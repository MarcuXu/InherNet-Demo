from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SVD_BACKEND_AUTO = "auto"
SVD_BACKEND_DEVICE = "device"
SVD_BACKEND_CPU = "cpu"
CPU_SVD_DTYPE = torch.float64


class StableSVDDecompositionError(RuntimeError):
    pass


def _candidate_svd_backends(reference_device: torch.device, requested_backend: str) -> list[str]:
    if requested_backend == SVD_BACKEND_AUTO:
        backends = [SVD_BACKEND_DEVICE]
        if reference_device.type != "cpu":
            backends.append(SVD_BACKEND_CPU)
        return backends
    if requested_backend in {SVD_BACKEND_DEVICE, SVD_BACKEND_CPU}:
        return [requested_backend]
    raise ValueError(f"Unsupported SVD backend: {requested_backend}")


def _move_tensor_for_svd_backend(tensor: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == SVD_BACKEND_CPU:
        target_dtype = CPU_SVD_DTYPE if tensor.dtype.is_floating_point else tensor.dtype
        return tensor.detach().to(device="cpu", dtype=target_dtype)
    return tensor.detach()


def _restore_tensor_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return tensor.to(device=reference.device, dtype=reference.dtype)


def _ensure_finite_tensors(context: str, **named_tensors: torch.Tensor) -> None:
    non_finite = [name for name, tensor in named_tensors.items() if not torch.isfinite(tensor).all()]
    if non_finite:
        raise StableSVDDecompositionError(
            f"{context} produced non-finite values for {', '.join(non_finite)}."
        )


def _checked_svd(weight: torch.Tensor, backend: str, context: str, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    working_weight = _move_tensor_for_svd_backend(weight, backend)
    try:
        u, s, v_h = torch.linalg.svd(working_weight, full_matrices=False)
    except RuntimeError as exc:
        raise StableSVDDecompositionError(f"{context} SVD failed on backend={backend}: {exc}") from exc
    _ensure_finite_tensors(f"{context} SVD on backend={backend}", u=u, s=s, v_h=v_h)
    return (
        _restore_tensor_like(u, reference),
        _restore_tensor_like(s, reference),
        _restore_tensor_like(v_h, reference),
    )


def _checked_inverse(matrix: torch.Tensor, context: str) -> torch.Tensor:
    try:
        inverse = torch.linalg.inv(matrix)
    except RuntimeError as exc:
        try:
            inverse = torch.linalg.pinv(matrix)
        except RuntimeError as pinv_exc:
            raise StableSVDDecompositionError(
                f"{context} inversion failed: {exc}; pseudo-inverse also failed: {pinv_exc}"
            ) from pinv_exc
    _ensure_finite_tensors(f"{context} inversion", inverse=inverse)
    return inverse


def _build_balanced_truncated_svd_factors(
    u: torch.Tensor,
    s: torch.Tensor,
    v_h: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r = max(1, min(rank, s.numel()))
    s_sqrt = torch.sqrt(torch.clamp(s[:r], min=1e-12))
    left_factor = (u[:, :r] @ torch.diag(s_sqrt)).contiguous()
    right_factor = (torch.diag(s_sqrt) @ v_h[:r, :]).contiguous()
    return left_factor, right_factor


class GatedSumLinear(nn.Module):
    def __init__(self, linear_list: nn.ModuleList, input_dim: int, head_num: int) -> None:
        super().__init__()
        self.linear_list = linear_list
        self.head_num = head_num
        self.gate = nn.Linear(input_dim, head_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        gating_scores = self.gate(x_flat)
        gating_weights = F.softmax(gating_scores, dim=-1)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.linear_list], dim=-1)
        output = torch.sum(gating_weights.unsqueeze(1) * expert_outputs, dim=-1)
        return output.reshape(*original_shape, output.shape[-1])


class GatedSumConv2d(nn.Module):
    def __init__(self, conv_list: nn.ModuleList, input_dim: int, head_num: int) -> None:
        super().__init__()
        self.conv_list = conv_list
        self.head_num = head_num
        self.gate = nn.Linear(input_dim, head_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        pooled = torch.mean(x, dim=(2, 3))
        gating_scores = self.gate(pooled)
        gating_weights = F.softmax(gating_scores, dim=-1)
        expert_outputs = torch.stack([conv(x) for conv in self.conv_list], dim=-1)
        gating_weights = gating_weights.view(batch_size, 1, 1, 1, self.head_num)
        return torch.sum(gating_weights * expert_outputs, dim=-1)


class DecoupledGatedSVDLinear(nn.Module):
    def __init__(
        self,
        linear1: nn.Linear,
        linear_list: nn.ModuleList,
        gate_input_dim: int,
        head_num: int,
        use_uncompressed_gate: bool = False,
    ) -> None:
        super().__init__()
        self.linear1 = linear1
        self.linear_list = linear_list
        self.head_num = head_num
        self.use_uncompressed_gate = use_uncompressed_gate
        self.gate = nn.Linear(gate_input_dim, head_num)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self._last_gating_probs: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        compressed = self.linear1(x_flat)
        expert_outputs = torch.stack([layer(compressed) for layer in self.linear_list], dim=-1)
        gate_feat = x_flat if self.use_uncompressed_gate else compressed
        gating_scores = self.gate(gate_feat)
        gating_probs = F.softmax(gating_scores, dim=-1)
        self._last_gating_probs = gating_probs
        output = torch.sum(gating_probs.unsqueeze(1) * expert_outputs, dim=-1)
        return output.reshape(*original_shape, output.shape[-1])

    def load_balance_loss(self) -> torch.Tensor | None:
        if self._last_gating_probs is None:
            return None
        mean_probs = self._last_gating_probs.mean(dim=0)
        return (mean_probs * mean_probs).sum() * self.head_num


class DecoupledGatedSVDConv2d(nn.Module):
    def __init__(
        self,
        conv1: nn.Conv2d,
        conv_list: nn.ModuleList,
        gate_input_dim: int,
        head_num: int,
        use_uncompressed_gate: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = conv1
        self.conv_list = conv_list
        self.head_num = head_num
        self.use_uncompressed_gate = use_uncompressed_gate
        self.gate = nn.Linear(gate_input_dim, head_num)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self._last_gating_probs: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        compressed = self.conv1(x)
        expert_outputs = torch.stack([conv(compressed) for conv in self.conv_list], dim=-1)
        if self.use_uncompressed_gate:
            gate_feat = torch.mean(x, dim=(2, 3))
        else:
            gate_feat = torch.mean(compressed, dim=(2, 3))
        gating_scores = self.gate(gate_feat)
        gating_probs = F.softmax(gating_scores, dim=-1)
        self._last_gating_probs = gating_probs
        gating_weights = gating_probs.view(batch_size, 1, 1, 1, self.head_num)
        return torch.sum(gating_weights * expert_outputs, dim=-1)

    def load_balance_loss(self) -> torch.Tensor | None:
        if self._last_gating_probs is None:
            return None
        mean_probs = self._last_gating_probs.mean(dim=0)
        return (mean_probs * mean_probs).sum() * self.head_num


class BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def load_dense_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.backbone.load_state_dict(state_dict)

    def _collect_target_layers(self) -> OrderedDict[str, nn.Module]:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers[name] = module
        return layers

    def _get_parent_module(self, module_name: str) -> tuple[nn.Module, str]:
        if "." not in module_name:
            return self.backbone, module_name
        parent_name, child_name = module_name.rsplit(".", 1)
        return self.backbone.get_submodule(parent_name), child_name

    def _match_module_device_dtype(self, replacement: nn.Module, reference: nn.Module) -> nn.Module:
        return replacement.to(device=reference.weight.device, dtype=reference.weight.dtype)


class GenericInherNet(BackboneWrapper):
    def _replace_linear_with_svd(
        self,
        module: nn.Linear,
        rank: int,
        head_num: int,
        *,
        svd_backend: str,
    ) -> nn.Module:
        weight = module.weight.data
        u, s, v_h = _checked_svd(
            weight,
            svd_backend,
            f"{module.__class__.__name__}({module.out_features},{module.in_features})",
            module.weight,
        )
        if rank >= s.numel():
            return module
        r = min(rank, s.numel())
        expert_weight, compressed_weight = _build_balanced_truncated_svd_factors(u, s, v_h, r)
        linear1 = nn.Linear(module.in_features, r, bias=False)
        linear1.weight.data = compressed_weight
        expert_layers = nn.ModuleList()
        for _ in range(head_num):
            linear2 = nn.Linear(r, module.out_features, bias=module.bias is not None)
            linear2.weight.data = expert_weight.clone()
            if module.bias is not None:
                linear2.bias.data = module.bias.data.clone()
            expert_layers.append(linear2)
        return nn.Sequential(linear1, GatedSumLinear(expert_layers, r, head_num))

    def _replace_conv_with_svd(
        self,
        module: nn.Conv2d,
        rank: int,
        head_num: int,
        *,
        svd_backend: str,
    ) -> nn.Module:
        if module.groups != 1:
            return module
        weight = module.weight.data
        c_out, c_in, k_h, k_w = weight.shape
        weight_flat = weight.view(c_out, -1)
        u, s, v_h = _checked_svd(
            weight_flat,
            svd_backend,
            f"{module.__class__.__name__}({c_out},{c_in * k_h * k_w})",
            module.weight,
        )
        if rank >= s.numel():
            return module
        r = min(rank, s.numel())
        expert_weight, compressed_weight = _build_balanced_truncated_svd_factors(u, s, v_h, r)
        conv1 = nn.Conv2d(
            c_in,
            r,
            kernel_size=(k_h, k_w),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            bias=False,
        )
        conv1.weight.data = compressed_weight.view(r, c_in, k_h, k_w)
        expert_layers = nn.ModuleList()
        for _ in range(head_num):
            conv2 = nn.Conv2d(r, c_out, kernel_size=1, stride=1, padding=0, bias=module.bias is not None)
            conv2.weight.data = expert_weight.clone().view(c_out, r, 1, 1)
            if module.bias is not None:
                conv2.bias.data = module.bias.data.clone()
            expert_layers.append(conv2)
        return nn.Sequential(conv1, GatedSumConv2d(expert_layers, r, head_num))

    def _replace_module_with_svd(
        self,
        module: nn.Module,
        rank: int,
        head_num: int,
        *,
        svd_backend: str,
    ) -> nn.Module:
        if isinstance(module, nn.Conv2d):
            replacement = self._replace_conv_with_svd(module, rank, head_num, svd_backend=svd_backend)
            return self._match_module_device_dtype(replacement, module)
        if isinstance(module, nn.Linear):
            replacement = self._replace_linear_with_svd(module, rank, head_num, svd_backend=svd_backend)
            return self._match_module_device_dtype(replacement, module)
        return module

    def _build_svd_replacements(
        self,
        rank: int,
        head_num: int,
        *,
        svd_backend: str,
    ) -> list[tuple[nn.Module, str, nn.Module]]:
        replacements: list[tuple[nn.Module, str, nn.Module]] = []
        for name, module in self._collect_target_layers().items():
            parent, child_name = self._get_parent_module(name)
            replacement = self._replace_module_with_svd(module, rank, head_num, svd_backend=svd_backend)
            replacements.append((parent, child_name, replacement))
        return replacements

    def apply_svd(self, rank: int, head_num: int, svd_backend: str = SVD_BACKEND_AUTO) -> str:
        last_error: StableSVDDecompositionError | None = None
        reference_device = next(self.parameters()).device
        for backend in _candidate_svd_backends(reference_device, svd_backend):
            try:
                replacements = self._build_svd_replacements(rank, head_num, svd_backend=backend)
            except StableSVDDecompositionError as exc:
                last_error = exc
                continue
            for parent, child_name, replacement in replacements:
                parent._modules[child_name] = replacement
            return backend
        if last_error is not None:
            raise last_error
        raise RuntimeError("No SVD backend candidate was available for InherNet.")


class GenericHeteroNet(BackboneWrapper):
    def _collect_hetero_target_layers(self, include_linear: bool = False) -> OrderedDict[str, nn.Module]:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) or (include_linear and isinstance(module, nn.Linear)):
                layers[name] = module
        return layers

    def _build_zero_mean_expert_noise(
        self,
        base_weight: torch.Tensor,
        head_num: int,
        noise_scale: float,
    ) -> torch.Tensor | None:
        if head_num <= 1 or noise_scale <= 0:
            return None
        base_std = base_weight.detach().std().clamp_min(1e-12)
        noise = torch.randn(
            (head_num, *base_weight.shape),
            device=base_weight.device,
            dtype=base_weight.dtype,
        )
        noise = noise * (noise_scale * base_std)
        return noise - noise.mean(dim=0, keepdim=True)

    def _extract_input_features(self, module: nn.Module, layer_input: torch.Tensor) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            return torch.mean(layer_input, dim=(2, 3))
        return layer_input.reshape(-1, layer_input.shape[-1])

    def _move_inputs_to_device(self, inputs, device: torch.device):
        if isinstance(inputs, Mapping):
            return {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
        return inputs.to(device)

    def _forward_with_inputs(self, inputs):
        if isinstance(inputs, Mapping):
            return self(**inputs)
        return self(inputs)

    def _stable_cholesky(self, matrix: torch.Tensor, base_eps: float = 1e-5) -> torch.Tensor:
        original_dtype = matrix.dtype
        working = torch.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        working = 0.5 * (working + working.transpose(0, 1))
        working = working.to(dtype=torch.float64)
        eye = torch.eye(working.shape[0], device=working.device, dtype=working.dtype)
        scale = torch.diagonal(working).abs().mean().clamp_min(1.0)
        jitter = base_eps * scale

        def finalize(chol: torch.Tensor) -> torch.Tensor:
            chol = torch.nan_to_num(chol, nan=0.0, posinf=1e6, neginf=-1e6)
            chol = torch.clamp(chol, min=-1e6, max=1e6)
            chol = chol.to(dtype=original_dtype)
            diag_idx = torch.arange(chol.shape[0], device=chol.device)
            chol[diag_idx, diag_idx] = chol[diag_idx, diag_idx].clamp_min(1e-6)
            return chol

        for _ in range(5):
            try:
                chol = torch.linalg.cholesky(working + jitter * eye)
                return finalize(chol)
            except RuntimeError:
                jitter *= 10.0
        eigvals, eigvecs = torch.linalg.eigh(working)
        eigvals = torch.clamp(eigvals, min=base_eps * scale)
        repaired = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.transpose(0, 1)
        chol = torch.linalg.cholesky(repaired + jitter * eye)
        return finalize(chol)

    def _estimate_input_covariances(
        self,
        calib_loader: DataLoader,
        max_batches: int = 16,
        eps: float = 1e-5,
        include_linear: bool = False,
    ) -> dict[str, torch.Tensor]:
        target_layers = self._collect_hetero_target_layers(include_linear=include_linear)
        stats = {
            name: {"sum": None, "sum_outer": None, "count": 0}
            for name in target_layers.keys()
        }
        handles = []

        def make_hook(layer_name: str, layer_module: nn.Module):
            def hook(_, layer_input, __):
                features = self._extract_input_features(layer_module, layer_input[0].detach())
                features = features.view(features.shape[0], -1)
                features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
                features = torch.clamp(features, min=-1e6, max=1e6)
                sum_vec = features.sum(dim=0)
                sum_outer = features.t().matmul(features)
                if stats[layer_name]["sum"] is None:
                    stats[layer_name]["sum"] = sum_vec
                    stats[layer_name]["sum_outer"] = sum_outer
                else:
                    stats[layer_name]["sum"] += sum_vec
                    stats[layer_name]["sum_outer"] += sum_outer
                stats[layer_name]["count"] += features.shape[0]

            return hook

        for name, module in target_layers.items():
            handles.append(module.register_forward_hook(make_hook(name, module)))

        was_training = self.training
        self.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(calib_loader):
                if batch_idx >= max_batches:
                    break
                moved_inputs = self._move_inputs_to_device(inputs, next(self.parameters()).device)
                _ = self._forward_with_inputs(moved_inputs)
        for handle in handles:
            handle.remove()
        if was_training:
            self.train()

        covariances: dict[str, torch.Tensor] = {}
        for name, module in target_layers.items():
            in_dim = module.in_channels if isinstance(module, nn.Conv2d) else module.in_features
            layer_stats = stats[name]
            if layer_stats["count"] == 0:
                covariances[name] = torch.eye(in_dim, device=module.weight.device, dtype=module.weight.dtype)
                continue
            mean = layer_stats["sum"] / layer_stats["count"]
            exx = layer_stats["sum_outer"] / layer_stats["count"]
            cov = exx - torch.outer(mean, mean)
            cov = 0.5 * (cov + cov.transpose(0, 1))
            cov = torch.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
            cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            covariances[name] = cov.to(device=module.weight.device, dtype=module.weight.dtype)
        return covariances

    def _whiten_weight(self, module: nn.Module, weight: torch.Tensor, chol_c: torch.Tensor) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            c_out, c_in, k_h, k_w = weight.shape
            weight_perm = weight.permute(0, 2, 3, 1).reshape(-1, c_in)
            whitened = weight_perm.matmul(chol_c)
            return whitened.view(c_out, k_h, k_w, c_in).permute(0, 3, 1, 2).contiguous()
        return weight.matmul(chol_c)

    def _compute_spectral_entropies(
        self,
        covariances: Mapping[str, torch.Tensor],
        *,
        svd_backend: str,
        include_linear: bool = False,
    ) -> tuple[dict[str, float], dict[str, int], dict[str, dict[str, torch.Tensor]]]:
        target_layers = self._collect_hetero_target_layers(include_linear=include_linear)
        entropies: dict[str, float] = {}
        max_ranks: dict[str, int] = {}
        svd_cache: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in target_layers.items():
            working_weight = _move_tensor_for_svd_backend(module.weight.data, svd_backend)
            working_covariance = _move_tensor_for_svd_backend(covariances[name], svd_backend)
            chol_c = self._stable_cholesky(working_covariance)
            _ensure_finite_tensors(f"{name} cholesky on backend={svd_backend}", chol_c=chol_c)
            whiten_inv = _checked_inverse(chol_c, f"{name} whitening matrix")
            whitened_weight = self._whiten_weight(module, working_weight, chol_c)
            _ensure_finite_tensors(f"{name} whitened weight on backend={svd_backend}", whitened_weight=whitened_weight)
            weight_flat = whitened_weight.view(whitened_weight.shape[0], -1)
            u, s, v_h = _checked_svd(weight_flat, svd_backend, f"{name} spectral entropy", module.weight)
            s_sq = _move_tensor_for_svd_backend(s, svd_backend) ** 2
            sigma_sum = s_sq.sum().clamp_min(1e-12)
            probs = (s_sq / sigma_sum).clamp_min(1e-12)
            entropies[name] = (-(probs * torch.log(probs)).sum()).item()
            max_ranks[name] = s.numel()
            svd_cache[name] = {
                "u": u,
                "s": s,
                "v_h": v_h,
                "whiten_inv": _restore_tensor_like(whiten_inv, module.weight),
            }
        return entropies, max_ranks, svd_cache

    def _allocate_ranks_by_entropy(
        self,
        entropies: Mapping[str, float],
        max_ranks: Mapping[str, int],
        budget_ratio: float,
        min_rank: int,
        temperature: float,
    ) -> dict[str, int]:
        layer_names = list(entropies.keys())
        total_max = sum(max_ranks[name] for name in layer_names)
        budget = int(max(len(layer_names) * min_rank, round(total_max * budget_ratio)))
        budget = min(budget, total_max)

        floor_budget = len(layer_names) * min_rank
        remaining_budget = max(0, budget - floor_budget)
        smoothed = {
            name: entropies[name] ** (1.0 / max(temperature, 1e-6))
            for name in layer_names
        }
        smoothed_sum = sum(smoothed.values())
        if smoothed_sum <= 0:
            raw = {name: remaining_budget / max(len(layer_names), 1) for name in layer_names}
        else:
            raw = {name: smoothed[name] / smoothed_sum * remaining_budget for name in layer_names}

        ranks = {
            name: min(max_ranks[name], min_rank + int(round(raw[name])))
            for name in layer_names
        }
        current_total = sum(ranks.values())
        if current_total < budget:
            order = sorted(layer_names, key=lambda item: raw[item] - int(raw[item]), reverse=True)
            idx = 0
            while current_total < budget and order:
                name = order[idx % len(order)]
                if ranks[name] < max_ranks[name]:
                    ranks[name] += 1
                    current_total += 1
                idx += 1
                if idx > len(order) * (max(max_ranks.values()) + 1):
                    break
        elif current_total > budget:
            order = sorted(layer_names, key=lambda item: raw[item] - int(raw[item]))
            idx = 0
            while current_total > budget and order:
                name = order[idx % len(order)]
                if ranks[name] > min_rank:
                    ranks[name] -= 1
                    current_total -= 1
                idx += 1
                if idx > len(order) * (max(max_ranks.values()) + 1):
                    break
        return ranks

    def _replace_conv_with_hetero_svd(
        self,
        module: nn.Conv2d,
        rank: int,
        head_num: int,
        compress_threshold: int,
        svd_pack: Mapping[str, torch.Tensor],
        expert_noise_scale: float,
    ) -> nn.Module:
        if module.groups != 1:
            return module
        c_out, c_in, k_h, k_w = module.weight.shape
        u = svd_pack["u"]
        s = svd_pack["s"]
        v_h = svd_pack["v_h"]
        whiten_inv = svd_pack["whiten_inv"]
        rank = max(1, min(rank, s.numel()))
        u_trunc = u[:, :rank]
        s_trunc = s[:rank]
        v_h_trunc = v_h[:rank, :]
        s_sqrt = torch.sqrt(torch.clamp(s_trunc, min=1e-12))
        v_scaled = torch.diag(s_sqrt) @ v_h_trunc
        v_4d = v_scaled.view(rank, c_in, k_h, k_w)
        v_perm = v_4d.permute(0, 2, 3, 1).reshape(-1, c_in)
        v_unwhiten = v_perm.matmul(whiten_inv)
        conv1_weight = v_unwhiten.view(rank, k_h, k_w, c_in).permute(0, 3, 1, 2).contiguous()
        conv1 = nn.Conv2d(
            c_in,
            rank,
            kernel_size=(k_h, k_w),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            bias=False,
        )
        conv1.weight.data = conv1_weight
        expert_weight = (u_trunc @ torch.diag(s_sqrt)).contiguous().view(c_out, rank, 1, 1)
        expert_noise = self._build_zero_mean_expert_noise(
            expert_weight,
            head_num,
            expert_noise_scale,
        )
        expert_layers = nn.ModuleList()
        for head_idx in range(head_num):
            conv2 = nn.Conv2d(rank, c_out, kernel_size=1, stride=1, padding=0, bias=module.bias is not None)
            conv2.weight.data = expert_weight.clone()
            if expert_noise is not None:
                conv2.weight.data.add_(expert_noise[head_idx])
            if module.bias is not None:
                conv2.bias.data = module.bias.data.clone()
            expert_layers.append(conv2)
        use_uncompressed_gate = rank < compress_threshold
        gate_input_dim = c_in if use_uncompressed_gate else rank
        return DecoupledGatedSVDConv2d(
            conv1,
            expert_layers,
            gate_input_dim,
            head_num,
            use_uncompressed_gate,
        )

    def _replace_linear_with_hetero_svd(
        self,
        module: nn.Linear,
        rank: int,
        head_num: int,
        compress_threshold: int,
        svd_pack: Mapping[str, torch.Tensor],
        expert_noise_scale: float,
    ) -> nn.Module:
        u = svd_pack["u"]
        s = svd_pack["s"]
        v_h = svd_pack["v_h"]
        whiten_inv = svd_pack["whiten_inv"]
        rank = max(1, min(rank, s.numel()))
        u_trunc = u[:, :rank]
        s_trunc = s[:rank]
        v_h_trunc = v_h[:rank, :]
        s_sqrt = torch.sqrt(torch.clamp(s_trunc, min=1e-12))
        linear1_weight = (torch.diag(s_sqrt) @ v_h_trunc @ whiten_inv).contiguous()
        linear1 = nn.Linear(module.in_features, rank, bias=False)
        linear1.weight.data = linear1_weight
        expert_weight = (u_trunc @ torch.diag(s_sqrt)).contiguous()
        expert_noise = self._build_zero_mean_expert_noise(
            expert_weight,
            head_num,
            expert_noise_scale,
        )
        expert_layers = nn.ModuleList()
        for head_idx in range(head_num):
            linear2 = nn.Linear(rank, module.out_features, bias=module.bias is not None)
            linear2.weight.data = expert_weight.clone()
            if expert_noise is not None:
                linear2.weight.data.add_(expert_noise[head_idx])
            if module.bias is not None:
                linear2.bias.data = module.bias.data.clone()
            expert_layers.append(linear2)
        use_uncompressed_gate = rank < compress_threshold
        gate_input_dim = module.in_features if use_uncompressed_gate else rank
        return DecoupledGatedSVDLinear(
            linear1,
            expert_layers,
            gate_input_dim,
            head_num,
            use_uncompressed_gate,
        )

    def _replace_module_with_hetero_svd(
        self,
        module: nn.Module,
        rank: int,
        head_num: int,
        compress_threshold: int,
        svd_pack: Mapping[str, torch.Tensor],
        expert_noise_scale: float,
    ) -> nn.Module:
        if isinstance(module, nn.Conv2d):
            replacement = self._replace_conv_with_hetero_svd(
                module,
                rank,
                head_num,
                compress_threshold,
                svd_pack,
                expert_noise_scale,
            )
            return self._match_module_device_dtype(replacement, module)
        if isinstance(module, nn.Linear):
            replacement = self._replace_linear_with_hetero_svd(
                module,
                rank,
                head_num,
                compress_threshold,
                svd_pack,
                expert_noise_scale,
            )
            return self._match_module_device_dtype(replacement, module)
        return module

    def apply_hetero_svd(
        self,
        calib_loader: DataLoader,
        head_num: int = 3,
        budget_ratio: float = 0.35,
        min_rank: int = 8,
        compress_threshold: int = 12,
        temperature: float = 1.4,
        max_calib_batches: int = 16,
        svd_backend: str = SVD_BACKEND_AUTO,
        expert_noise_scale: float = 0.01,
        compress_linear: bool = False,
    ) -> tuple[dict[str, int], str]:
        last_error: StableSVDDecompositionError | None = None
        reference_device = next(self.parameters()).device
        covariances = self._estimate_input_covariances(
            calib_loader,
            max_batches=max_calib_batches,
            include_linear=compress_linear,
        )
        for backend in _candidate_svd_backends(reference_device, svd_backend):
            try:
                entropies, max_ranks, svd_cache = self._compute_spectral_entropies(
                    covariances,
                    svd_backend=backend,
                    include_linear=compress_linear,
                )
                rank_map = self._allocate_ranks_by_entropy(
                    entropies,
                    max_ranks,
                    budget_ratio=budget_ratio,
                    min_rank=min_rank,
                    temperature=temperature,
                )
                replacements: list[tuple[nn.Module, str, nn.Module]] = []
                for name, module in self._collect_hetero_target_layers(include_linear=compress_linear).items():
                    parent, child_name = self._get_parent_module(name)
                    replacement = self._replace_module_with_hetero_svd(
                        module,
                        rank=rank_map[name],
                        head_num=head_num,
                        compress_threshold=compress_threshold,
                        svd_pack=svd_cache[name],
                        expert_noise_scale=expert_noise_scale,
                    )
                    replacements.append((parent, child_name, replacement))
            except StableSVDDecompositionError as exc:
                last_error = exc
                continue
            for parent, child_name, replacement in replacements:
                parent._modules[child_name] = replacement
            return rank_map, backend
        if last_error is not None:
            raise last_error
        raise RuntimeError("No SVD backend candidate was available for HeteroNet.")


def compute_gating_load_balance_loss(model: nn.Module) -> torch.Tensor | None:
    losses = []
    for module in model.modules():
        if hasattr(module, "load_balance_loss"):
            aux_loss = module.load_balance_loss()
            if aux_loss is not None:
                losses.append(aux_loss)
    if not losses:
        return None
    return torch.stack(losses).mean()
