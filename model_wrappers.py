from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SVD_BACKEND_AUTO = "auto"
SVD_BACKEND_DEVICE = "device"
SVD_BACKEND_CPU = "cpu"
CPU_SVD_DTYPE = torch.float64
FINAL_HETERO_ALLOCATION = "weighted_uniform"
MAINTAINED_HETERO_POLICIES = (
    FINAL_HETERO_ALLOCATION,
    "unweighted_uniform",
)
RESEARCH_HETERO_RANK_POLICIES = (
    "research_nested_relative",
    "research_total_output",
    "research_relative",
)
HETERO_ALLOCATION_SCALES = (
    *MAINTAINED_HETERO_POLICIES,
    *RESEARCH_HETERO_RANK_POLICIES,
)


@dataclass(frozen=True)
class HeteroConfig:
    """Validated configuration for data-aware conditional inheritance."""

    head_num: int = 3
    reference_rank: int = 8
    max_calib_batches: int = 16
    expert_noise_scale: float = 0.01
    compress_linear: bool = False
    max_features_per_batch: int = 4096
    second_moment_shrinkage: float = 0.01
    allocation_scale: str = FINAL_HETERO_ALLOCATION
    research_protected_rank: int | None = None

    def __post_init__(self) -> None:
        if self.head_num <= 0:
            raise ValueError("head_num must be positive.")
        if self.reference_rank <= 0:
            raise ValueError("reference_rank must be positive.")
        if self.max_calib_batches <= 0:
            raise ValueError("max_calib_batches must be positive.")
        if self.expert_noise_scale < 0:
            raise ValueError("expert_noise_scale must be non-negative.")
        if self.max_features_per_batch <= 0:
            raise ValueError("max_features_per_batch must be positive.")
        if not 0.0 <= self.second_moment_shrinkage <= 1.0:
            raise ValueError("second_moment_shrinkage must be in [0, 1].")
        if self.allocation_scale not in HETERO_ALLOCATION_SCALES:
            raise ValueError(f"Unknown allocation_scale: {self.allocation_scale}")
        if self.research_protected_rank is not None and self.research_protected_rank <= 0:
            raise ValueError("research_protected_rank must be positive when provided.")
        if self.allocation_scale == "research_nested_relative" and self.research_protected_rank is None:
            raise ValueError("research_nested_relative requires research_protected_rank.")


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


def _checked_triangular_inverse(matrix: torch.Tensor, context: str) -> torch.Tensor:
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    try:
        inverse = torch.linalg.solve_triangular(matrix, identity, upper=False)
    except RuntimeError as exc:
        raise StableSVDDecompositionError(f"{context} triangular solve failed: {exc}") from exc
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
    left_factor = (u[:, :r] * s_sqrt.unsqueeze(0)).contiguous()
    right_factor = (s_sqrt.unsqueeze(1) * v_h[:r, :]).contiguous()
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


class LoadBalancedRouter(nn.Module):
    def __init__(self, head_num: int) -> None:
        super().__init__()
        self.head_num = head_num
        self._last_gating_probs: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self._attention_mask = attention_mask

    def load_balance_loss(self) -> torch.Tensor | None:
        if self._last_gating_probs is None:
            return None
        probabilities = self._last_gating_probs
        if self._attention_mask is not None and probabilities.shape[0] == self._attention_mask.numel():
            probabilities = probabilities[self._attention_mask.reshape(-1).bool()]
        if probabilities.shape[0] == 0:
            return None
        mean_probs = probabilities.mean(dim=0)
        return (mean_probs * mean_probs).sum() * self.head_num - 1.0


class GatedSVDLinear(LoadBalancedRouter):
    def __init__(
        self,
        linear1: nn.Linear,
        expert_weight: torch.Tensor,
        expert_bias: torch.Tensor | None,
        head_num: int,
    ) -> None:
        super().__init__(head_num)
        self.linear1 = linear1
        out_features = expert_weight.shape[0] // head_num
        self.out_features = out_features
        self.experts = nn.Linear(
            linear1.out_features,
            head_num * out_features,
            bias=expert_bias is not None,
        )
        with torch.no_grad():
            self.experts.weight.copy_(expert_weight)
            if self.experts.bias is not None:
                self.experts.bias.copy_(expert_bias)
        self.gate = nn.Linear(linear1.out_features, head_num)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        compressed = self.linear1(x_flat)
        combined = self.experts(compressed)
        expert_outputs = combined.view(-1, self.head_num, self.out_features)
        gating_scores = self.gate(compressed)
        gating_probs = F.softmax(gating_scores, dim=-1)
        self._last_gating_probs = gating_probs
        output = torch.sum(gating_probs.unsqueeze(-1) * expert_outputs, dim=1)
        return output.reshape(*original_shape, output.shape[-1])


class GatedSVDConv2d(LoadBalancedRouter):
    def __init__(
        self,
        conv1: nn.Conv2d,
        expert_weight: torch.Tensor,
        expert_bias: torch.Tensor | None,
        head_num: int,
    ) -> None:
        super().__init__(head_num)
        self.conv1 = conv1
        out_channels = expert_weight.shape[0] // head_num
        self.out_channels = out_channels
        self.experts = nn.Conv2d(
            conv1.out_channels,
            head_num * out_channels,
            kernel_size=1,
            bias=expert_bias is not None,
        )
        with torch.no_grad():
            self.experts.weight.copy_(expert_weight)
            if self.experts.bias is not None:
                self.experts.bias.copy_(expert_bias)
        self.gate = nn.Linear(conv1.out_channels, head_num)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        compressed = self.conv1(x)
        combined = self.experts(compressed)
        expert_outputs = combined.view(
            batch_size,
            self.head_num,
            self.out_channels,
            combined.shape[-2],
            combined.shape[-1],
        )
        gate_feat = torch.mean(compressed, dim=(2, 3))
        gating_scores = self.gate(gate_feat)
        gating_probs = F.softmax(gating_scores, dim=-1)
        self._last_gating_probs = gating_probs
        gating_weights = gating_probs.view(batch_size, self.head_num, 1, 1, 1)
        return torch.sum(gating_weights * expert_outputs, dim=1)

class BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def load_dense_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.backbone.load_state_dict(state_dict)

    def _collect_target_layers(self, *, include_linear: bool = True) -> OrderedDict[str, nn.Module]:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) or (include_linear and isinstance(module, nn.Linear)):
                layers[name] = module
        return layers

    def _get_parent_module(self, module_name: str) -> tuple[nn.Module, str]:
        if "." not in module_name:
            return self.backbone, module_name
        parent_name, child_name = module_name.rsplit(".", 1)
        return self.backbone.get_submodule(parent_name), child_name

    def _match_module_device_dtype(self, replacement: nn.Module, reference: nn.Module) -> nn.Module:
        replacement = replacement.to(device=reference.weight.device, dtype=reference.weight.dtype)
        replacement.train(reference.training)
        replacement.requires_grad_(reference.weight.requires_grad)
        return replacement


class GenericInherNet(BackboneWrapper):
    def _replace_linear_with_svd(
        self,
        module: nn.Linear,
        rank: int,
        head_num: int,
        *,
        svd_backend: str,
    ) -> nn.Module:
        weight = module.weight.detach()
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
        with torch.no_grad():
            linear1.weight.copy_(compressed_weight)
        expert_layers = nn.ModuleList()
        for _ in range(head_num):
            linear2 = nn.Linear(r, module.out_features, bias=module.bias is not None)
            with torch.no_grad():
                linear2.weight.copy_(expert_weight)
                if module.bias is not None:
                    linear2.bias.copy_(module.bias)
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
        weight = module.weight.detach()
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
            padding_mode=module.padding_mode,
            bias=False,
        )
        with torch.no_grad():
            conv1.weight.copy_(compressed_weight.view(r, c_in, k_h, k_w))
        expert_layers = nn.ModuleList()
        for _ in range(head_num):
            conv2 = nn.Conv2d(r, c_out, kernel_size=1, stride=1, padding=0, bias=module.bias is not None)
            with torch.no_grad():
                conv2.weight.copy_(expert_weight.view(c_out, r, 1, 1))
                if module.bias is not None:
                    conv2.bias.copy_(module.bias)
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
        include_linear: bool,
    ) -> list[tuple[nn.Module, str, nn.Module]]:
        replacements: list[tuple[nn.Module, str, nn.Module]] = []
        for name, module in self._collect_target_layers(include_linear=include_linear).items():
            parent, child_name = self._get_parent_module(name)
            replacement = self._replace_module_with_svd(module, rank, head_num, svd_backend=svd_backend)
            replacements.append((parent, child_name, replacement))
        return replacements

    def apply_svd(
        self,
        rank: int,
        head_num: int,
        svd_backend: str = SVD_BACKEND_AUTO,
        *,
        include_linear: bool = False,
    ) -> str:
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if head_num <= 0:
            raise ValueError("head_num must be positive.")
        last_error: StableSVDDecompositionError | None = None
        reference_device = next(self.parameters()).device
        for backend in _candidate_svd_backends(reference_device, svd_backend):
            try:
                replacements = self._build_svd_replacements(
                    rank,
                    head_num,
                    svd_backend=backend,
                    include_linear=include_linear,
                )
            except StableSVDDecompositionError as exc:
                last_error = exc
                continue
            for parent, child_name, replacement in replacements:
                setattr(parent, child_name, replacement)
            return backend
        if last_error is not None:
            raise last_error
        raise RuntimeError("No SVD backend candidate was available for InherNet.")


class GenericHeteroNet(BackboneWrapper):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__(backbone)
        self._cached_routers: tuple[LoadBalancedRouter, ...] = ()

    def forward(self, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        for router in self._cached_routers:
            router.set_attention_mask(attention_mask)
        return super().forward(*args, **kwargs)

    def _collect_hetero_target_layers(self, include_linear: bool = False) -> OrderedDict[str, nn.Module]:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for name, module in self.backbone.named_modules():
            if (
                isinstance(module, nn.Conv2d)
                and module.groups == 1
            ) or (include_linear and isinstance(module, nn.Linear)):
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
        base_std = base_weight.detach().std(unbiased=False).clamp_min(1e-12)
        noise = torch.randn(
            (head_num, *base_weight.shape),
            device=base_weight.device,
            dtype=base_weight.dtype,
        )
        noise = noise * (noise_scale * base_std)
        return noise - noise.mean(dim=0, keepdim=True)

    @staticmethod
    def _expert_lift_statistics(
        base_weight: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> dict[str, float]:
        """Measure diversity and numerical mean preservation across experts."""
        working_base = base_weight.detach().float()
        working_experts = expert_weights.detach().float()
        mean_expert = working_experts.mean(dim=0)
        base_norm = working_base.norm().clamp_min(1e-30)
        mean_shift = mean_expert.sub(working_base).norm() / base_norm
        diversity = (
            working_experts.sub(mean_expert.unsqueeze(0)).square().mean().sqrt()
            / working_base.square().mean().sqrt().clamp_min(1e-30)
        )
        return {
            "relative_expert_mean_shift": float(mean_shift.item()),
            "relative_expert_diversity": float(diversity.item()),
        }

    def _extract_input_features(
        self,
        module: nn.Module,
        layer_input: torch.Tensor,
        *,
        max_features: int,
    ) -> tuple[torch.Tensor, str, int]:
        """Return local features whose Gram matrix matches layer output error.

        Small convolutions use exact im2col patches. Wider convolutions use a
        channel-wise, per-location approximation; importantly, neither path
        averages an image down to one vector. Token linears discard padding
        when an attention mask from the current calibration batch is available.
        """
        if isinstance(module, nn.Conv2d):
            input_h, input_w = layer_input.shape[-2:]
            output_h = math.floor(
                (input_h + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1)
                / module.stride[0]
                + 1
            )
            output_w = math.floor(
                (input_w + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1)
                / module.stride[1]
                + 1
            )
            application_count = layer_input.shape[0] * output_h * output_w
            patch_dim = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            if patch_dim <= 256:
                features = self._sample_conv_patches(layer_input, module, max_features)
                mode = "exact_patch"
            else:
                # Block-diagonal channel approximation sampled from actual
                # convolution applications. This respects stride, dilation,
                # padding, and kernel offsets while avoiding a patch_dim^2
                # second moment.
                kernel_positions = module.kernel_size[0] * module.kernel_size[1]
                patch_quota = max(1, max_features // kernel_positions)
                patches = self._sample_conv_patches(layer_input, module, patch_quota)
                features = (
                    patches.view(patches.shape[0], module.in_channels, kernel_positions)
                    .transpose(1, 2)
                    .reshape(-1, module.in_channels)
                )
                mode = "channel_block"
        else:
            features = layer_input.reshape(-1, layer_input.shape[-1])
            mask = getattr(self, "_calibration_attention_mask", None)
            if (
                mask is not None
                and layer_input.ndim >= 3
                and tuple(layer_input.shape[:2]) == tuple(mask.shape)
            ):
                features = features[mask.reshape(-1).bool()]
            mode = "full" if module.in_features <= 512 else "diagonal"
            application_count = features.shape[0]

        if features.shape[0] > max_features:
            # Deterministic coverage of the complete spatial/token range.
            indices = torch.linspace(
                0,
                features.shape[0] - 1,
                steps=max_features,
                device=features.device,
            ).long()
            features = features.index_select(0, indices)
        return features, mode, application_count

    @staticmethod
    def _sample_conv_patches(
        layer_input: torch.Tensor,
        module: nn.Conv2d,
        max_patches: int,
    ) -> torch.Tensor:
        """Sample local convolution patches across the complete mini-batch."""
        image_count = min(layer_input.shape[0], max_patches)
        image_indices = torch.linspace(
            0,
            layer_input.shape[0] - 1,
            steps=image_count,
            device=layer_input.device,
        ).long()
        base_quota, remainder = divmod(max_patches, image_count)
        sampled: list[torch.Tensor] = []
        for position, image_index in enumerate(image_indices):
            image = layer_input[image_index : image_index + 1]
            unfold_padding = module.padding
            if module.padding_mode != "zeros" and any(module.padding):
                pad_h, pad_w = module.padding
                image = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode=module.padding_mode)
                unfold_padding = (0, 0)
            patches = F.unfold(
                image,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=unfold_padding,
                stride=module.stride,
            ).transpose(1, 2).squeeze(0)
            quota = min(patches.shape[0], base_quota + int(position < remainder))
            if quota < patches.shape[0]:
                indices = torch.linspace(
                    0,
                    patches.shape[0] - 1,
                    steps=quota,
                    device=patches.device,
                ).long()
                patches = patches.index_select(0, indices)
            sampled.append(patches)
        return torch.cat(sampled, dim=0)

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
        if not torch.isfinite(matrix).all():
            raise StableSVDDecompositionError("Cannot factor a non-finite second moment.")
        working_dtype = torch.float64 if matrix.device.type == "cpu" else torch.float32
        working = (0.5 * (matrix + matrix.transpose(0, 1))).to(dtype=working_dtype)
        eye = torch.eye(working.shape[0], device=working.device, dtype=working.dtype)
        scale = torch.diagonal(working).abs().mean().clamp_min(1e-12)
        jitter = base_eps * scale

        def finalize(chol: torch.Tensor) -> torch.Tensor:
            chol = chol.to(dtype=original_dtype)
            _ensure_finite_tensors("Cholesky factorization", chol=chol)
            return chol

        for _ in range(5):
            try:
                chol = torch.linalg.cholesky(working + jitter * eye)
                return finalize(chol)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise StableSVDDecompositionError(
                        "Cholesky factorization ran out of memory."
                    ) from exc
                jitter *= 10.0
        eigvals, eigvecs = torch.linalg.eigh(working)
        eigvals = torch.clamp(eigvals, min=base_eps * scale)
        repaired = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.transpose(0, 1)
        chol = torch.linalg.cholesky(repaired + jitter * eye)
        return finalize(chol)

    def _estimate_input_second_moments(
        self,
        calib_loader: DataLoader,
        max_batches: int = 16,
        eps: float = 1e-5,
        include_linear: bool = False,
        max_features_per_batch: int = 4096,
        shrinkage: float = 0.01,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object]]]:
        target_layers = self._collect_hetero_target_layers(include_linear=include_linear)
        stats = {
            name: {
                "sum_outer": None,
                "count": 0,
                "population": 0,
                "examples": 0,
                "mode": "unknown",
            }
            for name in target_layers.keys()
        }
        handles = []

        def make_hook(layer_name: str, layer_module: nn.Module):
            def hook(_, layer_input, __):
                features, mode, application_count = self._extract_input_features(
                    layer_module,
                    layer_input[0].detach(),
                    max_features=max_features_per_batch,
                )
                if features.numel() == 0:
                    return
                features = features.view(features.shape[0], -1)
                accumulation_features = (
                    features.double() if features.device.type == "cpu" else features.float()
                )
                if mode == "diagonal":
                    sum_outer = accumulation_features.square().sum(dim=0)
                else:
                    sum_outer = accumulation_features.t().matmul(accumulation_features)
                if stats[layer_name]["sum_outer"] is None:
                    stats[layer_name]["sum_outer"] = sum_outer
                else:
                    stats[layer_name]["sum_outer"] += sum_outer
                stats[layer_name]["count"] += features.shape[0]
                stats[layer_name]["population"] += application_count
                stats[layer_name]["examples"] += layer_input[0].shape[0]
                stats[layer_name]["mode"] = mode

            return hook

        for name, module in target_layers.items():
            handles.append(module.register_forward_hook(make_hook(name, module)))

        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                for batch_idx, (inputs, _) in enumerate(calib_loader):
                    if batch_idx >= max_batches:
                        break
                    moved_inputs = self._move_inputs_to_device(inputs, next(self.parameters()).device)
                    self._calibration_attention_mask = (
                        moved_inputs.get("attention_mask")
                        if isinstance(moved_inputs, Mapping)
                        else None
                    )
                    _ = self._forward_with_inputs(moved_inputs)
        finally:
            self._calibration_attention_mask = None
            for handle in handles:
                handle.remove()
            self.train(was_training)

        moments: dict[str, torch.Tensor] = {}
        metadata: dict[str, dict[str, object]] = {}
        for name, module in target_layers.items():
            layer_stats = stats[name]
            mode = str(layer_stats["mode"])
            if layer_stats["count"] == 0:
                raise ValueError(f"Calibration did not exercise target layer '{name}'.")
            raw = layer_stats["sum_outer"] / layer_stats["count"]
            if raw.ndim == 1:
                trace_scale = raw.mean().clamp_min(1e-12)
                moment = (1.0 - shrinkage) * raw + shrinkage * trace_scale
                moment = moment + eps * trace_scale
            else:
                moment = 0.5 * (raw + raw.transpose(0, 1))
                trace_scale = torch.diagonal(moment).mean().clamp_min(1e-12)
                eye = torch.eye(moment.shape[0], device=moment.device, dtype=moment.dtype)
                moment = (1.0 - shrinkage) * moment + shrinkage * trace_scale * eye
                moment = moment + eps * trace_scale * eye
            if not torch.isfinite(moment).all():
                raise ValueError(f"Non-finite second moment for layer '{name}'.")
            moment_dtype = (
                torch.float64 if module.weight.device.type == "cpu" else module.weight.dtype
            )
            moments[name] = moment.to(device=module.weight.device, dtype=moment_dtype)
            applications = layer_stats["population"] / max(layer_stats["examples"], 1)
            metadata[name] = {
                "mode": mode,
                "samples": int(layer_stats["count"]),
                "applications_per_example": float(applications),
            }
        return moments, metadata

    def _whiten_weight(self, module: nn.Module, weight: torch.Tensor, chol_c: torch.Tensor) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            c_out, c_in, k_h, k_w = weight.shape
            if chol_c.ndim == 1:
                return weight * chol_c.view(1, c_in, 1, 1)
            if chol_c.shape[0] == c_in * k_h * k_w:
                return weight.reshape(c_out, -1).matmul(chol_c)
            weight_perm = weight.permute(0, 2, 3, 1).reshape(-1, c_in)
            whitened = weight_perm.matmul(chol_c)
            return whitened.view(c_out, k_h, k_w, c_in).permute(0, 3, 1, 2).contiguous()
        return weight * chol_c if chol_c.ndim == 1 else weight.matmul(chol_c)

    def _compute_weighted_svd_cache(
        self,
        second_moments: Mapping[str, torch.Tensor],
        *,
        svd_backend: str,
        include_linear: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        target_layers = self._collect_hetero_target_layers(include_linear=include_linear)
        svd_cache: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in target_layers.items():
            working_weight = _move_tensor_for_svd_backend(module.weight.detach(), svd_backend)
            working_moment = _move_tensor_for_svd_backend(second_moments[name], svd_backend)
            working_weight = working_weight.to(dtype=working_moment.dtype)
            if working_moment.ndim == 1:
                chol_c = working_moment.clamp_min(1e-12).sqrt()
                whiten_inv = chol_c.reciprocal()
            else:
                chol_c = self._stable_cholesky(working_moment)
                whiten_inv = _checked_triangular_inverse(chol_c, f"{name} whitening matrix")
            _ensure_finite_tensors(
                f"{name} whitening on backend={svd_backend}",
                chol_c=chol_c,
                whiten_inv=whiten_inv,
            )
            whitened_weight = self._whiten_weight(module, working_weight, chol_c)
            _ensure_finite_tensors(f"{name} whitened weight on backend={svd_backend}", whitened_weight=whitened_weight)
            weight_flat = whitened_weight.view(whitened_weight.shape[0], -1)
            u, s, v_h = _checked_svd(weight_flat, svd_backend, f"{name} weighted SVD", module.weight)
            svd_cache[name] = {
                "u": u,
                "s": s,
                "v_h": v_h,
                "whiten_inv": _restore_tensor_like(whiten_inv, module.weight),
            }
        return svd_cache

    def _layer_parameter_costs(
        self,
        module: nn.Module,
        rank: int,
        head_num: int,
    ) -> tuple[int, int]:
        dense = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        if isinstance(module, nn.Conv2d):
            input_dim = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            output_dim = module.out_channels
        else:
            input_dim = module.in_features
            output_dim = module.out_features
        factorized = rank * input_dim + head_num * rank * output_dim
        if module.bias is not None:
            factorized += head_num * output_dim
        factorized += head_num * rank + head_num
        return dense, factorized

    def _inhernet_parameter_budget(
        self,
        target_layers: Mapping[str, nn.Module],
        rank: int,
        head_num: int,
    ) -> int:
        """Exact total size of the corresponding uniform-rank InherNet."""
        total = sum(parameter.numel() for parameter in self.parameters())
        for module in target_layers.values():
            dense, factorized = self._layer_parameter_costs(module, rank, head_num)
            max_rank = min(module.weight.shape[0], module.weight[0].numel())
            if rank < max_rank:
                total += factorized - dense
        return total

    def _registered_rank_configuration(
        self,
        target_layers: Mapping[str, nn.Module],
        svd_cache: Mapping[str, Mapping[str, torch.Tensor]],
        *,
        parameter_budget: int,
        head_num: int,
        reference_rank: int,
        applications_per_example: Mapping[str, float],
        allocation_scale: str,
    ) -> tuple[dict[str, int], dict[str, object]]:
        """Construct the maintained fixed-rank policy without invoking research allocation."""
        if allocation_scale not in {FINAL_HETERO_ALLOCATION, "unweighted_uniform"}:
            raise ValueError(f"Not a registered-rank Hetero policy: {allocation_scale}")
        source_total = sum(parameter.numel() for parameter in self.parameters())
        dense_target_total = sum(
            module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            for module in target_layers.values()
        )
        fixed_cost = source_total - dense_target_total
        selected_cost = fixed_cost
        rank_map: dict[str, int] = {}
        dense_choices: list[str] = []
        allocation_map: dict[str, int | str] = {}
        allocation_layers: dict[str, dict[str, float | int | str]] = {}
        for name, module in target_layers.items():
            singular_energy = svd_cache[name]["s"].detach().cpu().double().square()
            total_energy = float(singular_energy.sum().item())
            max_rank = int(singular_energy.numel())
            if reference_rank < max_rank:
                rank = reference_rank
                _, cost = self._layer_parameter_costs(module, rank, head_num)
                residual = float(singular_energy[rank:].sum().item())
                rank_map[name] = rank
                allocation_map[name] = rank
            else:
                cost = module.weight.numel() + (
                    module.bias.numel() if module.bias is not None else 0
                )
                residual = 0.0
                dense_choices.append(name)
                allocation_map[name] = "dense"
            selected_cost += cost
            relative_residual = residual / total_energy if total_energy > 0 else 0.0
            allocation_layers[name] = {
                "applications_per_example": float(applications_per_example[name]),
                "raw_total_energy": total_energy,
                "allocation_total_energy": total_energy,
                "max_rank": max_rank,
                "minimum_factorized_rank": reference_rank,
                "choice": allocation_map[name],
                "selected_parameter_cost": int(cost),
                "predicted_residual": residual,
                "relative_predicted_residual": relative_residual,
                "retained_energy_fraction": 1.0 - relative_residual,
            }
        if selected_cost != parameter_budget:
            raise RuntimeError(
                "Registered-rank Hetero construction disagrees with the InherNet count: "
                f"selected={selected_cost}, inhernet={parameter_budget}."
            )
        residuals = [
            float(layer["relative_predicted_residual"])
            for layer in allocation_layers.values()
        ]
        report: dict[str, object] = {
            "protocol": (
                "activation_weighted_registered_rank"
                if allocation_scale == FINAL_HETERO_ALLOCATION
                else "weight_only_registered_rank"
            ),
            "allocator": "fixed_registered_rank",
            "allocation_scale": allocation_scale,
            "allocation_objective": "fixed_registered_rank",
            "decomposition_metric": (
                "activation_weighted"
                if allocation_scale == FINAL_HETERO_ALLOCATION
                else "weight_only"
            ),
            "reference_inhernet_rank": reference_rank,
            "protected_inheritance_rank": None,
            "reference_inhernet_parameters": parameter_budget,
            "source_parameters": source_total,
            "fixed_parameters": fixed_cost,
            "dense_target_parameters": dense_target_total,
            "requested_parameters": parameter_budget,
            "requested_target_parameters": parameter_budget - fixed_cost,
            "minimum_feasible_parameters": selected_cost,
            "selected_parameters": selected_cost,
            "selected_target_parameters": selected_cost - fixed_cost,
            "achieved_ratio": selected_cost / max(source_total, 1),
            "achieved_target_ratio": (selected_cost - fixed_cost) / max(dense_target_total, 1),
            "budget_slack": 0,
            "budget_utilization": 1.0,
            "target_layer_count": len(target_layers),
            "factorized_layer_count": len(rank_map),
            "dense_layer_count": len(dense_choices),
            "allocation_map": allocation_map,
            "allocation_layers": allocation_layers,
            "kept_dense_layers": sorted(dense_choices),
            "max_predicted_relative_residual": max(residuals, default=0.0),
            "sum_predicted_relative_residual": sum(residuals),
        }
        return rank_map, report

    def _allocate_research_ranks_by_cost(
        self,
        target_layers: Mapping[str, nn.Module],
        svd_cache: Mapping[str, Mapping[str, torch.Tensor]],
        *,
        parameter_budget: int,
        head_num: int,
        reference_rank: int,
        protected_rank: int,
        applications_per_example: Mapping[str, float],
        allocation_scale: str,
    ) -> tuple[dict[str, int], dict[str, object]]:
        """Diagnostic rank allocator; unreachable from maintained Hetero runs."""
        if not allocation_scale.startswith("research_"):
            raise ValueError(f"Not a research rank policy: {allocation_scale}")
        source_total = sum(parameter.numel() for parameter in self.parameters())
        dense_target_total = sum(
            module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            for module in target_layers.values()
        )
        fixed_cost = source_total - dense_target_total
        requested_budget = parameter_budget
        requested_target_budget = requested_budget - fixed_cost
        candidates: dict[str, list[tuple[int, int, float]]] = {}
        layer_score_metadata: dict[str, dict[str, float | int]] = {}
        for name, module in target_layers.items():
            dense, _ = self._layer_parameter_costs(
                module, 1, head_num
            )
            singular_energy = svd_cache[name]["s"].detach().cpu().double().square()
            applications = applications_per_example[name]
            raw_total_energy = float(singular_energy.sum().item())
            if allocation_scale == "research_total_output":
                singular_energy = singular_energy * applications
            elif allocation_scale in {"research_nested_relative", "research_relative"}:
                singular_energy = singular_energy / max(raw_total_energy, 1e-30)
            max_rank = singular_energy.numel()
            minimum_rank = (
                min(max_rank, protected_rank)
                if allocation_scale == "research_nested_relative"
                else 1
            )
            tail_energy = torch.zeros(max_rank + 1, dtype=singular_energy.dtype)
            tail_energy[:-1] = torch.flip(
                torch.cumsum(torch.flip(singular_energy, dims=(0,)), dim=0),
                dims=(0,),
            )
            layer_candidates: list[tuple[int, int, float]] = []
            if allocation_scale == "research_nested_relative" and reference_rank >= max_rank:
                # Do not introduce a new bottleneck where matched InherNet leaves
                # the layer dense. Heterogeneity redistributes capacity only
                # among layers that the reference model actually factorizes.
                candidates[name] = [(0, dense, 0.0)]
            else:
                for rank in range(minimum_rank, max_rank + 1):
                    _, cost = self._layer_parameter_costs(module, rank, head_num)
                    if cost < dense:
                        distortion = float(tail_energy[rank].item())
                        layer_candidates.append((rank, cost, distortion))
                if not layer_candidates:
                    candidates[name] = [(0, dense, 0.0)]
                else:
                    # Dense is a valid final candidate and prevents forced approximation.
                    layer_candidates.append((0, dense, 0.0))
                    candidates[name] = layer_candidates
            layer_score_metadata[name] = {
                "applications_per_example": float(applications),
                "raw_total_energy": raw_total_energy,
                "allocation_total_energy": float(singular_energy.sum().item()),
                "max_rank": int(max_rank),
                "minimum_factorized_rank": int(minimum_rank),
            }

        positions: dict[str, int] = {name: 0 for name in candidates}
        selected_cost = fixed_cost
        for name, layer_candidates in candidates.items():
            selected_cost += layer_candidates[0][1]

        minimum_feasible = selected_cost
        minimum_target_cost = minimum_feasible - fixed_cost
        if requested_target_budget < minimum_target_cost:
            raise ValueError(
                "Hetero cannot fit under its reference InherNet parameter cap: "
                f"budget={requested_budget}, minimum={minimum_feasible}."
            )

        while True:
            best: tuple[float, str, int, int] | None = None
            for name, layer_candidates in candidates.items():
                position = positions[name]
                if position + 1 >= len(layer_candidates):
                    continue
                current = layer_candidates[position]
                nxt = layer_candidates[position + 1]
                delta_cost = nxt[1] - current[1]
                if delta_cost <= 0 or selected_cost + delta_cost > requested_budget:
                    continue
                gain_per_cost = (current[2] - nxt[2]) / delta_cost
                proposal = (gain_per_cost, name, position + 1, delta_cost)
                if best is None or proposal[0] > best[0] or (
                    proposal[0] == best[0] and proposal[1] < best[1]
                ):
                    best = proposal
            if best is None:
                break
            _, name, next_position, delta_cost = best
            positions[name] = next_position
            selected_cost += delta_cost

        rank_map: dict[str, int] = {}
        dense_choices: list[str] = []
        allocation_map: dict[str, int | str] = {}
        allocation_layers: dict[str, dict[str, float | int | str]] = {}
        for name, position in positions.items():
            rank = candidates[name][position][0]
            selected_distortion = float(candidates[name][position][2])
            if rank == 0:
                dense_choices.append(name)
                allocation_map[name] = "dense"
            else:
                rank_map[name] = rank
                allocation_map[name] = rank
            total_energy = float(layer_score_metadata[name]["allocation_total_energy"])
            relative_residual = selected_distortion / total_energy if total_energy > 0 else 0.0
            allocation_layers[name] = {
                **layer_score_metadata[name],
                "choice": allocation_map[name],
                "selected_parameter_cost": int(candidates[name][position][1]),
                "predicted_residual": selected_distortion,
                "relative_predicted_residual": relative_residual,
                "retained_energy_fraction": 1.0 - relative_residual,
            }
        report: dict[str, object] = {
            "protocol": "research_rank_allocation",
            "allocator": "research_marginal_gain_per_parameter",
            "allocation_scale": allocation_scale,
            "allocation_objective": "marginal_gain_per_parameter",
            "decomposition_metric": "activation_weighted",
            "reference_inhernet_rank": reference_rank,
            "protected_inheritance_rank": (
                protected_rank if allocation_scale == "research_nested_relative" else None
            ),
            "reference_inhernet_parameters": requested_budget,
            "source_parameters": source_total,
            "fixed_parameters": fixed_cost,
            "dense_target_parameters": dense_target_total,
            "requested_parameters": requested_budget,
            "requested_target_parameters": requested_target_budget,
            "minimum_feasible_parameters": minimum_feasible,
            "selected_parameters": selected_cost,
            "selected_target_parameters": selected_cost - fixed_cost,
            "achieved_ratio": selected_cost / max(source_total, 1),
            "achieved_target_ratio": (selected_cost - fixed_cost) / max(dense_target_total, 1),
            "budget_slack": requested_budget - selected_cost,
            "budget_utilization": selected_cost / max(requested_budget, 1),
            "target_layer_count": len(target_layers),
            "factorized_layer_count": len(rank_map),
            "dense_layer_count": len(dense_choices),
            "allocation_map": allocation_map,
            "allocation_layers": allocation_layers,
            "kept_dense_layers": sorted(dense_choices),
        }
        selected_residuals = [
            float(layer["relative_predicted_residual"]) for layer in allocation_layers.values()
        ]
        report["max_predicted_relative_residual"] = max(selected_residuals, default=0.0)
        report["sum_predicted_relative_residual"] = sum(selected_residuals)
        return rank_map, report

    def _replace_conv_with_hetero_svd(
        self,
        module: nn.Conv2d,
        rank: int,
        head_num: int,
        svd_pack: Mapping[str, torch.Tensor],
        expert_noise_scale: float,
    ) -> nn.Module:
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
        v_scaled = s_sqrt.unsqueeze(1) * v_h_trunc
        if whiten_inv.ndim == 1:
            v_4d = v_scaled.view(rank, c_in, k_h, k_w)
            conv1_weight = (v_4d * whiten_inv.view(1, c_in, 1, 1)).contiguous()
        elif whiten_inv.shape[0] == c_in * k_h * k_w:
            conv1_weight = v_scaled.matmul(whiten_inv).view(rank, c_in, k_h, k_w).contiguous()
        else:
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
            padding_mode=module.padding_mode,
            bias=False,
        )
        with torch.no_grad():
            conv1.weight.copy_(conv1_weight)
        expert_weight = (u_trunc * s_sqrt.unsqueeze(0)).contiguous().view(c_out, rank, 1, 1)
        expert_noise = self._build_zero_mean_expert_noise(
            expert_weight,
            head_num,
            expert_noise_scale,
        )
        fused_weight = expert_weight.unsqueeze(0).expand(head_num, -1, -1, -1, -1).clone()
        if expert_noise is not None:
            fused_weight.add_(expert_noise)
        lift_statistics = self._expert_lift_statistics(expert_weight, fused_weight)
        fused_weight = fused_weight.reshape(head_num * c_out, rank, 1, 1)
        fused_bias = module.bias.detach().repeat(head_num) if module.bias is not None else None
        replacement = GatedSVDConv2d(
            conv1,
            fused_weight,
            fused_bias,
            head_num,
        )
        replacement._hetero_lift_statistics = lift_statistics
        return replacement

    def _replace_linear_with_hetero_svd(
        self,
        module: nn.Linear,
        rank: int,
        head_num: int,
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
        weighted_factor = s_sqrt.unsqueeze(1) * v_h_trunc
        linear1_weight = (
            weighted_factor * whiten_inv
            if whiten_inv.ndim == 1
            else weighted_factor @ whiten_inv
        ).contiguous()
        linear1 = nn.Linear(module.in_features, rank, bias=False)
        with torch.no_grad():
            linear1.weight.copy_(linear1_weight)
        expert_weight = (u_trunc * s_sqrt.unsqueeze(0)).contiguous()
        expert_noise = self._build_zero_mean_expert_noise(
            expert_weight,
            head_num,
            expert_noise_scale,
        )
        fused_weight = expert_weight.unsqueeze(0).expand(head_num, -1, -1).clone()
        if expert_noise is not None:
            fused_weight.add_(expert_noise)
        lift_statistics = self._expert_lift_statistics(expert_weight, fused_weight)
        fused_weight = fused_weight.reshape(head_num * module.out_features, rank)
        fused_bias = module.bias.detach().repeat(head_num) if module.bias is not None else None
        replacement = GatedSVDLinear(
            linear1,
            fused_weight,
            fused_bias,
            head_num,
        )
        replacement._hetero_lift_statistics = lift_statistics
        return replacement

    def _replace_module_with_hetero_svd(
        self,
        module: nn.Module,
        rank: int,
        head_num: int,
        svd_pack: Mapping[str, torch.Tensor],
        expert_noise_scale: float,
    ) -> nn.Module:
        if isinstance(module, nn.Conv2d):
            replacement = self._replace_conv_with_hetero_svd(
                module,
                rank,
                head_num,
                svd_pack,
                expert_noise_scale,
            )
            return self._match_module_device_dtype(replacement, module)
        if isinstance(module, nn.Linear):
            replacement = self._replace_linear_with_hetero_svd(
                module,
                rank,
                head_num,
                svd_pack,
                expert_noise_scale,
            )
            return self._match_module_device_dtype(replacement, module)
        raise TypeError(f"Unsupported Hetero target layer: {type(module).__name__}")

    def apply_hetero_svd(
        self,
        calib_loader: DataLoader,
        head_num: int = 3,
        reference_rank: int = 8,
        max_calib_batches: int = 16,
        svd_backend: str = SVD_BACKEND_AUTO,
        expert_noise_scale: float = 0.01,
        compress_linear: bool = False,
        max_features_per_batch: int = 4096,
        second_moment_shrinkage: float = 0.01,
        allocation_scale: str = FINAL_HETERO_ALLOCATION,
        research_protected_rank: int | None = None,
        allow_research_rank_probe: bool = False,
    ) -> tuple[dict[str, int], str]:
        config = HeteroConfig(
            head_num=head_num,
            reference_rank=reference_rank,
            max_calib_batches=max_calib_batches,
            expert_noise_scale=expert_noise_scale,
            compress_linear=compress_linear,
            max_features_per_batch=max_features_per_batch,
            second_moment_shrinkage=second_moment_shrinkage,
            allocation_scale=allocation_scale,
            research_protected_rank=research_protected_rank,
        )
        if (
            config.allocation_scale in RESEARCH_HETERO_RANK_POLICIES
            and not allow_research_rank_probe
        ):
            raise ValueError(
                "Research rank policies require explicit diagnostics-only opt-in."
            )
        last_error: StableSVDDecompositionError | None = None
        reference_device = next(self.parameters()).device
        moments, moment_metadata = self._estimate_input_second_moments(
            calib_loader,
            max_batches=config.max_calib_batches,
            include_linear=config.compress_linear,
            max_features_per_batch=config.max_features_per_batch,
            shrinkage=config.second_moment_shrinkage,
        )
        if config.allocation_scale == "unweighted_uniform":
            moments = {
                name: (
                    torch.ones_like(moment)
                    if moment.ndim == 1
                    else torch.eye(moment.shape[0], device=moment.device, dtype=moment.dtype)
                )
                for name, moment in moments.items()
            }
        for backend in _candidate_svd_backends(reference_device, svd_backend):
            try:
                svd_cache = self._compute_weighted_svd_cache(
                    moments,
                    svd_backend=backend,
                    include_linear=config.compress_linear,
                )
                target_layers = self._collect_hetero_target_layers(include_linear=config.compress_linear)
                parameter_budget = self._inhernet_parameter_budget(
                    target_layers,
                    config.reference_rank,
                    config.head_num,
                )
                applications = {
                    name: float(layer_metadata["applications_per_example"])
                    for name, layer_metadata in moment_metadata.items()
                }
                if config.allocation_scale in {FINAL_HETERO_ALLOCATION, "unweighted_uniform"}:
                    rank_map, report = self._registered_rank_configuration(
                        target_layers,
                        svd_cache,
                        parameter_budget=parameter_budget,
                        head_num=config.head_num,
                        reference_rank=config.reference_rank,
                        applications_per_example=applications,
                        allocation_scale=config.allocation_scale,
                    )
                else:
                    rank_map, report = self._allocate_research_ranks_by_cost(
                        target_layers,
                        svd_cache,
                        parameter_budget=parameter_budget,
                        head_num=config.head_num,
                        reference_rank=config.reference_rank,
                        protected_rank=int(config.research_protected_rank or 1),
                        applications_per_example=applications,
                        allocation_scale=config.allocation_scale,
                    )
                replacements: list[tuple[str, nn.Module, str, nn.Module]] = []
                for name, module in target_layers.items():
                    if name not in rank_map:
                        continue
                    parent, child_name = self._get_parent_module(name)
                    replacement = self._replace_module_with_hetero_svd(
                        module,
                        rank=rank_map[name],
                        head_num=config.head_num,
                        svd_pack=svd_cache[name],
                        expert_noise_scale=config.expert_noise_scale,
                    )
                    replacements.append((name, parent, child_name, replacement))
            except StableSVDDecompositionError as exc:
                last_error = exc
                continue
            for _, parent, child_name, replacement in replacements:
                setattr(parent, child_name, replacement)
            self._cached_routers = tuple(
                module for module in self.modules() if isinstance(module, LoadBalancedRouter)
            )
            actual_parameters = sum(parameter.numel() for parameter in self.parameters())
            if actual_parameters != report["selected_parameters"]:
                raise RuntimeError(
                    "Hetero allocator/implementation parameter mismatch: "
                    f"predicted={report['selected_parameters']}, actual={actual_parameters}."
                )
            if actual_parameters > report["requested_parameters"]:
                raise RuntimeError("Hetero model exceeded its requested parameter budget.")
            if config.allocation_scale in {FINAL_HETERO_ALLOCATION, "unweighted_uniform"}:
                if actual_parameters != parameter_budget:
                    raise RuntimeError(
                        "Registered-rank Hetero must exactly match its InherNet parameter count: "
                        f"hetero={actual_parameters}, inhernet={parameter_budget}."
                    )
                if any(rank != config.reference_rank for rank in rank_map.values()):
                    raise RuntimeError("Registered-rank Hetero produced a non-uniform factor rank.")
            lift_layers = {
                name: dict(getattr(replacement, "_hetero_lift_statistics"))
                for name, _, _, replacement in replacements
            }
            mean_shifts = [
                float(layer["relative_expert_mean_shift"])
                for layer in lift_layers.values()
            ]
            diversities = [
                float(layer["relative_expert_diversity"])
                for layer in lift_layers.values()
            ]
            report["conditional_lift_probe"] = {
                "expert_noise_scale": config.expert_noise_scale,
                "factorized_layer_count": len(lift_layers),
                "mean_relative_expert_mean_shift": (
                    sum(mean_shifts) / len(mean_shifts) if mean_shifts else 0.0
                ),
                "max_relative_expert_mean_shift": max(mean_shifts, default=0.0),
                "mean_relative_expert_diversity": (
                    sum(diversities) / len(diversities) if diversities else 0.0
                ),
                "max_relative_expert_diversity": max(diversities, default=0.0),
                "per_layer": lift_layers,
            }
            report["actual_parameters"] = actual_parameters
            report["second_moments"] = moment_metadata
            self.rank_map = dict(rank_map)
            self.hetero_report = report
            return rank_map, backend
        if last_error is not None:
            raise last_error
        raise RuntimeError("No SVD backend candidate was available for HeteroNet.")


def _gating_routers(model: nn.Module) -> tuple[LoadBalancedRouter, ...]:
    routers = getattr(model, "_cached_routers", None)
    if routers is None:
        routers = tuple(module for module in model.modules() if isinstance(module, LoadBalancedRouter))
    return routers


def compute_gating_load_balance_loss(model: nn.Module) -> torch.Tensor | None:
    losses = []
    routers = _gating_routers(model)
    for router in routers:
        aux_loss = router.load_balance_loss()
        if aux_loss is not None:
            losses.append(aux_loss)
        router._last_gating_probs = None
    if not losses:
        return None
    return torch.stack(losses).mean()


def clear_gating_router_cache(model: nn.Module) -> None:
    for router in _gating_routers(model):
        router._last_gating_probs = None


def freeze_gating_routers(model: nn.Module) -> None:
    """Keep the initialized uniform router fixed while retaining all expert parameters."""
    for router in _gating_routers(model):
        router.gate.requires_grad_(False)
