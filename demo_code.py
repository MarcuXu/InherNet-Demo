from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import random
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, resnet50

from cifar100_models import PAIR_REGISTRY as CIFAR100_PAIR_REGISTRY
from cifar100_models import build_model as build_cifar100_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PYPLOT = None
_PLOT_THEME_APPLIED = False
RUN_LOG_ENV_VAR = "INHERNET_RUN_LOG"
SUITE_LOG_DIR_ENV_VAR = "INHERNET_SUITE_LOG_DIR"
RUN_METADATA_PREFIX = "RUN_METADATA"
RUN_METRICS_PREFIX = "RUN_METRICS"
PLOT_METRIC_SPECS = (
    ("train_loss", "Loss", "Train Loss", None),
    ("test_loss", "Loss", "Test Loss", None),
    ("train_accuracy", "Top-1 Accuracy (%)", "Train Accuracy", (0.0, 100.0)),
    ("test_accuracy", "Top-1 Accuracy (%)", "Test Accuracy", (0.0, 100.0)),
)
METHOD_CHOICES = ["teacher", "student", "student_kd", "inhernet", "hetero"]


@dataclass(frozen=True)
class TrainSettings:
    optimizer_name: str
    batch_size: int
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    lr_milestones: tuple[int, ...]
    lr_gamma: float = 0.1
    kd_temperature: float = 2.0
    kd_loss_weight: float = 9.0
    ce_loss_weight: float = 0.1
    default_head_num: int = 3


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    dataset_class: type
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    train_settings: TrainSettings
    pair_registry: Mapping[str, Mapping[str, Any]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return DEFAULT_DEVICE
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def build_cifar_torchvision_resnet(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif arch == "resnet50":
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown torchvision ResNet architecture: {arch}")
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


CIFAR10_PAIR_REGISTRY: dict[str, Mapping[str, Any]] = {
    "resnet50_to_resnet18": {
        "teacher_name": "resnet50",
        "student_name": "resnet18",
        "teacher_builder": lambda num_classes: build_cifar_torchvision_resnet("resnet50", num_classes),
        "student_builder": lambda num_classes: build_cifar_torchvision_resnet("resnet18", num_classes),
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
    }
}


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        num_classes=10,
        dataset_class=torchvision.datasets.CIFAR10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        train_settings=TrainSettings(
            optimizer_name="sgd",
            batch_size=128,
            epochs=200,
            lr=0.05,
            momentum=0.9,
            weight_decay=5e-4,
            lr_milestones=(100, 150, 180),
            kd_temperature=2.0,
            kd_loss_weight=9.0,
            ce_loss_weight=0.1,
            default_head_num=3,
        ),
        pair_registry=CIFAR10_PAIR_REGISTRY,
    ),
    "cifar100": DatasetSpec(
        num_classes=100,
        dataset_class=torchvision.datasets.CIFAR100,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        train_settings=TrainSettings(
            optimizer_name="sgd",
            batch_size=64,
            epochs=240,
            lr=0.05,
            momentum=0.9,
            weight_decay=5e-4,
            lr_milestones=(150, 180, 210),
            kd_temperature=2.0,
            kd_loss_weight=9.0,
            ce_loss_weight=0.1,
            default_head_num=3,
        ),
        pair_registry=CIFAR100_PAIR_REGISTRY,
    ),
}


SUITE_SPECS: dict[str, list[dict[str, Any]]] = {
    "baseline": [
        {"label": "01_teacher", "method": "teacher"},
        {"label": "02_student", "method": "student"},
        {"label": "03_student_kd", "method": "student_kd"},
    ],
    "comparison": [
        {"label": "01_teacher", "method": "teacher"},
        {"label": "02_student_kd", "method": "student_kd"},
        {"label": "03_inhernet_small", "method": "inhernet", "rank_preset": "small"},
        {"label": "04_inhernet_large", "method": "inhernet", "rank_preset": "large"},
        {"label": "05_hetero", "method": "hetero"},
    ],
    "all": [
        {"label": "01_teacher", "method": "teacher"},
        {"label": "02_student", "method": "student"},
        {"label": "03_student_kd", "method": "student_kd"},
        {"label": "04_inhernet_small", "method": "inhernet", "rank_preset": "small"},
        {"label": "05_inhernet_large", "method": "inhernet", "rank_preset": "large"},
        {"label": "06_hetero", "method": "hetero"},
    ],
}


def get_pair_spec(dataset_name: str, pair_name: str) -> Mapping[str, Any]:
    dataset_spec = DATASET_REGISTRY[dataset_name]
    if pair_name not in dataset_spec.pair_registry:
        available = ", ".join(sorted(dataset_spec.pair_registry.keys()))
        raise ValueError(f"Unknown pair '{pair_name}' for dataset '{dataset_name}'. Available: {available}")
    return dataset_spec.pair_registry[pair_name]


def get_role_name(pair_spec: Mapping[str, Any], role: str) -> str:
    if role == "teacher":
        return str(pair_spec.get("teacher_name", pair_spec["teacher"]))
    if role == "student":
        return str(pair_spec.get("student_name", pair_spec["student"]))
    raise KeyError(f"Unknown role: {role}")


def build_pair_model(dataset_name: str, pair_name: str, role: str, num_classes: int) -> nn.Module:
    pair_spec = get_pair_spec(dataset_name, pair_name)
    builder_key = f"{role}_builder"
    if builder_key in pair_spec:
        return pair_spec[builder_key](num_classes)
    model_name = get_role_name(pair_spec, role)
    if dataset_name == "cifar100":
        return build_cifar100_model(model_name, num_classes)
    raise ValueError(f"No builder registered for {dataset_name}:{pair_name}:{role}")


def get_transforms(dataset_name: str) -> tuple[transforms.Compose, transforms.Compose]:
    dataset_spec = DATASET_REGISTRY[dataset_name]
    normalize = transforms.Normalize(dataset_spec.mean, dataset_spec.std)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, test_transform


def get_dataset(dataset_name: str, root: str, train: bool, download: bool):
    dataset_spec = DATASET_REGISTRY[dataset_name]
    train_transform, test_transform = get_transforms(dataset_name)
    return dataset_spec.dataset_class(
        root=root,
        train=train,
        download=download,
        transform=train_transform if train else test_transform,
    )


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    root: str,
    download: bool,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_set = get_dataset(dataset_name, root=root, train=True, download=download)
    test_set = get_dataset(dataset_name, root=root, train=False, download=download)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=DEFAULT_DEVICE.type == "cuda",
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEFAULT_DEVICE.type == "cuda",
    )
    return train_loader, test_loader


class GatedSumLinear(nn.Module):
    def __init__(self, linear_list: nn.ModuleList, input_dim: int, head_num: int) -> None:
        super().__init__()
        self.linear_list = linear_list
        self.head_num = head_num
        self.gate = nn.Linear(input_dim, head_num)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        gating_scores = self.gate(x)
        gating_weights = F.softmax(gating_scores, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.linear_list], dim=-1)
        return torch.sum(gating_weights.unsqueeze(1) * expert_outputs, dim=-1)


class GatedSumConv2d(nn.Module):
    def __init__(self, conv_list: nn.ModuleList, input_dim: int, head_num: int) -> None:
        super().__init__()
        self.conv_list = conv_list
        self.head_num = head_num
        self.gate = nn.Linear(input_dim, head_num)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

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
        self._last_gating_probs: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        compressed = self.linear1(x)
        expert_outputs = torch.stack([layer(compressed) for layer in self.linear_list], dim=-1)
        gate_feat = x if self.use_uncompressed_gate else compressed
        gating_scores = self.gate(gate_feat)
        gating_probs = F.softmax(gating_scores, dim=-1)
        self._last_gating_probs = gating_probs
        return torch.sum(gating_probs.unsqueeze(1) * expert_outputs, dim=-1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

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
    def _replace_linear_with_svd(self, module: nn.Linear, rank: int, head_num: int) -> nn.Module:
        weight = module.weight.data
        u, s, v_h = torch.linalg.svd(weight, full_matrices=False)
        if rank >= s.numel():
            return module
        r = min(rank, s.numel())
        linear1 = nn.Linear(module.in_features, r, bias=False)
        linear1.weight.data = v_h[:r, :].contiguous()
        expert_layers = nn.ModuleList()
        for _ in range(head_num):
            linear2 = nn.Linear(r, module.out_features, bias=module.bias is not None)
            linear2.weight.data = (u[:, :r] @ torch.diag(s[:r])).contiguous()
            if module.bias is not None:
                linear2.bias.data = module.bias.data.clone()
            expert_layers.append(linear2)
        return nn.Sequential(linear1, GatedSumLinear(expert_layers, r, head_num))

    def _replace_conv_with_svd(self, module: nn.Conv2d, rank: int, head_num: int) -> nn.Module:
        if module.groups != 1:
            return module
        weight = module.weight.data
        c_out, c_in, k_h, k_w = weight.shape
        weight_flat = weight.view(c_out, -1)
        u, s, v_h = torch.linalg.svd(weight_flat, full_matrices=False)
        if rank >= s.numel():
            return module
        r = min(rank, s.numel())
        conv1 = nn.Conv2d(
            c_in,
            r,
            kernel_size=(k_h, k_w),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            bias=False,
        )
        conv1.weight.data = v_h[:r, :].contiguous().view(r, c_in, k_h, k_w)
        expert_layers = nn.ModuleList()
        for _ in range(head_num):
            conv2 = nn.Conv2d(r, c_out, kernel_size=1, stride=1, padding=0, bias=module.bias is not None)
            conv2.weight.data = (u[:, :r] @ torch.diag(s[:r])).contiguous().view(c_out, r, 1, 1)
            if module.bias is not None:
                conv2.bias.data = module.bias.data.clone()
            expert_layers.append(conv2)
        return nn.Sequential(conv1, GatedSumConv2d(expert_layers, r, head_num))

    def _replace_module_with_svd(self, module: nn.Module, rank: int, head_num: int) -> nn.Module:
        if isinstance(module, nn.Conv2d):
            replacement = self._replace_conv_with_svd(module, rank, head_num)
            return self._match_module_device_dtype(replacement, module)
        if isinstance(module, nn.Linear):
            replacement = self._replace_linear_with_svd(module, rank, head_num)
            return self._match_module_device_dtype(replacement, module)
        return module

    def apply_svd(self, rank: int, head_num: int) -> None:
        for name, module in self._collect_target_layers().items():
            parent, child_name = self._get_parent_module(name)
            parent._modules[child_name] = self._replace_module_with_svd(module, rank, head_num)


class GenericHeteroNet(BackboneWrapper):
    def _replace_linear_with_rank_map(
        self,
        module: nn.Linear,
        rank: int,
        head_num: int,
        compress_threshold: int,
    ) -> nn.Module:
        rank = max(1, min(rank, min(module.in_features, module.out_features)))
        linear1 = nn.Linear(module.in_features, rank, bias=False)
        expert_layers = nn.ModuleList(
            [
                nn.Linear(rank, module.out_features, bias=module.bias is not None)
                for _ in range(head_num)
            ]
        )
        use_uncompressed_gate = rank < compress_threshold
        gate_input_dim = module.in_features if use_uncompressed_gate else rank
        return DecoupledGatedSVDLinear(
            linear1,
            expert_layers,
            gate_input_dim,
            head_num,
            use_uncompressed_gate,
        )

    def _replace_conv_with_rank_map(
        self,
        module: nn.Conv2d,
        rank: int,
        head_num: int,
        compress_threshold: int,
    ) -> nn.Module:
        if module.groups != 1:
            return module
        rank = max(1, min(rank, min(module.in_channels * module.kernel_size[0] * module.kernel_size[1], module.out_channels)))
        conv1 = nn.Conv2d(
            module.in_channels,
            rank,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            bias=False,
        )
        expert_layers = nn.ModuleList(
            [
                nn.Conv2d(rank, module.out_channels, kernel_size=1, stride=1, padding=0, bias=module.bias is not None)
                for _ in range(head_num)
            ]
        )
        use_uncompressed_gate = rank < compress_threshold
        gate_input_dim = module.in_channels if use_uncompressed_gate else rank
        return DecoupledGatedSVDConv2d(
            conv1,
            expert_layers,
            gate_input_dim,
            head_num,
            use_uncompressed_gate,
        )

    def _extract_input_features(self, module: nn.Module, layer_input: torch.Tensor) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            return torch.mean(layer_input, dim=(2, 3))
        if layer_input.ndim > 2:
            return torch.flatten(layer_input, 1)
        return layer_input

    def _stable_cholesky(self, matrix: torch.Tensor, base_eps: float = 1e-5) -> torch.Tensor:
        eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        jitter = base_eps
        for _ in range(5):
            try:
                return torch.linalg.cholesky(matrix + jitter * eye)
            except RuntimeError:
                jitter *= 10.0
        return torch.linalg.cholesky(matrix + jitter * eye)

    def _estimate_input_covariances(
        self,
        calib_loader: DataLoader,
        max_batches: int = 16,
        eps: float = 1e-5,
    ) -> dict[str, torch.Tensor]:
        target_layers = self._collect_target_layers()
        stats = {
            name: {"sum": None, "sum_outer": None, "count": 0}
            for name in target_layers.keys()
        }
        handles = []

        def make_hook(layer_name: str, layer_module: nn.Module):
            def hook(_, layer_input, __):
                features = self._extract_input_features(layer_module, layer_input[0].detach())
                features = features.view(features.shape[0], -1)
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
                _ = self(inputs.to(next(self.parameters()).device))
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
    ) -> tuple[dict[str, float], dict[str, int], dict[str, dict[str, torch.Tensor]]]:
        target_layers = self._collect_target_layers()
        entropies: dict[str, float] = {}
        max_ranks: dict[str, int] = {}
        svd_cache: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in target_layers.items():
            weight = module.weight.data
            chol_c = self._stable_cholesky(covariances[name])
            whitened_weight = self._whiten_weight(module, weight, chol_c)
            weight_flat = whitened_weight.view(whitened_weight.shape[0], -1)
            u, s, v_h = torch.linalg.svd(weight_flat, full_matrices=False)
            s_sq = s**2
            sigma_sum = s_sq.sum().clamp_min(1e-12)
            probs = (s_sq / sigma_sum).clamp_min(1e-12)
            entropies[name] = (-(probs * torch.log(probs)).sum()).item()
            max_ranks[name] = s.numel()
            svd_cache[name] = {"u": u, "s": s, "v_h": v_h, "chol_c": chol_c}
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

    def _replace_linear_with_hetero_svd(
        self,
        module: nn.Linear,
        rank: int,
        head_num: int,
        compress_threshold: int,
        svd_pack: Mapping[str, torch.Tensor],
    ) -> nn.Module:
        weight = module.weight.data
        u = svd_pack["u"]
        s = svd_pack["s"]
        v_h = svd_pack["v_h"]
        chol_c = svd_pack["chol_c"]
        rank = max(1, min(rank, s.numel()))
        u_trunc = u[:, :rank]
        s_trunc = s[:rank]
        v_h_trunc = v_h[:rank, :]
        s_sqrt = torch.sqrt(torch.clamp(s_trunc, min=1e-12))
        whiten_inv = torch.linalg.inv(chol_c)
        linear1 = nn.Linear(module.in_features, rank, bias=False)
        linear1.weight.data = (torch.diag(s_sqrt) @ v_h_trunc).matmul(whiten_inv).contiguous()
        expert_layers = nn.ModuleList()
        for _ in range(head_num):
            linear2 = nn.Linear(rank, module.out_features, bias=module.bias is not None)
            base_weight = (u_trunc @ torch.diag(s_sqrt) / head_num).contiguous()
            noise_scale = 0.01 * base_weight.std().clamp_min(1e-12)
            linear2.weight.data = base_weight + torch.randn_like(base_weight) * noise_scale
            if module.bias is not None:
                linear2.bias.data = module.bias.data.clone() / head_num
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

    def _replace_conv_with_hetero_svd(
        self,
        module: nn.Conv2d,
        rank: int,
        head_num: int,
        compress_threshold: int,
        svd_pack: Mapping[str, torch.Tensor],
    ) -> nn.Module:
        if module.groups != 1:
            return module
        weight = module.weight.data
        c_out, c_in, k_h, k_w = weight.shape
        u = svd_pack["u"]
        s = svd_pack["s"]
        v_h = svd_pack["v_h"]
        chol_c = svd_pack["chol_c"]
        rank = max(1, min(rank, s.numel()))
        u_trunc = u[:, :rank]
        s_trunc = s[:rank]
        v_h_trunc = v_h[:rank, :]
        s_sqrt = torch.sqrt(torch.clamp(s_trunc, min=1e-12))
        whiten_inv = torch.linalg.inv(chol_c)
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
        expert_layers = nn.ModuleList()
        for _ in range(head_num):
            conv2 = nn.Conv2d(rank, c_out, kernel_size=1, stride=1, padding=0, bias=module.bias is not None)
            base_weight = (u_trunc @ torch.diag(s_sqrt) / head_num).contiguous().view(c_out, rank, 1, 1)
            noise_scale = 0.01 * base_weight.std().clamp_min(1e-12)
            conv2.weight.data = base_weight + torch.randn_like(base_weight) * noise_scale
            if module.bias is not None:
                conv2.bias.data = module.bias.data.clone() / head_num
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

    def _replace_module_with_hetero_svd(
        self,
        module: nn.Module,
        rank: int,
        head_num: int,
        compress_threshold: int,
        svd_pack: Mapping[str, torch.Tensor],
    ) -> nn.Module:
        if isinstance(module, nn.Conv2d):
            replacement = self._replace_conv_with_hetero_svd(module, rank, head_num, compress_threshold, svd_pack)
            return self._match_module_device_dtype(replacement, module)
        if isinstance(module, nn.Linear):
            replacement = self._replace_linear_with_hetero_svd(module, rank, head_num, compress_threshold, svd_pack)
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
    ) -> dict[str, int]:
        covariances = self._estimate_input_covariances(calib_loader, max_batches=max_calib_batches)
        entropies, max_ranks, svd_cache = self._compute_spectral_entropies(covariances)
        rank_map = self._allocate_ranks_by_entropy(
            entropies,
            max_ranks,
            budget_ratio=budget_ratio,
            min_rank=min_rank,
            temperature=temperature,
        )
        for name, module in self._collect_target_layers().items():
            parent, child_name = self._get_parent_module(name)
            parent._modules[child_name] = self._replace_module_with_hetero_svd(
                module,
                rank=rank_map[name],
                head_num=head_num,
                compress_threshold=compress_threshold,
                svd_pack=svd_cache[name],
            )
        return rank_map

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


class RunLogger:
    def __init__(self, log_path: str | None = None, echo: bool = True, store_info_to_file: bool = True) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.echo = echo
        self.store_info_to_file = store_info_to_file

    def _write_line(self, message: str, *, echo: bool | None = None, write_to_file: bool = True) -> None:
        effective_echo = self.echo if echo is None else echo
        if write_to_file and self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{message}\n")
        if effective_echo:
            print(message)

    def info(self, message: str) -> None:
        self._write_line(message, write_to_file=self.store_info_to_file)

    def epoch(self, message: str) -> None:
        if self.log_path is None:
            self._write_line(message, echo=True, write_to_file=False)
            return
        self._write_line(message, echo=False, write_to_file=True)

    def structured(self, prefix: str, payload: Mapping[str, Any], *, echo: bool = False) -> None:
        self._write_line(f"{prefix} {json.dumps(dict(payload), sort_keys=True)}", echo=echo, write_to_file=True)

    def metadata(self, payload: Mapping[str, Any]) -> None:
        self.structured(RUN_METADATA_PREFIX, payload)

    def metrics(self, payload: Mapping[str, Any]) -> None:
        self.structured(RUN_METRICS_PREFIX, payload)


def build_run_logger(
    log_path: str | None = None,
    *,
    echo: bool = True,
    store_info_to_file: bool = True,
) -> RunLogger:
    resolved_path = os.environ.get(RUN_LOG_ENV_VAR) if log_path is None else log_path
    return RunLogger(resolved_path, echo=echo, store_info_to_file=store_info_to_file)


def to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    try:
        return [float(value) for value in list(values)]
    except TypeError:
        return []


def create_history_template() -> dict[str, list[float]]:
    return {
        "train_objective": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }


def normalize_history(history: Mapping[str, Any] | None) -> dict[str, list[float]]:
    # New runs log test_accuracy directly. eval_accuracy is only read here
    # as a backward-compatibility fallback for older logs during plotting.
    raw_history = history or {}
    normalized = create_history_template()
    normalized["train_objective"] = to_float_list(raw_history.get("train_objective", raw_history.get("train_loss")))
    normalized["train_loss"] = to_float_list(raw_history.get("train_loss"))
    normalized["train_accuracy"] = to_float_list(raw_history.get("train_accuracy"))
    normalized["test_loss"] = to_float_list(raw_history.get("test_loss"))
    normalized["test_accuracy"] = to_float_list(raw_history.get("test_accuracy", raw_history.get("eval_accuracy")))
    return normalized


def evaluate_classification_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += (predictions == labels).sum().item()
    return {
        "loss": running_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    metrics = evaluate_classification_metrics(model, data_loader, device, nn.CrossEntropyLoss())
    return metrics["accuracy"]


def ensure_finite_scalar(value: float, context: str) -> float:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"Non-finite metric detected: {context}={value}")
    return float(value)


def ensure_finite_loss_tensor(loss: torch.Tensor, context: str) -> None:
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Non-finite optimization loss detected during {context}.")


def build_optimizer(model: nn.Module, settings: TrainSettings) -> optim.Optimizer:
    if settings.optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=settings.lr,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
        )
    if settings.optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    raise ValueError(f"Unsupported optimizer: {settings.optimizer_name}")


def build_scheduler(optimizer: optim.Optimizer, settings: TrainSettings):
    if not settings.lr_milestones:
        return None
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(settings.lr_milestones),
        gamma=settings.lr_gamma,
    )


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    aux_loss_weight: float = 0.0,
    logger: RunLogger | None = None,
    phase: str = "target",
) -> dict[str, list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, settings)
    scheduler = build_scheduler(optimizer, settings)
    history = create_history_template()
    logger = logger or build_run_logger()

    for epoch in range(settings.epochs):
        model.train()
        running_objective = 0.0
        running_ce_loss = 0.0
        running_correct = 0
        total_examples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            ce_loss = criterion(logits, labels)
            loss = ce_loss
            if aux_loss_weight > 0:
                aux_loss = compute_gating_load_balance_loss(model)
                if aux_loss is not None:
                    loss = loss + aux_loss_weight * aux_loss
            ensure_finite_loss_tensor(loss, f"{phase} epoch {epoch + 1} supervised training")
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            running_objective += loss.item() * batch_size
            running_ce_loss += ce_loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size
        if scheduler is not None:
            scheduler.step()
        train_objective = ensure_finite_scalar(
            running_objective / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_objective",
        )
        train_loss = ensure_finite_scalar(
            running_ce_loss / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_loss",
        )
        train_accuracy = ensure_finite_scalar(
            100.0 * running_correct / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_accuracy",
        )
        test_metrics = evaluate_classification_metrics(model, test_loader, device, criterion)
        test_metrics["loss"] = ensure_finite_scalar(
            test_metrics["loss"],
            f"{phase} epoch {epoch + 1} test_loss",
        )
        test_metrics["accuracy"] = ensure_finite_scalar(
            test_metrics["accuracy"],
            f"{phase} epoch {epoch + 1} test_accuracy",
        )
        history["train_objective"].append(train_objective)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_metrics["loss"])
        history["test_accuracy"].append(test_metrics["accuracy"])
        metrics_payload = {
            "epoch": epoch + 1,
            "epochs": settings.epochs,
            "phase": phase,
            "train_objective": train_objective,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        logger.metrics(metrics_payload)
        logger.epoch(
            f"[{phase}] Epoch {epoch + 1:03d}/{settings.epochs:03d} | "
            f"train_objective={train_objective:.4f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.2f}% | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.2f}%"
        )
    return history


def train_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    aux_loss_weight: float = 0.0,
    logger: RunLogger | None = None,
    phase: str = "target",
) -> dict[str, list[float]]:
    hard_loss = nn.CrossEntropyLoss()
    optimizer = build_optimizer(student_model, settings)
    scheduler = build_scheduler(optimizer, settings)
    history = create_history_template()
    logger = logger or build_run_logger()

    teacher_model.eval()
    for epoch in range(settings.epochs):
        student_model.train()
        running_objective = 0.0
        running_ce_loss = 0.0
        running_correct = 0
        total_examples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            student_logits = student_model(inputs)
            ce_loss = hard_loss(student_logits, labels)
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / settings.kd_temperature, dim=1),
                F.softmax(teacher_logits / settings.kd_temperature, dim=1),
                reduction="batchmean",
            )
            loss = (
                settings.ce_loss_weight * ce_loss
                + settings.kd_loss_weight * (settings.kd_temperature**2) * kd_loss
            )
            if aux_loss_weight > 0:
                aux_loss = compute_gating_load_balance_loss(student_model)
                if aux_loss is not None:
                    loss = loss + aux_loss_weight * aux_loss
            ensure_finite_loss_tensor(loss, f"{phase} epoch {epoch + 1} distillation training")
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            running_objective += loss.item() * batch_size
            running_ce_loss += ce_loss.item() * batch_size
            running_correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size
        if scheduler is not None:
            scheduler.step()
        train_objective = ensure_finite_scalar(
            running_objective / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_objective",
        )
        train_loss = ensure_finite_scalar(
            running_ce_loss / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_loss",
        )
        train_accuracy = ensure_finite_scalar(
            100.0 * running_correct / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_accuracy",
        )
        test_metrics = evaluate_classification_metrics(student_model, test_loader, device, hard_loss)
        test_metrics["loss"] = ensure_finite_scalar(
            test_metrics["loss"],
            f"{phase} epoch {epoch + 1} test_loss",
        )
        test_metrics["accuracy"] = ensure_finite_scalar(
            test_metrics["accuracy"],
            f"{phase} epoch {epoch + 1} test_accuracy",
        )
        history["train_objective"].append(train_objective)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_metrics["loss"])
        history["test_accuracy"].append(test_metrics["accuracy"])
        metrics_payload = {
            "epoch": epoch + 1,
            "epochs": settings.epochs,
            "phase": phase,
            "train_objective": train_objective,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        logger.metrics(metrics_payload)
        logger.epoch(
            f"[{phase}] Epoch {epoch + 1:03d}/{settings.epochs:03d} | "
            f"train_objective={train_objective:.4f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.2f}% | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.2f}%"
        )
    return history


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def format_float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def sanitize_tag(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "_")


def get_pyplot(plot_mode: str):
    global _PYPLOT
    if plot_mode == "none":
        return None
    if _PYPLOT is not None:
        return _PYPLOT
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg", force=True)
        _PYPLOT = importlib.import_module("matplotlib.pyplot")
        apply_publication_plot_theme(_PYPLOT)
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it in the active environment or run with --plot-mode none."
        ) from exc
    return _PYPLOT


def apply_publication_plot_theme(plt) -> None:
    global _PLOT_THEME_APPLIED
    if _PLOT_THEME_APPLIED:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#424A57",
            "axes.linewidth": 1.1,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelcolor": "#2F3640",
            "xtick.color": "#4A5160",
            "ytick.color": "#4A5160",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": "#D8DDE6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "lines.linewidth": 2.6,
            "lines.solid_capstyle": "round",
        }
    )
    _PLOT_THEME_APPLIED = True


def history_has_curves(history: Mapping[str, Any] | None) -> bool:
    normalized = normalize_history(history)
    return any(normalized[key] for key, *_ in PLOT_METRIC_SPECS)


def get_plot_method_key(method: str, metadata: Mapping[str, Any]) -> str:
    config_tag = str(metadata.get("config_tag", "default"))
    if method == "inhernet":
        rank_preset = str(metadata.get("rank_preset", ""))
        if rank_preset == "small" or config_tag.startswith("small_rank_"):
            return "inhernet_small"
        if rank_preset == "large" or config_tag.startswith("large_rank_"):
            return "inhernet_large"
        return "inhernet_custom"
    return method


def get_plot_style(method_key: str) -> dict[str, Any]:
    styles: dict[str, dict[str, Any]] = {
        "teacher": {"color": "#111111", "linestyle": "-", "linewidth": 3.1, "alpha": 1.0, "zorder": 7},
        "student": {"color": "#8B8B8B", "linestyle": "--", "linewidth": 2.2, "alpha": 0.95, "zorder": 3},
        "student_kd": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "inhernet_small": {"color": "#009E73", "linestyle": "-", "linewidth": 2.8, "alpha": 0.98, "zorder": 4},
        "inhernet_large": {"color": "#D55E00", "linestyle": "-", "linewidth": 2.8, "alpha": 0.98, "zorder": 4},
        "inhernet_custom": {"color": "#CC79A7", "linestyle": "-.", "linewidth": 2.6, "alpha": 0.97, "zorder": 4},
        "hetero": {"color": "#E69F00", "linestyle": "-", "linewidth": 3.0, "alpha": 1.0, "zorder": 6},
    }
    return styles.get(method_key, {"color": "#4C78A8", "linestyle": "-", "linewidth": 2.5, "alpha": 0.98, "zorder": 4})


def style_plot_axis(ax, ylabel: str, title: str) -> None:
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, axis="y", alpha=0.75)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.tick_params(length=4.5, width=1.0)
    ax.margins(x=0.02)


def set_axis_limits(ax, values: list[float], clamp: tuple[float | None, float | None] | None = None) -> None:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return
    min_value = min(finite_values)
    max_value = max(finite_values)
    span = max(max_value - min_value, 1e-6)
    lower = min_value - 0.08 * span
    upper = max_value + 0.12 * span
    if clamp is not None:
        lower_bound, upper_bound = clamp
        if lower_bound is not None:
            lower = max(lower_bound, lower)
        if upper_bound is not None:
            upper = min(upper_bound, upper)
    ax.set_ylim(lower, upper)


def add_endpoint_marker(ax, x_value: int, y_value: float, color: str) -> None:
    if not math.isfinite(float(y_value)):
        return
    ax.scatter(
        [x_value],
        [y_value],
        s=34,
        color=color,
        edgecolor="white",
        linewidth=0.9,
        zorder=10,
    )


def build_metric_summary(history: Mapping[str, Any]) -> str:
    normalized = normalize_history(history)
    summary_lines = []
    train_loss = [value for value in normalized["train_loss"] if math.isfinite(value)]
    test_loss = [value for value in normalized["test_loss"] if math.isfinite(value)]
    train_accuracy = [value for value in normalized["train_accuracy"] if math.isfinite(value)]
    test_accuracy = [value for value in normalized["test_accuracy"] if math.isfinite(value)]
    train_objective = [value for value in normalized["train_objective"] if math.isfinite(value)]
    if train_loss:
        summary_lines.append(f"Train loss   {train_loss[-1]:.3f}")
    if test_loss:
        summary_lines.append(f"Test loss    {test_loss[-1]:.3f}")
    if train_accuracy:
        summary_lines.append(f"Train acc    {train_accuracy[-1]:.2f}%")
    if test_accuracy:
        summary_lines.append(f"Test acc     {test_accuracy[-1]:.2f}%")
        summary_lines.append(f"Best test    {max(test_accuracy):.2f}%")
    if train_objective and (not train_loss or abs(train_objective[-1] - train_loss[-1]) > 1e-8):
        summary_lines.append(f"Objective    {train_objective[-1]:.3f}")
    return "\n".join(summary_lines)


def draw_unavailable_metric(ax, ylabel: str, title: str) -> None:
    style_plot_axis(ax, ylabel, title)
    ax.text(
        0.5,
        0.5,
        "Metric unavailable\nin this run",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.0,
        color="#7A8596",
        bbox={
            "boxstyle": "round,pad=0.36",
            "facecolor": "#F7F8FB",
            "edgecolor": "#D7DCE5",
            "linewidth": 0.8,
        },
    )


def plot_single_metric_panel(
    ax,
    values: list[float],
    style: Mapping[str, Any],
    ylabel: str,
    title: str,
    clamp: tuple[float | None, float | None] | None = None,
) -> None:
    finite_pairs = [(idx + 1, float(value)) for idx, value in enumerate(values) if math.isfinite(float(value))]
    if not finite_pairs:
        draw_unavailable_metric(ax, ylabel, title)
        return
    x_values = [item[0] for item in finite_pairs]
    y_values = [item[1] for item in finite_pairs]
    ax.plot(
        x_values,
        y_values,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=max(2.5, float(style["linewidth"])),
        alpha=style["alpha"],
        zorder=style["zorder"],
    )
    add_endpoint_marker(ax, x_values[-1], y_values[-1], str(style["color"]))
    style_plot_axis(ax, ylabel, title)
    set_axis_limits(ax, y_values, clamp=clamp)


def plot_comparison_metric_panel(
    ax,
    records: list[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    clamp: tuple[float | None, float | None] | None = None,
) -> None:
    plotted_values: list[float] = []
    for record in records:
        values = list(record["history"].get(metric_key, []))
        finite_pairs = [(idx + 1, float(value)) for idx, value in enumerate(values) if math.isfinite(float(value))]
        if not finite_pairs:
            continue
        x_values = [item[0] for item in finite_pairs]
        y_values = [item[1] for item in finite_pairs]
        style = get_plot_style(record["method_key"])
        ax.plot(
            x_values,
            y_values,
            label=record["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
        add_endpoint_marker(ax, x_values[-1], y_values[-1], str(style["color"]))
        plotted_values.extend(y_values)

    if not plotted_values:
        draw_unavailable_metric(ax, ylabel, title)
        return
    style_plot_axis(ax, ylabel, title)
    set_axis_limits(ax, plotted_values, clamp=clamp)


def build_plot_label(
    method: str,
    metadata: Mapping[str, Any],
    *,
    detailed: bool,
) -> str:
    method_key = get_plot_method_key(method, metadata)
    if method_key == "teacher":
        label = "Teacher"
        if detailed:
            label = f"Teacher - {metadata.get('teacher_arch', 'teacher')}"
    elif method_key == "student":
        label = "Student"
        if detailed:
            label = f"Student - {metadata.get('student_arch', 'student')}"
    elif method_key == "student_kd":
        label = "Student + KD"
        if detailed:
            teacher_arch = metadata.get("teacher_arch", "teacher")
            student_arch = metadata.get("student_arch", "student")
            label = f"Student + KD - {teacher_arch} -> {student_arch}"
    elif method_key == "inhernet_small":
        label = "InherNet-S"
        if detailed:
            label = f"InherNet-S - rank {metadata.get('rank', '?')}, heads {metadata.get('head_num', '?')}"
    elif method_key == "inhernet_large":
        label = "InherNet-L"
        if detailed:
            label = f"InherNet-L - rank {metadata.get('rank', '?')}, heads {metadata.get('head_num', '?')}"
    elif method_key == "inhernet_custom":
        label = "InherNet"
        if detailed:
            label = f"InherNet - rank {metadata.get('rank', '?')}, heads {metadata.get('head_num', '?')}"
    elif method_key == "hetero":
        rank_map = metadata.get("rank_map", {})
        avg_rank = None
        if isinstance(rank_map, Mapping) and rank_map:
            avg_rank = sum(int(value) for value in rank_map.values()) / len(rank_map)
        label = "Hetero"
        if detailed:
            budget_ratio = metadata.get("budget_ratio", "?")
            label = f"Hetero - heads {metadata.get('head_num', '?')}, budget {budget_ratio}"
            if avg_rank is None and "avg_rank" in metadata:
                avg_rank = float(metadata["avg_rank"])
            if avg_rank is not None:
                label += f", avg rank {avg_rank:.1f}"
    else:
        label = f"{method} ({metadata.get('config_tag', 'run')})"
    return label


def plot_single_history(
    plot_root: Path,
    metadata: Mapping[str, Any],
    history: Mapping[str, Any],
    plot_mode: str,
) -> Path | None:
    history = normalize_history(history)
    if not history_has_curves(history):
        return None
    dataset_name = str(metadata.get("dataset", "unknown_dataset"))
    pair_name = str(metadata.get("pair", "unknown_pair"))
    method = str(metadata.get("method", "unknown_method"))
    config_tag = sanitize_tag(str(metadata.get("config_tag", "default")))
    method_key = get_plot_method_key(method, metadata)
    style = get_plot_style(method_key)
    label = build_plot_label(method, metadata, detailed=True)
    plt = get_pyplot(plot_mode)

    fig, axes = plt.subplots(2, 2, figsize=(11.9, 8.2), dpi=300)
    for axis, (metric_key, ylabel, title, clamp) in zip(axes.flatten(), PLOT_METRIC_SPECS, strict=True):
        plot_single_metric_panel(axis, list(history.get(metric_key, [])), style, ylabel, title, clamp)

    summary_text = build_metric_summary(history)
    if summary_text:
        axes[1, 1].text(
            0.98,
            0.04,
            summary_text,
            transform=axes[1, 1].transAxes,
            ha="right",
            va="bottom",
            fontsize=9.1,
            family="DejaVu Sans Mono",
            color="#1F2530",
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": "#F7F8FB",
                "edgecolor": "#D7DCE5",
                "linewidth": 0.9,
            },
        )

    fig.suptitle(label, x=0.06, y=0.985, ha="left", fontsize=13.3, fontweight="bold")
    fig.text(
        0.06,
        0.945,
        f"{dataset_name} | {pair_name}",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5A6473",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.11, wspace=0.22, hspace=0.28)

    output_path = plot_root / dataset_name / pair_name / method / f"{config_tag}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def parse_structured_log_line(line: str, prefix: str) -> dict[str, Any] | None:
    token = f"{prefix} "
    if not line.startswith(token):
        return None
    payload = line[len(token) :].strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def parse_run_log(log_path: Path, *, phase: str = "target") -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    metadata: dict[str, Any] | None = None
    phase_histories: dict[str, dict[str, list[float]]] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            metadata_payload = parse_structured_log_line(line, RUN_METADATA_PREFIX)
            if metadata_payload is not None:
                metadata = metadata_payload
                continue
            metrics_payload = parse_structured_log_line(line, RUN_METRICS_PREFIX)
            if metrics_payload is None:
                continue
            phase_name = str(metrics_payload.get("phase", "target"))
            history = phase_histories.setdefault(phase_name, create_history_template())
            history["train_objective"].append(float(metrics_payload.get("train_objective", metrics_payload.get("train_loss", 0.0))))
            history["train_loss"].append(float(metrics_payload.get("train_loss", metrics_payload.get("train_objective", 0.0))))
            history["train_accuracy"].append(float(metrics_payload.get("train_accuracy", 0.0)))
            history["test_loss"].append(float(metrics_payload.get("test_loss", 0.0)))
            history["test_accuracy"].append(float(metrics_payload.get("test_accuracy", metrics_payload.get("eval_accuracy", 0.0))))

    if metadata is None:
        return None

    selected_history = phase_histories.get(phase, create_history_template())
    if not history_has_curves(selected_history):
        for candidate in phase_histories.values():
            if history_has_curves(candidate):
                selected_history = candidate
                break

    return {
        "log_path": log_path,
        "metadata": metadata,
        "history": normalize_history(selected_history),
    }


def collect_suite_comparison_records(suite_log_dir: Path) -> list[dict[str, Any]]:
    if not suite_log_dir.exists():
        return []

    method_order = {
        "teacher": 0,
        "student": 1,
        "student_kd": 2,
        "inhernet_small": 3,
        "inhernet_large": 4,
        "inhernet_custom": 5,
        "hetero": 6,
    }
    records: list[dict[str, Any]] = []
    for log_path in sorted(suite_log_dir.glob("[0-9][0-9]_*.log")):
        parsed = parse_run_log(log_path, phase="target")
        if parsed is None or not history_has_curves(parsed["history"]):
            continue
        metadata = parsed["metadata"]
        method = str(metadata.get("method", log_path.stem))
        method_key = get_plot_method_key(method, metadata)
        records.append(
            {
                "log_path": log_path,
                "history": parsed["history"],
                "metadata": metadata,
                "method": method,
                "method_key": method_key,
                "label": build_plot_label(method, metadata, detailed=False),
            }
        )

    label_counts = Counter(record["label"] for record in records)
    for record in records:
        if label_counts[record["label"]] > 1:
            config_tag = str(record["metadata"].get("config_tag", record["log_path"].stem))
            record["label"] = f"{record['label']} [{sanitize_tag(config_tag)}]"

    records.sort(
        key=lambda record: (
            method_order.get(record["method_key"], 99),
            record["label"],
            str(record["log_path"]),
        )
    )
    return records


def plot_comparison_histories_from_records(
    plot_root: Path,
    dataset_name: str,
    pair_name: str,
    records: list[dict[str, Any]],
    plot_mode: str,
) -> Path | None:
    if not records:
        return None
    plt = get_pyplot(plot_mode)

    fig, axes = plt.subplots(2, 2, figsize=(13.7, 8.5), dpi=300)
    for axis, (metric_key, ylabel, title, clamp) in zip(axes.flatten(), PLOT_METRIC_SPECS, strict=True):
        plot_comparison_metric_panel(axis, records, metric_key, ylabel, title, clamp)

    handles = []
    labels = []
    for axis in axes.flatten():
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.79, 0.50),
            ncol=1,
            frameon=False,
            handlelength=2.8,
            borderaxespad=0.0,
        )
    fig.suptitle("Model Comparison", x=0.06, y=0.985, ha="left", fontsize=13.4, fontweight="bold")
    fig.text(
        0.06,
        0.945,
        f"{dataset_name} | {pair_name}",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5A6473",
    )
    fig.subplots_adjust(left=0.08, right=0.77, top=0.84, bottom=0.11, wspace=0.24, hspace=0.30)

    output_path = plot_root / dataset_name / pair_name / "comparison" / "overview.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def plot_suite_comparison_from_logs(
    plot_root: Path,
    suite_log_dir: Path,
    dataset_name: str,
    pair_name: str,
    plot_mode: str,
) -> Path | None:
    records = collect_suite_comparison_records(suite_log_dir)
    return plot_comparison_histories_from_records(plot_root, dataset_name, pair_name, records, plot_mode)


def resolve_train_settings(dataset_spec: DatasetSpec, args: argparse.Namespace) -> TrainSettings:
    settings = dataset_spec.train_settings
    if args.optimizer is not None:
        settings = replace(settings, optimizer_name=args.optimizer)
    if args.batch_size is not None:
        settings = replace(settings, batch_size=args.batch_size)
    if args.epochs is not None:
        settings = replace(settings, epochs=args.epochs)
    if args.lr is not None:
        settings = replace(settings, lr=args.lr)
    if args.momentum is not None:
        settings = replace(settings, momentum=args.momentum)
    if args.weight_decay is not None:
        settings = replace(settings, weight_decay=args.weight_decay)
    if args.kd_temperature is not None:
        settings = replace(settings, kd_temperature=args.kd_temperature)
    if args.kd_weight is not None:
        settings = replace(settings, kd_loss_weight=args.kd_weight)
    if args.ce_weight is not None:
        settings = replace(settings, ce_loss_weight=args.ce_weight)
    return settings


def resolve_head_num(args: argparse.Namespace, pair_spec: Mapping[str, Any], settings: TrainSettings) -> int:
    if args.head_num is not None:
        return args.head_num
    if "default_head_num" in pair_spec:
        return int(pair_spec["default_head_num"])
    return settings.default_head_num


def resolve_fixed_rank(args: argparse.Namespace, pair_spec: Mapping[str, Any]) -> int:
    return resolve_fixed_rank_with_override(args, pair_spec)


def resolve_fixed_rank_with_override(
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any],
    rank_preset_override: str | None = None,
) -> int:
    if args.rank is not None and rank_preset_override is None:
        return args.rank
    rank_presets = pair_spec.get("rank_presets", {})
    preset_name = args.rank_preset if rank_preset_override is None else rank_preset_override
    if preset_name not in rank_presets:
        available = ", ".join(sorted(rank_presets.keys()))
        raise ValueError(f"No rank preset '{preset_name}'. Available: {available}")
    return int(rank_presets[preset_name])


def validate_args(args: argparse.Namespace, pair_spec: Mapping[str, Any]) -> None:
    requested_methods = [args.method] if args.method is not None else [entry["method"] for entry in SUITE_SPECS[args.suite]]
    if "hetero" in requested_methods and args.compress_threshold <= args.min_rank:
        raise ValueError("--compress-threshold must be greater than --min-rank for hetero gating to use both branches.")
    if args.suite is not None and args.rank is not None:
        raise ValueError("--rank cannot be used together with --suite because suite InherNet runs already define their own presets.")
    if "inhernet" in requested_methods:
        if args.suite is None:
            rank = resolve_fixed_rank_with_override(args, pair_spec)
            if rank <= 0:
                raise ValueError("Resolved InherNet rank must be positive.")
        else:
            for entry in SUITE_SPECS[args.suite]:
                if entry["method"] != "inhernet":
                    continue
                rank = resolve_fixed_rank_with_override(args, pair_spec, entry.get("rank_preset"))
                if rank <= 0:
                    raise ValueError("Resolved InherNet rank must be positive.")


def build_method_tag(
    method: str,
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any],
    settings: TrainSettings,
    rank_preset_override: str | None = None,
) -> str:
    head_num = resolve_head_num(args, pair_spec, settings)
    if method in {"teacher", "student", "student_kd"}:
        tag = "default"
    elif method == "inhernet":
        rank = resolve_fixed_rank_with_override(args, pair_spec, rank_preset_override)
        if rank_preset_override is not None:
            rank_source = rank_preset_override
        else:
            rank_source = "custom" if args.rank is not None else args.rank_preset
        tag = f"{rank_source}_rank_{rank}_heads_{head_num}"
    elif method == "hetero":
        tag = (
            f"heads_{head_num}_budget_{format_float_tag(args.budget_ratio)}_"
            f"min_{args.min_rank}_temp_{format_float_tag(args.hetero_temperature)}_"
            f"thr_{args.compress_threshold}_calib_{args.max_calib_batches}"
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return tag


def build_run_metadata(
    method: str,
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any],
    settings: TrainSettings,
    model: nn.Module,
    config_tag: str,
    *,
    suite_name: str | None = None,
    suite_label: str | None = None,
    rank_preset_override: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset": args.dataset,
        "pair": args.pair,
        "method": method,
        "config_tag": config_tag,
        "plot_tag": config_tag,
        "teacher_arch": get_role_name(pair_spec, "teacher"),
        "student_arch": get_role_name(pair_spec, "student"),
        "num_parameters": count_parameters(model),
        "train_settings": asdict(settings),
    }
    if suite_name is not None:
        metadata["suite_name"] = suite_name
    if suite_label is not None:
        metadata["suite_label"] = suite_label
    if method == "inhernet":
        metadata["rank_preset"] = (
            rank_preset_override
            if rank_preset_override is not None
            else ("custom" if args.rank is not None else args.rank_preset)
        )
    if extra is not None:
        metadata.update(extra)
    return metadata


def build_training_dataloaders(
    args: argparse.Namespace,
    settings: TrainSettings,
) -> tuple[DataLoader, DataLoader]:
    return get_dataloaders(
        args.dataset,
        batch_size=settings.batch_size,
        root=args.data_root,
        download=args.download,
        num_workers=args.num_workers,
    )


def get_suite_run_specs(suite_name: str) -> list[dict[str, Any]]:
    if suite_name not in SUITE_SPECS:
        available = ", ".join(sorted(SUITE_SPECS.keys()))
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {available}")
    return [dict(entry) for entry in SUITE_SPECS[suite_name]]


def resolve_suite_log_dir(args: argparse.Namespace) -> Path:
    env_value = os.environ.get(SUITE_LOG_DIR_ENV_VAR)
    if env_value:
        return Path(env_value)
    if args.suite is None:
        raise ValueError("Suite log directory can only be resolved for suite runs.")
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return PROJECT_DIR / "logs" / args.dataset / args.pair / args.suite / timestamp


def maybe_save_single_plot(
    plot_root: Path,
    metadata: Mapping[str, Any],
    history: Mapping[str, Any],
    plot_mode: str,
    logger: RunLogger | None = None,
) -> Path | None:
    if plot_mode not in {"single", "both"}:
        return None
    output_path = plot_single_history(plot_root, metadata, history, plot_mode)
    if output_path is not None and logger is not None:
        logger.info(f"Saved plot: {output_path}")
    return output_path


def maybe_save_suite_comparison_plot(
    plot_root: Path,
    suite_log_dir: Path,
    dataset_name: str,
    pair_name: str,
    plot_mode: str,
    logger: RunLogger | None = None,
) -> Path | None:
    if plot_mode not in {"compare", "both"}:
        return None
    output_path = plot_suite_comparison_from_logs(plot_root, suite_log_dir, dataset_name, pair_name, plot_mode)
    if output_path is not None and logger is not None:
        logger.info(f"Saved comparison plot: {output_path}")
    return output_path


def train_teacher_pretrain(
    args: argparse.Namespace,
    dataset_spec: DatasetSpec,
    settings: TrainSettings,
    device: torch.device,
    logger: RunLogger,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(args.seed)
    train_loader, test_loader = build_training_dataloaders(args, settings)
    teacher_model = build_pair_model(args.dataset, args.pair, "teacher", dataset_spec.num_classes).to(device)
    logger.info("Training teacher from scratch for the target method.")
    history = train_supervised(
        teacher_model,
        train_loader,
        test_loader,
        settings,
        device,
        logger=logger,
        phase="teacher_pretrain",
    )
    teacher_model.eval()
    return teacher_model, history


def train_method_from_scratch(
    args: argparse.Namespace,
    method: str,
    pair_spec: Mapping[str, Any],
    dataset_spec: DatasetSpec,
    settings: TrainSettings,
    device: torch.device,
    logger: RunLogger,
    *,
    teacher_model: nn.Module | None = None,
    suite_name: str | None = None,
    suite_label: str | None = None,
    rank_preset_override: str | None = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    set_seed(args.seed)
    train_loader, test_loader = build_training_dataloaders(args, settings)
    config_tag = build_method_tag(method, args, pair_spec, settings, rank_preset_override)
    head_num = resolve_head_num(args, pair_spec, settings)

    if method == "teacher":
        model = build_pair_model(args.dataset, args.pair, "teacher", dataset_spec.num_classes).to(device)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
        )
        logger.metadata(metadata)
        history = train_supervised(model, train_loader, test_loader, settings, device, logger=logger)
    elif method == "student":
        model = build_pair_model(args.dataset, args.pair, "student", dataset_spec.num_classes).to(device)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
        )
        logger.metadata(metadata)
        history = train_supervised(model, train_loader, test_loader, settings, device, logger=logger)
    elif method == "student_kd":
        if teacher_model is None:
            raise ValueError("student_kd requires an in-memory teacher model.")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        model = build_pair_model(args.dataset, args.pair, "student", dataset_spec.num_classes).to(device)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
        )
        logger.metadata(metadata)
        history = train_distillation(teacher_model, model, train_loader, test_loader, settings, device, logger=logger)
    elif method == "inhernet":
        if teacher_model is None:
            raise ValueError("inhernet requires an in-memory teacher model.")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        rank = resolve_fixed_rank_with_override(args, pair_spec, rank_preset_override)
        model = GenericInherNet(
            build_pair_model(args.dataset, args.pair, "teacher", dataset_spec.num_classes)
        ).to(device)
        model.load_dense_state_dict(teacher_model.state_dict())
        model.apply_svd(rank=rank, head_num=head_num)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
            rank_preset_override=rank_preset_override,
            extra={
                "rank": rank,
                "head_num": head_num,
            },
        )
        logger.metadata(metadata)
        history = train_distillation(teacher_model, model, train_loader, test_loader, settings, device, logger=logger)
    elif method == "hetero":
        if teacher_model is None:
            raise ValueError("hetero requires an in-memory teacher model.")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        model = GenericHeteroNet(
            build_pair_model(args.dataset, args.pair, "teacher", dataset_spec.num_classes)
        ).to(device)
        model.load_dense_state_dict(teacher_model.state_dict())
        rank_map = model.apply_hetero_svd(
            calib_loader=train_loader,
            head_num=head_num,
            budget_ratio=args.budget_ratio,
            min_rank=args.min_rank,
            compress_threshold=args.compress_threshold,
            temperature=args.hetero_temperature,
            max_calib_batches=args.max_calib_batches,
        )
        rank_values = list(rank_map.values())
        avg_rank = sum(rank_values) / len(rank_values)
        logger.info(
            "Hetero rank allocation: "
            f"min={min(rank_values)}, max={max(rank_values)}, avg={avg_rank:.2f}"
        )
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
            extra={
                "head_num": head_num,
                "budget_ratio": args.budget_ratio,
                "min_rank": args.min_rank,
                "compress_threshold": args.compress_threshold,
                "hetero_temperature": args.hetero_temperature,
                "max_calib_batches": args.max_calib_batches,
                "aux_loss_weight": args.aux_loss_weight,
                "rank_map": {name: int(rank) for name, rank in rank_map.items()},
                "avg_rank": avg_rank,
                "rank_min": min(rank_values),
                "rank_max": max(rank_values),
            },
        )
        logger.metadata(metadata)
        history = train_distillation(
            teacher_model,
            model,
            train_loader,
            test_loader,
            settings,
            device,
            aux_loss_weight=args.aux_loss_weight,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    model.eval()
    return model, history, metadata


def run_single_method_smoke_test(
    dataset_name: str,
    pair_name: str,
    method: str,
    args: argparse.Namespace,
    rank_preset_override: str | None = None,
) -> dict[str, Any]:
    dataset_spec = DATASET_REGISTRY[dataset_name]
    pair_spec = get_pair_spec(dataset_name, pair_name)
    settings = resolve_train_settings(dataset_spec, args)
    head_num = resolve_head_num(args, pair_spec, settings)
    sample = torch.randn(2, 3, 32, 32)
    calib_inputs = torch.randn(8, 3, 32, 32)
    calib_labels = torch.zeros(8, dtype=torch.long)
    calib_loader = DataLoader(TensorDataset(calib_inputs, calib_labels), batch_size=2, shuffle=False)

    if method == "teacher":
        model = build_pair_model(dataset_name, pair_name, "teacher", dataset_spec.num_classes)
        output = model(sample)
        return {"method": method, "shape": tuple(output.shape), "params": count_parameters(model)}
    if method == "student":
        model = build_pair_model(dataset_name, pair_name, "student", dataset_spec.num_classes)
        output = model(sample)
        return {"method": method, "shape": tuple(output.shape), "params": count_parameters(model)}
    if method == "student_kd":
        teacher = build_pair_model(dataset_name, pair_name, "teacher", dataset_spec.num_classes)
        student = build_pair_model(dataset_name, pair_name, "student", dataset_spec.num_classes)
        teacher_out = teacher(sample)
        student_out = student(sample)
        return {
            "method": method,
            "teacher_shape": tuple(teacher_out.shape),
            "student_shape": tuple(student_out.shape),
        }
    if method == "inhernet":
        teacher = build_pair_model(dataset_name, pair_name, "teacher", dataset_spec.num_classes)
        model = GenericInherNet(copy.deepcopy(teacher))
        model.load_dense_state_dict(teacher.state_dict())
        rank = resolve_fixed_rank_with_override(args, pair_spec, rank_preset_override)
        model.apply_svd(rank=rank, head_num=head_num)
        output = model(sample)
        return {
            "method": method,
            "shape": tuple(output.shape),
            "params": count_parameters(model),
            "rank": rank,
            "head_num": head_num,
        }
    if method == "hetero":
        teacher = build_pair_model(dataset_name, pair_name, "teacher", dataset_spec.num_classes)
        model = GenericHeteroNet(copy.deepcopy(teacher))
        model.load_dense_state_dict(teacher.state_dict())
        rank_map = model.apply_hetero_svd(
            calib_loader=calib_loader,
            head_num=head_num,
            budget_ratio=args.budget_ratio,
            min_rank=args.min_rank,
            compress_threshold=args.compress_threshold,
            temperature=args.hetero_temperature,
            max_calib_batches=min(args.max_calib_batches, len(calib_loader)),
        )
        output = model(sample)
        assert args.compress_threshold > args.min_rank
        return {
            "method": method,
            "shape": tuple(output.shape),
            "params": count_parameters(model),
            "head_num": head_num,
            "rank_min": min(rank_map.values()),
            "rank_max": max(rank_map.values()),
        }
    raise ValueError(f"Unknown method: {method}")


def run_single_method(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    dataset_spec = DATASET_REGISTRY[args.dataset]
    pair_spec = get_pair_spec(args.dataset, args.pair)
    settings = resolve_train_settings(dataset_spec, args)
    validate_args(args, pair_spec)
    logger = build_run_logger(echo=True, store_info_to_file=False)

    if args.smoke_test:
        result = run_single_method_smoke_test(args.dataset, args.pair, args.method, args)
        logger.info(f"Smoke test passed: {result}")
        return Path("<smoke-test>")

    get_pyplot(args.plot_mode)
    plot_root = Path(args.plot_root)

    if args.method in {"teacher", "student"}:
        logger.info(f"Training {args.method} from scratch.")
        _, history, metadata = train_method_from_scratch(
            args,
            args.method,
            pair_spec,
            dataset_spec,
            settings,
            device,
            logger,
        )
    else:
        teacher_model, _ = train_teacher_pretrain(args, dataset_spec, settings, device, logger)
        logger.info(f"Training {args.method} from scratch using the in-memory teacher.")
        _, history, metadata = train_method_from_scratch(
            args,
            args.method,
            pair_spec,
            dataset_spec,
            settings,
            device,
            logger,
            teacher_model=teacher_model,
        )

    plot_path = maybe_save_single_plot(plot_root, metadata, history, args.plot_mode, logger)
    return plot_path if plot_path is not None else Path("<no-plot>")


def run_suite_smoke_test(args: argparse.Namespace) -> Path:
    suite_log_dir = resolve_suite_log_dir(args)
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_logger = build_run_logger(str(suite_log_dir / "suite.log"), echo=True, store_info_to_file=True)
    suite_logger.info(
        f"Suite smoke test started: dataset={args.dataset}, pair={args.pair}, suite={args.suite}"
    )
    for spec in get_suite_run_specs(args.suite):
        label = str(spec["label"])
        child_logger = build_run_logger(str(suite_log_dir / f"{label}.log"), echo=False, store_info_to_file=True)
        result = run_single_method_smoke_test(
            args.dataset,
            args.pair,
            str(spec["method"]),
            args,
            rank_preset_override=spec.get("rank_preset"),
        )
        child_logger.info(f"Smoke test passed: {result}")
        suite_logger.info(f"[{label}] Smoke test passed.")
    suite_logger.info("Suite smoke test completed successfully.")
    return suite_log_dir


def run_suite(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    dataset_spec = DATASET_REGISTRY[args.dataset]
    pair_spec = get_pair_spec(args.dataset, args.pair)
    settings = resolve_train_settings(dataset_spec, args)
    validate_args(args, pair_spec)

    if args.smoke_test:
        return run_suite_smoke_test(args)

    get_pyplot(args.plot_mode)
    plot_root = Path(args.plot_root)
    suite_log_dir = resolve_suite_log_dir(args)
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_logger = build_run_logger(str(suite_log_dir / "suite.log"), echo=True, store_info_to_file=True)
    suite_logger.info(
        f"Suite started: dataset={args.dataset}, pair={args.pair}, suite={args.suite}. All runs train from scratch."
    )

    teacher_model: nn.Module | None = None
    for spec in get_suite_run_specs(args.suite):
        label = str(spec["label"])
        method = str(spec["method"])
        rank_preset_override = spec.get("rank_preset")
        child_log_path = suite_log_dir / f"{label}.log"
        child_logger = build_run_logger(str(child_log_path), echo=False, store_info_to_file=True)

        suite_logger.info(f"[{label}] Starting {method}.")
        child_logger.info(
            f"Run started: dataset={args.dataset}, pair={args.pair}, method={method}, suite={args.suite}, label={label}"
        )
        try:
            if method == "teacher":
                teacher_model, history, metadata = train_method_from_scratch(
                    args,
                    method,
                    pair_spec,
                    dataset_spec,
                    settings,
                    device,
                    child_logger,
                    suite_name=args.suite,
                    suite_label=label,
                    rank_preset_override=rank_preset_override,
                )
            elif method == "student":
                model, history, metadata = train_method_from_scratch(
                    args,
                    method,
                    pair_spec,
                    dataset_spec,
                    settings,
                    device,
                    child_logger,
                    suite_name=args.suite,
                    suite_label=label,
                    rank_preset_override=rank_preset_override,
                )
                del model
            else:
                if teacher_model is None:
                    raise RuntimeError("Suite execution requires the teacher step to complete before dependent methods.")
                model, history, metadata = train_method_from_scratch(
                    args,
                    method,
                    pair_spec,
                    dataset_spec,
                    settings,
                    device,
                    child_logger,
                    teacher_model=teacher_model,
                    suite_name=args.suite,
                    suite_label=label,
                    rank_preset_override=rank_preset_override,
                )
                del model

            plot_path = maybe_save_single_plot(plot_root, metadata, history, args.plot_mode, child_logger)
            if plot_path is not None:
                suite_logger.info(f"[{label}] Saved plot: {plot_path}")
            comparison_path = maybe_save_suite_comparison_plot(
                plot_root,
                suite_log_dir,
                args.dataset,
                args.pair,
                args.plot_mode,
                suite_logger,
            )
            if comparison_path is not None:
                suite_logger.info(f"[{label}] Refreshed comparison plot.")
            suite_logger.info(f"[{label}] Completed.")
        except Exception as exc:
            suite_logger.info(f"[{label}] Failed: {exc}")
            raise
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    suite_logger.info("Suite completed successfully.")
    return suite_log_dir


def run_training(args: argparse.Namespace) -> Path:
    if args.suite is not None:
        return run_suite(args)
    return run_single_method(args)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registry-driven InherNet runner for CIFAR-10 and CIFAR-100.")
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY.keys()), required=True)
    parser.add_argument("--pair", required=True, help="Dataset-specific teacher/student pair name.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--method", choices=METHOD_CHOICES)
    mode_group.add_argument("--suite", choices=sorted(SUITE_SPECS.keys()))
    parser.add_argument("--data-root", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--kd-temperature", type=float, default=None)
    parser.add_argument("--kd-weight", type=float, default=None)
    parser.add_argument("--ce-weight", type=float, default=None)
    parser.add_argument("--rank-preset", choices=["small", "large"], default="small")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--head-num", type=int, default=None)
    parser.add_argument("--budget-ratio", type=float, default=0.35)
    parser.add_argument("--min-rank", type=int, default=8)
    parser.add_argument("--compress-threshold", type=int, default=12)
    parser.add_argument("--hetero-temperature", type=float, default=1.4)
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument("--aux-loss-weight", type=float, default=0.01)
    parser.add_argument("--plot-mode", choices=["none", "single", "compare", "both"], default="both")
    parser.add_argument("--plot-root", default=str(PROJECT_DIR / "results"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
