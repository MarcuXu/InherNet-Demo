from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from cifar10_models import PAIR_REGISTRY as CIFAR10_PAIR_REGISTRY
from cifar100_models import PAIR_REGISTRY as CIFAR100_PAIR_REGISTRY
from cifar100_models import build_model as build_cifar100_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUITE_LOG_DIR_ENV_VAR = "INHERNET_SUITE_LOG_DIR"
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


def build_pair_model(dataset_name: str, pair_name: str, role: str, num_classes: int):
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
    *,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_set = get_dataset(dataset_name, root=root, train=True, download=download)
    test_set = get_dataset(dataset_name, root=root, train=False, download=download)
    generator = torch.Generator().manual_seed(seed)
    if pin_memory is None:
        pin_memory = DEFAULT_DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def format_float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def sanitize_tag(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "_")


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


def resolve_fixed_rank(args: argparse.Namespace, pair_spec: Mapping[str, Any]) -> int:
    return resolve_fixed_rank_with_override(args, pair_spec)


def validate_args(args: argparse.Namespace, pair_spec: Mapping[str, Any]) -> None:
    requested_methods = [args.method] if args.method is not None else [entry["method"] for entry in SUITE_SPECS[args.suite]]
    if "hetero" in requested_methods and args.compress_threshold <= args.min_rank:
        raise ValueError("--compress-threshold must be greater than --min-rank for hetero gating to use both branches.")
    if "hetero" in requested_methods and args.hetero_expert_noise_scale < 0:
        raise ValueError("--hetero-expert-noise-scale must be non-negative.")
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
            f"thr_{args.compress_threshold}_calib_{args.max_calib_batches}_"
            f"noise_{format_float_tag(args.hetero_expert_noise_scale)}"
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return tag


def build_training_dataloaders(
    args: argparse.Namespace,
    settings: TrainSettings,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    runtime_device = resolve_device(args.device) if device is None else device
    return get_dataloaders(
        args.dataset,
        batch_size=settings.batch_size,
        root=args.data_root,
        download=args.download,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=runtime_device.type == "cuda",
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
