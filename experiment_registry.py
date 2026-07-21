from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from cifar10_models import PAIR_REGISTRY as CIFAR10_PAIR_REGISTRY
from cifar10_models import build_model as build_cifar10_model
from cifar100_models import PAIR_REGISTRY as CIFAR100_PAIR_REGISTRY
from cifar100_models import build_model as build_cifar100_model
from glue_data import build_glue_dataloaders
from glue_models import PAIR_REGISTRY as GLUE_PAIR_REGISTRY
from glue_models import build_model as build_glue_model
from pet_models import PAIR_REGISTRY as OXFORD_PETS_PAIR_REGISTRY
from pet_models import build_model as build_pet_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHOD_CHOICES = ["teacher", "student", "student_kd", "inhernet", "hetero"]
TRAIN_SETTING_OVERRIDE_KEYS = {
    "optimizer_name",
    "batch_size",
    "epochs",
    "lr",
    "momentum",
    "weight_decay",
    "lr_milestones",
    "lr_gamma",
    "kd_temperature",
    "kd_loss_weight",
    "ce_loss_weight",
    "default_head_num",
    "scheduler_name",
    "warmup_ratio",
    "max_grad_norm",
    "exclude_bias_norm_from_weight_decay",
}


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
    scheduler_name: str = "multistep"
    warmup_ratio: float = 0.0
    max_grad_norm: float = 0.0
    exclude_bias_norm_from_weight_decay: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    dataset_class: type | None
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None
    train_settings: TrainSettings
    pair_registry: Mapping[str, Mapping[str, Any]]
    task_type: str = "vision"
    image_size: int = 32
    train_split: str | None = None
    test_split: str | None = None
    eval_split_name: str = "test"
    primary_metric_name: str = "accuracy"
    primary_metric_display: str = "Accuracy (%)"
    metric_names: tuple[str, ...] = ("accuracy",)
    problem_type: str = "classification"
    text_task_name: str | None = None
    text_max_length: int = 128
    final_test_split_name: str | None = None
    validation_fraction: float = 0.0
    validation_split_seed: int = 2026
    data_profile: str = "standard_train_test"


@dataclass(frozen=True)
class TrainingLoaders:
    train: DataLoader
    evaluation: DataLoader
    final_test: DataLoader | None = None
    calibration: DataLoader | None = None
    split_metadata: Mapping[str, Any] | None = None
    eval_split_name: str = "test"
    final_test_split_name: str | None = None
    restore_best_state: bool = False


GLUE_TRAIN_SETTINGS = TrainSettings(
    optimizer_name="adamw",
    batch_size=32,
    epochs=4,
    lr=5e-5,
    momentum=0.0,
    weight_decay=0.01,
    lr_milestones=(),
    kd_temperature=2.0,
    kd_loss_weight=1.0,
    ce_loss_weight=1.0,
    default_head_num=2,
    scheduler_name="linear",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    exclude_bias_norm_from_weight_decay=True,
)


def build_glue_dataset_spec(
    *,
    task_name: str,
    num_classes: int,
    eval_split_name: str = "validation",
    primary_metric_name: str = "accuracy",
    primary_metric_display: str = "GLUE Accuracy (%)",
    metric_names: tuple[str, ...] = ("accuracy",),
    problem_type: str = "classification",
) -> DatasetSpec:
    return DatasetSpec(
        num_classes=num_classes,
        dataset_class=None,
        mean=None,
        std=None,
        train_settings=GLUE_TRAIN_SETTINGS,
        pair_registry=GLUE_PAIR_REGISTRY,
        task_type="text",
        eval_split_name=eval_split_name,
        primary_metric_name=primary_metric_name,
        primary_metric_display=primary_metric_display,
        metric_names=metric_names,
        problem_type=problem_type,
        text_task_name=task_name,
        text_max_length=128,
    )


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
        primary_metric_display="Top-1 Accuracy (%)",
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
        primary_metric_display="Top-1 Accuracy (%)",
    ),
    "oxford_pets": DatasetSpec(
        num_classes=37,
        dataset_class=torchvision.datasets.OxfordIIITPet,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        train_settings=TrainSettings(
            optimizer_name="sgd",
            batch_size=32,
            epochs=30,
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4,
            lr_milestones=(15, 25),
            kd_temperature=2.0,
            kd_loss_weight=1.0,
            ce_loss_weight=1.0,
            default_head_num=3,
        ),
        pair_registry=OXFORD_PETS_PAIR_REGISTRY,
        image_size=224,
        train_split="trainval",
        test_split="test",
        eval_split_name="validation",
        final_test_split_name="test",
        validation_fraction=0.2,
        data_profile="oxford_pets_deterministic_stratified_transfer",
        primary_metric_name="balanced_accuracy",
        primary_metric_display="Mean Per-Class Accuracy (%)",
        metric_names=("accuracy", "macro_f1", "balanced_accuracy"),
    ),
    "glue_mrpc": build_glue_dataset_spec(
        task_name="mrpc",
        num_classes=2,
        metric_names=("accuracy", "f1"),
    ),
    "glue_qqp": build_glue_dataset_spec(
        task_name="qqp",
        num_classes=2,
        metric_names=("accuracy", "f1"),
    ),
    "glue_sst2": build_glue_dataset_spec(
        task_name="sst2",
        num_classes=2,
    ),
    "glue_mnli": build_glue_dataset_spec(
        task_name="mnli",
        num_classes=3,
        eval_split_name="validation_matched",
    ),
    "glue_rte": build_glue_dataset_spec(
        task_name="rte",
        num_classes=2,
    ),
    "glue_qnli": build_glue_dataset_spec(
        task_name="qnli",
        num_classes=2,
    ),
    "glue_cola": build_glue_dataset_spec(
        task_name="cola",
        num_classes=2,
        primary_metric_name="matthews_correlation",
        primary_metric_display="Matthews Correlation (%)",
        metric_names=("matthews_correlation", "accuracy"),
    ),
    "glue_stsb": build_glue_dataset_spec(
        task_name="stsb",
        num_classes=1,
        primary_metric_name="pearson",
        primary_metric_display="Pearson Correlation (%)",
        metric_names=("pearson", "spearmanr"),
        problem_type="regression",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    if role not in {"teacher", "student"}:
        raise KeyError(f"Unknown role: {role}")
    name_key = f"{role}_name"
    if name_key in pair_spec:
        return str(pair_spec[name_key])
    if role in pair_spec:
        return str(pair_spec[role])
    raise KeyError(f"Pair spec is missing '{name_key}' or '{role}'.")


def build_pair_model(
    dataset_name: str,
    pair_name: str,
    role: str,
    num_classes: int,
    *,
    initialize_pretrained: bool = True,
):
    pair_spec = get_pair_spec(dataset_name, pair_name)
    builder_key = f"{role}_builder"
    if builder_key in pair_spec:
        return pair_spec[builder_key](num_classes)
    model_name = get_role_name(pair_spec, role)
    if dataset_name == "cifar10":
        return build_cifar10_model(model_name, num_classes)
    if dataset_name == "cifar100":
        return build_cifar100_model(model_name, num_classes)
    if dataset_name == "oxford_pets":
        return build_pet_model(model_name, num_classes, pretrained=initialize_pretrained)
    if DATASET_REGISTRY[dataset_name].task_type == "text":
        return build_glue_model(model_name, num_classes, pretrained=initialize_pretrained)
    raise ValueError(f"No builder registered for {dataset_name}:{pair_name}:{role}")


def get_transforms(dataset_name: str) -> tuple[transforms.Compose, transforms.Compose]:
    dataset_spec = DATASET_REGISTRY[dataset_name]
    if dataset_spec.task_type != "vision":
        raise ValueError(f"Dataset '{dataset_name}' is not a vision dataset.")
    if dataset_spec.mean is None or dataset_spec.std is None:
        raise ValueError(f"Vision dataset '{dataset_name}' is missing normalization statistics.")
    normalize = transforms.Normalize(dataset_spec.mean, dataset_spec.std)
    if dataset_name in {"cifar10", "cifar100"}:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(dataset_spec.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        return train_transform, test_transform
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(dataset_spec.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(dataset_spec.image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, test_transform


def get_dataset(
    dataset_name: str,
    root: str,
    train: bool,
    download: bool,
    *,
    transform_override=None,
):
    dataset_spec = DATASET_REGISTRY[dataset_name]
    if dataset_spec.task_type != "vision":
        raise ValueError(f"Dataset '{dataset_name}' is not loaded through the vision dataset path.")
    if dataset_spec.dataset_class is None:
        raise ValueError(f"Vision dataset '{dataset_name}' does not define a dataset class.")
    train_transform, test_transform = get_transforms(dataset_name)
    selected_transform = (
        transform_override
        if transform_override is not None
        else (train_transform if train else test_transform)
    )
    if dataset_name in {"cifar10", "cifar100"}:
        return dataset_spec.dataset_class(
            root=root,
            train=train,
            download=download,
            transform=selected_transform,
        )
    if dataset_name == "oxford_pets":
        split = dataset_spec.train_split if train else dataset_spec.test_split
        return dataset_spec.dataset_class(
            root=root,
            split=split,
            target_types="category",
            download=download,
            transform=selected_transform,
        )
    raise ValueError(f"No vision dataset loader registered for {dataset_name}")


def build_stratified_split_indices(
    labels: list[int],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return train, validation, and class-interleaved calibration indices."""
    class_to_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        class_to_indices.setdefault(int(label), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    train_by_class: dict[int, list[int]] = {}
    validation_indices: list[int] = []
    for class_id in sorted(class_to_indices):
        class_indices = class_to_indices[class_id]
        order = torch.randperm(len(class_indices), generator=generator).tolist()
        validation_count = max(1, round(len(class_indices) * validation_fraction))
        validation_indices.extend(class_indices[position] for position in order[:validation_count])
        train_by_class[class_id] = [
            class_indices[position] for position in order[validation_count:]
        ]
    train_indices = sorted(index for indices in train_by_class.values() for index in indices)
    calibration_indices = build_class_interleaved_indices(
        labels,
        train_indices,
        seed=seed + 1,
    )
    return train_indices, sorted(validation_indices), calibration_indices


def build_class_interleaved_indices(
    labels: list[int],
    eligible_indices: list[int],
    *,
    seed: int,
) -> list[int]:
    """Return a seeded, class-round-robin order over eligible examples."""
    by_class: dict[int, list[int]] = {}
    for index in eligible_indices:
        by_class.setdefault(int(labels[index]), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    for class_id, indices in by_class.items():
        order = torch.randperm(len(indices), generator=generator).tolist()
        by_class[class_id] = [indices[position] for position in order]
    interleaved: list[int] = []
    max_class_size = max((len(indices) for indices in by_class.values()), default=0)
    for position in range(max_class_size):
        for class_id in sorted(by_class):
            indices = by_class[class_id]
            if position < len(indices):
                interleaved.append(indices[position])
    return interleaved


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    root: str,
    download: bool,
    num_workers: int,
    *,
    seed: int = 42,
    pin_memory: bool | None = None,
    validation_fraction: float | None = None,
    validation_split_seed: int | None = None,
) -> TrainingLoaders:
    dataset_spec = DATASET_REGISTRY[dataset_name]
    if dataset_spec.task_type != "vision":
        raise ValueError(f"Dataset '{dataset_name}' is not loaded through the vision dataloader path.")
    train_set = get_dataset(dataset_name, root=root, train=True, download=download)
    test_set = get_dataset(dataset_name, root=root, train=False, download=download)
    _, calibration_transform = get_transforms(dataset_name)
    calibration_set = get_dataset(
        dataset_name,
        root=root,
        train=True,
        download=False,
        transform_override=calibration_transform,
    )
    generator = torch.Generator().manual_seed(seed)
    if pin_memory is None:
        pin_memory = DEFAULT_DEVICE.type == "cuda"
    effective_validation_fraction = (
        dataset_spec.validation_fraction
        if validation_fraction is None
        else validation_fraction
    )
    effective_split_seed = (
        dataset_spec.validation_split_seed
        if validation_split_seed is None
        else validation_split_seed
    )
    evaluation_set = test_set
    final_test_set = None
    train_indices: list[int] = []
    validation_indices: list[int] = []
    calibration_indices: list[int] = []
    raw_labels = (
        getattr(train_set, "_labels")
        if hasattr(train_set, "_labels")
        else getattr(train_set, "targets")
    )
    labels = [int(label) for label in raw_labels]
    if effective_validation_fraction > 0:
        # Use identical deterministic, class-stratified indices with distinct
        # train/evaluation transforms. The official test split remains unseen
        # until the validation-selected model is restored.
        _, eval_transform = get_transforms(dataset_name)
        eval_source = get_dataset(
            dataset_name,
            root=root,
            train=True,
            download=False,
            transform_override=eval_transform,
        )
        train_indices, validation_indices, calibration_indices = build_stratified_split_indices(
            labels,
            validation_fraction=effective_validation_fraction,
            seed=effective_split_seed,
        )
        train_set = Subset(train_set, train_indices)
        evaluation_set = Subset(eval_source, validation_indices)
        calibration_set = Subset(calibration_set, calibration_indices)
        final_test_set = test_set
    else:
        train_indices = list(range(len(labels)))
        calibration_indices = build_class_interleaved_indices(
            labels,
            train_indices,
            seed=seed + 1,
        )
        calibration_set = Subset(calibration_set, calibration_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    evaluation_loader = DataLoader(
        evaluation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    final_test_loader = None
    if final_test_set is not None:
        final_test_loader = DataLoader(
            final_test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    calibration_loader = DataLoader(
        calibration_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    split_metadata = {
        "profile": "fixed_stratified_holdout" if effective_validation_fraction > 0 else "official_train",
        "train_examples": len(train_indices),
        "calibration_profile": "seeded_class_round_robin",
        "calibration_examples": len(calibration_indices),
        "calibration_seed": (
            effective_split_seed + 1 if effective_validation_fraction > 0 else seed + 1
        ),
    }
    if effective_validation_fraction > 0:
        split_metadata.update({
            "seed": effective_split_seed,
            "validation_fraction": effective_validation_fraction,
            "validation_examples": len(validation_indices),
        })
    uses_validation = effective_validation_fraction > 0
    return TrainingLoaders(
        train_loader,
        evaluation_loader,
        final_test_loader,
        calibration_loader,
        split_metadata,
        eval_split_name="validation" if uses_validation else dataset_spec.eval_split_name,
        final_test_split_name=(dataset_spec.test_split or "test") if uses_validation else None,
        restore_best_state=uses_validation,
    )

def format_float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def sanitize_tag(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "_")


def resolve_train_settings(
    dataset_spec: DatasetSpec,
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any] | None = None,
) -> TrainSettings:
    settings = dataset_spec.train_settings
    if pair_spec is not None:
        train_defaults = pair_spec.get("train_defaults", {})
        for key, value in train_defaults.items():
            if key not in TRAIN_SETTING_OVERRIDE_KEYS:
                raise ValueError(f"Unknown train_defaults key: {key}")
            settings = replace(settings, **{key: value})
    if args.optimizer is not None:
        settings = replace(settings, optimizer_name=args.optimizer)
    if args.batch_size is not None:
        settings = replace(settings, batch_size=args.batch_size)
    if args.epochs is not None:
        settings = replace(settings, epochs=args.epochs)
    if args.lr is not None:
        settings = replace(settings, lr=args.lr)
    settings = replace(settings, lr=settings.lr * args.lr_scale)
    if args.momentum is not None:
        settings = replace(settings, momentum=args.momentum)
    if args.weight_decay is not None:
        settings = replace(settings, weight_decay=args.weight_decay)
    if args.kd_temperature is not None:
        settings = replace(settings, kd_temperature=args.kd_temperature)
    if args.kd_fraction is not None:
        loss_weight_total = settings.kd_loss_weight + settings.ce_loss_weight
        settings = replace(
            settings,
            kd_loss_weight=loss_weight_total * args.kd_fraction,
            ce_loss_weight=loss_weight_total * (1.0 - args.kd_fraction),
        )
    if args.kd_weight is not None:
        settings = replace(settings, kd_loss_weight=args.kd_weight)
    if args.ce_weight is not None:
        settings = replace(settings, ce_loss_weight=args.ce_weight)
    return settings


def resolve_compressed_train_mode(args: argparse.Namespace, pair_spec: Mapping[str, Any]) -> str:
    train_mode = args.compressed_train_mode
    if train_mode is None:
        train_mode = pair_spec.get("compressed_train_mode", "distillation")
    if train_mode not in {"distillation", "supervised"}:
        raise ValueError(f"Unsupported compressed training mode: {train_mode}")
    return str(train_mode)


def resolve_head_num(args: argparse.Namespace, pair_spec: Mapping[str, Any], settings: TrainSettings) -> int:
    if args.head_num is not None:
        return args.head_num
    if "default_head_num" in pair_spec:
        return int(pair_spec["default_head_num"])
    return settings.default_head_num


def resolve_compress_linear(pair_spec: Mapping[str, Any]) -> bool:
    """Return the shared target-layer policy for both inherited methods."""
    return bool(pair_spec.get("compress_linear", False))


def resolve_capacity_size(args: argparse.Namespace) -> str:
    """Resolve the public capacity default without mutating parsed arguments."""
    if args.size is not None:
        return str(args.size)
    return "large" if args.method == "hetero" else "small"


def resolve_fixed_rank(args: argparse.Namespace, pair_spec: Mapping[str, Any]) -> int:
    if args.rank is not None:
        return args.rank
    rank_presets = pair_spec.get("rank_presets", {})
    preset_name = resolve_capacity_size(args)
    if preset_name not in rank_presets:
        available = ", ".join(sorted(rank_presets.keys()))
        raise ValueError(f"No rank preset '{preset_name}'. Available: {available}")
    return int(rank_presets[preset_name])


def validate_args(args: argparse.Namespace, pair_spec: Mapping[str, Any]) -> None:
    for name in ("epochs", "batch_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("lr", "kd_temperature"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.lr_scale <= 0:
        raise ValueError("--lr-scale must be positive.")
    for name in ("weight_decay", "kd_weight", "ce_weight", "momentum"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.kd_fraction is not None and not 0.0 <= args.kd_fraction <= 1.0:
        raise ValueError("--kd-fraction must be in [0, 1].")
    if args.kd_fraction is not None and (args.kd_weight is not None or args.ce_weight is not None):
        raise ValueError("--kd-fraction is mutually exclusive with --kd-weight and --ce-weight.")
    if args.head_num is not None and args.head_num <= 0:
        raise ValueError("--head-num must be positive.")
    if args.rank is not None and args.rank <= 0:
        raise ValueError("--rank must be positive.")
    if args.method == "hetero" and args.rank is not None:
        raise ValueError(
            "Hetero uses only registered --size small|large ranks; "
            "--rank is retained for InherNet baseline diagnostics."
        )
    if args.search_validation:
        supported = args.dataset in {"cifar10", "cifar100"} or args.dataset.startswith("glue_")
        if not supported:
            raise ValueError("--search-validation is only supported for CIFAR and GLUE datasets.")
    if args.method == "hetero" and args.hetero_expert_noise_scale < 0:
        raise ValueError("--hetero-expert-noise-scale must be non-negative.")
    if args.method == "hetero" and args.max_calib_batches <= 0:
        raise ValueError("--max-calib-batches must be positive.")
    if args.method == "hetero" and args.hetero_max_features_per_batch <= 0:
        raise ValueError("--hetero-max-features-per-batch must be positive.")
    if args.method == "hetero" and not 0.0 <= args.hetero_second_moment_shrinkage <= 1.0:
        raise ValueError("--hetero-second-moment-shrinkage must be in [0, 1].")
    if args.method == "hetero" and args.aux_loss_weight < 0:
        raise ValueError("--aux-loss-weight must be non-negative.")
    if (
        args.method == "hetero"
        and args.hetero_allocation_scale.startswith("research_")
        and not args.inheritance_diagnostics_only
    ):
        raise ValueError(
            "research_* rank policies are restricted to initialization-only pre-study runs."
        )
    if args.hetero_recipe_id is not None and args.method != "hetero":
        raise ValueError("--hetero-recipe-id applies only to --method hetero.")
    if args.freeze_hetero_router and args.method != "hetero":
        raise ValueError("--freeze-hetero-router applies only to --method hetero.")
    if args.inheritance_diagnostics and args.method not in {"inhernet", "hetero"}:
        raise ValueError("--inheritance-diagnostics applies only to inherited methods.")
    if args.inheritance_diagnostics_only and args.method not in {"inhernet", "hetero"}:
        raise ValueError("--inheritance-diagnostics-only applies only to inherited methods.")
    if args.method == "inhernet":
        rank = resolve_fixed_rank(args, pair_spec)
        if rank <= 0:
            raise ValueError("Resolved InherNet rank must be positive.")


def build_method_tag(
    method: str,
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any],
    settings: TrainSettings,
) -> str:
    head_num = resolve_head_num(args, pair_spec, settings)
    if method in {"teacher", "student", "student_kd"}:
        tag = "default"
    elif method == "inhernet":
        rank = resolve_fixed_rank(args, pair_spec)
        rank_source = "custom" if args.rank is not None else resolve_capacity_size(args)
        tag = f"{rank_source}_rank_{rank}_heads_{head_num}"
        if resolve_compress_linear(pair_spec):
            tag = f"{tag}_linear"
        compressed_train_mode = resolve_compressed_train_mode(args, pair_spec)
        if compressed_train_mode != "distillation":
            tag = f"{tag}_{compressed_train_mode}"
    elif method == "hetero":
        compress_linear = resolve_compress_linear(pair_spec)
        reference_rank = resolve_fixed_rank(args, pair_spec)
        capacity_size = resolve_capacity_size(args)
        tag = (
            f"{capacity_size}_matched_rank_{reference_rank}_heads_{head_num}_"
            f"calib_{args.max_calib_batches}_"
            f"samples_{args.hetero_max_features_per_batch}_"
            f"shrink_{format_float_tag(args.hetero_second_moment_shrinkage)}_"
            f"allocation_{sanitize_tag(args.hetero_allocation_scale)}_"
            f"noise_{format_float_tag(args.hetero_expert_noise_scale)}_"
            f"aux_{format_float_tag(args.aux_loss_weight)}"
        )
        if compress_linear:
            tag = f"{tag}_linear"
        if args.freeze_hetero_router:
            tag = f"{tag}_frozen_router"
        compressed_train_mode = resolve_compressed_train_mode(args, pair_spec)
        if compressed_train_mode != "distillation":
            tag = f"{tag}_{compressed_train_mode}"
    else:
        raise ValueError(f"Unknown method: {method}")
    if args.search_candidate:
        tag = f"search_{sanitize_tag(args.search_candidate)}_{tag}"
    return tag


def build_training_dataloaders(
    args: argparse.Namespace,
    settings: TrainSettings,
    device: torch.device | None = None,
) -> TrainingLoaders:
    runtime_device = resolve_device(args.device) if device is None else device
    dataset_spec = DATASET_REGISTRY[args.dataset]
    if dataset_spec.task_type == "text":
        pair_spec = get_pair_spec(args.dataset, args.pair)
        tokenizer_name = str(pair_spec.get("tokenizer_name", get_role_name(pair_spec, "teacher")))
        if dataset_spec.text_task_name is None:
            raise ValueError(f"Text dataset '{args.dataset}' is missing a GLUE task name.")
        train_loader, eval_loader, calibration_loader, split_metadata = build_glue_dataloaders(
            task_name=dataset_spec.text_task_name,
            eval_split_name=dataset_spec.eval_split_name,
            problem_type=dataset_spec.problem_type,
            root=args.data_root,
            batch_size=settings.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=runtime_device.type == "cuda",
            tokenizer_name=tokenizer_name,
            tokenizer_revision=str(pair_spec["tokenizer_revision"]),
            max_length=dataset_spec.text_max_length,
            search_validation=args.search_validation,
            validation_split_seed=dataset_spec.validation_split_seed,
        )
        split_metadata = dict(split_metadata)
        split_metadata["teacher_revision"] = str(pair_spec["teacher_revision"])
        return TrainingLoaders(
            train_loader,
            eval_loader,
            calibration=calibration_loader,
            split_metadata=split_metadata,
            eval_split_name=str(split_metadata["evaluation_split"]),
            restore_best_state=True,
        )
    search_validation = args.search_validation
    validation_fraction = 0.1 if search_validation and args.dataset in {"cifar10", "cifar100"} else None
    loaders = get_dataloaders(
        args.dataset,
        batch_size=settings.batch_size,
        root=args.data_root,
        download=args.download,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=runtime_device.type == "cuda",
        validation_fraction=validation_fraction,
    )
    if not args.final_test:
        return replace(loaders, final_test=None)
    return loaders
