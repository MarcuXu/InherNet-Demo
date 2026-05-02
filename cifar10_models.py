from __future__ import annotations

from typing import Callable, Dict, Mapping

import torch.nn as nn
from torchvision.models import resnet18, resnet50


DEMO_CODE_ORG_TRAIN_DEFAULTS = {
    "optimizer_name": "adam",
    "batch_size": 256,
    "epochs": 100,
    "lr": 0.001,
    "weight_decay": 0.0,
    "lr_milestones": (),
    "kd_temperature": 7.0,
    "kd_loss_weight": 0.7,
    "ce_loss_weight": 0.3,
    "legacy_eval_sticky": True,
}


def build_torchvision_resnet(arch: str, num_classes: int, *, cifar_stem: bool) -> nn.Module:
    if arch == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif arch == "resnet50":
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown torchvision ResNet architecture: {arch}")
    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def build_cifar_torchvision_resnet(arch: str, num_classes: int) -> nn.Module:
    return build_torchvision_resnet(arch, num_classes, cifar_stem=True)


def build_org_torchvision_resnet(arch: str, num_classes: int) -> nn.Module:
    return build_torchvision_resnet(arch, num_classes, cifar_stem=False)


MODEL_REGISTRY: Dict[str, Callable[[int], nn.Module]] = {
    "resnet18": lambda num_classes: build_cifar_torchvision_resnet("resnet18", num_classes),
    "resnet50": lambda num_classes: build_cifar_torchvision_resnet("resnet50", num_classes),
    "resnet18_org": lambda num_classes: build_org_torchvision_resnet("resnet18", num_classes),
    "resnet50_org": lambda num_classes: build_org_torchvision_resnet("resnet50", num_classes),
}


def make_resnet_pair(
    *,
    teacher_name: str,
    student_name: str,
    model_profile: str,
    train_defaults: Mapping[str, object] | None = None,
    compressed_source: str | None = None,
    compressed_train_mode: str | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "teacher_name": teacher_name,
        "student_name": student_name,
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
        "model_profile": model_profile,
    }
    if train_defaults is not None:
        spec["train_defaults"] = train_defaults
    if compressed_source is not None:
        spec["compressed_source"] = compressed_source
    if compressed_train_mode is not None:
        spec["compressed_train_mode"] = compressed_train_mode
    return spec


CIFAR_STEM_PAIR = make_resnet_pair(
    teacher_name="resnet50",
    student_name="resnet18",
    model_profile="cifar_resnet_stem",
)

DEMO_CODE_ORG_PAIR = make_resnet_pair(
    teacher_name="resnet50_org",
    student_name="resnet18_org",
    model_profile="demo_code_org_torchvision_stem",
    train_defaults=DEMO_CODE_ORG_TRAIN_DEFAULTS,
    compressed_source="student",
    compressed_train_mode="supervised",
)


PAIR_REGISTRY: dict[str, Mapping[str, object]] = {
    "resnet50_to_resnet18": dict(CIFAR_STEM_PAIR),
    "resnet50_to_resnet18_cifar_stem": dict(CIFAR_STEM_PAIR),
    "resnet50_to_resnet18_org": dict(DEMO_CODE_ORG_PAIR),
    "resnet50_to_resnet18_torchvision_stem": dict(DEMO_CODE_ORG_PAIR),
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown CIFAR-10 model: {model_name}")
    return MODEL_REGISTRY[model_name](num_classes)
