from __future__ import annotations

from typing import Callable, Dict, Mapping

import torch.nn as nn
from torchvision.models import resnet18, resnet50


def build_torchvision_resnet(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif arch == "resnet50":
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown torchvision ResNet architecture: {arch}")
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_cifar_torchvision_resnet(arch: str, num_classes: int) -> nn.Module:
    return build_torchvision_resnet(arch, num_classes)


MODEL_REGISTRY: Dict[str, Callable[[int], nn.Module]] = {
    "resnet18": lambda num_classes: build_cifar_torchvision_resnet("resnet18", num_classes),
    "resnet50": lambda num_classes: build_cifar_torchvision_resnet("resnet50", num_classes),
}


def make_resnet_pair(
    *,
    teacher_name: str,
    student_name: str,
    model_profile: str,
) -> dict[str, object]:
    return {
        "teacher_name": teacher_name,
        "student_name": student_name,
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
        "inhernet_protocol_source": "repository_extension",
        "inhernet_rank_source": "repository_defined",
        "model_profile": model_profile,
    }


PAIR_REGISTRY: dict[str, Mapping[str, object]] = {
    "resnet50_to_resnet18": make_resnet_pair(
        teacher_name="resnet50",
        student_name="resnet18",
        model_profile="cifar_resnet_stem",
    ),
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown CIFAR-10 model: {model_name}")
    return MODEL_REGISTRY[model_name](num_classes)
