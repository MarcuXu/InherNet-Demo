from __future__ import annotations

from typing import Callable, Dict, Mapping

import torch.nn as nn
from torchvision.models import resnet18, resnet50


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


PAIR_REGISTRY: dict[str, Mapping[str, object]] = {
    "resnet50_to_resnet18": {
        "teacher_name": "resnet50",
        "student_name": "resnet18",
        "teacher_builder": lambda num_classes: build_cifar_torchvision_resnet("resnet50", num_classes),
        "student_builder": lambda num_classes: build_cifar_torchvision_resnet("resnet18", num_classes),
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
        "model_profile": "cifar_resnet_stem",
    },
    "resnet50_to_resnet18_cifar_stem": {
        "teacher_name": "resnet50",
        "student_name": "resnet18",
        "teacher_builder": lambda num_classes: build_cifar_torchvision_resnet("resnet50", num_classes),
        "student_builder": lambda num_classes: build_cifar_torchvision_resnet("resnet18", num_classes),
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
        "model_profile": "cifar_resnet_stem",
    },
    "resnet50_to_resnet18_org": {
        "teacher_name": "resnet50_org",
        "student_name": "resnet18_org",
        "teacher_builder": lambda num_classes: build_org_torchvision_resnet("resnet50", num_classes),
        "student_builder": lambda num_classes: build_org_torchvision_resnet("resnet18", num_classes),
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
        "model_profile": "demo_code_org_torchvision_stem",
    },
    "resnet50_to_resnet18_torchvision_stem": {
        "teacher_name": "resnet50_org",
        "student_name": "resnet18_org",
        "teacher_builder": lambda num_classes: build_org_torchvision_resnet("resnet50", num_classes),
        "student_builder": lambda num_classes: build_org_torchvision_resnet("resnet18", num_classes),
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
        "model_profile": "demo_code_org_torchvision_stem",
    },
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown CIFAR-10 model: {model_name}")
    return MODEL_REGISTRY[model_name](num_classes)
