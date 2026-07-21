from __future__ import annotations

from typing import Callable, Dict, Mapping

import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected a torchvision ResNet with a linear fc head.")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet18_pet(num_classes: int = 37, *, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return _replace_classifier(resnet18(weights=weights), num_classes)


def resnet34_pet(num_classes: int = 37, *, pretrained: bool = True) -> nn.Module:
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    return _replace_classifier(resnet34(weights=weights), num_classes)


MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18_pet,
    "resnet34": resnet34_pet,
}


PET_INHERITANCE_DEFAULTS = {
    "compressed_train_mode": "distillation",
    "model_profile": "oxford_pets_imagenet_transfer",
}


PAIR_REGISTRY: Dict[str, Mapping[str, object]] = {
    "resnet34_to_resnet18": {
        **PET_INHERITANCE_DEFAULTS,
        "teacher_name": "resnet34",
        "student_name": "resnet18",
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
        "inhernet_protocol_source": "repository_extension",
        "inhernet_rank_source": "repository_defined",
    }
}


def build_model(model_name: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown Oxford-IIIT Pet model: {model_name}")
    return MODEL_REGISTRY[model_name](num_classes, pretrained=pretrained)
