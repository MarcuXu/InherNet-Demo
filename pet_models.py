from __future__ import annotations

from typing import Callable, Dict, Mapping

import torch.nn as nn
from torchvision.models import resnet18, resnet34


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected a torchvision ResNet with a linear fc head.")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet18_pet(num_classes: int = 37) -> nn.Module:
    return _replace_classifier(resnet18(weights=None), num_classes)


def resnet34_pet(num_classes: int = 37) -> nn.Module:
    return _replace_classifier(resnet34(weights=None), num_classes)


MODEL_REGISTRY: Dict[str, Callable[[int], nn.Module]] = {
    "resnet18": resnet18_pet,
    "resnet34": resnet34_pet,
}


PET_INHERITANCE_DEFAULTS = {
    "compressed_source": "teacher",
    "compressed_train_mode": "distillation",
    "model_profile": "oxford_pets_resnet_teacher_inheritance",
}


PAIR_REGISTRY: Dict[str, Mapping[str, object]] = {
    "resnet34_to_resnet18": {
        **PET_INHERITANCE_DEFAULTS,
        "teacher_name": "resnet34",
        "student_name": "resnet18",
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 3,
    }
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown Oxford-IIIT Pet model: {model_name}")
    return MODEL_REGISTRY[model_name](num_classes)
