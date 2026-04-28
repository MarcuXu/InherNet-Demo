from __future__ import annotations

from typing import Callable, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFARResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int = 100, width_mult: int = 1) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"Depth {depth} is invalid for a CIFAR ResNet.")
        blocks_per_stage = (depth - 2) // 6
        base_channels = 16 * width_mult
        self.in_planes = base_channels

        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.layer1 = self._make_layer(base_channels, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks_per_stage, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(CIFARBasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


VGG_CONFIGS = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}


class CIFARVGG(nn.Module):
    def __init__(self, name: str, num_classes: int = 100) -> None:
        super().__init__()
        if name not in VGG_CONFIGS:
            raise ValueError(f"Unknown VGG variant: {name}")
        self.features = self._make_layers(VGG_CONFIGS[name])
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg: list[object]) -> nn.Sequential:
        layers = []
        in_channels = 3
        for item in cfg:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(item)
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dropout_rate = dropout_rate
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class WideNetworkBlock(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_planes: int,
        out_planes: int,
        block: type[WideBasicBlock],
        stride: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        layers = []
        for idx in range(num_blocks):
            layers.append(
                block(
                    in_planes if idx == 0 else out_planes,
                    out_planes,
                    stride if idx == 0 else 1,
                    dropout_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_classes: int = 100,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"Depth {depth} is invalid for a WideResNet.")
        blocks_per_stage = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = WideNetworkBlock(
            blocks_per_stage,
            channels[0],
            channels[1],
            WideBasicBlock,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.block2 = WideNetworkBlock(
            blocks_per_stage,
            channels[1],
            channels[2],
            WideBasicBlock,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.block3 = WideNetworkBlock(
            blocks_per_stage,
            channels[2],
            channels[3],
            WideBasicBlock,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet8(num_classes: int = 100) -> nn.Module:
    return CIFARResNet(depth=8, num_classes=num_classes)


def resnet20(num_classes: int = 100) -> nn.Module:
    return CIFARResNet(depth=20, num_classes=num_classes)


def resnet32(num_classes: int = 100) -> nn.Module:
    return CIFARResNet(depth=32, num_classes=num_classes)


def resnet56(num_classes: int = 100) -> nn.Module:
    return CIFARResNet(depth=56, num_classes=num_classes)


def resnet110(num_classes: int = 100) -> nn.Module:
    return CIFARResNet(depth=110, num_classes=num_classes)


def resnet8x4(num_classes: int = 100) -> nn.Module:
    return CIFARResNet(depth=8, num_classes=num_classes, width_mult=4)


def resnet32x4(num_classes: int = 100) -> nn.Module:
    return CIFARResNet(depth=32, num_classes=num_classes, width_mult=4)


def vgg8(num_classes: int = 100) -> nn.Module:
    return CIFARVGG(name="vgg8", num_classes=num_classes)


def vgg13(num_classes: int = 100) -> nn.Module:
    return CIFARVGG(name="vgg13", num_classes=num_classes)


def wrn_16_2(num_classes: int = 100) -> nn.Module:
    return WideResNet(depth=16, widen_factor=2, num_classes=num_classes)


def wrn_40_1(num_classes: int = 100) -> nn.Module:
    return WideResNet(depth=40, widen_factor=1, num_classes=num_classes)


def wrn_40_2(num_classes: int = 100) -> nn.Module:
    return WideResNet(depth=40, widen_factor=2, num_classes=num_classes)


MODEL_REGISTRY: Dict[str, Callable[[int], nn.Module]] = {
    "resnet8": resnet8,
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet8x4": resnet8x4,
    "resnet32x4": resnet32x4,
    "vgg8": vgg8,
    "vgg13": vgg13,
    "wrn_16_2": wrn_16_2,
    "wrn_40_1": wrn_40_1,
    "wrn_40_2": wrn_40_2,
}


CIFAR100_INHERNET_WORKFLOW_DEFAULTS = {
    "compressed_source": "teacher",
    "compressed_train_mode": "distillation",
    "model_profile": "paper_cifar100_teacher_inheritance",
}


PAIR_REGISTRY: Dict[str, Mapping[str, object]] = {
    "resnet32_to_resnet8": {
        **CIFAR100_INHERNET_WORKFLOW_DEFAULTS,
        "teacher": "resnet32",
        "student": "resnet8",
        "rank_presets": {"small": 4, "large": 8},
        "default_head_num": 3,
    },
    "resnet32x4_to_resnet8x4": {
        **CIFAR100_INHERNET_WORKFLOW_DEFAULTS,
        "teacher": "resnet32x4",
        "student": "resnet8x4",
        "rank_presets": {"small": 4, "large": 8},
        "default_head_num": 3,
    },
    "vgg13_to_vgg8": {
        **CIFAR100_INHERNET_WORKFLOW_DEFAULTS,
        "teacher": "vgg13",
        "student": "vgg8",
        "rank_presets": {"small": 128, "large": 256},
        "default_head_num": 3,
    },
    "wrn40_2_to_wrn40_1": {
        **CIFAR100_INHERNET_WORKFLOW_DEFAULTS,
        "teacher": "wrn_40_2",
        "student": "wrn_40_1",
        "rank_presets": {"small": 16, "large": 32},
        "default_head_num": 3,
    },
    "wrn40_2_to_wrn16_2": {
        **CIFAR100_INHERNET_WORKFLOW_DEFAULTS,
        "teacher": "wrn_40_2",
        "student": "wrn_16_2",
        "rank_presets": {"small": 16, "large": 32},
        "default_head_num": 3,
    },
    "resnet56_to_resnet20": {
        **CIFAR100_INHERNET_WORKFLOW_DEFAULTS,
        "teacher": "resnet56",
        "student": "resnet20",
        "rank_presets": {"small": 8, "large": 16},
        "default_head_num": 3,
    },
    "resnet110_to_resnet32": {
        **CIFAR100_INHERNET_WORKFLOW_DEFAULTS,
        "teacher": "resnet110",
        "student": "resnet32",
        "rank_presets": {"small": 8, "large": 32},
        "default_head_num": 3,
    },
    "resnet110_to_resnet20": {
        **CIFAR100_INHERNET_WORKFLOW_DEFAULTS,
        "teacher": "resnet110",
        "student": "resnet20",
        "rank_presets": {"small": 4, "large": 8},
        "default_head_num": 3,
    },
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown CIFAR model: {model_name}")
    return MODEL_REGISTRY[model_name](num_classes)


def build_teacher_model(pair_name: str, num_classes: int) -> nn.Module:
    if pair_name not in PAIR_REGISTRY:
        raise KeyError(f"Unknown CIFAR pair: {pair_name}")
    return build_model(str(PAIR_REGISTRY[pair_name]["teacher"]), num_classes)


def build_student_model(pair_name: str, num_classes: int) -> nn.Module:
    if pair_name not in PAIR_REGISTRY:
        raise KeyError(f"Unknown CIFAR pair: {pair_name}")
    return build_model(str(PAIR_REGISTRY[pair_name]["student"]), num_classes)
