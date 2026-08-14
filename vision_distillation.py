"""Vision-distillation building blocks used by the CIFAR baselines.

The module deliberately contains no experiment registry, optimizer, checkpoint,
or training-loop policy.  Each auxiliary module owns only its trainable
adapter; the teacher remains an external frozen model.  This keeps the
published method objectives explicit when they are later wired into the
maintained training pipeline.

Implemented source recipes:

* CAT-KD: classifier-weighted CAM matching with the released two-by-two pool.
* SimKD: the factor-two transfer projector and reused frozen teacher classifier.
* ReviewKD: the ABF review hierarchy, HCL feature loss, and its warm-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

from cifar100_models import CIFARResNet, CIFARVGG, WideResNet


@dataclass(frozen=True)
class VisionFeatures:
    """Explicit intermediate outputs from a supported vision classifier.

    ``stage_maps`` are ordered shallow-to-deep using the source method's
    stage-boundary convention. ``final_map`` is the native post-activation
    convolutional map used by CAT-KD and CRD; ``pooled`` is the vector passed
    to ``classifier``.
    """

    logits: torch.Tensor
    stage_maps: tuple[torch.Tensor, ...]
    final_map: torch.Tensor
    pooled: torch.Tensor
    classifier: nn.Linear


class DistillationObjective(NamedTuple):
    """Loss components returned by each trainable vision distiller."""

    logits: torch.Tensor
    ce_loss: torch.Tensor
    feature_loss: torch.Tensor
    total_loss: torch.Tensor


@dataclass(frozen=True)
class DistillerParameterCounts:
    """Parameter accounting that distinguishes training and inference cost."""

    student_parameters: int
    replaced_student_parameters: int
    deployment_auxiliary_parameters: int
    training_only_auxiliary_parameters: int
    deployment_parameters: int
    optimization_parameters: int


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _teacher_features(teacher: nn.Module, inputs: torch.Tensor) -> VisionFeatures:
    teacher.eval()
    with torch.no_grad():
        return extract_vision_features(teacher, inputs)


def review_feature_maps(features: VisionFeatures) -> tuple[torch.Tensor, ...]:
    """Return ReviewKD's shallow-to-deep maps followed by the pooled map."""

    return (*features.stage_maps, features.pooled.unsqueeze(-1).unsqueeze(-1))


def simkd_feature_map(model: nn.Module, features: VisionFeatures) -> torch.Tensor:
    """Return SimKD's final spatial feature for a supported CIFAR model.

    The released WideResNet emits its final spatial feature immediately after
    ``block3``; the classifier's terminal BN/ReLU is applied only before
    pooling. Other maintained model families expose their native final map at
    the same source tap as ``features.final_map``.
    """

    if isinstance(model, WideResNet):
        return features.stage_maps[-1]
    return features.final_map


def _forward_cifar_residual_block_preact(
    block: nn.Module, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a maintained CIFAR residual block and expose its pre-ReLU sum."""

    if not hasattr(block, "shortcut"):
        raise TypeError(f"Unsupported CIFAR residual block: {type(block)!r}")
    output = F.relu(block.bn1(block.conv1(inputs)), inplace=False)
    output = block.bn2(block.conv2(output))
    preactivation = output + block.shortcut(inputs)
    return F.relu(preactivation, inplace=False), preactivation


def _forward_cifar_residual_stage_preact(
    stage: nn.Sequential, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = inputs
    final_preactivation: torch.Tensor | None = None
    for block in stage:
        output, final_preactivation = _forward_cifar_residual_block_preact(block, output)
    if final_preactivation is None:
        raise ValueError("A CIFAR residual stage must contain at least one block.")
    return output, final_preactivation


def extract_review_teacher_maps(
    teacher: nn.Module, inputs: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Return ReviewKD's released pre-activation teacher targets.

    ReviewKD supervises the output immediately before the last ReLU of each
    residual stage, followed by the pooled feature. The current formal
    ReviewKD registry contains only CIFAR ResNet and WideResNet teachers.
    """

    teacher.eval()
    with torch.no_grad():
        if isinstance(teacher, CIFARResNet):
            output = F.relu(teacher.bn1(teacher.conv1(inputs)), inplace=False)
            output, stage1 = _forward_cifar_residual_stage_preact(teacher.layer1, output)
            output, stage2 = _forward_cifar_residual_stage_preact(teacher.layer2, output)
            output, stage3 = _forward_cifar_residual_stage_preact(teacher.layer3, output)
            pooled = teacher.avgpool(output)
            return stage1, stage2, stage3, pooled
        if isinstance(teacher, WideResNet):
            output = teacher.conv1(inputs)
            stage1_output = teacher.block1(output)
            stage2_output = teacher.block2(stage1_output)
            stage3_output = teacher.block3(stage2_output)
            stage1 = teacher.block2.layer[0].bn1(stage1_output)
            stage2 = teacher.block3.layer[0].bn1(stage2_output)
            stage3 = teacher.bn(stage3_output)
            pooled = F.adaptive_avg_pool2d(F.relu(stage3, inplace=False), 1)
            return stage1, stage2, stage3, pooled
    raise TypeError(f"Unsupported ReviewKD teacher model: {type(teacher)!r}")


def _extract_cifar_resnet_features(model: CIFARResNet, inputs: torch.Tensor) -> VisionFeatures:
    output = F.relu(model.bn1(model.conv1(inputs)), inplace=True)
    stage1 = model.layer1(output)
    stage2 = model.layer2(stage1)
    stage3 = model.layer3(stage2)
    pooled = torch.flatten(model.avgpool(stage3), 1)
    logits = model.fc(pooled)
    return VisionFeatures(logits, (stage1, stage2, stage3), stage3, pooled, model.fc)


def _extract_cifar_vgg_features(model: CIFARVGG, inputs: torch.Tensor) -> VisionFeatures:
    output = inputs
    stage_maps: list[torch.Tensor] = []
    feature_layers = tuple(model.features)
    for index, layer in enumerate(feature_layers):
        output = layer(output)
        if (
            isinstance(layer, nn.ReLU)
            and index + 1 < len(feature_layers)
            and isinstance(feature_layers[index + 1], nn.MaxPool2d)
        ):
            # A VGG block ends at its last ReLU, before spatial pooling.  This
            # is the source implementation's feature/CAM convention.
            stage_maps.append(output)

    final_map = stage_maps[-1]
    pooled = torch.flatten(output, 1)
    logits = model.classifier(pooled)
    return VisionFeatures(logits, tuple(stage_maps), final_map, pooled, model.classifier)


def _extract_wide_resnet_features(model: WideResNet, inputs: torch.Tensor) -> VisionFeatures:
    output = model.conv1(inputs)
    stage1 = model.block1(output)
    stage2 = model.block2(stage1)
    stage3 = model.block3(stage2)
    final_map = F.relu(model.bn(stage3), inplace=True)
    pooled = torch.flatten(F.adaptive_avg_pool2d(final_map, 1), 1)
    logits = model.fc(pooled)
    return VisionFeatures(logits, (stage1, stage2, stage3), final_map, pooled, model.fc)


def _extract_torchvision_resnet_features(model: ResNet, inputs: torch.Tensor) -> VisionFeatures:
    output = model.conv1(inputs)
    output = model.bn1(output)
    output = model.relu(output)
    output = model.maxpool(output)
    stage1 = model.layer1(output)
    stage2 = model.layer2(stage1)
    stage3 = model.layer3(stage2)
    stage4 = model.layer4(stage3)
    pooled = torch.flatten(model.avgpool(stage4), 1)
    logits = model.fc(pooled)
    return VisionFeatures(logits, (stage1, stage2, stage3, stage4), stage4, pooled, model.fc)


def extract_vision_features(model: nn.Module, inputs: torch.Tensor) -> VisionFeatures:
    """Run a supported current vision model while exposing its feature maps.

    The extraction paths reproduce the models' native forwards rather than
    using hooks. This avoids hidden interaction with their in-place activations
    and keeps batch-normalization behavior identical to ordinary training.
    """

    if isinstance(model, CIFARResNet):
        return _extract_cifar_resnet_features(model, inputs)
    if isinstance(model, CIFARVGG):
        return _extract_cifar_vgg_features(model, inputs)
    if isinstance(model, WideResNet):
        return _extract_wide_resnet_features(model, inputs)
    if isinstance(model, ResNet):
        return _extract_torchvision_resnet_features(model, inputs)
    raise TypeError(f"Unsupported vision model for feature distillation: {type(model)!r}")


def classifier_weight_cams(feature_map: torch.Tensor, classifier: nn.Linear) -> torch.Tensor:
    """Return the bias-free per-class CAMs used by CAT-KD.

    The released CAT-KD models use a bias-free one-by-one classifier convolution.
    Applying a current model's linear classifier weights in this form is exactly
    the same operation; its ordinary classifier bias is intentionally excluded.
    """

    return F.conv2d(feature_map, classifier.weight.unsqueeze(-1).unsqueeze(-1))


def cat_kd_loss(
    student: VisionFeatures,
    teacher: VisionFeatures,
    *,
    beta: float,
    cam_resolution: int = 2,
) -> torch.Tensor:
    """Return CAT-KD's feature term without the external cross-entropy term.

    The official CIFAR-100 CAT-KD configurations use full, unnormalized CAM
    transfer at resolution two. ``beta`` is intentionally mandatory because it
    is a published pair-specific coefficient rather than a universal default.
    """

    student_cams = classifier_weight_cams(student.final_map, student.classifier)
    teacher_cams = F.conv2d(
        teacher.final_map.detach(),
        teacher.classifier.weight.detach().unsqueeze(-1).unsqueeze(-1),
    )
    student_cams = F.adaptive_avg_pool2d(
        student_cams, (cam_resolution, cam_resolution)
    )
    teacher_cams = F.adaptive_avg_pool2d(
        teacher_cams, (cam_resolution, cam_resolution)
    )
    return beta * F.mse_loss(student_cams, teacher_cams)


class SimKDProjector(nn.Module):
    """The published factor-two SimKD transfer module.

    The frozen teacher classifier is passed to :meth:`forward` rather than
    registered as a child module. The projector therefore exposes exactly its
    own trainable parameters, while the deployment prediction follows SimKD's
    reused-teacher-classifier rule.
    """

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        bottleneck_channels = teacher_channels // 2
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.transfer = nn.Sequential(
            nn.Conv2d(student_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True),
        )

    def project(self, student_map: torch.Tensor) -> torch.Tensor:
        """Apply the published transfer stack to a spatially aligned student map."""

        return self.transfer(student_map)

    def transfer_maps(
        self, student_map: torch.Tensor, teacher_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align spatial maps exactly as SimKD before feature matching."""

        if student_map.shape[-2] > teacher_map.shape[-2]:
            source = F.adaptive_avg_pool2d(student_map, teacher_map.shape[-2:])
            target = teacher_map.detach()
        else:
            source = student_map
            target = F.adaptive_avg_pool2d(teacher_map.detach(), student_map.shape[-2:])
        return self.project(source), target

    def forward(
        self,
        student_map: torch.Tensor,
        teacher_map: torch.Tensor,
        teacher_classifier: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return projected student map, frozen target map, and SimKD logits."""

        projected, target = self.transfer_maps(student_map, teacher_map)
        pooled = torch.flatten(self.avg_pool(projected), 1)
        teacher_bias = None if teacher_classifier.bias is None else teacher_classifier.bias.detach()
        logits = F.linear(pooled, teacher_classifier.weight.detach(), teacher_bias)
        return projected, target, logits


def simkd_loss(projected_student_map: torch.Tensor, teacher_target_map: torch.Tensor) -> torch.Tensor:
    """Return SimKD's sole source loss: MSE on aligned transferred features."""

    return F.mse_loss(projected_student_map, teacher_target_map.detach())


class ReviewABF(nn.Module):
    """Attention-based fusion block from ReviewKD."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        *,
        fuse: bool,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.att_conv = (
            nn.Sequential(nn.Conv2d(mid_channels * 2, 2, kernel_size=1), nn.Sigmoid())
            if fuse
            else None
        )
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(
        self,
        features: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = self.conv1(features)
        if self.att_conv is not None:
            residual = F.interpolate(residual, size=transformed.shape[-2:], mode="nearest")
            attention = self.att_conv(torch.cat([transformed, residual], dim=1))
            transformed = (
                transformed * attention[:, 0].unsqueeze(1)
                + residual * attention[:, 1].unsqueeze(1)
            )
        if transformed.shape[-2:] != output_size:
            transformed = F.interpolate(transformed, size=output_size, mode="nearest")
        return self.conv2(transformed), transformed


class ReviewKDAdapter(nn.Module):
    """Train-time deep-to-shallow ABF hierarchy for teacher-aligned features."""

    def __init__(
        self,
        student_channels: Sequence[int],
        teacher_channels: Sequence[int],
    ) -> None:
        super().__init__()
        mid_channels = min(512, student_channels[-1])
        abfs = [
            ReviewABF(
                student_channel,
                mid_channels,
                teacher_channel,
                fuse=index < len(student_channels) - 1,
            )
            for index, (student_channel, teacher_channel) in enumerate(
                zip(student_channels, teacher_channels)
            )
        ]
        self.abfs = nn.ModuleList(reversed(abfs))

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        student_reversed = tuple(reversed(student_features))
        teacher_reversed = tuple(reversed(teacher_features))
        output, residual = self.abfs[0](
            student_reversed[0], output_size=teacher_reversed[0].shape[-2:]
        )
        reviewed = [output]
        for student_feature, teacher_feature, abf in zip(
            student_reversed[1:], teacher_reversed[1:], self.abfs[1:]
        ):
            output, residual = abf(
                student_feature,
                residual,
                output_size=teacher_feature.shape[-2:],
            )
            reviewed.insert(0, output)
        return tuple(reviewed)


def review_hcl_loss(
    student_features: Sequence[torch.Tensor], teacher_features: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Hierarchical context loss (HCL) from ReviewKD."""

    total_loss = student_features[0].new_zeros(())
    for student_feature, teacher_feature in zip(student_features, teacher_features):
        frozen_teacher = teacher_feature.detach()
        height = student_feature.shape[-2]
        feature_loss = F.mse_loss(student_feature, frozen_teacher)
        coefficient = 1.0
        normalizer = 1.0
        for pooled_size in (4, 2, 1):
            if pooled_size >= height:
                continue
            coefficient /= 2.0
            feature_loss = feature_loss + coefficient * F.mse_loss(
                F.adaptive_avg_pool2d(student_feature, (pooled_size, pooled_size)),
                F.adaptive_avg_pool2d(frozen_teacher, (pooled_size, pooled_size)),
            )
            normalizer += coefficient
        total_loss = total_loss + feature_loss / normalizer
    return total_loss


def review_kd_loss(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
    *,
    epoch: int,
    weight: float,
    warmup_epochs: int = 20,
) -> torch.Tensor:
    """Return ReviewKD's warmed-up feature term; CE remains external."""

    warmup = min(epoch / warmup_epochs, 1.0)
    return weight * warmup * review_hcl_loss(student_features, teacher_features)


def _parameter_counts(
    student: nn.Module,
    *,
    replaced_student_parameters: int,
    deployment_auxiliary_parameters: int,
    training_only_auxiliary_parameters: int,
    optimization_module: nn.Module,
) -> DistillerParameterCounts:
    student_parameters = _count_parameters(student)
    return DistillerParameterCounts(
        student_parameters=student_parameters,
        replaced_student_parameters=replaced_student_parameters,
        deployment_auxiliary_parameters=deployment_auxiliary_parameters,
        training_only_auxiliary_parameters=training_only_auxiliary_parameters,
        deployment_parameters=(
            student_parameters
            - replaced_student_parameters
            + deployment_auxiliary_parameters
        ),
        optimization_parameters=_count_parameters(optimization_module),
    )


def _final_classifier_parameter_count(student: nn.Module) -> int:
    """Count the current classifier replaced by SimKD's transfer head."""

    if isinstance(student, CIFARVGG):
        return _count_parameters(student.classifier)
    if isinstance(student, (CIFARResNet, WideResNet, ResNet)):
        return _count_parameters(student.fc)
    raise TypeError(f"Unsupported vision model for SimKD: {type(student)!r}")


class CATKDDistiller(nn.Module):
    """CAT-KD wrapper: task CE plus pair-specific class-attention matching."""

    def __init__(
        self,
        student: nn.Module,
        *,
        beta: float,
        ce_weight: float = 1.0,
        cam_resolution: int = 2,
    ) -> None:
        super().__init__()
        self.student = student
        self.beta = beta
        self.ce_weight = ce_weight
        self.cam_resolution = cam_resolution

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.student(inputs)

    def training_objective(
        self,
        teacher: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        criterion: nn.Module,
    ) -> DistillationObjective:
        student_features = extract_vision_features(self.student, inputs)
        teacher_features = _teacher_features(teacher, inputs)
        ce_loss = self.ce_weight * criterion(student_features.logits, labels)
        feature_loss = cat_kd_loss(
            student_features,
            teacher_features,
            beta=self.beta,
            cam_resolution=self.cam_resolution,
        )
        return DistillationObjective(
            student_features.logits,
            ce_loss,
            feature_loss,
            ce_loss + feature_loss,
        )

    def parameter_counts(self) -> DistillerParameterCounts:
        return _parameter_counts(
            self.student,
            replaced_student_parameters=0,
            deployment_auxiliary_parameters=0,
            training_only_auxiliary_parameters=0,
            optimization_module=self,
        )


class SimKDDistiller(nn.Module):
    """SimKD wrapper with a persistent frozen reused-teacher classifier."""

    def __init__(
        self,
        student: nn.Module,
        *,
        student_channels: int,
        teacher_channels: int,
        teacher_classifier: nn.Linear,
        feature_weight: float = 1.0,
        projector_factor: int = 2,
    ) -> None:
        super().__init__()
        self.student = student
        if projector_factor != 2:
            raise ValueError("The maintained SimKD projector implements the released factor-two recipe.")
        self.projector = SimKDProjector(student_channels, teacher_channels)
        self.feature_weight = feature_weight
        self.replaced_student_parameters = _final_classifier_parameter_count(student)
        self.register_buffer(
            "teacher_classifier_weight", teacher_classifier.weight.detach().clone()
        )
        teacher_bias = (
            None if teacher_classifier.bias is None else teacher_classifier.bias.detach().clone()
        )
        self.register_buffer("teacher_classifier_bias", teacher_bias)

    def _frozen_classifier_logits(self, projected_map: torch.Tensor) -> torch.Tensor:
        pooled = torch.flatten(F.adaptive_avg_pool2d(projected_map, 1), 1)
        return F.linear(
            pooled,
            self.teacher_classifier_weight,
            self.teacher_classifier_bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        student_features = extract_vision_features(self.student, inputs)
        # All registered current CIFAR pairs have equal final-map resolutions;
        # source SimKD consequently applies the transfer directly at evaluation.
        return self._frozen_classifier_logits(
            self.projector.project(simkd_feature_map(self.student, student_features))
        )

    def training_objective(
        self,
        teacher: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        criterion: nn.Module,
    ) -> DistillationObjective:
        student_features = extract_vision_features(self.student, inputs)
        teacher_features = _teacher_features(teacher, inputs)
        projected, target = self.projector.transfer_maps(
            simkd_feature_map(self.student, student_features),
            simkd_feature_map(teacher, teacher_features),
        )
        logits = self._frozen_classifier_logits(projected)
        ce_loss = criterion(logits, labels)
        feature_loss = self.feature_weight * simkd_loss(projected, target)
        # Official SimKD uses cls=0, div=0, beta=1: its optimization objective
        # is feature MSE, while CE is retained here for transparent logging.
        return DistillationObjective(logits, ce_loss, feature_loss, feature_loss)

    def parameter_counts(self) -> DistillerParameterCounts:
        frozen_classifier_parameters = self.teacher_classifier_weight.numel()
        if self.teacher_classifier_bias is not None:
            frozen_classifier_parameters += self.teacher_classifier_bias.numel()
        return _parameter_counts(
            self.student,
            replaced_student_parameters=self.replaced_student_parameters,
            deployment_auxiliary_parameters=(
                _count_parameters(self.projector) + frozen_classifier_parameters
            ),
            training_only_auxiliary_parameters=0,
            optimization_module=self,
        )


class ReviewKDDistiller(nn.Module):
    """ReviewKD wrapper: task CE plus warmed-up multi-scale feature review."""

    def __init__(
        self,
        student: nn.Module,
        *,
        student_channels: Sequence[int],
        teacher_channels: Sequence[int],
        weight: float,
        warmup_epochs: int = 20,
        ce_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.student = student
        self.adapter = ReviewKDAdapter(student_channels, teacher_channels)
        self.weight = weight
        self.warmup_epochs = warmup_epochs
        self.ce_weight = ce_weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.student(inputs)

    def training_objective(
        self,
        teacher: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        criterion: nn.Module,
    ) -> DistillationObjective:
        student_features = extract_vision_features(self.student, inputs)
        teacher_maps = extract_review_teacher_maps(teacher, inputs)
        reviewed_student_maps = self.adapter(review_feature_maps(student_features), teacher_maps)
        ce_loss = self.ce_weight * criterion(student_features.logits, labels)
        feature_loss = review_kd_loss(
            reviewed_student_maps,
            teacher_maps,
            epoch=epoch,
            weight=self.weight,
            warmup_epochs=self.warmup_epochs,
        )
        return DistillationObjective(
            student_features.logits,
            ce_loss,
            feature_loss,
            ce_loss + feature_loss,
        )

    def parameter_counts(self) -> DistillerParameterCounts:
        return _parameter_counts(
            self.student,
            replaced_student_parameters=0,
            deployment_auxiliary_parameters=0,
            training_only_auxiliary_parameters=_count_parameters(self.adapter),
            optimization_module=self,
        )
