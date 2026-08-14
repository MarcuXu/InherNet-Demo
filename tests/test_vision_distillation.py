from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cifar10_models import build_model as build_cifar10_model
from cifar100_models import resnet8, resnet20, vgg8, wrn_16_2
from vision_distillation import (
    CATKDDistiller,
    ReviewKDDistiller,
    ReviewKDAdapter,
    SimKDDistiller,
    SimKDProjector,
    VisionFeatures,
    cat_kd_loss,
    classifier_weight_cams,
    extract_review_teacher_maps,
    extract_vision_features,
    review_hcl_loss,
    review_kd_loss,
    review_feature_maps,
    simkd_feature_map,
    simkd_loss,
)


class FeatureExtractionTests(unittest.TestCase):
    def _assert_native_logits(self, model: nn.Module, *, channels: int, spatial_size: int) -> None:
        model.eval()
        inputs = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            native_logits = model(inputs)
            extracted = extract_vision_features(model, inputs)
        torch.testing.assert_close(extracted.logits, native_logits, rtol=0.0, atol=0.0)
        self.assertEqual(extracted.final_map.shape, (1, channels, spatial_size, spatial_size))
        self.assertEqual(extracted.pooled.shape, (1, channels))

    def test_current_cifar100_model_family_extraction_matches_native_forward(self) -> None:
        self._assert_native_logits(resnet20(num_classes=7), channels=64, spatial_size=8)
        self._assert_native_logits(vgg8(num_classes=7), channels=512, spatial_size=2)
        wide_resnet = wrn_16_2(num_classes=7).eval()
        self._assert_native_logits(wide_resnet, channels=128, spatial_size=8)
        with torch.no_grad():
            review_maps = review_feature_maps(
                extract_vision_features(wide_resnet, torch.randn(1, 3, 32, 32))
            )
        self.assertEqual([feature.shape[1] for feature in review_maps], [32, 64, 128, 128])

    def test_wide_resnet_uses_source_raw_block3_taps_for_reviewkd_and_simkd(self) -> None:
        model = wrn_16_2(num_classes=7).eval()
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            features = extract_vision_features(model, inputs)
            stage1 = model.block1(model.conv1(inputs))
            stage2 = model.block2(stage1)
            raw_stage3 = model.block3(stage2)
            native_final_map = F.relu(model.bn(raw_stage3), inplace=False)

        torch.testing.assert_close(features.stage_maps[-1], raw_stage3)
        torch.testing.assert_close(review_feature_maps(features)[2], raw_stage3)
        torch.testing.assert_close(simkd_feature_map(model, features), raw_stage3)
        torch.testing.assert_close(features.final_map, native_final_map)

    def test_current_cifar10_torchvision_resnets_match_native_forward(self) -> None:
        self._assert_native_logits(
            build_cifar10_model("resnet18", num_classes=7), channels=512, spatial_size=4
        )
        self._assert_native_logits(
            build_cifar10_model("resnet50", num_classes=7), channels=2048, spatial_size=4
        )


def _features(final_map: torch.Tensor, classifier: nn.Linear) -> VisionFeatures:
    batch_size = final_map.shape[0]
    return VisionFeatures(
        logits=final_map.new_zeros((batch_size, classifier.out_features)),
        stage_maps=(),
        final_map=final_map,
        pooled=final_map.new_zeros((batch_size, classifier.in_features)),
        classifier=classifier,
    )


class CATKDTests(unittest.TestCase):
    def test_classifier_weight_cams_and_pair_scaled_loss_match_the_source_formula(self) -> None:
        student_map = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 1.0], [0.0, 2.0]]]],
            requires_grad=True,
        )
        teacher_map = torch.tensor(
            [[[[0.5, 1.5], [2.5, 3.5]], [[1.0, -1.0], [2.0, 0.0]]]],
            requires_grad=True,
        )
        student_classifier = nn.Linear(2, 3, bias=True)
        teacher_classifier = nn.Linear(2, 3, bias=True)
        with torch.no_grad():
            student_classifier.weight.copy_(
                torch.tensor([[1.0, 2.0], [-1.0, 3.0], [0.5, -0.5]])
            )
            teacher_classifier.weight.copy_(
                torch.tensor([[2.0, -1.0], [1.5, 0.5], [-2.0, 1.0]])
            )
            student_classifier.bias.fill_(100.0)
            teacher_classifier.bias.fill_(-100.0)

        student = _features(student_map, student_classifier)
        teacher = _features(teacher_map, teacher_classifier)
        cams = classifier_weight_cams(student_map, student_classifier)
        expected_cams = F.conv2d(student_map, student_classifier.weight[:, :, None, None])
        torch.testing.assert_close(cams, expected_cams)

        beta = 2.5
        loss = cat_kd_loss(student, teacher, beta=beta)
        expected = beta * F.mse_loss(
            F.adaptive_avg_pool2d(expected_cams, (2, 2)),
            F.adaptive_avg_pool2d(
                F.conv2d(teacher_map.detach(), teacher_classifier.weight.detach()[:, :, None, None]),
                (2, 2),
            ),
        )
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(student_map.grad)
        self.assertIsNotNone(student_classifier.weight.grad)
        self.assertIsNone(teacher_map.grad)
        self.assertIsNone(teacher_classifier.weight.grad)
        self.assertIsNone(teacher_classifier.bias.grad)


class SimKDTests(unittest.TestCase):
    def test_projector_matches_reused_frozen_teacher_classifier_contract(self) -> None:
        torch.manual_seed(4)
        projector = SimKDProjector(student_channels=4, teacher_channels=8)
        student_map = torch.randn(2, 4, 4, 4, requires_grad=True)
        teacher_map = torch.randn(2, 8, 2, 2, requires_grad=True)
        teacher_classifier = nn.Linear(8, 3)

        projected, target, logits = projector(student_map, teacher_map, teacher_classifier)
        self.assertEqual(projected.shape, (2, 8, 2, 2))
        self.assertEqual(target.shape, (2, 8, 2, 2))
        self.assertEqual(logits.shape, (2, 3))
        self.assertIsNone(projector.transfer[0].bias)
        self.assertNotIn(id(teacher_classifier.weight), {id(parameter) for parameter in projector.parameters()})
        expected_logits = F.linear(
            torch.flatten(F.adaptive_avg_pool2d(projected, 1), 1),
            teacher_classifier.weight.detach(),
            teacher_classifier.bias.detach(),
        )
        torch.testing.assert_close(logits, expected_logits)

        loss = simkd_loss(projected, target) + logits.sum()
        loss.backward()
        self.assertIsNotNone(student_map.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in projector.parameters()))
        self.assertIsNone(teacher_map.grad)
        self.assertIsNone(teacher_classifier.weight.grad)
        self.assertIsNone(teacher_classifier.bias.grad)

    def test_projector_aligns_the_teacher_when_the_student_map_is_smaller(self) -> None:
        projector = SimKDProjector(student_channels=4, teacher_channels=8)
        student_map = torch.randn(2, 4, 2, 2)
        teacher_map = torch.randn(2, 8, 4, 4)
        teacher_classifier = nn.Linear(8, 3)
        projected, target, _ = projector(student_map, teacher_map, teacher_classifier)
        self.assertEqual(projected.shape[-2:], (2, 2))
        self.assertEqual(target.shape[-2:], (2, 2))

    def test_wide_resnet_wrapper_uses_the_released_pre_bn_block3_feature(self) -> None:
        student = wrn_16_2(num_classes=3).eval()
        teacher = wrn_16_2(num_classes=3).eval()
        distiller = SimKDDistiller(
            student,
            student_channels=128,
            teacher_channels=128,
            teacher_classifier=teacher.fc,
        ).eval()
        inputs = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])

        with torch.no_grad():
            student_features = extract_vision_features(student, inputs)
            teacher_features = extract_vision_features(teacher, inputs)
            source_student_map = simkd_feature_map(student, student_features)
            source_teacher_map = simkd_feature_map(teacher, teacher_features)
            expected_logits = distiller._frozen_classifier_logits(
                distiller.projector.project(source_student_map)
            )
            expected_projected, expected_target = distiller.projector.transfer_maps(
                source_student_map, source_teacher_map
            )

        torch.testing.assert_close(distiller(inputs), expected_logits)
        objective = distiller.training_objective(
            teacher, inputs, labels, epoch=0, criterion=nn.CrossEntropyLoss()
        )
        torch.testing.assert_close(objective.logits, expected_logits)
        torch.testing.assert_close(
            objective.feature_loss,
            simkd_loss(expected_projected, expected_target),
        )


class ReviewKDTests(unittest.TestCase):
    def test_wide_resnet_wrapper_passes_raw_block3_student_map_to_review_adapter(self) -> None:
        student = wrn_16_2(num_classes=3).eval()
        teacher = wrn_16_2(num_classes=3).eval()
        distiller = ReviewKDDistiller(
            student,
            student_channels=[32, 64, 128, 128],
            teacher_channels=[32, 64, 128, 128],
            weight=0.6,
        ).eval()
        inputs = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        captured_student_maps: list[torch.Tensor] = []

        def capture_student_maps(_module: nn.Module, arguments: tuple[object, ...]) -> None:
            student_maps = arguments[0]
            assert isinstance(student_maps, tuple)
            captured_student_maps.append(student_maps[2].detach().clone())

        hook = distiller.adapter.register_forward_pre_hook(capture_student_maps)
        try:
            distiller.training_objective(
                teacher, inputs, labels, epoch=0, criterion=nn.CrossEntropyLoss()
            )
        finally:
            hook.remove()

        with torch.no_grad():
            stage1 = student.block1(student.conv1(inputs))
            stage2 = student.block2(stage1)
            raw_stage3 = student.block3(stage2)
        self.assertEqual(len(captured_student_maps), 1)
        torch.testing.assert_close(captured_student_maps[0], raw_stage3)

    def test_teacher_targets_use_source_preactivation_maps(self) -> None:
        teacher = resnet20(num_classes=3).eval()
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            preactivation_maps = extract_review_teacher_maps(teacher, inputs)
            postactivation_maps = review_feature_maps(
                extract_vision_features(teacher, inputs)
            )
        self.assertEqual(
            [feature.shape for feature in preactivation_maps],
            [feature.shape for feature in postactivation_maps],
        )
        self.assertTrue(any(bool((feature < 0).any()) for feature in preactivation_maps[:-1]))
        self.assertTrue(all(bool((feature >= 0).all()) for feature in postactivation_maps[:-1]))

    def test_hcl_and_warmup_match_the_published_weighting(self) -> None:
        student_features = (
            torch.zeros(1, 1, 4, 4, requires_grad=True),
            torch.zeros(1, 1, 2, 2, requires_grad=True),
        )
        teacher_features = (
            torch.ones(1, 1, 4, 4, requires_grad=True),
            torch.ones(1, 1, 2, 2, requires_grad=True),
        )
        raw_loss = review_hcl_loss(student_features, teacher_features)
        torch.testing.assert_close(raw_loss, torch.tensor(2.0))
        torch.testing.assert_close(
            review_kd_loss(student_features, teacher_features, epoch=0, weight=0.6),
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            review_kd_loss(student_features, teacher_features, epoch=10, weight=0.6),
            torch.tensor(0.6),
        )
        torch.testing.assert_close(
            review_kd_loss(student_features, teacher_features, epoch=20, weight=0.6),
            torch.tensor(1.2),
        )
        torch.testing.assert_close(
            review_kd_loss(
                student_features,
                teacher_features,
                epoch=5,
                weight=0.6,
                warmup_epochs=10,
            ),
            torch.tensor(0.6),
        )

    def test_abf_hierarchy_matches_teacher_shapes_and_keeps_teacher_external(self) -> None:
        torch.manual_seed(8)
        adapter = ReviewKDAdapter([4, 8, 16], [6, 12, 20])
        student_features = (
            torch.randn(2, 4, 8, 8, requires_grad=True),
            torch.randn(2, 8, 4, 4, requires_grad=True),
            torch.randn(2, 16, 2, 2, requires_grad=True),
        )
        teacher_features = (
            torch.randn(2, 6, 8, 8, requires_grad=True),
            torch.randn(2, 12, 4, 4, requires_grad=True),
            torch.randn(2, 20, 2, 2, requires_grad=True),
        )

        reviewed = adapter(student_features, teacher_features)
        self.assertEqual([feature.shape for feature in reviewed], [feature.shape for feature in teacher_features])
        self.assertIsNone(adapter.abfs[0].att_conv)
        self.assertIsNotNone(adapter.abfs[1].att_conv)

        loss = review_kd_loss(reviewed, teacher_features, epoch=20, weight=0.6)
        loss.backward()
        self.assertTrue(all(feature.grad is not None for feature in student_features))
        self.assertTrue(any(parameter.grad is not None for parameter in adapter.parameters()))
        self.assertTrue(all(feature.grad is None for feature in teacher_features))


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class DistillerWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)
        self.inputs = torch.randn(1, 3, 32, 32)
        self.labels = torch.tensor([0])
        self.criterion = nn.CrossEntropyLoss()

    def test_cat_wrapper_owns_only_student_and_optimizes_ce_plus_cam_loss(self) -> None:
        student = resnet8(num_classes=3)
        teacher = resnet8(num_classes=3)
        distiller = CATKDDistiller(student, beta=0.7)
        distiller.eval()

        objective = distiller.training_objective(
            teacher, self.inputs, self.labels, epoch=0, criterion=self.criterion
        )
        torch.testing.assert_close(objective.total_loss, objective.ce_loss + objective.feature_loss)
        torch.testing.assert_close(distiller(self.inputs), student(self.inputs))
        objective.total_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in student.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

        counts = distiller.parameter_counts()
        self.assertEqual(counts.student_parameters, _parameter_count(student))
        self.assertEqual(counts.replaced_student_parameters, 0)
        self.assertEqual(counts.deployment_auxiliary_parameters, 0)
        self.assertEqual(counts.training_only_auxiliary_parameters, 0)
        self.assertEqual(counts.deployment_parameters, _parameter_count(student))
        self.assertEqual(counts.optimization_parameters, _parameter_count(student))

    def test_simkd_wrapper_uses_buffered_classifier_and_restores_projector_state(self) -> None:
        student = resnet8(num_classes=3)
        teacher = resnet8(num_classes=3)
        distiller = SimKDDistiller(
            student,
            student_channels=64,
            teacher_channels=64,
            teacher_classifier=teacher.fc,
        )
        distiller.eval()

        expected_logits = distiller(self.inputs).detach().clone()
        objective = distiller.training_objective(
            teacher, self.inputs, self.labels, epoch=0, criterion=self.criterion
        )
        torch.testing.assert_close(objective.logits, expected_logits)
        torch.testing.assert_close(objective.total_loss, objective.feature_loss)
        self.assertGreater(float(objective.ce_loss), 0.0)
        self.assertIn("teacher_classifier_weight", dict(distiller.named_buffers()))
        self.assertIn("teacher_classifier_bias", dict(distiller.named_buffers()))
        self.assertNotIn("teacher_classifier_weight", dict(distiller.named_parameters()))

        objective.total_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in student.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in distiller.projector.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

        saved_state = {
            name: value.detach().clone() for name, value in distiller.state_dict().items()
        }
        with torch.no_grad():
            distiller.projector.transfer[0].weight.add_(1.0)
            distiller.teacher_classifier_weight.add_(1.0)
        self.assertFalse(torch.allclose(distiller(self.inputs), expected_logits))
        distiller.load_state_dict(saved_state)
        torch.testing.assert_close(distiller(self.inputs), expected_logits)

        counts = distiller.parameter_counts()
        classifier_count = teacher.fc.weight.numel() + teacher.fc.bias.numel()
        replaced_student_classifier_count = _parameter_count(student.fc)
        self.assertEqual(counts.student_parameters, _parameter_count(student))
        self.assertEqual(counts.replaced_student_parameters, replaced_student_classifier_count)
        self.assertEqual(
            counts.deployment_auxiliary_parameters,
            _parameter_count(distiller.projector) + classifier_count,
        )
        self.assertEqual(
            counts.deployment_parameters,
            _parameter_count(student)
            - replaced_student_classifier_count
            + _parameter_count(distiller.projector)
            + classifier_count,
        )
        self.assertEqual(counts.training_only_auxiliary_parameters, 0)
        self.assertEqual(
            counts.optimization_parameters,
            _parameter_count(student) + _parameter_count(distiller.projector),
        )

    def test_review_wrapper_uses_stage_maps_and_a_pooled_map_with_source_warmup(self) -> None:
        student = resnet8(num_classes=3)
        teacher = resnet8(num_classes=3)
        distiller = ReviewKDDistiller(
            student,
            student_channels=[16, 32, 64, 64],
            teacher_channels=[16, 32, 64, 64],
            weight=0.6,
        )
        distiller.eval()

        features = extract_vision_features(student, self.inputs)
        review_maps = review_feature_maps(features)
        self.assertEqual([feature.shape[1] for feature in review_maps], [16, 32, 64, 64])
        self.assertEqual(review_maps[-1].shape[-2:], (1, 1))

        at_start = distiller.training_objective(
            teacher, self.inputs, self.labels, epoch=0, criterion=self.criterion
        )
        torch.testing.assert_close(at_start.feature_loss, torch.zeros_like(at_start.feature_loss))
        torch.testing.assert_close(at_start.total_loss, at_start.ce_loss)
        at_full_weight = distiller.training_objective(
            teacher, self.inputs, self.labels, epoch=20, criterion=self.criterion
        )
        torch.testing.assert_close(
            at_full_weight.total_loss,
            at_full_weight.ce_loss + at_full_weight.feature_loss,
        )
        self.assertGreater(float(at_full_weight.feature_loss), 0.0)
        torch.testing.assert_close(distiller(self.inputs), student(self.inputs))

        at_full_weight.total_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in student.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in distiller.adapter.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

        counts = distiller.parameter_counts()
        self.assertEqual(counts.student_parameters, _parameter_count(student))
        self.assertEqual(counts.replaced_student_parameters, 0)
        self.assertEqual(counts.deployment_auxiliary_parameters, 0)
        self.assertEqual(counts.training_only_auxiliary_parameters, _parameter_count(distiller.adapter))
        self.assertEqual(counts.deployment_parameters, _parameter_count(student))
        self.assertEqual(
            counts.optimization_parameters,
            _parameter_count(student) + _parameter_count(distiller.adapter),
        )


if __name__ == "__main__":
    unittest.main()
