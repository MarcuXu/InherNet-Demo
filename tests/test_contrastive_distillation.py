from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from contrastive_distillation import (
    CRDDistiller,
    CRDLoss,
    ContrastMemory,
    ExactClassContrastiveDataset,
    build_crd_train_loader,
    contrastive_nce_loss,
    crd_objective,
)
from cifar100_models import resnet8, resnet20


class ExactClassContrastiveDatasetTests(unittest.TestCase):
    def test_subset_indices_are_local_and_negatives_have_a_different_class(self) -> None:
        inputs = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        base = TensorDataset(inputs, labels)
        base.targets = labels
        subset = Subset(base, [0, 2, 3, 5])
        wrapped = ExactClassContrastiveDataset(subset, num_negatives=5, seed=11)

        item_inputs, target, memory_index, contrast_indices = wrapped[1]
        torch.testing.assert_close(item_inputs, inputs[2])
        self.assertEqual(int(target), 1)
        self.assertEqual(memory_index, 1)
        self.assertEqual(int(contrast_indices[0]), memory_index)
        self.assertTrue(torch.all(contrast_indices >= 0))
        self.assertTrue(torch.all(contrast_indices < len(subset)))
        negative_labels = wrapped.targets.index_select(0, contrast_indices[1:])
        self.assertTrue(torch.all(negative_labels != target))

    def test_source_replacement_rule_depends_on_negative_pool_size(self) -> None:
        inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        base = TensorDataset(inputs, labels)
        base.targets = labels

        without_replacement = ExactClassContrastiveDataset(
            base, num_negatives=5, seed=13
        ).contrast_indices(0)[1:]
        self.assertEqual(without_replacement.unique().numel(), 5)

        two_class_subset = Subset(base, [0, 1, 2])
        with_replacement = ExactClassContrastiveDataset(
            two_class_subset, num_negatives=3, seed=13
        ).contrast_indices(2)[1:]
        self.assertLess(with_replacement.unique().numel(), with_replacement.numel())

    def test_sampling_is_seeded_epoch_varying_and_does_not_consume_global_rng(self) -> None:
        inputs = torch.arange(60, dtype=torch.float32).reshape(20, 3)
        labels = torch.arange(20) % 4
        base = TensorDataset(inputs, labels)
        base.targets = labels
        first = ExactClassContrastiveDataset(base, num_negatives=8, seed=17)
        second = ExactClassContrastiveDataset(base, num_negatives=8, seed=17)

        torch.testing.assert_close(first.contrast_indices(7), second.contrast_indices(7))
        first.set_epoch(3)
        second.set_epoch(3)
        torch.testing.assert_close(first.contrast_indices(7), second.contrast_indices(7))
        epoch_three = first.contrast_indices(7)
        first.set_epoch(4)
        self.assertFalse(torch.equal(epoch_three, first.contrast_indices(7)))

        torch.manual_seed(123)
        expected_next = torch.rand(4)
        torch.manual_seed(123)
        first.contrast_indices(5)
        torch.testing.assert_close(torch.rand(4), expected_next)


class CRDTrainLoaderTests(unittest.TestCase):
    @staticmethod
    def _subset() -> Subset:
        inputs = torch.arange(48, dtype=torch.float32).reshape(16, 3)
        labels = torch.arange(16) % 4
        base = TensorDataset(inputs, labels)
        base.targets = labels
        return Subset(base, [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15])

    def test_rebuilt_loader_preserves_batch_and_worker_policy(self) -> None:
        original = DataLoader(
            self._subset(),
            batch_size=3,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=3,
            persistent_workers=True,
        )
        rebuilt = build_crd_train_loader(original, num_negatives=4, seed=29)
        self.assertIsInstance(rebuilt.dataset, ExactClassContrastiveDataset)
        self.assertEqual(rebuilt.batch_size, 3)
        self.assertEqual(rebuilt.num_workers, 2)
        self.assertTrue(rebuilt.pin_memory)
        self.assertTrue(rebuilt.drop_last)
        self.assertEqual(rebuilt.prefetch_factor, 3)
        self.assertTrue(rebuilt.persistent_workers)

    def test_subset_batches_and_contrast_samples_are_deterministic(self) -> None:
        first = build_crd_train_loader(
            DataLoader(self._subset(), batch_size=4, shuffle=True),
            num_negatives=5,
            seed=31,
        )
        second = build_crd_train_loader(
            DataLoader(self._subset(), batch_size=4, shuffle=True),
            num_negatives=5,
            seed=31,
        )
        first_batches = list(first)
        second_batches = list(second)
        self.assertEqual(len(first_batches), len(second_batches))
        for first_batch, second_batch in zip(first_batches, second_batches):
            for first_tensor, second_tensor in zip(first_batch, second_batch):
                torch.testing.assert_close(first_tensor, second_tensor)
            _, labels, sample_indices, contrast_indices = first_batch
            torch.testing.assert_close(contrast_indices[:, 0], sample_indices)
            negative_labels = first.dataset.targets[contrast_indices[:, 1:]]
            self.assertTrue(torch.all(negative_labels != labels.unsqueeze(1)))


class ContrastMemoryTests(unittest.TestCase):
    def test_scores_and_nce_loss_match_source_equations_on_tiny_tensors(self) -> None:
        memory = ContrastMemory(
            embedding_dim=2,
            num_samples=3,
            num_negatives=1,
            temperature=0.5,
            momentum=0.5,
        )
        with torch.no_grad():
            memory.student_memory.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
            )
            memory.teacher_memory.copy_(
                torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
            )
            memory.student_partition.fill_(2.0)
            memory.teacher_partition.fill_(4.0)

        student = torch.tensor([[1.0, 0.0]])
        teacher = torch.tensor([[0.0, 1.0]])
        contrast = torch.tensor([[0, 2]])
        student_probabilities, teacher_probabilities = memory(
            student,
            teacher,
            torch.tensor([0]),
            contrast,
            update_memory=False,
        )
        expected_student = torch.exp(
            memory.teacher_memory.index_select(0, contrast.flatten()) @ student.T / 0.5
        ).T / 2.0
        expected_teacher = torch.exp(
            memory.student_memory.index_select(0, contrast.flatten()) @ teacher.T / 0.5
        ).T / 4.0
        torch.testing.assert_close(student_probabilities, expected_student)
        torch.testing.assert_close(teacher_probabilities, expected_teacher)

        probabilities = torch.tensor([[0.25, 0.10, 0.05], [0.40, 0.20, 0.30]])
        noise_mass = 2.0 / 5.0
        expected_nce = -(
            torch.log(
                probabilities[:, 0]
                / (probabilities[:, 0] + noise_mass + 1e-7)
            ).sum()
            + torch.log(
                noise_mass / (probabilities[:, 1:] + noise_mass + 1e-7)
            ).sum()
        ) / probabilities.shape[0]
        torch.testing.assert_close(contrastive_nce_loss(probabilities, 5), expected_nce)

    def test_positive_rows_are_updated_by_source_momentum_rule(self) -> None:
        memory = ContrastMemory(
            embedding_dim=2,
            num_samples=4,
            num_negatives=1,
            temperature=0.07,
            momentum=0.5,
        )
        with torch.no_grad():
            memory.student_memory.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            )
            memory.teacher_memory.copy_(memory.student_memory)
        before_student = memory.student_memory.clone()
        before_teacher = memory.teacher_memory.clone()
        student = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 1.0]]), dim=1)
        teacher = F.normalize(torch.tensor([[1.0, 1.0], [-1.0, 1.0]]), dim=1)
        positive_indices = torch.tensor([0, 2])
        contrast = torch.tensor([[0, 1], [2, 3]])

        memory(student, teacher, positive_indices, contrast)
        expected_student = F.normalize(
            before_student.index_select(0, positive_indices) * 0.5 + student * 0.5,
            dim=1,
        )
        expected_teacher = F.normalize(
            before_teacher.index_select(0, positive_indices) * 0.5 + teacher * 0.5,
            dim=1,
        )
        torch.testing.assert_close(
            memory.student_memory.index_select(0, positive_indices), expected_student
        )
        torch.testing.assert_close(
            memory.teacher_memory.index_select(0, positive_indices), expected_teacher
        )
        torch.testing.assert_close(memory.student_memory[1], before_student[1])
        torch.testing.assert_close(memory.teacher_memory[3], before_teacher[3])


class CRDLossTests(unittest.TestCase):
    def test_forward_backward_is_finite_and_teacher_features_are_frozen(self) -> None:
        torch.manual_seed(4)
        criterion = CRDLoss(
            student_dim=3,
            teacher_dim=5,
            num_samples=6,
            embedding_dim=4,
            num_negatives=2,
            memory_seed=9,
        )
        student_features = torch.randn(2, 3, requires_grad=True)
        teacher_features = torch.randn(2, 5, requires_grad=True)
        logits = torch.randn(2, 3, requires_grad=True)
        targets = torch.tensor([0, 2])
        positive_indices = torch.tensor([0, 3])
        contrast_indices = torch.tensor([[0, 1, 2], [3, 4, 5]])

        contrastive = criterion(
            student_features,
            teacher_features,
            positive_indices,
            contrast_indices,
        )
        terms = crd_objective(logits, targets, contrastive)
        expected_total = F.cross_entropy(logits, targets) + 0.8 * contrastive
        torch.testing.assert_close(terms.total, expected_total)
        self.assertTrue(math.isfinite(float(terms.total.detach())))
        terms.total.backward()

        self.assertIsNotNone(student_features.grad)
        self.assertTrue(torch.isfinite(student_features.grad).all())
        self.assertIsNone(teacher_features.grad)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in criterion.parameters()
            )
        )
        state = criterion.state_dict()
        self.assertIn("memory.student_memory", state)
        self.assertIn("memory.teacher_memory", state)
        self.assertIn("student_embedding.linear.weight", state)
        self.assertIn("teacher_embedding.linear.weight", state)


class CRDDistillerTests(unittest.TestCase):
    def test_evaluation_is_the_native_student_and_counts_separate_auxiliaries(self) -> None:
        student = resnet8(num_classes=5)
        distiller = CRDDistiller(
            student,
            student_dim=64,
            teacher_dim=64,
            num_samples=6,
            embedding_dim=8,
            num_negatives=2,
            memory_seed=3,
        )
        inputs = torch.randn(2, 3, 32, 32)
        distiller.eval()
        with torch.no_grad():
            expected_logits = student(inputs)
            actual_logits = distiller(inputs)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)
        self.assertEqual(
            distiller.deployment_parameter_count,
            sum(parameter.numel() for parameter in student.parameters()),
        )
        self.assertEqual(
            distiller.train_only_auxiliary_parameter_count,
            sum(parameter.numel() for parameter in distiller.crd_loss.parameters()),
        )

    def test_training_objective_freezes_teacher_and_trains_student_and_heads(self) -> None:
        torch.manual_seed(12)
        student = resnet8(num_classes=4)
        teacher = resnet20(num_classes=4)
        distiller = CRDDistiller(
            student,
            student_dim=64,
            teacher_dim=64,
            num_samples=6,
            embedding_dim=8,
            num_negatives=2,
            memory_seed=5,
        )
        distiller.train()
        teacher.train()
        teacher_running_mean = teacher.bn1.running_mean.clone()
        inputs = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([1, 3])
        output = distiller.training_objective(
            teacher,
            inputs,
            labels,
            torch.tensor([0, 3]),
            torch.tensor([[0, 1, 2], [3, 4, 5]]),
            epoch=7,
            criterion=nn.CrossEntropyLoss(),
        )
        torch.testing.assert_close(
            output.total,
            output.classification + 0.8 * output.contrastive,
        )
        self.assertEqual(output.logits.shape, (2, 4))
        self.assertTrue(torch.isfinite(output.total))
        self.assertFalse(teacher.training)
        torch.testing.assert_close(teacher.bn1.running_mean, teacher_running_mean)

        output.total.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in student.parameters()))
        self.assertTrue(
            all(parameter.grad is not None for parameter in distiller.crd_loss.parameters())
        )
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
