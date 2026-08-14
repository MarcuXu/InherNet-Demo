"""Source-faithful building blocks for contrastive representation distillation.

This module implements CRD's exact-instance positive, different-class negative
sampling and its two-memory-bank noise-contrastive objective.  It contains no
experiment or optimizer policy: callers wrap the current training dataset,
train the two projection heads alongside the student, and keep the memory banks
as registered buffers.

Reference: https://github.com/HobbitLong/RepDistiller
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from vision_distillation import extract_vision_features


CRD_EMBEDDING_DIM = 128
CRD_NUM_NEGATIVES = 16_384
CRD_TEMPERATURE = 0.07
CRD_MEMORY_MOMENTUM = 0.5
CRD_LOSS_WEIGHT = 0.8
_NCE_EPSILON = 1e-7


def dataset_targets(dataset: Dataset) -> torch.Tensor:
    """Return labels in the dataset's current, possibly-subset, index space."""

    if isinstance(dataset, Subset):
        parent_targets = dataset_targets(dataset.dataset)
        return parent_targets.index_select(
            0, torch.as_tensor(dataset.indices, dtype=torch.long)
        )
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    if hasattr(dataset, "_labels"):
        return torch.as_tensor(dataset._labels, dtype=torch.long)
    raise TypeError("The contrastive dataset requires a dataset with targets or _labels.")


class ExactClassContrastiveDataset(Dataset):
    """Add local instance indices and exact-class CRD samples to a dataset.

    Each item is ``(input, target, index, contrast_indices)``.  The first
    contrast index is the anchor itself; all remaining indices belong to other
    classes.  For a ``Subset``, indices are local to that subset and therefore
    directly address a memory bank of size ``len(dataset)``.

    Negative sampling uses a private generator, so it does not consume random
    numbers from DataLoader shuffling or stochastic transforms.  Call
    :meth:`set_epoch` before creating each epoch's DataLoader iterator to obtain
    deterministic, epoch-varying negatives.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_negatives: int = CRD_NUM_NEGATIVES,
        seed: int = 0,
        targets: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        self.dataset = dataset
        self.num_negatives = num_negatives
        self.seed = seed
        self.epoch = 0
        self.targets = (
            dataset_targets(dataset)
            if targets is None
            else torch.as_tensor(targets, dtype=torch.long)
        )
        if self.targets.numel() != len(dataset):
            raise ValueError("targets must contain one label per wrapped example.")
        self._negative_indices = {
            int(class_id): torch.nonzero(self.targets != class_id, as_tuple=False).flatten()
            for class_id in torch.unique(self.targets, sorted=True).tolist()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def contrast_indices(self, index: int) -> torch.Tensor:
        negative_pool = self._negative_indices[int(self.targets[index])]
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * len(self.dataset) + index
        )
        if self.num_negatives <= negative_pool.numel():
            positions = torch.randperm(negative_pool.numel(), generator=generator)[
                : self.num_negatives
            ]
        else:
            positions = torch.randint(
                negative_pool.numel(),
                (self.num_negatives,),
                generator=generator,
            )
        negatives = negative_pool.index_select(0, positions)
        return torch.cat((torch.tensor([index], dtype=torch.long), negatives))

    def __getitem__(self, index: int):
        inputs, target = self.dataset[index]
        return inputs, target, index, self.contrast_indices(index)


def build_crd_train_loader(
    train_loader: DataLoader,
    *,
    num_negatives: int = CRD_NUM_NEGATIVES,
    seed: int,
) -> DataLoader:
    """Rebuild a shuffled train loader with CRD instance samples.

    The wrapped dataset uses local memory-bank indices, including when the
    original loader contains a training ``Subset``.  Batching, collation,
    worker, and pinned-memory policy are retained; the explicit generator seed
    makes the rebuilt shuffle independent and reproducible.
    """

    wrapped_dataset = ExactClassContrastiveDataset(
        train_loader.dataset,
        num_negatives=num_negatives,
        seed=seed,
    )
    loader_options = {
        "dataset": wrapped_dataset,
        "batch_size": train_loader.batch_size,
        "shuffle": True,
        "num_workers": train_loader.num_workers,
        "collate_fn": train_loader.collate_fn,
        "pin_memory": train_loader.pin_memory,
        "drop_last": train_loader.drop_last,
        "timeout": train_loader.timeout,
        "worker_init_fn": train_loader.worker_init_fn,
        "generator": torch.Generator().manual_seed(seed),
    }
    if train_loader.num_workers > 0:
        loader_options.update(
            prefetch_factor=train_loader.prefetch_factor,
            persistent_workers=train_loader.persistent_workers,
            multiprocessing_context=train_loader.multiprocessing_context,
        )
    return DataLoader(**loader_options)


class CRDEmbedding(nn.Module):
    """Linear projection into CRD's normalized 128-dimensional space."""

    def __init__(self, input_dim: int, embedding_dim: int = CRD_EMBEDDING_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.linear(features.reshape(features.shape[0], -1))
        norm = projected.square().sum(dim=1, keepdim=True).sqrt()
        return projected / norm


class ContrastMemory(nn.Module):
    """CRD's student and teacher memory banks and partition estimates."""

    def __init__(
        self,
        embedding_dim: int,
        num_samples: int,
        *,
        num_negatives: int = CRD_NUM_NEGATIVES,
        temperature: float = CRD_TEMPERATURE,
        momentum: float = CRD_MEMORY_MOMENTUM,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.momentum = momentum

        generator = None if seed is None else torch.Generator().manual_seed(seed)
        bound = 1.0 / (embedding_dim / 3.0) ** 0.5
        student_memory = torch.empty(num_samples, embedding_dim)
        teacher_memory = torch.empty(num_samples, embedding_dim)
        student_memory.uniform_(-bound, bound, generator=generator)
        teacher_memory.uniform_(-bound, bound, generator=generator)
        self.register_buffer("student_memory", student_memory)
        self.register_buffer("teacher_memory", teacher_memory)
        self.register_buffer("student_partition", torch.tensor(-1.0))
        self.register_buffer("teacher_partition", torch.tensor(-1.0))

    def forward(
        self,
        student_embedding: torch.Tensor,
        teacher_embedding: torch.Tensor,
        sample_indices: torch.Tensor,
        contrast_indices: torch.Tensor,
        *,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, embedding_dim = student_embedding.shape
        flat_indices = contrast_indices.reshape(-1)

        teacher_candidates = self.teacher_memory.index_select(0, flat_indices).detach()
        teacher_candidates = teacher_candidates.reshape(
            batch_size, self.num_negatives + 1, embedding_dim
        )
        student_scores = torch.exp(
            torch.bmm(teacher_candidates, student_embedding.unsqueeze(2)).squeeze(2)
            / self.temperature
        )

        student_candidates = self.student_memory.index_select(0, flat_indices).detach()
        student_candidates = student_candidates.reshape(
            batch_size, self.num_negatives + 1, embedding_dim
        )
        teacher_scores = torch.exp(
            torch.bmm(student_candidates, teacher_embedding.unsqueeze(2)).squeeze(2)
            / self.temperature
        )

        if self.student_partition.item() < 0:
            self.student_partition.copy_(student_scores.detach().mean() * self.num_samples)
        if self.teacher_partition.item() < 0:
            self.teacher_partition.copy_(teacher_scores.detach().mean() * self.num_samples)
        student_scores = student_scores / self.student_partition
        teacher_scores = teacher_scores / self.teacher_partition

        if update_memory:
            self._update_memory(sample_indices, student_embedding, teacher_embedding)
        return student_scores.contiguous(), teacher_scores.contiguous()

    @torch.no_grad()
    def _update_memory(
        self,
        sample_indices: torch.Tensor,
        student_embedding: torch.Tensor,
        teacher_embedding: torch.Tensor,
    ) -> None:
        previous_student = self.student_memory.index_select(0, sample_indices)
        updated_student = (
            previous_student * self.momentum
            + student_embedding.detach() * (1.0 - self.momentum)
        )
        updated_student = updated_student / updated_student.square().sum(
            dim=1, keepdim=True
        ).sqrt()
        self.student_memory.index_copy_(0, sample_indices, updated_student)

        previous_teacher = self.teacher_memory.index_select(0, sample_indices)
        updated_teacher = (
            previous_teacher * self.momentum
            + teacher_embedding.detach() * (1.0 - self.momentum)
        )
        updated_teacher = updated_teacher / updated_teacher.square().sum(
            dim=1, keepdim=True
        ).sqrt()
        self.teacher_memory.index_copy_(0, sample_indices, updated_teacher)


def contrastive_nce_loss(probabilities: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Noise-contrastive loss from Eq. 18 of the CRD source implementation."""

    batch_size = probabilities.shape[0]
    num_negatives = probabilities.shape[1] - 1
    noise_mass = num_negatives / float(num_samples)
    positive = probabilities[:, 0]
    negative = probabilities[:, 1:]
    positive_log_probability = torch.log(
        positive / (positive + noise_mass + _NCE_EPSILON)
    )
    negative_log_probability = torch.log(
        noise_mass / (negative + noise_mass + _NCE_EPSILON)
    )
    return -(
        positive_log_probability.sum() + negative_log_probability.sum()
    ) / batch_size


class CRDLoss(nn.Module):
    """Symmetric CRD loss with trainable projections and persistent memories.

    ``parameters()`` exposes exactly the student and teacher projection heads
    for the optimizer.  ``memory`` exposes both memory banks and their learned
    partition constants as registered buffers.  Teacher features are detached
    at this boundary, while the teacher-side projection head remains trainable
    as in the source implementation.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_samples: int,
        *,
        embedding_dim: int = CRD_EMBEDDING_DIM,
        num_negatives: int = CRD_NUM_NEGATIVES,
        temperature: float = CRD_TEMPERATURE,
        momentum: float = CRD_MEMORY_MOMENTUM,
        memory_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.student_embedding = CRDEmbedding(student_dim, embedding_dim)
        self.teacher_embedding = CRDEmbedding(teacher_dim, embedding_dim)
        self.memory = ContrastMemory(
            embedding_dim,
            num_samples,
            num_negatives=num_negatives,
            temperature=temperature,
            momentum=momentum,
            seed=memory_seed,
        )
        self.num_samples = num_samples

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        sample_indices: torch.Tensor,
        contrast_indices: torch.Tensor,
        *,
        update_memory: bool = True,
    ) -> torch.Tensor:
        student_embedding = self.student_embedding(student_features)
        teacher_embedding = self.teacher_embedding(teacher_features.detach())
        student_probabilities, teacher_probabilities = self.memory(
            student_embedding,
            teacher_embedding,
            sample_indices,
            contrast_indices,
            update_memory=update_memory,
        )
        return contrastive_nce_loss(
            student_probabilities, self.num_samples
        ) + contrastive_nce_loss(teacher_probabilities, self.num_samples)


@dataclass(frozen=True)
class CRDTrainingOutput:
    logits: torch.Tensor
    classification: torch.Tensor
    contrastive: torch.Tensor
    total: torch.Tensor


class CRDDistiller(nn.Module):
    """Training wrapper that deploys as its ordinary student classifier.

    The student and both CRD projection heads are optimized during training.
    The teacher remains external, frozen, and absent from this module's state.
    CRD projections and memory banks are train-time auxiliaries and are not
    part of the deployed model returned through ``student``.
    """

    def __init__(
        self,
        student: nn.Module,
        *,
        student_dim: int,
        teacher_dim: int,
        num_samples: int,
        embedding_dim: int = CRD_EMBEDDING_DIM,
        num_negatives: int = CRD_NUM_NEGATIVES,
        temperature: float = CRD_TEMPERATURE,
        momentum: float = CRD_MEMORY_MOMENTUM,
        memory_seed: int | None = None,
        ce_weight: float = 1.0,
        contrastive_weight: float = CRD_LOSS_WEIGHT,
    ) -> None:
        super().__init__()
        self.student = student
        self.crd_loss = CRDLoss(
            student_dim,
            teacher_dim,
            num_samples,
            embedding_dim=embedding_dim,
            num_negatives=num_negatives,
            temperature=temperature,
            momentum=momentum,
            memory_seed=memory_seed,
        )
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight

    @property
    def deployment_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.student.parameters())

    @property
    def train_only_auxiliary_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.crd_loss.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.student(inputs)

    def training_objective(
        self,
        teacher: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        sample_indices: torch.Tensor,
        contrast_indices: torch.Tensor,
        epoch: int,
        criterion: nn.Module,
    ) -> CRDTrainingOutput:
        """Return logits and the published CE + 0.8 CRD training terms.

        ``epoch`` is part of the shared distiller training interface.  CRD has
        no epoch-dependent warm-up, so it does not alter this source objective.
        """

        _ = epoch
        student_features = extract_vision_features(self.student, inputs)
        teacher.eval()
        with torch.no_grad():
            teacher_features = extract_vision_features(teacher, inputs)
        contrastive = self.crd_loss(
            student_features.pooled,
            teacher_features.pooled,
            sample_indices,
            contrast_indices,
        )
        classification = self.ce_weight * criterion(student_features.logits, labels)
        total = classification + self.contrastive_weight * contrastive
        return CRDTrainingOutput(
            student_features.logits,
            classification,
            contrastive,
            total,
        )


@dataclass(frozen=True)
class CRDObjectiveTerms:
    classification: torch.Tensor
    contrastive: torch.Tensor
    total: torch.Tensor


def crd_objective(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    contrastive_loss: torch.Tensor,
) -> CRDObjectiveTerms:
    """Return the published same-family CIFAR objective: CE + 0.8 CRD."""

    classification = F.cross_entropy(student_logits, targets)
    total = classification + CRD_LOSS_WEIGHT * contrastive_loss
    return CRDObjectiveTerms(classification, contrastive_loss, total)
