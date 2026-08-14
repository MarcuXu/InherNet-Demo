from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.reuse_teacher_checkpoint import (
    checkpoint_matches_protocol,
    find_compatible_checkpoint,
    snapshot_checkpoint,
)


def checkpoint_payload(
    dataset: str,
    *,
    seed: int = 42,
    selection_protocol: bool = False,
) -> dict[str, object]:
    if dataset == "glue_sst2":
        split = {
            "profile": "huggingface_glue",
            "dataset": "nyu-mll/glue",
            "dataset_revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
            "task": "sst2",
            "train_split": "train_subset" if selection_protocol else "train",
            "evaluation_split": "train_holdout" if selection_protocol else "validation",
            "max_length": 128,
            "tokenizer": "bert-base-uncased",
            "tokenizer_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
            "teacher_revision": "387825ce42dbb39b87911cdf8e383ee3b25184f8",
        }
        if selection_protocol:
            split.update(
                {
                    "selection_split_seed": 2026,
                    "selection_validation_fraction": 0.1,
                }
            )
        return {
            "dataset": dataset,
            "pair": "bert4_to_bert2",
            "seed": seed,
            "architecture": "google/bert_uncased_L-4_H-256_A-4",
            "num_classes": 2,
            "model_profile": "glue_small_bert_teacher_inheritance",
            "data_profile": "standard_train_test",
            "selection_policy": (
                "best_train_holdout_accuracy"
                if selection_protocol
                else "best_validation_accuracy"
            ),
            "data_split": split,
        }
    if dataset == "cifar100":
        return {
            "dataset": dataset,
            "pair": "resnet56_to_resnet20",
            "seed": seed,
            "architecture": "resnet56",
            "num_classes": 100,
            "model_profile": "cifar100_teacher_inheritance",
            "data_profile": "standard_train_test",
            "selection_policy": "best_validation_accuracy",
            "data_split": {
                "profile": "fixed_stratified_holdout",
                "seed": 2026,
                "validation_fraction": 0.1,
            },
        }
    raise AssertionError(dataset)


class TeacherReuseTests(unittest.TestCase):
    def test_same_seed_cifar_registry_teacher_matches_selection_protocol(self) -> None:
        compatible, reason = checkpoint_matches_protocol(
            checkpoint_payload("cifar100"),
            dataset="cifar100",
            pair="resnet56_to_resnet20",
            seed=42,
            search_validation=True,
        )
        self.assertTrue(compatible, reason)

    def test_glue_registry_teacher_cannot_leak_into_selection_protocol(self) -> None:
        compatible, reason = checkpoint_matches_protocol(
            checkpoint_payload("glue_sst2", selection_protocol=False),
            dataset="glue_sst2",
            pair="bert4_to_bert2",
            seed=42,
            search_validation=True,
        )
        self.assertFalse(compatible)
        self.assertIn("selection_policy", reason)

    def test_glue_selection_teacher_cannot_replace_formal_teacher(self) -> None:
        compatible, reason = checkpoint_matches_protocol(
            checkpoint_payload("glue_sst2", selection_protocol=True),
            dataset="glue_sst2",
            pair="bert4_to_bert2",
            seed=42,
            search_validation=False,
        )
        self.assertFalse(compatible)
        self.assertIn("selection_policy", reason)

    def test_different_seed_is_never_reused(self) -> None:
        compatible, reason = checkpoint_matches_protocol(
            checkpoint_payload("cifar100", seed=123),
            dataset="cifar100",
            pair="resnet56_to_resnet20",
            seed=42,
            search_validation=True,
        )
        self.assertFalse(compatible)
        self.assertIn("seed=123", reason)

    def test_finder_skips_incompatible_candidate_and_snapshot_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "wrong.pt"
            source = root / "source.pt"
            destination = root / "selection" / "teacher.pt"
            torch.save(checkpoint_payload("cifar100", seed=123), wrong)
            torch.save(checkpoint_payload("cifar100"), source)
            with patch(
                "scripts.reuse_teacher_checkpoint._strict_validate_checkpoint"
            ) as strict_validate:
                selected = find_compatible_checkpoint(
                    (wrong, source),
                    dataset="cifar100",
                    pair="resnet56_to_resnet20",
                    seed=42,
                    search_validation=True,
                )
            self.assertEqual(selected, source.resolve())
            strict_validate.assert_called_once()
            snapshot_checkpoint(selected, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                snapshot_checkpoint(selected, destination)


if __name__ == "__main__":
    unittest.main()
