from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from demo_code import build_argparser, run_single_method_smoke_test
from contrastive_distillation import CRDDistiller, build_crd_train_loader
from cifar100_models import resnet8
from experiment_registry import (
    CAT_KD_REGISTRY,
    CRD_REGISTRY,
    REVIEW_KD_REGISTRY,
    SIM_KD_REGISTRY,
    build_method_tag,
    get_pair_spec,
    resolve_cat_kd_settings,
    resolve_crd_settings,
    resolve_review_kd_settings,
    resolve_sim_kd_settings,
    resolve_train_settings,
    DATASET_REGISTRY,
    TrainSettings,
)
from training_utils import RunLogger, train_vision_distillation
from vision_distillation import SimKDDistiller


PAPER_CIFAR100_PAIRS = {
    "resnet32x4_to_resnet8x4",
    "vgg13_to_vgg8",
    "wrn40_2_to_wrn40_1",
    "wrn40_2_to_wrn16_2",
    "resnet56_to_resnet20",
    "resnet110_to_resnet32",
    "resnet110_to_resnet20",
}


class ExtendedBaselineRegistryTests(unittest.TestCase):
    def test_source_scoped_registry_coverage_is_exact(self) -> None:
        cat_pairs = PAPER_CIFAR100_PAIRS - {"resnet110_to_resnet20"}
        review_pairs = {
            "resnet32x4_to_resnet8x4",
            "wrn40_2_to_wrn40_1",
            "wrn40_2_to_wrn16_2",
            "resnet56_to_resnet20",
            "resnet110_to_resnet32",
        }
        self.assertEqual(set(CAT_KD_REGISTRY), {("cifar100", pair) for pair in cat_pairs})
        self.assertEqual(
            set(SIM_KD_REGISTRY),
            {("cifar100", pair) for pair in PAPER_CIFAR100_PAIRS},
        )
        self.assertEqual(
            set(REVIEW_KD_REGISTRY),
            {("cifar100", pair) for pair in review_pairs},
        )
        self.assertEqual(
            set(CRD_REGISTRY),
            {("cifar100", pair) for pair in PAPER_CIFAR100_PAIRS},
        )

    def test_released_pair_coefficients_and_costs_are_registered(self) -> None:
        self.assertEqual(
            asdict(resolve_cat_kd_settings("cifar100", "resnet56_to_resnet20")),
            {
                "ce_weight": 1.0,
                "feature_weight": 0.7,
                "cam_resolution": 2,
                "source": "catkd_objective_config_adaptation_cifar100",
            },
        )
        self.assertEqual(
            resolve_sim_kd_settings("cifar100", "resnet56_to_resnet20").projector_factor,
            2,
        )
        review = resolve_review_kd_settings("cifar100", "resnet56_to_resnet20")
        self.assertEqual((review.feature_weight, review.warmup_epochs), (0.6, 20))
        crd = resolve_crd_settings("cifar100", "resnet56_to_resnet20")
        self.assertEqual(
            (crd.contrastive_weight, crd.embedding_dim, crd.num_negatives),
            (0.8, 128, 16_384),
        )
        with self.assertRaisesRegex(ValueError, "no released recipe"):
            resolve_cat_kd_settings("cifar100", "resnet110_to_resnet20")
        with self.assertRaisesRegex(ValueError, "no released recipe"):
            resolve_review_kd_settings("cifar100", "vgg13_to_vgg8")

    def test_method_tags_record_source_and_pair_recipe(self) -> None:
        parser = build_argparser()
        pair = "resnet56_to_resnet20"
        pair_spec = get_pair_spec("cifar100", pair)
        for method, token in (
            ("student_catkd", "catkd_objective_config_adaptation"),
            ("student_simkd", "official_simkd"),
            ("student_reviewkd", "official_reviewkd"),
            ("student_crd", "official_repdistiller"),
        ):
            args = parser.parse_args(
                ["--dataset", "cifar100", "--pair", pair, "--method", method]
            )
            settings = resolve_train_settings(DATASET_REGISTRY["cifar100"], args, pair_spec)
            self.assertIn(token, build_method_tag(method, args, pair_spec, settings))


class ExtendedBaselineSmokeTests(unittest.TestCase):
    def test_all_new_feature_baselines_execute_their_objective(self) -> None:
        parser = build_argparser()
        pair = "resnet56_to_resnet20"
        for method in ("student_catkd", "student_simkd", "student_reviewkd", "student_crd"):
            args = parser.parse_args(
                [
                    "--dataset",
                    "cifar100",
                    "--pair",
                    pair,
                    "--method",
                    method,
                    "--device",
                    "cpu",
                    "--smoke-test",
                ]
            )
            result = run_single_method_smoke_test("cifar100", pair, method, args)
            self.assertEqual(result["shape"], (2, 100))
            self.assertGreaterEqual(result["transfer_loss"], 0.0)

    def test_shared_training_loop_handles_simkd_selection_and_crd_batches(self) -> None:
        torch.manual_seed(5)
        dataset = TensorDataset(
            torch.randn(4, 3, 32, 32),
            torch.tensor([0, 1, 0, 1]),
        )
        dataset.targets = torch.tensor([0, 1, 0, 1])
        base_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        settings = TrainSettings(
            optimizer_name="sgd",
            batch_size=2,
            epochs=1,
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
            lr_milestones=(),
            scheduler_name="none",
        )
        logger = RunLogger(echo=False, store_info_to_file=False)

        sim_teacher = resnet8(num_classes=2).eval()
        simkd = SimKDDistiller(
            resnet8(num_classes=2),
            student_channels=64,
            teacher_channels=64,
            teacher_classifier=sim_teacher.fc,
        )
        sim_history = train_vision_distillation(
            sim_teacher,
            simkd,
            base_loader,
            base_loader,
            settings,
            torch.device("cpu"),
            logger=logger,
            num_labels=2,
            final_test_loader=base_loader,
        )
        self.assertEqual(len(sim_history["train_objective"]), 1)
        self.assertEqual(len(sim_history["final_test_accuracy"]), 1)

        crd_teacher = resnet8(num_classes=2).eval()
        crd_loader = build_crd_train_loader(base_loader, num_negatives=1, seed=5)
        crd = CRDDistiller(
            resnet8(num_classes=2),
            student_dim=64,
            teacher_dim=64,
            num_samples=4,
            num_negatives=1,
            memory_seed=5,
        )
        crd_history = train_vision_distillation(
            crd_teacher,
            crd,
            crd_loader,
            base_loader,
            settings,
            torch.device("cpu"),
            logger=logger,
            num_labels=2,
        )
        self.assertEqual(len(crd_history["train_objective"]), 1)
        self.assertTrue(all(parameter.grad is None for parameter in crd_teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
