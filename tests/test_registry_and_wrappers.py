from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from demo_code import (
    build_argparser,
    semantic_split_metadata,
    teacher_training_split_metadata,
)
from checkpointing import (
    load_teacher_checkpoint,
    save_teacher_checkpoint,
    teacher_training_fingerprint,
)
from experiment_registry import (
    DATASET_REGISTRY,
    build_stratified_split_indices,
    build_method_tag,
    build_pair_model,
    get_dataset,
    get_dataloaders,
    get_pair_spec,
    get_role_name,
    resolve_compressed_train_mode,
    resolve_compress_linear,
    resolve_capacity_size,
    resolve_fixed_rank,
    resolve_train_settings,
    validate_args,
)
from model_wrappers import (
    GatedSVDLinear,
    GenericHeteroNet,
    GenericInherNet,
    freeze_gating_routers,
)
from scripts.rank_search import average_normalized_ranks
from training_utils import (
    RunLogger,
    build_optimizer,
    build_scheduler,
    count_parameters,
    evaluate_inheritance_diagnostics,
    evaluate_router_gradient_probe,
    train_distillation,
    train_supervised,
)


class RankOneConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[3.0]], [[4.0]]], [[[6.0]], [[8.0]]]]))
            self.conv.bias.copy_(torch.tensor([0.1, -0.2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyConvLinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, bias=True)
        self.fc = nn.Linear(18, 30, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(torch.flatten(x, 1))


class RaisingConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.conv(x)
        raise RuntimeError("injected calibration failure")


class FixedHeavyLinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(1000, 64)
        self.fc = nn.Linear(64, 64)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embedding(token_ids).mean(dim=1))


class RegistryTests(unittest.TestCase):
    def test_method_aware_capacity_defaults_and_hetero_custom_rank_rejection(self) -> None:
        parser = build_argparser()
        pair = get_pair_spec("cifar100", "resnet56_to_resnet20")
        hetero_args = parser.parse_args(
            ["--dataset", "cifar100", "--pair", "resnet56_to_resnet20", "--method", "hetero"]
        )
        inhernet_args = parser.parse_args(
            ["--dataset", "cifar100", "--pair", "resnet56_to_resnet20", "--method", "inhernet"]
        )
        self.assertEqual(resolve_capacity_size(hetero_args), "large")
        self.assertEqual(resolve_fixed_rank(hetero_args, pair), pair["rank_presets"]["large"])
        self.assertEqual(resolve_capacity_size(inhernet_args), "small")
        self.assertEqual(resolve_fixed_rank(inhernet_args, pair), pair["rank_presets"]["small"])

        custom_hetero = parser.parse_args(
            [
                "--dataset", "cifar100", "--pair", "resnet56_to_resnet20",
                "--method", "hetero", "--rank", "13",
            ]
        )
        with self.assertRaisesRegex(ValueError, "registered"):
            validate_args(custom_hetero, pair)
        custom_inhernet = parser.parse_args(
            [
                "--dataset", "cifar100", "--pair", "resnet56_to_resnet20",
                "--method", "inhernet", "--rank", "13",
            ]
        )
        validate_args(custom_inhernet, pair)

    def test_research_rank_policies_are_diagnostics_only(self) -> None:
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--dataset", "oxford_pets",
                "--pair", "resnet34_to_resnet18",
                "--method", "hetero",
                "--hetero-allocation-scale", "research_nested_relative",
            ]
        )
        with self.assertRaisesRegex(ValueError, "initialization-only"):
            validate_args(args, get_pair_spec(args.dataset, args.pair))
        args.inheritance_diagnostics_only = True
        validate_args(args, get_pair_spec(args.dataset, args.pair))

    def test_oxford_transform_override_is_supplied_at_construction(self) -> None:
        marker = object()

        class CapturingOxfordDataset:
            def __init__(self, *, transform, **kwargs) -> None:
                self.transform = transform
                self.transforms = transform

        original = DATASET_REGISTRY["oxford_pets"]
        replacement = replace(original, dataset_class=CapturingOxfordDataset)
        with patch.dict(DATASET_REGISTRY, {"oxford_pets": replacement}):
            dataset = get_dataset(
                "oxford_pets",
                root="unused",
                train=True,
                download=False,
                transform_override=marker,
            )
        self.assertIs(dataset.transform, marker)
        self.assertIs(dataset.transforms, marker)

    def test_semantic_split_metadata_drops_legacy_integrity_fields(self) -> None:
        split = semantic_split_metadata(
            {
                "profile": "fixed_stratified_holdout",
                "seed": 2026,
                "validation_indices_sha256": "legacy",
                "train_fingerprint": "legacy",
            }
        )
        self.assertEqual(split, {"profile": "fixed_stratified_holdout", "seed": 2026})

    def test_teacher_split_comparison_ignores_calibration_only_fields(self) -> None:
        split = teacher_training_split_metadata(
            {
                "profile": "fixed_stratified_holdout",
                "seed": 2026,
                "train_examples": 2944,
                "calibration_profile": "seeded_class_round_robin",
                "calibration_seed": 43,
                "official_evaluation_split": "validation",
                "selection_validation_fraction": None,
            }
        )
        self.assertEqual(
            split,
            {
                "profile": "fixed_stratified_holdout",
                "seed": 2026,
                "train_examples": 2944,
            },
        )

    def test_cifar100_inhernet_ranks_follow_printed_paper_table(self) -> None:
        expected = {
            "resnet32x4_to_resnet8x4": (4, 8),
            "vgg13_to_vgg8": (128, 256),
            "wrn40_2_to_wrn40_1": (16, 32),
            "wrn40_2_to_wrn16_2": (16, 32),
            "resnet56_to_resnet20": (8, 16),
            "resnet110_to_resnet32": (8, 32),
            "resnet110_to_resnet20": (4, 8),
        }
        for pair, ranks in expected.items():
            spec = get_pair_spec("cifar100", pair)
            self.assertEqual(
                (spec["rank_presets"]["small"], spec["rank_presets"]["large"]),
                ranks,
            )
            self.assertEqual(spec["inhernet_rank_source"], "paper_table_rank")

    def test_glue_adamw_excludes_bias_and_layer_norm_from_decay(self) -> None:
        model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
        settings = DATASET_REGISTRY["glue_sst2"].train_settings
        optimizer = build_optimizer(model, settings)
        self.assertEqual(len(optimizer.param_groups), 2)

        decayed = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
        not_decayed = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
        trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        self.assertFalse(decayed & not_decayed)
        self.assertEqual(decayed | not_decayed, trainable)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], settings.weight_decay)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)
        for name, parameter in model.named_parameters():
            if name.endswith(".bias") or name.startswith("1."):
                self.assertIn(id(parameter), not_decayed)

    def test_glue_linear_scheduler_warms_up_and_decays_per_step(self) -> None:
        model = nn.Linear(2, 2)
        settings = replace(
            DATASET_REGISTRY["glue_sst2"].train_settings,
            epochs=2,
            warmup_ratio=0.2,
        )
        optimizer = build_optimizer(model, settings)
        scheduler = build_scheduler(optimizer, settings, steps_per_epoch=5)
        self.assertIsNotNone(scheduler)
        self.assertEqual(scheduler.get_last_lr()[0], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], settings.lr * 0.5)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], settings.lr)
        for _ in range(8):
            optimizer.step()
            scheduler.step()
        self.assertEqual(scheduler.get_last_lr()[0], 0.0)

    def test_search_ranking_uses_average_ranks_for_ties(self) -> None:
        rows = [
            {"candidate": "a", "best_validation_metric": 90.0},
            {"candidate": "b", "best_validation_metric": 90.0},
            {"candidate": "c", "best_validation_metric": 80.0},
        ]
        scores = average_normalized_ranks(rows)
        self.assertEqual(scores, {"a": 0.75, "b": 0.75, "c": 0.0})

    def test_validation_training_restores_best_epoch_without_final_test(self) -> None:
        inputs = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.5, 0.5], [-0.5, -0.5]])
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
        base = nn.Linear(2, 2)
        one_epoch = copy.deepcopy(base)
        selected = copy.deepcopy(base)
        settings = replace(
            DATASET_REGISTRY["cifar100"].train_settings,
            epochs=1,
            lr=0.1,
            lr_milestones=(),
        )
        fake_metrics = {"accuracy": 100.0, "loss": 0.1}
        with patch("training_utils._finalize_test_metrics", return_value=fake_metrics):
            train_supervised(
                one_epoch,
                loader,
                loader,
                settings,
                torch.device("cpu"),
                logger=RunLogger(log_path="/dev/null", echo=False, store_info_to_file=False),
                num_labels=2,
            )
        metric_sequence = [
            {"accuracy": 100.0, "loss": 0.1},
            {"accuracy": 50.0, "loss": 0.2},
        ]
        with patch("training_utils._finalize_test_metrics", side_effect=metric_sequence):
            train_supervised(
                selected,
                loader,
                loader,
                replace(settings, epochs=2),
                torch.device("cpu"),
                logger=RunLogger(log_path="/dev/null", echo=False, store_info_to_file=False),
                num_labels=2,
                eval_split_name="validation",
                restore_best_state=True,
            )

        for name, tensor in selected.state_dict().items():
            self.assertTrue(torch.equal(tensor, one_epoch.state_dict()[name]))

    def test_distillation_never_updates_teacher(self) -> None:
        torch.manual_seed(5)
        teacher = nn.Sequential(nn.Linear(4, 2))
        student = nn.Sequential(nn.Linear(4, 2))
        teacher_state = copy.deepcopy(teacher.state_dict())
        student_state = copy.deepcopy(student.state_dict())
        inputs = torch.randn(12, 4)
        labels = torch.randint(0, 2, (12,))
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
        settings = replace(
            DATASET_REGISTRY["cifar100"].train_settings,
            epochs=1,
            lr=0.01,
            lr_milestones=(),
        )
        train_distillation(
            teacher,
            student,
            loader,
            loader,
            settings,
            torch.device("cpu"),
            logger=RunLogger(log_path="/dev/null", echo=False, store_info_to_file=False),
            num_labels=2,
        )
        for name, tensor in teacher.state_dict().items():
            self.assertTrue(torch.equal(tensor, teacher_state[name]))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertTrue(
            any(not torch.equal(tensor, student_state[name]) for name, tensor in student.state_dict().items())
        )

    def test_calibration_order_is_class_interleaved(self) -> None:
        labels = [class_id for class_id in range(37) for _ in range(100)]
        train_indices, validation_indices, calibration_indices = build_stratified_split_indices(
            labels,
            validation_fraction=0.2,
            seed=2026,
        )
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(set(train_indices) | set(validation_indices), set(range(len(labels))))
        prefix_labels = [labels[index] for index in calibration_indices[: 16 * 32]]
        counts = {class_id: prefix_labels.count(class_id) for class_id in range(37)}
        self.assertEqual(len([count for count in counts.values() if count > 0]), 37)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_holdout_calibration_seed_follows_the_split_seed(self) -> None:
        labels = torch.tensor([class_id for class_id in range(2) for _ in range(10)])
        dataset = TensorDataset(torch.randn(20, 1), labels)
        dataset.targets = labels.tolist()
        with patch("experiment_registry.get_dataset", return_value=dataset), patch(
            "experiment_registry.get_transforms", return_value=(object(), object())
        ):
            loaders = get_dataloaders(
                "cifar10",
                batch_size=4,
                root="unused",
                num_workers=0,
                download=False,
                validation_fraction=0.2,
                validation_split_seed=99,
            )
        self.assertEqual(loaders.split_metadata["seed"], 99)
        self.assertEqual(loaders.split_metadata["calibration_seed"], 100)

    def test_teacher_checkpoint_round_trip_and_validation(self) -> None:
        model = RankOneConvNet().eval()
        expected_state = copy.deepcopy(model.state_dict())
        settings = DATASET_REGISTRY["cifar100"].train_settings
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            info = save_teacher_checkpoint(
                path,
                model,
                dataset="toy",
                pair="toy_pair",
                architecture="toy_teacher",
                num_classes=2,
                seed=7,
                settings=settings,
                model_profile="toy_profile",
                data_profile="toy_data_profile",
                selection_policy="final_epoch",
                selected_epoch=1,
                metrics={"accuracy": 1.0},
            )
            with torch.no_grad():
                model.conv.weight.zero_()
            loaded_info = load_teacher_checkpoint(
                path,
                model,
                dataset="toy",
                pair="toy_pair",
                architecture="toy_teacher",
                num_classes=2,
                seed=7,
                model_profile="toy_profile",
                data_profile="toy_data_profile",
                expected_settings=settings,
            )

            self.assertEqual(info["sha256"], loaded_info["sha256"])
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor.cpu(), expected_state[name]))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_teacher_checkpoint(
                    path,
                    RankOneConvNet(),
                    dataset="toy",
                    pair="toy_pair",
                    architecture="toy_teacher",
                    num_classes=2,
                    seed=8,
                    model_profile="toy_profile",
                    data_profile="toy_data_profile",
                    expected_settings=settings,
                )
            corrupted_path = Path(directory) / "teacher_corrupted_settings.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["train_settings"]["lr"] *= 2
            torch.save(payload, corrupted_path)
            with self.assertRaisesRegex(ValueError, "training_fingerprint"):
                load_teacher_checkpoint(
                    corrupted_path,
                    RankOneConvNet(),
                    dataset="toy",
                    pair="toy_pair",
                    architecture="toy_teacher",
                    num_classes=2,
                    seed=7,
                    model_profile="toy_profile",
                    data_profile="toy_data_profile",
                    expected_settings=settings,
                )
            stale_path = Path(directory) / "teacher_stale_but_consistent.pt"
            stale_payload = torch.load(path, map_location="cpu", weights_only=True)
            stale_settings = replace(settings, lr=settings.lr * 2)
            stale_payload["train_settings"] = asdict(stale_settings)
            stale_payload["training_fingerprint"] = teacher_training_fingerprint(
                dataset="toy",
                pair="toy_pair",
                architecture="toy_teacher",
                num_classes=2,
                seed=7,
                settings=stale_settings,
                model_profile="toy_profile",
                data_profile="toy_data_profile",
            )
            torch.save(stale_payload, stale_path)
            with self.assertRaisesRegex(ValueError, "registered protocol"):
                load_teacher_checkpoint(
                    stale_path,
                    RankOneConvNet(),
                    dataset="toy",
                    pair="toy_pair",
                    architecture="toy_teacher",
                    num_classes=2,
                    seed=7,
                    model_profile="toy_profile",
                    data_profile="toy_data_profile",
                    expected_settings=settings,
                )

    def test_registered_pairs_resolve_teacher_and_student_roles(self) -> None:
        cifar10_pair = get_pair_spec("cifar10", "resnet50_to_resnet18")
        cifar100_pair = get_pair_spec("cifar100", "resnet56_to_resnet20")

        self.assertEqual(get_role_name(cifar10_pair, "teacher"), "resnet50")
        self.assertEqual(get_role_name(cifar10_pair, "student"), "resnet18")
        self.assertEqual(get_role_name(cifar100_pair, "teacher"), "resnet56")
        self.assertEqual(get_role_name(cifar100_pair, "student"), "resnet20")

    def test_cifar10_pair_uses_cifar_stem(self) -> None:
        cifar_stem_model = build_pair_model("cifar10", "resnet50_to_resnet18", "student", 10)

        self.assertEqual(tuple(cifar_stem_model.conv1.kernel_size), (3, 3))
        self.assertEqual(tuple(cifar_stem_model.conv1.stride), (1, 1))
        self.assertIsInstance(cifar_stem_model.maxpool, nn.Identity)

    def test_cifar100_pair_defaults_are_supervised(self) -> None:
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--dataset",
                "cifar100",
                "--pair",
                "resnet56_to_resnet20",
                "--method",
                "inhernet",
            ]
        )
        pair = get_pair_spec(args.dataset, args.pair)
        settings = resolve_train_settings(DATASET_REGISTRY[args.dataset], args, pair)

        self.assertEqual(settings.optimizer_name, "sgd")
        self.assertEqual(settings.batch_size, 64)
        self.assertEqual(settings.epochs, 240)
        self.assertEqual(settings.lr_milestones, (150, 180, 210))
        self.assertEqual(resolve_compressed_train_mode(args, pair), "supervised")
        self.assertEqual(build_method_tag("inhernet", args, pair, settings), "small_rank_8_heads_3_supervised")
        self.assertFalse(resolve_compress_linear(pair))

    def test_kd_fraction_preserves_registered_loss_weight_total(self) -> None:
        parser = build_argparser()
        for dataset, pair_name, expected_total in (
            ("cifar10", "resnet50_to_resnet18", 9.1),
            ("glue_sst2", "bert4_to_bert2", 2.0),
        ):
            args = parser.parse_args(
                [
                    "--dataset", dataset,
                    "--pair", pair_name,
                    "--method", "hetero",
                    "--kd-fraction", "0.25",
                ]
            )
            settings = resolve_train_settings(
                DATASET_REGISTRY[dataset], args, get_pair_spec(dataset, pair_name)
            )
            self.assertAlmostEqual(settings.kd_loss_weight, expected_total * 0.25)
            self.assertAlmostEqual(settings.ce_loss_weight, expected_total * 0.75)
            self.assertAlmostEqual(
                settings.kd_loss_weight + settings.ce_loss_weight, expected_total
            )

    def test_added_dataset_registries_are_small_a6000_targets(self) -> None:
        parser = build_argparser()
        pet_args = parser.parse_args(
            [
                "--dataset",
                "oxford_pets",
                "--pair",
                "resnet34_to_resnet18",
                "--method",
                "hetero",
            ]
        )
        pet_spec = DATASET_REGISTRY[pet_args.dataset]
        pet_pair = get_pair_spec(pet_args.dataset, pet_args.pair)
        pet_settings = resolve_train_settings(pet_spec, pet_args, pet_pair)

        self.assertEqual(pet_spec.num_classes, 37)
        self.assertEqual(pet_spec.task_type, "vision")
        self.assertEqual(pet_spec.image_size, 224)
        self.assertEqual(pet_spec.eval_split_name, "validation")
        self.assertEqual(pet_spec.final_test_split_name, "test")
        self.assertEqual(pet_spec.primary_metric_name, "balanced_accuracy")
        self.assertEqual(pet_spec.primary_metric_display, "Mean Per-Class Accuracy (%)")
        self.assertEqual(pet_spec.metric_names, ("accuracy", "macro_f1", "balanced_accuracy"))
        self.assertEqual(get_role_name(pet_pair, "teacher"), "resnet34")
        self.assertEqual(get_role_name(pet_pair, "student"), "resnet18")
        self.assertEqual(pet_settings.epochs, 30)
        self.assertEqual(pet_settings.lr, 0.001)
        self.assertFalse(resolve_compress_linear(pet_pair))

        glue_args = parser.parse_args(
            [
                "--dataset",
                "glue_sst2",
                "--pair",
                "bert4_to_bert2",
                "--method",
                "hetero",
            ]
        )
        glue_spec = DATASET_REGISTRY[glue_args.dataset]
        glue_pair = get_pair_spec(glue_args.dataset, glue_args.pair)
        glue_settings = resolve_train_settings(glue_spec, glue_args, glue_pair)

        self.assertEqual(glue_spec.num_classes, 2)
        self.assertEqual(glue_spec.task_type, "text")
        self.assertEqual(glue_spec.problem_type, "classification")
        self.assertEqual(glue_spec.text_task_name, "sst2")
        self.assertEqual(glue_spec.eval_split_name, "validation")
        self.assertEqual(glue_spec.primary_metric_display, "GLUE Accuracy (%)")
        self.assertEqual(get_role_name(glue_pair, "teacher"), "google/bert_uncased_L-4_H-256_A-4")
        self.assertEqual(get_role_name(glue_pair, "student"), "google/bert_uncased_L-2_H-128_A-2")
        self.assertEqual(glue_settings.epochs, 4)
        self.assertEqual(glue_settings.optimizer_name, "adamw")
        self.assertEqual(glue_settings.scheduler_name, "linear")
        self.assertEqual(glue_settings.warmup_ratio, 0.1)
        self.assertEqual(glue_settings.max_grad_norm, 1.0)
        self.assertTrue(glue_settings.exclude_bias_norm_from_weight_decay)
        self.assertTrue(resolve_compress_linear(glue_pair))
        self.assertIn("_linear", build_method_tag("hetero", glue_args, glue_pair, glue_settings))

        offline_glue = build_pair_model(
            "glue_sst2", "bert4_to_bert2", "teacher", 2, initialize_pretrained=False
        )
        self.assertEqual(offline_glue.config.hidden_size, 256)
        self.assertEqual(offline_glue.config.num_hidden_layers, 4)

        glue_expectations = {
            "glue_mrpc": ("mrpc", 2, "validation", "accuracy", ("accuracy", "f1"), "classification"),
            "glue_qqp": ("qqp", 2, "validation", "accuracy", ("accuracy", "f1"), "classification"),
            "glue_sst2": ("sst2", 2, "validation", "accuracy", ("accuracy",), "classification"),
            "glue_mnli": ("mnli", 3, "validation_matched", "accuracy", ("accuracy",), "classification"),
            "glue_rte": ("rte", 2, "validation", "accuracy", ("accuracy",), "classification"),
            "glue_qnli": ("qnli", 2, "validation", "accuracy", ("accuracy",), "classification"),
            "glue_cola": (
                "cola",
                2,
                "validation",
                "matthews_correlation",
                ("matthews_correlation", "accuracy"),
                "classification",
            ),
            "glue_stsb": ("stsb", 1, "validation", "pearson", ("pearson", "spearmanr"), "regression"),
        }
        for dataset_name, expectation in glue_expectations.items():
            task_name, num_classes, eval_split, primary_metric, metric_names, problem_type = expectation
            spec = DATASET_REGISTRY[dataset_name]
            self.assertEqual(spec.text_task_name, task_name)
            self.assertEqual(spec.num_classes, num_classes)
            self.assertEqual(spec.eval_split_name, eval_split)
            self.assertEqual(spec.primary_metric_name, primary_metric)
            self.assertEqual(spec.metric_names, metric_names)
            self.assertEqual(spec.problem_type, problem_type)


class WrapperTests(unittest.TestCase):
    def test_inhernet_rank_one_conv_preserves_rank_one_weight(self) -> None:
        torch.manual_seed(7)
        dense_model = RankOneConvNet().eval()
        sample = torch.randn(4, 2, 5, 5)

        inhernet = GenericInherNet(copy.deepcopy(dense_model)).eval()
        inhernet.load_dense_state_dict(dense_model.state_dict())
        inhernet.apply_svd(rank=1, head_num=3, svd_backend="cpu")

        with torch.no_grad():
            expected = dense_model(sample)
            actual = inhernet(sample)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_inhernet_multi_head_gate_breaks_expert_gradient_symmetry(self) -> None:
        torch.manual_seed(13)
        dense_model = RankOneConvNet().train()
        sample = torch.randn(6, 2, 5, 5)

        inhernet = GenericInherNet(copy.deepcopy(dense_model)).train()
        inhernet.load_dense_state_dict(dense_model.state_dict())
        inhernet.apply_svd(rank=1, head_num=3, svd_backend="cpu")

        output = inhernet(sample)
        output.square().mean().backward()
        gated_conv = next(module for module in inhernet.modules() if hasattr(module, "conv_list"))
        expert_grads = [expert.weight.grad.detach().clone() for expert in gated_conv.conv_list]

        self.assertFalse(torch.allclose(expert_grads[0], expert_grads[1]))

    def test_hetero_can_optionally_compress_linear_layers(self) -> None:
        torch.manual_seed(11)
        dense_model = TinyConvLinearNet().eval()
        sample = torch.randn(5, 2, 3, 3)
        calib_inputs = torch.randn(12, 2, 3, 3)
        calib_loader = DataLoader(
            TensorDataset(calib_inputs, torch.zeros(12, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )

        hetero = GenericHeteroNet(copy.deepcopy(dense_model)).eval()
        hetero.load_dense_state_dict(dense_model.state_dict())
        rank_map, backend = hetero.apply_hetero_svd(
            calib_loader=calib_loader,
            head_num=3,
            reference_rank=1,
            max_calib_batches=3,
            svd_backend="cpu",
            expert_noise_scale=0.01,
            compress_linear=True,
        )

        with torch.no_grad():
            actual = hetero(sample)

        self.assertEqual(backend, "cpu")
        self.assertIn("fc", rank_map)
        self.assertTrue(any(isinstance(module, GatedSVDLinear) for module in hetero.modules()))
        self.assertEqual(tuple(actual.shape), (5, 30))
        self.assertEqual(
            count_parameters(hetero), hetero.hetero_report["reference_inhernet_parameters"]
        )
        self.assertEqual(hetero.hetero_report["selected_parameters"], count_parameters(hetero))
        lift_probe = hetero.hetero_report["conditional_lift_probe"]
        self.assertEqual(
            lift_probe["factorized_layer_count"],
            hetero.hetero_report["factorized_layer_count"],
        )
        self.assertLess(lift_probe["max_relative_expert_mean_shift"], 1e-6)
        self.assertGreater(lift_probe["mean_relative_expert_diversity"], 0.0)
        routed = next(module for module in hetero.modules() if isinstance(module, GatedSVDLinear))
        self.assertEqual(routed.gate.in_features, rank_map["fc"])

    def test_text_hetero_matches_uniform_inhernet_parameter_count(self) -> None:
        token_ids = torch.randint(0, 1000, (12, 5))
        loader = DataLoader(
            TensorDataset(token_ids, torch.zeros(12, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )
        for reference_rank in (1, 2):
            dense_model = FixedHeavyLinearNet().eval()
            hetero = GenericHeteroNet(copy.deepcopy(dense_model)).eval()
            hetero.apply_hetero_svd(
                calib_loader=loader,
                head_num=2,
                reference_rank=reference_rank,
                max_calib_batches=3,
                svd_backend="cpu",
                compress_linear=True,
            )

            report = hetero.hetero_report
            inhernet = GenericInherNet(copy.deepcopy(dense_model)).eval()
            inhernet.apply_svd(
                rank=reference_rank,
                head_num=2,
                svd_backend="cpu",
                include_linear=True,
            )
            self.assertEqual(report["reference_inhernet_rank"], reference_rank)
            self.assertEqual(report["reference_inhernet_parameters"], count_parameters(inhernet))
            self.assertEqual(report["selected_parameters"], report["reference_inhernet_parameters"])
            self.assertEqual(report["selected_parameters"], count_parameters(hetero))
            self.assertEqual(report["target_layer_count"], len(report["allocation_map"]))
            self.assertEqual(
                report["target_layer_count"],
                report["factorized_layer_count"] + report["dense_layer_count"],
            )
            self.assertGreater(report["budget_utilization"], 0.0)
            self.assertLessEqual(report["budget_utilization"], 1.0)

    def test_vision_hetero_matches_uniform_inhernet_parameter_count(self) -> None:
        inputs = torch.randn(8, 2, 3, 3)
        loader = DataLoader(
            TensorDataset(inputs, torch.zeros(8, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )
        dense_model = TinyConvLinearNet().eval()
        hetero = GenericHeteroNet(copy.deepcopy(dense_model)).eval()
        hetero.apply_hetero_svd(
            calib_loader=loader,
            head_num=3,
            reference_rank=1,
            max_calib_batches=2,
            svd_backend="cpu",
            compress_linear=False,
        )
        inhernet = GenericInherNet(copy.deepcopy(dense_model)).eval()
        inhernet.apply_svd(
            rank=1,
            head_num=3,
            svd_backend="cpu",
            include_linear=False,
        )

        report = hetero.hetero_report
        self.assertEqual(report["reference_inhernet_parameters"], count_parameters(inhernet))
        self.assertEqual(count_parameters(hetero), count_parameters(inhernet))
        self.assertEqual(report["selected_parameters"], count_parameters(hetero))

    def test_weighted_uniform_policy_uses_the_registered_rank(self) -> None:
        inputs = torch.randn(8, 8, 5, 5)
        loader = DataLoader(
            TensorDataset(inputs, torch.zeros(8, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )
        hetero = GenericHeteroNet(
            nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding=1)).eval()
        ).eval()
        rank_map, _ = hetero.apply_hetero_svd(
            calib_loader=loader,
            head_num=3,
            reference_rank=1,
            max_calib_batches=2,
            svd_backend="cpu",
            compress_linear=False,
            allocation_scale="weighted_uniform",
        )
        self.assertEqual(set(rank_map.values()), {1})
        self.assertEqual(hetero.hetero_report["allocation_scale"], "weighted_uniform")
        self.assertEqual(hetero.hetero_report["allocator"], "fixed_registered_rank")
        self.assertEqual(
            hetero.hetero_report["decomposition_metric"], "activation_weighted"
        )

    def test_weighted_uniform_matches_inhernet_parameter_count_for_linear_targets(self) -> None:
        token_ids = torch.randint(0, 1000, (12, 5))
        loader = DataLoader(
            TensorDataset(token_ids, torch.zeros(12, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )
        dense_model = FixedHeavyLinearNet().eval()
        hetero = GenericHeteroNet(copy.deepcopy(dense_model)).eval()
        rank_map, _ = hetero.apply_hetero_svd(
            calib_loader=loader,
            head_num=2,
            reference_rank=2,
            max_calib_batches=3,
            svd_backend="cpu",
            compress_linear=True,
            allocation_scale="weighted_uniform",
        )
        inhernet = GenericInherNet(copy.deepcopy(dense_model)).eval()
        inhernet.apply_svd(
            rank=2,
            head_num=2,
            svd_backend="cpu",
            include_linear=True,
        )
        self.assertEqual(set(rank_map.values()), {2})
        self.assertEqual(count_parameters(hetero), count_parameters(inhernet))

    def test_research_nested_policy_preserves_the_registered_lite_rank(self) -> None:
        inputs = torch.randn(8, 8, 5, 5)
        loader = DataLoader(
            TensorDataset(inputs, torch.zeros(8, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )
        hetero = GenericHeteroNet(
            nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding=1)).eval()
        ).eval()
        rank_map, _ = hetero.apply_hetero_svd(
            calib_loader=loader,
            head_num=3,
            reference_rank=4,
            max_calib_batches=2,
            svd_backend="cpu",
            allocation_scale="research_nested_relative",
            research_protected_rank=2,
            allow_research_rank_probe=True,
        )
        self.assertTrue(rank_map)
        self.assertGreaterEqual(min(rank_map.values()), 2)
        self.assertEqual(hetero.hetero_report["protected_inheritance_rank"], 2)
        self.assertEqual(hetero.hetero_report["protocol"], "research_rank_allocation")

    def test_research_relative_policy_is_explicit(self) -> None:
        inputs = torch.randn(8, 8, 5, 5)
        loader = DataLoader(
            TensorDataset(inputs, torch.zeros(8, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )
        hetero = GenericHeteroNet(
            nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding=1)).eval()
        ).eval()
        with self.assertRaisesRegex(ValueError, "explicit diagnostics-only opt-in"):
            hetero.apply_hetero_svd(
                calib_loader=loader,
                head_num=3,
                reference_rank=4,
                max_calib_batches=2,
                svd_backend="cpu",
                allocation_scale="research_relative",
            )
        hetero.apply_hetero_svd(
            calib_loader=loader,
            head_num=3,
            reference_rank=4,
            max_calib_batches=2,
            svd_backend="cpu",
            allocation_scale="research_relative",
            allow_research_rank_probe=True,
        )
        self.assertEqual(
            hetero.hetero_report["allocation_scale"], "research_relative"
        )

    def test_default_auto_backend_supports_cpu_hetero(self) -> None:
        inputs = torch.randn(8, 8, 5, 5)
        loader = DataLoader(
            TensorDataset(inputs, torch.zeros(8, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )
        hetero = GenericHeteroNet(
            nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding=1)).eval()
        ).eval()
        rank_map, backend = hetero.apply_hetero_svd(
            calib_loader=loader,
            head_num=3,
            reference_rank=4,
            max_calib_batches=2,
        )
        self.assertTrue(rank_map)
        self.assertEqual(backend, "device")

    def test_initialization_diagnostics_are_exact_for_identical_models(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(8, 2)).train()
        inherited = copy.deepcopy(model).eval()
        loader = DataLoader(
            TensorDataset(torch.randn(6, 2, 2, 2), torch.randint(0, 2, (6,))),
            batch_size=3,
            shuffle=False,
        )
        diagnostics = evaluate_inheritance_diagnostics(
            model,
            inherited,
            loader,
            torch.device("cpu"),
            problem_type="classification",
            num_labels=2,
            metric_names=("accuracy",),
        )
        self.assertEqual(diagnostics["relative_output_squared_error"], 0.0)
        self.assertEqual(diagnostics["output_squared_error_per_example"], 0.0)
        self.assertAlmostEqual(diagnostics["prediction_agreement"], 1.0)
        self.assertAlmostEqual(diagnostics["teacher_to_inherited_kl"], 0.0, places=6)
        self.assertTrue(model.training)
        self.assertFalse(inherited.training)

    def test_hetero_diagnostics_include_held_out_local_operator_probe(self) -> None:
        torch.manual_seed(19)
        teacher = TinyConvLinearNet().train()
        inputs = torch.randn(8, 2, 3, 3)
        labels = torch.randint(0, 30, (8,))
        loader = DataLoader(
            TensorDataset(inputs, labels), batch_size=2, shuffle=False
        )
        inherited = GenericHeteroNet(copy.deepcopy(teacher)).eval()
        inherited.apply_hetero_svd(
            calib_loader=loader,
            head_num=3,
            reference_rank=1,
            max_calib_batches=2,
            svd_backend="cpu",
            expert_noise_scale=0.01,
            compress_linear=False,
        )

        diagnostics = evaluate_inheritance_diagnostics(
            teacher,
            inherited,
            loader,
            torch.device("cpu"),
            problem_type="classification",
            num_labels=30,
            metric_names=("accuracy",),
            evaluation_split="validation",
            local_operator_max_batches=2,
        )

        probe = diagnostics["local_operator_probe"]
        layer_rows = list(probe["per_layer"].values())
        self.assertEqual(probe["evaluation_split"], "validation")
        self.assertEqual(probe["max_batches"], 2)
        self.assertEqual(probe["batches"], 2)
        self.assertEqual(probe["examples"], 4)
        self.assertEqual(probe["factorized_layer_count"], len(layer_rows))
        self.assertAlmostEqual(
            probe["squared_error_sum"],
            sum(row["squared_error_sum"] for row in layer_rows),
        )
        self.assertAlmostEqual(
            probe["dense_squared_sum"],
            sum(row["dense_squared_sum"] for row in layer_rows),
        )
        self.assertAlmostEqual(
            probe["relative_squared_error"],
            probe["squared_error_sum"] / probe["dense_squared_sum"],
        )
        self.assertTrue(all(row["moment_mode"] != "unknown" for row in layer_rows))
        self.assertTrue(teacher.training)
        self.assertFalse(inherited.training)

    def test_inhernet_vision_profile_leaves_linear_classifier_dense(self) -> None:
        dense_model = TinyConvLinearNet().eval()
        inhernet = GenericInherNet(copy.deepcopy(dense_model)).eval()
        inhernet.apply_svd(rank=1, head_num=3, svd_backend="cpu")

        self.assertIsInstance(inhernet.backbone.fc, nn.Linear)
        self.assertIsInstance(inhernet.backbone.conv, nn.Sequential)

    def test_oxford_formal_targets_exclude_the_linear_classifier(self) -> None:
        dense_model = build_pair_model(
            "oxford_pets",
            "resnet34_to_resnet18",
            "teacher",
            37,
            initialize_pretrained=False,
        )
        inhernet = GenericInherNet(copy.deepcopy(dense_model))
        hetero = GenericHeteroNet(copy.deepcopy(dense_model))

        self.assertIsInstance(dense_model.fc, nn.Linear)
        self.assertNotIn("fc", inhernet._collect_target_layers(include_linear=False))
        self.assertNotIn("fc", hetero._collect_hetero_target_layers(include_linear=False))

    def test_inhernet_text_profile_can_target_linear_layers(self) -> None:
        dense_model = TinyConvLinearNet().eval()
        inhernet = GenericInherNet(copy.deepcopy(dense_model)).eval()
        inhernet.apply_svd(rank=1, head_num=3, svd_backend="cpu", include_linear=True)

        self.assertIsInstance(inhernet.backbone.fc, nn.Sequential)

    def test_hetero_uses_uncentered_second_moment(self) -> None:
        linear_model = nn.Sequential(nn.Linear(2, 2, bias=False)).eval()
        inputs = torch.tensor([[10.0, 0.0], [12.0, 0.0], [11.0, 0.0], [9.0, 0.0]])
        loader = DataLoader(
            TensorDataset(inputs, torch.zeros(4, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )
        hetero = GenericHeteroNet(linear_model).eval()
        moments, metadata = hetero._estimate_input_second_moments(
            loader,
            max_batches=1,
            include_linear=True,
            shrinkage=0.0,
        )

        moment = moments["0"]
        self.assertGreater(float(moment[0, 0]), 100.0)
        self.assertAlmostEqual(float(moment[0, 0]), 111.5, places=2)
        self.assertEqual(metadata["0"]["samples"], 4)

    def test_wide_linear_second_moment_stays_diagonal(self) -> None:
        linear_model = nn.Sequential(nn.Linear(513, 8, bias=False)).eval()
        inputs = torch.randn(4, 513)
        loader = DataLoader(TensorDataset(inputs, torch.zeros(4, dtype=torch.long)), batch_size=4)
        hetero = GenericHeteroNet(linear_model).eval()
        moments, metadata = hetero._estimate_input_second_moments(
            loader,
            max_batches=1,
            include_linear=True,
        )
        self.assertEqual(tuple(moments["0"].shape), (513,))
        self.assertEqual(metadata["0"]["mode"], "diagonal")

    def test_hetero_balance_loss_is_zero_for_uniform_router(self) -> None:
        down = nn.Linear(2, 1, bias=False)
        experts = nn.ModuleList([nn.Linear(1, 2) for _ in range(3)])
        expert_weight = torch.cat([expert.weight for expert in experts], dim=0)
        expert_bias = torch.cat([expert.bias for expert in experts], dim=0)
        module = GatedSVDLinear(down, expert_weight, expert_bias, 3)
        _ = module(torch.randn(5, 2))

        self.assertAlmostEqual(float(module.load_balance_loss()), 0.0, places=6)

    def test_frozen_router_control_keeps_experts_trainable(self) -> None:
        down = nn.Linear(2, 1, bias=False)
        expert_weight = torch.randn(6, 1)
        expert_bias = torch.randn(6)
        module = GatedSVDLinear(down, expert_weight, expert_bias, 3)

        freeze_gating_routers(module)

        self.assertTrue(all(not parameter.requires_grad for parameter in module.gate.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in module.experts.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in module.linear1.parameters()))

    def test_uniform_router_preserves_zero_mean_expert_perturbation(self) -> None:
        down = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            down.weight.copy_(torch.tensor([[2.0, -1.0]]))
        base_weight = torch.tensor([[3.0], [-2.0]])
        base_bias = torch.tensor([0.25, -0.5])
        noise = torch.tensor(
            [
                [[0.3], [-0.6]],
                [[-0.1], [0.2]],
                [[-0.2], [0.4]],
            ]
        )
        fused_weight = (base_weight.unsqueeze(0) + noise).reshape(6, 1)
        fused_bias = base_bias.repeat(3)
        module = GatedSVDLinear(down, fused_weight, fused_bias, 3)
        inputs = torch.randn(7, 2)

        actual = module(inputs)
        expected = torch.nn.functional.linear(down(inputs), base_weight, base_bias)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        actual[0, 0].backward()
        compressed_value = down(inputs)[0, 0].detach()
        expected_router_gradient = noise[:, 0, 0] * compressed_value / 3.0
        self.assertTrue(
            torch.allclose(
                module.gate.bias.grad,
                expected_router_gradient,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_router_probe_separates_identical_and_zero_mean_experts(self) -> None:
        down = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            down.weight.copy_(torch.tensor([[1.0, -0.5]]))
        base_weight = torch.tensor([[1.5], [-1.0]])
        teacher = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            teacher.weight.copy_(base_weight @ down.weight + torch.tensor([[0.2, 0.0], [0.0, -0.2]]))
        noise = torch.tensor(
            [
                [[0.3], [-0.1]],
                [[-0.1], [0.2]],
                [[-0.2], [-0.1]],
            ]
        )
        identical = GatedSVDLinear(
            copy.deepcopy(down), base_weight.repeat(3, 1), None, 3
        )
        conditional = GatedSVDLinear(
            copy.deepcopy(down),
            (base_weight.unsqueeze(0) + noise).reshape(6, 1),
            None,
            3,
        )
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [-0.5, 0.5]])
        loader = DataLoader(
            TensorDataset(inputs, torch.zeros(4, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )

        identical_probe = evaluate_router_gradient_probe(
            teacher,
            identical,
            loader,
            torch.device("cpu"),
            problem_type="classification",
            evaluation_split="validation",
        )
        conditional_probe = evaluate_router_gradient_probe(
            teacher, conditional, loader, torch.device("cpu"), problem_type="classification"
        )

        self.assertEqual(identical_probe["active_router_fraction"], 0.0)
        self.assertEqual(identical_probe["evaluation_split"], "validation")
        self.assertEqual(identical_probe["batch_index"], 0)
        self.assertEqual(identical_probe["mean_relative_expert_diversity"], 0.0)
        self.assertGreater(conditional_probe["active_router_fraction"], 0.0)
        self.assertGreater(conditional_probe["router_weight_gradient_rms"], 0.0)
        self.assertGreater(conditional_probe["mean_relative_expert_diversity"], 0.0)
        self.assertAlmostEqual(
            conditional_probe["mean_normalized_route_entropy"], 1.0, places=6
        )

    def test_hetero_conv_statistics_use_local_patches(self) -> None:
        model = GenericHeteroNet(nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, padding=1)))
        checkerboard = torch.tensor(
            [[[[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]]]
        )
        loader = DataLoader(
            TensorDataset(checkerboard, torch.zeros(1, dtype=torch.long)),
            batch_size=1,
        )
        moments, metadata = model._estimate_input_second_moments(
            loader,
            max_batches=1,
            shrinkage=0.0,
        )

        self.assertEqual(tuple(moments["0"].shape), (9, 9))
        self.assertEqual(metadata["0"]["mode"], "exact_patch")
        self.assertGreater(float(torch.trace(moments["0"])), 0.0)

    def test_hetero_calibration_restores_mode_and_removes_hooks_on_error(self) -> None:
        backbone = RaisingConvNet().train()
        model = GenericHeteroNet(backbone).train()
        loader = DataLoader(
            TensorDataset(torch.randn(2, 2, 3, 3), torch.zeros(2, dtype=torch.long)),
            batch_size=1,
        )

        with self.assertRaisesRegex(RuntimeError, "injected calibration failure"):
            model._estimate_input_second_moments(loader, max_batches=1)

        self.assertTrue(model.training)
        self.assertEqual(len(backbone.conv._forward_hooks), 0)

    def test_wide_stride_conv_uses_output_application_count(self) -> None:
        convolution = nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1)
        model = GenericHeteroNet(nn.Sequential(convolution))
        features, mode, applications = model._extract_input_features(
            convolution,
            torch.randn(2, 32, 8, 8),
            max_features=4096,
        )

        self.assertEqual(mode, "channel_block")
        self.assertEqual(tuple(features.shape), (2 * 4 * 4 * 3 * 3, 32))
        self.assertEqual(applications, 2 * 4 * 4)


if __name__ == "__main__":
    unittest.main()
