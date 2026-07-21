from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import csv

import torch

from demo_code import _selection_details
from experiment_registry import DATASET_REGISTRY, TrainSettings
from glue_data import split_glue_training_data
from scripts.summarize_search import build_rows, parse_structured_log
from scripts.summarize_prestudy import build_rows as build_prestudy_rows
from scripts.plot_prestudy_progression import (
    plot as plot_prestudy_progression,
    validate_data as validate_progression_data,
)
from scripts.plot_prestudy_allocation import (
    plot as plot_prestudy_allocation,
    validate_data as validate_allocation_data,
)
from scripts.plot_prestudy_router_activity import (
    plot as plot_prestudy_router_activity,
    validate_data as validate_router_data,
)
from scripts.plot_prestudy_local_operator import (
    plot as plot_prestudy_local_operator,
    validate_data as validate_local_operator_data,
)
from scripts.hetero_recipes import (
    dataset_profile,
    load_confirmation,
    load_registered_reference,
    load_selected,
    objective_arguments,
    recipe_arguments,
    supervised_control_arguments,
)
from scripts.validate_completed_log import expected_evaluation_split
from training_utils import (
    RunLogger,
    _average_train_batch_ms,
    _compute_classification_metric_values,
    _pearson_correlation_percent,
    _record_epoch_metrics,
    _summarize_history,
    compute_task_metric_values,
    create_history_template,
)


class SearchReportingTests(unittest.TestCase):
    def test_prestudy_summary_aligns_dense_and_factorized_layer_profiles(self) -> None:
        metadata = {
            "dataset": "oxford_pets",
            "pair": "resnet34_to_resnet18",
            "seed": 42,
            "method": "hetero",
            "search_candidate": "prestudy_weighted_uniform",
            "primary_metric_name": "balanced_accuracy",
            "num_parameters": 100,
            "inheritance_setup_seconds": 1.25,
            "max_calib_batches": 2,
            "rank_map": {"factorized": 4},
            "hetero_allocation_scale": "weighted_uniform",
            "hetero_report": {
                "decomposition_metric": "activation_weighted",
                "reference_inhernet_parameters": 100,
                "budget_utilization": 1.0,
                "target_layer_count": 2,
                "factorized_layer_count": 1,
                "dense_layer_count": 1,
                "max_predicted_relative_residual": 0.2,
                "sum_predicted_relative_residual": 0.3,
                "second_moments": {
                    "factorized": {"mode": "exact_patch"},
                    "dense": {"mode": "channel_block"},
                },
                "allocation_layers": {
                    "factorized": {"choice": 4, "max_rank": 8},
                    "dense": {"choice": "dense", "max_rank": 16},
                },
            },
        }
        diagnostics = {
            "teacher_metrics": {"balanced_accuracy": 90.0},
            "inherited_metrics": {"balanced_accuracy": 70.0},
            "relative_output_squared_error": 0.2,
            "output_cosine_similarity": 0.9,
            "teacher_to_inherited_kl": 0.1,
            "prediction_agreement": 0.8,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weighted_uniform.log").write_text(
                "RUN_METADATA " + json.dumps(metadata) + "\n"
                "INHERITANCE_DIAGNOSTICS " + json.dumps(diagnostics) + "\n",
                encoding="utf-8",
            )
            rows = build_prestudy_rows(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0]["rank_profile"]), {"factorized": 4, "dense": 16})
        self.assertEqual(json.loads(rows[0]["moment_modes"]), {"channel_block": 1, "exact_patch": 1})
        self.assertIn("max_calibration_metric_residual_proxy", rows[0])

    def test_prestudy_summary_uses_semantic_layers_and_explicit_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "dataset": "oxford_pets",
                "pair": "resnet34_to_resnet18",
                "method": "hetero",
                "seed": 42,
                "search_candidate": "prestudy_weighted_uniform",
                "hetero_allocation_scale": "weighted_uniform",
                "num_parameters": 100,
                "primary_metric_name": "balanced_accuracy",
                "rank_map": {"layer.a": 4},
                "hetero_report": {
                    "decomposition_metric": "activation_weighted",
                    "reference_inhernet_parameters": 100,
                    "budget_utilization": 1.0,
                    "target_layer_count": 2,
                    "factorized_layer_count": 1,
                    "dense_layer_count": 1,
                    "allocation_layers": {
                        "layer.a": {"choice": 4, "max_rank": 8},
                        "layer.b": {"choice": "dense", "max_rank": 3},
                    },
                },
            }
            diagnostics = {
                "teacher_metrics": {"balanced_accuracy": 90.0},
                "inherited_metrics": {"balanced_accuracy": 70.0},
                "relative_output_squared_error": 0.2,
                "output_cosine_similarity": 0.8,
                "teacher_to_inherited_kl": 0.3,
                "prediction_agreement": 0.75,
            }
            (root / "weighted_uniform.log").write_text(
                "RUN_METADATA " + json.dumps(metadata) + "\n"
                "INHERITANCE_DIAGNOSTICS " + json.dumps(diagnostics) + "\n",
                encoding="utf-8",
            )
            rows = build_prestudy_rows(root)
            self.assertEqual(json.loads(rows[0]["rank_profile"]), {"layer.a": 4, "layer.b": 3})
            self.assertEqual(rows[0]["prediction_agreement_fraction"], 0.75)
            plots = (
                (validate_progression_data, plot_prestudy_progression, "progression.png"),
                (validate_local_operator_data, plot_prestudy_local_operator, "local_operator.png"),
                (validate_router_data, plot_prestudy_router_activity, "router.png"),
                (validate_allocation_data, plot_prestudy_allocation, "allocation.png"),
            )
            for validate, plot, filename in plots:
                validate()
                output = root / filename
                plot(output)
                self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))

    def test_resume_validation_resolves_search_evaluation_splits(self) -> None:
        self.assertEqual(
            expected_evaluation_split("cifar100", ["--search-validation"]),
            "validation",
        )
        self.assertEqual(
            expected_evaluation_split("glue_sst2", ["--search-validation"]),
            "train_holdout",
        )
        self.assertEqual(expected_evaluation_split("glue_sst2", []), "validation")

    def test_selected_hetero_recipes_cover_transfer_profiles(self) -> None:
        selected = load_selected()
        self.assertEqual(
            set(selected),
            {"cifar10", "cifar100", "oxford_pets", "glue_classification", "glue_regression"},
        )
        self.assertEqual(dataset_profile("glue_sst2"), "glue_classification")
        self.assertEqual(dataset_profile("glue_stsb"), "glue_regression")
        self.assertEqual(
            {row["aux_loss_weight"] for row in selected.values()},
            {"0.01"},
        )
        args = recipe_arguments(selected["oxford_pets"])
        self.assertIn("--hetero-recipe-id", args)
        self.assertIn("--compressed-train-mode", args)
        self.assertIn("--hetero-allocation-scale", args)
        self.assertEqual(
            objective_arguments(selected["oxford_pets"]),
            ["--compressed-train-mode", "distillation", "--kd-temperature", "2.0"],
        )
        supervised = supervised_control_arguments(selected["oxford_pets"])
        self.assertIn("weighted_uniform_supervised_control", supervised)
        self.assertNotIn("--kd-temperature", supervised)
        self.assertEqual(
            objective_arguments(selected["cifar100"]),
            ["--compressed-train-mode", "supervised"],
        )
        self.assertEqual(
            objective_arguments(load_registered_reference("glue_sst2")),
            ["--compressed-train-mode", "distillation", "--kd-temperature", "2.0"],
        )
        self.assertEqual(set(load_confirmation()), {"weighted_uniform"})
        for row in selected.values():
            self.assertEqual(row["allocation_scale"], "weighted_uniform")

    def test_mechanism_hpo_does_not_search_rank_allocation(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/hetero_search_candidates.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 9)
        self.assertNotIn("allocation_scale", rows[0])
        self.assertIn("noise_002", {row["candidate_id"] for row in rows})
        self.assertNotIn("joint_regularized", {row["candidate_id"] for row in rows})

    def test_glue_selection_split_is_fixed_and_classification_is_stratified(self) -> None:
        class RecordingDataset:
            def __init__(self) -> None:
                self.kwargs = None

            def train_test_split(self, **kwargs):
                self.kwargs = kwargs
                return {"train": "train_subset", "test": "holdout"}

        classification = RecordingDataset()
        result = split_glue_training_data(
            classification,
            problem_type="classification",
            validation_fraction=0.1,
            validation_split_seed=2026,
        )
        self.assertEqual(result, {"train": "train_subset", "test": "holdout"})
        self.assertEqual(
            classification.kwargs,
            {"test_size": 0.1, "seed": 2026, "stratify_by_column": "label"},
        )

        regression = RecordingDataset()
        split_glue_training_data(
            regression,
            problem_type="regression",
            validation_fraction=0.1,
            validation_split_seed=2026,
        )
        self.assertEqual(regression.kwargs, {"test_size": 0.1, "seed": 2026})

    def test_average_train_batch_time_excludes_evaluation_time(self) -> None:
        self.assertEqual(_average_train_batch_ms(2.0, 4), 500.0)

    def test_glue_metrics_match_known_values(self) -> None:
        predictions = torch.tensor([1, 1, 0, 0])
        labels = torch.tensor([1, 0, 1, 0])
        metrics = _compute_classification_metric_values(
            predictions,
            labels,
            num_labels=2,
            metric_names=("accuracy", "f1", "matthews_correlation"),
        )
        self.assertEqual(metrics["accuracy"], 50.0)
        self.assertEqual(metrics["f1"], 50.0)
        self.assertEqual(metrics["matthews_correlation"], 0.0)

        values = torch.tensor([1.0, 2.0, 2.0, 4.0])
        self.assertAlmostEqual(_pearson_correlation_percent(values, values), 100.0, places=3)
        correlations = compute_task_metric_values(
            values,
            values,
            problem_type="regression",
            num_labels=1,
            metric_names=("pearson", "spearmanr"),
        )
        self.assertAlmostEqual(correlations["spearmanr"], 100.0, places=3)

    def test_summary_keeps_secondary_metrics_at_primary_selected_epoch(self) -> None:
        history = create_history_template()
        logger = RunLogger(log_path="/dev/null", echo=False)
        settings = TrainSettings(
            optimizer_name="sgd",
            lr=0.1,
            batch_size=2,
            epochs=2,
            momentum=0.0,
            weight_decay=0.0,
            lr_milestones=(),
        )
        for epoch, accuracy, f1 in ((1, 80.0, 90.0), (2, 70.0, 95.0)):
            _record_epoch_metrics(
                history,
                logger,
                epoch=epoch,
                settings=settings,
                phase="target",
                train_objective=1.0,
                train_loss=1.0,
                train_metrics={"accuracy": accuracy, "f1": f1},
                test_metrics={"loss": 1.0, "accuracy": accuracy, "f1": f1},
                epoch_time_seconds=1.0,
                train_time_seconds=0.8,
                eval_time_seconds=0.2,
                avg_train_batch_ms=1.0,
                eval_split_name="validation",
                primary_metric_name="accuracy",
            )
        summary = _summarize_history(
            history,
            eval_split_name="validation",
            primary_metric_name="accuracy",
            primary_metric_display="Accuracy (%)",
        )
        self.assertEqual(summary["best_eval_epoch"], 1)
        self.assertEqual(summary["selected_eval_metrics"], {"accuracy": 80.0, "f1": 90.0})
        self.assertEqual(summary["final_eval_metrics"], {"accuracy": 70.0, "f1": 95.0})

        policy, selected_epoch, checkpoint_metrics = _selection_details(
            history,
            DATASET_REGISTRY["glue_mrpc"],
            settings,
            {"eval_split": "validation"},
        )
        self.assertEqual(policy, "best_validation_accuracy")
        self.assertEqual(selected_epoch, 1)
        self.assertEqual(checkpoint_metrics["selected_evaluation_metric"], 80.0)
        self.assertEqual(
            checkpoint_metrics["selected_evaluation_metrics"],
            {"accuracy": 80.0, "f1": 90.0},
        )

    def test_search_summary_is_semantic_and_does_not_export_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "mechanism" / "size_small" / "candidate.log"
            log_path.parent.mkdir(parents=True)
            metadata = {
                "dataset": "glue_mrpc",
                "pair": "bert4_to_bert2",
                "method": "hetero",
                "seed": 42,
                "size": "small",
                "search_candidate": "mechanism_reference",
                "teacher_checkpoint": {"sha256": "must_not_escape"},
            }
            summary = {
                "primary_metric_name": "accuracy",
                "best_eval_metric": 80.0,
                "best_eval_epoch": 1,
                "epochs_completed": 4,
                "selected_eval_metrics": {"accuracy": 80.0, "f1": 88.0},
            }
            log_path.write_text(
                f"RUN_METADATA {json.dumps(metadata)}\nRUN_SUMMARY {json.dumps(summary)}\n",
                encoding="utf-8",
            )
            rows = build_rows(root)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("teacher_sha256", rows[0])
        self.assertNotIn("must_not_escape", json.dumps(rows[0]))
        self.assertEqual(
            json.loads(str(rows[0]["selected_eval_metrics"])),
            {"accuracy": 80.0, "f1": 88.0},
        )

    def test_search_summary_recovers_selected_metrics_from_legacy_epoch_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "teacher.log"
            metadata = {
                "dataset": "glue_mrpc",
                "pair": "bert4_to_bert2",
                "method": "teacher",
                "seed": 42,
            }
            epoch_one = {
                "epoch": 1,
                "eval_accuracy": 80.0,
                "eval_f1": 88.0,
                "eval_loss": 0.5,
                "eval_split": "validation",
            }
            epoch_two = {
                "epoch": 2,
                "eval_accuracy": 79.0,
                "eval_f1": 90.0,
                "eval_loss": 0.4,
                "eval_split": "validation",
            }
            legacy_summary = {
                "primary_metric_name": "accuracy",
                "best_eval_metric": 80.0,
                "best_eval_epoch": 1,
                "epochs_completed": 2,
            }
            log_path.write_text(
                "\n".join(
                    (
                        f"RUN_METADATA {json.dumps(metadata)}",
                        f"RUN_METRICS {json.dumps(epoch_one)}",
                        f"RUN_METRICS {json.dumps(epoch_two)}",
                        f"RUN_SUMMARY {json.dumps(legacy_summary)}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            rows = build_rows(root)
        self.assertEqual(
            json.loads(str(rows[0]["selected_eval_metrics"])),
            {"accuracy": 80.0, "f1": 88.0},
        )

    def test_structured_log_rejects_non_object_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "malformed.log"
            log_path.write_text("RUN_SUMMARY []\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected a JSON object"):
                parse_structured_log(log_path)


if __name__ == "__main__":
    unittest.main()
