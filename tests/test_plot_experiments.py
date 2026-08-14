from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for path in (PROJECT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_experiments import (
    plot_ablation_deltas,
    plot_search_scores,
    prepare_ablation_deltas,
    prepare_search_scores,
)
from plotting_utils import build_plot_label, get_plot_method_key, get_plot_style


def search_row(
    *, candidate: str, metric: float, dataset: str, metric_name: str = "accuracy"
) -> dict[str, object]:
    return {
        "stage": "mechanism",
        "dataset": dataset,
        "pair": "pair",
        "method": "inheract",
        "size": "small",
        "seed": 42,
        "candidate": candidate,
        "metric": metric_name,
        "best_validation_metric": metric,
    }


def ablation_row(
    *, candidate: str, metric: float, size: str, seed: int = 42
) -> dict[str, object]:
    return {
        "stage": "ablation",
        "dataset": "toy",
        "pair": "teacher_to_student",
        "method": "inhernet" if candidate == "ablation_inhernet" else "inheract",
        "size": size,
        "seed": seed,
        "candidate": candidate,
        "metric": "accuracy",
        "best_validation_metric": metric,
    }


class ExperimentFigureTests(unittest.TestCase):
    def test_published_cifar100_kd_variants_have_distinct_public_plot_labels(self) -> None:
        self.assertEqual(
            build_plot_label("student_kd_logit_standardized", {}, detailed=False),
            "Student + Logit Std. KD",
        )
        self.assertEqual(
            build_plot_label("student_dkd", {}, detailed=False),
            "Student + DKD",
        )
        self.assertNotEqual(
            get_plot_style("student_kd_logit_standardized"),
            get_plot_style("student_kd"),
        )
        expected = {
            "student_ctkd": "Student + CTKD",
            "student_catkd": "Student + CAT-KD",
            "student_simkd": "Student + SimKD",
            "student_reviewkd": "Student + ReviewKD",
            "student_crd": "Student + CRD",
        }
        for method, label in expected.items():
            self.assertEqual(build_plot_label(method, {}, detailed=False), label)

    def test_inhernet_labels_expose_the_registered_objective(self) -> None:
        self.assertEqual(
            build_plot_label(
                "inhernet",
                {"size": "small", "compressed_train_mode": "distillation"},
                detailed=False,
            ),
            "InherNet-S + KD",
        )
        self.assertEqual(
            build_plot_label(
                "inhernet",
                {"size": "large", "compressed_train_mode": "supervised"},
                detailed=False,
            ),
            "InherNet-L (supervised)",
        )

    def test_public_inheract_capacity_names_preserve_internal_sizes(self) -> None:
        self.assertEqual(build_plot_label("inheract", {"size": "large"}, detailed=False), "InherAct")
        self.assertEqual(
            build_plot_label("inheract", {"size": "small"}, detailed=False),
            "InherAct-Lite",
        )

    def test_direct_svd_reference_has_a_distinct_public_plot_label(self) -> None:
        metadata = {
            "size": "large",
            "rank": 16,
            "head_num": 1,
            "search_candidate": "ablation_direct_svd",
        }
        self.assertEqual(get_plot_method_key("inhernet", metadata), "direct_svd_inheritance")
        self.assertEqual(
            build_plot_label("inhernet", metadata, detailed=False),
            "Direct SVD inheritance",
        )
        self.assertEqual(
            build_plot_label("inhernet", metadata, detailed=True),
            "Direct SVD inheritance - rank 16, one head",
        )

    def test_search_uses_within_cell_ranks_across_incompatible_metrics(self) -> None:
        rows = [
            search_row(candidate="mechanism_reference", metric=90.0, dataset="vision"),
            search_row(candidate="mechanism_noise_0", metric=80.0, dataset="vision"),
            search_row(candidate="mechanism_reference", metric=0.3, dataset="text", metric_name="mcc"),
            search_row(candidate="mechanism_noise_0", metric=0.8, dataset="text", metric_name="mcc"),
        ]
        scores, cells = prepare_search_scores(rows, stage="mechanism")
        self.assertEqual(len(cells), 2)
        self.assertEqual(scores["mechanism_reference"], [1.0, 0.0])
        self.assertEqual(scores["mechanism_noise_0"], [0.0, 1.0])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search.png"
            plot_search_scores(scores, stage="mechanism", output=output)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))

    def test_ablation_is_paired_with_full_inheract_by_size_and_seed(self) -> None:
        rows = [
            ablation_row(candidate="ablation_full", metric=80.0, size="small"),
            ablation_row(candidate="ablation_no_noise", metric=78.5, size="small"),
            ablation_row(candidate="ablation_full", metric=85.0, size="large"),
            ablation_row(candidate="ablation_no_noise", metric=85.5, size="large"),
        ]
        deltas = prepare_ablation_deltas(rows)
        target = ("toy", "teacher_to_student")
        self.assertEqual(deltas[target]["ablation_no_noise"]["small"], [-1.5])
        self.assertEqual(deltas[target]["ablation_no_noise"]["large"], [0.5])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ablation.png"
            plot_ablation_deltas(deltas, output=output)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))

    def test_ablation_rejects_unpaired_variants(self) -> None:
        rows = [
            ablation_row(candidate="ablation_full", metric=80.0, size="small"),
            ablation_row(candidate="ablation_full", metric=85.0, size="large"),
            ablation_row(candidate="ablation_no_noise", metric=79.0, size="small"),
        ]
        with self.assertRaisesRegex(ValueError, "Unpaired ablation"):
            prepare_ablation_deltas(rows)

    def test_search_rejects_metric_mismatch_inside_a_cell(self) -> None:
        rows = [
            search_row(candidate="mechanism_reference", metric=90.0, dataset="vision"),
            search_row(
                candidate="mechanism_noise_0", metric=0.8, dataset="vision", metric_name="mcc"
            ),
        ]
        with self.assertRaisesRegex(ValueError, "different metrics"):
            prepare_search_scores(rows, stage="mechanism")


if __name__ == "__main__":
    unittest.main()
