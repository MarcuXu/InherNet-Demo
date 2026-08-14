#!/usr/bin/env python3
"""Build a deterministic CSV index from structured hyperparameter-search logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.inheract_artifacts import canonicalize_metadata


PREFIXES = (
    "RUN_METADATA",
    "RUN_SUMMARY",
    "RUN_FINAL_TEST",
    "INHERITANCE_DIAGNOSTICS",
    "TEACHER_CHECKPOINT",
)
METRICS_PREFIX = "RUN_METRICS"


def _parse_record(path: Path, prefix: str, payload: str) -> dict[str, Any]:
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {prefix} record in {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"Malformed {prefix} record in {path}: expected a JSON object.")
    return record


def parse_structured_log(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            for prefix in PREFIXES:
                marker = f"{prefix} "
                if not line.startswith(marker):
                    continue
                records[prefix] = _parse_record(path, prefix, line[len(marker) :])
    return records


def selected_eval_metrics(path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return selected-epoch metrics, reconstructing them for pre-field logs."""
    if "selected_eval_metrics" in summary:
        metrics = summary["selected_eval_metrics"]
        if not isinstance(metrics, Mapping):
            raise ValueError(
                f"Malformed RUN_SUMMARY record in {path}: "
                "selected_eval_metrics must be a JSON object."
            )
        return dict(metrics)

    try:
        selected_epoch = int(summary.get("best_eval_epoch", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Malformed RUN_SUMMARY record in {path}: best_eval_epoch must be an integer."
        ) from exc
    if selected_epoch < 1:
        return {}

    selected_record: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        marker = f"{METRICS_PREFIX} "
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith(marker):
                continue
            record = _parse_record(path, METRICS_PREFIX, line[len(marker) :])
            try:
                epoch = int(record.get("epoch", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed {METRICS_PREFIX} record in {path}: epoch must be an integer."
                ) from exc
            if epoch == selected_epoch:
                selected_record = record
    if selected_record is None:
        return {}
    return {
        key.removeprefix("eval_"): value
        for key, value in selected_record.items()
        if key.startswith("eval_") and key not in {"eval_loss", "eval_split"}
    }


def final_test_metrics(final_test: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return reportable held-out metrics without protocol bookkeeping fields."""
    if final_test is None:
        return {}
    return {
        key: value
        for key, value in final_test.items()
        if key not in {"phase", "selection_epoch", "split", "primary_metric_name"}
    }


def infer_stage(relative_path: Path, *, has_final_test: bool = False) -> str:
    if relative_path.name == "teacher.log":
        return "teacher"
    for part in relative_path.parts:
        if part in {"mechanism", "optimization", "distillation", "ablation"}:
            return part
    if has_final_test:
        return "formal"
    return "unknown"


def training_objective(metadata: Mapping[str, Any]) -> str | None:
    compressed_mode = metadata.get("compressed_train_mode")
    if isinstance(compressed_mode, str):
        return compressed_mode
    method = metadata.get("method")
    if method in {
        "student_kd",
        "student_dkd",
        "student_kd_logit_standardized",
        "student_ctkd",
        "student_catkd",
        "student_simkd",
        "student_reviewkd",
        "student_crd",
    }:
        return "distillation"
    if method in {"teacher", "student"}:
        return "supervised"
    return None


def distillation_configuration(
    metadata: Mapping[str, Any],
    train_settings: Mapping[str, Any],
    objective: str | None,
) -> str | None:
    if objective != "distillation":
        return None
    for field, kind in (
        ("decoupled_distillation", "dkd"),
        ("logit_standardized_distillation", "logit_standardized_kd"),
        ("curriculum_temperature_distillation", "ctkd"),
        ("baseline_settings", str(metadata.get("method", "feature_distillation"))),
    ):
        settings = metadata.get(field)
        if isinstance(settings, Mapping):
            return json.dumps({"type": kind, **settings}, sort_keys=True)
    return json.dumps(
        {
            "type": "standard_kd",
            "temperature": train_settings.get("kd_temperature"),
            "kd_loss_weight": train_settings.get("kd_loss_weight"),
            "ce_loss_weight": train_settings.get("ce_loss_weight"),
        },
        sort_keys=True,
    )


def build_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.log")):
        records = parse_structured_log(path)
        metadata = records.get("RUN_METADATA")
        summary = records.get("RUN_SUMMARY")
        if metadata is None or summary is None:
            continue
        metadata = canonicalize_metadata(metadata)
        final_test = records.get("RUN_FINAL_TEST")
        final_metrics = final_test_metrics(final_test)
        primary_metric_name = summary.get("primary_metric_name")
        relative_path = path.relative_to(root)
        inheract_report = metadata.get("inheract_report", {})
        is_inhernet = metadata.get("method") == "inhernet"
        train_settings = metadata.get("train_settings", {})
        if not isinstance(train_settings, Mapping):
            train_settings = {}
        teacher_checkpoint = metadata.get("teacher_checkpoint", {})
        if not isinstance(teacher_checkpoint, Mapping):
            teacher_checkpoint = {}
        if metadata.get("method") == "teacher":
            teacher_checkpoint_record = records.get("TEACHER_CHECKPOINT", {})
            if isinstance(teacher_checkpoint_record, Mapping):
                teacher_checkpoint = teacher_checkpoint_record
        objective = training_objective(metadata)
        rows.append(
            {
                "dataset": metadata.get("dataset"),
                "pair": metadata.get("pair"),
                "method": metadata.get("method"),
                "seed": metadata.get("seed"),
                "stage": infer_stage(relative_path, has_final_test=final_test is not None),
                "candidate": (
                    metadata.get("search_candidate")
                    or metadata.get("inheract_recipe_id")
                    or metadata.get("config_tag", "default")
                ),
                "size": metadata.get("size"),
                "reference_inhernet_rank": (
                    metadata.get("rank")
                    if is_inhernet
                    else inheract_report.get("reference_inhernet_rank")
                ),
                "reference_inhernet_parameters": (
                    metadata.get("num_parameters")
                    if is_inhernet
                    else inheract_report.get("reference_inhernet_parameters")
                ),
                "achieved_target_ratio": inheract_report.get("achieved_target_ratio"),
                "achieved_whole_model_ratio": inheract_report.get("achieved_ratio"),
                "parameters": metadata.get("num_parameters"),
                "deployment_parameters": metadata.get(
                    "deployment_parameters", metadata.get("num_parameters")
                ),
                "optimization_parameters": metadata.get(
                    "optimization_parameters", metadata.get("num_parameters")
                ),
                "training_only_auxiliary_parameters": metadata.get(
                    "training_only_auxiliary_parameters", 0
                ),
                "compressed_train_mode": metadata.get("compressed_train_mode"),
                "training_objective": objective,
                "lr_scale": metadata.get("lr_scale", 1.0),
                "resolved_lr": train_settings.get("lr"),
                "distillation_config": distillation_configuration(
                    metadata,
                    train_settings,
                    objective,
                ),
                "teacher_checkpoint_path": teacher_checkpoint.get("path"),
                "teacher_selected_epoch": teacher_checkpoint.get("selected_epoch"),
                "metric": primary_metric_name,
                "best_validation_metric": summary.get("best_eval_metric"),
                "selected_eval_metrics": json.dumps(
                    selected_eval_metrics(path, summary), sort_keys=True
                ),
                "best_epoch": summary.get("best_eval_epoch"),
                "epochs_completed": summary.get("epochs_completed"),
                "eval_split": summary.get("eval_split"),
                "final_test_split": final_test.get("split") if final_test is not None else None,
                "final_test_selection_epoch": (
                    final_test.get("selection_epoch") if final_test is not None else None
                ),
                "final_test_metric": (
                    final_metrics.get(primary_metric_name)
                    if isinstance(primary_metric_name, str)
                    else None
                ),
                "final_test_metrics": json.dumps(final_metrics, sort_keys=True),
                "log": str(relative_path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.log_root / "summary.csv"
    rows = build_rows(args.log_root)
    fieldnames = [
        "dataset",
        "pair",
        "method",
        "seed",
        "stage",
        "candidate",
        "size",
        "reference_inhernet_rank",
        "reference_inhernet_parameters",
        "achieved_target_ratio",
        "achieved_whole_model_ratio",
        "parameters",
        "deployment_parameters",
        "optimization_parameters",
        "training_only_auxiliary_parameters",
        "compressed_train_mode",
        "training_objective",
        "lr_scale",
        "resolved_lr",
        "distillation_config",
        "teacher_checkpoint_path",
        "teacher_selected_epoch",
        "metric",
        "best_validation_metric",
        "selected_eval_metrics",
        "best_epoch",
        "epochs_completed",
        "eval_split",
        "final_test_split",
        "final_test_selection_epoch",
        "final_test_metric",
        "final_test_metrics",
        "log",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} completed search runs to {output}")


if __name__ == "__main__":
    main()
