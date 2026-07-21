#!/usr/bin/env python3
"""Build a deterministic CSV index from structured hyperparameter-search logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


PREFIXES = ("RUN_METADATA", "RUN_SUMMARY", "INHERITANCE_DIAGNOSTICS")
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


def infer_stage(relative_path: Path) -> str:
    if relative_path.name == "teacher.log":
        return "teacher"
    for part in relative_path.parts:
        if part in {"mechanism", "optimization", "distillation", "confirmation", "ablation"}:
            return part
    return "unknown"


def build_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.log")):
        records = parse_structured_log(path)
        metadata = records.get("RUN_METADATA")
        summary = records.get("RUN_SUMMARY")
        if metadata is None or summary is None:
            continue
        relative_path = path.relative_to(root)
        hetero_report = metadata.get("hetero_report", {})
        rows.append(
            {
                "dataset": metadata.get("dataset"),
                "pair": metadata.get("pair"),
                "method": metadata.get("method"),
                "seed": metadata.get("seed"),
                "stage": infer_stage(relative_path),
                "candidate": (
                    metadata.get("search_candidate")
                    or metadata.get("hetero_recipe_id")
                    or metadata.get("config_tag", "default")
                ),
                "size": metadata.get("size"),
                "reference_inhernet_rank": hetero_report.get("reference_inhernet_rank"),
                "reference_inhernet_parameters": hetero_report.get("reference_inhernet_parameters"),
                "achieved_target_ratio": hetero_report.get("achieved_target_ratio"),
                "achieved_whole_model_ratio": hetero_report.get("achieved_ratio"),
                "parameters": metadata.get("num_parameters"),
                "metric": summary.get("primary_metric_name"),
                "best_validation_metric": summary.get("best_eval_metric"),
                "selected_eval_metrics": json.dumps(
                    selected_eval_metrics(path, summary), sort_keys=True
                ),
                "best_epoch": summary.get("best_eval_epoch"),
                "epochs_completed": summary.get("epochs_completed"),
                "eval_split": summary.get("eval_split"),
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
        "metric",
        "best_validation_metric",
        "selected_eval_metrics",
        "best_epoch",
        "epochs_completed",
        "eval_split",
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
