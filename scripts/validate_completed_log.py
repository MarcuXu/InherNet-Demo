#!/usr/bin/env python3
"""Validate that a completed log matches the command and current protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from demo_code import build_argparser
from experiment_registry import (
    DATASET_REGISTRY,
    build_method_tag,
    get_pair_spec,
    resolve_train_settings,
    validate_args,
)
from scripts.summarize_search import parse_structured_log
from scripts.inheract_artifacts import canonicalize_metadata


def option_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    return None


def expected_evaluation_split(dataset: str, arguments: list[str]) -> str:
    if "--search-validation" in arguments:
        if dataset in {"cifar10", "cifar100"}:
            return "validation"
        if dataset.startswith("glue_"):
            return "train_holdout"
    return DATASET_REGISTRY[dataset].eval_split_name


def expected_final_test_split(resolved_args) -> str | None:
    """Return the held-out split produced by this command, if any."""
    if not resolved_args.final_test:
        return None
    dataset_spec = DATASET_REGISTRY[resolved_args.dataset]
    if dataset_spec.task_type == "text":
        return dataset_spec.eval_split_name if resolved_args.search_validation else None
    uses_validation = (
        resolved_args.dataset in {"cifar10", "cifar100"}
        and resolved_args.search_validation
    ) or dataset_spec.validation_fraction > 0
    return (dataset_spec.test_split or "test") if uses_validation else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Accept a completed initialization-diagnostics run with no training summary.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    expected = args.command[1:] if args.command[:1] == ["--"] else args.command
    records = parse_structured_log(args.log)
    metadata = records.get("RUN_METADATA")
    summary = records.get("RUN_SUMMARY")
    diagnostics = records.get("INHERITANCE_DIAGNOSTICS")
    if metadata is None or (summary is None and not (args.diagnostics_only and diagnostics is not None)):
        raise SystemExit(f"Incomplete structured log: {args.log}")
    metadata = canonicalize_metadata(metadata)
    recorded_argv = metadata.get("argv")
    if not isinstance(recorded_argv, list) or recorded_argv[1:] != expected:
        raise SystemExit(
            f"Completed log command differs from the requested command: {args.log}. "
            "Use a new candidate/recipe identifier or move the stale log."
        )
    dataset = option_value(expected, "--dataset")
    if dataset not in DATASET_REGISTRY:
        raise SystemExit(f"Cannot resolve the dataset protocol for {args.log}")
    resolved_args = build_argparser().parse_args(expected)
    pair_spec = get_pair_spec(resolved_args.dataset, resolved_args.pair)
    validate_args(resolved_args, pair_spec)
    expected_settings = resolve_train_settings(
        DATASET_REGISTRY[dataset], resolved_args, pair_spec
    )
    expected_settings_json = json.loads(json.dumps(asdict(expected_settings)))
    if metadata.get("train_settings") != expected_settings_json:
        raise SystemExit(f"Completed log has stale resolved training settings: {args.log}")
    expected_tag = build_method_tag(
        resolved_args.method,
        resolved_args,
        pair_spec,
        expected_settings,
    )
    if metadata.get("config_tag") != expected_tag:
        raise SystemExit(f"Completed log has a stale resolved method configuration: {args.log}")
    expected_split = expected_evaluation_split(dataset, expected)
    if args.diagnostics_only:
        if "--inheritance-diagnostics-only" not in expected or diagnostics is None:
            raise SystemExit(f"Completed log is not a diagnostics-only run: {args.log}")
        if metadata.get("eval_split") != expected_split:
            raise SystemExit(f"Completed log has the wrong evaluation split: {args.log}")
        try:
            examples = int(diagnostics["examples"])
            relative_error = float(diagnostics["relative_output_squared_error"])
            router_probe = diagnostics["router_probe"]
            router_count = int(router_probe["router_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Completed log has malformed inheritance diagnostics: {args.log}") from exc
        if examples <= 0 or router_count < 0 or not math.isfinite(relative_error):
            raise SystemExit(f"Completed log has invalid inheritance diagnostics: {args.log}")
        if metadata.get("method") == "inheract":
            try:
                local_probe = diagnostics["local_operator_probe"]
                local_error = float(local_probe["relative_squared_error"])
                local_numerator = float(local_probe["squared_error_sum"])
                local_denominator = float(local_probe["dense_squared_sum"])
                local_layers = int(local_probe["factorized_layer_count"])
                local_batches = int(local_probe["batches"])
                local_max_batches = int(local_probe["max_batches"])
                local_split = str(local_probe["evaluation_split"])
                local_aggregation = str(local_probe["aggregation"])
                router_split = str(router_probe["evaluation_split"])
                router_batch_index = int(router_probe["batch_index"])
                lift_probe = metadata["inheract_report"]["conditional_lift_probe"]
                lift_layers = int(lift_probe["factorized_layer_count"])
                mean_shift = float(lift_probe["mean_relative_expert_mean_shift"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Completed InherAct log predates the causal diagnostics: {args.log}. "
                    "Move the stale log before rerunning this cell."
                ) from exc
            if (
                local_layers <= 0
                or local_batches <= 0
                or local_max_batches != 4
                or local_split != expected_split
                or local_aggregation != "ratio_of_summed_squared_errors"
                or router_split != expected_split
                or router_batch_index != 0
                or lift_layers != local_layers
                or not math.isfinite(local_error)
                or local_numerator < 0
                or local_denominator <= 0
                or not math.isclose(
                    local_error,
                    local_numerator / local_denominator,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or not math.isfinite(mean_shift)
                or mean_shift < 0
            ):
                raise SystemExit(f"Completed InherAct log has invalid causal diagnostics: {args.log}")
        return
    if summary is None:
        raise SystemExit(f"Completed log has no training summary: {args.log}")
    expected_epochs = expected_settings.epochs
    if int(summary.get("epochs_completed") or 0) != expected_epochs:
        raise SystemExit(f"Completed log has the wrong epoch horizon: {args.log}")
    if summary.get("eval_split") != expected_split:
        raise SystemExit(f"Completed log has the wrong evaluation split: {args.log}")
    try:
        metric = float(summary["best_eval_metric"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Completed log has no numeric selected metric: {args.log}") from exc
    if not math.isfinite(metric):
        raise SystemExit(f"Completed log has a non-finite selected metric: {args.log}")

    final_test_split = expected_final_test_split(resolved_args)
    if final_test_split is None:
        return
    final_test = records.get("RUN_FINAL_TEST")
    if final_test is None:
        raise SystemExit(f"Completed log is missing held-out final-test results: {args.log}")
    if final_test.get("split") != final_test_split:
        raise SystemExit(f"Completed log has the wrong final-test split: {args.log}")
    if final_test.get("primary_metric_name") != summary.get("primary_metric_name"):
        raise SystemExit(f"Completed log has the wrong final-test metric: {args.log}")
    try:
        selection_epoch = int(final_test["selection_epoch"])
        expected_epoch = int(summary["best_eval_epoch"])
        final_metric = float(final_test[str(summary["primary_metric_name"])])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Completed log has malformed held-out final-test results: {args.log}") from exc
    if selection_epoch != expected_epoch or not math.isfinite(final_metric):
        raise SystemExit(f"Completed log has invalid held-out final-test results: {args.log}")


if __name__ == "__main__":
    main()
