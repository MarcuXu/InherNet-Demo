#!/usr/bin/env python3
"""Extract fixed, training-free inheritance diagnostics as JSON."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.summarize_search import parse_structured_log


DISPLAY_NAMES = {
    "prestudy_inhernet": "InherNet",
    "prestudy_unweighted_uniform": "Hetero, weight-only",
    "prestudy_weighted_uniform": "Activation-aware base",
    "prestudy_weighted_uniform_noise_001": "Hetero, conditional lift",
    "prestudy_research_relative": "Relative allocation",
    "prestudy_research_nested_relative": "Nested allocation",
    "prestudy_research_total_output": "Raw-output allocation",
}
def _metric(metrics: Any, name: str | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name) if name else None
    if value is None and metrics:
        value = next(iter(metrics.values()))
    return float(value) if value is not None else None


def build_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.log")):
        records = parse_structured_log(path)
        metadata = records.get("RUN_METADATA")
        diagnostics = records.get("INHERITANCE_DIAGNOSTICS")
        if metadata is None or diagnostics is None:
            continue
        report = metadata.get("hetero_report") or {}
        router_probe = diagnostics.get("router_probe") or {}
        local_probe = diagnostics.get("local_operator_probe") or {}
        lift_probe = report.get("conditional_lift_probe") or {}
        candidate = str(metadata.get("search_candidate") or path.stem)
        rank_map = metadata.get("rank_map") or {}
        allocation_layers = report.get("allocation_layers") or {}
        moment_modes = Counter(
            str(layer.get("mode", "unknown"))
            for layer in (report.get("second_moments") or {}).values()
        )
        rank_profile = {
            name: (
                int(layer["max_rank"])
                if layer.get("choice") == "dense"
                else int(layer["choice"])
            )
            for name, layer in allocation_layers.items()
        }
        rank_values = [int(value) for value in rank_map.values()]
        primary_metric = str(metadata.get("primary_metric_name") or "")
        rows.append(
            {
                "dataset": metadata.get("dataset"),
                "pair": metadata.get("pair"),
                "seed": metadata.get("seed"),
                "variant": candidate,
                "display_name": DISPLAY_NAMES.get(candidate, candidate),
                "allocation": metadata.get("hetero_allocation_scale") or "registered_rank_svd",
                "decomposition_metric": report.get("decomposition_metric") or "weight_only",
                "parameters": metadata.get("num_parameters"),
                "inheritance_setup_seconds": metadata.get("inheritance_setup_seconds"),
                "max_calibration_batches": metadata.get("max_calib_batches"),
                "reference_parameters": (
                    metadata.get("num_parameters")
                    if metadata.get("method") == "inhernet"
                    else report.get("reference_inhernet_parameters")
                ),
                "budget_utilization": (
                    1.0
                    if metadata.get("method") == "inhernet"
                    else report.get("budget_utilization")
                ),
                "target_layers": report.get("target_layer_count"),
                "factorized_layers": report.get("factorized_layer_count"),
                "dense_layers": report.get("dense_layer_count"),
                "moment_modes": json.dumps(dict(sorted(moment_modes.items()))),
                "rank_min": min(rank_values) if rank_values else metadata.get("rank"),
                "factorized_rank_mean": (
                    sum(rank_values) / len(rank_values) if rank_values else metadata.get("rank")
                ),
                "rank_max": max(rank_values) if rank_values else metadata.get("rank"),
                "primary_metric": primary_metric,
                "teacher_metric": _metric(diagnostics.get("teacher_metrics"), primary_metric),
                "inherited_metric": _metric(diagnostics.get("inherited_metrics"), primary_metric),
                "relative_output_squared_error": diagnostics.get("relative_output_squared_error"),
                "output_cosine_similarity": diagnostics.get("output_cosine_similarity"),
                "teacher_to_inherited_kl": diagnostics.get("teacher_to_inherited_kl"),
                "prediction_agreement_fraction": diagnostics.get("prediction_agreement"),
                "router_probe_objective": router_probe.get("objective"),
                "router_probe_objective_value": router_probe.get("objective_value"),
                "router_probe_evaluation_split": router_probe.get("evaluation_split"),
                "router_probe_batch_index": router_probe.get("batch_index"),
                "router_count": router_probe.get("router_count"),
                "active_router_fraction": router_probe.get("active_router_fraction"),
                "router_weight_gradient_rms": router_probe.get(
                    "router_weight_gradient_rms"
                ),
                "router_bias_gradient_rms": router_probe.get("router_bias_gradient_rms"),
                "mean_normalized_route_entropy": router_probe.get(
                    "mean_normalized_route_entropy"
                ),
                "mean_relative_expert_diversity": router_probe.get(
                    "mean_relative_expert_diversity"
                ),
                "per_router_gradient_l2": json.dumps(
                    router_probe.get("per_router_gradient_l2") or {}, sort_keys=False
                ),
                "local_operator_evaluation_split": local_probe.get("evaluation_split"),
                "local_operator_max_batches": local_probe.get("max_batches"),
                "local_operator_batches": local_probe.get("batches"),
                "local_operator_examples": local_probe.get("examples"),
                "local_operator_factorized_layers": local_probe.get(
                    "factorized_layer_count"
                ),
                "local_operator_squared_error_sum": local_probe.get(
                    "squared_error_sum"
                ),
                "local_operator_dense_squared_sum": local_probe.get(
                    "dense_squared_sum"
                ),
                "local_operator_relative_squared_error": local_probe.get(
                    "relative_squared_error"
                ),
                "local_operator_per_layer": json.dumps(
                    local_probe.get("per_layer") or {}, sort_keys=False
                ),
                "conditional_lift_factorized_layers": lift_probe.get(
                    "factorized_layer_count"
                ),
                "conditional_lift_mean_relative_expert_mean_shift": lift_probe.get(
                    "mean_relative_expert_mean_shift"
                ),
                "conditional_lift_max_relative_expert_mean_shift": lift_probe.get(
                    "max_relative_expert_mean_shift"
                ),
                "conditional_lift_mean_relative_expert_diversity": lift_probe.get(
                    "mean_relative_expert_diversity"
                ),
                "conditional_lift_max_relative_expert_diversity": lift_probe.get(
                    "max_relative_expert_diversity"
                ),
                "conditional_lift_per_layer": json.dumps(
                    lift_probe.get("per_layer") or {}, sort_keys=False
                ),
                "max_calibration_metric_residual_proxy": report.get(
                    "max_predicted_relative_residual"
                ),
                "sum_calibration_metric_residual_proxy": report.get(
                    "sum_predicted_relative_residual"
                ),
                "rank_map": json.dumps(rank_map, sort_keys=False),
                "rank_profile": json.dumps(rank_profile, sort_keys=False),
                "log": path.name,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_root",
        type=Path,
        nargs="+",
        help="One or more directories searched recursively for pre-study logs.",
    )
    args = parser.parse_args()
    rows = [row for root in args.log_root for row in build_rows(root)]
    print(json.dumps(rows, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
