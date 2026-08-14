#!/usr/bin/env python3
"""Create publication figures from completed search and ablation logs.

The search view aggregates only normalized ranks computed inside comparable
dataset/pair/method/size/seed cells.  The ablation view reports paired changes
from the full InherAct configuration; it never pools raw metrics across tasks.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plotting_utils import get_pyplot
from scripts.inheract_artifacts import canonicalize_metadata
from scripts.rank_search import (
    DEFAULT_SEARCH_SEEDS,
    STAGE_TARGETS,
    applicable_candidates,
    average_normalized_ranks,
    candidate_ids,
    validate_search_protocol_rows,
)
from scripts.summarize_search import build_rows


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#C7CBD1"
SEARCH_STAGES = ("mechanism", "optimization", "distillation")
ABLATION_LABELS = {
    "ablation_unweighted_uniform": "No activation weighting",
    "ablation_no_noise": "No expert noise",
    "ablation_no_balance": "No balance loss",
    "ablation_no_noise_no_balance": "No noise / balance",
    "ablation_frozen_router": "Fixed uniform router",
}
NON_COMPONENT_ROWS = {
    "ablation_inhernet_small",
    "ablation_inhernet_large",
    "ablation_inheract_lite",
}


def _finite_metric(row: dict[str, Any]) -> float:
    try:
        value = float(row["best_validation_metric"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric best_validation_metric in row: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite best_validation_metric in row: {row}")
    return value


def _save_png(fig: Any, output: Path, plt: Any) -> Path:
    if output.suffix.lower() != ".png":
        raise ValueError("Publication figure output must use the .png suffix.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output


def _pretty_candidate(candidate: str, stage: str) -> str:
    prefix = f"{stage}_"
    value = candidate[len(prefix) :] if candidate.startswith(prefix) else candidate
    replacements = {
        "reference": "Reference",
        "lr_05": "LR × 0.5",
        "lr_1": "LR × 1",
        "lr_2": "LR × 2",
        "aux_0": "Balance = 0",
        "aux_003": "Balance = 0.03",
        "shrink_0": "Shrinkage = 0",
        "shrink_005": "Shrinkage = 0.05",
        "noise_0": "Noise = 0",
        "noise_0005": "Noise = 0.005",
        "noise_002": "Noise = 0.02",
        "joint_sparse": "Joint sparse",
        "distill_reference": "KD reference",
        "mix_reference": "KD fraction = 0.50",
        "supervised": "Supervised (no KD)",
        "registered_reference": "Registered reference",
        "temperature_1": "Temperature = 1",
        "temperature_4": "Temperature = 4",
        "kd_fraction_025": "KD fraction = 0.25",
        "kd_fraction_075": "KD fraction = 0.75",
    }
    return replacements.get(value, value.replace("_", " ").title())


def prepare_search_scores(
    rows: Sequence[dict[str, Any]],
    *,
    stage: str,
    expected_cells: Iterable[tuple[Any, ...]] | None = None,
) -> tuple[dict[str, list[float]], dict[tuple[Any, ...], dict[str, float]]]:
    """Validate completed search rows and compute within-cell normalized ranks."""
    if stage not in SEARCH_STAGES:
        raise ValueError(f"Unknown search stage: {stage}")
    allowed_candidates = set(candidate_ids(stage))
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("stage") != stage:
            continue
        candidate = str(row.get("candidate"))
        if candidate not in allowed_candidates:
            raise ValueError(f"Unknown {stage} candidate in results: {candidate}")
        _finite_metric(row)
        key = (
            row.get("dataset"),
            row.get("pair"),
            row.get("method"),
            row.get("size"),
            int(row.get("seed")),
            candidate,
        )
        if key in indexed:
            raise ValueError(f"Duplicate completed search cell: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"No completed {stage} rows were found.")

    if expected_cells is not None:
        expected = set(expected_cells)
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        if missing:
            preview = "; ".join("/".join(map(str, cell)) for cell in missing[:3])
            raise ValueError(f"Incomplete {stage} matrix: missing {len(missing)} cells; first: {preview}")
        if extra:
            preview = "; ".join("/".join(map(str, cell)) for cell in extra[:3])
            raise ValueError(f"Unexpected {stage} cells outside the requested matrix; first: {preview}")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for key, row in indexed.items():
        grouped[key[:-1]].append(row)
    for cell, cell_rows in grouped.items():
        if len(cell_rows) < 2:
            raise ValueError(f"Search cell needs at least two candidates for ranking: {cell}")
        metrics = {str(row.get("metric")) for row in cell_rows}
        if len(metrics) != 1:
            raise ValueError(f"Search candidates use different metrics inside cell {cell}: {sorted(metrics)}")

    cell_scores = {cell: average_normalized_ranks(cell_rows) for cell, cell_rows in grouped.items()}
    scores: dict[str, list[float]] = defaultdict(list)
    for candidate_scores in cell_scores.values():
        for candidate, score in candidate_scores.items():
            scores[candidate].append(score)
    return dict(scores), cell_scores


def _expected_search_cells(
    stage: str,
    targets: Sequence[tuple[str, str]],
    seeds: Sequence[int],
) -> set[tuple[Any, ...]]:
    methods = ("inheract",)
    candidates = candidate_ids(stage)
    expected: set[tuple[Any, ...]] = set()
    for dataset, pair in targets:
        applicable = applicable_candidates(stage, dataset, candidates)
        for method in methods:
            for size in ("large",):
                for seed in seeds:
                    for candidate in applicable:
                        expected.add((dataset, pair, method, size, seed, candidate))
    return expected


def plot_search_scores(
    scores: dict[str, list[float]],
    *,
    stage: str,
    output: Path,
) -> Path:
    if not scores:
        raise ValueError("Cannot plot an empty search result.")
    ordered = sorted(scores, key=lambda key: (-sum(scores[key]) / len(scores[key]), key))
    plt = get_pyplot("single")
    fig, ax = plt.subplots(figsize=(6.8, max(3.0, 0.42 * len(ordered) + 1.25)))
    for y, candidate in enumerate(ordered):
        values = scores[candidate]
        mean = sum(values) / len(values)
        ax.plot([min(values), max(values)], [y, y], color=LIGHT_GRAY, linewidth=2.0, zorder=1)
        ax.scatter(values, [y] * len(values), s=19, color=GRAY, alpha=0.36, linewidth=0, zorder=2)
        ax.scatter(
            [mean], [y], s=66, color=BLUE, marker="D", edgecolor="white", linewidth=0.8,
            label="Mean" if y == 0 else None, zorder=4,
        )
        ax.text(1.015, y, f"n={len(values)}", va="center", fontsize=8.0, color="#5E6672")
    ax.set_yticks(range(len(ordered)), [_pretty_candidate(candidate, stage) for candidate in ordered])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.09)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Normalized within-cell rank (higher is better)")
    ax.set_title(f"InherAct {stage} search", loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.7)
    ax.grid(False, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035))
    fig.subplots_adjust(left=0.31, right=0.91, bottom=0.23, top=0.89)
    return _save_png(fig, output, plt)


def prepare_ablation_deltas(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, dict[str, list[float]]]]:
    """Return per-target, per-variant, per-size paired deltas from the full configuration."""
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("stage") != "ablation":
            continue
        candidate = str(row.get("candidate"))
        if candidate in NON_COMPONENT_ROWS:
            continue
        if candidate != "ablation_full" and candidate not in ABLATION_LABELS:
            raise ValueError(f"Unknown ablation variant: {candidate}")
        _finite_metric(row)
        key = (
            str(row.get("dataset")), str(row.get("pair")), str(row.get("size")),
            int(row.get("seed")), candidate,
        )
        if key in indexed:
            raise ValueError(f"Duplicate completed ablation cell: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError("No completed ablation rows were found.")

    baseline_keys = {key[:-1] for key in indexed if key[-1] == "ablation_full"}
    if not baseline_keys:
        raise ValueError("Ablation results contain no ablation_full reference configuration.")
    variants = sorted({key[-1] for key in indexed if key[-1] != "ablation_full"})
    if not variants:
        raise ValueError("Ablation results contain no variants to compare with the full configuration.")
    for variant in variants:
        variant_keys = {key[:-1] for key in indexed if key[-1] == variant}
        if variant_keys != baseline_keys:
            missing = baseline_keys - variant_keys
            extra = variant_keys - baseline_keys
            raise ValueError(
                f"Unpaired ablation variant {variant}: {len(missing)} missing and {len(extra)} extra cells."
            )

    result: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for dataset, pair, size, seed in sorted(baseline_keys):
        baseline = indexed[(dataset, pair, size, seed, "ablation_full")]
        baseline_metric = str(baseline.get("metric"))
        baseline_value = _finite_metric(baseline)
        for variant in variants:
            row = indexed[(dataset, pair, size, seed, variant)]
            if str(row.get("metric")) != baseline_metric:
                raise ValueError(
                    f"Metric mismatch in paired ablation {dataset}/{pair}/{size}/seed_{seed}: "
                    f"{baseline_metric} versus {row.get('metric')}"
                )
            result[(dataset, pair)][variant][size].append(_finite_metric(row) - baseline_value)
    return {
        target: {variant: dict(by_size) for variant, by_size in by_variant.items()}
        for target, by_variant in result.items()
    }


def plot_ablation_deltas(
    deltas: dict[tuple[str, str], dict[str, dict[str, list[float]]]],
    *,
    output: Path,
) -> Path:
    if not deltas:
        raise ValueError("Cannot plot empty ablation deltas.")
    targets = sorted(deltas)
    plt = get_pyplot("single")
    fig, axes = plt.subplots(
        1, len(targets), squeeze=False,
        figsize=(max(5.2, 4.15 * len(targets)), 3.8), sharex=False, sharey=True,
    )
    axes_flat = list(axes.flat)
    for target_index, target in enumerate(targets):
        ax = axes_flat[target_index]
        by_variant = deltas[target]
        variants = sorted(by_variant, key=lambda value: (ABLATION_LABELS.get(value, value), value))
        for y, variant in enumerate(variants):
            for size, offset, color, marker in (
                ("small", -0.11, BLUE, "o"), ("large", 0.11, ORANGE, "s")
            ):
                values = by_variant[variant].get(size, [])
                if not values:
                    continue
                mean = sum(values) / len(values)
                ax.plot([min(values), max(values)], [y + offset, y + offset], color=color, alpha=0.35, linewidth=1.7)
                ax.scatter(values, [y + offset] * len(values), color=color, alpha=0.30, s=18, marker=marker, linewidth=0)
                ax.scatter(
                    [mean], [y + offset], color=color, s=54, marker=marker,
                    edgecolor="white", linewidth=0.8,
                    label=("InherAct-Lite mean" if size == "small" else "InherAct mean")
                    if target_index == 0 and y == 0 else None,
                    zorder=4,
                )
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0, zorder=0)
        ax.set_yticks(range(len(variants)), [ABLATION_LABELS.get(value, value) for value in variants])
        ax.invert_yaxis()
        ax.set_title(f"{target[0]}\n{target[1]}", fontsize=10.5, pad=8)
        ax.set_xlabel("Change from full configuration (validation-metric points)")
        ax.grid(True, axis="x", alpha=0.7)
        ax.grid(False, axis="y")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.22, top=0.78, wspace=0.28)
    return _save_png(fig, output, plt)


def _filter_rows(
    rows: Sequence[dict[str, Any]],
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        canonicalize_metadata(row)
        for row in rows
        if (not datasets or row.get("dataset") in datasets)
        and (not seeds or int(row.get("seed")) in seeds)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="figure", required=True)
    search_parser = subparsers.add_parser("search", help="Plot normalized hyperparameter-search ranks.")
    search_parser.add_argument("log_root", type=Path)
    search_parser.add_argument("--stage", required=True, choices=SEARCH_STAGES)
    search_parser.add_argument("--dataset", action="append", default=[], help="Restrict targets; repeat as needed.")
    search_parser.add_argument("--seed", action="append", type=int, default=[], help="Restrict seeds; repeat as needed.")
    search_parser.add_argument("--output", required=True, type=Path)
    ablation_parser = subparsers.add_parser("ablation", help="Plot paired ablation deltas from the full configuration.")
    ablation_parser.add_argument("log_root", type=Path)
    ablation_parser.add_argument("--dataset", action="append", default=[], help="Restrict targets; repeat as needed.")
    ablation_parser.add_argument("--seed", action="append", type=int, default=[], help="Restrict seeds; repeat as needed.")
    ablation_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _filter_rows(build_rows(args.log_root), datasets=args.dataset, seeds=args.seed)
    try:
        if args.figure == "search":
            if args.stage in {"optimization", "distillation"} and not args.dataset:
                raise ValueError(
                    f"{args.stage} settings are selected by training family; pass --dataset to plot one "
                    "prespecified decision group instead of pooling unrelated families."
                )
            targets = [
                target for target in STAGE_TARGETS[args.stage]
                if not args.dataset or target[0] in args.dataset
            ]
            if not targets:
                raise ValueError("No prespecified search target matches the dataset filter.")
            seeds = args.seed or list(DEFAULT_SEARCH_SEEDS)
            expected = _expected_search_cells(args.stage, targets, seeds)
            scoped_rows = [
                row for row in rows
                if (row.get("dataset"), row.get("pair")) in set(targets)
                and int(row.get("seed")) in seeds
                and row.get("method") == "inheract"
                and row.get("size") == "large"
            ]
            validate_search_protocol_rows(scoped_rows)
            scores, _ = prepare_search_scores(
                scoped_rows, stage=args.stage, expected_cells=expected
            )
            output = plot_search_scores(scores, stage=args.stage, output=args.output)
        else:
            deltas = prepare_ablation_deltas(rows)
            output = plot_ablation_deltas(deltas, output=args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote publication figure: {output}")


if __name__ == "__main__":
    main()
