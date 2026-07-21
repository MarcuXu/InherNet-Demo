#!/usr/bin/env python3
"""Rank complete search matrices without averaging incompatible task metrics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.summarize_search import build_rows


DEFAULT_SEARCH_SEEDS = (42, 123, 2026)
EXPECTED_EPOCHS = {
    "cifar10": 200,
    "cifar100": 240,
    "oxford_pets": 30,
    "glue_sst2": 4,
    "glue_stsb": 4,
}
EXPECTED_EVAL_SPLIT = {
    "cifar10": "validation",
    "cifar100": "validation",
    "oxford_pets": "validation",
    "glue_sst2": "train_holdout",
    "glue_stsb": "train_holdout",
}
STAGE_CONFIG = {
    "mechanism": PROJECT_DIR / "configs/hetero_search_candidates.csv",
    "optimization": PROJECT_DIR / "configs/lr_scale_search_candidates.csv",
    "distillation": PROJECT_DIR / "configs/distillation_search_candidates.csv",
    "confirmation": PROJECT_DIR / "configs/hetero_confirmation_candidates.csv",
}
STAGE_TARGETS = {
    "mechanism": (
        ("cifar10", "resnet50_to_resnet18"),
        ("cifar100", "resnet56_to_resnet20"),
        ("oxford_pets", "resnet34_to_resnet18"),
        ("glue_sst2", "bert4_to_bert2"),
        ("glue_stsb", "bert4_to_bert2"),
    ),
    "optimization": (
        ("cifar10", "resnet50_to_resnet18"),
        ("cifar100", "resnet56_to_resnet20"),
        ("oxford_pets", "resnet34_to_resnet18"),
        ("glue_sst2", "bert4_to_bert2"),
        ("glue_stsb", "bert4_to_bert2"),
    ),
    "distillation": (
        ("cifar10", "resnet50_to_resnet18"),
        ("oxford_pets", "resnet34_to_resnet18"),
        ("glue_sst2", "bert4_to_bert2"),
        ("glue_stsb", "bert4_to_bert2"),
    ),
    "confirmation": (
        ("cifar10", "resnet50_to_resnet18"),
        ("cifar100", "resnet56_to_resnet20"),
        ("oxford_pets", "resnet34_to_resnet18"),
        ("glue_sst2", "bert4_to_bert2"),
        ("glue_stsb", "bert4_to_bert2"),
    ),
}


def candidate_ids(stage: str) -> list[str]:
    with STAGE_CONFIG[stage].open(newline="", encoding="utf-8") as handle:
        return list(dict.fromkeys(
            f"{stage}_{row['candidate_id']}" for row in csv.DictReader(handle)
        ))


def validate_search_protocol_rows(rows: list[dict[str, object]]) -> None:
    metrics_by_target: dict[tuple[object, object], set[object]] = defaultdict(set)
    for row in rows:
        dataset = str(row["dataset"])
        if row.get("size") != "large":
            raise ValueError(f"Search row is not headline Hetero capacity: {row.get('log')}")
        if int(row.get("epochs_completed") or 0) != EXPECTED_EPOCHS[dataset]:
            raise ValueError(f"Wrong search horizon for {row.get('log')}")
        if row.get("eval_split") != EXPECTED_EVAL_SPLIT[dataset]:
            raise ValueError(f"Wrong selection split for {row.get('log')}")
        try:
            metric_value = float(row["best_validation_metric"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing numeric search metric for {row.get('log')}") from exc
        if not math.isfinite(metric_value):
            raise ValueError(f"Non-finite search metric for {row.get('log')}")
        metrics_by_target[(row["dataset"], row["pair"])].add(row.get("metric"))
    inconsistent = [target for target, metrics in metrics_by_target.items() if len(metrics) != 1]
    if inconsistent:
        raise ValueError(f"Inconsistent primary metrics for search targets: {inconsistent}")


def applicable_candidates(stage: str, dataset: str, candidates: list[str]) -> list[str]:
    if stage != "distillation":
        return candidates
    result = []
    for candidate in candidates:
        short_name = candidate.removeprefix("distillation_")
        if dataset == "glue_stsb" and short_name.startswith("temperature_"):
            continue
        if dataset != "cifar10" and short_name == "registered_reference":
            continue
        result.append(candidate)
    return result


def average_normalized_ranks(rows: list[dict[str, object]]) -> dict[str, float]:
    ranked = sorted(rows, key=lambda row: -float(row["best_validation_metric"]))
    denominator = max(len(ranked) - 1, 1)
    scores: dict[str, float] = {}
    start = 0
    while start < len(ranked):
        end = start + 1
        value = float(ranked[start]["best_validation_metric"])
        while end < len(ranked) and float(ranked[end]["best_validation_metric"]) == value:
            end += 1
        average_rank = (start + end - 1) / 2
        for row in ranked[start:end]:
            scores[str(row["candidate"])] = 1.0 - average_rank / denominator
        start = end
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_CONFIG))
    parser.add_argument("--dataset", action="append", default=[], help="Restrict targets; repeat as needed.")
    parser.add_argument("--seed", action="append", type=int, default=[], help="Restrict seeds; repeat as needed.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Restrict candidates; repeat as needed and omit the stage prefix if desired.",
    )
    args = parser.parse_args()
    if args.stage in {"optimization", "distillation"} and not args.dataset:
        raise SystemExit(
            f"{args.stage} settings are selected by training family; pass --dataset."
        )

    available_candidates = candidate_ids(args.stage)
    requested_candidates = [
        candidate if candidate.startswith(f"{args.stage}_") else f"{args.stage}_{candidate}"
        for candidate in args.candidate
    ]
    selected_candidates = requested_candidates or available_candidates
    unknown = sorted(set(selected_candidates) - set(available_candidates))
    if unknown:
        raise SystemExit(f"Unknown {args.stage} candidates: {', '.join(unknown)}")

    targets = [
        target
        for target in STAGE_TARGETS[args.stage]
        if not args.dataset or target[0] in args.dataset
    ]
    if not targets:
        raise SystemExit("No prespecified targets match the dataset filter.")
    target_set = set(targets)
    methods = ("hetero",)

    all_rows = build_rows(args.log_root)
    seeds = args.seed or list(DEFAULT_SEARCH_SEEDS)

    rows = [
        row
        for row in all_rows
        if row["stage"] == args.stage
        and (row["dataset"], row["pair"]) in target_set
        and row["method"] in methods
        and int(row["seed"]) in seeds
        and row["candidate"] in selected_candidates
    ]
    try:
        validate_search_protocol_rows(rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    indexed: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["dataset"], row["pair"], row["method"], row["size"], row["seed"], row["candidate"]
        )
        if key in indexed:
            raise SystemExit(f"Duplicate completed search cell: {key}")
        indexed[key] = row

    expected: set[tuple[object, ...]] = set()
    for dataset, pair in targets:
        applicable_candidates_for_target = applicable_candidates(
            args.stage, dataset, selected_candidates
        )
        for method in methods:
            for size in ("large",):
                for seed in seeds:
                    for candidate in applicable_candidates_for_target:
                        expected.add((dataset, pair, method, size, seed, candidate))
    missing = sorted(expected - set(indexed))
    if not expected:
        raise SystemExit("No requested candidate applies to the selected targets.")
    if missing:
        preview = "; ".join("/".join(map(str, cell)) for cell in missing[:5])
        raise SystemExit(
            f"Incomplete {args.stage} matrix: missing {len(missing)} of {len(expected)} cells. "
            f"First missing: {preview}"
        )

    by_cell: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for key in sorted(expected):
        row = indexed[key]
        by_cell[key[:-1]].append(row)

    scores: dict[str, list[float]] = defaultdict(list)
    for cell_rows in by_cell.values():
        for candidate, score in average_normalized_ranks(cell_rows).items():
            scores[candidate].append(score)

    print("candidate,mean_normalized_rank,coverage")
    for candidate, values in sorted(
        scores.items(),
        key=lambda item: (-sum(item[1]) / len(item[1]), item[0]),
    ):
        print(f"{candidate},{sum(values) / len(values):.6f},{len(values)}")


if __name__ == "__main__":
    main()
