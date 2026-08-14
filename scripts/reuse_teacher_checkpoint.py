#!/usr/bin/env python3
"""Validate and atomically snapshot a compatible teacher checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any, Mapping, Sequence

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from checkpointing import load_teacher_checkpoint
from experiment_registry import (
    DATASET_REGISTRY,
    build_pair_model,
    get_pair_spec,
    get_role_name,
)
from glue_data import GLUE_DATASET, GLUE_DATASET_REVISION


def expected_evaluation_split(dataset: str, *, search_validation: bool) -> str:
    dataset_spec = DATASET_REGISTRY[dataset]
    if dataset.startswith("glue_") and search_validation:
        return "train_holdout"
    if dataset in {"cifar10", "cifar100"} and search_validation:
        return "validation"
    return dataset_spec.eval_split_name


def checkpoint_matches_protocol(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    pair: str,
    seed: int,
    search_validation: bool,
) -> tuple[bool, str]:
    """Check identity and the teacher-training/selection split contract."""
    dataset_spec = DATASET_REGISTRY[dataset]
    pair_spec = get_pair_spec(dataset, pair)
    expected_identity = {
        "dataset": dataset,
        "pair": pair,
        "seed": seed,
        "architecture": get_role_name(pair_spec, "teacher"),
        "num_classes": dataset_spec.num_classes,
        "model_profile": str(pair_spec.get("model_profile", "unspecified")),
        "data_profile": dataset_spec.data_profile,
    }
    mismatches = [
        f"{key}={payload.get(key)!r} (expected {value!r})"
        for key, value in expected_identity.items()
        if payload.get(key) != value
    ]
    if mismatches:
        return False, "identity mismatch: " + ", ".join(mismatches)

    split = payload.get("data_split")
    if not isinstance(split, Mapping):
        return False, "missing data_split metadata"

    expected_eval = expected_evaluation_split(
        dataset,
        search_validation=search_validation,
    )
    expected_policy = f"best_{expected_eval}_{dataset_spec.primary_metric_name}"
    if payload.get("selection_policy") != expected_policy:
        return (
            False,
            f"selection_policy={payload.get('selection_policy')!r} "
            f"(expected {expected_policy!r})",
        )

    if dataset.startswith("glue_"):
        expected_train = "train_subset" if search_validation else "train"
        required = {
            "profile": "huggingface_glue",
            "dataset": GLUE_DATASET,
            "dataset_revision": GLUE_DATASET_REVISION,
            "task": dataset_spec.text_task_name,
            "train_split": expected_train,
            "evaluation_split": expected_eval,
            "max_length": dataset_spec.text_max_length,
            "tokenizer": str(pair_spec["tokenizer_name"]),
            "tokenizer_revision": str(pair_spec["tokenizer_revision"]),
            "teacher_revision": str(pair_spec["teacher_revision"]),
        }
        if search_validation:
            required.update(
                {
                    "selection_split_seed": dataset_spec.validation_split_seed,
                    "selection_validation_fraction": 0.1,
                }
            )
    else:
        validation_fraction = (
            0.1
            if dataset in {"cifar10", "cifar100"} and search_validation
            else dataset_spec.validation_fraction
        )
        if validation_fraction <= 0:
            required = {"profile": "official_train"}
        else:
            required = {
                "profile": "fixed_stratified_holdout",
                "seed": dataset_spec.validation_split_seed,
                "validation_fraction": validation_fraction,
            }

    split_mismatches = [
        f"{key}={split.get(key)!r} (expected {value!r})"
        for key, value in required.items()
        if split.get(key) != value
    ]
    if split_mismatches:
        return False, "training split mismatch: " + ", ".join(split_mismatches)
    return True, "compatible"


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot read checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root is not a mapping")
    return payload


def _strict_validate_checkpoint(
    path: Path,
    *,
    dataset: str,
    pair: str,
    seed: int,
) -> None:
    dataset_spec = DATASET_REGISTRY[dataset]
    pair_spec = get_pair_spec(dataset, pair)
    model = build_pair_model(
        dataset,
        pair,
        "teacher",
        dataset_spec.num_classes,
        initialize_pretrained=False,
    )
    load_teacher_checkpoint(
        path,
        model,
        dataset=dataset,
        pair=pair,
        architecture=get_role_name(pair_spec, "teacher"),
        num_classes=dataset_spec.num_classes,
        seed=seed,
        model_profile=str(pair_spec.get("model_profile", "unspecified")),
        data_profile=dataset_spec.data_profile,
        expected_settings=dataset_spec.train_settings,
    )


def find_compatible_checkpoint(
    candidates: Sequence[Path],
    *,
    dataset: str,
    pair: str,
    seed: int,
    search_validation: bool,
) -> Path | None:
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = _load_payload(candidate)
            compatible, reason = checkpoint_matches_protocol(
                payload,
                dataset=dataset,
                pair=pair,
                seed=seed,
                search_validation=search_validation,
            )
            if not compatible:
                print(f"Ignoring incompatible teacher {candidate}: {reason}", file=sys.stderr)
                continue
            _strict_validate_checkpoint(
                candidate,
                dataset=dataset,
                pair=pair,
                seed=seed,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            print(f"Ignoring incompatible teacher {candidate}: {exc}", file=sys.stderr)
            continue
        return candidate.resolve()
    return None


def snapshot_checkpoint(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(sorted(DATASET_REGISTRY)), required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--search-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()

    source = find_compatible_checkpoint(
        args.candidates,
        dataset=args.dataset,
        pair=args.pair,
        seed=args.seed,
        search_validation=args.search_validation,
    )
    if source is None:
        raise SystemExit(1)
    if not args.dry_run:
        snapshot_checkpoint(source, args.destination.resolve())
    print(source)


if __name__ == "__main__":
    main()
