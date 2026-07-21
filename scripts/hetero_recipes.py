#!/usr/bin/env python3
"""Validate and resolve manually reviewed Hetero experiment recipes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model_wrappers import FINAL_HETERO_ALLOCATION


SELECTED_PATH = PROJECT_DIR / "configs/hetero_selected_recipes.csv"
CONFIRMATION_PATH = PROJECT_DIR / "configs/hetero_confirmation_candidates.csv"
REGISTERED_REFERENCE_ID = "weighted_uniform"
PROFILES = (
    "cifar10",
    "cifar100",
    "oxford_pets",
    "glue_classification",
    "glue_regression",
)
RECIPE_FIELDS = (
    "aux_loss_weight",
    "second_moment_shrinkage",
    "expert_noise_scale",
    "allocation_scale",
    "lr_scale",
    "train_mode",
    "kd_temperature",
    "kd_fraction",
)
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def dataset_profile(dataset: str) -> str:
    if dataset in {"cifar10", "cifar100", "oxford_pets"}:
        return dataset
    if dataset == "glue_stsb":
        return "glue_regression"
    if dataset.startswith("glue_"):
        return "glue_classification"
    raise ValueError(f"No Hetero recipe profile for dataset: {dataset}")


def _number(row: dict[str, str], field: str, *, allow_blank: bool = False) -> float | None:
    value = row[field].strip()
    if allow_blank and not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}={value!r} for profile {row.get('profile')}") from exc


def validate_recipe(row: dict[str, str]) -> None:
    missing = [field for field in ("profile", *RECIPE_FIELDS) if field not in row]
    if missing:
        raise ValueError(f"Recipe is missing columns: {', '.join(missing)}")
    if row["profile"] not in PROFILES:
        raise ValueError(f"Unknown Hetero recipe profile: {row['profile']}")
    aux = _number(row, "aux_loss_weight")
    shrinkage = _number(row, "second_moment_shrinkage")
    noise = _number(row, "expert_noise_scale")
    lr_scale = _number(row, "lr_scale")
    kd_temperature = _number(row, "kd_temperature", allow_blank=True)
    kd_fraction = _number(row, "kd_fraction", allow_blank=True)
    if aux is None or aux < 0 or noise is None or noise < 0:
        raise ValueError("Auxiliary weight and expert noise must be non-negative.")
    if shrinkage is None or not 0 <= shrinkage <= 1:
        raise ValueError("Second-moment shrinkage must be in [0, 1].")
    if lr_scale is None or lr_scale <= 0:
        raise ValueError("Learning-rate scale must be positive.")
    if kd_fraction is not None and not 0 <= kd_fraction <= 1:
        raise ValueError("KD fraction must be in [0, 1].")
    mode = row["train_mode"]
    if row["allocation_scale"] != FINAL_HETERO_ALLOCATION:
        raise ValueError(
            "Selected and confirmation recipes must use the frozen Hetero mechanism "
            f"{FINAL_HETERO_ALLOCATION!r}; research controls belong only in pre-study."
        )
    if mode not in {"supervised", "distillation"}:
        raise ValueError(f"Unknown Hetero train mode: {mode}")
    if mode == "distillation" and (kd_temperature is None or kd_temperature <= 0):
        raise ValueError("Distillation recipes require a positive KD temperature.")
    if mode == "supervised" and (kd_temperature is not None or kd_fraction is not None):
        raise ValueError("Supervised recipes must leave KD fields blank.")


def load_selected(path: Path = SELECTED_PATH) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if not IDENTIFIER.fullmatch(row.get("recipe_id", "").strip()):
            raise ValueError("Every selected recipe requires a recipe_id.")
        validate_recipe(row)
        profile = row["profile"]
        if profile in selected:
            raise ValueError(f"Duplicate selected Hetero profile: {profile}")
        selected[profile] = row
    missing = sorted(set(PROFILES) - set(selected))
    if missing:
        raise ValueError(f"Selected Hetero recipes are missing profiles: {', '.join(missing)}")
    return selected


def load_confirmation(path: Path = CONFIRMATION_PATH) -> dict[str, dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        candidate = row.get("candidate_id", "").strip()
        if not IDENTIFIER.fullmatch(candidate):
            raise ValueError("Every confirmation recipe requires a candidate_id.")
        validate_recipe(row)
        by_profile = candidates.setdefault(candidate, {})
        profile = row["profile"]
        if profile in by_profile:
            raise ValueError(f"Duplicate confirmation row: {candidate}/{profile}")
        by_profile[profile] = row
    for candidate, by_profile in candidates.items():
        missing = sorted(set(PROFILES) - set(by_profile))
        if missing:
            raise ValueError(
                f"Confirmation candidate {candidate} is missing profiles: {', '.join(missing)}"
            )
    return candidates


def load_registered_reference(dataset: str) -> dict[str, str]:
    return load_confirmation()[REGISTERED_REFERENCE_ID][dataset_profile(dataset)]


def recipe_arguments(row: dict[str, str]) -> list[str]:
    args = [
        "--hetero-recipe-id", row.get("recipe_id") or row["candidate_id"],
        "--aux-loss-weight", row["aux_loss_weight"],
        "--hetero-second-moment-shrinkage", row["second_moment_shrinkage"],
        "--hetero-expert-noise-scale", row["expert_noise_scale"],
        "--hetero-allocation-scale", row["allocation_scale"],
        "--lr-scale", row["lr_scale"],
        "--compressed-train-mode", row["train_mode"],
    ]
    if row["train_mode"] == "distillation":
        args.extend(("--kd-temperature", row["kd_temperature"]))
        if row["kd_fraction"].strip():
            args.extend(("--kd-fraction", row["kd_fraction"]))
    return args


def objective_arguments(row: dict[str, str]) -> list[str]:
    args = ["--compressed-train-mode", row["train_mode"]]
    if row["train_mode"] == "distillation":
        args.extend(("--kd-temperature", row["kd_temperature"]))
        if row["kd_fraction"].strip():
            args.extend(("--kd-fraction", row["kd_fraction"]))
    return args


def supervised_control_arguments(row: dict[str, str]) -> list[str]:
    control = dict(row)
    control["recipe_id"] = f"{row['recipe_id']}_supervised_control"
    control["train_mode"] = "supervised"
    control["kd_temperature"] = ""
    control["kd_fraction"] = ""
    return recipe_arguments(control)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "selected",
        "selected-objective",
        "selected-supervised",
        "registered-objective",
    ):
        selected_parser = subparsers.add_parser(command)
        selected_parser.add_argument("dataset")
    subparsers.add_parser("validate-confirmation")
    args = parser.parse_args()
    if args.command != "validate-confirmation":
        row = (
            load_registered_reference(args.dataset)
            if args.command == "registered-objective"
            else load_selected()[dataset_profile(args.dataset)]
        )
        if args.command in {"selected-objective", "registered-objective"}:
            resolved = objective_arguments(row)
        elif args.command == "selected-supervised":
            resolved = supervised_control_arguments(row)
        else:
            resolved = recipe_arguments(row)
        for argument in resolved:
            print(argument)
    else:
        candidates = load_confirmation()
        print(len(candidates))


if __name__ == "__main__":
    main()
