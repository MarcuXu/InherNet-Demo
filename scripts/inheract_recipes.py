#!/usr/bin/env python3
"""Validate and resolve manually reviewed InherAct experiment recipes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model_wrappers import FINAL_INHERACT_ALLOCATION


SELECTED_PATH = PROJECT_DIR / "configs/inheract_selected_recipes.csv"
REFERENCE_PATH = PROJECT_DIR / "configs/inheract_reference_recipes.csv"
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
    raise ValueError(f"No InherAct recipe profile for dataset: {dataset}")


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
        raise ValueError(f"Unknown InherAct recipe profile: {row['profile']}")
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
    if row["allocation_scale"] != FINAL_INHERACT_ALLOCATION:
        raise ValueError(
            "Selected and reference recipes must use the frozen InherAct mechanism "
            f"{FINAL_INHERACT_ALLOCATION!r}; research controls belong only in pre-study."
        )
    if mode not in {"supervised", "distillation"}:
        raise ValueError(f"Unknown InherAct train mode: {mode}")
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
            raise ValueError(f"Duplicate selected InherAct profile: {profile}")
        selected[profile] = row
    missing = sorted(set(PROFILES) - set(selected))
    if missing:
        raise ValueError(f"Selected InherAct recipes are missing profiles: {', '.join(missing)}")
    return selected


def load_reference(path: Path = REFERENCE_PATH) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reference: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("recipe_id", "").strip() != "weighted_uniform":
            raise ValueError("Every reference recipe requires recipe_id=weighted_uniform.")
        validate_recipe(row)
        profile = row["profile"]
        if profile in reference:
            raise ValueError(f"Duplicate reference InherAct profile: {profile}")
        reference[profile] = row
    missing = sorted(set(PROFILES) - set(reference))
    if missing:
        raise ValueError(f"Reference InherAct recipes are missing profiles: {', '.join(missing)}")
    return reference


def load_reference_recipe(dataset: str) -> dict[str, str]:
    return load_reference()[dataset_profile(dataset)]


def recipe_arguments(row: dict[str, str]) -> list[str]:
    args = [
        "--inheract-recipe-id", row["recipe_id"],
        *mechanism_arguments(row),
        *optimizer_arguments(row),
        *objective_arguments(row),
    ]
    return args


def mechanism_arguments(row: dict[str, str]) -> list[str]:
    return [
        "--aux-loss-weight", row["aux_loss_weight"],
        "--inheract-second-moment-shrinkage", row["second_moment_shrinkage"],
        "--inheract-expert-noise-scale", row["expert_noise_scale"],
        "--inheract-allocation-scale", row["allocation_scale"],
    ]


def optimizer_arguments(row: dict[str, str]) -> list[str]:
    return ["--lr-scale", row["lr_scale"]]


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
        "selected-optimizer",
        "selected-supervised",
        "reference-objective",
        "reference-mechanism",
        "reference-optimizer",
    ):
        selected_parser = subparsers.add_parser(command)
        selected_parser.add_argument("dataset")
    args = parser.parse_args()
    row = (
        load_reference_recipe(args.dataset)
        if args.command.startswith("reference-")
        else load_selected()[dataset_profile(args.dataset)]
    )
    if args.command in {"selected-objective", "reference-objective"}:
        resolved = objective_arguments(row)
    elif args.command == "selected-optimizer":
        resolved = optimizer_arguments(row)
    elif args.command == "reference-mechanism":
        resolved = mechanism_arguments(row)
    elif args.command == "reference-optimizer":
        resolved = optimizer_arguments(row)
    elif args.command == "selected-supervised":
        resolved = supervised_control_arguments(row)
    else:
        resolved = recipe_arguments(row)
    for argument in resolved:
        print(argument)


if __name__ == "__main__":
    main()
