from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from experiment_registry import TrainSettings


TEACHER_CHECKPOINT_SCHEMA_VERSION = 2


def _teacher_settings_payload(settings: TrainSettings) -> dict[str, Any]:
    """Return teacher-relevant settings with legacy defaults normalized."""
    settings_payload = asdict(settings)
    for target_only_key in (
        "kd_temperature",
        "kd_loss_weight",
        "ce_loss_weight",
        "default_head_num",
    ):
        settings_payload.pop(target_only_key, None)
    if (
        settings.scheduler_name == "multistep"
        and settings.warmup_ratio == 0.0
        and settings.max_grad_norm == 0.0
        and not settings.exclude_bias_norm_from_weight_decay
    ):
        for legacy_default_key in (
            "scheduler_name",
            "warmup_ratio",
            "max_grad_norm",
            "exclude_bias_norm_from_weight_decay",
        ):
            settings_payload.pop(legacy_default_key, None)
    return settings_payload


def teacher_training_fingerprint(
    *,
    dataset: str,
    pair: str,
    architecture: str,
    num_classes: int,
    seed: int,
    settings: TrainSettings,
    model_profile: str,
    data_profile: str,
    data_split: Mapping[str, Any] | None = None,
) -> str:
    settings_payload = _teacher_settings_payload(settings)
    payload = {
        "dataset": dataset,
        "pair": pair,
        "architecture": architecture,
        "num_classes": num_classes,
        "seed": seed,
        "teacher_train_settings": settings_payload,
        "model_profile": model_profile,
        "data_profile": data_profile,
        "data_split": dict(data_split) if data_split is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_teacher_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    dataset: str,
    pair: str,
    architecture: str,
    num_classes: int,
    seed: int,
    settings: TrainSettings,
    model_profile: str,
    data_profile: str,
    selection_policy: str,
    selected_epoch: int,
    metrics: Mapping[str, Any],
    data_split: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not 1 <= selected_epoch <= settings.epochs:
        raise ValueError(
            f"selected_epoch must be within [1, {settings.epochs}], got {selected_epoch}."
        )
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"Teacher checkpoint already exists: {checkpoint_path}. "
            "Pass --overwrite-teacher-checkpoint to replace it."
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = teacher_training_fingerprint(
        dataset=dataset,
        pair=pair,
        architecture=architecture,
        num_classes=num_classes,
        seed=seed,
        settings=settings,
        model_profile=model_profile,
        data_profile=data_profile,
        data_split=data_split,
    )
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    non_finite_state = [
        name for name, tensor in state_dict.items()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all()
    ]
    if non_finite_state:
        raise ValueError(f"Teacher state contains non-finite tensors: {', '.join(non_finite_state)}")
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"Teacher metric '{name}' is non-finite: {value}")
    payload = {
        "schema_version": TEACHER_CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "trained_teacher",
        "dataset": dataset,
        "pair": pair,
        "architecture": architecture,
        "num_classes": num_classes,
        "seed": seed,
        "train_settings": asdict(settings),
        "model_profile": model_profile,
        "data_profile": data_profile,
        "data_split": dict(data_split) if data_split is not None else None,
        "training_fingerprint": fingerprint,
        "selection_policy": selection_policy,
        "selected_epoch": selected_epoch,
        "metrics": dict(metrics),
        "model_state_dict": state_dict,
    }
    temporary_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": file_sha256(checkpoint_path),
        "schema_version": TEACHER_CHECKPOINT_SCHEMA_VERSION,
        "training_fingerprint": fingerprint,
        "selection_policy": selection_policy,
        "selected_epoch": selected_epoch,
    }


def load_teacher_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    dataset: str,
    pair: str,
    architecture: str,
    num_classes: int,
    seed: int,
    model_profile: str,
    data_profile: str,
    expected_settings: TrainSettings,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load teacher checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Teacher checkpoint root must be a mapping.")
    expected = {
        "schema_version": TEACHER_CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "trained_teacher",
        "dataset": dataset,
        "pair": pair,
        "architecture": architecture,
        "num_classes": num_classes,
        "seed": seed,
        "model_profile": model_profile,
        "data_profile": data_profile,
    }
    mismatches = {
        key: (payload.get(key), expected_value)
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual!r}, expected={wanted!r}"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"Teacher checkpoint is incompatible ({details}).")
    saved_settings = payload.get("train_settings")
    try:
        checkpoint_settings = TrainSettings(**saved_settings)
    except (TypeError, ValueError) as exc:
        raise ValueError("Teacher checkpoint contains invalid train_settings.") from exc
    expected_fingerprint = teacher_training_fingerprint(
        dataset=dataset,
        pair=pair,
        architecture=architecture,
        num_classes=num_classes,
        seed=seed,
        settings=checkpoint_settings,
        model_profile=model_profile,
        data_profile=data_profile,
        data_split=payload.get("data_split"),
    )
    if payload.get("training_fingerprint") != expected_fingerprint:
        mismatches["training_fingerprint"] = (
            payload.get("training_fingerprint"),
            expected_fingerprint,
        )
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual!r}, expected={wanted!r}"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"Teacher checkpoint is incompatible ({details}).")
    if _teacher_settings_payload(checkpoint_settings) != _teacher_settings_payload(expected_settings):
        raise ValueError(
            "Teacher checkpoint is incompatible (saved teacher training settings do not match "
            "the current registered protocol)."
        )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Teacher checkpoint is missing a model_state_dict mapping.")
    invalid_state = [
        name for name, tensor in state_dict.items()
        if not isinstance(tensor, torch.Tensor)
        or (tensor.is_floating_point() and not torch.isfinite(tensor).all())
    ]
    if invalid_state:
        raise ValueError(f"Teacher checkpoint has invalid state tensors: {', '.join(invalid_state)}")
    selected_epoch = int(payload.get("selected_epoch", 0))
    if not 1 <= selected_epoch <= checkpoint_settings.epochs:
        raise ValueError("Teacher checkpoint has an invalid selected_epoch.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": file_sha256(checkpoint_path),
        "schema_version": payload["schema_version"],
        "training_fingerprint": payload["training_fingerprint"],
        "selection_policy": payload.get("selection_policy", "unknown"),
        "selected_epoch": selected_epoch,
        "metrics": dict(payload.get("metrics", {})),
        "data_split": (
            dict(payload["data_split"])
            if isinstance(payload.get("data_split"), Mapping)
            else None
        ),
    }
