from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from contrastive_distillation import CRDDistiller
from experiment_registry import (
    CurriculumTemperatureDistillationSettings,
    DecoupledDistillationSettings,
    LogitStandardizedKDSettings,
    TrainSettings,
)
from model_wrappers import clear_gating_router_cache, compute_gating_load_balance_loss
from vision_distillation import CATKDDistiller, ReviewKDDistiller, SimKDDistiller


RUN_LOG_ENV_VAR = "INHERNET_RUN_LOG"
RUN_METADATA_PREFIX = "RUN_METADATA"
RUN_METRICS_PREFIX = "RUN_METRICS"
RUN_SUMMARY_PREFIX = "RUN_SUMMARY"
RUN_FINAL_TEST_PREFIX = "RUN_FINAL_TEST"
INHERITANCE_DIAGNOSTICS_PREFIX = "INHERITANCE_DIAGNOSTICS"


class _CTKDGradientReversal(torch.autograd.Function):
    """Identity in the forward pass; multiply the temperature gradient by lambda."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, gradient_scale: float) -> torch.Tensor:
        ctx.gradient_scale = float(gradient_scale)
        return value.clone()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient * ctx.gradient_scale, None


class GlobalCurriculumTemperature(nn.Module):
    """CTKD's released one-parameter global-temperature module."""

    def __init__(self, settings: CurriculumTemperatureDistillationSettings) -> None:
        super().__init__()
        self.raw_temperature = nn.Parameter(torch.ones(1))
        self.t_start = float(settings.t_start)
        self.t_end = float(settings.t_end)

    def forward(self, gradient_scale: float) -> torch.Tensor:
        raw_temperature = _CTKDGradientReversal.apply(
            self.raw_temperature,
            gradient_scale,
        )
        return self.t_start + self.t_end * torch.sigmoid(raw_temperature)

    def current_temperature(self) -> torch.Tensor:
        return self.t_start + self.t_end * torch.sigmoid(self.raw_temperature)


class RunLogger:
    def __init__(self, log_path: str | None = None, echo: bool = True, store_info_to_file: bool = True) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.echo = echo
        self.store_info_to_file = store_info_to_file

    def _write_line(self, message: str, *, echo: bool | None = None, write_to_file: bool = True) -> None:
        effective_echo = self.echo if echo is None else echo
        if write_to_file and self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{message}\n")
        if effective_echo:
            print(message)

    def info(self, message: str) -> None:
        self._write_line(message, write_to_file=self.store_info_to_file)

    def epoch(self, message: str) -> None:
        if self.log_path is None:
            self._write_line(message, echo=True, write_to_file=False)
            return
        self._write_line(message, echo=False, write_to_file=True)

    def structured(self, prefix: str, payload: Mapping[str, Any], *, echo: bool = False) -> None:
        self._write_line(f"{prefix} {json.dumps(dict(payload), sort_keys=True)}", echo=echo, write_to_file=True)

    def metadata(self, payload: Mapping[str, Any]) -> None:
        self.structured(RUN_METADATA_PREFIX, payload)

    def metrics(self, payload: Mapping[str, Any]) -> None:
        self.structured(RUN_METRICS_PREFIX, payload)


def build_run_logger(
    log_path: str | None = None,
    *,
    echo: bool = True,
    store_info_to_file: bool = True,
) -> RunLogger:
    resolved_path = os.environ.get(RUN_LOG_ENV_VAR) if log_path is None else log_path
    return RunLogger(resolved_path, echo=echo, store_info_to_file=store_info_to_file)


def to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    try:
        return [float(value) for value in list(values)]
    except TypeError:
        return []


def create_history_template() -> dict[str, list[float]]:
    return {
        "train_objective": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }


def normalize_history(history: Mapping[str, Any] | None) -> dict[str, list[float]]:
    raw_history = history or {}
    normalized = create_history_template()
    normalized["train_objective"] = to_float_list(raw_history.get("train_objective", raw_history.get("train_loss")))
    normalized["train_loss"] = to_float_list(raw_history.get("train_loss"))
    normalized["train_accuracy"] = to_float_list(raw_history.get("train_accuracy"))
    normalized["test_loss"] = to_float_list(raw_history.get("test_loss"))
    normalized["test_accuracy"] = to_float_list(raw_history.get("test_accuracy", raw_history.get("eval_accuracy")))
    return normalized


def move_batch_to_device(inputs: Any, device: torch.device):
    if isinstance(inputs, Mapping):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
    return inputs.to(device)


def forward_logits(model: nn.Module, inputs: Any) -> torch.Tensor:
    outputs = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
    if isinstance(outputs, torch.Tensor):
        return outputs
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Model output type {type(outputs)!r} does not expose logits.")


def build_task_criterion(problem_type: str) -> nn.Module:
    if problem_type == "classification":
        return nn.CrossEntropyLoss()
    if problem_type == "regression":
        return nn.MSELoss()
    raise ValueError(f"Unsupported problem type: {problem_type}")


def prepare_labels(labels: torch.Tensor, problem_type: str) -> torch.Tensor:
    return labels.float() if problem_type == "regression" else labels.long()


def prepare_regression_outputs(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim > 1 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits.reshape(-1)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _compute_classification_metric_values(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_labels: int,
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    predictions = predictions.detach().cpu().long().view(-1)
    labels = labels.detach().cpu().long().view(-1)
    total = int(labels.numel())
    metrics: dict[str, float] = {}
    if total == 0:
        return {metric_name: 0.0 for metric_name in metric_names}

    correct = int((predictions == labels).sum().item())
    if "accuracy" in metric_names:
        metrics["accuracy"] = 100.0 * correct / total

    recalls: list[float] = []
    f1_values: list[float] = []
    for class_idx in range(num_labels):
        pred_pos = predictions == class_idx
        label_pos = labels == class_idx
        true_pos = int((pred_pos & label_pos).sum().item())
        false_pos = int((pred_pos & ~label_pos).sum().item())
        false_neg = int((~pred_pos & label_pos).sum().item())
        support = int(label_pos.sum().item())
        if support > 0:
            recalls.append(_safe_divide(true_pos, true_pos + false_neg))
        precision = _safe_divide(true_pos, true_pos + false_pos)
        recall = _safe_divide(true_pos, true_pos + false_neg)
        f1_values.append(_safe_divide(2.0 * precision * recall, precision + recall))

    if "balanced_accuracy" in metric_names:
        metrics["balanced_accuracy"] = 100.0 * (sum(recalls) / len(recalls) if recalls else 0.0)
    if "macro_f1" in metric_names:
        metrics["macro_f1"] = 100.0 * (sum(f1_values) / len(f1_values) if f1_values else 0.0)
    if "f1" in metric_names:
        positive_label = 1 if num_labels > 1 else 0
        pred_pos = predictions == positive_label
        label_pos = labels == positive_label
        true_pos = int((pred_pos & label_pos).sum().item())
        false_pos = int((pred_pos & ~label_pos).sum().item())
        false_neg = int((~pred_pos & label_pos).sum().item())
        precision = _safe_divide(true_pos, true_pos + false_pos)
        recall = _safe_divide(true_pos, true_pos + false_neg)
        metrics["f1"] = 100.0 * _safe_divide(2.0 * precision * recall, precision + recall)
    if "matthews_correlation" in metric_names:
        if num_labels != 2:
            metrics["matthews_correlation"] = 0.0
        else:
            pred_pos = predictions == 1
            label_pos = labels == 1
            tp = int((pred_pos & label_pos).sum().item())
            tn = int((~pred_pos & ~label_pos).sum().item())
            fp = int((pred_pos & ~label_pos).sum().item())
            fn = int((~pred_pos & label_pos).sum().item())
            denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            metrics["matthews_correlation"] = 100.0 * _safe_divide(float(tp * tn - fp * fn), denominator)
    return metrics


def _rankdata_average(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().float().view(-1)
    if values.numel() == 0:
        return values
    sorted_indices = torch.argsort(values, stable=True)
    sorted_values = values[sorted_indices]
    ranks = torch.empty_like(values)
    start = 0
    count = int(values.numel())
    while start < count:
        end = start + 1
        while end < count and float(sorted_values[end].item()) == float(sorted_values[start].item()):
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[sorted_indices[start:end]] = average_rank
        start = end
    return ranks


def _pearson_correlation_percent(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = predictions.detach().cpu().float().view(-1)
    labels = labels.detach().cpu().float().view(-1)
    if predictions.numel() < 2:
        return 0.0
    pred_centered = predictions - predictions.mean()
    label_centered = labels - labels.mean()
    denominator = torch.linalg.vector_norm(pred_centered) * torch.linalg.vector_norm(label_centered)
    if float(denominator.item()) <= 0:
        return 0.0
    return 100.0 * float(torch.dot(pred_centered, label_centered).item() / denominator.item())


def _compute_regression_metric_values(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "pearson" in metric_names:
        metrics["pearson"] = _pearson_correlation_percent(predictions, labels)
    if "spearmanr" in metric_names:
        metrics["spearmanr"] = _pearson_correlation_percent(
            _rankdata_average(predictions),
            _rankdata_average(labels),
        )
    return metrics


def compute_task_metric_values(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    problem_type: str,
    num_labels: int,
    metric_names: tuple[str, ...],
) -> dict[str, float]:
    if problem_type == "classification":
        return _compute_classification_metric_values(
            predictions,
            labels,
            num_labels=num_labels,
            metric_names=metric_names,
        )
    if problem_type == "regression":
        return _compute_regression_metric_values(predictions, labels, metric_names=metric_names)
    raise ValueError(f"Unsupported problem type: {problem_type}")


def evaluate_task_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    problem_type: str = "classification",
    num_labels: int = 0,
    metric_names: tuple[str, ...] = ("accuracy",),
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = move_batch_to_device(inputs, device)
            labels = prepare_labels(labels.to(device), problem_type)
            logits = forward_logits(model, inputs)
            if problem_type == "regression":
                predictions = prepare_regression_outputs(logits)
                loss = criterion(predictions, labels.view_as(predictions))
                metric_labels = labels.view_as(predictions)
            else:
                loss = criterion(logits, labels)
                predictions = logits.argmax(dim=1)
                metric_labels = labels
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            all_predictions.append(predictions.detach().cpu())
            all_labels.append(metric_labels.detach().cpu())
    prediction_tensor = torch.cat(all_predictions) if all_predictions else torch.empty(0)
    label_tensor = torch.cat(all_labels) if all_labels else torch.empty(0)
    metrics = compute_task_metric_values(
        prediction_tensor,
        label_tensor,
        problem_type=problem_type,
        num_labels=max(num_labels, 1),
        metric_names=metric_names,
    )
    metrics["loss"] = running_loss / max(total, 1)
    return metrics


def evaluate_inheritance_diagnostics(
    teacher_model: nn.Module,
    inherited_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    problem_type: str,
    num_labels: int,
    metric_names: tuple[str, ...],
    evaluation_split: str = "validation",
    local_operator_max_batches: int = 4,
) -> dict[str, Any]:
    """Measure inherited-model fidelity before the first optimizer update."""
    teacher_was_training = teacher_model.training
    inherited_was_training = inherited_model.training
    teacher_model.eval()
    inherited_model.eval()
    criterion = build_task_criterion(problem_type)
    total = 0
    teacher_loss_sum = 0.0
    inherited_loss_sum = 0.0
    squared_error_sum = 0.0
    teacher_squared_sum = 0.0
    dot_sum = 0.0
    teacher_vector_squared_sum = 0.0
    inherited_vector_squared_sum = 0.0
    kl_sum = 0.0
    agreement_sum = 0
    labels_all: list[torch.Tensor] = []
    teacher_predictions: list[torch.Tensor] = []
    inherited_predictions: list[torch.Tensor] = []

    try:
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = move_batch_to_device(inputs, device)
                labels = prepare_labels(labels.to(device), problem_type)
                teacher_logits = forward_logits(teacher_model, inputs)
                inherited_logits = forward_logits(inherited_model, inputs)
                batch_size = int(labels.size(0))
                total += batch_size

                teacher_flat = teacher_logits.float().reshape(batch_size, -1)
                inherited_flat = inherited_logits.float().reshape(batch_size, -1)
                difference = inherited_flat - teacher_flat
                squared_error_sum += float(difference.square().sum().item())
                teacher_squared_sum += float(teacher_flat.square().sum().item())
                dot_sum += float((teacher_flat * inherited_flat).sum().item())
                teacher_vector_squared_sum += float(teacher_flat.square().sum().item())
                inherited_vector_squared_sum += float(inherited_flat.square().sum().item())

                if problem_type == "regression":
                    teacher_values = prepare_regression_outputs(teacher_logits)
                    inherited_values = prepare_regression_outputs(inherited_logits)
                    label_values = labels.view_as(teacher_values)
                    teacher_loss = criterion(teacher_values, label_values)
                    inherited_loss = criterion(inherited_values, label_values)
                    teacher_prediction = teacher_values
                    inherited_prediction = inherited_values
                    metric_labels = label_values
                else:
                    teacher_loss = criterion(teacher_logits, labels)
                    inherited_loss = criterion(inherited_logits, labels)
                    teacher_prediction = teacher_logits.argmax(dim=1)
                    inherited_prediction = inherited_logits.argmax(dim=1)
                    metric_labels = labels
                    agreement_sum += int((teacher_prediction == inherited_prediction).sum().item())
                    kl_sum += float(
                        F.kl_div(
                            F.log_softmax(inherited_logits.float(), dim=1),
                            F.softmax(teacher_logits.float(), dim=1),
                            reduction="batchmean",
                        ).item()
                    ) * batch_size

                teacher_loss_sum += float(teacher_loss.item()) * batch_size
                inherited_loss_sum += float(inherited_loss.item()) * batch_size
                teacher_predictions.append(teacher_prediction.detach().cpu())
                inherited_predictions.append(inherited_prediction.detach().cpu())
                labels_all.append(metric_labels.detach().cpu())
    finally:
        teacher_model.train(teacher_was_training)
        inherited_model.train(inherited_was_training)

    teacher_prediction_tensor = torch.cat(teacher_predictions) if teacher_predictions else torch.empty(0)
    inherited_prediction_tensor = torch.cat(inherited_predictions) if inherited_predictions else torch.empty(0)
    label_tensor = torch.cat(labels_all) if labels_all else torch.empty(0)
    teacher_metrics = compute_task_metric_values(
        teacher_prediction_tensor,
        label_tensor,
        problem_type=problem_type,
        num_labels=max(num_labels, 1),
        metric_names=metric_names,
    )
    inherited_metrics = compute_task_metric_values(
        inherited_prediction_tensor,
        label_tensor,
        problem_type=problem_type,
        num_labels=max(num_labels, 1),
        metric_names=metric_names,
    )
    cosine_denominator = math.sqrt(
        max(teacher_vector_squared_sum * inherited_vector_squared_sum, 0.0)
    )
    diagnostics: dict[str, Any] = {
        "examples": total,
        "teacher_loss": teacher_loss_sum / max(total, 1),
        "inherited_loss": inherited_loss_sum / max(total, 1),
        "teacher_metrics": teacher_metrics,
        "inherited_metrics": inherited_metrics,
        "output_squared_error_per_example": squared_error_sum / max(total, 1),
        "relative_output_squared_error": squared_error_sum / max(teacher_squared_sum, 1e-30),
        "output_cosine_similarity": dot_sum / cosine_denominator if cosine_denominator > 0 else 0.0,
    }
    if problem_type == "classification":
        diagnostics["teacher_to_inherited_kl"] = kl_sum / max(total, 1)
        diagnostics["prediction_agreement"] = agreement_sum / max(total, 1)
    else:
        diagnostics["teacher_inherited_pearson"] = (
            _pearson_correlation_percent(teacher_prediction_tensor, inherited_prediction_tensor)
            / 100.0
        )
    diagnostics["router_probe"] = evaluate_router_gradient_probe(
        teacher_model,
        inherited_model,
        data_loader,
        device,
        problem_type=problem_type,
        evaluation_split=evaluation_split,
    )
    local_operator_probe = evaluate_local_operator_probe(
        teacher_model,
        inherited_model,
        data_loader,
        device,
        evaluation_split=evaluation_split,
        max_batches=local_operator_max_batches,
    )
    if local_operator_probe is not None:
        diagnostics["local_operator_probe"] = local_operator_probe
    return diagnostics


def evaluate_local_operator_probe(
    teacher_model: nn.Module,
    inherited_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    evaluation_split: str,
    max_batches: int = 4,
) -> dict[str, Any] | None:
    """Compare dense and inherited local operators on held-out teacher inputs."""
    if max_batches <= 0:
        raise ValueError("local-operator diagnostics require max_batches > 0.")
    report = getattr(inherited_model, "inheract_report", None)
    backbone = getattr(inherited_model, "backbone", None)
    if not isinstance(report, Mapping) or not isinstance(backbone, nn.Module):
        return None
    allocation_layers = report.get("allocation_layers") or {}
    layer_names = [
        name
        for name, layer in allocation_layers.items()
        if isinstance(layer, Mapping) and layer.get("choice") != "dense"
    ]
    if not layer_names:
        return None

    second_moments = report.get("second_moments") or {}
    accumulators = {
        name: {
            "squared_error_sum": 0.0,
            "dense_squared_sum": 0.0,
            "applications": 0,
            "output_elements": 0,
        }
        for name in layer_names
    }
    current_attention_mask: torch.Tensor | None = None
    hooks: list[Any] = []

    def make_hook(name: str, inherited_layer: nn.Module):
        def hook(_module: nn.Module, module_inputs: tuple[Any, ...], dense_output: Any) -> None:
            if not module_inputs or not isinstance(module_inputs[0], torch.Tensor):
                raise TypeError(f"Local-operator probe requires tensor input for layer {name!r}.")
            if not isinstance(dense_output, torch.Tensor):
                raise TypeError(f"Local-operator probe requires tensor output for layer {name!r}.")
            inherited_output = inherited_layer(module_inputs[0])
            if not isinstance(inherited_output, torch.Tensor):
                raise TypeError(f"Inherited layer {name!r} did not return a tensor.")
            dense_values = dense_output.detach().float()
            inherited_values = inherited_output.detach().float()
            applications = dense_values.numel() // max(int(dense_values.shape[-1]), 1)
            if dense_values.ndim == 4:
                applications = dense_values.numel() // max(int(dense_values.shape[1]), 1)
            elif (
                current_attention_mask is not None
                and dense_values.ndim >= 3
                and tuple(current_attention_mask.shape) == tuple(dense_values.shape[:2])
            ):
                mask = current_attention_mask.to(device=dense_values.device, dtype=torch.bool)
                dense_values = dense_values[mask]
                inherited_values = inherited_values[mask]
                applications = int(mask.sum().item())
            difference = inherited_values - dense_values
            accumulator = accumulators[name]
            accumulator["squared_error_sum"] += float(difference.square().sum().item())
            accumulator["dense_squared_sum"] += float(dense_values.square().sum().item())
            accumulator["applications"] += applications
            accumulator["output_elements"] += dense_values.numel()

        return hook

    teacher_was_training = teacher_model.training
    inherited_was_training = inherited_model.training
    teacher_model.eval()
    inherited_model.eval()
    batches = 0
    examples = 0
    try:
        for name in layer_names:
            teacher_layer = teacher_model.get_submodule(name)
            inherited_layer = backbone.get_submodule(name)
            hooks.append(teacher_layer.register_forward_hook(make_hook(name, inherited_layer)))
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(data_loader):
                if batch_index >= max_batches:
                    break
                inputs = move_batch_to_device(inputs, device)
                current_attention_mask = (
                    inputs.get("attention_mask")
                    if isinstance(inputs, Mapping)
                    and isinstance(inputs.get("attention_mask"), torch.Tensor)
                    else None
                )
                forward_logits(teacher_model, inputs)
                batches += 1
                examples += int(labels.size(0))
    finally:
        current_attention_mask = None
        for hook in hooks:
            hook.remove()
        clear_gating_router_cache(inherited_model)
        teacher_model.train(teacher_was_training)
        inherited_model.train(inherited_was_training)

    per_layer: dict[str, dict[str, Any]] = {}
    total_squared_error = 0.0
    total_dense_squared = 0.0
    for name in layer_names:
        accumulator = accumulators[name]
        numerator = float(accumulator["squared_error_sum"])
        denominator = float(accumulator["dense_squared_sum"])
        total_squared_error += numerator
        total_dense_squared += denominator
        moment = second_moments.get(name) or {}
        per_layer[name] = {
            **accumulator,
            "relative_squared_error": numerator / max(denominator, 1e-30),
            "moment_mode": moment.get("mode", "unknown"),
        }
    return {
        "evaluation_split": evaluation_split,
        "max_batches": max_batches,
        "batches": batches,
        "examples": examples,
        "factorized_layer_count": len(per_layer),
        "squared_error_sum": total_squared_error,
        "dense_squared_sum": total_dense_squared,
        "relative_squared_error": total_squared_error / max(total_dense_squared, 1e-30),
        "aggregation": "ratio_of_summed_squared_errors",
        "per_layer": per_layer,
    }


def evaluate_router_gradient_probe(
    teacher_model: nn.Module,
    inherited_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    problem_type: str,
    evaluation_split: str = "validation",
    batch_index: int = 0,
    active_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Probe teacher-matching router gradients without updating model state."""
    routers = [
        (name, module)
        for name, module in inherited_model.named_modules()
        if isinstance(getattr(module, "gate", None), nn.Linear)
    ]
    if not routers:
        return {
            "objective": "teacher_kl" if problem_type == "classification" else "teacher_mse",
            "evaluation_split": evaluation_split,
            "batch_index": batch_index,
            "router_count": 0,
            "active_tolerance": active_tolerance,
        }

    try:
        probe_batch = next(
            batch for index, batch in enumerate(data_loader) if index == batch_index
        )
    except StopIteration as exc:
        raise ValueError(
            f"Router-gradient diagnostics require batch index {batch_index}, but it is unavailable."
        ) from exc
    inputs, _ = probe_batch
    inputs = move_batch_to_device(inputs, device)
    teacher_was_training = teacher_model.training
    inherited_was_training = inherited_model.training
    teacher_model.eval()
    inherited_model.eval()
    clear_gating_router_cache(inherited_model)
    try:
        with torch.no_grad():
            teacher_logits = forward_logits(teacher_model, inputs).detach()
        inherited_logits = forward_logits(inherited_model, inputs)
        if problem_type == "classification":
            objective = F.kl_div(
                F.log_softmax(inherited_logits.float(), dim=-1),
                F.softmax(teacher_logits.float(), dim=-1),
                reduction="batchmean",
            )
            objective_name = "teacher_kl"
        else:
            objective = F.mse_loss(
                prepare_regression_outputs(inherited_logits).float(),
                prepare_regression_outputs(teacher_logits).float(),
            )
            objective_name = "teacher_mse"

        parameter_entries: list[tuple[str, str, torch.Tensor]] = []
        for name, router in routers:
            parameter_entries.append((name, "weight", router.gate.weight))
            if router.gate.bias is not None:
                parameter_entries.append((name, "bias", router.gate.bias))
        gradients = torch.autograd.grad(
            objective,
            [entry[2] for entry in parameter_entries],
            allow_unused=True,
        )

        squared_sums = {"weight": 0.0, "bias": 0.0}
        element_counts = {"weight": 0, "bias": 0}
        per_router: dict[str, dict[str, float]] = {name: {} for name, _ in routers}
        for (name, kind, parameter), gradient in zip(parameter_entries, gradients):
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            squared_sum = float(gradient.detach().float().square().sum().item())
            squared_sums[kind] += squared_sum
            element_counts[kind] += gradient.numel()
            per_router[name][f"{kind}_l2"] = math.sqrt(squared_sum)

        active_router_count = sum(
            max(values.get("weight_l2", 0.0), values.get("bias_l2", 0.0))
            > active_tolerance
            for values in per_router.values()
        )
        entropies: list[float] = []
        diversity_values: list[float] = []
        for _, router in routers:
            probabilities = getattr(router, "_last_gating_probs", None)
            if probabilities is not None and probabilities.numel() > 0:
                entropy = -(
                    probabilities.float().clamp_min(1e-30)
                    * probabilities.float().clamp_min(1e-30).log()
                ).sum(dim=-1).mean()
                entropies.append(float((entropy / math.log(router.gate.out_features)).item()))
            experts = getattr(router, "experts", None)
            head_num = int(getattr(router, "head_num", 0))
            if experts is not None and head_num > 1:
                weights = experts.weight.detach().float().reshape(head_num, -1)
                mean_weight = weights.mean(dim=0, keepdim=True)
                deviation_rms = weights.sub(mean_weight).square().mean().sqrt()
                mean_rms = mean_weight.square().mean().sqrt().clamp_min(1e-30)
                diversity_values.append(float((deviation_rms / mean_rms).item()))

        return {
            "objective": objective_name,
            "objective_value": float(objective.detach().item()),
            "evaluation_split": evaluation_split,
            "batch_index": batch_index,
            "batch_examples": int(teacher_logits.shape[0]),
            "router_count": len(routers),
            "active_tolerance": active_tolerance,
            "active_router_count": active_router_count,
            "active_router_fraction": active_router_count / len(routers),
            "router_weight_gradient_rms": math.sqrt(
                squared_sums["weight"] / max(element_counts["weight"], 1)
            ),
            "router_bias_gradient_rms": math.sqrt(
                squared_sums["bias"] / max(element_counts["bias"], 1)
            ),
            "mean_normalized_route_entropy": (
                sum(entropies) / len(entropies) if entropies else None
            ),
            "mean_relative_expert_diversity": (
                sum(diversity_values) / len(diversity_values)
                if diversity_values
                else None
            ),
            "per_router_gradient_l2": per_router,
        }
    finally:
        clear_gating_router_cache(inherited_model)
        teacher_model.train(teacher_was_training)
        inherited_model.train(inherited_was_training)


def ensure_finite_scalar(value: float, context: str) -> float:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"Non-finite metric detected: {context}={value}")
    return float(value)


def compute_distillation_objective(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    settings: TrainSettings,
    *,
    student_model: nn.Module | None = None,
    aux_loss_weight: float = 0.0,
    criterion: nn.Module | None = None,
    problem_type: str = "classification",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    hard_loss = build_task_criterion(problem_type) if criterion is None else criterion
    if problem_type == "regression":
        student_values = prepare_regression_outputs(student_logits)
        teacher_values = prepare_regression_outputs(teacher_logits).detach()
        label_values = labels.float().view_as(student_values)
        ce_loss = hard_loss(student_values, label_values)
        kd_loss = F.mse_loss(student_values, teacher_values.view_as(student_values))
        kd_scale = 1.0
    else:
        ce_loss = hard_loss(student_logits, labels.long())
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / settings.kd_temperature, dim=1),
            F.softmax(teacher_logits / settings.kd_temperature, dim=1),
            reduction="batchmean",
        )
        kd_scale = settings.kd_temperature**2
    aux_loss = None
    total_loss = (
        settings.ce_loss_weight * ce_loss
        + settings.kd_loss_weight * kd_scale * kd_loss
    )
    if aux_loss_weight > 0 and student_model is not None:
        aux_loss = compute_gating_load_balance_loss(student_model)
        if aux_loss is not None:
            total_loss = total_loss + aux_loss_weight * aux_loss
    elif student_model is not None:
        clear_gating_router_cache(student_model)
    return ce_loss, kd_loss, aux_loss, total_loss


def compute_logit_standardized_distillation_objective(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    settings: LogitStandardizedKDSettings,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """Compute the published CVPR 2024 logit-standardized KD objective."""
    labels = labels.long().reshape(-1)
    student_standardized = (student_logits - student_logits.mean(dim=1, keepdim=True)) / (
        student_logits.std(dim=1, keepdim=True) + 1e-7
    )
    teacher_standardized = (teacher_logits - teacher_logits.mean(dim=1, keepdim=True)) / (
        teacher_logits.std(dim=1, keepdim=True) + 1e-7
    )
    kd_loss = F.kl_div(
        F.log_softmax(student_standardized / settings.temperature, dim=1),
        F.softmax(teacher_standardized / settings.temperature, dim=1),
        reduction="batchmean",
    )
    ce_loss = criterion(student_logits, labels)
    total_loss = (
        settings.ce_weight * ce_loss
        + settings.kd_weight * settings.temperature**2 * kd_loss
    )
    return ce_loss, kd_loss, None, total_loss


def curriculum_temperature_gradient_scale(
    epoch: int,
    settings: CurriculumTemperatureDistillationSettings,
) -> float:
    """Return CTKD's released cosine multiplier for the global-temperature GRL."""
    position = min(max(int(epoch), 0), settings.decay_loops)
    cosine = (math.cos(position * math.pi / settings.decay_loops) + 1.0) * 0.5
    return cosine * (settings.decay_max - settings.decay_min) + settings.decay_min


def compute_curriculum_temperature_distillation_objective(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    settings: CurriculumTemperatureDistillationSettings,
    temperature_module: GlobalCurriculumTemperature,
    gradient_scale: float,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """Compute the released global-CTKD objective for a registered vision pair."""
    labels = labels.long().reshape(-1)
    temperature = temperature_module(gradient_scale)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature.square().squeeze()
    ce_loss = criterion(student_logits, labels)
    total_loss = settings.ce_weight * ce_loss + settings.kd_weight * kd_loss
    return ce_loss, kd_loss, None, total_loss


def compute_decoupled_distillation_objective(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    settings: DecoupledDistillationSettings,
    epoch: int,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
    """Compute the published DKD objective for a registered vision baseline."""
    labels = labels.long().reshape(-1)
    gt_mask = torch.zeros_like(student_logits, dtype=torch.bool).scatter_(
        1, labels.unsqueeze(1), True
    )
    other_mask = ~gt_mask
    student_prob = F.softmax(student_logits / settings.temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / settings.temperature, dim=1)
    student_binary = torch.cat(
        [
            (student_prob * gt_mask).sum(dim=1, keepdim=True),
            (student_prob * other_mask).sum(dim=1, keepdim=True),
        ],
        dim=1,
    )
    teacher_binary = torch.cat(
        [
            (teacher_prob * gt_mask).sum(dim=1, keepdim=True),
            (teacher_prob * other_mask).sum(dim=1, keepdim=True),
        ],
        dim=1,
    )
    temperature_scale = settings.temperature**2 / labels.shape[0]
    target_class_kd = F.kl_div(
        torch.log(student_binary), teacher_binary, reduction="sum"
    ) * temperature_scale
    teacher_non_target = F.softmax(
        teacher_logits / settings.temperature - 1000.0 * gt_mask, dim=1
    )
    student_non_target = F.log_softmax(
        student_logits / settings.temperature - 1000.0 * gt_mask, dim=1
    )
    non_target_kd = F.kl_div(
        student_non_target, teacher_non_target, reduction="sum"
    ) * temperature_scale
    dkd_loss = settings.alpha * target_class_kd + settings.beta * non_target_kd
    ce_loss = criterion(student_logits, labels)
    warmup = min(epoch / settings.warmup_epochs, 1.0)
    total_loss = settings.ce_weight * ce_loss + warmup * dkd_loss
    return ce_loss, dkd_loss, None, total_loss


def build_optimizer(model: nn.Module, settings: TrainSettings) -> optim.Optimizer:
    if settings.optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=settings.lr,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
        )
    if settings.optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    if settings.optimizer_name.lower() == "adamw":
        if settings.exclude_bias_norm_from_weight_decay:
            norm_parameter_ids = {
                id(parameter)
                for module in model.modules()
                if isinstance(module, nn.LayerNorm)
                for parameter in module.parameters(recurse=False)
            }
            decay_parameters = []
            no_decay_parameters = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.endswith(".bias") or id(parameter) in norm_parameter_ids:
                    no_decay_parameters.append(parameter)
                else:
                    decay_parameters.append(parameter)
            parameter_groups = [
                {"params": decay_parameters, "weight_decay": settings.weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ]
            return optim.AdamW(parameter_groups, lr=settings.lr)
        return optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    raise ValueError(f"Unsupported optimizer: {settings.optimizer_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    settings: TrainSettings,
    *,
    steps_per_epoch: int,
):
    if settings.scheduler_name == "none":
        return None
    if settings.scheduler_name == "multistep":
        if not settings.lr_milestones:
            return None
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(settings.lr_milestones),
            gamma=settings.lr_gamma,
        )
    if settings.scheduler_name == "linear":
        total_steps = max(1, settings.epochs * steps_per_epoch)
        warmup_steps = round(total_steps * settings.warmup_ratio)

        def lr_multiplier(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(warmup_steps)
            return max(
                0.0,
                float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
            )

        return optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    raise ValueError(f"Unsupported scheduler: {settings.scheduler_name}")


def scheduler_steps_per_batch(settings: TrainSettings) -> bool:
    return settings.scheduler_name == "linear"


def _average_train_batch_ms(train_time_seconds: float, batch_count: int) -> float:
    return 1000.0 * train_time_seconds / max(batch_count, 1)


def _finalize_test_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    phase: str,
    epoch: int,
    *,
    problem_type: str,
    num_labels: int,
    metric_names: tuple[str, ...],
    primary_metric_name: str,
) -> dict[str, float]:
    test_metrics = evaluate_task_metrics(
        model,
        test_loader,
        device,
        criterion,
        problem_type=problem_type,
        num_labels=num_labels,
        metric_names=metric_names,
    )
    test_metrics["loss"] = ensure_finite_scalar(
        test_metrics["loss"],
        f"{phase} epoch {epoch} test_loss",
    )
    for metric_name in metric_names:
        if metric_name in test_metrics:
            test_metrics[metric_name] = ensure_finite_scalar(
                test_metrics[metric_name],
                f"{phase} epoch {epoch} test_{metric_name}",
            )
    if primary_metric_name not in test_metrics:
        raise KeyError(f"Primary metric '{primary_metric_name}' missing from evaluation metrics.")
    return test_metrics


def _record_epoch_metrics(
    history: dict[str, list[float]],
    logger: RunLogger,
    *,
    epoch: int,
    settings: TrainSettings,
    phase: str,
    train_objective: float,
    train_loss: float,
    train_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
    epoch_time_seconds: float,
    train_time_seconds: float,
    eval_time_seconds: float,
    avg_train_batch_ms: float,
    eval_split_name: str = "test",
    primary_metric_name: str = "accuracy",
    train_components: Mapping[str, float] | None = None,
) -> None:
    eval_loss = float(test_metrics["loss"])
    train_primary = float(train_metrics[primary_metric_name])
    eval_primary = float(test_metrics[primary_metric_name])
    train_accuracy = float(train_metrics.get("accuracy", train_primary))
    eval_accuracy = float(test_metrics.get("accuracy", eval_primary))
    split_key_prefix = eval_split_name.replace("-", "_").replace(" ", "_")
    history["train_objective"].append(train_objective)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_primary)
    history["test_loss"].append(eval_loss)
    history["test_accuracy"].append(eval_primary)
    for component_name, component_value in (train_components or {}).items():
        history.setdefault(f"train_component_{component_name}", []).append(float(component_value))
    for metric_name, metric_value in train_metrics.items():
        history.setdefault(f"train_metric_{metric_name}", []).append(float(metric_value))
    for metric_name, metric_value in test_metrics.items():
        if metric_name != "loss":
            history.setdefault(f"eval_metric_{metric_name}", []).append(float(metric_value))
    metrics_payload = {
        "epoch": epoch,
        "epochs": settings.epochs,
        "phase": phase,
        "eval_split": eval_split_name,
        "train_objective": train_objective,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        f"{split_key_prefix}_loss": eval_loss,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": eval_primary,
        "train_primary_metric_value": train_primary,
        "epoch_time_seconds": epoch_time_seconds,
        "train_time_seconds": train_time_seconds,
        "eval_time_seconds": eval_time_seconds,
        "avg_train_batch_ms": avg_train_batch_ms,
    }
    if split_key_prefix == "test":
        metrics_payload["test_loss"] = eval_loss
    for metric_name, metric_value in train_metrics.items():
        metrics_payload[f"train_{metric_name}"] = float(metric_value)
    for metric_name, metric_value in test_metrics.items():
        if metric_name == "loss":
            continue
        metric_value = float(metric_value)
        metrics_payload[f"eval_{metric_name}"] = metric_value
        metrics_payload[f"{split_key_prefix}_{metric_name}"] = metric_value
    for component_name, component_value in (train_components or {}).items():
        metrics_payload[f"train_{component_name}"] = float(component_value)
    logger.metrics(metrics_payload)
    train_metric_label = f"train_{primary_metric_name}"
    eval_metric_label = f"{eval_split_name}_{primary_metric_name}"
    logger.epoch(
        f"[{phase}] Epoch {epoch:03d}/{settings.epochs:03d} | "
        f"train_objective={train_objective:.4f} | "
        f"train_loss={train_loss:.4f} | "
        f"{train_metric_label}={train_primary:.2f} | "
        f"{eval_split_name}_loss={eval_loss:.4f} | "
        f"{eval_metric_label}={eval_primary:.2f} | "
        f"train_time={train_time_seconds:.2f}s | "
        f"eval_time={eval_time_seconds:.2f}s | "
        f"epoch_time={epoch_time_seconds:.2f}s | "
        f"avg_batch={avg_train_batch_ms:.2f}ms"
    )


def _summarize_history(
    history: Mapping[str, list[float]],
    *,
    eval_split_name: str,
    primary_metric_name: str,
    primary_metric_display: str,
) -> dict[str, Any]:
    train_loss = history.get("train_loss", [])
    train_primary = history.get("train_accuracy", [])
    eval_loss = history.get("test_loss", [])
    eval_primary = history.get("test_accuracy", [])
    best_eval_metric = max(eval_primary) if eval_primary else None
    best_eval_epoch = eval_primary.index(best_eval_metric) + 1 if best_eval_metric is not None else 0
    selected_eval_metrics = {
        key.removeprefix("eval_metric_"): float(values[best_eval_epoch - 1])
        for key, values in history.items()
        if key.startswith("eval_metric_") and best_eval_epoch > 0 and len(values) >= best_eval_epoch
    }
    final_eval_metrics = {
        key.removeprefix("eval_metric_"): float(values[-1])
        for key, values in history.items()
        if key.startswith("eval_metric_") and values
    }
    summary: dict[str, Any] = {
        "epochs_completed": len(eval_primary),
        "eval_split": eval_split_name,
        "primary_metric_name": primary_metric_name,
        "primary_metric_display": primary_metric_display,
        "best_eval_metric": best_eval_metric,
        "best_eval_epoch": best_eval_epoch,
        "selected_eval_metrics": selected_eval_metrics,
        "final_eval_metrics": final_eval_metrics,
    }
    if primary_metric_name == "accuracy":
        summary["best_eval_accuracy"] = best_eval_metric
    if train_loss:
        summary["final_train_loss"] = train_loss[-1]
    if train_primary:
        summary["final_train_primary_metric"] = train_primary[-1]
        if primary_metric_name == "accuracy":
            summary["final_train_accuracy"] = train_primary[-1]
    if eval_loss:
        summary["final_eval_loss"] = eval_loss[-1]
    if eval_primary:
        summary["final_eval_primary_metric"] = eval_primary[-1]
        if primary_metric_name == "accuracy":
            summary["final_eval_accuracy"] = eval_primary[-1]
    return summary


def _log_training_summary(
    history: Mapping[str, list[float]],
    logger: RunLogger,
    *,
    eval_split_name: str,
    primary_metric_name: str,
    primary_metric_display: str,
) -> None:
    summary = _summarize_history(
        history,
        eval_split_name=eval_split_name,
        primary_metric_name=primary_metric_name,
        primary_metric_display=primary_metric_display,
    )
    logger.structured(RUN_SUMMARY_PREFIX, summary)
    if summary["epochs_completed"] == 0:
        logger.info("Training summary | no epochs completed")
        return
    logger.info(
        "Training summary | "
        f"best_{eval_split_name}_{primary_metric_name}={float(summary['best_eval_metric']):.2f} "
        f"@ epoch {int(summary['best_eval_epoch'])} | "
        f"final_{eval_split_name}_{primary_metric_name}={float(summary['final_eval_primary_metric']):.2f} | "
        f"final_{eval_split_name}_loss={float(summary['final_eval_loss']):.4f}"
    )


def _copy_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _restore_best_and_evaluate_final_test(
    model: nn.Module,
    best_state: Mapping[str, torch.Tensor] | None,
    best_epoch: int,
    final_test_loader: DataLoader | None,
    device: torch.device,
    criterion: nn.Module,
    logger: RunLogger,
    *,
    phase: str,
    problem_type: str,
    num_labels: int,
    metric_names: tuple[str, ...],
    primary_metric_name: str,
    final_test_split_name: str,
) -> dict[str, float] | None:
    if best_state is not None:
        model.load_state_dict(best_state)
    if final_test_loader is None:
        return None
    metrics = _finalize_test_metrics(
        model,
        final_test_loader,
        device,
        criterion,
        phase,
        best_epoch,
        problem_type=problem_type,
        num_labels=max(num_labels, 1),
        metric_names=metric_names,
        primary_metric_name=primary_metric_name,
    )
    payload: dict[str, float | int | str] = {
        "phase": phase,
        "selection_epoch": best_epoch,
        "split": final_test_split_name,
        "primary_metric_name": primary_metric_name,
    }
    payload.update({name: float(value) for name, value in metrics.items()})
    logger.structured(RUN_FINAL_TEST_PREFIX, payload)
    logger.info(
        f"Final {final_test_split_name} evaluation | selected_epoch={best_epoch} | "
        f"{primary_metric_name}={float(metrics[primary_metric_name]):.2f} | "
        f"loss={float(metrics['loss']):.4f}"
    )
    return metrics


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    aux_loss_weight: float = 0.0,
    logger: RunLogger | None = None,
    phase: str = "target",
    eval_split_name: str = "test",
    primary_metric_name: str = "accuracy",
    primary_metric_display: str = "Accuracy (%)",
    metric_names: tuple[str, ...] = ("accuracy",),
    problem_type: str = "classification",
    num_labels: int = 0,
    final_test_loader: DataLoader | None = None,
    final_test_split_name: str = "test",
    restore_best_state: bool = False,
) -> dict[str, list[float]]:
    criterion = build_task_criterion(problem_type)
    optimizer = build_optimizer(model, settings)
    scheduler = build_scheduler(optimizer, settings, steps_per_epoch=len(train_loader))
    history = create_history_template()
    logger = logger or build_run_logger()
    best_eval_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(settings.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_objective = 0.0
        running_ce_loss = torch.zeros((), device=device)
        running_aux_loss = torch.zeros((), device=device)
        total_examples = 0
        batch_count = 0
        train_predictions: list[torch.Tensor] = []
        train_labels: list[torch.Tensor] = []
        for batch_count, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = move_batch_to_device(inputs, device)
            labels = prepare_labels(labels.to(device), problem_type)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(model, inputs)
            if problem_type == "regression":
                predictions = prepare_regression_outputs(logits)
                label_values = labels.view_as(predictions)
                ce_loss = criterion(predictions, label_values)
                metric_labels = label_values
            else:
                ce_loss = criterion(logits, labels)
                predictions = logits.argmax(dim=1)
                metric_labels = labels
            loss = ce_loss
            if aux_loss_weight > 0:
                aux_loss = compute_gating_load_balance_loss(model)
                if aux_loss is not None:
                    loss = loss + aux_loss_weight * aux_loss
                    running_aux_loss += aux_loss.detach() * labels.size(0)
            else:
                clear_gating_router_cache(model)
            loss_value = ensure_finite_scalar(
                float(loss.detach()),
                f"{phase} epoch {epoch + 1} supervised training",
            )
            loss.backward()
            if settings.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            optimizer.step()
            if scheduler is not None and scheduler_steps_per_batch(settings):
                scheduler.step()
            batch_size = labels.size(0)
            running_objective += loss_value * batch_size
            running_ce_loss += ce_loss.detach() * batch_size
            total_examples += batch_size
            train_predictions.append(predictions.detach().cpu())
            train_labels.append(metric_labels.detach().cpu())
        if scheduler is not None and not scheduler_steps_per_batch(settings):
            scheduler.step()
        train_time_seconds = time.perf_counter() - epoch_start

        train_objective = ensure_finite_scalar(
            running_objective / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_objective",
        )
        train_loss = ensure_finite_scalar(
            float(running_ce_loss) / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_loss",
        )
        train_metrics = compute_task_metric_values(
            torch.cat(train_predictions) if train_predictions else torch.empty(0),
            torch.cat(train_labels) if train_labels else torch.empty(0),
            problem_type=problem_type,
            num_labels=max(num_labels, 1),
            metric_names=metric_names,
        )
        for metric_name, metric_value in list(train_metrics.items()):
            train_metrics[metric_name] = ensure_finite_scalar(
                metric_value,
                f"{phase} epoch {epoch + 1} train_{metric_name}",
            )
        eval_start = time.perf_counter()
        test_metrics = _finalize_test_metrics(
            model,
            test_loader,
            device,
            criterion,
            phase,
            epoch + 1,
            problem_type=problem_type,
            num_labels=max(num_labels, 1),
            metric_names=metric_names,
            primary_metric_name=primary_metric_name,
        )
        eval_time_seconds = time.perf_counter() - eval_start
        current_metric = float(test_metrics[primary_metric_name])
        if (restore_best_state or final_test_loader is not None) and current_metric > best_eval_metric:
            best_eval_metric = current_metric
            best_epoch = epoch + 1
            best_state = _copy_state_dict_to_cpu(model)
        epoch_time_seconds = time.perf_counter() - epoch_start
        avg_train_batch_ms = _average_train_batch_ms(train_time_seconds, batch_count)
        _record_epoch_metrics(
            history,
            logger,
            epoch=epoch + 1,
            settings=settings,
            phase=phase,
            train_objective=train_objective,
            train_loss=train_loss,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            epoch_time_seconds=epoch_time_seconds,
            train_time_seconds=train_time_seconds,
            eval_time_seconds=eval_time_seconds,
            avg_train_batch_ms=avg_train_batch_ms,
            eval_split_name=eval_split_name,
            primary_metric_name=primary_metric_name,
            train_components=(
                {"aux_loss": float(running_aux_loss) / max(total_examples, 1)}
                if aux_loss_weight > 0
                else None
            ),
        )
    _log_training_summary(
        history,
        logger,
        eval_split_name=eval_split_name,
        primary_metric_name=primary_metric_name,
        primary_metric_display=primary_metric_display,
    )
    final_metrics = _restore_best_and_evaluate_final_test(
        model,
        best_state,
        best_epoch,
        final_test_loader,
        device,
        criterion,
        logger,
        phase=phase,
        problem_type=problem_type,
        num_labels=num_labels,
        metric_names=metric_names,
        primary_metric_name=primary_metric_name,
        final_test_split_name=final_test_split_name,
    )
    if final_metrics is not None:
        history["final_test_loss"] = [float(final_metrics["loss"])]
        history["final_test_accuracy"] = [float(final_metrics[primary_metric_name])]
    return history


def train_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    aux_loss_weight: float = 0.0,
    logger: RunLogger | None = None,
    phase: str = "target",
    run_label: str | None = None,
    eval_split_name: str = "test",
    primary_metric_name: str = "accuracy",
    primary_metric_display: str = "Accuracy (%)",
    metric_names: tuple[str, ...] = ("accuracy",),
    problem_type: str = "classification",
    num_labels: int = 0,
    final_test_loader: DataLoader | None = None,
    final_test_split_name: str = "test",
    restore_best_state: bool = False,
    dkd_settings: DecoupledDistillationSettings | None = None,
    logit_standardized_kd_settings: LogitStandardizedKDSettings | None = None,
    ctkd_settings: CurriculumTemperatureDistillationSettings | None = None,
) -> dict[str, list[float]]:
    hard_loss = build_task_criterion(problem_type)
    ctkd_temperature = (
        GlobalCurriculumTemperature(ctkd_settings).to(device)
        if ctkd_settings is not None
        else None
    )
    trainable_model = (
        nn.ModuleList([student_model, ctkd_temperature])
        if ctkd_temperature is not None
        else student_model
    )
    optimizer = build_optimizer(trainable_model, settings)
    scheduler = build_scheduler(optimizer, settings, steps_per_epoch=len(train_loader))
    history = create_history_template()
    logger = logger or build_run_logger()
    run_label = run_label or phase
    best_eval_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    teacher_model.eval()
    for epoch in range(settings.epochs):
        epoch_start = time.perf_counter()
        student_model.train()
        if ctkd_temperature is not None:
            ctkd_temperature.train()
        ctkd_gradient_scale = (
            curriculum_temperature_gradient_scale(epoch + 1, ctkd_settings)
            if ctkd_settings is not None
            else None
        )
        running_objective = 0.0
        running_ce_loss = torch.zeros((), device=device)
        running_kd_loss = torch.zeros((), device=device)
        running_aux_loss = torch.zeros((), device=device)
        total_examples = 0
        batch_count = 0
        train_predictions: list[torch.Tensor] = []
        train_labels: list[torch.Tensor] = []
        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            batch_count = batch_idx
            inputs = move_batch_to_device(inputs, device)
            labels = prepare_labels(labels.to(device), problem_type)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = forward_logits(teacher_model, inputs)
            student_logits = forward_logits(student_model, inputs)
            if ctkd_settings is not None:
                ce_loss, kd_loss, aux_loss, loss = (
                    compute_curriculum_temperature_distillation_objective(
                        teacher_logits,
                        student_logits,
                        labels,
                        ctkd_settings,
                        ctkd_temperature,
                        ctkd_gradient_scale,
                        hard_loss,
                    )
                )
            elif dkd_settings is not None:
                ce_loss, kd_loss, aux_loss, loss = compute_decoupled_distillation_objective(
                    teacher_logits,
                    student_logits,
                    labels,
                    dkd_settings,
                    epoch + 1,
                    hard_loss,
                )
            elif logit_standardized_kd_settings is not None:
                ce_loss, kd_loss, aux_loss, loss = (
                    compute_logit_standardized_distillation_objective(
                        teacher_logits,
                        student_logits,
                        labels,
                        logit_standardized_kd_settings,
                        hard_loss,
                    )
                )
            else:
                ce_loss, kd_loss, aux_loss, loss = compute_distillation_objective(
                    teacher_logits,
                    student_logits,
                    labels,
                    settings,
                    student_model=student_model,
                    aux_loss_weight=aux_loss_weight,
                    criterion=hard_loss,
                    problem_type=problem_type,
                )
            loss_value = ensure_finite_scalar(
                float(loss.detach()),
                f"{run_label} epoch {epoch + 1} batch {batch_idx}",
            )
            loss.backward()
            if settings.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable_model.parameters(), settings.max_grad_norm)
            optimizer.step()
            if scheduler is not None and scheduler_steps_per_batch(settings):
                scheduler.step()
            batch_size = labels.size(0)
            running_objective += loss_value * batch_size
            running_ce_loss += ce_loss.detach() * batch_size
            running_kd_loss += kd_loss.detach() * batch_size
            if aux_loss is not None:
                running_aux_loss += aux_loss.detach() * batch_size
            total_examples += batch_size
            if problem_type == "regression":
                predictions = prepare_regression_outputs(student_logits)
                metric_labels = labels.view_as(predictions)
            else:
                predictions = student_logits.argmax(dim=1)
                metric_labels = labels
            train_predictions.append(predictions.detach().cpu())
            train_labels.append(metric_labels.detach().cpu())
        if scheduler is not None and not scheduler_steps_per_batch(settings):
            scheduler.step()
        train_time_seconds = time.perf_counter() - epoch_start

        train_objective = ensure_finite_scalar(
            running_objective / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_objective",
        )
        train_loss = ensure_finite_scalar(
            float(running_ce_loss) / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_loss",
        )
        train_metrics = compute_task_metric_values(
            torch.cat(train_predictions) if train_predictions else torch.empty(0),
            torch.cat(train_labels) if train_labels else torch.empty(0),
            problem_type=problem_type,
            num_labels=max(num_labels, 1),
            metric_names=metric_names,
        )
        for metric_name, metric_value in list(train_metrics.items()):
            train_metrics[metric_name] = ensure_finite_scalar(
                metric_value,
                f"{phase} epoch {epoch + 1} train_{metric_name}",
            )
        eval_start = time.perf_counter()
        test_metrics = _finalize_test_metrics(
            student_model,
            test_loader,
            device,
            hard_loss,
            phase,
            epoch + 1,
            problem_type=problem_type,
            num_labels=max(num_labels, 1),
            metric_names=metric_names,
            primary_metric_name=primary_metric_name,
        )
        eval_time_seconds = time.perf_counter() - eval_start
        current_metric = float(test_metrics[primary_metric_name])
        if (restore_best_state or final_test_loader is not None) and current_metric > best_eval_metric:
            best_eval_metric = current_metric
            best_epoch = epoch + 1
            best_state = _copy_state_dict_to_cpu(student_model)
        epoch_time_seconds = time.perf_counter() - epoch_start
        avg_train_batch_ms = _average_train_batch_ms(train_time_seconds, batch_count)
        train_components = {
            "kd_loss": float(running_kd_loss) / max(total_examples, 1),
            "aux_loss": float(running_aux_loss) / max(total_examples, 1),
        }
        if ctkd_temperature is not None:
            train_components["ctkd_temperature"] = float(
                ctkd_temperature.current_temperature().detach()
            )
            train_components["ctkd_gradient_scale"] = float(ctkd_gradient_scale)
        _record_epoch_metrics(
            history,
            logger,
            epoch=epoch + 1,
            settings=settings,
            phase=phase,
            train_objective=train_objective,
            train_loss=train_loss,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            epoch_time_seconds=epoch_time_seconds,
            train_time_seconds=train_time_seconds,
            eval_time_seconds=eval_time_seconds,
            avg_train_batch_ms=avg_train_batch_ms,
            eval_split_name=eval_split_name,
            primary_metric_name=primary_metric_name,
            train_components=train_components,
        )
    _log_training_summary(
        history,
        logger,
        eval_split_name=eval_split_name,
        primary_metric_name=primary_metric_name,
        primary_metric_display=primary_metric_display,
    )
    final_metrics = _restore_best_and_evaluate_final_test(
        student_model,
        best_state,
        best_epoch,
        final_test_loader,
        device,
        hard_loss,
        logger,
        phase=phase,
        problem_type=problem_type,
        num_labels=num_labels,
        metric_names=metric_names,
        primary_metric_name=primary_metric_name,
        final_test_split_name=final_test_split_name,
    )
    if final_metrics is not None:
        history["final_test_loss"] = [float(final_metrics["loss"])]
        history["final_test_accuracy"] = [float(final_metrics[primary_metric_name])]
    return history


def train_vision_distillation(
    teacher_model: nn.Module,
    distiller: CATKDDistiller | SimKDDistiller | ReviewKDDistiller | CRDDistiller,
    train_loader: DataLoader,
    test_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    *,
    logger: RunLogger | None = None,
    phase: str = "target",
    run_label: str = "feature_distillation",
    eval_split_name: str = "test",
    primary_metric_name: str = "accuracy",
    primary_metric_display: str = "Accuracy (%)",
    metric_names: tuple[str, ...] = ("accuracy",),
    num_labels: int = 0,
    final_test_loader: DataLoader | None = None,
    final_test_split_name: str = "test",
    restore_best_state: bool = False,
) -> dict[str, list[float]]:
    """Train a released vision feature-distillation baseline.

    All method-specific mathematics lives in the distiller.  This loop owns
    only the repository's shared optimizer, selection, logging, and final-test
    protocol, so feature baselines are evaluated identically to other formal
    cells.
    """

    if num_labels <= 1:
        raise ValueError("Vision distillation requires a classification task.")
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(distiller, settings)
    scheduler = build_scheduler(optimizer, settings, steps_per_epoch=len(train_loader))
    history = create_history_template()
    logger = logger or build_run_logger()
    best_eval_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    selection_model = distiller if isinstance(distiller, SimKDDistiller) else distiller.student

    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    distiller.to(device)
    for epoch in range(settings.epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        distiller.train()
        running_objective = 0.0
        running_ce_loss = 0.0
        running_transfer_loss = 0.0
        total_examples = 0
        batch_count = 0
        train_predictions: list[torch.Tensor] = []
        train_labels: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch_count = batch_idx
            if isinstance(distiller, CRDDistiller):
                inputs, labels, sample_indices, contrast_indices = batch
                sample_indices = sample_indices.to(device)
                contrast_indices = contrast_indices.to(device)
            else:
                inputs, labels = batch
                sample_indices = contrast_indices = None
            inputs = move_batch_to_device(inputs, device)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            if isinstance(distiller, CRDDistiller):
                output = distiller.training_objective(
                    teacher_model,
                    inputs,
                    labels,
                    sample_indices,
                    contrast_indices,
                    epoch + 1,
                    criterion,
                )
                transfer_loss = output.contrastive
                loss = output.total
                logits = output.logits
                ce_loss = output.classification
            else:
                output = distiller.training_objective(
                    teacher_model, inputs, labels, epoch, criterion
                )
                transfer_loss = output.feature_loss
                loss = output.total_loss
                logits = output.logits
                ce_loss = output.ce_loss
            loss_value = ensure_finite_scalar(
                float(loss.detach()),
                f"{run_label} epoch {epoch + 1} batch {batch_idx}",
            )
            loss.backward()
            if settings.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(distiller.parameters(), settings.max_grad_norm)
            optimizer.step()
            if scheduler is not None and scheduler_steps_per_batch(settings):
                scheduler.step()
            batch_size = labels.size(0)
            running_objective += loss_value * batch_size
            running_ce_loss += float(ce_loss.detach()) * batch_size
            running_transfer_loss += float(transfer_loss.detach()) * batch_size
            total_examples += batch_size
            train_predictions.append(logits.argmax(dim=1).detach().cpu())
            train_labels.append(labels.detach().cpu())
        if scheduler is not None and not scheduler_steps_per_batch(settings):
            scheduler.step()
        train_time_seconds = time.perf_counter() - epoch_start
        train_metrics = compute_task_metric_values(
            torch.cat(train_predictions),
            torch.cat(train_labels),
            problem_type="classification",
            num_labels=num_labels,
            metric_names=metric_names,
        )
        eval_start = time.perf_counter()
        test_metrics = _finalize_test_metrics(
            distiller,
            test_loader,
            device,
            criterion,
            phase,
            epoch + 1,
            problem_type="classification",
            num_labels=num_labels,
            metric_names=metric_names,
            primary_metric_name=primary_metric_name,
        )
        eval_time_seconds = time.perf_counter() - eval_start
        current_metric = float(test_metrics[primary_metric_name])
        if (restore_best_state or final_test_loader is not None) and current_metric > best_eval_metric:
            best_eval_metric = current_metric
            best_epoch = epoch + 1
            best_state = _copy_state_dict_to_cpu(selection_model)
        _record_epoch_metrics(
            history,
            logger,
            epoch=epoch + 1,
            settings=settings,
            phase=phase,
            train_objective=running_objective / max(total_examples, 1),
            train_loss=running_ce_loss / max(total_examples, 1),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            epoch_time_seconds=time.perf_counter() - epoch_start,
            train_time_seconds=train_time_seconds,
            eval_time_seconds=eval_time_seconds,
            avg_train_batch_ms=_average_train_batch_ms(train_time_seconds, batch_count),
            eval_split_name=eval_split_name,
            primary_metric_name=primary_metric_name,
            train_components={
                "transfer_loss": running_transfer_loss / max(total_examples, 1)
            },
        )
    _log_training_summary(
        history,
        logger,
        eval_split_name=eval_split_name,
        primary_metric_name=primary_metric_name,
        primary_metric_display=primary_metric_display,
    )
    if best_state is not None:
        selection_model.load_state_dict(best_state)
    final_metrics = _restore_best_and_evaluate_final_test(
        distiller,
        None,
        best_epoch,
        final_test_loader,
        device,
        criterion,
        logger,
        phase=phase,
        problem_type="classification",
        num_labels=num_labels,
        metric_names=metric_names,
        primary_metric_name=primary_metric_name,
        final_test_split_name=final_test_split_name,
    )
    if final_metrics is not None:
        history["final_test_loss"] = [float(final_metrics["loss"])]
        history["final_test_accuracy"] = [float(final_metrics[primary_metric_name])]
    return history


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
