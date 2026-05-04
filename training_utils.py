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

from experiment_registry import TrainSettings
from model_wrappers import compute_gating_load_balance_loss


RUN_LOG_ENV_VAR = "INHERNET_RUN_LOG"
RUN_METADATA_PREFIX = "RUN_METADATA"
RUN_METRICS_PREFIX = "RUN_METRICS"
RUN_SUMMARY_PREFIX = "RUN_SUMMARY"


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


def evaluate_classification_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    return evaluate_task_metrics(
        model,
        data_loader,
        device,
        criterion,
        problem_type="classification",
        metric_names=("accuracy",),
    )


def ensure_finite_scalar(value: float, context: str) -> float:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"Non-finite metric detected: {context}={value}")
    return float(value)


def ensure_finite_loss_tensor(loss: torch.Tensor, context: str) -> None:
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Non-finite optimization loss detected during {context}.")


def _tensor_is_finite(value: torch.Tensor | None) -> bool:
    if value is None:
        return True
    return bool(torch.isfinite(value).all().item())


def _format_scalar(value: torch.Tensor | None) -> str:
    if value is None:
        return "n/a"
    detached = value.detach()
    if detached.numel() == 0:
        return "empty"
    scalar = detached.item() if detached.numel() == 1 else detached.mean().item()
    return f"{float(scalar):.6g}"


def _format_max_abs(value: torch.Tensor | None) -> str:
    if value is None:
        return "n/a"
    detached = value.detach()
    if detached.numel() == 0:
        return "empty"
    return f"{float(detached.abs().max().item()):.6g}"


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _collect_non_finite_distillation_issues(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    ce_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    total_loss: torch.Tensor,
    aux_loss: torch.Tensor | None,
) -> list[str]:
    issues: list[str] = []
    for name, value in (
        ("teacher_logits", teacher_logits),
        ("student_logits", student_logits),
        ("ce_loss", ce_loss),
        ("kd_loss", kd_loss),
        ("aux_loss", aux_loss),
        ("total_loss", total_loss),
    ):
        if not _tensor_is_finite(value):
            issues.append(name)
    return issues


def _safe_max_abs_tensor(value: torch.Tensor) -> float | None:
    detached = value.detach()
    if detached.numel() == 0:
        return 0.0
    finite_values = detached[torch.isfinite(detached)]
    if finite_values.numel() == 0:
        return None
    return float(finite_values.abs().max().item())


def _collect_gradient_stats(model: nn.Module) -> tuple[float | None, float | None, str | None]:
    total_sq = 0.0
    max_abs = 0.0
    saw_grad = False
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        saw_grad = True
        detached = grad.detach()
        if detached.numel() == 0:
            continue
        if not torch.isfinite(detached).all():
            return None, _safe_max_abs_tensor(detached), name
        grad_float = detached.float()
        total_sq += float(torch.sum(grad_float * grad_float).item())
        grad_max_abs = _safe_max_abs_tensor(detached)
        if grad_max_abs is not None:
            max_abs = max(max_abs, grad_max_abs)
    if not saw_grad:
        return 0.0, 0.0, None
    return math.sqrt(total_sq), max_abs, None


def _collect_parameter_stats(model: nn.Module) -> tuple[float | None, str | None]:
    max_abs = 0.0
    for name, parameter in model.named_parameters():
        detached = parameter.detach()
        if detached.numel() == 0:
            continue
        if not torch.isfinite(detached).all():
            return _safe_max_abs_tensor(detached), name
        param_max_abs = _safe_max_abs_tensor(detached)
        if param_max_abs is not None:
            max_abs = max(max_abs, param_max_abs)
    return max_abs, None


def _should_collect_detailed_finiteness_stats(
    *,
    epoch: int,
    batch_idx: int,
    detailed_check_batches: int,
) -> bool:
    return detailed_check_batches > 0 and epoch == 1 and batch_idx <= detailed_check_batches


def _build_non_finite_distillation_error(
    *,
    phase: str,
    epoch: int,
    batch_idx: int,
    run_label: str,
    stage: str,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    ce_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    total_loss: torch.Tensor,
    aux_loss: torch.Tensor | None,
    backend_label: str | None = None,
    grad_norm: float | None = None,
    grad_max_abs: float | None = None,
    param_max_abs: float | None = None,
    non_finite_grad: str | None = None,
    non_finite_param: str | None = None,
) -> RuntimeError:
    issues = _collect_non_finite_distillation_issues(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        ce_loss=ce_loss,
        kd_loss=kd_loss,
        total_loss=total_loss,
        aux_loss=aux_loss,
    )
    if non_finite_grad is not None:
        issues.append("gradients")
    if non_finite_param is not None:
        issues.append("parameters")
    issue_text = ", ".join(issues) if issues else "unknown"
    backend_text = f", backend={backend_label}" if backend_label is not None else ""
    return RuntimeError(
        "Non-finite distillation values detected for "
        f"{run_label} during {phase} epoch {epoch} batch {batch_idx} "
        f"(stage={stage}{backend_text}). "
        f"issues={issue_text} | "
        f"teacher_max_abs={_format_max_abs(teacher_logits)} | "
        f"student_max_abs={_format_max_abs(student_logits)} | "
        f"ce_loss={_format_scalar(ce_loss)} | "
        f"kd_loss={_format_scalar(kd_loss)} | "
        f"aux_loss={_format_scalar(aux_loss)} | "
        f"total_loss={_format_scalar(total_loss)} | "
        f"grad_norm={_format_float(grad_norm)} | "
        f"grad_max_abs={_format_float(grad_max_abs)} | "
        f"param_max_abs={_format_float(param_max_abs)} | "
        f"non_finite_grad={non_finite_grad or 'n/a'} | "
        f"non_finite_param={non_finite_param or 'n/a'}"
    )


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
    return ce_loss, kd_loss, aux_loss, total_loss


def probe_distillation_warmup(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    *,
    aux_loss_weight: float = 0.0,
    gradient_clip_norm: float | None = None,
    warmup_batches: int = 20,
    run_label: str = "distillation_probe",
    phase: str = "warmup_probe",
    backend_label: str | None = None,
    check_post_step_finiteness: bool = True,
    detailed_check_batches: int = 0,
    problem_type: str = "classification",
) -> None:
    if warmup_batches <= 0:
        return

    optimizer = build_optimizer(student_model, settings)
    hard_loss = build_task_criterion(problem_type)
    teacher_was_training = teacher_model.training
    student_was_training = student_model.training
    teacher_model.eval()
    student_model.train()
    try:
        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            if batch_idx > warmup_batches:
                break
            inputs = move_batch_to_device(inputs, device)
            labels = prepare_labels(labels.to(device), problem_type)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = forward_logits(teacher_model, inputs)
            student_logits = forward_logits(student_model, inputs)
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
            collect_detailed_stats = check_post_step_finiteness and _should_collect_detailed_finiteness_stats(
                epoch=1,
                batch_idx=batch_idx,
                detailed_check_batches=detailed_check_batches,
            )
            param_max_abs = None
            if collect_detailed_stats:
                param_max_abs, _ = _collect_parameter_stats(student_model)
            if _collect_non_finite_distillation_issues(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                ce_loss=ce_loss,
                kd_loss=kd_loss,
                total_loss=loss,
                aux_loss=aux_loss,
            ):
                if param_max_abs is None:
                    param_max_abs, _ = _collect_parameter_stats(student_model)
                raise _build_non_finite_distillation_error(
                    phase=phase,
                    epoch=1,
                    batch_idx=batch_idx,
                    run_label=run_label,
                    stage="pre-step",
                    teacher_logits=teacher_logits,
                    student_logits=student_logits,
                    ce_loss=ce_loss,
                    kd_loss=kd_loss,
                    total_loss=loss,
                    aux_loss=aux_loss,
                    backend_label=backend_label,
                    param_max_abs=param_max_abs,
                )
            loss.backward()
            grad_norm = None
            grad_max_abs = None
            if collect_detailed_stats:
                grad_norm, grad_max_abs, non_finite_grad = _collect_gradient_stats(student_model)
                if non_finite_grad is not None:
                    raise _build_non_finite_distillation_error(
                        phase=phase,
                        epoch=1,
                        batch_idx=batch_idx,
                        run_label=run_label,
                        stage="post-backward",
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        ce_loss=ce_loss,
                        kd_loss=kd_loss,
                        total_loss=loss,
                        aux_loss=aux_loss,
                        backend_label=backend_label,
                        grad_norm=grad_norm,
                        grad_max_abs=grad_max_abs,
                        param_max_abs=param_max_abs,
                        non_finite_grad=non_finite_grad,
                    )
            if gradient_clip_norm is not None:
                clipped_grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=gradient_clip_norm)
                if bool(torch.isfinite(clipped_grad_norm).item()):
                    if grad_norm is None:
                        grad_norm = float(clipped_grad_norm.item())
                elif check_post_step_finiteness:
                    grad_norm, grad_max_abs, non_finite_grad = _collect_gradient_stats(student_model)
                    if param_max_abs is None:
                        param_max_abs, _ = _collect_parameter_stats(student_model)
                    raise _build_non_finite_distillation_error(
                        phase=phase,
                        epoch=1,
                        batch_idx=batch_idx,
                        run_label=run_label,
                        stage="post-backward",
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        ce_loss=ce_loss,
                        kd_loss=kd_loss,
                        total_loss=loss,
                        aux_loss=aux_loss,
                        backend_label=backend_label,
                        grad_norm=grad_norm,
                        grad_max_abs=grad_max_abs,
                        param_max_abs=param_max_abs,
                        non_finite_grad=non_finite_grad or "unknown",
                    )
            optimizer.step()
            if collect_detailed_stats:
                param_max_abs, non_finite_param = _collect_parameter_stats(student_model)
                if non_finite_param is not None:
                    raise _build_non_finite_distillation_error(
                        phase=phase,
                        epoch=1,
                        batch_idx=batch_idx,
                        run_label=run_label,
                        stage="post-step",
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        ce_loss=ce_loss,
                        kd_loss=kd_loss,
                        total_loss=loss,
                        aux_loss=aux_loss,
                        backend_label=backend_label,
                        grad_norm=grad_norm,
                        grad_max_abs=grad_max_abs,
                        param_max_abs=param_max_abs,
                        non_finite_param=non_finite_param,
                    )
    finally:
        teacher_model.train(teacher_was_training)
        student_model.train(student_was_training)


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
    raise ValueError(f"Unsupported optimizer: {settings.optimizer_name}")


def build_scheduler(optimizer: optim.Optimizer, settings: TrainSettings):
    if not settings.lr_milestones:
        return None
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(settings.lr_milestones),
        gamma=settings.lr_gamma,
    )


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
    avg_train_batch_ms: float,
    eval_split_name: str = "test",
    primary_metric_name: str = "accuracy",
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
    metrics_payload = {
        "epoch": epoch,
        "epochs": settings.epochs,
        "phase": phase,
        "eval_split": eval_split_name,
        "train_objective": train_objective,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "test_loss": eval_loss,
        f"{split_key_prefix}_loss": eval_loss,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": eval_primary,
        "train_primary_metric_value": train_primary,
        "epoch_time_seconds": epoch_time_seconds,
        "avg_train_batch_ms": avg_train_batch_ms,
    }
    for metric_name, metric_value in train_metrics.items():
        metrics_payload[f"train_{metric_name}"] = float(metric_value)
    for metric_name, metric_value in test_metrics.items():
        if metric_name == "loss":
            continue
        metric_value = float(metric_value)
        metrics_payload[f"eval_{metric_name}"] = metric_value
        metrics_payload[f"test_{metric_name}"] = metric_value
        metrics_payload[f"{split_key_prefix}_{metric_name}"] = metric_value
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
        f"epoch_time={epoch_time_seconds:.2f}s | "
        f"avg_batch={avg_train_batch_ms:.2f}ms"
    )


def _summarize_history(
    history: Mapping[str, list[float]],
    *,
    eval_split_name: str,
    primary_metric_name: str,
    primary_metric_display: str,
) -> dict[str, float | int | str | None]:
    train_loss = history.get("train_loss", [])
    train_primary = history.get("train_accuracy", [])
    eval_loss = history.get("test_loss", [])
    eval_primary = history.get("test_accuracy", [])
    best_eval_metric = max(eval_primary) if eval_primary else None
    best_eval_epoch = eval_primary.index(best_eval_metric) + 1 if best_eval_metric is not None else 0
    summary: dict[str, float | int | str | None] = {
        "epochs_completed": len(eval_primary),
        "eval_split": eval_split_name,
        "primary_metric_name": primary_metric_name,
        "primary_metric_display": primary_metric_display,
        "best_eval_metric": best_eval_metric,
        "best_eval_epoch": best_eval_epoch,
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
) -> dict[str, list[float]]:
    criterion = build_task_criterion(problem_type)
    optimizer = build_optimizer(model, settings)
    scheduler = build_scheduler(optimizer, settings)
    history = create_history_template()
    logger = logger or build_run_logger()

    if settings.legacy_eval_sticky:
        model.train()
    for epoch in range(settings.epochs):
        epoch_start = time.perf_counter()
        if not settings.legacy_eval_sticky:
            model.train()
        running_objective = 0.0
        running_ce_loss = 0.0
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
            ensure_finite_loss_tensor(loss, f"{phase} epoch {epoch + 1} supervised training")
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            running_objective += loss.item() * batch_size
            running_ce_loss += ce_loss.item() * batch_size
            total_examples += batch_size
            train_predictions.append(predictions.detach().cpu())
            train_labels.append(metric_labels.detach().cpu())
        if scheduler is not None:
            scheduler.step()

        train_objective = ensure_finite_scalar(
            running_objective / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_objective",
        )
        train_loss = ensure_finite_scalar(
            running_ce_loss / max(total_examples, 1),
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
        epoch_time_seconds = ensure_finite_scalar(
            time.perf_counter() - epoch_start,
            f"{phase} epoch {epoch + 1} epoch_time_seconds",
        )
        avg_train_batch_ms = ensure_finite_scalar(
            1000.0 * epoch_time_seconds / max(batch_count, 1),
            f"{phase} epoch {epoch + 1} avg_train_batch_ms",
        )
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
            avg_train_batch_ms=avg_train_batch_ms,
            eval_split_name=eval_split_name,
            primary_metric_name=primary_metric_name,
        )
    _log_training_summary(
        history,
        logger,
        eval_split_name=eval_split_name,
        primary_metric_name=primary_metric_name,
        primary_metric_display=primary_metric_display,
    )
    return history


def train_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    aux_loss_weight: float = 0.0,
    gradient_clip_norm: float | None = None,
    logger: RunLogger | None = None,
    phase: str = "target",
    run_label: str | None = None,
    backend_label: str | None = None,
    check_post_step_finiteness: bool = False,
    detailed_check_batches: int = 0,
    eval_split_name: str = "test",
    primary_metric_name: str = "accuracy",
    primary_metric_display: str = "Accuracy (%)",
    metric_names: tuple[str, ...] = ("accuracy",),
    problem_type: str = "classification",
    num_labels: int = 0,
) -> dict[str, list[float]]:
    hard_loss = build_task_criterion(problem_type)
    optimizer = build_optimizer(student_model, settings)
    scheduler = build_scheduler(optimizer, settings)
    history = create_history_template()
    logger = logger or build_run_logger()
    run_label = run_label or phase

    teacher_model.eval()
    if settings.legacy_eval_sticky:
        student_model.train()
    for epoch in range(settings.epochs):
        epoch_start = time.perf_counter()
        if not settings.legacy_eval_sticky:
            student_model.train()
        running_objective = 0.0
        running_ce_loss = 0.0
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
            collect_detailed_stats = check_post_step_finiteness and _should_collect_detailed_finiteness_stats(
                epoch=epoch + 1,
                batch_idx=batch_idx,
                detailed_check_batches=detailed_check_batches,
            )
            param_max_abs = None
            if collect_detailed_stats:
                param_max_abs, _ = _collect_parameter_stats(student_model)
            if _collect_non_finite_distillation_issues(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                ce_loss=ce_loss,
                kd_loss=kd_loss,
                total_loss=loss,
                aux_loss=aux_loss,
            ):
                if param_max_abs is None:
                    param_max_abs, _ = _collect_parameter_stats(student_model)
                raise _build_non_finite_distillation_error(
                    phase=phase,
                    epoch=epoch + 1,
                    batch_idx=batch_idx,
                    run_label=run_label,
                    stage="pre-step",
                    teacher_logits=teacher_logits,
                    student_logits=student_logits,
                    ce_loss=ce_loss,
                    kd_loss=kd_loss,
                    total_loss=loss,
                    aux_loss=aux_loss,
                    backend_label=backend_label,
                    param_max_abs=param_max_abs,
                )
            loss.backward()
            grad_norm = None
            grad_max_abs = None
            if collect_detailed_stats:
                grad_norm, grad_max_abs, non_finite_grad = _collect_gradient_stats(student_model)
                if non_finite_grad is not None:
                    raise _build_non_finite_distillation_error(
                        phase=phase,
                        epoch=epoch + 1,
                        batch_idx=batch_idx,
                        run_label=run_label,
                        stage="post-backward",
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        ce_loss=ce_loss,
                        kd_loss=kd_loss,
                        total_loss=loss,
                        aux_loss=aux_loss,
                        backend_label=backend_label,
                        grad_norm=grad_norm,
                        grad_max_abs=grad_max_abs,
                        param_max_abs=param_max_abs,
                        non_finite_grad=non_finite_grad,
                    )
            if gradient_clip_norm is not None:
                clipped_grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=gradient_clip_norm)
                if bool(torch.isfinite(clipped_grad_norm).item()):
                    if grad_norm is None:
                        grad_norm = float(clipped_grad_norm.item())
                elif check_post_step_finiteness:
                    grad_norm, grad_max_abs, non_finite_grad = _collect_gradient_stats(student_model)
                    if param_max_abs is None:
                        param_max_abs, _ = _collect_parameter_stats(student_model)
                    raise _build_non_finite_distillation_error(
                        phase=phase,
                        epoch=epoch + 1,
                        batch_idx=batch_idx,
                        run_label=run_label,
                        stage="post-backward",
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        ce_loss=ce_loss,
                        kd_loss=kd_loss,
                        total_loss=loss,
                        aux_loss=aux_loss,
                        backend_label=backend_label,
                        grad_norm=grad_norm,
                        grad_max_abs=grad_max_abs,
                        param_max_abs=param_max_abs,
                        non_finite_grad=non_finite_grad or "unknown",
                    )
            optimizer.step()
            if collect_detailed_stats:
                param_max_abs, non_finite_param = _collect_parameter_stats(student_model)
                if non_finite_param is not None:
                    raise _build_non_finite_distillation_error(
                        phase=phase,
                        epoch=epoch + 1,
                        batch_idx=batch_idx,
                        run_label=run_label,
                        stage="post-step",
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        ce_loss=ce_loss,
                        kd_loss=kd_loss,
                        total_loss=loss,
                        aux_loss=aux_loss,
                        backend_label=backend_label,
                        grad_norm=grad_norm,
                        grad_max_abs=grad_max_abs,
                        param_max_abs=param_max_abs,
                        non_finite_param=non_finite_param,
                    )
            batch_size = labels.size(0)
            running_objective += loss.item() * batch_size
            running_ce_loss += ce_loss.item() * batch_size
            total_examples += batch_size
            if problem_type == "regression":
                predictions = prepare_regression_outputs(student_logits)
                metric_labels = labels.view_as(predictions)
            else:
                predictions = student_logits.argmax(dim=1)
                metric_labels = labels
            train_predictions.append(predictions.detach().cpu())
            train_labels.append(metric_labels.detach().cpu())
        if scheduler is not None:
            scheduler.step()

        train_objective = ensure_finite_scalar(
            running_objective / max(total_examples, 1),
            f"{phase} epoch {epoch + 1} train_objective",
        )
        train_loss = ensure_finite_scalar(
            running_ce_loss / max(total_examples, 1),
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
        epoch_time_seconds = ensure_finite_scalar(
            time.perf_counter() - epoch_start,
            f"{phase} epoch {epoch + 1} epoch_time_seconds",
        )
        avg_train_batch_ms = ensure_finite_scalar(
            1000.0 * epoch_time_seconds / max(batch_count, 1),
            f"{phase} epoch {epoch + 1} avg_train_batch_ms",
        )
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
            avg_train_batch_ms=avg_train_batch_ms,
            eval_split_name=eval_split_name,
            primary_metric_name=primary_metric_name,
        )
    _log_training_summary(
        history,
        logger,
        eval_split_name=eval_split_name,
        primary_metric_name=primary_metric_name,
        primary_metric_display=primary_metric_display,
    )
    return history


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
