from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_registry import (
    DATASET_REGISTRY,
    METHOD_CHOICES,
    PROJECT_DIR,
    SUITE_SPECS,
    DatasetSpec,
    TrainSettings,
    build_method_tag,
    build_pair_model,
    build_training_dataloaders,
    get_pair_spec,
    get_role_name,
    get_suite_run_specs,
    resolve_compressed_source,
    resolve_compressed_train_mode,
    resolve_device,
    resolve_fixed_rank_with_override,
    resolve_hetero_compress_linear,
    resolve_head_num,
    resolve_suite_log_dir,
    resolve_train_settings,
    set_seed,
    validate_args,
)
from model_wrappers import GenericHeteroNet, GenericInherNet
from plotting_utils import get_pyplot, plot_single_history, plot_suite_comparison_from_logs
from training_utils import (
    RunLogger,
    build_task_criterion,
    build_run_logger,
    compute_distillation_objective,
    count_parameters,
    forward_logits,
    move_batch_to_device,
    probe_distillation_warmup,
    train_distillation,
    train_supervised,
)

SUITE_BACKGROUND_ENV_VAR = "INHERNET_SUITE_BACKGROUND"
COMPRESSED_MODEL_WARMUP_BATCHES = 4
COMPRESSED_MODEL_DETAILED_CHECK_BATCHES = 64


class CompressedModelSanityError(RuntimeError):
    pass


def should_echo_suite_logger() -> bool:
    return os.environ.get(SUITE_BACKGROUND_ENV_VAR) != "1"


def describe_svd_backend(backend: str, device: torch.device) -> str:
    if backend == "cpu" or device.type == "cpu":
        return "cpu"
    return device.type


def _format_debug_scalar(value: torch.Tensor | None) -> str:
    if value is None:
        return "n/a"
    detached = value.detach()
    if detached.numel() == 0:
        return "empty"
    scalar = detached.item() if detached.numel() == 1 else detached.mean().item()
    return f"{float(scalar):.6g}"


def _format_debug_max_abs(value: torch.Tensor | None) -> str:
    if value is None:
        return "n/a"
    detached = value.detach()
    if detached.numel() == 0:
        return "empty"
    return f"{float(detached.abs().max().item()):.6g}"


def validate_compressed_distillation_setup(
    *,
    method: str,
    backend_label: str,
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    aux_loss_weight: float = 0.0,
    problem_type: str = "classification",
) -> None:
    try:
        inputs, labels = next(iter(train_loader))
    except StopIteration:
        return

    inputs = move_batch_to_device(inputs, device)
    labels = labels.to(device)
    criterion = build_task_criterion(problem_type)
    teacher_was_training = teacher_model.training
    student_was_training = student_model.training
    teacher_model.eval()
    student_model.eval()
    try:
        with torch.no_grad():
            teacher_logits = forward_logits(teacher_model, inputs)
            student_logits = forward_logits(student_model, inputs)
            ce_loss, kd_loss, aux_loss, total_loss = compute_distillation_objective(
                teacher_logits,
                student_logits,
                labels,
                settings,
                student_model=student_model,
                aux_loss_weight=aux_loss_weight,
                criterion=criterion,
                problem_type=problem_type,
            )
    finally:
        teacher_model.train(teacher_was_training)
        student_model.train(student_was_training)

    issues: list[str] = []
    for name, value in (
        ("teacher_logits", teacher_logits),
        ("student_logits", student_logits),
        ("ce_loss", ce_loss),
        ("kd_loss", kd_loss),
        ("aux_loss", aux_loss),
        ("total_loss", total_loss),
    ):
        if value is not None and not torch.isfinite(value).all():
            issues.append(name)
    if issues:
        raise CompressedModelSanityError(
            f"{method} sanity check failed using {backend_label}. "
            f"issues={', '.join(issues)} | "
            f"teacher_max_abs={_format_debug_max_abs(teacher_logits)} | "
            f"student_max_abs={_format_debug_max_abs(student_logits)} | "
            f"ce_loss={_format_debug_scalar(ce_loss)} | "
            f"kd_loss={_format_debug_scalar(kd_loss)} | "
            f"aux_loss={_format_debug_scalar(aux_loss)} | "
            f"total_loss={_format_debug_scalar(total_loss)}"
        )


def validate_compressed_distillation_warmup(
    *,
    method: str,
    backend_label: str,
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    aux_loss_weight: float = 0.0,
    gradient_clip_norm: float | None = None,
    warmup_batches: int = COMPRESSED_MODEL_WARMUP_BATCHES,
    check_post_step_finiteness: bool = True,
    detailed_check_batches: int = COMPRESSED_MODEL_WARMUP_BATCHES,
    problem_type: str = "classification",
) -> None:
    try:
        probe_distillation_warmup(
            teacher_model,
            student_model,
            train_loader,
            settings,
            device,
            aux_loss_weight=aux_loss_weight,
            gradient_clip_norm=gradient_clip_norm,
            warmup_batches=warmup_batches,
            run_label=method,
            backend_label=backend_label,
            check_post_step_finiteness=check_post_step_finiteness,
            detailed_check_batches=detailed_check_batches,
            problem_type=problem_type,
        )
    except RuntimeError as exc:
        raise CompressedModelSanityError(
            f"{method} warmup probe failed using {backend_label}: {exc}"
        ) from exc


def build_validated_compressed_model(
    *,
    method: str,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    settings: TrainSettings,
    device: torch.device,
    logger: RunLogger,
    build_model: Callable[[str], tuple[nn.Module, dict[str, Any], str]],
    aux_loss_weight: float = 0.0,
    initial_svd_backend: str = "auto",
    retry_svd_backend: str | None = "cpu",
    gradient_clip_norm: float | None = None,
    warmup_batches: int = COMPRESSED_MODEL_WARMUP_BATCHES,
    check_post_step_finiteness: bool = True,
    detailed_check_batches: int = COMPRESSED_MODEL_DETAILED_CHECK_BATCHES,
    problem_type: str = "classification",
) -> tuple[nn.Module, dict[str, Any]]:
    loader_generator = getattr(train_loader, "generator", None)

    def build_and_probe_model(requested_backend: str) -> tuple[nn.Module, dict[str, Any], str]:
        python_rng_state = random.getstate()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        loader_generator_state = loader_generator.get_state() if loader_generator is not None else None
        probe_model: nn.Module | None = None
        try:
            probe_model, _, probed_backend = build_model(requested_backend)
            backend_label = describe_svd_backend(probed_backend, device)
            validate_compressed_distillation_setup(
                method=method,
                backend_label=backend_label,
                teacher_model=teacher_model,
                student_model=probe_model,
                train_loader=train_loader,
                settings=settings,
                device=device,
                aux_loss_weight=aux_loss_weight,
                problem_type=problem_type,
            )
            validate_compressed_distillation_warmup(
                method=method,
                backend_label=backend_label,
                teacher_model=teacher_model,
                student_model=probe_model,
                train_loader=train_loader,
                settings=settings,
                device=device,
                aux_loss_weight=aux_loss_weight,
                gradient_clip_norm=gradient_clip_norm,
                warmup_batches=warmup_batches,
                check_post_step_finiteness=check_post_step_finiteness,
                detailed_check_batches=min(detailed_check_batches, warmup_batches),
                problem_type=problem_type,
            )
        finally:
            if probe_model is not None:
                del probe_model
            random.setstate(python_rng_state)
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            if loader_generator is not None and loader_generator_state is not None:
                loader_generator.set_state(loader_generator_state)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model, extra, svd_backend = build_model(probed_backend)
        backend_label = describe_svd_backend(svd_backend, device)
        extra = dict(extra)
        extra["svd_backend"] = backend_label
        logger.info(f"{method} decomposition backend: {backend_label}")
        return model, extra, svd_backend

    try:
        model, extra, svd_backend = build_and_probe_model(initial_svd_backend)
    except CompressedModelSanityError as exc:
        if retry_svd_backend is None or initial_svd_backend == retry_svd_backend or device.type == "cpu":
            raise
        logger.info(str(exc))
        logger.info(f"{method} retrying with {retry_svd_backend.upper()} SVD fallback.")
        model, extra, svd_backend = build_and_probe_model(retry_svd_backend)
    return model, extra


def build_run_metadata(
    method: str,
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any],
    settings: TrainSettings,
    model: nn.Module,
    config_tag: str,
    *,
    suite_name: str | None = None,
    suite_label: str | None = None,
    rank_preset_override: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_spec = DATASET_REGISTRY[args.dataset]
    metadata: dict[str, Any] = {
        "dataset": args.dataset,
        "pair": args.pair,
        "method": method,
        "task_type": dataset_spec.task_type,
        "problem_type": dataset_spec.problem_type,
        "num_classes": dataset_spec.num_classes,
        "train_split": dataset_spec.train_split or "train",
        "eval_split": dataset_spec.eval_split_name,
        "primary_metric_name": dataset_spec.primary_metric_name,
        "primary_metric_display": dataset_spec.primary_metric_display,
        "metric_names": list(dataset_spec.metric_names),
        "config_tag": config_tag,
        "plot_tag": config_tag,
        "teacher_arch": get_role_name(pair_spec, "teacher"),
        "student_arch": get_role_name(pair_spec, "student"),
        "num_parameters": count_parameters(model),
        "train_settings": asdict(settings),
    }
    if dataset_spec.task_type == "vision":
        metadata["image_size"] = dataset_spec.image_size
    if dataset_spec.text_task_name is not None:
        metadata["text_task_name"] = dataset_spec.text_task_name
        metadata["text_max_length"] = dataset_spec.text_max_length
    if "model_profile" in pair_spec:
        metadata["model_profile"] = str(pair_spec["model_profile"])
    if suite_name is not None:
        metadata["suite_name"] = suite_name
    if suite_label is not None:
        metadata["suite_label"] = suite_label
    if method == "inhernet":
        metadata["rank_preset"] = (
            rank_preset_override
            if rank_preset_override is not None
            else ("custom" if args.rank is not None else args.rank_preset)
        )
    if extra is not None:
        metadata.update(extra)
    return metadata


def maybe_save_single_plot(
    plot_root: Path,
    metadata: Mapping[str, Any],
    history: Mapping[str, Any],
    plot_mode: str,
    logger: RunLogger | None = None,
) -> Path | None:
    if plot_mode not in {"single", "both"}:
        return None
    output_path = plot_single_history(plot_root, metadata, history, plot_mode)
    if output_path is not None and logger is not None:
        logger.info(f"Saved plot: {output_path}")
    return output_path


def maybe_save_suite_comparison_plot(
    plot_root: Path,
    suite_log_dir: Path,
    dataset_name: str,
    pair_name: str,
    plot_mode: str,
    logger: RunLogger | None = None,
) -> Path | None:
    if plot_mode not in {"compare", "both"}:
        return None
    output_path = plot_suite_comparison_from_logs(plot_root, suite_log_dir, dataset_name, pair_name, plot_mode)
    if output_path is not None and logger is not None:
        logger.info(f"Saved comparison plot: {output_path}")
    return output_path


def train_teacher_pretrain(
    args: argparse.Namespace,
    dataset_spec: DatasetSpec,
    settings: TrainSettings,
    device: torch.device,
    logger: RunLogger,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(args.seed)
    train_loader, test_loader = build_training_dataloaders(args, settings, device)
    teacher_model = build_pair_model(args.dataset, args.pair, "teacher", dataset_spec.num_classes).to(device)
    logger.info("Training teacher from scratch for the target method.")
    history = train_supervised(
        teacher_model,
        train_loader,
        test_loader,
        settings,
        device,
        logger=logger,
        phase="teacher_pretrain",
        eval_split_name=dataset_spec.eval_split_name,
        primary_metric_name=dataset_spec.primary_metric_name,
        primary_metric_display=dataset_spec.primary_metric_display,
        metric_names=dataset_spec.metric_names,
        problem_type=dataset_spec.problem_type,
        num_labels=dataset_spec.num_classes,
    )
    teacher_model.eval()
    return teacher_model, history


def train_student_pretrain(
    args: argparse.Namespace,
    dataset_spec: DatasetSpec,
    settings: TrainSettings,
    device: torch.device,
    logger: RunLogger,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(args.seed)
    train_loader, test_loader = build_training_dataloaders(args, settings, device)
    student_model = build_pair_model(args.dataset, args.pair, "student", dataset_spec.num_classes).to(device)
    logger.info("Training student source model from scratch for compressed-model initialization.")
    history = train_supervised(
        student_model,
        train_loader,
        test_loader,
        settings,
        device,
        logger=logger,
        phase="student_source_pretrain",
        eval_split_name=dataset_spec.eval_split_name,
        primary_metric_name=dataset_spec.primary_metric_name,
        primary_metric_display=dataset_spec.primary_metric_display,
        metric_names=dataset_spec.metric_names,
        problem_type=dataset_spec.problem_type,
        num_labels=dataset_spec.num_classes,
    )
    student_model.eval()
    return student_model, history


def train_method_from_scratch(
    args: argparse.Namespace,
    method: str,
    pair_spec: Mapping[str, Any],
    dataset_spec: DatasetSpec,
    settings: TrainSettings,
    device: torch.device,
    logger: RunLogger,
    *,
    teacher_model: nn.Module | None = None,
    source_student_model: nn.Module | None = None,
    suite_name: str | None = None,
    suite_label: str | None = None,
    rank_preset_override: str | None = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    set_seed(args.seed)
    train_loader, test_loader = build_training_dataloaders(args, settings, device)
    config_tag = build_method_tag(method, args, pair_spec, settings, rank_preset_override)
    head_num = resolve_head_num(args, pair_spec, settings)
    compressed_source = resolve_compressed_source(args, pair_spec)
    compressed_train_mode = resolve_compressed_train_mode(args, pair_spec)
    hetero_compress_linear = resolve_hetero_compress_linear(args, pair_spec)
    metric_log_kwargs = {
        "eval_split_name": dataset_spec.eval_split_name,
        "primary_metric_name": dataset_spec.primary_metric_name,
        "primary_metric_display": dataset_spec.primary_metric_display,
        "metric_names": dataset_spec.metric_names,
        "problem_type": dataset_spec.problem_type,
        "num_labels": dataset_spec.num_classes,
    }

    if method == "teacher":
        model = build_pair_model(args.dataset, args.pair, "teacher", dataset_spec.num_classes).to(device)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
        )
        logger.metadata(metadata)
        history = train_supervised(
            model,
            train_loader,
            test_loader,
            settings,
            device,
            logger=logger,
            **metric_log_kwargs,
        )
    elif method == "student":
        model = build_pair_model(args.dataset, args.pair, "student", dataset_spec.num_classes).to(device)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
        )
        logger.metadata(metadata)
        history = train_supervised(
            model,
            train_loader,
            test_loader,
            settings,
            device,
            logger=logger,
            **metric_log_kwargs,
        )
    elif method == "student_kd":
        if teacher_model is None:
            raise ValueError("student_kd requires an in-memory teacher model.")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        model = build_pair_model(args.dataset, args.pair, "student", dataset_spec.num_classes).to(device)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
        )
        logger.metadata(metadata)
        history = train_distillation(
            teacher_model,
            model,
            train_loader,
            test_loader,
            settings,
            device,
            logger=logger,
            run_label=method,
            **metric_log_kwargs,
        )
    elif method == "inhernet":
        if teacher_model is None:
            raise ValueError("inhernet requires an in-memory teacher model.")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        if compressed_source == "teacher":
            dense_source_model = teacher_model
            dense_source_role = "teacher"
        elif compressed_source == "student":
            if source_student_model is None:
                raise ValueError("inhernet with --compressed-source student requires an in-memory source student model.")
            dense_source_model = source_student_model.to(device)
            dense_source_model.eval()
            dense_source_role = "student"
        else:
            raise ValueError(f"Unsupported compressed source: {compressed_source}")
        rank = resolve_fixed_rank_with_override(args, pair_spec, rank_preset_override)
        dense_state_cpu = {
            name: tensor.detach().cpu().clone()
            for name, tensor in dense_source_model.state_dict().items()
        }

        def build_inhernet_model(svd_backend: str) -> tuple[nn.Module, dict[str, Any], str]:
            build_device = torch.device("cpu") if svd_backend == "cpu" else device
            model = GenericInherNet(
                build_pair_model(args.dataset, args.pair, dense_source_role, dataset_spec.num_classes)
            ).to(build_device)
            model.load_dense_state_dict(dense_state_cpu)
            used_backend = model.apply_svd(rank=rank, head_num=head_num, svd_backend=svd_backend)
            if build_device != device:
                model = model.to(device)
            return (
                model,
                {
                    "rank": rank,
                    "head_num": head_num,
                    "compressed_from": dense_source_role,
                    "compressed_train_mode": compressed_train_mode,
                },
                used_backend,
            )

        model, inhernet_extra = build_validated_compressed_model(
            method=method,
            teacher_model=teacher_model,
            train_loader=train_loader,
            settings=settings,
            device=device,
            logger=logger,
            build_model=build_inhernet_model,
            initial_svd_backend=args.svd_backend,
            retry_svd_backend="cpu" if args.svd_backend != "cpu" else None,
            gradient_clip_norm=5.0,
            detailed_check_batches=COMPRESSED_MODEL_DETAILED_CHECK_BATCHES,
            problem_type=dataset_spec.problem_type,
        )
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
            rank_preset_override=rank_preset_override,
            extra=inhernet_extra,
        )
        logger.metadata(metadata)
        if compressed_train_mode == "supervised":
            history = train_supervised(
                model,
                train_loader,
                test_loader,
                settings,
                device,
                logger=logger,
                **metric_log_kwargs,
            )
        else:
            history = train_distillation(
                teacher_model,
                model,
                train_loader,
                test_loader,
                settings,
                device,
                gradient_clip_norm=5.0,
                logger=logger,
                run_label=method,
                backend_label=str(inhernet_extra["svd_backend"]),
                check_post_step_finiteness=True,
                detailed_check_batches=COMPRESSED_MODEL_DETAILED_CHECK_BATCHES,
                **metric_log_kwargs,
            )
    elif method == "hetero":
        if teacher_model is None:
            raise ValueError("hetero requires an in-memory teacher model.")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        if compressed_source == "teacher":
            dense_source_model = teacher_model
            dense_source_role = "teacher"
        elif compressed_source == "student":
            if source_student_model is None:
                raise ValueError("hetero with --compressed-source student requires an in-memory source student model.")
            dense_source_model = source_student_model.to(device)
            dense_source_model.eval()
            dense_source_role = "student"
        else:
            raise ValueError(f"Unsupported compressed source: {compressed_source}")
        dense_state = dense_source_model.state_dict()
        dense_state_cpu = {name: tensor.detach().cpu().clone() for name, tensor in dense_state.items()}

        def build_hetero_model(svd_backend: str) -> tuple[nn.Module, dict[str, Any], str]:
            build_device = torch.device("cpu") if svd_backend == "cpu" else device
            model = GenericHeteroNet(
                build_pair_model(args.dataset, args.pair, dense_source_role, dataset_spec.num_classes)
            ).to(build_device)
            model.load_dense_state_dict(dense_state_cpu if build_device.type == "cpu" else dense_state)
            rank_map, used_backend = model.apply_hetero_svd(
                calib_loader=train_loader,
                head_num=head_num,
                budget_ratio=args.budget_ratio,
                min_rank=args.min_rank,
                compress_threshold=args.compress_threshold,
                temperature=args.hetero_temperature,
                max_calib_batches=args.max_calib_batches,
                svd_backend=svd_backend,
                expert_noise_scale=args.hetero_expert_noise_scale,
                compress_linear=hetero_compress_linear,
            )
            rank_values = list(rank_map.values())
            avg_rank = sum(rank_values) / len(rank_values)
            target_layer_types = ["conv2d"]
            if hetero_compress_linear:
                target_layer_types.append("linear")
            extra = {
                "head_num": head_num,
                "budget_ratio": args.budget_ratio,
                "min_rank": args.min_rank,
                "compress_threshold": args.compress_threshold,
                "hetero_temperature": args.hetero_temperature,
                "max_calib_batches": args.max_calib_batches,
                "aux_loss_weight": args.aux_loss_weight,
                "hetero_expert_noise_scale": args.hetero_expert_noise_scale,
                "hetero_compress_linear": hetero_compress_linear,
                "target_layer_types": target_layer_types,
                "rank_map": {name: int(rank) for name, rank in rank_map.items()},
                "avg_rank": avg_rank,
                "rank_min": min(rank_values),
                "rank_max": max(rank_values),
                "compressed_from": dense_source_role,
                "compressed_train_mode": compressed_train_mode,
            }
            if build_device != device:
                model = model.to(device)
            return model, extra, used_backend

        model, hetero_extra = build_validated_compressed_model(
            method=method,
            teacher_model=teacher_model,
            train_loader=train_loader,
            settings=settings,
            device=device,
            logger=logger,
            build_model=build_hetero_model,
            aux_loss_weight=args.aux_loss_weight,
            initial_svd_backend=args.svd_backend,
            retry_svd_backend="cpu" if args.svd_backend != "cpu" else None,
            gradient_clip_norm=5.0,
            detailed_check_batches=COMPRESSED_MODEL_DETAILED_CHECK_BATCHES,
            problem_type=dataset_spec.problem_type,
        )
        rank_map = hetero_extra["rank_map"]
        avg_rank = float(hetero_extra["avg_rank"])
        rank_values = [int(rank) for rank in rank_map.values()]
        logger.info(
            "Hetero rank allocation: "
            f"min={min(rank_values)}, max={max(rank_values)}, avg={avg_rank:.2f}"
        )
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            suite_name=suite_name,
            suite_label=suite_label,
            extra=hetero_extra,
        )
        logger.metadata(metadata)
        if compressed_train_mode == "supervised":
            history = train_supervised(
                model,
                train_loader,
                test_loader,
                settings,
                device,
                aux_loss_weight=args.aux_loss_weight,
                logger=logger,
                **metric_log_kwargs,
            )
        else:
            history = train_distillation(
                teacher_model,
                model,
                train_loader,
                test_loader,
                settings,
                device,
                aux_loss_weight=args.aux_loss_weight,
                gradient_clip_norm=5.0,
                logger=logger,
                run_label=method,
                backend_label=str(hetero_extra["svd_backend"]),
                check_post_step_finiteness=True,
                detailed_check_batches=COMPRESSED_MODEL_DETAILED_CHECK_BATCHES,
                **metric_log_kwargs,
            )
    else:
        raise ValueError(f"Unsupported method: {method}")

    model.eval()
    return model, history, metadata


def build_smoke_sample(dataset_spec: DatasetSpec, device: torch.device):
    if dataset_spec.task_type == "text":
        return {
            "input_ids": torch.randint(0, 1000, (2, min(dataset_spec.text_max_length, 16)), device=device),
            "attention_mask": torch.ones(2, min(dataset_spec.text_max_length, 16), dtype=torch.long, device=device),
        }
    return torch.randn(2, 3, dataset_spec.image_size, dataset_spec.image_size, device=device)


def build_smoke_calibration_loader(dataset_spec: DatasetSpec) -> DataLoader:
    if dataset_spec.task_type == "text":
        sequence_length = min(dataset_spec.text_max_length, 16)
        input_ids = torch.randint(0, 1000, (8, sequence_length))
        attention_mask = torch.ones(8, sequence_length, dtype=torch.long)
        labels = torch.zeros(8, dtype=torch.long)
        tensor_dataset = TensorDataset(input_ids, attention_mask, labels)

        def collate_text_smoke(batch):
            ids, masks, batch_labels = zip(*batch)
            return (
                {
                    "input_ids": torch.stack(ids),
                    "attention_mask": torch.stack(masks),
                },
                torch.stack(batch_labels),
            )

        return DataLoader(tensor_dataset, batch_size=2, shuffle=False, collate_fn=collate_text_smoke)

    calib_inputs = torch.randn(8, 3, dataset_spec.image_size, dataset_spec.image_size)
    calib_labels = torch.zeros(8, dtype=torch.long)
    return DataLoader(TensorDataset(calib_inputs, calib_labels), batch_size=2, shuffle=False)


def run_single_method_smoke_test(
    dataset_name: str,
    pair_name: str,
    method: str,
    args: argparse.Namespace,
    rank_preset_override: str | None = None,
) -> dict[str, Any]:
    dataset_spec = DATASET_REGISTRY[dataset_name]
    pair_spec = get_pair_spec(dataset_name, pair_name)
    settings = resolve_train_settings(dataset_spec, args, pair_spec)
    head_num = resolve_head_num(args, pair_spec, settings)
    compressed_source = resolve_compressed_source(args, pair_spec)
    hetero_compress_linear = resolve_hetero_compress_linear(args, pair_spec)
    device = resolve_device(args.device)
    sample = build_smoke_sample(dataset_spec, device)
    calib_loader = build_smoke_calibration_loader(dataset_spec)

    if method == "teacher":
        model = build_pair_model(dataset_name, pair_name, "teacher", dataset_spec.num_classes).to(device)
        with torch.no_grad():
            output = forward_logits(model, sample)
        return {"method": method, "shape": tuple(output.shape), "params": count_parameters(model)}
    if method == "student":
        model = build_pair_model(dataset_name, pair_name, "student", dataset_spec.num_classes).to(device)
        with torch.no_grad():
            output = forward_logits(model, sample)
        return {"method": method, "shape": tuple(output.shape), "params": count_parameters(model)}
    if method == "student_kd":
        teacher = build_pair_model(dataset_name, pair_name, "teacher", dataset_spec.num_classes).to(device)
        student = build_pair_model(dataset_name, pair_name, "student", dataset_spec.num_classes).to(device)
        with torch.no_grad():
            teacher_out = forward_logits(teacher, sample)
            student_out = forward_logits(student, sample)
        return {
            "method": method,
            "teacher_shape": tuple(teacher_out.shape),
            "student_shape": tuple(student_out.shape),
        }
    if method == "inhernet":
        source_role = "student" if compressed_source == "student" else "teacher"
        dense_source = build_pair_model(dataset_name, pair_name, source_role, dataset_spec.num_classes).to(device)
        model = GenericInherNet(copy.deepcopy(dense_source)).to(device)
        model.load_dense_state_dict(dense_source.state_dict())
        rank = resolve_fixed_rank_with_override(args, pair_spec, rank_preset_override)
        svd_backend = model.apply_svd(rank=rank, head_num=head_num, svd_backend=args.svd_backend)
        with torch.no_grad():
            output = forward_logits(model, sample)
        return {
            "method": method,
            "shape": tuple(output.shape),
            "params": count_parameters(model),
            "rank": rank,
            "head_num": head_num,
            "compressed_from": source_role,
            "svd_backend": describe_svd_backend(svd_backend, device),
        }
    if method == "hetero":
        source_role = "student" if compressed_source == "student" else "teacher"
        dense_source = build_pair_model(dataset_name, pair_name, source_role, dataset_spec.num_classes).to(device)
        model = GenericHeteroNet(copy.deepcopy(dense_source)).to(device)
        model.load_dense_state_dict(dense_source.state_dict())
        rank_map, svd_backend = model.apply_hetero_svd(
            calib_loader=calib_loader,
            head_num=head_num,
            budget_ratio=args.budget_ratio,
            min_rank=args.min_rank,
            compress_threshold=args.compress_threshold,
            temperature=args.hetero_temperature,
            max_calib_batches=min(args.max_calib_batches, len(calib_loader)),
            svd_backend=args.svd_backend,
            expert_noise_scale=args.hetero_expert_noise_scale,
            compress_linear=hetero_compress_linear,
        )
        with torch.no_grad():
            output = forward_logits(model, sample)
        assert args.compress_threshold > args.min_rank
        return {
            "method": method,
            "shape": tuple(output.shape),
            "params": count_parameters(model),
            "head_num": head_num,
            "rank_min": min(rank_map.values()),
            "rank_max": max(rank_map.values()),
            "compressed_from": source_role,
            "svd_backend": describe_svd_backend(svd_backend, device),
        }
    raise ValueError(f"Unknown method: {method}")


def run_single_method(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    dataset_spec = DATASET_REGISTRY[args.dataset]
    pair_spec = get_pair_spec(args.dataset, args.pair)
    settings = resolve_train_settings(dataset_spec, args, pair_spec)
    validate_args(args, pair_spec)
    logger = build_run_logger(echo=True, store_info_to_file=True)

    if args.smoke_test:
        result = run_single_method_smoke_test(args.dataset, args.pair, args.method, args)
        logger.info(f"Smoke test passed: {result}")
        return Path("<smoke-test>")

    get_pyplot(args.plot_mode)
    plot_root = Path(args.plot_root)

    if args.method in {"teacher", "student"}:
        logger.info(f"Training {args.method} from scratch.")
        _, history, metadata = train_method_from_scratch(
            args,
            args.method,
            pair_spec,
            dataset_spec,
            settings,
            device,
            logger,
        )
    else:
        teacher_model, _ = train_teacher_pretrain(args, dataset_spec, settings, device, logger)
        source_student_model = None
        compressed_source = resolve_compressed_source(args, pair_spec)
        if args.method in {"inhernet", "hetero"} and compressed_source == "student":
            source_student_model, _ = train_student_pretrain(args, dataset_spec, settings, device, logger)
        logger.info(
            f"Training {args.method} from scratch using compressed_source={compressed_source} "
            "and the in-memory teacher for KD when requested."
        )
        _, history, metadata = train_method_from_scratch(
            args,
            args.method,
            pair_spec,
            dataset_spec,
            settings,
            device,
            logger,
            teacher_model=teacher_model,
            source_student_model=source_student_model,
        )

    plot_path = maybe_save_single_plot(plot_root, metadata, history, args.plot_mode, logger)
    return plot_path if plot_path is not None else Path("<no-plot>")


def run_suite_smoke_test(args: argparse.Namespace) -> Path:
    suite_log_dir = resolve_suite_log_dir(args)
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_logger = build_run_logger(
        str(suite_log_dir / "suite.log"),
        echo=should_echo_suite_logger(),
        store_info_to_file=True,
    )
    suite_logger.info(
        f"Suite smoke test started: dataset={args.dataset}, pair={args.pair}, suite={args.suite}"
    )
    for spec in get_suite_run_specs(args.suite):
        label = str(spec["label"])
        child_logger = build_run_logger(str(suite_log_dir / f"{label}.log"), echo=False, store_info_to_file=True)
        result = run_single_method_smoke_test(
            args.dataset,
            args.pair,
            str(spec["method"]),
            args,
            rank_preset_override=spec.get("rank_preset"),
        )
        child_logger.info(f"Smoke test passed: {result}")
        suite_logger.info(f"[{label}] Smoke test passed.")
    suite_logger.info("Suite smoke test completed successfully.")
    return suite_log_dir


def run_suite(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    dataset_spec = DATASET_REGISTRY[args.dataset]
    pair_spec = get_pair_spec(args.dataset, args.pair)
    settings = resolve_train_settings(dataset_spec, args, pair_spec)
    validate_args(args, pair_spec)

    if args.smoke_test:
        return run_suite_smoke_test(args)

    get_pyplot(args.plot_mode)
    plot_root = Path(args.plot_root)
    suite_log_dir = resolve_suite_log_dir(args)
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_logger = build_run_logger(
        str(suite_log_dir / "suite.log"),
        echo=should_echo_suite_logger(),
        store_info_to_file=True,
    )
    suite_logger.info(
        f"Suite started: dataset={args.dataset}, pair={args.pair}, suite={args.suite}. All runs train from scratch."
    )

    teacher_model: nn.Module | None = None
    source_student_model: nn.Module | None = None
    compressed_source = resolve_compressed_source(args, pair_spec)
    for spec in get_suite_run_specs(args.suite):
        label = str(spec["label"])
        method = str(spec["method"])
        rank_preset_override = spec.get("rank_preset")
        child_log_path = suite_log_dir / f"{label}.log"
        child_logger = build_run_logger(str(child_log_path), echo=False, store_info_to_file=True)

        suite_logger.info(f"[{label}] Starting {method}.")
        child_logger.info(
            f"Run started: dataset={args.dataset}, pair={args.pair}, method={method}, suite={args.suite}, label={label}"
        )
        try:
            if method == "teacher":
                teacher_model, history, metadata = train_method_from_scratch(
                    args,
                    method,
                    pair_spec,
                    dataset_spec,
                    settings,
                    device,
                    child_logger,
                    suite_name=args.suite,
                    suite_label=label,
                    rank_preset_override=rank_preset_override,
                )
            elif method == "student":
                model, history, metadata = train_method_from_scratch(
                    args,
                    method,
                    pair_spec,
                    dataset_spec,
                    settings,
                    device,
                    child_logger,
                    suite_name=args.suite,
                    suite_label=label,
                    rank_preset_override=rank_preset_override,
                )
                if compressed_source == "student":
                    source_student_model = model
                else:
                    del model
            else:
                if teacher_model is None:
                    raise RuntimeError("Suite execution requires the teacher step to complete before dependent methods.")
                if method in {"inhernet", "hetero"} and compressed_source == "student" and source_student_model is None:
                    child_logger.info(
                        "Pretraining source student because this suite does not include a reusable student step before compressed methods."
                    )
                    source_student_model, _ = train_student_pretrain(args, dataset_spec, settings, device, child_logger)
                model, history, metadata = train_method_from_scratch(
                    args,
                    method,
                    pair_spec,
                    dataset_spec,
                    settings,
                    device,
                    child_logger,
                    teacher_model=teacher_model,
                    source_student_model=source_student_model,
                    suite_name=args.suite,
                    suite_label=label,
                    rank_preset_override=rank_preset_override,
                )
                del model

            plot_path = maybe_save_single_plot(plot_root, metadata, history, args.plot_mode, child_logger)
            if plot_path is not None:
                suite_logger.info(f"[{label}] Saved plot: {plot_path}")
            comparison_path = maybe_save_suite_comparison_plot(
                plot_root,
                suite_log_dir,
                args.dataset,
                args.pair,
                args.plot_mode,
                suite_logger,
            )
            if comparison_path is not None:
                suite_logger.info(f"[{label}] Refreshed comparison plot.")
            suite_logger.info(f"[{label}] Completed.")
        except Exception as exc:
            suite_logger.info(f"[{label}] Failed: {exc}")
            raise
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    suite_logger.info("Suite completed successfully.")
    return suite_log_dir


def run_training(args: argparse.Namespace) -> Path:
    if args.suite is not None:
        return run_suite(args)
    return run_single_method(args)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registry-driven HeteroInherNet runner for vision and GLUE tasks.")
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY.keys()), required=True)
    parser.add_argument("--pair", required=True, help="Dataset-specific teacher/student pair name.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--method", choices=METHOD_CHOICES)
    mode_group.add_argument("--suite", choices=sorted(SUITE_SPECS.keys()))
    parser.add_argument("--data-root", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--kd-temperature", type=float, default=None)
    parser.add_argument("--kd-weight", type=float, default=None)
    parser.add_argument("--ce-weight", type=float, default=None)
    parser.add_argument(
        "--legacy-eval-sticky",
        action="store_true",
        help=(
            "Force demo_code_org.py train/eval behavior, where epoch evaluation leaves the model "
            "in eval mode for the next training epoch. Some compatibility pairs enable this by default."
        ),
    )
    parser.add_argument("--rank-preset", choices=["small", "large"], default="small")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--head-num", type=int, default=None)
    parser.add_argument(
        "--compressed-source",
        choices=["teacher", "student"],
        default=None,
        help=(
            "Dense model to decompose for InherNet/Hetero. Defaults come from the pair: "
            "teacher for paper-style pairs, student for demo_code_org.py compatibility pairs."
        ),
    )
    parser.add_argument(
        "--compressed-train-mode",
        choices=["distillation", "supervised"],
        default=None,
        help=(
            "Training objective for compressed InherNet/Hetero models after decomposition. "
            "Defaults come from the pair."
        ),
    )
    parser.add_argument(
        "--svd-backend",
        choices=["auto", "device", "cpu"],
        default="auto",
        help="SVD backend for compressed models. auto tries the model device first and then CPU fallback.",
    )
    parser.add_argument("--budget-ratio", type=float, default=0.35)
    parser.add_argument("--min-rank", type=int, default=8)
    parser.add_argument("--compress-threshold", type=int, default=12)
    parser.add_argument("--hetero-temperature", type=float, default=1.4)
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument("--aux-loss-weight", type=float, default=0.01)
    parser.add_argument("--hetero-expert-noise-scale", type=float, default=0.01)
    parser.add_argument(
        "--hetero-compress-linear",
        action="store_true",
        help="Also apply Hetero SVD to linear layers. The default keeps prior CIFAR experiments conv-only.",
    )
    parser.add_argument("--plot-mode", choices=["none", "single", "compare", "both"], default="both")
    parser.add_argument("--plot-root", default=str(PROJECT_DIR / "results"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
