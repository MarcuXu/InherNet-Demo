from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from checkpointing import load_teacher_checkpoint, save_teacher_checkpoint
from experiment_registry import (
    DATASET_REGISTRY,
    METHOD_CHOICES,
    PROJECT_DIR,
    DatasetSpec,
    TrainSettings,
    build_method_tag,
    build_pair_model,
    build_training_dataloaders,
    get_pair_spec,
    get_role_name,
    resolve_compressed_train_mode,
    resolve_capacity_size,
    resolve_device,
    resolve_fixed_rank,
    resolve_compress_linear,
    resolve_head_num,
    resolve_train_settings,
    set_seed,
    validate_args,
)
from model_wrappers import (
    FINAL_HETERO_ALLOCATION,
    HETERO_ALLOCATION_SCALES,
    RESEARCH_HETERO_RANK_POLICIES,
    GenericHeteroNet,
    GenericInherNet,
    freeze_gating_routers,
)
from plotting_utils import get_pyplot, plot_single_history
from training_utils import (
    INHERITANCE_DIAGNOSTICS_PREFIX,
    RunLogger,
    build_run_logger,
    count_parameters,
    evaluate_inheritance_diagnostics,
    forward_logits,
    train_distillation,
    train_supervised,
)


def maybe_log_inheritance_diagnostics(
    args: argparse.Namespace,
    teacher_model: nn.Module,
    inherited_model: nn.Module,
    evaluation_loader: DataLoader,
    evaluation_split: str,
    dataset_spec: DatasetSpec,
    device: torch.device,
    logger: RunLogger,
) -> None:
    if not (args.inheritance_diagnostics or args.inheritance_diagnostics_only):
        return
    diagnostics = evaluate_inheritance_diagnostics(
        teacher_model,
        inherited_model,
        evaluation_loader,
        device,
        problem_type=dataset_spec.problem_type,
        num_labels=dataset_spec.num_classes,
        metric_names=dataset_spec.metric_names,
        evaluation_split=evaluation_split,
    )
    logger.structured(INHERITANCE_DIAGNOSTICS_PREFIX, diagnostics, echo=True)

def describe_svd_backend(backend: str, device: torch.device) -> str:
    if backend == "cpu" or device.type == "cpu":
        return "cpu"
    return device.type


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_run_metadata(
    method: str,
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any],
    settings: TrainSettings,
    model: nn.Module,
    config_tag: str,
    *,
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
        "seed": args.seed,
        "argv": list(sys.argv),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(next(model.parameters()).device),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "training_rng_reset_after_initialization": True,
    }
    if args.search_candidate is not None:
        metadata["search_candidate"] = args.search_candidate
    if dataset_spec.task_type == "vision":
        metadata["image_size"] = dataset_spec.image_size
    if dataset_spec.text_task_name is not None:
        metadata["text_task_name"] = dataset_spec.text_task_name
        metadata["text_max_length"] = dataset_spec.text_max_length
    if "model_profile" in pair_spec:
        metadata["model_profile"] = str(pair_spec["model_profile"])
    for provenance_key in ("inhernet_protocol_source", "inhernet_rank_source"):
        if provenance_key in pair_spec:
            metadata[provenance_key] = str(pair_spec[provenance_key])
    for revision_key in ("teacher_revision", "student_revision", "tokenizer_revision"):
        if revision_key in pair_spec:
            metadata[revision_key] = str(pair_spec[revision_key])
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


def resolve_teacher_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.teacher_checkpoint is not None:
        return Path(args.teacher_checkpoint).expanduser().resolve()
    return (
        Path(args.checkpoint_root)
        / args.dataset
        / args.pair
        / f"teacher_seed_{args.seed}.pt"
    ).resolve()


def _selection_details(
    history: Mapping[str, list[float]],
    dataset_spec: DatasetSpec,
    settings: TrainSettings,
    run_metadata: Mapping[str, Any],
) -> tuple[str, int, dict[str, Any]]:
    evaluation_values = history.get("test_accuracy", [])
    eval_split_name = str(run_metadata.get("eval_split", dataset_spec.eval_split_name))
    if eval_split_name != "test" and evaluation_values:
        best_value = max(evaluation_values)
        selected_epoch = evaluation_values.index(best_value) + 1
        policy = f"best_{eval_split_name}_{dataset_spec.primary_metric_name}"
    else:
        selected_epoch = settings.epochs
        policy = "final_epoch"
    metrics: dict[str, Any] = {
        "selected_evaluation_metric": (
            evaluation_values[selected_epoch - 1]
            if evaluation_values and selected_epoch > 0
            else None
        ),
        "selected_evaluation_metrics": {
            key.removeprefix("eval_metric_"): values[selected_epoch - 1]
            for key, values in history.items()
            if key.startswith("eval_metric_")
            and selected_epoch > 0
            and len(values) >= selected_epoch
        },
        "final_test_metric": (
            history.get("final_test_accuracy", [None])[-1]
            if history.get("final_test_accuracy")
            else None
        ),
    }
    return policy, selected_epoch, metrics


def persist_teacher_checkpoint(
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any],
    dataset_spec: DatasetSpec,
    settings: TrainSettings,
    model: nn.Module,
    history: Mapping[str, list[float]],
    run_metadata: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    policy, selected_epoch, metrics = _selection_details(
        history,
        dataset_spec,
        settings,
        run_metadata,
    )
    return save_teacher_checkpoint(
        path,
        model,
        dataset=args.dataset,
        pair=args.pair,
        architecture=get_role_name(pair_spec, "teacher"),
        num_classes=dataset_spec.num_classes,
        seed=args.seed,
        settings=settings,
        model_profile=str(pair_spec.get("model_profile", "unspecified")),
        data_profile=dataset_spec.data_profile,
        selection_policy=policy,
        selected_epoch=selected_epoch,
        metrics=metrics,
        data_split=run_metadata.get("data_split"),
        overwrite=args.overwrite_teacher_checkpoint,
    )


def semantic_split_metadata(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Drop legacy integrity-only fields before split comparison or reporting."""
    if value is None:
        return None
    return {
        key: item
        for key, item in value.items()
        if "fingerprint" not in key.lower()
        and "sha" not in key.lower()
        and "hash" not in key.lower()
    }


def teacher_training_split_metadata(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return only fields that affect which examples trained/selected the teacher."""
    semantic = semantic_split_metadata(value)
    if semantic is None:
        return None
    return {
        key: item
        for key, item in semantic.items()
        if item is not None
        and not key.startswith("calibration_")
        and not key.startswith("official_evaluation_")
    }


def load_frozen_teacher(
    args: argparse.Namespace,
    pair_spec: Mapping[str, Any],
    dataset_spec: DatasetSpec,
    device: torch.device,
    path: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    teacher = build_pair_model(
        args.dataset,
        args.pair,
        "teacher",
        dataset_spec.num_classes,
        initialize_pretrained=False,
    )
    info = load_teacher_checkpoint(
        path,
        teacher,
        dataset=args.dataset,
        pair=args.pair,
        architecture=get_role_name(pair_spec, "teacher"),
        num_classes=dataset_spec.num_classes,
        seed=args.seed,
        model_profile=str(pair_spec.get("model_profile", "unspecified")),
        data_profile=dataset_spec.data_profile,
        expected_settings=dataset_spec.train_settings,
    )
    return teacher.to(device), info


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
    teacher_checkpoint_info: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    set_seed(args.seed)
    loaders = build_training_dataloaders(args, settings, device)
    train_loader, test_loader = loaders.train, loaders.evaluation
    config_tag = build_method_tag(method, args, pair_spec, settings)
    head_num = resolve_head_num(args, pair_spec, settings)
    compressed_train_mode = resolve_compressed_train_mode(args, pair_spec)
    compress_linear = resolve_compress_linear(pair_spec)
    metric_log_kwargs = {
        "eval_split_name": loaders.eval_split_name,
        "primary_metric_name": dataset_spec.primary_metric_name,
        "primary_metric_display": dataset_spec.primary_metric_display,
        "metric_names": dataset_spec.metric_names,
        "problem_type": dataset_spec.problem_type,
        "num_labels": dataset_spec.num_classes,
        "final_test_loader": loaders.final_test,
        "final_test_split_name": loaders.final_test_split_name or "test",
        "restore_best_state": loaders.restore_best_state,
    }
    run_context: dict[str, Any] = {}
    if loaders.split_metadata is not None:
        run_context["data_split"] = semantic_split_metadata(loaders.split_metadata)
    run_context["eval_split"] = loaders.eval_split_name
    if loaders.final_test_split_name is not None:
        run_context["final_test_split"] = loaders.final_test_split_name
    if teacher_checkpoint_info is not None:
        checkpoint_split = teacher_training_split_metadata(
            teacher_checkpoint_info.get("data_split")
        )
        current_split = teacher_training_split_metadata(run_context.get("data_split"))
        if checkpoint_split != current_split:
            raise ValueError(
                "Teacher checkpoint data split does not match the current experiment split."
            )
    teacher_lineage = dict(run_context)
    if teacher_checkpoint_info is not None:
        teacher_lineage["teacher_checkpoint"] = dict(teacher_checkpoint_info)

    if method == "teacher":
        model = build_pair_model(args.dataset, args.pair, "teacher", dataset_spec.num_classes).to(device)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            extra=run_context,
        )
        logger.metadata(metadata)
        set_seed(args.seed)
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
            extra=run_context,
        )
        logger.metadata(metadata)
        set_seed(args.seed)
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
            raise ValueError("student_kd requires a frozen teacher model loaded from a checkpoint.")
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
            extra=teacher_lineage,
        )
        logger.metadata(metadata)
        set_seed(args.seed)
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
            raise ValueError("InherNet requires a frozen teacher model loaded from a checkpoint.")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        dense_source_model = teacher_model
        dense_source_role = "teacher"
        rank = resolve_fixed_rank(args, pair_spec)
        model = GenericInherNet(
            build_pair_model(
                args.dataset,
                args.pair,
                dense_source_role,
                dataset_spec.num_classes,
                initialize_pretrained=False,
            )
        ).to(device)
        model.load_dense_state_dict(dense_source_model.state_dict())
        synchronize_device(device)
        setup_start = time.perf_counter()
        used_backend = model.apply_svd(
            rank=rank,
            head_num=head_num,
            svd_backend=args.svd_backend,
            include_linear=compress_linear,
        )
        synchronize_device(device)
        inheritance_setup_seconds = time.perf_counter() - setup_start
        inhernet_extra = {
            "rank": rank,
            "size": "custom" if args.rank is not None else resolve_capacity_size(args),
            "head_num": head_num,
            "compressed_from": dense_source_role,
            "compressed_train_mode": compressed_train_mode,
            "protocol": "inhernet_shared_down_normalized_router",
            "compress_linear": compress_linear,
            "target_layer_types": ["linear"] if compress_linear else ["conv2d"],
            "inheritance_setup_seconds": inheritance_setup_seconds,
        }
        inhernet_extra["svd_backend"] = describe_svd_backend(used_backend, device)
        logger.info(f"{method} decomposition backend: {inhernet_extra['svd_backend']}")
        inhernet_extra.update(teacher_lineage)
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            extra=inhernet_extra,
        )
        logger.metadata(metadata)
        maybe_log_inheritance_diagnostics(
            args,
            dense_source_model,
            model,
            test_loader,
            loaders.eval_split_name,
            dataset_spec,
            device,
            logger,
        )
        set_seed(args.seed)
        if args.inheritance_diagnostics_only:
            history = {}
        elif compressed_train_mode == "supervised":
            teacher_model.to("cpu")
            dense_source_model = None
            teacher_model = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
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
                logger=logger,
                run_label=method,
                **metric_log_kwargs,
            )
    elif method == "hetero":
        if teacher_model is None:
            raise ValueError("Hetero requires a frozen teacher model loaded from a checkpoint.")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        dense_source_model = teacher_model
        dense_source_role = "teacher"
        model = GenericHeteroNet(
            build_pair_model(
                args.dataset,
                args.pair,
                dense_source_role,
                dataset_spec.num_classes,
                initialize_pretrained=False,
            )
        ).to(device)
        model.load_dense_state_dict(dense_source_model.state_dict())
        reference_rank = resolve_fixed_rank(args, pair_spec)
        research_protected_rank = (
            min(reference_rank, int(pair_spec["rank_presets"]["small"]))
            if args.hetero_allocation_scale == "research_nested_relative"
            else None
        )
        synchronize_device(device)
        setup_start = time.perf_counter()
        rank_map, used_backend = model.apply_hetero_svd(
            calib_loader=loaders.calibration or train_loader,
            head_num=head_num,
            reference_rank=reference_rank,
            max_calib_batches=args.max_calib_batches,
            svd_backend=args.svd_backend,
            expert_noise_scale=args.hetero_expert_noise_scale,
            compress_linear=compress_linear,
            max_features_per_batch=args.hetero_max_features_per_batch,
            second_moment_shrinkage=args.hetero_second_moment_shrinkage,
            allocation_scale=args.hetero_allocation_scale,
            research_protected_rank=research_protected_rank,
            allow_research_rank_probe=args.inheritance_diagnostics_only,
        )
        if args.freeze_hetero_router:
            freeze_gating_routers(model)
        synchronize_device(device)
        inheritance_setup_seconds = time.perf_counter() - setup_start
        rank_values = list(rank_map.values())
        avg_rank = sum(rank_values) / len(rank_values) if rank_values else 0.0
        target_layer_types = ["linear"] if dataset_spec.task_type == "text" else ["conv2d"]
        if compress_linear and "linear" not in target_layer_types:
            target_layer_types.append("linear")
        hetero_extra = {
            "head_num": head_num,
            "size": resolve_capacity_size(args),
            "reference_inhernet_rank": reference_rank,
            "max_calib_batches": args.max_calib_batches,
            "aux_loss_weight": args.aux_loss_weight,
            "hetero_expert_noise_scale": args.hetero_expert_noise_scale,
            "compress_linear": compress_linear,
            "hetero_max_features_per_batch": args.hetero_max_features_per_batch,
            "hetero_second_moment_shrinkage": args.hetero_second_moment_shrinkage,
            "hetero_allocation_scale": args.hetero_allocation_scale,
            "freeze_hetero_router": args.freeze_hetero_router,
            "hetero_report": model.hetero_report,
            "target_layer_types": target_layer_types,
            "rank_map": {name: int(rank) for name, rank in rank_map.items()},
            "avg_rank": avg_rank,
            "rank_min": min(rank_values) if rank_values else None,
            "rank_max": max(rank_values) if rank_values else None,
            "compressed_from": dense_source_role,
            "compressed_train_mode": compressed_train_mode,
            "hetero_recipe_id": args.hetero_recipe_id,
            "inheritance_setup_seconds": inheritance_setup_seconds,
        }
        hetero_extra["svd_backend"] = describe_svd_backend(used_backend, device)
        logger.info(f"{method} decomposition backend: {hetero_extra['svd_backend']}")
        hetero_extra.update(teacher_lineage)
        if rank_values:
            decomposition_report = hetero_extra["hetero_report"]
            if args.hetero_allocation_scale in RESEARCH_HETERO_RANK_POLICIES:
                logger.info(
                    "Research rank probe: "
                    f"factorized={decomposition_report['factorized_layer_count']}/"
                    f"{decomposition_report['target_layer_count']}, "
                    f"min={min(rank_values)}, max={max(rank_values)}, avg={avg_rank:.2f}, "
                    f"budget_utilization="
                    f"{100.0 * decomposition_report['budget_utilization']:.2f}%"
                )
            else:
                logger.info(
                    "Hetero registered-rank decomposition: "
                    f"factorized={decomposition_report['factorized_layer_count']}/"
                    f"{decomposition_report['target_layer_count']}, "
                    f"rank={min(rank_values)}, exact_matched_parameters="
                    f"{decomposition_report['actual_parameters']}"
                )
        else:
            decomposition_report = hetero_extra["hetero_report"]
            logger.info(
                "Hetero decomposition kept all eligible layers dense; "
                f"parameters={decomposition_report['actual_parameters']}"
            )
        metadata = build_run_metadata(
            method,
            args,
            pair_spec,
            settings,
            model,
            config_tag,
            extra=hetero_extra,
        )
        logger.metadata(metadata)
        maybe_log_inheritance_diagnostics(
            args,
            dense_source_model,
            model,
            test_loader,
            loaders.eval_split_name,
            dataset_spec,
            device,
            logger,
        )
        set_seed(args.seed)
        if args.inheritance_diagnostics_only:
            history = {}
        elif compressed_train_mode == "supervised":
            teacher_model.to("cpu")
            dense_source_model = None
            teacher_model = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
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
                logger=logger,
                run_label=method,
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
) -> dict[str, Any]:
    dataset_spec = DATASET_REGISTRY[dataset_name]
    pair_spec = get_pair_spec(dataset_name, pair_name)
    settings = resolve_train_settings(dataset_spec, args, pair_spec)
    head_num = resolve_head_num(args, pair_spec, settings)
    compress_linear = resolve_compress_linear(pair_spec)
    device = resolve_device(args.device)
    sample = build_smoke_sample(dataset_spec, device)
    calib_loader = build_smoke_calibration_loader(dataset_spec)

    def build_smoke_model(role: str) -> nn.Module:
        return build_pair_model(
            dataset_name,
            pair_name,
            role,
            dataset_spec.num_classes,
            initialize_pretrained=False,
        )

    if method == "teacher":
        model = build_smoke_model("teacher").to(device)
        with torch.no_grad():
            output = forward_logits(model, sample)
        return {"method": method, "shape": tuple(output.shape), "params": count_parameters(model)}
    if method == "student":
        model = build_smoke_model("student").to(device)
        with torch.no_grad():
            output = forward_logits(model, sample)
        return {"method": method, "shape": tuple(output.shape), "params": count_parameters(model)}
    if method == "student_kd":
        teacher = build_smoke_model("teacher").to(device)
        student = build_smoke_model("student").to(device)
        with torch.no_grad():
            teacher_out = forward_logits(teacher, sample)
            student_out = forward_logits(student, sample)
        return {
            "method": method,
            "teacher_shape": tuple(teacher_out.shape),
            "student_shape": tuple(student_out.shape),
        }
    if method == "inhernet":
        source_role = "teacher"
        dense_source = build_smoke_model(source_role).to(device)
        model = GenericInherNet(copy.deepcopy(dense_source)).to(device)
        model.load_dense_state_dict(dense_source.state_dict())
        rank = resolve_fixed_rank(args, pair_spec)
        synchronize_device(device)
        setup_start = time.perf_counter()
        svd_backend = model.apply_svd(
            rank=rank,
            head_num=head_num,
            svd_backend=args.svd_backend,
            include_linear=compress_linear,
        )
        synchronize_device(device)
        inheritance_setup_seconds = time.perf_counter() - setup_start
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
            "inheritance_setup_seconds": inheritance_setup_seconds,
        }
    if method == "hetero":
        source_role = "teacher"
        dense_source = build_smoke_model(source_role).to(device)
        model = GenericHeteroNet(copy.deepcopy(dense_source)).to(device)
        model.load_dense_state_dict(dense_source.state_dict())
        reference_rank = resolve_fixed_rank(args, pair_spec)
        research_protected_rank = (
            min(reference_rank, int(pair_spec["rank_presets"]["small"]))
            if args.hetero_allocation_scale == "research_nested_relative"
            else None
        )
        synchronize_device(device)
        setup_start = time.perf_counter()
        rank_map, svd_backend = model.apply_hetero_svd(
            calib_loader=calib_loader,
            head_num=head_num,
            reference_rank=reference_rank,
            max_calib_batches=min(args.max_calib_batches, len(calib_loader)),
            svd_backend=args.svd_backend,
            expert_noise_scale=args.hetero_expert_noise_scale,
            compress_linear=compress_linear,
            max_features_per_batch=args.hetero_max_features_per_batch,
            second_moment_shrinkage=args.hetero_second_moment_shrinkage,
            allocation_scale=args.hetero_allocation_scale,
            research_protected_rank=research_protected_rank,
            allow_research_rank_probe=args.inheritance_diagnostics_only,
        )
        if args.freeze_hetero_router:
            freeze_gating_routers(model)
        synchronize_device(device)
        inheritance_setup_seconds = time.perf_counter() - setup_start
        with torch.no_grad():
            output = forward_logits(model, sample)
        rank_values = list(rank_map.values())
        return {
            "method": method,
            "shape": tuple(output.shape),
            "params": count_parameters(model),
            "head_num": head_num,
            "size": resolve_capacity_size(args),
            "reference_inhernet_rank": reference_rank,
            "hetero_allocation_scale": args.hetero_allocation_scale,
            "freeze_hetero_router": args.freeze_hetero_router,
            "achieved_ratio": model.hetero_report.get("achieved_ratio"),
            "rank_min": min(rank_values) if rank_values else None,
            "rank_max": max(rank_values) if rank_values else None,
            "compressed_from": source_role,
            "svd_backend": describe_svd_backend(svd_backend, device),
            "inheritance_setup_seconds": inheritance_setup_seconds,
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
        model, history, metadata = train_method_from_scratch(
            args,
            args.method,
            pair_spec,
            dataset_spec,
            settings,
            device,
            logger,
        )
        if args.method == "teacher":
            checkpoint_path = resolve_teacher_checkpoint_path(args)
            checkpoint_info = persist_teacher_checkpoint(
                args,
                pair_spec,
                dataset_spec,
                settings,
                model,
                history,
                metadata,
                checkpoint_path,
            )
            public_checkpoint_info = {
                key: checkpoint_info[key]
                for key in ("path", "schema_version", "selection_policy", "selected_epoch")
            }
            logger.structured("TEACHER_CHECKPOINT", public_checkpoint_info, echo=True)
    else:
        checkpoint_path = resolve_teacher_checkpoint_path(args)
        teacher_model, checkpoint_info = load_frozen_teacher(
            args,
            pair_spec,
            dataset_spec,
            device,
            checkpoint_path,
        )
        logger.info(f"Loaded frozen teacher checkpoint: {checkpoint_info['path']}")
        checkpoint_info = {
            key: checkpoint_info[key]
            for key in (
                "path",
                "schema_version",
                "selection_policy",
                "selected_epoch",
                "metrics",
                "data_split",
            )
        }
        checkpoint_info["data_split"] = semantic_split_metadata(checkpoint_info.get("data_split"))
        logger.info(f"Training {args.method}; the checkpoint-loaded teacher is frozen.")
        _, history, metadata = train_method_from_scratch(
            args,
            args.method,
            pair_spec,
            dataset_spec,
            settings,
            device,
            logger,
            teacher_model=teacher_model,
            teacher_checkpoint_info=checkpoint_info,
        )

    if args.inheritance_diagnostics_only:
        return Path("<diagnostics-only>")
    plot_path = maybe_save_single_plot(plot_root, metadata, history, args.plot_mode, logger)
    return plot_path if plot_path is not None else Path("<no-plot>")


def run_training(args: argparse.Namespace) -> Path:
    return run_single_method(args)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registry-driven Hetero runner for vision and GLUE tasks.")
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY.keys()), required=True)
    parser.add_argument("--pair", required=True, help="Dataset-specific teacher/student pair name.")
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--data-root", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Teacher .pt artifact to save for --method teacher or load for dependent methods.",
    )
    parser.add_argument("--checkpoint-root", default=str(PROJECT_DIR / "checkpoints"))
    parser.add_argument(
        "--overwrite-teacher-checkpoint",
        action="store_true",
        help="Allow a teacher command to replace an existing checkpoint.",
    )
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--kd-temperature", type=float, default=None)
    parser.add_argument("--kd-weight", type=float, default=None)
    parser.add_argument("--ce-weight", type=float, default=None)
    parser.add_argument(
        "--kd-fraction",
        type=float,
        default=None,
        help=(
            "Set KD to this fraction of the dataset's registered KD+label-loss weight total; "
            "mutually exclusive with --kd-weight and --ce-weight."
        ),
    )
    parser.add_argument(
        "--size",
        choices=["small", "large"],
        default=None,
        help=(
            "Registered capacity preset. When omitted, direct Hetero runs use "
            "the headline large preset and InherNet uses small. Hetero exactly "
            "matches the corresponding InherNet rank and parameter count."
        ),
    )
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--head-num", type=int, default=None)
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
    parser.add_argument("--max-calib-batches", type=int, default=16)
    parser.add_argument("--aux-loss-weight", type=float, default=0.01)
    parser.add_argument("--hetero-expert-noise-scale", type=float, default=0.01)
    parser.add_argument("--hetero-max-features-per-batch", type=int, default=4096)
    parser.add_argument("--hetero-second-moment-shrinkage", type=float, default=0.01)
    parser.add_argument(
        "--hetero-allocation-scale",
        choices=HETERO_ALLOCATION_SCALES,
        default=FINAL_HETERO_ALLOCATION,
        help=(
            "Hetero decomposition policy. weighted_uniform is the maintained method: "
            "activation-aware decomposition at InherNet's registered rank. "
            "unweighted_uniform is its weight-only ablation; research_* policies are "
            "explicit pre-study diagnostics and never formal/HPO settings."
        ),
    )
    parser.add_argument(
        "--hetero-recipe-id",
        default=None,
        help="Reviewed recipe identifier recorded for formal, confirmation, and ablation runs.",
    )
    parser.add_argument(
        "--freeze-hetero-router",
        action="store_true",
        help="Ablation control: keep the zero-initialized uniform Hetero routers fixed.",
    )
    parser.add_argument(
        "--inheritance-diagnostics",
        action="store_true",
        help="Log teacher-versus-inherited metrics before the first optimizer update.",
    )
    parser.add_argument(
        "--inheritance-diagnostics-only",
        action="store_true",
        help="Run initialization diagnostics and stop before optimizer construction.",
    )
    parser.add_argument("--plot-mode", choices=["none", "single", "compare", "both"], default="both")
    parser.add_argument(
        "--final-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate a dataset's held-out final test split after validation selection.",
    )
    parser.add_argument(
        "--search-validation",
        action="store_true",
        help="Use a fixed training holdout for CIFAR or GLUE hyperparameter search.",
    )
    parser.add_argument(
        "--search-candidate",
        default=None,
        help="Stable candidate identifier recorded by hyperparameter searches.",
    )
    parser.add_argument("--plot-root", default=str(PROJECT_DIR / "results"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
