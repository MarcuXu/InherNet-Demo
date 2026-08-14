from __future__ import annotations

import importlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from experiment_registry import sanitize_tag
from scripts.inheract_artifacts import canonicalize_metadata
from training_utils import (
    RUN_METADATA_PREFIX,
    RUN_METRICS_PREFIX,
    create_history_template,
    normalize_history,
)


_PYPLOT = None
_PLOT_THEME_APPLIED = False
PLOT_METRIC_SPECS = (
    ("train_loss", "Loss", "Train Loss", None),
    ("test_loss", "Loss", "Test Loss", None),
    ("train_accuracy", "Accuracy (%)", "Train Accuracy", (0.0, 100.0)),
    ("test_accuracy", "Accuracy (%)", "Test Accuracy", (0.0, 100.0)),
)
CIFAR100_PLOT_METRIC_SPECS = (
    ("train_loss", "Loss", "Train Loss", None),
    ("test_loss", "Loss", "Test Loss", None),
    ("train_accuracy", "Top-1 Accuracy (%)", "Train Top-1", (0.0, 100.0)),
    ("test_accuracy", "Top-1 Accuracy (%)", "Test Top-1", (0.0, 100.0)),
)


def get_pyplot(plot_mode: str):
    global _PYPLOT
    if plot_mode == "none":
        return None
    if _PYPLOT is not None:
        return _PYPLOT
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg", force=True)
        _PYPLOT = importlib.import_module("matplotlib.pyplot")
        apply_publication_plot_theme(_PYPLOT)
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it in the active environment or run with --plot-mode none."
        ) from exc
    return _PYPLOT


def apply_publication_plot_theme(plt) -> None:
    global _PLOT_THEME_APPLIED
    if _PLOT_THEME_APPLIED:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#424A57",
            "axes.linewidth": 1.1,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelcolor": "#2F3640",
            "xtick.color": "#4A5160",
            "ytick.color": "#4A5160",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": "#D8DDE6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "lines.linewidth": 2.6,
            "lines.solid_capstyle": "round",
        }
    )
    _PLOT_THEME_APPLIED = True


def history_has_curves(history: Mapping[str, Any] | None) -> bool:
    normalized = normalize_history(history)
    return any(normalized[key] for key, *_ in PLOT_METRIC_SPECS)


def get_plot_metric_specs(
    dataset_name: str,
    *,
    eval_split_name: str = "test",
    primary_metric_display: str = "Accuracy (%)",
):
    if dataset_name == "cifar100":
        return CIFAR100_PLOT_METRIC_SPECS
    split_label = eval_split_name.replace("_", " ").title()
    return (
        ("train_loss", "Loss", "Train Loss", None),
        ("test_loss", "Loss", f"{split_label} Loss", None),
        ("train_accuracy", primary_metric_display, f"Train {primary_metric_display}", (0.0, 100.0)),
        ("test_accuracy", primary_metric_display, f"{split_label} {primary_metric_display}", (0.0, 100.0)),
    )


def get_plot_method_key(method: str, metadata: Mapping[str, Any]) -> str:
    method = str(canonicalize_metadata({"method": method})["method"])
    config_tag = str(metadata.get("config_tag", "default"))
    if method == "inhernet":
        if metadata.get("search_candidate") == "ablation_direct_svd":
            return "direct_svd_inheritance"
        size = str(metadata.get("size", metadata.get("rank_preset", "")))
        if size == "small" or config_tag.startswith("small_rank_"):
            return "inhernet_small"
        if size == "large" or config_tag.startswith("large_rank_"):
            return "inhernet_large"
        return "inhernet_custom"
    return method


def get_plot_style(method_key: str) -> dict[str, Any]:
    styles: dict[str, dict[str, Any]] = {
        "teacher": {"color": "#111111", "linestyle": "-", "linewidth": 3.1, "alpha": 1.0, "zorder": 7},
        "student": {"color": "#8B8B8B", "linestyle": "--", "linewidth": 2.2, "alpha": 0.95, "zorder": 3},
        "student_kd": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "student_kd_logit_standardized": {"color": "#56B4E9", "linestyle": "-.", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "student_dkd": {"color": "#CC79A7", "linestyle": "--", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "student_ctkd": {"color": "#6F4E9C", "linestyle": "-.", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "student_catkd": {"color": "#E76F51", "linestyle": "--", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "student_simkd": {"color": "#2A9D8F", "linestyle": "-.", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "student_reviewkd": {"color": "#7A8F37", "linestyle": "--", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "student_crd": {"color": "#3D5A80", "linestyle": ":", "linewidth": 2.7, "alpha": 0.98, "zorder": 5},
        "direct_svd_inheritance": {"color": "#7B61A8", "linestyle": "--", "linewidth": 2.7, "alpha": 0.98, "zorder": 4},
        "inhernet_small": {"color": "#009E73", "linestyle": "-", "linewidth": 2.8, "alpha": 0.98, "zorder": 4},
        "inhernet_large": {"color": "#D55E00", "linestyle": "-", "linewidth": 2.8, "alpha": 0.98, "zorder": 4},
        "inhernet_custom": {"color": "#CC79A7", "linestyle": "-.", "linewidth": 2.6, "alpha": 0.97, "zorder": 4},
        "inheract": {"color": "#E69F00", "linestyle": "-", "linewidth": 3.0, "alpha": 1.0, "zorder": 6},
    }
    return styles.get(method_key, {"color": "#4C78A8", "linestyle": "-", "linewidth": 2.5, "alpha": 0.98, "zorder": 4})


def style_plot_axis(ax, ylabel: str, title: str) -> None:
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, axis="y", alpha=0.75)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.tick_params(length=4.5, width=1.0)
    ax.margins(x=0.02)


def set_axis_limits(ax, values: list[float], clamp: tuple[float | None, float | None] | None = None) -> None:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return
    min_value = min(finite_values)
    max_value = max(finite_values)
    span = max(max_value - min_value, 1e-6)
    lower = min_value - 0.08 * span
    upper = max_value + 0.12 * span
    if clamp is not None:
        lower_bound, upper_bound = clamp
        if lower_bound is not None:
            lower = max(lower_bound, lower)
        if upper_bound is not None:
            upper = min(upper_bound, upper)
    if math.isfinite(lower) and math.isfinite(upper):
        ax.set_ylim(lower, upper)


def add_endpoint_marker(ax, x_value: int, y_value: float, color: str) -> None:
    if not math.isfinite(float(y_value)):
        return
    ax.scatter(
        [x_value],
        [y_value],
        s=34,
        color=color,
        edgecolor="white",
        linewidth=0.9,
        zorder=10,
    )


def build_metric_summary(history: Mapping[str, Any], eval_split_name: str = "test") -> str:
    normalized = normalize_history(history)
    summary_lines = []
    train_loss = [value for value in normalized["train_loss"] if math.isfinite(value)]
    test_loss = [value for value in normalized["test_loss"] if math.isfinite(value)]
    train_accuracy = [value for value in normalized["train_accuracy"] if math.isfinite(value)]
    test_accuracy = [value for value in normalized["test_accuracy"] if math.isfinite(value)]
    train_objective = [value for value in normalized["train_objective"] if math.isfinite(value)]
    if train_loss:
        summary_lines.append(f"Train loss   {train_loss[-1]:.3f}")
    split_label = eval_split_name.replace("_", " ").title()
    if test_loss:
        summary_lines.append(f"{split_label} loss {test_loss[-1]:.3f}")
    if train_accuracy:
        summary_lines.append(f"Train acc    {train_accuracy[-1]:.2f}%")
    if test_accuracy:
        summary_lines.append(f"{split_label} metric {test_accuracy[-1]:.2f}%")
        summary_lines.append(f"Best {split_label.lower()} {max(test_accuracy):.2f}%")
    if train_objective and (not train_loss or abs(train_objective[-1] - train_loss[-1]) > 1e-8):
        summary_lines.append(f"Objective    {train_objective[-1]:.3f}")
    return "\n".join(summary_lines)


def draw_unavailable_metric(ax, ylabel: str, title: str) -> None:
    style_plot_axis(ax, ylabel, title)
    ax.text(
        0.5,
        0.5,
        "Metric unavailable\nin this run",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.0,
        color="#7A8596",
        bbox={
            "boxstyle": "round,pad=0.36",
            "facecolor": "#F7F8FB",
            "edgecolor": "#D7DCE5",
            "linewidth": 0.8,
        },
    )


def plot_single_metric_panel(
    ax,
    values: list[float],
    style: Mapping[str, Any],
    ylabel: str,
    title: str,
    clamp: tuple[float | None, float | None] | None = None,
) -> None:
    finite_pairs = [(idx + 1, float(value)) for idx, value in enumerate(values) if math.isfinite(float(value))]
    if not finite_pairs:
        draw_unavailable_metric(ax, ylabel, title)
        return
    x_values = [item[0] for item in finite_pairs]
    y_values = [item[1] for item in finite_pairs]
    ax.plot(
        x_values,
        y_values,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=max(2.5, float(style["linewidth"])),
        alpha=style["alpha"],
        zorder=style["zorder"],
    )
    add_endpoint_marker(ax, x_values[-1], y_values[-1], str(style["color"]))
    style_plot_axis(ax, ylabel, title)
    set_axis_limits(ax, y_values, clamp=clamp)


def plot_comparison_metric_panel(
    ax,
    records: list[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    clamp: tuple[float | None, float | None] | None = None,
) -> None:
    plotted_values: list[float] = []
    for record in records:
        values = list(record["history"].get(metric_key, []))
        finite_pairs = [(idx + 1, float(value)) for idx, value in enumerate(values) if math.isfinite(float(value))]
        if not finite_pairs:
            continue
        x_values = [item[0] for item in finite_pairs]
        y_values = [item[1] for item in finite_pairs]
        style = get_plot_style(record["method_key"])
        ax.plot(
            x_values,
            y_values,
            label=record["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
        add_endpoint_marker(ax, x_values[-1], y_values[-1], str(style["color"]))
        plotted_values.extend(y_values)

    if not plotted_values:
        draw_unavailable_metric(ax, ylabel, title)
        return
    style_plot_axis(ax, ylabel, title)
    set_axis_limits(ax, plotted_values, clamp=clamp)


def build_plot_label(
    method: str,
    metadata: Mapping[str, Any],
    *,
    detailed: bool,
) -> str:
    method_key = get_plot_method_key(method, metadata)
    if method_key == "teacher":
        label = "Teacher"
        if detailed:
            label = f"Teacher - {metadata.get('teacher_arch', 'teacher')}"
    elif method_key == "student":
        label = "Student"
        if detailed:
            label = f"Student - {metadata.get('student_arch', 'student')}"
    elif method_key == "student_kd":
        label = "Student + KD"
        if detailed:
            teacher_arch = metadata.get("teacher_arch", "teacher")
            student_arch = metadata.get("student_arch", "student")
            label = f"Student + KD - {teacher_arch} -> {student_arch}"
    elif method_key == "student_kd_logit_standardized":
        label = "Student + Logit Std. KD"
        if detailed:
            label = "Student + Logit Std. KD - published CIFAR-100 protocol"
    elif method_key == "student_dkd":
        label = "Student + DKD"
        if detailed:
            scope = (
                "repository CIFAR-10 adaptation"
                if metadata.get("dataset") == "cifar10"
                else "released CIFAR-100 recipe"
            )
            label = f"Student + DKD - {scope}"
    elif method_key == "student_ctkd":
        label = "Student + CTKD"
        if detailed:
            scope = (
                "repository CIFAR-10 adaptation"
                if metadata.get("dataset") == "cifar10"
                else "released CIFAR-100 recipe"
            )
            label = f"Student + CTKD - {scope}"
    elif method_key == "student_catkd":
        label = "Student + CAT-KD"
        if detailed:
            label = "Student + CAT-KD - class-attention transfer"
    elif method_key == "student_simkd":
        label = "Student + SimKD"
        if detailed:
            label = "Student + SimKD - reused teacher classifier"
    elif method_key == "student_reviewkd":
        label = "Student + ReviewKD"
        if detailed:
            label = "Student + ReviewKD - multi-scale feature review"
    elif method_key == "student_crd":
        label = "Student + CRD"
        if detailed:
            label = "Student + CRD - contrastive representation distillation"
    elif method_key == "direct_svd_inheritance":
        label = "Direct SVD inheritance"
        if detailed:
            label = f"Direct SVD inheritance - rank {metadata.get('rank', '?')}, one head"
    elif method_key == "inhernet_small":
        train_mode = str(metadata.get("compressed_train_mode", "distillation"))
        label = "InherNet-S + KD" if train_mode == "distillation" else "InherNet-S (supervised)"
        if detailed:
            label = f"{label} - rank {metadata.get('rank', '?')}, heads {metadata.get('head_num', '?')}"
    elif method_key == "inhernet_large":
        train_mode = str(metadata.get("compressed_train_mode", "supervised"))
        label = (
            "InherNet-L (supervised)"
            if train_mode == "supervised"
            else "InherNet-L + matched KD"
        )
        if detailed:
            label = f"{label} - rank {metadata.get('rank', '?')}, heads {metadata.get('head_num', '?')}"
    elif method_key == "inhernet_custom":
        label = "InherNet"
        if detailed:
            label = f"InherNet - rank {metadata.get('rank', '?')}, heads {metadata.get('head_num', '?')}"
    elif method_key == "inheract":
        rank_map = metadata.get("rank_map", {})
        avg_rank = None
        if isinstance(rank_map, Mapping) and rank_map:
            avg_rank = sum(int(value) for value in rank_map.values()) / len(rank_map)
        size = metadata.get("size")
        label = "InherAct-Lite" if size == "small" else "InherAct"
        if detailed:
            label = f"{label} - heads {metadata.get('head_num', '?')}"
            if avg_rank is None and "avg_rank" in metadata:
                avg_rank = float(metadata["avg_rank"])
            if avg_rank is not None:
                label += f", avg rank {avg_rank:.1f}"
    else:
        label = f"{method} ({metadata.get('config_tag', 'run')})"
    return label


def plot_single_history(
    plot_root: Path,
    metadata: Mapping[str, Any],
    history: Mapping[str, Any],
    plot_mode: str,
) -> Path | None:
    history = normalize_history(history)
    if not history_has_curves(history):
        return None
    dataset_name = str(metadata.get("dataset", "unknown_dataset"))
    pair_name = str(metadata.get("pair", "unknown_pair"))
    metadata = canonicalize_metadata(metadata)
    method = str(metadata.get("method", "unknown_method"))
    config_tag = sanitize_tag(str(metadata.get("config_tag", "default")))
    method_key = get_plot_method_key(method, metadata)
    style = get_plot_style(method_key)
    label = build_plot_label(method, metadata, detailed=True)
    plt = get_pyplot(plot_mode)

    fig, axes = plt.subplots(2, 2, figsize=(11.9, 8.2), dpi=300)
    eval_split_name = str(metadata.get("eval_split", "test"))
    metric_specs = get_plot_metric_specs(
        dataset_name,
        eval_split_name=eval_split_name,
        primary_metric_display=str(metadata.get("primary_metric_display", "Accuracy (%)")),
    )
    for axis, (metric_key, ylabel, title, clamp) in zip(axes.flatten(), metric_specs, strict=True):
        plot_single_metric_panel(axis, list(history.get(metric_key, [])), style, ylabel, title, clamp)

    summary_text = build_metric_summary(history, eval_split_name)
    if summary_text:
        axes[1, 1].text(
            0.98,
            0.04,
            summary_text,
            transform=axes[1, 1].transAxes,
            ha="right",
            va="bottom",
            fontsize=9.1,
            family="DejaVu Sans Mono",
            color="#1F2530",
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": "#F7F8FB",
                "edgecolor": "#D7DCE5",
                "linewidth": 0.9,
            },
        )

    fig.suptitle(label, x=0.06, y=0.985, ha="left", fontsize=13.3, fontweight="bold")
    fig.text(
        0.06,
        0.945,
        f"{dataset_name} | {pair_name}",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5A6473",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.11, wspace=0.22, hspace=0.28)

    output_path = plot_root / dataset_name / pair_name / method / f"{config_tag}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def parse_structured_log_line(line: str, prefix: str) -> dict[str, Any] | None:
    token = f"{prefix} "
    if not line.startswith(token):
        return None
    payload = line[len(token) :].strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def parse_run_log(log_path: Path, *, phase: str = "target") -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    metadata: dict[str, Any] | None = None
    phase_histories: dict[str, dict[str, list[float]]] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            metadata_payload = parse_structured_log_line(line, RUN_METADATA_PREFIX)
            if metadata_payload is not None:
                metadata = canonicalize_metadata(metadata_payload)
                continue
            metrics_payload = parse_structured_log_line(line, RUN_METRICS_PREFIX)
            if metrics_payload is None:
                continue
            phase_name = str(metrics_payload.get("phase", "target"))
            history = phase_histories.setdefault(phase_name, create_history_template())
            history["train_objective"].append(float(metrics_payload.get("train_objective", metrics_payload.get("train_loss", 0.0))))
            history["train_loss"].append(float(metrics_payload.get("train_loss", metrics_payload.get("train_objective", 0.0))))
            history["train_accuracy"].append(float(metrics_payload.get("train_accuracy", 0.0)))
            history["test_loss"].append(float(metrics_payload.get("test_loss", 0.0)))
            history["test_accuracy"].append(float(metrics_payload.get("test_accuracy", metrics_payload.get("eval_accuracy", 0.0))))

    if metadata is None:
        return None

    selected_history = phase_histories.get(phase, create_history_template())
    if not history_has_curves(selected_history):
        for candidate in phase_histories.values():
            if history_has_curves(candidate):
                selected_history = candidate
                break

    return {
        "log_path": log_path,
        "metadata": metadata,
        "history": normalize_history(selected_history),
    }


def collect_suite_comparison_records(suite_log_dir: Path) -> list[dict[str, Any]]:
    if not suite_log_dir.exists():
        return []

    method_order = {
        "teacher": 0,
        "student": 1,
        "student_kd": 2,
        "student_kd_logit_standardized": 3,
        "student_ctkd": 4,
        "student_dkd": 5,
        "student_catkd": 6,
        "student_simkd": 7,
        "student_reviewkd": 8,
        "student_crd": 9,
        "direct_svd_inheritance": 10,
        "inhernet_small": 11,
        "inhernet_large": 12,
        "inhernet_custom": 13,
        "inheract": 14,
    }
    records: list[dict[str, Any]] = []
    for log_path in sorted(suite_log_dir.glob("[0-9][0-9]_*.log")):
        parsed = parse_run_log(log_path, phase="target")
        if parsed is None or not history_has_curves(parsed["history"]):
            continue
        metadata = parsed["metadata"]
        method = str(metadata.get("method", log_path.stem))
        method_key = get_plot_method_key(method, metadata)
        records.append(
            {
                "log_path": log_path,
                "history": parsed["history"],
                "metadata": metadata,
                "method": method,
                "method_key": method_key,
                "label": build_plot_label(method, metadata, detailed=False),
            }
        )

    label_counts = Counter(record["label"] for record in records)
    for record in records:
        if label_counts[record["label"]] > 1:
            config_tag = str(record["metadata"].get("config_tag", record["log_path"].stem))
            record["label"] = f"{record['label']} [{sanitize_tag(config_tag)}]"

    records.sort(
        key=lambda record: (
            method_order.get(record["method_key"], 99),
            record["label"],
            str(record["log_path"]),
        )
    )
    return records


def plot_comparison_histories_from_records(
    plot_root: Path,
    dataset_name: str,
    pair_name: str,
    records: list[dict[str, Any]],
    plot_mode: str,
) -> Path | None:
    if not records:
        return None
    plt = get_pyplot(plot_mode)

    fig, axes = plt.subplots(2, 2, figsize=(13.7, 8.5), dpi=300)
    first_metadata = records[0].get("metadata", {})
    metric_specs = get_plot_metric_specs(
        dataset_name,
        eval_split_name=str(first_metadata.get("eval_split", "test")),
        primary_metric_display=str(first_metadata.get("primary_metric_display", "Accuracy (%)")),
    )
    for axis, (metric_key, ylabel, title, clamp) in zip(axes.flatten(), metric_specs, strict=True):
        plot_comparison_metric_panel(axis, records, metric_key, ylabel, title, clamp)

    handles = []
    labels = []
    for axis in axes.flatten():
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.79, 0.50),
            ncol=1,
            frameon=False,
            handlelength=2.8,
            borderaxespad=0.0,
        )
    fig.suptitle("Model Comparison", x=0.06, y=0.985, ha="left", fontsize=13.4, fontweight="bold")
    fig.text(
        0.06,
        0.945,
        f"{dataset_name} | {pair_name}",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5A6473",
    )
    fig.subplots_adjust(left=0.08, right=0.77, top=0.84, bottom=0.11, wspace=0.24, hspace=0.30)

    output_path = plot_root / dataset_name / pair_name / "comparison" / "overview.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def plot_suite_comparison_from_logs(
    plot_root: Path,
    suite_log_dir: Path,
    dataset_name: str,
    pair_name: str,
    plot_mode: str,
) -> Path | None:
    records = collect_suite_comparison_records(suite_log_dir)
    return plot_comparison_histories_from_records(plot_root, dataset_name, pair_name, records, plot_mode)
