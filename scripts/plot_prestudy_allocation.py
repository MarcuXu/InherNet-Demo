#!/usr/bin/env python3
"""Plot allocation fidelity and task retention relative to fixed-rank Hetero."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from plotting_utils import get_pyplot


# Exact seed-42 zero-step values. Allocation controls stay within the fixed-rank
# parameter cap and use the same activation-weighted decomposition metric.
ALLOCATION_DATA = {
    "oxford_pets": {
        "label": "Oxford Pets",
        "color": "#0072B2",
        "marker": "o",
        "methods": {
            "Fixed rank": {"relative_sse": 0.28378891885026997, "teacher_kl": 0.4857463674700778, "task_metric": 77.46799431009957},
            "Relative": {"relative_sse": 0.40548027187926294, "teacher_kl": 0.9304731565973033, "task_metric": 61.51493598862021},
            "Nested": {"relative_sse": 0.30278324383882527, "teacher_kl": 0.5126885292322739, "task_metric": 75.84637268847794},
        },
    },
    "cifar100": {
        "label": "CIFAR-100",
        "color": "#D55E00",
        "marker": "D",
        "methods": {
            "Fixed rank": {"relative_sse": 0.8368906319195847, "teacher_kl": 3.7369920372009275, "task_metric": 7.42},
            "Relative": {"relative_sse": 0.8108404980478308, "teacher_kl": 3.6761052928924562, "task_metric": 6.8},
            "Nested": {"relative_sse": 0.8612547694229533, "teacher_kl": 3.763845914840698, "task_metric": 5.2},
        },
    },
}

METHODS = ("Fixed rank", "Relative", "Nested")


def normalized(dataset: dict[str, object], method: str, metric: str) -> float:
    methods = dataset["methods"]
    return 100.0 * methods[method][metric] / methods["Fixed rank"][metric]


def validate_data() -> None:
    for dataset in ALLOCATION_DATA.values():
        assert tuple(dataset["methods"]) == METHODS


def plot(output: Path) -> None:
    validate_data()
    plt = get_pyplot("single")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, left = plt.subplots(figsize=(6.5, 4.1))
    right = left.twinx()
    centers = list(range(len(METHODS)))
    dataset_offsets = {"oxford_pets": -0.19, "cifar100": 0.19}
    bar_offset = 0.055
    bar_width = 0.105

    for dataset_name, dataset in ALLOCATION_DATA.items():
        group_positions = [center + dataset_offsets[dataset_name] for center in centers]
        color = dataset["color"]
        left.bar(
            [position - bar_offset for position in group_positions],
            [normalized(dataset, method, "relative_sse") for method in METHODS],
            width=bar_width,
            color=color,
            alpha=0.27,
            edgecolor=color,
            linewidth=1.0,
            zorder=2,
        )
        left.bar(
            [position + bar_offset for position in group_positions],
            [normalized(dataset, method, "teacher_kl") for method in METHODS],
            width=bar_width,
            color="white",
            edgecolor=color,
            hatch="///",
            linewidth=1.05,
            zorder=2,
        )
        right.scatter(
            group_positions,
            [normalized(dataset, method, "task_metric") for method in METHODS],
            s=55,
            marker=dataset["marker"],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

    legend_handles = [
        Patch(facecolor="#0072B2", edgecolor="#0072B2", alpha=0.27, label="Oxford Pets"),
        Patch(facecolor="#D55E00", edgecolor="#D55E00", alpha=0.27, label="CIFAR-100"),
        Patch(facecolor="#9CA3AF", edgecolor="#6B7280", alpha=0.35, label="Output SSE"),
        Patch(facecolor="white", edgecolor="#6B7280", hatch="///", label="Teacher KL"),
        Line2D([], [], color="#374151", marker="D", linestyle="None", markersize=6,
               label="Task retention"),
    ]
    left.set_xticks(centers, METHODS)
    left.get_xticklabels()[0].set_fontweight("bold")
    left.set_ylabel("Inheritance discrepancy (% of fixed rank) ↓")
    right.set_ylabel("Task retention (% of fixed rank) ↑")
    left.set_ylim(0, 205)
    right.set_ylim(0, 108)
    left.set_xlim(-0.55, len(METHODS) - 0.45)
    left.grid(False)
    right.grid(False)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color("#424A57")
    left.set_title("Rank allocation: surrogate fidelity and task retention", pad=34)
    left.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=5,
        columnspacing=0.9,
        handletextpad=0.45,
        fontsize=8.3,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "figures" / "prestudy_allocation_tradeoff.png",
    )
    args = parser.parse_args()
    plot(args.output)
    print(f"Wrote 300-DPI figure: {args.output}")


if __name__ == "__main__":
    main()
