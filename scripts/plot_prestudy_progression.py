#!/usr/bin/env python3
"""Plot the fixed-capacity inheritance progression for both vision datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from plotting_utils import get_pyplot


# Exact seed-42 zero-step measurements from the completed pre-study. All four
# constructions use the registered large rank and the same parameter count.
PROGRESSION_DATA = {
    "oxford_pets": {
        "label": "Oxford Pets",
        "metric": "balanced accuracy",
        "color": "#0072B2",
        "marker": "o",
        "methods": {
            "InherNet": {"relative_sse": 0.8477581888307537, "task_metric": 18.157894736842103},
            "Weight SVD": {"relative_sse": 0.8477418906401061, "task_metric": 18.293029871977236},
            "Activation-aware": {"relative_sse": 0.28378891885026997, "task_metric": 77.46799431009957},
            "InherAct": {"relative_sse": 0.28384779482539896, "task_metric": 77.46799431009957},
        },
    },
    "cifar100": {
        "label": "CIFAR-100",
        "metric": "accuracy",
        "color": "#D55E00",
        "marker": "D",
        "methods": {
            "InherNet": {"relative_sse": 1.0155290510783406, "task_metric": 2.54},
            "Weight SVD": {"relative_sse": 1.0155332510683595, "task_metric": 2.54},
            "Activation-aware": {"relative_sse": 0.8368906319195847, "task_metric": 7.42},
            "InherAct": {"relative_sse": 0.8368925308302901, "task_metric": 7.42},
        },
    },
}

METHODS = ("InherNet", "Weight SVD", "Activation-aware", "InherAct")


def normalized_sse(dataset: dict[str, object], method: str) -> float:
    methods = dataset["methods"]
    return 100.0 * methods[method]["relative_sse"] / methods["InherNet"]["relative_sse"]


def validate_data() -> None:
    for dataset in PROGRESSION_DATA.values():
        assert tuple(dataset["methods"]) == METHODS
        assert normalized_sse(dataset, "Activation-aware") < normalized_sse(dataset, "Weight SVD")
        assert dataset["methods"]["InherAct"]["task_metric"] == dataset["methods"]["Activation-aware"]["task_metric"]


def plot(output: Path) -> None:
    validate_data()
    plt = get_pyplot("single")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, left = plt.subplots(figsize=(6.5, 4.1))
    right = left.twinx()
    centers = list(range(len(METHODS)))
    offsets = {"oxford_pets": -0.16, "cifar100": 0.16}

    legend_handles = []
    for dataset_name, dataset in PROGRESSION_DATA.items():
        positions = [center + offsets[dataset_name] for center in centers]
        sse_values = [normalized_sse(dataset, method) for method in METHODS]
        task_values = [dataset["methods"][method]["task_metric"] for method in METHODS]
        color = dataset["color"]
        marker = dataset["marker"]
        left.bar(
            positions,
            sse_values,
            width=0.27,
            color=color,
            alpha=0.24,
            edgecolor=color,
            linewidth=1.2,
            zorder=2,
        )
        right.plot(
            positions,
            task_values,
            color=color,
            marker=marker,
            markersize=6.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            zorder=4,
        )
        legend_handles.extend(
            [
                Patch(facecolor=color, edgecolor=color, alpha=0.24,
                      label=f"{dataset['label']} · output SSE"),
                Line2D([], [], color=color, marker=marker, linewidth=2.0,
                       markeredgecolor="white", label=f"{dataset['label']} · {dataset['metric']}"),
            ]
        )

    left.set_xticks(centers, METHODS)
    left.get_xticklabels()[-1].set_fontweight("bold")
    left.set_ylabel("Teacher-output SSE (% of InherNet) ↓")
    right.set_ylabel("Zero-step task metric (%) ↑")
    left.set_ylim(0, 112)
    right.set_ylim(0, 85)
    left.set_xlim(-0.55, len(METHODS) - 0.45)
    left.grid(False)
    right.grid(False)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color("#424A57")
    left.set_title("Inheritance progression at fixed capacity", pad=34)
    left.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        columnspacing=1.5,
        handletextpad=0.55,
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
        default=PROJECT_DIR / "figures" / "prestudy_inheritance_progression.png",
    )
    args = parser.parse_args()
    plot(args.output)
    print(f"Wrote 300-DPI figure: {args.output}")


if __name__ == "__main__":
    main()
