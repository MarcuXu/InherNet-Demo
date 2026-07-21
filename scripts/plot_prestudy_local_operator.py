#!/usr/bin/env python3
"""Plot held-out local operator error before and after activation weighting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from plotting_utils import get_pyplot


# Filled from the fixed-teacher held-out local-operator diagnostic. Each named
# layer record stores weight_only, activation_aware, and moment_mode fields.
LOCAL_OPERATOR_DATA = {
    "oxford_pets": {
        "label": "Oxford Pets",
        "color": "#0072B2",
        "layers": {
            "layer2.0.conv1": {"weight_only": 0.14225154702787773, "activation_aware": 0.03656849159608076, "moment_mode": "channel_block"},
            "layer2.0.conv2": {"weight_only": 0.08284807376231207, "activation_aware": 0.03605150893721171, "moment_mode": "channel_block"},
            "layer2.1.conv1": {"weight_only": 0.06803331010943281, "activation_aware": 0.03371320613954367, "moment_mode": "channel_block"},
            "layer2.1.conv2": {"weight_only": 0.07065083448517485, "activation_aware": 0.044179027917725947, "moment_mode": "channel_block"},
            "layer2.2.conv1": {"weight_only": 0.16674814757216563, "activation_aware": 0.06927375009409109, "moment_mode": "channel_block"},
            "layer2.2.conv2": {"weight_only": 0.12154420081252724, "activation_aware": 0.08718454303046957, "moment_mode": "channel_block"},
            "layer2.3.conv1": {"weight_only": 0.08732138695170788, "activation_aware": 0.04976704086412327, "moment_mode": "channel_block"},
            "layer2.3.conv2": {"weight_only": 0.11493422710416801, "activation_aware": 0.08716088307064299, "moment_mode": "channel_block"},
            "layer3.0.conv1": {"weight_only": 0.20530229862398028, "activation_aware": 0.08787586625478845, "moment_mode": "channel_block"},
            "layer3.0.conv2": {"weight_only": 0.29686345802434483, "activation_aware": 0.16286410902810894, "moment_mode": "channel_block"},
            "layer3.0.downsample.0": {"weight_only": 0.0860788534642977, "activation_aware": 0.02622655056517053, "moment_mode": "exact_patch"},
            "layer3.1.conv1": {"weight_only": 0.1609543752349524, "activation_aware": 0.09277453585197867, "moment_mode": "channel_block"},
            "layer3.1.conv2": {"weight_only": 0.21726400796676199, "activation_aware": 0.1560001836707126, "moment_mode": "channel_block"},
            "layer3.2.conv1": {"weight_only": 0.14155534098870118, "activation_aware": 0.0868412719350394, "moment_mode": "channel_block"},
            "layer3.2.conv2": {"weight_only": 0.23767000727717902, "activation_aware": 0.1723838187538458, "moment_mode": "channel_block"},
            "layer3.3.conv1": {"weight_only": 0.1493926688396493, "activation_aware": 0.08250163201102008, "moment_mode": "channel_block"},
            "layer3.3.conv2": {"weight_only": 0.20302882581597084, "activation_aware": 0.13737781408412983, "moment_mode": "channel_block"},
            "layer3.4.conv1": {"weight_only": 0.13181535944526432, "activation_aware": 0.08152619186746034, "moment_mode": "channel_block"},
            "layer3.4.conv2": {"weight_only": 0.20666167485711173, "activation_aware": 0.1429943263653006, "moment_mode": "channel_block"},
            "layer3.5.conv1": {"weight_only": 0.16481428649430183, "activation_aware": 0.10330663392620207, "moment_mode": "channel_block"},
            "layer3.5.conv2": {"weight_only": 0.19276511856875592, "activation_aware": 0.13757561744920424, "moment_mode": "channel_block"},
            "layer4.0.conv1": {"weight_only": 0.24514408613804817, "activation_aware": 0.1592238624375803, "moment_mode": "channel_block"},
            "layer4.0.conv2": {"weight_only": 0.5027359752765213, "activation_aware": 0.2740897943864321, "moment_mode": "channel_block"},
            "layer4.0.downsample.0": {"weight_only": 0.268828918461019, "activation_aware": 0.1389819335287115, "moment_mode": "exact_patch"},
            "layer4.1.conv1": {"weight_only": 0.1254038749597525, "activation_aware": 0.06991474998699335, "moment_mode": "channel_block"},
            "layer4.1.conv2": {"weight_only": 0.4330264932976878, "activation_aware": 0.24093511288147604, "moment_mode": "channel_block"},
            "layer4.2.conv1": {"weight_only": 0.11090197927155403, "activation_aware": 0.050833556455867616, "moment_mode": "channel_block"},
            "layer4.2.conv2": {"weight_only": 0.4964113097710287, "activation_aware": 0.22020419856605994, "moment_mode": "channel_block"},
        },
    },
    "cifar100": {
        "label": "CIFAR-100",
        "color": "#D55E00",
        "layers": {
            "layer2.0.conv1": {"weight_only": 0.09560223604964396, "activation_aware": 0.03132493753086241, "moment_mode": "exact_patch"},
            "layer2.0.conv2": {"weight_only": 0.054039761963238106, "activation_aware": 0.03177496349070881, "moment_mode": "channel_block"},
            "layer2.1.conv1": {"weight_only": 0.12094091779984695, "activation_aware": 0.05931101926628184, "moment_mode": "channel_block"},
            "layer2.1.conv2": {"weight_only": 0.0393315737695347, "activation_aware": 0.02950459786280547, "moment_mode": "channel_block"},
            "layer2.2.conv1": {"weight_only": 0.11519793609660448, "activation_aware": 0.07037167870099909, "moment_mode": "channel_block"},
            "layer2.2.conv2": {"weight_only": 0.08894839767229848, "activation_aware": 0.07344753713742491, "moment_mode": "channel_block"},
            "layer2.3.conv1": {"weight_only": 0.12201697391399126, "activation_aware": 0.0846307086706312, "moment_mode": "channel_block"},
            "layer2.3.conv2": {"weight_only": 0.0680221687158589, "activation_aware": 0.056595221095744934, "moment_mode": "channel_block"},
            "layer2.4.conv1": {"weight_only": 0.23249601252643912, "activation_aware": 0.10852172061849613, "moment_mode": "channel_block"},
            "layer2.4.conv2": {"weight_only": 0.14723252790483807, "activation_aware": 0.11787202824666658, "moment_mode": "channel_block"},
            "layer2.5.conv1": {"weight_only": 0.2601758286903604, "activation_aware": 0.09062906307342904, "moment_mode": "channel_block"},
            "layer2.5.conv2": {"weight_only": 0.13492996502831656, "activation_aware": 0.11560320687021657, "moment_mode": "channel_block"},
            "layer2.6.conv1": {"weight_only": 0.1831246365230058, "activation_aware": 0.07963216204950975, "moment_mode": "channel_block"},
            "layer2.6.conv2": {"weight_only": 0.09923327135309834, "activation_aware": 0.08188766904223102, "moment_mode": "channel_block"},
            "layer2.7.conv1": {"weight_only": 0.1032105911266406, "activation_aware": 0.05530201367105064, "moment_mode": "channel_block"},
            "layer2.7.conv2": {"weight_only": 0.08086579287557034, "activation_aware": 0.0671041641525128, "moment_mode": "channel_block"},
            "layer2.8.conv1": {"weight_only": 0.04352228082928123, "activation_aware": 0.03499257262837927, "moment_mode": "channel_block"},
            "layer2.8.conv2": {"weight_only": 0.10995520007177463, "activation_aware": 0.09283840480890006, "moment_mode": "channel_block"},
            "layer3.0.conv1": {"weight_only": 0.26662776720224524, "activation_aware": 0.13167390758310646, "moment_mode": "channel_block"},
            "layer3.0.conv2": {"weight_only": 0.42821987647449006, "activation_aware": 0.3127541260291662, "moment_mode": "channel_block"},
            "layer3.0.shortcut.0": {"weight_only": 0.12590295942539484, "activation_aware": 0.03751197074279963, "moment_mode": "exact_patch"},
            "layer3.1.conv1": {"weight_only": 0.4297951327814443, "activation_aware": 0.25357339419671066, "moment_mode": "channel_block"},
            "layer3.1.conv2": {"weight_only": 0.33179463849391694, "activation_aware": 0.2787074608845723, "moment_mode": "channel_block"},
            "layer3.2.conv1": {"weight_only": 0.3707936324439094, "activation_aware": 0.2503254493875023, "moment_mode": "channel_block"},
            "layer3.2.conv2": {"weight_only": 0.34238280832699913, "activation_aware": 0.2939608611225675, "moment_mode": "channel_block"},
            "layer3.3.conv1": {"weight_only": 0.41920925144956384, "activation_aware": 0.2806767567233396, "moment_mode": "channel_block"},
            "layer3.3.conv2": {"weight_only": 0.437345787795433, "activation_aware": 0.3776739032704541, "moment_mode": "channel_block"},
            "layer3.4.conv1": {"weight_only": 0.373849209358177, "activation_aware": 0.22771359342518138, "moment_mode": "channel_block"},
            "layer3.4.conv2": {"weight_only": 0.37658302754018014, "activation_aware": 0.34069296203268584, "moment_mode": "channel_block"},
            "layer3.5.conv1": {"weight_only": 0.3307981607922517, "activation_aware": 0.21614989810813293, "moment_mode": "channel_block"},
            "layer3.5.conv2": {"weight_only": 0.39945523333384136, "activation_aware": 0.3595213948141143, "moment_mode": "channel_block"},
            "layer3.6.conv1": {"weight_only": 0.3576387723515516, "activation_aware": 0.25257741345941037, "moment_mode": "channel_block"},
            "layer3.6.conv2": {"weight_only": 0.470303807769831, "activation_aware": 0.4192409553581576, "moment_mode": "channel_block"},
            "layer3.7.conv1": {"weight_only": 0.38711710022092344, "activation_aware": 0.23198449528880372, "moment_mode": "channel_block"},
            "layer3.7.conv2": {"weight_only": 0.25782628161328347, "activation_aware": 0.21794249592591572, "moment_mode": "channel_block"},
            "layer3.8.conv1": {"weight_only": 0.41650634988040663, "activation_aware": 0.33909196165753835, "moment_mode": "channel_block"},
            "layer3.8.conv2": {"weight_only": 0.5349149420687105, "activation_aware": 0.4386628694559568, "moment_mode": "channel_block"},
        },
    },
}


def validate_data() -> None:
    assert sum(len(dataset["layers"]) for dataset in LOCAL_OPERATOR_DATA.values()) == 65
    for dataset in LOCAL_OPERATOR_DATA.values():
        for layer in dataset["layers"].values():
            assert layer["activation_aware"] < layer["weight_only"]
            assert layer["moment_mode"] in {"channel_block", "exact_patch"}


def plot(output: Path) -> None:
    validate_data()
    plt = get_pyplot("single")
    fig, ax = plt.subplots(figsize=(5.1, 4.15))

    all_values = []
    for dataset in LOCAL_OPERATOR_DATA.values():
        x_values = [layer["weight_only"] for layer in dataset["layers"].values()]
        y_values = [layer["activation_aware"] for layer in dataset["layers"].values()]
        all_values.extend(x_values)
        all_values.extend(y_values)
        ax.scatter(
            x_values,
            y_values,
            s=28,
            color=dataset["color"],
            alpha=0.68,
            edgecolor="white",
            linewidth=0.45,
            label=dataset["label"],
        )

    lower = min(all_values) * 0.8
    upper = max(all_values) * 1.25
    ax.plot([lower, upper], [lower, upper], color="#6B7280", linestyle="--",
            linewidth=1.2, label="Equal error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Weight-SVD held-out local relative SSE")
    ax.set_ylabel("Activation-aware held-out local relative SSE")
    ax.set_title("Held-out local operator fidelity", pad=9)
    ax.grid(False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "figures" / "prestudy_local_operator.png",
    )
    args = parser.parse_args()
    plot(args.output)
    print(f"Wrote 300-DPI figure: {args.output}")


if __name__ == "__main__":
    main()
