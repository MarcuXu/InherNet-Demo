from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from checkpointing import load_teacher_checkpoint
from experiment_registry import (
    DATASET_REGISTRY,
    build_pair_model,
    get_pair_spec,
    get_role_name,
)

def audit_teacher_checkpoints(
    seed: int,
    group: str = "all",
    *,
    dataset_filter: str | None = None,
    pair_filter: str | None = None,
    checkpoint_root: Path | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    checkpoint_root = (checkpoint_root or PROJECT_DIR / "checkpoints" / "search").resolve()
    for dataset_name, dataset_spec in DATASET_REGISTRY.items():
        if group == "glue" and not dataset_name.startswith("glue_"):
            continue
        if group == "vision" and dataset_name.startswith("glue_"):
            continue
        if dataset_filter is not None and dataset_name != dataset_filter:
            continue
        for pair_name in dataset_spec.pair_registry:
            if pair_filter is not None and pair_name != pair_filter:
                continue
            pair_spec = get_pair_spec(dataset_name, pair_name)
            checkpoint_path = (
                checkpoint_root
                / dataset_name
                / pair_name
                / f"teacher_seed_{seed}.pt"
            )
            model = build_pair_model(
                dataset_name,
                pair_name,
                "teacher",
                dataset_spec.num_classes,
                initialize_pretrained=False,
            )
            info = load_teacher_checkpoint(
                checkpoint_path,
                model,
                dataset=dataset_name,
                pair=pair_name,
                architecture=get_role_name(pair_spec, "teacher"),
                num_classes=dataset_spec.num_classes,
                seed=seed,
                model_profile=str(pair_spec.get("model_profile", "unspecified")),
                data_profile=dataset_spec.data_profile,
                expected_settings=dataset_spec.train_settings,
            )
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError(f"Loaded teacher is not frozen in evaluation mode: {checkpoint_path}")
            records.append(
                {
                    "dataset": dataset_name,
                    "pair": pair_name,
                    "teacher": get_role_name(pair_spec, "teacher"),
                    "path": str(checkpoint_path.relative_to(PROJECT_DIR)),
                    "bytes": checkpoint_path.stat().st_size,
                    "selection_policy": info["selection_policy"],
                    "selected_epoch": info["selected_epoch"],
                    "selected_metrics": info["metrics"],
                    "split_provenance": "embedded" if info["data_split"] is not None else "legacy_unrecorded",
                }
            )
            del model
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate all registered teacher checkpoints.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group", choices=("all", "glue", "vision"), default="all")
    parser.add_argument("--dataset", choices=tuple(sorted(DATASET_REGISTRY)), default=None)
    parser.add_argument("--pair", default=None)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "search",
    )
    parser.add_argument("--json", action="store_true", help="Print a stable JSON manifest.")
    args = parser.parse_args()
    if (args.dataset is None) != (args.pair is None):
        parser.error("--dataset and --pair must be supplied together.")
    records = audit_teacher_checkpoints(
        args.seed,
        args.group,
        dataset_filter=args.dataset,
        pair_filter=args.pair,
        checkpoint_root=args.checkpoint_root,
    )
    if not records:
        parser.error("No registered teacher matched the requested filters.")
    if args.json:
        print(json.dumps({"schema_version": 1, "teachers": records}, indent=2, sort_keys=True))
        return
    total_bytes = sum(int(record["bytes"]) for record in records)
    for record in records:
        print(
            f"OK {record['dataset']}/{record['pair']} "
            f"epoch={record['selected_epoch']}"
        )
    print(f"Validated {len(records)} checkpoints ({total_bytes / 1024**2:.1f} MiB).")


if __name__ == "__main__":
    main()
