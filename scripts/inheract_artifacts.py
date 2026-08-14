"""Normalize pre-rename experiment records at the artifact-reader boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping


# Historical compatibility is deliberately isolated here; new code emits only
# the canonical InherAct spelling.
_LEGACY_PREFIX = "hetero"
_CANONICAL_PREFIX = "inheract"
_LEGACY_FIELD_NAMES = {
    f"{_LEGACY_PREFIX}_{suffix}": f"{_CANONICAL_PREFIX}_{suffix}"
    for suffix in (
        "expert_noise_scale",
        "max_features_per_batch",
        "second_moment_shrinkage",
        "allocation_scale",
        "recipe_id",
        "report",
    )
}
_LEGACY_FIELD_NAMES[
    f"freeze_{_LEGACY_PREFIX}_router"
] = f"freeze_{_CANONICAL_PREFIX}_router"


def canonicalize_argv(arguments: list[Any]) -> list[Any]:
    """Translate the historical public spelling without changing raw artifacts."""
    normalized: list[Any] = []
    for argument in arguments:
        if not isinstance(argument, str):
            normalized.append(argument)
        elif argument == _LEGACY_PREFIX:
            normalized.append(_CANONICAL_PREFIX)
        elif argument == f"--method={_LEGACY_PREFIX}":
            normalized.append(f"--method={_CANONICAL_PREFIX}")
        elif argument.startswith(f"--{_LEGACY_PREFIX}-"):
            normalized.append(
                f"--{_CANONICAL_PREFIX}-" + argument.removeprefix(f"--{_LEGACY_PREFIX}-")
            )
        elif argument == f"--freeze-{_LEGACY_PREFIX}-router":
            normalized.append(f"--freeze-{_CANONICAL_PREFIX}-router")
        else:
            normalized.append(argument)
    return normalized


def canonicalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow canonical view of one structured run-metadata record."""
    normalized = dict(metadata)
    if normalized.get("method") == _LEGACY_PREFIX:
        normalized["method"] = _CANONICAL_PREFIX
    for legacy_name, canonical_name in _LEGACY_FIELD_NAMES.items():
        if canonical_name not in normalized and legacy_name in normalized:
            normalized[canonical_name] = normalized[legacy_name]
    argv = normalized.get("argv")
    if isinstance(argv, list):
        normalized["argv"] = canonicalize_argv(argv)
    return normalized


def legacy_log_path(path: Path) -> Path | None:
    """Locate the historical method-directory spelling for a new log path."""
    parts = list(path.parts)
    try:
        index = parts.index("inheract")
    except ValueError:
        return None
    parts[index] = _LEGACY_PREFIX
    return Path(*parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_log", type=Path)
    args = parser.parse_args()
    legacy = legacy_log_path(args.canonical_log)
    if legacy is not None and legacy.is_file():
        print(legacy)


if __name__ == "__main__":
    main()
