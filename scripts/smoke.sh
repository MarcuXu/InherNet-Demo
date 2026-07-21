#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_dataset_pair "$@"

dataset="$1"
pair="$2"
shift 2
extra=("$@")
reject_identity_overrides "${extra[@]}"
reject_formal_training_overrides "${extra[@]}"
load_hetero_recipe "$dataset"
common=(--dataset "$dataset" --pair "$pair" --device "${DEVICE:-cuda}" --smoke-test --plot-mode none "${extra[@]}")

for method in teacher student student_kd; do
    run_case "$method smoke" "${common[@]}" --method "$method"
done
for rank in small large; do
    run_case "InherNet $rank smoke" "${common[@]}" --method inhernet --size "$rank"
    run_case "$(hetero_display_name "$rank") smoke" "${common[@]}" --method hetero --size "$rank" \
        "${HETERO_RECIPE_ARGS[@]}"
done
