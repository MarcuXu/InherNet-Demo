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
load_inheract_recipe "$dataset"
common=(--dataset "$dataset" --pair "$pair" --device "${DEVICE:-cuda}" --smoke-test --plot-mode none "${extra[@]}")

for method in teacher student student_kd; do
    run_case "$method smoke" "${common[@]}" --method "$method"
done

mapfile -t registered_baseline_methods < <(
    PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" - "$dataset" "$pair" <<'PY'
import sys
from experiment_registry import (
    CAT_KD_REGISTRY, CRD_REGISTRY, CURRICULUM_TEMPERATURE_DISTILLATION_REGISTRY,
    DECOUPLED_DISTILLATION_REGISTRY, LOGIT_STANDARDIZED_KD_REGISTRY,
    REVIEW_KD_REGISTRY, SIM_KD_REGISTRY,
)

key = (sys.argv[1], sys.argv[2])
for registry, method in (
    (DECOUPLED_DISTILLATION_REGISTRY, "student_dkd"),
    (LOGIT_STANDARDIZED_KD_REGISTRY, "student_kd_logit_standardized"),
    (CURRICULUM_TEMPERATURE_DISTILLATION_REGISTRY, "student_ctkd"),
    (CAT_KD_REGISTRY, "student_catkd"),
    (SIM_KD_REGISTRY, "student_simkd"),
    (REVIEW_KD_REGISTRY, "student_reviewkd"),
    (CRD_REGISTRY, "student_crd"),
):
    if key in registry:
        print(method)
PY
)
for method in "${registered_baseline_methods[@]}"; do
    run_case "$method smoke" "${common[@]}" --method "$method"
done
for rank in small large; do
    run_case "InherNet $rank smoke" "${common[@]}" --method inhernet --size "$rank"
    run_case "$(inheract_display_name "$rank") smoke" "${common[@]}" --method inheract --size "$rank" \
        "${INHERACT_RECIPE_ARGS[@]}"
done
run_case "Direct SVD inheritance reference (one head) smoke" "${common[@]}" \
    --method inhernet --size large --head-num 1
