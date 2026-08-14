#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
original_args=("$@")
require_dataset_pair "$@"

dataset="$1"
pair="$2"
shift 2
extra=("$@")
reject_identity_overrides "${extra[@]}"
reject_formal_training_overrides "${extra[@]}"
export DEVICE="${DEVICE:-cuda}"
if [[ "${OVERWRITE_TEACHER:-0}" == "1" ]]; then
    echo "Formal runs never overwrite teachers. Start a new formal run namespace instead." >&2
    exit 2
fi
if [[ "${RESUME:-0}" == "1" && -z "${FORMAL_RUN_ID:-}" ]]; then
    echo "RESUME=1 requires FORMAL_RUN_ID=<existing formal run namespace>." >&2
    exit 2
fi
export RESUME="${RESUME:-0}"
export FORMAL_RUN_ID="${FORMAL_RUN_ID:-$(new_formal_run_id)}"
validate_formal_run_id "$FORMAL_RUN_ID"
load_inheract_recipe "$dataset"
inheract_recipe_id="${INHERACT_RECIPE_ARGS[1]}"
mapfile -t INHERACT_OBJECTIVE_ARGS < <(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/inheract_recipes.py" selected-objective "$dataset"
)
mapfile -t INHERACT_OPTIMIZER_ARGS < <(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/inheract_recipes.py" selected-optimizer "$dataset"
)
inheract_train_mode="${INHERACT_OBJECTIVE_ARGS[1]}"
if [[ "$inheract_train_mode" == "distillation" ]]; then
    mapfile -t INHERACT_SUPERVISED_ARGS < <(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/inheract_recipes.py" selected-supervised "$dataset"
    )
fi
IFS=',' read -r -a seeds <<<"${SEEDS:-7,17,27,37}"
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ || -n "${seen_seeds[$seed]:-}" ]]; then
        echo "SEEDS must contain distinct non-negative integers: ${SEEDS:-7,17,27,37}" >&2
        exit 2
    fi
    seen_seeds[$seed]=1
done
default_background
if maybe_launch_background formal "${original_args[@]}"; then
    echo "Formal run namespace: $FORMAL_RUN_ID"
    exit 0
fi
validation_args=()
if [[ "$dataset" == "cifar10" || "$dataset" == "cifar100" || "$dataset" == glue_* ]]; then
    validation_args+=(--search-validation)
fi
mapfile -t registered_baseline_methods < <(
    PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" - "$dataset" "$pair" <<'PY'
import sys
from experiment_registry import (
    CAT_KD_REGISTRY,
    CRD_REGISTRY,
    CURRICULUM_TEMPERATURE_DISTILLATION_REGISTRY,
    DECOUPLED_DISTILLATION_REGISTRY,
    LOGIT_STANDARDIZED_KD_REGISTRY,
    REVIEW_KD_REGISTRY,
    SIM_KD_REGISTRY,
)

key = (sys.argv[1], sys.argv[2])
for registry, method in (
    (LOGIT_STANDARDIZED_KD_REGISTRY, "student_kd_logit_standardized"),
    (CURRICULUM_TEMPERATURE_DISTILLATION_REGISTRY, "student_ctkd"),
    (DECOUPLED_DISTILLATION_REGISTRY, "student_dkd"),
    (CAT_KD_REGISTRY, "student_catkd"),
    (SIM_KD_REGISTRY, "student_simkd"),
    (REVIEW_KD_REGISTRY, "student_reviewkd"),
    (CRD_REGISTRY, "student_crd"),
):
    if key in registry:
        print(method)
PY
)

run_formal_case() {
    local label="$1" log_path="$2"
    shift 2
    if [[ -f "$log_path" ]]; then
        if [[ "$RESUME" == "1" ]] && rg -q '^RUN_SUMMARY ' "$log_path"; then
            "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_completed_log.py" "$log_path" -- "$@"
            echo "Skipping completed formal run: $label"
            return
        fi
        echo "Formal log already exists: $log_path" >&2
        exit 1
    fi
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        mkdir -p "$(dirname "$log_path")"
    fi
    INHERNET_RUN_LOG="$log_path" run_case "$label" "$@"
}

run_registered_baselines() {
    local method label
    for method in "${registered_baseline_methods[@]}"; do
        case "$method" in
            student_kd_logit_standardized) label="student logit-standardized KD" ;;
            student_ctkd) label="student CTKD" ;;
            student_dkd) label="student DKD" ;;
            student_catkd) label="student CAT-KD" ;;
            student_simkd) label="student SimKD" ;;
            student_reviewkd) label="student ReviewKD" ;;
            student_crd) label="student CRD" ;;
        esac
        run_formal_case "$label seed=$seed" "$log_root/$method.log" \
            "${common[@]}" --method "$method" --teacher-checkpoint "$checkpoint"
    done
}

for seed in "${seeds[@]}"; do
    checkpoint="$(formal_checkpoint_for "$FORMAL_RUN_ID" "$dataset" "$pair" "$seed")"
    log_root="$PROJECT_DIR/logs/formal/$FORMAL_RUN_ID/$dataset/$pair/seed_${seed}"
    common=(--dataset "$dataset" --pair "$pair" --seed "$seed" --device "$DEVICE" \
        --plot-mode none "${validation_args[@]}" "${extra[@]}")
    search_validation_enabled=0
    ((${#validation_args[@]} > 0)) && search_validation_enabled=1
    teacher_checkpoint_args=(--teacher-checkpoint "$checkpoint")
    if [[ -f "$checkpoint" ]]; then
        validate_compatible_teacher_checkpoint \
            "$checkpoint" "$dataset" "$pair" "$seed" "$search_validation_enabled"
        if [[ ! -f "$log_root/teacher.log" ]]; then
            echo "Formal teacher checkpoint exists without its teacher log: $checkpoint" >&2
            echo "Use a new FORMAL_RUN_ID rather than mixing artifacts from different runs." >&2
            exit 1
        fi
        run_formal_case "teacher seed=$seed" "$log_root/teacher.log" \
            "${common[@]}" --method teacher "${teacher_checkpoint_args[@]}"
    else
        existing_formal_log=$(rg --files "$log_root" 2>/dev/null | rg '\.log$' | head -n 1 || true)
        if [[ -n "$existing_formal_log" ]]; then
            echo "Formal teacher is missing while formal logs exist: $existing_formal_log" >&2
            echo "Use a new FORMAL_RUN_ID rather than mixing artifacts from different runs." >&2
            exit 1
        fi
        run_formal_case "teacher seed=$seed" "$log_root/teacher.log" \
            "${common[@]}" --method teacher "${teacher_checkpoint_args[@]}"
    fi
    run_formal_case "student seed=$seed" "$log_root/student.log" \
        "${common[@]}" --method student
    run_formal_case "student KD seed=$seed" "$log_root/student_kd.log" \
        "${common[@]}" --method student_kd --teacher-checkpoint "$checkpoint"
    run_registered_baselines
    run_formal_case "InherNet small with KD seed=$seed" "$log_root/inhernet_small.log" \
        "${common[@]}" --method inhernet --size small \
        --teacher-checkpoint "$checkpoint" --compressed-train-mode distillation
    run_formal_case "InherNet large seed=$seed" "$log_root/inhernet_large.log" \
        "${common[@]}" --method inhernet --size large \
        --teacher-checkpoint "$checkpoint" --compressed-train-mode supervised
    if [[ "$inheract_train_mode" == "distillation" ]]; then
        run_formal_case "InherNet large with InherAct-matched objective and optimizer seed=$seed" \
            "$log_root/inhernet_large_matched_inheract_objective.log" \
            "${common[@]}" --method inhernet --size large \
            --teacher-checkpoint "$checkpoint" \
            --search-candidate formal_inhernet_large_matched_inheract_objective_optimizer \
            "${INHERACT_OBJECTIVE_ARGS[@]}" "${INHERACT_OPTIMIZER_ARGS[@]}"
    fi
    run_formal_case "InherAct recipe=$inheract_recipe_id seed=$seed" \
        "$log_root/inheract_${inheract_recipe_id}.log" \
        "${common[@]}" --method inheract --size large --teacher-checkpoint "$checkpoint" \
        "${INHERACT_RECIPE_ARGS[@]}"
    if [[ "$inheract_train_mode" == "distillation" ]]; then
        run_formal_case "InherAct supervised control recipe=$inheract_recipe_id seed=$seed" \
            "$log_root/inheract_${inheract_recipe_id}_supervised_control.log" \
            "${common[@]}" --method inheract --size large --teacher-checkpoint "$checkpoint" \
            --search-candidate formal_inheract_supervised_control \
            "${INHERACT_SUPERVISED_ARGS[@]}"
    fi
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/summarize_search.py" "$log_root" \
            --output "$log_root/summary.csv"
    fi
done
