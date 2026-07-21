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
export RESUME="${RESUME:-1}"
load_hetero_recipe "$dataset"
hetero_recipe_id="${HETERO_RECIPE_ARGS[1]}"
mapfile -t HETERO_OBJECTIVE_ARGS < <(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/hetero_recipes.py" selected-objective "$dataset"
)
hetero_train_mode="${HETERO_OBJECTIVE_ARGS[1]}"
if [[ "$hetero_train_mode" == "distillation" ]]; then
    mapfile -t HETERO_SUPERVISED_ARGS < <(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/hetero_recipes.py" selected-supervised "$dataset"
    )
fi
default_background
if maybe_launch_background formal "${original_args[@]}"; then exit 0; fi
IFS=',' read -r -a seeds <<<"${SEEDS:-7,17,27,37}"
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ || -n "${seen_seeds[$seed]:-}" ]]; then
        echo "SEEDS must contain distinct non-negative integers: ${SEEDS:-7,17,27,37}" >&2
        exit 2
    fi
    seen_seeds[$seed]=1
done
validation_args=()
if [[ "$dataset" == "cifar10" || "$dataset" == "cifar100" ]]; then
    validation_args+=(--search-validation)
fi

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
    mkdir -p "$(dirname "$log_path")"
    INHERNET_RUN_LOG="$log_path" run_case "$label" "$@"
}

for seed in "${seeds[@]}"; do
    checkpoint="$(checkpoint_for "$dataset" "$pair" "$seed")"
    log_root="$PROJECT_DIR/logs/formal/$dataset/$pair/seed_${seed}"
    common=(--dataset "$dataset" --pair "$pair" --seed "$seed" --device "$DEVICE" \
        --plot-mode none "${validation_args[@]}" "${extra[@]}")
    teacher_checkpoint_args=(--teacher-checkpoint "$checkpoint")
    if [[ "${OVERWRITE_TEACHER:-0}" == "1" ]]; then
        existing_formal_log=$(rg --files "$log_root" 2>/dev/null | rg '\.log$' | head -n 1 || true)
        if [[ -n "$existing_formal_log" ]]; then
            echo "Cannot replace a formal teacher after dependent logs exist: $existing_formal_log" >&2
            exit 1
        fi
        teacher_checkpoint_args+=(--overwrite-teacher-checkpoint)
    fi
    if [[ -f "$checkpoint" && "${OVERWRITE_TEACHER:-0}" != "1" ]]; then
        echo "Reusing teacher checkpoint: $checkpoint"
        if [[ "${DRY_RUN:-0}" != "1" ]]; then
            "$PYTHON_BIN" "$PROJECT_DIR/scripts/audit_teachers.py" \
                --checkpoint-root "$PROJECT_DIR/checkpoints" \
                --dataset "$dataset" --pair "$pair" --seed "$seed"
        fi
    else
        if [[ -f "$log_root/teacher.log" ]]; then
            echo "Teacher checkpoint must not be replaced while formal logs exist: $log_root" >&2
            exit 1
        fi
        run_formal_case "teacher seed=$seed" "$log_root/teacher.log" \
            "${common[@]}" --method teacher "${teacher_checkpoint_args[@]}"
    fi
    run_formal_case "student seed=$seed" "$log_root/student.log" \
        "${common[@]}" --method student
    run_formal_case "student KD seed=$seed" "$log_root/student_kd.log" \
        "${common[@]}" --method student_kd --teacher-checkpoint "$checkpoint"
    for size in small large; do
        run_formal_case "InherNet $size seed=$seed" "$log_root/inhernet_${size}.log" \
            "${common[@]}" --method inhernet \
            --size "$size" --teacher-checkpoint "$checkpoint" \
            --compressed-train-mode supervised
    done
    if [[ "$hetero_train_mode" == "distillation" ]]; then
        run_formal_case "InherNet large with Hetero-matched objective seed=$seed" \
            "$log_root/inhernet_large_matched_hetero_objective.log" \
            "${common[@]}" --method inhernet --size large \
            --teacher-checkpoint "$checkpoint" \
            --search-candidate formal_inhernet_large_matched_hetero_objective \
            "${HETERO_OBJECTIVE_ARGS[@]}"
    fi
    run_formal_case "Hetero recipe=$hetero_recipe_id seed=$seed" \
        "$log_root/hetero_${hetero_recipe_id}.log" \
        "${common[@]}" --method hetero --size large --teacher-checkpoint "$checkpoint" \
        "${HETERO_RECIPE_ARGS[@]}"
    if [[ "$hetero_train_mode" == "distillation" ]]; then
        run_formal_case "Hetero supervised control recipe=$hetero_recipe_id seed=$seed" \
            "$log_root/hetero_${hetero_recipe_id}_supervised_control.log" \
            "${common[@]}" --method hetero --size large --teacher-checkpoint "$checkpoint" \
            --search-candidate formal_hetero_supervised_control \
            "${HETERO_SUPERVISED_ARGS[@]}"
    fi
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/summarize_search.py" "$log_root" \
            --output "$log_root/summary.csv"
    fi
done
