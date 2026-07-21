#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
if maybe_launch_background ablation "$@"; then exit 0; fi
require_dataset_pair "$@"

dataset="$1"
pair="$2"
shift 2
seed="${SEED:-42}"
selection_checkpoint="$PROJECT_DIR/checkpoints/search/selection/$dataset/$pair/teacher_seed_${seed}.pt"
checkpoint="${TEACHER_CHECKPOINT:-$selection_checkpoint}"
if [[ ! -f "$checkpoint" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "Missing teacher checkpoint: $checkpoint" >&2
    echo "Run SEARCH_SEEDS=$seed scripts/search.sh teachers $dataset $pair first." >&2
    exit 1
fi
extra=("$@")
reject_identity_overrides "${extra[@]}"
reject_formal_training_overrides "${extra[@]}"
export RESUME="${RESUME:-1}"
load_hetero_recipe "$dataset"
hetero_recipe_id="${HETERO_RECIPE_ARGS[1]}"
common=(--dataset "$dataset" --pair "$pair" --seed "$seed" --device "${DEVICE:-cuda}" --plot-mode none \
    --teacher-checkpoint "$checkpoint" --no-final-test "${extra[@]}")
if [[ "$dataset" == "cifar10" || "$dataset" == "cifar100" || "$dataset" == glue_* ]]; then
    common+=(--search-validation)
fi

log_root="$PROJECT_DIR/logs/ablation/$dataset/$pair/seed_${seed}/recipe_${hetero_recipe_id}"
run_ablation_case() {
    local label="$1" size="$2" variant="$3"
    shift 3
    local log_path="$log_root/size_${size}/${variant}.log"
    local -a command=("${common[@]}" --size "$size" \
        --search-candidate "ablation_${variant}" "$@")
    mkdir -p "$(dirname "$log_path")"
    if [[ -f "$log_path" ]]; then
        if [[ "${RESUME:-0}" == "1" ]] && rg -q '^RUN_SUMMARY ' "$log_path"; then
            "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_completed_log.py" "$log_path" -- \
                "${command[@]}"
            echo "Skipping completed ablation: $label"
            return
        fi
        echo "Ablation log already exists: $log_path (set RESUME=1 to skip completed runs)" >&2
        exit 1
    fi
    INHERNET_RUN_LOG="$log_path" run_case "$label" "${command[@]}"
}

resolve_ablation_recipe() {
    local variant="$1"
    shift
    local -a resolved=("${HETERO_RECIPE_ARGS[@]}") filtered=()
    local option value argument index
    while (($# > 0)); do
        option="$1"
        value="$2"
        shift 2
        filtered=()
        for ((index=0; index<${#resolved[@]}; index++)); do
            argument="${resolved[$index]}"
            if [[ "$argument" == "$option" ]]; then
                ((index+=1))
            else
                filtered+=("$argument")
            fi
        done
        resolved=("${filtered[@]}")
        if [[ "$value" == "__FLAG__" ]]; then
            resolved+=("$option")
        elif [[ "$value" != "__REMOVE__" ]]; then
            resolved+=("$option" "$value")
        fi
    done
    if [[ "$variant" == "full" ]]; then
        ABLATION_RECIPE_ARGS=("${resolved[@]}")
        return
    fi
    filtered=()
    for ((index=0; index<${#resolved[@]}; index++)); do
        argument="${resolved[$index]}"
        if [[ "$argument" == "--hetero-recipe-id" ]]; then
            ((index+=1))
        else
            filtered+=("$argument")
        fi
    done
    ABLATION_RECIPE_ARGS=("${filtered[@]}" --hetero-recipe-id "${hetero_recipe_id}_${variant}")
}

for size in small large; do
    run_ablation_case "InherNet size=$size" "$size" "inhernet_${size}" \
        --method inhernet --compressed-train-mode supervised
done

resolve_ablation_recipe full
run_ablation_case "Hetero-Lite capacity control" small hetero_lite \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"
run_ablation_case "Hetero full" large full \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"

resolve_ablation_recipe calibration_4_batches --max-calib-batches 4
run_ablation_case "Hetero with 4 calibration batches" large calibration_4_batches \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"
resolve_ablation_recipe calibration_8_batches --max-calib-batches 8
run_ablation_case "Hetero with 8 calibration batches" large calibration_8_batches \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"

resolve_ablation_recipe unweighted_uniform --hetero-allocation-scale unweighted_uniform
run_ablation_case "Hetero without activation weighting" large unweighted_uniform \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"
resolve_ablation_recipe no_noise --hetero-expert-noise-scale 0
run_ablation_case "Hetero without expert perturbation" large no_noise \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"
resolve_ablation_recipe no_balance --aux-loss-weight 0
run_ablation_case "Hetero without balance loss" large no_balance \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"
resolve_ablation_recipe no_noise_no_balance \
    --hetero-expert-noise-scale 0 --aux-loss-weight 0
run_ablation_case "Hetero without expert perturbation or balance" large no_noise_no_balance \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"
resolve_ablation_recipe frozen_router --freeze-hetero-router __FLAG__
run_ablation_case "Hetero with fixed uniform routers" large frozen_router \
    --method hetero "${ABLATION_RECIPE_ARGS[@]}"

[[ "${DRY_RUN:-0}" == "1" ]] || "$PYTHON_BIN" "$PROJECT_DIR/scripts/summarize_search.py" \
    "$log_root" --output "$log_root/summary.csv"
