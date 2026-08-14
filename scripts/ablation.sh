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
export RESUME="${RESUME:-1}"
load_inheract_recipe "$dataset"
inheract_recipe_id="${INHERACT_RECIPE_ARGS[1]}"
mapfile -t INHERACT_OBJECTIVE_ARGS < <(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/inheract_recipes.py" selected-objective "$dataset"
)
mapfile -t INHERACT_OPTIMIZER_ARGS < <(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/inheract_recipes.py" selected-optimizer "$dataset"
)

IFS=',' read -r -a ablation_seeds <<<"${ABLATION_SEEDS:-7,17,27}"
if ((${#ablation_seeds[@]} == 0)); then
    echo "ABLATION_SEEDS must contain at least one non-negative integer." >&2
    exit 2
fi
declare -A seen_seeds=()
for seed in "${ablation_seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ || -n "${seen_seeds[$seed]:-}" ]]; then
        echo "ABLATION_SEEDS must contain distinct non-negative integers: ${ABLATION_SEEDS:-7,17,27}" >&2
        exit 2
    fi
    seen_seeds[$seed]=1
done
if [[ -n "${TEACHER_CHECKPOINT:-}" && ${#ablation_seeds[@]} -ne 1 ]]; then
    echo "TEACHER_CHECKPOINT requires exactly one ABLATION_SEEDS value." >&2
    echo "The paired default matrix uses its seed-matched FORMAL_RUN_ID teacher artifacts." >&2
    exit 2
fi
if [[ -z "${FORMAL_RUN_ID:-}" ]]; then
    echo "Ablations require FORMAL_RUN_ID=<completed formal run namespace>." >&2
    echo "Use the same identifier that produced the paired formal teacher checkpoints." >&2
    exit 2
fi
validate_formal_run_id "$FORMAL_RUN_ID"
if maybe_launch_background ablation "${original_args[@]}"; then exit 0; fi

run_ablation_case() {
    local label="$1" size="$2" variant="$3"
    shift 3
    local log_path="$log_root/size_${size}/${variant}.log"
    local -a command=("${common[@]}" --size "$size" \
        --search-candidate "ablation_${variant}" "$@")
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
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        mkdir -p "$(dirname "$log_path")"
    fi
    INHERNET_RUN_LOG="$log_path" run_case "$label" "${command[@]}"
}

resolve_ablation_recipe() {
    local variant="$1"
    shift
    local -a resolved=("${INHERACT_RECIPE_ARGS[@]}") filtered=()
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
        if [[ "$argument" == "--inheract-recipe-id" ]]; then
            ((index+=1))
        else
            filtered+=("$argument")
        fi
    done
    ABLATION_RECIPE_ARGS=("${filtered[@]}" --inheract-recipe-id "${inheract_recipe_id}_${variant}")
}

run_seed_matrix() {
    local size inhernet_train_mode
    for size in small large; do
        if [[ "$size" == "small" ]]; then
            inhernet_train_mode=distillation
        else
            inhernet_train_mode=supervised
        fi
        run_ablation_case "InherNet size=$size" "$size" "inhernet_${size}" \
            --method inhernet --compressed-train-mode "$inhernet_train_mode"
    done

    # One head makes the softmax exactly one: a static rank-r SVD inheritance
    # control at the headline rank, with the selected InherAct objective/step scale.
    run_ablation_case "Direct SVD inheritance control (one head)" large direct_svd \
        --method inhernet --head-num 1 \
        "${INHERACT_OPTIMIZER_ARGS[@]}" "${INHERACT_OBJECTIVE_ARGS[@]}"

    resolve_ablation_recipe full
    run_ablation_case "InherAct-Lite capacity control" small inheract_lite \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"
    run_ablation_case "InherAct full" large full \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"

    resolve_ablation_recipe calibration_4_batches --max-calib-batches 4
    run_ablation_case "InherAct with 4 calibration batches" large calibration_4_batches \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"
    resolve_ablation_recipe calibration_8_batches --max-calib-batches 8
    run_ablation_case "InherAct with 8 calibration batches" large calibration_8_batches \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"

    resolve_ablation_recipe unweighted_uniform --inheract-allocation-scale unweighted_uniform
    run_ablation_case "InherAct without activation weighting" large unweighted_uniform \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"
    resolve_ablation_recipe no_noise --inheract-expert-noise-scale 0
    run_ablation_case "InherAct without expert perturbation" large no_noise \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"
    resolve_ablation_recipe no_balance --aux-loss-weight 0
    run_ablation_case "InherAct without balance loss" large no_balance \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"
    resolve_ablation_recipe no_noise_no_balance \
        --inheract-expert-noise-scale 0 --aux-loss-weight 0
    run_ablation_case "InherAct without expert perturbation or balance" large no_noise_no_balance \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"
    resolve_ablation_recipe frozen_router --freeze-inheract-router __FLAG__
    run_ablation_case "InherAct with fixed uniform routers" large frozen_router \
        --method inheract "${ABLATION_RECIPE_ARGS[@]}"
}

for seed in "${ablation_seeds[@]}"; do
    checkpoint="${TEACHER_CHECKPOINT:-$(formal_checkpoint_for "$FORMAL_RUN_ID" "$dataset" "$pair" "$seed")}"
    if [[ ! -f "$checkpoint" && "${DRY_RUN:-0}" != "1" ]]; then
        echo "Missing formal teacher checkpoint: $checkpoint" >&2
        echo "Run FORMAL_RUN_ID=$FORMAL_RUN_ID scripts/formal.sh $dataset $pair first." >&2
        exit 1
    fi
    ablation_search_validation=0
    if [[ "$dataset" == "cifar10" || "$dataset" == "cifar100" || "$dataset" == glue_* ]]; then
        ablation_search_validation=1
    fi
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        validate_compatible_teacher_checkpoint \
            "$checkpoint" "$dataset" "$pair" "$seed" "$ablation_search_validation"
    fi
    common=(--dataset "$dataset" --pair "$pair" --seed "$seed" --device "${DEVICE:-cuda}" --plot-mode none \
        --teacher-checkpoint "$checkpoint" --no-final-test "${extra[@]}")
    if [[ "$ablation_search_validation" == "1" ]]; then
        common+=(--search-validation)
    fi
    log_root="$PROJECT_DIR/logs/ablation/$FORMAL_RUN_ID/$dataset/$pair/seed_${seed}/recipe_${inheract_recipe_id}"
    run_seed_matrix
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/summarize_search.py" \
            "$log_root" --output "$log_root/summary.csv"
    fi
done
