#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
original_args=("$@")

phase="${1:-all}"
if (($# > 0)); then shift; fi
case "$phase" in teachers|mechanism|optimization|distillation|confirmation|all) ;; *)
    echo "Usage: $0 {teachers|mechanism|optimization|distillation|confirmation|all} [DATASET PAIR] [runtime arguments...]" >&2
    exit 2
esac

mechanism_targets=(
    "cifar10 resnet50_to_resnet18"
    "cifar100 resnet56_to_resnet20"
    "oxford_pets resnet34_to_resnet18"
    "glue_sst2 bert4_to_bert2"
    "glue_stsb bert4_to_bert2"
)
optimization_targets=(
    "cifar10 resnet50_to_resnet18"
    "cifar100 resnet56_to_resnet20"
    "oxford_pets resnet34_to_resnet18"
    "glue_sst2 bert4_to_bert2"
    "glue_stsb bert4_to_bert2"
)
distillation_targets=(
    "cifar10 resnet50_to_resnet18"
    "oxford_pets resnet34_to_resnet18"
    "glue_sst2 bert4_to_bert2"
    "glue_stsb bert4_to_bert2"
)
confirmation_targets=("${mechanism_targets[@]}")
search_teacher_targets=(
    "cifar10 resnet50_to_resnet18"
    "cifar100 resnet56_to_resnet20"
    "oxford_pets resnet34_to_resnet18"
    "glue_sst2 bert4_to_bert2"
    "glue_stsb bert4_to_bert2"
)
teacher_targets=(
    "cifar10 resnet50_to_resnet18"
    "cifar100 resnet32_to_resnet8"
    "cifar100 resnet32x4_to_resnet8x4"
    "cifar100 vgg13_to_vgg8"
    "cifar100 wrn40_2_to_wrn40_1"
    "cifar100 wrn40_2_to_wrn16_2"
    "cifar100 resnet56_to_resnet20"
    "cifar100 resnet110_to_resnet32"
    "cifar100 resnet110_to_resnet20"
    "oxford_pets resnet34_to_resnet18"
    "glue_mrpc bert4_to_bert2"
    "glue_qqp bert4_to_bert2"
    "glue_sst2 bert4_to_bert2"
    "glue_mnli bert4_to_bert2"
    "glue_rte bert4_to_bert2"
    "glue_qnli bert4_to_bert2"
    "glue_cola bert4_to_bert2"
    "glue_stsb bert4_to_bert2"
)
targets=("${mechanism_targets[@]}")
explicit_target=0

if (($# >= 1)) && [[ "$1" != --* ]]; then
    if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "DATASET and PAIR must be provided together." >&2
        exit 2
    fi
    targets=("$1 $2")
    explicit_target=1
    shift 2
fi
extra=("$@")
reject_identity_overrides "${extra[@]}"
reject_search_overrides "${extra[@]}"

export DEVICE="${DEVICE:-cuda}"
export RESUME="${RESUME:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
search_artifact_scope="${SEARCH_ARTIFACT_SCOPE:-selection}"
case "$search_artifact_scope" in selection|registry) ;; *)
    echo "Unknown SEARCH_ARTIFACT_SCOPE: $search_artifact_scope" >&2
    exit 2
esac
if [[ "$search_artifact_scope" == "registry" && "$phase" != "teachers" ]]; then
    echo "Registry artifact scope is restricted to teacher maintenance." >&2
    exit 2
fi
search_seed_spec="${SEARCH_SEEDS:-42,123,2026}"
IFS=',' read -r -a search_seeds <<<"$search_seed_spec"
if ((${#search_seeds[@]} == 0)); then
    echo "SEARCH_SEEDS must contain at least one integer seed." >&2
    exit 2
fi
declare -A seen_seeds=()
for search_seed in "${search_seeds[@]}"; do
    if [[ ! "$search_seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid SEARCH_SEEDS entry: $search_seed" >&2
        exit 2
    fi
    if [[ -n "${seen_seeds[$search_seed]:-}" ]]; then
        echo "Duplicate SEARCH_SEEDS entry: $search_seed" >&2
        exit 2
    fi
    seen_seeds[$search_seed]=1
done
export SEARCH_SEEDS="$search_seed_spec"

read -r reference_aux_weight reference_shrinkage reference_noise_scale < <(
    awk -F, '$1 == "reference" { print $2, $3, $4; exit }' \
        "$PROJECT_DIR/configs/hetero_search_candidates.csv"
)
REFERENCE_HETERO_MECHANISM_ARGS=(
    --aux-loss-weight "$reference_aux_weight"
    --hetero-second-moment-shrinkage "$reference_shrinkage"
    --hetero-expert-noise-scale "$reference_noise_scale"
    --hetero-allocation-scale weighted_uniform
)
REFERENCE_HETERO_OPTIMIZER_ARGS=(--lr-scale 1.0)

device="$DEVICE"

candidate_selected() {
    local candidate_id="$1"
    [[ -z "${SEARCH_CANDIDATES:-}" ]] && return 0
    [[ ",${SEARCH_CANDIDATES}," == *",${candidate_id},"* ]]
}

validate_candidate_filter() {
    local selected_phase="$1" config requested candidate_id
    [[ -z "${SEARCH_CANDIDATES:-}" ]] && return
    if [[ "$selected_phase" == "teachers" ]]; then
        echo "SEARCH_CANDIDATES does not apply to the $selected_phase phase." >&2
        exit 2
    fi
    case "$selected_phase" in
        mechanism) config="$PROJECT_DIR/configs/hetero_search_candidates.csv" ;;
        optimization) config="$PROJECT_DIR/configs/lr_scale_search_candidates.csv" ;;
        distillation) config="$PROJECT_DIR/configs/distillation_search_candidates.csv" ;;
        confirmation) config="$PROJECT_DIR/configs/hetero_confirmation_candidates.csv" ;;
    esac
    IFS=',' read -r -a requested <<<"$SEARCH_CANDIDATES"
    for candidate_id in "${requested[@]}"; do
        if ! awk -F, -v id="$candidate_id" 'NR > 1 && $1 == id { found=1 } END { exit !found }' "$config"; then
            echo "Unknown $selected_phase search candidate: $candidate_id" >&2
            exit 2
        fi
    done
}

run_logged_case() {
    local label="$1" log_path="$2"
    shift 2
    if [[ -f "$log_path" ]]; then
        if [[ "${RESUME:-0}" == "1" ]] && rg -q '^RUN_SUMMARY ' "$log_path"; then
            "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_completed_log.py" "$log_path" -- "$@"
            echo "Skipping completed candidate: $label"
            return
        fi
        if rg -q '^RUN_SUMMARY ' "$log_path"; then
            echo "Completed search log already exists: $log_path" >&2
            echo "Use the default RESUME=1 behavior to skip it." >&2
        else
            echo "Search log exists but is incomplete: $log_path" >&2
            echo "Inspect or move it before retrying; search logs are never overwritten." >&2
        fi
        exit 1
    fi
    INHERNET_RUN_LOG="$log_path" run_case "$label" "$@"
}

run_target_phase() {
    local selected_phase="$1" dataset="$2" pair="$3"
    local checkpoint log_root candidate_id lr_scale aux_weight shrinkage noise_scale
    local temperature kd_weight ce_weight size candidate_log replacement_log applicable method
    local train_mode profile target_profile
    local -a requested_candidates
    if [[ "$search_artifact_scope" == "selection" ]]; then
        checkpoint="$PROJECT_DIR/checkpoints/search/selection/$dataset/$pair/teacher_seed_${seed}.pt"
        log_root="$PROJECT_DIR/logs/search/selection/$dataset/$pair/seed_${seed}"
    else
        checkpoint="$PROJECT_DIR/checkpoints/search/$dataset/$pair/teacher_seed_${seed}.pt"
        log_root="$PROJECT_DIR/logs/search/$dataset/$pair/seed_${seed}"
    fi
    mkdir -p "$(dirname "$checkpoint")" "$log_root"
    validation_args=()
    if [[ "$dataset" == "cifar10" || "$dataset" == "cifar100" || \
        ( "$search_artifact_scope" == "selection" && "$dataset" == glue_* ) ]]; then
        validation_args+=(--search-validation)
    fi
    common=(--dataset "$dataset" --pair "$pair" --seed "$seed" --device "$device" \
        --plot-mode none --no-final-test "${validation_args[@]}" "${extra[@]}")
    mapfile -t REGISTERED_OBJECTIVE_ARGS < <(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/hetero_recipes.py" registered-objective "$dataset"
    )

    if [[ "$selected_phase" == "teachers" ]]; then
        if [[ "$search_artifact_scope" == "selection" && "${OVERWRITE_TEACHER:-0}" == "1" ]]; then
            inherited_log=$(rg --files "$log_root" 2>/dev/null \
                | rg '/(mechanism|optimization|distillation|confirmation)/.*\.log$' \
                | head -n 1 || true)
            if [[ -n "$inherited_log" ]]; then
                echo "Cannot replace a selection teacher after inherited logs exist: $inherited_log" >&2
                echo "Use a fresh selection namespace instead of mixing teacher states." >&2
                exit 1
            fi
        fi
        if [[ -f "$checkpoint" && "${OVERWRITE_TEACHER:-0}" != "1" ]]; then
            echo "Reusing teacher checkpoint: $checkpoint"
            if [[ "${DRY_RUN:-0}" != "1" ]]; then
                if [[ "$search_artifact_scope" == "selection" ]]; then
                    checkpoint_root="$PROJECT_DIR/checkpoints/search/selection"
                else
                    checkpoint_root="$PROJECT_DIR/checkpoints/search"
                fi
                "$PYTHON_BIN" "$PROJECT_DIR/scripts/audit_teachers.py" \
                    --checkpoint-root "$checkpoint_root" \
                    --dataset "$dataset" --pair "$pair" --seed "$seed"
            fi
            return
        fi
        teacher_args=(--teacher-checkpoint "$checkpoint")
        [[ "${OVERWRITE_TEACHER:-0}" == "1" ]] && teacher_args+=(--overwrite-teacher-checkpoint)
        if [[ "${OVERWRITE_TEACHER:-0}" == "1" ]]; then
            replacement_log="$log_root/.teacher_replacement_${BASHPID}.log"
            run_logged_case "search teacher $dataset/$pair" "$replacement_log" \
                "${common[@]}" --method teacher "${teacher_args[@]}"
            if [[ "${DRY_RUN:-0}" != "1" ]]; then
                mv -f "$replacement_log" "$log_root/teacher.log"
            fi
        else
            if [[ -f "$log_root/teacher.log" ]]; then
                echo "Teacher checkpoint is missing but its prior log still exists: $log_root/teacher.log" >&2
                echo "Inspect or move the log before retraining; teacher evidence is never overwritten." >&2
                exit 1
            fi
            run_logged_case "search teacher $dataset/$pair" "$log_root/teacher.log" \
                "${common[@]}" --method teacher "${teacher_args[@]}"
        fi
        return
    fi

    if [[ ! -f "$checkpoint" && "${DRY_RUN:-0}" != "1" ]]; then
        echo "Missing search teacher checkpoint: $checkpoint" >&2
        exit 1
    fi

    if [[ "$selected_phase" == "mechanism" ]]; then
        while IFS=, read -r candidate_id aux_weight shrinkage noise_scale; do
            [[ "$candidate_id" == "candidate_id" || -z "$candidate_id" ]] && continue
            candidate_selected "$candidate_id" || continue
            candidate_log="$log_root/mechanism/hetero/size_large/${candidate_id}.log"
            mkdir -p "$(dirname "$candidate_log")"
            run_logged_case "$dataset Hetero candidate=$candidate_id" "$candidate_log" \
                "${common[@]}" --method hetero --teacher-checkpoint "$checkpoint" \
                --size large --search-candidate "mechanism_${candidate_id}" \
                --aux-loss-weight "$aux_weight" \
                --hetero-second-moment-shrinkage "$shrinkage" \
                --hetero-expert-noise-scale "$noise_scale" \
                --hetero-allocation-scale weighted_uniform \
                "${REFERENCE_HETERO_OPTIMIZER_ARGS[@]}" \
                "${REGISTERED_OBJECTIVE_ARGS[@]}"
        done <"$PROJECT_DIR/configs/hetero_search_candidates.csv"
    elif [[ "$selected_phase" == "optimization" ]]; then
        while IFS=, read -r candidate_id lr_scale; do
            [[ "$candidate_id" == "candidate_id" || -z "$candidate_id" ]] && continue
            candidate_selected "$candidate_id" || continue
            candidate_log="$log_root/optimization/hetero/size_large/${candidate_id}.log"
            mkdir -p "$(dirname "$candidate_log")"
            run_logged_case "$dataset Hetero candidate=$candidate_id" "$candidate_log" \
                "${common[@]}" --method hetero --teacher-checkpoint "$checkpoint" \
                --size large --search-candidate "optimization_${candidate_id}" \
                "${REFERENCE_HETERO_MECHANISM_ARGS[@]}" \
                "${REGISTERED_OBJECTIVE_ARGS[@]}" \
                --lr-scale "$lr_scale"
        done <"$PROJECT_DIR/configs/lr_scale_search_candidates.csv"
    elif [[ "$selected_phase" == "distillation" ]]; then
        if [[ "$dataset" == "cifar100" ]]; then
            echo "Skipping distillation search for supervised-default dataset: $dataset"
            return
        fi
        distillation_candidate_applies() {
            local candidate="$1"
            [[ "$dataset" == "glue_stsb" && "$candidate" == temperature_* ]] && return 1
            [[ "$dataset" != "cifar10" && "$candidate" == "registered_reference" ]] && return 1
            return 0
        }
        if [[ -n "${SEARCH_CANDIDATES:-}" ]]; then
            IFS=',' read -r -a requested_candidates <<<"$SEARCH_CANDIDATES"
            applicable=0
            for candidate_id in "${requested_candidates[@]}"; do
                distillation_candidate_applies "$candidate_id" && applicable=1
            done
            if [[ "$applicable" == "0" ]]; then
                echo "No requested distillation candidate applies to $dataset." >&2
                exit 2
            fi
        fi
        while IFS=, read -r candidate_id temperature kd_fraction; do
            [[ "$candidate_id" == "candidate_id" || -z "$candidate_id" ]] && continue
            candidate_selected "$candidate_id" || continue
            distillation_candidate_applies "$candidate_id" || continue
            candidate_log="$log_root/distillation/hetero/size_large/${candidate_id}.log"
            mkdir -p "$(dirname "$candidate_log")"
            if [[ "$candidate_id" == "supervised" ]]; then
                distillation_args=(--compressed-train-mode supervised)
            else
                distillation_args=(--compressed-train-mode distillation --kd-temperature "$temperature")
                [[ -n "$kd_fraction" ]] && distillation_args+=(--kd-fraction "$kd_fraction")
            fi
            run_logged_case "$dataset Hetero candidate=$candidate_id" "$candidate_log" \
                "${common[@]}" --method hetero --teacher-checkpoint "$checkpoint" \
                --size large --search-candidate "distillation_${candidate_id}" \
                "${REFERENCE_HETERO_MECHANISM_ARGS[@]}" \
                "${REFERENCE_HETERO_OPTIMIZER_ARGS[@]}" \
                "${distillation_args[@]}"
        done <"$PROJECT_DIR/configs/distillation_search_candidates.csv"
    elif [[ "$selected_phase" == "confirmation" ]]; then
        target_profile="$(hetero_recipe_profile "$dataset")"
        while IFS=, read -r candidate_id profile aux_weight shrinkage noise_scale allocation_scale lr_scale \
            train_mode temperature kd_fraction; do
            [[ "$candidate_id" == "candidate_id" || -z "$candidate_id" ]] && continue
            candidate_selected "$candidate_id" || continue
            [[ "$profile" == "$target_profile" ]] || continue
            candidate_log="$log_root/confirmation/hetero/size_large/${candidate_id}.log"
            mkdir -p "$(dirname "$candidate_log")"
            confirmation_args=(
                --hetero-recipe-id "$candidate_id"
                --aux-loss-weight "$aux_weight"
                --hetero-second-moment-shrinkage "$shrinkage"
                --hetero-expert-noise-scale "$noise_scale"
                --hetero-allocation-scale "$allocation_scale"
                --lr-scale "$lr_scale"
            )
            if [[ "$train_mode" == "supervised" ]]; then
                confirmation_args+=(--compressed-train-mode supervised)
            else
                confirmation_args+=(
                    --compressed-train-mode distillation
                    --kd-temperature "$temperature"
                )
                [[ -n "$kd_fraction" ]] && confirmation_args+=(--kd-fraction "$kd_fraction")
            fi
            run_logged_case "$dataset Hetero finalist=$candidate_id" "$candidate_log" \
                "${common[@]}" --method hetero --teacher-checkpoint "$checkpoint" \
                --size large --search-candidate "confirmation_${candidate_id}" \
                "${confirmation_args[@]}"
        done <"$PROJECT_DIR/configs/hetero_confirmation_candidates.csv"
    fi
    [[ "${DRY_RUN:-0}" == "1" ]] || "$PYTHON_BIN" "$PROJECT_DIR/scripts/summarize_search.py" "$log_root"
}

run_phase() {
    local selected_phase="$1" item dataset pair
    local -a phase_targets=("${targets[@]}")
    if [[ "$explicit_target" == "0" ]]; then
        case "$selected_phase" in
            teachers)
                phase_targets=()
                if [[ "$search_artifact_scope" == "selection" ]]; then
                    phase_targets=("${search_teacher_targets[@]}")
                else case "${TEACHER_GROUP:-all}" in
                    all) phase_targets=("${teacher_targets[@]}") ;;
                    glue)
                        for item in "${teacher_targets[@]}"; do
                            [[ "$item" == glue_* ]] && phase_targets+=("$item")
                        done
                        ;;
                    vision)
                        for item in "${teacher_targets[@]}"; do
                            [[ "$item" != glue_* ]] && phase_targets+=("$item")
                        done
                        ;;
                    *)
                        echo "Unknown TEACHER_GROUP: ${TEACHER_GROUP} (expected all, glue, or vision)" >&2
                        exit 2
                        ;;
                esac
                fi
                ;;
            mechanism) phase_targets=("${mechanism_targets[@]}") ;;
            optimization) phase_targets=("${optimization_targets[@]}") ;;
            distillation) phase_targets=("${distillation_targets[@]}") ;;
            confirmation) phase_targets=("${confirmation_targets[@]}") ;;
        esac
    fi
    for item in "${phase_targets[@]}"; do
        read -r dataset pair <<<"$item"
        run_target_phase "$selected_phase" "$dataset" "$pair"
    done
}

if [[ "$phase" == "all" && -n "${SEARCH_CANDIDATES:-}" ]]; then
    echo "SEARCH_CANDIDATES requires an explicit non-all phase." >&2
    exit 2
elif [[ "$phase" != "all" ]]; then
    validate_candidate_filter "$phase"
fi
if [[ "$phase" == "confirmation" ]]; then
    finalist_count=$("$PYTHON_BIN" "$PROJECT_DIR/scripts/hetero_recipes.py" validate-confirmation)
    if ((finalist_count < 2)); then
        echo "Confirmation requires the registered reference and at least one manually committed finalist." >&2
        exit 2
    fi
fi
if [[ "$phase" == "teachers" && "$explicit_target" == "0" ]]; then
    if [[ "$search_artifact_scope" == "selection" && "${TEACHER_GROUP:-all}" != "all" ]]; then
        echo "TEACHER_GROUP applies only to registry teacher maintenance." >&2
        exit 2
    elif [[ "$search_artifact_scope" == "registry" ]]; then
        case "${TEACHER_GROUP:-all}" in
            all|glue|vision) ;;
            *)
                echo "Unknown TEACHER_GROUP: ${TEACHER_GROUP} (expected all, glue, or vision)" >&2
                exit 2
                ;;
        esac
    fi
fi
default_background
if maybe_launch_background search "${original_args[@]}"; then exit 0; fi

for seed in "${search_seeds[@]}"; do
    echo
    echo "######## Search seed $seed ########"
    if [[ "$phase" == "all" ]]; then
        for item in "${search_teacher_targets[@]}"; do
            read -r dataset pair <<<"$item"
            run_target_phase teachers "$dataset" "$pair"
        done
        run_phase mechanism
        run_phase optimization
        run_phase distillation
    else
        run_phase "$phase"
    fi
done
