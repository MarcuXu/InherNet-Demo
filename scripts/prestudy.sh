#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
original_args=("$@")

target="${1:-all}"
if (($# > 0)); then shift; fi
case "$target" in
    oxford_pets) targets=("oxford_pets resnet34_to_resnet18") ;;
    cifar100) targets=("cifar100 resnet56_to_resnet20") ;;
    all) targets=(
        "oxford_pets resnet34_to_resnet18"
        "cifar100 resnet56_to_resnet20"
    ) ;;
    *)
        echo "Usage: $0 {oxford_pets|cifar100|all} [runtime arguments...]" >&2
        exit 2
        ;;
esac

extra=("$@")
reject_identity_overrides "${extra[@]}"
reject_formal_training_overrides "${extra[@]}"
export DEVICE="${DEVICE:-cuda}"
export RESUME="${RESUME:-1}"
scope="${PRESTUDY_SCOPE:-maintained}"
case "$scope" in
    maintained|research|all) ;;
    *)
        echo "PRESTUDY_SCOPE must be maintained, research, or all: $scope" >&2
        exit 2
        ;;
esac
seed="${PRESTUDY_SEED:-42}"
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "PRESTUDY_SEED must be a non-negative integer: $seed" >&2
    exit 2
fi
default_background
if maybe_launch_background prestudy "${original_args[@]}"; then exit 0; fi

run_diagnostic() {
    local label="$1" log_path="$2"
    shift 2
    if [[ -f "$log_path" ]]; then
        if [[ "$RESUME" == "1" ]] && rg -q '^INHERITANCE_DIAGNOSTICS ' "$log_path"; then
            if "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_completed_log.py" \
                --diagnostics-only "$log_path" -- "$@"; then
                echo "Skipping completed pre-study diagnostic: $label"
                return
            fi
            echo "Pre-study log is stale; move it before rerunning this cell: $log_path" >&2
            exit 1
        fi
        echo "Pre-study log already exists or is incomplete: $log_path" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$log_path")"
    INHERNET_RUN_LOG="$log_path" run_case "$label" "$@"
}

for target_spec in "${targets[@]}"; do
    read -r dataset pair <<<"$target_spec"
    checkpoint="${TEACHER_CHECKPOINT:-$PROJECT_DIR/checkpoints/search/$dataset/$pair/teacher_seed_${seed}.pt}"
    if [[ ! -f "$checkpoint" && "${DRY_RUN:-0}" != "1" ]]; then
        echo "Missing pre-study teacher: $checkpoint" >&2
        echo "Train it with: SEARCH_ARTIFACT_SCOPE=registry SEARCH_SEEDS=$seed scripts/search.sh teachers $dataset $pair" >&2
        exit 1
    fi
    validation_args=()
    [[ "$dataset" == "cifar100" ]] && validation_args+=(--search-validation)
    common=(--dataset "$dataset" --pair "$pair" --seed "$seed" --device "$DEVICE" \
        --plot-mode none --no-final-test --inheritance-diagnostics-only \
        --teacher-checkpoint "$checkpoint" "${validation_args[@]}" "${extra[@]}")
    log_root="$PROJECT_DIR/logs/prestudy/$dataset/$pair/seed_${seed}"

    if [[ "$scope" == "maintained" || "$scope" == "all" ]]; then
        run_diagnostic "$dataset InherNet registered-rank reference" "$log_root/inhernet.log" \
            "${common[@]}" --method inhernet --size large \
            --compressed-train-mode supervised --search-candidate prestudy_inhernet

        for allocation in unweighted_uniform weighted_uniform; do
            run_diagnostic "$dataset Hetero allocation=$allocation" "$log_root/${allocation}.log" \
                "${common[@]}" --method hetero --size large \
                --compressed-train-mode supervised --search-candidate "prestudy_${allocation}" \
                --hetero-allocation-scale "$allocation" \
                --hetero-expert-noise-scale 0 --aux-loss-weight 0
        done

        run_diagnostic "$dataset Hetero mean-preserving conditional lift" \
            "$log_root/weighted_uniform_noise_001.log" \
            "${common[@]}" --method hetero --size large \
            --compressed-train-mode supervised \
            --search-candidate prestudy_weighted_uniform_noise_001 \
            --hetero-allocation-scale weighted_uniform \
            --hetero-expert-noise-scale 0.01 --aux-loss-weight 0
    fi

    if [[ "$scope" == "research" || "$scope" == "all" ]]; then
        for allocation in research_relative research_nested_relative research_total_output; do
            run_diagnostic "$dataset Hetero research allocation=$allocation" \
                "$log_root/${allocation}.log" \
                "${common[@]}" --method hetero --size large \
                --compressed-train-mode supervised --search-candidate "prestudy_${allocation}" \
                --hetero-allocation-scale "$allocation" \
                --hetero-expert-noise-scale 0 --aux-loss-weight 0
        done
    fi
done
