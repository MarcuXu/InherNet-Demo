#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT_DIR/scripts/run.sh"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    project_python="${HOME}/miniconda3/envs/inherdemo/bin/python"
    if [[ "${CONDA_DEFAULT_ENV:-}" == "inherdemo" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif [[ -x "$project_python" ]]; then
        PYTHON_BIN="$project_python"
    else
        echo "Cannot find the inherdemo Python environment." >&2
        echo "Activate it with: conda activate inherdemo" >&2
        echo "Or set PYTHON_BIN=/path/to/python explicitly." >&2
        exit 1
    fi
fi
if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 1
fi
python_dir="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
PYTHON_BIN="$python_dir/$(basename "$PYTHON_BIN")"
export PYTHON_BIN

default_background() {
    if [[ "${INHERNET_BACKGROUND_CHILD:-0}" == "1" ]]; then
        return
    fi
    if [[ "${FOREGROUND:-0}" == "1" ]]; then
        export BACKGROUND=0
    elif [[ -z "${BACKGROUND+x}" ]]; then
        export BACKGROUND=1
    fi
}

hetero_display_name() {
    case "$1" in
        large) printf 'Hetero\n' ;;
        small) printf 'Hetero-Lite\n' ;;
        *) echo "Unknown Hetero size: $1" >&2; return 2 ;;
    esac
}

hetero_recipe_profile() {
    case "$1" in
        cifar10|cifar100|oxford_pets) printf '%s\n' "$1" ;;
        glue_stsb) printf 'glue_regression\n' ;;
        glue_*) printf 'glue_classification\n' ;;
        *) echo "No Hetero recipe profile for dataset: $1" >&2; return 2 ;;
    esac
}

require_dataset_pair() {
    if (($# < 2)); then
        echo "Usage: $0 DATASET PAIR [additional demo_code.py arguments...]" >&2
        exit 2
    fi
}

checkpoint_for() {
    local dataset="$1"
    local pair="$2"
    local seed="$3"
    printf '%s/checkpoints/%s/%s/teacher_seed_%s.pt\n' "$PROJECT_DIR" "$dataset" "$pair" "$seed"
}

reject_identity_overrides() {
    local argument
    for argument in "$@"; do
        case "$argument" in
            --dataset|--dataset=*|--pair|--pair=*|--seed|--seed=*|--method|--method=*|\
            --teacher-checkpoint|--teacher-checkpoint=*|\
            --checkpoint-root|--checkpoint-root=*|--overwrite-teacher-checkpoint|\
            --size|--size=*|--compressed-train-mode|\
            --compressed-train-mode=*|--smoke-test|--device|--device=*|\
            --search-candidate|--search-candidate=*|--search-validation|\
            --hetero-recipe-id|--hetero-recipe-id=*|--inheritance-diagnostics-only)
                echo "Reserved experiment-matrix argument is not allowed in extras: $argument" >&2
                exit 2
                ;;
        esac
    done
}

load_hetero_recipe() {
    local dataset="$1"
    mapfile -t HETERO_RECIPE_ARGS < <(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/hetero_recipes.py" selected "$dataset"
    )
    if ((${#HETERO_RECIPE_ARGS[@]} == 0)); then
        echo "No selected Hetero recipe resolved for $dataset." >&2
        return 1
    fi
}

reject_search_overrides() {
    local argument
    for argument in "$@"; do
        case "$argument" in
            --epochs|--epochs=*|--lr|--lr=*|--lr-scale|--lr-scale=*|--aux-loss-weight|\
            --aux-loss-weight=*|--hetero-second-moment-shrinkage|\
            --hetero-second-moment-shrinkage=*|--hetero-expert-noise-scale|\
            --hetero-expert-noise-scale=*|--hetero-allocation-scale|\
            --hetero-allocation-scale=*|--kd-temperature|--kd-temperature=*|\
            --freeze-hetero-router|\
            --kd-weight|--kd-weight=*|--ce-weight|--ce-weight=*|--kd-fraction|\
            --kd-fraction=*|--optimizer|--optimizer=*|\
            --batch-size|--batch-size=*|--momentum|--momentum=*|--weight-decay|\
            --weight-decay=*|--rank|--rank=*|--head-num|--head-num=*|\
            --max-calib-batches|--max-calib-batches=*|--hetero-max-features-per-batch|\
            --hetero-max-features-per-batch=*|--final-test|--no-final-test)
                echo "Search-controlled argument is not allowed in extras: $argument" >&2
                exit 2
                ;;
        esac
    done
}

reject_formal_training_overrides() {
    local argument
    for argument in "$@"; do
        case "$argument" in
            --optimizer|--optimizer=*|--batch-size|--batch-size=*|--epochs|--epochs=*|\
            --lr|--lr=*|--lr-scale|--lr-scale=*|--momentum|--momentum=*|\
            --weight-decay|--weight-decay=*|--kd-temperature|--kd-temperature=*|\
            --kd-weight|--kd-weight=*|--ce-weight|--ce-weight=*|--kd-fraction|\
            --kd-fraction=*|--rank|--rank=*|\
            --size|--size=*|--head-num|--head-num=*|\
            --aux-loss-weight|--aux-loss-weight=*|\
            --max-calib-batches|--max-calib-batches=*|--hetero-max-features-per-batch|\
            --hetero-max-features-per-batch=*|--hetero-second-moment-shrinkage|\
            --hetero-second-moment-shrinkage=*|--hetero-expert-noise-scale|\
            --hetero-expert-noise-scale=*|--hetero-allocation-scale|\
            --hetero-allocation-scale=*|--freeze-hetero-router|\
            --final-test|--no-final-test)
                echo "Formal launcher training override is not allowed: $argument" >&2
                echo "Freeze settings through search, then run the selected method explicitly with scripts/run.sh." >&2
                exit 2
                ;;
        esac
    done
}

maybe_launch_background() {
    local label="$1"
    shift
    if [[ "${DRY_RUN:-0}" == "1" || "${BACKGROUND:-0}" != "1" || "${INHERNET_BACKGROUND_CHILD:-0}" == "1" ]]; then
        return 1
    fi
    local timestamp job_dir stdout_log pid_file
    timestamp="$(date -u +%Y%m%d_%H%M%S_%N)"
    job_dir="$PROJECT_DIR/logs/jobs/${label}_${timestamp}"
    stdout_log="$job_dir/stdout.log"
    pid_file="$job_dir/job.pid"
    mkdir -p "$job_dir"
    nohup setsid env INHERNET_BACKGROUND_CHILD=1 BACKGROUND=0 "$0" "$@" \
        </dev/null >"$stdout_log" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    echo "Started background job: $label"
    echo "PID: $pid"
    echo "PID file: $pid_file"
    echo "Console log: $stdout_log"
    return 0
}

run_case() {
    local label="$1"
    shift
    echo
    echo "=== $label ==="
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "$RUNNER" "$@"
        printf '\n'
    else
        "$RUNNER" "$@"
    fi
}
