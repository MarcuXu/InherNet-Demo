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

inheract_display_name() {
    case "$1" in
        large) printf 'InherAct\n' ;;
        small) printf 'InherAct-Lite\n' ;;
        *) echo "Unknown InherAct size: $1" >&2; return 2 ;;
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

formal_checkpoint_for() {
    local run_id="$1"
    local dataset="$2"
    local pair="$3"
    local seed="$4"
    printf '%s/checkpoints/formal/%s/%s/%s/teacher_seed_%s.pt\n' \
        "$PROJECT_DIR" "$run_id" "$dataset" "$pair" "$seed"
}

new_formal_run_id() {
    printf 'formal_%s\n' "$(date -u +%Y%m%d_%H%M%S_%N)"
}

validate_formal_run_id() {
    local run_id="$1"
    if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "FORMAL_RUN_ID must contain only letters, digits, '.', '_' or '-' and start with a letter or digit." >&2
        exit 2
    fi
}

reuse_compatible_teacher_checkpoint() {
    local destination="$1" dataset="$2" pair="$3" seed="$4" search_validation="$5"
    shift 5
    local source
    local -a command=(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/reuse_teacher_checkpoint.py"
        --destination "$destination"
        --dataset "$dataset"
        --pair "$pair"
        --seed "$seed"
    )
    [[ "$search_validation" == "1" ]] && command+=(--search-validation)
    [[ "${DRY_RUN:-0}" == "1" ]] && command+=(--dry-run)
    command+=("$@")
    if ! source="$("${command[@]}")"; then
        return 1
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "Would snapshot compatible teacher: $source -> $destination"
    else
        echo "Snapshotted compatible teacher: $source -> $destination"
    fi
}

validate_compatible_teacher_checkpoint() {
    local checkpoint="$1" dataset="$2" pair="$3" seed="$4" search_validation="$5"
    local -a command=(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/reuse_teacher_checkpoint.py"
        --dry-run
        --destination "$checkpoint"
        --dataset "$dataset"
        --pair "$pair"
        --seed "$seed"
    )
    [[ "$search_validation" == "1" ]] && command+=(--search-validation)
    command+=("$checkpoint")
    "${command[@]}" >/dev/null
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
            --inheract-recipe-id|--inheract-recipe-id=*|--inheritance-diagnostics-only)
                echo "Reserved experiment-matrix argument is not allowed in extras: $argument" >&2
                exit 2
                ;;
        esac
    done
}

load_inheract_recipe() {
    local dataset="$1"
    mapfile -t INHERACT_RECIPE_ARGS < <(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/inheract_recipes.py" selected "$dataset"
    )
    if ((${#INHERACT_RECIPE_ARGS[@]} == 0)); then
        echo "No selected InherAct recipe resolved for $dataset." >&2
        return 1
    fi
}

reject_search_overrides() {
    local argument
    for argument in "$@"; do
        case "$argument" in
            --epochs|--epochs=*|--lr|--lr=*|--lr-scale|--lr-scale=*|--aux-loss-weight|\
            --aux-loss-weight=*|--inheract-second-moment-shrinkage|\
            --inheract-second-moment-shrinkage=*|--inheract-expert-noise-scale|\
            --inheract-expert-noise-scale=*|--inheract-allocation-scale|\
            --inheract-allocation-scale=*|--kd-temperature|--kd-temperature=*|\
            --freeze-inheract-router|\
            --kd-weight|--kd-weight=*|--ce-weight|--ce-weight=*|--kd-fraction|\
            --kd-fraction=*|--optimizer|--optimizer=*|\
            --batch-size|--batch-size=*|--momentum|--momentum=*|--weight-decay|\
            --weight-decay=*|--rank|--rank=*|--head-num|--head-num=*|\
            --max-calib-batches|--max-calib-batches=*|--inheract-max-features-per-batch|\
            --inheract-max-features-per-batch=*|--final-test|--no-final-test)
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
            --max-calib-batches|--max-calib-batches=*|--inheract-max-features-per-batch|\
            --inheract-max-features-per-batch=*|--inheract-second-moment-shrinkage|\
            --inheract-second-moment-shrinkage=*|--inheract-expert-noise-scale|\
            --inheract-expert-noise-scale=*|--inheract-allocation-scale|\
            --inheract-allocation-scale=*|--freeze-inheract-router|\
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
