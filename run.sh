#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

BACKGROUND=0
LOG_FILE=""
PID_FILE="$PROJECT_DIR/run.pid"
PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE=""
DATASET=""
PAIR=""
METHOD=""
PRE_DASH_EXTRA=()
SHARED_ARGS=()
AFTER_DOUBLE_DASH=0

usage() {
    cat <<'USAGE'
Usage:
  ./run.sh [--background] [--log-file PATH] --dataset DATASET --pair PAIR --method METHOD [demo_code.py args...]
  ./run.sh [--background] --suite {baseline,comparison,all} --dataset DATASET --pair PAIR -- [shared demo_code.py args...]

Launcher options:
  --background           Launch the selected run in the background with nohup.
  --log-file PATH        Single-run only custom log path.
  --suite NAME           Serial preset suite: baseline, comparison, or all.
                         Suite mode always targets one dataset/pair, not every registered pair.
  --                    Separator before shared demo_code.py args in suite mode.
                         Everything after -- is forwarded directly to demo_code.py.

Command structure:
  Single run:
    Use --method when you want to train or test one specific model.
  Suite run:
    Use --suite when you want a preset collection of methods for one dataset/pair.
    Example: --suite all runs every supported method for the chosen pair.

Notes:
  Suite runs are always serial and train all requested models from scratch.
  Shared options such as --epochs, --download, and --plot-mode must come after -- in suite mode.

Examples:
  Train all models for one CIFAR-100 pair for 5 epochs:
    ./run.sh --suite all --dataset cifar100 --pair resnet56_to_resnet20 -- --epochs 5 --download

  Train only the comparison methods for one CIFAR-100 pair:
    ./run.sh --suite comparison --dataset cifar100 --pair resnet32_to_resnet8 -- --download

  Run one suite in the background with nohup-managed logs and PID reporting:
    ./run.sh --background --suite all --dataset cifar100 --pair resnet56_to_resnet20 -- --epochs 5 --download

  Run one CIFAR-100 method directly:
    ./run.sh --dataset cifar100 --pair vgg13_to_vgg8 --method hetero --download

  Quick CIFAR-10 single-model smoke test:
    ./run.sh --dataset cifar10 --pair resnet50_to_resnet18 --method teacher --smoke-test --plot-mode none

  Single run in the background with nohup-managed logging:
    ./run.sh --background --dataset cifar100 --pair resnet56_to_resnet20 --method student_kd --download

  Single run with a custom log path:
    ./run.sh --log-file logs/custom_teacher.log --dataset cifar100 --pair resnet56_to_resnet20 --method teacher --download

  If the dataset is already present locally, omit --download:
    ./run.sh --dataset cifar100 --pair resnet56_to_resnet20 --method teacher

  Use a different Python interpreter explicitly:
    PYTHON_BIN=/path/to/python ./run.sh --dataset cifar100 --pair vgg13_to_vgg8 --method inhernet --rank-preset large --download
USAGE
}

launch_background_job() {
    local pid_file="$1"
    local output_log="$2"
    shift 2
    local -a cmd=("$@")

    mkdir -p "$(dirname "$pid_file")"
    mkdir -p "$(dirname "$output_log")"

    nohup "${cmd[@]}" >>"$output_log" 2>&1 &
    local pid=$!
    echo "$pid" >"$pid_file"
    echo "PID: $pid"
    echo "PID file: $pid_file"
}

while (($#)); do
    if ((AFTER_DOUBLE_DASH)); then
        SHARED_ARGS+=("$1")
        shift
        continue
    fi

    case "$1" in
        --)
            AFTER_DOUBLE_DASH=1
            shift
            ;;
        --background)
            BACKGROUND=1
            shift
            ;;
        --log-file)
            if (($# < 2)); then
                echo "--log-file requires a path" >&2
                exit 1
            fi
            LOG_FILE="$2"
            shift 2
            ;;
        --suite)
            if (($# < 2)); then
                echo "--suite requires a value" >&2
                exit 1
            fi
            SUITE="$2"
            shift 2
            ;;
        --schedule|--max-jobs)
            echo "--schedule and --max-jobs are no longer supported; suite runs are always serial." >&2
            exit 1
            ;;
        --dataset)
            if (($# < 2)); then
                echo "--dataset requires a value" >&2
                exit 1
            fi
            DATASET="$2"
            shift 2
            ;;
        --pair)
            if (($# < 2)); then
                echo "--pair requires a value" >&2
                exit 1
            fi
            PAIR="$2"
            shift 2
            ;;
        --method)
            if (($# < 2)); then
                echo "--method requires a value" >&2
                exit 1
            fi
            METHOD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PRE_DASH_EXTRA+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$DATASET" || -z "$PAIR" ]]; then
    usage
    echo "Both --dataset and --pair are required." >&2
    exit 1
fi

if [[ -n "$SUITE" && -n "$METHOD" ]]; then
    usage
    echo "Use either --method or --suite, not both." >&2
    exit 1
fi

if [[ -z "$SUITE" && -z "$METHOD" ]]; then
    usage
    echo "One of --method or --suite is required." >&2
    exit 1
fi

if [[ -n "$SUITE" ]]; then
    if [[ -n "$LOG_FILE" ]]; then
        echo "--log-file is single-run only in suite mode. demo_code.py manages suite.log and child logs." >&2
        exit 1
    fi
    if ((${#PRE_DASH_EXTRA[@]} > 0)); then
        echo "Suite demo_code.py arguments must come after --. Unexpected arguments: ${PRE_DASH_EXTRA[*]}" >&2
        exit 1
    fi

    TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
    SUITE_DIR="$PROJECT_DIR/logs/$DATASET/$PAIR/$SUITE/$TIMESTAMP"
    SUITE_LOG_FILE="$SUITE_DIR/suite.log"
    SUITE_PID_FILE="$SUITE_DIR/suite.pid"
    mkdir -p "$SUITE_DIR"

    CMD=(
        "$PYTHON_BIN"
        -u
        "$PROJECT_DIR/demo_code.py"
        --dataset "$DATASET"
        --pair "$PAIR"
        --suite "$SUITE"
        "${SHARED_ARGS[@]}"
    )

    if ((BACKGROUND)); then
        echo "Starting serial suite in background."
        launch_background_job \
            "$SUITE_PID_FILE" \
            "$SUITE_LOG_FILE" \
            env \
            INHERNET_SUITE_LOG_DIR="$SUITE_DIR" \
            INHERNET_SUITE_BACKGROUND=1 \
            "${CMD[@]}"
        echo "Suite log directory: $SUITE_DIR"
        echo "Suite log: $SUITE_LOG_FILE"
    else
        export INHERNET_SUITE_LOG_DIR="$SUITE_DIR"
        echo "Launching serial suite: ${CMD[*]}"
        "${CMD[@]}"
        echo "Suite logs: $SUITE_DIR"
    fi
    exit 0
fi

PY_ARGS=(
    --dataset "$DATASET"
    --pair "$PAIR"
    --method "$METHOD"
    "${PRE_DASH_EXTRA[@]}"
    "${SHARED_ARGS[@]}"
)

if [[ -z "$LOG_FILE" ]]; then
    TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
    mkdir -p "$PROJECT_DIR/logs"
    LOG_FILE="$PROJECT_DIR/logs/run_${TIMESTAMP}.log"
else
    mkdir -p "$(dirname "$LOG_FILE")"
fi

CMD=(
    "$PYTHON_BIN"
    -u
    "$PROJECT_DIR/demo_code.py"
    "${PY_ARGS[@]}"
)

if ((BACKGROUND)); then
    echo "Starting in background. Logs: $LOG_FILE"
    launch_background_job \
        "$PID_FILE" \
        "$LOG_FILE" \
        env \
        INHERNET_RUN_LOG="$LOG_FILE" \
        "${CMD[@]}"
    echo "Log file: $LOG_FILE"
else
    echo "Running: ${CMD[*]}"
    echo "Log file: $LOG_FILE"
    env INHERNET_RUN_LOG="$LOG_FILE" "${CMD[@]}"
fi
