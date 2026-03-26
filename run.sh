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
  ./run.sh --suite {baseline,comparison,all} --dataset DATASET --pair PAIR -- [shared demo_code.py args...]

Launcher options:
  --background           Single-run only. Launch in the background with nohup.
  --log-file PATH        Single-run only custom log path.
  --suite NAME           Serial preset suite: baseline, comparison, or all.
                         Suite mode always targets one dataset/pair, not every registered pair.
  --                    Separator before shared demo_code.py args in suite mode.

Notes:
  Suite runs are always serial and train all requested models from scratch.
  The old --schedule and --max-jobs options are no longer supported.

Examples:
  ./run.sh --dataset cifar100 --pair resnet56_to_resnet20 --method hetero
  ./run.sh --dataset cifar10 --pair resnet50_to_resnet18 --method teacher --smoke-test
  ./run.sh --suite all --dataset cifar100 --pair resnet56_to_resnet20 -- --epochs 5 --download
  ./run.sh --suite comparison --dataset cifar100 --pair resnet32_to_resnet8 -- --download
  PYTHON_BIN=/path/to/python ./run.sh --background --dataset cifar100 --pair vgg13_to_vgg8 --method inhernet --rank-preset large
USAGE
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
    if ((BACKGROUND)); then
        echo "--background is single-run only. For suites, wrap the whole command in nohup ... & if needed." >&2
        exit 1
    fi
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
    mkdir -p "$SUITE_DIR"
    export INHERNET_SUITE_LOG_DIR="$SUITE_DIR"

    CMD=(
        "$PYTHON_BIN"
        -u
        "$PROJECT_DIR/demo_code.py"
        --dataset "$DATASET"
        --pair "$PAIR"
        --suite "$SUITE"
        "${SHARED_ARGS[@]}"
    )

    echo "Launching serial suite: ${CMD[*]}"
    "${CMD[@]}"
    echo "Suite logs: $SUITE_DIR"
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
    nohup env INHERNET_RUN_LOG="$LOG_FILE" "${CMD[@]}" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "PID saved to $PID_FILE"
else
    echo "Running: ${CMD[*]}"
    env INHERNET_RUN_LOG="$LOG_FILE" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
