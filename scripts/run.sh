#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "$PROJECT_DIR"

if [[ -z "${INHERNET_RUN_LOG:-}" ]]; then
    timestamp="$(date -u +%Y%m%d_%H%M%S_%N)"
    mkdir -p "$PROJECT_DIR/logs"
    export INHERNET_RUN_LOG="$PROJECT_DIR/logs/run_${timestamp}.log"
fi

echo "Running: $PYTHON_BIN -u $PROJECT_DIR/demo_code.py $*"
echo "Log file: $INHERNET_RUN_LOG"
exec "$PYTHON_BIN" -u "$PROJECT_DIR/demo_code.py" "$@"
