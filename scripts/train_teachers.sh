#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

teacher_group="${1:-all}"
case "$teacher_group" in all|glue|vision) ;; *)
    echo "Usage: $0 [all|glue|vision] [runtime arguments...]" >&2
    exit 2
esac
if (($# > 0)); then shift; fi

# Teacher checkpoints used for hyperparameter selection must share each
# target's validation split. Run all registered pair-bound teachers
# sequentially on one GPU, detached by default.
if [[ "${INHERNET_BACKGROUND_CHILD:-0}" != "1" && "${FOREGROUND:-0}" != "1" ]]; then
    BACKGROUND=1
fi
if maybe_launch_background teachers "$teacher_group" "$@"; then exit 0; fi

export DEVICE="${DEVICE:-cuda}"
export SEARCH_SEEDS="${SEARCH_SEEDS:-42}"
export SEARCH_ARTIFACT_SCOPE=registry
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
export TEACHER_GROUP="$teacher_group"
"$PROJECT_DIR/scripts/search.sh" teachers --download --num-workers "${NUM_WORKERS:-4}" "$@"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

if [[ "$teacher_group" != "all" ]]; then
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/audit_teachers.py" --group "$teacher_group"
    exit 0
fi

manifest_tmp="$PROJECT_DIR/.teacher_checkpoints.${BASHPID}.json"
trap 'rm -f "$manifest_tmp"' EXIT
"$PYTHON_BIN" "$PROJECT_DIR/scripts/audit_teachers.py" --json >"$manifest_tmp"
mv -f "$manifest_tmp" "$PROJECT_DIR/teacher_checkpoints.json"
trap - EXIT
