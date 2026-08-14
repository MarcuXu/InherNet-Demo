#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

group="${1:-all}"
case "$group" in
    all|vision|cifar100|glue) ;;
    *)
        echo "Usage: $0 [all|vision|cifar100|glue] [permitted runtime arguments...]" >&2
        exit 2
        ;;
esac
if (($# > 0)); then shift; fi
extra=("$@")

vision_targets=(
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
)
cifar100_targets=(
    "cifar100 resnet32_to_resnet8"
    "cifar100 resnet32x4_to_resnet8x4"
    "cifar100 vgg13_to_vgg8"
    "cifar100 wrn40_2_to_wrn40_1"
    "cifar100 wrn40_2_to_wrn16_2"
    "cifar100 resnet56_to_resnet20"
    "cifar100 resnet110_to_resnet32"
    "cifar100 resnet110_to_resnet20"
)
glue_targets=(
    "glue_mrpc bert4_to_bert2"
    "glue_qqp bert4_to_bert2"
    "glue_sst2 bert4_to_bert2"
    "glue_mnli bert4_to_bert2"
    "glue_rte bert4_to_bert2"
    "glue_qnli bert4_to_bert2"
    "glue_cola bert4_to_bert2"
    "glue_stsb bert4_to_bert2"
)

case "$group" in
    all) targets=("${vision_targets[@]}" "${glue_targets[@]}") ;;
    vision) targets=("${vision_targets[@]}") ;;
    cifar100) targets=("${cifar100_targets[@]}") ;;
    glue) targets=("${glue_targets[@]}") ;;
esac

if [[ "${OVERWRITE_TEACHER:-0}" == "1" ]]; then
    echo "Formal runs never overwrite teachers. Start a new formal run namespace instead." >&2
    exit 2
fi
if [[ "${RESUME:-0}" == "1" && -z "${FORMAL_RUN_ID:-}" ]]; then
    echo "RESUME=1 requires FORMAL_RUN_ID=<existing formal run namespace>." >&2
    exit 2
fi
export RESUME="${RESUME:-0}"
export FORMAL_RUN_ID="${FORMAL_RUN_ID:-$(new_formal_run_id)}"
validate_formal_run_id "$FORMAL_RUN_ID"

default_background
if maybe_launch_background formal_all "$group" "${extra[@]}"; then
    echo "Formal run namespace: $FORMAL_RUN_ID"
    exit 0
fi

echo "Formal run namespace: $FORMAL_RUN_ID"

for target in "${targets[@]}"; do
    read -r dataset pair <<<"$target"
    echo
    echo "######## Formal target: $dataset / $pair ########"
    FOREGROUND=1 BACKGROUND=0 "$PROJECT_DIR/scripts/formal.sh" \
        "$dataset" "$pair" "${extra[@]}"
done
