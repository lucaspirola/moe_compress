#!/usr/bin/env bash
# Submit the Strategy A compression pipeline to HF Jobs.
#
# Usage:
#   ./hf_jobs/submit.sh                   # default: a100-large, 0.30 target
#   FLAVOR=h200 ./hf_jobs/submit.sh       # override hardware
#   TARGET_RATIO=0.25 ./hf_jobs/submit.sh # override compression
#   DETACH=1 ./hf_jobs/submit.sh          # return immediately; check logs separately
#
# The job auto-releases the GPU when the entrypoint exits (success or failure).
# --timeout is a hard ceiling; the A100 spec says ~2.75h wall-clock so 5h is
# comfortable headroom.

set -euo pipefail

FLAVOR="${FLAVOR:-a100-large}"
TIMEOUT="${TIMEOUT:-5h}"
CODE_REPO="${CODE_REPO:-pirola/moe-compress-code}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-35B-A3B}"
RESULT_REPO="${RESULT_REPO:-}"
TARGET_RATIO="${TARGET_RATIO:-0.30}"
BUCKET="${BUCKET:-hf://buckets/pirola/moe-cache}"
MOUNT="${MOUNT:-/mnt/cache}"
ENTRYPOINT="$(cd "$(dirname "$0")" && pwd)/entrypoint.py"
DETACH="${DETACH:-0}"
RESUME_FROM_STAGE="${RESUME_FROM_STAGE:-0}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-6}"

if [[ ! -f "$ENTRYPOINT" ]]; then
    echo "entrypoint.py not found at $ENTRYPOINT" >&2
    exit 1
fi

DETACH_FLAG=""
if [[ "$DETACH" == "1" ]]; then
    DETACH_FLAG="--detach"
fi

echo ">>> Submitting Strategy A pipeline"
echo "    flavor       : $FLAVOR"
echo "    timeout      : $TIMEOUT"
echo "    code repo    : $CODE_REPO"
echo "    model repo   : $MODEL_REPO"
echo "    target ratio : $TARGET_RATIO"
echo "    resume from  : stage $RESUME_FROM_STAGE"
echo "    stop after   : stage $STOP_AFTER_STAGE"
echo "    bucket mount : $BUCKET → $MOUNT"
echo "    result repo  : ${RESULT_REPO:-<auto: pirola/qwen3-6-35b-a3b-strategy-a-<pct>pct[-stopN]-<ts>>}"
echo

exec hf jobs uv run "$ENTRYPOINT" \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --volume "$BUCKET:$MOUNT" \
    --secrets HF_TOKEN \
    --env "CODE_REPO=$CODE_REPO" \
    --env "MODEL_REPO=$MODEL_REPO" \
    --env "RESULT_REPO=$RESULT_REPO" \
    --env "TARGET_RATIO=$TARGET_RATIO" \
    --env "CACHE_MOUNT=$MOUNT" \
    --env "CONFIG_PATH=configs/qwen36_35b_a3b_30pct.yaml" \
    --env "RESUME_FROM_STAGE=$RESUME_FROM_STAGE" \
    --env "STOP_AFTER_STAGE=$STOP_AFTER_STAGE" \
    $DETACH_FLAG
