#!/usr/bin/env bash
# Submit the Stage 2 v2 ablation harness to HF Jobs.
#
# Pipeline shape per ablation: Stage 1 (shared once) → Stage 2 → Stage 2.5 → Stage 6.
# 12 ablations sequential in one job; resume-safe across job timeouts.
#
# Usage:
#   ./hf_jobs/submit_ablations.sh                  # default: a100-large, 48h, all 12
#   ONLY=A0 ./hf_jobs/submit_ablations.sh          # run a subset (debug / smoke)
#   FLAVOR=h200 ./hf_jobs/submit_ablations.sh      # pin H200 if available
#   NUM_SEQUENCES=2000 ./hf_jobs/submit_ablations.sh   # bigger calibration
#   DETACH=1 ./hf_jobs/submit_ablations.sh         # return immediately
set -euo pipefail

FLAVOR="${FLAVOR:-a100-large}"
TIMEOUT="${TIMEOUT:-48h}"
CODE_REPO="${CODE_REPO:-pirola/moe-compress}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-35B-A3B}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
ONLY="${ONLY:-}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
BUCKET="${BUCKET:-hf://buckets/pirola/moe-ablations}"
MOUNT="${MOUNT:-/mnt/cache}"
ENTRYPOINT="$(cd "$(dirname "$0")" && pwd)/entrypoint_ablations.py"
DETACH="${DETACH:-0}"

if [[ ! -f "$ENTRYPOINT" ]]; then
    echo "entrypoint_ablations.py not found at $ENTRYPOINT" >&2
    exit 1
fi

DETACH_FLAG=""
if [[ "$DETACH" == "1" ]]; then
    DETACH_FLAG="--detach"
fi

echo ">>> Submitting Stage 2 v2 ablation matrix"
echo "    flavor         : $FLAVOR"
echo "    timeout        : $TIMEOUT"
echo "    code repo      : $CODE_REPO"
echo "    model repo     : $MODEL_REPO"
echo "    num sequences  : $NUM_SEQUENCES (calibration size for ablations)"
echo "    only           : ${ONLY:-<all 12: A0..A11>}"
echo "    preflight-only : $PREFLIGHT_ONLY (1 = run Stage 1 then exit)"
echo "    bucket mount   : $BUCKET → $MOUNT"
echo "    Trackio URL    : https://huggingface.co/spaces/pirola/trackio"
echo

exec hf jobs uv run "$ENTRYPOINT" \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --volume "$BUCKET:$MOUNT" \
    --secrets HF_TOKEN \
    --env "CODE_REPO=$CODE_REPO" \
    --env "MODEL_REPO=$MODEL_REPO" \
    --env "NUM_SEQUENCES=$NUM_SEQUENCES" \
    --env "ONLY=$ONLY" \
    --env "PREFLIGHT_ONLY=$PREFLIGHT_ONLY" \
    --env "CACHE_MOUNT=$MOUNT" \
    --env "PYTORCH_ALLOC_CONF=expandable_segments:True" \
    $DETACH_FLAG
