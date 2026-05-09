#!/usr/bin/env bash
# Submit the Phase A MA-formation detector diagnostic to HF Jobs.
#
# Reuses the same bucket as the ablation runs so the diagnostic can pick up
# the production ablation_config.yaml and the resident model snapshot
# (no model re-download needed).
#
# Wall time on H200: ~5 min model load + ~3 min Phase A = ~8 min total.
# Cost: ~$0.70 on h200 ($5/h × 8 min) — vs $25+ to re-run full Stage 1.
#
# Usage:
#   ./hf_jobs/submit_phase_a_diagnostic.sh                    # default: h200, 1h, 256 samples
#   FLAVOR=a100-large ./hf_jobs/submit_phase_a_diagnostic.sh  # if h200 queue stalled
#   NUM_SAMPLES=128 ./hf_jobs/submit_phase_a_diagnostic.sh    # smaller sample set
#   DETACH=1 ./hf_jobs/submit_phase_a_diagnostic.sh           # return immediately
set -euo pipefail

FLAVOR="${FLAVOR:-h200}"
TIMEOUT="${TIMEOUT:-1h}"
CODE_REPO="${CODE_REPO:-pirola/moe-compress}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-35B-A3B}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
BUCKET="${BUCKET:-hf://buckets/pirola/moe-ablations}"
MOUNT="${MOUNT:-/mnt/cache}"
ENTRYPOINT="$(cd "$(dirname "$0")" && pwd)/entrypoint_phase_a_diagnostic.py"
DETACH="${DETACH:-0}"

if [[ ! -f "$ENTRYPOINT" ]]; then
    echo "entrypoint_phase_a_diagnostic.py not found at $ENTRYPOINT" >&2
    exit 1
fi

DETACH_FLAG=""
if [[ "$DETACH" == "1" ]]; then
    DETACH_FLAG="--detach"
fi

echo ">>> Submitting Phase A diagnostic"
echo "    flavor       : $FLAVOR"
echo "    timeout      : $TIMEOUT"
echo "    code repo    : $CODE_REPO"
echo "    model repo   : $MODEL_REPO"
echo "    num samples  : $NUM_SAMPLES (Phase A only — 256 is plenty for layer_max)"
echo "    bucket mount : $BUCKET → $MOUNT"
echo "    output       : $MOUNT/diagnostics/phase_a_<timestamp>.json"
echo

exec hf jobs uv run "$ENTRYPOINT" \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --volume "$BUCKET:$MOUNT" \
    --secrets HF_TOKEN \
    --env "CODE_REPO=$CODE_REPO" \
    --env "MODEL_REPO=$MODEL_REPO" \
    --env "NUM_SAMPLES=$NUM_SAMPLES" \
    --env "CACHE_MOUNT=$MOUNT" \
    --env "PYTORCH_ALLOC_CONF=expandable_segments:True" \
    $DETACH_FLAG
