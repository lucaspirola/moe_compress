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
# HF Jobs default timeout is 30 min — far too short for any heavy stage. We
# size for one heavy stage with headroom; combined-stage runs need an override.
# Stage 2 alone ≈ 5 h, Stage 3 alone ≈ 1–2 h, Stage 5 KD ≈ 3–4 h.
TIMEOUT="${TIMEOUT:-8h}"
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
PRIOR_STAGE_REPO="${PRIOR_STAGE_REPO:-}"

if [[ ! -f "$ENTRYPOINT" ]]; then
    echo "entrypoint.py not found at $ENTRYPOINT" >&2
    exit 1
fi

# Opt-in preflight: run the load_compressed_model regression suite locally
# before spending compute. Catches in <2s the bug classes we burned hours on
# (state_dict pinning, dtype/shape silent corruption, _grad leakage). Pass
# PRECHECK=1 explicitly — keeps the default path zero-friction.
if [[ "${PRECHECK:-0}" == "1" ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    PYBIN="${PYBIN:-/home/lucas/ai/venv/bin/python}"
    if [[ -x "$PYBIN" ]]; then
        echo ">>> Preflight: tests/test_load_compressed_model.py"
        ( cd "$REPO_ROOT" && "$PYBIN" -m pytest tests/test_load_compressed_model.py -q ) \
            || { echo "Preflight FAILED — refusing to submit" >&2; exit 1; }
        echo
    else
        echo "PRECHECK=1 set but $PYBIN not executable; skipping preflight" >&2
    fi
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
echo "    prior repo   : ${PRIOR_STAGE_REPO:-<none>}"
echo "    bucket mount : $BUCKET → $MOUNT"
echo "    result repo  : ${RESULT_REPO:-<auto: pirola/qwen3-6-35b-a3b-strategy-a-<pct>pct[-stopN]-<ts>>}"
echo

if (( STOP_AFTER_STAGE > RESUME_FROM_STAGE )); then
    echo "Note: combining stages $RESUME_FROM_STAGE..$STOP_AFTER_STAGE in one job."
    echo "      Per-stage Hub upload (run_pipeline.py) makes intermediate stages"
    echo "      durable as soon as they complete, but cancelling mid-stage still"
    echo "      forfeits work in the current stage. Prefer one stage per job."
    echo "      See docs/huggingface_jobs_and_buckets.md."
    echo
fi


TIMEOUT_FLAG=()
if [[ -n "$TIMEOUT" ]]; then
    TIMEOUT_FLAG=(--timeout "$TIMEOUT")
fi

exec hf jobs uv run "$ENTRYPOINT" \
    --flavor "$FLAVOR" \
    "${TIMEOUT_FLAG[@]}" \
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
    --env "PRIOR_STAGE_REPO=$PRIOR_STAGE_REPO" \
    --env "PYTORCH_ALLOC_CONF=expandable_segments:True" \
    $DETACH_FLAG
