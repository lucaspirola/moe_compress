#!/usr/bin/env bash
# Hardened launcher for the vLLM self-traces calibration driver.
#
# Wraps `build_self_traces_calib_vllm.py` with the host-level fixes that the
# bare driver cannot do for itself (system packages, package surgery, durable
# compile caches). Every fix here was learned running the driver on fresh
# cloud GPU boxes; see docs/calibration_vllm_runbook.md for root causes.
#
# Idempotent and re-runnable: safe to invoke again after a spot preemption —
# the driver's --resume continues from the last completed chunk, and the
# persisted compile caches make restart fast (~6-8 min to resume generation).
#
# Tunables via env (all have sensible defaults):
#   DATA_ROOT     base dir on durable storage         (default /mnt/data)
#   REPO          repo checkout root                  (default $DATA_ROOT/moe_compress)
#   VENV          python venv to activate             (default $DATA_ROOT/gpu_venv)
#   OUTPUT        self_traces.jsonl path              (default $DATA_ROOT/artifacts/_shared/self_traces.jsonl)
#   TEACHER       teacher repo id                     (default Qwen/Qwen3.6-35B-A3B)
#   PROMPTS       mix name or JSONL path              (default qwen3-pretrain-mix-v2)
#   NUM_PROMPTS   raw prompts (pre-completeness filt) (default 8000)
#   DTYPE         bfloat16|float16|auto|fp8           (default bfloat16)
#   CALIB_KEEP_KERNELS=1  keep the `kernels` pkg (needed for an FP8 teacher)
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/data}"
REPO="${REPO:-$DATA_ROOT/moe_compress}"
VENV="${VENV:-$DATA_ROOT/gpu_venv}"
OUTPUT="${OUTPUT:-$DATA_ROOT/artifacts/_shared/self_traces.jsonl}"
TEACHER="${TEACHER:-Qwen/Qwen3.6-35B-A3B}"
PROMPTS="${PROMPTS:-qwen3-pretrain-mix-v2}"
NUM_PROMPTS="${NUM_PROMPTS:-8000}"
DTYPE="${DTYPE:-bfloat16}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- Fix 1: host build prerequisite -----------------------------------------
# The first forward pass JIT-compiles the FlashInfer GDN prefill kernel, which
# needs Python.h (cc fails late with "Python.h: No such file or directory" on
# images that ship python venv but not the dev headers).
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq python3.12-dev build-essential >/dev/null 2>&1 || true

# --- Fix 2: compile-OOM cap + durable compile caches ------------------------
# (The driver also setdefault()s these; we set them here so the env is correct
# even if the launcher is used with a different entrypoint.)
export MAX_JOBS="${MAX_JOBS:-16}"          # cap parallel cicc; ~6 GB each
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$DATA_ROOT/vllm_cache}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$DATA_ROOT/flashinfer_cache}"
mkdir -p "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_DIR"
# FlashInfer ignores the env var above and hardcodes ~/.cache/flashinfer, so
# symlink it onto durable storage to persist the GDN compile across restarts.
rm -rf ~/.cache/flashinfer; mkdir -p ~/.cache
ln -sfn "$FLASHINFER_WORKSPACE_DIR" ~/.cache/flashinfer

# --- venv -------------------------------------------------------------------
# shellcheck disable=SC1091
. "$VENV/bin/activate"

# --- Fix 3: kernels package surgery -----------------------------------------
# `kernels` is required ONLY for an FP8 KD teacher. With a non-FP8 teacher and
# recent transformers it makes `from vllm import LLM` raise inside transformers'
# hub-kernels import. Remove it unless explicitly kept / using fp8.
if [ "${CALIB_KEEP_KERNELS:-0}" != "1" ] && [ "$DTYPE" != "fp8" ]; then
    pip uninstall -y kernels >/dev/null 2>&1 || true
fi

cd "$REPO"
export PYTHONPATH="$REPO/max_quality/src:${PYTHONPATH:-}"
mkdir -p "$(dirname "$OUTPUT")"

log "launching calibration: prompts=$PROMPTS num=$NUM_PROMPTS dtype=$DTYPE -> $OUTPUT"
python max_quality/scripts/build_self_traces_calib_vllm.py \
    --teacher "$TEACHER" \
    --prompts "$PROMPTS" \
    --num-prompts "$NUM_PROMPTS" \
    --seed 1337 \
    --max-new-tokens 16384 \
    --reasoning-budget 4096 \
    --max-model-len 20480 \
    --dtype "$DTYPE" \
    --gpu-memory-utilization 0.90 \
    --logits-top-k 50 \
    --output "$OUTPUT" \
    --resume \
    --capture-imatrix --capture-reap-scores --capture-input-covariance \
    --capture-wanda-scalar-row --capture-stage2-profile --capture-per-expert-max \
    --capture-routing-stats --capture-router-logits-stats \
    --capture-layer-input-reservoir --capture-output-reservoir \
    --capture-block-outputs
rc=$?
log "RUN EXIT rc=$rc"
exit $rc
