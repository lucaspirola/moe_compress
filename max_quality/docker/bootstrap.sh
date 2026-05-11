#!/usr/bin/env bash
# Container ENTRYPOINT for the moe_compress portable image.
#
# Mirrors hf_jobs/entrypoint_ablations.py modulo HF-Jobs-specific bits:
# clones code from GitHub instead of HF dataset, uses a host-mounted
# /cache volume instead of HF Bucket auto-mount.
#
# Usage:
#   docker run --gpus all \
#     -v /workspace/cache:/cache \
#     -e HF_TOKEN=hf_... \
#     [-e ONLY=A0] \
#     [-e PREFLIGHT_ONLY=1] \
#     [-e UPLOAD_ON_SUCCESS=1] \
#     ghcr.io/lucaspirola/moe-compress:latest
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (env-driven; all have defaults except HF_TOKEN)
# ---------------------------------------------------------------------------
: "${HF_TOKEN:?HF_TOKEN is required (set via -e HF_TOKEN=...)}"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-35B-A3B}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
ONLY="${ONLY:-}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen36_35b_a3b_30pct.yaml}"
CODE_REPO_URL="${CODE_REPO_URL:-https://github.com/lucaspirola/moe_compress.git}"
CODE_REF="${CODE_REF:-main}"
CACHE_MOUNT="${CACHE_MOUNT:-/cache}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-pirola/trackio}"
HF_ARTIFACTS_BUCKET="${HF_ARTIFACTS_BUCKET:-pirola/moe-ablations}"
UPLOAD_ON_SUCCESS="${UPLOAD_ON_SUCCESS:-0}"
DESTROY_HINT="${DESTROY_HINT:-vastai destroy instance \$VAST_CONTAINERLABEL}"

export HF_HOME="${HF_HOME:-$CACHE_MOUNT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TRACKIO_SPACE_ID
# Forwarded to entrypoint of run_ablations.py; the env var is read by the
# harness via os.environ in run_ablations.main() if/when used.

log() { printf '[bootstrap] %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# 1. Sanity
# ---------------------------------------------------------------------------
log "================================================================"
log " moe_compress portable runtime"
log "================================================================"
log "MODEL_REPO        = $MODEL_REPO"
log "CACHE_MOUNT       = $CACHE_MOUNT"
log "HF_HOME           = $HF_HOME"
log "CODE_REF          = $CODE_REF"
log "NUM_SEQUENCES     = $NUM_SEQUENCES"
log "ONLY              = ${ONLY:-(all 12)}"
log "PREFLIGHT_ONLY    = $PREFLIGHT_ONLY"
log "UPLOAD_ON_SUCCESS = $UPLOAD_ON_SUCCESS"
log "TRACKIO_SPACE_ID  = $TRACKIO_SPACE_ID"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "FATAL: nvidia-smi not on PATH — was the container started with --gpus all?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ---------------------------------------------------------------------------
# 2. Cache layout
# ---------------------------------------------------------------------------
mkdir -p "$CACHE_MOUNT/hf" "$CACHE_MOUNT/ablations" "$CACHE_MOUNT/code"

# ---------------------------------------------------------------------------
# 3. Code clone (or pull if already on the persistent volume)
# ---------------------------------------------------------------------------
CODE_DIR="$CACHE_MOUNT/code/moe_compress"
if [[ -d "$CODE_DIR/.git" ]]; then
    log "Code repo already cloned at $CODE_DIR — fetching $CODE_REF"
    git -C "$CODE_DIR" fetch --depth=1 origin "$CODE_REF"
    git -C "$CODE_DIR" checkout "$CODE_REF"
    git -C "$CODE_DIR" reset --hard "origin/$CODE_REF"
else
    log "Cloning $CODE_REPO_URL@$CODE_REF → $CODE_DIR"
    git clone --depth=1 --branch "$CODE_REF" "$CODE_REPO_URL" "$CODE_DIR"
fi
HEAD_SHA="$(git -C "$CODE_DIR" rev-parse --short HEAD)"
log "Code HEAD = $HEAD_SHA"

# ---------------------------------------------------------------------------
# 4. HF auth — no explicit login required.
# `huggingface_hub` (used by snapshot_download below, by trackio, and by the
# Stage 1/2 upload paths) reads $HF_TOKEN directly from the environment on every
# API call. We had a `huggingface-cli login` step here previously, but newer
# huggingface_hub releases (1.x) renamed the CLI to `hf` and the legacy
# `huggingface-cli login` is a deprecated no-op — it printed a warning and
# returned non-zero, which `set -e` then killed bootstrap with. `HF_TOKEN`
# was already in env from the docker `-e HF_TOKEN=...` flag, so the explicit
# login was redundant. Dropping it keeps the script forward-compatible.
# ---------------------------------------------------------------------------
log "HF_TOKEN in env — huggingface_hub will read it from \$HF_TOKEN at each call"

# ---------------------------------------------------------------------------
# 5. Model snapshot prefetch (skipped automatically if already cached)
# ---------------------------------------------------------------------------
log "Prefetching model snapshot $MODEL_REPO → $HF_HOME/hub"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', cache_dir='$HF_HOME/hub', allow_patterns=['*'])
print('snapshot_download complete')
"

# ---------------------------------------------------------------------------
# 6. Invoke harness
# ---------------------------------------------------------------------------
HARNESS_DIR="$CODE_DIR/max_quality"
log "Invoking run_ablations from $HARNESS_DIR"

ARGS=(
    "--config" "$HARNESS_DIR/$CONFIG_PATH"
    "--model" "$MODEL_REPO"
    "--ablations-root" "$CACHE_MOUNT/ablations"
    "--num-sequences" "$NUM_SEQUENCES"
)
if [[ -n "$ONLY" ]]; then
    ARGS+=("--only" "$ONLY")
fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    ARGS+=("--preflight-only")
fi

log "Command: python -m moe_compress.run_ablations ${ARGS[*]}"
cd "$HARNESS_DIR"
PYTHONPATH="$HARNESS_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    python -m moe_compress.run_ablations "${ARGS[@]}"
HARNESS_RC=$?

if [[ $HARNESS_RC -ne 0 ]]; then
    log "Harness exited non-zero: $HARNESS_RC"
    exit $HARNESS_RC
fi

# ---------------------------------------------------------------------------
# 7. Optional artifact upload
# ---------------------------------------------------------------------------
if [[ "$UPLOAD_ON_SUCCESS" == "1" ]]; then
    log "Uploading artifacts to bucket $HF_ARTIFACTS_BUCKET"
    hf upload --repo-type bucket "$HF_ARTIFACTS_BUCKET" "$CACHE_MOUNT/ablations" \
        --include "_shared/**" \
        --include "*/stage6_eval.json" \
        --include "_summary.json" \
        || log "WARNING: artifact upload failed (non-fatal; artifacts remain on $CACHE_MOUNT)"
fi

# ---------------------------------------------------------------------------
# 8. Final hint
# ---------------------------------------------------------------------------
log "================================================================"
log " >>> RUN COMPLETE"
log " >>> destroy instance with: $DESTROY_HINT"
log "================================================================"
