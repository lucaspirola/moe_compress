#!/usr/bin/env bash
# Replay-path smoke: the go/no-go gate before the full data-parallel run.
#
# Loads the model with cudagraphs ON + moe_backend=triton and runs the ACTUAL
# --replay-from path over ~N rows of a shard (max_tokens=1), then loads the
# produced reap_scores.pt and asserts:
#   (a) token_counts.sum() > 0            (kernel-interior reap fired)
#   (b) a traced-region signal fired      (block_outputs sidecar present + >0)
#
# This matches the real run (the same fixed-wheel code path), not generic
# generation, so a green smoke is the gate on the patched wheel before
# renting the full N-GPU box. See tasks/CALIB_PARALLEL_CAPTURE_DESIGN.md §smoke.
#
# Usage:
#   replay_smoke.sh \
#       --jsonl <a shard or any v9 self-traces JSONL> \
#       [--rows 8] \
#       [--max-model-len 20480] \
#       [--teacher Qwen/Qwen3.6-35B-A3B] \
#       [--py python3] \
#       [--extra-args "--gpu-memory-utilization 0.90"]

set -euo pipefail

JSONL=""
ROWS=8
MAX_MODEL_LEN=20480
TEACHER="Qwen/Qwen3.6-35B-A3B"
PY="python3"
EXTRA_ARGS=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../src" && pwd)"
DRIVER="${SCRIPT_DIR}/build_self_traces_calib_vllm.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jsonl) JSONL="$2"; shift 2 ;;
        --rows) ROWS="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --teacher) TEACHER="$2"; shift 2 ;;
        --py) PY="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${JSONL}" || ! -f "${JSONL}" ]]; then
    echo "ERROR: --jsonl required and must exist (got '${JSONL}')" >&2
    exit 2
fi

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# Build a tiny ~ROWS-row smoke shard.
SMOKE_DIR="$(dirname "${JSONL}")/replay_smoke"
mkdir -p "${SMOKE_DIR}"
SMOKE_JSONL="${SMOKE_DIR}/smoke.jsonl"
head -n "${ROWS}" "${JSONL}" > "${SMOKE_JSONL}"
N_ACTUAL="$(wc -l < "${SMOKE_JSONL}")"
echo "smoke: replaying ${N_ACTUAL} rows from ${JSONL} -> ${SMOKE_JSONL}"

# Clean any prior smoke sidecars so the assertion sees fresh output.
rm -rf "${SMOKE_DIR}/sidecars"

echo "--- running ACTUAL --replay-from path (cudagraphs ON, moe_backend=triton) ---"
"${PY}" "${DRIVER}" \
    --replay-from "${SMOKE_JSONL}" \
    --teacher "${TEACHER}" \
    --moe-backend triton \
    --max-model-len "${MAX_MODEL_LEN}" \
    --capture-reap-scores \
    --capture-block-outputs \
    ${EXTRA_ARGS}

echo "--- assert reap token_counts > 0 AND a traced-region (block_out) signal fired ---"
"${PY}" - "${SMOKE_JSONL}" <<'PYEOF'
import sys
from pathlib import Path
import torch
from moe_compress.utils.cached_calibration_signals import (
    load_reap_scores, load_block_hidden,
)

smoke_jsonl = Path(sys.argv[1])

# (a) reap kernel-interior signal.
reap = load_reap_scores(smoke_jsonl)
assert reap is not None, "GO/NO-GO FAIL: no reap_scores.pt produced"
tc = int(reap.token_counts.sum().item())
print(f"reap token_counts.sum() = {tc}")
assert tc > 0, "GO/NO-GO FAIL: reap token_counts.sum() == 0 (capture empty)"

# (b) traced-region signal (block_outputs -> per-layer block_hidden shards).
sidecar_dir = smoke_jsonl.parent / "sidecars" / smoke_jsonl.stem / "block_hidden"
assert sidecar_dir.is_dir(), (
    f"GO/NO-GO FAIL: no block_hidden dir at {sidecar_dir} "
    "(block_outputs traced-region signal did not fire)")
layers = sorted(sidecar_dir.glob("layer_*.pt"))
assert layers, "GO/NO-GO FAIL: no block_hidden layer shards written"
bp = load_block_hidden(smoke_jsonl, int(layers[0].stem.split("_")[-1]))
assert bp is not None, "GO/NO-GO FAIL: block_hidden shard unreadable"
ptr = int(bp.hidden_states.shape[0])
print(f"block_hidden layer0 hidden_states rows = {ptr}")
assert ptr > 0, "GO/NO-GO FAIL: block_hidden hidden_states empty"

print("GO: reap + block_outputs both fired in the replay path. "
      "Cleared for the full N-GPU run.")
PYEOF

echo "=== replay smoke: GO ==="
