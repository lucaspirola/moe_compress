#!/usr/bin/env bash
# Data-parallel calibration capture orchestration.
#
# Runs the in-graph capture over a self-traces JSONL as N independent
# single-GPU replay processes (one per GPU), each over a disjoint shard, then
# merges the N partial sidecars into one identical to a single full run.
# See tasks/CALIB_PARALLEL_CAPTURE_DESIGN.md.
#
# Pipeline:
#   1. shard_split.py  -> shard_0..N-1  (+ disjoint/complete HARD assert)
#   2. launch N procs  -> CUDA_VISIBLE_DEVICES=k ... --replay-from shard_k ...
#   3. wait for all N  -> poll each shard's reap_scores.pt + .done marker
#   4. merge_sidecars  -> into the canonical 8000-jsonl sidecar dir
#   5. verify merged reap_scores (shape, non-zero, token_counts sum)
#
# Robust to one shard failing: reports which shard(s) failed and aborts the
# merge (a partial merge would silently undercount).
#
# Usage:
#   run_parallel_capture.sh \
#       --jsonl artifacts/_shared/self_traces.jsonl \
#       --n 4 \
#       [--captures "--capture-reap-scores --capture-per-expert-max"] \
#       [--max-model-len 20480] \
#       [--teacher Qwen/Qwen3.6-35B-A3B] \
#       [--out-dir <dir for shards>] \
#       [--py python3] \
#       [--extra-args "--gpu-memory-utilization 0.90"]
#
# Env knobs (override defaults):
#   PARALLEL_CAPTURE_POLL_SECS   poll interval while waiting (default 30)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults + arg parsing
# ---------------------------------------------------------------------------
JSONL=""
N=4
CAPTURES="--capture-reap-scores"
MAX_MODEL_LEN=20480
TEACHER="Qwen/Qwen3.6-35B-A3B"
OUT_DIR=""
PY="python3"
EXTRA_ARGS=""
MOE_BACKEND="triton"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../src" && pwd)"
DRIVER="${SCRIPT_DIR}/build_self_traces_calib_vllm.py"
SHARD_SPLIT="${SCRIPT_DIR}/shard_split.py"

POLL_SECS="${PARALLEL_CAPTURE_POLL_SECS:-30}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jsonl) JSONL="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --captures) CAPTURES="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --teacher) TEACHER="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --py) PY="$2"; shift 2 ;;
        --moe-backend) MOE_BACKEND="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${JSONL}" ]]; then
    echo "ERROR: --jsonl is required" >&2
    exit 2
fi
if [[ ! -f "${JSONL}" ]]; then
    echo "ERROR: --jsonl not found: ${JSONL}" >&2
    exit 2
fi
if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="$(dirname "${JSONL}")/parallel_capture"
fi

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

echo "=== parallel capture ==="
echo "  jsonl         = ${JSONL}"
echo "  n_shards      = ${N}"
echo "  captures      = ${CAPTURES}"
echo "  moe-backend   = ${MOE_BACKEND}"
echo "  max-model-len = ${MAX_MODEL_LEN}"
echo "  teacher       = ${TEACHER}"
echo "  out-dir       = ${OUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Shard split (HARD disjoint/complete verify inside).
# ---------------------------------------------------------------------------
echo "--- step 1: shard_split ---"
"${PY}" "${SHARD_SPLIT}" "${JSONL}" "${N}" --out-dir "${OUT_DIR}"

# ---------------------------------------------------------------------------
# 2. Launch N replay processes (one per GPU).
# ---------------------------------------------------------------------------
echo "--- step 2: launch ${N} replay processes ---"
PIDS=()
for ((k = 0; k < N; k++)); do
    SHARD_JSONL="${OUT_DIR}/shard_${k}/shard_${k}.jsonl"
    LOG="${OUT_DIR}/shard_${k}/replay.log"
    DONE="${OUT_DIR}/shard_${k}/.done"
    rm -f "${DONE}"
    if [[ ! -f "${SHARD_JSONL}" ]]; then
        echo "ERROR: shard file missing: ${SHARD_JSONL}" >&2
        exit 1
    fi
    echo "  shard ${k}: CUDA_VISIBLE_DEVICES=${k} -> ${SHARD_JSONL}"
    (
        if CUDA_VISIBLE_DEVICES="${k}" "${PY}" "${DRIVER}" \
            --replay-from "${SHARD_JSONL}" \
            --teacher "${TEACHER}" \
            --moe-backend "${MOE_BACKEND}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            ${CAPTURES} \
            ${EXTRA_ARGS} > "${LOG}" 2>&1; then
            touch "${DONE}"
        else
            echo "FAILED" > "${DONE}.failed"
        fi
    ) &
    PIDS+=($!)
done

# ---------------------------------------------------------------------------
# 3. Wait for all N (poll for each shard's reap_scores.pt + .done marker).
# ---------------------------------------------------------------------------
echo "--- step 3: wait for ${N} shards (poll every ${POLL_SECS}s) ---"
FAILED_SHARDS=()
for ((k = 0; k < N; k++)); do
    wait "${PIDS[$k]}" || true
done

for ((k = 0; k < N; k++)); do
    SHARD_DIR="${OUT_DIR}/shard_${k}"
    REAP_PT="${SHARD_DIR}/sidecars/shard_${k}/reap_scores.pt"
    if [[ -f "${SHARD_DIR}/.done.failed" ]]; then
        echo "  shard ${k}: FAILED (see ${SHARD_DIR}/replay.log)" >&2
        FAILED_SHARDS+=("${k}")
    elif [[ ! -f "${SHARD_DIR}/.done" ]]; then
        echo "  shard ${k}: NO done marker (crashed?) -> ${SHARD_DIR}/replay.log" >&2
        FAILED_SHARDS+=("${k}")
    elif [[ "${CAPTURES}" == *"--capture-reap-scores"* && ! -f "${REAP_PT}" ]]; then
        echo "  shard ${k}: reap_scores.pt MISSING (${REAP_PT})" >&2
        FAILED_SHARDS+=("${k}")
    else
        echo "  shard ${k}: OK"
    fi
done

if [[ ${#FAILED_SHARDS[@]} -gt 0 ]]; then
    echo "ERROR: ${#FAILED_SHARDS[@]} shard(s) failed: ${FAILED_SHARDS[*]}" >&2
    echo "Aborting merge (a partial merge would undercount)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Merge into the canonical 8000-jsonl sidecar dir.
# ---------------------------------------------------------------------------
echo "--- step 4: merge_sidecars -> canonical ${JSONL} ---"
SHARD_SIDECAR_DIRS=()
for ((k = 0; k < N; k++)); do
    SHARD_SIDECAR_DIRS+=("${OUT_DIR}/shard_${k}/sidecars/shard_${k}")
done
"${PY}" -m moe_compress.utils.merge_sidecars \
    --out-jsonl "${JSONL}" \
    "${SHARD_SIDECAR_DIRS[@]}"

# ---------------------------------------------------------------------------
# 5. Verify merged reap_scores (load, token_counts sum, shape, non-zero).
# ---------------------------------------------------------------------------
if [[ "${CAPTURES}" == *"--capture-reap-scores"* ]]; then
    echo "--- step 5: verify merged reap_scores ---"
    "${PY}" - "${JSONL}" "${OUT_DIR}" "${N}" <<'PYEOF'
import sys
from pathlib import Path
import torch
from moe_compress.utils.cached_calibration_signals import load_reap_scores

out_jsonl = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
n = int(sys.argv[3])

merged = load_reap_scores(out_jsonl)
assert merged is not None, "merged reap_scores not found"
shape = tuple(merged.reap_scores.shape)
merged_total = int(merged.token_counts.sum().item())
print(f"merged reap_scores shape={shape} token_counts.sum()={merged_total}")

# Sum of per-shard token_counts must equal merged total (conservation).
def _jsonl_for(k):
    return out_dir / f"shard_{k}" / f"shard_{k}.jsonl"

shard_total = 0
for k in range(n):
    p = load_reap_scores(_jsonl_for(k))
    assert p is not None, f"shard {k} reap_scores missing"
    shard_total += int(p.token_counts.sum().item())

assert merged_total == shard_total, (
    f"token_counts mismatch: merged={merged_total} != "
    f"sum(shards)={shard_total}")
assert merged_total > 0, "merged token_counts.sum() == 0 (empty capture!)"
assert merged.reap_scores.abs().sum().item() > 0, "merged reap all-zero!"
print(f"VERIFY OK: token_counts conserved ({merged_total}), non-zero scores.")
PYEOF
fi

echo "=== parallel capture COMPLETE ==="
