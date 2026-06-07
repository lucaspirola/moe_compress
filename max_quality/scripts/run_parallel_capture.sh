#!/usr/bin/env bash
# Data-parallel calibration orchestration (replay-capture OR generation).
#
# --mode replay   (default): shard an EXISTING self-traces JSONL into N
#     disjoint shards, run forward-only --replay-from capture on each (one per
#     GPU), then merge the N partial sidecars into the canonical JSONL's
#     sidecar dir. This is the probe's immediate path.
#
# --mode generate (NEW): shard by PROMPT OFFSET into the seeded mix. Each
#     process GENERATES a disjoint slice [C_k, C_{k+1}) of the deterministic
#     per-subset shuffle via the driver's --num-prompts / --prev-num-prompts
#     ladder (same --seed everywhere), writing shard_k/self_traces.jsonl.
#     Disjoint prompt slices -> disjoint completions. Merge = CONCATENATE the
#     N output JSONLs in shard order (with a disjoint+complete assert), plus
#     the sidecar-merge if --captures were set during generation.
#
# Disjointness proof (generate mode): _iter_prompts_from_qwen3_pretrain_mix
# draws each subset from a per-subset deterministic shuffle keyed by
# (seed, subset) — INDEPENDENT of --num-prompts. iter(N) yields the first
# int(N*weight) of each subset; iter(M, prev=N) (M>N) yields positions
# [int(N*weight), int(M*weight)). Same floor formula + same shuffle => slice
# boundaries align with NO gaps and NO overlap. A strictly-increasing count
# ladder C_0=0<C_1<...<C_N gives N provably-disjoint slices whose union is
# iter(C_N). No new driver arg needed.
#
# Pipeline (replay):
#   1. shard_split.py -> shard_0..N-1  (+ disjoint/complete HARD assert)
#   2. launch N procs -> CUDA_VISIBLE_DEVICES=k ... --replay-from shard_k ...
#   3. poll+wait      -> each shard's .done marker + every active signal's
#                        per-shard sidecar present (reap also non-empty)
#   4. merge_sidecars -> into the canonical JSONL's sidecar dir
#   5. verify merged reap_scores (shape, non-zero, token_counts conserved)
#
# Pipeline (generate):
#   1. compute the count ladder C_0..C_N over --num-prompts
#   2. launch N procs -> CUDA_VISIBLE_DEVICES=k ... --num-prompts C_{k+1}
#                        --prev-num-prompts C_k --output shard_k/self_traces.jsonl
#   3. poll+wait      -> each shard's .done + shard_k/self_traces.jsonl present
#   4. concat_jsonls  -> final corpus (disjoint+complete assert); sidecar-merge
#                        if --captures were set during generation
#
# Usage:
#   run_parallel_capture.sh --mode replay --jsonl <corpus.jsonl> --n 4 \
#       [--captures "--capture-reap-scores --capture-per-expert-max"] \
#       [--max-model-len 20480] [--teacher REPO] [--out-dir DIR] \
#       [--py python3] [--moe-backend triton] [--extra-args "..."]
#
#   run_parallel_capture.sh --mode generate --jsonl <final_corpus.jsonl> \
#       --n 4 --num-prompts 8000 --prompts qwen3-pretrain-mix-v2 --seed 1337 \
#       [--captures "..."] [--teacher REPO] [--out-dir DIR] [--py python3] \
#       [--moe-backend triton] [--extra-args "..."]
#
# Env knobs:
#   PARALLEL_CAPTURE_POLL_SECS   poll interval while waiting (default 30)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults + arg parsing
# ---------------------------------------------------------------------------
MODE="replay"
JSONL=""
N=4
CAPTURES=""
MAX_MODEL_LEN=20480
TEACHER="Qwen/Qwen3.6-35B-A3B"
OUT_DIR=""
PY="python3"
EXTRA_ARGS=""
MOE_BACKEND="triton"
NUM_PROMPTS=8000
PROMPTS="qwen3-pretrain-mix-v2"
SEED=1337

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../src" && pwd)"
DRIVER="${SCRIPT_DIR}/build_self_traces_calib_vllm.py"
SHARD_SPLIT="${SCRIPT_DIR}/shard_split.py"

POLL_SECS="${PARALLEL_CAPTURE_POLL_SECS:-30}"

# Default captures: replay needs at least one --capture-*; generate defaults
# to none (pure generation). Set after parsing if still empty.
_CAPTURES_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --jsonl) JSONL="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --captures) CAPTURES="$2"; _CAPTURES_SET=1; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --teacher) TEACHER="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --py) PY="$2"; shift 2 ;;
        --moe-backend) MOE_BACKEND="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --prompts) PROMPTS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ "${MODE}" != "replay" && "${MODE}" != "generate" ]]; then
    echo "ERROR: --mode must be 'replay' or 'generate' (got '${MODE}')" >&2
    exit 2
fi
if [[ -z "${JSONL}" ]]; then
    echo "ERROR: --jsonl is required" >&2
    exit 2
fi
if [[ "${MODE}" == "replay" && ! -f "${JSONL}" ]]; then
    echo "ERROR: replay mode requires an existing --jsonl: ${JSONL}" >&2
    exit 2
fi
if [[ "${_CAPTURES_SET}" -eq 0 && "${MODE}" == "replay" ]]; then
    CAPTURES="--capture-reap-scores"
fi
if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="$(dirname "${JSONL}")/parallel_capture"
fi
mkdir -p "${OUT_DIR}"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

echo "=== parallel ${MODE} ==="
echo "  jsonl         = ${JSONL}"
echo "  n             = ${N}"
echo "  captures      = ${CAPTURES:-<none>}"
echo "  moe-backend   = ${MOE_BACKEND}"
echo "  teacher       = ${TEACHER}"
echo "  out-dir       = ${OUT_DIR}"
if [[ "${MODE}" == "replay" ]]; then
    echo "  max-model-len = ${MAX_MODEL_LEN}"
else
    echo "  num-prompts   = ${NUM_PROMPTS}"
    echo "  prompts       = ${PROMPTS}"
    echo "  seed          = ${SEED}"
fi

# ---------------------------------------------------------------------------
# Helper: map a --capture-FLAG to its per-shard sidecar relative path.
# Used by the M3 generalized existence gate. block-outputs -> a dir.
# ---------------------------------------------------------------------------
_sidecar_rel_for_flag() {
    case "$1" in
        --capture-reap-scores) echo "reap_scores.pt" ;;
        --capture-per-expert-max) echo "per_expert_max.pt" ;;
        --capture-routing-stats) echo "routing_stats.pt" ;;
        --capture-router-logits-stats) echo "router_logits_stats.pt" ;;
        --capture-output-reservoir) echo "output_reservoir.pt" ;;
        --capture-input-covariance) echo "covariance.pt" ;;
        --capture-stage2-profile) echo "stage2_profile.pt" ;;
        --capture-wanda-scalar-row) echo "wanda_scalar_row.pt" ;;
        --capture-block-outputs) echo "block_hidden" ;;   # dir
        *) echo "" ;;   # imatrix (.dat sibling) / unknown -> not gated here
    esac
}

# Verify each ACTIVE --capture-* flag produced its per-shard sidecar (M3).
# reap is additionally checked non-empty. Returns 0 if all good, 1 otherwise.
_check_shard_sidecars() {
    local shard_sidecar_dir="$1"
    local ok=0
    for flag in ${CAPTURES}; do
        case "${flag}" in --capture-*) ;; *) continue ;; esac
        local rel; rel="$(_sidecar_rel_for_flag "${flag}")"
        [[ -z "${rel}" ]] && continue
        local target="${shard_sidecar_dir}/${rel}"
        if [[ "${rel}" == "block_hidden" ]]; then
            if [[ ! -d "${target}" ]] || ! ls "${target}"/layer_*.pt >/dev/null 2>&1; then
                echo "    missing sidecar for ${flag}: ${target}/layer_*.pt" >&2
                ok=1
            fi
        elif [[ ! -f "${target}" ]]; then
            echo "    missing sidecar for ${flag}: ${target}" >&2
            ok=1
        fi
    done
    return ${ok}
}

# ---------------------------------------------------------------------------
# Step 1 (+launch args) differ by mode.
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "replay" ]]; then
    echo "--- step 1: shard_split ---"
    "${PY}" "${SHARD_SPLIT}" "${JSONL}" "${N}" --out-dir "${OUT_DIR}"
else
    # Compute the count ladder C_0=0 < C_1 < ... < C_N = NUM_PROMPTS.
    # Even split of NUM_PROMPTS across N (last slice absorbs remainder).
    echo "--- step 1: compute generate count ladder over ${NUM_PROMPTS} ---"
    LADDER=()
    for ((k = 0; k <= N; k++)); do
        LADDER+=("$(( NUM_PROMPTS * k / N ))")
    done
    echo "  ladder = ${LADDER[*]}"

    # HIGH fix: derive ONE shuffle-buffer from the GLOBAL total and pass the
    # SAME value to every process so all N share an identical shuffle order
    # (otherwise the per-process count-derived buffer diverges once a subset
    # count exceeds ~1000 -> offset slices overlap or gap). Mirror the
    # driver's clamp [10000, 200000] with the global total as the basis.
    #   buffer = min(max(10000, 10*NUM_PROMPTS), 200000)
    GLOBAL_SHUFFLE_BUFFER=$(( 10 * NUM_PROMPTS ))
    [[ "${GLOBAL_SHUFFLE_BUFFER}" -lt 10000 ]] && GLOBAL_SHUFFLE_BUFFER=10000
    [[ "${GLOBAL_SHUFFLE_BUFFER}" -gt 200000 ]] && GLOBAL_SHUFFLE_BUFFER=200000
    echo "  shared shuffle-buffer = ${GLOBAL_SHUFFLE_BUFFER} (same for all ${N} shards)"
fi

# ---------------------------------------------------------------------------
# Step 2: launch N processes (one per GPU).
# ---------------------------------------------------------------------------
echo "--- step 2: launch ${N} ${MODE} processes ---"
PIDS=()
for ((k = 0; k < N; k++)); do
    SHARD_DIR="${OUT_DIR}/shard_${k}"
    mkdir -p "${SHARD_DIR}"
    LOG="${SHARD_DIR}/${MODE}.log"
    DONE="${SHARD_DIR}/.done"
    rm -f "${DONE}" "${DONE}.failed"

    if [[ "${MODE}" == "replay" ]]; then
        SHARD_JSONL="${SHARD_DIR}/shard_${k}.jsonl"
        if [[ ! -f "${SHARD_JSONL}" ]]; then
            echo "ERROR: shard file missing: ${SHARD_JSONL}" >&2
            exit 1
        fi
        echo "  shard ${k}: CUDA_VISIBLE_DEVICES=${k} replay -> ${SHARD_JSONL}"
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
    else
        C_LO="${LADDER[$k]}"
        C_HI="${LADDER[$((k + 1))]}"
        SHARD_OUT="${SHARD_DIR}/self_traces.jsonl"
        PREV_ARG=()
        [[ "${C_LO}" -gt 0 ]] && PREV_ARG=(--prev-num-prompts "${C_LO}")
        echo "  shard ${k}: CUDA_VISIBLE_DEVICES=${k} generate slice [${C_LO},${C_HI}) -> ${SHARD_OUT}"
        (
            if CUDA_VISIBLE_DEVICES="${k}" "${PY}" "${DRIVER}" \
                --prompts "${PROMPTS}" \
                --num-prompts "${C_HI}" \
                "${PREV_ARG[@]}" \
                --seed "${SEED}" \
                --shuffle-buffer "${GLOBAL_SHUFFLE_BUFFER}" \
                --teacher "${TEACHER}" \
                --moe-backend "${MOE_BACKEND}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --output "${SHARD_OUT}" \
                ${CAPTURES} \
                ${EXTRA_ARGS} > "${LOG}" 2>&1; then
                touch "${DONE}"
            else
                echo "FAILED" > "${DONE}.failed"
            fi
        ) &
    fi
    PIDS+=($!)
done

# ---------------------------------------------------------------------------
# Step 3: poll for progress until all .done/.failed markers appear, then wait.
# POLL_SECS controls the visible-progress interval (M2: wired to a real loop).
# ---------------------------------------------------------------------------
echo "--- step 3: wait for ${N} shards (poll every ${POLL_SECS}s) ---"
while true; do
    n_done=0
    n_failed=0
    for ((k = 0; k < N; k++)); do
        [[ -f "${OUT_DIR}/shard_${k}/.done" ]] && n_done=$((n_done + 1))
        [[ -f "${OUT_DIR}/shard_${k}/.done.failed" ]] && n_failed=$((n_failed + 1))
    done
    echo "  progress: ${n_done} done, ${n_failed} failed, of ${N}"
    [[ $((n_done + n_failed)) -ge ${N} ]] && break
    sleep "${POLL_SECS}"
done
# Reap the background jobs (markers are already written; this collects exit).
for ((k = 0; k < N; k++)); do
    wait "${PIDS[$k]}" || true
done

# ---------------------------------------------------------------------------
# Step 3b: per-shard success gate.
#   replay : .done present + (M3) every active capture's sidecar present.
#   generate: .done present + shard_k/self_traces.jsonl present.
# ---------------------------------------------------------------------------
FAILED_SHARDS=()
for ((k = 0; k < N; k++)); do
    SHARD_DIR="${OUT_DIR}/shard_${k}"
    if [[ -f "${SHARD_DIR}/.done.failed" ]]; then
        echo "  shard ${k}: FAILED (see ${SHARD_DIR}/${MODE}.log)" >&2
        FAILED_SHARDS+=("${k}"); continue
    fi
    if [[ ! -f "${SHARD_DIR}/.done" ]]; then
        echo "  shard ${k}: NO done marker (crashed?) -> ${SHARD_DIR}/${MODE}.log" >&2
        FAILED_SHARDS+=("${k}"); continue
    fi
    if [[ "${MODE}" == "replay" ]]; then
        if ! _check_shard_sidecars "${SHARD_DIR}/sidecars/shard_${k}"; then
            echo "  shard ${k}: missing active-signal sidecar(s)" >&2
            FAILED_SHARDS+=("${k}"); continue
        fi
    else
        if [[ ! -s "${SHARD_DIR}/self_traces.jsonl" ]]; then
            echo "  shard ${k}: self_traces.jsonl missing/empty" >&2
            FAILED_SHARDS+=("${k}"); continue
        fi
        if [[ -n "${CAPTURES}" ]] \
           && ! _check_shard_sidecars "${SHARD_DIR}/sidecars/self_traces"; then
            echo "  shard ${k}: missing active-signal sidecar(s)" >&2
            FAILED_SHARDS+=("${k}"); continue
        fi
    fi
    echo "  shard ${k}: OK"
done

if [[ ${#FAILED_SHARDS[@]} -gt 0 ]]; then
    echo "ERROR: ${#FAILED_SHARDS[@]} shard(s) failed: ${FAILED_SHARDS[*]}" >&2
    echo "Aborting merge (a partial merge would undercount)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 4: merge.
#   replay : merge_sidecars into the canonical JSONL's sidecar dir.
#   generate: concat the N output JSONLs into the final corpus, then
#             sidecar-merge if --captures were active during generation.
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "replay" ]]; then
    echo "--- step 4: merge_sidecars -> canonical ${JSONL} ---"
    SHARD_SIDECAR_DIRS=()
    for ((k = 0; k < N; k++)); do
        SHARD_SIDECAR_DIRS+=("${OUT_DIR}/shard_${k}/sidecars/shard_${k}")
    done
    "${PY}" -m moe_compress.utils.merge_sidecars \
        --out-jsonl "${JSONL}" "${SHARD_SIDECAR_DIRS[@]}"
else
    echo "--- step 4: concat ${N} generated JSONLs -> ${JSONL} ---"
    SHARD_JSONLS=()
    for ((k = 0; k < N; k++)); do
        SHARD_JSONLS+=("${OUT_DIR}/shard_${k}/self_traces.jsonl")
    done
    # GAP guard (completeness): when every prompt is expected to yield exactly
    # one row (deterministic completion, no teacher drops), set
    # PARALLEL_CAPTURE_ENFORCE_COMPLETE=1 so concat asserts the merged row
    # count == NUM_PROMPTS — this catches a pure shuffle-buffer GAP that the
    # disjointness check alone cannot see. Off by default because generation
    # can legitimately drop incomplete rows (then only (a)+(b) apply).
    EXPECTED_TOTAL_ARG="-1"
    if [[ "${PARALLEL_CAPTURE_ENFORCE_COMPLETE:-0}" == "1" ]]; then
        EXPECTED_TOTAL_ARG="${NUM_PROMPTS}"
        echo "  completeness gate ON: expecting ${NUM_PROMPTS} rows"
    fi
    "${PY}" - "${JSONL}" "${EXPECTED_TOTAL_ARG}" "${SHARD_JSONLS[@]}" <<'PYEOF'
import sys
from pathlib import Path
from moe_compress.utils.merge_sidecars import concat_jsonls
out = Path(sys.argv[1])
expected = int(sys.argv[2])
shards = [Path(p) for p in sys.argv[3:]]
n = concat_jsonls(shards, out,
                  expected_total=(expected if expected >= 0 else None))
print(f"concat_jsonls: {n} rows -> {out}")
PYEOF
    if [[ -n "${CAPTURES}" ]]; then
        echo "--- step 4b: merge generation-time sidecars -> ${JSONL} ---"
        SHARD_SIDECAR_DIRS=()
        for ((k = 0; k < N; k++)); do
            SHARD_SIDECAR_DIRS+=("${OUT_DIR}/shard_${k}/sidecars/self_traces")
        done
        "${PY}" -m moe_compress.utils.merge_sidecars \
            --out-jsonl "${JSONL}" "${SHARD_SIDECAR_DIRS[@]}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 5: verify merged reap_scores (only when reap was captured).
# ---------------------------------------------------------------------------
if [[ "${CAPTURES}" == *"--capture-reap-scores"* ]]; then
    echo "--- step 5: verify merged reap_scores ---"
    if [[ "${MODE}" == "replay" ]]; then SHARD_STEM="shard"; else SHARD_STEM="self_traces"; fi
    "${PY}" - "${JSONL}" "${OUT_DIR}" "${N}" "${MODE}" <<'PYEOF'
import sys
from pathlib import Path
import torch
from moe_compress.utils.cached_calibration_signals import load_reap_scores

out_jsonl = Path(sys.argv[1]); out_dir = Path(sys.argv[2])
n = int(sys.argv[3]); mode = sys.argv[4]

merged = load_reap_scores(out_jsonl)
assert merged is not None, "merged reap_scores not found"
merged_total = int(merged.token_counts.sum().item())
print(f"merged reap_scores shape={tuple(merged.reap_scores.shape)} "
      f"token_counts.sum()={merged_total}")

def _shard_jsonl(k):
    if mode == "replay":
        return out_dir / f"shard_{k}" / f"shard_{k}.jsonl"
    return out_dir / f"shard_{k}" / "self_traces.jsonl"

shard_total = 0
for k in range(n):
    p = load_reap_scores(_shard_jsonl(k))
    assert p is not None, f"shard {k} reap_scores missing"
    shard_total += int(p.token_counts.sum().item())

assert merged_total == shard_total, (
    f"token_counts mismatch: merged={merged_total} != sum(shards)={shard_total}")
assert merged_total > 0, "merged token_counts.sum() == 0 (empty capture!)"
assert merged.reap_scores.abs().sum().item() > 0, "merged reap all-zero!"
print(f"VERIFY OK: token_counts conserved ({merged_total}), non-zero scores.")
PYEOF
fi

echo "=== parallel ${MODE} COMPLETE ==="
