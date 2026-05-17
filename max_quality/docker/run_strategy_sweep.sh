#!/usr/bin/env bash
# Strategy-sweep orchestrator — runs the 5 "beyond greedy" directions D/B/A/C/E
# end to end: baseline S0 -> budget-retune -> SA -> SAB -> SC -> SCD -> SE.
#
# Runs INSIDE the ghcr.io/lucaspirola/moe-compress image (pure Python env);
# all orchestration lives here in the repo so changing it never needs an image
# rebuild. Invoke with the image's entrypoint overridden, e.g.:
#
#   docker run -d --gpus all -v /data/cache:/cache \
#     -e HF_TOKEN=hf_... \
#     --entrypoint bash ghcr.io/lucaspirola/moe-compress:latest -c '
#       set -e
#       if [ -d /cache/code/moe_compress/.git ]; then
#         git -C /cache/code/moe_compress fetch origin main
#         git -C /cache/code/moe_compress reset --hard origin/main
#       else
#         git clone --depth 1 -b main https://github.com/lucaspirola/moe_compress \
#           /cache/code/moe_compress
#       fi
#       exec bash /cache/code/moe_compress/max_quality/docker/run_strategy_sweep.sh'
#
# Why a dedicated script (not bootstrap.sh): Direction A needs a budget_retune
# step BETWEEN S0 and the SA/SAB rows — bootstrap.sh runs run_ablations exactly
# once and has no hook for that. It also needs --stage6-mode thermometer and
# MOE_KEEP_STAGE2_PARTIAL=1, neither of which bootstrap.sh passes.
#
# Disk: each row's Stage-2/2.5 model dirs (stage2_pruned, stage2p5_final) are
# ~50 GB. A scalar-bpt_gap sweep does not need to hoard 6 compressed models, so
# each row's heavy dirs are deleted once its thermometer eval is on disk (and,
# via run_ablations, uploaded to the bucket). KEEP_HEAVY_ARTIFACTS=1 disables.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required (set via -e HF_TOKEN=...)}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-35B-A3B}"
TEACHER_MODEL_REPO="${TEACHER_MODEL_REPO:-Qwen/Qwen3.6-35B-A3B-FP8}"   # H200 needs FP8 teacher (BF16 OOMs)
CONFIG_PATH="${CONFIG_PATH:-configs/qwen36_35b_a3b_30pct.yaml}"
CACHE_MOUNT="${CACHE_MOUNT:-/cache}"
CODE_DIR="${CODE_DIR:-$CACHE_MOUNT/code/moe_compress}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
STAGE6_MODE="${STAGE6_MODE:-thermometer}"
HF_ARTIFACTS_BUCKET="${HF_ARTIFACTS_BUCKET:-pirola/moe-ablations}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-pirola/trackio}"
KEEP_HEAVY_ARTIFACTS="${KEEP_HEAVY_ARTIFACTS:-0}"
ABLATIONS_ROOT="$CACHE_MOUNT/ablations"

export HF_HOME="${HF_HOME:-$CACHE_MOUNT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_ARTIFACTS_BUCKET TRACKIO_SPACE_ID

log() { printf '[strategy-sweep] %s :: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

log "================================================================"
log " moe_compress strategy sweep — directions D/B/A/C/E"
log " MODEL=$MODEL_REPO TEACHER=$TEACHER_MODEL_REPO STAGE6=$STAGE6_MODE"
log " ablations_root=$ABLATIONS_ROOT  num_sequences=$NUM_SEQUENCES"
log "================================================================"

command -v nvidia-smi >/dev/null 2>&1 || { log "FATAL: no nvidia-smi (start container with --gpus all)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

mkdir -p "$CACHE_MOUNT/hf" "$ABLATIONS_ROOT" "$CACHE_MOUNT/code"

[[ -d "$CODE_DIR/.git" ]] || { log "FATAL: code not cloned at $CODE_DIR"; exit 1; }
HARNESS_DIR="$CODE_DIR/max_quality"
log "code HEAD = $(git -C "$CODE_DIR" rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
# Model snapshot prefetch (idempotent — skipped if already in $HF_HOME/hub).
# ---------------------------------------------------------------------------
log "prefetching model snapshot $MODEL_REPO"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', cache_dir='$HF_HOME/hub', allow_patterns=['*'])
print('snapshot_download complete')
"

# ---------------------------------------------------------------------------
# FP8 KD-teacher kernel prefetch + metadata self-heal (mirrors bootstrap.sh
# step 5.5 — required because we use the FP8 teacher). The deep-gemm /
# finegrained-fp8 kernels lazy-load mid-Stage-2.5; their metadata.json is
# missing name/id that kernels>=0.14.0 strict-validates. Fetch + patch eagerly.
# ---------------------------------------------------------------------------
log "pre-fetching + patching FP8 KD-teacher kernels"
python3 - <<'PYEOF'
import glob, hashlib, json, os, sys
try:
    from kernels import get_kernel
    for repo_id in ("kernels-community/deep-gemm", "kernels-community/finegrained-fp8"):
        try:
            get_kernel(repo_id)
            print(f"[kernels-prefetch] {repo_id}: cached", flush=True)
        except Exception as exc:
            print(f"[kernels-prefetch] {repo_id} fetch failed (patch will retry): {exc}", flush=True)
except ImportError as exc:
    print(f"[kernels-prefetch] `kernels` not installed: {exc} — skipping", flush=True)
    sys.exit(0)
patched = 0
for p in glob.glob(f"{os.environ.get('HF_HOME', '/cache/hf')}/hub/kernels--*/snapshots/*/build/*/metadata.json"):
    try:
        with open(p) as f: m = json.load(f)
    except Exception:
        continue
    changed = False
    if "name" not in m:
        for part in p.split("/"):
            if part.startswith("kernels--") and part.count("--") >= 2:
                m["name"] = part.split("--", 2)[2].replace("_", "-"); changed = True; break
    if "id" not in m:
        m["id"] = "_" + hashlib.md5(p.encode()).hexdigest()[:14]; changed = True
    if changed:
        with open(p, "w") as f: json.dump(m, f, indent=2)
        patched += 1
print(f"[kernels-prefetch] complete: {patched} metadata file(s) patched", flush=True)
PYEOF

cd "$HARNESS_DIR"
export PYTHONPATH="$HARNESS_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

run_ablations() {  # $1 = comma-separated ablation ids
    log "run_ablations --only $1 --stage6-mode $STAGE6_MODE"
    python -m moe_compress.run_ablations \
        --config "$HARNESS_DIR/$CONFIG_PATH" \
        --model "$MODEL_REPO" \
        --ablations-root "$ABLATIONS_ROOT" \
        --num-sequences "$NUM_SEQUENCES" \
        --stage6-mode "$STAGE6_MODE" \
        --teacher-model-repo "$TEACHER_MODEL_REPO" \
        --only "$1"
}

# Drop a completed row's ~50 GB model dirs; keep the eval JSON + stage1 inputs.
# Only runs once the row's thermometer result is on disk (so a failed/partial
# row keeps its artifacts for diagnosis).
clean_row() {  # $1 = row id
    local d="$ABLATIONS_ROOT/$1"
    [[ "$KEEP_HEAVY_ARTIFACTS" == "1" ]] && return 0
    [[ -f "$d/stage6alt_eval.json" ]] || { log "clean_row $1: no result yet — keeping artifacts"; return 0; }
    rm -rf "$d/stage2_pruned" "$d/stage2p5_final" "$d/_stage2_partial"
    log "clean_row $1: dropped stage2_pruned / stage2p5_final / _stage2_partial"
}

row_done() {  # $1 = row id — true if the thermometer eval landed
    [[ -f "$ABLATIONS_ROOT/$1/stage6alt_eval.json" ]]
}

# ---------------------------------------------------------------------------
# Step 1 — Stage-1 pre-flight, then baseline S0.
#
# Pre-flight MUST be its own process: run_ablations._preflight loads the model
# for Stage 1 IN-PROCESS, which leaves ~114 GiB of model/CKA tensors resident
# on the GPU in the parent process. If S0 then runs in that same process its
# Stage-2 subprocess only sees ~26 GiB free and OOMs. A separate
# --preflight-only invocation frees all of it on exit; the subsequent --only S0
# process starts GPU-clean and its Stage-2 subprocess gets the full 141 GiB.
# (This is the split the docker README documents as the "recommended pattern".)
# S0's Stage 2 keeps _stage2_partial/ (MOE_KEEP_STAGE2_PARTIAL=1) for the retune.
# ---------------------------------------------------------------------------
export MOE_KEEP_STAGE2_PARTIAL=1
log "STEP 1/5 — Stage-1 pre-flight (separate process)"
python -m moe_compress.run_ablations \
    --config "$HARNESS_DIR/$CONFIG_PATH" \
    --model "$MODEL_REPO" \
    --ablations-root "$ABLATIONS_ROOT" \
    --num-sequences "$NUM_SEQUENCES" \
    --stage6-mode "$STAGE6_MODE" \
    --teacher-model-repo "$TEACHER_MODEL_REPO" \
    --preflight-only
log "STEP 1/5 — baseline S0"
run_ablations "S0"
row_done S0 || { log "FATAL: S0 did not complete — cannot retune. See $ABLATIONS_ROOT/S0"; exit 1; }

# ---------------------------------------------------------------------------
# Step 2 — Direction A: retune the per-layer expert budget against S0's
# measured Stage-2 merge damage.
# ---------------------------------------------------------------------------
log "STEP 2/5 — budget_retune on S0"
RETUNED="$ABLATIONS_ROOT/S0/stage1_budgets.retuned.json"
python -m moe_compress.budget_retune "$ABLATIONS_ROOT/S0" --output-path "$RETUNED" --verbose
[[ -f "$RETUNED" ]] || { log "FATAL: budget_retune produced no $RETUNED"; exit 1; }

# ---------------------------------------------------------------------------
# Step 3 — pre-place the retuned budgets into SA/ and SAB/ so _seed_stage1_-
# artifacts keeps them (its _hardlink_or_copy skips a dest that already exists).
# Then S0 no longer needs its heavy dirs.
# ---------------------------------------------------------------------------
log "STEP 3/5 — pre-placing retuned budgets into SA/ and SAB/"
for row in SA SAB; do
    mkdir -p "$ABLATIONS_ROOT/$row"
    cp "$RETUNED" "$ABLATIONS_ROOT/$row/stage1_budgets.json"
    log "  $row/stage1_budgets.json <- retuned"
done
clean_row S0
# Non-S0 rows do not need the partials kept — let Stage 2 auto-delete them.
unset MOE_KEEP_STAGE2_PARTIAL

# ---------------------------------------------------------------------------
# Step 4 — the remaining rows, one at a time, cleaning each on completion.
# ---------------------------------------------------------------------------
log "STEP 4/5 — SA, SAB, SC, SCD, SE"
for row in SA SAB SC SCD SE; do
    log "--- row $row ---"
    run_ablations "$row" || log "row $row: run_ablations returned non-zero (see leaderboard)"
    clean_row "$row"
done

# ---------------------------------------------------------------------------
# Step 5 — regenerate the full leaderboard. Every row is _is_complete (its
# stage6alt_eval.json is kept), so this does no GPU work — it just reloads the
# six results and rewrites _leaderboard.md / _summary.json.
# ---------------------------------------------------------------------------
log "STEP 5/5 — regenerating combined leaderboard"
run_ablations "S0,SA,SAB,SC,SCD,SE" || true

log "================================================================"
log " >>> STRATEGY SWEEP COMPLETE"
log " >>> leaderboard: $ABLATIONS_ROOT/_leaderboard.md"
for r in S0 SA SAB SC SCD SE; do
    row_done "$r" && log "   $r: done" || log "   $r: MISSING result"
done
log "================================================================"
