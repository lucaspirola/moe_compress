#!/usr/bin/env bash
# Strategy-sweep orchestrator — runs the 5 "beyond greedy" directions D/B/A/C/E
# end to end: baseline S0 -> budget-retune -> SA -> SAB -> SC -> SCD -> SE.
#
# Runs INSIDE the ghcr.io/lucaspirola/moe-compress image (pure Python env);
# all orchestration lives here in the repo so changing it never needs an image
# rebuild. Invoke with the image's entrypoint overridden, e.g.:
#
#   docker run --gpus all -v /data/cache:/cache \
#     -e HF_TOKEN=hf_... \
#     --entrypoint bash ghcr.io/lucaspirola/moe-compress:latest -c '
#       set -e
#       git clone --depth 1 -b main https://github.com/lucaspirola/moe_compress \
#         /cache/code/moe_compress 2>/dev/null \
#         || git -C /cache/code/moe_compress fetch origin main \
#            && git -C /cache/code/moe_compress reset --hard origin/main
#       exec bash /cache/code/moe_compress/max_quality/docker/run_strategy_sweep.sh'
#
# Why a dedicated script (not bootstrap.sh): Direction A needs a budget_retune
# step BETWEEN S0 and the SA/SAB rows — bootstrap.sh runs run_ablations exactly
# once and has no hook for that. It also needs --stage6-mode thermometer and
# MOE_KEEP_STAGE2_PARTIAL=1, neither of which bootstrap.sh passes.
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
ABLATIONS_ROOT="$CACHE_MOUNT/ablations"

export HF_HOME="${HF_HOME:-$CACHE_MOUNT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_ARTIFACTS_BUCKET TRACKIO_SPACE_ID
# Direction A: budget_retune reads the per-layer damage from _stage2_partial/,
# which Stage 2 deletes on a successful finish unless this is set. Exported so
# every run_ablations subprocess inherits it (harmless extra disk for non-A rows).
export MOE_KEEP_STAGE2_PARTIAL=1

log() { printf '[strategy-sweep] %s\n' "$*" >&2; }

log "================================================================"
log " moe_compress strategy sweep — directions D/B/A/C/E"
log " MODEL=$MODEL_REPO TEACHER=$TEACHER_MODEL_REPO STAGE6=$STAGE6_MODE"
log " ablations_root=$ABLATIONS_ROOT  num_sequences=$NUM_SEQUENCES"
log "================================================================"

command -v nvidia-smi >/dev/null 2>&1 || { log "FATAL: no nvidia-smi (start container with --gpus all)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

mkdir -p "$CACHE_MOUNT/hf" "$ABLATIONS_ROOT" "$CACHE_MOUNT/code"

# ---------------------------------------------------------------------------
# Code: the wrapper -c above already cloned/updated it; verify + record HEAD.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Step 1 — baseline S0 (also runs the shared Stage-1 pre-flight). S0's Stage 2
# keeps _stage2_partial/ (MOE_KEEP_STAGE2_PARTIAL=1) for the retune below.
# ---------------------------------------------------------------------------
log "STEP 1/4 — baseline S0 (+ Stage-1 pre-flight)"
run_ablations "S0"

# ---------------------------------------------------------------------------
# Step 2 — Direction A: retune the per-layer expert budget against S0's
# measured Stage-2 merge damage.
# ---------------------------------------------------------------------------
log "STEP 2/4 — budget_retune on S0"
RETUNED="$ABLATIONS_ROOT/S0/stage1_budgets.retuned.json"
python -m moe_compress.budget_retune "$ABLATIONS_ROOT/S0" --output-path "$RETUNED" --verbose
[[ -f "$RETUNED" ]] || { log "FATAL: budget_retune produced no $RETUNED"; exit 1; }

# ---------------------------------------------------------------------------
# Step 3 — pre-place the retuned budgets into SA/ and SAB/ so _seed_stage1_-
# artifacts keeps them (its _hardlink_or_copy skips a dest that already exists).
# ---------------------------------------------------------------------------
log "STEP 3/4 — pre-placing retuned budgets into SA/ and SAB/"
for row in SA SAB; do
    mkdir -p "$ABLATIONS_ROOT/$row"
    cp "$RETUNED" "$ABLATIONS_ROOT/$row/stage1_budgets.json"
    log "  $row/stage1_budgets.json <- retuned"
done

# ---------------------------------------------------------------------------
# Step 4 — the remaining rows: A, A+B, C, C+D, E.
# ---------------------------------------------------------------------------
log "STEP 4/4 — SA, SAB, SC, SCD, SE"
run_ablations "SA,SAB,SC,SCD,SE"

log "================================================================"
log " >>> STRATEGY SWEEP COMPLETE — leaderboard: $ABLATIONS_ROOT/_leaderboard.md"
log "================================================================"
