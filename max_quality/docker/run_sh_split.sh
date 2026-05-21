#!/usr/bin/env bash
# SH split-machine orchestrator — runs the single `SH` row (merge-heal with
# the LR schedule + cross-domain WikiText holdout telemetry on the
# `feat/heal-lr-schedule` branch) across TWO machines via an attached
# persistent volume:
#
#   PHASE 1 (MOE_PHASE=stage2) — typically on an RTX 6000 Pro Blackwell
#     ($2.405/hr on Spheron spheron-es US Central 1). Pulls Stage 1 _shared/
#     artifacts from HF, runs Stage 1 preflight + Stage 2 with --skip-stage2p5,
#     uploads stage2_pruned/ to HF as background insurance, exits.
#
#   PHASE 2 (MOE_PHASE=stage2p5) — typically on an H200 ($4.615/hr same
#     provider/region). With the volume re-attached so stage2_pruned/ is
#     in place, runs Stage 2.5 + Stage 6 alt. Uploads final artifacts.
#
# Runs INSIDE ghcr.io/lucaspirola/moe-compress:latest. The Spheron orchestrator
# (max_quality/docker/spheron_launch.py) is what spins up the boxes and
# attaches the volume; this script is the per-container bootstrap.
#
# Why a dedicated script (not run_strategy_sweep.sh): that script runs the
# 5-row strategy sweep (S0/SA/SAB/SC/SCD/SE). For our single SH row split
# across machines the sweep's budget_retune step is dead weight, and the
# strategy sweep has no notion of phase 1 vs phase 2.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required (set via -e HF_TOKEN=...)}"
: "${MOE_PHASE:?MOE_PHASE is required: 'phase0' (CPU prep), 'stage2' (Phase 1: RTX 6000 Pro) or 'stage2p5' (Phase 2: H200)}"
case "$MOE_PHASE" in phase0|stage2|stage2p5) ;; *) echo "FATAL: MOE_PHASE=$MOE_PHASE must be 'phase0', 'stage2' or 'stage2p5'" >&2; exit 2 ;; esac

MOE_BRANCH="${MOE_BRANCH:-main}"
ROW_ID="${ROW_ID:-SH}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-35B-A3B}"
TEACHER_MODEL_REPO="${TEACHER_MODEL_REPO:-Qwen/Qwen3.6-35B-A3B-FP8}"   # H200 only; Phase 1 ignores it
CONFIG_PATH="${CONFIG_PATH:-configs/qwen36_35b_a3b_30pct.yaml}"
CACHE_MOUNT="${CACHE_MOUNT:-/cache}"
CODE_DIR="${CODE_DIR:-$CACHE_MOUNT/code/moe_compress}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
STAGE6_MODE="${STAGE6_MODE:-thermometer}"
HF_ARTIFACTS_BUCKET="${HF_ARTIFACTS_BUCKET:-pirola/moe-strategy-35pct}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-pirola/trackio}"
ABLATIONS_ROOT="$CACHE_MOUNT/ablations"

export HF_HOME="${HF_HOME:-$CACHE_MOUNT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_ARTIFACTS_BUCKET TRACKIO_SPACE_ID

log() { printf '[sh-split %s] %s :: %s\n' "$MOE_PHASE" "$(date -u +%H:%M:%S)" "$*" >&2; }

log "================================================================"
log " moe_compress SH split run — PHASE=$MOE_PHASE  ROW=$ROW_ID"
log " BRANCH=$MOE_BRANCH  MODEL=$MODEL_REPO"
log " ablations_root=$ABLATIONS_ROOT  config=$CONFIG_PATH"
log "================================================================"

mkdir -p "$CACHE_MOUNT/hf" "$ABLATIONS_ROOT" "$CACHE_MOUNT/code"

[[ -d "$CODE_DIR/.git" ]] || { log "FATAL: code not cloned at $CODE_DIR"; exit 1; }
HARNESS_DIR="$CODE_DIR/max_quality"

# Optional config overrides (MOE_TOKEN_CAP / MOE_XD_HOLDOUT_TOKENS) live
# in a function so we can call it after the phase1/phase2 git reset wipes
# the working tree. Phase 0 doesn't need this — phase0_prep.py takes
# --token-cap directly on the CLI and writes its own pool-size metadata.
maybe_override_config() {
    [[ -z "${MOE_TOKEN_CAP:-}" && -z "${MOE_XD_HOLDOUT_TOKENS:-}" ]] && return 0
    local cfg="$HARNESS_DIR/$CONFIG_PATH"
    [[ -f "$cfg" ]] || { log "FATAL: config not found at $cfg"; exit 1; }
    if [[ -n "${MOE_TOKEN_CAP:-}" ]]; then
        log "overriding merge_heal_token_cap=$MOE_TOKEN_CAP in $cfg"
        sed -i -E "s/^([[:space:]]*merge_heal_token_cap:)[[:space:]]*[0-9]+(.*)$/\\1 $MOE_TOKEN_CAP\\2/" "$cfg"
    fi
    if [[ -n "${MOE_XD_HOLDOUT_TOKENS:-}" ]]; then
        log "overriding merge_heal_xd_holdout_tokens=$MOE_XD_HOLDOUT_TOKENS in $cfg"
        sed -i -E "s/^([[:space:]]*merge_heal_xd_holdout_tokens:)[[:space:]]*[0-9]+(.*)$/\\1 $MOE_XD_HOLDOUT_TOKENS\\2/" "$cfg"
    fi
    log "post-override config values:"
    grep -E "^[[:space:]]*(merge_heal_token_cap|merge_heal_xd_holdout_tokens):" "$cfg" | sed 's/^/  /'
}

# ---------------------------------------------------------------------------
# Phase 0 (CPU prep) — early dispatch
# ---------------------------------------------------------------------------
# Runs `phase0_prep.py` on the volume: snapshots the base + FP8 teacher
# models from HF into $HF_HOME/hub, pulls Stage-1 GRAPE artifacts from the
# strategy bucket, and tokenizes the calibration corpus once on CPU. Skips
# every GPU stage. Spheron-ES has no CPU-only offer, so this runs on the
# same RTX 6000 Pro hardware as Phase 1 — we eat the GPU rate during prep
# in exchange for keeping Phase 1's bootstrap minimal and warm-cached.
#
# Env-var overrides recognised in phase0:
#   PHASE0_TOKEN_CAP        — override merge_heal_token_cap (e.g. 26214400 for 100×)
#   PHASE0_SEQUENCE_LENGTH  — override calibration sequence_length
#   PHASE0_NO_TEACHER       — set to "1" to skip the FP8 teacher snapshot
#   PHASE0_ARTIFACTS_BUCKET — override the Stage-1 bucket (default pirola/moe-strategy-35pct)
if [[ "$MOE_PHASE" == "phase0" ]]; then
    log "PHASE 0 (CPU prep) — running phase0_prep.py"
    PHASE0_ARGS=(--volume-root "$CACHE_MOUNT" --config "$HARNESS_DIR/$CONFIG_PATH" \
                 --model-repo "$MODEL_REPO" --teacher-repo "$TEACHER_MODEL_REPO" \
                 --artifacts-bucket "${PHASE0_ARTIFACTS_BUCKET:-$HF_ARTIFACTS_BUCKET}")
    if [[ -n "${PHASE0_TOKEN_CAP:-}" ]]; then
        PHASE0_ARGS+=(--token-cap "$PHASE0_TOKEN_CAP")
    fi
    if [[ -n "${PHASE0_SEQUENCE_LENGTH:-}" ]]; then
        PHASE0_ARGS+=(--sequence-length "$PHASE0_SEQUENCE_LENGTH")
    fi
    if [[ "${PHASE0_NO_TEACHER:-0}" == "1" ]]; then
        PHASE0_ARGS+=(--no-teacher)
    fi
    log "phase0_prep.py args: ${PHASE0_ARGS[*]}"
    PYTHONPATH="$HARNESS_DIR/src" python3 "$HARNESS_DIR/scripts/phase0_prep.py" "${PHASE0_ARGS[@]}"
    log "PHASE 0 DONE — volume prepared at $CACHE_MOUNT"
    exit 0
fi

# Phase 1 / Phase 2 GPU checks (skipped on phase0).
command -v nvidia-smi >/dev/null 2>&1 || { log "FATAL: no nvidia-smi (start container with --gpus all)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Honor MOE_BRANCH: fetch + reset --hard so the volume's checked-out branch
# matches what this phase intends to run (the volume persists across phases,
# so without this Phase 2 would silently use whatever Phase 1 left behind).
log "fetching + resetting $CODE_DIR to origin/$MOE_BRANCH"
git -C "$CODE_DIR" fetch --depth=1 origin "$MOE_BRANCH"
git -C "$CODE_DIR" checkout "$MOE_BRANCH" 2>/dev/null || git -C "$CODE_DIR" checkout -b "$MOE_BRANCH" "origin/$MOE_BRANCH"
git -C "$CODE_DIR" reset --hard "origin/$MOE_BRANCH"
log "code HEAD = $(git -C "$CODE_DIR" rev-parse --short HEAD) on $(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD)"

# Re-apply any config overrides AFTER the git reset (which would otherwise
# wipe the sed edits made in a prior phase or earlier in this run).
maybe_override_config

# ---------------------------------------------------------------------------
# Model snapshot prefetch (idempotent). Phase 1 hits this fresh on a new
# volume (~70 GB BF16 download, ~30 min). Phase 2 sees it already cached.
# ---------------------------------------------------------------------------
log "prefetching model snapshot $MODEL_REPO"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', cache_dir='$HF_HOME/hub', allow_patterns=['*'])
print('snapshot_download complete')
"

# ---------------------------------------------------------------------------
# Phase 2 only: seed FP8 KD-teacher kernels from the image-baked copy.
# ---------------------------------------------------------------------------
if [[ "$MOE_PHASE" == "stage2p5" && -n "${TEACHER_MODEL_REPO:-}" ]]; then
    if [ -d /opt/kernels-hub/hub ]; then
        log "seeding FP8 kernels from image-baked /opt/kernels-hub (no HF fetch)"
        cp -rn /opt/kernels-hub/hub/kernels--* "$HF_HOME/hub/" 2>/dev/null || true
    fi
    # Runtime fallback + metadata self-heal: the image bake is permissive on
    # 429 storms, so the baked cache may be partial. Without this block a
    # partial bake silently lands a Stage 2.5 ValueError ~20-90 min in (after
    # the heavy lifting), wasting H200 wall time. Lifted from bootstrap.sh
    # 218-271 — the canonical fix for the kernels>=0.14 metadata format.
    log "pre-fetching FP8 KD-teacher kernels + patching metadata.json (TEACHER=$TEACHER_MODEL_REPO)"
    python3 - <<'PYEOF'
import glob, hashlib, json, os, sys
try:
    from kernels import get_kernel
    for repo_id in ("kernels-community/deep-gemm", "kernels-community/finegrained-fp8"):
        try:
            get_kernel(repo_id)
            print(f"[kernels-prefetch] {repo_id}: cached", flush=True)
        except Exception as exc:
            print(f"[kernels-prefetch] {repo_id} fetch failed (patch will salvage): {exc}", flush=True)
except ImportError as exc:
    print(f"[kernels-prefetch] `kernels` package not installed: {exc} — skipping", flush=True)
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
fi

# ---------------------------------------------------------------------------
# Pull Stage 1 _shared/ artifacts from the HF bucket if not on the volume yet.
# Per project_stage1_artifacts_reuse the canonical set lives on the bucket;
# Stage 1 is row-independent so we never need to re-derive it.
# ---------------------------------------------------------------------------
log "Seeding Stage 1 _shared/ from bucket $HF_ARTIFACTS_BUCKET (idempotent — skips if already on volume)"
python3 - <<PYEOF
import os, sys, pathlib, logging
logging.basicConfig(level=logging.WARNING)

bucket = "$HF_ARTIFACTS_BUCKET"
ablations_root = pathlib.Path("$ABLATIONS_ROOT")
shared_dir = ablations_root / "_shared"
token = os.environ.get("HF_TOKEN")

from huggingface_hub import HfApi
api = HfApi(token=token)

needed = {"_shared/stage1_blacklist.json",
          "_shared/stage1_budgets.json",
          "_shared/budget_decomposition.json"}
local_ok = all((ablations_root / f).exists() for f in needed)
if local_ok:
    print("[seed] _shared/ already on volume — skipping bucket fetch", flush=True)
    sys.exit(0)

try:
    remote = {it.path for it in api.list_bucket_tree(bucket, recursive=True)
              if hasattr(it, "size")}
except Exception as e:
    print(f"[seed] cannot list bucket {bucket}: {e} — Stage 1 will re-derive", flush=True)
    sys.exit(0)

if not needed.issubset(remote):
    print("[seed] _shared/ incomplete on bucket — Stage 1 will re-derive", flush=True)
    sys.exit(0)

to_pull = [(f, str(ablations_root / f)) for f in sorted(remote)
           if f.startswith("_shared/") and not f.endswith(".lock")]
for _, local in to_pull:
    pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
api.download_bucket_files(bucket_id=bucket, files=to_pull)
print(f"[seed] downloaded {len(to_pull)} _shared/ file(s) from bucket", flush=True)
PYEOF

cd "$HARNESS_DIR"
export PYTHONPATH="$HARNESS_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Stage 1 pre-flight runs in its own subprocess (frees model GPU memory on
# exit so the per-row Stage-2 subprocess starts clean). On Phase 2 the
# _shared/ artifacts are already on the volume — preflight is a fast no-op.
# ---------------------------------------------------------------------------
log "Stage 1 pre-flight (separate process; idempotent against _shared/)"
python3 -m moe_compress.run_ablations \
    --config "$HARNESS_DIR/$CONFIG_PATH" \
    --model "$MODEL_REPO" \
    --ablations-root "$ABLATIONS_ROOT" \
    --num-sequences "$NUM_SEQUENCES" \
    --stage6-mode "$STAGE6_MODE" \
    --teacher-model-repo "$TEACHER_MODEL_REPO" \
    --preflight-only

# ---------------------------------------------------------------------------
# Run the SH row. Phase 1 uses --skip-stage2p5 to exit after Stage 2; Phase 2
# does NOT pass that flag, so the existing --resume-from-stage 2 shortcut
# picks up the cached stage2_pruned/ and runs Stage 2.5 + Stage 6 alt.
# ---------------------------------------------------------------------------
PIPELINE_ARGS=(
    --config "$HARNESS_DIR/$CONFIG_PATH"
    --model "$MODEL_REPO"
    --ablations-root "$ABLATIONS_ROOT"
    --num-sequences "$NUM_SEQUENCES"
    --stage6-mode "$STAGE6_MODE"
    --teacher-model-repo "$TEACHER_MODEL_REPO"
    --only "$ROW_ID"
)
if [[ "$MOE_PHASE" == "stage2" ]]; then
    PIPELINE_ARGS+=(--skip-stage2p5)
fi

log "running ablation $ROW_ID (phase=$MOE_PHASE)"
python3 -m moe_compress.run_ablations "${PIPELINE_ARGS[@]}"

# ---------------------------------------------------------------------------
# Completion check + summary log.
# ---------------------------------------------------------------------------
ROW_DIR="$ABLATIONS_ROOT/$ROW_ID"
case "$MOE_PHASE" in
    stage2)
        [[ -d "$ROW_DIR/stage2_pruned" ]] || { log "FATAL: phase 1 finished without writing $ROW_DIR/stage2_pruned/"; exit 1; }
        log "PHASE 1 DONE — stage2_pruned/ on volume at $ROW_DIR/stage2_pruned"
        log "Volume detach-and-reattach to a Stage-2.5-capable GPU (H200) for phase 2."
        ;;
    stage2p5)
        [[ -f "$ROW_DIR/stage6alt_eval.json" ]] || { log "FATAL: phase 2 finished without writing $ROW_DIR/stage6alt_eval.json"; exit 1; }
        log "PHASE 2 DONE — final eval at $ROW_DIR/stage6alt_eval.json"
        log "    bpt_gap: $(python3 -c "import json; print(json.load(open('$ROW_DIR/stage6alt_eval.json'))['bpt_gap'])" 2>/dev/null || echo "?")"
        log "Artifacts uploaded to bucket $HF_ARTIFACTS_BUCKET by run_ablations background thread."
        ;;
esac

log "================================================================"
log " >>> SH split — PHASE $MOE_PHASE COMPLETE"
log "================================================================"
