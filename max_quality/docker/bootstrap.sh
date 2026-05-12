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
# Stage 5 / Stage 2.5 VRAM-reduction levers. Empty = use config defaults.
# See max_quality/configs/qwen36_35b_a3b_30pct.yaml and
# stage5_router_kd._get_teacher for what each knob does.
TEACHER_MODEL_REPO="${TEACHER_MODEL_REPO:-}"
STAGE5_MAX_CALIBRATION_SAMPLES="${STAGE5_MAX_CALIBRATION_SAMPLES:-}"
STAGE5_MAX_SEQUENCE_LENGTH="${STAGE5_MAX_SEQUENCE_LENGTH:-}"

export HF_HOME="${HF_HOME:-$CACHE_MOUNT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TRACKIO_SPACE_ID
export HF_ARTIFACTS_BUCKET
# HF_ARTIFACTS_BUCKET and HF_TOKEN are exported so run_ablations.py can read
# them for per-ablation streaming uploads (uploads in background thread while
# GPU runs the next ablation).

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
log "TEACHER_MODEL_REPO= ${TEACHER_MODEL_REPO:-(default BF16 from config.model)}"
log "STAGE5_MAX_CAL_SS = ${STAGE5_MAX_CALIBRATION_SAMPLES:-(config default)}"
log "STAGE5_MAX_SEQ_LN = ${STAGE5_MAX_SEQUENCE_LENGTH:-(config default)}"

# vast.ai's /.launch writes authorized_keys with group/world-writable permissions,
# which sshd rejects under StrictModes yes (the default). Fix early so SSH works
# as soon as bootstrap starts.
if [[ -d /root/.ssh ]]; then
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
fi

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
# 2.5. Seed artifacts from HF Hub (idempotent)
#      Pulls _shared/ (Stage 1 artifacts) and any A*/stage6_eval.json already
#      on Hub so _is_complete() skips them on this fresh instance. Enables
#      GPU-switching mid-sweep: destroy instance, launch cheaper one, resume.
# ---------------------------------------------------------------------------
if [[ -n "${HF_ARTIFACTS_BUCKET:-}" ]]; then
    log "Seeding artifacts from bucket $HF_ARTIFACTS_BUCKET → $CACHE_MOUNT/ablations"
    python3 - <<PYEOF
import os, sys, pathlib, logging
logging.basicConfig(level=logging.WARNING)

bucket = "$HF_ARTIFACTS_BUCKET"
ablations_root = pathlib.Path("$CACHE_MOUNT/ablations")
shared_dir = ablations_root / "_shared"
token = os.environ.get("HF_TOKEN")

try:
    from huggingface_hub import HfApi
except ImportError:
    print("[seed] huggingface_hub not available — skipping", flush=True)
    sys.exit(0)

api = HfApi(token=token)
try:
    # list_bucket_tree returns BucketFile (has .size) and BucketFolder (no .size)
    remote_files = {item.path for item in api.list_bucket_tree(bucket, recursive=True)
                    if hasattr(item, "size")}
except Exception as e:
    print(f"[seed] cannot list bucket {bucket}: {e} — skipping", flush=True)
    sys.exit(0)

needed_shared = {"_shared/stage1_blacklist.json", "_shared/stage1_budgets.json",
                 "_shared/budget_decomposition.json"}
shared_ok = all((shared_dir / f.split("/", 1)[1]).exists() for f in needed_shared)

to_pull = []  # list of (remote_path, local_path)
if not shared_ok:
    missing = [f for f in needed_shared if f in remote_files]
    if len(missing) == len(needed_shared):
        for f in remote_files:
            if f.startswith("_shared/") and not f.endswith(".lock"):
                local = ablations_root / f
                to_pull.append((f, str(local)))
        print("[seed] will pull _shared/ (Stage 1 artifacts)", flush=True)
    else:
        print("[seed] _shared/ incomplete on bucket — Stage 1 will run fresh", flush=True)

for f in sorted(remote_files):
    parts = f.split("/")
    if len(parts) == 2 and parts[1] == "stage6_eval.json":
        dst = ablations_root / parts[0] / "stage6_eval.json"
        if not dst.exists():
            to_pull.append((f, str(dst)))
            print(f"[seed] will pull {f} (marks {parts[0]} complete)", flush=True)

if not to_pull:
    print("[seed] nothing to pull — cache already up to date", flush=True)
    sys.exit(0)

# Ensure parent dirs exist (download_bucket_files writes to absolute paths)
for _, local in to_pull:
    pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)

api.download_bucket_files(bucket_id=bucket, files=to_pull)
print(f"[seed] downloaded {len(to_pull)} file(s)", flush=True)
PYEOF
fi

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
# 5.5. FP8 KD-teacher kernel prefetch + metadata self-heal
#
# transformers.integrations.finegrained_fp8 lazily loads `deep-gemm` and
# `finegrained-fp8` kernels via the `kernels` package during Stage 2.5's
# first FP8 forward pass (~90 min into a run). Two known upstream gotchas:
#
#   (1) `kernels-community/deep-gemm`'s metadata.json is missing the
#       `name` and `id` fields that kernels>=0.14.0 strict-validates;
#       load raises ValueError mid-Stage-2.5.
#   (2) Lazy-load happens deep into the run, so a failure there wastes
#       all of Stage 2's wall-time.
#
# Fix: trigger the kernel downloads eagerly here (each is small — MBs),
# then walk the resulting metadata.json files and inject sensible name/id
# values for any kernel that lacks them. Only applies the patch if the
# kernel teacher path is actually being used (TEACHER_MODEL_REPO non-empty);
# default BF16 teacher path doesn't need this and we want to skip the
# extra download.
# ---------------------------------------------------------------------------
if [[ -n "${TEACHER_MODEL_REPO:-}" ]]; then
    log "Pre-fetching FP8 KD-teacher kernels (TEACHER_MODEL_REPO=$TEACHER_MODEL_REPO) and patching metadata.json"
    python3 - <<'PYEOF'
import glob, hashlib, json, os, sys

# Trigger lazy downloads. The kernels API caches into HF_HOME/hub/kernels--*/.
# Best-effort — even if a fetch fails (e.g., HF rate limit), the patch step
# below fixes whatever did land on disk.
try:
    from kernels import get_kernel
    for repo_id in ("kernels-community/deep-gemm", "kernels-community/finegrained-fp8"):
        try:
            get_kernel(repo_id)
            print(f"[kernels-prefetch] {repo_id}: cached", flush=True)
        except Exception as exc:
            print(f"[kernels-prefetch] {repo_id} fetch failed (will patch + retry next time): {exc}",
                  flush=True)
except ImportError as exc:
    print(f"[kernels-prefetch] `kernels` package not installed: {exc} — skipping", flush=True)
    sys.exit(0)

# Patch metadata.json files missing required fields. kernels>=0.14.0 requires
# both `name` and `id`. Name must satisfy the strict format (lowercase letters,
# digits, dashes, start with letter, end with letter/digit) — we infer it from
# the cache path which uses the kernels--<org>--<name>/ layout. Underscores in
# the inferred name are mapped to dashes per the spec.
patched = 0
for p in glob.glob(f"{os.environ.get('HF_HOME', '/cache/hf')}/hub/kernels--*/snapshots/*/build/*/metadata.json"):
    try:
        with open(p) as f:
            m = json.load(f)
    except Exception:
        continue
    changed = False
    if "name" not in m:
        for part in p.split("/"):
            if part.startswith("kernels--") and part.count("--") >= 2:
                m["name"] = part.split("--", 2)[2].replace("_", "-")
                changed = True
                break
    if "id" not in m:
        # Stable id derived from the path so re-runs land on the same value
        # (helps the kernels cache key stay consistent across invocations).
        m["id"] = "_" + hashlib.md5(p.encode()).hexdigest()[:14]
        changed = True
    if changed:
        with open(p, "w") as f:
            json.dump(m, f, indent=2)
        patched += 1
        print(f"[kernels-prefetch] patched: {p}", flush=True)

print(f"[kernels-prefetch] complete: {patched} metadata file(s) patched", flush=True)
PYEOF
fi

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
# Stage 5 VRAM-reduction levers — opt-in. Each only adds to ARGS if set;
# unset = harness reads the config default (BF16 teacher, full calibration).
if [[ -n "$TEACHER_MODEL_REPO" ]]; then
    ARGS+=("--teacher-model-repo" "$TEACHER_MODEL_REPO")
fi
if [[ -n "$STAGE5_MAX_CALIBRATION_SAMPLES" ]]; then
    ARGS+=("--stage5-max-calibration-samples" "$STAGE5_MAX_CALIBRATION_SAMPLES")
fi
if [[ -n "$STAGE5_MAX_SEQUENCE_LENGTH" ]]; then
    ARGS+=("--stage5-max-sequence-length" "$STAGE5_MAX_SEQUENCE_LENGTH")
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
# 7. Final artifact upload safety net
#    Per-ablation uploads (stage6_eval.json + uploaded.flag) are handled by
#    run_ablations.py streaming threads — they run while the GPU works on the
#    next ablation. This block is a safety net for anything that didn't get
#    uploaded (e.g. run crashed before a thread completed), and for _shared/.
# ---------------------------------------------------------------------------
if [[ "$UPLOAD_ON_SUCCESS" == "1" ]]; then
    log "Final artifact sync to bucket $HF_ARTIFACTS_BUCKET"
    hf buckets sync "$CACHE_MOUNT/ablations/" "hf://buckets/$HF_ARTIFACTS_BUCKET/" \
        --include "_shared/**" \
        --include "*/stage6_eval.json" \
        --include "_summary.json" \
        || log "WARNING: final artifact sync failed (non-fatal; artifacts remain on $CACHE_MOUNT)"
fi

# ---------------------------------------------------------------------------
# 8. Final hint
# ---------------------------------------------------------------------------
log "================================================================"
log " >>> RUN COMPLETE"
log " >>> destroy instance with: $DESTROY_HINT"
log "================================================================"
