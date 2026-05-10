#!/usr/bin/env bash
# kdr vast.ai container bootstrap.
#
# Pulls the moe_compress repo (the image is shared with max_quality and ships
# only deps + this script), installs Zyphra's transformers fork over the base
# image's stock transformers, snapshot-downloads teacher + student into the
# cache mount, derives the deterministic run_id, queries HF Hub for prior
# partials, and invokes the trainer with --resume-from if applicable.
#
# Implements LLR-0008 (CLI surface), LLR-0031 (run_id), LLR-0032 (env-var
# validation), LLR-0033 (partials query), LLR-0035 (Zyphra fork install).
#
# Exit codes:
#   2 — missing/malformed env var (LLR-0032)
#   3 — Zyphra transformers fork install failed (LLR-0035)
#   4 — git clone, snapshot_download, or hub-side resolution failed
#   5 — trainer or final-upload failed
#   6 — final-artifact load-back round-trip failed (third-party-loadability check)
#
# Inline Python blocks read inputs via ``os.environ[...]`` rather than
# string-interpolated shell vars — this prevents shell-side injection / quoting
# bugs when paths contain quotes or apostrophes (e.g. /workspace/user's-cache).
#
# REQ: LLR-0032
# REQ: LLR-0035

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 1. Env-var validation (LLR-0032)
# ─────────────────────────────────────────────────────────────────────────────

usage() {
    cat >&2 <<'USAGE'
kdr bootstrap.sh — vast.ai container entrypoint.

Required environment variables:
  HF_TOKEN          HF Hub token with write access to the partials + final repos.
                    Read directly by huggingface_hub at API call time; we do
                    NOT call `huggingface-cli login` (which would write the
                    token to ~/.gitconfig — undesirable on persistent volumes).
  STUDENT_REPO      HF Hub repo ID of the student model (e.g. Zyphra/ZAYA1-reasoning-base).
  CACHE_MOUNT       Absolute path on the vast.ai instance where teacher + student
                    snapshots are downloaded (e.g. /workspace/cache).
  KDR_CONFIG        Path to the YAML config (relative to the cloned repo or absolute).
  KDR_MODE          One of "bf16" | "da_qad". Embedded in run_id derivation.

Optional:
  PARTIALS_REPO_PREFIX     Default "pirola/kdr-partials". Final repo: "{prefix}-{run_id}".
  RECOVERED_REPO_PREFIX    Default "pirola/kdr-recovered". Final repo: "{prefix}-{run_id}".
  MOE_COMPRESS_GIT_URL     Default https://github.com/lucaspirola/moe_compress.git
  MOE_COMPRESS_GIT_REF     Default "main".
USAGE
    exit 2
}

require_env() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        echo "ERROR: required environment variable ${var_name} is unset or empty." >&2
        usage
    fi
}

require_env HF_TOKEN
require_env STUDENT_REPO
require_env CACHE_MOUNT
require_env KDR_CONFIG
require_env KDR_MODE

# Validate KDR_MODE early — saves a 17 GB teacher download if mode is malformed.
case "${KDR_MODE}" in
    bf16|da_qad) ;;
    *)
        echo "ERROR: KDR_MODE must be 'bf16' or 'da_qad' (got '${KDR_MODE}')." >&2
        exit 2
        ;;
esac

PARTIALS_REPO_PREFIX="${PARTIALS_REPO_PREFIX:-pirola/kdr-partials}"
RECOVERED_REPO_PREFIX="${RECOVERED_REPO_PREFIX:-pirola/kdr-recovered}"
MOE_COMPRESS_GIT_URL="${MOE_COMPRESS_GIT_URL:-https://github.com/lucaspirola/moe_compress.git}"
MOE_COMPRESS_GIT_REF="${MOE_COMPRESS_GIT_REF:-main}"

# Export HF_TOKEN so huggingface_hub picks it up automatically (no `login` step).
export HF_TOKEN
# Pin the HF cache to the persistent mount so re-launches reuse the snapshot.
export HF_HOME="${CACHE_MOUNT}/hf-cache"

mkdir -p "${CACHE_MOUNT}" "${HF_HOME}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Repo clone (depth=1, ref-pinned, hard-reset on re-launch)
# ─────────────────────────────────────────────────────────────────────────────

REPO_DIR="${CACHE_MOUNT}/moe_compress"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo ">>> Cloning ${MOE_COMPRESS_GIT_URL}@${MOE_COMPRESS_GIT_REF} into ${REPO_DIR}"
    if ! git clone --depth=1 --branch "${MOE_COMPRESS_GIT_REF}" "${MOE_COMPRESS_GIT_URL}" "${REPO_DIR}"; then
        echo "ERROR: git clone failed." >&2
        exit 4
    fi
else
    # Re-launch on a persistent volume: discard any local mutations from the
    # prior run (partial pip installs, half-applied patches) before fetching.
    echo ">>> Repo already cloned at ${REPO_DIR}; hard-resetting to origin/${MOE_COMPRESS_GIT_REF}"
    git -C "${REPO_DIR}" fetch --depth=1 origin "${MOE_COMPRESS_GIT_REF}"
    git -C "${REPO_DIR}" reset --hard "origin/${MOE_COMPRESS_GIT_REF}"
fi

cd "${REPO_DIR}/knowledge_distillation_recovery/kdr"
pip install -e . --no-deps --quiet || {
    echo "ERROR: kdr package install failed." >&2
    exit 4
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Zyphra transformers fork install (LLR-0035)
# ─────────────────────────────────────────────────────────────────────────────

echo ">>> Installing Zyphra transformers fork (zaya1 branch)"
if ! pip install --upgrade --force-reinstall --quiet \
        "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1"; then
    echo "ERROR: Zyphra transformers fork install failed (LLR-0035)." >&2
    exit 3
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Snapshot-download student (HF_TOKEN read from env by huggingface_hub)
# ─────────────────────────────────────────────────────────────────────────────

export STUDENT_LOCAL_DIR="${CACHE_MOUNT}/student"
echo ">>> Snapshot-downloading student ${STUDENT_REPO} into ${STUDENT_LOCAL_DIR}"
if ! python -c '
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["STUDENT_REPO"],
    local_dir=os.environ["STUDENT_LOCAL_DIR"],
)
'; then
    echo "ERROR: snapshot_download of student ${STUDENT_REPO} failed." >&2
    exit 4
fi

# Fetch the student's HF Hub revision SHA — feeds run_id derivation.
echo ">>> Resolving student HF Hub SHA"
STUDENT_REPO_SHA="$(python -c '
import os
from huggingface_hub import HfApi
sha = HfApi().model_info(os.environ["STUDENT_REPO"]).sha
print(sha or "")
')"
if [[ -z "${STUDENT_REPO_SHA}" || "${STUDENT_REPO_SHA}" == "None" ]]; then
    echo "ERROR: could not resolve HF Hub SHA for ${STUDENT_REPO}." >&2
    exit 4
fi
export STUDENT_REPO_SHA

# ─────────────────────────────────────────────────────────────────────────────
# 5. Compute run_id (LLR-0031) + query HF Hub for resume seed (LLR-0033)
# ─────────────────────────────────────────────────────────────────────────────

# Resolve KDR_CONFIG to an absolute path — the bootstrap may be invoked with
# either a repo-relative or absolute path.
if [[ "${KDR_CONFIG}" = /* ]]; then
    CONFIG_PATH="${KDR_CONFIG}"
else
    CONFIG_PATH="${REPO_DIR}/${KDR_CONFIG}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: KDR_CONFIG=${KDR_CONFIG} resolved to ${CONFIG_PATH}, which does not exist." >&2
    exit 2
fi
export CONFIG_PATH

echo ">>> Deriving run_id from (config, student_sha=${STUDENT_REPO_SHA:0:8}, mode=${KDR_MODE})"
RUN_ID="$(python -c '
import os
import yaml
from kdr.config import Config
from kdr.io.run_id import derive_run_id

with open(os.environ["CONFIG_PATH"]) as f:
    cfg = Config.model_validate(yaml.safe_load(f))
print(derive_run_id(cfg, os.environ["STUDENT_REPO_SHA"], os.environ["KDR_MODE"]))
')"
if [[ -z "${RUN_ID}" ]]; then
    echo "ERROR: run_id derivation failed (empty hash)." >&2
    exit 4
fi
echo ">>> run_id=${RUN_ID}"

PARTIALS_REPO="${PARTIALS_REPO_PREFIX}-${RUN_ID}"
RECOVERED_REPO="${RECOVERED_REPO_PREFIX}-${RUN_ID}"
export PARTIALS_REPO RECOVERED_REPO

# Query the partials repo for the highest-step ``_SAVE_COMPLETE``-d partial.
# A missing repo or zero candidates => start from scratch.
ARTIFACTS_DIR="${CACHE_MOUNT}/artifacts"
mkdir -p "${ARTIFACTS_DIR}"
export ARTIFACTS_DIR

echo ">>> Querying ${PARTIALS_REPO} for resume seed"
RESUME_INFO="$(python -c '
import os
from pathlib import Path
from kdr.io.resume import find_latest_partial_on_hub, download_partial_from_hub
result = find_latest_partial_on_hub(os.environ["PARTIALS_REPO"])
if result is None:
    print("NONE")
else:
    dir_name, step = result
    target = Path(os.environ["ARTIFACTS_DIR"])
    download_partial_from_hub(os.environ["PARTIALS_REPO"], dir_name, target)
    print(f"{target}/{dir_name}|{step}")
')"

# Build the resume arg as an array — protects against word-splitting if any
# upstream path component contains spaces (M4 from the Phase 6 review).
resume_args=()
if [[ "${RESUME_INFO}" != "NONE" ]]; then
    RESUME_PATH="${RESUME_INFO%%|*}"
    RESUME_STEP="${RESUME_INFO##*|}"
    echo ">>> Resuming from step ${RESUME_STEP} at ${RESUME_PATH}"
    resume_args=("--resume-from" "${RESUME_PATH}")
else
    echo ">>> No prior partials; starting from scratch"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Invoke the trainer
# ─────────────────────────────────────────────────────────────────────────────

# Export the partials/recovered repo names so the trainer's save callback
# can upload to the correct hub repos.
export KDR_PARTIALS_REPO="${PARTIALS_REPO}"
export KDR_RECOVERED_REPO="${RECOVERED_REPO}"

echo ">>> Invoking kdr.cli.train (mode=${KDR_MODE})"
if ! python -m kdr.cli.train \
        --config "${CONFIG_PATH}" \
        --student "${STUDENT_LOCAL_DIR}" \
        --mode "${KDR_MODE}" \
        --artifacts-dir "${ARTIFACTS_DIR}" \
        "${resume_args[@]}"; then
    echo "ERROR: trainer invocation failed." >&2
    exit 5
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. Final-checkpoint upload (LLR-0030)
# ─────────────────────────────────────────────────────────────────────────────

FINAL_DIR="${ARTIFACTS_DIR}/kdr_${KDR_MODE}_recovered"
export FINAL_DIR
if [[ -d "${FINAL_DIR}" ]]; then
    echo ">>> Uploading final artifact to ${RECOVERED_REPO}"
    if ! python -c '
import os
from pathlib import Path
from kdr.io.resume import upload_final_to_hub
url = upload_final_to_hub(Path(os.environ["FINAL_DIR"]), os.environ["RECOVERED_REPO"])
print(f"Final artifact: {url}")
'; then
        echo "ERROR: final-artifact upload failed." >&2
        exit 5
    fi
else
    echo "WARNING: final dir ${FINAL_DIR} not found — trainer did not produce a final save." >&2
fi

# ─────────────────────────────────────────────────────────────────────────────
# 8. Load-back round-trip (third-party-loadability sanity)
# ─────────────────────────────────────────────────────────────────────────────
#
# Validates that the just-uploaded artifact is consumable from a fresh
# Python process via the canonical HF API — closes the loop from "we wrote
# something" to "what we wrote is usable downstream". The trainer's resume
# path already exercises kdr-internal load; this exercises the API surface
# any external consumer would use.
#
# bf16: AutoModelForCausalLM round-trip; trivial check that the safetensors
# + config.json + tokenizer files form a valid HF checkpoint.
#
# da_qad: deferred to Phase 7.2 — needs a compressed-tensors-aware loader
# (compressed-tensors package is gated behind kdr's `[compressed]` extra
# which the default --no-deps install above does not pull).

if [[ -d "${FINAL_DIR}" && "${KDR_MODE}" == "bf16" ]]; then
    echo ">>> Load-back round-trip: pulling ${RECOVERED_REPO} in a fresh Python process"
    if ! python - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = os.environ["RECOVERED_REPO"]
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="cuda"
)
ids = tok("hello world", return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    logits = model(input_ids=ids).logits
assert torch.isfinite(logits).all(), "NaN or Inf in logits — artifact is corrupt"
print(f"load-back OK: logits shape={tuple(logits.shape)}, dtype={logits.dtype}, finite ✓")
PY
    then
        echo "ERROR: final-artifact load-back round-trip failed." >&2
        exit 6
    fi
elif [[ -d "${FINAL_DIR}" && "${KDR_MODE}" == "da_qad" ]]; then
    echo ">>> Load-back round-trip skipped for da_qad — Phase 7.2 will add the compressed-tensors round-trip."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 9. Destroy hint
# ─────────────────────────────────────────────────────────────────────────────

cat <<'HINT'

────────────────────────────────────────────────────────────────────────
kdr bootstrap complete.

Final artifact on HF Hub. Destroy this vast.ai instance to stop billing:

    curl -sSf "https://console.vast.ai/api/v0/instances/${VAST_INSTANCE_ID}/" \
         -H "Authorization: Bearer ${VAST_API_KEY}" -X DELETE

────────────────────────────────────────────────────────────────────────
HINT
