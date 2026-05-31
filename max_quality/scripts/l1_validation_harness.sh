#!/bin/bash
# l1_validation_harness.sh -- installer + runner for the L1 validation
# harness on HF Jobs.
#
# VALIDATION HARNESS -- NOT PART OF THE PRODUCTION CODEBASE.
#
# Container: nvidia/cuda:13.0.0-devel-ubuntu24.04 (matches the wheel build env).
# Flavor   : a10g-small (4 vCPU, 15 GB RAM, 24 GB A10G, $1/hr)
# Inputs   : env HF_TOKEN passed via `hf jobs run --secrets HF_TOKEN`,
#            env L1_HARNESS_COMMIT (git SHA hosting the .py harness)
#
# Phases:
#   1. apt-install minimal packages (python, build deps).
#   2. venv + pip install torch 2.11.0+cu130 (matches the wheel) and the
#      patched vLLM wheel from pirola/vllm-patched-calib.
#   3. pip install transformers (for the side-by-side comparison).
#   4. curl the harness .py from the moe_compress GitHub repo at
#      ${L1_HARNESS_COMMIT}.
#   5. Run the harness. Captures /tmp/l1_validation_results.json.
#   6. Upload the JSON to pirola/l1-validation-results (HF dataset).

set -e
set -o pipefail

L1_HARNESS_COMMIT="${L1_HARNESS_COMMIT:-feat/calibration-v2}"
L1_RESULTS_REPO="${L1_RESULTS_REPO:-pirola/l1-validation-results}"
VLLM_WHEEL_REPO="${VLLM_WHEEL_REPO:-pirola/vllm-patched-calib}"
VLLM_WHEEL_FILE="${VLLM_WHEEL_FILE:-vllm-0.21.1.dev0+gad7125a43.d20260531-cp312-cp312-linux_x86_64.whl}"

echo "[$(date)] === Phase 1: apt-install ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    git python3 python3-pip python3-venv python3-dev \
    curl ca-certificates

echo "[$(date)] === Phase 2: venv + torch 2.11.0+cu130 ==="
python3 -m venv /tmp/venv
# shellcheck disable=SC1091
. /tmp/venv/bin/activate
pip install --quiet --upgrade pip wheel "setuptools>=77"
# NumPy first so torch can initialize its NumPy bridge cleanly.
pip install --quiet "numpy<2.0"
pip install --quiet torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

echo "[$(date)] === Phase 3: install patched vLLM wheel from ${VLLM_WHEEL_REPO} ==="
# Use the Python API instead of the `hf` CLI -- the CLI's click/typer
# deps are no longer transitive in huggingface_hub 1.16+, and the
# Python API is what Phase 6 already uses for the results upload.
pip install --quiet huggingface_hub
mkdir -p /tmp/wheels
python - <<PYEOF
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="${VLLM_WHEEL_REPO}",
    filename="${VLLM_WHEEL_FILE}",
    local_dir="/tmp/wheels",
    token=os.environ.get("HF_TOKEN"),
)
print(f"downloaded -> {path}")
PYEOF
ls -lh /tmp/wheels/
# Install with deps so transitive vLLM requirements (xformers, ray, etc.)
# get resolved against PyPI -- this is slower than --no-deps but correct.
pip install "/tmp/wheels/${VLLM_WHEEL_FILE}"
python -c "import vllm; print('vllm:', vllm.__version__)"

echo "[$(date)] === Phase 4: install transformers + accelerate ==="
# `accelerate` is required as soon as `device_map=` is used in
# from_pretrained, which the harness does for the side-by-side
# HF reference model.
pip install --quiet "transformers>=4.51.0" "accelerate>=0.30.0"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"

echo "[$(date)] === Phase 5: fetch + run the harness ==="
HARNESS_URL="https://raw.githubusercontent.com/lucaspirola/moe_compress/${L1_HARNESS_COMMIT}/max_quality/scripts/l1_validation_harness.py"
echo "  source: ${HARNESS_URL}"
curl -sL -o /tmp/l1_validation_harness.py "${HARNESS_URL}"
wc -l /tmp/l1_validation_harness.py
md5sum /tmp/l1_validation_harness.py

# Pre-export the vLLM env-flags the harness expects. expert_in capture
# is flipped at runtime inside the harness, but other writers need to
# stay OFF lest they crash (no sidecar paths configured).
export VLLM_CALIB_CAPTURE_EXPERT=0
export VLLM_CALIB_CAPTURE_ROUTER=0
export VLLM_CALIB_CAPTURE_BLOCK=0
export VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=0
export VLLM_CALIB_CAPTURE_EXPERT_MID=0
export VLLM_CALIB_CAPTURE_IMATRIX=0
export VLLM_CALIB_CAPTURE_INPUT_COV=0
# Disable the L2 early-exit by default.
export VLLM_CALIB_MAX_LAYER=-1

set +e
python /tmp/l1_validation_harness.py
HARNESS_RC=$?
set -e
echo "[$(date)] harness exit code: ${HARNESS_RC}"

echo "[$(date)] === Phase 6: upload results to ${L1_RESULTS_REPO} ==="
if [ ! -f /tmp/l1_validation_results.json ]; then
    echo "WARNING: /tmp/l1_validation_results.json missing -- harness crashed before writing"
    # Synthesize a minimal results file so the upload still happens.
    python - <<PYEOF
import json, time
json.dump({
    "harness_version": 1,
    "status": "CRASHED_BEFORE_RESULTS_WRITE",
    "exit_code": ${HARNESS_RC},
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}, open("/tmp/l1_validation_results.json", "w"), indent=2)
PYEOF
fi

python - <<PYEOF
import os, time
from huggingface_hub import HfApi, create_repo, upload_file

repo_id = "${L1_RESULTS_REPO}"
token = os.environ["HF_TOKEN"]
create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False, token=token)

stamp = time.strftime("%Y%m%dT%H%M%S")
path_in_repo = f"results/l1_validation_{stamp}.json"
upload_file(
    path_or_fileobj="/tmp/l1_validation_results.json",
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
    token=token,
)
print(f"uploaded -> https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}")

# Also upload as latest.json for easy polling.
upload_file(
    path_or_fileobj="/tmp/l1_validation_results.json",
    path_in_repo="results/latest.json",
    repo_id=repo_id,
    repo_type="dataset",
    token=token,
)
print(f"uploaded -> https://huggingface.co/datasets/{repo_id}/blob/main/results/latest.json")
PYEOF

echo "[$(date)] === done (harness rc=${HARNESS_RC}) ==="
exit ${HARNESS_RC}
