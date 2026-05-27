#!/bin/bash
# v2_validation_harness.sh — installer + runner for the v2 calibration
# validation harness on HF Jobs.
#
# VALIDATION HARNESS — NOT PART OF THE PRODUCTION CODEBASE.
#
# Container: nvidia/cuda:13.0.0-devel-ubuntu24.04 (matches the wheel
#            build env that produced pirola/vllm-patched-calib).
# Flavor   : h200 (1x H200, 141 GB VRAM, $5.00/hr) — fits BF16
#            Qwen3.6-35B-A3B (~70 GB weights) with ~71 GB free for
#            KV cache + activations. L40S/48 GB rejected because the
#            Phase B build script has no --quantization flag (BF16
#            only) and 48 GB cannot hold a BF16 35B model.
#            A100-large/80 GB was the cheapest fitting flavor but its
#            ~10 GB headroom over the 70 GB weights is too tight for
#            max_new_tokens=16384 KV cache. rtx-pro-6000 at $2.75/hr
#            would have been ideal (96 GB, $2.25 cheaper/hr) but is
#            not present in the hf-CLI's flavor enum (CLI is stale
#            against the live hf jobs hardware list).
# Inputs   : env HF_TOKEN passed via `hf jobs run --secrets HF_TOKEN`,
#            env V2_VAL_COMMIT (git SHA hosting the harness .py + .sh,
#            pinned to a3a946a by default).
#
# Phases:
#   1. apt-install minimal packages (python, build deps, curl).
#   2. venv + pip install torch 2.11.0+cu130 (matches the wheel) +
#      NumPy<2 + huggingface_hub + transformers + accelerate +
#      datasets.
#   3. Pull patched vLLM wheel from pirola/vllm-patched-calib via the
#      hf_hub_download Python API and pip install it.
#   4. git clone the moe_compress repo at V2_VAL_COMMIT into
#      /tmp/moe_compress so the harness can subprocess-invoke the
#      v2 build script.
#   5. curl the harness .py from GitHub raw at V2_VAL_COMMIT.
#   6. Run the harness. Captures /tmp/v2_validation_results.json.
#      (Harness handles its own upload to pirola/calibration-v2-validation.)

set -e
set -o pipefail

V2_VAL_COMMIT="${V2_VAL_COMMIT:-a3a946a}"
V2_VAL_RESULTS_REPO="${V2_VAL_RESULTS_REPO:-pirola/calibration-v2-validation}"
VLLM_WHEEL_REPO="${VLLM_WHEEL_REPO:-pirola/vllm-patched-calib}"
VLLM_WHEEL_FILE="${VLLM_WHEEL_FILE:-vllm-0.21.1.dev0+gad7125a43.d20260526-cp312-cp312-linux_x86_64.whl}"
REPO_ROOT="${V2_VAL_REPO_ROOT:-/tmp/moe_compress}"

echo "[$(date)] === Phase 1: apt-install ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    git python3 python3-pip python3-venv python3-dev \
    curl ca-certificates

echo "[$(date)] === Phase 2: venv + torch 2.11.0+cu130 + base deps ==="
python3 -m venv /tmp/venv
# shellcheck disable=SC1091
. /tmp/venv/bin/activate
pip install --quiet --upgrade pip wheel "setuptools>=77"
# NumPy first so torch can initialize its NumPy bridge cleanly.
pip install --quiet "numpy<2.0"
pip install --quiet torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

# huggingface_hub for both the wheel download AND the results upload.
pip install --quiet "huggingface_hub>=1.16"

echo "[$(date)] === Phase 3: install patched vLLM wheel from ${VLLM_WHEEL_REPO} ==="
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
# Install with deps so transitive vLLM requirements (xformers, ray,
# datasets, etc.) get resolved against PyPI.
pip install "/tmp/wheels/${VLLM_WHEEL_FILE}"
python -c "import vllm; print('vllm:', vllm.__version__)"

echo "[$(date)] === Phase 4: install transformers + accelerate + datasets ==="
# accelerate required when device_map= is used in from_pretrained.
# datasets required for the harness's MoT prompt sampling (streaming
# load_dataset) AND for the v2 build-script subprocess.
pip install --quiet "transformers>=4.51.0" "accelerate>=0.30.0" "datasets>=2.20.0"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"
python -c "import datasets; print('datasets:', datasets.__version__)"

echo "[$(date)] === Phase 5: git clone moe_compress @ ${V2_VAL_COMMIT} ==="
if [ -d "${REPO_ROOT}" ]; then
    rm -rf "${REPO_ROOT}"
fi
git clone --quiet --no-checkout \
    https://github.com/lucaspirola/moe_compress.git "${REPO_ROOT}"
git -C "${REPO_ROOT}" fetch --quiet --depth 1 origin "${V2_VAL_COMMIT}"
git -C "${REPO_ROOT}" checkout --quiet "${V2_VAL_COMMIT}"
git -C "${REPO_ROOT}" log -1 --oneline

# The v2 build script imports moe_compress.utils.calibration; expose the
# package via PYTHONPATH for the harness's subprocess invocation.
export PYTHONPATH="${REPO_ROOT}/max_quality/src:${PYTHONPATH:-}"

echo "[$(date)] === Phase 6: fetch + run the harness ==="
HARNESS_URL="https://raw.githubusercontent.com/lucaspirola/moe_compress/${V2_VAL_COMMIT}/max_quality/scripts/v2_validation_harness.py"
echo "  source: ${HARNESS_URL}"
curl -sL -o /tmp/v2_validation_harness.py "${HARNESS_URL}"
wc -l /tmp/v2_validation_harness.py
md5sum /tmp/v2_validation_harness.py

# Export the env-flags the harness expects. expert_in capture is flipped
# at runtime inside the harness via vllm.calibration_hooks._CAPTURE_EXPERT
# but we also export it so any pre-import code paths see the right value.
export VLLM_CALIB_CAPTURE_EXPERT=1
export VLLM_CALIB_CAPTURE_ROUTER=0
export VLLM_CALIB_CAPTURE_BLOCK=0
export VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=0
export VLLM_CALIB_CAPTURE_EXPERT_MID=0
export VLLM_CALIB_CAPTURE_IMATRIX=0
export VLLM_CALIB_CAPTURE_INPUT_COV=0
export VLLM_CALIB_CAPTURE_REAP_SCORES=0
export VLLM_CALIB_CAPTURE_PER_EXPERT_MAX=0
export VLLM_CALIB_MAX_LAYER=-1
export V2_VAL_COMMIT="${V2_VAL_COMMIT}"
export V2_VAL_REPO_ROOT="${REPO_ROOT}"
export V2_VAL_RESULTS_REPO="${V2_VAL_RESULTS_REPO}"

set +e
python /tmp/v2_validation_harness.py
HARNESS_RC=$?
set -e
echo "[$(date)] harness exit code: ${HARNESS_RC}"

# The harness uploads its own results to ${V2_VAL_RESULTS_REPO}. If it
# crashed before writing /tmp/v2_validation_results.json, synthesize a
# minimal stub and upload that so the supervisor can poll for *some*
# result file.
if [ ! -f /tmp/v2_validation_results.json ]; then
    echo "[$(date)] WARNING: /tmp/v2_validation_results.json missing — synthesizing stub"
    python - <<PYEOF
import json, os, time
from huggingface_hub import create_repo, upload_file
repo = "${V2_VAL_RESULTS_REPO}"
token = os.environ["HF_TOKEN"]
payload = {
    "harness_version": 1,
    "harness_name": "v2_validation_harness",
    "status": "CRASHED_BEFORE_RESULTS_WRITE",
    "exit_code": ${HARNESS_RC},
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
stub = "/tmp/v2_validation_results.json"
open(stub, "w").write(json.dumps(payload, indent=2))
create_repo(repo, repo_type="dataset", exist_ok=True, private=False, token=token)
stamp = time.strftime("%Y%m%dT%H%M%S")
upload_file(
    path_or_fileobj=stub,
    path_in_repo=f"results/v2_validation_{stamp}_CRASHED.json",
    repo_id=repo, repo_type="dataset", token=token,
)
upload_file(
    path_or_fileobj=stub,
    path_in_repo="results/latest.json",
    repo_id=repo, repo_type="dataset", token=token,
)
print(f"stub uploaded -> https://huggingface.co/datasets/{repo}/blob/main/results/latest.json")
PYEOF
fi

echo "[$(date)] === done (harness rc=${HARNESS_RC}) ==="
exit ${HARNESS_RC}
