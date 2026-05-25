#!/bin/bash
# hf_jobs_build_patched_vllm.sh — Build vLLM 0.21.0 multi-arch wheel with our
# calibration hooks patch applied, inside an HF Jobs container.
#
# Inputs (env vars):
#   HF_TOKEN — HF API token, passed via `hf jobs run --secrets HF_TOKEN`
#
# Container: nvidia/cuda:13.0.0-devel-ubuntu24.04 (CUDA toolkit pre-installed)
# Flavor:   cpu-performance (32 vCPU / 256 GB RAM)
# Output:   wheel uploaded to pirola/vllm-patched-calib on HF Hub
#
# Invocation (from the host):
#   hf jobs run --flavor cpu-performance --detach --timeout 6h \
#       --secrets HF_TOKEN \
#       nvidia/cuda:13.0.0-devel-ubuntu24.04 \
#       bash -c "curl -sL https://raw.githubusercontent.com/lucaspirola/moe_compress/feat/calibration-v2/max_quality/scripts/hf_jobs_build_patched_vllm.sh | bash"

set -e
set -o pipefail

echo "[$(date)] === Phase 1: install base packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    git python3 python3-pip python3-venv \
    build-essential ninja-build cmake curl ca-certificates

echo "[$(date)] === Phase 2: make venv + install build prerequisites ==="
python3 -m venv /tmp/venv
# shellcheck disable=SC1091
. /tmp/venv/bin/activate
# Pin setuptools<77: newer setuptools enforces strict pyproject.toml license
# schema; vLLM 0.21.0's pyproject uses the SPDX-string form which the strict
# schema rejects. setuptools 75.x and earlier accept it.
pip install --quiet --upgrade pip "wheel<0.50" "setuptools<77"
pip install --quiet setuptools_scm pybind11 huggingface_hub

echo "[$(date)] === Phase 3: install torch 2.11.0+cu130 ==="
pip install --quiet torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

echo "[$(date)] === Phase 4: clone vLLM v0.21.0 ==="
cd /tmp
rm -rf vllm-patched
git clone --depth 1 --branch v0.21.0 \
    https://github.com/vllm-project/vllm vllm-patched
cd vllm-patched
echo "vllm commit: $(git rev-parse HEAD)"   # should be ad7125a

echo "[$(date)] === Phase 5: fetch and apply calibration hooks patch ==="
curl -sL \
    https://raw.githubusercontent.com/lucaspirola/moe_compress/feat/calibration-v2/max_quality/patches/vllm_calibration_hooks.patch \
    -o /tmp/calib.patch
wc -l /tmp/calib.patch
md5sum /tmp/calib.patch
# Expected MD5: 9effe235a95940d806f626ee1dc841c8 (3087 lines)
git apply --check /tmp/calib.patch
git apply /tmp/calib.patch
echo "Applied. Status:"
git status --short

echo "[$(date)] === Phase 6: build multi-arch wheel ==="
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:${PATH}
export TORCH_CUDA_ARCH_LIST="8.0;9.0a;10.0;12.0"
export MAX_JOBS=16
export NVCC_THREADS=2
nvcc --version | tail -3
echo "CPU count: $(nproc)"
echo "Memory: $(free -h | head -2)"
mkdir -p /tmp/wheels
# Build the wheel (no editable install — produces a portable .whl)
pip wheel . --no-deps -w /tmp/wheels --no-build-isolation 2>&1 | tail -50
ls -la /tmp/wheels/

echo "[$(date)] === Phase 7: upload wheel to HF Hub ==="
python <<'PYEOF'
import os, glob, sys
from huggingface_hub import HfApi, upload_file, create_repo

repo_id = "pirola/vllm-patched-calib"
token = os.environ["HF_TOKEN"]

api = HfApi()
print(f"Creating repo {repo_id} (exist_ok=True)...")
create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=token)

wheels = glob.glob("/tmp/wheels/*.whl")
if not wheels:
    print("ERROR: no wheels found in /tmp/wheels/", file=sys.stderr)
    sys.exit(1)

for w in wheels:
    name = os.path.basename(w)
    size_mb = os.path.getsize(w) / (1024 * 1024)
    print(f"Uploading {name} ({size_mb:.1f} MB)...")
    upload_file(
        path_or_fileobj=w,
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"  -> https://huggingface.co/{repo_id}/blob/main/{name}")

# Also upload the patch itself for traceability
patch_path = "/tmp/calib.patch"
if os.path.exists(patch_path):
    print("Uploading patch artifact...")
    upload_file(
        path_or_fileobj=patch_path,
        path_in_repo="vllm_calibration_hooks.patch",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

# Upload a small README with build metadata
readme = f"""---
license: apache-2.0
tags:
  - vllm
  - calibration
  - patched
---

# vllm-patched-calib

vLLM 0.21.0 (commit `ad7125a`) with calibration-v2 hooks patch applied.

- Source repo: https://github.com/lucaspirola/moe_compress (branch `feat/calibration-v2`, immutable tag `calib-v2-patch-locked`)
- Patch artifact (3087 lines, MD5 `9effe235a95940d806f626ee1dc841c8`): also uploaded to this repo as `vllm_calibration_hooks.patch`
- Architectures: sm_80 (A100), sm_90a (H100/H200), sm_100 (B200), sm_120 (RTX 6000 Pro Blackwell)
- Build host: HF Jobs (cpu-performance)
- torch: 2.11.0+cu130
- CUDA toolkit: 13.0

## Install on a fresh GPU host

```bash
hf download pirola/vllm-patched-calib --include "*.whl" --local-dir /tmp/wheels
pip install /tmp/wheels/vllm-*.whl
```

## Calibration capture flags

The patched vLLM accepts new env vars to enable calibration data capture:

- `VLLM_CALIB_CAPTURE_ROUTER=1`     — per-layer router logits + topk
- `VLLM_CALIB_CAPTURE_EXPERT=1`     — per-expert inputs + weighted outputs
- `VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1` — kernel-level pre-weight per-expert outputs (Triton backend; forces VLLM_USE_FLASHINFER_MOE_FP16=0)
- `VLLM_CALIB_CAPTURE_EXPERT_MID=1` — silu(gate)·up intermediate (input to down_proj; Triton backend)
- `VLLM_CALIB_CAPTURE_BLOCK=1`      — MoE block pre-residual output
- `VLLM_CALIB_CAPTURE_IMATRIX=1`    — per-input-channel sum-of-squares for every linear layer (writes llama.cpp-compatible `.imatrix.dat`)
"""

api.upload_file(
    path_or_fileobj=readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    token=token,
)

print("UPLOAD DONE")
PYEOF

echo "[$(date)] === BUILD COMPLETE ==="
