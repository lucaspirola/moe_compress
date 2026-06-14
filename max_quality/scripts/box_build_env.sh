#!/usr/bin/env bash
# CUDA-12.8 H200 env build for the s234 ablation (proven pins).
# Idempotent-ish; FAILS LOUD. Run on the box from the repo root with REPO=/root/work/repo.
set -euo pipefail

REPO="${REPO:-/root/work/repo}"
VENV="${VENV:-/root/work/venv}"
cd "${REPO}"

echo "=== [build] apt prereqs (venv, dev headers, ninja) ==="
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
apt-get update -qq
apt-get install -y -qq python3.12-venv python3.12-dev ninja-build build-essential git

echo "=== [build] venv ${VENV} ==="
python3 -m venv "${VENV}"
PY="${VENV}/bin/python"
"${PY}" -m pip install -q --upgrade pip wheel setuptools

echo "=== [build] torch 2.11.0 + cu128 (nvcc-major must match = 12) ==="
"${PY}" -m pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== [build] pipeline requirements (NOTE: tilelang/kernels overridden below) ==="
# requirements pins torch==2.11.0 (satisfied) + tilelang>=0.1.7 (would pull the
# BAD 0.1.11 -> SIGABRT) + kernels>=0.14.0 (breaks bf16 transformers import).
# Install reqs, then HARD-override tilelang + tvm-ffi to the proven CUDA-12 pins
# and uninstall kernels.
"${PY}" -m pip install -q -r requirements.txt

echo "=== [build] proven CUDA-12 GDN pins (decisive: apache-tvm-ffi==0.1.9) ==="
"${PY}" -m pip install -q --force-reinstall --no-deps \
    tilelang==0.1.7.post3 apache-tvm-ffi==0.1.9

# Uninstall kernels BEFORE setup_gpu_env.sh: that script's verify gate imports
# transformers (via fla), and kernels>=0.14 makes transformers' hub_kernels.py
# raise "Either a revision or a version must be specified" → the gate trips on a
# still-present kernels. Order matters (learned on the box 2026-06-14).
echo "=== [build] uninstall kernels (breaks bf16 transformers import) ==="
"${PY}" -m pip uninstall -y kernels || true

echo "=== [build] fast-path kernels (causal-conv1d compile + fla git) via canonical script ==="
# setup_gpu_env.sh: apt headers, nvcc==torch major gate, causal-conv1d build,
# fla from git, fast-path verification gate.
bash scripts/setup_gpu_env.sh "${VENV}"

echo "=== [build] freeze ==="
"${PY}" -m pip list 2>/dev/null | grep -iE "^torch |tilelang|apache-tvm-ffi|flash-linear|causal-conv1d|triton |transformers " || true

echo "=== [build] verify gate (torch+cuda+tilelang import) ==="
"${PY}" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0), "ngpu", torch.cuda.device_count())
assert torch.version.cuda.startswith("12.8"), torch.version.cuda
import tilelang; print("tilelang import OK", tilelang.__version__)
import fla; print("fla import OK")
import causal_conv1d; print("causal_conv1d import OK")
PYEOF
echo "BUILD_ENV_DONE rc=0"
