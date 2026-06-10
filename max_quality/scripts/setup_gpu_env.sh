#!/usr/bin/env bash
# Canonical fast-path GPU environment setup for moe_compress (Qwen3.6-35B-A3B).
#
# WHY THIS EXISTS
# --------------
# Qwen3.6 is a hybrid GDN (Gated DeltaNet) + fused-MoE model. Two independent
# things must be true for it to run *correctly and fast* on a GPU:
#
#   1. MoE path: transformers calls `torch._grouped_mm`, which only exists on
#      Hopper (sm_90) and — with torch>=2.11 — Blackwell (sm_120). torch 2.8
#      raises "grouped_mm is only supported on CUDA devices with compute
#      capability = 9.0" on Blackwell. => torch MUST be >= 2.11.
#
#   2. GDN fast path: transformers gates it ALL-OR-NOTHING on BOTH
#      `flash-linear-attention` AND `causal-conv1d` being importable
#      (modeling_qwen3_5_moe.py: `is_fast_path_available = all((...))`).
#      Missing either => slow native chunked fallback (GPU util ~0-46%,
#      CPU-bound python chunk loop). With both => fla kernel, GPU util ~100%.
#
# THE THREE TRAPS THIS SCRIPT REMOVES (each cost real GPU-hours to rediscover):
#   T1. causal-conv1d is a CUDA C++ extension. Its build fails with
#       "fatal error: Python.h: No such file" without python3-dev, and silently
#       falls back to the slow distutils backend without ninja. INSTALL BOTH.
#   T2. causal-conv1d's setup.py HARD-ERRORS if nvcc's CUDA major != the CUDA
#       major torch was built against ("detected CUDA version mismatches ...").
#       => nvcc and torch must agree (e.g. CUDA-13 base + torch cu130).
#   T3. pip's newest flash-linear-attention is 0.5.0, which crashes on torch
#       2.11 ("module 'torch.cpu' has no attribute 'device'"). The fix is only
#       on git (>= 0.5.1). INSTALL FLA FROM GIT, NOT PYPI.
#   T-bonus. A base image's torchvision/torchaudio pinned to an older torch
#       breaks `import transformers` ("operator torchvision::nms does not
#       exist") after a torch upgrade. We don't need them — remove on mismatch.
#
# The script is IDEMPOTENT and FAILS LOUD: it ends with a hard verification
# gate that asserts the full fast path is live, so a broken env is caught here
# (seconds) instead of mid-run (hours, $$$).
#
# USAGE
#   max_quality/scripts/setup_gpu_env.sh [/path/to/venv]
# If a venv path is given (or $VIRTUAL_ENV is set) its python is used; else the
# system python3. Run as root (or with sudo) so apt can install build deps.
#
# Companion to docker/Dockerfile (which bakes the same recipe into an image).

set -euo pipefail

# --------------------------------------------------------------------------
# Resolve the target python interpreter
# --------------------------------------------------------------------------
VENV="${1:-${VIRTUAL_ENV:-}}"
if [[ -n "${VENV}" && -x "${VENV}/bin/python" ]]; then
    PY="${VENV}/bin/python"
else
    PY="$(command -v python3 || command -v python || true)"
fi
[[ -n "${PY}" ]] || { echo "[setup] FATAL: no python3/python found and no venv given (arg 1 or \$VIRTUAL_ENV)."; exit 1; }
echo "[setup] target python: ${PY}"
PYVER="$(${PY} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

# --------------------------------------------------------------------------
# T1: system build prerequisites (python headers + ninja + nvcc)
# --------------------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
    echo "[setup] installing build prerequisites (python${PYVER}-dev, ninja, build-essential)"
    export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
    apt-get update -qq
    # python<ver>-dev provides Python.h; fall back to python3-dev if the
    # versioned package name is unavailable.
    apt-get install -y -qq "python${PYVER}-dev" ninja-build build-essential \
        || apt-get install -y -qq python3-dev ninja-build build-essential
else
    echo "[setup] WARNING: apt-get not found; ensure python headers + ninja + a C++ toolchain exist"
fi

# nvcc must be present to compile causal-conv1d. Prefer the /usr/local/cuda
# symlink (the host's active toolkit); else fall back to the HIGHEST installed
# cuda-<ver> so a stale CUDA-12 alongside CUDA-13 doesn't get picked.
if ! command -v nvcc >/dev/null 2>&1; then
    for c in /usr/local/cuda/bin $(ls -d /usr/local/cuda-*/bin 2>/dev/null | sort -t- -k2 -Vr); do
        [[ -x "${c}/nvcc" ]] && export PATH="${c}:${PATH}" && break
    done
fi
command -v nvcc >/dev/null 2>&1 || { echo "[setup] FATAL: nvcc not found. Use a CUDA -devel base image (e.g. nvidia/cuda:13.0.3-cudnn-devel)."; exit 1; }

# --------------------------------------------------------------------------
# T2: enforce nvcc CUDA major == torch CUDA major (causal-conv1d build gate)
# --------------------------------------------------------------------------
"${PY}" - "$(nvcc --version)" <<'PYEOF'
import re, sys
import torch
nvcc_text = sys.argv[1]
m = re.search(r"release (\d+)\.(\d+)", nvcc_text)
if not m:
    print("[setup] FATAL: could not parse nvcc version"); sys.exit(1)
nvcc_major = int(m.group(1))
tv = torch.version.cuda          # e.g. "13.0"
if tv is None:
    print("[setup] FATAL: this torch is CPU-only (torch.version.cuda is None)."); sys.exit(1)
torch_major = int(tv.split(".")[0])
print(f"[setup] nvcc CUDA major={nvcc_major}  torch CUDA major={torch_major}  torch={torch.__version__}")
# torch >= 2.11 required for Blackwell grouped_mm; warn (not fatal) on Hopper.
tmaj, tmin = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if (tmaj, tmin) < (2, 11):
    print(f"[setup] FATAL: torch {torch.__version__} < 2.11 — MoE grouped_mm will fail on Blackwell. "
          f"Reinstall: pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu{torch_major}0")
    sys.exit(1)
if nvcc_major != torch_major:
    print(f"[setup] FATAL: nvcc CUDA {nvcc_major} != torch CUDA {torch_major}. "
          f"causal-conv1d will refuse to build. Fix by matching them: either install a CUDA-{torch_major} "
          f"toolkit, or reinstall torch built for CUDA {nvcc_major} "
          f"(pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu{nvcc_major}0).")
    sys.exit(1)
PYEOF

# --------------------------------------------------------------------------
# T-bonus: drop torchvision/torchaudio if they pin a different torch than the
# one installed (they break `import transformers` with "operator
# torchvision::nms does not exist"; the pipeline does not use them).
# --------------------------------------------------------------------------
MISMATCHED="$(${PY} - <<'PYEOF'
import importlib.metadata as md
try:
    torch_v = md.version("torch")            # e.g. "2.11.0+cu130"
except md.PackageNotFoundError:
    raise SystemExit
# Compare on the public release (drop the +cuXXX local segment): torchvision's
# requirement is "torch==2.11.0" with NO local part, so a raw string/endswith
# compare against "2.11.0+cu130" would wrongly flag a MATCHING install.
try:
    from packaging.requirements import Requirement
    from packaging.version import Version
    torch_rel = Version(torch_v).release            # (2, 11, 0)
    def pinned_mismatch(reqs):
        for r in reqs:
            try:
                req = Requirement(r)
            except Exception:
                continue
            if req.name.lower() != "torch":
                continue
            for spec in req.specifier:
                if spec.operator in ("==", "===", "~="):
                    try:
                        spec_rel = Version(spec.version).release
                    except Exception:
                        continue  # unparseable specifier — don't assume mismatch (stay fail-safe)
                    n = min(len(spec_rel), len(torch_rel))
                    if spec_rel[:n] != torch_rel[:n]:
                        return True
        return False
except Exception:
    # packaging missing (shouldn't happen — pip depends on it). Conservative
    # base-version string compare as a fallback.
    torch_base = torch_v.split("+", 1)[0]
    def pinned_mismatch(reqs):
        import re
        for r in reqs:
            m = re.match(r"\s*torch\s*==\s*([0-9][0-9A-Za-z.\-+!]*)", r)
            if m and m.group(1).split("+", 1)[0] != torch_base:
                return True
        return False
out = []
for pkg in ("torchvision", "torchaudio"):
    try:
        reqs = md.requires(pkg) or []
    except md.PackageNotFoundError:
        continue
    if pinned_mismatch(reqs):
        out.append(pkg)
print(" ".join(out))
PYEOF
)"
if [[ -n "${MISMATCHED// }" ]]; then
    echo "[setup] removing torch-mismatched (would break transformers import): ${MISMATCHED}"
    "${PY}" -m pip uninstall -y ${MISMATCHED} || true
fi

# --------------------------------------------------------------------------
# Install the fast-path kernels
# --------------------------------------------------------------------------
echo "[setup] installing ninja (build backend) into the venv"
"${PY}" -m pip install -q ninja

echo "[setup] T1/T2: building causal-conv1d (CUDA extension, --no-build-isolation)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}" \
    "${PY}" -m pip install -q --no-build-isolation "causal-conv1d>=1.4.0"

echo "[setup] T3: installing flash-linear-attention from git (pip 0.5.0 crashes on torch 2.11)"
"${PY}" -m pip install -q --no-deps \
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"

# --------------------------------------------------------------------------
# HARD VERIFICATION GATE — fail loud here, not mid-run
# --------------------------------------------------------------------------
echo "[setup] verifying fast path ..."
"${PY}" - <<'PYEOF'
import sys
import torch
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
# MoE grouped_mm support: Hopper (9,0) or Blackwell (12,0)+ with torch>=2.11
print(f"[verify] device={name} capability={cap} torch={torch.__version__} cuda={torch.version.cuda}")

import causal_conv1d  # noqa: F401
import fla            # noqa: F401
print(f"[verify] causal_conv1d OK | fla {fla.__version__}")

# exercise the exact fla call that crashed on the buggy 0.5.0 / torch 2.11
from fla.utils import custom_device_ctx
with custom_device_ctx(0):
    pass
print("[verify] fla.custom_device_ctx OK (no torch.cpu.device bug)")

from transformers.utils.import_utils import (
    is_flash_linear_attention_available as _fla,
    is_causal_conv1d_available as _cc,
)
assert _fla(), "transformers does not see flash-linear-attention"
assert _cc(), "transformers does not see causal-conv1d"
print("[verify] transformers fast-path: fla=True causal=True")

# Training-readiness is SEPARATE from the inference fast path above. The gated
# DeltaNet BACKWARD (chunk_bwd_dqkwg) RAISES on Hopper + Triton>=3.4 without
# tilelang (fla PR #827 / issue #640). This script does NOT install tilelang —
# it is un-installable on CUDA-13 (apache-tvm-ffi double-registers the tvm-ffi
# runtime -> SIGABRT). So warn LOUDLY: this env can do forward/inference, but
# Router-KD TRAINING (Stage 2.5/5) will crash at loss.backward(). Train on a
# CUDA-12 box + tilelang. (Non-fatal: inference-only setups are valid here.)
import importlib.util as _ilu
import triton as _tr
from packaging.version import Version as _V
_hopper = tuple(torch.cuda.get_device_capability(0)) == (9, 0)
_tri_ge_34 = _V(_tr.__version__) >= _V("3.4.0")
_tilelang = _ilu.find_spec("tilelang") is not None
if _hopper and _tri_ge_34 and not _tilelang:
    print("[verify] ##############################################################")
    print("[verify] ## TRAINING WARNING: Hopper + Triton>=3.4 + NO tilelang.    ##")
    print("[verify] ## Router-KD TRAINING (Stage 2.5/5) WILL CRASH at backward  ##")
    print("[verify] ## (fla #827). tilelang cannot install on CUDA-13. Forward/ ##")
    print("[verify] ## inference is fine; train on a CUDA-12 box + tilelang.    ##")
    print("[verify] ##############################################################")
    print("TRAINING_BACKWARD_UNSUPPORTED")
else:
    print("[verify] training backward: OK (tilelang present, or off-Hopper)")

print("FAST_PATH_READY")
PYEOF

echo "[setup] DONE — fast path verified. GDN runs the fla kernel (GPU-saturating), MoE uses grouped_mm."
