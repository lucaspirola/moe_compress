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
#       bash -c "apt-get update -qq && apt-get install -y -qq curl ca-certificates && curl -sL https://raw.githubusercontent.com/lucaspirola/moe_compress/main/max_quality/scripts/hf_jobs_build_patched_vllm.sh | bash"
#
# NOTE: the base nvidia/cuda:*-ubuntu24.04 image does NOT ship with curl,
# so the bootstrap MUST apt-install it BEFORE the curl|bash pipe — otherwise
# the job fails at line 1 with "curl: command not found". The script's own
# Phase 1 installs the rest of the build prereqs (curl included, idempotent).

set -e
set -o pipefail

echo "[$(date)] === Phase 1: install base packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    git python3 python3-pip python3-venv python3-dev python3.12-dev \
    build-essential ninja-build cmake curl ca-certificates

echo "[$(date)] === Phase 2: make venv + install build prerequisites ==="
python3 -m venv /tmp/venv
# shellcheck disable=SC1091
. /tmp/venv/bin/activate
# Two-pronged license fix:
#   - Use setuptools>=77 (accepts PEP 639 license-files field in pyproject)
#   - Rewrite SPDX-string license to dict form in Phase 5b below (setuptools>=77 rejects it)
# An older setuptools<77 would accept the SPDX string but reject license-files.
pip install --quiet --upgrade pip wheel "setuptools>=77"
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

echo "[$(date)] === Phase 5: fetch and apply BOTH calibration patches ==="
# Two patches in this wheel:
#  1. vllm_calibration_hooks.patch (10846 lines, MD5 c19ffcc5...) — the
#     CUDA-graph-safe in-graph capture rearchitecture: calibration_hooks.py
#     (GPU accumulator dicts, custom-op import, NO Python callback registry),
#     calibration_custom_ops.py (the 3 named-arg mutates_args custom ops:
#     calib_imatrix_dense_accum / calib_block_out_accum / calib_layer_in_accum),
#     the 10 writer modules rewritten to GPU accumulators + in-graph ops
#     (block_outputs / imatrix / input_cov / output_reservoir / per_expert_max /
#     reap_scores / router_logits_stats / routing_stats / wanda_scalar_row),
#     the envs.py mirror (+ new VLLM_CALIB_INPUT_COV_MODE / BUF_ROWS_FLOOR /
#     STAGE2_PROFILE_MODE / WANDA gate), the MoERunner + TritonExperts interior
#     accumulation, and the qwen3_moe/qwen3_next/linear/logits_processor
#     custom-op call sites. Every writer keeps a public captured_entry_count().
#  2. vllm_calibration_stage2_profile.patch (941 lines, MD5 fca4c01c...) —
#     creates calibration_stage2_profile.py, now REPLAY-ONLY by default
#     (VLLM_CALIB_STAGE2_PROFILE_MODE=replay): no live callbacks are registered
#     (the graph-unsafe callback registry was removed); the legacy "live" mode
#     raises NotImplementedError. The profile is reconstructed post-capture by
#     the feat/calib-v3-replay forward-only driver.
# Both patches are needed; #2 creates a file #1 does not.
# See max_quality/patches/MANIFEST.md.

curl -sL \
    https://raw.githubusercontent.com/lucaspirola/moe_compress/main/max_quality/patches/vllm_calibration_hooks.patch \
    -o /tmp/calib.patch
wc -l /tmp/calib.patch
md5sum /tmp/calib.patch
# Expected: 10846 lines, MD5 c19ffcc5d03fec7dc20caebfad80d7c1
# (In-graph cudagraph-safe capture + the fused grouped-SYRK input_cov kernel:
#  calibration_custom_ops.py, GPU-accumulator rewrite of all 10 writers,
#  named-arg mutates_args custom ops, AND csrc/calibration/gram_grouped_accum.cu
#  + a CMakeLists VLLM_EXT_SRC append. See MANIFEST.md.)

curl -sL \
    https://raw.githubusercontent.com/lucaspirola/moe_compress/main/max_quality/patches/vllm_calibration_stage2_profile.patch \
    -o /tmp/calib2.patch
wc -l /tmp/calib2.patch
md5sum /tmp/calib2.patch
# Expected: 941 lines, MD5 fca4c01cc5cef188b30c323bb919cd72
# (In-graph rearchitecture: stage2_profile is now replay-only by default;
#  the legacy live CPU-callback path raises NotImplementedError under
#  VLLM_CALIB_STAGE2_PROFILE_MODE=live.)

git apply --check /tmp/calib.patch
git apply /tmp/calib.patch
git apply --check /tmp/calib2.patch
git apply /tmp/calib2.patch
echo "Applied both. Status:"
git status --short

# Build-hygiene gate: confirm the in-graph capture rearchitecture landed.
# Under the cudagraph-safe rearchitecture the Python ``_ch.dispatch`` callback
# path is REMOVED; the qwen3_next.py block hook now calls the
# ``torch.ops.vllm.calib_block_out_accum`` custom op instead. The custom-op
# registration module + the block hook custom-op call must both be present.
echo "[$(date)] === Phase 5 gate: verify in-graph capture hooks applied ==="
grep -q 'def captured_entry_count' vllm/calibration_block_outputs.py \
    || { echo "FATAL: captured_entry_count() missing from calibration_block_outputs.py"; exit 1; }
test -f vllm/calibration_custom_ops.py \
    || { echo "FATAL: calibration_custom_ops.py missing (in-graph custom ops)"; exit 1; }
grep -q 'calib_block_out_accum' vllm/model_executor/models/qwen3_next.py \
    || { echo "FATAL: in-graph block hook (calib_block_out_accum) missing from qwen3_next.py"; exit 1; }
echo "In-graph capture gate OK (custom_ops + block hook present)."

# Fused grouped-SYRK input_cov kernel gate: the .cu and its CMakeLists
# registration must both have landed from patch 1, else the wheel builds
# without torch.ops.calib_gram.gram_grouped_accum and input_cov capture aborts
# at the first forward.
echo "[$(date)] === Phase 5 gate: verify grouped-SYRK kernel applied ==="
test -f csrc/calibration/gram_grouped_accum.cu \
    || { echo "FATAL: csrc/calibration/gram_grouped_accum.cu missing (input_cov kernel)"; exit 1; }
grep -q 'csrc/calibration/gram_grouped_accum.cu' CMakeLists.txt \
    || { echo "FATAL: gram_grouped_accum.cu not registered in CMakeLists.txt VLLM_EXT_SRC"; exit 1; }
grep -q 'TORCH_LIBRARY(calib_gram' csrc/calibration/gram_grouped_accum.cu \
    || { echo "FATAL: calib_gram op registration missing from the kernel"; exit 1; }
echo "grouped-SYRK kernel gate OK (.cu + CMakeLists + op registration present)."

# down_proj input_cov gate: the SECOND grouped-SYRK call (post-SwiGLU
# intermediate -> down_proj input cov) must have landed -- the down accumulator
# globals in calibration_hooks.py AND the in-graph SYRK call inside
# TritonExperts.apply. Without these the wheel captures gate_proj only.
echo "[$(date)] === Phase 5 gate: verify down_proj input_cov SYRK applied ==="
grep -q '_INPUT_COV_DOWN_GPU' vllm/calibration_hooks.py \
    || { echo "FATAL: _INPUT_COV_DOWN_GPU missing from calibration_hooks.py"; exit 1; }
grep -q '_CAPTURE_INPUT_COV_DOWN' \
    vllm/model_executor/layers/fused_moe/experts/triton_moe.py \
    || { echo "FATAL: down_proj SYRK (CAPTURE_INPUT_COV_DOWN) missing from triton_moe.py"; exit 1; }
echo "down_proj input_cov gate OK (down accumulator + in-graph SYRK present)."

echo "[$(date)] === Phase 5b: strip license + license-files lines from pyproject.toml ==="
# vLLM 0.21.0 has BOTH `license = "Apache-2.0"` (SPDX-string, deprecated)
# AND `license-files = [...]` (PEP 639). The schema validator in the
# build chain rejects various combinations regardless of which setuptools
# version we use. Since we don't need license metadata in the wheel to
# install + run it, just strip both lines. The Apache-2.0 license still
# applies via the LICENSE file in the source.
python3 - <<'PYEOF'
import re, pathlib
p = pathlib.Path("pyproject.toml")
src = p.read_text()
# Strip the single-line license = "..." form
new = re.sub(r'^license\s*=\s*".+?"\s*$\n?', '', src, flags=re.MULTILINE)
# Strip the multi-line license-files = [...] block (handles list form across lines)
new = re.sub(
    r'^license-files\s*=\s*\[[^\]]*\]\s*$\n?',
    '',
    new,
    flags=re.MULTILINE | re.DOTALL,
)
# Also handle inline list `license-files = ["..."]`
new = re.sub(r'^license-files\s*=\s*\[.*?\]\s*$\n?', '', new, flags=re.MULTILINE)
if new != src:
    p.write_text(new)
    print("pyproject.toml: license + license-files stripped")
else:
    print("pyproject.toml: no license lines found (already clean)")
PYEOF
echo "Remaining license refs in pyproject.toml:"
grep -nE "^license" pyproject.toml || echo "(none — clean)"

echo "[$(date)] === Phase 6: build multi-arch wheel ==="
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:${PATH}
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0a;10.0;12.0}"
export MAX_JOBS=16
export NVCC_THREADS=2
nvcc --version | tail -3
echo "CPU count: $(nproc)"
echo "Memory: $(free -h | head -2)"

echo "[$(date)] === Phase 6a: manual cmake configure (to see actual cmake errors) ==="
# pip wheel hides cmake stdout/stderr behind a Python traceback. Run cmake
# manually first to capture the real configure errors in the build log.
mkdir -p /tmp/cmake-test
cmake -S /tmp/vllm-patched -B /tmp/cmake-test \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DVLLM_TARGET_DEVICE=cuda \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    -DVLLM_PYTHON_EXECUTABLE=/tmp/venv/bin/python3 \
    2>&1 | tee /tmp/cmake_configure.log
CMAKE_RC=${PIPESTATUS[0]}
echo "cmake configure exit code: ${CMAKE_RC}"
if [ ${CMAKE_RC} -ne 0 ]; then
    echo "[$(date)] === cmake configure FAILED; aborting before pip wheel ==="
    echo "Last 40 lines of cmake log:"
    tail -40 /tmp/cmake_configure.log
    exit 1
fi

echo "[$(date)] === Phase 6b: pip wheel with --verbose ==="
mkdir -p /tmp/wheels
# --verbose keeps subprocess output streaming (no buffering).
pip wheel . --no-deps -w /tmp/wheels --no-build-isolation --verbose 2>&1
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

# Also upload both patches themselves for traceability
for src_path, dest_name in [
    ("/tmp/calib.patch", "vllm_calibration_hooks.patch"),
    ("/tmp/calib2.patch", "vllm_calibration_stage2_profile.patch"),
]:
    if os.path.exists(src_path):
        print(f"Uploading patch artifact {dest_name}...")
        upload_file(
            path_or_fileobj=src_path,
            path_in_repo=dest_name,
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

In-graph cudagraph-safe capture (this build): every capture signal now
accumulates via pure GPU tensor ops recorded INTO the captured graph (the prior
Python `_ch.dispatch` callback path yielded zero under cudagraphs, the
production default). Traced-region signals (imatrix dense / block_out /
layer_in) use named-arg `mutates_args` custom ops in `calibration_custom_ops.py`
so Inductor cannot DCE them; custom-op-interior signals accumulate inline in
MoERunner / TritonExperts. All 10 writers keep public `captured_entry_count()`
for the driver fail-fast and dump from GPU accumulators (checkpoint schema v2).
Runs cudagraph-fast (NOT enforce_eager).

- Source repo: https://github.com/lucaspirola/moe_compress (branch `feat/b0-hook-fix`)
- Patch artifacts (also uploaded to this repo for traceability):
  - `vllm_calibration_hooks.patch` — 10846 lines, MD5 `c19ffcc5d03fec7dc20caebfad80d7c1`
  - `vllm_calibration_stage2_profile.patch` — 941 lines, MD5 `fca4c01cc5cef188b30c323bb919cd72`
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
- `VLLM_CALIB_CAPTURE_INPUT_COV=1`  — per-(layer, expert, "gate_proj") teacher input covariance Σ_in (requires `VLLM_CALIB_CAPTURE_EXPERT=1`; writes dict-shaped `sidecars/covariance.pt`, schema v2)
- `VLLM_CALIB_MAX_LAYER=<N>`        — L2 early-exit gate: truncate `Qwen3MoeModel.forward` after decoder layer `N` (inclusive); skips `N+1..end`. Default `-1` / unset = disabled. Orthogonal to the capture gates above. Useful as the foundation for L1 (sequential REAP+REAM per-layer profiling) and as a standalone optimisation for any writer whose payload comes from layer `L` or earlier.

## Spot-preemption resumability

The driver `build_self_traces_calib_vllm.py` writes a periodic
`<jsonl>.imatrix.ckpt` checkpoint at every chunk boundary (CLI:
`--imatrix-checkpoint-every-chunks=1` by default). On `--resume`, the
checkpoint is hydrated into the live accumulators in-place and the
cumulative prompt counter is restored. The final `.imatrix.dat` and
the periodic `.imatrix.ckpt` both use the temp-file + `os.replace`
atomic-rename pattern so a kill mid-write leaves the previous file
intact. `.npz` logit sidecars are also written atomically.

JSONL resume is hardened against trailing partial lines: each line
is JSON-validated on resume; the first parse failure triggers a
truncate to the last good byte offset before counting resumes.
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
