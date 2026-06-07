# syntax=docker/dockerfile:1
# moe_compress portable runtime image.
#
# Targets vast.ai / RunPod / Lambda / any GPU host that supports `docker run --gpus all`.
# Companion to hf_jobs/entrypoint_ablations.py (HF Jobs path) — both consume
# the same canonical max_quality/requirements.txt dep list.
#
# Build context: max_quality/  (so requirements.txt + docker/bootstrap.sh are at the root)
# Build:   docker build -f max_quality/docker/Dockerfile -t moe-compress:dev max_quality/
# Push:    handled by .github/workflows/docker-build.yml → ghcr.io/lucaspirola/moe-compress
#
# Image carries dependencies only. Code is git-cloned at container start by
# bootstrap.sh; model snapshot is downloaded into the persistent /cache volume.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/hub \
    PYTORCH_ALLOC_CONF=expandable_segments:True

# Ubuntu 24.04 ships Python 3.12 (final release — no BlockFinder/inspect.py
# RC bugs). git for runtime code clone; build tooling for compiled wheels
# (flash-linear-attention and causal-conv1d build CUDA kernels — need ninja +
# nvcc; the cuda:13.0.3-devel base provides nvcc with sm_100 codegen support).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        git \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools packaging

# torch from the cu130 wheel index (matches CUDA 13.0 base; cu130 wheels
# add native sm_100 / Blackwell B200 support — see commit message + memory/
# project_grouped_mm_blackwell.md for the speed-recovery context).
# Capture the installed wheel as a constraint so transitive deps cannot
# pull a different build variant via the default PyPI index.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.11.0" \
    && pip freeze | grep '^torch==' > /tmp/torch-constraint.txt

# Bulk install of pinned deps. The constraint file locks torch to the exact
# cu130 wheel above.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -c /tmp/torch-constraint.txt

# GDN fast-path kernels. transformers gates the Qwen3.6 GDN fast path
# ALL-OR-NOTHING on BOTH of these importing (modeling_qwen3_5_moe.py:
# `is_fast_path_available = all((causal_conv1d_*, *_gated_delta_rule))`);
# missing either => slow native chunked fallback (GPU util ~0-46%). See
# memory/project_grouped_mm_blackwell.md.
#
#  - causal-conv1d: CUDA C++ extension; --no-build-isolation so it links the
#    already-installed torch + nvcc. Needs python3-dev (Python.h) + ninja
#    (both installed above) and nvcc CUDA major == torch CUDA major (the
#    cuda:13.0.3 base + cu130 torch satisfy this).
#  - flash-linear-attention: MUST come from git. PyPI's newest is 0.5.0, which
#    crashes on torch>=2.11 ("module 'torch.cpu' has no attribute 'device'");
#    the fix is git-only (>=0.5.1). Do NOT use a PyPI version pin here.
#    --no-deps: fla's runtime deps (einops, triton, transformers, ...) are
#    already pinned/installed by requirements.txt above; --no-deps stops the
#    git install from pulling unpinned/duplicate variants over them.
RUN pip install --no-build-isolation -c /tmp/torch-constraint.txt \
        "causal-conv1d>=1.4.0" \
    && pip install --no-deps \
        "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"

# Fail the build LOUDLY if the fast path is not actually live — catch a broken
# kernel/torch combo here (seconds) instead of mid-run on a paid GPU (hours).
# Build-time = no GPU, so we can't enter custom_device_ctx(0); instead we (a)
# import both kernels, (b) reject the known-bad fla 0.5.0 by version (the
# torch.cpu.device crash is call-time, not import-time, so an import alone
# would NOT catch a regression to it), and (c) assert transformers sees both.
# The full call-exercise (with custom_device_ctx(0)) runs in setup_gpu_env.sh
# on the live GPU box.
RUN python3 - <<'PYEOF'
import importlib
import fla, causal_conv1d  # noqa: F401
from packaging.version import Version
assert Version(fla.__version__) >= Version("0.5.1"), (
    f"flash-linear-attention {fla.__version__} has the torch>=2.11 "
    "custom_device_ctx crash; install >=0.5.1 from git"
)
from fla.utils import custom_device_ctx  # noqa: F401  symbol that broke on 0.5.0
from transformers.utils.import_utils import (
    is_flash_linear_attention_available as _fla,
    is_causal_conv1d_available as _cc,
)
assert _fla() and _cc(), f"fast path not importable: fla={_fla()} causal={_cc()}"
print(f"[docker] fast-path verified: fla {fla.__version__} + causal-conv1d, transformers sees both")
PYEOF

# Pre-bake the FP8 KD-teacher kernels (deep-gemm + finegrained-fp8) into the
# image. A fresh container otherwise fetches ~1200 small deep-gemm files from
# HF at the first FP8 forward — that request burst trips HF's 429 rate limiter
# and stalls Stage 2.5 for 20-40 min. Baked into /opt/kernels-hub (NOT /cache,
# which is a VOLUME that would shadow anything baked under it); bootstrap.sh /
# run_strategy_sweep.sh seed the runtime HF cache from this copy. metadata.json
# is self-healed here (kernels>=0.14.0 strict-validates name/id). The build
# variant resolves from the installed torch (no GPU needed at fetch time).
#
# The fetch is AUTHENTICATED via a BuildKit secret (id=hf_token, wired in
# .github/workflows/docker-build.yml). Unauthenticated, HF's anonymous rate
# limit makes the ~1200-file burst 429-storm for hours until the build job
# times out — that is why every build after the bake was first added failed.
# The secret is optional: if absent, HF_TOKEN is empty and the step degrades to
# the old unauthenticated behaviour (and the runtime fallback still re-fetches).
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    HF_HOME=/opt/kernels-hub python3 - <<'PYEOF'
import glob, hashlib, json
from kernels import get_kernel
for repo in ("kernels-community/deep-gemm", "kernels-community/finegrained-fp8"):
    # A transient HF 429 during the build must NOT fail the image build — the
    # runtime fallback in run_strategy_sweep.sh re-fetches when /opt/kernels-hub
    # is absent/empty. So warn and continue rather than aborting the RUN layer.
    try:
        get_kernel(repo)
        print(f"[bake] cached {repo}", flush=True)
    except Exception as exc:
        print(f"[bake] {repo} fetch failed ({exc}) — runtime will fall back", flush=True)
patched = 0
for p in glob.glob("/opt/kernels-hub/hub/kernels--*/snapshots/*/build/*/metadata.json"):
    with open(p) as f:
        m = json.load(f)
    ch = False
    if "name" not in m:
        for part in p.split("/"):
            if part.startswith("kernels--") and part.count("--") >= 2:
                m["name"] = part.split("--", 2)[2].replace("_", "-")
                ch = True
                break
    if "id" not in m:
        m["id"] = "_" + hashlib.md5(p.encode()).hexdigest()[:14]
        ch = True
    if ch:
        with open(p, "w") as f:
            json.dump(m, f, indent=2)
        patched += 1
print(f"[bake] kernels baked + {patched} metadata file(s) patched", flush=True)
PYEOF

WORKDIR /workspace

COPY docker/bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh

# Default mountpoint for persistent storage (model snapshot, ablation artifacts,
# cloned code). On vast.ai mount as `-v /workspace/cache:/cache`.
VOLUME ["/cache"]

ENTRYPOINT ["/usr/local/bin/bootstrap.sh"]
