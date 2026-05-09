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
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/hub \
    PYTORCH_ALLOC_CONF=expandable_segments:True

# Python 3.11 + git (for runtime code clone) + build tooling for compiled wheels
# (flash-linear-attention and causal-conv1d build CUDA kernels — need ninja + nvcc;
# the cuda:12.6.3-devel base provides nvcc).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools packaging

# torch first against the cu124 wheel index — separate from requirements.txt so
# we don't accidentally pull a CPU-only build via the default PyPI resolver.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        "torch>=2.5.0,<2.11.0"

# Bulk install of pinned deps. requirements.txt also lists torch (for reference);
# pip will see it's already satisfied and skip.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Compiled fast-path kernels need --no-build-isolation so they can see the
# already-installed torch + cuda toolkit during their setup.py.
# Re-pinned here in case a base image already had stale versions.
RUN pip install --no-build-isolation \
        "causal-conv1d>=1.4.0" \
        "flash-linear-attention>=0.1.2"

WORKDIR /workspace

COPY docker/bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh

# Default mountpoint for persistent storage (model snapshot, ablation artifacts,
# cloned code). On vast.ai mount as `-v /workspace/cache:/cache`.
VOLUME ["/cache"]

ENTRYPOINT ["/usr/local/bin/bootstrap.sh"]
