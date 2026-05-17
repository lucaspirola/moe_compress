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

# Compiled fast-path kernels need --no-build-isolation so they can see the
# already-installed torch + cuda toolkit during their setup.py.
# Re-pinned here in case a base image already had stale versions.
RUN pip install --no-build-isolation -c /tmp/torch-constraint.txt \
        "causal-conv1d>=1.4.0" \
        "flash-linear-attention>=0.1.2"

# Pre-bake the FP8 KD-teacher kernels (deep-gemm + finegrained-fp8) into the
# image. A fresh container otherwise fetches ~1200 small deep-gemm files from
# HF at the first FP8 forward — that request burst trips HF's 429 rate limiter
# and stalls Stage 2.5 for 20-40 min. Baked into /opt/kernels-hub (NOT /cache,
# which is a VOLUME that would shadow anything baked under it); bootstrap.sh /
# run_strategy_sweep.sh seed the runtime HF cache from this copy. metadata.json
# is self-healed here (kernels>=0.14.0 strict-validates name/id). The build
# variant resolves from the installed torch (no GPU needed at fetch time).
RUN HF_HOME=/opt/kernels-hub python3 - <<'PYEOF'
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
