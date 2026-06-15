#!/usr/bin/env bash
# Download the s234-ablation artifacts from HF onto the box. Uses SYSTEM python
# (+ ensurepip) so it can run IN PARALLEL with the venv build. Authoritative
# torch-based CONTENT verify of the covariance is box_verify_cov.py (post-build).
# Layout produced (consumed by assert_covariance_resolves / prep_box_config):
#   ${BASE}/models/Qwen3.6-35B-A3B/                      (teacher+student weights)
#   ${BASE}/run/self_traces_489ee0e1b17b43b0.jsonl       (replay source)
#   ${BASE}/run/sidecars/self_traces_489ee0e1b17b43b0/covariance.pt (91GB) + small sidecars
set -euo pipefail

BASE="${BASE:-/root/work}"
STEM="self_traces_489ee0e1b17b43b0"
DS="pirola/calib-v2-self-traces"
MODEL="Qwen/Qwen3.6-35B-A3B"

# System python + pip. The base CUDA image has python3 but NO pip, and its
# ensurepip is stripped — so bootstrap pip via apt before any `python3 -m pip`.
# (Lock::Timeout waits out the parallel venv-build's apt lock instead of failing.)
# Caught live on box 41023933 (2026-06-15): without this the first `python3 -m
# pip` died with "No module named pip" and the download aborted under set -e.
export HF_HUB_ENABLE_HF_TRANSFER=1 DEBIAN_FRONTEND=noninteractive
if ! python3 -m pip --version >/dev/null 2>&1; then
    apt-get -o DPkg::Lock::Timeout=300 update -qq
    apt-get -o DPkg::Lock::Timeout=300 install -y -qq python3-pip
fi
PIP="python3 -m pip install -q --break-system-packages"   # PEP 668 base image
${PIP} huggingface_hub hf_transfer

mkdir -p "${BASE}/models" "${BASE}/run"

# Stable Python API (hf_hub 1.19 dropped the commands.huggingface_cli module path).
BASE="${BASE}" STEM="${STEM}" DS="${DS}" MODEL="${MODEL}" python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
base, stem, ds, model = os.environ["BASE"], os.environ["STEM"], os.environ["DS"], os.environ["MODEL"]
print(f"=== [dl] model {model} ===", flush=True)
snapshot_download(model, local_dir=f"{base}/models/Qwen3.6-35B-A3B", max_workers=16)
print("=== [dl] jsonl ===", flush=True)
snapshot_download(ds, repo_type="dataset", local_dir=f"{base}/run",
                  allow_patterns=[f"{stem}.jsonl"], max_workers=16)
print("=== [dl] sidecars (covariance.pt 91GB + small .pt + manifests; NOT block_hidden) ===", flush=True)
snapshot_download(ds, repo_type="dataset", local_dir=f"{base}/run",
                  allow_patterns=[f"sidecars/{stem}/*.pt", f"sidecars/{stem}/*.MANIFEST.json"],
                  max_workers=16)
print("[dl] snapshot_download all OK", flush=True)
PYEOF

COV="${BASE}/run/sidecars/${STEM}/covariance.pt"
[[ -f "${COV}" ]] || { echo "[dl] FATAL covariance.pt missing"; exit 1; }
SZ=$(stat -c %s "${COV}")
echo "[dl] covariance.pt present, $((SZ/1024/1024/1024)) GiB (content verify = box_verify_cov.py post-build)"
# Stale-manifest guard (prior 85.9GB-vs-91.3GB incident): drop the cov manifest
# so the manifested loader self-checks the in-payload content.
rm -f "${COV}.MANIFEST.json" && echo "[dl] dropped covariance.pt.MANIFEST.json"
echo "DOWNLOAD_ARTIFACTS_DONE rc=0"
