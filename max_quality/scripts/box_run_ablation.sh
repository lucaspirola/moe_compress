#!/usr/bin/env bash
# The real s234 ablation on the box: reap-s234 (resumes HF stage2p5_final, Stage
# 3->6) + ream-s234 (Stage 2->6). whitening_cov=shift (paper-fidelity acov).
# SINGLE-H200 strategy (--num-gpus 1): the 35B student + single-pass CPU-accum cov
# fit one card, so the 1-GPU path injects NO multi-GPU overlay (byte-identical) and
# is far cheaper than the paused 2-GPU 4000-seq run. No arm runs Stage 1 (_shared/
# is the gen_shared stub).
#
# REQUIRES box_reap.yaml prepped WITH the Stage-3 cov optimizations ON (default-OFF
# in the library), e.g.:
#   python scripts/prep_box_config.py configs/qwen36_35b_a3b_reap_faithful.yaml \
#     "${BASE}/box_reap.yaml" --base "${BASE}" \
#     --single-pass-cov --cov-num-sequences 512 --spectra-workers "$(nproc)"
# (do NOT pass --shard: single GPU).
set -uo pipefail

BASE="${BASE:-/root/work}"
REPO="${REPO:-/root/work/repo}"
VENV="${VENV:-/root/work/venv}"
PY="${VENV}/bin/python"
MODEL="${BASE}/models/Qwen3.6-35B-A3B"
CFG="${BASE}/box_reap.yaml"
PROBE="${BASE}/probe"           # already holds _shared/ (gen_shared stub)
export PYTHONPATH="${REPO}/src"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export HF_HUB_DISABLE_XET=1      # xet writer error on the 52GB seed (2026-06-14)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"

echo "=================== s234 ABLATION (whitening=shift, 1-GPU single-pass) ==================="
"${PY}" -u -m moe_compress.run_reap_ream_35pct \
  --config "${CFG}" --model "${MODEL}" \
  --probe-root "${PROBE}" \
  --only reap-s234,ream-s234 \
  --whitening-cov shift --num-gpus 1 --num-sequences 512
RC=$?
echo "ABLATION_RC=${RC}"

echo "=================== RESULT ==================="
"${PY}" - "${PROBE}/reap-s234/stage6alt_eval.json" "${PROBE}/ream-s234/stage6alt_eval.json" <<'PYEOF'
import json, sys
def bpt(p):
    try:
        d = json.load(open(p))
        for k in ("student_bpt", "bpt", "student_bits_per_token"):
            if k in d: return d[k]
        return d
    except Exception as e:
        return f"<missing: {e}>"
print(f"reap-s234 student_bpt = {bpt(sys.argv[1])}")
print(f"ream-s234 student_bpt = {bpt(sys.argv[2])}")
print("baselines: reap-rkd=3.1686 / ream-rkd=3.1839 (lower=better)")
PYEOF
echo "ABLATION_DONE RC=${RC}"
