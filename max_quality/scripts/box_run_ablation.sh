#!/usr/bin/env bash
# The real s234 ablation on the box: reap-s234 (resumes HF stage2p5_final, Stage
# 3->6) + ream-s234 (Stage 2->6). whitening_cov=shift (paper-fidelity acov).
#
# 2×H200 SHARDED strategy (--num-gpus 2 + device_map=balanced via prep --shard):
# the Stage-3 cross-cov holds BOTH 35B models resident (~126GB), which on ONE card
# leaves no GPU room → the cov SYRK falls back to CPU (~50min, GPU idle). Sharding
# the dual model across 2 cards frees room so the SYRK GEMM runs ON GPU; the 354GB
# of fp32 per-expert Grams still can't fit GPU, so KEEP cov_single_pass=true (CPU
# hot-accum — only the GEMM result migrates to CPU). This is the agent-validated
# fast path. NOTE: --num-gpus>=2 injects the RESULT-PRESERVING multi-GPU subset
# (Stage-3 SVD/alpha + Stage-4 EoRA task-parallel + Stage-2.5/5 DDP); it does NOT
# enable multi_gpu.cov_replicas (data-parallel cov is NOT bitwise-equal, excluded).
# No arm runs Stage 1 (_shared/ is the gen_shared stub).
#
# REQUIRES box_reap.yaml prepped WITH --shard (device_map=balanced) + the Stage-3
# cov optimizations ON (default-OFF in the library):
#   python scripts/prep_box_config.py configs/qwen36_35b_a3b_reap_faithful.yaml \
#     "${BASE}/box_reap.yaml" --base "${BASE}" --shard \
#     --single-pass-cov --cov-num-sequences 512 --spectra-workers "$(nproc)"
set -uo pipefail

# Raise the open-files limit. The Stage-3 spectra spawn-ProcessPool
# (stage3_svd.spectra_workers) opens many pipes/sockets per worker plus heavy
# per-worker re-imports; the base image default (ulimit -n 1024) exhausts FDs
# → "OSError: [Errno 24] Too many open files" kills the arm. Caught on box
# 41049095 (2026-06-15). Raise to the hard max (best-effort).
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true

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

# Pre-flight HARD gate: the self-traces JSONL must be domain-tagged AND the 512
# draw must preserve the calibration domain mix (else stratification silently
# no-ops to a flat draw). Abort before spending GPU on a mis-calibrated run.
JSONL="${BASE}/run/self_traces_489ee0e1b17b43b0.jsonl"
echo "=================== PRE-FLIGHT: domain-mix guardrail ==================="
"${PY}" "${REPO}/scripts/box_verify_domain_mix.py" "${JSONL}" 512 2 \
  || { echo "DOMAIN_MIX_FAIL — JSONL not domain-tagged or 512 draw drifts; aborting"; exit 1; }

echo "=================== s234 ABLATION (whitening=shift, 2-GPU sharded single-pass) ==================="
"${PY}" -u -m moe_compress.run_reap_ream_35pct \
  --config "${CFG}" --model "${MODEL}" \
  --probe-root "${PROBE}" \
  --only reap-s234,ream-s234 \
  --whitening-cov shift --num-gpus 2 --num-sequences 512
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
