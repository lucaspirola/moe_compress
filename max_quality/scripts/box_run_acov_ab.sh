#!/usr/bin/env bash
# acov A/B on the box: run the reap arm (resumes from HF stage2p5_final) TWICE —
# whitening_cov=anchor (historical) vs shift (post-2.5 paper-fidelity) — with the
# result-preserving multi-GPU subset (--num-gpus 2: DDP + SVD/EoRA task-parallel,
# NO auto-batch cov-DP). Same Stage-3 cov (1-GPU, deterministic) for both, so the
# only difference is Stage-4 EoRA whitening. Compares stage6alt student_bpt.
set -uo pipefail

BASE="${BASE:-/root/work}"
REPO="${REPO:-/root/work/repo}"
VENV="${VENV:-/root/work/venv}"
PY="${VENV}/bin/python"
MODEL="${BASE}/models/Qwen3.6-35B-A3B"
CFG="${BASE}/box_reap.yaml"
SHARED="${BASE}/probe/_shared"
export PYTHONPATH="${REPO}/src"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# xet backend hit "Internal Writer Error: Background writer channel closed" on the
# 52GB stage2p5_final seed (2026-06-14). Force the standard HTTPS download path.
export HF_HUB_DISABLE_XET=1
cd "${REPO}"

run_arm () {
  local arm="$1"            # anchor | shift
  local proot="${BASE}/ab_${arm}"
  echo "=================== ACOV ARM: ${arm} ==================="
  mkdir -p "${proot}/_shared"
  cp -f "${SHARED}"/*.json "${proot}/_shared/"     # Stage-1 stub gate (both arms)
  "${PY}" -u -m moe_compress.run_reap_ream_35pct \
    --config "${CFG}" --model "${MODEL}" \
    --probe-root "${proot}" --only reap-s234 \
    --whitening-cov "${arm}" --num-gpus 2
  local rc=$?
  echo "ARM_${arm}_RC=${rc}"
  return ${rc}
}

# Sequential (each arm uses both GPUs for DDP). anchor first, then shift.
run_arm anchor; A_RC=$?
run_arm shift;  B_RC=$?

echo "=================== ACOV A/B RESULT ==================="
"${PY}" - "${BASE}/ab_anchor/reap-s234/stage6alt_eval.json" \
          "${BASE}/ab_shift/reap-s234/stage6alt_eval.json" <<'PYEOF'
import json, sys
def bpt(p):
    try:
        d = json.load(open(p))
        for k in ("student_bpt", "bpt", "student_bits_per_token"):
            if k in d: return d[k]
        return d
    except Exception as e:
        return f"<missing: {e}>"
a, s = bpt(sys.argv[1]), bpt(sys.argv[2])
print(f"anchor student_bpt = {a}")
print(f"shift  student_bpt = {s}")
try:
    print(f"delta (shift - anchor) = {float(s) - float(a):+.4f}  "
          f"({'shift WINS (lower bpt)' if float(s) < float(a) else 'anchor wins'})")
except Exception:
    pass
PYEOF
echo "ACOV_AB_DONE A_RC=${A_RC} B_RC=${B_RC}"
