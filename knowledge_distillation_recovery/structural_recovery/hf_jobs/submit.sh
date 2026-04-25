#!/usr/bin/env bash
# Submit Chapter 1 — Structural Recovery to HF Jobs.
#
# Verified flavors (via `hf jobs hardware` 2026-04-25):
#   a100x4   4× A100-80GB (320 GB), 568 GB host RAM, $10/h  ← Light tier default
#   h200x2   2× H200      (282 GB), 512 GB host RAM, $10/h  (Light alt)
#   a100x8   8× A100      (640 GB), 1136 GB host RAM, $20/h (Moderate/Heavy)
#   h200     1× H200      (141 GB), 256 GB host RAM, $5/h   ← Smoke tier default
#
# NOTE: a100-large (1× A100, 80 GB) is NOT viable — FP8 teacher (~37 GB) +
# BF16 student (~70 GB) = 107 GB, exceeds the 80 GB budget.
#
# Usage:
#   STUDENT_REPO=pirola/qwen3-... ./hf_jobs/submit.sh                   # Light
#   SMOKE=1 STUDENT_REPO=... ./hf_jobs/submit.sh                        # smoke (h200)
#   FLAVOR=h200x2 STUDENT_REPO=... ./hf_jobs/submit.sh                  # Light alt
#   DETACH=1 STUDENT_REPO=... ./hf_jobs/submit.sh                       # background

set -euo pipefail

# Smoke runs default to the smoke YAML on h200 (single H200, 141 GB);
# Light runs default to a100x4. a100-large (single A100-80GB) cannot fit
# the FP8 teacher (~37 GB) + BF16 student (~70 GB) at once.
SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
    FLAVOR="${FLAVOR:-h200}"
    CONFIG_PATH="${CONFIG_PATH:-configs/qwen36_35b_a3b_chapter1_smoke.yaml}"
    TIMEOUT="${TIMEOUT:-2h}"
else
    FLAVOR="${FLAVOR:-a100x4}"
    CONFIG_PATH="${CONFIG_PATH:-configs/qwen36_35b_a3b_chapter1_light.yaml}"
    TIMEOUT="${TIMEOUT:-12h}"
fi

CODE_REPO="${CODE_REPO:-pirola/moe-compress-code}"
RECOVERY_REPO="${RECOVERY_REPO:-pirola/structural-recovery-code}"
STUDENT_REPO="${STUDENT_REPO:-}"
RESULT_REPO="${RESULT_REPO:-}"
BUCKET="${BUCKET:-hf://buckets/pirola/moe-cache}"
MOUNT="${MOUNT:-/mnt/cache}"
SKIP_TEACHER_CORRECTION="${SKIP_TEACHER_CORRECTION:-0}"
ENTRYPOINT="$(cd "$(dirname "$0")" && pwd)/entrypoint.py"
DETACH="${DETACH:-0}"

if [[ -z "$STUDENT_REPO" ]]; then
    echo "ERROR: STUDENT_REPO is required (the result repo from your max_quality run)." >&2
    echo "       e.g. STUDENT_REPO=pirola/qwen3-6-35b-a3b-strategy-a-30pct-20260424-1230" >&2
    exit 1
fi

if [[ ! -f "$ENTRYPOINT" ]]; then
    echo "entrypoint.py not found at $ENTRYPOINT" >&2
    exit 1
fi

DETACH_FLAG=""
if [[ "$DETACH" == "1" ]]; then
    DETACH_FLAG="--detach"
fi

echo ">>> Submitting Chapter 1 — Structural Recovery"
echo "    flavor          : $FLAVOR"
echo "    timeout         : $TIMEOUT"
echo "    code repo       : $CODE_REPO"
echo "    recovery repo   : $RECOVERY_REPO"
echo "    student repo    : $STUDENT_REPO"
echo "    config          : $CONFIG_PATH"
echo "    smoke           : $SMOKE"
echo "    skip teacher cc : $SKIP_TEACHER_CORRECTION"
echo "    bucket mount    : $BUCKET → $MOUNT"
echo "    result repo     : ${RESULT_REPO:-<auto>}"
echo

# Build --env list, omitting empty values to keep job env clean.
ENV_ARGS=(
    --env "CODE_REPO=$CODE_REPO"
    --env "RECOVERY_REPO=$RECOVERY_REPO"
    --env "STUDENT_REPO=$STUDENT_REPO"
    --env "CACHE_MOUNT=$MOUNT"
    --env "CONFIG_PATH=$CONFIG_PATH"
    --env "SMOKE=$SMOKE"
    --env "SKIP_TEACHER_CORRECTION=$SKIP_TEACHER_CORRECTION"
    --env "PYTORCH_ALLOC_CONF=expandable_segments:True"
)
if [[ -n "$RESULT_REPO" ]]; then
    ENV_ARGS+=(--env "RESULT_REPO=$RESULT_REPO")
fi

exec hf jobs uv run "$ENTRYPOINT" \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --volume "$BUCKET:$MOUNT" \
    --secrets HF_TOKEN \
    "${ENV_ARGS[@]}" \
    $DETACH_FLAG
