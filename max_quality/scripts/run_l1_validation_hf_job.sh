#!/bin/bash
# run_l1_validation_hf_job.sh -- launch the L1 validation harness on HF Jobs.
#
# VALIDATION HARNESS -- NOT PART OF THE PRODUCTION CODEBASE.
#
# Usage (from a host with `hf` CLI authenticated to namespace `pirola`):
#   bash max_quality/scripts/run_l1_validation_hf_job.sh
#
# Optional env overrides:
#   L1_HARNESS_COMMIT   git ref hosting the harness .py + .sh (default: feat/calibration-v2)
#   FLAVOR              HF Jobs flavor (default: a10g-small)
#   TIMEOUT             HF Jobs timeout (default: 2h)
#   IMAGE               container image (default: nvidia/cuda:13.0.0-devel-ubuntu24.04)

set -euo pipefail

L1_HARNESS_COMMIT="${L1_HARNESS_COMMIT:-feat/calibration-v2}"
FLAVOR="${FLAVOR:-a10g-small}"
TIMEOUT="${TIMEOUT:-2h}"
IMAGE="${IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}"

INSTALLER_URL="https://raw.githubusercontent.com/lucaspirola/moe_compress/${L1_HARNESS_COMMIT}/max_quality/scripts/l1_validation_harness.sh"

echo "Launching L1 validation harness on HF Jobs"
echo "  ref     : ${L1_HARNESS_COMMIT}"
echo "  flavor  : ${FLAVOR}"
echo "  timeout : ${TIMEOUT}"
echo "  image   : ${IMAGE}"
echo "  source  : ${INSTALLER_URL}"

hf jobs run \
    --flavor "${FLAVOR}" \
    --detach \
    --timeout "${TIMEOUT}" \
    --secrets HF_TOKEN \
    --env "L1_HARNESS_COMMIT=${L1_HARNESS_COMMIT}" \
    "${IMAGE}" \
    bash -c "curl -sL ${INSTALLER_URL} | L1_HARNESS_COMMIT=${L1_HARNESS_COMMIT} bash"
