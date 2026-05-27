#!/bin/bash
# run_v2_validation_hf_job.sh — launch the v2 calibration validation
# harness (Phase A R1-alignment + Phase B smoke build) on HF Jobs.
#
# VALIDATION HARNESS — NOT PART OF THE PRODUCTION CODEBASE.
#
# Usage (from a host with `hf` CLI authenticated to namespace `pirola`):
#   bash max_quality/scripts/run_v2_validation_hf_job.sh
#
# Optional env overrides:
#   V2_VAL_COMMIT   git ref hosting the harness .py + .sh (default: a3a946a)
#   FLAVOR          HF Jobs flavor (default: l40sx1)
#   TIMEOUT         HF Jobs timeout (default: 4h)
#   IMAGE           container image (default: nvidia/cuda:13.0.0-devel-ubuntu24.04)

set -euo pipefail

V2_VAL_COMMIT="${V2_VAL_COMMIT:-a3a946a}"
FLAVOR="${FLAVOR:-h200}"
TIMEOUT="${TIMEOUT:-4h}"
IMAGE="${IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}"

INSTALLER_URL="https://raw.githubusercontent.com/lucaspirola/moe_compress/${V2_VAL_COMMIT}/max_quality/scripts/v2_validation_harness.sh"

echo "Launching v2 validation harness on HF Jobs"
echo "  ref     : ${V2_VAL_COMMIT}"
echo "  flavor  : ${FLAVOR}"
echo "  timeout : ${TIMEOUT}"
echo "  image   : ${IMAGE}"
echo "  source  : ${INSTALLER_URL}"

hf jobs run \
    --flavor "${FLAVOR}" \
    --detach \
    --timeout "${TIMEOUT}" \
    --secrets HF_TOKEN \
    --env "V2_VAL_COMMIT=${V2_VAL_COMMIT}" \
    "${IMAGE}" \
    bash -c "export DEBIAN_FRONTEND=noninteractive && apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates && curl -sL ${INSTALLER_URL} | V2_VAL_COMMIT=${V2_VAL_COMMIT} bash"
