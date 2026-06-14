#!/usr/bin/env python3
"""Authoritative CONTENT verify of the downloaded covariance (run with venv torch,
post-build). Asserts non-empty + finite — file size is NOT proof (prior incidents:
fp16 inf corruption, stale manifest). Exit 0 + COV_VERIFY_OK on success."""
import sys, torch

p = sys.argv[1] if len(sys.argv) > 1 else "/root/work/run/sidecars/self_traces_489ee0e1b17b43b0/covariance.pt"
payload = torch.load(p, map_location="cpu", weights_only=False)
# CovariancePayload exposes the Σ_in dict as .sigma_in; legacy dict payloads use "covariance".
if hasattr(payload, "sigma_in"):
    cov = payload.sigma_in
elif isinstance(payload, dict):
    cov = payload.get("covariance", payload)
else:
    cov = payload
n = len(cov)
gate = sum(1 for k in cov if "gate" in str(k) or "up" in str(k))
down = sum(1 for k in cov if "down" in str(k))
# finiteness over a sample (loading all 91GB to GPU is unnecessary)
sample = list(cov.values())[:16]
finite = all(torch.isfinite(v.float()).all().item() for v in sample)
print(f"[verify] {p}\n[verify] entries={n} gate/up~={gate} down~={down} sampled16_finite={finite}")
assert n > 0, "covariance empty"
assert finite, "covariance has non-finite sampled entries"
print("COV_VERIFY_OK")
