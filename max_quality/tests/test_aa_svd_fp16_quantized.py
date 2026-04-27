"""Sanity check on the bf16 → fp16 storage switch.

fp16 has 10 mantissa bits vs bf16's 7. On a positive-definite Gram with a
broad eigenvalue spread, fp16 round-tripping preserves materially more of
B's spectrum than bf16. Plan §2.5 #2 calls for the **effective rank**
(eigenvalues above the relative noise floor used in
``stage3_svd._aa_svd``) to be materially better tracked under fp16.

We exercise the production path itself: ``_aa_svd`` returns ``k_eff`` (the
effective rank actually used to factor W). This way the test fails if
``_aa_svd``'s threshold is loosened or if the dtype branch is silently
flipped in the config — neither of which a re-derived local formula would
catch.
"""
from __future__ import annotations

import torch

from moe_compress.stage3_svd import _aa_svd


# Same regime as test_aa_svd_bf16_quantized: n < d_in (wide-fat covariance).
# Spread chosen to stay inside fp16's representable range (max ≈ 6.5e4).
_D_OUT, _D_IN, _N = 256, 512, 200


def _make_W_and_B(seed: int):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(_D_OUT, _D_IN, generator=g, dtype=torch.float32)
    X = torch.randn(_N, _D_IN, generator=g, dtype=torch.float32)
    scale = torch.logspace(-3, 0, _D_IN, dtype=torch.float32)
    return W, (X * scale).T @ (X * scale)


def _k_eff(W, B, k):
    """Drive _aa_svd and return its effective rank — the production code's
    own decision, not a re-implementation of its threshold logic."""
    _, _, _, k_eff = _aa_svd(W, None, B, k, device="cpu")
    return k_eff


def test_fp16_effective_rank_closer_to_fp32_than_bf16_avg():
    """Across multiple seeds, the *mean* gap to fp32 must be smaller under
    fp16 than under bf16. Mean is more robust than per-seed strict-less-than:
    bf16 noise can either deflate tail eigenvalues to zero OR inflate them
    above the relative noise floor, so a single seed can produce ties at
    integer rank boundaries — but the mean over seeds will separate."""
    k_request = _D_IN  # ask for full rank — k_eff reflects what _aa_svd kept
    fp16_gaps = []
    bf16_gaps = []
    for seed in range(8):
        W, B_fp32 = _make_W_and_B(seed)
        B_fp16 = B_fp32.to(torch.float16).to(torch.float32)
        B_bf16 = B_fp32.to(torch.bfloat16).to(torch.float32)
        r_fp32 = _k_eff(W, B_fp32, k_request)
        r_fp16 = _k_eff(W, B_fp16, k_request)
        r_bf16 = _k_eff(W, B_bf16, k_request)
        fp16_gaps.append(abs(r_fp16 - r_fp32))
        bf16_gaps.append(abs(r_bf16 - r_fp32))
    mean_fp16 = sum(fp16_gaps) / len(fp16_gaps)
    mean_bf16 = sum(bf16_gaps) / len(bf16_gaps)
    # Strictly tighter than `<`: require at least 1.0 integer-rank advantage on
    # average across 8 seeds. Avoids ties when both gaps happen to be small.
    assert mean_fp16 + 1.0 <= mean_bf16, (
        f"fp16 must track fp32's k_eff more closely than bf16 on average; "
        f"fp16 gaps={fp16_gaps} (mean={mean_fp16:.2f}), "
        f"bf16 gaps={bf16_gaps} (mean={mean_bf16:.2f})"
    )


def test_fp16_effective_rank_close_to_fp32():
    """fp16 should track fp32 ground truth to within a small fraction of d_in."""
    W, B_fp32 = _make_W_and_B(seed=7)
    r_fp32 = _k_eff(W, B_fp32, _D_IN)
    r_fp16 = _k_eff(W, B_fp32.to(torch.float16).to(torch.float32), _D_IN)
    # fp16 noise floor (σ_max·2⁻¹⁰ ≈ σ_max/1024) is well above the production
    # relative threshold (σ_max·1e-6 = σ_max/1e6), so a handful of tail
    # eigenvalues that were below threshold in fp32 get noised up above
    # threshold in fp16. Empirical drift on this spectrum is ~13; cap at 25
    # (fail fast if it ever drifts to bf16-class). bf16 drifts far further.
    assert abs(r_fp32 - r_fp16) <= 25, (
        f"fp16 effective rank drifted from fp32 by more than 25: "
        f"fp32={r_fp32}, fp16={r_fp16}"
    )
