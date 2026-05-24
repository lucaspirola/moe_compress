"""Regression: AA-SVD must remain numerically sane when B is round-tripped
through bfloat16 storage.

The original Stage 3 pipeline persisted the input covariance sidecar in
bfloat16 (7 mantissa bits). Round-tripping a positive-definite Gram through
bf16 quantizes its small eigenvalues to zero — making B rank-deficient.
The previous Cholesky-of-(B+1e-6·I) path then fit noise into the killed
directions and produced rel_err on the order of 10²–10⁶ (a "relative" error
whose exact-arithmetic upper bound is 1).

The fix is the eigh-based factorization with effective-rank clipping. This
test pins that fix: with bf16-quantized B, rel_err must stay in [0, 1] and
shrink monotonically with rank.
"""
from __future__ import annotations

import torch

from moe_compress.stage3_svd import _aa_svd


def _bf16_roundtrip_B(d_in: int, n: int, seed: int = 0) -> torch.Tensor:
    """Build a bf16-roundtripped Gram in the regime the plan calls for:

    n < d_in (under-determined calibration) means rank(X) ≤ n, so XᵀX has at
    most n nonzero eigenvalues in d_in dimensions — exactly the wide-fat
    covariance regime Stage 2 produces in production. Combined with bf16's
    7-mantissa-bit storage, this reliably manufactures rank-deficient B (the
    OLD code path then fits noise into the dropped directions and returns
    rel_err in the 10²–10⁶ range).
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d_in, generator=g, dtype=torch.float32)
    scale = torch.logspace(-3, 0, d_in, dtype=torch.float32)
    X = X * scale
    B = X.T @ X
    return B.to(torch.bfloat16).to(torch.float32)


# Plan §2.5 #1 calls for d_in ≥ 512 with n < d_in; pick (640, 200) as a
# CPU-friendly stand-in for production's (6144, ~16k_tokens).
_D_OUT, _D_IN, _N = 256, 640, 200


def test_aa_svd_bf16_B_rel_err_in_unit_interval():
    """Across multiple seeds, rel_err must stay in [0, 1]. Pre-fix code
    produced 10²–10⁶ on this regime, so even loose `<= 1.0 + 1e-3` fails old
    code by 5+ orders of magnitude."""
    d_out, d_in = _D_OUT, _D_IN
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
        B = _bf16_roundtrip_B(d_in, _N, seed=seed + 100)
        for k in (16, 64, 200):
            _, _, rel_err, _ = _aa_svd(W, None, B, k, device="cpu")
            assert 0.0 <= rel_err <= 1.0 + 1e-3, (
                f"seed={seed} k={k}: rel_err escaped [0,1]: {rel_err}"
            )


def test_aa_svd_bf16_B_rel_err_monotonic_in_rank():
    """rel_err must be monotonically non-increasing in k (across seeds). k
    values stop at the underlying rank floor n=200; beyond that, the eigh
    path clips to k_eff=r_eff and rel_err plateaus, which is correct but not
    interesting to assert."""
    d_out, d_in = _D_OUT, _D_IN
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
        B = _bf16_roundtrip_B(d_in, _N, seed=seed + 200)
        errs = []
        for k in (16, 32, 64, 128, 200):
            _, _, rel_err, _ = _aa_svd(W, None, B, k, device="cpu")
            errs.append(rel_err)
        for prev, cur in zip(errs[:-1], errs[1:]):
            assert cur <= prev + 1e-4, (
                f"seed={seed}: rel_err not monotonically non-increasing in k; "
                f"got {errs}"
            )


def test_aa_svd_bf16_B_converges_at_full_effective_rank():
    """At k ≥ r_eff, the eigh path captures all spannable directions in B,
    so rel_err (singular-value tail of M = W·L_B beyond k_eff) collapses to
    bf16-quantization noise. We pin a loose median-over-seeds bound to avoid
    seed-flakiness from a single eigenvalue straddling the relative noise
    floor; the property under test is "rel_err is small at full k", not a
    specific number.
    """
    d_out, d_in = _D_OUT, _D_IN
    errs = []
    for seed in range(5):
        g = torch.Generator().manual_seed(seed + 300)
        W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
        B = _bf16_roundtrip_B(d_in, _N, seed=seed + 400)
        _, _, rel_err, k_eff = _aa_svd(W, None, B, d_in, device="cpu")
        # k_eff is clipped to r_eff (the count of eigvals above the relative
        # noise floor). True rank ≤ n=200; bf16 quantization can inflate
        # apparent rank by ~30%. Cap at d_in/2 = 320 — catches both total
        # threshold disablement AND a 3-4 OoM threshold loosening (which
        # would let k_eff balloon to 500-600).
        assert k_eff <= _D_IN // 2, (
            f"seed={seed}: k_eff={k_eff} > d_in/2; eigh threshold appears "
            "loose enough to keep noise-inflated directions"
        )
        errs.append(rel_err)
    median = sorted(errs)[len(errs) // 2]
    assert median < 1e-2, (
        f"At k=d_in, median rel_err over 5 seeds should be near 0; got {errs} "
        f"(median {median:.4e})"
    )


def test_aa_svd_bf16_B_does_not_fall_back_to_plain_svd(monkeypatch):
    """The eigh path must succeed on bf16-quantized B; the plain-SVD fallback
    is a defensive last-resort and should not be hit on this input.
    """
    # S3-5: ``_aa_svd`` was relocated to ``stage3/plugins/aa_svd_factor`` and
    # now logs the "fallback to plain SVD" warning through THAT module's
    # ``log``. Spy on the new module's logger — a spy on ``stage3_svd.log``
    # would silently never fire and this test would pass vacuously.
    import moe_compress.stage3.plugins.aa_svd_factor as aa

    fallback_calls = {"n": 0}
    real_warning = aa.log.warning

    def _spy(msg, *a, **kw):
        if isinstance(msg, str) and "fallback to plain SVD" in msg:
            fallback_calls["n"] += 1
        return real_warning(msg, *a, **kw)

    monkeypatch.setattr(aa.log, "warning", _spy)

    d_out, d_in = _D_OUT, _D_IN
    g = torch.Generator().manual_seed(5)
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
    B = _bf16_roundtrip_B(d_in, _N, seed=6)
    _aa_svd(W, None, B, 200, device="cpu")
    assert fallback_calls["n"] == 0, "plain-SVD fallback fired on bf16-quantized B"


def test_aa_svd_fallback_warning_fires_on_bad_B(monkeypatch):
    """Positive counterpart to the test above: when the eigh path genuinely
    cannot proceed, ``_aa_svd`` MUST log the fallback warning and still return
    a valid plain-SVD factorization.

    Without this test, a future change that silenced
    ``aa_svd_factor.log.warning`` would go undetected — the
    "does_not_fall_back" test only asserts the spy stays at zero, which a
    permanently-mute logger also satisfies.

    Trigger: an all-NaN covariance B. NaN propagates through
    ``B = 0.5*(B + B.T)`` and ``torch.linalg.eigh``; the NaN eigenvalues then
    fail the ``eigvals > thresh`` comparison (NaN compares False), so
    ``_precompute_eigh`` raises ``ValueError`` ("no positive eigenvalues above
    threshold") — the cheapest deterministic way to force the fallback branch
    without a real model (the implementer's note: "NaN B → eigh raises").
    """
    import moe_compress.stage3.plugins.aa_svd_factor as aa

    fallback_calls = {"n": 0, "msgs": []}
    real_warning = aa.log.warning

    def _spy(msg, *a, **kw):
        if isinstance(msg, str) and "fallback to plain SVD" in msg:
            fallback_calls["n"] += 1
            fallback_calls["msgs"].append(msg)
        return real_warning(msg, *a, **kw)

    monkeypatch.setattr(aa.log, "warning", _spy)

    d_out, d_in = _D_OUT, _D_IN
    g = torch.Generator().manual_seed(7)
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
    # All-NaN B: a structurally-valid-shape covariance whose contents force
    # _precompute_eigh down the failure path and thus _aa_svd into fallback.
    B = torch.full((d_in, d_in), float("nan"), dtype=torch.float32)

    k = 64
    U_k, V_k, rel_err, k_eff = _aa_svd(W, None, B, k, device="cpu")

    # (a) the spy genuinely fired.
    assert fallback_calls["n"] >= 1, (
        "fallback warning did NOT fire on all-NaN B — the eigh path either "
        "did not fail or the warning is silenced"
    )
    # (b) the warning message names the fallback.
    assert all(
        "fallback to plain SVD" in m for m in fallback_calls["msgs"]
    ), f"unexpected fallback warning text: {fallback_calls['msgs']}"
    # (c) _aa_svd still returns a valid plain-SVD factorization — it falls
    # back, it does not crash, and the factors are finite.
    assert U_k.shape == (d_out, k) and V_k.shape == (k, d_in), (
        f"fallback returned wrong shapes: U={U_k.shape} V={V_k.shape}"
    )
    assert torch.isfinite(U_k).all() and torch.isfinite(V_k).all(), (
        "fallback produced non-finite factors"
    )
    assert 0.0 <= rel_err <= 1.0 + 1e-3, (
        f"fallback rel_err escaped [0,1]: {rel_err}"
    )
    assert k_eff == k, f"plain-SVD fallback should report k_eff=k; got {k_eff}"
