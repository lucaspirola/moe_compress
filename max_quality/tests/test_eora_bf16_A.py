"""Stage 4 EoRA must remain numerically sane when A is round-tripped through bf16.

Same root cause as the Stage 3 bug: bf16 quantization kills small eigenvalues
of the input Gram, making A rank-deficient. The previous A + 1e-6·I + eigh
path then picked up noise eigenvectors. This module pins:

  1. ``test_eora_bf16_A_beats_isotropic_in_A_norm`` — A-aware EoRA strictly
     beats isotropic SVD in the **A-weighted norm** ‖(δ-UV)·L_A‖_F (the
     objective ``_compute_eora_factors`` actually optimizes). Plain Frobenius
     would be the wrong target here: by Eckart–Young, isotropic truncated SVD
     is Frobenius-optimal and A-aware can only match or do worse.

  2. ``test_eora_zero_pad_path_used_when_take_lt_r`` — exercise the
     **zero-pad branch** in production code: A whose effective rank
     (after the relative noise floor) is strictly between 0 and r. The
     previous all-zero-A test short-circuited to the plain-SVD fallback
     and never entered the zero-pad path.
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.stage4_eora import _compute_eora_factors


# Plan §2.5 #3 → "n < d_in" so XᵀX is rank-≤n in d_in dims; pick CPU-friendly
# stand-in for the (4096, ~16k) production regime that still reproduces the
# bf16 rank-killing pathology.
_D_OUT, _D_IN, _N = 256, 512, 200


def _make_inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    delta = torch.randn(_D_OUT, _D_IN, generator=g, dtype=torch.float32) * 0.1
    X = torch.randn(_N, _D_IN, generator=g, dtype=torch.float32)
    scale = torch.logspace(-3, 0, _D_IN, dtype=torch.float32)
    X = X * scale
    A_fp32 = X.T @ X
    A = A_fp32.to(torch.bfloat16).to(torch.float32)   # bf16 round-trip
    return delta, A, A_fp32


def _A_weighted_residual(delta, U, V, A_fp32):
    """‖(δ - U·V)·L_A‖_F where A_fp32 = L_A·L_Aᵀ — the EoRA objective."""
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (A_fp32 + A_fp32.T))
    sigma_max = float(eigvals[-1].clamp_min(0).item())
    keep = eigvals > max(sigma_max * 1e-6, 1e-12)
    L_A = eigvecs[:, keep] * eigvals[keep].clamp_min(0).sqrt().unsqueeze(0)
    R = (delta - U.to(torch.float32) @ V.to(torch.float32)) @ L_A
    return float(R.norm().item())


def test_eora_bf16_A_beats_isotropic_in_A_norm():
    """A-aware EoRA must strictly decrease the A-weighted residual vs isotropic.

    Pinned in the A-weighted norm (the objective the function optimizes), not
    in plain Frobenius — by Eckart–Young, isotropic SVD is Frobenius-optimal
    and A-aware cannot beat it there.
    """
    r = 32
    for seed in (1, 2, 3):
        delta, A, A_fp32 = _make_inputs(seed)
        U_iso, V_iso = _compute_eora_factors(delta, None, r, "cpu")
        U_a, V_a = _compute_eora_factors(delta, A, r, "cpu")
        res_iso = _A_weighted_residual(delta, U_iso, V_iso, A_fp32)
        res_a = _A_weighted_residual(delta, U_a, V_a, A_fp32)
        assert res_a < res_iso - 1e-6, (
            f"seed={seed}: A-aware EoRA did not beat isotropic in A-weighted "
            f"norm (the objective): a={res_a:.4e}, iso={res_iso:.4e}"
        )


def test_eora_bf16_A_shape_stable():
    delta, A, _ = _make_inputs(seed=4)
    r = 32
    U, V = _compute_eora_factors(delta, A, r, "cpu")
    assert U.shape == (_D_OUT, r)
    assert V.shape == (r, _D_IN)


def test_eora_zero_pad_path_used_when_take_lt_r():
    """Construct an A whose effective rank is strictly between 0 and r so the
    eigh path runs (not the plain-SVD fallback) AND the zero-pad branch
    triggers. The previous all-zero-A test bypassed both via the
    keep_idx.numel()==0 short-circuit.
    """
    g = torch.Generator().manual_seed(5)
    d_out, d_in = 64, 48
    delta = torch.randn(d_out, d_in, generator=g, dtype=torch.float32) * 0.1
    # A has exactly 5 sizeable eigenvalues; the rest are below the relative
    # noise floor (1e-6 of sigma_max). With r=12, take_eff = 5 < r=12 ⇒
    # zero-pad path runs.
    Q, _ = torch.linalg.qr(torch.randn(d_in, d_in, generator=g, dtype=torch.float32))
    eigvals = torch.cat([
        torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5], dtype=torch.float32),
        torch.full((d_in - 5,), 1e-12, dtype=torch.float32),
    ])
    # Kept in fp32 deliberately: bf16 round-tripping injects O(σ_max·2⁻⁷)
    # off-diagonal noise that inflates the 1e-12 tail above the production
    # relative threshold, defeating the controlled-rank construction. This
    # test pins the SHAPE/zero-pad CONTRACT of the eigh branch; bf16 numerical
    # robustness is covered by `test_eora_bf16_A_beats_isotropic_in_A_norm`.
    A = (Q * eigvals.unsqueeze(0)) @ Q.T

    r = 12
    U, V = _compute_eora_factors(delta, A, r, "cpu")
    assert U.shape == (d_out, r), f"zero-pad path produced wrong U shape: {U.shape}"
    assert V.shape == (r, d_in), f"zero-pad path produced wrong V shape: {V.shape}"
    # The padded trailing columns/rows must be exactly zero.
    assert torch.equal(U[:, 5:], torch.zeros(d_out, r - 5)), (
        "zero-pad path did not zero the trailing U columns"
    )
    assert torch.equal(V[5:, :], torch.zeros(r - 5, d_in)), (
        "zero-pad path did not zero the trailing V rows"
    )


def test_eora_non_square_shapes_and_residual():
    """Production has both gate/up (d_in < d_out) and down (d_in > d_out).
    Both directions must produce shape-stable [d_out, r] / [r, d_in] outputs
    AND a residual smaller than δ itself (i.e. the correction is doing work,
    not returning zero or noise).
    """
    for d_out, d_in in [(64, 192), (192, 64)]:
        g = torch.Generator().manual_seed(d_out)
        delta = torch.randn(d_out, d_in, generator=g, dtype=torch.float32) * 0.1
        X = torch.randn(80, d_in, generator=g, dtype=torch.float32)
        A = (X.T @ X).to(torch.bfloat16).to(torch.float32)
        r = 16
        U, V = _compute_eora_factors(delta, A, r, "cpu")
        assert U.shape == (d_out, r), f"({d_out},{d_in}): U shape {U.shape}"
        assert V.shape == (r, d_in), f"({d_out},{d_in}): V shape {V.shape}"
        residual = (delta - U @ V).norm().item()
        assert residual < delta.norm().item(), (
            f"({d_out},{d_in}): correction did not reduce residual: "
            f"‖δ‖={delta.norm().item():.4e}, ‖δ-UV‖={residual:.4e}"
        )


@pytest.mark.parametrize("a_provider", [
    pytest.param(lambda d: None, id="A_none"),
    pytest.param(lambda d: torch.eye(d, dtype=torch.float32), id="A_eye"),
])
def test_eora_r_zero_returns_empty(a_provider):
    """r=0 hits the `if r <= 0` early return in both A=None and A=present
    code paths (parametrized to make that explicit; both currently land on
    the same line, but the parametrization guards against a future split)."""
    delta = torch.randn(64, 48, dtype=torch.float32)
    U, V = _compute_eora_factors(delta, a_provider(48), r=0, device="cpu")
    assert U.shape == (64, 0)
    assert V.shape == (0, 48)
