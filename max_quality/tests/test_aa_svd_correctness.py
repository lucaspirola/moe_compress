"""Numerical correctness tests for stage3_svd._aa_svd.

This module pins the most important invariant of the AA-SVD factorization:
the returned (U_k, V_k) must satisfy U_k @ V_k ≈ W (in the B-weighted
Frobenius norm), because every downstream consumer assumes it:

  - FactoredExperts.forward computes y = U @ V @ x (must equal W @ x)
  - Stage 4 EoRA computes the residual delta = W - U @ V (must be small)
  - _per_matrix_refine minimizes tr((W - UV)^T (W - UV) A) (target = W)

The previous implementation accidentally produced U_k @ V_k ≈ W @ A (from
an erroneous extra A factor in the M matrix), which corrupted the forward
pass and broke EoRA's residual target. These tests would have caught that
immediately.
"""
from __future__ import annotations

import math

import torch

from moe_compress.stage3_svd import _aa_svd


def _make_inputs(d_out, d_in, seed=0, scale_low=0.1, scale_high=10.0):
    """Random W and a non-isotropic input Gram B = X^T X.

    A is built from a different scale pattern so it's distinct from B —
    this lets us catch any code path that confuses A and B or includes A
    where it shouldn't be.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
    n = max(d_in * 4, 256)
    X = torch.randn(n, d_in, generator=g, dtype=torch.float32)
    scale = torch.linspace(scale_low, scale_high, d_in, dtype=torch.float32)
    X = X * scale
    B = X.transpose(0, 1) @ X
    # A: different non-isotropic Gram, distinct spectrum from B.
    Y = torch.randn(n, d_in, generator=g, dtype=torch.float32)
    scale_a = torch.linspace(scale_high, scale_low, d_in, dtype=torch.float32)
    Y = Y * scale_a
    A = Y.transpose(0, 1) @ Y
    return W, A, B


def test_aa_svd_recovers_W_with_isotropic_B():
    """With B ≈ I (isotropic), AA-SVD must match plain truncated SVD of W.

    This is the canonical sanity check: under no activation weighting, the
    algorithm should produce the standard SVD-optimal rank-k approximation.
    """
    W, _A, _B = _make_inputs(d_out=20, d_in=16, seed=0)
    B = torch.eye(16, dtype=torch.float32)
    A = torch.eye(16, dtype=torch.float32) * 1.5  # different scale, must be ignored
    k = min(W.shape) - 1
    U_k, V_k, _, _ = _aa_svd(W, A, B, k, device="cpu")
    U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
    expected = U_full[:, :k] @ torch.diag(S_full[:k]) @ Vh_full[:k, :]
    err = (U_k @ V_k - expected).norm() / expected.norm()
    assert err.item() < 1e-3, (
        f"AA-SVD with isotropic B should match plain SVD; got "
        f"||UV_aa - UV_plain|| / ||UV_plain|| = {err.item():.6f}"
    )


def test_aa_svd_minimizes_B_weighted_error():
    """Returned (U,V) must minimize ||(W - UV) L_B||_F at the chosen rank.

    Equivalently: the rel_err returned by _aa_svd must equal the optimal
    rank-k truncation error of M = W @ L_B, in the M-Frobenius sense.
    """
    W, A, B = _make_inputs(d_out=32, d_in=24, seed=1)
    k = 16
    _, _, rel_err, _ = _aa_svd(W, A, B, k, device="cpu")

    # The optimal rank-k error in the B-weighted norm is the tail of M's
    # singular values: sqrt(sum_{i>=k} σ_i^2(M)) / sqrt(sum_i σ_i^2(M)).
    B_reg = B + 1e-6 * torch.eye(B.shape[0], dtype=B.dtype)
    L_B = torch.linalg.cholesky(B_reg)
    M = W @ L_B
    _U, S, _Vh = torch.linalg.svd(M, full_matrices=False)
    optimal_rel = float((S[k:].pow(2).sum().sqrt() / S.pow(2).sum().sqrt()).item())

    assert math.isclose(rel_err, optimal_rel, rel_tol=1e-3, abs_tol=1e-5), (
        f"_aa_svd is not optimal in the B-weighted norm: "
        f"rel_err={rel_err:.6f}, optimal={optimal_rel:.6f}"
    )


def test_aa_svd_target_is_W_not_W_at_A():
    """Critical regression: U @ V must approximate W, never W @ A.

    The bug we are guarding against: a previous implementation included an
    extra A factor in M, producing U @ V ≈ W @ A. With non-isotropic A,
    ||W @ A - W||_F is large, so the bug would show up as the recon being
    far from W and (suspiciously) close to W @ A.
    """
    W, A, B = _make_inputs(d_out=24, d_in=20, seed=2)
    k = min(W.shape) - 1   # near-full rank → tight approximation
    U_k, V_k, _, _ = _aa_svd(W, A, B, k, device="cpu")
    recon = U_k @ V_k

    err_to_W = (recon - W).norm().item()
    err_to_WA = (recon - W @ A).norm().item()
    # The recon must be much closer to W than to W @ A.
    assert err_to_W < err_to_WA * 0.1, (
        f"U @ V is not approximating W: ||recon - W|| = {err_to_W:.4f}, "
        f"||recon - W@A|| = {err_to_WA:.4f}. The W @ A bug has reappeared."
    )


def test_aa_svd_ignores_A_factor():
    """Result must not depend on A — A is reserved for L-BFGS refinement only."""
    W, A, B = _make_inputs(d_out=20, d_in=16, seed=3)
    k = 8
    U1, V1, r1, _ = _aa_svd(W, A, B, k, device="cpu")
    U2, V2, r2, _ = _aa_svd(W, A * 100.0, B, k, device="cpu")  # rescale A 100x
    U3, V3, r3, _ = _aa_svd(W, None, B, k, device="cpu")       # drop A entirely

    diff_U_12 = (U1 - U2).norm().item() / (U1.norm().item() + 1e-9)
    diff_V_12 = (V1 - V2).norm().item() / (V1.norm().item() + 1e-9)
    diff_U_13 = (U1 - U3).norm().item() / (U1.norm().item() + 1e-9)
    diff_V_13 = (V1 - V3).norm().item() / (V1.norm().item() + 1e-9)

    assert diff_U_12 < 1e-5, f"U_k changed when A was scaled 100x: {diff_U_12:.2e}"
    assert diff_V_12 < 1e-5, f"V_k changed when A was scaled 100x: {diff_V_12:.2e}"
    assert diff_U_13 < 1e-5, f"U_k changed when A was set None: {diff_U_13:.2e}"
    assert diff_V_13 < 1e-5, f"V_k changed when A was set None: {diff_V_13:.2e}"
    assert math.isclose(r1, r2, rel_tol=1e-5)
    assert math.isclose(r1, r3, rel_tol=1e-5)


def test_aa_svd_rel_err_in_unit_interval():
    """Returned rel_err must be in [0, 1] — it's a relative weighted error."""
    W, A, B = _make_inputs(d_out=24, d_in=20, seed=4)
    for k in (4, 8, 12, 18):
        _, _, rel_err, _ = _aa_svd(W, A, B, k, device="cpu")
        assert 0.0 <= rel_err <= 1.0 + 1e-6, (
            f"rel_err out of [0,1] at k={k}: {rel_err}"
        )


def test_aa_svd_rel_err_decreases_with_rank():
    """Higher rank → lower weighted reconstruction error (monotonic non-increasing)."""
    W, A, B = _make_inputs(d_out=32, d_in=24, seed=5)
    errs = []
    for k in (2, 4, 8, 16, 22):
        _, _, rel_err, _ = _aa_svd(W, A, B, k, device="cpu")
        errs.append(rel_err)
    for prev, cur in zip(errs[:-1], errs[1:]):
        assert cur <= prev + 1e-5, (
            f"rel_err should be monotonically non-increasing in k; got {errs}"
        )


def test_aa_svd_plain_fallback_when_B_missing():
    """With B=None, falls back to plain SVD; rel_err matches optimal Frobenius."""
    W, _, _ = _make_inputs(d_out=20, d_in=16, seed=6)
    k = 8
    U_k, V_k, rel_err, _ = _aa_svd(W, None, None, k, device="cpu")
    recon = U_k @ V_k
    _U_full, S_full, _ = torch.linalg.svd(W, full_matrices=False)
    optimal_err = float((S_full[k:].pow(2).sum().sqrt() / W.norm()).item())
    actual_err = ((W - recon).norm() / W.norm()).item()
    assert math.isclose(actual_err, optimal_err, rel_tol=1e-3), (
        f"Plain SVD fallback should match optimal rank-{k} Frobenius error: "
        f"got {actual_err:.6f}, optimal {optimal_err:.6f}"
    )
    assert math.isclose(rel_err, actual_err, rel_tol=1e-3)
