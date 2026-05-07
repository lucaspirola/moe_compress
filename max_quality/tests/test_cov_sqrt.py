"""Phase 1 tests for the activation-aware whitening helper module.

Covers ``moe_compress.utils.cov_sqrt``:
- ``compute_a_sqrt`` modes ``"none"``, ``"diag"``, ``"full"``
- ``whitened_residual`` matches direct computation of ``||ΔW · A^{1/2}||_F``
  with A^{1/2} on the **right** of ΔW (per AA-SVD lineage and the spec-review
  Round 1 dimensional fix)
- Eigen-sqrt is self-consistent: ``A^{1/2} · A^{1/2} ≈ A`` to fp32 precision
- Diag-mode equals full-mode when A is exactly diagonal (sanity bridge)
- ``CovSqrtCache`` LRU eviction and key isolation
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.utils.cov_sqrt import (
    CovSqrtCache,
    compute_a_sqrt,
    whitened_residual,
)


# ---------------------------------------------------------------------------
# compute_a_sqrt: basic correctness
# ---------------------------------------------------------------------------


def test_compute_a_sqrt_full_squared_returns_a():
    """For PSD A, A^{1/2} · A^{1/2} should reproduce A within numerical
    tolerance (matrix square root of a symmetric PSD is uniquely defined
    when restricted to the PSD cone)."""
    torch.manual_seed(0)
    M = torch.randn(8, 8)
    A = M @ M.T + torch.eye(8)  # well-conditioned PSD

    a_sqrt = compute_a_sqrt(A, mode="full")
    assert a_sqrt.shape == A.shape
    diff = (A - a_sqrt @ a_sqrt).abs().max().item()
    assert diff < 1e-4  # fp32 eigh precision


def test_compute_a_sqrt_diag_returns_sqrt_of_diagonal():
    A = torch.tensor([[4.0, 0.5], [0.5, 9.0]])
    a_sqrt = compute_a_sqrt(A, mode="diag")
    assert a_sqrt.shape == (2,)
    expected = torch.sqrt(torch.tensor([4.0, 9.0]))
    assert torch.allclose(a_sqrt, expected)


def test_compute_a_sqrt_none_returns_scalar_one():
    A = torch.randn(3, 3) @ torch.randn(3, 3).T + torch.eye(3)
    sentinel = compute_a_sqrt(A, mode="none")
    assert sentinel.shape == ()
    assert sentinel.item() == pytest.approx(1.0)


def test_compute_a_sqrt_full_matches_diag_when_a_is_diagonal():
    """Sanity bridge: when A is exactly diagonal, the full eigen-sqrt and
    the diag-of-sqrt should produce the same effective whitening (the
    full-mode result equals diag(sqrt(diag(A)))).
    """
    A = torch.diag(torch.tensor([1.0, 4.0, 9.0, 16.0]))
    a_sqrt_full = compute_a_sqrt(A, mode="full")
    a_sqrt_diag = compute_a_sqrt(A, mode="diag")

    # full mode: V·diag(sqrt(λ))·V^T; for diagonal A, V is a permutation of
    # the identity and the result equals diag(sqrt(diag(A))).
    expected = torch.diag(a_sqrt_diag)
    assert torch.allclose(a_sqrt_full, expected, atol=1e-5)


def test_compute_a_sqrt_clamps_tiny_eigenvalues():
    """Near-singular A should not produce NaN/Inf in the sqrt — the
    implementation clamps eigenvalues to a relative noise floor."""
    A = torch.diag(torch.tensor([1e-30, 1.0, 1.0, 1.0]))
    a_sqrt = compute_a_sqrt(A, mode="full")
    assert torch.isfinite(a_sqrt).all()


def test_compute_a_sqrt_rejects_non_square_input():
    A = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="square"):
        compute_a_sqrt(A, mode="full")


def test_compute_a_sqrt_unknown_mode_raises():
    A = torch.eye(3)
    with pytest.raises(ValueError, match="unknown mode"):
        compute_a_sqrt(A, mode="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# whitened_residual: A^{1/2} multiplies ΔW on the RIGHT (input axis).
# This is the key dimensional fix from the spec-review Round 1.
# ---------------------------------------------------------------------------


def test_whitened_residual_full_matches_direct_right_multiply():
    """Verify the output equals ‖ΔW · A^{1/2}‖_F exactly.

    Random ΔW with shape (out, in) and A with shape (in, in) ensures that
    a left-multiply ‖A^{1/2} · ΔW‖_F would have a different shape (in, in)
    and a different Frobenius norm — so this test would fail under the
    pre-fix dimensional bug.
    """
    torch.manual_seed(0)
    out, in_ = 5, 8
    delta_w = torch.randn(out, in_)
    M = torch.randn(in_, in_)
    A = M @ M.T + torch.eye(in_)

    a_sqrt = compute_a_sqrt(A, mode="full")
    direct = torch.linalg.matrix_norm(delta_w @ a_sqrt, ord="fro")

    via_helper = whitened_residual(delta_w, a_sqrt, mode="full")
    assert via_helper == pytest.approx(direct.item(), rel=1e-6)


def test_whitened_residual_diag_matches_direct_column_scaling():
    out, in_ = 4, 6
    delta_w = torch.randn(out, in_)
    diag_vec = torch.rand(in_) + 0.1  # strictly positive

    direct = torch.linalg.matrix_norm(delta_w * diag_vec, ord="fro")
    via_helper = whitened_residual(delta_w, diag_vec, mode="diag")
    assert via_helper == pytest.approx(direct.item(), rel=1e-6)


def test_whitened_residual_none_returns_plain_frobenius():
    delta_w = torch.randn(3, 5)
    plain = torch.linalg.matrix_norm(delta_w, ord="fro")
    via_helper = whitened_residual(delta_w, torch.tensor(1.0), mode="none")
    assert via_helper == pytest.approx(plain.item(), rel=1e-6)


def test_whitened_residual_full_rejects_dim_mismatch():
    """If a_sqrt's input axis doesn't match ΔW's input axis, raise — this
    catches accidental left-multiplication or wrong-covariance inputs."""
    delta_w = torch.randn(4, 8)  # input axis = 8
    a_sqrt = torch.eye(5)  # mismatched input axis
    with pytest.raises(ValueError, match="dim 5 does not match"):
        whitened_residual(delta_w, a_sqrt, mode="full")


def test_whitened_residual_diag_rejects_2d_input():
    delta_w = torch.randn(3, 4)
    a_sqrt_matrix = torch.eye(4)
    with pytest.raises(ValueError, match="expected 1-D"):
        whitened_residual(delta_w, a_sqrt_matrix, mode="diag")


def test_whitened_residual_full_rejects_1d_input():
    delta_w = torch.randn(3, 4)
    a_sqrt_vec = torch.ones(4)
    with pytest.raises(ValueError, match="expected square 2-D"):
        whitened_residual(delta_w, a_sqrt_vec, mode="full")


# ---------------------------------------------------------------------------
# Numerical sanity: whitening preserves the cost-matrix norm structure.
# ‖ΔW · A^{1/2}‖_F² = E_x[‖ΔW · x‖²] over x ~ N(0, A) (when A is the cov).
# ---------------------------------------------------------------------------


def test_whitened_residual_matches_expected_squared_error_under_sampling():
    """If x is drawn from N(0, A) and Y = ΔW · x, then E[‖Y‖²] should
    equal ‖ΔW · A^{1/2}‖_F². Verify with Monte Carlo."""
    torch.manual_seed(42)
    out, in_ = 3, 6
    delta_w = torch.randn(out, in_) * 0.3
    M = torch.randn(in_, in_)
    A = M @ M.T + torch.eye(in_)
    a_sqrt = compute_a_sqrt(A, mode="full")

    # Whitened residual squared:
    expected = whitened_residual(delta_w, a_sqrt, mode="full") ** 2

    # Monte Carlo: x ~ N(0, A) ↔ x = a_sqrt @ z, z ~ N(0, I)
    n_samples = 50_000
    z = torch.randn(in_, n_samples)
    x = a_sqrt @ z
    y = delta_w @ x  # (out, n_samples)
    mc_estimate = (y ** 2).sum(dim=0).mean()  # mean ‖y‖²

    # Loose tolerance for Monte-Carlo noise.
    assert mc_estimate.item() == pytest.approx(expected.item(), rel=0.05)


# ---------------------------------------------------------------------------
# CovSqrtCache: LRU eviction + isolated keys
# ---------------------------------------------------------------------------


def test_cov_sqrt_cache_get_put_roundtrip():
    cache = CovSqrtCache(max_entries=4)
    t = torch.eye(3)
    cache.put(("k1",), t)
    assert cache.get(("k1",)) is t
    assert cache.get(("missing",)) is None


def test_cov_sqrt_cache_lru_eviction():
    cache = CovSqrtCache(max_entries=2)
    a, b, c = torch.eye(3), torch.eye(3) * 2, torch.eye(3) * 3
    cache.put(("k1",), a)
    cache.put(("k2",), b)
    # Touch k1 to mark it most-recently-used.
    _ = cache.get(("k1",))
    # Inserting k3 should evict k2 (least recent).
    cache.put(("k3",), c)
    assert cache.get(("k1",)) is a
    assert cache.get(("k2",)) is None
    assert cache.get(("k3",)) is c
    assert len(cache) == 2


def test_cov_sqrt_cache_key_isolates_layer_expert_matrix_mode():
    """Different (layer, expert, matrix, mode) keys must not collide."""
    cache = CovSqrtCache(max_entries=8)
    diag_t = torch.tensor([1.0, 2.0])
    full_t = torch.eye(2)
    cache.put((0, 1, "gate_proj", "diag"), diag_t)
    cache.put((0, 1, "gate_proj", "full"), full_t)
    assert cache.get((0, 1, "gate_proj", "diag")) is diag_t
    assert cache.get((0, 1, "gate_proj", "full")) is full_t


def test_cov_sqrt_cache_clear():
    cache = CovSqrtCache(max_entries=4)
    cache.put(("k",), torch.eye(2))
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert cache.get(("k",)) is None
