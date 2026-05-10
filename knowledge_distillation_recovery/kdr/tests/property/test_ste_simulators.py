"""Hypothesis property tests for NativeBackend STE simulators (LLR-0045).

Four properties per public simulator function:

  * identity-on-forward      — quant(x) is "close" to x (zero for representable values)
  * dequant-quant idempotence — quant(quant(x)) == quant(x)
  * granularity correctness   — per-slice scales differ when source slices differ
  * gradient flow through STE — d(quant(x))/dx == 1 (identity backward)

# REQ: LLR-0045
# VERIFIES: LLR-0015
# VERIFIES: LLR-0045
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from kdr.quant.native_backend.ste_simulators import (
    int_quant_ste,
    mxfp4_kv_ste,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shape strategies
# ─────────────────────────────────────────────────────────────────────────────


# Small but multi-axis shapes — keeps tests fast while still exercising
# per-slice scale broadcasting.
_SHAPES_2D = st.tuples(
    st.integers(min_value=2, max_value=8),
    st.integers(min_value=2, max_value=8),
)
_SHAPES_3D = st.tuples(
    st.integers(min_value=2, max_value=6),
    st.integers(min_value=2, max_value=6),
    st.integers(min_value=2, max_value=6),
)


def _random_tensor(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> torch.Tensor:
    """Reproducible fp32 tensor at a given seed (Hypothesis shrinks on seed)."""
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(shape, generator=g, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# int_quant_ste — INT3 / INT2 / INT4 / INT8
# ─────────────────────────────────────────────────────────────────────────────


@given(shape=_SHAPES_2D, seed=st.integers(min_value=0, max_value=10_000), bits=st.sampled_from([2, 3, 4, 8]))
@settings(max_examples=30, deadline=None)
def test_int_ste_idempotence(shape: tuple[int, int], seed: int, bits: int) -> None:
    """quant(quant(x)) == quant(x): once snapped to the grid, second pass is a no-op."""
    x = _random_tensor(shape, seed)
    q1 = int_quant_ste(x, bits=bits, axis=0)
    q2 = int_quant_ste(q1, bits=bits, axis=0)
    # `q2` runs through the STE again; the forward output is the snap of q1's
    # values to the same per-slice grid → bit-equal to q1.
    assert torch.equal(q1, q2), (
        f"int_quant_ste idempotence failed (bits={bits}): "
        f"max diff = {(q1 - q2).abs().max().item()}"
    )


@given(shape=_SHAPES_2D, seed=st.integers(min_value=0, max_value=10_000), bits=st.sampled_from([3, 4, 8]))
@settings(max_examples=30, deadline=None)
def test_int_ste_grid_bound(shape: tuple[int, int], seed: int, bits: int) -> None:
    """Snapped values are within the [-qmax*scale, +qmax*scale] interval per slice."""
    x = _random_tensor(shape, seed)
    q = int_quant_ste(x, bits=bits, axis=0)
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    # Per-slice scale: max-abs / qmax along axis-0 (so reduce dim=1).
    abs_max = x.abs().amax(dim=1, keepdim=True).clamp(min=torch.finfo(x.dtype).tiny)
    scale = abs_max / qmax
    upper = qmax * scale
    lower = qmin * scale
    # Use a small float-ε slack to absorb the rounding bias of floor/ceil.
    slack = 1e-5 * scale.amax()
    assert (q <= upper + slack).all(), "values exceed +qmax * scale"
    assert (q >= lower - slack).all(), "values undercut -qmax * scale"


@given(shape=_SHAPES_3D, seed=st.integers(min_value=0, max_value=10_000), bits=st.sampled_from([3, 4]))
@settings(max_examples=30, deadline=None)
def test_int_ste_axis_granularity(
    shape: tuple[int, int, int], seed: int, bits: int
) -> None:
    """Per-slice scales along ``axis`` differ when source slices differ.

    Build x where slice 0 along axis-0 has 100x larger magnitude than slice 1.
    The per-slice scales must therefore differ by ~100x; if our axis logic
    were wrong (e.g. reducing along the wrong axes) the scales would be equal.
    """
    if shape[0] < 2:
        return  # Need ≥2 slices along axis-0 to compare.
    x = _random_tensor(shape, seed)
    # Make slice 0 100x larger than slice 1.
    x[0] = x[0] * 100.0
    x[1] = x[1] * 0.01

    q = int_quant_ste(x, bits=bits, axis=0)
    abs0 = q[0].abs().max().item()
    abs1 = q[1].abs().max().item()
    # If granularity is per-axis-0, slice 0 has ~100x larger range than slice 1.
    # Allow a wide tolerance — the test checks orders of magnitude, not equality.
    assert abs0 > 100 * abs1 * 0.5, (
        f"per-slice scaling: slice 0 max-abs={abs0:.4f}, slice 1 max-abs={abs1:.4f} "
        f"(expected slice 0 ≫ slice 1)"
    )


@given(shape=_SHAPES_2D, seed=st.integers(min_value=0, max_value=10_000), bits=st.sampled_from([3, 4]))
@settings(max_examples=30, deadline=None)
def test_int_ste_gradient_is_identity(
    shape: tuple[int, int], seed: int, bits: int
) -> None:
    """∂(int_quant_ste(x))/∂x == 1 — STE forwards the upstream gradient unchanged."""
    x = _random_tensor(shape, seed)
    x.requires_grad_(True)
    y = int_quant_ste(x, bits=bits, axis=0)
    grad_out = torch.ones_like(y)
    y.backward(grad_out)
    assert x.grad is not None
    assert torch.equal(x.grad, grad_out), (
        f"gradient is not identity: max |grad - 1| = {(x.grad - grad_out).abs().max().item()}"
    )


def test_int_ste_rejects_invalid_bits() -> None:
    """bits ∉ [2, 8] is a programming error — raise."""
    x = torch.randn(4, 4)
    import pytest

    with pytest.raises(ValueError, match="bits must be in"):
        int_quant_ste(x, bits=1, axis=0)
    with pytest.raises(ValueError, match="bits must be in"):
        int_quant_ste(x, bits=9, axis=0)


def test_int_ste_rejects_scalar() -> None:
    """0-D tensor has no meaningful axis — raise."""
    import pytest

    with pytest.raises(ValueError, match="at least a 1-D tensor"):
        int_quant_ste(torch.tensor(1.0), bits=4, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# mxfp4_kv_ste — MXFP4 (E2M1 + E8M0)
# ─────────────────────────────────────────────────────────────────────────────


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=20, deadline=None)
def test_mxfp4_idempotence(seed: int) -> None:
    """quant(quant(x)) == quant(x) along the MXFP4 axis."""
    # Use a shape whose last axis is a multiple of 32 so no padding kicks in
    # — the test focuses on the snap-to-grid invariant, not padding logic.
    x = _random_tensor((4, 64), seed)
    q1 = mxfp4_kv_ste(x, axis=-1)
    q2 = mxfp4_kv_ste(q1, axis=-1)
    # Floating-point block-scaling has a tiny rounding window; allow a 1-ulp
    # slack rather than `torch.equal` to keep this robust on diverse seeds.
    assert torch.allclose(q1, q2, rtol=0, atol=1e-6), (
        f"mxfp4 idempotence failed: max diff = {(q1 - q2).abs().max().item()}"
    )


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=20, deadline=None)
def test_mxfp4_pads_uneven_axis(seed: int) -> None:
    """Axis size that's not a multiple of 32 still produces output of input shape."""
    x = _random_tensor((3, 47), seed)
    q = mxfp4_kv_ste(x, axis=-1)
    assert q.shape == x.shape


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=20, deadline=None)
def test_mxfp4_gradient_is_identity(seed: int) -> None:
    """STE: ∂y/∂x == 1 for MXFP4 too."""
    x = _random_tensor((4, 64), seed)
    x.requires_grad_(True)
    y = mxfp4_kv_ste(x, axis=-1)
    grad_out = torch.ones_like(y)
    y.backward(grad_out)
    assert x.grad is not None
    assert torch.equal(x.grad, grad_out)


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=20, deadline=None)
def test_mxfp4_per_block_scale_isolation(seed: int) -> None:
    """A block's scale depends only on its own elements — perturbing one block
    must NOT change the snapped values of a neighbouring block.

    Builds a tensor with two MXFP4 blocks (size 64 = 2x32) along axis -1; one
    block has a huge value, the other has small values. The small-value
    block's snap should be unaffected by the large-value block's presence.
    """
    x = torch.zeros(4, 64, dtype=torch.float32)
    x[:, :32] = _random_tensor((4, 32), seed) * 0.1   # small block
    x[:, 32:] = _random_tensor((4, 32), seed + 1) * 100.0  # large block
    q_full = mxfp4_kv_ste(x, axis=-1)

    # Now snap only the small block in isolation.
    x_small_only = torch.zeros(4, 32, dtype=torch.float32)
    x_small_only[:] = x[:, :32]
    q_small_solo = mxfp4_kv_ste(x_small_only, axis=-1)

    # The first block's snapped values must match the standalone snap.
    assert torch.allclose(q_full[:, :32], q_small_solo, rtol=0, atol=1e-6), (
        "MXFP4 block scale leaked across blocks: "
        f"max diff = {(q_full[:, :32] - q_small_solo).abs().max().item()}"
    )


def test_mxfp4_rejects_scalar() -> None:
    """0-D tensor has no axis — raise."""
    import pytest

    with pytest.raises(ValueError, match="at least a 1-D tensor"):
        mxfp4_kv_ste(torch.tensor(1.0), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Identity-on-representable-values
# ─────────────────────────────────────────────────────────────────────────────


def test_int_ste_identity_on_grid_value() -> None:
    """If `x` already lies on the per-slice grid, the snap is a no-op (within fp ε).

    Constructs a tensor whose values are exactly multiples of the per-slice
    scale that ``int_quant_ste`` would compute, then asserts ``q ≈ x``.
    """
    bits = 4
    # Per-slice scale: max-abs / qmax (qmax=7 for INT4). Build a slice with
    # max-abs = 7.0 so scale = 1.0.
    x = torch.tensor([
        [-7.0, -3.0, 0.0, 5.0, 7.0],
        [-3.5, 0.0, 1.5, 3.5, 0.5],   # max-abs = 3.5 → scale = 0.5
    ], dtype=torch.float32)
    # After scale = 0.5 for slice 1: each value /0.5 must be an integer in [-7, 7]:
    # -7, 0, 3, 7, 1 → all integer. So the snap should be a no-op.
    q = int_quant_ste(x, bits=bits, axis=0)
    assert torch.allclose(q, x, atol=1e-6), (
        f"identity-on-grid failed: x={x}, q={q}"
    )
