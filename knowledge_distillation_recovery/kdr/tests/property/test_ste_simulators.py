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

from kdr.quant.native_backend.gguf_codebooks import (
    KSIGNS_IQ2XS,
    KVALUES_IQ4NL,
    get_iq2xs_grid,
    get_ksigns_iq2xs,
)
from kdr.quant.native_backend.ste_simulators import (
    int_quant_ste,
    iq2_xs_quant_ste,
    iq4_xs_quant_ste,
    mxfp4_kv_ste,
    q3_k_quant_ste,
    q5_k_quant_ste,
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


# ─────────────────────────────────────────────────────────────────────────────
# GGUF codebook STE simulators (Phase 7.2 Task 4)
# ─────────────────────────────────────────────────────────────────────────────

# Shape strategy for GGUF formats: super-block size 256 along the last axis.
_GGUF_SHAPES = st.tuples(
    st.integers(min_value=1, max_value=4),
    st.sampled_from([256, 512]),
)


# ─────────────────────────────────────────────────────────────────────────────
# Codebook transcription guards (B.T2)
# ─────────────────────────────────────────────────────────────────────────────


def test_iq2_xs_magnitude_grid_loaded() -> None:
    """``get_iq2xs_grid(...)`` returns a (512, 8) tensor with the canonical
    entry-0 packing (= eight magnitudes of 8 → ``0x0808080808080808``)."""
    grid = get_iq2xs_grid(torch.device("cpu"), torch.float32)
    assert grid.shape == (512, 8)
    # Entry 0: all eight bytes are 0x08 = 8.
    assert torch.equal(grid[0], torch.full((8,), 8.0))
    # Last entry of ggml's iq2xs_grid is 0x2b2b2b2b2b2b2b2b (eight 43s).
    assert torch.equal(grid[511], torch.full((8,), 43.0))
    # Magnitudes are drawn from the 3-element set {8, 25, 43}.
    unique_mags = torch.unique(grid).tolist()
    assert unique_mags == [8.0, 25.0, 43.0]


def test_iq2_xs_sign_codebook_loaded() -> None:
    """``KSIGNS_IQ2XS`` has 128 entries, even parity per row, with spot-
    checks against the source bit-packed values."""
    assert len(KSIGNS_IQ2XS) == 128
    ks = get_ksigns_iq2xs(torch.device("cpu"), torch.float32)
    assert ks.shape == (128, 8)
    # Entry 0 = all-positive (bit pattern 0 → all signs +1).
    assert torch.equal(ks[0], torch.full((8,), 1.0))
    # Entry 127 (last) = all bits set (0xFF) → all signs -1.
    assert torch.equal(ks[127], torch.full((8,), -1.0))
    # Even parity: product of signs per row is +1.
    parities = ks.prod(dim=-1)
    assert torch.equal(parities, torch.full((128,), 1.0)), (
        f"non-even parity rows: {(parities != 1.0).sum().item()}"
    )


def test_iq4nl_codebook_loaded() -> None:
    """KVALUES_IQ4NL must match the canonical 16-entry signed table."""
    assert KVALUES_IQ4NL == (
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113,
    )


# ─────────────────────────────────────────────────────────────────────────────
# IQ2_XS property tests
# ─────────────────────────────────────────────────────────────────────────────


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=8, deadline=None)
def test_iq2_xs_idempotence(rows: int, cols: int, seed: int) -> None:
    """Re-snapping a snapped tensor is bounded by the original snap error.

    Strict bit-equal idempotence does not hold for IQ2_XS because the
    sub-block amax shifts on already-snapped values (the joint
    magnitude+sign argmin may pick a different codeword when the
    fitted scale recovers from ``43 * scale`` to ``max_used * scale``
    for ``max_used in {8, 25, 43}``). The looser stability invariant we
    can guarantee: ``||q(q(x)) - q(x)|| <= ||q(x) - x||`` — second snap
    cannot ADD more error than the first. A bound exceedance here would
    still flag a wholesale codebook transcription bug.
    """
    x = _random_tensor((rows, cols), seed)
    q1 = iq2_xs_quant_ste(x, axis=-1)
    q2 = iq2_xs_quant_ste(q1, axis=-1)
    first_err = (q1 - x).abs().mean().item()
    second_step = (q2 - q1).abs().mean().item()
    assert second_step <= first_err + 1e-4, (
        f"iq2_xs re-snap added error: |q1-x|.mean={first_err:.4e}, "
        f"|q2-q1|.mean={second_step:.4e}"
    )


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=8, deadline=None)
def test_iq2_xs_forward_bound(rows: int, cols: int, seed: int) -> None:
    """Average |q(x) - x| <= per-super-block max-abs * 0.15.

    Codebook-physics floor: IQ2_XS magnitudes are drawn from ``{8, 25, 43}``
    (no zero codeword), so the worst-case error on a value snapping from
    near zero to the ``8 * scale`` codeword is bounded below by
    ``8 / (2 * 43) ≈ 0.093 * sub_amax``. Empirically, with this
    non-iterative encoder, the mean reconstruction error per super-block
    is ``~0.10 * super_amax`` on Gaussian inputs and edges up to ``~0.11``
    on adversarial Hypothesis-shrunk seeds (single super-block, amax
    concentrated in one sub-block).

    The spec sketch suggested 0.10; we use a slightly looser 0.15 to
    absorb the codebook noise floor plus inter-sub-block amax variation
    without false alarms on the property fuzzer. A wholesale
    transcription error in the magnitude or sign codebook would still
    blow this bound up to O(amax) (the original 0.30 was unnecessarily
    loose; the new 0.15 still rejects the C2 / B.T1 property-2 bug
    class without producing flaky failures). See ``_iq2xs_snap_block``
    docstring for the codebook-physics note."""
    x = _random_tensor((rows, cols), seed)
    q = iq2_xs_quant_ste(x, axis=-1)
    # Super-block amax: shape-aware max-abs over each 256-element block.
    n_super = cols // 256
    blocks = x.view(rows, n_super, 256)
    amax_per_block = blocks.abs().amax(dim=-1)  # (rows, n_super)
    # Mean absolute reconstruction error per super-block: a more stable
    # statistic than worst-case for a low-bpw codebook.
    diff = (q - x).abs().view(rows, n_super, 256)
    mean_err = diff.mean(dim=-1)  # (rows, n_super)
    bound = 0.15 * amax_per_block
    assert (mean_err <= bound + 1e-4).all(), (
        f"iq2_xs forward bound exceeded: max mean-err = {mean_err.max().item():.4f}, "
        f"min bound = {bound.min().item():.4f}"
    )


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=8, deadline=None)
def test_iq2_xs_gradient_is_identity(rows: int, cols: int, seed: int) -> None:
    """STE: ∂y/∂x == 1."""
    x = _random_tensor((rows, cols), seed)
    x.requires_grad_(True)
    y = iq2_xs_quant_ste(x, axis=-1)
    grad_out = torch.ones_like(y)
    y.backward(grad_out)
    assert x.grad is not None
    assert torch.equal(x.grad, grad_out)


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=8, deadline=None)
def test_iq2_xs_shape_preserved(rows: int, cols: int, seed: int) -> None:
    x = _random_tensor((rows, cols), seed)
    q = iq2_xs_quant_ste(x, axis=-1)
    assert q.shape == x.shape


def test_iq2_xs_preserves_sign() -> None:
    """All-positive input → element-wise non-negative output; all-negative
    input → element-wise non-positive output (review H7).

    Catches the C2-class bug (missing sign codebook) directly: a
    magnitude-only encoder maps negative inputs to positive outputs."""
    pos = torch.rand(2, 256, dtype=torch.float32) + 0.1
    out_pos = iq2_xs_quant_ste(pos, axis=-1)
    assert (out_pos >= 0).all(), (
        f"sign preservation (positive) failed: min={out_pos.min().item()}"
    )

    neg = -(torch.rand(2, 256, dtype=torch.float32) + 0.1)
    out_neg = iq2_xs_quant_ste(neg, axis=-1)
    assert (out_neg <= 0).all(), (
        f"sign preservation (negative) failed: max={out_neg.max().item()}"
    )


def test_iq2_xs_rejects_non_multiple_of_256() -> None:
    """Axis size that's not a multiple of 256 → ValueError per H6."""
    import pytest

    with pytest.raises(ValueError, match="must be a multiple of"):
        iq2_xs_quant_ste(torch.randn(2, 100), axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Q3_K property tests
# ─────────────────────────────────────────────────────────────────────────────


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_q3_k_idempotence(rows: int, cols: int, seed: int) -> None:
    x = _random_tensor((rows, cols), seed)
    q1 = q3_k_quant_ste(x, axis=-1)
    q2 = q3_k_quant_ste(q1, axis=-1)
    # Q3_K's symmetric grid is closed under round+clamp on already-snapped
    # values, so bit-exact idempotence holds (modulo fp ε).
    assert torch.allclose(q2, q1, atol=1e-5), (
        f"q3_k idempotence diff = {(q2 - q1).abs().max().item():.4e}"
    )


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_q3_k_forward_bound(rows: int, cols: int, seed: int) -> None:
    """|q(x) - x| <= per-sub-block max-abs / 4."""
    x = _random_tensor((rows, cols), seed)
    q = q3_k_quant_ste(x, axis=-1)
    # Per-sub-block amax: 16 sub-blocks of 16 elements per 256 super-block.
    n_super = cols // 256
    sub = x.view(rows, n_super, 16, 16)
    sub_amax = sub.abs().amax(dim=-1, keepdim=True)
    bound = (sub_amax / 4.0).expand_as(sub).reshape(rows, cols)
    diff = (q - x).abs()
    assert (diff <= bound + 1e-4).all(), (
        f"q3_k forward bound exceeded: max diff/bound = "
        f"{(diff / bound.clamp(min=1e-6)).max().item():.4f}"
    )


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_q3_k_gradient_is_identity(rows: int, cols: int, seed: int) -> None:
    x = _random_tensor((rows, cols), seed)
    x.requires_grad_(True)
    y = q3_k_quant_ste(x, axis=-1)
    grad_out = torch.ones_like(y)
    y.backward(grad_out)
    assert x.grad is not None
    assert torch.equal(x.grad, grad_out)


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_q3_k_shape_preserved(rows: int, cols: int, seed: int) -> None:
    x = _random_tensor((rows, cols), seed)
    q = q3_k_quant_ste(x, axis=-1)
    assert q.shape == x.shape


def test_q3_k_rejects_non_multiple_of_256() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be a multiple of"):
        q3_k_quant_ste(torch.randn(2, 100), axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# IQ4_XS property tests
# ─────────────────────────────────────────────────────────────────────────────


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_iq4_xs_idempotence(rows: int, cols: int, seed: int) -> None:
    """quant(quant(x)) ≈ quant(x) — the codebook + per-sub-block scale
    fitting recovers a slightly different scale on re-snap when no input
    in the sub-block lands on the extreme codeword ±127. Tolerance set
    accordingly."""
    x = _random_tensor((rows, cols), seed)
    q1 = iq4_xs_quant_ste(x, axis=-1)
    q2 = iq4_xs_quant_ste(q1, axis=-1)
    # Generous tolerance: codebook level spacing x sub-block scale.
    diff = (q2 - q1).abs().max().item()
    amax = x.abs().amax().item()
    assert diff <= 0.30 * amax + 1e-5, (
        f"iq4_xs idempotence diff = {diff:.4e} (amax={amax:.4f})"
    )


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_iq4_xs_forward_bound(rows: int, cols: int, seed: int) -> None:
    """|q(x) - x| roughly within per-sub-block max-abs x 0.15.

    Codebook-physics floor: ``kvalues_iq4nl`` has widest gap of 24
    (between 89 and 113); the worst-case half-gap error is
    ``12 / kmax_pos = 12 / 113 ≈ 0.106 * sub_amax``. The 0.15 threshold
    sits comfortably above this floor while still rejecting the
    C2 / B.T1 property-2 bug class (a transcription-broken codebook
    would exceed ``sub_amax``).

    The spec sketch suggested 0.10 (tighter than the codebook physics
    permit on the positive arm); the original loose 0.25 was wider than
    needed and tolerated the pre-fix asymmetric-clip bug. The 0.15
    threshold captures both improvements: tight enough to reject the
    asymmetric-clip class, loose enough to absorb the 24-spaced
    codebook gap without flaky failures on Hypothesis-shrunk seeds."""
    x = _random_tensor((rows, cols), seed)
    q = iq4_xs_quant_ste(x, axis=-1)
    n_super = cols // 256
    sub = x.view(rows, n_super, 8, 32)
    sub_amax = sub.abs().amax(dim=-1, keepdim=True)
    bound = (sub_amax * 0.15).expand_as(sub).reshape(rows, cols)
    diff = (q - x).abs()
    assert (diff <= bound + 1e-4).all(), (
        f"iq4_xs forward bound exceeded: max diff = {diff.max().item():.4f}"
    )


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_iq4_xs_gradient_is_identity(rows: int, cols: int, seed: int) -> None:
    x = _random_tensor((rows, cols), seed)
    x.requires_grad_(True)
    y = iq4_xs_quant_ste(x, axis=-1)
    grad_out = torch.ones_like(y)
    y.backward(grad_out)
    assert x.grad is not None
    assert torch.equal(x.grad, grad_out)


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_iq4_xs_shape_preserved(rows: int, cols: int, seed: int) -> None:
    x = _random_tensor((rows, cols), seed)
    q = iq4_xs_quant_ste(x, axis=-1)
    assert q.shape == x.shape


def test_iq4_xs_rejects_non_multiple_of_256() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be a multiple of"):
        iq4_xs_quant_ste(torch.randn(2, 100), axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Q5_K property tests
# ─────────────────────────────────────────────────────────────────────────────


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_q5_k_idempotence(rows: int, cols: int, seed: int) -> None:
    """Asymmetric (scale, min) fit on already-snapped values reproduces
    the same grid points (within fp ε)."""
    x = _random_tensor((rows, cols), seed)
    q1 = q5_k_quant_ste(x, axis=-1)
    q2 = q5_k_quant_ste(q1, axis=-1)
    assert torch.allclose(q2, q1, atol=1e-5), (
        f"q5_k idempotence diff = {(q2 - q1).abs().max().item():.4e}"
    )


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_q5_k_forward_bound(rows: int, cols: int, seed: int) -> None:
    """|q(x) - x| <= per-sub-block range / (2 * 31) ≈ amax / 16."""
    x = _random_tensor((rows, cols), seed)
    q = q5_k_quant_ste(x, axis=-1)
    n_super = cols // 256
    sub = x.view(rows, n_super, 8, 32)
    sub_range = sub.amax(dim=-1, keepdim=True) - sub.amin(dim=-1, keepdim=True)
    # Worst-case 5-bit unsigned quant error is half a step.
    bound = (sub_range / (2 * 31.0)).expand_as(sub).reshape(rows, cols)
    diff = (q - x).abs()
    assert (diff <= bound + 1e-4).all(), (
        f"q5_k forward bound exceeded: max diff = {diff.max().item():.4f}"
    )


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_q5_k_gradient_is_identity(rows: int, cols: int, seed: int) -> None:
    x = _random_tensor((rows, cols), seed)
    x.requires_grad_(True)
    y = q5_k_quant_ste(x, axis=-1)
    grad_out = torch.ones_like(y)
    y.backward(grad_out)
    assert x.grad is not None
    assert torch.equal(x.grad, grad_out)


@given(rows=st.integers(min_value=1, max_value=3),
       cols=st.sampled_from([256, 512]),
       seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=10, deadline=None)
def test_q5_k_shape_preserved(rows: int, cols: int, seed: int) -> None:
    x = _random_tensor((rows, cols), seed)
    q = q5_k_quant_ste(x, axis=-1)
    assert q.shape == x.shape


def test_q5_k_rejects_non_multiple_of_256() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be a multiple of"):
        q5_k_quant_ste(torch.randn(2, 100), axis=-1)
