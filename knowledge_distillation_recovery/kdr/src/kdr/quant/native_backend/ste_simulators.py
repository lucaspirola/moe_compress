"""Straight-through-estimator quant simulators for INT3, INT2, MXFP4-KV (LLR-0015).

Three pure functions, each accepting an fp32/bf16 input tensor and returning a
fake-quantized tensor whose forward equals the dequant-quant snap and whose
backward is the identity over `x` (the STE pattern). The functions own
nothing; backend code calls them inside forward hooks.

Granularity is parametrised by `axis`: scales/zero-points are computed
*per-slice* along that axis (one scale per index of `axis`, shared across all
other axes).

  * K per-channel along `head_dim`  → `axis = head_dim_axis` (typically -1)
  * V per-token along `seq_len`     → `axis = seq_len_axis` (typically 1)

Hypothesis property tests in ``tests/property/test_ste_simulators.py`` verify:

  * identity-on-forward (forward(x) ≈ x for representable x)
  * dequant-quant idempotence (quant(quant(x)) == quant(x))
  * granularity correctness (per-slice scales differ along `axis`)
  * gradient flow through the STE (d out / d x == 1)
"""

# REQ: LLR-0015

from __future__ import annotations

from typing import cast

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Symmetric integer STE (INT8 / INT4 / INT3 / INT2)
# ─────────────────────────────────────────────────────────────────────────────


def int_quant_ste(x: torch.Tensor, bits: int, *, axis: int) -> torch.Tensor:
    """Symmetric integer fake-quantization with straight-through gradient.

    Forward: ``round(clamp(x / scale, qmin, qmax)) * scale`` per slice along
    ``axis``. Backward: ``∂y/∂x = 1`` so gradients flow as identity.

    Args:
        x: input tensor (fp32 or bf16). At least 1-D.
        bits: integer bit-width in [2, 8]. Sign bit included → integer range
            is ``[-2**(bits-1), 2**(bits-1) - 1]``.
        axis: axis along which scales differ (one scalar scale per index of
            ``axis``, shared across all other axes). Negative values index
            from the end (PyTorch convention).

    Returns:
        Fake-quantized tensor with the same shape and dtype as ``x``.

    Raises:
        ValueError: if ``bits`` is outside [2, 8] or ``x.ndim == 0``.
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8]; got {bits}")
    if x.ndim == 0:
        raise ValueError("int_quant_ste requires at least a 1-D tensor")

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    axis_pos = axis % x.ndim
    reduce_axes = tuple(i for i in range(x.ndim) if i != axis_pos)

    # Per-slice abs-max → per-slice scale. `keepdim=True` preserves broadcasting.
    # `.detach()` keeps the scale out of the autograd graph (it's data-derived
    # but not a learnable parameter).
    abs_max = x.detach().abs().amax(dim=reduce_axes, keepdim=True)
    eps = torch.finfo(x.dtype).tiny
    scale = (abs_max / qmax).clamp(min=eps)

    q = torch.clamp(torch.round(x / scale), min=qmin, max=qmax)
    x_q = q * scale

    # STE: forward == x_q (by construction), backward == identity over x.
    # `(x_q - x).detach()` has zero gradient → ∂(x + (x_q - x).detach())/∂x = 1.
    return cast(torch.Tensor, x + (x_q - x).detach())


# ─────────────────────────────────────────────────────────────────────────────
# MXFP4 (E2M1 + E8M0 power-of-two block scales) STE
# ─────────────────────────────────────────────────────────────────────────────

# E2M1 representable magnitudes (sign bit handled separately): 8 distinct
# non-negative values. `0.0` appears once; the rest are the standard FP4-E2M1
# subnormal+normal grid.
_E2M1_POSITIVE: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX: float = 6.0

# OCP MXFP4 block size: 32 elements share one E8M0 scale.
_MXFP4_BLOCK_SIZE: int = 32


def _snap_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Snap each element to the nearest signed E2M1 representable value.

    E2M1 has 8 positive levels (incl. 0); the negative half is the mirror.
    Snap-to-nearest is implemented via a single broadcast-distance argmin
    against the level table (fast for small blocks, no float-trickery).
    """
    levels = torch.tensor(_E2M1_POSITIVE, device=x.device, dtype=x.dtype)
    abs_x = x.abs()
    sign = torch.sign(x)
    # Distances: |abs_x - levels|, shape (..., 8). argmin over the last axis.
    dists = (abs_x.unsqueeze(-1) - levels).abs()
    nearest_idx = dists.argmin(dim=-1)
    nearest_mag = levels[nearest_idx]
    return sign * nearest_mag


def mxfp4_kv_ste(x: torch.Tensor, *, axis: int) -> torch.Tensor:
    """MXFP4 (E2M1 mantissa + E8M0 power-of-two block scale) STE simulator.

    Used only when ``feature_matrix.SUPPORTED_QUANTS`` says modelopt's
    installed version lacks MXFP4-KV support. The OCP MXFP4 layout shares one
    E8M0 scale across each block of ``_MXFP4_BLOCK_SIZE = 32`` elements along
    ``axis``; within a block each element is E2M1.

    Forward: snap each block to (E2M1 ⊗ E8M0); backward: identity over ``x``.

    Args:
        x: input tensor.
        axis: axis along which the MXFP4 blocks are laid out (groups of 32
            elements). Padding to a multiple of 32 is handled internally.

    Returns:
        Fake-quantized tensor with the same shape and dtype as ``x``.
    """
    if x.ndim == 0:
        raise ValueError("mxfp4_kv_ste requires at least a 1-D tensor")

    axis_pos = axis % x.ndim
    # Move `axis` to the last position so blocks lie along the trailing axis.
    perm = [i for i in range(x.ndim) if i != axis_pos] + [axis_pos]
    x_perm = x.permute(perm).contiguous()

    last = x_perm.shape[-1]
    pad_n = (-last) % _MXFP4_BLOCK_SIZE
    if pad_n:
        x_perm = torch.nn.functional.pad(x_perm, (0, pad_n))

    *lead, n = x_perm.shape
    blocks = x_perm.reshape(*lead, n // _MXFP4_BLOCK_SIZE, _MXFP4_BLOCK_SIZE)

    # Per-block max-abs → choose the smallest power-of-two scale `s` such
    # that `block / s` fits within E2M1's max magnitude (6.0). E8M0 stores
    # `log2(s)` as a signed integer; rounding via `ceil(log2(...))` gives the
    # smallest valid power-of-two.
    block_max = blocks.detach().abs().amax(dim=-1, keepdim=True)
    # `tiny` keeps log2 finite when a block is all-zero; the resulting scale
    # is effectively zero in significance (block stays at 0).
    eps = torch.finfo(x.dtype).tiny
    scale_log2 = torch.ceil(torch.log2((block_max / _E2M1_MAX).clamp(min=eps)))
    scale = torch.pow(2.0, scale_log2)

    snapped = _snap_to_e2m1(blocks / scale) * scale
    snapped = snapped.reshape(*lead, n)

    if pad_n:
        snapped = snapped[..., :last]

    # Inverse permute.
    inv_perm = [0] * x.ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    snapped = snapped.permute(inv_perm).contiguous()

    return x + (snapped - x).detach()
