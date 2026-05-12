"""Straight-through-estimator quant simulators (LLR-0015, LLR-0045).

Pure functions, each accepting an fp32/bf16 input tensor and returning a
fake-quantized tensor whose forward equals the dequant-quant snap and whose
backward is the identity over `x` (the STE pattern). The functions own
nothing; backend code calls them inside forward hooks.

Granularity is parametrised by `axis`: scales/zero-points are computed
*per-slice* along that axis (one scale per index of `axis`, shared across all
other axes) — except for the GGUF super-block STEs, where the super-block
layout (256-element groups) is itself the per-slice unit.

  * K per-channel along `head_dim`  → `axis = head_dim_axis` (typically -1)
  * V per-token along `seq_len`     → `axis = seq_len_axis` (typically 1)
  * GGUF super-block formats        → `axis = -1` (in-features axis of an
    `nn.Linear` weight; ggml super-blocks lie along the last contiguous axis)

Hypothesis property tests in ``tests/property/test_ste_simulators.py`` verify:

  * identity-on-forward (forward(x) ≈ x for representable x)
  * dequant-quant idempotence (quant(quant(x)) == quant(x))
  * granularity correctness (per-slice scales differ along `axis`)
  * gradient flow through the STE (d out / d x == 1)

Axis convention note (review L5): the existing :func:`int_quant_ste` uses
`axis=0` (per-output-channel scaling — natural for INT-N quant of
`nn.Linear.weight[out, in]`). The four new codebook STEs in this module
(:func:`iq2_xs_quant_ste`, :func:`q3_k_quant_ste`, :func:`iq4_xs_quant_ste`,
:func:`q5_k_quant_ste`) use `axis=-1` (the in-features axis of an
`nn.Linear` weight, where the GGUF super-block-of-256 layout lives — ggml
tensors are row-major with super-blocks along the last contiguous axis).
Both conventions are correct for their domains; the asymmetry is
intentional and should NOT be normalised by switching either side.
"""

# REQ: LLR-0015
# REQ: LLR-0045

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

from .gguf_codebooks import (
    KVALUES_IQ4NL,
    get_iq2xs_grid,
    get_ksigns_iq2xs,
)

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


# ─────────────────────────────────────────────────────────────────────────────
# GGUF super-block STE machinery (Phase 7.2 Task 4)
# ─────────────────────────────────────────────────────────────────────────────

# All GGUF super-block formats share this constant — 256 elements per
# super-block, per ggml's QK_K. The sub-block size varies per format.
_GGUF_SUPER_BLOCK: int = 256


def _block_quantize(
    x: torch.Tensor,
    axis: int,
    super_block_size: int,
    quant_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Move ``axis`` to last, reshape into super-blocks, apply ``quant_fn``
    per super-block, undo reshape + permute, return same-shape result.

    Raises ``ValueError`` if the size along ``axis`` is not a multiple of
    ``super_block_size`` (review H6 — Q5_K's asymmetric quant is sensitive
    to zero-padding bias, and Profile J's ZAYA1 axes are all multiples of
    256). A future v1 may add a format-aware padding path once a
    correctness gate against ggml's reference encoder exists.

    ``quant_fn`` operates on a tensor of shape ``(..., n_super_blocks,
    super_block_size)`` and returns the dequant'd tensor of the same
    shape. Internally everything runs in fp32 for numerical stability;
    the caller is responsible for casting back to the input dtype.

    The helper does NOT pad; padding is the caller's choice, and Task 4's
    choice is "raise".
    """
    if x.ndim == 0:
        raise ValueError(
            "GGUF super-block STE requires at least a 1-D tensor; got 0-D"
        )

    axis_pos = axis % x.ndim
    perm = [i for i in range(x.ndim) if i != axis_pos] + [axis_pos]
    x_perm = x.permute(perm).contiguous()

    last = x_perm.shape[-1]
    if last % super_block_size != 0:
        raise ValueError(
            f"_block_quantize: size along axis={axis} (post-permute "
            f"trailing axis = {last}) must be a multiple of "
            f"super_block_size={super_block_size}; got remainder "
            f"{last % super_block_size}. Profile-J/ZAYA1 axes are all "
            f"multiples of 256 by design; a future v1 may add a "
            f"format-aware padding path."
        )

    *lead, n = x_perm.shape
    n_super = n // super_block_size
    blocks = x_perm.reshape(*lead, n_super, super_block_size)

    snapped = quant_fn(blocks)
    snapped = snapped.reshape(*lead, n)

    inv_perm = [0] * x.ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return snapped.permute(inv_perm).contiguous()


def _ste_wrap(
    x: torch.Tensor, snap_fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Common STE wrapper: forward = snap, backward = identity over ``x``.

    Runs ``snap_fn`` in fp32 regardless of input dtype (the codebook
    constants are integers cast to fp32; doing the snap in fp32 keeps the
    reduction stable). The result is cast back to ``x.dtype`` at the very
    end.
    """
    snapped_fp32 = snap_fn(x.detach().float())
    snapped = snapped_fp32.to(x.dtype)
    return x + (snapped - x).detach()


# ─────────────────────────────────────────────────────────────────────────────
# IQ2_XS (2.3125 bpw) — codebook STE
# ─────────────────────────────────────────────────────────────────────────────

# IQ2_XS sub-block size (32 elements share one 4-bit scale inside a
# super-block of 256). Each sub-block holds 4 chunks of 8 elements; each
# chunk is encoded via the (magnitude, sign) codebook pair.
_IQ2XS_SUB_BLOCK: int = 32
_IQ2XS_CHUNK: int = 8

# IQ2_XS argmin tiling (review M5): the naive broadcast
#   (chunks[..., None, :] - grid[None, :, :]).norm(dim=-1).argmin(dim=-1)
# materialises a [n_chunks, 512] distance tensor per super-block. For a
# [4096, 4096] Linear weight that's about 268 MB fp32 in flight. We chunk
# the argmin to bound peak memory; the tile size is small enough that the
# cache line dominates the cost rather than DRAM bandwidth.
_IQ2XS_ARGMIN_TILE: int = 64


def _iq2xs_snap_block(super_blocks: torch.Tensor) -> torch.Tensor:
    """Snap ``super_blocks`` (shape ``(..., n_super, 256)``, fp32) to the
    IQ2_XS magnitude+sign codebook joint quantization grid.

    Per super-block:
      1. Split into 8 sub-blocks of 32 elements (the IQ2_XS scale grain).
      2. Per sub-block: 4-bit scale ``s = db_quant(amax_sub / 31)`` —
         we use the per-sub-block abs-max divided by the magnitude codebook
         max (43.0) since the codebook stores raw byte values 8/25/43.
      3. Per 8-element chunk: 65 536-way argmin against the joint
         (magnitudes x signs) codebook → pick the codeword that minimises
         per-chunk reconstruction error.
      4. Dequant = magnitudes * signs * sub_scale.

    Codebook-physics note: IQ2_XS encodes magnitudes drawn from
    ``{8, 25, 43}`` — there is NO zero codeword, so the worst-case
    mean reconstruction error against a Gaussian input is bounded
    below by ``8 / (2 * 43) ≈ 0.093`` of the per-sub-block max-abs
    (a typical noise-floor for the 2.3125-bpw representation). The
    forward-bound property test uses a slightly looser threshold that
    reflects this floor + a small slack for inter-sub-block max-abs
    variation; see ``test_iq2_xs_forward_bound`` for the rationale.
    """
    # Materialise codebooks lazily; cached per (device, dtype).
    device = super_blocks.device
    dtype = super_blocks.dtype  # fp32 by _ste_wrap contract
    grid = get_iq2xs_grid(device, dtype)  # (512, 8)
    ksigns = get_ksigns_iq2xs(device, dtype)  # (128, 8)

    # Reshape: (..., n_super, 256) -> (..., n_super * 8 sub-blocks, 32)
    lead = super_blocks.shape[:-1]
    # super_blocks is already (..., n_super, 256); split into chunks of 8.
    # Shape transform: keep all leading dims, then chunks of 8 per super-block.
    n_chunks_per_super = _GGUF_SUPER_BLOCK // _IQ2XS_CHUNK  # 32 chunks/super
    chunks = super_blocks.view(*lead, -1, n_chunks_per_super, _IQ2XS_CHUNK)
    # chunks shape: (..., n_super, 32 chunks, 8 elements/chunk)

    # Per sub-block (32 elements = 4 chunks): scale = amax / max_codebook_mag.
    # Reshape to expose sub-blocks: 32 chunks/super -> 8 sub-blocks * 4 chunks.
    sub = chunks.view(*lead, -1, _GGUF_SUPER_BLOCK // _IQ2XS_SUB_BLOCK, 4, _IQ2XS_CHUNK)
    # sub shape: (..., n_super, 8 sub-blocks, 4 chunks, 8 elements)
    sub_amax = sub.detach().abs().amax(dim=(-2, -1), keepdim=True)
    # Codebook max magnitude is 43 (= 0x2b). For an all-zero sub-block we
    # divide by eps so the chunk stays at zero after the argmin.
    eps = torch.finfo(dtype).tiny
    grid_max = 43.0
    sub_scale = (sub_amax / grid_max).clamp(min=eps)  # (..., n_super, 8, 1, 1)

    # Normalise chunks by their sub-block scale → values in roughly [-43, 43].
    chunks_norm = sub / sub_scale  # broadcast over (4, 8)

    # Flatten back to chunks dimension: (..., n_super, 32 chunks, 8 elements).
    chunks_norm_flat = chunks_norm.view(*lead, -1, n_chunks_per_super, _IQ2XS_CHUNK)
    sub_scale_per_chunk = sub_scale.expand(
        *lead, -1, sub_scale.shape[-3], 4, 1
    ).reshape(*lead, -1, n_chunks_per_super, 1)

    # Joint magnitude+sign argmin per chunk.
    # Best (magnitude, sign) pair minimises ||chunk_norm - mag * sign||^2.
    # We process chunks in tiles to bound the in-flight distance tensor.
    flat = chunks_norm_flat.reshape(-1, _IQ2XS_CHUNK)  # (total_chunks, 8)
    n_total = flat.shape[0]
    snapped_chunks = torch.empty_like(flat)

    # Precompute the full joint codebook: (512, 128, 8) = magnitudes * signs.
    # That's 512*128*8 = ~524 K fp32 entries = ~2 MB; fits easily in cache.
    joint = grid.unsqueeze(1) * ksigns.unsqueeze(0)  # (512, 128, 8)
    joint_flat = joint.view(512 * 128, _IQ2XS_CHUNK)  # (65536, 8)
    # Hoisted: invariant across tiles (depends only on the codebook).
    code_sq = (joint_flat * joint_flat).sum(dim=-1)  # (65536,)

    for start in range(0, n_total, _IQ2XS_ARGMIN_TILE):
        stop = min(start + _IQ2XS_ARGMIN_TILE, n_total)
        tile = flat[start:stop]  # (tile_n, 8)
        # Squared L2 distance: ||tile||^2 + ||code||^2 - 2 <tile, code>.
        # We only need argmin, so ||tile||^2 cancels and we compare
        # ||code||^2 - 2 <tile, code>.
        inner = tile @ joint_flat.t()  # (tile_n, 65536)
        dist = code_sq.unsqueeze(0) - 2.0 * inner  # (tile_n, 65536)
        best = dist.argmin(dim=-1)  # (tile_n,)
        # Decode best index → (magnitude_idx, sign_idx) and look up the
        # actual reconstructed chunk.
        snapped_chunks[start:stop] = joint_flat[best]

    # Rescale snapped chunks by sub-block scale and reshape back to super-block.
    snapped_unscaled = snapped_chunks.view(*lead, -1, n_chunks_per_super, _IQ2XS_CHUNK)
    snapped = snapped_unscaled * sub_scale_per_chunk

    return snapped.view(*lead, -1, _GGUF_SUPER_BLOCK)


def iq2_xs_quant_ste(x: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    """IQ2_XS (2.3125 bpw) codebook STE simulator.

    IQ2_XS is a TWO-codebook format (magnitude + sign). A 512-way
    magnitude-only argmin would force all outputs to positive magnitudes
    regardless of input sign; the joint encoding below also picks the
    per-chunk sign pattern from the 128-entry ``ksigns_iq2xs`` table.

    Per 8-element chunk inside a 32-element sub-block (8 sub-blocks per
    256-element super-block):

      1. Per-sub-block abs-max scale; divide.
      2. 512 magnitudes x 128 signs = 65 536 joint codewords; argmin
         against ``magnitudes * signs - chunk_normalised``.
      3. Dequant = ``magnitudes * signs * sub_scale``.

    Backward: identity over ``x`` (STE).

    Args:
        x: input tensor (fp32 or bf16). At least 1-D; the size along
            ``axis`` MUST be a multiple of 256 (raises ``ValueError``
            otherwise — see "Padding policy" in the module docstring).
        axis: axis along which the IQ2_XS super-blocks are laid out
            (typically the in-features axis of an ``nn.Linear`` weight,
            i.e. -1 by GGUF convention).

    Returns:
        Fake-quantized tensor with the same shape and dtype as ``x``.

    Source: ``quantize_row_iq2_xs`` / ``dequantize_row_iq2_xs`` in
    llama.cpp ``ggml/src/ggml-quants.c`` (commit pinned in
    :mod:`.gguf_codebooks`).
    """
    return _ste_wrap(
        x,
        lambda x_fp32: _block_quantize(
            x_fp32, axis=axis, super_block_size=_GGUF_SUPER_BLOCK,
            quant_fn=_iq2xs_snap_block,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Q3_K (3.4375 bpw) — non-codebook STE
# ─────────────────────────────────────────────────────────────────────────────

_Q3K_SUB_BLOCK: int = 16  # 16 sub-blocks of 16 elements per 256 super-block
_Q3K_QMIN: int = -4
_Q3K_QMAX: int = 3


def _q3k_snap_block(super_blocks: torch.Tensor) -> torch.Tensor:
    """Snap ``super_blocks`` (shape ``(..., n_super, 256)``, fp32) to the
    Q3_K grid.

    Q3_K stores 16 sub-blocks of 16 elements per super-block. Each
    super-block carries an FP16 ``d`` and 16 per-sub-block 6-bit signed
    scales (range [-32, 31]). Each element is a 3-bit signed integer in
    [-4, 3]. Dequant = ``d * sub_scale * element``.

    The simulator uses a simpler-but-equivalent fitting that captures the
    grid: per sub-block abs-max -> 6-bit signed scale (fitted relative to
    the super-block ``d``); per element 3-bit signed quant.
    """
    # Reshape super-block (256) -> 16 sub-blocks of 16.
    lead = super_blocks.shape[:-1]
    sub = super_blocks.view(*lead, -1, _GGUF_SUPER_BLOCK // _Q3K_SUB_BLOCK, _Q3K_SUB_BLOCK)
    # sub shape: (..., n_super, 16 sub-blocks, 16 elements)

    # Per sub-block scale (signed): abs-max / qmax (where qmax = 3 for 3-bit).
    eps = torch.finfo(super_blocks.dtype).tiny
    sub_amax = sub.detach().abs().amax(dim=-1, keepdim=True)
    sub_scale = (sub_amax / float(_Q3K_QMAX)).clamp(min=eps)

    # 3-bit signed snap, then rescale.
    q = torch.clamp(torch.round(sub / sub_scale), min=_Q3K_QMIN, max=_Q3K_QMAX)
    snapped_sub = q * sub_scale

    return snapped_sub.view(*lead, -1, _GGUF_SUPER_BLOCK)


def q3_k_quant_ste(x: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    """Q3_K (3.4375 bpw) STE simulator.

    Forward: per-super-block of 256 elements along ``axis``:
      1. Split into 16 sub-blocks of 16 elements.
      2. Per sub-block: signed scale = ``abs-max / 3``.
      3. 3-bit signed quant of each element via round-and-clamp to
         ``[-4, 3]``.
      4. Dequant = ``sub_scale * round_value``.

    Backward: identity over ``x`` (STE).

    Args:
        x: input tensor (fp32 or bf16). At least 1-D; the size along
            ``axis`` MUST be a multiple of 256.
        axis: axis along which the Q3_K super-blocks are laid out
            (typically -1 — the in-features axis).

    Returns:
        Fake-quantized tensor with the same shape and dtype as ``x``.

    Source: ``quantize_row_q3_K_ref`` in llama.cpp ``ggml/src/ggml-quants.c``
    (commit pinned in :mod:`.gguf_codebooks`).
    """
    return _ste_wrap(
        x,
        lambda x_fp32: _block_quantize(
            x_fp32, axis=axis, super_block_size=_GGUF_SUPER_BLOCK,
            quant_fn=_q3k_snap_block,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# IQ4_XS (4.25 bpw) — non-linear 4-bit codebook STE
# ─────────────────────────────────────────────────────────────────────────────

_IQ4XS_SUB_BLOCK: int = 32  # 8 sub-blocks of 32 elements per super-block


def _iq4xs_snap_block(super_blocks: torch.Tensor) -> torch.Tensor:
    """Snap ``super_blocks`` to IQ4_XS: per sub-block of 32 elements, 4-bit
    unsigned index into the 16-entry signed ``kvalues_iq4nl`` codebook.

    Per super-block of 256:
      1. Split into 8 sub-blocks of 32.
      2. Per sub-block scale derived from BOTH codebook extremes, since
         ``kvalues_iq4nl`` is asymmetric: ``min(k) = -127``, ``max(k) = 113``.
         ``scale = max(pos_max / 113, |neg_min| / 127)`` guarantees both
         extremes fit inside the codebook range without one-sided clipping
         (a uniform ``scale = amax / 127`` would crush the positive arm and
         introduce a 12% extra error on positive outliers).
      3. Per element: argmin over the 16-entry codebook.
      4. Dequant = sub_scale * codebook_value.
    """
    lead = super_blocks.shape[:-1]
    sub = super_blocks.view(
        *lead, -1, _GGUF_SUPER_BLOCK // _IQ4XS_SUB_BLOCK, _IQ4XS_SUB_BLOCK
    )
    # sub shape: (..., n_super, 8 sub-blocks, 32 elements)

    device = super_blocks.device
    dtype = super_blocks.dtype
    kvalues = torch.tensor(KVALUES_IQ4NL, dtype=dtype, device=device)  # (16,)
    # Asymmetric range: negative arm is wider than positive arm.
    kmax_pos = float(max(KVALUES_IQ4NL))   # 113
    kmax_neg = float(-min(KVALUES_IQ4NL))  # 127

    eps = torch.finfo(dtype).tiny
    sub_max = sub.detach().amax(dim=-1, keepdim=True).clamp(min=0.0)
    sub_min = sub.detach().amin(dim=-1, keepdim=True).clamp(max=0.0)
    # The "no-clip" scale: anything smaller pushes one extreme outside the
    # codebook range and forces a hard clip on that side.
    sub_scale = torch.maximum(
        sub_max / kmax_pos, (-sub_min) / kmax_neg
    ).clamp(min=eps)

    sub_norm = sub / sub_scale  # values inside [kmin, kmax] of the codebook

    # Argmin over the 16 levels: |sub_norm - kvalues|.
    dists = (sub_norm.unsqueeze(-1) - kvalues).abs()  # (..., 32, 16)
    idx = dists.argmin(dim=-1)  # (..., 32)
    snapped_norm = kvalues[idx]  # gather → (..., 32)

    snapped_sub = snapped_norm * sub_scale
    return snapped_sub.view(*lead, -1, _GGUF_SUPER_BLOCK)


def iq4_xs_quant_ste(x: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    """IQ4_XS (4.25 bpw) codebook STE simulator.

    Forward: per-super-block of 256 elements along ``axis``:
      1. Split into 8 sub-blocks of 32 elements.
      2. Per sub-block: abs-max-scaled (divided by 127, the codebook max).
      3. Per element: 4-bit unsigned index into ``KVALUES_IQ4NL``.
      4. Dequant = ``sub_scale * KVALUES_IQ4NL[index]``.

    Backward: identity over ``x`` (STE).

    Args:
        x: input tensor (fp32 or bf16). At least 1-D; the size along
            ``axis`` MUST be a multiple of 256.
        axis: axis along which the IQ4_XS super-blocks are laid out
            (typically -1).

    Returns:
        Fake-quantized tensor with the same shape and dtype as ``x``.

    Source: ``quantize_row_iq4_xs`` / ``dequantize_row_iq4_xs`` in
    llama.cpp ``ggml/src/ggml-quants.c`` (commit pinned in
    :mod:`.gguf_codebooks`).
    """
    return _ste_wrap(
        x,
        lambda x_fp32: _block_quantize(
            x_fp32, axis=axis, super_block_size=_GGUF_SUPER_BLOCK,
            quant_fn=_iq4xs_snap_block,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Q5_K (5.5 bpw) — asymmetric per-sub-block STE
# ─────────────────────────────────────────────────────────────────────────────

_Q5K_SUB_BLOCK: int = 32
_Q5K_QMAX: int = 31  # 5-bit unsigned: [0, 31]


def _q5k_snap_block(super_blocks: torch.Tensor) -> torch.Tensor:
    """Snap to Q5_K: per sub-block of 32 elements, asymmetric 5-bit
    unsigned quant via (scale, min).

    Per super-block of 256:
      1. Split into 8 sub-blocks of 32.
      2. Per sub-block: sub_min = min, sub_max = max.
      3. sub_scale = (sub_max - sub_min) / 31 (clamped via eps).
      4. q = round((x - sub_min) / sub_scale) clamped to [0, 31].
      5. Dequant = sub_scale * q + sub_min.
    """
    lead = super_blocks.shape[:-1]
    sub = super_blocks.view(
        *lead, -1, _GGUF_SUPER_BLOCK // _Q5K_SUB_BLOCK, _Q5K_SUB_BLOCK
    )

    eps = torch.finfo(super_blocks.dtype).tiny
    sub_min = sub.detach().amin(dim=-1, keepdim=True)
    sub_max = sub.detach().amax(dim=-1, keepdim=True)
    sub_scale = ((sub_max - sub_min) / float(_Q5K_QMAX)).clamp(min=eps)

    q = torch.clamp(
        torch.round((sub - sub_min) / sub_scale), min=0.0, max=float(_Q5K_QMAX)
    )
    snapped_sub = sub_scale * q + sub_min

    return snapped_sub.view(*lead, -1, _GGUF_SUPER_BLOCK)


def q5_k_quant_ste(x: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    """Q5_K (5.5 bpw) STE simulator.

    Forward: per-super-block of 256 elements along ``axis``:
      1. Split into 8 sub-blocks of 32.
      2. Per sub-block: ``sub_min = min(x)``, ``sub_max = max(x)``,
         ``sub_scale = (sub_max - sub_min) / 31``.
      3. Per element: 5-bit unsigned quant ``q = round((x - sub_min) / sub_scale)``
         clamped to [0, 31].
      4. Dequant = ``sub_scale * q + sub_min``.

    Backward: identity over ``x`` (STE).

    Args:
        x: input tensor (fp32 or bf16). At least 1-D; the size along
            ``axis`` MUST be a multiple of 256 (Q5_K is asymmetric and the
            naive zero-pad distorts the per-block ``dmin`` fit).
        axis: axis along which the Q5_K super-blocks are laid out
            (typically -1).

    Returns:
        Fake-quantized tensor with the same shape and dtype as ``x``.

    Source: ``quantize_row_q5_K_ref`` in llama.cpp ``ggml/src/ggml-quants.c``
    (commit pinned in :mod:`.gguf_codebooks`).
    """
    return _ste_wrap(
        x,
        lambda x_fp32: _block_quantize(
            x_fp32, axis=axis, super_block_size=_GGUF_SUPER_BLOCK,
            quant_fn=_q5k_snap_block,
        ),
    )
