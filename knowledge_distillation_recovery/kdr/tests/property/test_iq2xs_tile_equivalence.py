"""LLR-0061: bit-identity of `_iq2xs_snap_block` across tile sizes.

The IQ2_XS argmin tile size was bumped from 64 → 8192 to amortise CUDA
kernel launch overhead. The tile is a memory-bounded chunking of the
SAME matmul + argmin; output must be bit-identical regardless of tile
size. This file pins that invariant.

# REQ: LLR-0061
# VERIFIES: LLR-0061
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from kdr.quant.native_backend.gguf_codebooks import (
    get_iq2xs_grid,
    get_ksigns_iq2xs,
)
from kdr.quant.native_backend.ste_simulators import (
    _iq2xs_snap_block,
)

_GGUF_SUPER_BLOCK = 256
_IQ2XS_SUB_BLOCK = 32
_IQ2XS_CHUNK = 8


def _snap_with_tile(super_blocks: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Reference replay of `_iq2xs_snap_block`'s loop with a configurable
    tile size. The body matches the production implementation byte-for-byte
    except for the literal tile constant — used to assert bit-equivalence
    across tile choices.
    """
    device = super_blocks.device
    dtype = super_blocks.dtype
    grid = get_iq2xs_grid(device, dtype)
    ksigns = get_ksigns_iq2xs(device, dtype)

    lead = super_blocks.shape[:-1]
    n_chunks_per_super = _GGUF_SUPER_BLOCK // _IQ2XS_CHUNK
    chunks = super_blocks.view(*lead, -1, n_chunks_per_super, _IQ2XS_CHUNK)
    sub = chunks.view(
        *lead, -1, _GGUF_SUPER_BLOCK // _IQ2XS_SUB_BLOCK, 4, _IQ2XS_CHUNK
    )
    sub_amax = sub.detach().abs().amax(dim=(-2, -1), keepdim=True)
    eps = torch.finfo(dtype).tiny
    grid_max = 43.0
    sub_scale = (sub_amax / grid_max).clamp(min=eps)
    chunks_norm = sub / sub_scale
    chunks_norm_flat = chunks_norm.view(*lead, -1, n_chunks_per_super, _IQ2XS_CHUNK)
    sub_scale_per_chunk = sub_scale.expand(
        *lead, -1, sub_scale.shape[-3], 4, 1
    ).reshape(*lead, -1, n_chunks_per_super, 1)

    flat = chunks_norm_flat.reshape(-1, _IQ2XS_CHUNK)
    n_total = flat.shape[0]
    snapped_chunks = torch.empty_like(flat)
    joint = grid.unsqueeze(1) * ksigns.unsqueeze(0)
    joint_flat = joint.view(512 * 128, _IQ2XS_CHUNK)
    code_sq = (joint_flat * joint_flat).sum(dim=-1)

    for start in range(0, n_total, tile_size):
        stop = min(start + tile_size, n_total)
        tile = flat[start:stop]
        inner = tile @ joint_flat.t()
        dist = code_sq.unsqueeze(0) - 2.0 * inner
        best = dist.argmin(dim=-1)
        snapped_chunks[start:stop] = joint_flat[best]

    snapped_unscaled = snapped_chunks.view(
        *lead, -1, n_chunks_per_super, _IQ2XS_CHUNK
    )
    snapped = snapped_unscaled * sub_scale_per_chunk
    return snapped.view(*lead, -1, _GGUF_SUPER_BLOCK)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-seed regression at curated shapes (LLR-0061 AC)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape",
    [
        (1, 256),      # smallest valid GGUF super-block
        (1, 512),      # two super-blocks
        (2, 4096),     # batch of 2, 16 super-blocks each
        (1, 8192),     # exactly the tile boundary (32 super-blocks)
    ],
)
def test_snap_block_bit_identical_to_old_tile_64(shape: tuple[int, int]) -> None:
    """LLR-0061 AC: new tile size produces bit-identical output to the
    pre-refactor tile=64 implementation across curated shapes."""
    torch.manual_seed(0xC0DE)
    # Inputs must be reshaped to (..., n_super, 256) — _iq2xs_snap_block's
    # contract. Build a flat tensor and reshape so the function sees the
    # super-block axis.
    rows, last = shape
    n_super = last // _GGUF_SUPER_BLOCK
    x = torch.randn(rows, n_super, _GGUF_SUPER_BLOCK)
    out_new = _iq2xs_snap_block(x)
    out_old = _snap_with_tile(x, tile_size=64)
    assert torch.equal(out_new, out_old), (
        f"tile-size refactor changed output for shape {shape}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis property: random shapes (multiples of 256) and seeds
# ─────────────────────────────────────────────────────────────────────────────


@settings(deadline=None, max_examples=16)
@given(
    n_super=st.integers(min_value=1, max_value=32),
    rows=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_snap_block_tile_invariance_hypothesis(
    n_super: int, rows: int, seed: int
) -> None:
    """LLR-0061 AC: across random shapes (last-axis = n_super × 256 in
    [256, 8192]) and seeds, the tile-size-8192 output equals the
    tile-size-64 reference output bit-for-bit."""
    torch.manual_seed(seed)
    x = torch.randn(rows, n_super, _GGUF_SUPER_BLOCK)
    out_new = _iq2xs_snap_block(x)
    out_old = _snap_with_tile(x, tile_size=64)
    assert torch.equal(out_new, out_old)
