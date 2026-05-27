"""Opt B1 — output-space perm_cache write tests.

Per SC_FAST_PLAN_V3.md §4-B1 lines 211–229: ``_tentative_merged_weights``
persists the freshly-computed Hungarian permutation under
``(li, centroid_id, child_id)`` on cache-miss so the downstream merge step
(`merging.py:140`) and the EM-refine pass (`em_refine.py:190`) can reuse it
instead of re-running ``scipy.optimize.linear_sum_assignment`` for the same
expert pair. Side-effect only: cost matrix is byte-identical.

These tests cover:
  T1 — cache-miss path actually writes to the cache.
  T2 — cache-hit path returns the same merged weights as the original miss
       (the stored perm matches the perm used to construct the merge).
  T3 — cache-hit path does NOT overwrite an existing entry (idempotency).
  T4 — ``perm_cache=None`` path is unaffected (the write is guarded).

All tests are CPU-only and use a synthetic 2-expert top-1 MoE layer
(hidden=4, d_int=3, n_exp=2, top_k=1) mirroring
``test_output_cost_hand_checked_scalar`` in ``test_stage2_output_cost.py``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from moe_compress.stage2.permutation_align import _PermAlignCache
from moe_compress.stage2.plugins.output_space_cost import _tentative_merged_weights
from moe_compress.utils.model_io import MATRIX_NAMES, MoELayerRef, build_banks


# ---------------------------------------------------------------------------
# Synthetic-MoE fixture (mirrors test_output_cost_hand_checked_scalar)
# ---------------------------------------------------------------------------


def _make_layer_ref(seed: int = 7) -> MoELayerRef:
    """Build a minimal 2-expert top-1 MoE layer for cache-write tests.

    Matches the fused-experts shape (gate_up_proj + down_proj parameters)
    that ``build_banks`` knows how to decompose into per-expert
    gate_proj / up_proj / down_proj views.
    """
    hidden, d_int, n_exp, top_k = 4, 3, 2, 1
    torch.manual_seed(seed)

    class _Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = n_exp
            # fused gate_up: (n_exp, 2*d_int, hidden); down: (n_exp, hidden, d_int)
            self.gate_up_proj = nn.Parameter(torch.randn(n_exp, 2 * d_int, hidden))
            self.down_proj = nn.Parameter(torch.randn(n_exp, hidden, d_int))

    class _Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = top_k
            self.hidden_dim = hidden
            w = torch.zeros(n_exp, hidden)
            w[0, 0] = 50.0
            self.weight = nn.Parameter(w)

    class _MLP(nn.Module):
        def __init__(self, experts, router):
            super().__init__()
            self.experts = experts
            self.gate = router

    experts = _Experts()
    router = _Router()
    mlp = _MLP(experts, router)
    return MoELayerRef(
        layer_idx=0, layer_module=mlp, mlp=mlp, router=router,
        experts_module=experts, shared_expert=None, layer_type="full_attention",
    )


# ---------------------------------------------------------------------------
# T1 — cache-miss path writes the cache
# ---------------------------------------------------------------------------


def test_cache_miss_path_writes_perm_to_cache():
    """First call into ``_tentative_merged_weights`` with a fresh cache must
    leave the freshly-computed permutation under ``(li, centroid_id, child_id)``.
    """
    layer_ref = _make_layer_ref()
    cache = _PermAlignCache()
    li = layer_ref.layer_idx
    centroid_id, child_id = 0, 1

    # Sanity: nothing in the cache before the call.
    assert not cache.has((li, centroid_id, child_id))

    _ = _tentative_merged_weights(
        layer_ref,
        centroid_id=centroid_id,
        child_id=child_id,
        freq={0: 2, 1: 2},
        ream_acc=None,
        perm_cache=cache,
    )

    # The write must have happened.
    assert cache.has((li, centroid_id, child_id)), (
        "cache-miss path must persist the Hungarian permutation for downstream reuse"
    )

    stored = cache.get((li, centroid_id, child_id))
    assert stored is not None
    perm_stored, residual_stored = stored

    # Shape: 1-D integer array of length intermediate_size (d_int == 3).
    d_int = 3
    assert isinstance(perm_stored, np.ndarray)
    assert perm_stored.shape == (d_int,)
    assert np.issubdtype(perm_stored.dtype, np.integer)
    # Residual is None for the output path (no whitened Frobenius computed).
    assert residual_stored is None

    # Recompute the expected perm independently using the same inputs to
    # directly assert that the stored perm equals the perm that
    # ``_permutation_align_to_centroid`` produces for these inputs.
    from moe_compress.stage2.permutation_align import _permutation_align_to_centroid
    banks = build_banks(layer_ref)
    ref_gate_fp32   = banks["gate_proj"].get(centroid_id).to(torch.float32)
    ref_up_fp32     = banks["up_proj"].get(centroid_id).to(torch.float32)
    child_gate_fp32 = banks["gate_proj"].get(child_id).to(torch.float32)
    child_up_fp32   = banks["up_proj"].get(child_id).to(torch.float32)
    expected_perm = _permutation_align_to_centroid(
        ref_gate_fp32, ref_up_fp32, child_gate_fp32, child_up_fp32,
        ref_act_mean=None, child_act_mean=None,
    )
    assert np.array_equal(perm_stored, expected_perm), (
        f"stored perm {perm_stored} != expected {expected_perm}"
    )


# ---------------------------------------------------------------------------
# T2 — cached perm matches the perm used in the merge
# ---------------------------------------------------------------------------


def test_cached_perm_yields_identical_merge_on_second_call():
    """The merge built from the cached perm (cache-hit) must equal the merge
    built when the perm was first computed (cache-miss). Proves the stored
    perm is exactly the one the cost matrix used."""
    layer_ref = _make_layer_ref()
    cache = _PermAlignCache()
    li = layer_ref.layer_idx
    centroid_id, child_id = 0, 1
    freq = {0: 2, 1: 2}

    # First call: cache miss, perm computed and stored.
    merged_first = _tentative_merged_weights(
        layer_ref,
        centroid_id=centroid_id,
        child_id=child_id,
        freq=freq,
        ream_acc=None,
        perm_cache=cache,
    )

    # Confirm the write happened so the second call genuinely takes the
    # cache-hit branch.
    assert cache.has((li, centroid_id, child_id))

    # Second call: cache hit, perm pulled from cache.
    merged_second = _tentative_merged_weights(
        layer_ref,
        centroid_id=centroid_id,
        child_id=child_id,
        freq=freq,
        ream_acc=None,
        perm_cache=cache,
    )

    for name in MATRIX_NAMES:
        assert torch.allclose(merged_first[name], merged_second[name], atol=1e-6), (
            f"{name}: cache-hit merge must match the original cache-miss merge"
        )


# ---------------------------------------------------------------------------
# T3 — cache-hit path does NOT overwrite (idempotency)
# ---------------------------------------------------------------------------


def test_cache_hit_path_does_not_overwrite_existing_entry():
    """When the entry already exists, ``_tentative_merged_weights`` must take
    the cache-hit branch and NOT call ``put`` again. We prove this by stamping
    a sentinel residual onto the existing entry and asserting it survives a
    follow-up call."""
    layer_ref = _make_layer_ref()
    cache = _PermAlignCache()
    li = layer_ref.layer_idx
    centroid_id, child_id = 0, 1
    freq = {0: 2, 1: 2}

    # First call populates the cache with residual=None.
    _ = _tentative_merged_weights(
        layer_ref,
        centroid_id=centroid_id,
        child_id=child_id,
        freq=freq,
        ream_acc=None,
        perm_cache=cache,
    )
    perm_stored, _ = cache.get((li, centroid_id, child_id))

    # Stamp a sentinel residual so we can detect an unwanted overwrite.
    sentinel_residual = 42.0
    cache.put((li, centroid_id, child_id), perm_stored, residual=sentinel_residual)

    # Second call should hit the cache and leave the sentinel untouched.
    _ = _tentative_merged_weights(
        layer_ref,
        centroid_id=centroid_id,
        child_id=child_id,
        freq=freq,
        ream_acc=None,
        perm_cache=cache,
    )

    perm_after, residual_after = cache.get((li, centroid_id, child_id))
    assert residual_after == sentinel_residual, (
        "cache-hit path must not overwrite the existing entry"
    )
    # Perm must also be byte-identical (no silent re-write of perm alone).
    np.testing.assert_array_equal(perm_after, perm_stored)


# ---------------------------------------------------------------------------
# T4 — perm_cache=None path unchanged
# ---------------------------------------------------------------------------


def test_none_perm_cache_path_unchanged():
    """With ``perm_cache=None`` the write must be silently skipped. The call
    must succeed and produce finite merged weights of the correct shapes —
    and, crucially, must be byte-identical to a ``perm_cache``-enabled call
    on the same inputs (no semantic drift between the two code paths)."""
    layer_ref = _make_layer_ref()
    centroid_id, child_id = 0, 1
    freq = {0: 2, 1: 2}

    merged = _tentative_merged_weights(
        layer_ref,
        centroid_id=centroid_id,
        child_id=child_id,
        freq=freq,
        ream_acc=None,
        perm_cache=None,
    )

    banks = build_banks(layer_ref)
    for name in MATRIX_NAMES:
        # Shape must match the per-expert weight shape.
        expected_shape = banks[name].get(centroid_id).shape
        assert merged[name].shape == expected_shape, (
            f"{name}: merged shape {tuple(merged[name].shape)} != "
            f"per-expert shape {tuple(expected_shape)}"
        )
        assert torch.isfinite(merged[name]).all(), (
            f"{name}: merged weights contain non-finite values"
        )

    # L2 cross-check: perm_cache=None must produce byte-identical merged
    # weights to a perm_cache-enabled call on the same inputs.
    cache = _PermAlignCache()
    merged_with_cache = _tentative_merged_weights(
        layer_ref,
        centroid_id=centroid_id,
        child_id=child_id,
        freq=freq,
        ream_acc=None,
        perm_cache=cache,
    )
    merged_no_cache = _tentative_merged_weights(
        layer_ref,
        centroid_id=centroid_id,
        child_id=child_id,
        freq=freq,
        ream_acc=None,
        perm_cache=None,
    )
    for name in MATRIX_NAMES:
        assert torch.equal(merged_no_cache[name], merged_with_cache[name]), (
            f"semantic drift on {name}: perm_cache=None deviates from "
            f"perm_cache=enabled"
        )
