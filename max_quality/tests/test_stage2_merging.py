"""Tests for moe_compress.stage2.permutation_align and stage2.merging
(Task 4 of the plugin refactor).

Scope:
  _PermAlignCache round-trip: get / put / has / clear / __len__ on the new
  module path. Behavioural coverage of the merge engine itself stays in
  test_stage2_merge.py — duplicating it here would be redundant.
"""
from __future__ import annotations

import numpy as np

from moe_compress.stage2.merging import (
    _merge_experts_inplace,
    _resize_router_for_kept_experts,
)
from moe_compress.stage2.permutation_align import (
    _PermAlignCache,
    _aligned_whitened_residual,
    _permutation_align_to_centroid,
)


# ---------------------------------------------------------------------------
# _PermAlignCache - get/put/has/clear/__len__ round-trip
# ---------------------------------------------------------------------------


def test_perm_align_cache_put_get_has_returns_stored_values():
    cache = _PermAlignCache()
    perm = np.array([2, 0, 1, 3], dtype=np.int64)
    cache.put((0, 5, 7), perm, 0.42)

    got = cache.get((0, 5, 7))
    assert got is not None
    stored_perm, stored_residual = got
    assert np.array_equal(stored_perm, perm)
    assert stored_residual == 0.42
    assert cache.has((0, 5, 7))
    assert len(cache) == 1


def test_perm_align_cache_residual_may_be_none_legacy_v1_path():
    """The v1 merge path stores entries with residual=None (only perm known).
    The cache must accept and round-trip that value without coercion."""
    cache = _PermAlignCache()
    perm = np.array([0, 1, 2], dtype=np.int64)
    cache.put((1, 0, 1), perm, None)

    got = cache.get((1, 0, 1))
    assert got is not None
    assert got[1] is None


def test_perm_align_cache_clear_empties_store():
    cache = _PermAlignCache()
    cache.put((0, 0, 1), np.array([0, 1], dtype=np.int64), 1.0)
    cache.put((0, 0, 2), np.array([1, 0], dtype=np.int64), 2.0)
    assert len(cache) == 2

    cache.clear()
    assert len(cache) == 0
    assert cache.get((0, 0, 1)) is None
    assert not cache.has((0, 0, 1))
