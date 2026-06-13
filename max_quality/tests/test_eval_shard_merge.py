"""Task 2 — gen merge by index (order-preserving concat).

``_merge_completions`` concatenates contiguous, group-aligned shard results in
``start`` order, recovering the original index order. A gap/overlap is a logic
error and must raise (defensive).
"""
from __future__ import annotations

import pytest

from moe_compress.tools.eval_shard import _merge_completions


def test_merge_three_shards_original_order():
    # shards cover [0,8), [8,16), [16,20)
    s0 = (0, 8, [f"c{i}" for i in range(0, 8)])
    s1 = (8, 16, [f"c{i}" for i in range(8, 16)])
    s2 = (16, 20, [f"c{i}" for i in range(16, 20)])
    # Pass them out of order to prove sorting by `start`.
    merged = _merge_completions([s2, s0, s1])
    assert merged == [f"c{i}" for i in range(20)]


def test_merge_single_shard():
    s = (0, 5, ["a", "b", "c", "d", "e"])
    assert _merge_completions([s]) == ["a", "b", "c", "d", "e"]


def test_merge_gap_raises():
    s0 = (0, 8, [f"c{i}" for i in range(8)])
    s1 = (10, 14, [f"c{i}" for i in range(4)])  # gap [8,10)
    with pytest.raises(ValueError, match="gap|overlap|contiguous"):
        _merge_completions([s0, s1])


def test_merge_overlap_raises():
    s0 = (0, 8, [f"c{i}" for i in range(8)])
    s1 = (6, 12, [f"c{i}" for i in range(6)])  # overlaps [6,8)
    with pytest.raises(ValueError, match="gap|overlap|contiguous"):
        _merge_completions([s0, s1])


def test_merge_length_mismatch_raises():
    # completions list length must equal end-start.
    s0 = (0, 8, [f"c{i}" for i in range(7)])  # 7 != 8
    with pytest.raises(ValueError, match="length|count|expected"):
        _merge_completions([s0])


def test_merge_does_not_start_at_zero_raises():
    s0 = (2, 8, [f"c{i}" for i in range(6)])
    with pytest.raises(ValueError, match="start|0|contiguous"):
        _merge_completions([s0])
