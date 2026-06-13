"""Task 1 — group-aligned + even split helpers for Stage-6 DP eval-shard.

The gen path is METRIC_PINNED (bf16 reduction order + left-pad geometry flip
near-tied argmax). A completion is a pure function of its group-of-8 + within-
group order, so the shard boundary MUST be a multiple of ``PINNED_GEN_BATCH_SIZE``
(=8) or ``n`` — never split a group. The PPL path is BATCH_INVARIANT, so its
split has no group constraint.

These are pure-arithmetic helpers (no torch); the tests pin the exact code
output the plan specifies.
"""
from __future__ import annotations

import pytest

from moe_compress.tools.eval_shard import (
    PINNED_GEN_BATCH_SIZE,
    _even_split,
    _group_aligned_split,
)


def test_pinned_gen_batch_size_is_eight():
    """Single source of truth: re-exported from tools.eval_harness."""
    from moe_compress.tools.eval_harness import (
        PINNED_GEN_BATCH_SIZE as HARNESS_PIN,
    )

    assert PINNED_GEN_BATCH_SIZE == 8
    assert PINNED_GEN_BATCH_SIZE == HARNESS_PIN


def test_group_aligned_split_humaneval_164_2():
    """164 prompts → 21 groups (last group of 4); base = 21//2 = 10.
    GPU0 owns groups 0..9 = [0:80], GPU1 owns groups 10..20 = [80:164]
    (incl. the short final group [160:164]). Pinned exactly per the plan.
    """
    assert _group_aligned_split(164, 2) == [(0, 80), (80, 164)]


def test_group_aligned_boundaries_are_multiples_of_8_or_n():
    for n in (8, 16, 20, 164, 500, 37):
        for replicas in (2, 3, 4):
            bounds = _group_aligned_split(n, replicas)
            for start, end in bounds:
                assert start % PINNED_GEN_BATCH_SIZE == 0 or start == n
                assert end % PINNED_GEN_BATCH_SIZE == 0 or end == n


def test_group_aligned_reconstructs_range_no_gap_overlap():
    for n in (8, 16, 20, 164, 500, 37, 1):
        for replicas in (1, 2, 3, 4, 5):
            bounds = _group_aligned_split(n, replicas)
            # Contiguous, in order, covering [0, n).
            assert bounds[0][0] == 0
            assert bounds[-1][1] == n
            for (s0, e0), (s1, e1) in zip(bounds, bounds[1:]):
                assert e0 == s1, f"gap/overlap n={n} r={replicas}: {bounds}"
                assert e0 > s0


def test_group_aligned_clamps_replicas_to_n_groups():
    # 8 prompts = 1 group → at most 1 shard regardless of replicas requested.
    assert _group_aligned_split(8, 5) == [(0, 8)]
    # 20 prompts = 3 groups (8,8,4) → at most 3 shards.
    assert len(_group_aligned_split(20, 10)) == 3


def test_group_aligned_single_replica_is_whole():
    assert _group_aligned_split(164, 1) == [(0, 164)]
    assert _group_aligned_split(0, 4) == [(0, 0)]


def test_even_split_500_2():
    assert _even_split(500, 2) == [(0, 250), (250, 500)]


def test_even_split_remainder_to_last_shard():
    # Mirror _shard_calib: last shard absorbs the remainder.
    assert _even_split(501, 2) == [(0, 250), (250, 501)]
    assert _even_split(10, 3) == [(0, 3), (3, 6), (6, 10)]


def test_even_split_single_and_clamp():
    assert _even_split(500, 1) == [(0, 500)]
    assert _even_split(0, 4) == [(0, 0)]
    # More replicas than rows → clamp to n.
    assert _even_split(2, 5) == [(0, 1), (1, 2)]
