"""Tests for Phase D ablation filter and the legacy Phase F helper."""
from moe_compress.stage1_post_hoc_ablation import (
    _apply_threshold_filter,
    rank_top_nonblacklisted,
)


def test_rank_top_nonblacklisted_excludes_blacklisted():
    per_expert_max = {
        (5, 0): 1.0, (5, 1): 9.0, (5, 2): 5.0, (5, 3): 8.0,
        (5, 4): 7.0, (5, 5): 2.0,
    }
    blacklist = {5: [1]}  # expert 1 is blacklisted (would otherwise top the list)
    L = {5}
    out = rank_top_nonblacklisted(per_expert_max, blacklist, L, top_k=3)
    # Expected: experts 3, 4, 2 (descending by per_expert_max, excluding 1)
    assert out == {5: [3, 4, 2]}


def test_rank_top_nonblacklisted_only_in_L():
    per_expert_max = {(5, 0): 100.0, (9, 0): 200.0}
    L = {5}
    out = rank_top_nonblacklisted(per_expert_max, blacklist={}, L=L, top_k=2)
    assert out == {5: [0]}  # only layer 5 considered (L = {5})


def test_run_ablation_filter_threshold_logic():
    """Phase D filter cuts ΔNLL ≤ threshold; passes ΔNLL > threshold; sorts per-layer."""
    deltas = {
        (10, 5): 0.005,    # > threshold → kept
        (10, 2): 0.002,    # > threshold → kept (sorted before 5)
        (10, 9): 0.0005,   # ≤ threshold → dropped
        (10, 1): -0.01,    # negative ΔNLL (false positive) → dropped
        (34, 7): 0.001,    # exactly at threshold → dropped (strict >)
        (34, 4): 0.1,      # well above → kept
    }
    threshold = 0.001
    bl = _apply_threshold_filter(deltas, threshold)
    assert bl == {10: [2, 5], 34: [4]}


def test_run_ablation_filter_threshold_logic_empty():
    """Empty deltas → empty blacklist."""
    assert _apply_threshold_filter({}, threshold=0.001) == {}


def test_run_ablation_filter_threshold_logic_all_below():
    """All candidates below threshold → empty blacklist (matches v4-style false positives)."""
    deltas = {(0, 0): -0.001, (0, 1): 0.0, (0, 2): 0.0009}
    assert _apply_threshold_filter(deltas, threshold=0.001) == {}
