"""Tests for Phase F post-hoc causal-ablation validation."""
from moe_compress.stage1_post_hoc_ablation import rank_top_nonblacklisted


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
