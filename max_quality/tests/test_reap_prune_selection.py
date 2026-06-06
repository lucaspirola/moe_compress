"""Unit — faithful REAP pruner top-K selection (§6 test 1).

Covers the pure ``compute_final_kept_ids`` helper: drop the bottom-saliency
``n_prune`` experts (== keep the top ``n_experts − n_prune``), never drop
protected, and emit the dropped complement as ``pruned_expert_ids``. Mirrors
upstream ``torch.topk(saliency, n_prune, largest=False)`` (CerebrasResearch/reap
``prune.py:101``).
"""
from __future__ import annotations

import numpy as np
import pytest

from moe_compress.stage2.plugins.reap_prune import compute_final_kept_ids


def test_keeps_top_k_drops_bottom():
    # scores: expert 1 highest, expert 0 lowest.
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    kept, pruned = compute_final_kept_ids(
        scores, n_experts=4, n_prune=2, protected=[],
    )
    # keep top 2 (experts 1, 3); drop bottom 2 (experts 0, 2).
    assert kept == [1, 3]
    assert pruned == [0, 2]


def test_protected_never_dropped():
    # expert 0 has the lowest score but is protected → must be kept; the next
    # lowest non-protected (expert 2) is dropped instead to hit kept-count==2.
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    kept, pruned = compute_final_kept_ids(
        scores, n_experts=4, n_prune=2, protected=[0],
    )
    # faithful_target = 2; protected {0} kept; one non-protected kept (top: 1).
    assert kept == [0, 1]
    assert pruned == [2, 3]
    assert 0 not in pruned


def test_kept_count_equals_target():
    scores = np.array([0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4])
    kept, pruned = compute_final_kept_ids(
        scores, n_experts=8, n_prune=3, protected=[],
    )
    assert len(kept) == 5            # 8 − 3
    assert len(pruned) == 3
    assert set(kept) | set(pruned) == set(range(8))
    assert set(kept) & set(pruned) == set()


def test_n_prune_zero_keeps_all():
    scores = np.array([0.1, 0.9, 0.3])
    kept, pruned = compute_final_kept_ids(
        scores, n_experts=3, n_prune=0, protected=[],
    )
    assert kept == [0, 1, 2]
    assert pruned == []


def test_target_below_protected_raises():
    # n_prune=3 → faithful_target=1, but 2 protected → infeasible.
    scores = np.array([0.1, 0.9, 0.3, 0.7])
    with pytest.raises(RuntimeError, match="smaller than the protected"):
        compute_final_kept_ids(scores, n_experts=4, n_prune=3, protected=[0, 1])


def test_tie_break_prefers_lower_index():
    # experts 0 and 1 tie at the boundary; our deterministic tie-break keeps the
    # lower index (drops the higher). This is NOT a torch.topk(largest=False)
    # parity claim — torch's tie order is implementation-defined and differs
    # CPU/CUDA; here we only assert our own stable keep-lowest-index behaviour.
    scores = np.array([0.5, 0.5, 0.9])
    kept, pruned = compute_final_kept_ids(
        scores, n_experts=3, n_prune=1, protected=[],
    )
    assert kept == [0, 2]
    assert pruned == [1]
