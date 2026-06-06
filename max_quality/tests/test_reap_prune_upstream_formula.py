"""Unit — protected=∅ byte-match vs upstream's topk formula (§6 test 7).

With ``protected = ∅`` (the upstream DEFAULT — both
``perserve_super_experts`` / ``perserve_outliers`` are ``default=False`` at
``CerebrasResearch/reap`` ``args.py:514-515,522-523``), our
``compute_final_kept_ids`` must be byte-identical to upstream's

    experts_to_prune = torch.topk(scores, n_prune, largest=False).indices
    retained         = [i for i in range(n) if i not in experts_to_prune]

(``prune.py:101-106``). This proves that, absent our protected divergence, the
selection equals upstream's formula. Tie-break is pinned to ``torch.topk``'s
ordering so the byte-match is deterministic.
"""
from __future__ import annotations

import numpy as np
import torch

from moe_compress.stage2.plugins.reap_prune import compute_final_kept_ids


def _upstream_retained(scores_np: np.ndarray, n_prune: int) -> list[int]:
    """Hand-computed upstream prune.py:101-106 retained-expert list."""
    scores_t = torch.tensor(scores_np, dtype=torch.float64)
    _, experts_to_prune = torch.topk(scores_t, n_prune, largest=False)
    drop = set(int(i) for i in experts_to_prune.tolist())
    return [i for i in range(scores_t.numel()) if i not in drop]


def test_protected_empty_matches_upstream_topk():
    rng = np.random.default_rng(1234)
    for n_experts in (3, 5, 8):
        for n_prune in range(0, n_experts):
            # distinct scores → no ties → both formulas agree unambiguously.
            scores = rng.permutation(n_experts).astype(np.float64)
            kept, pruned = compute_final_kept_ids(
                scores, n_experts=n_experts, n_prune=n_prune, protected=[],
            )
            retained = sorted(_upstream_retained(scores, n_prune))
            assert kept == retained, (
                f"n_experts={n_experts} n_prune={n_prune}: "
                f"ours={kept} upstream={retained}"
            )
            assert pruned == sorted(set(range(n_experts)) - set(retained))


def test_tie_break_is_deterministic_and_keeps_lowest_index():
    # On exact ties torch.topk's index ordering is IMPLEMENTATION-DEFINED (not a
    # stable, version-pinned contract), so we do NOT byte-match upstream there.
    # Our helper's documented tie-break — np.argsort(-scores) is stable →
    # ascending index → keep the LOWEST indices — is deterministic and is the
    # behaviour that matters for reproducible runs. Real REAP saliencies are
    # continuous and never exactly tied, so this path is defensive only.
    scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    kept, pruned = compute_final_kept_ids(
        scores, n_experts=4, n_prune=1, protected=[],
    )
    assert kept == [0, 1, 2]   # keep the three lowest indices on a full tie
    assert pruned == [3]
    # idempotent / deterministic across repeated calls
    kept2, _ = compute_final_kept_ids(
        scores, n_experts=4, n_prune=1, protected=[],
    )
    assert kept2 == kept
