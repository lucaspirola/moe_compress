"""Direction D — correctness tests for the Stage 2 2-opt refinement pass.

``_two_opt_refine`` is a strictly-improving local search over an already
feasible greedy child→centroid assignment. These tests pin the four
invariants the spec demands, all on CPU-only synthetic cost matrices (no
model, no GPU):

  (a) 2-opt total cost <= greedy total cost, always.
  (b) per-centroid capacity caps respected after refinement.
  (c) no-op when the input is already locally optimal.
  (d) it terminates (no infinite pass loop).

The pass is model-agnostic: it only sees an assignment list, a cost matrix
and a capacity cap, so the synthetic instances below are a faithful test of
the production code path.
"""
from __future__ import annotations

import numpy as np
import pytest

from moe_compress.stage2.orchestrator import (
    _assign_greedy,
    _two_opt_refine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _total_cost(assignment: list[int], cost: np.ndarray) -> float:
    """Sum of cost[child, assigned_centroid] over assigned children."""
    return float(
        sum(cost[ch, g] for ch, g in enumerate(assignment) if g >= 0)
    )


def _group_sizes(assignment: list[int], n_centroids: int) -> list[int]:
    sizes = [0] * n_centroids
    for g in assignment:
        if g >= 0:
            sizes[g] += 1
    return sizes


# ---------------------------------------------------------------------------
# (a) 2-opt total cost <= greedy total cost — random fuzz
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(40))
def test_two_opt_never_worse_than_greedy(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_children = int(rng.integers(2, 14))
    n_centroids = int(rng.integers(2, 7))
    cap = int(rng.integers(1, n_children + 2))
    # Ensure feasibility: total capacity must cover all children.
    if n_centroids * cap < n_children:
        cap = -(-n_children // n_centroids)  # ceil div

    cost = rng.random((n_children, n_centroids)).astype(np.float64)

    greedy = _assign_greedy(cost, n_children, n_centroids, cap)
    refined = _two_opt_refine(greedy, cost, cap)

    assert _total_cost(refined, cost) <= _total_cost(greedy, cost) + 1e-9, (
        f"seed={seed}: 2-opt regressed vs greedy "
        f"({_total_cost(refined, cost)} > {_total_cost(greedy, cost)})"
    )
    # Refinement must not strand a previously-assigned child.
    for ch in range(n_children):
        if greedy[ch] >= 0:
            assert refined[ch] >= 0, f"seed={seed}: child {ch} lost its assignment"


def test_two_opt_strictly_improves_known_swap_case() -> None:
    """A hand-built instance where greedy is sub-optimal and a swap fixes it."""
    # 2 children, 2 centroids, cap 1 each → exactly one child per centroid.
    # Greedy iterates centroid 0 first and grabs its cheapest child (child 0),
    # leaving child 1 → centroid 1. Optimal is the swap.
    #            centroid0  centroid1
    cost = np.array([
        [0.10,      0.00],   # child 0: much prefers centroid 1
        [0.20,      0.90],   # child 1: prefers centroid 0
    ])
    cap = 1
    greedy = _assign_greedy(cost, 2, 2, cap)
    refined = _two_opt_refine(greedy, cost, cap)
    assert _total_cost(refined, cost) < _total_cost(greedy, cost)
    # Optimal assignment: child0->1, child1->0.
    assert refined == [1, 0]


# ---------------------------------------------------------------------------
# (b) capacity caps respected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(40))
def test_two_opt_respects_capacity(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n_children = int(rng.integers(2, 16))
    n_centroids = int(rng.integers(2, 7))
    cap = -(-n_children // n_centroids)  # tight ceil-div cap
    cost = rng.random((n_children, n_centroids)).astype(np.float64)

    greedy = _assign_greedy(cost, n_children, n_centroids, cap)
    refined = _two_opt_refine(greedy, cost, cap)

    for g, size in enumerate(_group_sizes(refined, n_centroids)):
        assert size <= cap, (
            f"seed={seed}: centroid {g} has {size} children, cap={cap}"
        )


def test_two_opt_uncapped_path() -> None:
    """max_group_cap <= 0 is the uncapped ablation path — groups unbounded."""
    rng = np.random.default_rng(7)
    n_children, n_centroids = 12, 3
    cost = rng.random((n_children, n_centroids)).astype(np.float64)
    greedy = _assign_greedy(cost, n_children, n_centroids, 0)
    refined = _two_opt_refine(greedy, cost, 0)
    # Uncapped greedy already assigns each child to its global argmin; 2-opt
    # has nothing to improve and must leave it byte-identical.
    assert refined == greedy
    assert _total_cost(refined, cost) <= _total_cost(greedy, cost) + 1e-9


# ---------------------------------------------------------------------------
# (c) no-op when input is already locally optimal
# ---------------------------------------------------------------------------
def test_two_opt_noop_on_locally_optimal_input() -> None:
    """Each child strictly prefers its current centroid → no move improves."""
    #            centroid0  centroid1  centroid2
    cost = np.array([
        [0.01,      0.90,      0.95],   # child 0 -> 0
        [0.92,      0.02,      0.97],   # child 1 -> 1
        [0.93,      0.94,      0.03],   # child 2 -> 2
    ])
    assignment = [0, 1, 2]
    refined = _two_opt_refine(assignment, cost, max_group_cap=2)
    assert refined == assignment
    assert refined is not assignment  # returns a copy, never mutates input


def test_two_opt_does_not_mutate_input() -> None:
    rng = np.random.default_rng(99)
    cost = rng.random((8, 3)).astype(np.float64)
    greedy = _assign_greedy(cost, 8, 3, 4)
    original = list(greedy)
    _ = _two_opt_refine(greedy, cost, 4)
    assert greedy == original, "_two_opt_refine mutated its input list"


# ---------------------------------------------------------------------------
# (d) termination — idempotence proves the loop fixed-points
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(25))
def test_two_opt_terminates_and_is_idempotent(seed: int) -> None:
    rng = np.random.default_rng(5000 + seed)
    n_children = int(rng.integers(2, 18))
    n_centroids = int(rng.integers(2, 8))
    cap = -(-n_children // n_centroids)
    cost = rng.random((n_children, n_centroids)).astype(np.float64)

    greedy = _assign_greedy(cost, n_children, n_centroids, cap)
    # First call terminates (pytest would hang otherwise).
    refined = _two_opt_refine(greedy, cost, cap)
    # A second pass on the output must change nothing — the result is a
    # fixed point, which is what guarantees the while-loop halts.
    refined2 = _two_opt_refine(refined, cost, cap)
    assert refined2 == refined, f"seed={seed}: 2-opt output is not a fixed point"


# ---------------------------------------------------------------------------
# Edge cases — empty / degenerate inputs
# ---------------------------------------------------------------------------
def test_two_opt_empty_inputs() -> None:
    assert _two_opt_refine([], np.zeros((0, 0)), 4) == []
    assert _two_opt_refine([], np.zeros((0, 3)), 4) == []
    # Single child, single centroid — nothing to swap or move.
    cost = np.array([[0.5]])
    assert _two_opt_refine([0], cost, 1) == [0]


def test_two_opt_preserves_unassigned_children() -> None:
    """A -1 (unassigned, all-inf row) child stays unassigned; 2-opt never
    promotes it into a group."""
    cost = np.array([
        [0.10, 0.20],
        [np.inf, np.inf],   # child 1: no feasible centroid
        [0.30, 0.15],
    ])
    assignment = [0, -1, 1]
    refined = _two_opt_refine(assignment, cost, max_group_cap=3)
    assert refined[1] == -1
    assert _total_cost(refined, cost) <= _total_cost(assignment, cost) + 1e-9
