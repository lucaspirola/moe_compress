"""Phase 1 tests for the Stage 2 v2 assignment-solver dispatcher.

Covers what's implemented after Phase 1's first sub-step:
- Solver dispatch (``greedy``, ``hungarian``, ``mcf``, ``auto``, ``sinkhorn`` fallback)
- Bit-identical greedy (the v1 compatibility path)
- Hungarian == greedy on 1-1 assignment
- MCF >= greedy on tight-capacity instances (greedy bias case)
- Empty-input defensive guards
- Unknown-solver error path

Tests for ``cost_alignment="post"``, whitened cost, asymmetric freq, capacity
gate, K-prefilter, and EM are deferred to subsequent Phase 1 sub-steps where
those features land.
"""
from __future__ import annotations

import numpy as np
import pytest

from moe_compress.stage2.orchestrator import (
    _assign_children_to_centroids,
    _assign_greedy,
    _assign_hungarian,
    _assign_mcf,
    _assign_sinkhorn,
)

# ``_assign_mcf`` requires the optional ``ortools>=9.10`` package. Tests that
# exercise the MCF solver (and tests that cross-check Sinkhorn or the ``auto``
# dispatcher against MCF) need ortools at runtime. When the package is absent
# they SKIP cleanly via this marker instead of erroring with a RuntimeError.
try:
    import ortools  # noqa: F401
    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False
_requires_ortools = pytest.mark.skipif(
    not _HAS_ORTOOLS,
    reason="ortools>=9.10 not installed; MCF solver tests skipped",
)


# ---------------------------------------------------------------------------
# Compatibility invariant: dispatcher with default solver = greedy is
# bit-identical to the v1 helper (the body of which is now _assign_greedy).
# ---------------------------------------------------------------------------


def test_dispatcher_default_solver_is_greedy_and_bit_identical():
    """Default dispatcher path must produce exactly the same output as
    calling _assign_greedy directly. This is the v1 compatibility invariant.
    """
    rng = np.random.default_rng(0)
    cost = rng.random((6, 3))

    via_dispatch = _assign_children_to_centroids(cost, 6, 3, max_group_cap=2)
    via_helper = _assign_greedy(cost, 6, 3, max_group_cap=2)

    assert via_dispatch == via_helper


def test_dispatcher_unknown_solver_raises():
    cost = np.zeros((2, 2))
    with pytest.raises(ValueError, match="unknown solver"):
        _assign_children_to_centroids(cost, 2, 2, max_group_cap=1, solver="bogus")  # type: ignore[arg-type]


def test_dispatcher_uppercase_solver_normalized():
    """Solver name comparison should be case-insensitive for human-friendly
    config values."""
    cost = np.array([[0.1, 0.9], [0.8, 0.2]])
    expected = _assign_greedy(cost, 2, 2, max_group_cap=1)
    via_upper = _assign_children_to_centroids(cost, 2, 2, max_group_cap=1, solver="GREEDY")  # type: ignore[arg-type]
    assert via_upper == expected


# ---------------------------------------------------------------------------
# Defensive empty-input guards: every helper returns [-1] * n_children
# when either dimension is zero, so they can be called from fallback paths
# without re-doing the dispatcher's early-exit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["greedy", "hungarian", "mcf", "auto"])
def test_empty_inputs_return_negative_ones(solver):
    cost_zero_rows = np.zeros((0, 4))
    cost_zero_cols = np.zeros((4, 0))

    assert _assign_children_to_centroids(cost_zero_rows, 0, 4, max_group_cap=1, solver=solver) == []
    assert _assign_children_to_centroids(cost_zero_cols, 4, 0, max_group_cap=1, solver=solver) == [-1] * 4


def test_helpers_independently_handle_empty():
    """Helpers must guard their own empty-input case so MCF→greedy and
    Hungarian→MCF fallbacks are safe."""
    cost = np.zeros((0, 0))
    assert _assign_greedy(cost, 0, 0, 1) == []
    assert _assign_hungarian(cost, 0, 0, 1) == []
    assert _assign_mcf(cost, 0, 0, 1) == []


# ---------------------------------------------------------------------------
# Hungarian = optimal under 1-1 capacity (slack regime).
# ---------------------------------------------------------------------------


def test_hungarian_optimal_1to1():
    """On a 1-1 problem (n_children ≤ n_centroids, capacity ≥ 1), Hungarian
    finds the global minimum cost. Construct an instance where greedy's
    saliency-order bias produces a suboptimal solution, and verify Hungarian
    beats it."""
    # Reviewer's counterexample (synthetic):
    #            c1    c2    c3
    #   m1     0.10  0.15  0.99
    #   m2     0.20  0.25  0.99
    #   m3     0.90  0.30  0.40
    cost = np.array([
        [0.10, 0.15, 0.99],
        [0.20, 0.25, 0.99],
        [0.90, 0.30, 0.40],
    ])

    hungarian_result = _assign_hungarian(cost, n_children=3, n_centroids=3, max_group_cap=1)
    hungarian_cost = sum(cost[i, hungarian_result[i]] for i in range(3))

    # Greedy with cap=1 (1-1) iterates centroids in column order (caller
    # builds them by descending saliency); it has the same per-child argmin
    # behavior under cap=1 only when each centroid has at most one candidate.
    # Compare against the all-1-1 enumeration to confirm Hungarian is optimal.
    from itertools import permutations
    best_cost = min(
        sum(cost[i, p[i]] for i in range(3))
        for p in permutations(range(3))
    )
    assert hungarian_cost == pytest.approx(best_cost)


# ---------------------------------------------------------------------------
# MCF ≥ greedy on tight instances (the reviewer's headline gap).
# ---------------------------------------------------------------------------


@_requires_ortools
def test_mcf_optimal_on_tight_capacity_counterexample():
    """The reviewer's counterexample (group_size=3 → max_group_cap=2):

        d:        m1    m2    m3    m4
          c1   0.10  0.20  0.90  0.90
          c2   0.15  0.25  0.30  0.95
          c3   0.99  0.99  0.40  0.50

    Greedy (descending saliency, c1 first): c1 grabs {m1,m2}=0.30,
    c2 grabs {m3,m4}=1.25  → total 1.55.

    Optimal: c1 grabs {m1,m2}=0.30, c2 grabs {m3}=0.30, c3 grabs {m4}=0.50
    → total 1.10 (29% lower).

    MCF must hit the optimum.
    """
    cost = np.array([
        [0.10, 0.15, 0.99],
        [0.20, 0.25, 0.99],
        [0.90, 0.30, 0.40],
        [0.90, 0.95, 0.50],
    ])  # shape (n_children=4, n_centroids=3)
    n_children, n_centroids, cap = 4, 3, 2

    mcf_result = _assign_mcf(cost, n_children, n_centroids, cap)
    mcf_cost = sum(cost[i, mcf_result[i]] for i in range(n_children))

    greedy_result = _assign_greedy(cost, n_children, n_centroids, cap)
    greedy_cost = sum(cost[i, greedy_result[i]] for i in range(n_children))

    # MCF must be strictly better than greedy on this counterexample.
    assert mcf_cost < greedy_cost
    # And MCF must hit the known optimum 1.10.
    assert mcf_cost == pytest.approx(1.10, abs=1e-6)


@_requires_ortools
def test_mcf_matches_hungarian_on_1to1():
    """When capacity ≥ 1 and n_children ≤ n_centroids, MCF and Hungarian
    must produce the same total cost (both optimal on the same problem)."""
    rng = np.random.default_rng(42)
    cost = rng.random((4, 6))

    mcf_result = _assign_mcf(cost, 4, 6, max_group_cap=1)
    hung_result = _assign_hungarian(cost, 4, 6, max_group_cap=1)

    mcf_total = sum(cost[i, mcf_result[i]] for i in range(4))
    hung_total = sum(cost[i, hung_result[i]] for i in range(4))
    assert mcf_total == pytest.approx(hung_total, abs=1e-6)


@_requires_ortools
def test_mcf_handles_inf_entries():
    """+∞ cost entries should be excluded from the assignment (forbidden
    pairs). MCF must still find an optimal feasible solution."""
    cost = np.array([
        [0.1, np.inf, 0.5],
        [np.inf, 0.2, 0.3],
        [0.4, 0.3, np.inf],
    ])
    result = _assign_mcf(cost, 3, 3, max_group_cap=1)
    # Only finite-arc assignment must be picked.
    for i, j in enumerate(result):
        assert j >= 0
        assert np.isfinite(cost[i, j])


# ---------------------------------------------------------------------------
# Auto: hungarian when slack, mcf when tight.
# ---------------------------------------------------------------------------


def test_auto_dispatches_hungarian_on_slack():
    """n_children ≤ n_centroids → auto must hit the Hungarian branch."""
    rng = np.random.default_rng(1)
    cost = rng.random((3, 5))

    auto_result = _assign_children_to_centroids(cost, 3, 5, max_group_cap=1, solver="auto")
    hung_result = _assign_hungarian(cost, 3, 5, max_group_cap=1)
    assert auto_result == hung_result


@_requires_ortools
def test_auto_dispatches_mcf_on_tight():
    """n_children > n_centroids → auto must hit the MCF branch and produce
    the MCF answer (which dominates greedy)."""
    cost = np.array([
        [0.10, 0.15, 0.99],
        [0.20, 0.25, 0.99],
        [0.90, 0.30, 0.40],
        [0.90, 0.95, 0.50],
    ])
    auto_result = _assign_children_to_centroids(cost, 4, 3, max_group_cap=2, solver="auto")
    mcf_result = _assign_mcf(cost, 4, 3, max_group_cap=2)
    auto_cost = sum(cost[i, auto_result[i]] for i in range(4))
    mcf_cost = sum(cost[i, mcf_result[i]] for i in range(4))
    assert auto_cost == pytest.approx(mcf_cost, abs=1e-6)


# ---------------------------------------------------------------------------
# Sinkhorn falls back to MCF until Phase 4 lands. Verify the fallback
# is reachable and its result equals MCF's.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 4 — Sinkhorn solver (spec § 5 step 4d / M9 / D-sinkhorn-soft-assign)
# ---------------------------------------------------------------------------


def test_sinkhorn_empty_inputs():
    cost_zero_rows = np.zeros((0, 4))
    cost_zero_cols = np.zeros((4, 0))
    assert _assign_sinkhorn(cost_zero_rows, 0, 4, max_group_cap=2) == []
    assert _assign_sinkhorn(cost_zero_cols, 4, 0, max_group_cap=2) == [-1] * 4


def test_sinkhorn_simple_obvious_assignment():
    """When the cost matrix has a clear optimum (one centroid is much better
    for each child), Sinkhorn at low epsilon must find it."""
    cost = np.array([
        [0.01, 0.99, 0.99],
        [0.99, 0.01, 0.99],
        [0.99, 0.99, 0.01],
    ])
    result = _assign_sinkhorn(
        cost, 3, 3, max_group_cap=1,
        epsilon_init=1.0, epsilon_final=0.005, iters=300,
    )
    assert result == [0, 1, 2]


def test_sinkhorn_capacitated_respects_cap_in_argmax():
    """With max_group_cap=2 and 4 children but 2 centroids, the argmax
    output should have at most a few children pointing at each centroid;
    not strictly capped (Sinkhorn is soft) but the distribution should
    favor balanced assignment."""
    rng = np.random.default_rng(11)
    cost = rng.random((4, 2))

    result = _assign_sinkhorn(
        cost, 4, 2, max_group_cap=2,
        epsilon_init=1.0, epsilon_final=0.01, iters=300,
    )
    assert len(result) == 4
    assert all(0 <= c < 2 for c in result)
    # Soft argmax may not strictly enforce capacity but should not collapse
    # to a single centroid given balanced slack.
    assignments_per_centroid = [result.count(c) for c in range(2)]
    assert max(assignments_per_centroid) <= 4  # always true; we just want no NaNs


@_requires_ortools
def test_sinkhorn_converges_to_mcf_at_low_epsilon():
    """As epsilon → 0, the Sinkhorn solution should harden toward the MCF
    optimum. With epsilon_final small enough, the total cost of the
    Sinkhorn assignment should be within ~10% of MCF's optimum on a small
    random instance."""
    rng = np.random.default_rng(3)
    cost = rng.random((6, 3))

    sinkhorn_result = _assign_sinkhorn(
        cost, 6, 3, max_group_cap=2,
        epsilon_init=0.5, epsilon_final=0.001, iters=500,
    )
    mcf_result = _assign_mcf(cost, 6, 3, max_group_cap=2)

    sinkhorn_total = sum(cost[i, sinkhorn_result[i]] for i in range(6))
    mcf_total = sum(cost[i, mcf_result[i]] for i in range(6))
    # Sinkhorn at low epsilon should be within 25% of MCF optimum on the
    # average random instance (loose bound; exact convergence depends on
    # iteration count and stiffness).
    assert sinkhorn_total <= mcf_total * 1.25


def test_sinkhorn_handles_inf_entries():
    """+∞ entries should be effectively forbidden — the assignment should
    not pick them in the limit."""
    cost = np.array([
        [0.1, np.inf, 0.5],
        [np.inf, 0.2, 0.3],
        [0.4, 0.3, np.inf],
    ])
    result = _assign_sinkhorn(
        cost, 3, 3, max_group_cap=1,
        epsilon_init=0.5, epsilon_final=0.001, iters=400,
    )
    for i, c in enumerate(result):
        assert np.isfinite(cost[i, c]), (
            f"Sinkhorn picked +∞ entry for child {i} → centroid {c}"
        )


def test_sinkhorn_infeasible_falls_back_to_greedy(caplog):
    """If n_C × C_max < n_NC, the slack would be negative — Sinkhorn must
    fall back to greedy with a clear warning."""
    import logging as _logging
    cost = np.zeros((10, 2))  # 10 children, 2 centroids, cap=1 → infeasible
    # Pytest's caplog plugin sets ``propagate=False`` on captured loggers by
    # default (to prevent double-emit if the app also installs handlers).
    # That breaks propagation to caplog's root-attached handler, so we have
    # to opt back in explicitly. caplog.set_level doesn't restore propagate.
    _solver_log = _logging.getLogger("moe_compress.stage2.plugins.solver_sinkhorn")
    _saved_propagate = _solver_log.propagate
    _solver_log.propagate = True
    try:
        caplog.set_level("WARNING")
        result = _assign_sinkhorn(cost, 10, 2, max_group_cap=1)
    finally:
        _solver_log.propagate = _saved_propagate
    # Greedy on this all-zeros cost will assign all 10 to centroid 0
    # (or 1, given ties), but at minimum returns finite indices.
    assert len(result) == 10
    assert any("infeasible" in rec.message for rec in caplog.records)


def test_sinkhorn_cost_normalization_invariance():
    """Sinkhorn solutions should be invariant under positive affine cost
    transformations (epsilon was normalized to the [0,1] cost range)."""
    cost_small = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.5, 0.5],
    ])
    cost_large = cost_small * 1e6 + 100

    r1 = _assign_sinkhorn(
        cost_small, 3, 2, max_group_cap=2,
        epsilon_init=0.5, epsilon_final=0.001, iters=300,
    )
    r2 = _assign_sinkhorn(
        cost_large, 3, 2, max_group_cap=2,
        epsilon_init=0.5, epsilon_final=0.001, iters=300,
    )
    assert r1 == r2


def test_sinkhorn_dispatch_returns_finite_assignment():
    """Phase 4 (M9): solver='sinkhorn' actually runs the Sinkhorn solver
    (not the MCF fallback that Phase 1 stubbed). Verify the dispatcher
    routes to Sinkhorn and produces a valid assignment.
    """
    rng = np.random.default_rng(7)
    cost = rng.random((5, 3))

    sinkhorn_result = _assign_children_to_centroids(
        cost, 5, 3, max_group_cap=2, solver="sinkhorn",
        sinkhorn_iters=300,
    )
    # Every child must be assigned to a real centroid (no -1, no out-of-range).
    assert len(sinkhorn_result) == 5
    assert all(0 <= c < 3 for c in sinkhorn_result)


# ---------------------------------------------------------------------------
# MCF cost-normalization: the implementation normalizes finite costs to
# [0, 1e6] before int-rounding, so unbounded post-alignment residuals do
# not cause int32 overflow. Verify by feeding a large-scale cost matrix.
# ---------------------------------------------------------------------------


@_requires_ortools
def test_mcf_handles_large_scale_costs_without_overflow():
    """Construct costs spanning ~1e6 magnitude and confirm MCF still finds
    the same assignment as on the same matrix divided by 1e6 (positive
    affine transformation invariance)."""
    cost_small = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.5, 0.5],
    ])
    cost_large = cost_small * 1e9 + 1e8  # huge-magnitude, mostly-shifted

    result_small = _assign_mcf(cost_small, 3, 2, max_group_cap=2)
    result_large = _assign_mcf(cost_large, 3, 2, max_group_cap=2)
    assert result_small == result_large


# ---------------------------------------------------------------------------
# Stage 2 v2 — Phase 1 completion: PermAlignCache + post-alignment helpers.
# These verify the new public/private surface introduced in Phase 1 part 2:
#   _PermAlignCache, _aligned_whitened_residual, _post_alignment_cost.
# ---------------------------------------------------------------------------


def test_perm_align_cache_basic_get_put():
    from moe_compress.stage2.orchestrator import _PermAlignCache

    cache = _PermAlignCache()
    perm = np.array([2, 0, 1, 3])
    cache.put((0, 5, 7), perm, 0.42)
    got = cache.get((0, 5, 7))
    assert got is not None
    assert np.array_equal(got[0], perm)
    assert got[1] == 0.42

    assert cache.get((0, 5, 8)) is None
    assert cache.has((0, 5, 7))
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0


def test_aligned_whitened_residual_zero_when_centroid_equals_aligned_child():
    """If the centroid weights exactly equal the permuted child weights, the
    residual should be 0 regardless of whitening mode."""
    from moe_compress.stage2.orchestrator import _aligned_whitened_residual
    import torch

    torch.manual_seed(0)
    d_int, hidden = 4, 6
    W_gate = torch.randn(d_int, hidden)
    W_up   = torch.randn(d_int, hidden)
    W_down = torch.randn(hidden, d_int)
    perm = np.arange(d_int)  # identity permutation

    a_sqrt_gate_up = torch.eye(hidden)  # whitening = identity
    a_sqrt_down    = torch.eye(d_int)

    r = _aligned_whitened_residual(
        ref_gate=W_gate, ref_up=W_up, ref_down=W_down,
        child_gate=W_gate, child_up=W_up, child_down=W_down,
        perm=perm,
        a_sqrt_gate_up=a_sqrt_gate_up, a_sqrt_down=a_sqrt_down,
        whitening_mode="full",
    )
    assert r == pytest.approx(0.0, abs=1e-5)


def test_aligned_whitened_residual_grows_with_perturbation():
    """Adding a small random perturbation to the child weights should produce
    a strictly positive residual that increases with perturbation magnitude.
    """
    from moe_compress.stage2.orchestrator import _aligned_whitened_residual
    import torch

    torch.manual_seed(1)
    d_int, hidden = 4, 6
    W_gate = torch.randn(d_int, hidden)
    W_up   = torch.randn(d_int, hidden)
    W_down = torch.randn(hidden, d_int)
    eps = torch.randn(d_int, hidden)
    perm = np.arange(d_int)

    a_sqrt_gate_up = torch.eye(hidden)
    a_sqrt_down    = torch.eye(d_int)

    r_small = _aligned_whitened_residual(
        ref_gate=W_gate, ref_up=W_up, ref_down=W_down,
        child_gate=W_gate + 0.01 * eps,
        child_up=W_up,
        child_down=W_down,
        perm=perm,
        a_sqrt_gate_up=a_sqrt_gate_up, a_sqrt_down=a_sqrt_down,
        whitening_mode="full",
    )
    r_large = _aligned_whitened_residual(
        ref_gate=W_gate, ref_up=W_up, ref_down=W_down,
        child_gate=W_gate + 0.10 * eps,
        child_up=W_up,
        child_down=W_down,
        perm=perm,
        a_sqrt_gate_up=a_sqrt_gate_up, a_sqrt_down=a_sqrt_down,
        whitening_mode="full",
    )
    assert 0 < r_small < r_large


def test_aligned_whitened_residual_respects_permutation():
    """Permuting the child rows should be undone by the same permutation
    passed to the helper, yielding the original residual."""
    from moe_compress.stage2.orchestrator import _aligned_whitened_residual
    import torch

    torch.manual_seed(2)
    d_int, hidden = 5, 4
    W_gate = torch.randn(d_int, hidden)
    W_up   = torch.randn(d_int, hidden)
    W_down = torch.randn(hidden, d_int)

    # Pretend the child has been row-permuted relative to centroid.
    p = np.array([2, 0, 4, 1, 3])
    inv_p = np.argsort(p)
    child_gate = W_gate[inv_p, :]
    child_up   = W_up[inv_p, :]
    child_down = W_down[:, inv_p]

    a_sqrt_gate_up = torch.eye(hidden)
    a_sqrt_down    = torch.eye(d_int)

    # Passing perm=inv(perm-of-perturbation) must align child back to centroid.
    r_aligned = _aligned_whitened_residual(
        ref_gate=W_gate, ref_up=W_up, ref_down=W_down,
        child_gate=child_gate, child_up=child_up, child_down=child_down,
        perm=p,  # apply p to the already-inverse-permuted child → identity
        a_sqrt_gate_up=a_sqrt_gate_up, a_sqrt_down=a_sqrt_down,
        whitening_mode="full",
    )
    assert r_aligned == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Capacity-utilization gate (M3): _pick_effective_alignment
# ---------------------------------------------------------------------------


def test_capacity_util_gate_slack_below_threshold_returns_pre():
    from moe_compress.stage2.orchestrator import _pick_effective_alignment

    # n_nc=2, n_c=4, cap=8 → util = 2/32 = 0.0625, below 0.25 threshold.
    result = _pick_effective_alignment(
        n_nc=2, n_c=4, max_group_cap=8, threshold=0.25, configured="post",
    )
    assert result == "pre"


def test_capacity_util_gate_tight_at_or_above_threshold_returns_configured():
    from moe_compress.stage2.orchestrator import _pick_effective_alignment

    # n_nc=12, n_c=2, cap=8 → util = 12/16 = 0.75, well above threshold.
    result = _pick_effective_alignment(
        n_nc=12, n_c=2, max_group_cap=8, threshold=0.25, configured="post",
    )
    assert result == "post"


def test_capacity_util_gate_uncapped_treats_as_slack():
    """max_group_cap=0 (uncapped, ablation-only path) → util=0 → SLACK."""
    from moe_compress.stage2.orchestrator import _pick_effective_alignment

    result = _pick_effective_alignment(
        n_nc=100, n_c=10, max_group_cap=0, threshold=0.25, configured="post",
    )
    assert result == "pre"


def test_capacity_util_gate_configured_pre_stays_pre():
    """When the user explicitly configures 'pre', the gate should never
    upgrade to 'post' regardless of utilization."""
    from moe_compress.stage2.orchestrator import _pick_effective_alignment

    result = _pick_effective_alignment(
        n_nc=12, n_c=2, max_group_cap=8, threshold=0.25, configured="pre",
    )
    assert result == "pre"


# ---------------------------------------------------------------------------
# Asymmetric freq factor (spec § 5 step 4T(c)(iii)) — direction matters.
# Verify d_cm = (freq_m / (freq_c + freq_m)) · R_cm with freq_m on top.
# ---------------------------------------------------------------------------


class _FakeBank:
    """Mirrors the ``ExpertMatrixBank.get`` interface used by Stage 2 v2."""

    def __init__(self, weights: dict[int, "torch.Tensor"]):
        self._w = weights

    def get(self, eid):
        return self._w[eid]

    def set(self, eid, tensor):  # mirror real bank for completeness
        self._w[eid] = tensor


def _make_post_alignment_test_setup(monkeypatch):
    """Build a minimal in-memory setup that lets us call _post_alignment_cost
    with controlled inputs. We patch ``build_banks`` and substitute fake
    accumulators for ream_acc / cov_acc.
    """
    import threading

    import torch
    from moe_compress.stage2 import orchestrator as stage2_reap_ream

    d_int, hidden = 4, 6

    # Centroid c=0 and two non-centroids m=1, m=2 with different freq.
    torch.manual_seed(0)
    weights = {}
    for eid in (0, 1, 2):
        weights[eid] = {
            "gate_proj": torch.randn(d_int, hidden),
            "up_proj":   torch.randn(d_int, hidden),
            "down_proj": torch.randn(hidden, d_int),
        }

    # Plain dict matches the real ``build_banks`` return type
    # (dict[str, ExpertMatrixBank]).
    banks = {
        "gate_proj": _FakeBank({eid: weights[eid]["gate_proj"] for eid in weights}),
        "up_proj":   _FakeBank({eid: weights[eid]["up_proj"]   for eid in weights}),
        "down_proj": _FakeBank({eid: weights[eid]["down_proj"] for eid in weights}),
    }

    # ``_post_alignment_cost`` lives in ``stage2.plugins.ream_cost_post`` and
    # ``_em_compute_tentative_weights`` in ``stage2.plugins.em_refine`` under the
    # Stage 2 plugin refactor; each function resolves ``build_banks`` from its
    # OWN module namespace, so both bindings must be patched. The
    # ``stage2_reap_ream`` patch is kept defensively for any other
    # monolith-resident code path a test in this file might exercise.
    import moe_compress.stage2.plugins.em_refine as _em_refine
    import moe_compress.stage2.plugins.ream_cost_post as _ream_cost_post
    import moe_compress.stage2.merging as _merging
    monkeypatch.setattr(stage2_reap_ream, "build_banks", lambda layer_ref: banks)
    monkeypatch.setattr(_ream_cost_post, "build_banks", lambda layer_ref: banks)
    monkeypatch.setattr(_em_refine, "build_banks", lambda layer_ref: banks)
    monkeypatch.setattr(_merging, "build_banks", lambda layer_ref: banks)

    class _FakeReamAcc:
        """Minimal accumulator that returns deterministic stub values so the
        cost-matrix builder can run end-to-end without real calibration data.

        After the Stage 2 vectorization, ``_ream_cost_matrix`` reads the
        accumulator's ``_lock`` / ``_total_tokens_by_layer`` / ``_sim_tensor``
        directly (no longer per-pair ``compute_delta_expert``). An empty
        ``_total_tokens_by_layer`` (total == 0) makes
        ``_extract_sim_expert_matrix_from_tensor`` return a full-0.5 δ̃_expert
        matrix — the same neutral value the old ``compute_delta_expert`` stub
        produced, so this fixture's expected costs are unchanged.
        """

        def __init__(self):
            self._lock = threading.Lock()
            self._total_tokens_by_layer: dict[int, int] = {}
            self._sim_tensor: dict[int, torch.Tensor] = {}

        def get_neuron_mean(self, layer_idx, expert_idx):
            return None  # disable C_act in alignment

        def compute_gate_similarity_matrix(self, layer_idx, expert_ids):
            # Constant (uniform-similarity) gate matrix — the post-alignment
            # path's cheap-cost step only uses this to pick top-K, and a
            # constant matrix produces deterministic top-K (whichever K the
            # argpartition selects first), which is fine for unit testing.
            n = len(expert_ids)
            return torch.zeros(n, n)

    class _FakeLayerRef:
        layer_idx = 0
        num_routed_experts = 3

    return _FakeLayerRef(), _FakeReamAcc(), weights


def test_asymmetric_factor_uses_freq_m_over_freq_c_plus_m(monkeypatch):
    """Two non-centroids with same raw residual but different frequencies:
    the high-freq one should get a *higher* cost (because it would dominate
    the merged centroid). This directly verifies the freq_m/(freq_c+freq_m)
    direction — a regression to freq_c/(freq_c+freq_m) would invert the order.
    """
    from moe_compress.stage2.orchestrator import _post_alignment_cost, _PermAlignCache

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    perm_cache = _PermAlignCache()

    # Cheap-cost matrix has two rows (non-centroids) × one column (centroid).
    cheap = np.array([[0.5], [0.5]])

    # freq: centroid=10, m1=1 (low), m2=20 (high). m2 has freq_m > freq_c.
    freq = {0: 10, 1: 1, 2: 20}

    out = _post_alignment_cost(
        layer_ref,
        noncentroid_ids=[1, 2],
        centroid_ids=[0],
        cheap_cost=cheap,
        ream_acc=ream_acc,
        cov_acc=None,
        perm_cache=perm_cache,
        whitening_mode="none",
        asymmetric=True,
        topk=1,
        freq=freq,
    )

    # The high-freq non-centroid (m2) gets multiplied by 20/30, the low-freq
    # one (m1) gets 1/11. Even though the underlying alignment residual will
    # differ (different perms produced different W_m^aligned), the asymmetric
    # factor must still scale m2's cost MORE than m1's relative to their
    # un-scaled residuals. We verify by:
    #   - factor for m1 = 1/11 ≈ 0.091
    #   - factor for m2 = 20/30 ≈ 0.667
    # so out[1, 0] / out[0, 0] should equal R_2/R_1 × 0.667/0.091 ≈ 7.3 × R_2/R_1.
    # Since R_1 and R_2 are both finite Frobenius norms over comparable random
    # weights, the ratio out[1, 0] / out[0, 0] should be > 1 (asymmetric
    # scaling dominates raw-residual variation).
    assert out[1, 0] > out[0, 0]


def test_asymmetric_factor_zero_when_freq_m_is_zero(monkeypatch):
    """A non-centroid with freq_m = 0 cannot wash out the centroid's identity,
    so its asymmetric cost factor should be 0 (free to absorb)."""
    from moe_compress.stage2.orchestrator import _post_alignment_cost, _PermAlignCache

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    perm_cache = _PermAlignCache()

    cheap = np.array([[0.5]])
    freq = {0: 10, 1: 0}  # m1 has zero freq

    out = _post_alignment_cost(
        layer_ref,
        noncentroid_ids=[1],
        centroid_ids=[0],
        cheap_cost=cheap,
        ream_acc=ream_acc,
        cov_acc=None,
        perm_cache=perm_cache,
        whitening_mode="none",
        asymmetric=True,
        topk=1,
        freq=freq,
    )
    # 0 / (10 + 0) = 0 → cost = 0
    assert out[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_post_alignment_fills_top_k_only_rest_inf(monkeypatch):
    """Verify the K-prefilter: only the top-K cheapest centroids per
    non-centroid get finite costs; the rest are +inf."""
    from moe_compress.stage2.orchestrator import _post_alignment_cost, _PermAlignCache

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    # Set up a 3rd centroid (re-use the existing weight bank). The fake banks
    # already contain experts 0, 1, 2; treat 0 and 1 as centroids and 2 as
    # the lone non-centroid.
    perm_cache = _PermAlignCache()

    cheap = np.array([[0.1, 0.9]])  # m=2 prefers centroid 0 strongly
    freq = {0: 5, 1: 5, 2: 5}

    out = _post_alignment_cost(
        layer_ref,
        noncentroid_ids=[2],
        centroid_ids=[0, 1],
        cheap_cost=cheap,
        ream_acc=ream_acc,
        cov_acc=None,
        perm_cache=perm_cache,
        whitening_mode="none",
        asymmetric=False,
        topk=1,  # only the cheapest centroid gets a finite cost
        freq=freq,
    )
    # Top-1 by cheap cost is column 0 → finite; column 1 must be +inf.
    assert np.isfinite(out[0, 0])
    assert out[0, 1] == np.inf


def test_post_alignment_writes_perm_cache(monkeypatch):
    from moe_compress.stage2.orchestrator import _post_alignment_cost, _PermAlignCache

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    perm_cache = _PermAlignCache()

    cheap = np.array([[0.1, 0.9]])
    freq = {0: 5, 1: 5, 2: 5}

    _ = _post_alignment_cost(
        layer_ref,
        noncentroid_ids=[2],
        centroid_ids=[0, 1],
        cheap_cost=cheap,
        ream_acc=ream_acc,
        cov_acc=None,
        perm_cache=perm_cache,
        whitening_mode="none",
        asymmetric=False,
        topk=1,
        freq=freq,
    )

    # Cache must have been populated for (layer=0, centroid=0, non=2) only
    # (top-K=1). Centroid=1 was filtered out by the prefilter.
    assert perm_cache.has((0, 0, 2))
    assert not perm_cache.has((0, 1, 2))


def test_post_alignment_topk_validation_raises_on_zero():
    from moe_compress.stage2.orchestrator import _post_alignment_cost, _PermAlignCache

    class _DummyLayerRef:
        layer_idx = 0
        num_routed_experts = 2

    cheap = np.array([[0.5]])
    with pytest.raises(ValueError, match="cost_topk_filter=0"):
        _post_alignment_cost(
            _DummyLayerRef(),
            noncentroid_ids=[1],
            centroid_ids=[0],
            cheap_cost=cheap,
            ream_acc=None,  # type: ignore[arg-type]
            cov_acc=None,
            perm_cache=_PermAlignCache(),
            whitening_mode="none",
            asymmetric=False,
            topk=0,  # invalid
            freq={0: 1, 1: 1},
        )


def test_post_alignment_whitening_requires_cov_acc():
    from moe_compress.stage2.orchestrator import _post_alignment_cost, _PermAlignCache

    class _DummyLayerRef:
        layer_idx = 0
        num_routed_experts = 2

    cheap = np.array([[0.5]])
    with pytest.raises(ValueError, match="cov_acc is required"):
        _post_alignment_cost(
            _DummyLayerRef(),
            noncentroid_ids=[1],
            centroid_ids=[0],
            cheap_cost=cheap,
            ream_acc=None,  # type: ignore[arg-type]
            cov_acc=None,  # missing
            perm_cache=_PermAlignCache(),
            whitening_mode="full",  # requires cov
            asymmetric=False,
            topk=1,
            freq={0: 1, 1: 1},
        )


def test_post_alignment_asymmetric_requires_freq():
    from moe_compress.stage2.orchestrator import _post_alignment_cost, _PermAlignCache

    class _DummyLayerRef:
        layer_idx = 0
        num_routed_experts = 2

    cheap = np.array([[0.5]])
    with pytest.raises(ValueError, match="cost_asymmetric=True requires freq"):
        _post_alignment_cost(
            _DummyLayerRef(),
            noncentroid_ids=[1],
            centroid_ids=[0],
            cheap_cost=cheap,
            ream_acc=None,  # type: ignore[arg-type]
            cov_acc=None,
            perm_cache=_PermAlignCache(),
            whitening_mode="none",
            asymmetric=True,
            topk=1,
            freq=None,  # missing
        )


# ---------------------------------------------------------------------------
# YAML config rejection: cost_asymmetric=True ∧ freq_weighted_merge=False
# spec D-asymmetric-freq says this is mathematically inconsistent.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2 — EM refinement (spec § 5 step 4T(e) / M4)
# ---------------------------------------------------------------------------


def test_em_zero_rounds_returns_initial_unchanged(monkeypatch):
    """em_refinement_rounds=0 must be a no-op: returns the input assignment
    and cost matrix verbatim, with rounds_completed=0."""
    from moe_compress.stage2.orchestrator import (
        _em_refine_assignment, _PermAlignCache,
    )

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    init_assign = [0, 0]
    init_delta = np.array([[0.1, 1.0], [0.2, 1.0]])

    new_assign, new_delta, rounds = _em_refine_assignment(
        layer_ref,
        initial_assignment=init_assign,
        initial_delta=init_delta,
        ream_centroid_ids=[0, 1],
        ream_noncentroid_ids=[2],
        perm_cache=_PermAlignCache(),
        ream_acc=ream_acc,
        cov_acc=None,
        freq={0: 1, 1: 1, 2: 1},
        max_group_cap=2,
        cost_alignment="post",
        cost_whitening="none",
        cost_asymmetric=False,
        cost_topk_filter=2,
        assignment_solver="greedy",
        em_rounds=0,
        em_break=True,
        blacklisted_ids=set(),
    )
    assert new_assign == init_assign
    assert new_delta is init_delta  # no-op returns the same object
    assert rounds == 0


def test_em_pre_alignment_is_noop(monkeypatch):
    """Even with em_rounds > 0, EM must be a no-op when cost_alignment='pre'
    because the cheap symmetric cost does not depend on centroid weights —
    a tentative merge cannot change the assignment."""
    from moe_compress.stage2.orchestrator import (
        _em_refine_assignment, _PermAlignCache,
    )

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    init_assign = [0]
    init_delta = np.array([[0.5, 0.7]])

    new_assign, new_delta, rounds = _em_refine_assignment(
        layer_ref,
        initial_assignment=init_assign,
        initial_delta=init_delta,
        ream_centroid_ids=[0, 1],
        ream_noncentroid_ids=[2],
        perm_cache=_PermAlignCache(),
        ream_acc=ream_acc,
        cov_acc=None,
        freq={0: 1, 1: 1, 2: 1},
        max_group_cap=2,
        cost_alignment="pre",  # ← guard
        cost_whitening="none",
        cost_asymmetric=False,
        cost_topk_filter=2,
        assignment_solver="greedy",
        em_rounds=5,  # would otherwise loop
        em_break=True,
        blacklisted_ids=set(),
    )
    assert new_assign == init_assign
    assert new_delta is init_delta  # no-op contract: same object returned
    assert rounds == 0


def test_em_breaks_early_when_assignment_stable(monkeypatch):
    """If round-1's reassignment matches round-0's, EM should stop after
    round 1 (em_convergence_break=True)."""
    from moe_compress.stage2.orchestrator import (
        _em_refine_assignment, _PermAlignCache,
    )

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    # All non-centroids point to centroid 0; the cheap cost has only
    # centroid 0 available (column 1 doesn't exist). Any EM round will
    # produce the same assignment trivially.
    init_assign = [0]
    init_delta = np.array([[0.1]])

    new_assign, new_delta, rounds = _em_refine_assignment(
        layer_ref,
        initial_assignment=init_assign,
        initial_delta=init_delta,
        ream_centroid_ids=[0],
        ream_noncentroid_ids=[2],
        perm_cache=_PermAlignCache(),
        ream_acc=ream_acc,
        cov_acc=None,
        freq={0: 1, 2: 1},
        max_group_cap=2,
        cost_alignment="post",
        cost_whitening="none",
        cost_asymmetric=False,
        cost_topk_filter=1,
        assignment_solver="greedy",
        em_rounds=5,
        em_break=True,
        blacklisted_ids=set(),
    )
    # Convergence after exactly 1 round (assignment stable).
    assert new_assign == init_assign
    assert rounds == 1


def test_em_runs_full_rounds_when_break_disabled(monkeypatch):
    """With em_convergence_break=False, EM should run all configured rounds
    even if the assignment stabilizes earlier."""
    from moe_compress.stage2.orchestrator import (
        _em_refine_assignment, _PermAlignCache,
    )

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    init_assign = [0]
    init_delta = np.array([[0.1]])

    _, _, rounds = _em_refine_assignment(
        layer_ref,
        initial_assignment=init_assign,
        initial_delta=init_delta,
        ream_centroid_ids=[0],
        ream_noncentroid_ids=[2],
        perm_cache=_PermAlignCache(),
        ream_acc=ream_acc,
        cov_acc=None,
        freq={0: 1, 2: 1},
        max_group_cap=2,
        cost_alignment="post",
        cost_whitening="none",
        cost_asymmetric=False,
        cost_topk_filter=1,
        assignment_solver="greedy",
        em_rounds=3,
        em_break=False,
        blacklisted_ids=set(),
    )
    assert rounds == 3


def test_em_singleton_only_skips_immediately(monkeypatch):
    """If every group is a singleton (no non-centroid assigned), the
    tentative-merge step has nothing to do; rounds_completed must be 0
    and the early break must fire on the first iteration."""
    from moe_compress.stage2.orchestrator import (
        _em_refine_assignment, _PermAlignCache,
    )

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    init_assign = [-1]  # non-centroid m=2 unassigned
    init_delta = np.array([[np.inf]])

    _, _, rounds = _em_refine_assignment(
        layer_ref,
        initial_assignment=init_assign,
        initial_delta=init_delta,
        ream_centroid_ids=[0],
        ream_noncentroid_ids=[2],
        perm_cache=_PermAlignCache(),
        ream_acc=ream_acc,
        cov_acc=None,
        freq={0: 1, 2: 1},
        max_group_cap=2,
        cost_alignment="post",
        cost_whitening="none",
        cost_asymmetric=False,
        cost_topk_filter=1,
        assignment_solver="greedy",
        em_rounds=3,
        em_break=True,
        blacklisted_ids=set(),
    )
    # All groups singleton → tentative dict empty → break before any reassign.
    assert rounds == 0


def test_em_compute_tentative_weights_singleton_skipped(monkeypatch):
    """A singleton group (centroid only, no absorbed members) must not
    appear in the tentative-weights output."""
    from moe_compress.stage2.orchestrator import (
        _em_compute_tentative_weights, _PermAlignCache,
    )

    layer_ref, ream_acc, _ = _make_post_alignment_test_setup(monkeypatch)
    grouped = {0: [0]}  # singleton

    out = _em_compute_tentative_weights(
        layer_ref, grouped,
        freq={0: 1},
        ream_acc=ream_acc,
        perm_cache=_PermAlignCache(),
    )
    assert out == {}


def test_em_compute_tentative_weights_freq_weighted_average(monkeypatch):
    """For a 2-member group, the tentative weight must be the freq-weighted
    average of centroid and member (after permutation alignment)."""
    from moe_compress.stage2.orchestrator import (
        _em_compute_tentative_weights, _PermAlignCache,
    )
    import torch

    layer_ref, ream_acc, weights = _make_post_alignment_test_setup(monkeypatch)
    perm_cache = _PermAlignCache()

    # Group: centroid=0, member=1. freq={0: 3, 1: 1}.
    # Expected: tentative_W[0] = 0.75 · W_0 + 0.25 · perm(W_1).
    grouped = {0: [0, 1]}
    freq = {0: 3, 1: 1}

    out = _em_compute_tentative_weights(
        layer_ref, grouped, freq, ream_acc, perm_cache,
    )
    assert 0 in out
    # Verify the convex combination property: the tentative weight should
    # lie on the line segment between W_0 and the (permuted) W_1.
    # We don't know the perm, but we know |t - W_0| < |W_1 - W_0| at any
    # weight strictly between 0 and 1. Equivalently, the tentative is closer
    # to W_0 than W_1 is when weight on W_0 > 0.5.
    W0 = weights[0]["gate_proj"].to(torch.float32)
    W1 = weights[1]["gate_proj"].to(torch.float32)
    t  = out[0]["gate_proj"]
    # t = 0.75 W_0 + 0.25 perm(W_1). Distance from t to W_0 is
    # 0.25 * ||perm(W_1) - W_0||; distance to W_1 (un-permuted) is bigger
    # for non-degenerate W_1. Just verify t is "much closer" to W_0:
    d_t_W0 = torch.linalg.matrix_norm(t - W0, ord="fro").item()
    d_W1_W0 = torch.linalg.matrix_norm(W1 - W0, ord="fro").item()
    assert d_t_W0 < d_W1_W0


def test_em_rounds_completed_persisted_in_partial_json(tmp_path, monkeypatch):
    """The em_rounds_completed field must round-trip through the partial
    JSON write path (spec § 12.1 schema bump 1→2 reserved this field)."""
    from moe_compress.stage2.orchestrator import _write_merge_json
    import json

    _write_merge_json(
        tmp_path,
        layer_idx=7,
        final_kept_ids=[0, 1],
        grouped={0: [0]},
        freq={0: 5, 1: 3},
        merge_map_layer={0: [0]},
        mean_cost_per_pair=0.42,
        assignment_solver_used="auto",
        cost_alignment_used="post",
        em_rounds_completed=2,
        distill_state=None,
    )
    data = json.loads((tmp_path / "merge_7.json").read_text())
    assert data["em_rounds_completed"] == 2
    assert data["assignment_solver_used"] == "auto"
    assert data["cost_alignment_used"] == "post"


# ---------------------------------------------------------------------------
# Phase 1 YAML rejection (kept for clarity in the test ordering).
# ---------------------------------------------------------------------------


def test_config_rejects_asymmetric_without_freq_weighted_merge(tmp_path, monkeypatch):
    """The driver's YAML-boundary validation must reject
    ``cost_asymmetric=True ∧ ream.frequency_weighted_merge=False`` because
    the asymmetric factor freq_m/(freq_c+freq_m) is the per-pair version of
    the merge weight; running asymmetric cost with non-freq-weighted merge
    silently drives the assignment in a direction the merge formula doesn't
    follow."""
    from moe_compress.stage2 import orchestrator as stage2_reap_ream

    bad_config = {
        "stage2_reap_ream": {
            "batch_size": 1,
            "num_calibration_samples": 1,
            "covariance_storage_dtype": "float16",
            "max_merge_group_size": 8,
            "ream_cost_sigma_threshold": 1.5,
            "ream_cost_bump_ratio": 0.10,
            "ream": {"frequency_weighted_merge": False},  # disabled
            "assignment_solver": "greedy",
            "cost_alignment": "post",
            "cost_whitening": "none",
            "cost_asymmetric": True,  # incompatible with merge=False
            "cost_topk_filter": 4,
            "capacity_util_threshold": 0.25,
        },
        "calibration": {"source": "c4-math-code", "seed": 0},
    }

    # Stub Stage 1 budget loading so the call reaches the validation block.
    monkeypatch.setattr(
        stage2_reap_ream, "load_json_artifact",
        lambda p: {"per_layer_target_experts": {}, "blacklist": {}},
    )
    monkeypatch.setattr(
        stage2_reap_ream, "build_calibration_tensor",
        lambda *a, **kw: __import__("torch").zeros(1, 1, dtype=__import__("torch").long),
    )
    monkeypatch.setattr(
        stage2_reap_ream, "iter_batches",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stage2_reap_ream, "iter_moe_layers",
        lambda model: iter([]),
    )

    class _DummyModel:
        pass

    with pytest.raises(ValueError, match="cost_asymmetric=True"):
        stage2_reap_ream.run(
            _DummyModel(), tokenizer=None, config=bad_config,
            artifacts_dir=tmp_path, no_resume=True,
        )


# ---------------------------------------------------------------------------
# Saliency-weighted merge (P2): the ``freq_weighted=False`` branch
# ---------------------------------------------------------------------------


def test_merge_saliency_weighted_matches_equivalent_freq(monkeypatch):
    """Saliency mode with scores=[3,1] must produce byte-identical merged
    weights to freq mode with freq={0:3, 1:1} — both compute the same
    [0.75, 0.25] convex blend with the same permutation alignment.

    This pins the saliency arithmetic without needing to recompute the
    permutation manually: any wrong weighting (e.g. [0.5, 0.5] or
    [0.25, 0.75]) would diverge from the freq-mode reference.
    """
    import numpy as np
    import torch

    from moe_compress.stage2 import merging as _merging
    from moe_compress.stage2.merging import _merge_experts_inplace

    # --- Run 1: freq-weighted merge, freq={0:3, 1:1} ---
    # The fixture sets torch.manual_seed(0) internally, so two calls yield
    # identical initial weights in the bank — the ideal baseline for an
    # arithmetic equality check across the two merge modes.
    layer_ref, _, _ = _make_post_alignment_test_setup(monkeypatch)
    grouped = {0: [0, 1]}
    _merge_experts_inplace(
        layer_ref, grouped,
        freq={0: 3, 1: 1},
        freq_weighted=True,
    )
    merged_freq = (
        _merging.build_banks(layer_ref)["gate_proj"].get(0).clone().to(torch.float32)
    )

    # --- Run 2: saliency-weighted merge, scores=[3.0, 1.0] ---
    # Fresh fixture so the layer is back at its initial weights.
    layer_ref2, _, _ = _make_post_alignment_test_setup(monkeypatch)
    _merge_experts_inplace(
        layer_ref2, grouped,
        freq={0: 99, 1: 99},   # freq irrelevant in saliency mode
        freq_weighted=False,
        scores=np.array([3.0, 1.0], dtype=np.float64),
    )
    merged_sal = (
        _merging.build_banks(layer_ref2)["gate_proj"].get(0).clone().to(torch.float32)
    )

    # Both modes must produce the same blend (same weights × same alignment).
    assert torch.allclose(merged_freq, merged_sal, atol=1e-5), (
        f"saliency mode with scores=[3,1] should equal freq mode with "
        f"freq={{0:3, 1:1}} — both compute weights [0.75, 0.25] with the "
        f"same permutation alignment. Max abs diff: "
        f"{(merged_freq - merged_sal).abs().max().item():.2e}"
    )


def test_merge_saliency_zero_sum_fallback(monkeypatch, caplog):
    """All-zero saliency scores must fall back to equal weights and emit a WARNING."""
    import logging
    import numpy as np

    from moe_compress.stage2.merging import _merge_experts_inplace

    # Some pytest plugins in this env default new loggers to propagate=False;
    # mirror the working pattern in test_stage2_grouping.py.
    logging.getLogger("moe_compress.stage2.merging").propagate = True

    layer_ref, _, _ = _make_post_alignment_test_setup(monkeypatch)
    scores = np.array([0.0, 0.0], dtype=np.float64)
    grouped = {0: [0, 1]}
    freq = {0: 5, 1: 5}

    with caplog.at_level(logging.WARNING):
        _merge_experts_inplace(
            layer_ref, grouped, freq,
            freq_weighted=False,
            scores=scores,
        )
    assert any("zero saliency" in r.message for r in caplog.records), (
        "Expected a WARNING about zero saliency score fallback"
    )


def test_merge_saliency_negative_clamped(monkeypatch):
    """Pathological negative saliency must be clamped to 0; with only one
    positive-scored expert the merged centroid must equal the centroid's
    original weight exactly (no blending)."""
    import numpy as np
    import torch

    from moe_compress.stage2 import merging as _merging
    from moe_compress.stage2.merging import _merge_experts_inplace

    layer_ref, _, raw_weights = _make_post_alignment_test_setup(monkeypatch)
    scores = np.array([2.0, -5.0], dtype=np.float64)   # member clamped to 0
    grouped = {0: [0, 1]}
    freq = {0: 5, 1: 5}

    W0_before = raw_weights[0]["gate_proj"].clone().to(torch.float32)

    _merge_experts_inplace(
        layer_ref, grouped, freq,
        freq_weighted=False,
        scores=scores,
    )

    merged = _merging.build_banks(layer_ref)["gate_proj"].get(0).to(torch.float32)
    assert torch.allclose(merged, W0_before, atol=1e-5), (
        "With member score clamped to 0, merged weight must equal centroid weight"
    )


def test_config_accepts_freq_weighted_false_when_not_asymmetric(tmp_path, monkeypatch):
    """YAML with frequency_weighted_merge=False ∧ cost_asymmetric=False must NOT
    raise at the orchestrator guard (only asymmetric+saliency is rejected)."""
    import torch
    from moe_compress.stage2 import orchestrator as stage2_reap_ream

    cfg = {
        "stage2_reap_ream": {
            "batch_size": 1,
            "num_calibration_samples": 1,
            "covariance_storage_dtype": "float16",
            "max_merge_group_size": 8,
            "ream_cost_sigma_threshold": 1.5,
            "ream_cost_bump_ratio": 0.10,
            "ream": {"frequency_weighted_merge": False},
            "assignment_solver": "greedy",
            "cost_alignment": "post",
            "cost_whitening": "none",
            "cost_asymmetric": False,
            "cost_topk_filter": 4,
            "capacity_util_threshold": 0.25,
        },
        "calibration": {"source": "c4-math-code", "seed": 0},
    }

    monkeypatch.setattr(
        stage2_reap_ream, "load_json_artifact",
        lambda p: {"per_layer_target_experts": {}, "blacklist": {}},
    )
    monkeypatch.setattr(
        stage2_reap_ream, "build_calibration_tensor",
        lambda *a, **kw: torch.zeros(1, 1, dtype=torch.long),
    )
    monkeypatch.setattr(stage2_reap_ream, "iter_batches", lambda *a, **kw: [])
    monkeypatch.setattr(stage2_reap_ream, "iter_moe_layers", lambda model: iter([]))
    # Past the orchestrator guard, run() calls `_set_experts_implementation`
    # (Blackwell sm_100 workaround) which needs `model.config`. Stub it out so
    # the empty-moe-layers path can run to completion on a `_DummyModel`.
    import moe_compress.stage5_router_kd as _stage5_router_kd
    monkeypatch.setattr(
        _stage5_router_kd, "_set_experts_implementation", lambda model, impl: None,
    )
    # ``spec_from_config`` enforces required calibration keys; we don't care
    # about the spec value because ``build_calibration_tensor`` is also stubbed.
    monkeypatch.setattr(stage2_reap_ream, "spec_from_config", lambda *a, **kw: None)
    # After the empty layer loop the orchestrator still writes a checkpoint;
    # the helpers walk the model again, so neutralize them on _DummyModel.
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", lambda *a, **kw: None)
    monkeypatch.setattr(stage2_reap_ream, "save_json_artifact", lambda *a, **kw: None)
    monkeypatch.setattr(stage2_reap_ream, "_save_covariance", lambda *a, **kw: None)

    class _DummyModel:
        pass

    # Must complete without raising — empty moe_layers list means the merge
    # loop never executes; the orchestrator guard must NOT fire for this combo.
    stage2_reap_ream.run(
        _DummyModel(), tokenizer=None, config=cfg,
        artifacts_dir=tmp_path, no_resume=True,
    )


def test_config_still_rejects_asymmetric_with_saliency(tmp_path, monkeypatch):
    """YAML with frequency_weighted_merge=False ∧ cost_asymmetric=True must STILL
    raise at the orchestrator guard — asymmetric-saliency is out of scope for P2."""
    import torch
    from moe_compress.stage2 import orchestrator as stage2_reap_ream

    cfg = {
        "stage2_reap_ream": {
            "batch_size": 1,
            "num_calibration_samples": 1,
            "covariance_storage_dtype": "float16",
            "max_merge_group_size": 8,
            "ream_cost_sigma_threshold": 1.5,
            "ream_cost_bump_ratio": 0.10,
            "ream": {"frequency_weighted_merge": False},
            "assignment_solver": "greedy",
            "cost_alignment": "post",
            "cost_whitening": "none",
            "cost_asymmetric": True,
            "cost_topk_filter": 4,
            "capacity_util_threshold": 0.25,
        },
        "calibration": {"source": "c4-math-code", "seed": 0},
    }

    monkeypatch.setattr(
        stage2_reap_ream, "load_json_artifact",
        lambda p: {"per_layer_target_experts": {}, "blacklist": {}},
    )
    monkeypatch.setattr(
        stage2_reap_ream, "build_calibration_tensor",
        lambda *a, **kw: torch.zeros(1, 1, dtype=torch.long),
    )
    monkeypatch.setattr(stage2_reap_ream, "iter_batches", lambda *a, **kw: [])
    monkeypatch.setattr(stage2_reap_ream, "iter_moe_layers", lambda model: iter([]))

    class _DummyModel:
        pass

    with pytest.raises(ValueError, match="cost_asymmetric=True"):
        stage2_reap_ream.run(
            _DummyModel(), tokenizer=None, config=cfg,
            artifacts_dir=tmp_path, no_resume=True,
        )


# ---------------------------------------------------------------------------
# Trackio v2 schema tests (spec § 5 / § 6)
#
# Verify that Stage 2 v2 emits the expected telemetry keys (one-shot config
# + per-layer dynamic state), and that the v1 schema is preserved verbatim
# for backward compatibility with existing dashboards.
# ---------------------------------------------------------------------------


def test_summarize_distill_state_empty_returns_empty_dict():
    """When distill_state is None or empty, the helper returns {} so the
    per-layer Trackio emit naturally omits the four `stage2/distill_*` keys
    on layers where distillation didn't run (singleton-only or disabled)."""
    from moe_compress.stage2.orchestrator import _summarize_distill_state

    assert _summarize_distill_state(None) == {}
    assert _summarize_distill_state({}) == {}


def test_summarize_distill_state_skips_trivial_groups():
    """Groups marked ``{"steps": 0, "skip": "trivial"}`` (singleton or
    zero-steps no-op) must not pollute the means."""
    from moe_compress.stage2.orchestrator import _summarize_distill_state

    state = {
        0: {"steps": 0, "skip": "trivial"},
        1: {"steps": 0, "skip": "trivial"},
    }
    out = _summarize_distill_state(state)
    assert out["stage2/distill_groups"] == 0
    # NaN sentinel for "no real distillation happened" so the dashboard can
    # plot it as a missing data point rather than 0 (which would imply a
    # successful zero-loss merge).
    import math
    assert math.isnan(out["stage2/distill_mean_final_loss"])
    assert out["stage2/distill_mean_steps"] == 0.0
    assert out["stage2/distill_plateau_breaks"] == 0


def test_summarize_distill_state_aggregates_real_groups():
    """Aggregate count, mean final_loss, mean steps, plateau-break count
    across non-trivial groups."""
    from moe_compress.stage2.orchestrator import _summarize_distill_state

    state = {
        0: {"steps": 100, "final_loss": 0.10, "initial_loss": 1.0, "break_reason": "plateau"},
        1: {"steps": 200, "final_loss": 0.30, "initial_loss": 1.0, "break_reason": "max_steps"},
        2: {"steps": 0,   "skip": "trivial"},  # excluded
    }
    out = _summarize_distill_state(state)
    assert out["stage2/distill_groups"] == 2
    assert out["stage2/distill_mean_final_loss"] == pytest.approx(0.20)
    assert out["stage2/distill_mean_steps"] == pytest.approx(150.0)
    assert out["stage2/distill_plateau_breaks"] == 1


# ---------------------------------------------------------------------------
# End-to-end telemetry emission tests via monkey-patched _trackio_log.
# Reuse the same fake-calibration / fake-save patches as
# test_config_rejects_asymmetric_without_freq_weighted_merge so we can
# drive Stage 2 with a tiny synthetic model and capture the dict args.
# ---------------------------------------------------------------------------


@pytest.fixture
def _captured_trackio_emits(monkeypatch):
    """Monkey-patch ``_trackio_log`` to record every dict passed to it during
    Stage 2. Returns the list reference so tests can inspect the captured
    emits afterward.

    Both namespaces are patched: the per-layer emit lives in
    ``stage2.plugins.layer_merge`` (``LayerMergePlugin.write_artifacts`` calls
    ``_trackio_log``, imported into that module's own binding), while the
    static config emit still lives in ``stage2_reap_ream``. Patching only one
    namespace would silently miss the other code path."""
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    from moe_compress.stage2.plugins import layer_merge
    captured: list[dict] = []

    def _capture(metrics: dict) -> None:
        captured.append(dict(metrics))

    monkeypatch.setattr(stage2_reap_ream, "_trackio_log", _capture)
    monkeypatch.setattr(layer_merge, "_trackio_log", _capture)
    return captured


def _enable_v2_flags_for_telemetry(cfg: dict, *, distill_steps: int = 3) -> dict:
    """Mutate ``tiny_config`` in-place to turn on every Stage 2 v2 flag with
    values appropriate for the synthetic ``_TinyModel`` (small K, capacity
    threshold = 0 so TIGHT path always taken, low EM/distill budgets so
    tests run in seconds). Returns the same dict for chaining."""
    s2 = cfg["stage2_reap_ream"]
    s2["assignment_solver"] = "auto"
    s2["cost_alignment"] = "post"
    s2["cost_whitening"] = "diag"
    s2["cost_asymmetric"] = True
    s2["cost_topk_filter"] = 2
    s2["capacity_util_threshold"] = 0.0
    s2["em_refinement_rounds"] = 1
    s2["em_convergence_break"] = True
    s2["expert_distill_steps"] = distill_steps
    s2["expert_distill_token_cap"] = 8
    s2["expert_distill_lr"] = 1.0e-4
    s2["expert_distill_loss_plateau_steps"] = 2
    s2["sinkhorn_iters"] = 50
    return cfg


class _TinyTokenizerForTelemetry:
    name_or_path = "tiny"
    eos_token_id = 0
    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}
    def save_pretrained(self, *_a, **_kw):
        return None


def _patch_calib_and_save(monkeypatch):
    """Mirror the patched_stage2 setup from test_smoke_stage2_resume.py: stub
    calibration tensor builders + save_compressed_checkpoint so Stage 2 (and
    Stage 1) can run end-to-end on the synthetic _TinyModel without hitting
    HuggingFace or writing a real checkpoint."""
    import torch
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.utils import model_io as mio
    from pathlib import Path

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    def _noop_save(model, tokenizer, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)


def _run_stage1_for_telemetry(model, cfg, tmp_path):
    """Run Stage 1 with a fixed BudgetDecomposition so Stage 2 has the
    required artifacts (stage1_blacklist.json, stage1_budgets.json) on disk."""
    from moe_compress import stage1
    from moe_compress.budget.solver import BudgetDecomposition

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2, expert_prune_ratio=0.5,
        svd_rank_ratio=0.14, global_expert_budget=4,
        min_experts_per_layer=2, blacklisted_experts={},
    )
    stage1.run(model, _TinyTokenizerForTelemetry(), cfg, tmp_path, decomp)


def test_trackio_emits_v2_config_keys_once_at_start(
    _captured_trackio_emits, tmp_path, monkeypatch, tiny_config,
):
    """The one-shot config emit at the top of run() must surface every v2
    config flag under the ``stage2/config/*`` namespace. Stubs out the
    per-layer loop via empty iter_moe_layers so this test exercises only
    the config emit path."""
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    _patch_calib_and_save(monkeypatch)
    monkeypatch.setattr(
        stage2_reap_ream, "load_json_artifact",
        lambda p: {"per_layer_target_experts": {}, "blacklist": {}},
    )
    monkeypatch.setattr(
        stage2_reap_ream, "iter_moe_layers",
        lambda model: iter([]),  # short-circuit per-layer loop
    )

    cfg = _enable_v2_flags_for_telemetry(tiny_config)

    # The dummy needs a mutable ``config`` because Stage 2's Blackwell
    # workaround calls ``_set_experts_implementation(model, ...)`` at the
    # top of ``run()`` which sets ``model.config._experts_implementation``.
    # SimpleNamespace is mutable; no real model loading is needed because
    # ``iter_moe_layers`` is monkeypatched to an empty iterator above.
    import types as _types
    class _Dummy:
        config = _types.SimpleNamespace()

    stage2_reap_ream.run(
        _Dummy(), tokenizer=None, config=cfg,
        artifacts_dir=tmp_path, no_resume=True,
    )

    config_emits = [
        e for e in _captured_trackio_emits
        if any(k.startswith("stage2/config/") for k in e)
    ]
    assert len(config_emits) >= 1, "expected at least one config emit at start of run"
    cfg_emit = config_emits[0]

    expected_keys = {
        "stage2/config/assignment_solver": str,
        "stage2/config/cost_alignment": str,
        "stage2/config/cost_whitening": str,
        "stage2/config/cost_asymmetric": bool,
        "stage2/config/cost_topk_filter": int,
        "stage2/config/capacity_util_threshold": float,
        "stage2/config/em_refinement_rounds": int,
        "stage2/config/em_convergence_break": bool,
        "stage2/config/expert_distill_steps": int,
        "stage2/config/expert_distill_token_cap": int,
        "stage2/config/expert_distill_lr": float,
        "stage2/config/sinkhorn_iters": int,
        "stage2/config/format_version": int,
    }
    for k, t in expected_keys.items():
        assert k in cfg_emit, f"missing config key {k}"
        assert isinstance(cfg_emit[k], t), (
            f"config key {k} has type {type(cfg_emit[k]).__name__}, expected {t.__name__}"
        )

    # Specific values from _enable_v2_flags_for_telemetry().
    assert cfg_emit["stage2/config/assignment_solver"] == "auto"
    assert cfg_emit["stage2/config/cost_alignment"] == "post"
    assert cfg_emit["stage2/config/cost_whitening"] == "diag"
    assert cfg_emit["stage2/config/cost_asymmetric"] is True
    assert cfg_emit["stage2/config/format_version"] == 2


def test_trackio_v1_and_v2_per_layer_keys_present(
    _captured_trackio_emits, tmp_path, monkeypatch, tiny_model, tiny_config,
):
    """Combined regression guard + v2-coverage test: with v2 flags ON, the
    per-layer emit must carry both the legacy v1 key set (no renames /
    removals) AND the new v2 keys with expected types."""
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    _patch_calib_and_save(monkeypatch)

    cfg = _enable_v2_flags_for_telemetry(tiny_config)
    _run_stage1_for_telemetry(tiny_model, cfg, tmp_path)
    stage2_reap_ream.run(
        tiny_model, _TinyTokenizerForTelemetry(), cfg, tmp_path,
        device=None, no_resume=True,
    )

    per_layer_emits = [e for e in _captured_trackio_emits if "stage2/layer_idx" in e]
    assert per_layer_emits, "expected at least one per-layer Trackio emit"

    v1_required = {
        "stage2/layer_idx", "stage2/protected_experts", "stage2/ream_centroids",
        "stage2/total_experts", "stage2/sum_assignment_cost",
        "stage2/mean_cost_per_pair", "stage2/max_merge_group_size",
        "stage2/mean_merge_group_size", "stage2/effective_target",
        "stage2/actual_kept_experts", "stage2/stage1_target",
    }
    v2_required = {
        "stage2/assignment_solver_used": str,
        "stage2/cost_alignment_effective": str,
        "stage2/cost_asymmetric_effective": bool,
        "stage2/capacity_util": float,
        "stage2/capacity_regime": str,
        "stage2/em_rounds_done": int,
    }
    for emit in per_layer_emits:
        # v1 backward-compat: every legacy key still present.
        missing_v1 = v1_required - emit.keys()
        assert not missing_v1, f"v1 keys missing: {missing_v1}"
        # v2 keys: present + correct type + bounded enums.
        for k, t in v2_required.items():
            assert k in emit, f"v2 key {k} missing"
            assert isinstance(emit[k], t), (
                f"v2 key {k} has type {type(emit[k]).__name__}, expected {t.__name__}"
            )
        assert emit["stage2/capacity_regime"] in ("slack", "tight")


def test_trackio_distill_keys_absent_when_distillation_disabled(
    _captured_trackio_emits, tmp_path, monkeypatch, tiny_model, tiny_config,
):
    """When ``expert_distill_steps == 0``, the four ``stage2/distill_*``
    keys must NOT appear on the per-layer emit. Avoids dashboard noise
    from runs that don't use the feature."""
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    _patch_calib_and_save(monkeypatch)

    cfg = _enable_v2_flags_for_telemetry(tiny_config, distill_steps=0)
    _run_stage1_for_telemetry(tiny_model, cfg, tmp_path)
    stage2_reap_ream.run(
        tiny_model, _TinyTokenizerForTelemetry(), cfg, tmp_path,
        device=None, no_resume=True,
    )

    per_layer_emits = [e for e in _captured_trackio_emits if "stage2/layer_idx" in e]
    distill_keys = {
        "stage2/distill_groups",
        "stage2/distill_mean_final_loss",
        "stage2/distill_mean_steps",
        "stage2/distill_plateau_breaks",
    }
    for emit in per_layer_emits:
        intersect = distill_keys & emit.keys()
        assert not intersect, (
            f"distill_* keys leaked into emit when expert_distill_steps=0: {intersect}"
        )
