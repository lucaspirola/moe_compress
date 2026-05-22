"""Task 13 — assignment-solver plugin extraction tests.

Covers the structural T13 contract (plugin is_enabled selectors, registry
dispatch equivalence). Deep algorithm coverage stays in
test_stage2_assignment_v2.py — this file does NOT re-test solver internals.
"""
from __future__ import annotations

import numpy as np
import pytest

import moe_compress.stage2_reap_ream as s2
from moe_compress.stage2.plugins import (
    solver_auto, solver_dispatch, solver_greedy,
    solver_hungarian, solver_mcf, solver_sinkhorn,
)


# --- solver_mcf importable without ortools --------------------------------
def test_solver_mcf_module_imports_without_ortools():
    # The module imported fine at file top; assert the symbols exist. The
    # ortools import is function-scope, so module import never needs ortools.
    assert callable(solver_mcf._assign_mcf)
    assert solver_mcf.McfSolverPlugin.name == "solver_mcf"


# --- plugin is_enabled selectors ------------------------------------------
@pytest.mark.parametrize(
    "plugin_cls,solver_name",
    [
        (solver_greedy.GreedySolverPlugin, "greedy"),
        (solver_hungarian.HungarianSolverPlugin, "hungarian"),
        (solver_mcf.McfSolverPlugin, "mcf"),
        (solver_sinkhorn.SinkhornSolverPlugin, "sinkhorn"),
        (solver_auto.AutoSolverPlugin, "auto"),
    ],
)
def test_is_enabled_matches_only_its_solver(plugin_cls, solver_name):
    for candidate in ["greedy", "hungarian", "mcf", "sinkhorn", "auto"]:
        cfg = {"stage2_reap_ream": {"assignment_solver": candidate}}
        assert plugin_cls().is_enabled(cfg) is (candidate == solver_name)


def test_is_enabled_missing_key_defaults_to_greedy():
    # assignment_solver defaults to "greedy" -> only GreedySolverPlugin on.
    empty = {"stage2_reap_ream": {}}
    assert solver_greedy.GreedySolverPlugin().is_enabled(empty) is True
    for cls in (solver_hungarian.HungarianSolverPlugin,
                solver_mcf.McfSolverPlugin,
                solver_sinkhorn.SinkhornSolverPlugin,
                solver_auto.AutoSolverPlugin):
        assert cls().is_enabled(empty) is False


# --- dispatcher == direct solver call (greedy/hungarian/sinkhorn) ---------
def _cost():
    # 4 children, 3 centroids, deterministic finite cost matrix.
    rng = np.random.default_rng(13)
    return rng.random((4, 3)).astype(np.float64)


def test_dispatch_greedy_equals_direct():
    cost = _cost()
    assert (s2._assign_children_to_centroids(cost, 4, 3, 2, solver="greedy")
            == solver_greedy._assign_greedy(cost, 4, 3, 2))


def test_dispatch_hungarian_equals_direct():
    # n_children <= n_centroids so hungarian is not routed to mcf.
    rng = np.random.default_rng(7)
    cost = rng.random((3, 4)).astype(np.float64)
    assert (s2._assign_children_to_centroids(cost, 3, 4, 1, solver="hungarian")
            == solver_hungarian._assign_hungarian(cost, 3, 4, 1))


def test_dispatch_sinkhorn_equals_direct():
    cost = _cost()
    via_dispatch = s2._assign_children_to_centroids(
        cost, 4, 3, 2, solver="sinkhorn",
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.01,
        sinkhorn_iters=50,
    )
    direct = solver_sinkhorn._assign_sinkhorn(
        cost, 4, 3, 2, epsilon_init=1.0, epsilon_final=0.01, iters=50,
    )
    assert via_dispatch == direct


def test_dispatch_mcf_equals_direct_if_ortools():
    pytest.importorskip("ortools")
    cost = _cost()
    assert (s2._assign_children_to_centroids(cost, 4, 3, 2, solver="mcf")
            == solver_mcf._assign_mcf(cost, 4, 3, 2))


# --- registry shape -------------------------------------------------------
def test_solvers_registry_keys():
    assert set(solver_dispatch.SOLVERS) == {
        "greedy", "hungarian", "mcf", "auto", "sinkhorn"}


def test_unknown_solver_still_raises():
    cost = _cost()
    with pytest.raises(ValueError, match="unknown solver"):
        s2._assign_children_to_centroids(cost, 4, 3, 2, solver="nope")  # type: ignore[arg-type]
