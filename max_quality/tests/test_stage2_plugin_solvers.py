"""Task 13 — assignment-solver plugin extraction tests.

Covers the structural T13 contract (plugin is_enabled selectors, registry
dispatch equivalence). Deep algorithm coverage stays in
test_stage2_assignment_v2.py — this file does NOT re-test solver internals.
"""
from __future__ import annotations

import numpy as np
import pytest

from moe_compress.stage2 import orchestrator as s2
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


# --- S2-8: live solve_assignment slot -------------------------------------
import pathlib  # noqa: E402

from moe_compress.pipeline.context import PipelineContext  # noqa: E402
from moe_compress.pipeline.registry import PluginRegistry  # noqa: E402


def _solver_ctx(n_nc: int, n_c: int) -> PipelineContext:
    """Build a per-layer ctx with the two _iter_* scratch slots the live
    solve_assignment slot reads."""
    ctx = PipelineContext()
    ctx.set("_iter_n_ream_nc", n_nc)
    ctx.set("_iter_n_ream_c", n_c)
    return ctx


def test_greedy_plugin_solve_assignment_byte_identical():
    """GreedySolverPlugin.solve_assignment produces the same assignment as a
    direct _assign_children_to_centroids call."""
    cost = _cost()  # 4 children, 3 centroids
    plugin = solver_greedy.GreedySolverPlugin(
        max_group_cap=2, assignment_solver="greedy",
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.01,
        sinkhorn_iters=200,
    )
    ctx = _solver_ctx(n_nc=4, n_c=3)
    via_plugin = plugin.solve_assignment(ctx, cost)
    direct = s2._assign_children_to_centroids(
        cost, 4, 3, 2, solver="greedy",
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.01,
        sinkhorn_iters=200,
    )
    assert via_plugin == direct
    assert via_plugin == solver_greedy._assign_greedy(cost, 4, 3, 2)


def test_hungarian_plugin_solve_assignment_byte_identical():
    """HungarianSolverPlugin.solve_assignment == direct dispatcher call."""
    rng = np.random.default_rng(7)
    cost = rng.random((3, 4)).astype(np.float64)  # n_nc <= n_c
    plugin = solver_hungarian.HungarianSolverPlugin(
        max_group_cap=1, assignment_solver="hungarian",
    )
    ctx = _solver_ctx(n_nc=3, n_c=4)
    via_plugin = plugin.solve_assignment(ctx, cost)
    direct = s2._assign_children_to_centroids(
        cost, 3, 4, 1, solver="hungarian",
    )
    assert via_plugin == direct


def test_sinkhorn_plugin_solve_assignment_byte_identical():
    """SinkhornSolverPlugin.solve_assignment == direct dispatcher call,
    threading the sinkhorn_* knobs from the plugin ctor."""
    cost = _cost()
    plugin = solver_sinkhorn.SinkhornSolverPlugin(
        max_group_cap=2, assignment_solver="sinkhorn",
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.01,
        sinkhorn_iters=50,
    )
    ctx = _solver_ctx(n_nc=4, n_c=3)
    via_plugin = plugin.solve_assignment(ctx, cost)
    direct = s2._assign_children_to_centroids(
        cost, 4, 3, 2, solver="sinkhorn",
        sinkhorn_epsilon_init=1.0, sinkhorn_epsilon_final=0.01,
        sinkhorn_iters=50,
    )
    assert via_plugin == direct


def test_mcf_plugin_solve_assignment_byte_identical():
    """McfSolverPlugin.solve_assignment == direct dispatcher call (ortools)."""
    pytest.importorskip("ortools")
    cost = _cost()
    plugin = solver_mcf.McfSolverPlugin(
        max_group_cap=2, assignment_solver="mcf",
    )
    ctx = _solver_ctx(n_nc=4, n_c=3)
    via_plugin = plugin.solve_assignment(ctx, cost)
    direct = s2._assign_children_to_centroids(cost, 4, 3, 2, solver="mcf")
    assert via_plugin == direct


def test_auto_plugin_solve_assignment_byte_identical():
    """AutoSolverPlugin.solve_assignment == direct dispatcher call. The auto
    rule routes a capacitated problem to mcf -> needs ortools."""
    pytest.importorskip("ortools")
    cost = _cost()  # 4 children > 3 centroids -> auto picks mcf
    plugin = solver_auto.AutoSolverPlugin(
        max_group_cap=2, assignment_solver="auto",
    )
    ctx = _solver_ctx(n_nc=4, n_c=3)
    via_plugin = plugin.solve_assignment(ctx, cost)
    direct = s2._assign_children_to_centroids(cost, 4, 3, 2, solver="auto")
    assert via_plugin == direct


# --- S2-8: PluginRegistry wiring ------------------------------------------

class _AlwaysOnPlugin:
    """Minimal always-enabled plugin standing in for the LayerMergePlugin."""

    name = "always_on_adapter_stub"

    def is_enabled(self, config: dict) -> bool:
        return True


_SOLVER_PLUGIN_CLASSES = [
    solver_greedy.GreedySolverPlugin,
    solver_hungarian.HungarianSolverPlugin,
    solver_mcf.McfSolverPlugin,
    solver_sinkhorn.SinkhornSolverPlugin,
    solver_auto.AutoSolverPlugin,
]


@pytest.mark.parametrize(
    "solver_name",
    ["greedy", "hungarian", "mcf", "sinkhorn", "auto"],
)
def test_registry_wiring_one_solver_enabled_before_adapter(solver_name):
    """Exactly one solver plugin is enabled per assignment_solver, and it is
    ordered before the (always-on) adapter stand-in."""
    kwargs = dict(assignment_solver=solver_name)
    plugins = [cls(**kwargs) for cls in _SOLVER_PLUGIN_CLASSES]
    adapter_stub = _AlwaysOnPlugin()
    registry = PluginRegistry(plugins + [adapter_stub])

    enabled = registry.enabled(
        {"stage2_reap_ream": {"assignment_solver": solver_name}}
    )
    solver_plugins = [p for p in enabled if p is not adapter_stub]
    assert len(solver_plugins) == 1, (
        f"expected exactly one solver plugin enabled for {solver_name!r}, "
        f"got {[p.name for p in solver_plugins]}"
    )
    assert adapter_stub in enabled
    # Ordered before the adapter so it wins the solve_assignment dispatch_first.
    assert enabled.index(solver_plugins[0]) < enabled.index(adapter_stub)


def test_orchestrator_registers_solvers_after_skip_merge_before_adapter():
    """The orchestrator source registers the five solver plugins after
    ``SkipMergeFloorPlugin`` and before the merge spine in the registry list.
    S2-12: the merge-spine entry is the ``layer_merge`` (``LayerMergePlugin``)
    instance that replaced the retired ``LegacyAdapter``."""
    src = (
        pathlib.Path(__file__).parents[1]
        / "src/moe_compress/stage2/orchestrator.py"
    ).read_text()
    smf = src.index("SkipMergeFloorPlugin(skip_merge_percentile=")
    adapter = src.index("\n        layer_merge,\n")
    for name in (
        "GreedySolverPlugin(**_solver_plugin_kwargs)",
        "HungarianSolverPlugin(**_solver_plugin_kwargs)",
        "McfSolverPlugin(**_solver_plugin_kwargs)",
        "SinkhornSolverPlugin(**_solver_plugin_kwargs)",
        "AutoSolverPlugin(**_solver_plugin_kwargs)",
    ):
        pos = src.index(name)
        assert smf < pos < adapter, f"{name} not wired between SMF and adapter"
