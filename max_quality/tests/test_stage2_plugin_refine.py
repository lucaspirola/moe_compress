"""S2-9 — the refine_assignment CHAIN (TwoOptRefinePlugin then EmRefinePlugin).

``refine_assignment`` is a CHAIN, not a single-winner ``dispatch_first`` slot:
both refiners may run, in registry order (two-opt first, then EM). A chain
link declining the slot (returning None) is skipped cleanly.

Coverage:
  * plugin contract — both plugins satisfy ``PipelinePlugin``;
  * constructors — the new ``__init__`` kwarg sets;
  * ``TwoOptRefinePlugin.refine_assignment`` byte-identity (greedy →
    ``_two_opt_refine`` result; non-greedy → unchanged asg + the ``elif``
    warning);
  * ``EmRefinePlugin.refine_assignment`` byte-identity vs a direct
    ``_em_refine_assignment`` call;
  * chain-order — a probe records two-opt before EM, ``em_rounds_done`` is
    threaded from EM's info dict;
  * neither-enabled no-op;
  * registry wiring — two-opt before EM before the adapter.

Deep algorithm coverage stays in ``test_stage2_two_opt.py`` /
``test_stage2_assignment_v2.py`` — this file tests the *chain wiring* only.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage2.plugins.em_refine import (
    EmRefinePlugin,
    _em_refine_assignment,
)
from moe_compress.stage2.plugins.two_opt_refine import (
    TwoOptRefinePlugin,
    _two_opt_refine,
)


# ---------------------------------------------------------------------------
# plugin contract + constructors
# ---------------------------------------------------------------------------
def test_two_opt_plugin_conforms_to_pipeline_plugin():
    assert isinstance(TwoOptRefinePlugin(), PipelinePlugin)


def test_em_plugin_conforms_to_pipeline_plugin():
    assert isinstance(EmRefinePlugin(), PipelinePlugin)


def test_two_opt_constructor_stores_knobs():
    p = TwoOptRefinePlugin(
        two_opt_refine=True, assignment_solver="hungarian", max_group_cap=4,
    )
    assert p.two_opt_refine is True
    assert p.assignment_solver == "hungarian"
    assert p.max_group_cap == 4


def test_two_opt_constructor_defaults():
    p = TwoOptRefinePlugin()
    assert p.two_opt_refine is False
    assert p.assignment_solver == "greedy"
    assert p.max_group_cap == 0


def test_two_opt_reads_no_ctx_slots():
    """Two-opt operates purely on call args — it declares no ctx reads."""
    assert TwoOptRefinePlugin.reads == ()


def test_em_constructor_stores_knobs():
    sentinel = object()
    p = EmRefinePlugin(
        em_refinement_rounds=3,
        em_convergence_break=False,
        max_group_cap=8,
        assignment_solver="mcf",
        cost_whitening="diag",
        cost_asymmetric=True,
        cost_topk_filter=16,
        skip_merge_percentile=90.0,
        cov_acc=sentinel,
        sinkhorn_epsilon_init=2.0,
        sinkhorn_epsilon_final=0.02,
        sinkhorn_iters=50,
    )
    assert p.em_refinement_rounds == 3
    assert p.em_convergence_break is False
    assert p.max_group_cap == 8
    assert p.assignment_solver == "mcf"
    assert p.cost_whitening == "diag"
    assert p.cost_asymmetric is True
    assert p.cost_topk_filter == 16
    assert p.skip_merge_percentile == 90.0
    assert p.cov_acc is sentinel
    assert p.sinkhorn_epsilon_init == 2.0
    assert p.sinkhorn_epsilon_final == 0.02
    assert p.sinkhorn_iters == 50


def test_em_constructor_defaults():
    p = EmRefinePlugin()
    assert p.em_refinement_rounds == 0
    assert p.em_convergence_break is True
    assert p.cov_acc is None


# ---------------------------------------------------------------------------
# TwoOptRefinePlugin.refine_assignment byte-identity
# ---------------------------------------------------------------------------
def _two_opt_test_inputs():
    """A small assignment + cost matrix where a 2-opt swap strictly improves."""
    # 2 children, 2 centroids; the initial assignment [0, 0] over-fills
    # centroid 0; swapping child 1 to centroid 1 lowers cost.
    asg = [0, 1]
    cost = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float64)
    return asg, cost


def test_two_opt_refine_greedy_byte_identical():
    """Greedy + flag set → refine_assignment returns the exact _two_opt_refine
    result; info dict is {"two_opt": True} with NO em_rounds key."""
    asg, cost = _two_opt_test_inputs()
    plugin = TwoOptRefinePlugin(
        two_opt_refine=True, assignment_solver="greedy", max_group_cap=2,
    )
    expected = _two_opt_refine(asg, cost, 2)

    out_asg, out_delta, info = plugin.refine_assignment(None, list(asg), cost)
    assert out_asg == expected
    assert out_delta is cost  # delta passes through unchanged
    assert info == {"two_opt": True}
    assert "em_rounds" not in info


class _ListHandler(logging.Handler):
    """Minimal log sink — collects records into a list. Used instead of the
    ``caplog`` fixture, which does not capture in this repo's test env."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def test_two_opt_refine_non_greedy_warns_and_passes_through():
    """Non-greedy solver + flag set → asg is returned UNCHANGED and the elif
    warning fires."""
    asg, cost = _two_opt_test_inputs()
    plugin = TwoOptRefinePlugin(
        two_opt_refine=True, assignment_solver="hungarian", max_group_cap=2,
    )
    plugin_log = logging.getLogger("moe_compress.stage2.plugins.two_opt_refine")
    handler = _ListHandler()
    handler.setLevel(logging.WARNING)
    plugin_log.addHandler(handler)
    try:
        out_asg, out_delta, info = plugin.refine_assignment(None, list(asg), cost)
    finally:
        plugin_log.removeHandler(handler)
    assert out_asg == asg  # unchanged — 2-opt did not run
    assert out_delta is cost
    assert info == {"two_opt": True}
    messages = [rec.getMessage() for rec in handler.records]
    assert any(
        "two_opt_refine=true is ignored" in m
        and "assignment_solver='hungarian'" in m
        for m in messages
    ), f"expected the elif warning, got: {messages}"


def test_two_opt_refine_flag_off_is_passthrough():
    """Flag off → asg returned verbatim, no warning, info still {"two_opt": True}."""
    asg, cost = _two_opt_test_inputs()
    plugin = TwoOptRefinePlugin(two_opt_refine=False, assignment_solver="greedy")
    out_asg, out_delta, info = plugin.refine_assignment(None, list(asg), cost)
    assert out_asg == asg
    assert out_delta is cost
    assert info == {"two_opt": True}


# ---------------------------------------------------------------------------
# EmRefinePlugin.refine_assignment byte-identity
# ---------------------------------------------------------------------------
def _make_em_layer_ctx(*, cost_alignment="pre"):
    """Build a layer ctx carrying every slot EmRefinePlugin.refine_assignment
    reads. cost_alignment="pre" makes _em_refine_assignment a guaranteed no-op
    (the cheap symmetric cost is centroid-independent) so the byte-identity
    check needs no real model — both the plugin and the direct call return the
    input unchanged."""
    ctx = PipelineContext()
    ctx.set("layer_ref", object())
    ctx.set("ream_acc", object())
    ctx.set("perm_cache", object())
    ctx.set("freq", {0: 1, 1: 1, 2: 1})
    ctx.set("protected", ())
    ctx.set("_iter_ream_centroid_ids", (0, 1))
    ctx.set("_iter_ream_noncentroid_ids", (2,))
    ctx.set("effective_cost_alignment", cost_alignment)
    ctx.set("effective_cost_asymmetric", False)
    return ctx


def test_em_refine_byte_identical_vs_direct_call():
    """EmRefinePlugin.refine_assignment threads exactly the args
    ``_em_refine_assignment`` expects: its output equals a direct
    ``_em_refine_assignment`` call.

    Uses cost_alignment="pre" so EM is a no-op (no model needed) — the path
    under test is the ctx-slot read + arg threading, not EM's internals."""
    ctx = _make_em_layer_ctx(cost_alignment="pre")
    asg = [0, 1]
    delta = np.array([[0.3, 0.7]], dtype=np.float64)

    plugin = EmRefinePlugin(
        em_refinement_rounds=2,
        em_convergence_break=True,
        max_group_cap=2,
        assignment_solver="greedy",
        cost_whitening="none",
        cost_asymmetric=False,
        cost_topk_filter=2,
        skip_merge_percentile=100.0,
        cov_acc=None,
        sinkhorn_epsilon_init=1.0,
        sinkhorn_epsilon_final=0.01,
        sinkhorn_iters=10,
    )
    out_asg, out_delta, info = plugin.refine_assignment(ctx, list(asg), delta)

    direct_asg, direct_delta, direct_rounds = _em_refine_assignment(
        ctx.get("layer_ref"),
        initial_assignment=list(asg),
        initial_delta=delta,
        skip_merge_percentile=100.0,
        ream_centroid_ids=[0, 1],
        ream_noncentroid_ids=[2],
        perm_cache=ctx.get("perm_cache"),
        ream_acc=ctx.get("ream_acc"),
        cov_acc=None,
        freq={0: 1, 1: 1, 2: 1},
        max_group_cap=2,
        cost_alignment="pre",
        cost_whitening="none",
        cost_asymmetric=False,
        cost_topk_filter=2,
        assignment_solver="greedy",
        em_rounds=2,
        em_break=True,
        blacklisted_ids=set(),
        sinkhorn_epsilon_init=1.0,
        sinkhorn_epsilon_final=0.01,
        sinkhorn_iters=10,
    )
    assert out_asg == direct_asg
    assert out_delta is direct_delta
    assert info == {"em_rounds": direct_rounds}


def test_em_refine_emits_em_rounds_key():
    """EmRefinePlugin.refine_assignment's info dict carries an em_rounds key
    (the orchestrator reads em_rounds_done out of it)."""
    ctx = _make_em_layer_ctx(cost_alignment="pre")
    plugin = EmRefinePlugin(em_refinement_rounds=0)
    _asg, _delta, info = plugin.refine_assignment(
        ctx, [0], np.array([[0.1, 0.2]], dtype=np.float64),
    )
    assert "em_rounds" in info
    assert info["em_rounds"] == 0  # em_rounds=0 → zero rounds completed


# ---------------------------------------------------------------------------
# chain order + em_rounds threading
# ---------------------------------------------------------------------------
def _run_refine_chain(plugins, ctx, asg, delta):
    """Replica of the orchestrator's S2-9 refine chain loop — calls every
    plugin's refine_assignment in order, threads the result forward, and pulls
    em_rounds out of the info dict. Returns (asg, delta, em_rounds_done)."""
    em_rounds_done = 0
    for p in plugins:
        hook = getattr(p, "refine_assignment", None)
        if not callable(hook):
            continue
        result = hook(ctx, asg, delta)
        if result is None:
            continue
        asg, delta, info = result
        if "em_rounds" in info:
            em_rounds_done = int(info["em_rounds"])
    return asg, delta, em_rounds_done


def test_chain_runs_two_opt_before_em_and_threads_em_rounds():
    """The chain calls two-opt THEN EM; em_rounds_done comes from EM's info."""
    order: list[str] = []

    class _ProbeTwoOpt:
        name = "probe_two_opt"

        def refine_assignment(self, ctx, asg, delta):
            order.append("two_opt")
            return asg, delta, {"two_opt": True}

    class _ProbeEm:
        name = "probe_em"

        def refine_assignment(self, ctx, asg, delta):
            order.append("em")
            return asg, delta, {"em_rounds": 7}

    asg, delta, em_rounds = _run_refine_chain(
        [_ProbeTwoOpt(), _ProbeEm()], None, [0], np.empty((0, 0)),
    )
    assert order == ["two_opt", "em"], "chain must run two-opt before EM"
    # em_rounds_done is taken from EM's info dict, not two-opt's (no key there).
    assert em_rounds == 7


def test_chain_skips_plugins_returning_none():
    """A chain link declining the slot (returns None — e.g. a refiner whose
    gate is off this layer) is skipped cleanly."""

    class _DeclineAll:
        name = "decline"

        def refine_assignment(self, ctx, asg, delta):
            return None

    class _ProbeEm:
        name = "probe_em"

        def refine_assignment(self, ctx, asg, delta):
            return asg, delta, {"em_rounds": 3}

    asg, delta, em_rounds = _run_refine_chain(
        [_DeclineAll(), _ProbeEm(), _DeclineAll()], None, [1], np.empty((0, 0)),
    )
    assert asg == [1]
    assert em_rounds == 3


def test_chain_neither_refiner_enabled_is_noop():
    """When registry.enabled drops both refiners, the refine_assignment chain
    is empty — a stand-in link that declines the slot (returns None) leaves the
    chain a clean no-op and em_rounds_done stays 0."""

    class _RefineDecliner:
        name = "refine_decliner"

        def refine_assignment(self, ctx, asg, delta):
            return None

    asg_in = [0, 1, 0]
    delta_in = np.array([[0.1, 0.2]], dtype=np.float64)
    asg, delta, em_rounds = _run_refine_chain(
        [_RefineDecliner()], None, asg_in, delta_in,
    )
    assert asg == asg_in
    assert delta is delta_in
    assert em_rounds == 0


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------
def test_registry_orders_two_opt_before_em_before_adapter():
    """In the Stage-2 registry the two refiners appear after the solver
    plugins, two-opt FIRST then EM, immediately before the LayerMergePlugin."""
    import inspect

    from moe_compress.stage2 import orchestrator

    src = inspect.getsource(orchestrator.run)
    i_two_opt = src.index("TwoOptRefinePlugin(")
    i_em = src.index("EmRefinePlugin(")
    # S2-12: the merge-spine registry entry is the ``layer_merge`` instance
    # (the ``LayerMergePlugin`` that replaced the retired ``LegacyAdapter``).
    i_adapter = src.index("\n        layer_merge,\n")
    i_auto = src.index("AutoSolverPlugin(**_solver_plugin_kwargs)")
    assert i_auto < i_two_opt < i_em < i_adapter, (
        "registry order must be ...AutoSolverPlugin, TwoOptRefinePlugin, "
        "EmRefinePlugin, layer_merge"
    )
