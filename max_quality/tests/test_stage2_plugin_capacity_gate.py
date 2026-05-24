"""Task 11 / S2-10 — capacity-utilization gate plugin module.

Pins the `CapacityGatePlugin` contract, the gate decision logic, the live
``select_alignment`` slot, and a monkeypatch-drift guard (T9/T10 lesson).
"""
from __future__ import annotations

import pathlib

import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage2.plugins.capacity_gate import (
    CapacityGatePlugin,
    _pick_effective_alignment,
)


# Representative gate knobs — the same shape the orchestrator passes (the four
# parsed ``run()`` locals the gate stores).
def _gate_kwargs(*, cost_alignment_cfg="pre", max_group_cap=8,
                 capacity_util_threshold=0.25, cost_asymmetric=False):
    return dict(
        max_group_cap=max_group_cap,
        capacity_util_threshold=capacity_util_threshold,
        cost_alignment_cfg=cost_alignment_cfg,
        cost_asymmetric=cost_asymmetric,
    )


def test_ream_cost_plugin_imports_from_capacity_gate():
    """``CapacityGatePlugin.select_alignment`` resolves the gate decision from
    the sibling ``capacity_gate`` module — not the monolith — so monkeypatching
    ``stage2_reap_ream`` will not silently no-op the gate.

    (Pre-S2-5 this guarded the monolithic ``compute_assignment`` hook; S2-5
    decomposed that, S2-6 moved the cost path into
    ``ream_cost._compute_cost_for_plugin``, and S2-10 moved the gate itself
    into ``CapacityGatePlugin.select_alignment``.)"""
    import inspect
    src = inspect.getsource(CapacityGatePlugin.select_alignment)
    assert "_pick_effective_alignment(" in src
    assert "from ...stage2_reap_ream import _pick_effective_alignment" not in src


# --- CapacityGatePlugin contract -------------------------------------------

def test_plugin_conforms_to_pipeline_plugin():
    assert isinstance(CapacityGatePlugin(**_gate_kwargs()), PipelinePlugin)


def test_plugin_name():
    assert CapacityGatePlugin.name == "capacity_gate"


def test_plugin_always_enabled():
    """The gate always runs → is_enabled is True for every config shape."""
    assert CapacityGatePlugin(**_gate_kwargs()).is_enabled({}) is True
    assert CapacityGatePlugin(**_gate_kwargs()).is_enabled(
        {"stage2_reap_ream": {}}) is True
    assert CapacityGatePlugin(**_gate_kwargs()).is_enabled(
        {"stage2_reap_ream": {"cost_alignment": "post"}}
    ) is True


def test_select_alignment_writes_gate_slots_and_returns_winner():
    """S2-10: ``select_alignment`` is the live gate slot — it writes the three
    gate slots to ctx and returns a non-None winner so ``dispatch_first``
    registers it.

    ``max_group_cap=8``, ``n_nc=12``, ``n_c=2`` → u = 12/(2*8) = 0.75 ≥ 0.25
    threshold → TIGHT → the configured ``post`` wins.
    """
    ctx = PipelineContext()
    ctx.set("_iter_n_ream_c", 2)
    ctx.set("_iter_n_ream_nc", 12)

    plugin = CapacityGatePlugin(**_gate_kwargs(
        cost_alignment_cfg="post", cost_asymmetric=True))
    result = plugin.select_alignment(ctx)

    # Non-None winner so PluginRegistry.dispatch_first registers this plugin.
    assert result is not None
    assert result == "post"
    # The three gate slots were published to ctx.
    assert ctx.get("effective_cost_alignment") == "post"
    assert ctx.get("effective_cost_asymmetric") is True
    assert ctx.get("capacity_util_value") == 12 / (2 * 8)


def test_select_alignment_slack_downgrades_to_pre():
    """A slack-capacity layer (u < threshold) downgrades the configured
    ``post`` → ``pre``; ``effective_cost_asymmetric`` follows to False."""
    ctx = PipelineContext()
    ctx.set("_iter_n_ream_c", 4)
    ctx.set("_iter_n_ream_nc", 2)  # u = 2/(4*8) = 0.0625 < 0.25 → SLACK

    plugin = CapacityGatePlugin(**_gate_kwargs(
        cost_alignment_cfg="post", cost_asymmetric=True))
    result = plugin.select_alignment(ctx)

    assert result == "pre"
    assert ctx.get("effective_cost_alignment") == "pre"
    assert ctx.get("effective_cost_asymmetric") is False
    assert ctx.get("capacity_util_value") == 2 / (4 * 8)


# --- _pick_effective_alignment decision logic ------------------------------

def test_slack_below_threshold_returns_pre():
    """u = 2/(4*8) = 0.0625 < 0.25 → SLACK → 'pre' despite configured 'post'."""
    assert _pick_effective_alignment(
        n_nc=2, n_c=4, max_group_cap=8, threshold=0.25, configured="post",
    ) == "pre"


def test_tight_above_threshold_returns_configured_post():
    """u = 12/(2*8) = 0.75 >= 0.25 → TIGHT → configured 'post' wins."""
    assert _pick_effective_alignment(
        n_nc=12, n_c=2, max_group_cap=8, threshold=0.25, configured="post",
    ) == "post"


def test_tight_above_threshold_returns_configured_output():
    """Direction C 'output' is gated identically to 'post' (TIGHT → passthrough)."""
    assert _pick_effective_alignment(
        n_nc=12, n_c=2, max_group_cap=8, threshold=0.25, configured="output",
    ) == "output"


def test_configured_pre_stays_pre_when_tight():
    """A 'pre' config is never upgraded — TIGHT regime still yields 'pre'."""
    assert _pick_effective_alignment(
        n_nc=12, n_c=2, max_group_cap=8, threshold=0.25, configured="pre",
    ) == "pre"


def test_uncapped_max_group_cap_zero_treated_as_slack():
    """max_group_cap == 0 (uncapped ablation path) → util forced to 0 → SLACK."""
    assert _pick_effective_alignment(
        n_nc=100, n_c=10, max_group_cap=0, threshold=0.25, configured="post",
    ) == "pre"


def test_negative_max_group_cap_treated_as_slack():
    """max_group_cap <= 0 branch: a negative cap also forces util=0 → SLACK."""
    assert _pick_effective_alignment(
        n_nc=100, n_c=10, max_group_cap=-1, threshold=0.25, configured="post",
    ) == "pre"


def test_n_c_zero_capacity_floored_to_one():
    """n_c == 0 with cap > 0 → capacity = max(0, 1) = 1 → util = n_nc.
    With n_nc=5 > threshold the gate stays TIGHT and returns the configured
    value; the max(..., 1) floor prevents a ZeroDivisionError."""
    assert _pick_effective_alignment(
        n_nc=5, n_c=0, max_group_cap=8, threshold=0.25, configured="post",
    ) == "post"


def test_util_exactly_at_threshold_is_tight():
    """u == threshold is NOT below threshold → TIGHT (configured wins).
    u = 4/(2*8) = 0.25 exactly == 0.25 threshold."""
    assert _pick_effective_alignment(
        n_nc=4, n_c=2, max_group_cap=8, threshold=0.25, configured="post",
    ) == "post"


def test_zero_threshold_always_tight():
    """threshold == 0.0 → util < 0 is impossible → always configured value."""
    assert _pick_effective_alignment(
        n_nc=0, n_c=4, max_group_cap=8, threshold=0.0, configured="post",
    ) == "post"


# --- monkeypatch-drift guard (T9/T10 lesson) -------------------------------

def test_no_monolith_only_monkeypatch_of_moved_symbol():
    """No test may patch `_pick_effective_alignment` via the monolith namespace
    only — after T11 such a patch silently no-ops for the live plugin path
    (which now imports from `capacity_gate`). Investigation found none; this
    guard fails loudly if a future edit introduces one without a matching
    capacity_gate patch."""
    tests_dir = pathlib.Path(__file__).parent
    needle = 'setattr(stage2_reap_ream, "_pick_effective_alignment"'
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        if needle in text and "capacity_gate" not in text:
            offenders.append(f"{path.name}: patches _pick_effective_alignment "
                             "on monolith only")
    assert not offenders, (
        "monolith-only monkeypatch of the moved gate symbol — add a matching "
        "capacity_gate patch (T9 dual-patch lesson):\n" + "\n".join(offenders)
    )
