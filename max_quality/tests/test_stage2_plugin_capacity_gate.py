"""Task 11 — capacity-utilization gate plugin module.

Pins the `CapacityGatePlugin` contract, the gate decision logic, and a
monkeypatch-drift guard (T9/T10 lesson).
"""
from __future__ import annotations

import pathlib

import pytest

from moe_compress.stage2._framework.base import Stage2Plugin
from moe_compress.stage2.plugins.capacity_gate import (
    CapacityGatePlugin,
    _pick_effective_alignment,
)


def test_legacy_adapter_imports_from_capacity_gate():
    """LegacyAdapter resolves the gate from the sibling module, not the monolith."""
    import inspect
    import moe_compress.stage2.plugins.legacy_adapter as legacy_adapter
    src = inspect.getsource(legacy_adapter.LegacyAdapter.compute_assignment)
    assert "from .capacity_gate import _pick_effective_alignment" in src
    assert "_pick_effective_alignment," not in src  # not in the monolith tuple


# --- CapacityGatePlugin contract -------------------------------------------

def test_plugin_is_stage2plugin_subclass():
    assert issubclass(CapacityGatePlugin, Stage2Plugin)


def test_plugin_name():
    assert CapacityGatePlugin.name == "capacity_gate"


def test_plugin_always_enabled():
    """enabled_by is empty → is_enabled is True for every config shape."""
    assert CapacityGatePlugin.enabled_by == ()
    assert CapacityGatePlugin.is_enabled({}) is True
    assert CapacityGatePlugin.is_enabled({"stage2_reap_ream": {}}) is True
    assert CapacityGatePlugin.is_enabled(
        {"stage2_reap_ream": {"cost_alignment": "post"}}
    ) is True


def test_plugin_compute_cost_is_noop():
    """T11 shell: compute_cost returns None so dispatch_first skips it."""
    assert CapacityGatePlugin().compute_cost(ctx=None) is None


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
    only — after T11 such a patch silently no-ops for LegacyAdapter (which now
    imports from `capacity_gate`). Investigation found none; this guard fails
    loudly if a future edit introduces one without a matching capacity_gate
    patch."""
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
