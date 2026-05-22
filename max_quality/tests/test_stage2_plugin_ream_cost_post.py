"""Task 9 — post-alignment REAM cost plugin module.

Pins the ``ReamCostPostPlugin.is_enabled`` truth table. Algorithm coverage is
provided by the existing ``test_stage2_assignment_v2.py`` /
``test_stage2_output_cost.py`` suites.
"""
from __future__ import annotations

import pytest

from moe_compress.stage2.plugins.ream_cost_post import ReamCostPostPlugin


# --- ReamCostPostPlugin.is_enabled truth table ------------------------------

@pytest.mark.parametrize("cost_alignment,expected", [
    ("post", True),
    ("POST", True),      # case-insensitive (matches run() .lower() normalize)
    ("pre", False),
    ("output", False),
])
def test_is_enabled_explicit(cost_alignment, expected):
    cfg = {"stage2_reap_ream": {"cost_alignment": cost_alignment}}
    assert ReamCostPostPlugin.is_enabled(cfg) is expected


def test_is_enabled_default_missing_key():
    """Missing `cost_alignment` -> default 'pre' -> post plugin disabled."""
    assert ReamCostPostPlugin.is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_missing_block():
    """Missing `stage2_reap_ream` block -> default 'pre' -> post disabled."""
    assert ReamCostPostPlugin.is_enabled({}) is False


def test_compute_cost_is_noop():
    """T9: compute_cost is a documented no-op (legacy bump loop still owns it)."""
    assert ReamCostPostPlugin().compute_cost(ctx=None) is None  # type: ignore[arg-type]


def test_plugin_name():
    assert ReamCostPostPlugin.name == "ream_cost_post"
