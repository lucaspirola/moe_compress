"""Task 10 — output-space (Direction C) cost plugin module.

Pins the ``OutputSpaceCostPlugin.is_enabled`` truth table. Algorithm coverage
is provided by the existing ``test_stage2_output_cost.py`` suite. Also guards
against monkeypatch drift: no test may patch one of the 4 moved symbols via
the monolith namespace only.
"""
from __future__ import annotations

import pathlib

import pytest

from moe_compress.stage2.plugins.output_space_cost import OutputSpaceCostPlugin


def test_output_branch_uses_plugin_module():
    """The `output` branch of `_ream_cost_matrix` imports `_output_space_cost`
    from the sibling plugin module, not the monolith."""
    import inspect
    import moe_compress.stage2.plugins.ream_cost as ream_cost
    src = inspect.getsource(ream_cost._ream_cost_matrix)
    assert "from .output_space_cost import _output_space_cost" in src
    assert "from ...stage2_reap_ream import _output_space_cost" not in src


# --- OutputSpaceCostPlugin.is_enabled truth table ---------------------------

@pytest.mark.parametrize("cost_alignment,expected", [
    ("output", True),
    ("OUTPUT", True),    # case-insensitive (matches run() .lower() normalize)
    ("pre", False),
    ("post", False),
])
def test_is_enabled_explicit(cost_alignment, expected):
    cfg = {"stage2_reap_ream": {"cost_alignment": cost_alignment}}
    assert OutputSpaceCostPlugin().is_enabled(cfg) is expected


def test_is_enabled_default_missing_key():
    """Missing `cost_alignment` -> default 'pre' -> output plugin disabled."""
    assert OutputSpaceCostPlugin().is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_missing_block():
    """Missing `stage2_reap_ream` block -> default 'pre' -> output disabled."""
    assert OutputSpaceCostPlugin().is_enabled({}) is False


def test_compute_cost_is_noop():
    """T10: compute_cost is a documented no-op (legacy bump loop still owns it)."""
    assert OutputSpaceCostPlugin().compute_cost(ctx=None) is None  # type: ignore[arg-type]


def test_plugin_name():
    assert OutputSpaceCostPlugin.name == "output_space_cost"


# --- monkeypatch-drift guard (the T9 lesson) --------------------------------

def test_no_monolith_only_monkeypatch_of_moved_symbols():
    """Guard: no test patches a moved output-space symbol via the
    `stage2_reap_ream` namespace only.

    The 4 functions resolve `build_banks` / `_permutation_align_to_centroid`
    from `output_space_cost`'s own namespace. A test that does
    `monkeypatch.setattr(stage2_reap_ream, "<sym>", ...)` for one of the moved
    functions OR for a symbol they use internally would silently no-op after
    T10. Investigation found none; this test fails loudly if a future edit
    introduces one without the matching `output_space_cost` patch.
    """
    moved = {
        "_output_space_cost",
        "_tentative_merged_weights",
        "_router_routing_weights",
        "_swiglu_forward",
    }
    tests_dir = pathlib.Path(__file__).parent
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        for sym in moved:
            needle = f'setattr(stage2_reap_ream, "{sym}"'
            if needle in text and "output_space_cost" not in text:
                offenders.append(f"{path.name}: patches {sym} on monolith only")
    assert not offenders, (
        "monolith-only monkeypatch of a moved output-space symbol — add a "
        "matching output_space_cost patch (T9 dual-patch lesson):\n"
        + "\n".join(offenders)
    )
