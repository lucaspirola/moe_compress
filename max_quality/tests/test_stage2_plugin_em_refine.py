"""Task 15 — EM refinement plugin module.

Structural T15 contract: the EmRefinePlugin contract, the numeric
(em_refinement_rounds > 0) is_enabled gate, and a monkeypatch-drift guard
(T9-T14 lesson). Deep algorithm coverage stays in
test_stage2_assignment_v2.py (the `em_`-keyed tests) — this file does NOT
re-test EM internals.
"""
from __future__ import annotations

import pathlib

from moe_compress.stage2._framework.base import Stage2Plugin
from moe_compress.stage2.plugins.em_refine import EmRefinePlugin


# --- plugin contract ------------------------------------------------------
def test_plugin_is_stage2plugin_subclass():
    assert issubclass(EmRefinePlugin, Stage2Plugin)


def test_plugin_name():
    assert EmRefinePlugin.name == "em_refine"


def test_enabled_by_is_empty():
    """Numeric-threshold gate → enabled_by stays empty, is_enabled overridden."""
    assert EmRefinePlugin.enabled_by == ()


def test_overrides_is_enabled():
    """em_refinement_rounds is an int threshold, not a bool flag — the plugin
    must override the base AND-of-flags is_enabled (mirrors ReamCost*Plugin)."""
    assert EmRefinePlugin.is_enabled.__func__ is not Stage2Plugin.is_enabled.__func__


# --- is_enabled numeric gate ---------------------------------------------
def test_is_enabled_true_when_rounds_positive():
    assert EmRefinePlugin.is_enabled(
        {"stage2_reap_ream": {"em_refinement_rounds": 1}}
    ) is True


def test_is_enabled_true_when_rounds_large():
    assert EmRefinePlugin.is_enabled(
        {"stage2_reap_ream": {"em_refinement_rounds": 5}}
    ) is True


def test_is_enabled_false_when_rounds_zero():
    assert EmRefinePlugin.is_enabled(
        {"stage2_reap_ream": {"em_refinement_rounds": 0}}
    ) is False


def test_is_enabled_false_when_rounds_negative():
    """A negative round count is as inert as 0 — EM's own guard is em_rounds<=0."""
    assert EmRefinePlugin.is_enabled(
        {"stage2_reap_ream": {"em_refinement_rounds": -1}}
    ) is False


def test_is_enabled_false_when_key_missing():
    assert EmRefinePlugin.is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_false_when_block_missing():
    assert EmRefinePlugin.is_enabled({}) is False


def test_is_enabled_false_when_non_numeric():
    """A non-numeric value falls back to disabled rather than crashing."""
    assert EmRefinePlugin.is_enabled(
        {"stage2_reap_ream": {"em_refinement_rounds": "abc"}}
    ) is False


def test_is_enabled_coerces_numeric_string():
    """A numeric string is coerced via int() — '2' enables the plugin."""
    assert EmRefinePlugin.is_enabled(
        {"stage2_reap_ream": {"em_refinement_rounds": "2"}}
    ) is True


# --- refine_assignment is a documented no-op (scaffold) -------------------
def test_refine_assignment_returns_none():
    """T15 scaffold: refine_assignment defers (returns None) so dispatch_first
    skips it; the LegacyAdapter still owns the live EM call."""
    assert EmRefinePlugin().refine_assignment(ctx=None, asg=[0], delta=None) is None


# --- monkeypatch-drift guard (T9-T14 lesson) -----------------------------
def test_no_stale_monkeypatch_of_em_symbols():
    """`_em_refine_assignment` / `_em_compute_tentative_weights` moved to
    pipeline.plugins.em_refine in T15. Any test that patches either on the
    monolith namespace must also patch it on the new module (or the live
    LegacyAdapter / legacy-loop path drifts). Fails loudly otherwise."""
    tests_dir = pathlib.Path(__file__).parent
    needles = (
        'setattr(stage2_reap_ream, "_em_refine_assignment"',
        'setattr(stage2_reap_ream, "_em_compute_tentative_weights"',
    )
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        for needle in needles:
            # Strip the needle text itself before scanning for the plugin
            # module name — "_em_refine_assignment" contains "em_refine" as a
            # substring, so a naive `"em_refine" not in text` check would
            # never fire for that needle.
            if needle in text and "em_refine" not in text.replace(needle, ""):
                offenders.append(
                    f"{path.name}: patches an EM symbol on monolith only"
                )
    assert not offenders, (
        "monolith-only monkeypatch of an EM symbol — also patch it on "
        "pipeline.plugins.em_refine:\n" + "\n".join(offenders)
    )
