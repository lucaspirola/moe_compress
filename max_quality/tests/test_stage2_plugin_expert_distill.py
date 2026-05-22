"""Task 16 — Per-group expert distillation plugin module.

Structural T16 contract: the ExpertDistillPlugin contract, the numeric
(expert_distill_steps > 0) is_enabled gate, and a monkeypatch-drift guard
(T9-T15 lesson). Deep algorithm coverage stays in
test_stage2_expert_distill.py — this file does NOT re-test the
distillation internals.
"""
from __future__ import annotations

import pathlib

from moe_compress.stage2._framework.base import Stage2Plugin
from moe_compress.stage2.plugins.expert_distill import ExpertDistillPlugin


# --- plugin contract ------------------------------------------------------
def test_plugin_is_stage2plugin_subclass():
    assert issubclass(ExpertDistillPlugin, Stage2Plugin)


def test_plugin_name():
    assert ExpertDistillPlugin.name == "expert_distill"


def test_enabled_by_is_empty():
    """Numeric-threshold gate → enabled_by stays empty, is_enabled overridden."""
    assert ExpertDistillPlugin.enabled_by == ()


def test_overrides_is_enabled():
    """expert_distill_steps is an int threshold, not a bool flag — the plugin
    must override the base AND-of-flags is_enabled (mirrors EmRefinePlugin)."""
    assert (
        ExpertDistillPlugin.is_enabled.__func__
        is not Stage2Plugin.is_enabled.__func__
    )


# --- is_enabled numeric gate ---------------------------------------------
def test_is_enabled_true_when_steps_positive():
    assert ExpertDistillPlugin.is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": 1}}
    ) is True


def test_is_enabled_true_when_steps_large():
    assert ExpertDistillPlugin.is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": 500}}
    ) is True


def test_is_enabled_false_when_steps_zero():
    assert ExpertDistillPlugin.is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": 0}}
    ) is False


def test_is_enabled_false_when_steps_negative():
    """A negative step count is as inert as 0 — the distill guard is steps<=0."""
    assert ExpertDistillPlugin.is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": -1}}
    ) is False


def test_is_enabled_false_when_key_missing():
    assert ExpertDistillPlugin.is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_false_when_block_missing():
    assert ExpertDistillPlugin.is_enabled({}) is False


def test_is_enabled_false_when_non_numeric():
    """A non-numeric value falls back to disabled rather than crashing."""
    assert ExpertDistillPlugin.is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": "abc"}}
    ) is False


def test_is_enabled_coerces_numeric_string():
    """A numeric string is coerced via int() — '3' enables the plugin."""
    assert ExpertDistillPlugin.is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": "3"}}
    ) is True


# --- pre_merge_snapshot / post_merge are documented no-ops (scaffold) -----
def test_pre_merge_snapshot_returns_none():
    """T16 scaffold: pre_merge_snapshot defers; the LegacyAdapter still owns
    the live _snapshot_pre_merge_layer_experts call."""
    assert ExpertDistillPlugin().pre_merge_snapshot(ctx=None) is None


def test_post_merge_returns_none():
    """T16 scaffold: post_merge defers; the LegacyAdapter still owns the live
    _distill_merged_group call."""
    assert ExpertDistillPlugin().post_merge(ctx=None) is None


# --- monkeypatch-drift guard (T9-T15 lesson) -----------------------------
def test_no_stale_monkeypatch_of_distill_symbols():
    """`_distill_merged_group` / `_snapshot_pre_merge_layer_experts` moved to
    pipeline.plugins.expert_distill in T16. Any test that patches either on the
    monolith namespace must also patch it on the new module (or the live
    LegacyAdapter / legacy-loop path drifts). Fails loudly otherwise.

    No existing test patches these symbols (verified during T16 planning) —
    this guard is anticipatory, protecting against future drift.
    """
    tests_dir = pathlib.Path(__file__).parent
    needles = (
        'setattr(stage2_reap_ream, "_distill_merged_group"',
        'setattr(stage2_reap_ream, "_snapshot_pre_merge_layer_experts"',
    )
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        for needle in needles:
            # Strip the needle text itself before scanning for the plugin
            # module name — "_distill_merged_group" does NOT contain
            # "expert_distill" as a substring, but strip defensively to mirror
            # the em_refine guard's robustness.
            if needle in text and "expert_distill" not in text.replace(needle, ""):
                offenders.append(
                    f"{path.name}: patches a distill symbol on monolith only"
                )
    assert not offenders, (
        "monolith-only monkeypatch of a distill symbol — also patch it on "
        "pipeline.plugins.expert_distill:\n" + "\n".join(offenders)
    )
