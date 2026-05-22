"""Task 17 — Per-layer merge-heal plugin module.

Structural T17 contract: the MergeHealPlugin contract, the boolean
(merge_heal_enabled) is_enabled gate, and a monkeypatch-drift guard
(T9-T16 lesson). Deep algorithm coverage stays in
test_stage2_merge_heal.py — this file does NOT re-test the heal internals.
"""
from __future__ import annotations

import pathlib

from moe_compress.stage2._framework.base import Stage2Plugin
from moe_compress.stage2.plugins.merge_heal import MergeHealPlugin

_HEAL_NAMES = (
    "_HealConfig",
    "_make_shared_out_fn",
    "_capture_mlp_io",
    "_heal_student_moe_output",
    "_heal_lr_at_step",
    "_heal_layer",
    "_summarize_distill_state",
)


# --- plugin contract ------------------------------------------------------
def test_plugin_is_stage2plugin_subclass():
    assert issubclass(MergeHealPlugin, Stage2Plugin)


def test_plugin_name():
    assert MergeHealPlugin.name == "merge_heal"


def test_enabled_by_is_merge_heal_enabled():
    """Plain bool flag — base AND-of-flags is_enabled is used directly."""
    assert MergeHealPlugin.enabled_by == ("merge_heal_enabled",)


def test_does_not_override_is_enabled():
    """merge_heal_enabled is a plain bool flag, so the base AND-of-flags
    is_enabled expresses the gate exactly — no override (unlike
    ExpertDistillPlugin / EmRefinePlugin)."""
    assert (
        MergeHealPlugin.is_enabled.__func__
        is Stage2Plugin.is_enabled.__func__
    )


# --- is_enabled boolean gate ---------------------------------------------
def test_is_enabled_true_when_flag_true():
    assert MergeHealPlugin.is_enabled(
        {"stage2_reap_ream": {"merge_heal_enabled": True}}
    ) is True


def test_is_enabled_false_when_flag_false():
    assert MergeHealPlugin.is_enabled(
        {"stage2_reap_ream": {"merge_heal_enabled": False}}
    ) is False


def test_is_enabled_false_when_key_missing():
    assert MergeHealPlugin.is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_false_when_block_missing():
    assert MergeHealPlugin.is_enabled({}) is False


# --- pre_merge_snapshot / post_merge / write_artifacts are no-ops (scaffold)
def test_pre_merge_snapshot_returns_none():
    assert MergeHealPlugin().pre_merge_snapshot(ctx=None) is None


def test_post_merge_returns_none():
    assert MergeHealPlugin().post_merge(ctx=None) is None


def test_write_artifacts_returns_empty_dict():
    assert MergeHealPlugin().write_artifacts(ctx=None, partial_dir=None) == {}


# --- monkeypatch-drift guard (T9-T16 lesson) -----------------------------
def test_no_stale_monkeypatch_of_heal_symbols():
    """The 7 heal symbols moved to pipeline.plugins.merge_heal in T17. Any
    test that patches one on the monolith namespace must also patch it on the
    new module (or the live LegacyAdapter / legacy-loop path drifts). Fails
    loudly otherwise.

    No existing test patches these symbols (verified during T17 planning —
    test_stage2_merge_heal.py patches only torch.optim.AdamW.step;
    test_smoke_stage2_resume.py patches only _profile_layer /
    build_calibration_tensor / save_compressed_checkpoint) — this guard is
    anticipatory, protecting against future drift.
    """
    tests_dir = pathlib.Path(__file__).parent
    needles = tuple(
        f'setattr(stage2_reap_ream, "{name}"' for name in _HEAL_NAMES
    )
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        for needle in needles:
            if needle in text and "merge_heal" not in text.replace(needle, ""):
                offenders.append(
                    f"{path.name}: patches a heal symbol on monolith only"
                )
    assert not offenders, (
        "monolith-only monkeypatch of a heal symbol — also patch it on "
        "pipeline.plugins.merge_heal:\n" + "\n".join(offenders)
    )
