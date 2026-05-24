"""Task 14 — two-opt refinement plugin module.

Structural T14 contract: the TwoOptRefinePlugin contract, the boolean
is_enabled gate, and a monkeypatch-drift guard (T9-T11 lesson). Deep
algorithm coverage stays in test_stage2_two_opt.py — this file does NOT
re-test 2-opt internals.
"""
from __future__ import annotations

import pathlib

from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage2.plugins.two_opt_refine import TwoOptRefinePlugin


# --- plugin contract ------------------------------------------------------
def test_plugin_conforms_to_pipeline_plugin():
    assert isinstance(TwoOptRefinePlugin(), PipelinePlugin)


def test_plugin_name():
    assert TwoOptRefinePlugin.name == "two_opt_refine"


# --- is_enabled boolean gate ----------------------------------------------
def test_is_enabled_true_when_flag_set():
    assert TwoOptRefinePlugin().is_enabled(
        {"stage2_reap_ream": {"two_opt_refine": True}}
    ) is True


def test_is_enabled_false_when_flag_false():
    assert TwoOptRefinePlugin().is_enabled(
        {"stage2_reap_ream": {"two_opt_refine": False}}
    ) is False


def test_is_enabled_false_when_key_missing():
    assert TwoOptRefinePlugin().is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_false_when_block_missing():
    assert TwoOptRefinePlugin().is_enabled({}) is False


# --- monkeypatch-drift guard (T9-T11 lesson) ------------------------------
def test_no_stale_monkeypatch_of_two_opt_refine():
    """`_two_opt_refine` moved to pipeline.plugins.two_opt_refine in T14. Any
    test that patches it on the monolith namespace must also patch it on the
    new module (or the live plugin path drifts). Fails loudly otherwise."""
    tests_dir = pathlib.Path(__file__).parent
    needle = 'setattr(stage2_reap_ream, "_two_opt_refine"'
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        if needle in text and "two_opt_refine" not in text.replace(needle, ""):
            offenders.append(f"{path.name}: patches _two_opt_refine on monolith only")
    assert not offenders, (
        "monolith-only monkeypatch of _two_opt_refine — also patch it on "
        "pipeline.plugins.two_opt_refine:\n" + "\n".join(offenders)
    )
