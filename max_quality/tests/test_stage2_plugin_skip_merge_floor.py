"""Task 12 — skip-merge floor plugin module.

Pins the SkipMergeFloorPlugin contract, the is_enabled numeric gate, the
apply_cost_mask delegation to grouping._apply_skip_merge_floor, the OFF-sentinel
passthrough, and a monkeypatch-drift guard (T9–T11 lesson).
"""
from __future__ import annotations

import pathlib

import numpy as np

from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage2.grouping import _apply_skip_merge_floor
from moe_compress.stage2.plugins.skip_merge_floor import (
    SkipMergeFloorPlugin,
    make_skip_merge_floor_plugin,
)


# --- plugin contract --------------------------------------------------------

def test_plugin_conforms_to_pipeline_plugin():
    assert isinstance(SkipMergeFloorPlugin(), PipelinePlugin)


def test_plugin_name():
    assert SkipMergeFloorPlugin.name == "skip_merge_floor"


# --- is_enabled numeric gate ------------------------------------------------

def test_is_enabled_true_below_100():
    assert SkipMergeFloorPlugin().is_enabled(
        {"stage2_reap_ream": {"skip_merge_percentile": 95.0}}
    ) is True


def test_is_enabled_false_at_off_sentinel():
    """100.0 is the OFF sentinel → disabled."""
    assert SkipMergeFloorPlugin().is_enabled(
        {"stage2_reap_ream": {"skip_merge_percentile": 100.0}}
    ) is False


def test_is_enabled_false_when_key_missing():
    """Missing key defaults to 100.0 → disabled."""
    assert SkipMergeFloorPlugin().is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_false_when_block_missing():
    assert SkipMergeFloorPlugin().is_enabled({}) is False


def test_is_enabled_false_above_100():
    """Values > 100 are also OFF (nothing sits strictly above a >100 clamp)."""
    assert SkipMergeFloorPlugin().is_enabled(
        {"stage2_reap_ream": {"skip_merge_percentile": 150.0}}
    ) is False


def test_is_enabled_boundary_just_below_100():
    assert SkipMergeFloorPlugin().is_enabled(
        {"stage2_reap_ream": {"skip_merge_percentile": 99.9}}
    ) is True


# --- factory ----------------------------------------------------------------

def test_factory_reads_percentile_from_cfg():
    plugin = make_skip_merge_floor_plugin(
        {"stage2_reap_ream": {"skip_merge_percentile": 60.0}}
    )
    assert isinstance(plugin, SkipMergeFloorPlugin)
    assert plugin.skip_merge_percentile == 60.0


def test_factory_defaults_to_off_when_missing():
    plugin = make_skip_merge_floor_plugin({})
    assert plugin.skip_merge_percentile == 100.0


# --- apply_cost_mask --------------------------------------------------------

def test_apply_cost_mask_masks_high_cost_entries():
    """apply_cost_mask delegates to _apply_skip_merge_floor: entries strictly
    above the percentile become +inf, byte-identical to the helper."""
    rng = np.random.default_rng(0)
    delta = rng.random((4, 5)).astype(np.float64) * 10.0
    plugin = SkipMergeFloorPlugin(skip_merge_percentile=50.0)
    masked, info = plugin.apply_cost_mask(ctx=None, delta=delta)

    ref_masked, ref_n = _apply_skip_merge_floor(delta, 50.0)
    np.testing.assert_array_equal(masked, ref_masked)
    assert info["n_masked"] == ref_n
    assert info["percentile"] == 50.0
    # at least one entry was pushed to +inf (sanity: P50 of random data)
    assert np.isinf(masked).any()
    assert info["n_masked"] > 0


def test_apply_cost_mask_does_not_mutate_input():
    delta = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    before = delta.copy()
    SkipMergeFloorPlugin(skip_merge_percentile=50.0).apply_cost_mask(None, delta)
    np.testing.assert_array_equal(delta, before)


def test_apply_cost_mask_off_sentinel_returns_input_unchanged():
    """At percentile 100.0 the plugin returns the SAME array object (no copy),
    matching the LegacyAdapter live path which skips the helper entirely."""
    delta = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    plugin = SkipMergeFloorPlugin(skip_merge_percentile=100.0)
    masked, info = plugin.apply_cost_mask(None, delta)
    assert masked is delta
    assert info == {"n_masked": 0, "percentile": 100.0}


def test_apply_cost_mask_default_construction_is_off():
    """Bare SkipMergeFloorPlugin() defaults to the OFF sentinel."""
    delta = np.array([[1.0, 2.0]], dtype=np.float64)
    masked, info = SkipMergeFloorPlugin().apply_cost_mask(None, delta)
    assert masked is delta
    assert info["n_masked"] == 0


# --- monkeypatch-drift guard (T9–T11 lesson) --------------------------------

def test_no_stale_monkeypatch_of_skip_merge_floor():
    """`_apply_skip_merge_floor` did not move in T12 (it stayed in grouping.py
    since T5), so no namespace went stale. This guard fails loudly if a future
    edit patches the symbol on the monolith namespace only while the live
    LegacyAdapter path imports it from `pipeline.grouping`."""
    tests_dir = pathlib.Path(__file__).parent
    needle = 'setattr(stage2_reap_ream, "_apply_skip_merge_floor"'
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        if needle in text and "grouping" not in text:
            offenders.append(f"{path.name}: patches _apply_skip_merge_floor "
                             "on monolith only")
    assert not offenders, (
        "monolith-only monkeypatch of _apply_skip_merge_floor — also patch it "
        "on pipeline.grouping (T9 dual-patch lesson):\n" + "\n".join(offenders)
    )
