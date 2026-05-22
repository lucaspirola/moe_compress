"""Task 12 / S2-7 — skip-merge floor plugin module.

Pins the SkipMergeFloorPlugin contract, the is_enabled numeric gate, the
apply_cost_mask delegation to grouping._apply_skip_merge_floor, the OFF-sentinel
passthrough, a monkeypatch-drift guard (T9–T11 lesson), and (S2-7) the live
``apply_cost_mask`` slot: the INFO log line and the ``PluginRegistry`` wiring.
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.pipeline.registry import PluginRegistry
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
    """At percentile 100.0 the plugin returns the SAME array object (no copy):
    the OFF sentinel skips the ``_apply_skip_merge_floor`` helper entirely."""
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
    plugin path imports it from `pipeline.grouping`."""
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


# --- S2-7: live apply_cost_mask slot ----------------------------------------

class _LayerRefStub:
    """Minimal layer_ref stub exposing only ``.layer_idx`` (read by the log)."""

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx


def test_apply_cost_mask_emits_log_when_masked(caplog):
    """With a real ctx carrying a layer_ref, the INFO log fires when masking
    actually pushes entries to +inf."""
    rng = np.random.default_rng(1)
    delta = rng.random((4, 5)).astype(np.float64) * 10.0
    ctx = PipelineContext()
    ctx.set("layer_ref", _LayerRefStub(layer_idx=7))
    plugin = SkipMergeFloorPlugin(skip_merge_percentile=50.0)

    # Some pytest plugins in this env default new loggers to propagate=False;
    # the real stage2 logger has propagate=True, so mirror that here so the
    # caplog handler on the root logger receives the INFO record.
    smf_logger = logging.getLogger("moe_compress.stage2.plugins.skip_merge_floor")
    smf_logger.propagate = True

    with caplog.at_level(logging.INFO):
        masked, info = plugin.apply_cost_mask(ctx, delta)

    assert info["n_masked"] > 0
    records = [
        r for r in caplog.records
        if r.name == "moe_compress.stage2.plugins.skip_merge_floor"
        and r.levelno == logging.INFO
    ]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "layer 7" in msg
    assert "skip-merge floor (P50.0)" in msg
    assert f"masked {info['n_masked']}/{masked.size}" in msg


# --- S2-7: PluginRegistry wiring --------------------------------------------

class _AlwaysOnPlugin:
    """Minimal always-enabled plugin standing in for the LayerMergePlugin."""

    name = "always_on_adapter_stub"

    def is_enabled(self, config: dict) -> bool:
        return True


def test_registry_wiring_skip_merge_floor():
    """The plugin is enabled below 100.0, dropped at/above it, and ordered
    before the (always-on) adapter stand-in."""
    skip = SkipMergeFloorPlugin(skip_merge_percentile=50.0)
    adapter_stub = _AlwaysOnPlugin()
    registry = PluginRegistry([skip, adapter_stub])

    enabled = registry.enabled(
        {"stage2_reap_ream": {"skip_merge_percentile": 50.0}}
    )
    assert skip in enabled
    assert adapter_stub in enabled
    # Ordered before the adapter so it wins the apply_cost_mask dispatch_first.
    assert enabled.index(skip) < enabled.index(adapter_stub)

    dropped = registry.enabled(
        {"stage2_reap_ream": {"skip_merge_percentile": 100.0}}
    )
    assert skip not in dropped
    assert adapter_stub in dropped


def test_orchestrator_registers_skip_merge_floor_before_adapter():
    """The orchestrator source registers ``SkipMergeFloorPlugin`` after the
    three cost plugins and before the merge spine in the ``PluginRegistry``
    list. S2-12: the merge-spine entry is the ``layer_merge``
    (``LayerMergePlugin``) instance that replaced the retired ``LegacyAdapter``."""
    src = (
        pathlib.Path(__file__).parents[1]
        / "src/moe_compress/stage2/orchestrator.py"
    ).read_text()
    smf = src.index("SkipMergeFloorPlugin(skip_merge_percentile=")
    out_cost = src.index("OutputSpaceCostPlugin(**_cost_plugin_kwargs)")
    adapter = src.index("\n        layer_merge,\n")
    assert out_cost < smf < adapter
