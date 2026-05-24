"""S2-12 — the always-on ``LayerMergePlugin`` (per-layer merge spine).

Dedicated unit coverage for the plugin that S2-12a carved out of the retired
``LegacyAdapter``: it owns the SIX live phase hooks (``on_layer_setup`` /
``on_profile`` / ``merge`` / ``post_merge`` / ``write_artifacts`` /
``on_layer_teardown``).

Coverage here is deliberately narrow — the plugin contract, the always-on
``is_enabled`` gate, and the two accumulator-lifecycle hooks
(``on_layer_setup`` / ``on_layer_teardown``). The merge / profile / artifact
hooks and the ``_run_assignment`` integration are exercised end-to-end by
``test_stage2_pipeline_run_layer.py``; this file does not duplicate them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage2.permutation_align import _PermAlignCache
from moe_compress.stage2.plugins.layer_merge import LayerMergePlugin
from moe_compress.utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _LayerRefStub:
    """Minimal layer_ref — ``on_layer_setup`` reads only ``.layer_idx``."""

    def __init__(self, layer_idx: int = 0) -> None:
        self.layer_idx = layer_idx


def _build_plugin(tiny_config, tmp_path, *, expert_distill_steps=0,
                  cost_alignment_cfg="pre"):
    """Construct a ``LayerMergePlugin`` with the trimmed knob set its
    ``__init__`` requires — the same superset the orchestrator passes."""
    s2 = tiny_config["stage2_reap_ream"]
    from moe_compress.stage2 import orchestrator as _srr
    return LayerMergePlugin(
        s2_cfg=s2, heal_cfg=_srr._HealConfig(s2),
        batches=[], model=None,
        cov_acc=InputCovarianceAccumulator(), merge_map={},
        layer_mean_costs=[], partial_dir=tmp_path,
        max_group_cap=0, cost_sigma=float("inf"),
        cost_bump_ratio=0.1, min_active_tokens=1,
        assignment_solver="greedy", cost_alignment_cfg=cost_alignment_cfg,
        cost_output_token_cap=8, cost_asymmetric=False,
        expert_distill_steps=expert_distill_steps, expert_distill_token_cap=8,
        blacklist={}, device=None,
    )


# ---------------------------------------------------------------------------
# plugin contract
# ---------------------------------------------------------------------------


def test_layer_merge_structural_conformance():
    """LayerMergePlugin satisfies the universal PipelinePlugin contract.

    The class carries every metadata attribute with the right type and the
    two universal core methods; the class object structurally satisfies the
    runtime_checkable Protocol.
    """
    for attr in ("name", "paper", "config_key", "reads", "writes", "provides"):
        assert hasattr(LayerMergePlugin, attr), f"LayerMergePlugin missing {attr!r}"
    assert isinstance(LayerMergePlugin.name, str)
    assert isinstance(LayerMergePlugin.paper, str)
    assert isinstance(LayerMergePlugin.config_key, str)
    assert isinstance(LayerMergePlugin.reads, tuple)
    assert isinstance(LayerMergePlugin.writes, tuple)
    assert isinstance(LayerMergePlugin.provides, tuple)
    assert callable(getattr(LayerMergePlugin, "is_enabled", None))
    assert callable(getattr(LayerMergePlugin, "contribute_artifact", None))
    assert isinstance(LayerMergePlugin, PipelinePlugin)


def test_layer_merge_owns_the_six_live_phase_hooks():
    """LayerMergePlugin declares exactly the SIX live phase hooks relocated
    out of the retired LegacyAdapter (S2-12a)."""
    for hook in ("on_layer_setup", "on_profile", "merge", "post_merge",
                 "write_artifacts", "on_layer_teardown"):
        assert callable(getattr(LayerMergePlugin, hook, None)), (
            f"LayerMergePlugin missing live phase hook {hook!r}"
        )
    # The dead dispatch_first fallbacks never relocated here — they died with
    # the LegacyAdapter (S2-12b). The live cost/solve/refine slots are
    # serviced by the cost / solver / refine plugins instead.
    for dead_slot in ("compute_cost", "apply_cost_mask",
                      "solve_assignment", "refine_assignment",
                      "compute_assignment"):
        assert not hasattr(LayerMergePlugin, dead_slot), (
            f"LayerMergePlugin must not declare {dead_slot!r}"
        )


# ---------------------------------------------------------------------------
# is_enabled — always on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", [
    {},
    {"stage2_reap_ream": {}},
    {"stage2_reap_ream": {"cost_alignment": "post"}},
    {"unrelated": {"key": "value"}},
])
def test_is_enabled_always_true(config, tiny_config, tmp_path):
    """The per-layer merge spine runs on every Stage-2 run — is_enabled
    returns True for any config, including an empty one."""
    plugin = _build_plugin(tiny_config, tmp_path)
    assert plugin.is_enabled(config) is True


# ---------------------------------------------------------------------------
# on_layer_setup / on_layer_teardown accumulator lifecycle
# ---------------------------------------------------------------------------


def test_on_layer_setup_populates_per_layer_accumulators(tiny_config, tmp_path):
    """on_layer_setup builds ream_acc + perm_cache; layer_input_acc is None in
    the default mode (expert_distill_steps=0, cost_alignment='pre')."""
    plugin = _build_plugin(tiny_config, tmp_path)
    ctx = PipelineContext()
    ctx.set("layer_ref", _LayerRefStub(layer_idx=0))
    plugin.on_layer_setup(ctx)
    assert isinstance(ctx.get("ream_acc"), ReamCostAccumulator)
    assert isinstance(ctx.get("perm_cache"), _PermAlignCache)
    # No input capture: distillation off and cost_alignment != "output".
    assert ctx.get("layer_input_acc") is None


def test_on_layer_setup_enables_layer_input_acc_for_distillation(tiny_config,
                                                                 tmp_path):
    """When expert distillation is on, on_layer_setup builds a
    layer_input_acc (the per-expert distillation calibration buffer)."""
    from moe_compress.stage2.profiling import _LayerInputAccumulator

    plugin = _build_plugin(tiny_config, tmp_path, expert_distill_steps=3)
    ctx = PipelineContext()
    ctx.set("layer_ref", _LayerRefStub(layer_idx=2))
    plugin.on_layer_setup(ctx)
    assert isinstance(ctx.get("layer_input_acc"), _LayerInputAccumulator)


def test_on_layer_teardown_nulls_per_layer_slots(tiny_config, tmp_path):
    """on_layer_teardown nulls every per-layer slot unconditionally
    (overwrite=True is an upsert — no pre-seed needed)."""
    plugin = _build_plugin(tiny_config, tmp_path)
    ctx = PipelineContext()
    ctx.set("layer_ref", _LayerRefStub(layer_idx=0))
    plugin.on_layer_setup(ctx)
    assert ctx.get("ream_acc") is not None
    assert ctx.get("perm_cache") is not None

    plugin.on_layer_teardown(ctx)
    for slot in ("reap_acc", "ream_acc", "perm_cache", "layer_input_acc",
                 "pre_merge_weights", "distill_state",
                 "nemo_writer", "xd_writer"):
        assert ctx.get(slot) is None, f"slot {slot!r} not nulled by teardown"
