"""Per-group expert distillation plugin module.

Structural contract: the ExpertDistillPlugin contract, the numeric
(expert_distill_steps > 0) is_enabled gate, a monkeypatch-drift guard
(T9-T15 lesson), the LIVE S2-11 hook coverage (the plugin exposes
``pre_merge_snapshot`` / ``merge``), and an ON-path equivalence test that
drives the plugin's hooks with distillation ENABLED and asserts byte-equality
vs. a direct ``_distill_merged_group`` reference. Deep algorithm coverage stays
in test_stage2_expert_distill.py — this file does NOT re-test the
distillation internals.
"""
from __future__ import annotations

import copy
import pathlib

import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage2.merging import _merge_experts_inplace
from moe_compress.stage2.plugins.expert_distill import (
    ExpertDistillPlugin,
    _distill_merged_group,
    _snapshot_pre_merge_layer_experts,
)


# Default knob set for a constructed (live) plugin. Mirrors the
# ``expert_distill_*`` block of ``stage2_reap_ream.run()``.
_DISTILL_KNOBS = dict(
    expert_distill_steps=0,
    expert_distill_lr=1e-4,
    expert_distill_betas=(0.9, 0.95),
    expert_distill_token_cap=8,
    expert_distill_skip_singletons=True,
    expert_distill_plateau_steps=2,
    expert_distill_plateau_eps=1e-4,
)


def _make_plugin(**overrides):
    """Construct an ExpertDistillPlugin with the default knob set + overrides."""
    knobs = dict(_DISTILL_KNOBS)
    knobs.update(overrides)
    return ExpertDistillPlugin(**knobs)


# --- plugin contract ------------------------------------------------------
def test_plugin_conforms_to_pipeline_plugin():
    assert isinstance(_make_plugin(), PipelinePlugin)


def test_plugin_name():
    assert ExpertDistillPlugin.name == "expert_distill"


# --- is_enabled numeric gate ---------------------------------------------
def test_is_enabled_true_when_steps_positive():
    assert _make_plugin().is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": 1}}
    ) is True


def test_is_enabled_true_when_steps_large():
    assert _make_plugin().is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": 500}}
    ) is True


def test_is_enabled_false_when_steps_zero():
    assert _make_plugin().is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": 0}}
    ) is False


def test_is_enabled_false_when_steps_negative():
    """A negative step count is as inert as 0 — the distill guard is steps<=0."""
    assert _make_plugin().is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": -1}}
    ) is False


def test_is_enabled_false_when_key_missing():
    assert _make_plugin().is_enabled({"stage2_reap_ream": {}}) is False


def test_is_enabled_false_when_block_missing():
    assert _make_plugin().is_enabled({}) is False


def test_is_enabled_false_when_non_numeric():
    """A non-numeric value falls back to disabled rather than crashing."""
    assert _make_plugin().is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": "abc"}}
    ) is False


def test_is_enabled_coerces_numeric_string():
    """A numeric string is coerced via int() — '3' enables the plugin."""
    assert _make_plugin().is_enabled(
        {"stage2_reap_ream": {"expert_distill_steps": "3"}}
    ) is True


# --- LIVE S2-11 hooks: structural ----------------------------------------
def test_exposes_live_phase_hooks():
    """S2-11: the plugin owns the ``pre_merge_snapshot`` + ``merge`` phases."""
    plugin = _make_plugin()
    assert callable(getattr(plugin, "pre_merge_snapshot", None))
    assert callable(getattr(plugin, "merge", None))
    # post_merge is NOT a hook of this plugin (merge-heal owns post_merge).
    assert not hasattr(plugin, "post_merge")


def test_pre_merge_snapshot_disabled_writes_none():
    """With distillation OFF the snapshot hook writes ``pre_merge_weights=None``
    (no host-RAM cost) — but the slot IS published so downstream reads resolve.
    """
    ctx = PipelineContext()
    ctx.set("layer_ref", object())  # not read when steps == 0
    _make_plugin(expert_distill_steps=0).pre_merge_snapshot(ctx)
    assert ctx.get("pre_merge_weights") is None


def test_merge_disabled_overwrites_distill_state_none():
    """With distillation OFF the merge hook overwrites ``distill_state`` (which
    LayerMergePlugin.merge defaults to None) — still None, no crash."""
    ctx = PipelineContext()
    ctx.set("layer_ref", object())
    ctx.set("grouped", {})
    ctx.set("freq", {})
    ctx.set("layer_input_acc", None)
    ctx.set("pre_merge_weights", None)
    ctx.set("distill_state", None)  # LayerMergePlugin.merge default
    _make_plugin(expert_distill_steps=0).merge(ctx)
    assert ctx.get("distill_state") is None


# --- LIVE S2-11 hooks: ON-path equivalence (distill ENABLED) -------------
def test_distill_on_path_matches_reference(tiny_model):
    """ON-path equivalence: drive ExpertDistillPlugin's pre_merge_snapshot +
    merge hooks with distillation ENABLED and assert byte-equality vs. a direct
    ``_merge_experts_inplace`` + ``_distill_merged_group`` reference run on an
    independent deepcopy.

    Both paths faithfully reproduce the production composite merge sequence:
    ``LayerMergePlugin.merge`` runs ``_merge_experts_inplace`` FIRST (mutating
    the centroid bank weights) and ``ExpertDistillPlugin.merge`` runs the
    ``_distill_merged_group`` loop SECOND — so distillation starts from
    POST-MERGE centroid weights. The pre-merge snapshot (the distillation
    target) is captured BEFORE the merge on both paths.

    The tiny_config gate (test_stage2_pipeline_run_layer) only exercises the
    distill-OFF path; this covers the ON path. ``_distill_merged_group`` is
    deterministic (layer-idx-seeded token subsample, no other randomness), so
    two runs on identical inputs produce bit-identical bank weights.
    """
    from moe_compress.utils.model_io import build_banks, iter_moe_layers

    distill_kwargs = dict(
        expert_distill_steps=8,
        expert_distill_lr=5e-3,
        expert_distill_betas=(0.9, 0.95),
        expert_distill_token_cap=16,
        expert_distill_skip_singletons=True,
        expert_distill_plateau_steps=100,  # disable plateau-break
        expert_distill_plateau_eps=0.0,
    )

    # --- Reference run: direct _snapshot + _distill_merged_group ----------
    ref_model = copy.deepcopy(tiny_model)
    ref_layer = list(iter_moe_layers(ref_model))[0]
    grouped = {0: [0, 1], 2: [2]}  # one non-singleton group + one singleton
    freq = {0: 3, 1: 1, 2: 5}
    torch.manual_seed(1234)
    layer_inputs = torch.randn(16, ref_layer.experts_module.hidden_dim) * 0.1

    # Snapshot the ORIGINAL pre-merge expert weights — these are the
    # distillation target. The snapshot is taken BEFORE _merge_experts_inplace
    # mutates the centroid bank weights.
    ref_pre = _snapshot_pre_merge_layer_experts(ref_layer)
    # Production sequence: LayerMergePlugin.merge runs _merge_experts_inplace
    # FIRST (mutating the centroid bank weights), THEN ExpertDistillPlugin.merge
    # distills the merged centroid. Replicate that here so the reference starts
    # from POST-MERGE centroid weights, exactly like the plugin path.
    _merge_experts_inplace(ref_layer, grouped, freq, freq_weighted=True)
    ref_target_device = ref_layer.layer_module.parameters().__next__().device
    ref_state = _distill_merged_group(
        layer_ref=ref_layer, centroid_id=0, members=[0, 1], freq=freq,
        pre_merge_weights=ref_pre, layer_inputs=layer_inputs,
        steps=distill_kwargs["expert_distill_steps"],
        lr=distill_kwargs["expert_distill_lr"],
        betas=distill_kwargs["expert_distill_betas"],
        plateau_steps=distill_kwargs["expert_distill_plateau_steps"],
        plateau_eps=distill_kwargs["expert_distill_plateau_eps"],
        token_cap=distill_kwargs["expert_distill_token_cap"],
        device=ref_target_device,
    )
    ref_banks = build_banks(ref_layer)
    ref_weights = {
        name: ref_banks[name].get(0).clone() for name in ("gate_proj", "up_proj", "down_proj")
    }

    # --- Plugin run: drive pre_merge_snapshot + merge through the hooks ---
    plug_model = copy.deepcopy(tiny_model)
    plug_layer = list(iter_moe_layers(plug_model))[0]

    class _StubAcc:
        """Minimal layer_input_acc — .get() returns the same buffer."""
        def __init__(self, buf):
            self._buf = buf

        def get(self):
            return self._buf

    plugin = _make_plugin(**distill_kwargs)
    ctx = PipelineContext()
    ctx.set("layer_ref", plug_layer)
    ctx.set("grouped", grouped)
    ctx.set("freq", freq)
    ctx.set("layer_input_acc", _StubAcc(layer_inputs.clone()))
    ctx.set("distill_state", None)  # LayerMergePlugin.merge default

    plugin.pre_merge_snapshot(ctx)
    assert ctx.get("pre_merge_weights") is not None

    # The orchestrator runs LayerMergePlugin.merge (which calls
    # _merge_experts_inplace) BEFORE the ExpertDistillPlugin.merge hook —
    # replicate that here so distillation starts from POST-MERGE centroid
    # weights, identically to the reference path above.
    _merge_experts_inplace(plug_layer, grouped, freq, freq_weighted=True)

    plugin.merge(ctx)

    distill_state = ctx.get("distill_state")
    assert distill_state is not None
    # Singleton group 2 is skipped (skip_singletons=True); group 0 distilled.
    assert set(distill_state.keys()) == {0}
    assert distill_state[0]["steps"] == ref_state["steps"]
    assert distill_state[0]["final_loss"] == ref_state["final_loss"]

    plug_banks = build_banks(plug_layer)
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert torch.equal(plug_banks[name].get(0), ref_weights[name]), (
            f"plugin-path bank weight {name} diverged from the "
            f"_distill_merged_group reference"
        )


# --- monkeypatch-drift guard (T9-T15 lesson) -----------------------------
def test_no_stale_monkeypatch_of_distill_symbols():
    """`_distill_merged_group` / `_snapshot_pre_merge_layer_experts` moved to
    pipeline.plugins.expert_distill in T16. Any test that patches either on the
    monolith namespace must also patch it on the new module (or the live
    plugin path drifts). Fails loudly otherwise.

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
