"""ExpertDistillPlugin v2: skip-on-empty vs run-on-populated reservoir.

Plan §7.c (CRITICAL-1 / E-1). Verifies that the v2 distillation path is
gated correctly on ``layer_input_acc.buffer``: an empty reservoir logs a
warning and produces ``distill_state=None``; a populated reservoir runs
the per-merge-group distill loop and populates ``distill_state``.

These two cases bracket the user-visible behaviour change unlocked by
the new ``layer_in`` hook: pre-CRITICAL-1 production runs always had
empty reservoirs and silently skipped v2; post-CRITICAL-1 runs with
``--capture-layer-input-reservoir`` populate the buffer and exercise
the paper Eq. 2 per-token routing target.
"""
from __future__ import annotations

import copy
import logging
import pathlib

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.merging import _merge_experts_inplace
from moe_compress.stage2.plugins.expert_distill import ExpertDistillPlugin


# Reuse the tiny-model fixture from the shared conftest used by
# test_stage2_plugin_expert_distill.py. The fixture file's location is
# detected via the same import path; pytest discovers conftest.py
# automatically.


_KNOBS = dict(
    expert_distill_steps=2,
    expert_distill_lr=5e-3,
    expert_distill_betas=(0.9, 0.95),
    expert_distill_token_cap=8,
    expert_distill_skip_singletons=True,
    expert_distill_plateau_steps=100,
    expert_distill_plateau_eps=0.0,
    expert_distill_use_ce_term=False,
    expert_distill_ce_lambda=1.0,
    expert_distill_target_version="v2",
)


class _StubLayerInputAcc:
    """Mirror of ``_LayerInputAccumulator.get()`` for plugin tests."""

    def __init__(self, buf):
        self._buf = buf

    def get(self):
        return self._buf


def _build_ctx_for_layer(layer_ref, *, layer_input_buf, pre_merge_weights):
    """Construct the minimal ctx the merge hook reads."""
    ctx = PipelineContext()
    ctx.set("layer_ref", layer_ref)
    ctx.set("grouped", {0: [0, 1], 2: [2]})  # one non-singleton + one singleton
    ctx.set("freq", {0: 3, 1: 1, 2: 5})
    ctx.set("layer_input_acc", _StubLayerInputAcc(layer_input_buf))
    ctx.set("pre_merge_weights", pre_merge_weights)
    ctx.set("distill_state", None)  # LayerMergePlugin.merge default
    return ctx


def test_v2_skips_when_reservoir_empty(tiny_model, capsys):
    """Empty layer_input_acc.buffer → warning logged + distill_state stays None."""
    from moe_compress.stage2.plugins.expert_distill import (
        _snapshot_pre_merge_layer_experts,
    )
    from moe_compress.utils.model_io import iter_moe_layers

    model = copy.deepcopy(tiny_model)
    layer_ref = list(iter_moe_layers(model))[0]
    pre_merge_weights = _snapshot_pre_merge_layer_experts(layer_ref)
    _merge_experts_inplace(
        layer_ref, {0: [0, 1], 2: [2]}, {0: 3, 1: 1, 2: 5},
        freq_weighted=True,
    )

    ctx = _build_ctx_for_layer(
        layer_ref,
        layer_input_buf=None,  # empty reservoir
        pre_merge_weights=pre_merge_weights,
    )
    plugin = ExpertDistillPlugin(**_KNOBS)
    plugin.merge(ctx)

    assert ctx.get("distill_state") is None
    # The plugin logs a WARNING that names the cause; capsys captures it
    # from the stderr stream (the test repo's logging config writes there).
    captured = capsys.readouterr()
    assert (
        "no layer-input samples were captured" in captured.err
        or "no layer-input samples were captured" in captured.out
    )


def test_v2_runs_when_reservoir_populated(tiny_model):
    """Populated reservoir → v2 path runs; distill_state contains per-group entries."""
    from moe_compress.stage2.plugins.expert_distill import (
        _snapshot_pre_merge_layer_experts,
    )
    from moe_compress.utils.model_io import iter_moe_layers

    model = copy.deepcopy(tiny_model)
    layer_ref = list(iter_moe_layers(model))[0]
    pre_merge_weights = _snapshot_pre_merge_layer_experts(layer_ref)
    _merge_experts_inplace(
        layer_ref, {0: [0, 1], 2: [2]}, {0: 3, 1: 1, 2: 5},
        freq_weighted=True,
    )

    torch.manual_seed(2026)
    hidden = layer_ref.experts_module.hidden_dim
    layer_inputs = torch.randn(64, hidden) * 0.1

    ctx = _build_ctx_for_layer(
        layer_ref,
        layer_input_buf=layer_inputs,
        pre_merge_weights=pre_merge_weights,
    )
    plugin = ExpertDistillPlugin(**_KNOBS)
    plugin.merge(ctx)

    distill_state = ctx.get("distill_state")
    # Singletons are skipped (config knob); only centroid 0 ([0, 1]) ran.
    assert distill_state is not None
    assert isinstance(distill_state, dict)
    assert 0 in distill_state
    assert 2 not in distill_state  # singleton group skipped
    # Per-group state records the number of steps executed plus
    # initial / final losses (D-expert-distill-state-schema).
    state = distill_state[0]
    assert state["steps"] == _KNOBS["expert_distill_steps"]
    assert "initial_loss" in state and "final_loss" in state
