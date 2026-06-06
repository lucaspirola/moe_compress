"""Unit — faithful drop applies to fused tensors + router, NO rescale (§6 test 2).

Drives ``ReapPrunePlugin.compute_assignment`` + ``post_merge`` against the tiny
fused-experts model and asserts:
  - gate_up_proj / down_proj leading dim == kept count,
  - router (gate) weight rows == kept count, router.num_experts == kept,
  - the SURVIVING router rows are byte-equal to the original rows at the kept
    indices (NO post-drop rescale) — locks upstream "drop-only" fidelity
    (CerebrasResearch/reap ``prune.py:142``, no renorm).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.reap_prune import ReapPrunePlugin
from moe_compress.utils.model_io import iter_moe_layers


def _make_plugin(prune_fraction):
    return ReapPrunePlugin(
        prune_fraction=prune_fraction,
        blacklist={},
        merge_map={},
        partial_dir=None,
    )


def test_drop_slices_experts_and_router_no_rescale():
    from .conftest import _TinyModel  # local synthetic fused MoE

    torch.manual_seed(0)
    model = _TinyModel(hidden=16, intermediate=8, num_layers=1,
                       num_experts=8, top_k=1)
    layer_ref = next(iter_moe_layers(model))
    n_experts = layer_ref.num_routed_experts
    assert n_experts == 8

    # snapshot the pre-drop router weight rows.
    router = layer_ref.router
    orig_router_w = router.weight.detach().clone()
    orig_gate_up = layer_ref.experts_module.gate_up_proj.detach().clone()

    # fixed scores: drop the 2 lowest-saliency experts (n_prune = int(8*0.25)=2).
    scores = np.array([0.10, 0.90, 0.30, 0.70, 0.50, 0.20, 0.80, 0.40])
    plugin = _make_plugin(prune_fraction=0.25)

    ctx = PipelineContext()
    ctx.set("layer_ref", layer_ref)
    ctx.set("scores", scores)
    ctx.set("n_experts", n_experts)
    # Sentinel: a real sidecar payload was loaded (the fail-loud guard gates on
    # this slot, not on ``scores``). Direct unit test of the selection/drop
    # math, so any non-None payload satisfies the guard.
    ctx.set("reap_scores_payload", object())

    plugin.compute_assignment(ctx)
    final_kept_ids = list(ctx.get("final_kept_ids"))
    # bottom-2 by score are experts 0 (0.10) and 5 (0.20).
    assert final_kept_ids == [1, 2, 3, 4, 6, 7]
    assert list(ctx.get("pruned_expert_ids")) == [0, 5]

    plugin.merge(ctx)
    plugin.post_merge(ctx)

    kept = len(final_kept_ids)
    # expert tensors sliced
    assert layer_ref.experts_module.gate_up_proj.shape[0] == kept
    assert layer_ref.experts_module.down_proj.shape[0] == kept
    # router sliced + counts updated
    assert router.weight.shape[0] == kept
    assert router.num_experts == kept
    assert layer_ref.mlp.num_experts == kept

    # NO rescale: surviving router rows byte-equal to original rows at kept ids.
    expected_rows = orig_router_w[torch.tensor(final_kept_ids)]
    assert torch.equal(router.weight.detach(), expected_rows)

    # NO rescale on expert tensors either: kept rows byte-equal originals.
    expected_gate_up = orig_gate_up[torch.tensor(final_kept_ids)]
    assert torch.equal(
        layer_ref.experts_module.gate_up_proj.detach(), expected_gate_up
    )


def test_compute_assignment_fails_loud_without_scores():
    """Faithful mode FAILS LOUD if no REAP scores are available.

    Per the project decision: the pruner must source saliency from the vLLM
    --capture-reap-scores sidecar and must NOT silently run its own HF
    forward-pass rescore. With no 'scores' slot, compute_assignment raises a
    descriptive RuntimeError pointing the operator at the calibration step.
    """
    plugin = _make_plugin(prune_fraction=0.25)
    ctx = PipelineContext()

    class _Ref:
        layer_idx = 0

    ctx.set("layer_ref", _Ref())
    ctx.set("n_experts", 8)
    # NOTE: 'scores' is deliberately NOT set.
    with pytest.raises(RuntimeError, match="capture-reap-scores"):
        plugin.compute_assignment(ctx)


def test_fails_loud_on_full_schedule_with_no_sidecar():
    """Regression for the bypassable-guard HIGH (review of 53e0062).

    Reproduces the REAL cache-miss path: run the per-layer pre-assign schedule
    (ReapScoringPlugin.on_layer_setup → on_score) with NO sidecar payload, then
    compute_assignment. ReapScoringPlugin publishes an ALL-ZERO ``scores`` vector
    (its live ReapAccumulator is empty — faithful mode drops the on_profile
    feeder), so ``ctx.has("scores")`` is True. The guard MUST still fire because
    it now gates on ``reap_scores_payload`` (only set on a real sidecar hit),
    NOT on ``scores``. Without the fix the pruner would silently argsort(-zeros)
    and drop the highest-indexed experts.
    """
    from moe_compress.stage2.plugins.reap_scoring import ReapScoringPlugin
    from .conftest import _TinyModel

    torch.manual_seed(0)
    model = _TinyModel(hidden=16, intermediate=8, num_layers=1,
                       num_experts=8, top_k=1)
    layer_ref = next(iter_moe_layers(model))

    reap_scoring = ReapScoringPlugin()
    pruner = _make_plugin(prune_fraction=0.25)

    ctx = PipelineContext()
    ctx.set("layer_ref", layer_ref)
    ctx.set("n_experts", layer_ref.num_routed_experts)

    # Pre-assign schedule WITHOUT a sidecar: on_layer_setup builds an empty
    # ReapAccumulator, on_score finalizes it and publishes all-zero scores.
    reap_scoring.on_layer_setup(ctx)
    reap_scoring.on_score(ctx)
    assert ctx.has("scores"), "ReapScoringPlugin should have published scores"
    assert float(np.abs(ctx.get("scores")).sum()) == 0.0, (
        "empty accumulator must yield an all-zero scores vector (the bug input)"
    )
    assert not ctx.has("reap_scores_payload"), "no sidecar → no payload"

    # The guard must fire despite the published (all-zero) scores.
    with pytest.raises(RuntimeError, match="capture-reap-scores"):
        pruner.compute_assignment(ctx)
