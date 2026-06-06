"""Golden snapshot — faithful REAP prune selection + sliced weights (§6 test 5).

Deterministic tiny model + fixed REAP scores → assert ``final_kept_ids`` per
layer and the byte-hash of the sliced ``gate.weight`` / ``gate_up_proj`` match a
committed golden. Regenerate on purpose with ``REGEN_REAP_PRUNE_GOLDEN=1``.

Plus an inert-by-default guard: ``ReapPrunePlugin.is_enabled`` is False for the
default (merge) config, proving the plugin never perturbs the merge path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.reap_prune import ReapPrunePlugin
from moe_compress.utils.model_io import iter_moe_layers

_GOLDEN = Path(__file__).parent / "golden" / "reap_prune" / "selection.json"


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().to(torch.float64).numpy().tobytes()
    ).hexdigest()


def _run_prune(model, prune_fraction, scores_per_layer):
    """Drive compute_assignment + post_merge per layer; return the record."""
    plugin = ReapPrunePlugin(
        prune_fraction=prune_fraction, blacklist={}, merge_map={},
        partial_dir=None,
    )
    record = {}
    for ref in iter_moe_layers(model):
        ctx = PipelineContext()
        ctx.set("layer_ref", ref)
        ctx.set("scores", np.asarray(scores_per_layer[ref.layer_idx]))
        ctx.set("n_experts", ref.num_routed_experts)
        # Sentinel payload (fail-loud guard gates on this, not on ``scores``).
        ctx.set("reap_scores_payload", object())
        plugin.compute_assignment(ctx)
        plugin.merge(ctx)
        plugin.post_merge(ctx)
        record[str(ref.layer_idx)] = {
            "final_kept_ids": list(ctx.get("final_kept_ids")),
            "pruned_expert_ids": list(ctx.get("pruned_expert_ids")),
            "gate_weight_sha": _sha(ref.router.weight),
            "gate_up_proj_sha": _sha(ref.experts_module.gate_up_proj),
        }
    return record


def test_reap_prune_golden_selection_and_slice():
    from .conftest import _TinyModel

    torch.manual_seed(7)
    model = _TinyModel(hidden=16, intermediate=8, num_layers=2,
                       num_experts=8, top_k=1)
    # Fixed, distinct scores per layer (no ties) → deterministic selection.
    scores_per_layer = {
        0: [0.10, 0.90, 0.30, 0.70, 0.50, 0.20, 0.80, 0.40],
        1: [0.85, 0.15, 0.65, 0.25, 0.95, 0.05, 0.45, 0.55],
    }
    record = _run_prune(model, prune_fraction=0.5, scores_per_layer=scores_per_layer)

    if os.environ.get("REGEN_REAP_PRUNE_GOLDEN") == "1":
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(record, indent=2), encoding="utf-8")

    assert _GOLDEN.is_file(), (
        "golden missing — regenerate with REGEN_REAP_PRUNE_GOLDEN=1"
    )
    golden = json.loads(_GOLDEN.read_text())
    assert record == golden, (
        "faithful prune golden mismatch — selection or sliced-weight hash "
        "changed; regenerate on purpose with REGEN_REAP_PRUNE_GOLDEN=1 if "
        "intended."
    )


def test_plugin_inert_by_default_merge_mode():
    """In the default (merge) config the pruner is_enabled=False — proves the
    new plugin is dropped by registry.enabled and the merge path is unperturbed.
    """
    plugin = ReapPrunePlugin(
        prune_fraction=0.0, blacklist={}, merge_map={}, partial_dir=None,
    )
    assert plugin.is_enabled({"stage2_reap_ream": {}}) is False
    assert plugin.is_enabled({"stage2_reap_ream": {"prune_mode": "merge"}}) is False
    assert plugin.is_enabled(
        {"stage2_reap_ream": {"prune_mode": "faithful_prune"}}
    ) is True
