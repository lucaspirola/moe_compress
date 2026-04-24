"""Stage 1 — GRAPE: redundancy ranking and budget allocation."""
from __future__ import annotations

import pytest
import torch

from moe_compress import stage1_grape
from moe_compress.budget.solver import BudgetDecomposition


def test_highly_redundant_layer_gets_smaller_budget(tiny_model, tiny_config, tmp_path):
    # Make layer 0's experts near-identical (high redundancy) and layer 1's
    # experts diverse. GRAPE should allocate fewer experts to layer 0.
    with torch.no_grad():
        base_expert = tiny_model.model.layers[0].mlp.experts[0]
        for e in tiny_model.model.layers[0].mlp.experts[1:]:
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(e, name).weight.copy_(
                    getattr(base_expert, name).weight + 1e-4 * torch.randn_like(
                        getattr(base_expert, name).weight
                    )
                )
        for e in tiny_model.model.layers[1].mlp.experts:
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(e, name).weight.copy_(torch.randn_like(getattr(e, name).weight))

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,          # target: keep 5 of 8 total routed experts
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, tiny_config, tmp_path, decomp)

    import json
    out = json.loads((tmp_path / "stage1_budgets.json").read_text())
    budgets = {int(k): v for k, v in out["per_layer_target_experts"].items()}
    assert sum(budgets.values()) == 5
    # Layer 0 (redundant) should get ≤ layer 1 (diverse).
    assert budgets[0] <= budgets[1]
