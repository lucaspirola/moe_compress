"""Stage 1 — GRAPE redundancy ranking on fused-experts fixture."""
from __future__ import annotations

import pytest
import torch

from moe_compress import stage1_grape
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.utils.model_io import build_banks, iter_moe_layers


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0
    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}
    def save_pretrained(self, *_args, **_kwargs):
        return None


def test_highly_redundant_layer_gets_smaller_budget(tiny_model, tiny_config, tmp_path):
    # Make layer 0's experts near-identical (high redundancy) by copying the
    # first expert's rows into the others directly on the fused tensors.
    with torch.no_grad():
        for ref in iter_moe_layers(tiny_model):
            banks = build_banks(ref)
            if ref.layer_idx == 0:
                W0 = banks["gate_proj"].get(0).clone()
                Wu = banks["up_proj"].get(0).clone()
                Wd = banks["down_proj"].get(0).clone()
                for e in range(1, ref.num_routed_experts):
                    banks["gate_proj"].set(e, W0 + 1e-4 * torch.randn_like(W0))
                    banks["up_proj"].set(e, Wu + 1e-4 * torch.randn_like(Wu))
                    banks["down_proj"].set(e, Wd + 1e-4 * torch.randn_like(Wd))
            else:  # layer 1: randomize every expert
                for e in range(ref.num_routed_experts):
                    for name in ("gate_proj", "up_proj", "down_proj"):
                        banks[name].set(e, torch.randn_like(banks[name].get(e)))

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    import json
    out = json.loads((tmp_path / "stage1_budgets.json").read_text())
    budgets = {int(k): v for k, v in out["per_layer_target_experts"].items()}
    assert sum(budgets.values()) == 5
    assert budgets[0] <= budgets[1]
