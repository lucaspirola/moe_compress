"""Stage 1 — end-to-end contract tests via the plugin orchestrator.

These three tests exercise ``stage1.run()`` on the tiny-model fixture and
assert the immutable Stage 1 ↔ Stage 2 contract: GRAPE budget ordering and
the artifact JSON schemas (7 top-level keys, 15 inner ``config`` keys).
Symbol-level helper coverage lives in the per-plugin test files
(``test_stage1_plugin_*.py``).
"""
from __future__ import annotations

import json

import torch

from moe_compress import stage1
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
    stage1.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    out = json.loads((tmp_path / "stage1_budgets.json").read_text())
    budgets = {int(k): v for k, v in out["per_layer_target_experts"].items()}
    assert sum(budgets.values()) == 5
    assert budgets[0] <= budgets[1]


def test_blacklist_schema_seven_top_level_keys(tiny_model, tiny_config, tmp_path):
    """stage1_blacklist.json must have exactly the 7 documented top-level keys.

    Locks the v6 schema (blacklist is ablation-validated; aimer/sink_token blocks
    expose `candidates` not `auto_extended`; blacklist_provenance is a list per
    entry). Top-level key set unchanged from v5 — only inner shapes changed.
    """
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    bl = json.loads((tmp_path / "stage1_blacklist.json").read_text())
    assert set(bl.keys()) == {
        "blacklist", "per_expert_max", "config",
        "blacklist_provenance", "dual_signal", "aimer", "sink_token",
    }


def test_blacklist_inner_config_keys(tiny_model, tiny_config, tmp_path):
    """stage1_blacklist.json['config'] inner block must pin exactly these 15 keys.

    Locks the spec §4 Phase C/D config schema so a future addition/removal is
    caught here rather than silently broken by downstream consumers. Extended
    from 12 → 15 keys at the v6 candidates+ablation-filter rewrite (added
    sink_token_max_per_layer_cap, magnitude_topk_per_l_layer, ablation_filter_threshold).
    """
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    bl = json.loads((tmp_path / "stage1_blacklist.json").read_text())
    expected_keys = {
        "a_max_fraction",
        "ma_ratio",
        "ma_growth_ratio",
        "moe_output_growth_ratio",
        "ma_formation_layers",
        "p995_threshold",
        "a_max_absolute",
        "a_max_threshold",
        "aimer_bottom_pct",
        "aimer_layer_max_fraction",
        "sink_token_score_ratio",
        "sink_token_freq_threshold",
        "sink_token_max_per_layer_cap",
        "magnitude_topk_per_l_layer",
        "ablation_filter_threshold",
    }
    assert set(bl["config"].keys()) == expected_keys, (
        f"stage1_blacklist.json['config'] keys drift: "
        f"got {set(bl['config'].keys())}, expected {expected_keys}"
    )
