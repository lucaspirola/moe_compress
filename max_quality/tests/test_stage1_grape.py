"""Stage 1 — GRAPE redundancy ranking on fused-experts fixture."""
from __future__ import annotations

import json

import numpy as np
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

    out = json.loads((tmp_path / "stage1_budgets.json").read_text())
    budgets = {int(k): v for k, v in out["per_layer_target_experts"].items()}
    assert sum(budgets.values()) == 5
    assert budgets[0] <= budgets[1]


def test_blacklist_schema_three_top_level_keys(tiny_model, tiny_config, tmp_path):
    """stage1_blacklist.json must have exactly {blacklist, per_expert_max, config}."""
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    bl = json.loads((tmp_path / "stage1_blacklist.json").read_text())
    assert set(bl.keys()) == {"blacklist", "per_expert_max", "config"}


def test_three_way_AND_criterion():
    """Construct a synthetic per-expert magnitude where exactly one expert satisfies all
    three conditions: P99.5 ∧ 0.1·a_max ∧ l ∈ L. That expert must be the only one in
    the resulting blacklist."""
    # L contains layer 5. Build a per_expert_max so:
    # - One expert (layer 5, expert 7) has very large magnitude (clear winner).
    # - Many other experts have small magnitudes (form the bulk of the population for
    #   stable P99.5 computation).
    # - One expert (layer 9, expert 3) has even larger magnitude but l=9 is NOT in L,
    #   so the three-way AND must reject it.
    L = {5}
    per_expert_max: dict[tuple[int, int], float] = {}
    # Populate layer 5 with 200 small-valued experts plus 1 outlier.
    for e in range(200):
        per_expert_max[(5, e)] = 1.0
    per_expert_max[(5, 7)] = 100.0  # winner: high enough to pass P99.5 + 0.1*a_max
    # Populate layer 9 with a magnitude > P99.5 but l=9 ∉ L → rejected.
    per_expert_max[(9, 3)] = 10000.0

    p995, a_max = stage1_grape._compute_se_thresholds(per_expert_max, L)
    a_max_threshold = 0.1 * a_max
    bl = stage1_grape._apply_paper_criterion(per_expert_max, L, p995, a_max_threshold)
    # Only layer 5 has any entries, and exactly one expert (the outlier) qualifies.
    assert bl == {5: [7]}


def test_ma_formation_fallback_when_dynamic_empty(tiny_model, tiny_config, tmp_path, monkeypatch):
    """If the dynamic detector finds nothing, the 0.75-depth fallback must populate L
    with the first-75% of MoE layer indices."""
    # Capture detector output via a trampoline: force ma_ratio + ma_growth_ratio so
    # high that no layer can satisfy them, then verify the returned L is the
    # 0.75-depth fallback set.
    captured: dict[str, set] = {}
    real_detect = stage1_grape._detect_ma_layers

    def _spy(model, batches, moe_layers, device, **kwargs):
        # Force impossible thresholds so dynamic detector returns ∅.
        kwargs["ma_ratio"] = 1.0e30
        kwargs["ma_growth_ratio"] = 1.0e30
        L = real_detect(model, batches, moe_layers, device, **kwargs)
        captured["L"] = set(L)
        return L

    monkeypatch.setattr(stage1_grape, "_detect_ma_layers", _spy)

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    # tiny_model has 2 layers; round(0.75 * 2) = 2 → fallback L = MoE layers with idx < 2 = {0, 1}.
    moe_indices = sorted(ref.layer_idx for ref in iter_moe_layers(tiny_model))
    total_layers = tiny_model.config.num_hidden_layers
    cutoff = round(0.75 * total_layers)
    expected = {li for li in moe_indices if li < cutoff}
    assert captured["L"] == expected, (
        f"fallback L={captured['L']} but expected {expected} "
        f"(cutoff={cutoff}, moe_indices={moe_indices})"
    )


def test_cka_distance_contract():
    """Given two identical CKA matrices, the 1 - CKA distance matrix must have all-zero
    diagonal."""
    # Construct two identical [n_tokens, d_out] matrices.
    n_tokens, d_out = 32, 16
    R = torch.randn(n_tokens, d_out)

    # Compute CKA(R, R) via the same mechanics used in _cka_distance_matrix.
    # By construction, CKA(X, X) = 1, so 1 - CKA = 0 on the diagonal.
    # We assert this directly via a 2-expert layer where both experts hold the same R.
    class _FakeAcc:
        def get_representations(self_, li, e):  # noqa: N805
            return R.clone()

    class _FakeRef:
        layer_idx = 0
        num_routed_experts = 2

    D = stage1_grape._cka_distance_matrix(_FakeAcc(), _FakeRef())
    # Diagonal must be zero (default-initialized by torch.zeros) — the contract.
    assert torch.allclose(torch.diag(D), torch.zeros(2))
    # Off-diagonal: identical reps → CKA ≈ 1 → distance ≈ 0.
    assert D[0, 1].item() == pytest.approx(0.0, abs=1e-4)
    assert D[1, 0].item() == pytest.approx(0.0, abs=1e-4)
