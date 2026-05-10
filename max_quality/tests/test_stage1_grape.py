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
    stage1_grape.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

    bl = json.loads((tmp_path / "stage1_blacklist.json").read_text())
    assert set(bl.keys()) == {
        "blacklist", "per_expert_max", "config",
        "blacklist_provenance", "dual_signal", "aimer", "sink_token",
    }


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


def test_phase_a_dual_signal_or_rule():
    """Phase A flags layer if EITHER residual OR MoE-output growth exceeds threshold."""
    from moe_compress.stage1_grape import _flag_layer_dual_signal

    # Layer 5: residual growth = 2.5 (below 3.0) but MoE growth = 2.5 (above 2.0) → flag
    assert _flag_layer_dual_signal(
        residual_ratio=2.5, moe_ratio=2.5,
        residual_threshold=3.0, moe_threshold=2.0,
    ) is True

    # Layer 6: residual growth = 4.0 (above 3.0), MoE growth below threshold → flag
    assert _flag_layer_dual_signal(
        residual_ratio=4.0, moe_ratio=1.5,
        residual_threshold=3.0, moe_threshold=2.0,
    ) is True

    # Layer 7: below both thresholds → don't flag
    assert _flag_layer_dual_signal(
        residual_ratio=2.5, moe_ratio=1.5,
        residual_threshold=3.0, moe_threshold=2.0,
    ) is False

    # Both above → flag (completes the OR truth table)
    assert _flag_layer_dual_signal(
        residual_ratio=4.0, moe_ratio=2.5,
        residual_threshold=3.0, moe_threshold=2.0,
    ) is True


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
        kwargs["moe_output_growth_ratio"] = 1.0e30
        result = real_detect(model, batches, moe_layers, device, **kwargs)
        L = result[0]
        captured["L"] = set(L)
        return result

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
    stage1_grape.run(tiny_model, _TinyTokenizer(), tiny_config, tmp_path, decomp)

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


def test_grape_greedy_merge_with_se_blacklist():
    """D-se-blacklist-merge contract (regression for code-vs-spec N-2).

    With a non-empty SE blacklist, _grape_greedy_merge must:
      (a) zero blacklisted rows/cols in D_work (so SEs never become i_star/j_star),
      (b) compute effective_budget = global_budget − total_SEs (subtract SE slots),
      (c) reduce per-layer floor by |SE_l| (floor applied to non-SE pool only).

    Build a 2-layer setup with N=4 experts each, blacklist {0:[1,2]}, and pick a
    global_budget that forces some merges so we can observe the contract.
    """
    n = 4
    # Layer 0: experts 0 and 3 are non-blacklisted; experts 1 and 2 are SE.
    # Make non-SE pair (0, 3) the most similar so merge selects j_star = 3 (or 0).
    D0 = torch.full((n, n), 0.9, dtype=torch.float32)
    D0[0, 3] = D0[3, 0] = 0.05  # very similar non-SE pair
    D0.fill_diagonal_(0.0)
    # Layer 1: all distances roughly equal; only one merge should happen here.
    D1 = torch.full((n, n), 0.6, dtype=torch.float32)
    D1.fill_diagonal_(0.0)

    blacklist = {0: [1, 2]}
    per_layer_counts = {0: n, 1: n}
    # global_budget counts TOTAL surviving experts including blacklisted; pick 6
    # so effective_budget = 6 − 2 = 4 → must drop from (4-2)+4 = 6 to 4 non-bl.
    # Per-layer floors: layer 0 → max(4//2 − 2, 0) = 0; layer 1 → max(4//2 − 0, 0) = 2.
    budgets = stage1_grape._grape_greedy_merge(
        D_matrices={0: D0, 1: D1},
        global_budget=6,
        per_layer_counts=per_layer_counts,
        blacklist=blacklist,
        gamma=1.0,  # disable entropy gate so the merge actually proceeds
    )

    # (a) SE rows/cols zeroed: SEs (1, 2) must NOT have been selected as j_star.
    #     Survivors include SEs unconditionally; total surviving in each layer ≥ |SE_l|.
    assert budgets[0] >= 2, f"layer 0 must keep both SEs; got {budgets[0]}"

    # (b) Total surviving experts must equal effective_budget + total_SEs = 4 + 2 = 6.
    #     But achieved may be larger if floors block — verify it's at least met from non-SE side.
    total_surviving = sum(budgets.values())
    assert total_surviving == 6, (
        f"sum(budgets)={total_surviving} but global_budget=6 with bl={blacklist}; "
        f"effective_budget should be 6 − 2 = 4 non-blacklisted survivors plus 2 SEs"
    )

    # (c) Layer 1 floor is 4 // 2 = 2 (no SEs); layer 0 floor is max(2 − 2, 0) = 0
    #     so layer 0 may drop to just its 2 SEs if entropy permits.
    assert budgets[1] >= 2, f"layer 1 floor=2 (n//2 with no SE); got budgets[1]={budgets[1]}"


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


def test_phase_c_candidate_set_union():
    """_collect_candidates unions four detectors with multi-source provenance tags.

    Synthetic single-layer (l=0) scenario:
      - expert 7: huge magnitude → flagged by phase_c (three-way AND) AND magnitude_topk
      - expert 5: moderate magnitude in top-K, low AIMER → flagged by aimer + magnitude_topk
      - expert 3: small magnitude, sink-token-dominated → flagged by sink_token only
      - expert 8: moderate-low magnitude in top-K but no other signal → magnitude_topk only
      - other experts (0..N) act as the population from which P99.5/a_max are computed.
    """
    from types import SimpleNamespace

    L = {0}
    # 200 filler experts in layer 0, magnitudes near 1.0 (population for percentile).
    per_expert_max: dict[tuple[int, int], float] = {(0, e): 1.0 for e in range(200)}
    per_expert_max[(0, 7)] = 100.0   # huge — passes three-way AND
    per_expert_max[(0, 5)] = 5.0     # moderate — top-K only
    per_expert_max[(0, 8)] = 4.0     # moderate — top-K only
    per_expert_max[(0, 3)] = 0.5     # small — sink-token only

    p995, a_max = stage1_grape._compute_se_thresholds(per_expert_max, L)
    a_max_threshold = 0.1 * a_max

    # AIMER: only expert 5 has a "concentrated" (low) score; all others are 1.0.
    # pct=0.005 → k=max(1, round(200*0.005))=1, so the bottom-pct picks only e=5.
    aimer_scores = {(0, e): 1.0 for e in range(200)}
    aimer_scores[(0, 5)] = 0.001
    bottom_pct = stage1_grape.aimer_bottom_pct_per_layer(aimer_scores, pct=0.005)

    # Sink-token mock: only expert 3 is sink-dominated (high freq + high ratio).
    sink_acc = SimpleNamespace(
        mean_router_score_sink={(0, 3): 1.0, (0, 5): 0.05, (0, 7): 0.05},
        mean_router_score_normal={(0, 3): 0.05, (0, 5): 1.0, (0, 7): 1.0},
        freq_on_sink={(0, 3): 1.0, (0, 5): 0.0, (0, 7): 0.0},
    )

    candidates = stage1_grape._collect_candidates(
        per_expert_max=per_expert_max,
        L=L,
        p995=p995, a_max=a_max, a_max_threshold=a_max_threshold,
        aimer_scores=aimer_scores,
        bottom_pct_by_layer=bottom_pct,
        aimer_enabled=True,
        aimer_layer_max_fraction=0.1,
        sink_acc=sink_acc,
        sink_enabled=True,
        sink_score_ratio=10.0,
        sink_freq_threshold=0.99,
        sink_max_per_layer_cap=10,
        magnitude_topk_per_l_layer=3,
    )

    assert candidates[(0, 7)] == ["magnitude_topk", "phase_c"]
    assert candidates[(0, 5)] == ["aimer", "magnitude_topk"]
    assert candidates[(0, 8)] == ["magnitude_topk"]
    assert candidates[(0, 3)] == ["sink_token"]
    # No spurious entries beyond these four.
    assert set(candidates.keys()) == {(0, 7), (0, 5), (0, 8), (0, 3)}
