"""End-to-end Stages 1/2 on the fused-experts synthetic fixture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from moe_compress import stage1_grape, stage2_reap_ream
from moe_compress.budget import solver


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


@pytest.fixture
def patched_calibration(monkeypatch, tiny_config):
    from moe_compress.utils import calibration as cal_mod

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    return tiny_config


def test_stage1_smoke(tiny_model, patched_calibration, tmp_path):
    decomp = solver.BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, _TinyTokenizer(), patched_calibration, tmp_path, decomp)
    payload = json.loads((tmp_path / "stage1_budgets.json").read_text())
    # The output schema names this `requested_budget` (carries
    # decomposition.global_expert_budget); the test was stale from a rename.
    assert payload["requested_budget"] == 5
    assert all(v >= 2 for v in payload["per_layer_target_experts"].values())


def test_stage2_smoke_full_chain(tiny_model, patched_calibration, tmp_path):
    # 1
    decomp = solver.BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.0,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, _TinyTokenizer(), patched_calibration, tmp_path, decomp)
    # 2 — monkey-patch save_compressed_checkpoint to a no-op since tiny model
    # isn't HF-pretrained.
    from moe_compress.utils import model_io as mio

    def _noop_save(model, tokenizer, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    mio.save_compressed_checkpoint = _noop_save              # type: ignore[assignment]
    stage2_reap_ream.save_compressed_checkpoint = _noop_save  # type: ignore[attr-defined]

    stage2_reap_ream.run(
        tiny_model, _TinyTokenizer(), patched_calibration, tmp_path, device=None,
    )
    budgets = {
        int(k): int(v)
        for k, v in json.loads((tmp_path / "stage1_budgets.json").read_text())[
            "per_layer_target_experts"
        ].items()
    }
    for li, layer in enumerate(tiny_model.model.layers):
        if li not in budgets:
            continue  # non-MoE layer (dense MLP) — not in the budget dict
        assert layer.mlp.experts.num_experts == budgets[li]
        assert layer.mlp.experts.gate_up_proj.shape[0] == budgets[li]
        assert layer.mlp.experts.down_proj.shape[0] == budgets[li]
        assert layer.mlp.gate.weight.shape[0] == budgets[li]

    cov_path = tmp_path / "_stage2_input_covariance.pt"
    assert cov_path.exists()
    cov = torch.load(cov_path, map_location="cpu")
    for (li, e, name), _tensor in cov["covariance"].items():
        assert 0 <= e < budgets[li], (
            f"covariance key has expert index {e} beyond layer {li}'s budget {budgets[li]}"
        )


def test_stage2_max_merge_group_size_enforced(tiny_model, patched_calibration, tmp_path):
    """max_merge_group_size=2 must force the bump loop to keep enough experts
    that no centroid ends up with more than 2 members (itself + 1 child).

    Tiny model: 2 MoE layers × 4 experts.  We set global_budget=4 (2/layer)
    which means 2 non-centroids per layer; each centroid absorbs 1 child
    → max_group=2, which is exactly the cap — no bump expected.
    Then we tighten to global_budget=2 (1/layer) where 3 non-centroids would
    pile onto 1 centroid (max_group=4 > cap=2) → bump must fire and raise the
    effective target until max_group ≤ 2.
    """
    import copy

    from moe_compress.utils import model_io as mio

    model = copy.deepcopy(tiny_model)

    # Budget of 1 per layer (below min_experts_per_layer=2 in config, so we
    # override that key) to force very aggressive merging.
    cfg = copy.deepcopy(patched_calibration)
    cfg["stage1_grape"]["min_experts_per_layer"] = 1
    cfg["stage2_reap_ream"]["max_merge_group_size"] = 2

    decomp = solver.BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.0,
        global_expert_budget=2,   # 1 expert per MoE layer → max_group would be 4 without cap
        min_experts_per_layer=1,
        blacklisted_experts={},
    )
    stage1_grape.run(model, _TinyTokenizer(), cfg, tmp_path, decomp)

    def _noop_save(m, tok, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    mio.save_compressed_checkpoint = _noop_save              # type: ignore[assignment]
    stage2_reap_ream.save_compressed_checkpoint = _noop_save  # type: ignore[attr-defined]

    stage2_reap_ream.run(model, _TinyTokenizer(), cfg, tmp_path, device=None)

    # With max_merge_group_size=2: ceil(4 experts / 2) = 2 centroids minimum.
    for layer in model.model.layers:
        n = layer.mlp.experts.num_experts
        assert n >= 2, f"got {n} experts, expected ≥ 2 with max_merge_group_size=2"
