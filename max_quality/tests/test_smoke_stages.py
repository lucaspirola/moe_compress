"""End-to-end Stages 0/1/2 on the fused-experts synthetic fixture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from moe_compress import stage0_super_experts, stage1_grape, stage2_reap_ream
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
    monkeypatch.setattr(stage0_super_experts, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    return tiny_config


def test_stage0_smoke(tiny_model, patched_calibration, tmp_path):
    stage0_super_experts.run(
        tiny_model, _TinyTokenizer(), patched_calibration, tmp_path, device=None,
    )
    payload = json.loads((tmp_path / "stage0_blacklist.json").read_text())
    assert "blacklist" in payload
    # Per-expert max entries should number num_layers × num_experts (2 × 4 = 8).
    assert len(payload["per_expert_max"]) == 8
    # Every entry should be a positive float (forward actually ran).
    assert all(v >= 0 for v in payload["per_expert_max"].values())


def test_stage1_smoke(tiny_model, patched_calibration, tmp_path):
    decomp = solver.BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=5,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, patched_calibration, tmp_path, decomp)
    payload = json.loads((tmp_path / "stage1_budgets.json").read_text())
    assert payload["global_budget"] == 5
    assert all(v >= 2 for v in payload["per_layer_target_experts"].values())


def test_stage2_smoke_full_chain(tiny_model, patched_calibration, tmp_path):
    # 0
    stage0_super_experts.run(
        tiny_model, _TinyTokenizer(), patched_calibration, tmp_path, device=None,
    )
    # 1
    decomp = solver.BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.0,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, patched_calibration, tmp_path, decomp)
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
