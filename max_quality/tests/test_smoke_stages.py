"""End-to-end smoke of Stages 0, 1, 2 on the tiny synthetic MoE.

The real orchestrator imports `datasets` to build the C4 calibration tensor
(expensive and requires network). We monkey-patch `build_calibration_tensor`
and `build_super_expert_slice` to return a small deterministic tensor so
the test runs in <5 seconds locally without touching the Hub.

Stages 3+ need LAPACK on CPU, which the locally-built PyTorch nightly was
compiled without, so we stop after Stage 2. The same stages run on real
hardware (HF Jobs) for Phase B supervision.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from moe_compress import stage0_super_experts, stage1_grape, stage2_reap_ream
from moe_compress.budget import solver


class _TinyTokenizer:
    """Minimal tokenizer stand-in for calibration payloads."""
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        # One arbitrary token per character, capped at vocab.
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}


@pytest.fixture
def patched_calibration(monkeypatch, tiny_config):
    """Replace C4 download with a tiny deterministic token tensor."""
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
    # Stages import these names directly — patch at the import site too.
    monkeypatch.setattr(stage0_super_experts, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    return tiny_config


def test_stage0_smoke(tiny_model, patched_calibration, tmp_path):
    stage0_super_experts.run(
        tiny_model, _TinyTokenizer(), patched_calibration, tmp_path, device=None,
    )
    payload = json.loads((tmp_path / "stage0_blacklist.json").read_text())
    assert "blacklist" in payload
    assert "per_expert_max" in payload
    # Either a small or empty blacklist is fine; what matters is the format.
    for layer_idx, experts in payload["blacklist"].items():
        assert isinstance(layer_idx, str)                  # JSON round-trips
        assert all(isinstance(e, int) for e in experts)


def test_stage1_smoke(tiny_model, patched_calibration, tmp_path):
    # Stage 1 depends only on the model, not on Stage 0's artifact.
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
    """Run Stage 0, 1, 2 in sequence and verify Stage 2's artifacts."""
    # 0
    stage0_super_experts.run(
        tiny_model, _TinyTokenizer(), patched_calibration, tmp_path, device=None,
    )
    # 1
    decomp = solver.BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.0,
        global_expert_budget=4,                     # keep 4 of 8 experts total
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(tiny_model, patched_calibration, tmp_path, decomp)
    # 2 — patch save_checkpoint to a no-op since tiny model isn't HF-saveable
    from moe_compress.utils import model_io as mio

    def _noop_save(model, tokenizer, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    mio.save_checkpoint = _noop_save                # type: ignore[assignment]
    stage2_reap_ream.save_checkpoint = _noop_save   # type: ignore[attr-defined]

    stage2_reap_ream.run(
        tiny_model, _TinyTokenizer(), patched_calibration, tmp_path,
        device=None,
    )
    # Verify each layer has the expected number of experts post-merge.
    budgets = {
        int(k): int(v)
        for k, v in json.loads((tmp_path / "stage1_budgets.json").read_text())[
            "per_layer_target_experts"
        ].items()
    }
    for li, layer in enumerate(tiny_model.model.layers):
        assert len(layer.mlp.experts) == budgets[li], (
            f"layer {li}: expected {budgets[li]} experts, got {len(layer.mlp.experts)}"
        )
        assert layer.mlp.gate.weight.shape[0] == budgets[li]
        assert layer.mlp.num_experts == budgets[li]

    # merge_map sanity
    mm_path = tmp_path / "stage2_pruned" / "merge_map.json"
    assert mm_path.exists(), "merge_map.json was not written"
    mm = json.loads(mm_path.read_text())
    for li, groups in mm.items():
        assert len(groups) == budgets[int(li)]

    # Covariance snapshot should exist and have post-remap keys.
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    assert cov_path.exists()
    cov = torch.load(cov_path, map_location="cpu")
    for (li, e, name), _tensor in cov["covariance"].items():
        assert 0 <= e < budgets[li], (
            f"covariance key has expert index {e} beyond layer {li}'s budget {budgets[li]}"
        )
