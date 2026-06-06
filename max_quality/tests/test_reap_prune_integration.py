"""Integration — tiny CPU end-to-end faithful_prune Stage 2 (§6 test 3).

Runs ``stage2.run()`` in ``prune_mode: faithful_prune`` on the tiny fused MoE
model with a tiny calibration batch (CPU only) and asserts:
  - the run completes and produces a checkpoint dir + merge_map.json,
  - every MoE layer's expert count dropped to the faithful target,
  - the merge machinery never ran (merge_map groups are all singletons; the
    merge JSON carries the faithful payload + ``pruned_expert_ids``),
  - no covariance accumulated (faithful mode collects none).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from moe_compress import stage1
from moe_compress.stage2 import orchestrator as stage2_reap_ream
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    Stage2ReapPayload,
    save_reap_scores,
)
from moe_compress.utils.model_io import iter_moe_layers


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _noop_save(model, tokenizer, path, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def _write_reap_sidecar(jsonl_path, n_layers, n_experts):
    """Write a --capture-reap-scores sidecar with descending per-layer scores.

    Score for expert e = (n_experts − e), so expert 0 is highest-saliency and
    expert n−1 is lowest → the faithful pruner drops the highest-index experts.
    Deterministic so the golden / parity assertions are stable.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("", encoding="utf-8")  # placeholder JSONL
    row = torch.tensor([float(n_experts - e) for e in range(n_experts)])
    scores = row.unsqueeze(0).repeat(n_layers, 1).contiguous()
    payload = Stage2ReapPayload(
        schema_version=SCHEMA_VERSIONS["reap_scores"],
        n_experts=n_experts,
        n_layers=n_layers,
        reap_scores=scores,
        token_counts=torch.full((n_layers, n_experts), 11, dtype=torch.int64),
    )
    save_reap_scores(payload, jsonl_path)


@pytest.fixture
def faithful_config(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["stage2_reap_ream"]["prune_mode"] = "faithful_prune"
    cfg["stage2_reap_ream"]["prune_fraction"] = 0.5  # 8 experts → drop 4, keep 4
    # Source REAP scores from a --capture-reap-scores sidecar (faithful mode
    # FAILS LOUD without it — it never runs its own HF rescore).
    jsonl = tmp_path / "self_traces.jsonl"
    cfg["calibration"]["jsonl_path"] = str(jsonl)
    _write_reap_sidecar(jsonl, n_layers=2, n_experts=8)
    return cfg


@pytest.fixture
def patched_calib(monkeypatch):
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

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)


def _run_stage1(model, config, tmp_path):
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=6,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, _TinyTokenizer(), config, tmp_path, decomp)


def test_faithful_prune_end_to_end(faithful_config, patched_calib, tmp_path):
    from .conftest import _TinyModel

    torch.manual_seed(0)
    model = _TinyModel(hidden=16, intermediate=8, num_layers=2,
                       num_experts=8, top_k=1)
    n_experts_pre = next(iter_moe_layers(model)).num_routed_experts
    assert n_experts_pre == 8

    _run_stage1(model, faithful_config, tmp_path)

    out_dir = stage2_reap_ream.run(
        model, _TinyTokenizer(), faithful_config, tmp_path,
        device=None, no_resume=True,
    )
    assert out_dir == tmp_path / "stage2_pruned"
    assert out_dir.is_dir()

    # n_prune = int(8 * 0.5) = 4 → keep 4 per (homogeneous) layer.
    for ref in iter_moe_layers(model):
        assert ref.router.weight.shape[0] == 4
        assert ref.num_routed_experts == 4
        assert ref.experts_module.gate_up_proj.shape[0] == 4
        assert ref.experts_module.down_proj.shape[0] == 4

    # merge_map: all groups singletons (no merges happened). The finalize step
    # wraps the bare dict under a "merge_map" envelope (S-2 SVC audit).
    envelope = json.loads((out_dir / "merge_map.json").read_text())
    merge_map = envelope["merge_map"]
    assert merge_map
    for _layer_key, groups in merge_map.items():
        assert isinstance(groups, dict)
        for _centroid, members in groups.items():
            assert len(members) == 1, "faithful prune must emit singleton groups"


def test_faithful_prune_merge_json_payload(faithful_config, patched_calib,
                                           tmp_path, monkeypatch):
    """The per-layer merge JSON carries the faithful payload + pruned_expert_ids,
    and the partial dir holds BOTH merge_{idx}.json AND the sentinel layer_{idx}.pt.
    """
    from .conftest import _TinyModel

    torch.manual_seed(0)
    model = _TinyModel(hidden=16, intermediate=8, num_layers=2,
                       num_experts=8, top_k=1)
    _run_stage1(model, faithful_config, tmp_path)

    # resume mode (no_resume=False) → partial dir is written. Keep it past
    # finalize (run() rmtree's it on success) so we can inspect the artifacts.
    monkeypatch.setenv("MOE_KEEP_STAGE2_PARTIAL", "1")
    stage2_reap_ream.run(model, _TinyTokenizer(), faithful_config, tmp_path,
                         device=None)

    partial = tmp_path / "_stage2_partial"
    assert (partial / "merge_0.json").is_file()
    assert (partial / "layer_0.pt").is_file(), "sentinel cov .pt must exist"

    data = json.loads((partial / "merge_0.json").read_text())
    assert data["format_version"] == 2
    # descending sidecar scores → keep top-4 (experts 0-3), drop 4-7.
    assert data["final_kept_ids"] == [0, 1, 2, 3]
    assert "pruned_expert_ids" in data
    assert data["pruned_expert_ids"] == [4, 5, 6, 7]
    # freq covers the FULL original expert set (resume derives n_pre_merge).
    assert set(int(k) for k in data["freq"].keys()) == set(range(8))
    # grouped + merge_map are singletons.
    for members in data["merge_map_layer"].values():
        assert len(members) == 1
    # the sentinel .pt is empty-but-valid.
    payload = torch.load(partial / "layer_0.pt", weights_only=False)
    assert payload["format_version"] == 1
    assert payload["covariance"] == {}
    assert payload["tokens"] == {}
