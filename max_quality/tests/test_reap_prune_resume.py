"""Faithful-mode resume round-trip (§6 test 6 / finding #1).

Regression lock for the "resume silently re-runs every layer" bug. Faithful
mode collects no covariance, so ``_snapshot_cov_layer`` would never write the
``layer_{idx}.pt`` the ``resume.py:135`` both-files gate requires. The §5 fix
writes an empty sentinel ``.pt`` from ``ReapPrunePlugin.write_artifacts``.

Asserts, after a crash-after-layer-0 + resume:
  - ``merge_0.json`` AND the sentinel ``layer_0.pt`` both exist,
  - the second run RECOGNIZES layer 0 as completed (does not re-profile it),
  - ``cov_acc.load_layer_from_disk`` accepts the empty sentinel (no cov),
  - the resumed model's per-layer expert tensors + router are byte-equal to a
    clean (no-resume) run AND ``final_kept_ids`` matches. The byte-equality
    locks ``merging.py:162`` — replay runs ``_merge_experts_inplace`` (a no-op
    for faithful singletons ONLY because it skips ``len(members) <= 1`` groups).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from moe_compress import stage1
from moe_compress.stage2 import orchestrator as stage2_reap_ream
from moe_compress.stage2.plugins import reap_prune as reap_prune_mod
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    Stage2ReapPayload,
    save_reap_scores,
)
from moe_compress.utils.model_io import iter_moe_layers


def _write_reap_sidecar(jsonl_path, n_layers, n_experts):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("", encoding="utf-8")
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


@pytest.fixture
def faithful_config(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["stage2_reap_ream"]["prune_mode"] = "faithful_prune"
    cfg["stage2_reap_ream"]["prune_fraction"] = 0.5
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


def _snapshot_layers(model):
    """{layer_idx: (router_w, gate_up, down)} for byte-equality comparison."""
    out = {}
    for ref in iter_moe_layers(model):
        out[ref.layer_idx] = (
            ref.router.weight.detach().clone(),
            ref.experts_module.gate_up_proj.detach().clone(),
            ref.experts_module.down_proj.detach().clone(),
        )
    return out


def test_faithful_resume_round_trip(faithful_config, patched_calib, tmp_path,
                                    monkeypatch):
    from .conftest import _TinyModel

    torch.manual_seed(0)
    model = _TinyModel(hidden=16, intermediate=8, num_layers=2,
                       num_experts=8, top_k=1)
    _run_stage1(model, faithful_config, tmp_path)

    pre_s2 = copy.deepcopy(model)
    moe_layers = list(iter_moe_layers(model))
    assert len(moe_layers) >= 2

    # First run: crash after layer 0's drop is applied + artifacts written.
    # Faithful mode runs NO profiling forward (LayerMergePlugin.on_profile is
    # dropped; scores come from the sidecar), so we trigger the crash from the
    # router-resize primitive on the 2nd layer — layer 0 is fully processed
    # (its merge JSON + sentinel .pt already written by write_artifacts, which
    # runs after post_merge in the post-assign schedule).
    orig_resize = reap_prune_mod._resize_router_for_kept_experts
    resize_calls = [0]

    def _crashing_resize(layer_ref, kept_ids):
        resize_calls[0] += 1
        if resize_calls[0] > 1:
            raise RuntimeError("simulated crash after layer 0")
        return orig_resize(layer_ref, kept_ids)

    monkeypatch.setattr(reap_prune_mod, "_resize_router_for_kept_experts",
                        _crashing_resize)
    with pytest.raises(RuntimeError, match="simulated crash"):
        stage2_reap_ream.run(model, _TinyTokenizer(), faithful_config,
                             tmp_path, device=None)

    # Layer 0's artifacts (the §5 fix): merge JSON + sentinel .pt both present.
    partial = tmp_path / "_stage2_partial"
    assert (partial / "merge_0.json").is_file()
    assert (partial / "layer_0.pt").is_file(), (
        "faithful mode must write a sentinel layer_0.pt so resume.py:135 "
        "recognizes layer 0 as completed"
    )
    # The sentinel loads + accumulates no covariance (empty maps).
    from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
    _cov = InputCovarianceAccumulator()
    assert _cov.load_layer_from_disk(0, partial) is True
    assert not [k for k in _cov.covariance if k[0] == 0]

    # Resume: restart from the pre-Stage-2 snapshot. Spy on the pruner's
    # compute_assignment to prove layer 0 is REPLAYED (not re-selected).
    monkeypatch.setattr(reap_prune_mod, "_resize_router_for_kept_experts",
                        orig_resize)
    resume_model = copy.deepcopy(pre_s2)
    selected_layers = []
    orig_ca = reap_prune_mod.ReapPrunePlugin.compute_assignment

    def _spy_ca(self, ctx):
        selected_layers.append(ctx.get("layer_ref").layer_idx)
        return orig_ca(self, ctx)

    monkeypatch.setattr(reap_prune_mod.ReapPrunePlugin, "compute_assignment",
                        _spy_ca)
    stage2_reap_ream.run(resume_model, _TinyTokenizer(), faithful_config,
                         tmp_path, device=None)
    assert 0 not in selected_layers, "layer 0 must be replayed, not reprocessed"

    # Clean no-resume baseline from the same pre-S2 snapshot.
    clean_dir = tmp_path.parent / (tmp_path.name + "_clean")
    clean_dir.mkdir()
    clean_model = copy.deepcopy(pre_s2)
    _run_stage1(clean_model, faithful_config, clean_dir)
    stage2_reap_ream.run(clean_model, _TinyTokenizer(), faithful_config,
                         clean_dir, device=None, no_resume=True)

    # Byte-equal surviving expert tensors + router rows AND final_kept_ids.
    resumed = _snapshot_layers(resume_model)
    clean = _snapshot_layers(clean_model)
    assert set(resumed) == set(clean)
    for li in resumed:
        r_router, r_gu, r_dn = resumed[li]
        c_router, c_gu, c_dn = clean[li]
        assert torch.equal(r_router, c_router), f"layer {li} router diverged"
        assert torch.equal(r_gu, c_gu), f"layer {li} gate_up diverged"
        assert torch.equal(r_dn, c_dn), f"layer {li} down diverged"

    # final_kept_ids parity via the aggregate merge_map envelopes.
    r_map = json.loads(
        (tmp_path / "stage2_pruned" / "merge_map.json").read_text())["merge_map"]
    c_map = json.loads(
        (clean_dir / "stage2_pruned" / "merge_map.json").read_text())["merge_map"]
    assert r_map == c_map
