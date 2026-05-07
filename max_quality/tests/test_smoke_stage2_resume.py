"""Crash-resume tests for Stage 2 (REAP + REAM).

Verifies that when Stage 2 is interrupted after completing some layers,
a re-run skips those layers (reads from _stage2_partial/) and produces
identical output to a clean run.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from moe_compress import stage1_grape, stage2_reap_ream
from moe_compress.budget.solver import BudgetDecomposition
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


@pytest.fixture
def patched_stage2(monkeypatch, tiny_config):
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

    return tiny_config


def _run_stages_01(model, config, tmp_path):
    tokenizer = _TinyTokenizer()
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(model, tokenizer, config, tmp_path, decomp)
    return decomp


def test_stage2_resume_skips_completed_layers(tiny_model, patched_stage2, tmp_path, monkeypatch):
    """Stage 2 interrupted after layer 0 resumes correctly and skips layer 0."""
    _run_stages_01(tiny_model, patched_stage2, tmp_path)

    # Save a deep copy of the model after stages 0+1 (pre-stage-2 state).
    # We use deepcopy instead of state_dict because the merge operation changes
    # tensor shapes (bank.select slices stacked tensors) and load_state_dict
    # requires exact shape matches.
    model_before_s2 = copy.deepcopy(tiny_model)

    moe_layers = list(iter_moe_layers(tiny_model))
    assert len(moe_layers) >= 2, "Need at least 2 MoE layers for this test"

    # --- First run: crash after layer 0 is fully processed ---
    original_profile = stage2_reap_ream._profile_layer
    call_count = [0]

    def _crashing_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs):
        # Forward-compatible kwargs handling (see test_stage2_resume_produces_same_merge_map).
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("simulated crash after layer 0")
        return original_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs)

    monkeypatch.setattr(stage2_reap_ream, "_profile_layer", _crashing_profile)

    with pytest.raises(RuntimeError, match="simulated crash after layer 0"):
        stage2_reap_ream.run(tiny_model, _TinyTokenizer(), patched_stage2, tmp_path, device=None)

    # Partial files for layer 0 must exist.
    partial_dir = tmp_path / "_stage2_partial"
    layer0_idx = moe_layers[0].layer_idx
    assert (partial_dir / f"merge_{layer0_idx}.json").exists(), \
        f"merge_{layer0_idx}.json not written before crash"
    assert (partial_dir / f"layer_{layer0_idx}.pt").exists(), \
        f"layer_{layer0_idx}.pt not written before crash"

    # Verify merge JSON structure.
    data = json.loads((partial_dir / f"merge_{layer0_idx}.json").read_text())
    # Stage 2 v2 (spec § 12.1): format_version bumped 1 → 2 for the new
    # assignment_solver / cost_alignment / EM / distill_state forensic fields.
    # No backward-compat shim — operators on a v1 partial must delete and re-run.
    assert data["format_version"] == 2
    assert "final_kept_ids" in data
    assert "grouped" in data
    assert "freq" in data
    assert "merge_map_layer" in data
    assert "assignment_solver_used" in data
    assert "cost_alignment_used" in data
    assert "em_rounds_completed" in data
    assert "distill_state" in data

    # --- Restore model to pre-crash (pre-stage-2) state using the deep copy ---
    model_for_resume = copy.deepcopy(model_before_s2)

    # --- Second run: resume (no crash) ---
    monkeypatch.setattr(stage2_reap_ream, "_profile_layer", original_profile)
    second_call_count = [0]

    def _counting_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs):
        # Forward-compatible kwargs handling: Phase 3 added ``layer_input_acc``.
        second_call_count[0] += 1
        return original_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs)

    monkeypatch.setattr(stage2_reap_ream, "_profile_layer", _counting_profile)

    stage2_reap_ream.run(model_for_resume, _TinyTokenizer(), patched_stage2, tmp_path, device=None)

    # Layer 0 must have been skipped (not re-profiled).
    assert second_call_count[0] == len(moe_layers) - 1, (
        f"Expected {len(moe_layers) - 1} profile calls on resume "
        f"(layer 0 skipped), got {second_call_count[0]}"
    )

    # Partial dir cleaned up on success.
    assert not partial_dir.exists(), "_stage2_partial/ not cleaned up after successful Stage 2"

    # Output artifacts must exist.
    assert (tmp_path / "stage2_pruned" / "merge_map.json").exists()
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    assert cov_path.exists()
    cov_payload = torch.load(cov_path, map_location="cpu")
    assert len(cov_payload["covariance"]) > 0, \
        "Covariance file is empty — spill must not have removed data from memory"


def test_stage2_partial_dir_cleaned_up_on_clean_run(tiny_model, patched_stage2, tmp_path):
    """A clean (no-crash) Stage 2 run must remove _stage2_partial/ on success."""
    _run_stages_01(tiny_model, patched_stage2, tmp_path)
    stage2_reap_ream.run(tiny_model, _TinyTokenizer(), patched_stage2, tmp_path, device=None)
    assert not (tmp_path / "_stage2_partial").exists(), \
        "_stage2_partial/ should not exist after a successful Stage 2 run"


def test_stage2_resume_produces_same_merge_map(tiny_model, patched_stage2, tmp_path, monkeypatch):
    """Merge map from a resumed run matches a clean run for non-resumed layers."""
    _run_stages_01(tiny_model, patched_stage2, tmp_path)
    model_before_s2 = copy.deepcopy(tiny_model)

    moe_layers = list(iter_moe_layers(tiny_model))
    if len(moe_layers) < 2:
        pytest.skip("Need ≥2 MoE layers")

    # Clean run first — capture merge_map (use fresh copy so model is unmodified).
    clean_model = copy.deepcopy(model_before_s2)
    stage2_reap_ream.run(clean_model, _TinyTokenizer(), patched_stage2, tmp_path, device=None)
    clean_merge_map = json.loads((tmp_path / "stage2_pruned" / "merge_map.json").read_text())

    # Crash-resume run: crash after layer 0, then resume.
    crash_model = copy.deepcopy(model_before_s2)
    original_profile = stage2_reap_ream._profile_layer
    call_count = [0]

    def _crash_after_first(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs):
        # Phase 3 added the optional ``layer_input_acc`` kwarg; accept any
        # forward-compatible kwargs via **kwargs so this fixture doesn't have
        # to be updated every time _profile_layer's signature grows.
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("crash")
        return original_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs)

    monkeypatch.setattr(stage2_reap_ream, "_profile_layer", _crash_after_first)
    with pytest.raises(RuntimeError, match="crash"):
        stage2_reap_ream.run(crash_model, _TinyTokenizer(), patched_stage2, tmp_path, device=None)

    # Restore to pre-stage-2 and complete the run.
    resume_model = copy.deepcopy(model_before_s2)
    monkeypatch.setattr(stage2_reap_ream, "_profile_layer", original_profile)
    stage2_reap_ream.run(resume_model, _TinyTokenizer(), patched_stage2, tmp_path, device=None)

    resume_merge_map = json.loads((tmp_path / "stage2_pruned" / "merge_map.json").read_text())

    # The merge maps should be identical (same deterministic computation).
    assert clean_merge_map == resume_merge_map, (
        "Merge map from resumed run differs from clean run — resume broke determinism"
    )


def test_stage2_resume_deletes_orphaned_pt(tiny_model, patched_stage2, tmp_path):
    """Orphaned layer_N.pt (no matching merge_N.json) must be deleted on resume."""
    _run_stages_01(tiny_model, patched_stage2, tmp_path)
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()

    # Simulate orphaned state: .pt exists, .json does not
    orphan = partial_dir / "layer_0.pt"
    torch.save({"format_version": 1, "covariance": {}, "tokens": {}}, orphan)
    assert orphan.exists()

    model, tokenizer = tiny_model, _TinyTokenizer()
    stage2_reap_ream.run(model, tokenizer, patched_stage2, tmp_path, device=None)

    # Orphan must be gone; layer was reprocessed
    assert not orphan.exists()
