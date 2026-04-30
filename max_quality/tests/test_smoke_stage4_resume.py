"""Crash-resume tests for Stage 4 (EoRA residual compensation).

Stage 4 layers are fully independent: each layer reads from immutable sidecars
(_stage3_original_weights.pt, _stage2_input_covariance.pt) and writes to the
model's FactoredExperts via widen_rank(). The partial dir stores the complete
post-widen state per layer so a resume can skip it.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from moe_compress import stage0_super_experts, stage1_grape, stage2_reap_ream, stage3_svd
from moe_compress import stage4_eora
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.utils.model_io import MATRIX_NAMES, FactoredExperts, iter_moe_layers


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
def patched_stages(monkeypatch, tiny_config):
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
    monkeypatch.setattr(stage3_svd, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage3_svd, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage4_eora, "save_compressed_checkpoint", _noop_save)

    return tiny_config


def _run_stages_0123(model, config, tmp_path):
    stage0_super_experts.run(model, _TinyTokenizer(), config, tmp_path, device=None)
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1_grape.run(model, _TinyTokenizer(), config, tmp_path, decomp)
    stage2_reap_ream.run(model, _TinyTokenizer(), config, tmp_path, device=None)
    stage3_svd.run(model, _TinyTokenizer(), config, tmp_path, decomp, device=None)
    return decomp


def test_stage4_resume_skips_completed_layers(
    tiny_model, patched_stages, tmp_path, monkeypatch
):
    """Stage 4 interrupted after layer 0 resumes correctly and skips layer 0."""
    _run_stages_0123(tiny_model, patched_stages, tmp_path)

    # Snapshot post-stage-3 model state for reset.
    state_after_3 = copy.deepcopy(tiny_model.state_dict())

    layers = [ref for ref in iter_moe_layers(tiny_model)
               if isinstance(ref.experts_module, FactoredExperts)]
    assert len(layers) >= 2, "Need at least 2 FactoredExperts layers for this test"

    # --- First run: crash after layer 0's spill is written ---
    # We intercept _spill_layer: let the first call succeed (layer 0 done),
    # then raise on the second call (layer 1 attempt).
    original_spill = stage4_eora._spill_layer
    spill_call_count = [0]

    def _crash_after_first_spill(partial_dir, layer_idx, fe, rank_map_layer, compensated):
        spill_call_count[0] += 1
        if spill_call_count[0] == 1:
            return original_spill(partial_dir, layer_idx, fe, rank_map_layer, compensated)
        raise RuntimeError("simulated crash after layer 0")

    monkeypatch.setattr(stage4_eora, "_spill_layer", _crash_after_first_spill)

    with pytest.raises(RuntimeError, match="simulated crash after layer 0"):
        stage4_eora.run(tiny_model, _TinyTokenizer(), patched_stages, tmp_path)

    partial_dir = tmp_path / "_stage4_partial"
    layer0_idx = layers[0].layer_idx
    assert (partial_dir / f"layer_{layer0_idx}.pt").exists(), \
        f"layer_{layer0_idx}.pt not written before crash"

    # Validate spill file structure.
    payload = torch.load(partial_dir / f"layer_{layer0_idx}.pt", map_location="cpu")
    assert payload["format_version"] == 1
    assert "ranks" in payload
    assert "rank_map_layer" in payload
    for name in MATRIX_NAMES:
        assert f"{name}_U" in payload
        assert f"{name}_V" in payload

    # --- Restore post-stage-3 model state ---
    tiny_model.load_state_dict(state_after_3)

    # --- Second run: resume without crash ---
    monkeypatch.setattr(stage4_eora, "_spill_layer", original_spill)
    resumed_spill_count = [0]

    def _counting_spill(partial_dir, layer_idx, fe, rank_map_layer, compensated):
        resumed_spill_count[0] += 1
        return original_spill(partial_dir, layer_idx, fe, rank_map_layer, compensated)

    monkeypatch.setattr(stage4_eora, "_spill_layer", _counting_spill)

    stage4_eora.run(tiny_model, _TinyTokenizer(), patched_stages, tmp_path)

    # Partial dir must be cleaned up.
    assert not partial_dir.exists(), "_stage4_partial/ not cleaned up after successful Stage 4"

    # Output files must exist.
    assert (tmp_path / "stage4_eora" / "eora_ranks.json").exists()

    # On resume, layer 0 was loaded from spill (not re-spilled), so _spill_layer
    # should only be called for remaining layers (len(layers) - 1 times).
    assert resumed_spill_count[0] == len(layers) - 1, (
        f"Expected {len(layers) - 1} spill calls on resume "
        f"(layer 0 skipped), got {resumed_spill_count[0]}"
    )

    fe = layers[0].experts_module
    assert isinstance(fe, FactoredExperts)


def test_stage4_partial_dir_cleaned_up_on_clean_run(
    tiny_model, patched_stages, tmp_path
):
    """A clean Stage 4 run must remove _stage4_partial/ on success."""
    _run_stages_0123(tiny_model, patched_stages, tmp_path)
    stage4_eora.run(tiny_model, _TinyTokenizer(), patched_stages, tmp_path)
    assert not (tmp_path / "_stage4_partial").exists(), \
        "_stage4_partial/ should not exist after a successful Stage 4 run"


def test_stage4_resume_restores_factored_experts_from_spill(
    tiny_model, patched_stages, tmp_path, monkeypatch
):
    """FactoredExperts restored from a spill file must match what was computed."""
    _run_stages_0123(tiny_model, patched_stages, tmp_path)
    state_after_3 = copy.deepcopy(tiny_model.state_dict())

    layers = [ref for ref in iter_moe_layers(tiny_model)
               if isinstance(ref.experts_module, FactoredExperts)]
    if not layers:
        pytest.skip("No FactoredExperts layers — EoRA added no rank for any layer")

    # Run Stage 4 to completion — capture FactoredExperts tensors from layer 0.
    stage4_eora.run(tiny_model, _TinyTokenizer(), patched_stages, tmp_path)
    layer0_idx = layers[0].layer_idx
    fe_after = layers[0].experts_module
    saved = {name: (getattr(fe_after, f"{name}_U").data.clone(),
                    getattr(fe_after, f"{name}_V").data.clone())
             for name in MATRIX_NAMES}

    # Reset to post-stage-3 and pre-seed layer 0's spill file manually.
    tiny_model.load_state_dict(state_after_3)
    partial_dir = tmp_path / "_stage4_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    fe0 = layers[0].experts_module
    from moe_compress.stage4_eora import _spill_layer
    _spill_layer(partial_dir, layer0_idx, fe0,
                 rank_map_layer={f"L{layer0_idx}_{n}": fe0.ranks[n] for n in MATRIX_NAMES},
                 compensated_params_layer=0)

    # Overwrite with the tensors from the clean run so the spill has the "correct" data.
    # Also update ranks to reflect the widened shape — the spill seeded above has pre-widen
    # ranks, but we're patching in post-widen tensors, so ranks must match U.shape[-1].
    import os
    payload = torch.load(partial_dir / f"layer_{layer0_idx}.pt", map_location="cpu")
    for name in MATRIX_NAMES:
        payload[f"{name}_U"] = saved[name][0].cpu()
        payload[f"{name}_V"] = saved[name][1].cpu()
        payload["ranks"][name] = int(saved[name][0].shape[-1])   # widened rank dim
    tmp_pt = partial_dir / f"layer_{layer0_idx}.pt.tmp"
    torch.save(payload, tmp_pt)
    os.replace(tmp_pt, partial_dir / f"layer_{layer0_idx}.pt")

    # Reset model again and re-run Stage 4 — layer 0 should come from the spill.
    tiny_model.load_state_dict(state_after_3)
    stage4_eora.run(tiny_model, _TinyTokenizer(), patched_stages, tmp_path)

    fe_resumed = layers[0].experts_module
    for name in MATRIX_NAMES:
        U_expected, V_expected = saved[name]
        U_actual = getattr(fe_resumed, f"{name}_U").data
        V_actual = getattr(fe_resumed, f"{name}_V").data
        assert torch.allclose(U_actual.cpu(), U_expected.cpu(), atol=1e-5), \
            f"L{layer0_idx}/{name}_U mismatch after resume from spill"
        assert torch.allclose(V_actual.cpu(), V_expected.cpu(), atol=1e-5), \
            f"L{layer0_idx}/{name}_V mismatch after resume from spill"
        # Critical invariant: fe.ranks[name] must match the actual tensor rank dim.
        assert fe_resumed.ranks[name] == U_actual.shape[-1], (
            f"L{layer0_idx}/{name}: fe.ranks={fe_resumed.ranks[name]} "
            f"but U.shape[-1]={U_actual.shape[-1]} — ranks dict inconsistent after resume"
        )


def test_stage4_double_widen_raises(
    tiny_model, patched_stages, tmp_path
):
    """Running Stage 4 twice on the same in-process model must raise AssertionError."""
    import shutil

    # Run stages 0-3 to get a model with FactoredExperts
    _run_stages_0123(tiny_model, patched_stages, tmp_path)

    layers = [ref for ref in iter_moe_layers(tiny_model)
              if isinstance(ref.experts_module, FactoredExperts)]
    if not layers:
        pytest.skip("No FactoredExperts layers — EoRA added no rank for any layer")

    # First Stage 4 run succeeds
    stage4_eora.run(tiny_model, _TinyTokenizer(), patched_stages, tmp_path)

    # Clear partial dir so the second run doesn't skip via resume
    partial_dir = tmp_path / "_stage4_partial"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)

    # Second in-process run must fail with double-widen detected
    with pytest.raises(AssertionError, match="double-widen"):
        stage4_eora.run(tiny_model, _TinyTokenizer(), patched_stages, tmp_path)
