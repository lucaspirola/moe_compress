"""Crash-resume tests for Stage 5 (Router KD).

Verifies that periodic checkpoints are written at the correct interval and
that a resume correctly restores router weights + optimizer state and skips
already-processed batches.

Stage 5 uses the student as its own teacher (no live teacher, no logits cache)
so these tests run on CPU without loading any real model.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from moe_compress import stage0_super_experts, stage1_grape, stage2_reap_ream
from moe_compress import stage5_router_kd
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
def patched_stage5(monkeypatch, tiny_config):
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
    monkeypatch.setattr(stage5_router_kd, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage5_router_kd, "save_compressed_checkpoint", _noop_save)

    # Stage 5 needs a merge_map. Build one from stages 0+1+2.
    cfg = dict(tiny_config)
    return cfg


def _prepare_model_and_merge_map(model, config, tmp_path, monkeypatch):
    """Run stages 0+1+2 to produce merge_map.json at tmp_path/stage2_pruned/."""
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

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)

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

    # Overwrite merge_map.json with a trivial identity map (each new expert → itself).
    # Stage 2's merge_map maps new_idx → [original_expert_ids], but when teacher = student
    # (same post-stage2 model), _pool_teacher_logits would index original expert IDs into
    # a tensor whose last dim = num_new_experts — causing an out-of-bounds error. A trivial
    # identity map (each expert maps to itself) avoids the pooling step entirely.
    moe_layer_refs = list(iter_moe_layers(model))
    trivial_map = {
        str(ref.layer_idx): {str(i): [i] for i in range(ref.num_routed_experts)}
        for ref in moe_layer_refs
    }
    (tmp_path / "stage2_pruned").mkdir(parents=True, exist_ok=True)
    (tmp_path / "stage2_pruned" / "merge_map.json").write_text(json.dumps(trivial_map))


def _make_stage5_config(base_config: dict, ckpt_every: int = 0) -> dict:
    """Return a copy of base_config with stage5 set up for fast testing.

    Uses the student as the teacher by omitting teacher_load_in_4bit and
    teacher_logits_cache. The student model also serves as the teacher in
    tests via capture_router_outputs — any KD signal is valid for wiring.
    """
    import copy as _copy
    cfg = _copy.deepcopy(base_config)
    s5 = cfg["stage5_router_kd"]
    s5["epochs"] = 1
    s5["batch_size"] = 1
    s5["gradient_accumulation"] = 1
    s5["max_calibration_samples"] = 4   # 4 batches of size 1 = 4 optimizer steps
    s5["checkpoint_every_n_steps"] = ckpt_every
    s5.pop("teacher_load_in_4bit", None)
    s5.pop("teacher_logits_cache", None)
    return cfg


def _run_stage5_self_kd(student, config, tmp_path, monkeypatch):
    """Run Stage 5 with the student acting as its own teacher via hook capture.

    NOTE: teacher and student are the same Python object. After the first optimizer
    step the teacher (= updated student) always produces the same logits as the
    student, making KL divergence = 0. This is intentional in these tests — we
    are testing the crash-resume wiring, not the quality of the KD signal.
    """
    from moe_compress.utils.activation_hooks import capture_router_outputs

    original_run = stage5_router_kd.run

    def _patched_run(student, tokenizer, config, artifacts_dir, *, device=None):
        # Monkeypatch load_model inside stage5 so teacher = student.
        from moe_compress.utils import model_io as mio
        original_load = mio.load_model

        def _load_student(*args, **kwargs):
            return student, tokenizer

        monkeypatch.setattr(mio, "load_model", _load_student)
        monkeypatch.setattr(stage5_router_kd, "load_model", _load_student)
        try:
            return original_run(student, tokenizer, config, artifacts_dir, device=device)
        finally:
            monkeypatch.setattr(mio, "load_model", original_load)
            monkeypatch.setattr(stage5_router_kd, "load_model", original_load)

    return _patched_run(student, _TinyTokenizer(), config, tmp_path)


def test_stage5_checkpoint_written_at_interval(
    tiny_model, patched_stage5, tmp_path, monkeypatch
):
    """With checkpoint_every_n_steps=1, checkpoints exist after each optimizer step."""
    _prepare_model_and_merge_map(tiny_model, patched_stage5, tmp_path, monkeypatch)
    state_after_2 = copy.deepcopy(tiny_model.state_dict())

    cfg = _make_stage5_config(patched_stage5, ckpt_every=1)
    tiny_model.load_state_dict(state_after_2)

    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)

    partial_dir = tmp_path / "_stage5_partial"
    ckpts = sorted(partial_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    assert ckpts, "No checkpoint files found in _stage5_partial/"
    # With 4 steps and keep-last-2, we should have step_3.pt and step_4.pt.
    latest_step = int(ckpts[-1].stem.split("_")[1])
    assert latest_step == 4, f"Expected latest checkpoint at step 4, got step {latest_step}"
    assert len(ckpts) <= 2, f"Expected at most 2 checkpoints (keep-last-2), got {len(ckpts)}"

    # Validate checkpoint structure.
    payload = torch.load(ckpts[-1], map_location="cpu")
    assert payload["format_version"] == 1
    assert "router_state" in payload
    assert "optim_state" in payload
    assert "step" in payload
    assert "epoch" in payload
    assert "batch_idx" in payload


def test_stage5_resume_restores_router_weights(
    tiny_model, patched_stage5, tmp_path, monkeypatch
):
    """Resume restores router weights from the latest checkpoint."""
    _prepare_model_and_merge_map(tiny_model, patched_stage5, tmp_path, monkeypatch)
    state_after_2 = copy.deepcopy(tiny_model.state_dict())

    cfg = _make_stage5_config(patched_stage5, ckpt_every=1)
    tiny_model.load_state_dict(state_after_2)

    # First run: train to completion.
    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)

    # Read the final router weights from the latest checkpoint.
    partial_dir = tmp_path / "_stage5_partial"
    ckpts = sorted(partial_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    assert ckpts, "No checkpoint found after first run"
    latest_payload = torch.load(ckpts[-1], map_location="cpu")
    router_state_from_ckpt = {k: v.clone() for k, v in latest_payload["router_state"].items()}

    # Corrupt the model's router weights.
    tiny_model.load_state_dict(state_after_2)

    # Second run: resume from checkpoint.
    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)

    # After resume (all batches fast-forwarded), router weights should match checkpoint.
    for pname, expected in router_state_from_ckpt.items():
        parts = pname.split(".")
        obj = tiny_model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        actual = getattr(obj, parts[-1]).data.cpu()
        assert torch.allclose(actual, expected, atol=1e-6), (
            f"Router weight {pname} mismatch after resume: "
            f"max diff = {(actual - expected).abs().max().item():.2e}"
        )


def test_stage5_resume_step_counter_continues(
    tiny_model, patched_stage5, tmp_path, monkeypatch
):
    """Resume starts from the correct step number, not from 0."""
    _prepare_model_and_merge_map(tiny_model, patched_stage5, tmp_path, monkeypatch)
    state_after_2 = copy.deepcopy(tiny_model.state_dict())

    # 4 batches, ckpt_every=2 → checkpoints at steps 2 and 4.
    cfg = _make_stage5_config(patched_stage5, ckpt_every=2)
    tiny_model.load_state_dict(state_after_2)

    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)

    partial_dir = tmp_path / "_stage5_partial"
    ckpts = sorted(partial_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    assert ckpts
    latest_step = int(ckpts[-1].stem.split("_")[1])
    assert latest_step == 4, f"Last checkpoint should be at step 4, got {latest_step}"

    # Reset and resume — verify step counter restores correctly.
    tiny_model.load_state_dict(state_after_2)
    steps_seen: list[int] = []

    original_save = stage5_router_kd._save_stage5_checkpoint

    def _capture_save(partial_dir, step, epoch, batch_idx, student, optim):
        steps_seen.append(step)
        return original_save(partial_dir, step, epoch, batch_idx, student, optim)

    monkeypatch.setattr(stage5_router_kd, "_save_stage5_checkpoint", _capture_save)

    _run_stage5_self_kd(tiny_model, cfg, tmp_path, monkeypatch)

    # On resume from step=4, all 4 batches are fast-forwarded → no new steps → no saves.
    assert steps_seen == [], (
        f"Resume from step 4 should not write new checkpoints (all batches fast-forwarded), "
        f"but got checkpoint writes at steps: {steps_seen}"
    )
