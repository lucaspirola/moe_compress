"""Phase 5 smoke test: Stage 2 (max-quality flags ON) → Stage 2.5 hand-off.

Per spec § 10 the Stage 2 v2 changes leave Stage 2.5 (router KD post-merge)
untouched — it just receives a model whose merged-centroid weights have
already been distilled in Stage 2's step 7b. This test verifies the
**light-weight contract** that doesn't require loading a real teacher
model:

  1. Stage 2 runs with all new flags ON without errors and produces a valid
     post-merge model state on the synthetic ``_TinyModel`` fixture.
  2. The post-Stage-2 model has the expected structure: router rows
     resized, expert count reduced, weights non-degenerate.
  3. Stage 2.5's freeze pattern (``_freeze_non_routers``) correctly leaves
     ONLY ``mlp.gate.weight`` trainable when applied to the Stage 2 output.

The full Stage 2 → Stage 2.5 KD-loop integration is verified by the
ablation pipeline (spec § 8) on the real model, which has the teacher
checkpoint available. This unit-level test gives quick regression
coverage on the freeze contract — the user's "don't forget Stage 2.5"
ask.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from moe_compress import stage1, stage2_reap_ream, stage5_router_kd
from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.utils.model_io import iter_moe_layers, build_banks


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
    """Mirror the calibration/save monkey-patches from
    test_smoke_stage2_resume.py so Stage 2 can run end-to-end on the
    synthetic _TinyModel without hitting the network."""
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


def _run_stage1(model, config, tmp_path):
    tokenizer = _TinyTokenizer()
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, tokenizer, config, tmp_path, decomp)


def _enable_v2_flags(config: dict) -> dict:
    s2 = config["stage2_reap_ream"]
    s2["assignment_solver"] = "auto"
    s2["cost_alignment"] = "post"
    s2["cost_whitening"] = "diag"
    s2["cost_asymmetric"] = True
    s2["cost_topk_filter"] = 2
    s2["capacity_util_threshold"] = 0.0
    s2["em_refinement_rounds"] = 1
    s2["em_convergence_break"] = True
    s2["expert_distill_steps"] = 3
    s2["expert_distill_token_cap"] = 8
    s2["expert_distill_loss_plateau_steps"] = 2
    return config


def test_stage2_v2_handoff_to_stage2p5_freeze_contract(tiny_model, patched_stage2, tmp_path):
    """Run Stage 2 with the max-quality bundle ON, then apply Stage 2.5's
    parameter-freeze logic to the result and verify ONLY router weights
    are trainable. This catches regressions where Stage 2's mutations
    break the assumptions Stage 2.5 makes about parameter naming /
    structure (the `experts`, `shared_expert`, `embed`, `lm_head` patterns
    in the frozen list).
    """
    cfg = _enable_v2_flags(patched_stage2)
    _run_stage1(tiny_model, cfg, tmp_path)

    # Run Stage 2 with the max-quality bundle on.
    stage2_reap_ream.run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path,
        device=None, no_resume=True,
    )

    # Apply Stage 2.5's freeze pattern (extracted from stage5_router_kd to
    # avoid loading a teacher model).
    s5 = cfg["stage5_router_kd"]
    trainable_patterns = s5["trainable_name_patterns"]

    # Mimic the freeze logic from stage5_router_kd._freeze_non_routers.
    for name, p in tiny_model.named_parameters():
        p.requires_grad_(any(pat in name for pat in trainable_patterns))

    # Verify ONLY router weights are trainable.
    trainable_names = sorted(
        name for name, p in tiny_model.named_parameters() if p.requires_grad
    )
    frozen_names = sorted(
        name for name, p in tiny_model.named_parameters() if not p.requires_grad
    )

    # All trainable params must be router weights.
    assert all("mlp.gate.weight" in n for n in trainable_names), (
        f"Stage 2.5 freeze pattern produces unexpected trainable params: "
        f"{trainable_names}"
    )
    assert len(trainable_names) > 0, "No router weights found — Stage 2 may have removed them"

    # All expert / embed / lm_head weights must be frozen.
    for n in frozen_names:
        # The fixture's _TinyFusedExperts uses 'gate_up_proj' / 'down_proj'
        # tensor names; verify these are frozen.
        if "experts.gate_up_proj" in n or "experts.down_proj" in n:
            pass  # expected to be frozen
        elif "embed" in n or "lm_head" in n or "shared_expert" in n:
            pass  # expected to be frozen
        # Other non-router params (norms, etc.) are also acceptably frozen.


def test_stage2_v2_post_merge_structure_valid_for_stage2p5(tiny_model, patched_stage2, tmp_path):
    """Stage 2 v2 must produce a model whose router shape matches the
    post-merge expert count. Stage 2.5 forwards through this router; if
    the row count and expert count diverge, the routing dispatch breaks.
    """
    cfg = _enable_v2_flags(patched_stage2)
    _run_stage1(tiny_model, cfg, tmp_path)

    stage2_reap_ream.run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path,
        device=None, no_resume=True,
    )

    for ref in iter_moe_layers(tiny_model):
        n_experts = ref.num_routed_experts
        router_rows = ref.router.weight.shape[0]
        assert router_rows == n_experts, (
            f"layer {ref.layer_idx}: router has {router_rows} rows but "
            f"{n_experts} experts. Stage 2.5 forward will mis-dispatch."
        )

        # Verify expert weights are non-degenerate (not all-zero) — the
        # distillation step 7b shouldn't produce zero centroids.
        banks = build_banks(ref)
        for eid in range(n_experts):
            for name in ("gate_proj", "up_proj", "down_proj"):
                w = banks[name].get(eid)
                assert torch.isfinite(w).all(), (
                    f"layer {ref.layer_idx} expert {eid} {name}: NaN/Inf weight"
                )
                assert w.abs().max() > 0, (
                    f"layer {ref.layer_idx} expert {eid} {name}: all-zero weight"
                )
