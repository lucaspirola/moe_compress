"""enable_student_training: scope -> requires_grad assignment.

CPU-only. Builds a tiny synthetic Qwen3.5MoE-shaped model with a real
``FactoredExperts`` so ``iter_moe_layers`` (called by enable_student_training)
recognises every layer as a MoE layer.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from structural_recovery.distillation import enable_student_training


# Skip the whole module if max_quality isn't on sys.path. conftest.py tries
# to add it, but if max_quality isn't checked out next to us we can't run
# these tests.
moe_compress = pytest.importorskip("moe_compress.utils.model_io",
                                    reason="max_quality must be on sys.path")
FactoredExperts = moe_compress.FactoredExperts


# ---------------------------------------------------------------------------
# Synthetic Qwen3.5MoE-like model
# ---------------------------------------------------------------------------


class _MoELayer(nn.Module):
    """Minimal stand-in for one Qwen3_5MoeDecoderLayer."""
    def __init__(self, hidden: int, intermediate: int, n_experts: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        # Attention (irrelevant beyond having params under self_attn).
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False)
        # MLP: router + FactoredExperts + shared expert.
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(hidden, n_experts, bias=False)   # router
        self.mlp.experts = FactoredExperts(
            num_experts=n_experts,
            hidden_dim=hidden,
            intermediate_dim=intermediate,
            ranks={"gate_proj": 4, "up_proj": 4, "down_proj": 4},
            dtype=torch.float32,
            device="cpu",
        )
        # Strategy A protects the shared expert; it must NOT be unfrozen
        # by ``experts_only`` (substring matching used to be a foot-gun).
        self.mlp.shared_expert = nn.Linear(hidden, hidden, bias=False)


class _SyntheticTower(nn.Module):
    """Qwen3.5MoE inner tower — owns ``.layers`` (what iter_moe_layers walks)."""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.norm = nn.LayerNorm(16)
        self.layers = nn.ModuleList([
            _MoELayer(hidden=16, intermediate=8, n_experts=4),
            _MoELayer(hidden=16, intermediate=8, n_experts=4),
        ])


class _SyntheticModel(nn.Module):
    """Wraps a tower + lm_head, mirroring AutoModelForCausalLM nesting."""
    def __init__(self):
        super().__init__()
        self.model = _SyntheticTower()
        self.lm_head = nn.Linear(16, 32, bias=False)


@pytest.fixture
def student() -> nn.Module:
    """Built fresh per test to avoid cross-test requires_grad bleed."""
    return _SyntheticModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _trainable_names(m: nn.Module) -> set[str]:
    return {n for n, p in m.named_parameters() if p.requires_grad}


def test_scope_full_marks_everything_trainable(student):
    enable_student_training(student, scope="full")
    all_names = {n for n, _ in student.named_parameters()}
    assert _trainable_names(student) == all_names


def test_scope_experts_only_targets_factored_banks_and_router(student):
    enable_student_training(student, scope="experts_only")
    trainable = _trainable_names(student)
    # Every FactoredExperts U/V bank should be trainable.
    for li in (0, 1):
        for mat in ("gate_proj", "up_proj", "down_proj"):
            for f in ("U", "V"):
                pname = f"model.layers.{li}.mlp.experts.{mat}_{f}"
                assert pname in trainable, f"{pname} should be trainable"
    # Router weight: trainable.
    for li in (0, 1):
        assert f"model.layers.{li}.mlp.gate.weight" in trainable

    # Frozen targets — Strategy A protects shared expert from compression
    # so it should match the teacher exactly and need no distillation.
    for li in (0, 1):
        assert f"model.layers.{li}.mlp.shared_expert.weight" not in trainable
        assert f"model.layers.{li}.self_attn.q_proj.weight" not in trainable
        assert f"model.layers.{li}.self_attn.o_proj.weight" not in trainable
        assert f"model.layers.{li}.input_layernorm.weight" not in trainable
        assert f"model.layers.{li}.input_layernorm.bias" not in trainable
    assert "model.embed_tokens.weight" not in trainable
    assert "model.norm.weight" not in trainable
    assert "lm_head.weight" not in trainable


def test_scope_factored_only_targets_uv_banks_only(student):
    enable_student_training(student, scope="factored_only")
    trainable = _trainable_names(student)
    # Six FactoredExperts banks per layer.
    for li in (0, 1):
        for mat in ("gate_proj", "up_proj", "down_proj"):
            for f in ("U", "V"):
                assert f"model.layers.{li}.mlp.experts.{mat}_{f}" in trainable
    # Router NOT trainable here (factored_only is stricter than experts_only).
    for li in (0, 1):
        assert f"model.layers.{li}.mlp.gate.weight" not in trainable
        assert f"model.layers.{li}.mlp.shared_expert.weight" not in trainable
        assert f"model.layers.{li}.self_attn.q_proj.weight" not in trainable
    assert "lm_head.weight" not in trainable


def test_scope_unknown_raises(student):
    with pytest.raises(ValueError, match="trainable_scope"):
        enable_student_training(student, scope="banana")


def test_count_returned_matches_actual(student):
    """The returned count should equal sum of numel for trainable params."""
    n = enable_student_training(student, scope="experts_only")
    actual = sum(p.numel() for p in student.parameters() if p.requires_grad)
    assert n == actual
