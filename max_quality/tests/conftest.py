"""Test fixtures for the MoE compression pipeline.

The main fixture builds a tiny synthetic MoE model (2 layers × 4 routed
experts × 1 shared expert, hidden=16, intermediate=8) that mirrors the
module structure Qwen3_5MoeSparseMoeBlock exposes. That lets us exercise
Stages 0–3 without downloading the real 35 B checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make the `src/` tree importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _TinyExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyMoEBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([
            _TinyExpert(hidden, intermediate) for _ in range(num_experts)
        ])
        self.shared_expert = _TinyExpert(hidden, intermediate)

    def forward(self, x):
        logits = self.gate(x)
        topk_vals, topk_idx = logits.softmax(dim=-1).topk(self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for e_idx, expert in enumerate(self.experts):
            mask = (topk_idx == e_idx)                      # [B, T, k]
            if not mask.any():
                continue
            tok_mask = mask.any(dim=-1)                     # [B, T]
            selected = x[tok_mask]                          # [N, hidden]
            if selected.numel() == 0:
                continue
            # Per-token gate weight = sum of gate vals over the slots where
            # this token picked expert e_idx (typically 0 or 1 per top-k).
            w = (topk_vals * mask.to(topk_vals.dtype)).sum(dim=-1)[tok_mask]
            out[tok_mask] = out[tok_mask] + expert(selected) * w.unsqueeze(-1)
        return out + self.shared_expert(x)


class _TinyLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int, num_experts: int, top_k: int):
        super().__init__()
        self.mlp = _TinyMoEBlock(hidden, intermediate, num_experts, top_k)

    def forward(self, x):
        return x + self.mlp(x)


class _TinyTower(nn.Module):
    def __init__(self, num_layers: int, hidden: int, intermediate: int,
                 num_experts: int, top_k: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _TinyLayer(hidden, intermediate, num_experts, top_k)
            for _ in range(num_layers)
        ])


class _TinyConfig:
    def __init__(self, num_experts: int, num_layers: int):
        self.num_hidden_layers = num_layers
        self.layer_types = ["full_attention"] * num_layers
        self.num_experts = num_experts
        self.text_config = self


class _TinyModel(nn.Module):
    def __init__(
        self,
        *,
        hidden: int = 16,
        intermediate: int = 8,
        num_layers: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(32, hidden)
        self.model = _TinyTower(num_layers, hidden, intermediate, num_experts, top_k)
        self.lm_head = nn.Linear(hidden, 32, bias=False)
        self.config = _TinyConfig(num_experts, num_layers)

    def forward(self, input_ids=None, labels=None, output_router_logits=False, **_ignored):
        x = self.embed(input_ids)
        rls: list[torch.Tensor] = []
        for layer in self.model.layers:
            # Capture router logits as a side effect.
            if output_router_logits:
                rls.append(layer.mlp.gate(x))
            x = layer(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                labels[..., 1:].reshape(-1),
                ignore_index=-100,
            )
        class _Out:
            pass
        out = _Out()
        out.logits = logits
        out.loss = loss
        out.router_logits = rls if output_router_logits else None
        return out


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return _TinyModel()


@pytest.fixture
def tiny_config():
    return {
        "model": {
            "name_or_path": "tiny",
            "revision": "main",
            "torch_dtype": "float32",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "load_in_4bit": False,
            "trust_remote_code": False,
        },
        "target": {
            "total_reduction_ratio": 0.25,
            "initial_expert_reduction": 0.25,
            "initial_svd_reduction": 0.10,
        },
        "calibration": {
            "dataset": "allenai/c4",
            "subset": "en",
            "split": "train",
            "seed": 0,
            "num_sequences": 8,
            "sequence_length": 16,
            "super_expert_num_samples": 4,
            "domain_mix": {"c4": 1.0, "math": 0.0, "code": 0.0},
            "math_dataset": "unused",
            "code_dataset": "unused",
        },
        "stage0_super_experts": {
            "zscore_threshold": 1.0,
            "max_blacklisted_per_layer": 1,
            "global_blacklist_cap_pct": 0.50,
        },
        "stage1_grape": {
            "similarity_metric": "cosine",
            "min_experts_per_layer": 2,
            "early_layer_bonus": 0,
            "early_layer_bonus_depth": 0,
            "include_shared_in_similarity": False,
            "target_total_experts_per_layer_avg": 3,
        },
        "stage2_reap_ream": {
            "batch_size": 1,
            "num_calibration_samples": 4,
            "reap_min_active_tokens": 1,
            "ream": {
                "gate_weight": 1.0,
                "expert_weight": 1.0,
                "hungarian": True,
                "frequency_weighted_merge": True,
            },
            "sequential_recompute": True,
            "per_layer_mse_sigma_threshold": 3.0,
            "per_layer_mse_bump_ratio": 0.10,
        },
        "stage3_svd": {
            "scope": "moe_experts_only",
            "d_rank": {"parameter_cost_omega_mode": "auto"},
            "swift_svd_plus": {
                "alpha_grid": [0.5],
                "validation_samples": 2,
                "metric": "wikitext2_ppl",
                "per_group_type": True,
            },
            "aa_svd": {"use_post_prune_inputs": True},
            "block_refine": {"enabled": False, "lbfgs_steps": 5, "lbfgs_history": 2,
                             "per_block_loss": "mse"},
        },
        "stage4_eora": {
            "per_expert": True,
            "compensation_budget_pct": 0.03,
            "eigenspace_rank_cap": 4,
        },
        "stage5_router_kd": {
            "optimizer": "adamw",
            "learning_rate": 5.0e-5,
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "max_sequence_length": 16,
            "kd_temperature": 1.0,
            "max_calibration_samples": 4,
            "trainable_name_patterns": ["mlp.gate.weight"],
            "frozen_name_patterns": ["experts", "shared_expert", "embed", "lm_head"],
            "enable_output_router_logits": True,
        },
        "stage6_validate": {
            "wikitext2": {"enabled": False, "dataset": "wikitext", "subset": "wikitext-2-raw-v1",
                          "split": "test", "sequence_length": 16},
            "zero_shot": {"enabled": False, "tasks": []},
            "generative": {"enabled": False},
            "thresholds": {
                "wikitext2_ppl_relative_max_increase": 0.05,
                "arc_c_absolute_max_drop": 1.0,
                "hellaswag_absolute_max_drop": 1.0,
                "humaneval_absolute_max_drop": 1.0,
                "math500_absolute_max_drop": 1.0,
                "measured_reduction_min": 0.0,
            },
        },
        "logging": {"level": "INFO", "log_every_n_steps": 5,
                    "save_intermediate_every_n_layers": 1},
    }
