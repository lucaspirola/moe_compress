"""Test fixtures: synthetic MoE that mirrors Qwen3_5Moe's fused layout.

The pipeline targets fused experts — a single ``nn.Module`` per layer owns
all expert weights as stacked tensors. The fixture replicates that exactly:
``mlp.experts`` is a ``_TinyFusedExperts`` with ``gate_up_proj`` and
``down_proj`` parameters shaped ``[num_experts, 2·d_int, d_hid]`` and
``[num_experts, d_hid, d_int]`` respectively.

The fixture's forward implements the same sparse per-expert loop as the
reference ``Qwen3_5MoeExperts.forward`` so ``instrument_experts`` callbacks
fire with shapes that match the real model.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# Synthetic fused-experts module (structural twin of Qwen3_5MoeExperts)
# ---------------------------------------------------------------------------


class _TinyFusedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int, intermediate: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.intermediate_dim = intermediate
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate) * 0.02
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for e_idx in hit:
            e = e_idx[0]
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[e])
            sel = hidden_states[token_idx]
            gate_up = F.linear(sel, self.gate_up_proj[e])
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = self.act_fn(gate) * up
            down = F.linear(intermediate, self.down_proj[e])
            down = down * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, down.to(final.dtype))
        return final


class _TinyRouter(nn.Module):
    def __init__(self, num_experts: int, hidden: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.top_k = top_k
        self.weight = nn.Parameter(torch.randn(num_experts, hidden) * 0.02)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(hidden_states, self.weight)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)
        topv = topv.to(logits.dtype)
        return logits, topv, topi


class _TinyMoEBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = _TinyRouter(num_experts, hidden, top_k)
        self.experts = _TinyFusedExperts(num_experts, hidden, intermediate)
        # Minimal shared expert (unfused, protected).
        self.shared_expert = nn.Sequential()
        self.shared_expert.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.shared_expert.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.shared_expert.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        _, weights, indices = self.gate(flat)
        exp_out = self.experts(flat, indices, weights)
        return exp_out.reshape(B, T, H)


class _TinyLayer(nn.Module):
    def __init__(self, hidden, intermediate, num_experts, top_k):
        super().__init__()
        self.mlp = _TinyMoEBlock(hidden, intermediate, num_experts, top_k)

    def forward(self, x):
        return x + self.mlp(x)


class _TinyTower(nn.Module):
    def __init__(self, num_layers, hidden, intermediate, num_experts, top_k):
        super().__init__()
        self.layers = nn.ModuleList([
            _TinyLayer(hidden, intermediate, num_experts, top_k)
            for _ in range(num_layers)
        ])


class _TinyConfig:
    def __init__(self, num_experts, num_layers, hidden, intermediate, top_k):
        self.num_hidden_layers = num_layers
        self.layer_types = ["full_attention"] * num_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = top_k
        self.hidden_size = hidden
        self.moe_intermediate_size = intermediate
        self.text_config = self


class _TinyModel(nn.Module):
    def __init__(
        self, *,
        hidden: int = 16, intermediate: int = 8,
        num_layers: int = 2, num_experts: int = 4, top_k: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(32, hidden)
        self.model = _TinyTower(num_layers, hidden, intermediate, num_experts, top_k)
        self.lm_head = nn.Linear(hidden, 32, bias=False)
        self.config = _TinyConfig(num_experts, num_layers, hidden, intermediate, top_k)

    def forward(self, input_ids=None, labels=None, **_ignored):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                labels[..., 1:].reshape(-1),
                ignore_index=-100,
            )
        class _Out:
            pass
        out = _Out()
        out.logits = logits
        out.loss = loss
        return out


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return _TinyModel()


@pytest.fixture
def tiny_config():
    return {
        "model": {
            "name_or_path": "tiny", "revision": "main",
            "torch_dtype": "float32", "device_map": "cpu",
            "attn_implementation": "sdpa",
            "load_in_4bit": False, "trust_remote_code": False,
        },
        "target": {
            "total_reduction_ratio": 0.25,
            "initial_expert_reduction": 0.25,
            "initial_svd_reduction": 0.10,
        },
        "calibration": {
            # Tests never hit the real loader (monkey-patched), but the
            # spec_from_config parser requires these keys.
            "source": "c4-math-code",
            "dataset": "allenai/c4", "subset": "en", "split": "train",
            "seed": 0, "num_sequences": 8, "sequence_length": 16,
            "super_expert_num_samples": 4,
            "domain_mix": {"c4": 1.0, "math": 0.0, "code": 0.0},
            "math_dataset": "unused", "code_dataset": "unused",
        },
        "stage1_grape": {
            "num_calibration_samples": 4,
            "similarity_metric": "cosine", "min_experts_per_layer": 2,
            "early_layer_bonus": 0, "early_layer_bonus_depth": 0,
            "late_layer_bonus": 0, "late_layer_bonus_depth": 0,
            "target_total_experts_per_layer_avg": 3,
            "super_expert_detection": {
                "zscore_threshold": 1.0, "max_blacklisted_per_layer": 1,
                "global_blacklist_cap_pct": 0.50,
            },
        },
        "stage2_reap_ream": {
            "batch_size": 1, "num_calibration_samples": 4,
            "reap_min_active_tokens": 1,
            "covariance_storage_dtype": "float32",
            "max_merge_group_size": 0,
            "ream_cost_sigma_threshold": float("inf"),
            "ream_cost_bump_ratio": 0.10,
            "ream": {
                "hungarian": True, "frequency_weighted_merge": True,
            },
        },
        "stage3_svd": {
            "scope": "moe_experts_only",
            "d_rank": {"parameter_cost_omega_mode": "auto"},
            "swift_svd_plus": {
                "alpha_grid": [0.5], "validation_samples": 2,
                "metric": "wikitext2_ppl", "per_group_type": True,
                "alpha_search_min_host_ram_gb": 0.0,
            },
            "aa_svd": {"use_post_prune_inputs": True, "cross_covariance": False},
            "block_refine": {"enabled": False, "lbfgs_steps": 5,
                             "lbfgs_history": 2, "per_block_loss": "mse"},
        },
        "stage4_eora": {
            "per_expert": True, "compensation_budget_pct": 0.03,
            "eigenspace_rank_cap": 4,
        },
        "stage5_router_kd": {
            "optimizer": "adamw", "learning_rate": 5.0e-5, "epochs": 1,
            "batch_size": 1, "gradient_accumulation": 1,
            "max_sequence_length": 16, "kd_temperature": 1.0,
            "max_calibration_samples": 4,
            "trainable_name_patterns": ["mlp.gate.weight"],
            "frozen_name_patterns": ["experts", "shared_expert", "embed", "lm_head"],
            "enable_output_router_logits": True,
        },
        "stage6_validate": {
            "wikitext2": {"enabled": False, "dataset": "wikitext",
                          "subset": "wikitext-2-raw-v1", "split": "test",
                          "sequence_length": 16},
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


@pytest.fixture
def tiny_config_bf16(tiny_config):
    """Same as tiny_config but with bf16 covariance storage on both stages.

    Use for smoke runs that must continue to work after the bf16→fp16
    storage switch (defense in depth: if a future config flips the dtype
    back to bf16, the eigh-based AA-SVD must still tolerate it).
    """
    cfg = copy.deepcopy(tiny_config)
    cfg["stage2_reap_ream"]["covariance_storage_dtype"] = "bfloat16"
    cfg["stage3_svd"]["bcov_storage_dtype"] = "bfloat16"
    return cfg
