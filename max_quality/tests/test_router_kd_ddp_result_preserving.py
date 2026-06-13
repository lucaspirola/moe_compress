"""Task 10 — RESULT-PRESERVATION GATE (the acceptance test).

Single-process (world_size=1) vs DDP world_size=2 on CPU/gloo, with IDENTICAL
batch_size=4 (per_gpu=2) AND identical len(batches), a NON-degenerate teacher
(per-row-distinct logits), must produce matching final router weights + loss
trace within rtol=1e-5, atol=1e-7. This proves average-gradient DDP at the same
effective global batch ≡ single-GPU full-batch, exercising the per-step
row-split + all-reduce + per-rank teacher handling (per_gpu=2 defeats the
teacher==student / per_gpu=1 mask of the double-slice bug).

Both the world=1 and world=2 runs go through spawn_ddp_workers so the SAME code
path (DdpConfig, row-split, all-reduce, rank-0 export) runs in both — only
world_size differs. Each worker builds the student + teacher deterministically
(seeded) in-process; monkeypatch cannot cross the spawn boundary.
"""
from __future__ import annotations

import copy

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_compress.router_kd.ddp_runtime import spawn_ddp_workers


# ---- deterministic tiny model + tokenizer (redeclared; no cross-test import) --
class _TinyFusedExperts(nn.Module):
    def __init__(self, num_experts, hidden, intermediate):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.intermediate_dim = intermediate
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate, hidden) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate) * 0.02)
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
            inter = self.act_fn(gate) * up
            down = F.linear(inter, self.down_proj[e])
            down = down * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, down.to(final.dtype))
        return final


class _TinyRouter(nn.Module):
    def __init__(self, num_experts, hidden, top_k):
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
    def __init__(self, hidden, intermediate, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = _TinyRouter(num_experts, hidden, top_k)
        self.experts = _TinyFusedExperts(num_experts, hidden, intermediate)
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
        self.vocab_size = 32
        self.text_config = self


class _TinyModel(nn.Module):
    def __init__(self, *, hidden=16, intermediate=8, num_layers=2, num_experts=4, top_k=2):
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
        class _Out:
            pass
        out = _Out()
        out.logits = logits
        out.loss = None
        return out


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_a, **_k):
        return None


def _tiny_config():
    return {
        "model": {
            "name_or_path": "tiny", "revision": "main",
            "torch_dtype": "float32", "device_map": "cpu",
            "attn_implementation": "sdpa",
            "load_in_4bit": False, "trust_remote_code": False,
        },
        "calibration": {
            "source": "c4-math-code", "dataset": "allenai/c4", "subset": "en",
            "split": "train", "seed": 0, "num_sequences": 16, "sequence_length": 8,
            "super_expert_num_samples": 4,
            "domain_mix": {"c4": 1.0, "math": 0.0, "code": 0.0},
            "math_dataset": "unused", "code_dataset": "unused",
        },
        "stage5_router_kd": {
            "optimizer": "adamw", "learning_rate": 5.0e-4, "epochs": 1,
            "batch_size": 4, "gradient_accumulation": 1,
            "max_sequence_length": 8, "kd_temperature": 1.0,
            "max_calibration_samples": 16,
            "rkd_recipe": "current",
            "trainable_name_patterns": ["mlp.gate.weight"],
            "frozen_name_patterns": ["experts", "shared_expert", "embed", "lm_head"],
            "enable_output_router_logits": True,
            "checkpoint_every_n_steps": 1000,
        },
        "logging": {"level": "INFO", "log_every_n_steps": 1,
                    "save_intermediate_every_n_layers": 1},
    }


# The shared worker: builds student + non-degenerate teacher deterministically,
# runs _run_single_process, returns rank-0's (loss_trace, router_state).
def _rp_worker(*, rank, world_size, ddp_world_size, backend, tmpdir):
    import pathlib
    torch.manual_seed(1234)
    torch.set_num_threads(1)

    from moe_compress.router_kd import orchestrator as rk_orchestrator
    from moe_compress.router_kd.plugins import teacher as rk_teacher
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.router_kd.ddp_config import DdpConfig

    # NON-degenerate teacher: a DIFFERENT seeded tiny model (distinct weights →
    # per-row-distinct logits). Same seed in both runs → identical KD target.
    def _make_teacher():
        torch.manual_seed(777)
        return _TinyModel(), _TinyTokenizer()

    def _load_model_stub(*_a, **_k):
        return _make_teacher()

    rk_teacher.load_model = _load_model_stub

    # Deterministic calibration: identical token tensor for both runs.
    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    rk_orchestrator.build_calibration_tensor = _fake_build
    cal_mod.build_calibration_tensor = _fake_build

    captured = []
    rk_orchestrator._trackio_log = lambda payload: (
        captured.append(dict(payload)) if "stage5/loss" in payload else None
    )

    # Build the student (identical on every rank / both runs via the seed).
    torch.manual_seed(4242)
    student = _TinyModel()

    ddp_cfg = (
        DdpConfig(enabled=True, world_size=ddp_world_size, backend=backend)
        if ddp_world_size > 1 else None
    )
    artifacts = pathlib.Path(tmpdir) / f"ws{ddp_world_size}"
    artifacts.mkdir(parents=True, exist_ok=True)

    rk_orchestrator._run_single_process(
        student, _TinyTokenizer(), _tiny_config(), artifacts,
        device=None, no_resume=True, stage_key="stage5",
        rank=rank, world_size=world_size, ddp=ddp_cfg,
    )

    if rank != 0:
        return "ok"
    base = rk_orchestrator.unwrap_student(student)
    router_state = {
        n: p.detach().cpu().clone()
        for n, p in base.named_parameters() if p.requires_grad
    }
    trace = [(p["stage5/step"], p["stage5/loss"], p["stage5/raw_kl"]) for p in captured]
    # Return via a file (torch tensors through a mp queue share shm that is torn
    # down on child exit → FileNotFoundError in the parent). The parent loads it.
    out_file = pathlib.Path(tmpdir) / f"result_ws{ddp_world_size}.pt"
    torch.save({"trace": trace, "router_state": router_state}, out_file)
    return str(out_file)


def test_ddp2_matches_single_process(tmp_path):
    # Run single-process (world=1) and DDP (world=2) through the SAME spawn
    # driver so only world_size differs (save_compressed_checkpoint is stubbed
    # to a no-op inside each worker — the tiny model is not a real HF ckpt).
    payload1 = dict(ddp_world_size=1, backend="gloo", tmpdir=str(tmp_path))
    f1 = spawn_ddp_workers(1, backend="gloo", payload=payload1,
                           worker_fn=_rp_worker_export_noop)

    payload2 = dict(ddp_world_size=2, backend="gloo", tmpdir=str(tmp_path))
    f2 = spawn_ddp_workers(2, backend="gloo", payload=payload2,
                           worker_fn=_rp_worker_export_noop)

    r1 = torch.load(f1, weights_only=False)
    r2 = torch.load(f2, weights_only=False)
    trace1, state1 = r1["trace"], r1["router_state"]
    trace2, state2 = r2["trace"], r2["router_state"]

    assert len(trace1) == len(trace2) and len(trace1) > 0
    import math
    for (s1, l1, k1), (s2, l2, k2) in zip(trace1, trace2):
        assert s1 == s2
        assert math.isclose(l1, l2, rel_tol=1e-5, abs_tol=1e-7), (s1, l1, l2)
        assert math.isclose(k1, k2, rel_tol=1e-5, abs_tol=1e-7), (s1, k1, k2)

    assert set(state1.keys()) == set(state2.keys())
    for k in state1:
        assert torch.allclose(state1[k], state2[k], rtol=1e-5, atol=1e-7), (
            k, float((state1[k] - state2[k]).abs().max()))


# Worker variant that ALSO stubs the final export to a no-op (the tiny model is
# not a real HF checkpoint, so save_compressed_checkpoint's config.save_pretrained
# would fail). Defined at module level for the spawn handoff.
def _rp_worker_export_noop(*, rank, world_size, ddp_world_size, backend, tmpdir):
    import pathlib
    from moe_compress.router_kd import orchestrator as rk_orchestrator

    def _noop_save(model, tokenizer, path, **kw):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(path)

    rk_orchestrator.save_compressed_checkpoint = _noop_save
    return _rp_worker(rank=rank, world_size=world_size,
                      ddp_world_size=ddp_world_size, backend=backend, tmpdir=tmpdir)
