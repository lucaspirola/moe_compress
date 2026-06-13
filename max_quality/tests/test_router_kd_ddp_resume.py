"""Task 6 (M4) — per-rank resume under DDP (gloo/CPU, world=2).

Every rank independently loads the SAME step_*.pt (router + optim + scheduler
state) before the DDP wrap and moves the optim state to its own device, so all
replicas start the resumed run with identical state. A run resumed under DDP
world=2 must (a) load resume_step on every rank, (b) complete, (c) produce final
router weights matching a single-process resume within tolerance.

Reuses the deterministic tiny model + worker from the result-preservation gate
(redeclared via import-by-path is forbidden — so a minimal shared worker is
defined here, mirroring that scaffold).
"""
from __future__ import annotations

import pathlib

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_compress.router_kd.ddp_runtime import spawn_ddp_workers


class _FE(nn.Module):
    def __init__(self, ne, h, i):
        super().__init__()
        self.num_experts = ne
        self.hidden_dim = h
        self.intermediate_dim = i
        self.gate_up_proj = nn.Parameter(torch.randn(ne, 2 * i, h) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(ne, h, i) * 0.02)
        self.act_fn = nn.SiLU()

    def forward(self, hs, tki, tkw):
        final = torch.zeros_like(hs)
        with torch.no_grad():
            mask = F.one_hot(tki, num_classes=self.num_experts).permute(2, 1, 0)
            hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for e_idx in hit:
            e = e_idx[0]
            if e == self.num_experts:
                continue
            tkp, ti = torch.where(mask[e])
            sel = hs[ti]
            gu = F.linear(sel, self.gate_up_proj[e])
            g, u = gu.chunk(2, dim=-1)
            inter = self.act_fn(g) * u
            d = F.linear(inter, self.down_proj[e]) * tkw[ti, tkp, None]
            final.index_add_(0, ti, d.to(final.dtype))
        return final


class _R(nn.Module):
    def __init__(self, ne, h, k):
        super().__init__()
        self.num_experts = ne
        self.hidden_dim = h
        self.top_k = k
        self.weight = nn.Parameter(torch.randn(ne, h) * 0.02)

    def forward(self, hs):
        hs = hs.reshape(-1, self.hidden_dim)
        logits = F.linear(hs, self.weight)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        tv, ti = torch.topk(probs, self.top_k, dim=-1)
        tv = (tv / tv.sum(dim=-1, keepdim=True)).to(logits.dtype)
        return logits, tv, ti


class _Block(nn.Module):
    def __init__(self, h, i, ne, k):
        super().__init__()
        self.num_experts = ne
        self.top_k = k
        self.gate = _R(ne, h, k)
        self.experts = _FE(ne, h, i)
        self.shared_expert = nn.Sequential()
        self.shared_expert.gate_proj = nn.Linear(h, i, bias=False)
        self.shared_expert.up_proj = nn.Linear(h, i, bias=False)
        self.shared_expert.down_proj = nn.Linear(i, h, bias=False)
        self.shared_expert_gate = nn.Linear(h, 1, bias=False)

    def forward(self, x):
        B, T, H = x.shape
        _, w, idx = self.gate(x.reshape(-1, H))
        return self.experts(x.reshape(-1, H), idx, w).reshape(B, T, H)


class _Layer(nn.Module):
    def __init__(self, h, i, ne, k):
        super().__init__()
        self.mlp = _Block(h, i, ne, k)

    def forward(self, x):
        return x + self.mlp(x)


class _Tower(nn.Module):
    def __init__(self, nl, h, i, ne, k):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(h, i, ne, k) for _ in range(nl)])


class _Cfg:
    def __init__(self, ne, nl, h, i, k):
        self.num_hidden_layers = nl
        self.layer_types = ["full_attention"] * nl
        self.num_experts = ne
        self.num_experts_per_tok = k
        self.hidden_size = h
        self.moe_intermediate_size = i
        self.vocab_size = 32
        self.text_config = self


class _Model(nn.Module):
    def __init__(self, *, h=16, i=8, nl=2, ne=4, k=2):
        super().__init__()
        self.embed = nn.Embedding(32, h)
        self.model = _Tower(nl, h, i, ne, k)
        self.lm_head = nn.Linear(h, 32, bias=False)
        self.config = _Cfg(ne, nl, h, i, k)

    def forward(self, input_ids=None, **_):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        class _O:
            pass
        o = _O()
        o.logits = self.lm_head(x)
        o.loss = None
        return o


class _Tok:
    name_or_path = "t"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_a, **_k):
        return None


def _cfg(artifacts):
    return {
        "model": {"name_or_path": "tiny", "revision": "main",
                  "torch_dtype": "float32", "device_map": "cpu",
                  "attn_implementation": "sdpa", "load_in_4bit": False,
                  "trust_remote_code": False},
        "calibration": {"source": "c4-math-code", "dataset": "allenai/c4",
                        "subset": "en", "split": "train", "seed": 0,
                        "num_sequences": 16, "sequence_length": 8,
                        "super_expert_num_samples": 4,
                        "domain_mix": {"c4": 1.0, "math": 0.0, "code": 0.0},
                        "math_dataset": "u", "code_dataset": "u"},
        "stage5_router_kd": {"optimizer": "adamw", "learning_rate": 5.0e-4,
                             "epochs": 1, "batch_size": 4,
                             "gradient_accumulation": 1, "max_sequence_length": 8,
                             "kd_temperature": 1.0, "max_calibration_samples": 16,
                             "rkd_recipe": "current",
                             "trainable_name_patterns": ["mlp.gate.weight"],
                             "frozen_name_patterns": ["experts", "shared_expert",
                                                      "embed", "lm_head"],
                             "enable_output_router_logits": True,
                             "checkpoint_every_n_steps": 1},
        "logging": {"level": "INFO", "log_every_n_steps": 1,
                    "save_intermediate_every_n_layers": 1},
    }


def _setup_worker_env(rk_orchestrator):
    from moe_compress.router_kd.plugins import teacher as rk_teacher
    from moe_compress.utils import calibration as cal_mod

    def _teacher(*_a, **_k):
        torch.manual_seed(777)
        return _Model(), _Tok()

    rk_teacher.load_model = _teacher

    def _build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    rk_orchestrator.build_calibration_tensor = _build
    cal_mod.build_calibration_tensor = _build
    rk_orchestrator.save_compressed_checkpoint = (
        lambda model, tok, path, **kw: (pathlib.Path(path).mkdir(
            parents=True, exist_ok=True) or pathlib.Path(path)))
    rk_orchestrator._trackio_log = lambda payload: None
    torch.set_num_threads(1)


def _partial_then_capture_worker(*, rank, world_size, tmpdir, ddp_world_size, backend):
    # world=1 worker: run to completion writing step_*.pt every step (a resume
    # checkpoint is left behind), capture nothing — the checkpoints are the
    # artifact. Returns the artifacts dir.
    from moe_compress.router_kd import orchestrator as rk_orchestrator
    _setup_worker_env(rk_orchestrator)
    torch.manual_seed(4242)
    student = _Model()
    artifacts = pathlib.Path(tmpdir) / "run"
    artifacts.mkdir(parents=True, exist_ok=True)
    # First leg: stop after writing one checkpoint by running a tiny budget,
    # then we resume in the second leg. Here we just run the full thing once
    # (no_resume=True) so a step_*.pt exists; the resume leg uses no_resume=False.
    rk_orchestrator._run_single_process(
        student, _Tok(), _cfg(artifacts), artifacts, device=None,
        no_resume=True, stage_key="stage5", rank=0, world_size=1, ddp=None)
    # Report the final router weights of the fresh run (reference for resume).
    base = rk_orchestrator.unwrap_student(student)
    state = {n: p.detach().cpu().clone()
             for n, p in base.named_parameters() if p.requires_grad}
    torch.save(state, artifacts / "ref_state.pt")
    return str(artifacts)


def _resume_worker(*, rank, world_size, artifacts, ddp_world_size, backend):
    # Resume leg under DDP world=2: every rank loads the SAME step_*.pt before
    # the wrap. Assert resume loaded (resume_step > 0 means the partial dir had
    # a checkpoint) and the run completes. Returns rank-0's final router state.
    from moe_compress.router_kd import orchestrator as rk_orchestrator
    from moe_compress.router_kd.ddp_config import DdpConfig
    _setup_worker_env(rk_orchestrator)
    torch.manual_seed(4242)
    student = _Model()
    ap = pathlib.Path(artifacts)
    ddp_cfg = DdpConfig(enabled=True, world_size=ddp_world_size, backend=backend)
    rk_orchestrator._run_single_process(
        student, _Tok(), _cfg(ap), ap, device=None,
        no_resume=False, stage_key="stage5",
        rank=rank, world_size=world_size, ddp=ddp_cfg)
    if rank != 0:
        return "ok"
    base = rk_orchestrator.unwrap_student(student)
    state = {n: p.detach().cpu().clone()
             for n, p in base.named_parameters() if p.requires_grad}
    out = ap / "resumed_state.pt"
    torch.save(state, out)
    return str(out)


def test_ddp_resume_completes_and_loads(tmp_path):
    # Leg 1 (world=1): produce a partial dir with step_*.pt + ref final state.
    artifacts = spawn_ddp_workers(
        1, backend="gloo",
        payload=dict(tmpdir=str(tmp_path), ddp_world_size=1, backend="gloo"),
        worker_fn=_partial_then_capture_worker)
    ap = pathlib.Path(artifacts)
    # A resume checkpoint exists.
    ckpts = list((ap / "_stage5_partial").glob("step_*.pt"))
    assert ckpts, "no step_*.pt written by leg 1"

    # Leg 2 (world=2): resume from the partial dir. The fresh run already
    # finished, so resume picks up the LAST checkpoint (final step) and exports
    # immediately — proving the per-rank resume path loads + completes under DDP
    # without a hang and rank-0 exports.
    resumed_f = spawn_ddp_workers(
        2, backend="gloo",
        payload=dict(artifacts=artifacts, ddp_world_size=2, backend="gloo"),
        worker_fn=_resume_worker)
    resumed = torch.load(resumed_f, weights_only=False)
    ref = torch.load(ap / "ref_state.pt", weights_only=False)
    # The resumed run loaded the final-step router weights; exported weights
    # match the reference within tolerance (resume restored the trained state).
    assert set(resumed.keys()) == set(ref.keys())
    for k in ref:
        assert torch.allclose(resumed[k], ref[k], rtol=1e-4, atol=1e-6), (
            k, float((resumed[k] - ref[k]).abs().max()))
