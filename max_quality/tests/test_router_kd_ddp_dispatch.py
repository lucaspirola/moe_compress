"""Task 1 — orchestrator dispatch fork (single-process vs DDP).

The default (no ``ddp`` key) MUST take the existing single-process path and
NEVER call the DDP spawn. With ``ddp.enabled true, world_size 2`` the
orchestrator dispatches to ``_spawn_ddp_workers`` and returns its Path.

Scaffold helpers are redeclared verbatim (codebase discipline — no cross-test
imports), mirroring ``test_router_kd_orchestrator.py``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress import stage1
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    from moe_compress.budget.solver import BudgetDecomposition
    from moe_compress.router_kd import orchestrator as rk_orchestrator
except Exception as e:  # pragma: no cover
    pytest.skip(f"Router-KD imports unavailable: {e}", allow_module_level=True)


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


def _fake_build(tokenizer, spec, cache_dir=None):
    torch.manual_seed(spec.seed)
    return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                         dtype=torch.long)


def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
    torch.manual_seed(spec.seed + 1)
    return torch.randint(0, 32, (num_samples, spec.sequence_length),
                         dtype=torch.long)


def _prepare_model_and_merge_map(model, config, tmp_path, monkeypatch):
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.utils.model_io import iter_moe_layers

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2, expert_prune_ratio=0.5, svd_rank_ratio=0.14,
        global_expert_budget=4, min_experts_per_layer=2, blacklisted_experts={},
    )
    stage1.run(model, _TinyTokenizer(), config, tmp_path, decomp)
    stage2_reap_ream.run(model, _TinyTokenizer(), config, tmp_path, device=None)

    moe_layer_refs = list(iter_moe_layers(model))
    trivial_map = {
        str(ref.layer_idx): {str(i): [i] for i in range(ref.num_routed_experts)}
        for ref in moe_layer_refs
    }
    (tmp_path / "stage2_pruned").mkdir(parents=True, exist_ok=True)
    (tmp_path / "stage2_pruned" / "merge_map.json").write_text(json.dumps(trivial_map))


def _patch_common(monkeypatch, tiny_model):
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.utils import model_io as mio
    from moe_compress.router_kd.plugins import teacher as rk_teacher

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(rk_orchestrator, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(rk_orchestrator, "_trackio_log", lambda payload: None)

    def _load_student(*_a, **_k):
        return tiny_model, _TinyTokenizer()

    monkeypatch.setattr(mio, "load_model", _load_student)
    monkeypatch.setattr(rk_teacher, "load_model", _load_student)


def test_default_takes_single_process(tiny_model, tiny_config, tmp_path, monkeypatch):
    cfg = copy.deepcopy(tiny_config)
    cfg["stage5_router_kd"]["rkd_recipe"] = "current"
    _patch_common(monkeypatch, tiny_model)
    monkeypatch.setattr(
        tiny_model.config, "save_pretrained", lambda *a, **k: None, raising=False,
    )

    called = {"ddp": False}

    def _sentinel(*a, **k):
        called["ddp"] = True
        return Path("SHOULD-NOT-BE-USED")

    monkeypatch.setattr(rk_orchestrator, "_spawn_ddp_workers", _sentinel)

    _prepare_model_and_merge_map(tiny_model, cfg, tmp_path, monkeypatch)
    out = rk_orchestrator.run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None, stage_key="stage5",
    )
    assert called["ddp"] is False
    assert out == tmp_path / "stage5_final"


def test_ddp_enabled_dispatches_spawn(tiny_model, tiny_config, tmp_path, monkeypatch):
    cfg = copy.deepcopy(tiny_config)
    cfg["stage5_router_kd"]["rkd_recipe"] = "current"
    cfg["stage5_router_kd"]["batch_size"] = 2
    cfg["stage5_router_kd"]["ddp"] = {
        "enabled": True, "world_size": 2, "backend": "gloo",
    }
    _patch_common(monkeypatch, tiny_model)

    fake = tmp_path / "stage5_final"

    captured = {}

    def _stub_spawn(student, tokenizer, config, artifacts_dir, *, no_resume, stage_key, ddp):
        captured["world_size"] = ddp.world_size
        return fake

    monkeypatch.setattr(rk_orchestrator, "_spawn_ddp_workers", _stub_spawn)
    # Force device_count so DdpConfig resolves world_size=2 on this CPU box.
    monkeypatch.setattr(
        rk_orchestrator.DdpConfig, "from_config",
        classmethod(lambda cls, c: cls(enabled=True, world_size=2, backend="gloo")),
    )

    out = rk_orchestrator.run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None, stage_key="stage5",
    )
    assert captured["world_size"] == 2
    assert out == fake
