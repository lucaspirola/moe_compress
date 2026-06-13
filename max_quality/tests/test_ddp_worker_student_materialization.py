"""Task 8 — _run_ddp_worker materializes the LIVE compressed student per rank.

The H1 fix: each rank trains the LIVE compressed student (post-merge / post-
EoRA), reconstructed from a parent-written temp dir via load_compressed_model —
NOT load_model(config["model"]["name_or_path"]) (the original uncompressed repo).

The byte round-trip of save_compressed_checkpoint → load_compressed_model is
covered by test_load_compressed_model.py; here we pin the WORKER ORCHESTRATION:
it calls the compressed loader (never the original-repo loader), threads the DDP
config, and returns rank-0's out_dir Path. Serializers are monkeypatched (the
tiny fixture is not a real HF checkpoint).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import torch
import torch.nn as nn

from moe_compress.router_kd import orchestrator as rk_orchestrator
from moe_compress.router_kd.ddp_config import DdpConfig


def _cfg():
    return {
        "model": {
            "name_or_path": "ORIGINAL-REPO-MUST-NOT-LOAD",
            "revision": "main", "torch_dtype": "float32",
            "attn_implementation": "sdpa",
        },
        "stage5_router_kd": {"epochs": 1, "teacher_load_in_4bit": True},
    }


def test_each_rank_reloads_compressed(monkeypatch, tmp_path):
    calls = {"compressed": 0, "original": 0}

    def _fake_load_compressed(path, **kwargs):
        calls["compressed"] += 1
        return nn.Linear(2, 2), object(), {}

    def _fake_load_model(name_or_path, **kwargs):
        calls["original"] += 1
        return nn.Linear(2, 2), object()

    captured = {}

    def _fake_run_single(student, tokenizer, config, artifacts_dir, **kwargs):
        captured["rank"] = kwargs.get("rank")
        captured["world_size"] = kwargs.get("world_size")
        captured["ddp_enabled"] = kwargs.get("ddp").enabled
        return Path(artifacts_dir) / "stage5_final"

    monkeypatch.setattr(rk_orchestrator, "load_compressed_model", _fake_load_compressed)
    monkeypatch.setattr(rk_orchestrator, "_run_single_process", _fake_run_single)
    # load_model lives in the teacher plugin; assert it is NEVER called here.
    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "load_model", _fake_load_model)

    out = rk_orchestrator._run_ddp_worker(
        rank=0, world_size=2, config=_cfg(),
        artifacts_dir=str(tmp_path), student_src=str(tmp_path / "_src"),
        no_resume=False, stage_key="stage5",
        ddp_world_size=2, backend="gloo",
    )
    assert calls["compressed"] == 1
    assert calls["original"] == 0
    assert captured["ddp_enabled"] is True
    assert captured["world_size"] == 2
    assert out == tmp_path / "stage5_final"


def test_spawn_materializes_live_student(monkeypatch, tmp_path):
    # _spawn_ddp_workers must serialize the LIVE student via
    # save_compressed_checkpoint (NOT name_or_path) before spawning, validate
    # the teacher strategy, and clean up the temp dir.
    saved = {"path": None, "model": None}

    def _fake_save(model, tokenizer, out_dir, *, pipeline_stage, **kw):
        saved["path"] = Path(out_dir)
        saved["model"] = model
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return Path(out_dir)

    def _fake_spawn(world_size, *, backend, payload, worker_fn):
        # Echo the payload so we can assert student_src is threaded.
        assert "student_src" in payload
        assert payload["ddp_world_size"] == 2
        return Path(payload["artifacts_dir"]) / "stage5_final"

    monkeypatch.setattr(rk_orchestrator, "save_compressed_checkpoint", _fake_save)
    monkeypatch.setattr(rk_orchestrator, "spawn_ddp_workers", _fake_spawn)

    student = nn.Linear(2, 2)
    ddp = DdpConfig(enabled=True, world_size=2, backend="gloo")
    cfg = _cfg()

    out = rk_orchestrator._spawn_ddp_workers(
        student, object(), cfg, tmp_path,
        no_resume=False, stage_key="stage5", ddp=ddp,
    )
    assert out == tmp_path / "stage5_final"
    # The LIVE student was serialized (the unwrapped module object).
    assert saved["model"] is student
    assert saved["path"] == tmp_path / "_ddp_student_src"
    # Temp dir cleaned up after the join.
    assert not (tmp_path / "_ddp_student_src").exists()


def test_spawn_validates_teacher_strategy(monkeypatch, tmp_path):
    # epochs>1 + no quantized teacher → the teacher-strategy precondition fires
    # BEFORE any spawn.
    monkeypatch.setattr(
        rk_orchestrator, "spawn_ddp_workers",
        lambda *a, **k: pytest.fail("spawn must not be reached"),
    )
    cfg = _cfg()
    cfg["stage5_router_kd"] = {"epochs": 2}  # no cache, no quantized teacher
    ddp = DdpConfig(enabled=True, world_size=2, backend="gloo")
    with pytest.raises(RuntimeError, match="epochs>1"):
        rk_orchestrator._spawn_ddp_workers(
            nn.Linear(2, 2), object(), cfg, tmp_path,
            no_resume=False, stage_key="stage5", ddp=ddp,
        )
