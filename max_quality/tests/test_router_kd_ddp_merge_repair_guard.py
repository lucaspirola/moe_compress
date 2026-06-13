"""Task 12 — merge-repair + DDP is not yet supported (clear guard error).

DDP ships FIRST without merge-repair (Stage-2.5-only, opt-in, default-OFF). When
merge_repair.enabled at stage2p5 + DDP, _spawn_ddp_workers must raise a clear
"not yet supported" error BEFORE any spawn. Non-merge-repair DDP is unaffected;
stage5 (where merge-repair is never enabled) is unaffected.
"""
from __future__ import annotations

import pytest

import torch.nn as nn

from moe_compress.router_kd import orchestrator as rk_orchestrator
from moe_compress.router_kd.ddp_config import DdpConfig


def _ddp():
    return DdpConfig(enabled=True, world_size=2, backend="gloo")


def _cfg(merge_repair_enabled):
    return {
        "model": {
            "name_or_path": "tiny", "revision": "main",
            "torch_dtype": "float32", "attn_implementation": "sdpa",
        },
        "stage5_router_kd": {
            "epochs": 1, "teacher_load_in_4bit": True,
            "merge_repair": {"enabled": merge_repair_enabled},
        },
    }


def test_merge_repair_ddp_raises_not_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rk_orchestrator, "spawn_ddp_workers",
        lambda *a, **k: pytest.fail("must not spawn"),
    )
    monkeypatch.setattr(
        rk_orchestrator, "save_compressed_checkpoint",
        lambda *a, **k: tmp_path,
    )
    with pytest.raises(RuntimeError, match="merge.?repair"):
        rk_orchestrator._spawn_ddp_workers(
            nn.Linear(2, 2), object(), _cfg(True), tmp_path,
            no_resume=False, stage_key="stage2p5", ddp=_ddp(),
        )


def test_merge_repair_off_ddp_ok_at_stage2p5(monkeypatch, tmp_path):
    # merge_repair off → no guard; spawn proceeds (stubbed).
    monkeypatch.setattr(
        rk_orchestrator, "spawn_ddp_workers",
        lambda ws, **k: tmp_path / "stage2p5_final",
    )
    monkeypatch.setattr(
        rk_orchestrator, "save_compressed_checkpoint",
        lambda *a, **k: tmp_path,
    )
    out = rk_orchestrator._spawn_ddp_workers(
        nn.Linear(2, 2), object(), _cfg(False), tmp_path,
        no_resume=False, stage_key="stage2p5", ddp=_ddp(),
    )
    assert out == tmp_path / "stage2p5_final"


def test_merge_repair_enabled_but_stage5_ok(monkeypatch, tmp_path):
    # merge_repair.enabled but stage_key=stage5 (where merge-repair is NEVER
    # active) → no guard; spawn proceeds.
    monkeypatch.setattr(
        rk_orchestrator, "spawn_ddp_workers",
        lambda ws, **k: tmp_path / "stage5_final",
    )
    monkeypatch.setattr(
        rk_orchestrator, "save_compressed_checkpoint",
        lambda *a, **k: tmp_path,
    )
    out = rk_orchestrator._spawn_ddp_workers(
        nn.Linear(2, 2), object(), _cfg(True), tmp_path,
        no_resume=False, stage_key="stage5", ddp=_ddp(),
    )
    assert out == tmp_path / "stage5_final"
