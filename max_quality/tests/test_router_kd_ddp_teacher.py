"""Task 7 — teacher VRAM strategy: validated precondition for DDP.

epochs=1 + cache → OK (path A, faithful, zero per-rank teacher VRAM).
epochs>1 (paper default) + DDP + no quantized teacher → RAISE.
epochs>1 + 4-bit → OK (path B, quality-trade, explicitly configured).
"""
from __future__ import annotations

import pytest

from moe_compress.router_kd.ddp_config import DdpConfig
from moe_compress.router_kd.orchestrator import validate_ddp_teacher_strategy


def _ddp():
    return DdpConfig(enabled=True, world_size=2, backend="gloo")


def test_ddp_epoch1_allows_cache():
    cfg = {"stage5_router_kd": {
        "epochs": 1, "teacher_logits_cache": "tc.pt",
    }}
    # No raise.
    validate_ddp_teacher_strategy(cfg, _ddp())


def test_ddp_multiepoch_requires_quantized_teacher():
    cfg = {"stage5_router_kd": {"epochs": 2}}
    with pytest.raises(RuntimeError, match="4-bit|FP8|teacher_load_in_4bit"):
        validate_ddp_teacher_strategy(cfg, _ddp())


def test_ddp_multiepoch_4bit_ok():
    cfg = {"stage5_router_kd": {"epochs": 2, "teacher_load_in_4bit": True}}
    validate_ddp_teacher_strategy(cfg, _ddp())


def test_ddp_multiepoch_fp8_repo_ok():
    cfg = {"stage5_router_kd": {"epochs": 2, "teacher_model_repo": "some/fp8"}}
    validate_ddp_teacher_strategy(cfg, _ddp())


def test_ddp_epoch1_no_cache_quantized_ok():
    # epochs==1 without cache but with quantized teacher: allowed (each rank
    # loads its own teacher; operator's call).
    cfg = {"stage5_router_kd": {"epochs": 1, "teacher_load_in_4bit": True}}
    validate_ddp_teacher_strategy(cfg, _ddp())
