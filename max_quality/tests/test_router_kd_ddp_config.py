"""Task 0 — DdpConfig schema + resolver for the new ``ddp`` knob.

These tests pin the default-OFF contract, explicit world_size, auto
device-count resolution, the METRIC_PINNED divisibility cap, and YAML
string coercion. No behaviour change to the training loop yet.
"""
from __future__ import annotations

import pytest

from moe_compress.router_kd.ddp_config import DdpConfig


def _cfg(ddp=None, batch_size=2):
    s5 = {"batch_size": batch_size}
    if ddp is not None:
        s5["ddp"] = ddp
    return {"stage5_router_kd": s5}


def test_ddp_disabled_by_default():
    d = DdpConfig.from_config(_cfg())
    assert d.enabled is False
    assert d.world_size == 1


def test_ddp_explicit_world_size():
    d = DdpConfig.from_config(
        _cfg({"enabled": True, "world_size": 2}, batch_size=2),
        device_count_fn=lambda: 2,
    )
    assert d.enabled is True
    assert d.world_size == 2


def test_ddp_auto_world_size():
    d = DdpConfig.from_config(
        _cfg({"enabled": True}, batch_size=8),
        device_count_fn=lambda: 4,
    )
    assert d.enabled is True
    assert d.world_size == 4


def test_ddp_world_size_capped_by_global_batch():
    with pytest.raises(ValueError, match="divisible"):
        DdpConfig.from_config(
            _cfg({"enabled": True, "world_size": 4}, batch_size=2),
            device_count_fn=lambda: 4,
        )


def test_ddp_string_false_coerced():
    d = DdpConfig.from_config(_cfg({"enabled": "false"}, batch_size=2))
    assert d.enabled is False
    assert d.world_size == 1


def test_ddp_string_true_coerced():
    d = DdpConfig.from_config(
        _cfg({"enabled": "true", "world_size": 2}, batch_size=4),
        device_count_fn=lambda: 2,
    )
    assert d.enabled is True
    assert d.world_size == 2


def test_ddp_enabled_but_single_gpu_collapses_to_disabled():
    # enabled:true but only one device available → world_size 1 → enabled False
    # so the orchestrator takes the single-process path (backward-compat).
    d = DdpConfig.from_config(
        _cfg({"enabled": True}, batch_size=2),
        device_count_fn=lambda: 1,
    )
    assert d.enabled is False
    assert d.world_size == 1


def test_ddp_backend_default_nccl_and_override():
    d = DdpConfig.from_config(
        _cfg({"enabled": True, "world_size": 2, "backend": "gloo"}, batch_size=4),
        device_count_fn=lambda: 2,
    )
    assert d.backend == "gloo"
    d2 = DdpConfig.from_config(
        _cfg({"enabled": True, "world_size": 2}, batch_size=4),
        device_count_fn=lambda: 2,
    )
    assert d2.backend == "nccl"
