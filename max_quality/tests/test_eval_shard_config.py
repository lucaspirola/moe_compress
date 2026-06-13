"""Task 6 — EvalShardConfig + default-OFF resolver (_should_shard).

Default single-GPU byte-identical: enabled:false / replicas<=1 /
device_count()<2 -> 0 (in-process). Mirrors AutoBatchConfig.from_dict's
bool-coerce so a stray YAML string ("true") can't silently flip behaviour the
wrong way (here it must flip ON only when explicitly "true").
"""
from __future__ import annotations

import pytest

from moe_compress.tools.eval_shard import EvalShardConfig, _should_shard


def test_config_defaults_off():
    cfg = EvalShardConfig.from_dict(None)
    assert cfg.enabled is False
    assert cfg.replicas == 0
    assert cfg.gpus_per_replica == 1
    assert cfg.ppl is False


def test_config_string_true_coerced():
    cfg = EvalShardConfig.from_dict({"enabled": "true", "replicas": 4, "ppl": "true"})
    assert cfg.enabled is True
    assert cfg.replicas == 4
    assert cfg.ppl is True


def test_config_string_false_stays_off():
    cfg = EvalShardConfig.from_dict({"enabled": "false", "replicas": 4})
    assert cfg.enabled is False


def test_config_unknown_keys_ignored():
    cfg = EvalShardConfig.from_dict({"enabled": True, "replicas": 2, "bogus": 99})
    assert cfg.enabled is True
    assert cfg.replicas == 2


def test_should_shard_disabled_returns_zero(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 8)
    cfg = EvalShardConfig.from_dict({"enabled": False, "replicas": 4})
    assert _should_shard(cfg, n_examples=100) == 0


def test_should_shard_clamps_to_device_count(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    cfg = EvalShardConfig.from_dict({"enabled": "true", "replicas": 4})
    # 4 requested, only 2 GPUs -> 2
    assert _should_shard(cfg, n_examples=100) == 2


def test_should_shard_clamps_to_n_groups_for_gen(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 8)
    cfg = EvalShardConfig.from_dict({"enabled": "true", "replicas": 8})
    # 24 examples = 3 groups (8,8,8) -> at most 3 replicas on the gen path.
    assert _should_shard(cfg, n_examples=24, group_aligned=True) == 3


def test_should_shard_ppl_no_group_clamp(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 8)
    cfg = EvalShardConfig.from_dict({"enabled": "true", "replicas": 8})
    # PPL path: no group constraint, clamp only to device_count + n_examples.
    assert _should_shard(cfg, n_examples=24, group_aligned=False) == 8
    assert _should_shard(cfg, n_examples=3, group_aligned=False) == 3


def test_should_shard_under_2_gpus_forces_inprocess(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    cfg = EvalShardConfig.from_dict({"enabled": "true", "replicas": 4})
    assert _should_shard(cfg, n_examples=100) == 0


def test_should_shard_replicas_one_is_inprocess(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 8)
    cfg = EvalShardConfig.from_dict({"enabled": "true", "replicas": 1})
    assert _should_shard(cfg, n_examples=100) == 0
