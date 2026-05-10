"""Tests for `kdr.training.optim` (LLR-0039).

# VERIFIES: LLR-0039
"""

from __future__ import annotations

from typing import Literal, cast

import pytest
import torch
import torch.nn as nn

from kdr.config import DistillationConfig
from kdr.training.optim import build_optimizer, cosine_with_warmup, set_lr

_OptName = Literal["adamw_bnb_8bit", "deepspeed_cpu_adam"]


def _make_dconf(optimizer: str) -> DistillationConfig:
    return DistillationConfig(
        loss="forward_kld",
        temperature=1.0,
        optimizer=cast("_OptName", optimizer),
        learning_rate=3e-5,
        min_learning_rate=3e-7,
        weight_decay=0.0,
        betas=[0.9, 0.95],
        grad_clip_norm=1.0,
        warmup_steps=10,
        total_tokens=1_000_000,
        per_device_batch_size=1,
        gradient_accumulation=1,
        sequence_length=128,
        log_every_n_steps=1,
        eval_every_n_steps=10,
        save_every_n_steps=0,
        trainable_scope="full",
    )


def test_build_optimizer_rejects_no_trainable_params() -> None:
    student = nn.Linear(8, 8)
    for p in student.parameters():
        p.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        build_optimizer(student, _make_dconf("adamw_bnb_8bit"))


def test_build_optimizer_rejects_unknown_optimizer() -> None:
    """Bypass Pydantic to feed an unknown name — verifies the runtime
    branch's error path independently of the schema's Literal."""
    student = nn.Linear(8, 8)
    dconf = _make_dconf("adamw_bnb_8bit")
    object.__setattr__(dconf, "optimizer", "unknown_optimizer")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(student, dconf)


def test_build_optimizer_bnb_path_raises_when_bnb_missing() -> None:
    """Smoke: bnb is not installed in the kdr venv. The branch must call into
    the import → ImportError. We assert it propagates rather than swallowing."""
    student = nn.Linear(8, 8)
    with pytest.raises(ModuleNotFoundError):
        build_optimizer(student, _make_dconf("adamw_bnb_8bit"))


def test_build_optimizer_dscpuadam_path_raises_when_deepspeed_missing() -> None:
    student = nn.Linear(8, 8)
    with pytest.raises(ModuleNotFoundError):
        build_optimizer(student, _make_dconf("deepspeed_cpu_adam"))


def test_cosine_with_warmup_warmup_phase() -> None:
    # Step 0 → first warmup increment.
    lr = cosine_with_warmup(
        step=0, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr == pytest.approx(1e-3 * 1 / 10)

    # Step 9 → end of warmup.
    lr = cosine_with_warmup(
        step=9, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr == pytest.approx(1e-3)


def test_cosine_with_warmup_decay_phase() -> None:
    # Step warmup_steps → cosine starts at lr_max.
    lr = cosine_with_warmup(
        step=10, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr == pytest.approx(1e-3, rel=1e-9)

    # Step total_steps - 1 → near lr_min.
    lr_end = cosine_with_warmup(
        step=99, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr_end < 1e-3
    assert lr_end >= 1e-6 - 1e-9


def test_cosine_with_warmup_clamps_past_total() -> None:
    lr = cosine_with_warmup(
        step=200, warmup_steps=10, total_steps=100, lr_max=1e-3, lr_min=1e-6
    )
    assert lr == 1e-6


def test_set_lr_overwrites_all_param_groups() -> None:
    student = nn.Linear(8, 8)
    optim = torch.optim.AdamW(student.parameters(), lr=1.0)
    set_lr(optim, 0.42)
    for g in optim.param_groups:
        assert g["lr"] == 0.42
