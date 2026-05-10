"""Optimizer tier dispatch + LR scheduler helpers.

Direct port from `structural_recovery/distillation.py:159-218`.

Two tiers, picked from the YAML's `distillation.optimizer` field:

* `adamw_bnb_8bit` — single-GPU smoke runs (no DeepSpeed). 8-bit AdamW from
  bitsandbytes; ~75% optimizer-state reduction vs fp32 AdamW. **Does not
  compose with DeepSpeed ZeRO-3** — bnb owns its `optim.step` and would
  fight DS's fp32 reduce.
* `deepspeed_cpu_adam` — multi-GPU under ZeRO-3. Optimizer state lives on
  the host CPU; per-rank GPU footprint is teacher (sharded) + student
  (sharded) + grads (sharded) + activations.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn

from ..config import DistillationConfig


# REQ: LLR-0039
def build_optimizer(student: nn.Module, dconf: DistillationConfig) -> torch.optim.Optimizer:
    """Return an optimizer instance for `student` per `dconf.optimizer`.

    `'adamw_bnb_8bit'` returns `bitsandbytes.optim.AdamW8bit` (a
    `torch.optim.Optimizer` subclass). `'deepspeed_cpu_adam'` returns
    `deepspeed.ops.adam.DeepSpeedCPUAdam(adamw_mode=True)` (also a
    `torch.optim.Optimizer` subclass).

    Both branches collect only `requires_grad=True` params — calling this
    after toggling trainable scope to e.g. `experts_only` would put only the
    expert params under the optimizer.
    """
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "build_optimizer: no trainable parameters. Did you set "
            "`requires_grad=True` on the student before calling?"
        )
    name = dconf.optimizer
    lr = dconf.learning_rate
    betas = (dconf.betas[0], dconf.betas[1])
    wd = dconf.weight_decay

    if name == "adamw_bnb_8bit":
        import bitsandbytes as bnb

        return cast(
            torch.optim.Optimizer,
            bnb.optim.AdamW8bit(trainable, lr=lr, betas=betas, weight_decay=wd),
        )

    if name == "deepspeed_cpu_adam":
        # adamw_mode=True selects AdamW (decoupled weight decay), matching bnb.
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        return cast(
            torch.optim.Optimizer,
            DeepSpeedCPUAdam(
                trainable, lr=lr, betas=betas, weight_decay=wd, adamw_mode=True
            ),
        )

    raise ValueError(
        f"Unknown optimizer: {name!r}. "
        "Expected 'adamw_bnb_8bit' or 'deepspeed_cpu_adam'."
    )


def cosine_with_warmup(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """Linear warmup `[0..warmup-1]` → cosine decay `[warmup..total_steps-1]`.

    `step` is the 0-based optimizer-step index about to be taken (called
    BEFORE `optim.step`). The cosine approaches `lr_min` from above as
    `step` approaches `total_steps - 1` (it never dips below `lr_min`); for
    `step >= total_steps` the function hard-clamps to `lr_min`.
    """
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    """Set every param-group's `lr` to `lr` in-place."""
    for g in optim.param_groups:
        g["lr"] = lr
