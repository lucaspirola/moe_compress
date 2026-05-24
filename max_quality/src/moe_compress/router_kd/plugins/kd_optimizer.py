"""KD-optimizer concern (RK-3 of the Router-KD plugin-architecture refactor).

Home of the Router-KD optimizer concern, extracted from the legacy
``stage5_router_kd.py`` monolith. RK-3 covers THREE pieces with TWO patterns:

Piece A — relocated verbatim (the S3-2/S3-3/S4-3 / RK-2 pattern):
  ``_move_optimizer_state_to_device`` is a STANDALONE function in the monolith.
  It is relocated here character-for-character; the ``stage5_router_kd.py``
  monolith re-imports it (``# noqa: F401`` block) so ``run()`` and its two
  resume-path call sites keep their import path.

Piece B — reproduced in an inert hook (the S3-4/S4-2 / RK-2 pattern):
  the split-param-group AdamW construction and the ``_lr_lambda`` LR-scheduler
  closure are INLINE ``run()`` code in the monolith (one a code block, the
  other a closure) — there is nothing standalone to relocate. The
  ``build_optimizer`` hook below REPRODUCES that inline logic faithfully; the
  monolith ``run()`` is NOT modified for them. This is an intentional,
  temporary logic duplication that resolves at RK-8 when the monolith ``run()``
  is deleted and this hook is wired live.

Circular-import note (mirror of ``trainable_scope.py``): this module imports
only from ``...pipeline.*`` / ``..context`` / stdlib / torch — NEVER from
``stage5_router_kd`` or ``router_kd.orchestrator`` at any scope (module-top OR
function-local). The monolith re-imports *this* module at load time, so a
``from ..stage5_router_kd import ...`` here would deadlock the import; nothing
in this module does that.

``KdOptimizerPlugin`` is registered-but-INERT at RK-3 — no orchestrator walk
or test invokes its ``build_optimizer`` hook. RK-8 plugs the hook into the
live Router-KD plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch

from ..context import PipelineContext

log = logging.getLogger(__name__)


def _move_optimizer_state_to_device(optim: torch.optim.Optimizer, device) -> None:
    """Move all optimizer state tensors to the target device.

    Required after load_state_dict() when the checkpoint was saved on CPU
    but the training params live on a CUDA device — otherwise the first
    optimizer step silently mixes CPU and CUDA tensors.
    """
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


class KdOptimizerPlugin:
    """Router-KD optimizer plugin (RK-3 — registered-but-INERT).

    Owns the Router-KD optimizer concern: constructing the AdamW optimizer
    over the trainable parameters (with a split router/expert param-group
    layout when merge-repair is active) and the ``LambdaLR`` warmup+cosine
    learning-rate scheduler, plus the resume-path
    ``_move_optimizer_state_to_device`` helper (relocated verbatim above).

    RK-3 covers a mixed pattern: ``_move_optimizer_state_to_device`` is
    relocated verbatim (the monolith re-imports it), while the split-param-group
    AdamW construction and the ``_lr_lambda`` LR-scheduler closure — inline
    ``run()`` code in the monolith — are reproduced in the ``build_optimizer``
    hook below; the monolith ``run()`` is NOT modified for them (see module
    docstring). RK-3 wires this class into the plugin registry as metadata only
    — no walk or test invokes ``build_optimizer``. RK-8 plugs the hook into the
    live Router-KD plugin sequencer.
    """

    name = "kd_optimizer"
    paper = "Router Knowledge Distillation (paper 2603.02217, Eq. 3)."
    config_key = "stage5_router_kd.learning_rate"
    # ``student``/``model`` are the primary/fallback slot for the model the
    # hook reads (RK-8 will canonicalize to one); both are declared here.
    # ``merge_repair_grad_handles`` / ``total_optim_steps`` are optional
    # upstream slots — guarded with has() in the hook.
    reads: tuple[str, ...] = (
        "student", "model", "config",
        "merge_repair_grad_handles", "total_optim_steps",
    )
    # The hook publishes the constructed optimizer + LR scheduler.
    writes: tuple[str, ...] = ("optimizer", "lr_scheduler")
    # Empty: building the optimizer needs no calibration pass.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — building the optimizer is UNCONDITIONAL.

        Every Router-KD run must construct an optimizer + LR scheduler before
        training; ``config_key`` only names the learning rate, it never gates
        the plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def build_optimizer(self, ctx: PipelineContext) -> None:
        """Phase hook — Router-KD optimizer construction (RK-8 wiring surface).

        INERT at RK-3: no orchestrator walk or test invokes this hook. RK-8
        replaces the Router-KD orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        optimizer + LR-scheduler construction. The body below reproduces those
        two inline blocks faithfully — it is dead code at RK-3 but RK-8 relies
        on it once the monolith ``run()`` is deleted.

        Reproduces (in monolith order): the split-param-group AdamW
        construction (router group ``weight_decay=_wd`` + expert group ``0.0``
        when ``merge_repair_grad_handles`` is present, else a single
        ``weight_decay=_wd`` group), then the warmup+cosine ``LambdaLR``
        scheduler — including the load-bearing ``(current_step + 1)``
        off-by-one in the warmup branch.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. The student may be published under either "student"
        # or "model"; has()-guard the preferred slot.
        student = ctx.get("student") if ctx.has("student") else ctx.get("model")
        config = ctx.get("config")
        s5 = config["stage5_router_kd"]
        # Optional upstream slots — absent is a valid state (no merge-repair /
        # scheduler not yet sized). has()-guard them.
        _merge_repair_grad_handles = (
            ctx.get("merge_repair_grad_handles")
            if ctx.has("merge_repair_grad_handles")
            else None
        )
        # RK-8 wiring contract: the orchestrator MUST populate total_optim_steps
        # before dispatching this hook. The 0 fallback exists only so the inert
        # hook never KeyErrors at RK-3; total_optim_steps == 0 yields a degenerate
        # (warmup_steps == 1, flat) scheduler — never a valid live run.
        total_optim_steps = (
            ctx.get("total_optim_steps")
            if ctx.has("total_optim_steps")
            else 0
        )

        # --- Piece (a): split-param-group AdamW construction ---
        # Optimizer constructed AFTER freezing so it only receives parameters
        # that have requires_grad=True at construction time. weight_decay is
        # config-driven (default 0.0 to match the pre-2026-05-13 baseline).
        _wd = float(s5.get("weight_decay", 0.0))
        _trainable_params = [p for p in student.parameters() if p.requires_grad]
        if _merge_repair_grad_handles:
            # merge-repair unfroze whole stacked expert tensors; a gradient-mask
            # hook zeroes every non-centroid row so only the merged centroids
            # get gradient. AdamW weight decay, however, is applied to the
            # *parameter* independently of its gradient — with weight_decay>0 it
            # would drift every non-centroid expert row too. Put the expert
            # tensors in their own param group with weight_decay=0.0 (the mask
            # still selects rows).
            _expert_ids = set(_merge_repair_grad_handles)
            _expert_params = [p for p in _trainable_params if id(p) in _expert_ids]
            _router_params = [p for p in _trainable_params if id(p) not in _expert_ids]
            optim = torch.optim.AdamW(
                [
                    {"params": _router_params, "weight_decay": _wd},
                    {"params": _expert_params, "weight_decay": 0.0},
                ],
                lr=s5["learning_rate"],
            )
        else:
            optim = torch.optim.AdamW(
                _trainable_params,
                lr=s5["learning_rate"],
                weight_decay=_wd,
            )

        # --- Piece (b): warmup+cosine LambdaLR scheduler ---
        # Constructed AFTER total_optim_steps is known so the warmup horizon and
        # cosine endpoint align with the real step count.
        _lr_schedule = str(s5.get("lr_schedule", "none"))
        _warmup_ratio = float(s5.get("warmup_ratio", 0.05))
        _lr_min_ratio = float(s5.get("lr_min_ratio", 0.10))
        warmup_steps = max(1, int(total_optim_steps * _warmup_ratio))

        def _lr_lambda(current_step: int) -> float:
            if _lr_schedule == "none":
                return 1.0
            # Off-by-one: LambdaLR with last_epoch=-1 advances to current_step=0
            # on the first .step() call. Use (current_step + 1) in the warmup
            # branch so step 0 fires at LR = 1/warmup_steps, not 0.
            if current_step < warmup_steps:
                return (current_step + 1) / warmup_steps
            progress = (current_step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return _lr_min_ratio + (1.0 - _lr_min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)

        ctx.set("optimizer", optim)
        ctx.set("lr_scheduler", scheduler)
