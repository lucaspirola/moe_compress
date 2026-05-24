"""KD-optimizer concern (RK-3 of the Router-KD plugin-architecture refactor).

Paper
-----
Hyeon & Do, "Is Retraining-Free Enough? The Necessity of Router
Calibration for Efficient MoE Compression" — arXiv:2603.02217 (§F.3,
Eq. 3, Table 1). audit/spec_compliance/01_papers/2603.02217/source.md.

Equation 3 (abbreviated; per-sequence form): the vocab-KL distillation
objective
    L_KD = KL(softmax(s_t / τ) || softmax(s_s / τ)) · τ²
where ``s_t``, ``s_s`` are the teacher and student vocabulary logits
and ``τ`` is the distillation temperature. The full paper form
``L_KD = (τ²/N_x) Σ_t m_{t+1} · KL(...)`` includes a per-sequence
``1/N_x`` normalizer and a next-token loss mask ``m_{t+1}``; both are
applied in the :mod:`router_kd.plugins.vocab_kd` plugin (which owns the
loss). This module owns only the optimizer / scheduler construction —
the abbreviated form is sufficient context here.

§F.3 fixes the calibration data and hyperparameters; Table 1 reports
the resulting recovery on Mixtral/Qwen-MoE post-pruning/post-merging.

Official code
-------------
**None published.** Verified 2026-05: the paper's source.md contains
no code link; first author Sieun Hyeon (Seoul National University) has
no public router-KD repo.

Calibration deviation D11 (SHARED with Stage 2 / Stage 2.5)
-----------------------------------------------------------
Paper §F.3 Table 1 uses ``c4``. The project uses multi-domain
Nemotron-Cascade-2-SFT-Data with weighted subsets — task-aware
calibration better matches target deployment distribution. The D11
row's canonical owner is :mod:`stage2.plugins.reap_scoring`.

Home of the Router-KD optimizer concern, extracted from the legacy
``stage5_router_kd.py`` monolith. RK-3 covers two pieces:

Piece A — relocated verbatim (the S3-2/S3-3/S4-3 / RK-2 pattern):
  ``_move_optimizer_state_to_device`` is a STANDALONE function originally
  defined in the monolith. It is relocated here character-for-character; the
  ``stage5_router_kd.py`` shim re-imports it (``# noqa: F401`` block) so
  external callers keep their import path resolvable.

Piece B — the ``build_optimizer`` hook (now LIVE, post-RK-8):
  the split-param-group AdamW construction and the ``_lr_lambda`` LR-scheduler
  closure used to be INLINE ``run()`` code in the monolith. As of RK-8 the
  monolith has been demoted to a thin shim (see :mod:`moe_compress.stage5_router_kd`)
  and ``router_kd.orchestrator.run`` is the real sequencer: it calls
  ``walk_phases(("build_optimizer",), plugins, run_ctx)`` and consumes
  ``optimizer`` / ``lr_scheduler`` from the context. This hook is now the SOLE
  OWNER of optimizer + LR-scheduler construction — there is no duplicate
  inline path anywhere else in the codebase.

Deviation D-merge-repair-grad-flow (xref — canonical owner: merge_repair plugin)
-------------------------------------------------------------------------------
Paper §5 (source.md L824–834) specifies the Eq. 3 contract that "gradients are
backpropagated and applied exclusively to the student router parameters θR,
while all expert and backbone parameters remain frozen". When the
Stage-2.5 *merge-repair* path (Direction E) is active, ``merge_repair`` unfreezes
the merged centroid rows of the stacked expert tensors and registers a
gradient-mask hook so only those rows accumulate gradient. The
:class:`KdOptimizerPlugin` builds a SECOND AdamW param group (``weight_decay=0.0``)
holding those expert tensors so that the optimizer actually steps the unfrozen
rows. This is a real deviation from the paper's strict frozen-experts contract;
its canonical declaration / audit row lives with the owner of the unfreeze
decision — :mod:`router_kd.plugins.merge_repair` — and the split-group AdamW
branch in :meth:`KdOptimizerPlugin.build_optimizer` is the downstream realization
of it. The flag-off path (merge-repair disabled) is byte-identical to a single
``weight_decay=_wd`` AdamW group over the trainable router params and is paper-
faithful w.r.t. Eq. 3 frozen-parameter scope.

Hyperparameter audit note (project additions vs paper baseline)
---------------------------------------------------------------
Paper Table 1 (source.md L3582–3594) specifies LR = 5×10⁻⁵, epochs = 1,
batch size 2, gradient accumulation 4, max sequence length 512, τ = 1.0,
and 3000 calibration samples. It does NOT specify warmup, cosine decay,
weight decay, or split param groups. ``lr_schedule``, ``warmup_ratio``,
``lr_min_ratio`` and ``weight_decay`` are project additions; their defaults
(``lr_schedule="none"``, ``weight_decay=0.0``) reproduce the paper baseline
(constant LR, no WD). The split-group AdamW branch only activates when
merge-repair is enabled (see D-merge-repair-grad-flow above).

Circular-import note (mirror of ``trainable_scope.py``): this module imports
only from ``...pipeline.*`` / ``..context`` / stdlib / torch — NEVER from
``stage5_router_kd`` or ``router_kd.orchestrator`` at any scope (module-top OR
function-local). The shim re-imports *this* module at load time, so a
``from ..stage5_router_kd import ...`` here would deadlock the import; nothing
in this module does that.

``KdOptimizerPlugin`` is LIVE post-RK-8: the orchestrator's
``walk_phases(("build_optimizer",), ...)`` call (orchestrator.py L318) is the
sole driver of the ``build_optimizer`` hook; the resulting ``optimizer`` and
``lr_scheduler`` ctx slots are consumed by the orchestrator's training loop.
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
    """Router-KD optimizer plugin (LIVE post-RK-8 — sole owner).

    Owns the Router-KD optimizer concern: constructing the AdamW optimizer
    over the trainable parameters (with a split router/expert param-group
    layout when merge-repair is active) and the ``LambdaLR`` warmup+cosine
    learning-rate scheduler, plus the resume-path
    ``_move_optimizer_state_to_device`` helper (relocated verbatim above).

    Post-RK-8 this plugin is LIVE: ``router_kd.orchestrator.run`` calls
    ``walk_phases(("build_optimizer",), plugins, run_ctx)`` and consumes the
    ``optimizer`` / ``lr_scheduler`` ctx slots this hook publishes. The legacy
    monolith ``run()`` has been demoted to a thin shim that delegates to the
    orchestrator (see :mod:`moe_compress.stage5_router_kd`); there is no
    duplicate inline AdamW/LambdaLR path anywhere in the codebase.
    """

    name = "kd_optimizer"
    paper = (
        "Router KD vocab-KL distillation Eq. 3 — arXiv:2603.02217 "
        "(Hyeon & Do); no official code. Concern: split-param-group AdamW + scheduler + optimizer-state device-move. "
        "Calibration D11 (SHARED — see :mod:`stage2.plugins.reap_scoring`). "
        "See module docstring."
    )
    # Dotted-path form encodes the nested location of the descriptive key
    # (config["stage5_router_kd"]["learning_rate"]); ``config_key`` here is
    # descriptive metadata only — it does NOT gate the plugin (see
    # ``is_enabled`` below, which is unconditionally True).
    config_key = "stage5_router_kd.learning_rate"
    # ``student``/``model`` are the primary/fallback slot for the model the
    # hook reads (the orchestrator may publish either depending on phase);
    # both are declared here. ``merge_repair_grad_handles`` is an optional
    # upstream slot (absent when merge-repair is disabled); ``total_optim_steps``
    # is REQUIRED at dispatch time (the hook raises if missing).
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
        """Phase hook — Router-KD optimizer construction (LIVE post-RK-8).

        Driven by ``router_kd.orchestrator.run`` via
        ``walk_phases(("build_optimizer",), plugins, run_ctx)`` (orchestrator.py
        L318); the orchestrator consumes ``optimizer`` / ``lr_scheduler`` from
        the context immediately afterward. This hook is the SOLE owner of
        Router-KD optimizer + scheduler construction.

        Builds (in two pieces): the split-param-group AdamW (router group
        ``weight_decay=_wd`` + expert group ``0.0`` when
        ``merge_repair_grad_handles`` is present — see
        ``D-merge-repair-grad-flow`` in the module docstring; else a single
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
        # Optional upstream slots — absent is a valid state (no merge-repair).
        _merge_repair_grad_handles = (
            ctx.get("merge_repair_grad_handles")
            if ctx.has("merge_repair_grad_handles")
            else None
        )
        # Live-wiring contract: the orchestrator MUST populate total_optim_steps
        # before dispatching this hook (orchestrator.py L310-312). Missing or
        # zero is a wiring bug — fail loudly here rather than silently
        # constructing a degenerate (warmup_steps == 1, flat) scheduler.
        if not ctx.has("total_optim_steps"):
            raise RuntimeError(
                "KdOptimizerPlugin.build_optimizer: ctx slot 'total_optim_steps' "
                "is missing — the orchestrator must compute it before dispatching "
                "the build_optimizer phase (see router_kd/orchestrator.py L310-312)."
            )
        total_optim_steps = int(ctx.get("total_optim_steps"))
        if total_optim_steps <= 0:
            raise RuntimeError(
                "KdOptimizerPlugin.build_optimizer: total_optim_steps must be "
                f"positive; got {total_optim_steps}. This indicates an empty "
                "calibration batches list or epochs<=0 — a Router-KD run with "
                "zero optimizer steps is never valid."
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
