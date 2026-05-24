"""Early-stop concern (RK-7 of the Router-KD plugin-architecture refactor).

Paper
-----
Hyeon & Do, "Is Retraining-Free Enough? The Necessity of Router
Calibration for Efficient MoE Compression" — arXiv:2603.02217 (§F.3,
Eq. 3, Table 1). audit/spec_compliance/01_papers/2603.02217/source.md.

Equation 3: the per-batch vocab-KL distillation objective
    L_KD = KL(softmax(s_t / τ) || softmax(s_s / τ)) · τ²
where ``s_t``, ``s_s`` are the teacher and student vocabulary logits
and ``τ`` is the distillation temperature.

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

Home of the Router-KD *best-tracker + early-stop* concern — the last
Router-KD plugin extraction — carved out of the legacy
``stage5_router_kd.py`` monolith. RK-7 is a MIXED pattern: one Pattern-A
relocation plus one Pattern-B inline reproduction.

Piece A — relocated verbatim (Pattern A, the RK-2/RK-3/RK-4/RK-6 pattern):
  ONE STANDALONE module-level function is relocated here character-for-
  character — ``_save_best_router_state`` (the atomic ``best.pt`` writer:
  it snapshots the trainable router params to a slim payload and rewrites
  ``best.pt`` with the ``torch.save`` → ``os.fsync`` → ``os.replace`` atomic
  dance). It is relocated verbatim; the ``stage5_router_kd.py`` monolith
  re-imports it (``# noqa: F401`` block) so ``run()`` keeps its import path.

Piece B — the inert hooks (Pattern B): :class:`EarlyStopPlugin` carries
four phase hooks — ``setup_early_stop`` / ``update_best_tracker`` /
``check_early_stop`` / ``reload_best_checkpoint`` — that REPRODUCE the
inline best-tracker / early-stop glue scattered through the monolith
``run()`` (the best-tracker + early-stop state init and the resume-restore;
the per-log-window EMA-update / ``best.pt``-save / patience-counter loop
body; the early-stop break DECISION; the end-of-training ``best.pt``
reload). They are INERT at RK-7 — no orchestrator walk or test invokes
them, and the monolith ``run()`` is NOT modified for any of the Pattern-B
glue: the inline early-stop code stays exactly where it was. The hooks
exist as the RK-8 wiring surface; RK-8 plugs them into the live Router-KD
plugin sequencer and deletes the monolith ``run()``.

NOTE — the orchestrator/monolith boundary: :meth:`check_early_stop` only
makes the early-stop DECISION (it sets the ``early_stop_should_stop`` flag
and logs the stop line). The actual loop ``break`` and the final
crash-resume checkpoint write stay orchestrator-level / monolith-owned —
the plugin owns the *policy*, not the loop control flow.

Unconditional ``is_enabled`` (mirror of ``VocabKdPlugin``, NOT the
stage-gated ``MergeRepairPlugin``): every Router-KD run carries a
best-tracker (it always exports the best ``best.pt`` it saw), so the plugin
is always enabled. The early-stop *patience* block is gated INTERNALLY on
``early_stop_patience > 0`` inside the hooks — ``config_key`` only names
that knob, it does not gate the plugin as a whole.

Circular-import note (mirror of ``vocab_kd.py`` / ``merge_repair.py``):
this module imports only from ``..context`` / stdlib / torch — NEVER from
``stage5_router_kd`` or ``router_kd.orchestrator`` at any scope (module-top
OR function-local). The monolith re-imports *this* module at load time, so
a ``from ..stage5_router_kd import ...`` here would deadlock the import;
nothing in this module does that.

``EarlyStopPlugin`` is registered-but-INERT at RK-7 — RK-8 plugs the hooks
into the live Router-KD plugin sequencer.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..context import PipelineContext

log = logging.getLogger(__name__)


def _save_best_router_state(
    partial_dir: Path,
    student: nn.Module,
    step: int,
    epoch: int,
    raw_kl_ema: float,
) -> None:
    """Atomically rewrite best.pt with the trainable (router) params only.

    File size is ~10-50 MB (router weights only) vs ~5 GB for the full
    optim+student checkpoint, so we can afford to rewrite on every
    improvement. The slim payload also keeps the end-of-training reload
    boundaried: only trainable params land via load_state_dict(strict=False).
    """
    unwrapped = getattr(student, "_orig_mod", student)
    router_state = {
        name: p.data.cpu().clone()
        for name, p in unwrapped.named_parameters()
        if p.requires_grad
    }
    payload = {
        "format_version": 1,  # best.pt format; independent of step_*.pt versioning
        "step": int(step),
        "epoch": int(epoch),
        "raw_kl_ema": float(raw_kl_ema),
        "router_state": router_state,
    }
    tmp = partial_dir / "best.pt.tmp"
    final = partial_dir / "best.pt"
    torch.save(payload, tmp)
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)


class EarlyStopPlugin:
    """Router-KD best-tracker + early-stop plugin (RK-7 — registered-but-INERT).

    Owns the Router-KD *best-tracker + early-stop* concern: the EMA-smoothed
    raw-KL tracking, the ``best.pt`` save-on-improvement, the patience-based
    early-stop decision and the end-of-training best-checkpoint reload. The
    one standalone function above (``_save_best_router_state``) is relocated
    verbatim (the monolith re-imports it).

    Best-tracking is UNCONDITIONAL — every Router-KD run exports the best
    ``best.pt`` it saw — so :meth:`is_enabled` always returns ``True``
    (mirroring ``VocabKdPlugin``, NOT the stage-gated ``MergeRepairPlugin``).
    The early-stop *patience* block is gated INTERNALLY on
    ``early_stop_patience > 0`` inside the hooks.

    The four hooks (``setup_early_stop`` / ``update_best_tracker`` /
    ``check_early_stop`` / ``reload_best_checkpoint``) REPRODUCE the inline
    best-tracker / early-stop ``run()`` glue; they are INERT at RK-7 — no
    orchestrator walk or test invokes them, and the monolith ``run()`` is
    untouched. RK-8 plugs them into the live Router-KD plugin sequencer.
    """

    name = "early_stop"
    paper = (
        "Router KD vocab-KL distillation Eq. 3 — arXiv:2603.02217 "
        "(Hyeon & Do); no official code. Concern: best-tracker EMA + best.pt save + patience early-stop. "
        "Calibration D11 (SHARED — see :mod:`stage2.plugins.reap_scoring`). "
        "See module docstring."
    )
    config_key = "stage5_router_kd.early_stop_patience"
    # ``config`` drives the one-time setup (EMA alpha, save_best, patience);
    # ``partial_dir`` is the best.pt / resume-checkpoint directory; ``student``
    # is snapshotted into best.pt and reloaded from it; ``step`` / ``epoch`` /
    # ``raw_kl_val`` are the per-window training-loop signals; ``stage_key`` is
    # read only for log lines. The resume-restore reads the optional
    # ``resume_*`` slots the checkpoint loader publishes; the per-window /
    # decision hooks re-read the best-tracker state the setup hook published.
    reads: tuple[str, ...] = (
        "config", "partial_dir", "student", "stage_key",
        "step", "epoch", "raw_kl_val",
        "resume_best_raw_kl_ema", "resume_best_step", "resume_prev_ema",
        "resume_no_improve_windows", "resume_es_ref_ema",
        "best_ema_alpha", "save_best", "best_raw_kl_ema", "best_step",
        "prev_ema", "early_stop_patience", "no_improve_windows",
        "es_ref_ema",
    )
    # The setup hook publishes the best-tracker + early-stop state; the
    # per-window hook updates the EMA / running minima / patience counter and
    # publishes the window EMA; the decision hook publishes the stop flag.
    writes: tuple[str, ...] = (
        "best_ema_alpha", "save_best", "best_raw_kl_ema", "best_step",
        "prev_ema", "early_stop_patience", "no_improve_windows",
        "es_ref_ema", "raw_kl_ema", "early_stop_should_stop",
    )
    # Empty: the best-tracker / early-stop needs no separate calibration pass.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — best-tracking is UNCONDITIONAL.

        Every Router-KD run tracks the best EMA-smoothed raw-KL and exports
        the corresponding ``best.pt``; ``config_key`` only names the
        early-stop patience knob, it never gates the plugin as a whole. The
        patience block itself is gated INTERNALLY on
        ``early_stop_patience > 0`` inside the hooks — so this mirrors
        ``VocabKdPlugin``'s unconditional gate, not ``MergeRepairPlugin``'s
        stage-gated one.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def setup_early_stop(self, ctx: PipelineContext) -> None:
        """One-time setup hook — best-tracker + early-stop state init.

        INERT at RK-7: no orchestrator walk or test invokes this hook. RK-8
        calls it once before the epoch loop. The body reproduces the monolith
        ``run()`` inline best-tracker + early-stop setup block (and the
        resume-restore) faithfully — dead code at RK-7 but RK-8 + unit tests
        rely on it.

        Reproduces (in monolith order): read ``best_metric_ema_alpha`` /
        ``save_best`` from ``stage5_router_kd`` config and seed the
        best-tracker (``best_raw_kl_ema`` / ``best_step`` / ``prev_ema``);
        read + validate ``early_stop_patience`` (must be >= 0) and seed the
        early-stop state (``no_improve_windows`` / ``es_ref_ema``); then apply
        the resume-restore — when a v2 resume checkpoint published the
        optional ``resume_*`` slots, overwrite the freshly-seeded state from
        them so a crash-resume does not lose accumulated patience. Publishes
        ``best_ema_alpha`` / ``save_best`` / ``best_raw_kl_ema`` /
        ``best_step`` / ``prev_ema`` / ``early_stop_patience`` /
        ``no_improve_windows`` / ``es_ref_ema``.
        """
        config = ctx.get("config")
        s5 = config["stage5_router_kd"]

        # Best-tracker state. Initialized here so the resume-restore below can
        # overwrite when restarting from a v2 checkpoint.
        best_ema_alpha = float(s5.get("best_metric_ema_alpha", 0.2))
        save_best = bool(s5.get("save_best", True))
        best_raw_kl_ema = float("inf")
        best_step = -1
        prev_ema = float("inf")

        # --- Early stopping (2026-05-17 overfit fix) ---
        # Patience-based on the SAME raw_kl EMA the best-tracker uses. After
        # `early_stop_patience` consecutive log windows with no improvement,
        # training stops cleanly. 0 = disabled. The no-improve counter is
        # persisted in the resume checkpoint so a crash-resume does not lose
        # accumulated patience.
        early_stop_patience = int(s5.get("early_stop_patience", 0))
        if early_stop_patience < 0:
            raise ValueError(
                f"Stage 5: early_stop_patience={early_stop_patience} must be "
                ">= 0 (0 disables early stopping)."
            )
        no_improve_windows = 0
        # Running-minimum raw_kl EMA used purely by the early-stop patience
        # test. Distinct from best_raw_kl_ema, which is only updated when
        # save_best is on — es_ref_ema must track improvement even with
        # save_best=false so early stopping is independent of the
        # checkpoint-export knob.
        es_ref_ema = float("inf")

        # --- Resume restore for the best-tracker + early-stop state ---
        # A v2 resume checkpoint publishes the optional resume_* slots; absent
        # is a valid state (v1 checkpoint or no resume) — the freshly-seeded
        # +inf / 0 state then stands and the next log boundary bootstraps.
        if ctx.has("resume_best_raw_kl_ema"):
            resume_best_raw_kl_ema = ctx.get("resume_best_raw_kl_ema")
            if resume_best_raw_kl_ema is not None:
                best_raw_kl_ema = float(resume_best_raw_kl_ema)
                resume_best_step = (
                    ctx.get("resume_best_step")
                    if ctx.has("resume_best_step")
                    else None
                )
                best_step = (
                    int(resume_best_step)
                    if resume_best_step is not None
                    else -1
                )
        if ctx.has("resume_prev_ema"):
            resume_prev_ema = ctx.get("resume_prev_ema")
            if resume_prev_ema is not None:
                prev_ema = float(resume_prev_ema)
        if ctx.has("resume_no_improve_windows"):
            resume_no_improve_windows = ctx.get("resume_no_improve_windows")
            if resume_no_improve_windows is not None:
                no_improve_windows = int(resume_no_improve_windows)
        if ctx.has("resume_es_ref_ema"):
            resume_es_ref_ema = ctx.get("resume_es_ref_ema")
            if resume_es_ref_ema is not None:
                es_ref_ema = float(resume_es_ref_ema)

        ctx.set("best_ema_alpha", best_ema_alpha)
        ctx.set("save_best", save_best)
        ctx.set("best_raw_kl_ema", best_raw_kl_ema)
        ctx.set("best_step", best_step)
        ctx.set("prev_ema", prev_ema)
        ctx.set("early_stop_patience", early_stop_patience)
        ctx.set("no_improve_windows", no_improve_windows)
        ctx.set("es_ref_ema", es_ref_ema)

    def update_best_tracker(self, ctx: PipelineContext) -> None:
        """Per-log-window hook — EMA update + best.pt save + patience counter.

        INERT at RK-7: no orchestrator walk or test invokes this hook. RK-8
        dispatches it once per log window inside the optimizer-step block. The
        body reproduces the monolith ``run()`` inline per-window best-tracker
        + early-stop-counter block faithfully — dead code at RK-7 but RK-8 +
        unit tests rely on it.

        Reproduces (in monolith order): EMA of ``raw_kl_val`` across log
        boundaries (``prev_ema=+inf`` bootstraps ``ema = raw_kl_val``); the
        save-best comparison (``save_best`` AND ``ema < best_raw_kl_ema`` →
        update best + ``_save_best_router_state``); then the early-stop
        counter — gated on ``early_stop_patience > 0`` — which resets
        ``no_improve_windows`` on a window improving the ``es_ref_ema``
        running minimum and increments it otherwise (``es_ref_ema`` advances
        independent of ``save_best``, so early stopping works even with
        ``save_best=false``). Publishes the window ``raw_kl_ema`` and the
        updated ``prev_ema`` / ``best_raw_kl_ema`` / ``best_step`` /
        ``no_improve_windows`` / ``es_ref_ema``.
        """
        partial_dir = ctx.get("partial_dir")
        student = ctx.get("student")
        step = int(ctx.get("step"))
        epoch = int(ctx.get("epoch"))
        raw_kl_val = float(ctx.get("raw_kl_val"))
        best_ema_alpha = float(ctx.get("best_ema_alpha"))
        save_best = bool(ctx.get("save_best"))
        best_raw_kl_ema = float(ctx.get("best_raw_kl_ema"))
        best_step = int(ctx.get("best_step"))
        prev_ema = float(ctx.get("prev_ema"))
        early_stop_patience = int(ctx.get("early_stop_patience"))
        no_improve_windows = int(ctx.get("no_improve_windows"))
        es_ref_ema = float(ctx.get("es_ref_ema"))

        # EMA of raw_kl across log boundaries. prev_ema=+inf on first
        # observation triggers a bootstrap (ema = raw_kl_val).
        if math.isinf(prev_ema):
            ema = raw_kl_val
        else:
            ema = best_ema_alpha * raw_kl_val + (1.0 - best_ema_alpha) * prev_ema
        prev_ema = ema

        # Save-best by EMA-smoothed raw KL. +inf seed of best_raw_kl_ema
        # guarantees the first log boundary always writes a best.pt, so the
        # run always exports SOMETHING even if it crashes before any
        # improvement.
        if save_best and ema < best_raw_kl_ema:
            best_raw_kl_ema = ema
            best_step = step
            _save_best_router_state(partial_dir, student, step, epoch, ema)

        # --- Early-stopping counter (2026-05-17 overfit fix) ---
        # `es_ref_ema` holds the best raw_kl EMA seen so far; it is
        # independent of `save_best` (which gates only the best.pt WRITE) so
        # early stopping works even with save_best=false. A window that
        # improves on the running minimum resets the no-improve counter; one
        # that does not increments it. With early_stop_patience == 0 the whole
        # block is skipped.
        if early_stop_patience > 0:
            if ema < es_ref_ema:
                no_improve_windows = 0
            else:
                no_improve_windows += 1
            es_ref_ema = min(es_ref_ema, ema)

        ctx.set("raw_kl_ema", ema, overwrite=True)
        ctx.set("prev_ema", prev_ema, overwrite=True)
        ctx.set("best_raw_kl_ema", best_raw_kl_ema, overwrite=True)
        ctx.set("best_step", best_step, overwrite=True)
        ctx.set("no_improve_windows", no_improve_windows, overwrite=True)
        ctx.set("es_ref_ema", es_ref_ema, overwrite=True)

    def check_early_stop(self, ctx: PipelineContext) -> None:
        """Per-window hook — the early-stop DECISION (RK-8 wiring surface).

        INERT at RK-7: no orchestrator walk or test invokes this hook. RK-8
        dispatches it right after :meth:`update_best_tracker`. The body
        reproduces the monolith ``run()`` early-stop DECISION faithfully —
        dead code at RK-7 but RK-8 + unit tests rely on it.

        BOUNDARY: this hook only makes the *decision* — it sets the
        ``early_stop_should_stop`` flag (``True`` once
        ``no_improve_windows >= early_stop_patience``, with
        ``early_stop_patience > 0``) and logs the stop line. The actual loop
        ``break`` and the final crash-resume checkpoint write stay
        orchestrator-level / monolith-owned: the plugin owns the early-stop
        *policy*, not the training-loop control flow. With
        ``early_stop_patience == 0`` the flag stays ``False`` — byte-identical
        to pre-2026-05-17 ``main``.

        Reproduces: the ``no_improve_windows >= early_stop_patience`` test and
        the stop-line ``log.info``. Publishes ``early_stop_should_stop``.
        """
        early_stop_patience = int(ctx.get("early_stop_patience"))
        no_improve_windows = int(ctx.get("no_improve_windows"))
        should_stop = (
            early_stop_patience > 0
            and no_improve_windows >= early_stop_patience
        )
        if should_stop:
            stage_key = ctx.get("stage_key")
            step = int(ctx.get("step"))
            best_raw_kl_ema = float(ctx.get("best_raw_kl_ema"))
            best_step = int(ctx.get("best_step"))
            log.info(
                "Stage %s: early stopping at step %d — raw_kl EMA did not "
                "improve for %d consecutive log windows "
                "(early_stop_patience=%d). best_ema=%.6f@step%d.",
                stage_key, step, no_improve_windows, early_stop_patience,
                best_raw_kl_ema, best_step,
            )
        ctx.set("early_stop_should_stop", should_stop, overwrite=ctx.has(
            "early_stop_should_stop"
        ))

    def reload_best_checkpoint(self, ctx: PipelineContext) -> None:
        """End-of-training hook — best.pt param swap (RK-8 wiring surface).

        INERT at RK-7: no orchestrator walk or test invokes this hook. RK-8
        calls it once after the epoch loop, before the final export. The body
        reproduces the monolith ``run()`` inline best-checkpoint reload block
        faithfully — dead code at RK-7 but RK-8 + unit tests rely on it.

        Reproduces: if ``save_best`` was active and a ``best.pt`` was written
        during training, load it and swap the trainable (router) params for
        the best snapshot before export. The bulk of the model (frozen, not
        in best.pt) stays at its current state — that is the whole point of
        saving only the trainable subset. ``best.pt`` carrying any non-router
        (unexpected) key is fail-loud. No-op when ``save_best`` is off or no
        ``best.pt`` exists (the best-tracker never fired).
        """
        save_best = bool(ctx.get("save_best"))
        if not save_best:
            return
        partial_dir = ctx.get("partial_dir")
        student = ctx.get("student")
        stage_key = ctx.get("stage_key")
        best_path = partial_dir / "best.pt"
        if best_path.exists():
            best_blob = torch.load(best_path, map_location="cpu")
            base = getattr(student, "_orig_mod", student)
            missing, unexpected = base.load_state_dict(
                best_blob["router_state"], strict=False
            )
            log.info(
                "Stage %s: reloaded best router state from step=%d "
                "(raw_kl_ema=%.6f); missing=%d (expected — non-router params "
                "not in best), unexpected=%d",
                stage_key, int(best_blob.get("step", -1)),
                float(best_blob.get("raw_kl_ema", float("nan"))),
                len(missing), len(unexpected),
            )
            if unexpected:
                raise RuntimeError(
                    f"Stage {stage_key}: best.pt contains unexpected keys "
                    f"(non-router params leaked into best snapshot): "
                    f"{unexpected[:5]}"
                )
        else:
            log.warning(
                "Stage %s: save_best=true but no best.pt found in %s — "
                "exporting last-step state (best-tracker never fired)",
                stage_key, partial_dir,
            )


__all__ = [
    "EarlyStopPlugin",
    "_save_best_router_state",
]
