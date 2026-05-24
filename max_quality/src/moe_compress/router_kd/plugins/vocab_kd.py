"""Vocab-KD KD-loss concern (RK-4 of the Router-KD plugin-architecture refactor).

Home of the Router-KD KD-loss concern, extracted from the legacy
``stage5_router_kd.py`` monolith. RK-4 is a PURE Pattern A relocation: FIVE
STANDALONE module-level functions are relocated here character-for-character —
nothing is reproduced inline.

Piece A — relocated verbatim (the S3-2/S3-3/S4-3 / RK-2 / RK-3 pattern):
  ``_chunked_vocab_kl`` (the temperature-scaled chunked vocab-KL kernel),
  ``_combine_kd_loss`` (the ``kl + w·mse`` loss combiner) and the three NaN
  sanity probes ``_log_first_batch_sanity`` / ``_dump_nan_diagnostics`` /
  ``_check_param_sanity`` are STANDALONE functions in the monolith. They are
  relocated here verbatim; the ``stage5_router_kd.py`` monolith re-imports them
  (``# noqa: F401`` block) so ``run()`` and external callers/tests
  (``test_stage5_merge_repair.py``) keep their import paths. The probes were
  the "Debug instrumentation" block added 2026-05-13 after the Stage 2.5 NaN
  crash on vast.ai B200 contract 36639423; they provide a first-batch sanity
  probe, a NaN tripwire and a periodic param-sanity scan around the KD loss.

Circular-import note (mirror of ``trainable_scope.py`` / ``kd_optimizer.py``):
this module imports only from ``...pipeline.*`` / ``..context`` / stdlib /
torch — NEVER from ``stage5_router_kd`` or ``router_kd.orchestrator`` at any
scope (module-top OR function-local). The monolith re-imports *this* module at
load time, so a ``from ..stage5_router_kd import ...`` here would deadlock the
import; nothing in this module does that.

``VocabKdPlugin`` is registered-but-INERT at RK-4 — no orchestrator walk or
test invokes its ``compute_kd_loss`` hook. RK-8 plugs the hook into the live
Router-KD plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..context import PipelineContext

log = logging.getLogger(__name__)


def _chunked_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute vocab-level KL(teacher ‖ student) in sequence chunks.

    Processes ``chunk_size`` sequence positions at a time to bound peak
    intermediate memory. At chunk_size=128 with |V|=150K and B=4:
      Peak intermediate per chunk ≈ 4 × 128 × 150K × 4 bytes ≈ 300 MB
      vs ≈1.2 GB for the full sequence at L=512.

    Returns scalar loss = (τ²/N_tokens) × Σ_t KL(teacher_t ‖ student_t).

    Note: n_tokens = B × (L−1) is the per-position-mean denominator (paper
    Eq. 3's N_x for fully-packed sequences with no padding).

    ASSUMPTION: fully-packed sequences (no padding) — see spec §8 N_x note.
    Under this invariant, paper Eq. 3's mask `m_{t+1}=1` everywhere and
    `N_x = Σ_t m_{t+1} = B × (L−1) = n_tokens`, so the `+ ε` zero-mask
    safety constant from paper Eq. 3 is unnecessary. If a future calibration
    source ever introduces padding, this normalization (and the `+ ε`) must
    be revisited.
    """
    B, L, V = student_logits.shape
    total_kl = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    n_tokens = 0
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        s_chunk = student_logits[:, start:end, :]
        t_chunk = teacher_logits[:, start:end, :]
        t_p = F.softmax(t_chunk / temperature, dim=-1)
        s_lp = F.log_softmax(s_chunk / temperature, dim=-1)
        chunk_kl = F.kl_div(s_lp, t_p, reduction="none").sum(dim=-1)  # [B, chunk_len]
        total_kl = total_kl + chunk_kl.sum()
        n_tokens += chunk_kl.numel()
        del t_p, s_lp, chunk_kl  # free intermediates eagerly
    return (total_kl / max(n_tokens, 1)) * (temperature ** 2)


def _combine_kd_loss(
    kl_loss: torch.Tensor,
    mse_term: "torch.Tensor | None",
    mse_weight: float,
) -> torch.Tensor:
    """Stage-2.5 total loss = vocab-KL + weighted merge-repair MSE.

    When merge-repair is off the caller passes ``mse_term=None`` and this
    returns the *exact* ``kl_loss`` tensor object — so the flag-off loss is
    byte-identical to pre-Direction-E ``main``. When on, the MSE term is cast
    to the KL's dtype and added with the config-scalar ``mse_weight``.
    """
    if mse_term is None:
        return kl_loss
    return kl_loss + mse_weight * mse_term.to(kl_loss.dtype)


def _log_first_batch_sanity(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """First-batch sanity probe — log forward-pass stats and abort if any
    NaN/Inf is present BEFORE the optimizer runs."""
    try:
        t_finite = bool(torch.isfinite(teacher_logits).all())
        s_finite = bool(torch.isfinite(student_logits).all())
        loss_finite = bool(torch.isfinite(loss))
        log.info(
            "Stage 5 first-batch sanity: "
            "teacher shape=%s dtype=%s finite=%s abs_max=%.3e mean=%.3e std=%.3e ; "
            "student shape=%s dtype=%s finite=%s abs_max=%.3e mean=%.3e std=%.3e ; "
            "initial_loss=%.6e finite=%s",
            tuple(teacher_logits.shape), teacher_logits.dtype, t_finite,
            float(teacher_logits.detach().abs().max()),
            float(teacher_logits.detach().mean()),
            float(teacher_logits.detach().std()),
            tuple(student_logits.shape), student_logits.dtype, s_finite,
            float(student_logits.detach().abs().max()),
            float(student_logits.detach().mean()),
            float(student_logits.detach().std()),
            float(loss.detach()), loss_finite,
        )
        if not (t_finite and s_finite and loss_finite):
            raise RuntimeError(
                "Stage 5 first-batch sanity FAILED: "
                f"teacher_finite={t_finite} student_finite={s_finite} loss_finite={loss_finite}. "
                "Halting before any optimizer step to surface the actual failure mode "
                "(teacher vs student vs KL) instead of training 50 batches of NaN."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        log.warning("Stage 5 first-batch sanity probe raised %s (non-fatal — continuing)", exc)


def _dump_nan_diagnostics(
    *,
    loss: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    student: nn.Module,
    epoch: int,
    step: int,
    batch_i: int,
) -> None:
    """Structured dump on non-finite loss — teacher/student stats + first 5
    non-finite trainable params (routers only)."""
    try:
        def _stats(t: torch.Tensor) -> str:
            t_det = t.detach()
            n_total = max(1, t_det.numel())
            n_nan = int(torch.isnan(t_det).sum())
            n_inf = int(torch.isinf(t_det).sum())
            return (
                f"shape={tuple(t.shape)} dtype={t.dtype} "
                f"abs_max={float(t_det.abs().max()):.3e} mean={float(t_det.mean()):.3e} "
                f"pct_nan={100.0 * n_nan / n_total:.2f} pct_inf={100.0 * n_inf / n_total:.2f}"
            )
        log.error("Stage 5 NaN-tripwire at epoch=%d step=%d batch=%d: loss=%s",
                  epoch, step, batch_i, float(loss.detach()))
        log.error("  teacher logits: %s", _stats(teacher_logits))
        log.error("  student logits: %s", _stats(student_logits))
        bad_params = _check_param_sanity(student, step)
        if bad_params:
            log.error("  non-finite trainable params (first %d): %s", len(bad_params), bad_params)
        else:
            log.error("  all trainable params still finite — NaN originates in forward, not weights.")
    except Exception as exc:
        log.error("Stage 5 NaN diagnostics raised: %s", exc)


def _check_param_sanity(student: nn.Module, step: int) -> list[str]:
    """Cheap O(params) scan: names of trainable params containing NaN/Inf,
    capped at 5 for log brevity."""
    bad: list[str] = []
    base = getattr(student, "_orig_mod", student)
    for name, p in base.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.isfinite(p.data).all():
            bad.append(name)
            if len(bad) >= 5:
                break
    return bad


class VocabKdPlugin:
    """Router-KD vocab-KD KD-loss plugin (RK-4 — registered-but-INERT).

    Owns the Router-KD KD-loss concern: the temperature-scaled chunked
    vocab-KL kernel (``_chunked_vocab_kl``), the ``kl + w·mse`` loss combiner
    (``_combine_kd_loss``) and the three NaN sanity probes
    (``_log_first_batch_sanity`` / ``_dump_nan_diagnostics`` /
    ``_check_param_sanity``) — all relocated verbatim above.

    RK-4 is a PURE Pattern A relocation: the five functions are relocated
    verbatim (the monolith re-imports them) and nothing is reproduced inline.
    RK-4 wires this class into the plugin registry as metadata only — no walk
    or test invokes ``compute_kd_loss``. RK-8 plugs the hook into the live
    Router-KD plugin sequencer.
    """

    name = "vocab_kd"
    paper = "Router Knowledge Distillation (paper 2603.02217, Eq. 3)."
    config_key = "stage5_router_kd.kd_temperature"
    # ``teacher_logits``/``student_logits`` carry the already causally-shifted
    # vocab logits the KD loss consumes; ``merge_repair_mse_term`` /
    # ``merge_repair_mse_weight`` are the optional Direction-E merge-repair
    # slots — guarded with has() in the hook.
    reads: tuple[str, ...] = (
        "teacher_logits", "student_logits", "config",
        "merge_repair_mse_term", "merge_repair_mse_weight",
    )
    # The hook publishes the combined KD loss + the raw vocab-KL term.
    writes: tuple[str, ...] = ("kd_loss", "vocab_kl")
    # Empty: computing the KD loss needs no calibration pass.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — computing the KD loss is UNCONDITIONAL.

        Every Router-KD run distills via the vocab-KL loss; ``config_key``
        only names the distillation temperature, it never gates the plugin as
        a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def compute_kd_loss(self, ctx: PipelineContext) -> None:
        """Phase hook — Router-KD KD-loss assembly (RK-8 wiring surface).

        INERT at RK-4: no orchestrator walk or test invokes this hook. RK-8
        replaces the Router-KD orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()`` per-
        batch loss assembly. The body below reproduces that inline block
        faithfully — it is dead code at RK-4 but RK-8 + unit tests rely on it.

        The ``teacher_logits`` / ``student_logits`` ctx slots carry the
        ALREADY causally-shifted ``[:, :-1, :]`` logits — shifting (predict
        token t+1 from position t) is the caller's job, not this hook's.

        Reproduces (in monolith order): read ``kd_temperature`` /
        ``kd_seq_chunk_size`` from ``stage5_router_kd`` config, compute the
        chunked vocab-KL via ``_chunked_vocab_kl``, then combine it with the
        optional Direction-E merge-repair MSE term via ``_combine_kd_loss``
        (absent term → ``None`` / weight ``0.0``, so the combined loss is the
        exact KL tensor). Publishes the combined ``kd_loss`` and the raw
        ``vocab_kl`` term.
        """
        config = ctx.get("config")
        s5 = config["stage5_router_kd"]
        # The shifted teacher/student vocab logits — required slots.
        teacher_logits = ctx.get("teacher_logits")
        student_logits = ctx.get("student_logits")
        T = float(s5.get("kd_temperature", 1.0))
        seq_chunk = int(s5.get("kd_seq_chunk_size", 512))
        kl_loss = _chunked_vocab_kl(student_logits, teacher_logits, T, chunk_size=seq_chunk)

        # --- Direction E: per-layer merge-repair MSE term ---
        # Optional upstream slots — absent is a valid state (merge-repair off).
        # has()-guard them: no term -> None / weight 0.0, so _combine_kd_loss
        # returns the exact kl_loss tensor (flag-off byte-identity).
        mse_term = (
            ctx.get("merge_repair_mse_term")
            if ctx.has("merge_repair_mse_term")
            else None
        )
        mse_weight = (
            float(ctx.get("merge_repair_mse_weight"))
            if ctx.has("merge_repair_mse_weight")
            else 0.0
        )
        loss = _combine_kd_loss(kl_loss, mse_term, mse_weight)

        ctx.set("kd_loss", loss)
        ctx.set("vocab_kl", kl_loss)
