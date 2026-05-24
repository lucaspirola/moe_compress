"""Thermometer bits-per-token (BPT) metric — Stage 6alt plugin.

Paper / spec source
--------------------
Standard bits-per-token (BPT) metric — ``BPT = mean_NLL / log(2)``.
BPT is the cross-entropy-in-bits-per-token equivalent of PPL
(``PPL = exp(mean_NLL)``); both characterize the model's next-token
distribution on a fixed corpus. No specific paper for the metric —
it is the natural information-theoretic counterpart of PPL going back
to the language-modelling literature (Shannon 1951 entropy +
modern-LM PPL conventions).

Project-original sweep usage: BPT is preferred over PPL for the
thermometer because differences across compressed models are
linear-in-bits (additive) rather than exponential-in-bits
(multiplicative — small NLL gaps compound under ``exp``), giving a
more readable side-by-side ablation table.

Pure forward pass, no generation. ``_bpt_from_nll`` turns mean NLL
into BPT.

Home of the Stage 6alt thermometer BPT-measurement concern. The
thermometer's primary metric is bits-per-token: mean next-token NLL (in
bits) over the fixed evaluation corpus that ``ThermoCorpusPlugin``
produced. Pure forward pass, no generation — ``_bpt_from_nll`` is the
helper that turns a model + ``(num_seqs, seq_len)`` int64 calib tensor
into a single BPT float (plus an optional per-token argmax tensor for
the ``top1_agreement`` metric).

Wiring
------
``BptMetricPlugin`` is **live-wired** by ``stage6alt.orchestrator``:
the orchestrator constructs ``BptMetricPlugin()`` in its
``PluginRegistry`` and dispatches ``walk_phases(("compute_bpt",), ...)``
on the run context, which invokes the ``compute_bpt`` hook below. The
hook reads ``model`` / ``calib_ids`` / ``config`` from the context and
publishes ``student_bpt`` / ``student_argmax`` for downstream phases
(zero-shot subset, teacher-side, report assembly).

The dedup of the NLL loop against Stage 6's ``_wikitext2_ppl`` (a sibling
NLL loop in ``stage6.plugins.wikitext_ppl``) is intentionally NOT done:
the two metrics serve different orchestrators with different attention
contracts and corpus shapes, and keeping them independent avoids
cross-stage coupling.

Circular-import contract (mirror of ``stage6alt/plugins/thermo_corpus.py``):
this module imports only from ``..context`` / ``...utils.calibration``
/ stdlib / torch — NEVER from ``stage6alt.orchestrator`` at any scope
(module-top OR function-local), so the orchestrator can freely import
this module without risk of an import cycle.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch

from ..context import PipelineContext
from ...utils.calibration import iter_batches

log = logging.getLogger(__name__)


# Module-local copy of the Stage 6 eager-attention contract. The monolith
# imports ``_STAGE6_ATTN_IMPLEMENTATION`` from ``stage6_validate`` for the
# same purpose; mirror the relocation discipline of
# ``stage6/plugins/wikitext_ppl.py`` (each module carries its own copy of
# the constant rather than chaining the import).
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"


# ---------------------------------------------------------------------------
# Bits-per-token
# ---------------------------------------------------------------------------


def _bpt_from_nll(model, calib_ids: torch.Tensor, *, device, batch_size: int,
                  collect_argmax: bool = False):
    """Mean next-token NLL in bits over a pre-tokenized calibration tensor.

    Adapted from `stage6_validate._wikitext2_ppl`'s NLL loop, but returns
    bits-per-token (mean NLL in nats / ln 2) instead of exp(mean NLL), and
    takes a ready-made `(num_seqs, seq_len)` int64 tensor instead of loading
    WikiText.

    Returns `float("inf")` if any batch is skipped — a loud failure rather
    than a partial-corpus number that would corrupt a directional comparison.

    When `collect_argmax=True`, returns `(bpt, argmax)` where `argmax` is a
    CPU int64 tensor of shape `(num_seqs, seq_len-1)` holding the model's
    predicted next-token id at each position — used by the top1_agreement
    metric. On the skip/inf path `argmax` is `None`. When `collect_argmax`
    is False (default) the bare `float` is returned, as before.
    """
    # Batch-size-invariant numerics require eager attention (same requirement
    # as Stage 6's PPL / lm-eval paths). The student is loaded eager by
    # run_pipeline._load_for_stage; the teacher must be loaded eager explicitly.
    _attn_impl = getattr(model.config, "_attn_implementation", None)
    if _attn_impl != _STAGE6_ATTN_IMPLEMENTATION:
        raise RuntimeError(
            f"stage6alt _bpt_from_nll: model.config._attn_implementation="
            f"{_attn_impl!r}, expected {_STAGE6_ATTN_IMPLEMENTATION!r} "
            "(batch-size-invariant NLL requires eager attention)."
        )
    model.train(False)  # inference mode (equivalent to model.eval())

    _dev = device
    if _dev is None:
        try:
            _dev = next(model.parameters()).device
        except StopIteration:
            pass

    nll_sum = 0.0
    tok_count = 0
    skipped = 0
    total = 0
    argmax_chunks: list[torch.Tensor] = []
    n_seqs = calib_ids.shape[0]
    log.info("Stage 6alt BPT: %d sequences x len=%d, batch_size=%d",
             n_seqs, calib_ids.shape[1], batch_size)
    with torch.no_grad():
        for i, batch in enumerate(iter_batches(calib_ids, batch_size=batch_size)):
            total += 1
            if _dev is not None:
                batch = batch.to(_dev)
            try:
                out = model(input_ids=batch, labels=batch)
                if out.loss is None:
                    log.warning("stage6alt _bpt_from_nll: None loss; skipping batch")
                    skipped += 1
                    continue
                loss_val = float(out.loss.item())
                if not math.isfinite(loss_val):
                    log.warning("stage6alt _bpt_from_nll: non-finite loss %.2e; "
                                "skipping batch", loss_val)
                    skipped += 1
                    continue
                # (batch.numel() - batch.shape[0]) == B*(seq_len-1): the count
                # of predicted tokens under the standard causal-LM label shift.
                predicted = batch.numel() - batch.shape[0]
                nll_sum += loss_val * predicted
                tok_count += predicted
                if collect_argmax:
                    # logits[:, t] predicts token t+1 → predicted-next id at
                    # positions 0..L-2. Move to CPU so 32 batches' worth of
                    # predictions don't accumulate on the GPU.
                    argmax_chunks.append(
                        out.logits[:, :-1, :].argmax(dim=-1).to("cpu")
                    )
            except Exception as exc:           # noqa: BLE001
                log.warning("stage6alt _bpt_from_nll: batch error (%s); skipping", exc)
                skipped += 1
                continue
            # round (not floor) so batch_size > 64 still rounds to every-batch
            # logging instead of dividing by zero — keeps the "every ~64
            # sequences" intent symmetric around the 64-token boundary.
            # Mirrors stage6/plugins/wikitext_ppl.py cadence discipline.
            if (i + 1) % max(1, round(64 / batch_size)) == 0:
                log.info("  BPT forward %d/%d batches", i + 1,
                         math.ceil(n_seqs / batch_size))
    if skipped > 0:
        log.error("stage6alt _bpt_from_nll: %d/%d batches skipped — returning inf "
                  "(directional comparison must not run on a partial corpus).",
                  skipped, total)
        return (float("inf"), None) if collect_argmax else float("inf")
    if tok_count == 0:
        log.error("stage6alt _bpt_from_nll: corpus produced no tokens "
                  "(empty calib_ids?) — returning inf.")
        return (float("inf"), None) if collect_argmax else float("inf")
    # BPT = mean NLL in nats / ln(2). Computed directly from the running sum —
    # never round-tripped through exp().
    bpt = nll_sum / tok_count / math.log(2)
    if collect_argmax:
        return bpt, torch.cat(argmax_chunks, dim=0)
    return bpt


class BptMetricPlugin:
    """Stage 6alt thermometer BPT-metric plugin.

    Owns the Stage 6alt BPT-measurement concern: the ``_bpt_from_nll``
    helper plus the ``compute_bpt`` phase hook that runs the student-side
    BPT pass.

    Live-wired by ``stage6alt.orchestrator``: registered in the
    ``PluginRegistry`` and invoked via ``walk_phases(("compute_bpt",), ...)``
    after the corpus is built and before the zero-shot subset phase.
    """

    name = "bpt_metric"
    paper = "Stage 6alt thermometer BPT metric — mean_NLL / log(2) (standard information-theoretic BPT; no specific paper). See module docstring."
    config_key = "stage6_validate.thermometer"
    reads: tuple[str, ...] = ("model", "calib_ids", "config")
    writes: tuple[str, ...] = ("student_bpt", "student_argmax")
    # No calibration-pass accumulator — BPT is a forward-pass-only metric
    # that consumes the corpus tensor ``ThermoCorpusPlugin`` built and
    # produces a scalar (plus optional argmax tensor) directly.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — every thermometer run must score BPT.

        ``config_key`` only names the thermometer config sub-tree
        (``bpt_batch_size`` lives there); it never gates the plugin as a
        whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def compute_bpt(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6alt thermometer student-BPT.

        Live-wired: ``stage6alt.orchestrator.run`` dispatches
        ``walk_phases(("compute_bpt",), plugins, run_ctx)`` after
        ``build_corpus`` and before ``compute_zero_shot_subset``, which
        invokes this method.

        Reads ``model`` / ``calib_ids`` / ``config`` from ``ctx``, resolves
        ``bpt_batch_size`` from ``config.stage6_validate.thermometer``,
        calls ``_bpt_from_nll`` with ``collect_argmax=True``, and writes
        the two return values to the ``student_bpt`` / ``student_argmax``
        ctx slots for the downstream report-assembly phase.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise.
        model = ctx.get("model")
        calib = ctx.get("calib_ids")
        config = ctx.get("config")
        therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
        bpt_batch = int(therm.get("bpt_batch_size", 8))

        # ``device`` is an optional ctx slot — the monolith threads it through
        # as a keyword argument (``device=None`` is a valid default; the helper
        # then falls back to ``next(model.parameters()).device``).
        device = ctx.get("device") if ctx.has("device") else None

        bpt, argmax = _bpt_from_nll(
            model, calib, device=device, batch_size=bpt_batch,
            collect_argmax=True,
        )

        ctx.set("student_bpt", bpt)
        ctx.set("student_argmax", argmax)


__all__ = ["_bpt_from_nll", "BptMetricPlugin"]
