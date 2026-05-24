"""WikiText-2 perplexity eval (S6-3 of the Stage 6 plugin-architecture refactor).

Paper / dataset
----------------
WikiText-2 perplexity protocol — Merity et al. 2017 "Pointer Sentinel
Mixture Models" (ICLR 2017, arXiv:1609.07843). PPL = ``exp(mean per-token NLL)``
over the 2048-token-chunked corpus with row-join ``"\n\n"`` (project
convention matching HF / lm-eval recipe and the imatrix calibration
corpus build).

Declared deviations from Merity et al. 2017 Table 4
---------------------------------------------------
1. **Non-overlapping chunking** (sliding-window in paper). Each
   2048-token chunk's first ~k tokens are scored with little-to-no
   left context, inflating PPL versus the paper's sliding-window
   full-context recipe. This is the canonical HF / lm-eval recipe
   convention. **Reported numbers are NOT directly comparable to
   Merity et al. 2017 Table 4.**
2. **Drop-last-partial** (paper reports over ALL tokens). The trailing
   ``len(all_ids) % 2048`` tokens are discarded — the reported PPL is
   computed over ``n_full × 2048`` tokens. On the standard
   ``wikitext-2-raw-v1`` test split this covers ~99% of the corpus,
   but for short / non-standard splits the dropped fraction can shift
   the reported number non-trivially.

Stage 6 implementation note: configurable ``batch_size``;
**numerically identical to ``batch_size=1``** (no per-batch noise).
This is the project's ``VALIDATED_STRATEGIES`` §Stage 6 Optimization
#1.

Home of the Stage 6 WikiText-2 perplexity concern, extracted from the legacy
``stage6_validate.py`` monolith. WikiText-2 PPL is the first sub-metric of the
Stage 6 validation gate: standard next-token NLL → ``exp(mean_NLL)`` over the
2048-token chunked corpus (Spec §9, Optimization #1 — configurable batch_size,
numerically identical to ``batch_size=1``).

Pattern A vs Pattern B
----------------------
S6-3's wikitext slice covers a MIXED pattern (mirror of S6-2):

* **Pattern A — relocated verbatim**: ``_wikitext2_ppl`` below is a
  character-identical copy of the monolith body. ``stage6_validate.py``
  re-imports it (a ``# noqa: F401`` block) so ``run()`` and external
  callers/tests (e.g. ``stage6alt_thermometer``) keep their original import
  path.
* **Pattern B — reproduced in an inert hook**: the ``run()`` student-side
  WikiText-PPL *call site* (the ``s6["wikitext2"]["enabled"]`` gate + the
  ``_wikitext2_ppl(...)`` invocation that lands the result in
  ``results["student"]["wikitext2_ppl"]``) is INLINE ``run()`` code in the
  monolith — there is nothing standalone to relocate. The ``eval_task`` hook
  below REPRODUCES that inline call faithfully; the monolith ``run()`` is NOT
  modified for it. This is an intentional, temporary logic duplication that
  resolves at S6-8 when the monolith ``run()`` is deleted and this hook is
  wired live.

Circular-import contract (mirror of ``stage6/plugins/eval_environment.py``):
this module imports only from ``..context`` / ``...utils`` / stdlib / torch —
NEVER from ``stage6_validate``, ``stage6.orchestrator`` or ``orchestrator`` at
any scope (module-top OR function-local). The monolith re-imports *this* module
at load time, so a ``from ..stage6_validate import ...`` here would deadlock the
import; nothing in this module does that.

``WikitextPplPlugin`` is registered-but-INERT at S6-3 — no orchestrator walk or
test invokes its ``eval_task`` hook. S6-8 plugs the hook into the live Stage 6
plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch

from ..context import PipelineContext
from ...utils.calibration import iter_batches

log = logging.getLogger(__name__)


# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Constant — never override at call sites. This is
# a module-LOCAL copy of the monolith's ``_STAGE6_ATTN_IMPLEMENTATION``: the
# monolith keeps its own definition and is NOT imported here (circular-import
# contract). Both copies must stay in sync until S6-8 collapses the monolith.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"

# N3: single source of truth for the WikiText PPL forward batch size default.
# Referenced by both ``_wikitext2_ppl(... batch_size=...)`` and the eval_task
# hook's ``s6.get("ppl_batch_size", _DEFAULT_PPL_BATCH_SIZE)`` so the two sites
# cannot drift out of sync via manual edit. Optimization #1 guarantees the
# computed PPL is numerically identical across batch sizes, so the default's
# value here affects only throughput / log cadence, not the reported number.
_DEFAULT_PPL_BATCH_SIZE: int = 8


# ---------------------------------------------------------------------------
# WikiText-2 perplexity (Optimization #1: configurable batch_size)
# ---------------------------------------------------------------------------


def _wikitext2_ppl(model, tokenizer, cfg: dict, *, device=None, collect=None,
                   batch_size: int = _DEFAULT_PPL_BATCH_SIZE,
                   dataset_revisions: dict[str, str | None] | None = None) -> float:
    """Standard next-token NLL → exp(mean_NLL), seq_len=2048.

    Batching doesn't change NLL computation — each sequence is scored
    independently; out.loss is the mean over tokens in each batch element,
    and we scale by (batch.numel() - batch.shape[0]) to recover the sum.
    Numerically identical to batch_size=1.

    F-C-H-3: pass revision= to load_dataset when a pinned SHA is configured.

    F-iter4-CRIT-1: assert that the model is running under eager attention.
    Spec §9 lines 821, 838 require eager attn for the Stage 6 gate run for
    BOTH teacher and student to remove cross-batch sdpa-kernel variance and
    preserve the bs=1 ↔ bs>1 numerical-equivalence claim of Optimization #1.
    F-iter4-L-4: assert sequence_length == 2048 (Spec §9 line 769).
    """
    # F-iter4-L-4: PPL chunk length is fixed by spec.
    assert int(cfg.get("sequence_length", 0)) == 2048, (
        f"_wikitext2_ppl: cfg['sequence_length'] must be 2048 per Spec §9 "
        f"line 769; got {cfg.get('sequence_length')!r}"
    )
    # F-iter4-CRIT-1: verify eager attention pin took effect.
    try:
        model_attn = getattr(model.config, "_attn_implementation", None)
    except Exception:  # noqa: BLE001
        model_attn = None
    assert model_attn == _STAGE6_ATTN_IMPLEMENTATION, (
        f"_wikitext2_ppl: model.config._attn_implementation={model_attn!r} "
        f"but Spec §9 lines 821, 838 require {_STAGE6_ATTN_IMPLEMENTATION!r} "
        f"for the Stage 6 gate run. The student must be loaded with "
        f"attn_implementation='eager' (see run_pipeline._load_for_stage)."
    )
    from datasets import load_dataset

    revision = (dataset_revisions or {}).get("wikitext_ppl")
    try:
        ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"], revision=revision)
    except Exception as exc:
        log.warning("_wikitext2_ppl: load_dataset failed (%s); returning inf PPL", exc)
        return float("inf")
    # F-CR2-H-1/H-2 (Spec §9 / F-S-C-1): tokenize the entire concatenated corpus
    # in a single call with add_special_tokens=True. This:
    #   - applies BOS exactly once (closes F-CR2-H-2),
    #   - inserts no inter-row separator tokens beyond the natural newline that
    #     wikitext-2-raw-v1 already uses (closes F-CR2-H-1),
    #   - matches the canonical HF / lm-eval WikiText-2 PPL recipe.
    rows: list[str] = []
    for row in ds:
        text = row.get("text", "")
        # Spec §9 / F-S-C-1 says BOS is applied once on the concatenated text;
        # the row-to-row joiner is "\n\n" matching the canonical HF / lm-eval
        # PPL recipe. Empty rows ARE preserved here (canonical recipe keeps
        # them, producing a "\n\n" + "" + "\n\n" sequence that turns into the
        # expected paragraph-spacing tokens). Filtering empties would change
        # the token stream and chunk boundaries vs. the cited recipe.
        if collect is not None and text.strip():
            collect.append(text)
        rows.append(text)
    concatenated = "\n\n".join(rows)
    all_ids: list[int] = tokenizer(
        concatenated, add_special_tokens=True, return_tensors=None,
    )["input_ids"]

    seq_len = cfg["sequence_length"]
    n_full = len(all_ids) // seq_len
    if n_full == 0:
        log.warning("WikiText-2 has no full-length sequences; returning inf.")
        return float("inf")
    chunks = torch.tensor(all_ids[: n_full * seq_len], dtype=torch.long).view(n_full, seq_len)

    nll_sum = 0.0
    tok_count = 0
    log.info("Stage 6 PPL: %d sequences × len=%d, batch_size=%d", n_full, seq_len, batch_size)
    # Infer device from model when not explicitly set (e.g. device_map="auto")
    _ppl_dev = device
    if _ppl_dev is None:
        try:
            _ppl_dev = next(model.parameters()).device
        except StopIteration:
            pass

    skipped_batches = 0
    total_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(iter_batches(chunks, batch_size=batch_size)):
            total_batches += 1
            if _ppl_dev is not None:
                batch = batch.to(_ppl_dev)
            # out.loss is the mean NLL over all B*(seq_len-1) predicted tokens in the batch.
            try:
                out = model(input_ids=batch, labels=batch)
                if out.loss is None:
                    log.warning("_wikitext2_ppl: model returned None loss for batch; skipping")
                    skipped_batches += 1
                    continue
                loss_val = float(out.loss.item())
                if not math.isfinite(loss_val):
                    log.warning("_wikitext2_ppl: non-finite loss %.2e for batch; skipping", loss_val)
                    skipped_batches += 1
                    continue
                # L-3: Assumes the model uses the standard causal LM convention of
                # shifting labels by one position, computing loss over (seq_len - 1)
                # tokens per row.  The factor (batch.numel() - batch.shape[0]) equals
                # B * (seq_len - 1), recovering the total NLL sum from the mean loss.
                # Incorrect for models with non-standard label conventions (prefix
                # labels, pad-masked losses, etc.).
                nll = loss_val * (batch.numel() - batch.shape[0])
                nll_sum += nll
                tok_count += batch.numel() - batch.shape[0]
            except Exception as exc:
                log.warning("_wikitext2_ppl: error processing batch (%s); skipping", exc)
                skipped_batches += 1
                continue
            # N2: round (not floor) so batch_size > 64 still rounds to every-batch
            # logging instead of dividing by zero — keeps the "every ~64 sequences"
            # intent symmetric around the 64-token boundary.
            if (i + 1) % max(1, round(64 / batch_size)) == 0:  # log every ~64 sequences regardless of batch size
                log.info("  PPL forward %d/%d batches (%d/%d seqs)",
                         i + 1, math.ceil(n_full / batch_size), min((i + 1) * batch_size, n_full), n_full)
    if tok_count == 0:
        # M1: All batches were skipped — PPL is entirely undefined, not just degraded.
        log.error(
            "_wikitext2_ppl: All batches skipped (%d/%d); PPL is undefined — returning inf",
            skipped_batches, total_batches,
        )
        return float("inf")
    if skipped_batches > 0:
        # Spec §9 PPL formula is over ALL retained chunks (drop-last-partial only).
        # A runtime skip would silently change the reported PPL's domain, so a
        # gating decision could pass on a partially-broken student. Force inf
        # to fail the gate rather than report a sub-corpus PPL.
        log.error(
            "_wikitext2_ppl: %d/%d batches were skipped (non-finite loss or runtime error); "
            "spec §9 mandates PPL over the full retained-chunk corpus — returning inf to "
            "fail the gate rather than report a sub-corpus PPL.",
            skipped_batches, total_batches,
        )
        return float("inf")
    try:
        return math.exp(nll_sum / tok_count)
    except OverflowError:
        return float("inf")


class WikitextPplPlugin:
    """Stage 6 WikiText-2 perplexity plugin (S6-3 — registered-but-INERT).

    Owns the Stage 6 WikiText-2 PPL sub-metric: the relocated ``_wikitext2_ppl``
    helper (Pattern A) plus an inert ``eval_task`` hook (Pattern B) that
    reproduces the monolith's inline student-side WikiText-PPL call site.

    S6-3 wires this class into the plugin registry as metadata only — no
    orchestrator walk or test invokes ``eval_task``. S6-8 plugs the hook into
    the live Stage 6 plugin sequencer and deletes the monolith ``run()``.
    """

    name = "wikitext_ppl"
    paper = (
        "WikiText-2 PPL — Merity et al. 2017 (ICLR) arXiv:1609.07843; "
        "project batched bs-invariant. NOTE: this implementation uses the "
        "HF / lm-eval non-overlapping 2048-token chunking convention with "
        "drop-last-partial, so reported PPL is NOT directly comparable to "
        "Merity et al. 2017 Table 4 (sliding-window full-context). See "
        "module docstring for the full deviation list."
    )
    config_key = "stage6_validate.wikitext2.enabled"
    reads: tuple[str, ...] = ("model", "tokenizer", "config", "dataset_revisions")
    writes: tuple[str, ...] = ("eval_results",)
    # eval_results is a shared collector the orchestrator pre-creates per side
    # and every eval plugin appends to; it is NOT a calibration-pass accumulator,
    # so it belongs in `writes`, not `provides`. (S6-8 wires the collector.)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Gate on ``stage6_validate.wikitext2.enabled`` (default False).

        Mirrors the monolith ``run()``'s ``if s6["wikitext2"]["enabled"]``
        guard. Uses ``.get()`` chains so a missing ``stage6_validate`` or
        ``wikitext2`` subdict resolves to disabled rather than raising.
        """
        return bool(
            (config.get("stage6_validate", {}) or {})
            .get("wikitext2", {})
            .get("enabled", False)
        )

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def eval_task(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6 WikiText-2 PPL eval (S6-8 wiring surface).

        INERT at S6-3: no orchestrator walk or test invokes this hook. S6-8
        replaces the Stage 6 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        WikiText-PPL block. The body below reproduces that inline call site
        faithfully — it is dead code at S6-3 but S6-8 relies on it once the
        monolith ``run()`` is deleted.

        Reproduces the monolith ``run()``'s student-side call:

            results["student"]["wikitext2_ppl"] = _wikitext2_ppl(
                model, tokenizer, s6["wikitext2"], device=device,
                collect=eval_text_concat, batch_size=ppl_batch_size,
                dataset_revisions=dataset_revisions,
            )

        The result lands in the pre-existing ``eval_results`` ctx slot (the
        analogue of the monolith's ``results["student"]`` dict) under the
        ``wikitext2_ppl`` key. This hook does NOT ``ctx.set`` ``eval_results``
        — it mutates the dict another plugin/the orchestrator already created.

        The monolith parses ``ppl_batch_size`` from ``s6.get("ppl_batch_size",
        8)`` and passes the run-scoped ``device`` / ``eval_text_concat``
        side-channel; the hook reproduces the ``ppl_batch_size`` default and
        passes ``device`` / ``collect`` from optional ctx slots so the call
        shape matches even though those side-channels are not S6-3's concern.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        dataset_revisions = ctx.get("dataset_revisions")
        s6 = config["stage6_validate"]

        # Reproduces the monolith's `ppl_batch_size = int(s6.get("ppl_batch_size", 8))`.
        # N3: default sourced from module-level _DEFAULT_PPL_BATCH_SIZE so this
        # site and `_wikitext2_ppl`'s own kwarg default cannot drift apart.
        ppl_batch_size = int(s6.get("ppl_batch_size", _DEFAULT_PPL_BATCH_SIZE))
        # The run-scoped `device` / `eval_text_concat` are optional context
        # side-channels in the plugin world (the monolith threads them through
        # run()); default to None when a wiring stage has not provided them.
        device = ctx.get("device") if ctx.has("device") else None
        collect = ctx.get("eval_text_concat") if ctx.has("eval_text_concat") else None

        log.info("Stage 6: WikiText-2 PPL (student), batch_size=%d", int(ppl_batch_size))
        eval_results = ctx.get("eval_results")
        eval_results["wikitext2_ppl"] = _wikitext2_ppl(
            model, tokenizer, s6["wikitext2"], device=device, collect=collect,
            batch_size=ppl_batch_size, dataset_revisions=dataset_revisions,
        )


__all__ = ["_wikitext2_ppl", "WikitextPplPlugin"]
