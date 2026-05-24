"""Zero-shot lm-eval harness (S6-3 of the Stage 6 plugin-architecture refactor).

Paper / dataset
----------------
lm-evaluation-harness (Gao et al. 2024) standard zero-shot suite —
ARC-Challenge (Clark et al. 2018, arXiv:1803.05457) + HellaSwag
(Zellers et al. 2019, arXiv:1905.07830). Loglikelihood scoring via
``lm_eval.simple_evaluate(...)``.

Stage 6 implementation note: ``batch_size=auto:8`` (lm-eval
deterministic loglikelihood, numerically identical to
``batch_size=1``). This is the project's ``VALIDATED_STRATEGIES``
§Stage 6 Optimization #2.

Reference code
--------------
EleutherAI/lm-evaluation-harness — standard library; no project-pinned
SHA. Invoked via the ``lm_eval`` package.

Home of the Stage 6 zero-shot concern, extracted from the legacy
``stage6_validate.py`` monolith. The zero-shot sub-metric of the Stage 6
validation gate delegates to lm-eval's ``simple_evaluate`` for ARC-Challenge
and HellaSwag accuracy (Spec §9, Optimization #2 — ``batch_size=auto:8``,
deterministic loglikelihood scoring, numerically identical to
``batch_size=1``).

Pattern A vs Pattern B
----------------------
S6-3's zero-shot slice covers a MIXED pattern (mirror of S6-2):

* **Pattern A — relocated verbatim**: ``_ZERO_SHOT_TASKS`` (the canonical
  metric-key frozenset) and ``_lm_eval_tasks`` (the harness wrapper) below are
  character-identical copies of the monolith definitions. ``stage6_validate.py``
  re-imports them (a ``# noqa: F401`` block) so ``run()`` and external
  callers/tests (e.g. ``stage6alt_thermometer``) keep their original import
  path.
* **Pattern B — reproduced in an inert hook**: the ``run()`` student-side
  zero-shot *call site* (the ``s6["zero_shot"]["enabled"]`` gate + the
  ``results["student"].update(_lm_eval_tasks(...))`` invocation) is INLINE
  ``run()`` code in the monolith — there is nothing standalone to relocate.
  The ``eval_task`` hook below REPRODUCES that inline call faithfully; the
  monolith ``run()`` is NOT modified for it. This is an intentional, temporary
  logic duplication that resolves at S6-8 when the monolith ``run()`` is
  deleted and this hook is wired live.

Circular-import contract (mirror of ``stage6/plugins/eval_environment.py``):
this module imports only from ``..context`` / stdlib — NEVER from
``stage6_validate``, ``stage6.orchestrator`` or ``orchestrator`` at any scope
(module-top OR function-local). The monolith re-imports *this* module at load
time, so a ``from ..stage6_validate import ...`` here would deadlock the
import; nothing in this module does that.

``ZeroShotLmEvalPlugin`` is registered-but-INERT at S6-3 — no orchestrator walk
or test invokes its ``eval_task`` hook. S6-8 plugs the hook into the live
Stage 6 plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
from typing import Any

from ..context import PipelineContext

log = logging.getLogger(__name__)


# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Constant — never override at call sites. This is
# a module-LOCAL copy of the monolith's ``_STAGE6_ATTN_IMPLEMENTATION``: the
# monolith keeps its own definition and is NOT imported here (circular-import
# contract). Both copies must stay in sync until S6-8 collapses the monolith.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"


_ZERO_SHOT_TASKS: frozenset[str] = frozenset({"arc_challenge_acc", "hellaswag_acc"})


# ---------------------------------------------------------------------------
# Zero-shot (ARC-C + HellaSwag) via lm-eval (Optimization #2: batch_size=auto:8)
# ---------------------------------------------------------------------------


def _lm_eval_tasks(model, tokenizer, tasks: list[str], *, collect=None,
                   batch_size="auto:8", limit=None) -> dict:
    """Delegate to lm-eval's simple_evaluate with configurable batch_size.

    lm-eval's 0-shot loglikelihood scoring is deterministic and batch-size-
    independent. Numerically identical to batch_size=1.

    `limit` (int or float in (0,1], default None) caps the number of docs
    per task — used by the stage6alt thermometer to subsample ARC-Easy /
    HellaSwag for a cheap directional signal. None = evaluate the full task,
    which is the behavior every full-Stage-6 caller relies on.
    """
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except Exception as err:           # noqa: BLE001
        log.warning("lm-eval not available (%s); skipping zero-shot.", err)
        return {}

    # Spec §9 #2: lm-eval batch-size invariance requires eager attention.
    _attn_impl = getattr(model.config, "_attn_implementation", None)
    if _attn_impl != _STAGE6_ATTN_IMPLEMENTATION:
        raise RuntimeError(
            f"Stage 6 _lm_eval_tasks: model.config._attn_implementation="
            f"{_attn_impl!r}, expected {_STAGE6_ATTN_IMPLEMENTATION!r} per "
            "spec §9 #2 (batch-size-independent loglikelihood requires eager attn)."
        )
    try:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        out = simple_evaluate(
            model=lm, tasks=list(tasks), num_fewshot=0,
            log_samples=(collect is not None),
            limit=limit,
        )
        results = out.get("results", {})
        flat: dict = {}
        for task, metrics in results.items():
            # ARC-C canonical metric is acc_norm,none (normalized); prefer it first.
            # Use key-existence check (not truthiness) so acc=0.0 is not skipped.
            for _k in ("acc_norm,none", "acc,none", "acc"):
                if _k in metrics:
                    acc = metrics[_k]
                    break
            else:
                acc = None
            if acc is not None:
                flat[f"{task}_acc"] = float(acc)
        if collect is not None and "samples" in out:
            for task_samples in out["samples"].values():
                seen: set[str] = set()
                for s in task_samples:
                    try:
                        args = s.get("arguments", ())
                        ctx = args[0] if args else None
                        if ctx and isinstance(ctx, str) and ctx not in seen:
                            seen.add(ctx)
                            collect.append(ctx)
                    except (KeyError, IndexError, TypeError):
                        pass
        return flat
    except Exception as err:           # noqa: BLE001
        log.warning("lm-eval evaluation failed: %s", err, exc_info=True)
        return {}


class ZeroShotLmEvalPlugin:
    """Stage 6 zero-shot lm-eval plugin (S6-3 — registered-but-INERT).

    Owns the Stage 6 zero-shot sub-metric: the relocated ``_ZERO_SHOT_TASKS``
    constant and ``_lm_eval_tasks`` helper (Pattern A) plus an inert
    ``eval_task`` hook (Pattern B) that reproduces the monolith's inline
    student-side zero-shot call site.

    S6-3 wires this class into the plugin registry as metadata only — no
    orchestrator walk or test invokes ``eval_task``. S6-8 plugs the hook into
    the live Stage 6 plugin sequencer and deletes the monolith ``run()``.
    """

    name = "zero_shot_lm_eval"
    paper = "lm-eval zero-shot (Gao et al. 2024) ARC-Challenge + HellaSwag — EleutherAI/lm-evaluation-harness. See module docstring."
    config_key = "stage6_validate.zero_shot.enabled"
    reads: tuple[str, ...] = ("model", "tokenizer", "config")
    writes: tuple[str, ...] = ("eval_results",)
    # eval_results is a shared collector the orchestrator pre-creates per side
    # and every eval plugin appends to; it is NOT a calibration-pass accumulator,
    # so it belongs in `writes`, not `provides`. (S6-8 wires the collector.)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Gate on ``stage6_validate.zero_shot.enabled`` (default False).

        Mirrors the monolith ``run()``'s ``if s6["zero_shot"]["enabled"]``
        guard. Uses ``.get()`` chains so a missing ``stage6_validate`` or
        ``zero_shot`` subdict resolves to disabled rather than raising.
        """
        return bool(
            (config.get("stage6_validate", {}) or {})
            .get("zero_shot", {})
            .get("enabled", False)
        )

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def eval_task(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6 zero-shot eval (S6-8 wiring surface).

        INERT at S6-3: no orchestrator walk or test invokes this hook. S6-8
        replaces the Stage 6 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        zero-shot block. The body below reproduces that inline call site
        faithfully — it is dead code at S6-3 but S6-8 relies on it once the
        monolith ``run()`` is deleted.

        Reproduces the monolith ``run()``'s student-side call:

            results["student"].update(
                _lm_eval_tasks(model, tokenizer, s6["zero_shot"]["tasks"],
                               collect=eval_text_concat,
                               batch_size=lm_eval_batch_size)
            )

        The harness's flat ``{task}_acc`` dict is merged into the pre-existing
        ``eval_results`` ctx slot (the analogue of the monolith's
        ``results["student"]`` dict) via ``dict.update``. This hook does NOT
        ``ctx.set`` ``eval_results`` — it mutates the dict another plugin/the
        orchestrator already created.

        The monolith parses/validates ``lm_eval_batch_size`` from
        ``s6.get("lm_eval_batch_size", "auto:8")`` (accepting a positive int,
        an int-string, or the ``auto[:N]`` pattern) and passes the run-scoped
        ``eval_text_concat`` side-channel as ``collect``; the hook reproduces
        that batch-size resolution and threads ``collect`` from an optional
        ctx slot so the call shape matches even though that side-channel is not
        S6-3's concern.
        """
        import re

        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        s6 = config["stage6_validate"]

        # Reproduces the monolith's lm_eval_batch_size parse/validation block.
        _raw_lebs = s6.get("lm_eval_batch_size", "auto:8")
        if isinstance(_raw_lebs, int):
            if _raw_lebs <= 0:
                raise ValueError(
                    f"stage6_validate.lm_eval_batch_size must be > 0; got {_raw_lebs}"
                )
            lm_eval_batch_size = _raw_lebs
        elif isinstance(_raw_lebs, str):
            if not (re.fullmatch(r"\d+", _raw_lebs) or re.fullmatch(r"auto(:\d+)?", _raw_lebs)):
                raise ValueError(
                    f"stage6_validate.lm_eval_batch_size must be a positive int or "
                    f"match 'auto' / 'auto:N'; got {_raw_lebs!r}"
                )
            lm_eval_batch_size = int(_raw_lebs) if _raw_lebs.isdigit() else _raw_lebs
        else:
            raise TypeError(
                f"stage6_validate.lm_eval_batch_size must be int or str; "
                f"got {type(_raw_lebs).__name__}"
            )

        # The run-scoped `eval_text_concat` is an optional context side-channel
        # in the plugin world (the monolith threads it through run()); default
        # to None when a wiring stage has not provided it.
        collect = ctx.get("eval_text_concat") if ctx.has("eval_text_concat") else None

        log.info("Stage 6: zero-shot harness, batch_size=%s", lm_eval_batch_size)
        eval_results = ctx.get("eval_results")
        eval_results.update(
            _lm_eval_tasks(model, tokenizer, s6["zero_shot"]["tasks"],
                           collect=collect, batch_size=lm_eval_batch_size)
        )


__all__ = ["_ZERO_SHOT_TASKS", "_lm_eval_tasks", "ZeroShotLmEvalPlugin"]
