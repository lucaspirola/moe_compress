"""Thermometer ARC-Easy + HellaSwag zero-shot subset (Stage 6alt plugin).

Paper / dataset
----------------
Stage 6alt thermometer ARC-Easy + HellaSwag subset — same lm-eval-harness
(Gao et al. 2024) wrappers as :mod:`stage6.plugins.zero_shot_lm_eval`,
but with **small-subsample limits** (``ARC-Easy limit=N``,
``HellaSwag limit=M``) so the thermometer runs in seconds rather than
minutes.

Datasets: ARC-Easy + ARC-Challenge are Clark et al. 2018
(arXiv:1803.05457); HellaSwag is Zellers et al. 2019
(arXiv:1905.07830).

Stage 6alt implementation note: ``batch_size=auto:8`` (lm-eval
deterministic loglikelihood, numerically identical to ``batch_size=1``)
— this is the project's ``VALIDATED_STRATEGIES`` §Stage 6 Optimization
#2, inherited via the sibling :mod:`stage6.plugins.zero_shot_lm_eval`
(which also pins the lm-eval ``>=0.4.5,<0.5`` version contract that
this plugin transitively relies on through ``_lm_eval_tasks``).

Reference code
--------------
EleutherAI/lm-evaluation-harness — standard library, no
project-pinned SHA.

Home of the Stage 6alt thermometer zero-shot-subset concern, extracted
from the legacy ``stage6alt_thermometer.py`` monolith. The thermometer
runs a small subsample of ARC-Easy + HellaSwag via lm-eval to produce
a cheap secondary signal alongside BPT; ``_lm_eval_subset`` wraps two
``_lm_eval_tasks`` calls (one per task, the two limits differ) and
returns the two per-task acc_norm scalars and their sum.

Deviation D-arc-acc-norm (inherited)
------------------------------------
``_lm_eval_tasks`` is imported from the sibling
:mod:`stage6.plugins.zero_shot_lm_eval` and its metric-key preference
list — ``("acc_norm,none", "acc,none")`` — picks ``acc_norm,none`` first
whenever it is present, which it always is for ARC-Easy in lm-eval
``>=0.4.x``. That means the ``arc_easy_acc_norm`` scalar this plugin
writes (and the downstream ``student_arc_easy_acc_norm`` ctx slot) is
length-normalised accuracy, NOT the Clark 2018 (arXiv:1803.05457) raw
``acc`` reported in the original ARC paper. See the sibling's "Deviation
D-arc-acc-norm" docstring for the full rationale (Open-LLM-Leaderboard
convention, length-bias correction, dashboard continuity). HellaSwag
(Zellers et al. 2019) already uses ``acc_norm`` in its original paper so
no deviation applies there.

Pattern A vs Pattern B
----------------------
* **Pattern A — relocated**: ``_lm_eval_subset`` below owns the helper;
  ``stage6alt_thermometer.py`` re-exports it (the ``# noqa: F401`` block
  at ``stage6alt_thermometer.py:71-74``) purely for monkeypatch-by-
  attribute back-compat (the S6A-0 golden snapshot still patches
  ``stage6alt_thermometer._lm_eval_subset`` via ``monkeypatch.setattr``).
  The underlying ``_lm_eval_tasks`` harness wrapper is NOT relocated
  here: it already lives in ``stage6.plugins.zero_shot_lm_eval`` and is
  imported from that home — re-relocating it would create two divergent
  copies of the same harness wrapper.
* **Pattern B — live orchestrator entry point**: the
  ``compute_zero_shot_subset`` hook below is the live student-side
  ARC-Easy + HellaSwag call site. ``stage6alt.orchestrator.run``
  registers ``ZeroShotSubsetPlugin()`` and dispatches
  ``walk_phases(("compute_zero_shot_subset",), ...)`` against it; the
  monolith ``stage6alt_thermometer.run`` is now a thin shim that
  delegates to the orchestrator.

Circular-import contract (mirror of ``stage6alt/plugins/thermo_corpus.py``):
this module imports only from ``..context`` / ``...stage6.plugins.zero_shot_lm_eval``
/ stdlib — NEVER from ``stage6alt_thermometer`` or
``stage6alt.orchestrator`` at any scope (module-top OR function-local).
The monolith re-imports *this* module's symbols at load time, so a
``from ..stage6alt_thermometer import ...`` here would deadlock the
import; nothing in this module does that.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..context import PipelineContext
from ...stage6.plugins.zero_shot_lm_eval import _lm_eval_tasks

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# lm-eval subset
# ---------------------------------------------------------------------------


def _lm_eval_subset(model, tokenizer, *, arc_limit: int, hellaswag_limit: int,
                    batch_size) -> dict:
    """ARC-Easy + HellaSwag zero-shot on a subsample. Two calls (limits differ).

    Returns {arc_easy_acc_norm, hellaswag_acc_norm, acc_norm_sum}. Any metric
    that lm-eval could not produce (e.g. lm-eval not installed) is recorded as
    None and acc_norm_sum is None — BPT alone still carries the signal.
    """
    arc = _lm_eval_tasks(model, tokenizer, ["arc_easy"],
                         batch_size=batch_size, limit=arc_limit)
    hsw = _lm_eval_tasks(model, tokenizer, ["hellaswag"],
                         batch_size=batch_size, limit=hellaswag_limit)
    arc_acc = arc.get("arc_easy_acc")
    hsw_acc = hsw.get("hellaswag_acc")
    acc_sum = (arc_acc + hsw_acc) if (arc_acc is not None and hsw_acc is not None) else None
    return {
        "arc_easy_acc_norm": arc_acc,
        "hellaswag_acc_norm": hsw_acc,
        "acc_norm_sum": acc_sum,
    }


class ZeroShotSubsetPlugin:
    """Stage 6alt thermometer zero-shot-subset plugin (live).

    Owns the Stage 6alt ARC-Easy + HellaSwag zero-shot concern: the
    ``_lm_eval_subset`` helper (Pattern A) plus the live
    ``compute_zero_shot_subset`` hook (Pattern B) that the Stage 6alt
    orchestrator dispatches against the student. The underlying
    ``_lm_eval_tasks`` harness wrapper stays in its sibling home
    (``stage6.plugins.zero_shot_lm_eval``) and is imported there.
    """

    name = "zero_shot_subset"
    paper = (
        "Stage 6alt thermometer zero-shot subset — ARC-Easy + HellaSwag "
        "(Clark 2018 / Zellers 2019) via lm-eval (Gao et al. 2024); "
        "small-N limits for sweep speed. See module docstring. "
        "Deviation D-arc-acc-norm applies (ARC-Easy)."
    )
    config_key = "stage6_validate.thermometer"
    reads: tuple[str, ...] = ("model", "tokenizer", "config")
    writes: tuple[str, ...] = (
        "student_arc_easy_acc_norm",
        "student_hellaswag_acc_norm",
        "student_acc_norm_sum",
    )
    # No calibration-pass accumulator — the lm-eval harness handles its own
    # batching internally; this plugin just calls the wrapper.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — every thermometer run scores the ARC-Easy +
        HellaSwag subset. ``config_key`` only names the thermometer
        config sub-tree (``arc_easy_limit`` / ``hellaswag_limit`` /
        ``lm_eval_batch_size`` live there); it never gates the plugin
        as a whole. If lm-eval is not installed the helper records
        ``None`` for each metric — BPT alone still carries the signal.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def compute_zero_shot_subset(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6alt thermometer zero-shot subset (live).

        The Stage 6alt orchestrator (``stage6alt.orchestrator.run``)
        registers this plugin and dispatches
        ``walk_phases(("compute_zero_shot_subset",), ...)`` against it as
        the student-side ARC-Easy + HellaSwag call site. The monolith
        ``stage6alt_thermometer.run`` is a thin shim that delegates to the
        orchestrator.

        The hook reproduces the historical inline call:

            arc_limit = int(therm.get("arc_easy_limit", 100))
            hsw_limit = int(therm.get("hellaswag_limit", 200))
            lm_batch = therm.get("lm_eval_batch_size", "auto:8")
            student_lm = _lm_eval_subset(model, tokenizer,
                                         arc_limit=arc_limit,
                                         hellaswag_limit=hsw_limit,
                                         batch_size=lm_batch)

        The three result-dict entries are written to
        ``student_arc_easy_acc_norm`` / ``student_hellaswag_acc_norm`` /
        ``student_acc_norm_sum`` ctx slots.

        ``lm_eval_batch_size`` is validated inline (positive int or the
        ``auto`` / ``auto:N`` pattern), mirroring the sibling
        :mod:`stage6.plugins.zero_shot_lm_eval` validation block so both
        zero-shot call sites fail loud on the same malformed inputs.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise.
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
        arc_limit = int(therm.get("arc_easy_limit", 100))
        hellaswag_limit = int(therm.get("hellaswag_limit", 200))

        # Mirror the sibling stage6.plugins.zero_shot_lm_eval batch-size
        # parse/validation block so both zero-shot call sites reject the
        # same malformed inputs (positive int OR int-string OR auto[:N]).
        _raw_lebs = therm.get("lm_eval_batch_size", "auto:8")
        if isinstance(_raw_lebs, int):
            if _raw_lebs <= 0:
                raise ValueError(
                    "stage6_validate.thermometer.lm_eval_batch_size must be > 0; "
                    f"got {_raw_lebs}"
                )
            batch_size = _raw_lebs
        elif isinstance(_raw_lebs, str):
            if not (re.fullmatch(r"\d+", _raw_lebs) or re.fullmatch(r"auto(:\d+)?", _raw_lebs)):
                raise ValueError(
                    "stage6_validate.thermometer.lm_eval_batch_size must be a "
                    "positive int or match 'auto' / 'auto:N'; "
                    f"got {_raw_lebs!r}"
                )
            batch_size = int(_raw_lebs) if _raw_lebs.isdigit() else _raw_lebs
        else:
            raise TypeError(
                "stage6_validate.thermometer.lm_eval_batch_size must be int or "
                f"str; got {type(_raw_lebs).__name__}"
            )

        result = _lm_eval_subset(
            model, tokenizer,
            arc_limit=arc_limit, hellaswag_limit=hellaswag_limit,
            batch_size=batch_size,
        )

        ctx.set("student_arc_easy_acc_norm", result["arc_easy_acc_norm"])
        ctx.set("student_hellaswag_acc_norm", result["hellaswag_acc_norm"])
        ctx.set("student_acc_norm_sum", result["acc_norm_sum"])


__all__ = ["_lm_eval_subset", "ZeroShotSubsetPlugin"]
