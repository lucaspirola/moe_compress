"""Thermometer ARC-Easy + HellaSwag zero-shot subset (S6A-3 of the Stage 6alt plugin-architecture refactor).

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

Reference code
--------------
EleutherAI/lm-evaluation-harness — standard library, no
project-pinned SHA.

Home of the Stage 6alt thermometer zero-shot-subset concern, extracted
from the legacy ``stage6alt_thermometer.py`` monolith. The thermometer
runs a small subsample of ARC-Easy + HellaSwag via lm-eval to produce
a cheap secondary signal alongside BPT; ``_lm_eval_subset`` wraps two
``_lm_eval_tasks`` calls (one per task, the two limits differ) and
returns the three result keys + their sum.

Pattern A vs Pattern B
----------------------
S6A-3's zero-shot-subset slice covers a MIXED pattern:

* **Pattern A — relocated verbatim**: ``_lm_eval_subset`` below is a
  character-identical copy of the monolith body. ``stage6alt_thermometer.py``
  re-imports it (the ``# noqa: F401`` block) so ``run()`` and any external
  caller / test that monkey-patches ``stage6alt_thermometer._lm_eval_subset``
  keeps working unchanged — the re-import puts the SAME function object
  on the monolith namespace. The underlying ``_lm_eval_tasks`` helper is
  NOT relocated here: it already lives in ``stage6.plugins.zero_shot_lm_eval``
  (per S6-3) and is imported from that home — re-relocating it would create
  two divergent copies of the same harness wrapper.
* **Pattern B — reproduced in an inert hook**: the monolith ``run()``'s
  student-side ``_lm_eval_subset`` call site (writing the three
  ``student_arc_easy_acc_norm`` / ``student_hellaswag_acc_norm`` /
  ``student_acc_norm_sum`` slots) is reproduced in the inert
  ``compute_zero_shot_subset`` hook below. The monolith ``run()`` is NOT
  modified for it. This is an intentional, temporary logic duplication
  that resolves at S6A-6 when the orchestrator flip wires this hook live
  and the monolith ``run()`` becomes a thin shim.

Circular-import contract (mirror of ``stage6alt/plugins/thermo_corpus.py``):
this module imports only from ``..context`` / ``...stage6.plugins.zero_shot_lm_eval``
/ stdlib — NEVER from ``stage6alt_thermometer`` or
``stage6alt.orchestrator`` at any scope (module-top OR function-local).
The monolith re-imports *this* module's symbols at load time, so a
``from ..stage6alt_thermometer import ...`` here would deadlock the
import; nothing in this module does that.

``ZeroShotSubsetPlugin`` is registered-but-INERT at S6A-3 — no orchestrator
walk or test invokes its ``compute_zero_shot_subset`` hook. S6A-6 plugs
the hook into the live Stage 6alt plugin sequencer.
"""
from __future__ import annotations

import logging
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
    """Stage 6alt thermometer zero-shot-subset plugin (S6A-3 — registered-but-INERT).

    Owns the Stage 6alt ARC-Easy + HellaSwag zero-shot concern: the relocated
    ``_lm_eval_subset`` helper (Pattern A) plus an inert
    ``compute_zero_shot_subset`` hook (Pattern B) that reproduces the
    monolith's student-side call site. The underlying ``_lm_eval_tasks``
    harness wrapper stays in its S6-3 home
    (``stage6.plugins.zero_shot_lm_eval``) and is imported there.

    S6A-3 wires this class into the plugin registry as metadata only — no
    orchestrator walk or test invokes ``compute_zero_shot_subset``. S6A-6
    plugs the hook into the live Stage 6alt plugin sequencer.
    """

    name = "zero_shot_subset"
    paper = "Stage 6alt thermometer zero-shot subset — ARC-Easy + HellaSwag (Clark 2018 / Zellers 2019) via lm-eval (Gao et al. 2024); small-N limits for sweep speed. See module docstring."
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
        """Phase hook — Stage 6alt thermometer zero-shot subset (S6A-6 wiring surface).

        INERT at S6A-3: no orchestrator walk or test invokes this hook. S6A-6
        replaces the Stage 6alt orchestrator body with the plugin sequencer
        and dispatches this hook in place of the monolith ``run()``'s inline
        ``_lm_eval_subset`` student-side call. The body below reproduces that
        inline call faithfully — it is dead code at S6A-3 but S6A-6 relies on
        it once the monolith ``run()`` becomes a thin shim.

        Reproduces the monolith ``run()``'s student-side call:

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
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise.
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
        arc_limit = int(therm.get("arc_easy_limit", 100))
        hellaswag_limit = int(therm.get("hellaswag_limit", 200))
        batch_size = therm.get("lm_eval_batch_size", "auto:8")

        result = _lm_eval_subset(
            model, tokenizer,
            arc_limit=arc_limit, hellaswag_limit=hellaswag_limit,
            batch_size=batch_size,
        )

        ctx.set("student_arc_easy_acc_norm", result["arc_easy_acc_norm"])
        ctx.set("student_hellaswag_acc_norm", result["hellaswag_acc_norm"])
        ctx.set("student_acc_norm_sum", result["acc_norm_sum"])


__all__ = ["_lm_eval_subset", "ZeroShotSubsetPlugin"]
