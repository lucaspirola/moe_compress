"""Thermometer final-report assembly (S6A-5 of the Stage 6alt plugin-architecture refactor).

Home of the Stage 6alt thermometer FINAL-REPORT concern, extracted from
the legacy ``stage6alt_thermometer.py`` monolith. The thermo-report
plugin owns the post-eval assembly of the ``stage6alt_eval.json``
artifact: the per-token ``top1_agreement`` computation (student vs
teacher argmax), the ``bpt_gap`` computation (with ``math.isfinite``
guard on both operands), the ``acc_norm_sum_gap`` computation, the
final ``results`` dict assembly with the 16 top-level keys pinned by
the S6A-0 golden snapshot, and the ``stage6alt_eval.json`` artifact
write.

Pattern A vs Pattern B
----------------------
S6A-5 is **pure Pattern B**:

* **No Pattern A** — there is nothing standalone to relocate. The
  monolith ``run()``'s final-assembly block (top-1 agreement, gap
  computations, results dict assembly, ``save_json_artifact`` call) is
  ALL INLINE ``run()`` code; no module-level helper exists for this
  concern. Therefore S6A-5 introduces ZERO new top-level symbols
  re-exported by the monolith — only the plugin class.
* **Pattern B — reproduced in ONE inert hook**: the
  ``assemble_thermo_report`` hook below REPRODUCES the monolith's
  inline final-assembly block faithfully; the monolith ``run()`` is
  NOT modified for it. This is an intentional, temporary logic
  duplication that resolves at S6A-6 when the orchestrator flip wires
  this hook live and the monolith ``run()`` becomes a thin shim.

The S6A-0 golden snapshot's existing ``stage6alt_eval.json`` pins the
exact key set (16 top-level keys) and the exact gap-None semantics
(both operands must be finite/not-None, else the gap is ``None``); any
deviation in this hook would break the future S6A-6 orchestrator flip.

Circular-import contract (mirror of ``stage6alt/plugins/thermo_teacher_provider.py``):
this module imports only from ``..context`` / ``...utils.model_io`` /
stdlib / torch — NEVER from ``stage6alt_thermometer`` or
``stage6alt.orchestrator`` at any scope (module-top OR function-local).
The monolith re-imports *this* module at load time, so a
``from ..stage6alt_thermometer import ...`` here would deadlock the
import; nothing in this module does that.

``ThermoReportPlugin`` is registered-but-INERT at S6A-5 — no
orchestrator walk or test invokes its ``assemble_thermo_report`` hook.
S6A-6 plugs the hook into the live Stage 6alt plugin sequencer and
deletes the monolith ``run()``'s inline final-assembly block.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch

from ..context import PipelineContext
from ...utils.model_io import save_json_artifact

log = logging.getLogger(__name__)


class ThermoReportPlugin:
    """Stage 6alt thermometer final-report plugin (S6A-5 — registered-but-INERT).

    Owns the Stage 6alt thermometer final-report assembly: the
    ``top1_agreement`` computation, the ``bpt_gap`` / ``acc_norm_sum_gap``
    computations (with finite/None guards), the ``results`` dict
    assembly with the 16 top-level keys pinned by the S6A-0 golden, the
    ``stage6alt_eval.json`` artifact write, and the
    ``stage6alt_eval_path`` ctx publish.

    Pure Pattern B — no standalone helpers to relocate; the monolith
    ``run()``'s inline final-assembly block is reproduced in the inert
    ``assemble_thermo_report`` hook below. The monolith ``run()`` is
    NOT modified by S6A-5.

    S6A-5 wires this class into the plugin registry as metadata only —
    no orchestrator walk or test invokes ``assemble_thermo_report``.
    S6A-6 plugs the hook into the live Stage 6alt plugin sequencer and
    deletes the monolith ``run()``'s final-assembly block.
    """

    name = "thermo_report"
    paper = (
        "Stage 6alt thermometer final-report assembly — top1_agreement "
        "(per-token student-vs-teacher argmax match rate), bpt_gap "
        "(student_bpt - teacher_bpt, finite-guarded), acc_norm_sum_gap "
        "(student_acc_sum - teacher_acc_sum, None-guarded), the 16-key "
        "results dict pinned by the S6A-0 golden, and the "
        "stage6alt_eval.json artifact write. See stage6alt_thermometer.py "
        "module docstring for the bpt_gap / top1_agreement interpretation."
    )
    config_key = "stage6_validate.thermometer"
    reads: tuple[str, ...] = (
        "config",
        "artifacts_dir",
        "student_bpt",
        "student_argmax",
        "student_arc_easy_acc_norm",
        "student_hellaswag_acc_norm",
        "student_acc_norm_sum",
        "teacher_results",
        "teacher_cache_hit",
        "teacher_cache_path",
        "teacher_cache_key",
        "corpus_meta",
    )
    writes: tuple[str, ...] = (
        "stage6alt_eval_path",
    )
    # The final-report concern has no calibration-pass accumulators — it
    # only consumes already-computed scalars / argmax tensors and emits
    # the artifact + ctx slot. Same convention as the sibling
    # ``ThermoTeacherProviderPlugin`` and ``ValidationReportPlugin``.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """The thermometer final-report concern always runs.

        The artifact (``stage6alt_eval.json``) is the deliverable of
        Stage 6alt; every thermometer run produces it (with whatever
        student / teacher scalars the upstream plugins computed).
        ``is_enabled`` returns ``True`` unconditionally.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def assemble_thermo_report(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6alt thermometer final-report (S6A-6 wiring surface).

        INERT at S6A-5: no orchestrator walk or test invokes this hook.
        S6A-6 replaces the Stage 6alt orchestrator body with the plugin
        sequencer and dispatches this hook in place of the monolith
        ``run()``'s inline final-assembly block. The body below
        reproduces that inline block faithfully — it is dead code at
        S6A-5 but S6A-6 relies on it once the monolith ``run()`` is
        deleted.

        Reproduces, in order, the monolith ``run()``'s final-assembly
        block:

        1. **Read required slots** from ctx — config, artifacts_dir,
           student_bpt, student_argmax, the three student lm-eval
           scalars, teacher_results dict, teacher_cache_{hit,path,key},
           corpus_meta.
        2. **Re-derive arc / hsw limits** from the thermometer config
           sub-tree, matching the monolith run()'s top-of-function
           ``int(therm.get("arc_easy_limit", 100))`` /
           ``int(therm.get("hellaswag_limit", 200))`` calls.
        3. **top1_agreement** — None unless both ``student_argmax`` and
           ``teacher_results["teacher_argmax"]`` are present AND their
           shapes match; otherwise log a shape-mismatch warning and
           leave None.
        4. **bpt_gap** — ``student_bpt - teacher_bpt`` only if BOTH are
           finite (``math.isfinite``); else ``None``.
        5. **acc_norm_sum_gap** — ``student_acc_sum - teacher_acc_sum``
           only if BOTH are not None; else ``None``.
        6. **Assemble results** — 16-key dict in the exact order +
           shape pinned by the S6A-0 golden snapshot.
        7. **Save** ``stage6alt_eval.json`` via ``save_json_artifact``.
        8. **Log** completion line (corpus name, student / teacher
           BPT, bpt_gap, top1_agreement, path).
        9. **Publish** ``ctx.stage6alt_eval_path`` for downstream
           consumers.

        Required ctx slots:
          * ``config`` (dict)
          * ``artifacts_dir`` (Path)
          * ``student_bpt`` (float)
          * ``student_argmax`` (torch.Tensor | None)
          * ``student_arc_easy_acc_norm`` (float | None)
          * ``student_hellaswag_acc_norm`` (float | None)
          * ``student_acc_norm_sum`` (float | None)
          * ``teacher_results`` (dict — carries ``teacher_bpt`` /
            ``teacher_argmax`` / ``teacher_arc_easy_acc_norm`` /
            ``teacher_hellaswag_acc_norm`` / ``teacher_acc_norm_sum``)
          * ``teacher_cache_hit`` (bool)
          * ``teacher_cache_path`` (Path | str)
          * ``teacher_cache_key`` (str)
          * ``corpus_meta`` (dict — at minimum has ``name``)
        """
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")

        student_bpt = ctx.get("student_bpt")
        student_argmax = ctx.get("student_argmax")
        student_arc = ctx.get("student_arc_easy_acc_norm")
        student_hsw = ctx.get("student_hellaswag_acc_norm")
        student_acc_sum = ctx.get("student_acc_norm_sum")

        teacher_results = ctx.get("teacher_results")
        teacher_cache_hit = ctx.get("teacher_cache_hit")
        cache_path = ctx.get("teacher_cache_path")
        cache_key = ctx.get("teacher_cache_key")
        corpus_meta = ctx.get("corpus_meta")

        # Re-derive arc / hsw limits from the thermometer config sub-tree.
        # Mirrors the monolith run()'s top-of-function logic; the limits
        # are emitted into the ``lm_eval`` sub-dict of the results.
        s6 = config["stage6_validate"]
        therm = s6.get("thermometer", {}) or {}
        arc_limit = int(therm.get("arc_easy_limit", 100))
        hsw_limit = int(therm.get("hellaswag_limit", 200))

        teacher_bpt = teacher_results["teacher_bpt"]
        teacher_acc_sum = teacher_results.get("teacher_acc_norm_sum")

        # top1_agreement — fraction of corpus positions where student and
        # teacher argmax the same next token. Unlike bpt_gap this does not
        # depend on what text the student was trained on, so it is a fair
        # compression-damage signal on ANY corpus. None if either model
        # skipped a BPT batch.
        top1_agreement = None
        _t_argmax = teacher_results.get("teacher_argmax")
        if student_argmax is not None and _t_argmax is not None:
            _teacher_argmax = torch.as_tensor(_t_argmax, dtype=torch.long)
            if _teacher_argmax.shape == student_argmax.shape:
                top1_agreement = float(
                    (student_argmax == _teacher_argmax).float().mean()
                )
            else:
                log.warning("Stage 6alt: student/teacher argmax shape mismatch "
                            "(%s vs %s) — top1_agreement left None",
                            tuple(student_argmax.shape),
                            tuple(_teacher_argmax.shape))

        results = {
            "stage": "6alt",
            "mode": "thermometer",
            "student_bpt": student_bpt,
            "teacher_bpt": teacher_bpt,
            "bpt_gap": (student_bpt - teacher_bpt
                        if math.isfinite(student_bpt) and math.isfinite(teacher_bpt)
                        else None),
            "student_arc_easy_acc_norm": student_arc,
            "student_hellaswag_acc_norm": student_hsw,
            "student_acc_norm_sum": student_acc_sum,
            "teacher_arc_easy_acc_norm": teacher_results.get("teacher_arc_easy_acc_norm"),
            "teacher_hellaswag_acc_norm": teacher_results.get("teacher_hellaswag_acc_norm"),
            "teacher_acc_norm_sum": teacher_acc_sum,
            "acc_norm_sum_gap": (student_acc_sum - teacher_acc_sum
                                 if (student_acc_sum is not None
                                     and teacher_acc_sum is not None)
                                 else None),
            "top1_agreement": top1_agreement,
            "corpus": corpus_meta,
            "teacher_cache": {
                "path": str(cache_path),
                "key": cache_key,
                "hit": teacher_cache_hit,
            },
            "lm_eval": {
                "arc_easy_limit": arc_limit,
                "hellaswag_limit": hsw_limit,
            },
        }
        path = artifacts_dir / "stage6alt_eval.json"
        save_json_artifact(results, path)
        log.info("Stage 6alt complete: corpus=%s student_bpt=%.4f teacher_bpt=%.4f "
                 "bpt_gap=%s top1_agreement=%s -> %s",
                 corpus_meta.get("name"), student_bpt, teacher_bpt,
                 results["bpt_gap"], top1_agreement, path)

        ctx.set("stage6alt_eval_path", path)


__all__ = ["ThermoReportPlugin"]
