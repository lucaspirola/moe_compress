"""Validation final-report (S6-7 of the Stage 6 plugin-architecture refactor).

Home of the Stage 6 FINAL-REPORT concern, extracted from the legacy
``stage6_validate.py`` monolith. The validation-report plugin owns the
post-eval assembly of the ``stage6_eval.json`` artifact: the per-metric
``student``/``teacher``/``delta`` triples (via ``_deltas``), the
measured-reduction ratio (via ``_measured_reduction``), the
threshold-check pass/fail / skipped-checks dict (via ``_check_thresholds``),
the ``overall_pass`` aggregation across boolean threshold results, the
JSON-artifact write, and the Trackio scalar flatten.

Pattern A vs Pattern B
----------------------
S6-7 covers a MIXED pattern:

* **Pattern A -- relocated verbatim**: ``_deltas``,
  ``_measured_reduction`` and ``_check_thresholds`` below are
  character-identical copies of the monolith bodies.
  ``stage6_validate.py`` re-imports the 3 FUNCTIONS (a ``# noqa: F401``
  block) so ``run()`` and external callers/tests keep their original
  import path. **Master plan §8 HOTSPOT** -- ``_deltas`` byte-identity is
  load-bearing for the S6-0 golden snapshot: every NaN/Inf branch
  (student-non-finite, teacher-non-finite, missing-key continue,
  non-numeric continue, the defensive ``not math.isfinite(delta)`` branch
  that is unreachable in IEEE 754 but preserved, and the
  ``_non_finite_skipped`` / ``_teacher_non_finite_skipped`` sentinel-list
  assembly) is preserved character-for-character. ``_check_thresholds``
  is similarly byte-identical -- the ``skipped_checks`` sub-dict's exact
  key names (``arc_challenge_acc_drop_ok``, ``hellaswag_acc_drop_ok``,
  ``humaneval_pass_at_1_drop_ok``, ``math500_accuracy_drop_ok``,
  ``measured_reduction_ok``, ``wikitext2_ppl_increase_ok``) are pinned by
  the S6-0 golden's ``stage6_eval.json`` and must not drift.
* **Pattern B -- reproduced in ONE inert hook**: the FINAL-REPORT block
  at the tail of the monolith ``run()`` is INLINE ``run()`` code -- there
  is nothing standalone to relocate. The ``assemble_report`` hook below
  REPRODUCES that inline block faithfully; the monolith ``run()`` is NOT
  modified for it. This is an intentional, temporary logic duplication
  that resolves at S6-8 when the monolith ``run()`` is deleted and this
  hook is wired live.

Circular-import contract (mirror of ``stage6/plugins/teacher_provider.py``
/ ``stage6/plugins/imatrix_export.py``): this module imports only from
``..context`` / ``...utils`` / sibling plugin modules
(``zero_shot_lm_eval``) / stdlib -- NEVER from ``stage6_validate``,
``stage6.orchestrator`` or ``orchestrator`` at any scope (module-top OR
function-local). The monolith re-imports *this* module at load time, so
a ``from ..stage6_validate import ...`` here would deadlock the import;
nothing in this module does that.

``ValidationReportPlugin`` is registered-but-INERT at S6-7 -- no
orchestrator walk or test invokes its ``assemble_report`` hook. S6-8
plugs the hook into the live Stage 6 plugin sequencer and deletes the
monolith ``run()``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from ..context import PipelineContext
from ...utils.model_io import (
    count_expert_parameters,
    count_parameters_effective,
    load_model,
    save_json_artifact,
)
from ...utils.trackio_log import trackio_log as _trackio_log
from .zero_shot_lm_eval import _ZERO_SHOT_TASKS

log = logging.getLogger(__name__)


# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Module-LOCAL constant -- the monolith keeps its
# own copy (each module's `_STAGE6_ATTN_IMPLEMENTATION` is module-local; see
# the S6-4 docstring note in `stage6_validate.py`). Referenced INSIDE
# `_measured_reduction`'s fallback CPU-load path below.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"


# ---------------------------------------------------------------------------
# Deltas + threshold check
# ---------------------------------------------------------------------------


def _deltas(student: dict, teacher: dict) -> dict:
    # delta = student - teacher: positive means student is worse for PPL
    # (higher is worse), negative means student is worse for accuracy tasks
    # (lower is worse). _check_thresholds interprets each metric's sign.
    out = {}
    non_finite: list[str] = []           # student non-finite → auto-fail in _check_thresholds
    teacher_non_finite: list[str] = []   # teacher non-finite → skip check (not a student failure)
    for k in sorted(set(student) | set(teacher)):
        s = student.get(k)
        t = teacher.get(k)
        if s is None or t is None:
            continue
        try:
            s_finite = math.isfinite(s)
            t_finite = math.isfinite(t)
        except (TypeError, ValueError):
            log.warning("_deltas: non-numeric value for key %r (student=%r, teacher=%r); skipping", k, s, t)
            continue
        # M-1: Check each operand independently so a non-finite *teacher* value
        # (e.g. teacher eval failed → inf PPL) does not trigger auto-failure of
        # the student threshold check.
        if not s_finite:
            # Student non-finite → auto-fail downstream.
            log.warning(
                "_deltas: student value non-finite for key %r (student=%s, teacher=%s); "
                "recording as student non-finite",
                k, s, t,
            )
            non_finite.append(k)
        elif not t_finite:
            # Teacher non-finite → skip threshold check entirely (teacher issue, not student).
            log.warning(
                "_deltas: teacher value non-finite for key %r (teacher=%s); "
                "skipping threshold check for this metric",
                k, t,
            )
            teacher_non_finite.append(k)
        else:
            delta = s - t
            if not math.isfinite(delta):
                # Both operands finite but difference is not (e.g. inf - inf).
                log.warning(
                    "_deltas: delta non-finite for key %r (student=%s, teacher=%s) "
                    "despite finite operands; treating as student non-finite",
                    k, s, t,
                )
                non_finite.append(k)
            else:
                out[k] = {"student": s, "teacher": t, "delta": delta}
    # Record skipped keys so downstream consumers can distinguish "not computed"
    # from "computed but non-finite and omitted".
    if non_finite:
        out["_non_finite_skipped"] = non_finite
    if teacher_non_finite:
        out["_teacher_non_finite_skipped"] = teacher_non_finite
    return out


def _measured_reduction(
    student_model,
    *,
    student_total: int | None = None,
    student_expert: int | None = None,
    teacher_model=None,
    cached_teacher_param_counts: dict | None = None,
    config: dict | None = None,
) -> dict:
    # F-iter4-CRIT-2: use count_parameters_effective to honor FactoredExperts
    # per-expert effective ranks (Spec §9 line 785). For models with no
    # FactoredExperts (e.g. teacher) this equals count_parameters().
    s_total = student_total if student_total is not None else count_parameters_effective(student_model)
    s_expert = student_expert if student_expert is not None else count_expert_parameters(student_model, routed_only=True)

    if teacher_model is not None:
        t_total = count_parameters_effective(teacher_model)
        t_expert = count_expert_parameters(teacher_model, routed_only=True)
    elif cached_teacher_param_counts is not None:
        t_total = cached_teacher_param_counts["total"]
        t_expert = cached_teacher_param_counts["expert"]
        log.info("Using cached teacher param counts: total=%d, expert=%d", t_total, t_expert)
    else:
        if config is None:
            raise RuntimeError("_measured_reduction: config required when teacher_model and cached_teacher_param_counts are both None")
        log.info("Computing teacher param counts via CPU model load")
        try:
            _load_in_4bit = config["model"].get("load_in_4bit", False)
            if _load_in_4bit:
                log.warning(
                    "_measured_reduction: load_in_4bit=True is incompatible with "
                    "device_map='cpu'; loading in full precision."
                )
                _load_in_4bit = False
            # F-C-H-1: attn_implementation="eager" pinned per Spec F-S-M-1.
            teacher_tmp, _ = load_model(
                config["model"]["name_or_path"],
                revision=config["model"].get("revision", "main"),
                torch_dtype=config["model"]["torch_dtype"],
                device_map="cpu",
                attn_implementation=_STAGE6_ATTN_IMPLEMENTATION,
                load_in_4bit=_load_in_4bit,
                trust_remote_code=config["model"].get("trust_remote_code", False),
            )
            try:
                # F-iter4-CRIT-2: effective count (no FactoredExperts in teacher,
                # so equivalent to count_parameters).
                t_total = count_parameters_effective(teacher_tmp)
                t_expert = count_expert_parameters(teacher_tmp, routed_only=True)
            finally:
                del teacher_tmp
        except Exception as exc:
            log.warning("Could not load teacher for param counting (%s) — using 0", exc)
            t_total = 0
            t_expert = 0

    # L-2: When teacher total param count is 0 (param counting failed), the
    # total_reduction_ratio formula produces a meaningless result (1.0 always).
    # Return None so _check_thresholds can skip this check instead of treating it
    # as a pass.
    if t_total == 0:
        log.warning(
            "_measured_reduction: teacher total_params=0 (count failed); "
            "total_reduction_ratio is unreliable — skipping measured_reduction threshold check"
        )
        return {
            "total_student": s_total,
            "total_teacher": t_total,
            "total_reduction_ratio": None,
            "expert_student": s_expert,
            "expert_teacher": t_expert,
            "expert_reduction_ratio": None,
        }

    # F-2: When t_expert == 0 (non-MoE teacher), max(t_expert, 1) yields 1 and
    # s_expert is also 0, so the formula would produce expert_reduction_ratio=1.0 —
    # misleadingly suggesting 100% expert reduction.  Use None instead.
    expert_reduction_ratio = (
        None if t_expert == 0
        else 1.0 - (s_expert / t_expert)
    )
    return {
        "total_student": s_total,
        "total_teacher": t_total,
        "total_reduction_ratio": 1.0 - (s_total / max(t_total, 1)),
        "expert_student": s_expert,
        "expert_teacher": t_expert,
        "expert_reduction_ratio": expert_reduction_ratio,
    }


def _check_thresholds(results: dict, thresholds: dict, *, s6_cfg: dict | None = None) -> dict:
    """Return a dict with boolean per-check results plus a 'skipped_checks' sub-dict.

    The 'skipped_checks' dict maps threshold key names to a reason string so
    downstream consumers can distinguish "threshold not configured" from
    "eval disabled, threshold configured but skipped".
    """
    checks: dict[str, bool] = {}
    # Keys whose threshold was configured but whose eval was disabled — value is reason string.
    skipped_checks: dict[str, str] = {}

    delta = results.get("delta", {})
    wt = delta.get("wikitext2_ppl")
    wt_thresh = thresholds.get("wikitext2_ppl_relative_max_increase", None)
    # Elif ordering matters — non-finite auto-fails must come BEFORE the
    # `wt is None and wt_thresh is None` catch-all, otherwise a student with
    # inf/nan PPL passes silently when no threshold is configured.
    # Actual branch order:
    #   1.  Both wt and wt_thresh present → perform the relative check.
    #   2.  wikitext2_ppl student non-finite → auto-FAIL (H3/M5), regardless of threshold.
    #   3.  wikitext2_ppl teacher non-finite → skip (teacher issue, not student failure).
    #   4.  wt is None AND wt_thresh is None → no eval, no threshold; debug only.
    #   5.  wt_thresh is None → threshold unconfigured; skip (no penalty).
    #   6.  wt is None → data missing despite threshold being set.
    if wt is not None and wt_thresh is not None:
        # Use pre-computed delta (student - teacher): positive = student PPL higher = worse.
        if wt["teacher"] <= 0:
            log.warning(
                "_check_thresholds: teacher PPL <= 0 (%s); skipping relative wikitext2 check",
                wt["teacher"],
            )
            skipped_checks["wikitext2_ppl_increase_ok"] = f"teacher PPL <= 0 ({wt['teacher']})"
        else:
            # rel is a fraction (e.g. 0.03 = 3%).  wt_thresh is stored as a fraction
            # in config (e.g. 0.03 for the ≤ 3% spec limit).  Both sides use the
            # same unit so the comparison is correct.  Log as % for human readability.
            #
            # Defensive sanity check: a threshold > 1.0 means > 100% relative PPL
            # increase is acceptable, which is almost certainly a misconfigured
            # percentage (e.g. 3 instead of 0.03).  Warn loudly so operators catch it.
            if wt_thresh > 1.0:
                log.warning(
                    "_check_thresholds: wikitext2_ppl_relative_max_increase=%.4g looks like "
                    "a percentage (>1.0); expected a fraction (e.g. 0.03 for 3%%).  "
                    "Check your config — the quality gate may be too lenient.",
                    wt_thresh,
                )
            rel = wt["delta"] / wt["teacher"]
            passed = rel <= wt_thresh
            log.info(
                "_check_thresholds: wikitext2_ppl relative increase = %.4f%% "
                "(threshold %.4f%%) → %s",
                rel * 100, wt_thresh * 100, "PASS" if passed else "FAIL",
            )
            checks["wikitext2_ppl_increase_ok"] = passed
    elif "wikitext2_ppl" in delta.get("_non_finite_skipped", []):
        # H3 / M5: A non-finite student PPL (inf/nan) is an automatic failure.
        # MUST be checked BEFORE the `wt is None and wt_thresh is None` catch-all:
        # when wt_thresh is unconfigured, both conditions are true and the catch-all
        # would silence this auto-fail, allowing overall_pass=True for a model with
        # infinite PPL.  Non-finite student values always auto-fail regardless of
        # whether a threshold was configured.
        log.warning(
            "_check_thresholds: wikitext2_ppl was non-finite (student PPL=inf/nan); "
            "treating as automatic threshold FAILURE rather than a skipped check.",
        )
        checks["wikitext2_ppl_increase_ok"] = False
    elif "wikitext2_ppl" in delta.get("_teacher_non_finite_skipped", []):
        # M-1: Teacher PPL was non-finite (teacher eval failed); this is a teacher issue,
        # not a student failure — skip the check rather than auto-failing the student.
        log.warning(
            "_check_thresholds: wikitext2_ppl teacher value was non-finite; "
            "skipping threshold check (teacher eval issue, not student failure).",
        )
        skipped_checks["wikitext2_ppl_increase_ok"] = "teacher wikitext2_ppl non-finite (teacher eval issue)"
    elif wt is None and wt_thresh is None:
        # Neither eval result nor threshold is present — nothing to do.
        # N-2: This is a by-design configuration, not an unexpected condition; use DEBUG.
        log.debug("Threshold key 'wikitext2_ppl_relative_max_increase' missing from config and no wikitext2_ppl result — skipping check")
    elif wt_thresh is None:
        # Threshold not configured but wt result exists — unconfigured threshold, skip.
        log.warning("Threshold key 'wikitext2_ppl_relative_max_increase' missing from config — skipping check")
    else:  # wt is None, wt_thresh is not None
        wikitext2_enabled = (s6_cfg or {}).get("wikitext2", {}).get("enabled", True)
        if not wikitext2_enabled:
            log.warning("wikitext2_ppl threshold configured but eval was disabled; skipping check.")
            skipped_checks["wikitext2_ppl_increase_ok"] = "wikitext2 eval disabled in config"
        else:
            log.warning("wikitext2_ppl threshold configured but no result was produced; marking as failed.")
            checks["wikitext2_ppl_increase_ok"] = False
    for task, key_name in [
        ("arc_challenge_acc", "arc_c_absolute_max_drop"),
        ("hellaswag_acc", "hellaswag_absolute_max_drop"),
        ("humaneval_pass_at_1", "humaneval_absolute_max_drop"),
        ("math500_accuracy", "math500_absolute_max_drop"),
    ]:
        thresh = thresholds.get(key_name, None)
        if thresh is None:
            log.warning("Threshold key '%s' missing from config — skipping check for %s",
                        key_name, task)
            continue
        # Defensive sanity check: accuracy drop thresholds > 1.0 (i.e. > 100pp)
        # almost certainly mean the config stored a percentage instead of a fraction.
        if thresh > 1.0:
            log.warning(
                "_check_thresholds: %s threshold %.4g looks like a percentage (>1.0); "
                "expected a fraction (e.g. 0.015 for 1.5pp).  "
                "Check your config — the quality gate may be too lenient.",
                key_name, thresh,
            )
        d = delta.get(task)
        if d is not None:
            # delta = student - teacher (from _deltas); for accuracy tasks a negative
            # delta means student is worse. drop = teacher - student = -delta.
            # thresh is stored as a fraction in config (e.g. 0.015 for 1.5pp).
            drop = -d["delta"]
            passed = drop <= thresh
            log.info(
                "_check_thresholds: %s drop = %.4f (%.2fpp), threshold %.4f (%.2fpp) → %s",
                task, drop, drop * 100, thresh, thresh * 100, "PASS" if passed else "FAIL",
            )
            checks[f"{task}_drop_ok"] = passed
        else:
            # Metric absent from delta dict — check whether the eval was disabled,
            # whether the teacher value was non-finite (skip), or whether the
            # student value was non-finite (auto-fail).
            _non_finite_skipped = delta.get("_non_finite_skipped", [])
            _teacher_non_finite_skipped = delta.get("_teacher_non_finite_skipped", [])
            if task in _teacher_non_finite_skipped:
                # M-1: Teacher value was non-finite — teacher eval issue, not student
                # failure.  Skip the check rather than auto-failing the student.
                log.warning(
                    "Threshold check for %s: teacher value non-finite (teacher eval issue); "
                    "skipping check (not a student failure).",
                    task,
                )
                skipped_checks[f"{task}_drop_ok"] = "teacher value non-finite (teacher eval issue)"
            elif task in _non_finite_skipped:
                # H3 / M5: Non-finite student value is an automatic failure, not a skip.
                # Putting it in skipped_checks would allow overall_pass=True even though
                # the student produced inf/nan for this metric.
                log.warning(
                    "Threshold check for %s: non-finite student value (inf/nan); "
                    "treating as automatic FAILURE rather than a skipped check.",
                    task,
                )
                checks[f"{task}_drop_ok"] = False
            else:
                if task in _ZERO_SHOT_TASKS:
                    eval_enabled = (s6_cfg or {}).get("zero_shot", {}).get("enabled", True)
                    eval_name = "zero_shot"
                else:
                    eval_enabled = (s6_cfg or {}).get("generative", {}).get("enabled", True)
                    eval_name = "generative"
                if not eval_enabled:
                    log.warning(
                        "Threshold check for %s skipped — %s eval was disabled in config",
                        task, eval_name,
                    )
                    skipped_checks[f"{task}_drop_ok"] = f"{eval_name} eval disabled in config"
                else:
                    log.warning(
                        "Threshold check for %s failed — metric missing from results "
                        "(lm-eval task name mismatch or evaluation error)", task,
                    )
                    checks[f"{task}_drop_ok"] = False
    mr_thresh = thresholds.get("measured_reduction_min", None)
    if mr_thresh is not None:
        mr = results.get("measured_reduction", {})
        mr_ratio = mr.get("total_reduction_ratio")
        # L-2: total_reduction_ratio is None when teacher param count failed (t_total=0).
        # Also skip when it is NaN (shouldn't normally occur, but guard defensively).
        if mr_ratio is None or (isinstance(mr_ratio, float) and math.isnan(mr_ratio)):
            log.warning(
                "_check_thresholds: measured_reduction.total_reduction_ratio is %s "
                "(teacher param count failed); skipping measured_reduction threshold check",
                mr_ratio,
            )
            skipped_checks["measured_reduction_ok"] = (
                "total_reduction_ratio unavailable (teacher param count failed)"
            )
        else:
            checks["measured_reduction_ok"] = mr_ratio >= mr_thresh
    else:
        log.warning("Threshold key 'measured_reduction_min' missing from config — skipping check")

    # Merge skipped_checks into output so artifact consumers can distinguish
    # "not configured" (key absent) from "configured but eval disabled" (key in skipped_checks).
    result_dict: dict = dict(checks)
    if skipped_checks:
        result_dict["skipped_checks"] = skipped_checks
    return result_dict


class ValidationReportPlugin:
    """Stage 6 validation-report plugin (S6-7 -- registered-but-INERT).

    Owns the Stage 6 final-report assembly: per-metric ``student``/``teacher``/
    ``delta`` triples via the relocated ``_deltas``, the measured-reduction
    ratio via the relocated ``_measured_reduction``, the threshold-check
    results dict via the relocated ``_check_thresholds``, the ``overall_pass``
    boolean aggregation across the boolean-valued threshold results, the
    ``stage6_eval.json`` artifact write, and the Trackio scalar flatten. The
    3 standalone helpers (Pattern A) are relocated verbatim above and
    re-imported by the monolith; the inline-in-``run()`` assembly glue is
    reproduced in the ``assemble_report`` hook below (Pattern B).

    S6-7 wires this class into the plugin registry as metadata only -- no
    orchestrator walk or test invokes ``assemble_report``. S6-8 plugs the
    hook into the live Stage 6 plugin sequencer and deletes the monolith
    ``run()``.
    """

    name = "validation_report"
    paper = (
        "Stage 6 final-report assembly -- per-metric student/teacher/delta "
        "triples (_deltas), measured-reduction ratio (_measured_reduction), "
        "threshold gating (_check_thresholds), overall_pass aggregation, "
        "stage6_eval.json artifact write + Trackio scalar flatten."
    )
    config_key = "stage6_validate"
    reads: tuple[str, ...] = (
        "config",
        "artifacts_dir",
        "student_results",
        "teacher_results",
        "student_param_counts",
        "teacher_param_counts",
    )
    writes: tuple[str, ...] = (
        "stage6_results_path",
        "overall_pass",
    )
    # The final-report concern has no calibration-pass accumulators -- it
    # only consumes already-computed metric scalars + param counts and emits
    # the artifact / Trackio scalars / context flags. Same convention as the
    # sibling plugins.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """The final-report concern always runs.

        The artifact (``stage6_eval.json``) is the deliverable of Stage 6;
        even a Stage 6 config that disables every sub-eval still produces a
        report (with empty student/teacher dicts and a ``skipped_checks``
        sub-dict reflecting the disabled evals) -- this is the S6-0 golden
        snapshot's exact shape. So ``is_enabled`` returns ``True``
        unconditionally.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def assemble_report(self, ctx: PipelineContext) -> None:
        """Phase hook -- final-report assembly (S6-8 wiring surface).

        INERT at S6-7: no orchestrator walk or test invokes this hook. S6-8
        replaces the Stage 6 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        final-block. The body below reproduces that inline block faithfully
        -- it is dead code at S6-7 but S6-8 relies on it once the monolith
        ``run()`` is deleted.

        Reproduces the monolith ``run()``'s final block:

        1. Compute ``results["delta"]`` via ``_deltas(student, teacher)``.
        2. Compute ``results["measured_reduction"]`` via
           ``_measured_reduction(...)``; on exception, record an empty dict.
        3. Compute ``results["thresholds"]`` via ``_check_thresholds(...)``.
        4. Aggregate ``overall_pass`` across only the boolean-valued entries
           of ``results["thresholds"]`` (``skipped_checks`` is a dict, not a
           bool -- excluded).
        5. Save ``stage6_eval.json`` via ``save_json_artifact``.
        6. Flatten the metric scalars into the Trackio ``flat`` dict, with
           the ``stage6/student/...`` / ``stage6/teacher/...`` /
           ``stage6/delta/.../{student,teacher,delta}`` /
           ``stage6/measured_reduction/...`` / ``stage6/non_finite_count``
           / ``stage6/overall_pass`` key naming the monolith uses. Note the
           dict-vs-list asymmetry for delta entries: dict-valued entries are
           per-metric ``{student, teacher, delta}`` triples; list-valued
           entries (``_non_finite_skipped`` / ``_teacher_non_finite_skipped``)
           are sentinel keys counted into ``non_finite_count`` instead.
        7. Publish ``ctx.stage6_results_path`` and ``ctx.overall_pass`` for
           downstream consumers.

        Required ctx slots:
          * ``config`` (dict)
          * ``artifacts_dir`` (Path)
          * ``student_results`` (dict[str, float])
          * ``teacher_results`` (dict[str, float])
          * ``student_param_counts`` (dict with ``total`` / ``expert`` keys)
          * ``teacher_param_counts`` (dict with ``total`` / ``expert`` keys
            -- the monolith path that loads the teacher from disk on
            cache-miss falls back to count_parameters_effective on the live
            ``teacher`` model; in the plugin path the counts are supplied
            via this ctx slot, by the teacher-provider plugin)

        Optional ctx slots:
          * ``student_model`` (nn.Module | None -- only consulted when
            ``student_param_counts`` is missing, matching the monolith
            ``run()``'s ``_measured_reduction(model, student_total=...,
            student_expert=...)`` call).
          * ``imatrix_skipped`` (bool -- if present and ``True``, surfaced
            on the results dict; mirrors the monolith ``run()``'s
            ``results["imatrix_skipped"] = True`` assignment on the
            timeout-skip branch).
        """
        config = ctx.get("config")
        artifacts_dir = ctx.get("artifacts_dir")
        s6 = config["stage6_validate"]

        student_results = ctx.get("student_results")
        teacher_results = ctx.get("teacher_results")
        student_pc = ctx.get("student_param_counts")
        teacher_pc = ctx.get("teacher_param_counts")
        student_model = ctx.get("student_model") if ctx.has("student_model") else None

        results: dict = {
            "student": dict(student_results),
            "teacher": dict(teacher_results),
            "delta": {},
            "thresholds": {},
        }

        # 6. Deltas and threshold checks
        results["delta"] = _deltas(results["student"], results["teacher"])
        try:
            meas = _measured_reduction(
                student_model,
                student_total=student_pc.get("total") if student_pc else None,
                student_expert=student_pc.get("expert") if student_pc else None,
                teacher_model=None,  # always use cached counts in the plugin path
                cached_teacher_param_counts=teacher_pc,
                config=config,
            )
        except Exception as exc:
            log.warning("_measured_reduction failed (%s); recording empty dict", exc)
            meas = {}
        results["measured_reduction"] = meas
        # L3: results["thresholds"] has a mixed schema: most values are bool (per-check
        # pass/fail results), but the key "skipped_checks" maps to a dict[str, str]
        # (reason strings for checks that were configured but not performed).
        results["thresholds"] = _check_thresholds(results, s6["thresholds"], s6_cfg=s6)

        path = artifacts_dir / "stage6_eval.json"

        # Mirror the monolith's `results["imatrix_skipped"] = True` assignment
        # on the F-CR2-M-1 timeout-skip branch when the upstream
        # ImatrixExportPlugin has signalled the skip via ctx.imatrix_skipped.
        if ctx.has("imatrix_skipped") and ctx.get("imatrix_skipped"):
            results["imatrix_skipped"] = True

        # Only boolean entries in thresholds count toward overall_pass; skipped_checks is a dict.
        _bool_checks = {k: v for k, v in results["thresholds"].items() if isinstance(v, bool)}
        if not _bool_checks:
            log.warning("Stage 6: no threshold checks were performed (all keys missing from config); overall_pass=False")
            overall_pass = False
        else:
            overall_pass = all(_bool_checks.values())
        results["overall_pass"] = overall_pass
        save_json_artifact(results, path)
        log.info("Stage 6 complete — thresholds %s; detail → %s",
                 "PASS" if overall_pass else "FAIL", path)

        # Trackio: flatten the metric scalars so they appear on the dashboard.
        flat: dict[str, float] = {}
        for side in ("student", "teacher"):
            for k, v in results.get(side, {}).items():
                try:
                    flat[f"stage6/{side}/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
        # F-C-L-1: surface _non_finite_skipped sentinel keys as a single counter
        # so the dashboard sees the failure-mode signal. _deltas writes these as
        # *list* values (NOT dicts), so they are skipped by the per-metric triple
        # block above and would otherwise be invisible on Trackio.
        non_finite_count = 0
        for k, triple in results.get("delta", {}).items():
            if isinstance(triple, dict):
                for sub in ("student", "teacher", "delta"):
                    if sub in triple:
                        try:
                            flat[f"stage6/delta/{k}/{sub}"] = float(triple[sub])
                        except (TypeError, ValueError):
                            pass
            elif isinstance(triple, list) and k in ("_non_finite_skipped", "_teacher_non_finite_skipped"):
                non_finite_count += len(triple)
        flat["stage6/non_finite_count"] = float(non_finite_count)
        for k, v in results.get("measured_reduction", {}).items():
            try:
                flat[f"stage6/measured_reduction/{k}"] = float(v)
            except (TypeError, ValueError):
                pass
        flat["stage6/overall_pass"] = 1.0 if overall_pass else 0.0
        _trackio_log(flat)
        if not overall_pass:
            log.error(
                "One or more quality gates FAILED: %s",
                {k: v for k, v in _bool_checks.items() if not v},
            )

        # Publish the artifact path + overall_pass for downstream consumers.
        ctx.set("stage6_results_path", path)
        ctx.set("overall_pass", overall_pass)


__all__ = [
    "_deltas",
    "_measured_reduction",
    "_check_thresholds",
    "ValidationReportPlugin",
]
