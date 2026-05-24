"""Stage 6alt orchestrator — the real plugin-driven phase sequencer (S6A-6).

S6A-1 shipped this module as a thin delegation to the legacy Stage 6alt
thermometer monolith :func:`moe_compress.stage6alt_thermometer.run`.
S6A-2..S6A-5 extracted the Stage 6alt thermometer algorithm (environment
setup, calibration-corpus build, BPT measurement, ARC-Easy + HellaSwag
zero-shot subset, sweep-shared teacher cache, final-report assembly) into
``stage6alt/plugins/``. S6A-6 flips the relationship: :func:`run` here is
now the REAL orchestrator and ``stage6alt_thermometer.run`` is a thin shim
that delegates to it.

The schedule
------------
SETUP: ``setup_thermo_environment`` (ThermoEnvironmentPlugin — experts-
implementation shim + cu130/Hopper kernel patches; publishes
``experts_impl`` on ctx).

CORPUS: ``build_corpus`` (ThermoCorpusPlugin — selects nemotron held-out
slice or WikiText-2 test split per ``thermometer.corpus`` and tokenizes
the ``(num_seqs, seq_len)`` int64 calib tensor; publishes ``calib_ids`` /
``corpus_meta`` / ``corpus_id`` on ctx).

STUDENT BPT: ``compute_bpt`` (BptMetricPlugin — runs the student forward
pass, returning a finite mean-NLL-in-bits and an optional per-token
argmax; publishes ``student_bpt`` / ``student_argmax`` on ctx).

STUDENT ZERO-SHOT: ``compute_zero_shot_subset`` (ZeroShotSubsetPlugin —
runs ARC-Easy + HellaSwag at small per-task limits; publishes the three
``student_*_acc_norm`` scalars on ctx).

TEACHER SIDE: ``provide_thermo_teacher_side`` (ThermoTeacherProviderPlugin
— internally owns the cache-hit shortcut AND the cache-miss
student-to-CPU swap → eager-attn ``load_model`` → kernel patches +
experts-impl shim → ``_bpt_from_nll`` + ``_lm_eval_subset`` →
``_save_thermo_teacher_cache`` → student-restore path; publishes
``teacher_results`` / ``teacher_bpt`` / ``teacher_argmax`` /
``teacher_cache_hit`` / ``teacher_cache_path`` / ``teacher_cache_key`` on
ctx).

REPORT: ``assemble_thermo_report`` (ThermoReportPlugin — top1_agreement +
bpt_gap + acc_norm_sum_gap computations, the 16-key results dict, the
``stage6alt_eval.json`` artifact write; publishes ``stage6alt_eval_path``
on ctx).

FINALIZE: return ``ctx.get("stage6alt_eval_path")``.

Division of labour
------------------
Unlike Stage 6's orchestrator, the Stage 6alt orchestrator has NO
run-glue between phases: the teacher cache-hit shortcut, the student-to-
CPU swap discipline, the cache key/path resolution — all of it is
internal to ThermoTeacherProviderPlugin. The orchestrator's job is purely
to thread one :class:`PipelineContext` through the six phases sequentially.

Monkeypatch survival (HAZARD H3)
--------------------------------
The S6A-0 golden patches six names. Pre-S6A-6 those were attributes of
``stage6alt_thermometer`` (re-imported from the plugin modules per S6A-2..
S6A-4); post-S6A-6 they must live on the plugin modules themselves so
``monkeypatch.setattr`` on the plugin module attribute bites the call site:

* ``_set_experts_implementation_s6`` — patched on
  ``stage6alt.plugins.thermo_environment``; called by
  ``ThermoEnvironmentPlugin.setup_thermo_environment`` and by
  ``ThermoTeacherProviderPlugin.provide_thermo_teacher_side`` (the latter
  imports it from the same eval-environment helper module).
* ``_apply_stage6_kernel_patches`` — patched on
  ``stage6alt.plugins.thermo_environment``; same call sites as above.
* ``_build_thermo_corpus`` — patched on ``stage6alt.plugins.thermo_corpus``;
  called by ``ThermoCorpusPlugin.build_corpus``.
* ``_bpt_from_nll`` — patched on ``stage6alt.plugins.bpt_metric``; called
  by ``BptMetricPlugin.compute_bpt`` and (re-imported on the teacher-
  provider module) by ``ThermoTeacherProviderPlugin.provide_thermo_teacher_side``.
* ``_lm_eval_subset`` — patched on ``stage6alt.plugins.zero_shot_subset``;
  called by ``ZeroShotSubsetPlugin.compute_zero_shot_subset`` and (re-
  imported on the teacher-provider module) by the teacher-side hook.
* ``_load_thermo_teacher_cache`` — patched on
  ``stage6alt.plugins.thermo_teacher_provider``; called by the teacher-
  side hook directly.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch  # noqa: F401  -- kept for symmetry with stage6/orchestrator

from ..pipeline.context import PipelineContext
from ..pipeline.registry import PluginRegistry
from ..tools.phase_walker import walk_phases

from .plugins.bpt_metric import BptMetricPlugin
from .plugins.thermo_corpus import ThermoCorpusPlugin
from .plugins.thermo_environment import ThermoEnvironmentPlugin
from .plugins.thermo_report import ThermoReportPlugin
from .plugins.thermo_teacher_provider import ThermoTeacherProviderPlugin
from .plugins.zero_shot_subset import ZeroShotSubsetPlugin

log = logging.getLogger(__name__)


def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    """Run Stage 6alt thermometer via the plugin pipeline.

    Threads one :class:`PipelineContext` through the Stage 6alt phase
    schedule (setup_thermo_environment → build_corpus → compute_bpt →
    compute_zero_shot_subset → provide_thermo_teacher_side →
    assemble_thermo_report). Returns the ``stage6alt_eval.json`` path
    written by the report plugin, same as the legacy monolith.
    """
    # ---- one PipelineContext: input slots only -------------------------
    # Unlike Stage 6's orchestrator, there is no run-glue accumulator to
    # pre-set — every intermediate slot is produced by a plugin hook.
    run_ctx = PipelineContext()
    run_ctx.set("model", model)
    run_ctx.set("tokenizer", tokenizer)
    run_ctx.set("config", config)
    run_ctx.set("artifacts_dir", artifacts_dir)
    run_ctx.set("device", device)

    registry = PluginRegistry([
        ThermoEnvironmentPlugin(),
        ThermoCorpusPlugin(),
        BptMetricPlugin(),
        ZeroShotSubsetPlugin(),
        ThermoTeacherProviderPlugin(),
        ThermoReportPlugin(),
    ])
    plugins = registry.enabled(config)

    log.info("=== Stage 6alt — Thermometer ===")

    # ---- setup_thermo_environment --------------------------------------
    # ThermoEnvironmentPlugin.setup_thermo_environment performs (in order):
    # the experts-implementation shim and the cu130/Hopper kernel patches
    # on the student. Publishes ``experts_impl`` on run_ctx so the later
    # teacher-side hook can apply the matching shim without re-resolving
    # the env-var-vs-config override.
    walk_phases(("setup_thermo_environment",), plugins, run_ctx)

    # ---- build_corpus ---------------------------------------------------
    # ThermoCorpusPlugin.build_corpus calls _build_thermo_corpus and
    # publishes the three return values as ``calib_ids`` / ``corpus_meta``
    # / ``corpus_id`` on run_ctx.
    walk_phases(("build_corpus",), plugins, run_ctx)

    # ---- compute_bpt (student) ------------------------------------------
    # BptMetricPlugin.compute_bpt runs _bpt_from_nll on the student and
    # publishes ``student_bpt`` + ``student_argmax`` on run_ctx.
    walk_phases(("compute_bpt",), plugins, run_ctx)

    # ---- compute_zero_shot_subset (student) -----------------------------
    # ZeroShotSubsetPlugin.compute_zero_shot_subset runs the small
    # ARC-Easy + HellaSwag lm-eval subset on the student and publishes
    # the three student_*_acc_norm scalars on run_ctx.
    walk_phases(("compute_zero_shot_subset",), plugins, run_ctx)

    # ---- provide_thermo_teacher_side ------------------------------------
    # ThermoTeacherProviderPlugin.provide_thermo_teacher_side owns the
    # cache-hit shortcut AND the cache-miss full path (student-to-CPU
    # swap, eager-attn load_model, kernel patches + experts-impl shim,
    # _bpt_from_nll + _lm_eval_subset on the teacher, cache save,
    # teacher-free + student-restore). Publishes teacher_results,
    # teacher_bpt, teacher_argmax, teacher_cache_hit, teacher_cache_path,
    # teacher_cache_key on run_ctx.
    walk_phases(("provide_thermo_teacher_side",), plugins, run_ctx)

    # ---- assemble_thermo_report -----------------------------------------
    # ThermoReportPlugin.assemble_thermo_report computes top1_agreement,
    # bpt_gap, acc_norm_sum_gap, assembles the 16-key results dict,
    # writes stage6alt_eval.json via save_json_artifact, and publishes
    # ``stage6alt_eval_path`` on run_ctx.
    walk_phases(("assemble_thermo_report",), plugins, run_ctx)

    return run_ctx.get("stage6alt_eval_path")


__all__ = ["run"]
