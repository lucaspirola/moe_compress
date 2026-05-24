"""Stage 6 orchestrator — the real plugin-driven phase sequencer (S6-8).

S6-1 shipped this module as a thin delegation to the legacy Stage 6 monolith
:func:`moe_compress.stage6_validate.run`. S6-2..S6-7 extracted the Stage 6
validation algorithm (eval-environment setup, WikiText-2 PPL, zero-shot
lm-eval, HumanEval, MATH-500, teacher provider, imatrix export, validation
report) into ``stage6/plugins/``. S6-8 flips the relationship: :func:`run`
here is now the REAL orchestrator and ``stage6_validate.run`` is a thin shim
that delegates to it.

The schedule
------------
SETUP: ``setup_environment`` (EvalEnvironmentPlugin — experts-impl shim,
``model.eval()``, strict revision pinning, imatrix calibration-corpus build,
cu130/Hopper kernel patches, ``torch.compile``, ``masking_utils`` patch).

STUDENT EVALS: ``eval_task`` walked over the four eval plugins
(WikitextPpl, ZeroShotLmEval, HumanEval, Math500); each is gated by its own
``is_enabled``, so only the enabled sub-evals run. Each plugin mutates the
shared ``eval_results`` dict the orchestrator pre-creates.

TEACHER CACHE LOOKUP (orchestrator preamble): resolve the cache key + path,
call ``_load_teacher_cache`` only when ``teacher_eval_cache.enabled`` is
True; publish ``cached_teacher_results`` / ``cached_teacher_param_counts``
on the ctx (None on miss).

BACKGROUND TEACHER PRELOAD (cache-MISS only): start a daemon thread running
``_preload_teacher_to_cpu`` so the teacher download overlaps with the
student-side compute already finished above.

STUDENT PARAM COUNT SNAPSHOT: capture ``count_parameters_effective`` +
``count_expert_parameters`` BEFORE any model.to("cpu") side-effects.

STUDENT GPU FREE (cache-MISS only): move student to CPU and empty the CUDA
cache before the teacher is loaded into VRAM.

GGUF KICKOFF (cache-MISS only): walk ``start_gguf_convert``
(ImatrixExportPlugin) so the background F16-GGUF conversion runs in
parallel with the teacher work. On cache-HIT publish ``gguf_thread=None``,
``gguf_result={}`` so the late imatrix phase falls through to the sequential
fallback that internally short-circuits when imatrix is disabled.

TEACHER SIDE: ``provide_teacher_side`` (TeacherProviderPlugin) handles the
cache-hit short-circuit and the cache-miss full path (preload-join +
fallback load, eager-attn pin, kernel patches, experts-impl shim, optional
torch.compile, the four conditional eval calls, cache save).

STUDENT GPU FREE (cache-HIT only): on the cache-HIT path no teacher was
loaded, so the orchestrator moves the student to CPU here before
llama-imatrix tries to claim the GPU.

POST-EVAL IMATRIX: ``export_imatrix`` (ImatrixExportPlugin — join the GGUF
thread, run llama-imatrix or sequential fallback, write the eval_text_concat
debug side-channel).

REPORT: ``assemble_report`` (ValidationReportPlugin — deltas, measured
reduction, threshold check, ``stage6_eval.json`` write, Trackio flatten).

FINALIZE: return ``run_ctx.get("stage6_results_path")``.

Division of labour
------------------
The plugin hooks own only their algorithm; the orchestrator owns the
teacher-cache lookup, the preload-thread launch, the student-param snapshot,
the GPU-free discipline, and the conditional GGUF kickoff. Every run-glue
block below is a verbatim copy from the monolith ``run()``, just reorganized
around the ``walk_phases`` calls.

Monkeypatch survival (HAZARD H3)
--------------------------------
The S6-0 golden monkeypatches three names on the surface that ``run()``
calls them from. Pre-S6-8 those were attributes of ``stage6_validate`` —
post-S6-8 they live here in the orchestrator module:

* ``_load_teacher_cache`` — imported by direct ``from .plugins.teacher_provider
  import ...`` and called in the orchestrator preamble.
* ``_trackio_log`` — imported by direct ``from ..utils.trackio_log import ...``
  and used for the one-shot Stage 6 config emit.

The third patch, ``_build_imatrix_calibration_corpus``, the plugin calls
from its OWN module scope (``stage6.plugins.eval_environment``); the
golden's monkeypatch repoints the plugin module attribute, not this
orchestrator's.
"""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import torch

from ..pipeline.context import PipelineContext
from ..pipeline.registry import PluginRegistry
from ..tools.phase_walker import walk_phases
from ..utils.model_io import count_expert_parameters, count_parameters_effective
from ..utils.trackio_log import trackio_log as _trackio_log

from .plugins.eval_environment import EvalEnvironmentPlugin
from .plugins.humaneval import HumanEvalPlugin
from .plugins.imatrix_export import ImatrixExportPlugin
from .plugins.math500 import Math500Plugin
from .plugins.teacher_provider import (
    TeacherProviderPlugin,
    _load_teacher_cache,
    _preload_teacher_to_cpu,
    _teacher_cache_key,
)
from .plugins.validation_report import ValidationReportPlugin
from .plugins.wikitext_ppl import WikitextPplPlugin
from .plugins.zero_shot_lm_eval import ZeroShotLmEvalPlugin

log = logging.getLogger(__name__)


def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    """Run Stage 6 validation via the plugin pipeline.

    Threads one :class:`PipelineContext` through the Stage 6 phase schedule
    (setup_environment → eval_task → start_gguf_convert → provide_teacher_side
    → export_imatrix → assemble_report). Returns the ``stage6_eval.json`` path
    written by the validation-report plugin, same as the legacy monolith.
    """
    s6 = config["stage6_validate"]

    # ---- one PipelineContext: input slots + run-glue intermediates --------
    run_ctx = PipelineContext()
    run_ctx.set("model", model)
    run_ctx.set("tokenizer", tokenizer)
    run_ctx.set("config", config)
    run_ctx.set("artifacts_dir", artifacts_dir)
    run_ctx.set("device", device)

    registry = PluginRegistry([
        EvalEnvironmentPlugin(),
        WikitextPplPlugin(),
        ZeroShotLmEvalPlugin(),
        HumanEvalPlugin(),
        Math500Plugin(),
        TeacherProviderPlugin(),
        ImatrixExportPlugin(),
        ValidationReportPlugin(),
    ])
    plugins = registry.enabled(config)

    # One-shot Trackio emit: Stage 6 sub-suite shape and toggles. All values
    # are config reads — pure additive emit, no logic change. Verbatim from
    # the monolith — the exact 6 keys + value derivations must match for the
    # S6-0 golden capture to stay byte-identical.
    _wt2_cfg = (s6.get("wikitext2") or {})
    _zs_cfg = (s6.get("zero_shot") or {})
    _gen_cfg = (s6.get("generative") or {})
    _trackio_log({
        "stage6/config/wikitext2_enabled": bool(_wt2_cfg.get("enabled", False)),
        "stage6/config/wikitext2_seq_len": int(_wt2_cfg.get("sequence_length", 0)),
        "stage6/config/zero_shot_enabled": bool(_zs_cfg.get("enabled", False)),
        "stage6/config/zero_shot_n_tasks": int(len(_zs_cfg.get("tasks", []))),
        "stage6/config/generative_enabled": bool(_gen_cfg.get("enabled", False)),
        "stage6/config/torch_compile": bool(s6.get("torch_compile", False)),
    })

    # ---- setup_environment ----------------------------------------------
    # EvalEnvironmentPlugin.setup_environment performs (in order): the
    # experts-implementation shim, model.eval(), strict revision pinning,
    # the imatrix calibration-corpus build, the cu130/Hopper kernel
    # patches, the torch.compile setup (stashing pre-compile forward),
    # and the masking_utils linear-attention passthrough. It publishes
    # experts_impl / dataset_revisions / imatrix_calib_path /
    # use_torch_compile / pre_compile_forward onto run_ctx.
    walk_phases(("setup_environment",), plugins, run_ctx)

    # Pre-set the shared eval-results collector + the eval-text-concat
    # debug side-channel. The four eval plugins (wikitext / zero_shot /
    # humaneval / math500) mutate `eval_results` in place; the imatrix
    # plugin reads `eval_text_concat` to write the debug artifact.
    run_ctx.set("eval_text_concat", [])
    run_ctx.set("eval_results", {})

    # ---- eval_task (student side) ---------------------------------------
    # Walks the four student-side sub-eval plugins; each is internally
    # gated on its own enabled flag via PluginRegistry.enabled(config)
    # above, so only the enabled sub-evals actually run.
    walk_phases(("eval_task",), plugins, run_ctx)

    # ---- Teacher cache lookup (orchestrator preamble) -------------------
    # Verbatim from the monolith: resolve the cache key + path, call
    # _load_teacher_cache only when teacher_eval_cache.enabled is True;
    # publish the resolved values onto run_ctx so TeacherProviderPlugin's
    # provide_teacher_side hook can read them through the cache-hit
    # short-circuit.
    teacher_cache_cfg = s6.get("teacher_eval_cache", {})
    teacher_cache_enabled = teacher_cache_cfg.get("enabled", False)
    cache_key = _teacher_cache_key(config)
    cache_path = Path(teacher_cache_cfg.get("cache_path") or
                      str(artifacts_dir / "teacher_eval_cache.json"))
    cached_teacher = _load_teacher_cache(cache_path, cache_key) if teacher_cache_enabled else None
    cached_teacher_results = cached_teacher["results"] if cached_teacher else None
    cached_teacher_param_counts = (cached_teacher["param_counts"] if cached_teacher else None)
    run_ctx.set("cached_teacher_results", cached_teacher_results)
    run_ctx.set("cached_teacher_param_counts", cached_teacher_param_counts)

    # ---- Background teacher preload (cache-MISS only) -------------------
    # Optimization #6: overlap the ~3-5 min teacher download/deserialize
    # with the GPU compute done above by the student-side sub-eval
    # plugins. Verbatim from the monolith — same Queue depth, same
    # thread name, same daemon flag, same args order.
    teacher_preload_q: queue.Queue = queue.Queue(maxsize=1)
    preload_thread = None
    if cached_teacher_results is None:
        preload_thread = threading.Thread(
            target=_preload_teacher_to_cpu,
            args=(config, teacher_preload_q),
            daemon=True,
            name="teacher-preload",
        )
        preload_thread.start()
        log.info("Stage 6: teacher preload started in background thread")
    run_ctx.set("teacher_preload_q", teacher_preload_q)
    run_ctx.set("preload_thread", preload_thread)

    # ---- Student param-count snapshot -----------------------------------
    # F-iter4-CRIT-2: use count_parameters_effective so FactoredExperts
    # U/V factors are counted at their per-expert effective ranks (Spec
    # §9 line 785), not the padded slot width allocated by ranks.
    # F-iter4-M-4: snapshot order — AFTER torch.compile (no parameter
    # mutation there) but BEFORE the student is moved to CPU.
    student_total = count_parameters_effective(model)
    student_expert = count_expert_parameters(model, routed_only=True)
    run_ctx.set(
        "student_param_counts",
        {"total": student_total, "expert": student_expert},
    )

    # Bridge: ValidationReportPlugin.assemble_report reads
    # `student_results`, the sub-eval plugins mutate `eval_results`; both
    # point to the SAME dict so the bridge is a single ctx.set, not a
    # copy. Required for the late assemble_report to see the per-eval
    # results the eval_task plugins wrote.
    run_ctx.set("student_results", run_ctx.get("eval_results"))

    # ---- Student GPU free (cache-MISS only) -----------------------------
    # Verbatim from the monolith: free the student GPU before loading
    # the teacher. On the cache-HIT path no teacher load happens, so
    # this is gated to the MISS branch only; the cache-HIT GPU free
    # happens later (mirrors monolith lines 660-671).
    if cached_teacher_results is None:
        try:
            model.to("cpu")
        except Exception as exc:
            log.warning("Could not move student to CPU before teacher load: %s", exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Conditional GGUF start (cache-MISS only) -----------------------
    # Optimization #8: start the F16-GGUF conversion in a background CPU
    # thread so it overlaps with teacher compute on GPU. The monolith
    # only starts this thread on the non-cache-hit path; on cache HIT no
    # GGUF thread is created and the late export_imatrix hook falls
    # through to the sequential fallback (whose internal `enabled` guard
    # handles the imatrix-disabled config).
    if cached_teacher_results is None:
        walk_phases(("start_gguf_convert",), plugins, run_ctx)
    else:
        run_ctx.set("gguf_thread", None)
        run_ctx.set("gguf_result", {})

    # ---- provide_teacher_side -------------------------------------------
    # TeacherProviderPlugin owns the cache-hit short-circuit AND the
    # cache-miss full path (preload-join + fallback direct load,
    # eager-attn pin, kernel patches, experts-impl shim, optional
    # torch.compile, the four conditional teacher-side sub-eval calls,
    # cache save). Publishes teacher_results + teacher_param_counts on
    # run_ctx.
    walk_phases(("provide_teacher_side",), plugins, run_ctx)

    # ---- Student GPU free (cache-HIT only) ------------------------------
    # Mirrors the monolith's else-branch at lines 660-671: on cache HIT
    # no teacher was loaded, so the student is the only resident model
    # and llama-imatrix would race for GPU residency. Move the student
    # to CPU here so the imatrix subprocess has the GPU to itself.
    if cached_teacher_results is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- export_imatrix -------------------------------------------------
    # ImatrixExportPlugin.export_imatrix joins the background GGUF
    # thread, runs llama-imatrix against the WikiText-2-train
    # calibration corpus (or the sequential fallback), and writes the
    # eval_text_concat debug artifact unconditionally.
    walk_phases(("export_imatrix",), plugins, run_ctx)

    # ---- assemble_report ------------------------------------------------
    # ValidationReportPlugin.assemble_report computes deltas + measured
    # reduction + threshold checks, writes stage6_eval.json, flattens
    # the metric scalars to Trackio, and publishes stage6_results_path /
    # overall_pass onto run_ctx.
    walk_phases(("assemble_report",), plugins, run_ctx)

    return run_ctx.get("stage6_results_path")


__all__ = ["run"]
