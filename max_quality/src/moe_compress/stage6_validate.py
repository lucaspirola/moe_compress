"""Stage 6 — Validation (compute-time optimized).

Metrics (from VALIDATED_STRATEGIES §Stage 6):

- **WikiText-2 PPL** — primary quality signal.
- **Zero-shot**: ARC-C, HellaSwag. We defer to ``lm-eval`` harness for these
  since reimplementing MC-format scoring per-task is fraught.
- **Generative**: HumanEval (code), MATH-500 (math). These two are light-touch
  — they primarily guard against catastrophic collapse of the compressed
  model on generation-heavy tasks. Full pass@k evaluation is expensive; we
  sample ``num_samples_per_task`` completions per prompt and score with the
  dataset's reference judge.

The uncompressed baseline is re-loaded once at the end and evaluated on the
same prompt slices for apples-to-apples deltas — **unless** teacher eval
caching is enabled, in which case the cached teacher results are used directly.

Artifact: ``stage6_eval.json`` with absolute metrics + deltas + threshold
pass/fail summary.

**Security note — HumanEval code execution (H1, F-C-L-3):**
HumanEval scoring runs model-generated Python in ``spawn`` ProcessPool CHILD
PROCESSES (Item-2), with a shared wall-clock deadline and hard termination of
stuck workers.  This provides subprocess isolation — strictly stronger than the
legacy in-process daemon thread — but is still *best-effort*: there is **no
seccomp / landlock / container boundary**.  Malicious or runaway generated code
can still access the filesystem, network, and its own interpreter state.  Use
only in trusted environments or behind an external sandbox.

Known limitations of the subprocess sandbox (Item-2 supersedes the old
in-process daemon-thread design):
  * The former daemon-thread LEAK (timed-out threads that ran until interpreter
    exit) is FIXED: a worker that exceeds the shared deadline is hard-terminated
    via ``ProcessPoolExecutor.shutdown(wait=False, cancel_futures=True)`` and
    reaped by the OS.
  * A worker stuck in a C extension may briefly ignore ``SIGTERM`` before the
    OS reaps it on pool shutdown; this is strictly better than the old
    never-dies daemon thread.
  * No syscall filter (no seccomp/landlock); generated code can open sockets,
    write to ``/tmp``, run binaries, etc., subject only to OS-level permissions.

**Compute-time optimizations (2026-04-30):**
All optimizations are purely computational scheduling — larger batches, cached
known-constants, overlapped I/O, and torch.compile. No metric, formula,
threshold, or evaluation methodology is changed. All outputs are numerically
identical to the batch_size=1 baseline.

  #1 — WikiText-2 PPL batch_size 1 → configurable (default 8 on H200)
  #2 — lm-eval batch_size=1 → batch_size="auto:8"
  #3 — HumanEval: batched model.generate() (groups of 8–16)
  #4 — MATH-500: batched model.generate() (groups of 8–16)
  #5 — torch.compile for prefill-dominant forward paths
  #6 — Overlap teacher I/O loading with student generative evals
  #7 — Cache teacher baselines (deterministic teacher = same results)
  #8 — Overlap GGUF conversion with teacher eval (CPU-bound)
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# S6-2: eval-environment setup (dataset revision pinning, the cu130/Hopper
# kernel patches, the MoE experts-implementation shim, the imatrix
# calibration-corpus build + its atomic-write helper) relocated to
# stage6/plugins/eval_environment. Re-imported so run() + external callers/
# tests keep their stage6_validate import paths.
from .stage6.plugins.eval_environment import (  # noqa: F401
    # _CANONICAL_DATASET_REVISION_KEYS is NOT consumed by surviving monolith
    # code — it is re-exported solely to keep the `stage6_validate` external/
    # test API surface stable (callers/tests that imported it pre-S6-2 still
    # resolve it here). Do NOT delete it as "unused".
    _CANONICAL_DATASET_REVISION_KEYS,
    _resolve_dataset_revisions,
    _enforce_revision_pinning,
    _atomic_write_text,
    _IMATRIX_CALIB_FILENAME,
    _build_imatrix_calibration_corpus,
    _set_experts_implementation_s6,
    _apply_stage6_kernel_patches,
)

# S6-3: WikiText-2 PPL + zero-shot lm-eval relocated to stage6/plugins/
# {wikitext_ppl,zero_shot_lm_eval}. Re-imported so run(), _check_thresholds()
# and external callers/tests (e.g. stage6alt_thermometer) keep their
# stage6_validate import paths. _wikitext2_ppl / _lm_eval_tasks are the
# Pattern-A relocated functions; _ZERO_SHOT_TASKS is the relocated constant
# (re-imported, not re-declared, so its identity stays single-sourced).
from .stage6.plugins.wikitext_ppl import _wikitext2_ppl  # noqa: F401
from .stage6.plugins.zero_shot_lm_eval import (  # noqa: F401
    _ZERO_SHOT_TASKS,
    _lm_eval_tasks,
)

# S6-4: generative-eval helpers relocated. The shared batched-generation +
# chat-format primitives moved to tools/eval_harness; HumanEval moved to
# stage6/plugins/humaneval; MATH-500 (incl. its boxed-answer grading helpers
# and the optional-SymPy guard) moved to stage6/plugins/math500. Re-imported so
# run(), _check_thresholds() and external callers/tests keep their
# stage6_validate import paths. These are the Pattern-A relocated symbols;
# _STAGE6_ATTN_IMPLEMENTATION is NOT re-imported — each module keeps its own
# module-local copy (see the constant's definition below).
from .tools.eval_harness import (  # noqa: F401
    _generate_batched,
    _stage6_enable_thinking,
    _chat_format_prompts,
    _THINK_BLOCK_RE,
    _PY_FENCE_RE,
    _TRAILING_PROSE_RE,
    _extract_code_from_chat_response,
)
from .stage6.plugins.humaneval import (  # noqa: F401
    _humaneval,
    _check_humaneval,
)
from .stage6.plugins.math500 import (  # noqa: F401
    _math500,
    _check_math,
    _extract_boxed,
    _last_numeric,
    _math_fallback_extract,
)

# S6-5: teacher-eval-cache machinery (cache key + format version + load/save +
# the CPU preload helper) relocated to stage6/plugins/teacher_provider.
# Re-imported so run() and external callers/tests (e.g.
# test_teacher_eval_cache_key_invariant) keep their stage6_validate import
# paths. These are the Pattern-A relocated symbols; the teacher-side eval
# block in run() is NOT modified (S6-5 is mixed Pattern A + Pattern B; the
# Pattern-B teacher-side hook is reproduced in the plugin module and stays
# INERT until S6-8 deletes run()).
from .stage6.plugins.teacher_provider import (  # noqa: F401
    TEACHER_CACHE_FORMAT_VERSION,
    _safe_pkg_version,
    _teacher_cache_key,
    _load_teacher_cache,
    _save_teacher_cache,
    _preload_teacher_to_cpu,
)

# S6-6: imatrix / GGUF pipeline (_background_gguf_convert,
# _write_eval_text_concat, _run_llama_imatrix_with_prebuilt_gguf,
# _generate_imatrix and _find_llama_cpp_dir) relocated to
# stage6/plugins/imatrix_export. Re-imported so run() keeps calling them via
# their original names. The _EVAL_TEXT_CONCAT_FILENAME constant is NOT
# re-imported here -- only the relocated _write_eval_text_concat references
# it, and the plugin module's module-local copy is the single source of
# truth. The stdlib imports (`subprocess`, `shutil`, `sys`, `time`) that the
# monolith previously needed for these bodies were removed in the import
# block at the top of this module; only callers (`run()`) of these symbols
# remain.
from .stage6.plugins.imatrix_export import (  # noqa: F401
    _background_gguf_convert,
    _write_eval_text_concat,
    _run_llama_imatrix_with_prebuilt_gguf,
    _generate_imatrix,
    _find_llama_cpp_dir,
)

# S6-7: final-report concern (_deltas, _measured_reduction, _check_thresholds)
# relocated to stage6/plugins/validation_report. Re-imported so run() and
# external callers/tests keep their stage6_validate import paths. These are
# the Pattern-A relocated symbols; the inline final-block in run() (results
# dict assembly + JSON write + Trackio flatten) is intentionally UNCHANGED
# at S6-7 (Pattern B is reproduced in the plugin's inert ``assemble_report``
# hook; the monolith run() owns the live code until S6-8). The `import math`
# previously needed by these three function bodies was removed in the import
# block at the top of this module; only callers (`run()`) of these symbols
# remain, and `run()` does not reference `math.` directly.
from .stage6.plugins.validation_report import (  # noqa: F401
    _deltas,
    _measured_reduction,
    _check_thresholds,
)

# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Constant — never override at call sites.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"

# S6-6: _EVAL_TEXT_CONCAT_FILENAME relocated to stage6/plugins/imatrix_export
# alongside _write_eval_text_concat (its sole consumer). It is NOT re-imported
# here — surviving monolith run() code never references the constant directly;
# only the relocated _write_eval_text_concat does, and that function resolves
# the constant from the plugin module's own module-local copy.


# S6-5: teacher-eval-cache machinery (TEACHER_CACHE_FORMAT_VERSION constant +
# _safe_pkg_version / _teacher_cache_key / _load_teacher_cache /
# _save_teacher_cache / _preload_teacher_to_cpu) relocated to
# stage6/plugins/teacher_provider — re-imported in the S6-5 ``# noqa: F401``
# block near the top of this module. The teacher-side INLINE block in run()
# is intentionally UNCHANGED at S6-5 (Pattern B is reproduced in the plugin's
# inert ``provide_teacher_side`` hook; the monolith run() owns the live code
# until S6-8).


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# S6-2: _set_experts_implementation_s6 + _apply_stage6_kernel_patches relocated
# to stage6/plugins/eval_environment — re-imported in the S6-2 ``# noqa: F401``
# block near the top of this module.


def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    """Run Stage 6 validation. Thin shim delegating to the plugin orchestrator (S6-8).

    S6-8 flipped the relationship: the REAL Stage 6 orchestration now lives
    in :func:`moe_compress.stage6.orchestrator.run` (a ``PipelineContext`` +
    ``PluginRegistry`` driving the eight Stage 6 plugins through the
    schedule ``setup_environment -> eval_task -> start_gguf_convert ->
    provide_teacher_side -> export_imatrix -> assemble_report``). This
    module retains ``stage6_validate.run`` only as the stable legacy entry
    point — ``run_pipeline.py`` and the golden / smoke tests still call
    ``stage6_validate.run``.

    The import of the orchestrator is function-local: the orchestrator
    does not import ``stage6_validate`` itself (and must not — that would
    close an import cycle through the S6-2..S6-7 re-import blocks above),
    so a module-top ``from .stage6.orchestrator import run`` here is
    unnecessary churn at import time and adds no API value.
    """
    from .stage6.orchestrator import run as _orchestrator_run
    return _orchestrator_run(model, tokenizer, config, artifacts_dir, device=device)


# ---------------------------------------------------------------------------
# Background GGUF conversion (Optimization #8) + post-eval imatrix pipeline
# (_background_gguf_convert, _write_eval_text_concat,
# _run_llama_imatrix_with_prebuilt_gguf, _generate_imatrix and
# _find_llama_cpp_dir) relocated to stage6/plugins/imatrix_export by S6-6.
# All five symbols are re-imported in the S6-6 # noqa: F401 block near the top
# of this module so run() keeps calling them via their original names.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WikiText-2 perplexity (Optimization #1) + zero-shot lm-eval (Optimization #2)
# relocated to stage6/plugins/{wikitext_ppl,zero_shot_lm_eval} by S6-3.
# _wikitext2_ppl, _ZERO_SHOT_TASKS and _lm_eval_tasks are re-imported in the
# S6-3 # noqa: F401 block near the top of this module.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Batched generation (Optimizations #3, #4) + chat-format / thinking-mode
# helpers relocated to tools/eval_harness; the generative evals themselves
# (HumanEval pass@1, MATH-500 accuracy + its boxed-answer grading helpers and
# the optional-SymPy guard) relocated to stage6/plugins/{humaneval,math500}
# by S6-4. _generate_batched, _stage6_enable_thinking, _chat_format_prompts,
# the _THINK_BLOCK_RE/_PY_FENCE_RE/_TRAILING_PROSE_RE regexes,
# _extract_code_from_chat_response, _humaneval, _check_humaneval, _math500,
# _check_math, _extract_boxed, _last_numeric and _math_fallback_extract are
# re-imported in the S6-4 # noqa: F401 block near the top of this module.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Deltas + threshold check (_deltas, _measured_reduction, _check_thresholds)
# relocated to stage6/plugins/validation_report by S6-7. All three symbols are
# re-imported in the S6-7 # noqa: F401 block near the top of this module so
# run() keeps calling them via their original names.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# imatrix calibration + GGUF conversion (full sequential path) — relocated
# to stage6/plugins/imatrix_export by S6-6 along with _find_llama_cpp_dir.
# Both _generate_imatrix and _find_llama_cpp_dir are re-imported in the S6-6
# # noqa: F401 block near the top of this module.
# ---------------------------------------------------------------------------


