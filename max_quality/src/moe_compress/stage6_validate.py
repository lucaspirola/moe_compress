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
``_check_humaneval`` executes model-generated Python code via ``exec()`` inside
a daemon thread with a wall-clock timeout.  This provides *best-effort*
sandboxing only — there is **no process isolation** (no subprocess, no
seccomp, no container boundary).  Malicious or runaway generated code can
access the filesystem, network, and interpreter state.  Use only in trusted
environments or behind an external sandbox.

Known limitations of the in-process sandbox:
  * Daemon threads that exceed the timeout are NOT killed — they leak silently
    until interpreter exit (counted via ``_leaked_counter`` and surfaced as a
    warning at the end of the eval).
  * Wall-clock timeouts via ``Thread.join(timeout=...)`` do not interrupt
    long-running C extensions or syscalls inside the exec body. (POSIX ``signal``
    -based timeouts would interrupt syscalls but are not used here because they
    only work on the main thread; signal-based timeouts are POSIX-only anyway.)
  * No syscall filter (no seccomp/landlock); generated code can open sockets,
    write to ``/tmp``, exec binaries, etc., subject only to OS-level permissions.

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
import os
import queue
import re
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

from .utils.model_io import (
    count_expert_parameters,
    count_parameters_effective,
    load_model,
    save_json_artifact,
)
from .utils.trackio_log import trackio_log as _trackio_log

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
    s6 = config["stage6_validate"]

    # Set MoE forward dispatch (default 'batched_mm' to work around the
    # grouped_mm Blackwell deadlock — see project memory
    # `project_grouped_mm_blackwell.md`). Same shim as stage5_router_kd;
    # env var `EXPERTS_IMPLEMENTATION` overrides YAML for quick A/B.
    _experts_impl = os.environ.get(
        "EXPERTS_IMPLEMENTATION", s6.get("experts_implementation", "batched_mm")
    )
    _set_experts_implementation_s6(model, _experts_impl)

    model.eval()   # stage5 leaves model in train(); set eval before any sub-metric
    results: dict = {"student": {}, "teacher": {}, "delta": {}, "thresholds": {}}

    # One-shot Trackio emit: Stage 6 eval-suite shape and toggles. All values
    # are config reads — pure additive emit, no logic change.
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

    # F-C-H-3: enforce strict revision pinning early — fail fast on a misconfigured
    # production run rather than after expensive teacher loads / evals.
    dataset_revisions = _enforce_revision_pinning(config)

    # F-C-C-1: build the imatrix calibration corpus from WikiText-2 *train* split,
    # written to artifacts_dir/calibration_wiki_train.txt. The eval-text concat
    # below is captured separately as a debug side-channel only — it is NOT used
    # by imatrix anymore (Spec §9 mandate).
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    imatrix_calib_path = _build_imatrix_calibration_corpus(artifacts_dir, dataset_revisions)
    eval_text_concat: list[str] = []  # debug side-channel only — see eval_text_concat.txt

    # Optimization #5: torch.compile for prefill-dominant paths.
    # Compile model.forward before evaluations begin; model.generate also benefits
    # since it calls model.forward internally for each prefill step.
    # dynamic=True handles variable-length padded batches from lm-eval.
    # One-time compilation cost (~3-5 min) is amortized across 1000+ forward passes.
    #
    # mode='default' (not 'reduce-overhead'): the 2026-05-13 04:31 A0 run with
    # reduce-overhead hung in lm-eval's loglikelihood loop after ~10s of activity
    # (faulthandler thread dump showed `torch._inductor/compile_worker/subproc_pool.py`
    # — same CUDA-graph deadlock that hung Stage 2.5's grouped_mm). lm-eval issues
    # 44k+ requests with many distinct input shapes; reduce-overhead's CUDA graph
    # capture/replay can't keep up with shape churn (we also saw 8 `recompile_limit`
    # warnings before death). `mode='default'` keeps TorchDynamo + TorchInductor
    # fusion but drops graph capture, eliminating the deadlock at the cost of
    # ~10-15% per-forward speed.
    use_torch_compile = s6.get("torch_compile", False)
    # Apply the cu130/Hopper segfault-fix patches to the student
    # UNCONDITIONALLY (not gated on use_torch_compile). The fla kernel
    # crashes happen during eager generate() regardless of compile state,
    # and the helper is a no-op on models that don't have GatedDeltaNet
    # modules. Mirrors the unconditional teacher-side patch call below.
    _apply_stage6_kernel_patches(model, role="student")

    # If we compile model.forward below, stash the pre-compile bound method
    # here so the generative block (HumanEval/MATH-500) can restore it.
    # torch.compile(dynamic=True) on autoregressive generate() drives an
    # Inductor recompile storm on the growing cache_position plus exposes
    # a Triton/Inductor codegen path that's unstable on cu130 for batch=1
    # decode shapes — we keep compile ON for PPL + lm_eval (prefill-only,
    # works today) and revert to eager for generate() (~minutes of slowdown,
    # <1% of full ablation wall).
    _pre_compile_forward = None
    if use_torch_compile:
        log.info("Stage 6: applying torch.compile(dynamic=True, mode='default') to model.forward")
        try:
            # NOTE: previously set `model.generation_config.cache_implementation = "static"`
            # here to dodge dynamic-cache recompile storms during autoregressive
            # generate(). With our other Stage 6 fixes in place (Dynamo bypass on
            # GatedDeltaNet + torch-native fla/tilelang fallback +
            # TORCHDYNAMO_CACHE_SIZE_LIMIT=512), the dynamic-cache path now works
            # without storm. Keeping StaticCache active triggered a transformers
            # bug at modeling_qwen3_5_moe.py:1396 → create_causal_mask, where the
            # static-cache prefill path passes a dict instead of a tensor (raises
            # `AttributeError: 'dict' object has no attribute 'ndim'` during the
            # FIRST HumanEval forward). The torch.compile path itself is robust
            # to dynamic shapes here because we capped recompile_limit at 512;
            # actual unique attention-layer shapes per HumanEval/MATH run are far
            # below that. So we leave generation_config alone and let HF's default
            # DynamicCache handle the prefill+decode.
            # Capture the pre-compile bound method BEFORE wrapping so the
            # generative block can restore it (Option C: keep compile for
            # prefill-only paths; eager for generate()).
            _pre_compile_forward = model.forward
            model.forward = torch.compile(model.forward, dynamic=True, mode="default")
            log.info("Stage 6: torch.compile applied successfully")
        except Exception as exc:
            log.warning("Stage 6: torch.compile failed (%s) — continuing without compilation", exc)
            use_torch_compile = False
            _pre_compile_forward = None

    # transformers' LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING is missing an
    # entry for 'linear_attention' in 4.x, but Qwen3.5-MoE's GatedDeltaNet
    # layers register that pattern. create_masks_for_generate (called by
    # generate's prefill path when cache_implementation='static' is active)
    # then raises KeyError: 'linear_attention' at masking_utils.py:1479
    # before the first HumanEval token is produced. Register a passthrough
    # mapping to the same function as 'full_attention' — GatedDeltaNet
    # doesn't consume the attention mask anyway (it derives causality from
    # internal conv1d state via the torch-native fallback we just installed).
    # Same math, same outputs, no quality compromise.
    try:
        from transformers import masking_utils as _mu
        _mapping = getattr(_mu, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", None)
        if isinstance(_mapping, dict) and "linear_attention" not in _mapping:
            if "full_attention" in _mapping:
                _mapping["linear_attention"] = _mapping["full_attention"]
                log.info("Stage 6: registered 'linear_attention' → full_attention mask "
                         "in LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING (transformers missing "
                         "entry for Qwen3.5-MoE GatedDeltaNet)")
    except ImportError:
        pass

    # Read batch size configs with defaults tuned for H200.
    ppl_batch_size = int(s6.get("ppl_batch_size", 8))
    # F-iter4-LOW-2: validate lm_eval_batch_size — accept positive int, an
    # int-string, or the "auto[:N]" pattern. Reject anything else early so an
    # invalid config doesn't surface as a confusing lm-eval traceback later.
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
    gen_batch_size = int(s6.get("gen_batch_size", 8))
    if gen_batch_size <= 0:
        raise ValueError(
            f"stage6_validate.gen_batch_size must be a positive int; got {gen_batch_size!r}."
        )

    # 1. WikiText-2 PPL on student (Optimization #1: batch_size=8)
    if s6["wikitext2"]["enabled"]:
        log.info("Stage 6: WikiText-2 PPL (student), batch_size=%d", int(ppl_batch_size))
        results["student"]["wikitext2_ppl"] = _wikitext2_ppl(
            model, tokenizer, s6["wikitext2"], device=device, collect=eval_text_concat,
            batch_size=ppl_batch_size, dataset_revisions=dataset_revisions,
        )

    # 2. Zero-shot via lm-eval (ARC-C + HellaSwag) (Optimization #2: batch_size=auto:8)
    if s6["zero_shot"]["enabled"]:
        log.info("Stage 6: zero-shot harness, batch_size=%s", lm_eval_batch_size)
        results["student"].update(
            _lm_eval_tasks(model, tokenizer, s6["zero_shot"]["tasks"],
                           collect=eval_text_concat, batch_size=lm_eval_batch_size)
        )

    # Optimization #6: Begin preloading teacher weights to host RAM in a background
    # thread while student generative evals (HumanEval, MATH-500) run on GPU.
    # This overlaps the ~3-5 min teacher download/deserialize with GPU compute.
    teacher_cache_cfg = s6.get("teacher_eval_cache", {})
    teacher_cache_enabled = teacher_cache_cfg.get("enabled", False)
    cache_key = _teacher_cache_key(config)
    cache_path = Path(teacher_cache_cfg.get("cache_path") or
                      str(artifacts_dir / "teacher_eval_cache.json"))
    cached_teacher = _load_teacher_cache(cache_path, cache_key) if teacher_cache_enabled else None
    cached_teacher_results = cached_teacher["results"] if cached_teacher else None
    cached_teacher_param_counts = (cached_teacher["param_counts"] if cached_teacher else None)

    teacher_preload_q: queue.Queue = queue.Queue(maxsize=1)
    preload_thread = None
    if cached_teacher_results is None:
        # We need the teacher — start preloading to CPU RAM in background.
        preload_thread = threading.Thread(
            target=_preload_teacher_to_cpu,
            args=(config, teacher_preload_q),
            daemon=True,
            name="teacher-preload",
        )
        preload_thread.start()
        log.info("Stage 6: teacher preload started in background thread")

    # 3. Generative — HumanEval + MATH-500 (Optimizations #3, #4: batched generate)
    if s6["generative"]["enabled"]:
        # Restore uncompiled forward for generate(): see _pre_compile_forward
        # rationale above. PPL + lm_eval ran with the compiled forward (works
        # today); generate() runs eager to dodge cu130 decode-shape codegen
        # bugs and Inductor recompile storms on growing cache_position.
        if _pre_compile_forward is not None:
            model.forward = _pre_compile_forward
            log.info("Stage 6: restored uncompiled model.forward for generative "
                     "block (keep PPL/lm_eval compiled, generative eager)")
        # Switch MoE dispatch to batched_mm for the generative block only.
        # torch._grouped_mm on cu130 crashes on B=1 decode-shape (tiny per-
        # expert groups + changing `offs` tensor every decode step). batched_mm
        # is ~5-10% slower per step but generative is single-digit % of Stage 6
        # wall, so absolute cost is minutes. PPL + lm_eval keep whatever the
        # YAML/env specified (grouped_mm by default on Hopper).
        _gen_experts_impl = os.environ.get("EXPERTS_IMPLEMENTATION_GENERATIVE", "batched_mm")
        if _gen_experts_impl != _experts_impl:
            log.info("Stage 6: switching experts_implementation %r → %r for "
                     "generative block (cu130 _grouped_mm decode-shape workaround)",
                     _experts_impl, _gen_experts_impl)
            _set_experts_implementation_s6(model, _gen_experts_impl)
        log.info("Stage 6: generative (HumanEval + MATH-500), gen_batch_size=%d", int(gen_batch_size))
        if "humaneval" in s6["generative"]:
            # F-CR2-L-1: schema preservation — accept `num_samples_per_task` for
            # future operators who may want to ablate, but assert it equals 1.
            # Spec D-humaneval-greedy mandates greedy single-sample pass@1 (NOT
            # Chen-2021-style stochastic pass@1 that requires k>=10 samples).
            _humaneval_cfg = s6["generative"]["humaneval"]
            _nspt = _humaneval_cfg.get("num_samples_per_task", 1)
            if int(_nspt) != 1:
                raise ValueError(
                    f"stage6_validate.generative.humaneval.num_samples_per_task must be 1 "
                    f"(spec D-humaneval-greedy: greedy single-sample pass@1); got {_nspt}. "
                    f"Stochastic pass@1 (Chen 2021) requires a different harness — not supported here."
                )
            results["student"]["humaneval_pass_at_1"] = _humaneval(
                model, tokenizer, s6["generative"]["humaneval"], device=device,
                collect=eval_text_concat, batch_size=gen_batch_size,
                dataset_revisions=dataset_revisions,
            )
        if "math500" in s6["generative"]:
            results["student"]["math500_accuracy"] = _math500(
                model, tokenizer, s6["generative"]["math500"], device=device,
                collect=eval_text_concat, batch_size=gen_batch_size,
                dataset_revisions=dataset_revisions,
            )

    # 4. Snapshot student param counts BEFORE loading teacher.
    # F-iter4-CRIT-2: use count_parameters_effective so FactoredExperts U/V
    # factors are counted at their per-expert effective ranks (Spec §9 line 785),
    # not the padded slot width allocated by ranks. The padded zero columns are
    # not real parameters.
    # F-iter4-M-4: snapshot order — AFTER torch.compile (no parameter mutation
    # there) but BEFORE the student is moved to CPU (a CPU move does not
    # change numel() so this is for pinning the lifecycle order, not numerics).
    student_total = count_parameters_effective(model)
    student_expert = count_expert_parameters(model, routed_only=True)

    # Initialize gguf_thread and gguf_result at this scope level so the
    # imatrix dispatch below can reference them regardless of which branch runs.
    # L5: Cross-thread dict mutation contract — _background_gguf_convert writes
    # to gguf_result (specifically gguf_result["f16_path"]) only before it exits.
    # All reads of gguf_result in this function occur after gguf_thread.join(),
    # which ensures the background thread has fully exited and its writes are
    # visible.  Do NOT read gguf_result before gguf_thread.join() completes.
    gguf_thread = None
    gguf_result: dict = {}

    # Optimization #7: Use cached teacher results if available.
    teacher = None  # ensure teacher is bound even on the cache-hit path so the cleanup block below is always valid
    if cached_teacher_results is not None:
        log.info("Stage 6: using cached teacher results (key=%s)", cache_key)
        results["teacher"] = cached_teacher_results
    else:
        # 5. Free student GPU memory before loading teacher.
        try:
            model.to("cpu")
        except Exception as exc:
            log.warning("Could not move student to CPU before teacher load: %s", exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Wait for preload thread to finish (teacher weights should be in host RAM).
        teacher_preloaded = None
        if preload_thread is not None:
            log.info("Stage 6: waiting for teacher preload thread to complete")
            preload_thread.join(timeout=3600)
            if preload_thread.is_alive():
                log.warning("Preload thread did not complete within 3600s; proceeding without preloaded teacher.")
            else:
                teacher_preloaded = teacher_preload_q.get_nowait() if not teacher_preload_q.empty() else None
        if teacher_preloaded is not None:
            # Teacher was preloaded to CPU — move to GPU.
            teacher = teacher_preloaded
            log.info("Stage 6: moving preloaded teacher to GPU")
            teacher.to(device or "cuda")
        else:
            # Preload failed or wasn't started — load directly.
            # F-C-H-1: pin attn_implementation="eager" per Spec F-S-M-1 regardless of config.
            log.info(
                "Stage 6: loading uncompressed baseline for delta computation "
                "(attn_implementation=%r forced per spec F-S-M-1)",
                _STAGE6_ATTN_IMPLEMENTATION,
            )
            teacher, _ = load_model(
                config["model"]["name_or_path"],
                revision=config["model"].get("revision", "main"),
                torch_dtype=config["model"]["torch_dtype"],
                device_map=config["model"]["device_map"],
                attn_implementation=_STAGE6_ATTN_IMPLEMENTATION,
                load_in_4bit=config["model"].get("load_in_4bit", False),
                trust_remote_code=config["model"].get("trust_remote_code", False),
            )
        teacher.eval()

        # Apply the same cu130/Hopper segfault-fix patches to the teacher.
        # Without this the teacher's HumanEval generate() segfaults exactly
        # like the student's used to (same Qwen3.5-MoE architecture + same
        # fla FusedRMSNormGated Triton kernel crash on decode shape).
        _apply_stage6_kernel_patches(teacher, role="teacher")

        # Mirror the student-side initial experts_implementation set so the
        # later "switch for generative" comparison has a valid baseline (else
        # `_experts_implementation` is None and the switch log misleadingly
        # reads "None → batched_mm" on every run).
        _set_experts_implementation_s6(teacher, _experts_impl)

        # Optimization #5: torch.compile on teacher too. Mode 'default'
        # (was 'reduce-overhead'): same rationale as student compile —
        # reduce-overhead's CUDA-graph capture can't keep up with shape
        # churn under lm-eval and hangs; default mode is robust.
        _teacher_pre_compile_forward = None
        if use_torch_compile:
            try:
                _teacher_pre_compile_forward = teacher.forward
                teacher.forward = torch.compile(teacher.forward, dynamic=True, mode="default")
                log.info("Stage 6: torch.compile applied to teacher (mode=default)")
            except Exception as exc:
                log.warning("Stage 6: torch.compile on teacher failed (%s)", exc)
                _teacher_pre_compile_forward = None

        # Optimization #8: Start GGUF conversion in background (CPU-bound)
        # while teacher evaluation runs on GPU. The student checkpoint on disk
        # is already available from Stage 5.
        if s6.get("imatrix", {}).get("enabled", True):
            gguf_thread = threading.Thread(
                target=_background_gguf_convert,
                args=(s6.get("imatrix", {}), artifacts_dir, gguf_result),
                daemon=True,
                name="gguf-convert",
            )
            gguf_thread.start()
            log.info("Stage 6: GGUF conversion started in background (CPU-bound)")

        if s6["wikitext2"]["enabled"]:
            results["teacher"]["wikitext2_ppl"] = _wikitext2_ppl(
                teacher, tokenizer, s6["wikitext2"], device=device,
                batch_size=ppl_batch_size, dataset_revisions=dataset_revisions,
            )
        if s6["zero_shot"]["enabled"]:
            results["teacher"].update(
                _lm_eval_tasks(teacher, tokenizer, s6["zero_shot"]["tasks"],
                               batch_size=lm_eval_batch_size)
            )
        if s6["generative"]["enabled"]:
            # Same cu130 generative workarounds as the student-side block:
            # restore uncompiled teacher.forward (eager generate dodges Inductor
            # recompile storm on growing cache_position + decode-shape codegen
            # crashes) and switch experts_implementation to batched_mm
            # (`torch._grouped_mm` crashes on B=1 decode-shape on cu130).
            if _teacher_pre_compile_forward is not None:
                teacher.forward = _teacher_pre_compile_forward
                log.info("Stage 6: restored uncompiled teacher.forward for "
                         "generative block (keep PPL/lm_eval compiled, "
                         "generative eager)")
            _teacher_gen_experts_impl = os.environ.get(
                "EXPERTS_IMPLEMENTATION_GENERATIVE", "batched_mm"
            )
            _teacher_cfg = getattr(teacher, "_orig_mod", teacher).config
            _teacher_current_impl = getattr(_teacher_cfg, "_experts_implementation", None)
            if _teacher_gen_experts_impl != _teacher_current_impl:
                log.info("Stage 6: switching teacher experts_implementation "
                         "%r → %r for generative block",
                         _teacher_current_impl, _teacher_gen_experts_impl)
                _set_experts_implementation_s6(teacher, _teacher_gen_experts_impl)
            if "humaneval" in s6["generative"]:
                results["teacher"]["humaneval_pass_at_1"] = _humaneval(
                    teacher, tokenizer, s6["generative"]["humaneval"], device=device,
                    batch_size=gen_batch_size, dataset_revisions=dataset_revisions,
                )
            if "math500" in s6["generative"]:
                results["teacher"]["math500_accuracy"] = _math500(
                    teacher, tokenizer, s6["generative"]["math500"], device=device,
                    batch_size=gen_batch_size, dataset_revisions=dataset_revisions,
                )

        # Save teacher results to cache for future runs.
        if teacher_cache_enabled:
            # F-iter4-CRIT-2: teacher has no FactoredExperts modules so the
            # effective count == physical count, but use the same effective
            # function for symmetry and so the cached value compares apples-to-
            # apples with the student's effective live param count.
            teacher_pc = {
                "total": count_parameters_effective(teacher),
                "expert": count_expert_parameters(teacher, routed_only=True),
            }
            try:
                _save_teacher_cache(cache_path, cache_key, results["teacher"],
                                    teacher_param_counts=teacher_pc)
            except Exception as exc:
                log.warning("_save_teacher_cache: failed (%s); continuing without cache", exc)

    # 6. Deltas and threshold checks
    results["delta"] = _deltas(results["student"], results["teacher"])
    try:
        meas = _measured_reduction(
            model,
            student_total=student_total, student_expert=student_expert,
            teacher_model=teacher,  # may be None if cached
            cached_teacher_param_counts=cached_teacher_param_counts,
            config=config,
        )
    except Exception as exc:
        log.warning("_measured_reduction failed (%s); recording empty dict", exc)
        meas = {}
    results["measured_reduction"] = meas
    # L3: results["thresholds"] has a mixed schema: most values are bool (per-check
    # pass/fail results), but the key "skipped_checks" maps to a dict[str, str]
    # (reason strings for checks that were configured but not performed).
    # Callers that want only the boolean check results should filter with:
    #   {k: v for k, v in results["thresholds"].items() if isinstance(v, bool)}
    results["thresholds"] = _check_thresholds(results, s6["thresholds"], s6_cfg=s6)

    path = artifacts_dir / "stage6_eval.json"

    # Free teacher GPU memory before llama-imatrix subprocess uses the GPU.
    if teacher is not None:
        try:
            teacher.to("cpu")
            del teacher
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Cache-HIT path — teacher was never loaded, so the student is the
        # only resident model. llama-imatrix will load the F16 GGUF onto the
        # same GPU; on a 35B-class model this would push GPU residency to
        # ~140 GB and risk OOM on H200. Move the student to CPU here so the
        # imatrix subprocess has the GPU to itself, mirroring the non-cache
        # path's free-before-imatrix discipline.
        try:
            model.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("Stage 6: starting post-eval imatrix pipeline")
    # Optimization #8: If GGUF conversion was running in background, wait for it.
    # Then run llama-imatrix (which needs the GPU, now freed from teacher).
    gguf_thread_timed_out = False
    imatrix_skipped = False
    if gguf_thread is not None:
        log.info("Stage 6: waiting for background GGUF conversion to complete")
        gguf_thread.join(timeout=3700)
        if gguf_thread.is_alive():
            # F-CR2-M-1: SKIP imatrix entirely when the bg thread is still alive after
            # the timeout. The daemon bg thread continues writing to model_f16.gguf.tmp
            # and would race with _generate_imatrix's sequential fallback, both of which
            # call os.replace on the same target path. By skipping, no concurrent writer
            # exists; the bg thread's eventual replace just updates the GGUF for the next
            # run. The prebuilt-only GGUF (without imatrix) remains acceptable for
            # downstream serving.
            log.error(
                "GGUF convert thread still alive after %.0f s timeout; SKIPPING imatrix "
                "entirely to avoid concurrent-writer race on model_f16.gguf",
                3700,
            )
            gguf_thread_timed_out = True
            imatrix_skipped = True
    f16_path = None if gguf_thread_timed_out else gguf_result.get("f16_path")
    if imatrix_skipped:
        # Sentinel: surface to dashboard via trackio. Do NOT call _generate_imatrix:
        # it would spawn a sequential GGUF write that races the still-live bg thread.
        _trackio_log({"stage6/imatrix_skipped": 1.0})
        results["imatrix_skipped"] = True
        # eval_text_concat.txt is an unconditional debug side-channel per spec
        # §9 — write it even on the skipped-imatrix path so the dashboard
        # has the captured prompts available for triage.
        try:
            _write_eval_text_concat(eval_text_concat, artifacts_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("imatrix-skipped path: eval_text_concat write failed (%s)", exc)
    elif cached_teacher_results is None and f16_path is not None:
        _run_llama_imatrix_with_prebuilt_gguf(
            eval_text_concat, s6.get("imatrix", {}), artifacts_dir, gguf_result,
        )
    else:
        # This else covers two sub-cases:
        #   (a) Teacher was cached — no background GGUF conversion was started, so
        #       gguf_result is empty and we fall through here. _generate_imatrix
        #       performs its own GGUF conversion sequentially if imatrix is enabled;
        #       if imatrix is disabled it returns immediately via its `enabled` guard.
        #   (b) Background GGUF conversion was started but failed/produced no output —
        #       cached_teacher_results is None but gguf_result has no f16_path.
        #       _generate_imatrix will retry the full GGUF + imatrix pipeline.
        # In both cases _generate_imatrix's internal `enabled` guard ensures we do
        # nothing unnecessary when imatrix is disabled in config.
        _generate_imatrix(eval_text_concat, s6.get("imatrix", {}), artifacts_dir)

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
    return path


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


