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

import hashlib
import importlib.metadata as _md
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
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

# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Constant — never override at call sites.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"

# F-C-C-1: Spec §9 — the eval-text concat (eval prompts seen by the model
# during PPL/zero-shot/generative) is captured to eval_text_concat.txt as a
# debugging side-channel ONLY. The imatrix calibration corpus filename
# (_IMATRIX_CALIB_FILENAME) is re-imported in the S6-2 block above.
_EVAL_TEXT_CONCAT_FILENAME: str = "eval_text_concat.txt"


# ---------------------------------------------------------------------------
# Teacher eval caching (Optimization #7)
# ---------------------------------------------------------------------------

# F-iter4-LOW-5: bump this whenever the cache file schema changes. _load_teacher_cache
# rejects (and triggers re-evaluation) when the on-disk version does not match.
TEACHER_CACHE_FORMAT_VERSION: int = 1


def _safe_pkg_version(name: str) -> str:
    """Return importlib.metadata.version(name) or 'unknown' if not installed.

    Avoids hard-failing the cache key when an optional package isn't installed
    (e.g. lm-eval missing in a smoke environment).
    """
    try:
        return _md.version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def _teacher_cache_key(config: dict) -> str:
    """Compute a deterministic SHA-256 cache key from the 9 spec-mandated components.

    Per spec F-S-H-3, the cache key MUST cover every input that can change the
    teacher's evaluation numbers, so a stale cache cannot mask a meaningful
    config change.

    Components (sorted-keys JSON, no whitespace):
      1. model_name              — config.model.name_or_path
      2. model_revision          — config.model.revision (default "main")
      3. tokenizer_revision      — config.model.tokenizer_revision (default model_revision)
      4. dataset_revisions       — canonical sorted-keys mapping from config
      5. lm_eval_version         — importlib.metadata.version("lm-eval")
      6. transformers_version    — importlib.metadata.version("transformers")
      7. dtype                   — config.model.torch_dtype
      8. attn_impl               — pinned to "eager" per F-S-M-1
      9. eval_config_subset      — wikitext2 + zero_shot + generative subdicts
    """
    s6 = config["stage6_validate"]
    model_cfg = config["model"]
    model_revision = model_cfg.get("revision") or "main"
    tokenizer_revision = model_cfg.get("tokenizer_revision") or model_revision
    dataset_revisions = _resolve_dataset_revisions(config)
    # F-iter4-NIT-3: explicitly canonicalize dataset_revisions to a sorted-keys
    # JSON string; do not rely solely on the outer json.dumps(..., sort_keys=)
    # for nested-dict canonicalization (sort_keys recurses but specifying it
    # explicitly here documents the contract — Spec §9 line 816 states the
    # mapping is "JSON-canonicalized ... with sorted keys before concatenation").
    dataset_revisions_canonical = json.dumps(
        dataset_revisions, sort_keys=True, separators=(",", ":"),
    )
    # F-iter4-HIGH-1: fold the lm-eval task list (and any per-task config we
    # configure here) into the cache key so the cache invalidates if the
    # lm-eval task set changes — even when we cannot pin per-dataset SHAs.
    lm_eval_task_config = {
        "tasks": list(s6.get("zero_shot", {}).get("tasks", [])),
        "lm_eval_batch_size": s6.get("lm_eval_batch_size"),
    }
    lm_eval_task_config_hash = hashlib.sha256(
        json.dumps(lm_eval_task_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "model_name": model_cfg["name_or_path"],
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "dataset_revisions_canonical": dataset_revisions_canonical,
        "lm_eval_version": _safe_pkg_version("lm-eval"),
        "lm_eval_task_config_hash": lm_eval_task_config_hash,
        "transformers_version": _safe_pkg_version("transformers"),
        "dtype": str(model_cfg.get("torch_dtype", "bfloat16")),
        "attn_impl": _STAGE6_ATTN_IMPLEMENTATION,
        "eval_config_subset": {
            "wikitext2": s6.get("wikitext2", {}),
            "zero_shot": s6.get("zero_shot", {}),
            "generative": s6.get("generative", {}),
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _load_teacher_cache(cache_path: Path, cache_key: str) -> dict | None:
    """Load cached teacher eval results if they exist and the key matches.

    Returns a dict with keys "results" and optionally "param_counts", or None.
    """
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        # F-iter4-LOW-5: reject mismatched schema versions so a stale cache
        # written by an older format never silently feeds wrong values.
        on_disk_version = data.get("format_version")
        if on_disk_version is not None and on_disk_version != TEACHER_CACHE_FORMAT_VERSION:
            log.warning(
                "Teacher cache format_version mismatch (expected %d, found %r) — "
                "re-evaluating.",
                TEACHER_CACHE_FORMAT_VERSION, on_disk_version,
            )
            return None
        if data.get("cache_key") != cache_key:
            log.info("Teacher cache key mismatch (expected %s, found %s) — re-evaluating.",
                     cache_key, data.get("cache_key"))
            return None
        # F-CR2-N-2: prefer .get() with explicit None check + a precise warning
        # message over relying on a broad except KeyError.
        teacher_results = data.get("teacher_results")
        if teacher_results is None:
            log.warning(
                "Teacher cache invalid: 'teacher_results' key missing from cache file %s "
                "— re-evaluating.",
                cache_path,
            )
            return None
        param_counts = data.get("teacher_param_counts")
        if param_counts is None:
            log.warning(
                "Teacher eval cache HIT (%s) but cache lacks 'teacher_param_counts' "
                "(legacy cache file). _measured_reduction will load the teacher model "
                "from scratch to count parameters — this may take several minutes.",
                cache_key,
            )
        else:
            log.info("Teacher eval cache HIT (%s) — skipping teacher load+eval entirely.", cache_key)
        return {
            "results": teacher_results,
            "param_counts": param_counts,
        }
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("Teacher cache corrupted (%s) — re-evaluating.", exc)
        return None


def _save_teacher_cache(
    cache_path: Path, cache_key: str, teacher_results: dict,
    *, teacher_param_counts: dict | None = None,
) -> None:
    """Save teacher eval results + param counts to cache file (atomic write)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cache_key": cache_key,
        "teacher_results": teacher_results,
    }
    if teacher_param_counts is not None:
        data["teacher_param_counts"] = teacher_param_counts
    # F-iter4-LOW-5: stamp a format_version so a future schema bump can be
    # detected at load time (see _load_teacher_cache).
    data["format_version"] = TEACHER_CACHE_FORMAT_VERSION
    # F-iter4-LOW-1: use the same `<file>.<ext>.tmp` convention as
    # _atomic_write_text (e.g. "teacher_eval_cache.json.tmp"); the previous
    # `.with_suffix(".tmp")` produced "teacher_eval_cache.tmp" which dropped
    # the .json extension and made temp files harder to identify.
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    # F-3: Split the try/except into two blocks so that a parent-dir fsync
    # failure after a successful os.replace does not cause a misleading re-raise.
    # After os.replace the cache file is durably on disk; the parent fsync is a
    # belt-and-suspenders flush and its failure should not invalidate the write.
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # F-CR2-N-1: open read-only solely to fsync — no bytes are written here.
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, cache_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    # Parent-dir fsync: best-effort only — file is already on disk after os.replace.
    try:
        parent_fd = os.open(str(cache_path.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception as _fsync_exc:
        log.debug(
            "_save_teacher_cache: parent-dir fsync failed (%s); "
            "cache file was already written by os.replace — continuing.",
            _fsync_exc,
        )
    log.info("Teacher eval cache saved → %s (key=%s)", cache_path, cache_key)


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
# Teacher preload (Optimization #6)
# ---------------------------------------------------------------------------

def _preload_teacher_to_cpu(config: dict, result_q: queue.Queue) -> None:
    """Load teacher model weights to CPU RAM in a background thread."""
    # H2: 4-bit quantisation requires CUDA; skip CPU preload to avoid guaranteed crash.
    if config.get("model", {}).get("load_in_4bit", False):
        log.warning(
            "_preload_teacher_to_cpu: skipping CPU preload because load_in_4bit=True requires CUDA"
        )
        return  # get_nowait() will return None → main thread does direct load
    try:
        # F-C-H-1: attn_implementation="eager" pinned per Spec F-S-M-1.
        log.info(
            "Teacher preload: loading %s to CPU (attn_implementation=%r forced per spec F-S-M-1)...",
            config["model"]["name_or_path"], _STAGE6_ATTN_IMPLEMENTATION,
        )
        t0 = time.monotonic()
        teacher, _ = load_model(
            config["model"]["name_or_path"],
            revision=config["model"].get("revision", "main"),
            torch_dtype=config["model"]["torch_dtype"],
            device_map="cpu",
            attn_implementation=_STAGE6_ATTN_IMPLEMENTATION,
            load_in_4bit=config["model"].get("load_in_4bit", False),
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )
        dt = time.monotonic() - t0
        # M4: Log "complete" before put_nowait so this message only fires when
        # the teacher was successfully loaded; if put_nowait fails the load was
        # still successful but the result won't be available to the main thread.
        log.info("Teacher preload complete in %.1fs (on CPU)", dt)
        try:
            result_q.put_nowait(teacher)
        except Exception as exc:
            log.debug("_preload_teacher_to_cpu: put_nowait failed (%s); main thread will load directly", exc)
    except Exception as exc:
        log.warning("Teacher preload failed (%s) — will fall back to direct load", exc)


# ---------------------------------------------------------------------------
# Background GGUF conversion (Optimization #8)
# ---------------------------------------------------------------------------

def _background_gguf_convert(icfg: dict, artifacts_dir: Path, result: dict) -> None:
    """Convert the student model to F16 GGUF in background (CPU-bound)."""
    if not icfg.get("enabled", True):
        return

    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
        log.warning("GGUF convert (background): llama.cpp not found — skipping.")
        return

    convert_py = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        log.warning("GGUF convert (background): convert script missing — skipping.")
        return

    model_dir = artifacts_dir / "stage5_final"
    if not model_dir.exists():
        log.warning("GGUF convert (background): stage5_final not found — skipping.")
        return

    free_gb = shutil.disk_usage(artifacts_dir).free / 1e9
    if free_gb < 40:
        log.warning("GGUF convert (background): only %.1f GB free — skipping.", free_gb)
        return

    f16_path = artifacts_dir / "model_f16.gguf"
    f16_tmp = artifacts_dir / "model_f16.gguf.tmp"
    log.info("GGUF convert (background): %s → F16 GGUF", model_dir)
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": str(llama_cpp_dir / "build" / "bin") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    try:
        t0 = time.monotonic()
        stderr_log = artifacts_dir / "gguf_convert_stderr.log"
        with open(stderr_log, "w") as _stderr_fh:
            subprocess.run(
                [sys.executable, str(convert_py), str(model_dir),
                 "--outtype", "f16", "--outfile", str(f16_tmp)],
                env=env, check=True, timeout=3600, stderr=_stderr_fh,
            )
        os.replace(f16_tmp, f16_path)
        dt = time.monotonic() - t0
        result["f16_path"] = f16_path
        log.info("GGUF convert (background): done in %.1fs (%.1f GB)",
                 dt, f16_path.stat().st_size / 1e9)
    except subprocess.TimeoutExpired as exc:
        log.error("GGUF convert (background): timed out after 3600s (%s)", exc)
        f16_tmp.unlink(missing_ok=True)
        return
    except subprocess.CalledProcessError as exc:
        stderr_snippet = ""
        if stderr_log.exists():
            try:
                stderr_snippet = stderr_log.read_text(errors="replace")[-2000:]
            except Exception:
                pass
        log.warning("GGUF convert (background): failed (%s): %s", exc, stderr_snippet)
        f16_tmp.unlink(missing_ok=True)
        return
    except Exception as exc:
        log.warning("GGUF convert (background): failed (%s)", exc)
        f16_tmp.unlink(missing_ok=True)
        return


def _write_eval_text_concat(texts: list[str], artifacts_dir: Path) -> Path:
    """Write the eval-text concat (debug side-channel) atomically.

    F-C-C-1: this file is the concatenation of every prompt seen by the model
    during PPL/zero-shot/generative evals. It is NOT the imatrix calibration
    corpus — that is `calibration_wiki_train.txt` per Spec §9.
    """
    path = artifacts_dir / _EVAL_TEXT_CONCAT_FILENAME
    joined = "\n\n".join(t.strip() for t in texts if t and t.strip())
    _atomic_write_text(path, joined)
    log.info(
        "eval_text_concat: %d docs (%d chars) → %s (debug only — not used by imatrix)",
        len(texts), len(joined), path,
    )
    return path


def _run_llama_imatrix_with_prebuilt_gguf(
    eval_text_concat: list[str], icfg: dict, artifacts_dir: Path, gguf_result: dict,
) -> None:
    """Run llama-imatrix using the pre-built F16 GGUF from background thread.

    F-C-C-1: imatrix calibration corpus is the WikiText-2 *train* split
    (calibration_wiki_train.txt). The eval_text_concat list captured during
    PPL/zero-shot/generative evals is written as a debugging side-channel
    (eval_text_concat.txt) but is NOT used by llama-imatrix.
    """
    # Always write eval_text_concat.txt as a debug artifact.
    _write_eval_text_concat(eval_text_concat, artifacts_dir)

    if not icfg.get("enabled", True):
        return

    # F-C-C-1: imatrix calibration source is the WikiText-2 *train* split.
    calib_path = artifacts_dir / _IMATRIX_CALIB_FILENAME
    if not calib_path.exists() or calib_path.stat().st_size == 0:
        log.warning(
            "imatrix: calibration corpus %s missing/empty; skipping imatrix generation. "
            "Spec §9 requires this file (WikiText-2 train split). "
            "It is built automatically at the top of run() — check the earlier warning "
            "from _build_imatrix_calibration_corpus.",
            calib_path,
        )
        return

    f16_path = gguf_result.get("f16_path")
    if f16_path is None or not f16_path.exists():
        log.warning("imatrix: pre-built GGUF not available — falling back to full pipeline")
        _generate_imatrix(eval_text_concat, icfg, artifacts_dir)
        return

    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
        log.warning("llama_cpp_dir not found; skipping imatrix generation via prebuilt GGUF")
        return

    imatrix_bin = llama_cpp_dir / "build" / "bin" / "llama-imatrix"
    if not imatrix_bin.exists():
        log.warning("imatrix: llama-imatrix binary not found — skipping.")
        return

    imatrix_out = artifacts_dir / "imatrix.gguf"
    ngl = int(icfg.get("ngl", 99))
    ctx = int(icfg.get("ctx_size", 2048))
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": str(llama_cpp_dir / "build" / "bin") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    log.info("imatrix: running llama-imatrix (ngl=%d, ctx=%d) → %s", ngl, ctx, imatrix_out)
    imatrix_stderr_log = artifacts_dir / "llama_imatrix_stderr.log"
    try:
        with open(imatrix_stderr_log, "w") as _stderr_fh:
            subprocess.run(
                [str(imatrix_bin),
                 "-m", str(f16_path), "-f", str(calib_path),
                 "-o", str(imatrix_out), "--output-format", "gguf",
                 "--no-ppl", "-ngl", str(ngl), "-c", str(ctx)],
                env=env, check=True, timeout=7200, stderr=_stderr_fh,
            )
        log.info("imatrix: saved (%.1f MB)", imatrix_out.stat().st_size / 1e6)
    except subprocess.TimeoutExpired as exc:
        log.warning("imatrix subprocess timed out after %ss; skipping imatrix", exc.timeout)
        return
    except subprocess.CalledProcessError as exc:
        stderr_snippet = ""
        if imatrix_stderr_log.exists():
            try:
                stderr_snippet = imatrix_stderr_log.read_text(errors="replace")[-2000:]
            except Exception:
                pass
        log.warning("imatrix: llama-imatrix failed (%s): %s. Calibration text at %s.",
                    exc, stderr_snippet, calib_path)
    except Exception as exc:
        log.warning("imatrix: llama-imatrix failed (%s). Calibration text at %s.", exc, calib_path)


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


# ---------------------------------------------------------------------------
# imatrix calibration + GGUF conversion (full sequential path)
# ---------------------------------------------------------------------------


def _generate_imatrix(eval_text_concat: list[str], icfg: dict, artifacts_dir: Path) -> None:
    """Sequential GGUF + llama-imatrix fallback path.

    F-C-C-1: imatrix calibration corpus is `calibration_wiki_train.txt` (the
    WikiText-2 train split written at the top of run()). The eval_text_concat
    list captured during evals is written as a debug-only side-channel.
    """
    # Always write the eval-text concat as a debug artifact, regardless of imatrix enable.
    _write_eval_text_concat(eval_text_concat, artifacts_dir)

    if not icfg.get("enabled", True):
        log.info("imatrix: disabled via config.")
        return

    calib_path = artifacts_dir / _IMATRIX_CALIB_FILENAME
    if not calib_path.exists() or calib_path.stat().st_size == 0:
        log.warning(
            "imatrix: calibration corpus %s missing/empty; skipping imatrix generation. "
            "Spec §9 requires this file (WikiText-2 train split).",
            calib_path,
        )
        return

    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
        log.warning("imatrix: llama.cpp not found; skipping imatrix generation.")
        return

    imatrix_bin = llama_cpp_dir / "build" / "bin" / "llama-imatrix"
    convert_py  = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not imatrix_bin.exists() or not convert_py.exists():
        log.warning("imatrix: binaries missing under %s; skipping.", llama_cpp_dir)
        return

    model_dir = artifacts_dir / "stage5_final"
    if not model_dir.exists():
        log.warning("imatrix: stage5_final not found at %s; skipping.", model_dir)
        return

    free_gb = shutil.disk_usage(artifacts_dir).free / 1e9
    if free_gb < 40:
        log.warning("imatrix: only %.1f GB free; skipping GGUF conversion.", free_gb)
        return

    f16_path = artifacts_dir / "model_f16.gguf"
    f16_tmp = artifacts_dir / "model_f16.gguf.tmp"
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": str(llama_cpp_dir / "build" / "bin") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    log.info("imatrix: converting %s → F16 GGUF", model_dir)
    stderr_log = artifacts_dir / "gguf_convert_stderr.log"
    try:
        f16_tmp.unlink(missing_ok=True)
        with open(stderr_log, "w") as stderr_fh:
            subprocess.run(
                [sys.executable, str(convert_py), str(model_dir),
                 "--outtype", "f16", "--outfile", str(f16_tmp)],
                env=env, check=True, timeout=3600, stderr=stderr_fh,
            )
        os.replace(f16_tmp, f16_path)
        log.info("imatrix: GGUF ready (%.1f GB)", f16_path.stat().st_size / 1e9)
    except subprocess.TimeoutExpired as exc:
        f16_tmp.unlink(missing_ok=True)
        log.warning("imatrix: GGUF conversion timed out after %ss: %s; skipping.", exc.timeout, exc)
        return
    except subprocess.CalledProcessError as exc:
        f16_tmp.unlink(missing_ok=True)
        try:
            tail = stderr_log.read_text()[-2000:]
        except Exception:
            tail = ""
        log.warning("imatrix: GGUF conversion failed (%s): %s; skipping.", exc, tail)
        return
    except Exception as exc:  # noqa: BLE001
        f16_tmp.unlink(missing_ok=True)
        log.warning("imatrix: GGUF conversion failed (%s); skipping.", exc)
        return

    imatrix_out = artifacts_dir / "imatrix.gguf"
    ngl = int(icfg.get("ngl", 99))
    ctx = int(icfg.get("ctx_size", 2048))
    log.info("imatrix: running llama-imatrix (ngl=%d, ctx=%d) → %s", ngl, ctx, imatrix_out)
    imatrix_stderr_log = artifacts_dir / "llama_imatrix_stderr.log"
    try:
        with open(imatrix_stderr_log, "w") as stderr_fh:
            subprocess.run(
                [str(imatrix_bin),
                 "-m", str(f16_path), "-f", str(calib_path),
                 "-o", str(imatrix_out), "--output-format", "gguf",
                 "--no-ppl", "-ngl", str(ngl), "-c", str(ctx)],
                env=env, check=True, timeout=7200, stderr=stderr_fh,
            )
        log.info("imatrix: saved (%.1f MB)", imatrix_out.stat().st_size / 1e6)
    except subprocess.TimeoutExpired as exc:
        log.error("imatrix: llama-imatrix timed out after 7200s (%s). Calibration text at %s.", exc, calib_path)
        return
    except subprocess.CalledProcessError as exc:
        try:
            tail = imatrix_stderr_log.read_text()[-2000:]
        except Exception:
            tail = ""
        log.warning("imatrix: llama-imatrix failed (%s): %s. Calibration text at %s.", exc, tail, calib_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("imatrix: llama-imatrix failed (%s). Calibration text at %s.", exc, calib_path)


def _find_llama_cpp_dir(override: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    on_path = shutil.which("llama-imatrix")
    if on_path:
        candidates.append(Path(on_path).parent.parent.parent)
    # No further fallback beyond the three candidates above (config, env var, PATH search).

    for p in candidates:
        if (p / "build" / "bin" / "llama-imatrix").exists() and (p / "convert_hf_to_gguf.py").exists():
            return p
    return None


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
