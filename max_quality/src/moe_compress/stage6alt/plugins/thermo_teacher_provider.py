"""Thermometer teacher-cache provider — Stage 6alt teacher-side eval (live).

Paper / spec source
--------------------
No upstream paper for the thermometer teacher provider per se; this
plugin owns the Stage 6alt thermometer teacher-side concern:

- **Sweep-shared cache**: the thermometer runs many ablation rows against
  the SAME uncompressed teacher on the SAME fixed corpus, so the
  teacher's BPT + zero-shot-subset + per-token argmax are constant
  across rows and computed ONCE per sweep. The first row evaluates the
  teacher; subsequent rows take the HIT path and never reload it.
- **Cache-key invariant**: SHA-256 over an 8-component payload —
  ``format_version`` / ``corpus_id`` / ``teacher_repo`` /
  ``teacher_revision`` / ``torch_dtype`` / ``attn_implementation`` /
  ``arc_easy_limit`` / ``hellaswag_limit``. ``teacher_revision`` is a
  revision *name string* (default "main"), NOT a content SHA — pin the
  revision to a commit SHA in YAML if you need full content sensitivity
  (same revision-vs-SHA caveat as the Stage 6 sibling). The tokenizer
  name is folded in indirectly via ``corpus_id`` (``thermo_corpus.py``
  L216-228 hashes ``tok_id`` into the corpus id), so a tokenizer change
  invalidates the cache transitively through the corpus.
- **Cache scope-of-validity**: thermometer caches are scoped to a SINGLE
  SWEEP (a short-lived host process). The cache key intentionally OMITS
  ``lm_eval_version`` / ``transformers_version`` /
  ``dataset_revisions_canonical`` for arc_easy/hellaswag — a sweep does
  not survive long enough to encounter an lm-eval or HF-dataset bump,
  and the on-disk file is regenerated on the next sweep boot. Do NOT
  reuse a thermometer cache file across host processes whose lm-eval /
  HF-datasets / transformers versions differ; either delete the file or
  bump ``THERMO_TEACHER_CACHE_FORMAT_VERSION``.
- **Sidecar to the Stage 6 cache**: stored as a separate file with a
  separate key namespace; the thermometer cache cannot collide with
  ``teacher_eval_cache``.

Live wiring
-----------
``ThermoTeacherProviderPlugin`` is the live Stage 6alt thermometer
teacher-side entry point. ``stage6alt.orchestrator.run()`` registers it
(orchestrator L121-128) and invokes
``walk_phases(("provide_thermo_teacher_side",), plugins, run_ctx)``
(orchestrator L166). The legacy ``stage6alt_thermometer.run()`` is now
a 2-line shim delegating into the orchestrator. Tests invoke this hook
directly via
``ThermoTeacherProviderPlugin().provide_thermo_teacher_side(ctx)``
(mirror of the pattern documented in
``stage6/plugins/teacher_provider.py:30-35``).

The standalone helpers (``THERMO_TEACHER_CACHE_FORMAT_VERSION`` /
``_thermo_teacher_cache_key`` / ``_load_thermo_teacher_cache`` /
``_save_thermo_teacher_cache``) are re-exported through the monolith
namespace (``stage6alt_thermometer.py:84-90``) so tests that
monkey-patch ``stage6alt_thermometer._load_thermo_teacher_cache`` (etc.)
keep biting — the re-import puts the SAME function object on the
monolith namespace.

Unlike the Stage 6 ``TeacherProviderPlugin``, the thermometer teacher
HAS NO PRELOAD THREAD and HAS NO PARAM-COUNT TRACKING: the load is
direct (after a guarded student-to-CPU swap), and the cached payload is
the raw ``teacher_results`` dict (``teacher_bpt`` / ``teacher_argmax`` /
``teacher_arc_easy_acc_norm`` / ``teacher_hellaswag_acc_norm`` /
``teacher_acc_norm_sum``) — NOT a wrapper carrying ``{"results": ...,
"param_counts": ...}`` like Stage 6. ``_load_thermo_teacher_cache``
returns that raw dict directly on a HIT (or ``None`` on miss / mismatch
/ malformed-payload rejection).

Circular-import contract (mirror of ``stage6alt/plugins/bpt_metric.py``):
this module imports only from ``..context`` / ``...stage6.plugins.eval_environment``
/ ``...utils.model_io`` / sibling plugin modules (``bpt_metric``,
``zero_shot_subset``) / stdlib / torch — NEVER from
``stage6alt_thermometer`` or ``stage6alt.orchestrator`` at any scope
(module-top OR function-local). The monolith re-imports *this* module's
symbols at load time, so a ``from ..stage6alt_thermometer import ...``
here would deadlock the import; nothing in this module does that.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from ..context import PipelineContext
from ...stage6.plugins.eval_environment import (
    _apply_stage6_kernel_patches,
    _set_experts_implementation_s6,
)
from ...utils.model_io import load_model
from .bpt_metric import _bpt_from_nll
from .zero_shot_subset import _lm_eval_subset

log = logging.getLogger(__name__)


# Module-local copy of the Stage 6 eager-attention contract. Mirror of the
# relocation discipline used by ``stage6alt/plugins/bpt_metric.py``: each
# module carries its own copy of the constant rather than chaining the
# import from ``stage6_validate`` (which would couple the import graph).
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"


# ---------------------------------------------------------------------------
# Teacher BPT cache (sweep-shared) — format-version constant + helpers
# ---------------------------------------------------------------------------


# Bump on any change to the thermometer_teacher_cache.json schema.
# v2: added `teacher_argmax` (per-token predictions for the top1_agreement
#     metric) and switched the cache key from a Nemotron-only spec hash to a
#     corpus-agnostic `corpus_id` (so wikitext/nemotron results never collide).
THERMO_TEACHER_CACHE_FORMAT_VERSION = 2


def _thermo_teacher_cache_key(config: dict, corpus_id: str) -> str:
    """SHA-256 over the 8-component payload that scopes the sweep cache.

    Components (sorted-keys JSON):

      1. format_version       — ``THERMO_TEACHER_CACHE_FORMAT_VERSION``
      2. corpus_id            — from ``_build_thermo_corpus`` (folds corpus
                                kind / size / seed-spec / dataset revisions /
                                tokenizer ``tok_id`` together; this is how
                                tokenizer churn invalidates the cache)
      3. teacher_repo         — config.model.name_or_path
      4. teacher_revision     — config.model.revision (default "main"; a
                                revision *name string*, not a content SHA —
                                same caveat as the Stage 6 sibling)
      5. torch_dtype          — config.model.torch_dtype (default bfloat16)
      6. attn_implementation  — pinned to "eager" per Spec F-S-M-1
      7. arc_easy_limit       — thermometer.arc_easy_limit (default 100)
      8. hellaswag_limit      — thermometer.hellaswag_limit (default 200)

    Scope-of-validity boundary
    --------------------------
    Intentionally OMITTED relative to the Stage 6 sibling:
    ``lm_eval_version``, ``transformers_version``, and
    ``dataset_revisions_canonical`` for arc_easy/hellaswag. The thermometer
    cache is scoped to a SINGLE SWEEP (short-lived host process) — a sweep
    does not survive across an lm-eval / transformers / HF-dataset bump,
    so folding those into the key would be wasted work. Do NOT carry a
    thermometer cache file across host processes whose lm-eval /
    HF-datasets / transformers versions differ; either delete the file or
    bump ``THERMO_TEACHER_CACHE_FORMAT_VERSION``.
    """
    therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
    model_cfg = config["model"]
    payload = json.dumps({
        "format_version": THERMO_TEACHER_CACHE_FORMAT_VERSION,
        "corpus_id": corpus_id,
        "teacher_repo": model_cfg["name_or_path"],
        "teacher_revision": model_cfg.get("revision") or "main",
        "torch_dtype": str(model_cfg.get("torch_dtype", "bfloat16")),
        "attn_implementation": _STAGE6_ATTN_IMPLEMENTATION,
        "arc_easy_limit": int(therm.get("arc_easy_limit", 100)),
        "hellaswag_limit": int(therm.get("hellaswag_limit", 200)),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_thermo_teacher_cache(cache_path: Path, cache_key: str) -> dict | None:
    """Return cached teacher_results on a key+version+schema match, else None.

    Schema validation: a HIT is only returned when the cached
    ``teacher_results`` dict carries the four scalar keys the cache-hit
    branch of ``provide_thermo_teacher_side`` publishes to ctx without
    re-deriving (``teacher_bpt`` / ``teacher_argmax`` /
    ``teacher_arc_easy_acc_norm`` / ``teacher_hellaswag_acc_norm``).
    Missing-key payloads are rejected here (mirror of the Stage 6 sibling's
    ``teacher_provider.py`` L188-197 pattern) so the hit path can use
    required-key access and a malformed cache cannot publish ``None`` to
    downstream ctx slots.
    """
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:                   # noqa: BLE001
        log.warning("stage6alt: teacher cache unreadable (%s); recomputing", exc)
        return None
    if data.get("format_version") != THERMO_TEACHER_CACHE_FORMAT_VERSION:
        log.info("stage6alt: teacher cache format_version mismatch; recomputing")
        return None
    if data.get("cache_key") != cache_key:
        log.info("stage6alt: teacher cache key mismatch; recomputing")
        return None
    teacher_results = data.get("teacher_results")
    if not isinstance(teacher_results, dict):
        log.warning(
            "stage6alt: teacher cache invalid: 'teacher_results' missing or "
            "non-dict in %s — recomputing.", cache_path,
        )
        return None
    # F-iter1-L3: required-key schema check — any of the four scalars below
    # being absent means the cache-hit branch would publish a None to ctx
    # and break downstream bpt_gap / acc_norm_sum / top1_agreement with a
    # confusing AttributeError far from the cache. Reject here instead.
    required_keys = (
        "teacher_bpt",
        "teacher_argmax",
        "teacher_arc_easy_acc_norm",
        "teacher_hellaswag_acc_norm",
    )
    missing = [k for k in required_keys if k not in teacher_results]
    if missing:
        log.warning(
            "stage6alt: teacher cache invalid: missing keys %s in %s — "
            "recomputing.", missing, cache_path,
        )
        return None
    return teacher_results


def _save_thermo_teacher_cache(cache_path: Path, cache_key: str,
                               teacher_results: dict) -> None:
    """Atomic write of teacher_results to the sweep-shared cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format_version": THERMO_TEACHER_CACHE_FORMAT_VERSION,
        "cache_key": cache_key,
        "teacher_results": teacher_results,
    }
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, cache_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log.info("stage6alt: teacher BPT cache saved -> %s (key=%s)", cache_path, cache_key)


# ---------------------------------------------------------------------------
# Plugin — live entry point for the thermometer teacher-side phase
# ---------------------------------------------------------------------------


class ThermoTeacherProviderPlugin:
    """Stage 6alt thermometer teacher-provider plugin (live).

    Owns the Stage 6alt thermometer teacher-side concern: the cache-key /
    load / save helpers plus the ``provide_thermo_teacher_side`` hook —
    the cache-hit shortcut, the cache-miss student-to-CPU swap, the
    direct teacher ``load_model`` (attn pinned to ``eager`` per Spec
    F-S-M-1), the cu130/Hopper kernel patches + experts-impl shim, the
    ``_bpt_from_nll`` + ``_lm_eval_subset`` calls, the cache save, and the
    teacher-free + student-restore path.

    The Stage 6alt orchestrator invokes this hook via
    ``walk_phases(("provide_thermo_teacher_side",), plugins, run_ctx)``
    at ``stage6alt/orchestrator.py`` L166.
    """

    name = "thermo_teacher_provider"
    paper = "Stage 6alt thermometer teacher-cache provider — SHA-256-keyed per-sweep cache (project-original sweep harness). See module docstring."
    config_key = "stage6_validate.thermometer"
    reads: tuple[str, ...] = (
        "config", "artifacts_dir", "tokenizer", "calib_ids", "corpus_id",
        "device", "experts_impl", "model",
    )
    writes: tuple[str, ...] = (
        "teacher_results",
        "teacher_bpt",
        "teacher_argmax",
        "teacher_cache_hit",
        "teacher_cache_path",
        "teacher_cache_key",
    )
    # No calibration-pass accumulator — the teacher block is a forward-pass-
    # only metric path (BPT + lm-eval harness) that consumes the corpus
    # tensor ``ThermoCorpusPlugin`` already built. The thermometer teacher
    # results are written wholesale to ctx via ``writes``; nothing here
    # needs a `provides` accumulator.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — every thermometer run must produce teacher results.

        ``config_key`` only names the thermometer config sub-tree (the
        ``teacher_cache_path`` / ``arc_easy_limit`` / ``hellaswag_limit`` /
        ``bpt_batch_size`` / ``lm_eval_batch_size`` knobs live there); it
        never gates the plugin as a whole. The hook itself contains the
        internal cache-hit shortcut.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def provide_thermo_teacher_side(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6alt thermometer teacher-side eval (live).

        Dispatched by the Stage 6alt orchestrator via
        ``walk_phases(("provide_thermo_teacher_side",), plugins, run_ctx)``
        (orchestrator L166). Executes, in order:

        1. **Cache key + path** — ``cache_path`` from ``therm[
           "teacher_cache_path"]`` (falling back to ``artifacts_dir /
           "thermometer_teacher_cache.json"``); ``cache_key`` from
           ``_thermo_teacher_cache_key(config, corpus_id)``.
        2. **Cache-hit shortcut** — ``_load_thermo_teacher_cache(cache_path,
           cache_key)``; HIT publishes ``teacher_results`` /
           ``teacher_bpt`` / ``teacher_argmax`` /
           ``teacher_cache_hit=True`` / ``teacher_cache_path`` /
           ``teacher_cache_key`` to ctx and returns. Required-key access on
           the hit path is safe because ``_load_thermo_teacher_cache``
           rejects malformed payloads up front (schema validation L1).
        3. **Cache miss** — guard ``model.to("cpu")`` + empty CUDA cache,
           ``load_model(...)`` with ``attn_implementation="eager"``,
           ``teacher.train(False)``, ``_set_experts_implementation_s6`` +
           ``_apply_stage6_kernel_patches`` (role="teacher"), then
           ``_bpt_from_nll(..., collect_argmax=True)`` and
           ``_lm_eval_subset(...)``; assemble ``teacher_results`` with
           the five keys (``teacher_bpt`` / ``teacher_arc_easy_acc_norm``
           / ``teacher_hellaswag_acc_norm`` / ``teacher_acc_norm_sum`` /
           ``teacher_argmax`` as ``.tolist()`` or None);
           ``_save_thermo_teacher_cache`` then ``del teacher`` + empty
           CUDA cache + guarded ``model.to(device or "cuda")``.
        4. **ctx writes** — ``teacher_results`` /
           ``teacher_bpt`` / ``teacher_argmax`` / ``teacher_cache_hit`` /
           ``teacher_cache_path`` / ``teacher_cache_key``.

        ``load_model`` call leniency
        -----------------------------
        Unlike the Stage 6 sibling (``teacher_provider.py`` L482-490) which
        requires ``config["model"]["device_map"]`` and
        ``config["model"]["torch_dtype"]`` to be set, this hook uses
        ``model_cfg.get(...)`` with defaults for ``device_map`` /
        ``torch_dtype`` / ``trust_remote_code`` / ``load_in_4bit``. The
        thermometer is run by sweep harnesses that may set up a minimal
        ``model`` sub-tree (``name_or_path`` only), so leniency here is
        intentional — the defaults match what ``load_model`` would resolve
        on a fresh import.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise.
        config = ctx.get("config")
        tokenizer = ctx.get("tokenizer")
        calib_ids = ctx.get("calib_ids")
        artifacts_dir = ctx.get("artifacts_dir")
        corpus_id = ctx.get("corpus_id")

        # Optional side-channels (analogues of monolith run()'s locals).
        device = ctx.get("device") if ctx.has("device") else None
        s6 = config["stage6_validate"]
        therm = s6.get("thermometer", {}) or {}
        bpt_batch = int(therm.get("bpt_batch_size", 8))
        lm_batch = therm.get("lm_eval_batch_size", "auto:8")
        arc_limit = int(therm.get("arc_easy_limit", 100))
        hsw_limit = int(therm.get("hellaswag_limit", 200))

        # Resolve experts_impl matching the monolith run()'s top-of-block
        # logic: ctx (set by ThermoEnvironmentPlugin) wins; otherwise mirror
        # env-var-first → config → "batched_mm" default so the value is
        # NEVER None (matches the monolith, which calls
        # _set_experts_implementation_s6(...) unconditionally with this).
        if ctx.has("experts_impl"):
            experts_impl = ctx.get("experts_impl")
        else:
            experts_impl = os.environ.get(
                "EXPERTS_IMPLEMENTATION",
                s6.get("experts_implementation", "batched_mm"),
            )

        # Step 1: resolve cache locals exactly as the monolith run() does.
        cache_path = Path(
            therm.get("teacher_cache_path")
            or str(artifacts_dir / "thermometer_teacher_cache.json")
        )
        cache_key = _thermo_teacher_cache_key(config, corpus_id)

        # Step 2: cache-hit shortcut. Mirrors the monolith's
        #   teacher_results = _load_thermo_teacher_cache(cache_path, cache_key)
        #   teacher_cache_hit = teacher_results is not None
        # branch that skips the teacher load + eval entirely.
        teacher_results = _load_thermo_teacher_cache(cache_path, cache_key)
        teacher_cache_hit = teacher_results is not None
        if teacher_cache_hit:
            log.info(
                "Stage 6alt: teacher BPT cache HIT (%s) — skipping teacher load",
                cache_path,
            )
            ctx.set("teacher_results", teacher_results)
            # F-iter1-L3: required-key access — _load_thermo_teacher_cache
            # has already validated that the cached payload carries
            # 'teacher_bpt' / 'teacher_argmax' (rejecting on missing). Using
            # subscript here mirrors the cache-miss branch and prevents a
            # silent None publish if validation ever loosens.
            ctx.set("teacher_bpt", teacher_results["teacher_bpt"])
            ctx.set("teacher_argmax", teacher_results["teacher_argmax"])
            ctx.set("teacher_cache_hit", True)
            ctx.set("teacher_cache_path", cache_path)
            ctx.set("teacher_cache_key", cache_key)
            return

        # Step 3: cache-miss path. Mirrors the monolith run() teacher load
        # block exactly.
        log.info("Stage 6alt: teacher BPT cache miss — loading teacher")
        model = ctx.get("model") if ctx.has("model") else None
        # Free the student's GPU memory before the teacher comes on. Guarded:
        # an accelerate-sharded student (device_map="auto" spanning >1 GPU)
        # raises NotImplementedError on .to() — mirror stage6_validate.py's
        # teacher-swap block, which logs and continues rather than aborting.
        if model is not None:
            try:
                model.to("cpu")
            except Exception as exc:           # noqa: BLE001
                log.warning("Stage 6alt: could not move student to CPU before "
                            "teacher load (%s) — teacher may OOM on a multi-GPU "
                            "host; single-GPU runs are unaffected.", exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_cfg = config["model"]
        teacher, _ = load_model(
            model_cfg["name_or_path"],
            revision=model_cfg.get("revision") or "main",
            torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
            device_map=model_cfg.get("device_map", "auto"),
            # eager is mandatory for batch-invariant NLL — do NOT inherit
            # model_cfg["attn_implementation"] (typically "sdpa").
            attn_implementation=_STAGE6_ATTN_IMPLEMENTATION,
            load_in_4bit=model_cfg.get("load_in_4bit", False),
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        teacher.train(False)
        # Same cu130/Hopper segfault-fix patches as the student — mirrors
        # stage6_validate.run()'s teacher-side patch calls.
        _set_experts_implementation_s6(teacher, experts_impl)
        _apply_stage6_kernel_patches(teacher, role="teacher")
        teacher_bpt, teacher_argmax = _bpt_from_nll(
            teacher, calib_ids, device=device, batch_size=bpt_batch,
            collect_argmax=True,
        )
        teacher_lm = _lm_eval_subset(
            teacher, tokenizer, arc_limit=arc_limit,
            hellaswag_limit=hsw_limit, batch_size=lm_batch,
        )
        teacher_results = {
            "teacher_bpt": teacher_bpt,
            "teacher_arc_easy_acc_norm": teacher_lm["arc_easy_acc_norm"],
            "teacher_hellaswag_acc_norm": teacher_lm["hellaswag_acc_norm"],
            "teacher_acc_norm_sum": teacher_lm["acc_norm_sum"],
            # Per-token argmax for top1_agreement, cached so cache-hit sweep
            # rows still score agreement against the (constant) teacher.
            # None when BPT skipped a batch (partial corpus).
            "teacher_argmax": (teacher_argmax.tolist()
                               if teacher_argmax is not None else None),
        }
        _save_thermo_teacher_cache(cache_path, cache_key, teacher_results)
        # Free the teacher and restore the student to GPU. Guarded for the
        # same multi-GPU-shard reason as the .to("cpu") above.
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if model is not None:
            try:
                model.to(device or "cuda")
            except Exception as exc:           # noqa: BLE001
                log.warning("Stage 6alt: could not restore student to GPU "
                            "(%s) — harmless, the pipeline ends after Stage 6.", exc)

        # Step 4: publish ctx slots.
        ctx.set("teacher_results", teacher_results)
        ctx.set("teacher_bpt", teacher_results["teacher_bpt"])
        ctx.set("teacher_argmax", teacher_results["teacher_argmax"])
        ctx.set("teacher_cache_hit", False)
        ctx.set("teacher_cache_path", cache_path)
        ctx.set("teacher_cache_key", cache_key)


__all__ = [
    "THERMO_TEACHER_CACHE_FORMAT_VERSION",
    "_thermo_teacher_cache_key",
    "_load_thermo_teacher_cache",
    "_save_thermo_teacher_cache",
    "ThermoTeacherProviderPlugin",
]
