"""Thermometer teacher-cache provider (S6A-4 of the Stage 6alt plugin-architecture refactor).

Paper / spec source
--------------------
Sweep-cache mirror of :mod:`stage6.plugins.teacher_provider`. The
thermometer runs many ablation rows against the SAME uncompressed
teacher on the SAME fixed corpus, so the teacher's BPT +
zero-shot-subset + per-token argmax are constant across rows and
computed ONCE per sweep.

Cache invariant: SHA-256 key over teacher SHA, corpus selector, lm-eval
task config, and tokenizer SHA. Stored as a sidecar parallel to the
Stage 6 ``teacher_eval_cache`` (separate key namespace so the thermometer
cache cannot collide with the Stage 6 cache).

Project-original sweep harness; no upstream paper.

Home of the Stage 6alt thermometer teacher-side concern, extracted from
the legacy ``stage6alt_thermometer.py`` monolith. The thermometer's
teacher BPT + lm-eval subset + per-token argmax are constant across all
12 ablation rows (same uncompressed model, same fixed corpus), so the
teacher results are computed ONCE per sweep and cached on disk under a
``corpus_id``-keyed SHA-256. Sweep-shared cache: the first row evaluates
the teacher; rows 2..12 take the HIT path and never reload it.

Pattern A vs Pattern B
----------------------
S6A-4's teacher-cache slice covers a MIXED pattern:

* **Pattern A — relocated verbatim**: ``THERMO_TEACHER_CACHE_FORMAT_VERSION``,
  ``_thermo_teacher_cache_key``, ``_load_thermo_teacher_cache``, and
  ``_save_thermo_teacher_cache`` below are character-identical copies of
  the monolith bodies. ``stage6alt_thermometer.py`` re-imports them (a
  ``# noqa: F401`` block) so ``run()`` and any external caller / test
  that monkey-patches ``stage6alt_thermometer._load_thermo_teacher_cache``
  (etc.) keeps working unchanged — the re-import puts the SAME function
  object on the monolith namespace.
* **Pattern B — reproduced in an inert hook**: the monolith ``run()``'s
  teacher-side block (cache-hit shortcut + cache-miss
  CPU-swap → ``load_model`` → kernel patches → ``_bpt_from_nll`` →
  ``_lm_eval_subset`` → cache save → student-restore path) is INLINE
  ``run()`` code in the monolith — there is nothing standalone to
  relocate. The ``provide_thermo_teacher_side`` hook below REPRODUCES
  that inline block faithfully; the monolith ``run()`` is NOT modified
  for it. This is an intentional, temporary logic duplication that
  resolves at S6A-6 when the orchestrator flip wires this hook live and
  the monolith ``run()`` becomes a thin shim.

Unlike S6-5's ``TeacherProviderPlugin``, the thermometer teacher-cache
HAS NO PRELOAD THREAD and HAS NO PARAM-COUNT TRACKING: the load is
direct (after a guarded student-to-CPU swap), and the cached payload is
the raw ``teacher_results`` dict (``teacher_bpt`` / ``teacher_argmax`` /
``teacher_arc_easy_acc_norm`` / ``teacher_hellaswag_acc_norm`` /
``teacher_acc_norm_sum``) — NOT a wrapper carrying ``{"results": ...,
"param_counts": ...}`` like S6-5. ``_load_thermo_teacher_cache`` returns
that raw dict directly on a HIT (or ``None`` on miss / mismatch).

Circular-import contract (mirror of ``stage6alt/plugins/bpt_metric.py``):
this module imports only from ``..context`` / ``...stage6.plugins.eval_environment``
/ ``...utils.model_io`` / sibling plugin modules (``bpt_metric``,
``zero_shot_subset``) / stdlib / torch — NEVER from
``stage6alt_thermometer`` or ``stage6alt.orchestrator`` at any scope
(module-top OR function-local). The monolith re-imports *this* module's
symbols at load time, so a ``from ..stage6alt_thermometer import ...``
here would deadlock the import; nothing in this module does that.

``ThermoTeacherProviderPlugin`` is registered-but-INERT at S6A-4 — no
orchestrator walk or test invokes its ``provide_thermo_teacher_side``
hook. S6A-6 plugs the hook into the live Stage 6alt plugin sequencer.
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
# Teacher BPT cache (sweep-shared) — Pattern A: relocated verbatim
# ---------------------------------------------------------------------------


# Bump on any change to the thermometer_teacher_cache.json schema.
# v2: added `teacher_argmax` (per-token predictions for the top1_agreement
#     metric) and switched the cache key from a Nemotron-only spec hash to a
#     corpus-agnostic `corpus_id` (so wikitext/nemotron results never collide).
THERMO_TEACHER_CACHE_FORMAT_VERSION = 2


def _thermo_teacher_cache_key(config: dict, corpus_id: str) -> str:
    """SHA-256 over everything that affects teacher BPT + lm-eval subset.

    `corpus_id` (from `_build_thermo_corpus`) already identifies the corpus
    fully — kind (nemotron/wikitext), size, seed/spec, dataset, and tokenizer
    name — so a corpus, seed-offset, or wikitext-subset change auto-
    invalidates the sweep-shared teacher cache.
    """
    therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
    model_cfg = config["model"]
    payload = json.dumps({
        "format_version": THERMO_TEACHER_CACHE_FORMAT_VERSION,
        "corpus_id": corpus_id,
        "teacher_repo": model_cfg["name_or_path"],
        "teacher_revision": model_cfg.get("revision", "main"),
        "torch_dtype": str(model_cfg.get("torch_dtype", "bfloat16")),
        "attn_implementation": _STAGE6_ATTN_IMPLEMENTATION,
        "arc_easy_limit": int(therm.get("arc_easy_limit", 100)),
        "hellaswag_limit": int(therm.get("hellaswag_limit", 200)),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_thermo_teacher_cache(cache_path: Path, cache_key: str) -> dict | None:
    """Return cached teacher_results on a key+version match, else None."""
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
    return data.get("teacher_results")


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
# Plugin scaffold — Pattern B: inert hook reproducing the monolith block
# ---------------------------------------------------------------------------


class ThermoTeacherProviderPlugin:
    """Stage 6alt thermometer teacher-provider plugin (S6A-4 — registered-but-INERT).

    Owns the Stage 6alt thermometer teacher-side concern: the relocated
    cache-key / load / save helpers (Pattern A) plus an inert
    ``provide_thermo_teacher_side`` hook (Pattern B) that reproduces the
    monolith's teacher block — the cache-hit shortcut, the cache-miss
    student-to-CPU swap, the direct teacher ``load_model`` (attn pinned to
    ``eager`` per Spec F-S-M-1), the cu130/Hopper kernel patches +
    experts-impl shim, the ``_bpt_from_nll`` + ``_lm_eval_subset`` calls,
    the cache save, and the teacher-free + student-restore path.

    S6A-4 wires this class into the plugin registry as metadata only — no
    orchestrator walk or test invokes ``provide_thermo_teacher_side``.
    S6A-6 plugs the hook into the live Stage 6alt plugin sequencer.
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
        """Phase hook — Stage 6alt thermometer teacher-side (S6A-6 wiring surface).

        INERT at S6A-4: no orchestrator walk or test invokes this hook.
        S6A-6 replaces the Stage 6alt orchestrator body with the plugin
        sequencer and dispatches this hook in place of the monolith
        ``run()``'s inline teacher block. The body below reproduces that
        inline block faithfully — it is dead code at S6A-4 but S6A-6
        relies on it once the monolith ``run()`` becomes a thin shim.

        Reproduces, in order, the monolith ``run()``'s teacher block:

        1. **Cache key + path** — ``cache_path`` from ``therm[
           "teacher_cache_path"]`` (falling back to ``artifacts_dir /
           "thermometer_teacher_cache.json"``); ``cache_key`` from
           ``_thermo_teacher_cache_key(config, corpus_id)``.
        2. **Cache load** — ``_load_thermo_teacher_cache(cache_path,
           cache_key)``; HIT publishes ``teacher_results`` /
           ``teacher_cache_hit=True`` / ``teacher_cache_path`` /
           ``teacher_cache_key`` to ctx and returns.
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
            ctx.set("teacher_bpt", teacher_results.get("teacher_bpt"))
            ctx.set("teacher_argmax", teacher_results.get("teacher_argmax"))
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
            revision=model_cfg.get("revision", "main"),
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
