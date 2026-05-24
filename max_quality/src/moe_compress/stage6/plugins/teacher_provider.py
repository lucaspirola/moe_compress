"""Teacher provider (S6-5 of the Stage 6 plugin-architecture refactor).

Paper / spec source
--------------------
No upstream paper for the teacher provider per se; this plugin owns
the Stage 6 teacher-side eval loop:

- **Cache-key invariant** (project ``VALIDATED_STRATEGIES`` §Stage 6
  Optimization #9): SHA-256 cache key over teacher checkpoint SHA,
  per-task ``dataset_revisions`` (wikitext_ppl, humaneval, math500 —
  the lm-eval-managed tasks like hellaswag/arc_challenge are NOT in
  the key; they're handled by ``lm_eval_version`` + a SHA-256 of the
  lm-eval task config), greedy/sampling protocol, tokenizer SHA, and
  batched-vs-bs=1 invariance flag.
- **Background CPU preload** + teacher-side eval loop running the
  same four eval families against the teacher.
- **Cache load/save** with atomic ``.tmp + os.replace`` writes
  (project §11 durability contract).

The cache hit invariance is what makes Stage 6 ~8-12× faster — the
teacher's eval results are deterministic given the cache key, so a
re-run of the student against the same teacher reuses the cache.

Home of the Stage 6 teacher-provider concern, extracted from the legacy
``stage6_validate.py`` monolith. The teacher provider owns the teacher
side of the Stage 6 validation gate: the eval-cache key + load/save, the
background CPU preload, and the teacher-side eval loop (WikiText-2 PPL +
lm-eval zero-shot + HumanEval + MATH-500) whose results are compared
against the student's metrics to produce per-task deltas.

Pattern A vs Pattern B
----------------------
S6-5 covers a MIXED pattern (mirror of S6-2 / S6-3 / S6-4):

* **Pattern A -- relocated verbatim**: ``TEACHER_CACHE_FORMAT_VERSION``,
  ``_safe_pkg_version``, ``_teacher_cache_key``, ``_load_teacher_cache``,
  ``_save_teacher_cache`` and ``_preload_teacher_to_cpu`` below are
  character-identical copies of the monolith bodies. ``stage6_validate.py``
  re-imports them (a ``# noqa: F401`` block) so ``run()`` and external
  callers/tests (e.g. ``test_teacher_eval_cache_key_invariant``) keep their
  original import path.
* **Pattern B -- reproduced in an inert hook**: the ``run()`` teacher-side
  block (cache-hit shortcut, preload-thread join + queue.get_nowait
  fallback, ``teacher.eval()`` / kernel patches / experts-impl shim /
  optional torch.compile, the four conditional teacher-side eval calls,
  cache save) is INLINE ``run()`` code in the monolith -- there is nothing
  standalone to relocate. The ``provide_teacher_side`` hook below
  REPRODUCES that inline block faithfully; the monolith ``run()`` is NOT
  modified for it. This is an intentional, temporary logic duplication
  that resolves at S6-8 when the monolith ``run()`` is deleted and this
  hook is wired live.

Circular-import contract (mirror of ``stage6/plugins/eval_environment.py``):
this module imports only from ``..context`` / ``...utils`` / sibling
plugin modules (``eval_environment``, ``wikitext_ppl``,
``zero_shot_lm_eval``, ``humaneval``, ``math500``) / stdlib / torch --
NEVER from ``stage6_validate``, ``stage6.orchestrator`` or
``orchestrator`` at any scope (module-top OR function-local). The monolith
re-imports *this* module at load time, so a ``from ..stage6_validate
import ...`` here would deadlock the import; nothing in this module does
that.

``TeacherProviderPlugin`` is registered-but-INERT at S6-5 -- no orchestrator
walk or test invokes its ``provide_teacher_side`` hook. S6-8 plugs the hook
into the live Stage 6 plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _md
import json
import logging
import os
import queue
import re
import time
from pathlib import Path
from typing import Any

import torch

from ..context import PipelineContext
from ...utils.model_io import (
    count_expert_parameters,
    count_parameters_effective,
    load_model,
)
from .eval_environment import (
    _apply_stage6_kernel_patches,
    _resolve_dataset_revisions,
    _set_experts_implementation_s6,
)
from .humaneval import _humaneval
from .math500 import _math500
from .wikitext_ppl import _wikitext2_ppl
from .zero_shot_lm_eval import _lm_eval_tasks

log = logging.getLogger(__name__)


# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Constant -- never override at call sites. This is
# a module-LOCAL copy of the monolith's ``_STAGE6_ATTN_IMPLEMENTATION``: the
# monolith keeps its own definition and is NOT imported here (circular-import
# contract). Both copies must stay in sync until S6-8 collapses the monolith.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"


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


class TeacherProviderPlugin:
    """Stage 6 teacher-provider plugin (S6-5 -- registered-but-INERT).

    Owns the Stage 6 teacher concern: the eval-cache key + load/save, the
    background CPU preload, and the teacher-side eval loop (WikiText-2 PPL +
    lm-eval zero-shot + HumanEval + MATH-500). The standalone helpers
    (Pattern A) are relocated verbatim above and re-imported by the monolith;
    the ordering glue around them (cache-hit shortcut, preload-join,
    post-load patches, the four conditional teacher-side eval calls, cache
    save) is reproduced in the ``provide_teacher_side`` hook below
    (Pattern B).

    S6-5 wires this class into the plugin registry as metadata only -- no
    orchestrator walk or test invokes ``provide_teacher_side``. S6-8 plugs
    the hook into the live Stage 6 plugin sequencer and deletes the
    monolith ``run()``.
    """

    name = "teacher_provider"
    paper = "Stage 6 teacher provider — cache-key invariant + background preload (no upstream paper; VALIDATED_STRATEGIES §Stage 6 Opt #9). See module docstring."
    config_key = "stage6_validate.teacher_eval_cache"
    reads: tuple[str, ...] = (
        "config", "artifacts_dir", "tokenizer", "dataset_revisions",
        "experts_impl", "use_torch_compile",
    )
    writes: tuple[str, ...] = ("teacher_results", "teacher_param_counts")
    # teacher_results is a per-side collector dict (analogue of the monolith's
    # `results["teacher"]`) -- a result collector is NOT a calibration-pass
    # accumulator, so it belongs in `writes`, not `provides`. (S6-8 wires the
    # collector.) Mirrors the S6-3 zero_shot / wikitext convention.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True -- the teacher side is UNCONDITIONAL.

        Every Stage 6 run must produce teacher metrics (either from the cache
        or from a fresh teacher load + eval) so the delta-vs-student gate can
        be computed; ``config_key`` only names *where* the cache lives, it
        never gates the plugin as a whole. The hook itself contains the
        internal cache-hit shortcut and the per-sub-metric ``enabled``
        guards that the monolith ``run()`` applies inline.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def provide_teacher_side(self, ctx: PipelineContext) -> None:
        """Phase hook -- Stage 6 teacher-side eval (S6-8 wiring surface).

        INERT at S6-5: no orchestrator walk or test invokes this hook. S6-8
        replaces the Stage 6 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        teacher-side block. The body below reproduces that inline block
        faithfully -- it is dead code at S6-5 but S6-8 relies on it once the
        monolith ``run()`` is deleted.

        Reproduces, in order, the monolith ``run()``'s teacher-side block:

        1. **Cache-hit shortcut** -- if ``cached_teacher_results`` is already
           in ctx (the orchestrator pre-resolved the cache), write it
           straight to ``teacher_results`` and return without touching the
           model.
        2. **Preload-thread join + queue.get_nowait fallback** -- wait for
           the background CPU preload to finish; if the queue is empty or
           the thread did not start, load the teacher directly via
           ``load_model(...)`` (attn pinned to ``eager`` per Spec F-S-M-1).
        3. **inference-mode + kernel patches + experts-impl shim** --
           switch to inference mode, apply the cu130/Hopper segfault-fix
           patches, mirror the student-side experts-impl so the
           generative-switch comparison has a baseline.
        4. **Optional ``torch.compile``** -- guarded by ``use_torch_compile``
           ctx slot; on failure logs a warning and falls through.
        5. **Conditional teacher-side eval calls** -- gated by the same
           ``s6["wikitext2"]["enabled"]`` / ``s6["zero_shot"]["enabled"]`` /
           ``s6["generative"]["enabled"]`` flags ``run()`` uses, writing
           each result to a local ``teacher_results`` dict. The generative
           sub-block restores uncompiled ``teacher.forward`` and switches
           ``experts_implementation`` to ``batched_mm`` (cu130 _grouped_mm
           decode-shape workaround), same as the student-side block.
        6. **Cache save** -- if ``teacher_cache_enabled`` was on, save the
           results + param counts via ``_save_teacher_cache``; failure is a
           warning, never a re-raise.
        7. **ctx writes** -- ``teacher_results`` and ``teacher_param_counts``
           (the latter is None on the cache-HIT path -- that's the same
           lifetime the monolith honors).

        S6-8 will add the gguf thread start when the imatrix plugin is wired
        -- at S6-5 the ``_background_gguf_convert`` thread launch from the
        monolith is intentionally OMITTED here (it is an imatrix concern,
        not the teacher provider's, and the imatrix plugin has not been
        extracted yet).
        """
        # Required slots -- direct get(): a missing one is a wiring bug and
        # SHOULD raise.
        config = ctx.get("config")
        tokenizer = ctx.get("tokenizer")
        artifacts_dir = ctx.get("artifacts_dir")
        s6 = config["stage6_validate"]

        # Optional side-channels (analogues of monolith run()'s locals).
        device = ctx.get("device") if ctx.has("device") else None
        dataset_revisions = (
            ctx.get("dataset_revisions") if ctx.has("dataset_revisions") else {}
        )
        # Resolve experts_impl matching the monolith run()'s top-of-block
        # logic: ctx (set by EvalEnvironmentPlugin) wins; otherwise mirror
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
        use_torch_compile = (
            ctx.get("use_torch_compile") if ctx.has("use_torch_compile") else False
        )

        # Resolve cache locals exactly as run() does so the eventual save
        # path lines up.
        teacher_cache_cfg = s6.get("teacher_eval_cache", {})
        teacher_cache_enabled = teacher_cache_cfg.get("enabled", False)
        cache_key = _teacher_cache_key(config)
        cache_path = Path(
            teacher_cache_cfg.get("cache_path")
            or str(artifacts_dir / "teacher_eval_cache.json")
        )

        # Step 1: cache-hit shortcut. The orchestrator pre-resolves the cache
        # and exposes the results dict (and param counts) via the
        # `cached_teacher_results` / `cached_teacher_param_counts` ctx slots;
        # mirrors the monolith run()'s
        #   `if cached_teacher_results is not None:`
        # branch that skips the teacher load + eval entirely.
        cached_teacher_results = (
            ctx.get("cached_teacher_results")
            if ctx.has("cached_teacher_results") else None
        )
        cached_teacher_param_counts = (
            ctx.get("cached_teacher_param_counts")
            if ctx.has("cached_teacher_param_counts") else None
        )
        if cached_teacher_results is not None:
            log.info("Stage 6: using cached teacher results (key=%s)", cache_key)
            ctx.set("teacher_results", cached_teacher_results)
            ctx.set("teacher_param_counts", cached_teacher_param_counts)
            return

        # Step 2: wait for the background CPU preload, fall back to a direct
        # load if the queue is empty or the thread did not start.
        teacher_preload_q = (
            ctx.get("teacher_preload_q") if ctx.has("teacher_preload_q") else None
        )
        preload_thread = (
            ctx.get("preload_thread") if ctx.has("preload_thread") else None
        )
        teacher_preloaded = None
        if preload_thread is not None:
            log.info("Stage 6: waiting for teacher preload thread to complete")
            preload_thread.join(timeout=3600)
            if preload_thread.is_alive():
                log.warning(
                    "Preload thread did not complete within 3600s; proceeding without preloaded teacher."
                )
            else:
                if teacher_preload_q is not None and not teacher_preload_q.empty():
                    teacher_preloaded = teacher_preload_q.get_nowait()
        if teacher_preloaded is not None:
            teacher = teacher_preloaded
            log.info("Stage 6: moving preloaded teacher to GPU")
            teacher.to(device or "cuda")
        else:
            # Preload failed or wasn't started -- load directly.
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

        # Step 3: apply the cu130/Hopper segfault-fix patches to the teacher.
        # Without this the teacher's HumanEval generate() segfaults exactly
        # like the student's used to (same Qwen3.5-MoE architecture + same
        # fla FusedRMSNormGated Triton kernel crash on decode shape).
        _apply_stage6_kernel_patches(teacher, role="teacher")

        # Mirror the student-side initial experts_implementation set so the
        # later "switch for generative" comparison has a valid baseline.
        # Unconditional — `experts_impl` is resolved to a non-None default
        # above (matches the monolith run() which calls this unconditionally).
        _set_experts_implementation_s6(teacher, experts_impl)

        # Step 4: optional torch.compile on teacher too. Mode 'default'
        # (was 'reduce-overhead'): same rationale as student compile --
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

        # S6-8 will add the gguf thread start when the imatrix plugin is
        # wired (the monolith run() starts _background_gguf_convert here,
        # immediately before the teacher-side eval calls below; it is an
        # imatrix concern, not the teacher provider's).

        # Read eval batch-size configs the same way run() does so the
        # teacher-side calls match the student-side ones from the wikitext /
        # zero_shot / generative plugins. Faithful reproduction of monolith
        # F-iter4-LOW-2 validation: reject invalid configs early instead of
        # surfacing them as confusing tracebacks deep inside lm-eval / generate.
        ppl_batch_size = int(s6.get("ppl_batch_size", 8))
        _raw_lebs = s6.get("lm_eval_batch_size", "auto:8")
        if isinstance(_raw_lebs, int):
            if _raw_lebs <= 0:
                raise ValueError(
                    f"stage6_validate.lm_eval_batch_size must be > 0; got {_raw_lebs}"
                )
            lm_eval_batch_size: Any = _raw_lebs
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

        # Step 5: conditional teacher-side eval calls (gated by the same
        # flags run() uses).
        teacher_results: dict[str, Any] = {}
        if s6["wikitext2"]["enabled"]:
            teacher_results["wikitext2_ppl"] = _wikitext2_ppl(
                teacher, tokenizer, s6["wikitext2"], device=device,
                batch_size=ppl_batch_size, dataset_revisions=dataset_revisions,
            )
        if s6["zero_shot"]["enabled"]:
            teacher_results.update(
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
                teacher_results["humaneval_pass_at_1"] = _humaneval(
                    teacher, tokenizer, s6["generative"]["humaneval"], device=device,
                    batch_size=gen_batch_size, dataset_revisions=dataset_revisions,
                )
            if "math500" in s6["generative"]:
                teacher_results["math500_accuracy"] = _math500(
                    teacher, tokenizer, s6["generative"]["math500"], device=device,
                    batch_size=gen_batch_size, dataset_revisions=dataset_revisions,
                )

        # Step 6: save teacher results to cache for future runs. teacher_pc
        # is computed ONLY when the cache is enabled — matches the monolith,
        # which leaves `teacher_param_counts` unbound on the non-cache path
        # and lets _measured_reduction count live teacher params from the
        # model itself. When caching is disabled the ctx slot is set to None.
        # F-iter4-CRIT-2: teacher has no FactoredExperts modules so the
        # effective count == physical count, but use the same effective
        # function for symmetry and so the cached value compares apples-to-
        # apples with the student's effective live param count.
        teacher_pc: dict[str, int] | None = None
        if teacher_cache_enabled:
            teacher_pc = {
                "total": count_parameters_effective(teacher),
                "expert": count_expert_parameters(teacher, routed_only=True),
            }
            try:
                _save_teacher_cache(
                    cache_path, cache_key, teacher_results,
                    teacher_param_counts=teacher_pc,
                )
            except Exception as exc:
                log.warning("_save_teacher_cache: failed (%s); continuing without cache", exc)

        # Step 7: ctx writes.
        ctx.set("teacher_results", teacher_results)
        ctx.set("teacher_param_counts", teacher_pc)


__all__ = [
    "TEACHER_CACHE_FORMAT_VERSION",
    "_safe_pkg_version",
    "_teacher_cache_key",
    "_load_teacher_cache",
    "_save_teacher_cache",
    "_preload_teacher_to_cpu",
    "TeacherProviderPlugin",
]
