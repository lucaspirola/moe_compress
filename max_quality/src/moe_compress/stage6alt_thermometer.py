"""Stage 6alt — "thermometer" cheap directional eval.

A lightweight alternative to the full Stage 6 validation suite. Where Stage 6
runs WikiText PPL + lm-eval + 164-problem HumanEval + 500-problem MATH-500 on
both student and teacher (~$50-120 per ablation row at thinking-mode generation
speeds), this stage measures a single forward-pass signal good enough to tell
the operator whether an ablation knob HELPED or HURT vs the prior row.

Primary metric — bits-per-token (BPT): mean next-token NLL (in bits) over a
fixed 64-seq x 2048-token corpus. Pure forward pass, no generation. Reports
`student_bpt`, `teacher_bpt`, and `bpt_gap = student_bpt - teacher_bpt`.

CORPUS CHOICE (config `thermometer.corpus`) — critical for interpreting bpt_gap:
  - "wikitext": WikiText-2 test split. General text the student was NOT
    Stage-2.5-trained on, so `bpt_gap` is a FAIR teacher-vs-student
    compression-damage number (expected sign: positive — student worse).
  - "nemotron" (default): a held-out slice of the Nemotron-Cascade SFT data.
    This is the SAME distribution Stage 2.5 Router KD trains the student on,
    so `bpt_gap` here CONFLATES compression damage with the student's
    distribution adaptation (it can go negative). Trust it only for cross-row
    A0..A11 RANKING — where the adaptation is common-mode and cancels — never
    as an absolute teacher-vs-student claim.

Secondary signals — ARC-Easy/HellaSwag zero-shot `acc_norm` summed, and
`top1_agreement`: the fraction of corpus positions where student and teacher
argmax the same next token. Agreement is a training-distribution-independent
damage measure (it asks "did the student stay faithful to the teacher",
not "did the student get good at this text").

The teacher BPT, lm-eval, and per-token argmax are computed once and cached to
a sweep-shared file so all 12 ablation rows reuse them (teacher is constant).

Selected via `config["stage6_validate"]["mode"] == "thermometer"`; see
run_pipeline.py's Stage 6 dispatch. Default mode is "full" (stage6_validate).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path

import torch

from .stage6_validate import (
    _STAGE6_ATTN_IMPLEMENTATION,
    _apply_stage6_kernel_patches,
    _set_experts_implementation_s6,
)
from .utils.model_io import load_model, save_json_artifact

# S6A-2: re-export the Stage 6alt thermometer corpus Pattern-A symbols
# (constants + functions) from their plugin home so existing import paths
# (this module's own `run()`, plus external callers like
# `stage2/orchestrator.py`'s xD calibration that imports
# `_thermo_wikitext_tensor`) keep working unchanged. The plugin classes
# `ThermoEnvironmentPlugin` / `ThermoCorpusPlugin` are imported alongside
# so an external registry walker can pick them up via the monolith too.
from .stage6alt.plugins.thermo_corpus import (  # noqa: F401
    THERMO_SEED_OFFSET,
    _DEFAULT_SUBSET_WEIGHTS,
    _thermo_corpus_spec,
    _thermo_wikitext_tensor,
    _build_thermo_corpus,
    ThermoCorpusPlugin,
)
from .stage6alt.plugins.thermo_environment import ThermoEnvironmentPlugin  # noqa: F401

# S6A-3: re-export the Stage 6alt thermometer BPT-metric + zero-shot-subset
# Pattern-A symbols (the two helper functions) from their plugin homes so the
# existing import path keeps working. The S6A-0 golden snapshot patches
# ``stage6alt_thermometer._bpt_from_nll`` / ``stage6alt_thermometer._lm_eval_subset``
# directly via ``monkeypatch.setattr``; the re-import puts the SAME function
# object on the monolith namespace, so that patch-by-attribute keeps biting.
# The plugin classes ``BptMetricPlugin`` / ``ZeroShotSubsetPlugin`` are imported
# alongside so an external registry walker can pick them up via the monolith too.
from .stage6alt.plugins.bpt_metric import (  # noqa: F401
    _bpt_from_nll,
    BptMetricPlugin,
)
from .stage6alt.plugins.zero_shot_subset import (  # noqa: F401
    _lm_eval_subset,
    ZeroShotSubsetPlugin,
)

log = logging.getLogger(__name__)

# S6A-2: `THERMO_SEED_OFFSET` and `_DEFAULT_SUBSET_WEIGHTS` are relocated to
# `stage6alt/plugins/thermo_corpus.py` and re-imported above (see the
# `# noqa: F401` block) so existing call sites in this module / external
# callers still resolve them via `stage6alt_thermometer.THERMO_SEED_OFFSET`.

# Bump on any change to the thermometer_teacher_cache.json schema.
# v2: added `teacher_argmax` (per-token predictions for the top1_agreement
#     metric) and switched the cache key from a Nemotron-only spec hash to a
#     corpus-agnostic `corpus_id` (so wikitext/nemotron results never collide).
THERMO_TEACHER_CACHE_FORMAT_VERSION = 2


# S6A-3: `_bpt_from_nll` is relocated to `stage6alt/plugins/bpt_metric.py` and
# re-imported above (see the `# noqa: F401` block); call sites below resolve
# it through that re-import. The orphaned `iter_batches` import (only used by
# `_bpt_from_nll`) was removed alongside the relocation.


# S6A-2: `_thermo_corpus_spec`, `_thermo_wikitext_tensor`, `_build_thermo_corpus`
# are relocated to `stage6alt/plugins/thermo_corpus.py` and re-imported above
# (see the `# noqa: F401` block); call sites below resolve them through that
# re-import.


# ---------------------------------------------------------------------------
# Teacher BPT cache (sweep-shared)
# ---------------------------------------------------------------------------


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


# S6A-3: `_lm_eval_subset` is relocated to `stage6alt/plugins/zero_shot_subset.py`
# and re-imported above (see the `# noqa: F401` block); call sites below resolve
# it through that re-import.


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    """Stage 6alt thermometer. Same contract as stage6_validate.run.

    Writes `stage6alt_eval.json` to `artifacts_dir` and returns its Path.
    """
    s6 = config["stage6_validate"]
    therm = s6.get("thermometer", {}) or {}
    bpt_batch = int(therm.get("bpt_batch_size", 8))
    lm_batch = therm.get("lm_eval_batch_size", "auto:8")
    arc_limit = int(therm.get("arc_easy_limit", 100))
    hsw_limit = int(therm.get("hellaswag_limit", 200))

    log.info("=== Stage 6alt — Thermometer ===")

    # Apply the cu130/Hopper segfault-fix patches to the student before any
    # forward pass. Mirrors stage6_validate.run(): without these, fla's
    # FusedRMSNormGated Triton kernel and chunk_gated_delta_rule SIGSEGV on
    # H-series GPUs partway through lm-eval loglikelihood scoring. Both helpers
    # are idempotent and no-op on models without GatedDeltaNet modules.
    _experts_impl = os.environ.get(
        "EXPERTS_IMPLEMENTATION", s6.get("experts_implementation", "batched_mm")
    )
    _set_experts_implementation_s6(model, _experts_impl)
    _apply_stage6_kernel_patches(model, role="student")

    # 1. Build the evaluation corpus — see _build_thermo_corpus + module docstring.
    calib, corpus_meta, corpus_id = _build_thermo_corpus(
        config, tokenizer, artifacts_dir,
    )

    # 2. STUDENT — already on GPU from run_pipeline._load_for_stage.
    log.info("Stage 6alt: scoring student")
    student_bpt, student_argmax = _bpt_from_nll(
        model, calib, device=device, batch_size=bpt_batch, collect_argmax=True,
    )
    student_lm = _lm_eval_subset(model, tokenizer, arc_limit=arc_limit,
                                 hellaswag_limit=hsw_limit, batch_size=lm_batch)

    # 3. TEACHER — cache-first; the teacher is identical across all 12 rows.
    cache_path = Path(
        therm.get("teacher_cache_path")
        or artifacts_dir / "thermometer_teacher_cache.json"
    )
    cache_key = _thermo_teacher_cache_key(config, corpus_id)
    teacher_results = _load_thermo_teacher_cache(cache_path, cache_key)
    teacher_cache_hit = teacher_results is not None

    if teacher_cache_hit:
        log.info("Stage 6alt: teacher BPT cache HIT (%s) — skipping teacher load",
                 cache_path)
    else:
        log.info("Stage 6alt: teacher BPT cache miss — loading teacher")
        # Free the student's GPU memory before the teacher comes on. Guarded:
        # an accelerate-sharded student (device_map="auto" spanning >1 GPU)
        # raises NotImplementedError on .to() — mirror stage6_validate.py's
        # teacher-swap block, which logs and continues rather than aborting.
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
        _set_experts_implementation_s6(teacher, _experts_impl)
        _apply_stage6_kernel_patches(teacher, role="teacher")
        teacher_bpt, teacher_argmax = _bpt_from_nll(
            teacher, calib, device=device, batch_size=bpt_batch,
            collect_argmax=True,
        )
        teacher_lm = _lm_eval_subset(teacher, tokenizer, arc_limit=arc_limit,
                                     hellaswag_limit=hsw_limit, batch_size=lm_batch)
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
        try:
            model.to(device or "cuda")
        except Exception as exc:           # noqa: BLE001
            log.warning("Stage 6alt: could not restore student to GPU "
                        "(%s) — harmless, the pipeline ends after Stage 6.", exc)

    # 4. Assemble + write results.
    teacher_bpt = teacher_results["teacher_bpt"]
    teacher_acc_sum = teacher_results.get("teacher_acc_norm_sum")
    student_acc_sum = student_lm["acc_norm_sum"]

    # top1_agreement — fraction of corpus positions where student and teacher
    # argmax the same next token. Unlike bpt_gap this does not depend on what
    # text the student was trained on, so it is a fair compression-damage
    # signal on ANY corpus. None if either model skipped a BPT batch.
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
        "student_arc_easy_acc_norm": student_lm["arc_easy_acc_norm"],
        "student_hellaswag_acc_norm": student_lm["hellaswag_acc_norm"],
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
    return path
