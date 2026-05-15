"""Stage 6alt — "thermometer" cheap directional eval.

A lightweight alternative to the full Stage 6 validation suite. Where Stage 6
runs WikiText PPL + lm-eval + 164-problem HumanEval + 500-problem MATH-500 on
both student and teacher (~$50-120 per ablation row at thinking-mode generation
speeds), this stage measures a single forward-pass signal good enough to tell
the operator whether an ablation knob HELPED or HURT vs the prior row.

Primary metric — bits-per-token (BPT): mean next-token NLL (in bits) over a
fixed, deterministic 64-seq x 2048-token held-out slice of the Nemotron-Cascade
SFT dataset. Pure forward pass, no generation. Reports `student_bpt` and
`bpt_gap = student_bpt - teacher_bpt`.

Secondary tiebreaker — ARC-Easy (limit 100) + HellaSwag (limit 200) zero-shot
`acc_norm`, summed. Consult only when two rows' `bpt_gap` are within ~0.02 bits.

The teacher BPT is computed once and cached to a sweep-shared file so all 12
ablation rows reuse it (the teacher is identical across rows).

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

from .stage6_validate import _STAGE6_ATTN_IMPLEMENTATION, _lm_eval_tasks
from .utils.calibration import (
    CalibrationSpec,
    build_calibration_tensor,
    iter_batches,
    spec_from_config,
)
from .utils.model_io import load_model, save_json_artifact

log = logging.getLogger(__name__)

# Held-out draw: shifts the calibration seed so the thermometer's eval
# sequences do not overlap the Stage 2/2.5 training draw. Bumping this value
# changes the effective seed inside CalibrationSpec.cache_key, which in turn
# changes _thermo_teacher_cache_key — so the teacher cache auto-invalidates.
THERMO_SEED_OFFSET = 715

# Bump on any change to the thermometer_teacher_cache.json schema.
THERMO_TEACHER_CACHE_FORMAT_VERSION = 1

# Default eval subset mix — reasoning-heavy, independent of the chat-dominant
# calibration.subset_weights used for compression. Overridable via the
# `thermometer.subset_weights` config key.
_DEFAULT_SUBSET_WEIGHTS = {"math": 0.35, "swe": 0.25, "chat": 0.25, "science": 0.15}


# ---------------------------------------------------------------------------
# Bits-per-token
# ---------------------------------------------------------------------------


def _bpt_from_nll(model, calib_ids: torch.Tensor, *, device, batch_size: int) -> float:
    """Mean next-token NLL in bits over a pre-tokenized calibration tensor.

    Adapted from `stage6_validate._wikitext2_ppl`'s NLL loop, but returns
    bits-per-token (mean NLL in nats / ln 2) instead of exp(mean NLL), and
    takes a ready-made `(num_seqs, seq_len)` int64 tensor instead of loading
    WikiText. Returns `float("inf")` if any batch is skipped — a loud failure
    rather than a partial-corpus number that would corrupt a directional
    comparison.
    """
    # Batch-size-invariant numerics require eager attention (same requirement
    # as Stage 6's PPL / lm-eval paths). The student is loaded eager by
    # run_pipeline._load_for_stage; the teacher must be loaded eager explicitly.
    _attn_impl = getattr(model.config, "_attn_implementation", None)
    if _attn_impl != _STAGE6_ATTN_IMPLEMENTATION:
        raise RuntimeError(
            f"stage6alt _bpt_from_nll: model.config._attn_implementation="
            f"{_attn_impl!r}, expected {_STAGE6_ATTN_IMPLEMENTATION!r} "
            "(batch-size-invariant NLL requires eager attention)."
        )
    model.train(False)  # inference mode (equivalent to model.eval())

    _dev = device
    if _dev is None:
        try:
            _dev = next(model.parameters()).device
        except StopIteration:
            pass

    nll_sum = 0.0
    tok_count = 0
    skipped = 0
    total = 0
    n_seqs = calib_ids.shape[0]
    log.info("Stage 6alt BPT: %d sequences x len=%d, batch_size=%d",
             n_seqs, calib_ids.shape[1], batch_size)
    with torch.no_grad():
        for i, batch in enumerate(iter_batches(calib_ids, batch_size=batch_size)):
            total += 1
            if _dev is not None:
                batch = batch.to(_dev)
            try:
                out = model(input_ids=batch, labels=batch)
                if out.loss is None:
                    log.warning("stage6alt _bpt_from_nll: None loss; skipping batch")
                    skipped += 1
                    continue
                loss_val = float(out.loss.item())
                if not math.isfinite(loss_val):
                    log.warning("stage6alt _bpt_from_nll: non-finite loss %.2e; "
                                "skipping batch", loss_val)
                    skipped += 1
                    continue
                # (batch.numel() - batch.shape[0]) == B*(seq_len-1): the count
                # of predicted tokens under the standard causal-LM label shift.
                predicted = batch.numel() - batch.shape[0]
                nll_sum += loss_val * predicted
                tok_count += predicted
            except Exception as exc:           # noqa: BLE001
                log.warning("stage6alt _bpt_from_nll: batch error (%s); skipping", exc)
                skipped += 1
                continue
            if (i + 1) % max(1, 64 // batch_size) == 0:
                log.info("  BPT forward %d/%d batches", i + 1,
                         math.ceil(n_seqs / batch_size))
    if tok_count == 0 or skipped > 0:
        log.error("stage6alt _bpt_from_nll: %d/%d batches skipped — returning inf "
                  "(directional comparison must not run on a partial corpus).",
                  skipped, total)
        return float("inf")
    # BPT = mean NLL in nats / ln(2). Computed directly from the running sum —
    # never round-tripped through exp().
    return nll_sum / tok_count / math.log(2)


# ---------------------------------------------------------------------------
# Corpus spec
# ---------------------------------------------------------------------------


def _thermo_corpus_spec(config: dict) -> CalibrationSpec:
    """Build the held-out CalibrationSpec for the thermometer corpus.

    Copies `config["calibration"]`, overlays the thermometer's own
    `subset_weights` (reasoning-heavy, not the chat-dominant training mix),
    and applies `THERMO_SEED_OFFSET` so the draw is disjoint from Stage 2/2.5.
    Never mutates `config["calibration"]` — Stage 2/2.5 read it.
    """
    therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
    cal_cfg = dict(config["calibration"])  # shallow copy — we replace one key
    cal_cfg["subset_weights"] = dict(
        therm.get("subset_weights") or _DEFAULT_SUBSET_WEIGHTS
    )
    return spec_from_config(
        cal_cfg,
        num_sequences_override=int(therm.get("num_sequences", 64)),
        sequence_length_override=int(therm.get("sequence_length", 2048)),
        seed_offset=THERMO_SEED_OFFSET,
    )


# ---------------------------------------------------------------------------
# Teacher BPT cache (sweep-shared)
# ---------------------------------------------------------------------------


def _thermo_teacher_cache_key(config: dict, spec: CalibrationSpec, tokenizer) -> str:
    """SHA-256 over everything that affects teacher BPT + lm-eval subset.

    The corpus spec hash already folds in num_sequences, sequence_length, the
    effective seed (incl. THERMO_SEED_OFFSET), subset_weights, dataset, and
    tokenizer name — so a corpus or seed-offset change auto-invalidates.
    """
    therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
    model_cfg = config["model"]
    payload = json.dumps({
        "format_version": THERMO_TEACHER_CACHE_FORMAT_VERSION,
        "corpus_spec_hash": spec.cache_key(tokenizer.name_or_path),
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
# lm-eval subset
# ---------------------------------------------------------------------------


def _lm_eval_subset(model, tokenizer, *, arc_limit: int, hellaswag_limit: int,
                    batch_size) -> dict:
    """ARC-Easy + HellaSwag zero-shot on a subsample. Two calls (limits differ).

    Returns {arc_easy_acc_norm, hellaswag_acc_norm, acc_norm_sum}. Any metric
    that lm-eval could not produce (e.g. lm-eval not installed) is recorded as
    None and acc_norm_sum is None — BPT alone still carries the signal.
    """
    arc = _lm_eval_tasks(model, tokenizer, ["arc_easy"],
                         batch_size=batch_size, limit=arc_limit)
    hsw = _lm_eval_tasks(model, tokenizer, ["hellaswag"],
                         batch_size=batch_size, limit=hellaswag_limit)
    arc_acc = arc.get("arc_easy_acc")
    hsw_acc = hsw.get("hellaswag_acc")
    acc_sum = (arc_acc + hsw_acc) if (arc_acc is not None and hsw_acc is not None) else None
    return {
        "arc_easy_acc_norm": arc_acc,
        "hellaswag_acc_norm": hsw_acc,
        "acc_norm_sum": acc_sum,
    }


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

    # 1. Build the held-out corpus tensor (cached on disk by build_calibration_tensor).
    spec = _thermo_corpus_spec(config)
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache",
    )

    # 2. STUDENT — already on GPU from run_pipeline._load_for_stage.
    log.info("Stage 6alt: scoring student")
    student_bpt = _bpt_from_nll(model, calib, device=device, batch_size=bpt_batch)
    student_lm = _lm_eval_subset(model, tokenizer, arc_limit=arc_limit,
                                 hellaswag_limit=hsw_limit, batch_size=lm_batch)

    # 3. TEACHER — cache-first; the teacher is identical across all 12 rows.
    cache_path = Path(
        therm.get("teacher_cache_path")
        or artifacts_dir / "thermometer_teacher_cache.json"
    )
    cache_key = _thermo_teacher_cache_key(config, spec, tokenizer)
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
        teacher_bpt = _bpt_from_nll(teacher, calib, device=device,
                                    batch_size=bpt_batch)
        teacher_lm = _lm_eval_subset(teacher, tokenizer, arc_limit=arc_limit,
                                     hellaswag_limit=hsw_limit, batch_size=lm_batch)
        teacher_results = {
            "teacher_bpt": teacher_bpt,
            "teacher_arc_easy_acc_norm": teacher_lm["arc_easy_acc_norm"],
            "teacher_hellaswag_acc_norm": teacher_lm["hellaswag_acc_norm"],
            "teacher_acc_norm_sum": teacher_lm["acc_norm_sum"],
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
        "corpus": {
            "num_sequences": spec.num_sequences,
            "sequence_length": spec.sequence_length,
            "effective_seed": spec.seed,
            "seed_offset": THERMO_SEED_OFFSET,
            "subset_weights": spec.subset_weights,
        },
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
    log.info("Stage 6alt complete: student_bpt=%.4f teacher_bpt=%.4f bpt_gap=%s "
             "-> %s", student_bpt, teacher_bpt, results["bpt_gap"], path)
    return path
