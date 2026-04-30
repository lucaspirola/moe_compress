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
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .utils.calibration import iter_batches
from .utils.model_io import (
    count_expert_parameters,
    count_parameters,
    load_model,
    save_json_artifact,
)
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher eval caching (Optimization #7)
# ---------------------------------------------------------------------------

def _teacher_cache_key(config: dict) -> str:
    """Compute a deterministic cache key from teacher model identity + eval config.

    Key components: model name, revision, and the subset of stage6 config that
    affects teacher evaluation (wikitext2, zero_shot, generative settings).
    The teacher is deterministic — same model + same eval = same numbers.
    """
    s6 = config["stage6_validate"]
    payload = json.dumps({
        "model_name_or_path": config["model"]["name_or_path"],
        "model_revision": config["model"].get("revision") or "main",
        "torch_dtype": config["model"].get("torch_dtype", "bfloat16"),
        "wikitext2": s6.get("wikitext2", {}),
        "zero_shot": s6.get("zero_shot", {}),
        "generative": s6.get("generative", {}),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_teacher_cache(cache_path: Path, cache_key: str) -> dict | None:
    """Load cached teacher eval results if they exist and the key matches.

    Returns a dict with keys "results" and optionally "param_counts", or None.
    """
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        if data.get("cache_key") != cache_key:
            log.info("Teacher cache key mismatch (expected %s, found %s) — re-evaluating.",
                     cache_key, data.get("cache_key"))
            return None
        log.info("Teacher eval cache HIT (%s) — skipping teacher load+eval entirely.", cache_key)
        return {
            "results": data["teacher_results"],
            "param_counts": data.get("teacher_param_counts"),
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
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
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(str(tmp), str(cache_path))
    log.info("Teacher eval cache saved → %s (key=%s)", cache_path, cache_key)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    s6 = config["stage6_validate"]
    model.eval()   # stage5 leaves model in train(); set eval before any sub-metric
    results: dict = {"student": {}, "teacher": {}, "delta": {}, "thresholds": {}}

    calib_texts: list[str] = []  # accumulates eval text for imatrix calibration

    # Optimization #5: torch.compile for prefill-dominant paths.
    # Compile model.forward (NOT model.generate) before evaluations begin.
    # dynamic=True handles variable-length padded batches from lm-eval.
    # One-time compilation cost (~3-5 min) is amortized across 1000+ forward passes.
    use_torch_compile = s6.get("torch_compile", False)
    if use_torch_compile:
        log.info("Stage 6: applying torch.compile(dynamic=True, mode='reduce-overhead') to model.forward")
        try:
            model.forward = torch.compile(model.forward, dynamic=True, mode="reduce-overhead")
            log.info("Stage 6: torch.compile applied successfully")
        except Exception as exc:
            log.warning("Stage 6: torch.compile failed (%s) — continuing without compilation", exc)
            use_torch_compile = False

    # Read batch size configs with defaults tuned for H200.
    ppl_batch_size = s6.get("ppl_batch_size", 8)
    lm_eval_batch_size = s6.get("lm_eval_batch_size", "auto:8")
    gen_batch_size = s6.get("gen_batch_size", 8)

    # 1. WikiText-2 PPL on student (Optimization #1: batch_size=8)
    if s6["wikitext2"]["enabled"]:
        log.info("Stage 6: WikiText-2 PPL (student), batch_size=%d", ppl_batch_size)
        results["student"]["wikitext2_ppl"] = _wikitext2_ppl(
            model, tokenizer, s6["wikitext2"], device=device, collect=calib_texts,
            batch_size=ppl_batch_size,
        )

    # 2. Zero-shot via lm-eval (ARC-C + HellaSwag) (Optimization #2: batch_size=auto:8)
    if s6["zero_shot"]["enabled"]:
        log.info("Stage 6: zero-shot harness, batch_size=%s", lm_eval_batch_size)
        results["student"].update(
            _lm_eval_tasks(model, tokenizer, s6["zero_shot"]["tasks"],
                           collect=calib_texts, batch_size=lm_eval_batch_size)
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

    teacher_preload: dict = {}  # will hold {"model": teacher_model} when ready
    preload_thread = None
    if cached_teacher_results is None:
        # We need the teacher — start preloading to CPU RAM in background.
        preload_thread = threading.Thread(
            target=_preload_teacher_to_cpu,
            args=(config, teacher_preload),
            daemon=True,
            name="teacher-preload",
        )
        preload_thread.start()
        log.info("Stage 6: teacher preload started in background thread")

    # 3. Generative — HumanEval + MATH-500 (Optimizations #3, #4: batched generate)
    if s6["generative"]["enabled"]:
        log.info("Stage 6: generative (HumanEval + MATH-500), gen_batch_size=%d", gen_batch_size)
        if "humaneval" in s6["generative"]:
            results["student"]["humaneval_pass_at_1"] = _humaneval(
                model, tokenizer, s6["generative"]["humaneval"], device=device,
                collect=calib_texts, batch_size=gen_batch_size,
            )
        if "math500" in s6["generative"]:
            results["student"]["math500_accuracy"] = _math500(
                model, tokenizer, s6["generative"]["math500"], device=device,
                collect=calib_texts, batch_size=gen_batch_size,
            )

    # 4. Snapshot student param counts BEFORE loading teacher.
    student_total = count_parameters(model)
    student_expert = count_expert_parameters(model, routed_only=True)

    # Initialize gguf_thread and gguf_result at this scope level so the
    # imatrix dispatch below can reference them regardless of which branch runs.
    gguf_thread = None
    gguf_result: dict = {}

    # Optimization #7: Use cached teacher results if available.
    if cached_teacher_results is not None:
        log.info("Stage 6: using cached teacher results (key=%s)", cache_key)
        results["teacher"] = cached_teacher_results
        teacher = None  # no teacher model needed
    else:
        # 5. Free student GPU memory before loading teacher.
        try:
            model.to("cpu")
        except Exception as exc:
            log.warning("Could not move student to CPU before teacher load: %s", exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Wait for preload thread to finish (teacher weights should be in host RAM).
        if preload_thread is not None:
            log.info("Stage 6: waiting for teacher preload thread to complete")
            preload_thread.join()

        if "model" in teacher_preload:
            # Teacher was preloaded to CPU — move to GPU.
            teacher = teacher_preload["model"]
            log.info("Stage 6: moving preloaded teacher to GPU")
            teacher.to(device or "cuda")
        else:
            # Preload failed or wasn't started — load directly.
            log.info("Stage 6: loading uncompressed baseline for delta computation")
            teacher, _ = load_model(
                config["model"]["name_or_path"],
                revision=config["model"].get("revision", "main"),
                torch_dtype=config["model"]["torch_dtype"],
                device_map=config["model"]["device_map"],
                attn_implementation=config["model"]["attn_implementation"],
                load_in_4bit=config["model"].get("load_in_4bit", False),
                trust_remote_code=config["model"].get("trust_remote_code", False),
            )
        teacher.eval()

        # Optimization #5: torch.compile on teacher too.
        if use_torch_compile:
            try:
                teacher.forward = torch.compile(teacher.forward, dynamic=True, mode="reduce-overhead")
                log.info("Stage 6: torch.compile applied to teacher")
            except Exception as exc:
                log.warning("Stage 6: torch.compile on teacher failed (%s)", exc)

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
                batch_size=ppl_batch_size,
            )
        if s6["zero_shot"]["enabled"]:
            results["teacher"].update(
                _lm_eval_tasks(teacher, tokenizer, s6["zero_shot"]["tasks"],
                               batch_size=lm_eval_batch_size)
            )
        if s6["generative"]["enabled"]:
            if "humaneval" in s6["generative"]:
                results["teacher"]["humaneval_pass_at_1"] = _humaneval(
                    teacher, tokenizer, s6["generative"]["humaneval"], device=device,
                    batch_size=gen_batch_size,
                )
            if "math500" in s6["generative"]:
                results["teacher"]["math500_accuracy"] = _math500(
                    teacher, tokenizer, s6["generative"]["math500"], device=device,
                    batch_size=gen_batch_size,
                )

        # Save teacher results to cache for future runs.
        if teacher_cache_enabled:
            teacher_pc = {
                "total": count_parameters(teacher),
                "expert": count_expert_parameters(teacher, routed_only=True),
            }
            _save_teacher_cache(cache_path, cache_key, results["teacher"],
                                teacher_param_counts=teacher_pc)

    # 5. Deltas and threshold checks
    results["delta"] = _deltas(results["student"], results["teacher"])
    results["measured_reduction"] = _measured_reduction(
        model,
        student_total=student_total, student_expert=student_expert,
        teacher_model=teacher,  # may be None if cached
        cached_teacher_param_counts=cached_teacher_param_counts,
        config=config,
    )
    results["thresholds"] = _check_thresholds(results, s6["thresholds"])

    path = artifacts_dir / "stage6_eval.json"
    save_json_artifact(results, path)

    # Free teacher GPU memory before llama-imatrix subprocess uses the GPU.
    if teacher is not None:
        try:
            teacher.to("cpu")
            del teacher
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Optimization #8: If GGUF conversion was running in background, wait for it.
    # Then run llama-imatrix (which needs the GPU, now freed from teacher).
    if cached_teacher_results is None and gguf_thread is not None:
        log.info("Stage 6: waiting for background GGUF conversion to complete")
        gguf_thread.join()
        _run_llama_imatrix_with_prebuilt_gguf(
            calib_texts, s6.get("imatrix", {}), artifacts_dir, gguf_result,
        )
    else:
        # Either teacher was cached (no GGUF in background), or no gguf_thread.
        _generate_imatrix(calib_texts, s6.get("imatrix", {}), artifacts_dir)

    overall_pass = all(results["thresholds"].values())
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
    for k, triple in results.get("delta", {}).items():
        if isinstance(triple, dict):
            for sub in ("student", "teacher", "delta"):
                if sub in triple:
                    try:
                        flat[f"stage6/delta/{k}/{sub}"] = float(triple[sub])
                    except (TypeError, ValueError):
                        pass
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
            {k: v for k, v in results["thresholds"].items() if not v},
        )
    return path


# ---------------------------------------------------------------------------
# Teacher preload (Optimization #6)
# ---------------------------------------------------------------------------

def _preload_teacher_to_cpu(config: dict, result: dict) -> None:
    """Load teacher model weights to CPU RAM in a background thread."""
    try:
        log.info("Teacher preload: loading %s to CPU...", config["model"]["name_or_path"])
        t0 = time.monotonic()
        teacher, _ = load_model(
            config["model"]["name_or_path"],
            revision=config["model"].get("revision", "main"),
            torch_dtype=config["model"]["torch_dtype"],
            device_map="cpu",
            attn_implementation=config["model"]["attn_implementation"],
            load_in_4bit=config["model"].get("load_in_4bit", False),
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )
        result["model"] = teacher
        dt = time.monotonic() - t0
        log.info("Teacher preload complete in %.1fs (on CPU)", dt)
    except Exception as exc:
        log.warning("Teacher preload failed (%s) — will fall back to direct load", exc)


# ---------------------------------------------------------------------------
# Background GGUF conversion (Optimization #8)
# ---------------------------------------------------------------------------

def _background_gguf_convert(icfg: dict, artifacts_dir: Path, result: dict) -> None:
    """Convert the student model to F16 GGUF in background (CPU-bound)."""
    import shutil
    import subprocess

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
    try:
        t0 = time.monotonic()
        subprocess.run(
            [sys.executable, str(convert_py), str(model_dir),
             "--outtype", "f16", "--outfile", str(f16_tmp)],
            check=True, timeout=3600,
        )
        os.replace(f16_tmp, f16_path)
        dt = time.monotonic() - t0
        result["f16_path"] = f16_path
        log.info("GGUF convert (background): done in %.1fs (%.1f GB)",
                 dt, f16_path.stat().st_size / 1e9)
    except Exception as exc:
        log.warning("GGUF convert (background): failed (%s)", exc)
        f16_tmp.unlink(missing_ok=True)


def _run_llama_imatrix_with_prebuilt_gguf(
    texts: list[str], icfg: dict, artifacts_dir: Path, gguf_result: dict,
) -> None:
    """Run llama-imatrix using the pre-built F16 GGUF from background thread."""
    import subprocess

    if not icfg.get("enabled", True):
        return

    calib_path = artifacts_dir / "calibration_imatrix.txt"
    joined = "\n\n".join(t.strip() for t in texts if t and t.strip())
    calib_path.write_text(joined, encoding="utf-8")
    log.info("imatrix: calibration file written (%d docs, %d chars) → %s",
             len(texts), len(joined), calib_path)

    f16_path = gguf_result.get("f16_path")
    if f16_path is None or not Path(f16_path).exists():
        log.warning("imatrix: pre-built GGUF not available — falling back to full pipeline")
        _generate_imatrix(texts, icfg, artifacts_dir)
        return

    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
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
    try:
        subprocess.run(
            [str(imatrix_bin),
             "-m", str(f16_path), "-f", str(calib_path),
             "-o", str(imatrix_out), "--output-format", "gguf",
             "--no-ppl", "-ngl", str(ngl), "-c", str(ctx)],
            env=env, check=True, timeout=7200,
        )
        log.info("imatrix: saved (%.1f MB)", imatrix_out.stat().st_size / 1e6)
    except Exception as exc:
        log.warning("imatrix: llama-imatrix failed (%s). Calibration text at %s.", exc, calib_path)


# ---------------------------------------------------------------------------
# WikiText-2 perplexity (Optimization #1: configurable batch_size)
# ---------------------------------------------------------------------------


def _wikitext2_ppl(model, tokenizer, cfg: dict, *, device=None, collect=None,
                   batch_size: int = 8) -> float:
    """Standard next-token NLL → exp(mean_NLL), seq_len=2048.

    Batching doesn't change NLL computation — each sequence is scored
    independently; out.loss is the mean over tokens in each batch element,
    and we scale by (batch.numel() - batch.shape[0]) to recover the sum.
    Numerically identical to batch_size=1.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])
    eos = tokenizer.eos_token_id or 0
    all_ids: list[int] = []
    for row in ds:
        text = row.get("text", "")
        if not text.strip():
            continue
        if collect is not None:
            collect.append(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        # Avoid double-EOS at document boundaries when the tokenizer already
        # appends EOS as part of the text encoding.
        if not ids or ids[-1] != eos:
            all_ids.append(eos)

    seq_len = cfg["sequence_length"]
    n_full = len(all_ids) // seq_len
    if n_full == 0:
        log.warning("WikiText-2 has no full-length sequences; returning inf.")
        return float("inf")
    chunks = torch.tensor(all_ids[: n_full * seq_len], dtype=torch.long).view(n_full, seq_len)

    model.eval()
    nll_sum = 0.0
    tok_count = 0
    log.info("Stage 6 PPL: %d sequences × len=%d, batch_size=%d", n_full, seq_len, batch_size)
    with torch.no_grad():
        for i, batch in enumerate(iter_batches(chunks, batch_size=batch_size)):
            if device is not None:
                batch = batch.to(device)
            out = model(input_ids=batch, labels=batch)
            nll = float(out.loss.item()) * (batch.numel() - batch.shape[0])
            nll_sum += nll
            tok_count += batch.numel() - batch.shape[0]
            if (i + 1) % max(1, 64 // batch_size) == 0:
                log.info("  PPL forward %d/%d batches (%d/%d seqs)",
                         i + 1, math.ceil(n_full / batch_size), min((i + 1) * batch_size, n_full), n_full)
    if tok_count == 0:
        return float("inf")
    return math.exp(nll_sum / tok_count)


# ---------------------------------------------------------------------------
# Zero-shot (ARC-C + HellaSwag) via lm-eval (Optimization #2: batch_size=auto:8)
# ---------------------------------------------------------------------------


def _lm_eval_tasks(model, tokenizer, tasks: list[str], *, collect=None,
                   batch_size="auto:8") -> dict:
    """Delegate to lm-eval's simple_evaluate with configurable batch_size.

    lm-eval's 0-shot loglikelihood scoring is deterministic and batch-size-
    independent. Numerically identical to batch_size=1.
    """
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except Exception as err:           # noqa: BLE001
        log.warning("lm-eval not available (%s); skipping zero-shot.", err)
        return {}

    try:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        out = simple_evaluate(
            model=lm, tasks=list(tasks), num_fewshot=0,
            log_samples=(collect is not None),
        )
        results = out.get("results", {})
        flat: dict = {}
        for task, metrics in results.items():
            # ARC-C canonical metric is acc_norm,none (normalized); prefer it first.
            # Use key-existence check (not truthiness) so acc=0.0 is not skipped.
            for _k in ("acc_norm,none", "acc,none", "acc"):
                if _k in metrics:
                    acc = metrics[_k]
                    break
            else:
                acc = None
            if acc is not None:
                flat[f"{task}_acc"] = float(acc)
        if collect is not None and "samples" in out:
            for task_samples in out["samples"].values():
                seen: set[str] = set()
                for s in task_samples:
                    try:
                        args = s.get("arguments", ())
                        ctx = args[0] if args else None
                        if ctx and isinstance(ctx, str) and ctx not in seen:
                            seen.add(ctx)
                            collect.append(ctx)
                    except (KeyError, IndexError, TypeError):
                        pass
        return flat
    except Exception as err:           # noqa: BLE001
        log.warning("lm-eval evaluation failed: %s", err)
        return {}


# ---------------------------------------------------------------------------
# Batched generation (Optimizations #3, #4)
# ---------------------------------------------------------------------------


def _generate_batched(model, tokenizer, prompts: list[str], *, max_new: int,
                      device, batch_size: int = 8) -> list[str]:
    """Batched model.generate() for greedy decoding (do_sample=False).

    Left-pads prompts to the longest in each batch group. Greedy decoding
    produces deterministic outputs regardless of batching.
    Numerically identical to serial generation.
    """
    original_padding_side = tokenizer.padding_side
    original_pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: list[str] = [""] * len(prompts)
    model.eval()

    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            encoded = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                truncation=False, add_special_tokens=True,
            )
            if device is not None:
                encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = encoded["input_ids"].shape[1]  # padded width, same for all in batch
            for j in range(len(batch_prompts)):
                # Slice from input_len (not attention_mask.sum()) because with left-padding
                # the shorter prompts have prompt_len < padded_len, causing out[j, prompt_len:]
                # to include trailing pad tokens from the input as "generated" tokens.
                generated_ids = out[j, input_len:]
                results[i + j] = tokenizer.decode(generated_ids, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.pad_token_id = original_pad_token_id

    return results


# ---------------------------------------------------------------------------
# Generative — HumanEval pass@1, MATH-500 accuracy
# ---------------------------------------------------------------------------


def _humaneval(model, tokenizer, cfg: dict, *, device=None, collect=None,
               batch_size: int = 8) -> float:
    try:
        from datasets import load_dataset
    except Exception as err:           # noqa: BLE001
        log.warning("datasets not available (%s); skipping HumanEval.", err)
        return float("nan")
    try:
        ds = load_dataset("openai_humaneval", split="test")
    except Exception as err:           # noqa: BLE001
        log.warning("HumanEval dataset load failed (%s); skipping.", err)
        return float("nan")

    max_new = int(cfg.get("max_new_tokens", 512))
    model.eval()

    prompts = [row["prompt"] for row in ds]
    tests = [row["test"] for row in ds]
    entry_points = [row["entry_point"] for row in ds]

    if collect is not None:
        collect.extend(prompts)

    log.info("Stage 6 HumanEval: %d problems, batch_size=%d", len(prompts), batch_size)
    completions = _generate_batched(
        model, tokenizer, prompts, max_new=max_new,
        device=device, batch_size=batch_size,
    )

    passes = 0
    total = len(prompts)
    for i, (prompt, completion, test, ep) in enumerate(
        zip(prompts, completions, tests, entry_points)
    ):
        if _check_humaneval(prompt, completion, test, ep):
            passes += 1
        if (i + 1) % 16 == 0:
            log.info("  HumanEval eval %d/%d (pass=%d)", i + 1, total, passes)
    log.info("  HumanEval final: %d/%d = %.3f", passes, total, passes / max(total, 1))
    return passes / max(total, 1)


def _math500(model, tokenizer, cfg: dict, *, device=None, collect=None,
             batch_size: int = 8) -> float:
    try:
        from datasets import load_dataset
    except Exception as err:           # noqa: BLE001
        log.warning("datasets not available; skipping MATH-500.")
        return float("nan")
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as err:           # noqa: BLE001
        log.warning("MATH-500 dataset load failed (%s); skipping.", err)
        return float("nan")

    max_new = int(cfg.get("max_new_tokens", 1024))
    n = int(cfg.get("num_samples", 500))
    n_total = min(n, len(ds))
    model.eval()

    selected = ds.select(range(n_total))
    prompts = [f"Problem: {row['problem']}\nAnswer:" for row in selected]
    answers = [row.get("answer", "") for row in selected]

    if collect is not None:
        collect.extend(prompts)

    log.info("Stage 6 MATH-500: %d problems, batch_size=%d", n_total, batch_size)
    completions = _generate_batched(
        model, tokenizer, prompts, max_new=max_new,
        device=device, batch_size=batch_size,
    )

    correct = 0
    for i, (completion, answer) in enumerate(zip(completions, answers)):
        if _check_math(completion, answer):
            correct += 1
        if (i + 1) % 25 == 0:
            log.info("  MATH-500 eval %d/%d (correct=%d)", i + 1, n_total, correct)
    log.info("  MATH-500 final: %d/%d = %.3f", correct, n_total, correct / max(n_total, 1))
    return correct / max(n_total, 1)


def _check_humaneval(prompt: str, completion: str, test_src: str, entry_point: str) -> bool:
    src = prompt + completion + "\n" + test_src + f"\ncheck({entry_point})\n"
    try:
        ns: dict = {}
        exec(src, ns, ns)               # noqa: S102 — controlled benchmark use
        return True
    except Exception:                    # noqa: BLE001
        return False


def _check_math(completion: str, reference: str) -> bool:
    import re

    def _extract_boxed(s: str) -> str | None:
        # Balanced-brace extraction handles nested braces (e.g. \boxed{\frac{1}{2}}).
        # The simple regex [^}]* stops at the first } and misses nested LaTeX.
        results = []
        idx = 0
        while True:
            m = re.search(r'\\boxed\{', s[idx:])
            if not m:
                break
            start = idx + m.end()
            depth = 1
            i = start
            while i < len(s) and depth > 0:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                results.append(s[start:i - 1])
                idx = start
            else:
                # Unclosed \boxed{ — truncated output; stop scanning to avoid
                # misidentifying nested \boxed{} inside the open group as top-level.
                break
        return results[-1] if results else None

    def _last_numeric(s: str) -> str | None:
        nums = re.findall(r"-?\d+\.?\d*", s)
        return nums[-1] if nums else None

    comp_answer = _extract_boxed(completion)
    ref_answer = _extract_boxed(reference)
    if comp_answer is None:
        comp_answer = _last_numeric(completion)
    if ref_answer is None:
        ref_answer = _last_numeric(reference)

    if comp_answer is None or ref_answer is None:
        return False
    if comp_answer.strip() == ref_answer.strip():
        return True

    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex
        try:
            a = parse_latex(comp_answer)
        except Exception:
            a = sympify(comp_answer)
        try:
            b = parse_latex(ref_answer)
        except Exception:
            b = sympify(ref_answer)
        return bool(simplify(a - b) == 0)
    except Exception:
        pass

    a = _last_numeric(comp_answer)
    b = _last_numeric(ref_answer)
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Deltas + threshold check
# ---------------------------------------------------------------------------


def _deltas(student: dict, teacher: dict) -> dict:
    # delta = student - teacher: positive means student is worse for PPL
    # (higher is worse), negative means student is worse for accuracy tasks
    # (lower is worse). _check_thresholds interprets each metric's sign.
    out = {}
    for k in set(student) | set(teacher):
        s = student.get(k)
        t = teacher.get(k)
        if s is None or t is None:
            continue
        out[k] = {"student": s, "teacher": t, "delta": s - t}
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
    s_total = student_total if student_total is not None else count_parameters(student_model)
    s_expert = student_expert if student_expert is not None else count_expert_parameters(student_model, routed_only=True)

    if teacher_model is not None:
        t_total = count_parameters(teacher_model)
        t_expert = count_expert_parameters(teacher_model, routed_only=True)
    elif cached_teacher_param_counts is not None:
        t_total = cached_teacher_param_counts["total"]
        t_expert = cached_teacher_param_counts["expert"]
        log.info("Using cached teacher param counts: total=%d, expert=%d", t_total, t_expert)
    else:
        log.info("Computing teacher param counts via CPU model load")
        try:
            teacher_tmp, _ = load_model(
                config["model"]["name_or_path"],
                revision=config["model"].get("revision", "main"),
                torch_dtype=config["model"]["torch_dtype"],
                device_map="cpu",
                attn_implementation=config["model"]["attn_implementation"],
                load_in_4bit=config["model"].get("load_in_4bit", False),
                trust_remote_code=config["model"].get("trust_remote_code", False),
            )
            t_total = count_parameters(teacher_tmp)
            t_expert = count_expert_parameters(teacher_tmp, routed_only=True)
            del teacher_tmp
        except Exception as exc:
            log.warning("Could not load teacher for param counting (%s) — using 0", exc)
            t_total = 0
            t_expert = 0

    return {
        "total_student": s_total,
        "total_teacher": t_total,
        "total_reduction_ratio": 1.0 - (s_total / max(t_total, 1)),
        "expert_student": s_expert,
        "expert_teacher": t_expert,
        "expert_reduction_ratio": 1.0 - (s_expert / max(t_expert, 1)),
    }


# ---------------------------------------------------------------------------
# imatrix calibration + GGUF conversion (full sequential path)
# ---------------------------------------------------------------------------


def _generate_imatrix(texts: list[str], icfg: dict, artifacts_dir: Path) -> None:
    import shutil
    import subprocess

    if not icfg.get("enabled", True):
        log.info("imatrix: disabled via config.")
        return

    calib_path = artifacts_dir / "calibration_imatrix.txt"
    joined = "\n\n".join(t.strip() for t in texts if t and t.strip())
    calib_path.write_text(joined, encoding="utf-8")
    log.info("imatrix: calibration file written (%d docs, %d chars) → %s",
             len(texts), len(joined), calib_path)

    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
        log.warning("imatrix: llama.cpp not found. Calibration text preserved at %s.", calib_path)
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
    try:
        f16_tmp.unlink(missing_ok=True)
        subprocess.run(
            [sys.executable, str(convert_py), str(model_dir),
             "--outtype", "f16", "--outfile", str(f16_tmp)],
            check=True, timeout=3600,
        )
        os.replace(f16_tmp, f16_path)
        log.info("imatrix: GGUF ready (%.1f GB)", f16_path.stat().st_size / 1e9)
    except Exception as exc:  # noqa: BLE001
        f16_tmp.unlink(missing_ok=True)
        log.warning("imatrix: GGUF conversion failed (%s); skipping.", exc)
        return

    imatrix_out = artifacts_dir / "imatrix.gguf"
    ngl = int(icfg.get("ngl", 99))
    ctx = int(icfg.get("ctx_size", 2048))
    log.info("imatrix: running llama-imatrix (ngl=%d, ctx=%d) → %s", ngl, ctx, imatrix_out)
    try:
        subprocess.run(
            [str(imatrix_bin),
             "-m", str(f16_path), "-f", str(calib_path),
             "-o", str(imatrix_out), "--output-format", "gguf",
             "--no-ppl", "-ngl", str(ngl), "-c", str(ctx)],
            env=env, check=True, timeout=7200,
        )
        log.info("imatrix: saved (%.1f MB)", imatrix_out.stat().st_size / 1e6)
    except Exception as exc:  # noqa: BLE001
        log.warning("imatrix: llama-imatrix failed (%s). Calibration text at %s.", exc, calib_path)


def _find_llama_cpp_dir(override: str | None = None) -> Path | None:
    import shutil

    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    on_path = shutil.which("llama-imatrix")
    if on_path:
        candidates.append(Path(on_path).parent.parent.parent)
    candidates.append(Path("/home/lucas/ai/tools/llama.cpp"))

    for p in candidates:
        if (p / "build" / "bin" / "llama-imatrix").exists() and (p / "convert_hf_to_gguf.py").exists():
            return p
    return None


def _check_thresholds(results: dict, thresholds: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    wt = results["delta"].get("wikitext2_ppl")
    if wt is not None:
        rel = (wt["student"] - wt["teacher"]) / max(wt["teacher"], 1e-9)
        checks["wikitext2_ppl_increase_ok"] = rel <= thresholds["wikitext2_ppl_relative_max_increase"]
    for task, key, thresh in [
        ("arc_challenge_acc", "arc_c_absolute_max_drop", thresholds["arc_c_absolute_max_drop"]),
        ("hellaswag_acc", "hellaswag_absolute_max_drop", thresholds["hellaswag_absolute_max_drop"]),
        ("humaneval_pass_at_1", "humaneval_absolute_max_drop", thresholds["humaneval_absolute_max_drop"]),
        ("math500_accuracy", "math500_absolute_max_drop", thresholds["math500_absolute_max_drop"]),
    ]:
        d = results["delta"].get(task)
        if d is not None:
            drop = d["teacher"] - d["student"]
            checks[f"{task}_drop_ok"] = drop <= thresh
        else:
            log.warning("Threshold check for %s failed — metric missing from results "
                        "(lm-eval task name mismatch or evaluation error)", task)
            checks[f"{task}_drop_ok"] = False
    mr = results["measured_reduction"]["total_reduction_ratio"]
    checks["measured_reduction_ok"] = mr >= thresholds["measured_reduction_min"]
    return checks
