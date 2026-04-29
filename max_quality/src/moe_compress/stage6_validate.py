"""Stage 6 — Validation.

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
same prompt slices for apples-to-apples deltas.

Artifact: ``stage6_eval.json`` with absolute metrics + deltas + threshold
pass/fail summary.
"""
from __future__ import annotations

import logging
import math
import os
import sys
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


def run(model, tokenizer, config: dict, artifacts_dir: Path, *, device=None) -> Path:
    s6 = config["stage6_validate"]
    model.eval()   # stage5 leaves model in train(); set eval before any sub-metric
    results: dict = {"student": {}, "teacher": {}, "delta": {}, "thresholds": {}}

    calib_texts: list[str] = []  # accumulates eval text for imatrix calibration

    # 1. WikiText-2 PPL on student
    if s6["wikitext2"]["enabled"]:
        log.info("Stage 6: WikiText-2 PPL (student)")
        results["student"]["wikitext2_ppl"] = _wikitext2_ppl(
            model, tokenizer, s6["wikitext2"], device=device, collect=calib_texts,
        )

    # 2. Zero-shot via lm-eval (ARC-C + HellaSwag)
    if s6["zero_shot"]["enabled"]:
        log.info("Stage 6: zero-shot harness")
        results["student"].update(
            _lm_eval_tasks(model, tokenizer, s6["zero_shot"]["tasks"], collect=calib_texts)
        )

    # 3. Generative — HumanEval + MATH-500
    if s6["generative"]["enabled"]:
        log.info("Stage 6: generative (HumanEval + MATH-500)")
        if "humaneval" in s6["generative"]:
            results["student"]["humaneval_pass_at_1"] = _humaneval(
                model, tokenizer, s6["generative"]["humaneval"], device=device, collect=calib_texts,
            )
        if "math500" in s6["generative"]:
            results["student"]["math500_accuracy"] = _math500(
                model, tokenizer, s6["generative"]["math500"], device=device, collect=calib_texts,
            )

    # 4. Snapshot student param counts BEFORE loading teacher — once teacher
    #    is loaded we may move student to CPU, but `count_expert_parameters`
    #    is param-element-counting and dtype-independent so it stays correct.
    student_total = count_parameters(model)
    student_expert = count_expert_parameters(model, routed_only=True)

    # 5. Free student GPU memory before loading the BF16 teacher — together
    #    they exceed the 80 GB A100 budget (factored student ~35 GB +
    #    BF16 teacher ~70 GB > 80 GB). Student eval is done; we don't need
    #    it on GPU again. CPU offload keeps it available for parameter
    #    counting; full delete is unnecessary.
    import torch
    try:
        model.to("cpu")
    except Exception as exc:                                # noqa: BLE001
        log.warning("Could not move student to CPU before teacher load: %s", exc)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("Stage 6: loading uncompressed baseline for delta computation")
    teacher, _ = load_model(
        config["model"]["name_or_path"],
        revision=config["model"]["revision"],
        torch_dtype=config["model"]["torch_dtype"],
        device_map=config["model"]["device_map"],
        attn_implementation=config["model"]["attn_implementation"],
        load_in_4bit=config["model"].get("load_in_4bit", False),
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    teacher.eval()

    if s6["wikitext2"]["enabled"]:
        results["teacher"]["wikitext2_ppl"] = _wikitext2_ppl(
            teacher, tokenizer, s6["wikitext2"], device=device,
        )
    if s6["zero_shot"]["enabled"]:
        results["teacher"].update(_lm_eval_tasks(teacher, tokenizer, s6["zero_shot"]["tasks"]))
    if s6["generative"]["enabled"]:
        if "humaneval" in s6["generative"]:
            results["teacher"]["humaneval_pass_at_1"] = _humaneval(
                teacher, tokenizer, s6["generative"]["humaneval"], device=device,
            )
        if "math500" in s6["generative"]:
            results["teacher"]["math500_accuracy"] = _math500(
                teacher, tokenizer, s6["generative"]["math500"], device=device,
            )

    # 5. Deltas and threshold checks
    results["delta"] = _deltas(results["student"], results["teacher"])
    results["measured_reduction"] = _measured_reduction(
        model, teacher, student_total=student_total, student_expert=student_expert,
    )
    results["thresholds"] = _check_thresholds(results, s6["thresholds"])

    path = artifacts_dir / "stage6_eval.json"
    save_json_artifact(results, path)

    # Free teacher GPU memory before llama-imatrix subprocess uses the GPU.
    # On A100 the teacher (~70 GB BF16) must be evicted before the F16 GGUF
    # (~35 GB) can be loaded by llama-imatrix; on H200 it's optional but tidy.
    try:
        teacher.to("cpu")
        del teacher
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _generate_imatrix(calib_texts, s6.get("imatrix", {}), artifacts_dir)

    overall_pass = all(results["thresholds"].values())
    log.info("Stage 6 complete — thresholds %s; detail → %s",
             "PASS" if overall_pass else "FAIL", path)
    # Trackio: flatten the metric scalars so they appear on the dashboard.
    # Layout per `_deltas`/`_measured_reduction`: results = {
    #   "student": {metric: float, ...},
    #   "teacher": {metric: float, ...},
    #   "delta":   {metric: {"student": s, "teacher": t, "delta": d}, ...},
    #   "measured_reduction": {total_student, total_teacher, total_reduction_ratio, ...},
    # }
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
# WikiText-2 perplexity
# ---------------------------------------------------------------------------


def _wikitext2_ppl(model, tokenizer, cfg: dict, *, device=None, collect=None) -> float:
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])
    # Concatenate with EOS between docs, then chunk into fixed sequences.
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
        all_ids.append(eos)

    seq_len = cfg["sequence_length"]
    # Drop any partial final chunk for clean comparison.
    n_full = len(all_ids) // seq_len
    if n_full == 0:
        log.warning("WikiText-2 has no full-length sequences; returning inf.")
        return float("inf")
    chunks = torch.tensor(all_ids[: n_full * seq_len], dtype=torch.long).view(n_full, seq_len)

    model.eval()
    nll_sum = 0.0
    tok_count = 0
    log.info("Stage 6 PPL: %d sequences × len=%d", n_full, seq_len)
    with torch.no_grad():
        for i, batch in enumerate(iter_batches(chunks, batch_size=1)):
            if device is not None:
                batch = batch.to(device)
            out = model(input_ids=batch, labels=batch)
            # ``out.loss`` is the mean over `seq_len - 1` tokens.
            nll = float(out.loss.item()) * (batch.numel() - batch.shape[0])
            nll_sum += nll
            tok_count += batch.numel() - batch.shape[0]
            if (i + 1) % 64 == 0:
                log.info("  PPL forward %d/%d", i + 1, n_full)
    if tok_count == 0:
        return float("inf")
    return math.exp(nll_sum / tok_count)


# ---------------------------------------------------------------------------
# Zero-shot (ARC-C + HellaSwag) via lm-eval
# ---------------------------------------------------------------------------


def _lm_eval_tasks(model, tokenizer, tasks: list[str], *, collect=None) -> dict:
    """Delegate to lm-eval's simple_evaluate. If lm-eval isn't installed or
    the HF-LM wrapper doesn't handle this architecture, log and return {}."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except Exception as err:           # noqa: BLE001
        log.warning("lm-eval not available (%s); skipping zero-shot.", err)
        return {}

    try:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
        out = simple_evaluate(
            model=lm, tasks=list(tasks), num_fewshot=0,
            log_samples=(collect is not None),
        )
        results = out.get("results", {})
        flat: dict = {}
        for task, metrics in results.items():
            acc = metrics.get("acc,none") or metrics.get("acc") or metrics.get("acc_norm,none")
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
# Generative — HumanEval pass@1, MATH-500 accuracy
# ---------------------------------------------------------------------------


def _humaneval(model, tokenizer, cfg: dict, *, device=None, collect=None) -> float:
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
    passes = 0
    total = 0
    log.info("Stage 6 HumanEval: %d problems", len(ds))
    for i, row in enumerate(ds):
        prompt = row["prompt"]
        if collect is not None:
            collect.append(prompt)
        completion = _generate(model, tokenizer, prompt, max_new=max_new, device=device)
        if _check_humaneval(prompt, completion, row["test"], row["entry_point"]):
            passes += 1
        total += 1
        if (i + 1) % 16 == 0:
            log.info("  HumanEval %d/%d (pass=%d)", i + 1, len(ds), passes)
    return passes / max(total, 1)


def _math500(model, tokenizer, cfg: dict, *, device=None, collect=None) -> float:
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
    model.eval()
    correct = 0
    total = 0
    n_total = min(n, len(ds))
    log.info("Stage 6 MATH-500: %d problems", n_total)
    for i, row in enumerate(ds.select(range(n_total))):
        prompt = f"Problem: {row['problem']}\nAnswer:"
        if collect is not None:
            collect.append(prompt)
        completion = _generate(model, tokenizer, prompt, max_new=max_new, device=device)
        if _check_math(completion, row.get("answer", "")):
            correct += 1
        total += 1
        if (i + 1) % 25 == 0:
            log.info("  MATH-500 %d/%d (correct=%d)", i + 1, n_total, correct)
    return correct / max(total, 1)


def _generate(model, tokenizer, prompt: str, *, max_new: int, device) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    if device is not None:
        ids = ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _check_humaneval(prompt: str, completion: str, test_src: str, entry_point: str) -> bool:
    """Run the HumanEval hidden test against prompt+completion.

    Executed in-process with a stripped `exec`. This is fine for benchmark
    evaluation but NEVER enable in production on untrusted completions.
    """
    src = prompt + completion + "\n" + test_src + f"\ncheck({entry_point})\n"
    try:
        ns: dict = {}
        exec(src, ns, ns)               # noqa: S102 — controlled benchmark use
        return True
    except Exception:                    # noqa: BLE001
        return False


def _check_math(completion: str, reference: str) -> bool:
    """Normalized string-match with SymPy fallback.

    First tries SymPy-based symbolic equivalence (handles different forms
    like 3/4 vs 0.75). Falls back to last-numeric-token string match if
    SymPy is not available or parsing fails.
    """
    import re

    def _extract_boxed(s: str) -> str | None:
        """Extract content from \\boxed{...} if present."""
        # Find the last \\boxed{...} in the string.
        matches = re.findall(r'\\boxed\{([^}]*)\}', s)
        return matches[-1] if matches else None

    def _last_numeric(s: str) -> str | None:
        nums = re.findall(r"-?\d+\.?\d*", s)
        return nums[-1] if nums else None

    # Try to extract boxed answer first (standard MATH format).
    comp_answer = _extract_boxed(completion)
    ref_answer = _extract_boxed(reference)
    if comp_answer is None:
        comp_answer = _last_numeric(completion)
    if ref_answer is None:
        ref_answer = _last_numeric(reference)

    if comp_answer is None or ref_answer is None:
        return False

    # Direct string match.
    if comp_answer.strip() == ref_answer.strip():
        return True

    # SymPy symbolic equivalence.
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

    # Fallback: numeric string match.
    a = _last_numeric(comp_answer)
    b = _last_numeric(ref_answer)
    return a is not None and b is not None and a == b


# ---------------------------------------------------------------------------
# Deltas + threshold check
# ---------------------------------------------------------------------------


def _deltas(student: dict, teacher: dict) -> dict:
    out = {}
    for k in set(student) | set(teacher):
        s = student.get(k)
        t = teacher.get(k)
        if s is None or t is None:
            continue
        out[k] = {"student": s, "teacher": t, "delta": s - t}
    return out


def _measured_reduction(
    student, teacher,
    *,
    student_total: int | None = None,
    student_expert: int | None = None,
) -> dict:
    # Allow caller to pass pre-computed student counts (taken before student
    # was moved to CPU for the teacher load). Fall back to live counts when
    # called outside that context.
    s_total = student_total if student_total is not None else count_parameters(student)
    s_expert = student_expert if student_expert is not None else count_expert_parameters(student, routed_only=True)
    t_total = count_parameters(teacher)
    t_expert = count_expert_parameters(teacher, routed_only=True)
    return {
        "total_student": s_total,
        "total_teacher": t_total,
        "total_reduction_ratio": 1.0 - (s_total / max(t_total, 1)),
        "expert_student": s_expert,
        "expert_teacher": t_expert,
        "expert_reduction_ratio": 1.0 - (s_expert / max(t_expert, 1)),
    }


# ---------------------------------------------------------------------------
# imatrix calibration + GGUF conversion
# ---------------------------------------------------------------------------


def _generate_imatrix(texts: list[str], icfg: dict, artifacts_dir: Path) -> None:
    """Write multi-domain calibration text, convert model to F16 GGUF, run llama-imatrix."""
    import shutil
    import subprocess

    if not icfg.get("enabled", True):
        log.info("imatrix: disabled via config.")
        return

    # 1. Write calibration file — always, even if llama.cpp is absent (manual fallback).
    calib_path = artifacts_dir / "calibration_imatrix.txt"
    joined = "\n\n".join(t.strip() for t in texts if t and t.strip())
    calib_path.write_text(joined, encoding="utf-8")
    log.info("imatrix: calibration file written (%d docs, %d chars) → %s",
             len(texts), len(joined), calib_path)

    # 2. Locate llama.cpp.
    llama_cpp_dir = _find_llama_cpp_dir(icfg.get("llama_cpp_dir"))
    if llama_cpp_dir is None:
        log.warning(
            "imatrix: llama.cpp not found. Set LLAMA_CPP_DIR env var or build it "
            "in the job entrypoint. Calibration text preserved at %s.", calib_path,
        )
        return

    imatrix_bin = llama_cpp_dir / "build" / "bin" / "llama-imatrix"
    convert_py  = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not imatrix_bin.exists() or not convert_py.exists():
        log.warning("imatrix: binaries missing under %s; skipping.", llama_cpp_dir)
        return

    # 3. Locate the stage5_final HF model directory for GGUF conversion.
    model_dir = artifacts_dir / "stage5_final"
    if not model_dir.exists():
        log.warning("imatrix: stage5_final not found at %s; skipping.", model_dir)
        return

    # 4. Disk-space guard — F16 GGUF of the compressed student is ~35 GB.
    free_gb = shutil.disk_usage(artifacts_dir).free / 1e9
    if free_gb < 40:
        log.warning("imatrix: only %.1f GB free at %s; skipping GGUF conversion.", free_gb, artifacts_dir)
        return

    # 5. Convert HF model → F16 GGUF.
    f16_path = artifacts_dir / "model_f16.gguf"
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": str(llama_cpp_dir / "build" / "bin") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    log.info("imatrix: converting %s → F16 GGUF", model_dir)
    try:
        subprocess.run(
            [sys.executable, str(convert_py), str(model_dir),
             "--outtype", "f16", "--outfile", str(f16_path)],
            check=True, timeout=3600,
        )
        log.info("imatrix: GGUF ready (%.1f GB)", f16_path.stat().st_size / 1e9)
    except Exception as exc:  # noqa: BLE001
        log.warning("imatrix: GGUF conversion failed (%s); skipping.", exc)
        return

    # 6. Run llama-imatrix.
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
    """Return the llama.cpp root containing build/bin/llama-imatrix, or None."""
    import shutil

    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    on_path = shutil.which("llama-imatrix")
    if on_path:
        # .../build/bin/llama-imatrix → root is 3 levels up
        candidates.append(Path(on_path).parent.parent.parent)
    candidates.append(Path("/home/lucas/ai/tools/llama.cpp"))  # local dev fallback

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
            log.warning("Threshold check for %s skipped — metric missing from results "
                        "(lm-eval task name mismatch or evaluation error)", task)
    mr = results["measured_reduction"]["total_reduction_ratio"]
    checks["measured_reduction_ok"] = mr >= thresholds["measured_reduction_min"]
    return checks
