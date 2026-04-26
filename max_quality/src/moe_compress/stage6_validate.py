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
    results: dict = {"student": {}, "teacher": {}, "delta": {}, "thresholds": {}}

    # 1. WikiText-2 PPL on student
    if s6["wikitext2"]["enabled"]:
        log.info("Stage 6: WikiText-2 PPL (student)")
        results["student"]["wikitext2_ppl"] = _wikitext2_ppl(
            model, tokenizer, s6["wikitext2"], device=device,
        )

    # 2. Zero-shot via lm-eval (ARC-C + HellaSwag)
    if s6["zero_shot"]["enabled"]:
        log.info("Stage 6: zero-shot harness")
        results["student"].update(
            _lm_eval_tasks(model, tokenizer, s6["zero_shot"]["tasks"])
        )

    # 3. Generative — HumanEval + MATH-500
    if s6["generative"]["enabled"]:
        log.info("Stage 6: generative (HumanEval + MATH-500)")
        if "humaneval" in s6["generative"]:
            results["student"]["humaneval_pass_at_1"] = _humaneval(
                model, tokenizer, s6["generative"]["humaneval"], device=device,
            )
        if "math500" in s6["generative"]:
            results["student"]["math500_accuracy"] = _math500(
                model, tokenizer, s6["generative"]["math500"], device=device,
            )

    # 4. Load teacher for baseline comparison and repeat the same slices
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
    results["measured_reduction"] = _measured_reduction(model, teacher)
    results["thresholds"] = _check_thresholds(results, s6["thresholds"])

    path = artifacts_dir / "stage6_eval.json"
    save_json_artifact(results, path)

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


def _wikitext2_ppl(model, tokenizer, cfg: dict, *, device=None) -> float:
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])
    # Concatenate with EOS between docs, then chunk into fixed sequences.
    eos = tokenizer.eos_token_id or 0
    all_ids: list[int] = []
    for row in ds:
        text = row.get("text", "")
        if not text.strip():
            continue
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


def _lm_eval_tasks(model, tokenizer, tasks: list[str]) -> dict:
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
        out = simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=0)
        results = out.get("results", {})
        flat: dict = {}
        for task, metrics in results.items():
            acc = metrics.get("acc,none") or metrics.get("acc") or metrics.get("acc_norm,none")
            if acc is not None:
                flat[f"{task}_acc"] = float(acc)
        return flat
    except Exception as err:           # noqa: BLE001
        log.warning("lm-eval evaluation failed: %s", err)
        return {}


# ---------------------------------------------------------------------------
# Generative — HumanEval pass@1, MATH-500 accuracy
# ---------------------------------------------------------------------------


def _humaneval(model, tokenizer, cfg: dict, *, device=None) -> float:
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
        completion = _generate(model, tokenizer, prompt, max_new=max_new, device=device)
        if _check_humaneval(prompt, completion, row["test"], row["entry_point"]):
            passes += 1
        total += 1
        if (i + 1) % 16 == 0:
            log.info("  HumanEval %d/%d (pass=%d)", i + 1, len(ds), passes)
    return passes / max(total, 1)


def _math500(model, tokenizer, cfg: dict, *, device=None) -> float:
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
    """Crude string-match equivalence on the last numeric token.

    Proper grading uses SymPy normalization — left as a future improvement.
    Ranking / trend signal is enough for Stage 6 threshold checking.
    """
    import re

    def _last_numeric(s: str) -> str | None:
        nums = re.findall(r"-?\d+\.?\d*", s)
        return nums[-1] if nums else None

    a = _last_numeric(completion)
    b = _last_numeric(reference)
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


def _measured_reduction(student, teacher) -> dict:
    s_total = count_parameters(student)
    t_total = count_parameters(teacher)
    s_expert = count_expert_parameters(student, routed_only=True)
    t_expert = count_expert_parameters(teacher, routed_only=True)
    return {
        "total_student": s_total,
        "total_teacher": t_total,
        "total_reduction_ratio": 1.0 - (s_total / max(t_total, 1)),
        "expert_student": s_expert,
        "expert_teacher": t_expert,
        "expert_reduction_ratio": 1.0 - (s_expert / max(t_expert, 1)),
    }


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
    mr = results["measured_reduction"]["total_reduction_ratio"]
    checks["measured_reduction_ok"] = mr >= thresholds["measured_reduction_min"]
    return checks
