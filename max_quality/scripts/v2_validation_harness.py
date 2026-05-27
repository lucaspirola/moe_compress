"""v2 calibration-mix validation harness — single-Job dual-phase.

VALIDATION HARNESS — NOT PART OF THE PRODUCTION CODEBASE.

Combines two validation tests that share one GPU rental:

    Phase A — R1-vs-Qwen3.6 router-pattern alignment (test 7.1 from
              tasks/CALIBRATION_MIX_V2_DESIGN.md). Samples 100 prompts
              from MoT-math/code/science, generates fresh thinking-mode
              completions with the teacher, and compares per-layer
              expert-routing top-k overlap between
              (prompt + canonical_R1_completion) and
              (prompt + teacher_generated_completion). Outputs a
              verdict (PASS / SOFT_WARN / HARD_FAIL) that gates the
              TEACHER_FORCED policy for MoT subsets in v2.

    Phase B — End-to-end smoke build of qwen3-pretrain-mix-v2 at
              --num-prompts 200 via build_self_traces_calib_vllm.py
              (test 7.4). Verifies JSONL shape: row count ~= 200, all
              12 subsets represented, completion_source distribution,
              schema_version=9, TEACHER_FORCED rows have
              n_gen_tokens=0 / _complete=True.

    Phase C — Upload a single JSON report (with optional gzipped
              smoke JSONL alongside) to the HF dataset
              pirola/calibration-v2-validation.

The harness is invoked by v2_validation_harness.sh which sets up the
venv, installs the patched vLLM wheel, and clones the repo at the
pinned commit (a3a946a). Phase A and Phase B each run in their own
subprocess so the teacher's VRAM is fully released between them
(neither vLLM's caching allocator nor CUDA-graph-pinned memory
survives a process exit). The harness re-loads the teacher twice (one
extra ~5-10 min) in exchange for OOM-free isolation between phases.

Wall-clock budget (rtx-pro-6000 / 96 GB BF16):
    * Teacher load A (BF16):                           ~5-10 min
    * Phase A (100 prompts × 1 gen + 2 forwards each): ~30-50 min
    * Teacher load B (BF16, in build-script subprocess): ~5-10 min
    * Phase B (smoke build, 200 prompts):              ~60-90 min
    * Phase C (upload):                                ~2 min
    * Total: ~2.5-3 hours; --timeout 4h on the Job has headroom.

Results
=======
Writes /tmp/v2_validation_results.json; the wrapper shell uploads it
to pirola/calibration-v2-validation. The smoke JSONL is also uploaded
(gzipped if > 5 MB) for archival.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

TEACHER_MODEL = os.environ.get("V2_VAL_TEACHER", "Qwen/Qwen3.6-35B-A3B")
PINNED_COMMIT = os.environ.get("V2_VAL_COMMIT", "a3a946a")

# Phase A sample sizes — total 100 prompts split per the design doc.
PHASE_A_N_MATH = 33
PHASE_A_N_CODE = 33
PHASE_A_N_SCIENCE = 34
PHASE_A_N_TOTAL = PHASE_A_N_MATH + PHASE_A_N_CODE + PHASE_A_N_SCIENCE

# Phase A generation budget. Qwen3.6 thinking-mode responses are long;
# 4096 captures the bulk of the routing signal without ballooning the
# wall-clock too much. The downstream forward passes use the realized
# (possibly shorter) generation length.
PHASE_A_GEN_MAX_TOKENS = 4096

# Phase A overlap thresholds (per task brief).
PHASE_A_PASS_MEDIAN = 0.5
PHASE_A_PASS_MIN_LAYER_MEDIAN = 0.2
PHASE_A_SOFT_WARN_MEDIAN = 0.3
PHASE_A_SOFT_WARN_MIN_LAYER_MEDIAN = 0.1
# HARD_FAIL: median < 0.3 OR >=3 layers with median < 0.1.

# Phase B parameters.
PHASE_B_NUM_PROMPTS = 200
PHASE_B_MAX_NEW_TOKENS = 16384
PHASE_B_OUTPUT_PATH = Path("/tmp/v2_smoke_self_traces.jsonl")
PHASE_B_LOG_PATH = Path("/tmp/v2_smoke_build.log")
PHASE_B_EXPECTED_SUBSETS = {
    "tulu3", "math", "qa", "creative", "multilingual", "fineweb", "papers",
    "mot_math", "mot_code", "mot_science", "swe_smith", "function_calling",
}

# Repo paths set up by the .sh wrapper. The wrapper clones the repo at
# PINNED_COMMIT to REPO_ROOT before running this script.
REPO_ROOT = Path(os.environ.get("V2_VAL_REPO_ROOT", "/tmp/moe_compress"))

# Output paths.
RESULTS_PATH = Path(os.environ.get("V2_VAL_RESULTS_PATH", "/tmp/v2_validation_results.json"))
RESULTS_REPO = os.environ.get("V2_VAL_RESULTS_REPO", "pirola/calibration-v2-validation")

# vLLM teacher load knobs. The Phase B build script defaults to BF16
# (no --quantization flag), so we match that here to keep both phases
# on the same dtype — same router patterns, same logits, no FP8↔BF16
# delta confounding Phase A's overlap metric.
VLLM_DTYPE = os.environ.get("V2_VAL_VLLM_DTYPE", "bfloat16")
VLLM_GPU_UTIL = float(os.environ.get("V2_VAL_GPU_UTIL", "0.85"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("V2_VAL_MAX_MODEL_LEN", "20480"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _env_banner() -> dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "vram_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            if torch.cuda.is_available() else None
        ),
    }
    try:
        import vllm  # noqa: F401
        info["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        info["vllm_import_error"] = repr(e)
    return info


# ---------------------------------------------------------------------------
# Phase A — R1 alignment
# ---------------------------------------------------------------------------


def _sample_mot_prompts(seed: int = 1337) -> list[dict[str, Any]]:
    """Pull (PHASE_A_N_MATH, PHASE_A_N_CODE, PHASE_A_N_SCIENCE) prompts
    from open-r1/Mixture-of-Thoughts via streaming.

    Each entry: {"subset": "math"|"code"|"science", "user": str, "canonical": str}.
    """
    from datasets import load_dataset  # noqa: WPS433 — late import (heavy)

    out: list[dict[str, Any]] = []
    for cfg, count in [
        ("math", PHASE_A_N_MATH),
        ("code", PHASE_A_N_CODE),
        ("science", PHASE_A_N_SCIENCE),
    ]:
        print(f"[{_now()}] phase-A: streaming MoT/{cfg} for {count} prompts", flush=True)
        ds = load_dataset(
            "open-r1/Mixture-of-Thoughts", cfg, split="train", streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=2048)
        taken = 0
        for row in ds:
            msgs = row.get("messages") or []
            if (
                len(msgs) >= 2
                and msgs[0].get("role") == "user"
                and msgs[1].get("role") == "assistant"
            ):
                u = (msgs[0].get("content") or "").strip()
                c = (msgs[1].get("content") or "").strip()
                if u and c:
                    out.append({"subset": cfg, "user": u, "canonical": c})
                    taken += 1
                    if taken >= count:
                        break
        print(f"[{_now()}] phase-A: MoT/{cfg} yielded {taken}", flush=True)
    return out


def _render_chat(tokenizer, user: str, assistant: str | None) -> str:
    msgs = [{"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    try:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=(assistant is None),
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=(assistant is None),
        )


def _generate_thinking(llm, tokenizer, user_prompt: str) -> str:
    """Generate a thinking-mode completion from a user prompt via vLLM.

    Returns the assistant-content string (with the <think>...</think>
    block intact if Qwen3.6 emits one).
    """
    from vllm import SamplingParams  # noqa: WPS433 — late import

    rendered = _render_chat(tokenizer, user_prompt, assistant=None)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=PHASE_A_GEN_MAX_TOKENS,
        seed=0,
    )
    out = llm.generate([rendered], sp, use_tqdm=False)
    return out[0].outputs[0].text


def _forward_capture_topk(
    llm, tokenizer, user: str, assistant: str,
) -> tuple[dict[int, torch.Tensor], int]:
    """Forward (prompt + assistant) through vLLM with the expert_in
    hook capturing per-(layer, token) topk_ids. We use generate(...,
    max_tokens=1) so the prefill phase processes every token in the
    rendered string in a single forward; the hook fires once per layer
    with topk_ids shaped [n_prefill_tokens, top_k].

    Returns (layer -> tensor[n_tokens, top_k], n_total_prefill_tokens).
    """
    from vllm import SamplingParams  # noqa: WPS433
    import vllm.calibration_hooks as ch  # noqa: WPS433 — patched module

    captures: dict[int, list[torch.Tensor]] = defaultdict(list)

    def on_expert_in(layer_idx, hidden_states, topk_ids, **_kw):
        # Only need topk_ids; keep on CPU/int64 for downstream set ops.
        captures[int(layer_idx)].append(
            topk_ids.detach().to("cpu", torch.long).clone()
        )

    ch._CAPTURE_EXPERT = True
    ch.register_callback("expert_in", on_expert_in)
    try:
        rendered = _render_chat(tokenizer, user, assistant=assistant)
        sp = SamplingParams(temperature=0.0, max_tokens=1, seed=0)
        _ = llm.generate([rendered], sp, use_tqdm=False)
    finally:
        ch.register_callback("expert_in", None)

    # Merge per-layer captures. Prefill emits a single tensor per layer
    # of shape [n_prefill_tokens, top_k]; the +1 generated token adds a
    # second small capture which we drop (we only care about prefill).
    per_layer: dict[int, torch.Tensor] = {}
    n_prefill = 0
    for layer_idx, lst in captures.items():
        if not lst:
            continue
        # Prefill is the first/largest capture; subsequent are
        # decode-step captures of shape [1, top_k] we ignore.
        prefill = max(lst, key=lambda t: t.shape[0])
        per_layer[layer_idx] = prefill
        n_prefill = max(n_prefill, prefill.shape[0])
    return per_layer, n_prefill


def _topk_set_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """For two top-k id vectors of shape [k], return |A∩B|/k."""
    sa = set(a.tolist())
    sb = set(b.tolist())
    k = max(len(sa), len(sb))
    if k == 0:
        return 0.0
    return len(sa & sb) / k


def _phase_a_run_prompt(
    llm, tokenizer, sample: dict[str, Any],
) -> dict[str, Any]:
    """Run one Phase-A prompt end-to-end. Returns per-layer overlap
    arrays + metadata for one (canonical, gen) pair.
    """
    user = sample["user"]
    canonical = sample["canonical"]

    # 1. Generate fresh thinking-mode completion from prompt-only.
    t_gen_start = time.perf_counter()
    gen = _generate_thinking(llm, tokenizer, user)
    t_gen = time.perf_counter() - t_gen_start

    # 2. Forward (prompt + canonical) and (prompt + gen) capturing topk.
    t_f1_start = time.perf_counter()
    topk_canon, n_canon = _forward_capture_topk(llm, tokenizer, user, canonical)
    t_f1 = time.perf_counter() - t_f1_start

    t_f2_start = time.perf_counter()
    topk_gen, n_gen = _forward_capture_topk(llm, tokenizer, user, gen)
    t_f2 = time.perf_counter() - t_f2_start

    # Align by token position over the SHORTER of the two prefills.
    # Routing comparison at position t: top-k set of canonical vs top-k
    # set of gen at the same position, even though the token at that
    # position differs (that's the point — different tokens, similar
    # contextual routing → high overlap).
    layers = sorted(set(topk_canon) & set(topk_gen))
    if not layers:
        return {
            "subset": sample["subset"],
            "error": "no MoE layers captured",
            "n_canonical_prefill": int(n_canon),
            "n_gen_prefill": int(n_gen),
            "t_gen_s": float(t_gen),
            "t_forward1_s": float(t_f1),
            "t_forward2_s": float(t_f2),
        }

    per_layer_median: dict[int, float] = {}
    per_layer_mean: dict[int, float] = {}
    per_layer_n_compared: dict[int, int] = {}
    for L in layers:
        a = topk_canon[L]  # [n_canon, k]
        b = topk_gen[L]    # [n_gen, k]
        n_compare = min(a.shape[0], b.shape[0])
        if n_compare == 0:
            continue
        a = a[:n_compare]
        b = b[:n_compare]
        # Per-position set overlap.
        overlaps = [_topk_set_overlap(a[t], b[t]) for t in range(n_compare)]
        ot = torch.tensor(overlaps, dtype=torch.float32)
        per_layer_median[L] = float(ot.median().item())
        per_layer_mean[L] = float(ot.mean().item())
        per_layer_n_compared[L] = int(n_compare)

    medians = list(per_layer_median.values())
    overall_median = float(torch.tensor(medians).median().item()) if medians else float("nan")
    overall_mean = float(torch.tensor(medians).mean().item()) if medians else float("nan")

    return {
        "subset": sample["subset"],
        "n_canonical_prefill": int(n_canon),
        "n_gen_prefill": int(n_gen),
        "n_layers_compared": len(layers),
        "per_layer_median": per_layer_median,
        "per_layer_mean": per_layer_mean,
        "per_layer_n_compared": per_layer_n_compared,
        "prompt_median_overlap": overall_median,
        "prompt_mean_of_median": overall_mean,
        "t_gen_s": float(t_gen),
        "t_forward1_s": float(t_f1),
        "t_forward2_s": float(t_f2),
        "gen_preview": gen[:240],
        "canonical_preview": canonical[:240],
    }


def _phase_a_verdict(
    layer_medians: dict[int, list[float]],
) -> tuple[str, dict[str, Any]]:
    """Aggregate per-layer medians across prompts → global verdict.

    layer_medians[layer] = [median_overlap_per_prompt_1, ..., per_prompt_N]
    """
    if not layer_medians:
        return "HARD_FAIL", {"reason": "no layers captured at all"}

    # Per-layer median-across-prompts of the per-prompt median overlap.
    per_layer = {
        L: float(torch.tensor(v, dtype=torch.float32).median().item())
        for L, v in layer_medians.items()
    }
    sorted_layers = sorted(per_layer)
    medians_array = [per_layer[L] for L in sorted_layers]
    overall_median = float(torch.tensor(medians_array, dtype=torch.float32).median().item())
    n_below_02 = sum(1 for v in medians_array if v < 0.2)
    n_below_01 = sum(1 for v in medians_array if v < 0.1)

    details = {
        "overall_median_across_layers": overall_median,
        "n_layers": len(medians_array),
        "n_layers_median_below_0.2": n_below_02,
        "n_layers_median_below_0.1": n_below_01,
        "per_layer_median": {str(L): v for L, v in per_layer.items()},
    }

    if overall_median < PHASE_A_SOFT_WARN_MEDIAN or n_below_01 >= 3:
        verdict = "HARD_FAIL"
        details["reason"] = (
            f"median={overall_median:.3f} < {PHASE_A_SOFT_WARN_MEDIAN} "
            f"OR n_layers(median<0.1)={n_below_01} >= 3"
        )
    elif (
        overall_median >= PHASE_A_PASS_MEDIAN
        and n_below_02 == 0
    ):
        verdict = "PASS"
        details["reason"] = (
            f"median={overall_median:.3f} >= {PHASE_A_PASS_MEDIAN} "
            f"AND all layers median >= 0.2"
        )
    else:
        verdict = "SOFT_WARN"
        details["reason"] = (
            f"median={overall_median:.3f} in [{PHASE_A_SOFT_WARN_MEDIAN}, "
            f"{PHASE_A_PASS_MEDIAN}) OR n_layers(median<0.2)={n_below_02} > 0"
        )
    return verdict, details


def run_phase_a(llm, tokenizer) -> dict[str, Any]:
    """Drive Phase A: sample 100 prompts, run alignment per prompt,
    aggregate, render verdict.
    """
    print(f"\n[{_now()}] === Phase A: R1 alignment ===", flush=True)
    t0 = time.perf_counter()
    try:
        samples = _sample_mot_prompts(seed=1337)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "FAILED_PROMPT_SAMPLING",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "verdict": "HARD_FAIL",
            "wall_clock_s": time.perf_counter() - t0,
        }

    print(f"[{_now()}] phase-A: {len(samples)} samples ready", flush=True)

    per_prompt: list[dict[str, Any]] = []
    layer_medians: dict[int, list[float]] = defaultdict(list)
    per_subset_layer_medians: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for i, sample in enumerate(samples):
        try:
            rec = _phase_a_run_prompt(llm, tokenizer, sample)
        except Exception as e:  # noqa: BLE001
            rec = {
                "subset": sample.get("subset"),
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
        per_prompt.append(rec)
        if "per_layer_median" in rec:
            for L, m in rec["per_layer_median"].items():
                layer_medians[int(L)].append(m)
                per_subset_layer_medians[rec["subset"]][int(L)].append(m)
        if (i + 1) % 10 == 0 or i + 1 == len(samples):
            avg = (
                sum(r.get("prompt_median_overlap", 0.0) for r in per_prompt if "prompt_median_overlap" in r)
                / max(1, sum(1 for r in per_prompt if "prompt_median_overlap" in r))
            )
            print(
                f"[{_now()}] phase-A: progress {i+1}/{len(samples)} | "
                f"running mean_of_medians={avg:.3f}",
                flush=True,
            )

    verdict, details = _phase_a_verdict(layer_medians)
    per_subset_verdict: dict[str, dict[str, Any]] = {}
    for subset, lm in per_subset_layer_medians.items():
        v, d = _phase_a_verdict(lm)
        per_subset_verdict[subset] = {"verdict": v, **d}

    print(f"\n[{_now()}] phase-A VERDICT: {verdict}", flush=True)
    print(f"  reason: {details.get('reason')}", flush=True)

    return {
        "status": "COMPLETED",
        "verdict": verdict,
        "global": details,
        "per_subset": per_subset_verdict,
        "per_prompt": per_prompt,
        "n_samples": len(samples),
        "wall_clock_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Phase B — smoke build
# ---------------------------------------------------------------------------


def run_phase_b() -> dict[str, Any]:
    """Drive Phase B: invoke build_self_traces_calib_vllm.py at the
    pinned commit, then sanity-check the produced JSONL.
    """
    print(f"\n[{_now()}] === Phase B: smoke build (--num-prompts 200) ===", flush=True)
    t0 = time.perf_counter()

    build_script = REPO_ROOT / "max_quality" / "scripts" / "build_self_traces_calib_vllm.py"
    if not build_script.exists():
        return {
            "status": "FAILED_BUILD_SCRIPT_MISSING",
            "expected_path": str(build_script),
            "wall_clock_s": time.perf_counter() - t0,
        }

    # The build script uses --no-cache-suffix so the output lands at
    # exactly PHASE_B_OUTPUT_PATH (no extra cache_key suffix).
    if PHASE_B_OUTPUT_PATH.exists():
        PHASE_B_OUTPUT_PATH.unlink()

    cmd = [
        sys.executable,
        str(build_script),
        "--teacher", TEACHER_MODEL,
        "--prompts", "qwen3-pretrain-mix-v2",
        "--num-prompts", str(PHASE_B_NUM_PROMPTS),
        "--max-new-tokens", str(PHASE_B_MAX_NEW_TOKENS),
        "--output", str(PHASE_B_OUTPUT_PATH),
        "--no-cache-suffix",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(VLLM_GPU_UTIL),
    ]
    print(f"[{_now()}] phase-B exec: {' '.join(cmd)}", flush=True)

    # Stream the subprocess output into a log file AND echo to stdout.
    PHASE_B_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Build script imports moe_compress.utils.calibration; the harness
    # PYTHONPATHs the cloned repo's src/ dir in the .sh wrapper, but
    # belt-and-suspenders here in case the script is invoked standalone.
    src_dir = str(REPO_ROOT / "max_quality" / "src")
    env["PYTHONPATH"] = (
        f"{src_dir}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_dir
    )

    try:
        with PHASE_B_LOG_PATH.open("w") as log_f:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, bufsize=1, universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            rc = proc.wait()
    except Exception as e:  # noqa: BLE001
        return {
            "status": "FAILED_SUBPROCESS_LAUNCH",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "wall_clock_s": time.perf_counter() - t0,
        }

    if rc != 0:
        return {
            "status": "FAILED_BUILD_SCRIPT_EXIT_NONZERO",
            "exit_code": rc,
            "log_tail": _read_log_tail(PHASE_B_LOG_PATH, n_lines=200),
            "wall_clock_s": time.perf_counter() - t0,
        }

    if not PHASE_B_OUTPUT_PATH.exists():
        return {
            "status": "FAILED_NO_OUTPUT_JSONL",
            "exit_code": rc,
            "log_tail": _read_log_tail(PHASE_B_LOG_PATH, n_lines=200),
            "wall_clock_s": time.perf_counter() - t0,
        }

    # ----- sanity checks -----
    rows = []
    with PHASE_B_OUTPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n_rows = len(rows)
    domain_counter = Counter(r.get("domain") for r in rows)
    completion_counter = Counter(r.get("completion_source") for r in rows)
    complete_counter = Counter(bool(r.get("_complete")) for r in rows)
    schema_counter: Counter[Any] = Counter()
    # The build script emits a manifest-style row OR a per-row schema_version
    # field; v9 writes per-row "completion_source" but schema_version is
    # carried in the cache-suffix manifest header (first .stats line printed
    # to log). Best-effort introspection: detect via field presence.
    schema_field_present = all("completion_source" in r for r in rows)

    # TEACHER_FORCED rows must have n_gen_tokens=0 and _complete=True.
    tf_rows = [r for r in rows if r.get("completion_source") == "canonical"]
    tf_violations = [
        {
            "domain": r.get("domain"),
            "n_gen_tokens": r.get("n_gen_tokens"),
            "_complete": r.get("_complete"),
        }
        for r in tf_rows
        if r.get("n_gen_tokens") != 0 or not r.get("_complete")
    ]

    # Policy distribution check: design says 56% GENERATE + 44% TF; allow
    # ±10pp wiggle for the small n=200 sample.
    n_gen_rows = completion_counter.get("teacher_generated", 0)
    n_tf_rows = completion_counter.get("canonical", 0)
    pct_gen = (n_gen_rows / n_rows * 100.0) if n_rows else 0.0
    pct_tf = (n_tf_rows / n_rows * 100.0) if n_rows else 0.0
    target_pct_gen, target_pct_tf = 56.0, 44.0
    policy_within_tolerance = (
        abs(pct_gen - target_pct_gen) <= 10.0
        and abs(pct_tf - target_pct_tf) <= 10.0
    )

    # Subset-coverage check: at least 1 row from every expected subset.
    subsets_seen = set(domain_counter)
    missing_subsets = sorted(PHASE_B_EXPECTED_SUBSETS - subsets_seen)

    verdict_parts: list[str] = []
    if n_rows < int(0.7 * PHASE_B_NUM_PROMPTS):
        verdict_parts.append(f"row_count_low: {n_rows}/{PHASE_B_NUM_PROMPTS}")
    if missing_subsets:
        verdict_parts.append(f"missing_subsets: {missing_subsets}")
    if tf_violations:
        verdict_parts.append(f"tf_violations: {len(tf_violations)}")
    if not policy_within_tolerance:
        verdict_parts.append(
            f"policy_dist_off: gen={pct_gen:.1f}% (target ~56±10%), "
            f"tf={pct_tf:.1f}% (target ~44±10%)"
        )
    if not schema_field_present:
        verdict_parts.append("completion_source field missing on some rows")

    verdict = "PASS" if not verdict_parts else "WARN"

    print(f"\n[{_now()}] phase-B VERDICT: {verdict}", flush=True)
    if verdict_parts:
        for p in verdict_parts:
            print(f"  ! {p}", flush=True)

    return {
        "status": "COMPLETED",
        "verdict": verdict,
        "verdict_issues": verdict_parts,
        "n_rows": n_rows,
        "domain_counts": dict(domain_counter),
        "completion_source_counts": dict(completion_counter),
        "complete_counts": {str(k): v for k, v in complete_counter.items()},
        "pct_teacher_generated": pct_gen,
        "pct_canonical": pct_tf,
        "policy_within_tolerance": policy_within_tolerance,
        "missing_subsets": missing_subsets,
        "n_teacher_forced_rows": len(tf_rows),
        "n_tf_violations": len(tf_violations),
        "tf_violations_sample": tf_violations[:10],
        "schema_field_completion_source_present_all_rows": schema_field_present,
        "smoke_jsonl_path": str(PHASE_B_OUTPUT_PATH),
        "smoke_jsonl_size_bytes": (
            PHASE_B_OUTPUT_PATH.stat().st_size if PHASE_B_OUTPUT_PATH.exists() else 0
        ),
        "wall_clock_s": time.perf_counter() - t0,
    }


def _read_log_tail(path: Path, n_lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [L.rstrip() for L in lines[-n_lines:]]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Phase C — upload
# ---------------------------------------------------------------------------


def run_phase_c(results: dict[str, Any]) -> dict[str, Any]:
    """Upload the JSON report (and gzipped smoke JSONL if present) to
    the HF dataset.
    """
    print(f"\n[{_now()}] === Phase C: upload to {RESULTS_REPO} ===", flush=True)
    t0 = time.perf_counter()
    try:
        from huggingface_hub import create_repo, upload_file  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        return {
            "status": "FAILED_HF_HUB_IMPORT",
            "error": repr(e),
            "wall_clock_s": time.perf_counter() - t0,
        }

    token = os.environ.get("HF_TOKEN")
    if not token:
        return {
            "status": "FAILED_NO_HF_TOKEN",
            "wall_clock_s": time.perf_counter() - t0,
        }

    try:
        create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True,
                    private=False, token=token)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "FAILED_CREATE_REPO",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "wall_clock_s": time.perf_counter() - t0,
        }

    stamp = _stamp()
    uploaded: list[str] = []

    # 1. results JSON (timestamped + latest.json).
    try:
        upload_file(
            path_or_fileobj=str(RESULTS_PATH),
            path_in_repo=f"results/v2_validation_{stamp}.json",
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            token=token,
        )
        uploaded.append(f"results/v2_validation_{stamp}.json")
        upload_file(
            path_or_fileobj=str(RESULTS_PATH),
            path_in_repo="results/latest.json",
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            token=token,
        )
        uploaded.append("results/latest.json")
    except Exception as e:  # noqa: BLE001
        return {
            "status": "FAILED_UPLOAD_RESULTS",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "uploaded_before_error": uploaded,
            "wall_clock_s": time.perf_counter() - t0,
        }

    # 2. smoke JSONL — gzip if > 5 MB.
    if PHASE_B_OUTPUT_PATH.exists():
        try:
            src = PHASE_B_OUTPUT_PATH
            size = src.stat().st_size
            if size > 5 * 1024 * 1024:
                gz = Path(str(src) + ".gz")
                with src.open("rb") as fin, gzip.open(gz, "wb", compresslevel=6) as fout:
                    shutil.copyfileobj(fin, fout)
                upload_path = f"smoke_jsonl/v2_smoke_{stamp}.jsonl.gz"
                upload_file(
                    path_or_fileobj=str(gz),
                    path_in_repo=upload_path,
                    repo_id=RESULTS_REPO,
                    repo_type="dataset",
                    token=token,
                )
            else:
                upload_path = f"smoke_jsonl/v2_smoke_{stamp}.jsonl"
                upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=upload_path,
                    repo_id=RESULTS_REPO,
                    repo_type="dataset",
                    token=token,
                )
            uploaded.append(upload_path)
        except Exception as e:  # noqa: BLE001
            # Non-fatal: results JSON already uploaded.
            return {
                "status": "PARTIAL_UPLOAD_RESULTS_OK_JSONL_FAIL",
                "uploaded": uploaded,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "wall_clock_s": time.perf_counter() - t0,
            }

    return {
        "status": "COMPLETED",
        "uploaded": uploaded,
        "repo": RESULTS_REPO,
        "wall_clock_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _load_teacher_llm():
    """Construct the vLLM LLM instance, trying FP8 first then falling
    back to bf16 if FP8 init fails.

    Sets VLLM_CALIB_CAPTURE_EXPERT=1 BEFORE the LLM is built so the
    expert_in hook is wired during model load (vllm.calibration_hooks
    samples this env once at module-import time).
    """
    # Ensure expert_in capture is enabled — Phase A relies on it.
    os.environ["VLLM_CALIB_CAPTURE_EXPERT"] = "1"
    # Keep all other writers OFF so no sidecar paths are required.
    for k in (
        "VLLM_CALIB_CAPTURE_ROUTER",
        "VLLM_CALIB_CAPTURE_BLOCK",
        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED",
        "VLLM_CALIB_CAPTURE_EXPERT_MID",
        "VLLM_CALIB_CAPTURE_IMATRIX",
        "VLLM_CALIB_CAPTURE_INPUT_COV",
        "VLLM_CALIB_CAPTURE_REAP_SCORES",
        "VLLM_CALIB_CAPTURE_PER_EXPERT_MAX",
    ):
        os.environ.setdefault(k, "0")
    os.environ.setdefault("VLLM_CALIB_MAX_LAYER", "-1")

    from vllm import LLM  # noqa: WPS433

    # Always try the model's intrinsic dtype first via "auto" (which
    # respects the config's torch_dtype — Qwen3.6-35B-A3B ships in
    # bf16 by default; FP8 quantization, if requested, is configured
    # via --quantization not --dtype).
    print(f"[{_now()}] loading vLLM teacher: {TEACHER_MODEL}", flush=True)
    print(f"  dtype={VLLM_DTYPE} gpu_util={VLLM_GPU_UTIL} max_model_len={VLLM_MAX_MODEL_LEN}", flush=True)

    # Single-attempt BF16 (rtx-pro-6000 has 96 GB; BF16 weights ~70 GB
    # fit with ~26 GB free for KV cache + activations). If init fails
    # we surface the error rather than silently downgrading to FP8 —
    # Phase B's build script can't consume FP8 weights so a mid-phase
    # dtype split would invalidate the smoke verdict.
    print(f"[{_now()}]   attempting LLM(... dtype={VLLM_DTYPE})", flush=True)
    try:
        llm = LLM(
            model=TEACHER_MODEL,
            dtype=VLLM_DTYPE,
            enforce_eager=False,
            gpu_memory_utilization=VLLM_GPU_UTIL,
            max_model_len=VLLM_MAX_MODEL_LEN,
            trust_remote_code=False,
        )
        print(f"[{_now()}]   teacher loaded (dtype={VLLM_DTYPE})", flush=True)
        return llm, {"dtype": VLLM_DTYPE, "quantization": None}
    except Exception as e:  # noqa: BLE001
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"teacher-load failed at dtype={VLLM_DTYPE}: {e!r}") from e


def _run_phase_a_inproc() -> int:
    """Phase-A subprocess entry point: load teacher, run alignment,
    dump result JSON to /tmp/v2_phase_a_result.json, exit. Running
    Phase A in its own subprocess guarantees the teacher's VRAM is
    released to the OS / CUDA driver before Phase B (the build-script
    subprocess) tries to allocate its own teacher.
    """
    print(f"[{_now()}] === Phase A subprocess ===", flush=True)
    out_path = Path("/tmp/v2_phase_a_result.json")
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=False)
        llm, load_info = _load_teacher_llm()
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"!! Phase A teacher-load failure: {e!r}\n{tb}", flush=True)
        _atomic_write_json(out_path, {
            "status": "FAILED_TEACHER_LOAD",
            "error": repr(e),
            "traceback": tb,
            "verdict": "HARD_FAIL",
        })
        return 1

    try:
        phase_a = run_phase_a(llm, tokenizer)
        phase_a["teacher_load_info"] = load_info
    except Exception as e:  # noqa: BLE001
        phase_a = {
            "status": "CRASHED",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "verdict": "HARD_FAIL",
            "teacher_load_info": load_info,
        }

    _atomic_write_json(out_path, phase_a)
    print(f"[{_now()}] Phase A result written -> {out_path}", flush=True)
    return 0 if phase_a.get("status") == "COMPLETED" else 1


def main() -> int:
    # Subprocess re-entry: if invoked with --phase a, run that phase
    # and exit. The parent process orchestrates and aggregates.
    if len(sys.argv) >= 2 and sys.argv[1] == "--phase":
        phase = sys.argv[2] if len(sys.argv) >= 3 else ""
        if phase == "a":
            return _run_phase_a_inproc()
        raise SystemExit(f"unknown --phase value: {phase!r}")

    print(f"[{_now()}] === v2 validation harness (parent) ===", flush=True)
    print(f"  pinned_commit : {PINNED_COMMIT}", flush=True)
    print(f"  teacher       : {TEACHER_MODEL}", flush=True)
    print(f"  repo_root     : {REPO_ROOT}", flush=True)
    print(f"  results_path  : {RESULTS_PATH}", flush=True)
    print(f"  results_repo  : {RESULTS_REPO}", flush=True)
    env = _env_banner()
    for k, v in env.items():
        print(f"  {k:14s}: {v}", flush=True)

    results: dict[str, Any] = {
        "harness_version": 1,
        "harness_name": "v2_validation_harness",
        "pinned_commit": PINNED_COMMIT,
        "teacher_model": TEACHER_MODEL,
        "env": env,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phases": {},
    }
    _atomic_write_json(RESULTS_PATH, results)

    # ------ Phase A in a subprocess so its VRAM is fully released ------
    phase_a_result_path = Path("/tmp/v2_phase_a_result.json")
    if phase_a_result_path.exists():
        phase_a_result_path.unlink()
    print(f"\n[{_now()}] spawning Phase A subprocess", flush=True)
    sub_env = os.environ.copy()
    # Inherit PYTHONPATH (set by the .sh wrapper) so the subprocess can
    # import moe_compress.utils.calibration if Phase A ever needs it.
    cmd_a = [sys.executable, str(Path(__file__).resolve()), "--phase", "a"]
    rc_a = subprocess.call(cmd_a, env=sub_env)
    print(f"[{_now()}] Phase A subprocess exit code: {rc_a}", flush=True)
    if phase_a_result_path.exists():
        try:
            phase_a = json.loads(phase_a_result_path.read_text())
        except Exception as e:  # noqa: BLE001
            phase_a = {
                "status": "CRASHED",
                "error": f"failed to parse {phase_a_result_path}: {e!r}",
                "verdict": "HARD_FAIL",
            }
    else:
        phase_a = {
            "status": "CRASHED_NO_RESULT_FILE",
            "subprocess_exit_code": rc_a,
            "verdict": "HARD_FAIL",
        }
    results["phases"]["phase_a_alignment"] = phase_a
    _atomic_write_json(RESULTS_PATH, results)

    # ------ Phase B in a subprocess (build script) ------
    try:
        phase_b = run_phase_b()
    except Exception as e:  # noqa: BLE001
        phase_b = {
            "status": "CRASHED",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "verdict": "HARD_FAIL",
        }
    results["phases"]["phase_b_smoke_build"] = phase_b
    _atomic_write_json(RESULTS_PATH, results)

    # ------ Phase C ------
    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _atomic_write_json(RESULTS_PATH, results)
    try:
        phase_c = run_phase_c(results)
    except Exception as e:  # noqa: BLE001
        phase_c = {
            "status": "CRASHED",
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
    results["phases"]["phase_c_upload"] = phase_c
    _atomic_write_json(RESULTS_PATH, results)

    # ------ summary ------
    print("\n=== V2 VALIDATION SUMMARY ===", flush=True)
    print(f"  Phase A (alignment) : {phase_a.get('verdict', '?')} "
          f"(status={phase_a.get('status', '?')})", flush=True)
    print(f"  Phase B (smoke)     : {phase_b.get('verdict', '?')} "
          f"(status={phase_b.get('status', '?')})", flush=True)
    print(f"  Phase C (upload)    : {phase_c.get('status', '?')}", flush=True)
    print(f"  results @ {RESULTS_PATH}", flush=True)

    # Exit code policy: 0 if BOTH phases A and B completed (regardless
    # of verdict — the verdict goes in the JSON for the supervisor to
    # interpret). Non-zero only on hard infrastructure failure.
    phase_a_ok = phase_a.get("status") == "COMPLETED"
    phase_b_ok = phase_b.get("status") == "COMPLETED"
    return 0 if (phase_a_ok and phase_b_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
