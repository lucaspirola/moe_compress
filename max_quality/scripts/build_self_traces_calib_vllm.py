#!/usr/bin/env python3
"""Build the ``self-traces`` calibration JSONL via vLLM offline inference.

Why this exists alongside ``build_self_traces_calib.py``
--------------------------------------------------------
The transformers-based build script (the HF-generate path) hits two walls on
Qwen3-thinking models running on 96 GB-class GPUs:

  1. ``output_scores=True`` materializes a per-step [bs, vocab] tensor that
     accumulates with token count and OOMs around step 4-8k.
  2. The model has a documented "endless reasoning loops" failure mode (see
     ``Qwen/Qwen3.6-35B-A3B`` discussion #19) where simple prompts trigger
     dozens of self-questioning cycles. With ``max_new_tokens=16384`` and no
     budget enforcement, every such prompt burns the full 16k tokens. At
     bs=1 + bf16 we measured 267 s/trace average → 480+ hours total.

This script replaces HF generate with vLLM. vLLM provides:

  * **Continuous batching + PagedAttention** — orders-of-magnitude higher
    throughput than HF generate on the same hardware; the bs ceiling we
    kept hitting goes away.
  * **Native per-token top-k logprobs** via ``SamplingParams(logprobs=K)``
    — the teacher's predicted-next-token distribution at every position,
    no LogitsProcessor scaffolding required.
  * **``reasoning_budget`` support** (vLLM PR #20859, merged to main) —
    caps tokens emitted inside ``<think>...</think>`` via the model's
    reasoning parser. Forces the close tag once the budget is reached;
    the model then emits a final answer immediately. Eliminates the
    endless-loop tail without dropping --max-new-tokens.

Output schema is byte-identical to the HF script: same JSONL row shape with
``_complete`` / ``_attempt_idx`` / ``completion_source``, same ``.npz``
logit-cache sidecar layout. ``completion_source`` (added in schema v9, the
companion bump to the HF script's schema v7) records whether the assistant
content came from vLLM generation (``"teacher_generated"``) or directly
from the source dataset's canonical assistant turn (``"canonical"`` — only
the v2 mix's TEACHER_FORCED subsets). The cache_key folds
``inference_engine="vllm"`` so vLLM and HF outputs never collide on disk.

Usage
-----
.. code-block:: bash

    python max_quality/scripts/build_self_traces_calib_vllm.py \\
        --teacher Qwen/Qwen3.6-35B-A3B \\
        --prompts qwen3-pretrain-mix \\
        --num-prompts 6500 \\
        --max-new-tokens 16384 \\
        --reasoning-budget 4096 \\
        --logits-top-k 50 \\
        --gpu-memory-utilization 0.90 \\
        --chunk-size 200 \\
        --output artifacts/_shared/self_traces.jsonl

The ``--chunk-size`` controls how many prompts vLLM batches per
``LLM.generate`` call. vLLM internally continuous-batches up to its
configured concurrency; the chunk size only affects how often we flush
JSONL rows to disk for crash-recovery.

Cost expectation (Qwen3.6-35B-A3B BF16, reasoning_budget=4096):
  * H200 SXM5 (141 GB) — ~2-3 h, $7-10 at $3.39/hr (DataCrunch)
  * H100 SXM5 (80 GB)   — ~3-5 h with FP8 quant, $8-12
  * B300 SXM6 (262 GB)  — ~1-1.5 h, $9-11 at $6.99/hr

Determinism
-----------
``temperature=0`` + ``seed`` + a fixed teacher revision + fixed prompts
gives reproducible token sequences in vLLM's deterministic mode. Note: vLLM
non-determinism CAN appear when ``tensor_parallel_size > 1`` or when
``enforce_eager=False`` and CUDA graphs are recompiled between runs. The
default config below pins both to deterministic settings.

This is a ONE-SHOT pre-step — see the HF script's docstring for the
downstream pipeline integration (Stage 2.5 router-KD consumes the JSONL
+ logit cache).
"""
from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

log = logging.getLogger("build_self_traces_calib_vllm")

# B0/C1 — run the vLLM V1 EngineCore IN-PROCESS (no engine subprocess).
#
# vLLM V1 defaults VLLM_ENABLE_V1_MULTIPROCESSING=1, which runs EngineCore (and
# therefore the model forward + every capture *dispatch* site: MoERunner.apply,
# TritonExperts.apply, LinearBase, LogitsProcessor) in a separate WORKER
# subprocess. Our calibration writers call register_callback() in the DRIVER
# process, so under MP=1 the worker's callback registry is empty and every
# dispatch is a no-op; the driver's _resolve_model() also finds no
# model_executor (set only `if not multiprocess_mode`), so _N_LAYERS=0 and
# every sidecar is written empty ("no MoE layers seen").
#
# Forcing MP=0 makes EngineCore run in-process so setup + dispatch + dump all
# share one process. Calibration is single-GPU (tp=1 -> UniProcExecutor), so
# in-process is correct here and also more deterministic for capture.
#
# This is a DRIVER INVARIANT, not a default: this script CANNOT capture with
# MP=1. We therefore HARD-SET it (not setdefault) so a stale shell
# `export VLLM_ENABLE_V1_MULTIPROCESSING=1` cannot survive and silently
# re-break captures -- that would only be caught by the fail-fast after a
# wasted chunk + full GPU spin-up. Must be set BEFORE vllm is first imported
# anywhere (the import is lazy, inside _load_teacher_vllm), so it lives at
# module level here.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
log.info("calibration driver: forcing VLLM_ENABLE_V1_MULTIPROCESSING=0 "
         "(in-process EngineCore required for capture hooks to fire)")

# Reuse prompt loaders + per-domain stats helpers from the HF script — they
# don't depend on transformers / vLLM, only on utils/calibration.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_self_traces_calib import (  # type: ignore
    _iter_prompts_from_qwen3_pretrain_mix,
    _iter_prompts_from_qwen3_pretrain_mix_v2,
    _iter_prompts_from_jsonl,
    _coerce_eos_ids,
    _trim_at_first_eos,
)


# ---------------------------------------------------------------------------
# Cache-key — extends the HF script's cache_key with vLLM-specific fields so
# the two engines' outputs never collide on disk.
# ---------------------------------------------------------------------------


def _trace_cache_key_vllm(
    teacher_repo: str,
    teacher_revision: str,
    prompts_source: str,
    num_prompts: int,
    seed: int,
    max_new_tokens: int,
    reasoning_budget: int,
    dtype: str,
    logits_top_k: int,
) -> str:
    """Compute the cache_key for vLLM runs.

    Fields:
      * ``inference_engine="vllm"`` — partitions vLLM outputs from the HF
        script's outputs.
      * ``reasoning_budget`` — affects what tokens land inside <think> and
        thus the saved logits; runs with different budgets are NOT
        interchangeable.
      * ``dtype`` — bf16 / fp8 / awq runs produce different teacher logits.
      * ``logits_top_k`` — always folded (this script always saves logits;
        the HF script made it optional).
      * ``schema_version=9`` — bumped 8→9 in Step 6 of
        tasks/CALIBRATION_MIX_V2_PLAN.md. v9 is the version that carries
        the new ``completion_source`` field on every row
        (``"teacher_generated"`` for rows produced by vLLM generation;
        ``"canonical"`` for v2 TEACHER_FORCED rows synthesized directly
        from the source dataset's canonical assistant turn). v8 was the
        Items 8+9 metadata bundle (``n_prompt_tokens``, ``n_gen_tokens``,
        ``has_think``, ``refusal_flag``, ``subset``, ``seed_idx``).
        Existing v8 runs are NOT cache-hit by v9 runs.
    """
    payload = json.dumps({
        "teacher_repo": teacher_repo,
        "teacher_revision": teacher_revision,
        "prompts_source": prompts_source,
        "num_prompts": num_prompts,
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "reasoning_budget": int(reasoning_budget),
        "dtype": str(dtype),
        "logits_top_k": int(logits_top_k),
        "decode": "greedy",
        "inference_engine": "vllm",
        "schema_version": 9,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# B0 fail-fast: enabled captures must produce nonzero entries.
# ---------------------------------------------------------------------------

# Maps each --capture-* arg name to the vllm writer module that holds its
# accumulator. ``layer_input_reservoir`` rides stage2_profile (no separate
# module), so it is intentionally absent here -- its non-emptiness is covered
# by the stage2_profile writer's count.
_CAPTURE_WRITER_MODULES: dict[str, str] = {
    "capture_imatrix": "vllm.calibration_imatrix",
    "capture_reap_scores": "vllm.calibration_reap_scores",
    "capture_input_covariance": "vllm.calibration_input_cov",
    "capture_wanda_scalar_row": "vllm.calibration_wanda_scalar_row",
    "capture_stage2_profile": "vllm.calibration_stage2_profile",
    "capture_per_expert_max": "vllm.calibration_per_expert_max",
    "capture_routing_stats": "vllm.calibration_routing_stats",
    "capture_router_logits_stats": "vllm.calibration_router_logits_stats",
    "capture_output_reservoir": "vllm.calibration_output_reservoir",
    "capture_block_outputs": "vllm.calibration_block_outputs",
}


def _default_writer_resolver(module_name: str):
    """Import a vllm calibration writer module by dotted name.

    Separated out so tests can inject a fake resolver returning stub writer
    objects (no monkeypatch of production code, no real vllm wheel required).
    """
    import importlib

    return importlib.import_module(module_name)


def assert_enabled_captures_nonempty(
    enabled_captures,
    *,
    model_class: str = "<unknown>",
    allow_empty: bool = False,
    writer_resolver=_default_writer_resolver,
) -> dict[str, int]:
    """B0 fail-fast: verify every ENABLED capture has nonzero entries.

    Called once, right after the first chunk that actually invoked
    ``llm.generate()``. For each enabled ``--capture-*`` it resolves the
    writer module and reads its PUBLIC ``captured_entry_count() -> int``
    (no private ``_state`` reach-in, no monkeypatch).

    Behaviour per writer:
      * writer exposes ``captured_entry_count`` and returns 0  -> EMPTY
        (a hook/model mismatch -- the B0 bug class).
      * writer exposes it and returns > 0                      -> OK.
      * writer (installed wheel) predates the method            -> SKIP with a
        WARNING (cannot prove emptiness; do not crash on an old wheel).

    If any enabled capture is EMPTY: log ERROR naming every empty capture +
    the resolved model class, then ``SystemExit(2)`` BEFORE any checkpoint --
    unless ``allow_empty`` (``--allow-empty-captures``) downgrades it to a
    warning.

    Returns ``{capture_arg_name: count}`` for the captures that exposed the
    count method (diagnostic; SKIPs are omitted).
    """
    counts: dict[str, int] = {}
    empty: list[str] = []
    skipped: list[str] = []
    for cap in enabled_captures:
        module_name = _CAPTURE_WRITER_MODULES.get(cap)
        if module_name is None:
            # Not a count-checkable capture (e.g. layer_input_reservoir which
            # rides stage2_profile). Skip silently.
            continue
        try:
            writer = writer_resolver(module_name)
        except Exception as exc:  # pragma: no cover - import-time failure
            log.warning(
                "B0 fail-fast: could not import writer %s for --%s (%s); "
                "skipping its non-empty check.",
                module_name, cap.replace("_", "-"), exc,
            )
            skipped.append(cap)
            continue
        counter = getattr(writer, "captured_entry_count", None)
        if not callable(counter):
            log.warning(
                "B0 fail-fast: installed %s has no captured_entry_count() "
                "(pre-B0 wheel); skipping its non-empty check. Rebuild the "
                "patched wheel to enable this gate for --%s.",
                module_name, cap.replace("_", "-"),
            )
            skipped.append(cap)
            continue
        n = int(counter())
        counts[cap] = n
        if n <= 0:
            empty.append(cap)

    if empty:
        names = ", ".join("--" + c.replace("_", "-") for c in sorted(empty))
        msg = (
            "B0 fail-fast: %d enabled capture(s) produced ZERO entries after "
            "the first generate chunk: %s. Resolved model class=%s. The "
            "capture hooks did not bind to this model's MoE path -- see "
            "tasks/PLAN_B0_HOOK_FIX.md (check VLLM_ENABLE_V1_MULTIPROCESSING=0 "
            "and that the wheel carries the C2 predicate + M3 block hook)."
        )
        if allow_empty:
            log.warning(msg + " (continuing: --allow-empty-captures set)",
                        len(empty), names, model_class)
        else:
            log.error(msg, len(empty), names, model_class)
            raise SystemExit(2)
    else:
        log.info(
            "B0 fail-fast: all %d checkable enabled capture(s) nonempty "
            "(%s)%s.",
            len(counts),
            ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none",
            f"; {len(skipped)} skipped (pre-B0 wheel)" if skipped else "",
        )
    return counts


# ---------------------------------------------------------------------------
# vLLM driver
# ---------------------------------------------------------------------------


def _harden_runtime_env(output_path: str, dtype: str) -> None:
    """Apply operational fixes learned running this driver on fresh cloud GPU
    boxes (see ``docs/calibration_vllm_runbook.md``).

    All of these are SAFE: they touch only the build/cache environment, never
    the sampling path, so generated tokens and the trace cache_key are
    unchanged.

    1. Cap JIT-compiler fan-out. The first forward pass JIT-compiles the
       FlashInfer GDN (gated-delta-net) prefill kernel, spawning one ``cicc``
       (CUDA compiler) per translation unit — uncapped that is ~one per vCPU,
       each ~6 GB RSS, which host-RAM-OOM-kills the run on a 180 GB box.
       ``MAX_JOBS`` caps the fan-out; ``NVCC_THREADS=1`` keeps per-job RSS low.
    2. Persist the vLLM / torch.compile cache next to the output (durable
       storage) so a restart after a spot preemption skips recompilation.
    3. Warn about the two host prerequisites that fail loudly and late:
       missing ``python3-dev`` (no ``Python.h`` -> the kernel JIT cc fails) and
       a present-but-incompatible ``kernels`` package (needed only for an FP8
       teacher; with a non-FP8 teacher it can break ``from vllm import LLM``).

    Idempotent and override-friendly: every env var uses ``setdefault`` so an
    operator can still pin their own values.
    """
    import importlib.util
    import sysconfig

    cpu = os.cpu_count() or 8
    os.environ.setdefault("MAX_JOBS", str(max(1, min(16, cpu // 2))))
    os.environ.setdefault("NVCC_THREADS", "1")
    if "VLLM_CACHE_ROOT" not in os.environ:
        cache_dir = Path(output_path).resolve().parent / ".vllm_compile_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = str(cache_dir)
    log.info("runtime env: MAX_JOBS=%s NVCC_THREADS=%s VLLM_CACHE_ROOT=%s",
             os.environ["MAX_JOBS"], os.environ["NVCC_THREADS"],
             os.environ["VLLM_CACHE_ROOT"])

    py_h = Path(sysconfig.get_path("include")) / "Python.h"
    if not py_h.exists():
        log.warning("Python.h not found at %s — the FlashInfer GDN kernel JIT "
                    "will fail to compile. Install the python dev headers "
                    "(e.g. `apt-get install python3-dev`).", py_h)

    if dtype != "fp8" and importlib.util.find_spec("kernels") is not None:
        log.warning("`kernels` is installed but teacher dtype is %s (not fp8). "
                    "With some transformers versions this makes "
                    "`from vllm import LLM` raise during the hub-kernels "
                    "import. If vLLM import fails, `pip uninstall -y kernels`.",
                    dtype)


def _load_teacher_vllm(
    repo: str, revision: str, dtype: str,
    gpu_memory_utilization: float, max_model_len: int,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_logprobs: int = 50,
    moe_backend: str | None = None,
):
    """Instantiate vLLM's offline LLM for the teacher.

    ``reasoning_parser="qwen3"`` is required for ``reasoning_budget`` to
    engage — vLLM's ReasoningBudgetLogitsProcessor reads the start/end
    think-token ids from this parser. Qwen3-thinking-mode tokenizers
    register ``<think>`` and ``</think>`` as single tokens that the parser
    matches against.

    ``enforce_eager=False`` (the default) lets vLLM build CUDA graphs for
    decode. That's where the throughput wins live. We document it for the
    operator since the determinism contract depends on it being stable
    across runs (i.e., same vLLM version + same teacher revision).

    ``moe_backend`` (when not ``None``) is threaded into the LLM's
    ``kernel_config={"moe_backend": <val>}``. This is REQUIRED for the
    in-graph calibration capture path: the kernel-interior signals
    (reap_scores, per_expert_max, output_reservoir, plus the per-layer
    block_outputs hook) only dispatch inside ``TritonExperts.apply`` /
    ``MoERunner.apply``. With the default ``"auto"`` backend, vLLM picks
    FlashInfer's fused-MoE kernel on Hopper/Blackwell, which never reaches
    the patched Triton dispatch sites — every capture sidecar would come
    out empty (the B0 failure class). The calibration orchestration always
    passes ``moe_backend="triton"`` (the driver's ``--moe-backend`` default)
    so the replay-capture path runs ``TritonExperts``. When ``None`` the
    kwarg is omitted entirely so vLLM's auto-selection is unchanged (kept
    for non-capture / pure-generate experimentation).

    ``max_num_seqs`` and ``max_num_batched_tokens`` are vLLM's continuous-
    batching knobs:
      * max_num_seqs — the cap on concurrent sequences in flight. Default 256;
        with H200 (141 GB) + Qwen3-class 35B + 16k token sequences we can
        often push to 384-512 if VRAM headroom allows.
      * max_num_batched_tokens — the cap on tokens scheduled per forward pass.
        Higher = better GPU utilization during prefill; trades off latency on
        long contexts. Default ~8192-16384 depending on vLLM version.
    Both ``None`` means use vLLM defaults — set explicitly via CLI for
    throughput tuning when steady-state VRAM observation shows headroom.
    """
    from vllm import LLM  # type: ignore

    log.info("loading teacher via vLLM: %s (revision=%s, dtype=%s, "
             "max_num_seqs=%s, max_num_batched_tokens=%s)",
             repo, revision, dtype, max_num_seqs, max_num_batched_tokens)
    kwargs: dict = dict(
        model=repo,
        revision=revision,
        dtype=dtype,                         # "bfloat16" | "float16" | "auto"
        tensor_parallel_size=1,               # single-GPU determinism
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        # Required for SamplingParams.reasoning_budget to engage on Qwen3.
        reasoning_parser="qwen3",
        # Trust remote code — Qwen3.6's modeling files use custom Python.
        trust_remote_code=True,
        # vLLM 0.21 defaults max_logprobs=20; we need 50 for the topk teacher
        # cache. Raise it to match --logits-top-k. Pure runtime validation
        # cap (not deterministic-output-affecting), so cache_key unchanged.
        max_logprobs=int(max_logprobs),
    )
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if moe_backend is not None:
        # Force the fused-MoE backend. "triton" is required for the
        # in-graph capture hooks to fire (see docstring). Passed via
        # kernel_config, which vLLM forwards to the MoE method selection.
        kwargs["kernel_config"] = {"moe_backend": str(moe_backend)}
        log.info("vLLM kernel_config: forcing moe_backend=%s "
                 "(required for in-graph calibration capture)", moe_backend)
    return LLM(**kwargs)


def _resolve_moe_backend(args) -> "str | None":
    """Resolve the fused-MoE backend to force on the LLM.

    Returns ``args.moe_backend`` (default ``"triton"``) so the value
    threads uniformly into both the generate and replay ``_load_teacher_vllm``
    call sites. The replay (``--replay-from``) path REQUIRES a non-auto
    backend (triton) for the kernel-interior captures to fire; if an
    operator explicitly downgrades to ``--moe-backend auto`` while ANY
    ``--capture-*`` flag is set, this is a misconfiguration that would
    silently produce empty sidecars, so we hard-force ``"triton"`` and log
    a warning rather than honour the auto downgrade. An explicit non-auto
    override (e.g. another patched backend) is honoured as-is.

    Passing the literal string ``"none"`` (case-insensitive) maps to
    ``None`` -> omit kernel_config entirely (escape hatch for benchmarking
    vLLM's own selection on a pure-generate run).
    """
    backend = getattr(args, "moe_backend", "triton")
    if backend is not None and str(backend).lower() == "none":
        return None
    enabled_captures = any(
        getattr(args, cap, False) for cap in _CAPTURE_WRITER_MODULES
    )
    if enabled_captures and (backend is None or str(backend).lower() == "auto"):
        log.warning(
            "moe_backend=%r requested but --capture-* flags are enabled; "
            "auto/None would route to FlashInfer and yield EMPTY capture "
            "sidecars. Hard-forcing moe_backend='triton'.", backend,
        )
        return "triton"
    return backend


def _render_prompts(tokenizer, prompts: Iterable[str]) -> list[str]:
    """Render a list of user prompts through the model's chat template
    with the thinking-mode opener appended (apply_chat_template handles
    the ``<think>`` injection automatically for Qwen3-thinking tokenizers)."""
    out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        out.append(text)
    return out


def _extract_topk_from_vllm_logprobs(
    step_logprobs, top_k: int,
):
    """Convert vLLM's per-step ``List[Dict[int, Logprob]]`` into
    ``(top_ids: int32[T,K], top_logprobs: fp16[T,K])`` numpy arrays.

    vLLM gives us ``logprobs=K`` candidates per step already sorted by
    descending logprob. We just unpack into dense arrays. If a step has
    fewer than ``top_k`` entries (rare; happens when the next-token
    distribution is sharply peaked), we pad with -inf logprob and a
    sentinel id of -1.
    """
    import numpy as np

    T = len(step_logprobs)
    top_ids = np.full((T, top_k), -1, dtype=np.int32)
    top_lp  = np.full((T, top_k), -1e9, dtype=np.float16)
    for t, step in enumerate(step_logprobs):
        if step is None:
            continue
        # step: Dict[int token_id -> Logprob(logprob, rank, decoded_token)]
        # Sort by logprob descending, then take first top_k.
        sorted_items = sorted(step.items(), key=lambda kv: -kv[1].logprob)
        for k, (tok_id, lp_obj) in enumerate(sorted_items[:top_k]):
            top_ids[t, k] = int(tok_id)
            top_lp[t, k]  = float(lp_obj.logprob)
    return top_ids, top_lp


# JSONL row schema v8 (Items 8+9): per-row metadata bundle.
#
# ``_REFUSAL_PATTERN`` — heuristic match against the assistant answer
# (post-strip, AFTER any ``<think>...</think>`` block — see
# ``_strip_think_block``). Matches at the start of the answer body, since
# refusals open with the apology/disclaimer phrase. Pattern intentionally
# narrow (5 canonical openers) to avoid false positives on legitimate
# answers that happen to contain "sorry" or "can't" mid-sentence.
#
# Detected openers:
#   * "I cannot"
#   * "I can't"
#   * "I'm sorry"
#   * "I am sorry"
#   * "Sorry, I" / "Sorry I"
# Case-insensitive; matches optional leading whitespace.
_REFUSAL_PATTERN = re.compile(
    r"^\s*(i\s+cannot\b|i\s+can['’]t\b|i['’]m\s+sorry\b|"
    r"i\s+am\s+sorry\b|sorry,?\s+i\b)",
    re.IGNORECASE,
)

# Regex stripping the leading ``<think>...</think>`` block (if present)
# so the refusal-heuristic sees the answer body, not the reasoning trace
# (the model often types "I'm sorry, I need to think about this..." INSIDE
# the think block, which is reasoning prose, not a refusal of the user's
# task). DOTALL because ``<think>`` content can span newlines.
_THINK_BLOCK_PATTERN = re.compile(
    r"^\s*<think>.*?</think>\s*", re.DOTALL,
)


def _has_think_block(answer: str) -> bool:
    """True iff the assistant answer contains a ``<think>...</think>``
    block. Used as an Item-8 ``has_think`` metadata flag. Closed tag is
    required: an unterminated ``<think>`` (which can occur on a
    ``finish_reason='length'`` truncation tail) is NOT counted, matching
    the existing ``is_complete`` predicate that also requires
    ``"</think>" in ans``."""
    return "<think>" in answer and "</think>" in answer


def _detect_refusal(answer: str) -> bool:
    """Heuristic refusal detector — see ``_REFUSAL_PATTERN`` docstring for
    the matched phrases. Strips any leading ``<think>...</think>`` block
    first so the heuristic fires on the assistant's final answer, not on
    in-think reasoning prose that happens to mention "sorry"."""
    body = _THINK_BLOCK_PATTERN.sub("", answer, count=1)
    return bool(_REFUSAL_PATTERN.search(body))


def _process_outputs(
    outputs, prompts_chunk, attempt_idx_chunk, eos_ids, logits_top_k,
    logits_dir, domain_stats, max_new_tokens,
):
    """Per chunk: convert vLLM ``RequestOutput`` results into our JSONL row
    shape + the .npz logit-cache sidecars. Yields one dict per output.

    ``prompts_chunk`` may be a list of 2-tuples ``(prompt, domain)`` (v1
    / JSONL iterators) or 4-tuples ``(prompt, domain, canonical, policy)``
    (v2 iterator). _process_outputs reads only the first two positions;
    the policy field is consulted by the caller (chunk loop) which is
    responsible for partitioning GENERATE rows (forwarded here) from
    TEACHER_FORCED rows (handled by ``_synth_teacher_forced_rows``).

    JSONL row schema v9 (Step 6 of CALIBRATION_MIX_V2_PLAN.md)
    -------------------------------------------------------------------
    Every row produced by this function carries
    ``completion_source="teacher_generated"`` (TEACHER_FORCED rows go
    through ``_synth_teacher_forced_rows`` and get ``"canonical"``).

    Each yielded dict carries the original ``messages`` / ``domain`` /
    ``_complete`` / ``_attempt_idx`` fields plus the v8 metadata bundle:

      * ``n_prompt_tokens`` (int) — vLLM-tokenized prompt length, from
        ``out.prompt_token_ids`` (the rendered chat-templated prompt that
        was fed to ``LLM.generate``). Excludes generated tokens.
      * ``n_gen_tokens`` (int) — emitted token count, ``len(gen.token_ids)``
        BEFORE EOS-trim. Includes ``<think>`` content if present and any
        EOS sentinel vLLM appended; we keep the un-trimmed count because
        downstream cost/length analyses want the raw decode work, not the
        post-trim signal length.
      * ``has_think`` (bool) — whether the answer contains a closed
        ``<think>...</think>`` block (see ``_has_think_block``).
      * ``refusal_flag`` (bool) — heuristic refusal detector (see
        ``_detect_refusal``).
      * ``subset`` (str) — the prompt's domain/subset, duplicated from
        the existing ``domain`` field for consumer convenience (matches
        the plan-doc field name).
      * ``seed_idx`` (int) — duplicate of ``_attempt_idx`` under the
        Item-9 name. The attempt index is the prompt's position in the
        shuffled ``CalibrationSpec`` source ordering, which is what the
        plan refers to as the deterministic per-prompt seed index. We
        keep both keys for backward compatibility — existing consumers
        reading ``_attempt_idx`` continue to work unchanged.
    """
    import numpy as np

    for prompt_text, attempt_idx, out in zip(prompts_chunk, attempt_idx_chunk, outputs):
        # vLLM returns RequestOutput with .outputs[0] being the first (only)
        # generated completion (we don't request n>1).
        gen = out.outputs[0]
        full_text = gen.text or ""

        # Token sequence + EOS detection. vLLM's ``finish_reason`` is
        # ``"stop"`` when EOS or stop string was hit, ``"length"`` when
        # max_tokens was reached without EOS. ``"stop"`` is our complete
        # signal; ``"length"`` means truncation.
        token_ids = list(gen.token_ids)
        n_emit = len(token_ids)
        saw_eos = (gen.finish_reason == "stop")

        # We still apply the EOS-trim defensively (some stop-string paths
        # land the eos in the sequence). Same logic as the HF script.
        trimmed_ids = _trim_at_first_eos(token_ids, eos_ids)

        # Strip the assistant turn from the rendered text. vLLM does NOT
        # echo the prompt in gen.text, so full_text is already
        # assistant-only. Just clean whitespace.
        ans = full_text.strip()

        domain = prompt_text[1]  # we packed as (text, domain) tuples
        prompt_str = prompt_text[0]

        # Completeness: same predicate as the HF script.
        is_complete = bool(saw_eos and "</think>" in ans)
        stats = domain_stats.setdefault(domain, [0, 0])
        stats[1] += 1
        if is_complete:
            stats[0] += 1

        # Logit sidecar — only for complete rows (truncated tails are
        # mid-thought noise and would poison KD).
        if is_complete and gen.logprobs is not None and logits_dir is not None:
            n_keep = len(trimmed_ids)
            if n_keep > 0:
                top_ids, top_lp = _extract_topk_from_vllm_logprobs(
                    gen.logprobs[:n_keep], logits_top_k,
                )
                fp = logits_dir / f"{int(attempt_idx):07d}.npz"
                # F-C-1 fix: durable .npz write via atomic_io.
                # The previous tmp_fp = fp.with_suffix(fp.suffix + ".tmp")
                # was broken: np.savez_compressed(str_path, …)
                # auto-appends ".npz" to any path that doesn't end in
                # ".npz", so the call wrote "000.npz.tmp.npz" and the
                # subsequent os.replace(tmp_fp, fp) raised
                # FileNotFoundError. atomic_npz_save passes an open
                # binary file HANDLE which numpy does NOT auto-extend,
                # and adds fsync(fd) + fsync(parent_dir) for true
                # durability under eviction-class SIGKILL.
                from moe_compress.utils.atomic_io import atomic_npz_save
                atomic_npz_save(
                    fp,
                    token_ids=np.asarray(trimmed_ids, dtype=np.int32),
                    top_ids=top_ids,
                    top_logprobs=top_lp,
                    attempt_idx=np.int64(attempt_idx),
                    top_k=np.int32(logits_top_k),
                )

        # Item 8+9: per-row metadata bundle. ``prompt_token_ids`` on
        # RequestOutput is the rendered+tokenized prompt vLLM actually
        # consumed (not the raw user text), so it matches the model's
        # forward-pass view 1:1. ``n_gen_tokens`` is the un-trimmed
        # emit count — see _process_outputs docstring rationale.
        prompt_token_ids = getattr(out, "prompt_token_ids", None) or []
        n_prompt_tokens = len(prompt_token_ids)
        n_gen_tokens = n_emit
        has_think = _has_think_block(ans)
        refusal_flag = _detect_refusal(ans)

        yield {
            "messages": [
                {"role": "user", "content": prompt_str},
                {"role": "assistant", "content": ans},
            ],
            "domain": domain,
            "_complete": is_complete,
            "_attempt_idx": int(attempt_idx),
            # --- Item 8+9 metadata bundle (JSONL schema v8) ---------------
            "n_prompt_tokens": int(n_prompt_tokens),
            "n_gen_tokens": int(n_gen_tokens),
            "has_think": bool(has_think),
            "refusal_flag": bool(refusal_flag),
            "subset": str(domain),
            "seed_idx": int(attempt_idx),
            # --- v9 (CALIBRATION_MIX_V2_PLAN.md Step 6) -------------------
            # GENERATE path always writes teacher_generated; the
            # TEACHER_FORCED path is handled by _synth_teacher_forced_rows
            # which emits "canonical" instead.
            "completion_source": "teacher_generated",
        }


def _synth_teacher_forced_rows(
    tf_prompts, tf_attempt_idx, tokenizer, domain_stats,
):
    """Emit JSONL rows for TEACHER_FORCED chunk entries — no vLLM
    generation involved.

    Each ``tf_prompts`` entry is a 4-tuple ``(prompt, domain, canonical,
    policy)`` produced by the v2 iterator. For each entry we:

      1. Render ``messages=[{user: prompt}, {assistant: canonical}]``
         via ``apply_chat_template`` to compute the prompt-tokens count
         (which the v8 metadata bundle exposes as ``n_prompt_tokens``).
         We do NOT render add_generation_prompt because there is no
         generation step.
      2. Tokenize the rendered string with the tokenizer (no padding /
         truncation — we just want the length count). Cheap enough at
         8-K-token TF rows that we skip a length cache.
      3. Compute the same per-row metadata flags as the GENERATE path:
         ``has_think`` from ``_has_think_block`` on the canonical
         completion, ``refusal_flag`` via ``_detect_refusal(canonical)``
         — same heuristic as the GENERATE path; canonical R1 / SWE-smith
         traces evaluate to False in practice, but we run the detector
         for consistency.
      4. Yield a row dict with ``completion_source="canonical"``,
         ``_complete=True``, ``n_gen_tokens=0`` (no generation
         occurred), and the v8 metadata bundle.

    Logit-sidecar emission is skipped intentionally — there's no
    per-step logprobs distribution to capture from a canonical trace
    (the row arrives pre-decided; no model.generate call ever happens).

    ``domain_stats`` is mutated in place to reflect the canonical rows
    as "complete" so the end-of-run completeness summary stays correct.
    """
    for entry, attempt_idx in zip(tf_prompts, tf_attempt_idx):
        # Plan ties TF rows to 4-tuples by construction; reject 2-tuples
        # loudly rather than silently treating them as GENERATE.
        if len(entry) != 4:
            raise ValueError(
                f"_synth_teacher_forced_rows: expected 4-tuple "
                f"(prompt, domain, canonical, policy), got len={len(entry)}: "
                f"{entry!r}"
            )
        prompt, domain, canonical, policy = entry
        if policy != "TEACHER_FORCED":
            raise ValueError(
                f"_synth_teacher_forced_rows: expected policy="
                f"TEACHER_FORCED, got {policy!r} for domain={domain!r}"
            )
        if not canonical:
            # Defensive: iterator should have skipped this row already.
            continue

        # Render+tokenize to get n_prompt_tokens. Mirrors what the
        # downstream calibration consumer will see (its first forward
        # pass renders+tokenizes the same messages list).
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": canonical},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=True,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        try:
            token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            n_prompt_tokens = len(token_ids)
        except Exception:  # noqa: BLE001 — tokenizer drift shouldn't tank row.
            n_prompt_tokens = 0

        has_think = _has_think_block(canonical)
        refusal_flag = _detect_refusal(canonical)

        stats = domain_stats.setdefault(domain, [0, 0])
        stats[0] += 1   # canonical rows are by construction complete
        stats[1] += 1

        yield {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": canonical},
            ],
            "domain": domain,
            "_complete": True,
            "_attempt_idx": int(attempt_idx),
            # v8 metadata bundle (same fields as the GENERATE path).
            "n_prompt_tokens": int(n_prompt_tokens),
            "n_gen_tokens": 0,
            "has_think": bool(has_think),
            "refusal_flag": bool(refusal_flag),
            "subset": str(domain),
            "seed_idx": int(attempt_idx),
            # v9 (CALIBRATION_MIX_V2_PLAN.md Step 6): canonical-source
            # rows.
            "completion_source": "canonical",
        }


# ---------------------------------------------------------------------------
# v3 forward-only replay helpers — module scope so unit tests can exercise
# them without spinning up vLLM (test_replay_helpers.py).
# ---------------------------------------------------------------------------
def _render_row_for_replay(
    row: dict,
    tokenizer,
    max_model_len: int,
) -> "tuple[list[int], int] | None":
    """Tokenize a saved JSONL row for forward-only replay (v3).

    Renders messages=[{role:user, content:prompt},{role:assistant,
    content:answer}] through the chat template with
    add_generation_prompt=False, enable_thinking=True. Mirrors
    _synth_teacher_forced_rows lines 726-733 exactly, and works
    uniformly for completion_source="teacher_generated" and "canonical".

    Returns (token_ids: list[int], n_tokens: int) if
    n_tokens <= max_model_len. Returns None if over-length.
    Never silently truncates. Pure function; testable without GPU.
    """
    messages = row.get("messages", [])
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    token_ids: list[int] = tokenizer(
        rendered, add_special_tokens=False,
    )["input_ids"]
    if len(token_ids) > max_model_len:
        return None
    return token_ids, len(token_ids)


def _build_replay_subset_tally(
    rows: "list[dict]",
    token_counts: "list[int]",
) -> "dict[str, dict[str, int]]":
    """Per-subset token+row tally for successfully replayed rows.

    Args:
        rows: row dicts that passed the length gate and were submitted
              to vLLM (non-skipped rows only).
        token_counts: parallel list of token counts (same length as rows).

    Returns {subset_name: {"n_rows": int, "n_tokens": int}}.
    Pure function; testable without GPU.
    """
    tally: dict[str, dict[str, int]] = {}
    for row, n_tok in zip(rows, token_counts):
        subset = str(row.get("subset") or row.get("domain") or "unknown")
        entry = tally.setdefault(subset, {"n_rows": 0, "n_tokens": 0})
        entry["n_rows"] += 1
        entry["n_tokens"] += n_tok
    return tally


def _assert_code_science_nonzero(
    tally: "dict[str, dict[str, int]]",
    code_subsets: "frozenset[str]" = frozenset({"mot_code", "swe_smith"}),
    science_subsets: "frozenset[str]" = frozenset({"mot_science"}),
) -> None:
    """Correctness gate: assert code and science subsets contributed tokens.

    The purpose of v3 replay is to cover code/science which TEACHER_FORCED
    rows never covered in v2. Zero tokens in both means the JSONL is not
    the v2 corpus or the subset names changed.

    Logs full per-subset breakdown before asserting. Pure function.
    Raises AssertionError with an actionable message on failure.
    """
    code_tokens = sum(
        tally[s]["n_tokens"] for s in code_subsets if s in tally
    )
    sci_tokens = sum(
        tally[s]["n_tokens"] for s in science_subsets if s in tally
    )
    log.info("replay tally by subset (replayed rows only):")
    for subset, counts in sorted(tally.items()):
        log.info(
            "  %-24s  %5d rows  %9d tokens",
            subset, counts["n_rows"], counts["n_tokens"],
        )
    log.info(
        "replay: code subsets %s -> %d tokens; "
        "science subsets %s -> %d tokens",
        sorted(code_subsets & tally.keys()), code_tokens,
        sorted(science_subsets & tally.keys()), sci_tokens,
    )
    assert code_tokens > 0, (
        f"replay correctness gate FAILED: no code-subset tokens captured. "
        f"Checked: {sorted(code_subsets)}. "
        f"Present in corpus: {sorted(tally)}. "
        f"Verify input JSONL is the v2 corpus and mot_code/swe_smith "
        f"rows have non-empty messages fields."
    )
    assert sci_tokens > 0, (
        f"replay correctness gate FAILED: no science-subset tokens captured. "
        f"Checked: {sorted(science_subsets)}. "
        f"Present in corpus: {sorted(tally)}."
    )


def _write_replay_ckpt(
    path: "Path",
    rows_done: int,
    captured_done: int,
) -> None:
    """Atomically write the replay row-index + capture-counter checkpoint.

    Two counters stored separately (M1 fix):
      rows_done     -- total rows consumed from the JSONL (replayed +
                       skipped). Used to slice the JSONL on resume.
                       Per-row n_skipped += 1 is the SOLE accumulator;
                       no other site may increment n_skipped.
      captured_done -- rows that contributed to captures (no skips).
                       Passed to _setup_all_writers as already_done for
                       per-writer ckpt counter validation.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"rows_done": rows_done, "captured_done": captured_done}),
        encoding="utf-8",
    )
    os.replace(tmp, path)


@dataclasses.dataclass
class _WriterState:
    """Per-writer checkpoint paths returned by _setup_all_writers.

    Every field is None when the corresponding --capture-* flag is off.
    Dataclass gives AttributeError on typo instead of silent None.

    Ten fields; layer_input_reservoir has no field because it rides
    inside stage2_profile with no separate dump call (H2).
    """
    imatrix_ckpt_path: "Path | None" = None
    reap_ckpt_path: "Path | None" = None
    input_cov_ckpt_path: "Path | None" = None
    wsr_ckpt_path: "Path | None" = None
    s2p_ckpt_path: "Path | None" = None
    pem_ckpt_path: "Path | None" = None
    rts_ckpt_path: "Path | None" = None
    router_logits_ckpt_path: "Path | None" = None
    or_ckpt_path: "Path | None" = None
    bo_ckpt_path: "Path | None" = None


def _setup_all_writers(
    args,
    out_path: "Path",
    llm,
    tokenizer,
    already_done: int,
) -> "_WriterState":
    """Pre-allocate all enabled capture writer accumulators.

    Extracted from main() (was inline lines 1659-2115) so both the
    generate path and _run_replay() call it identically.

    ``out_path`` determines checkpoint file locations:
      generate path: the output JSONL tmp path
      replay path  : the input JSONL (replay_jsonl)
    All per-writer .ckpt files land as siblings of out_path.

    ``already_done`` for the generate path: JSONL rows already written.
    For the replay path: captured_done (rows that contributed to
    captures, EXCLUDING skipped/over-length rows). This is the correct
    counter for _ckpt_counter_check which validates capture coverage.

    The _check_ckpt_counter closure from main() (~line 1659) is
    recreated as a lambda capturing already_done and
    args.allow_counter_divergence from the caller's scope.

    Returns _WriterState with all ten ckpt path fields (None if disabled).
    """
    ws = _WriterState()

    def _check(signal_name: str, loaded_prompts: int, ckpt_path: "Path"):
        _ckpt_counter_check(
            signal_name, loaded_prompts, already_done, ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )

    # ---- imatrix --------------------------------------------------------
    if args.capture_imatrix:
        import vllm.calibration_imatrix as _im  # type: ignore
        _im.setup(llm)
        log.info("imatrix: setup complete -- accumulators pre-allocated")
        ws.imatrix_ckpt_path = out_path.with_suffix(".imatrix.ckpt")
        _ckpt_existence_check(
            "imatrix", args.capture_imatrix, already_done,
            ws.imatrix_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.imatrix_ckpt_path.exists():
            try:
                loaded = _im.load_imatrix_checkpoint(
                    str(ws.imatrix_ckpt_path))
                _check("imatrix", loaded, ws.imatrix_ckpt_path)
                log.info("imatrix: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "imatrix: ckpt schema mismatch (%s); deleting", exc)
                ws.imatrix_ckpt_path.unlink()

    # ---- reap-scores ----------------------------------------------------
    if args.capture_reap_scores:
        import vllm.calibration_reap_scores as _reap  # type: ignore
        _reap.setup(llm)
        log.info("reap-scores: setup complete")
        ws.reap_ckpt_path = out_path.with_suffix(".reap_scores.ckpt")
        _ckpt_existence_check(
            "reap_scores", args.capture_reap_scores, already_done,
            ws.reap_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.reap_ckpt_path.exists():
            try:
                loaded = _reap.load_reap_scores_checkpoint(
                    str(ws.reap_ckpt_path))
                _check("reap-scores", loaded, ws.reap_ckpt_path)
                log.info("reap-scores: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "reap-scores: ckpt schema mismatch (%s); deleting", exc)
                ws.reap_ckpt_path.unlink()

    # ---- input-covariance -----------------------------------------------
    if args.capture_input_covariance:
        import vllm.calibration_input_cov as _icov  # type: ignore
        _icov.setup(llm)
        log.info("input-cov: setup complete")
        ws.input_cov_ckpt_path = out_path.with_suffix(".input_cov.ckpt")
        _ckpt_existence_check(
            "input_covariance", args.capture_input_covariance, already_done,
            ws.input_cov_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.input_cov_ckpt_path.exists():
            try:
                loaded = _icov.load_input_cov_checkpoint(
                    str(ws.input_cov_ckpt_path))
                _check("input-cov", loaded, ws.input_cov_ckpt_path)
                log.info("input-cov: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "input-cov: ckpt schema mismatch (%s); deleting", exc)
                ws.input_cov_ckpt_path.unlink()

    # ---- wanda scalar_row -----------------------------------------------
    if args.capture_wanda_scalar_row:
        import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
        _wsr.setup(llm)
        log.info("wanda-scalar-row: setup complete")
        ws.wsr_ckpt_path = out_path.with_suffix(".wanda_scalar_row.ckpt")
        _ckpt_existence_check(
            "wanda_scalar_row", args.capture_wanda_scalar_row, already_done,
            ws.wsr_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.wsr_ckpt_path.exists():
            try:
                loaded = _wsr.load_wanda_scalar_row_checkpoint(
                    str(ws.wsr_ckpt_path))
                _check("wanda-scalar-row", loaded, ws.wsr_ckpt_path)
                log.info("wanda-scalar-row: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "wanda-scalar-row: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.wsr_ckpt_path.unlink()

    # ---- stage2-profile -------------------------------------------------
    if args.capture_stage2_profile:
        import vllm.calibration_stage2_profile as _s2p  # type: ignore
        _s2p.setup(llm,
                   cov_storage_dtype=args.stage2_profile_cov_storage_dtype)
        log.info("stage2-profile: setup complete (cov_dtype=%s)",
                 args.stage2_profile_cov_storage_dtype)
        ws.s2p_ckpt_path = out_path.with_suffix(".stage2_profile.ckpt")
        _ckpt_existence_check(
            "stage2_profile", args.capture_stage2_profile, already_done,
            ws.s2p_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.s2p_ckpt_path.exists():
            try:
                loaded = _s2p.load_stage2_profile_checkpoint(
                    str(ws.s2p_ckpt_path))
                _check("stage2-profile", loaded, ws.s2p_ckpt_path)
                log.info("stage2-profile: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "stage2-profile: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.s2p_ckpt_path.unlink()

    # ---- per-expert-max -------------------------------------------------
    if args.capture_per_expert_max:
        import vllm.calibration_per_expert_max as _pem  # type: ignore
        _pem.setup(llm)
        log.info("per-expert-max: setup complete")
        ws.pem_ckpt_path = out_path.with_suffix(".per_expert_max.ckpt")
        _ckpt_existence_check(
            "per_expert_max", args.capture_per_expert_max, already_done,
            ws.pem_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.pem_ckpt_path.exists():
            try:
                loaded = _pem.load_per_expert_max_checkpoint(
                    str(ws.pem_ckpt_path))
                _check("per-expert-max", loaded, ws.pem_ckpt_path)
                log.info("per-expert-max: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "per-expert-max: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.pem_ckpt_path.unlink()

    # ---- routing-stats --------------------------------------------------
    if args.capture_routing_stats:
        import vllm.calibration_routing_stats as _rts  # type: ignore
        _rts.setup(llm)
        log.info("routing-stats: setup complete")
        ws.rts_ckpt_path = out_path.with_suffix(".routing_stats.ckpt")
        _ckpt_existence_check(
            "routing_stats", args.capture_routing_stats, already_done,
            ws.rts_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.rts_ckpt_path.exists():
            try:
                loaded = _rts.load_routing_stats_checkpoint(
                    str(ws.rts_ckpt_path))
                _check("routing-stats", loaded, ws.rts_ckpt_path)
                log.info("routing-stats: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "routing-stats: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.rts_ckpt_path.unlink()

    # ---- router-logits-stats --------------------------------------------
    if args.capture_router_logits_stats:
        import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
        _bos = getattr(tokenizer, "bos_token_id", None)
        _rlsx.setup(llm, bos_token_id=_bos)
        log.info("router-logits-stats: setup complete (bos_token_id=%s)",
                 _bos)
        ws.router_logits_ckpt_path = out_path.with_suffix(
            ".router_logits_stats.ckpt")
        _ckpt_existence_check(
            "router_logits_stats", args.capture_router_logits_stats,
            already_done, ws.router_logits_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.router_logits_ckpt_path.exists():
            try:
                loaded = _rlsx.load_router_logits_stats_checkpoint(
                    str(ws.router_logits_ckpt_path))
                _check("router-logits-stats", loaded,
                       ws.router_logits_ckpt_path)
                log.info("router-logits-stats: hydrated %d-prompt ckpt",
                         loaded)
            except ValueError as exc:
                log.error(
                    "router-logits-stats: ckpt schema mismatch (%s); "
                    "deleting", exc)
                ws.router_logits_ckpt_path.unlink()

    # ---- output-reservoir -----------------------------------------------
    if args.capture_output_reservoir:
        import vllm.calibration_output_reservoir as _or  # type: ignore
        _or.setup(llm)
        log.info("output-reservoir: setup complete (cap=%d)",
                 args.output_reservoir_cap)
        ws.or_ckpt_path = out_path.with_suffix(".output_reservoir.ckpt")
        _ckpt_existence_check(
            "output_reservoir", args.capture_output_reservoir, already_done,
            ws.or_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.or_ckpt_path.exists():
            try:
                loaded = _or.load_output_reservoir_checkpoint(
                    str(ws.or_ckpt_path))
                _check("output-reservoir", loaded, ws.or_ckpt_path)
                log.info("output-reservoir: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "output-reservoir: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.or_ckpt_path.unlink()

    # ---- block-outputs --------------------------------------------------
    if args.capture_block_outputs:
        import vllm.calibration_block_outputs as _bo  # type: ignore
        _bo.setup(llm)
        log.info("block-outputs: setup complete (subset_size=%d)",
                 args.block_outputs_subset_size)
        ws.bo_ckpt_path = out_path.with_suffix(".block_outputs.ckpt")
        _ckpt_existence_check(
            "block_outputs", args.capture_block_outputs, already_done,
            ws.bo_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.bo_ckpt_path.exists():
            try:
                loaded = _bo.load_block_outputs_checkpoint(
                    str(ws.bo_ckpt_path))
                _check("block-outputs", loaded, ws.bo_ckpt_path)
                log.info(
                    "block-outputs: hydrated %d-prompt ckpt (closed=%s)",
                    loaded, _bo._SUBSET_CLOSED)
            except ValueError as exc:
                log.error(
                    "block-outputs: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.bo_ckpt_path.unlink()

    return ws


# ---------------------------------------------------------------------------
# Free helpers — exposed at module scope so unit tests can exercise them
# without spinning up the full vLLM pipeline. (NIT-4 / LOW-4 fold.)
# ---------------------------------------------------------------------------
def _ckpt_counter_check(
    signal_name: str,
    loaded_prompts: int,
    already_done: int,
    ckpt_path: "Path",
    *,
    allow_counter_divergence: bool,
    log_: logging.Logger | None = None,
) -> None:
    """F-H-6 enforcement, extracted to module scope (NIT-4 / LOW-4).

    Hard-fail (default) or WARN-only (with ``allow_counter_divergence=True``)
    when an accumulator checkpoint and the JSONL row count disagree. The
    function is pure — it takes counts + flags as args and raises
    ``ValueError`` on a hard fail. The in-``main`` closure
    ``_check_ckpt_counter`` (kept for backward-compat with the existing
    call sites that already capture ``args`` and ``already_done`` from
    the enclosing scope) now forwards to this free function.

    Raises:
        ValueError: when loaded_prompts != already_done and
            allow_counter_divergence is False. Message includes the
            checkpoint path so operators know what to delete.
    """
    if loaded_prompts == already_done:
        return
    msg = (
        f"{signal_name}: checkpoint has {loaded_prompts} prompts "
        f"but JSONL has {already_done} rows. A SIGKILL between "
        f"JSONL flush and the next ckpt dump silently undercounts "
        f"the accumulator (the JSONL claims more prompts than the "
        f"accumulator saw)."
    )
    if allow_counter_divergence:
        (log_ or log).warning(
            "%s Proceeding with the smaller counter "
            "(--allow-counter-divergence is set). Sidecar will be "
            "computed over a SUBSET of the calibration data.",
            msg,
        )
        return
    raise ValueError(
        f"{msg} Delete the checkpoint file ({ckpt_path}) so the "
        "accumulator restarts from zero and re-walks the prompts "
        "from this run's resume base, OR re-run with "
        "--allow-counter-divergence to tolerate the under-count."
    )


def _ckpt_existence_check(
    signal_name: str,
    capture_enabled: bool,
    already_done: int,
    ckpt_path: "Path",
    *,
    allow_counter_divergence: bool,
    log_: logging.Logger | None = None,
) -> None:
    """C-1 enforcement: hard-fail when a writer is fresh-on-resume.

    Fires when ``args.resume`` is set, the operator enabled the writer
    (``capture_enabled=True``), the JSONL has rows from a prior session
    (``already_done > 0``), but the writer's checkpoint does not exist.
    In that state, the writer would start from zero and silently
    under-cover the calibration data, while the sidecar's eventual
    ``n_prompts_accumulated = already_done + n_new`` would inflate the
    coverage claim. Refuse to start unless ``--allow-counter-divergence``
    is set (in which case warn).

    Raises:
        ValueError: when capture_enabled is True, already_done > 0,
            the checkpoint does not exist, and
            allow_counter_divergence is False. Message names the
            writer, the row count, the checkpoint path, and the
            escape-hatch flag so operators have an actionable
            recovery path.
    """
    if not capture_enabled or already_done == 0 or ckpt_path.exists():
        return
    msg = (
        f"{signal_name}: --capture-{signal_name} enabled on a resume "
        f"with already_done={already_done} JSONL rows, but no "
        f"checkpoint exists at {ckpt_path}. The writer would silently "
        f"under-cover the prior {already_done} prompts; the sidecar's "
        f"n_prompts_accumulated would claim full coverage."
    )
    if allow_counter_divergence:
        (log_ or log).warning(
            "%s Proceeding with the smaller actual coverage "
            "(--allow-counter-divergence is set). Sidecar metadata will "
            "OVER-state coverage; downstream consumers that rely on "
            "n_prompts_accumulated for normalization will be biased.",
            msg,
        )
        return
    raise ValueError(
        f"{msg} Either re-run without --resume to start fresh, OR "
        f"delete the .jsonl.tmp file ({ckpt_path.parent}) so the "
        f"writer starts from prompt 0, OR pass "
        f"--allow-counter-divergence to tolerate the under-coverage."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--teacher-revision", default="main")
    p.add_argument("--prompts", default="qwen3-pretrain-mix",
                   help="'qwen3-pretrain-mix' (v1, 8 subsets, all GENERATE), "
                        "'qwen3-pretrain-mix-v2' (12 subsets, hybrid GENERATE + "
                        "TEACHER_FORCED — see tasks/CALIBRATION_MIX_V2_PLAN.md), "
                        "or path to JSONL with {'prompt': '...', "
                        "'domain': '...'} rows.")
    p.add_argument("--num-prompts", type=int, default=6500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-new-tokens", type=int, default=16384,
                   help="Hard token cap per row.")
    p.add_argument("--reasoning-budget", type=int, default=4096,
                   help="vLLM reasoning_budget — forces </think> after N "
                        "tokens inside <think>...</think>. Caps overthinking "
                        "without dropping --max-new-tokens (which still bounds "
                        "the post-</think> answer block).")
    p.add_argument("--logits-top-k", type=int, default=50,
                   help="K for the teacher-logit topk cache. vLLM returns "
                        "this many logprobs per generated position natively.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "auto", "fp8"],
                   help="Teacher precision. bf16 is the determinism reference; "
                        "fp8 fits on smaller GPUs but yields slightly different "
                        "logits (cache_key folds dtype, so no collision).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="Fraction of free GPU VRAM vLLM reserves at startup.")
    p.add_argument("--max-model-len", type=int, default=20480,
                   help="vLLM context-length budget = prompt (≤2048) + "
                        "max_new_tokens. Slightly larger than the sum to "
                        "give vLLM scheduler headroom.")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="Cap on concurrent sequences vLLM batches in flight. "
                        "Default (None) uses vLLM's built-in default (256). "
                        "Bump on large VRAM (e.g. 384-512 on H200) when "
                        "steady-state VRAM observation shows >30 GB free. "
                        "Does NOT change output bytes (vLLM scheduling is "
                        "deterministic under fixed seed + temp=0), so OK to "
                        "tune across runs without invalidating cache_key.")
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="Cap on total tokens scheduled per forward pass. "
                        "Default (None) uses vLLM's default. Higher values "
                        "improve prefill GPU utilization on H200/B300 but "
                        "trade off latency. Like --max-num-seqs, doesn't "
                        "alter output bytes.")
    p.add_argument("--moe-backend", type=str, default="triton",
                   help="Force vLLM's fused-MoE backend via "
                        "kernel_config={'moe_backend': <val>}. Default "
                        "'triton' is REQUIRED for the in-graph calibration "
                        "capture path: the kernel-interior signals "
                        "(reap_scores / per_expert_max / output_reservoir / "
                        "block_outputs) only dispatch inside TritonExperts; "
                        "the 'auto' default would pick FlashInfer on "
                        "Hopper/Blackwell and every capture sidecar would be "
                        "empty (B0 failure class). Pass 'auto' to restore "
                        "vLLM's own backend selection (only safe for "
                        "pure-generate runs with no --capture-* flags).")
    p.add_argument("--chunk-size", type=int, default=200,
                   help="How many prompts to submit per LLM.generate call. "
                        "Affects crash-recovery granularity, not throughput "
                        "(vLLM continuous-batches internally).")
    p.add_argument("--output", default="artifacts/_shared/self_traces.jsonl")
    p.add_argument("--no-cache-suffix", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts that already have rows in the .tmp file.")
    p.add_argument(
        "--allow-counter-divergence",
        action="store_true", default=False,
        help=(
            "F-H-6 escape hatch: by default, when an accumulator "
            "checkpoint's prompt counter disagrees with the JSONL row "
            "count on resume (indicating SIGKILL between JSONL flush "
            "and ckpt dump → silently-undercounted accumulator), the "
            "script hard-fails with a ValueError instructing the "
            "operator to delete the affected .ckpt file and re-run. "
            "Passing this flag downgrades that to a WARNING and "
            "proceeds with the smaller counter — the legacy behavior. "
            "Recommended ONLY for ablation sweeps where minor "
            "under-counting is tolerable; production runs should "
            "keep the default hard-fail. "
            "Also downgrades the C-1 (fresh-writer-on-resume) abort "
            "to a WARN: when a writer's --capture-X flag is enabled "
            "on a resume but no prior checkpoint exists, the writer "
            "integrates over only the post-resume chunks (a SUBSET "
            "of the calibration data); the sidecar's "
            "n_prompts_accumulated metadata will OVER-state coverage."
        ),
    )
    p.add_argument("--prev-num-prompts", type=int, default=0,
                   help="Per-subset extension mode: yield ONLY the prompts that "
                        "an earlier --num-prompts=N run would NOT have yielded. "
                        "For each subset in qwen3-pretrain-mix, the iterator "
                        "computes its previous per-subset count from N (=PREV) "
                        "and its target count from --num-prompts (=NEW), then "
                        "yields rows at deterministic-shuffle positions "
                        "[prev_count, new_count). The resulting prompts are by "
                        "construction non-overlapping with the earlier run. "
                        "Cache_key incorporates this so the output file is "
                        "distinct from the prev=0 run with the same --num-prompts.")
    p.add_argument("--shuffle-buffer", type=int, default=0,
                   help="Fix the HF streaming shuffle buffer_size INDEPENDENTLY "
                        "of --num-prompts (clamped to [10000, 200000]). Default "
                        "0 = unset = the historical count-derived formula "
                        "min(max(10000, 10*per_subset_count), 200000), which is "
                        "count-DEPENDENT. REQUIRED for data-parallel generate "
                        "mode: the per-subset shuffle order must be IDENTICAL "
                        "across all N processes for the --prev-num-prompts "
                        "offset slices to align. The generate orchestration "
                        "computes ONE buffer from the GLOBAL total and passes "
                        "the SAME --shuffle-buffer (and --seed) to every shard. "
                        "Generate-mode INVARIANT: shards MUST share seed AND "
                        "shuffle-buffer. Only affects the mix iterators; "
                        "replay/JSONL paths ignore it.")
    p.add_argument("--capture-imatrix", action="store_true", default=False,
                   help="Capture per-input-channel squared-activation statistics "
                        "for every linear layer reached during calibration and "
                        "write a llama.cpp-compatible '.imatrix.dat' sidecar at "
                        "run end. Requires the vLLM calibration-hooks patch "
                        "(vllm.calibration_imatrix). Auto-enables "
                        "VLLM_CALIB_CAPTURE_IMATRIX=1, VLLM_CALIB_CAPTURE_EXPERT=1, "
                        "and VLLM_CALIB_CAPTURE_EXPERT_MID=1 BEFORE any vllm "
                        "import (the gates are sampled at "
                        "vllm.calibration_hooks module load). The sidecar is "
                        "written next to the JSONL with extension '.imatrix.dat'. "
                        "Failures during dump are logged but do NOT re-raise -- "
                        "the JSONL is more valuable than the imatrix.")
    p.add_argument("--imatrix-checkpoint-every-chunks", type=int, default=1,
                   help="When --capture-imatrix is set, dump a checkpoint "
                        "(.imatrix.ckpt) of the live accumulator state every "
                        "N chunked LLM.generate calls. Default 1 = checkpoint "
                        "at every JSONL flush boundary, matching the existing "
                        "crash-recovery granularity. Set 0 to disable periodic "
                        "checkpointing (final-dump-only). On --resume, the "
                        "checkpoint at <jsonl>.imatrix.ckpt is hydrated "
                        "automatically if it exists.")
    p.add_argument("--capture-reap-scores", action="store_true", default=False,
                   help="Capture per-(layer, expert) REAP saliency scores "
                        "(S_j = (1/|X_j|)·Σ g_j(x)·‖f_j(x)‖₂, arXiv:2510.13999 "
                        "Eq. 9) during calibration and write a "
                        "moe_compress-side sidecar at "
                        "<jsonl>/sidecars/reap_scores.pt at run end. Requires "
                        "the vLLM calibration-hooks patch "
                        "(vllm.calibration_reap_scores). Auto-enables "
                        "VLLM_CALIB_CAPTURE_REAP_SCORES=1, "
                        "VLLM_CALIB_CAPTURE_ROUTER=1, "
                        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1, AND "
                        "VLLM_USE_FLASHINFER_MOE_FP16=0 (forces Triton MoE "
                        "backend; FlashInfer's monolithic path does not "
                        "expose expert_out_unweighted) BEFORE any vllm "
                        "import. Failures during dump are logged but do NOT "
                        "re-raise -- the JSONL is more valuable than the "
                        "REAP sidecar.")
    p.add_argument("--reap-scores-checkpoint-every-chunks", type=int, default=1,
                   help="When --capture-reap-scores is set, dump a checkpoint "
                        "(.reap_scores.ckpt) of the live REAP accumulator "
                        "state every N chunked LLM.generate calls. Default 1 "
                        "= checkpoint at every JSONL flush boundary. Set 0 to "
                        "disable periodic checkpointing (final-dump-only). On "
                        "--resume, the checkpoint at <jsonl>.reap_scores.ckpt "
                        "is hydrated automatically if it exists.")
    p.add_argument("--capture-input-covariance", action="store_true",
                   default=False,
                   help="Capture per-(layer, expert, 'gate_proj') teacher "
                        "input covariance Σ_in = Σ_t x_t^T x_t during "
                        "calibration and write a moe_compress-side sidecar at "
                        "<jsonl>/sidecars/covariance.pt at run end "
                        "(schema v2, dict-shaped, byte-compatible with the "
                        "Stage 2 writer's _stage2_input_covariance.pt). "
                        "Requires the vLLM calibration-hooks patch "
                        "(vllm.calibration_input_cov). Auto-enables "
                        "VLLM_CALIB_CAPTURE_INPUT_COV=1 and "
                        "VLLM_CALIB_CAPTURE_EXPERT=1 (so the expert_in hook "
                        "fires) BEFORE any vllm import. Failures during dump "
                        "are logged but do NOT re-raise -- the JSONL is more "
                        "valuable than the covariance sidecar.")
    p.add_argument("--input-cov-checkpoint-every-chunks", type=int, default=1,
                   help="When --capture-input-covariance is set, dump a "
                        "checkpoint (.input_cov.ckpt) of the live covariance "
                        "accumulator state every N chunked LLM.generate "
                        "calls. Default 1 = checkpoint at every JSONL flush "
                        "boundary. Set 0 to disable periodic checkpointing "
                        "(final-dump-only). On --resume, the checkpoint at "
                        "<jsonl>.input_cov.ckpt is hydrated automatically "
                        "if it exists.")
    p.add_argument("--input-cov-offload", action="store_true", default=False,
                   help="Capture input_covariance via the per-layer CPU-offload "
                        "(windowed-resident) path instead of the all-resident "
                        "setup. REQUIRED for the full Gram on a big model: the "
                        "all-resident Gram is ~172 GB (40 layers x 256 experts x "
                        "2048^2 x 4B) and OOMs. The offload allocates only a "
                        "WINDOW of MoE layers at a time, early-exits the forward "
                        "after the window's top layer (set_calibration_max_layer), "
                        "snapshots each window's Gram to CPU, frees the GPU "
                        "slice, and advances -- re-running the corpus once per "
                        "window. Only valid in --replay-from mode with "
                        "--capture-input-covariance as the SOLE capture flag "
                        "(the per-window early-exit is incompatible with the "
                        "other full-forward signals). Forces "
                        "VLLM_CALIB_INPUT_COV_MODE=resident.")
    p.add_argument("--input-cov-window-size", type=int, default=0,
                   help="Number of MoE layers allocated resident per offload "
                        "window. 0 (default) = auto-size from free GPU memory "
                        "after model load (free // per-layer-Gram-bytes, with "
                        "headroom). Smaller windows = lower peak VRAM but more "
                        "corpus passes (cost ~ sum(window_top_layer)/n_layers x "
                        "full forward). Only used with --input-cov-offload.")
    # W-1: Wanda scalar_row sidecar (audit/PLAN_W1).
    p.add_argument("--capture-wanda-scalar-row", action="store_true",
                   default=False,
                   help="Capture Wanda scalar_row = E[(x*g_e)^2] per "
                        "(layer, expert, gate_proj) during calibration; "
                        "write sidecar wanda_scalar_row.pt (schema v1). "
                        "Auto-enables "
                        "VLLM_CALIB_CAPTURE_{WANDA_SCALAR_ROW,ROUTER,EXPERT}=1. "
                        "Full contract: vllm.calibration_wanda_scalar_row "
                        "module docstring.")
    p.add_argument("--wanda-scalar-row-checkpoint-every-chunks", type=int,
                   default=1, help="When --capture-wanda-scalar-row is set, "
                                   "dump a .wanda_scalar_row.ckpt every N "
                                   "chunks; mirrors "
                                   "--input-cov-checkpoint-every-chunks.")
    # Plugin #12 REDO -- Optimization A profile-pass sidecar.
    p.add_argument("--capture-stage2-profile", action="store_true",
                   default=False,
                   help="Capture Stage 2 REAM profile (gate-logit Gram + gated "
                        "outputs + gate_proj/down_proj covariance + layer-input "
                        "reservoir) and write a sidecar at "
                        "<jsonl>/sidecars/stage2_profile.pt "
                        "(schema v4). Requires the vLLM patch "
                        "vllm.calibration_stage2_profile (canonical source: "
                        "moe_compress.calibration.stage2_profile_writer). "
                        "Auto-enables VLLM_CALIB_CAPTURE_STAGE2_PROFILE=1 + "
                        "VLLM_CALIB_CAPTURE_ROUTER=1 + "
                        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                        "VLLM_CALIB_CAPTURE_EXPERT=1 (gate_proj cov) + "
                        "VLLM_CALIB_CAPTURE_EXPERT_MID=1 (down_proj cov) + "
                        "VLLM_USE_FLASHINFER_MOE_FP16=0 BEFORE any vllm "
                        "import. DISK: the gate+down per-(layer,expert) cov "
                        "adds ~91 GB at cov_storage_dtype=float16 for a "
                        "256-expert / 40-layer / hidden=2048 / "
                        "moe_intermediate=512 model (size it accordingly); "
                        "this is the price of skipping the live Stage-2 cov "
                        "forward on full-hit layers. NOTE: layer-input "
                        "reservoir is reserved "
                        "for future use; current production sidecars omit "
                        "it pending a `layer_in` callback hook in the vLLM "
                        "patch. Until then, SC cost_alignment='output' will "
                        "fall back to the live forward pass on full-hit "
                        "layers (the reader skips reservoir hydration when "
                        "the payload entry is empty). Failures during dump "
                        "are logged but do NOT re-raise.")
    p.add_argument("--stage2-profile-checkpoint-every-chunks", type=int,
                   default=1,
                   help="When --capture-stage2-profile is set, dump a "
                        "checkpoint (.stage2_profile.ckpt) every N chunks. "
                        "Default 1. Set 0 to disable. On --resume, the "
                        "checkpoint is hydrated automatically.")
    p.add_argument("--stage2-profile-cov-storage-dtype", type=str,
                   default="float16",
                   choices=["float16", "bfloat16", "float32"],
                   help="When --capture-stage2-profile is set, configure "
                        "the InputCovarianceAccumulator.storage_dtype used "
                        "by the writer. MUST MATCH the Stage 2 config's "
                        "covariance_storage_dtype (default 'float16' per "
                        "stage2 orchestrator). On mismatch the reader "
                        "fails loud at load time with 'Delete the sidecar "
                        "to regenerate'.")
    # CRITICAL-1: vLLM `layer_in` hook landing -- writer subscription
    # populates the previously-empty layer_input_reservoir field of the
    # stage2_profile sidecar. Production default flipped to ON after
    # the hook landed in the preceding commit (per plan §3.c / user
    # OQ Q3 = same-branch follow-up commit).
    p.add_argument("--capture-layer-input-reservoir",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="When --capture-stage2-profile is set, ALSO "
                        "populate the layer_input_reservoir field of "
                        "the stage2_profile sidecar. Requires the vLLM "
                        "patch's layer_in hook (vllm.calibration_hooks "
                        "VALID_HOOK_NAMES). Auto-enables "
                        "VLLM_CALIB_CAPTURE_LAYER_IN=1 BEFORE any vllm "
                        "import. Default ON (post-CRITICAL-1 production "
                        "flip): the layer_in hook is now part of the "
                        "patch and the writer subscribes via "
                        "_layer_in_handler. Use "
                        "--no-capture-layer-input-reservoir to opt out "
                        "(e.g. to reproduce the legacy empty-placeholder "
                        "sidecar shape for back-compat experiments). "
                        "Unblocks SC cost_alignment='output' full-hit "
                        "path (Plugin #12 + Plugin #1 combined ~37-57 "
                        "min/SC row saved per PLAN_PLUGIN_14 §5 row 1).")
    p.add_argument("--capture-per-expert-max", action="store_true",
                   default=False,
                   help="Capture per-(layer, expert) down_proj output max "
                        "L_inf during calibration and write a moe_compress-"
                        "side sidecar at <jsonl>/sidecars/per_expert_max.pt "
                        "at run end (schema v1, shape [n_layers, n_experts] "
                        "float32). Consumed by Stage 1's three-way / "
                        "magnitude-topk / ablation_filter cheap-pruning "
                        "scoring. Requires the vLLM calibration-hooks patch "
                        "(vllm.calibration_per_expert_max). Auto-enables "
                        "VLLM_CALIB_CAPTURE_PER_EXPERT_MAX=1 + "
                        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                        "VLLM_USE_FLASHINFER_MOE_FP16=0 BEFORE any vllm "
                        "import (Triton MoE backend is required). Failures "
                        "during dump are logged but do NOT re-raise.")
    p.add_argument("--per-expert-max-checkpoint-every-chunks", type=int,
                   default=1,
                   help="When --capture-per-expert-max is set, dump a "
                        "checkpoint (.per_expert_max.ckpt) of the live "
                        "accumulator state every N chunked LLM.generate "
                        "calls. Default 1 = checkpoint at every JSONL flush "
                        "boundary. Set 0 to disable periodic checkpointing "
                        "(final-dump-only). On --resume, the checkpoint at "
                        "<jsonl>.per_expert_max.ckpt is hydrated "
                        "automatically if it exists.")
    p.add_argument("--capture-routing-stats", action="store_true",
                   default=False,
                   help="Capture per-(layer, expert) routing frequency + "
                        "mean routing weight during calibration and write a "
                        "moe_compress-side sidecar at "
                        "<jsonl>/sidecars/routing_stats.pt at run end "
                        "(schema v1, shape [n_layers, n_experts] int64 freq + "
                        "float32 mean_weight). Item 3 of the calibration-v2 "
                        "writers campaign. NOTE: as of 2026-05 there is "
                        "still NO production consumer of "
                        "ctx.routing_stats_payload -- the writer + Stage 1/2 "
                        "cache readers are infrastructure-only, awaiting "
                        "the planned routing-aware ablation gating / "
                        "mean-weight-weighted REAP variant plugins. Skip "
                        "this flag unless a consumer has landed. Requires the "
                        "vLLM calibration-hooks patch "
                        "(vllm.calibration_routing_stats). Auto-enables "
                        "VLLM_CALIB_CAPTURE_ROUTING_STATS=1 + "
                        "VLLM_CALIB_CAPTURE_ROUTER=1 BEFORE any vllm import. "
                        "Works on EVERY MoE backend (no FlashInfer or "
                        "EXPERT_UNWEIGHTED requirement -- the router hook "
                        "fires regardless). Failures during dump are logged "
                        "but do NOT re-raise.")
    p.add_argument("--routing-stats-checkpoint-every-chunks", type=int,
                   default=1,
                   help="When --capture-routing-stats is set, dump a "
                        "checkpoint (.routing_stats.ckpt) of the live "
                        "accumulator state every N chunked LLM.generate "
                        "calls. Default 1 = checkpoint at every JSONL flush "
                        "boundary. Set 0 to disable periodic checkpointing "
                        "(final-dump-only). On --resume, the checkpoint at "
                        "<jsonl>.routing_stats.ckpt is hydrated "
                        "automatically if it exists.")
    p.add_argument("--capture-router-logits-stats", action="store_true",
                   default=False,
                   help="Capture per-(layer, expert) sink-vs-normal "
                        "router-score aggregates during calibration and "
                        "write a moe_compress-side sidecar at "
                        "<jsonl>/sidecars/router_logits_stats.pt at run "
                        "end (schema v1, per-(layer, expert) "
                        "score_sink_sum / score_normal_sum float32 + "
                        "fire_on_sink int64 + per-layer n_sink_tokens / "
                        "n_normal_tokens int64 + bos_token_id). Item 4 "
                        "of the calibration-v2 writers campaign. "
                        "Hydrates Stage 1's SinkTokenRoutingAccumulator "
                        "from the sidecar, allowing the sink-token "
                        "detector to skip its live router-logits + "
                        "softmax + top-k accumulator pass entirely. "
                        "Auto-enables VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS=1 + "
                        "VLLM_CALIB_CAPTURE_ROUTER=1 BEFORE any vllm "
                        "import. Works on EVERY MoE backend (no FlashInfer "
                        "or EXPERT_UNWEIGHTED requirement -- the router "
                        "hook fires regardless). Failures during dump are "
                        "logged but do NOT re-raise.")
    p.add_argument("--router-logits-stats-checkpoint-every-chunks", type=int,
                   default=1,
                   help="When --capture-router-logits-stats is set, dump "
                        "a checkpoint (.router_logits_stats.ckpt) of the "
                        "live accumulator state every N chunked "
                        "LLM.generate calls. Default 1 = checkpoint at "
                        "every JSONL flush boundary. Set 0 to disable "
                        "periodic checkpointing (final-dump-only). On "
                        "--resume, the checkpoint at "
                        "<jsonl>.router_logits_stats.ckpt is hydrated "
                        "automatically if it exists.")
    p.add_argument("--capture-output-reservoir", action="store_true",
                   default=False,
                   help="Capture per-(layer, expert) unweighted expert-"
                        "output reservoirs during calibration and write a "
                        "moe_compress-side sidecar at "
                        "<jsonl>/sidecars/output_reservoir.pt at run end "
                        "(schema v1, dense [n_layers, n_experts, "
                        "max_tokens, hidden] bf16 tensor + per-(layer, "
                        "expert) valid_count / total_seen int64 + "
                        "max_tokens scalar). Item 6 of the calibration-v2 "
                        "writers campaign. Hydrates Stage 1's "
                        "ExpertOutputAccumulator from the sidecar, "
                        "allowing the CKADistancePlugin to skip its live "
                        "Phase-B reservoir-build forward pass entirely. "
                        "Auto-enables VLLM_CALIB_CAPTURE_OUTPUT_RESERVOIR=1 + "
                        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                        "VLLM_USE_FLASHINFER_MOE_FP16=0 + "
                        "VLLM_CALIB_OUTPUT_RESERVOIR_CAP=<value> BEFORE "
                        "any vllm import. Failures during dump are "
                        "logged but do NOT re-raise.")
    p.add_argument("--output-reservoir-cap", type=int, default=256,
                   help="Per-(layer, expert) reservoir capacity (max "
                        "tokens stored per cell). Mirrors "
                        "ExpertOutputAccumulator.max_tokens_per_expert "
                        "(default 256). Increasing this linearly scales "
                        "both peak memory during the run and the on-disk "
                        "sidecar size. Only consulted when "
                        "--capture-output-reservoir is set; baked into "
                        "VLLM_CALIB_OUTPUT_RESERVOIR_CAP before vllm "
                        "import.")
    p.add_argument("--output-reservoir-checkpoint-every-chunks", type=int,
                   default=1,
                   help="When --capture-output-reservoir is set, dump a "
                        "checkpoint (.output_reservoir.ckpt) of the live "
                        "reservoir state every N chunked LLM.generate "
                        "calls. Default 1 = checkpoint at every JSONL "
                        "flush boundary. Set 0 to disable periodic "
                        "checkpointing (final-dump-only). On --resume, "
                        "the checkpoint at "
                        "<jsonl>.output_reservoir.ckpt is hydrated "
                        "automatically if it exists.")
    p.add_argument("--capture-block-outputs", action="store_true",
                   default=False,
                   help="Capture per-MoE-block output hidden states on a "
                        "FIXED subset (size controlled by "
                        "--block-outputs-subset-size, default 128) during "
                        "calibration and write per-layer sidecars at "
                        "<jsonl>/sidecars/block_hidden/layer_{idx:04d}.pt "
                        "at run end (schema v1, [n_tokens, hidden] bf16 "
                        "tensor + layer_idx + n_prompts_in_subset). Item 7 "
                        "of the calibration-v2 writers campaign. Hydrates "
                        "Stage 3 block_refine's teacher targets so the "
                        "live teacher block forward can be skipped. "
                        "Auto-enables VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS=1 + "
                        "VLLM_CALIB_CAPTURE_BLOCK=1 + "
                        "VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE=<value> "
                        "BEFORE any vllm import. After "
                        "--block-outputs-subset-size prompts have been "
                        "processed the driver calls "
                        "vllm.calibration_block_outputs.close_subset() "
                        "to lock the accumulators so subsequent "
                        "block_out dispatches are no-ops. Failures "
                        "during dump are logged but do NOT re-raise.")
    p.add_argument("--block-outputs-subset-size", type=int, default=128,
                   help="Number of prompts to include in the block-"
                        "outputs subset. Matches the calibration-v2 "
                        "campaign plan's documented 128-prompt size. "
                        "Larger values linearly grow the on-disk per-"
                        "layer sidecar size (e.g., Qwen3-30B-A3B at 128 "
                        "prompts × ~2700 tokens × 48 layers × 2048 "
                        "hidden × 2 bytes bf16 ≈ 64 GiB total). Only "
                        "consulted when --capture-block-outputs is "
                        "set; baked into "
                        "VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE before "
                        "vllm import.")
    p.add_argument("--block-outputs-checkpoint-every-chunks", type=int,
                   default=1,
                   help="When --capture-block-outputs is set, dump a "
                        "checkpoint (.block_outputs.ckpt) of the live "
                        "per-rank accumulator state every N chunked "
                        "LLM.generate calls. Default 1 = checkpoint at "
                        "every JSONL flush boundary. Set 0 to disable "
                        "periodic checkpointing (final-dump-only). On "
                        "--resume, the checkpoint at "
                        "<jsonl>.block_outputs.ckpt is hydrated "
                        "automatically if it exists (including the "
                        "subset-closed flag).")
    p.add_argument("--allow-empty-captures", action="store_true",
                   default=False,
                   help="B0 fail-fast bypass. By default, after the first "
                        "chunk that actually ran vLLM generate(), the driver "
                        "asserts every ENABLED --capture-* writer has a "
                        "nonzero captured_entry_count() and SystemExit(2)s "
                        "if any enabled capture is still empty (a hook/model "
                        "mismatch -- see tasks/PLAN_B0_HOOK_FIX.md). Set this "
                        "flag to downgrade that abort to a warning (debug "
                        "only).")
    p.add_argument("--replay-from", type=str, default=None,
                   help="v3 forward-only replay mode. Path to an existing "
                        "self-traces JSONL. When set, the driver skips "
                        "generation entirely: it tokenizes each row's full "
                        "(prompt+answer) sequence, submits it to vLLM as a "
                        "single prefill-only forward (max_tokens=1), and writes "
                        "capture sidecars to the input JSONL's sidecar "
                        "namespace. Requires at least one --capture-* flag. "
                        "See tasks/CALIBRATION_V3_CAPTURE_REPLAY_DESIGN.md.")
    args = p.parse_args()

    # Apply operational env hardening (compile-OOM cap, durable compile cache,
    # host-prereq warnings) before any vllm.* import. Safe: build/cache only.
    # Skip in replay mode: _harden_runtime_env uses setdefault, so calling it
    # here with args.output would pin VLLM_CACHE_ROOT next to the *output*
    # path and the later call in _run_replay (with the replay JSONL path)
    # would be a no-op. _run_replay owns hardening with the correct path.
    if args.replay_from is None:
        _harden_runtime_env(args.output, args.dtype)

    # Pre-import env gates for the imatrix path. These MUST be set before any
    # vllm.* import because vllm.calibration_hooks samples them at module
    # import time (see vllm/calibration_hooks.py for the strict-string rule).
    if args.capture_imatrix:
        os.environ["VLLM_CALIB_CAPTURE_IMATRIX"] = "1"
        # VLLM_CALIB_CAPTURE_EXPERT is REQUIRED so the expert_in callback
        # dispatches; vllm.calibration_imatrix registers a handler against it
        # to scatter-reduce per-expert hidden_states into ffn_gate_exps /
        # ffn_up_exps accumulators.
        os.environ["VLLM_CALIB_CAPTURE_EXPERT"] = "1"
        # VLLM_CALIB_CAPTURE_EXPERT_MID is REQUIRED so the expert_mid hook
        # fires from inside TritonExperts.apply() between SwiGLU and the
        # pre-down quantize; the imatrix callback consumes the per-expert
        # silu(gate)·up activations to populate the real ffn_down_exps
        # entries (replaces the prior uniform-ones placeholder).
        os.environ["VLLM_CALIB_CAPTURE_EXPERT_MID"] = "1"
        log.info("--capture-imatrix: enabled VLLM_CALIB_CAPTURE_IMATRIX=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT_MID=1 "
                 "(must precede vllm import)")

    # Pre-import env gates for the REAP-scores path. Same strict-string
    # rule as imatrix: vllm.calibration_hooks samples these once at
    # module import. Forces the Triton MoE backend because the
    # expert_out_unweighted hook is NOT available on FlashInfer's
    # monolithic path (MoERunner asserts at model load).
    if args.capture_reap_scores:
        os.environ["VLLM_CALIB_CAPTURE_REAP_SCORES"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_ROUTER"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
        log.info("--capture-reap-scores: enabled "
                 "VLLM_CALIB_CAPTURE_REAP_SCORES=1 + "
                 "VLLM_CALIB_CAPTURE_ROUTER=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                 "VLLM_USE_FLASHINFER_MOE_FP16=0 "
                 "(must precede vllm import)")

    # Pre-import env gates for the input-covariance path. Same strict-
    # string rule: vllm.calibration_hooks samples these once at module
    # import. VLLM_CALIB_CAPTURE_EXPERT is REQUIRED so the expert_in
    # callback dispatches; vllm.calibration_input_cov registers a
    # handler against it to scatter-reduce per-expert hidden-state
    # covariance into the dict-shaped accumulator.
    if args.capture_input_covariance:
        os.environ["VLLM_CALIB_CAPTURE_INPUT_COV"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_EXPERT"] = "1"
        # The in-graph Gram accumulation in moe_runner.py is guarded by
        # ``_INPUT_COV_MODE == "resident"``. Without this the accumulation
        # never fires and the sidecar is silently empty (one of the two
        # original empty-capture root causes). The offload path uses the
        # SAME resident accumulation kernel -- it just restricts which layers
        # are allocated -- so it also runs in "resident" mode.
        os.environ["VLLM_CALIB_INPUT_COV_MODE"] = "resident"
        # CAPTURE_EXPERT requires the non-monolithic (Triton) MoE path;
        # MoERunner.__init__ hard-asserts against monolithic backends when
        # CAPTURE_EXPERT is set. _resolve_moe_backend already forces triton,
        # but disable FlashInfer's monolithic fp16 path defensively (mirrors
        # the reap-scores gate) so a stray backend selection can't trip the
        # assert / silently miss the Gram.
        os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
        log.info("--capture-input-covariance: enabled "
                 "VLLM_CALIB_CAPTURE_INPUT_COV=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT=1 + "
                 "VLLM_CALIB_INPUT_COV_MODE=resident + "
                 "VLLM_USE_FLASHINFER_MOE_FP16=0 "
                 "(must precede vllm import)")

    # W-1: Pre-import env gates for the Wanda scalar_row path. Same strict-
    # string rule. VLLM_CALIB_CAPTURE_ROUTER is REQUIRED so the router hook
    # fires (for the topk_weights stash); VLLM_CALIB_CAPTURE_EXPERT is
    # REQUIRED so the expert_in hook fires (for the hidden-state read).
    if args.capture_wanda_scalar_row:
        os.environ["VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_ROUTER"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_EXPERT"] = "1"
        log.info("--capture-wanda-scalar-row: enabled "
                 "VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW=1 + "
                 "VLLM_CALIB_CAPTURE_ROUTER=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT=1 "
                 "(must precede vllm import)")

    # Pre-import env gates for the per-expert-max path. Same strict-string
    # rule: vllm.calibration_hooks samples these once at module import.
    # VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED is REQUIRED so the
    # expert_out_unweighted callback dispatches; the per-expert-max
    # callback shares the hook with REAP-scores via the chained-callback
    # registry. FlashInfer monolithic path is disabled because
    # expert_out_unweighted is not available on that backend.
    if args.capture_per_expert_max:
        os.environ["VLLM_CALIB_CAPTURE_PER_EXPERT_MAX"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
        log.info("--capture-per-expert-max: enabled "
                 "VLLM_CALIB_CAPTURE_PER_EXPERT_MAX=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                 "VLLM_USE_FLASHINFER_MOE_FP16=0 "
                 "(must precede vllm import)")

    # Pre-import env gates for the routing-stats path. Same strict-string
    # rule: vllm.calibration_hooks samples these once at module import.
    # VLLM_CALIB_CAPTURE_ROUTER is REQUIRED so the router callback
    # dispatches. NO FlashInfer / EXPERT_UNWEIGHTED requirement: the
    # router hook fires on every MoE backend, so this writer works
    # alongside any other writer combination (or alone).
    if args.capture_routing_stats:
        os.environ["VLLM_CALIB_CAPTURE_ROUTING_STATS"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_ROUTER"] = "1"
        log.info("--capture-routing-stats: enabled "
                 "VLLM_CALIB_CAPTURE_ROUTING_STATS=1 + "
                 "VLLM_CALIB_CAPTURE_ROUTER=1 "
                 "(must precede vllm import)")

    # Plugin #12 REDO -- Optimization A. Pre-import env gates: the writer
    # needs the router callback (gate logits) AND the
    # expert_out_unweighted callback (gated outputs) to fire. FlashInfer
    # is disabled because expert_out_unweighted is unavailable there.
    if args.capture_stage2_profile:
        os.environ["VLLM_CALIB_CAPTURE_STAGE2_PROFILE"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_ROUTER"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED"] = "1"
        # Cov-capture gates: expert_in feeds gate_proj cov, expert_mid feeds
        # down_proj cov (both mirror the live instrument_experts row set).
        # expert_mid also needs _current_layer_idx, which is set whenever
        # _CAPTURE_EXPERT_UNWEIGHTED or _CAPTURE_EXPERT_MID is on (satisfied).
        os.environ["VLLM_CALIB_CAPTURE_EXPERT"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_EXPERT_MID"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
        log.info("--capture-stage2-profile: enabled "
                 "VLLM_CALIB_CAPTURE_STAGE2_PROFILE=1 + "
                 "VLLM_CALIB_CAPTURE_ROUTER=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT_MID=1 + "
                 "VLLM_USE_FLASHINFER_MOE_FP16=0 "
                 "(must precede vllm import)")

    # CRITICAL-1: opt-in for the new vLLM `layer_in` hook + writer
    # subscription. Requires --capture-stage2-profile (the
    # layer_input_reservoir field lives inside the stage2_profile
    # sidecar). Strict-string env: vllm.calibration_hooks samples this
    # once at module import with ``os.getenv(...) == "1"``.
    if args.capture_layer_input_reservoir:
        if not args.capture_stage2_profile:
            # Post-CRITICAL-1 production flip: the flag defaults ON.
            # Runs that don't capture stage2_profile have nothing to
            # populate, so silently no-op instead of raising
            # SystemExit (which would break every legacy invocation
            # that didn't pass --capture-stage2-profile).
            log.debug(
                "--capture-layer-input-reservoir is on (default) but "
                "--capture-stage2-profile is off; no-op "
                "(layer_input_reservoir lives inside the stage2_profile "
                "sidecar). Pass --no-capture-layer-input-reservoir to "
                "suppress this message."
            )
        else:
            os.environ["VLLM_CALIB_CAPTURE_LAYER_IN"] = "1"
            log.info("--capture-layer-input-reservoir: enabled "
                     "VLLM_CALIB_CAPTURE_LAYER_IN=1 "
                     "(must precede vllm import)")

    # Pre-import env gates for the router-logits-stats path. Same strict-
    # string rule: vllm.calibration_hooks samples these once at module
    # import. VLLM_CALIB_CAPTURE_ROUTER is REQUIRED so the router callback
    # dispatches. NO FlashInfer / EXPERT_UNWEIGHTED requirement.
    if args.capture_router_logits_stats:
        os.environ["VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_ROUTER"] = "1"
        log.info("--capture-router-logits-stats: enabled "
                 "VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS=1 + "
                 "VLLM_CALIB_CAPTURE_ROUTER=1 "
                 "(must precede vllm import)")

    # Pre-import env gates for the output-reservoir path. Same strict-
    # string rule: vllm.calibration_hooks samples these once at module
    # import. VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED is REQUIRED so the
    # expert_out_unweighted callback dispatches; output-reservoir shares
    # the hook with REAP-scores + per-expert-max via the chained-callback
    # registry. FlashInfer monolithic path is disabled because
    # expert_out_unweighted is not available on that backend.
    # VLLM_CALIB_OUTPUT_RESERVOIR_CAP is sampled at writer-module import
    # alongside the gate so it must also be set BEFORE the first vllm
    # import.
    if args.capture_output_reservoir:
        os.environ["VLLM_CALIB_CAPTURE_OUTPUT_RESERVOIR"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED"] = "1"
        os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
        os.environ["VLLM_CALIB_OUTPUT_RESERVOIR_CAP"] = str(
            args.output_reservoir_cap
        )
        log.info("--capture-output-reservoir: enabled "
                 "VLLM_CALIB_CAPTURE_OUTPUT_RESERVOIR=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                 "VLLM_USE_FLASHINFER_MOE_FP16=0 + "
                 "VLLM_CALIB_OUTPUT_RESERVOIR_CAP=%d "
                 "(must precede vllm import)", args.output_reservoir_cap)

    # Pre-import env gates for the block-outputs path. Same strict-
    # string rule: vllm.calibration_hooks samples these once at module
    # import. VLLM_CALIB_CAPTURE_BLOCK is REQUIRED so the block_out hook
    # fires from Qwen3MoeSparseMoeBlock.forward. No FlashInfer
    # restriction: the block_out hook is dispatched from the model-level
    # forward, not from a kernel path, so it works on any MoE backend.
    # The subset size is sampled at writer-module import alongside the
    # gate so it must also be set BEFORE the first vllm import.
    if args.capture_block_outputs:
        os.environ["VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS"] = "1"
        os.environ["VLLM_CALIB_CAPTURE_BLOCK"] = "1"
        os.environ["VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE"] = str(
            args.block_outputs_subset_size
        )
        log.info("--capture-block-outputs: enabled "
                 "VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS=1 + "
                 "VLLM_CALIB_CAPTURE_BLOCK=1 + "
                 "VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE=%d "
                 "(must precede vllm import)",
                 args.block_outputs_subset_size)

    # --- v3 forward-only replay dispatch --------------------------------
    # Placed after all VLLM_CALIB_CAPTURE_* env gates (so the capture hooks
    # are armed before the vllm import inside _run_replay) and after the
    # B0/C1 env invariants, but before the generate-path cache_key block
    # (which _run_replay does not use). _run_replay owns the rest of main().
    if args.replay_from is not None:
        if args.input_cov_offload:
            # The per-window early-exit (set_calibration_max_layer) is
            # incompatible with the other full-forward signals: it would
            # truncate their accumulation at the window's top layer. Enforce
            # input_covariance as the SOLE capture flag for the offload path.
            _other = [
                cap for cap in _CAPTURE_WRITER_MODULES
                if cap != "capture_input_covariance"
                and getattr(args, cap, False)
            ]
            if not args.capture_input_covariance:
                log.error(
                    "--input-cov-offload requires --capture-input-covariance.")
                return 1
            if _other:
                log.error(
                    "--input-cov-offload must be the SOLE capture flag "
                    "(per-window early-exit truncates full-forward signals). "
                    "Also enabled: %s. Run those in a separate replay pass.",
                    _other)
                return 1
            return _run_input_cov_offload(args)
        return _run_replay(args)

    # --- cache_key + paths ----------------------------------------------
    # prev_num_prompts is folded into the prompts_source field so an extended
    # run writes to a separate cache_key (different filename) from a fresh
    # run with the same --num-prompts.
    _prev_suffix = f"#prev{args.prev_num_prompts}" if args.prev_num_prompts else ""
    cache_key = _trace_cache_key_vllm(
        args.teacher, args.teacher_revision,
        f"{args.prompts}#{args.num_prompts}#{args.seed}{_prev_suffix}",
        args.num_prompts, args.seed,
        args.max_new_tokens, args.reasoning_budget,
        args.dtype, args.logits_top_k,
    )
    out_path = Path(args.output)
    if not args.no_cache_suffix:
        out_path = out_path.with_name(
            f"{out_path.stem}_{cache_key}{out_path.suffix}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if args.resume:
            log.info("output already exists: %s — --resume given; nothing to do.",
                     out_path)
            return 0
        log.warning("output exists at %s — refusing to overwrite (no --resume).",
                    out_path)
        return 1
    logits_dir = out_path.with_name(f"{out_path.stem}_logits")
    logits_dir.mkdir(parents=True, exist_ok=True)
    log.info("output -> %s (cache_key=%s)", out_path, cache_key)
    log.info("logits cache -> %s/ (top-k=%d, fp16)", logits_dir, args.logits_top_k)

    # --- prompt gather --------------------------------------------------
    if args.prompts == "qwen3-pretrain-mix":
        prompts_iter = _iter_prompts_from_qwen3_pretrain_mix(
            args.num_prompts, args.seed,
            prev_num_prompts=(args.prev_num_prompts or None),
            shuffle_buffer=(args.shuffle_buffer or None),
        )
    elif args.prompts == "qwen3-pretrain-mix-v2":
        prompts_iter = _iter_prompts_from_qwen3_pretrain_mix_v2(
            args.num_prompts, args.seed,
            prev_num_prompts=(args.prev_num_prompts or None),
            shuffle_buffer=(args.shuffle_buffer or None),
        )
    else:
        if args.prev_num_prompts:
            log.error("--prev-num-prompts only supported with --prompts=qwen3-pretrain-mix{,-v2}")
            return 1
        prompts_iter = _iter_prompts_from_jsonl(Path(args.prompts))

    # 2-tuple (v1 / JSONL) or 4-tuple (v2). The chunk loop below
    # partitions by entry[3] when present.
    prompts: list[tuple] = []
    for entry in prompts_iter:
        prompts.append(tuple(entry))
        if len(prompts) >= args.num_prompts:
            break
    if not prompts:
        log.error("no prompts gathered.")
        return 1
    log.info("gathered %d prompts", len(prompts))

    # --- resume ---------------------------------------------------------
    # Hardening: validate every line as JSON and TRUNCATE the file at the
    # last good offset before counting. This prevents a trailing partial
    # line (from a kill mid-`f.write`) from being silently counted as a
    # "done" row, which would skip that prompt forever AND leave garbage
    # in the eventual finalized .jsonl.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    already_done = 0
    if args.resume and tmp_path.exists():
        last_good_offset = 0
        bad_line_found = False
        with tmp_path.open("rb") as f:
            while True:
                line_start = f.tell()
                raw = f.readline()
                if not raw:
                    break
                # Empty/whitespace lines: don't count, but don't truncate
                # either -- they're harmless.
                stripped = raw.strip()
                if not stripped:
                    last_good_offset = f.tell()
                    continue
                try:
                    json.loads(stripped)
                except json.JSONDecodeError:
                    log.warning(
                        "resume: dropping partial/corrupt line at byte "
                        "offset %d (len=%d) -- file will be truncated",
                        line_start, len(raw),
                    )
                    bad_line_found = True
                    # Don't advance last_good_offset; truncate at line_start.
                    break
                already_done += 1
                last_good_offset = f.tell()
        if bad_line_found:
            with tmp_path.open("r+b") as f:
                f.truncate(last_good_offset)
            log.warning(
                "resume: truncated %s to %d bytes after dropping partial "
                "row(s); %d good rows recovered.",
                tmp_path, last_good_offset, already_done,
            )
        log.info("resume: %d rows already in %s", already_done, tmp_path)
        if already_done >= len(prompts):
            log.info("resume: all prompts already done — finalizing.")
            os.replace(tmp_path, out_path)
            return 0
        prompts = prompts[already_done:]

    # --- load teacher ---------------------------------------------------
    llm = _load_teacher_vllm(
        args.teacher, args.teacher_revision, args.dtype,
        args.gpu_memory_utilization, args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_logprobs=args.logits_top_k,
        moe_backend=_resolve_moe_backend(args),
    )
    tokenizer = llm.get_tokenizer()
    eos_ids = _coerce_eos_ids(getattr(tokenizer, "eos_token_id", None))

    # --- per-writer accumulator setup (extracted to _setup_all_writers) -
    # All ten capture writers are pre-allocated + resume-hydrated here so
    # the generate path and _run_replay() share identical setup logic.
    ws = _setup_all_writers(args, out_path, llm, tokenizer, already_done)


    # --- sampling params ------------------------------------------------
    from vllm import SamplingParams  # type: ignore
    # vLLM's reasoning_budget is exposed via `extra_args` (PR #20859 path).
    # Some vLLM versions accept it as a top-level SamplingParams kwarg; try
    # both with graceful fallback.
    sp_kwargs = dict(
        temperature=0.0,
        top_p=1.0,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
        logprobs=args.logits_top_k,
    )
    try:
        sp = SamplingParams(reasoning_budget=args.reasoning_budget, **sp_kwargs)
        log.info("SamplingParams: reasoning_budget=%d (top-level kwarg)",
                 args.reasoning_budget)
    except TypeError:
        sp = SamplingParams(
            extra_args={"reasoning_budget": args.reasoning_budget},
            **sp_kwargs,
        )
        log.info("SamplingParams: reasoning_budget=%d (extra_args)",
                 args.reasoning_budget)

    # --- generate in chunks --------------------------------------------
    domain_stats: dict[str, list[int]] = {}
    n_new = 0
    mode = "a" if already_done > 0 else "w"
    t0 = time.monotonic()
    # B0 fail-fast state: run the non-empty assertion exactly once, after the
    # first chunk that actually called llm.generate() (an all-TF first chunk
    # legitimately captures nothing, so defer until a real generate chunk).
    first_gen_chunk_checked = False
    _enabled_captures = [
        cap for cap in _CAPTURE_WRITER_MODULES
        if getattr(args, cap, False)
    ]
    with tmp_path.open(mode, encoding="utf-8") as f:
        for chunk_start in range(0, len(prompts), args.chunk_size):
            chunk = prompts[chunk_start:chunk_start + args.chunk_size]
            chunk_attempt_idx = [
                already_done + chunk_start + k for k in range(len(chunk))
            ]
            # Partition by policy. v1 / JSONL paths emit 2-tuples which
            # we treat as GENERATE (entry[3] only exists on the v2
            # 4-tuple shape — fall back to "GENERATE" for shorter tuples).
            def _policy_of(entry):
                return entry[3] if len(entry) >= 4 else "GENERATE"

            gen_chunk = []
            gen_attempt = []
            tf_chunk = []
            tf_attempt = []
            for k, entry in enumerate(chunk):
                if _policy_of(entry) == "TEACHER_FORCED":
                    tf_chunk.append(entry)
                    tf_attempt.append(chunk_attempt_idx[k])
                else:
                    gen_chunk.append(entry)
                    gen_attempt.append(chunk_attempt_idx[k])

            # Emit TF rows FIRST so a SIGINT during the (slower)
            # generate() call doesn't lose the cheap canonical rows.
            #
            # DEPRECATION (v3): TEACHER_FORCED synthesis writes JSONL rows but
            # runs NO forward pass, so it captures nothing for code/science
            # subsets. The v3 forward-only replay path (--replay-from) replays
            # an existing corpus through a real prefill so every subset
            # contributes activations. Prefer --replay-from over the TF path
            # for capture coverage. See
            # tasks/CALIBRATION_V3_CAPTURE_REPLAY_DESIGN.md.
            if tf_chunk:
                log.info(
                    "chunk %d-%d: synthesizing %d TEACHER_FORCED rows "
                    "(skipping vLLM generate)",
                    chunk_start, chunk_start + len(chunk), len(tf_chunk),
                )
                for row in _synth_teacher_forced_rows(
                    tf_chunk, tf_attempt, tokenizer, domain_stats,
                ):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n_new += 1

            if gen_chunk:
                chunk_prompts_text = [entry[0] for entry in gen_chunk]
                rendered = _render_prompts(tokenizer, chunk_prompts_text)
                log.info("chunk %d-%d: submitting %d prompts to vLLM",
                         chunk_start, chunk_start + len(chunk), len(gen_chunk))
                chunk_t0 = time.monotonic()
                outputs = llm.generate(rendered, sp)
                chunk_elapsed = time.monotonic() - chunk_t0
                log.info("chunk done in %.1fs (%.2f s/prompt avg)",
                         chunk_elapsed, chunk_elapsed / max(len(gen_chunk), 1))

                for row in _process_outputs(
                    outputs, gen_chunk, gen_attempt, eos_ids,
                    args.logits_top_k, logits_dir, domain_stats,
                    args.max_new_tokens,
                ):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n_new += 1
            elif tf_chunk:
                # All-TF chunk — no generate() call; the accumulator-
                # checkpoint blocks below run unchanged and observe the
                # current accumulator state (no new GENERATE forward
                # pass means no new sidecar samples from this chunk; the
                # checkpoint just persists whatever was captured before).
                log.info(
                    "chunk %d-%d: all-TF chunk; no vLLM generate call "
                    "this iteration",
                    chunk_start, chunk_start + len(chunk),
                )

            # F-H-5: per-row f.flush() pushed Python's userspace buffer to
            # the kernel page-cache but did NOT durably flush to disk. A
            # kernel-panic / power-loss between flush and the next
            # pdflush cycle (5-30 s on ext4) would lose all rows from
            # this chunk — but the per-chunk accumulator checkpoints
            # below would also be lost, so the JSONL/ckpt counter pair
            # remains internally consistent (the F-H-6 hard-fail covers
            # the remaining edge case). The fsync here promotes the
            # entire chunk's rows to durable storage BEFORE the
            # accumulator checkpoints are written, establishing the
            # ordering invariant "JSONL is durable >= ckpt counter".
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError as exc:
                # LOW-3: narrow the swallow to errno {EINVAL, ENOTSUP}.
                # FUSE / tmpfs reject fsync on regular files with these
                # errnos; any OTHER OSError (EIO, ENOSPC, EBADF) is a
                # real problem the JSONL caller must see immediately
                # rather than waiting for the next pdflush cycle to
                # surface the loss.
                if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                    raise
                # Logging at DEBUG so production runs on real ext4/xfs
                # don't see noise.
                log.debug(
                    "F-H-5: fsync(jsonl_fd) raised OSError (errno=%s) — "
                    "non-POSIX filesystem (HF Jobs FUSE mount?); relying "
                    "on rename atomicity instead.",
                    exc.errno,
                )

            total_done = already_done + n_new
            total_target = already_done + len(prompts)
            session_elapsed = time.monotonic() - t0
            log.info(
                "[%d/%d traces yielded] — %.0fs session elapsed "
                "(%.1f s/trace session-avg)",
                total_done, total_target,
                session_elapsed, session_elapsed / max(n_new, 1),
            )

            # B0 fail-fast: after the FIRST chunk that actually ran
            # llm.generate(), assert every enabled capture has nonzero
            # entries. Runs once (first_gen_chunk_checked guard) and only on a
            # real generate chunk (an all-TF first chunk captures nothing
            # legitimately). Placed BEFORE close_subset so we abort with
            # partial JSONL intact and no checkpoint written.
            if gen_chunk and not first_gen_chunk_checked:
                first_gen_chunk_checked = True
                if _enabled_captures:
                    try:
                        _model_cls = type(
                            llm.llm_engine.model_executor.driver_worker
                            .model_runner.model
                        ).__name__
                    except Exception:
                        _model_cls = "<unresolved>"
                    assert_enabled_captures_nonempty(
                        _enabled_captures,
                        model_class=_model_cls,
                        allow_empty=args.allow_empty_captures,
                    )

            # Block-outputs subset gate. Close as soon as the cumulative
            # prompt counter reaches the configured subset size so any
            # later chunks no-op the block_out dispatch (saves the
            # post-subset clone + CPU bf16 copy cost for the remaining
            # prompts of the calibration run). Idempotent: calling
            # close_subset() twice is a no-op.
            if (args.capture_block_outputs
                    and total_done >= args.block_outputs_subset_size):
                try:
                    import vllm.calibration_block_outputs as _bo  # type: ignore
                    if not _bo._SUBSET_CLOSED:
                        _bo.close_subset()
                        log.info(
                            "block-outputs: subset closed at %d prompts "
                            "(>= subset_size=%d); subsequent block_out "
                            "dispatches are no-ops.",
                            total_done, args.block_outputs_subset_size,
                        )
                except Exception as exc:
                    log.error("block-outputs close_subset failed: %s",
                              exc, exc_info=True)

            # Periodic imatrix checkpoint. Same cadence as the JSONL flush
            # so a preemption between the two never leaves the checkpoint
            # ahead of the JSONL. Atomic via tmp+rename inside the dumper,
            # so a kill during the dump leaves any previous .imatrix.ckpt
            # intact.
            if (args.capture_imatrix
                    and args.imatrix_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.imatrix_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_imatrix as _im  # type: ignore
                        # Driver-owned cumulative counter: reflects all
                        # prompts ever folded in (across instance lifetimes).
                        _im.set_n_prompts_accumulated(already_done + n_new)
                        _im.dump_imatrix_checkpoint(str(ws.imatrix_ckpt_path))
                        log.info(
                            "imatrix: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.imatrix_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "imatrix checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Periodic REAP-scores checkpoint -- mirrors imatrix cadence.
            if (args.capture_reap_scores
                    and args.reap_scores_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.reap_scores_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_reap_scores as _reap  # type: ignore
                        _reap.set_n_prompts_accumulated(already_done + n_new)
                        _reap.dump_reap_scores_checkpoint(str(ws.reap_ckpt_path))
                        log.info(
                            "reap-scores: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.reap_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "reap-scores checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Periodic input-covariance checkpoint -- mirrors imatrix /
            # reap-scores cadence.
            if (args.capture_input_covariance
                    and args.input_cov_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.input_cov_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_input_cov as _icov  # type: ignore
                        _icov.set_n_prompts_accumulated(already_done + n_new)
                        _icov.dump_input_cov_checkpoint(
                            str(ws.input_cov_ckpt_path),
                        )
                        log.info(
                            "input-cov: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.input_cov_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "input-cov checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Periodic Wanda scalar_row checkpoint -- mirrors input-cov
            # cadence. W-1 (audit/PLAN_W1) §6.3.
            if (args.capture_wanda_scalar_row
                    and args.wanda_scalar_row_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.wanda_scalar_row_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
                        _wsr.set_n_prompts_accumulated(already_done + n_new)
                        _wsr.dump_wanda_scalar_row_checkpoint(
                            str(ws.wsr_ckpt_path),
                        )
                        log.info(
                            "wanda-scalar-row: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.wsr_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "wanda-scalar-row checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Plugin #12 REDO -- periodic stage2-profile checkpoint.
            if (args.capture_stage2_profile
                    and args.stage2_profile_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.stage2_profile_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_stage2_profile as _s2p  # type: ignore
                        _s2p.set_n_prompts_accumulated(already_done + n_new)
                        _s2p.dump_stage2_profile_checkpoint(str(ws.s2p_ckpt_path))
                        log.info(
                            "stage2-profile: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.s2p_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "stage2-profile checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Periodic per-expert-max checkpoint -- same cadence pattern.
            if (args.capture_per_expert_max
                    and args.per_expert_max_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.per_expert_max_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_per_expert_max as _pem  # type: ignore
                        _pem.set_n_prompts_accumulated(already_done + n_new)
                        _pem.dump_per_expert_max_checkpoint(
                            str(ws.pem_ckpt_path),
                        )
                        log.info(
                            "per-expert-max: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.pem_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "per-expert-max checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Periodic routing-stats checkpoint -- same cadence pattern.
            if (args.capture_routing_stats
                    and args.routing_stats_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.routing_stats_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_routing_stats as _rts  # type: ignore
                        _rts.set_n_prompts_accumulated(already_done + n_new)
                        _rts.dump_routing_stats_checkpoint(
                            str(ws.rts_ckpt_path),
                        )
                        log.info(
                            "routing-stats: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.rts_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "routing-stats checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Periodic router-logits-stats checkpoint -- same cadence.
            if (args.capture_router_logits_stats
                    and args.router_logits_stats_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.router_logits_stats_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
                        _rlsx.set_n_prompts_accumulated(already_done + n_new)
                        _rlsx.dump_router_logits_stats_checkpoint(
                            str(ws.router_logits_ckpt_path),
                        )
                        log.info(
                            "router-logits-stats: checkpointed %d prompts "
                            "-> %s",
                            already_done + n_new, ws.router_logits_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "router-logits-stats checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Periodic output-reservoir checkpoint -- same cadence pattern.
            if (args.capture_output_reservoir
                    and args.output_reservoir_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.output_reservoir_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_output_reservoir as _or  # type: ignore
                        _or.set_n_prompts_accumulated(already_done + n_new)
                        _or.dump_output_reservoir_checkpoint(
                            str(ws.or_ckpt_path),
                        )
                        log.info(
                            "output-reservoir: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.or_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "output-reservoir checkpoint failed: %s",
                            exc, exc_info=True,
                        )

            # Periodic block-outputs checkpoint -- same cadence pattern.
            # Capture only fires until the driver calls close_subset, but
            # the checkpoint still serializes the closed-flag so a resumed
            # run that already closed the subset stays a no-op.
            if (args.capture_block_outputs
                    and args.block_outputs_checkpoint_every_chunks > 0):
                chunk_idx = chunk_start // args.chunk_size
                every = args.block_outputs_checkpoint_every_chunks
                if (chunk_idx + 1) % every == 0:
                    try:
                        import vllm.calibration_block_outputs as _bo  # type: ignore
                        _bo.set_n_prompts_accumulated(already_done + n_new)
                        _bo.dump_block_outputs_checkpoint(
                            str(ws.bo_ckpt_path),
                        )
                        log.info(
                            "block-outputs: checkpointed %d prompts -> %s",
                            already_done + n_new, ws.bo_ckpt_path,
                        )
                    except Exception as exc:
                        log.error(
                            "block-outputs checkpoint failed: %s",
                            exc, exc_info=True,
                        )

    # --- imatrix dump ---------------------------------------------------
    # Run BEFORE the JSONL finalize so a failure here can't corrupt the
    # rename, but in a try/except so a failure doesn't lose the JSONL.
    if args.capture_imatrix:
        imatrix_path = out_path.with_suffix(".imatrix.dat")
        try:
            import vllm.calibration_imatrix as _im  # type: ignore
            # Final cumulative-counter sync; this is also what gets written
            # into the m_last_chunk field of the .imatrix.dat header.
            _im.set_n_prompts_accumulated(already_done + n_new)
            total_prompts_processed = _im.get_n_prompts_accumulated()
            _im.dump_imatrix(str(imatrix_path),
                             chunk_count=total_prompts_processed)
            log.info("imatrix -> %s (%d entries from %d prompts)",
                     imatrix_path, len(_im._accumulators),
                     total_prompts_processed)
            # Periodic checkpoint served its purpose; remove it so the
            # next clean run (without --resume) doesn't hydrate stale state.
            if ws.imatrix_ckpt_path is not None and ws.imatrix_ckpt_path.exists():
                ws.imatrix_ckpt_path.unlink()
        except Exception as exc:
            # Imatrix is a sidecar; cal JSONL is the primary deliverable.
            # Log and continue.
            log.error("imatrix dump failed: %s", exc, exc_info=True)

    # --- REAP-scores dump ----------------------------------------------
    # Run BEFORE the JSONL finalize so a failure here can't corrupt the
    # rename, but in a try/except so a failure doesn't lose the JSONL.
    if args.capture_reap_scores:
        try:
            import vllm.calibration_reap_scores as _reap  # type: ignore
            # Final cumulative-counter sync.
            _reap.set_n_prompts_accumulated(already_done + n_new)
            _reap.dump_reap_scores(out_path)
            log.info(
                "reap-scores: dumped sidecar from %d prompts (next to %s)",
                _reap.get_n_prompts_accumulated(), out_path,
            )
            # Periodic checkpoint served its purpose; remove it so the
            # next clean run (without --resume) doesn't hydrate stale state.
            if ws.reap_ckpt_path is not None and ws.reap_ckpt_path.exists():
                ws.reap_ckpt_path.unlink()
        except Exception as exc:
            log.error("reap-scores dump failed: %s", exc, exc_info=True)

    # --- input-covariance dump -----------------------------------------
    # Same try/except policy as imatrix / reap-scores: the JSONL is the
    # primary deliverable.
    if args.capture_input_covariance:
        try:
            import vllm.calibration_input_cov as _icov  # type: ignore
            _icov.set_n_prompts_accumulated(already_done + n_new)
            _icov.dump_input_cov(out_path)
            log.info(
                "input-cov: dumped sidecar from %d prompts (next to %s)",
                _icov.get_n_prompts_accumulated(), out_path,
            )
            if ws.input_cov_ckpt_path is not None and ws.input_cov_ckpt_path.exists():
                ws.input_cov_ckpt_path.unlink()
        except Exception as exc:
            log.error("input-cov dump failed: %s", exc, exc_info=True)

    # --- Wanda scalar_row dump (W-1) -----------------------------------
    # Same try/except policy: the JSONL is the primary deliverable;
    # dump failures are logged but never re-raised.
    if args.capture_wanda_scalar_row:
        try:
            import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
            _wsr.set_n_prompts_accumulated(already_done + n_new)
            _wsr.dump_wanda_scalar_row(out_path)
            log.info(
                "wanda-scalar-row: dumped sidecar from %d prompts (next to %s)",
                _wsr.get_n_prompts_accumulated(), out_path,
            )
            if ws.wsr_ckpt_path is not None and ws.wsr_ckpt_path.exists():
                ws.wsr_ckpt_path.unlink()
        except Exception as exc:
            log.error("wanda-scalar-row dump failed: %s", exc, exc_info=True)

    # --- Plugin #12 REDO -- stage2-profile dump ------------------------
    # Same try/except policy as the other writers: the JSONL is the
    # primary deliverable; dump failures are logged but never re-raised.
    if args.capture_stage2_profile:
        try:
            import vllm.calibration_stage2_profile as _s2p  # type: ignore
            _s2p.set_n_prompts_accumulated(already_done + n_new)
            _s2p.dump_stage2_profile(out_path)
            log.info(
                "stage2-profile: dumped sidecar from %d prompts (next to %s)",
                _s2p.get_n_prompts_accumulated(), out_path,
            )
            if ws.s2p_ckpt_path is not None and ws.s2p_ckpt_path.exists():
                ws.s2p_ckpt_path.unlink()
        except Exception as exc:
            log.error("stage2-profile dump failed: %s", exc, exc_info=True)

    # --- per-expert-max dump -------------------------------------------
    # Same try/except policy as imatrix / reap-scores / input-cov: the
    # JSONL is the primary deliverable.
    if args.capture_per_expert_max:
        try:
            import vllm.calibration_per_expert_max as _pem  # type: ignore
            _pem.set_n_prompts_accumulated(already_done + n_new)
            _pem.dump_per_expert_max(out_path)
            log.info(
                "per-expert-max: dumped sidecar from %d prompts (next to %s)",
                _pem.get_n_prompts_accumulated(), out_path,
            )
            if ws.pem_ckpt_path is not None and ws.pem_ckpt_path.exists():
                ws.pem_ckpt_path.unlink()
        except Exception as exc:
            log.error("per-expert-max dump failed: %s", exc, exc_info=True)

    # --- routing-stats dump --------------------------------------------
    # Same try/except policy: the JSONL is the primary deliverable.
    if args.capture_routing_stats:
        try:
            import vllm.calibration_routing_stats as _rts  # type: ignore
            _rts.set_n_prompts_accumulated(already_done + n_new)
            _rts.dump_routing_stats(out_path)
            log.info(
                "routing-stats: dumped sidecar from %d prompts (next to %s)",
                _rts.get_n_prompts_accumulated(), out_path,
            )
            if ws.rts_ckpt_path is not None and ws.rts_ckpt_path.exists():
                ws.rts_ckpt_path.unlink()
        except Exception as exc:
            log.error("routing-stats dump failed: %s", exc, exc_info=True)

    # --- router-logits-stats dump --------------------------------------
    # Same try/except policy: the JSONL is the primary deliverable.
    if args.capture_router_logits_stats:
        try:
            import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
            _rlsx.set_n_prompts_accumulated(already_done + n_new)
            _rlsx.dump_router_logits_stats(out_path)
            log.info(
                "router-logits-stats: dumped sidecar from %d prompts "
                "(next to %s)",
                _rlsx.get_n_prompts_accumulated(), out_path,
            )
            if (ws.router_logits_ckpt_path is not None
                    and ws.router_logits_ckpt_path.exists()):
                ws.router_logits_ckpt_path.unlink()
        except Exception as exc:
            log.error(
                "router-logits-stats dump failed: %s", exc, exc_info=True,
            )

    # --- output-reservoir dump -----------------------------------------
    # Same try/except policy: the JSONL is the primary deliverable.
    if args.capture_output_reservoir:
        try:
            import vllm.calibration_output_reservoir as _or  # type: ignore
            _or.set_n_prompts_accumulated(already_done + n_new)
            _or.dump_output_reservoir(out_path)
            log.info(
                "output-reservoir: dumped sidecar from %d prompts "
                "(next to %s)",
                _or.get_n_prompts_accumulated(), out_path,
            )
            if ws.or_ckpt_path is not None and ws.or_ckpt_path.exists():
                ws.or_ckpt_path.unlink()
        except Exception as exc:
            log.error(
                "output-reservoir dump failed: %s", exc, exc_info=True,
            )

    # --- block-outputs dump --------------------------------------------
    # Same try/except policy: the JSONL is the primary deliverable.
    # close_subset() is called belt-and-braces here even though the
    # in-loop gate above should already have fired; the subset MUST be
    # closed before the dump so the n_prompts_in_subset field on the
    # sidecar payload reflects the actual frozen subset count, not the
    # (potentially larger) total accumulated.
    if args.capture_block_outputs:
        try:
            import vllm.calibration_block_outputs as _bo  # type: ignore
            _bo.set_n_prompts_accumulated(already_done + n_new)
            if not _bo._SUBSET_CLOSED:
                _bo.close_subset()
                log.info(
                    "block-outputs: subset closed pre-dump (run ended "
                    "with %d prompts < subset_size=%d -- shipping the "
                    "partial subset).",
                    already_done + n_new, args.block_outputs_subset_size,
                )
            _bo.dump_block_outputs(out_path)
            log.info(
                "block-outputs: dumped per-layer sidecars from %d "
                "prompts (next to %s)",
                _bo.get_n_prompts_accumulated(), out_path,
            )
            if ws.bo_ckpt_path is not None and ws.bo_ckpt_path.exists():
                ws.bo_ckpt_path.unlink()
        except Exception as exc:
            log.error(
                "block-outputs dump failed: %s", exc, exc_info=True,
            )

    # --- finalize -------------------------------------------------------
    os.replace(tmp_path, out_path)
    log.info("wrote %d traces (%d resumed + %d new) -> %s",
             already_done + n_new, already_done, n_new, out_path)

    # Per-domain completeness summary.
    if domain_stats:
        agg_c = sum(c for c, _ in domain_stats.values())
        agg_t = sum(t for _, t in domain_stats.values())
        log.info("completeness: %d/%d (%.1f%%) complete across domains",
                 agg_c, agg_t, 100.0 * agg_c / max(agg_t, 1))
        for d in sorted(domain_stats):
            c, t = domain_stats[d]
            log.info("  %-14s  %4d/%-4d  (%5.1f%%)", d, c, t,
                     100.0 * c / max(t, 1))
    return 0


def _run_replay(args) -> int:
    """v3 forward-only replay mode.

    Called from main() when --replay-from is set. Capture env gates
    (VLLM_CALIB_CAPTURE_*) and B0/C1 invariants are already applied
    by main() before this function is called.

    Reads an existing self-traces JSONL, tokenizes each row's full
    (prompt+answer) sequence, submits to vLLM as a single prefill-only
    forward (max_tokens=1, max_num_batched_tokens=256), and writes
    capture sidecars to the canonical sidecar_path namespace of the
    input JSONL.

    Imatrix dump is special-cased: dump_imatrix(str, chunk_count=int)
    writes <jsonl>.imatrix.dat alongside the JSONL (NOT under sidecars/).
    All other nine writers use dump_<signal>(Path) and compute their
    sidecar path internally.

    Counter contract (M1 fix):
      n_skipped is incremented ONLY in the per-row rendering loop.
      No other site may increment it. rows_done = rows_done_base +
      n_replayed + n_skipped is always the correct JSONL position.
      captured_done = captured_done_base + n_replayed excludes skips
      and is passed to _setup_all_writers for writer counter checks.

    Returns 0 on success, 1 on configuration/validation error.
    SystemExit(2) if B0 hard-fails (assert_enabled_captures_nonempty).
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    replay_jsonl = Path(args.replay_from).resolve()
    if not replay_jsonl.is_file():
        log.error("--replay-from: file not found: %s", replay_jsonl)
        return 1

    _enabled_captures = [
        cap for cap in _CAPTURE_WRITER_MODULES if getattr(args, cap, False)
    ]
    if not _enabled_captures:
        log.error(
            "--replay-from requires at least one --capture-* flag. "
            "Nothing to capture; exiting."
        )
        return 1

    log.info(
        "v3 replay mode: input=%s, enabled captures=%s",
        replay_jsonl, _enabled_captures,
    )

    # Runtime env hardening (compile cache, JIT cap).
    _harden_runtime_env(str(replay_jsonl), args.dtype)

    # ------------------------------------------------------------------
    # 2. Load + validate input JSONL
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    with replay_jsonl.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                log.error(
                    "replay: invalid JSON at line %d: %s", lineno, exc)
                return 1
            msgs = row.get("messages", [])
            if len(msgs) < 2:
                log.error(
                    "replay: line %d: messages field missing or has <2 "
                    "entries. Expected [{role:user,...},{role:assistant,...}]. "
                    "Is this a v8+ schema JSONL?", lineno)
                return 1
            all_rows.append(row)

    if not all_rows:
        log.error("replay: no rows found in %s", replay_jsonl)
        return 1

    log.info("replay: loaded %d rows from %s", len(all_rows), replay_jsonl)

    n_generate_rows = sum(
        1 for r in all_rows
        if r.get("completion_source") == "teacher_generated"
    )
    log.info(
        "replay: %d GENERATE rows, %d TEACHER_FORCED rows",
        n_generate_rows, len(all_rows) - n_generate_rows,
    )
    if n_generate_rows == 0:
        log.warning(
            "replay: no GENERATE rows (completion_source=teacher_generated). "
            "Continuing -- may indicate an unexpected corpus."
        )

    # ------------------------------------------------------------------
    # 3. Pre-flight length scan (uses stored metadata; no tokenization)
    # ------------------------------------------------------------------
    max_required = 0
    over_count_preflight = 0
    for row in all_rows:
        needed = row.get("n_prompt_tokens", 0) + row.get("n_gen_tokens", 0)
        max_required = max(max_required, needed)
        if needed > args.max_model_len:
            over_count_preflight += 1
    log.info(
        "replay pre-flight: corpus max tokens (n_prompt+n_gen)=%d, "
        "--max-model-len=%d, ~%d/%d rows may be skipped "
        "(actual tokenization may differ from stored counts)",
        max_required, args.max_model_len,
        over_count_preflight, len(all_rows),
    )
    if over_count_preflight > 0:
        log.warning(
            "replay: ~%d rows may exceed --max-model-len=%d. "
            "These will be SKIPPED (not truncated). "
            "Consider --max-model-len %d (~5%% over corpus max).",
            over_count_preflight, args.max_model_len,
            int(max_required * 1.05),
        )

    # ------------------------------------------------------------------
    # 4. Resume handling -- two counters (M1 fix)
    #
    # rows_done    = total rows consumed (replayed + skipped).
    #               Used to slice the JSONL on resume.
    # captured_done = rows that contributed to captures (no skips).
    #               Passed to _setup_all_writers as already_done.
    # ------------------------------------------------------------------
    replay_ckpt = replay_jsonl.with_suffix(
        replay_jsonl.suffix + ".replay.ckpt"
    )
    rows_done_base = 0
    captured_done_base = 0
    if args.resume and replay_ckpt.exists():
        try:
            _rc = json.loads(
                replay_ckpt.read_text(encoding="utf-8"))
            rows_done_base = int(_rc["rows_done"])
            captured_done_base = int(_rc["captured_done"])
            log.info(
                "replay resume: rows_done=%d, captured_done=%d (per %s)",
                rows_done_base, captured_done_base, replay_ckpt,
            )
        except Exception as exc:
            log.warning(
                "replay: could not read resume checkpoint (%s); "
                "starting from row 0.", exc)
            rows_done_base = 0
            captured_done_base = 0

    replay_rows = all_rows[rows_done_base:]
    if not replay_rows:
        log.info("replay: all %d rows already processed.", len(all_rows))
        return 0

    # ------------------------------------------------------------------
    # 5. Load teacher
    #
    # THROUGHPUT: the patched wheel sizes the in-graph capture side-store to
    # _calib_buf_rows = max(max_cudagraph_capture_size, max_num_batched_tokens,
    # 512), so a LARGE max_num_batched_tokens scales the buffer WITH the batch
    # (no OOB capture skip) and keeps the GPU busy. The old hard 256 cap (only
    # needed before the buf_rows fix) is removed; default high and let
    # --max-num-batched-tokens / --max-num-seqs override. The C2 assert
    # (buf_rows >= max_num_batched_tokens) below is the correctness safety net.
    # ------------------------------------------------------------------
    _REPLAY_MAX_BATCHED_TOKENS = args.max_num_batched_tokens or 8192
    _REPLAY_MAX_NUM_SEQS = args.max_num_seqs or 256

    llm = _load_teacher_vllm(
        args.teacher,
        args.teacher_revision,
        args.dtype,
        args.gpu_memory_utilization,
        args.max_model_len,
        max_num_seqs=_REPLAY_MAX_NUM_SEQS,
        max_num_batched_tokens=_REPLAY_MAX_BATCHED_TOKENS,
        max_logprobs=1,
        moe_backend=_resolve_moe_backend(args),
    )
    tokenizer = llm.get_tokenizer()

    # ------------------------------------------------------------------
    # 6. C2 runtime buffer-size assertion (L-NEW-1 hardening)
    #
    # Read back the actual max_cudagraph_capture_size from the live
    # engine. If buf_rows < _REPLAY_MAX_BATCHED_TOKENS the
    # expert_out_unweighted hook would silently skip prefill chunks and
    # reap/per_expert_max/output_reservoir would only capture the single
    # decode token -- the whole chunked-prefill fix would silently fail.
    #
    # Two attribute paths tried before falling back to the formula:
    #   Path 1: llm.llm_engine.vllm_config.compilation_config...
    #   Path 2: llm.llm_engine.model_executor.driver_worker
    #             .model_runner.vllm_config.compilation_config...
    #           (same spine used by the generate path for model_cls,
    #            lines 2277-2279 of the driver)
    # Only if BOTH raise AttributeError: fall back to formula and log
    # ERROR (not WARNING) -- the GPU smoke is then the sole guarantee.
    # ------------------------------------------------------------------
    # The ACTUAL side-store row capacity is TritonExperts._calib_buf_rows =
    # max(max_cudagraph_capture_size, max_num_batched_tokens, 512) in the
    # patched wheel (H2 fix). Introspect a live TritonExperts instance and
    # assert that real value >= max_num_batched_tokens, so a forward batch can
    # never exceed the buffer (the OOB-skip condition). At mbt > cudagraph
    # capture size the large prefill simply runs eager (Python scatter executes)
    # -> capture still fires; the only thing that matters is buf_rows >= batch.
    buf_rows = -1
    _buf_rows_source = "unresolved"
    try:
        _model = (llm.llm_engine.model_executor.driver_worker
                  .model_runner.model)
        for _name, _mod in _model.named_modules():
            if getattr(_mod, "global_num_experts", None) is not None and \
                    hasattr(_mod, "quant_method"):
                _te = getattr(getattr(_mod.quant_method, "moe_kernel", None),
                              "fused_experts", None)
                _br = getattr(_te, "_calib_buf_rows", None)
                if isinstance(_br, int) and _br > 0:
                    buf_rows = _br
                    _buf_rows_source = "TritonExperts._calib_buf_rows (actual)"
                break
    except Exception as _exc:  # noqa: BLE001
        log.warning("C2: could not introspect _calib_buf_rows (%s)", _exc)
    if buf_rows < 0:
        # Fall back to the wheel's formula so the check is not falsely strict.
        try:
            _ccs = (llm.llm_engine.vllm_config
                    .compilation_config.max_cudagraph_capture_size) or 0
        except AttributeError:
            _ccs = 0
        buf_rows = max(_ccs, _REPLAY_MAX_BATCHED_TOKENS, 512)
        _buf_rows_source = "formula max(cg,mbt,512)"
        log.warning(
            "C2: using formula buf_rows=%d (could not read the live "
            "TritonExperts._calib_buf_rows). If the installed wheel lacks the "
            "H2 buf_rows fix this may overstate the real buffer; the smoke's "
            "captured-token check is the backstop.", buf_rows,
        )

    log.info(
        "C2 check: _calib_buf_rows=%d (source: %s); "
        "must be >= max_num_batched_tokens=%d",
        buf_rows, _buf_rows_source, _REPLAY_MAX_BATCHED_TOKENS,
    )

    if buf_rows < _REPLAY_MAX_BATCHED_TOKENS:
        log.error(
            "C2 HARD FAIL: _calib_buf_rows=%d < max_num_batched_tokens=%d. "
            "expert_out_unweighted would skip forward batches > buf_rows. "
            "The wheel's H2 fix should make buf_rows=max(cg,mbt,512) >= mbt; "
            "if not, rebuild the wheel or lower --max-num-batched-tokens to %d. "
            "Aborting.",
            buf_rows, _REPLAY_MAX_BATCHED_TOKENS, buf_rows,
        )
        return 1

    log.info(
        "C2 check passed: _calib_buf_rows=%d >= max_num_batched_tokens=%d. "
        "expert_out_unweighted fires on all forward tokens.",
        buf_rows, _REPLAY_MAX_BATCHED_TOKENS,
    )

    # ------------------------------------------------------------------
    # 7. Writer setup
    #
    # out_path = replay_jsonl so .ckpt files land next to the input JSONL.
    # captured_done_base passed as already_done (excludes skips -- M1).
    # ------------------------------------------------------------------
    out_path = replay_jsonl
    ws = _setup_all_writers(
        args, out_path, llm, tokenizer, captured_done_base,
    )

    # ------------------------------------------------------------------
    # 8. SamplingParams for replay
    # ------------------------------------------------------------------
    from vllm import SamplingParams  # type: ignore
    sp_replay = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=args.seed,
    )
    log.info(
        "replay SamplingParams: temperature=0.0, max_tokens=1, seed=%d "
        "(1 decode token emitted per row; outputs discarded).",
        args.seed,
    )

    # ------------------------------------------------------------------
    # 9. Replay loop
    #
    # Counter invariant (C-NEW-1 fix):
    #   n_skipped is incremented ONLY by the per-row `n_skipped += 1`
    #   inside the rendering loop. No other site touches it.
    #   rows_done = rows_done_base + n_replayed + n_skipped is always
    #   the correct total rows consumed from the JSONL.
    # ------------------------------------------------------------------
    n_replayed = 0          # rows submitted to vLLM (contribute to captures)
    n_skipped = 0           # rows dropped (over max_model_len)
    skipped_by_subset: dict[str, int] = {}
    replayed_rows: list[dict] = []
    replayed_token_counts: list[int] = []
    first_chunk_checked = False   # M2: gate B0 on first actual submission

    t0 = time.monotonic()

    for chunk_start in range(0, len(replay_rows), args.chunk_size):
        chunk = replay_rows[chunk_start: chunk_start + args.chunk_size]

        # Render all rows in this chunk. Per-row n_skipped += 1 is the
        # SOLE site that increments n_skipped (C-NEW-1).
        rendered_chunk: list[tuple[dict, list[int], int]] = []
        for row in chunk:
            result = _render_row_for_replay(
                row, tokenizer, args.max_model_len)
            if result is None:
                n_skipped += 1   # SOLE increment site (C-NEW-1)
                subset = str(
                    row.get("subset") or row.get("domain") or "unknown")
                skipped_by_subset[subset] = (
                    skipped_by_subset.get(subset, 0) + 1)
            else:
                tok_ids, n_tok = result
                rendered_chunk.append((row, tok_ids, n_tok))

        # After the rendering loop, n_skipped already reflects ALL skips
        # from this chunk (C-NEW-1). rows_done_base + n_replayed + n_skipped
        # is the correct JSONL position at this point.

        if not rendered_chunk:
            # All rows in this chunk were over max_model_len.
            log.warning(
                "replay chunk %d-%d: all %d rows over max_model_len=%d; "
                "skipping chunk.",
                chunk_start, chunk_start + len(chunk),
                len(chunk), args.max_model_len,
            )
            # Write checkpoint using the already-correct counters.
            # n_skipped already includes this chunk's skips from the
            # per-row loop above. DO NOT add len(chunk) again (C-NEW-1).
            _write_replay_ckpt(
                replay_ckpt,
                rows_done=rows_done_base + n_replayed + n_skipped,
                captured_done=captured_done_base + n_replayed,
            )
            continue

        # Build requests. Primary shape: prompt_token_ids dict.
        # Fallback (N1): if the pinned wheel rejects dicts, re-render
        # as string and verify round-trip token equality before submitting.
        requests = [
            {"prompt_token_ids": tok_ids}
            for _, tok_ids, _ in rendered_chunk
        ]

        log.info(
            "replay chunk %d-%d: submitting %d requests "
            "(%d over-length skipped); token range [%d, %d]",
            rows_done_base + chunk_start,
            rows_done_base + chunk_start + len(chunk),
            len(rendered_chunk),
            len(chunk) - len(rendered_chunk),
            min(n for _, _, n in rendered_chunk),
            max(n for _, _, n in rendered_chunk),
        )
        chunk_t0 = time.monotonic()
        try:
            outputs = llm.generate(requests, sp_replay)
        except TypeError as exc:
            # N1 fallback: dict input not accepted; re-render as string
            # and verify round-trip token equality (no lossy decode).
            log.warning(
                "replay: LLM.generate rejected prompt_token_ids dict "
                "(%s); falling back to string rendering. "
                "Verifying round-trip token equality.", exc,
            )
            string_inputs = []
            for row, expected_ids, _ in rendered_chunk:
                messages = row.get("messages", [])
                try:
                    rendered_str = tokenizer.apply_chat_template(
                        messages, tokenize=False,
                        add_generation_prompt=False, enable_thinking=True,
                    )
                except TypeError:
                    rendered_str = tokenizer.apply_chat_template(
                        messages, tokenize=False,
                        add_generation_prompt=False,
                    )
                recheck_ids = tokenizer(
                    rendered_str, add_special_tokens=False,
                )["input_ids"]
                if recheck_ids != expected_ids:
                    log.error(
                        "replay N1: round-trip token mismatch for row "
                        "(subset=%s): expected %d tokens, got %d. "
                        "String fallback would produce different activations. "
                        "Aborting.",
                        row.get("subset", "?"),
                        len(expected_ids), len(recheck_ids),
                    )
                    raise SystemExit(2)
                string_inputs.append(rendered_str)
            outputs = llm.generate(string_inputs, sp_replay)

        chunk_elapsed = time.monotonic() - chunk_t0
        log.info(
            "replay chunk done in %.1fs (%.2f s/row avg)",
            chunk_elapsed, chunk_elapsed / max(len(rendered_chunk), 1),
        )
        del outputs  # outputs discarded; captures live in writer state

        # Update tally accumulators.
        for row, _, n_tok in rendered_chunk:
            replayed_rows.append(row)
            replayed_token_counts.append(n_tok)
            n_replayed += 1

        total_done_captures = captured_done_base + n_replayed

        # Progress log.
        session_elapsed = time.monotonic() - t0
        log.info(
            "[%d/%d rows consumed] %d replayed, %d skipped -- "
            "%.0fs elapsed (%.2f s/replayed-row avg)",
            rows_done_base + n_replayed + n_skipped,
            rows_done_base + len(replay_rows),
            n_replayed, n_skipped,
            session_elapsed,
            session_elapsed / max(n_replayed, 1),
        )

        # B0 fail-fast (M2: gate on first actual submission only).
        if not first_chunk_checked and _enabled_captures:
            first_chunk_checked = True
            try:
                _model_cls = type(
                    llm.llm_engine.model_executor.driver_worker
                    .model_runner.model
                ).__name__
            except Exception:
                _model_cls = "<unresolved>"
            assert_enabled_captures_nonempty(
                _enabled_captures,
                model_class=_model_cls,
                allow_empty=args.allow_empty_captures,
            )

        # Block-outputs subset gate.
        total_rows_consumed = rows_done_base + n_replayed + n_skipped
        if (
            args.capture_block_outputs
            and total_rows_consumed >= args.block_outputs_subset_size
        ):
            try:
                import vllm.calibration_block_outputs as _bo  # type: ignore
                if not _bo._SUBSET_CLOSED:
                    _bo.close_subset()
                    log.info(
                        "block-outputs: subset closed at %d rows consumed "
                        "(>= subset_size=%d).",
                        total_rows_consumed, args.block_outputs_subset_size,
                    )
            except Exception as exc:
                log.error("block-outputs close_subset failed: %s", exc,
                          exc_info=True)

        # ---- Periodic per-writer checkpoints ---------------------------
        # All counters use total_done_captures (captured rows only, no skips).
        chunk_idx = chunk_start // args.chunk_size

        if (args.capture_imatrix
                and args.imatrix_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.imatrix_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_imatrix as _im  # type: ignore
                    _im.set_n_prompts_accumulated(total_done_captures)
                    _im.dump_imatrix_checkpoint(str(ws.imatrix_ckpt_path))
                    log.info("imatrix: ckpt %d prompts -> %s",
                             total_done_captures, ws.imatrix_ckpt_path)
                except Exception as exc:
                    log.error("imatrix ckpt failed: %s", exc, exc_info=True)

        if (args.capture_reap_scores
                and args.reap_scores_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.reap_scores_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_reap_scores as _reap  # type: ignore
                    _reap.set_n_prompts_accumulated(total_done_captures)
                    _reap.dump_reap_scores_checkpoint(str(ws.reap_ckpt_path))
                    log.info("reap-scores: ckpt %d -> %s",
                             total_done_captures, ws.reap_ckpt_path)
                except Exception as exc:
                    log.error("reap-scores ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_input_covariance
                and args.input_cov_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.input_cov_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_input_cov as _icov  # type: ignore
                    _icov.set_n_prompts_accumulated(total_done_captures)
                    _icov.dump_input_cov_checkpoint(
                        str(ws.input_cov_ckpt_path))
                    log.info("input-cov: ckpt %d -> %s",
                             total_done_captures, ws.input_cov_ckpt_path)
                except Exception as exc:
                    log.error("input-cov ckpt failed: %s", exc, exc_info=True)

        if (args.capture_wanda_scalar_row
                and args.wanda_scalar_row_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.wanda_scalar_row_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
                    _wsr.set_n_prompts_accumulated(total_done_captures)
                    _wsr.dump_wanda_scalar_row_checkpoint(
                        str(ws.wsr_ckpt_path))
                    log.info("wanda-scalar-row: ckpt %d -> %s",
                             total_done_captures, ws.wsr_ckpt_path)
                except Exception as exc:
                    log.error("wanda-scalar-row ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_stage2_profile
                and args.stage2_profile_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.stage2_profile_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_stage2_profile as _s2p  # type: ignore
                    _s2p.set_n_prompts_accumulated(total_done_captures)
                    _s2p.dump_stage2_profile_checkpoint(str(ws.s2p_ckpt_path))
                    log.info("stage2-profile: ckpt %d -> %s",
                             total_done_captures, ws.s2p_ckpt_path)
                except Exception as exc:
                    log.error("stage2-profile ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_per_expert_max
                and args.per_expert_max_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.per_expert_max_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_per_expert_max as _pem  # type: ignore
                    _pem.set_n_prompts_accumulated(total_done_captures)
                    _pem.dump_per_expert_max_checkpoint(str(ws.pem_ckpt_path))
                    log.info("per-expert-max: ckpt %d -> %s",
                             total_done_captures, ws.pem_ckpt_path)
                except Exception as exc:
                    log.error("per-expert-max ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_routing_stats
                and args.routing_stats_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.routing_stats_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_routing_stats as _rts  # type: ignore
                    _rts.set_n_prompts_accumulated(total_done_captures)
                    _rts.dump_routing_stats_checkpoint(str(ws.rts_ckpt_path))
                    log.info("routing-stats: ckpt %d -> %s",
                             total_done_captures, ws.rts_ckpt_path)
                except Exception as exc:
                    log.error("routing-stats ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_router_logits_stats
                and args.router_logits_stats_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.router_logits_stats_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
                    _rlsx.set_n_prompts_accumulated(total_done_captures)
                    _rlsx.dump_router_logits_stats_checkpoint(
                        str(ws.router_logits_ckpt_path))
                    log.info("router-logits-stats: ckpt %d -> %s",
                             total_done_captures, ws.router_logits_ckpt_path)
                except Exception as exc:
                    log.error("router-logits-stats ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_output_reservoir
                and args.output_reservoir_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.output_reservoir_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_output_reservoir as _or  # type: ignore
                    _or.set_n_prompts_accumulated(total_done_captures)
                    _or.dump_output_reservoir_checkpoint(str(ws.or_ckpt_path))
                    log.info("output-reservoir: ckpt %d -> %s",
                             total_done_captures, ws.or_ckpt_path)
                except Exception as exc:
                    log.error("output-reservoir ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_block_outputs
                and args.block_outputs_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.block_outputs_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_block_outputs as _bo  # type: ignore
                    _bo.set_n_prompts_accumulated(total_done_captures)
                    _bo.dump_block_outputs_checkpoint(str(ws.bo_ckpt_path))
                    log.info("block-outputs: ckpt %d -> %s",
                             total_done_captures, ws.bo_ckpt_path)
                except Exception as exc:
                    log.error("block-outputs ckpt failed: %s", exc,
                              exc_info=True)

        # Row-index + capture-counter checkpoint (M1 fix: both counters).
        # n_skipped already reflects all skips accumulated so far (C-NEW-1).
        _write_replay_ckpt(
            replay_ckpt,
            rows_done=rows_done_base + n_replayed + n_skipped,
            captured_done=captured_done_base + n_replayed,
        )

    # End of loop.
    total_done_captures = captured_done_base + n_replayed

    # ------------------------------------------------------------------
    # 10. Skip histogram
    # ------------------------------------------------------------------
    if n_skipped > 0:
        log.warning(
            "replay: %d/%d rows skipped (over --max-model-len=%d). "
            "By subset: %s",
            n_skipped, len(replay_rows), args.max_model_len,
            skipped_by_subset,
        )
        skip_frac = n_skipped / max(len(replay_rows), 1)
        if skip_frac > 0.10:
            log.warning(
                "replay: %.1f%% rows skipped -- consider "
                "--max-model-len %d (corpus max ~%d tokens).",
                100.0 * skip_frac, int(max_required * 1.05), max_required,
            )
    log.info(
        "replay loop done: %d replayed, %d skipped.",
        n_replayed, n_skipped,
    )

    # ------------------------------------------------------------------
    # 11. Correctness gates
    #
    # N3 fix: generate-vs-replay routing match removed -- per-row
    # generate-time captures are not retained; the gate is infeasible.
    # Justification is the causal-masking argument in the design doc.
    # Automated gates: (a) B0 non-empty, (b) code+science tokens > 0,
    # (c) buf_rows >= max_num_batched_tokens (asserted before loop).
    # ------------------------------------------------------------------
    tally = _build_replay_subset_tally(replayed_rows, replayed_token_counts)
    _assert_code_science_nonzero(tally)

    if _enabled_captures:
        assert_enabled_captures_nonempty(
            _enabled_captures,
            model_class="<post-replay>",
            allow_empty=args.allow_empty_captures,
        )

    if n_generate_rows > 0:
        log.info(
            "replay: %d GENERATE rows were in the corpus. "
            "Read-through == generation is justified by causal masking "
            "(identical activations for fixed token sequences). "
            "See tasks/CALIBRATION_V3_CAPTURE_REPLAY_DESIGN.md.",
            n_generate_rows,
        )

    # ------------------------------------------------------------------
    # 12. Sidecar dumps
    #
    # IMATRIX IS SPECIAL:
    #   dump_imatrix(path: str, chunk_count: int)
    #   writes <jsonl>.imatrix.dat (sibling of the input JSONL, NOT
    #   under sidecars/). Requires chunk_count argument.
    #
    # ALL OTHER NINE WRITERS (uniform interface):
    #   dump_<signal>(out_path: Path)
    #   each computes sidecar_path(out_path, signal) internally ->
    #   <jsonl>.parent/sidecars/<jsonl.stem>/<signal>.pt
    # ------------------------------------------------------------------

    # -- imatrix (SPECIAL: sibling .dat + required chunk_count arg) ------
    if args.capture_imatrix:
        imatrix_path = out_path.with_suffix(".imatrix.dat")
        try:
            import vllm.calibration_imatrix as _im  # type: ignore
            _im.set_n_prompts_accumulated(total_done_captures)
            total_p = _im.get_n_prompts_accumulated()
            _im.dump_imatrix(str(imatrix_path), chunk_count=total_p)
            log.info("imatrix -> %s (%d entries from %d prompts)",
                     imatrix_path, len(_im._accumulators), total_p)
            if ws.imatrix_ckpt_path and ws.imatrix_ckpt_path.exists():
                ws.imatrix_ckpt_path.unlink()
        except Exception as exc:
            log.error("imatrix dump failed: %s", exc, exc_info=True)

    # -- reap-scores (uniform: dump_reap_scores(Path)) --------------------
    if args.capture_reap_scores:
        try:
            import vllm.calibration_reap_scores as _reap  # type: ignore
            _reap.set_n_prompts_accumulated(total_done_captures)
            _reap.dump_reap_scores(out_path)
            log.info("reap-scores: dumped sidecar from %d prompts",
                     _reap.get_n_prompts_accumulated())
            if ws.reap_ckpt_path and ws.reap_ckpt_path.exists():
                ws.reap_ckpt_path.unlink()
        except Exception as exc:
            log.error("reap-scores dump failed: %s", exc, exc_info=True)

    # -- input-covariance (uniform: dump_input_cov(Path)) ----------------
    if args.capture_input_covariance:
        try:
            import vllm.calibration_input_cov as _icov  # type: ignore
            _icov.set_n_prompts_accumulated(total_done_captures)
            _icov.dump_input_cov(out_path)
            log.info("input-cov: dumped sidecar from %d prompts",
                     _icov.get_n_prompts_accumulated())
            if ws.input_cov_ckpt_path and ws.input_cov_ckpt_path.exists():
                ws.input_cov_ckpt_path.unlink()
        except Exception as exc:
            log.error("input-cov dump failed: %s", exc, exc_info=True)

    # -- wanda scalar_row (uniform: dump_wanda_scalar_row(Path)) ----------
    if args.capture_wanda_scalar_row:
        try:
            import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
            _wsr.set_n_prompts_accumulated(total_done_captures)
            _wsr.dump_wanda_scalar_row(out_path)
            log.info("wanda-scalar-row: dumped from %d prompts",
                     _wsr.get_n_prompts_accumulated())
            if ws.wsr_ckpt_path and ws.wsr_ckpt_path.exists():
                ws.wsr_ckpt_path.unlink()
        except Exception as exc:
            log.error("wanda-scalar-row dump failed: %s", exc, exc_info=True)

    # -- stage2-profile (uniform: dump_stage2_profile(Path)) --------------
    # layer_input_reservoir rides inside this sidecar; no separate
    # dump needed (H2).
    if args.capture_stage2_profile:
        try:
            import vllm.calibration_stage2_profile as _s2p  # type: ignore
            _s2p.set_n_prompts_accumulated(total_done_captures)
            _s2p.dump_stage2_profile(out_path)
            log.info("stage2-profile: dumped from %d prompts",
                     _s2p.get_n_prompts_accumulated())
            if ws.s2p_ckpt_path and ws.s2p_ckpt_path.exists():
                ws.s2p_ckpt_path.unlink()
        except Exception as exc:
            log.error("stage2-profile dump failed: %s", exc, exc_info=True)

    # -- per-expert-max (uniform: dump_per_expert_max(Path)) --------------
    if args.capture_per_expert_max:
        try:
            import vllm.calibration_per_expert_max as _pem  # type: ignore
            _pem.set_n_prompts_accumulated(total_done_captures)
            _pem.dump_per_expert_max(out_path)
            log.info("per-expert-max: dumped from %d prompts",
                     _pem.get_n_prompts_accumulated())
            if ws.pem_ckpt_path and ws.pem_ckpt_path.exists():
                ws.pem_ckpt_path.unlink()
        except Exception as exc:
            log.error("per-expert-max dump failed: %s", exc, exc_info=True)

    # -- routing-stats (uniform: dump_routing_stats(Path)) ----------------
    if args.capture_routing_stats:
        try:
            import vllm.calibration_routing_stats as _rts  # type: ignore
            _rts.set_n_prompts_accumulated(total_done_captures)
            _rts.dump_routing_stats(out_path)
            log.info("routing-stats: dumped from %d prompts",
                     _rts.get_n_prompts_accumulated())
            if ws.rts_ckpt_path and ws.rts_ckpt_path.exists():
                ws.rts_ckpt_path.unlink()
        except Exception as exc:
            log.error("routing-stats dump failed: %s", exc, exc_info=True)

    # -- router-logits-stats (uniform: dump_router_logits_stats(Path)) ----
    if args.capture_router_logits_stats:
        try:
            import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
            _rlsx.set_n_prompts_accumulated(total_done_captures)
            _rlsx.dump_router_logits_stats(out_path)
            log.info("router-logits-stats: dumped from %d prompts",
                     _rlsx.get_n_prompts_accumulated())
            if (ws.router_logits_ckpt_path
                    and ws.router_logits_ckpt_path.exists()):
                ws.router_logits_ckpt_path.unlink()
        except Exception as exc:
            log.error("router-logits-stats dump failed: %s", exc,
                      exc_info=True)

    # -- output-reservoir (uniform: dump_output_reservoir(Path)) ----------
    if args.capture_output_reservoir:
        try:
            import vllm.calibration_output_reservoir as _or  # type: ignore
            _or.set_n_prompts_accumulated(total_done_captures)
            _or.dump_output_reservoir(out_path)
            log.info("output-reservoir: dumped from %d prompts",
                     _or.get_n_prompts_accumulated())
            if ws.or_ckpt_path and ws.or_ckpt_path.exists():
                ws.or_ckpt_path.unlink()
        except Exception as exc:
            log.error("output-reservoir dump failed: %s", exc, exc_info=True)

    # -- block-outputs (uniform: dump_block_outputs(Path)) ----------------
    if args.capture_block_outputs:
        try:
            import vllm.calibration_block_outputs as _bo  # type: ignore
            _bo.set_n_prompts_accumulated(total_done_captures)
            if not _bo._SUBSET_CLOSED:
                _bo.close_subset()
                log.info(
                    "block-outputs: subset closed pre-dump (%d prompts "
                    "< subset_size=%d -- partial subset shipped).",
                    total_done_captures, args.block_outputs_subset_size,
                )
            _bo.dump_block_outputs(out_path)
            log.info(
                "block-outputs: dumped per-layer sidecars from %d prompts",
                _bo.get_n_prompts_accumulated())
            if ws.bo_ckpt_path and ws.bo_ckpt_path.exists():
                ws.bo_ckpt_path.unlink()
        except Exception as exc:
            log.error("block-outputs dump failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 13. Cleanup + summary
    # ------------------------------------------------------------------
    if replay_ckpt.exists():
        replay_ckpt.unlink()

    sidecar_dir = replay_jsonl.parent / "sidecars" / replay_jsonl.stem
    log.info(
        "v3 replay complete: %d rows replayed, %d skipped. "
        "Sidecars at %s/",
        n_replayed, n_skipped, sidecar_dir,
    )
    return 0


def _introspect_calib_buf_rows(llm, fallback_mbt: int) -> int:
    """Read the live TritonExperts._calib_buf_rows (side-store row capacity).

    Mirrors the C2 introspection in _run_replay: the per-forward Gram temp is
    sized [E, buf_rows+1, H], so buf_rows must match the wheel's actual buffer
    or the scatter overflows / underflows. Falls back to the wheel formula
    max(cudagraph, max_num_batched_tokens, 512) if introspection fails.
    """
    try:
        _model = (llm.llm_engine.model_executor.driver_worker
                  .model_runner.model)
        for _name, _mod in _model.named_modules():
            if getattr(_mod, "global_num_experts", None) is not None and \
                    hasattr(_mod, "quant_method"):
                _te = getattr(getattr(_mod.quant_method, "moe_kernel", None),
                              "fused_experts", None)
                _br = getattr(_te, "_calib_buf_rows", None)
                if isinstance(_br, int) and _br > 0:
                    return _br
                break
    except Exception as _exc:  # noqa: BLE001
        log.warning("input-cov-offload: could not introspect "
                    "_calib_buf_rows (%s)", _exc)
    try:
        _ccs = (llm.llm_engine.vllm_config
                .compilation_config.max_cudagraph_capture_size) or 0
    except AttributeError:
        _ccs = 0
    return max(_ccs, fallback_mbt, 512)


def _run_input_cov_offload(args) -> int:
    """input_covariance capture via the per-layer CPU-offload (windowed) path.

    The all-resident Gram ([n_layers, E, H, H] fp32) is ~172 GB on the target
    model and OOMs. This path allocates only a WINDOW of MoE layers at a time:

      for each window [lo..hi] of MoE layer ids:
        - allocate _ch._INPUT_COV_GPU[li]/_COUNT_GPU[li] for li in window;
          point _ch._INPUT_COV_TEMP_GPU[li] at a single shared temp buffer.
        - set_calibration_max_layer(hi)  -> forward early-exits after hi.
        - run the full corpus (forward-only, prefill) once.
        - snapshot each window layer's Gram + counts to a CPU dict; free the
          GPU slice; advance.

    Each MoE layer is in exactly one window, so every layer integrates over the
    full corpus exactly once (no double counting). The in-graph accumulation in
    moe_runner.py is the SAME resident kernel; it fires for a layer iff that
    layer is present in _ch._INPUT_COV_GPU (per-layer guard), so absent layers
    (prior/future windows) are silently skipped.

    Resume is at WINDOW granularity via <jsonl>.input_cov_offload.ckpt.
    On success writes the canonical input_cov sidecar (same on-disk shape as
    dump_input_cov) and returns 0.
    """
    # ------------------------------------------------------------------
    # 1. Input validation + corpus load (mirrors _run_replay steps 1-2)
    # ------------------------------------------------------------------
    replay_jsonl = Path(args.replay_from).resolve()
    if not replay_jsonl.is_file():
        log.error("--replay-from: file not found: %s", replay_jsonl)
        return 1

    _harden_runtime_env(str(replay_jsonl), args.dtype)

    all_rows: list[dict] = []
    with replay_jsonl.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                log.error("input-cov-offload: invalid JSON at line %d: %s",
                          lineno, exc)
                return 1
            if len(row.get("messages", [])) < 2:
                log.error("input-cov-offload: line %d: messages missing/<2.",
                          lineno)
                return 1
            all_rows.append(row)
    if not all_rows:
        log.error("input-cov-offload: no rows in %s", replay_jsonl)
        return 1
    log.info("input-cov-offload: loaded %d rows from %s",
             len(all_rows), replay_jsonl)

    # ------------------------------------------------------------------
    # 2. Load teacher (forward-only; keep gpu-memory-utilization LOW so the
    #    windowed Gram has room -- vLLM otherwise reserves most VRAM for KV).
    # ------------------------------------------------------------------
    _MBT = args.max_num_batched_tokens or 2048
    _NUM_SEQS = args.max_num_seqs or 256
    llm = _load_teacher_vllm(
        args.teacher,
        args.teacher_revision,
        args.dtype,
        args.gpu_memory_utilization,
        args.max_model_len,
        max_num_seqs=_NUM_SEQS,
        max_num_batched_tokens=_MBT,
        max_logprobs=1,
        moe_backend=_resolve_moe_backend(args),
    )
    tokenizer = llm.get_tokenizer()

    import torch  # noqa: PLC0415  (after vLLM import to keep CUDA init order)
    import vllm.calibration_hooks as _ch  # type: ignore
    import vllm.calibration_input_cov as _icov  # type: ignore
    from moe_compress.utils.cached_calibration_signals import (  # type: ignore
        CovariancePayload, save_covariance, SCHEMA_VERSIONS,
    )

    if _ch._INPUT_COV_MODE != "resident":
        log.error(
            "input-cov-offload: VLLM_CALIB_INPUT_COV_MODE=%r (expected "
            "'resident'). The accumulation kernel is gated on 'resident'; the "
            "env gate in main() should have set it. Aborting.",
            _ch._INPUT_COV_MODE)
        return 1

    # ------------------------------------------------------------------
    # 3. Discover MoE topology + buffer size
    # ------------------------------------------------------------------
    model = _icov._resolve_model(llm)
    if model is None:
        log.error("input-cov-offload: could not resolve model.")
        return 1
    layer_ids, n_experts, d_in, device = _icov._discover_moe_layers(model)
    if not layer_ids or n_experts == 0 or d_in == 0:
        log.error(
            "input-cov-offload: incomplete MoE discovery "
            "(layers=%d, experts=%d, d_in=%d).",
            len(layer_ids), n_experts, d_in)
        return 1
    if device is None:
        device = torch.device("cuda")
    n_layers = len(layer_ids)
    buf_rows = _introspect_calib_buf_rows(llm, _MBT)
    if buf_rows < _MBT:
        log.error(
            "input-cov-offload: _calib_buf_rows=%d < max_num_batched_tokens="
            "%d; forward batches > buf_rows would be skipped. Lower "
            "--max-num-batched-tokens to %d. Aborting.",
            buf_rows, _MBT, buf_rows)
        return 1

    gram_bytes = n_experts * d_in * d_in * 4          # per-layer Gram [E,H,H]
    temp_bytes = n_experts * (buf_rows + 1) * d_in * 4  # shared temp
    log.info(
        "input-cov-offload: topology n_layers=%d, n_experts=%d, d_in=%d, "
        "buf_rows=%d; per-layer Gram=%.2f GB, shared temp=%.2f GB.",
        n_layers, n_experts, d_in, buf_rows,
        gram_bytes / 2**30, temp_bytes / 2**30)

    # ------------------------------------------------------------------
    # 4. Window sizing (auto from free VRAM after model+KV are allocated)
    # ------------------------------------------------------------------
    if args.input_cov_window_size > 0:
        window_size = min(args.input_cov_window_size, n_layers)
        log.info("input-cov-offload: window_size=%d (from "
                 "--input-cov-window-size).", window_size)
    else:
        free_b, total_b = torch.cuda.mem_get_info()
        headroom = max(8 * 2**30, int(0.10 * total_b))
        usable = free_b - temp_bytes - headroom
        window_size = max(0, int(usable // gram_bytes))
        window_size = min(window_size, n_layers)
        log.info(
            "input-cov-offload: auto window_size=%d (free=%.1f GB, total=%.1f "
            "GB, temp=%.2f GB, headroom=%.1f GB, gram/layer=%.2f GB).",
            window_size, free_b / 2**30, total_b / 2**30,
            temp_bytes / 2**30, headroom / 2**30, gram_bytes / 2**30)
        if window_size < 1:
            log.error(
                "input-cov-offload: not enough free VRAM for even ONE layer's "
                "Gram (%.2f GB) + temp (%.2f GB). Lower "
                "--gpu-memory-utilization (vLLM reserves most VRAM for KV "
                "cache) or --max-num-batched-tokens. Aborting.",
                gram_bytes / 2**30, temp_bytes / 2**30)
            return 1

    windows = [layer_ids[i:i + window_size]
               for i in range(0, n_layers, window_size)]
    log.info("input-cov-offload: %d window(s) over %d layers: %s",
             len(windows), n_layers,
             [(w[0], w[-1]) for w in windows])

    # ------------------------------------------------------------------
    # 5. Pre-render the corpus once (reused across every window pass)
    # ------------------------------------------------------------------
    rendered: list[list[int]] = []
    n_skipped = 0
    for row in all_rows:
        result = _render_row_for_replay(row, tokenizer, args.max_model_len)
        if result is None:
            n_skipped += 1
        else:
            tok_ids, _ = result
            rendered.append(tok_ids)
    if not rendered:
        log.error("input-cov-offload: all %d rows over --max-model-len=%d.",
                  len(all_rows), args.max_model_len)
        return 1
    log.info("input-cov-offload: %d rows renderable, %d skipped (over "
             "--max-model-len=%d).", len(rendered), n_skipped,
             args.max_model_len)

    from vllm import SamplingParams  # type: ignore
    sp = SamplingParams(temperature=0.0, max_tokens=1, seed=args.seed)

    # ------------------------------------------------------------------
    # 6. Resume (window granularity)
    # ------------------------------------------------------------------
    offload_ckpt = replay_jsonl.with_suffix(
        replay_jsonl.suffix + ".input_cov_offload.ckpt")
    cpu_sigma: dict = {}        # (layer, expert, "gate_proj") -> Tensor[H,H]
    cpu_counts: dict = {}       # (layer, expert, "gate_proj") -> int
    windows_done = 0
    if args.resume and offload_ckpt.exists():
        try:
            _ck = torch.load(offload_ckpt, map_location="cpu",
                             weights_only=False)
            if (int(_ck.get("n_experts", -1)) == n_experts
                    and int(_ck.get("d_in", -1)) == d_in
                    and int(_ck.get("window_size", -1)) == window_size):
                cpu_sigma = _ck["cpu_sigma"]
                cpu_counts = _ck["cpu_counts"]
                windows_done = int(_ck["windows_done"])
                log.info("input-cov-offload: resume -- %d/%d windows done, "
                         "%d entries hydrated.", windows_done, len(windows),
                         len(cpu_sigma))
            else:
                log.warning("input-cov-offload: ckpt topology mismatch; "
                            "ignoring and restarting from window 0.")
        except Exception as exc:  # noqa: BLE001
            log.warning("input-cov-offload: could not read ckpt (%s); "
                        "restarting from window 0.", exc)

    # ------------------------------------------------------------------
    # 7. Shared per-forward temp (allocated ONCE; reused every window/layer).
    #    Sequential layer execution within a forward zeroes+folds it per
    #    layer before the next, so a single buffer is correct.
    # ------------------------------------------------------------------
    shared_temp = torch.zeros(
        n_experts, buf_rows + 1, d_in, dtype=torch.float32, device=device)

    t0 = time.monotonic()
    for w_idx, window in enumerate(windows):
        if w_idx < windows_done:
            continue
        hi = window[-1]
        # Allocate this window's Gram slices; point temp at the shared buffer.
        _ch._INPUT_COV_GPU.clear()
        _ch._INPUT_COV_COUNT_GPU.clear()
        _ch._INPUT_COV_TEMP_GPU.clear()
        for li in window:
            _ch._INPUT_COV_GPU[li] = torch.zeros(
                n_experts, d_in, d_in, dtype=torch.float32, device=device)
            _ch._INPUT_COV_COUNT_GPU[li] = torch.zeros(
                n_experts, dtype=torch.int64, device=device)
            _ch._INPUT_COV_TEMP_GPU[li] = shared_temp
        _ch.set_calibration_max_layer(hi)
        log.info(
            "input-cov-offload: window %d/%d layers [%d..%d] -- "
            "max_layer=%d, %d Gram slices allocated (%.1f GB).",
            w_idx + 1, len(windows), window[0], hi, hi, len(window),
            len(window) * gram_bytes / 2**30)

        # Forward the full corpus (prefill-only) in chunks.
        w_t0 = time.monotonic()
        for cstart in range(0, len(rendered), args.chunk_size):
            chunk = rendered[cstart: cstart + args.chunk_size]
            requests = [{"prompt_token_ids": ids} for ids in chunk]
            outputs = llm.generate(requests, sp)
            del outputs
            if (cstart // args.chunk_size) % 10 == 0:
                log.info(
                    "input-cov-offload: window %d/%d, %d/%d rows, "
                    "%.0fs elapsed.",
                    w_idx + 1, len(windows),
                    min(cstart + args.chunk_size, len(rendered)),
                    len(rendered), time.monotonic() - t0)
        log.info("input-cov-offload: window %d/%d forward done in %.0fs.",
                 w_idx + 1, len(windows), time.monotonic() - w_t0)

        # Snapshot this window's layers to CPU, then free GPU slices.
        for li in window:
            cov_cpu = _ch._INPUT_COV_GPU[li].cpu()       # [E, H, H]
            cnt_cpu = _ch._INPUT_COV_COUNT_GPU[li].cpu()  # [E]
            for e in range(n_experts):
                c = int(cnt_cpu[e].item())
                if c <= 0:
                    continue
                key = (int(li), int(e), "gate_proj")
                cpu_sigma[key] = cov_cpu[e].clone()
                cpu_counts[key] = c
        _ch._INPUT_COV_GPU.clear()
        _ch._INPUT_COV_COUNT_GPU.clear()
        _ch._INPUT_COV_TEMP_GPU.clear()
        torch.cuda.empty_cache()

        windows_done = w_idx + 1
        tmp = str(offload_ckpt) + ".tmp"
        torch.save({
            "schema": 1,
            "n_experts": n_experts,
            "d_in": d_in,
            "n_layers": n_layers,
            "window_size": window_size,
            "windows_done": windows_done,
            "cpu_sigma": cpu_sigma,
            "cpu_counts": cpu_counts,
        }, tmp)
        os.replace(tmp, offload_ckpt)
        log.info("input-cov-offload: window %d/%d checkpointed "
                 "(%d entries so far).", windows_done, len(windows),
                 len(cpu_sigma))

    _ch.set_calibration_max_layer(None)
    del shared_temp
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 8. Assemble + write the canonical sidecar (same shape as dump_input_cov)
    # ------------------------------------------------------------------
    if not cpu_sigma:
        log.error("input-cov-offload: no entries captured; refusing to write "
                  "an empty sidecar.")
        return 1
    payload = CovariancePayload(
        schema_version=SCHEMA_VERSIONS["covariance"],
        n_experts=n_experts,
        n_layers=n_layers,
        sigma_in=cpu_sigma,
        token_counts=cpu_counts,
    )
    save_covariance(payload, replay_jsonl)
    log.info(
        "input-cov-offload: wrote %d (layer, expert) entries (%d layers x %d "
        "experts) for %s in %.0fs.",
        len(cpu_sigma), n_layers, n_experts, replay_jsonl,
        time.monotonic() - t0)

    if offload_ckpt.exists():
        offload_ckpt.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
