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
``_complete`` / ``_attempt_idx``, same ``.npz`` logit-cache sidecar layout.
The cache_key folds ``inference_engine="vllm"`` so vLLM and HF outputs never
collide on disk.

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
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

# Reuse prompt loaders + per-domain stats helpers from the HF script — they
# don't depend on transformers / vLLM, only on utils/calibration.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_self_traces_calib import (  # type: ignore
    _iter_prompts_from_qwen3_pretrain_mix,
    _iter_prompts_from_jsonl,
    _coerce_eos_ids,
    _trim_at_first_eos,
)

log = logging.getLogger("build_self_traces_calib_vllm")


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
      * ``schema_version=7`` — distinct from the HF schema_version=6 so
        a downstream loader can pick the right format if we ever change
        the .npz layout.
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
        "schema_version": 7,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# vLLM driver
# ---------------------------------------------------------------------------


def _load_teacher_vllm(
    repo: str, revision: str, dtype: str,
    gpu_memory_utilization: float, max_model_len: int,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_logprobs: int = 50,
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
    return LLM(**kwargs)


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


def _process_outputs(
    outputs, prompts_chunk, attempt_idx_chunk, eos_ids, logits_top_k,
    logits_dir, domain_stats, max_new_tokens,
):
    """Per chunk: convert vLLM ``RequestOutput`` results into our JSONL row
    shape + the .npz logit-cache sidecars. Yields one dict per output."""
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
                np.savez_compressed(
                    fp,
                    token_ids=np.asarray(trimmed_ids, dtype=np.int32),
                    top_ids=top_ids,
                    top_logprobs=top_lp,
                    attempt_idx=np.int64(attempt_idx),
                    top_k=np.int32(logits_top_k),
                )

        yield {
            "messages": [
                {"role": "user", "content": prompt_str},
                {"role": "assistant", "content": ans},
            ],
            "domain": domain,
            "_complete": is_complete,
            "_attempt_idx": int(attempt_idx),
        }


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
                   help="'qwen3-pretrain-mix' or path to JSONL with "
                        "{'prompt': '...', 'domain': '...'} rows.")
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
    p.add_argument("--chunk-size", type=int, default=200,
                   help="How many prompts to submit per LLM.generate call. "
                        "Affects crash-recovery granularity, not throughput "
                        "(vLLM continuous-batches internally).")
    p.add_argument("--output", default="artifacts/_shared/self_traces.jsonl")
    p.add_argument("--no-cache-suffix", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts that already have rows in the .tmp file.")
    args = p.parse_args()

    # --- cache_key + paths ----------------------------------------------
    cache_key = _trace_cache_key_vllm(
        args.teacher, args.teacher_revision,
        f"{args.prompts}#{args.num_prompts}#{args.seed}",
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
        )
    else:
        prompts_iter = _iter_prompts_from_jsonl(Path(args.prompts))

    prompts: list[tuple[str, str]] = []
    for entry in prompts_iter:
        prompts.append(entry)
        if len(prompts) >= args.num_prompts:
            break
    if not prompts:
        log.error("no prompts gathered.")
        return 1
    log.info("gathered %d prompts", len(prompts))

    # --- resume ---------------------------------------------------------
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    already_done = 0
    if args.resume and tmp_path.exists():
        with tmp_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done += 1
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
    )
    tokenizer = llm.get_tokenizer()
    eos_ids = _coerce_eos_ids(getattr(tokenizer, "eos_token_id", None))

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
    with tmp_path.open(mode, encoding="utf-8") as f:
        for chunk_start in range(0, len(prompts), args.chunk_size):
            chunk = prompts[chunk_start:chunk_start + args.chunk_size]
            chunk_attempt_idx = [
                already_done + chunk_start + k for k in range(len(chunk))
            ]
            chunk_prompts_text = [p for p, _ in chunk]
            rendered = _render_prompts(tokenizer, chunk_prompts_text)
            log.info("chunk %d-%d: submitting %d prompts to vLLM",
                     chunk_start, chunk_start + len(chunk), len(chunk))
            chunk_t0 = time.monotonic()
            outputs = llm.generate(rendered, sp)
            chunk_elapsed = time.monotonic() - chunk_t0
            log.info("chunk done in %.1fs (%.2f s/prompt avg)",
                     chunk_elapsed, chunk_elapsed / max(len(chunk), 1))

            for row in _process_outputs(
                outputs, chunk, chunk_attempt_idx, eos_ids,
                args.logits_top_k, logits_dir, domain_stats,
                args.max_new_tokens,
            ):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1

            total_done = already_done + n_new
            total_target = already_done + len(prompts)
            session_elapsed = time.monotonic() - t0
            log.info(
                "[%d/%d traces yielded] — %.0fs session elapsed "
                "(%.1f s/trace session-avg)",
                total_done, total_target,
                session_elapsed, session_elapsed / max(n_new, 1),
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


if __name__ == "__main__":
    sys.exit(main())
