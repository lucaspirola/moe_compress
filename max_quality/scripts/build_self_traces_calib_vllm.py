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
            "keep the default hard-fail."
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
                   help="Capture Stage 2 REAM profile (gate logits + gated "
                        "outputs + covariance + layer-input reservoir) and "
                        "write a sidecar at <jsonl>/sidecars/stage2_profile.pt "
                        "(schema v3). Requires the vLLM patch "
                        "vllm.calibration_stage2_profile (canonical source: "
                        "moe_compress.calibration.stage2_profile_writer). "
                        "Auto-enables VLLM_CALIB_CAPTURE_STAGE2_PROFILE=1 + "
                        "VLLM_CALIB_CAPTURE_ROUTER=1 + "
                        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                        "VLLM_USE_FLASHINFER_MOE_FP16=0 BEFORE any vllm "
                        "import. NOTE: layer-input reservoir is reserved "
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
    args = p.parse_args()

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
        log.info("--capture-input-covariance: enabled "
                 "VLLM_CALIB_CAPTURE_INPUT_COV=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT=1 "
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
        os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
        log.info("--capture-stage2-profile: enabled "
                 "VLLM_CALIB_CAPTURE_STAGE2_PROFILE=1 + "
                 "VLLM_CALIB_CAPTURE_ROUTER=1 + "
                 "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 + "
                 "VLLM_USE_FLASHINFER_MOE_FP16=0 "
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
        )
    elif args.prompts == "qwen3-pretrain-mix-v2":
        prompts_iter = _iter_prompts_from_qwen3_pretrain_mix_v2(
            args.num_prompts, args.seed,
            prev_num_prompts=(args.prev_num_prompts or None),
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
    )
    tokenizer = llm.get_tokenizer()
    eos_ids = _coerce_eos_ids(getattr(tokenizer, "eos_token_id", None))

    # Per-writer checkpoint paths. Hoisted out of the per-feature ``if``
    # blocks so a NameError can't sneak in if a setup() raises before
    # binding the local (the inner gated references below check
    # ``is not None`` rather than relying on the feature flag's
    # truthiness alone -- belt + braces).
    imatrix_ckpt_path = None
    reap_ckpt_path = None
    input_cov_ckpt_path = None
    pem_ckpt_path = None
    rts_ckpt_path = None
    router_logits_ckpt_path = None
    or_ckpt_path = None
    bo_ckpt_path = None

    def _check_ckpt_counter(
        signal_name: str,
        loaded_prompts: int,
        ckpt_path: "Path",
    ) -> None:
        """F-H-6 enforcement: hard-fail on JSONL/ckpt counter divergence.

        Thin wrapper over the module-level :func:`_ckpt_counter_check`
        (NIT-4 / LOW-4 extraction); preserves the closure semantics so
        existing call sites that already capture ``already_done`` +
        ``args.allow_counter_divergence`` from the enclosing scope keep
        working without modification.
        """
        _ckpt_counter_check(
            signal_name,
            loaded_prompts,
            already_done,
            ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )

    # --- imatrix accumulator setup (pre-CUDA-graph) ---------------------
    # The imatrix module must pre-allocate accumulator tensors for every
    # LinearBase / ParallelLMHead instance BEFORE the first captured
    # forward; lazy-alloc during CUDA-graph capture is forbidden.
    if args.capture_imatrix:
        import vllm.calibration_imatrix as _im  # type: ignore
        _im.setup(llm)
        log.info("imatrix: setup complete -- accumulators pre-allocated")

        # Spot-preemption resumability: if a prior run wrote a checkpoint
        # next to the JSONL, hydrate the accumulators NOW (after setup
        # pre-allocated the buffers CUDA-graph capture needs). The loader
        # does an in-place .copy_() so the pinned buffers survive.
        imatrix_ckpt_path = out_path.with_suffix(".imatrix.ckpt")
        if args.resume and imatrix_ckpt_path.exists():
            try:
                loaded_prompts = _im.load_imatrix_checkpoint(
                    str(imatrix_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence (or WARN if
                # --allow-counter-divergence is set). Replaces the
                # legacy WARN-only flow whose silent under-counting
                # would slip past standing-auth pipelines.
                _check_ckpt_counter(
                    "imatrix", loaded_prompts, imatrix_ckpt_path,
                )
                log.info(
                    "imatrix: hydrated %d-prompt checkpoint from %s",
                    loaded_prompts, imatrix_ckpt_path,
                )
            except ValueError as exc:
                # Schema mismatch -- delete and restart cleanly.
                log.error(
                    "imatrix: checkpoint schema mismatch (%s); deleting "
                    "stale checkpoint and starting accumulators from zero.",
                    exc,
                )
                imatrix_ckpt_path.unlink()

    # --- REAP-scores accumulator setup (pre-CUDA-graph) -----------------
    # Mirrors the imatrix block: setup() pre-allocates per-(layer, expert)
    # CPU accumulators so the router + expert_out_unweighted callbacks
    # have somewhere to land. Resume: hydrate the checkpoint that sits
    # next to the JSONL at <jsonl>.reap_scores.ckpt.
    if args.capture_reap_scores:
        import vllm.calibration_reap_scores as _reap  # type: ignore
        _reap.setup(llm)
        log.info("reap-scores: setup complete -- accumulators pre-allocated")

        reap_ckpt_path = out_path.with_suffix(".reap_scores.ckpt")
        if args.resume and reap_ckpt_path.exists():
            try:
                loaded_prompts = _reap.load_reap_scores_checkpoint(
                    str(reap_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence.
                _check_ckpt_counter(
                    "reap-scores", loaded_prompts, reap_ckpt_path,
                )
                log.info(
                    "reap-scores: hydrated %d-prompt checkpoint from %s",
                    loaded_prompts, reap_ckpt_path,
                )
            except ValueError as exc:
                log.error(
                    "reap-scores: checkpoint schema mismatch (%s); deleting "
                    "stale checkpoint and starting accumulators from zero.",
                    exc,
                )
                reap_ckpt_path.unlink()

    # --- input-covariance accumulator setup (pre-CUDA-graph) ------------
    # Mirrors the reap-scores block: setup() registers the expert_in
    # callback and builds the layer→rank map. Resume: hydrate the
    # checkpoint that sits next to the JSONL at <jsonl>.input_cov.ckpt.
    if args.capture_input_covariance:
        import vllm.calibration_input_cov as _icov  # type: ignore
        _icov.setup(llm)
        log.info("input-cov: setup complete -- expert_in callback registered")

        input_cov_ckpt_path = out_path.with_suffix(".input_cov.ckpt")
        if args.resume and input_cov_ckpt_path.exists():
            try:
                loaded_prompts = _icov.load_input_cov_checkpoint(
                    str(input_cov_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence.
                _check_ckpt_counter(
                    "input-cov", loaded_prompts, input_cov_ckpt_path,
                )
                log.info(
                    "input-cov: hydrated %d-prompt checkpoint from %s",
                    loaded_prompts, input_cov_ckpt_path,
                )
            except ValueError as exc:
                log.error(
                    "input-cov: checkpoint schema mismatch (%s); deleting "
                    "stale checkpoint and starting accumulators from zero.",
                    exc,
                )
                input_cov_ckpt_path.unlink()

    # --- Wanda scalar_row accumulator setup (W-1) ----------------------
    # Mirrors the input-cov block: setup() registers the router +
    # expert_in callbacks. Resume: hydrate the checkpoint at
    # <jsonl>.wanda_scalar_row.ckpt.
    wsr_ckpt_path = None
    if args.capture_wanda_scalar_row:
        import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
        _wsr.setup(llm)
        log.info(
            "wanda-scalar-row: setup complete -- router + expert_in "
            "callbacks registered"
        )

        wsr_ckpt_path = out_path.with_suffix(".wanda_scalar_row.ckpt")
        if args.resume and wsr_ckpt_path.exists():
            try:
                loaded_prompts = _wsr.load_wanda_scalar_row_checkpoint(
                    str(wsr_ckpt_path),
                )
                _check_ckpt_counter(
                    "wanda-scalar-row", loaded_prompts, wsr_ckpt_path,
                )
                log.info(
                    "wanda-scalar-row: hydrated %d-prompt checkpoint from %s",
                    loaded_prompts, wsr_ckpt_path,
                )
            except ValueError as exc:
                log.error(
                    "wanda-scalar-row: checkpoint schema mismatch (%s); "
                    "deleting stale checkpoint and starting accumulators "
                    "from zero.",
                    exc,
                )
                wsr_ckpt_path.unlink()

    # --- Stage 2 profile-pass writer setup (Plugin #12 REDO) -----------
    # Mirrors the input-cov block: setup() pins cov_storage_dtype IMMEDIATELY
    # (overriding the default fp32 at activation_hooks.py:961) and
    # registers the router + expert_out_unweighted callbacks. Resume:
    # hydrate the checkpoint at <jsonl>.stage2_profile.ckpt.
    s2p_ckpt_path = None
    if args.capture_stage2_profile:
        import vllm.calibration_stage2_profile as _s2p  # type: ignore
        _s2p.setup(llm, cov_storage_dtype=args.stage2_profile_cov_storage_dtype)
        log.info(
            "stage2-profile: setup complete -- router + "
            "expert_out_unweighted callbacks registered; "
            "cov_storage_dtype=%s",
            args.stage2_profile_cov_storage_dtype,
        )
        s2p_ckpt_path = out_path.with_suffix(".stage2_profile.ckpt")
        if args.resume and s2p_ckpt_path.exists():
            try:
                loaded_prompts = _s2p.load_stage2_profile_checkpoint(
                    str(s2p_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence.
                _check_ckpt_counter(
                    "stage2-profile", loaded_prompts, s2p_ckpt_path,
                )
                log.info(
                    "stage2-profile: hydrated %d-prompt checkpoint from %s",
                    loaded_prompts, s2p_ckpt_path,
                )
            except ValueError as exc:
                log.error(
                    "stage2-profile: checkpoint schema mismatch (%s); "
                    "deleting", exc,
                )
                s2p_ckpt_path.unlink()

    # --- per-expert-max accumulator setup (pre-CUDA-graph) --------------
    # Mirrors the reap-scores / input-cov blocks: setup() registers the
    # expert_out_unweighted callback (chained alongside REAP-scores' own
    # subscriber via the multi-callback registry) and builds the
    # layer→rank map. Resume: hydrate the checkpoint that sits next to
    # the JSONL at <jsonl>.per_expert_max.ckpt.
    if args.capture_per_expert_max:
        import vllm.calibration_per_expert_max as _pem  # type: ignore
        _pem.setup(llm)
        log.info("per-expert-max: setup complete -- "
                 "expert_out_unweighted callback registered")

        pem_ckpt_path = out_path.with_suffix(".per_expert_max.ckpt")
        if args.resume and pem_ckpt_path.exists():
            try:
                loaded_prompts = _pem.load_per_expert_max_checkpoint(
                    str(pem_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence.
                _check_ckpt_counter(
                    "per-expert-max", loaded_prompts, pem_ckpt_path,
                )
                log.info(
                    "per-expert-max: hydrated %d-prompt checkpoint "
                    "from %s",
                    loaded_prompts, pem_ckpt_path,
                )
            except ValueError as exc:
                log.error(
                    "per-expert-max: checkpoint schema mismatch (%s); "
                    "deleting stale checkpoint and starting accumulators "
                    "from zero.",
                    exc,
                )
                pem_ckpt_path.unlink()

    # --- routing-stats accumulator setup (pre-CUDA-graph) ---------------
    # Mirrors the per-expert-max block but subscribes the ``router`` hook
    # (which fires on every MoE backend). Resume: hydrate the checkpoint
    # that sits next to the JSONL at <jsonl>.routing_stats.ckpt.
    if args.capture_routing_stats:
        import vllm.calibration_routing_stats as _rts  # type: ignore
        _rts.setup(llm)
        log.info("routing-stats: setup complete -- "
                 "router callback registered")

        rts_ckpt_path = out_path.with_suffix(".routing_stats.ckpt")
        if args.resume and rts_ckpt_path.exists():
            try:
                loaded_prompts = _rts.load_routing_stats_checkpoint(
                    str(rts_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence.
                _check_ckpt_counter(
                    "routing-stats", loaded_prompts, rts_ckpt_path,
                )
                log.info(
                    "routing-stats: hydrated %d-prompt checkpoint "
                    "from %s",
                    loaded_prompts, rts_ckpt_path,
                )
            except ValueError as exc:
                log.error(
                    "routing-stats: checkpoint schema mismatch (%s); "
                    "deleting stale checkpoint and starting accumulators "
                    "from zero.",
                    exc,
                )
                rts_ckpt_path.unlink()

    # --- router-logits-stats accumulator setup (pre-CUDA-graph) ---------
    # Mirrors the routing-stats block but emits sink-vs-normal
    # aggregates: per-(layer, expert) score_sink_sum / score_normal_sum +
    # fire_on_sink + per-layer n_sink_tokens / n_normal_tokens, plus the
    # bos_token_id captured from the tokenizer. The Stage 1 cache reader
    # hydrates a SinkTokenRoutingAccumulator from this payload. Resume:
    # hydrate the checkpoint that sits next to the JSONL at
    # <jsonl>.router_logits_stats.ckpt.
    if args.capture_router_logits_stats:
        import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
        # Pull the BOS id from the tokenizer so the (forward-compatible)
        # BOS-id sink-mask branch lights up if/when the upstream router
        # dispatch grows the ``input_ids`` kwarg. Today's dispatch does
        # not surface input_ids; the writer falls back to position-0-
        # only sink classification.
        _bos = getattr(tokenizer, "bos_token_id", None)
        _rlsx.setup(llm, bos_token_id=_bos)
        log.info("router-logits-stats: setup complete -- "
                 "router callback registered (bos_token_id=%s)", _bos)

        router_logits_ckpt_path = out_path.with_suffix(
            ".router_logits_stats.ckpt"
        )
        if args.resume and router_logits_ckpt_path.exists():
            try:
                loaded_prompts = _rlsx.load_router_logits_stats_checkpoint(
                    str(router_logits_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence.
                _check_ckpt_counter(
                    "router-logits-stats",
                    loaded_prompts,
                    router_logits_ckpt_path,
                )
                log.info(
                    "router-logits-stats: hydrated %d-prompt "
                    "checkpoint from %s",
                    loaded_prompts, router_logits_ckpt_path,
                )
            except ValueError as exc:
                log.error(
                    "router-logits-stats: checkpoint schema mismatch "
                    "(%s); deleting stale checkpoint and starting "
                    "accumulators from zero.",
                    exc,
                )
                router_logits_ckpt_path.unlink()

    # --- output-reservoir accumulator setup (pre-CUDA-graph) -----------
    # Mirrors the per-expert-max / reap-scores blocks: setup() registers
    # the expert_out_unweighted callback (chained alongside REAP-scores
    # and per-expert-max via the multi-callback registry) and builds the
    # layer->rank map. Resume: hydrate the checkpoint that sits next to
    # the JSONL at <jsonl>.output_reservoir.ckpt.
    if args.capture_output_reservoir:
        import vllm.calibration_output_reservoir as _or  # type: ignore
        _or.setup(llm)
        log.info("output-reservoir: setup complete -- "
                 "expert_out_unweighted callback registered (cap=%d)",
                 args.output_reservoir_cap)

        or_ckpt_path = out_path.with_suffix(".output_reservoir.ckpt")
        if args.resume and or_ckpt_path.exists():
            try:
                loaded_prompts = _or.load_output_reservoir_checkpoint(
                    str(or_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence.
                _check_ckpt_counter(
                    "output-reservoir", loaded_prompts, or_ckpt_path,
                )
                log.info(
                    "output-reservoir: hydrated %d-prompt "
                    "checkpoint from %s",
                    loaded_prompts, or_ckpt_path,
                )
            except ValueError as exc:
                log.error(
                    "output-reservoir: checkpoint schema mismatch "
                    "(%s); deleting stale checkpoint and starting "
                    "reservoirs from zero.",
                    exc,
                )
                or_ckpt_path.unlink()

    # --- block-outputs accumulator setup (pre-CUDA-graph) ---------------
    # Subscribes to the existing ``block_out`` hook dispatched by
    # Qwen3MoeSparseMoeBlock.forward. The driver -- not the writer --
    # owns the prompt counter; once ``already_done + n_new`` reaches
    # args.block_outputs_subset_size we call close_subset() to lock the
    # accumulators. Resume hydrates the closed-flag from the checkpoint,
    # so a resumed run that already closed the subset stays a no-op for
    # the rest of the JSONL.
    if args.capture_block_outputs:
        import vllm.calibration_block_outputs as _bo  # type: ignore
        _bo.setup(llm)
        log.info("block-outputs: setup complete -- "
                 "block_out callback registered (subset_size=%d)",
                 args.block_outputs_subset_size)

        bo_ckpt_path = out_path.with_suffix(".block_outputs.ckpt")
        if args.resume and bo_ckpt_path.exists():
            try:
                loaded_prompts = _bo.load_block_outputs_checkpoint(
                    str(bo_ckpt_path),
                )
                # F-H-6: hard-fail on counter divergence.
                _check_ckpt_counter(
                    "block-outputs", loaded_prompts, bo_ckpt_path,
                )
                log.info(
                    "block-outputs: hydrated %d-prompt checkpoint "
                    "from %s (subset_closed=%s)",
                    loaded_prompts, bo_ckpt_path, _bo._SUBSET_CLOSED,
                )
            except ValueError as exc:
                log.error(
                    "block-outputs: checkpoint schema mismatch (%s); "
                    "deleting stale checkpoint and starting accumulators "
                    "from zero.",
                    exc,
                )
                bo_ckpt_path.unlink()

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
                        _im.dump_imatrix_checkpoint(str(imatrix_ckpt_path))
                        log.info(
                            "imatrix: checkpointed %d prompts -> %s",
                            already_done + n_new, imatrix_ckpt_path,
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
                        _reap.dump_reap_scores_checkpoint(str(reap_ckpt_path))
                        log.info(
                            "reap-scores: checkpointed %d prompts -> %s",
                            already_done + n_new, reap_ckpt_path,
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
                            str(input_cov_ckpt_path),
                        )
                        log.info(
                            "input-cov: checkpointed %d prompts -> %s",
                            already_done + n_new, input_cov_ckpt_path,
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
                            str(wsr_ckpt_path),
                        )
                        log.info(
                            "wanda-scalar-row: checkpointed %d prompts -> %s",
                            already_done + n_new, wsr_ckpt_path,
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
                        _s2p.dump_stage2_profile_checkpoint(str(s2p_ckpt_path))
                        log.info(
                            "stage2-profile: checkpointed %d prompts -> %s",
                            already_done + n_new, s2p_ckpt_path,
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
                            str(pem_ckpt_path),
                        )
                        log.info(
                            "per-expert-max: checkpointed %d prompts -> %s",
                            already_done + n_new, pem_ckpt_path,
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
                            str(rts_ckpt_path),
                        )
                        log.info(
                            "routing-stats: checkpointed %d prompts -> %s",
                            already_done + n_new, rts_ckpt_path,
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
                            str(router_logits_ckpt_path),
                        )
                        log.info(
                            "router-logits-stats: checkpointed %d prompts "
                            "-> %s",
                            already_done + n_new, router_logits_ckpt_path,
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
                            str(or_ckpt_path),
                        )
                        log.info(
                            "output-reservoir: checkpointed %d prompts -> %s",
                            already_done + n_new, or_ckpt_path,
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
                            str(bo_ckpt_path),
                        )
                        log.info(
                            "block-outputs: checkpointed %d prompts -> %s",
                            already_done + n_new, bo_ckpt_path,
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
            if imatrix_ckpt_path is not None and imatrix_ckpt_path.exists():
                imatrix_ckpt_path.unlink()
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
            if reap_ckpt_path is not None and reap_ckpt_path.exists():
                reap_ckpt_path.unlink()
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
            if input_cov_ckpt_path is not None and input_cov_ckpt_path.exists():
                input_cov_ckpt_path.unlink()
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
            if wsr_ckpt_path is not None and wsr_ckpt_path.exists():
                wsr_ckpt_path.unlink()
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
            if s2p_ckpt_path is not None and s2p_ckpt_path.exists():
                s2p_ckpt_path.unlink()
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
            if pem_ckpt_path is not None and pem_ckpt_path.exists():
                pem_ckpt_path.unlink()
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
            if rts_ckpt_path is not None and rts_ckpt_path.exists():
                rts_ckpt_path.unlink()
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
            if (router_logits_ckpt_path is not None
                    and router_logits_ckpt_path.exists()):
                router_logits_ckpt_path.unlink()
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
            if or_ckpt_path is not None and or_ckpt_path.exists():
                or_ckpt_path.unlink()
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
            if bo_ckpt_path is not None and bo_ckpt_path.exists():
                bo_ckpt_path.unlink()
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


if __name__ == "__main__":
    sys.exit(main())
