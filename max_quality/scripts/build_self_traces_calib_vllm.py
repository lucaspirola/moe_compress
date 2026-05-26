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
                # Atomic write: tmp + os.replace so a kill mid-write
                # leaves any previous .npz at `fp` intact (or no file
                # at all). The paired JSONL row is written only AFTER
                # this returns, so resume's re-run of the prompt will
                # cleanly overwrite either way.
                tmp_fp = fp.with_suffix(fp.suffix + ".tmp")
                np.savez_compressed(
                    tmp_fp,
                    token_ids=np.asarray(trimmed_ids, dtype=np.int32),
                    top_ids=top_ids,
                    top_logprobs=top_lp,
                    attempt_idx=np.int64(attempt_idx),
                    top_k=np.int32(logits_top_k),
                )
                os.replace(tmp_fp, fp)

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
    else:
        if args.prev_num_prompts:
            log.error("--prev-num-prompts only supported with --prompts=qwen3-pretrain-mix")
            return 1
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
                if loaded_prompts != already_done:
                    # JSONL/checkpoint divergence. Two cases:
                    #   loaded > already_done : checkpoint is newer than
                    #     JSONL -- impossible given we checkpoint AFTER
                    #     flush, unless the user manually edited the JSONL.
                    #     Trust the checkpoint; warn loudly.
                    #   loaded < already_done : JSONL was flushed but the
                    #     subsequent checkpoint dump didn't complete before
                    #     preemption. The imatrix will have fewer prompts'
                    #     worth of data than the JSONL claims, but that's
                    #     honestly recorded in m_last_chunk.
                    log.warning(
                        "imatrix: checkpoint has %d prompts but JSONL has "
                        "%d rows -- proceeding with the cumulative counter "
                        "from the checkpoint; m_last_chunk in the final "
                        ".imatrix.dat will reflect actual accumulated count.",
                        loaded_prompts, already_done,
                    )
                else:
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
                if loaded_prompts != already_done:
                    log.warning(
                        "reap-scores: checkpoint has %d prompts but JSONL "
                        "has %d rows -- proceeding with the cumulative "
                        "counter from the checkpoint.",
                        loaded_prompts, already_done,
                    )
                else:
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
                if loaded_prompts != already_done:
                    log.warning(
                        "input-cov: checkpoint has %d prompts but JSONL "
                        "has %d rows -- proceeding with the cumulative "
                        "counter from the checkpoint.",
                        loaded_prompts, already_done,
                    )
                else:
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
            if imatrix_ckpt_path.exists():
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
            if reap_ckpt_path.exists():
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
            if input_cov_ckpt_path.exists():
                input_cov_ckpt_path.unlink()
        except Exception as exc:
            log.error("input-cov dump failed: %s", exc, exc_info=True)

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
