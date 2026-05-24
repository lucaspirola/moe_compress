#!/usr/bin/env python3
"""Build the ``self-traces`` calibration JSONL by running the teacher
with ``enable_thinking=True`` over a prompt set and capturing its deterministic
``<think>...</think>answer`` traces.

Why
---
Every other calibration source (nvidia-cascade, tulu3-sft-mix, qwen3-pretrain-
mix) renders the chat template around the OUTER role headers but the assistant
content is plain SFT/raw-text — none contain ``<think>...</think>`` blocks.
For Qwen3-thinking-mode models the highest-reasoning-quality token positions
live INSIDE that block. The routers (Stage 2.5 KD) and merged expert weights
(Stage 2 SH heal) are never supervised on those positions today.

This script closes the gap: it runs THE SAME teacher the pipeline distils
against, in thinking-mode, over a prompt set, and writes the full chat-
formatted sequences as a JSONL the ``self-traces`` corpus loader reads.

Usage
-----
.. code-block:: bash

    # H200, BF16 teacher (default):
    python max_quality/scripts/build_self_traces_calib.py \
        --teacher Qwen/Qwen3.6-35B-A3B \
        --prompts qwen3-pretrain-mix \
        --num-prompts 5000 \
        --max-new-tokens 4096 \
        --output artifacts/_shared/self_traces.jsonl


Determinism
-----------
Greedy decoding (``do_sample=False``) under a fixed teacher + tokenizer +
prompt set produces byte-identical traces, so the JSONL is reproducible and
the cache invalidates correctly when ANY of the ten cache-key components
changes. The full key is:

  1. ``teacher_repo``        — HF repo id of the teacher
  2. ``teacher_revision``    — git revision / branch / tag at the repo
  3. ``prompts_source``      — composite of source-id + num_prompts + seed
  4. ``num_prompts``         — count of prompts requested (also in prompts_source)
  5. ``seed``                — RNG seed driving prompt shuffling (also in prompts_source)
  6. ``max_new_tokens``      — generation cap
  7. ``batch_size``          — folded in because eager-attention softmax
                                reduction order varies with batch shape
  8. ``load_in_4bit``        — NF4 vs BF16 teacher are not interchangeable
  9. ``decode``              — currently fixed to "greedy"
  10. ``schema_version``     — bump to invalidate ALL prior caches at once

The cache-key is folded into the output filename so multiple variants can
coexist on disk.

Output schema
-------------
JSONL where each row is::

    {"messages": [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": "<think>...</think>\\n\\n<answer>"}
    ]}

The downstream loader in ``utils/calibration.py`` renders each row through
``_render_messages(..., enable_thinking=True)`` so the Qwen3-thinking
tokenizer keeps the ``<think>`` markers in the calibration token stream.

Cost
----
~6h on a single H200 SXM5 for 4000 traces at avg 2000 tokens (greedy,
batch=8, BF16 teacher, kernel-cached). Pass ``--load-in-4bit`` (bitsandbytes
NF4 with ``bnb_4bit_compute_dtype=bfloat16``) to roughly halve VRAM at the
cost of ~10-15% throughput; FP8 throughput is achieved by passing an
already-FP8-quantized teacher repo (no validation; operator's responsibility).

This is a ONE-SHOT pre-step — the JSONL is reused across every Stage-2 / 2.5
run that points ``calibration.source: self-traces``. Re-run only when
the teacher revision or the prompt set changes.
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

log = logging.getLogger("build_self_traces_calib")


# ---------------------------------------------------------------------------
# Prompt sourcing — reuse the qwen3-pretrain-mix faucet for prompt diversity.
# We strip the assistant content (if any) and keep only the user prompt; the
# teacher will re-generate the assistant turn in thinking mode.
# ---------------------------------------------------------------------------


def _iter_prompts_from_qwen3_pretrain_mix(
    num_prompts: int, seed: int,
) -> Iterator[tuple[str, str]]:
    """Yield ``(prompt, domain)`` pairs drawn from the qwen3-pretrain-mix
    datasets, with per-subset counts that preserve the corpus's intended mix.

    Reuses the corpus's existing dataset loaders but strips back to the user
    turn only — the teacher generates fresh thinking-mode responses below.
    The ``domain`` tag travels into the output JSONL so the downstream
    self-traces loader can preserve the same percentages at every draw.
    """
    from moe_compress.utils.calibration import (  # type: ignore
        _QWEN3_MIX_DATASET,
        _QWEN3_MIX_WEIGHTS,
        _shuffled_stream,
        _make_subset_seed,
    )

    per_subset = {
        subset: max(1, int(num_prompts * weight))
        for subset, weight in _QWEN3_MIX_WEIGHTS.items()
    }
    # Warn when num_prompts is too small for the smallest-weight subset to
    # get its expected share. Threshold = 2 * num_subsets / min_weight; below
    # that point the int()-floor in per_subset truncates the smallest subsets
    # to 1 row each and the iteration-order short-circuit at the end of the
    # function may drop later subsets entirely.
    try:
        min_w = min(_QWEN3_MIX_WEIGHTS.values())
        threshold = int(2 * len(_QWEN3_MIX_WEIGHTS) / min_w) if min_w > 0 else 0
        if num_prompts < threshold:
            log.warning(
                "num_prompts=%d is below the diversity threshold (~%d) for "
                "qwen3-pretrain-mix's %d subsets (min weight=%.2f). Smallest "
                "domains may be under-represented or dropped by iteration-"
                "order short-circuit. Raise --num-prompts for production runs.",
                num_prompts, threshold, len(_QWEN3_MIX_WEIGHTS), min_w,
            )
    except Exception:  # noqa: BLE001 — diagnostic only
        pass

    total_yielded = 0
    for subset, count in per_subset.items():
        ds_name = _QWEN3_MIX_DATASET[subset]
        s = _make_subset_seed(seed, subset)
        log.info("prompts: %s — pulling %d from %s (seed=%d)",
                 subset, count, ds_name, s)
        try:
            ds, _ = _shuffled_stream(ds_name, count, s)
        except Exception as err:  # noqa: BLE001
            log.error("prompts: %s failed (%s) — skipping", subset, err)
            continue
        n = 0
        for row in ds:
            if subset == "tulu3":
                msgs = row.get("messages") or []
                user = next(
                    (m.get("content") for m in msgs if m.get("role") == "user"),
                    None,
                )
                if isinstance(user, str) and user.strip():
                    yield user.strip(), subset
                    n += 1
            elif subset == "fineweb":
                text = (row.get("text") or "").strip()
                if text:
                    # Wrap as a "summarize/extend" prompt to give the teacher
                    # something to reason about.
                    yield (
                        f"Read the following passage and explain its key ideas:\n\n{text[:2000]}",
                        subset,
                    )
                    n += 1
            elif subset == "math":
                problem = (row.get("problem") or "").strip()
                if problem:
                    yield problem, subset
                    n += 1
            elif subset == "code":
                instr = (row.get("instruction") or "").strip()
                inp = (row.get("input") or "").strip()
                if instr:
                    yield (instr + (("\n\n" + inp) if inp else ""), subset)
                    n += 1
            elif subset == "qa":
                # databricks-dolly-15k: instruction/context/response.
                instr = (row.get("instruction") or "").strip()
                ctx = (row.get("context") or "").strip()
                if instr:
                    yield (instr + (("\n\n" + ctx) if ctx else ""), subset)
                    n += 1
            elif subset == "creative":
                # euclaise/writingprompts: prompt/story — use prompt as the
                # user turn (teacher generates a new story / reasoning trace).
                prompt = (row.get("prompt") or "").strip()
                if prompt:
                    yield prompt, subset
                    n += 1
            elif subset == "multilingual":
                # CohereForAI/aya_dataset: inputs/targets across 65+ languages.
                inputs = (row.get("inputs") or "").strip()
                if inputs:
                    yield inputs, subset
                    n += 1
            elif subset == "papers":
                # gfissore/arxiv-abstracts-2021: title/abstract — ask the
                # teacher to write the abstract given the paper's title.
                title = (row.get("title") or "").strip()
                if title:
                    yield (
                        f"Write the abstract for an academic paper titled:\n\n{title}",
                        subset,
                    )
                    n += 1
            if n >= count:
                break
        total_yielded += n
        if total_yielded >= num_prompts:
            break


def _iter_prompts_from_jsonl(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(prompt, domain)`` pairs from a JSONL file. Each row must
    carry ``{"prompt": "...", "domain": "..."}`` (or aliases ``user`` /
    ``input`` for the prompt). Rows without a ``domain`` are tagged
    ``"unknown"`` so the downstream loader's domain-mix partitioning still
    has somewhere to put them.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            p = row.get("prompt") or row.get("user") or row.get("input")
            d = row.get("domain") or "unknown"
            if isinstance(p, str) and p.strip():
                yield p.strip(), str(d)


# ---------------------------------------------------------------------------
# Cache-key for the output JSONL — invalidates when teacher / prompts / decode
# params change. The key is folded into the output filename so multiple variants
# coexist on disk.
# ---------------------------------------------------------------------------


def _trace_cache_key(
    teacher_repo: str,
    teacher_revision: str,
    prompts_source: str,
    num_prompts: int,
    seed: int,
    max_new_tokens: int,
    batch_size: int,
    load_in_4bit: bool,
) -> str:
    """Compute the cache key.

    ``batch_size`` and ``load_in_4bit`` are folded in because:
      * eager attention softmax reduction order varies with batch shape, so
        byte-identical greedy output is only guaranteed when batch_size is
        held constant;
      * NF4 quantization (load_in_4bit) materially changes teacher logits vs
        BF16, so the two are NOT interchangeable runs.
    ``seed`` is a standalone payload field for symmetry with ``num_prompts``
    (which is also embedded in ``prompts_source``); the redundancy makes the
    invariance contract obvious from the payload alone.
    Bumping ``schema_version`` invalidates ALL prior caches at once.
    """
    payload = json.dumps({
        "teacher_repo": teacher_repo,
        "teacher_revision": teacher_revision,
        "prompts_source": prompts_source,
        "num_prompts": num_prompts,
        "seed": int(seed),
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "load_in_4bit": bool(load_in_4bit),
        "decode": "greedy",
        "schema_version": 3,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Teacher load + batched generation
# ---------------------------------------------------------------------------


def _load_teacher(repo: str, revision: str, load_in_4bit: bool, tokenizer=None):
    """Load the teacher model. Optionally accepts a pre-loaded ``tokenizer``
    to avoid the duplicate HF-hub fetch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = "int4-nf4(compute=bf16)" if load_in_4bit else "bfloat16"
    log.info("loading teacher %s (revision=%s, dtype=%s)",
             repo, revision, dtype_name)
    tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
        repo, revision=revision,
    )
    kwargs: dict = {
        "revision": revision,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        # BF16 compute dtype is required for sane numerics on H100/H200;
        # without it bnb defaults to FP32 compute and the 4-bit path silently
        # diverges from the BF16 path in subtle (slow + slightly off) ways.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    # Switch PyTorch nn.Module to inference mode (disables dropout/batchnorm
    # statistic updates). This is the standard nn.Module.eval() method — NOT
    # Python's built-in eval() function.
    model.eval()
    return model, tok


def _coerce_eos_ids(eos_token_id) -> set[int]:
    """Normalize ``tokenizer.eos_token_id`` (int, list, tuple, or None) into
    a set of ints. Qwen3-thinking tokenizers often expose multiple EOS ids
    (e.g. ``<|im_end|>`` and ``<|endoftext|>``) as a list."""
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(t) for t in eos_token_id if t is not None}
    # Fallback for exotic tokenizer types — coerce via iteration.
    try:
        return {int(t) for t in eos_token_id if t is not None}  # type: ignore[union-attr]
    except TypeError:
        return set()


def _trim_at_first_eos(row_ids: Iterable[int], eos_ids: set[int]) -> list[int]:
    """Return the prefix of ``row_ids`` up to (but not including) the first
    occurrence of any token in ``eos_ids``. If no eos token is present, the
    full row is returned."""
    out: list[int] = []
    for t in row_ids:
        if int(t) in eos_ids:
            break
        out.append(int(t))
    return out


def _generate_traces(
    model, tokenizer, prompts: list[tuple[str, str]], *,
    batch_size: int, max_new_tokens: int,
    already_done: int = 0,
) -> Iterator[dict]:
    """Greedy-generate teacher traces; yield
    ``{"messages": [...], "domain": "..."}`` rows. The ``domain`` field
    propagates the source-subset tag from the prompt iterator so the
    downstream self-traces loader can preserve the empirical domain mix
    at every draw.

    ``already_done`` is the number of rows the caller has already persisted
    (e.g. from a prior --resume scan). It is added to the progress LOG ONLY
    so the operator sees absolute progress across the full requested set,
    not just this session's slice. It does NOT shift the loop indexing —
    ``prompts`` is the post-resume slice the caller passes in.
    """
    import torch

    # Only fall back to eos as pad when the tokenizer truly lacks a pad
    # token (Qwen3 ships with a distinct <|endoftext|> pad; preserving it
    # keeps eos distinguishable from pad in gen_only). If the tokenizer
    # exposes eos only as a list of ids (no string form / no eos_token_id),
    # peel off the first id and set pad_token_id directly so we don't end up
    # with pad_token=None silently.
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        eos_str = getattr(tokenizer, "eos_token", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_str is not None:
            tokenizer.pad_token = eos_str
        elif isinstance(eos_id, (list, tuple)) and eos_id:
            tokenizer.pad_token_id = int(eos_id[0])
        elif isinstance(eos_id, int):
            tokenizer.pad_token_id = eos_id
        else:
            raise RuntimeError(
                f"tokenizer {type(tokenizer).__name__} has neither pad_token "
                "nor a usable eos_token / eos_token_id to fall back on; "
                "batched generation will fail without padding. Set pad_token "
                "explicitly on this tokenizer before running."
            )
    # Left-pad so batched generate output is right-aligned for slicing.
    tokenizer.padding_side = "left"

    # Precompute the EOS-id set once. Qwen3-thinking exposes multiple eos
    # ids (list); trimming by token-id before decoding is the only way to drop
    # the trailing padding/eos cleanly without relying on the string form of
    # `tokenizer.eos_token` (which is None when the tokenizer has multiple).
    eos_ids = _coerce_eos_ids(getattr(tokenizer, "eos_token_id", None))

    total = len(prompts)
    t0 = time.monotonic()
    yielded = 0  # count what we actually emit, not what we tried.
    for i in range(0, total, batch_size):
        batch = prompts[i:i + batch_size]
        batch_prompts = [p for p, _ in batch]
        batch_domains = [d for _, d in batch]
        # Render each prompt with add_generation_prompt=True so the template
        # lands at the assistant cursor — the model's generation extends from
        # there with the <think> opener.
        rendered: list[str] = []
        for p in batch_prompts:
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
            rendered.append(text)

        # Detect truncation. Tokenize once without a max_length cap so we
        # can compare the natural length against the 2048 ceiling and warn the
        # operator when a prompt is actually being clipped (math/code prompts
        # with long context windows). The double-tokenize cost is negligible
        # vs the teacher forward pass.
        natural = tokenizer(rendered, padding=False, truncation=False)
        max_natural = max((len(ids) for ids in natural["input_ids"]), default=0)
        if max_natural > 2048:
            log.warning(
                "prompt(s) in this batch exceed 2048 tokens (max=%d); "
                "truncating to 2048. Consider raising the truncation ceiling "
                "if your teacher's context window allows it.", max_natural,
            )

        inputs = tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_only = out_ids[:, inputs.input_ids.shape[1]:]
        # Trim by token-id BEFORE decode. Handles single int eos_token_id,
        # list-of-ints eos_token_id, and rows with no eos in gen uniformly.
        trimmed_rows = [
            _trim_at_first_eos(row.tolist(), eos_ids) for row in gen_only
        ]
        decoded = [
            tokenizer.decode(ids, skip_special_tokens=False).strip()
            for ids in trimmed_rows
        ]
        for prompt, domain, ans in zip(batch_prompts, batch_domains, decoded):
            if not ans:
                continue
            yielded += 1
            yield {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ans},
                ],
                "domain": domain,
            }

        elapsed = time.monotonic() - t0
        done = i + len(batch_prompts)
        # Report both throughput metrics — prompts/s for batch-pacing,
        # s/trace based on what actually landed in the JSONL. The bracketed
        # counters are absolute (lifetime) progress across resumed runs, but
        # the s/trace + s/prompt averages are computed from this session's
        # elapsed clock only — label them "session-avg" so the operator
        # doesn't misread them as lifetime averages after --resume.
        s_per_trace = elapsed / yielded if yielded > 0 else float("inf")
        # Surface ABSOLUTE progress across the full requested set, not just
        # this session's slice. Loop indexing is unchanged — we only offset
        # the numbers in the log.
        log_done = already_done + done
        log_total = already_done + total
        log_yielded = already_done + yielded
        log.info(
            "[%d/%d prompts processed, %d traces yielded] — "
            "%.1fs session elapsed (%.1f s/trace session-avg, "
            "%.1f s/prompt session-avg)",
            log_done, log_total, log_yielded,
            elapsed, s_per_trace, elapsed / max(done, 1),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _repo_exists_on_hub(repo: str) -> bool:
    """Best-effort HF Hub preflight. Returns True if the repo id is
    reachable OR if the huggingface_hub helper is unavailable (don't block on
    a missing optional dep). Returns False ONLY when the hub responds with a
    clean "not found" for the repo id.

    Note: ``huggingface_hub.repo_exists`` checks repo-id existence only
    (signature: ``(repo_id, *, repo_type=None, token=None)``); it does NOT
    accept a ``revision=`` kwarg — revision-level existence is a separate
    ``revision_exists()`` helper. For this preflight we just need to catch
    typo'd / private / missing repo ids before paying the model-load bill;
    revision / structural / version-mismatch errors are surfaced by
    AutoModel.from_pretrained downstream.
    """
    try:
        from huggingface_hub import repo_exists  # type: ignore
    except Exception:  # noqa: BLE001
        log.info("repo_exists preflight skipped (huggingface_hub unavailable).")
        return True
    try:
        return bool(repo_exists(repo))
    except Exception as err:  # noqa: BLE001
        log.info("repo_exists preflight inconclusive for %s (%s) — "
                 "proceeding; AutoModel.from_pretrained will surface "
                 "revision / structural / version-mismatch errors.",
                 repo, err)
        return True


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s :: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="Qwen/Qwen3.6-35B-A3B",
                   help="HF repo id of the teacher to distill against. "
                        "Default targets this project's production model "
                        "(see max_quality/configs/qwen36_35b_a3b_30pct.yaml).")
    p.add_argument("--teacher-revision", default="main")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load teacher in 4-bit (bnb NF4 with bf16 compute) "
                        "for lower VRAM. NOTE: changes teacher logits vs BF16 "
                        "and so produces a DIFFERENT output cache key.")
    p.add_argument("--prompts", default="qwen3-pretrain-mix",
                   help="Prompt source: 'qwen3-pretrain-mix' or path to a "
                        "JSONL with {'prompt': '...'} rows.")
    p.add_argument("--num-prompts", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--output", default="artifacts/_shared/self_traces.jsonl")
    p.add_argument("--no-cache-suffix", action="store_true",
                   help="Skip the cache-key suffix on the output filename.")
    p.add_argument("--resume", action="store_true",
                   help="If a .tmp file from a prior run exists, count its "
                        "valid rows and skip that many prompts (in deterministic "
                        "gather order) before generating. Refuses to resume if "
                        "the first OR last row's prompt doesn't match the "
                        "corresponding gathered prompt (seed / prompt-source "
                        "drift detected). If the .tmp already covers the full "
                        "current prompt set, it is promoted to the final file.")
    args = p.parse_args()

    # HF Hub preflight catches the user's typo before we burn ~1-2s on
    # tokenizer-load + several seconds on model-load. We don't block on
    # network/auth failures — only on a confirmed "repo not found".
    if not _repo_exists_on_hub(args.teacher):
        log.error(
            "teacher %s does not exist on the Hugging Face Hub. Check "
            "for a typo or a private repo you're not authenticated for. "
            "(Revision %s is validated downstream by AutoModel.from_pretrained.)",
            args.teacher, args.teacher_revision,
        )
        return 1

    # Determinism: teacher.generate(do_sample=False) is greedy → deterministic
    # under fixed (teacher_repo, revision, prompts, max_new_tokens, batch_size,
    # load_in_4bit).
    cache_key = _trace_cache_key(
        args.teacher, args.teacher_revision,
        f"{args.prompts}#{args.num_prompts}#{args.seed}",
        args.num_prompts, args.seed, args.max_new_tokens,
        args.batch_size, args.load_in_4bit,
    )
    out_path = Path(args.output)
    if not args.no_cache_suffix:
        out_path = out_path.with_name(
            f"{out_path.stem}_{cache_key}{out_path.suffix}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        # --resume + already-finished output is a no-op success, not an
        # error. The cache_key in the filename guarantees the prior run was
        # produced by the same (teacher, prompts, decode) tuple.
        if args.resume:
            log.info("output already exists: %s — --resume given and the "
                     "final file is present; nothing to do.", out_path)
            return 0
        log.warning("output already exists: %s — refusing to overwrite. "
                    "Delete the file (or pick a different --output) to regenerate.",
                    out_path)
        return 1

    log.info("output -> %s (cache_key=%s)", out_path, cache_key)

    # Resolve prompts. Preload the tokenizer once (reused by _load_teacher
    # below) — the prompt iterator itself doesn't need it, but the cached
    # load avoids a duplicate hub round-trip.
    bootstrap_tok = None
    if args.prompts == "qwen3-pretrain-mix":
        from transformers import AutoTokenizer
        bootstrap_tok = AutoTokenizer.from_pretrained(
            args.teacher, revision=args.teacher_revision,
        )
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
        log.error("no prompts gathered — check --prompts source.")
        return 1

    # Underfill detection. When the source is exhausted before we hit
    # --num-prompts the cache_key (which contains num_prompts) lies about the
    # actual content. Loud-warn so the operator notices on log scan, and stop
    # the run unless the user explicitly opts in to a smaller dataset by
    # rerunning with the actual count.
    if len(prompts) < args.num_prompts:
        log.error(
            "prompt source exhausted: gathered %d but --num-prompts=%d. The "
            "cache_key in the output filename encodes the REQUESTED count, "
            "not the actual count, so silently continuing would let a smaller "
            "dataset masquerade as the full one. Re-run with --num-prompts=%d "
            "(or fix the prompt source) to proceed.",
            len(prompts), args.num_prompts, len(prompts),
        )
        return 1

    # Log empirical per-domain breakdown so the operator can verify the mix
    # before paying the teacher-forward bill.
    from collections import Counter
    domain_counts = Counter(d for _, d in prompts)
    total_p = len(prompts)
    log.info("gathered %d prompts; per-domain mix: %s", total_p, {
        d: f"{c} ({c / total_p:.1%})" for d, c in sorted(domain_counts.items())
    })

    # --- Crash recovery via --resume -------------------------------------
    # If a .tmp file from a prior run exists, count its valid rows and skip
    # that many prompts. Determinism guarantees the gather order matches the
    # prior run (same seed + same prompt source); we sanity-check by
    # comparing the .tmp's first AND last user-prompts against
    # prompts[0]/prompts[already_done-1] and refuse to resume if either
    # differs.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    already_done = 0
    if args.resume and tmp_path.exists():
        log.info("resume: scanning existing .tmp at %s ...", tmp_path)
        first_existing_prompt = None
        last_existing_prompt = None
        with tmp_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("resume: malformed JSONL at line %d — stopping "
                                "the resume scan here (%d valid rows so far).",
                                i + 1, already_done)
                    break
                msgs = row.get("messages") or []
                if not msgs:
                    continue
                first_user = msgs[0].get("content")
                if first_existing_prompt is None:
                    first_existing_prompt = first_user
                last_existing_prompt = first_user
                already_done += 1
        if already_done > 0:
            # Critical guard: if the .tmp holds more rows than the current
            # run's prompt list, the naive `prompts = prompts[already_done:]`
            # slice would silently empty the list and the "no more work"
            # branch would promote a .tmp with MORE rows than the requested
            # num_prompts to a final file whose cache_key claims a smaller
            # count. Abort loudly instead — the user must either delete the
            # .tmp or raise --num-prompts to match.
            if already_done > len(prompts):
                log.error(
                    "resume: .tmp at %s contains %d valid rows but the current "
                    "run only gathered %d prompts. Continuing would promote "
                    "more rows to %s than the cache_key (num_prompts=%d) "
                    "advertises — refusing to corrupt the cache. Either:\n"
                    "  (a) delete the .tmp and start fresh, or\n"
                    "  (b) re-run with --num-prompts >= %d to match the .tmp.",
                    tmp_path, already_done, len(prompts), out_path,
                    args.num_prompts, already_done,
                )
                return 1
            # Compare both ends of the resumed range. Drift at row 2+ is
            # caught by the LAST-row check; row-0 drift was already caught.
            if first_existing_prompt != prompts[0][0]:
                log.error(
                    "resume: .tmp's FIRST user-prompt does NOT match the first "
                    "gathered prompt under the current seed/source. The prompt "
                    "stream has drifted since the prior run — refusing to mix "
                    "runs silently. Delete %s and start fresh, or restore the "
                    "prior seed / --prompts source.", tmp_path,
                )
                return 1
            expected_last = prompts[already_done - 1][0]
            if last_existing_prompt != expected_last:
                log.error(
                    "resume: .tmp's LAST user-prompt (row %d) does NOT match "
                    "the corresponding gathered prompt under the current "
                    "seed/source. Mid-stream drift detected — refusing to "
                    "mix runs silently. Delete %s and start fresh, or restore "
                    "the prior seed / --prompts source.",
                    already_done, tmp_path,
                )
                return 1
            log.info("resume: %d valid rows in .tmp; skipping first %d of %d "
                     "gathered prompts.", already_done, already_done, len(prompts))
            if already_done == len(prompts):
                log.info("resume: .tmp already contains all %d prompts — "
                         "promoting %s → %s and exiting cleanly.",
                         already_done, tmp_path, out_path)
                os.replace(tmp_path, out_path)
                return 0
            prompts = prompts[already_done:]
    elif args.resume and not tmp_path.exists():
        # --resume but no .tmp is a benign "fresh start"; surface it so
        # the operator notices if they expected a recovery.
        log.info("resume: --resume requested but no .tmp found at %s; "
                 "starting a fresh run.", tmp_path)
    elif tmp_path.exists() and not args.resume:
        log.warning("existing .tmp at %s will be OVERWRITTEN (--resume not set). "
                    "Pass --resume to recover its rows instead.", tmp_path)

    # Reuse the bootstrap tokenizer when present to avoid a second hub
    # round-trip for the exact same repo+revision.
    model, tokenizer = _load_teacher(
        args.teacher, args.teacher_revision, args.load_in_4bit,
        tokenizer=bootstrap_tok,
    )

    # Open in append mode when resuming with rows already present, else
    # write/truncate. The os.replace at the end gives an atomic rename.
    mode = "a" if already_done > 0 else "w"
    n_new = 0
    with tmp_path.open(mode, encoding="utf-8") as f:
        for row in _generate_traces(
            model, tokenizer, prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            already_done=already_done,
        ):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
    n_total = already_done + n_new
    os.replace(tmp_path, out_path)
    log.info("wrote %d traces (%d resumed + %d new) -> %s",
             n_total, already_done, n_new, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
