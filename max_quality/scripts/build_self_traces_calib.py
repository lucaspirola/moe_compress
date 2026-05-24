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
the cache invalidates correctly when (teacher_revision, prompts_hash,
max_new_tokens) change. The cache-key is folded into the output filename so
multiple variants can coexist on disk.

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
batch=8, BF16 teacher, kernel-cached). FP8 teacher cuts this to ~4h.

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
from typing import Iterator

log = logging.getLogger("build_self_traces_calib")


# ---------------------------------------------------------------------------
# Prompt sourcing — reuse the qwen3-pretrain-mix faucet for prompt diversity.
# We strip the assistant content (if any) and keep only the user prompt; the
# teacher will re-generate the assistant turn in thinking mode.
# ---------------------------------------------------------------------------


def _iter_prompts_from_qwen3_pretrain_mix(
    tokenizer, num_prompts: int, seed: int,
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
    max_new_tokens: int,
) -> str:
    payload = json.dumps({
        "teacher_repo": teacher_repo,
        "teacher_revision": teacher_revision,
        "prompts_source": prompts_source,
        "num_prompts": num_prompts,
        "max_new_tokens": max_new_tokens,
        "decode": "greedy",
        "schema_version": 1,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Teacher load + batched generation
# ---------------------------------------------------------------------------


def _load_teacher(repo: str, revision: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading teacher %s (revision=%s, 4bit=%s)",
             repo, revision, load_in_4bit)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    kwargs = {
        "revision": revision,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "eager",
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    # Switch PyTorch nn.Module to inference mode (disables dropout/batchnorm
    # statistic updates). This is the standard nn.Module.eval() method — NOT
    # Python's built-in eval() function.
    model.eval()
    return model, tok


def _generate_traces(
    model, tokenizer, prompts: list[tuple[str, str]], *,
    batch_size: int, max_new_tokens: int,
) -> Iterator[dict]:
    """Greedy-generate teacher traces; yield
    ``{"messages": [...], "domain": "..."}`` rows. The ``domain`` field
    propagates the source-subset tag from the prompt iterator so the
    downstream self-traces loader can preserve the empirical domain mix
    at every draw.
    """
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-pad so batched generate output is right-aligned for slicing.
    tokenizer.padding_side = "left"

    total = len(prompts)
    t0 = time.monotonic()
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
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=False)
        for prompt, domain, completion in zip(batch_prompts, batch_domains, decoded):
            # Trim trailing pad/eos tokens — keep the <think>...</think> and
            # final answer intact.
            ans = completion.split(tokenizer.eos_token)[0] if tokenizer.eos_token else completion
            ans = ans.strip()
            if not ans:
                continue
            yield {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ans},
                ],
                "domain": domain,
            }

        elapsed = time.monotonic() - t0
        done = i + len(batch_prompts)
        log.info("[%d/%d] traces generated — %.1fs elapsed (%.1f s/trace avg)",
                 done, total, elapsed, elapsed / max(done, 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s :: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="Qwen/Qwen3.6-35B-A3B",
                   help="HF repo id of the teacher to distill against.")
    p.add_argument("--teacher-revision", default="main")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load teacher in 4-bit via bitsandbytes (lower VRAM).")
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
                        "the first row's prompt doesn't match the first gathered "
                        "prompt (seed / prompt-source drift detected).")
    args = p.parse_args()

    # Determinism: teacher.generate(do_sample=False) is greedy → deterministic
    # under fixed (teacher_repo, revision, prompts, max_new_tokens).
    cache_key = _trace_cache_key(
        args.teacher, args.teacher_revision,
        f"{args.prompts}#{args.num_prompts}#{args.seed}",
        args.num_prompts, args.max_new_tokens,
    )
    out_path = Path(args.output)
    if not args.no_cache_suffix:
        out_path = out_path.with_name(
            f"{out_path.stem}_{cache_key}{out_path.suffix}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        log.warning("output already exists: %s — refusing to overwrite. "
                    "Delete the file (or pick a different --output) to regenerate.",
                    out_path)
        return 1

    log.info("output -> %s (cache_key=%s)", out_path, cache_key)

    # Resolve prompts.
    if args.prompts == "qwen3-pretrain-mix":
        from transformers import AutoTokenizer
        bootstrap_tok = AutoTokenizer.from_pretrained(
            args.teacher, revision=args.teacher_revision,
        )
        prompts_iter = _iter_prompts_from_qwen3_pretrain_mix(
            bootstrap_tok, args.num_prompts, args.seed,
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
    # Log empirical per-domain breakdown so the operator can verify the mix
    # before paying the teacher-forward bill.
    from collections import Counter as _Counter
    domain_counts = _Counter(d for _, d in prompts)
    total_p = len(prompts)
    log.info("gathered %d prompts; per-domain mix: %s", total_p, {
        d: f"{c} ({c / total_p:.1%})" for d, c in sorted(domain_counts.items())
    })

    # --- Crash recovery via --resume -------------------------------------
    # If a .tmp file from a prior run exists, count its valid rows and skip
    # that many prompts. Determinism guarantees the gather order matches the
    # prior run (same seed + same prompt source); we sanity-check by
    # comparing the .tmp's first user-prompt against prompts[0] and refuse
    # to resume if they differ.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    already_done = 0
    if args.resume and tmp_path.exists():
        log.info("resume: scanning existing .tmp at %s ...", tmp_path)
        first_existing_prompt = None
        with tmp_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("resume: malformed JSONL at line %d — stopping "
                                "the resume scan here (%d valid rows so far).",
                                i + 1, already_done)
                    break
                if not row.get("messages"):
                    continue
                if first_existing_prompt is None:
                    first_existing_prompt = row["messages"][0].get("content")
                already_done += 1
        if already_done > 0:
            if first_existing_prompt != prompts[0][0]:
                log.error(
                    "resume: .tmp's first user-prompt does NOT match the first "
                    "gathered prompt under the current seed/source. The prompt "
                    "stream has drifted since the prior run — refusing to mix "
                    "runs silently. Delete %s and start fresh, or restore the "
                    "prior seed / --prompts source.", tmp_path,
                )
                return 1
            log.info("resume: %d valid rows in .tmp; skipping first %d of %d "
                     "gathered prompts.", already_done, already_done, len(prompts))
            prompts = prompts[already_done:]
            if not prompts:
                log.info("resume: .tmp already contains all %d prompts — "
                         "promoting %s → %s and exiting cleanly.",
                         already_done, tmp_path, out_path)
                os.replace(tmp_path, out_path)
                return 0
    elif tmp_path.exists() and not args.resume:
        log.warning("existing .tmp at %s will be OVERWRITTEN (--resume not set). "
                    "Pass --resume to recover its rows instead.", tmp_path)

    model, tokenizer = _load_teacher(
        args.teacher, args.teacher_revision, args.load_in_4bit,
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
