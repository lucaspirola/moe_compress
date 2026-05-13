"""Lightweight inline evaluator: WikiText-2 PPL on a small slice.

Direct port from `structural_recovery/eval_quick.py:81-134`.

DeepSpeed correctness rules (P0):

  * Forward passes are COLLECTIVE under ZeRO-3 — every rank must call
    `student(input_ids=...)` together. Eval data is rank-replicated (no
    distributed sampler) so every rank computes the same loss; only rank 0
    logs.
  * `student.generate(...)` requires full params resident. Wrap in
    `deepspeed.zero.GatheredParameters` so all ranks gather, only rank 0
    actually generates, then params re-shard on context exit.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as fn
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from ..config import EvalConfig, WikiText2Config
from ..training.zero3_init import is_zero3

log = logging.getLogger(__name__)


def run(
    student: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    eval_cfg: EvalConfig,
    accelerator: Accelerator,
) -> float | None:
    """Single-shot eval — every rank participates in the forward.

    Returns the WikiText-2 PPL when the eval is enabled, else `None`.
    Resilient: any exception inside eval is caught and logged from rank 0,
    never aborts training. Eval is a diagnostic, not a gate.
    """
    if not eval_cfg.wikitext2.enabled:
        return None
    try:
        ppl = wikitext2_ppl(student, tokenizer, eval_cfg.wikitext2, accelerator)
    except Exception as err:
        # REQ: LLR-0038
        if accelerator.is_main_process:
            log.warning("eval failed (continuing training): %s", err)
        return None
    # REQ: LLR-0038
    if accelerator.is_main_process:
        log.info("eval :: wikitext2_ppl=%.4f", ppl)
    return ppl


# REQ: LLR-0036
def wikitext2_ppl(
    student: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    cfg: WikiText2Config,
    accelerator: Accelerator,
) -> float:
    """Rank-replicated forward computing WikiText-2 perplexity.

    Loads the WikiText-2 raw test split, concatenates all non-empty rows,
    tokenizes once, slices into `cfg.num_sequences` x `cfg.sequence_length`
    fixed-length blocks, then iterates the collective forward and sums
    cross-entropy on shifted labels. Returns `exp(avg_nll)`.

    No `DistributedSampler` is constructed — every rank sees identical
    `input_ids`, so the forward is collective and the per-rank PPLs match.
    """
    from datasets import load_dataset

    seq_len = cfg.sequence_length
    n_seqs_req = cfg.num_sequences
    eval_bs = 1  # micro-batch size for eval; small to keep memory bounded.

    # REQ: LLR-0038
    if accelerator.is_main_process:
        log.info(
            "eval :: loading WikiText-2 (test, %d x %d requested)",
            n_seqs_req,
            seq_len,
        )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # `load_dataset` returns `DatasetDict | Dataset | IterableDataset`; the
    # `split="test"` form is concretely a `Dataset`, but the static type is
    # broader — iterate dynamically, mypy would otherwise reject the index op.
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())

    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    ids_list = encoded["input_ids"]
    # Clamp to what the corpus actually provides — never EOS-pad beyond the
    # real text, which biases PPL low.
    available = max(1, len(ids_list) // seq_len)
    n_seqs = min(n_seqs_req, available)
    # REQ: LLR-0038
    if n_seqs < n_seqs_req and accelerator.is_main_process:
        log.info(
            "eval :: WikiText-2 only fits %d full sequences of %d tokens "
            "(requested %d) — clamping.",
            n_seqs,
            seq_len,
            n_seqs_req,
        )

    ids_list = ids_list[: n_seqs * seq_len]
    # REQ: LLR-0062
    # Construct directly on the target device — avoids a gratuitous CPU
    # allocation + H2D copy. Numerically identical to the prior pattern
    # `torch.tensor(...).to(accelerator.device)`.
    inp = torch.tensor(
        ids_list, dtype=torch.long, device=accelerator.device
    ).view(n_seqs, seq_len)

    # Data is rank-replicated (no distributed sampler) — every rank computes
    # the *same* nll on the *same* sequences. We deliberately do not reduce
    # across ranks; if a future refactor introduces a distributed sampler,
    # also add an `accelerator.gather`/`accelerator.reduce` to avoid silently
    # producing per-rank-different PPLs.
    student.eval()
    nll_total = 0.0
    tok_total = 0
    try:
        with torch.no_grad():
            for i in range(0, n_seqs, eval_bs):
                seq = inp[i : i + eval_bs]
                # COLLECTIVE forward: every rank must call.
                logits = student(input_ids=seq).logits  # [B, T, V]
                shift_logits = logits[:, :-1, :].float()
                shift_labels = seq[:, 1:]
                loss = fn.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="sum",
                )
                nll_total += float(loss.item())
                tok_total += int(shift_labels.numel())
    finally:
        student.train()

    avg_nll = nll_total / max(1, tok_total)
    return float(math.exp(avg_nll))


# REQ: LLR-0037
def log_samples(
    student: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    accelerator: Accelerator,
    *,
    max_new_tokens: int = 128,
) -> None:
    """Greedy generations from fixed prompts.

    Under ZeRO-3, wraps `.generate` in `deepspeed.zero.GatheredParameters` so
    all ranks gather params, then ONLY rank 0 calls generate while the other
    ranks block inside the context. Without the gather, generate either hangs
    or produces garbage on a sharded model.
    """
    if not prompts:
        return
    student.eval()
    try:
        if is_zero3(accelerator):
            import deepspeed

            with deepspeed.zero.GatheredParameters(
                list(student.parameters()), modifier_rank=0
            ):
                if accelerator.is_main_process:
                    _do_generations(
                        student, tokenizer, prompts, max_new_tokens, accelerator
                    )
        else:
            # REQ: LLR-0038
            if accelerator.is_main_process:
                _do_generations(
                    student, tokenizer, prompts, max_new_tokens, accelerator
                )
    finally:
        student.train()


def _do_generations(
    student: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
    accelerator: Accelerator,
) -> None:
    # Defensive: callers (`log_samples`) already guard with
    # `accelerator.is_main_process` before invoking, so this is a redundant
    # check today. Inlining the guard here keeps LLR-0038's "no log message
    # escapes the rank-0 guard" invariant local — a future call site that
    # forgets the outer guard cannot accidentally spam logs from N ranks.
    if not accelerator.is_main_process:
        return
    model: Any = accelerator.unwrap_model(student)
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                accelerator.device
            )
            try:
                out = model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(
                    out[0, ids.shape[-1] :], skip_special_tokens=True
                )
                # REQ: LLR-0038
                log.info("eval :: sample %r -> %r", prompt, text[:200])
            except Exception as err:
                # REQ: LLR-0038
                log.warning("eval :: generate failed for %r: %s", prompt, err)
