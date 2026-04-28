"""Lightweight inline evaluator: WikiText-2 PPL on a small slice + sample generations.

Called between training intervals. Does NOT replace
``moe_compress.stage6_validate`` — that gauntlet (lm-eval ARC-C/HellaSwag,
HumanEval, MATH-500) is run manually after Chapter 1 finishes.

DeepSpeed correctness rules (P0):
  * Forward passes are COLLECTIVE under ZeRO-3 — every rank must call
    ``student(input_ids=...)`` together. Eval data is rank-replicated (no
    distributed sampler) so every rank computes the same loss; only rank 0
    logs.
  * ``student.generate(...)`` requires full params resident. Wrap in
    ``deepspeed.zero.GatheredParameters`` so all ranks gather, only rank 0
    actually generates, then params re-shard on context exit.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def run(student, tokenizer, config: dict[str, Any], accelerator) -> None:
    """Single-shot eval. Logs to root logger from rank 0 only.

    EVERY rank must call this — the forward pass under ZeRO-3 is collective
    and a rank-0-only call would deadlock.

    Resilient: any exception inside eval is caught and logged, never aborts
    training. Eval is a diagnostic, not a gate.
    """
    eval_cfg = config.get("eval", {})
    try:
        if eval_cfg.get("wikitext2", {}).get("enabled", False):
            ppl = _wikitext2_ppl(student, tokenizer, eval_cfg["wikitext2"], accelerator)
            if accelerator.is_main_process:
                log.info("eval :: wikitext2_ppl=%.4f", ppl)
        if eval_cfg.get("sample_generations", {}).get("enabled", False):
            _log_samples(student, tokenizer, eval_cfg["sample_generations"], accelerator)
    except Exception as err:                                     # noqa: BLE001
        if accelerator.is_main_process:
            log.warning("eval failed (continuing training): %s", err)


def final_report(student, tokenizer, config: dict[str, Any], accelerator) -> dict:
    """End-of-run report. COLLECTIVE — every rank participates in the forward.

    Only rank 0 returns a populated dict; other ranks return ``{}``.

    Resilient like ``run`` — final report is a diagnostic that must not abort
    the job after the (already-saved) checkpoint exists.
    """
    out: dict[str, Any] = {}
    eval_cfg = config.get("eval", {})
    try:
        if eval_cfg.get("wikitext2", {}).get("enabled", False):
            # Final pass uses 4× the smoke slice for a tighter estimate.
            cfg = dict(eval_cfg["wikitext2"])
            cfg["num_sequences"] = int(cfg.get("num_sequences", 256)) * 4
            ppl = _wikitext2_ppl(student, tokenizer, cfg, accelerator)
            if accelerator.is_main_process:
                out["wikitext2_ppl"] = ppl
    except Exception as err:                                         # noqa: BLE001
        if accelerator.is_main_process:
            log.warning("final_report eval failed (continuing): %s", err)
    if accelerator.is_main_process:
        log.info("final_report: %s", out)
    return out


# ---------------------------------------------------------------------------
# WikiText-2 PPL — collective forward, rank-0 logs.
# ---------------------------------------------------------------------------


def _wikitext2_ppl(student, tokenizer, cfg: dict[str, Any], accelerator) -> float:
    from datasets import load_dataset

    seq_len = int(cfg.get("sequence_length", 2048))
    n_seqs_req = int(cfg.get("num_sequences", 256))

    if accelerator.is_main_process:
        log.info("eval :: loading WikiText-2 (test, %d×%d requested)",
                 n_seqs_req, seq_len)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())

    ids = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
    # P2: clamp n_seqs to what the corpus actually provides — never EOS-pad
    # beyond the real text, which biases PPL low.
    available = max(1, len(ids) // seq_len)
    n_seqs = min(n_seqs_req, available)
    if n_seqs < n_seqs_req and accelerator.is_main_process:
        log.info("eval :: WikiText-2 only fits %d full sequences of %d tokens "
                 "(requested %d) — clamping.", n_seqs, seq_len, n_seqs_req)

    ids = ids[: n_seqs * seq_len]
    inp = torch.tensor(ids, dtype=torch.long).view(n_seqs, seq_len).to(accelerator.device)

    # NOTE: data is rank-replicated (no distributed sampler), so every rank
    # computes the *same* nll on the *same* sequences. We deliberately do not
    # reduce across ranks. If a future refactor introduces a distributed
    # sampler, also add an `accelerator.gather`/`accelerator.reduce` here to
    # avoid silently producing per-rank-different PPLs.
    student.eval()
    nll_total = 0.0
    tok_total = 0
    try:
        with torch.no_grad():
            for i in range(n_seqs):
                seq = inp[i:i + 1]
                # COLLECTIVE forward: every rank must call.
                logits = student(input_ids=seq).logits         # [1, T, V]
                shift_logits = logits[:, :-1, :].float()
                shift_labels = seq[:, 1:]
                loss = F.cross_entropy(
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


# ---------------------------------------------------------------------------
# Sample generations — gather params under ZeRO-3, generate on rank 0.
# ---------------------------------------------------------------------------


def _log_samples(student, tokenizer, cfg: dict[str, Any], accelerator) -> None:
    """Greedy generations from fixed prompts. Qualitative sanity only.

    Under ZeRO-3 we wrap in ``deepspeed.zero.GatheredParameters`` so all ranks
    gather params, then ONLY rank 0 calls ``generate`` while the other ranks
    block inside the context. Without the gather, generate either hangs or
    produces garbage on a sharded model.
    """
    prompts = list(cfg.get("prompts", []))
    if not prompts:
        return
    max_new = int(cfg.get("max_new_tokens", 128))

    # Detect ZeRO-3 the same way distillation does (avoid circular import).
    is_zero3 = False
    try:
        from accelerate.utils import DistributedType
        is_ds = accelerator.distributed_type == DistributedType.DEEPSPEED
        if is_ds:
            plugin = getattr(accelerator.state, "deepspeed_plugin", None)
            is_zero3 = plugin is not None and int(plugin.zero_stage) >= 3
    except Exception:                                            # noqa: BLE001
        pass

    student.eval()
    try:
        if is_zero3:
            import deepspeed
            with deepspeed.zero.GatheredParameters(
                list(student.parameters()), modifier_rank=0,
            ):
                if accelerator.is_main_process:
                    _do_generations(student, tokenizer, prompts, max_new, accelerator)
        else:
            if accelerator.is_main_process:
                _do_generations(student, tokenizer, prompts, max_new, accelerator)
    finally:
        student.train()


def _do_generations(student, tokenizer, prompts, max_new, accelerator) -> None:
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)
            try:
                # accelerator.unwrap_model so generate() sees the underlying
                # nn.Module (DeepSpeed engine doesn't implement generate).
                model = accelerator.unwrap_model(student)
                out = model.generate(
                    ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
                log.info("eval :: sample %r -> %r", prompt, text[:200])
            except Exception as err:                              # noqa: BLE001
                log.warning("eval :: generate failed for %r: %s", prompt, err)
