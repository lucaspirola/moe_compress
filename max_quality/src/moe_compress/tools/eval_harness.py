"""``tools/eval_harness`` — batched generation + chat-format helpers (S6-4).

Shared low-level harness for the Stage 6 *generative* evals: batched greedy
``model.generate()`` plus the chat-template / thinking-mode / code-extraction
helpers. Extracted from the legacy ``stage6_validate.py`` monolith by S6-4 so
the HumanEval and MATH-500 plugins (``stage6/plugins/{humaneval,math500}.py``)
share a single source for these primitives; ``stage6alt`` reuses them too.

Per the ``tools/`` package contract this module is a leaf utility: it imports
only stdlib + ``torch`` and MUST NEVER import any stage module (e.g.
``stage6_validate``, ``stage6.orchestrator``) or any ``pipeline/`` module.
``tools/`` may depend on ``pipeline/`` in general, but this module's whole
purpose -- being reusable by stage6 and stage6alt without an import cycle --
requires it to stay pipeline-free. The monolith re-imports *this* module at
load time, so a ``from ..stage6_validate import ...`` here would deadlock the
import; nothing in this module does that.

Pattern A (S6-4): every symbol below is a character-identical copy of the
monolith definition. ``stage6_validate.py`` re-imports them (a ``# noqa: F401``
block) so ``run()`` and external callers/tests keep their original import path.
"""
from __future__ import annotations

import logging
import os
import re

import torch

log = logging.getLogger(__name__)


# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Constant -- never override at call sites. This
# is a module-LOCAL copy of the monolith's ``_STAGE6_ATTN_IMPLEMENTATION``: the
# monolith keeps its own definition and is NOT imported here (the leaf-utility
# contract above). Both copies must stay in sync until S6-8 collapses the
# monolith.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"


# ---------------------------------------------------------------------------
# Batched generation (Optimizations #3, #4)
# ---------------------------------------------------------------------------


def _generate_batched(model, tokenizer, prompts: list[str], *, max_new: int,
                      device, batch_size: int = 8) -> list[str]:
    """Batched model.generate() for greedy decoding (do_sample=False).

    Left-pads prompts to the longest in each batch group. Greedy decoding
    produces deterministic outputs regardless of batching.
    Numerically identical to serial generation.
    """
    # H2: Mutates shared tokenizer state (padding_side, pad_token_id) and then
    # restores it in a finally block.  Not safe for concurrent callers — must
    # be called from a single thread or protected by an external lock.
    # Concurrent callers would race on both the save and the restore, producing
    # non-deterministic tokenizer state mid-batch.  Use a copy of the tokenizer
    # if concurrent access is required.
    # Spec §9 #3/#4: argmax-identity bs=1 ↔ batched generate requires
    # eager attention (sdpa/flash can drift batched logits by ~1e-5 and
    # flip near-tied argmax). Defensive re-assertion (the load-time pin
    # in run_pipeline._load_for_stage already enforces this; this guard
    # catches a future regression where someone wraps generate() or
    # swaps attn impl mid-run).
    _attn_impl = getattr(model.config, "_attn_implementation", None)
    if _attn_impl != _STAGE6_ATTN_IMPLEMENTATION:
        raise RuntimeError(
            f"Stage 6 _generate_batched: model.config._attn_implementation="
            f"{_attn_impl!r}, expected {_STAGE6_ATTN_IMPLEMENTATION!r} per "
            "spec §9 #3/#4 binding requirement (bs=1↔batched argmax-identity)."
        )
    original_padding_side = tokenizer.padding_side
    original_pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: list[str] = [""] * len(prompts)

    # Infer device from model when not explicitly set (e.g. device_map="auto")
    _gen_dev = device
    if _gen_dev is None:
        try:
            _gen_dev = next(model.parameters()).device
        except StopIteration:
            pass

    # N-4: Hoist eos_id lookup out of the per-batch inner loop — it does not
    # change between iterations and re-reading it each time is unnecessary.
    eos_id = tokenizer.eos_token_id

    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            # pad_to_multiple_of=64 buckets prompt lengths so torch.compile sees
            # a small set of stable input shapes across batches — combined with
            # StaticCache (set at the compile site), the entire HumanEval +
            # MATH-500 phase trips at most ~2 recompile events instead of one
            # per (seq_len, kv_len) pair. Avoids the autoregressive-recompile
            # segfault inside aot_dispatch_base_graph on torch 2.11+cu130.
            encoded = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                pad_to_multiple_of=64,
                truncation=False, add_special_tokens=False,
            )
            if _gen_dev is not None:
                encoded = {k: v.to(_gen_dev) for k, v in encoded.items()}

            with torch.no_grad():
                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new,
                    # PRIOR: min_new_tokens=max_new was forced to dodge a
                    # torch.compile recompile cascade on variable decode
                    # length. That made the model emit `<think>...` filler
                    # past natural EOS — turning chat-tuned reasoning models
                    # into 0/164 scorers on HumanEval (project_a0_student_
                    # diagnosis_2026_05_15.md). With FX-cache-disable +
                    # spawn-worker + recompile-limit=512 already set as cu130
                    # mitigations, the original recompile concern is moot.
                    # Let generate() early-exit on EOS — chat models actually
                    # finish responses.
                    do_sample=False,
                    # tokenizer.pad_token_id is set to eos_token_id by the guard
                    # above; fall back to 0 if both are absent so generate() never
                    # receives None.
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                )

            input_len = encoded["input_ids"].shape[1]  # padded width, same for all in batch
            for j in range(len(batch_prompts)):
                # Slice from input_len (not attention_mask.sum()) because with left-padding
                # the shorter prompts have prompt_len < padded_len, causing out[j, prompt_len:]
                # to include trailing pad tokens from the input as "generated" tokens.
                gen_ids = out[j, input_len:]
                # Truncate at the first EOS token to avoid garbage when
                # pad_token_id != eos_token_id.
                if eos_id is not None:
                    eos_pos = (gen_ids == eos_id).nonzero(as_tuple=False)
                    if len(eos_pos):
                        gen_ids = gen_ids[:eos_pos[0].item()]
                results[i + j] = tokenizer.decode(gen_ids, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.pad_token_id = original_pad_token_id

    return results


# ---------------------------------------------------------------------------
# Chat-template + thinking-mode helpers for generative evals
# ---------------------------------------------------------------------------


def _stage6_enable_thinking() -> bool:
    """Env knob for thinking-mode in HumanEval/MATH-500.

    Default True: we are verifying that the compressed student retains
    chain-of-thought reasoning capability. Set STAGE6_ENABLE_THINKING=false
    for pure-completion code-gen evals (faster, lower scores).
    """
    val = os.environ.get("STAGE6_ENABLE_THINKING", "true").strip().lower()
    return val not in ("0", "false", "no", "off")


def _chat_format_prompts(
    tokenizer,
    raw_prompts: list[str],
    *,
    system: str | None = None,
) -> list[str]:
    """Wrap each raw user prompt with the model's chat template.

    Returns text strings (not token ids) — `_generate_batched` will re-tokenize
    them with left-padding. Adds `add_generation_prompt=True` so the templated
    string ends at the assistant-open tag and generation starts there.

    Qwen3.5/3.6 chat models default to thinking-mode ON. We pass an explicit
    `enable_thinking` flag per `_stage6_enable_thinking()`.
    """
    enable_thinking = _stage6_enable_thinking()
    out: list[str] = []
    # De-spam: log the first degradation as a warning, downgrade subsequent
    # ones to debug. A non-chat tokenizer would otherwise emit 164 warnings
    # for HumanEval / 500 for MATH-500 and bury legitimate signals.
    _warned_once = False
    for p in raw_prompts:
        msgs = []
        if system is not None:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": p})
        # Try (a) full kwargs (Qwen3.x with thinking control), (b) without
        # enable_thinking (older tokenizers), (c) degrade to raw prompt
        # (non-chat tokenizer / jinja template raises). Catches the broad
        # set of exception types apply_chat_template can raise: TypeError
        # (unknown kwarg), ValueError (malformed messages / role rejected),
        # jinja2.TemplateError (template missing or rejects), AttributeError
        # (method missing on tokenizer), plus anything else — fail safe to
        # raw prompt rather than crashing the whole eval.
        text = None
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            except Exception as exc:           # noqa: BLE001
                (log.warning if not _warned_once else log.debug)(
                    "Stage 6: apply_chat_template fallback failed (%s); "
                    "using raw prompt for this entry", exc,
                )
                _warned_once = True
        except Exception as exc:               # noqa: BLE001
            (log.warning if not _warned_once else log.debug)(
                "Stage 6: apply_chat_template raised %s — using raw prompt "
                "for this entry (tokenizer has no usable chat template)",
                exc,
            )
            _warned_once = True
        # Treat empty-string template output as "no usable output" — fall
        # back to the raw prompt rather than silently sending an empty
        # input to generate() (which would score the problem as a failure
        # for the wrong reason).
        out.append(text if text else p)
    return out


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# After a `def <name>(...):` body, the function ends at the first line that
# is non-blank, not indented, and not a `def`/`class`/`async def`/decorator/
# import. That line (and everything after) is trailing prose like
# "This function works by..." — must be trimmed before running. The negative
# lookahead lets us keep continuations (another top-level def, a decorator,
# imports needed by the body).
_TRAILING_PROSE_RE = re.compile(
    r"\n(?=[A-Za-z][^\n]*$)"            # next line starts with a letter (not indent, not symbol)
    r"(?!def |class |async def |@|from |import )",  # but is not Python top-level construct
    re.MULTILINE,
)


def _extract_code_from_chat_response(text: str, entry_point: str) -> str:
    """Pull executable Python out of a chat-formatted completion.

    Steps:
      1. Strip any <think>...</think> reasoning block (Qwen thinking-mode).
      2. Prefer a ```python ... ``` fenced block (most common).
      3. Fall back to text starting at the first `def <entry_point>(`,
         trimmed at the first top-level prose line so trailing explanation
         ("This function works by...") doesn't break exec().
      4. Return empty string if no code can be located → check_humaneval fails.
    """
    s = _THINK_BLOCK_RE.sub("", text)
    m = _PY_FENCE_RE.search(s)
    if m:
        return m.group(1)
    needle = f"def {entry_point}("
    idx = s.find(needle)
    if idx >= 0:
        body = s[idx:]
        trim = _TRAILING_PROSE_RE.search(body)
        return body[:trim.start()] if trim else body
    return ""


__all__ = [
    "_generate_batched",
    "_stage6_enable_thinking",
    "_chat_format_prompts",
    "_THINK_BLOCK_RE",
    "_PY_FENCE_RE",
    "_TRAILING_PROSE_RE",
    "_extract_code_from_chat_response",
]
