"""Tests for the calibration-v2 JSONL row schema v8 (Items 8+9).

Items 8+9 of the calibration-v2 writers campaign (see
``max_quality/docs/calibration_v2_data_capture_plan.md``) bump the JSONL
schema from v7 to v8 by adding a per-row metadata bundle:

  * ``n_prompt_tokens`` — vLLM-tokenized prompt length.
  * ``n_gen_tokens`` — generated token count (pre-EOS-trim).
  * ``has_think`` — closed ``<think>...</think>`` block present.
  * ``refusal_flag`` — heuristic refusal-opener detector.
  * ``subset`` — duplicate of ``domain`` under the plan-doc name.
  * ``seed_idx`` — duplicate of ``_attempt_idx`` under the plan-doc name.

These tests synthesize fake vLLM ``RequestOutput`` objects and call
``_process_outputs`` directly (no live vLLM engine required).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# The driver lives under max_quality/scripts/; add it to sys.path so we can
# import it as a flat module (the script's own ``__main__`` block does the
# same trick to import build_self_traces_calib).
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_vllm_mod = importlib.import_module("build_self_traces_calib_vllm")


# ---------------------------------------------------------------------------
# Synthetic vLLM RequestOutput builders
# ---------------------------------------------------------------------------


def _make_request_output(
    prompt_token_ids: list[int],
    gen_text: str,
    gen_token_ids: list[int],
    finish_reason: str = "stop",
):
    """Mimic the subset of vllm.RequestOutput / CompletionOutput surface
    that ``_process_outputs`` actually reads. Keeping the duck-typed shape
    means we don't pull vLLM into the test path."""
    gen = SimpleNamespace(
        text=gen_text,
        token_ids=gen_token_ids,
        finish_reason=finish_reason,
        logprobs=None,                          # logits-sidecar path not exercised here
    )
    return SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        outputs=[gen],
    )


def _drive_single(
    prompt_str: str,
    domain: str,
    attempt_idx: int,
    prompt_token_ids: list[int],
    gen_text: str,
    gen_token_ids: list[int],
    finish_reason: str = "stop",
):
    """Run a single-row chunk through ``_process_outputs`` and return the
    yielded JSONL row dict."""
    out_obj = _make_request_output(
        prompt_token_ids, gen_text, gen_token_ids, finish_reason,
    )
    rows = list(_vllm_mod._process_outputs(
        outputs=[out_obj],
        prompts_chunk=[(prompt_str, domain)],
        attempt_idx_chunk=[attempt_idx],
        eos_ids=[151643],                       # arbitrary; not exercised
        logits_top_k=50,
        logits_dir=None,                        # skip the .npz sidecar path
        domain_stats={},
        max_new_tokens=16384,
    ))
    assert len(rows) == 1
    return rows[0]


# ---------------------------------------------------------------------------
# Item 8: per-row metadata schema
# ---------------------------------------------------------------------------


def test_jsonl_row_contains_v8_metadata():
    """All six v8 metadata fields must be present with the correct types."""
    row = _drive_single(
        prompt_str="What is 2 + 2?",
        domain="math",
        attempt_idx=42,
        prompt_token_ids=list(range(17)),       # n_prompt_tokens=17
        gen_text="<think>compute</think>The answer is 4.",
        gen_token_ids=[100, 101, 102, 103, 104],
        finish_reason="stop",
    )

    for field in (
        "n_prompt_tokens",
        "n_gen_tokens",
        "has_think",
        "refusal_flag",
        "subset",
        "seed_idx",
    ):
        assert field in row, f"v8 schema missing field {field!r}"

    assert isinstance(row["n_prompt_tokens"], int)
    assert isinstance(row["n_gen_tokens"], int)
    assert isinstance(row["has_think"], bool)
    assert isinstance(row["refusal_flag"], bool)
    assert isinstance(row["subset"], str)
    assert isinstance(row["seed_idx"], int)

    # Existing v7 fields preserved (backward-compat invariant).
    for legacy in ("messages", "domain", "_complete", "_attempt_idx"):
        assert legacy in row, f"v8 must keep legacy field {legacy!r}"


def test_has_think_detection():
    """``has_think`` is True iff the answer carries a closed
    ``<think>...</think>`` block (mirrors the ``is_complete`` predicate's
    closed-tag requirement)."""
    # 1. With think block.
    row_with = _drive_single(
        prompt_str="Q",
        domain="reasoning",
        attempt_idx=1,
        prompt_token_ids=[1, 2, 3],
        gen_text="<think>step by step</think>Final answer: 42.",
        gen_token_ids=[10, 11, 12],
    )
    assert row_with["has_think"] is True

    # 2. Without any think tag.
    row_without = _drive_single(
        prompt_str="Q",
        domain="reasoning",
        attempt_idx=2,
        prompt_token_ids=[1, 2, 3],
        gen_text="Final answer: 42.",
        gen_token_ids=[10, 11, 12],
    )
    assert row_without["has_think"] is False

    # 3. Unterminated <think> (e.g. length truncation): closed tag required,
    #    so has_think stays False.
    row_unterm = _drive_single(
        prompt_str="Q",
        domain="reasoning",
        attempt_idx=3,
        prompt_token_ids=[1, 2, 3],
        gen_text="<think>step by step but never closed...",
        gen_token_ids=[10, 11, 12],
        finish_reason="length",
    )
    assert row_unterm["has_think"] is False


@pytest.mark.parametrize(
    "answer,expected",
    [
        # Canonical refusal openers — should fire.
        ("I cannot help with that request.", True),
        ("I can't assist with this.", True),
        ("I can’t do that.", True),                          # curly apostrophe
        ("I'm sorry, but I can't comply.", True),
        ("I am sorry, but this is not appropriate.", True),
        ("Sorry, I can't help with that.", True),
        ("Sorry I cannot do this.", True),
        # Case-insensitivity.
        ("i cannot do that.", True),
        # Normal answers — should NOT fire.
        ("The answer is 42.", False),
        ("Let me think about this carefully.", False),
        ("I think the answer is 7.", False),
        # "sorry" mid-sentence is not a refusal opener.
        ("The result, sorry to say, is wrong.", False),
        # Refusal phrase INSIDE the think block is reasoning, not refusal —
        # the heuristic strips a leading think block before matching.
        (
            "<think>I'm sorry, but let me reconsider this approach.</think>"
            "The answer is 42.",
            False,
        ),
        # Refusal OUTSIDE the think block — should still fire.
        (
            "<think>let me reason</think>I cannot answer that.",
            True,
        ),
    ],
)
def test_refusal_flag_heuristic(answer: str, expected: bool):
    """The ``refusal_flag`` heuristic fires on the canonical 5 openers and
    does NOT false-positive on normal answers or reasoning-internal
    "sorry" mentions inside ``<think>...</think>``."""
    row = _drive_single(
        prompt_str="Q",
        domain="chat",
        attempt_idx=0,
        prompt_token_ids=[1, 2, 3],
        gen_text=answer,
        gen_token_ids=[10, 11],
    )
    assert row["refusal_flag"] is expected, (
        f"refusal_flag mismatch for answer={answer!r}: "
        f"got {row['refusal_flag']!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Item 9: deterministic seed_idx
# ---------------------------------------------------------------------------


def test_seed_idx_matches_attempt_idx():
    """``seed_idx`` is the duplicate-key alias of ``_attempt_idx`` (the
    row's position in the shuffled CalibrationSpec source). Both keys
    must coexist (backward compat) and carry the same int value."""
    for idx in (0, 1, 7, 1337, 999_999):
        row = _drive_single(
            prompt_str="Q",
            domain="math",
            attempt_idx=idx,
            prompt_token_ids=[1, 2],
            gen_text="A",
            gen_token_ids=[10],
        )
        assert row["seed_idx"] == idx
        assert row["_attempt_idx"] == idx
        assert row["seed_idx"] == row["_attempt_idx"]


# ---------------------------------------------------------------------------
# Token-count fields match the underlying lengths
# ---------------------------------------------------------------------------


def test_n_prompt_tokens_n_gen_tokens_match_lengths():
    """``n_prompt_tokens`` mirrors ``len(out.prompt_token_ids)`` and
    ``n_gen_tokens`` mirrors ``len(gen.token_ids)`` (un-trimmed emit
    count — see _process_outputs docstring rationale)."""
    prompt_ids = list(range(123))               # 123 prompt tokens
    gen_ids = list(range(1000, 1057))           # 57 generated tokens

    row = _drive_single(
        prompt_str="anything",
        domain="science",
        attempt_idx=5,
        prompt_token_ids=prompt_ids,
        gen_text="<think>x</think>y",
        gen_token_ids=gen_ids,
        finish_reason="stop",
    )

    assert row["n_prompt_tokens"] == 123
    assert row["n_gen_tokens"] == 57

    # Subset duplicates domain.
    assert row["subset"] == "science"
    assert row["domain"] == "science"


def test_n_prompt_tokens_robust_to_missing_attribute():
    """If a future vLLM build omits ``prompt_token_ids`` on RequestOutput,
    the writer falls back to 0 rather than crashing — keeps the JSONL row
    well-formed even on engine surface drift."""
    # Hand-build an object lacking prompt_token_ids entirely.
    gen = SimpleNamespace(
        text="ok",
        token_ids=[1, 2, 3],
        finish_reason="stop",
        logprobs=None,
    )
    out_obj = SimpleNamespace(outputs=[gen])    # NOTE: no prompt_token_ids

    rows = list(_vllm_mod._process_outputs(
        outputs=[out_obj],
        prompts_chunk=[("hi", "chat")],
        attempt_idx_chunk=[0],
        eos_ids=[151643],
        logits_top_k=50,
        logits_dir=None,
        domain_stats={},
        max_new_tokens=16384,
    ))
    assert len(rows) == 1
    assert rows[0]["n_prompt_tokens"] == 0
    assert rows[0]["n_gen_tokens"] == 3


# ---------------------------------------------------------------------------
# Schema-version bump assertion (defensive — the cache_key payload must
# carry schema_version=8 once items 8+9 ship).
# ---------------------------------------------------------------------------


def test_cache_key_carries_schema_version_8():
    """Calling ``_trace_cache_key_vllm`` with the v7-shape arguments must
    fold ``schema_version=8`` so v7 runs do NOT cache-hit v8 runs."""
    import json as _json

    # Re-derive the cache key with a known input and compare against a
    # locally-computed payload that asserts schema_version=8.
    key = _vllm_mod._trace_cache_key_vllm(
        teacher_repo="Qwen/Qwen3.6-35B-A3B",
        teacher_revision="main",
        prompts_source="qwen3-pretrain-mix",
        num_prompts=6500,
        seed=1337,
        max_new_tokens=16384,
        reasoning_budget=4096,
        dtype="bfloat16",
        logits_top_k=50,
    )

    # Independent re-derivation — if someone bumps to v9 without updating
    # this test, both sides shift together; the assertion below is the
    # canary that the writer's schema_version is exactly 8 right now.
    import hashlib as _hashlib
    expected_payload = _json.dumps({
        "teacher_repo": "Qwen/Qwen3.6-35B-A3B",
        "teacher_revision": "main",
        "prompts_source": "qwen3-pretrain-mix",
        "num_prompts": 6500,
        "seed": 1337,
        "max_new_tokens": 16384,
        "reasoning_budget": 4096,
        "dtype": "bfloat16",
        "logits_top_k": 50,
        "decode": "greedy",
        "inference_engine": "vllm",
        "schema_version": 8,
    }, sort_keys=True)
    expected_key = _hashlib.sha256(expected_payload.encode()).hexdigest()[:16]
    assert key == expected_key, (
        "cache_key changed: either schema_version is no longer 8, or one "
        "of the folded fields was renamed. If you bumped to v9 on purpose, "
        "update this test's expected payload accordingly."
    )
