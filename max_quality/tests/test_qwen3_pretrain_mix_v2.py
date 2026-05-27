"""Unit tests for the ``qwen3-pretrain-mix-v2`` calibration corpus.

Step 1 coverage (constants + registration + cache_key):
  * F.5.1 — weights sum to 1.0; dict keys match across all six sole-truth dicts.
  * F.5.2 — every policy is in {GENERATE, TEACHER_FORCED}.
  * F.5.3 — corpus is registered AND v1 corpus is still registered.
  * F.5.4 — v1 vs v2 CalibrationSpec cache_keys differ.

Steps 3+4 add parametric row-extraction tests (F.5.5) and per-subset
behaviour tests (F.5.6 / F.5.7 / F.5.8 / F.5.9).

All tests are CPU-only with no network access — they exercise constants
and registry behavior only.
"""
from __future__ import annotations

import json as _json

import pytest

from moe_compress.utils import calibration as _calib
from moe_compress.utils.calibration import (
    CalibrationSpec,
    _QWEN3_MIX_V2_AVG_TOKENS,
    _QWEN3_MIX_V2_DATASET,
    _QWEN3_MIX_V2_DATASET_CONFIG,
    _QWEN3_MIX_V2_DATASET_SPLIT,
    _QWEN3_MIX_V2_POLICY,
    _QWEN3_MIX_V2_WEIGHTS,
    _extract_glaive_first_user,
    _shuffled_stream,
    _stream_glaive_function_calling,
    _stream_messages_with_config,
    _stream_swe_smith_xml,
    get_corpus_adapter,
)


class _StubTokenizer:
    """Minimal tokenizer stub for the _stream_* helpers — they call
    ``_render_messages`` which falls back to a role/content concat string
    when ``apply_chat_template`` is missing.  We DON'T provide one, so the
    fallback path renders deterministically without pulling transformers
    into the test."""

    name_or_path = "stub-tokenizer"

    # Intentionally NO apply_chat_template — the fallback path in
    # _render_messages serializes role + content + "\n\n" between turns.


# The 12 subsets the plan locks in.
_EXPECTED_SUBSETS = {
    "tulu3",
    "math",
    "qa",
    "creative",
    "multilingual",
    "fineweb",
    "papers",
    "mot_math",
    "mot_code",
    "mot_science",
    "swe_smith",
    "function_calling",
}


# ---------------------------------------------------------------------------
# F.5.1 — weights / dict-key consistency
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    """The 12 weights must sum to exactly 1.0 within float tolerance."""
    total = sum(_QWEN3_MIX_V2_WEIGHTS.values())
    assert total == pytest.approx(1.0, abs=1e-9), (
        f"weights sum to {total!r}, expected 1.0"
    )


def test_all_dicts_share_the_same_subset_keys():
    """The six sole-truth dicts must agree on the subset set — otherwise
    a subset can be silently missing config/policy/dataset/etc. lookups
    at dispatch time."""
    keys = set(_QWEN3_MIX_V2_WEIGHTS)
    assert keys == _EXPECTED_SUBSETS, (
        f"_QWEN3_MIX_V2_WEIGHTS keys {sorted(keys)} != expected "
        f"{sorted(_EXPECTED_SUBSETS)}"
    )
    assert keys == set(_QWEN3_MIX_V2_AVG_TOKENS)
    assert keys == set(_QWEN3_MIX_V2_DATASET)
    assert keys == set(_QWEN3_MIX_V2_DATASET_CONFIG)
    assert keys == set(_QWEN3_MIX_V2_DATASET_SPLIT)
    assert keys == set(_QWEN3_MIX_V2_POLICY)


# ---------------------------------------------------------------------------
# F.5.2 — policy validity
# ---------------------------------------------------------------------------


def test_policy_values_are_valid():
    """Every per-subset policy must be one of the two recognized values.
    The build scripts dispatch on this string; any drift would silently
    skip a subset."""
    valid = {"GENERATE", "TEACHER_FORCED"}
    for subset, policy in _QWEN3_MIX_V2_POLICY.items():
        assert policy in valid, (
            f"subset {subset!r}: invalid policy {policy!r} (expected one of {valid})"
        )


# ---------------------------------------------------------------------------
# F.5.3 — corpus registration (v1 + v2 coexist)
# ---------------------------------------------------------------------------


def test_v2_corpus_registered():
    """The v2 adapter must be registered and look up by name."""
    adapter = get_corpus_adapter("qwen3-pretrain-mix-v2")
    assert adapter.name == "qwen3-pretrain-mix-v2"
    assert callable(adapter.parse_yaml)
    assert callable(adapter.stream_texts)


def test_v1_corpus_still_registered():
    """Backward compat: registering v2 must not displace or override v1."""
    adapter = get_corpus_adapter("qwen3-pretrain-mix")
    assert adapter.name == "qwen3-pretrain-mix"


# ---------------------------------------------------------------------------
# F.5.4 — v1 vs v2 cache-key uniqueness
# ---------------------------------------------------------------------------


def test_shuffled_stream_forwards_config_and_split_kwargs(monkeypatch):
    """The Step-2 widening of ``_shuffled_stream`` must pass ``config`` and
    ``split`` through to ``load_dataset`` exactly, while preserving v1
    behavior when both kwargs are omitted.

    v1 callsite (no kwargs) → ``load_dataset(name, split="train", streaming=True)``.
    v2 mot_* callsite (config="math", split="train") → ``load_dataset(name, "math", split="train", streaming=True)``.
    v2 swe_smith callsite (config=None, split="xml") → ``load_dataset(name, split="xml", streaming=True)``.
    """
    captured: list[tuple[tuple, dict]] = []

    class _StubDS:
        def shuffle(self, *_args, **_kwargs):
            return self

    def _fake_load_dataset(*args, **kwargs):
        captured.append((args, dict(kwargs)))
        return _StubDS()

    # Monkeypatch the symbol where _shuffled_stream looks it up — the lazy
    # ``from datasets import load_dataset`` happens inside the function, so
    # we patch the `datasets` module attribute.
    import datasets as _datasets

    monkeypatch.setattr(_datasets, "load_dataset", _fake_load_dataset)

    # Case 1: v1-style (no kwargs).
    _shuffled_stream("some/dataset", count=5, seed=0)
    assert captured[-1][0] == ("some/dataset",)
    assert captured[-1][1] == {"split": "train", "streaming": True}

    # Case 2: v2 MoT-style (config + split=train).
    _shuffled_stream(
        "open-r1/Mixture-of-Thoughts", count=5, seed=0,
        config="math", split="train",
    )
    assert captured[-1][0] == ("open-r1/Mixture-of-Thoughts", "math")
    assert captured[-1][1] == {"split": "train", "streaming": True}

    # Case 3: v2 SWE-smith-style (config=None, split="xml").
    _shuffled_stream(
        "SWE-bench/SWE-smith-trajectories", count=5, seed=0,
        config=None, split="xml",
    )
    assert captured[-1][0] == ("SWE-bench/SWE-smith-trajectories",)
    assert captured[-1][1] == {"split": "xml", "streaming": True}


# ---------------------------------------------------------------------------
# Step 3 — helper-level row extraction (CPU-only, hand-built fixtures)
# ---------------------------------------------------------------------------


def _patch_shuffled_stream(monkeypatch, rows: list[dict]):
    """Monkeypatch ``_shuffled_stream`` to yield the given canned rows.

    Returns the rows list so callers can mutate it between assertions if
    needed. Replaces the function via monkeypatch on the calibration
    module — safe per pytest's fixture scoping (auto-reverted after the
    test).
    """
    def _fake(name, count, seed, *, config=None, split="train"):
        # circuit_limit just needs to be larger than `count`; the iter
        # below short-circuits on `len(out) >= count`.
        return iter(rows), 10_000

    monkeypatch.setattr(_calib, "_shuffled_stream", _fake)
    return rows


def test_stream_messages_with_config_renders_mot_row(monkeypatch):
    """The MoT-style helper feeds {messages, num_tokens, source} rows
    through ``_render_messages`` with enable_thinking=True. The fallback
    role/content concat path (no apply_chat_template on the stub) produces
    a non-empty string containing both turns."""
    _patch_shuffled_stream(monkeypatch, [
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant",
                 "content": "<think>two plus two</think>4"},
            ],
            "num_tokens": 12,
            "source": "test",
        },
    ])
    out = _stream_messages_with_config(
        "open-r1/Mixture-of-Thoughts", "math", "train",
        count=1, tokenizer=_StubTokenizer(), seed=0,
    )
    assert len(out) == 1
    rendered = out[0]
    assert "What is 2 + 2?" in rendered
    assert "<think>two plus two</think>4" in rendered


def test_stream_swe_smith_xml_flattens_to_first_pair(monkeypatch):
    """swe_smith stores ``messages`` as a JSON string. The helper must
    parse it, drop the system message, keep only the first
    (user, assistant) pair, and discard subsequent turns. The first
    assistant turn carries literal ``<function=...>`` blocks as plain
    string content."""
    raw_messages = [
        {"role": "system", "content": "You are an agent. Use bash."},
        {"role": "user", "content": "<uploaded>repo/</uploaded>Fix bug X."},
        {"role": "assistant",
         "content": "<function=bash><parameter=command>ls</parameter></function>"},
        {"role": "tool", "content": "main.py\n"},
        {"role": "assistant",
         "content": "Now I'll patch main.py with the fix."},
    ]
    _patch_shuffled_stream(monkeypatch, [
        {
            "messages": _json.dumps(raw_messages),
            "instance_id": "x",
            "resolved": True,
            "model": "claude",
            "traj_id": "y",
            "patch": "",
        },
    ])
    out = _stream_swe_smith_xml(
        "SWE-bench/SWE-smith-trajectories", "xml",
        count=1, tokenizer=_StubTokenizer(), seed=0,
    )
    assert len(out) == 1
    rendered = out[0]
    # First user turn present.
    assert "Fix bug X." in rendered
    # First assistant turn (tool-call block) preserved literal.
    assert "<function=bash>" in rendered
    assert "<parameter=command>ls</parameter>" in rendered
    # System content from SWE-smith dropped (Qwen3 reinjects its own).
    assert "You are an agent. Use bash." not in rendered
    # Subsequent assistant turn dropped.
    assert "Now I'll patch main.py" not in rendered


def test_stream_swe_smith_xml_skips_malformed_rows(monkeypatch):
    """Defensive: rows whose ``messages`` is not a JSON-encoded list of
    dicts must be skipped without raising."""
    _patch_shuffled_stream(monkeypatch, [
        {"messages": "not-valid-json{[]"},
        {"messages": None},
        {"messages": "{\"role\": \"user\"}"},   # parses but not a list
        {
            "messages": _json.dumps([
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]),
        },
    ])
    out = _stream_swe_smith_xml(
        "SWE-bench/SWE-smith-trajectories", "xml",
        count=4, tokenizer=_StubTokenizer(), seed=0,
    )
    # Only the well-formed row should yield text.
    assert len(out) == 1
    assert "hello" in out[0]
    assert "world" in out[0]


def test_extract_glaive_first_user_basic():
    """Pulls the first ``USER:`` segment, stops at the next ``ASSISTANT:``
    or ``FUNCTION RESPONSE:`` boundary, strips whitespace and
    ``<|endoftext|>`` sentinels."""
    chat = (
        "USER: pick a function to call <|endoftext|>"
        " ASSISTANT: <functioncall>{...}</functioncall> <|endoftext|>"
        " FUNCTION RESPONSE: ok <|endoftext|>"
        " ASSISTANT: done <|endoftext|>"
        " USER: ignored"
    )
    out = _extract_glaive_first_user(chat)
    assert out is not None
    assert out.startswith("pick a function")
    assert "ignored" not in out
    assert "<functioncall>" not in out
    assert "<|endoftext|>" not in out


def test_extract_glaive_first_user_returns_none_when_no_user():
    """No ``USER:`` marker → None (helper must not crash)."""
    assert _extract_glaive_first_user("ASSISTANT: hi") is None
    assert _extract_glaive_first_user("") is None


def test_stream_glaive_function_calling_combines_system_and_user(monkeypatch):
    """Glaive helper renders the (stripped) system schema + first USER
    turn as one combined user message — Option (i) from plan §B.6.

    The stub tokenizer's fallback path appends ``USER: <combined>`` so
    we should see BOTH the schema and the first user turn in the
    rendered string.
    """
    _patch_shuffled_stream(monkeypatch, [
        {
            "system": (
                "SYSTEM: You are a helpful assistant with access to the "
                "following functions. {\"name\": \"get_weather\"}"
            ),
            "chat": (
                "USER: What's the weather in Paris?  "
                "ASSISTANT: <functioncall>... <|endoftext|> "
                "USER: dropped"
            ),
        },
    ])
    out = _stream_glaive_function_calling(
        "glaiveai/glaive-function-calling-v2",
        count=1, tokenizer=_StubTokenizer(), seed=0,
    )
    assert len(out) == 1
    rendered = out[0]
    # System schema (sans SYSTEM: prefix) and first user turn both present.
    assert "{\"name\": \"get_weather\"}" in rendered
    assert "What's the weather in Paris?" in rendered
    # Second USER turn dropped.
    assert "dropped" not in rendered


def test_cache_key_distinct_for_v1_v2():
    """CalibrationSpec.cache_key folds ``source`` into the payload; v1 and
    v2 specs that differ only in ``source`` must produce different keys.
    A collision would let v2 overwrite v1's cache file (or vice versa)."""
    v1 = CalibrationSpec(
        num_sequences=4000,
        sequence_length=2048,
        seed=1337,
        source="qwen3-pretrain-mix",
    )
    v2 = CalibrationSpec(
        num_sequences=4000,
        sequence_length=2048,
        seed=1337,
        source="qwen3-pretrain-mix-v2",
    )
    k1 = v1.cache_key("Qwen/Qwen3.6-35B-A3B")
    k2 = v2.cache_key("Qwen/Qwen3.6-35B-A3B")
    assert k1 != k2, (
        f"v1 and v2 cache_keys collide ({k1!r} == {k2!r}). Cache files "
        "would clobber each other on disk."
    )
