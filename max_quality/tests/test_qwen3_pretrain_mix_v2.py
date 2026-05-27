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


# ---------------------------------------------------------------------------
# Step 4 — HF build-script iterator F.5.5/6/7/8/9
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _build_self_traces_calib():
    """Import the build script as a flat module (its __main__ block does the
    same trick from the vLLM sibling). Module-scoped to amortize the cost
    across the parametric F.5.5 test."""
    import importlib
    import sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("build_self_traces_calib")


# Per-subset fixture rows matching the verified Section B schemas. One row
# per subset is enough — the iterator's per-subset target count is at least
# 1, so a single hit suffices to verify the 4-tuple shape + payload.
_SUBSET_FIXTURE_ROWS: dict[str, dict] = {
    "tulu3": {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi back"},
        ],
    },
    "math": {
        "problem": "What is 2 + 2?",
        "generated_solution": "4",
    },
    "qa": {
        "instruction": "Name the largest planet.",
        "context": "",
        "response": "Jupiter.",
    },
    "creative": {
        "prompt": "Write a story about a fox.",
        "story": "Once upon a time...",
    },
    "multilingual": {
        "inputs": "Hola, ¿cómo estás?",
        "targets": "Estoy bien, gracias.",
    },
    "fineweb": {
        "text": "Photosynthesis converts sunlight into chemical energy.",
    },
    "papers": {
        "title": "On the convergence of stochastic gradient descent",
        "abstract": "We show that SGD converges in expectation...",
    },
    "mot_math": {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>two plus two</think>4"},
        ],
        "num_tokens": 100,
        "source": "test",
    },
    "mot_code": {
        "messages": [
            {"role": "user", "content": "Write FizzBuzz."},
            {"role": "assistant",
             "content": "<think>standard pattern</think>def fizzbuzz(n): ..."},
        ],
        "num_tokens": 100,
        "source": "test",
    },
    "mot_science": {
        "messages": [
            {"role": "user", "content": "Explain photosynthesis."},
            {"role": "assistant",
             "content": "<think>light + CO2 + H2O</think>It is the process..."},
        ],
        "num_tokens": 100,
        "source": "test",
    },
    "swe_smith": {
        "messages": _json.dumps([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "<uploaded>fake</uploaded>Fix X"},
            {"role": "assistant",
             "content": "<function=bash><parameter=command>ls</parameter></function>"},
        ]),
        "instance_id": "x",
        "resolved": True,
        "model": "claude",
        "traj_id": "y",
        "patch": "",
    },
    "function_calling": {
        "system": (
            "SYSTEM: You are helpful. Use {\"name\": \"get_weather\"}."
        ),
        "chat": (
            "USER: What's the weather in Paris? <|endoftext|> "
            "ASSISTANT: <functioncall>{...} <|endoftext|>"
        ),
    },
}


@pytest.mark.parametrize("subset", sorted(_EXPECTED_SUBSETS))
def test_iter_prompts_v2_returns_4tuples(
    subset, monkeypatch, _build_self_traces_calib,
):
    """F.5.5 — for each of the 12 subsets, the iterator yields a 4-tuple
    with the expected shape: prompt is non-empty str, domain matches the
    subset key, canonical_completion is None iff policy=GENERATE, policy
    matches the sole-truth dict.

    Single-row fixture per subset (matches Section B schemas).
    Monkeypatches ``_shuffled_stream`` on the calibration module — the
    build-script iterator imports that symbol at call time, so the patch
    on the calibration module takes effect when the iterator dispatches.
    """
    fixture_row = _SUBSET_FIXTURE_ROWS[subset]

    def _fake_shuffled_stream(name, count, seed, *, config=None, split="train"):
        # Only the asked-for subset's fixture must yield; every other
        # subset gets an empty iter so the iterator emits ONLY this
        # subset's row.
        if _QWEN3_MIX_V2_DATASET[subset] == name:
            return iter([fixture_row]), 10_000
        return iter([]), 10_000

    monkeypatch.setattr(_calib, "_shuffled_stream", _fake_shuffled_stream)

    iterator = _build_self_traces_calib._iter_prompts_from_qwen3_pretrain_mix_v2(
        num_prompts=8000, seed=1337,
    )
    yielded = list(iterator)
    # Find the tuple(s) for this subset (others are empty by design).
    matching = [t for t in yielded if t[1] == subset]
    assert matching, (
        f"subset {subset!r}: iterator yielded no tuples for fixture row "
        f"(yielded {len(yielded)} other-subset tuples)"
    )
    tup = matching[0]
    assert len(tup) == 4, f"subset {subset!r}: tuple shape {tup!r}"
    prompt, domain, canonical, policy = tup
    assert isinstance(prompt, str) and prompt.strip(), (
        f"subset {subset!r}: prompt is empty"
    )
    assert domain == subset
    assert policy == _QWEN3_MIX_V2_POLICY[subset]
    if policy == "GENERATE":
        assert canonical is None, (
            f"subset {subset!r}: GENERATE row carried canonical {canonical!r}"
        )
    else:
        assert isinstance(canonical, str) and canonical.strip(), (
            f"subset {subset!r}: TEACHER_FORCED row missing canonical"
        )


def test_iter_prompts_v2_swe_smith_drops_subsequent_turns(
    monkeypatch, _build_self_traces_calib,
):
    """F.5.6 — multi-turn SWE-smith row: keep only first (user, asst1);
    drop system, tool, asst2."""
    raw_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first user turn"},
        {"role": "assistant", "content": "first assistant turn with <function=...>"},
        {"role": "tool", "content": "tool out"},
        {"role": "user", "content": "should-be-dropped second user"},
        {"role": "assistant", "content": "should-be-dropped second assistant"},
    ]
    row = {
        "messages": _json.dumps(raw_messages),
        "instance_id": "x",
        "resolved": True,
        "model": "claude",
        "traj_id": "y",
        "patch": "",
    }

    def _fake(name, count, seed, *, config=None, split="train"):
        if name == _QWEN3_MIX_V2_DATASET["swe_smith"]:
            return iter([row]), 10_000
        return iter([]), 10_000

    monkeypatch.setattr(_calib, "_shuffled_stream", _fake)

    tuples = list(
        _build_self_traces_calib._iter_prompts_from_qwen3_pretrain_mix_v2(
            num_prompts=8000, seed=1337,
        )
    )
    swe = [t for t in tuples if t[1] == "swe_smith"]
    assert len(swe) >= 1
    prompt, _domain, canonical, policy = swe[0]
    assert prompt == "first user turn"
    assert canonical == "first assistant turn with <function=...>"
    assert policy == "TEACHER_FORCED"
    # No second-turn leakage in either field.
    assert "should-be-dropped" not in prompt
    assert "should-be-dropped" not in (canonical or "")


def test_iter_prompts_v2_glaive_extracts_first_user(
    monkeypatch, _build_self_traces_calib,
):
    """F.5.7 — Glaive: extracted prompt contains the system schema +
    first USER turn; second USER turn is dropped."""
    row = {
        "system": "SYSTEM: You are helpful. {\"name\": \"f\"}",
        "chat": (
            "USER: pick a function  "
            "ASSISTANT: <functioncall> {...} <|endoftext|>  "
            "FUNCTION RESPONSE: ok  "
            "ASSISTANT: done <|endoftext|>  "
            "USER: ignored-second-user"
        ),
    }

    def _fake(name, count, seed, *, config=None, split="train"):
        if name == _QWEN3_MIX_V2_DATASET["function_calling"]:
            return iter([row]), 10_000
        return iter([]), 10_000

    monkeypatch.setattr(_calib, "_shuffled_stream", _fake)

    tuples = list(
        _build_self_traces_calib._iter_prompts_from_qwen3_pretrain_mix_v2(
            num_prompts=8000, seed=1337,
        )
    )
    fc = [t for t in tuples if t[1] == "function_calling"]
    assert len(fc) >= 1
    prompt, _domain, canonical, policy = fc[0]
    assert "pick a function" in prompt
    assert "{\"name\": \"f\"}" in prompt   # system schema preserved
    assert "ignored-second-user" not in prompt
    assert canonical is None
    assert policy == "GENERATE"


def _diversity_warnings(records):
    """Collector for diversity-threshold warnings — kept as a helper so the
    F.5.8 / F.5.9 tests share the substring contract verbatim."""
    return [
        rec for rec in records
        if "diversity threshold" in rec.getMessage()
    ]


@pytest.fixture
def _attach_caplog_to_build_script_logger(caplog, _build_self_traces_calib):
    """Attach the caplog handler directly to the build script's logger.

    The ROS-on-system ``launch-testing-ros`` plugin overrides
    ``logging.getLogger`` with a ``LaunchLogger`` that pins
    ``propagate=False`` — so records emitted via that logger never reach
    the root logger where caplog's handler is mounted. We attach caplog's
    handler to the build-script's logger directly for the duration of
    the test, then restore on teardown.
    """
    bs_logger = _build_self_traces_calib.log
    bs_logger.setLevel("WARNING")
    bs_logger.addHandler(caplog.handler)
    yield caplog
    bs_logger.removeHandler(caplog.handler)


def test_iter_prompts_v2_diversity_floor_no_warning_at_8000(
    monkeypatch, _build_self_traces_calib,
    _attach_caplog_to_build_script_logger,
):
    """F.5.8 — at num_prompts=8000 (far above 2*12/0.05 = 480), the
    diversity-threshold warning must NOT fire."""
    caplog = _attach_caplog_to_build_script_logger

    def _fake(name, count, seed, *, config=None, split="train"):
        return iter([]), 10_000

    monkeypatch.setattr(_calib, "_shuffled_stream", _fake)

    # Exhaust the iterator so subset-skip logging fires (the diversity
    # check happens BEFORE the iteration loop regardless).
    list(
        _build_self_traces_calib._iter_prompts_from_qwen3_pretrain_mix_v2(
            num_prompts=8000, seed=1337,
        )
    )

    matched = _diversity_warnings(caplog.records)
    assert not matched, (
        f"unexpected diversity-threshold warning at num_prompts=8000: "
        f"{[r.getMessage() for r in matched]}"
    )


def test_iter_prompts_v2_diversity_floor_warns_at_low_num_prompts(
    monkeypatch, _build_self_traces_calib,
    _attach_caplog_to_build_script_logger,
):
    """F.5.9 — at num_prompts=300 (< threshold 480), the warning fires
    exactly once."""
    caplog = _attach_caplog_to_build_script_logger

    def _fake(name, count, seed, *, config=None, split="train"):
        return iter([]), 10_000

    monkeypatch.setattr(_calib, "_shuffled_stream", _fake)

    list(
        _build_self_traces_calib._iter_prompts_from_qwen3_pretrain_mix_v2(
            num_prompts=300, seed=1337,
        )
    )

    matched = _diversity_warnings(caplog.records)
    assert len(matched) == 1, (
        f"expected exactly one diversity-threshold warning, got "
        f"{len(matched)}: {[r.getMessage() for r in matched]}"
    )


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
