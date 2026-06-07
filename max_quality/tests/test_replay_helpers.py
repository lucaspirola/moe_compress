"""Unit tests for _run_replay helper functions (v3 calibration replay).

Tests _render_row_for_replay, _build_replay_subset_tally,
_assert_code_science_nonzero. No model load; no vllm import;
no monkeypatching of production code (project rule).

The stub tokenizer is a self-contained class satisfying the contract:
  apply_chat_template(messages, tokenize, add_generation_prompt, **kw)
  __call__(text, add_special_tokens=False) -> {"input_ids": list[int]}
"""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_self_traces_calib_vllm import (  # type: ignore  # noqa: E402
    _render_row_for_replay,
    _build_replay_subset_tally,
    _assert_code_science_nonzero,
)


class _StubTokenizer:
    """Minimal tokenizer stub: one token per character."""
    def __init__(self, *, raise_on_enable_thinking: bool = False):
        self._raise = raise_on_enable_thinking

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking=None, **kw):
        if self._raise and enable_thinking is not None:
            raise TypeError("enable_thinking not supported by this tokenizer")
        return "".join(m.get("content", "") for m in messages)

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": list(range(len(text)))}


_ROW = {
    "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ],
    "subset": "mot_code",
    "domain": "mot_code",
    "completion_source": "canonical",
}
# "helloworld" = 10 chars = 10 tokens under _StubTokenizer


def test_render_happy_path():
    result = _render_row_for_replay(_ROW, _StubTokenizer(), max_model_len=100)
    assert result is not None
    ids, n = result
    assert n == 10
    assert len(ids) == 10


def test_render_over_length_returns_none():
    result = _render_row_for_replay(_ROW, _StubTokenizer(), max_model_len=9)
    assert result is None


def test_render_exactly_at_limit_accepted():
    result = _render_row_for_replay(_ROW, _StubTokenizer(), max_model_len=10)
    assert result is not None


def test_render_enable_thinking_fallback():
    tok = _StubTokenizer(raise_on_enable_thinking=True)
    result = _render_row_for_replay(_ROW, tok, max_model_len=100)
    assert result is not None  # fallback path returned a result


def test_render_add_generation_prompt_is_false():
    """apply_chat_template must always be called with add_generation_prompt=False."""
    calls: list[bool] = []

    class _Recording(_StubTokenizer):
        def apply_chat_template(self, messages, tokenize,
                                add_generation_prompt, **kw):
            calls.append(add_generation_prompt)
            return "x"

    _render_row_for_replay(_ROW, _Recording(), max_model_len=100)
    assert calls, "apply_chat_template was never called"
    assert all(v is False for v in calls), (
        f"add_generation_prompt must be False; got {calls}")


def test_render_generate_row_works_identically():
    """completion_source=teacher_generated rows use same rendering path."""
    row = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
        "subset": "mot_math",
        "completion_source": "teacher_generated",
    }
    result = _render_row_for_replay(row, _StubTokenizer(), max_model_len=100)
    assert result is not None
    ids, n = result
    assert n == 2  # "QA" = 2 chars


def test_tally_basic():
    rows = [
        {"subset": "mot_code"},
        {"subset": "mot_code"},
        {"subset": "mot_science"},
    ]
    tally = _build_replay_subset_tally(rows, [100, 200, 50])
    assert tally["mot_code"]["n_rows"] == 2
    assert tally["mot_code"]["n_tokens"] == 300
    assert tally["mot_science"]["n_rows"] == 1
    assert tally["mot_science"]["n_tokens"] == 50


def test_tally_fallback_domain():
    rows = [{"domain": "math"}]
    tally = _build_replay_subset_tally(rows, [42])
    assert "math" in tally
    assert tally["math"]["n_tokens"] == 42


def test_tally_unknown_fallback():
    rows = [{}]
    tally = _build_replay_subset_tally(rows, [5])
    assert "unknown" in tally


def test_tally_empty():
    assert _build_replay_subset_tally([], []) == {}


def test_assert_code_science_passes():
    tally = {
        "mot_code": {"n_rows": 5, "n_tokens": 1000},
        "mot_science": {"n_rows": 3, "n_tokens": 500},
        "math": {"n_rows": 10, "n_tokens": 2000},
    }
    _assert_code_science_nonzero(tally)  # must not raise


def test_assert_fails_no_code():
    tally = {
        "mot_science": {"n_rows": 3, "n_tokens": 500},
        "math": {"n_rows": 10, "n_tokens": 2000},
    }
    with pytest.raises(AssertionError, match="code-subset"):
        _assert_code_science_nonzero(tally)


def test_assert_fails_no_science():
    tally = {
        "mot_code": {"n_rows": 5, "n_tokens": 1000},
        "math": {"n_rows": 10, "n_tokens": 2000},
    }
    with pytest.raises(AssertionError, match="science-subset"):
        _assert_code_science_nonzero(tally)


def test_assert_custom_subsets():
    tally = {
        "my_code": {"n_rows": 1, "n_tokens": 100},
        "my_sci": {"n_rows": 1, "n_tokens": 50},
    }
    _assert_code_science_nonzero(
        tally,
        code_subsets=frozenset({"my_code"}),
        science_subsets=frozenset({"my_sci"}),
    )
