"""Tests for the shared ``tools/eval_harness`` module (S6-4).

``tools/eval_harness`` is the leaf utility that holds the batched-generation +
chat-format primitives reused by the Stage 6 generative-eval plugins
(``stage6/plugins/{humaneval,math500}.py``) and by ``stage6alt``. This file
follows the ``tools/`` test convention (``test_tools_phase_walker.py`` etc.):

* the module imports and exposes its ``__all__`` symbols;
* ``_generate_batched`` re-asserts the eager-attention pin (a model whose
  ``config._attn_implementation`` is non-eager raises ``RuntimeError``);
* ``_chat_format_prompts`` degrades to the raw prompt when the tokenizer has no
  usable ``apply_chat_template``;
* ``_extract_code_from_chat_response`` pulls Python out of a fenced /
  think-blocked chat reply;
* ``test_no_stage_import_at_any_scope`` — an AST walk proving the module
  imports NO stage module and NO ``pipeline`` module (the leaf-utility
  contract in the module docstring).

All tests are CPU-only and run no real generation.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.tools import eval_harness  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"tools/eval_harness imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


class _Config:
    """Minimal stand-in for a model ``.config`` carrying the attn-impl flag."""

    def __init__(self, attn_impl: str) -> None:
        self._attn_implementation = attn_impl


class _StubModel:
    """Model stub: only the ``.config`` attribute ``_generate_batched`` reads
    before its eager-attention guard. ``generate()`` is never reached in these
    tests — the guard raises first when attn-impl is non-eager."""

    def __init__(self, attn_impl: str) -> None:
        self.config = _Config(attn_impl)


class _PlainTokenizer:
    """Tokenizer with NO ``apply_chat_template`` — ``_chat_format_prompts``
    must degrade to the raw prompt for every entry."""

    def apply_chat_template(self, *_args, **_kwargs):  # pragma: no cover
        raise AttributeError("no chat template on this tokenizer")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eval_harness_module_imports():
    """All ``__all__`` symbols are importable from the module."""
    from moe_compress.tools.eval_harness import (
        _chat_format_prompts,
        _extract_code_from_chat_response,
        _generate_batched,
        _PY_FENCE_RE,
        _stage6_enable_thinking,
        _THINK_BLOCK_RE,
        _TRAILING_PROSE_RE,
    )

    assert callable(_generate_batched)
    assert callable(_stage6_enable_thinking)
    assert callable(_chat_format_prompts)
    assert callable(_extract_code_from_chat_response)
    # The three regexes are compiled pattern objects.
    for rx in (_THINK_BLOCK_RE, _PY_FENCE_RE, _TRAILING_PROSE_RE):
        assert hasattr(rx, "search")


def test_all_lists_public_symbols():
    """``__all__`` lists exactly the seven relocated Pattern-A symbols."""
    assert set(eval_harness.__all__) == {
        "_generate_batched",
        "_stage6_enable_thinking",
        "_chat_format_prompts",
        "_THINK_BLOCK_RE",
        "_PY_FENCE_RE",
        "_TRAILING_PROSE_RE",
        "_extract_code_from_chat_response",
    }


def test_generate_batched_eager_attn_assert():
    """``_generate_batched`` raises ``RuntimeError`` under non-eager attention.

    Spec §9 #3/#4 binds the bs=1 ↔ batched argmax-identity claim to eager
    attention; the function re-asserts the pin before doing any work.
    """
    from moe_compress.tools.eval_harness import _generate_batched

    model = _StubModel(attn_impl="sdpa")  # non-eager → must raise
    with pytest.raises(RuntimeError, match="_attn_implementation"):
        _generate_batched(model, object(), ["hello"], max_new=4, device=None)


def test_chat_format_prompts_fallback_without_template():
    """``_chat_format_prompts`` returns the raw prompt when no chat template.

    A tokenizer whose ``apply_chat_template`` raises must not crash the eval —
    each entry degrades to the raw prompt string.
    """
    from moe_compress.tools.eval_harness import _chat_format_prompts

    raw = ["solve x+1", "print hello"]
    out = _chat_format_prompts(_PlainTokenizer(), raw, system="be helpful")
    assert out == raw


def test_extract_code_from_fenced_response():
    """``_extract_code_from_chat_response`` pulls a ```python fenced block."""
    from moe_compress.tools.eval_harness import _extract_code_from_chat_response

    reply = (
        "Here is the solution:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
        "That completes it."
    )
    code = _extract_code_from_chat_response(reply, "add")
    assert "def add(a, b):" in code
    assert "return a + b" in code
    # Trailing prose outside the fence is not included.
    assert "That completes it" not in code


def test_extract_code_strips_think_block():
    """A <think>...</think> block is stripped before code extraction."""
    from moe_compress.tools.eval_harness import _extract_code_from_chat_response

    reply = (
        "<think>I should add the two numbers. Let me write a fenced block "
        "def add( ... wait no.</think>\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```"
    )
    code = _extract_code_from_chat_response(reply, "add")
    assert "def add(a, b):" in code
    assert "return a + b" in code
    assert "wait no" not in code  # reasoning text removed


def test_extract_code_no_code_returns_empty():
    """When no code can be located the helper returns the empty string."""
    from moe_compress.tools.eval_harness import _extract_code_from_chat_response

    assert _extract_code_from_chat_response("just prose, no code here", "add") == ""


def test_no_stage_import_at_any_scope():
    """The module imports NO stage module and NO ``pipeline`` module.

    ``tools/eval_harness`` is a leaf utility: its whole purpose is to be
    reusable by stage6 *and* stage6alt without an import cycle, so it must
    import only stdlib + ``torch``. Parse the source with ``ast`` and walk the
    FULL tree so a function-local forbidden import cannot slip past. For
    ``ImportFrom`` both ``node.module`` and each imported name (plus its
    ``asname``) are checked.
    """
    src = Path(eval_harness.__file__).read_text()
    tree = ast.parse(src)
    forbidden = ("stage6_validate", "stage6", "orchestrator", "pipeline")
    for node in ast.walk(tree):  # any nesting level, not just module-top
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(
                    f in alias.name or f in (alias.asname or "")
                    for f in forbidden
                ), f"forbidden import at any scope: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert not any(f in mod_name for f in forbidden), (
                f"forbidden import-from at any scope: {mod_name}"
            )
            for alias in node.names:
                assert not any(
                    f in alias.name or f in (alias.asname or "")
                    for f in forbidden
                ), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )
