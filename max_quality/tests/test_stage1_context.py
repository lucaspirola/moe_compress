"""Unit tests for ``moe_compress.stage1.context``.

``Stage1Context`` is a one-line :class:`PipelineContext` subclass; full
coverage for the base class behaviour lives in
``test_pipeline_context.py``. These three tests confirm the subclass
inherits the contract without shadowing anything.
"""

from __future__ import annotations

import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.context import Stage1Context


def test_stage1_context_is_pipeline_context():
    """The subclass must be a :class:`PipelineContext`."""
    assert isinstance(Stage1Context(), PipelineContext)


def test_stage1_context_inherits_set_get():
    """``set`` / ``get`` are inherited verbatim from the base class."""
    ctx = Stage1Context()
    ctx.set("foo", 42)
    assert ctx.get("foo") == 42


def test_stage1_context_inherits_set_once_guard():
    """The inherited set-once guard must still fire through the subclass."""
    ctx = Stage1Context()
    ctx.set("foo", 1)
    with pytest.raises(KeyError):
        ctx.set("foo", 2)
    # ``overwrite=True`` is honoured.
    ctx.set("foo", 2, overwrite=True)
    assert ctx.get("foo") == 2
