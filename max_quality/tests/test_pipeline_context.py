"""Tests for :class:`moe_compress.pipeline.context.PipelineContext`.

Covers strict get/set semantics, set-once guarding, ``None`` as a legal value,
``drop``, local enumeration (``keys`` / ``in`` / iteration), and parent/child
scoping — including the intentional asymmetry where ``get`` / ``has`` chain
through parents but ``keys`` / ``in`` / iteration / ``drop`` are local-scope only.
"""

from __future__ import annotations

import pytest

from moe_compress.pipeline.context import PipelineContext


def test_get_after_set_returns_value():
    ctx = PipelineContext()
    ctx.set("foo", 42)
    assert ctx.get("foo") == 42


def test_set_twice_raises_without_overwrite():
    ctx = PipelineContext()
    ctx.set("foo", 1)
    with pytest.raises(KeyError):
        ctx.set("foo", 2)


def test_set_with_overwrite_replaces():
    ctx = PipelineContext()
    ctx.set("foo", 1)
    ctx.set("foo", 2, overwrite=True)
    assert ctx.get("foo") == 2


def test_get_missing_raises_informative_keyerror():
    ctx = PipelineContext()
    with pytest.raises(KeyError) as excinfo:
        ctx.get("missing")
    message = str(excinfo.value)
    assert "missing" in message
    assert "[]" in message


def test_get_missing_message_lists_written_slots():
    ctx = PipelineContext()
    ctx.set("a", 1)
    ctx.set("b", 2)
    with pytest.raises(KeyError) as excinfo:
        ctx.get("missing")
    message = str(excinfo.value)
    assert "a" in message
    assert "b" in message


def test_has_true_for_written_false_otherwise():
    ctx = PipelineContext()
    ctx.set("foo", 1)
    assert ctx.has("foo") is True
    assert ctx.has("bar") is False


def test_supports_none_values():
    ctx = PipelineContext()
    ctx.set("foo", None)
    assert ctx.has("foo") is True
    assert ctx.get("foo") is None


def test_drop_removes_slot():
    ctx = PipelineContext()
    ctx.set("foo", 1)
    ctx.drop("foo")
    assert ctx.has("foo") is False


def test_drop_missing_raises_keyerror():
    ctx = PipelineContext()
    with pytest.raises(KeyError):
        ctx.drop("missing")


def test_keys_iter_contains_local_insertion_order():
    ctx = PipelineContext()
    ctx.set("a", 1)
    ctx.set("b", 2)
    ctx.set("c", 3)
    assert ctx.keys() == ("a", "b", "c")
    assert list(iter(ctx)) == ["a", "b", "c"]
    assert "a" in ctx
    assert "z" not in ctx


def test_child_reads_fall_through_to_parent():
    parent = PipelineContext()
    parent.set("x", 1)
    child = parent.child()
    assert child.get("x") == 1


def test_child_set_is_local_parent_unaffected():
    parent = PipelineContext()
    child = parent.child()
    child.set("y", 9)
    assert parent.has("y") is False
    assert "y" not in parent
    assert "y" not in parent.keys()


def test_child_can_shadow_parent_slot():
    parent = PipelineContext()
    parent.set("x", 1)
    child = parent.child()
    child.set("x", 2)  # must NOT raise — local-only set-once check
    assert child.get("x") == 2
    assert parent.get("x") == 1


def test_child_set_once_within_child_scope():
    parent = PipelineContext()
    child = parent.child()
    child.set("y", 1)
    with pytest.raises(KeyError):
        child.set("y", 2)


def test_drop_on_child_cannot_remove_parent_slot():
    parent = PipelineContext()
    parent.set("x", 1)
    child = parent.child()
    assert child.has("x") is True
    with pytest.raises(KeyError):
        child.drop("x")
    assert parent.has("x") is True


def test_drop_local_child_slot_removes_only_child():
    parent = PipelineContext()
    parent.set("x", 1)
    child = parent.child()
    child.set("x", 2)
    child.drop("x")
    assert child.get("x") == 1  # falls through to parent again
    assert parent.get("x") == 1


def test_has_chains_but_contains_is_local():
    parent = PipelineContext()
    parent.set("x", 1)
    child = parent.child()
    assert child.has("x") is True
    assert "x" not in child
    assert child.keys() == ()


def test_nested_grandchild_fall_through():
    parent = PipelineContext()
    parent.set("x", 1)
    child = parent.child()
    grandchild = child.child()
    assert grandchild.get("x") == 1
    assert grandchild.has("x") is True


def test_child_get_missing_raises():
    parent = PipelineContext()
    child = parent.child()
    with pytest.raises(KeyError):
        child.get("missing")


def test_child_get_missing_error_lists_child_slots_not_parent():
    parent = PipelineContext()
    parent.set("parent_slot", 1)
    child = parent.child()
    child.set("child_slot", 99)
    with pytest.raises(KeyError) as excinfo:
        child.get("missing")
    message = str(excinfo.value)
    assert "child_slot" in message
    assert "parent_slot" not in message


def test_child_factory_returns_distinct_instance():
    parent = PipelineContext()
    child = parent.child()
    assert child is not parent
    assert isinstance(child, PipelineContext)
