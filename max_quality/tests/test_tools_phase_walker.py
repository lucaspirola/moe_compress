"""Tests for the reflective phase scheduler — ``walk_phases`` + ``loop_over``.

Covers phase-major / plugin-minor ordering, reflective hook discovery (present
hooks called, missing hooks skipped, non-callable colliding attributes skipped),
in-place ``ctx`` mutation, and the per-item child-scope semantics of
``loop_over`` (one distinct child per item, ``item_key`` binding, parent
fall-through reads, local-only writes, harvesting per-item results).
"""

from __future__ import annotations

import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.tools.phase_walker import walk_phases, loop_over


# ---------------------------------------------------------------------------
# Configurable structural fake plugin
# ---------------------------------------------------------------------------


class _FakePlugin:
    """Structurally-complete plugin with reflectively-installed phase hooks.

    Declares all six metadata attributes to satisfy ``PipelinePlugin``
    structurally. The constructor installs a callable hook for each name in
    ``hook_phases``; every hook appends ``(plugin_name, phase_name, id(ctx))``
    to the shared ``call_log`` list when invoked.
    """

    paper = "Doe et al. 2025"
    config_key = "fake.enabled"
    reads = ()
    writes = ()
    provides = ()

    def __init__(self, name: str, hook_phases, call_log: list) -> None:
        self.name = name
        self._call_log = call_log
        for phase in hook_phases:
            setattr(self, phase, self._make_hook(phase))

    def _make_hook(self, phase: str):
        def hook(ctx) -> None:
            self._call_log.append((self.name, phase, id(ctx)))
        return hook

    def is_enabled(self, config: dict) -> bool:
        return True

    def contribute_artifact(self, ctx) -> dict:
        return {}


# ---------------------------------------------------------------------------
# walk_phases
# ---------------------------------------------------------------------------


def test_walk_phases_calls_present_hooks():
    log: list = []
    p = _FakePlugin("p1", ["A", "B"], log)
    ctx = PipelineContext()
    walk_phases(["A", "B"], [p], ctx)
    assert [(n, ph) for (n, ph, _) in log] == [("p1", "A"), ("p1", "B")]


def test_walk_phases_skips_missing_hook():
    log: list = []
    p = _FakePlugin("p1", ["A"], log)  # no "B" hook
    ctx = PipelineContext()
    walk_phases(["A", "B"], [p], ctx)  # must not raise AttributeError
    assert [(n, ph) for (n, ph, _) in log] == [("p1", "A")]


def test_walk_phases_skips_non_callable_colliding_attribute():
    log: list = []
    p = _FakePlugin("p1", ["A"], log)
    p.B = 123  # non-callable attribute colliding with phase name "B"
    ctx = PipelineContext()
    walk_phases(["A", "B"], [p], ctx)  # must not raise TypeError
    assert [(n, ph) for (n, ph, _) in log] == [("p1", "A")]


def test_walk_phases_phase_major_order():
    log: list = []
    p1 = _FakePlugin("p1", ["A", "B"], log)
    p2 = _FakePlugin("p2", ["A", "B"], log)
    ctx = PipelineContext()
    walk_phases(["A", "B"], [p1, p2], ctx)
    assert [(n, ph) for (n, ph, _) in log] == [
        ("p1", "A"), ("p2", "A"), ("p1", "B"), ("p2", "B"),
    ]


def test_walk_phases_plugin_order_within_phase():
    log: list = []
    p1 = _FakePlugin("p1", ["A"], log)
    p2 = _FakePlugin("p2", ["A"], log)
    p3 = _FakePlugin("p3", ["A"], log)
    ctx = PipelineContext()
    walk_phases(["A"], [p1, p2, p3], ctx)
    assert [n for (n, _, _) in log] == ["p1", "p2", "p3"]


def test_walk_phases_same_ctx_passed_to_all_hooks():
    log: list = []
    p1 = _FakePlugin("p1", ["A", "B"], log)
    p2 = _FakePlugin("p2", ["A", "B"], log)
    ctx = PipelineContext()
    walk_phases(["A", "B"], [p1, p2], ctx)
    ctx_ids = {cid for (_, _, cid) in log}
    assert ctx_ids == {id(ctx)}


def test_walk_phases_empty_phases_is_noop():
    log: list = []
    p = _FakePlugin("p1", ["A"], log)
    ctx = PipelineContext()
    walk_phases([], [p], ctx)
    assert log == []


def test_walk_phases_empty_plugins_is_noop():
    ctx = PipelineContext()
    walk_phases(["A", "B"], [], ctx)  # must not raise
    assert list(ctx) == []


def test_walk_phases_returns_none():
    log: list = []
    p = _FakePlugin("p1", ["A"], log)
    ctx = PipelineContext()
    assert walk_phases(["A"], [p], ctx) is None


def test_walk_phases_plugin_implementing_only_some_phases():
    log: list = []
    p1 = _FakePlugin("p1", ["A", "B"], log)
    p2 = _FakePlugin("p2", ["B"], log)  # only phase B
    ctx = PipelineContext()
    walk_phases(["A", "B"], [p1, p2], ctx)
    assert [(n, ph) for (n, ph, _) in log] == [
        ("p1", "A"), ("p1", "B"), ("p2", "B"),
    ]


def test_walk_phases_hook_can_mutate_ctx():
    ctx = PipelineContext()

    class _Writer:
        name = "writer"
        paper = ""
        config_key = ""
        reads = ()
        writes = ("slot",)
        provides = ()

        def is_enabled(self, config: dict) -> bool:
            return True

        def contribute_artifact(self, ctx) -> dict:
            return {}

        def A(self, ctx) -> None:
            ctx.set("slot", 42)

    walk_phases(["A"], [_Writer()], ctx)
    assert ctx.get("slot") == 42


# ---------------------------------------------------------------------------
# loop_over
# ---------------------------------------------------------------------------


def test_loop_over_one_child_per_item():
    log: list = []
    p = _FakePlugin("p1", ["A"], log)
    parent = PipelineContext()
    children = loop_over([10, 20, 30], [p], ["A"], parent, item_key="item")
    assert len(children) == 3


def test_loop_over_returns_distinct_pipeline_contexts():
    parent = PipelineContext()
    children = loop_over([1, 2, 3], [], ["A"], parent, item_key="item")
    for child in children:
        assert isinstance(child, PipelineContext)
        assert child is not parent
    # Each child is distinct from every other.
    assert len({id(c) for c in children}) == 3


def test_loop_over_item_key_set_on_each_child():
    parent = PipelineContext()
    children = loop_over(["a", "b", "c"], [], ["A"], parent, item_key="layer")
    assert [c.get("layer") for c in children] == ["a", "b", "c"]


def test_loop_over_child_reads_fall_through_to_parent():
    parent = PipelineContext()
    parent.set("shared", "from_parent")
    children = loop_over([1, 2], [], ["A"], parent, item_key="item")
    for child in children:
        assert child.get("shared") == "from_parent"


def test_loop_over_child_writes_stay_local():
    parent = PipelineContext()

    class _Writer:
        name = "writer"
        paper = ""
        config_key = ""
        reads = ()
        writes = ("result",)
        provides = ()

        def is_enabled(self, config: dict) -> bool:
            return True

        def contribute_artifact(self, ctx) -> dict:
            return {}

        def A(self, ctx) -> None:
            ctx.set("result", ctx.get("item") * 2)

    children = loop_over([1, 2, 3], [_Writer()], ["A"], parent, item_key="item")
    # Parent must not have picked up any child-local write.
    assert "result" not in parent
    assert parent.has("result") is False
    # Each child carries its own per-item result.
    assert [c.get("result") for c in children] == [2, 4, 6]


def test_loop_over_harvest_per_item_results():
    parent = PipelineContext()

    class _Squarer:
        name = "squarer"
        paper = ""
        config_key = ""
        reads = ("n",)
        writes = ("squared",)
        provides = ()

        def is_enabled(self, config: dict) -> bool:
            return True

        def contribute_artifact(self, ctx) -> dict:
            return {}

        def compute(self, ctx) -> None:
            ctx.set("squared", ctx.get("n") ** 2)

    children = loop_over([2, 3, 4], [_Squarer()], ["compute"], parent, item_key="n")
    assert [c.get("squared") for c in children] == [4, 9, 16]


def test_loop_over_phases_run_in_order_per_item():
    log: list = []
    p = _FakePlugin("p1", ["A", "B"], log)
    parent = PipelineContext()
    loop_over([1, 2], [p], ["A", "B"], parent, item_key="item")
    assert [ph for (_, ph, _) in log] == ["A", "B", "A", "B"]


def test_loop_over_empty_items_returns_empty_list():
    log: list = []
    p = _FakePlugin("p1", ["A"], log)
    parent = PipelineContext()
    children = loop_over([], [p], ["A"], parent, item_key="item")
    assert children == []
    assert log == []


def test_loop_over_item_key_is_keyword_only():
    parent = PipelineContext()
    with pytest.raises(TypeError):
        # item_key passed positionally must raise.
        loop_over([1], [], ["A"], parent, "item")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol sanity check
# ---------------------------------------------------------------------------


def test_fake_plugins_satisfy_protocol():
    log: list = []
    p = _FakePlugin("p1", ["A", "B"], log)
    assert isinstance(p, PipelinePlugin)
