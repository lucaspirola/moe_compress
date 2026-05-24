"""Tests for ``PluginRegistry`` — the universal, immutable plugin registry.

Covers construction (empty, order-preservation, immutability, generator
input), the duplicate-name guard, the dunders (``__len__`` / ``__iter__`` /
``names``), :meth:`enabled` config-driven filtering, :meth:`provides`
first-occurrence union over the enabled subset, and the ``dispatch_first``
slot helper — first non-None wins, hook-absent / non-callable skipping,
argument forwarding, short-circuiting, and the falsy-non-None guard against a
truthiness regression.
"""

from __future__ import annotations

import pytest

from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.pipeline.plugin import PipelinePlugin


# --------------------------------------------------------------------------
# Structural fake plugins satisfying ``PipelinePlugin``.
# --------------------------------------------------------------------------
class _FakePlugin:
    """Configurable structural plugin: all six metadata attrs + core hooks.

    ``name`` / ``enabled`` / ``provides`` are constructor-configurable.
    """

    paper = "Doe et al. 2025"
    config_key = "fake.enabled"
    reads = ()
    writes = ()

    def __init__(self, name: str, *, enabled: bool = True, provides=()):
        self.name = name
        self._enabled = enabled
        self.provides = tuple(provides)

    def is_enabled(self, config: dict) -> bool:
        return self._enabled

    def contribute_artifact(self, ctx) -> dict:
        return {}


class _ConfigDrivenPlugin(_FakePlugin):
    """Plugin whose ``is_enabled`` reads its own key out of ``config``."""

    def is_enabled(self, config: dict) -> bool:
        return config.get(self.name, True)


class _SlotPlugin(_FakePlugin):
    """Plugin exposing a named hook returning a configurable value.

    The hook value may be ``None`` (defer) or any other value (including a
    falsy non-None one). A call counter records hook invocations so tests can
    assert short-circuit behaviour.
    """

    def __init__(self, name: str, *, hook_name: str, hook_value, **kw):
        super().__init__(name, **kw)
        self._hook_name = hook_name
        self._hook_value = hook_value
        self.calls = 0
        setattr(self, hook_name, self._hook)

    def _hook(self, *args, **kwargs):
        self.calls += 1
        self._last_args = args
        self._last_kwargs = kwargs
        return self._hook_value


class _NoHookPlugin(_FakePlugin):
    """Plugin with no slot hook at all (dispatch_first must skip it)."""


class _NonCallableHookPlugin(_FakePlugin):
    """Plugin with a NON-callable attribute colliding with a hook name."""

    def __init__(self, name: str, *, hook_name: str, **kw):
        super().__init__(name, **kw)
        setattr(self, hook_name, "not-callable")  # plain string, not a method


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def test_empty_registry_constructs():
    r = PluginRegistry([])
    assert len(r) == 0
    assert list(r) == []


def test_construction_preserves_order():
    a, b, c = _FakePlugin("a"), _FakePlugin("b"), _FakePlugin("c")
    r = PluginRegistry([a, b, c])
    assert list(r) == [a, b, c]
    assert r.names() == ("a", "b", "c")


def test_registry_is_immutable_no_register_method():
    r = PluginRegistry([_FakePlugin("a")])
    assert not hasattr(r, "register")
    assert not hasattr(r, "active")
    assert not hasattr(r, "classes")


def test_mutating_input_list_after_construction_does_not_change_registry():
    plugins = [_FakePlugin("a"), _FakePlugin("b")]
    r = PluginRegistry(plugins)
    plugins.append(_FakePlugin("c"))
    plugins.clear()
    assert r.names() == ("a", "b")
    assert len(r) == 2


def test_construction_accepts_a_generator():
    gen = (_FakePlugin(n) for n in ("a", "b", "c"))
    r = PluginRegistry(gen)
    assert r.names() == ("a", "b", "c")
    assert len(r) == 3


def test_construction_stores_plugins_as_tuple():
    r = PluginRegistry([_FakePlugin("a")])
    assert isinstance(r._plugins, tuple)


# --------------------------------------------------------------------------
# Duplicate-name guard
# --------------------------------------------------------------------------
def test_duplicate_name_raises_value_error():
    with pytest.raises(ValueError):
        PluginRegistry([_FakePlugin("dup"), _FakePlugin("dup")])


def test_duplicate_name_message_contains_the_name():
    with pytest.raises(ValueError, match="dup"):
        PluginRegistry([_FakePlugin("dup"), _FakePlugin("dup")])


def test_duplicate_name_message_mentions_registry():
    with pytest.raises(ValueError, match="Duplicate plugin names"):
        PluginRegistry([_FakePlugin("x"), _FakePlugin("x")])


def test_distinct_names_construct_fine():
    r = PluginRegistry([_FakePlugin("a"), _FakePlugin("b"), _FakePlugin("c")])
    assert len(r) == 3


def test_duplicate_among_many_still_caught():
    with pytest.raises(ValueError, match="bad"):
        PluginRegistry(
            [_FakePlugin("a"), _FakePlugin("bad"), _FakePlugin("c"), _FakePlugin("bad")]
        )


# --------------------------------------------------------------------------
# Dunders
# --------------------------------------------------------------------------
def test_len_reports_plugin_count():
    assert len(PluginRegistry([])) == 0
    assert len(PluginRegistry([_FakePlugin("a")])) == 1
    assert len(PluginRegistry([_FakePlugin("a"), _FakePlugin("b")])) == 2


def test_iter_yields_plugins_in_order_by_identity():
    a, b, c = _FakePlugin("a"), _FakePlugin("b"), _FakePlugin("c")
    r = PluginRegistry([a, b, c])
    iterated = list(r)
    assert iterated[0] is a
    assert iterated[1] is b
    assert iterated[2] is c


def test_iter_is_repeatable():
    r = PluginRegistry([_FakePlugin("a"), _FakePlugin("b")])
    first = list(r)
    second = list(r)
    assert first == second
    assert len(first) == 2


def test_names_returns_a_tuple():
    r = PluginRegistry([_FakePlugin("a")])
    assert isinstance(r.names(), tuple)


# --------------------------------------------------------------------------
# enabled()
# --------------------------------------------------------------------------
def test_enabled_filters_by_is_enabled():
    on = _FakePlugin("on", enabled=True)
    off = _FakePlugin("off", enabled=False)
    r = PluginRegistry([on, off])
    assert r.enabled({}) == (on,)


def test_enabled_preserves_order_with_middle_plugin_disabled():
    a = _FakePlugin("a", enabled=True)
    b = _FakePlugin("b", enabled=False)
    c = _FakePlugin("c", enabled=True)
    r = PluginRegistry([a, b, c])
    assert r.enabled({}) == (a, c)


def test_enabled_all_true_returns_all():
    a, b = _FakePlugin("a"), _FakePlugin("b")
    r = PluginRegistry([a, b])
    assert r.enabled({}) == (a, b)


def test_enabled_all_false_returns_empty_tuple():
    r = PluginRegistry([_FakePlugin("a", enabled=False), _FakePlugin("b", enabled=False)])
    assert r.enabled({}) == ()


def test_enabled_returns_a_tuple():
    r = PluginRegistry([_FakePlugin("a")])
    assert isinstance(r.enabled({}), tuple)


def test_enabled_honors_the_config_argument():
    p = _ConfigDrivenPlugin("toggle")
    r = PluginRegistry([p])
    assert r.enabled({"toggle": True}) == (p,)
    assert r.enabled({"toggle": False}) == ()


# --------------------------------------------------------------------------
# provides()
# --------------------------------------------------------------------------
def test_provides_unions_enabled_plugins():
    p1 = _FakePlugin("p1", provides=("a", "b"))
    p2 = _FakePlugin("p2", provides=("c",))
    r = PluginRegistry([p1, p2])
    assert r.provides({}) == ("a", "b", "c")


def test_provides_first_occurrence_order_with_overlap():
    p1 = _FakePlugin("p1", provides=("a", "b"))
    p2 = _FakePlugin("p2", provides=("b", "c"))
    p3 = _FakePlugin("p3", provides=("a", "d"))
    r = PluginRegistry([p1, p2, p3])
    assert r.provides({}) == ("a", "b", "c", "d")


def test_provides_excludes_disabled_plugins():
    on = _FakePlugin("on", enabled=True, provides=("a",))
    off = _FakePlugin("off", enabled=False, provides=("b",))
    r = PluginRegistry([on, off])
    assert r.provides({}) == ("a",)


def test_provides_empty_when_no_plugin_provides_anything():
    r = PluginRegistry([_FakePlugin("a"), _FakePlugin("b")])
    assert r.provides({}) == ()


def test_provides_empty_for_empty_registry():
    r = PluginRegistry([])
    assert r.provides({}) == ()


def test_provides_returns_a_tuple():
    p = _FakePlugin("p", provides=("a",))
    r = PluginRegistry([p])
    assert isinstance(r.provides({}), tuple)


def test_provides_respects_config_driven_enablement():
    p1 = _ConfigDrivenPlugin("p1", provides=("a",))
    p2 = _ConfigDrivenPlugin("p2", provides=("b",))
    r = PluginRegistry([p1, p2])
    assert r.provides({"p2": False}) == ("a",)
    assert r.provides({"p1": False}) == ("b",)


# --------------------------------------------------------------------------
# dispatch_first()
# --------------------------------------------------------------------------
def test_dispatch_first_returns_first_non_none():
    a = _SlotPlugin("a", hook_name="solve", hook_value=None)
    b = _SlotPlugin("b", hook_name="solve", hook_value="winner")
    c = _SlotPlugin("c", hook_name="solve", hook_value="late")
    assert PluginRegistry.dispatch_first([a, b, c], "solve") == "winner"


def test_dispatch_first_all_defer_returns_none():
    a = _SlotPlugin("a", hook_name="solve", hook_value=None)
    b = _SlotPlugin("b", hook_name="solve", hook_value=None)
    assert PluginRegistry.dispatch_first([a, b], "solve") is None


def test_dispatch_first_empty_list_returns_none():
    assert PluginRegistry.dispatch_first([], "solve") is None


def test_dispatch_first_skips_plugin_lacking_the_hook():
    no_hook = _NoHookPlugin("no_hook")
    winner = _SlotPlugin("winner", hook_name="solve", hook_value="ok")
    # No AttributeError despite no_hook lacking 'solve'.
    assert PluginRegistry.dispatch_first([no_hook, winner], "solve") == "ok"


def test_dispatch_first_all_lack_hook_returns_none():
    a, b = _NoHookPlugin("a"), _NoHookPlugin("b")
    assert PluginRegistry.dispatch_first([a, b], "solve") is None


def test_dispatch_first_skips_non_callable_colliding_attribute():
    bad = _NonCallableHookPlugin("bad", hook_name="solve")
    winner = _SlotPlugin("winner", hook_name="solve", hook_value="ok")
    # No TypeError despite bad.solve being a non-callable string.
    assert PluginRegistry.dispatch_first([bad, winner], "solve") == "ok"


def test_dispatch_first_passes_positional_and_keyword_args():
    p = _SlotPlugin("p", hook_name="solve", hook_value="done")
    PluginRegistry.dispatch_first([p], "solve", 1, 2, key="value")
    assert p._last_args == (1, 2)
    assert p._last_kwargs == {"key": "value"}


def test_dispatch_first_stops_at_first_winner():
    winner = _SlotPlugin("winner", hook_name="solve", hook_value="first")
    later = _SlotPlugin("later", hook_name="solve", hook_value="second")
    result = PluginRegistry.dispatch_first([winner, later], "solve")
    assert result == "first"
    assert winner.calls == 1
    assert later.calls == 0  # never invoked — short-circuited


def test_dispatch_first_invokes_deferring_plugins_before_winner():
    deferring = _SlotPlugin("deferring", hook_name="solve", hook_value=None)
    winner = _SlotPlugin("winner", hook_name="solve", hook_value="ok")
    PluginRegistry.dispatch_first([deferring, winner], "solve")
    assert deferring.calls == 1
    assert winner.calls == 1


def test_dispatch_first_is_a_staticmethod():
    # Callable straight off the class with no instance.
    p = _SlotPlugin("p", hook_name="solve", hook_value="x")
    assert PluginRegistry.dispatch_first([p], "solve") == "x"


def test_dispatch_first_typical_usage_with_enabled():
    on = _SlotPlugin("on", hook_name="solve", hook_value="hit", enabled=True)
    off = _SlotPlugin("off", hook_name="solve", hook_value="skipped", enabled=False)
    r = PluginRegistry([off, on])
    # off is filtered out by enabled(), so 'on' wins even though it is later.
    assert PluginRegistry.dispatch_first(r.enabled({}), "solve") == "hit"


@pytest.mark.parametrize("falsy", [0, False, "", []])
def test_dispatch_first_returns_falsy_non_none_value(falsy):
    """A hook returning a falsy-but-non-None value must count as a winner —
    guards against an ``if result:`` truthiness regression."""
    winner = _SlotPlugin("winner", hook_name="solve", hook_value=falsy)
    later = _SlotPlugin("later", hook_name="solve", hook_value="should-not-win")
    result = PluginRegistry.dispatch_first([winner, later], "solve")
    assert result == falsy
    assert later.calls == 0  # winner already produced a non-None result


def test_fake_plugins_satisfy_protocol():
    """Sanity check: the structural fakes really do satisfy ``PipelinePlugin``."""
    assert isinstance(_FakePlugin("a"), PipelinePlugin)
    assert isinstance(_ConfigDrivenPlugin("b"), PipelinePlugin)
    assert isinstance(_SlotPlugin("c", hook_name="h", hook_value=1), PipelinePlugin)
    assert isinstance(_NoHookPlugin("d"), PipelinePlugin)
    assert isinstance(_NonCallableHookPlugin("e", hook_name="h"), PipelinePlugin)
