"""Unit tests for ``moe_compress.stage1._framework.plugin``.

Verifies the ``StagePlugin`` Protocol shape and ``PluginRegistry`` behaviour
without depending on any Stage-1 fixtures (no ``tiny_model`` / ``tiny_config``).
"""

from __future__ import annotations

import pytest

from moe_compress.stage1._framework.plugin import PluginRegistry, StagePlugin


class _FakePlugin:
    """Minimal stand-alone plugin used across tests.

    Construct with ``name`` / ``enabled`` / ``accumulators`` to assemble
    varied registries — kept here (not in conftest) because these tests
    deliberately do not share fixtures with the Stage 1 suite.
    """

    paper = "fake-2026"
    config_key = "stage1_grape.fake"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("foo",)

    def __init__(
        self,
        name: str = "fake",
        *,
        enabled: bool = True,
        accumulators: tuple[str, ...] = ("downproj_max",),
    ) -> None:
        self.name = name
        self.accumulators = accumulators
        self._enabled = enabled

    def is_enabled(self, config: dict) -> bool:
        return self._enabled

    def run(self, ctx) -> None:  # pragma: no cover - protocol only
        return None

    def contribute_artifact(self, ctx) -> dict:  # pragma: no cover - protocol only
        return {}


def test_fake_plugin_satisfies_stage_plugin_protocol():
    """``runtime_checkable`` Protocol must accept a structurally-matching class."""
    plugin = _FakePlugin()
    assert isinstance(plugin, StagePlugin)


def test_registry_preserves_order():
    a = _FakePlugin(name="a_name")
    b = _FakePlugin(name="b_name")
    c = _FakePlugin(name="c_name")
    reg = PluginRegistry([a, b, c])
    assert reg.names() == ("a_name", "b_name", "c_name")
    assert tuple(reg) == (a, b, c)
    assert len(reg) == 3


def test_registry_rejects_duplicate_names():
    a = _FakePlugin(name="dup")
    b = _FakePlugin(name="dup")
    with pytest.raises(ValueError) as exc:
        PluginRegistry([a, b])
    assert "dup" in str(exc.value)


def test_registry_enabled_filters_by_is_enabled():
    on = _FakePlugin(name="on", enabled=True)
    off = _FakePlugin(name="off", enabled=False)
    reg = PluginRegistry([on, off])
    assert reg.enabled({}) == (on,)


def test_registry_required_accumulators_first_occurrence_order():
    a = _FakePlugin(name="a", accumulators=("downproj_max",))
    b = _FakePlugin(name="b", accumulators=("downproj_max", "output_reservoir"))
    reg = PluginRegistry([a, b])
    assert reg.required_accumulators({}) == ("downproj_max", "output_reservoir")


def test_registry_required_accumulators_skips_disabled():
    a = _FakePlugin(name="a", accumulators=("downproj_max",), enabled=True)
    b = _FakePlugin(name="b", accumulators=("output_reservoir",), enabled=False)
    reg = PluginRegistry([a, b])
    assert reg.required_accumulators({}) == ("downproj_max",)
