"""Tests for the universal plugin framework — ``PipelinePlugin`` + ``BasePlugin``.

Covers structural conformance to the ``@runtime_checkable`` Protocol (a plain
class with the right attributes/methods passes ``isinstance``; omitting any
attribute or method fails it), and the ``BasePlugin`` convenience base —
metadata defaults, no-op core hooks, and override behaviour.
"""

from __future__ import annotations

import sys

import pytest

from moe_compress.pipeline.plugin import PipelinePlugin, BasePlugin


class _FakePlugin:
    """Structurally-complete plugin that does NOT subclass ``BasePlugin``.

    Declares all six metadata attributes as class attributes and both
    universal-core methods — enough to satisfy ``PipelinePlugin`` structurally.
    """

    name = "fake"
    paper = "Doe et al. 2025"
    config_key = "fake.enabled"
    reads = ("model",)
    writes = ("fake_artifact",)
    provides = ("fake_accumulator",)

    def is_enabled(self, config: dict) -> bool:
        return True

    def contribute_artifact(self, ctx) -> dict:
        return {}


def test_fake_plugin_satisfies_protocol():
    assert isinstance(_FakePlugin(), PipelinePlugin)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="runtime_checkable attribute-presence checks require Python 3.12+",
)
def test_missing_attribute_fails_protocol():
    class _NoProvides:
        name = "x"
        paper = "x"
        config_key = "x"
        reads = ()
        writes = ()
        # 'provides' deliberately omitted entirely

        def is_enabled(self, config: dict) -> bool:
            return True

        def contribute_artifact(self, ctx) -> dict:
            return {}

    assert isinstance(_NoProvides(), PipelinePlugin) is False


def test_missing_method_fails_protocol():
    class _NoContribute:
        name = "x"
        paper = "x"
        config_key = "x"
        reads = ()
        writes = ()
        provides = ()

        def is_enabled(self, config: dict) -> bool:
            return True

        # 'contribute_artifact' deliberately omitted entirely

    assert isinstance(_NoContribute(), PipelinePlugin) is False


def test_missing_is_enabled_fails_protocol():
    class _NoIsEnabled:
        name = "x"
        paper = "x"
        config_key = "x"
        reads = ()
        writes = ()
        provides = ()

        # 'is_enabled' deliberately omitted entirely

        def contribute_artifact(self, ctx) -> dict:
            return {}

    assert isinstance(_NoIsEnabled(), PipelinePlugin) is False


def test_baseplugin_subclass_satisfies_protocol():
    class _Sub(BasePlugin):
        name = "sub"

    assert isinstance(_Sub(), PipelinePlugin)


def test_baseplugin_instance_satisfies_protocol():
    assert isinstance(BasePlugin(), PipelinePlugin)


def test_baseplugin_metadata_defaults():
    p = BasePlugin()
    assert p.name == ""
    assert p.paper == ""
    assert p.config_key == ""
    assert p.reads == ()
    assert p.writes == ()
    assert p.provides == ()
    assert isinstance(p.reads, tuple)
    assert isinstance(p.writes, tuple)
    assert isinstance(p.provides, tuple)


def test_baseplugin_is_enabled_default_true():
    p = BasePlugin()
    assert p.is_enabled({}) is True
    assert p.is_enabled({"x": 1}) is True


def test_baseplugin_contribute_artifact_default_empty():
    p = BasePlugin()
    a = p.contribute_artifact(None)
    b = p.contribute_artifact(None)
    assert a == {}
    assert b == {}
    # Each call must return a distinct, fresh object.
    assert a is not b
    # Mutating one return value must not affect the next.
    a["mutated"] = True
    assert p.contribute_artifact(None) == {}


def test_baseplugin_subclass_can_override():
    class _Override(BasePlugin):
        name = "override"

        def is_enabled(self, config: dict) -> bool:
            return False

        def contribute_artifact(self, ctx) -> dict:
            return {"k": 1}

    p = _Override()
    assert isinstance(p, PipelinePlugin)
    assert p.is_enabled({}) is False
    assert p.contribute_artifact(None) == {"k": 1}
