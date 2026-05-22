"""Tests for the orchestrator-facing stage contract — ``Stage`` Protocol.

Covers structural conformance to the ``@runtime_checkable`` Protocol: a plain
class with ``stage_id`` plus the ``is_enabled`` and ``run`` methods passes
``isinstance``; omitting the attribute or either method fails it. Also verifies
that both ways a real stage will satisfy the contract — a base-class-derived
subclass and a factory-built instance — conform.
"""

from __future__ import annotations

import sys

import pytest

from moe_compress.pipeline.stage import Stage


class _FakeStage:
    """Structurally-complete stage — enough to satisfy ``Stage`` structurally."""

    stage_id = "1"

    def is_enabled(self, config: dict) -> bool:
        return True

    def run(self, ctx) -> None:
        return None


def test_fake_stage_satisfies_protocol():
    assert isinstance(_FakeStage(), Stage)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="runtime_checkable attribute-presence checks require Python 3.12+",
)
def test_missing_stage_id_fails_protocol():
    class _NoStageId:
        # 'stage_id' deliberately omitted entirely

        def is_enabled(self, config: dict) -> bool:
            return True

        def run(self, ctx) -> None:
            return None

    assert isinstance(_NoStageId(), Stage) is False


def test_missing_run_fails_protocol():
    class _NoRun:
        stage_id = "x"

        def is_enabled(self, config: dict) -> bool:
            return True

        # 'run' deliberately omitted entirely

    assert isinstance(_NoRun(), Stage) is False


def test_missing_is_enabled_fails_protocol():
    class _NoIsEnabled:
        stage_id = "x"

        # 'is_enabled' deliberately omitted entirely

        def run(self, ctx) -> None:
            return None

    assert isinstance(_NoIsEnabled(), Stage) is False


class _BaseStageLike:
    """Optional-base-style stage: metadata default + no-op core methods."""

    stage_id: str = ""

    def is_enabled(self, config: dict) -> bool:
        return True

    def run(self, ctx) -> None:
        return None


class _Sub(_BaseStageLike):
    stage_id = "2.5"


def test_baseplugin_like_stage_satisfies_protocol():
    assert isinstance(_Sub(), Stage)


def make_fake_stage(stage_id: str) -> Stage:
    """Factory returning a structurally-complete ``Stage`` instance."""

    class _FactoryStage:
        def __init__(self, stage_id: str) -> None:
            self.stage_id = stage_id

        def is_enabled(self, config: dict) -> bool:
            return True

        def run(self, ctx) -> None:
            return None

    return _FactoryStage(stage_id)


def test_factory_built_stage_satisfies_protocol():
    # Deliberately exercises the instance-attribute (factory) pattern:
    # stage_id is set in __init__, mirroring Router-KD's make_router_kd_stage.
    assert isinstance(make_fake_stage("6alt"), Stage)
