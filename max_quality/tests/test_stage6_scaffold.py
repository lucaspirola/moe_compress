"""Tests for the ``stage6/`` package surface.

S6-1 created the ``stage6/`` package skeleton — an ``__init__`` re-exporting
``run``, an ``orchestrator`` module, a ``context`` re-export shim, and an
(empty) ``plugins`` package. S6-2..S6-7 extracted the Stage 6 validation
algorithm into ``stage6/plugins/``; S6-8 made ``stage6/orchestrator.run`` the
REAL plugin-driven orchestrator and flipped ``stage6_validate.run`` to a thin
shim that delegates to IT (the inverse of the S6-1 scaffold direction).

These tests guard the package surface only. The S6-1 delegation-direction
test (``stage6.orchestrator.run`` → ``stage6_validate.run``) was retired at
S6-8: the orchestrator no longer delegates, it IS the implementation, and
``stage6_validate.run`` is now the delegator. The byte-identity of the new
orchestrator is covered by ``test_stage6_golden_snapshot.py``.
"""
from __future__ import annotations

import inspect

from moe_compress import stage6
from moe_compress import stage6_validate
from moe_compress.pipeline.stage import Stage
from moe_compress.stage6 import STAGE6


def test_stage6_package_imports():
    """The ``stage6`` package and its modules import cleanly."""
    from moe_compress.stage6 import orchestrator, context  # noqa: F401
    import moe_compress.stage6.plugins  # noqa: F401

    assert callable(stage6.run)


def test_stage6_shim_delegates_to_orchestrator(monkeypatch):
    """S6-8: ``stage6_validate.run`` is now the thin shim — it forwards every
    argument unchanged to ``stage6.orchestrator.run`` (4 positionals + the
    kw-only ``device``). Pure unit test, no model."""
    from moe_compress.stage6 import orchestrator

    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    # The shim does a function-local ``from .stage6.orchestrator import run``;
    # patching the ``run`` attribute on the orchestrator module is what it
    # resolves.
    monkeypatch.setattr(orchestrator, "run", _sentinel)

    model = object()
    tokenizer = object()
    config = {"stage6_validate": {}}
    artifacts_dir = object()
    device = object()

    result = stage6_validate.run(
        model, tokenizer, config, artifacts_dir, device=device,
    )

    assert result is sentinel_result
    assert len(calls) == 1, "stage6.orchestrator.run must be called exactly once"

    args, kwargs = calls[0]
    assert args == (model, tokenizer, config, artifacts_dir)
    assert kwargs == {"device": device}


def test_stage6_orchestrator_signature_matches_monolith():
    """``stage6.orchestrator.run`` and ``stage6_validate.run`` have identical
    signatures — parameter names, kinds, defaults, annotations, and return
    annotation. The delegating shim and the real orchestrator must stay
    swap-compatible."""
    from moe_compress.stage6 import orchestrator

    orch_sig = inspect.signature(orchestrator.run)
    mono_sig = inspect.signature(stage6_validate.run)

    orch_params = list(orch_sig.parameters.values())
    mono_params = list(mono_sig.parameters.values())

    assert len(orch_params) == len(mono_params)
    for op, mp in zip(orch_params, mono_params):
        assert op.name == mp.name, f"name mismatch: {op.name} != {mp.name}"
        assert op.kind == mp.kind, f"kind mismatch for {op.name}"
        assert op.default == mp.default, f"default mismatch for {op.name}"
        assert op.annotation == mp.annotation, f"annotation mismatch for {op.name}"
    assert orch_sig.return_annotation == mono_sig.return_annotation


def test_stage6_stage_object():
    """``STAGE6`` is a ``Stage``-conforming object — it satisfies the
    structural :class:`Stage` Protocol, exposes ``stage_id == "6"``, and has
    callable ``is_enabled`` / ``run`` methods. Mirrors the
    ``test_stage3_stage_object`` / ``test_stage4_stage_object`` pattern.
    """
    assert isinstance(STAGE6, Stage)
    assert STAGE6.stage_id == "6"
    assert callable(STAGE6.is_enabled)
    assert callable(STAGE6.run)
    # is_enabled never gates Stage 6 itself — stage selection belongs to the
    # universal orchestrator, not to the stage. Mirrors STAGE3 / STAGE4.
    assert STAGE6.is_enabled({}) is True
