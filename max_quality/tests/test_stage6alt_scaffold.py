"""Tests for the ``stage6alt/`` package surface.

S6A-1 created the ``stage6alt/`` package skeleton — an ``__init__``
re-exporting ``run``, an ``orchestrator`` module, a ``context`` re-export
shim, and an (empty) ``plugins`` package. S6A-2..S6A-5 extracted the
Stage 6alt thermometer algorithm into ``stage6alt/plugins/``; S6A-6 made
``stage6alt/orchestrator.run`` the REAL plugin-driven orchestrator and
flipped ``stage6alt_thermometer.run`` to a thin shim that delegates to IT
(the inverse of the S6A-1 scaffold direction), and introduced the
``STAGE6ALT`` ``Stage``-conforming object.

These tests guard the package surface only. The S6A-1 delegation-
direction test (``stage6alt.orchestrator.run`` → ``stage6alt_thermometer.run``)
was retired at S6A-6: the orchestrator no longer delegates, it IS the
implementation, and ``stage6alt_thermometer.run`` is now the delegator.
The byte-identity of the new orchestrator is covered by
``test_stage6alt_golden_snapshot.py``.
"""
from __future__ import annotations

import inspect

from moe_compress import stage6alt
from moe_compress import stage6alt_thermometer
from moe_compress.pipeline.stage import Stage
from moe_compress.stage6alt import STAGE6ALT


def test_stage6alt_package_imports():
    """The ``stage6alt`` package and its modules import cleanly."""
    from moe_compress.stage6alt import orchestrator, context  # noqa: F401
    import moe_compress.stage6alt.plugins  # noqa: F401

    assert callable(stage6alt.run)


def test_stage6alt_shim_delegates_to_orchestrator(monkeypatch):
    """S6A-6: ``stage6alt_thermometer.run`` is now the thin shim — it forwards
    every argument unchanged to ``stage6alt.orchestrator.run`` (4 positionals +
    the kw-only ``device``). Pure unit test, no model."""
    from moe_compress.stage6alt import orchestrator

    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    # The shim does a function-local ``from .stage6alt.orchestrator import run``;
    # patching the ``run`` attribute on the orchestrator module is what it
    # resolves.
    monkeypatch.setattr(orchestrator, "run", _sentinel)

    model = object()
    tokenizer = object()
    config = {"stage6_validate": {}}
    artifacts_dir = object()
    device = object()

    result = stage6alt_thermometer.run(
        model, tokenizer, config, artifacts_dir, device=device,
    )

    assert result is sentinel_result
    assert len(calls) == 1, "stage6alt.orchestrator.run must be called exactly once"

    args, kwargs = calls[0]
    assert args == (model, tokenizer, config, artifacts_dir)
    assert kwargs == {"device": device}


def test_stage6alt_orchestrator_signature_matches_monolith():
    """``stage6alt.orchestrator.run`` and ``stage6alt_thermometer.run`` have
    identical signatures — parameter names, kinds, defaults, annotations,
    and return annotation. The delegating shim and the real orchestrator
    must stay swap-compatible (post-S6A-6 the seam is unchanged)."""
    from moe_compress.stage6alt import orchestrator

    orch_sig = inspect.signature(orchestrator.run)
    mono_sig = inspect.signature(stage6alt_thermometer.run)

    orch_params = list(orch_sig.parameters.values())
    mono_params = list(mono_sig.parameters.values())

    assert len(orch_params) == len(mono_params)
    for op, mp in zip(orch_params, mono_params):
        assert op.name == mp.name, f"name mismatch: {op.name} != {mp.name}"
        assert op.kind == mp.kind, f"kind mismatch for {op.name}"
        assert op.default == mp.default, f"default mismatch for {op.name}"
        assert op.annotation == mp.annotation, f"annotation mismatch for {op.name}"
    assert orch_sig.return_annotation == mono_sig.return_annotation


def test_stage6alt_stage_object():
    """``STAGE6ALT`` is a ``Stage``-conforming object — it satisfies the
    structural :class:`Stage` Protocol, exposes ``stage_id == "6alt"``, and
    has callable ``is_enabled`` / ``run`` methods. Mirrors the
    ``test_stage6_stage_object`` pattern.
    """
    assert isinstance(STAGE6ALT, Stage)
    assert STAGE6ALT.stage_id == "6alt"
    assert callable(STAGE6ALT.is_enabled)
    assert callable(STAGE6ALT.run)
    # is_enabled never gates Stage 6alt itself — stage selection (full vs
    # thermometer) is a run_pipeline-level dispatch on
    # ``stage6_validate.mode``, not on the stage object. Mirrors STAGE6.
    assert STAGE6ALT.is_enabled({}) is True
