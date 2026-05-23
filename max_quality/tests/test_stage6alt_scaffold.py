"""Tests for the ``stage6alt/`` package surface.

S6A-1 creates the ``stage6alt/`` package skeleton — an ``__init__``
re-exporting ``run``, an ``orchestrator`` module, a ``context`` re-export
shim, and an (empty) ``plugins`` package. Tasks S6A-2..S6A-5 will extract
the Stage 6alt thermometer algorithm into ``stage6alt/plugins/``; S6A-6
flips the relationship so ``stage6alt.orchestrator.run`` becomes the REAL
orchestrator and ``stage6alt_thermometer.run`` becomes the thin shim that
delegates to it; S6A-6 also introduces the ``STAGE6ALT`` plugin manifest.

These tests guard the scaffold package surface: the package imports
cleanly, ``stage6alt.orchestrator.run`` delegates to
``stage6alt_thermometer.run`` with the matching signature, the signatures
are identical, and no ``STAGE6ALT`` symbol is exported yet (deferred to
S6A-6). The byte-identity of the monolith is covered by
``test_stage6alt_golden_snapshot.py``.

At S6A-6 these tests will be updated to reflect the flipped delegation
direction and the introduction of the ``STAGE6ALT`` manifest (as was
done for stage6 at its flip).
"""
from __future__ import annotations

import inspect

import pytest

from moe_compress import stage6alt
from moe_compress import stage6alt_thermometer


def test_stage6alt_package_imports():
    """The ``stage6alt`` package and its modules import cleanly, and the
    ``STAGE6ALT`` manifest object is NOT exported yet (deferred to S6A-6).
    """
    from moe_compress.stage6alt import orchestrator, context  # noqa: F401
    import moe_compress.stage6alt.plugins  # noqa: F401

    assert callable(stage6alt.run)

    # The STAGE6ALT plugin manifest is deliberately NOT introduced until
    # S6A-6 — guard against accidental early export.
    assert not hasattr(stage6alt, "STAGE6ALT"), (
        "STAGE6ALT must not be exported until S6A-6 flips the orchestrator"
    )


def test_stage6alt_orchestrator_delegates_to_monolith(monkeypatch):
    """S6A-1 scaffold: ``stage6alt.orchestrator.run`` is the thin shim — it
    forwards every argument unchanged to ``stage6alt_thermometer.run`` (4
    positionals + the kw-only ``device``). Pure unit test, no model."""
    import moe_compress.stage6alt_thermometer as _monolith

    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    monkeypatch.setattr(_monolith, "run", _sentinel)

    model = object()
    tokenizer = object()
    config = {"stage6alt_thermometer": {}}
    artifacts_dir = object()
    device = object()

    result = stage6alt.run(
        model, tokenizer, config, artifacts_dir, device=device,
    )

    assert result is sentinel_result
    assert len(calls) == 1, "stage6alt_thermometer.run must be called exactly once"

    args, kwargs = calls[0]
    assert args == (model, tokenizer, config, artifacts_dir)
    assert kwargs == {"device": device}


def test_stage6alt_orchestrator_signature_matches_monolith():
    """``stage6alt.orchestrator.run`` and ``stage6alt_thermometer.run`` have
    identical signatures — parameter names, kinds, defaults, annotations,
    and return annotation. The delegating shim and the legacy monolith
    must stay swap-compatible (the seam S6A-6 will swap)."""
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
