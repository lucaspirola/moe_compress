"""Tests for the ``stage4/`` package surface.

S4-1 creates the ``stage4/`` package skeleton — an ``__init__`` re-exporting
``run``, an ``orchestrator`` module whose ``run`` is a thin delegation to the
legacy ``stage4_eora.run`` monolith, a ``context`` re-export shim, and a
``plugins`` package. Tasks S4-2..S4-3 extract the EoRA algorithm into
``stage4/plugins/``; S4-4 then makes ``stage4/orchestrator.run`` the REAL
plugin-driven orchestrator and deletes the monolith.

These tests guard the S4-1 scaffold surface: the package imports cleanly, and
``stage4.orchestrator.run`` delegates to ``stage4_eora.run`` with a matching
signature (the seam S4-4 swaps).
"""
from __future__ import annotations

import inspect

from moe_compress import stage4
from moe_compress import stage4_eora


def test_stage4_package_imports():
    """The ``stage4`` package and its modules import cleanly."""
    from moe_compress.stage4 import orchestrator, context  # noqa: F401
    import moe_compress.stage4.plugins  # noqa: F401

    assert callable(stage4.run)


def test_stage4_orchestrator_delegates_to_legacy(monkeypatch):
    """S4-1: ``stage4.orchestrator.run`` is a thin shim — it forwards every
    argument unchanged to ``stage4_eora.run`` (4 positionals + the kw-only
    ``no_resume``). Pure unit test, no model."""
    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    # The orchestrator binds the monolith as ``_legacy_stage4`` (the module
    # object) and calls ``_legacy_stage4.run`` — patching the ``run``
    # attribute on the ``stage4_eora`` module is what it resolves.
    monkeypatch.setattr(stage4_eora, "run", _sentinel)

    model = object()
    tokenizer = object()
    config = {"stage4_eora": {}}
    artifacts_dir = object()

    result = stage4.orchestrator.run(
        model, tokenizer, config, artifacts_dir, no_resume=True,
    )

    assert result is sentinel_result
    assert len(calls) == 1, "stage4_eora.run must be called exactly once"

    args, kwargs = calls[0]
    assert args == (model, tokenizer, config, artifacts_dir)
    assert kwargs == {"no_resume": True}


def test_stage4_orchestrator_signature_matches_legacy():
    """``stage4.orchestrator.run`` and ``stage4_eora.run`` have identical
    signatures — parameter names, kinds, defaults, annotations, and return
    annotation. The delegating shim and the legacy monolith must stay
    swap-compatible (the seam S4-4 swaps)."""
    orch_sig = inspect.signature(stage4.orchestrator.run)
    legacy_sig = inspect.signature(stage4_eora.run)

    orch_params = list(orch_sig.parameters.values())
    legacy_params = list(legacy_sig.parameters.values())

    assert len(orch_params) == len(legacy_params)
    for op, lp in zip(orch_params, legacy_params):
        assert op.name == lp.name, f"name mismatch: {op.name} != {lp.name}"
        assert op.kind == lp.kind, f"kind mismatch for {op.name}"
        assert op.default == lp.default, f"default mismatch for {op.name}"
        assert op.annotation == lp.annotation, f"annotation mismatch for {op.name}"
    assert orch_sig.return_annotation == legacy_sig.return_annotation
