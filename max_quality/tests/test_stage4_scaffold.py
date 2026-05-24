"""Tests for the ``stage4/`` package surface.

S4-1 created the ``stage4/`` package skeleton — an ``__init__`` re-exporting
``run``, an ``orchestrator`` module, a ``context`` re-export shim, and a
``plugins`` package. S4-2..S4-3 extracted the EoRA algorithm into
``stage4/plugins/``; S4-4a made ``stage4/orchestrator.run`` the REAL
plugin-driven orchestrator and flipped ``stage4_eora.run`` to a thin shim that
delegates to IT (the inverse of the S4-1 scaffold direction).

These tests guard the package surface: the package imports cleanly, and
``stage4_eora.run`` delegates to ``stage4.orchestrator.run`` with a matching
signature. The byte-identity of the new orchestrator is covered by
``test_stage4_golden_snapshot.py``.
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


def test_stage4_eora_run_delegates_to_orchestrator(monkeypatch):
    """S4-4a: ``stage4_eora.run`` is now the thin shim — it forwards every
    argument unchanged to ``stage4.orchestrator.run`` (4 positionals + the
    kw-only ``no_resume``). Pure unit test, no model."""
    from moe_compress.stage4 import orchestrator

    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    # The shim does a function-local ``from .stage4.orchestrator import run``;
    # patching the ``run`` attribute on the orchestrator module is what it
    # resolves.
    monkeypatch.setattr(orchestrator, "run", _sentinel)

    model = object()
    tokenizer = object()
    config = {"stage4_eora": {}}
    artifacts_dir = object()

    result = stage4_eora.run(
        model, tokenizer, config, artifacts_dir, no_resume=True,
    )

    assert result is sentinel_result
    assert len(calls) == 1, "stage4.orchestrator.run must be called exactly once"

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
