"""Tests for the ``stage6/`` package surface.

S6-1 creates the ``stage6/`` package skeleton — an ``__init__`` re-exporting
``run``, an ``orchestrator`` module, a ``context`` re-export shim, and an
(empty) ``plugins`` package. Tasks S6-2..S6-7 will extract the Stage 6
validation algorithm into ``stage6/plugins/``; S6-8 flips the relationship
so ``stage6.orchestrator.run`` becomes the REAL orchestrator and
``stage6_validate.run`` becomes the thin shim that delegates to it.

These tests guard the scaffold package surface: the package imports cleanly,
``stage6.orchestrator.run`` delegates to ``stage6_validate.run`` with the
matching signature, and the signatures are identical. The byte-identity of
the monolith is covered by ``test_stage6_golden_snapshot.py``.

At S6-8 these tests will be updated to reflect the flipped delegation
direction (as was done for stage3 at S3-7a and stage4 at S4-4a).
"""
from __future__ import annotations

import inspect

from moe_compress import stage6
from moe_compress import stage6_validate


def test_stage6_package_imports():
    """The ``stage6`` package and its modules import cleanly."""
    from moe_compress.stage6 import orchestrator, context  # noqa: F401
    import moe_compress.stage6.plugins  # noqa: F401

    assert callable(stage6.run)


def test_stage6_orchestrator_delegates_to_monolith(monkeypatch):
    """S6-1 scaffold: ``stage6.orchestrator.run`` is the thin shim — it
    forwards every argument unchanged to ``stage6_validate.run`` (4
    positionals + the kw-only ``device``). Pure unit test, no model."""
    import moe_compress.stage6_validate as _monolith

    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    monkeypatch.setattr(_monolith, "run", _sentinel)

    model = object()
    tokenizer = object()
    config = {"stage6_validate": {}}
    artifacts_dir = object()
    device = object()

    result = stage6.run(
        model, tokenizer, config, artifacts_dir, device=device,
    )

    assert result is sentinel_result
    assert len(calls) == 1, "stage6_validate.run must be called exactly once"

    args, kwargs = calls[0]
    assert args == (model, tokenizer, config, artifacts_dir)
    assert kwargs == {"device": device}


def test_stage6_orchestrator_signature_matches_monolith():
    """``stage6.orchestrator.run`` and ``stage6_validate.run`` have identical
    signatures — parameter names, kinds, defaults, annotations, and return
    annotation. The delegating shim and the legacy monolith must stay
    swap-compatible (the seam S6-8 will swap)."""
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
