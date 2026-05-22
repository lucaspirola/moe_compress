"""Tests for the S3-1 ``stage3/`` package scaffold.

S3-1 creates the ``stage3/`` package skeleton — an ``__init__`` re-exporting
``run``, an ``orchestrator`` whose ``run`` thinly delegates to the legacy
``stage3_svd.run`` monolith, a ``context`` re-export shim, and an empty
``plugins`` package. These tests guard the package surface and the
pass-through delegation contract; the SVD algorithm itself stays in
``stage3_svd.py`` until S3-2..S3-7. Pure unit tests — no model required.
"""
from __future__ import annotations

import inspect

from moe_compress import stage3
from moe_compress import stage3_svd


def test_stage3_package_imports():
    """The ``stage3`` package and its scaffold modules import cleanly."""
    from moe_compress.stage3 import orchestrator, context  # noqa: F401
    import moe_compress.stage3.plugins  # noqa: F401

    assert callable(stage3.run)


def test_stage3_orchestrator_delegates_to_legacy(monkeypatch):
    """``stage3.orchestrator.run`` forwards every argument unchanged to
    ``stage3_svd.run`` — ``decomposition`` in the 5th positional slot and
    ``device`` / ``no_resume`` as keyword arguments."""
    from moe_compress.stage3 import orchestrator

    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    monkeypatch.setattr(stage3_svd, "run", _sentinel)

    model = object()
    tokenizer = object()
    config = {"stage3_svd": {}}
    artifacts_dir = object()
    decomposition = object()
    device = object()

    result = orchestrator.run(
        model, tokenizer, config, artifacts_dir, decomposition,
        device=device, no_resume=True,
    )

    assert result is sentinel_result
    assert len(calls) == 1, "legacy stage3_svd.run must be called exactly once"

    args, kwargs = calls[0]
    assert args == (model, tokenizer, config, artifacts_dir, decomposition)
    assert args[4] is decomposition, "decomposition must be the 5th positional arg"
    assert kwargs == {"device": device, "no_resume": True}


def test_stage3_orchestrator_signature_matches_legacy():
    """``stage3.orchestrator.run`` signature is identical to the legacy
    ``stage3_svd.run`` — parameter names, kinds, and defaults. Guards the
    pass-through contract S3-7 relies on."""
    scaffold_sig = inspect.signature(stage3.orchestrator.run)
    legacy_sig = inspect.signature(stage3_svd.run)

    scaffold_params = list(scaffold_sig.parameters.values())
    legacy_params = list(legacy_sig.parameters.values())

    assert len(scaffold_params) == len(legacy_params)
    for sp, lp in zip(scaffold_params, legacy_params):
        assert sp.name == lp.name, f"name mismatch: {sp.name} != {lp.name}"
        assert sp.kind == lp.kind, f"kind mismatch for {sp.name}"
        assert sp.default == lp.default, f"default mismatch for {sp.name}"
