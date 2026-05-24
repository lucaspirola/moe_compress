"""Tests for the ``stage3/`` package surface.

S3-1 created the ``stage3/`` package skeleton — an ``__init__`` re-exporting
``run``, an ``orchestrator`` module, a ``context`` re-export shim, and a
``plugins`` package. S3-2..S3-6 extracted the SVD algorithm into
``stage3/plugins/``; S3-7a made ``stage3/orchestrator.run`` the REAL
plugin-driven orchestrator and flipped ``stage3_svd.run`` to a thin shim that
delegates to IT (the inverse of the S3-1 scaffold direction).

These tests guard the package surface only. The S3-1 delegation-direction
tests (``stage3.orchestrator.run`` → ``stage3_svd.run``) were retired at
S3-7a: the orchestrator no longer delegates, it IS the implementation, and
``stage3_svd.run`` is now the delegator. The byte-identity of the new
orchestrator is covered by ``test_stage3_golden_snapshot.py``.
"""
from __future__ import annotations

import inspect

from moe_compress import stage3
from moe_compress import stage3_svd


def test_stage3_package_imports():
    """The ``stage3`` package and its modules import cleanly."""
    from moe_compress.stage3 import orchestrator, context  # noqa: F401
    import moe_compress.stage3.plugins  # noqa: F401

    assert callable(stage3.run)


def test_stage3_svd_run_delegates_to_orchestrator(monkeypatch):
    """S3-7a: ``stage3_svd.run`` is now the thin shim — it forwards every
    argument unchanged to ``stage3.orchestrator.run`` (``decomposition`` in
    the 5th positional slot, ``device`` / ``no_resume`` as keywords)."""
    from moe_compress.stage3 import orchestrator

    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    # The shim does a function-local ``from .stage3.orchestrator import run``;
    # patching the attribute on the orchestrator module is what it resolves.
    monkeypatch.setattr(orchestrator, "run", _sentinel)

    model = object()
    tokenizer = object()
    config = {"stage3_svd": {}}
    artifacts_dir = object()
    decomposition = object()
    device = object()

    result = stage3_svd.run(
        model, tokenizer, config, artifacts_dir, decomposition,
        device=device, no_resume=True,
    )

    assert result is sentinel_result
    assert len(calls) == 1, "stage3.orchestrator.run must be called exactly once"

    args, kwargs = calls[0]
    assert args == (model, tokenizer, config, artifacts_dir, decomposition)
    assert args[4] is decomposition, "decomposition must be the 5th positional arg"
    assert kwargs == {"device": device, "no_resume": True}


def test_stage3_run_signatures_match():
    """``stage3_svd.run`` and ``stage3.orchestrator.run`` have identical
    signatures — parameter names, kinds, and defaults. The shim and the real
    orchestrator must stay swap-compatible."""
    shim_sig = inspect.signature(stage3_svd.run)
    orch_sig = inspect.signature(stage3.orchestrator.run)

    shim_params = list(shim_sig.parameters.values())
    orch_params = list(orch_sig.parameters.values())

    assert len(shim_params) == len(orch_params)
    for sp, op in zip(shim_params, orch_params):
        assert sp.name == op.name, f"name mismatch: {sp.name} != {op.name}"
        assert sp.kind == op.kind, f"kind mismatch for {sp.name}"
        assert sp.default == op.default, f"default mismatch for {sp.name}"
