"""Tests for the ``router_kd/`` package surface.

RK-1 created the ``router_kd/`` package skeleton — an ``__init__`` re-exporting
``run`` and ``make_router_kd_stage``, an ``orchestrator`` module, a ``context``
re-export shim, an (empty) ``plugins`` package, and a ``stage`` module exposing
the ``make_router_kd_stage(stage_id)`` factory. RK-2..RK-7 extracted the unified
Router-KD algorithm into ``router_kd/plugins/``; RK-8 made
``router_kd.orchestrator.run`` the real plugin-driven orchestrator and flipped
``stage5_router_kd.run`` to a thin shim that delegates TO it.

These tests guard the package surface: the package imports cleanly,
``stage5_router_kd.run`` delegates to ``router_kd.orchestrator.run`` with a
matching signature, and the ``make_router_kd_stage`` factory produces
``Stage``-conforming objects that thread the correct ``stage_key``. The
byte-identity of the orchestrator is covered by
``test_router_kd_golden_snapshot.py``.
"""
from __future__ import annotations

import inspect

import pytest

from moe_compress import router_kd
from moe_compress import stage5_router_kd
from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.stage import Stage
from moe_compress.router_kd import make_router_kd_stage


def test_router_kd_package_imports():
    """The ``router_kd`` package and its modules import cleanly."""
    from moe_compress.router_kd import orchestrator, context, stage  # noqa: F401
    import moe_compress.router_kd.plugins  # noqa: F401

    assert callable(router_kd.run)
    assert callable(router_kd.make_router_kd_stage)


def test_legacy_run_delegates_to_orchestrator(monkeypatch):
    """RK-8: ``stage5_router_kd.run`` is a thin shim — it forwards every
    argument unchanged to the real ``router_kd.orchestrator.run`` orchestrator
    (4 positionals + the kw-only ``device`` / ``no_resume`` / ``stage_key``).
    Pure unit test, no model."""
    from moe_compress.router_kd import orchestrator

    calls: list[tuple] = []
    sentinel_result = object()

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_result

    # ``stage5_router_kd.run`` does a function-local ``from
    # .router_kd.orchestrator import run as _orchestrator_run`` then
    # ``_orchestrator_run(...)`` — patching the ``run`` attribute on the
    # orchestrator module is what that import resolves.
    monkeypatch.setattr(orchestrator, "run", _sentinel)

    student = object()
    tokenizer = object()
    config = {"stage5_router_kd": {}}
    artifacts_dir = object()
    device = object()

    result = stage5_router_kd.run(
        student, tokenizer, config, artifacts_dir,
        device=device, no_resume=True, stage_key="stage2p5",
    )

    assert result is sentinel_result
    assert len(calls) == 1, "router_kd.orchestrator.run must be called exactly once"

    args, kwargs = calls[0]
    assert args == (student, tokenizer, config, artifacts_dir)
    assert kwargs == {"device": device, "no_resume": True, "stage_key": "stage2p5"}


def test_orchestrator_signature_matches_legacy():
    """``router_kd.orchestrator.run`` and ``stage5_router_kd.run`` have
    identical signatures — parameter names, kinds, defaults, annotations, and
    return annotation. The real orchestrator and the thin shim must stay
    swap-compatible (the seam RK-8 swapped)."""
    orch_sig = inspect.signature(router_kd.orchestrator.run)
    legacy_sig = inspect.signature(stage5_router_kd.run)

    orch_params = list(orch_sig.parameters.values())
    legacy_params = list(legacy_sig.parameters.values())

    assert len(orch_params) == len(legacy_params)
    for op, lp in zip(orch_params, legacy_params):
        assert op.name == lp.name, f"name mismatch: {op.name} != {lp.name}"
        assert op.kind == lp.kind, f"kind mismatch for {op.name}"
        assert op.default == lp.default, f"default mismatch for {op.name}"
        assert op.annotation == lp.annotation, f"annotation mismatch for {op.name}"
    assert orch_sig.return_annotation == legacy_sig.return_annotation


def test_make_router_kd_stage_produces_conforming_stages():
    """``make_router_kd_stage`` produces ``Stage``-conforming objects for both
    canonical stage_id values ``"2.5"`` and ``"5"``."""
    for stage_id in ("2.5", "5"):
        s = make_router_kd_stage(stage_id)
        assert isinstance(s, Stage)
        assert s.stage_id == stage_id
        assert callable(s.run)
        assert s.is_enabled({}) is True
        assert s.is_enabled({"x": 1}) is True


def test_make_router_kd_stage_rejects_unknown_id():
    """``make_router_kd_stage`` rejects anything but the canonical ``"2.5"`` /
    ``"5"`` stage_id values — including the monolith's ``stage_key`` form
    ``"stage5"``, which is NOT an accepted ``stage_id``."""
    with pytest.raises(ValueError):
        make_router_kd_stage("5.5")
    with pytest.raises(ValueError):
        make_router_kd_stage("stage5")
    with pytest.raises(ValueError):
        make_router_kd_stage("2p5")


def test_stage_run_threads_correct_stage_key(monkeypatch):
    """``_RouterKdStage.run`` unwraps ``ctx`` and threads the correct
    ``stage_key`` into the orchestrator, then writes the namespaced
    ``router_kd_<stage_key>_path`` slot back. Pure unit test, no model."""
    from moe_compress.router_kd import stage as stage_mod

    for stage_id, stage_key in (("2.5", "stage2p5"), ("5", "stage5")):
        captured: dict = {}
        sentinel_out = object()

        def _sentinel(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return sentinel_out

        # ``stage.py`` binds the orchestrator at module-import time via
        # ``from .orchestrator import run as _orchestrator_run`` — that copies
        # the function reference, so the name the factory actually resolves at
        # call time is ``stage._orchestrator_run``. Patch it there.
        monkeypatch.setattr(stage_mod, "_orchestrator_run", _sentinel)

        ctx = PipelineContext()
        student = object()
        tokenizer = object()
        config = {"stage5_router_kd": {}}
        artifacts_dir = object()
        ctx.set("student", student)
        ctx.set("tokenizer", tokenizer)
        ctx.set("config", config)
        ctx.set("artifacts_dir", artifacts_dir)

        result = make_router_kd_stage(stage_id).run(ctx)

        assert result is None
        assert captured["args"] == (student, tokenizer, config, artifacts_dir)
        assert captured["kwargs"]["stage_key"] == stage_key
        assert ctx.get(f"router_kd_{stage_key}_path") is sentinel_out
