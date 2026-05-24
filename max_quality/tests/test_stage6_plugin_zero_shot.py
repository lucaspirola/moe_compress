"""S6-3 — Stage 6 zero-shot lm-eval plugin extraction tests.

Verifies the S6-3 ``ZeroShotLmEvalPlugin`` scaffolding in
``stage6/plugins/zero_shot_lm_eval.py``:

* the Pattern-A symbols ``_ZERO_SHOT_TASKS`` + ``_lm_eval_tasks`` import from
  the plugin module;
* the ``stage6_validate`` monolith re-exports the SAME relocated objects (the
  ``# noqa: F401`` re-import block is load-bearing);
* ``ZeroShotLmEvalPlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries correct metadata, gates on
  ``stage6_validate.zero_shot.enabled`` and exposes the (S6-8) ``eval_task``
  phase hook;
* the module never imports the ``stage6_validate`` monolith or
  ``stage6.orchestrator`` at any scope (the circular-import contract);
* the relocated ``_lm_eval_tasks`` keeps its graceful-fallback branch (returns
  ``{}`` when lm-eval is unavailable) and its eager-attention RuntimeError;
* the inert ``eval_task`` hook merges the harness result into ``eval_results``.

S6-3 covers a MIXED pattern: ``_ZERO_SHOT_TASKS`` / ``_lm_eval_tasks`` are
relocated verbatim (the monolith re-imports them); the inline ``run()``
student-side call site is reproduced in the inert ``eval_task`` hook (the
monolith ``run()`` is NOT modified for it). The byte-identical behavioral gate
is the S6-0 golden snapshot (``test_stage6_golden_snapshot.py``); this file
only checks the relocation plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6.plugins import zero_shot_lm_eval  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6 zero-shot lm-eval imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Local helpers (mirror test_stage6_plugin_environment.py discipline)
# ---------------------------------------------------------------------------


class _FakeConfigModel:
    """Minimal model stub exposing only ``config._attn_implementation``.

    ``_lm_eval_tasks`` reads ``model.config._attn_implementation`` for its
    eager-attention guard before touching the harness.
    """

    def __init__(self, attn_implementation="eager"):
        self.config = types.SimpleNamespace(
            _attn_implementation=attn_implementation
        )


def _fake_harness_module(results):
    """Build a fake ``lm_eval`` package whose ``simple_evaluate`` returns ``results``.

    Mirrors the harness API surface ``_lm_eval_tasks`` touches:
    ``from lm_eval import simple_evaluate`` and
    ``from lm_eval.models.huggingface import HFLM``. Both imports are
    function-local, so the fake is injected via
    ``monkeypatch.setitem(sys.modules, ...)``.
    """
    pkg = types.ModuleType("lm_eval")

    def simple_evaluate(*_args, **_kwargs):
        return {"results": results}

    pkg.simple_evaluate = simple_evaluate

    models = types.ModuleType("lm_eval.models")
    huggingface = types.ModuleType("lm_eval.models.huggingface")

    class HFLM:
        def __init__(self, *_args, **_kwargs):
            pass

    huggingface.HFLM = HFLM
    models.huggingface = huggingface
    pkg.models = models
    return pkg, models, huggingface


def _inject_fake_harness(monkeypatch, results):
    """Register the fake ``lm_eval`` package (+ submodules) in ``sys.modules``."""
    pkg, models, huggingface = _fake_harness_module(results)
    monkeypatch.setitem(sys.modules, "lm_eval", pkg)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", huggingface)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_zero_shot_module_imports():
    """``_ZERO_SHOT_TASKS`` + ``_lm_eval_tasks`` + ``ZeroShotLmEvalPlugin`` import."""
    from moe_compress.stage6.plugins.zero_shot_lm_eval import (
        ZeroShotLmEvalPlugin,
        _ZERO_SHOT_TASKS,
        _lm_eval_tasks,
    )

    assert isinstance(ZeroShotLmEvalPlugin, type)
    assert isinstance(_ZERO_SHOT_TASKS, frozenset)
    assert callable(_lm_eval_tasks)


def test_monolith_reexports_zero_shot_symbols():
    """The monolith re-exports the SAME relocated zero-shot symbols.

    ``_lm_eval_tasks`` is ``is``-identity checked (a function object); the
    immutable ``_ZERO_SHOT_TASKS`` frozenset is equality-checked.
    """
    from moe_compress import stage6_validate
    from moe_compress.stage6.plugins import zero_shot_lm_eval

    assert stage6_validate._lm_eval_tasks is zero_shot_lm_eval._lm_eval_tasks
    assert stage6_validate._ZERO_SHOT_TASKS == zero_shot_lm_eval._ZERO_SHOT_TASKS


def test_plugin_satisfies_protocol():
    """``ZeroShotLmEvalPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6.plugins.zero_shot_lm_eval import ZeroShotLmEvalPlugin

    assert isinstance(ZeroShotLmEvalPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed reads-writes-provides."""
    from moe_compress.stage6.plugins.zero_shot_lm_eval import ZeroShotLmEvalPlugin

    plugin = ZeroShotLmEvalPlugin()
    assert plugin.name == "zero_shot_lm_eval"
    assert plugin.config_key == "stage6_validate.zero_shot.enabled"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.reads == (
        "config",
        "model",
        "tokenizer",
        "eval_results",
        "eval_text_concat",
    )
    assert plugin.writes == ("eval_results",)
    # eval_results is a shared collector (in `writes`), not a calibration-pass
    # accumulator — `provides` is empty.
    assert plugin.provides == ()
    assert "eval_results" in plugin.writes


def test_zero_shot_tasks_constant():
    """``_ZERO_SHOT_TASKS`` is the canonical ARC-C + HellaSwag metric-key set."""
    from moe_compress.stage6.plugins.zero_shot_lm_eval import _ZERO_SHOT_TASKS

    assert _ZERO_SHOT_TASKS == frozenset({"arc_challenge_acc", "hellaswag_acc"})


def test_plugin_is_enabled_gating():
    """``is_enabled`` gates on ``stage6_validate.zero_shot.enabled``.

    Empty config → False; ``enabled=True`` → True; ``enabled=False`` → False.
    """
    from moe_compress.stage6.plugins.zero_shot_lm_eval import ZeroShotLmEvalPlugin

    plugin = ZeroShotLmEvalPlugin()
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled(
        {"stage6_validate": {"zero_shot": {"enabled": True}}}
    ) is True
    assert plugin.is_enabled(
        {"stage6_validate": {"zero_shot": {"enabled": False}}}
    ) is False


def test_plugin_has_eval_task_hook():
    """The S6-8 phase hook ``eval_task`` is present and callable."""
    from moe_compress.stage6.plugins.zero_shot_lm_eval import ZeroShotLmEvalPlugin

    plugin = ZeroShotLmEvalPlugin()
    assert callable(getattr(plugin, "eval_task", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6_validate`` / ``stage6.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator) at any scope — module-top OR function-local — since either
    would risk an import cycle (the monolith re-imports *this* module at load
    time). Parse the source with ``ast`` and walk the FULL tree so a
    function-local ``import stage6_validate`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported symbols) —
    so the cycle-causing ``from moe_compress import stage6_validate`` form is
    also caught. Each alias's ``asname`` is checked alongside its ``name`` so a
    renamed import cannot slip past either.
    """
    from moe_compress.stage6.plugins import zero_shot_lm_eval as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = ("stage6_validate", "stage6.orchestrator", "orchestrator")
    for node in ast.walk(tree):  # any nesting level, not just module-top
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(
                    f in alias.name or f in (alias.asname or "")
                    for f in forbidden
                ), f"forbidden import at any scope: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert not any(f in mod_name for f in forbidden), (
                f"forbidden import-from at any scope: {mod_name}"
            )
            for alias in node.names:
                assert not any(
                    f in alias.name or f in (alias.asname or "")
                    for f in forbidden
                ), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


def test_lm_eval_tasks_returns_empty_when_unavailable(monkeypatch):
    """``_lm_eval_tasks`` returns ``{}`` when the harness cannot be imported.

    The monolith's relocated body wraps ``from lm_eval import simple_evaluate``
    in ``try/except Exception`` and returns ``{}`` on failure. Setting the
    ``lm_eval`` ``sys.modules`` entry to ``None`` makes ``from lm_eval import
    ...`` raise ``ImportError`` — caught by that branch.
    """
    from moe_compress.stage6.plugins.zero_shot_lm_eval import _lm_eval_tasks

    monkeypatch.setitem(sys.modules, "lm_eval", None)
    out = _lm_eval_tasks(_FakeConfigModel(), object(), ["arc_challenge"])
    assert out == {}


def test_lm_eval_tasks_eager_attn_assert(monkeypatch):
    """``_lm_eval_tasks`` raises ``RuntimeError`` under non-eager attention.

    With a fake ``lm_eval`` module injected (so the import succeeds), a model
    whose ``config._attn_implementation`` is not ``'eager'`` must trip the
    Spec §9 #2 eager-attention guard.
    """
    from moe_compress.stage6.plugins.zero_shot_lm_eval import _lm_eval_tasks

    _inject_fake_harness(monkeypatch, results={})
    model = _FakeConfigModel(attn_implementation="sdpa")
    with pytest.raises(RuntimeError, match="_attn_implementation"):
        _lm_eval_tasks(model, object(), ["arc_challenge"])


def test_eval_task_hook_writes_eval_results(monkeypatch):
    """The inert ``eval_task`` hook merges the harness result into ``eval_results``.

    Injects a fake ``lm_eval`` whose ``simple_evaluate`` returns a canned
    ARC-Challenge result, builds a ``PipelineContext`` with a pre-existing
    ``eval_results={}`` slot, calls the hook, and asserts the canned metric
    landed in ``eval_results``. The hook mutates the existing dict via
    ``dict.update`` — it does NOT ``ctx.set`` ``eval_results``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins.zero_shot_lm_eval import ZeroShotLmEvalPlugin

    # _lm_eval_tasks reads metrics[<key>] preferring "acc_norm,none".
    _inject_fake_harness(monkeypatch, results={
        "arc_challenge": {"acc_norm,none": 0.42, "acc,none": 0.40},
    })

    ctx = PipelineContext()
    ctx.set("model", _FakeConfigModel(attn_implementation="eager"))
    ctx.set("tokenizer", object())
    ctx.set("config", {
        "stage6_validate": {
            "lm_eval_batch_size": "auto:8",
            "zero_shot": {"enabled": True, "tasks": ["arc_challenge"]},
        }
    })
    eval_results: dict = {}
    ctx.set("eval_results", eval_results)

    plugin = ZeroShotLmEvalPlugin()
    plugin.eval_task(ctx)

    out = ctx.get("eval_results")
    assert out is eval_results, "eval_task must mutate the existing dict in place"
    assert out.get("arc_challenge_acc") == pytest.approx(0.42)
