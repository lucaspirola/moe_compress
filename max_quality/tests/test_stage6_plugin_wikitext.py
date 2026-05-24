"""S6-3 — Stage 6 WikiText-2 PPL plugin extraction tests.

Verifies the S6-3 ``WikitextPplPlugin`` scaffolding in
``stage6/plugins/wikitext_ppl.py``:

* the Pattern-A symbol ``_wikitext2_ppl`` imports from the plugin module;
* the ``stage6_validate`` monolith re-exports the SAME relocated function
  object (the ``# noqa: F401`` re-import block is load-bearing);
* ``WikitextPplPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, gates on ``stage6_validate.wikitext2.enabled`` and
  exposes the (S6-8) ``eval_task`` phase hook;
* the module never imports the ``stage6_validate`` monolith or
  ``stage6.orchestrator`` at any scope (the circular-import contract);
* the relocated ``_wikitext2_ppl`` keeps its spec asserts and returns a
  finite PPL on a tiny CPU model;
* the inert ``eval_task`` hook lands the wikitext PPL key in ``eval_results``.

S6-3 covers a MIXED pattern: ``_wikitext2_ppl`` is relocated verbatim (the
monolith re-imports it); the inline ``run()`` student-side call site is
reproduced in the inert ``eval_task`` hook (the monolith ``run()`` is NOT
modified for it). The byte-identical behavioral gate is the S6-0 golden
snapshot (``test_stage6_golden_snapshot.py``); this file only checks the
relocation plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6.plugins import wikitext_ppl  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6 wikitext-ppl imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Local helpers (mirror test_stage6_plugin_environment.py discipline)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer: maps every character to a small-vocab token id.

    ``_wikitext2_ppl`` calls ``tokenizer(text, add_special_tokens=...,
    return_tensors=None)["input_ids"]`` — a flat list of ints. The tiny CPU
    model has vocab 32, so token ids are kept in ``[0, 32)``.
    """

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [(ord(c) % 31) + 1 for c in text]
        return {"input_ids": ids}


def _fake_datasets_module(rows):
    """Build a fake ``datasets`` module whose ``load_dataset`` yields ``rows``.

    Each row is a ``{"text": ...}`` dict, matching the wikitext-2-raw-v1 schema
    ``_wikitext2_ppl`` iterates. The module is injected via
    ``monkeypatch.setitem(sys.modules, "datasets", ...)`` because the
    ``from datasets import load_dataset`` inside ``_wikitext2_ppl`` is
    function-local.
    """
    mod = types.ModuleType("datasets")

    def load_dataset(*_args, **_kwargs):
        return [{"text": r} for r in rows]

    mod.load_dataset = load_dataset
    return mod


def _eager(model):
    """Pin the tiny model's config to eager attention and return it.

    ``_wikitext2_ppl`` asserts ``model.config._attn_implementation == 'eager'``;
    the tiny-model fixture's config has no such attribute by default.
    """
    model.config._attn_implementation = "eager"
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wikitext_ppl_module_imports():
    """``_wikitext2_ppl`` + ``WikitextPplPlugin`` import from the plugin module."""
    from moe_compress.stage6.plugins.wikitext_ppl import (
        WikitextPplPlugin,
        _wikitext2_ppl,
    )

    assert isinstance(WikitextPplPlugin, type)
    assert callable(_wikitext2_ppl)


def test_monolith_reexports_wikitext_ppl():
    """The monolith re-exports the SAME relocated ``_wikitext2_ppl`` object.

    Proves the ``# noqa: F401`` re-import block in ``stage6_validate.py`` keeps
    ``run()`` and external callers/tests on their original import path.
    """
    from moe_compress import stage6_validate
    from moe_compress.stage6.plugins import wikitext_ppl

    assert stage6_validate._wikitext2_ppl is wikitext_ppl._wikitext2_ppl


def test_plugin_satisfies_protocol():
    """``WikitextPplPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6.plugins.wikitext_ppl import WikitextPplPlugin

    assert isinstance(WikitextPplPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed reads-writes-provides."""
    from moe_compress.stage6.plugins.wikitext_ppl import WikitextPplPlugin

    plugin = WikitextPplPlugin()
    assert plugin.name == "wikitext_ppl"
    assert plugin.config_key == "stage6_validate.wikitext2.enabled"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.reads == ("model", "tokenizer", "config", "dataset_revisions")
    assert plugin.writes == ("eval_results",)
    # eval_results is a shared collector (in `writes`), not a calibration-pass
    # accumulator — `provides` is empty.
    assert plugin.provides == ()
    assert "eval_results" in plugin.writes


def test_plugin_is_enabled_gating():
    """``is_enabled`` gates on ``stage6_validate.wikitext2.enabled``.

    Empty config → False; ``enabled=True`` → True; ``enabled=False`` → False.
    """
    from moe_compress.stage6.plugins.wikitext_ppl import WikitextPplPlugin

    plugin = WikitextPplPlugin()
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled(
        {"stage6_validate": {"wikitext2": {"enabled": True}}}
    ) is True
    assert plugin.is_enabled(
        {"stage6_validate": {"wikitext2": {"enabled": False}}}
    ) is False


def test_plugin_has_eval_task_hook():
    """The S6-8 phase hook ``eval_task`` is present and callable."""
    from moe_compress.stage6.plugins.wikitext_ppl import WikitextPplPlugin

    plugin = WikitextPplPlugin()
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
    from moe_compress.stage6.plugins import wikitext_ppl as mod

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


def test_wikitext2_ppl_sequence_length_assert():
    """``_wikitext2_ppl`` asserts ``cfg['sequence_length'] == 2048`` (Spec §9)."""
    from moe_compress.stage6.plugins.wikitext_ppl import _wikitext2_ppl

    cfg = {
        "dataset": "wikitext", "subset": "wikitext-2-raw-v1",
        "split": "test", "sequence_length": 1024,  # wrong — must be 2048
    }
    with pytest.raises(AssertionError, match="sequence_length"):
        _wikitext2_ppl(object(), object(), cfg)


def test_wikitext2_ppl_eager_attn_assert(tiny_model):
    """``_wikitext2_ppl`` asserts the model runs under eager attention."""
    from moe_compress.stage6.plugins.wikitext_ppl import _wikitext2_ppl

    tiny_model.config._attn_implementation = "sdpa"  # non-eager → must raise
    cfg = {
        "dataset": "wikitext", "subset": "wikitext-2-raw-v1",
        "split": "test", "sequence_length": 2048,
    }
    with pytest.raises(AssertionError, match="_attn_implementation"):
        _wikitext2_ppl(tiny_model, _FakeTokenizer(), cfg)


def test_wikitext2_ppl_returns_finite_float(tiny_model, monkeypatch):
    """``_wikitext2_ppl`` returns a finite positive PPL on a tiny CPU model.

    ``datasets`` is mocked via ``monkeypatch.setitem(sys.modules, "datasets",
    ...)`` because the ``from datasets import load_dataset`` inside
    ``_wikitext2_ppl`` is function-local. The fake corpus is long enough to
    yield at least one full 2048-token chunk.
    """
    import math as _math

    from moe_compress.stage6.plugins.wikitext_ppl import _wikitext2_ppl

    # ~3000 chars → one full 2048-token chunk after the _FakeTokenizer maps
    # each char to one token id.
    rows = ["the quick brown fox jumps over the lazy dog " * 8 for _ in range(10)]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module(rows))

    cfg = {
        "dataset": "wikitext", "subset": "wikitext-2-raw-v1",
        "split": "test", "sequence_length": 2048,
    }
    ppl = _wikitext2_ppl(_eager(tiny_model), _FakeTokenizer(), cfg, batch_size=2)
    assert isinstance(ppl, float)
    assert _math.isfinite(ppl)
    assert ppl > 0.0


def test_eval_task_hook_writes_eval_results(tiny_model, monkeypatch):
    """The inert ``eval_task`` hook lands the wikitext PPL key in ``eval_results``.

    Builds a ``PipelineContext`` with ``model`` / ``tokenizer`` / ``config`` /
    ``dataset_revisions`` / a pre-existing ``eval_results={}`` slot, mocks
    ``datasets``, calls the hook, and asserts the ``wikitext2_ppl`` key was
    added as a finite float. The hook mutates the existing dict — it does NOT
    ``ctx.set`` ``eval_results``.
    """
    import math as _math

    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins.wikitext_ppl import WikitextPplPlugin

    rows = ["the quick brown fox jumps over the lazy dog " * 8 for _ in range(10)]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module(rows))

    ctx = PipelineContext()
    ctx.set("model", _eager(tiny_model))
    ctx.set("tokenizer", _FakeTokenizer())
    ctx.set("config", {
        "stage6_validate": {
            "ppl_batch_size": 2,
            "wikitext2": {
                "enabled": True,
                "dataset": "wikitext", "subset": "wikitext-2-raw-v1",
                "split": "test", "sequence_length": 2048,
            },
        }
    })
    ctx.set("dataset_revisions", {})
    eval_results: dict = {}
    ctx.set("eval_results", eval_results)

    plugin = WikitextPplPlugin()
    plugin.eval_task(ctx)

    out = ctx.get("eval_results")
    assert out is eval_results, "eval_task must mutate the existing dict in place"
    assert "wikitext2_ppl" in out
    assert isinstance(out["wikitext2_ppl"], float)
    assert _math.isfinite(out["wikitext2_ppl"])
