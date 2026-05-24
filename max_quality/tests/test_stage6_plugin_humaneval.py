"""S6-4 -- Stage 6 HumanEval pass@1 plugin extraction tests.

Verifies the S6-4 ``HumanEvalPlugin`` scaffolding in
``stage6/plugins/humaneval.py``:

* the Pattern-A symbols ``_humaneval`` + ``_check_humaneval`` import from the
  plugin module;
* the ``stage6_validate`` monolith re-exports the SAME relocated objects (the
  ``# noqa: F401`` re-import block is load-bearing);
* ``HumanEvalPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, gates on ``stage6_validate.generative.enabled`` +
  a ``humaneval`` sub-key, and exposes the (S6-8) ``eval_task`` phase hook;
* the module never imports the ``stage6_validate`` monolith or
  ``stage6.orchestrator`` at any scope (the circular-import contract);
* the relocated ``_check_humaneval`` grades a known-safe completion correctly;
* the inert ``eval_task`` hook lands the ``humaneval_pass_at_1`` key in
  ``eval_results``.

S6-4 covers a MIXED pattern: ``_humaneval`` / ``_check_humaneval`` are
relocated verbatim (the monolith re-imports them); the inline ``run()``
student-side call site is reproduced in the inert ``eval_task`` hook (the
monolith ``run()`` is NOT modified for it). The byte-identical behavioral gate
is the S6-0 golden snapshot (``test_stage6_golden_snapshot.py``); this file
only checks the relocation plumbing and the relocated logic.

SECURITY: ``_check_humaneval`` runs code in a daemon thread. These tests pass
ONLY the test's own known-safe snippets (a trivial ``def add(a, b)``). No
model-generated or untrusted code is run, no real generation runs, no datasets
are fetched.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6.plugins import humaneval  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6 humaneval imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_humaneval_module_imports():
    """``_humaneval`` + ``_check_humaneval`` + ``HumanEvalPlugin`` import."""
    from moe_compress.stage6.plugins.humaneval import (
        HumanEvalPlugin,
        _check_humaneval,
        _humaneval,
    )

    assert isinstance(HumanEvalPlugin, type)
    assert callable(_humaneval)
    assert callable(_check_humaneval)


def test_monolith_reexports_humaneval():
    """The monolith re-exports the SAME relocated ``_humaneval`` /
    ``_check_humaneval`` objects (``is``-identity).

    Proves the ``# noqa: F401`` re-import block in ``stage6_validate.py`` keeps
    ``run()`` and external callers/tests on their original import path.
    """
    from moe_compress import stage6_validate
    from moe_compress.stage6.plugins import humaneval as he

    assert stage6_validate._humaneval is he._humaneval
    assert stage6_validate._check_humaneval is he._check_humaneval


def test_plugin_satisfies_protocol():
    """``HumanEvalPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6.plugins.humaneval import HumanEvalPlugin

    assert isinstance(HumanEvalPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata -- name / config_key / tuple-typed reads-writes-provides."""
    from moe_compress.stage6.plugins.humaneval import HumanEvalPlugin

    plugin = HumanEvalPlugin()
    assert plugin.name == "humaneval"
    assert plugin.config_key == "stage6_validate.generative.enabled"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.reads == (
        "model",
        "tokenizer",
        "config",
        "dataset_revisions",
        "device",
        "eval_text_concat",
        "eval_results",
        "pre_compile_forward",
        "experts_implementation_generative",
    )
    assert plugin.writes == ("eval_results",)
    # eval_results is a shared collector (in `writes`), not a calibration-pass
    # accumulator -- `provides` is empty.
    assert plugin.provides == ()
    assert "eval_results" in plugin.writes


def test_plugin_is_enabled_gating():
    """``is_enabled`` gates on ``generative.enabled`` AND a ``humaneval`` key.

    Empty config -> False; ``enabled=True`` + ``humaneval`` key -> True;
    ``enabled=False`` -> False; ``enabled=True`` with NO ``humaneval`` key ->
    False (mirrors the monolith's ``if "humaneval" in s6["generative"]`` gate).
    """
    from moe_compress.stage6.plugins.humaneval import HumanEvalPlugin

    plugin = HumanEvalPlugin()
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled(
        {"stage6_validate": {"generative": {"enabled": True, "humaneval": {}}}}
    ) is True
    assert plugin.is_enabled(
        {"stage6_validate": {"generative": {"enabled": False, "humaneval": {}}}}
    ) is False
    # enabled=True but no humaneval sub-key -> still disabled.
    assert plugin.is_enabled(
        {"stage6_validate": {"generative": {"enabled": True, "math500": {}}}}
    ) is False


def test_plugin_has_eval_task_hook():
    """The S6-8 phase hook ``eval_task`` is present and callable."""
    from moe_compress.stage6.plugins.humaneval import HumanEvalPlugin

    plugin = HumanEvalPlugin()
    assert callable(getattr(plugin, "eval_task", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6_validate`` / ``stage6.orchestrator``.

    The circular-import contract forbids importing the monolith (or the
    orchestrator) at any scope -- the monolith re-imports *this* module at load
    time. Parse the source with ``ast`` and walk the FULL tree so a
    function-local forbidden import cannot slip past. For ``ImportFrom`` both
    ``node.module`` and each imported name (plus its ``asname``) are checked.
    """
    from moe_compress.stage6.plugins import humaneval as mod

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


def test_check_humaneval_correct_solution():
    """``_check_humaneval`` returns True for a correct, known-safe completion.

    SECURITY: the snippet run here is the test's OWN trivial ``def add`` plus a
    passing assertion -- no model output, no untrusted code.
    """
    from moe_compress.stage6.plugins.humaneval import _check_humaneval

    prompt = "def add(a, b):\n"
    completion = (
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    test_src = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
    assert _check_humaneval(prompt, completion, test_src, "add") is True


def test_check_humaneval_wrong_solution():
    """``_check_humaneval`` returns False when the completion fails the test.

    SECURITY: the snippet run here is the test's OWN ``def add`` (wrong body)
    plus an assertion -- no model output, no untrusted code.
    """
    from moe_compress.stage6.plugins.humaneval import _check_humaneval

    prompt = "def add(a, b):\n"
    completion = (
        "```python\n"
        "def add(a, b):\n"
        "    return a - b\n"   # wrong
        "```\n"
    )
    test_src = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
    assert _check_humaneval(prompt, completion, test_src, "add") is False


def test_eval_task_hook_writes_eval_results(monkeypatch):
    """The inert ``eval_task`` hook lands ``humaneval_pass_at_1`` in
    ``eval_results``.

    Monkeypatches the plugin module's ``_humaneval`` to a stub returning 0.5
    (no real generation / dataset download), builds a ``PipelineContext`` with
    a pre-existing ``eval_results={}`` slot, calls the hook, and asserts the
    key was added. The hook mutates the existing dict -- it does NOT ``ctx.set``
    ``eval_results``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins import humaneval as he

    def _stub_humaneval(*_args, **_kwargs):
        return 0.5

    monkeypatch.setattr(he, "_humaneval", _stub_humaneval)

    ctx = PipelineContext()
    ctx.set("model", object())
    ctx.set("tokenizer", object())
    ctx.set("config", {
        "stage6_validate": {
            "gen_batch_size": 8,
            "generative": {
                "enabled": True,
                "humaneval": {},
            },
        }
    })
    ctx.set("dataset_revisions", {})
    eval_results = {}
    ctx.set("eval_results", eval_results)

    plugin = he.HumanEvalPlugin()
    plugin.eval_task(ctx)

    out = ctx.get("eval_results")
    assert out is eval_results, "eval_task must mutate the existing dict in place"
    assert out["humaneval_pass_at_1"] == 0.5
