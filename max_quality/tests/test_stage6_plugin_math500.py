"""S6-4 — Stage 6 MATH-500 accuracy plugin extraction tests.

Verifies the S6-4 ``Math500Plugin`` scaffolding in
``stage6/plugins/math500.py``:

* the Pattern-A symbols ``_math500`` + the boxed-answer grading helpers import
  from the plugin module;
* the ``stage6_validate`` monolith re-exports the SAME relocated objects (the
  ``# noqa: F401`` re-import block is load-bearing);
* ``Math500Plugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, gates on ``stage6_validate.generative.enabled`` +
  a ``math500`` sub-key, and exposes the (S6-8) ``eval_task`` phase hook;
* the module never imports the ``stage6_validate`` monolith or
  ``stage6.orchestrator`` at any scope (the circular-import contract);
* the relocated ``_extract_boxed`` / ``_check_math`` grade genuine inputs;
* the inert ``eval_task`` hook lands the ``math500_accuracy`` key in
  ``eval_results``.

S6-4 covers a MIXED pattern: ``_math500`` and its grading helpers are
relocated verbatim (the monolith re-imports them); the inline ``run()``
student-side call site is reproduced in the inert ``eval_task`` hook (the
monolith ``run()`` is NOT modified for it). The byte-identical behavioral gate
is the S6-0 golden snapshot (``test_stage6_golden_snapshot.py``); this file
only checks the relocation plumbing and the relocated logic.

All tests are CPU-only; no real generation runs and no datasets are fetched.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6.plugins import math500  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6 math500 imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_math500_module_imports():
    """``_math500`` + grading helpers + ``Math500Plugin`` import."""
    from moe_compress.stage6.plugins.math500 import (
        Math500Plugin,
        _check_math,
        _extract_boxed,
        _last_numeric,
        _math500,
        _math_fallback_extract,
    )

    assert isinstance(Math500Plugin, type)
    assert callable(_math500)
    assert callable(_extract_boxed)
    assert callable(_last_numeric)
    assert callable(_check_math)
    assert callable(_math_fallback_extract)


def test_monolith_reexports_math500():
    """The monolith re-exports the SAME relocated objects (``is``-identity).

    Proves the ``# noqa: F401`` re-import block in ``stage6_validate.py`` keeps
    ``run()`` and external callers/tests on their original import path.
    """
    from moe_compress import stage6_validate
    from moe_compress.stage6.plugins import math500 as m5

    assert stage6_validate._math500 is m5._math500
    assert stage6_validate._check_math is m5._check_math
    assert stage6_validate._extract_boxed is m5._extract_boxed
    assert stage6_validate._last_numeric is m5._last_numeric
    assert stage6_validate._math_fallback_extract is m5._math_fallback_extract


def test_plugin_satisfies_protocol():
    """``Math500Plugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6.plugins.math500 import Math500Plugin

    assert isinstance(Math500Plugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed reads-writes-provides."""
    from moe_compress.stage6.plugins.math500 import Math500Plugin

    plugin = Math500Plugin()
    assert plugin.name == "math500"
    assert plugin.config_key == "stage6_validate.generative.enabled"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    # C1: pre_compile_forward + experts_implementation_generative are consumed
    # by `eval_task` before generate() (eval_environment.py L498-518 contract).
    assert plugin.reads == (
        "model",
        "tokenizer",
        "config",
        "dataset_revisions",
        "pre_compile_forward",
        "experts_implementation_generative",
    )
    assert plugin.writes == ("eval_results",)
    # eval_results is a shared collector (in `writes`), not a calibration-pass
    # accumulator — `provides` is empty.
    assert plugin.provides == ()
    assert "eval_results" in plugin.writes


def test_plugin_is_enabled_gating():
    """``is_enabled`` gates on ``generative.enabled`` AND a ``math500`` key.

    Empty config → False; ``enabled=True`` + ``math500`` key → True;
    ``enabled=False`` → False; ``enabled=True`` with NO ``math500`` key →
    False (mirrors the monolith's ``if "math500" in s6["generative"]`` gate).
    """
    from moe_compress.stage6.plugins.math500 import Math500Plugin

    plugin = Math500Plugin()
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled(
        {"stage6_validate": {"generative": {"enabled": True, "math500": {}}}}
    ) is True
    assert plugin.is_enabled(
        {"stage6_validate": {"generative": {"enabled": False, "math500": {}}}}
    ) is False
    # enabled=True but no math500 sub-key → still disabled.
    assert plugin.is_enabled(
        {"stage6_validate": {"generative": {"enabled": True, "humaneval": {}}}}
    ) is False


def test_plugin_has_eval_task_hook():
    """The S6-8 phase hook ``eval_task`` is present and callable."""
    from moe_compress.stage6.plugins.math500 import Math500Plugin

    plugin = Math500Plugin()
    assert callable(getattr(plugin, "eval_task", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6_validate`` / ``stage6.orchestrator``.

    The circular-import contract forbids importing the monolith (or the
    orchestrator) at any scope — the monolith re-imports *this* module at load
    time. Parse the source with ``ast`` and walk the FULL tree so a
    function-local forbidden import cannot slip past. For ``ImportFrom`` both
    ``node.module`` and each imported name (plus its ``asname``) are checked.
    """
    from moe_compress.stage6.plugins import math500 as mod

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


def test_extract_boxed_basic():
    """``_extract_boxed`` extracts a simple \\boxed{...} value."""
    from moe_compress.stage6.plugins.math500 import _extract_boxed

    assert _extract_boxed(r"The answer is \boxed{42}.") == "42"
    # No \boxed{} → None.
    assert _extract_boxed("no box here") is None


def test_extract_boxed_nested():
    """``_extract_boxed`` handles nested braces via balanced-brace scanning."""
    from moe_compress.stage6.plugins.math500 import _extract_boxed

    assert _extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    # The LAST \boxed{} wins when there are several.
    assert _extract_boxed(r"\boxed{1} then \boxed{2}") == "2"


def test_check_math_exact_match():
    """``_check_math`` grades a correct boxed answer as True."""
    from moe_compress.stage6.plugins.math500 import _check_math

    completion = r"After working it out, \boxed{7}."
    reference = r"\boxed{7}"
    assert _check_math(completion, reference) is True


def test_check_math_wrong_answer():
    """``_check_math`` grades an incorrect boxed answer as False."""
    from moe_compress.stage6.plugins.math500 import _check_math

    completion = r"I think the answer is \boxed{8}."
    reference = r"\boxed{7}"
    assert _check_math(completion, reference) is False


def test_check_math_no_boxed_fallback():
    """Completion without ``\\boxed{}`` forces ``_math_fallback_extract`` —
    the last-numeric / LaTeX-fallback grader path, not the exact-match path.

    The reference is the boxed reference; the completion is a free-form prose
    answer with the numeric value at the end. ``_check_math`` strips its own
    boxed wrapper, fails to find one in the completion, then defers to the
    fallback. The fallback's ``_last_numeric`` finds the trailing integer and
    grades it against the (boxed-extracted) reference."""
    from moe_compress.stage6.plugins.math500 import _check_math

    # Completion has no \boxed{}; the trailing numeric should match the
    # reference's boxed value via the fallback path.
    assert _check_math("After some work, the answer is 7", r"\boxed{7}") is True
    # Same shape, wrong numeric → fallback grades False.
    assert _check_math("After some work, the answer is 8", r"\boxed{7}") is False


def test_eval_task_hook_writes_eval_results(monkeypatch):
    """The inert ``eval_task`` hook lands ``math500_accuracy`` in ``eval_results``.

    Monkeypatches the plugin module's ``_math500`` to a stub returning 0.5 (no
    real generation / dataset download), builds a ``PipelineContext`` with a
    pre-existing ``eval_results={}`` slot, calls the hook, and asserts the key
    was added. The hook mutates the existing dict — it does NOT ``ctx.set``
    ``eval_results``.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins import math500 as m5

    def _stub_math500(*_args, **_kwargs):
        return 0.5

    monkeypatch.setattr(m5, "_math500", _stub_math500)

    ctx = PipelineContext()
    ctx.set("model", object())
    ctx.set("tokenizer", object())
    ctx.set("config", {
        "stage6_validate": {
            "gen_batch_size": 8,
            "generative": {
                "enabled": True,
                "math500": {},
            },
        }
    })
    ctx.set("dataset_revisions", {})
    eval_results: dict = {}
    ctx.set("eval_results", eval_results)

    plugin = m5.Math500Plugin()
    plugin.eval_task(ctx)

    out = ctx.get("eval_results")
    assert out is eval_results, "eval_task must mutate the existing dict in place"
    assert out["math500_accuracy"] == 0.5
