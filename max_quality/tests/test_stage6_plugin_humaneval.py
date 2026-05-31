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


# ---------------------------------------------------------------------------
# Item-2 — ProcessPool scoring (timeout-robustness, leak fix, order-independence)
# ---------------------------------------------------------------------------


def test_score_humaneval_one_picklable():
    """The spawn worker must be picklable by reference (no closure capture)."""
    import pickle

    from moe_compress.stage6.plugins._humaneval_worker import _score_humaneval_one

    loaded = pickle.loads(pickle.dumps(_score_humaneval_one))
    assert loaded is _score_humaneval_one


def test_humaneval_worker_module_is_torch_free():
    """Importing the worker leaf module must NOT pull torch into sys.modules.

    Run in a FRESH subprocess so the parent's already-imported torch does not
    mask a regression. This is the H1 spawn-cost guard: every spawn child
    re-imports the worker's defining module, so it must stay torch-free.
    """
    import subprocess
    import sys

    src = (
        "import sys; "
        "import moe_compress.stage6.plugins._humaneval_worker as w; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m); "
        "print('OK')"
    )
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True, text=True,
        env={"PYTHONPATH": repo_src, "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert proc.returncode == 0, (
        f"worker import pulled torch or failed:\nSTDOUT={proc.stdout}\nSTDERR={proc.stderr}"
    )
    assert "OK" in proc.stdout


def test_humaneval_pool_timeout_scores_false():
    """A hanging completion scores False without blocking the others, and no
    live worker child remains after the shared-deadline shutdown (leak fix)."""
    import multiprocessing

    from moe_compress.stage6.plugins.humaneval import _score_all_humaneval

    # Three problems: a fast-pass, a HANG (infinite loop), a fast-pass. The hang
    # must NOT prevent the two good problems from scoring True.
    good = (
        "def f():\n    return 1\n",          # raw stub
        "```python\ndef f():\n    return 1\n```\n",  # completion
        "def check(candidate):\n    assert candidate() == 1\n",  # test
        "f",                                  # entry point
    )
    hang = (
        "def g():\n",
        "```python\ndef g():\n    while True:\n        pass\n```\n",
        "def check(candidate):\n    candidate()\n",
        "g",
    )
    raw_prompts = [good[0], hang[0], good[0]]
    completions = [good[1], hang[1], good[1]]
    tests = [good[2], hang[2], good[2]]
    entry_points = [good[3], hang[3], good[3]]

    children_before = set(multiprocessing.active_children())
    passes = _score_all_humaneval(
        raw_prompts, completions, tests, entry_points,
        exec_timeout_secs=2,
    )
    # Two good problems pass; the hang scores False (terminated at the deadline).
    assert passes == 2

    # No worker child should survive the shutdown. Give the OS a brief moment to
    # reap terminated children, then assert nothing new is alive.
    import time as _time
    deadline = _time.monotonic() + 10
    while _time.monotonic() < deadline:
        leftover = set(multiprocessing.active_children()) - children_before
        if not leftover:
            break
        _time.sleep(0.1)
    leftover = set(multiprocessing.active_children()) - children_before
    assert not leftover, f"leaked worker processes after shutdown: {leftover}"


def test_humaneval_pool_order_independent():
    """passes/total is order-independent: shuffling the problem order yields the
    same pass count (greedy decode -> each score is a pure function)."""
    from moe_compress.stage6.plugins.humaneval import _score_all_humaneval

    pass_one = (
        "def f():\n",
        "```python\ndef f():\n    return 1\n```\n",
        "def check(candidate):\n    assert candidate() == 1\n",
        "f",
    )
    fail_one = (
        "def f():\n",
        "```python\ndef f():\n    return 2\n```\n",
        "def check(candidate):\n    assert candidate() == 1\n",
        "f",
    )
    problems = [pass_one, fail_one, pass_one, pass_one, fail_one]

    def _score(order):
        rp = [p[0] for p in order]
        co = [p[1] for p in order]
        te = [p[2] for p in order]
        ep = [p[3] for p in order]
        return _score_all_humaneval(rp, co, te, ep, exec_timeout_secs=10)

    forward = _score(problems)
    reversed_ = _score(list(reversed(problems)))
    assert forward == 3
    assert forward == reversed_


# ---------------------------------------------------------------------------
# Item-1 — gen_batch_size pin advisory WARN at all three parse sites
# ---------------------------------------------------------------------------


def test_generate_batched_docstring_scopes_invariance():
    """_generate_batched docstring must drop the false 'Numerically identical'
    claim and explicitly label generate() as NOT bit-identical (Item-1)."""
    from moe_compress.tools.eval_harness import _generate_batched

    doc = _generate_batched.__doc__ or ""
    assert "Numerically identical" not in doc
    assert "NOT bit-identical" in doc


def test_pinned_gen_batch_size_mirror_in_sync():
    """teacher_provider's local mirror must equal the canonical eval_harness
    constant (mirror-drift guard; tp may not cross-import it)."""
    from moe_compress.stage6.plugins import teacher_provider as tp
    from moe_compress.tools.eval_harness import PINNED_GEN_BATCH_SIZE

    assert tp.PINNED_GEN_BATCH_SIZE == PINNED_GEN_BATCH_SIZE


def _ctx_with_gen_batch_size(gbs):
    from moe_compress.pipeline.context import PipelineContext

    ctx = PipelineContext()
    ctx.set("model", object())
    ctx.set("tokenizer", object())
    ctx.set("config", {
        "stage6_validate": {
            "gen_batch_size": gbs,
            "generative": {"enabled": True, "humaneval": {}, "math500": {}},
        }
    })
    ctx.set("dataset_revisions", {})
    ctx.set("eval_results", {})
    return ctx


def _assert_off_pin_warns(caplog, logger_name, run_eval_task):
    """Run an eval_task with an off-pin gen_batch_size and assert the advisory
    WARN fired. The parse/WARN is emitted BEFORE any real generation, so we let
    the subsequent (dummy-model) generation blow up and only inspect caplog —
    no monkeypatching of production code."""
    import logging

    # Some pytest plugins in this env default new loggers to propagate=False;
    # the real plugin loggers have propagate=True, so mirror that here so the
    # caplog handler on the root logger receives the WARNING record (same
    # documented pattern as test_stage2_plugin_skip_merge_floor.py).
    logging.getLogger(logger_name).propagate = True

    with caplog.at_level(logging.WARNING, logger=logger_name):
        try:
            run_eval_task()
        except Exception:  # noqa: BLE001 — dummy model fails AFTER the warn
            pass
    assert any(
        "differs from the pinned generative geometry" in r.getMessage()
        for r in caplog.records
    ), f"expected an advisory WARN on off-pin gen_batch_size ({logger_name})"


def test_gen_batch_size_pin_warns_humaneval(caplog):
    """HumanEval eval_task emits an advisory WARN (no raise) when gen_batch_size
    differs from PINNED_GEN_BATCH_SIZE (Item-1, site 1)."""
    from moe_compress.stage6.plugins import humaneval as he

    ctx = _ctx_with_gen_batch_size(4)
    _assert_off_pin_warns(caplog, he.log.name,
                          lambda: he.HumanEvalPlugin().eval_task(ctx))


def test_gen_batch_size_pin_warns_math500(caplog):
    """MATH-500 eval_task emits the same advisory WARN; PINNED_GEN_BATCH_SIZE is
    imported DIRECTLY from eval_harness (Item-1, site 3)."""
    from moe_compress.stage6.plugins import math500 as m5

    ctx = _ctx_with_gen_batch_size(4)
    _assert_off_pin_warns(caplog, m5.log.name,
                          lambda: m5.Math500Plugin().eval_task(ctx))


def test_gen_batch_size_pin_warns_teacher_provider_source():
    """teacher_provider's parse site emits the SAME advisory WARN against its
    module-local PINNED_GEN_BATCH_SIZE mirror (Item-1, site 2).

    The live parse site sits after a real teacher-model load
    (`provide_teacher_side`), which a unit test must not trigger (GPU
    discipline / tiny-fixtures-only). We therefore assert the WARN is wired at
    the source level and that the mirror it compares against is in sync (the
    runtime warn text is identical to the two caplog-exercised sites above)."""
    import inspect

    from moe_compress.stage6.plugins import teacher_provider as tp

    src = inspect.getsource(tp.TeacherProviderPlugin.provide_teacher_side)
    assert "if gen_batch_size != PINNED_GEN_BATCH_SIZE:" in src
    assert "differs from the pinned generative geometry" in src
    # And the mirror it compares against equals the canonical constant.
    from moe_compress.tools.eval_harness import PINNED_GEN_BATCH_SIZE
    assert tp.PINNED_GEN_BATCH_SIZE == PINNED_GEN_BATCH_SIZE
