"""S6-7 — Stage 6 validation-report plugin extraction tests.

Verifies the S6-7 ``ValidationReportPlugin`` scaffolding in
``stage6/plugins/validation_report.py``:

* the 3 Pattern-A function symbols (``_deltas``, ``_measured_reduction``,
  ``_check_thresholds``) import from the plugin module;
* the ``stage6_validate`` monolith re-exports the SAME relocated function
  objects (the ``# noqa: F401`` re-import block is load-bearing);
* ``ValidationReportPlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries correct metadata, has an unconditional ``is_enabled``
  and exposes the (S6-8) ``assemble_report`` phase hook;
* the module never imports the ``stage6_validate`` monolith or
  ``stage6.orchestrator`` at any scope (the circular-import contract);
* ``_deltas`` master plan §8 NaN/Inf hotspot — every branch preserved
  character-identical;
* ``_measured_reduction`` direct branches (t_total=0 early return,
  normal path, t_expert=0 → expert_reduction_ratio=None);
* ``_check_thresholds`` direct branches including the S6-0 golden's
  exact six ``skipped_checks`` key names;
* the inert ``assemble_report`` hook writes ``stage6_eval.json`` with the
  expected shape and flattens the correct scalars to Trackio.

S6-7 covers a MIXED pattern: the 3 standalone functions are relocated
verbatim (the monolith re-imports them); the inline final-block of the
monolith ``run()`` (results dict assembly + JSON write + Trackio flatten)
is reproduced in the inert ``assemble_report`` hook (the monolith
``run()`` is NOT modified for it). The byte-identical behavioral gate is
the S6-0 golden snapshot (``test_stage6_golden_snapshot.py``); this file
only checks the relocation plumbing and the relocated logic.
"""
from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.stage6.plugins import validation_report  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6 validation-report imports unavailable: {e}",
        allow_module_level=True,
    )


def test_validation_report_module_imports():
    """All 3 Pattern-A functions + ``ValidationReportPlugin`` import."""
    from moe_compress.stage6.plugins.validation_report import (
        ValidationReportPlugin,
        _check_thresholds,
        _deltas,
        _measured_reduction,
    )

    assert isinstance(ValidationReportPlugin, type)
    for fn in (_deltas, _measured_reduction, _check_thresholds):
        assert callable(fn)


def test_monolith_reexports_pattern_a_functions():
    """The monolith re-exports the SAME relocated FUNCTION objects.

    Proves the S6-7 ``# noqa: F401`` re-import block in
    ``stage6_validate.py`` keeps ``run()`` and external callers/tests on
    their original import path. ``is``-identity checks for all 3
    relocated functions.
    """
    from moe_compress import stage6_validate
    from moe_compress.stage6.plugins import validation_report

    for name in ("_deltas", "_measured_reduction", "_check_thresholds"):
        assert getattr(stage6_validate, name) is getattr(validation_report, name), (
            f"monolith re-export mismatch for {name}"
        )


def test_plugin_satisfies_protocol():
    """``ValidationReportPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6.plugins.validation_report import ValidationReportPlugin

    assert isinstance(ValidationReportPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed fields / writes slots."""
    from moe_compress.stage6.plugins.validation_report import ValidationReportPlugin

    plugin = ValidationReportPlugin()
    assert plugin.name == "validation_report"
    assert plugin.config_key == "stage6_validate"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert plugin.provides == ()
    assert "stage6_results_path" in plugin.writes
    assert "overall_pass" in plugin.writes


def test_plugin_is_enabled_unconditional():
    """``is_enabled`` returns ``True`` regardless of config — the report
    is the deliverable of Stage 6 (even when every sub-eval is disabled
    the S6-0 golden's ``stage6_eval.json`` is still emitted).
    """
    from moe_compress.stage6.plugins.validation_report import ValidationReportPlugin

    plugin = ValidationReportPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled({"stage6_validate": {}}) is True
    assert plugin.is_enabled(
        {"stage6_validate": {"wikitext2": {"enabled": False}}}
    ) is True


def test_plugin_has_assemble_report_hook():
    """The S6-8 phase hook (``assemble_report``) is present + callable."""
    from moe_compress.stage6.plugins.validation_report import ValidationReportPlugin

    plugin = ValidationReportPlugin()
    assert callable(getattr(plugin, "assemble_report", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6_validate`` / ``stage6.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or
    the orchestrator) at any scope — module-top OR function-local — since
    either would risk an import cycle (the monolith re-imports *this*
    module at load time). Parse the source with ``ast`` and walk the FULL
    tree so a function-local ``import stage6_validate`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported
    symbols) — so the cycle-causing
    ``from moe_compress import stage6_validate`` form is also caught.

    Each alias's ``asname`` is checked alongside its ``name`` so a
    renamed import cannot slip past the name check either.
    """
    from moe_compress.stage6.plugins import validation_report as mod

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


# ---------------------------------------------------------------------------
# _deltas — master plan §8 NaN/Inf hotspot
# ---------------------------------------------------------------------------


def test_deltas_empty_both_returns_empty():
    """S6-0 golden's path: both sides empty → ``{}``.

    The golden ``stage6_eval.json`` pins ``"delta": {}`` for the
    all-disabled Stage 6 run; this test guards that exact behaviour.
    """
    from moe_compress.stage6.plugins.validation_report import _deltas

    assert _deltas({}, {}) == {}


def test_deltas_normal_path_computes_triple():
    """Both operands finite → ``{student, teacher, delta}`` triple."""
    from moe_compress.stage6.plugins.validation_report import _deltas

    out = _deltas({"wikitext2_ppl": 12.5}, {"wikitext2_ppl": 10.0})
    assert "wikitext2_ppl" in out
    triple = out["wikitext2_ppl"]
    assert triple["student"] == pytest.approx(12.5)
    assert triple["teacher"] == pytest.approx(10.0)
    assert triple["delta"] == pytest.approx(2.5)
    assert "_non_finite_skipped" not in out
    assert "_teacher_non_finite_skipped" not in out


def test_deltas_student_nan_records_non_finite_sentinel():
    """student value NaN → ``_non_finite_skipped`` contains the key."""
    from moe_compress.stage6.plugins.validation_report import _deltas

    out = _deltas({"wikitext2_ppl": float("nan")}, {"wikitext2_ppl": 10.0})
    assert "wikitext2_ppl" not in out
    assert out["_non_finite_skipped"] == ["wikitext2_ppl"]


def test_deltas_student_inf_records_non_finite_sentinel():
    """student value +inf → ``_non_finite_skipped`` contains the key."""
    from moe_compress.stage6.plugins.validation_report import _deltas

    out = _deltas({"wikitext2_ppl": float("inf")}, {"wikitext2_ppl": 10.0})
    assert "wikitext2_ppl" not in out
    assert out["_non_finite_skipped"] == ["wikitext2_ppl"]


def test_deltas_teacher_nan_records_teacher_non_finite_sentinel():
    """teacher value NaN → ``_teacher_non_finite_skipped`` contains the key."""
    from moe_compress.stage6.plugins.validation_report import _deltas

    out = _deltas({"wikitext2_ppl": 12.5}, {"wikitext2_ppl": float("nan")})
    assert "wikitext2_ppl" not in out
    assert out["_teacher_non_finite_skipped"] == ["wikitext2_ppl"]


def test_deltas_teacher_inf_records_teacher_non_finite_sentinel():
    """teacher value +inf → ``_teacher_non_finite_skipped`` contains the key."""
    from moe_compress.stage6.plugins.validation_report import _deltas

    out = _deltas({"wikitext2_ppl": 12.5}, {"wikitext2_ppl": float("inf")})
    assert "wikitext2_ppl" not in out
    assert out["_teacher_non_finite_skipped"] == ["wikitext2_ppl"]


def test_deltas_missing_key_skipped():
    """Key present on only one side (s or t is None) → key absent from result."""
    from moe_compress.stage6.plugins.validation_report import _deltas

    # Only on student side
    out = _deltas({"only_student": 1.0}, {"wikitext2_ppl": 10.0})
    assert "only_student" not in out
    # The wikitext2_ppl entry has teacher=10.0 but no student (None) → also absent.
    assert "wikitext2_ppl" not in out
    # Only on teacher side
    out2 = _deltas({"x": 1.0}, {"only_teacher": 1.0})
    assert "only_teacher" not in out2
    assert "x" not in out2


def test_deltas_non_numeric_value_skipped():
    """Non-numeric value (TypeError in math.isfinite) → key absent."""
    from moe_compress.stage6.plugins.validation_report import _deltas

    out = _deltas({"foo": "not-a-number"}, {"foo": 1.0})
    assert "foo" not in out
    assert "_non_finite_skipped" not in out
    assert "_teacher_non_finite_skipped" not in out


def test_deltas_defensive_delta_non_finite_branch_present():
    """The defensive ``not math.isfinite(delta)`` branch is PRESENT in
    the relocated body.

    The branch is unreachable in IEEE 754 (math.isfinite(a-b) is True
    whenever both a and b are finite), but the master plan §8 hotspot
    contract requires preserving every branch character-identical.
    Inspect the source to confirm the defensive guard literal is still
    there.
    """
    from moe_compress.stage6.plugins.validation_report import _deltas

    src = inspect.getsource(_deltas)
    assert "if not math.isfinite(delta):" in src, (
        "defensive `not math.isfinite(delta)` branch missing from _deltas"
    )
    # Also confirm the comment that pins this is a defensive (unreachable-in-IEEE-754) guard.
    assert "inf - inf" in src


# ---------------------------------------------------------------------------
# _measured_reduction — direct branches
# ---------------------------------------------------------------------------


def test_measured_reduction_t_total_zero_early_return():
    """S6-0 golden's path: cached teacher counts = (0, 0) → ratios=None.

    The golden ``stage6_eval.json`` pins
    ``"total_reduction_ratio": null`` + ``"expert_reduction_ratio": null``
    + ``"total_teacher": 0`` for the all-disabled Stage 6 run; this
    test guards the early-return shape that produces it. We pass
    ``student_model=None`` + explicit student counts so the function
    never touches any nn.Module.
    """
    from moe_compress.stage6.plugins.validation_report import _measured_reduction

    out = _measured_reduction(
        None,
        student_total=5024,
        student_expert=3072,
        teacher_model=None,
        cached_teacher_param_counts={"total": 0, "expert": 0},
        config=None,
    )
    assert out == {
        "total_student": 5024,
        "total_teacher": 0,
        "total_reduction_ratio": None,
        "expert_student": 3072,
        "expert_teacher": 0,
        "expert_reduction_ratio": None,
    }


def test_measured_reduction_normal_path():
    """Normal computation with realistic counts → numeric ratios."""
    from moe_compress.stage6.plugins.validation_report import _measured_reduction

    out = _measured_reduction(
        None,
        student_total=8_000,
        student_expert=4_000,
        teacher_model=None,
        cached_teacher_param_counts={"total": 10_000, "expert": 5_000},
        config=None,
    )
    assert out["total_student"] == 8_000
    assert out["total_teacher"] == 10_000
    assert out["total_reduction_ratio"] == pytest.approx(0.2)
    assert out["expert_student"] == 4_000
    assert out["expert_teacher"] == 5_000
    assert out["expert_reduction_ratio"] == pytest.approx(0.2)


def test_measured_reduction_t_expert_zero():
    """t_expert == 0 (non-MoE teacher) → expert_reduction_ratio is None."""
    from moe_compress.stage6.plugins.validation_report import _measured_reduction

    out = _measured_reduction(
        None,
        student_total=8_000,
        student_expert=0,
        teacher_model=None,
        cached_teacher_param_counts={"total": 10_000, "expert": 0},
        config=None,
    )
    # t_total > 0 → total_reduction_ratio computed normally
    assert out["total_reduction_ratio"] == pytest.approx(0.2)
    # t_expert == 0 → expert_reduction_ratio is None (not 1.0)
    assert out["expert_reduction_ratio"] is None


# ---------------------------------------------------------------------------
# _check_thresholds — direct branches
# ---------------------------------------------------------------------------


_GOLDEN_SKIPPED_KEYS = frozenset({
    "arc_challenge_acc_drop_ok",
    "hellaswag_acc_drop_ok",
    "humaneval_pass_at_1_drop_ok",
    "math500_accuracy_drop_ok",
    "measured_reduction_ok",
    "wikitext2_ppl_increase_ok",
})


def test_check_thresholds_all_disabled():
    """S6-0 golden scenario — all sub-evals disabled, results empty.

    Asserts ``skipped_checks`` contains EXACTLY the 6 key names the
    golden ``stage6_eval.json`` pins (and no other boolean check is
    produced, since every metric branch hits the eval-disabled skip
    path).
    """
    from moe_compress.stage6.plugins.validation_report import _check_thresholds

    results = {
        "student": {},
        "teacher": {},
        "delta": {},
        "measured_reduction": {
            "total_reduction_ratio": None,
            "expert_reduction_ratio": None,
        },
    }
    thresholds = {
        "wikitext2_ppl_relative_max_increase": 0.03,
        "arc_c_absolute_max_drop": 0.015,
        "hellaswag_absolute_max_drop": 0.015,
        "humaneval_absolute_max_drop": 0.015,
        "math500_absolute_max_drop": 0.015,
        "measured_reduction_min": 0.30,
    }
    s6_cfg = {
        "wikitext2": {"enabled": False},
        "zero_shot": {"enabled": False},
        "generative": {"enabled": False},
    }
    out = _check_thresholds(results, thresholds, s6_cfg=s6_cfg)
    assert "skipped_checks" in out
    assert frozenset(out["skipped_checks"].keys()) == _GOLDEN_SKIPPED_KEYS
    # No boolean keys should be present (every check hit a skip branch).
    bool_keys = [k for k, v in out.items() if isinstance(v, bool)]
    assert bool_keys == []


def test_check_thresholds_ppl_pass():
    """wikitext2_ppl: 3% increase ≤ 3% threshold → PASS."""
    from moe_compress.stage6.plugins.validation_report import _check_thresholds

    results = {
        "delta": {
            "wikitext2_ppl": {"student": 10.3, "teacher": 10.0, "delta": 0.3},
        },
        "measured_reduction": {},
    }
    out = _check_thresholds(
        results, {"wikitext2_ppl_relative_max_increase": 0.03}, s6_cfg={},
    )
    assert out.get("wikitext2_ppl_increase_ok") is True


def test_check_thresholds_ppl_fail():
    """wikitext2_ppl: 10% increase > 3% threshold → FAIL."""
    from moe_compress.stage6.plugins.validation_report import _check_thresholds

    results = {
        "delta": {
            "wikitext2_ppl": {"student": 11.0, "teacher": 10.0, "delta": 1.0},
        },
        "measured_reduction": {},
    }
    out = _check_thresholds(
        results, {"wikitext2_ppl_relative_max_increase": 0.03}, s6_cfg={},
    )
    assert out.get("wikitext2_ppl_increase_ok") is False


def test_check_thresholds_student_non_finite_auto_fail():
    """student PPL non-finite → auto-FAIL (NOT in skipped_checks).

    H3 / M5 contract: a non-finite student value is an automatic failure
    regardless of whether a threshold is configured. The check key must
    be False on the boolean side, NOT a string reason on skipped_checks.
    """
    from moe_compress.stage6.plugins.validation_report import _check_thresholds

    results = {
        "delta": {"_non_finite_skipped": ["wikitext2_ppl"]},
        "measured_reduction": {},
    }
    out = _check_thresholds(results, {}, s6_cfg={})
    # Auto-fail on the boolean side; not in skipped_checks.
    assert out.get("wikitext2_ppl_increase_ok") is False
    assert "wikitext2_ppl_increase_ok" not in out.get("skipped_checks", {})


def test_check_thresholds_teacher_non_finite_skip():
    """teacher PPL non-finite → skip (in skipped_checks, not in boolean checks).

    M-1 contract: teacher non-finite is a teacher issue, not a student
    failure — record under skipped_checks rather than auto-failing.
    """
    from moe_compress.stage6.plugins.validation_report import _check_thresholds

    results = {
        "delta": {"_teacher_non_finite_skipped": ["wikitext2_ppl"]},
        "measured_reduction": {},
    }
    out = _check_thresholds(results, {}, s6_cfg={})
    # Not on the boolean side.
    assert "wikitext2_ppl_increase_ok" not in {
        k for k, v in out.items() if isinstance(v, bool)
    }
    # In the skipped_checks sub-dict with a teacher-issue reason.
    sk = out.get("skipped_checks", {})
    assert "wikitext2_ppl_increase_ok" in sk
    assert "teacher" in sk["wikitext2_ppl_increase_ok"]


def test_check_thresholds_measured_reduction_pass():
    """measured_reduction passes when ratio ≥ threshold."""
    from moe_compress.stage6.plugins.validation_report import _check_thresholds

    results = {
        "delta": {},
        "measured_reduction": {"total_reduction_ratio": 0.5},
    }
    out = _check_thresholds(
        results, {"measured_reduction_min": 0.3}, s6_cfg={},
    )
    assert out.get("measured_reduction_ok") is True


def test_check_thresholds_measured_reduction_none_skip():
    """measured_reduction ratio=None → skipped, not failed."""
    from moe_compress.stage6.plugins.validation_report import _check_thresholds

    results = {
        "delta": {},
        "measured_reduction": {"total_reduction_ratio": None},
    }
    out = _check_thresholds(
        results, {"measured_reduction_min": 0.3}, s6_cfg={},
    )
    # Not on the boolean side.
    assert "measured_reduction_ok" not in {
        k for k, v in out.items() if isinstance(v, bool)
    }
    # In skipped_checks with the param-count-failed reason.
    sk = out.get("skipped_checks", {})
    assert "measured_reduction_ok" in sk
    assert "total_reduction_ratio unavailable" in sk["measured_reduction_ok"]


# ---------------------------------------------------------------------------
# assemble_report — integration test for the inert Pattern-B hook
# ---------------------------------------------------------------------------


def test_assemble_report_hook(tmp_path, monkeypatch):
    """The inert ``assemble_report`` hook reproduces the monolith's final
    block: writes ``stage6_eval.json`` with the expected shape, flattens
    the metric scalars to Trackio, and publishes ``stage6_results_path``
    + ``overall_pass`` on the context.

    Uses the S6-0 golden's all-disabled scenario as the integration
    fixture so the hook output structurally matches what the monolith
    ``run()`` would produce on the same inputs.
    """
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.stage6.plugins import validation_report as _vr_mod
    from moe_compress.stage6.plugins.validation_report import ValidationReportPlugin

    captured: list[dict] = []
    monkeypatch.setattr(_vr_mod, "_trackio_log", lambda d: captured.append(dict(d)))

    config = {
        "stage6_validate": {
            "wikitext2": {"enabled": False},
            "zero_shot": {"enabled": False},
            "generative": {"enabled": False},
            "thresholds": {
                "wikitext2_ppl_relative_max_increase": 0.03,
                "arc_c_absolute_max_drop": 0.015,
                "hellaswag_absolute_max_drop": 0.015,
                "humaneval_absolute_max_drop": 0.015,
                "math500_absolute_max_drop": 0.015,
                "measured_reduction_min": 0.30,
            },
        },
    }
    plugin = ValidationReportPlugin()
    ctx = PipelineContext()
    ctx.set("config", config)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("student_results", {})
    ctx.set("teacher_results", {})
    ctx.set("student_param_counts", {"total": 5024, "expert": 3072})
    ctx.set("teacher_param_counts", {"total": 0, "expert": 0})

    plugin.assemble_report(ctx)

    # The JSON artifact lives at the expected path.
    out_path = tmp_path / "stage6_eval.json"
    assert out_path.exists()
    blob = json.loads(out_path.read_text())

    # Structural shape matches the S6-0 golden's all-disabled scenario.
    assert blob["overall_pass"] is False
    assert blob["delta"] == {}
    assert blob["student"] == {}
    assert blob["teacher"] == {}
    # measured_reduction present with the expected nullable ratios.
    assert "measured_reduction" in blob
    assert blob["measured_reduction"]["total_reduction_ratio"] is None
    assert blob["measured_reduction"]["expert_reduction_ratio"] is None
    # thresholds.skipped_checks present and matches the golden's exact keys.
    assert "thresholds" in blob
    assert "skipped_checks" in blob["thresholds"]
    assert frozenset(blob["thresholds"]["skipped_checks"].keys()) == _GOLDEN_SKIPPED_KEYS

    # Trackio: at least one log call happened and the last one carried
    # the overall_pass scalar key.
    assert len(captured) >= 1
    assert "stage6/overall_pass" in captured[-1]
    # non_finite_count surfaced as expected (zero on the empty-delta path).
    assert captured[-1].get("stage6/non_finite_count") == 0.0

    # Context publishes the path + overall_pass for downstream consumers.
    assert ctx.get("stage6_results_path") == out_path
    assert isinstance(ctx.get("overall_pass"), bool)
    assert ctx.get("overall_pass") is False
