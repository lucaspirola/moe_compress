"""S6A-5 — Stage 6alt thermometer final-report plugin extraction tests.

Verifies the S6A-5 ``ThermoReportPlugin`` scaffolding in
``stage6alt/plugins/thermo_report.py``:

* the ``ThermoReportPlugin`` class imports from the plugin module;
* ``ThermoReportPlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries the declared metadata, is unconditionally enabled,
  and exposes the (S6A-6) ``assemble_thermo_report`` phase hook;
* the module never imports the ``stage6alt_thermometer`` monolith or
  ``stage6alt.orchestrator`` at any scope (the circular-import contract);
* the inert ``assemble_thermo_report`` hook produces a results dict with
  the exact 16 top-level keys pinned by the S6A-0 golden snapshot,
  computes ``bpt_gap`` correctly when both BPTs are finite, computes
  ``top1_agreement`` correctly when both argmax tensors match, leaves
  ``top1_agreement`` None on a shape mismatch, leaves ``bpt_gap`` None
  when the student BPT is non-finite, and publishes
  ``stage6alt_eval_path`` to ctx.

S6A-5 is **pure Pattern B**: no standalone helpers to relocate. The
final-assembly block is reproduced in the inert
``assemble_thermo_report`` hook (the monolith ``run()`` is NOT modified
for it). The byte-identical behavioral gate is the S6A-0 golden
snapshot (``test_stage6alt_golden_snapshot.py``); this file only checks
the plugin scaffolding and the reproduced logic on synthetic inputs.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

try:
    import torch
    from moe_compress.stage6alt.plugins import thermo_report  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(
        f"Stage 6alt thermo_report imports unavailable: {e}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _make_thermo_config(tiny_config) -> dict:
    """Augment ``tiny_config`` with a minimal thermometer sub-tree.

    The final-report hook re-derives ``arc_easy_limit`` / ``hellaswag_limit``
    from ``stage6_validate.thermometer`` to emit them into the
    ``lm_eval`` sub-dict of the results. The tiny_config fixture
    already provides ``stage6_validate``; we just add the thermometer
    sub-dict.
    """
    cfg = dict(tiny_config)
    cfg["stage6_validate"] = dict(cfg["stage6_validate"])
    cfg["stage6_validate"]["thermometer"] = {
        "bpt_batch_size": 4,
        "arc_easy_limit": 5,
        "hellaswag_limit": 7,
        "lm_eval_batch_size": "auto:4",
    }
    return cfg


def _seed_ctx(tmp_path, cfg, *, student_bpt=3.0, teacher_bpt=2.5,
              student_argmax=None, teacher_argmax=None,
              student_arc=0.45, student_hsw=0.55, student_acc_sum=1.0,
              teacher_arc=0.50, teacher_hsw=0.60, teacher_acc_sum=1.10):
    """Build a populated PipelineContext for the assemble hook.

    Mirrors the slots the upstream plugins
    (``BptMetricPlugin``, ``ZeroShotSubsetPlugin``,
    ``ThermoTeacherProviderPlugin``, ``ThermoCorpusPlugin``) would
    publish before the report assembly runs.
    """
    from moe_compress.pipeline.context import PipelineContext

    ctx = PipelineContext()
    ctx.set("config", cfg)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("student_bpt", student_bpt)
    ctx.set("student_argmax", student_argmax)
    ctx.set("student_arc_easy_acc_norm", student_arc)
    ctx.set("student_hellaswag_acc_norm", student_hsw)
    ctx.set("student_acc_norm_sum", student_acc_sum)
    ctx.set("teacher_results", {
        "teacher_bpt": teacher_bpt,
        "teacher_arc_easy_acc_norm": teacher_arc,
        "teacher_hellaswag_acc_norm": teacher_hsw,
        "teacher_acc_norm_sum": teacher_acc_sum,
        "teacher_argmax": teacher_argmax,
    })
    ctx.set("teacher_cache_hit", True)
    ctx.set("teacher_cache_path", tmp_path / "thermometer_teacher_cache.json")
    ctx.set("teacher_cache_key", "k" * 64)
    ctx.set("corpus_meta", {"name": "synthetic", "n_sequences": 4})
    return ctx


# ---------------------------------------------------------------------------
# Tests — module imports + protocol conformance + metadata
# ---------------------------------------------------------------------------


def test_thermo_report_module_imports():
    """``ThermoReportPlugin`` imports from the plugin module."""
    from moe_compress.stage6alt.plugins.thermo_report import ThermoReportPlugin

    assert isinstance(ThermoReportPlugin, type)


def test_plugin_satisfies_protocol():
    """``ThermoReportPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage6alt.plugins.thermo_report import ThermoReportPlugin

    assert isinstance(ThermoReportPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / config_key / tuple-typed fields / writes slot."""
    from moe_compress.stage6alt.plugins.thermo_report import ThermoReportPlugin

    plugin = ThermoReportPlugin()
    assert plugin.name == "thermo_report"
    assert plugin.config_key == "stage6_validate.thermometer"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    # provides=() — the final-report concern has no calibration-pass
    # accumulators, mirroring the sibling ThermoTeacherProviderPlugin and
    # ValidationReportPlugin convention.
    assert plugin.provides == ()
    assert "stage6alt_eval_path" in plugin.writes
    # Spot-check that the 12 reads slots declared by the plan are present.
    for slot in (
        "config", "artifacts_dir",
        "student_bpt", "student_argmax",
        "student_arc_easy_acc_norm", "student_hellaswag_acc_norm",
        "student_acc_norm_sum",
        "teacher_results", "teacher_cache_hit",
        "teacher_cache_path", "teacher_cache_key",
        "corpus_meta",
    ):
        assert slot in plugin.reads, f"missing reads slot: {slot}"


def test_plugin_is_enabled_unconditional():
    """``is_enabled`` is True for any config (the artifact is always emitted)."""
    from moe_compress.stage6alt.plugins.thermo_report import ThermoReportPlugin

    plugin = ThermoReportPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled({"stage6_validate": {}}) is True
    assert plugin.is_enabled({
        "stage6_validate": {"thermometer": {"arc_easy_limit": 5}}
    }) is True


def test_plugin_has_assemble_thermo_report_hook():
    """The S6A-6 phase hook ``assemble_thermo_report`` is present and callable."""
    from moe_compress.stage6alt.plugins.thermo_report import ThermoReportPlugin

    plugin = ThermoReportPlugin()
    assert callable(getattr(plugin, "assemble_thermo_report", None))


# ---------------------------------------------------------------------------
# Tests — circular-import AST guard
# ---------------------------------------------------------------------------


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage6alt_thermometer`` / orchestrator.

    The plugin docstring's contract says NEVER import the monolith (or
    the orchestrator) at any scope — module-top OR function-local —
    since either would risk an import cycle (the monolith re-imports
    *this* module at load time). Parse the source with ``ast`` and walk
    the FULL tree so a function-local ``import stage6alt_thermometer``
    cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` AND
    ``node.names`` — so the cycle-causing
    ``from moe_compress import stage6alt_thermometer`` form is also
    caught. Each alias's ``asname`` is checked alongside its ``name`` so
    a renamed import cannot slip past either.
    """
    from moe_compress.stage6alt.plugins import thermo_report as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = (
        "stage6alt_thermometer",
        "stage6alt.orchestrator",
    )
    for node in ast.walk(tree):
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
# Tests — inert ``assemble_thermo_report`` hook
# ---------------------------------------------------------------------------


def test_assemble_thermo_report_hook_json_shape(tmp_path, monkeypatch, tiny_config):
    """Hook produces a results dict with the exact 16 top-level keys.

    Patches ``save_json_artifact`` on the thermo_report module to capture
    the results dict in-memory rather than touching disk; builds a
    ``PipelineContext`` populated with all 12 required slots; calls the
    hook; asserts the captured dict matches the S6A-0 golden shape
    (16 top-level keys, ``stage=="6alt"`` / ``mode=="thermometer"``,
    ``bpt_gap=0.5`` since 3.0-2.5 and both finite, ``teacher_cache.hit
    is True``, ``top1_agreement is None`` since both argmax are None),
    and that ``stage6alt_eval_path`` is published on ctx as a Path
    ending in ``stage6alt_eval.json``.
    """
    from moe_compress.stage6alt.plugins import thermo_report as _rp_mod

    captured = {}

    def _capture(obj, path):  # noqa: ANN001
        captured["obj"] = obj
        captured["path"] = path
        return path

    monkeypatch.setattr(_rp_mod, "save_json_artifact", _capture)

    cfg = _make_thermo_config(tiny_config)
    ctx = _seed_ctx(tmp_path, cfg,
                    student_bpt=3.0, teacher_bpt=2.5,
                    student_argmax=None, teacher_argmax=None)
    plugin = _rp_mod.ThermoReportPlugin()
    plugin.assemble_thermo_report(ctx)

    results = captured["obj"]
    assert isinstance(results, dict)
    expected_keys = {
        "stage", "mode",
        "student_bpt", "teacher_bpt", "bpt_gap",
        "student_arc_easy_acc_norm", "student_hellaswag_acc_norm",
        "student_acc_norm_sum",
        "teacher_arc_easy_acc_norm", "teacher_hellaswag_acc_norm",
        "teacher_acc_norm_sum",
        "acc_norm_sum_gap", "top1_agreement",
        "corpus", "teacher_cache", "lm_eval",
    }
    assert set(results.keys()) == expected_keys, (
        f"key set mismatch — extra {set(results.keys()) - expected_keys}, "
        f"missing {expected_keys - set(results.keys())}"
    )
    assert len(expected_keys) == 16  # sanity: the golden pins 16 keys
    assert results["stage"] == "6alt"
    assert results["mode"] == "thermometer"
    assert results["bpt_gap"] == pytest.approx(0.5)
    assert results["teacher_cache"]["hit"] is True
    assert results["teacher_cache"]["path"] == str(
        tmp_path / "thermometer_teacher_cache.json"
    )
    assert results["top1_agreement"] is None
    assert results["lm_eval"] == {"arc_easy_limit": 5, "hellaswag_limit": 7}

    eval_path = ctx.get("stage6alt_eval_path")
    assert isinstance(eval_path, Path)
    assert eval_path.name == "stage6alt_eval.json"


def test_assemble_thermo_report_top1_agreement_computed(tmp_path, monkeypatch, tiny_config):
    """When student/teacher argmax are identical, top1_agreement == 1.0.

    Both argmaxes are ``torch.ones(2, 3, dtype=long)`` — same shape,
    same values. The hook should compute the mean equality as a
    float, yielding exactly 1.0.
    """
    from moe_compress.stage6alt.plugins import thermo_report as _rp_mod

    monkeypatch.setattr(_rp_mod, "save_json_artifact",
                        lambda obj, path: path)

    student_argmax = torch.ones(2, 3, dtype=torch.long)
    teacher_argmax = torch.ones(2, 3, dtype=torch.long).tolist()  # cache stores list

    cfg = _make_thermo_config(tiny_config)
    ctx = _seed_ctx(tmp_path, cfg,
                    student_argmax=student_argmax,
                    teacher_argmax=teacher_argmax)
    plugin = _rp_mod.ThermoReportPlugin()

    # Capture results via a second monkeypatch slot since we need them.
    captured = {}
    monkeypatch.setattr(_rp_mod, "save_json_artifact",
                        lambda obj, path: captured.setdefault("obj", obj))

    plugin.assemble_thermo_report(ctx)
    assert captured["obj"]["top1_agreement"] == pytest.approx(1.0)


def test_assemble_thermo_report_top1_agreement_shape_mismatch(
    tmp_path, monkeypatch, tiny_config,
):
    """Shape mismatch between student/teacher argmax → top1_agreement is None.

    Student is ``[2, 3]``; teacher is ``[2, 4]``. The hook's shape
    check rejects the comparison and leaves top1_agreement at None
    (and logs a warning).
    """
    from moe_compress.stage6alt.plugins import thermo_report as _rp_mod

    captured = {}
    monkeypatch.setattr(_rp_mod, "save_json_artifact",
                        lambda obj, path: captured.setdefault("obj", obj))

    student_argmax = torch.ones(2, 3, dtype=torch.long)
    teacher_argmax = torch.ones(2, 4, dtype=torch.long).tolist()

    cfg = _make_thermo_config(tiny_config)
    ctx = _seed_ctx(tmp_path, cfg,
                    student_argmax=student_argmax,
                    teacher_argmax=teacher_argmax)
    plugin = _rp_mod.ThermoReportPlugin()
    plugin.assemble_thermo_report(ctx)

    assert captured["obj"]["top1_agreement"] is None


def test_assemble_thermo_report_bpt_gap_none_when_inf(
    tmp_path, monkeypatch, tiny_config,
):
    """When student_bpt is +inf, bpt_gap is None (math.isfinite guard).

    The monolith's ``bpt_gap`` formula requires BOTH operands to be
    finite. A non-finite student BPT (e.g. partial-corpus skip) must
    leave bpt_gap as None rather than emit an inf scalar.
    """
    from moe_compress.stage6alt.plugins import thermo_report as _rp_mod

    captured = {}
    monkeypatch.setattr(_rp_mod, "save_json_artifact",
                        lambda obj, path: captured.setdefault("obj", obj))

    cfg = _make_thermo_config(tiny_config)
    ctx = _seed_ctx(tmp_path, cfg,
                    student_bpt=float("inf"), teacher_bpt=2.5)
    plugin = _rp_mod.ThermoReportPlugin()
    plugin.assemble_thermo_report(ctx)

    assert captured["obj"]["bpt_gap"] is None
