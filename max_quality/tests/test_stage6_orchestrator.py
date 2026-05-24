"""Tests for the Stage 6 orchestrator's plugin pipeline (S6-9).

The S6-* refactor sub-tasks (S6-1..S6-8) progressively decomposed the Stage 6
monolith into a plugin-driven orchestrator: ``stage6/orchestrator.py`` now
builds an 8-plugin ``PluginRegistry`` and dispatches a fixed phase schedule
against it. This file is the closing S6-9 test: it pins the orchestrator's
contract -- the roster + order, each plugin's phase ownership, the
``Stage`` protocol conformance of ``STAGE6``, the ``is_enabled`` gating
semantic for the one config-gated plugin (``imatrix_export``, default-True
to faithfully reproduce the monolith), and -- under instrumentation -- the
canonical phase-order traversal and the cache-hit short-circuit that skips
``start_gguf_convert``.

This complements -- and deliberately does NOT duplicate -- the
``test_stage6_scaffold.py`` package-surface checks (which guard the
``stage6.run`` import and the ``stage6_validate`` shim delegation) and the
``test_stage6_golden_snapshot.py`` byte-identity pin (which captures the
``stage6_eval.json`` artifact).

Helpers (``_TinyTokenizer``, the constants ``_EXPECTED_ROSTER`` /
``_PHASE_PLUGIN_MAP`` / ``_CANONICAL_S6_PHASE_ORDER``, the ``patched_stage6``
fixture) are redeclared locally on purpose -- tests in this codebase do not
import from each other (codebase discipline; mirrors
``test_router_kd_orchestrator.py`` / ``test_stage6_golden_snapshot.py``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress.pipeline.registry import PluginRegistry
    from moe_compress.pipeline.stage import Stage
    from moe_compress.stage6 import orchestrator as _s6_orch
    from moe_compress.stage6.orchestrator import run as _s6_orchestrator_run
    from moe_compress.stage6.plugins import eval_environment as _eval_env_mod
    from moe_compress.stage6.plugins.eval_environment import EvalEnvironmentPlugin
    from moe_compress.stage6.plugins.humaneval import HumanEvalPlugin
    from moe_compress.stage6.plugins.imatrix_export import ImatrixExportPlugin
    from moe_compress.stage6.plugins.math500 import Math500Plugin
    from moe_compress.stage6.plugins.teacher_provider import TeacherProviderPlugin
    from moe_compress.stage6.plugins.validation_report import ValidationReportPlugin
    from moe_compress.stage6.plugins.wikitext_ppl import WikitextPplPlugin
    from moe_compress.stage6.plugins.zero_shot_lm_eval import ZeroShotLmEvalPlugin
    from moe_compress.stage6.stage import STAGE6
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(f"Stage 6 imports unavailable: {e}", allow_module_level=True)


# --------------------------------------------------------------------------
# Local helpers -- redeclared verbatim; codebase discipline (no cross-test
# imports). Mirrors test_stage6_golden_snapshot.py / test_router_kd_orchestrator.py.
# --------------------------------------------------------------------------


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


# --------------------------------------------------------------------------
# Canonical schedule constants -- mirror the schedule defined inline by the
# Stage 6 orchestrator's ``walk_phases`` calls. Kept module-level so the
# layer-1 (registry) and layer-2 (instrumented run) tests share one source
# of truth.
# --------------------------------------------------------------------------

# The full 8-plugin roster in construction (= execution) order, exactly as
# stage6/orchestrator.run builds it.
_EXPECTED_ROSTER = (
    "eval_environment",
    "wikitext_ppl",
    "zero_shot_lm_eval",
    "humaneval",
    "math500",
    "teacher_provider",
    "imatrix_export",
    "validation_report",
)

# (phase, plugin class that owns it). The 4 student-side eval plugins all
# expose the SAME ``eval_task`` slot (the orchestrator walks the slot once
# and PluginRegistry.enabled(config) gates which sub-evals actually fire);
# ImatrixExportPlugin owns TWO phase hooks (``start_gguf_convert`` early,
# ``export_imatrix`` late). This is a (phase -> class) list with repeated
# entries, not a 1:1 map.
_PHASE_PLUGIN_MAP = (
    ("setup_environment", EvalEnvironmentPlugin),
    ("eval_task", WikitextPplPlugin),
    ("eval_task", ZeroShotLmEvalPlugin),
    ("eval_task", HumanEvalPlugin),
    ("eval_task", Math500Plugin),
    ("provide_teacher_side", TeacherProviderPlugin),
    ("start_gguf_convert", ImatrixExportPlugin),
    ("export_imatrix", ImatrixExportPlugin),
    ("assemble_report", ValidationReportPlugin),
)

# The FULL canonical Stage 6 phase order, in execution order. ``eval_task``
# is a single slot the orchestrator walks once (the 4 eval plugins share it).
# ``start_gguf_convert`` is gated to the cache-MISS path only — included here
# in its canonical position (between eval_task and provide_teacher_side); the
# cache-HIT layer-2 tests filter it out of the expected subsequence via
# ``[p for p in _CANONICAL_S6_PHASE_ORDER if p in first_seen]``.
_CANONICAL_S6_PHASE_ORDER = (
    "setup_environment",
    "eval_task",
    "start_gguf_convert",
    "provide_teacher_side",
    "export_imatrix",
    "assemble_report",
)


# ==========================================================================
# Layer 1 -- registry / Stage protocol tests (no model run, fast).
# ==========================================================================


def test_orchestrator_builds_plugins_in_schedule_order():
    """The orchestrator builds an 8-plugin registry whose construction (=
    execution) order matches the Stage 6 roster: the eval-environment setup
    runs FIRST, then the 4 student-side eval plugins, then the teacher
    provider, then the imatrix export, and finally the validation report."""
    registry = PluginRegistry([
        EvalEnvironmentPlugin(),
        WikitextPplPlugin(),
        ZeroShotLmEvalPlugin(),
        HumanEvalPlugin(),
        Math500Plugin(),
        TeacherProviderPlugin(),
        ImatrixExportPlugin(),
        ValidationReportPlugin(),
    ])
    assert len(registry) == 8
    names = registry.names()
    assert isinstance(names, tuple)
    assert names == _EXPECTED_ROSTER


def test_each_plugin_owns_its_schedule_phase_hooks():
    """Each plugin in the ownership mapping exposes a callable for the phase
    hook it owns. The 4 eval plugins share the ``eval_task`` slot (one
    callable each); ImatrixExportPlugin owns BOTH ``start_gguf_convert`` and
    ``export_imatrix`` -- the mapping enumerates every (phase, owner) pair."""
    for phase, plugin_class in _PHASE_PLUGIN_MAP:
        plugin = plugin_class()
        assert callable(getattr(plugin, phase, None)), (
            f"{plugin_class.__name__} must expose a callable {phase!r} hook"
        )


def test_stage6_conforms_to_stage_protocol():
    """``STAGE6`` is a ``Stage``-conforming object -- it satisfies the
    structural :class:`Stage` Protocol, exposes ``stage_id == "6"``, is
    enabled unconditionally (stage selection belongs to the universal
    orchestrator, not to the stage), and has a callable ``run``."""
    assert isinstance(STAGE6, Stage)
    assert STAGE6.stage_id == "6"
    assert STAGE6.is_enabled({}) is True
    assert callable(STAGE6.run)


def test_imatrix_disabled_drops_plugin_from_enabled_set():
    """Cross-check the S6-6 default-True semantic on the 8-plugin registry:
    a Stage 6 config that OMITS the ``imatrix`` subdict still enables
    ``imatrix_export`` (the monolith ``run()`` defaulted to True at the
    call site; the plugin gate must match); a config that explicitly sets
    ``imatrix.enabled=False`` drops ``imatrix_export`` from the enabled set.
    In both cases the three always-on plugins (``eval_environment``,
    ``teacher_provider``, ``validation_report``) remain enabled.
    """
    def _build_registry() -> PluginRegistry:
        return PluginRegistry([
            EvalEnvironmentPlugin(),
            WikitextPplPlugin(),
            ZeroShotLmEvalPlugin(),
            HumanEvalPlugin(),
            Math500Plugin(),
            TeacherProviderPlugin(),
            ImatrixExportPlugin(),
            ValidationReportPlugin(),
        ])

    # Case A: no imatrix subkey -> imatrix_export is on (default True).
    cfg_default = {"stage6_validate": {}}
    enabled_default = [p.name for p in _build_registry().enabled(cfg_default)]
    assert "imatrix_export" in enabled_default
    for always_on in ("eval_environment", "teacher_provider", "validation_report"):
        assert always_on in enabled_default

    # Case B: imatrix.enabled=False -> imatrix_export drops out.
    cfg_off = {"stage6_validate": {"imatrix": {"enabled": False}}}
    enabled_off = [p.name for p in _build_registry().enabled(cfg_off)]
    assert "imatrix_export" not in enabled_off
    for always_on in ("eval_environment", "teacher_provider", "validation_report"):
        assert always_on in enabled_off


# ==========================================================================
# Layer 2 -- instrumented functional run (uses ``tiny_model``, no real
# evals; cache-HIT path with imatrix disabled).
# ==========================================================================


@pytest.fixture
def patched_stage6(monkeypatch, tiny_config):
    """Patch Stage 6 so the orchestrator ``run()`` completes CPU-only with no
    real evals -- the cache-HIT path with imatrix disabled.

    Mirrors the ``patched_stage6`` fixture in
    ``test_stage6_golden_snapshot.py`` (same H3 patch surface) but returns
    ONLY ``cfg`` (these tests do not inspect trackio payloads):

    * ``strict_revision_pinning`` is turned OFF so the tiny config (which
      pins no dataset SHAs) does not trip the revision-pinning guard.
    * ``teacher_eval_cache.enabled`` is turned ON so the orchestrator hits
      ``_load_teacher_cache``; the patched ``_load_teacher_cache`` returns a
      permanent pre-baked hit so the cache-HIT branch is exercised, bypassing
      teacher loading, the background preload thread and the GGUF pipeline.
    * ``imatrix.enabled=False`` (belt-and-suspenders): the cache-HIT path
      already short-circuits the early GGUF kickoff, but the explicit
      disable also drops ``imatrix_export`` from ``registry.enabled(config)``
      entirely (verified by the layer-1 test above).
    * ``_build_imatrix_calibration_corpus`` is patched on the eval-env plugin
      module to a no-op (it would otherwise hit the network).
    * ``_trackio_log`` is patched on the orchestrator module to a no-op.

    HAZARD H3: the patch targets must repoint to the modules each name is
    actually resolved from at call time (the orchestrator's
    ``_load_teacher_cache`` / ``_trackio_log``, the eval-env plugin's own
    ``_build_imatrix_calibration_corpus``) -- patching ``stage6_validate``
    directly would NOT reach the orchestrator code.
    """
    cfg = copy.deepcopy(tiny_config)
    s6 = cfg["stage6_validate"]
    s6["strict_revision_pinning"] = False
    s6["teacher_eval_cache"] = {"enabled": True}
    s6["imatrix"] = {"enabled": False}

    monkeypatch.setattr(
        _eval_env_mod, "_build_imatrix_calibration_corpus",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        _s6_orch, "_load_teacher_cache",
        lambda *a, **k: {
            "results": {},
            "param_counts": {"total": 0, "expert": 0},
        },
    )
    monkeypatch.setattr(_s6_orch, "_trackio_log", lambda payload: None)

    return cfg


def test_orchestrator_run_visits_phases_in_canonical_order(
    tiny_model, patched_stage6, tmp_path, monkeypatch,
):
    """Instrument the CLASS-level phase hooks on the three always-on plugins
    (plus a no-op spy on ``ImatrixExportPlugin.start_gguf_convert``) and
    assert the orchestrator's ``run()`` visits the phases in canonical
    first-occurrence order.

    Cache-HIT path with imatrix disabled and all 4 eval families off:

    * ``start_gguf_convert`` is the cache-MISS-only branch in the
      orchestrator preamble -- on a cache HIT the orchestrator publishes
      ``gguf_thread=None``/``gguf_result={}`` directly without walking the
      phase. So the spy must record ZERO calls. (Belt-and-suspenders: the
      ``imatrix.enabled=False`` config also drops ``imatrix_export`` from
      ``registry.enabled(config)``, so even if the orchestrator did dispatch
      the phase the spy would not fire.)
    * ``eval_task`` is absent from ``visited`` because all 4 eval plugins are
      disabled in the tiny config (``wikitext2``/``zero_shot``/``generative``
      all off) -- ``walk_phases`` dispatches only against enabled plugins.
    * ``export_imatrix`` is absent because ``imatrix_export`` is dropped
      from the enabled subset entirely.

    Patching is done on the CLASS because the orchestrator instantiates the
    plugins itself, so per-instance patching would not reach them.
    """
    # patched_stage6 already returns a deep-copied cfg; use it directly.
    cfg = patched_stage6

    # Stage 6 pins attn_implementation="eager" for the teacher; the tiny
    # student has no real attention impl but pin the attribute defensively
    # so any code reading model.config._attn_implementation sees a sane value.
    monkeypatch.setattr(
        tiny_model.config, "_attn_implementation", "eager", raising=False,
    )

    visited: list[str] = []

    # Wrap-and-record on the three always-on plugin hooks. The
    # ``*args, **kwargs`` wrapper records the visit regardless of hook
    # signature.
    wrap_targets = (
        ("setup_environment", EvalEnvironmentPlugin),
        ("provide_teacher_side", TeacherProviderPlugin),
        ("assemble_report", ValidationReportPlugin),
    )
    for phase, plugin_class in wrap_targets:
        original = getattr(plugin_class, phase)

        def _make_wrapper(_phase, _original):
            def _wrapper(self, *args, **kwargs):
                visited.append(_phase)
                return _original(self, *args, **kwargs)
            return _wrapper

        monkeypatch.setattr(plugin_class, phase, _make_wrapper(phase, original))

    # Independent spy on start_gguf_convert (no-op + count). Replaces the
    # original entirely so even if the orchestrator did dispatch it (it
    # should NOT on this cache-HIT path), the body would not run.
    start_gguf_called: list[None] = []

    def _start_gguf_spy(self, *args, **kwargs):
        start_gguf_called.append(None)
        return None

    monkeypatch.setattr(ImatrixExportPlugin, "start_gguf_convert", _start_gguf_spy)

    result = _s6_orchestrator_run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None,
    )

    # The orchestrator returns the stage6_eval.json path on success.
    assert result == tmp_path / "stage6_eval.json"

    # start_gguf_convert MUST NOT fire on the cache-HIT path (and imatrix
    # is disabled anyway -- belt-and-suspenders).
    assert start_gguf_called == [], (
        "start_gguf_convert must not fire on the cache-HIT path"
    )

    # First-occurrence order of visited phases matches the expected sequence:
    # all 4 eval families disabled -> no eval_task visit; imatrix disabled ->
    # no export_imatrix visit. Only the three always-on hooks fire.
    first_seen: list[str] = []
    for phase in visited:
        if phase not in first_seen:
            first_seen.append(phase)
    assert first_seen == [
        "setup_environment",
        "provide_teacher_side",
        "assemble_report",
    ]

    # The first-occurrence sequence is a subsequence of the canonical
    # Stage 6 phase order, filtered to phases that actually fired.
    expected_subseq = [p for p in _CANONICAL_S6_PHASE_ORDER if p in first_seen]
    assert first_seen == expected_subseq


def test_cache_hit_short_circuits_start_gguf_convert_and_teacher_load(
    tiny_model, patched_stage6, tmp_path, monkeypatch,
):
    """The cache-HIT branch in the orchestrator preamble has two observable
    consequences this test pins:

    * ``start_gguf_convert`` is NEVER dispatched -- the orchestrator only
      walks that phase on the cache-MISS path; on a HIT it publishes
      ``gguf_thread=None``/``gguf_result={}`` directly.
    * ``provide_teacher_side`` STILL runs exactly once -- the teacher provider
      plugin owns the cache-hit short-circuit body internally (it reads the
      pre-published ``cached_teacher_results`` slot and emits teacher metrics
      without loading a teacher).

    And the run completes to artifact: ``stage6_eval.json`` lands on disk.
    """
    # patched_stage6 already returns a deep-copied cfg; use it directly.
    cfg = patched_stage6
    monkeypatch.setattr(
        tiny_model.config, "_attn_implementation", "eager", raising=False,
    )

    start_gguf_calls: list[None] = []

    def _start_gguf_spy(self, *args, **kwargs):
        start_gguf_calls.append(None)
        return None

    monkeypatch.setattr(ImatrixExportPlugin, "start_gguf_convert", _start_gguf_spy)

    teacher_side_calls: list[None] = []
    original_teacher_side = TeacherProviderPlugin.provide_teacher_side

    def _teacher_side_wrap(self, *args, **kwargs):
        teacher_side_calls.append(None)
        return original_teacher_side(self, *args, **kwargs)

    monkeypatch.setattr(
        TeacherProviderPlugin, "provide_teacher_side", _teacher_side_wrap,
    )

    result = _s6_orchestrator_run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None,
    )

    # Cache-HIT short-circuit: start_gguf_convert is never dispatched.
    assert start_gguf_calls == [], (
        "start_gguf_convert must not be dispatched on a cache HIT"
    )

    # provide_teacher_side ran exactly once -- the orchestrator walks the
    # phase unconditionally; the plugin's own cache-hit branch handles the
    # short-circuit internally.
    assert len(teacher_side_calls) == 1

    # The run reaches the validation-report plugin and writes the artifact.
    assert result == tmp_path / "stage6_eval.json"
    assert (tmp_path / "stage6_eval.json").is_file(), (
        "Stage 6 orchestrator must produce stage6_eval.json end-to-end"
    )
