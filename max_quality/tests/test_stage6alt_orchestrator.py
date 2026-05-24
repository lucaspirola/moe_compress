"""Tests for the Stage 6alt orchestrator's plugin pipeline (S6A-7).

The S6A-* refactor sub-tasks (S6A-1..S6A-6) progressively extracted the
Stage 6alt thermometer monolith into a 6-plugin orchestrator:
``stage6alt/orchestrator.py`` now builds a
``PluginRegistry`` of 6 plugins and dispatches a fixed phase schedule
against it. This file is the closing S6A-7 test: it pins the
orchestrator's contract -- the roster + order, each plugin's phase
ownership, the ``Stage`` protocol conformance of ``STAGE6ALT``, and --
under instrumentation -- the canonical phase-order traversal and the
teacher-cache-hit short-circuit that skips ``load_model``.

Unlike Stage 6 there is no conditional dispatch in the Stage 6alt
orchestrator (no cache-MISS-only phase, no imatrix gate); every one of
the six phases is walked unconditionally on every run. The teacher
cache-hit shortcut is internal to ``ThermoTeacherProviderPlugin``.

This complements -- and deliberately does NOT duplicate -- the
``test_stage6alt_scaffold.py`` package-surface checks and the
``test_stage6alt_golden_snapshot.py`` byte-identity pin (which captures
the ``stage6alt_eval.json`` artifact).

Helpers (``_TinyTokenizer``, the constants ``_EXPECTED_ROSTER`` /
``_PHASE_PLUGIN_MAP`` / ``_CANONICAL_S6ALT_PHASE_ORDER``, the
``patched_stage6alt`` fixture) are redeclared locally on purpose --
tests in this codebase do not import from each other (codebase
discipline; mirrors ``test_stage6_orchestrator.py`` /
``test_stage6alt_golden_snapshot.py``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    import torch
    from moe_compress.pipeline.registry import PluginRegistry
    from moe_compress.pipeline.stage import Stage
    from moe_compress.stage6alt import orchestrator as _s6alt_orch  # noqa: F401
    from moe_compress.stage6alt.orchestrator import run as _s6alt_orchestrator_run
    from moe_compress.stage6alt.plugins.bpt_metric import BptMetricPlugin
    from moe_compress.stage6alt.plugins.thermo_corpus import ThermoCorpusPlugin
    from moe_compress.stage6alt.plugins.thermo_environment import ThermoEnvironmentPlugin
    from moe_compress.stage6alt.plugins.thermo_report import ThermoReportPlugin
    from moe_compress.stage6alt.plugins.thermo_teacher_provider import (
        ThermoTeacherProviderPlugin,
    )
    from moe_compress.stage6alt.plugins.zero_shot_subset import ZeroShotSubsetPlugin
    from moe_compress.stage6alt.stage import STAGE6ALT
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(f"Stage 6alt imports unavailable: {e}", allow_module_level=True)


# --------------------------------------------------------------------------
# Local helpers -- redeclared verbatim; codebase discipline (no cross-test
# imports). Mirrors test_stage6alt_golden_snapshot.py /
# test_stage6_orchestrator.py.
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
# Stage 6alt orchestrator's ``walk_phases`` calls. Kept module-level so
# the layer-1 (registry) and layer-2 (instrumented run) tests share one
# source of truth.
# --------------------------------------------------------------------------

# The full 6-plugin roster in construction (= execution) order, exactly as
# stage6alt/orchestrator.run builds it.
_EXPECTED_ROSTER = (
    "thermo_environment",
    "thermo_corpus",
    "bpt_metric",
    "zero_shot_subset",
    "thermo_teacher_provider",
    "thermo_report",
)

# (phase, plugin class that owns it). Unlike Stage 6 there is a strict 1:1
# mapping between phases and plugins -- no shared slots, no plugin owning
# more than one phase.
_PHASE_PLUGIN_MAP = (
    ("setup_thermo_environment", ThermoEnvironmentPlugin),
    ("build_corpus", ThermoCorpusPlugin),
    ("compute_bpt", BptMetricPlugin),
    ("compute_zero_shot_subset", ZeroShotSubsetPlugin),
    ("provide_thermo_teacher_side", ThermoTeacherProviderPlugin),
    ("assemble_thermo_report", ThermoReportPlugin),
)

# The FULL canonical Stage 6alt phase order, in execution order. Every
# phase fires unconditionally on every run -- no conditional dispatch, no
# cache-MISS-only branch (the teacher cache-hit shortcut is internal to
# the teacher-provider plugin).
_CANONICAL_S6ALT_PHASE_ORDER = (
    "setup_thermo_environment",
    "build_corpus",
    "compute_bpt",
    "compute_zero_shot_subset",
    "provide_thermo_teacher_side",
    "assemble_thermo_report",
)


# ==========================================================================
# Layer 1 -- registry / Stage protocol tests (no model run, fast).
# ==========================================================================


def test_orchestrator_builds_plugins_in_schedule_order():
    """The orchestrator builds a 6-plugin registry whose construction (=
    execution) order matches the Stage 6alt roster: environment setup runs
    FIRST, then the calibration-corpus build, then the student-side BPT
    metric, then the student-side zero-shot subset, then the teacher
    provider, then the final report assembly."""
    registry = PluginRegistry([
        ThermoEnvironmentPlugin(),
        ThermoCorpusPlugin(),
        BptMetricPlugin(),
        ZeroShotSubsetPlugin(),
        ThermoTeacherProviderPlugin(),
        ThermoReportPlugin(),
    ])
    assert len(registry) == 6
    names = registry.names()
    assert isinstance(names, tuple)
    assert names == _EXPECTED_ROSTER


def test_each_plugin_owns_its_schedule_phase_hooks():
    """Each plugin in the ownership mapping exposes a callable for the
    phase hook it owns. Stage 6alt's mapping is strict 1:1 -- every plugin
    owns exactly one phase and no phase is shared between plugins."""
    for phase, plugin_class in _PHASE_PLUGIN_MAP:
        plugin = plugin_class()
        assert callable(getattr(plugin, phase, None)), (
            f"{plugin_class.__name__} must expose a callable {phase!r} hook"
        )


def test_stage6alt_conforms_to_stage_protocol():
    """``STAGE6ALT`` is a ``Stage``-conforming object -- it satisfies the
    structural :class:`Stage` Protocol, exposes ``stage_id == "6alt"``, is
    enabled unconditionally (stage selection / mode dispatch belongs to
    the universal orchestrator, not to the stage), and has a callable
    ``run``."""
    assert isinstance(STAGE6ALT, Stage)
    assert STAGE6ALT.stage_id == "6alt"
    assert STAGE6ALT.is_enabled({}) is True
    assert callable(STAGE6ALT.run)


# ==========================================================================
# Layer 2 -- instrumented functional run (uses ``tiny_model``, no real
# evals; teacher cache-HIT path).
# ==========================================================================


@pytest.fixture
def patched_stage6alt(monkeypatch, tiny_config):
    """Patch Stage 6alt so the orchestrator ``run()`` completes CPU-only
    with no real evals -- the teacher cache-HIT path.

    Mirrors the ``patched_stage6alt`` fixture in
    ``test_stage6alt_golden_snapshot.py`` (same H3 patch surface) but
    returns ONLY ``cfg`` (these tests do not byte-pin a snapshot):

    * ``_set_experts_implementation_s6`` and ``_apply_stage6_kernel_patches``
      on ``stage6alt.plugins.thermo_environment`` → no-op (tiny_model has
      no fused-experts switch / no fla / no GatedDeltaNet).
    * ``_build_thermo_corpus`` on ``stage6alt.plugins.thermo_corpus``
      → returns a constant ``(calib_ids, corpus_meta, corpus_id)`` so no
      dataset is loaded.
    * ``_bpt_from_nll`` on ``stage6alt.plugins.bpt_metric`` → returns
      ``(3.0, None)``.
    * ``_lm_eval_subset`` on ``stage6alt.plugins.zero_shot_subset`` →
      returns all-None metrics.
    * ``_load_thermo_teacher_cache`` on
      ``stage6alt.plugins.thermo_teacher_provider`` → returns a pre-baked
      teacher dict (cache HIT, ``teacher_bpt=2.5``, no argmax) so the
      teacher-load branch is bypassed entirely.

    Additionally the config's ``thermometer`` sub-dict is overlaid with a
    fixed ``teacher_cache_path`` (``/dev/null/stub_teacher_cache.json``)
    so the JSON's ``teacher_cache.path`` field does not embed pytest's
    volatile ``tmp_path``.

    HAZARD H3: the patch targets must repoint to the plugin modules that
    own the call sites (post-S6A-6 flip) -- patching
    ``stage6alt_thermometer`` directly would NOT reach the orchestrator
    code.
    """
    # Function-local imports of the plugin modules -- the H3 patches now
    # target the plugin modules that own the call sites (post-S6A-6
    # flip), not the legacy ``stage6alt_thermometer`` monolith. Importing
    # at function scope mirrors the test-isolation discipline used by the
    # sibling stage6alt golden.
    from moe_compress.stage6alt.plugins import (
        bpt_metric,
        thermo_corpus,
        thermo_environment,
        thermo_teacher_provider,
        zero_shot_subset,
    )

    cfg = copy.deepcopy(tiny_config)
    s6 = cfg["stage6_validate"]
    s6["mode"] = "thermometer"
    s6["thermometer"] = {
        "corpus": "nemotron",
        "num_sequences": 4,
        "sequence_length": 16,
        "bpt_batch_size": 4,
        "lm_eval_batch_size": "auto:4",
        "arc_easy_limit": 2,
        "hellaswag_limit": 2,
        "teacher_cache_path": "/dev/null/stub_teacher_cache.json",
    }

    # 1. Kernel/impl switches → no-op.
    monkeypatch.setattr(
        thermo_environment, "_set_experts_implementation_s6",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        thermo_environment, "_apply_stage6_kernel_patches",
        lambda *a, **k: None,
    )

    # 2. Corpus build → constant (calib_ids, corpus_meta, corpus_id).
    _calib = torch.zeros(4, 16, dtype=torch.long)
    _corpus_meta = {
        "name": "nemotron",
        "num_sequences": 4,
        "sequence_length": 16,
        "effective_seed": 0,
        "seed_offset": 715,
        "subset_weights": {
            "math": 0.35, "swe": 0.25, "chat": 0.25, "science": 0.15,
        },
    }
    monkeypatch.setattr(
        thermo_corpus, "_build_thermo_corpus",
        lambda *a, **k: (_calib, _corpus_meta, "nemotron:stub"),
    )

    # 3. BPT → (3.0, None). Finite BPT + no argmax → top1_agreement None.
    monkeypatch.setattr(
        bpt_metric, "_bpt_from_nll",
        lambda *a, **k: (3.0, None),
    )

    # 4. lm-eval subset → all-None.
    monkeypatch.setattr(
        zero_shot_subset, "_lm_eval_subset",
        lambda *a, **k: {
            "arc_easy_acc_norm": None,
            "hellaswag_acc_norm": None,
            "acc_norm_sum": None,
        },
    )

    # 5. Teacher cache → HIT with teacher_bpt=2.5, no argmax.
    monkeypatch.setattr(
        thermo_teacher_provider, "_load_thermo_teacher_cache",
        lambda *a, **k: {
            "teacher_bpt": 2.5,
            "teacher_arc_easy_acc_norm": None,
            "teacher_hellaswag_acc_norm": None,
            "teacher_acc_norm_sum": None,
            "teacher_argmax": None,
        },
    )

    return cfg


def test_orchestrator_run_visits_phases_in_canonical_order(
    tiny_model, patched_stage6alt, tmp_path, monkeypatch,
):
    """Instrument the CLASS-level phase hooks on every plugin and assert
    the orchestrator's ``run()`` visits the six phases in canonical
    first-occurrence order.

    Unlike Stage 6 there is no conditional dispatch in the Stage 6alt
    orchestrator -- ``walk_phases`` is invoked once per phase, every
    phase, on every run. All six hooks must therefore fire exactly once.

    Patching is done on the CLASS because the orchestrator instantiates
    the plugins itself, so per-instance patching would not reach them.
    """
    # patched_stage6alt already returns a deep-copied cfg; use it directly.
    cfg = patched_stage6alt

    # Stage 6alt's _bpt_from_nll guards on
    # model.config._attn_implementation == "eager"; the helper itself is
    # patched away, but pin the attribute defensively so any code path
    # reading it sees a sane value.
    monkeypatch.setattr(
        tiny_model.config, "_attn_implementation", "eager", raising=False,
    )

    visited: list[str] = []

    # Wrap-and-record on every (phase, plugin) pair. The ``*args, **kwargs``
    # wrapper records the visit regardless of hook signature.
    for phase, plugin_class in _PHASE_PLUGIN_MAP:
        original = getattr(plugin_class, phase)

        def _make_wrapper(_phase, _original):
            def _wrapper(self, *args, **kwargs):
                visited.append(_phase)
                return _original(self, *args, **kwargs)
            return _wrapper

        monkeypatch.setattr(plugin_class, phase, _make_wrapper(phase, original))

    result = _s6alt_orchestrator_run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None,
    )

    # The orchestrator returns the stage6alt_eval.json path on success.
    assert result == tmp_path / "stage6alt_eval.json"

    # Every phase fires exactly once -- no conditional dispatch.
    for phase, _plugin_class in _PHASE_PLUGIN_MAP:
        assert visited.count(phase) == 1, (
            f"phase {phase!r} fired {visited.count(phase)} times; expected 1"
        )

    # First-occurrence order matches the canonical Stage 6alt phase order.
    first_seen: list[str] = []
    for phase in visited:
        if phase not in first_seen:
            first_seen.append(phase)
    assert first_seen == list(_CANONICAL_S6ALT_PHASE_ORDER)

    # The first-occurrence sequence is a subsequence of the canonical
    # Stage 6alt phase order, filtered to phases that actually fired
    # (here: all of them).
    expected_subseq = [p for p in _CANONICAL_S6ALT_PHASE_ORDER if p in first_seen]
    assert first_seen == expected_subseq


def test_stage6alt_run_writes_eval_json(
    tiny_model, patched_stage6alt, tmp_path, monkeypatch,
):
    """The orchestrator runs end-to-end and the report plugin lands
    ``stage6alt_eval.json`` at ``tmp_path``; the orchestrator's return
    value equals that path."""
    cfg = patched_stage6alt
    monkeypatch.setattr(
        tiny_model.config, "_attn_implementation", "eager", raising=False,
    )

    result = _s6alt_orchestrator_run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None,
    )

    assert result == tmp_path / "stage6alt_eval.json"
    assert (tmp_path / "stage6alt_eval.json").is_file(), (
        "Stage 6alt orchestrator must produce stage6alt_eval.json end-to-end"
    )


def test_cache_hit_skips_teacher_load(
    tiny_model, patched_stage6alt, tmp_path, monkeypatch,
):
    """The teacher-cache-HIT branch in
    ``ThermoTeacherProviderPlugin.provide_thermo_teacher_side`` has two
    observable consequences this test pins:

    * ``load_model`` (imported on ``stage6alt.plugins.thermo_teacher_provider``
      and called only by the cache-MISS path) is NEVER invoked. We patch
      ``load_model`` on the plugin module to raise -- if the cache-hit
      branch correctly short-circuits, the raise is never tripped and
      the orchestrator returns cleanly. (The ``patched_stage6alt``
      fixture pre-installs a cache-HIT stub for ``_load_thermo_teacher_cache``.)
    * The resulting ``stage6alt_eval.json`` records
      ``teacher_cache.hit == True``.
    """
    from moe_compress.stage6alt.plugins import thermo_teacher_provider

    cfg = patched_stage6alt
    monkeypatch.setattr(
        tiny_model.config, "_attn_implementation", "eager", raising=False,
    )

    def _load_model_must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "load_model must not be called on the teacher-cache-HIT path"
        )

    monkeypatch.setattr(
        thermo_teacher_provider, "load_model",
        _load_model_must_not_be_called,
    )

    result = _s6alt_orchestrator_run(
        tiny_model, _TinyTokenizer(), cfg, tmp_path, device=None,
    )

    # The run reaches the report plugin and lands the artifact.
    assert result == tmp_path / "stage6alt_eval.json"
    assert (tmp_path / "stage6alt_eval.json").is_file()

    # Verify the resulting JSON records the cache-hit branch (the
    # orchestrator does not expose ctx to the test; the JSON is the
    # observable side-effect of ``ctx.set("teacher_cache_hit", True)``
    # in the cache-hit branch).
    payload = json.loads((tmp_path / "stage6alt_eval.json").read_bytes())
    assert payload["teacher_cache"]["hit"] is True
