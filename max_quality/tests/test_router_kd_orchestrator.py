"""Tests for the Router-KD orchestrator's plugin pipeline.

Covers the Router-KD orchestrator's registry construction (the 7-plugin
roster + order), the canonical phase schedule, and — the part unique to
Router-KD — the DUAL-INVOCATION factory: Router-KD serves BOTH Stage 2.5 and
Stage 5, so it exposes ``make_router_kd_stage(stage_id)`` (a factory) rather
than a single ``STAGE`` singleton. Layer 1 is a set of fast registry/factory
tests that need no model run; Layer 2 instruments a real stage1->2->2.5/5 run
to verify the ``stage_id`` -> ``stage_key`` -> ``{stage_key}_final/`` dir-name
propagation for BOTH invocations.

This complements -- and deliberately does NOT duplicate -- the
``make_router_kd_stage`` conformance basics in ``test_router_kd_scaffold.py``
(package imports, the legacy-shim delegation, signature equality, the
unknown-id rejection, the stage-key threading). It also does not re-pin the
byte-level golden, which ``test_router_kd_golden_snapshot.py`` owns.

Helpers (``_TinyTokenizer``, ``_noop_save``, ``patched_router_kd``,
``_prepare_model_and_merge_map``) are redeclared locally on purpose -- tests in
this codebase do not import from each other (codebase discipline; mirrors
``test_router_kd_golden_snapshot.py`` / ``test_stage3_orchestrator.py``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    from moe_compress import stage1
    from moe_compress.stage2 import orchestrator as stage2_reap_ream
    from moe_compress.budget.solver import BudgetDecomposition
    from moe_compress.pipeline.context import PipelineContext
    from moe_compress.pipeline.registry import PluginRegistry
    from moe_compress.pipeline.stage import Stage
    from moe_compress.router_kd import make_router_kd_stage
    from moe_compress.router_kd.plugins.trainable_scope import TrainableScopePlugin
    from moe_compress.router_kd.plugins.kd_optimizer import KdOptimizerPlugin
    from moe_compress.router_kd.plugins.vocab_kd import VocabKdPlugin
    from moe_compress.router_kd.plugins.teacher import (
        TeacherCachePlugin,
        TeacherLivePlugin,
    )
    from moe_compress.router_kd.plugins.merge_repair import MergeRepairPlugin
    from moe_compress.router_kd.plugins.early_stop import EarlyStopPlugin
except Exception as e:  # pragma: no cover - import-time guard
    pytest.skip(f"Router-KD imports unavailable: {e}", allow_module_level=True)


# --------------------------------------------------------------------------
# Local helpers -- copied verbatim from test_router_kd_golden_snapshot.py
# (no cross-test imports; codebase discipline).
# --------------------------------------------------------------------------


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0

    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

    def save_pretrained(self, *_args, **_kwargs):
        return None


def _noop_save(model, tokenizer, path, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def _prepare_model_and_merge_map(model, config, tmp_path, monkeypatch):
    """Run stages 1+2 and write an identity merge_map.json at stage2_pruned/.

    Copied from ``test_router_kd_golden_snapshot.py``: Stage 2's real merge_map
    maps new_idx -> [original_expert_ids], but with teacher == student (same
    post-stage-2 model) ``_pool_teacher_logits`` would index original expert
    IDs into a tensor whose last dim is num_new_experts -- an out-of-bounds
    error. A trivial identity map (each expert maps to itself) avoids the
    pooling step entirely.
    """
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.utils.model_io import iter_moe_layers

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)

    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, _TinyTokenizer(), config, tmp_path, decomp)
    stage2_reap_ream.run(model, _TinyTokenizer(), config, tmp_path, device=None)

    moe_layer_refs = list(iter_moe_layers(model))
    trivial_map = {
        str(ref.layer_idx): {str(i): [i] for i in range(ref.num_routed_experts)}
        for ref in moe_layer_refs
    }
    (tmp_path / "stage2_pruned").mkdir(parents=True, exist_ok=True)
    (tmp_path / "stage2_pruned" / "merge_map.json").write_text(json.dumps(trivial_map))


@pytest.fixture
def patched_router_kd(monkeypatch, tiny_config):
    """Patch calibration loaders, the stage-2 saver, and Router-KD's teacher.

    Mirrors the ``patched_router_kd`` fixture in
    ``test_router_kd_golden_snapshot.py``: seeded fake calibration loaders on
    ``utils.calibration`` and on the modules that bind those names by direct
    import (``stage2.orchestrator`` and ``router_kd.orchestrator``), a no-op
    ``save_compressed_checkpoint`` on ``utils.model_io`` and
    ``stage2.orchestrator``, and a ``load_model`` patch on
    ``router_kd.plugins.teacher`` so the teacher == the student.

    Unlike the golden fixture, ``router_kd.orchestrator.save_compressed_checkpoint``
    is ALSO stubbed to the no-op saver -- this test only asserts the
    ``{stage_key}_final/`` directory NAME, not the bytes of the metadata file
    inside it, so the cheap no-op (which still creates the dir) is sufficient.

    ``router_kd.orchestrator._trackio_log`` is patched to a no-op so the run
    emits nothing -- this test does not inspect the loss trace.

    Returns the unchanged ``tiny_config``.
    """
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.router_kd import orchestrator as rk_orchestrator

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(rk_orchestrator, "build_calibration_tensor", _fake_build)
    if hasattr(rk_orchestrator, "build_super_expert_slice"):
        monkeypatch.setattr(rk_orchestrator, "build_super_expert_slice", _fake_slice)

    from moe_compress.utils import model_io as mio
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _noop_save)
    # This test asserts the {stage_key}_final/ dir NAME, not the bytes inside,
    # so the no-op saver (which still mkdir's the dir) is sufficient here.
    monkeypatch.setattr(rk_orchestrator, "save_compressed_checkpoint", _noop_save)

    monkeypatch.setattr(rk_orchestrator, "_trackio_log", lambda payload: None)

    return tiny_config


def _load_student_factory(student, tokenizer, monkeypatch):
    """Patch load_model so teacher == student.

    Mirrors the helper in ``test_router_kd_golden_snapshot.py``: the
    live-teacher plugin (``router_kd.plugins.teacher``) binds ``load_model`` by
    direct import -- patch it there; ``utils.model_io`` is patched too.
    """
    from moe_compress.utils import model_io as mio
    from moe_compress.router_kd.plugins import teacher as rk_teacher

    def _load_student(*_args, **_kwargs):
        return student, tokenizer

    monkeypatch.setattr(mio, "load_model", _load_student)
    monkeypatch.setattr(rk_teacher, "load_model", _load_student)


# --------------------------------------------------------------------------
# Canonical schedule -- the orchestrator has NO _ROUTER_KD_PHASES constant; the
# schedule is expressed by inline walk_phases / dispatch_first calls. This
# constant + mapping mirror that schedule for the conformance tests below.
# --------------------------------------------------------------------------

_CANONICAL_ROUTER_KD_SCHEDULE = (
    # SETUP phases (once, before the epoch loop).
    "load_teacher_cache",
    "setup_trainable_scope",
    "setup_merge_repair",
    "build_optimizer",
    "setup_early_stop",
    # PER-BATCH phases (once per microbatch).
    "provide_teacher_logits",
    "compute_merge_repair_mse",
    "compute_kd_loss",
    # PER-LOG-WINDOW phases.
    "update_best_tracker",
    "check_early_stop",
    # FINALIZE phases (once, after the epoch loop).
    "teardown_merge_repair",
    "reload_best_checkpoint",
)

# (schedule phase, plugin class that owns it). Several plugins own multiple
# hooks -- MergeRepairPlugin owns 3, EarlyStopPlugin owns 4 -- so this is a
# (phase -> class) list with repeated classes, not a 1:1 map. Note
# provide_teacher_logits is owned by TWO plugins (cache + live, the
# dispatch_first slot chain); it is verified separately in
# test_provide_teacher_logits_owned_by_both_teacher_plugins.
_PHASE_PLUGIN_MAP = (
    ("load_teacher_cache", TeacherCachePlugin),
    ("setup_trainable_scope", TrainableScopePlugin),
    ("setup_merge_repair", MergeRepairPlugin),
    ("build_optimizer", KdOptimizerPlugin),
    ("setup_early_stop", EarlyStopPlugin),
    ("compute_merge_repair_mse", MergeRepairPlugin),
    ("compute_kd_loss", VocabKdPlugin),
    ("update_best_tracker", EarlyStopPlugin),
    ("check_early_stop", EarlyStopPlugin),
    ("teardown_merge_repair", MergeRepairPlugin),
    ("reload_best_checkpoint", EarlyStopPlugin),
)

# The full 7-plugin roster in construction (= execution) order, exactly as
# router_kd/orchestrator.run builds it. TeacherCachePlugin is registered BEFORE
# TeacherLivePlugin so dispatch_first prefers the cache on a hit.
_EXPECTED_ROSTER = (
    "trainable_scope",
    "kd_optimizer",
    "vocab_kd",
    "teacher_cache",
    "teacher_live",
    "merge_repair",
    "early_stop",
)


# ==========================================================================
# Layer 1 -- registry / factory tests (no model run, fast).
# ==========================================================================


def test_orchestrator_builds_plugins_in_schedule_order():
    """The orchestrator builds a 7-plugin registry whose construction (=
    execution) order matches the Router-KD roster -- with TeacherCachePlugin
    registered BEFORE TeacherLivePlugin so ``dispatch_first`` prefers the
    cache on a hit."""
    registry = PluginRegistry([
        TrainableScopePlugin(),
        KdOptimizerPlugin(),
        VocabKdPlugin(),
        TeacherCachePlugin(),
        TeacherLivePlugin(),
        MergeRepairPlugin(stage_key="stage5"),
        EarlyStopPlugin(),
    ])
    assert len(registry) == 7
    names = registry.names()
    assert isinstance(names, tuple)
    assert names == _EXPECTED_ROSTER
    # TeacherCachePlugin must precede TeacherLivePlugin for dispatch_first.
    assert names.index("teacher_cache") < names.index("teacher_live")


def test_each_plugin_owns_its_schedule_phase_hooks():
    """Each plugin in the ownership mapping exposes a callable for the phase
    hook it owns. Several plugins own multiple hooks (MergeRepairPlugin: 3;
    EarlyStopPlugin: 4) -- the mapping enumerates every (phase, owner) pair."""
    for phase, plugin_class in _PHASE_PLUGIN_MAP:
        plugin = (
            plugin_class(stage_key="stage2p5")
            if plugin_class is MergeRepairPlugin
            else plugin_class()
        )
        assert callable(getattr(plugin, phase, None)), (
            f"{plugin_class.__name__} must expose a callable {phase!r} hook"
        )


def test_provide_teacher_logits_owned_by_both_teacher_plugins():
    """``provide_teacher_logits`` is the ONE slot-style hook owned by TWO
    plugins -- TeacherCachePlugin and TeacherLivePlugin -- forming the
    ``dispatch_first`` chain (cache wins on a hit, live answers on a miss)."""
    assert callable(getattr(TeacherCachePlugin(), "provide_teacher_logits", None))
    assert callable(getattr(TeacherLivePlugin(), "provide_teacher_logits", None))


def test_make_router_kd_stage_both_factory_outputs_conform():
    """Stage conformance for BOTH factory outputs (the master-plan-named
    check): for ``stage_id`` in ``("2.5", "5")`` the factory yields a
    ``Stage``-conforming object whose ``stage_id`` round-trips, that is
    enabled, and whose ``run`` is callable."""
    for stage_id in ("2.5", "5"):
        s = make_router_kd_stage(stage_id)
        assert isinstance(s, Stage)
        assert s.stage_id == stage_id
        assert s.is_enabled({}) is True
        assert callable(s.run)


def test_merge_repair_enabled_only_for_stage2p5():
    """The RK-6 stage-gate, verified at the orchestrator-registry level: the
    orchestrator constructs ``MergeRepairPlugin(stage_key=...)``, and
    ``registry.enabled(config)`` -- with ``merge_repair.enabled=True`` -- keeps
    ``merge_repair`` ONLY for the stage2p5 construction, dropping it for
    stage5."""
    cfg = {
        "stage5_router_kd": {"merge_repair": {"enabled": True}},
    }

    def _build_registry(stage_key: str) -> PluginRegistry:
        return PluginRegistry([
            TrainableScopePlugin(),
            KdOptimizerPlugin(),
            VocabKdPlugin(),
            TeacherCachePlugin(),
            TeacherLivePlugin(),
            MergeRepairPlugin(stage_key=stage_key),
            EarlyStopPlugin(),
        ])

    enabled_2p5 = [p.name for p in _build_registry("stage2p5").enabled(cfg)]
    enabled_s5 = [p.name for p in _build_registry("stage5").enabled(cfg)]

    # stage2p5 construction + merge_repair.enabled=True -> merge_repair is on.
    assert "merge_repair" in enabled_2p5
    # stage5 construction -> merge_repair is off regardless of the config flag.
    assert "merge_repair" not in enabled_s5

    # The other plugins are unaffected by the stage_key (teacher_cache is the
    # only one config-gated off here: no teacher_logits_cache path set).
    assert "teacher_cache" not in enabled_2p5
    assert enabled_s5 == [
        "trainable_scope", "kd_optimizer", "vocab_kd",
        "teacher_live", "early_stop",
    ]


# ==========================================================================
# Layer 2 -- functional dual-invocation test (one stage1->2->2.5/5 run each).
# ==========================================================================


@pytest.mark.parametrize(
    "stage_id, stage_key", [("2.5", "stage2p5"), ("5", "stage5")]
)
def test_dual_invocation_stage_id_propagates_to_dir_name(
    tiny_model, patched_router_kd, stage_id, stage_key, tmp_path, monkeypatch,
):
    """The master-plan-named ``stage_id`` -> dir-name propagation, end-to-end,
    for BOTH factory invocations.

    For each ``stage_id``: prep the stage1->2 prereq, build a
    ``PipelineContext`` with the required slots, call ``make_router_kd_stage``,
    and ``stage.run(ctx)``. Asserts the run returns ``None``, that the
    orchestrator wrote the per-stage output directory ``{stage_key}_final/``
    (proving ``stage_id`` -> ``stage_key`` -> ``{stage_key}_final/``), and that
    the namespaced ``router_kd_{stage_key}_path`` ctx slot is set to that
    directory.
    """
    cfg = copy.deepcopy(patched_router_kd)
    assert cfg["stage5_router_kd"]["epochs"] == 1, (
        "dual-invocation test assumes the 1-epoch tiny_config default"
    )

    _prepare_model_and_merge_map(tiny_model, cfg, tmp_path, monkeypatch)
    _load_student_factory(tiny_model, _TinyTokenizer(), monkeypatch)

    ctx = PipelineContext()
    ctx.set("student", tiny_model)
    ctx.set("tokenizer", _TinyTokenizer())
    ctx.set("config", cfg)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("device", None)
    ctx.set("no_resume", True)

    stage = make_router_kd_stage(stage_id)
    assert stage.stage_id == stage_id

    result = stage.run(ctx)

    # Stage.run is purely side-effecting -- it returns None.
    assert result is None

    # stage_id -> stage_key -> {stage_key}_final/ : the output directory name.
    expected_out = tmp_path / f"{stage_key}_final"
    assert expected_out.is_dir(), (
        f"Router-KD (stage_id={stage_id}) did not write {expected_out}"
    )

    # The namespaced per-stage output ctx slot points at that directory.
    slot = f"router_kd_{stage_key}_path"
    assert ctx.has(slot), f"_RouterKdStage.run must publish the {slot!r} slot"
    assert ctx.get(slot) == expected_out

    # Cross-check: this invocation produces ONLY its own stage's artifacts --
    # it must not also emit the OTHER stage's namespaced slot or {key}_final/
    # dir. This guards against a hard-coded / mis-keyed output name. (The two
    # stage_ids run as isolated pytest invocations, so this confirms per-stage
    # keying within a run, not live cross-invocation coexistence.)
    other_key = "stage5" if stage_key == "stage2p5" else "stage2p5"
    assert not ctx.has(f"router_kd_{other_key}_path")
    assert not (tmp_path / f"{other_key}_final").exists()


def test_orchestrator_run_visits_phases_in_canonical_order(
    tiny_model, patched_router_kd, tmp_path, monkeypatch,
):
    """Instrument every plugin phase hook on the CLASS and assert the
    orchestrator's ``run`` (stage 5) visits the SETUP phases once and in
    canonical order, the per-batch phases at least once after setup, and the
    FINALIZE phase once after the per-batch phases.

    Run at stage 5, so the merge-repair phases (``setup_merge_repair`` /
    ``compute_merge_repair_mse`` / ``teardown_merge_repair``) and
    ``load_teacher_cache`` are ABSENT -- ``walk_phases`` dispatches only against
    ``registry.enabled(config)``, and both ``MergeRepairPlugin(stage_key=
    "stage5")`` and the cache plugin (no cache path) are disabled. This is the
    flip side of the RK-6 stage gate verified statically in
    ``test_merge_repair_enabled_only_for_stage2p5``.

    The hooks have heterogeneous signatures (``provide_teacher_logits`` takes
    keyword-only ``input_ids`` / ``epoch`` / ... ; the others take just
    ``ctx``) -- a ``*args, **kwargs`` wrapper records the visit regardless.
    Patching is done on the CLASS because the orchestrator instantiates the
    plugins itself, so per-instance patching would not reach them.
    """
    cfg = copy.deepcopy(patched_router_kd)
    _prepare_model_and_merge_map(tiny_model, cfg, tmp_path, monkeypatch)
    _load_student_factory(tiny_model, _TinyTokenizer(), monkeypatch)

    visited: list[str] = []

    # Patch each (phase, owner) hook. provide_teacher_logits is patched on the
    # live-teacher plugin (the one that actually answers on a cache miss --
    # TeacherCachePlugin is disabled here with no cache configured).
    instrument = list(_PHASE_PLUGIN_MAP) + [
        ("provide_teacher_logits", TeacherLivePlugin),
    ]
    for phase, plugin_class in instrument:
        original = getattr(plugin_class, phase)

        def _make_wrapper(_phase, _original):
            def _wrapper(self, *args, **kwargs):
                visited.append(_phase)
                return _original(self, *args, **kwargs)
            return _wrapper

        monkeypatch.setattr(plugin_class, phase, _make_wrapper(phase, original))

    ctx = PipelineContext()
    ctx.set("student", tiny_model)
    ctx.set("tokenizer", _TinyTokenizer())
    ctx.set("config", cfg)
    ctx.set("artifacts_dir", tmp_path)
    ctx.set("device", None)
    ctx.set("no_resume", True)

    result = make_router_kd_stage("5").run(ctx)
    assert result is None

    # Phases ABSENT at stage 5 -- confirmed against the real orchestrator:
    #  * load_teacher_cache  -- TeacherCachePlugin is disabled (no cache path
    #    configured), so it is not in registry.enabled(config) and walk_phases
    #    never dispatches its hook.
    #  * setup_merge_repair / compute_merge_repair_mse / teardown_merge_repair
    #    -- MergeRepairPlugin(stage_key="stage5") is disabled (the RK-6 stage
    #    gate), so it is dropped from the enabled subset entirely. walk_phases
    #    dispatches only against enabled plugins, so its hooks NEVER run at
    #    stage 5 -- they are not no-op'd, they are absent. (At stage 2.5 with
    #    merge_repair.enabled=True they would run.)
    for absent in ("load_teacher_cache", "setup_merge_repair",
                   "compute_merge_repair_mse", "teardown_merge_repair"):
        assert visited.count(absent) == 0, (
            f"{absent!r} must NOT run at stage 5 (its plugin is disabled)"
        )

    # SETUP phases that DO run -- each exactly once, in canonical order,
    # before the first per-batch phase.
    for phase in ("setup_trainable_scope", "build_optimizer",
                  "setup_early_stop"):
        assert visited.count(phase) == 1, (
            f"setup phase {phase!r} should run exactly once"
        )
    idx_scope = visited.index("setup_trainable_scope")
    idx_optim = visited.index("build_optimizer")
    idx_early = visited.index("setup_early_stop")
    assert idx_scope < idx_optim < idx_early

    # PER-BATCH phases -- run at least once, after all setup phases.
    idx_first_teacher = visited.index("provide_teacher_logits")
    idx_first_kd = visited.index("compute_kd_loss")
    assert idx_early < idx_first_teacher
    assert idx_first_teacher < idx_first_kd
    # provide_teacher_logits and compute_kd_loss run once per microbatch.
    assert visited.count("provide_teacher_logits") >= 1
    assert visited.count("compute_kd_loss") == visited.count("provide_teacher_logits")

    # FINALIZE phase -- reload_best_checkpoint runs once, after every per-batch
    # phase. (teardown_merge_repair is absent at stage 5; see above.)
    assert visited.count("reload_best_checkpoint") == 1
    idx_reload = visited.index("reload_best_checkpoint")
    last_per_batch = max(
        i for i, p in enumerate(visited)
        if p in ("provide_teacher_logits", "compute_kd_loss")
    )
    assert last_per_batch < idx_reload

    # First-occurrence order of all visited phases is a subsequence of the
    # canonical schedule (the merge-repair + load_teacher_cache phases are
    # absent at stage 5 -> the remaining phases still appear in canonical
    # order).
    first_seen: list[str] = []
    for phase in visited:
        if phase not in first_seen:
            first_seen.append(phase)
    canonical = [p for p in _CANONICAL_ROUTER_KD_SCHEDULE if p in first_seen]
    assert first_seen == canonical
