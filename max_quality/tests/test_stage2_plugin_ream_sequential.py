"""Plugin #10 / row S2_SEQ — REAM sequential merging cache invalidator.

Pins the ``Stage2ReamSequentialPlugin`` contract:

* ``PipelinePlugin`` Protocol conformance.
* ``is_enabled`` gates on ``stage2_reap_ream.sequential_reprofile`` truthiness;
  every flavour of OFF (absent block, absent key, ``False``, ``0``, ``""``,
  non-dict configs) leaves the plugin out, preserving the default
  byte-identical non-sequential path.
* ``on_post_merge`` invalidates the three per-layer cost-accumulator ctx
  slots (``cov_acc`` / ``ream_acc`` / ``layer_input_acc``) by upserting them
  to ``None`` — works whether the slots were ever written on this scope
  or not.
* ``PluginRegistry`` wiring: with the knob ON the plugin is enabled, with
  the knob OFF (default) it is dropped.
* Source-string contract: ``orchestrator.py`` registers this plugin AFTER
  ``MergeHealPlugin`` in the registry list, so any future plugin's
  ``on_post_merge`` hook runs **before** this one's invalidation.

Spec: ``tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md`` §523-532 (row S2_SEQ),
§5.2 (REAM paper anchor, arXiv:2604.04356), §582 (Risk R2). Foundational
dep: Plugin #6 (commit ``ce49a1b``) added the ``on_post_merge`` phase.
"""
from __future__ import annotations

import pathlib

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.stage2.plugins.ream_sequential import Stage2ReamSequentialPlugin


# --- plugin contract --------------------------------------------------------

def test_plugin_conforms_to_pipeline_plugin():
    assert isinstance(Stage2ReamSequentialPlugin(), PipelinePlugin)


def test_plugin_name():
    assert Stage2ReamSequentialPlugin.name == "ream_sequential"


def test_plugin_config_key():
    assert Stage2ReamSequentialPlugin.config_key == (
        "stage2_reap_ream.sequential_reprofile"
    )


def test_plugin_paper_cites_arxiv_2604_04356():
    """Paper attribution must cite the REAM arXiv id and the SAILMontreal repo."""
    paper = Stage2ReamSequentialPlugin.paper
    assert "2604.04356" in paper
    assert "SamsungSAILMontreal/ream" in paper
    # The three documented deviations must be enumerated in the citation
    # so callers see them without opening the module docstring.
    assert "D-seq-1" in paper
    assert "D-seq-2" in paper
    assert "D-seq-3" in paper


def test_plugin_writes_three_cache_slots():
    """``writes`` declares the three invalidated ctx slots."""
    assert set(Stage2ReamSequentialPlugin.writes) == {
        "cov_acc", "ream_acc", "layer_input_acc",
    }


def test_plugin_reads_and_provides_empty():
    """Cache invalidation has no read deps and provides no accumulators."""
    assert Stage2ReamSequentialPlugin.reads == ()
    assert Stage2ReamSequentialPlugin.provides == ()


def test_plugin_contribute_artifact_is_empty_dict():
    """Empty dict per Plugin Protocol — fresh literal every call."""
    plugin = Stage2ReamSequentialPlugin()
    first = plugin.contribute_artifact(ctx=None)
    second = plugin.contribute_artifact(ctx=None)
    assert first == {}
    assert second == {}
    assert first is not second  # fresh literal, not a shared module-level dict


# --- is_enabled gate --------------------------------------------------------

def test_is_enabled_true_when_knob_true():
    assert Stage2ReamSequentialPlugin().is_enabled(
        {"stage2_reap_ream": {"sequential_reprofile": True}}
    ) is True


def test_is_enabled_false_when_knob_false():
    assert Stage2ReamSequentialPlugin().is_enabled(
        {"stage2_reap_ream": {"sequential_reprofile": False}}
    ) is False


def test_is_enabled_false_when_key_missing():
    """Missing ``sequential_reprofile`` key → OFF (the default)."""
    assert Stage2ReamSequentialPlugin().is_enabled(
        {"stage2_reap_ream": {}}
    ) is False


def test_is_enabled_false_when_block_missing():
    """Missing ``stage2_reap_ream`` block → OFF."""
    assert Stage2ReamSequentialPlugin().is_enabled({}) is False


def test_is_enabled_false_when_config_is_not_dict():
    """Non-dict configs are tolerated — gate returns False."""
    assert Stage2ReamSequentialPlugin().is_enabled(None) is False  # type: ignore[arg-type]
    assert Stage2ReamSequentialPlugin().is_enabled([]) is False  # type: ignore[arg-type]


def test_is_enabled_false_when_block_is_not_dict():
    """Malformed ``stage2_reap_ream`` block (non-dict) → OFF."""
    assert Stage2ReamSequentialPlugin().is_enabled(
        {"stage2_reap_ream": "not-a-dict"}
    ) is False


def test_is_enabled_false_for_falsy_values():
    """``0`` / ``""`` / ``None`` are all treated as OFF."""
    plugin = Stage2ReamSequentialPlugin()
    for falsy in (0, "", None, []):
        assert plugin.is_enabled(
            {"stage2_reap_ream": {"sequential_reprofile": falsy}}
        ) is False, f"value {falsy!r} should be OFF"


def test_is_enabled_true_for_truthy_values():
    """Non-bool truthy values (1, "yes") also enable the plugin."""
    plugin = Stage2ReamSequentialPlugin()
    for truthy in (True, 1, "yes", "true"):
        assert plugin.is_enabled(
            {"stage2_reap_ream": {"sequential_reprofile": truthy}}
        ) is True, f"value {truthy!r} should be ON"


# --- on_post_merge cache invalidation --------------------------------------

class _Sentinel:
    """Distinctive non-None object so the post-call ``is None`` assertion is
    meaningful (compared to e.g. having the test write ``None`` explicitly)."""


def test_on_post_merge_invalidates_three_caches():
    """All three cache slots are set to ``None`` on ``on_post_merge``."""
    ctx = PipelineContext()
    cov_sentinel = _Sentinel()
    ream_sentinel = _Sentinel()
    layer_input_sentinel = _Sentinel()
    ctx.set("cov_acc", cov_sentinel)
    ctx.set("ream_acc", ream_sentinel)
    ctx.set("layer_input_acc", layer_input_sentinel)
    # sanity: ctx really did hold the sentinels
    assert ctx.get("cov_acc") is cov_sentinel
    assert ctx.get("ream_acc") is ream_sentinel
    assert ctx.get("layer_input_acc") is layer_input_sentinel

    Stage2ReamSequentialPlugin().on_post_merge(ctx)

    assert ctx.get("cov_acc") is None
    assert ctx.get("ream_acc") is None
    assert ctx.get("layer_input_acc") is None


def test_on_post_merge_works_when_slots_never_set():
    """``overwrite=True`` upserts — no KeyError if the slot was never written.

    This is the production case for ``cov_acc``: in current code it lives
    as run-scope plugin-instance state on ``LayerMergePlugin``, not a ctx
    slot, so when ``on_post_merge`` first fires for layer 0 the slot is
    not yet written on the per-layer scope.
    """
    ctx = PipelineContext()
    Stage2ReamSequentialPlugin().on_post_merge(ctx)
    # All three slots now exist and resolve to None.
    assert ctx.get("cov_acc") is None
    assert ctx.get("ream_acc") is None
    assert ctx.get("layer_input_acc") is None


def test_on_post_merge_works_in_child_scope():
    """The plugin sets slots on the LOCAL scope (per ``PipelineContext.set``
    semantics), shadowing any parent-scope binding — exactly the behaviour
    required by the per-layer ``ctx = run_ctx.child()`` pattern in
    ``stage2.orchestrator.run``."""
    root = PipelineContext()
    root.set("cov_acc", _Sentinel())  # parent-scope binding survives
    parent_cov = root.get("cov_acc")
    child = root.child()
    # Before: child resolves the parent-scope binding via ``get``.
    assert child.get("cov_acc") is parent_cov

    Stage2ReamSequentialPlugin().on_post_merge(child)

    # After: child sees None locally; the parent-scope binding is unchanged.
    assert child.get("cov_acc") is None
    assert root.get("cov_acc") is parent_cov


def test_on_post_merge_idempotent():
    """Calling the hook twice is safe — each call upserts the slots to None."""
    ctx = PipelineContext()
    ctx.set("cov_acc", _Sentinel())
    ctx.set("ream_acc", _Sentinel())
    ctx.set("layer_input_acc", _Sentinel())
    plugin = Stage2ReamSequentialPlugin()
    plugin.on_post_merge(ctx)
    plugin.on_post_merge(ctx)  # must not raise
    assert ctx.get("cov_acc") is None
    assert ctx.get("ream_acc") is None
    assert ctx.get("layer_input_acc") is None


def test_on_post_merge_returns_none():
    """Phase-walk hooks return ``None`` (the walker discards return values)."""
    ctx = PipelineContext()
    result = Stage2ReamSequentialPlugin().on_post_merge(ctx)
    assert result is None


def test_on_post_merge_tolerates_missing_layer_ref():
    """The optional DEBUG breadcrumb must not KeyError when ``layer_ref`` is
    absent on the ctx — unit-test ctxs do not always populate it."""
    import logging
    ctx = PipelineContext()
    # Force DEBUG so the breadcrumb branch executes.
    logger = logging.getLogger("moe_compress.stage2.plugins.ream_sequential")
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        # Must not raise even though ``layer_ref`` is unset.
        Stage2ReamSequentialPlugin().on_post_merge(ctx)
    finally:
        logger.setLevel(prev_level)


# --- registry wiring --------------------------------------------------------

class _AlwaysOnLayerMergeStub:
    """Minimal always-enabled plugin standing in for ``LayerMergePlugin``."""

    name = "layer_merge"

    def is_enabled(self, config: dict) -> bool:
        return True


def test_registry_includes_plugin_when_knob_on():
    """With ``sequential_reprofile: true``, ``registry.enabled`` includes the
    plugin AFTER the (always-on) layer-merge stand-in — matching the
    orchestrator's "registered last" wiring."""
    layer_merge_stub = _AlwaysOnLayerMergeStub()
    plugin = Stage2ReamSequentialPlugin()
    registry = PluginRegistry([layer_merge_stub, plugin])

    enabled = registry.enabled(
        {"stage2_reap_ream": {"sequential_reprofile": True}}
    )
    assert plugin in enabled
    assert layer_merge_stub in enabled
    # Plugin must run AFTER the layer-merge spine in the enabled subset so
    # any future ``LayerMergePlugin.on_post_merge`` hook fires first.
    assert enabled.index(layer_merge_stub) < enabled.index(plugin)


def test_registry_drops_plugin_when_knob_off():
    """At the OFF default, the plugin is filtered out of ``registry.enabled``
    — guarantees the byte-identical existing path."""
    layer_merge_stub = _AlwaysOnLayerMergeStub()
    plugin = Stage2ReamSequentialPlugin()
    registry = PluginRegistry([layer_merge_stub, plugin])

    # OFF flavour 1: knob set to False
    off_explicit = registry.enabled(
        {"stage2_reap_ream": {"sequential_reprofile": False}}
    )
    assert plugin not in off_explicit
    assert layer_merge_stub in off_explicit

    # OFF flavour 2: knob absent (the default config)
    off_default = registry.enabled({"stage2_reap_ream": {}})
    assert plugin not in off_default
    assert layer_merge_stub in off_default

    # OFF flavour 3: block absent (no Stage 2 config at all)
    off_no_block = registry.enabled({})
    assert plugin not in off_no_block
    assert layer_merge_stub in off_no_block


def test_default_byte_identical_path():
    """The composite "default = byte-identical existing path" claim:
    ``is_enabled`` is False for the default empty config, and
    ``registry.enabled`` therefore drops the plugin entirely."""
    plugin = Stage2ReamSequentialPlugin()
    # 1. Plugin opts itself out at the default.
    assert plugin.is_enabled({}) is False
    # 2. Registry-level: the plugin is not in the enabled tuple.
    registry = PluginRegistry([plugin])
    assert plugin not in registry.enabled({})
    assert registry.enabled({}) == ()


# --- orchestrator source-string contract -----------------------------------

def _orchestrator_source() -> str:
    """Read the orchestrator source for source-string assertions."""
    src_path = (
        pathlib.Path(__file__).parents[1]
        / "src/moe_compress/stage2/orchestrator.py"
    )
    return src_path.read_text()


def test_orchestrator_imports_stage2_ream_sequential_plugin():
    """The orchestrator must import ``Stage2ReamSequentialPlugin``."""
    src = _orchestrator_source()
    assert "from .plugins.ream_sequential import Stage2ReamSequentialPlugin" in src


def test_orchestrator_registers_after_merge_heal():
    """``Stage2ReamSequentialPlugin`` is registered AFTER ``MergeHealPlugin``
    in the ``PluginRegistry`` list, so its ``on_post_merge`` hook runs last
    in the phase-major walk. This is the wiring contract the brief pins:
    "must run AFTER LayerMergePlugin in on_post_merge phase" — LayerMerge
    has no ``on_post_merge``, but ordering ours last also covers any
    future plugin that adopts the hook."""
    src = _orchestrator_source()
    # Both anchors must appear, and the new plugin must come after MergeHeal.
    mh_idx = src.index("MergeHealPlugin(")
    seq_idx = src.index("Stage2ReamSequentialPlugin()")
    assert mh_idx < seq_idx, (
        "Stage2ReamSequentialPlugin must be registered AFTER MergeHealPlugin "
        "in the PluginRegistry list (it owns ``on_post_merge`` cache "
        "invalidation and should fire last)."
    )


def test_orchestrator_registers_after_layer_merge():
    """The brief's explicit ordering constraint: the new plugin's hook must
    run AFTER ``LayerMergePlugin``'s hooks. ``LayerMergePlugin`` does not
    implement ``on_post_merge`` (its merge work happens in ``merge`` /
    ``post_merge``), so this contract is satisfied by phase-schedule
    semantics; for defense-in-depth we still assert registry order."""
    src = _orchestrator_source()
    # ``layer_merge`` is the registered LayerMergePlugin instance variable.
    lm_idx = src.index("\n        layer_merge,\n")
    seq_idx = src.index("Stage2ReamSequentialPlugin()")
    assert lm_idx < seq_idx, (
        "Stage2ReamSequentialPlugin must be registered AFTER the "
        "``layer_merge`` LayerMergePlugin instance in the PluginRegistry list."
    )


# --- integration: Position-B schedule contract (C1/P1 regression guard) ----

def test_orchestrator_walks_phases_with_sequential_reprofile_and_partial_dir(
    tmp_path,
):
    """Regression test for the Position-A crash (Plugin #10 review C1/P1).

    The original Plugin #6 commit placed ``on_post_merge`` BETWEEN
    ``post_merge`` and ``write_artifacts`` (Position A). That made the
    S2_SEQ invalidator fire before ``write_artifacts`` read ``ream_acc``,
    which in production yielded ``AttributeError: 'NoneType' object has
    no attribute ...`` deep inside ``_snapshot_neuron_means_layer`` when
    ``partial_dir`` was set.

    Commit ``8b2177e`` moved the phase to Position B (AFTER
    ``write_artifacts``). This test exercises the live phase walk over
    ``_STAGE2_POST_ASSIGN_PHASES`` with both:

    * a ``ream_acc`` sentinel populated on the layer ctx (the production
      hot path),
    * the real ``Stage2ReamSequentialPlugin`` registered alongside a
      stub spine that reads ``ream_acc`` in its ``write_artifacts`` hook
      and would AttributeError if the slot had been nulled first.

    No AttributeError == the schedule still walks Position B.
    """
    from moe_compress.stage2.orchestrator import _STAGE2_POST_ASSIGN_PHASES
    from moe_compress.tools.phase_walker import walk_phases

    # Build a fake spine plugin that mirrors the LayerMergePlugin contract
    # the reviewer flagged: ``write_artifacts`` reads ``ream_acc`` and
    # dereferences an attribute on it (the production crash signature).
    class _SpineStub:
        name = "layer_merge_stub"
        write_artifacts_saw: object = None

        def is_enabled(self, config: dict) -> bool:
            return True

        def write_artifacts(self, ctx: PipelineContext) -> dict:
            ream_acc = ctx.get("ream_acc")
            # Production code (layer_merge.write_artifacts:625) dereferences
            # ream_acc via ``_snapshot_neuron_means_layer``. Mimic that
            # by reading an attribute — None would AttributeError here, which
            # is the exact crash signature the schedule fix prevents.
            self.write_artifacts_saw = ream_acc.layer_data  # type: ignore[union-attr]
            return {}

    class _Ream:
        layer_data = "live"

    spine = _SpineStub()
    seq = Stage2ReamSequentialPlugin()

    # Per-layer ctx mirroring orchestrator.run's layer child: ream_acc
    # populated as a real (non-None) accumulator, partial_dir present.
    ctx = PipelineContext()
    ctx.set("ream_acc", _Ream())
    ctx.set("layer_ref", object())
    ctx.set("partial_dir", tmp_path / "_partial")
    # Pre-seed the other slots the invalidator will overwrite, so we can
    # independently assert they were nulled after the walk.
    ctx.set("cov_acc", object())
    ctx.set("layer_input_acc", object())

    # Drive the LIVE post-assign phase tuple (imported from orchestrator).
    # If the schedule ever regresses to Position A, the spine's
    # ``write_artifacts`` will AttributeError on ``None.layer_data``.
    walk_phases(_STAGE2_POST_ASSIGN_PHASES, (spine, seq), ctx)

    # write_artifacts ran first and saw the LIVE ream_acc (no Nullification).
    assert spine.write_artifacts_saw == "live", (
        "write_artifacts must observe a live ream_acc; if it sees None, "
        "the ``on_post_merge`` phase has regressed to Position A "
        "(Plugin #10 review C1/P1)."
    )
    # on_post_merge ran AFTER and nulled all three accumulators.
    assert ctx.get("ream_acc") is None
    assert ctx.get("cov_acc") is None
    assert ctx.get("layer_input_acc") is None


def test_post_assign_phases_place_on_post_merge_after_write_artifacts():
    """Pin the Position-B schedule directly on the live phase tuple.

    Complements the integration test above: even if a future refactor
    swapped the phase-walk driver, this assertion catches a regression in
    the static schedule itself (Plugin #6 commit ``8b2177e``)."""
    from moe_compress.stage2.orchestrator import _STAGE2_POST_ASSIGN_PHASES

    phases = list(_STAGE2_POST_ASSIGN_PHASES)
    assert "write_artifacts" in phases
    assert "on_post_merge" in phases
    assert phases.index("write_artifacts") < phases.index("on_post_merge"), (
        "Plugin #10 review C1/P1: ``on_post_merge`` must run AFTER "
        "``write_artifacts`` (Position B). A Position-A schedule will "
        "AttributeError in production when ``sequential_reprofile=True`` "
        "and ``partial_dir != None``."
    )
