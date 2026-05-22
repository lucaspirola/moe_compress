"""Unit tests for the Stage 2 pipeline scaffolding (Task 1).

Validates the plugin ABC contract, registry ordering + filtering,
``dispatch_first`` short-circuit semantics, and ``Stage2Pipeline`` setup /
teardown fan-out. No torch / numpy needed at this layer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from moe_compress.stage2._framework import (
    LayerContext,
    PluginRegistry,
    RunContext,
    Stage2Pipeline,
    Stage2Plugin,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingPlugin(Stage2Plugin):
    """Records every lifecycle call so tests can assert on dispatch order."""

    name = "recording"
    enabled_by = ()

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def on_run_setup(self, run_ctx):
        self.calls.append(("on_run_setup", run_ctx))

    def on_run_teardown(self, run_ctx):
        self.calls.append(("on_run_teardown", run_ctx))


class _FlaggedPlugin(Stage2Plugin):
    name = "flagged"
    enabled_by = ("my_flag",)


class _UnflaggedPlugin(Stage2Plugin):
    name = "unflagged"
    enabled_by = ()


class _MultiFlagPlugin(Stage2Plugin):
    name = "multi"
    enabled_by = ("flag_a", "flag_b")


class _NoneCostPlugin(Stage2Plugin):
    name = "none_cost"
    enabled_by = ()
    # compute_cost inherits the no-op default returning None.


class _FixedCostPlugin(Stage2Plugin):
    name = "fixed_cost"
    enabled_by = ()

    def compute_cost(self, ctx):  # type: ignore[override]
        return "delta-from-fixed"


class _BoomCostPlugin(Stage2Plugin):
    name = "boom"
    enabled_by = ()

    def compute_cost(self, ctx):  # type: ignore[override]
        raise AssertionError("dispatch_first should have short-circuited before me")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_ctx(tmp_path: Path, cfg: dict | None = None) -> RunContext:
    return RunContext(
        model=object(),
        tokenizer=object(),
        config=cfg or {},
        artifacts_dir=tmp_path,
        partial_dir=tmp_path / "_partial",
        device="cpu",
    )


def _make_layer_ctx() -> LayerContext:
    return LayerContext(layer_idx=0, layer_ref=object(), n_experts=4, target=2)


# ---------------------------------------------------------------------------
# Hook surface
# ---------------------------------------------------------------------------


def test_stage2plugin_exposes_all_lifecycle_hooks():
    """Stage2Plugin must expose every lifecycle hook as a callable for subclasses to override."""
    hook_names = [
        "on_run_setup",
        "on_run_teardown",
        "on_layer_setup",
        "on_profile",
        "on_score",
        "compute_cost",
        "apply_cost_mask",
        "solve_assignment",
        "refine_assignment",
        "compute_assignment",
        "pre_merge_snapshot",
        "merge",
        "post_merge",
        "write_artifacts",
        "on_layer_teardown",
    ]
    for hook in hook_names:
        attr = getattr(Stage2Plugin, hook, None)
        assert callable(attr), f"Stage2Plugin missing callable hook: {hook}"


def test_default_hooks_are_noops():
    """All lifecycle hooks have safe no-op defaults that return None (or {} for write_artifacts)."""

    class _Bare(Stage2Plugin):
        name = "bare"

    plugin = _Bare()
    run_ctx = _make_run_ctx(Path("/tmp"))
    layer_ctx = _make_layer_ctx()
    assert plugin.on_run_setup(run_ctx) is None
    assert plugin.on_run_teardown(run_ctx) is None
    assert plugin.on_layer_setup(layer_ctx) is None
    assert plugin.on_profile(layer_ctx) is None
    assert plugin.on_score(layer_ctx) is None
    assert plugin.compute_cost(layer_ctx) is None
    assert plugin.apply_cost_mask(layer_ctx, delta=None) is None
    assert plugin.solve_assignment(layer_ctx, delta=None) is None
    assert plugin.refine_assignment(layer_ctx, asg=None, delta=None) is None
    assert plugin.compute_assignment(layer_ctx) is None
    assert plugin.pre_merge_snapshot(layer_ctx) is None
    assert plugin.merge(layer_ctx) is None
    assert plugin.post_merge(layer_ctx) is None
    assert plugin.write_artifacts(layer_ctx, partial_dir=Path("/tmp")) == {}
    assert plugin.on_layer_teardown(layer_ctx) is None


def test_is_enabled_default_no_flags_returns_true():
    """A plugin with no enabled_by flags is always enabled."""
    assert _UnflaggedPlugin.is_enabled({}) is True
    assert _UnflaggedPlugin.is_enabled({"stage2_reap_ream": {}}) is True


def test_is_enabled_requires_all_flags_truthy():
    """is_enabled returns True iff every flag in enabled_by is truthy in cfg['stage2_reap_ream']."""
    assert _FlaggedPlugin.is_enabled({"stage2_reap_ream": {"my_flag": True}}) is True
    assert _FlaggedPlugin.is_enabled({"stage2_reap_ream": {"my_flag": False}}) is False
    assert _FlaggedPlugin.is_enabled({"stage2_reap_ream": {}}) is False
    assert _FlaggedPlugin.is_enabled({}) is False
    # Multi-flag: all must be truthy
    assert _MultiFlagPlugin.is_enabled(
        {"stage2_reap_ream": {"flag_a": 1, "flag_b": "yes"}}
    ) is True
    assert _MultiFlagPlugin.is_enabled(
        {"stage2_reap_ream": {"flag_a": 1, "flag_b": 0}}
    ) is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_register_preserves_insertion_order():
    """PluginRegistry.classes() returns the registered classes in insertion order."""
    reg = PluginRegistry()
    reg.register(_UnflaggedPlugin)
    reg.register(_FlaggedPlugin)
    reg.register(_RecordingPlugin)
    assert reg.classes() == [_UnflaggedPlugin, _FlaggedPlugin, _RecordingPlugin]


def test_register_rejects_non_plugin_classes():
    """Registering a non-Stage2Plugin class raises TypeError."""
    reg = PluginRegistry()
    with pytest.raises(TypeError):
        reg.register(object)  # type: ignore[arg-type]


def test_active_filters_by_is_enabled_and_preserves_order():
    """active(cfg) instantiates only enabled plugins, preserving registration order."""
    reg = PluginRegistry()
    reg.register(_FlaggedPlugin)         # gated by 'my_flag'
    reg.register(_UnflaggedPlugin)       # always on
    reg.register(_MultiFlagPlugin)       # gated by 'flag_a' + 'flag_b'

    # Only the unflagged plugin is enabled.
    cfg_off: dict = {"stage2_reap_ream": {}}
    active_off = reg.active(cfg_off)
    assert [type(p) for p in active_off] == [_UnflaggedPlugin]

    # All three enabled — registration order preserved.
    cfg_on: dict = {
        "stage2_reap_ream": {"my_flag": True, "flag_a": True, "flag_b": True}
    }
    active_on = reg.active(cfg_on)
    assert [type(p) for p in active_on] == [
        _FlaggedPlugin,
        _UnflaggedPlugin,
        _MultiFlagPlugin,
    ]


def test_dispatch_first_returns_first_non_none():
    """dispatch_first returns the first non-None result and skips later plugins."""
    plugins = [_NoneCostPlugin(), _FixedCostPlugin(), _BoomCostPlugin()]
    ctx = _make_layer_ctx()
    result = PluginRegistry.dispatch_first(plugins, "compute_cost", ctx)
    assert result == "delta-from-fixed"


def test_dispatch_first_returns_none_when_no_plugin_provides():
    """dispatch_first returns None when every plugin's hook returns None."""
    plugins = [_NoneCostPlugin(), _NoneCostPlugin()]
    ctx = _make_layer_ctx()
    assert PluginRegistry.dispatch_first(plugins, "compute_cost", ctx) is None


# ---------------------------------------------------------------------------
# Pipeline shell
# ---------------------------------------------------------------------------


def test_pipeline_phases_are_declared_in_canonical_order():
    """Stage2Pipeline.phases documents the per-layer execution order.

    The T7 tuple is 9 entries. The four fine-grained sub-hooks
    (``compute_cost``, ``apply_cost_mask``, ``solve_assignment``,
    ``refine_assignment``) remain DECLARED on ``Stage2Plugin`` but are not
    iterated by ``Stage2Pipeline.run_layer`` — the LegacyAdapter folds them
    into ``compute_assignment`` to preserve the bump-loop control flow.
    Tasks 8/9/13/14/15 will re-introduce them when the algorithm plugins
    take over from the legacy adapter.
    """
    assert Stage2Pipeline.phases == (
        "on_layer_setup",
        "on_profile",
        "on_score",
        "compute_assignment",
        "pre_merge_snapshot",
        "merge",
        "post_merge",
        "write_artifacts",
        "on_layer_teardown",
    )


def test_run_setup_and_teardown_fan_out_in_order(tmp_path):
    """run_setup / run_teardown invoke every plugin's hook in registration order."""
    a, b = _RecordingPlugin(), _RecordingPlugin()
    pipeline = Stage2Pipeline(plugins=[a, b])
    run_ctx = _make_run_ctx(tmp_path)

    pipeline.run_setup(run_ctx)
    pipeline.run_teardown(run_ctx)

    assert [name for name, _ in a.calls] == ["on_run_setup", "on_run_teardown"]
    assert [name for name, _ in b.calls] == ["on_run_setup", "on_run_teardown"]
    # Same RunContext instance passed to every plugin.
    assert all(arg is run_ctx for _, arg in a.calls)
    assert all(arg is run_ctx for _, arg in b.calls)


def test_run_context_is_frozen():
    """RunContext is immutable so plugins cannot accidentally mutate run-scope state."""
    rc = _make_run_ctx(Path("/tmp"))
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError / dataclasses.FrozenInstanceError
        rc.device = "cuda"  # type: ignore[misc]


def test_run_context_has_no_extras_field():
    """RunContext intentionally has no `extras` escape hatch — plugin-private state lives on LayerContext."""
    rc = _make_run_ctx(Path("/tmp"))
    assert not hasattr(rc, "extras")


def test_layer_context_is_mutable_and_has_extras_dict():
    """LayerContext is mutable and provides a free-form extras dict for plugins."""
    lc = _make_layer_ctx()
    lc.scores = "fake-scores"
    lc.extras["my_plugin"] = {"foo": 1}
    assert lc.scores == "fake-scores"
    assert lc.extras == {"my_plugin": {"foo": 1}}
