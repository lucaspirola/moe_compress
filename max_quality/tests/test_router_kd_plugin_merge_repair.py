"""RK-6 — Router-KD Stage-2.5 merge-repair plugin extraction tests.

Verifies the RK-6 ``MergeRepairPlugin`` scaffolding in
``router_kd/plugins/merge_repair.py``:

* the seven relocated symbols (``_load_merge_map`` / ``_merged_centroid_rows`` /
  ``_select_merge_repair_layers`` / ``_experts_param_tensors`` /
  ``_unfreeze_merged_experts`` / ``_LayerOutputCapture`` /
  ``_merge_repair_mse``) and ``MergeRepairPlugin`` import from the plugin
  module;
* the ``stage5_router_kd`` monolith re-exports the SAME seven objects (the
  ``# noqa: F401`` re-import block is load-bearing — ``test_stage5_merge_repair``
  imports all seven from the monolith);
* ``MergeRepairPlugin`` satisfies the universal ``PipelinePlugin`` Protocol,
  carries correct metadata, and exposes the (RK-8) merge-repair phase hooks;
* the STAGE-GATED ``is_enabled``: enabled only when bound to ``"stage2p5"``
  AND ``merge_repair.enabled`` is true — always disabled at ``"stage5"`` and
  for a default-constructed plugin;
* the module never imports the ``stage5_router_kd`` monolith or
  ``router_kd.orchestrator`` at any scope (the circular-import contract);
* a light check of the relocated logic itself.

RK-6 is a Pattern A relocation (the seven symbols relocated verbatim, the
monolith re-imports them) plus a Pattern B ``MergeRepairPlugin`` reproducing
the inline ``run()`` merge-repair glue in three INERT hooks. The byte-
identical behavioral gate is the RK-0 golden snapshot
(``test_router_kd_golden_snapshot.py``); the heavy behavioral coverage of the
relocated functions is in ``test_stage5_merge_repair.py``. This file only
checks the relocation plumbing and a light slice of the logic.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


def test_merge_repair_module_imports():
    """All eight RK-6 symbols import from the plugin module."""
    from moe_compress.router_kd.plugins.merge_repair import (
        MergeRepairPlugin,
        _load_merge_map,
        _merged_centroid_rows,
        _select_merge_repair_layers,
        _experts_param_tensors,
        _unfreeze_merged_experts,
        _LayerOutputCapture,
        _merge_repair_mse,
    )

    assert isinstance(MergeRepairPlugin, type)
    assert callable(_load_merge_map)
    assert callable(_merged_centroid_rows)
    assert callable(_select_merge_repair_layers)
    assert callable(_experts_param_tensors)
    assert callable(_unfreeze_merged_experts)
    assert isinstance(_LayerOutputCapture, type)
    assert callable(_merge_repair_mse)


def test_monolith_reexports_merge_repair_symbols():
    """The monolith re-exports the SAME seven merge-repair objects.

    ``is``-identity for ALL seven proves the ``# noqa: F401`` re-import block
    in ``stage5_router_kd.py`` keeps ``run()`` and
    ``test_stage5_merge_repair.py`` on their original import paths.
    """
    from moe_compress import stage5_router_kd
    from moe_compress.router_kd.plugins import merge_repair

    assert stage5_router_kd._load_merge_map is merge_repair._load_merge_map
    assert (
        stage5_router_kd._merged_centroid_rows
        is merge_repair._merged_centroid_rows
    )
    assert (
        stage5_router_kd._select_merge_repair_layers
        is merge_repair._select_merge_repair_layers
    )
    assert (
        stage5_router_kd._experts_param_tensors
        is merge_repair._experts_param_tensors
    )
    assert (
        stage5_router_kd._unfreeze_merged_experts
        is merge_repair._unfreeze_merged_experts
    )
    assert (
        stage5_router_kd._LayerOutputCapture
        is merge_repair._LayerOutputCapture
    )
    assert stage5_router_kd._merge_repair_mse is merge_repair._merge_repair_mse


def test_plugin_satisfies_protocol():
    """``MergeRepairPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.router_kd.plugins.merge_repair import MergeRepairPlugin

    assert isinstance(MergeRepairPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.router_kd.plugins.merge_repair import MergeRepairPlugin

    plugin = MergeRepairPlugin()
    assert plugin.name == "merge_repair"
    assert "2603.02217" in plugin.paper
    assert plugin.config_key == "stage5_router_kd.merge_repair.enabled"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    assert "merge_repair_mse_term" in plugin.writes
    assert "merge_repair_mse_weight" in plugin.writes


def test_is_enabled_stage2p5_gated_on_config():
    """At Stage 2.5 ``is_enabled`` ANDs the stage with ``merge_repair.enabled``.

    No ``merge_repair`` block → False; ``enabled=False`` → False;
    ``enabled=True`` → True.
    """
    from moe_compress.router_kd.plugins.merge_repair import MergeRepairPlugin

    plugin = MergeRepairPlugin(stage_key="stage2p5")
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled(
        {"stage5_router_kd": {"merge_repair": {"enabled": False}}}
    ) is False
    assert plugin.is_enabled(
        {"stage5_router_kd": {"merge_repair": {"enabled": True}}}
    ) is True


def test_is_enabled_stage5_always_disabled():
    """At Stage 5 ``is_enabled`` is False even with ``merge_repair.enabled=True``.

    Merge-repair is a Stage-2.5-only concern — Stage 5 has no Stage-2 merge to
    repair, so the stage gate hard-disables the plugin regardless of config.
    """
    from moe_compress.router_kd.plugins.merge_repair import MergeRepairPlugin

    plugin = MergeRepairPlugin(stage_key="stage5")
    assert plugin.is_enabled(
        {"stage5_router_kd": {"merge_repair": {"enabled": True}}}
    ) is False


def test_default_constructed_plugin_disabled():
    """A default-constructed plugin (stage_key=stage5) is always disabled."""
    from moe_compress.router_kd.plugins.merge_repair import MergeRepairPlugin

    plugin = MergeRepairPlugin()
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled(
        {"stage5_router_kd": {"merge_repair": {"enabled": True}}}
    ) is False


def test_invalid_stage_key_rejected():
    """``MergeRepairPlugin(stage_key="bogus")`` raises ``ValueError``."""
    from moe_compress.router_kd.plugins.merge_repair import MergeRepairPlugin

    with pytest.raises(ValueError, match="unsupported stage_key"):
        MergeRepairPlugin(stage_key="bogus")


def test_plugin_has_hooks():
    """The RK-8 merge-repair phase hooks are present and callable."""
    from moe_compress.router_kd.plugins.merge_repair import MergeRepairPlugin

    plugin = MergeRepairPlugin()
    assert callable(getattr(plugin, "setup_merge_repair", None))
    assert callable(getattr(plugin, "compute_merge_repair_mse", None))
    assert callable(getattr(plugin, "teardown_merge_repair", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage5_router_kd`` / ``router_kd.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator) at any scope — module-top OR function-local — since either
    would risk an import cycle (the monolith re-imports *this* module at load
    time). Parse the source with ``ast`` and walk the FULL tree so a
    function-local ``import stage5_router_kd`` cannot slip past.

    For ``ImportFrom`` both halves are checked: ``node.module`` (the
    ``from X import ...`` package) AND ``node.names`` (the imported symbols) —
    so the cycle-causing ``from moe_compress import stage5_router_kd`` form
    (``module="moe_compress"``, name ``stage5_router_kd``) is also caught.
    """
    from moe_compress.router_kd.plugins import merge_repair as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = ("stage5_router_kd", "router_kd.orchestrator", "orchestrator")
    for node in ast.walk(tree):  # any nesting level, not just module-top
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(f in alias.name for f in forbidden), (
                    f"forbidden import at any scope: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert not any(f in mod_name for f in forbidden), (
                f"forbidden import-from at any scope: {mod_name}"
            )
            # Also inspect the imported NAMES: ``from moe_compress import
            # stage5_router_kd`` carries the monolith as an ``alias.name``, not
            # in ``node.module`` — without this it would slip past undetected.
            for alias in node.names:
                assert not any(f in alias.name for f in forbidden), (
                    f"forbidden import-from name at any scope: "
                    f"from {mod_name} import {alias.name}"
                )


# ---------------------------------------------------------------------------
# Light logic checks — the heavy behavioral coverage lives in
# test_stage5_merge_repair.py.
# ---------------------------------------------------------------------------


def test_merged_centroid_rows_only_real_merges():
    """``_merged_centroid_rows`` returns only rows that absorbed >1 expert."""
    from moe_compress.router_kd.plugins.merge_repair import _merged_centroid_rows

    merge_map = {5: {0: [0, 1, 2], 1: [3], 2: [4, 5], 3: [6]}}
    # Rows 0 and 2 absorbed >1 expert -> merged centroids; rows 1, 3 are
    # length-1 -> untouched, must NOT be returned.
    assert _merged_centroid_rows(merge_map, 5) == [0, 2]
    # A layer not in the map has zero merged centroids.
    assert _merged_centroid_rows(merge_map, 99) == []


def test_merge_repair_mse_matches_manual_mean():
    """``_merge_repair_mse`` is the mean of per-layer MSE terms, with grad."""
    from moe_compress.router_kd.plugins.merge_repair import _merge_repair_mse

    torch.manual_seed(0)
    s = {0: torch.randn(2, 3, 5, requires_grad=True), 7: torch.randn(2, 3, 5)}
    t = {0: torch.randn(2, 3, 5), 7: torch.randn(2, 3, 5)}
    mse = _merge_repair_mse(s, t, [0, 7])
    manual = 0.5 * (
        ((s[0].detach() - t[0]) ** 2).mean() + ((s[7] - t[7]) ** 2).mean()
    )
    assert torch.allclose(mse, manual, atol=1e-6)
    # Grad flows back into the student layer outputs.
    mse.backward()
    assert s[0].grad is not None
    assert s[0].grad.abs().sum().item() > 0.0
