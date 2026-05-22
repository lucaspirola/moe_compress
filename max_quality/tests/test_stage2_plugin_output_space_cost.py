"""Task 10 / S2-6 — output-space (Direction C) cost plugin module.

Pins the ``OutputSpaceCostPlugin.is_enabled`` truth table and (S2-6) the live
``compute_cost`` slot. Algorithm coverage is provided by the existing
``test_stage2_output_cost.py`` suite. Also guards against monkeypatch drift:
no test may patch one of the 4 moved symbols via the monolith namespace only.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.output_space_cost import OutputSpaceCostPlugin
from moe_compress.utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from moe_compress.utils.model_io import iter_moe_layers


def _cost_kwargs(cov_acc=None, *, cost_alignment_cfg="output"):
    """Representative cost knobs — same shape the orchestrator passes.

    S2-10: the capacity-gate knobs (max_group_cap / capacity_util_threshold /
    cost_asymmetric) moved to CapacityGatePlugin and are no longer cost-plugin
    ctor args. The cost slot now READS the gate's decision off the ctx.
    """
    return dict(
        cov_acc=cov_acc if cov_acc is not None else InputCovarianceAccumulator(),
        cost_alignment_cfg=cost_alignment_cfg,
        cost_whitening="none",
        cost_topk_filter=2,
        cost_output_token_cap=8,
    )


def test_output_branch_uses_plugin_module():
    """The `output` branch of `_ream_cost_matrix` imports `_output_space_cost`
    from the sibling plugin module, not the monolith."""
    import inspect
    import moe_compress.stage2.plugins.ream_cost as ream_cost
    src = inspect.getsource(ream_cost._ream_cost_matrix)
    assert "from .output_space_cost import _output_space_cost" in src
    assert "from ...stage2_reap_ream import _output_space_cost" not in src


# --- OutputSpaceCostPlugin.is_enabled truth table ---------------------------

@pytest.mark.parametrize("cost_alignment,expected", [
    ("output", True),
    ("OUTPUT", True),    # case-insensitive (matches run() .lower() normalize)
    ("pre", False),
    ("post", False),
])
def test_is_enabled_explicit(cost_alignment, expected):
    cfg = {"stage2_reap_ream": {"cost_alignment": cost_alignment}}
    assert OutputSpaceCostPlugin(**_cost_kwargs()).is_enabled(cfg) is expected


def test_is_enabled_default_missing_key():
    """Missing `cost_alignment` -> default 'pre' -> output plugin disabled."""
    assert OutputSpaceCostPlugin(**_cost_kwargs()).is_enabled(
        {"stage2_reap_ream": {}}) is False


def test_is_enabled_missing_block():
    """Missing `stage2_reap_ream` block -> default 'pre' -> output disabled."""
    assert OutputSpaceCostPlugin(**_cost_kwargs()).is_enabled({}) is False


def test_plugin_name():
    assert OutputSpaceCostPlugin.name == "output_space_cost"


# --- S2-6: the live compute_cost slot ---------------------------------------

def test_compute_cost_is_live_slot(tiny_model):
    """S2-6 / S2-10: `OutputSpaceCostPlugin.compute_cost` is a live slot — it
    runs `_ream_cost_matrix` and returns a finite cost matrix of the right
    shape, NOT None.

    S2-10: the capacity gate moved out into ``CapacityGatePlugin.select_alignment``;
    ``compute_cost`` now READS ``effective_cost_alignment`` /
    ``effective_cost_asymmetric`` off the ctx. The test pre-sets them to ``pre``
    (the cheap symmetric path) — exactly what the gate publishes on a slack
    layer — so every entry is finite without the layer-input calibration buffer.
    """
    from moe_compress.stage2.permutation_align import _PermAlignCache

    layer_ref = list(iter_moe_layers(tiny_model))[0]
    n_exp = layer_ref.num_routed_experts
    noncentroid_ids = [0, 1]
    centroid_ids = [e for e in range(n_exp) if e not in (0, 1)]

    ctx = PipelineContext()
    ctx.set("layer_ref", layer_ref)
    ctx.set("ream_acc", ReamCostAccumulator())
    ctx.set("perm_cache", _PermAlignCache())
    ctx.set("layer_input_acc", None)
    ctx.set("freq", {e: 1 for e in range(n_exp)})
    ctx.set("protected", ())
    ctx.set("_iter_ream_centroid_ids", tuple(centroid_ids))
    ctx.set("_iter_ream_noncentroid_ids", tuple(noncentroid_ids))
    ctx.set("_iter_n_ream_c", len(centroid_ids))
    ctx.set("_iter_n_ream_nc", len(noncentroid_ids))
    # Gate decision the orchestrator's select_alignment slot publishes first.
    ctx.set("effective_cost_alignment", "pre")
    ctx.set("effective_cost_asymmetric", False)

    plugin = OutputSpaceCostPlugin(**_cost_kwargs())
    delta = plugin.compute_cost(ctx)

    assert delta is not None
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (len(noncentroid_ids), len(centroid_ids))
    assert np.isfinite(delta).all()


# --- monkeypatch-drift guard (the T9 lesson) --------------------------------

def test_no_monolith_only_monkeypatch_of_moved_symbols():
    """Guard: no test patches a moved output-space symbol via the
    `stage2_reap_ream` namespace only.

    The 4 functions resolve `build_banks` / `_permutation_align_to_centroid`
    from `output_space_cost`'s own namespace. A test that does
    `monkeypatch.setattr(stage2_reap_ream, "<sym>", ...)` for one of the moved
    functions OR for a symbol they use internally would silently no-op after
    T10. Investigation found none; this test fails loudly if a future edit
    introduces one without the matching `output_space_cost` patch.
    """
    moved = {
        "_output_space_cost",
        "_tentative_merged_weights",
        "_router_routing_weights",
        "_swiglu_forward",
    }
    tests_dir = pathlib.Path(__file__).parent
    offenders = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        for sym in moved:
            needle = f'setattr(stage2_reap_ream, "{sym}"'
            if needle in text and "output_space_cost" not in text:
                offenders.append(f"{path.name}: patches {sym} on monolith only")
    assert not offenders, (
        "monolith-only monkeypatch of a moved output-space symbol — add a "
        "matching output_space_cost patch (T9 dual-patch lesson):\n"
        + "\n".join(offenders)
    )
