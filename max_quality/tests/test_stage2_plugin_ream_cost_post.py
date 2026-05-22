"""Task 9 / S2-6 — post-alignment REAM cost plugin module.

Pins the ``ReamCostPostPlugin.is_enabled`` truth table and (S2-6) the live
``compute_cost`` slot. Algorithm coverage is provided by the existing
``test_stage2_assignment_v2.py`` / ``test_stage2_output_cost.py`` suites.
"""
from __future__ import annotations

import numpy as np
import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.ream_cost_post import ReamCostPostPlugin
from moe_compress.utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from moe_compress.utils.model_io import iter_moe_layers


def _cost_kwargs(cov_acc=None, *, cost_alignment_cfg="post"):
    """Representative cost knobs — same shape the orchestrator passes.

    ``max_group_cap=0`` (uncapped) makes the capacity gate treat every layer
    as fully slack (u = 0); ``capacity_util_threshold=1.0`` then downgrades the
    configured ``post`` -> ``pre`` per layer (the documented gate behaviour),
    so the live ``compute_cost`` slot runs the cheap symmetric path without
    needing input covariance.
    """
    return dict(
        cov_acc=cov_acc if cov_acc is not None else InputCovarianceAccumulator(),
        max_group_cap=0,
        capacity_util_threshold=1.0,
        cost_alignment_cfg=cost_alignment_cfg,
        cost_asymmetric=False,
        cost_whitening="none",
        cost_topk_filter=2,
        cost_output_token_cap=8,
    )


# --- ReamCostPostPlugin.is_enabled truth table ------------------------------

@pytest.mark.parametrize("cost_alignment,expected", [
    ("post", True),
    ("POST", True),      # case-insensitive (matches run() .lower() normalize)
    ("pre", False),
    ("output", False),
])
def test_is_enabled_explicit(cost_alignment, expected):
    cfg = {"stage2_reap_ream": {"cost_alignment": cost_alignment}}
    assert ReamCostPostPlugin(**_cost_kwargs()).is_enabled(cfg) is expected


def test_is_enabled_default_missing_key():
    """Missing `cost_alignment` -> default 'pre' -> post plugin disabled."""
    assert ReamCostPostPlugin(**_cost_kwargs()).is_enabled(
        {"stage2_reap_ream": {}}) is False


def test_is_enabled_missing_block():
    """Missing `stage2_reap_ream` block -> default 'pre' -> post disabled."""
    assert ReamCostPostPlugin(**_cost_kwargs()).is_enabled({}) is False


def test_plugin_name():
    assert ReamCostPostPlugin.name == "ream_cost_post"


# --- S2-6: the live compute_cost slot ---------------------------------------

def test_compute_cost_is_live_slot(tiny_model):
    """S2-6: `ReamCostPostPlugin.compute_cost` is a live slot — it runs the
    capacity-util gate + `_ream_cost_matrix` and returns a finite cost matrix
    of the right shape, NOT None.

    With ``max_group_cap=0`` the capacity gate downgrades ``post`` -> ``pre``
    (the configured ``post`` mode is unreachable when the cap is uncapped), so
    the cheap symmetric path runs and every entry is finite — exactly the
    documented per-layer gate behaviour.
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

    plugin = ReamCostPostPlugin(**_cost_kwargs())
    delta = plugin.compute_cost(ctx)

    assert delta is not None
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (len(noncentroid_ids), len(centroid_ids))
    assert np.isfinite(delta).all()
    # max_group_cap=0 -> capacity gate downgrades the configured 'post' to
    # 'pre' (uncapped layers always take the cheap symmetric path).
    assert ctx.get("effective_cost_alignment") == "pre"
