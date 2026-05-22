"""Task 8 / S2-6 — REAM cost-matrix plugin module.

Pins the ``ReamCostPrePlugin.is_enabled`` truth table, a CPU-only
``cost_alignment="pre"`` smoke test, and (S2-6) the live ``compute_cost`` slot:
the plugin services the ``compute_cost`` assignment slot and produces a finite
cost matrix.
"""
from __future__ import annotations

import numpy as np
import pytest

import moe_compress.stage2.plugins.ream_cost as ream_cost
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.ream_cost import ReamCostPrePlugin
from moe_compress.utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from moe_compress.utils.model_io import iter_moe_layers


# Representative cost knobs for constructing a cost plugin in a unit test —
# the same shape the orchestrator passes.
# S2-10: the capacity-gate knobs (max_group_cap / capacity_util_threshold /
# cost_asymmetric) moved to CapacityGatePlugin and are no longer cost-plugin
# ctor args.
def _cost_kwargs(cov_acc=None, *, cost_alignment_cfg="pre"):
    return dict(
        cov_acc=cov_acc if cov_acc is not None else InputCovarianceAccumulator(),
        cost_alignment_cfg=cost_alignment_cfg,
        cost_whitening="none",
        cost_topk_filter=2,
        cost_output_token_cap=8,
    )


# --- ReamCostPrePlugin.is_enabled truth table -------------------------------

@pytest.mark.parametrize("cost_alignment,expected", [
    ("pre", True),
    ("PRE", True),       # case-insensitive (matches run() .lower() normalize)
    ("post", False),
    ("output", False),
])
def test_is_enabled_explicit(cost_alignment, expected):
    cfg = {"stage2_reap_ream": {"cost_alignment": cost_alignment}}
    assert ReamCostPrePlugin(**_cost_kwargs()).is_enabled(cfg) is expected


def test_is_enabled_default_missing_key():
    """Missing `cost_alignment` -> default 'pre' -> enabled."""
    assert ReamCostPrePlugin(**_cost_kwargs()).is_enabled(
        {"stage2_reap_ream": {}}) is True


def test_is_enabled_missing_block():
    """Missing `stage2_reap_ream` block -> default 'pre' -> enabled."""
    assert ReamCostPrePlugin(**_cost_kwargs()).is_enabled({}) is True


def test_plugin_name():
    assert ReamCostPrePlugin.name == "ream_cost_pre"


# --- CPU-only "pre" smoke test ----------------------------------------------

def test_ream_cost_matrix_pre_smoke(tiny_model):
    """`_ream_cost_matrix(cost_alignment="pre")` on the synthetic layer:
    empty accumulator -> degenerate full-0.5 sim path -> finite cost in [0,1],
    correct shape, no monolith back-import touched."""
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    n_exp = layer_ref.num_routed_experts
    ream_acc = ReamCostAccumulator()  # empty -> neutral 0.5 sim_expert
    noncentroid_ids = [0, 1]
    centroid_ids = [e for e in range(n_exp) if e not in (0, 1)]

    cost = ream_cost._ream_cost_matrix(
        layer_ref, noncentroid_ids, centroid_ids,
        ream_acc=ream_acc, cost_alignment="pre",
    )
    assert cost.shape == (len(noncentroid_ids), len(centroid_ids))
    assert np.isfinite(cost).all()
    assert (cost >= 0.0).all() and (cost <= 1.0).all()


# --- S2-6: the live compute_cost slot ---------------------------------------

def _prepare_cost_ctx(tiny_model):
    """Build a per-layer ctx with every slot ``_compute_cost_for_plugin``
    reads, on the first synthetic MoE layer. Returns ``(layer_ref, ctx)``.

    The cost slot reads: ``layer_ref``, ``ream_acc``, ``perm_cache``,
    ``layer_input_acc``, ``freq``, ``protected`` and the four ``_iter_*``
    bump-loop scratch slots the orchestrator publishes before each call.
    """
    from moe_compress.stage2.permutation_align import _PermAlignCache

    layer_ref = list(iter_moe_layers(tiny_model))[0]
    n_exp = layer_ref.num_routed_experts
    noncentroid_ids = [0, 1]
    centroid_ids = [e for e in range(n_exp) if e not in (0, 1)]

    ctx = PipelineContext()
    ctx.set("layer_ref", layer_ref)
    ctx.set("ream_acc", ReamCostAccumulator())  # empty -> neutral 0.5 sim
    ctx.set("perm_cache", _PermAlignCache())
    ctx.set("layer_input_acc", None)
    ctx.set("freq", {e: 1 for e in range(n_exp)})
    ctx.set("protected", ())
    ctx.set("_iter_ream_centroid_ids", tuple(centroid_ids))
    ctx.set("_iter_ream_noncentroid_ids", tuple(noncentroid_ids))
    ctx.set("_iter_n_ream_c", len(centroid_ids))
    ctx.set("_iter_n_ream_nc", len(noncentroid_ids))
    return layer_ref, ctx


def test_compute_cost_is_live_slot(tiny_model):
    """S2-6 / S2-10: `ReamCostPrePlugin.compute_cost` is a live slot — it runs
    `_ream_cost_matrix` and returns a finite cost matrix of the right shape,
    NOT None.

    S2-10: the capacity gate moved out into ``CapacityGatePlugin.select_alignment``;
    ``compute_cost`` now READS ``effective_cost_alignment`` /
    ``effective_cost_asymmetric`` off the ctx, so the test pre-sets them (as the
    orchestrator's earlier ``select_alignment`` dispatch would).
    """
    layer_ref, ctx = _prepare_cost_ctx(tiny_model)
    n_nc = ctx.get("_iter_n_ream_nc")
    n_c = ctx.get("_iter_n_ream_c")
    # Gate decision the orchestrator's select_alignment slot publishes first.
    ctx.set("effective_cost_alignment", "pre")
    ctx.set("effective_cost_asymmetric", False)

    plugin = ReamCostPrePlugin(**_cost_kwargs())
    delta = plugin.compute_cost(ctx)

    assert delta is not None
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (n_nc, n_c)
    assert np.isfinite(delta).all()

