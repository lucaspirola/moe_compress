"""Task 9 / S2-6 — post-alignment REAM cost plugin module.

Pins the ``ReamCostPostPlugin.is_enabled`` truth table and (S2-6) the live
``compute_cost`` slot. Algorithm coverage is provided by the existing
``test_stage2_assignment_v2.py`` / ``test_stage2_output_cost.py`` suites.
"""
from __future__ import annotations

import numpy as np
import pytest

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.ream_cost_post import (
    ReamCostPostPlugin,
    _post_alignment_cost,
)
from moe_compress.utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from moe_compress.utils.model_io import iter_moe_layers


def _cost_kwargs(cov_acc=None, *, cost_alignment_cfg="post"):
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
    """S2-6 / S2-10: `ReamCostPostPlugin.compute_cost` is a live slot — it runs
    `_ream_cost_matrix` and returns a finite cost matrix of the right shape,
    NOT None.

    S2-10: the capacity gate moved out into ``CapacityGatePlugin.select_alignment``;
    ``compute_cost`` now READS ``effective_cost_alignment`` /
    ``effective_cost_asymmetric`` off the ctx. The test pre-sets them to ``pre``
    (the cheap symmetric path) — exactly what the gate publishes on a slack
    layer — so every entry is finite without needing input covariance.
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

    plugin = ReamCostPostPlugin(**_cost_kwargs())
    delta = plugin.compute_cost(ctx)

    assert delta is not None
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (len(noncentroid_ids), len(centroid_ids))
    assert np.isfinite(delta).all()


# --- Plugin #14 audit follow-up item 4 / Pattern H hoist polish -------------

def test_post_cost_topk_hoisting_byte_identical():
    """N2 (nitpick): hoisting np.argpartition out of the per-row loop in
    ``_post_alignment_cost`` must select the same K-smallest centroids as
    the pre-hoist per-row form. Mirrors Plugin #3's
    ``test_output_cost_topk_hoisting_byte_identical`` shape (see
    ``test_stage2_output_cost.py``).

    Two cases:
      * No-tie matrix — the vectorized argpartition and the per-row form
        return set-identical K-smallest indices (and the same set as
        ``argsort[:K]``).
      * With-tie matrix — argpartition's order among tied elements is
        implementation-defined; the *set* of K-smallest indices must
        still match per-row, and the K-smallest *values* must equal.

    Pins the hoist contract documented at ``ream_cost_post.py:208`` and
    Plugin #14 audit follow-up item 4.
    """
    # Case 1 — no ties: vectorized vs per-row, set-identical selection.
    rng = np.random.default_rng(seed=42)
    n_nc, n_c = 4, 6
    k_cand = 3
    cheap_cost = rng.random((n_nc, n_c)).astype(np.float64)
    assert len(np.unique(cheap_cost)) == n_nc * n_c, (
        "synthetic cheap_cost should have no ties"
    )

    vectorized = np.argpartition(cheap_cost, k_cand - 1, axis=1)[:, :k_cand]
    assert vectorized.shape == (n_nc, k_cand)

    per_row = np.array([
        np.argpartition(cheap_cost[ci], k_cand - 1)[:k_cand]
        for ci in range(n_nc)
    ])

    for ci in range(n_nc):
        assert set(vectorized[ci].tolist()) == set(per_row[ci].tolist()), (
            f"row {ci}: vectorized {vectorized[ci]} != per-row {per_row[ci]}"
        )
        # Every selected index is one of the K smallest.
        sorted_indices = np.argsort(cheap_cost[ci])[:k_cand]
        assert set(vectorized[ci].tolist()) == set(sorted_indices.tolist()), (
            f"row {ci}: selected indices are not the K smallest"
        )

    # Case 2 — with ties: argpartition order is implementation-defined among
    # tied elements; the set of K-smallest indices must still match per-row,
    # and the selected *values* must equal the K-smallest values.
    tied = np.array([
        [0.1, 0.1, 0.2, 0.3, 0.4, 0.5],  # 0.1 / 0.1 tied
        [0.9, 0.8, 0.8, 0.7, 0.6, 0.6],  # 0.6 / 0.6 tied
        [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],  # 3.0 / 3.0 tied outside top-K
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # all equal
    ], dtype=np.float64)
    k_cand = 3

    vectorized_tied = np.argpartition(tied, k_cand - 1, axis=1)[:, :k_cand]
    per_row_tied = np.array([
        np.argpartition(tied[ci], k_cand - 1)[:k_cand]
        for ci in range(tied.shape[0])
    ])

    for ci in range(tied.shape[0]):
        assert set(vectorized_tied[ci].tolist()) == set(per_row_tied[ci].tolist()), (
            f"tie-row {ci}: vectorized {vectorized_tied[ci]} != "
            f"per-row {per_row_tied[ci]}"
        )
        # Selected values must equal the K-smallest values, even when index
        # sets differ across tie-breaking strategies.
        selected_vals = sorted(tied[ci, vectorized_tied[ci]].tolist())
        smallest_vals = sorted(np.sort(tied[ci])[:k_cand].tolist())
        assert selected_vals == smallest_vals, (
            f"tie-row {ci}: selected values {selected_vals} != "
            f"K-smallest values {smallest_vals}"
        )


def test_post_cost_empty_matrix_early_return(tiny_model):
    """N3 (nitpick): exercise the ``n_nc == 0 or n_c == 0`` early-return at
    ``ream_cost_post.py:205``. The commit message itself (24c24df) flags
    this branch as untested. Two cases:

      * ``noncentroid_ids=[]`` -> ``(0, n_c)`` all-+inf matrix.
      * ``centroid_ids=[]``    -> ``(n_nc, 0)`` all-+inf matrix.

    The all-+inf init is the assignment-solver "forbidden arc" sentinel;
    the early-return preserves that contract without entering the
    argpartition path (which would IndexError on shape[1]==0).
    """
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    n_exp = layer_ref.num_routed_experts

    # Case 1: empty non-centroid set -> (0, n_c) result.
    centroid_ids = list(range(n_exp))
    out_empty_nc = _post_alignment_cost(
        layer_ref,
        noncentroid_ids=[],
        centroid_ids=centroid_ids,
        cheap_cost=np.empty((0, len(centroid_ids)), dtype=np.float64),
        ream_acc=ReamCostAccumulator(),
        cov_acc=None,
        perm_cache=None,
        whitening_mode="none",
        asymmetric=False,
        topk=2,
        freq=None,
    )
    assert isinstance(out_empty_nc, np.ndarray)
    assert out_empty_nc.shape == (0, len(centroid_ids))
    assert out_empty_nc.dtype == np.float64

    # Case 2: empty centroid set -> (n_nc, 0) result.
    noncentroid_ids = list(range(n_exp))
    out_empty_c = _post_alignment_cost(
        layer_ref,
        noncentroid_ids=noncentroid_ids,
        centroid_ids=[],
        cheap_cost=np.empty((len(noncentroid_ids), 0), dtype=np.float64),
        ream_acc=ReamCostAccumulator(),
        cov_acc=None,
        perm_cache=None,
        whitening_mode="none",
        asymmetric=False,
        topk=2,
        freq=None,
    )
    assert isinstance(out_empty_c, np.ndarray)
    assert out_empty_c.shape == (len(noncentroid_ids), 0)
    assert out_empty_c.dtype == np.float64
