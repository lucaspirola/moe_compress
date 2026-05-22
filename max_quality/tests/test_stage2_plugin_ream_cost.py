"""Task 8 — REAM cost-matrix plugin module.

Pins the ``ReamCostPrePlugin.is_enabled`` truth table and a CPU-only
``cost_alignment="pre"`` smoke test.
"""
from __future__ import annotations

import numpy as np
import pytest

import moe_compress.stage2.plugins.ream_cost as ream_cost
from moe_compress.stage2.plugins.ream_cost import ReamCostPrePlugin
from moe_compress.utils.activation_hooks import ReamCostAccumulator
from moe_compress.utils.model_io import iter_moe_layers


# --- ReamCostPrePlugin.is_enabled truth table -------------------------------

@pytest.mark.parametrize("cost_alignment,expected", [
    ("pre", True),
    ("PRE", True),       # case-insensitive (matches run() .lower() normalize)
    ("post", False),
    ("output", False),
])
def test_is_enabled_explicit(cost_alignment, expected):
    cfg = {"stage2_reap_ream": {"cost_alignment": cost_alignment}}
    assert ReamCostPrePlugin.is_enabled(cfg) is expected


def test_is_enabled_default_missing_key():
    """Missing `cost_alignment` -> default 'pre' -> enabled."""
    assert ReamCostPrePlugin.is_enabled({"stage2_reap_ream": {}}) is True


def test_is_enabled_missing_block():
    """Missing `stage2_reap_ream` block -> default 'pre' -> enabled."""
    assert ReamCostPrePlugin.is_enabled({}) is True


def test_compute_cost_is_noop():
    """T8: compute_cost is a documented no-op (legacy bump loop still owns it)."""
    assert ReamCostPrePlugin().compute_cost(ctx=None) is None  # type: ignore[arg-type]


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
