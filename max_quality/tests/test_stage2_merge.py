"""Stage 2 — covariance remap and router resize."""
from __future__ import annotations

import torch

from moe_compress.stage2_reap_ream import (
    _assign_children_to_centroids,
    _ream_cost_matrix,
    _remap_covariance_for_layer,
    _resize_router_for_kept_experts,
)
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
from moe_compress.utils.model_io import iter_moe_layers


def test_remap_covariance_keeps_only_centroids():
    cov = InputCovarianceAccumulator()
    cov.covariance = {
        (0, 0, "gate_proj"): torch.eye(3),
        (0, 1, "gate_proj"): torch.eye(3) * 2,
        (0, 2, "gate_proj"): torch.eye(3) * 3,
        (1, 0, "gate_proj"): torch.eye(3) * 9,
    }
    cov.token_count = {k: 10 for k in cov.covariance}

    # Keep original experts 0 and 2 from layer 0 (renumbered to new 0 and 1)
    _remap_covariance_for_layer(cov, layer_idx=0, centroid_ids=[0, 2])

    # Layer 0: new key 0 ← old 0, new key 1 ← old 2; expert 1 dropped.
    assert torch.equal(cov.covariance[(0, 0, "gate_proj")], torch.eye(3))
    assert torch.equal(cov.covariance[(0, 1, "gate_proj")], torch.eye(3) * 3)
    assert (0, 2, "gate_proj") not in cov.covariance
    # Layer 1: untouched
    assert torch.equal(cov.covariance[(1, 0, "gate_proj")], torch.eye(3) * 9)


def test_router_resize_updates_top_k_and_num_experts(tiny_model):
    layer_ref = next(iter_moe_layers(tiny_model))
    # original: num_experts=4, top_k=2
    assert layer_ref.mlp.num_experts == 4
    assert layer_ref.mlp.top_k == 2

    _resize_router_for_kept_experts(layer_ref, kept_ids=[0, 3])
    assert layer_ref.router.weight.shape[0] == 2
    assert layer_ref.mlp.num_experts == 2
    assert layer_ref.mlp.top_k == 2   # already ≤ 2, no clamp
    _resize_router_for_kept_experts(layer_ref, kept_ids=[0])   # further prune
    assert layer_ref.mlp.top_k == 1


def test_assign_children_when_more_children_than_centroids():
    import numpy as np
    cost = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.5],
    ])
    assn = _assign_children_to_centroids(cost, n_children=3, n_centroids=2)
    # Child 0 → centroid 0; child 1 → centroid 1 (Hungarian on first 2).
    assert assn[:2] == [0, 1]
    # Child 2 assigned via next-hungarian pass.
    assert assn[2] in (0, 1)
