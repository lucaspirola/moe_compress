"""Stage 2 — covariance remap + router resize + merge on fused experts."""
from __future__ import annotations

import torch

from moe_compress.stage2_reap_ream import (
    _assign_children_to_centroids,
    _remap_covariance_for_layer,
    _resize_router_for_kept_experts,
)
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
from moe_compress.utils.model_io import build_banks, iter_moe_layers


def test_remap_covariance_keeps_only_centroids():
    cov = InputCovarianceAccumulator()
    cov.covariance = {
        (0, 0, "gate_proj"): torch.eye(3),
        (0, 1, "gate_proj"): torch.eye(3) * 2,
        (0, 2, "gate_proj"): torch.eye(3) * 3,
        (1, 0, "gate_proj"): torch.eye(3) * 9,
    }
    cov.token_count = {k: 10 for k in cov.covariance}

    _remap_covariance_for_layer(cov, layer_idx=0, centroid_ids=[0, 2])

    assert torch.equal(cov.covariance[(0, 0, "gate_proj")], torch.eye(3))
    assert torch.equal(cov.covariance[(0, 1, "gate_proj")], torch.eye(3) * 3)
    assert (0, 2, "gate_proj") not in cov.covariance
    assert torch.equal(cov.covariance[(1, 0, "gate_proj")], torch.eye(3) * 9)


def test_router_resize_updates_top_k_and_num_experts(tiny_model):
    layer_ref = next(iter_moe_layers(tiny_model))
    assert layer_ref.num_routed_experts == 4
    assert layer_ref.router.top_k == 2

    _resize_router_for_kept_experts(layer_ref, kept_ids=[0, 3])
    assert layer_ref.router.weight.shape[0] == 2
    assert layer_ref.router.num_experts == 2
    assert layer_ref.router.top_k == 2

    _resize_router_for_kept_experts(layer_ref, kept_ids=[0])
    assert layer_ref.router.top_k == 1


def test_bank_select_slices_stacked_tensor(tiny_model):
    layer_ref = next(iter_moe_layers(tiny_model))
    banks = build_banks(layer_ref)
    assert banks["down_proj"].num_experts() == 4
    banks["down_proj"].select([0, 2])
    # Both banks that share gate_up_proj should observe the new expert count
    # after a single select call on either gate_proj or up_proj.
    banks_after = build_banks(layer_ref)
    banks_after["gate_proj"].select([0, 2])
    assert banks_after["gate_proj"].num_experts() == 2
    assert banks_after["up_proj"].num_experts() == 2


def test_assign_children_when_more_children_than_centroids():
    import numpy as np
    cost = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.5]])
    assn = _assign_children_to_centroids(cost, n_children=3, n_centroids=2)
    assert assn[:2] == [0, 1]
    assert assn[2] in (0, 1)
