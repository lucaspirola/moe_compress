"""Stage 2 — covariance remap + router resize + merge on fused experts."""
from __future__ import annotations

import torch

from moe_compress.stage2_reap_ream import (
    _assign_children_to_centroids,
    _merge_experts_inplace,
    _permutation_align_to_centroid,
    _remap_covariance_for_layer,
    _resize_router_for_kept_experts,
)
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator, ReamCostAccumulator
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


def test_permutation_align_c_act_breaks_tie():
    """C_act term flips the permutation when activation signal overwhelms C_wt."""
    d_int, d_hid = 4, 8
    torch.manual_seed(42)
    ref_gate = torch.randn(d_int, d_hid)
    ref_up   = torch.randn(d_int, d_hid)
    child_gate = ref_gate.clone()
    child_up   = ref_up.clone()

    # Use extreme activation values so C_act saving (200 units) overwhelms
    # any C_wt off-diagonal penalty (~8 units for unit-normal 8-d vectors).
    ref_act   = torch.tensor([100.0, 0.0, 0.5, 0.5])
    child_act = torch.tensor([  0.0, 100.0, 0.5, 0.5])  # first two swapped

    perm_no_act = _permutation_align_to_centroid(ref_gate, ref_up, child_gate, child_up)
    perm_with_act = _permutation_align_to_centroid(
        ref_gate, ref_up, child_gate, child_up,
        ref_act_mean=ref_act, child_act_mean=child_act,
    )

    # Without C_act: identity wins (diagonal C_wt = 0).
    assert list(perm_no_act) == [0, 1, 2, 3]
    # With C_act: swapping neurons 0↔1 saves 200 in activation cost.
    assert perm_with_act[0] == 1 and perm_with_act[1] == 0


def test_merge_experts_inplace_weight_only(tiny_model):
    """_merge_experts_inplace with ream_acc=None uses C_wt only — no crash, weights updated."""
    layer_ref = next(iter_moe_layers(tiny_model))
    banks_before = {
        name: bank.get(0).clone()
        for name, bank in build_banks(layer_ref).items()
    }
    grouped = {0: [0, 1]}  # merge expert 1 into centroid 0
    freq = {0: 3, 1: 1}
    _merge_experts_inplace(layer_ref, grouped, freq, freq_weighted=True, ream_acc=None)
    banks_after = build_banks(layer_ref)
    # Centroid slot must have changed (it's now the weighted average).
    assert not torch.equal(banks_after["gate_proj"].get(0), banks_before["gate_proj"])


def test_merge_experts_inplace_with_ream_acc(tiny_model):
    """_merge_experts_inplace with a populated ream_acc applies C_wt + C_act alignment."""
    layer_ref = next(iter_moe_layers(tiny_model))
    li = layer_ref.layer_idx
    d_int = layer_ref.experts_module.intermediate_dim

    ream_acc = ReamCostAccumulator()
    # Pre-populate neuron means for experts 0 and 1.
    ream_acc._neuron_act_sum[(li, 0)] = torch.ones(d_int)
    ream_acc._neuron_act_count[(li, 0)] = 1
    ream_acc._neuron_act_sum[(li, 1)] = torch.ones(d_int) * 2.0
    ream_acc._neuron_act_count[(li, 1)] = 1

    grouped = {0: [0, 1]}
    freq = {0: 1, 1: 1}
    # Should not raise — C_act branch is exercised.
    _merge_experts_inplace(layer_ref, grouped, freq, freq_weighted=False, ream_acc=ream_acc)
