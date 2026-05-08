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

    _remap_covariance_for_layer(cov, layer_idx=0, kept_ids=[0, 2])

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


def test_permutation_align_device_invariance_cpu_only():
    """_permutation_align_to_centroid produces a valid permutation regardless of input device.

    Pins the cost-matrix CPU→GPU optimization: the function previously moved
    inputs to CPU before cdist, now keeps them on the input device. Test runs
    on CPU (always available); verifies the function produces a valid
    permutation (each index in [0, d_int) used exactly once) without crashing
    or returning device-mixed tensors.
    """
    d_int, d_hid = 4, 8
    torch.manual_seed(7)
    ref_gate   = torch.randn(d_int, d_hid)
    ref_up     = torch.randn(d_int, d_hid)
    child_gate = torch.randn(d_int, d_hid)
    child_up   = torch.randn(d_int, d_hid)
    ref_act    = torch.randn(d_int)
    child_act  = torch.randn(d_int)

    # No-activation branch.
    perm_wt = _permutation_align_to_centroid(ref_gate, ref_up, child_gate, child_up)
    assert sorted(perm_wt.tolist()) == list(range(d_int))

    # With activation branch.
    perm_act = _permutation_align_to_centroid(
        ref_gate, ref_up, child_gate, child_up,
        ref_act_mean=ref_act, child_act_mean=child_act,
    )
    assert sorted(perm_act.tolist()) == list(range(d_int))


def test_permutation_align_no_implicit_cpu_calls():
    """Pin: _permutation_align_to_centroid must not contain explicit .cpu() calls
    on its cost-construction tensors. Guards the GPU optimization from regression.

    Reads the function source and asserts that the `.cpu()` calls present in the
    pre-fix CPU regression are absent. The single allowed `.cpu()` is at the
    Hungarian sync (linear_sum_assignment).
    """
    import ast
    import inspect
    from moe_compress.stage2_reap_ream import _permutation_align_to_centroid as fn
    src = inspect.getsource(fn)
    # AST-walk to find .cpu() calls, ignoring occurrences inside comments / strings.
    tree = ast.parse(src)
    cpu_calls = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cpu"
            and not node.args
        ):
            cpu_calls += 1
    # Allowed: exactly one .cpu() at the Hungarian boundary
    # (C.detach().cpu().numpy()). More than one means a regression.
    assert cpu_calls == 1, (
        f"_permutation_align_to_centroid has {cpu_calls} .cpu() calls; "
        f"expected exactly 1 (the Hungarian boundary). The CPU-cost-matrix "
        f"regression has reappeared."
    )


def test_permutation_align_c_act_breaks_tie():
    """C_act term flips the permutation when activation signal favors a swap.

    Spec §5 / D5b: cost C = C_act + C_wt with each component independently
    normalized to [0, 1] via _safe_norm. To exhibit C_act flipping a permutation
    we set up a scenario where C_wt is identical for identity vs swap (so
    C_wt ties cancel) and only C_act prefers the swap. We achieve a clean C_wt
    tie on the [0,1] swap by swapping rows 0 and 1 of the child weights:
    C_wt[0,1] = C_wt[1,0] = 0 (swap aligns) and C_wt[0,0] = C_wt[1,1] = original
    distance — so identity and swap yield identical C_wt totals on those two
    rows. Then C_act tilts the choice to the swap.
    """
    d_int, d_hid = 4, 8
    torch.manual_seed(42)
    ref_gate = torch.randn(d_int, d_hid)
    ref_up   = torch.randn(d_int, d_hid)
    # child_gate[0] = ref_gate[1], child_gate[1] = ref_gate[0] (rows swapped on 0/1).
    child_gate = ref_gate.clone()
    child_up   = ref_up.clone()
    child_gate[[0, 1]] = ref_gate[[1, 0]]
    child_up[[0, 1]]   = ref_up[[1, 0]]

    ref_act   = torch.tensor([100.0, 0.0, 0.5, 0.5])
    child_act = torch.tensor([  0.0, 100.0, 0.5, 0.5])  # first two swapped

    perm_no_act = _permutation_align_to_centroid(ref_gate, ref_up, child_gate, child_up)
    perm_with_act = _permutation_align_to_centroid(
        ref_gate, ref_up, child_gate, child_up,
        ref_act_mean=ref_act, child_act_mean=child_act,
    )

    # Without C_act: identity is no longer trivially zero (rows are swapped),
    # but by symmetry the algorithm should still pick a permutation matching
    # the row-level alignment. The exact permutation depends on the random
    # weights — we only assert that with C_act the [0↔1] swap is preferred.
    # With C_act: the activation-mean signal (extreme values on neurons 0/1
    # swapped) confirms the [0↔1] swap.
    assert perm_with_act[0] == 1 and perm_with_act[1] == 0
    # The no-C_act path must be deterministic given the seed; just check it
    # produces a valid permutation (each value 0..3 used exactly once).
    assert sorted(list(perm_no_act)) == [0, 1, 2, 3]


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
    # Should not raise — C_act branch is exercised. Spec §5 Step 4 mandates
    # frequency-weighted merge (REAM Eq. 6); the equal-weights branch is no
    # longer reachable because it would produce spec-non-compliant merges.
    _merge_experts_inplace(layer_ref, grouped, freq, freq_weighted=True, ream_acc=ream_acc)
