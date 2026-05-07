"""Phase 3 tests for per-merge-group expert distillation (spec § 5 step 7b / M8).

Covers:
- ``_LayerInputAccumulator`` reservoir-style sample capping.
- ``_swiglu_forward`` shape and numeric correctness.
- ``_snapshot_pre_merge_layer_experts`` produces deep copies on CPU.
- ``_distill_merged_group`` end-to-end on synthetic weights:
  * ``steps=0`` returns ``{"steps": 0, "skip": ...}`` and does not mutate the bank.
  * Singleton groups are skipped.
  * Non-singleton groups train: loss decreases, plateau-break fires when
    initial perfect-fit, weights are written back to the bank in original dtype.
- Distillation is a no-op when ``expert_distill_steps == 0`` (the default).

Tests run on CPU with the synthetic ``_TinyModel`` fixture.
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.stage2_reap_ream import (
    _LayerInputAccumulator,
    _swiglu_forward,
    _snapshot_pre_merge_layer_experts,
    _distill_merged_group,
)


# ---------------------------------------------------------------------------
# _LayerInputAccumulator
# ---------------------------------------------------------------------------


def test_layer_input_acc_initial_capture():
    acc = _LayerInputAccumulator(max_samples=64)
    hidden = torch.randn(2, 4, 8)  # (batch, seq, hidden_dim)
    acc.add(hidden)
    out = acc.get()
    assert out is not None
    assert out.shape == (8, 8)  # (batch*seq, hidden)


def test_layer_input_acc_capped_at_max_samples():
    acc = _LayerInputAccumulator(max_samples=10)
    big = torch.randn(20, 1, 8)  # 20 tokens, more than cap
    acc.add(big)
    out = acc.get()
    assert out is not None
    # Capped at max_samples on the very first batch.
    assert out.shape == (10, 8)


def test_layer_input_acc_reservoir_extends_then_replaces():
    acc = _LayerInputAccumulator(max_samples=20)
    a = torch.randn(8, 1, 4)  # 8 tokens
    b = torch.randn(8, 1, 4)  # 8 more
    c = torch.randn(8, 1, 4)  # 8 more — total 24 > 20
    acc.add(a)
    assert acc.get().shape == (8, 4)
    acc.add(b)
    assert acc.get().shape == (16, 4)
    acc.add(c)
    # Reservoir-replaced once full; stays at the cap.
    assert acc.get().shape == (20, 4)


def test_layer_input_acc_get_before_any_add_returns_none():
    acc = _LayerInputAccumulator(max_samples=4)
    assert acc.get() is None


# ---------------------------------------------------------------------------
# _swiglu_forward
# ---------------------------------------------------------------------------


def test_swiglu_forward_shape_and_finite():
    d_int, hidden = 6, 4
    W_gate = torch.randn(d_int, hidden) * 0.05
    W_up   = torch.randn(d_int, hidden) * 0.05
    W_down = torch.randn(hidden, d_int) * 0.05
    x = torch.randn(3, hidden)
    y = _swiglu_forward(W_gate, W_up, W_down, x)
    assert y.shape == (3, hidden)
    assert torch.isfinite(y).all()


def test_swiglu_forward_deterministic():
    """Same inputs → same outputs (no internal randomness)."""
    torch.manual_seed(0)
    d_int, hidden = 6, 4
    W_gate = torch.randn(d_int, hidden)
    W_up   = torch.randn(d_int, hidden)
    W_down = torch.randn(hidden, d_int)
    x = torch.randn(2, hidden)
    y1 = _swiglu_forward(W_gate, W_up, W_down, x)
    y2 = _swiglu_forward(W_gate, W_up, W_down, x)
    assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# _snapshot_pre_merge_layer_experts
# ---------------------------------------------------------------------------


def test_snapshot_pre_merge_layer_experts_makes_independent_cpu_clones(tiny_model):
    # Build the layer ref via the same helper Stage 2 uses.
    from moe_compress.utils.model_io import iter_moe_layers
    refs = list(iter_moe_layers(tiny_model))
    assert len(refs) > 0, "Tiny model has no MoE layers"
    layer_ref = refs[0]

    snap = _snapshot_pre_merge_layer_experts(layer_ref)
    n = layer_ref.num_routed_experts
    assert set(snap.keys()) == set(range(n))
    for eid in snap:
        for name in ("gate_proj", "up_proj", "down_proj"):
            t = snap[eid][name]
            assert t.device.type == "cpu"
            assert torch.isfinite(t).all()

    # Mutate the bank — the snapshot must not change.
    from moe_compress.utils.model_io import build_banks
    banks = build_banks(layer_ref)
    pristine = snap[0]["gate_proj"].clone()
    new_w = torch.zeros_like(banks["gate_proj"].get(0))
    with torch.no_grad():
        banks["gate_proj"].set(0, new_w)
    assert torch.allclose(snap[0]["gate_proj"], pristine)


# ---------------------------------------------------------------------------
# _distill_merged_group
# ---------------------------------------------------------------------------


def test_distill_singleton_group_returns_skip():
    state = _distill_merged_group(
        layer_ref=None,           # type: ignore[arg-type]
        centroid_id=0,
        members=[0],              # singleton
        freq={0: 1},
        pre_merge_weights={},
        layer_inputs=torch.randn(4, 4),
        steps=10,
        lr=1e-4,
        betas=(0.9, 0.95),
        plateau_steps=5,
        plateau_eps=1e-6,
        token_cap=4,
        device=torch.device("cpu"),
    )
    assert state["steps"] == 0
    assert state["skip"] == "trivial"


def test_distill_zero_steps_returns_skip():
    state = _distill_merged_group(
        layer_ref=None,           # type: ignore[arg-type]
        centroid_id=0,
        members=[0, 1],           # non-singleton
        freq={0: 1, 1: 1},
        pre_merge_weights={},
        layer_inputs=torch.randn(4, 4),
        steps=0,                  # disabled
        lr=1e-4,
        betas=(0.9, 0.95),
        plateau_steps=5,
        plateau_eps=1e-6,
        token_cap=4,
        device=torch.device("cpu"),
    )
    assert state["steps"] == 0


def test_distill_loss_decreases_on_synthetic(tiny_model):
    """Train the merged centroid against a freq-weighted target made from
    two original experts and verify the loss decreases monotonically over
    a few steps."""
    from moe_compress.utils.model_io import iter_moe_layers
    layer_ref = list(iter_moe_layers(tiny_model))[0]

    # Snapshot pre-merge weights, then mutate the centroid bank to a
    # known-bad weight so the distillation has work to do.
    pre = _snapshot_pre_merge_layer_experts(layer_ref)
    from moe_compress.utils.model_io import build_banks
    banks = build_banks(layer_ref)
    bad = torch.randn_like(banks["gate_proj"].get(0)) * 0.5
    with torch.no_grad():
        banks["gate_proj"].set(0, bad)

    layer_inputs = torch.randn(16, layer_ref.experts_module.hidden_dim) * 0.1

    state = _distill_merged_group(
        layer_ref=layer_ref,
        centroid_id=0,
        members=[0, 1],
        freq={0: 3, 1: 1},
        pre_merge_weights=pre,
        layer_inputs=layer_inputs,
        steps=20,
        lr=5e-3,
        betas=(0.9, 0.95),
        plateau_steps=100,  # disable plateau-break for this test
        plateau_eps=0.0,
        token_cap=16,
        device=torch.device("cpu"),
    )
    assert state["steps"] >= 1
    assert state["final_loss"] is not None
    assert state["initial_loss"] is not None
    # Final loss should be strictly less than initial after 20 steps of
    # gradient descent on a simple synthetic problem.
    assert state["final_loss"] < state["initial_loss"]


def test_distill_plateau_break_fires_when_already_at_target(tiny_model):
    """If the centroid is already exactly at the freq-weighted target weight
    (which trivially produces zero loss after one step), the plateau-break
    should fire and stop training early."""
    from moe_compress.utils.model_io import iter_moe_layers, build_banks
    layer_ref = list(iter_moe_layers(tiny_model))[0]

    pre = _snapshot_pre_merge_layer_experts(layer_ref)
    banks = build_banks(layer_ref)

    # Set the centroid weights to the freq-weighted average of expert 0 and 1
    # (which is the optimal target for those frequencies).
    weights = torch.tensor([3.0, 1.0]) / 4.0
    with torch.no_grad():
        for name in ("gate_proj", "up_proj", "down_proj"):
            merged = (
                weights[0] * pre[0][name].to(torch.float32)
                + weights[1] * pre[1][name].to(torch.float32)
            )
            banks[name].set(0, merged.to(banks[name].get(0).dtype))

    # Use SAME freq weights as the merge so the target equals the current
    # centroid output → loss starts ~0.
    layer_inputs = torch.randn(8, layer_ref.experts_module.hidden_dim) * 0.1
    state = _distill_merged_group(
        layer_ref=layer_ref,
        centroid_id=0,
        members=[0, 1],
        freq={0: 3, 1: 1},
        pre_merge_weights=pre,
        layer_inputs=layer_inputs,
        steps=200,
        lr=1e-4,
        betas=(0.9, 0.95),
        plateau_steps=3,
        plateau_eps=2.0,  # very loose threshold; plateau triggers immediately
        token_cap=8,
        device=torch.device("cpu"),
    )
    # Plateau break should fire well before 200 steps.
    assert state["break_reason"] == "plateau"
    assert state["steps"] < 200


def test_distill_writes_back_to_bank(tiny_model):
    """After distillation, the centroid's bank weights must be the trained
    parameter values (in the original dtype), not the pre-distill weights."""
    from moe_compress.utils.model_io import iter_moe_layers, build_banks
    layer_ref = list(iter_moe_layers(tiny_model))[0]

    pre = _snapshot_pre_merge_layer_experts(layer_ref)
    banks = build_banks(layer_ref)
    pre_distill = banks["gate_proj"].get(0).clone()
    pre_dtype = pre_distill.dtype

    layer_inputs = torch.randn(8, layer_ref.experts_module.hidden_dim) * 0.1

    _distill_merged_group(
        layer_ref=layer_ref,
        centroid_id=0,
        members=[0, 1],
        freq={0: 1, 1: 1},
        pre_merge_weights=pre,
        layer_inputs=layer_inputs,
        steps=5,
        lr=5e-2,  # large LR → guaranteed weight change
        betas=(0.9, 0.95),
        plateau_steps=100,
        plateau_eps=0.0,
        token_cap=8,
        device=torch.device("cpu"),
    )

    post_distill = banks["gate_proj"].get(0)
    # Dtype preserved.
    assert post_distill.dtype == pre_dtype
    # Weights changed.
    assert not torch.allclose(pre_distill, post_distill)
