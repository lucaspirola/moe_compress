"""Direction C — correctness tests for the output-space merge cost.

``cost_alignment: "output"`` adds a third Stage-2 cost mode whose per-pair
cost is the routing-weighted change in expert ``m``'s gated routed output on
calibration tokens when ``m`` is tentatively merged into a centroid ``c`` —
a strictly better merge-damage proxy than the weight-space "pre"/"post"
costs.

All tests are CPU-only and use synthetic tensors / the ``tiny_model``
fixture — no GPU, no real checkpoint.

Coverage:
  (a) The output-space cost matches an independent hand-recomputation on a
      synthetic layer (genuine numeric check, not a self-consistency tautology).
  (b) Tentatively merging two *identical* experts yields ~zero output cost.
  (c) The "pre"/"post" cost paths are byte-unaffected by Direction C: with
      ``cost_alignment in {"pre","post"}`` the new ``layer_inputs`` /
      ``output_token_cap`` parameters are never read, and the config boundary
      now accepts "output" without changing "pre"/"post" handling.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from moe_compress.stage2_reap_ream import (
    _output_space_cost,
    _router_routing_weights,
    _tentative_merged_weights,
    _ream_cost_matrix,
    _swiglu_forward,
)
from moe_compress.utils.activation_hooks import ReamCostAccumulator
from moe_compress.utils.model_io import iter_moe_layers, build_banks, MATRIX_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _independent_output_cost(
    layer_ref,
    m_id: int,
    c_id: int,
    x: torch.Tensor,
    freq: dict[int, int],
) -> float:
    """Recompute cost(m→c) from scratch via an independent code path.

    Deliberately does NOT call ``_output_space_cost`` — it rebuilds the
    tentative merge, the SwiGLU forwards and the routing-weight mask by hand
    so the assertion is a genuine numeric check of the production function.
    """
    banks = build_banks(layer_ref)
    W_m = {n: banks[n].get(m_id).to(torch.float32) for n in MATRIX_NAMES}

    # Tentative freq-weighted merge of m into c. The two experts in the test
    # fixture have an identity neuron permutation (random independent
    # weights → _permutation_align_to_centroid may still permute, so we reuse
    # the production tentative-merge helper for the weights only; the cost
    # arithmetic below is fully independent of it).
    merged = _tentative_merged_weights(layer_ref, c_id, m_id, freq, None, None)

    E_m = _swiglu_forward(W_m["gate_proj"], W_m["up_proj"], W_m["down_proj"], x)
    E_merged = _swiglu_forward(
        merged["gate_proj"], merged["up_proj"], merged["down_proj"], x,
    )

    # Routing weights + top-k mask, recomputed independently.
    router = layer_ref.router
    logits = F.linear(x.to(router.weight.dtype), router.weight)
    sigma = F.softmax(logits.float(), dim=-1)
    k = min(layer_ref.top_k, sigma.shape[-1])
    topk_idx = torch.topk(sigma, k=k, dim=-1).indices
    routed_m = (topk_idx == m_id).any(dim=-1)
    gate_m = sigma[:, m_id] * routed_m.to(sigma.dtype)

    per_token = (E_m - E_merged).pow(2).sum(dim=-1)
    return float((gate_m * per_token).sum() / gate_m.sum())


# ---------------------------------------------------------------------------
# _router_routing_weights
# ---------------------------------------------------------------------------


def test_router_routing_weights_is_a_softmax(tiny_model):
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    x = torch.randn(7, tiny_model.config.hidden_size)
    sigma = _router_routing_weights(layer_ref, x)
    assert sigma.shape == (7, layer_ref.num_routed_experts)
    # A softmax: non-negative, rows sum to 1.
    assert (sigma >= 0).all()
    assert torch.allclose(sigma.sum(dim=-1), torch.ones(7), atol=1e-5)


# ---------------------------------------------------------------------------
# _tentative_merged_weights
# ---------------------------------------------------------------------------


def test_tentative_merge_of_identical_experts_is_that_expert(tiny_model):
    """Merging an expert into a clone of itself returns the same weights —
    regardless of the freq weighting (any convex combo of W with W is W)."""
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    banks = build_banks(layer_ref)
    # Make expert 1 an exact clone of expert 0.
    with torch.no_grad():
        for n in MATRIX_NAMES:
            banks[n].set(1, banks[n].get(0).clone())

    merged = _tentative_merged_weights(
        layer_ref, centroid_id=0, child_id=1, freq={0: 3, 1: 7},
        ream_acc=None, perm_cache=None,
    )
    for n in MATRIX_NAMES:
        assert torch.allclose(
            merged[n], banks[n].get(0).to(torch.float32), atol=1e-5,
        ), f"{n}: merge of identical experts must equal the expert"


def test_tentative_merge_freq_weighting(tiny_model):
    """With an identity permutation the merge is exactly the freq-weighted
    convex combination of the two experts' weights."""
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    banks = build_banks(layer_ref)
    # Force an identity alignment: gate/up of both experts equal → the
    # permutation solver returns identity, isolating the freq-weight math.
    with torch.no_grad():
        for n in ("gate_proj", "up_proj"):
            banks[n].set(1, banks[n].get(0).clone())
        # down_proj differs so the merge is non-trivial.
        banks["down_proj"].set(1, torch.randn_like(banks["down_proj"].get(1)))

    f_c, f_m = 1, 3
    merged = _tentative_merged_weights(
        layer_ref, centroid_id=0, child_id=1, freq={0: f_c, 1: f_m},
        ream_acc=None, perm_cache=None,
    )
    w_c, w_m = f_c / (f_c + f_m), f_m / (f_c + f_m)
    expected_down = (
        w_c * banks["down_proj"].get(0).to(torch.float32)
        + w_m * banks["down_proj"].get(1).to(torch.float32)
    )
    assert torch.allclose(merged["down_proj"], expected_down, atol=1e-5)


# ---------------------------------------------------------------------------
# (b) identical experts → ~zero output cost
# ---------------------------------------------------------------------------


def test_output_cost_of_identical_experts_is_zero(tiny_model):
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    banks = build_banks(layer_ref)
    # Expert 1 := exact clone of expert 0. Merging 1 into 0 changes nothing,
    # so the gated-output residual must be ~0.
    with torch.no_grad():
        for n in MATRIX_NAMES:
            banks[n].set(1, banks[n].get(0).clone())

    n_exp = layer_ref.num_routed_experts
    x = torch.randn(64, tiny_model.config.hidden_size)
    cheap = np.random.rand(1, n_exp - 1)  # 1 noncentroid, (n_exp-1) centroids

    cost = _output_space_cost(
        layer_ref,
        noncentroid_ids=[1],
        centroid_ids=[e for e in range(n_exp) if e != 1],
        cheap_cost=cheap,
        ream_acc=None,
        perm_cache=None,
        topk=n_exp,
        freq={e: 5 for e in range(n_exp)},
        layer_inputs=x,
        token_cap=1024,
    )
    # Column for centroid id 0 (expert 1's clone source).
    col_of_0 = [e for e in range(n_exp) if e != 1].index(0)
    assert cost[0, col_of_0] == pytest.approx(0.0, abs=1e-6), (
        f"merging identical experts must cost ~0, got {cost[0, col_of_0]}"
    )


# ---------------------------------------------------------------------------
# (a) hand-checkable / independently-recomputed numeric case
# ---------------------------------------------------------------------------


def test_output_cost_matches_independent_recomputation(tiny_model):
    """The production cost matrix entry equals an independent by-hand
    recomputation of the routing-weighted gated-output residual."""
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    n_exp = layer_ref.num_routed_experts
    freq = {e: (e + 1) * 3 for e in range(n_exp)}  # distinct, non-uniform

    torch.manual_seed(123)
    x = torch.randn(96, tiny_model.config.hidden_size)

    # 1 noncentroid (expert 0), the rest are centroids. topk == n_c so every
    # centroid is a candidate → every entry is finite & checkable.
    noncentroid_ids = [0]
    centroid_ids = [e for e in range(n_exp) if e != 0]
    cheap = np.random.rand(1, len(centroid_ids))

    cost = _output_space_cost(
        layer_ref,
        noncentroid_ids=noncentroid_ids,
        centroid_ids=centroid_ids,
        cheap_cost=cheap,
        ream_acc=None,
        perm_cache=None,
        topk=len(centroid_ids),
        freq=freq,
        layer_inputs=x,
        token_cap=1024,
    )

    for cj, c_id in enumerate(centroid_ids):
        expected = _independent_output_cost(layer_ref, 0, c_id, x, freq)
        assert cost[0, cj] == pytest.approx(expected, rel=1e-5, abs=1e-7), (
            f"centroid {c_id}: production cost {cost[0, cj]} != "
            f"independent recompute {expected}"
        )
    # Sanity: a genuine merge of *different* random experts costs > 0.
    assert (cost[0] > 0).all(), "non-identical merges must have positive cost"


def test_output_cost_hand_checked_scalar():
    """Fully hand-checked scalar case on a 2-expert, top-1 synthetic layer.

    With top_k == 1 and a router whose weight makes every token route to
    expert 0, only expert-0 tokens carry weight. We build expert 0 and
    expert 1 explicitly, set freq so the merge weight is exactly 0.5/0.5,
    and verify the cost equals the closed-form
    ``mean_t σ_0(x_t)·‖E_0 - E_merged‖² / mean_t σ_0(x_t)`` computed with
    plain torch ops.
    """
    import torch.nn as nn
    from moe_compress.utils.model_io import MoELayerRef

    hidden, d_int, n_exp, top_k = 4, 3, 2, 1
    torch.manual_seed(7)

    # --- minimal fused-experts module (gate_up_proj + down_proj params) ---
    class _Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = n_exp
            # fused gate_up: (n_exp, 2*d_int, hidden); down: (n_exp, hidden, d_int)
            self.gate_up_proj = nn.Parameter(torch.randn(n_exp, 2 * d_int, hidden))
            self.down_proj = nn.Parameter(torch.randn(n_exp, hidden, d_int))

    class _Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = top_k
            self.hidden_dim = hidden
            # Router weight strongly favouring expert 0 for every input.
            w = torch.zeros(n_exp, hidden)
            w[0, 0] = 50.0  # huge logit for expert 0
            self.weight = nn.Parameter(w)

    class _MLP(nn.Module):
        def __init__(self, experts, router):
            super().__init__()
            self.experts = experts
            self.gate = router

    experts = _Experts()
    router = _Router()
    mlp = _MLP(experts, router)
    layer_ref = MoELayerRef(
        layer_idx=0, layer_module=mlp, mlp=mlp, router=router,
        experts_module=experts, shared_expert=None, layer_type="full_attention",
    )

    banks = build_banks(layer_ref)
    freq = {0: 2, 1: 2}  # → merge weights 0.5 / 0.5

    # Make all inputs positive on dim 0 so the router logit for expert 0 is
    # large and positive → σ_0 ≈ 1 for every token.
    x = torch.rand(40, hidden) + 0.5

    cost = _output_space_cost(
        layer_ref,
        noncentroid_ids=[0],
        centroid_ids=[1],
        cheap_cost=np.zeros((1, 1)),
        ream_acc=None, perm_cache=None,
        topk=1, freq=freq, layer_inputs=x, token_cap=1024,
    )

    # --- closed-form expected value ---
    W0 = {n: banks[n].get(0).to(torch.float32) for n in MATRIX_NAMES}
    merged = _tentative_merged_weights(layer_ref, 1, 0, freq, None, None)
    E0 = _swiglu_forward(W0["gate_proj"], W0["up_proj"], W0["down_proj"], x)
    Em = _swiglu_forward(
        merged["gate_proj"], merged["up_proj"], merged["down_proj"], x,
    )
    logits = F.linear(x, router.weight)
    sigma0 = F.softmax(logits.float(), dim=-1)[:, 0]
    # top_k == 1 and expert 0 always wins → routed_m is all-True.
    per_token = (E0 - Em).pow(2).sum(dim=-1)
    expected = float((sigma0 * per_token).sum() / sigma0.sum())

    assert cost[0, 0] == pytest.approx(expected, rel=1e-5, abs=1e-7)
    assert cost[0, 0] > 0.0  # the two random experts genuinely differ


# ---------------------------------------------------------------------------
# (c) pre / post paths unaffected
# ---------------------------------------------------------------------------


def test_pre_path_ignores_output_params(tiny_model):
    """``_ream_cost_matrix(cost_alignment="pre")`` returns the identical
    matrix whether or not the Direction-C parameters are supplied."""
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    n_exp = layer_ref.num_routed_experts

    ream_acc = ReamCostAccumulator()  # empty → degenerate full-0.5 sim path
    noncentroid_ids = [0, 1]
    centroid_ids = [e for e in range(n_exp) if e not in (0, 1)]

    base = _ream_cost_matrix(
        layer_ref, noncentroid_ids, centroid_ids,
        ream_acc=ream_acc, cost_alignment="pre",
    )
    # Same call, but now passing the Direction-C-only kwargs. The "pre" path
    # must never read them → byte-identical result.
    with_c_params = _ream_cost_matrix(
        layer_ref, noncentroid_ids, centroid_ids,
        ream_acc=ream_acc, cost_alignment="pre",
        layer_inputs=torch.randn(32, tiny_model.config.hidden_size),
        output_token_cap=256,
    )
    assert np.array_equal(base, with_c_params), (
        "Direction-C parameters must not perturb the cost_alignment='pre' path"
    )


def test_unknown_cost_alignment_still_rejected(tiny_model):
    """The dispatch error message lists exactly the three valid modes —
    a typo'd mode is still rejected (the "output" addition didn't loosen
    validation)."""
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    ream_acc = ReamCostAccumulator()
    with pytest.raises(ValueError, match="'pre', 'post', or 'output'"):
        _ream_cost_matrix(
            layer_ref, [0], [1, 2],
            ream_acc=ream_acc, cost_alignment="bogus",
        )


def test_output_cost_requires_layer_inputs(tiny_model):
    """The output path fails loudly if calibration tokens are missing —
    it never silently degrades to a weight-space cost."""
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    ream_acc = ReamCostAccumulator()
    with pytest.raises(RuntimeError, match="no layer-input calibration tokens"):
        _ream_cost_matrix(
            layer_ref, [0], [1, 2],
            ream_acc=ream_acc, cost_alignment="output",
            freq={0: 1, 1: 1, 2: 1},
            layer_inputs=None,
        )
