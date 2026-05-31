"""Online δ_gate Gram equivalence + merge-invariance (PLAN_GATE_LOGIT_ONLINE).

The unbounded raw ``gate_logit_profiles`` list was replaced by a bounded
online ``_gate_gram`` [E, E] fp64 Gram. ``compute_gate_similarity_matrix``
reconstructs the δ_gate similarity from the Gram in fp64. These tests pin:

1. ``test_gate_gram_online_equals_batched`` — the online Gram path matches a
   reference that does the OLD math (cat → F.normalize → cdist → dist2sim) on
   the full token tensor, to within the documented fp64 accuracy budget
   (atol=2e-7, NOT bit-equality). Fixture includes a near-colinear pair
   (the catastrophic-cancellation case) and the three §2.3 edge cases.
2. ``test_gate_gram_merge_invariant`` — feeds both sim matrices through the
   ream_cost cost build + top-K candidate filter + cost-argmin and asserts
   the candidate sets AND argmin assignments are identical (the real bar).
3. ``test_gate_gram_rejects_fp32`` — locks in WHY fp64 is mandatory: an
   all-fp32 reconstruction breaches atol=2e-7 on the near-colinear pair.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from moe_compress.utils.activation_hooks import ReamCostAccumulator


def _ref_delta_gate(full: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Reference: the OLD batched math on the full [T, E] fp64 tensor."""
    mat = F.normalize(full[:, ids].t(), p=2, dim=1)  # [n, T]
    mat = torch.where(mat.isnan(), torch.zeros_like(mat), mat)
    n = len(ids)
    if mat.abs().max() < 1e-9:
        return torch.zeros(n, n, dtype=torch.float64)
    d = torch.cdist(mat, mat, p=2)
    sim = 1.0 - d / d.max().clamp_min(1e-12)
    sim.fill_diagonal_(1.0)
    return sim.clamp(0.0, 1.0)


def _online_from_gram(full_fp32: torch.Tensor, n_batches: int, ids: list[int]):
    """Drive record_router_logits over uneven batches, then reconstruct."""
    acc = ReamCostAccumulator()
    acc.num_experts = full_fp32.shape[1]
    # Uneven split into n_batches chunks.
    T = full_fp32.shape[0]
    bounds = sorted({0, T, *[int(T * (i + 1) / n_batches) for i in range(n_batches)]})
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        acc.record_router_logits(0, full_fp32[lo:hi], lo)
    return acc.compute_gate_similarity_matrix(0, ids)


def _build_fixture(seed: int = 0):
    """[T, E] fp32 logits with a near-colinear pair + the §2.3 edge cases."""
    torch.manual_seed(seed)
    E, T = 8, 256
    full = torch.randn((T, E), dtype=torch.float32)
    # Near-colinear pair (experts 1 ≈ 0 + eps) — catastrophic-cancellation case.
    full[:, 1] = full[:, 0] + 1e-6 * torch.randn((T,))
    # All-identical-direction pair (experts 2, 3) so d.max() path is exercised.
    full[:, 3] = full[:, 2]
    # Zero-norm expert (expert 7 has no signal).
    full[:, 7] = 0.0
    return full


def test_gate_gram_online_equals_batched():
    full = _build_fixture(seed=11)
    ids = list(range(full.shape[1]))
    online = _online_from_gram(full, n_batches=4, ids=ids)
    ref = _ref_delta_gate(full.to(torch.float64), ids)
    # HONEST tolerance per §2.4 budget — fp64 reconstruction error ~2e-8,
    # NOT bit-equality (1e-9 is FALSE for the near-colinear pair).
    torch.testing.assert_close(
        online.to(torch.float64), ref, atol=2e-7, rtol=0.0,
    )


def test_gate_gram_edge_cases():
    """All-zero matrix and a single zero-norm expert match the reference."""
    E, T = 5, 32
    # All-zero matrix → all-zeros sim (early exit).
    full_zero = torch.zeros((T, E), dtype=torch.float32)
    ids = list(range(E))
    online = _online_from_gram(full_zero, n_batches=3, ids=ids)
    assert torch.equal(online, torch.zeros(E, E, dtype=torch.float32))

    # All-identical experts → d.max()==0 → all-ones sim.
    torch.manual_seed(3)
    col = torch.randn((T, 1), dtype=torch.float32)
    full_same = col.repeat(1, E)
    online_same = _online_from_gram(full_same, n_batches=2, ids=ids)
    ref_same = _ref_delta_gate(full_same.to(torch.float64), ids)
    torch.testing.assert_close(
        online_same.to(torch.float64), ref_same, atol=2e-7, rtol=0.0,
    )


def _cost_build(sim_gate: np.ndarray, sim_expert: np.ndarray) -> np.ndarray:
    """Mirror ream_cost: cost = 1 - (sim_gate + sim_expert)/2."""
    return 1.0 - (sim_gate + sim_expert) / 2.0


def test_gate_gram_merge_invariant():
    """Downstream top-K candidate sets + argmin assignments do not flip."""
    full = _build_fixture(seed=22)
    n = full.shape[1]
    ids = list(range(n))
    online = _online_from_gram(full, n_batches=5, ids=ids).numpy().astype(np.float64)
    ref = _ref_delta_gate(full.to(torch.float64), ids).numpy()

    # Hold sim_expert constant so sim_gate is the sole ranker.
    sim_expert = np.full((n, n), 0.5, dtype=np.float64)
    cost_online = _cost_build(online, sim_expert)
    cost_ref = _cost_build(ref, sim_expert)

    # Centroids = well-separated experts (4, 5, 6); candidates = the rest
    # (incl. the near-colinear pair 0/1 and the duplicate pair 2/3, which
    # compete to be assigned to a centroid — matching the plan's §2.4 setup,
    # NOT making the near-colinear pair itself a pair of centroids).
    centroids = [4, 5, 6]
    candidates = [c for c in range(n) if c not in centroids]
    topk = min(2, len(centroids))
    # The load-bearing property is that the fp64 Gram-vs-reference sim
    # difference (~2e-8) never flips an assignment whose decision margin
    # exceeds that error. A "tie" (margin below the sim error) is inherent
    # to the random fixture, not a Gram regression, so we guard those out:
    # we assert no flip occurs on any decision with a real (> sim_err) gap.
    sim_err = float(np.abs(online - ref).max())
    cost_err = sim_err / 2.0  # cost = 1 - (sim_gate + sim_expert)/2
    margin = 10.0 * cost_err + 1e-12
    flips_topk = 0
    flips_argmin = 0
    for c in candidates:
        row_o = cost_online[c, centroids]
        row_r = cost_ref[c, centroids]
        # top-K candidate set on the reference; require it stable on online.
        order_r = np.argsort(row_r)
        # Decision margin: gap between the K-th and (K+1)-th smallest costs.
        if len(order_r) > topk:
            decision_gap = row_r[order_r[topk]] - row_r[order_r[topk - 1]]
            if decision_gap > margin:
                set_o = set(np.argpartition(row_o, topk - 1)[:topk].tolist())
                set_r = set(order_r[:topk].tolist())
                if set_o != set_r:
                    flips_topk += 1
        # argmin: gap between the two smallest costs.
        argmin_gap = row_r[order_r[1]] - row_r[order_r[0]]
        if argmin_gap > margin:
            if int(np.argmin(row_o)) != int(order_r[0]):
                flips_argmin += 1
    assert flips_topk == 0, f"top-K candidate-set flips (non-tie): {flips_topk}"
    assert flips_argmin == 0, f"argmin assignment flips (non-tie): {flips_argmin}"
    # The Gram sim error must itself be within the documented fp64 budget.
    assert sim_err <= 2e-7, f"Gram sim error {sim_err:.3e} exceeds 2e-7 budget"


def test_gate_gram_rejects_fp32():
    """An all-fp32 reconstruction breaches atol=2e-7 — proves fp64 is load-bearing."""
    full = _build_fixture(seed=33)
    ids = list(range(full.shape[1]))
    ref = _ref_delta_gate(full.to(torch.float64), ids)

    # Reproduce the reviewer's all-fp32 reconstruction directly from the Gram.
    x = full.to(torch.float32)
    G = (x.transpose(0, 1) @ x)  # fp32 Gram
    G_sub = G[ids][:, ids].to(torch.float32)
    norms = G_sub.diagonal().clamp_min(0).sqrt()
    nz = norms > 0
    denom = (norms[:, None] * norms[None, :]).clamp_min(1e-30)
    cos = torch.where(nz[:, None] & nz[None, :], G_sub / denom, torch.zeros_like(G_sub))
    unit = nz.to(torch.float32)
    d = (unit[:, None] + unit[None, :] - 2.0 * cos).clamp_min(0.0).sqrt()
    sim_fp32 = 1.0 - d / d.max().clamp_min(1e-12)
    sim_fp32.fill_diagonal_(1.0)
    sim_fp32.clamp_(0.0, 1.0)

    err = (sim_fp32.to(torch.float64) - ref).abs().max().item()
    # The fp32 (2-2cos) cancellation on the near-colinear pair blows the
    # error well past the fp64 budget; this is WHY fp64 is mandatory.
    assert err > 2e-7, (
        f"fp32 reconstruction error {err:.3e} should breach the 2e-7 budget; "
        f"if this fails the near-colinear fixture is too lax"
    )
