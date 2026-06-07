# SPDX-License-Identifier: Apache-2.0
"""Pure-tensor unit tests for the in-graph calibration accumulation math.

These validate the GPU-op accumulation kernels used by the cudagraph-safe
calibration capture rearchitecture (tasks/PLAN_CALIB_INGRAPH_CAPTURE.md)
INDEPENDENT of vLLM, CUDA, or torch.compile. We reimplement the exact tensor
math each writer / custom-op body performs on small CPU tensors and assert it
matches a straightforward reference. No vllm import, no GPU, no monkeypatch of
production code.

The math under test (one helper per writer signal):
  * reap_scores       : gate-weighted ||f||_2 scatter_add + token counts
  * per_expert_max    : per-token |f|_inf scatter_reduce(amax)
  * routing_stats     : per-expert freq scatter_add + gate-weight sum
  * router_logits     : per-expert Σ logit / Σ logit² / count
  * output_reservoir  : deterministic fixed-stride local-rank via one-hot+cumsum
  * block_out pointer : monotonic-pointer write w/ OOB sentinel (no clobber)
  * input_cov clamp   : H-NEW-1 overflow mask + per-expert baddbmm Gram
  * imatrix dense     : Σ x² over the token dim
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------
# Reference reimplementations of the exact production math.
# --------------------------------------------------------------------------
def _reap_accum(unweighted, topk_ids, topk_weights, n_experts):
    """[T, tk, K], [T, tk], [T, tk] -> ([E] score, [E] count)."""
    scores = torch.zeros(n_experts)
    counts = torch.zeros(n_experts, dtype=torch.int64)
    norms = unweighted.float().norm(dim=-1)        # [T, tk]
    weighted = norms * topk_weights                # [T, tk]
    flat_ids = topk_ids.reshape(-1)
    scores.scatter_add_(0, flat_ids, weighted.reshape(-1))
    ones = torch.ones(flat_ids.shape[0], dtype=torch.int64)
    counts.scatter_add_(0, flat_ids, ones)
    return scores, counts


def _per_expert_max_accum(unweighted, topk_ids, n_experts):
    acc = torch.full((n_experts,), float("-inf"))
    mags = unweighted.float().abs().amax(dim=-1)   # [T, tk]
    flat_ids = topk_ids.reshape(-1)
    acc.scatter_reduce_(0, flat_ids, mags.reshape(-1),
                        reduce="amax", include_self=True)
    return acc


def _routing_accum(topk_ids, topk_weights, n_experts):
    freq = torch.zeros(n_experts, dtype=torch.int64)
    wsum = torch.zeros(n_experts)
    flat_ids = topk_ids.reshape(-1)
    freq.scatter_add_(0, flat_ids,
                      torch.ones(flat_ids.shape[0], dtype=torch.int64))
    wsum.scatter_add_(0, flat_ids, topk_weights.reshape(-1).float())
    return freq, wsum


def _router_logits_accum(router_logits):
    rl = router_logits.float()                     # [T, E]
    lsum = rl.sum(dim=0)
    lsq = (rl * rl).sum(dim=0)
    lcnt = torch.tensor(rl.shape[0], dtype=torch.int64)
    return lsum, lsq, lcnt


def _reservoir_local_rank(ids_k, n_experts):
    """One-hot + cumsum within-dispatch local rank per token (0-based)."""
    T = ids_k.shape[0]
    indicator = torch.zeros(T, n_experts, dtype=torch.int64)
    indicator.scatter_(1, ids_k.unsqueeze(1), 1)
    local_rank = (indicator.cumsum(dim=0)
                  .gather(1, ids_k.unsqueeze(1))
                  .squeeze(1) - 1)
    return local_rank, indicator.sum(dim=0)


def _block_out_write(accum, ptr, arange, hidden):
    """Monotonic-pointer write with OOB sentinel. accum has cap+1 rows."""
    T = hidden.shape[0]
    cap = accum.shape[0] - 1
    offs = ptr + arange[:T]
    valid = offs < cap
    write_idx = torch.where(valid, offs, torch.full_like(offs, cap))
    src = hidden * valid.unsqueeze(1).to(hidden.dtype)
    accum.scatter_(0, write_idx.unsqueeze(1).expand(-1, accum.shape[1]),
                   src.to(accum.dtype))
    ptr.copy_((ptr + T).clamp(max=cap))


def _input_cov_clamp(x, ids_k, n_experts, buf_rows):
    """H-NEW-1 overflow clamp+mask with OOB sentinel column.

    Returns (tmp[E, buf_rows, H] -- sentinel column stripped, valid[T]).
    OOB tokens are routed to a sacrificial sentinel column so a clamped OOB
    write can never clobber a valid slot at buf_rows-1.
    """
    T = x.shape[0]
    H = x.shape[1]
    bufp1 = buf_rows + 1
    tmp = torch.zeros(n_experts, bufp1, H)
    indicator = torch.zeros(T, n_experts, dtype=torch.int64)
    indicator.scatter_(1, ids_k.unsqueeze(1), 1)
    local_rank = (indicator.cumsum(dim=0)
                  .gather(1, ids_k.unsqueeze(1))
                  .squeeze(1) - 1)
    valid = local_rank < buf_rows
    col = torch.where(valid, local_rank, torch.full_like(local_rank, buf_rows))
    lin = (ids_k * bufp1 + col)
    xm = x * valid.unsqueeze(1).to(x.dtype)
    tmp.view(-1, H).scatter_(0, lin.unsqueeze(1).expand(-1, H), xm)
    return tmp[:, :buf_rows, :], valid.to(torch.int64)


def _imatrix_dense(x):
    return x.float().pow(2).sum(dim=0)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_reap_scatter_add_matches_reference():
    torch.manual_seed(0)
    T, tk, K, E = 5, 2, 4, 6
    uw = torch.randn(T, tk, K)
    ids = torch.randint(0, E, (T, tk))
    tw = torch.rand(T, tk)
    scores, counts = _reap_accum(uw, ids, tw, E)

    # Reference: explicit python loop.
    ref_s = torch.zeros(E)
    ref_c = torch.zeros(E, dtype=torch.int64)
    for t in range(T):
        for k in range(tk):
            e = int(ids[t, k])
            ref_s[e] += float(uw[t, k].norm()) * float(tw[t, k])
            ref_c[e] += 1
    assert torch.allclose(scores, ref_s, atol=1e-5)
    assert torch.equal(counts, ref_c)
    assert int(counts.sum()) == T * tk


def test_per_expert_max_amax_matches_reference():
    torch.manual_seed(1)
    T, tk, K, E = 4, 2, 3, 5
    uw = torch.randn(T, tk, K)
    ids = torch.randint(0, E, (T, tk))
    acc = _per_expert_max_accum(uw, ids, E)

    ref = torch.full((E,), float("-inf"))
    for t in range(T):
        for k in range(tk):
            e = int(ids[t, k])
            ref[e] = max(ref[e], float(uw[t, k].abs().max()))
    # experts that saw no token stay -inf in both.
    finite = torch.isfinite(ref)
    assert torch.allclose(acc[finite], ref[finite], atol=1e-5)
    assert torch.equal(torch.isfinite(acc), finite)


def test_routing_stats_freq_and_wsum():
    torch.manual_seed(2)
    T, tk, E = 7, 3, 4
    ids = torch.randint(0, E, (T, tk))
    tw = torch.rand(T, tk)
    freq, wsum = _routing_accum(ids, tw, E)
    assert int(freq.sum()) == T * tk
    ref_w = torch.zeros(E)
    for t in range(T):
        for k in range(tk):
            ref_w[int(ids[t, k])] += float(tw[t, k])
    assert torch.allclose(wsum, ref_w, atol=1e-5)


def test_router_logits_sum_sq_count():
    torch.manual_seed(3)
    T, E = 6, 5
    rl = torch.randn(T, E)
    lsum, lsq, lcnt = _router_logits_accum(rl)
    assert torch.allclose(lsum, rl.sum(dim=0), atol=1e-5)
    assert torch.allclose(lsq, (rl ** 2).sum(dim=0), atol=1e-5)
    assert int(lcnt) == T


def test_reservoir_local_rank_is_per_expert_dense_arange():
    # Tokens routed to the same expert get distinct 0,1,2,... local ranks.
    ids = torch.tensor([0, 0, 1, 0, 1])
    E = 2
    lr, delta = _reservoir_local_rank(ids, E)
    # expert 0 appears at positions 0,1,3 -> ranks 0,1,2
    # expert 1 appears at positions 2,4   -> ranks 0,1
    assert lr.tolist() == [0, 1, 0, 2, 1]
    assert delta.tolist() == [3, 2]
    # No two tokens for the same expert share a rank.
    for e in range(E):
        ranks = lr[ids == e]
        assert len(set(ranks.tolist())) == len(ranks)


def test_reservoir_fixed_stride_slot_wraps_without_overlap():
    cap = 3
    ids = torch.tensor([0, 0, 0, 0])  # 4 tokens to expert 0, cap=3
    E = 1
    lr, delta = _reservoir_local_rank(ids, E)
    base_ctr = torch.tensor([0])
    slots = (base_ctr.gather(0, ids) + lr) % cap
    # slots 0,1,2,0 -> the 4th token wraps to slot 0 (ring buffer).
    assert slots.tolist() == [0, 1, 2, 0]


def test_block_out_monotonic_pointer_advances():
    cap = 4
    buf = torch.zeros(cap + 1, 2)
    ptr = torch.zeros((), dtype=torch.int64)
    arange = torch.arange(cap + 1)
    _block_out_write(buf, ptr, arange, torch.tensor([[1., 1.], [2., 2.]]))
    assert int(ptr) == 2
    _block_out_write(buf, ptr, arange, torch.tensor([[3., 3.]]))
    assert int(ptr) == 3
    assert buf[:3].tolist() == [[1., 1.], [2., 2.], [3., 3.]]


def test_block_out_oob_sentinel_does_not_clobber_valid_rows():
    cap = 3
    buf = torch.zeros(cap + 1, 2)
    ptr = torch.zeros((), dtype=torch.int64)
    arange = torch.arange(cap + 1)
    _block_out_write(buf, ptr, arange, torch.tensor([[5., 6.], [7., 8.]]))
    # 2 more rows: only 1 slot left (idx 2); the 2nd is OOB and must NOT
    # overwrite the valid row written to idx 2.
    _block_out_write(buf, ptr, arange, torch.tensor([[9., 9.], [1., 1.]]))
    assert int(ptr) == 3
    assert buf[:cap].tolist() == [[5., 6.], [7., 8.], [9., 9.]]
    # sentinel row absorbed the OOB write (zeroed by the valid-mask, i.e.
    # the OOB token's data is dropped, not retained).
    assert buf[cap].tolist() == [0., 0.]


def test_input_cov_overflow_tokens_dropped_and_gram_correct():
    # 3 tokens to expert 0, buf_rows=2 -> the 3rd overflows and is dropped.
    x = torch.tensor([[1., 0.], [0., 1.], [5., 5.]])
    ids = torch.tensor([0, 0, 0])
    E, buf = 1, 2
    tmp, valid = _input_cov_clamp(x, ids, E, buf)
    assert valid.tolist() == [1, 1, 0]            # 3rd token masked
    # Gram = tmp.T @ tmp per expert == sum of outer products of kept rows.
    cov = torch.baddbmm(
        torch.zeros(E, 2, 2), tmp.transpose(1, 2), tmp)
    ref = torch.zeros(2, 2)
    for t in range(2):  # only the first two tokens count
        ref += torch.outer(x[t], x[t])
    assert torch.allclose(cov[0], ref, atol=1e-5)


def test_input_cov_count_excludes_overflow():
    x = torch.randn(5, 3)
    ids = torch.tensor([0, 0, 0, 0, 1])  # 4 to expert 0, buf=2 -> 2 dropped
    E, buf = 2, 2
    _, valid = _input_cov_clamp(x, ids, E, buf)
    # expert-0 valid count = 2 (first two), expert-1 valid count = 1.
    per_expert = torch.zeros(E, dtype=torch.int64)
    per_expert.scatter_add_(0, ids, valid)
    assert per_expert.tolist() == [2, 1]


def test_imatrix_dense_sum_of_squares():
    x = torch.tensor([[1., 2., 3.], [1., 1., 1.]])
    acc = _imatrix_dense(x)
    assert acc.tolist() == [2., 5., 10.]
    # accumulation is additive across calls.
    acc2 = acc + _imatrix_dense(torch.tensor([[2., 0., 0.]]))
    assert acc2.tolist() == [6., 5., 10.]
