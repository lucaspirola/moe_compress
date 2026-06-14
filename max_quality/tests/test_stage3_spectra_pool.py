"""Stage-3 F2 — parallel fp64-CPU rank-spectra pool.

Validates the spawn-ProcessPool group-stat parallelization is FIDELITY-SAFE:
parallel (workers>1) == 1-thread-pinned-serial on effective_rank (exact float)
and singular_values_mean (torch.equal); 1-thread pinning is the bit invariant;
the worker wrapper is top-level/serializable.
"""
from __future__ import annotations

import copy
import torch


def _make_payload(seed=7):
    from moe_compress.stage3.spectra_pool import _GroupStatPayload
    torch.manual_seed(seed)
    d_out, d_in, n = 16, 12, 4
    weights = [torch.randn(d_out, d_in, dtype=torch.float32) for _ in range(n)]
    m = torch.randn(d_in, d_in, dtype=torch.float32)
    a_g = (m @ m.T + d_in * torch.eye(d_in)).to(torch.float32)
    return _GroupStatPayload(layer_idx=0, name="gate_proj", n_experts=n,
                             weights_cpu=weights, a_g_cpu=a_g)


def test_payload_round_trips_through_pool_serialization():
    """The payload survives the spawn pool's submit path (deepcopy stands in for
    the cross-process round-trip; a real workers=2 run in test 4 exercises the
    actual spawn pickling end-to-end)."""
    payload = _make_payload()
    clone = copy.deepcopy(payload)
    assert clone.name == "gate_proj"
    assert torch.equal(clone.weights_cpu[0], payload.weights_cpu[0])


def test_group_stat_payload_matches_direct_group_stat():
    """The wrapper's output equals calling _group_stat directly (same numbers)."""
    from moe_compress.stage3.spectra_pool import _group_stat_payload, _ListBank
    from moe_compress.stage3.plugins.d_rank_allocate import _group_stat
    payload = _make_payload()
    torch.set_num_threads(1)
    key, gs = _group_stat_payload(payload)
    ref = _group_stat(payload.n_experts, _ListBank(payload.weights_cpu), A_g=payload.a_g_cpu)
    assert key == (0, "gate_proj")
    assert gs.effective_rank == ref.effective_rank
    assert torch.equal(gs.singular_values_mean, ref.singular_values_mean)


def test_pin_one_thread_initializer():
    from moe_compress.stage3.spectra_pool import _pin_one_thread
    saved = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        _pin_one_thread()
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(saved)


def _eff_rank_under_threads(payload, nthreads):
    from moe_compress.stage3.plugins.d_rank_allocate import _group_stat
    from moe_compress.stage3.spectra_pool import _ListBank
    saved = torch.get_num_threads()
    try:
        torch.set_num_threads(nthreads)
        gs = _group_stat(payload.n_experts, _ListBank(payload.weights_cpu),
                         A_g=payload.a_g_cpu)
        return gs.effective_rank, gs.singular_values_mean.clone()
    finally:
        torch.set_num_threads(saved)


def test_thread_pinning_holds_the_bits():
    """The 1-thread path is the canonical reduction; the pool worker reproduces
    it bit-for-bit. (The 1-vs-N-thread numeric DIFFERENCE is matrix-size
    dependent — the F2 doc measured ~1e-11 on real [2048,1536] shapes; the tiny
    16x12 fixture may not reassociate, so we do NOT hard-assert inequality. The
    load-bearing contract is worker(1-thread) == reference(1-thread).)"""
    payload = _make_payload(seed=7)
    er1, sv1 = _eff_rank_under_threads(payload, 1)
    from moe_compress.stage3.spectra_pool import _group_stat_payload, _pin_one_thread
    _pin_one_thread()
    assert torch.get_num_threads() == 1
    _, gs_w = _group_stat_payload(payload)
    assert gs_w.effective_rank == er1
    assert torch.equal(gs_w.singular_values_mean, sv1)


def _build_multigroup_payloads():
    from moe_compress.stage3.spectra_pool import _GroupStatPayload
    torch.manual_seed(13)
    d_out, d_in, n = 16, 12, 4
    payloads = []
    for li in range(2):
        for name in ("gate_proj", "up_proj", "down_proj"):
            weights = [torch.randn(d_out, d_in, dtype=torch.float32) for _ in range(n)]
            m = torch.randn(d_in, d_in, dtype=torch.float32)
            a_g = (m @ m.T + d_in * torch.eye(d_in)).to(torch.float32)
            payloads.append(_GroupStatPayload(li, name, n, weights, a_g))
    return payloads


def _run_group_stats(payloads, workers):
    from moe_compress.stage3.spectra_pool import run_group_stats_pool
    return run_group_stats_pool(payloads, workers=workers)


def test_group_stat_parallel_equals_serial():
    """PRIMARY gate: parallel (workers=2, spawn, 1-thread-pinned) == 1-thread
    serial on effective_rank (exact float) + singular_values_mean (torch.equal)."""
    payloads = _build_multigroup_payloads()
    serial = _run_group_stats(payloads, workers=1)
    parallel = _run_group_stats(payloads, workers=2)
    assert set(serial) == set(parallel)
    for key in serial:
        assert serial[key].effective_rank == parallel[key].effective_rank, key
        assert torch.equal(serial[key].singular_values_mean,
                           parallel[key].singular_values_mean), key


def test_group_stat_rank_map_equal_across_workers():
    """Downstream int rank_map identical across workers in {1,2,4}, swept over
    a couple of T budgets (a borderline round() can't hide a float drift)."""
    from moe_compress.stage3.plugins.d_rank_allocate import (
        _d_rank_allocate, _compute_T_budget,
    )
    payloads = _build_multigroup_payloads()
    base = _run_group_stats(payloads, workers=1)
    for w in (2, 4):
        cand = _run_group_stats(payloads, workers=w)
        for ratio in (0.2, 0.3, 0.5):
            Tb = _compute_T_budget(base, svd_rank_ratio=ratio)
            Tc = _compute_T_budget(cand, svd_rank_ratio=ratio)
            assert _d_rank_allocate(base, Tb) == _d_rank_allocate(cand, Tc), (w, ratio)


def test_group_stat_worker_order_invariant():
    """workers in {1,2,4} → identical effective_rank (exact) + singular_values_mean."""
    payloads = _build_multigroup_payloads()
    ref = _run_group_stats(payloads, workers=1)
    for w in (2, 4):
        got = _run_group_stats(payloads, workers=w)
        for key in ref:
            assert got[key].effective_rank == ref[key].effective_rank, (w, key)
            assert torch.equal(got[key].singular_values_mean,
                               ref[key].singular_values_mean), (w, key)
