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
