"""Stage-3 Swift-SVD+ α-spectra vectorization parity.

The per-expert fp64 ``cholesky`` + ``svdvals(W @ L_C)`` is the Stage-3 α-phase
bottleneck (~1 h single-threaded on the 35B). ``_build_grouped_svs`` fans it
across a ``ThreadPoolExecutor`` (svdvals/cholesky release the GIL) instead of
the prior serial loop. These tests pin the parity that makes the speedup safe:

* serial (``svd_workers=1``) and parallel (``svd_workers>1``) builds are
  BYTE-IDENTICAL (``torch.equal``) — completion order never leaks into the
  per-expert spectrum (the dict is keyed by ``(layer, expert)``);
* the threaded build equals the inline ``svdvals(W @ L_C)`` reference;
* α selection + per-expert rank maps are identical across worker counts;
* a fresh parallel build (cache absent — the orchestrator α-resume path) gives
  the same ranks as the threaded proxy cache;
* the D-raw-svd-fallback warn-once latch stays exactly-once under the pool.

Fixture builders are verbatim local copies (the codebase rule is tests do not
import each other) of the private builders in ``test_stage3_alpha_determinism``.
Matrix sizes are kept below the BLAS multi-thread threshold so the spectra are
thread-count-invariant and the ``torch.equal`` parity is exact.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Verbatim local fixture builders (no cross-test imports).
# ---------------------------------------------------------------------------
class _FusedExperts(nn.Module):
    """Minimal fused-layout experts module matching ``_is_fused_experts``."""

    def __init__(self, n_experts, d_int, d_hid, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_experts = n_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(n_experts, 2 * d_int, d_hid, generator=g)
        )
        self.down_proj = nn.Parameter(
            torch.randn(n_experts, d_hid, d_int, generator=g)
        )


def _make_layer_ref(layer_idx, experts):
    from moe_compress.utils.model_io import MoELayerRef

    dummy = nn.Identity()
    return MoELayerRef(
        layer_idx=layer_idx,
        layer_module=dummy,
        mlp=dummy,
        router=dummy,
        experts_module=experts,
        shared_expert=None,
        layer_type="unknown",
    )


def _build_acov(layer_idx, experts, d_hid, d_int, seed=1):
    A_cov = {}
    n = experts.num_experts
    for e in range(n):
        g = torch.Generator().manual_seed(seed + e)
        for name, dim in (("gate_proj", d_hid), ("up_proj", d_hid),
                          ("down_proj", d_int)):
            M = torch.randn(dim, dim, generator=g)
            A_cov[(layer_idx, e, name)] = (M @ M.T) + torch.eye(dim)
    return A_cov


def _group_stats_for(layer_idx, experts, d_int, d_hid):
    from moe_compress.stage3.plugins.d_rank_allocate import _GroupStats

    n = experts.num_experts
    dims = {"gate_proj": (d_int, d_hid), "up_proj": (d_int, d_hid),
            "down_proj": (d_hid, d_int)}
    gs = {}
    for name, (d_out, d_in) in dims.items():
        gs[(layer_idx, name)] = _GroupStats(
            d_out=d_out, d_in=d_in, n_experts=n,
            singular_values_mean=torch.ones(min(d_out, d_in)),
            effective_rank=float(min(d_out, d_in)) / 2.0,
            omega=n * (d_out + d_in),
        )
    return gs


# A two-layer, several-expert problem so the pool sees real fan-out.
def _problem(seed=3, n=8, d_int=48, d_hid=64):
    refs, gs, A_cov = [], {}, {}
    for layer_idx in (0, 1):
        experts = _FusedExperts(n, d_int, d_hid, seed=seed + layer_idx)
        refs.append(_make_layer_ref(layer_idx, experts))
        gs.update(_group_stats_for(layer_idx, experts, d_int, d_hid))
        A_cov.update(_build_acov(layer_idx, experts, d_hid, d_int,
                                 seed=1 + 100 * layer_idx))
    base_ranks = {k: 7 for k in gs}
    return refs, gs, base_ranks, A_cov


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_resolve_svd_workers_clamps():
    from moe_compress.stage3.plugins.swift_svd_alpha import _resolve_svd_workers

    # None → os.cpu_count(), capped to n_tasks (never more workers than experts).
    assert _resolve_svd_workers(None, 1) == 1
    assert _resolve_svd_workers(None, 3) <= 3
    assert _resolve_svd_workers(None, 3) >= 1
    # Explicit int clamps to [1, n_tasks].
    assert _resolve_svd_workers(99, 4) == 4
    assert _resolve_svd_workers(2, 4) == 2
    assert _resolve_svd_workers(0, 4) == 1
    assert _resolve_svd_workers(-5, 4) == 1
    # No tasks → 1 (degenerate, never divides by zero downstream).
    assert _resolve_svd_workers(None, 0) == 1


def test_build_grouped_svs_serial_parallel_byte_identical():
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _build_grouped_svs, _alpha_whiten_factor,
    )
    from moe_compress.utils.model_io import build_banks

    refs, gs, _base, A_cov = _problem()

    serial = _build_grouped_svs(refs, gs, A_cov, svd_workers=1)
    parallel = _build_grouped_svs(refs, gs, A_cov, svd_workers=8)
    parallel2 = _build_grouped_svs(refs, gs, A_cov, svd_workers=8)

    # Serial (no pool), pool, and a second pool run are all BYTE-IDENTICAL:
    # the 1-thread-BLAS pin makes the per-expert spectrum independent of worker
    # count and completion order. These 64×48 fixtures sit BELOW the typical
    # BLAS multi-thread threshold, so they would already be identical by
    # coincidence; the pin makes the guarantee explicit and size-independent
    # (production 2048² matrices DO drift ~1e-13 across thread counts without
    # it). `torch.equal` here proves serial==pool==pool2 under the pin.
    assert serial.keys() == parallel.keys()
    for name in serial:
        assert serial[name].keys() == parallel[name].keys()
        for key in serial[name]:
            assert torch.equal(serial[name][key], parallel[name][key]), \
                f"serial != parallel at {name} {key}"
            assert torch.equal(parallel[name][key], parallel2[name][key]), \
                f"pool not reproducible at {name} {key}"

    # ...and byte-identical to the inline svdvals(W @ L_C) reference computed
    # under the SAME 1-thread BLAS pin.
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for ref in refs:
            li = ref.layer_idx
            banks = build_banks(ref)
            n_e = gs[(li, "gate_proj")].n_experts
            for name in ("gate_proj", "up_proj", "down_proj"):
                for e in range(n_e):
                    W = banks[name].get(e).detach().to(
                        device="cpu", dtype=torch.float64)
                    A = A_cov[(li, e, name)].to(
                        device="cpu", dtype=torch.float64)
                    A = 0.5 * (A + A.T)
                    inline = torch.linalg.svdvals(W @ _alpha_whiten_factor(A))
                    assert torch.equal(parallel[name][(li, e)], inline)
    finally:
        torch.set_num_threads(saved)


def test_build_grouped_svs_restores_thread_count():
    from moe_compress.stage3.plugins.swift_svd_alpha import _build_grouped_svs

    refs, gs, _base, A_cov = _problem()
    before = torch.get_num_threads()
    _build_grouped_svs(refs, gs, A_cov, svd_workers=8)
    assert torch.get_num_threads() == before, \
        "intra-op thread count not restored after the parallel section"


def test_alpha_search_invariant_across_worker_counts():
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _swift_svd_plus_alpha_search,
    )
    refs, gs, base, A_cov = _problem()
    grid = [0.0, 0.5, 1.0]

    a1, svs1 = _swift_svd_plus_alpha_search(
        refs, gs, base, grid, per_group_type=True, A_cov=A_cov,
        return_svs=True, svd_workers=1)
    a8, svs8 = _swift_svd_plus_alpha_search(
        refs, gs, base, grid, per_group_type=True, A_cov=A_cov,
        return_svs=True, svd_workers=8)

    assert a1 == a8, f"alpha_by_type differs across workers: {a1} vs {a8}"
    for name in svs1:
        for key in svs1[name]:
            assert torch.equal(svs1[name][key], svs8[name][key])


def test_redistribute_fresh_build_equals_cache_and_workers():
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _swift_svd_plus_alpha_search, _redistribute_ranks_swift_svd_plus,
    )
    refs, gs, base, A_cov = _problem()
    grid = [0.0, 0.5, 1.0]

    alpha, cache = _swift_svd_plus_alpha_search(
        refs, gs, base, grid, per_group_type=True, A_cov=A_cov,
        return_svs=True, svd_workers=4)

    # (a) reuse the threaded proxy cache (production select_alpha path)
    r_cache = _redistribute_ranks_swift_svd_plus(
        refs, gs, base, alpha, grouped_svs_cache=cache, A_cov=A_cov)
    # (b) cache absent → fresh parallel build (orchestrator α-resume path)
    r_fresh_8 = _redistribute_ranks_swift_svd_plus(
        refs, gs, base, alpha, A_cov=A_cov, svd_workers=8)
    # (c) cache absent, serial reference
    r_fresh_1 = _redistribute_ranks_swift_svd_plus(
        refs, gs, base, alpha, A_cov=A_cov, svd_workers=1)

    assert r_cache == r_fresh_8 == r_fresh_1
    # Budget conservation as a sanity backstop.
    for (li, name), g in gs.items():
        k_group = base[(li, name)]
        total = sum(r_cache[(li, name, e)] for e in range(g.n_experts))
        assert total == k_group * g.n_experts


def test_raw_svd_fallback_warns_once_under_pool():
    """A_cov=None across many experts in the parallel build → the warn-once
    latch fires EXACTLY once (the threading.Lock guard, not a racy global).

    A dedicated handler is attached to the module logger (standard logging API
    — not a monkeypatch) so the count is captured deterministically even though
    the warning is emitted from worker threads.
    """
    import logging
    from moe_compress.stage3.plugins import swift_svd_alpha as ssa

    refs, gs, _base, _A = _problem()
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger(ssa.__name__)
    handler = _Collector()
    logger.addHandler(handler)
    ssa._reset_raw_svd_fallback_warning()
    try:
        ssa._build_grouped_svs(refs, gs, A_cov=None, svd_workers=8)
    finally:
        logger.removeHandler(handler)
        ssa._reset_raw_svd_fallback_warning()

    fallback = [r for r in records if "D-raw-svd-fallback" in r.getMessage()]
    assert len(fallback) == 1, \
        f"expected exactly one warn-once message, got {len(fallback)}"
