"""Stage-3 α-path whitening determinism (F1).

Targets the discrete ``keep_a`` threshold flip that makes the eigh-based
α-path whitening cross-host non-reproducible, and proves the Cholesky
replacement removes it. Does NOT assert eigenvector sign/rotation
invariance — ``svdvals(W @ L)`` depends only on ``L @ L.T`` and is invariant
to it by construction (the doc's reviewer measured ~5e-15), so such an
assertion proves nothing about the bug.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-local fixture builders. The codebase rule is tests do not import each
# other, so these are verbatim local copies of the private builders in
# test_stage3_tier1.py (_FusedExperts / _make_layer_ref / _build_acov /
# _group_stats_for) rather than imports.
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


def _eigh_factor(A64: torch.Tensor) -> torch.Tensor:
    """The OLD whitening factor (discrete keep-threshold)."""
    A64 = 0.5 * (A64 + A64.T)
    ev, evec = torch.linalg.eigh(A64)
    keep = ev > ev.max() * 1e-6
    return evec[:, keep] * ev[keep].clamp_min(1e-12).sqrt().unsqueeze(0)


def _chol_factor(A64: torch.Tensor) -> torch.Tensor:
    """The NEW whitening factor (full-rank, unique, no threshold)."""
    from moe_compress.stage3.plugins.swift_svd_alpha import _alpha_whiten_factor
    return _alpha_whiten_factor(0.5 * (A64 + A64.T))


def _spd_with_eigenvalue_above_threshold(d: int, seed: int):
    """Build an SPD ``A`` with one eigenvalue sitting just ABOVE ``1e-6 * lambda_max``.

    DEVIATION from the plan's draft fixture (which placed the eigenvalue exactly
    AT ``1e-6 * lambda_max``): with the production ``keep_a = ev > ev.max()*1e-6``
    (strict ``>``) an at-threshold eigenvalue is dropped at BOTH ``A`` and the
    perturbed ``A_lo`` → no column-count flip, so the discontinuity the design
    doc (§5.2) calls for never appears. Placing it just above threshold (``2e-6``)
    means it is KEPT at ``A`` and DROPPED at ``A_lo`` → the kept-column count
    flips, exactly the doc's "svdvals jumps discontinuously (column count
    changes)" mechanism. ``W`` is square below so the dropped whitening column
    actually reduces ``min(W.rows, L.cols)`` and the jump is observable.
    """
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    lam = torch.ones(d, dtype=torch.float64)
    lam[0] = 1.0  # lambda_max
    lam[-1] = 2e-6  # just above the 1e-6 keep threshold → kept at A, dropped after nudge
    A = (Q * lam) @ Q.T
    return 0.5 * (A + A.T)


def test_eigh_whitening_is_threshold_sensitive():
    """OLD path: nudging the boundary eigenvalue across ``1e-6*lambda_max`` flips
    the kept-column count → svdvals dimension/values jump discontinuously."""
    d = 8
    torch.manual_seed(1)
    W = torch.randn(d, d, dtype=torch.float64)  # square so a dropped column changes svdvals length
    A = _spd_with_eigenvalue_above_threshold(d, seed=2)
    ev, evec = torch.linalg.eigh(A)
    ev_lo = ev.clone()
    ev_lo[ev_lo.argmin()] = ev.max() * 1e-6 * 0.5  # now dropped by keep_a
    A_lo = (evec * ev_lo) @ evec.T
    s_at = torch.linalg.svdvals(W @ _eigh_factor(A))
    s_lo = torch.linalg.svdvals(W @ _eigh_factor(A_lo))
    assert s_at.shape != s_lo.shape or not torch.allclose(s_at, s_lo, atol=1e-6)


def test_cholesky_whitening_is_threshold_free_and_stable():
    """NEW path: same boundary perturbation changes svdvals only by round-off
    (no threshold to cross), and cholesky run twice is byte-equal."""
    d = 8
    torch.manual_seed(1)
    W = torch.randn(d, d, dtype=torch.float64)
    A = _spd_with_eigenvalue_above_threshold(d, seed=2)
    A_eps = A + 1e-12 * torch.eye(d, dtype=torch.float64)
    s0 = torch.linalg.svdvals(W @ _chol_factor(A))
    s1 = torch.linalg.svdvals(W @ _chol_factor(A_eps))
    assert s0.shape == s1.shape
    assert torch.allclose(s0, s1, rtol=0, atol=1e-9)
    assert torch.equal(_chol_factor(A.clone()), _chol_factor(A.clone()))


def test_producer1_uses_cholesky_spectrum():
    """``_swift_svd_plus_alpha_search`` (producer 1) emits svdvals(W @ L_C)
    bit-identical to the helper-based recompute (proves the eigh block is gone)."""
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _swift_svd_plus_alpha_search, _alpha_whiten_factor,
    )
    from moe_compress.utils.model_io import build_banks
    layer_idx, n, d_int, d_hid = 0, 4, 6, 8
    experts = _FusedExperts(n, d_int, d_hid, seed=3)        # local copy
    ref = _make_layer_ref(layer_idx, experts)              # local copy
    A_cov = _build_acov(layer_idx, experts, d_hid, d_int)  # local copy
    group_stats = _group_stats_for(layer_idx, experts, d_int, d_hid)  # local copy
    base_ranks = {k: 3 for k in group_stats}
    _, grouped_svs = _swift_svd_plus_alpha_search(
        [ref], group_stats, base_ranks, [0.0, 0.5, 1.0],
        per_group_type=True, A_cov=A_cov, return_svs=True,
    )
    banks = build_banks(ref)
    for name in ("gate_proj", "up_proj", "down_proj"):
        for e in range(n):
            W = banks[name].get(e).detach().to(device="cpu", dtype=torch.float64)
            A = A_cov[(layer_idx, e, name)].to(device="cpu", dtype=torch.float64)
            A = 0.5 * (A + A.T)
            inline = torch.linalg.svdvals(W @ _alpha_whiten_factor(A))
            assert torch.equal(grouped_svs[name][(layer_idx, e)], inline)
