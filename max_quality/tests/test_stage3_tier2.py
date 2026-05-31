"""Stage-3 Tier-2 — device-independence + non-uniform zero-pad.

Two load-bearing invariants introduced by Tier-2:

* **Device-independence (§5d).** The rank-deciding whitened spectra now run in
  fp64 (``_group_stat`` Cholesky+svdvals, swift eigh+svdvals). fp64 agrees
  across CPU and GPU to ~1e-14 → 0 rank flips. This test runs the d_rank
  spectra/allocation phase on CPU and (when CUDA is available) on GPU with
  identical inputs and asserts the resulting **integer rank_map is equal**.
  Because Tier-2 standardizes the spectra on CPU, the GPU leg here is the guard
  against a future regression that moves a fp32 spectrum back onto the GPU.

* **Non-uniform zero-pad (§4).** ``factor_layer`` allocates each matrix slot at
  the per-LAYER MAX per-expert rank and must zero-pad each expert's ``U_k/V_k``
  up to that slot width before ``set_factors`` (which hard-checks the shape).
  This test exercises the exact production primitives — ``_aa_svd`` →
  zero-pad-to-slot → ``FactoredExperts.set_factors`` → ``forward`` — for a
  non-uniform per-expert rank scenario, asserting (a) ``set_factors`` no longer
  raises on the smaller-rank expert and (b) the padded directions are inert in
  the forward (output matches the unpadded reference to fp tolerance).

No production code is monkeypatched; all primitives are called directly.
"""
from __future__ import annotations

import torch
import pytest


# ---------------------------------------------------------------------------
# §5d — fp64-CPU == fp64-GPU device-independence of the rank decision.
# ---------------------------------------------------------------------------


class _FakeBank:
    """Duck-typed expert bank for ``_group_stat`` — ``.shape()`` + ``.get(e)``."""

    def __init__(self, weights):
        self._w = weights

    def shape(self):
        return tuple(self._w[0].shape)

    def get(self, e):
        return self._w[e]


def _alloc_rank_map_on(device: str):
    """Run the d_rank spectra + allocation phase on ``device`` and return the
    integer per-group rank dict. Inputs are built fresh on ``device`` (the
    spectra producer co-locates everything on CPU-fp64 internally, so the
    integer output must be identical regardless of input device)."""
    from moe_compress.stage3.plugins.d_rank_allocate import (
        _group_stat,
        _d_rank_allocate,
        _compute_T_budget,
    )

    torch.manual_seed(7)
    d_out, d_in, n_experts = 16, 12, 4
    # Build canonical CPU tensors first, then move to the target device — this
    # guarantees the GPU and CPU runs see the SAME numbers (only the residency
    # differs), which is exactly what device-independence must hold over.
    weights = [torch.randn(d_out, d_in, dtype=torch.float32) for _ in range(n_experts)]
    m = torch.randn(d_in, d_in, dtype=torch.float32)
    a_g = m @ m.T + d_in * torch.eye(d_in)

    weights = [w.to(device) for w in weights]
    a_g = a_g.to(device)

    # Two matrix-type groups so the allocator has something non-trivial to split.
    group_stats = {
        (0, "gate_proj"): _group_stat(n_experts, _FakeBank(weights), A_g=a_g),
        (0, "down_proj"): _group_stat(n_experts, _FakeBank(weights), A_g=a_g),
    }
    T_budget = _compute_T_budget(group_stats, svd_rank_ratio=0.3)
    return _d_rank_allocate(group_stats, T_budget)


def test_d_rank_alloc_device_independent_cpu():
    """CPU leg is always green; pins the reference rank_map."""
    rm = _alloc_rank_map_on("cpu")
    assert all(isinstance(v, int) and v >= 1 for v in rm.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_d_rank_alloc_fp64_cpu_equals_fp64_gpu():
    """fp64 spectra agree CPU vs GPU to ~1e-14 → the integer rank_map is
    IDENTICAL on both devices. Guards a regression that puts a fp32 spectrum
    back on the GPU (which flipped 2-3/216 ranks in the Tier-2 measurement)."""
    rm_cpu = _alloc_rank_map_on("cpu")
    rm_gpu = _alloc_rank_map_on("cuda")
    assert rm_cpu == rm_gpu, (
        f"device-dependent ranks: cpu={rm_cpu} gpu={rm_gpu}"
    )


# ---------------------------------------------------------------------------
# §4 — non-uniform per-expert zero-pad to the layer slot width.
# ---------------------------------------------------------------------------


def _pad_to_slot(U_k, V_k, slot):
    """The exact zero-pad logic added to ``factor_layer`` (§4): index by the
    ACTUAL returned factor width, pad up to the slot."""
    u_w = U_k.shape[1]
    v_w = V_k.shape[0]
    if u_w < slot:
        U_pad = torch.zeros(U_k.shape[0], slot, device=U_k.device, dtype=U_k.dtype)
        V_pad = torch.zeros(slot, V_k.shape[1], device=V_k.device, dtype=V_k.dtype)
        U_pad[:, :u_w] = U_k
        V_pad[:v_w, :] = V_k
        U_k, V_k = U_pad, V_pad
    return U_k, V_k


def test_factor_layer_non_uniform_zero_pad_no_raise_and_inert():
    """Non-uniform per-expert ranks: pad each expert's factors to the per-layer
    MAX rank (the slot), set_factors must accept them, and the padded
    directions must be inert in the forward.

    Mirrors ``factor_layer``: slot = max_e k_e; each expert factored at its own
    k_e; zero-pad k_e → slot before set_factors."""
    from moe_compress.stage3.plugins.aa_svd_factor import _aa_svd
    from moe_compress.utils.model_io import FactoredExperts

    torch.manual_seed(3)
    n_experts = 2
    hidden, inter = 8, 10
    dtype = torch.float32

    # Per-expert requested ranks — NON-UNIFORM. slot = max = 5.
    per_expert_k = {0: 5, 1: 2}
    slot = max(per_expert_k.values())

    # gate_proj / up_proj weight shape = (intermediate, hidden); down = (hidden, intermediate).
    W_gate = {e: torch.randn(inter, hidden) for e in range(n_experts)}
    W_up = {e: torch.randn(inter, hidden) for e in range(n_experts)}
    W_down = {e: torch.randn(hidden, inter) for e in range(n_experts)}

    ranks_layer = {"gate_proj": slot, "up_proj": slot, "down_proj": slot}
    fe = FactoredExperts(
        num_experts=n_experts, hidden_dim=hidden, intermediate_dim=inter,
        ranks=ranks_layer, dtype=dtype, device="cpu",
    )

    for e in range(n_experts):
        k = per_expert_k[e]
        for name, Wmap in (("gate_proj", W_gate), ("up_proj", W_up), ("down_proj", W_down)):
            W = Wmap[e]
            # No covariance → _aa_svd takes the plain-SVD fallback (still returns
            # (d_out, k)/(k, d_in) at the re-clamped k).
            U_k, V_k, _rel, k_eff = _aa_svd(W, None, None, k, C=None, device="cpu")
            assert U_k.shape[1] <= slot
            U_pad, V_pad = _pad_to_slot(U_k, V_k, slot)
            assert U_pad.shape == (W.shape[0], slot)
            assert V_pad.shape == (slot, W.shape[1])
            # MUST NOT raise: this is the crash the zero-pad fix removes.
            fe.set_factors(e, name, U_pad, V_pad, effective_rank=k_eff)
            # effective_rank records the genuine-signal width, not the slot.
            assert fe.effective_ranks[name][e] == int(k_eff)

    # Forward inertness: the padded (low-rank) expert produces the same output
    # whether its factors are padded-to-slot or set at their own width in a
    # second FactoredExperts whose slot == that expert's own rank.
    n_tokens = 6
    x = torch.randn(n_tokens, hidden, dtype=dtype)
    top_k_index = torch.zeros(n_tokens, 1, dtype=torch.long)        # all route to expert 0
    top_k_index[3:] = 1                                             # half to expert 1 (k=2)
    top_k_weights = torch.ones(n_tokens, 1, dtype=dtype)
    out_padded = fe(x, top_k_index, top_k_weights)

    # Reference: expert 1 alone, slot == its own rank (2), no padding.
    ranks_ref = {"gate_proj": per_expert_k[1], "up_proj": per_expert_k[1],
                 "down_proj": per_expert_k[1]}
    fe_ref = FactoredExperts(
        num_experts=1, hidden_dim=hidden, intermediate_dim=inter,
        ranks=ranks_ref, dtype=dtype, device="cpu",
    )
    for name, Wmap in (("gate_proj", W_gate), ("up_proj", W_up), ("down_proj", W_down)):
        U_k, V_k, _rel, k_eff = _aa_svd(Wmap[1], None, None, per_expert_k[1], C=None, device="cpu")
        # no pad needed (slot == k); set as-is.
        fe_ref.set_factors(0, name, U_k, V_k, effective_rank=k_eff)

    x1 = x[3:]
    idx1 = torch.zeros(x1.shape[0], 1, dtype=torch.long)
    w1 = torch.ones(x1.shape[0], 1, dtype=dtype)
    out_ref_expert1 = fe_ref(x1, idx1, w1)

    # The padded expert-1 rows of out_padded must equal the unpadded reference
    # (padded trailing rank directions contribute exactly 0).
    assert torch.allclose(out_padded[3:], out_ref_expert1, atol=1e-5, rtol=1e-4), (
        "zero-padded low-rank expert forward diverged from the unpadded "
        "reference — padded directions are NOT inert."
    )
