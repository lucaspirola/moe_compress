"""FactoredExperts forward equivalence with the fused reference."""
from __future__ import annotations

import pytest
import torch

from moe_compress.utils.model_io import FactoredExperts


def _lapack_available() -> bool:
    try:
        torch.linalg.svd(torch.eye(2), full_matrices=False)
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(not _lapack_available(), reason="PyTorch built without CPU LAPACK")
def test_factored_matches_fused_when_ranks_full():
    """When U @ V == W exactly (full-rank factorization), FactoredExperts
    output must match a reference fused forward within numerical tolerance."""
    torch.manual_seed(7)
    num_experts, hidden, intermediate = 3, 8, 4
    top_k = 2
    T = 6

    # Reference: fused weights drawn at random.
    gate_up_proj = torch.randn(num_experts, 2 * intermediate, hidden) * 0.1
    down_proj = torch.randn(num_experts, hidden, intermediate) * 0.1

    # Build FactoredExperts at the max rank (equal to min(d_out, d_in)).
    fe = FactoredExperts(
        num_experts=num_experts, hidden_dim=hidden, intermediate_dim=intermediate,
        ranks={"gate_proj": intermediate, "up_proj": intermediate,
               "down_proj": min(hidden, intermediate)},
        dtype=torch.float32,
    )

    # Fill factors from the fused weights via full-rank SVD.
    with torch.no_grad():
        for e in range(num_experts):
            gate = gate_up_proj[e][:intermediate]            # [int, hid]
            up = gate_up_proj[e][intermediate:]               # [int, hid]
            down = down_proj[e]                               # [hid, int]
            fe.set_factors_from_weight(e, "gate_proj", gate)
            fe.set_factors_from_weight(e, "up_proj", up)
            fe.set_factors_from_weight(e, "down_proj", down)

    # Build a reference "Qwen3_5MoeExperts"-like forward inline.
    import torch.nn.functional as F
    act = torch.nn.SiLU()
    hidden_states = torch.randn(T, hidden)
    top_k_index = torch.stack([
        torch.randperm(num_experts)[:top_k] for _ in range(T)
    ])
    top_k_weights = torch.full((T, top_k), 1.0 / top_k)

    # Reference dispatch
    def ref_forward():
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
        hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for e_idx in hit:
            e = e_idx[0]
            if e == num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[e])
            sel = hidden_states[token_idx]
            g_up = F.linear(sel, gate_up_proj[e])
            g, u = g_up.chunk(2, dim=-1)
            inter = act(g) * u
            d = F.linear(inter, down_proj[e])
            d = d * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, d)
        return final

    y_ref = ref_forward()
    y_fact = fe(hidden_states, top_k_index, top_k_weights)
    err = (y_fact - y_ref).abs().max().item()
    assert err < 1e-4, f"FactoredExperts diverged from fused reference: max abs = {err}"


@pytest.mark.skipif(not _lapack_available(), reason="PyTorch built without CPU LAPACK")
def test_widen_rank_appends_correctly():
    torch.manual_seed(0)
    fe = FactoredExperts(num_experts=2, hidden_dim=6, intermediate_dim=3,
                         ranks={"gate_proj": 2, "up_proj": 2, "down_proj": 2},
                         dtype=torch.float32)
    # Append rank 1 to gate_proj via widen_rank
    U_new = torch.randn(2, 3, 1)
    V_new = torch.randn(2, 1, 6)
    fe.widen_rank("gate_proj", U_new, V_new)
    assert fe.ranks["gate_proj"] == 3
    assert fe.gate_proj_U.shape == (2, 3, 3)
    assert fe.gate_proj_V.shape == (2, 3, 6)
