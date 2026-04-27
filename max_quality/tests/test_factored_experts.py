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


def test_count_expert_parameters_uses_effective_ranks():
    """Zero-padded columns from AA-SVD `k_eff < k` or EoRA `take_eff < r`
    must NOT be counted as real parameters by `count_expert_parameters`,
    which Stage 6's measured-reduction gate depends on. Regression for
    review B5: prior to the fix, `widen_rank` set `ranks[name] = full_r`
    and `count_expert_parameters` summed `numel()` over the full tensor."""
    from moe_compress.utils.model_io import count_expert_parameters
    import torch.nn as nn

    fe = FactoredExperts(
        num_experts=4, hidden_dim=8, intermediate_dim=4,
        ranks={"gate_proj": 4, "up_proj": 4, "down_proj": 4},
        dtype=torch.float32,
    )
    # Mark experts 0 and 1 as having only effective rank 2 (zero-padded).
    fe.effective_ranks["gate_proj"] = [2, 2, 4, 4]

    # Wrap in a minimal MoE-shaped module so iter_moe_layers picks it up.
    class _Layer(nn.Module):
        def __init__(self, fe):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = fe
            self.mlp.gate = nn.Linear(8, 4, bias=False)
            self.mlp.shared_expert = nn.Sequential()
            self.mlp.shared_expert.gate_proj = nn.Linear(8, 4, bias=False)
            self.mlp.shared_expert.up_proj = nn.Linear(8, 4, bias=False)
            self.mlp.shared_expert.down_proj = nn.Linear(4, 8, bias=False)
            self.mlp.shared_expert_gate = nn.Linear(8, 1, bias=False)

    class _Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(fe)])

    class _Cfg:
        def __init__(self):
            self.num_hidden_layers = 1
            self.layer_types = ["full_attention"]
            self.num_experts = 4
            self.num_experts_per_tok = 2
            self.hidden_size = 8
            self.moe_intermediate_size = 4
            self.text_config = self

    class _Mdl(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Tower()
            self.config = _Cfg()

    model = _Mdl()

    # Honest count: gate_proj has effective_ranks [2, 2, 4, 4] = 12; up/down
    # are still full r=4 across 4 experts = 16 each. (d_out + d_in) per matrix
    # is (4 + 8) = 12 for gate/up, (8 + 4) = 12 for down. So the contribution
    # is 12·(12 + 16 + 16) = 12·44 = 528.
    expected = 12 * (12 + 16 + 16)
    assert count_expert_parameters(model, routed_only=True) == expected, (
        "count_expert_parameters did not honor effective_ranks — "
        "Stage 6 measured_reduction would be inflated"
    )


def test_factored_experts_state_dict_round_trip():
    """`save_pretrained` (the path used by `save_compressed_checkpoint`)
    serializes FactoredExperts as `nn.Parameter`s, and `load_compressed_model`
    reads them back via streaming `_assign_storage`. This pins the
    parameter naming round-trip — a name mismatch (`gate_proj_U` vs `gate_U`,
    or expert-axis ordering changes) would only surface on a real HF Jobs
    Stage 3+ resume hours into the run.

    We exercise just the FactoredExperts ↔ state_dict path (not the full
    HF AutoConfig flow, which requires the real model class) to keep the
    test self-contained and fast.
    """
    src = FactoredExperts(
        num_experts=3, hidden_dim=8, intermediate_dim=4,
        ranks={"gate_proj": 3, "up_proj": 2, "down_proj": 4},
        dtype=torch.float32,
    )
    # Fill with deterministic values.
    torch.manual_seed(42)
    for name in ("gate_proj", "up_proj", "down_proj"):
        for attr_suffix in ("_U", "_V"):
            t = getattr(src, name + attr_suffix)
            t.data.copy_(torch.randn_like(t))
    src.effective_ranks["gate_proj"] = [2, 3, 1]   # mixed effective ranks

    sd = src.state_dict()
    # Expected keys: gate/up/down × U/V = 6 keys.
    expected_keys = {
        f"{n}_{s}" for n in ("gate_proj", "up_proj", "down_proj") for s in ("U", "V")
    }
    assert set(sd.keys()) == expected_keys, (
        f"FactoredExperts state_dict keys diverged from expected: got {set(sd.keys())}"
    )

    # Build a fresh skeleton at the same shapes and stream the state_dict in.
    dst = FactoredExperts(
        num_experts=3, hidden_dim=8, intermediate_dim=4,
        ranks={"gate_proj": 3, "up_proj": 2, "down_proj": 4},
        dtype=torch.float32,
    )
    missing, unexpected = dst.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (
        f"state_dict load left missing/unexpected keys: missing={missing}, unexpected={unexpected}"
    )

    # Tensors must match bit-for-bit.
    for name in ("gate_proj", "up_proj", "down_proj"):
        for attr_suffix in ("_U", "_V"):
            attr = name + attr_suffix
            assert torch.equal(
                getattr(src, attr), getattr(dst, attr)
            ), f"round-trip diverged on {attr}"

    # Forward output equivalence on identical input.
    T = 5
    hidden_states = torch.randn(T, 8)
    top_k_index = torch.stack([torch.randperm(3)[:2] for _ in range(T)])
    top_k_weights = torch.full((T, 2), 0.5)
    y_src = src(hidden_states, top_k_index, top_k_weights)
    y_dst = dst(hidden_states, top_k_index, top_k_weights)
    assert torch.allclose(y_src, y_dst, atol=1e-6), (
        "round-tripped FactoredExperts diverged on forward — name/order mismatch"
    )

    # NOTE: effective_ranks is metadata, not in state_dict. The save_compressed
    # _checkpoint flow persists ranks in `compressed_metadata.json`. Persisting
    # effective_ranks the same way is a follow-up; not in scope here.


def test_widen_rank_updates_effective_ranks():
    """`widen_rank` must update `effective_ranks` per expert when
    `added_effective_per_expert` is supplied (Stage 4's zero-pad path)."""
    fe = FactoredExperts(
        num_experts=3, hidden_dim=6, intermediate_dim=3,
        ranks={"gate_proj": 2, "up_proj": 2, "down_proj": 2},
        dtype=torch.float32,
    )
    # Append r=4 columns; experts 0/1 only have effective rank 1, expert 2
    # has full effective rank 4.
    U_new = torch.zeros(3, 3, 4)
    V_new = torch.zeros(3, 4, 6)
    fe.widen_rank("gate_proj", U_new, V_new, added_effective_per_expert=[1, 1, 4])

    assert fe.ranks["gate_proj"] == 6                # slot width grew
    assert fe.effective_ranks["gate_proj"] == [3, 3, 6], (
        "effective_ranks not updated correctly per expert"
    )
