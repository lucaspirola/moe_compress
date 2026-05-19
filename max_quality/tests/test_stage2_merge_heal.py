"""Component tests for the Stage-2 per-layer merge-heal.

Covers the correctness-critical pieces of the opt-in merge-heal feature
(``stage2_reap_ream``):

- ``_heal_student_moe_output`` — the faithful replica of the Qwen MoE block's
  routed forward (softmax → top-k → renormalize → weighted SwiGLU sum + shared
  expert). Verified against hand-computed references, not a re-implementation.
- ``_build_heal_optimizers`` — the Muon (2D) + AdamW (1D) optimizer split.
- ``_HealConfig`` — validation gating (inert when disabled, strict when on).
- ``_CascadeBuffer`` — advancing decoder layers reproduces a plain forward
  (catches the stale-KV-cache and wrong-mask classes of bug).
- ``_heal_layer`` — runs end-to-end on a tiny real MoE *after* ``bank.select``
  has re-indexed the expert banks (catches original-id-vs-position bugs).

The last two use a tiny randomly-initialized ``Qwen3MoeForCausalLM`` — small
enough to run on CPU; the full-scale behaviour is validated by the on-GPU
smoke run.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_compress.stage2_reap_ream import (
    _build_heal_optimizers,
    _CascadeBuffer,
    _HealConfig,
    _heal_layer,
    _heal_student_moe_output,
    _resize_router_for_kept_experts,
    _swiglu_forward,
)
from moe_compress.utils.model_io import (
    _find_text_tower,
    build_banks,
    iter_moe_layers,
)
from moe_compress.utils.muon import Muon


def _tiny_moe_model():
    """A tiny randomly-initialized Qwen3-MoE causal LM (CPU, fp32, inference)."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        vocab_size=128, hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True, decoder_sparse_step=1, max_position_embeddings=64,
        head_dim=8,
    )
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg)
    model.train(False)
    return model


def _random_expert(hidden: int, d_int: int) -> dict[str, torch.Tensor]:
    return {
        "gate_proj": torch.randn(d_int, hidden),
        "up_proj": torch.randn(d_int, hidden),
        "down_proj": torch.randn(hidden, d_int),
    }


# ---------------------------------------------------------------------------
# _heal_student_moe_output — routing replica
# ---------------------------------------------------------------------------


def test_heal_moe_output_shape():
    torch.manual_seed(0)
    hidden, d_int, n_kept, T = 8, 6, 4, 10
    x = torch.randn(T, hidden)
    out = _heal_student_moe_output(
        x=x,
        router_weight=torch.randn(n_kept, hidden),
        router_bias=None,
        esc_bias=None,
        expert_params={c: _random_expert(hidden, d_int) for c in range(n_kept)},
        centroid_order=list(range(n_kept)),
        top_k=2,
        shared_out=torch.zeros(T, hidden),
    )
    assert out.shape == (T, hidden)


def test_heal_moe_output_top1_selects_argmax_expert():
    """With top_k=1 the renormalized weight is exactly 1.0, so the routed
    output must equal the argmax expert's SwiGLU plus the shared output."""
    torch.manual_seed(1)
    hidden, d_int, n_kept, T = 8, 6, 4, 16
    x = torch.randn(T, hidden)
    router_weight = torch.randn(n_kept, hidden)
    experts = {c: _random_expert(hidden, d_int) for c in range(n_kept)}
    shared = torch.randn(T, hidden)

    out = _heal_student_moe_output(
        x=x, router_weight=router_weight, router_bias=None, esc_bias=None,
        expert_params=experts, centroid_order=list(range(n_kept)),
        top_k=1, shared_out=shared,
    )

    sel = torch.argmax(x @ router_weight.T, dim=-1)  # (T,)
    expected = shared.clone()
    for t in range(T):
        e = experts[int(sel[t])]
        expected[t] += _swiglu_forward(
            e["gate_proj"], e["up_proj"], e["down_proj"], x[t:t + 1]
        )[0]
    assert torch.allclose(out, expected, atol=1e-5)


def test_heal_moe_output_renormalizes_topk_weights():
    """When every kept expert is identical, the top-k weights renormalize to
    sum 1, so the routed output equals one expert's SwiGLU regardless of k."""
    torch.manual_seed(2)
    hidden, d_int, n_kept, T = 8, 6, 5, 12
    x = torch.randn(T, hidden)
    one = _random_expert(hidden, d_int)
    experts = {c: one for c in range(n_kept)}  # all identical
    shared = torch.randn(T, hidden)
    expected = shared + _swiglu_forward(
        one["gate_proj"], one["up_proj"], one["down_proj"], x
    )

    for k in (1, 2, 3, n_kept):
        out = _heal_student_moe_output(
            x=x, router_weight=torch.randn(n_kept, hidden),
            router_bias=None, esc_bias=None, expert_params=experts,
            centroid_order=list(range(n_kept)), top_k=k, shared_out=shared,
        )
        assert torch.allclose(out, expected, atol=1e-5), f"k={k}"


def test_heal_moe_output_centroid_order_indirection():
    """centroid_order maps router rows → expert ids; routing must follow it."""
    torch.manual_seed(3)
    hidden, d_int, T = 8, 6, 8
    x = torch.randn(T, hidden)
    # Two experts under non-identity ids; router row 0 → id 7, row 1 → id 3.
    experts = {7: _random_expert(hidden, d_int), 3: _random_expert(hidden, d_int)}
    order = [7, 3]
    router_weight = torch.randn(2, hidden)
    out = _heal_student_moe_output(
        x=x, router_weight=router_weight, router_bias=None, esc_bias=None,
        expert_params=experts, centroid_order=order, top_k=1,
        shared_out=torch.zeros(T, hidden),
    )
    sel = torch.argmax(x @ router_weight.T, dim=-1)
    expected = torch.zeros(T, hidden)
    for t in range(T):
        e = experts[order[int(sel[t])]]
        expected[t] = _swiglu_forward(
            e["gate_proj"], e["up_proj"], e["down_proj"], x[t:t + 1]
        )[0]
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# _build_heal_optimizers
# ---------------------------------------------------------------------------


def test_build_heal_optimizers_muon_splits_2d_and_1d():
    cfg = _HealConfig({"merge_heal_optimizer": "muon"}, Path("/tmp"))
    p2d = [torch.nn.Parameter(torch.randn(4, 4))]
    p1d = [torch.nn.Parameter(torch.randn(4))]
    optims = _build_heal_optimizers(p2d, p1d, cfg)
    assert len(optims) == 2
    assert any(isinstance(o, Muon) for o in optims)
    assert any(isinstance(o, torch.optim.AdamW) for o in optims)


def test_build_heal_optimizers_muon_no_1d_params():
    cfg = _HealConfig({"merge_heal_optimizer": "muon"}, Path("/tmp"))
    optims = _build_heal_optimizers([torch.nn.Parameter(torch.randn(4, 4))], [], cfg)
    assert len(optims) == 1 and isinstance(optims[0], Muon)


def test_build_heal_optimizers_adamw_single_group():
    cfg = _HealConfig({"merge_heal_optimizer": "adamw"}, Path("/tmp"))
    p2d = [torch.nn.Parameter(torch.randn(4, 4))]
    p1d = [torch.nn.Parameter(torch.randn(4))]
    optims = _build_heal_optimizers(p2d, p1d, cfg)
    assert len(optims) == 1 and isinstance(optims[0], torch.optim.AdamW)


# ---------------------------------------------------------------------------
# _HealConfig
# ---------------------------------------------------------------------------


def test_heal_config_disabled_is_inert():
    """A disabled config must not validate — an empty block is legal."""
    cfg = _HealConfig({}, Path("/tmp"))
    assert cfg.enabled is False


def test_heal_config_enabled_requires_sidecar_path():
    with pytest.raises(RuntimeError, match="sidecar_path"):
        _HealConfig({"merge_heal_enabled": True}, Path("/tmp"))


def test_heal_config_rejects_bad_optimizer():
    with pytest.raises(ValueError, match="merge_heal_optimizer"):
        _HealConfig(
            {"merge_heal_enabled": True, "merge_heal_sidecar_path": "/tmp/x.pt",
             "merge_heal_optimizer": "sgd"},
            Path("/tmp"),
        )


def test_heal_config_rejects_bad_holdout_fraction():
    with pytest.raises(ValueError, match="holdout_fraction"):
        _HealConfig(
            {"merge_heal_enabled": True, "merge_heal_sidecar_path": "/tmp/x.pt",
             "merge_heal_holdout_fraction": 1.5},
            Path("/tmp"),
        )


def test_heal_config_relative_sidecar_resolves_under_artifacts():
    cfg = _HealConfig(
        {"merge_heal_enabled": True, "merge_heal_sidecar_path": "sub/x.pt"},
        Path("/art"),
    )
    assert cfg.sidecar_path == Path("/art/sub/x.pt")


# ---------------------------------------------------------------------------
# _CascadeBuffer — on a tiny real Qwen3-MoE model
# ---------------------------------------------------------------------------


def test_cascade_buffer_matches_reference_forward():
    """Advancing the cascade buffer through decoder layers must reproduce the
    plain model forward's layer input. This catches (a) the stale-KV-cache bug
    — replaying with a captured DynamicCache would append KV and crash/corrupt
    every full-attention layer — and (b) any wrong attention mask.
    """
    model = _tiny_moe_model()
    tower = _find_text_tower(model)
    n_seq, seq_len = 4, 8
    ids = torch.randint(0, model.config.vocab_size, (n_seq, seq_len))

    # Reference: capture the input hidden-state of layer 2 in a plain forward.
    ref: dict = {}

    def _grab(_m, args, _kw):
        ref["x"] = args[0].detach()

    h = tower.layers[2].register_forward_pre_hook(_grab, with_kwargs=True)
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()

    # Cascade buffer: seed + advance through layers 0,1 → should hold layer-2 input.
    cb = _CascadeBuffer(
        model, tower, ids, seq_len=seq_len, batch_size=2, device=torch.device("cpu"),
    )
    cb.advance_to(2)
    buf = cb.buffer.reshape(n_seq, seq_len, -1).float()

    # The buffer runs in bf16; compare with a tolerance via per-row cosine sim.
    a = buf.reshape(-1, buf.shape[-1])
    b = ref["x"].reshape(-1, ref["x"].shape[-1]).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    assert cos.min().item() > 0.98, f"cascade buffer diverged: min cos {cos.min()}"


# ---------------------------------------------------------------------------
# _heal_layer — end-to-end after bank.select() re-indexing
# ---------------------------------------------------------------------------


def test_heal_layer_runs_after_bank_select_reindexing():
    """`_heal_layer` runs AFTER `bank.select()` has re-indexed the expert banks
    to 0..n_kept-1. It must index banks by post-select POSITION, not by the
    original expert id — kept ids deliberately include ids >= n_kept here, so
    an original-id index would raise (or read the wrong expert)."""
    model = _tiny_moe_model()
    tower = _find_text_tower(model)
    refs = list(iter_moe_layers(model))
    ref = refs[1]  # heal the middle layer

    n_seq, seq_len = 4, 8
    ids = torch.randint(0, model.config.vocab_size, (n_seq, seq_len))
    cb = _CascadeBuffer(
        model, tower, ids, seq_len=seq_len, batch_size=2, device=torch.device("cpu"),
    )
    cb.advance_to(ref.layer_idx)

    # Simulate a merge: keep 4 of 8 experts, ids chosen so max kept id (7) > n_kept (4).
    final_kept_ids = [0, 2, 5, 7]
    grouped = {0: [0, 1], 2: [2], 5: [5], 7: [7, 4, 6]}  # 0 and 7 absorbed children
    banks = build_banks(ref)
    for bank in banks.values():
        bank.select(final_kept_ids)
    _resize_router_for_kept_experts(ref, final_kept_ids)

    # Pre-heal snapshot of a merged centroid (post-select position 0 == id 0).
    before = banks["gate_proj"].get(0).detach().clone()

    heal_cfg = _HealConfig(
        {
            "merge_heal_enabled": True, "merge_heal_sidecar_path": "dummy.pt",
            "merge_heal_optimizer": "muon", "merge_heal_max_steps": 30,
            "merge_heal_eval_interval": 10, "merge_heal_patience": 3,
            "merge_heal_token_cap": 256, "merge_heal_minibatch_size": 16,
        },
        Path("/tmp"),
    )
    teacher_out = torch.randn(cb.n_tokens, model.config.hidden_size, dtype=torch.bfloat16)

    state = _heal_layer(
        layer_ref=ref, grouped=grouped, final_kept_ids=final_kept_ids,
        cascade_buffer=cb, teacher_layer_output=teacher_out,
        heal_cfg=heal_cfg, device=torch.device("cpu"),
    )

    assert state["steps"] > 0
    assert state["stop_reason"] in ("max_steps", "patience")
    # A merged centroid must have been updated by the heal.
    after = build_banks(ref)["gate_proj"].get(0).detach()
    assert not torch.equal(before, after), "merged centroid weights did not change"
