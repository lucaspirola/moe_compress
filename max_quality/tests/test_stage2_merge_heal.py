"""Component tests for the Stage-2 per-layer merge-heal (self-distillation v2).

Covers the correctness-critical pieces of the opt-in merge-heal feature
(``stage2_reap_ream``):

- ``_heal_student_moe_output`` — the faithful replica of the Qwen MoE block's
  routed forward (softmax → top-k → renormalize → weighted SwiGLU sum + shared
  expert). Verified against hand-computed references, not a re-implementation.
- ``_HealConfig`` — validation gating (inert when disabled, strict when on).
- ``_capture_mlp_io`` — captures row-aligned (mlp_input, mlp_output) pools at
  the MoE-block boundary (the correct hook point for self-distillation).
- ``_heal_layer`` — runs end-to-end on a tiny real MoE *after* ``bank.select``
  has re-indexed the expert banks; checks ALL kept experts train, the router
  flag, and the monotone-safe accept/reject guard.
- ``_write_heal_weights`` / ``_load_heal_weights`` — the per-layer checkpoint
  round-trip (all kept experts + router, format v2).

The model-level tests use a tiny randomly-initialized ``Qwen3MoeForCausalLM`` —
small enough to run on CPU; full-scale behaviour is validated by the on-GPU
smoke run.
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.stage2_reap_ream import (
    _HealConfig,
    _capture_mlp_io,
    _heal_layer,
    _heal_student_moe_output,
    _load_heal_weights,
    _resize_router_for_kept_experts,
    _swiglu_forward,
    _write_heal_weights,
)
from moe_compress.utils.model_io import (
    MATRIX_NAMES,
    build_banks,
    iter_moe_layers,
)

_CPU = torch.device("cpu")


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
    # SwiGLU weight shapes are asymmetric: gate_proj / up_proj map hidden -> d_int
    # so they are [d_int, hidden]; down_proj maps the intermediate d_int back to
    # hidden so it is [hidden, d_int]. This matches the Qwen MoE expert layout.
    return {
        "gate_proj": torch.randn(d_int, hidden),
        "up_proj": torch.randn(d_int, hidden),
        "down_proj": torch.randn(hidden, d_int),
    }


def _merge_layer(ref, final_kept_ids: list[int]) -> None:
    """Apply a merge to one layer: select kept experts + resize the router.

    This is the post-merge state `_heal_layer` expects (it runs AFTER
    `bank.select()` has re-indexed the banks to 0..n_kept-1).
    """
    banks = build_banks(ref)
    for bank in banks.values():
        bank.select(final_kept_ids)
    _resize_router_for_kept_experts(ref, final_kept_ids)


def _id_batches(model, n_seq: int = 16, seq_len: int = 8, chunk: int = 4,
                seed: int = 0):
    """Deterministic synthetic token batches.

    The token draw is seeded via an explicit `torch.Generator` so the accept-
    based heal tests are reproducible run-to-run (an unseeded `torch.randint`
    made them flaky near the accept margin).
    """
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (n_seq, seq_len),
                        generator=gen)
    return [ids[i:i + chunk] for i in range(0, n_seq, chunk)]


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
# _HealConfig
# ---------------------------------------------------------------------------


def test_heal_config_disabled_is_inert():
    """A disabled config must not validate — an empty block is legal."""
    cfg = _HealConfig({})
    assert cfg.enabled is False


def test_heal_config_enabled_defaults():
    cfg = _HealConfig({"merge_heal_enabled": True})
    assert cfg.enabled is True
    assert cfg.train_router is True          # default ON
    assert cfg.lr == pytest.approx(1.0e-4)
    assert cfg.token_cap == 262144


def test_heal_config_rejects_bad_holdout_fraction():
    with pytest.raises(ValueError, match="holdout_fraction"):
        _HealConfig({"merge_heal_enabled": True,
                     "merge_heal_holdout_fraction": 1.5})


def test_heal_config_rejects_bad_grad_clip():
    with pytest.raises(ValueError, match="grad_clip"):
        _HealConfig({"merge_heal_enabled": True, "merge_heal_grad_clip": 0.0})


# ---------------------------------------------------------------------------
# _capture_mlp_io — on a tiny real Qwen3-MoE model
# ---------------------------------------------------------------------------


def test_capture_mlp_io_pools_aligned():
    """`_capture_mlp_io` records row-aligned (mlp_input, mlp_output) pairs at
    the MoE-block boundary. Feeding the captured input back through the (still
    original) mlp must reproduce the captured target by ABSOLUTE error —
    confirms the hook is on the right module, the i/o correspond, AND the
    captured magnitudes are correct (a scale bug would survive a scale-
    invariant check but not an allclose)."""
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[1]
    batches = _id_batches(model, n_seq=4, seq_len=8, chunk=2)

    cap_in, cap_out = _capture_mlp_io(
        model, ref, batches, device=_CPU, pool_size=10_000,
    )
    assert cap_in.shape == cap_out.shape
    assert cap_in.shape == (4 * 8, model.config.hidden_size)
    assert cap_in.dtype == torch.bfloat16 and cap_out.dtype == torch.bfloat16

    x3d = cap_in.to(torch.float32).reshape(1, -1, model.config.hidden_size)
    with torch.no_grad():
        out = ref.mlp(x3d)
    out = (out[0] if isinstance(out, tuple) else out).reshape(
        -1, model.config.hidden_size)
    # Absolute-error check is the primary assertion — it catches magnitude
    # bugs that a scale-invariant cosine would miss. Tolerances are loose
    # because the captured pools are stored bf16.
    assert torch.allclose(out.float(), cap_out.float(), rtol=1e-2, atol=1e-2), (
        f"capture i/o magnitude mismatch: max abs err "
        f"{(out.float() - cap_out.float()).abs().max().item()}"
    )


def test_capture_mlp_io_respects_pool_size():
    """The pool is capped at pool_size; the forward loop stops early. The cap
    is a genuine PREFIX — capturing the same batches uncapped must reproduce
    the capped pool as its leading rows."""
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[0]
    batches = _id_batches(model, n_seq=16, seq_len=8, chunk=4)  # 128 tokens
    cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU, pool_size=40)
    assert cap_in.shape[0] == 40 and cap_out.shape[0] == 40

    # Same batches, large pool_size — the capped capture must be its prefix.
    cap_in_big, cap_out_big = _capture_mlp_io(
        model, ref, batches, device=_CPU, pool_size=10_000)
    assert cap_in_big.shape[0] == 128
    assert torch.equal(cap_in, cap_in_big[:40]), "pool cap is not a prefix"
    assert torch.equal(cap_out, cap_out_big[:40]), "pool cap is not a prefix"


# ---------------------------------------------------------------------------
# _heal_layer — end-to-end after bank.select() re-indexing
# ---------------------------------------------------------------------------

# Kept ids chosen so the max kept id (7) exceeds n_kept (4): a heal that indexed
# banks by original expert id instead of post-select position would raise.
_KEPT = [0, 2, 5, 7]


def _heal_cfg(**overrides) -> _HealConfig:
    base = {
        "merge_heal_enabled": True, "merge_heal_lr": 1.0e-3,
        "merge_heal_max_steps": 300, "merge_heal_eval_interval": 20,
        "merge_heal_patience": 20, "merge_heal_minibatch_size": 16,
    }
    base.update(overrides)
    return _HealConfig(base)


def test_heal_layer_runs_after_bank_select_reindexing():
    """`_heal_layer` runs AFTER `bank.select()` re-indexed the banks to
    0..n_kept-1; it must index by post-select POSITION. With a reachable
    self-distillation target the heal is accepted by a comfortable margin and
    the weights move."""
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[1]
    batches = _id_batches(model, n_seq=64)  # ~512 tokens -> ~51 holdout

    # Capture BEFORE merge — target = the layer's own pre-merge MoE output.
    cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU, pool_size=10_000)
    _merge_layer(ref, _KEPT)
    before = build_banks(ref)["gate_proj"].get(0).detach().clone()

    state = _heal_layer(
        layer_ref=ref, final_kept_ids=_KEPT,
        captured_input=cap_in, captured_target=cap_out,
        heal_cfg=_heal_cfg(), device=_CPU,
    )
    assert state["steps"] > 0
    # The heal must clear the accept threshold by a comfortable margin, not
    # marginally — a near-tie would be flaky. The heal is fully deterministic
    # (all RNG seeded) and stably converges to ~0.70× the plain-merged MSE on
    # this tiny model; 0.85 leaves ample headroom yet still rejects a near-no-op
    # heal (ratio -> 1.0).
    assert state["holdout_mse"] < 0.85 * state["plain_merged_holdout_mse"], (
        f"heal barely improved: holdout {state['holdout_mse']:.6e} vs "
        f"plain-merged {state['plain_merged_holdout_mse']:.6e}"
    )
    assert state["accepted"] is True
    after = build_banks(ref)["gate_proj"].get(0).detach()
    assert not torch.equal(before, after), "merged centroid weights did not change"


def test_heal_layer_all_experts_trainable():
    """Every kept expert trains — including singletons. (The old SH design
    wrongly froze singletons.) Position 1 (id 2) is a singleton kept expert;
    its weights must move when the heal is accepted."""
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[1]
    batches = _id_batches(model, n_seq=64)  # ~512 tokens -> ~51 holdout
    cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU, pool_size=10_000)
    _merge_layer(ref, _KEPT)
    singleton_before = build_banks(ref)["down_proj"].get(1).detach().clone()

    state = _heal_layer(
        layer_ref=ref, final_kept_ids=_KEPT,
        captured_input=cap_in, captured_target=cap_out,
        heal_cfg=_heal_cfg(), device=_CPU,
    )
    assert state["holdout_mse"] < 0.85 * state["plain_merged_holdout_mse"]
    assert state["accepted"] is True
    singleton_after = build_banks(ref)["down_proj"].get(1).detach()
    assert not torch.equal(singleton_before, singleton_after), (
        "singleton kept expert was frozen — all kept experts must train"
    )


def test_heal_layer_accept_reject_revert():
    """Monotone-safe guard: when the heal cannot beat the plain merge the layer
    is REVERTED. Capturing the target AFTER the merge makes the plain-merged
    output the target — the heal starts at ~0 MSE and (with a large lr) can
    only move away → rejected, banks + router byte-identical to the merge."""
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[1]
    # Default n_seq=16 pool is fine here: the reject path starts at the bf16
    # noise floor, so the verdict does not depend on pool size — no need for
    # the n_seq=64 pool the accept tests use.
    batches = _id_batches(model)
    _merge_layer(ref, _KEPT)
    # Target == the plain-merged layer's own output.
    cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU, pool_size=10_000)

    banks = build_banks(ref)
    snap = {n: [banks[n].get(p).detach().clone() for p in range(len(_KEPT))]
            for n in MATRIX_NAMES}
    router_snap = ref.router.weight.detach().clone()

    state = _heal_layer(
        layer_ref=ref, final_kept_ids=_KEPT,
        captured_input=cap_in, captured_target=cap_out,
        heal_cfg=_heal_cfg(merge_heal_lr=1.0e-2, merge_heal_max_steps=60,
                           merge_heal_eval_interval=10, merge_heal_patience=3),
        device=_CPU,
    )
    assert state["accepted"] is False
    # Guard against a degenerate _heal_layer that returned accepted=False
    # without ever training: training ran, and the rejection came from
    # patience exhaustion (held-out MSE never beat the plain merge).
    assert state["steps"] > 0
    assert state["stop_reason"] == "patience"
    banks2 = build_banks(ref)
    for n in MATRIX_NAMES:
        for p in range(len(_KEPT)):
            assert torch.equal(banks2[n].get(p), snap[n][p]), (
                f"rejected heal left {n}[{p}] modified — must revert to plain merge"
            )
    assert torch.equal(ref.router.weight.detach(), router_snap)


def test_heal_layer_router_flag():
    """`merge_heal_train_router` gates router training: the resized router
    changes iff the flag is True. Experts always train regardless — so when
    the router is frozen the kept experts must STILL move."""
    for train_router in (True, False):
        model = _tiny_moe_model()
        ref = list(iter_moe_layers(model))[1]
        batches = _id_batches(model, n_seq=64)  # ~512 tokens -> ~51 holdout
        cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU,
                                          pool_size=10_000)
        _merge_layer(ref, _KEPT)
        router_after_resize = ref.router.weight.detach().clone()
        experts_before = build_banks(ref)["gate_proj"].get(0).detach().clone()

        state = _heal_layer(
            layer_ref=ref, final_kept_ids=_KEPT,
            captured_input=cap_in, captured_target=cap_out,
            heal_cfg=_heal_cfg(merge_heal_train_router=train_router),
            device=_CPU,
        )
        assert state["holdout_mse"] < 0.85 * state["plain_merged_holdout_mse"]
        assert state["accepted"] is True

        router_now = ref.router.weight.detach()
        router_changed = not torch.equal(router_now, router_after_resize)
        assert router_changed is train_router, (
            f"train_router={train_router} but router_changed={router_changed}"
        )
        if train_router:
            # A trained router must move by a MEANINGFUL magnitude, not just
            # a numerically-noisy `not equal`.
            delta = (router_now - router_after_resize).abs().max().item()
            assert delta > 1e-4, f"router barely moved: max |Δ| {delta}"
        else:
            # F14/L5: router frozen — but experts still all train, so the
            # kept-expert banks must have changed.
            experts_after = build_banks(ref)["gate_proj"].get(0).detach()
            assert not torch.equal(experts_before, experts_after), (
                "router frozen but experts did not heal — all kept experts "
                "must train regardless of merge_heal_train_router"
            )


# ---------------------------------------------------------------------------
# _write_heal_weights / _load_heal_weights — checkpoint round-trip
# ---------------------------------------------------------------------------


def test_heal_weights_roundtrip_all_experts(tmp_path):
    """The per-layer healed-weight checkpoint round-trips the genuinely
    POST-HEAL weights of EVERY kept expert (merged + singleton) and the
    router, under format_version 2.

    The checkpoint is written AFTER `_heal_layer` runs (with a pre-merge
    target so the heal is accepted and the weights genuinely move) — writing
    immediately after `_merge_layer` would only round-trip plain-merged
    weights and silently pass even if the heal output were dropped.
    """
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[1]
    batches = _id_batches(model, n_seq=64)

    # Capture BEFORE merge so the heal has a reachable target and is accepted.
    cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU,
                                      pool_size=10_000)
    _merge_layer(ref, _KEPT)
    plain_merged = build_banks(ref)["gate_proj"].get(0).detach().clone()

    state = _heal_layer(
        layer_ref=ref, final_kept_ids=_KEPT,
        captured_input=cap_in, captured_target=cap_out,
        heal_cfg=_heal_cfg(), device=_CPU,
    )
    assert state["accepted"] is True, "heal must be accepted for this test"

    # Snapshot the POST-HEAL banks + router — these moved off the plain merge.
    banks = build_banks(ref)
    expected = {n: [banks[n].get(p).detach().clone() for p in range(len(_KEPT))]
                for n in MATRIX_NAMES}
    router_expected = ref.router.weight.detach().clone()
    assert not torch.equal(expected["gate_proj"][0], plain_merged), (
        "heal did not move the weights — round-trip would be vacuous"
    )

    _write_heal_weights(tmp_path, ref, _KEPT, accepted=True)
    pt = tmp_path / f"_heal_weights_layer_{ref.layer_idx}.pt"
    assert pt.exists()
    payload = torch.load(pt, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 2
    assert payload["accepted"] is True
    assert len(payload["healed_experts"]) == len(_KEPT)  # ALL kept experts

    # Perturb the banks, then reload — every kept expert must be restored to
    # its POST-HEAL state.
    banks2 = build_banks(ref)
    with torch.no_grad():
        for n in MATRIX_NAMES:
            for p in range(len(_KEPT)):
                banks2[n].set(p, torch.zeros_like(banks2[n].get(p)))
    _load_heal_weights(tmp_path, ref, _KEPT)

    banks3 = build_banks(ref)
    for n in MATRIX_NAMES:
        for p in range(len(_KEPT)):
            assert torch.allclose(banks3[n].get(p), expected[n][p], atol=1e-5), (
                f"{n}[{p}] did not round-trip"
            )
    assert torch.allclose(ref.router.weight.detach(), router_expected, atol=1e-5)

    # F14: the `accepted` payload field round-trips False as well as True.
    _write_heal_weights(tmp_path, ref, _KEPT, accepted=False)
    payload_rej = torch.load(pt, map_location="cpu", weights_only=True)
    assert payload_rej["accepted"] is False
