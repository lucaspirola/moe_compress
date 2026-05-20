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
    _heal_lr_at_step,
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
# LR schedule — _HealConfig knobs, the helper, and end-to-end wiring
# ---------------------------------------------------------------------------


def test_heal_config_lr_schedule_defaults_are_inert():
    """With only `merge_heal_enabled: true` the three LR-schedule knobs default
    to inert values that reproduce the constant-LR Adam path: warmup=0, decay=0,
    lr_min == lr."""
    cfg = _HealConfig({"merge_heal_enabled": True})
    assert cfg.lr_warmup_steps == 0
    assert cfg.lr_decay_steps == 0
    assert cfg.lr_min == cfg.lr


def test_heal_config_rejects_bad_lr_min():
    """`lr_min <= 0` and `lr_min > lr` both raise — out of (0, lr] range."""
    with pytest.raises(ValueError, match="lr_min"):
        _HealConfig({
            "merge_heal_enabled": True, "merge_heal_lr": 1.0e-4,
            "merge_heal_lr_min": 0.0,
        })
    with pytest.raises(ValueError, match="lr_min"):
        _HealConfig({
            "merge_heal_enabled": True, "merge_heal_lr": 1.0e-4,
            "merge_heal_lr_min": 1.0e-3,  # > lr
        })


def test_heal_config_rejects_bad_decay_steps():
    """Negative counts raise; warmup or warmup+decay exceeding max_steps raises;
    a cosine with `lr_min == lr` (no-op) raises."""
    # Negative warmup.
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        _HealConfig({
            "merge_heal_enabled": True, "merge_heal_lr_warmup_steps": -1,
        })
    # Negative decay.
    with pytest.raises(ValueError, match="lr_decay_steps"):
        _HealConfig({
            "merge_heal_enabled": True, "merge_heal_lr_decay_steps": -1,
        })
    # Warmup >= max_steps (warmup never completes).
    with pytest.raises(ValueError, match="warmup would never complete"):
        _HealConfig({
            "merge_heal_enabled": True,
            "merge_heal_lr_warmup_steps": 2000, "merge_heal_max_steps": 2000,
        })
    # Warmup + decay > max_steps (cosine cannot reach lr_min).
    with pytest.raises(ValueError, match="cosine schedule cannot reach lr_min"):
        _HealConfig({
            "merge_heal_enabled": True,
            "merge_heal_lr_warmup_steps": 200,
            "merge_heal_lr_decay_steps": 2000,
            "merge_heal_lr_min": 1.0e-5,
            "merge_heal_max_steps": 2000,
        })
    # decay > 0 with lr_min == lr: cosine would be a no-op.
    with pytest.raises(ValueError, match="cosine.*would be a no-op"):
        _HealConfig({
            "merge_heal_enabled": True, "merge_heal_lr_decay_steps": 100,
        })
    # lr_min < lr with decay_steps == 0: asymptote unreachable.
    with pytest.raises(ValueError, match="asymptote is unreachable"):
        _HealConfig({
            "merge_heal_enabled": True, "merge_heal_lr": 1.0e-4,
            "merge_heal_lr_min": 1.0e-5,
            "merge_heal_lr_decay_steps": 0,
        })


def test_lr_schedule_warmup_then_cosine_then_floor():
    """Pure-math test of `_heal_lr_at_step` — sample LRs at every phase boundary
    and at one interior point of each phase; assert linear ramp, cosine endpoint
    equalities, monotone decrease across the cosine, and held floor past end."""
    lr, lr_min, warmup, decay = 1.0e-4, 1.0e-5, 100, 1000
    kw = dict(lr=lr, lr_min=lr_min, warmup_steps=warmup, decay_steps=decay)

    # Defensive: step < 0 raises.
    with pytest.raises(ValueError, match="step="):
        _heal_lr_at_step(-1, **kw)

    # Linear warmup: step s takes lr * (s+1)/warmup.
    assert _heal_lr_at_step(0, **kw) == pytest.approx(lr / warmup)         # first ramp step
    assert _heal_lr_at_step(warmup // 2, **kw) == pytest.approx(
        lr * (warmup // 2 + 1) / warmup)
    assert _heal_lr_at_step(warmup - 1, **kw) == pytest.approx(lr)         # last ramp step

    # Cosine first step (t=0) is continuous with warmup endpoint.
    assert _heal_lr_at_step(warmup, **kw) == pytest.approx(lr)
    # Mid-cosine (t = decay/2): cos(π/2) = 0, cos_term = 0.5.
    assert _heal_lr_at_step(warmup + decay // 2, **kw) == pytest.approx(
        lr_min + (lr - lr_min) * 0.5, rel=1e-9)
    # Cosine endpoint (step = warmup+decay) lands at lr_min via the flat branch.
    assert _heal_lr_at_step(warmup + decay, **kw) == pytest.approx(lr_min)
    # Held floor past the end.
    assert _heal_lr_at_step(warmup + decay + 9999, **kw) == pytest.approx(lr_min)

    # Cosine is monotone non-increasing on its interval.
    cosine_vals = [
        _heal_lr_at_step(warmup + t, **kw)
        for t in range(0, decay + 1, 50)
    ]
    for a, b in zip(cosine_vals, cosine_vals[1:]):
        assert b <= a + 1e-15, f"cosine not monotone: {a} → {b}"

    # Inert defaults: warmup=0, decay=0, lr_min==lr ⇒ constant lr at every step.
    for s in (0, 1, 100, 99999):
        assert _heal_lr_at_step(
            s, lr=lr, lr_min=lr, warmup_steps=0, decay_steps=0
        ) == pytest.approx(lr)


def test_heal_config_cross_domain_holdout_inert_by_default():
    """With only `merge_heal_enabled: true` the cross-domain holdout is OFF,
    so Stage 2 will skip the WikiText capture entirely — disabled-default
    contract."""
    cfg = _HealConfig({"merge_heal_enabled": True})
    assert cfg.cross_domain_holdout_enabled is False
    # The token count still has a sensible default (only consulted when on).
    assert cfg.xd_holdout_tokens >= 1


def test_heal_config_rejects_bad_xd_holdout_tokens():
    """When the cross-domain holdout is enabled, `xd_holdout_tokens < 1` is
    rejected so a malformed YAML can't silently disable the capture."""
    with pytest.raises(ValueError, match="xd_holdout_tokens"):
        _HealConfig({
            "merge_heal_enabled": True,
            "merge_heal_cross_domain_holdout_enabled": True,
            "merge_heal_xd_holdout_tokens": 0,
        })


def test_heal_layer_with_xd_holdout_reports_telemetry_and_leaves_decision_unchanged():
    """When `_heal_layer` receives `xd_input`/`xd_target`, the returned state
    dict carries the two cross-domain numbers — and accept/reject is keyed on
    the SAME Nemotron metric the no-xd path uses, i.e. the verdict is identical
    whether xd is fed in or not. (Determinism note: every RNG used by
    `_heal_layer` is seeded by `layer_idx`, so a back-to-back invocation on
    the same model state is byte-deterministic.)"""
    # Build the same scenario the existing accept tests use, then run the heal
    # twice: once without xd, once with a hand-crafted xd pool (random tokens
    # the heal has never seen). The xd path must:
    #   1. Populate `holdout_mse_xd` and `plain_merged_holdout_mse_xd` as
    #      finite floats (no NaN passthrough when xd is supplied).
    #   2. Leave `holdout_mse`, `accepted`, and `steps` byte-equal to the
    #      no-xd run.
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[1]
    batches = _id_batches(model, n_seq=64)  # the accept-margin pool
    cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU, pool_size=10_000)

    # A second model freshly seeded the same way and merged identically — the
    # heal must see the same starting weights on both runs, so we don't reuse
    # the same `ref` (which gets mutated by _merge_layer + _heal_layer).
    model_a = _tiny_moe_model()
    ref_a = list(iter_moe_layers(model_a))[1]
    _merge_layer(ref_a, _KEPT)
    no_xd = _heal_layer(
        layer_ref=ref_a, final_kept_ids=_KEPT,
        captured_input=cap_in, captured_target=cap_out,
        heal_cfg=_heal_cfg(), device=_CPU,
    )
    # `holdout_mse_xd` keys must exist in the no-xd return shape, NaN-valued,
    # so downstream consumers can rely on the shape regardless of toggle.
    assert "holdout_mse_xd" in no_xd
    assert "plain_merged_holdout_mse_xd" in no_xd
    import math as _math
    assert _math.isnan(no_xd["holdout_mse_xd"])
    assert _math.isnan(no_xd["plain_merged_holdout_mse_xd"])

    # Build a synthetic xd pool: half the rows of cap_in, drawn from the tail
    # so they are different rows from anything the heal will train on. Using
    # the same `cap_out` rows as targets is fine — the test only checks that
    # the MSE *computation* works and that xd is plumbed through; it does NOT
    # check cross-corpus interpretability (that's what the H200 run is for).
    n = cap_in.shape[0]
    xd_in = cap_in[n // 2:].clone()
    xd_out = cap_out[n // 2:].clone()

    model_b = _tiny_moe_model()
    ref_b = list(iter_moe_layers(model_b))[1]
    _merge_layer(ref_b, _KEPT)
    with_xd = _heal_layer(
        layer_ref=ref_b, final_kept_ids=_KEPT,
        captured_input=cap_in, captured_target=cap_out,
        xd_input=xd_in, xd_target=xd_out,
        heal_cfg=_heal_cfg(), device=_CPU,
    )
    # xd telemetry populated as finite floats.
    assert _math.isfinite(with_xd["holdout_mse_xd"])
    assert _math.isfinite(with_xd["plain_merged_holdout_mse_xd"])
    assert with_xd["holdout_mse_xd"] >= 0.0
    assert with_xd["plain_merged_holdout_mse_xd"] >= 0.0
    # Accept/reject + step count + every Nemotron-derived metric is unaffected
    # by the added telemetry — the xd path is read-only. Assert byte-equality
    # (not approx) because both runs see identical seeded model state, identical
    # captured tensors, and identical optimiser trajectories; pytest's `==`
    # failure message is informative enough on a leak.
    assert with_xd["accepted"] == no_xd["accepted"]
    assert with_xd["steps"] == no_xd["steps"]
    assert with_xd["stop_reason"] == no_xd["stop_reason"]
    assert with_xd["holdout_mse"] == no_xd["holdout_mse"]
    assert with_xd["plain_merged_holdout_mse"] == no_xd["plain_merged_holdout_mse"]
    assert with_xd["heal_gap"] == no_xd["heal_gap"]
    assert with_xd["train_mse"] == no_xd["train_mse"]
    assert with_xd["train_mse_at_best"] == no_xd["train_mse_at_best"]


def test_heal_layer_with_xd_holdout_raises_on_misaligned_pools():
    """A shape mismatch between `xd_input` and `xd_target` indicates a token
    alignment bug at the capture site — fail fast, don't silently produce a
    bogus MSE."""
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[1]
    batches = _id_batches(model, n_seq=16)
    cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU, pool_size=10_000)
    _merge_layer(ref, _KEPT)
    # Same first-axis size but a deliberately-broken second axis: simulates a
    # capture-site bug where the input pool's hidden dim diverges from the
    # target pool's.
    bad_xd_in = cap_in.clone()
    bad_xd_out = cap_out[:, :-1].clone()  # one fewer hidden dim
    # The XD shape-mismatch site raises ValueError (Nemotron mismatch raises
    # RuntimeError); pin to "cross-domain" so a future refactor that
    # converted the Nemotron message to ValueError can't make this test
    # silently match the wrong site.
    with pytest.raises(ValueError, match="cross-domain.*token alignment"):
        _heal_layer(
            layer_ref=ref, final_kept_ids=_KEPT,
            captured_input=cap_in, captured_target=cap_out,
            xd_input=bad_xd_in, xd_target=bad_xd_out,
            heal_cfg=_heal_cfg(), device=_CPU,
        )


def test_heal_layer_applies_lr_schedule(monkeypatch):
    """`_heal_layer` mutates `opt.param_groups[0]['lr']` each step per the
    schedule. Record the LR seen by `AdamW.step` over a short heal and assert
    the sequence matches `_heal_lr_at_step` exactly — proves the schedule is
    actually wired into the optimiser, not just computed and discarded."""
    model = _tiny_moe_model()
    ref = list(iter_moe_layers(model))[1]
    batches = _id_batches(model, n_seq=16)
    cap_in, cap_out = _capture_mlp_io(model, ref, batches, device=_CPU, pool_size=10_000)
    _merge_layer(ref, _KEPT)

    lr_peak, lr_min = 1.0e-3, 1.0e-4
    warmup, decay = 4, 20
    cfg = _heal_cfg(
        merge_heal_lr=lr_peak,
        merge_heal_lr_warmup_steps=warmup,
        merge_heal_lr_min=lr_min,
        merge_heal_lr_decay_steps=decay,
        # Cap right at end-of-cosine so we exercise warmup + cosine + final-step,
        # while staying under the validator's `warmup+decay <= max_steps` rule.
        merge_heal_max_steps=warmup + decay,
        merge_heal_eval_interval=2,
        # Disable patience as a stopping mechanism — we want the full schedule.
        merge_heal_patience=10_000,
    )

    recorded: list[float] = []
    orig_step = torch.optim.AdamW.step

    def record_step(self, *args, **kwargs):
        recorded.append(self.param_groups[0]["lr"])
        return orig_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)

    _heal_layer(
        layer_ref=ref, final_kept_ids=_KEPT,
        captured_input=cap_in, captured_target=cap_out,
        heal_cfg=cfg, device=_CPU,
    )

    assert len(recorded) == warmup + decay, (
        f"expected {warmup + decay} optimiser steps, got {len(recorded)}"
    )
    expected = [
        _heal_lr_at_step(s, lr=lr_peak, lr_min=lr_min,
                         warmup_steps=warmup, decay_steps=decay)
        for s in range(len(recorded))
    ]
    assert recorded == expected, (
        f"applied LR sequence diverged from schedule.\n"
        f"  recorded[:6] = {recorded[:6]}\n  expected[:6] = {expected[:6]}"
    )


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
