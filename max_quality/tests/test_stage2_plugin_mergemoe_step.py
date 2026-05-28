"""Plugin #9 / S2_MM — MergeMoE T₁=Q·P† merge step.

Pins:
  * The MergeMoE closed-form math on a small hand-checkable case.
  * The byte-identical guarantee of ``merge_step="freq_weighted"`` (the default).
  * The ``merge_step="mergemoe"`` branch routes through the new path.
  * Defensive fallbacks: empty/None ``layer_inputs`` and rank-deficient ``P``.
  * Orchestrator config validation.

See ``tasks/PLAN_PLUGIN_09_s2_mm.md`` for the spec.
"""
from __future__ import annotations

import copy
import logging
import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from moe_compress.stage2.merging import _merge_experts_inplace
from moe_compress.stage2.mergemoe import (
    _COND_THRESHOLD,
    _mergemoe_compute_merged_down,
    _swiglu_intermediate,
)
from moe_compress.utils.activation_hooks import ReamCostAccumulator
from moe_compress.utils.model_io import build_banks, iter_moe_layers


# ---------------------------------------------------------------------------
# Section 1 — _mergemoe_compute_merged_down: hand-checked closed-form
# ---------------------------------------------------------------------------


def _hand_compute_mergemoe_down(
    W_G: list[torch.Tensor],
    W_U: list[torch.Tensor],
    W_D: list[torch.Tensor],
    b: list[float],
    X_hat: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of the MergeMoE T₁=Q·P† closed-form.

    Mirrors the paper's notation exactly (paper Eqs. 3–6). Used only by the
    hand-checked test below — independent of the production helper so that a
    bug in either path surfaces as a mismatch.
    """
    N = len(W_G)
    # Merged gate/up: freq-weighted average (T₂/T₃ collapse, Eq. 4).
    W_G_merged = sum(b[j] * W_G[j] for j in range(N))
    W_U_merged = sum(b[j] * W_U[j] for j in range(N))

    # P = σ(W_G_merged·X̂) ⊙ (W_U_merged·X̂)   shape (T, d_int)
    gate_p = F.linear(X_hat, W_G_merged)
    up_p   = F.linear(X_hat, W_U_merged)
    P = F.silu(gate_p) * up_p

    # Q (paper convention shape (N·d_int, T)). We compute Qᵀ (T, N·d_int)
    # because PyTorch lstsq is row-major.
    Q_cols = []
    for j in range(N):
        gate_q = F.linear(X_hat, W_G[j])
        up_q   = F.linear(X_hat, W_U[j])
        Q_cols.append(F.silu(gate_q) * up_q)  # (T, d_int)
    Q_T = torch.cat(Q_cols, dim=1)  # (T, N·d_int)

    # Solve P · X = Q_T  for  X = P† · Q_T  shape (d_int, N·d_int).
    # Paper's T₁ is the transpose: shape (N·d_int, d_int).
    X = torch.linalg.lstsq(P, Q_T, driver="gelsd").solution
    T1 = X.transpose(0, 1).contiguous()  # (N·d_int, d_int)

    # W_D^merged = Σ_j b_j · W_D^j · T₁_block_j
    d_hidden, d_int = W_D[0].shape
    T1_blocks = T1.view(N, d_int, d_int)
    merged = torch.zeros(d_hidden, d_int, dtype=W_D[0].dtype, device=W_D[0].device)
    for j in range(N):
        merged = merged + b[j] * (W_D[j] @ T1_blocks[j])
    return merged


def test_mergemoe_closed_form_matches_hand_computation():
    """T₁=Q·P† closed-form: production helper matches an independent hand
    computation on a small case.

    Two experts, d_hidden=4, d_int=3, T=8 calibration tokens. Frequency
    weights b = (0.75, 0.25). Sub-sample disabled (T_full == token_cap),
    so the deterministic random sub-sample is a no-op and the two paths
    consume the exact same X̂.
    """
    torch.manual_seed(0)
    d_hidden, d_int, T = 4, 3, 8
    b = [0.75, 0.25]

    W_G = [torch.randn(d_int, d_hidden) for _ in range(2)]
    W_U = [torch.randn(d_int, d_hidden) for _ in range(2)]
    W_D = [torch.randn(d_hidden, d_int) for _ in range(2)]
    X_hat = torch.randn(T, d_hidden)

    expected = _hand_compute_mergemoe_down(W_G, W_U, W_D, b, X_hat)
    actual = _mergemoe_compute_merged_down(
        member_gates=W_G, member_ups=W_U, member_downs=W_D,
        weights=b, layer_inputs=X_hat, token_cap=T, seed=0,
    )
    # The two implementations should match to fp32 precision; both solve the
    # same lstsq via the same driver ("gelsd").
    assert actual.shape == (d_hidden, d_int)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_mergemoe_with_more_experts():
    """Closed-form matches hand computation with N=3 cluster members."""
    torch.manual_seed(1)
    d_hidden, d_int, T = 6, 4, 16
    b = [0.5, 0.3, 0.2]
    W_G = [torch.randn(d_int, d_hidden) for _ in range(3)]
    W_U = [torch.randn(d_int, d_hidden) for _ in range(3)]
    W_D = [torch.randn(d_hidden, d_int) for _ in range(3)]
    X_hat = torch.randn(T, d_hidden)

    expected = _hand_compute_mergemoe_down(W_G, W_U, W_D, b, X_hat)
    actual = _mergemoe_compute_merged_down(
        member_gates=W_G, member_ups=W_U, member_downs=W_D,
        weights=b, layer_inputs=X_hat, token_cap=T, seed=1,
    )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_mergemoe_dtype_preserved():
    """Helper returns a tensor in the dtype of ``member_downs[0]``."""
    torch.manual_seed(2)
    d_hidden, d_int, T = 4, 3, 8
    W_G = [torch.randn(d_int, d_hidden, dtype=torch.float32) for _ in range(2)]
    W_U = [torch.randn(d_int, d_hidden, dtype=torch.float32) for _ in range(2)]
    W_D = [torch.randn(d_hidden, d_int, dtype=torch.float32) for _ in range(2)]
    X_hat = torch.randn(T, d_hidden, dtype=torch.float32)
    out = _mergemoe_compute_merged_down(
        member_gates=W_G, member_ups=W_U, member_downs=W_D,
        weights=[0.5, 0.5], layer_inputs=X_hat, token_cap=T, seed=0,
    )
    assert out.dtype == torch.float32


def test_mergemoe_swiglu_intermediate_matches_manual():
    """Local SwiGLU helper produces σ(gate·x) ⊙ (up·x)."""
    torch.manual_seed(3)
    d_hidden, d_int, T = 4, 3, 5
    W_g = torch.randn(d_int, d_hidden)
    W_u = torch.randn(d_int, d_hidden)
    x = torch.randn(T, d_hidden)
    expected = F.silu(F.linear(x, W_g)) * F.linear(x, W_u)
    actual = _swiglu_intermediate(W_g, W_u, x)
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Section 2 — _merge_experts_inplace: byte-identical default + opt-in branch
# ---------------------------------------------------------------------------


def _snapshot_banks(layer_ref) -> dict[str, torch.Tensor]:
    """Snapshot all expert weights into independent tensors."""
    banks = build_banks(layer_ref)
    return {
        name: bank._stacked().detach().clone()
        for name, bank in banks.items()
    }


def test_merge_experts_inplace_default_is_byte_identical(tiny_model):
    """``merge_step`` unset and ``merge_step="freq_weighted"`` produce
    byte-identical merged weights — the legacy code path is unchanged.

    Snapshot the merged centroid weights after the default call; reset the
    model from a deepcopy and re-merge with the explicit ``freq_weighted``
    keyword; assert byte-identical tensors (no ``atol``).
    """
    model_a = copy.deepcopy(tiny_model)
    model_b = copy.deepcopy(tiny_model)

    layer_a = next(iter_moe_layers(model_a))
    layer_b = next(iter_moe_layers(model_b))
    grouped = {0: [0, 1, 2]}
    freq = {0: 4, 1: 2, 2: 1}

    # Default call — no merge_step kwarg.
    _merge_experts_inplace(layer_a, grouped, freq, freq_weighted=True)
    snap_a = _snapshot_banks(layer_a)

    # Explicit freq_weighted.
    _merge_experts_inplace(
        layer_b, grouped, freq, freq_weighted=True, merge_step="freq_weighted",
    )
    snap_b = _snapshot_banks(layer_b)

    for name in snap_a:
        assert torch.equal(snap_a[name], snap_b[name]), (
            f"{name}: default and merge_step='freq_weighted' diverged "
            "— the legacy path is no longer byte-identical."
        )


def test_merge_experts_inplace_mergemoe_routes_through_new_path(tiny_model):
    """``merge_step="mergemoe"`` actually invokes the closed-form solve.

    Gate/up should match the freq-weighted result (algebraically identical
    by paper Eq. 4). Down should differ (otherwise the new branch did not
    fire). The result must be finite.
    """
    model_freq = copy.deepcopy(tiny_model)
    model_mm = copy.deepcopy(tiny_model)

    layer_freq = next(iter_moe_layers(model_freq))
    layer_mm = next(iter_moe_layers(model_mm))
    d_hidden = layer_mm.experts_module.hidden_dim
    grouped = {0: [0, 1, 2]}
    freq = {0: 3, 1: 2, 2: 1}

    _merge_experts_inplace(layer_freq, grouped, freq, freq_weighted=True)
    snap_freq = _snapshot_banks(layer_freq)

    torch.manual_seed(7)
    X_hat = torch.randn(32, d_hidden)
    _merge_experts_inplace(
        layer_mm, grouped, freq, freq_weighted=True,
        merge_step="mergemoe", layer_inputs=X_hat, token_cap=32,
    )
    snap_mm = _snapshot_banks(layer_mm)

    # gate / up unchanged.
    assert torch.equal(snap_mm["gate_proj"][0], snap_freq["gate_proj"][0]), (
        "MergeMoE merged gate diverged from freq-weighted gate — these must "
        "be algebraically identical (paper Eq. 4 T₂ = freq-weights)."
    )
    assert torch.equal(snap_mm["up_proj"][0], snap_freq["up_proj"][0]), (
        "MergeMoE merged up diverged from freq-weighted up — these must be "
        "algebraically identical (paper Eq. 4 T₃ = freq-weights)."
    )
    # Down differs (the new path fired) and is finite.
    assert not torch.equal(snap_mm["down_proj"][0], snap_freq["down_proj"][0]), (
        "MergeMoE merged down did not diverge from freq-weighted — the new "
        "branch is silently a no-op."
    )
    assert torch.isfinite(snap_mm["down_proj"][0]).all()


def test_merge_experts_inplace_mergemoe_fallback_on_empty_layer_inputs(
    tiny_model, capsys,
):
    """``merge_step="mergemoe"`` with ``layer_inputs=None`` falls back to
    freq-weighted with a WARNING.
    """
    model_freq = copy.deepcopy(tiny_model)
    model_mm = copy.deepcopy(tiny_model)
    layer_freq = next(iter_moe_layers(model_freq))
    layer_mm = next(iter_moe_layers(model_mm))
    grouped = {0: [0, 1]}
    freq = {0: 2, 1: 1}

    _merge_experts_inplace(layer_freq, grouped, freq, freq_weighted=True)

    # Mirror of 9330988: the moe_compress.stage2.merging logger's handler chain
    # bypasses pytest's `caplog` under full-suite imports, but the warning
    # reaches stderr via the default StreamHandler. Assert on captured stderr.
    _merge_experts_inplace(
        layer_mm, grouped, freq, freq_weighted=True,
        merge_step="mergemoe", layer_inputs=None,
    )

    captured = capsys.readouterr()
    assert (
        "falling back to freq-weighted merge" in captured.err
    ), (
        f"expected a WARNING about the freq-weighted fallback in captured "
        f"stderr, got: {captured.err!r}"
    )

    # Merged weights should match the freq-weighted result.
    snap_freq = _snapshot_banks(layer_freq)
    snap_mm = _snapshot_banks(layer_mm)
    for name in snap_freq:
        assert torch.equal(snap_freq[name], snap_mm[name])


def test_merge_experts_inplace_invalid_merge_step_raises(tiny_model):
    """Unknown ``merge_step`` value raises ValueError with the legal values."""
    layer = next(iter_moe_layers(tiny_model))
    with pytest.raises(ValueError, match="freq_weighted.*mergemoe"):
        _merge_experts_inplace(
            layer, {0: [0, 1]}, {0: 1, 1: 1},
            freq_weighted=True, merge_step="banana",
        )


# ---------------------------------------------------------------------------
# Section 3 — Conditioning fallback
# ---------------------------------------------------------------------------


def test_mergemoe_cond_threshold_fallback_warns_and_falls_back(capsys):
    """When cond(P) > 1e8 the helper falls back to freq-weighted down + WARNs."""
    torch.manual_seed(4)
    d_hidden, d_int = 4, 3
    W_G = [torch.randn(d_int, d_hidden) for _ in range(2)]
    W_U = [torch.randn(d_int, d_hidden) for _ in range(2)]
    W_D = [torch.randn(d_hidden, d_int) for _ in range(2)]
    b = [0.6, 0.4]
    # Rank-1 calibration: every token is the same vector → P has rank ≤ 1,
    # cond(P) is essentially infinite. Stack 8 copies of one row.
    x1 = torch.randn(1, d_hidden)
    X_hat = x1.repeat(8, 1)

    # Ensure the moe_compress.stage2.mergemoe logger reaches stderr.
    # Some full-suite imports configure loggers to bypass `caplog`; assert
    # against the captured stderr instead (where log.warning ends up via the
    # default StreamHandler chain).
    out = _mergemoe_compute_merged_down(
        member_gates=W_G, member_ups=W_U, member_downs=W_D,
        weights=b, layer_inputs=X_hat, token_cap=8, seed=0,
    )

    captured = capsys.readouterr()
    assert (
        "cond(P)=" in captured.err and "D-mergemoe-cond-fallback" in captured.err
    ), (
        f"expected cond-fallback WARNING in captured stderr, got: {captured.err!r}"
    )

    # Falls back to freq-weighted down.
    expected_freq = b[0] * W_D[0] + b[1] * W_D[1]
    assert torch.allclose(out, expected_freq, atol=1e-6)


def test_mergemoe_cond_threshold_constant_matches_plan():
    """Pin: ``_COND_THRESHOLD`` matches the comprehensive-plan §581 spec.

    The risk-mitigation table in SC_STAGE12_COMPREHENSIVE_PLAN.md §581
    pins the per-cluster ``cond(P)`` threshold at ``1e8``. Hard-coding the
    expected value here guards against a silent threshold change.
    """
    assert _COND_THRESHOLD == 1e8


def test_mergemoe_requires_at_least_two_members():
    """N<2 raises ValueError; caller must filter singletons."""
    W_G = [torch.randn(3, 4)]
    W_U = [torch.randn(3, 4)]
    W_D = [torch.randn(4, 3)]
    X_hat = torch.randn(8, 4)
    with pytest.raises(ValueError, match="N=1"):
        _mergemoe_compute_merged_down(
            member_gates=W_G, member_ups=W_U, member_downs=W_D,
            weights=[1.0], layer_inputs=X_hat, token_cap=8, seed=0,
        )


def test_mergemoe_mismatched_lengths_raise():
    """member_gates/ups/downs/weights must agree in length."""
    W_G = [torch.randn(3, 4), torch.randn(3, 4)]
    W_U = [torch.randn(3, 4)]  # wrong length
    W_D = [torch.randn(4, 3), torch.randn(4, 3)]
    X_hat = torch.randn(8, 4)
    with pytest.raises(ValueError, match="same length"):
        _mergemoe_compute_merged_down(
            member_gates=W_G, member_ups=W_U, member_downs=W_D,
            weights=[0.5, 0.5], layer_inputs=X_hat, token_cap=8, seed=0,
        )


def test_mergemoe_empty_layer_inputs_raises():
    """Helper requires a non-empty calibration buffer."""
    W_G = [torch.randn(3, 4), torch.randn(3, 4)]
    W_U = [torch.randn(3, 4), torch.randn(3, 4)]
    W_D = [torch.randn(4, 3), torch.randn(4, 3)]
    X_hat = torch.empty(0, 4)
    with pytest.raises(ValueError, match="zero tokens"):
        _mergemoe_compute_merged_down(
            member_gates=W_G, member_ups=W_U, member_downs=W_D,
            weights=[0.5, 0.5], layer_inputs=X_hat, token_cap=8, seed=0,
        )


# ---------------------------------------------------------------------------
# Section 4 — _tentative_merged_weights signature compatibility
# ---------------------------------------------------------------------------


def test_tentative_merged_weights_default_still_freq_weighted(tiny_model):
    """``_tentative_merged_weights`` without the new kwargs is byte-identical
    to its legacy callers' expectations.

    Pins that callers in :mod:`stage2.plugins.output_space_cost` (the cost
    matrix loop) which do NOT pass ``merge_step`` continue to receive the
    freq-weighted merged dict — i.e. the cost matrix path is unchanged.
    """
    from moe_compress.stage2.plugins.output_space_cost import _tentative_merged_weights
    from moe_compress.stage2.permutation_align import _PermAlignCache

    layer_ref = next(iter_moe_layers(tiny_model))
    banks = build_banks(layer_ref)
    perm_cache = _PermAlignCache()

    merged_default = _tentative_merged_weights(
        layer_ref, centroid_id=0, child_id=1,
        freq={0: 3, 1: 1}, ream_acc=None, perm_cache=perm_cache,
        banks=banks,
    )
    merged_explicit = _tentative_merged_weights(
        layer_ref, centroid_id=0, child_id=1,
        freq={0: 3, 1: 1}, ream_acc=None, perm_cache=_PermAlignCache(),
        banks=banks,
        merge_step="freq_weighted",
    )
    for name in merged_default:
        assert torch.equal(merged_default[name], merged_explicit[name])


def test_tentative_merged_weights_mergemoe_branch_routes(tiny_model):
    """``merge_step="mergemoe"`` with valid ``layer_inputs`` flips down_proj."""
    from moe_compress.stage2.plugins.output_space_cost import _tentative_merged_weights
    from moe_compress.stage2.permutation_align import _PermAlignCache

    layer_ref = next(iter_moe_layers(tiny_model))
    banks = build_banks(layer_ref)
    d_hidden = layer_ref.experts_module.hidden_dim

    merged_freq = _tentative_merged_weights(
        layer_ref, centroid_id=0, child_id=1,
        freq={0: 3, 1: 1}, ream_acc=None, perm_cache=_PermAlignCache(),
        banks=banks,
    )
    torch.manual_seed(9)
    X_hat = torch.randn(32, d_hidden)
    merged_mm = _tentative_merged_weights(
        layer_ref, centroid_id=0, child_id=1,
        freq={0: 3, 1: 1}, ream_acc=None, perm_cache=_PermAlignCache(),
        banks=banks,
        merge_step="mergemoe", layer_inputs=X_hat, token_cap=32,
    )
    # gate/up unchanged.
    assert torch.equal(merged_mm["gate_proj"], merged_freq["gate_proj"])
    assert torch.equal(merged_mm["up_proj"], merged_freq["up_proj"])
    # down_proj differs (and is finite).
    assert not torch.equal(merged_mm["down_proj"], merged_freq["down_proj"])
    assert torch.isfinite(merged_mm["down_proj"]).all()


# ---------------------------------------------------------------------------
# Section 5 — Orchestrator config validation
# ---------------------------------------------------------------------------


def test_orchestrator_rejects_invalid_merge_step(tiny_config, tiny_model, tmp_path, monkeypatch):
    """``stage2_reap_ream.merge_step`` validates at run() entry."""
    from moe_compress.stage2 import orchestrator as orch

    tiny_config.setdefault("stage2_reap_ream", {})
    tiny_config["stage2_reap_ream"] = {
        "merge_step": "banana",
        "ream": {"frequency_weighted_merge": True},
        "num_calibration_samples": 1,
        "batch_size": 1,
    }
    # Provide the bare-minimum stage1 artifact so config-load doesn't crash
    # before the merge_step check; if validation runs first the test still
    # passes (pytest.raises catches before file IO).
    (tmp_path / "stage1_budgets.json").write_text('{"per_layer_target_experts": {}}')
    (tmp_path / "stage1_blacklist.json").write_text('{"blacklist": {}}')

    with pytest.raises(ValueError, match=r"merge_step=.*freq_weighted.*mergemoe"):
        orch.run(tiny_model, tokenizer=None, config=tiny_config,
                 artifacts_dir=tmp_path)


def test_orchestrator_default_merge_step_is_freq_weighted():
    """Pin: the default of ``stage2_reap_ream.merge_step`` is
    ``"freq_weighted"`` — guards against an inadvertent flip that would
    silently re-route every Stage-2 run through the MergeMoE branch.
    """
    import inspect
    from moe_compress.stage2 import orchestrator as orch
    src = inspect.getsource(orch.run)
    assert 's2.get("merge_step", "freq_weighted")' in src


def test_orchestrator_rejects_sequential_reprofile_with_profile_sidecar(
    tiny_config, tiny_model, tmp_path,
):
    """Plugin #14 audit HIGH-2: hard mutual-exclusion between Plugin #10's
    ``sequential_reprofile`` and Plugin #12's ``profile_sidecar.enabled``.

    REAM §4 stale-stats hazard — the sidecar serves pre-merge state, the
    sequential invalidator clears post-merge state; combining them is a
    silent corruption path. The orchestrator must fail-fast at run() entry.
    """
    from moe_compress.stage2 import orchestrator as orch

    tiny_config["stage2_reap_ream"] = {
        "sequential_reprofile": True,
        "profile_sidecar": {"enabled": True},
        "ream": {"frequency_weighted_merge": True},
        "num_calibration_samples": 1,
        "batch_size": 1,
    }
    (tmp_path / "stage1_budgets.json").write_text('{"per_layer_target_experts": {}}')
    (tmp_path / "stage1_blacklist.json").write_text('{"blacklist": {}}')

    with pytest.raises(ValueError, match=r"sequential_reprofile.*profile_sidecar"):
        orch.run(tiny_model, tokenizer=None, config=tiny_config,
                 artifacts_dir=tmp_path)


def test_orchestrator_allows_sequential_reprofile_alone(tiny_config, tiny_model, tmp_path):
    """Sanity: enabling sequential_reprofile without profile_sidecar is OK
    (does NOT trigger the mutual-exclusion guard). Tests the boundary so a
    future refactor cannot accidentally tighten the guard.
    """
    from moe_compress.stage2 import orchestrator as orch

    tiny_config["stage2_reap_ream"] = {
        "sequential_reprofile": True,
        "profile_sidecar": {"enabled": False},  # explicit False — not raised
        "ream": {"frequency_weighted_merge": True},
        "num_calibration_samples": 1,
        "batch_size": 1,
    }
    (tmp_path / "stage1_budgets.json").write_text('{"per_layer_target_experts": {}}')
    (tmp_path / "stage1_blacklist.json").write_text('{"blacklist": {}}')

    # The mutual-exclusion guard must NOT raise. The run will fail later
    # on a different validation (tiny_config minimal stage1 artifact) — but
    # the failure must NOT be the specific ValueError this guard raises.
    # Negative match is anchored to (ValueError, regex) — substring-only
    # matching would be fragile against any downstream message that happens
    # to mention either knob name for unrelated reasons.
    with pytest.raises(Exception) as exc_info:
        orch.run(tiny_model, tokenizer=None, config=tiny_config,
                 artifacts_dir=tmp_path)
    assert not (
        isinstance(exc_info.value, ValueError)
        and re.search(r"sequential_reprofile.*profile_sidecar", str(exc_info.value))
    )


def test_orchestrator_allows_profile_sidecar_alone(tiny_config, tiny_model, tmp_path):
    """Sanity: enabling profile_sidecar without sequential_reprofile is OK."""
    from moe_compress.stage2 import orchestrator as orch

    tiny_config["stage2_reap_ream"] = {
        "sequential_reprofile": False,
        "profile_sidecar": {"enabled": True},
        "ream": {"frequency_weighted_merge": True},
        "num_calibration_samples": 1,
        "batch_size": 1,
    }
    (tmp_path / "stage1_budgets.json").write_text('{"per_layer_target_experts": {}}')
    (tmp_path / "stage1_blacklist.json").write_text('{"blacklist": {}}')

    with pytest.raises(Exception) as exc_info:
        orch.run(tiny_model, tokenizer=None, config=tiny_config,
                 artifacts_dir=tmp_path)
    assert not (
        isinstance(exc_info.value, ValueError)
        and re.search(r"sequential_reprofile.*profile_sidecar", str(exc_info.value))
    )


def test_orchestrator_default_config_does_not_trigger_mutex(
    tiny_config, tiny_model, tmp_path,
):
    """Pin: the default ``tiny_config`` (no overrides) must NOT trigger the
    sequential_reprofile ↔ profile_sidecar mutual-exclusion guard.

    Symmetrical to ``test_orchestrator_default_merge_step_is_freq_weighted``:
    guards against an inadvertent default flip that would make every Stage-2
    run hit the guard. The run will still fail downstream on tiny_config's
    minimal stage1 artifact, but the failure must not be the mutex guard.
    """
    from moe_compress.stage2 import orchestrator as orch

    (tmp_path / "stage1_budgets.json").write_text('{"per_layer_target_experts": {}}')
    (tmp_path / "stage1_blacklist.json").write_text('{"blacklist": {}}')

    with pytest.raises(Exception) as exc_info:
        orch.run(tiny_model, tokenizer=None, config=tiny_config,
                 artifacts_dir=tmp_path)
    assert not (
        isinstance(exc_info.value, ValueError)
        and re.search(r"sequential_reprofile.*profile_sidecar", str(exc_info.value))
    )


# ---------------------------------------------------------------------------
# Section 6 — D-mergemoe-resume-fallback (Plugin #9 reviewer fix)
# ---------------------------------------------------------------------------


def _patch_stage2_io(monkeypatch):
    """Common helper: stub calibration loader + checkpoint save for the
    resume-path tests below so they never hit HF datasets or real disk
    writes for the compressed checkpoint. Returns nothing — applies the
    monkey-patches on the supplied fixture.
    """
    from moe_compress.stage2 import orchestrator as orch
    from moe_compress.utils import calibration as cal_mod
    from moe_compress.utils import model_io as mio

    def _fake_build(tokenizer, spec, cache_dir=None):
        torch.manual_seed(spec.seed)
        return torch.randint(0, 32, (spec.num_sequences, spec.sequence_length),
                             dtype=torch.long)

    def _fake_slice(tokenizer, spec, num_samples, cache_dir=None):
        torch.manual_seed(spec.seed + 1)
        return torch.randint(0, 32, (num_samples, spec.sequence_length),
                             dtype=torch.long)

    def _noop_save(model, tokenizer, path, **kwargs):
        from pathlib import Path as _P
        _P(path).mkdir(parents=True, exist_ok=True)
        return _P(path)

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(orch, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(mio, "save_compressed_checkpoint", _noop_save)
    monkeypatch.setattr(orch, "save_compressed_checkpoint", _noop_save)


class _TinyTok:
    name_or_path = "tiny-tokenizer"
    eos_token_id = 0
    def __call__(self, text, *_, **__):
        return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}
    def save_pretrained(self, *_a, **_k):
        return None


def _run_tiny_stage1(model, tokenizer, config, tmp_path):
    from moe_compress import stage1
    from moe_compress.budget.solver import BudgetDecomposition
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2, expert_prune_ratio=0.5,
        svd_rank_ratio=0.14, global_expert_budget=4,
        min_experts_per_layer=2, blacklisted_experts={},
    )
    stage1.run(model, tokenizer, config, tmp_path, decomp)


def test_resume_path_forces_freq_weighted_and_warns_for_mergemoe(
    tiny_model, tiny_config, tmp_path, monkeypatch, caplog,
):
    """Resume with ``merge_step="mergemoe"`` forces freq_weighted + warns.

    Pins D-mergemoe-resume-fallback (Plugin #9):
      * Resume does NOT crash even though the per-layer
        ``_LayerInputAccumulator`` calibration buffer is not on disk.
      * A single ``log.warning`` fires (gated on configured ``"mergemoe"``
        and non-empty ``resumed_records``) — surfaces the silent downgrade.
      * Every replayed layer is forced through the ``freq_weighted`` branch
        (asserted by spying on ``_merge_experts_inplace`` and checking the
        ``merge_step`` kwarg the orchestrator passes).

    The orchestrator's resume gate fires when the partial-dir is populated
    AND ``no_resume=False`` — manufactured here by crashing Stage 2 mid-run
    via a monkey-patched ``_profile_layer`` (mirrors the pattern already
    used by ``test_stage2_pipeline_run_layer.py::
    test_stage2_pipeline_resume_skips_completed_layers``). The crashed
    layer's checkpoint persists in ``_stage2_partial/`` and is picked up
    by the second invocation.
    """
    import copy
    from moe_compress.stage2 import orchestrator as orch
    from moe_compress.utils.model_io import iter_moe_layers

    _patch_stage2_io(monkeypatch)

    # --- Stage 1 setup.
    pre_s2 = copy.deepcopy(tiny_model)
    _run_tiny_stage1(tiny_model, _TinyTok(), tiny_config, tmp_path)

    moe_layers = list(iter_moe_layers(tiny_model))
    assert len(moe_layers) >= 2, "need ≥2 MoE layers to manufacture a crash-resume"

    # --- First run: crash after layer 0 completes. The completed-layer
    # checkpoints (merge_0.json, layer_0.pt, optional _neuron_means_layer0.pt)
    # are flushed to ``_stage2_partial/`` before the crash propagates.
    original_profile = orch._profile_layer
    call_count = [0]

    def _crashing_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs):
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("simulated crash after layer 0")
        return original_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs)

    monkeypatch.setattr(orch, "_profile_layer", _crashing_profile)
    tiny_config["stage2_reap_ream"]["merge_step"] = "freq_weighted"
    with pytest.raises(RuntimeError, match="simulated crash"):
        orch.run(tiny_model, _TinyTok(), tiny_config, tmp_path, device=None)

    # Restore profile for the resume run.
    monkeypatch.setattr(orch, "_profile_layer", original_profile)

    # Sanity: partial-dir has at least one completed layer.
    partial_dir = tmp_path / "_stage2_partial"
    assert partial_dir.exists(), "_stage2_partial/ should survive the crash"
    merge_jsons = sorted(partial_dir.glob("merge_*.json"))
    assert len(merge_jsons) >= 1, (
        "expected at least one completed-layer record after the crash"
    )

    # --- Second run: flip merge_step to "mergemoe". Spy on
    # ``_merge_experts_inplace`` so we can prove every resume-loop call
    # forwards ``merge_step="freq_weighted"`` regardless of the config.
    spy_calls: list[dict] = []
    real_merge = orch._merge_experts_inplace

    def _spy_merge(*args, **kwargs):
        spy_calls.append({"args_len": len(args), "kwargs": dict(kwargs)})
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(orch, "_merge_experts_inplace", _spy_merge)

    tiny_config["stage2_reap_ream"]["merge_step"] = "mergemoe"
    resume_model = copy.deepcopy(pre_s2)

    # The resume loop runs BEFORE the orchestrator processes any
    # post-resume-point layer, so the D-mergemoe-resume-fallback warning
    # and the forced-freq_weighted ``_merge_experts_inplace`` calls fire
    # before the live MergeMoE solve on layer 1. Layer-1 MergeMoE may
    # still raise on environments that lack a LAPACK-enabled PyTorch
    # (CPU lstsq needs LAPACK) — catch *any* exception thereafter and
    # rely on the assertions below to pin the resume-loop behaviour.
    #
    # Pytest's caplog plugin sets ``propagate=False`` on the captured
    # logger; force it back on so caplog actually sees the warning.
    orch_log = logging.getLogger("moe_compress.stage2.orchestrator")
    _saved_propagate = orch_log.propagate
    orch_log.propagate = True
    try:
        caplog.set_level(logging.WARNING, logger="moe_compress.stage2.orchestrator")
        try:
            orch.run(resume_model, _TinyTok(), tiny_config, tmp_path,
                     device=None, no_resume=False)
        except RuntimeError as exc:
            # Acceptable: post-resume MergeMoE solve hit LAPACK-not-found
            # on the local CI box. The resume-loop assertions below still
            # cover this fixer-pin. Re-raise anything else.
            if "LAPACK" not in str(exc):
                raise
    finally:
        orch_log.propagate = _saved_propagate

    # --- Assertion 1: D-mergemoe-resume-fallback fired exactly once.
    matches = [
        r for r in caplog.records
        if "D-mergemoe-resume-fallback" in r.message
    ]
    assert len(matches) == 1, (
        f"expected exactly one D-mergemoe-resume-fallback warning "
        f"(found {len(matches)}): {[r.message for r in caplog.records]}"
    )
    assert matches[0].levelname == "WARNING"
    assert "forcing merge_step=freq_weighted" in matches[0].message

    # --- Assertion 2: at least one ``_merge_experts_inplace`` call on
    # the resume path forwarded ``merge_step="freq_weighted"`` (i.e. the
    # orchestrator did NOT honor the config's "mergemoe" for replayed
    # layers — see resume loop in stage2/orchestrator.py).
    resume_loop_calls = [
        c for c in spy_calls
        if c["kwargs"].get("merge_step") == "freq_weighted"
        and c["kwargs"].get("scores", "<missing>") is None
    ]
    assert len(resume_loop_calls) >= 1, (
        "expected at least one resume-loop _merge_experts_inplace call "
        "with merge_step='freq_weighted' (scores=None marks the resume "
        f"path); got: {spy_calls}"
    )


def test_resume_warning_does_not_fire_for_freq_weighted_config(
    tiny_model, tiny_config, tmp_path, monkeypatch, caplog,
):
    """Resume with ``merge_step="freq_weighted"`` (the default) does NOT
    emit the D-mergemoe-resume-fallback warning — it would be noise.

    The gate in ``stage2/orchestrator.py`` reads:
      ``if … merge_step == "mergemoe" and resumed_records:``
    so a freq_weighted config must short-circuit it.
    """
    import copy
    from moe_compress.stage2 import orchestrator as orch
    from moe_compress.utils.model_io import iter_moe_layers

    _patch_stage2_io(monkeypatch)
    pre_s2 = copy.deepcopy(tiny_model)
    _run_tiny_stage1(tiny_model, _TinyTok(), tiny_config, tmp_path)

    moe_layers = list(iter_moe_layers(tiny_model))
    assert len(moe_layers) >= 2

    original_profile = orch._profile_layer
    call_count = [0]

    def _crashing_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs):
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("simulated crash after layer 0")
        return original_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs)

    monkeypatch.setattr(orch, "_profile_layer", _crashing_profile)
    tiny_config["stage2_reap_ream"]["merge_step"] = "freq_weighted"
    with pytest.raises(RuntimeError, match="simulated crash"):
        orch.run(tiny_model, _TinyTok(), tiny_config, tmp_path, device=None)
    monkeypatch.setattr(orch, "_profile_layer", original_profile)

    assert (tmp_path / "_stage2_partial").exists()
    resume_model = copy.deepcopy(pre_s2)
    orch_log = logging.getLogger("moe_compress.stage2.orchestrator")
    _saved_propagate = orch_log.propagate
    orch_log.propagate = True
    try:
        caplog.set_level(logging.WARNING, logger="moe_compress.stage2.orchestrator")
        orch.run(resume_model, _TinyTok(), tiny_config, tmp_path,
                 device=None, no_resume=False)
    finally:
        orch_log.propagate = _saved_propagate

    matches = [
        r for r in caplog.records
        if "D-mergemoe-resume-fallback" in r.message
    ]
    assert matches == [], (
        f"D-mergemoe-resume-fallback warning fired for a freq_weighted "
        f"config — would be noise: {[r.message for r in matches]}"
    )
