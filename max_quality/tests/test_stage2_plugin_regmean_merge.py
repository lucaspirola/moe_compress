"""RegMean Stage-2 merge-step — clean-room re-impl tests.

Pins:
  * Plugin metadata + PipelinePlugin Protocol conformance.
  * Default-disabled: ``merge_step`` config-knob defaults to
    ``"freq_weighted"`` (NOT regmean); no opt-in surface flipped.
  * Pattern C config-validation: ``RegMeanMergeStepPlugin.validate_cov_acc_present``
    raises with an actionable message when ``cov_acc`` is missing.
  * Closed-form math correctness on hand-derived cases (1D and 2D).
  * Cross-check against upstream RegMean formula on synthetic data.
  * Per-cluster D-regmean-zero-cov-fallback path is exercised.
  * Per-cluster D-regmean-cond-fallback path is exercised.
  * Manifest order tests still pass.
  * The default ``"freq_weighted"`` path stays byte-identical with the
    new optional ``cov_acc`` kwarg threaded through.

See ``tasks/INVESTIGATION_FUSION_BENCH.md`` § Headline-Recommendation #4
for the spec; ``stage2/regmean.py`` for the math (with paper-cite
arXiv:2212.09849); ``stage2/plugins/regmean_merge.py`` for the shim.
"""
from __future__ import annotations

import copy
import logging

import pytest
import torch

from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage2.merging import _merge_experts_inplace
from moe_compress.stage2.plugins.regmean_merge import RegMeanMergeStepPlugin
from moe_compress.stage2.regmean import (
    _COND_THRESHOLD,
    _DAMPING_RATIO,
    _regmean_solve_one_linear,
)
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
from moe_compress.utils.model_io import build_banks, iter_moe_layers


# ---------------------------------------------------------------------------
# Section 1 — Plugin metadata + Protocol conformance + default-disabled
# ---------------------------------------------------------------------------


def test_regmean_plugin_is_pipeline_plugin_protocol():
    """RegMean plugin satisfies the ``PipelinePlugin`` Protocol."""
    p = RegMeanMergeStepPlugin()
    assert isinstance(p, PipelinePlugin)


def test_regmean_plugin_metadata_fields_populated():
    """Pin: required Protocol attrs are non-empty / informative."""
    p = RegMeanMergeStepPlugin()
    assert p.name == "regmean_merge_step"
    assert "2212.09849" in p.paper, (
        "RegMean plugin must cite the canonical paper arXiv:2212.09849."
    )
    assert "fusion_bench" in p.paper, (
        "RegMean plugin must attribute the upstream reference implementation."
    )
    assert "MIT" in p.paper, "Plugin paper string must record upstream license."
    assert p.config_key == "stage2_reap_ream.merge_step"
    assert p.reads == ()
    assert p.writes == ()
    assert p.provides == ()
    assert p.contribute_artifact(None) == {}


def test_regmean_plugin_disabled_when_merge_step_not_regmean():
    """Pin: ``is_enabled`` is False for the default + every non-regmean
    value; True only for case-insensitive 'regmean'."""
    p = RegMeanMergeStepPlugin()
    # Default-empty config: plugin OFF (matches no-opt-in semantics).
    assert p.is_enabled({}) is False
    assert p.is_enabled({"stage2_reap_ream": {}}) is False
    # Explicit non-regmean values: plugin OFF.
    assert p.is_enabled({"stage2_reap_ream": {"merge_step": "freq_weighted"}}) is False
    assert p.is_enabled({"stage2_reap_ream": {"merge_step": "mergemoe"}}) is False
    # Case-insensitive 'regmean': plugin ON.
    assert p.is_enabled({"stage2_reap_ream": {"merge_step": "regmean"}}) is True
    assert p.is_enabled({"stage2_reap_ream": {"merge_step": "RegMean"}}) is True
    assert p.is_enabled({"stage2_reap_ream": {"merge_step": "REGMEAN"}}) is True


def test_regmean_plugin_validate_cov_acc_raises_actionable():
    """Pattern C: ``validate_cov_acc_present(None)`` raises a ValueError
    naming the remediation (set merge_step back to freq_weighted)."""
    with pytest.raises(ValueError) as exc_info:
        RegMeanMergeStepPlugin.validate_cov_acc_present(None)
    msg = str(exc_info.value)
    assert "regmean" in msg.lower()
    assert "InputCovarianceAccumulator" in msg
    assert "freq_weighted" in msg, (
        "Pattern C error must name the remediation knob value."
    )


def test_regmean_plugin_validate_cov_acc_accepts_non_none():
    """``validate_cov_acc_present`` is a no-op when ``cov_acc`` is non-None."""
    acc = InputCovarianceAccumulator()
    # Must not raise.
    RegMeanMergeStepPlugin.validate_cov_acc_present(acc)


# ---------------------------------------------------------------------------
# Section 2 — Closed-form math correctness
# ---------------------------------------------------------------------------


def _reference_regmean_solve(
    W_list: list[torch.Tensor],
    G_list: list[torch.Tensor],
    damping_ratio: float = _DAMPING_RATIO,
) -> torch.Tensor:
    """Independent reference implementation of the RegMean formula.

    Computes ``W_M = ((Σ G + λ·tr/d·I)^{-1} · Σ G·W^T)^T`` directly with
    ``torch.linalg.solve``. Independent from the production helper so a
    bug in either path surfaces as a mismatch.

    Mirrors the paper's Eq. 2 plus the project's documented
    D-regmean-damping deviation.
    """
    d_out, d_in = W_list[0].shape
    G_sum = torch.zeros(d_in, d_in, dtype=torch.float32, device=W_list[0].device)
    GW_sum = torch.zeros(d_in, d_out, dtype=torch.float32, device=W_list[0].device)
    for W, G in zip(W_list, G_list):
        G_f = G.to(torch.float32)
        W_f = W.to(torch.float32)
        G_sum.add_(G_f)
        GW_sum.add_(G_f @ W_f.transpose(0, 1))
    trace_mean = (G_sum.diagonal().sum() / float(d_in)).item()
    damping = damping_ratio * trace_mean if trace_mean > 0.0 else 1.0
    G_reg = G_sum + damping * torch.eye(d_in, dtype=torch.float32, device=G_sum.device)
    WT = torch.linalg.solve(G_reg, GW_sum)
    return WT.transpose(0, 1).contiguous().to(W_list[0].dtype)


def test_regmean_math_1d_hand_derived():
    """Hand-derived 1D case: W_M = (Σ G W^T) / (Σ G + λ·tr/d).

    Two sources, 1-dim feature:
      W1 = [[2.0]], G1 = [[4.0]] (i.e. 4 calibration tokens of value 1.0)
      W2 = [[4.0]], G2 = [[1.0]] (i.e. 1 calibration token of value 1.0)
    Σ G = 5, Σ G·W^T = 4*2 + 1*4 = 12
    Damping: 1e-3 * trace(5)/1 = 5e-3 → denominator = 5.005
    W_M = 12 / 5.005 = 2.397602...
    """
    W = [torch.tensor([[2.0]]), torch.tensor([[4.0]])]
    G = [torch.tensor([[4.0]]), torch.tensor([[1.0]])]
    out = _regmean_solve_one_linear(
        weights_per_member=W, grams_per_member=G, alpha_per_member=[0.5, 0.5],
    )
    expected = 12.0 / (5.0 + _DAMPING_RATIO * 5.0)
    assert out.shape == (1, 1)
    assert abs(out.item() - expected) < 1e-5, (
        f"RegMean 1D solve diverged from hand-derived expected: "
        f"got {out.item()}, expected {expected}"
    )


def test_regmean_math_identity_gram_is_simple_average():
    """When all Gram matrices are equal multiples of identity, RegMean
    collapses to the unweighted average (up to a vanishing damping bias).

    Algebra (N=3, G_i = c·I, all matrices d_in × d_in):
        Σ G       = 3c · I
        Σ G·W_i^T = c · (W_1 + W_2 + W_3)^T = 3c · simple_avg^T
        trace_mean(Σ G) = 3c · d_in / d_in = 3c
        damping   = λ · 3c
        W_M^T     = (3c·I + λ·3c·I)^{-1} · 3c · simple_avg^T
                  = (1 / (1 + λ)) · simple_avg^T
        W_M       = simple_avg / (1 + λ)
    """
    torch.manual_seed(0)
    d_out, d_in = 2, 4
    N = 3
    W = [torch.randn(d_out, d_in) for _ in range(N)]
    G = [torch.eye(d_in) * 10.0 for _ in range(N)]
    out = _regmean_solve_one_linear(
        weights_per_member=W, grams_per_member=G, alpha_per_member=[1.0 / N] * N,
    )
    simple_avg = sum(W) / N
    # Per algebra above: damped factor is 1 / (1 + λ).
    expected = simple_avg / (1.0 + _DAMPING_RATIO)
    assert torch.allclose(out, expected, atol=2e-5, rtol=1e-5), (
        f"RegMean with identity Grams must equal damped simple-average "
        f"(simple_avg / (1 + λ)). max diff: {(out - expected).abs().max().item()}"
    )


def test_regmean_math_2d_matches_reference_solve():
    """Cross-check the production helper against an independent
    ``torch.linalg.solve``-based reference on a 2D synthetic case."""
    torch.manual_seed(1)
    d_out, d_in = 5, 7
    N = 4
    W = [torch.randn(d_out, d_in) for _ in range(N)]
    # Per-source random Grams: X_i ∈ R^{T_i, d_in} with T_i varying.
    G = []
    for T in (16, 32, 8, 48):
        X = torch.randn(T, d_in)
        G.append(X.T @ X)
    out = _regmean_solve_one_linear(
        weights_per_member=W, grams_per_member=G,
        alpha_per_member=[T / sum([16, 32, 8, 48]) for T in (16, 32, 8, 48)],
    )
    expected = _reference_regmean_solve(W, G)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4), (
        f"RegMean production helper diverged from independent reference. "
        f"max diff: {(out - expected).abs().max().item()}"
    )


def test_regmean_math_two_sources_unequal_grams():
    """Two sources with very different per-source Grams: the merge tilts
    toward the source with larger Gram (more calibration weight)."""
    torch.manual_seed(2)
    d_out, d_in = 3, 4
    W1 = torch.zeros(d_out, d_in)
    W2 = torch.ones(d_out, d_in)
    # G1 dominant (100× bigger trace) ⇒ RegMean output should be close to W1.
    G1 = torch.eye(d_in) * 100.0
    G2 = torch.eye(d_in) * 1.0
    out_dominant_W1 = _regmean_solve_one_linear(
        weights_per_member=[W1, W2], grams_per_member=[G1, G2],
        alpha_per_member=[0.5, 0.5],
    )
    # Expected: ~ (100*0 + 1*1) / (101 + damp) ≈ 1/101 ≈ 0.0099 on each entry.
    assert out_dominant_W1.abs().max().item() < 0.015, (
        f"RegMean did not tilt toward dominant Gram: out_max={out_dominant_W1.abs().max().item()}"
    )
    # Swap: G2 dominant ⇒ output close to W2.
    out_dominant_W2 = _regmean_solve_one_linear(
        weights_per_member=[W1, W2], grams_per_member=[G2, G1],
        alpha_per_member=[0.5, 0.5],
    )
    # Expected: ~ (1*0 + 100*1) / 101 ≈ 0.99 on each entry.
    assert (out_dominant_W2 - 1.0).abs().max().item() < 0.015, (
        f"RegMean did not tilt toward dominant Gram (swap): max diff "
        f"from 1.0 = {(out_dominant_W2 - 1.0).abs().max().item()}"
    )


def test_regmean_math_returns_dtype_of_input():
    """Output dtype matches ``weights_per_member[0]`` (project D-regmean-fp32-solve)."""
    torch.manual_seed(3)
    for dtype in (torch.float32, torch.float64):
        W = [torch.randn(2, 3, dtype=dtype) for _ in range(2)]
        G = [torch.eye(3, dtype=dtype) for _ in range(2)]
        out = _regmean_solve_one_linear(
            weights_per_member=W, grams_per_member=G, alpha_per_member=[0.5, 0.5],
        )
        assert out.dtype == dtype, (
            f"output dtype {out.dtype} != input dtype {dtype}"
        )


def test_regmean_math_singleton_rejected():
    """N<2 raises ValueError; caller must filter singletons."""
    W = [torch.randn(2, 3)]
    G = [torch.eye(3)]
    with pytest.raises(ValueError, match=r"N=1"):
        _regmean_solve_one_linear(
            weights_per_member=W, grams_per_member=G, alpha_per_member=[1.0],
        )


def test_regmean_math_length_mismatch_rejected():
    """All three sequences must have the same length."""
    W = [torch.randn(2, 3), torch.randn(2, 3)]
    G = [torch.eye(3)]
    with pytest.raises(ValueError, match=r"same length"):
        _regmean_solve_one_linear(
            weights_per_member=W, grams_per_member=G, alpha_per_member=[0.5, 0.5],
        )


def test_regmean_math_wrong_gram_shape_rejected():
    """Gram shape mismatch surfaces with an explicit error message."""
    W = [torch.randn(2, 3), torch.randn(2, 3)]
    G = [torch.eye(3), torch.eye(4)]  # wrong inner-dim
    with pytest.raises(ValueError, match=r"Gram shape"):
        _regmean_solve_one_linear(
            weights_per_member=W, grams_per_member=G, alpha_per_member=[0.5, 0.5],
        )


def test_regmean_math_cond_fallback_to_freq_weighted(caplog, monkeypatch):
    """When ``cond(Σ G + λ·tr/d·I) > _COND_THRESHOLD`` the solve falls
    back to the α-weighted simple average and logs the documented
    warning. Threshold-lowered via monkeypatch because the production
    damping λ=1e-3 is aggressive enough that triggering the cond gate
    with any naturally-degenerate Gram requires either ``λ=0`` or an
    artificially-low ``_COND_THRESHOLD``; the latter is the cleaner
    test surface.
    """
    torch.manual_seed(4)
    d_out, d_in = 3, 4
    # Rank-1 Gram for one source ⇒ Σ G is rank-1; damping makes cond
    # ~ d_in / λ ≈ 4000.  Lower _COND_THRESHOLD below that so the
    # condition gate fires.
    monkeypatch.setattr(
        "moe_compress.stage2.regmean._COND_THRESHOLD", 100.0,
    )
    x = torch.randn(1, d_in)
    G = [x.T @ x, x.T @ x]
    W = [torch.randn(d_out, d_in), torch.randn(d_out, d_in)]
    alpha = [0.7, 0.3]
    regmean_log = logging.getLogger("moe_compress.stage2.regmean")
    saved = regmean_log.propagate
    regmean_log.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="moe_compress.stage2.regmean"):
            out = _regmean_solve_one_linear(
                weights_per_member=W, grams_per_member=G, alpha_per_member=alpha,
            )
    finally:
        regmean_log.propagate = saved
    fallback_msgs = [r.message for r in caplog.records if "D-regmean-cond-fallback" in r.message]
    assert len(fallback_msgs) >= 1, (
        f"expected D-regmean-cond-fallback warning, got: {[r.message for r in caplog.records]}"
    )
    # Result matches α-weighted average.
    expected = alpha[0] * W[0] + alpha[1] * W[1]
    assert torch.allclose(out, expected, atol=1e-5)


def test_regmean_cond_threshold_constant():
    """Pin: ``_COND_THRESHOLD`` matches the design (1e8) — same as MergeMoE."""
    assert _COND_THRESHOLD == 1e8


def test_regmean_damping_ratio_constant():
    """Pin: ``_DAMPING_RATIO`` matches the documented default (1e-3)."""
    assert _DAMPING_RATIO == 1e-3


# ---------------------------------------------------------------------------
# Section 3 — Merge integration: _merge_experts_inplace routes to RegMean
# ---------------------------------------------------------------------------


def _snapshot_banks(layer_ref) -> dict[str, torch.Tensor]:
    banks = build_banks(layer_ref)
    return {
        name: bank._stacked().detach().clone()
        for name, bank in banks.items()
    }


def _populate_cov_acc_from_layer(layer_ref, cov_acc, n_tokens: int = 64,
                                 seed: int = 0) -> None:
    """Run a tiny forward pass that populates ``cov_acc`` for every expert
    of ``layer_ref`` so the RegMean per-cluster Gram lookup hits.

    The synthetic flow mirrors the production calibration: for each
    expert, feed ``n_tokens`` synthetic samples through the gate_proj
    (== gate_up_proj first half) input axis. The accumulator stores the
    Pearson-style X^T·X — we hand-construct two batches per expert with
    different distributions so the per-source Gram tilts the RegMean
    solve away from a simple average.
    """
    g = torch.Generator().manual_seed(seed)
    em = layer_ref.experts_module
    d_hidden = em.hidden_dim
    d_int = em.intermediate_dim
    n_experts = em.num_experts
    for e in range(n_experts):
        # gate_proj / up_proj share input ⇒ accumulator aliases on the key.
        x_gate = torch.randn(n_tokens, d_hidden, generator=g) + 0.1 * e
        cov_acc.update(layer_ref.layer_idx, e, "gate_proj", x_gate)
        # down_proj input lives in the intermediate space.
        x_down = torch.randn(n_tokens, d_int, generator=g) + 0.1 * e
        cov_acc.update(layer_ref.layer_idx, e, "down_proj", x_down)
    cov_acc.finalize_layer(layer_ref.layer_idx)


def test_merge_regmean_routes_through_new_path(tiny_model):
    """``merge_step="regmean"`` actually changes the merged weights vs
    the freq-weighted default (otherwise the new branch is a no-op)."""
    model_freq = copy.deepcopy(tiny_model)
    model_rm = copy.deepcopy(tiny_model)
    layer_freq = next(iter_moe_layers(model_freq))
    layer_rm = next(iter_moe_layers(model_rm))

    grouped = {0: [0, 1, 2]}
    freq = {0: 3, 1: 2, 2: 1}

    _merge_experts_inplace(layer_freq, grouped, freq, freq_weighted=True)
    snap_freq = _snapshot_banks(layer_freq)

    cov_acc = InputCovarianceAccumulator()
    _populate_cov_acc_from_layer(layer_rm, cov_acc, n_tokens=64, seed=7)
    _merge_experts_inplace(
        layer_rm, grouped, freq, freq_weighted=True,
        merge_step="regmean", cov_acc=cov_acc,
    )
    snap_rm = _snapshot_banks(layer_rm)

    # All three projections differ — RegMean is per-Linear, not per-block,
    # so unlike MergeMoE (gate/up unchanged) ALL three accumulators are
    # replaced by the closed-form solve.
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert not torch.equal(snap_rm[name][0], snap_freq[name][0]), (
            f"RegMean merged {name} did not diverge from freq-weighted — "
            "the new branch is silently a no-op."
        )
        assert torch.isfinite(snap_rm[name][0]).all(), (
            f"RegMean merged {name} produced non-finite entries"
        )


def test_merge_regmean_falls_back_when_cov_acc_none(tiny_model, caplog):
    """``merge_step="regmean"`` with ``cov_acc=None`` falls back to
    freq-weighted with a WARNING (defensive fast-path)."""
    model_freq = copy.deepcopy(tiny_model)
    model_rm = copy.deepcopy(tiny_model)
    layer_freq = next(iter_moe_layers(model_freq))
    layer_rm = next(iter_moe_layers(model_rm))
    grouped = {0: [0, 1]}
    freq = {0: 2, 1: 1}

    _merge_experts_inplace(layer_freq, grouped, freq, freq_weighted=True)

    merging_log = logging.getLogger("moe_compress.stage2.merging")
    saved = merging_log.propagate
    merging_log.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="moe_compress.stage2.merging"):
            _merge_experts_inplace(
                layer_rm, grouped, freq, freq_weighted=True,
                merge_step="regmean", cov_acc=None,
            )
    finally:
        merging_log.propagate = saved
    assert any(
        "merge_step='regmean'" in r.message and "cov_acc is None" in r.message
        for r in caplog.records
    ), (
        "expected the cov_acc-None fallback warning, got: "
        f"{[r.message for r in caplog.records]}"
    )
    snap_freq = _snapshot_banks(layer_freq)
    snap_rm = _snapshot_banks(layer_rm)
    for name in snap_freq:
        assert torch.equal(snap_freq[name], snap_rm[name]), (
            f"{name}: cov_acc=None fallback diverged from freq-weighted"
        )


def test_merge_regmean_falls_back_when_member_gram_missing(tiny_model, caplog):
    """``merge_step="regmean"`` falls back per-cluster when a member's
    Gram is missing (zero calibration traffic for that expert) — see
    D-regmean-zero-cov-fallback."""
    model_freq = copy.deepcopy(tiny_model)
    model_rm = copy.deepcopy(tiny_model)
    layer_freq = next(iter_moe_layers(model_freq))
    layer_rm = next(iter_moe_layers(model_rm))
    grouped = {0: [0, 1, 2]}
    freq = {0: 1, 1: 1, 2: 1}

    _merge_experts_inplace(layer_freq, grouped, freq, freq_weighted=True)

    # Populate cov_acc for experts 0 and 1 ONLY. Expert 2 → missing Gram.
    cov_acc = InputCovarianceAccumulator()
    em = layer_rm.experts_module
    g = torch.Generator().manual_seed(11)
    for e in (0, 1):
        x_gate = torch.randn(32, em.hidden_dim, generator=g)
        cov_acc.update(layer_rm.layer_idx, e, "gate_proj", x_gate)
        x_down = torch.randn(32, em.intermediate_dim, generator=g)
        cov_acc.update(layer_rm.layer_idx, e, "down_proj", x_down)
    cov_acc.finalize_layer(layer_rm.layer_idx)
    assert cov_acc.get((layer_rm.layer_idx, 2, "gate_proj")) is None, (
        "test setup error: expert 2 should have no Gram"
    )

    merging_log = logging.getLogger("moe_compress.stage2.merging")
    saved = merging_log.propagate
    merging_log.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="moe_compress.stage2.merging"):
            _merge_experts_inplace(
                layer_rm, grouped, freq, freq_weighted=True,
                merge_step="regmean", cov_acc=cov_acc,
            )
    finally:
        merging_log.propagate = saved
    fallback_msgs = [
        r.message for r in caplog.records
        if "D-regmean-zero-cov-fallback" in r.message
    ]
    assert len(fallback_msgs) >= 1, (
        f"expected D-regmean-zero-cov-fallback warning; got: "
        f"{[r.message for r in caplog.records]}"
    )
    snap_freq = _snapshot_banks(layer_freq)
    snap_rm = _snapshot_banks(layer_rm)
    for name in snap_freq:
        assert torch.equal(snap_freq[name], snap_rm[name]), (
            f"{name}: per-cluster zero-cov fallback must produce identical "
            f"output to freq-weighted merge"
        )


def test_merge_invalid_merge_step_rejected(tiny_model):
    """Unknown ``merge_step`` raises ValueError naming all legal values."""
    layer = next(iter_moe_layers(tiny_model))
    with pytest.raises(ValueError, match=r"freq_weighted.*mergemoe.*regmean"):
        _merge_experts_inplace(
            layer, {0: [0, 1]}, {0: 1, 1: 1},
            freq_weighted=True, merge_step="banana",
        )


def test_merge_default_freq_weighted_still_byte_identical_with_cov_acc_kwarg(tiny_model):
    """Threading the new ``cov_acc`` kwarg through to a freq-weighted
    call MUST be a no-op — the default path stays byte-identical."""
    model_a = copy.deepcopy(tiny_model)
    model_b = copy.deepcopy(tiny_model)
    layer_a = next(iter_moe_layers(model_a))
    layer_b = next(iter_moe_layers(model_b))
    grouped = {0: [0, 1, 2]}
    freq = {0: 4, 1: 2, 2: 1}

    cov_acc = InputCovarianceAccumulator()
    _populate_cov_acc_from_layer(layer_a, cov_acc, n_tokens=32, seed=0)

    # Default — no cov_acc kwarg.
    _merge_experts_inplace(layer_a, grouped, freq, freq_weighted=True)
    snap_a = _snapshot_banks(layer_a)

    # Explicit freq_weighted with cov_acc — the kwarg is unused on
    # freq-weighted and must not perturb the result.
    _merge_experts_inplace(
        layer_b, grouped, freq, freq_weighted=True,
        merge_step="freq_weighted", cov_acc=cov_acc,
    )
    snap_b = _snapshot_banks(layer_b)
    for name in snap_a:
        assert torch.equal(snap_a[name], snap_b[name]), (
            f"{name}: cov_acc kwarg perturbed the freq-weighted path"
        )


# ---------------------------------------------------------------------------
# Section 4 — Orchestrator config validation + default
# ---------------------------------------------------------------------------


def test_orchestrator_rejects_invalid_merge_step_message_lists_regmean(
    tiny_config, tiny_model, tmp_path,
):
    """``stage2_reap_ream.merge_step="banana"`` is rejected with a message
    that NAMES regmean as a legal value (operators must discover it)."""
    from moe_compress.stage2 import orchestrator as orch
    tiny_config["stage2_reap_ream"] = {
        "merge_step": "banana",
        "ream": {"frequency_weighted_merge": True},
        "num_calibration_samples": 1,
        "batch_size": 1,
    }
    (tmp_path / "stage1_budgets.json").write_text('{"per_layer_target_experts": {}}')
    (tmp_path / "stage1_blacklist.json").write_text('{"blacklist": {}}')
    with pytest.raises(ValueError, match=r"freq_weighted.*mergemoe.*regmean"):
        orch.run(tiny_model, tokenizer=None, config=tiny_config,
                 artifacts_dir=tmp_path)


def test_orchestrator_default_merge_step_unchanged_by_regmean_addition(
    tiny_config, tiny_model, tmp_path,
):
    """REGRESSION pin: adding RegMean must NOT flip the default away from
    freq_weighted. A tiny_config with no merge_step override must NOT
    trigger the validator (it would only trigger on banana)."""
    from moe_compress.stage2 import orchestrator as orch
    tiny_config["stage2_reap_ream"] = {
        "ream": {"frequency_weighted_merge": True},
        "num_calibration_samples": 1,
        "batch_size": 1,
    }
    # Make stage1 artifacts available so the validator runs before any
    # stage1-blocked error path.
    (tmp_path / "stage1_budgets.json").write_text('{"per_layer_target_experts": {}}')
    (tmp_path / "stage1_blacklist.json").write_text('{"blacklist": {}}')
    # The run will fail downstream (stage1 budgets are empty) but the
    # merge_step validator must NOT raise — that is the regression pin.
    with pytest.raises(Exception) as exc_info:
        orch.run(tiny_model, tokenizer=None, config=tiny_config,
                 artifacts_dir=tmp_path)
    msg = str(exc_info.value)
    assert "merge_step" not in msg, (
        f"adding regmean must not change the default merge_step validator "
        f"path; got merge_step-mentioning error: {msg}"
    )


# ---------------------------------------------------------------------------
# Section 5 — Manifest / registry order
# ---------------------------------------------------------------------------


def test_regmean_plugin_class_importable_from_plugins_namespace():
    """Importability + manifest pin: the new module sits under
    ``stage2/plugins/`` and exports ``RegMeanMergeStepPlugin`` so it can
    be enumerated by registry-introspection tooling.
    """
    import importlib
    mod = importlib.import_module("moe_compress.stage2.plugins.regmean_merge")
    assert hasattr(mod, "RegMeanMergeStepPlugin")


def test_regmean_math_module_importable():
    """``stage2/regmean.py`` lives next to ``stage2/mergemoe.py`` — both
    are merge-step math modules consumed by the merge spine."""
    import importlib
    mod = importlib.import_module("moe_compress.stage2.regmean")
    assert hasattr(mod, "_regmean_solve_one_linear")
    assert hasattr(mod, "_DAMPING_RATIO")
    assert hasattr(mod, "_COND_THRESHOLD")


# ---------------------------------------------------------------------------
# Section 6 — REAM / mergemoe regression: those paths still work
# ---------------------------------------------------------------------------


def test_freq_weighted_path_unchanged_byte_identical_default(tiny_model):
    """REAM freq-weighted (the default merge_step) MUST be byte-identical
    before and after the RegMean addition — defended by the matching
    test in test_stage2_plugin_mergemoe_step.py; this test re-pins the
    invariant here too, since the new `cov_acc` kwarg is non-trivial
    plumbing through `_merge_experts_inplace`."""
    model_a = copy.deepcopy(tiny_model)
    model_b = copy.deepcopy(tiny_model)
    layer_a = next(iter_moe_layers(model_a))
    layer_b = next(iter_moe_layers(model_b))
    grouped = {0: [0, 1, 2]}
    freq = {0: 4, 1: 2, 2: 1}

    _merge_experts_inplace(layer_a, grouped, freq, freq_weighted=True)
    snap_a = _snapshot_banks(layer_a)

    _merge_experts_inplace(
        layer_b, grouped, freq, freq_weighted=True, merge_step="freq_weighted",
    )
    snap_b = _snapshot_banks(layer_b)
    for name in snap_a:
        assert torch.equal(snap_a[name], snap_b[name]), (
            f"{name}: freq-weighted byte-identical pin broken by RegMean addition"
        )


def test_mergemoe_path_unchanged_by_regmean_addition(tiny_model):
    """MergeMoE remains functional after the RegMean addition (no
    accidental coupling)."""
    model_mm = copy.deepcopy(tiny_model)
    layer = next(iter_moe_layers(model_mm))
    grouped = {0: [0, 1, 2]}
    freq = {0: 3, 1: 2, 2: 1}
    d_hidden = layer.experts_module.hidden_dim
    torch.manual_seed(7)
    X_hat = torch.randn(32, d_hidden)
    # The MergeMoE branch must not raise; the cov_acc kwarg is unused.
    _merge_experts_inplace(
        layer, grouped, freq, freq_weighted=True,
        merge_step="mergemoe", layer_inputs=X_hat, token_cap=32,
        cov_acc=None,
    )
    banks = build_banks(layer)
    assert torch.isfinite(banks["down_proj"]._stacked()).all()


# ---------------------------------------------------------------------------
# Section 10 — R-1 (RegMean resume Trackio forensics) — orchestrator-level
# ---------------------------------------------------------------------------
#
# Pins the eager-init contract from
# ``tasks/PLAN_R1_REGMEAN_RESUME_TRACKIO.md`` § 2.2: by the time the one-shot
# Trackio config emit at ``orchestrator.run()`` fires,
# ``_effective_merge_step_per_layer`` MUST contain one entry per MoE layer.
# Resume forces ``"freq_weighted"`` for replayed layers (overriding the
# configured ``merge_step``); clean runs leave every entry at the configured
# value.
#
# Test 1 covers RegMean resume → downgrade counter / reason / dict shape.
# Test 2 covers RegMean clean run → zero downgrades, full per-layer coverage.
# Both tests reuse the resume fixture pattern from
# ``test_smoke_stage2_resume.py`` (see plan § 4 for the rationale of not
# parametrising).


def _r1_make_tiny_tokenizer():
    """Tokenizer shim mirroring ``test_smoke_stage2_resume._TinyTokenizer``.

    Inlined here so this module does not import the smoke test (which would
    couple two unrelated test files together).
    """
    class _TinyTokenizer:
        name_or_path = "tiny-tokenizer"
        eos_token_id = 0

        def __call__(self, text, *_, **__):
            return {"input_ids": [min(ord(c) % 32, 31) for c in (text or " ")]}

        def save_pretrained(self, *_args, **_kwargs):
            return None

    return _TinyTokenizer()


def _r1_noop_save(model, tokenizer, path, **kwargs):
    from pathlib import Path as _Path

    _Path(path).mkdir(parents=True, exist_ok=True)
    return _Path(path)


@pytest.fixture
def _r1_patched_stage2(monkeypatch, tiny_config):
    """Resume-fixture clone (CPU-only) configured for RegMean.

    Mirrors ``test_smoke_stage2_resume.patched_stage2`` but flips
    ``merge_step`` to ``"regmean"`` so the eager-init contract is exercised
    on the RegMean code paths. Returns the mutated config dict.
    """
    import copy as _copy

    from moe_compress.stage2 import orchestrator as stage2_reap_ream
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

    monkeypatch.setattr(cal_mod, "build_calibration_tensor", _fake_build)
    monkeypatch.setattr(cal_mod, "build_super_expert_slice", _fake_slice)
    monkeypatch.setattr(stage2_reap_ream, "build_calibration_tensor", _fake_build)

    monkeypatch.setattr(mio, "save_compressed_checkpoint", _r1_noop_save)
    monkeypatch.setattr(stage2_reap_ream, "save_compressed_checkpoint", _r1_noop_save)

    cfg = _copy.deepcopy(tiny_config)
    cfg["stage2_reap_ream"]["merge_step"] = "regmean"
    return cfg


def _r1_run_stages_01(model, config, tmp_path):
    """Stage-0/1 prep clone of ``test_smoke_stage2_resume._run_stages_01``."""
    from moe_compress import stage1
    from moe_compress.budget.solver import BudgetDecomposition

    tokenizer = _r1_make_tiny_tokenizer()
    decomp = BudgetDecomposition(
        total_reduction_ratio=0.2,
        expert_prune_ratio=0.5,
        svd_rank_ratio=0.14,
        global_expert_budget=4,
        min_experts_per_layer=2,
        blacklisted_experts={},
    )
    stage1.run(model, tokenizer, config, tmp_path, decomp)
    return decomp


def _r1_find_config_emit(captured_emits):
    """Locate the one-shot config emit in the captured Trackio call list.

    The emit is the dict containing ``stage2/config/merge_step``. We assert a
    single match to pin the eager-init contract on the ONE canonical emit
    (post-resume, pre-walker) — if the orchestrator ever splits this into
    two emits, this helper fails loudly.
    """
    matches = [p for p in captured_emits if "stage2/config/merge_step" in p]
    assert len(matches) == 1, (
        f"Expected exactly one config emit, found {len(matches)}; "
        "the R-1 contract pins the eager-init dict on the canonical emit "
        "at orchestrator.run() (post-resume, pre-walker)."
    )
    return matches[0]


def test_regmean_resume_emits_downgrade_counter(
    tiny_model, _r1_patched_stage2, tmp_path, monkeypatch
):
    """R-1: resume of a RegMean run reports the forced freq_weighted
    downgrade via the new ``stage2/effective/*`` Trackio keys.

    End-to-end proof:
      * ``stage2/effective/merge_step_downgrades_total`` equals the number of
        replayed layers (1 in this fixture: crash after layer 0).
      * ``stage2/effective/merge_step_downgrade_reason`` reads
        ``"regmean_resume_no_cov_load_before_merge"`` (RegMean branch).
      * ``stage2/effective/merge_step_per_layer`` is a JSON dict with FULL
        MoE coverage (one entry per layer in ``moe_layers``) — pins the
        eager-init contract from plan § 2.2. The replayed layer maps to
        ``"freq_weighted"``; every other MoE layer maps to ``"regmean"``.
    """
    import copy as _copy
    import json as _json

    from moe_compress.stage2 import orchestrator as stage2_reap_ream

    _r1_run_stages_01(tiny_model, _r1_patched_stage2, tmp_path)
    model_before_s2 = _copy.deepcopy(tiny_model)

    moe_layers = list(iter_moe_layers(tiny_model))
    assert len(moe_layers) >= 2, "Need at least 2 MoE layers for this test"
    layer0_idx = moe_layers[0].layer_idx

    # --- First run: crash after layer 0 is fully processed (writes partial) ---
    original_profile = stage2_reap_ream._profile_layer
    call_count = [0]

    def _crashing_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs):
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("simulated crash after layer 0")
        return original_profile(model, layer_ref, batches, reap_acc, cov_acc, ream_acc, **kwargs)

    monkeypatch.setattr(stage2_reap_ream, "_profile_layer", _crashing_profile)
    with pytest.raises(RuntimeError, match="simulated crash after layer 0"):
        stage2_reap_ream.run(
            tiny_model, _r1_make_tiny_tokenizer(), _r1_patched_stage2,
            tmp_path, device=None,
        )

    # Confirm partial files are present so the resume path is exercised.
    partial_dir = tmp_path / "_stage2_partial"
    assert (partial_dir / f"merge_{layer0_idx}.json").exists()
    assert (partial_dir / f"layer_{layer0_idx}.pt").exists()

    # --- Restore model + capture Trackio emits on the resume run ---
    model_for_resume = _copy.deepcopy(model_before_s2)
    monkeypatch.setattr(stage2_reap_ream, "_profile_layer", original_profile)

    captured: list[dict] = []

    def _capture_trackio(payload, *args, **kwargs):
        captured.append(dict(payload))

    monkeypatch.setattr(stage2_reap_ream, "_trackio_log", _capture_trackio)

    stage2_reap_ream.run(
        model_for_resume, _r1_make_tiny_tokenizer(), _r1_patched_stage2,
        tmp_path, device=None,
    )

    payload = _r1_find_config_emit(captured)

    # Configured value is unchanged.
    assert payload["stage2/config/merge_step"] == "regmean"

    # One replayed layer → one downgrade.
    assert payload["stage2/effective/merge_step_downgrades_total"] == 1

    # Reason string maps to the RegMean branch.
    assert payload["stage2/effective/merge_step_downgrade_reason"] == (
        "regmean_resume_no_cov_load_before_merge"
    )

    # Per-layer dict shape: JSON-encoded, FULL MoE coverage (eager-init
    # contract from plan § 2.2). Without eager init the non-resumed layers
    # would be missing from the dict at emit time — this assertion catches
    # a regression of the eager-init fix.
    per_layer = _json.loads(payload["stage2/effective/merge_step_per_layer"])
    expected_keys = {str(layer.layer_idx) for layer in moe_layers}
    assert set(per_layer.keys()) == expected_keys, (
        "Eager-init contract broken: per-layer dict missing entries for "
        f"non-resumed MoE layers. Expected {expected_keys}, "
        f"got {set(per_layer.keys())}."
    )

    # Replayed layer → "freq_weighted"; every other → "regmean".
    assert per_layer[str(layer0_idx)] == "freq_weighted"
    for layer in moe_layers:
        if layer.layer_idx == layer0_idx:
            continue
        assert per_layer[str(layer.layer_idx)] == "regmean", (
            f"Non-resumed layer {layer.layer_idx} carries "
            f"{per_layer[str(layer.layer_idx)]!r}; expected the configured "
            "'regmean' (eager-init default)."
        )


def test_regmean_clean_run_emits_zero_downgrades(
    tiny_model, _r1_patched_stage2, tmp_path, monkeypatch
):
    """R-1: a clean (no-resume) RegMean run pins zero downgrades AND full
    per-layer coverage of the configured ``"regmean"`` value.

    Guards against:
      * accidentally counting absence of resume as a "downgrade" (off-by-one);
      * the v1 ``NameError`` on the no-resume path (eager init hoisted
        OUTSIDE the ``else:`` branch makes the binding exist on both code
        paths — see plan § 2.2 CRITICAL-2).
    """
    import json as _json

    from moe_compress.stage2 import orchestrator as stage2_reap_ream

    _r1_run_stages_01(tiny_model, _r1_patched_stage2, tmp_path)
    moe_layers = list(iter_moe_layers(tiny_model))

    captured: list[dict] = []

    def _capture_trackio(payload, *args, **kwargs):
        captured.append(dict(payload))

    monkeypatch.setattr(stage2_reap_ream, "_trackio_log", _capture_trackio)

    stage2_reap_ream.run(
        tiny_model, _r1_make_tiny_tokenizer(), _r1_patched_stage2,
        tmp_path, device=None,
    )

    payload = _r1_find_config_emit(captured)

    # Zero downgrades + empty reason on the clean path.
    assert payload["stage2/effective/merge_step_downgrades_total"] == 0
    assert payload["stage2/effective/merge_step_downgrade_reason"] == ""

    # Per-layer dict: FULL MoE coverage (eager-init contract). Without eager
    # init the clean-run dict would be empty — no resume-loop writes occur,
    # so this assertion is the load-bearing pin for plan § 2.2.
    per_layer = _json.loads(payload["stage2/effective/merge_step_per_layer"])
    expected_keys = {str(layer.layer_idx) for layer in moe_layers}
    assert set(per_layer.keys()) == expected_keys, (
        "Eager-init contract broken on clean run: per-layer dict is "
        f"missing entries. Expected {expected_keys}, "
        f"got {set(per_layer.keys())}."
    )

    # Every layer carries the configured value.
    for layer in moe_layers:
        assert per_layer[str(layer.layer_idx)] == "regmean", (
            f"Clean-run layer {layer.layer_idx} carries "
            f"{per_layer[str(layer.layer_idx)]!r}; expected configured "
            "'regmean'."
        )
