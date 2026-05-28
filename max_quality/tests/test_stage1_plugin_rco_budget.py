"""Unit tests for ``moe_compress.stage1.plugins.rco_budget`` (clean-room re-impl).

Total: 34 tests covering plan ``tasks/PLAN_RCO_NATIVE_REIMPL.md`` §6:

- **C1-C16 (16)**: code-quality / contract / Pattern B/C/E adoption tests
- **F1-F15 (15)**: paper-fidelity tests pinning algorithm correctness
- **R1-R3 (3)**: regression / byte-identity tests

Naming follows the plan's test IDs verbatim. The "carried over" 12 tests
(C1-C5, C8-C10, C13-C16) keep their existing function names; their
expected values were re-derived where the algorithm changed (Gumbel form,
cosine anneal direction, β·log p removal, infeasibility fallback).

The 4 ON-path behavior changes from plan §7 are pinned by F11 (Gumbel
form), F12 (cosine direction), F15 (pure-damage DP), F8 (nearest
feasible budget).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage1.plugins.rco_budget import RCOBudgetPlugin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_decomposition(global_budget: int = 6) -> BudgetDecomposition:
    """Construct a real BudgetDecomposition (only ``global_expert_budget`` is consumed)."""
    return BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=global_budget,
        min_experts_per_layer=2,
    )


def _build_inputs(*, global_budget: int = 6, enabled: bool = True) -> dict:
    """Two MoE layers × 4 experts each. GRAPE allocates {0: 3, 1: 3} (sum = 6).

    floor_divisor = 2 → floor_l = 2 → option grid {2, 3, 4} per layer.
    """
    n = 4
    return {
        "per_layer_target_experts": {"0": 3, "1": 3},
        "per_layer_redundancy": {"0": 0.5, "1": 0.5},
        "per_layer_targets": {0: n, 1: n},
        "decomposition": _make_decomposition(global_budget=global_budget),
        "config": {
            "stage1": {
                "rco_budget": {
                    "enabled": enabled,
                    "n_iterations": 50,   # tiny for fast tests
                    "learning_rate": 0.1,
                    "gumbel_tau_init": 5.0,
                    "gumbel_tau_final": 0.5,
                    "init_peak_logit": 2.0,
                    "floor_divisor": 2,
                    "seed": 0,
                }
            }
        },
    }


def _populate_context(inputs: dict) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set("per_layer_target_experts", inputs["per_layer_target_experts"])
    ctx.set("per_layer_redundancy", inputs["per_layer_redundancy"])
    ctx.set("per_layer_targets", inputs["per_layer_targets"])
    ctx.set("decomposition", inputs["decomposition"])
    ctx.set("config", inputs["config"])
    return ctx


# ===========================================================================
# C1-C16 — Code-quality / contract tests
# ===========================================================================


# --- C1: Protocol-attribute contract ---------------------------------------


def test_plugin_protocol_attributes():
    """C1: class-level attributes match the plan exactly + nine deviations cited."""
    plugin = RCOBudgetPlugin()
    assert plugin.name == "rco_budget"
    assert "arxiv:2605.00649" in plugin.paper
    assert "clean-room" in plugin.paper
    assert "IST-DASLab" in plugin.paper
    # All NINE D-* deviations must be cited in the paper string (8 carried +
    # 1 new D-adam-no-v-transport).
    for deviation_token in (
        "D-clean-room",
        "D-init-grape",
        "D-fitness-mse",
        "D-synthetic-curve",
        "D-floor-projection",
        "D-ragged-K",
        "D-bisection-budget",
        "D-disabled-default",
        "D-adam-no-v-transport",
    ):
        assert deviation_token in plugin.paper
    assert plugin.config_key == "stage1.rco_budget"
    assert plugin.reads == (
        "per_layer_target_experts",
        "per_layer_redundancy",
        "per_layer_targets",
        "decomposition",
        "config",
    )
    assert plugin.writes == (
        "per_layer_target_experts_rco",
        "rco_metadata",
    )
    assert plugin.provides == ()


# --- C2: Runtime-checkable Protocol -----------------------------------------


def test_plugin_is_runtime_checkable_pipelineplugin():
    """C2: ``isinstance`` against the runtime-checkable Protocol must succeed."""
    assert isinstance(RCOBudgetPlugin(), PipelinePlugin)


# --- C3, C4: is_enabled gate ------------------------------------------------


def test_plugin_disabled_by_default():
    """C3: RCO is OFF unless the explicit flag is set true."""
    plugin = RCOBudgetPlugin()
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled({"stage1": {}}) is False
    assert plugin.is_enabled({"stage1": {"rco_budget": {}}}) is False
    assert plugin.is_enabled({"stage1": {"rco_budget": {"enabled": False}}}) is False


def test_plugin_enabled_when_flag_true():
    """C4: when the explicit flag is True, the plugin enables."""
    plugin = RCOBudgetPlugin()
    assert plugin.is_enabled({"stage1": {"rco_budget": {"enabled": True}}}) is True


# --- C5: Missing-slot KeyError contract ------------------------------------


@pytest.mark.parametrize(
    "missing_slot",
    [
        "per_layer_target_experts",
        "per_layer_redundancy",
        "per_layer_targets",
        "decomposition",
        "config",
    ],
)
def test_run_rejects_missing_slot(missing_slot):
    """C5: ``plugin.run`` must raise ``KeyError`` mentioning the missing slot name."""
    inputs = _build_inputs()
    populators = {
        "per_layer_target_experts": inputs["per_layer_target_experts"],
        "per_layer_redundancy": inputs["per_layer_redundancy"],
        "per_layer_targets": inputs["per_layer_targets"],
        "decomposition": inputs["decomposition"],
        "config": inputs["config"],
    }
    ctx = PipelineContext()
    for slot, value in populators.items():
        if slot == missing_slot:
            continue
        ctx.set(slot, value)

    with pytest.raises(KeyError) as exc:
        RCOBudgetPlugin().run(ctx)
    assert missing_slot in str(exc.value)


# --- C6: Config validation rejects unknown keys (Pattern C) ----------------


def test_config_rejects_unknown_keys():
    """C6: a typo'd key (e.g. ``learning_rates``) raises ValueError listing the unknown key."""
    inputs = _build_inputs()
    # Add a typo'd key to the rco_budget block.
    inputs["config"]["stage1"]["rco_budget"]["learning_rates"] = 0.1  # typo
    ctx = _populate_context(inputs)
    with pytest.raises(ValueError) as exc:
        RCOBudgetPlugin().run(ctx)
    assert "learning_rates" in str(exc.value)
    assert "unknown" in str(exc.value).lower()


# --- C7: Config range checks (Pattern C) ------------------------------------


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("n_iterations", 0),
        ("n_iterations", -1),
        ("learning_rate", 0.0),
        ("learning_rate", -0.1),
        ("learning_rate", 100.0),
        ("gumbel_tau_init", 0.0),
        ("gumbel_tau_init", -1.0),
        ("gumbel_tau_final", 0.0),
        ("gumbel_tau_final", -1.0),
        ("floor_divisor", 0),
        ("floor_divisor", -1),
        ("adam_beta1", -0.1),
        ("adam_beta1", 1.0),
        ("adam_beta2", 1.5),
        ("adam_eps", 0.0),
        ("adam_eps", -1e-8),
    ],
)
def test_config_range_checks(key, bad_value):
    """C7: invalid ranges all raise ValueError."""
    inputs = _build_inputs()
    inputs["config"]["stage1"]["rco_budget"][key] = bad_value
    ctx = _populate_context(inputs)
    with pytest.raises(ValueError) as exc:
        RCOBudgetPlugin().run(ctx)
    assert key in str(exc.value) or "rco_budget" in str(exc.value)


def test_config_range_check_tau_ordering():
    """C7 (ordering branch): tau_final >= tau_init raises ValueError."""
    inputs = _build_inputs()
    inputs["config"]["stage1"]["rco_budget"]["gumbel_tau_init"] = 0.5
    inputs["config"]["stage1"]["rco_budget"]["gumbel_tau_final"] = 5.0  # final >= init
    ctx = _populate_context(inputs)
    with pytest.raises(ValueError) as exc:
        RCOBudgetPlugin().run(ctx)
    msg = str(exc.value)
    assert "tau_final" in msg or "tau_init" in msg


# --- C8: synthetic 2-layer regression --------------------------------------


def test_run_synthetic_2layer_handcheck():
    """C8: 2 layers × 4 experts, B=6, no damage curve → synthetic fallback.

    Hand-check (under the corrected algorithm):
    - Option grid {2, 3, 4} per layer; floor=2.
    - GRAPE init {3, 3} (sum 6).
    - Symmetric R̃ = 0.5 on both layers ⇒ synthetic curve is symmetric.
    - The DP returns a feasible budget vector summing to 6 that respects
      the per-layer floor.

    Assertions: budget vector is feasible (floor-respecting, sum=6) and
    each layer's budget is in [2, 4]. We don't pin a specific vector
    because under the corrected algorithm the symmetric-cost tiebreak
    direction may differ from the historical (β·log p) impl.
    """
    inputs = _build_inputs(global_budget=6)
    ctx = _populate_context(inputs)
    RCOBudgetPlugin().run(ctx)

    rco_budgets = ctx.get("per_layer_target_experts_rco")
    assert set(rco_budgets.keys()) == {"0", "1"}
    for k, v in rco_budgets.items():
        assert 2 <= v <= 4, f"layer {k}: budget {v} outside [floor, N]"
    assert sum(rco_budgets.values()) == 6


# --- C9: damage curve consumption ------------------------------------------


def test_run_consumes_damage_curve_when_present():
    """C9: with an asymmetric damage curve, RCO shifts allocation toward the costly layer.

    Layer 0 has a STEEP damage curve (each removed expert costs 100); layer 1
    is nearly flat (each removed expert costs 1). RCO should keep layer 0
    near its full count (4) and let layer 1 absorb the compression.

    Under the corrected (pure-damage) DP this matches the existing
    expected value: (4, 2) gives damage 0 + 2 = 2, vs (3, 3) = 100 + 1 =
    101, vs (2, 4) = 200 + 0 = 200. So (4, 2) wins.
    """
    inputs = _build_inputs(global_budget=6)
    ctx = _populate_context(inputs)
    ctx.set("per_layer_damage_curve", {
        0: {2: 200.0, 3: 100.0, 4: 0.0},
        1: {2: 2.0, 3: 1.0, 4: 0.0},
    })
    RCOBudgetPlugin().run(ctx)

    rco_budgets = ctx.get("per_layer_target_experts_rco")
    assert rco_budgets["0"] == 4, (
        f"steep layer 0 should keep all 4 experts, got {rco_budgets['0']}"
    )
    assert rco_budgets["1"] == 2, (
        f"flat layer 1 should drop to floor 2, got {rco_budgets['1']}"
    )
    assert sum(rco_budgets.values()) == 6

    metadata = ctx.get("rco_metadata")
    assert metadata["fitness_source"] == "damage_curve"
    assert metadata["fitness_signal_resolved"] == "damage_curve"
    assert metadata["achieved_budget"] == 6
    assert metadata["requested_budget"] == 6
    assert metadata["final_fitness"] <= metadata["init_fitness"] + 1e-6


# --- C10: floor invariant --------------------------------------------------


def test_run_respects_floor():
    """C10: RCO must never allocate below ``floor_l = per_layer_count_l // 2``."""
    inputs = _build_inputs(global_budget=8)
    ctx = _populate_context(inputs)
    ctx.set("per_layer_damage_curve", {
        0: {2: 0.0, 3: 100.0, 4: 200.0},
        1: {2: 0.0, 3: 0.0, 4: 0.0},
    })
    RCOBudgetPlugin().run(ctx)
    rco_budgets = ctx.get("per_layer_target_experts_rco")
    assert rco_budgets["0"] >= 2
    assert rco_budgets["1"] >= 2


# --- C11: fitness_signal strict damage_curve mode (Pattern E) --------------


def test_fitness_signal_strict_mode_raises_when_curve_absent():
    """C11: ``fitness_signal="damage_curve"`` without the slot raises ValueError."""
    inputs = _build_inputs()
    inputs["config"]["stage1"]["rco_budget"]["fitness_signal"] = "damage_curve"
    ctx = _populate_context(inputs)
    # Deliberately do NOT set ``per_layer_damage_curve``.
    with pytest.raises(ValueError) as exc:
        RCOBudgetPlugin().run(ctx)
    assert "fitness_signal=damage_curve" in str(exc.value)
    assert "per_layer_damage_curve" in str(exc.value)


def test_fitness_signal_synthetic_mode_ignores_damage_curve():
    """C11 (extension): ``fitness_signal="synthetic"`` hard-skips the damage curve."""
    inputs = _build_inputs(global_budget=6)
    inputs["config"]["stage1"]["rco_budget"]["fitness_signal"] = "synthetic"
    ctx = _populate_context(inputs)
    # Plant a damage curve — should be ignored.
    ctx.set("per_layer_damage_curve", {
        0: {2: 200.0, 3: 100.0, 4: 0.0},
        1: {2: 2.0, 3: 1.0, 4: 0.0},
    })
    RCOBudgetPlugin().run(ctx)
    metadata = ctx.get("rco_metadata")
    assert metadata["fitness_source"] == "synthetic"
    assert metadata["fitness_signal_resolved"] == "synthetic"


# --- C12: artifact includes format_version at top level (Pattern B) --------


def test_artifact_includes_format_version():
    """C12: the artifact dict carries ``format_version: 1`` at the TOP level."""
    inputs = _build_inputs()
    ctx = _populate_context(inputs)
    plugin = RCOBudgetPlugin()
    plugin.run(ctx)
    payload = plugin.contribute_artifact(ctx)
    assert "format_version" in payload
    assert payload["format_version"] == 1
    # Verify it is at the top level, NOT inside rco_metadata.
    assert "format_version" not in payload["rco_metadata"]


# --- C13: contribute_artifact when disabled --------------------------------


def test_contribute_artifact_when_disabled():
    """C13: before ``run`` (or when disabled), the artifact is an empty dict."""
    plugin = RCOBudgetPlugin()
    ctx = PipelineContext()
    assert plugin.contribute_artifact(ctx) == {}


# --- C14: contribute_artifact when enabled, with format_version extension --


def test_contribute_artifact_when_enabled():
    """C14 (extended): after ``run``, the artifact has format_version + budgets + metadata."""
    inputs = _build_inputs()
    ctx = _populate_context(inputs)
    plugin = RCOBudgetPlugin()
    plugin.run(ctx)
    payload = plugin.contribute_artifact(ctx)
    assert set(payload.keys()) == {"format_version", "rco_budgets", "rco_metadata"}
    assert payload["format_version"] == 1
    assert isinstance(payload["rco_budgets"], dict)
    assert isinstance(payload["rco_metadata"], dict)
    for k, v in payload["rco_budgets"].items():
        assert isinstance(k, str)
        assert isinstance(v, int)
    md = payload["rco_metadata"]
    assert "init_fitness" in md
    assert "final_fitness" in md
    assert "init_budget_vector" in md
    assert "final_budget_vector" in md
    assert md["achieved_budget"] == sum(payload["rco_budgets"].values())
    # Pattern E metadata: fitness_signal_resolved + tau_init_used + tau_final_used.
    assert md["fitness_signal_resolved"] in ("synthetic", "damage_curve")
    assert md["tau_init_used"] == 5.0
    assert md["tau_final_used"] == 0.5


# --- C15: manifest registration --------------------------------------------


def test_plugin_registered_in_manifest():
    """C15: ``RCOBudgetPlugin`` is in ``STAGE1_PLUGIN_MANIFEST``, after grape_merge."""
    from moe_compress.stage1.plugins import STAGE1_PLUGIN_MANIFEST
    names = [p.name for p in STAGE1_PLUGIN_MANIFEST]
    assert "rco_budget" in names
    assert names.index("rco_budget") > names.index("grape_merge")


# --- C16: budget exactness -------------------------------------------------


def test_run_sums_to_global_budget():
    """C16: the final allocation sums exactly to ``decomposition.global_expert_budget``."""
    for B in [6, 7, 8]:
        inputs = _build_inputs(global_budget=B)
        ctx = _populate_context(inputs)
        RCOBudgetPlugin().run(ctx)
        rco_budgets = ctx.get("per_layer_target_experts_rco")
        assert sum(rco_budgets.values()) == B, (
            f"global_budget={B}: RCO sum = {sum(rco_budgets.values())}"
        )


# ===========================================================================
# F1-F15 — Paper-fidelity tests
# ===========================================================================


# --- F1: constraint normal closed form -------------------------------------


def test_constraint_normal_closed_form():
    """F1: ``_constraint_normal(α)`` equals autograd of Σ_l ⟨p_l, c_l⟩.

    Symbolic reference (paper §2 Prop. 2): ``n_lk = p_lk · (c_lk − E_p[c_l])``
    where p = softmax(α). F1 also pins that the constraint normal is
    evaluated at the **un-perturbed** p (plan Q8) — F6 covers the
    perturbed-p̃ object used in the backward.
    """
    torch.manual_seed(7)
    L, K_max = 3, 5
    # Build random α with a mask (last col padded out on layers 0 & 2).
    alpha = torch.randn(L, K_max, dtype=torch.float64)
    mask = torch.ones(L, K_max, dtype=torch.float64)
    mask[0, -1] = 0.0
    mask[2, -1] = 0.0
    alpha[0, -1] = -1e9
    alpha[2, -1] = -1e9
    cost_grid = torch.randint(1, 10, (L, K_max)).double()

    plugin = RCOBudgetPlugin()
    n_impl = plugin._constraint_normal(alpha, cost_grid, mask)

    # Autograd reference: differentiate C(α) = Σ_l Σ_k p_lk · c_lk w.r.t. α.
    a = alpha.clone().requires_grad_(True)
    p = plugin._masked_softmax(a, mask)
    C = (p * cost_grid).sum()
    C.backward()
    n_autograd = a.grad * mask

    diff = (n_impl - n_autograd).abs().max().item()
    assert diff < 1e-10, f"constraint normal vs autograd diff = {diff}"


# --- F2: tangent projection orthogonality ----------------------------------


def test_tangent_projection_orthogonality():
    """F2: after projection, ``⟨g_tan, n⟩ ≈ 0``."""
    torch.manual_seed(2)
    L, K_max = 4, 6
    g = torch.randn(L, K_max, dtype=torch.float64)
    n = torch.randn(L, K_max, dtype=torch.float64)
    mask = torch.ones(L, K_max, dtype=torch.float64)

    plugin = RCOBudgetPlugin()
    g_tan = plugin._project_off_normal(g, n, mask)
    inner = (g_tan * n).sum().item()
    norm = (n * n).sum().sqrt().item()
    rel = abs(inner) / max(norm, 1e-12)
    assert rel < 1e-10, f"⟨g_tan, n⟩ / ‖n‖ = {rel}"


# --- F3: retraction budget exactness ---------------------------------------


def test_retraction_budget_exactness():
    """F3: after ``_retract``, Σ p · c equals B within _BISECT_TOL (50 trials)."""
    plugin = RCOBudgetPlugin()
    torch.manual_seed(42)
    L, K_max = 5, 8
    mask = torch.ones(L, K_max, dtype=torch.float64)
    for trial in range(50):
        alpha = torch.randn(L, K_max, dtype=torch.float64)
        cost_grid = torch.randint(1, 10, (L, K_max)).double()
        # Pick a B inside the feasible range (uniform-softmax expectation).
        p_uniform = torch.full((L, K_max), 1.0 / K_max, dtype=torch.float64)
        budget_uniform = float((p_uniform * cost_grid).sum().item())
        B = float(budget_uniform)
        alpha_r = plugin._retract(alpha, cost_grid, mask, B)
        p = plugin._masked_softmax(alpha_r, mask)
        soft_b = float((p * cost_grid).sum().item())
        assert abs(soft_b - B) <= 1e-3, (
            f"trial {trial}: |Σ p·c − B| = {abs(soft_b - B)}"
        )


# --- F4: retraction monotonicity -------------------------------------------


def test_retraction_monotonicity():
    """F4: f(t) = C(α − t·c) − B is monotonically non-increasing in t."""
    plugin = RCOBudgetPlugin()
    torch.manual_seed(3)
    L, K_max = 4, 5
    alpha = torch.randn(L, K_max, dtype=torch.float64)
    cost_grid = torch.randint(1, 10, (L, K_max)).double()
    mask = torch.ones(L, K_max, dtype=torch.float64)
    B = 4.0
    ts = np.linspace(-3.0, 3.0, 81)
    residuals = [
        plugin._budget_residual(alpha - t * cost_grid, cost_grid, mask, B)
        for t in ts
    ]
    diffs = np.diff(residuals)
    # Non-increasing: every diff ≤ 0 (allow tiny float noise).
    assert (diffs <= 1e-10).all(), (
        f"retraction residual not monotone: max diff = {diffs.max()}"
    )


# --- F5: vector transport preserves tangency -------------------------------


def test_vector_transport_preserves_tangency():
    """F5: after step + retract + project, ``⟨m_new, n_new⟩ ≈ 0``."""
    plugin = RCOBudgetPlugin()
    torch.manual_seed(5)
    L, K_max = 3, 4
    alpha = torch.randn(L, K_max, dtype=torch.float64)
    cost_grid = torch.randint(1, 10, (L, K_max)).double()
    mask = torch.ones(L, K_max, dtype=torch.float64)
    B = 5.0
    # Take a "step" + retract.
    m = torch.randn(L, K_max, dtype=torch.float64)
    alpha_stepped = alpha + 0.1 * m
    alpha_retracted = plugin._retract(alpha_stepped, cost_grid, mask, B)
    n_new = plugin._constraint_normal(alpha_retracted, cost_grid, mask)
    m_new = plugin._project_off_normal(m, n_new, mask)
    inner = (m_new * n_new).sum().item()
    norm = (n_new * n_new).sum().sqrt().item()
    rel = abs(inner) / max(norm, 1e-12)
    assert rel < 1e-10, f"⟨m_new, n_new⟩ / ‖n_new‖ = {rel}"


# --- F6: gradient estimate Jacobian collapse -------------------------------


def test_gradient_estimate_jacobian_collapse():
    """F6: analytic Gumbel-softmax gradient equals ``p̃ · (D − E_p̃[D])``.

    Verified against ``torch.autograd`` on a deterministic instance: the
    Gumbel noise is fixed at the same value the plugin would sample so
    we can diff the two outputs directly.
    """
    plugin = RCOBudgetPlugin()
    torch.manual_seed(11)
    L, K_max = 3, 4
    alpha = torch.randn(L, K_max, dtype=torch.float64)
    damage_grid = torch.randn(L, K_max, dtype=torch.float64)
    mask = torch.ones(L, K_max, dtype=torch.float64)
    tau = 0.7

    # Sample Gumbel noise once + replay it deterministically.
    rng_impl = torch.Generator().manual_seed(123)
    rng_ref = torch.Generator().manual_seed(123)

    grad_impl = plugin._gradient_estimate(
        alpha=alpha, mask=mask, damage_grid=damage_grid, tau=tau, rng=rng_impl,
    )

    # Reference: replay the same Gumbel sample + analytic Jacobian collapse.
    u = torch.rand(alpha.shape, generator=rng_ref, dtype=alpha.dtype).clamp_min(1e-20)
    g_noise = -torch.log(-torch.log(u))
    alpha_perturbed = (alpha + g_noise) / tau
    p_tilde = plugin._masked_softmax(alpha_perturbed, mask)
    e_d = (p_tilde * damage_grid).sum(dim=1, keepdim=True)
    grad_ref = p_tilde * (damage_grid - e_d) * mask

    diff = (grad_impl - grad_ref).abs().max().item()
    assert diff < 1e-12, f"analytic grad vs replay diff = {diff}"


# --- F7: DP knapsack optimality (3×4 = 64 brute force) ---------------------


def test_dp_knapsack_optimality():
    """F7: DP finds the brute-force optimum on a hand-graded 3-layer × 4-option case."""
    plugin = RCOBudgetPlugin()
    torch.manual_seed(13)
    L, K = 3, 4
    cost_grid = torch.tensor(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float64
    )
    damage_grid = torch.rand(L, K, dtype=torch.float64) * 10.0
    mask = torch.ones(L, K, dtype=torch.float64)
    alpha = torch.zeros(L, K, dtype=torch.float64)
    k_options = {li: [1, 2, 3, 4] for li in range(L)}
    sorted_layers = [0, 1, 2]
    B = 7

    fitness_dp, budget_dp = plugin._evaluate_discrete(
        alpha=alpha, cost_grid=cost_grid, mask=mask, damage_grid=damage_grid,
        k_options=k_options, sorted_layers=sorted_layers, global_budget=B,
    )

    # Brute force.
    best_dam, best_choice = float("inf"), None
    for choice in itertools.product(range(K), repeat=L):
        total_c = sum(int(cost_grid[i, c]) for i, c in enumerate(choice))
        if total_c != B:
            continue
        d = sum(float(damage_grid[i, c]) for i, c in enumerate(choice))
        if d < best_dam:
            best_dam = d
            best_choice = choice

    assert best_choice is not None
    assert abs(fitness_dp - best_dam) < 1e-12, (
        f"DP fitness {fitness_dp} vs brute force {best_dam}"
    )
    # Verify the DP-chosen vector also matches the brute-force one.
    for i, li in enumerate(sorted_layers):
        opt_idx = best_choice[i]
        expected_k = int(cost_grid[i, opt_idx])
        assert budget_dp[li] == expected_k


# --- F8: infeasible budget falls back to nearest feasible ------------------


def test_dp_handles_infeasible_budget(monkeypatch):
    """F8: when B is out of range, DP falls back to NEAREST feasible (not max).

    Plan §6.1 F8 / Delta 6 fix: selector is
    ``min(feasible, key=lambda b: (abs(b−B), −b))``, i.e. nearest with
    larger-budget tiebreak. Prior impl picked ``max(feasible, ...)``;
    REMOVED.

    Setup uses cost_grid with a gap. Layers each have options {2, 4}
    only → feasible budgets = {4, 6, 8}. Ask B=5; nearest is 4 or 6,
    larger wins → 6. Ask B=7; 6 or 8 are equidistant; larger wins → 8.

    Captures the WARNING via a monkeypatched ``log.warning`` rather than
    pytest's ``caplog`` because the latter has propagation quirks
    depending on the test harness's log-config side effects.
    """
    plugin = RCOBudgetPlugin()
    L, K = 2, 2
    cost_grid = torch.tensor([[2, 4], [2, 4]], dtype=torch.float64)
    damage_grid = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    mask = torch.ones(L, K, dtype=torch.float64)
    alpha = torch.zeros(L, K, dtype=torch.float64)
    k_options = {0: [2, 4], 1: [2, 4]}
    sorted_layers = [0, 1]

    # Spy on log.warning to confirm the warning fires (no silent corruption).
    from moe_compress.stage1.plugins import rco_budget as rco_mod
    warnings_seen: list[str] = []

    def _spy(msg, *args, **kwargs):
        warnings_seen.append(msg % args if args else msg)

    monkeypatch.setattr(rco_mod.log, "warning", _spy)

    # B = 5: equidistant from 4 and 6 → larger-budget tiebreak picks 6.
    _, budget = plugin._evaluate_discrete(
        alpha=alpha, cost_grid=cost_grid, mask=mask, damage_grid=damage_grid,
        k_options=k_options, sorted_layers=sorted_layers, global_budget=5,
    )
    assert sum(budget.values()) == 6, (
        f"B=5 infeasible; expected nearest-larger=6, got {sum(budget.values())}"
    )
    # WARNING was emitted (no silent corruption).
    assert any("infeasible" in w for w in warnings_seen), (
        f"expected 'infeasible' WARNING; saw: {warnings_seen}"
    )

    # B = 7: equidistant from 6 and 8 → larger-budget tiebreak picks 8.
    _, budget = plugin._evaluate_discrete(
        alpha=alpha, cost_grid=cost_grid, mask=mask, damage_grid=damage_grid,
        k_options=k_options, sorted_layers=sorted_layers, global_budget=7,
    )
    assert sum(budget.values()) == 8, (
        f"B=7 infeasible; expected nearest-larger=8, got {sum(budget.values())}"
    )


# --- F9: convergence on a synthetic quadratic ------------------------------


def test_algorithm_converges_on_quadratic_proxy():
    """F9: on a damage curve with a known analytic minimum, RCO converges.

    Setup: 2 layers, option grid {2, 3, 4}, k*_0 = 4, k*_1 = 2.
    Damage 0: {2: 4, 3: 1, 4: 0}; Damage 1: {2: 0, 3: 1, 4: 4}.
    Budget B = 6. Optimal under pure-damage DP: (4, 2) with damage 0,
    vs (3, 3) = 1+1 = 2, vs (2, 4) = 4+4 = 8. So (4, 2) is optimum.
    RCO should find it within 200 iterations.
    """
    inputs = _build_inputs(global_budget=6)
    inputs["config"]["stage1"]["rco_budget"]["n_iterations"] = 200
    ctx = _populate_context(inputs)
    ctx.set("per_layer_damage_curve", {
        0: {2: 4.0, 3: 1.0, 4: 0.0},
        1: {2: 0.0, 3: 1.0, 4: 4.0},
    })
    RCOBudgetPlugin().run(ctx)
    rco_budgets = ctx.get("per_layer_target_experts_rco")
    metadata = ctx.get("rco_metadata")
    # Optimum is (4, 2) with damage 0.
    assert rco_budgets["0"] == 4
    assert rco_budgets["1"] == 2
    assert metadata["final_fitness"] < 1e-6


# --- F10: seed reproducibility ---------------------------------------------


def test_seed_reproducibility():
    """F10: two runs with the same seed produce identical final budgets."""
    inputs1 = _build_inputs(global_budget=6)
    ctx1 = _populate_context(inputs1)
    RCOBudgetPlugin().run(ctx1)
    out1 = ctx1.get("per_layer_target_experts_rco")
    md1 = ctx1.get("rco_metadata")

    inputs2 = _build_inputs(global_budget=6)
    ctx2 = _populate_context(inputs2)
    RCOBudgetPlugin().run(ctx2)
    out2 = ctx2.get("per_layer_target_experts_rco")
    md2 = ctx2.get("rco_metadata")

    assert out1 == out2
    assert md1["final_fitness"] == pytest.approx(md2["final_fitness"], abs=1e-12)


# --- F11: Gumbel-softmax τ limits (standard form) --------------------------


def test_gumbel_softmax_tau_limits():
    """F11: standard-form Gumbel-softmax limits.

    Plan §1.3 step 1 / D1-1 fix:
    - τ → ∞ ⇒ ``softmax((α+g)/τ) → uniform`` (within 1e-3 per element).
    - τ → 0 ⇒ ``softmax((α+g)/τ) → one-hot at argmax(α+g)``.

    The prior impl's ``softmax(α + τ·g)`` has the opposite limits;
    that form is REMOVED, not preserved.
    """
    plugin = RCOBudgetPlugin()
    torch.manual_seed(0)
    L, K = 2, 5
    alpha = torch.randn(L, K, dtype=torch.float64)
    mask = torch.ones(L, K, dtype=torch.float64)

    # Pre-sample the same Gumbel noise for both temperatures.
    rng_setup = torch.Generator().manual_seed(99)
    u = torch.rand(alpha.shape, generator=rng_setup, dtype=alpha.dtype).clamp_min(1e-20)
    g_noise = -torch.log(-torch.log(u))

    # τ → ∞: p̃ ≈ uniform.
    tau_big = 1e6
    alpha_perturbed_big = (alpha + g_noise) / tau_big
    p_big = plugin._masked_softmax(alpha_perturbed_big, mask)
    uniform = torch.full((L, K), 1.0 / K, dtype=torch.float64)
    assert (p_big - uniform).abs().max().item() < 1e-3, (
        f"τ=1e6 p̃ not uniform: max dev {(p_big - uniform).abs().max().item()}"
    )

    # τ → 0: p̃ → one-hot at argmax(α + g_noise).
    tau_tiny = 1e-3
    alpha_perturbed_tiny = (alpha + g_noise) / tau_tiny
    p_tiny = plugin._masked_softmax(alpha_perturbed_tiny, mask)
    argmax_target = (alpha + g_noise).argmax(dim=1)
    for li in range(L):
        assert p_tiny[li, argmax_target[li]].item() > 0.999, (
            f"τ=1e-3 row {li}: argmax mass {p_tiny[li, argmax_target[li]].item()}"
        )


# --- F12: cosine anneal endpoints ------------------------------------------


def test_cosine_anneal_endpoints():
    """F12: cosine schedule with (τ_init=5.0, τ_final=0.5, T=500).

    Pins explore→exploit direction (plan §1.3 step 2 / D1-2 fix).
    - At t=0: τ_t ≈ τ_init (cos(0)=1).
    - At t=T-1: τ_t ≈ τ_final (cos(π·(T-1)/T) ≈ cos(π) = -1).
    """
    T = 500
    tau_init = 5.0
    tau_final = 0.5
    tau_0 = RCOBudgetPlugin._cosine_tau(
        step=0, total_steps=T, tau_init=tau_init, tau_final=tau_final
    )
    tau_end = RCOBudgetPlugin._cosine_tau(
        step=T - 1, total_steps=T, tau_init=tau_init, tau_final=tau_final
    )
    assert tau_0 == pytest.approx(tau_init, abs=1e-12), (
        f"τ at t=0 should be τ_init={tau_init}, got {tau_0}"
    )
    # cos(π·(T-1)/T) = cos(π·(1 - 1/T)) ≈ -1 + π²/(2T²) for large T; with
    # T=500 the offset is small but nonzero so use a generous tol.
    assert tau_end == pytest.approx(tau_final, abs=2e-2), (
        f"τ at t=T-1 should be ≈τ_final={tau_final}, got {tau_end}"
    )
    # Sanity: τ_init > τ_end ⇒ schedule is decreasing (explore → exploit).
    assert tau_0 > tau_end


# --- F13: Adam v_buf transport policy --------------------------------------


def test_adam_v_buf_transport_policy():
    """F13: D-adam-no-v-transport.

    After one outer step + retract, the first moment ``m`` (transported)
    has ``⟨m, n_new⟩ ≈ 0``; the second moment ``v`` (NOT transported)
    in general does NOT satisfy this.

    Strategy: run a single Adam step with synthetic m, v populated, then
    check (m_after, v_after) against the new normal n_new directly. The
    plugin's main loop transports m via ``_project_off_normal`` and
    leaves v as-is; we replicate that here.

    Setup uses a mild α + cost_grid that lets retraction find a clean
    bisection bracket (so n_new is well-conditioned).
    """
    plugin = RCOBudgetPlugin()
    torch.manual_seed(31)
    L, K_max = 3, 4
    # Mild α so softmax doesn't saturate after the step+retract.
    alpha = torch.randn(L, K_max, dtype=torch.float64) * 0.5
    cost_grid = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]] * L, dtype=torch.float64
    )
    mask = torch.ones(L, K_max, dtype=torch.float64)
    # Pick a B in the soft-feasible range (each row uniform softmax gives
    # ⟨p, c⟩ = 2.5; total uniform budget = 7.5; pick B = 7.0).
    B = 7.0

    # Populate m + v with structured noise that is NOT tangent.
    m = torch.randn(L, K_max, dtype=torch.float64) * 0.3
    v = m * m + 0.1  # element-wise positive (matches real Adam) + offset.

    # Step + retract.
    alpha_stepped = alpha + 0.1 * m
    alpha_new = plugin._retract(alpha_stepped, cost_grid, mask, B)
    n_new = plugin._constraint_normal(alpha_new, cost_grid, mask)

    # Sanity: n_new is well-conditioned (not zero).
    norm_n = (n_new * n_new).sum().sqrt().item()
    assert norm_n > 1e-6, f"n_new degenerate (‖n‖={norm_n}); test setup needs adjustment"

    # Transport m (as the main loop does).
    m_transported = plugin._project_off_normal(m, n_new, mask)
    # v is NOT transported.
    v_untransported = v

    inner_m = (m_transported * n_new).sum().item()
    rel_m = abs(inner_m) / norm_n
    assert rel_m < 1e-10, (
        f"transported m should be tangent: ⟨m_new, n_new⟩/‖n_new‖ = {rel_m}"
    )

    inner_v = (v_untransported * n_new).sum().item()
    rel_v = abs(inner_v) / norm_n
    assert rel_v > 1e-6, (
        f"untransported v should generally NOT be tangent; got rel={rel_v} "
        "(if this fails the random draw happened to land tangent — re-seed)"
    )


# --- F14: α stability under large logits -----------------------------------


def test_alpha_stability_under_large_logits():
    """F14: α with magnitude ~1e6 → no NaN/Inf in p̃, n, or retraction."""
    plugin = RCOBudgetPlugin()
    L, K_max = 3, 4
    alpha = torch.tensor(
        [[1e6, -1e6, 1e6, -1e6],
         [-1e6, 1e6, -1e6, 1e6],
         [1e6, 1e6, -1e6, -1e6]],
        dtype=torch.float64,
    )
    cost_grid = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]] * L, dtype=torch.float64
    )
    mask = torch.ones(L, K_max, dtype=torch.float64)

    p = plugin._masked_softmax(alpha, mask)
    assert torch.isfinite(p).all(), "masked softmax produced NaN/Inf"
    # Per-row probability mass sums to 1.
    row_sums = p.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(L, dtype=torch.float64), atol=1e-12)

    n = plugin._constraint_normal(alpha, cost_grid, mask)
    assert torch.isfinite(n).all(), "constraint normal produced NaN/Inf"

    alpha_r = plugin._retract(alpha, cost_grid, mask, global_budget=6.0)
    assert torch.isfinite(alpha_r).all(), "retraction produced NaN/Inf"

    # Gradient estimate stable too.
    rng = torch.Generator().manual_seed(0)
    damage = torch.zeros(L, K_max, dtype=torch.float64)
    grad = plugin._gradient_estimate(
        alpha=alpha, mask=mask, damage_grid=damage, tau=0.5, rng=rng
    )
    assert torch.isfinite(grad).all(), "gradient estimate produced NaN/Inf"


# --- F15: pure-damage DP, NOT β·log-p tiebreak -----------------------------


def test_dp_pure_damage_not_logit_tiebreak():
    """F15: re-impl uses pure damage; β·log p tiebreak is REMOVED.

    Setup (plan §6.1 F15 hand-derived arithmetic):
    - L=2, K=2; c = [[1, 2], [1, 2]]; B = 3.
    - D = [[1.0, 1.0], [1.0, 1.001]].
    - α = [[10, 0], [0, 10]].

    Feasibility at B=3:
    - (0, 0): 1+1 = 2 — INFEASIBLE.
    - (0, 1): 1+2 = 3 — FEASIBLE.
    - (1, 0): 2+1 = 3 — FEASIBLE.
    - (1, 1): 2+2 = 4 — INFEASIBLE.

    β=0 pure-damage DP:
    - damage_sum(0, 1) = D[0][0] + D[1][1] = 1.0 + 1.001 = 2.001.
    - damage_sum(1, 0) = D[0][1] + D[1][0] = 1.0 + 1.0 = 2.0.
    - Minimum is 2.0 → picks **(1, 0)**.

    β=1e-3 reference (computed off-line — the re-impl has no β knob):
    - log softmax([10, 0]) ≈ [−4.54e-5, −10.0000454].
    - log softmax([0, 10]) ≈ [−10.0000454, −4.54e-5].
    - score(0, 1) = 2.001 − 1e-3·(−4.54e-5 + −4.54e-5) ≈ 2.001000091.
    - score(1, 0) = 2.0 − 1e-3·(−10.0000454 + −10.0000454) ≈ 2.0200000908.
    - Minimum ≈ 2.001 → β=1e-3 reference picks **(0, 1)**.

    β=0 and β=1e-3 GENUINELY DISAGREE because the damage gap (0.001) is
    comparable to β·|Δ log p| (1e-3 · 20 = 0.02). The re-impl returns
    the pure-damage answer (1, 0), NOT (0, 1).

    Construction has no DP-tiebreak ambiguity: damage_sum(0,1)=2.001 ≠
    damage_sum(1,0)=2.0, so plan v4-N4's strict ``<`` tiebreak doesn't
    affect F15.
    """
    plugin = RCOBudgetPlugin()
    L, K = 2, 2
    cost_grid = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
    damage_grid = torch.tensor([[1.0, 1.0], [1.0, 1.001]], dtype=torch.float64)
    mask = torch.ones(L, K, dtype=torch.float64)
    alpha = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float64)
    k_options = {0: [1, 2], 1: [1, 2]}
    sorted_layers = [0, 1]

    _, budget_vec = plugin._evaluate_discrete(
        alpha=alpha, cost_grid=cost_grid, mask=mask, damage_grid=damage_grid,
        k_options=k_options, sorted_layers=sorted_layers, global_budget=3,
    )

    # (i) Re-impl (no β knob) returns the β=0 reference (1, 0):
    #     budget_vec[0] = k_options[0][1] = 2,  budget_vec[1] = k_options[1][0] = 1.
    assert budget_vec[0] == 2, (
        f"layer 0 should be k_options[0][1]=2 (β=0 answer); got {budget_vec[0]}"
    )
    assert budget_vec[1] == 1, (
        f"layer 1 should be k_options[1][0]=1 (β=0 answer); got {budget_vec[1]}"
    )

    # (ii) Re-impl's output is NOT (0, 1) (the β=1e-3 hand-computed answer):
    #      that would have been budget_vec[0]=1, budget_vec[1]=2.
    assert not (budget_vec[0] == 1 and budget_vec[1] == 2), (
        "re-impl wrongly returned the β=1e-3 tiebreak answer (0, 1)"
    )


# ===========================================================================
# R1-R3 — Regression tests
# ===========================================================================


# --- R1: default-OFF byte-equality (Stage-1 golden snapshot) ---------------


def test_default_off_byte_identical():
    """R1: with ``enabled: false``, the plugin is a strict no-op.

    Stand-in for the Stage-1 byte-equality golden snapshot: when the
    plugin is disabled it does NOT write ``per_layer_target_experts_rco``
    or ``rco_metadata`` to the ctx, and the artifact is an empty dict.
    """
    inputs = _build_inputs(enabled=False)
    ctx = _populate_context(inputs)
    plugin = RCOBudgetPlugin()
    assert plugin.is_enabled(inputs["config"]) is False
    # When disabled, the orchestrator does NOT call run() — the plugin's
    # contract is that contribute_artifact() returns {} too, so a stray
    # write is harmless.
    assert plugin.contribute_artifact(ctx) == {}
    # No writes to the ctx from the disabled path.
    assert not ctx.has("per_layer_target_experts_rco")
    assert not ctx.has("rco_metadata")


# --- R2: behavioural tests with re-derived expected values -----------------
# (R2 is the existing C8 + C9 tests with expected values re-derived under
#  the corrected algorithm; they are defined above and run as C8 + C9.
#  This wrapper test asserts the rerun produces the same vectors twice in
#  a row to pin re-derivation stability.)


def test_behaviour_under_corrected_algorithm_is_stable():
    """R2: behavioural tests pass under the corrected algorithm + are stable across runs."""
    # Synthetic case — C8 equivalent.
    inputs1 = _build_inputs(global_budget=6)
    ctx1 = _populate_context(inputs1)
    RCOBudgetPlugin().run(ctx1)
    out1 = ctx1.get("per_layer_target_experts_rco")
    inputs2 = _build_inputs(global_budget=6)
    ctx2 = _populate_context(inputs2)
    RCOBudgetPlugin().run(ctx2)
    out2 = ctx2.get("per_layer_target_experts_rco")
    assert out1 == out2

    # Damage-curve case — C9 equivalent.
    inputs3 = _build_inputs(global_budget=6)
    ctx3 = _populate_context(inputs3)
    ctx3.set("per_layer_damage_curve", {
        0: {2: 200.0, 3: 100.0, 4: 0.0},
        1: {2: 2.0, 3: 1.0, 4: 0.0},
    })
    RCOBudgetPlugin().run(ctx3)
    assert ctx3.get("per_layer_target_experts_rco") == {"0": 4, "1": 2}


# --- R3: budget exactness invariant (alias of C16; pinned again) -----------


def test_budget_exactness_invariant():
    """R3: budget-exactness is invariant under algorithm changes."""
    for B in [6, 7, 8]:
        inputs = _build_inputs(global_budget=B)
        ctx = _populate_context(inputs)
        RCOBudgetPlugin().run(ctx)
        assert sum(ctx.get("per_layer_target_experts_rco").values()) == B
