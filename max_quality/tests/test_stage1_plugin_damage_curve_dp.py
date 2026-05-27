"""Unit + integration tests for ``moe_compress.stage1.plugins.damage_curve_dp``.

Verifies:

1. Plugin Protocol attributes match the contract.
2. ``is_enabled`` correctly gates on the YAML knob.
3. Damage curve construction on hand-checked synthetic distance matrices.
4. DP knapsack on a small hand-checked instance with a known optimum.
5. Marginal-prior derivation: at-optimum prior == ``D(k*+1) − D(k*)``;
   at-floor layers get ``+inf``; zero marginals get clamped to
   ``_PRIOR_EPS``.
6. Plugin enabled/disabled gating end-to-end on a PipelineContext.
7. Integration with ``GrapeMergePlugin``: when enabled, the published
   prior makes GRAPE's selection differ from the un-biased baseline.
8. Plugin appears in ``STAGE1_PLUGIN_MANIFEST`` in the correct slot
   (after ``cka_distance``, before ``grape_merge``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from moe_compress.budget.solver import BudgetDecomposition
from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.plugin import PipelinePlugin
from moe_compress.stage1.plugins import STAGE1_PLUGIN_MANIFEST
from moe_compress.stage1.plugins.damage_curve_dp import (
    DamageCurveDpPlugin,
    _PRIOR_EPS,
    _build_damage_curves,
    _solve_knapsack_dp,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _decomposition(global_budget: int) -> BudgetDecomposition:
    return BudgetDecomposition(
        total_reduction_ratio=0.20,
        expert_prune_ratio=0.25,
        svd_rank_ratio=0.0,
        global_expert_budget=global_budget,
        min_experts_per_layer=2,
    )


def _make_ctx(
    *,
    D_matrices: dict[int, torch.Tensor],
    per_layer_counts: dict[int, int],
    blacklist: dict[int, list[int]] | None = None,
    global_budget: int,
    enabled: bool = True,
    floor_divisor: int = 2,
) -> PipelineContext:
    ctx = PipelineContext()
    ctx.set("D_matrices", D_matrices)
    ctx.set("blacklist", blacklist or {})
    ctx.set("per_layer_targets", per_layer_counts)
    ctx.set("decomposition", _decomposition(global_budget))
    cfg = {
        "stage1_grape": {
            "damage_curve_dp": {"enabled": enabled},
            "grape_floor_divisor": floor_divisor,
        }
    }
    ctx.set("config", cfg)
    return ctx


# ---------------------------------------------------------------------------
# 1. Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    plugin = DamageCurveDpPlugin()
    assert plugin.name == "damage_curve_dp"
    assert "arXiv:2308.10438" in plugin.paper          # R4 anchor
    assert "arXiv:2410.08589" in plugin.paper          # R8 anchor
    assert "D-cka-substitute-for-output-mse" in plugin.paper
    assert "D-dp-prior-as-marginal" in plugin.paper
    assert "D-prior-floor-eps" in plugin.paper
    assert plugin.config_key == "stage1_grape.damage_curve_dp.enabled"
    assert plugin.reads == (
        "D_matrices", "blacklist", "per_layer_targets",
        "decomposition", "config",
    )
    assert plugin.writes == (
        "damage_curves", "dp_optimum", "merge_cost_prior_computed",
    )
    assert plugin.provides == ()
    # Structural Protocol conformance (Python >= 3.12 also checks attrs).
    assert isinstance(plugin, PipelinePlugin)


def test_plugin_in_manifest_between_cka_and_grape():
    names = [p.name for p in STAGE1_PLUGIN_MANIFEST]
    assert "damage_curve_dp" in names
    assert names.index("damage_curve_dp") == names.index("cka_distance") + 1
    assert names.index("damage_curve_dp") == names.index("grape_merge") - 1


# ---------------------------------------------------------------------------
# 2. is_enabled gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, False),
        ({"stage1_grape": {}}, False),
        ({"stage1_grape": {"damage_curve_dp": {}}}, False),
        ({"stage1_grape": {"damage_curve_dp": {"enabled": False}}}, False),
        ({"stage1_grape": {"damage_curve_dp": {"enabled": True}}}, True),
    ],
)
def test_is_enabled_gating(config, expected):
    plugin = DamageCurveDpPlugin()
    assert plugin.is_enabled(config) is expected


# ---------------------------------------------------------------------------
# 3. Damage curve formula on synthetic 4-layer × 8-expert distance matrices
# ---------------------------------------------------------------------------


def test_damage_curve_construction_4layers_8experts():
    """Hand-checked cumulative damage on 4 layers × 8 experts.

    For each layer we build an n=8 distance matrix with a known off-
    diagonal pattern, then verify the cumsum matches the sorted-
    ascending construction.
    """
    n = 8
    rng = np.random.default_rng(42)
    D_matrices: dict[int, torch.Tensor] = {}
    expected_pairs: dict[int, list[float]] = {}
    for li in range(4):
        # Random symmetric distance matrix, diagonal zero, entries in [0, 1].
        upper = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float64)
        upper = np.triu(upper, k=1)
        sym = upper + upper.T
        np.fill_diagonal(sym, 0.0)
        D_matrices[li] = torch.from_numpy(sym).float()
        # n*(n-1)/2 unique unordered pairs.
        iu, ju = np.triu_indices(n, k=1)
        pairs = sym[iu, ju].tolist()
        pairs.sort()
        expected_pairs[li] = pairs

    sorted_layers = [0, 1, 2, 3]
    blacklist: dict[int, list[int]] = {}
    k_max = {li: n - n // 2 for li in sorted_layers}     # 8 − 4 = 4

    curves = _build_damage_curves(
        D_matrices=D_matrices,
        blacklist=blacklist,
        sorted_layers=sorted_layers,
        k_max=k_max,
    )

    for li in sorted_layers:
        curve = curves[li]
        assert curve.shape == (k_max[li] + 1,)
        assert curve[0] == 0.0
        # Curve must equal cumsum of first k_max sorted pairs.
        expected_cum = np.cumsum(expected_pairs[li][: k_max[li]])
        np.testing.assert_allclose(curve[1:], expected_cum, atol=1e-6)
        # Monotone non-decreasing by construction.
        diffs = np.diff(curve)
        assert np.all(diffs >= -1e-9), f"layer {li} curve not monotone: {diffs}"


def test_damage_curve_excludes_blacklisted_experts():
    """Blacklisted experts must not contribute pairs to the cumsum."""
    n = 4
    # Distances: small values involve expert 0 (blacklist target).
    sym = torch.zeros(n, n)
    sym[0, 1] = sym[1, 0] = 0.01
    sym[0, 2] = sym[2, 0] = 0.02
    sym[0, 3] = sym[3, 0] = 0.03
    sym[1, 2] = sym[2, 1] = 0.5
    sym[1, 3] = sym[3, 1] = 0.6
    sym[2, 3] = sym[3, 2] = 0.7
    D_matrices = {0: sym}
    blacklist = {0: [0]}
    sorted_layers = [0]
    k_max = {0: 3}

    curves = _build_damage_curves(
        D_matrices=D_matrices,
        blacklist=blacklist,
        sorted_layers=sorted_layers,
        k_max=k_max,
    )
    curve = curves[0]
    # Only pairs (1,2), (1,3), (2,3) remain: distances 0.5, 0.6, 0.7
    # → cumsum at k=1,2,3 = 0.5, 1.1, 1.8.
    np.testing.assert_allclose(curve, [0.0, 0.5, 1.1, 1.8], atol=1e-6)


def test_damage_curve_pad_when_k_exceeds_available_pairs():
    """If k_max exceeds available pair count, curve plateaus at last value."""
    n = 3
    sym = torch.tensor([[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]])
    D_matrices = {0: sym}
    sorted_layers = [0]
    # Only 3 pairs available but ask for k_max=5.
    k_max = {0: 5}
    curves = _build_damage_curves(
        D_matrices=D_matrices, blacklist={}, sorted_layers=sorted_layers,
        k_max=k_max,
    )
    curve = curves[0]
    # Pairs sorted: 0.1, 0.2, 0.3 → cumsum 0.1, 0.3, 0.6 then plateau.
    np.testing.assert_allclose(curve, [0.0, 0.1, 0.3, 0.6, 0.6, 0.6], atol=1e-6)


# ---------------------------------------------------------------------------
# 4. DP knapsack on small hand-checked instance
# ---------------------------------------------------------------------------


def test_dp_knapsack_small_handchecked_optimum():
    """3 layers × 4 experts × budget=5 merges with a hand-checked optimum.

    Layer 0: D(k) = [0, 1, 3, 6, 10]   (steep)
    Layer 1: D(k) = [0, 0.5, 1, 1.5, 2.0] (cheap)
    Layer 2: D(k) = [0, 2, 5, 9, 14]   (steepest)

    Optimum for sum=5 merges: the DP should pick k_1=4 (cost 2.0) +
    k_0=1 (cost 1) + k_2=0 (cost 0) = 3.0, or k_1=3 (1.5) + k_0=2 (3) +
    k_2=0 (0) = 4.5, or k_1=4 + k_0=0 + k_2=1 = 4.0. Cheapest is
    k=(1, 4, 0) with total cost 3.0.
    """
    damage = {
        0: np.array([0.0, 1.0, 3.0, 6.0, 10.0]),
        1: np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        2: np.array([0.0, 2.0, 5.0, 9.0, 14.0]),
    }
    k_max = {0: 4, 1: 4, 2: 4}
    optimum = _solve_knapsack_dp(
        sorted_layers=[0, 1, 2],
        damage_curves=damage,
        k_max=k_max,
        global_merges=5,
    )
    assert sum(optimum.values()) == 5
    total_cost = sum(float(damage[li][optimum[li]]) for li in optimum)
    assert math.isclose(total_cost, 3.0, abs_tol=1e-9), \
        f"DP picked suboptimal allocation {optimum} with cost {total_cost}"
    assert optimum == {0: 1, 1: 4, 2: 0}


def test_dp_knapsack_respects_k_max():
    """No layer's k_ℓ exceeds its k_max."""
    damage = {
        0: np.array([0.0, 1.0]),
        1: np.array([0.0, 1.0, 2.0]),
        2: np.array([0.0, 1.0, 2.0, 3.0]),
    }
    k_max = {0: 1, 1: 2, 2: 3}
    optimum = _solve_knapsack_dp(
        sorted_layers=[0, 1, 2],
        damage_curves=damage,
        k_max=k_max,
        global_merges=6,
    )
    assert sum(optimum.values()) == 6
    for li, k in optimum.items():
        assert 0 <= k <= k_max[li], f"layer {li}: k={k} outside [0, {k_max[li]}]"


def test_dp_knapsack_zero_budget():
    """With global_merges=0, every layer's k* must be 0."""
    damage = {
        0: np.array([0.0, 1.0, 2.0]),
        1: np.array([0.0, 1.0, 2.0]),
    }
    k_max = {0: 2, 1: 2}
    optimum = _solve_knapsack_dp(
        sorted_layers=[0, 1],
        damage_curves=damage,
        k_max=k_max,
        global_merges=0,
    )
    assert optimum == {0: 0, 1: 0}


# ---------------------------------------------------------------------------
# 5. Plugin enabled/disabled gating end-to-end
# ---------------------------------------------------------------------------


def test_run_disabled_does_nothing():
    """When disabled, the plugin must write nothing to ctx or config."""
    n = 4
    D = torch.full((n, n), 0.3); D.fill_diagonal_(0.0)
    ctx = _make_ctx(
        D_matrices={0: D, 1: D.clone()},
        per_layer_counts={0: n, 1: n},
        global_budget=6,
        enabled=False,
    )
    plugin = DamageCurveDpPlugin()
    plugin.run(ctx)
    # ctx slots must NOT be written.
    assert "damage_curves" not in ctx
    assert "dp_optimum" not in ctx
    assert "merge_cost_prior_computed" not in ctx
    # config must NOT have merge_cost_prior set.
    s1 = ctx.get("config")["stage1_grape"]
    assert "merge_cost_prior" not in s1


def test_run_enabled_populates_ctx_and_config():
    """When enabled, run writes the three ctx slots + mutates the config."""
    n = 4
    # Distinct distance patterns per layer for a non-trivial DP.
    D0 = torch.tensor([
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.0, 0.4, 0.5],
        [0.2, 0.4, 0.0, 0.6],
        [0.3, 0.5, 0.6, 0.0],
    ])
    D1 = torch.tensor([
        [0.0, 0.8, 0.7, 0.9],
        [0.8, 0.0, 0.6, 0.7],
        [0.7, 0.6, 0.0, 0.5],
        [0.9, 0.7, 0.5, 0.0],
    ])
    ctx = _make_ctx(
        D_matrices={0: D0, 1: D1},
        per_layer_counts={0: n, 1: n},
        global_budget=6,    # 8 total experts - 6 = 2 merges across both layers
        enabled=True,
    )
    DamageCurveDpPlugin().run(ctx)

    curves = ctx.get("damage_curves")
    optimum = ctx.get("dp_optimum")
    prior = ctx.get("merge_cost_prior_computed")
    assert set(curves) == {0, 1}
    assert set(optimum) == {0, 1}
    assert set(prior) == {0, 1}
    # Total merges == 2.
    assert sum(optimum.values()) == 2
    # Prior is finite + non-negative + clamped to >= _PRIOR_EPS or +inf.
    for li, p in prior.items():
        assert p >= _PRIOR_EPS or p == math.inf
        assert p > 0.0

    # Config mutation: keys are string-indexed (matches GRAPE's contract).
    s1 = ctx.get("config")["stage1_grape"]
    assert "merge_cost_prior" in s1
    assert set(s1["merge_cost_prior"].keys()) == {"0", "1"}
    for k, v in s1["merge_cost_prior"].items():
        assert isinstance(k, str)
        assert isinstance(v, float)


def test_marginal_prior_matches_dp_traceback():
    """``prior_ℓ == D_ℓ(k*+1) − D_ℓ(k*)`` for layers below floor."""
    n = 4
    D0 = torch.tensor([
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.0, 0.4, 0.5],
        [0.2, 0.4, 0.0, 0.6],
        [0.3, 0.5, 0.6, 0.0],
    ])
    D1 = torch.tensor([
        [0.0, 0.8, 0.7, 0.9],
        [0.8, 0.0, 0.6, 0.7],
        [0.7, 0.6, 0.0, 0.5],
        [0.9, 0.7, 0.5, 0.0],
    ])
    ctx = _make_ctx(
        D_matrices={0: D0, 1: D1},
        per_layer_counts={0: n, 1: n},
        global_budget=6,
        enabled=True,
    )
    DamageCurveDpPlugin().run(ctx)
    curves = ctx.get("damage_curves")
    optimum = ctx.get("dp_optimum")
    prior = ctx.get("merge_cost_prior_computed")
    for li in optimum:
        k_star = optimum[li]
        curve = curves[li]
        k_max = len(curve) - 1
        if k_star >= k_max:
            assert prior[li] == math.inf
        else:
            expected = float(curve[k_star + 1] - curve[k_star])
            expected = max(expected, _PRIOR_EPS)
            assert math.isclose(prior[li], expected, abs_tol=1e-9)


def test_at_floor_layer_gets_inf_prior():
    """When DP places a layer at its k_max (= at the floor), prior = +inf."""
    # 2 layers, 4 experts each, floor_divisor=2 → floor=2 → k_max=2 per layer.
    # Budget=4 means 4 merges → forces both layers to k=k_max=2.
    n = 4
    D = torch.tensor([
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.0, 0.2, 0.3],
        [0.2, 0.2, 0.0, 0.3],
        [0.3, 0.3, 0.3, 0.0],
    ])
    ctx = _make_ctx(
        D_matrices={0: D, 1: D.clone()},
        per_layer_counts={0: n, 1: n},
        global_budget=4,    # 8 - 4 = 4 merges total, k_max=2 per layer
        enabled=True,
    )
    DamageCurveDpPlugin().run(ctx)
    prior = ctx.get("merge_cost_prior_computed")
    assert prior[0] == math.inf
    assert prior[1] == math.inf


def test_zero_marginal_clamped_to_eps():
    """A degenerate zero marginal must be clamped to ``_PRIOR_EPS``."""
    n = 4
    # All-zero off-diagonal: cumsum is 0 everywhere → marginal at any k = 0.
    D = torch.zeros(n, n)
    ctx = _make_ctx(
        D_matrices={0: D, 1: D.clone()},
        per_layer_counts={0: n, 1: n},
        global_budget=6,    # 2 merges, k_max=2 per layer → DP picks k<k_max somewhere
        enabled=True,
    )
    DamageCurveDpPlugin().run(ctx)
    prior = ctx.get("merge_cost_prior_computed")
    # At least one finite prior (we have feasibility).
    finite = [p for p in prior.values() if math.isfinite(p)]
    assert len(finite) >= 1
    # Every finite prior must be >= _PRIOR_EPS (clamped).
    for p in finite:
        assert p >= _PRIOR_EPS


def test_infeasible_target_falls_back_to_kmax():
    """When global_merges > Σ k_max, all layers pin at k_max and prior=+inf."""
    n = 4
    D = torch.full((n, n), 0.3); D.fill_diagonal_(0.0)
    ctx = _make_ctx(
        D_matrices={0: D, 1: D.clone()},
        per_layer_counts={0: n, 1: n},
        # floor=2 each → k_max=2 each → Σ k_max=4; budget=1 → merges=7 > 4.
        global_budget=1,
        enabled=True,
    )
    DamageCurveDpPlugin().run(ctx)
    optimum = ctx.get("dp_optimum")
    assert optimum == {0: 2, 1: 2}    # each pinned to k_max
    prior = ctx.get("merge_cost_prior_computed")
    assert all(p == math.inf for p in prior.values())


def test_zero_merges_target():
    """When the compression target is already met (global_merges=0), DP is trivial."""
    n = 4
    D = torch.full((n, n), 0.3); D.fill_diagonal_(0.0)
    ctx = _make_ctx(
        D_matrices={0: D, 1: D.clone()},
        per_layer_counts={0: n, 1: n},
        global_budget=8,    # 8 total experts; no merges needed
        enabled=True,
    )
    DamageCurveDpPlugin().run(ctx)
    optimum = ctx.get("dp_optimum")
    assert optimum == {0: 0, 1: 0}


# ---------------------------------------------------------------------------
# 6. Integration with GrapeMergePlugin
# ---------------------------------------------------------------------------


def test_grape_consumes_published_prior_when_enabled():
    """When S1_DP is enabled, GrapeMergePlugin reads its prior AND the
    prior actually changes GRAPE's allocation.

    The plugin mutates `config["stage1_grape"]["merge_cost_prior"]` and
    GRAPE's selection becomes ``argmin R[li] · prior[li]``. Concretely
    we verify:
      - The DP-published prior dict is present in the config when
        enabled, absent when disabled.
      - GRAPE runs to completion with the DP prior populated (no
        ``merge_cost_prior is missing entries`` ValueError).
      - GRAPE's final per-layer budget vector respects the floor and
        the global budget.
      - **The DP prior actually biases the allocation** —
        ``base_budgets != dp_budgets``. This guards against a future
        regression that silently neutralises the prior (e.g. dropping
        the config-side-channel write, mis-keying the dict).

    Uses ``entropy_tolerance = 0.1`` (the project default), NOT 1.0,
    so the entropy gate is active — disabling it would mask any
    regression that only manifested under realistic gamma.
    """
    from moe_compress.stage1.plugins.grape_merge import GrapeMergePlugin

    # Deterministic synthetic. Three layers, 8 experts each. Construct
    # the distance matrices so the un-biased GRAPE baseline (argmin R)
    # and the DP-biased GRAPE pick *different* layers for their merges:
    #
    #   * Layer 0: uniform distance 0.50 → R = 28 (smallest at iter 0
    #     ONLY among layers with no head outliers); D_0 cumsum is linear
    #     so DP gains nothing by merging here at low k.
    #   * Layer 1: 27 pair distances of 0.50 plus one tiny outlier
    #     (0.001) → R ≈ 27.002 (slightly smaller than layer 0's R, so
    #     baseline argmin-R picks it first). DP also wants to merge
    #     here (the head is cheap) and pins it at k_max=4 → +inf prior.
    #   * Layer 2: uniform 0.80 → R = 44.80 (largest); never picked.
    #
    # Effect under DP:
    #   - DP allocation = (0, 4, 0): layer 1 at k_max.
    #   - prior_0 = 0.50, prior_1 = +inf, prior_2 = 0.80.
    #   - GRAPE-biased scoring: R_0·0.50 = 14.0 (layer 0 wins);
    #     layer 1 gated out by +inf, layer 2 has 35.84.
    #   - Final DP-biased budgets: {0: 4, 1: 8, 2: 8}.
    # Effect under baseline (no prior, argmin R):
    #   - Iteration 0: layer 1 wins (R = 27.002 < 28).
    #   - GRAPE merges 4 in layer 1 (down to floor) using the cheap
    #     head pair first → final baseline budgets: {0: 8, 1: 4, 2: 8}.
    # The two allocations differ on layers 0 and 1 — the DP prior
    # gated out layer 1 (its "ideal" merge target per cheap head) and
    # forced GRAPE into layer 0.
    n = 8

    def _sym_from_upper(upper: np.ndarray) -> torch.Tensor:
        sym = upper + upper.T
        np.fill_diagonal(sym, 0.0)
        return torch.from_numpy(sym).float()

    L0 = np.triu(np.full((n, n), 0.50), k=1)

    # Layer 1: a CLUSTER of 4 tiny pair distances at the head, the
    # remaining 24 pairs at 0.50. The 4 tiny pairs make D_1's first
    # k_max entries near-zero so DP pins layer 1 at k_max=4 (+inf prior).
    # The tiny pairs also give layer 1 the smallest R, so the un-biased
    # baseline argmin-R selection picks layer 1 first and absorbs all
    # 4 merges there.
    L1 = np.triu(np.full((n, n), 0.50), k=1)
    L1[0, 1] = 0.001
    L1[0, 2] = 0.001
    L1[0, 3] = 0.001
    L1[1, 2] = 0.001

    L2 = np.triu(np.full((n, n), 0.80), k=1)

    D_matrices = {
        0: _sym_from_upper(L0),
        1: _sym_from_upper(L1),
        2: _sym_from_upper(L2),
    }

    per_layer_counts = {li: n for li in range(3)}
    # 3 layers × 8 experts = 24; budget=20 → 4 merges total; floor=4 per
    # layer → k_max=4 per layer.
    global_budget = 20

    # Disabled-baseline: no merge_cost_prior in config; GRAPE runs un-biased.
    ctx_base = _make_ctx(
        D_matrices={li: D.clone() for li, D in D_matrices.items()},
        per_layer_counts=per_layer_counts,
        global_budget=global_budget,
        enabled=False,
    )
    # Realistic gamma — entropy gate ACTIVE (not the 1.0 always-off setting).
    ctx_base.get("config")["stage1_grape"]["entropy_tolerance"] = 0.1
    DamageCurveDpPlugin().run(ctx_base)
    assert "merge_cost_prior" not in ctx_base.get("config")["stage1_grape"]
    GrapeMergePlugin().run(ctx_base)
    base_budgets = {
        int(k): v
        for k, v in ctx_base.get("per_layer_target_experts").items()
    }

    # Enabled-run: DP prior is published into the config and GRAPE consumes it.
    ctx_dp = _make_ctx(
        D_matrices={li: D.clone() for li, D in D_matrices.items()},
        per_layer_counts=per_layer_counts,
        global_budget=global_budget,
        enabled=True,
    )
    ctx_dp.get("config")["stage1_grape"]["entropy_tolerance"] = 0.1
    DamageCurveDpPlugin().run(ctx_dp)
    prior_in_cfg = ctx_dp.get("config")["stage1_grape"].get("merge_cost_prior")
    assert prior_in_cfg is not None
    assert set(prior_in_cfg.keys()) == {"0", "1", "2"}    # str-keyed per GRAPE contract
    # GRAPE consumes the prior and runs through without raising.
    GrapeMergePlugin().run(ctx_dp)
    dp_budgets = {
        int(k): v
        for k, v in ctx_dp.get("per_layer_target_experts").items()
    }

    # Both runs hit the global budget and respect the per-layer floor.
    assert sum(base_budgets.values()) == global_budget
    assert sum(dp_budgets.values()) == global_budget
    floor = n // 2
    for li, b in base_budgets.items():
        assert b >= floor, f"baseline floor violation: layer {li} budget {b}"
    for li, b in dp_budgets.items():
        assert b >= floor, f"DP-run floor violation: layer {li} budget {b}"

    # H1: the DP prior MUST shift GRAPE's selection vs the un-biased
    # baseline. If a future regression silently neutralises the prior
    # (e.g. the config mutation is dropped, or the dict is mis-keyed
    # such that GRAPE falls back to argmin R), this assertion fires.
    assert base_budgets != dp_budgets, (
        f"DP prior should bias GRAPE's selection vs the un-biased baseline; "
        f"base_budgets={base_budgets} dp_budgets={dp_budgets}"
    )
    # Direction-specific check: under the synthetic above, the DP plan
    # pins layer 1 at k_max (+inf prior) AND the prior shifts GRAPE
    # into layer 0. So layer 1's DP-biased final budget must be HIGHER
    # (fewer merges absorbed) than baseline's, and layer 0's must be
    # LOWER (more merges absorbed) than baseline's.
    assert dp_budgets[1] > base_budgets[1], (
        f"DP +inf prior on layer 1 should prevent GRAPE from merging there; "
        f"base[1]={base_budgets[1]} dp[1]={dp_budgets[1]}"
    )
    assert dp_budgets[0] < base_budgets[0], (
        f"DP prior should redirect merges into layer 0; "
        f"base[0]={base_budgets[0]} dp[0]={dp_budgets[0]}"
    )


def test_dp_at_floor_with_inf_priors_grape_still_converges():
    """H2: when DP places several layers at their floor (prior=+inf),
    GRAPE must still reach the budget by selecting from the few
    finite-prior layers — and must not hit max_iter.

    Construction: 6 layers (heterogeneous expert counts). Floor divisor 2.
      * Layers 0..3 carry 4 experts each → floor=2, k_max=2. Pair
        distances tiny (0.01..0.10) so DP places all four at their
        k_max (the at-floor side of the plan → prior == +inf).
      * Layers 4, 5 carry 16 experts each → floor=8, k_max=8. Pair
        distances 0.5 throughout (linear cumsum) so DP can split the
        residual budget freely between them — but the prior assignments
        on layers 4, 5 are *not* uniquely determined by the cost. Test
        only asserts: cheap layers all at-floor (+inf prior); at least
        one finite-prior layer exists; GRAPE converges through the
        finite-prior side without stalling.

    Total experts = 4·4 + 2·16 = 48. Pick global_budget=32 → 16 merges
    required: 8 by the cheap layers at k_max, 8 distributed across the
    pricey layers. Even if DP arbitrarily packs all 8 into one pricey
    layer (k_max=8), the single remaining finite-prior layer has enough
    capacity for the residual.
    """
    from moe_compress.stage1.plugins.grape_merge import GrapeMergePlugin

    def _sym_filled(size: int, val: float) -> torch.Tensor:
        upper = np.triu(np.full((size, size), val), k=1)
        sym = upper + upper.T
        np.fill_diagonal(sym, 0.0)
        return torch.from_numpy(sym).float()

    # Layers 0..3: 4 experts each, tiny pair distances → DP at k_max=2.
    cheap_layers: dict[int, torch.Tensor] = {}
    for li in range(4):
        d = _sym_filled(4, 0.10).numpy().astype(np.float64)
        d[0, 1] = d[1, 0] = 0.01
        d[0, 2] = d[2, 0] = 0.02
        d[0, 3] = d[3, 0] = 0.03
        cheap_layers[li] = torch.from_numpy(d).float()

    # Layers 4, 5: 16 experts each. Make their sorted-pair cumulative
    # damage curves strictly CONVEX (each marginal larger than the
    # previous one) so DP balances merges between them — packing all of
    # them into a single pricey layer would be a strictly worse
    # allocation. We achieve this by placing the pair distances on a
    # quadratic schedule along the upper-triangular order.
    n_pairs = 16 * 15 // 2  # 120 pairs per layer
    pair_vals = np.linspace(0.50, 1.00, num=n_pairs, dtype=np.float64)
    pair_vals = np.sort(pair_vals)    # already ascending, but be explicit
    # Each layer's distance matrix carries the SAME sorted distance set
    # so the two layers' cumsum curves are identical → DP must split the
    # residual 8 merges 4-4 between them (any imbalance strictly
    # increases cost on the convex curve).
    iu, ju = np.triu_indices(16, k=1)

    def _pricey(seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        d = np.zeros((16, 16), dtype=np.float64)
        # Same multiset of distances; permute the assignment to pairs so
        # the matrices are not bit-identical (this is just defensive — the
        # cumsum curve is identical because cumsum sorts).
        perm = rng.permutation(n_pairs)
        d[iu, ju] = pair_vals[perm]
        d[ju, iu] = pair_vals[perm]
        return torch.from_numpy(d).float()

    pricey_4 = _pricey(seed=0)
    pricey_5 = _pricey(seed=1)

    D_matrices = {
        0: cheap_layers[0],
        1: cheap_layers[1],
        2: cheap_layers[2],
        3: cheap_layers[3],
        4: pricey_4,
        5: pricey_5,
    }
    per_layer_counts = {0: 4, 1: 4, 2: 4, 3: 4, 4: 16, 5: 16}
    # Σ experts = 48. floor_divisor=2 → floor(0..3)=2, floor(4,5)=8;
    # Σ floors = 24, so min feasible budget=24. Pick 32 → 16 merges.
    # Cheap layers absorb 8 at k_max, pricey layers absorb 8 total —
    # even if DP packs all 8 into a single pricey layer (k_max=8), the
    # remaining finite-prior layer has the full capacity for it.
    global_budget = 32

    ctx = _make_ctx(
        D_matrices={li: D.clone() for li, D in D_matrices.items()},
        per_layer_counts=per_layer_counts,
        global_budget=global_budget,
        enabled=True,
    )
    ctx.get("config")["stage1_grape"]["entropy_tolerance"] = 0.1
    DamageCurveDpPlugin().run(ctx)

    prior = ctx.get("merge_cost_prior_computed")
    dp_optimum = ctx.get("dp_optimum")
    inf_layers = [li for li, p in prior.items() if p == math.inf]
    finite_layers = [li for li, p in prior.items() if math.isfinite(p)]
    # The cheap layers (0..3) MUST be at-floor with +inf prior. The
    # pricey layers may or may not be at-floor depending on DP tie-
    # breaking with the linear cumsum, so we only assert the cheap
    # half here.
    assert {0, 1, 2, 3}.issubset(set(inf_layers)), (
        f"expected the 4 cheap layers (0..3) at-floor with +inf prior; "
        f"got inf={inf_layers} finite={finite_layers}"
    )
    # ~2/3 (4 of 6) at-floor — the regression target. At LEAST one
    # finite-prior layer must remain so GRAPE has somewhere to merge.
    assert len(inf_layers) >= 4, (
        f"expected ≥ 4 layers at-floor; got {inf_layers}"
    )
    assert len(finite_layers) >= 1, (
        f"need at least one finite-prior layer for GRAPE convergence; "
        f"got finite={finite_layers}"
    )
    # Cheap-layer DP optimum: k_max=2 each.
    for li in (0, 1, 2, 3):
        assert dp_optimum[li] == 2, f"layer {li} should be at k_max=2"
    # Σ k* = global_merges by DP construction.
    assert sum(dp_optimum.values()) == sum(per_layer_counts.values()) - global_budget

    # GRAPE must converge to the global budget exactly, even though 4/6
    # layers are gated out by +inf priors. If GRAPE looped to max_iter,
    # `current_total` would not equal `effective_budget`.
    GrapeMergePlugin().run(ctx)
    final_budgets = {
        int(k): v
        for k, v in ctx.get("per_layer_target_experts").items()
    }
    assert sum(final_budgets.values()) == global_budget, (
        f"GRAPE failed to converge to global_budget={global_budget}; "
        f"got Σ budgets = {sum(final_budgets.values())} "
        f"(budgets={final_budgets}) — likely max_iter stall."
    )
    # Floor respected on every layer (DP at-floor side is exactly at
    # floor by construction; finite-prior side must not break floor).
    for li, b in final_budgets.items():
        assert b >= per_layer_counts[li] // 2, (
            f"floor violation layer {li}: budget {b} < "
            f"{per_layer_counts[li] // 2}"
        )
    # +inf-prior layers must NOT receive any merge from GRAPE.
    for li in inf_layers:
        assert final_budgets[li] == per_layer_counts[li], (
            f"+inf-prior layer {li} should not be merged by GRAPE "
            f"(prior=inf gates it out); got final budget "
            f"{final_budgets[li]} vs initial {per_layer_counts[li]}"
        )
    # GRAPE absorbs ALL the residual merges into the finite-prior layers.
    finite_surviving = sum(final_budgets[li] for li in finite_layers)
    finite_initial = sum(per_layer_counts[li] for li in finite_layers)
    total_merges_required = sum(per_layer_counts.values()) - global_budget
    expected_finite_surviving = finite_initial - total_merges_required
    assert finite_surviving == expected_finite_surviving, (
        f"finite-prior layers must absorb all merges: "
        f"finite_surviving={finite_surviving} "
        f"expected={expected_finite_surviving} (finite_initial={finite_initial} "
        f"− total_merges_required={total_merges_required})"
    )


# ---------------------------------------------------------------------------
# 7. contribute_artifact contract
# ---------------------------------------------------------------------------


def test_contribute_artifact_returns_empty_dict():
    plugin = DamageCurveDpPlugin()
    ctx = PipelineContext()
    out = plugin.contribute_artifact(ctx)
    assert out == {}
    # Each call returns a fresh dict (no shared module-level object).
    assert plugin.contribute_artifact(ctx) is not out


# ---------------------------------------------------------------------------
# 8. Missing required slots raise KeyError
# ---------------------------------------------------------------------------


def test_run_missing_d_matrices_raises():
    ctx = PipelineContext()
    ctx.set("config", {
        "stage1_grape": {"damage_curve_dp": {"enabled": True}}
    })
    ctx.set("blacklist", {})
    ctx.set("per_layer_targets", {0: 4})
    ctx.set("decomposition", _decomposition(global_budget=4))
    # Missing D_matrices.
    with pytest.raises(KeyError, match="D_matrices"):
        DamageCurveDpPlugin().run(ctx)


def test_run_invalid_floor_divisor_raises():
    n = 4
    D = torch.zeros(n, n)
    ctx = _make_ctx(
        D_matrices={0: D},
        per_layer_counts={0: n},
        global_budget=4,
        enabled=True,
        floor_divisor=0,    # invalid
    )
    with pytest.raises(ValueError, match="floor_divisor"):
        DamageCurveDpPlugin().run(ctx)
