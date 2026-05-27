"""Unit tests for ``moe_compress.stage1.plugins.rco_budget``.

Verifies:

1. The plugin's class-level Protocol attributes match the registry
   contract (``PipelinePlugin``).
2. ``is_enabled`` is False by default and True only when the explicit
   config flag is set.
3. ``run`` raises a clean ``KeyError`` (no silent degradation) when any
   required ctx slot is missing.
4. RCO Algorithm 1 on a small synthetic 2-layer case produces a
   budget-exact, floor-respecting allocation.
5. When a damage curve is supplied, RCO consumes it and the allocation
   shifts vs the synthetic fallback.
6. ``contribute_artifact`` returns an empty dict before ``run`` (so a
   disabled-but-mistakenly-called artifact write is safe) and a populated
   dict after ``run``.
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 1. Protocol-attribute contract
# ---------------------------------------------------------------------------


def test_plugin_protocol_attributes():
    """Class-level attributes match the plan exactly."""
    plugin = RCOBudgetPlugin()
    assert plugin.name == "rco_budget"
    # paper string must cite the arxiv id + the clean-room note + the
    # named deviations.
    assert "arxiv:2605.00649" in plugin.paper
    assert "clean-room" in plugin.paper
    assert "IST-DASLab" in plugin.paper
    for deviation_token in (
        "D-clean-room",
        "D-init-grape",
        "D-fitness-mse",
        "D-synthetic-curve",
        "D-floor-projection",
        "D-ragged-K",
        "D-bisection-budget",
        "D-disabled-default",
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


def test_plugin_is_runtime_checkable_pipelineplugin():
    """``isinstance`` against the runtime-checkable Protocol must succeed."""
    assert isinstance(RCOBudgetPlugin(), PipelinePlugin)


# ---------------------------------------------------------------------------
# 2. is_enabled gate
# ---------------------------------------------------------------------------


def test_plugin_disabled_by_default():
    """RCO is OFF unless the explicit flag is set true."""
    plugin = RCOBudgetPlugin()
    assert plugin.is_enabled({}) is False
    assert plugin.is_enabled({"stage1": {}}) is False
    assert plugin.is_enabled({"stage1": {"rco_budget": {}}}) is False
    assert plugin.is_enabled({"stage1": {"rco_budget": {"enabled": False}}}) is False


def test_plugin_enabled_when_flag_true():
    """When the explicit flag is True, the plugin enables."""
    plugin = RCOBudgetPlugin()
    assert plugin.is_enabled({"stage1": {"rco_budget": {"enabled": True}}}) is True


# ---------------------------------------------------------------------------
# 3. Missing-slot KeyError contract
# ---------------------------------------------------------------------------


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
    """``plugin.run`` must raise ``KeyError`` mentioning the missing slot name."""
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


# ---------------------------------------------------------------------------
# 4. Algorithm correctness on a small synthetic case
# ---------------------------------------------------------------------------


def test_run_synthetic_2layer_handcheck():
    """2 layers × 4 experts, B=6, no damage curve → synthetic fallback.

    Hand-check: option grid is {2, 3, 4} per layer. GRAPE init = {3, 3}
    (sum 6, the only "balanced" feasible). With symmetric redundancy
    R̃ = 0.5 on both layers, the synthetic curve gives equal cost per
    layer, so RCO has no signal to break the {3, 3} tie — final
    allocation should still sum to 6 and respect the floor.
    """
    inputs = _build_inputs(global_budget=6)
    ctx = _populate_context(inputs)
    RCOBudgetPlugin().run(ctx)

    rco_budgets = ctx.get("per_layer_target_experts_rco")
    assert set(rco_budgets.keys()) == {"0", "1"}
    # Floor = 2; per_layer_count = 4; option grid {2, 3, 4}.
    for k, v in rco_budgets.items():
        assert 2 <= v <= 4, f"layer {k}: budget {v} outside [floor, N]"
    # Sums to global budget.
    assert sum(rco_budgets.values()) == 6


def test_run_consumes_damage_curve_when_present():
    """With an asymmetric damage curve, RCO shifts allocation toward the costly layer.

    Layer 0 has a STEEP damage curve (each removed expert costs 100); layer 1
    is nearly flat (each removed expert costs 1). RCO should keep layer 0
    near its full count (4) and let layer 1 absorb most of the compression.
    """
    inputs = _build_inputs(global_budget=6)
    ctx = _populate_context(inputs)
    # Build damage curve: D_l(k) = damage from choosing k surviving experts.
    # Layer 0: steep (every removed expert costs 100). Layer 1: flat (cost 1).
    ctx.set("per_layer_damage_curve", {
        0: {2: 200.0, 3: 100.0, 4: 0.0},
        1: {2: 2.0, 3: 1.0, 4: 0.0},
    })
    RCOBudgetPlugin().run(ctx)

    rco_budgets = ctx.get("per_layer_target_experts_rco")
    # Sum is feasible: 4 (l0) + 2 (l1) = 6, or 3+3, or 2+4. With the
    # asymmetric cost, optimum is l0=4 (cheap to keep), l1=2 (cheap to cut):
    # combined damage = 0 + 2 = 2.
    # Alternative 3+3 = 100+1 = 101; 2+4 = 200+0 = 200. So 4+2 should win.
    assert rco_budgets["0"] == 4, f"steep layer 0 should keep all 4 experts, got {rco_budgets['0']}"
    assert rco_budgets["1"] == 2, f"flat layer 1 should drop to floor 2, got {rco_budgets['1']}"
    assert sum(rco_budgets.values()) == 6

    # Metadata sanity.
    metadata = ctx.get("rco_metadata")
    assert metadata["fitness_source"] == "damage_curve"
    assert metadata["achieved_budget"] == 6
    assert metadata["requested_budget"] == 6
    assert metadata["final_fitness"] <= metadata["init_fitness"] + 1e-6


def test_run_respects_floor():
    """RCO must never allocate below ``floor_l = per_layer_count_l // 2``.

    Even with a damage curve that *favors* dropping a layer entirely,
    the floor bakes a hard lower bound into the option grid.
    """
    inputs = _build_inputs(global_budget=8)  # plenty of budget for 4 floors of 2
    ctx = _populate_context(inputs)
    # Damage curve aggressively rewards dropping layer 0 entirely (impossible
    # because the option grid floor is 2).
    ctx.set("per_layer_damage_curve", {
        0: {2: 0.0, 3: 100.0, 4: 200.0},  # cheapest to keep just 2
        1: {2: 0.0, 3: 0.0, 4: 0.0},
    })
    RCOBudgetPlugin().run(ctx)
    rco_budgets = ctx.get("per_layer_target_experts_rco")
    assert rco_budgets["0"] >= 2
    assert rco_budgets["1"] >= 2


def test_run_sums_to_global_budget():
    """The final allocation sums exactly to ``decomposition.global_expert_budget``."""
    for B in [6, 7, 8]:
        inputs = _build_inputs(global_budget=B)
        ctx = _populate_context(inputs)
        RCOBudgetPlugin().run(ctx)
        rco_budgets = ctx.get("per_layer_target_experts_rco")
        assert sum(rco_budgets.values()) == B, (
            f"global_budget={B}: RCO sum = {sum(rco_budgets.values())}"
        )


# ---------------------------------------------------------------------------
# 5. Artifact contract
# ---------------------------------------------------------------------------


def test_contribute_artifact_when_disabled():
    """Before ``run`` (or when disabled), the artifact is an empty dict."""
    plugin = RCOBudgetPlugin()
    ctx = PipelineContext()
    assert plugin.contribute_artifact(ctx) == {}


def test_contribute_artifact_when_enabled():
    """After ``run``, ``contribute_artifact`` returns the budget + metadata dict."""
    inputs = _build_inputs()
    ctx = _populate_context(inputs)
    plugin = RCOBudgetPlugin()
    plugin.run(ctx)
    payload = plugin.contribute_artifact(ctx)
    assert set(payload.keys()) == {"rco_budgets", "rco_metadata"}
    assert isinstance(payload["rco_budgets"], dict)
    assert isinstance(payload["rco_metadata"], dict)
    # Budget dict keys are str(layer_idx); values are ints.
    for k, v in payload["rco_budgets"].items():
        assert isinstance(k, str)
        assert isinstance(v, int)
    # Metadata carries init + final fitness + budget vectors.
    md = payload["rco_metadata"]
    assert "init_fitness" in md
    assert "final_fitness" in md
    assert "init_budget_vector" in md
    assert "final_budget_vector" in md
    assert md["achieved_budget"] == sum(payload["rco_budgets"].values())


# ---------------------------------------------------------------------------
# 6. Manifest registration sanity
# ---------------------------------------------------------------------------


def test_plugin_registered_in_manifest():
    """``RCOBudgetPlugin`` must be in ``STAGE1_PLUGIN_MANIFEST`` so the
    orchestrator can look it up by name."""
    from moe_compress.stage1.plugins import STAGE1_PLUGIN_MANIFEST
    names = [p.name for p in STAGE1_PLUGIN_MANIFEST]
    assert "rco_budget" in names
    # Must come AFTER grape_merge so it can consume GRAPE's output.
    assert names.index("rco_budget") > names.index("grape_merge")
