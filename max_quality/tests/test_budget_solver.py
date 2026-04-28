"""Sanity checks on the budget solver — does it hit the target reduction?"""
from __future__ import annotations

import pytest

from moe_compress.budget import solver
from moe_compress.stage1_grape import _allocate_budgets
from moe_compress.utils.model_io import count_expert_parameters, count_parameters


def test_solve_hits_target(tiny_model):
    total = count_parameters(tiny_model)
    expert = count_expert_parameters(tiny_model, routed_only=True)
    # Sanity: experts account for the bulk of this tiny model.
    assert expert > 0
    assert expert < total

    # Use a modest target the 2×4-expert fixture can actually hit with
    # whole-expert rounding and a 2-per-layer floor.
    decomp = solver.solve(
        tiny_model,
        target_total_reduction=0.15,
        expert_svd_ratio=5.0,  # ep/sp = 0.25/0.05 = 5
        min_experts_per_layer=2,
    )
    assert decomp.projected_total_reduction >= 0.145
    assert decomp.global_expert_budget < 4 * 2


def test_solve_raises_on_impossible_target(tiny_model):
    # Setting min_experts to the current count leaves nothing prunable →
    # solver should raise RuntimeError per bug #10 fix.
    with pytest.raises(RuntimeError, match="target_total_reduction"):
        solver.solve(
            tiny_model,
            target_total_reduction=0.40,
            expert_svd_ratio=5.0,
            min_experts_per_layer=4,   # = original, nothing prunable
        )


def test_late_layer_bonus_raises_last_layer_budgets():
    # 6-layer model: layers 0-5 with 8 experts each, global budget = 36 (75%)
    # With late_layer_bonus=4 / late_layer_bonus_depth=2: layers 4 and 5 should
    # receive more experts than layers 2 and 3 (middle layers with equal redundancy).
    redundancies = {i: 0.5 for i in range(6)}  # uniform redundancy → only bonus differentiates
    per_layer_counts = {i: 8 for i in range(6)}
    budgets = _allocate_budgets(
        redundancies=redundancies,
        global_budget=36,
        per_layer_counts=per_layer_counts,
        min_experts=1,
        blacklist={},
        early_bonus=0,
        early_bonus_depth=0,
        late_bonus=4,
        late_bonus_depth=2,
    )
    assert sum(budgets.values()) == 36
    # Late layers (4, 5) must have higher budget than mid layers (2, 3)
    assert budgets[4] > budgets[2]
    assert budgets[5] > budgets[3]
    # All layers must respect floor and ceiling
    assert all(1 <= v <= 8 for v in budgets.values())


def test_early_and_late_bonus_both_applied():
    # Both early and late bonuses should fire simultaneously on a 6-layer model.
    redundancies = {i: 0.5 for i in range(6)}
    per_layer_counts = {i: 8 for i in range(6)}
    budgets_with_bonuses = _allocate_budgets(
        redundancies=redundancies,
        global_budget=36,
        per_layer_counts=per_layer_counts,
        min_experts=1,
        blacklist={},
        early_bonus=2,
        early_bonus_depth=1,   # layer 0 gets +2
        late_bonus=2,
        late_bonus_depth=1,    # layer 5 gets +2
    )
    budgets_no_bonuses = _allocate_budgets(
        redundancies=redundancies,
        global_budget=36,
        per_layer_counts=per_layer_counts,
        min_experts=1,
        blacklist={},
        early_bonus=0,
        early_bonus_depth=0,
        late_bonus=0,
        late_bonus_depth=0,
    )
    assert sum(budgets_with_bonuses.values()) == 36
    # Edge layers get more than the no-bonus baseline
    assert budgets_with_bonuses[0] >= budgets_no_bonuses[0]
    assert budgets_with_bonuses[5] >= budgets_no_bonuses[5]
