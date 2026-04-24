"""Sanity checks on the budget solver — does it hit the target reduction?"""
from __future__ import annotations

import pytest

from moe_compress.budget import solver
from moe_compress.utils.model_io import count_expert_parameters, count_parameters


def test_solve_hits_target(tiny_model):
    total = count_parameters(tiny_model)
    expert = count_expert_parameters(tiny_model, routed_only=True)
    # Sanity: experts account for the bulk of this tiny model.
    assert expert > 0
    assert expert < total

    decomp = solver.solve(
        tiny_model,
        target_total_reduction=0.20,
        initial_expert_reduction=0.25,
        initial_svd_reduction=0.05,
        min_experts_per_layer=2,
    )
    assert decomp.projected_total_reduction >= 0.18   # tolerance in the solver
    assert decomp.global_expert_budget < 4 * 2         # fewer than original 4x2


def test_solve_raises_on_impossible_target(tiny_model):
    # Setting min_experts to the current count leaves nothing prunable →
    # solver should raise RuntimeError per bug #10 fix.
    with pytest.raises(RuntimeError, match="target_total_reduction"):
        solver.solve(
            tiny_model,
            target_total_reduction=0.40,
            initial_expert_reduction=0.25,
            initial_svd_reduction=0.05,
            min_experts_per_layer=4,   # = original, nothing prunable
        )
