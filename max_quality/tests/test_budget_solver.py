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

    # Use a modest target the 2×4-expert fixture can actually hit with
    # whole-expert rounding and a 2-per-layer floor.
    decomp = solver.solve(
        tiny_model,
        target_total_reduction=0.15,
        ep_sp_knob_ratio=5.0,  # ep/sp = 0.25/0.05 = 5
        min_experts_per_layer=2,
    )
    assert 0.145 <= decomp.projected_total_reduction <= 0.165
    assert decomp.global_expert_budget < 4 * 2  # at least one expert pruned per layer on average
    assert decomp.global_expert_budget >= 2 * 2  # min_experts_per_layer=2 across 2 layers


def test_solve_raises_on_impossible_target(tiny_model):
    # Setting min_experts to the current count leaves nothing prunable →
    # solver should raise ValueError with min_pool=0 or all-protected message.
    with pytest.raises(ValueError, match=r"min_pool=0|All experts are protected"):
        solver.solve(
            tiny_model,
            target_total_reduction=0.40,
            ep_sp_knob_ratio=5.0,
            min_experts_per_layer=4,   # = original, nothing prunable
        )


def test_solve_floor_clamp_branch(tiny_model):
    """Floor-clamp branch: expert-pruning knob hits the protected-floor ceiling mid-solve.

    With 2 layers × 4 experts (min_pool = 2 when min_experts_per_layer=3),
    max_prunable_frac = 2/8 = 0.25.  A target of 0.25 with ep_sp_knob_ratio=5
    drives the scale-adjusted ep above 0.25 during the first iteration's scale
    step, triggering the floor-clamp branch (ep clamped to min_pool/total_routed,
    sp solved analytically from the discretisation-consistent formula).  The
    floor-clamp fires during the scale step, not from the analytical starting point.
    """
    decomp = solver.solve(
        tiny_model,
        target_total_reduction=0.25,
        ep_sp_knob_ratio=5.0,
        min_experts_per_layer=3,   # floor=3 → min_pool=2 → max_prunable_frac=0.25
    )
    # Floor-clamped: exactly 2 experts pruned across 2 layers (6 survive total).
    assert decomp.global_expert_budget == 6, (
        f"Expected exactly 6 surviving experts (2 pruned at floor), got {decomp.global_expert_budget}"
    )
    # ep is clamped to the floor; svd_rank_ratio absorbs the residual.
    assert decomp.expert_prune_ratio <= 0.25 + 1e-6, (
        f"ep should be clamped to ≤0.25 (floor), got {decomp.expert_prune_ratio:.6f}"
    )
    assert decomp.svd_rank_ratio > 0, (
        "sp must be positive: SVD must compensate for the floor-clamped expert pruning"
    )
    # Projected reduction must be within tolerance of the target.
    assert abs(decomp.projected_total_reduction - 0.25) <= 0.005, (
        f"projected_total_reduction={decomp.projected_total_reduction:.4f} not within 0.005 of 0.25"
    )


def test_solve_rejects_nonfinite_tolerance(tiny_model):
    """Solver must raise ValueError for non-finite or non-positive tolerance."""
    with pytest.raises(ValueError, match="tolerance"):
        solver.solve(
            tiny_model,
            target_total_reduction=0.15,
            ep_sp_knob_ratio=5.0,
            min_experts_per_layer=2,
            tolerance=float("inf"),
        )
    with pytest.raises(ValueError, match="tolerance"):
        solver.solve(
            tiny_model,
            target_total_reduction=0.15,
            ep_sp_knob_ratio=5.0,
            min_experts_per_layer=2,
            tolerance=float("nan"),
        )
    with pytest.raises(ValueError, match="tolerance"):
        solver.solve(
            tiny_model,
            target_total_reduction=0.15,
            ep_sp_knob_ratio=5.0,
            min_experts_per_layer=2,
            tolerance=0,
        )
    with pytest.raises(ValueError, match="tolerance"):
        solver.solve(
            tiny_model,
            target_total_reduction=0.15,
            ep_sp_knob_ratio=5.0,
            min_experts_per_layer=2,
            tolerance=-0.001,
        )


def test_blacklisted_experts_is_deep_copied(tiny_model):
    """Mutating the original blacklisted_experts dict after solve() must not affect decomp."""
    blacklisted = {0: [0, 1]}
    original_inner = blacklisted[0]  # save reference before solve
    decomp = solver.solve(
        tiny_model,
        target_total_reduction=0.15,
        ep_sp_knob_ratio=5.0,
        min_experts_per_layer=2,
        blacklisted_experts=blacklisted,
    )
    # Verify the stored inner list is a different object from the caller's original list.
    assert decomp.blacklisted_experts[0] is not original_inner, "inner list must be a copy"
    blacklisted[0].append(99)  # mutate original inner list
    blacklisted[99] = [0]      # add new key to original dict
    assert decomp.blacklisted_experts == {0: [0, 1]}, "decomp should be independent copy"


def test_as_dict_stringifies_blacklisted_keys(tiny_model):
    """as_dict() must convert int keys in blacklisted_experts to strings for JSON compat."""
    decomp = solver.solve(
        tiny_model,
        target_total_reduction=0.15,
        ep_sp_knob_ratio=5.0,
        min_experts_per_layer=2,
        blacklisted_experts={0: [0, 1], 1: [2]},
    )
    d = decomp.as_dict()
    assert all(isinstance(k, str) for k in d["blacklisted_experts"]), (
        "blacklisted_experts keys must be strings in as_dict() output"
    )


