"""CPU-only tests for the Direction-A budget-retune tool.

These build synthetic Stage-1 budgets + Stage-2 merge JSONs on disk (no model,
no GPU) and assert the four required properties:

  (a) total kept-expert count is conserved;
  (b) per-layer floors/ceilings are respected;
  (c) reallocation moves budget from low-damage to high-damage layers;
  (d) idempotence-ish: a second pass on already-optimal input is a no-op.
"""
from __future__ import annotations

import json

import pytest

from moe_compress.budget_retune import (
    NoDamageSignalError,
    assemble_layers,
    load_stage1_budgets,
    load_stage2_damage,
    retune_budgets,
    retune_from_artifacts,
)


# ---------------------------------------------------------------------------
# Synthetic-artifact helpers
# ---------------------------------------------------------------------------
def _write_stage1_budgets(artifacts_dir, per_layer_budget, *, name="stage1_budgets.json"):
    """Write a minimal but realistic stage1_budgets.json."""
    payload = {
        "per_layer_target_experts": {str(k): int(v) for k, v in per_layer_budget.items()},
        "per_layer_redundancy": {str(k): 0.0 for k in per_layer_budget},
        "achieved_budget": sum(per_layer_budget.values()),
        "requested_budget": sum(per_layer_budget.values()),
        "config": {"some": "stage1-config"},
    }
    path = artifacts_dir / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_merge_json(partial_dir, layer_idx, total_experts, mean_cost_per_pair):
    """Write one Stage-2 v2 merge_<layer>.json.

    Only the fields budget_retune actually reads are load-bearing here:
    ``format_version``, ``freq`` (its length == total_experts), and
    ``mean_cost_per_pair``. The rest mirror the real schema for realism.
    """
    payload = {
        "format_version": 2,
        "final_kept_ids": list(range(min(total_experts, total_experts))),
        "grouped": {"0": [0]},
        # freq keys must be range(total_experts) — Stage 2 enforces this and
        # budget_retune relies on len(freq) == N_l.
        "freq": {str(i): 1 for i in range(total_experts)},
        "merge_map_layer": {"0": [0]},
        "mean_cost_per_pair": mean_cost_per_pair,
        "assignment_solver_used": "greedy",
        "cost_alignment_used": "pre",
        "em_rounds_completed": 0,
        "distill_state": None,
    }
    path = partial_dir / f"merge_{layer_idx}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_run(
    artifacts_dir,
    layers,  # {layer_idx: (total_experts, current_budget, mean_cost_per_pair)}
):
    """Materialise a full synthetic Stage-1+Stage-2 artifacts dir."""
    partial_dir = artifacts_dir / "_stage2_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    per_layer_budget = {li: cur for li, (_, cur, _) in layers.items()}
    _write_stage1_budgets(artifacts_dir, per_layer_budget)
    for li, (total, _cur, mcp) in layers.items():
        _write_merge_json(partial_dir, li, total, mcp)
    return artifacts_dir


# ---------------------------------------------------------------------------
# (a) total kept-expert count is conserved
# ---------------------------------------------------------------------------
def test_total_kept_conserved(tmp_path):
    # 4 layers, 8 experts each, uniform budget 5. Damage varies widely.
    layers = {
        0: (8, 5, 0.01),   # cheap
        1: (8, 5, 0.05),
        2: (8, 5, 0.20),
        3: (8, 5, 0.90),   # expensive
    }
    _make_run(tmp_path, layers)
    result, out_path = retune_from_artifacts(tmp_path)

    assert sum(result.new_budgets.values()) == sum(result.old_budgets.values())
    assert result.total_kept == 4 * 5

    # The written artifact also conserves the total.
    written = json.loads(out_path.read_text())
    new_budgets = {int(k): v for k, v in written["per_layer_target_experts"].items()}
    assert sum(new_budgets.values()) == 4 * 5
    assert written["achieved_budget"] == 4 * 5


# ---------------------------------------------------------------------------
# (b) per-layer floors and ceilings respected
# ---------------------------------------------------------------------------
def test_floors_and_ceilings_respected(tmp_path):
    # Extreme damage gradient pushes the optimiser hard against the bounds.
    layers = {
        0: (8, 7, 0.001),   # very cheap, near ceiling: should be drained to floor 4
        1: (8, 7, 0.002),   # very cheap
        2: (8, 4, 0.500),   # expensive, at floor: should be filled toward ceiling 8
        3: (8, 4, 0.900),   # most expensive
    }
    _make_run(tmp_path, layers)
    result, _ = retune_from_artifacts(tmp_path)

    for li, (total, _cur, _mcp) in layers.items():
        floor = total // 2
        b = result.new_budgets[li]
        assert floor <= b <= total, f"layer {li}: budget {b} outside [{floor},{total}]"


# ---------------------------------------------------------------------------
# (c) reallocation moves budget from low-damage to high-damage layers
# ---------------------------------------------------------------------------
def test_moves_budget_toward_high_damage(tmp_path):
    layers = {
        0: (8, 6, 0.01),   # cheapest -> should LOSE budget
        1: (8, 6, 0.04),
        2: (8, 6, 0.30),
        3: (8, 6, 0.95),   # most expensive -> should GAIN budget
    }
    _make_run(tmp_path, layers)
    result, _ = retune_from_artifacts(tmp_path)

    # Cheapest layer lost experts; most-expensive layer gained them.
    assert result.new_budgets[0] < result.old_budgets[0]
    assert result.new_budgets[3] > result.old_budgets[3]
    # Net flow respects the damage ordering: a strictly cheaper layer never
    # ends up with more budget than a strictly more expensive one when both
    # started equal.
    assert result.new_budgets[0] <= result.new_budgets[3]
    # The optimiser actually reduced predicted damage.
    assert result.predicted_damage_after < result.predicted_damage_before
    assert result.transfers > 0


def test_optimal_allocation_at_bounds(tmp_path):
    # With a steep gradient and equal starting budgets the greedy optimum
    # drains cheap layers to the floor and fills the expensive one to ceiling.
    layers = {
        0: (8, 6, 0.01),
        1: (8, 6, 0.02),
        2: (8, 6, 0.03),
        3: (8, 6, 5.00),   # dominant cost
    }
    _make_run(tmp_path, layers)
    result, _ = retune_from_artifacts(tmp_path)
    # layer 3 should be filled to its ceiling (8); the freed 2 experts come
    # from the cheapest donors.
    assert result.new_budgets[3] == 8
    assert sum(result.new_budgets.values()) == sum(result.old_budgets.values())


# ---------------------------------------------------------------------------
# (d) idempotence: a second pass on already-optimal input is a no-op
# ---------------------------------------------------------------------------
def test_idempotent_second_pass(tmp_path):
    layers = {
        0: (8, 6, 0.01),
        1: (8, 6, 0.04),
        2: (8, 6, 0.30),
        3: (8, 6, 0.95),
    }
    _make_run(tmp_path, layers)
    result1, out_path1 = retune_from_artifacts(tmp_path)

    # Build a second synthetic run whose Stage-1 budgets ARE the retuned
    # output, but with the SAME measured per-layer damage. Re-running the
    # retune must produce zero transfers.
    run2 = tmp_path / "run2"
    partial2 = run2 / "_stage2_partial"
    partial2.mkdir(parents=True)
    _write_stage1_budgets(run2, result1.new_budgets)
    for li, (total, _cur, mcp) in layers.items():
        _write_merge_json(partial2, li, total, mcp)

    result2, _ = retune_from_artifacts(run2)
    assert result2.transfers == 0
    assert result2.new_budgets == result1.new_budgets
    assert result2.predicted_damage_after == pytest.approx(
        result2.predicted_damage_before
    )


def test_already_optimal_input_is_noop(tmp_path):
    # Hand-crafted optimal allocation: budget already maximally concentrated
    # on the expensive layer (at ceiling), cheap layers at floor.
    layers = {
        0: (8, 4, 0.01),   # at floor
        1: (8, 4, 0.02),   # at floor
        2: (8, 8, 0.50),   # at ceiling
        3: (8, 8, 0.90),   # at ceiling
    }
    _make_run(tmp_path, layers)
    result, _ = retune_from_artifacts(tmp_path)
    assert result.transfers == 0
    assert result.new_budgets == result.old_budgets


# ---------------------------------------------------------------------------
# Honesty: no usable damage signal -> refuse, do not invent
# ---------------------------------------------------------------------------
def test_no_damage_signal_raises(tmp_path):
    # Every layer has mean_cost_per_pair == null (Stage 2 merged nothing).
    layers = {
        0: (8, 5, None),
        1: (8, 5, None),
        2: (8, 5, None),
    }
    _make_run(tmp_path, layers)
    with pytest.raises(NoDamageSignalError):
        retune_from_artifacts(tmp_path)


def test_signal_less_layers_are_pinned(tmp_path):
    # Layers 0,1 have no signal; only 2,3 carry measured damage.
    layers = {
        0: (8, 6, None),    # pinned
        1: (8, 6, None),    # pinned
        2: (8, 6, 0.05),    # cheap signal layer
        3: (8, 6, 0.80),    # expensive signal layer
    }
    _make_run(tmp_path, layers)
    result, _ = retune_from_artifacts(tmp_path)
    # Signal-less layers must keep their original budget exactly.
    assert result.new_budgets[0] == 6
    assert result.new_budgets[1] == 6
    assert sorted(result.layers_without_signal) == [0, 1]
    # Transfer happened only between the two signal layers.
    assert result.new_budgets[2] < 6
    assert result.new_budgets[3] > 6
    assert result.new_budgets[2] + result.new_budgets[3] == 12


# ---------------------------------------------------------------------------
# Model-agnostic: non-uniform expert counts derived from artifacts
# ---------------------------------------------------------------------------
def test_non_uniform_expert_counts(tmp_path):
    # Layers with DIFFERENT total expert counts — floors/ceilings must be
    # derived per-layer from the artifacts, never assumed.
    layers = {
        0: (16, 12, 0.01),   # floor 8,  ceiling 16
        1: (4, 3, 0.02),     # floor 2,  ceiling 4
        2: (32, 20, 0.50),   # floor 16, ceiling 32
        3: (8, 5, 0.90),     # floor 4,  ceiling 8
    }
    _make_run(tmp_path, layers)
    result, _ = retune_from_artifacts(tmp_path)

    for li, (total, _cur, _mcp) in layers.items():
        floor = total // 2
        assert floor <= result.new_budgets[li] <= total
    assert sum(result.new_budgets.values()) == sum(result.old_budgets.values())
    # Expensive layers (2,3) should not lose budget; cheap ones should not gain.
    assert result.new_budgets[2] >= result.old_budgets[2]
    assert result.new_budgets[3] >= result.old_budgets[3]
    assert result.new_budgets[0] <= result.old_budgets[0]


# ---------------------------------------------------------------------------
# Loader / validation edge cases
# ---------------------------------------------------------------------------
def test_missing_stage2_partial_dir_raises(tmp_path):
    _write_stage1_budgets(tmp_path, {0: 5, 1: 5})
    with pytest.raises(FileNotFoundError, match="_stage2_partial"):
        retune_from_artifacts(tmp_path)


def test_output_must_not_clobber_input(tmp_path):
    layers = {0: (8, 5, 0.1), 1: (8, 5, 0.2)}
    _make_run(tmp_path, layers)
    with pytest.raises(ValueError, match="must differ"):
        retune_from_artifacts(
            tmp_path, output_path=tmp_path / "stage1_budgets.json"
        )


def test_layer_set_mismatch_raises(tmp_path):
    # Stage-1 budgets has layer 2 that Stage-2 never recorded.
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()
    _write_stage1_budgets(tmp_path, {0: 5, 1: 5, 2: 5})
    _write_merge_json(partial_dir, 0, 8, 0.1)
    _write_merge_json(partial_dir, 1, 8, 0.2)
    with pytest.raises(ValueError, match="Layer-set mismatch"):
        retune_from_artifacts(tmp_path)


def test_input_below_floor_rejected(tmp_path):
    # Stage-1 budget of 3 for an 8-expert layer is below the floor 4.
    layers = {0: (8, 3, 0.1), 1: (8, 6, 0.2)}
    _make_run(tmp_path, layers)
    with pytest.raises(ValueError, match="below the floor"):
        retune_from_artifacts(tmp_path)


def test_total_experts_derived_from_freq_length(tmp_path):
    # The per-layer ceiling must come from len(freq), not from any constant.
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()
    _write_merge_json(partial_dir, 0, total_experts=13, mean_cost_per_pair=0.1)
    damage = load_stage2_damage(tmp_path)
    assert damage[0][0] == 13  # total_experts == len(freq)


def test_zero_cost_treated_as_no_signal(tmp_path):
    # mean_cost_per_pair == 0.0 carries no usable gradient -> treated as
    # signal-less (and, if it's the only such case, retune refuses).
    layers = {
        0: (8, 5, 0.0),
        1: (8, 5, 0.0),
    }
    _make_run(tmp_path, layers)
    with pytest.raises(NoDamageSignalError):
        retune_from_artifacts(tmp_path)
