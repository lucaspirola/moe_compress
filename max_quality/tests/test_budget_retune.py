"""CPU-only tests for the redesigned Direction-A budget-retune tool.

These build synthetic Stage-1 budgets + Stage-2 merge JSONs on disk (no model,
no GPU) and assert the properties of the *damage-aware allocation re-solve*
(Direction A proposal, option d):

  * the GLOBAL kept-expert count is conserved (achieved compression pinned);
  * per-layer floors ``max(N//K, n_protected)`` and ceilings ``N`` are respected;
  * the re-solve moves budget toward high-damage layers;
  * a configurable ``N//K`` floor (K>2) opens donor freedom below ``N//2``;
  * ``BudgetInfeasibleError`` is raised when the floors make the global budget
    unreachable;
  * the redundancy prior scores signal-less layers so they can be re-allocated;
  * the default path (``K=2``, no prior) is unchanged vs. an equivalent
    measured-only solve.
"""
from __future__ import annotations

import json

import pytest

from moe_compress.budget_retune import (
    BudgetInfeasibleError,
    NoDamageSignalError,
    assemble_layers,
    load_protected_counts,
    load_stage1_budgets,
    load_stage2_damage,
    retune_budgets,
    retune_from_artifacts,
)


# ---------------------------------------------------------------------------
# Synthetic-artifact helpers
# ---------------------------------------------------------------------------
def _write_stage1_budgets(
    artifacts_dir, per_layer_budget, *, name="stage1_budgets.json",
    redundancy=None,
):
    """Write a minimal but realistic stage1_budgets.json.

    ``redundancy`` optionally maps layer_idx -> R̃^l; when None every layer
    gets 0.0 (GRAPE's value for a uniform-redundancy run).
    """
    if redundancy is None:
        redundancy = {k: 0.0 for k in per_layer_budget}
    payload = {
        "per_layer_target_experts": {str(k): int(v) for k, v in per_layer_budget.items()},
        "per_layer_redundancy": {str(k): float(redundancy[k]) for k in per_layer_budget},
        "achieved_budget": sum(per_layer_budget.values()),
        "requested_budget": sum(per_layer_budget.values()),
        "config": {"some": "stage1-config"},
    }
    path = artifacts_dir / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_merge_json(
    partial_dir, layer_idx, total_experts, mean_cost_per_pair, *, format_version=2
):
    """Write one Stage-2 v2 merge_<layer>.json.

    Only the fields budget_retune actually reads are load-bearing here:
    ``format_version``, ``freq`` (its length == total_experts), and
    ``mean_cost_per_pair``. The rest mirror the real schema for realism.
    """
    payload = {
        "format_version": format_version,
        "final_kept_ids": list(range(total_experts)),
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


def _write_blacklist(artifacts_dir, protected_counts):
    """Write a minimal stage1_blacklist.json.

    ``protected_counts`` maps ``layer_idx -> n_protected``; the artifact's
    ``blacklist`` key stores the (synthetic) protected expert-id lists.
    """
    payload = {
        "blacklist": {str(li): list(range(n)) for li, n in protected_counts.items()},
    }
    path = artifacts_dir / "stage1_blacklist.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _make_run(
    artifacts_dir,
    layers,  # {layer_idx: (total_experts, current_budget, mean_cost_per_pair)}
    blacklist=None,  # optional {layer_idx: n_protected}
    redundancy=None,  # optional {layer_idx: R̃^l}
):
    """Materialise a full synthetic Stage-1+Stage-2 artifacts dir."""
    partial_dir = artifacts_dir / "_stage2_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    per_layer_budget = {li: cur for li, (_, cur, _) in layers.items()}
    _write_stage1_budgets(artifacts_dir, per_layer_budget, redundancy=redundancy)
    for li, (total, _cur, mcp) in layers.items():
        _write_merge_json(partial_dir, li, total, mcp)
    if blacklist is not None:
        _write_blacklist(artifacts_dir, blacklist)
    return artifacts_dir


# ---------------------------------------------------------------------------
# (a) the GLOBAL kept-expert count is conserved
# ---------------------------------------------------------------------------
def test_global_budget_conserved(tmp_path):
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

    # The written artifact also conserves the global total.
    written = json.loads(out_path.read_text())
    new_budgets = {int(k): v for k, v in written["per_layer_target_experts"].items()}
    assert sum(new_budgets.values()) == 4 * 5
    assert written["achieved_budget"] == 4 * 5
    # Provenance records the conserved global budget.
    assert written["budget_retune"]["global_budget_conserved"] == 4 * 5
    assert written["budget_retune"]["floor_divisor"] == 2


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
# (c) re-solve moves budget from low-damage to high-damage layers
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

    # Cheapest layer drained to floor; most-expensive filled to ceiling.
    assert result.new_budgets[0] < result.old_budgets[0]
    assert result.new_budgets[3] > result.old_budgets[3]
    assert result.new_budgets[0] <= result.new_budgets[3]
    assert result.predicted_damage_after < result.predicted_damage_before
    assert result.transfers > 0


def test_resolve_drains_cheapest_layer_fully_first(tmp_path):
    # Constant-marginal optimum: with a steep gradient the cheapest layer is
    # drained all the way to its floor before the next-cheapest is touched.
    layers = {
        0: (8, 6, 0.01),   # cheapest -> drained to floor 4
        1: (8, 6, 0.02),
        2: (8, 6, 0.03),
        3: (8, 6, 5.00),   # dominant cost -> filled to ceiling 8
    }
    _make_run(tmp_path, layers)
    result, _ = retune_from_artifacts(tmp_path)
    # Global budget 24, 4 layers of ceiling 8 -> remove 8 experts. Cheapest
    # (layer 0) sheds 8-4=4, then layer 1 sheds 4. Layers 2 and 3 stay at 8.
    assert result.new_budgets[0] == 4
    assert result.new_budgets[1] == 4
    assert result.new_budgets[2] == 8
    assert result.new_budgets[3] == 8
    assert sum(result.new_budgets.values()) == 24


# ---------------------------------------------------------------------------
# (d) the N//K floor opens donor freedom below N//2
# ---------------------------------------------------------------------------
def test_floor_divisor_opens_donor_freedom(tmp_path):
    # Every merged layer sits exactly at N//2 == 4 (GRAPE's bimodal output),
    # and the unmerged layer is at the ceiling. With K=2 the merged layers
    # are floored and cannot donate further. With K=4 the floor drops to 2,
    # so the cheap merged layer CAN donate to the expensive one.
    layers = {
        0: (8, 4, 0.01),   # merged, at N//2, cheap
        1: (8, 4, 0.90),   # merged, at N//2, expensive
        2: (8, 8, 0.30),   # unmerged at ceiling (carries a signal here)
    }
    # K=2: floors are all 4; layer 0 already at floor, layer 1 already at
    # floor -> only layer 2 can shed. Re-solve removes from layer 2 only.
    _make_run(tmp_path, layers)
    res_k2, _ = retune_from_artifacts(tmp_path, output_path=tmp_path / "k2.json")
    assert res_k2.new_budgets[0] == 4  # cannot drop below N//2
    assert res_k2.floor_divisor == 2

    # K=4: floor drops to 2. Now the cheap merged layer 0 can donate below
    # N//2. Total budget is 16; the optimum drains layer 0 toward its floor 2.
    res_k4, _ = retune_from_artifacts(
        tmp_path, output_path=tmp_path / "k4.json", floor_divisor=4
    )
    assert res_k4.floor_divisor == 4
    assert res_k4.new_budgets[0] < 4, "K=4 must let the cheap layer drop below N//2"
    assert res_k4.new_budgets[0] >= 2, "but never below the N//4 floor"
    assert sum(res_k4.new_budgets.values()) == 16  # global budget still conserved


def test_floor_divisor_threaded_to_assemble_layers():
    # Unit-level: assemble_layers must compute the floor from floor_divisor.
    payload = {
        "per_layer_target_experts": {"0": 8, "1": 8},
        "per_layer_redundancy": {"0": 0.0, "1": 0.0},
    }
    damage = {0: (8, 0.1), 1: (8, 0.2)}
    layers_k2 = assemble_layers(payload, damage, floor_divisor=2)
    assert {ld.layer_idx: ld.floor for ld in layers_k2} == {0: 4, 1: 4}
    layers_k4 = assemble_layers(payload, damage, floor_divisor=4)
    assert {ld.layer_idx: ld.floor for ld in layers_k4} == {0: 2, 1: 2}


# ---------------------------------------------------------------------------
# (e) BudgetInfeasibleError when floors make the global budget unreachable
# ---------------------------------------------------------------------------
#
# Note on reachability: assemble_layers enforces floor_l <= current_l <= N_l
# per layer, which implies sum_floor <= global_budget <= sum_ceil. So the
# whole-pipeline path (retune_from_artifacts) cannot itself produce an
# infeasible global budget through the loader. BudgetInfeasibleError is a
# defence-in-depth guard for retune_budgets when it is handed LayerDamage rows
# directly — e.g. the proposal's two-pass procedure that constructs layers
# programmatically, or any future caller. These unit tests exercise that guard.
def test_budget_infeasible_below_sum_of_floors():
    # Global budget below sum(floor_l): no allocation respecting the floors
    # can be that small.
    payload = {
        "per_layer_target_experts": {"0": 8, "1": 8},
        "per_layer_redundancy": {"0": 0.0, "1": 0.0},
    }
    damage = {0: (8, 0.1), 1: (8, 0.2)}
    layers = assemble_layers(payload, damage, floor_divisor=2)  # floors 4,4
    # Drop one layer's budget below its floor *after* assembly to isolate the
    # global-feasibility guard (assemble_layers rejects below-floor inputs).
    layers[0].current_budget = 2  # global budget now 10 < sum_floor 8? no -> 10>8
    layers[1].current_budget = 2  # global budget now 4 < sum_floor 8
    with pytest.raises(BudgetInfeasibleError, match="feasible range"):
        retune_budgets(layers, floor_divisor=2)


def test_budget_infeasible_above_sum_of_ceilings():
    # Global budget above sum(N_l): no allocation can be that large.
    payload = {
        "per_layer_target_experts": {"0": 8, "1": 8},
        "per_layer_redundancy": {"0": 0.0, "1": 0.0},
    }
    damage = {0: (8, 0.1), 1: (8, 0.2)}
    layers = assemble_layers(payload, damage, floor_divisor=2)
    layers[0].current_budget = 20  # global budget 28 > sum_ceil 16
    with pytest.raises(BudgetInfeasibleError, match="feasible range"):
        retune_budgets(layers, floor_divisor=2)


def test_budget_feasible_at_sum_of_ceilings():
    # A global budget exactly at sum(N_l) is feasible (every layer untouched).
    payload = {
        "per_layer_target_experts": {"0": 8, "1": 8},
        "per_layer_redundancy": {"0": 0.0, "1": 0.0},
    }
    damage = {0: (8, 0.1), 1: (8, 0.2)}
    layers = assemble_layers(payload, damage, floor_divisor=2)
    res = retune_budgets(layers, floor_divisor=2)
    assert res.new_budgets == {0: 8, 1: 8}
    assert res.transfers == 0


# ---------------------------------------------------------------------------
# (f) the redundancy prior scores signal-less layers
# ---------------------------------------------------------------------------
def test_redundancy_prior_reallocates_signalless_layers(tmp_path):
    # Layers 0,1 have NO measured signal (GRAPE protected them at N). The
    # redundancy prior must let the re-solve treat them: layer 0 is highly
    # redundant (R̃=0.0 -> cheap to merge), layer 1 is diverse (R̃=1.0 ->
    # expensive). Layers 2,3 carry measured costs.
    layers = {
        0: (8, 8, None),    # no signal, very redundant -> should be drained
        1: (8, 8, None),    # no signal, diverse        -> should be protected
        2: (8, 6, 0.05),    # cheap measured
        3: (8, 6, 0.80),    # expensive measured
    }
    redundancy = {0: 0.0, 1: 1.0, 2: 0.5, 3: 0.5}
    _make_run(tmp_path, layers, redundancy=redundancy)
    result, out_path = retune_from_artifacts(tmp_path)

    # The signal-less layers are flagged as cost-predicted.
    assert sorted(result.layers_predicted) == [0, 1]
    assert sorted(result.layers_without_signal) == [0, 1]
    # The redundant signal-less layer 0 is drained (cheap prior); the diverse
    # signal-less layer 1 is left at the ceiling (expensive prior).
    assert result.new_budgets[0] < 8, "redundant signal-less layer must be drainable"
    assert result.new_budgets[1] == 8, "diverse signal-less layer must be protected"
    # Global budget conserved.
    assert sum(result.new_budgets.values()) == sum(result.old_budgets.values())
    # Provenance records the measured/predicted split.
    prov = json.loads(out_path.read_text())["budget_retune"]
    assert prov["n_layers_predicted"] == 2
    assert prov["n_layers_measured"] == 2
    assert sorted(prov["layers_cost_predicted"]) == [0, 1]


def test_signalless_layer_without_redundancy_is_conservative(tmp_path):
    # A signal-less layer with NO redundancy value at all must be treated as
    # the most-expensive layer (conservative) -> never drained unjustified.
    layers = {
        0: (8, 8, None),    # no signal, no redundancy -> max cost -> protected
        1: (8, 6, 0.05),    # cheap measured -> drained
        2: (8, 6, 0.80),    # expensive measured
    }
    # per_layer_redundancy intentionally omits layer 0.
    partial = tmp_path / "_stage2_partial"
    partial.mkdir(parents=True)
    payload = {
        "per_layer_target_experts": {"0": 8, "1": 6, "2": 6},
        "per_layer_redundancy": {"1": 0.5, "2": 0.5},  # no entry for layer 0
        "achieved_budget": 20,
    }
    (tmp_path / "stage1_budgets.json").write_text(json.dumps(payload), encoding="utf-8")
    for li, (total, _cur, mcp) in layers.items():
        _write_merge_json(partial, li, total, mcp)
    result, _ = retune_from_artifacts(tmp_path)
    # Layer 0 (no signal, no prior) treated as most expensive -> kept at ceiling.
    assert result.new_budgets[0] == 8
    assert 0 in result.layers_predicted


# ---------------------------------------------------------------------------
# (g) the default path (K=2, no prior) is unchanged
# ---------------------------------------------------------------------------
def test_default_path_matches_measured_only_solve(tmp_path):
    # When every layer carries a measured signal, the redundancy prior is
    # never consulted; the K=2 default solve depends only on measured costs.
    # Two runs with the SAME measured costs but DIFFERENT redundancy values
    # must produce identical allocations — proving the default path ignores
    # the prior entirely.
    layers = {
        0: (8, 6, 0.01),
        1: (8, 6, 0.04),
        2: (8, 6, 0.30),
        3: (8, 6, 0.95),
    }
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _make_run(run_a, layers, redundancy={k: 0.0 for k in layers})
    _make_run(run_b, layers, redundancy={k: 0.9 for k in layers})
    res_a, _ = retune_from_artifacts(run_a)
    res_b, _ = retune_from_artifacts(run_b)
    assert res_a.new_budgets == res_b.new_budgets
    assert res_a.layers_predicted == []
    assert res_b.layers_predicted == []


def test_idempotent_second_pass(tmp_path):
    # Re-running the re-solve on its own output (same measured damage) is a
    # fixed point: the allocation does not change.
    layers = {
        0: (8, 6, 0.01),
        1: (8, 6, 0.04),
        2: (8, 6, 0.30),
        3: (8, 6, 0.95),
    }
    _make_run(tmp_path, layers)
    result1, _ = retune_from_artifacts(tmp_path)

    run2 = tmp_path / "run2"
    partial2 = run2 / "_stage2_partial"
    partial2.mkdir(parents=True)
    _write_stage1_budgets(run2, result1.new_budgets)
    for li, (total, _cur, mcp) in layers.items():
        _write_merge_json(partial2, li, total, mcp)

    result2, _ = retune_from_artifacts(run2)
    assert result2.new_budgets == result1.new_budgets
    assert result2.transfers == 0
    assert result2.predicted_damage_after == pytest.approx(
        result2.predicted_damage_before
    )


# ---------------------------------------------------------------------------
# Honesty: no usable MEASURED damage signal -> refuse, do not invent
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


def test_zero_cost_treated_as_no_signal(tmp_path):
    # mean_cost_per_pair == 0.0 carries no usable gradient -> treated as
    # signal-less; with no other signal the tool refuses.
    layers = {
        0: (8, 5, 0.0),
        1: (8, 5, 0.0),
    }
    _make_run(tmp_path, layers)
    with pytest.raises(NoDamageSignalError):
        retune_from_artifacts(tmp_path)


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
    # Expensive layers (2,3) should not lose budget; cheapest (0) should not gain.
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
    # Stage-1 budget of 3 for an 8-expert layer is below the K=2 floor 4.
    layers = {0: (8, 3, 0.1), 1: (8, 6, 0.2)}
    _make_run(tmp_path, layers)
    with pytest.raises(ValueError, match="below the floor"):
        retune_from_artifacts(tmp_path)


def test_input_below_floor_accepted_with_larger_divisor(tmp_path):
    # The same budget of 3 IS above the K=4 floor 2 — a deliberately lowered
    # floor accepts an input the K=2 floor would reject.
    layers = {0: (8, 3, 0.1), 1: (8, 6, 0.2)}
    _make_run(tmp_path, layers)
    result, _ = retune_from_artifacts(tmp_path, floor_divisor=4)
    assert result.floor_divisor == 4
    for li in layers:
        assert result.new_budgets[li] >= 2  # N//4 floor


def test_total_experts_derived_from_freq_length(tmp_path):
    # The per-layer ceiling must come from len(freq), not from any constant.
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()
    _write_merge_json(partial_dir, 0, total_experts=13, mean_cost_per_pair=0.1)
    damage = load_stage2_damage(tmp_path)
    assert damage[0][0] == 13  # total_experts == len(freq)


def test_format_version_mismatch_rejected(tmp_path):
    # A pre-v2 merge JSON must be rejected loudly, not silently misparsed.
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()
    _write_stage1_budgets(tmp_path, {0: 5})
    _write_merge_json(partial_dir, 0, 8, 0.1, format_version=1)
    with pytest.raises(ValueError, match="format_version"):
        retune_from_artifacts(tmp_path)


def test_floor_divisor_below_one_rejected(tmp_path):
    layers = {0: (8, 5, 0.1), 1: (8, 5, 0.2)}
    _make_run(tmp_path, layers)
    with pytest.raises(ValueError, match="floor_divisor"):
        retune_from_artifacts(tmp_path, floor_divisor=0)


# ---------------------------------------------------------------------------
# Blacklist-aware floor: a layer is never dropped below its protected-expert
# count, even when that exceeds N//K (model-agnostic — heavy-blacklist models).
# ---------------------------------------------------------------------------
def test_load_protected_counts(tmp_path):
    path = _write_blacklist(tmp_path, {0: 3, 2: 0, 5: 7})
    assert load_protected_counts(path) == {0: 3, 2: 0, 5: 7}


def test_load_protected_counts_rejects_bad_artifact(tmp_path):
    bad = tmp_path / "stage1_blacklist.json"
    bad.write_text(json.dumps({"not_blacklist": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="blacklist"):
        load_protected_counts(bad)


def test_blacklist_raises_per_layer_floor(tmp_path):
    # Layer 0 is the cheap donor; 6 of its 8 experts are protected, so its
    # floor is max(8//2, 6) == 6, not 4. The re-solve must stop draining at 6.
    layers = {
        0: (8, 8, 0.01),   # cheap donor
        1: (8, 4, 0.90),   # expensive recipient, at its N//2 floor
    }
    _make_run(tmp_path, layers, blacklist={0: 6, 1: 0})
    result, _ = retune_from_artifacts(tmp_path)
    assert result.new_budgets[0] == 6, "protected floor must stop the drain at 6"
    assert result.new_budgets[1] == 6
    assert sum(result.new_budgets.values()) == 12  # global total conserved


def test_no_blacklist_drains_to_half_floor(tmp_path):
    # Same layers WITHOUT a blacklist artifact: layer 0 drains to N//2 == 4.
    layers = {
        0: (8, 8, 0.01),
        1: (8, 4, 0.90),
    }
    _make_run(tmp_path, layers)  # no blacklist
    result, _ = retune_from_artifacts(tmp_path)
    assert result.new_budgets[0] == 4
    assert result.new_budgets[1] == 8


def test_protected_count_above_current_budget_rejected(tmp_path):
    # Layer 0 keeps 5 experts but 6 are protected -> the input itself violates
    # the blacklist-raised floor (6); retune must refuse rather than proceed.
    layers = {
        0: (8, 5, 0.10),
        1: (8, 6, 0.20),
    }
    _make_run(tmp_path, layers, blacklist={0: 6, 1: 0})
    with pytest.raises(ValueError, match="below the floor"):
        retune_from_artifacts(tmp_path)
