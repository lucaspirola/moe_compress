"""Config-grep + driver-config tests for the 6-model REAP/REAM probe.

Asserts the GENERATED config dicts (build_probe_config on the faithful base)
carry every paper-core disable sentinel (§5), the REAM by-the-book params (§4),
the per-arm windowing (§1), the §3 keep==166 derivation intent, and the
uniform-166 budget pin (§2). All CPU, no model load.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from moe_compress.run_probe import (
    PROBE_PRUNE_FRACTION,
    PROBE_SURVIVORS,
    build_probe_config,
    is_heal_arm,
    pick_winner,
    probe_rows,
    seed_stage1_artifacts,
    write_uniform_budget,
)


def _faithful_base() -> dict:
    p = (Path(__file__).parent.parent / "configs"
         / "qwen36_35b_a3b_reap_faithful.yaml")
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Row matrix
# ---------------------------------------------------------------------------

def test_probe_rows_are_six_groups_x_arms():
    rows = probe_rows()
    assert [r[0] for r in rows] == [
        "reap-base", "reap-heal25", "reap-rkd",
        "ream-base", "ream-heal25", "ream-rkd",
    ]
    assert len(rows) == 6


# ---------------------------------------------------------------------------
# §5 paper-core OFF set — ALL 6 generated configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row_id,group,arm", probe_rows())
def test_all_refinements_off(row_id, group, arm):
    s2 = build_probe_config(_faithful_base(), group=group, arm=arm)["stage2_reap_ream"]
    assert s2["em_refinement_rounds"] == 0
    assert s2["expert_distill_steps"] == 0
    assert s2["merge_heal_enabled"] is False
    assert s2["two_opt_refine"] is False
    # capacity gate inert via cost_alignment="pre".
    assert s2["cost_alignment"] == "pre"


@pytest.mark.parametrize("row_id,group,arm", probe_rows())
def test_survivor_guard_enabled(row_id, group, arm):
    s2 = build_probe_config(_faithful_base(), group=group, arm=arm)["stage2_reap_ream"]
    assert s2["assert_survivors_match_target"] is True


# ---------------------------------------------------------------------------
# §3 REAP keep==166 intent + §4 REAM by-the-book
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["base", "heal25", "rkd"])
def test_reap_rows_faithful_prune_035(arm):
    s2 = build_probe_config(_faithful_base(), group="reap", arm=arm)["stage2_reap_ream"]
    assert s2["prune_mode"] == "faithful_prune"
    assert s2["prune_fraction"] == PROBE_PRUNE_FRACTION
    # round((1-0.35)*256) = 166 — the keep contract this fraction encodes.
    assert round((1.0 - s2["prune_fraction"]) * 256) == PROBE_SURVIVORS


@pytest.mark.parametrize("arm", ["base", "heal25", "rkd"])
def test_ream_rows_by_the_book(arm):
    s2 = build_probe_config(_faithful_base(), group="ream", arm=arm)["stage2_reap_ream"]
    assert s2["prune_mode"] == "merge"
    # max_merge_group_size counts NON-centroids → upstream group_size=16 ⇔ 15.
    assert s2["max_merge_group_size"] == 15
    assert s2["sequential_reprofile"] is True
    assert s2["cost_alignment"] == "pre"
    assert s2["ream"]["frequency_weighted_merge"] is True
    # skip-merge floor OFF (100.0) so merges proceed.
    assert s2["skip_merge_percentile"] == 100.0
    # mutual-exclusion: sequential_reprofile=true + profile_sidecar=true rejected.
    assert s2["profile_sidecar"]["enabled"] is False


# ---------------------------------------------------------------------------
# §1 arm windowing + eval-mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("group", ["reap", "ream"])
def test_base_arm_skip_intermediate_and_thermometer(group):
    cfg = build_probe_config(_faithful_base(), group=group, arm="base")
    assert cfg["pipeline"]["skip_intermediate_stages"] is True
    assert cfg["pipeline"]["evaluator"] == "stage6alt"
    assert cfg["stage6_validate"]["mode"] == "thermometer"


@pytest.mark.parametrize("group", ["reap", "ream"])
@pytest.mark.parametrize("arm", ["heal25", "rkd"])
def test_heal_arms_no_skip_and_explicit_thermometer(group, arm):
    cfg = build_probe_config(_faithful_base(), group=group, arm=arm)
    # heal arms run Stage 2.5; skip-intermediate eval-override does NOT fire,
    # so mode MUST be set directly (plan §1b).
    assert cfg["pipeline"]["skip_intermediate_stages"] is False
    assert cfg["stage6_validate"]["mode"] == "thermometer"


@pytest.mark.parametrize("group", ["reap", "ream"])
def test_rkd_arm_paper_dials_only(group):
    cfg = build_probe_config(_faithful_base(), group=group, arm="rkd")
    assert cfg["stage5_router_kd"]["rkd_recipe"] == "paper_dials_only"


@pytest.mark.parametrize("group", ["reap", "ream"])
def test_heal25_arm_explicit_current(group):
    cfg = build_probe_config(_faithful_base(), group=group, arm="heal25")
    # heal25 = the DEPRECATED current/production dials. Since the 2026-06-09
    # default flip (absent rkd_recipe → "paper_dials_only"), heal25 must pin
    # "current" EXPLICITLY, else it would silently become a second paper arm.
    assert cfg["stage5_router_kd"]["rkd_recipe"] == "current"


def test_is_heal_arm():
    assert is_heal_arm("heal25") and is_heal_arm("rkd")
    assert not is_heal_arm("base")


# ---------------------------------------------------------------------------
# Eval corpus inherited from the faithful base (wikitext + pinned SHA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row_id,group,arm", probe_rows())
def test_thermo_corpus_wikitext_with_pinned_sha(row_id, group, arm):
    cfg = build_probe_config(_faithful_base(), group=group, arm=arm)
    assert cfg["stage6_validate"]["thermometer"]["corpus"] == "wikitext"
    sha = cfg["stage6_validate"]["dataset_revisions"]["wikitext_ppl"]
    assert isinstance(sha, str) and len(sha) >= 8


# ---------------------------------------------------------------------------
# §2 uniform-166 budget pin
# ---------------------------------------------------------------------------

def _fake_shared(tmp_path: Path) -> Path:
    shared = tmp_path / "_shared"
    shared.mkdir()
    # Non-uniform per-layer budgets, mixed-type keys → all must become 166.
    (shared / "stage1_budgets.json").write_text(json.dumps({
        "per_layer_target_experts": {"0": 200, "1": 150, "2": 180},
        "per_layer_redundancy": {"0": 0.1, "1": 0.2, "2": 0.3},
    }))
    (shared / "stage1_blacklist.json").write_text(json.dumps(
        {"blacklist": {"0": [3]}}))
    (shared / "budget_decomposition.json").write_text(json.dumps({"x": 1}))
    return shared


def test_write_uniform_budget_pins_166_every_layer(tmp_path):
    shared = _fake_shared(tmp_path)
    out = tmp_path / "out_budgets.json"
    payload = write_uniform_budget(shared / "stage1_budgets.json", out)
    assert payload["per_layer_target_experts"] == {"0": 166, "1": 166, "2": 166}
    # Schema preserved (redundancy untouched).
    assert payload["per_layer_redundancy"] == {"0": 0.1, "1": 0.2, "2": 0.3}
    on_disk = json.loads(out.read_text())
    assert on_disk["per_layer_target_experts"] == {"0": 166, "1": 166, "2": 166}


def test_write_uniform_budget_rejects_empty_layer_set(tmp_path):
    shared = tmp_path / "_shared"
    shared.mkdir()
    (shared / "stage1_budgets.json").write_text(json.dumps(
        {"per_layer_target_experts": {}}))
    with pytest.raises(RuntimeError, match="per_layer_target_experts"):
        write_uniform_budget(shared / "stage1_budgets.json",
                             tmp_path / "o.json")


@pytest.mark.parametrize("group", ["reap", "ream"])
def test_seed_stage1_artifacts_writes_166_pin_for_both_groups(tmp_path, group):
    shared = _fake_shared(tmp_path)
    row_dir = tmp_path / f"{group}-base"
    seed_stage1_artifacts(row_dir, shared, group=group)
    budgets = json.loads((row_dir / "stage1_budgets.json").read_text())
    assert set(budgets["per_layer_target_experts"].values()) == {166}
    # blacklist + decomposition copied verbatim.
    assert (row_dir / "stage1_blacklist.json").exists()
    assert (row_dir / "budget_decomposition.json").exists()


# ---------------------------------------------------------------------------
# Winner pick (lowest student_bpt)
# ---------------------------------------------------------------------------

def test_pick_winner_lowest_bpt():
    results = {
        "reap-base": {"student_bpt": 3.5},
        "ream-heal25": {"student_bpt": 3.1},
        "reap-rkd": {"student_bpt": 3.9},
    }
    assert pick_winner(results) == "ream-heal25"


def test_pick_winner_none_when_no_finite():
    assert pick_winner({}) is None
    assert pick_winner({"a": {"student_bpt": float("inf")}}) is None
    assert pick_winner({"a": {}}) is None
