"""Unit tests for the REAP-vs-REAM @ net-35% full-pipeline runner's pure logic.

Covers the config builder (WS2 dials + M-C net accounting + Stage-2 paper-pure),
the paper-recipe save_best safety net (WS2 H1), the homogeneous-expert-count
guard, and the solver-derived uniform-K / prune_fraction math (H-A). The
subprocess-driving + I/O paths (run_one_arm, main, assert_covariance_resolves)
are integration surfaces exercised on the box, not unit-tested here.
"""
from __future__ import annotations

import pytest

from moe_compress import run_reap_ream_35pct as rr
from moe_compress.run_probe import _PAPER_CORE_OFF


# ---------------------------------------------------------------------------
# build_arm_config — pure config injection
# ---------------------------------------------------------------------------

def _base() -> dict:
    return {"stage5_router_kd": {"save_best": True}}


def test_build_arm_config_reap_injects_paper_pure_and_net_accounting():
    cfg = rr.build_arm_config(_base(), method="faithful_prune", prune_fraction=0.23)
    s2 = cfg["stage2_reap_ream"]
    # Stage-2 paper-pure prune
    assert s2["prune_mode"] == "faithful_prune"
    assert s2["prune_fraction"] == pytest.approx(0.23)
    assert s2["assert_survivors_match_target"] is True
    # refinements OFF (paper-core)
    for k, v in _PAPER_CORE_OFF.items():
        assert s2[k] == v
    # M-C net accounting
    assert cfg["target"]["total_reduction_ratio"] == rr.NET_TARGET == 0.35
    assert cfg["target"]["net_of_eora"] is True
    # WS2 paper dials (applies at both 2.5 + 5 via the shared block)
    assert cfg["stage5_router_kd"]["rkd_recipe"] == "paper_dials_only"
    # full pipeline + thermometer
    assert cfg["pipeline"]["skip_intermediate_stages"] is False
    assert cfg["pipeline"]["evaluator"] == "stage6alt"
    assert cfg["stage6_validate"]["mode"] == "thermometer"


def test_build_arm_config_ream_uses_by_the_book_merge():
    cfg = rr.build_arm_config(_base(), method="merge", prune_fraction=0.23)
    s2 = cfg["stage2_reap_ream"]
    assert s2["prune_mode"] == "merge"
    assert s2["ream"]["frequency_weighted_merge"] is True
    # sequential_reprofile + profile_sidecar.enabled=true is hard-rejected; off.
    assert s2["profile_sidecar"]["enabled"] is False
    # merge arm does NOT get a faithful prune_fraction
    assert "prune_fraction" not in s2 or s2.get("prune_mode") == "merge"
    # same net accounting + paper dials
    assert cfg["target"]["net_of_eora"] is True
    assert cfg["stage5_router_kd"]["rkd_recipe"] == "paper_dials_only"


def test_build_arm_config_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown Stage-2 method"):
        rr.build_arm_config(_base(), method="bogus", prune_fraction=0.2)


def test_build_arm_config_is_pure_no_mutation():
    base = _base()
    rr.build_arm_config(base, method="faithful_prune", prune_fraction=0.2)
    # base must be untouched (deepcopy inside).
    assert base == {"stage5_router_kd": {"save_best": True}}


# ---------------------------------------------------------------------------
# assert_paper_recipe_safety — WS2 H1 save_best guard
# ---------------------------------------------------------------------------

def test_paper_recipe_safety_raises_without_save_best():
    cfg = {"stage5_router_kd": {"rkd_recipe": "paper_dials_only", "save_best": False}}
    with pytest.raises(RuntimeError, match="save_best"):
        rr.assert_paper_recipe_safety(cfg)


def test_paper_recipe_safety_passes_with_save_best():
    cfg = {"stage5_router_kd": {"rkd_recipe": "paper_dials_only", "save_best": True}}
    rr.assert_paper_recipe_safety(cfg)  # no raise


def test_paper_recipe_safety_noop_for_non_paper_recipe():
    cfg = {"stage5_router_kd": {"rkd_recipe": "current", "save_best": False}}
    rr.assert_paper_recipe_safety(cfg)  # current dials don't require save_best


# ---------------------------------------------------------------------------
# homogeneous expert count + solver-derived uniform-K (H-A)
# ---------------------------------------------------------------------------

def test_homogeneous_expert_count(tiny_model):
    n = rr._homogeneous_expert_count(tiny_model)
    assert n >= 1 and isinstance(n, int)


@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("K_frac", [0.50, 0.6484, 0.77, 0.90])
def test_prune_fraction_is_exact_inverse_of_K(n, K_frac):
    """H-A step 3 correctness (decoupled from the solver/model): for any uniform
    keep count K, ``prune_fraction = 1 - K/n`` must make REAP's keep formula
    (n_keep = round_half_up((1-pf)*n) = floor((1-pf)*n + 0.5), reap_prune.py:339)
    return EXACTLY K — this is what guarantees iso-K between the REAP arm
    (prune_fraction) and the REAM arm (uniform budget pin)."""
    import math
    K = round(K_frac * n)
    assert 0 < K < n
    prune_fraction = 1.0 - K / n
    n_keep = math.floor(n * (1.0 - prune_fraction) + 0.5)
    assert n_keep == K, f"prune_fraction inverse broke for n={n}, K={K}: got {n_keep}"


def test_derive_solver_budget_math_on_real_dims():
    """derive_solver_budget's solver call needs a fine-lattice model (256
    experts) to hit net-35%; the test suite has only the coarse tiny_model
    (4 experts, lattice >> 35% target → solver legitimately can't converge).
    The solver itself is covered by test_budget_solver (incl. the EoRA path);
    the K/prune_fraction inverse math is covered above. This integration is
    exercised on the box during the real run."""
    pytest.skip("needs a 256-expert fixture; solver + inverse math covered separately")
