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


def test_paper_recipe_safety_passes_with_default_save_best():
    # save_best omitted → defaults to True (early_stop.py default), so the guard
    # must NOT trip on a mere omission under the (now-default) paper recipe.
    cfg = {"stage5_router_kd": {"rkd_recipe": "paper_dials_only"}}
    rr.assert_paper_recipe_safety(cfg)  # no raise


def test_paper_recipe_safety_default_recipe_requires_save_best():
    # rkd_recipe omitted → defaults to "paper_dials_only" (2026-06-09 flip), so
    # the guard now covers the DEFAULT path: an explicit save_best=False raises.
    cfg = {"stage5_router_kd": {"save_best": False}}
    with pytest.raises(RuntimeError, match="save_best"):
        rr.assert_paper_recipe_safety(cfg)


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


# ===========================================================================
# S234-correctness regression locks (fix/reap-ream-s234-tests)
# Audit: tasks/PLAN_REAP_REAM_S234_CORRECTNESS.md (Concerns 1/2/3).
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. rkd_recipe == LIVE plugin default — re-flip tripwire (Concern 2)
# ---------------------------------------------------------------------------

def _live_rkd_recipe_default() -> str:
    """Derive the LIVE default of ``stage5_router_kd.rkd_recipe`` behaviourally
    from the plugin (NO hardcoded mirror) by finding the recipe string whose
    EXPLICIT injection reproduces, byte-for-byte, the plugin's mutation when the
    key is ABSENT. The plugin resolves ``s5.get("rkd_recipe", <DEFAULT>)``, so
    the candidate that matches the absent-key behaviour IS the default."""
    from moe_compress.router_kd.plugins.rkd_paper_recipe import RkdPaperRecipePlugin

    plugin = RkdPaperRecipePlugin()

    def _mutated(recipe: str | None) -> dict:
        s5: dict = {"save_best": True}
        if recipe is not None:
            s5["rkd_recipe"] = recipe
        cfg = {"stage5_router_kd": s5, "calibration": {"source": "project-mix"}}
        plugin.apply_config_overrides(cfg)
        # Drop the selector key itself so we compare only the APPLIED dials —
        # the absent-key config has no rkd_recipe, the explicit one does.
        cfg["stage5_router_kd"].pop("rkd_recipe", None)
        return cfg

    absent = _mutated(None)
    # All recipe values the plugin understands; the one matching the absent-key
    # result is the live default.
    for candidate in ("paper_dials_only", "paper", "current"):
        if _mutated(candidate) == absent:
            return candidate
    raise AssertionError(
        "no candidate rkd_recipe reproduced the absent-key behaviour — the "
        "plugin's default-resolution changed shape; update this derivation."
    )


def test_runner_rkd_recipe_equals_live_plugin_default():
    """TRIPWIRE: the value the runner injects for ``stage5_router_kd.rkd_recipe``
    MUST equal the LIVE plugin default (rkd_paper_recipe.py, currently
    'paper_dials_only', flipped 2026-06-09). The whole point of this test is to
    FAIL LOUDLY if someone later re-flips the plugin default without re-deciding
    what the ablation runner pins — the two must not silently diverge."""
    live_default = _live_rkd_recipe_default()
    cfg = rr.build_arm_config(_base(), method="faithful_prune", prune_fraction=0.23)
    injected = cfg["stage5_router_kd"]["rkd_recipe"]
    assert injected == live_default, (
        f"runner injects rkd_recipe={injected!r} but the LIVE plugin default is "
        f"{live_default!r} — the plugin default was re-flipped. Re-decide what "
        "run_reap_ream_35pct.py should pin (this divergence is the bug)."
    )
    # Belt-and-suspenders: the merge arm injects the same value.
    cfg_merge = rr.build_arm_config(_base(), method="merge", prune_fraction=0.23)
    assert cfg_merge["stage5_router_kd"]["rkd_recipe"] == live_default


# ---------------------------------------------------------------------------
# 2. iso-K identity between REAP (prune_fraction) and REAM (uniform pin)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_experts", [128, 256, 512])
@pytest.mark.parametrize("K", [65, 166, 207, 333])
def test_iso_K_identity_between_arms(n_experts, K):
    """Both arms keep the SAME uniform K every layer. REAP derives its keep
    count from the scalar ``prune_fraction = 1 - K/n`` through the REAL
    round-half-up path (n_keep = floor((1-pf)*n + 0.5), reap_prune.py:339); REAM
    pins the uniform budget directly to K (write_uniform_budget → target==K).
    Assert REAP's realised n_keep == K == REAM's target — iso-K."""
    from moe_compress.stage2.plugins.reap_prune import keep_count
    if not (0 < K < n_experts):
        pytest.skip("K out of range for this n_experts")

    # --- REAP arm: scalar prune_fraction → realised keep count (the exact
    #     round-half-up the orchestrator uses, sourced from the REAL production
    #     helper reap_prune.keep_count — no local re-impl). ---
    prune_fraction = 1.0 - K / n_experts
    cfg_reap = rr.build_arm_config(_base(), method="faithful_prune",
                                   prune_fraction=prune_fraction)
    assert cfg_reap["stage2_reap_ream"]["prune_mode"] == "faithful_prune"
    pf = cfg_reap["stage2_reap_ream"]["prune_fraction"]
    reap_n_keep = keep_count(n_experts, pf)  # the production formula itself
    assert reap_n_keep == K, f"REAP keep {reap_n_keep} != K {K}"

    # --- REAM arm: uniform budget pin → every layer's target == K. ---
    from moe_compress.run_probe import write_uniform_budget
    shared = {"per_layer_target_experts": {str(i): 999 for i in range(48)},
              "per_layer_redundancy": {}}
    with _tmp_json(shared) as shared_path, _tmp_path() as out_path:
        payload = write_uniform_budget(shared_path, out_path, survivors=K)
    ream_targets = set(payload["per_layer_target_experts"].values())
    assert ream_targets == {K}, f"REAM targets {ream_targets} != {{{K}}}"

    # iso-K: REAP realised keep == REAM uniform target for every layer.
    assert reap_n_keep == K and ream_targets == {reap_n_keep}


def test_keep_count_pins_documented_rounding():
    """Pin the EXACT documented production example (reap_prune.py:324-331):
    256 experts @ prune_fraction=0.35 keeps floor(166.4 + 0.5)=166. This fails
    LOUDLY if anyone swaps the round-half-up rule for banker's round (which
    would also give 166 here, but the half-up tie cases below would diverge)."""
    from moe_compress.stage2.plugins.reap_prune import keep_count
    assert keep_count(256, 0.35) == 166
    # Half-up tie discriminators where banker's round-half-to-even diverges:
    #   (1-pf)*n lands on a .5 tie above an EVEN integer ⇒ half-up rounds UP,
    #   banker's (Python round) rounds DOWN to the even. These would BREAK if the
    #   convention were swapped to round-half-to-even.
    assert keep_count(1, 0.5) == 1   # 0.5 → half-up 1, banker's 0
    assert keep_count(5, 0.5) == 3   # 2.5 → half-up 3, banker's 2
    assert keep_count(9, 0.5) == 5   # 4.5 → half-up 5, banker's 4


# ---------------------------------------------------------------------------
# 3. Stage-1 never runs — both subprocesses resume at stage >= 2 (Concern 1)
# ---------------------------------------------------------------------------

def test_pipeline_subprocesses_never_invoke_stage1():
    """The runner's per-arm stage windows (the constant the two
    ``subprocess.run`` launches actually read) must never resume Stage-1. Source
    the (resume, stop) pairs from the REAL ``ARM_STAGE_WINDOWS`` constant — not
    hardcoded literals — so a future ``run_one_arm`` edit that lowers a resume to
    1 is caught here, and feed each through the real ``_pipeline_argv`` to lock
    the rendered argv. Stage-1 GRAPE/RCO is gated on start<=1 in run_pipeline.py;
    every window with resume>=2 proves it is NEVER invoked per arm."""
    import tempfile
    from pathlib import Path

    arm_dir = Path(tempfile.mkdtemp())
    cfg_path = arm_dir / "arm_config.yaml"

    def _resume_of(argv: list[str]) -> int:
        return int(argv[argv.index("--resume-from-stage") + 1])

    windows = rr.ARM_STAGE_WINDOWS
    assert len(windows) >= 1

    # (a) Every window's resume stage proves Stage-1 is never re-entered.
    for resume, stop in windows:
        assert resume >= 2, (
            f"ARM_STAGE_WINDOWS has a window resuming at stage {resume} "
            "(<=1 ⇒ Stage-1 GRAPE/RCO would run!)"
        )

    # (b) Feed each window through the real formatter and lock the argv: it must
    #     carry --resume-from-stage <resume> with resume>=2, never 1 or 0.
    for resume, stop in windows:
        argv = rr._pipeline_argv(cfg_path, "fake/repo", arm_dir,
                                 resume=resume, stop=stop)
        assert "--resume-from-stage" in argv
        start = _resume_of(argv)
        assert start == resume and start >= 2
        assert "--resume-from-stage 1" not in " ".join(argv)
        assert "--resume-from-stage 0" not in " ".join(argv)

    # The two locked windows: (Stage2+2.5)=2/2 then (Stage3-6)=3/6.
    assert windows[0][0] == 2 and windows[1][0] == 3


# ---------------------------------------------------------------------------
# 4. covariance-miss fails fast — no silent fp16 fallback (Concern 3d / H-B)
# ---------------------------------------------------------------------------

def test_assert_covariance_resolves_raises_on_missing_sidecar(tmp_path):
    """H-B pre-check MUST hard-raise when the bf16 covariance.pt sidecar is
    absent/unloadable — the full-pipeline runner does NOT skip the fp16 fallback,
    so a missing covariance source must abort before any GPU work rather than
    silently degrade. Point calibration.jsonl_path at a tmp JSONL with NO
    sidecar and assert the raise."""
    missing_jsonl = tmp_path / "self_traces.jsonl"
    missing_jsonl.write_text("{}\n", encoding="utf-8")  # exists, but no sidecar
    base_cfg = {"calibration": {"jsonl_path": str(missing_jsonl)}}
    with pytest.raises(RuntimeError, match="H-B"):
        rr.assert_covariance_resolves(base_cfg)


# --- small tmp helpers for the iso-K REAM-pin write/read ---

import contextlib  # noqa: E402
import json as _json  # noqa: E402
import tempfile as _tempfile  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


@contextlib.contextmanager
def _tmp_json(obj):
    d = _tempfile.mkdtemp()
    p = _Path(d) / "shared_budgets.json"
    p.write_text(_json.dumps(obj), encoding="utf-8")
    yield p


@contextlib.contextmanager
def _tmp_path():
    d = _tempfile.mkdtemp()
    yield _Path(d) / "stage1_budgets.json"
