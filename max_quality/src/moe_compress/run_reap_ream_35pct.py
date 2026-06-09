"""Full-pipeline REAP-vs-REAM @ net-35% — the two "ours" arms (WS3).

Implements the APPROVED plan
``tasks/PLAN_SOLVER_S4_AND_REAP_REAM_35PCT.md`` (plan-review CLOSED, round 4
all-none). This module builds ONLY the two new "ours" arms — the paper
baselines (``reap-rkd`` 3.1686 / ``ream-rkd``) ALREADY EXIST from the prior
``run_probe.py`` ablation and are NOT rebuilt here.

GOAL (plan GOAL section). Scientific question: at iso-35% compression, does
*keeping more experts + compressing them with Stage-3 SVD + Stage-4 EoRA*
(``reap-s234`` / ``ream-s234``) beat *bluntly dropping/merging experts*
(paper, Stage-2 only — the existing reap-rkd/ream-rkd baselines)?

    arm          Stage 2 method                    pipeline
    reap-s234    faithful_prune  → K_solver/layer   2 → 2.5 → 3 → 4 → 5 → 6, net-35%
    ream-s234    merge (by-book) → K_solver/layer   2 → 2.5 → 3 → 4 → 5 → 6, net-35%

Both arms consume the SAME solver output (same uniform K_solver, same Stage-3
``sp``) → clean iso-compression; ONLY the Stage-2 method differs. K_solver is
the solver's net-35% expert-keep count, NOT the paper baselines' K=166 — it is
HIGHER (lighter Stage-2) because Stage-3 SVD + Stage-4 EoRA absorb part of the
35% (plan GOAL line 17).

How each plan item is realized here (all implemented):

  * **WS1 + H-A — solver-derived K.** ``derive_solver_budget`` runs the
    Stage-4-aware solver ONCE (net target 0.35, ``net_of_eora`` on,
    ``eora_overhead_pct`` from ``stage4_eora.compensation_budget_pct``,
    ``ep_sp_knob_ratio`` from ``target.expert_svd_ratio``) on the loaded base
    model → ``ep``, ``sp``; ``K = round((1 - ep) * n_experts)`` with
    ``n_experts`` read from the homogeneous model (asserted homogeneous).
  * **H-A iso-K for BOTH arms.** REAP arm injects
    ``stage2.prune_fraction = 1 - K/n_experts`` (exact inverse of
    ``n_keep = round_half_up((1-pf)*n)``, reap_prune.py:339). BOTH arms write
    the uniform-K ``stage1_budgets.json`` via ``write_uniform_budget(...,
    survivors=K)`` (reused from run_probe) and set
    ``assert_survivors_match_target: true`` — the survivor guard reads
    stage1_budgets.json for both arms, so REAP needs the pin too.
  * **Stage-2 by-the-paper-only.** REAP=``faithful_prune``; REAM=``merge`` with
    run_probe's ``_REAM_BY_THE_BOOK`` params; ``_PAPER_CORE_OFF`` (em=0,
    distill=0, merge_heal off, two_opt off) on BOTH. No other Stage-2 plugins.
  * **WS2 — paper router-kd dials.** Inject
    ``stage5_router_kd.rkd_recipe = "paper_dials_only"`` (honored at BOTH
    Stage 2.5 and Stage 5). Assert ``save_best`` is true (load-bearing safety
    net for the recipe's epochs=2; fail loudly if false).
  * **M-C — net accounting ON.** Inject ``target.total_reduction_ratio = 0.35``
    AND ``target.net_of_eora = true`` under ``config["target"]``. Base config
    carries ``compensation_budget_pct: 0.03``.
  * **H-B — Stage-4 covariance.** Do NOT set ``MOE_SKIP_STAGE2_COV_SAVE=1``
    (keeps the fp16 fallback). ``assert_covariance_resolves`` fails fast if
    ``load_covariance(jsonl)`` cannot resolve the bf16
    ``sidecars/<stem>/covariance.pt`` (a_storage_dtype auto-resolves to
    bfloat16 on the hit path — input_cov_cache.py).
  * **Full pipeline run.** Per arm: Stage 2 → 2.5 (one invocation, resume@2
    stop@2) then Stage 3 → 4 → 5 → 6 (resume@3 stop@6), mirroring run_probe's
    subprocess staging. Stage 6 = thermometer (stage6alt, ``student_bpt``).
  * **Output.** Each arm's ``student_bpt`` collected into a summary JSON +
    printed comparison line (reap-s234 vs ream-s234). Comparison vs the
    existing reap-rkd(3.1686)/ream-rkd baselines is done SEPARATELY (those are
    not fetched here).

DEVIATIONS from the plan (project rule: deviations live in the docstring):

  D1. **Stage staging shape.** The plan (WS3 step 7) says "Stage 2 → 2.5 → 3 →
      4 → 5 → 6". run_pipeline runs Stage 2 + auto-Stage-2.5 in ONE block and
      Stages 3-6 in another; Stage 2.5 is NOT a standalone ``--resume-from-stage``
      entry (STAGE_REGISTRY in run_pipeline omits it). So this runner uses TWO
      subprocesses per arm: (a) ``--resume-from-stage 2 --stop-after-stage 2``
      (Stage 2 + auto Stage 2.5 → stage2p5_final/), then (b)
      ``--resume-from-stage 3 --stop-after-stage 6`` (Stage 3→4→5→6, loading
      stage2p5_final/ per STAGE_REGISTRY[3]). This is the only run_pipeline-
      supported decomposition of "2 → 2.5 → 3 → 4 → 5 → 6" and is behaviorally
      identical to a single resume@2→stop@6 run (kept split for per-stage
      supervision + crash-resume granularity, matching run_probe's idiom).
      ``skip_intermediate_stages`` stays FALSE so Stages 3/4/5 actually run.

  D2. **No shared-prune dedup across arms.** run_probe shares one Stage-2 prune
      across a group's three arms (base/heal25/rkd differ only in Stage 2.5).
      Here the two arms are DIFFERENT Stage-2 methods (faithful_prune vs merge)
      AND each runs its own Stage 2.5 → 6, so there is nothing to share; each
      arm is one self-contained run. (run_probe's ``run_shared_prune`` is not
      reused — it would be a no-op here.)

  D3. **base model load for the solver.** The solver needs a real ``nn.Module``
      (it calls ``count_parameters`` / ``iter_moe_layers``). The plan says "run
      the solver on the loaded base model", so this runner loads the full base
      model ONCE up front (CPU-or-device per config) purely to derive K + sp,
      then frees it before the per-arm subprocess pipeline runs. This is the
      same model run_pipeline Stage 1 would load; loading it here avoids
      re-deriving the budget inside each arm's Stage 1 (we pin K instead).

  D4. **Solver run is the ONLY budget authority; Stage 1 is skipped per arm.**
      Both arms are seeded with a uniform-K ``stage1_budgets.json`` +
      blacklist + budget_decomposition (from a shared Stage-1 run, exactly like
      run_probe's ``seed_stage1_artifacts``), so the per-arm pipeline starts at
      Stage 2. The solver's ``sp`` flows to Stage 3 via the seeded
      ``budget_decomposition.json`` (run_pipeline loads it when start>1).
      NOTE: the shared Stage-1 artifacts (incl. budget_decomposition with the
      solver's sp) must be produced by a prior ``--stop-after-stage 1`` run with
      net_of_eora ON; this runner injects net_of_eora into each arm's config so
      a from-Stage-1 run would also be net-aware, but the iso-K pin overrides
      the per-layer expert counts regardless.

ASSUMPTIONS / OPEN QUESTIONS:

  A1. The shared Stage-1 artifacts under ``<probe-root>/_shared/`` were produced
      with ``net_of_eora: true`` so that the seeded ``budget_decomposition.json``
      carries the net-35% Stage-3 ``sp``. If the shared run was gross-only, the
      seeded sp lands ~0.3pp light — the runner WARNS but does not block (the
      iso-K pin still makes both arms identical, preserving the comparison).
  A2. ``calibration.jsonl_path`` in the base config points at the bf16 capture
      JSONL whose ``sidecars/<stem>/covariance.pt`` exists (the H-B pre-check
      asserts this). If absent the runner aborts before any GPU work.

Launch (after the bf16 input-cov capture is uploaded; same H200 box):

    python -m moe_compress.run_reap_ream_35pct \\
        --config configs/qwen36_35b_a3b_reap_faithful.yaml \\
        --model Qwen/Qwen3.6-35B-A3B \\
        --probe-root ./artifacts/reap_ream_35pct \\
        --num-sequences 4000

NOTHING here runs on GPU at import time. The config-builder + budget helpers
are pure; only ``main`` (and ``derive_solver_budget``) touch the model.
"""
from __future__ import annotations

import os

# Match run_probe.py / run_ablations.py pre-torch env hardening (allocator +
# inductor cache/worker fixes). Set BEFORE any torch import; correctness-neutral.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "8")
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHDYNAMO_CACHE_SIZE_LIMIT", "512")
os.environ.setdefault("TORCHDYNAMO_RECOMPILE_LIMIT", "512")

import argparse
import copy
import gc
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .budget import solver as budget_solver
from .utils.model_io import iter_moe_layers, save_json_artifact

# Reuse run_probe's proven pure helpers (no copy): the REAM-by-the-book params,
# the paper-core OFF set, the uniform-K budget pin, the Stage-1 artifact seeding,
# the dotted-key setter, and the BPT ranking helper.
from .run_probe import (
    _PAPER_CORE_OFF,
    _REAM_BY_THE_BOOK,
    _set_nested,
    seed_stage1_artifacts,  # internally pins the uniform-K budget via write_uniform_budget
    _student_bpt,
)

log = logging.getLogger(__name__)

# The two "ours" arms. ``method`` selects the Stage-2 mechanism; both share the
# SAME solver K + sp (iso-compression) and the SAME paper router-kd dials.
ARMS: tuple[tuple[str, str], ...] = (
    ("reap-s234", "faithful_prune"),
    ("ream-s234", "merge"),
)

# Net compression target (M-C) — these arms target NET-35% after Stage-4 EoRA.
NET_TARGET = 0.35

# Final Stage-6alt artifact filename — completion gate (mirrors run_probe).
STAGE6ALT_ARTIFACT = "stage6alt_eval.json"

# Existing paper baselines (for the printed reminder only — NOT fetched here).
_PAPER_BASELINE_NOTE = (
    "reap-rkd student_bpt=3.1686 / ream-rkd (existing run_probe baselines, "
    "K=166 Stage-2-only) — compare SEPARATELY; not fetched by this runner."
)


# ---------------------------------------------------------------------------
# WS1 + H-A — solver-derived uniform K + Stage-3 sp (touches the model)
# ---------------------------------------------------------------------------

def _homogeneous_expert_count(model) -> int:
    """Return the per-layer routed-expert count, asserting the model is
    homogeneous (every MoE layer has the same count — Qwen3.6 is 256/layer).

    H-A step 2 requires a single ``n_experts`` to translate the solver's ``ep``
    into a uniform per-layer ``K``. A non-homogeneous model would make
    ``K = round((1-ep)*n_experts)`` ill-defined, so we fail loudly.
    """
    counts = sorted({ref.num_routed_experts for ref in iter_moe_layers(model)})
    if not counts:
        raise RuntimeError("model has no MoE layers — cannot derive uniform K")
    if len(counts) != 1:
        raise RuntimeError(
            f"model is NOT homogeneous in routed-expert count (saw {counts}); "
            "uniform-K derivation (H-A step 2) requires a single per-layer count"
        )
    return int(counts[0])


def derive_solver_budget(model, base_config: dict) -> dict[str, Any]:
    """Run the Stage-4-aware solver ONCE → uniform ``K`` + Stage-3 ``sp``.

    Net-35% with ``net_of_eora`` ON: ``eora_overhead_pct`` is read from
    ``stage4_eora.compensation_budget_pct`` (M-C / WS1), ``ep_sp_knob_ratio``
    from ``target.expert_svd_ratio``. Returns a dict with ``n_experts``, ``ep``,
    ``sp``, ``K``, ``prune_fraction`` (= 1 - K/n_experts) and the gross/net
    projections for the forensic summary.
    """
    n_experts = _homogeneous_expert_count(model)
    eora_overhead_pct = float(
        base_config.get("stage4_eora", {}).get("compensation_budget_pct", 0.0)
    )
    ep_sp_knob_ratio = float(base_config["target"]["expert_svd_ratio"])
    min_experts = int(base_config["stage1_grape"]["min_experts_per_layer"])

    decomp = budget_solver.solve(
        model,
        target_total_reduction=NET_TARGET,
        ep_sp_knob_ratio=ep_sp_knob_ratio,
        min_experts_per_layer=min_experts,
        blacklisted_experts={},
        eora_overhead_pct=eora_overhead_pct,
    )

    ep = float(decomp.expert_prune_ratio)
    sp = float(decomp.svd_rank_ratio)
    # H-A step 2: uniform per-layer keep count.
    K = int(round((1.0 - ep) * n_experts))
    if not (0 < K < n_experts):
        raise RuntimeError(
            f"derived K={K} is out of range (0, {n_experts}) for ep={ep:.4f}; "
            "solver/budget is degenerate — refusing to proceed"
        )
    # H-A step 3: exact inverse so faithful keep == K
    # (n_keep = round_half_up((1-pf)*n) — reap_prune.py:339).
    prune_fraction = 1.0 - K / n_experts

    budget = {
        "n_experts": n_experts,
        "ep": ep,
        "sp": sp,
        "K": K,
        "prune_fraction": prune_fraction,
        "eora_overhead_pct": eora_overhead_pct,
        "ep_sp_knob_ratio": ep_sp_knob_ratio,
        "net_target": NET_TARGET,
        "projected_gross_reduction": float(decomp.projected_total_reduction),
        "projected_net_reduction": float(decomp.projected_net_reduction),
        "global_expert_budget": int(decomp.global_expert_budget),
    }
    log.info(
        "solver-derived budget: n_experts=%d ep=%.4f sp=%.4f K=%d "
        "prune_fraction=%.6f (gross=%.4f net=%.4f, eora_overhead=%.3f)",
        n_experts, ep, sp, K, prune_fraction,
        decomp.projected_total_reduction, decomp.projected_net_reduction,
        eora_overhead_pct,
    )
    if K <= 166:
        log.warning(
            "derived K=%d <= 166 (the paper-baseline count). The plan expects "
            "K > 166 (lighter Stage-2; SVD+EoRA absorb part of 35%%). Check "
            "expert_svd_ratio / compensation_budget_pct — proceeding anyway.",
            K,
        )
    return budget


# ---------------------------------------------------------------------------
# Config builder — paper-pure Stage-2 + WS2 dials + M-C net accounting (pure)
# ---------------------------------------------------------------------------

def build_arm_config(
    base: dict, *, method: str, prune_fraction: float,
    num_sequences: int | None = None,
) -> dict:
    """Build one arm's config from the faithful base config.

    Pure — no I/O, no model load. Applies, for the given Stage-2 ``method``:

      * Stage-2 by-the-paper-only: REAP=``faithful_prune`` with the exact
        ``prune_fraction`` inverse of K; REAM=``merge`` + ``_REAM_BY_THE_BOOK``.
        ``_PAPER_CORE_OFF`` (em/distill/merge-heal/two-opt off) on BOTH.
      * H-A: ``assert_survivors_match_target = true`` (the uniform-K budget pin
        is seeded onto disk separately by ``seed_stage1_artifacts``).
      * WS2: ``stage5_router_kd.rkd_recipe = "paper_dials_only"`` (both 2.5+5).
      * M-C: ``target.total_reduction_ratio = 0.35`` + ``target.net_of_eora =
        true``.
      * Full pipeline: ``pipeline.skip_intermediate_stages = false`` (run 3/4/5),
        ``stage6_validate.mode = thermometer`` (stage6alt student_bpt).
    """
    if method not in ("faithful_prune", "merge"):
        raise ValueError(f"unknown Stage-2 method {method!r}")

    cfg = copy.deepcopy(base)

    # ---- M-C: net accounting (target block) ----
    cfg.setdefault("target", {})
    cfg["target"]["total_reduction_ratio"] = NET_TARGET
    cfg["target"]["net_of_eora"] = True

    # ---- Stage-2 by-the-paper-only ----
    cfg.setdefault("stage2_reap_ream", {})
    s2 = cfg["stage2_reap_ream"]
    s2.update(_PAPER_CORE_OFF)
    s2["assert_survivors_match_target"] = True

    if method == "faithful_prune":
        s2["prune_mode"] = "faithful_prune"
        # H-A step 3: exact inverse so faithful keep == K.
        s2["prune_fraction"] = float(prune_fraction)
    else:  # merge — by the book
        s2.update(_REAM_BY_THE_BOOK)
        s2.setdefault("ream", {})["frequency_weighted_merge"] = True
        # sequential_reprofile=true + profile_sidecar.enabled=true is HARD
        # rejected (orchestrator.py); force the sidecar OFF (mirrors run_probe).
        _set_nested(s2, "profile_sidecar.enabled", False)

    # ---- WS2: paper router-kd dials at BOTH Stage 2.5 and Stage 5 ----
    cfg.setdefault("stage5_router_kd", {})["rkd_recipe"] = "paper_dials_only"

    # ---- Full pipeline: run Stages 3/4/5; thermometer eval at Stage 6 ----
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["skip_intermediate_stages"] = False
    cfg["pipeline"]["evaluator"] = "stage6alt"
    cfg.setdefault("stage6_validate", {})["mode"] = "thermometer"

    if num_sequences is not None:
        cfg.setdefault("calibration", {})["num_sequences"] = int(num_sequences)
        s2["num_calibration_samples"] = int(num_sequences)

    return cfg


def assert_paper_recipe_safety(cfg: dict) -> None:
    """WS2 safety net: ``paper_dials_only`` sets epochs=2; ``save_best`` MUST be
    true (it exports the EMA-best ~step250 and discards the late-step overfit —
    plan WS2 H1). Fail loudly if the recipe is active but save_best is false.
    """
    s5 = cfg.get("stage5_router_kd", {})
    if s5.get("rkd_recipe") in ("paper", "paper_dials_only"):
        if not bool(s5.get("save_best", False)):
            raise RuntimeError(
                "stage5_router_kd.rkd_recipe is a paper recipe (epochs=2) but "
                "save_best is not true — the late-step overfit would be "
                "exported. Set stage5_router_kd.save_best: true (plan WS2 H1)."
            )


# ---------------------------------------------------------------------------
# H-B — Stage-4 covariance fail-fast pre-check
# ---------------------------------------------------------------------------

def assert_covariance_resolves(base_config: dict) -> Path:
    """Fail fast (H-B) unless the bf16 input-covariance sidecar resolves.

    Resolves the calibration JSONL exactly as the Stage-4 orchestrator does
    (``calibration.jsonl_path`` → ``_DEFAULT_SELF_TRACES_PATH``, made absolute
    against cwd), then calls ``load_covariance`` (standalone, returns None on
    miss). On the sidecar-HIT path ``a_storage_dtype`` auto-resolves to
    bfloat16 inside ``input_cov_cache.on_load``. Aborts with a clear error if
    the sidecar (``sidecars/<stem>/covariance.pt``) cannot be found.

    Returns the resolved JSONL path (for logging).
    """
    from .utils.calibration import _DEFAULT_SELF_TRACES_PATH
    from .utils.cached_calibration_signals import load_covariance, sidecar_path

    cal_cfg = base_config.get("calibration", {}) or {}
    jsonl_source = cal_cfg.get("jsonl_path", _DEFAULT_SELF_TRACES_PATH)
    jsonl_path = Path(jsonl_source)
    if not jsonl_path.is_absolute():
        jsonl_path = Path.cwd() / jsonl_path

    payload = None
    try:
        payload = load_covariance(jsonl_path)
    except Exception as exc:  # noqa: BLE001 — surface as an actionable abort
        raise RuntimeError(
            f"H-B: load_covariance({jsonl_path}) raised {exc!r}. The Stage-4 "
            "input covariance is unresolvable; fix the sidecar before launch."
        ) from exc
    if payload is None:
        expected = sidecar_path(jsonl_path, "covariance")
        raise RuntimeError(
            f"H-B: Stage-4 input covariance NOT found for calibration JSONL "
            f"{jsonl_path}.\nExpected the bf16 capture sidecar at {expected} "
            "(or its legacy non-namespaced sibling). Set calibration.jsonl_path "
            "to the bf16 self-traces JSONL whose sidecars/<stem>/covariance.pt "
            "exists, or re-run the input-covariance capture. Aborting before "
            "any GPU work (the full-pipeline runner does NOT skip the fp16 "
            "fallback, but Stage 4 still needs a covariance source)."
        )
    log.info(
        "H-B: input covariance resolves for %s (sidecar %s) — a_storage_dtype "
        "will auto-resolve to bfloat16 on the Stage-4 hit path.",
        jsonl_path, sidecar_path(jsonl_path, "covariance"),
    )
    return jsonl_path


# ---------------------------------------------------------------------------
# Per-arm subprocess runner (mirrors run_probe staging; see D1)
# ---------------------------------------------------------------------------

def _write_config(cfg: dict, arm_dir: Path) -> Path:
    arm_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = arm_dir / "arm_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def _pipeline_argv(cfg_path: Path, model_repo: str, arm_dir: Path,
                   resume: int, stop: int) -> list[str]:
    """run_pipeline invocation. ``--target-ratio 0.35`` mirrors run_probe; the
    config's ``target.net_of_eora=true`` (injected by build_arm_config) makes
    the solver net-aware (--target-ratio sets only the ratio, not net_of_eora —
    run_pipeline.py:89-90,181)."""
    return [
        sys.executable, "-m", "moe_compress.run_pipeline",
        "--config", str(cfg_path),
        "--model", model_repo,
        "--artifacts-dir", str(arm_dir),
        "--target-ratio", str(NET_TARGET),
        "--resume-from-stage", str(resume),
        "--stop-after-stage", str(stop),
    ]


def is_complete(arm_dir: Path) -> bool:
    """Idempotent gate: the Stage-6alt artifact marks a finished arm."""
    return (arm_dir / STAGE6ALT_ARTIFACT).exists()


def run_one_arm(
    *, arm_id: str, method: str, base_config: dict, budget: dict[str, Any],
    shared_dir: Path, probe_root: Path, model_repo: str, num_sequences: int,
) -> dict[str, Any]:
    """Drive one arm Stage 2 → 2.5 → 3 → 4 → 5 → 6 (two subprocesses; see D1).

    Idempotent: a present stage6alt_eval.json short-circuits. Seeds the
    uniform-K Stage-1 artifacts (H-A step 4 — pin for BOTH arms) before Stage 2.
    """
    arm_dir = probe_root / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)

    if is_complete(arm_dir):
        log.info("[%s] already complete — loading prior result", arm_id)
        result = json.loads((arm_dir / STAGE6ALT_ARTIFACT).read_text())
        result["_arm_id"] = arm_id
        return result

    cfg = build_arm_config(
        base_config, method=method,
        prune_fraction=budget["prune_fraction"], num_sequences=num_sequences,
    )
    assert_paper_recipe_safety(cfg)
    cfg_path = _write_config(cfg, arm_dir)

    # H-A step 4: uniform-K stage1_budgets.json pin (+ blacklist +
    # budget_decomposition) for BOTH arms — the survivor guard reads it for
    # REAP too. survivors=K passed EXPLICITLY (default is run_probe's 166).
    seed_stage1_artifacts(arm_dir, shared_dir, group="ream", survivors=budget["K"])

    # ---- (a) Stage 2 + auto Stage 2.5 → stage2p5_final/ ----
    log.info("[%s] Stage 2 + 2.5 (method=%s, K=%d) → stage2p5_final/",
             arm_id, method, budget["K"])
    rc1 = subprocess.run(
        _pipeline_argv(cfg_path, model_repo, arm_dir, resume=2, stop=2),
        check=False,
    ).returncode
    if rc1 != 0:
        raise RuntimeError(f"[{arm_id}] Stage 2/2.5 returned exit code {rc1}")
    if not (arm_dir / "stage2p5_final").exists():
        raise RuntimeError(
            f"[{arm_id}] Stage 2.5 exited 0 but stage2p5_final/ is absent — "
            "refusing to resume Stage 3 on a missing post-2.5 checkpoint."
        )

    # ---- (b) Stage 3 → 4 → 5 → 6 (loads stage2p5_final/ per STAGE_REGISTRY) ----
    log.info("[%s] Stage 3 → 4 → 5 → 6alt", arm_id)
    rc2 = subprocess.run(
        _pipeline_argv(cfg_path, model_repo, arm_dir, resume=3, stop=6),
        check=False,
    ).returncode
    if rc2 != 0:
        raise RuntimeError(f"[{arm_id}] Stage 3-6 returned exit code {rc2}")

    if not is_complete(arm_dir):
        raise RuntimeError(
            f"[{arm_id}] pipeline succeeded but {STAGE6ALT_ARTIFACT} missing"
        )

    result = json.loads((arm_dir / STAGE6ALT_ARTIFACT).read_text())
    result["_arm_id"] = arm_id
    result["_method"] = method
    return result


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full-pipeline REAP-vs-REAM @ net-35% (2 'ours' arms)")
    parser.add_argument("--config", required=True,
                        help="Base YAML (qwen36_35b_a3b_reap_faithful.yaml)")
    parser.add_argument("--model", required=True, help="HF model repo")
    parser.add_argument("--probe-root", required=True,
                        help="Root dir for per-arm artifacts + _shared/ Stage-1")
    parser.add_argument("--num-sequences", type=int, default=4000)
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of arm ids "
                             "(e.g. reap-s234,ream-s234)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    log.info("===== Full-pipeline REAP-vs-REAM @ net-35% (ours arms) =====")
    log.info("%s", _PAPER_BASELINE_NOTE)

    base_config = yaml.safe_load(Path(args.config).read_text())
    if args.model:
        base_config.setdefault("model", {})["name_or_path"] = args.model
    probe_root = Path(args.probe_root)
    probe_root.mkdir(parents=True, exist_ok=True)
    shared_dir = probe_root / "_shared"

    arms = list(ARMS)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        arms = [a for a in arms if a[0] in wanted]
    log.info("Will run %d arm(s): %s", len(arms), [a[0] for a in arms])

    # Shared Stage-1 artifacts gate (same contract as run_probe).
    if not all((shared_dir / n).exists() for n in (
        "stage1_blacklist.json", "stage1_budgets.json", "budget_decomposition.json",
    )):
        raise RuntimeError(
            f"Shared Stage-1 artifacts missing in {shared_dir}. Run the shared "
            "Stage-1 step (run_pipeline --stop-after-stage 1, net_of_eora ON) "
            "into _shared/ first (plan A1)."
        )

    # A1: warn (don't block) if the shared Stage-1 budget was computed gross-only
    # — the seeded Stage-3 sp then lands ~0.3pp light. The iso-K pin still makes
    # both arms identical, so the comparison holds; we just flag the provenance.
    try:
        _shared_budget = json.loads((shared_dir / "budget_decomposition.json").read_text())
        if not float(_shared_budget.get("eora_overhead_pct", 0.0)) > 0.0:
            log.warning(
                "shared budget_decomposition.json was produced GROSS-only "
                "(eora_overhead_pct=0): seeded Stage-3 sp is ~0.3pp light of net-%.2f. "
                "Both arms remain iso-K (pin overrides experts), so the comparison "
                "is unaffected — but re-run shared Stage 1 with net_of_eora ON for a "
                "fully net-aware sp.", NET_TARGET,
            )
    except (OSError, ValueError, TypeError) as e:
        log.warning("could not check shared budget net provenance: %s", e)

    # H-B: fail fast unless the bf16 input-covariance sidecar resolves — BEFORE
    # any model load / GPU work.
    assert_covariance_resolves(base_config)

    # WS1 + H-A: derive uniform K + Stage-3 sp ONCE from the loaded base model.
    log.info("Loading base model to derive the solver budget (one-shot) ...")
    from .utils.model_io import load_model
    model, _tok = load_model(
        base_config["model"]["name_or_path"],
        revision=base_config["model"].get("revision", "main"),
        torch_dtype=base_config["model"].get("torch_dtype", "bfloat16"),
        device_map=base_config["model"].get("device_map", "auto"),
        attn_implementation=base_config["model"].get("attn_implementation", "sdpa"),
        load_in_4bit=base_config["model"].get("load_in_4bit", False),
        trust_remote_code=base_config["model"].get("trust_remote_code", False),
    )
    budget = derive_solver_budget(model, base_config)
    save_json_artifact(budget, probe_root / "solver_budget.json")
    # Free the budget-derivation model before the per-arm subprocess pipeline.
    del model, _tok
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort cache release
        pass

    results: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    for arm_id, method in arms:
        try:
            results[arm_id] = run_one_arm(
                arm_id=arm_id, method=method, base_config=base_config,
                budget=budget, shared_dir=shared_dir, probe_root=probe_root,
                model_repo=args.model, num_sequences=args.num_sequences,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] failed", arm_id)
            failures.append((arm_id, str(exc)))

    summary = {
        "solver_budget": budget,
        "results": {k: {**v, "student_bpt": _student_bpt(v)} for k, v in results.items()},
        "failures": [{"arm_id": a, "error": e} for a, e in failures],
        "paper_baseline_note": _PAPER_BASELINE_NOTE,
    }
    save_json_artifact(summary, probe_root / "_reap_ream_35pct_summary.json")

    # Comparison line: reap-s234 vs ream-s234 (the two ours arms).
    reap = _student_bpt(results["reap-s234"]) if "reap-s234" in results else float("inf")
    ream = _student_bpt(results["ream-s234"]) if "ream-s234" in results else float("inf")
    log.info(
        "COMPARISON (ours @ net-35%%, iso-K=%d, sp=%.4f): "
        "reap-s234 student_bpt=%.4f  vs  ream-s234 student_bpt=%.4f  (lower=better)",
        budget["K"], budget["sp"], reap, ream,
    )
    log.info("vs paper baselines: %s", _PAPER_BASELINE_NOTE)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
