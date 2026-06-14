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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from huggingface_hub import snapshot_download

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

# Net compression target (M-C) — these arms target NET-35% after Stage-4 EoRA.
NET_TARGET = 0.35


def _default_num_gpus() -> int:
    """Detected CUDA device count (floor 1) for the ``--num-gpus`` default.

    Best-effort — a torch import / CUDA probe failure falls back to 1 (the
    1-GPU path injects no multi-GPU overlay)."""
    try:
        import torch
        return max(1, int(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001 — no torch / no CUDA ⇒ single-GPU default
        return 1


@dataclass(frozen=True)
class ArmSpec:
    """One "ours" arm: Stage-2 ``method`` + its run_pipeline stage windows + an
    optional HF seed repo for a resume-from-post-2.5 checkpoint.

    ``stage_windows`` is a tuple of ``(resume, stop)`` pairs run in order. Every
    resume stage is >= 2 ⇒ Stage-1 GRAPE/RCO never runs (run_pipeline gates
    start<=1). A ``seed_hub_repo`` arm has its post-2.5 ``stage2p5_final/`` placed
    on disk from the Hub before the loop, so it carries a single ``(3, 6)``
    window (skips Stage-2/2.5 entirely); a non-seeded arm runs the
    ``(2,2),(3,6)`` pair (its own Stage 2 + auto-2.5, then Stage 3→6).
    """

    arm_id: str
    method: str
    seed_hub_repo: str | None
    stage_windows: tuple[tuple[int, int], ...]


# The two "ours" arms. ``method`` selects the Stage-2 mechanism; both share the
# SAME solver K + sp (iso-compression) and the SAME paper router-kd dials.
#   * reap-s234 resumes from its HF-backed stage2p5_final → Stage 3→6 ONLY.
#   * ream-s234 runs its own Stage 2→2.5 then Stage 3→6 (unchanged).
ARM_SPECS: tuple[ArmSpec, ...] = (
    ArmSpec(
        arm_id="reap-s234",
        method="faithful_prune",
        seed_hub_repo="pirola/reap-s234-stage2p5-final",
        stage_windows=((3, 6),),
    ),
    ArmSpec(
        arm_id="ream-s234",
        method="merge",
        seed_hub_repo=None,
        stage_windows=((2, 2), (3, 6)),
    ),
)

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
# HF seed — place a post-2.5 stage2p5_final/ checkpoint on disk for a seeded arm
# ---------------------------------------------------------------------------

# The 3 required METADATA files a resume@3 needs in stage2p5_final/
# (shards handled separately below; run_pipeline.py:516-535).
_STAGE2P5_REQUIRED = (
    "config.json",
    "model.safetensors.index.json",
    "compressed_metadata.json",
)


def _seed_stage2p5_from_hub(
    repo: str,
    arm_dir: Path,
    *,
    _downloader: Callable[..., Any] = snapshot_download,
) -> Path:
    """Download a post-2.5 checkpoint from the Hub and materialize
    ``arm_dir/stage2p5_final/`` so a ``--resume-from-stage 3`` run loads it from
    disk (run_pipeline.py:516-535 — no in-process Hub download).

    Handles BOTH repo layouts: the checkpoint files at the repo ROOT, or under a
    ``stage2p5_final/`` subdir — detected by probing for ``compressed_metadata.json``.

    Idempotent: if ``arm_dir/stage2p5_final/compressed_metadata.json`` is already
    present, skips the download entirely (a re-run / resumed arm does not
    re-fetch 52 GB).

    CONTENT-VERIFY (per [[feedback_verify_content_not_filesize]]): after placing,
    asserts ``compressed_metadata.json`` parses AND every shard listed in
    ``model.safetensors.index.json`` exists on disk; RAISES loudly on any gap —
    a half-download must FAIL, never silently fall back to ``stage2_pruned``.

    The ``_downloader`` seam keeps tests off the network.

    Returns the placed ``stage2p5_final/`` dir.
    """
    final_dir = arm_dir / "stage2p5_final"
    meta_path = final_dir / "compressed_metadata.json"
    if meta_path.exists():
        log.info("[seed] %s already present — skipping Hub download (idempotent)",
                 meta_path)
        # Self-heal: if the existing dir is a half-checkpoint (e.g. a prior run
        # filled the disk mid-shard-copy), wipe it so this call re-downloads
        # rather than re-raising forever on every retry.
        try:
            _verify_stage2p5_content(final_dir)
        except Exception:
            log.warning("[seed] existing %s failed content-verify — wiping the "
                        "partial checkpoint and re-downloading", final_dir)
            shutil.rmtree(final_dir, ignore_errors=True)
        else:
            return final_dir

    final_dir.mkdir(parents=True, exist_ok=True)
    snap_root = arm_dir / "_hub_snapshot_stage2p5"
    snap_root.mkdir(parents=True, exist_ok=True)
    log.info("[seed] downloading %s → %s", repo, snap_root)
    _downloader(repo_id=repo, local_dir=str(snap_root))

    # Detect the layout: files at root, or nested under stage2p5_final/.
    if (snap_root / "compressed_metadata.json").exists():
        src_root = snap_root
    elif (snap_root / "stage2p5_final" / "compressed_metadata.json").exists():
        src_root = snap_root / "stage2p5_final"
    else:
        raise RuntimeError(
            f"[seed] {repo}: compressed_metadata.json not found at the snapshot "
            f"root ({snap_root}) nor under a stage2p5_final/ subdir — cannot "
            "locate the post-2.5 checkpoint. Refusing to resume Stage 3."
        )

    # Copy the checkpoint files (config + index + metadata + every shard) into
    # arm_dir/stage2p5_final/. The pipeline's save_compressed_checkpoint ALWAYS
    # emits the multi-shard + index layout, so shards are matched by the
    # model-*.safetensors glob.
    for name in _STAGE2P5_REQUIRED:
        src = src_root / name
        if not src.exists():
            raise RuntimeError(
                f"[seed] {repo}: required file {name} missing from the snapshot "
                f"({src}). The Hub checkpoint is incomplete; refusing to resume."
            )
        shutil.copy2(src, final_dir / name)
    for shard in src_root.glob("model-*.safetensors"):
        shutil.copy2(shard, final_dir / shard.name)

    # Optional _shared/ metadata ride-along: copy only what is MISSING locally
    # (do not clobber a locally-seeded _shared/).
    src_shared = src_root / "_shared"
    if src_shared.is_dir():
        dst_shared = arm_dir.parent / "_shared"
        dst_shared.mkdir(parents=True, exist_ok=True)
        for f in src_shared.iterdir():
            if f.is_file() and not (dst_shared / f.name).exists():
                shutil.copy2(f, dst_shared / f.name)

    # Self-heal: a FAILED post-copy verify (e.g. disk filled mid-shard-copy)
    # wipes the partial dir before raising, so the next call re-downloads rather
    # than short-circuiting forever on the present-but-incomplete metadata.
    try:
        _verify_stage2p5_content(final_dir)
    except Exception:
        shutil.rmtree(final_dir, ignore_errors=True)
        raise
    log.info("[seed] placed + content-verified %s", final_dir)
    return final_dir


def _verify_stage2p5_content(final_dir: Path) -> None:
    """CONTENT-VERIFY the placed stage2p5_final/: metadata parses + every shard
    the index lists exists on disk. RAISE loudly on any gap (a half-download
    must FAIL, never silently fall back to stage2_pruned)."""
    meta_path = final_dir / "compressed_metadata.json"
    try:
        json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"[seed] compressed_metadata.json at {meta_path} does not parse "
            f"({exc!r}) — the seeded post-2.5 checkpoint is corrupt."
        ) from exc

    index_path = final_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise RuntimeError(
            f"[seed] model.safetensors.index.json missing from {final_dir} — "
            "cannot verify shard completeness; refusing to resume Stage 3."
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"[seed] index {index_path} does not parse ({exc!r})."
        ) from exc
    weight_map = index.get("weight_map", {}) or {}
    shards = sorted(set(weight_map.values()))
    missing = [s for s in shards if not (final_dir / s).exists()]
    if missing:
        raise RuntimeError(
            f"[seed] {len(missing)} shard(s) listed in the index are absent from "
            f"{final_dir}: {missing[:5]}{'...' if len(missing) > 5 else ''}. The "
            "post-2.5 download is incomplete — refusing to resume Stage 3 on a "
            "half-checkpoint (would silently fall back to stage2_pruned). "
            f"Delete {final_dir} to retry."
        )


# ---------------------------------------------------------------------------
# Config builder — paper-pure Stage-2 + WS2 dials + M-C net accounting (pure)
# ---------------------------------------------------------------------------

def build_arm_config(
    base: dict, *, method: str, prune_fraction: float,
    num_sequences: int | None = None,
    num_gpus: int = 1, whitening_cov: str = "anchor",
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

    # ---- acov whitening (Task 5) ----
    if whitening_cov not in ("anchor", "shift", "anchored_adaptive"):
        raise ValueError(
            f"unknown whitening_cov {whitening_cov!r} (expected anchor/shift/"
            "anchored_adaptive)")
    if whitening_cov != "anchor":
        # Opt-in only: 'anchor' IS the plugin default (eora_inputs.py:363
        # s4.get("whitening_cov", "anchor")), so the default path emits NEITHER
        # key ⇒ byte-identical to the historical config. shift / anchored_adaptive
        # set whitening_cov AND REQUIRE the persisted post-2.5 shift cov
        # (eora_inputs.py:363-371 raises without persist_shift_covariance).
        cfg.setdefault("stage4_eora", {})["whitening_cov"] = whitening_cov
        cfg.setdefault("stage3_svd", {})["persist_shift_covariance"] = True

    # ---- Multi-GPU overlay (Task 4), gated on num_gpus>=2 ----
    # num_gpus<2 injects NONE of these ⇒ the 1-GPU path is byte-identical to the
    # historical config. All these knobs are default-OFF in their stage plugins.
    if num_gpus >= 2:
        # RESULT-PRESERVING SUBSET (live 2×H200 validation 2026-06-14):
        # the data-parallel AUTO-BATCH cov paths (multi_gpu.cov_replicas and
        # stage2_reap_ream.profile_dp) are EXCLUDED — validate_cov_dp_live.py
        # showed a 2-replica auto-batched reduce is NOT bitwise-equal to the
        # 1-GPU bs=1 cov (bf16 MoE/attention forward is not batch-invariant, so
        # replicas that auto-size a different batch drift the covariance at fp
        # level). validate_cov_sharded_live.py confirmed sharding + the reduce
        # MATH are bitwise-exact, so the drift is purely auto-batch. cov collection
        # is also not the bottleneck (the 35B student fits one H200), so we run it
        # 1-GPU and keep the covariance bitwise-faithful.
        #
        # KEPT (byte-identical / grad-avg-equivalent, high value):
        #   * Stage-3 per-expert SVD factor + α-grid task-parallel (independent
        #     per-expert work, no cross-device reduction) — multi_gpu.{factor,alpha}_workers
        #   * Stage-4 EoRA per-expert concurrency — multi_gpu.eora_workers
        #   * Stage-2.5 + Stage-5 Router-KD DDP (the runtime long pole)
        # All read top-level config.get("multi_gpu", {}) (stage3/orchestrator.py
        # :119,137 ; stage4/orchestrator.py:80).
        mg = cfg.setdefault("multi_gpu", {})
        mg["factor_workers"] = num_gpus
        mg["alpha_workers"] = num_gpus
        mg["eora_workers"] = num_gpus

        # Stage-2.5 + Stage-5 DDP (both read stage5_router_kd regardless of
        # stage_key — router_kd/orchestrator.py:200). NO stage6_validate.eval_shard
        # — this ablation evals via the stage6alt thermometer where eval_shard
        # does not apply; setting it would be misleading.
        cfg["stage5_router_kd"]["ddp"] = {
            "enabled": True, "world_size": num_gpus, "backend": "nccl",
        }
        assert_ddp_batch_divisible(cfg, num_gpus)
        # model.device_map left at base (auto): DP cov replicas / DDP ranks pin
        # their own GPU via CUDA_VISIBLE_DEVICES and DDP overrides device_map
        # per-rank, so the parent auto placement is safe.

    return cfg


def assert_ddp_batch_divisible(cfg: dict, world_size: int) -> None:
    """DDP guard: ``stage5_router_kd.batch_size`` must be divisible by
    ``world_size`` (per_gpu = batch / ws must be an integer — ddp_config.py:52-65).

    Validates the AS-WRITTEN config ONLY. ``rkd_paper_recipe`` runs later inside
    ``router_kd.run()`` (after this runner emits the config) but does NOT touch
    ``batch_size`` (plan M1), so this pre-check is valid. Reads ``batch_size``
    with ``.get`` (NOT ``[...]``): an absent batch_size (paper_dials_only does
    not set one) validates nothing here — the plugin resolves the effective
    batch downstream — rather than KeyError-ing."""
    s5 = cfg.get("stage5_router_kd", {}) or {}
    batch_size = s5.get("batch_size")
    if batch_size is None:
        return
    if int(batch_size) % int(world_size) != 0:
        raise RuntimeError(
            f"Router-KD DDP: stage5_router_kd.batch_size={batch_size} is not "
            f"divisible by world_size={world_size} (per-GPU batch = "
            f"batch_size/world_size must be an integer). Set batch_size to a "
            "multiple of world_size (and co-scale gradient_accumulation to keep "
            "the effective batch fixed) before launching the multi-GPU run."
        )


def assert_paper_recipe_safety(cfg: dict) -> None:
    """WS2 safety net: ``paper_dials_only`` sets epochs=2; ``save_best`` MUST be
    true (it exports the EMA-best ~step250 and discards the late-step overfit —
    plan WS2 H1). Fail loudly only if the recipe is active AND save_best is
    EXPLICITLY false. ``save_best`` already defaults to True in the operative
    path (early_stop.py: ``s5.get("save_best", True)``), so this net matches
    that default and never trips on a mere omission.
    """
    s5 = cfg.get("stage5_router_kd", {})
    # Defaults mirror the operative code (2026-06-09): absent rkd_recipe →
    # "paper_dials_only" (the plugin default), absent save_best → True
    # (early_stop.py default). So the guard covers the default path and only
    # raises on an EXPLICIT save_best: false under a paper recipe.
    if s5.get("rkd_recipe", "paper_dials_only") in ("paper", "paper_dials_only"):
        if not bool(s5.get("save_best", True)):
            raise RuntimeError(
                "stage5_router_kd.rkd_recipe is a paper recipe (epochs=2) but "
                "save_best is explicitly false — the late-step overfit would be "
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
    *, spec: ArmSpec, base_config: dict, budget: dict[str, Any],
    shared_dir: Path, probe_root: Path, model_repo: str, num_sequences: int,
    num_gpus: int = 1, whitening_cov: str = "anchor",
) -> dict[str, Any]:
    """Drive one arm through its ``spec.stage_windows`` (see D1).

    A ``seed_hub_repo`` arm (reap) materializes its post-2.5 ``stage2p5_final/``
    from the Hub BEFORE the loop, then runs ONLY its ``(3, 6)`` window (zero
    ``--resume-from-stage 2`` calls). A non-seeded arm (ream) runs all windows in
    order — its own Stage 2 + auto-2.5, then Stage 3→6 — with the
    ``stage2p5_final/`` existence guard between a stop@2 and the next window.

    ``seed_stage1_artifacts`` runs for BOTH arms (the survivor guard reads
    ``_shared/`` even for reap; harmless and already present).

    Idempotent: a present stage6alt_eval.json short-circuits; a seeded reap
    re-run does not re-download (the seed helper is itself idempotent).
    """
    arm_id, method = spec.arm_id, spec.method
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
        num_gpus=num_gpus, whitening_cov=whitening_cov,
    )
    assert_paper_recipe_safety(cfg)
    cfg_path = _write_config(cfg, arm_dir)

    # H-A step 4: uniform-K stage1_budgets.json pin (+ blacklist +
    # budget_decomposition) for BOTH arms — the survivor guard reads it for
    # REAP too. survivors=K passed EXPLICITLY (default is run_probe's 166).
    seed_stage1_artifacts(arm_dir, shared_dir, group="ream", survivors=budget["K"])

    # A seeded arm places its post-2.5 checkpoint on disk so its (3,6) window
    # resumes from a real stage2p5_final/ — no Stage-2/2.5 subprocess at all.
    if spec.seed_hub_repo is not None:
        log.info("[%s] seeding post-2.5 checkpoint from %s (Stage 2/2.5 SKIPPED)",
                 arm_id, spec.seed_hub_repo)
        _seed_stage2p5_from_hub(spec.seed_hub_repo, arm_dir)

    for resume, stop in spec.stage_windows:
        log.info("[%s] Stage %d → %d (method=%s, K=%d)",
                 arm_id, resume, stop, method, budget["K"])
        rc = subprocess.run(
            _pipeline_argv(cfg_path, model_repo, arm_dir, resume=resume, stop=stop),
            check=False,
        ).returncode
        if rc != 0:
            raise RuntimeError(
                f"[{arm_id}] Stage {resume}-{stop} returned exit code {rc}")
        # Post-2.5 (stop@2) checkpoint guard: the next window resumes@3 and must
        # find stage2p5_final/ on disk.
        if stop == 2 and not (arm_dir / "stage2p5_final").exists():
            raise RuntimeError(
                f"[{arm_id}] Stage 2.5 exited 0 but stage2p5_final/ is absent — "
                "refusing to resume Stage 3 on a missing post-2.5 checkpoint."
            )

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
    parser.add_argument(
        "--num-gpus", type=int, default=_default_num_gpus(),
        help="GPUs to fan the multi-GPU opt-in features across (Stage-3 cov "
             "DP/SVD, Stage-4 EoRA threads, Router-KD DDP). Default: detected "
             "torch.cuda.device_count() (floor 1). num_gpus<2 ⇒ 1-GPU path.")
    parser.add_argument(
        "--whitening-cov", default="anchor",
        choices=("anchor", "shift", "anchored_adaptive"),
        help="Stage-4 EoRA whitening covariance (acov A/B). Default 'anchor' "
             "(byte-identical historical path); 'shift'/'anchored_adaptive' "
             "also persist the Stage-3 shift covariance.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    log.info("===== Full-pipeline REAP-vs-REAM @ net-35% (ours arms) =====")
    log.info("%s", _PAPER_BASELINE_NOTE)
    log.info("num_gpus=%d (multi-GPU overlay %s), whitening_cov=%s",
             args.num_gpus, "ON" if args.num_gpus >= 2 else "OFF (1-GPU)",
             args.whitening_cov)

    base_config = yaml.safe_load(Path(args.config).read_text())
    if args.model:
        base_config.setdefault("model", {})["name_or_path"] = args.model
    probe_root = Path(args.probe_root)
    probe_root.mkdir(parents=True, exist_ok=True)
    shared_dir = probe_root / "_shared"

    arms = list(ARM_SPECS)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        arms = [a for a in arms if a.arm_id in wanted]
    log.info("Will run %d arm(s): %s", len(arms), [a.arm_id for a in arms])

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
    for spec in arms:
        try:
            results[spec.arm_id] = run_one_arm(
                spec=spec, base_config=base_config,
                budget=budget, shared_dir=shared_dir, probe_root=probe_root,
                model_repo=args.model, num_sequences=args.num_sequences,
                num_gpus=args.num_gpus, whitening_cov=args.whitening_cov,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] failed", spec.arm_id)
            failures.append((spec.arm_id, str(exc)))

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
