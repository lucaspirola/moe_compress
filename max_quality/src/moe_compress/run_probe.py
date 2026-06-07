"""6-model REAP/REAM healing-probe harness — Stage 2(+2.5) → Stage 6alt.

Six paper-core models on ``qwen3-pretrain-mix-v2``, 35% fewer experts, single
survivor count K=166/layer for BOTH groups (drop = round(0.35·256)=90).
ALL refinements OFF (em/expert-distill/merge-heal/two-opt off; capacity gate
inert via ``cost_alignment: "pre"``; skip-merge floor OFF for REAM rows). Two
groups × three arms:

    group  arm     mechanism
    reap   base    faithful_prune, prune_fraction=0.35, NO heal
    reap   heal25  + auto Stage-2.5 router-KD (current dials, our calib)
    reap   rkd     + auto Stage-2.5 router-KD (rkd_recipe: paper_dials_only)
    ream   base    merge, uniform-166 budget pin, NO heal
    ream   heal25  + auto Stage-2.5 router-KD (current dials)
    ream   rkd     + auto Stage-2.5 router-KD (paper_dials_only)

Arm windowing onto the EXISTING pipeline (no new stage-map mechanism):

  * **base** — ``pipeline.skip_intermediate_stages: true``; ONE invocation,
    default ``--stop-after-stage 6``. Stage 2.5/3/4/5 are skipped and control
    flows straight to Stage 6alt (run_pipeline.py:253-257,270,310). The eval
    mode is forced to thermometer by the skip-intermediate override
    (run_pipeline.py:102-113).
  * **heal25 / rkd** — ``skip_intermediate_stages: false``; TWO invocations,
    exactly run_ablations.py's shape: ``--resume-from-stage 2 --stop-after-stage
    2`` (Stage 2 + auto Stage 2.5, run_pipeline.py:263-270), then
    ``--resume-from-stage 6 --stop-after-stage 6`` (Stage 6alt off
    stage2p5_final/). The skip-intermediate eval-override does NOT fire on this
    path, so the heal configs set ``stage6_validate.mode: thermometer``
    DIRECTLY.

REAM 166: a uniform ``stage1_budgets.json`` pin
(``per_layer_target_experts[layer]=166`` for every MoE layer) seeded into the
row dir before Stage 2; the merge path reads it as the per-layer survivor
``target`` (orchestrator.py:785-789,1647). The faithful (REAP) path ignores
the budget for selection (count is fixed by ``prune_fraction``), but the same
166 pin makes ``target==166`` there too so the opt-in survivor==target guard
(``assert_survivors_match_target``) covers both groups.

Each row uploads its model + ``stage6alt_eval.json`` to
``pirola/calib-v2-probe-{group}-{arm}``. Winner = lowest ``student_bpt`` →
``pirola/calib-v2-probe-winner`` + optional local snapshot_download.

The probe RUN is BLOCKED on calibration sidecar re-capture (B0); this module
is the launch-ready machinery. NOTHING here runs on GPU at import time. All
config-builder + budget-pin helpers are pure and unit-tested (no model load).

Launch (after sidecar B0):

    MOE_SKIP_STAGE2_COV_SAVE=1 \
    python -m moe_compress.run_probe \
        --config configs/qwen36_35b_a3b_reap_faithful.yaml \
        --model Qwen/Qwen3.6-35B-A3B \
        --probe-root ./artifacts/probe \
        --num-sequences 4000
"""
from __future__ import annotations

import os

# Match run_ablations.py's pre-torch env hardening (allocator + thread caps +
# inductor cache/worker fixes). These are set BEFORE any torch import and are
# correctness-neutral; see run_ablations.py for the per-line rationale.
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
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .utils.model_io import load_json_artifact, save_json_artifact

log = logging.getLogger(__name__)

# Survivor count K = round((1 - 0.35) · 256) = 166. Both groups, every layer.
PROBE_SURVIVORS = 166
PROBE_PRUNE_FRACTION = 0.35

# Custom HF repo ids — NOT upload_stage_to_hub's f"{base}-stage{idx}" scheme.
HF_REPO_PREFIX = "pirola/calib-v2-probe"
HF_WINNER_REPO = "pirola/calib-v2-probe-winner"

# Final Stage-6alt artifact filename — the completion gate + upload target.
STAGE6ALT_ARTIFACT = "stage6alt_eval.json"


# ---------------------------------------------------------------------------
# Row matrix — {reap,ream} × {base,heal25,rkd}
# ---------------------------------------------------------------------------

GROUPS = ("reap", "ream")
ARMS = ("base", "heal25", "rkd")


def probe_rows() -> list[tuple[str, str, str]]:
    """The 6 rows as ``(row_id, group, arm)``; ``row_id`` = ``{group}-{arm}``."""
    return [(f"{g}-{a}", g, a) for g in GROUPS for a in ARMS]


# ---------------------------------------------------------------------------
# Config builder — paper-core deltas on the faithful base (pure; unit-tested)
# ---------------------------------------------------------------------------

# REAM by-the-book params (§4). ``max_merge_group_size`` counts NON-centroids,
# so upstream group_size=16 (total incl. centroid) ⇔ max_merge_group_size=15.
_REAM_BY_THE_BOOK: dict[str, Any] = {
    "prune_mode": "merge",
    "max_merge_group_size": 15,
    "sequential_reprofile": True,
    "cost_alignment": "pre",
    # skip-merge floor OFF (100.0) so merges actually proceed (the point of REAM).
    "skip_merge_percentile": 100.0,
    # capacity gate is inert under cost_alignment="pre" (capacity_gate.py:25,100,110).
    # cost_asymmetric stays at the faithful base value (false): the by-the-book
    # δ_REAM is the symmetric "pre" cost.
}

# Paper-core OFF set (§5) — applied to BOTH groups. All refinements disabled.
_PAPER_CORE_OFF: dict[str, Any] = {
    "em_refinement_rounds": 0,
    "expert_distill_steps": 0,
    "merge_heal_enabled": False,
    "two_opt_refine": False,
}


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    """Set ``d[a][b]... = value`` for a dotted key, creating dicts as needed."""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):  # pragma: no cover - defensive
            raise TypeError(f"{dotted}: {k} is not a dict")
    cur[keys[-1]] = value


def build_probe_config(
    base: dict, *, group: str, arm: str, num_sequences: int | None = None,
) -> dict:
    """Build one probe row's config dict from the faithful base.

    Deep-copies ``base`` (the faithful YAML, prune_mode=faithful_prune) and
    applies the per-group + per-arm deltas. Pure — no I/O, no model load.
    """
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; expected one of {GROUPS}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    cfg = copy.deepcopy(base)
    cfg.setdefault("stage2_reap_ream", {})
    s2 = cfg["stage2_reap_ream"]

    # Paper-core: ALL refinements OFF (both groups).
    s2.update(_PAPER_CORE_OFF)
    # Opt-in realised-survivor guard (orchestrator asserts kept==target/layer).
    s2["assert_survivors_match_target"] = True

    if group == "reap":
        # Faithful structural drop; keep = round((1-0.35)*256) = 166.
        s2["prune_mode"] = "faithful_prune"
        s2["prune_fraction"] = PROBE_PRUNE_FRACTION
    else:  # ream
        # By-the-book merge to 166 via the uniform stage1_budgets.json pin.
        s2.update(_REAM_BY_THE_BOOK)
        s2.setdefault("ream", {})["frequency_weighted_merge"] = True
        # sequential_reprofile=true + profile_sidecar.enabled=true is HARD
        # rejected (orchestrator.py:759-767); force the sidecar OFF.
        _set_nested(s2, "profile_sidecar.enabled", False)

    # Arm windowing.
    cfg.setdefault("pipeline", {})
    cfg.setdefault("stage6_validate", {})
    if arm == "base":
        # No heal: skip 2.5/3/4/5; the skip-intermediate override forces the
        # thermometer evaluator (run_pipeline.py:102-113). Set it explicitly
        # too so the generated config self-documents + the grep test passes.
        cfg["pipeline"]["skip_intermediate_stages"] = True
        cfg["pipeline"]["evaluator"] = "stage6alt"
        cfg["stage6_validate"]["mode"] = "thermometer"
    else:
        # heal arms: Stage 2 + auto Stage 2.5, then resume Stage 6alt. The
        # skip-intermediate eval-override does NOT fire here, so set the mode
        # DIRECTLY (plan §1b).
        cfg["pipeline"]["skip_intermediate_stages"] = False
        cfg["stage6_validate"]["mode"] = "thermometer"
        if arm == "rkd":
            # Paper dials, OUR calib.
            cfg.setdefault("stage5_router_kd", {})["rkd_recipe"] = "paper_dials_only"
        # heal25: rkd_recipe ABSENT → default "current" production dials.
        else:
            cfg.setdefault("stage5_router_kd", {}).pop("rkd_recipe", None)

    if num_sequences is not None:
        cfg.setdefault("calibration", {})["num_sequences"] = int(num_sequences)
        s2["num_calibration_samples"] = int(num_sequences)

    return cfg


def is_heal_arm(arm: str) -> bool:
    """heal25/rkd run the two-process Stage-2(+2.5)→resume-6 flow; base does not."""
    return arm in ("heal25", "rkd")


# ---------------------------------------------------------------------------
# Uniform-166 budget pin (§2) + Stage-1 artifact seeding (pure file I/O)
# ---------------------------------------------------------------------------

def write_uniform_budget(
    shared_budgets_path: Path, out_path: Path, *, survivors: int = PROBE_SURVIVORS,
) -> dict:
    """Write a ``stage1_budgets.json`` with ``per_layer_target_experts`` pinned
    to ``survivors`` for EVERY MoE layer.

    Reads the shared Stage-1 budgets to learn the layer set (and to preserve
    the rest of the payload schema, e.g. ``per_layer_redundancy``), then
    overwrites every per-layer target with ``survivors``. The merge path reads
    this as the per-layer survivor ``target`` (orchestrator.py:785-789,1647);
    166 > the ``min_experts_per_layer`` floor (256//2=128) so the floor never
    binds. Returns the written payload.
    """
    payload = dict(load_json_artifact(shared_budgets_path))
    per_layer = payload.get("per_layer_target_experts")
    if not isinstance(per_layer, dict) or not per_layer:
        raise RuntimeError(
            f"{shared_budgets_path} has no non-empty 'per_layer_target_experts'; "
            "cannot derive the MoE layer set for the uniform budget pin."
        )
    payload["per_layer_target_experts"] = {
        str(k): int(survivors) for k in per_layer
    }
    save_json_artifact(payload, out_path)
    return payload


def seed_stage1_artifacts(
    row_dir: Path, shared_dir: Path, *, group: str, survivors: int = PROBE_SURVIVORS,
) -> None:
    """Seed the row dir with the three Stage-1 artifacts Stage 2 consumes.

    ``stage1_blacklist.json`` + ``budget_decomposition.json`` are copied
    verbatim from the shared Stage-1 run. ``stage1_budgets.json`` is the
    uniform-166 pin for REAM rows; for REAP rows the budget is ignored for
    selection but is still pinned to 166 so ``target==166`` and the survivor
    guard holds for both groups.
    """
    row_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stage1_blacklist.json", "budget_decomposition.json"):
        src = shared_dir / name
        if not src.exists():
            raise RuntimeError(
                f"seed_stage1_artifacts: shared Stage-1 artifact missing: {src}. "
                "Run the shared Stage-1 step before the probe."
            )
        shutil.copy2(src, row_dir / name)
    shared_budgets = shared_dir / "stage1_budgets.json"
    if not shared_budgets.exists():
        raise RuntimeError(
            f"seed_stage1_artifacts: shared budgets missing: {shared_budgets}."
        )
    write_uniform_budget(shared_budgets, row_dir / "stage1_budgets.json",
                         survivors=survivors)


def _write_config(cfg: dict, row_dir: Path) -> Path:
    """Write the per-row config YAML for forensic record + subprocess pass-through."""
    row_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = row_dir / "probe_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# Idempotent completion gate
# ---------------------------------------------------------------------------

def is_complete(row_dir: Path) -> bool:
    """Skip-if-already-run: the Stage-6alt artifact is the gate."""
    return (row_dir / STAGE6ALT_ARTIFACT).exists()


# ---------------------------------------------------------------------------
# Per-row runner (subprocess; mirrors run_ablations.py memory discipline)
# ---------------------------------------------------------------------------

def _pipeline_argv(cfg_path: Path, model_repo: str, row_dir: Path,
                   resume: int, stop: int,
                   extra: "list[str] | None" = None) -> list[str]:
    argv = [
        sys.executable, "-m", "moe_compress.run_pipeline",
        "--config", str(cfg_path),
        "--model", model_repo,
        "--artifacts-dir", str(row_dir),
        "--target-ratio", "0.35",
        "--resume-from-stage", str(resume),
        "--stop-after-stage", str(stop),
    ]
    if extra:
        argv.extend(extra)
    return argv


def _pruned_checkpoint_complete(pruned: Path) -> bool:
    """A stage2_pruned/ dir is usable only if it has a config + weight shard.

    Guards against a killed prior prune leaving a half-written dir that the
    resume shortcut would otherwise load as a complete checkpoint (corrupting
    every arm that branches off it).
    """
    return (
        pruned.is_dir()
        and (pruned / "config.json").exists()
        and any(pruned.glob("*.safetensors"))
    )


def run_shared_prune(
    *, group: str, base_config: dict, shared_dir: Path, probe_root: Path,
    model_repo: str, num_sequences: int,
) -> Path:
    """Run the deterministic Stage-2 prune/merge for a GROUP exactly ONCE.

    The faithful prune (reap) and frequency-weighted merge (ream) are identical
    across that group's three arms (base/heal25/rkd differ only in Stage-2.5).
    Running them per-row re-did the same expensive Stage-2 work 3×. Instead we
    produce one ``_shared_prune/<group>/stage2_pruned/`` via the pipeline's
    designed split-machine flow (``--skip-stage2p5``: run Stage 2, write
    ``stage2_pruned/``, return), which each arm then symlinks + resumes from.

    Returns the path to the shared ``stage2_pruned/`` dir. Idempotent.
    """
    prune_dir = probe_root / "_shared_prune" / group
    pruned = prune_dir / "stage2_pruned"
    if pruned.exists():
        if _pruned_checkpoint_complete(pruned):
            log.info("[%s] shared prune already present → %s", group, pruned)
            return pruned
        # Partial/corrupt dir from a killed prior run: dir-existence alone is
        # not enough (resume shortcut would load it as complete → corrupt model
        # for all 3 arms). Discard and re-prune.
        log.warning("[%s] shared prune dir is incomplete (no config.json/.safetensors) "
                    "— discarding %s and re-pruning", group, pruned)
        shutil.rmtree(pruned)

    prune_dir.mkdir(parents=True, exist_ok=True)
    # Arm is irrelevant to the Stage-2 prune (arms differ only in Stage-2.5);
    # use "base" purely to materialise the group's prune_mode + survivor pin.
    cfg = build_probe_config(base_config, group=group, arm="base",
                             num_sequences=num_sequences)
    cfg_path = _write_config(cfg, prune_dir)
    seed_stage1_artifacts(prune_dir, shared_dir, group=group)

    log.info("[%s] shared prune: Stage-2 once (--skip-stage2p5) → %s", group, prune_dir)
    rc = subprocess.run(
        _pipeline_argv(cfg_path, model_repo, prune_dir, resume=2, stop=2,
                       extra=["--skip-stage2p5"]),
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"[{group}] shared prune returned exit code {rc}")
    if not _pruned_checkpoint_complete(pruned):
        raise RuntimeError(
            f"[{group}] shared prune reported success but {pruned} is missing or "
            "incomplete (no config.json / .safetensors)"
        )
    return pruned


def run_one_row(
    *, row_id: str, group: str, arm: str, base_config: dict, shared_dir: Path,
    probe_root: Path, model_repo: str, num_sequences: int,
    pruned_src: Path, hf_token: str | None = None,
) -> dict[str, Any]:
    """Drive one probe row to its Stage-6alt artifact + per-row HF upload.

    Branches off the GROUP's shared ``stage2_pruned/`` (``pruned_src``): the
    pruned model is symlinked into the row dir, so the pipeline's resume
    shortcut skips the (already-done) Stage-2 prune and loads it from disk.
    base arm: ONE subprocess (resume@2 → skip prune → Stage 6alt).
    heal arms: TWO subprocesses (resume@2 → skip prune, run Stage 2.5; then
    resume@6 → Stage 6alt). Idempotent: a present stage6alt_eval.json short-circuits.
    """
    row_dir = probe_root / row_id
    row_dir.mkdir(parents=True, exist_ok=True)

    if is_complete(row_dir):
        log.info("[%s] already complete — loading prior result", row_id)
        result = json.loads((row_dir / STAGE6ALT_ARTIFACT).read_text())
        result["_row_id"] = row_id
        return result

    cfg = build_probe_config(base_config, group=group, arm=arm,
                             num_sequences=num_sequences)
    cfg_path = _write_config(cfg, row_dir)
    seed_stage1_artifacts(row_dir, shared_dir, group=group)

    # Branch off the shared prune: symlink stage2_pruned/ so the resume shortcut
    # (run_pipeline.py:145-151) skips the deterministic Stage-2 work. It is
    # read-only on resume (Stage 2.5 writes a separate stage2p5_final/), so a
    # symlink is safe and avoids copying the ~46 GB pruned model per row.
    row_pruned = row_dir / "stage2_pruned"
    if not row_pruned.exists():
        os.symlink(Path(pruned_src).resolve(), row_pruned, target_is_directory=True)

    log.info("[%s] starting (group=%s arm=%s) — branched off shared prune", row_id, group, arm)
    if not is_heal_arm(arm):
        # base: single invocation, skip_intermediate → straight to Stage 6alt.
        rc = subprocess.run(
            _pipeline_argv(cfg_path, model_repo, row_dir, resume=2, stop=6),
            check=False,
        ).returncode
        if rc != 0:
            raise RuntimeError(f"[{row_id}] base pipeline returned exit code {rc}")
    else:
        # heal: Stage 2 + auto Stage 2.5 (stop@2), then Stage 6alt (resume@6).
        rc1 = subprocess.run(
            _pipeline_argv(cfg_path, model_repo, row_dir, resume=2, stop=2),
            check=False,
        ).returncode
        if rc1 != 0:
            raise RuntimeError(f"[{row_id}] Stage 2/2.5 returned exit code {rc1}")
        # Stage 2.5 must have produced stage2p5_final/. If it didn't (e.g. a
        # defensive exit-0 with no write), the resume@6 below would silently
        # fall back to stage2_pruned/ (the symlinked UN-healed model) and emit a
        # base-like BPT mislabeled as a heal arm — corrupting the comparison.
        if not (row_dir / "stage2p5_final").exists():
            raise RuntimeError(
                f"[{row_id}] Stage 2.5 exited 0 but stage2p5_final/ is absent — "
                "refusing to resume Stage 6 on the un-healed stage2_pruned/."
            )
        rc2 = subprocess.run(
            _pipeline_argv(cfg_path, model_repo, row_dir, resume=6, stop=6),
            check=False,
        ).returncode
        if rc2 != 0:
            raise RuntimeError(f"[{row_id}] Stage 6 returned exit code {rc2}")

    if not is_complete(row_dir):
        raise RuntimeError(
            f"[{row_id}] pipeline succeeded but {STAGE6ALT_ARTIFACT} missing"
        )

    result = json.loads((row_dir / STAGE6ALT_ARTIFACT).read_text())
    result["_row_id"] = row_id
    result["_group"] = group
    result["_arm"] = arm

    # Per-row HF upload to the custom repo id (NOT upload_stage_to_hub's scheme).
    if hf_token:
        _upload_row_model(row_id, row_dir, arm, hf_token)

    return result


def _row_model_dir(row_dir: Path, arm: str) -> Path:
    """The model checkpoint dir to publish: stage2p5_final/ for heal arms,
    stage2_pruned/ for base. For base, stage2_pruned/ is a symlink to the
    group's shared prune — resolve() so the upload targets the real dir."""
    sub = "stage2p5_final" if is_heal_arm(arm) else "stage2_pruned"
    return (row_dir / sub).resolve()


def _upload_row_model(row_id: str, row_dir: Path, arm: str, hf_token: str) -> None:
    """Upload the row's model + stage6alt_eval.json to pirola/calib-v2-probe-{row_id}."""
    try:
        from huggingface_hub import HfApi
    except ImportError:  # pragma: no cover - env without hub
        log.warning("[%s] huggingface_hub not installed — skipping upload", row_id)
        return
    repo_id = f"{HF_REPO_PREFIX}-{row_id}"
    model_dir = _row_model_dir(row_dir, arm)
    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        log.info("[%s] uploading %s → %s", row_id, model_dir, repo_id)
        api.upload_large_folder(folder_path=str(model_dir), repo_id=repo_id,
                                repo_type="model")
        eval_path = row_dir / STAGE6ALT_ARTIFACT
        if eval_path.exists():
            api.upload_file(path_or_fileobj=str(eval_path),
                            path_in_repo=STAGE6ALT_ARTIFACT,
                            repo_id=repo_id, repo_type="model")
        log.info("[%s] durable on Hub: %s", row_id, repo_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] upload failed (artifacts remain on disk): %s", row_id, exc)


# ---------------------------------------------------------------------------
# Winner pick (lowest student_bpt) + winner upload + local download
# ---------------------------------------------------------------------------

def _student_bpt(result: dict) -> float:
    """student_bpt for ranking; non-finite/missing sinks to the bottom."""
    v = result.get("student_bpt")
    return v if isinstance(v, (int, float)) and math.isfinite(v) else float("inf")


def pick_winner(results: dict[str, dict]) -> str | None:
    """Row id with the lowest student_bpt (lower = less damage), or None."""
    ranked = sorted(results.items(), key=lambda kv: _student_bpt(kv[1]))
    if not ranked or not math.isfinite(_student_bpt(ranked[0][1])):
        return None
    return ranked[0][0]


def promote_winner(
    winner_row_id: str, probe_root: Path, *, arm: str, hf_token: str | None,
    download_to: Path | None,
) -> None:
    """Upload the winner to pirola/calib-v2-probe-winner + optional local snapshot."""
    if hf_token:
        try:
            from huggingface_hub import HfApi
            model_dir = _row_model_dir(probe_root / winner_row_id, arm)
            api = HfApi(token=hf_token)
            api.create_repo(HF_WINNER_REPO, repo_type="model", private=True,
                            exist_ok=True)
            log.info("Promoting winner %s → %s", winner_row_id, HF_WINNER_REPO)
            api.upload_large_folder(folder_path=str(model_dir),
                                    repo_id=HF_WINNER_REPO, repo_type="model")
        except Exception as exc:  # noqa: BLE001
            log.warning("winner upload failed: %s", exc)
    if download_to is not None:
        try:
            from huggingface_hub import snapshot_download
            download_to.mkdir(parents=True, exist_ok=True)
            log.info("Downloading winner → %s", download_to)
            snapshot_download(repo_id=HF_WINNER_REPO, repo_type="model",
                              local_dir=str(download_to), token=hf_token)
        except Exception as exc:  # noqa: BLE001
            log.warning("winner download failed: %s", exc)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="6-model REAP/REAM healing probe")
    parser.add_argument("--config", required=True,
                        help="Base YAML (qwen36_35b_a3b_reap_faithful.yaml)")
    parser.add_argument("--model", required=True, help="HF model repo")
    parser.add_argument("--probe-root", required=True,
                        help="Root dir for per-row artifacts + _shared/ Stage-1")
    parser.add_argument("--num-sequences", type=int, default=4000)
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of row ids (e.g. reap-base,ream-rkd)")
    parser.add_argument("--download-winner-to", default=None,
                        help="Local dir to snapshot_download the winner (default off)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    log.info("========== 6-model REAP/REAM healing probe ==========")

    base_config = yaml.safe_load(Path(args.config).read_text())
    probe_root = Path(args.probe_root)
    probe_root.mkdir(parents=True, exist_ok=True)
    shared_dir = probe_root / "_shared"
    hf_token = os.environ.get("HF_TOKEN") or None

    rows = probe_rows()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        rows = [r for r in rows if r[0] in wanted]
    log.info("Will run %d row(s): %s", len(rows), [r[0] for r in rows])

    if not all((shared_dir / n).exists() for n in (
        "stage1_blacklist.json", "stage1_budgets.json", "budget_decomposition.json",
    )):
        raise RuntimeError(
            f"Shared Stage-1 artifacts missing in {shared_dir}. Run the shared "
            "Stage-1 step (run_pipeline --stop-after-stage 1) into _shared/ first."
        )

    # Prune/merge ONCE per group, then branch its arms off the shared
    # stage2_pruned/. A failed shared prune fails every arm in that group
    # (recorded per-row), but does not abort the other group.
    pruned_by_group: dict[str, Path] = {}
    prune_errors: dict[str, str] = {}
    for grp in list(dict.fromkeys(g for _, g, _ in rows)):  # ordered dedup of groups
        try:
            pruned_by_group[grp] = run_shared_prune(
                group=grp, base_config=base_config, shared_dir=shared_dir,
                probe_root=probe_root, model_repo=args.model,
                num_sequences=args.num_sequences,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] shared prune failed", grp)
            prune_errors[grp] = str(exc)

    results: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    for row_id, group, arm in rows:
        if group in prune_errors:
            failures.append((row_id, f"shared prune failed: {prune_errors[group]}"))
            continue
        try:
            results[row_id] = run_one_row(
                row_id=row_id, group=group, arm=arm, base_config=base_config,
                shared_dir=shared_dir, probe_root=probe_root,
                model_repo=args.model, num_sequences=args.num_sequences,
                pruned_src=pruned_by_group[group], hf_token=hf_token,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] failed", row_id)
            failures.append((row_id, str(exc)))

    save_json_artifact(
        {"results": results,
         "failures": [{"row_id": r, "error": e} for r, e in failures]},
        probe_root / "_probe_summary.json",
    )

    winner = pick_winner(results)
    if winner is not None:
        winner_arm = next(a for rid, _, a in rows if rid == winner)
        log.info("Winner: %s (student_bpt=%.4f)", winner,
                 _student_bpt(results[winner]))
        promote_winner(
            winner, probe_root, arm=winner_arm, hf_token=hf_token,
            download_to=(Path(args.download_winner_to)
                         if args.download_winner_to else None),
        )
    else:
        log.warning("No finite-BPT winner among %d completed rows", len(results))

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
