"""Stage 2 v2 ablation harness — runs A0..A11 sequentially in one job.

Pipeline shape per ablation: Stage 1 (shared) → Stage 2 → Stage 2.5 → Stage 6.
Stages 3, 4, 5 are skipped — Stage 2 alone hits the 35% total-reduction target
because we set ``target.expert_svd_ratio = 100.0`` so the budget solver
allocates ~all of the savings to expert pruning.

The driver is **idempotent**: ablations whose ``stage6_eval.json`` already
exists are skipped on re-invocation. This makes job-timeout + resubmit a
zero-config recovery path — the harness picks up where it left off.

Per-stage Hub uploads are intentionally disabled for ablation runs (set via
``PIPELINE_HUB_RESULT_REPO_BASE=""``); the bucket-mounted artifact dir is
the durability boundary, not the Hub. Twelve ablations × per-stage uploads
would create 36+ junk repos under ``pirola/``.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .run_pipeline import main as run_pipeline_main
from .utils.model_io import load_json_artifact, save_json_artifact
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation matrix — mirrors max_quality/docs/stage2_assignment_revision.md § 8
# ---------------------------------------------------------------------------

# Each row's deltas are applied on top of the base config's `stage2_reap_ream`
# block. All v2 flags absent from a row's deltas inherit their baseline value
# (greedy / pre / none / false / 0) from the YAML.
_A4_BASE: dict[str, Any] = {
    "assignment_solver": "auto",
    "cost_alignment": "post",
    "cost_whitening": "diag",
    "cost_asymmetric": True,
    "em_refinement_rounds": 2,
    "capacity_util_threshold": 0.25,
    "cost_topk_filter": 24,
}
_A7_BASE: dict[str, Any] = {
    "assignment_solver": "auto",
    "cost_alignment": "post",
    "cost_whitening": "full",
    "cost_asymmetric": True,
    "em_refinement_rounds": 3,
    "capacity_util_threshold": 0.25,
    "cost_topk_filter": 48,
}
_A8_BASE: dict[str, Any] = {**_A7_BASE, "expert_distill_steps": 500}

ABLATION_DELTAS: list[tuple[str, dict[str, Any]]] = [
    ("A0",  {}),
    ("A1",  {"assignment_solver": "auto"}),
    ("A2",  {"assignment_solver": "auto", "cost_alignment": "post", "cost_whitening": "diag"}),
    ("A3",  {"assignment_solver": "auto", "cost_alignment": "post", "cost_whitening": "diag", "cost_asymmetric": True}),
    ("A4",  _A4_BASE),
    ("A5",  {**_A4_BASE, "cost_topk_filter": 16}),
    ("A6",  {**_A4_BASE, "em_refinement_rounds": 3}),
    ("A7",  _A7_BASE),
    ("A8",  _A8_BASE),
    ("A9",  {**_A8_BASE, "assignment_solver": "sinkhorn"}),
    ("A10", {**_A8_BASE, "expert_distill_steps": 200}),
    ("A11", {**_A8_BASE, "expert_distill_min_freq_sum": 0.5}),
]


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _build_ablation_config(
    base: dict, deltas: dict, *, num_sequences: int,
    teacher_cache_path: Path,
) -> dict:
    """Apply per-ablation deltas + the 35%-via-Stage-2-only target to the
    base config. The teacher-cache path forces all 12 ablations to read /
    write the same Stage 6 cache file (filled by A0, hit by A1..A11)."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("target", {})
    cfg["target"]["total_reduction_ratio"] = 0.35
    cfg["target"]["expert_svd_ratio"] = 100.0  # force ~100% expert pruning, ~0% SVD
    cfg.setdefault("calibration", {})
    cfg["calibration"]["num_sequences"] = num_sequences
    s2 = cfg.setdefault("stage2_reap_ream", {})
    # Stage 2 reads its own per-stage knob (stage2_reap_ream.num_calibration_samples,
    # see stage2_reap_ream.py:111) — overriding only the global calibration.num_sequences
    # would leave Stage 2 at its YAML-specified value (4000 in prod). Cap it here so the
    # --num-sequences flag actually bounds Stage 2 work.
    s2["num_calibration_samples"] = num_sequences
    for k, v in deltas.items():
        s2[k] = v
    cfg.setdefault("stage6_validate", {})
    cfg["stage6_validate"].setdefault("teacher_eval_cache", {})
    cfg["stage6_validate"]["teacher_eval_cache"]["cache_path"] = str(teacher_cache_path)
    return cfg


# ---------------------------------------------------------------------------
# Per-ablation runner
# ---------------------------------------------------------------------------


def _is_complete(ablation_dir: Path) -> bool:
    """Skip-if-already-run signal. Stage 6's final artifact is the gate."""
    return (ablation_dir / "stage6_eval.json").exists()


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    """Hardlink src → dst when on the same filesystem; fall back to copy."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _bridge_stage25_to_stage6(ablation_dir: Path) -> None:
    """Stage 6 hard-codes its student input as ``stage5_final/``; Stage 2.5
    produces ``stage2p5_final/``. One symlink bridges them — no code change.
    """
    src_name = "stage2p5_final"
    src = ablation_dir / src_name
    dst = ablation_dir / "stage5_final"
    if not src.exists():
        raise RuntimeError(
            f"_bridge_stage25_to_stage6: {src} not found — Stage 2.5 did not "
            "produce its output dir. Cannot run Stage 6."
        )
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(src_name)  # relative symlink — survives bucket mount changes


def _seed_stage1_artifacts(ablation_dir: Path, shared_dir: Path) -> None:
    """Hardlink the three Stage 1 outputs from _shared/ into the ablation
    dir so Stage 2 reads them as if Stage 1 had run locally."""
    for name in ("stage1_blacklist.json", "stage1_budgets.json",
                 "budget_decomposition.json"):
        src = shared_dir / name
        if not src.exists():
            raise RuntimeError(
                f"_seed_stage1_artifacts: shared Stage 1 artifact missing: {src}. "
                "Pre-flight Stage 1 step must run before any ablation."
            )
        _hardlink_or_copy(src, ablation_dir / name)


def _write_ablation_config(cfg: dict, ablation_dir: Path) -> Path:
    """Write the per-ablation config YAML into the ablation dir for forensic
    record + pass-through to run_pipeline.main()."""
    cfg_path = ablation_dir / "ablation_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _run_one_ablation(
    *, ablation_id: str, deltas: dict[str, Any], base_config: dict,
    shared_dir: Path, ablations_root: Path, model_repo: str,
    num_sequences: int, teacher_cache_path: Path,
) -> dict[str, Any]:
    """Drive one ablation through Stage 2 → 2.5 → Stage 6. Stage 1 artifacts
    are seeded from ``shared_dir``; Stage 6 reads the shared teacher cache."""
    ablation_dir = ablations_root / ablation_id
    ablation_dir.mkdir(parents=True, exist_ok=True)

    if _is_complete(ablation_dir):
        log.info("[%s] already complete — loading prior result", ablation_id)
        return json.loads((ablation_dir / "stage6_eval.json").read_text())

    log.info("[%s] starting (deltas=%s)", ablation_id, deltas)
    t_start = time.monotonic()

    # Per-ablation Trackio run.
    try:
        import trackio
        try:
            run = trackio.init(
                project="moe-compress-strategy-a",
                name=f"ablation-{ablation_id}",
                space_id=os.environ.get("TRACKIO_SPACE_ID", "pirola/trackio"),
                config={"ablation_id": ablation_id, **deltas},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] trackio.init failed: %s — continuing without it", ablation_id, exc)
            run = None
    except ImportError:
        run = None

    try:
        cfg = _build_ablation_config(
            base_config, deltas,
            num_sequences=num_sequences,
            teacher_cache_path=teacher_cache_path,
        )
        cfg_path = _write_ablation_config(cfg, ablation_dir)
        _seed_stage1_artifacts(ablation_dir, shared_dir)

        # Stage 2 + Stage 2.5 — one pipeline call (Stage 2.5 is folded into the
        # Stage-2 block of run_pipeline.main, so --stop-after-stage 2 runs both).
        rc1 = run_pipeline_main([
            "--config", str(cfg_path),
            "--model", model_repo,
            "--artifacts-dir", str(ablation_dir),
            "--target-ratio", "0.35",
            "--resume-from-stage", "2",
            "--stop-after-stage", "2",
        ])
        if rc1 != 0:
            raise RuntimeError(f"[{ablation_id}] Stage 2/2.5 returned exit code {rc1}")

        # Bridge: Stage 6 expects stage5_final/.
        _bridge_stage25_to_stage6(ablation_dir)

        # Stage 6 — separate pipeline call so it loads from stage5_final.
        rc2 = run_pipeline_main([
            "--config", str(cfg_path),
            "--model", model_repo,
            "--artifacts-dir", str(ablation_dir),
            "--target-ratio", "0.35",
            "--resume-from-stage", "6",
            "--stop-after-stage", "6",
        ])
        if rc2 != 0:
            raise RuntimeError(f"[{ablation_id}] Stage 6 returned exit code {rc2}")

        if not _is_complete(ablation_dir):
            raise RuntimeError(
                f"[{ablation_id}] Stage 6 succeeded but stage6_eval.json missing"
            )

        elapsed = time.monotonic() - t_start
        result = json.loads((ablation_dir / "stage6_eval.json").read_text())
        result["_ablation_id"] = ablation_id
        result["_deltas"] = deltas
        result["_elapsed_seconds"] = elapsed
        log.info("[%s] complete in %.1f min", ablation_id, elapsed / 60.0)
        return result
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] trackio.run.finish failed: %s", ablation_id, exc)


# ---------------------------------------------------------------------------
# Pre-flight: shared Stage 1 + shared teacher cache
# ---------------------------------------------------------------------------


def _preflight(
    *, base_config: dict, shared_dir: Path, model_repo: str,
    num_sequences: int, teacher_cache_path: Path,
) -> None:
    """Run Stage 1 once on the base config. Idempotent — skips if artifacts
    already present. The teacher cache is filled by A0's Stage 6 (no separate
    teacher-only mode in Stage 6).
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    needed = ("stage1_blacklist.json", "stage1_budgets.json",
              "budget_decomposition.json")
    if all((shared_dir / n).exists() for n in needed):
        log.info("Pre-flight Stage 1 already complete in %s", shared_dir)
        return

    log.info("Pre-flight: running Stage 1 once into %s", shared_dir)
    cfg = _build_ablation_config(
        base_config, deltas={},
        num_sequences=num_sequences,
        teacher_cache_path=teacher_cache_path,
    )
    cfg_path = _write_ablation_config(cfg, shared_dir)
    rc = run_pipeline_main([
        "--config", str(cfg_path),
        "--model", model_repo,
        "--artifacts-dir", str(shared_dir),
        "--target-ratio", "0.35",
        "--resume-from-stage", "1",
        "--stop-after-stage", "1",
    ])
    if rc != 0:
        raise RuntimeError(f"Pre-flight Stage 1 failed with exit code {rc}")
    for n in needed:
        if not (shared_dir / n).exists():
            raise RuntimeError(
                f"Pre-flight Stage 1 returned 0 but {n} is missing from {shared_dir}"
            )
    log.info("Pre-flight Stage 1 complete")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2 v2 ablation harness (A0..A11)",
    )
    parser.add_argument("--config", required=True,
                        help="Base YAML config (e.g., qwen36_35b_a3b_30pct.yaml)")
    parser.add_argument("--model", required=True,
                        help="HF model repo for the base model")
    parser.add_argument("--ablations-root", required=True,
                        help="Root directory for per-ablation artifacts (will create A0..A11 subdirs)")
    parser.add_argument("--num-sequences", type=int, default=1000,
                        help="Calibration sequence count for ablations (default 1000)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of ablation IDs to run (e.g., A0,A4,A8). Default: all 12.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run A0 only with tiny calibration; for local plumbing verification")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    log.info("========== Stage 2 v2 Ablation Harness ==========")

    base_config = yaml.safe_load(Path(args.config).read_text())
    ablations_root = Path(args.ablations_root)
    ablations_root.mkdir(parents=True, exist_ok=True)
    shared_dir = ablations_root / "_shared"
    teacher_cache_path = shared_dir / "teacher_eval_cache.json"

    # Disable per-stage Hub upload for ablations — bucket is the durability layer.
    os.environ.pop("PIPELINE_HUB_RESULT_REPO_BASE", None)

    num_sequences = 8 if args.smoke_test else args.num_sequences

    # Filter ablations to run.
    if args.only:
        only_ids = {x.strip() for x in args.only.split(",") if x.strip()}
        rows = [(aid, d) for (aid, d) in ABLATION_DELTAS if aid in only_ids]
    elif args.smoke_test:
        rows = ABLATION_DELTAS[:1]  # A0 only
    else:
        rows = ABLATION_DELTAS

    log.info("Will run %d ablation(s): %s", len(rows), [r[0] for r in rows])

    # Pre-flight: shared Stage 1 (blacklist + budgets, identical across ablations).
    _preflight(
        base_config=base_config,
        shared_dir=shared_dir,
        model_repo=args.model,
        num_sequences=num_sequences,
        teacher_cache_path=teacher_cache_path,
    )

    # Per-ablation loop.
    results: dict[str, Any] = {}
    failures: list[tuple[str, str]] = []
    for i, (aid, deltas) in enumerate(rows):
        log.info("[%d/%d] === %s ===", i + 1, len(rows), aid)
        try:
            result = _run_one_ablation(
                ablation_id=aid, deltas=deltas, base_config=base_config,
                shared_dir=shared_dir, ablations_root=ablations_root,
                model_repo=args.model, num_sequences=num_sequences,
                teacher_cache_path=teacher_cache_path,
            )
            results[aid] = result
            # Sanity: A0 must populate the teacher cache.
            if aid == "A0" and not teacher_cache_path.exists():
                raise RuntimeError(
                    "A0 completed but teacher_eval_cache.json was not written. "
                    "Subsequent ablations would re-run teacher scoring. Halting."
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] failed", aid)
            failures.append((aid, str(exc)))
            # Continue to the next ablation; failed run is retryable on the
            # next job invocation (the harness is idempotent).

    # Aggregate.
    summary_path = ablations_root / "_summary.json"
    summary = {
        "results": results,
        "failures": [{"ablation_id": a, "error": e} for a, e in failures],
        "num_completed": len(results),
        "num_failed": len(failures),
        "total_ablations_planned": len(rows),
    }
    save_json_artifact(summary, summary_path)
    log.info("Wrote summary to %s", summary_path)

    # Concise dashboard line.
    log.info("=" * 60)
    log.info("Ablation summary (completed=%d, failed=%d):",
             len(results), len(failures))
    for aid, _ in rows:
        if aid in results:
            r = results[aid]
            ppl = r.get("student", {}).get("wikitext2_ppl", "?")
            log.info("  %s : ppl=%s", aid, ppl)
        else:
            log.info("  %s : FAILED or skipped", aid)
    log.info("=" * 60)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
