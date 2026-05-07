"""Orchestrator — runs Strategy A end-to-end with per-stage artifact checkpointing.

Usage:

    python -m moe_compress.run_pipeline \\
        --config configs/qwen36_35b_a3b_30pct.yaml \\
        --model Qwen/Qwen3.6-35B-A3B \\
        --artifacts-dir ./artifacts \\
        --target-ratio 0.30 \\
        [--resume-from-stage N]

Each stage:
1. verifies its dependency artifacts exist
2. loads the checkpoint from the previous stage (or the original for Stage 1)
3. runs its ``run(...)`` function
4. writes its artifact(s) atomically

Stage resume: ``--resume-from-stage N`` skips stages before N and loads the
Stage (N-1) checkpoint if it exists on disk.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

from .budget import solver as budget_solver
from . import (
    stage1_grape,
    stage2_reap_ream,
    stage3_svd,
    stage4_eora,
    stage5_router_kd,
    stage6_validate,
)
from .utils.hub_upload import (
    hub_repo_base_from_env,
    upload_stage_to_hub,
    wait_for_pending_uploads,
)
from .utils.model_io import load_json_artifact, load_model, load_compressed_model, save_json_artifact
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


def _finish_stage(stage_idx, t_start: float, repo_id: str | None) -> None:
    """Log stage completion + push timing scalar to Trackio."""
    dt = time.monotonic() - t_start
    h = int(dt // 3600); m = int((dt % 3600) // 60); s = int(dt % 60)
    log.info("Stage %s done in %dh%02dm%02ds — durable on Hub: %s",
             stage_idx, h, m, s, repo_id or "<not uploaded>")
    _trackio_log({f"pipeline/stage_{stage_idx}_seconds": dt})


STAGE_REGISTRY = {
    1: ("stage1_budgets.json",                 "original"),
    2: ("stage2_pruned",                       "original"),
    # Stage 2.5 (Router KD post-merge) is not in this registry because it is
    # always run immediately after Stage 2 (not resumable as a standalone entry
    # point via --resume-from-stage). Its output is "stage2p5_final".
    3: ("stage3_svd",                          "stage2p5_final"),
    4: ("stage4_eora",                         "stage3_svd"),
    5: ("stage5_final",                        "stage4_eora"),
    6: ("stage6_eval.json",                    "stage5_final"),
}


def main(argv=None) -> int:
    args = _parse(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    config = _load_config(args.config)
    artifacts_dir = Path(args.artifacts_dir).absolute()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Override a few fields from CLI
    if args.model:
        config["model"]["name_or_path"] = args.model
    if args.target_ratio is not None:
        config["target"]["total_reduction_ratio"] = args.target_ratio

    _validate_config(config)

    log.info("Artifacts directory: %s", artifacts_dir)
    log.info("Pipeline target: %.1f%% total parameter reduction", config["target"]["total_reduction_ratio"] * 100)

    start = args.resume_from_stage
    stop = args.stop_after_stage
    model, tokenizer = _load_for_stage(start, config, artifacts_dir,
                                       stop_after_stage=stop)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One-shot Trackio emit of run-level config so the dashboard's run-summary
    # carries model name + target compression ratio + device without parsing
    # per-stage logs. All keys read existing in-scope values; no new state.
    _trackio_log({
        "pipeline/config/model_name": str(config["model"]["name_or_path"]),
        "pipeline/config/target_reduction_ratio": float(config["target"]["total_reduction_ratio"]),
        "pipeline/config/expert_svd_ratio": float(config["target"]["expert_svd_ratio"]),
        "pipeline/config/device": device.type,
        "pipeline/config/resume_from_stage": int(start),
        "pipeline/config/stop_after_stage": int(stop),
    })

    if start <= 1 <= stop:
        log.info("=== Stage 1 — Super Expert Detection + GRAPE Budgets ===")
        t1 = time.monotonic()

        # First pass: approximate budget (blacklist unknown yet)
        decomposition = budget_solver.solve(
            model,
            target_total_reduction=config["target"]["total_reduction_ratio"],
            ep_sp_knob_ratio=config["target"]["expert_svd_ratio"],
            min_experts_per_layer=config["stage1_grape"]["min_experts_per_layer"],
            blacklisted_experts={},
        )

        # Stage 1: SE detection + CKA + GRAPE
        blacklist_path, budgets_path = stage1_grape.run(
            model, tokenizer, config, artifacts_dir, decomposition, device=device,
        )

        # Re-run budget solver with actual blacklist for accurate decomposition
        blacklist_payload = load_json_artifact(blacklist_path)
        blacklist = {int(k): list(v) for k, v in blacklist_payload.get("blacklist", {}).items()}
        decomposition = budget_solver.solve(
            model,
            target_total_reduction=config["target"]["total_reduction_ratio"],
            ep_sp_knob_ratio=config["target"]["expert_svd_ratio"],
            min_experts_per_layer=config["stage1_grape"]["min_experts_per_layer"],
            blacklisted_experts=blacklist,
        )
        save_json_artifact(decomposition.as_dict(), artifacts_dir / "budget_decomposition.json")
        _finish_stage(1, t1, None)
    else:
        decomp_path = artifacts_dir / "budget_decomposition.json"
        if decomp_path.exists():
            payload = load_json_artifact(decomp_path)
            decomposition = budget_solver.BudgetDecomposition(**{
                k: v for k, v in payload.items() if k in budget_solver.BudgetDecomposition.__dataclass_fields__
            })
        else:
            decomposition = None

    if stop < 2:
        log.info("Stopping after stage %d as requested.", stop)
        wait_for_pending_uploads()
        return 0

    # Make the optional save a no-op if the caller asked us to skip it.
    if args.skip_save:
        from .utils import model_io as _mio
        _mio.save_checkpoint = _skip_save_checkpoint

    hub_base = hub_repo_base_from_env()

    if start <= 2 <= stop:
        if start >= 2:
            _validate_stage1_artifacts(artifacts_dir)
        log.info("=== Stage 2 — REAP + REAM ===")
        t2 = time.monotonic()
        stage2_reap_ream.run(model, tokenizer, config, artifacts_dir, device=device,
                             no_resume=args.no_resume)
        repo2 = upload_stage_to_hub(2, artifacts_dir, repo_base=hub_base) if hub_base else None
        _finish_stage(2, t2, repo2)

        # Stage 2.5 — Router KD post-merge: recalibrate routers so Stage 3
        # covariance collection sees already-adapted routing decisions.
        # Always runs immediately after Stage 2 (no standalone resume entry point).
        log.info("=== Stage 2.5 — Post-Merge Router KD ===")
        t2p5 = time.monotonic()
        stage5_router_kd.run(model, tokenizer, config, artifacts_dir, device=device,
                             no_resume=args.no_resume, stage_key="stage2p5")
        repo2p5 = upload_stage_to_hub("2p5", artifacts_dir, repo_base=hub_base) if hub_base else None
        _finish_stage("2p5", t2p5, repo2p5)
    if stop < 3:
        log.info("Stopping after stage %d as requested.", stop)
        wait_for_pending_uploads()
        return 0

    if start <= 3 <= stop:
        log.info("=== Stage 3 — SVD ===")
        t3 = time.monotonic()
        stage3_svd.run(model, tokenizer, config, artifacts_dir, decomposition, device=device,
                       no_resume=args.no_resume)
        repo3 = upload_stage_to_hub(3, artifacts_dir, repo_base=hub_base) if hub_base else None
        _finish_stage(3, t3, repo3)
    if stop < 4:
        log.info("Stopping after stage %d as requested.", stop)
        wait_for_pending_uploads()
        return 0

    if start <= 4 <= stop:
        log.info("=== Stage 4 — EoRA ===")
        t4 = time.monotonic()
        stage4_eora.run(model, tokenizer, config, artifacts_dir, no_resume=args.no_resume)
        repo4 = upload_stage_to_hub(4, artifacts_dir, repo_base=hub_base) if hub_base else None
        _finish_stage(4, t4, repo4)
    if stop < 5:
        log.info("Stopping after stage %d as requested.", stop)
        wait_for_pending_uploads()
        return 0

    if start <= 5 <= stop:
        log.info("=== Stage 5 — Router KD ===")
        t5 = time.monotonic()
        stage5_router_kd.run(model, tokenizer, config, artifacts_dir, device=device,
                             no_resume=args.no_resume)
        repo5 = upload_stage_to_hub(5, artifacts_dir, repo_base=hub_base) if hub_base else None
        _finish_stage(5, t5, repo5)
    if stop < 6:
        log.info("Stopping after stage %d as requested.", stop)
        wait_for_pending_uploads()
        return 0

    if start <= 6 <= stop:
        log.info("=== Stage 6 — Validation ===")
        stage6_validate.run(model, tokenizer, config, artifacts_dir, device=device)

    wait_for_pending_uploads()
    log.info("Pipeline complete.")
    return 0


# ---------------------------------------------------------------------------


def _parse(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strategy A (Maximum Quality) MoE compression pipeline.")
    p.add_argument("--config", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--artifacts-dir", default="./artifacts")
    p.add_argument("--target-ratio", type=float, default=None)
    p.add_argument("--resume-from-stage", type=int, default=1)
    p.add_argument(
        "--stop-after-stage", type=int, default=6,
        help="Exit after the named stage completes (inclusive). Useful for "
             "per-stage supervision on HF Jobs. 6 = run everything.",
    )
    p.add_argument(
        "--skip-save", action="store_true",
        help="Skip save_checkpoint calls between stages. For in-memory smoke "
             "testing on tiny models that don't round-trip through HF save.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Disable crash-resume I/O for stages 2–5. Each stage runs from scratch.",
    )
    return p.parse_args(argv)


def _load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _skip_save_checkpoint(model, tokenizer, out_dir):
    """Replacement for save_checkpoint when --skip-save is passed."""
    from pathlib import Path as _Path
    out = _Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("[--skip-save] suppressing save_pretrained → %s", out)
    return out


def _validate_stage1_artifacts(artifacts_dir: Path) -> None:
    """Raise with a clear error if Stage 1 output artifacts are missing or corrupt."""
    required = [
        artifacts_dir / "stage1_blacklist.json",
        artifacts_dir / "stage1_budgets.json",
        artifacts_dir / "budget_decomposition.json",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Stage 1 artifact missing: {p}\n"
                "Run with --resume-from-stage 1 (or from scratch) to regenerate."
            )
        try:
            load_json_artifact(p)
        except Exception as exc:
            raise RuntimeError(
                f"Stage 1 artifact corrupted: {p}: {exc}\n"
                "Delete the file and re-run Stage 1."
            ) from exc


def _validate_config(config: dict) -> None:
    """Catch known-bad configurations before the pipeline starts a long run."""
    min_exp = config["stage1_grape"]["min_experts_per_layer"]
    if min_exp < 9:
        raise ValueError(
            f"stage1_grape.min_experts_per_layer={min_exp} is below the "
            "recommended floor of 9 (top-k=8 + 1 headroom)."
        )
    target = config["target"]["total_reduction_ratio"]
    if not (0.0 < target < 1.0):
        raise ValueError(f"target.total_reduction_ratio={target} must be in (0, 1).")
    ratio = config["target"]["expert_svd_ratio"]
    if ratio <= 0:
        raise ValueError(f"target.expert_svd_ratio={ratio} must be > 0.")


# F-iter4-CRIT-1: Spec §9 lines 821, 838 require BOTH teacher and student to
# run under attn_implementation="eager" for the Stage 6 quality gate. The
# teacher is pinned at load time inside stage6_validate.py; the student is
# loaded here, so we override the config's attn_implementation when this run
# will reach Stage 6 (the default for production runs).
_STAGE6_ATTN_IMPLEMENTATION = "eager"


def _load_for_stage(stage: int, config: dict, artifacts_dir: Path,
                    *, stop_after_stage: int = 6):
    """Load the model + tokenizer appropriate for starting at ``stage``."""
    # F-iter4-CRIT-1: Spec §9 lines 821, 838 require eager attn for the Stage 6
    # gate run for both teacher and student. The teacher is pinned at load time
    # inside stage6_validate.py; the student is loaded here, so override the
    # config's attn_implementation when this run will reach Stage 6.
    cfg_attn = config["model"]["attn_implementation"]
    will_run_stage6 = stop_after_stage >= 6
    student_attn = _STAGE6_ATTN_IMPLEMENTATION if will_run_stage6 else cfg_attn
    if will_run_stage6 and student_attn != cfg_attn:
        log.info(
            "Stage 6 will run (stop_after_stage=%d): overriding "
            "model.attn_implementation %r -> %r for student load to satisfy "
            "Spec §9 lines 821, 838 (eager attn for Stage 6 gate).",
            stop_after_stage, cfg_attn, student_attn,
        )
    if stage <= 2:
        return load_model(
            config["model"]["name_or_path"],
            revision=config["model"].get("revision", "main"),
            torch_dtype=config["model"]["torch_dtype"],
            device_map=config["model"]["device_map"],
            attn_implementation=student_attn,
            load_in_4bit=config["model"].get("load_in_4bit", False),
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )
    # Stage 3's predecessor is stage2p5_final (post-merge Router KD output).
    # If resuming directly from stage 3 without running stage 2.5 (e.g., the
    # stage2p5_final dir does not exist), fall back to stage2_pruned so that
    # the pipeline can still resume when stage2p5_final is on a Hub repo that
    # was downloaded by the job entrypoint.
    if stage == 3:
        for candidate in ("stage2p5_final", "stage2_pruned"):
            prev_path = artifacts_dir / candidate
            if prev_path.exists():
                if candidate == "stage2_pruned":
                    log.warning(
                        "Loading stage 3 input from %s — stage2p5_final not found; "
                        "Stage 2.5 router recalibration will be absent from this run.",
                        prev_path,
                    )
                else:
                    log.info("Loading stage 3 input from %s", prev_path)
                model, tokenizer, _ = load_compressed_model(
                    prev_path,
                    device_map=config["model"]["device_map"],
                    torch_dtype=config["model"]["torch_dtype"],
                    attn_implementation=student_attn,
                )
                return model, tokenizer
        raise FileNotFoundError(
            "Cannot resume from stage 3: neither stage2p5_final/ nor stage2_pruned/ "
            f"exists under {artifacts_dir}. Run stages 1–2.5 first."
        )
    prev_dir_name = STAGE_REGISTRY[stage][1]
    prev_path = artifacts_dir / prev_dir_name
    if not prev_path.exists():
        raise FileNotFoundError(
            f"Cannot resume from stage {stage}: expected checkpoint at {prev_path}.\n"
            f"  - If running locally: run stages 1..{stage - 1} first so "
            f"{prev_dir_name}/ exists under {artifacts_dir}.\n"
            f"  - If running on HF Jobs: set BOTH "
            f"RESUME_FROM_STAGE={stage} AND PRIOR_STAGE_REPO=<the Hub repo "
            f"holding the {prev_dir_name}/ output of stage {stage - 1}>."
        )
    log.info("Loading stage %d input from %s", stage, prev_path)
    model, tokenizer, _ = load_compressed_model(
        prev_path,
        device_map=config["model"]["device_map"],
        torch_dtype=config["model"]["torch_dtype"],
        attn_implementation=student_attn,
    )
    return model, tokenizer


if __name__ == "__main__":
    sys.exit(main())
