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
2. loads the checkpoint from the previous stage (or the original for Stage 0)
3. runs its ``run(...)`` function
4. writes its artifact(s) atomically

Stage resume: ``--resume-from-stage N`` skips 0..N-1 and loads the Stage (N-1)
checkpoint if it exists on disk.
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
    stage0_super_experts,
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
from .utils.model_io import load_json_artifact, load_model, load_compressed_model
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


def _finish_stage(stage_idx: int, t_start: float, repo_id: str | None) -> None:
    """Log stage completion + push timing scalar to Trackio. The repo name is
    a string and goes to stdout only — Trackio metrics are scalars."""
    dt = time.monotonic() - t_start
    h = int(dt // 3600); m = int((dt % 3600) // 60); s = int(dt % 60)
    log.info("Stage %d done in %dh%02dm%02ds — durable on Hub: %s",
             stage_idx, h, m, s, repo_id or "<not uploaded>")
    _trackio_log({f"pipeline/stage_{stage_idx}_seconds": dt})


STAGE_REGISTRY = {
    0: ("stage0_blacklist.json",               "original"),
    1: ("stage1_budgets.json",                 "original"),
    2: ("stage2_pruned",                       "original"),
    3: ("stage3_svd",                          "stage2_pruned"),
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

    # Figure out which checkpoint to load for the starting stage.
    start = args.resume_from_stage
    stop = args.stop_after_stage
    model, tokenizer = _load_for_stage(start, config, artifacts_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if start <= 0 <= stop:
        log.info("=== Stage 0 — Super Expert Detection ===")
        stage0_super_experts.run(model, tokenizer, config, artifacts_dir, device=device)
    if stop < 1:
        log.info("Stopping after stage %d as requested.", stop)
        return 0

    if start <= 1 <= stop:
        log.info("=== Budget Solver ===")
        blacklist_payload = load_json_artifact(artifacts_dir / "stage0_blacklist.json")
        blacklist = {int(k): list(v) for k, v in blacklist_payload.get("blacklist", {}).items()}
        decomposition = budget_solver.solve(
            model,
            target_total_reduction=config["target"]["total_reduction_ratio"],
            expert_svd_ratio=config["target"]["expert_svd_ratio"],
            min_experts_per_layer=config["stage1_grape"]["min_experts_per_layer"],
            blacklisted_experts=blacklist,
        )
        (artifacts_dir / "budget_decomposition.json").write_text(
            __import__("json").dumps(decomposition.as_dict(), indent=2)
        )
        log.info("=== Stage 1 — GRAPE Budgets ===")
        stage1_grape.run(model, config, artifacts_dir, decomposition)
    else:
        decomp_path = artifacts_dir / "budget_decomposition.json"
        if decomp_path.exists():
            payload = load_json_artifact(decomp_path)
            decomposition = budget_solver.BudgetDecomposition(**{
                k: v for k, v in payload.items() if k in budget_solver.BudgetDecomposition.__dataclass_fields__
            })
        else:
            decomposition = None  # not needed for Stage 4+
    if stop < 2:
        log.info("Stopping after stage %d as requested.", stop)
        return 0

    # FIX (review bug #3): keep the model alive across Stages 2-5. The
    # saved checkpoints between those stages are artifacts for post-mortem
    # / future custom-loader resumption only; HF `from_pretrained` cannot
    # reload a state_dict that contains per-layer-variable `num_experts` and
    # `_FactoredLinear` submodules without additional plumbing. For that
    # reason, ``--resume-from-stage`` values >2 fall back to the original
    # checkpoint today (documented limitation, see README.md Risk register).

    # Make the optional save a no-op if the caller asked us to skip it.
    if args.skip_save:
        from .utils import model_io as _mio
        _mio.save_checkpoint = _skip_save_checkpoint

    hub_base = hub_repo_base_from_env()

    if start <= 2 <= stop:
        log.info("=== Stage 2 — REAP + REAM ===")
        t2 = time.monotonic()
        stage2_reap_ream.run(model, tokenizer, config, artifacts_dir, device=device)
        repo2 = upload_stage_to_hub(2, artifacts_dir, repo_base=hub_base) if hub_base else None
        _finish_stage(2, t2, repo2)
    if stop < 3:
        log.info("Stopping after stage %d as requested.", stop)
        wait_for_pending_uploads()
        return 0

    if start <= 3 <= stop:
        log.info("=== Stage 3 — SVD ===")
        t3 = time.monotonic()
        stage3_svd.run(model, tokenizer, config, artifacts_dir, decomposition, device=device)
        repo3 = upload_stage_to_hub(3, artifacts_dir, repo_base=hub_base) if hub_base else None
        _finish_stage(3, t3, repo3)
    if stop < 4:
        log.info("Stopping after stage %d as requested.", stop)
        wait_for_pending_uploads()
        return 0

    if start <= 4 <= stop:
        log.info("=== Stage 4 — EoRA ===")
        t4 = time.monotonic()
        stage4_eora.run(model, tokenizer, config, artifacts_dir)
        repo4 = upload_stage_to_hub(4, artifacts_dir, repo_base=hub_base) if hub_base else None
        _finish_stage(4, t4, repo4)
    if stop < 5:
        log.info("Stopping after stage %d as requested.", stop)
        wait_for_pending_uploads()
        return 0

    if start <= 5 <= stop:
        log.info("=== Stage 5 — Router KD ===")
        t5 = time.monotonic()
        stage5_router_kd.run(model, tokenizer, config, artifacts_dir, device=device)
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
    p.add_argument("--resume-from-stage", type=int, default=0)
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


def _validate_config(config: dict) -> None:
    """Catch known-bad configurations before the pipeline starts a long run.

    This is a small wall against foot-guns like pruning below ``top_k``.
    """
    min_exp = config["stage1_grape"]["min_experts_per_layer"]
    # Reasonable lower bound: any MoE forward with top_k routing needs at
    # least that many experts to pick from. Qwen3.6-35B-A3B defaults to 8
    # routed experts per token; we require a generous headroom.
    if min_exp < 9:
        raise ValueError(
            f"stage1_grape.min_experts_per_layer={min_exp} is below the "
            "recommended floor of 9 (top-k=8 + 1 headroom). Pruning below "
            "top_k causes the router to emit fewer experts than it selects, "
            "triggering dispatch errors."
        )
    target = config["target"]["total_reduction_ratio"]
    if not (0.0 < target < 1.0):
        raise ValueError(f"target.total_reduction_ratio={target} must be in (0, 1).")
    ratio = config["target"]["expert_svd_ratio"]
    if ratio <= 0:
        raise ValueError(f"target.expert_svd_ratio={ratio} must be > 0.")


def _load_for_stage(stage: int, config: dict, artifacts_dir: Path):
    """Load the model + tokenizer appropriate for starting at ``stage``.

    Stages 0-2 load the original pretrained model.
    Stages 3-6 load the previous stage's compressed checkpoint via
    ``load_compressed_model`` which handles both pruned (Qwen3_5MoeExperts)
    and factored (FactoredExperts) layouts.
    """
    if stage <= 2:
        return load_model(
            config["model"]["name_or_path"],
            revision=config["model"]["revision"],
            torch_dtype=config["model"]["torch_dtype"],
            device_map=config["model"]["device_map"],
            attn_implementation=config["model"]["attn_implementation"],
            load_in_4bit=config["model"].get("load_in_4bit", False),
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )
    # Stages 3+: load the checkpoint produced by the preceding stage.
    prev_dir_name = STAGE_REGISTRY[stage][1]
    prev_path = artifacts_dir / prev_dir_name
    if not prev_path.exists():
        # Most common operator error: re-queueing a downstream stage in a
        # fresh job without setting PRIOR_STAGE_REPO. The HF Jobs bucket is
        # non-durable across cancel/timeout (see project memory), so a fresh
        # job starts with an empty artifacts_dir; the prior-stage checkpoint
        # must be hydrated from the per-stage Hub repo by entrypoint.py via
        # PRIOR_STAGE_REPO before _load_for_stage runs.
        raise FileNotFoundError(
            f"Cannot resume from stage {stage}: expected checkpoint at {prev_path}.\n"
            f"  - If running locally: run stages 0..{stage - 1} first so "
            f"{prev_dir_name}/ exists under {artifacts_dir}.\n"
            f"  - If running on HF Jobs: set BOTH "
            f"RESUME_FROM_STAGE={stage} AND PRIOR_STAGE_REPO=<the Hub repo "
            f"holding the {prev_dir_name}/ output of stage {stage - 1}>. "
            f"The entrypoint only hydrates the bucket when RESUME_FROM_STAGE "
            f">= 3 AND PRIOR_STAGE_REPO is non-empty; setting just one is "
            f"silently a no-op."
        )
    log.info("Loading stage %d input from %s", stage, prev_path)
    model, tokenizer, _ = load_compressed_model(
        prev_path,
        device_map=config["model"]["device_map"],
        torch_dtype=config["model"]["torch_dtype"],
        attn_implementation=config["model"]["attn_implementation"],
    )
    return model, tokenizer


def _load_from_dir(path: Path, config: dict):
    if not path.exists():
        raise FileNotFoundError(f"Expected prior stage checkpoint at {path}")
    return load_model(
        str(path),
        revision="main",
        torch_dtype=config["model"]["torch_dtype"],
        device_map=config["model"]["device_map"],
        attn_implementation=config["model"]["attn_implementation"],
        load_in_4bit=config["model"].get("load_in_4bit", False),
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )


if __name__ == "__main__":
    sys.exit(main())
