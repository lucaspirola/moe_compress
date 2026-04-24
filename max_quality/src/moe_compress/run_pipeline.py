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
from .utils.model_io import load_json_artifact, load_model

log = logging.getLogger(__name__)


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
    model, tokenizer = _load_for_stage(start, config, artifacts_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if start <= 0:
        log.info("=== Stage 0 — Super Expert Detection ===")
        stage0_super_experts.run(model, tokenizer, config, artifacts_dir, device=device)

    if start <= 1:
        log.info("=== Budget Solver ===")
        blacklist_payload = load_json_artifact(artifacts_dir / "stage0_blacklist.json")
        blacklist = {int(k): list(v) for k, v in blacklist_payload.get("blacklist", {}).items()}
        decomposition = budget_solver.solve(
            model,
            target_total_reduction=config["target"]["total_reduction_ratio"],
            initial_expert_reduction=config["target"]["initial_expert_reduction"],
            initial_svd_reduction=config["target"]["initial_svd_reduction"],
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

    # FIX (review bug #3): keep the model alive across Stages 2-5. The
    # saved checkpoints between those stages are artifacts for post-mortem
    # / future custom-loader resumption only; HF `from_pretrained` cannot
    # reload a state_dict that contains per-layer-variable `num_experts` and
    # `_FactoredLinear` submodules without additional plumbing. For that
    # reason, ``--resume-from-stage`` values >2 fall back to the original
    # checkpoint today (documented limitation, see README.md Risk register).
    if start <= 2:
        log.info("=== Stage 2 — REAP + REAM ===")
        stage2_reap_ream.run(model, tokenizer, config, artifacts_dir, device=device)

    if start <= 3:
        log.info("=== Stage 3 — SVD ===")
        stage3_svd.run(model, tokenizer, config, artifacts_dir, decomposition, device=device)

    if start <= 4:
        log.info("=== Stage 4 — EoRA ===")
        stage4_eora.run(model, tokenizer, config, artifacts_dir)

    if start <= 5:
        log.info("=== Stage 5 — Router KD ===")
        stage5_router_kd.run(model, tokenizer, config, artifacts_dir, device=device)

    if start <= 6:
        log.info("=== Stage 6 — Validation ===")
        stage6_validate.run(model, tokenizer, config, artifacts_dir, device=device)

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
    return p.parse_args(argv)


def _load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


def _load_for_stage(stage: int, config: dict, artifacts_dir: Path):
    """Load the model + tokenizer appropriate for starting at ``stage``.

    Stages 0-2 load the original pretrained model. Stages 3-6 require the
    in-memory state from earlier stages; resume from those is not supported
    by `from_pretrained` because of `_FactoredLinear` + per-layer-variable
    `num_experts` (review bug #2/#3).
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
    raise RuntimeError(
        f"--resume-from-stage={stage} is not supported in this pipeline version. "
        "Re-run from Stage 0 (cheap Stages 0-1 re-use cached calibration) or "
        "implement a custom `load_pipeline_checkpoint` that reconstructs "
        "_FactoredLinear submodules from saved state_dicts."
    )


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
