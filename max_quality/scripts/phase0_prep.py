"""Phase 0 — CPU-only pre-warm script.

Idempotent prep step that runs on a cheap CPU instance with a mounted volume
BEFORE the GPU stages spin up. Pre-fetches everything the Stage 2 + Stage 2.5
runs need from Hugging Face onto the volume so the GPU instances boot ready:

  1. Base model snapshot (Qwen3.6-35B-A3B, ~70 GB) → ``$HF_HOME/hub``
  2. FP8 teacher snapshot (Qwen3.6-35B-A3B-FP8, ~35 GB) — only used by Stage 2.5
     but pre-fetching now means the H200 boot doesn't wait on the HF CDN.
  3. Stage-1 GRAPE artifacts (three small JSON files) from the strategy bucket
     → ``<volume_root>/artifacts/_shared/`` so ``_preflight`` skips Stage 1.
  4. Calibration tensor (``calib_<sha>.pt``) — tokenized once on CPU, cached at
     ``<volume_root>/artifacts/_shared/_calibration_cache/``. At 100× pool
     (~26M tokens) tokenization is a non-trivial CPU task that we *do not*
     want to pay GPU rental for.

The script is idempotent: any step that already produced its artifact on the
volume is skipped. Safe to re-run after a crash or to top up a partial volume.

Run with PYTHONPATH including ``max_quality/src``, e.g.::

    PYTHONPATH=max_quality/src python3 max_quality/scripts/phase0_prep.py \\
        --volume-root /mnt/volume \\
        --config max_quality/configs/qwen36_35b_a3b_30pct.yaml \\
        --token-cap 26214400

Optional: ``--no-teacher`` skips the FP8 download, ``--skip-calibration`` skips
the calibration tensor build (useful when you only want to refresh the model
cache).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("phase0_prep")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Step 1+2: model + teacher snapshot download
# ---------------------------------------------------------------------------


def _snapshot_download(repo: str, hub_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    log.info("Prefetching snapshot %s → %s", repo, hub_dir)
    snapshot_download(repo, cache_dir=str(hub_dir), allow_patterns=["*"])
    log.info("snapshot_download %s complete", repo)


# ---------------------------------------------------------------------------
# Step 3: Stage-1 artifact bucket pull
# ---------------------------------------------------------------------------


_STAGE1_FILES = (
    "_shared/stage1_blacklist.json",
    "_shared/stage1_budgets.json",
    "_shared/budget_decomposition.json",
)


def _pull_stage1_artifacts(bucket: str, artifacts_root: Path) -> None:
    """Mirror the bootstrap.sh logic: list the bucket, download the 3 files
    into ``<artifacts_root>/_shared/`` if any are missing."""
    shared_dir = artifacts_root / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        f for f in _STAGE1_FILES
        if not (artifacts_root / f).exists()
    ]
    if not missing:
        log.info("Stage-1 artifacts already on volume (%d files in %s)",
                 len(_STAGE1_FILES), shared_dir)
        return

    from huggingface_hub import HfApi
    api = HfApi()
    log.info("Listing bucket %s for Stage-1 artifacts", bucket)
    try:
        remote_files = {
            item.path for item in api.list_bucket_tree(bucket, recursive=True)
            if hasattr(item, "size")
        }
    except Exception as e:
        raise RuntimeError(
            f"Cannot list bucket {bucket}: {e}. "
            f"If this is the first run on a new bucket, Stage 1 must be run "
            f"once on a GPU instance to populate the artifacts; the volume-swap "
            f"workflow assumes a prior run has uploaded them."
        ) from e

    to_pull = []
    for f in missing:
        if f not in remote_files:
            raise RuntimeError(
                f"Stage-1 artifact {f!r} missing from bucket {bucket}. "
                f"Available files: {sorted(remote_files)[:10]}..."
            )
        local = artifacts_root / f
        local.parent.mkdir(parents=True, exist_ok=True)
        to_pull.append((f, str(local)))

    log.info("Downloading %d Stage-1 artifacts from %s", len(to_pull), bucket)
    api.download_bucket_files(bucket_id=bucket, files=to_pull)
    log.info("Stage-1 artifacts pulled into %s", shared_dir)


# ---------------------------------------------------------------------------
# Step 4: calibration tensor build (CPU tokenization)
# ---------------------------------------------------------------------------


def _build_calibration(
    *,
    config_path: Path,
    artifacts_root: Path,
    hub_dir: Path,
    model_repo: str,
    token_cap_override: int | None,
    seq_len_override: int | None,
) -> Path:
    """Tokenize the calibration corpus on CPU and cache it on the volume.

    Returns the path to the produced calib tensor file.
    """
    import yaml
    from transformers import AutoTokenizer

    from moe_compress.utils.calibration import (
        build_calibration_tensor,
        spec_from_config,
    )

    cfg = yaml.safe_load(config_path.read_text())
    cal_cfg = cfg.get("calibration")
    if cal_cfg is None:
        raise KeyError(
            f"{config_path} has no top-level 'calibration:' block — cannot "
            f"build the calibration tensor without it."
        )

    # Compute the row count the heal step will actually consume. Stage 2's
    # ``num_calibration_samples`` (under ``stage2_reap_ream``) drives the
    # ``num_sequences`` Stage 2 requests at runtime; mirror that here so the
    # cache key matches what Stage 2 will look up.
    s2_cfg = cfg.get("stage2_reap_ream", {})
    num_sequences = int(
        s2_cfg.get("num_calibration_samples")
        or cal_cfg.get("num_sequences")
    )
    if token_cap_override is not None:
        # token_cap is in tokens, num_sequences is in sequences of seq_len tokens.
        # Bump num_sequences so the captured pool can be filled at the new cap.
        seq_len = int(seq_len_override or cal_cfg.get("sequence_length", 2048))
        # Add ~15% headroom — heal capture stops at exact pool_size but the
        # forward needs enough source rows to actually capture that many.
        num_sequences = max(
            num_sequences,
            int(token_cap_override / seq_len * 1.15),
        )
        log.info(
            "Overriding num_sequences to %d (token_cap=%d / seq_len=%d × 1.15)",
            num_sequences, token_cap_override, seq_len,
        )

    spec = spec_from_config(
        cal_cfg,
        num_sequences_override=num_sequences,
        sequence_length_override=seq_len_override,
    )
    log.info(
        "Calibration spec: source=%s, num_sequences=%d, sequence_length=%d, seed=%d",
        spec.source, spec.num_sequences, spec.sequence_length, spec.seed,
    )

    cache_dir = artifacts_root / "_shared" / "_calibration_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer %s (from local HF cache)", model_repo)
    tokenizer = AutoTokenizer.from_pretrained(
        model_repo, cache_dir=str(hub_dir), trust_remote_code=True,
    )

    tensor = build_calibration_tensor(
        tokenizer, spec, cache_dir=cache_dir,
    )
    key = spec.cache_key(
        getattr(tokenizer, "name_or_path", None)
        or f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
    )
    cache_file = cache_dir / f"calib_{key}.pt"
    log.info(
        "Calibration tensor ready: %s (shape=%s, dtype=%s, %.2f MB)",
        cache_file, tuple(tensor.shape), tensor.dtype,
        cache_file.stat().st_size / (1024 ** 2),
    )
    return cache_file


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _du(path: Path) -> str:
    """Human-readable directory size."""
    if not path.exists():
        return "missing"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024 or unit == "TB":
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _summary(volume_root: Path, artifacts_root: Path, hub_dir: Path) -> None:
    print("\n" + "=" * 70)
    print(f"Phase 0 prep summary — volume at {volume_root}")
    print("=" * 70)
    print(f"HF cache: {hub_dir} ({_du(hub_dir)})")
    print(f"Artifacts: {artifacts_root} ({_du(artifacts_root)})")
    shared = artifacts_root / "_shared"
    if shared.exists():
        print(f"  Stage-1 artifacts:")
        for f in _STAGE1_FILES:
            local = artifacts_root / f
            mark = "✓" if local.exists() else "✗"
            print(f"    [{mark}] {f}")
        calib_cache = shared / "_calibration_cache"
        if calib_cache.exists():
            calibs = list(calib_cache.glob("calib_*.pt"))
            print(f"  Calibration cache: {len(calibs)} file(s), {_du(calib_cache)}")
            for c in calibs:
                print(f"    {c.name} ({c.stat().st_size / (1024**2):.1f} MB)")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--volume-root", type=Path, required=True,
                        help="Mounted persistent volume root (e.g. /mnt/volume)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to the run YAML config (for calibration spec).")
    parser.add_argument("--model-repo", default="Qwen/Qwen3.6-35B-A3B",
                        help="HF repo for the base model")
    parser.add_argument("--teacher-repo", default="Qwen/Qwen3.6-35B-A3B-FP8",
                        help="HF repo for the FP8 teacher (used in Stage 2.5)")
    parser.add_argument("--no-teacher", action="store_true",
                        help="Skip the FP8 teacher snapshot download")
    parser.add_argument("--artifacts-bucket", default="pirola/moe-strategy-35pct",
                        help="HF bucket holding the pre-computed Stage-1 _shared/ artifacts")
    parser.add_argument("--token-cap", type=int, default=None,
                        help="Override merge_heal_token_cap (e.g. 26214400 for 100× pool). "
                             "When set, num_sequences is auto-scaled to provide enough rows.")
    parser.add_argument("--sequence-length", type=int, default=None,
                        help="Override calibration sequence_length")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip the calibration-tensor build step (use to refresh model cache only)")
    args = parser.parse_args(argv)

    volume_root: Path = args.volume_root.resolve()
    if not volume_root.is_dir():
        log.error("Volume root %s does not exist or is not a directory", volume_root)
        return 2

    hub_dir = volume_root / "hf_cache" / "hub"
    artifacts_root = volume_root / "artifacts"
    hub_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    # HF cache vars — point everything at the volume so re-mounts find the cache.
    os.environ["HF_HOME"] = str(volume_root / "hf_cache")
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    log.info("HF_HOME=%s HF_HUB_CACHE=%s", os.environ["HF_HOME"], os.environ["HF_HUB_CACHE"])

    # 1. Base model snapshot
    _snapshot_download(args.model_repo, hub_dir)

    # 2. FP8 teacher snapshot
    if not args.no_teacher:
        _snapshot_download(args.teacher_repo, hub_dir)
    else:
        log.info("Skipping teacher snapshot (--no-teacher)")

    # 3. Stage-1 artifacts
    _pull_stage1_artifacts(args.artifacts_bucket, artifacts_root)

    # 4. Calibration tensor
    if not args.skip_calibration:
        _build_calibration(
            config_path=args.config.resolve(),
            artifacts_root=artifacts_root,
            hub_dir=hub_dir,
            model_repo=args.model_repo,
            token_cap_override=args.token_cap,
            seq_len_override=args.sequence_length,
        )
    else:
        log.info("Skipping calibration tensor build (--skip-calibration)")

    _summary(volume_root, artifacts_root, hub_dir)
    log.info("Phase 0 prep complete. Volume %s is ready for the RTX 6000 Pro to attach.", volume_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
