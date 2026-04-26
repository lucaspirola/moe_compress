"""Per-stage Hub upload helper — synchronous uploads.

Each long-running stage (2-5) writes its checkpoint to ``artifacts/stage{N}_*/``
on the bucket. The bucket is NOT durable on SIGKILL (HF Jobs FUSE volume mounts
inherit hf-mount streaming-write semantics — see ``docs/huggingface_jobs_and_buckets.md``).
The Hub commit is the only durability boundary.

Each call blocks until the upload commits — Stage N is fully durable on Hub
before Stage N+1 begins. Pays ~$1/stage of GPU-idle (a100-large × ~25 min upload
for 50 GB) but eliminates the partial-upload-on-hard-kill window. The model
stays loaded across stages in the same job, so we are not paying for re-load.

``wait_for_pending_uploads`` remains as a no-op stub for callers that want
explicit pipeline-exit drain semantics; it does nothing in synchronous mode.

Layout in the per-stage repo matches what ``entrypoint.py:_restore_prior_checkpoint``
expects:

    <repo_root>/                      # stage_dir contents flat at root
        config.json
        model.safetensors[.index.json]
        compressed_metadata.json
        ...
    <repo_root>/artifacts/<sidecar>   # per-stage sidecars hoisted on download
        _stage2_input_covariance.pt
        ...
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


# Stage N → (stage subdir name, list of sidecar filenames in artifacts_dir).
# Only stages 2-5 produce a heavy checkpoint dir worth uploading per-stage;
# 0/1/6 produce small JSONs covered by the entrypoint's job-exit aux upload.
_STAGE_LAYOUT: dict[int, tuple[str, list[str]]] = {
    2: ("stage2_pruned", ["_stage2_input_covariance.pt", "stage2_layer_mse.json"]),
    3: ("stage3_svd",    ["_stage3_original_weights.pt"]),
    4: ("stage4_eora",   []),
    5: ("stage5_final",  []),
}


def upload_stage_to_hub(
    stage_idx: int,
    artifacts_dir: Path,
    *,
    repo_base: str,
) -> str | None:
    """Synchronously upload the ``stage{N}`` checkpoint + sidecars to Hub.

    Blocks until every commit returns 200, so the stage is fully durable on
    Hub before this function returns. Returns the repo_id, or None if no
    upload was performed (no stage dir, or huggingface_hub not installed).
    """
    layout = _STAGE_LAYOUT.get(stage_idx)
    if layout is None:
        return None
    subdir, sidecars = layout
    stage_dir = Path(artifacts_dir) / subdir
    if not stage_dir.exists():
        log.warning("Stage %d Hub upload skipped: %s does not exist", stage_idx, stage_dir)
        return None

    repo_id = f"{repo_base}-stage{stage_idx}"

    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.warning("Stage %d Hub upload skipped: huggingface_hub not installed", stage_idx)
        return None

    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as exc:                     # noqa: BLE001
        log.warning("create_repo(%s) failed: %s — continuing with existing", repo_id, exc)

    log.info("Uploading Stage %d checkpoint → https://huggingface.co/%s", stage_idx, repo_id)
    api.upload_large_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="model",
    )

    # Sidecars under artifacts/ in the repo so PRIOR_STAGE_REPO restore hoists them.
    for name in sidecars:
        p = Path(artifacts_dir) / name
        if not p.exists():
            continue
        log.info("Uploading sidecar %s (%.1f GB) → %s", name, p.stat().st_size / 1e9, repo_id)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f"artifacts/{name}",
            repo_id=repo_id,
            repo_type="model",
        )

    log.info("Stage %d durable on Hub: %s", stage_idx, repo_id)
    return repo_id


def wait_for_pending_uploads() -> None:
    """No-op in synchronous mode.

    Kept as a callable so ``run_pipeline.py`` can call it at every exit path
    without branching. In synchronous mode, ``upload_stage_to_hub`` already
    blocked until durable, so there is nothing to wait for here.
    """
    return


def hub_repo_base_from_env() -> str | None:
    """Read the per-stage repo base name from env. Set by ``hf_jobs/entrypoint.py``."""
    base = os.environ.get("PIPELINE_HUB_RESULT_REPO_BASE", "").strip()
    return base or None
