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
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:                              # noqa: BLE001
    HfApi = None                                 # type: ignore[assignment]
    HfHubHTTPError = Exception                   # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


# Stage key → (stage subdir name, list of sidecar filenames in artifacts_dir).
# Only stages 2-5 produce a heavy checkpoint dir worth uploading per-stage;
# 0/1/6 produce small JSONs covered by the entrypoint's job-exit aux upload.
# Stage 2.5 (router-KD post-merge) writes ``stage2p5_final/`` via
# ``stage5_router_kd.run(stage_key="stage2p5")`` — verified at
# stage5_router_kd.py:434 (``out_dir = artifacts_dir / f"{stage_key}_final"``).
# No new sidecars are produced beyond what Stage 2 already uploaded.
_STAGE_LAYOUT: dict[int | str, tuple[str, list[str]]] = {
    2:     ("stage2_pruned",  ["_stage2_input_covariance.pt", "stage2_layer_mse.json"]),
    "2p5": ("stage2p5_final", []),
    3:     ("stage3_svd",     ["_stage3_original_weights.pt"]),
    4:     ("stage4_eora",    []),
    5:     ("stage5_final",   []),
}


def _format_size(size_bytes: int) -> str:
    """Auto-scale a byte count to GB / MB / KB so kilobyte sidecars don't print as 0.0 GB."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1e9:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1e6:.1f} MB"
    return f"{size_bytes / 1e3:.1f} KB"


def _retry_upload(label: str, fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` with 3-attempt exponential backoff (2s, 4s).

    Retries transient HfHubHTTPError / ConnectionError / OSError. Re-raises
    on the final failure with a clear log line so the pipeline fails loud
    rather than recording a half-uploaded stage as durable.
    """
    last_exc: BaseException | None = None
    for attempt in range(3):
        try:
            return fn(*args, **kwargs)
        except (HfHubHTTPError, ConnectionError, OSError) as exc:  # noqa: PERF203
            last_exc = exc
            if attempt < 2:
                delay = 2 ** (attempt + 1)
                log.warning(
                    "%s failed (attempt %d/3): %s — retrying in %ds",
                    label, attempt + 1, exc, delay,
                )
                time.sleep(delay)
                continue
            log.error("%s failed after 3 attempts: %s", label, exc)
            raise
    # Defensive: should not be reachable because the final attempt either
    # returns or re-raises. Re-raise the last seen exception just in case.
    assert last_exc is not None
    raise last_exc


def upload_stage_to_hub(
    stage_idx: int | str,
    artifacts_dir: Path,
    *,
    repo_base: str | None,
) -> str | None:
    """Synchronously upload the ``stage{N}`` checkpoint + sidecars to Hub.

    Blocks until every commit returns 200, so the stage is fully durable on
    Hub before this function returns. Returns the repo_id, or None if no
    upload was performed (no stage dir, repo_base unset, or
    huggingface_hub not installed).
    """
    if repo_base is None:
        log.warning(
            "upload_stage_to_hub: repo_base is None (env var not set?) — skipping upload",
        )
        return None
    layout = _STAGE_LAYOUT.get(stage_idx)
    if layout is None:
        log.error(
            "upload_stage_to_hub: unknown stage %r — no upload performed", stage_idx,
        )
        return None
    if HfApi is None:
        log.warning(
            "Stage %s Hub upload skipped: huggingface_hub not installed", stage_idx,
        )
        return None
    subdir, sidecars = layout
    stage_dir = Path(artifacts_dir) / subdir
    if not stage_dir.exists():
        log.warning("Stage %s Hub upload skipped: %s does not exist", stage_idx, stage_dir)
        return None

    repo_id = f"{repo_base}-stage{stage_idx}"

    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as exc:                     # noqa: BLE001
        log.warning("create_repo(%s) failed: %s — continuing with existing", repo_id, exc)

    log.info("Uploading Stage %s checkpoint → https://huggingface.co/%s", stage_idx, repo_id)
    _retry_upload(
        f"upload_large_folder({stage_dir})",
        api.upload_large_folder,
        folder_path=stage_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    # Sidecars under artifacts/ in the repo so PRIOR_STAGE_REPO restore hoists them.
    for name in sidecars:
        p = Path(artifacts_dir) / name
        if not p.exists():
            continue
        log.info(
            "Uploading sidecar %s (%s) → %s",
            name, _format_size(p.stat().st_size), repo_id,
        )
        _retry_upload(
            f"upload_file({name})",
            api.upload_file,
            path_or_fileobj=p,
            path_in_repo=f"artifacts/{name}",
            repo_id=repo_id,
            repo_type="model",
        )

    log.info("Stage %s durable on Hub: %s", stage_idx, repo_id)
    return repo_id


def wait_for_pending_uploads() -> None:
    """no-op; placeholder for future async upload mode.

    Kept as a callable so ``run_pipeline.py`` can call it at every exit path
    without branching. In the current synchronous mode,
    ``upload_stage_to_hub`` already blocked until durable, so there is
    nothing to wait for here.
    """
    return


def hub_repo_base_from_env() -> str | None:
    """Read the per-stage repo base name from env. Set by ``hf_jobs/entrypoint.py``."""
    base = os.environ.get("PIPELINE_HUB_RESULT_REPO_BASE", "").strip()
    return base or None
