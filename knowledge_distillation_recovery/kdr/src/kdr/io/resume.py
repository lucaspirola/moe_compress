"""Partial-checkpoint discovery + HF Hub sync for resume + final upload.

Three concerns:

  * :func:`find_latest_partial` — local-disk scan for ``kdr_{mode}_partial_step*/``
    dirs whose ``_SAVE_COMPLETE`` sentinel is present (LLR-0027 / LLR-0029).
  * :func:`upload_partial_to_hub` / :func:`upload_final_to_hub` — synchronous
    HF Hub upload after a save (LLR-0030 final, LLR-0033 partial).
  * :func:`find_latest_partial_on_hub` / :func:`download_partial_from_hub` —
    bootstrap-time resume query (LLR-0033).

The HF Hub paths are intentionally mirror-symmetric: the partial dir's NAME
is the path-in-repo (``kdr_da_qad_partial_step100/``), so a hub repo holds
multiple partials side by side. The final-checkpoint repo is separate
(LLR-0030) so partial pruning can be aggressive without touching the final.
"""

# REQ: LLR-0027
# REQ: LLR-0029
# REQ: LLR-0030
# REQ: LLR-0033

from __future__ import annotations

import logging
from pathlib import Path

from ..modes import Mode
from .save import SAVE_COMPLETE_SENTINEL

log = logging.getLogger(__name__)


def find_latest_partial(
    artifacts_dir: Path, mode: Mode
) -> tuple[Path, int] | None:
    """Find the highest-step ``kdr_{mode}_partial_step{N}/`` whose
    ``_SAVE_COMPLETE`` sentinel is present.

    Returns ``(partial_dir, step)`` or ``None`` if no valid partial exists.
    Dirs lacking the sentinel are skipped with a warning (incomplete writes
    must NOT be picked up as resume seeds — they would silently load
    truncated weights).
    """
    if not artifacts_dir.exists():
        return None

    # Direct format string rather than `partial_dir_name(mode, 0).replace(...)`
    # — keeps the glob pattern decoupled from the canonical-name format so a
    # future change to `partial_dir_name` (e.g. zero-padded step indices)
    # doesn't silently regress this glob.
    pattern = f"kdr_{mode}_partial_step*"
    candidates: list[tuple[int, Path]] = []
    for p in artifacts_dir.glob(pattern):
        if not p.is_dir() or p.name.endswith(".tmp"):
            continue
        if not (p / SAVE_COMPLETE_SENTINEL).exists():
            log.warning(
                "Partial %s missing %s — skipping (incomplete write).",
                p.name,
                SAVE_COMPLETE_SENTINEL,
            )
            continue
        try:
            step = int(p.name.split("step")[-1])
        except ValueError:
            log.warning("Could not parse step from %s — skipping.", p.name)
            continue
        candidates.append((step, p))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_step, best_path = candidates[0]
    log.info("Found partial checkpoint at step=%d: %s", best_step, best_path)
    return best_path, best_step


# ─────────────────────────────────────────────────────────────────────────────
# HF Hub uploads (LLR-0030, LLR-0033)
# ─────────────────────────────────────────────────────────────────────────────


def upload_partial_to_hub(
    partial_dir: Path, repo_id: str, *, create_repo: bool = True
) -> str:
    """Synchronous upload of a single partial dir to ``repo_id`` on HF Hub.

    The partial's path-in-repo equals its directory name
    (``kdr_da_qad_partial_step100/...``), so the hub repo accumulates
    multiple partials side by side. The training loop's caller must own
    the pruning policy (oldest-N partials kept).

    Args:
        partial_dir: local path to the partial dir to upload. Must contain
            the ``_SAVE_COMPLETE`` sentinel; a partial without the sentinel
            represents a half-written save and SHOULD NOT be uploaded.
        repo_id: target HF Hub repo (e.g. ``"pirola/kdr-partials-{run_id}"``).
        create_repo: if True (default), create the repo on upload if it
            doesn't already exist (private). False for tests.

    Returns:
        The HF Hub model-page URL.

    Raises:
        ValueError: if ``partial_dir`` lacks the ``_SAVE_COMPLETE`` sentinel.
    """
    if not (partial_dir / SAVE_COMPLETE_SENTINEL).exists():
        raise ValueError(
            f"upload_partial_to_hub: {partial_dir} lacks "
            f"{SAVE_COMPLETE_SENTINEL!r} — refusing to upload an incomplete "
            "partial (would corrupt resume state)."
        )

    from huggingface_hub import HfApi

    api = HfApi()
    if create_repo:
        api.create_repo(repo_id, exist_ok=True, private=True, repo_type="model")
    api.upload_folder(
        folder_path=str(partial_dir),
        path_in_repo=partial_dir.name,
        repo_id=repo_id,
        repo_type="model",
    )
    url = f"https://huggingface.co/{repo_id}"
    log.info("Uploaded partial %s to %s", partial_dir.name, url)
    return url


# REQ: LLR-0030
def upload_final_to_hub(
    final_dir: Path, repo_id: str, *, create_repo: bool = True
) -> str:
    """Synchronous upload of the final compressed-tensors artifact.

    The final repo is distinct from the partials repo so partial-pruning
    policy can be aggressive without touching the final artifact.

    LLR-0030 AC: "Upload happens after the last `_save` returns
    successfully." Mirrors :func:`upload_partial_to_hub`'s sentinel-guard:
    a final dir without ``_SAVE_COMPLETE`` represents a crash between
    the atomic rename and the sentinel write — uploading it would publish
    a possibly-truncated artifact under a stable URL.

    Args:
        final_dir: local final-artifact dir (the output of ``save_kdr_artifact``
            or the bf16 ``save_partial(..., partial=False)``).
        repo_id: target repo (e.g. ``"pirola/kdr-recovered-{run_id}"``).
        create_repo: if True (default), create the repo on upload.

    Returns:
        The HF Hub model-page URL.

    Raises:
        ValueError: if ``final_dir`` lacks the ``_SAVE_COMPLETE`` sentinel.
    """
    if not (final_dir / SAVE_COMPLETE_SENTINEL).exists():
        raise ValueError(
            f"upload_final_to_hub: {final_dir} lacks "
            f"{SAVE_COMPLETE_SENTINEL!r} — refusing to upload an incomplete "
            "artifact (would publish a truncated checkpoint under a stable URL)."
        )

    from huggingface_hub import HfApi

    api = HfApi()
    if create_repo:
        api.create_repo(repo_id, exist_ok=True, private=True, repo_type="model")
    api.upload_folder(
        folder_path=str(final_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    url = f"https://huggingface.co/{repo_id}"
    log.info("Uploaded final artifact to %s", url)
    return url


# REQ: LLR-0033
def find_latest_partial_on_hub(repo_id: str) -> tuple[str, int] | None:
    """Query a partials repo, find the highest-step partial whose
    ``_SAVE_COMPLETE`` sentinel is present in the hub.

    Returns ``(partial_dir_name, step)`` or ``None`` if the repo doesn't
    exist, is unreachable, OR contains no valid partials. Per LLR-0033
    AC: a missing repo is not an error — the trainer starts from step 0.
    Network errors during the listing iteration are also swallowed
    (returning ``None``) so transient hub issues don't crash the bootstrap.

    Uses ``list_repo_tree`` (per LLR-0033's literal text) with
    ``recursive=True`` to walk every file path in one paginated stream.

    Args:
        repo_id: HF Hub repo ID. Need not exist; absent → ``None``.

    Returns:
        ``(dir_name, step)`` or ``None``.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    api = HfApi()
    candidates: dict[str, int] = {}
    try:
        for entry in api.list_repo_tree(repo_id, repo_type="model", recursive=True):
            path = getattr(entry, "path", None)
            if not isinstance(path, str):
                continue
            parts = path.split("/")
            if len(parts) < 2:
                continue
            dir_name = parts[0]
            if not dir_name.startswith("kdr_") or "_partial_step" not in dir_name:
                continue
            if parts[-1] != SAVE_COMPLETE_SENTINEL:
                continue
            try:
                step = int(dir_name.rsplit("step", 1)[-1])
            except ValueError:
                continue
            candidates[dir_name] = step
    except RepositoryNotFoundError:
        log.info("Partials repo %s does not exist — starting from step 0", repo_id)
        return None
    except HfHubHTTPError as e:
        # Network blip / 5xx mid-iteration: prefer "start fresh" over crashing
        # the bootstrap. The trainer will start from step 0; the next save
        # will re-establish the partials repo from this run's outputs.
        log.warning(
            "Partials repo %s listing failed (%s) — starting from step 0", repo_id, e
        )
        return None

    if not candidates:
        log.info("Partials repo %s has no valid partials", repo_id)
        return None
    best_dir, best_step = max(candidates.items(), key=lambda kv: kv[1])
    log.info(
        "Latest partial on hub: %s (step=%d)", best_dir, best_step
    )
    return best_dir, best_step


# REQ: LLR-0033
def download_partial_from_hub(
    repo_id: str, partial_dir_name: str, target: Path
) -> Path:
    """Snapshot-download a single partial subdir from the hub into ``target``.

    Args:
        repo_id: source repo on HF Hub.
        partial_dir_name: the partial's directory name (path-in-repo).
        target: local destination directory; created if absent.

    Returns:
        The local path to the downloaded partial dir.
    """
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{partial_dir_name}/*"],
        local_dir=str(target),
    )
    out = target / partial_dir_name
    log.info("Downloaded partial %s from %s to %s", partial_dir_name, repo_id, out)
    return out
