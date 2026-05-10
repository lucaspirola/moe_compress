"""Partial-checkpoint discovery for resume.

`find_latest_partial(artifacts_dir, mode)` scans for `kdr_{mode}_partial_step*/`
dirs whose `_SAVE_COMPLETE` sentinel is present and returns the highest-step
match (LLR-0027 / LLR-0029).

`upload_partial_to_hub` is the Phase 6 vast.ai bootstrap path — synchronous
HF Hub upload after a partial save. Still stubbed; only meaningful inside
the docker bootstrap flow.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..modes import Mode
from .save import SAVE_COMPLETE_SENTINEL

log = logging.getLogger(__name__)


# REQ: LLR-0027
# REQ: LLR-0029
def find_latest_partial(
    artifacts_dir: Path, mode: Mode
) -> tuple[Path, int] | None:
    """Find the highest-step `kdr_{mode}_partial_step{N}/` whose
    `_SAVE_COMPLETE` sentinel is present.

    Returns `(partial_dir, step)` or `None` if no valid partial exists.
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


def upload_partial_to_hub(partial_dir: Path, repo_id: str) -> str:
    """Synchronous upload to a private HF Hub model repo. Returns the repo URL.

    Phase 6: only meaningful inside the vast.ai bootstrap flow. Local-disk
    runs use `find_latest_partial` alone. Phase 3b leaves this stubbed.
    """
    raise NotImplementedError("Phase 6: upload_partial_to_hub")
