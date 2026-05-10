"""Atomic save path for kdr — partial (Phase 3b) and final (Phase 4).

`save_partial` is the Phase 3b / BF16-mode path: vanilla `save_pretrained`
+ tokenizer + optional `compressed_metadata.json` passthrough +
`_SAVE_COMPLETE` sentinel. Atomic via `.tmp` directory + `os.rename`.

`save_kdr_artifact` is the Phase 4 / `da_qad`-mode path: same shape but the
weight serialiser is the compressed-tensors converter from the active
`QuantBackend`. Still stubbed — Phase 4 lands the body.

Atomic-save invariant (LLR-0029): inside a partial dir, `_SAVE_COMPLETE` is
the LAST file written and is empty (zero bytes). Its presence is the
post-rename guarantee that every other file in the dir was committed before
the rename. Resume logic SHALL ignore dirs lacking it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import torch.nn as nn
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from ..config import QuantBlock
from ..modes import Mode

log = logging.getLogger(__name__)

# REQ: LLR-0029
SAVE_COMPLETE_SENTINEL = "_SAVE_COMPLETE"
"""Empty file written LAST inside a saved partial dir; its presence is the
post-atomic-rename invariant that the dir is fully committed."""

COMPRESSED_METADATA_FILENAME = "compressed_metadata.json"
"""HLR-0005 / LLR-0019: preserved verbatim from the input student if present."""


# REQ: LLR-0027
def partial_dir_name(mode: Mode, step: int) -> str:
    """`kdr_{mode}_partial_step{N}` — embeds both mode and step.

    Mode-prefixing avoids cross-mode resume contamination (a partial saved
    by `bf16` cannot be picked up as a resume seed for `da_qad` which
    has different module wrappers).
    """
    return f"kdr_{mode}_partial_step{step}"


# REQ: LLR-0027
def save_partial(
    student: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    *,
    artifacts_dir: Path,
    mode: Mode,
    step: int,
    source_metadata_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    partial: bool = True,
) -> Path:
    """Atomic save of `student` to a partial (or final) dir.

    Layout:
      `<artifacts_dir>/kdr_{mode}_partial_step{step}/` (partial=True), or
      `<artifacts_dir>/kdr_{mode}_recovered/` (partial=False).

    Steps (all rank-0):
      1. Build under `.tmp/` so a half-written dir never appears at the
         final path.
      2. `unwrapped.save_pretrained(tmp_dir, state_dict=…, safe_serialization=True)`
         — `state_dict` comes from `accelerator.get_state_dict(student)`,
         which under ZeRO-3 streams the consolidated tensors INTO CPU memory
         on rank 0 only (other ranks return `{}`).
      3. `tokenizer.save_pretrained(tmp_dir)`.
      4. Copy `compressed_metadata.json` verbatim from `source_metadata_path`
         if it exists (HLR-0005).
      5. Write `extra_metadata` (if provided) into a sidecar
         `kdr_run_metadata.json`.
      6. Atomically rename `tmp_dir` → final dir.
      7. Write empty `_SAVE_COMPLETE` LAST so its presence post-rename is
         the integrity sentinel (LLR-0029).

    All ranks call (`get_state_dict` is collective under DS); only rank 0
    actually writes.
    """
    accelerator.wait_for_everyone()

    out_name = partial_dir_name(mode, step) if partial else f"kdr_{mode}_recovered"
    out_dir = artifacts_dir / out_name
    tmp_dir = out_dir.parent / f"{out_dir.name}.tmp"

    # Collective: every rank participates in the consolidation.
    state_dict = accelerator.get_state_dict(student)
    unwrapped = accelerator.unwrap_model(student)

    if accelerator.is_main_process:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Clean any stale `.tmp` from a previous failed save.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        unwrapped.save_pretrained(
            tmp_dir, state_dict=state_dict, safe_serialization=True
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp_dir)

        # REQ: LLR-0019
        if source_metadata_path is not None and source_metadata_path.exists():
            shutil.copyfile(
                source_metadata_path, tmp_dir / COMPRESSED_METADATA_FILENAME
            )

        if extra_metadata is not None:
            (tmp_dir / "kdr_run_metadata.json").write_text(
                json.dumps(extra_metadata, indent=2, sort_keys=True)
            )

        # Atomic rename — replaces an existing final dir if present.
        _atomic_replace_dir(tmp_dir, out_dir)

        # REQ: LLR-0029
        # Sentinel written LAST, INTO the renamed final dir, EMPTY (zero bytes).
        # Mtime ordering guarantees every other file's mtime ≤ sentinel's.
        sentinel = out_dir / SAVE_COMPLETE_SENTINEL
        sentinel.touch(exist_ok=False)

        log.info(
            "Saved %s checkpoint to %s (step=%d)",
            "PARTIAL" if partial else "FINAL",
            out_dir,
            step,
        )

    accelerator.wait_for_everyone()
    return out_dir


# REQ: LLR-0018
def save_kdr_artifact(
    model: nn.Module,
    output_dir: Path,
    *,
    quant_block: QuantBlock | None,
    input_metadata: dict[str, object] | None = None,
) -> None:
    """Compressed-tensors final save — Phase 4.

    Phase 3b leaves this stubbed; the `da_qad`-mode final save lands when the
    `QuantBackend.save` path is implemented. The `bf16`-mode final save is
    handled by `save_partial(..., partial=False)`.
    """
    raise NotImplementedError("Phase 4: save_kdr_artifact (compressed-tensors)")


# ---------------------------------------------------------------------------
# Atomic helpers
# ---------------------------------------------------------------------------


def _atomic_replace_dir(src: Path, dst: Path) -> None:
    """Atomically replace `dst` with `src`. Both must be on the same FS.

    POSIX `rename(2)` (and Python's `os.rename`) refuses to replace a
    non-empty directory. We work around by moving the existing `dst` aside
    first; on rename failure, restore it.
    """
    if dst.exists():
        backup = dst.with_name(dst.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        os.rename(dst, backup)
        try:
            os.rename(src, dst)
        except Exception:
            # Restore the backup on any failure.
            os.rename(backup, dst)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.rename(src, dst)
