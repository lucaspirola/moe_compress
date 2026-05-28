"""F-S3-1: Stage 3 originals.pt must be written via atomic_torch_save +
write_manifest_last, and Stage 4 must read+validate the manifest before
opening the .pt.

We don't run the full Stage 3 orchestrator (too heavy); instead we
exercise the same write+read protocol the orchestrator uses end-to-end:

  1. atomic_torch_save(originals, _orig_path)
  2. write_manifest_last(_orig_path, _orig_manifest_path, schema_version=1)

then simulate a Stage-4-style read with truncation injected between
write and read.
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.utils.atomic_io import (
    ManifestMismatchError,
    atomic_torch_save,
    read_and_validate_manifest,
    write_manifest_last,
)


def _fake_originals():
    # Mimics dict[(layer, name)] -> tensor that Stage 3 snapshots.
    return {
        (0, "gate_proj"): torch.arange(16, dtype=torch.float32).reshape(4, 4),
        (0, "down_proj"): torch.arange(16, dtype=torch.float32).reshape(4, 4),
        (1, "gate_proj"): torch.zeros(4, 4),
    }


def test_stage3_originals_manifest_roundtrip(tmp_path):
    orig_path = tmp_path / "_stage3_original_weights.pt"
    manifest_path = tmp_path / "_stage3_original_weights.pt.MANIFEST.json"

    originals = _fake_originals()
    atomic_torch_save(originals, orig_path)
    write_manifest_last(
        orig_path, manifest_path, schema_version=1,
        extra_meta={"n_matrices": len(originals)},
    )

    # Stage 4 read path: validate manifest first, then torch.load.
    out = read_and_validate_manifest(orig_path, manifest_path, expected_schema_version=1)
    assert out["payload_name"] == orig_path.name
    assert out["extra"]["n_matrices"] == 3

    loaded = torch.load(orig_path, weights_only=False)
    assert set(loaded.keys()) == set(originals.keys())


def test_stage3_originals_torn_payload_fails_loudly(tmp_path):
    """The audit's worst-case: a kill mid-write leaves a TRUNCATED .pt at
    the final path. Without manifest validation, Stage 4 might
    .get(..., 0)-fallback and produce silently wrong residuals. With the
    fix, the manifest's size_bytes mismatch raises ManifestMismatchError
    BEFORE Stage 4 ever touches the corrupt file."""
    orig_path = tmp_path / "_stage3_original_weights.pt"
    manifest_path = tmp_path / "_stage3_original_weights.pt.MANIFEST.json"
    atomic_torch_save(_fake_originals(), orig_path)
    write_manifest_last(orig_path, manifest_path, schema_version=1)

    # Truncate the payload to simulate SIGKILL mid-write recovered by
    # the next pod (the manifest from the previous successful run still
    # exists, but the .pt is now half-size).
    real_size = orig_path.stat().st_size
    with open(orig_path, "r+b") as f:
        f.truncate(real_size // 2)

    with pytest.raises(ManifestMismatchError, match="size"):
        read_and_validate_manifest(orig_path, manifest_path, expected_schema_version=1)


def test_stage3_originals_missing_manifest_fails_loudly(tmp_path):
    """Kill BETWEEN atomic_torch_save and write_manifest_last leaves
    the .pt without its manifest. Reader treats this as torn."""
    orig_path = tmp_path / "_stage3_original_weights.pt"
    manifest_path = tmp_path / "_stage3_original_weights.pt.MANIFEST.json"
    atomic_torch_save(_fake_originals(), orig_path)
    # NO write_manifest_last call — simulating kill in between.

    with pytest.raises(ManifestMismatchError, match="missing"):
        read_and_validate_manifest(orig_path, manifest_path, expected_schema_version=1)


def test_stage3_originals_schema_bump_invalidates(tmp_path):
    """A schema_version bump in Stage 3 must invalidate stale manifests."""
    orig_path = tmp_path / "_stage3_original_weights.pt"
    manifest_path = tmp_path / "_stage3_original_weights.pt.MANIFEST.json"
    atomic_torch_save(_fake_originals(), orig_path)
    write_manifest_last(orig_path, manifest_path, schema_version=1)

    # Stage 4 from a future revision expects schema_version=2.
    with pytest.raises(ManifestMismatchError, match="schema_version"):
        read_and_validate_manifest(orig_path, manifest_path, expected_schema_version=2)


def test_stage3_originals_no_dotnpz_tmp_leftovers(tmp_path):
    """Belt-and-braces: writer leaves no .tmp orphans on success."""
    orig_path = tmp_path / "_stage3_original_weights.pt"
    manifest_path = tmp_path / "_stage3_original_weights.pt.MANIFEST.json"
    atomic_torch_save(_fake_originals(), orig_path)
    write_manifest_last(orig_path, manifest_path, schema_version=1)
    assert not list(tmp_path.glob("*.tmp"))


def test_stage3_originals_manifest_in_hub_upload_lists():
    """HIGH-2: F-S3-1 manifest must be in BOTH Hub upload lists, AFTER
    the .pt, so Pattern O's manifest-LAST invariant survives the Hub
    durability boundary too. A partial upload that drops the manifest
    leaves Stage 4 with a torn-write signature (missing manifest), which
    fails loudly instead of silently consuming a half-uploaded payload.
    """
    # _STAGE_LAYOUT (used by per-stage uploader).
    from moe_compress.utils.hub_upload import _STAGE_LAYOUT
    _stage3_subdir, stage3_sidecars = _STAGE_LAYOUT[3]
    assert "_stage3_original_weights.pt" in stage3_sidecars
    assert "_stage3_original_weights.pt.MANIFEST.json" in stage3_sidecars
    # Manifest must come AFTER the payload in the list (manifest-LAST).
    pt_idx = stage3_sidecars.index("_stage3_original_weights.pt")
    manifest_idx = stage3_sidecars.index("_stage3_original_weights.pt.MANIFEST.json")
    assert manifest_idx > pt_idx, (
        f"Pattern O violation: manifest at index {manifest_idx} must "
        f"come AFTER payload at index {pt_idx} in _STAGE_LAYOUT[3]"
    )

    # entrypoint.aux_files (used by job-exit aux uploader). Parse the
    # source so we don't import the script (it pulls in heavy deps).
    from pathlib import Path
    src = Path(__file__).parent.parent / "hf_jobs" / "entrypoint.py"
    text = src.read_text()
    assert '"_stage3_original_weights.pt"' in text, "aux_files missing .pt"
    assert '"_stage3_original_weights.pt.MANIFEST.json"' in text, (
        "aux_files missing MANIFEST.json"
    )
    # Manifest must appear AFTER the .pt in the file (manifest-LAST).
    pt_offset = text.index('"_stage3_original_weights.pt"')
    mf_offset = text.index('"_stage3_original_weights.pt.MANIFEST.json"')
    assert mf_offset > pt_offset, (
        "Pattern O violation: aux_files lists MANIFEST.json before .pt"
    )
