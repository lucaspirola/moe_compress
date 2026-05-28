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
    manifest_path = tmp_path / "_stage3_original_weights.MANIFEST.json"

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
    manifest_path = tmp_path / "_stage3_original_weights.MANIFEST.json"
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
    manifest_path = tmp_path / "_stage3_original_weights.MANIFEST.json"
    atomic_torch_save(_fake_originals(), orig_path)
    # NO write_manifest_last call — simulating kill in between.

    with pytest.raises(ManifestMismatchError, match="missing"):
        read_and_validate_manifest(orig_path, manifest_path, expected_schema_version=1)


def test_stage3_originals_schema_bump_invalidates(tmp_path):
    """A schema_version bump in Stage 3 must invalidate stale manifests."""
    orig_path = tmp_path / "_stage3_original_weights.pt"
    manifest_path = tmp_path / "_stage3_original_weights.MANIFEST.json"
    atomic_torch_save(_fake_originals(), orig_path)
    write_manifest_last(orig_path, manifest_path, schema_version=1)

    # Stage 4 from a future revision expects schema_version=2.
    with pytest.raises(ManifestMismatchError, match="schema_version"):
        read_and_validate_manifest(orig_path, manifest_path, expected_schema_version=2)


def test_stage3_originals_no_dotnpz_tmp_leftovers(tmp_path):
    """Belt-and-braces: writer leaves no .tmp orphans on success."""
    orig_path = tmp_path / "_stage3_original_weights.pt"
    manifest_path = tmp_path / "_stage3_original_weights.MANIFEST.json"
    atomic_torch_save(_fake_originals(), orig_path)
    write_manifest_last(orig_path, manifest_path, schema_version=1)
    assert not list(tmp_path.glob("*.tmp"))
