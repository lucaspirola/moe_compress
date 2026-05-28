"""F-RK-1: Stage 5 teacher_logits.pt must be written via atomic_torch_save
+ write_manifest_last, and the Stage 5 reader must validate the manifest
before opening the .pt with mmap=True.

Same shape of test as test_stage3_originals_manifest — we test the
write/read protocol directly, not the full Stage 5 stack.
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


def _fake_cache_payload(num_samples=4, seq_len=8, vocab=32):
    # Stage 5's teacher cache: a single big logits tensor + metadata.
    return {
        "logits": torch.zeros(num_samples * seq_len, vocab, dtype=torch.bfloat16),
        "num_samples": num_samples,
        "sequence_length": seq_len,
        "batch_size": 2,
        "model": "fake",
        "calibration_seed_offset": 5,
        "format_version": 1,
    }


def test_stage5_teacher_logits_manifest_roundtrip(tmp_path):
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, _fake_cache_payload())
    write_manifest_last(
        cache_path, manifest_path, schema_version=1,
        extra_meta={"artifact": "stage5_teacher_logits"},
    )

    out = read_and_validate_manifest(cache_path, manifest_path, expected_schema_version=1)
    assert out["payload_name"] == cache_path.name
    assert out["extra"]["artifact"] == "stage5_teacher_logits"

    # Reader can then mmap-load.
    loaded = torch.load(cache_path, mmap=True, weights_only=False)
    assert int(loaded["format_version"]) == 1
    assert loaded["logits"].shape == (4 * 8, 32)


def test_stage5_teacher_logits_torn_payload_caught(tmp_path):
    """Audit's worst-case: HF Jobs pod eviction mid-write leaves a truncated
    ~30 GB .pt. mmap=True opens cleanly and silently reads garbage past
    EOF → degenerate KD. With manifest validation, we catch this BEFORE
    mmap."""
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, _fake_cache_payload())
    write_manifest_last(cache_path, manifest_path, schema_version=1)

    real_size = cache_path.stat().st_size
    with open(cache_path, "r+b") as f:
        f.truncate(real_size // 3)

    with pytest.raises(ManifestMismatchError, match="size"):
        read_and_validate_manifest(cache_path, manifest_path, expected_schema_version=1)


def test_stage5_teacher_logits_missing_manifest_fails(tmp_path):
    """Kill between atomic_torch_save and write_manifest_last."""
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, _fake_cache_payload())

    with pytest.raises(ManifestMismatchError, match="missing"):
        read_and_validate_manifest(cache_path, manifest_path, expected_schema_version=1)


def test_stage5_teacher_logits_size_mismatch_after_replacement(tmp_path):
    """Pod B resumes from pod A's interrupted state — the manifest from
    pod A's previous successful round is still on disk, but pod B's
    re-write is mid-flight at the moment of inspection.

    Concretely: simulate the write-order dance by writing a v1 manifest
    for a v1 payload, then atomically replacing the payload with a
    different-size v1 payload WITHOUT updating the manifest.
    """
    cache_path = tmp_path / "_stage5_teacher_logits.pt"
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".MANIFEST.json")
    atomic_torch_save(cache_path, _fake_cache_payload(num_samples=4))
    write_manifest_last(cache_path, manifest_path, schema_version=1)
    # Replace payload with a larger one but keep the old manifest.
    atomic_torch_save(cache_path, _fake_cache_payload(num_samples=16))
    with pytest.raises(ManifestMismatchError, match="size"):
        read_and_validate_manifest(cache_path, manifest_path, expected_schema_version=1)
