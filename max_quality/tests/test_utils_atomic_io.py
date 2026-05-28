"""Tests for moe_compress.utils.atomic_io (Pattern O).

The audit (F-C-1, F-S3-1, F-RK-1, F-H-1..7) motivates one shared
durable-write module. These tests cover:

* Round-trip correctness for each writer.
* No .tmp file orphan on success.
* Previous file contents preserved on a mid-write crash.
* npz writer dodges numpy's .npz auto-suffix bug (F-C-1 regression).
* Manifest-last protocol: writer ordering + reader-side validation +
  loud failure on torn payload.
"""
from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import torch

from moe_compress.utils.atomic_io import (
    ManifestMismatchError,
    atomic_json_save,
    atomic_npz_save,
    atomic_safetensors_save,
    atomic_torch_save,
    atomic_write_text,
    durable_rename,
    read_and_validate_manifest,
    write_manifest_last,
)


# ---------------------------------------------------------------------------
# atomic_torch_save
# ---------------------------------------------------------------------------
def test_atomic_torch_save_roundtrip(tmp_path):
    p = tmp_path / "sub" / "out.pt"
    payload = {"k": torch.arange(4, dtype=torch.float32)}
    atomic_torch_save(p, payload)
    assert p.exists()
    assert not list(tmp_path.rglob("*.tmp"))
    reloaded = torch.load(p, weights_only=False)
    assert torch.equal(reloaded["k"], payload["k"])


def test_atomic_torch_save_preserves_previous_on_failure(tmp_path):
    p = tmp_path / "out.pt"
    atomic_torch_save(p, {"v": 1})

    # Simulate a crash *during* torch.save (before tmp closed).
    def boom(*a, **kw):
        raise RuntimeError("simulated SIGKILL")
    with mock.patch("moe_compress.utils.atomic_io.torch.save", boom):
        with pytest.raises(RuntimeError, match="simulated"):
            atomic_torch_save(p, {"v": 2})
    # Previous file still readable.
    reloaded = torch.load(p, weights_only=False)
    assert reloaded == {"v": 1}
    # No tmp orphan.
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_torch_save_kill_between_save_and_rename(tmp_path):
    """A kill AFTER torch.save (tmp file exists) but BEFORE os.replace
    must leave the previous final-path file intact."""
    p = tmp_path / "out.pt"
    atomic_torch_save(p, {"v": 1})

    # Inject a failure inside os.replace (post-write, pre-rename).
    def boom(src, dst):
        raise RuntimeError("simulated kill before rename")
    with mock.patch("moe_compress.utils.atomic_io.os.replace", boom):
        with pytest.raises(RuntimeError):
            atomic_torch_save(p, {"v": 2})
    reloaded = torch.load(p, weights_only=False)
    assert reloaded == {"v": 1}
    # tmp file is cleaned up by the except branch.
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# atomic_npz_save — F-C-1 regression
# ---------------------------------------------------------------------------
def test_atomic_npz_save_roundtrip(tmp_path):
    p = tmp_path / "logits" / "000.npz"
    atomic_npz_save(
        p,
        token_ids=np.arange(8, dtype=np.int32),
        top_ids=np.zeros((8, 5), dtype=np.int32),
    )
    assert p.exists()
    with np.load(p) as data:
        assert np.array_equal(data["token_ids"], np.arange(8, dtype=np.int32))
        assert data["top_ids"].shape == (8, 5)


def test_atomic_npz_save_no_double_extension_ghost(tmp_path):
    """F-C-1 regression: confirm `out.npz.tmp.npz` ghost file does NOT
    appear. The naive `np.savez_compressed("out.npz.tmp", …)` would
    write to `out.npz.tmp.npz` and leave the .tmp file nonexistent.
    """
    p = tmp_path / "out.npz"
    atomic_npz_save(p, arr=np.ones(3, dtype=np.float32))
    assert p.exists()
    # No ghost double-extension file.
    assert not (tmp_path / "out.npz.tmp.npz").exists()
    assert not (tmp_path / "out.npz.tmp").exists()
    # No stray .tmp files.
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_npz_save_preserves_previous_on_failure(tmp_path):
    p = tmp_path / "out.npz"
    atomic_npz_save(p, arr=np.array([1, 2, 3], dtype=np.int32))
    # Make np.savez_compressed blow up half-way.

    def boom(fh, **kw):
        # write some bytes then crash — simulates a partial flush.
        if hasattr(fh, "write"):
            fh.write(b"GARBAGE_PARTIAL")
        raise RuntimeError("kill mid-savez")

    with mock.patch("moe_compress.utils.atomic_io.np.savez_compressed", boom):
        with pytest.raises(RuntimeError):
            atomic_npz_save(p, arr=np.array([9, 9, 9], dtype=np.int32))
    # Previous file intact.
    with np.load(p) as data:
        assert np.array_equal(data["arr"], np.array([1, 2, 3], dtype=np.int32))
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# atomic_json_save / atomic_write_text
# ---------------------------------------------------------------------------
def test_atomic_json_save_roundtrip(tmp_path):
    p = tmp_path / "meta.json"
    atomic_json_save(p, {"x": 1, "y": [1, 2, 3]})
    loaded = json.loads(p.read_text())
    assert loaded == {"x": 1, "y": [1, 2, 3]}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_text_roundtrip(tmp_path):
    p = tmp_path / "hello.txt"
    atomic_write_text(p, "héllo")
    assert p.read_text() == "héllo"


# ---------------------------------------------------------------------------
# atomic_safetensors_save
# ---------------------------------------------------------------------------
def test_atomic_safetensors_save_roundtrip(tmp_path):
    from safetensors.torch import safe_open

    p = tmp_path / "shard.safetensors"
    atomic_safetensors_save(
        p,
        {"a": torch.arange(4, dtype=torch.float32), "b": torch.zeros(2, 3)},
    )
    assert p.exists()
    with safe_open(str(p), framework="pt", device="cpu") as f:
        assert torch.equal(f.get_tensor("a"), torch.arange(4, dtype=torch.float32))
        assert f.get_tensor("b").shape == (2, 3)
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# durable_rename
# ---------------------------------------------------------------------------
def test_durable_rename_basic(tmp_path):
    tmp = tmp_path / "foo.pt.tmp"
    final = tmp_path / "foo.pt"
    tmp.write_bytes(b"hello")
    durable_rename(tmp, final)
    assert not tmp.exists()
    assert final.read_bytes() == b"hello"


def test_durable_rename_call_order_is_fsync_replace_fsync(tmp_path):
    """MEDIUM-3: durable_rename's correctness contract is the CALL
    ORDER of (fsync_file, os.replace, fsync_dir), not just the
    end-state. A reorder (e.g. fsync AFTER replace) would still leave
    the right file on disk under happy-path tests but would lose the
    durability guarantee on power-loss. Verify the sequence directly.
    """
    tmp = tmp_path / "foo.pt.tmp"
    final = tmp_path / "foo.pt"
    tmp.write_bytes(b"hello")

    calls: list[str] = []
    import moe_compress.utils.atomic_io as aio

    real_fsync_file = aio._fsync_file
    real_fsync_dir = aio._fsync_dir
    real_replace = aio.os.replace

    def rec_fsync_file(p):
        calls.append("fsync_file")
        return real_fsync_file(p)

    def rec_replace(src, dst):
        calls.append("replace")
        return real_replace(src, dst)

    def rec_fsync_dir(d):
        calls.append("fsync_dir")
        return real_fsync_dir(d)

    with mock.patch.object(aio, "_fsync_file", side_effect=rec_fsync_file), \
         mock.patch.object(aio.os, "replace", side_effect=rec_replace), \
         mock.patch.object(aio, "_fsync_dir", side_effect=rec_fsync_dir):
        durable_rename(tmp, final)

    assert calls == ["fsync_file", "replace", "fsync_dir"], (
        f"expected sequence [fsync_file, replace, fsync_dir], got {calls}"
    )


# ---------------------------------------------------------------------------
# write_manifest_last + read_and_validate_manifest
# ---------------------------------------------------------------------------
def test_manifest_last_roundtrip(tmp_path):
    payload = tmp_path / "payload.pt"
    atomic_torch_save(payload, {"data": torch.arange(16)})
    manifest = payload.with_suffix(".MANIFEST.json")
    write_manifest_last(payload, manifest, schema_version=1)
    out = read_and_validate_manifest(payload, manifest, expected_schema_version=1)
    assert out["schema_version"] == 1
    assert out["payload_name"] == "payload.pt"
    assert out["size_bytes"] == payload.stat().st_size
    assert isinstance(out["sha256"], str)


def test_manifest_last_missing_manifest_raises(tmp_path):
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"ABCD")
    manifest = tmp_path / "payload.MANIFEST.json"
    with pytest.raises(ManifestMismatchError, match="missing"):
        read_and_validate_manifest(payload, manifest, expected_schema_version=1)


def test_manifest_last_torn_payload_detected_by_size(tmp_path):
    """Simulate F-S3-1 / F-RK-1: a kill mid-write leaves a TRUNCATED .pt at
    the final path. The manifest was written under a previous (whole) run
    with the full size; after a truncating reset of the payload the
    manifest's size_bytes no longer matches → reader fails loudly."""
    payload = tmp_path / "big.pt"
    atomic_torch_save(payload, {"data": torch.arange(1024)})
    manifest = payload.with_suffix(".MANIFEST.json")
    write_manifest_last(payload, manifest, schema_version=1)

    # Truncate the payload (simulate torn write recovered from disk).
    real_size = payload.stat().st_size
    with open(payload, "r+b") as f:
        f.truncate(real_size // 2)

    with pytest.raises(ManifestMismatchError, match="size"):
        read_and_validate_manifest(payload, manifest, expected_schema_version=1)


def test_manifest_last_schema_mismatch_raises(tmp_path):
    payload = tmp_path / "p.pt"
    atomic_torch_save(payload, {"data": 1})
    manifest = payload.with_suffix(".MANIFEST.json")
    write_manifest_last(payload, manifest, schema_version=1)
    with pytest.raises(ManifestMismatchError, match="schema_version"):
        read_and_validate_manifest(payload, manifest, expected_schema_version=2)


def test_manifest_last_sha256_validation(tmp_path):
    payload = tmp_path / "p.pt"
    atomic_torch_save(payload, {"data": torch.arange(8)})
    manifest = payload.with_suffix(".MANIFEST.json")
    write_manifest_last(payload, manifest, schema_version=1, compute_sha256=True)

    # Corrupt the payload but keep the same size — only sha256 catches this.
    size = payload.stat().st_size
    with open(payload, "r+b") as f:
        f.seek(size // 2)
        f.write(b"\xFF" * 8)

    with pytest.raises(ManifestMismatchError, match="sha256"):
        read_and_validate_manifest(
            payload, manifest, expected_schema_version=1, require_sha256=True,
        )


def test_manifest_last_extra_meta_preserved(tmp_path):
    payload = tmp_path / "p.pt"
    atomic_torch_save(payload, {"data": 1})
    manifest = payload.with_suffix(".MANIFEST.json")
    write_manifest_last(
        payload, manifest, schema_version=1,
        extra_meta={"layers": 32, "model": "fake"},
    )
    out = read_and_validate_manifest(payload, manifest, expected_schema_version=1)
    assert out["extra"] == {"layers": 32, "model": "fake"}


def test_manifest_last_includes_write_timestamp(tmp_path):
    """MEDIUM-2: every manifest now carries a UTC ISO-8601 timestamp
    for cross-host forensics. The field is informational only — never
    validated — but must be present on every newly-written manifest."""
    import re
    payload = tmp_path / "p.pt"
    atomic_torch_save(payload, {"x": 1})
    manifest = payload.with_suffix(".MANIFEST.json")
    write_manifest_last(payload, manifest, schema_version=1)
    out = read_and_validate_manifest(payload, manifest, expected_schema_version=1)
    assert "write_timestamp_iso" in out
    ts = out["write_timestamp_iso"]
    # ISO-8601 with timezone offset: 2026-05-28T...+00:00 or ...Z.
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts), (
        f"unexpected timestamp format: {ts!r}"
    )


def test_manifest_last_requires_payload_exists(tmp_path):
    """write_manifest_last MUST refuse to write a manifest for a
    nonexistent payload — that would be the worst possible bug:
    a forward-looking manifest that consumers trust."""
    payload = tmp_path / "missing.pt"
    manifest = tmp_path / "missing.MANIFEST.json"
    with pytest.raises(FileNotFoundError):
        write_manifest_last(payload, manifest, schema_version=1)


# ---------------------------------------------------------------------------
# Crash-injection: kill between fsync(file) and os.replace.
# ---------------------------------------------------------------------------
def test_atomic_torch_save_kill_after_fsync_before_rename(tmp_path):
    """Kill the process state after fsync but before the rename. The
    final-path file MUST be the PREVIOUS good version (or absent if no
    prior write). The tmp file MAY survive (resume sweeps it)."""
    p = tmp_path / "out.pt"
    atomic_torch_save(p, {"v": "first"})
    # Patch durable_rename to fail (between fsync and replace internally,
    # but at this granularity we simulate "rename never happened").
    with mock.patch(
        "moe_compress.utils.atomic_io.durable_rename",
        side_effect=RuntimeError("kill after fsync, before rename"),
    ):
        with pytest.raises(RuntimeError):
            atomic_torch_save(p, {"v": "second"})
    # Final-path file is the PREVIOUS good version.
    reloaded = torch.load(p, weights_only=False)
    assert reloaded == {"v": "first"}
