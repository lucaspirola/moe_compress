"""Durable atomic-write helpers for calibration-phase artifacts.

This module is the single source of truth for "kill-safe" file writes across
all calibration / training stages. It implements **Pattern O**
(architectural-patterns) — *atomic-write + manifest-last for durable
calibration artifacts*.

Threat model
------------
GPU eviction on vast.ai / DataCrunch / spot instances delivers a SIGKILL
with no shutdown hook. A fresh GPU instance re-attaches the HF cache and
the per-run output directory and resumes the calibration script. Every
on-disk write must be either:

* **Atomic on disk** — tmp + fsync(fd) + os.replace + fsync(parent_dir).
  A torn write leaves a stale .tmp file (ignorable) and the previous
  good final-path file intact.
* **Re-doable on resume** — artifact validated on read; corrupt → re-do
  the work from scratch, NEVER silently consume a partial file.

Helpers in this module
----------------------
* :func:`atomic_torch_save`         — tmp + fsync + os.replace + fsync parent.
* :func:`atomic_npz_save`           — same, dodging numpy's .npz auto-suffix bug.
* :func:`atomic_json_save`          — same, with json.dumps.
* :func:`atomic_safetensors_save`   — same, with safetensors.save_file.
* :func:`atomic_write_text`         — same, for plain text.
* :func:`write_manifest_last`       — write a sidecar manifest.json AFTER a
  payload file has been atomically flushed; computes sha256 (configurable)
  and embeds size + schema_version. Manifest is itself written atomically
  and is the LAST thing a write sequence does.
* :func:`read_and_validate_manifest` — load a manifest, optionally verify
  sha256 and size match the payload on disk; raises with an actionable
  message on any mismatch.

Backward-compat shim
--------------------
``stage2.shared_io._durable_rename`` is preserved as a thin shim that calls
:func:`durable_rename` here. Existing call sites do not need to change.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


__all__ = [
    "durable_rename",
    "atomic_torch_save",
    "atomic_npz_save",
    "atomic_json_save",
    "atomic_safetensors_save",
    "atomic_write_text",
    "write_manifest_last",
    "read_and_validate_manifest",
    "ManifestMismatchError",
]


class ManifestMismatchError(RuntimeError):
    """Raised when a manifest disagrees with the payload it describes.

    Distinct from a generic RuntimeError so callers can re-capture rather
    than crash the whole pipeline if they so choose. Default behavior is
    to propagate — silent fall-back is forbidden by Pattern O.
    """


# ---------------------------------------------------------------------------
# Internal primitives.
# ---------------------------------------------------------------------------
def _fsync_file(path: Path) -> None:
    """Open the file for read, fsync the fd, close. Best-effort across FS.

    POSIX guarantees that fsync() flushes the file's data + metadata. On
    ext4/xfs this is a real flush. On tmpfs / FUSE mounts (HF Jobs bucket)
    fsync may raise EINVAL/ENOTSUP; we swallow with a debug log because
    rename is already atomic at the FS level.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        # Some virtual filesystems (tmpfs, FUSE) don't support fsync on
        # regular files. The rename below is still atomic at the FS layer.
        log.debug("atomic_io: fsync(%s) raised %s — non-POSIX FS?", path, exc)


def _fsync_dir(directory: Path) -> None:
    """fsync the parent directory so that the rename entry survives power loss."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # tmpfs / FUSE mounts (HF Jobs) raise EINVAL — swallow.
        log.debug("atomic_io: fsync(dir=%s) raised OSError — non-POSIX FS?", directory)


def durable_rename(tmp: Path, final: Path) -> None:
    """Spec §11 durable rename: fsync(tmp) → os.replace → fsync(parent_dir).

    ``tmp`` must already be closed at the Python level (all userspace
    buffers flushed). The fsync inside this function flushes the kernel
    page-cache so the data blocks are durable before the rename commits.

    A kill between the fsync and the rename leaves a stale .tmp (caller
    should sweep on resume) and the previous final-path file intact. A
    kill between the rename and the parent-dir fsync leaves the new
    final-path file readable but its directory entry may be lost on
    power-loss / kernel-panic; the rename itself is still atomic at the
    POSIX level.
    """
    tmp = Path(tmp)
    final = Path(final)
    _fsync_file(tmp)
    os.replace(tmp, final)
    _fsync_dir(final.parent)


# ---------------------------------------------------------------------------
# Atomic writers (Pattern O — single source of truth).
# ---------------------------------------------------------------------------
def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Atomically write ``payload`` to ``path`` via torch.save.

    Sequence: mkdir parents → torch.save(payload, tmp) → fsync(tmp) →
    os.replace(tmp, path) → fsync(parent_dir). The tmp suffix is always
    ``.tmp`` appended to the FULL final filename (e.g. ``foo.pt.tmp``).

    On any exception the partial .tmp file is unlinked. The previous
    contents of ``path`` (if any) are untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        torch.save(payload, tmp)
        durable_rename(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_npz_save(path: str | Path, **arrays: np.ndarray) -> Path:
    """Atomically write a ``.npz`` archive.

    CRITICAL: ``np.savez_compressed(path_str, …)`` auto-appends ``.npz``
    to any path that does NOT already end in ``.npz``. A naive
    ``tmp = "out.npz.tmp"`` + ``np.savez_compressed(tmp, …)`` writes
    ``out.npz.tmp.npz`` and the subsequent ``os.replace(tmp, …)`` fails
    with FileNotFoundError (F-C-1).

    The fix: pass an open binary file HANDLE to numpy, which does NOT
    auto-append. Then fsync the handle, close, durable_rename.

    The signature is ``atomic_npz_save(path, **arrays)`` to mirror
    numpy's own ``savez_compressed(file, **arrays)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_json_save(
    path: str | Path,
    obj: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> Path:
    """Atomically write ``obj`` as JSON to ``path``.

    Encoding is UTF-8. Default formatting is human-readable
    (``indent=2``, ``sort_keys=True``); pass ``indent=None`` for a
    compact one-liner.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent, sort_keys=sort_keys)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_safetensors_save(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Atomically write a safetensors file.

    safetensors.save_file writes in place by default (no atomic helper in
    the upstream library). We pass it a tmp path, then fsync + rename.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        if metadata is None:
            save_file(tensors, str(tmp))
        else:
            save_file(tensors, str(tmp), metadata=metadata)
        durable_rename(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write a UTF-8 text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        with open(tmp, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


# ---------------------------------------------------------------------------
# Manifest-last protocol (Pattern O — for large artifacts).
# ---------------------------------------------------------------------------
_DEFAULT_CHUNK = 1 << 20  # 1 MiB


def _sha256_file(path: Path, chunk: int = _DEFAULT_CHUNK) -> str:
    """Stream a file through SHA-256. Returns the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def write_manifest_last(
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    schema_version: int,
    extra_meta: dict[str, Any] | None = None,
    compute_sha256: bool = True,
) -> Path:
    """Write a sidecar manifest.json describing ``payload_path``.

    MUST be called AFTER ``payload_path`` has been atomically written and
    fsynced (i.e. its data blocks are durable). The manifest itself is
    written atomically and contains:

    * ``schema_version`` (int)
    * ``payload_name`` (str — basename of payload_path)
    * ``size_bytes`` (int — payload's current size on disk)
    * ``sha256`` (str | None — payload's SHA-256, or None if
      compute_sha256=False; large artifacts may set this False to skip
      the I/O cost and rely on size + schema_version cross-checks
      instead)
    * ``extra`` (dict | absent — caller-supplied bag for forensics)

    The manifest is the LAST thing written by the producer. A reader
    that finds ``payload_path`` without ``manifest_path`` MUST treat
    the payload as torn (= mid-write SIGKILL) and either re-capture or
    fail loudly. See :func:`read_and_validate_manifest`.
    """
    payload_path = Path(payload_path)
    manifest_path = Path(manifest_path)

    if not payload_path.exists():
        raise FileNotFoundError(
            f"write_manifest_last: payload {payload_path} does not exist — "
            "the payload must be atomically written and fsynced before "
            "the manifest is written"
        )

    size_bytes = payload_path.stat().st_size
    sha256: str | None = None
    if compute_sha256:
        sha256 = _sha256_file(payload_path)

    manifest = {
        "schema_version": int(schema_version),
        "payload_name": payload_path.name,
        "size_bytes": int(size_bytes),
        "sha256": sha256,
    }
    if extra_meta:
        # Caller meta is kept under a sub-key so future top-level fields
        # we add here can't collide with caller forensics.
        manifest["extra"] = dict(extra_meta)

    return atomic_json_save(manifest_path, manifest, indent=2, sort_keys=True)


def read_and_validate_manifest(
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_schema_version: int,
    require_sha256: bool = False,
) -> dict[str, Any]:
    """Load the manifest and cross-validate it against the payload.

    Raises :class:`ManifestMismatchError` (a RuntimeError subclass) with
    an actionable message if any of the following are wrong:

    * manifest file is missing or unparseable → payload is presumed torn
    * schema_version disagrees with ``expected_schema_version``
    * payload's size on disk disagrees with manifest's ``size_bytes``
    * ``require_sha256=True`` and the manifest's ``sha256`` disagrees
      with the live SHA-256 of the payload (expensive — skipped by
      default for multi-GB payloads; reserve for security-sensitive
      reads)

    Returns the parsed manifest dict on success.

    The caller decides what to do with the failure: deleting the
    payload + manifest and re-running is the standard recovery path on
    a torn-write resume.
    """
    payload_path = Path(payload_path)
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise ManifestMismatchError(
            f"Manifest {manifest_path} is missing. The payload "
            f"{payload_path} may be torn (mid-write SIGKILL). Delete the "
            "payload and re-run to re-capture."
        )

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestMismatchError(
            f"Manifest {manifest_path} unreadable ({exc}). The payload "
            f"{payload_path} may be torn. Delete both and re-run."
        ) from exc

    sv = manifest.get("schema_version")
    if sv != expected_schema_version:
        raise ManifestMismatchError(
            f"Manifest {manifest_path} has schema_version={sv}, expected "
            f"{expected_schema_version}. Delete the manifest + payload "
            f"{payload_path} and re-run to re-capture under the new schema."
        )

    if not payload_path.exists():
        raise ManifestMismatchError(
            f"Manifest {manifest_path} references payload "
            f"{payload_path.name} but the payload is missing on disk. "
            "Delete the manifest and re-run."
        )

    expected_size = manifest.get("size_bytes")
    actual_size = payload_path.stat().st_size
    if expected_size is not None and int(actual_size) != int(expected_size):
        raise ManifestMismatchError(
            f"Payload {payload_path} size {actual_size} bytes disagrees "
            f"with manifest {manifest_path} size_bytes={expected_size}. "
            "Payload is torn — delete both and re-run."
        )

    if require_sha256:
        expected_sha = manifest.get("sha256")
        if expected_sha is None:
            raise ManifestMismatchError(
                f"Manifest {manifest_path} has no sha256 but require_sha256=True"
            )
        actual_sha = _sha256_file(payload_path)
        if actual_sha != expected_sha:
            raise ManifestMismatchError(
                f"Payload {payload_path} sha256={actual_sha} disagrees "
                f"with manifest sha256={expected_sha}. Payload corrupted — "
                "delete both and re-run."
            )

    return manifest
