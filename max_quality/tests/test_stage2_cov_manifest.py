"""S-2: Stage 2 ``_stage2_input_covariance.pt`` must be written via
``atomic_torch_save + write_manifest_last``, and all three readers
(Stage 3 covariance load, Stage 4 EoRA cache-miss, SVC audit) must read
+ validate the manifest before opening the .pt.

Mirrors ``test_stage3_originals_manifest.py`` 1:1, on a different payload.

We exercise the production ``_save_covariance`` writer directly (via a
minimal fake of ``InputCovarianceAccumulator`` matching the three attrs
the writer reads: ``_lock`` / ``covariance`` / ``token_count``) so the
lock-held snapshot, manifest-path derivation, and stale-manifest
unlink branches all get coverage — not just the bare
``atomic_torch_save + write_manifest_last`` pair.
"""
from __future__ import annotations

import json
import threading
from unittest import mock

import pytest
import torch

from moe_compress.stage2.shared_io import _save_covariance
from moe_compress.stage3.plugins.covariance_collection import (
    _load_stage2_covariance,
)
from moe_compress.utils.atomic_io import (
    ManifestMismatchError,
    read_and_validate_manifest,
)


class _FakeInputCovarianceAccumulator:
    """Minimal stand-in for ``InputCovarianceAccumulator``.

    Surface required by ``_save_covariance``:

      * ``_lock``: ``threading.Lock`` instance (the writer enters
        ``with cov._lock:`` before snapshotting).
      * ``covariance``: ``dict[(layer, expert, matrix) -> Tensor]``.
      * ``token_count``: ``dict[(layer, expert, matrix) -> int]``
        parallel-keyed.

    No methods of the real accumulator are invoked by the writer.
    """

    def __init__(self, covariance: dict, token_count: dict) -> None:
        self._lock = threading.Lock()
        self.covariance = covariance
        self.token_count = token_count


def _fake_cov(dtype: torch.dtype = torch.float16) -> _FakeInputCovarianceAccumulator:
    cov = {
        (0, 0, "gate_proj"): torch.eye(3, dtype=dtype),
        (0, 0, "down_proj"): torch.eye(3, dtype=dtype) * 2,
        (0, 1, "gate_proj"): torch.eye(3, dtype=dtype) * 3,
    }
    tokens = {k: 5 for k in cov}
    return _FakeInputCovarianceAccumulator(cov, tokens)


# ---------------------------------------------------------------------------
# T1 — round-trip via the real writer + Reader 1
# ---------------------------------------------------------------------------


def test_stage2_cov_manifest_roundtrip(tmp_path):
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    manifest_path = tmp_path / "_stage2_input_covariance.pt.MANIFEST.json"

    cov = _fake_cov()
    _save_covariance(cov, cov_path)

    assert cov_path.exists()
    assert manifest_path.exists()

    # Stage 4 / Stage 3 read path: validate manifest first, then torch.load.
    out = read_and_validate_manifest(
        cov_path, manifest_path, expected_schema_version=1,
    )
    assert out["payload_name"] == cov_path.name
    assert out["extra"]["artifact"] == "_stage2_input_covariance.pt"
    assert out["extra"]["n_keys"] == 3
    assert out["extra"]["covariance_storage_dtype"] == "float16"

    # Reader 1 returns the cov dict on a healthy artifact.
    loaded = _load_stage2_covariance(cov_path)
    assert set(loaded.keys()) == set(cov.covariance.keys())


# ---------------------------------------------------------------------------
# T2 — torn payload via Reader 1
# ---------------------------------------------------------------------------


def test_stage2_cov_torn_payload_fails_loudly(tmp_path):
    """The audit's worst-case: a kill mid-write leaves a TRUNCATED .pt at
    the final path. Without manifest validation, Stage 3 might
    .get(..., {})-fallback and produce silently wrong AA-SVD. With the
    fix, the manifest's size_bytes mismatch raises ManifestMismatchError
    BEFORE Stage 3 ever touches the corrupt file."""
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    _save_covariance(_fake_cov(), cov_path)

    # Truncate the payload to simulate SIGKILL mid-write recovered by
    # the next pod (the manifest from the previous successful run still
    # exists, but the .pt is now half-size).
    real_size = cov_path.stat().st_size
    with open(cov_path, "r+b") as f:
        f.truncate(real_size // 2)

    with pytest.raises(RuntimeError, match="re-run Stage 2"):
        _load_stage2_covariance(cov_path)


# ---------------------------------------------------------------------------
# T3 — missing manifest, NEW writer (kill between atomic_torch_save +
# write_manifest_last)
# ---------------------------------------------------------------------------


def test_stage2_cov_missing_manifest_after_new_writer_torn(tmp_path):
    """Simulate a kill between ``atomic_torch_save`` and
    ``write_manifest_last``: the .pt is durable but the manifest never
    landed. With the back-compat WARN-and-continue branch in Reader 1
    this should still load (because pre-S-2 .pt files have the same
    shape: payload but no manifest). It's the back-compat T7 path."""
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    manifest_path = tmp_path / "_stage2_input_covariance.pt.MANIFEST.json"
    _save_covariance(_fake_cov(), cov_path)
    # Delete the manifest the writer just wrote → the on-disk shape is
    # indistinguishable from a pre-S-2 .pt with no manifest. Reader 1
    # should WARN-and-continue.
    manifest_path.unlink()
    # No error — back-compat fallback returns the cov dict.
    loaded = _load_stage2_covariance(cov_path)
    assert len(loaded) == 3


# ---------------------------------------------------------------------------
# T4 — schema bump invalidates manifest
# ---------------------------------------------------------------------------


def test_stage2_cov_schema_bump_invalidates(tmp_path):
    """A schema_version bump in Stage 2 must invalidate stale manifests."""
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    manifest_path = tmp_path / "_stage2_input_covariance.pt.MANIFEST.json"
    _save_covariance(_fake_cov(), cov_path)

    # Future revision of Stage 2 expects schema_version=2; the manifest
    # on disk says schema_version=1.
    with pytest.raises(ManifestMismatchError, match="schema_version"):
        read_and_validate_manifest(
            cov_path, manifest_path, expected_schema_version=2,
        )


# ---------------------------------------------------------------------------
# T5 — manifest is written LAST (mock os.replace to prove ordering)
# ---------------------------------------------------------------------------


def test_stage2_cov_manifest_written_last(tmp_path):
    """Belt-and-braces: the writer must call ``write_manifest_last``
    AFTER ``atomic_torch_save`` returns. We spy on os.replace (the final
    durable rename inside atomic_*_save) and assert the payload rename
    fires before the manifest rename."""
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    manifest_path = tmp_path / "_stage2_input_covariance.pt.MANIFEST.json"

    import os as _os
    real_replace = _os.replace
    rename_order: list[str] = []

    def _spy_replace(src, dst, *args, **kwargs):
        # Record only the final-target basenames we care about. The
        # atomic helpers rename `<dst>.tmp` → `<dst>`, so the
        # destination basename is the file we want to track.
        from pathlib import Path as _P
        dst_name = _P(dst).name
        if dst_name in (cov_path.name, manifest_path.name):
            rename_order.append(dst_name)
        return real_replace(src, dst, *args, **kwargs)

    with mock.patch("os.replace", _spy_replace):
        _save_covariance(_fake_cov(), cov_path)

    # Both renames fired, in order: payload first, then manifest.
    assert rename_order == [cov_path.name, manifest_path.name], (
        f"manifest-LAST ordering violated: {rename_order}"
    )


# ---------------------------------------------------------------------------
# T6 — stale manifest is unlinked BEFORE the new payload write
# ---------------------------------------------------------------------------


def test_stage2_cov_stale_manifest_unlinked_before_write(tmp_path):
    """A previous run left a stale manifest pointing at a stale .pt.
    The new writer must unlink the stale manifest BEFORE the new payload
    lands, so an interrupted re-write cannot briefly leave a manifest
    that vouches for the OLD payload."""
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    manifest_path = tmp_path / "_stage2_input_covariance.pt.MANIFEST.json"

    # Plant a stale manifest claiming a tiny size. If the writer didn't
    # unlink before writing, atomic_torch_save would replace the .pt at
    # a moment when the stale manifest still "vouches" for an old size —
    # but with the unlink-first contract, the manifest is gone before
    # the new payload lands.
    stale = {
        "schema_version": 1,
        "payload_name": cov_path.name,
        "size_bytes": 1,
        "sha256": None,
        "write_timestamp_iso": "2020-01-01T00:00:00+00:00",
    }
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")

    # Spy on Path.unlink — the writer should call it on the manifest
    # path before atomic_torch_save runs.
    import os as _os
    real_replace = _os.replace
    events: list[tuple[str, str]] = []

    from pathlib import Path as _P
    real_unlink = _P.unlink

    def _spy_unlink(self, *args, **kwargs):
        if self.name == manifest_path.name:
            events.append(("unlink", self.name))
        return real_unlink(self, *args, **kwargs)

    def _spy_replace(src, dst, *args, **kwargs):
        dst_name = _P(dst).name
        if dst_name in (cov_path.name, manifest_path.name):
            events.append(("replace", dst_name))
        return real_replace(src, dst, *args, **kwargs)

    with mock.patch.object(_P, "unlink", _spy_unlink), \
            mock.patch("os.replace", _spy_replace):
        _save_covariance(_fake_cov(), cov_path)

    # The first event for the manifest must be 'unlink', and it must
    # come BEFORE the payload's 'replace' event.
    manifest_events = [e for e in events if e[1] == manifest_path.name]
    payload_events = [e for e in events if e[1] == cov_path.name]
    assert manifest_events[0] == ("unlink", manifest_path.name), (
        f"first manifest event must be unlink: {events}"
    )
    # Payload replace must come AFTER the stale-manifest unlink.
    first_unlink_idx = events.index(("unlink", manifest_path.name))
    first_payload_replace_idx = events.index(("replace", cov_path.name))
    assert first_unlink_idx < first_payload_replace_idx, (
        f"stale manifest must be unlinked BEFORE new payload write: {events}"
    )

    # Post-condition: the new manifest exists and vouches for the NEW size.
    assert manifest_path.exists()
    new_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert new_manifest["size_bytes"] == cov_path.stat().st_size


# ---------------------------------------------------------------------------
# T7 — legacy (no-manifest) .pt loads with a WARN
# ---------------------------------------------------------------------------


def test_stage2_cov_legacy_no_manifest_loads_with_warn(tmp_path):
    """Back-compat contract: a pre-S-2 ``.pt`` (bare ``torch.save`` with
    NO manifest sibling) MUST still load through Reader 1 — with a single
    WARNING. This is what keeps in-flight runs whose Stage 2 finished
    BEFORE this migration landed working without re-capture.

    We attach an explicit logging.Handler to the reader's logger rather
    than relying on pytest's ``caplog`` fixture: some conftest setups in
    this repo route logs in a way that bypasses caplog's default capture,
    so the handler-attach pattern is the robust idiom (used elsewhere in
    the suite, e.g. resume tests)."""
    import logging

    cov_path = tmp_path / "_stage2_input_covariance.pt"
    # Bare write — no manifest. Mimics a pre-S-2 Stage 2 finalize.
    torch.save(
        {
            "format_version": 1,
            "covariance": {
                (0, 0, "gate_proj"): torch.eye(3, dtype=torch.float16),
            },
            "tokens": {(0, 0, "gate_proj"): 5},
        },
        cov_path,
    )

    captured: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    reader_logger = logging.getLogger(
        "moe_compress.stage3.plugins.covariance_collection"
    )
    handler = _CaptureHandler(level=logging.WARNING)
    # Make sure the logger forwards WARNING to our handler regardless of
    # any prior basicConfig.
    prev_level = reader_logger.level
    reader_logger.setLevel(logging.WARNING)
    reader_logger.addHandler(handler)
    try:
        loaded = _load_stage2_covariance(cov_path)
    finally:
        reader_logger.removeHandler(handler)
        reader_logger.setLevel(prev_level)

    assert (0, 0, "gate_proj") in loaded
    matched = [m for m in captured if "pre-S-2" in m]
    assert len(matched) >= 1, (
        f"expected pre-S-2 WARN; got {captured}"
    )


# ---------------------------------------------------------------------------
# T8 — no .tmp leftovers
# ---------------------------------------------------------------------------


def test_stage2_cov_no_tmp_leftovers(tmp_path):
    """Belt-and-braces: writer leaves no .tmp orphans on success."""
    cov_path = tmp_path / "_stage2_input_covariance.pt"
    _save_covariance(_fake_cov(), cov_path)
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# T9 — Hub-upload lists include the manifest, AFTER the .pt
# ---------------------------------------------------------------------------


def test_stage2_cov_manifest_in_hub_upload_lists():
    """HIGH-2 / S-2 mirror: the manifest must be in BOTH Hub upload lists,
    AFTER the .pt, so Pattern O's manifest-LAST invariant survives the
    Hub durability boundary too. A partial upload that drops the manifest
    leaves Stage 3/4 with a torn-write signature (missing manifest),
    which surfaces as a back-compat WARN instead of silently consuming a
    half-uploaded payload that would later mismatch in size on a re-run.
    """
    # _STAGE_LAYOUT (per-stage uploader).
    from moe_compress.utils.hub_upload import _STAGE_LAYOUT
    _stage2_subdir, stage2_sidecars = _STAGE_LAYOUT[2]
    assert "_stage2_input_covariance.pt" in stage2_sidecars
    assert "_stage2_input_covariance.pt.MANIFEST.json" in stage2_sidecars
    # Manifest must come AFTER the payload in the list (manifest-LAST).
    pt_idx = stage2_sidecars.index("_stage2_input_covariance.pt")
    manifest_idx = stage2_sidecars.index(
        "_stage2_input_covariance.pt.MANIFEST.json"
    )
    assert manifest_idx > pt_idx, (
        f"Pattern O violation: manifest at index {manifest_idx} must "
        f"come AFTER payload at index {pt_idx} in _STAGE_LAYOUT[2]"
    )

    # entrypoint.aux_files (job-exit aux uploader). Parse the source so
    # we don't import the script (it pulls in heavy deps).
    from pathlib import Path
    src = Path(__file__).parent.parent / "hf_jobs" / "entrypoint.py"
    text = src.read_text()
    assert '"_stage2_input_covariance.pt"' in text, "aux_files missing .pt"
    assert '"_stage2_input_covariance.pt.MANIFEST.json"' in text, (
        "aux_files missing MANIFEST.json"
    )
    # Manifest must appear AFTER the .pt in the file (manifest-LAST).
    pt_offset = text.index('"_stage2_input_covariance.pt"')
    mf_offset = text.index('"_stage2_input_covariance.pt.MANIFEST.json"')
    assert mf_offset > pt_offset, (
        "Pattern O violation: aux_files lists MANIFEST.json before .pt"
    )
