"""CPU-only tests for the partial-checkpoint resume logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from structural_recovery.distillation import (
    _append_to_partials_manifest, _load_latest_partial, _prune_old_partials,
)


def _make_partial(base: Path, step: int, *, complete: bool = True) -> Path:
    """Create a fake partial checkpoint dir."""
    d = base / f"chapter1_recovered_partial_step{step}"
    d.mkdir(parents=True)
    (d / "compressed_metadata.json").write_text(json.dumps({"version": 1}))
    if complete:
        (d / "_SAVE_COMPLETE").write_text(json.dumps({"step": step, "partial": True}))
    return d


def test_no_partials_returns_none(tmp_path):
    assert _load_latest_partial(tmp_path) == (None, 0)


def test_single_valid_partial(tmp_path):
    d = _make_partial(tmp_path, 500)
    path, step = _load_latest_partial(tmp_path)
    assert step == 500
    assert path == d


def test_incomplete_partial_ignored(tmp_path):
    _make_partial(tmp_path, 500, complete=False)
    assert _load_latest_partial(tmp_path) == (None, 0)


def test_picks_highest_step(tmp_path):
    _make_partial(tmp_path, 500)
    d1000 = _make_partial(tmp_path, 1000)
    path, step = _load_latest_partial(tmp_path)
    assert step == 1000
    assert path == d1000


def test_skips_incomplete_picks_lower_complete(tmp_path):
    d500 = _make_partial(tmp_path, 500)
    _make_partial(tmp_path, 1000, complete=False)
    path, step = _load_latest_partial(tmp_path)
    assert step == 500
    assert path == d500


def test_tmp_dirs_ignored(tmp_path):
    # .tmp siblings from an interrupted atomic rename must never be loaded.
    tmp_dir = tmp_path / "chapter1_recovered_partial_step500.tmp"
    tmp_dir.mkdir()
    (tmp_dir / "_SAVE_COMPLETE").write_text("{}")
    assert _load_latest_partial(tmp_path) == (None, 0)


def test_unparseable_step_skipped(tmp_path):
    bad = tmp_path / "chapter1_recovered_partial_stepXYZ"
    bad.mkdir()
    (bad / "_SAVE_COMPLETE").write_text("{}")
    assert _load_latest_partial(tmp_path) == (None, 0)


def test_multiple_valid_partials_picks_latest(tmp_path):
    _make_partial(tmp_path, 100)
    _make_partial(tmp_path, 200)
    d300 = _make_partial(tmp_path, 300)
    path, step = _load_latest_partial(tmp_path)
    assert step == 300
    assert path == d300


def test_save_complete_present_metadata_missing_still_picked(tmp_path):
    """Discovery contract: ``_SAVE_COMPLETE`` is the integrity sentinel.

    If the metadata file is absent the partial is still "complete" per the
    discovery layer — load failures surface downstream when
    ``load_compressed_model`` reads the partial. This pins the contract so a
    future change that adds metadata-presence to discovery (which would mask
    real corruption with a silent fallback) breaks the test loudly.
    """
    d = tmp_path / "chapter1_recovered_partial_step500"
    d.mkdir()
    (d / "_SAVE_COMPLETE").write_text("{}")
    # Deliberately NO compressed_metadata.json.
    path, step = _load_latest_partial(tmp_path)
    assert step == 500
    assert path == d


def test_resume_after_prune_picks_surviving_partial(tmp_path):
    """End-to-end: write 3 partials with manifest, prune to 1, re-discover.

    Mirrors the production flow: train writes partial → manifest appended →
    next save triggers prune(keep=1) → resume after a kill must find the
    surviving partial, not the pruned one.
    """
    for step in (100, 200, 300):
        d = _make_partial(tmp_path, step)
        # Add a fake shard so _sha256_of_first_shard has something to hash.
        (d / "model-00001-of-00001.safetensors").write_bytes(
            b"step-" + str(step).encode()
        )
        _append_to_partials_manifest(
            tmp_path, step=step,
            path=f"chapter1_recovered_partial_step{step}",
            sha256_first_shard=f"sha-{step}",
        )

    _prune_old_partials(tmp_path, keep=1)

    # Disk: only step=300 survives.
    assert not (tmp_path / "chapter1_recovered_partial_step100").exists()
    assert not (tmp_path / "chapter1_recovered_partial_step200").exists()
    assert (tmp_path / "chapter1_recovered_partial_step300").exists()

    path, step = _load_latest_partial(tmp_path)
    assert step == 300
    assert path == tmp_path / "chapter1_recovered_partial_step300"

    manifest = json.loads((tmp_path / "partials.json").read_text())
    assert [e["step"] for e in manifest] == [300]
