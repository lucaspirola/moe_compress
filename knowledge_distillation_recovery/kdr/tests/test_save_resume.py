"""Tests for `kdr.io.save.save_partial` and `kdr.io.resume.find_latest_partial`.

# VERIFIES: LLR-0027
# VERIFIES: LLR-0029
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

from kdr.io.resume import find_latest_partial
from kdr.io.save import (
    COMPRESSED_METADATA_FILENAME,
    SAVE_COMPLETE_SENTINEL,
    partial_dir_name,
    save_partial,
)


def _fake_accelerator(*, is_main: bool = True) -> MagicMock:
    accel = MagicMock()
    accel.is_main_process = is_main
    accel.wait_for_everyone = MagicMock()
    # `get_state_dict` returns an empty dict (test save calls don't
    # exercise weight serialisation; that's `save_pretrained`'s concern).
    accel.get_state_dict = MagicMock(return_value={})
    accel.unwrap_model = lambda m: m
    return accel


def _fake_model() -> MagicMock:
    """A model that records `save_pretrained` invocations to verify shape."""
    m = MagicMock()

    def _save(out_dir: Path, **kw: object) -> None:
        out_dir = Path(out_dir)
        # Drop a fake shard so the directory looks plausibly populated.
        (out_dir / "model.safetensors").write_bytes(b"\x00" * 16)
        (out_dir / "config.json").write_text("{}")

    m.save_pretrained.side_effect = _save
    return m


def _fake_tokenizer() -> MagicMock:
    tok = MagicMock()

    def _save(out_dir: Path) -> None:
        Path(out_dir).joinpath("tokenizer.json").write_text("{}")

    tok.save_pretrained.side_effect = _save
    return tok


def test_partial_dir_name_format() -> None:
    assert partial_dir_name("bf16", 0) == "kdr_bf16_partial_step0"
    assert partial_dir_name("da_qad", 1234) == "kdr_da_qad_partial_step1234"


def test_save_partial_writes_sentinel_empty(tmp_path: Path) -> None:
    """LLR-0029 AC: sentinel is exactly zero bytes."""
    accel = _fake_accelerator()
    out = save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=10,
        partial=True,
    )
    sentinel = out / SAVE_COMPLETE_SENTINEL
    assert sentinel.exists()
    assert os.path.getsize(sentinel) == 0


def test_save_partial_sentinel_is_last(tmp_path: Path) -> None:
    """LLR-0029 AC: sentinel mtime >= every other file's mtime."""
    accel = _fake_accelerator()
    out = save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=20,
        partial=True,
    )
    sentinel = out / SAVE_COMPLETE_SENTINEL
    sentinel_mtime = sentinel.stat().st_mtime
    for f in out.iterdir():
        if f.name == SAVE_COMPLETE_SENTINEL:
            continue
        assert f.stat().st_mtime <= sentinel_mtime + 1e-6, f"{f.name} newer than sentinel"


def test_save_partial_dir_name_carries_mode_and_step(tmp_path: Path) -> None:
    """LLR-0027 AC: partial dir embeds both mode and step."""
    accel = _fake_accelerator()
    out_bf16 = save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=42,
        partial=True,
    )
    out_qad = save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="da_qad",
        step=42,
        partial=True,
    )
    assert out_bf16.name == "kdr_bf16_partial_step42"
    assert out_qad.name == "kdr_da_qad_partial_step42"


def test_save_partial_preserves_compressed_metadata(tmp_path: Path) -> None:
    """HLR-0005 / LLR-0019: compressed_metadata.json is preserved verbatim."""
    src_meta = tmp_path / "src" / COMPRESSED_METADATA_FILENAME
    src_meta.parent.mkdir()
    src_meta.write_text('{"version": 1, "factored_layers": [3, 7]}')

    accel = _fake_accelerator()
    out = save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path / "artifacts",
        mode="bf16",
        step=5,
        source_metadata_path=src_meta,
        partial=True,
    )
    out_meta = out / COMPRESSED_METADATA_FILENAME
    assert out_meta.exists()
    assert out_meta.read_text() == src_meta.read_text()


def test_save_partial_overwrites_existing_final_dir(tmp_path: Path) -> None:
    """Atomic replace: pre-existing final dir is swapped out cleanly."""
    accel = _fake_accelerator()
    out1 = save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=0,
        partial=False,
    )
    # Place a marker in the first save's dir so we can detect replacement.
    (out1 / "marker.txt").write_text("first")
    out2 = save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=0,
        partial=False,
    )
    assert out1 == out2
    assert not (out2 / "marker.txt").exists()
    # Backup dir from atomic replace must be cleaned up.
    assert not (tmp_path / "kdr_bf16_recovered.bak").exists()


def test_save_partial_skips_writes_off_main_process(tmp_path: Path) -> None:
    """Non-rank-0 processes don't write files but still participate in
    `wait_for_everyone` / `get_state_dict` (collective ops)."""
    accel = _fake_accelerator(is_main=False)
    save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=1,
        partial=True,
    )
    # Returned path is what rank-0 would create; non-main does not actually
    # have the dir on its own filesystem.
    assert accel.wait_for_everyone.call_count == 2
    assert accel.get_state_dict.call_count == 1


# ── find_latest_partial ─────────────────────────────────────────────────────


def test_find_latest_partial_returns_none_for_empty_dir(tmp_path: Path) -> None:
    assert find_latest_partial(tmp_path, "bf16") is None


def test_find_latest_partial_returns_none_for_missing_dir(tmp_path: Path) -> None:
    assert find_latest_partial(tmp_path / "does_not_exist", "bf16") is None


def test_find_latest_partial_picks_highest_step(tmp_path: Path) -> None:
    accel = _fake_accelerator()
    for step in (5, 100, 50):
        save_partial(
            _fake_model(),
            _fake_tokenizer(),
            accel,
            artifacts_dir=tmp_path,
            mode="bf16",
            step=step,
            partial=True,
        )
    result = find_latest_partial(tmp_path, "bf16")
    assert result is not None
    path, step = result
    assert step == 100
    assert path.name == "kdr_bf16_partial_step100"


def test_find_latest_partial_skips_dirs_without_sentinel(tmp_path: Path) -> None:
    """Incomplete writes (no _SAVE_COMPLETE) must be silently skipped — they
    represent crashes mid-write and would load truncated weights."""
    accel = _fake_accelerator()
    save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=10,
        partial=True,
    )
    # Manually create a higher-step dir that lacks the sentinel.
    bad = tmp_path / "kdr_bf16_partial_step99"
    bad.mkdir()
    (bad / "model.safetensors").write_bytes(b"\x00")

    result = find_latest_partial(tmp_path, "bf16")
    assert result is not None
    _, step = result
    # The valid partial wins despite the broken higher-step.
    assert step == 10


def test_find_latest_partial_filters_by_mode(tmp_path: Path) -> None:
    accel = _fake_accelerator()
    save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=10,
        partial=True,
    )
    save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="da_qad",
        step=20,
        partial=True,
    )
    bf16_result = find_latest_partial(tmp_path, "bf16")
    qad_result = find_latest_partial(tmp_path, "da_qad")
    assert bf16_result is not None and bf16_result[1] == 10
    assert qad_result is not None and qad_result[1] == 20


def test_find_latest_partial_skips_tmp_dirs(tmp_path: Path) -> None:
    """An aborted save leaves a `.tmp` sibling — must be ignored."""
    accel = _fake_accelerator()
    save_partial(
        _fake_model(),
        _fake_tokenizer(),
        accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=5,
        partial=True,
    )
    # Inject a `.tmp` sibling that even has a sentinel — must be ignored.
    bad = tmp_path / "kdr_bf16_partial_step99.tmp"
    bad.mkdir()
    (bad / SAVE_COMPLETE_SENTINEL).touch()

    result = find_latest_partial(tmp_path, "bf16")
    assert result is not None
    _, step = result
    assert step == 5


def test_find_latest_partial_skips_unparseable_step(tmp_path: Path) -> None:
    """A dir that matches the glob but whose suffix isn't an integer is
    skipped with a warning, not an exception."""
    bad = tmp_path / "kdr_bf16_partial_stepNaN"
    bad.mkdir()
    (bad / SAVE_COMPLETE_SENTINEL).touch()
    # mtime tweak to ensure ordering doesn't pick this up by accident.
    time.sleep(0.001)
    assert find_latest_partial(tmp_path, "bf16") is None
