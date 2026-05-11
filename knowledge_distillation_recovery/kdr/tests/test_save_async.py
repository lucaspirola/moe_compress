"""Phase 7+ async save_partial tests (LLR-0027 v2).

Covers the new `async_mode=True` semantics added in Phase C:

- async dispatches the disk write to a single-flight background thread,
- the next save_partial auto-joins the prior pending Future,
- save_partial_join() flushes any pending Future and re-raises exceptions,
- async_mode=True with partial=False is rejected (final save must be sync),
- the on-disk byte layout is identical to the sync path (LLR-0029 sentinel,
  atomic rename ordering all preserved).

These tests use the same MagicMock-based accelerator / model / tokenizer
fixtures as `test_save_resume.py`. The background thread is real (a
`ThreadPoolExecutor` with `max_workers=1`); each test calls
`save_partial_join()` in teardown to guarantee no Future bleeds across
tests.

# VERIFIES: LLR-0027 (v2 async semantics)
# VERIFIES: LLR-0029 (sentinel-last invariant preserved in async path)
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kdr.io.save import (
    SAVE_COMPLETE_SENTINEL,
    _reset_async_save_executor,
    save_partial,
    save_partial_join,
)


def _fake_accelerator(*, is_main: bool = True) -> MagicMock:
    accel = MagicMock()
    accel.is_main_process = is_main
    accel.wait_for_everyone = MagicMock()
    accel.get_state_dict = MagicMock(return_value={})
    accel.unwrap_model = lambda m: m
    return accel


def _fake_model(write_delay_s: float = 0.0) -> MagicMock:
    """A model whose `save_pretrained` writes a fake shard and optionally
    sleeps. The sleep is used to assert async dispatch returns BEFORE the
    background thread finishes."""
    m = MagicMock()

    def _save(out_dir: Path, **kw: object) -> None:
        if write_delay_s:
            time.sleep(write_delay_s)
        out_dir = Path(out_dir)
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


@pytest.fixture(autouse=True)
def _reset_executor() -> Iterator[None]:
    """Reset the module-global async save executor BEFORE and AFTER each
    test — prevents cross-test leakage of pending Futures.

    Pre-test reset: clears whatever the previous test left behind.
    Post-test reset (`yield` then reset): ensures THIS test's own
    pending Future does not bleed into the next test. Important under
    test-failure scenarios: if a test fails between submitting an async
    write and joining it, the next test would otherwise see a leaked
    `_pending` Future plus a background thread still writing to this
    test's (deleted) `tmp_path`.
    """
    _reset_async_save_executor()
    yield
    _reset_async_save_executor()


# ─── LLR-0027 v2: async dispatch returns before disk write completes ─────


def test_async_mode_returns_before_disk_write_completes(tmp_path: Path) -> None:
    """The whole point of async: control returns to the caller as soon as
    the collective `get_state_dict` is done, while the disk write runs in
    a background thread. Verified by a delayed-write fake model: if async
    were synchronous, the call would block ≥ delay; instead it returns
    immediately."""
    delay = 0.3
    accel = _fake_accelerator()
    model = _fake_model(write_delay_s=delay)
    tok = _fake_tokenizer()

    t0 = time.monotonic()
    out_dir = save_partial(
        model, tok, accel,
        artifacts_dir=tmp_path,
        mode="bf16",
        step=10,
        partial=True,
        async_mode=True,
    )
    dispatch_dt = time.monotonic() - t0
    # Allow generous slack — Python startup overhead can vary across CI
    # boxes, but it must be well under the artificial `delay` injected.
    assert dispatch_dt < delay * 0.5, (
        f"save_partial(async_mode=True) blocked for {dispatch_dt:.2f}s "
        f"vs delayed-write delay {delay}s — async dispatch is not running "
        f"in a background thread."
    )
    # The dir doesn't exist yet (write is still pending); join to flush.
    save_partial_join()
    # After join, the sentinel must exist (LLR-0029 invariant preserved).
    sentinel = out_dir / SAVE_COMPLETE_SENTINEL
    assert sentinel.exists()
    assert os.path.getsize(sentinel) == 0


def test_async_mode_byte_identical_to_sync(tmp_path: Path) -> None:
    """LLR-0027 AC: async path's on-disk layout is byte-identical to sync.
    Compare the file set produced by both paths."""
    accel = _fake_accelerator()

    # Sync save at /sync subdir
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    save_partial(
        _fake_model(), _fake_tokenizer(), accel,
        artifacts_dir=sync_dir, mode="bf16", step=10, partial=True,
        async_mode=False,
    )
    sync_files = sorted(
        p.name for p in (sync_dir / "kdr_bf16_partial_step10").iterdir()
    )

    # Async save at /async subdir
    async_dir = tmp_path / "async"
    async_dir.mkdir()
    save_partial(
        _fake_model(), _fake_tokenizer(), accel,
        artifacts_dir=async_dir, mode="bf16", step=10, partial=True,
        async_mode=True,
    )
    save_partial_join()
    async_files = sorted(
        p.name for p in (async_dir / "kdr_bf16_partial_step10").iterdir()
    )

    assert sync_files == async_files, (
        f"sync produced {sync_files}, async produced {async_files} — "
        f"file set divergence violates LLR-0027 AC."
    )


# ─── LLR-0027 v2: async_mode with partial=False is rejected ──────────────


def test_async_mode_rejected_for_final_save(tmp_path: Path) -> None:
    """Final save (partial=False) must be synchronous because its return
    Path is consumed immediately by the upload step."""
    accel = _fake_accelerator()
    with pytest.raises(ValueError, match="async_mode=True.*partial=False"):
        save_partial(
            _fake_model(), _fake_tokenizer(), accel,
            artifacts_dir=tmp_path, mode="bf16", step=10,
            partial=False, async_mode=True,
        )


# ─── LLR-0027 v2: single-flight queue auto-joins prior Future ────────────


def test_single_flight_queue_auto_joins_prior(tmp_path: Path) -> None:
    """When save_partial(async_mode=True) is called and a prior Future is
    pending, the new call MUST auto-join the prior before submitting. This
    guarantees monotone partial ordering on disk."""
    accel = _fake_accelerator()
    # First async call — long delay so the second submit catches it pending.
    save_partial(
        _fake_model(write_delay_s=0.4), _fake_tokenizer(), accel,
        artifacts_dir=tmp_path, mode="bf16", step=10, partial=True,
        async_mode=True,
    )
    # At this point the first write is still in flight. The second call
    # should not return until the first is joined.
    t0 = time.monotonic()
    save_partial(
        _fake_model(write_delay_s=0.0), _fake_tokenizer(), accel,
        artifacts_dir=tmp_path, mode="bf16", step=20, partial=True,
        async_mode=True,
    )
    submit2_dt = time.monotonic() - t0
    # The second submit had to wait for the first to finish (~0.4s delay).
    assert submit2_dt >= 0.3, (
        f"second async save_partial returned in {submit2_dt:.2f}s — "
        f"auto-join of prior Future did not block, violating monotone "
        f"partial ordering invariant."
    )
    save_partial_join()
    # Both partials are on disk after final join.
    assert (tmp_path / "kdr_bf16_partial_step10" / SAVE_COMPLETE_SENTINEL).exists()
    assert (tmp_path / "kdr_bf16_partial_step20" / SAVE_COMPLETE_SENTINEL).exists()


# ─── LLR-0027 v2: background exception propagates at join ────────────────


def test_background_exception_propagates_at_join(tmp_path: Path) -> None:
    """If the background disk-write raises, the exception MUST surface at
    the next save_partial_join() call (and at the next save_partial async
    submit). Silently dropping the exception would corrupt the resume
    contract."""
    accel = _fake_accelerator()
    bad_model = MagicMock()
    bad_model.save_pretrained.side_effect = RuntimeError(
        "synthetic disk-write failure"
    )

    save_partial(
        bad_model, _fake_tokenizer(), accel,
        artifacts_dir=tmp_path, mode="bf16", step=10, partial=True,
        async_mode=True,
    )
    # Exception must surface at the join site.
    with pytest.raises(RuntimeError, match="synthetic disk-write failure"):
        save_partial_join()


def test_background_exception_propagates_at_next_submit(tmp_path: Path) -> None:
    """If the user does NOT call save_partial_join() between failures,
    the next async submit auto-joins the prior Future and re-raises."""
    accel = _fake_accelerator()
    bad_model = MagicMock()
    bad_model.save_pretrained.side_effect = RuntimeError(
        "synthetic disk-write failure"
    )

    save_partial(
        bad_model, _fake_tokenizer(), accel,
        artifacts_dir=tmp_path, mode="bf16", step=10, partial=True,
        async_mode=True,
    )
    # The next submit auto-joins the prior failed Future and re-raises.
    with pytest.raises(RuntimeError, match="synthetic disk-write failure"):
        save_partial(
            _fake_model(), _fake_tokenizer(), accel,
            artifacts_dir=tmp_path, mode="bf16", step=20, partial=True,
            async_mode=True,
        )


# ─── LLR-0027 v2: save_partial_join() is a no-op when nothing pending ────


def test_join_noop_when_nothing_pending() -> None:
    """save_partial_join() must be safe to call when no Future is in
    flight — the trainer calls it before final save regardless of whether
    any async save_partial was previously dispatched."""
    save_partial_join()  # Should not raise.
    save_partial_join()  # Idempotent.


# ─── LLR-0027 v2: sync path unchanged (regression guard) ─────────────────


def test_sync_path_unchanged_by_async_refactor(tmp_path: Path) -> None:
    """Default `async_mode=False` produces the same disk artifact as the
    pre-Phase-C sync-only `save_partial`. Regression guard: if a future
    refactor accidentally rewrites the sync path through the executor,
    this would catch it (the test would block waiting for a Future)."""
    accel = _fake_accelerator()
    t0 = time.monotonic()
    out = save_partial(
        _fake_model(write_delay_s=0.2), _fake_tokenizer(), accel,
        artifacts_dir=tmp_path, mode="bf16", step=10, partial=True,
        # async_mode omitted → defaults to False
    )
    sync_dt = time.monotonic() - t0
    # Sync call MUST block for the write delay (or longer).
    assert sync_dt >= 0.15, (
        f"sync save_partial returned in {sync_dt:.2f}s but the model's "
        f"save_pretrained had a 0.2s delay — sync path may have been "
        f"accidentally routed through the async executor."
    )
    # On-disk state is correct.
    assert (out / SAVE_COMPLETE_SENTINEL).exists()
