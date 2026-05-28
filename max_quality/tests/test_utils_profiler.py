"""Tests for ``moe_compress.utils.profiler.SimpleProfilerMixin``.

Covers:

* Basic span recording (start < end, duration > 0).
* Nested spans (outer encompasses inner).
* Concurrent / overlapping spans with the same name (LIFO stack).
* JSON dump goes through Pattern O ``atomic_json_save`` and contains
  the documented schema (schema_version, sections, total_seconds).
* Thread-safety: two threads adding spans concurrently — all visible.
* ``format_profile_summary`` produces the expected sectioned text.
* ``reset_profile`` clears state.
* Error surfaces for invalid names / stop-without-start.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from moe_compress.utils.profiler import (
    PROFILER_SCHEMA_VERSION,
    SimpleProfilerMixin,
)


class _Subject(SimpleProfilerMixin):
    """Minimal concrete subclass used across tests."""


# ---------------------------------------------------------------------------
# Basic span recording.
# ---------------------------------------------------------------------------
def test_basic_span_records_positive_duration():
    subj = _Subject()
    with subj.profile("work"):
        # A short sleep so the duration is comfortably above the
        # perf_counter resolution floor on any platform.
        time.sleep(0.01)
    summary = subj.get_profile_summary()
    assert "work" in summary
    assert summary["work"]["count"] == 1
    assert summary["work"]["total_seconds"] > 0.0
    # Sanity: > sleep duration is not guaranteed on a busy CI, but a
    # 1ms lower bound rules out the perf_counter == 0 pathology.
    assert summary["work"]["total_seconds"] >= 1e-3
    assert summary["work"]["min_seconds"] == summary["work"]["max_seconds"]
    assert summary["work"]["mean_seconds"] == summary["work"]["total_seconds"]


def test_manual_start_stop_returns_duration():
    subj = _Subject()
    subj.start_profile("manual")
    time.sleep(0.005)
    d = subj.stop_profile("manual")
    assert d > 0.0
    summary = subj.get_profile_summary()
    assert summary["manual"]["count"] == 1
    assert summary["manual"]["total_seconds"] == pytest.approx(d, rel=1e-9)


def test_multiple_spans_accumulate():
    subj = _Subject()
    for _ in range(3):
        with subj.profile("repeat"):
            time.sleep(0.002)
    summary = subj.get_profile_summary()
    assert summary["repeat"]["count"] == 3
    assert summary["repeat"]["min_seconds"] <= summary["repeat"]["mean_seconds"]
    assert summary["repeat"]["mean_seconds"] <= summary["repeat"]["max_seconds"]
    assert summary["repeat"]["total_seconds"] == pytest.approx(
        summary["repeat"]["mean_seconds"] * 3, rel=1e-9
    )


# ---------------------------------------------------------------------------
# Nested spans.
# ---------------------------------------------------------------------------
def test_nested_spans_outer_includes_inner():
    """Outer span's wall-clock must be >= inner span's wall-clock."""
    subj = _Subject()
    with subj.profile("outer"):
        time.sleep(0.005)
        with subj.profile("inner"):
            time.sleep(0.010)
        time.sleep(0.005)
    summary = subj.get_profile_summary()
    assert summary["outer"]["count"] == 1
    assert summary["inner"]["count"] == 1
    assert summary["outer"]["total_seconds"] >= summary["inner"]["total_seconds"]
    # Outer wraps inner plus the two sleeps — must exceed inner alone.
    assert summary["outer"]["total_seconds"] > summary["inner"]["total_seconds"]


def test_same_name_can_nest_lifo():
    """Stacking the same span name uses LIFO ordering."""
    subj = _Subject()
    subj.start_profile("recur")
    time.sleep(0.002)
    subj.start_profile("recur")
    time.sleep(0.002)
    d_inner = subj.stop_profile("recur")
    d_outer = subj.stop_profile("recur")
    # The inner span (closed first) is the SHORTER of the two by
    # construction (started later).
    assert d_inner < d_outer
    summary = subj.get_profile_summary()
    assert summary["recur"]["count"] == 2


# ---------------------------------------------------------------------------
# Error handling.
# ---------------------------------------------------------------------------
def test_stop_without_start_raises():
    subj = _Subject()
    with pytest.raises(KeyError, match="no open span"):
        subj.stop_profile("ghost")


def test_empty_name_rejected():
    subj = _Subject()
    with pytest.raises(ValueError):
        subj.start_profile("")
    with pytest.raises(ValueError):
        subj.start_profile(None)  # type: ignore[arg-type]


def test_exception_in_span_still_closes():
    """If the wrapped block raises, the span MUST still be recorded."""
    subj = _Subject()
    with pytest.raises(RuntimeError, match="boom"):
        with subj.profile("explode"):
            time.sleep(0.002)
            raise RuntimeError("boom")
    summary = subj.get_profile_summary()
    assert summary["explode"]["count"] == 1
    assert summary["explode"]["total_seconds"] > 0.0


# ---------------------------------------------------------------------------
# JSON dump (Pattern O).
# ---------------------------------------------------------------------------
def test_dump_profile_json_writes_schema(tmp_path: Path):
    subj = _Subject()
    with subj.profile("a"):
        time.sleep(0.002)
    with subj.profile("b"):
        time.sleep(0.002)
    out = tmp_path / "sub" / "timings.json"
    written = subj.dump_profile_json(out)
    assert written == out
    assert out.exists()
    # No stray .tmp left behind — atomic_json_save invariant.
    assert not list(tmp_path.rglob("*.tmp"))
    blob = json.loads(out.read_text())
    assert blob["schema_version"] == PROFILER_SCHEMA_VERSION
    assert set(blob["sections"].keys()) == {"a", "b"}
    assert blob["total_seconds"] == pytest.approx(
        blob["sections"]["a"]["total_seconds"]
        + blob["sections"]["b"]["total_seconds"],
        rel=1e-9,
    )
    # Per-section schema keys.
    expected_keys = {"count", "total_seconds", "mean_seconds", "min_seconds", "max_seconds"}
    for stats in blob["sections"].values():
        assert set(stats.keys()) == expected_keys


def test_dump_profile_json_uses_atomic_io(tmp_path: Path, monkeypatch):
    """Verify the dump goes through ``atomic_json_save`` (Pattern O)."""
    import moe_compress.utils.profiler as prof_mod

    calls: list[tuple[Path, dict]] = []
    real_fn = prof_mod.atomic_json_save

    def spy(path, obj, **kw):
        calls.append((Path(path), obj))
        return real_fn(path, obj, **kw)

    monkeypatch.setattr(prof_mod, "atomic_json_save", spy)

    subj = _Subject()
    with subj.profile("x"):
        time.sleep(0.001)
    out = tmp_path / "t.json"
    subj.dump_profile_json(out)
    assert len(calls) == 1
    assert calls[0][0] == out
    assert calls[0][1]["schema_version"] == PROFILER_SCHEMA_VERSION


def test_dump_profile_json_empty(tmp_path: Path):
    """An instance with no spans still writes a valid JSON sidecar."""
    subj = _Subject()
    out = tmp_path / "empty.json"
    subj.dump_profile_json(out)
    blob = json.loads(out.read_text())
    assert blob == {
        "schema_version": PROFILER_SCHEMA_VERSION,
        "sections": {},
        "total_seconds": 0.0,
    }


# ---------------------------------------------------------------------------
# Thread-safety.
# ---------------------------------------------------------------------------
def test_thread_safety_two_threads_concurrent_spans():
    """Two threads recording many spans must not lose or corrupt any."""
    subj = _Subject()
    n_per_thread = 200
    barrier = threading.Barrier(2)

    def worker(name: str):
        barrier.wait()  # maximize concurrency
        for _ in range(n_per_thread):
            with subj.profile(name):
                # Negligible work — we want the lock to be hammered, not
                # the wall-clock.
                pass

    t1 = threading.Thread(target=worker, args=("alpha",))
    t2 = threading.Thread(target=worker, args=("beta",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    summary = subj.get_profile_summary()
    assert summary["alpha"]["count"] == n_per_thread
    assert summary["beta"]["count"] == n_per_thread


def test_thread_safety_shared_section_name():
    """Two threads sharing a section name accumulate cleanly."""
    subj = _Subject()
    n_per_thread = 150
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        for _ in range(n_per_thread):
            with subj.profile("shared"):
                pass

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    summary = subj.get_profile_summary()
    assert summary["shared"]["count"] == 2 * n_per_thread
    # No span should be left open after both threads return.
    assert subj._profile_open == {}


# ---------------------------------------------------------------------------
# print_profile_summary / format_profile_summary.
# ---------------------------------------------------------------------------
def test_format_profile_summary_layout():
    subj = _Subject()
    with subj.profile("fast"):
        pass
    with subj.profile("slow"):
        time.sleep(0.01)
    text = subj.format_profile_summary(title="Stage X timings")
    # Title appears as the first line.
    assert text.splitlines()[0] == "Stage X timings"
    # Both section names appear.
    assert "fast" in text
    assert "slow" in text
    # TOTAL row is present.
    assert "TOTAL" in text
    # Header columns documented in the docstring.
    for col in ("section", "count", "total", "mean", "max"):
        assert col in text
    # Slower section sorts above the faster one.
    slow_idx = text.find("slow")
    fast_idx = text.find("fast")
    assert slow_idx < fast_idx


def test_format_profile_summary_default_title():
    subj = _Subject()
    text = subj.format_profile_summary()
    assert text.splitlines()[0] == "Profiler summary"
    assert "(no spans recorded)" in text


def test_print_profile_summary_logs_at_info(caplog):
    """print_profile_summary uses the module logger at INFO."""
    import logging

    import moe_compress.utils.profiler as prof_mod

    subj = _Subject()
    with subj.profile("logme"):
        pass

    # Pattern N (architectural-patterns): non-root module loggers may
    # have propagate flipped by caplog; preserve + restore so the
    # assertion is reliable.
    prev_propagate = prof_mod.log.propagate
    prof_mod.log.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger=prof_mod.log.name):
            subj.print_profile_summary("ZZ")
    finally:
        prof_mod.log.propagate = prev_propagate

    msgs = [r.getMessage() for r in caplog.records if r.name == prof_mod.log.name]
    assert any("ZZ" in m and "logme" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# reset_profile.
# ---------------------------------------------------------------------------
def test_reset_profile_clears_state():
    subj = _Subject()
    with subj.profile("first"):
        pass
    subj.start_profile("never-closed")  # open span
    subj.reset_profile()
    assert subj.get_profile_summary() == {}
    assert subj._profile_open == {}
    # And we can keep using it after reset.
    with subj.profile("after"):
        pass
    assert "after" in subj.get_profile_summary()
    assert "first" not in subj.get_profile_summary()


# ---------------------------------------------------------------------------
# Independent instances do not share state.
# ---------------------------------------------------------------------------
def test_instances_are_independent():
    a = _Subject()
    b = _Subject()
    with a.profile("a-only"):
        pass
    assert "a-only" in a.get_profile_summary()
    assert b.get_profile_summary() == {}
