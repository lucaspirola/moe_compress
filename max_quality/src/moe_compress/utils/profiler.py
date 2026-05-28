"""Stage-/plugin-level wall-clock profiler mixin.

A lightweight, dependency-free mixin that lets any stage orchestrator or
plugin record named wall-clock spans with a context manager, print a
per-section breakdown at stage close, and dump a JSON sidecar for
downstream consumption (e.g. the A0..A11 ablation grid's per-stage
time-attribution column).

Pattern H — clean-room re-implementation
----------------------------------------
This module re-implements the API surface of
``fusion_bench.mixins.simple_profiler.SimpleProfilerMixin`` (MIT-licensed
upstream — github.com/tanganke/fusion_bench, ``fusion_bench/mixins/
simple_profiler.py``, accessed 2026-05-28) WITHOUT vendoring any source.
The upstream version delegates to ``lightning.pytorch.profilers.SimpleProfiler``
under the hood, which would drag in the full Lightning runtime as a
dependency. We intentionally re-derive the behavior from the upstream
docstring + public API (the ``profile`` context manager, ``start_profile``
/ ``stop_profile`` pair, and ``print_profile_summary``) using only stdlib
``time.perf_counter`` and ``threading.Lock``. Per-primitive citation
classification: **D-clean-room** (no verbatim vendoring, no license import,
no Lightning runtime).

Deviations from upstream — documented intentionally so reviewers can
diff against fusion_bench cleanly:

* No Lightning dependency — backed by ``time.perf_counter()`` directly.
* No ``@rank_zero_only`` on the summary printer — rank gating is a
  caller concern in this codebase; the mixin must remain runnable from
  unit-tests without a distributed init.
* Thread-safe (Pattern: ``threading.Lock`` around all mutations of the
  timings dict). Upstream relies on Lightning's profiler which is NOT
  thread-safe; we explicitly tighten the contract here because Stage 2
  plugins use ``futures.py`` worker pools.
* JSON dump via :func:`utils.atomic_io.atomic_json_save` (Pattern O,
  architectural-patterns) — never bare ``json.dump`` to disk in the
  calibration pipeline.
* Concurrent / overlapping spans with the SAME name are supported via a
  per-name stack of start-times (LIFO). Upstream's Lightning profiler
  errors on a double-start; we lift that restriction because nested
  recursion (e.g. ``profile("merge_step")`` inside a recursive helper
  also wrapped in ``profile("merge_step")``) is a real use case in the
  per-layer merging stages.

Usage
-----
::

    class Stage2Orchestrator(SimpleProfilerMixin):
        def run(self, ctx):
            with self.profile("load_calibration"):
                ctx = load_calibration(ctx)
            with self.profile("merge_step"):
                ctx = merge(ctx)
            self.print_profile_summary("Stage 2 timings")
            self.dump_profile_json(ctx.artifacts_dir / "stage2_timings.json")

The JSON sidecar schema is::

    {
      "schema_version": 1,
      "sections": {
        "load_calibration": {
          "count":         <int>,    # how many spans were recorded
          "total_seconds": <float>,  # sum of span durations
          "mean_seconds":  <float>,
          "min_seconds":   <float>,
          "max_seconds":   <float>
        },
        ...
      },
      "total_seconds": <float>       # sum across all sections
    }

Forward-only schema bumps (Pattern K).
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from .atomic_io import atomic_json_save

log = logging.getLogger(__name__)


__all__ = [
    "SimpleProfilerMixin",
    "PROFILER_SCHEMA_VERSION",
]


# Forward-only (Pattern K). Bump on additive field changes; never break
# old readers. Consumers MUST tolerate older schemas via .get(...).
PROFILER_SCHEMA_VERSION = 1


class SimpleProfilerMixin:
    """Mixin that adds named wall-clock spans + summary + JSON dump.

    State is held in two per-instance dicts:

    * ``_profile_durations``: ``{name: list[float]}`` — completed spans.
    * ``_profile_open``: ``{name: list[float]}`` — open spans' start
      times, kept as a stack so the same name can nest (LIFO close).

    Both dicts are guarded by ``_profile_lock`` — a re-entrant lock is
    NOT required because no public method calls another holding the
    lock; the lock scope is intentionally tight.

    Lazy init: dicts and lock are created on first access via
    :meth:`_ensure_profile_state` so subclasses do not need to remember
    to call ``super().__init__()``. This matches fusion_bench's lazy
    ``profiler`` property idiom.
    """

    # Class-level sentinels so static type checkers see the attributes
    # without requiring an __init__. Re-bound to per-instance objects on
    # first use; never read directly.
    _profile_durations: dict[str, list[float]] | None = None
    _profile_open: dict[str, list[float]] | None = None
    _profile_lock: threading.Lock | None = None

    # ------------------------------------------------------------------
    # Internal: lazy state init.
    # ------------------------------------------------------------------
    def _ensure_profile_state(self) -> None:
        """Idempotent: bind per-instance dicts + lock on first call.

        Safe to call from multiple threads — the lock-creation race is
        benign because every method that touches the state calls this
        first, and the worst case is a brief double-allocation that
        the GIL serializes.
        """
        # We check via the instance ``__dict__`` to distinguish "never
        # initialized on this instance" from "inherited the class-level
        # None sentinel". Once bound, the instance dict shadows the
        # class attribute.
        if "_profile_lock" not in self.__dict__ or self._profile_lock is None:
            self._profile_lock = threading.Lock()
        if "_profile_durations" not in self.__dict__ or self._profile_durations is None:
            self._profile_durations = {}
        if "_profile_open" not in self.__dict__ or self._profile_open is None:
            self._profile_open = {}

    # ------------------------------------------------------------------
    # Public API — start/stop and context manager.
    # ------------------------------------------------------------------
    def start_profile(self, name: str) -> None:
        """Begin a span. Must be paired with :meth:`stop_profile`.

        Multiple concurrent spans with the same name are allowed; they
        form a LIFO stack and each ``stop_profile(name)`` closes the
        most recent open span for that name.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"start_profile: name must be a non-empty str, got {name!r}"
            )
        self._ensure_profile_state()
        # perf_counter() OUTSIDE the lock — minimizes hold time so the
        # measurement is not contaminated by lock contention.
        t = time.perf_counter()
        assert self._profile_lock is not None  # for type checkers
        with self._profile_lock:
            assert self._profile_open is not None
            self._profile_open.setdefault(name, []).append(t)

    def stop_profile(self, name: str) -> float:
        """End the most recently opened span named ``name``.

        Returns the duration in seconds. Raises :class:`KeyError` if no
        span with this name is open — that is a programming bug, not a
        runtime condition, so we surface loudly per the no-silent-fallback
        rule.
        """
        # perf_counter() OUTSIDE the lock.
        t_end = time.perf_counter()
        self._ensure_profile_state()
        assert self._profile_lock is not None
        with self._profile_lock:
            assert self._profile_open is not None
            assert self._profile_durations is not None
            stack = self._profile_open.get(name)
            if not stack:
                raise KeyError(
                    f"stop_profile({name!r}): no open span. start_profile must "
                    f"be called before stop_profile (or the profile() context "
                    f"manager used)."
                )
            t_start = stack.pop()
            # Clean up empty stacks so the open-dict never grows.
            if not stack:
                del self._profile_open[name]
            duration = t_end - t_start
            self._profile_durations.setdefault(name, []).append(duration)
        return duration

    @contextmanager
    def profile(self, name: str) -> Generator[str, None, None]:
        """Context manager that wraps :meth:`start_profile` /
        :meth:`stop_profile`.

        Yields ``name`` for convenience (caller can ``with self.profile("x") as n``
        and use ``n`` for logging). The span is closed in the ``finally``
        block so exceptions raised inside the ``with`` still record
        their elapsed time before propagating — useful for diagnosing
        slow failures.
        """
        self.start_profile(name)
        try:
            yield name
        finally:
            # If something went VERY wrong (e.g. the span was already
            # closed manually), surface the original exception, not the
            # KeyError from stop_profile.
            try:
                self.stop_profile(name)
            except KeyError:
                log.warning(
                    "profile(%r) context-exit: span already closed; ignoring",
                    name,
                )

    # ------------------------------------------------------------------
    # Public API — summary + JSON dump.
    # ------------------------------------------------------------------
    def get_profile_summary(self) -> dict[str, dict[str, float]]:
        """Return a snapshot of the per-section statistics.

        Keys (per section name):

        * ``count`` (int — number of completed spans)
        * ``total_seconds`` (float)
        * ``mean_seconds`` (float — undefined / 0.0 if count==0)
        * ``min_seconds`` (float)
        * ``max_seconds`` (float)

        Open (un-closed) spans are NOT included — caller must close
        them first if they want them reflected.
        """
        self._ensure_profile_state()
        assert self._profile_lock is not None
        with self._profile_lock:
            assert self._profile_durations is not None
            # Snapshot copy under the lock — the returned dict is owned
            # by the caller and safe to mutate outside the lock.
            durations_copy = {
                name: list(samples)
                for name, samples in self._profile_durations.items()
            }
        out: dict[str, dict[str, float]] = {}
        for name, samples in durations_copy.items():
            if not samples:
                # Defensive: should never happen because we only insert
                # on a successful stop_profile.
                out[name] = {
                    "count": 0,
                    "total_seconds": 0.0,
                    "mean_seconds": 0.0,
                    "min_seconds": 0.0,
                    "max_seconds": 0.0,
                }
                continue
            total = sum(samples)
            out[name] = {
                "count": len(samples),
                "total_seconds": total,
                "mean_seconds": total / len(samples),
                "min_seconds": min(samples),
                "max_seconds": max(samples),
            }
        return out

    def format_profile_summary(self, title: str | None = None) -> str:
        """Return the human-readable summary as a string.

        Format (fixed-width columns; suitable for ``print()`` or
        ``log.info("\\n%s", summary)``)::

            Profiler summary
            ----------------------------------------------------------------
            section                       count       total       mean       max
            ----------------------------------------------------------------
            load_calibration                  1     12.345 s   12.345 s  12.345 s
            merge_step                       12      4.000 s    0.333 s   0.500 s
            ----------------------------------------------------------------
            TOTAL                           ---     16.345 s

        Sections are sorted by ``total_seconds`` descending (slowest
        first) so the bottleneck pops to the top.
        """
        summary = self.get_profile_summary()
        lines: list[str] = []
        if title:
            lines.append(str(title))
        else:
            lines.append("Profiler summary")
        sep = "-" * 80
        lines.append(sep)
        header = f"{'section':<32} {'count':>6} {'total':>12} {'mean':>12} {'max':>12}"
        lines.append(header)
        lines.append(sep)
        if not summary:
            lines.append("(no spans recorded)")
        else:
            sorted_items = sorted(
                summary.items(), key=lambda kv: kv[1]["total_seconds"], reverse=True
            )
            grand_total = 0.0
            for name, stats in sorted_items:
                grand_total += stats["total_seconds"]
                lines.append(
                    f"{name[:32]:<32} {int(stats['count']):>6d} "
                    f"{stats['total_seconds']:>10.3f} s "
                    f"{stats['mean_seconds']:>10.3f} s "
                    f"{stats['max_seconds']:>10.3f} s"
                )
            lines.append(sep)
            lines.append(f"{'TOTAL':<32} {'---':>6} {grand_total:>10.3f} s")
        return "\n".join(lines)

    def print_profile_summary(self, title: str | None = None) -> None:
        """Print the per-section summary table to the module logger.

        Logs at INFO level via this module's logger
        (``moe_compress.utils.profiler``) rather than ``print()`` so the
        output is captured by the project's existing log handlers and
        appears in the stage log files.
        """
        log.info("\n%s", self.format_profile_summary(title=title))

    def dump_profile_json(self, path: str | Path) -> Path:
        """Atomically write the JSON sidecar.

        Uses :func:`utils.atomic_io.atomic_json_save` per Pattern O
        (architectural-patterns) — never bare ``json.dump``. The on-disk
        schema is documented at the top of this module; reads MUST
        check ``schema_version`` and tolerate older versions.

        Returns the resolved final path on success.
        """
        summary = self.get_profile_summary()
        total = sum(stats["total_seconds"] for stats in summary.values())
        payload: dict[str, Any] = {
            "schema_version": PROFILER_SCHEMA_VERSION,
            "sections": summary,
            "total_seconds": total,
        }
        return atomic_json_save(path, payload, indent=2, sort_keys=True)

    def reset_profile(self) -> None:
        """Drop all recorded + open spans.

        Useful between phases when the caller wants per-phase JSON
        sidecars rather than a cumulative one.
        """
        self._ensure_profile_state()
        assert self._profile_lock is not None
        with self._profile_lock:
            assert self._profile_durations is not None
            assert self._profile_open is not None
            self._profile_durations.clear()
            self._profile_open.clear()
