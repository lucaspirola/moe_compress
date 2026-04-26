"""Tiny lazy wrapper around ``trackio.log``.

Each stage module imports ``trackio_log`` from here and calls it freely; if the
``trackio`` package isn't installed, or ``trackio.init`` was never called (e.g.
local smoke tests), the call is a silent no-op. Keeps stage code uncluttered
with try/except blocks while preserving the graceful-fallback contract from
``hf_jobs/entrypoint.py:_init_trackio``.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_trackio = None
# Two independent flags so a "trackio not installed" debug message doesn't
# silence subsequent "trackio.log failed" warnings (different failure modes,
# operator wants to see both at least once).
_warned_missing = False
_warned_log_failed = False


def _resolve():
    global _trackio, _warned_missing
    if _trackio is not None:
        return _trackio
    try:
        import trackio
        _trackio = trackio
        return _trackio
    except ImportError:
        if not _warned_missing:
            log.debug("trackio not installed — trackio_log is a no-op")
            _warned_missing = True
        return None


def trackio_log(metrics: dict[str, Any]) -> None:
    """Push a dict of scalar metrics to Trackio. Silent no-op on any failure."""
    mod = _resolve()
    if mod is None:
        return
    try:
        mod.log(metrics)
    except Exception as exc:                         # noqa: BLE001
        # One warning per process for log failures — further failures
        # suppressed (dashboard error or run-finished race shouldn't spam).
        global _warned_log_failed
        if not _warned_log_failed:
            log.warning("trackio.log failed (%s) — further failures suppressed", exc)
            _warned_log_failed = True
