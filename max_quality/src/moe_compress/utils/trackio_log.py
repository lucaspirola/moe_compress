"""Tiny lazy wrapper around ``trackio.log``.

Each stage module imports ``trackio_log`` from here and calls it freely; if the
``trackio`` package isn't installed, or ``trackio.init`` was never called (e.g.
local smoke tests), the call is a silent no-op.  Keeps stage code uncluttered
with try/except blocks while preserving the graceful-fallback contract from
``hf_jobs/entrypoint.py:_init_trackio``.

Threading contract
------------------
*Import-tracking state* (``_import_failed``, ``_warned_missing``,
``_warned_import_exception``, ``_warned_log_bad_attr_exc``,
``_warned_log_bad_attr_noncallable``) — protected by ``_lock``; all reads
and writes go through the lock so concurrent first calls are safe.
``_warned_import_exception`` gates the one-time warning for unexpected
exceptions raised during the ``import trackio`` step (i.e. not an
``ImportError``).

Two distinct synchronization mechanisms are in play: importlib's per-module
lock serializes the ``import trackio`` statement itself (preventing duplicate
module execution), while ``_lock`` (a ``threading.Lock``) is a separate
mechanism that protects the flag updates (``_trackio``, ``_import_failed``,
etc.) that happen *after* the import completes.  Do not conflate them.

*Logging-failure tracking state* (``_warned_log_failed``) — also protected by
``_lock``.  Each invocation of ``trackio_log`` still attempts ``mod.log()``
regardless of this flag; only the one-time warning is gated behind the flag.
The *first* ``mod.log()`` failure in the process lifetime emits a warning;
subsequent failures (including any post-recovery failures) are always silently
swallowed.

``mod.log()`` itself is **not** thread-safe — ``trackio.log`` may use
non-thread-safe state internally; treat the ``mod.log()`` call as
non-thread-safe.  This wrapper does **not** make it safe to call from
background threads.  Callers must ensure they invoke ``trackio_log`` from the
main thread only.

Import-failure policy
---------------------
``_import_failed`` is a permanent latch: once set, ``_import_trackio()``
returns ``None`` immediately on every subsequent call without re-attempting the
import.  Contrast with ``mod.log()`` failures: the *first* failure emits a
one-time warning, and all subsequent failures (including any post-recovery
ones) are silently swallowed — the call is retried on every invocation but
errors never propagate to the caller.  Note: ``trackio_log`` itself may also
latch ``_import_failed`` (e.g., when ``mod.log`` is found non-callable after a
successful import).
"""
from __future__ import annotations

import logging
import threading
import types
from typing import Any

_log = logging.getLogger(__name__)

_lock = threading.Lock()

# Assumption: once imported, trackio is never removed from sys.modules at
# runtime.  The cached reference below therefore stays valid for the process
# lifetime.
_trackio: types.ModuleType | None = None

# Import-failure state: _import_failed is the permanent latch; the warn flags
# are independent so a warning for one failure mode cannot silence warnings for
# the other.  Operator should see each category at least once.
# Note: _import_failed may also be latched by trackio_log (not just
# _import_trackio) when mod.log is found non-callable after a successful import.
_import_failed: bool = False
_warned_missing: bool = False           # covers: ImportError (package absent or broken dep)
_warned_import_exception: bool = False  # covers: unexpected Exception during import
_warned_log_bad_attr_exc: bool = False          # covers: getattr(mod, 'log') raised an exception
_warned_log_bad_attr_noncallable: bool = False  # covers: trackio installed but mod.log non-callable (version/install issue)

# Runtime log-failure flags:
_warned_log_failed: bool = False
_warned_flush_no_run: bool = False
_warned_flush_failed: bool = False


def _import_trackio() -> types.ModuleType | None:
    """Return the trackio module if available, else None.

    The lock protects flag reads/writes only; the import itself runs outside
    the lock so we never hold the lock during I/O.

    Relies on the interpreter's per-module import lock to prevent concurrent
    import; safe under CPython with GIL and under Python 3.13 free-threaded
    builds.
    """
    global _trackio, _import_failed, _warned_missing, _warned_import_exception

    with _lock:
        if _trackio is not None:
            return _trackio
        if _import_failed:
            return None

    # Import happens outside the lock to avoid blocking other threads during
    # the (potentially slow) import.
    try:
        import trackio  # noqa: PLC0415 — lazy import by design
        with _lock:
            if _trackio is None and not _import_failed:   # re-check under lock
                _trackio = trackio
            # Prefer the locally-imported module to guard against a TOCTOU race
            # where a concurrent except-branch set _import_failed before we
            # re-acquired the lock, leaving _trackio as None despite a successful
            # import on this thread.
            result = _trackio if _trackio is not None else trackio
        return result
    except ImportError:
        with _lock:
            _import_failed = True
            should_warn = not _warned_missing
            if should_warn:
                _warned_missing = True
        if should_warn:  # local var; safe to read outside lock
            _log.debug("trackio not installed — trackio_log is a no-op")
        return None
    except Exception:
        with _lock:
            _import_failed = True
            should_warn = not _warned_import_exception
            if should_warn:
                _warned_import_exception = True
        if should_warn:  # local var; safe to read outside lock
            _log.warning("trackio import failed unexpectedly", exc_info=True)
        return None


def trackio_log(metrics: dict[str, Any]) -> None:
    """Push a dict of scalar metrics to Trackio.

    If the loaded trackio module exposes no callable ``log`` attribute, latches
    ``_import_failed = True`` and emits a one-time ``log.warning``, then
    returns (silent no-op — consistent with the install-issue contract).
    Raises ``TypeError`` if ``metrics`` is not a dict (programming error).
    All other exceptions from inside ``mod.log()`` are caught and warned once,
    then silently swallowed on subsequent calls (silent no-op contract).

    Must be called from the main thread — ``trackio.log`` is not thread-safe.
    Raises ``RuntimeError`` if called from any non-main thread (F-C-N-1: enforce
    the contract that was previously documented but not checked).
    """
    global _import_failed, _trackio, _warned_log_failed, _warned_log_bad_attr_exc, _warned_log_bad_attr_noncallable

    # F-C-N-1: enforce the main-thread contract documented above.
    if threading.main_thread() is not threading.current_thread():
        raise RuntimeError(
            "trackio_log must be called from the main thread "
            f"(called from {threading.current_thread().name!r}); "
            "trackio.log is not thread-safe."
        )

    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be a dict, got {type(metrics).__name__!r}")

    mod = _import_trackio()
    if mod is None:
        return
    try:
        log_attr = getattr(mod, "log", None)
    except Exception as _exc:
        with _lock:
            _import_failed = True
            _trackio = None   # clear cached module ref so future callers see None once latch is set
            should_warn = not _warned_log_bad_attr_exc
            if should_warn:
                _warned_log_bad_attr_exc = True
        if should_warn:
            _log.warning("trackio: getattr(mod, 'log') raised %r — disabling trackio", _exc)
        return
    if not callable(log_attr):
        with _lock:
            _import_failed = True
            _trackio = None   # clear cached module ref so future callers see None once latch is set
            should_warn = not _warned_log_bad_attr_noncallable
            if should_warn:
                _warned_log_bad_attr_noncallable = True
        if should_warn:  # local var; safe to read outside lock
            _log.warning(
                "trackio has no callable 'log' attribute (check trackio version/installation): %r; "
                "disabling trackio logging permanently",
                mod,
            )
        return
    try:
        mod.log(metrics)
    except Exception:  # noqa: BLE001 — broad catch intentional; runtime errors from trackio are non-fatal
        # Retry-forever design: each call still attempts mod.log() on every
        # invocation.  Only the *first* failure in the process lifetime emits a
        # warning; all subsequent failures (including any post-recovery ones)
        # are silently swallowed so dashboard errors don't spam the log.
        with _lock:
            already_warned = _warned_log_failed
            if not already_warned:
                _warned_log_failed = True
        if not already_warned:  # local var; safe to read outside lock
            _log.warning("trackio.log failed; subsequent log failures will not emit a warning", exc_info=True)


def trackio_flush() -> None:
    """Best-effort drain of the active Run's queue + respawn of its sender thread.

    Trackio batches log emits and ships them to the remote Space every
    ``BATCH_SEND_INTERVAL`` (~0.5 s) via a background thread. If that thread
    dies silently, queued emits stop reaching the dashboard with no error
    surfaced anywhere. Calling this helper at known cadence points (e.g. the
    Stage 1 per-batch progress callback, or end-of-phase summary emits)
    ensures (a) the sender thread is alive and (b) the in-process queue is
    drained.

    Reaches into private trackio internals via getattr/no-op fallback:
    ``Run._ensure_sender_alive`` (respawns dead sender) and
    ``Run._flush_queues_inline`` (drains queues; for remote Runs this writes
    to the local-fallback file rather than the Space, but it still relieves
    queue pressure and makes the data recoverable on the next ``trackio.init``).

    Active-run lookup uses the same ContextVar that ``trackio.log`` itself
    uses (``trackio.context_vars.current_run``), so it stays in sync with the
    library's own notion of "current run".

    Must be called from the main thread (mirrors ``trackio_log``'s contract —
    trackio is not thread-safe).
    """
    global _warned_flush_no_run, _warned_flush_failed

    if threading.main_thread() is not threading.current_thread():
        raise RuntimeError(
            "trackio_flush must be called from the main thread "
            f"(called from {threading.current_thread().name!r}); "
            "trackio.log is not thread-safe."
        )

    mod = _import_trackio()
    if mod is None:
        return

    try:
        ctx = getattr(mod, "context_vars", None)
        run = ctx.current_run.get() if ctx is not None else None
    except Exception:  # noqa: BLE001 — defensive against trackio refactors
        run = None

    if run is None:
        with _lock:
            should_warn = not _warned_flush_no_run
            if should_warn:
                _warned_flush_no_run = True
        if should_warn:
            _log.debug(
                "trackio_flush: no active Run "
                "(trackio.init() not called yet?) — silent no-op"
            )
        return

    try:
        ensure_alive = getattr(run, "_ensure_sender_alive", None)
        if callable(ensure_alive):
            ensure_alive()
        flush = getattr(run, "_flush_queues_inline", None)
        if callable(flush):
            flush()
    except Exception:  # noqa: BLE001 — runtime errors from trackio are non-fatal
        with _lock:
            already_warned = _warned_flush_failed
            if not already_warned:
                _warned_flush_failed = True
        if not already_warned:
            _log.warning(
                "trackio_flush failed; subsequent flush failures will not emit a warning",
                exc_info=True,
            )
