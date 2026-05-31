"""Compute-parallel ThreadPool map for Stage-2 LSA (Hungarian) solves.

Why a dedicated module (not :mod:`utils.futures`): ``futures`` is scoped to
background-I/O drain semantics; this is a *compute*-parallel order-preserving
map with two extra concerns that the I/O helper has no business knowing about:

1. **scipy version-gating.** ``scipy.optimize.linear_sum_assignment`` (the LSA /
   Hungarian solver) is implemented in C. **scipy >= 1.12 releases the GIL** for
   the duration of the solve, so a Python ``ThreadPool`` of LSA solves runs them
   concurrently. On scipy < 1.12 the GIL is held and a thread pool merely adds
   management overhead on top of fully-serial work (a *regression*). So
   :func:`lsa_threads_enabled` gates threading on the installed scipy version at
   runtime — a stale image (scipy 1.11.x) silently and correctly falls back to
   serial (slow, never wrong), logging a one-shot WARNING so the no-op is
   visible.

2. **BLAS intra-op throttling.** The threaded bodies (Stage-2 cost rows / merge
   alignment) do cdist + Frobenius + eigh + SwiGLU — all BLAS-heavy. Running N
   outer worker threads each spawning the default intra-op BLAS thread count
   oversubscribes the CPU (N x cores). :func:`parallel_map` pins
   ``torch.set_num_threads`` to ``_INNER == 1`` for the duration of the pool and
   restores it afterward. ``torch.set_num_threads`` is **process-global, not
   thread-local**, so it is set ONCE on the calling (main) thread inside a
   ``try/finally`` — NEVER per worker (a per-worker set would race the global
   state and bleed into the main thread after the pool joins). The intra-op
   thread count does not change numerical results for a fixed BLAS build
   (deterministic per build), only timing — so pinning it is byte-irrelevant.

No ``threadpoolctl`` dependency: the global save/set/restore pattern is
sufficient and avoids adding a dep.

Byte-identicality: this helper only changes *when* independent pure functions
run, never *what* they compute. Callers reassemble results by precomputed index
(``ThreadPoolExecutor.map`` preserves input order), so worker completion order
is unobservable in every output.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

import torch

log = logging.getLogger(__name__)

_T = TypeVar("_T")
_R = TypeVar("_R")

# Intra-op BLAS threads per outer worker. Pinned to 1 to avoid
# (outer_workers x cores) oversubscription on the BLAS-heavy bodies. Numerically
# irrelevant for a fixed BLAS build — only timing changes.
_INNER = 1

# Minimal scipy version that releases the GIL inside linear_sum_assignment.
_MIN_SCIPY = (1, 12)

# One-shot WARNING latch + cached version-gate decision.
_warned_disabled = False
_enabled_cache: bool | None = None


def _parse_version(ver: str) -> tuple[int, int]:
    """Parse the leading ``MAJOR.MINOR`` of a version string to an int tuple.

    Tolerant of suffixes (``"1.12.0"``, ``"1.13.0rc1"``, ``"1.12"``). Returns
    ``(0, 0)`` on an unparseable string so the gate fails *closed* (serial).
    """
    parts = ver.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return (0, 0)
    return (major, minor)


def lsa_threads_enabled() -> bool:
    """True iff the installed scipy releases the GIL in ``linear_sum_assignment``.

    That is scipy >= 1.12. Cached one-shot. On a sub-1.12 scipy, logs a single
    WARNING so a stale image is *visibly* serial (slow) rather than silently
    wrong — and returns False so :func:`parallel_map` falls back to serial.
    """
    global _enabled_cache, _warned_disabled
    if _enabled_cache is not None:
        return _enabled_cache
    try:
        import scipy

        enabled = _parse_version(scipy.__version__) >= _MIN_SCIPY
    except Exception:  # pragma: no cover — scipy import failure ⇒ fail closed
        enabled = False
    if not enabled and not _warned_disabled:
        _warned_disabled = True
        try:
            import scipy

            ver = scipy.__version__
        except Exception:
            ver = "<unknown>"
        log.warning(
            "lsa_pool: scipy %s < 1.12 holds the GIL in "
            "linear_sum_assignment; Stage-2 LSA threading DISABLED (running "
            "serial — correct but slow). Bump scipy>=1.12 in the deployed "
            "image to enable the speedup.",
            ver,
        )
    _enabled_cache = enabled
    return enabled


def parallel_map(
    fn: Callable[[_T], _R],
    items: Iterable[_T],
    *,
    enabled: bool | None = None,
    max_workers: int | None = None,
) -> list[_R]:
    """Order-preserving map of ``fn`` over ``items``, optionally threaded.

    Returns a list whose order matches ``items`` (``ThreadPoolExecutor.map``
    preserves input order), so callers may zip / index the result against the
    input deterministically.

    Threading is enabled iff ``enabled`` (when not None) else
    :func:`lsa_threads_enabled`. The explicit ``enabled`` override lets the
    parity test exercise BOTH the serial and threaded branches on any host scipy
    WITHOUT monkeypatching ``scipy.__version__`` (repo policy: no monkeypatch).

    Worker count defaults to ``min(8, os.cpu_count() or 1)``. ``os.cpu_count()``
    can return ``None`` in cgroup/container contexts, so the ``or 1`` floor makes
    the helper degrade to serial instead of raising ``TypeError`` on
    ``min(8, None)``. ``max_workers <= 1`` ⇒ serial.

    BLAS throttle (``torch.set_num_threads(_INNER)``) is applied ONCE on the
    calling (main) thread and restored in a ``finally`` — never per worker
    (it is process-global).
    """
    materialized = list(items)
    if enabled is None:
        enabled = lsa_threads_enabled()

    if max_workers is None:
        workers = min(8, os.cpu_count() or 1)
    else:
        workers = max_workers
    workers = max(1, min(workers, len(materialized) if materialized else 1))

    # Serial fast-path: nothing to gain (or threading disabled / single item).
    if not enabled or workers <= 1 or len(materialized) <= 1:
        return [fn(x) for x in materialized]

    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(_INNER)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # .map preserves input order; materialize inside the pool context
            # so exceptions surface here.
            return list(pool.map(fn, materialized))
    finally:
        torch.set_num_threads(prev_threads)
