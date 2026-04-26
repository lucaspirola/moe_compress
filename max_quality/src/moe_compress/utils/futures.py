"""Lightweight helpers for background ThreadPoolExecutor I/O patterns."""
from __future__ import annotations


def drain_done_futures(futures: list) -> None:
    """Surface exceptions from completed futures without blocking.

    Removes done futures from the list in-place. Call before submitting more
    work so a stale failure doesn't silently keep the loop running.
    """
    still_pending = []
    for f in futures:
        if f.done():
            f.result()      # re-raises if the thread errored
        else:
            still_pending.append(f)
    futures[:] = still_pending
