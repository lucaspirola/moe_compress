"""Task 11 — deadlock / failure-mode tests (gloo/CPU, defensive).

M1: a NaN on ONE rank must make BOTH ranks raise (the finiteness all-reduce is a
collective; a missing flag would HANG here). The parent re-raises naming the
rank; no orphaned processes. Early-stop: a rank-0 stop decision broadcast must
make both ranks break (no desync hang).
"""
from __future__ import annotations

import pytest

import torch

from moe_compress.router_kd.ddp_runtime import (
    all_ranks_finite,
    broadcast_flag,
    spawn_ddp_workers,
)


def _m1_worker(*, rank, world_size):
    # Mirror the orchestrator's M1 guard: all_ranks_finite BEFORE backward.
    # rank-1's loss is NaN; the all-reduce(MIN) must propagate non-finite to
    # rank-0 too so BOTH raise together (no rank proceeds into a hang).
    loss = torch.tensor(float("nan")) if rank == 1 else torch.tensor(1.0)

    class _Ddp:
        pass

    local_finite = all_ranks_finite(loss, ddp=_Ddp())
    if not local_finite:
        raise RuntimeError(
            f"Stage 5 KD loss non-finite on at least one rank (rank={rank})"
        )
    return "trained"


def _early_stop_worker(*, rank, world_size):
    # rank-0 decides stop (patience tripped); broadcast must make BOTH ranks
    # break unanimously. Simulate a few log windows; on window 2 rank-0 trips.
    stopped = False
    for window in range(4):
        local_decision = (rank == 0 and window == 2)
        stopped = broadcast_flag(local_decision, src=0)
        if stopped:
            break
    # Both ranks must have stopped at the same window.
    return ("stopped", window)


def test_nonfinite_loss_all_ranks_raise_no_hang():
    # The finiteness all-reduce is a collective — a missing flag would HANG.
    # The parent must re-raise naming a failed rank; both child exitcodes set.
    with pytest.raises(RuntimeError, match="rank|non-finite"):
        spawn_ddp_workers(2, backend="gloo", payload={}, worker_fn=_m1_worker)


def test_early_stop_no_desync():
    out = spawn_ddp_workers(2, backend="gloo", payload={},
                            worker_fn=_early_stop_worker)
    # rank-0's collected result: stopped at window 2 (both ranks agree via the
    # broadcast — if they desynced, one would hang and the watchdog/queue would
    # not return a clean ("stopped", 2)).
    assert out == ("stopped", 2)


def test_join_watchdog_terminates_on_hang_smoke():
    # Belt-and-suspenders backstop (also covered in T3): a worker that blocks
    # forever is terminated by the bounded join + the run raises.
    def _noop():  # placeholder to keep the import local
        pass

    # Reuse a module-level hanging worker via a small finite timeout.
    with pytest.raises(RuntimeError, match="timeout|deadlock"):
        spawn_ddp_workers(2, backend="gloo", payload={},
                          worker_fn=_hang_worker, join_timeout_s=2)


def _hang_worker(*, rank, world_size):
    import time
    if rank == 1:
        time.sleep(3600)
    return "ok"
