"""Task 3 — process-group bootstrap/teardown + in-process spawn (gloo/CPU).

All collectives use the gloo backend (no CUDA/NCCL on this box).
"""
from __future__ import annotations

import pytest

import torch.distributed as dist

from moe_compress.router_kd.ddp_runtime import (
    _destroy_pg,
    _free_port,
    _init_pg,
    spawn_ddp_workers,
)


# Module-level worker fns so the spawn (serialization) handoff can import them.
def _ok_worker(*, rank, world_size):
    return f"rank{rank}"


def _raise_worker(*, rank, world_size):
    if rank == 1:
        raise ValueError("boom on rank 1")
    return f"rank{rank}"


def _hang_worker(*, rank, world_size):
    import time
    # rank 1 blocks forever to trip the join watchdog.
    if rank == 1:
        time.sleep(3600)
    return f"rank{rank}"


def test_bootstrap_teardown_world1():
    port = _free_port()
    try:
        _init_pg(rank=0, world_size=1, backend="gloo", master_port=port)
        assert dist.is_initialized() is True
        assert dist.get_rank() == 0
    finally:
        _destroy_pg()
    assert dist.is_initialized() is False


def test_spawn_two_workers_collect_result():
    out = spawn_ddp_workers(
        2, backend="gloo", payload={}, worker_fn=_ok_worker,
    )
    # Parent receives rank-0's value via the queue.
    assert out == "rank0"


def test_worker_nonzero_exit_raises():
    with pytest.raises(RuntimeError, match="rank 1"):
        spawn_ddp_workers(
            2, backend="gloo", payload={}, worker_fn=_raise_worker,
        )


def test_join_watchdog_terminates_on_hang():
    with pytest.raises(RuntimeError, match="timeout|deadlock"):
        spawn_ddp_workers(
            2, backend="gloo", payload={}, worker_fn=_hang_worker,
            join_timeout_s=3,
        )
