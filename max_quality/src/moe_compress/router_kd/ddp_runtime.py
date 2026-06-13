"""In-process DDP runtime for Router-KD: process-group bootstrap/teardown and
the spawn-N-rank-workers driver.

Mirrors the Stage-3 spawn precedent (``stage3/plugins/covariance_collection.py``
``ctx.Process(...).start()/join()`` + exit-code check) but layers
``torch.distributed`` collectives on top (NCCL on GPU, gloo on the CPU test).
The PARENT first materializes the live compressed student to a temp dir (Task 8);
each worker bootstraps the group, pins its GPU, reconstructs the student, trains,
and rank-0 returns the ``out_dir`` Path via a ``mp.SimpleQueue``.

The PRIMARY deadlock defense is the in-loop finiteness all-reduce (Task 5/M1);
the bounded ``join_timeout_s`` watchdog here is the backstop for a collective
hang that escapes it.
"""
from __future__ import annotations

import contextlib
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _init_pg(rank, world_size, *, backend, master_addr="127.0.0.1", master_port):
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)


def _destroy_pg():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(student, *, device, backend):
    """Wrap ``student`` in DistributedDataParallel (Task 5).

    Call AFTER freeze + optimizer build (+ optional torch.compile) so the
    optimizer already holds the leaf params and DDP wraps the SAME module —
    ``ddp.module.parameters()`` are the identical objects, no optimizer rebuild.

    ``device_ids`` is ``[device.index]`` on CUDA/NCCL and ``None`` for gloo/CPU.
    ``find_unused_parameters=False``: the router-only trainable scope is fully
    used every step (faster + correct). ``gradient_as_bucket_view=True`` avoids
    an extra grad copy. DDP averages gradients across ranks by default — exactly
    the per-token-mean semantics; NO manual ``*world_size`` rescale anywhere.
    """
    device_ids = None
    if device is not None and getattr(device, "type", None) == "cuda":
        device_ids = [device.index]
    return torch.nn.parallel.DistributedDataParallel(
        student,
        device_ids=device_ids,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def all_ranks_finite(loss, *, ddp) -> bool:
    """Return whether the loss is finite on EVERY rank (M1 — NaN→deadlock fix).

    Under DDP, a NaN on ONE rank would let that rank exit the loop while the
    others block forever in the next gradient all-reduce. Before the per-rank
    finiteness check, all-reduce a flag with ReduceOp.MIN (0 if ANY rank is
    non-finite) so every rank sees any rank's NaN and they raise TOGETHER.
    On the single-process path (``ddp is None``) this is a plain local check.
    """
    if ddp is None:
        return bool(torch.isfinite(loss))
    flag = torch.tensor(
        [1.0 if torch.isfinite(loss) else 0.0], device=loss.device
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def grad_sync_context(ddp_student, *, is_boundary, ddp_on):
    """Context manager for the microbatch backward (Task 5).

    Returns ``ddp_student.no_sync()`` on a NON-boundary grad-accum microbatch so
    the gradient all-reduce fires only on the boundary microbatch (one
    all-reduce per grad-accum window). On the boundary, or on the single-process
    path (``ddp_on`` False), returns a nullcontext.
    """
    if ddp_on and not is_boundary:
        return ddp_student.no_sync()
    return contextlib.nullcontext()


def _worker_entry(rank, world_size, backend, master_port, result_q, payload, worker_fn):
    try:
        _init_pg(rank, world_size, backend=backend, master_port=master_port)
        out = worker_fn(rank=rank, world_size=world_size, **payload)
        if rank == 0:
            result_q.put(("ok", out))
    except BaseException as exc:  # noqa: BLE001 — surfaced to the parent + re-raised
        result_q.put(("err", rank, repr(exc)))
        raise
    finally:
        _destroy_pg()


def spawn_ddp_workers(world_size, *, backend, payload, worker_fn, join_timeout_s=None):
    """Spawn ``world_size`` rank workers, return rank-0's result.

    ``join_timeout_s`` defaults to ``None`` (block indefinitely) for a long
    legitimate production run; the failure-mode tests pass a small finite value
    to assert the watchdog fires. A worker exception is re-raised on the parent
    naming the failed rank; a non-zero exit (no queued error) is also re-raised.
    """
    ctx = mp.get_context("spawn")
    result_q = ctx.SimpleQueue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_worker_entry,
            args=(r, world_size, backend, port, result_q, payload, worker_fn),
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()

    # Bounded join (watchdog): a collective hang that escapes the in-loop
    # finiteness all-reduce (Task 5) still terminates instead of wedging.
    for p in procs:
        p.join(timeout=join_timeout_s)
        if p.is_alive():
            for q in procs:
                q.terminate()
            for q in procs:
                q.join(timeout=5)
            raise RuntimeError(
                f"Router-KD DDP: worker exceeded join timeout {join_timeout_s}s "
                "(suspected collective deadlock); terminated all workers."
            )

    # Drain ALL queued messages (rank-0's "ok" + any rank's "err"); prefer the
    # first error so the failed rank is named even if rank-0 succeeded.
    err = None
    ok = None
    while not result_q.empty():
        msg = result_q.get()
        if msg[0] == "err" and err is None:
            err = msg
        elif msg[0] == "ok":
            ok = msg

    # Exit-code backstop (mirror the Stage-3 check).
    failed_codes = [
        (i, p.exitcode) for i, p in enumerate(procs)
        if p.exitcode not in (0, None)
    ]
    if err is not None:
        raise RuntimeError(f"Router-KD DDP rank {err[1]} failed: {err[2]}")
    if failed_codes:
        i, code = failed_codes[0]
        raise RuntimeError(
            f"Router-KD DDP: rank {i} worker exited with code {code}"
        )
    if ok is None:
        raise RuntimeError(
            "Router-KD DDP: rank-0 produced no result (all workers exited 0 "
            "but the result queue was empty)."
        )
    return ok[1]


__all__ = [
    "_init_pg",
    "_destroy_pg",
    "_free_port",
    "spawn_ddp_workers",
    "wrap_ddp",
    "all_ranks_finite",
    "grad_sync_context",
]
