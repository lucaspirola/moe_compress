"""Task 5 — DDP wrap (after freeze + optimizer) + grad-accum no_sync +
finiteness all-reduce (M1). gloo/CPU, world_size=2.

The grad-average + finiteness tests run inside spawned gloo workers (real
collectives); the parent asserts via the result queue.
"""
from __future__ import annotations

import pytest

import torch
import torch.nn as nn

from moe_compress.router_kd.ddp_runtime import (
    all_ranks_finite,
    spawn_ddp_workers,
    wrap_ddp,
)


def _wrap_after_freeze_worker(*, rank, world_size):
    torch.manual_seed(0)
    m = nn.Linear(4, 4)
    # Freeze all but the weight (router-only analogue).
    m.bias.requires_grad_(False)
    optim = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    ddp = wrap_ddp(m, device=None, backend="gloo")
    # The optimizer's leaf param IS the module's trainable param (same object).
    opt_params = {id(p) for g in optim.param_groups for p in g["params"]}
    mod_params = {id(p) for p in ddp.module.parameters() if p.requires_grad}
    same = opt_params == mod_params
    # requires_grad already set before wrap (freeze first).
    rg_ok = (ddp.module.weight.requires_grad and not ddp.module.bias.requires_grad)
    return ("ok" if (same and rg_ok) else "bad")


def _grad_avg_worker(*, rank, world_size):
    # Same 1-layer toy, identical init on both ranks; each rank gets ONE row of
    # the shared 2-row batch (row-split). After backward + DDP all-reduce, each
    # rank's .grad must equal the gradient of the 2-row per-token-MEAN loss
    # (i.e. the AVERAGE of the two per-row grads), NOT the sum.
    torch.manual_seed(0)
    lin = nn.Linear(3, 3, bias=False)
    ddp = wrap_ddp(lin, device=None, backend="gloo")

    full = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 rows
    target = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    per_gpu = full.shape[0] // world_size
    local = full[rank * per_gpu:(rank + 1) * per_gpu]
    local_t = target[rank * per_gpu:(rank + 1) * per_gpu]

    out = ddp(local)
    # Per-token MEAN loss on the local row (denominator equal across ranks).
    loss = ((out - local_t) ** 2).mean()
    loss.backward()  # DDP all-reduce-AVERAGES grads here
    ddp_grad = ddp.module.weight.grad.detach().clone()

    # Reference: the SAME mean loss over BOTH rows, single process.
    torch.manual_seed(0)
    ref = nn.Linear(3, 3, bias=False)
    ref_out = ref(full)
    ref_loss = ((ref_out - target) ** 2).mean()
    ref_loss.backward()
    ref_grad = ref.weight.grad.detach().clone()

    close = torch.allclose(ddp_grad, ref_grad, rtol=1e-5, atol=1e-7)
    if rank == 0:
        return "match" if close else f"MISMATCH max={float((ddp_grad-ref_grad).abs().max())}"
    return "ok"


def _finite_worker(*, rank, world_size):
    # rank-1 has a non-finite loss; the finiteness all-reduce must make BOTH
    # ranks see local_finite == False (so they raise together, no deadlock).
    loss = torch.tensor(float("nan")) if rank == 1 else torch.tensor(1.0)

    class _Ddp:  # truthy sentinel — all_ranks_finite uses it only as "DDP on"
        pass

    local_finite = all_ranks_finite(loss, ddp=_Ddp())
    return "finite" if local_finite else "nonfinite"


def test_ddp_wrap_after_freeze():
    out = spawn_ddp_workers(2, backend="gloo", payload={},
                            worker_fn=_wrap_after_freeze_worker)
    assert out == "ok"


def test_grad_avg_matches_local():
    out = spawn_ddp_workers(2, backend="gloo", payload={},
                            worker_fn=_grad_avg_worker)
    assert out == "match", out


def test_nonfinite_loss_all_ranks_raise():
    # Both ranks must report nonfinite (rank-0's loss is finite, but the
    # all-reduce MIN propagates rank-1's NaN flag to it).
    out = spawn_ddp_workers(2, backend="gloo", payload={},
                            worker_fn=_finite_worker)
    # rank-0's result is collected; it must be "nonfinite".
    assert out == "nonfinite"


def test_no_sync_on_nonboundary_microbatch():
    # grad_sync_context enters DDP.no_sync() on a NON-boundary microbatch and a
    # nullcontext on the boundary — so the all-reduce fires once per grad-accum
    # window, not per microbatch. Spy via a fake DDP exposing no_sync().
    from moe_compress.router_kd.ddp_runtime import grad_sync_context
    import contextlib

    entered = {"no_sync": 0}

    class _FakeDdp:
        @contextlib.contextmanager
        def no_sync(self):
            entered["no_sync"] += 1
            yield

    fake = _FakeDdp()
    # Non-boundary → no_sync entered.
    with grad_sync_context(fake, is_boundary=False, ddp_on=True):
        pass
    assert entered["no_sync"] == 1
    # Boundary → nullcontext, no_sync NOT entered.
    with grad_sync_context(fake, is_boundary=True, ddp_on=True):
        pass
    assert entered["no_sync"] == 1
    # ddp_on False (single-process) → always nullcontext.
    with grad_sync_context(None, is_boundary=False, ddp_on=False):
        pass
    assert entered["no_sync"] == 1


def test_no_manual_world_size_rescale():
    # Guard: the orchestrator must NOT multiply the loss by world_size anywhere
    # (DDP averages grads → per-token-mean semantics; a *world_size rescale
    # would double-count). grep the source.
    import pathlib
    src = pathlib.Path(
        __file__).resolve().parents[1] / "src" / "moe_compress" / "router_kd" / "orchestrator.py"
    text = src.read_text()
    assert "* world_size" not in text and "world_size *" not in text
