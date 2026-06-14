"""Lean, torch-light worker leaf module for the Stage-3 parallel rank-spectra
pool (F2). Top-level (serializable) payload + wrapper + 1-thread initializer.

Kept import-light so the spawn ProcessPool re-import per worker is cheap
(precedent: stage6/plugins/humaneval's torch-free worker leaf). ``_group_stat``
numerics are UNCHANGED — the wrapper reconstructs a tiny duck-typed bank from
the shipped CPU weight list and calls the existing ``_group_stat``.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # type-checking only; runtime import stays lazy (_pin_one_thread)


@dataclass
class _GroupStatPayload:
    """Serializable per-(layer,matrix) group unit. CPU tensors only (CPU torch
    tensor serialization is bit-exact)."""
    layer_idx: int
    name: str
    n_experts: int
    weights_cpu: list  # [torch.Tensor (d_out,d_in) cpu fp32]
    a_g_cpu: "torch.Tensor | None"  # (d_in,d_in) cpu fp32, or None


class _ListBank:
    """Duck-typed bank: ``.shape()`` + ``.get(e)`` over a CPU weight list
    (mirrors test_stage3_tier2._FakeBank)."""

    def __init__(self, weights):
        self._w = weights

    def shape(self):
        return tuple(self._w[0].shape)

    def get(self, e):
        return self._w[e]


def _pin_one_thread() -> None:
    """Pool initializer. 1 intra-op thread per worker is a FIDELITY invariant
    (multi-thread BLAS reassociates fp sums → bit drift in effective_rank), not
    just perf. See F2 doc §2."""
    # Lazy import keeps this leaf module torch-light at load time so each spawned
    # worker's re-import is cheap (torch is only needed here, at runtime).
    import torch
    torch.set_num_threads(1)


def _group_stat_payload(payload: "_GroupStatPayload"):
    """Top-level (serializable) worker entry. Returns ((layer_idx, name), _GroupStats)."""
    # Lazy import keeps the leaf module torch-light at import time but the heavy
    # numerics module is loaded once per spawned worker.
    from moe_compress.stage3.plugins.d_rank_allocate import _group_stat
    bank = _ListBank(payload.weights_cpu)
    gs = _group_stat(payload.n_experts, bank, A_g=payload.a_g_cpu)
    return (payload.layer_idx, payload.name), gs


def run_group_stats_pool(payloads, workers: int):
    """Compute {(layer_idx,name): _GroupStats} for all group payloads.

    workers<=1 → serial in-process (1-thread-pinned here), byte-identical to
    today's serial default path. workers>1 → spawn ProcessPool (CUDA-fork-safe),
    each worker 1-thread-pinned. Reassembly is order-free (dict keyed by
    (layer_idx,name)); the order-sensitive mean(0) stays INSIDE each worker by
    the group granularity.
    """
    if workers is None or workers <= 1:
        # Pin to 1 thread for the serial spectra determinism, then RESTORE the
        # parent's thread count — otherwise Stage 3/4/5 (EoRA eigh/SVD,
        # Router-KD) inherit a 1-thread cap for the rest of the run. The
        # parallel path's _pin_one_thread runs in WORKER processes (isolated),
        # never touching the parent — so only the serial leg needs this guard.
        import torch
        _saved = torch.get_num_threads()
        try:
            _pin_one_thread()
            return dict(_group_stat_payload(p) for p in payloads)
        finally:
            torch.set_num_threads(_saved)

    # FORCE spawn — the parent is CUDA-initialized (Stage 3 is GPU-resident);
    # fork-after-CUDA-init deadlocks the child. Precedent humaneval.py:374-381.
    ctx = multiprocessing.get_context("spawn")
    out: dict = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_pin_one_thread,
    ) as ex:
        for key, gs in ex.map(_group_stat_payload, payloads):
            out[key] = gs
    return out
