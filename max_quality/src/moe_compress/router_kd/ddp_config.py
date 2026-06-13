"""Config schema + resolver for the opt-in Router-KD DDP knob.

A new ``stage5_router_kd.ddp`` mapping (default absent → disabled) controls
whether the Router-KD training loop (Stage 2.5 heal + Stage 5 final) runs under
in-process DistributedDataParallel. When absent / ``enabled: false`` / resolved
``world_size <= 1`` the orchestrator takes the EXISTING single-process path and
``DdpConfig.enabled is False`` — backward-compat preserved.

The effective global batch is METRIC_PINNED (it determines the trained result),
so DDP keeps it FIXED and splits each optimizer step's ``batch_size`` rows
across replicas: ``per_gpu = batch_size // world_size`` must be an integer ≥ 1.
A requested ``world_size`` that does not divide ``batch_size`` is rejected.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DdpConfig:
    enabled: bool = False
    world_size: int = 1
    # "gloo" for the CPU tolerance test; "nccl" on GPU. Teacher strategy is
    # validated separately (Task 7); kept here for surfacing.
    backend: str = "nccl"

    @classmethod
    def from_config(cls, config: dict, *, device_count_fn=None) -> "DdpConfig":
        s5 = config.get("stage5_router_kd", {}) or {}
        raw = s5.get("ddp", {}) or {}
        enabled = (raw.get("enabled") is True) or (
            str(raw.get("enabled", "")).strip().lower() == "true"
        )
        if not enabled:
            return cls(enabled=False, world_size=1)
        if device_count_fn is None:
            import torch

            device_count_fn = torch.cuda.device_count
        avail = int(device_count_fn())
        requested = raw.get("world_size")
        ws = int(requested) if requested is not None else avail
        # avail == 0 only on the CPU test path (no CUDA); leave ws as requested
        # there so the gloo tolerance test can ask for world_size=2 on CPU.
        ws = min(ws, avail) if avail > 0 else ws
        backend = str(raw.get("backend", "nccl"))
        if ws <= 1:
            # enabled:true but only one device available (or world_size==1) →
            # collapse to the single-process path. Backward-compat on
            # single-GPU hosts even with the flag on.
            return cls(enabled=False, world_size=1)
        # METRIC_PINNED cap: per_gpu_batch = global_batch / world_size must be an
        # integer >= 1, and the effective batch must stay == single-GPU
        # global_batch.
        global_batch = int(s5["batch_size"])
        if global_batch % ws != 0:
            raise ValueError(
                f"Router-KD DDP: stage5_router_kd.batch_size={global_batch} is not "
                f"divisible by ddp.world_size={ws}. The effective global batch is "
                "METRIC_PINNED (it determines the trained result); per_gpu_batch = "
                "global_batch / world_size must be an integer. Either set world_size "
                "to a divisor of batch_size, raise batch_size to a multiple of "
                "world_size AND co-scale gradient_accumulation DOWN to keep the "
                "effective batch fixed, or reduce world_size."
            )
        return cls(enabled=True, world_size=ws, backend=backend)


__all__ = ["DdpConfig"]
