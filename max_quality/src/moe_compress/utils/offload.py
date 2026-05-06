# NOTE: This module is currently unused by run_pipeline.py and stage modules.
# Retained for future use of layer-streaming device maps. See utils/model_io.py
# for the active offload path.
"""Layer-by-layer device placement — smoke-testing on the 16 GB 5080 only.

Production A100 runs use ``device_map="auto"`` and ignore this file. We expose
two helpers:

1. ``build_layer_streaming_device_map`` — pins everything except the decoder
   layers to GPU, then materializes each decoder layer on GPU just-in-time via
   a pre-forward hook.

2. ``materialize_layer_on_device`` / ``release_layer_to_cpu`` — low-level moves
   stages can call directly (e.g. Stage 2's sequential recompute re-uses these
   to keep only layers [0..l] resident during profiling of layer l+1).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def build_layer_streaming_device_map(model: nn.Module, gpu: int = 0) -> dict[str, str | int]:
    """Best-effort device_map for smoke tests.

    NOTE: for accelerate ≥ 1.0 you can pass ``device_map="auto"`` with
    ``max_memory={0: "14GiB", "cpu": "40GiB"}`` and it will do something
    similar. We only build this map when the caller wants layer-streaming but
    can't accept accelerate-style sharding (e.g. Stage 2's sequential merge).
    """
    dmap: dict[str, str | int] = {}
    n_cpu = 0
    n_gpu = 0
    for name, _ in model.named_modules():
        if ".layers." in name and name.count(".") == 2:
            # e.g. "model.layers.0" — park on CPU, lift to GPU per layer
            dmap[name] = "cpu"
            n_cpu += 1
        elif name == "":
            continue
        else:
            dmap[name] = gpu
            n_gpu += 1
    log.info(
        "streaming layer-aware device map: parked %d layers on CPU, %d on GPU",
        n_cpu, n_gpu,
    )
    return dmap


def materialize_layer_on_device(
    layer: nn.Module,
    device: torch.device | str | int,
) -> None:
    # Normalize to torch.device so a malformed argument fails here, not deep
    # inside Tensor.to() with an opaque message.
    device = torch.device(device) if not isinstance(device, torch.device) else device
    layer.to(device, non_blocking=True)


def release_layer_to_cpu(layer: nn.Module) -> None:
    layer.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
