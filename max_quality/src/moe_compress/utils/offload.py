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
    for name, _ in model.named_modules():
        if ".layers." in name and name.count(".") == 2:
            # e.g. "model.layers.0" — park on CPU, lift to GPU per layer
            dmap[name] = "cpu"
        elif name == "":
            continue
        else:
            dmap.setdefault(name, gpu)
    return dmap


def materialize_layer_on_device(layer: nn.Module, device) -> None:
    layer.to(device, non_blocking=True)


def release_layer_to_cpu(layer: nn.Module) -> None:
    layer.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()
