"""ModelOptBackend (LLR-0014).

Wraps `mtq.quantize` and the compressed-tensors converter. Calls
`mtq.quantize` exactly once per `apply_quant`. Stubbed in Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from ..interface import QuantBlockSubset


class ModelOptBackend:
    """`QuantBackend` Protocol implementation for NVIDIA modelopt."""

    name: str = "modelopt"

    def apply_quant(self, model: nn.Module, quant_block: QuantBlockSubset) -> None:
        """Build a single modelopt config dict from `quant_block` and call
        `mtq.quantize` exactly once.

        Phase 2: stub.
        """
        raise NotImplementedError("Phase 4: ModelOptBackend.apply_quant")

    def save(self, model: nn.Module, output_dir: Path) -> None:
        """Emit compressed-tensors safetensors via the converter selected by
        `feature_matrix` for the format actually used.

        Phase 2: stub.
        """
        raise NotImplementedError("Phase 4: ModelOptBackend.save")
