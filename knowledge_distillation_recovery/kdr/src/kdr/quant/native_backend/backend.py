"""NativeBackend (LLR-0015).

Hand-rolled STE simulators for INT3, INT2, and any KV format modelopt's
installed version doesn't ship. Uses PyTorch forward hooks at adapter-
declared module paths — does NOT call `mtq.quantize`. Stubbed in Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from ..interface import QuantBlockSubset


class NativeBackend:
    """`QuantBackend` Protocol implementation with hand-rolled STE."""

    name: str = "native"

    def apply_quant(self, model: nn.Module, quant_block: QuantBlockSubset) -> None:
        """Install PyTorch forward hooks at adapter-declared module paths.

        Phase 2: stub.
        """
        raise NotImplementedError("Phase 4: NativeBackend.apply_quant")

    def save(self, model: nn.Module, output_dir: Path) -> None:
        """Emit compressed-tensors safetensors. Native uses the same
        compressed-tensors writer as ModelOpt's save path.

        Phase 2: stub.
        """
        raise NotImplementedError("Phase 4: NativeBackend.save")
