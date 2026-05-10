"""QuantBackend Protocol + QuantBlockSubset (LLR-0013).

The single `apply_quant` method shape preserves the "exactly one
`mtq.quantize` call" invariant from HLR-0002 AC #3 while still allowing
the factory to mix backends per quantizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch.nn as nn
from pydantic import BaseModel, ConfigDict

from .specs import KVQuantSpec, WeightQuantSpec


class QuantBlockSubset(BaseModel):
    """The portion of the YAML's `quant` block routed to a single backend.

    Field shape is intentionally flat (LLR-0013): the YAML mirrors how humans
    think about the recipe (K and V belong together under `kv_quant`), while
    the runtime sub-block flattens them so a backend sees three peers and acts
    on each independently.

    Exactly one `QuantBlockSubset` is dispatched to each routed backend per
    `apply_quant` call — never two — per LLR-0014. When the factory routes all
    three quantizers to the same backend (typically ModelOpt), the merged
    sub-block has all three fields populated.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    weight: WeightQuantSpec | None = None
    key: KVQuantSpec | None = None
    value: KVQuantSpec | None = None

    def is_empty(self) -> bool:
        """A backend that receives an empty sub-block raises (LLR-0013 AC)."""
        return self.weight is None and self.key is None and self.value is None


class QuantBackend(Protocol):
    """Interface implemented by `ModelOptBackend` and `NativeBackend`.

    The training loop only sees this Protocol — never modelopt directly. This
    isolates kdr from modelopt API churn (LLR-0016 feature_matrix is the only
    file that needs editing on modelopt bumps).
    """

    name: str

    def apply_quant(self, model: nn.Module, quant_block: QuantBlockSubset) -> None:
        """Install fake-quant on `model` for the populated fields of `quant_block`.

        Backends MUST raise `ValueError` on an empty sub-block. ModelOpt's
        implementation calls `mtq.quantize` exactly once; Native's implementation
        installs PyTorch forward hooks (no `mtq.quantize` call at all).
        """
        ...

    def save(self, model: nn.Module, output_dir: Path) -> None:
        """Emit the trained checkpoint as HF compressed-tensors safetensors.

        The output directory loads cleanly via `AutoModelForCausalLM.from_pretrained`.
        """
        ...
