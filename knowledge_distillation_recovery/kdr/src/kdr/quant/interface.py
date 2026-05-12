"""QuantBackend Protocol + QuantBlockSubset (LLR-0013).

The single `apply_quant` method shape preserves the "exactly one
`mtq.quantize` call" invariant from HLR-0002 AC #3 while still allowing
the factory to mix backends per quantizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch.nn as nn
from pydantic import BaseModel, ConfigDict, field_validator

from .specs import KVQuantSpec, WeightPatternSpec


class QuantBlockSubset(BaseModel):
    """The portion of the YAML's `quant` block routed to a single backend.

    ``weight`` is a list of :class:`WeightPatternSpec` entries. The Uniform
    case (existing YAMLs with a single global weight spec) normalizes to a
    single-entry list with ``pattern=""`` (matches every Linear). The Mixed
    case (Profile J et al.) carries the routed subset of the original
    ``spec_map``. ``None`` means no weight quantization landed on this
    backend.

    Other LLR-0013 invariants unchanged: exactly one subset is dispatched
    per backend; backends raise ``ValueError`` on ``is_empty()`` subsets.

    Field shape is intentionally flat (LLR-0013): the YAML mirrors how
    humans think about the recipe (K and V belong together under
    ``kv_quant``), while the runtime sub-block flattens them so a backend
    sees three peers and acts on each independently.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    weight: list[WeightPatternSpec] | None = None
    key: KVQuantSpec | None = None
    value: KVQuantSpec | None = None

    @field_validator("weight")
    @classmethod
    def _no_empty_list(
        cls, v: list[WeightPatternSpec] | None
    ) -> list[WeightPatternSpec] | None:
        """Reject ``weight=[]`` so ``is_empty()``'s ``weight is None`` check
        stays coherent (review H3).

        The factory's ``_subset_for`` collapses empty per-backend lists to
        ``None``; a direct constructor call with ``[]`` would otherwise
        produce a subset where ``is_empty()`` returns False but every
        backend's weight path silently no-ops over zero patterns.
        """
        if v is not None and len(v) == 0:
            raise ValueError(
                "QuantBlockSubset.weight must be None or a non-empty list "
                "of patterns; the factory's _subset_for collapses empty "
                "lists to None."
            )
        return v

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
