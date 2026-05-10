"""Per-quantizer backend routing factory (LLR-0017).

`partition_and_dispatch` examines the YAML's `QuantBlock`, consults
`feature_matrix.SUPPORTED_QUANTS`, partitions the quantizers per backend,
and invokes each backend's `apply_quant` exactly once. Stubbed in Phase 2.
"""

from __future__ import annotations

import torch.nn as nn

from ..config import QuantBlock
from .interface import QuantBackend


def partition_and_dispatch(model: nn.Module, quant_block: QuantBlock) -> list[QuantBackend]:
    """Partition `quant_block` per backend per `feature_matrix.SUPPORTED_QUANTS`,
    invoke each routed backend's `apply_quant`, and return the list of backends
    that were used (for downstream save dispatch).

    Phase 2: stub.
    """
    raise NotImplementedError("Phase 4: partition_and_dispatch")
