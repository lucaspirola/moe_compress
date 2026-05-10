"""YAML-spec → modelopt-config-dict translation (LLR-0014's helper).

Stubbed in Phase 2; real implementation lands in Phase 4.
"""

from __future__ import annotations

from ..interface import QuantBlockSubset

# `object` (rather than `Any`) keeps the strict-typing invariant from HLR-0012
# while still allowing modelopt's heterogeneous config dict at the boundary.
ModelOptConfig = dict[str, object]


def quant_block_to_modelopt_config(quant_block: QuantBlockSubset) -> ModelOptConfig:
    """Translate a typed `QuantBlockSubset` into a modelopt config dict.

    Phase 2: stub.
    """
    raise NotImplementedError("Phase 4: quant_block_to_modelopt_config")
