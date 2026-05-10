"""Straight-through-estimator quant simulators for INT3, INT2, MXFP4-KV (LLR-0015).

Stubbed in Phase 2; real implementation + hypothesis property tests land in Phase 4.
"""

from __future__ import annotations

import torch


def int_quant_ste(x: torch.Tensor, bits: int, *, axis: int) -> torch.Tensor:
    """Symmetric integer quantization via STE.

    Phase 2: stub.
    Phase 4: hypothesis-tested for round-trip, idempotence, axis correctness,
    gradient flow.
    """
    raise NotImplementedError("Phase 4: int_quant_ste")


def mxfp4_kv_ste(x: torch.Tensor, *, axis: int) -> torch.Tensor:
    """MXFP4 (E2M1 + E8M0 power-of-two scales) quantization via STE.

    Used only when `feature_matrix` says modelopt's installed version lacks
    MXFP4-KV support. Phase 2: stub.
    """
    raise NotImplementedError("Phase 4: mxfp4_kv_ste")
