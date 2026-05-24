"""Shared dtype noise-floor table (relocated by S4-3).

The ``_NOISE_FLOOR_BY_DTYPE`` dict maps a covariance storage dtype to the
relative eigenvalue noise floor below which an eigenvalue is treated as
storage-quantization noise rather than signal. It is consumed by two stages:

* Stage 3 — AA-SVD ``_precompute_eigh`` (``stage3/plugins/aa_svd_factor.py``);
* Stage 4 — EoRA ``_compute_eora_factors``
  (``stage4/plugins/eora_compensation.py``).

Before S4-3 the table lived in ``aa_svd_factor.py`` and Stage 4 reached it via
a function-scope ``from moe_compress.stage3_svd import _NOISE_FLOOR_BY_DTYPE``
— a stage4→stage3 cross-import. Relocating the literal here removes that
cross-import: ``tools/`` is the shared layer both stages may depend on.

This module imports NOTHING but ``torch`` (the dict keys are ``torch.dtype``
literals); it is a pure literal. Per the ``tools/`` package contract,
``tools/`` may import ``pipeline/`` but never a stage — and this module
imports neither.
"""
from __future__ import annotations

import torch

_NOISE_FLOOR_BY_DTYPE: dict[torch.dtype, float] = {
    # Relative threshold above which an eigenvalue of B is considered signal
    # rather than storage-quantization noise. Driven by the storage dtype's
    # mantissa bits: bf16 has 7 (~2⁻⁷ ≈ 8e-3 noise), fp16 has 10 (~2⁻¹⁰ ≈ 1e-3),
    # fp32 has 23 (~2⁻²³). Set the floor a small margin above noise to ensure
    # we don't keep noise-inflated directions.
    torch.bfloat16: 1e-2,
    torch.float16:  1e-3,
    torch.float32:  1e-6,
    torch.float64:  1e-12,
}

__all__ = ["_NOISE_FLOOR_BY_DTYPE"]
