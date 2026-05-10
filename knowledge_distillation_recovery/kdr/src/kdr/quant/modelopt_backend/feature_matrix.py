"""Declares which `(target, format, bits)` tuples ModelOpt supports (LLR-0016).

This is the single source of truth that makes modelopt API churn isolatable.
Bumping modelopt version → update this file (and `config_map.py` if API
shape changed). Nothing else needs to change.
"""

from __future__ import annotations

from typing import Literal

from ..specs import Format

Target = Literal["weight", "kv_key", "kv_value"]


SUPPORTED_QUANTS: dict[tuple[Target, Format], list[int]] = {
    # ──── Weight quantization ────────────────────────────────────────────
    ("weight", "int"): [8, 4],  # INT8 / INT4 widely supported
    ("weight", "fp8"): [8],  # E4M3 / E5M2
    ("weight", "nvfp4"): [4],  # Blackwell NVFP4
    ("weight", "mxfp4"): [4],  # OCP MXFP4
    # ──── KV-cache quantization ──────────────────────────────────────────
    ("kv_key", "fp8"): [8],
    ("kv_key", "nvfp4"): [4],
    ("kv_key", "int"): [8, 4],
    ("kv_value", "fp8"): [8],
    ("kv_value", "nvfp4"): [4],
    ("kv_value", "int"): [8, 4],
    # NOTE: INT3, INT2, and MXFP4-KV are NOT in this matrix — they fall
    # through to NativeBackend per LLR-0017.
}
"""The `(target, format)` tuples ModelOpt supports, with the bit widths it
covers. The factory in `quant.factory` consults this to decide which backend
to route each quantizer to."""
