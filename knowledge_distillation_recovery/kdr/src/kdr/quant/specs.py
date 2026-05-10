"""Typed quantization specs (LLR-0009, LLR-0010, LLR-0012).

Every Pydantic model here uses `ConfigDict(strict=True, extra='forbid')` per
HLR-0012 — kdr does not silently coerce types and does not silently ignore
unknown YAML fields.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# ─────────────────────────────────────────────────────────────────────────────
# Format / granularity / transform enums (LLR-0012)
# ─────────────────────────────────────────────────────────────────────────────

Format = Literal["int", "fp8", "mxfp4", "nvfp4"]
"""Numeric format for quantized weights or KV-cache entries."""

Granularity = Literal["tensor", "channel", "group", "block", "token"]
"""Quantization granularity: shape over which scale/zero-point are shared."""

Transform = Literal["none"]
"""Pre-quantization transforms.

v0 supports only `none`. Hadamard / FWHT are deferred to v1+ per the plan
(`/home/lucas/.claude/plans/radiant-pondering-beacon.md`, "Deferred to v1+").
v1 will extend this Literal with `"hadamard"` and `"fwht"` and route them in
`NativeBackend`; this LLR's structure does not change.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Per-quantizer specs (LLR-0009 K/V; LLR-0010 weight)
# ─────────────────────────────────────────────────────────────────────────────


class KVQuantSpec(BaseModel):
    """K or V quantizer config (LLR-0009).

    All four fields are required — kdr does not silently inject defaults so
    YAML recipes cannot drift. Asymmetric K/V (e.g. K bits != V bits) is
    expressed by setting different `KVQuantSpec` instances on the parent
    `kv_quant.key` and `kv_quant.value` blocks.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    bits: int
    format: Format
    granularity: Granularity
    transform: Transform


class WeightQuantSpec(BaseModel):
    """Weight quantizer config (LLR-0010)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    bits: int
    format: Format
    granularity: Granularity
    transform: Transform
