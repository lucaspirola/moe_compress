"""Typed quantization specs (LLR-0009, LLR-0010, LLR-0012).

Every Pydantic model here uses `ConfigDict(strict=True, extra='forbid')` per
HLR-0012 — kdr does not silently coerce types and does not silently ignore
unknown YAML fields.
"""

# REQ: LLR-0009
# REQ: LLR-0010
# REQ: LLR-0012

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# Format / granularity / transform enums (LLR-0012)
# ─────────────────────────────────────────────────────────────────────────────

Format = Literal[
    # Existing (Task 2 preserves all four)
    "int", "fp8", "mxfp4", "nvfp4",
    # GGUF legacy round-to-nearest
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    # GGUF K-quants (block K-means, super-blocks of 256)
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k",
    # GGUF I-quants (lattice/codebook)
    "iq1_s", "iq1_m",
    "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
    "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m",
    "iq4_xs", "iq4_nl",
]
"""Numeric format for quantized weights or KV-cache entries.

v0 STE coverage: `int` and `mxfp4` via Native; `nvfp4` and `fp8` via ModelOpt.
Other formats (GGUF Q/IQ family) are schema-accepted but raise at `apply_quant`
time until Task 4 implements them.
"""

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


class UniformWeightSpec(BaseModel):
    """Weight quantizer config (LLR-0010)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    bits: int
    format: Format
    granularity: Granularity
    transform: Transform


WeightQuantSpec = UniformWeightSpec
"""Deprecated alias retained for back-compat with pre-Phase-7.2 imports.
Prefer `UniformWeightSpec` for new code; `MixedWeightSpec` for per-pattern
quantization. The alias will be removed in a future cleanup pass after all
in-tree consumers migrate."""


class WeightPatternSpec(BaseModel):
    """One pattern → spec entry inside a `MixedWeightSpec.spec_map`.

    `pattern` is matched as a substring against `nn.Module.named_modules()`
    dotted paths (mirrors `fp32_carve_outs` semantics — see
    `adapters/zaya1_8b.py:fp32_carve_outs` and
    `quant/native_backend/backend.py:_is_carved_out`). The empty string
    matches every Linear (used internally for the uniform → mixed shim).

    Precedence at install time (LLR-0024 v2, applied in Task 5):
        1. fp32_carve_out match → tensor stays FP/BF16, no quantization.
        2. spec_map match (first-match-wins) → use this spec.
        3. unmatched Linear → strict error.

    Catch-all default (H8 / R4): the empty string ``pattern=""`` is a
    documented, user-authorable catch-all. Substring ``""`` matches every
    Linear, so when authored in a ``MixedWeightSpec.spec_map`` it should be
    placed LAST under first-match-wins precedence — every Linear that did
    not match an earlier, more specific pattern falls through to this
    entry. The factory's internal uniform→mixed shim emits exactly this
    single-entry list with ``pattern=""``; this is the degenerate case of
    the same semantics.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    pattern: str
    bits: int
    format: Format
    granularity: Granularity
    transform: Transform


class MixedWeightSpec(BaseModel):
    """Per-module-pattern weight quant specs (mixed-precision support).

    `spec_map` is order-sensitive: the first matching pattern wins. Empty
    spec_map is rejected (use `UniformWeightSpec` for the single-spec case).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    spec_map: list[WeightPatternSpec] = Field(..., min_length=1)

    @field_validator("spec_map")
    @classmethod
    def _no_duplicate_patterns(
        cls, v: list[WeightPatternSpec]
    ) -> list[WeightPatternSpec]:
        """Rejects any duplicate pattern string inside spec_map.

        Catches the footgun of two list entries with identical `pattern:`
        values at parse time rather than waiting for install-time on a GPU
        node.
        """
        seen: set[str] = set()
        for entry in v:
            if entry.pattern in seen:
                raise ValueError(
                    f"duplicate pattern in spec_map: {entry.pattern!r}"
                )
            seen.add(entry.pattern)
        return v
