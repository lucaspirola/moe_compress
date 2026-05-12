"""YAML-spec → modelopt-config-dict translation (LLR-0014's helper).

Translates a :class:`~kdr.quant.interface.QuantBlockSubset` into the single
config dict ``mtq.quantize`` consumes. The translation is purely structural —
no model is touched here — so it is unit-testable on a CPU.

Wildcard patterns
-----------------

Modelopt's config dict uses ``fnmatch``-style patterns to address quantizer
submodules by name. The defaults here are stock-HF-shape:

  * ``*weight_quantizer``        → every Linear's weight-side quantizer
  * ``*input_quantizer``         → disabled globally (kdr does not quantize
    activations as a recipe choice)
  * ``*[Kk]_proj*output_quantizer`` → K-projection outputs (per HF convention)
  * ``*[Vv]_proj*output_quantizer`` → V-projection outputs

ZAYA1's CCA uses different submodule names (``kv_compressor``, ``q_compressor``
per arxiv:2605.05365 §II-A1); the adapter-driven Phase 5 wiring overrides
these patterns by passing custom wildcards through this function.

The returned dict is ``dict[str, object]`` rather than ``dict[str, Any]`` to
keep the strict-typing invariant from HLR-0012; modelopt accepts it as a
generic mapping.
"""

# REQ: LLR-0014

from __future__ import annotations

from ..interface import QuantBlockSubset
from ..specs import Format, KVQuantSpec, WeightPatternSpec

# `object` (rather than `Any`) keeps the strict-typing invariant from HLR-0012
# while still allowing modelopt's heterogeneous config dict at the boundary.
ModelOptConfig = dict[str, object]


# Default wildcard patterns (stock HF). Phase 5 ZAYA1 adapter overrides.
_DEFAULT_WEIGHT_PATTERN = "*weight_quantizer"
_DEFAULT_INPUT_PATTERN = "*input_quantizer"
_DEFAULT_KEY_PATTERN = "*[Kk]_proj*output_quantizer"
_DEFAULT_VALUE_PATTERN = "*[Vv]_proj*output_quantizer"


def quant_block_to_modelopt_config(
    quant_block: QuantBlockSubset,
    *,
    ignore: list[str] | None = None,
    weight_target_pattern: str = _DEFAULT_WEIGHT_PATTERN,
    input_target_pattern: str = _DEFAULT_INPUT_PATTERN,
    key_target_pattern: str = _DEFAULT_KEY_PATTERN,
    value_target_pattern: str = _DEFAULT_VALUE_PATTERN,
) -> ModelOptConfig:
    """Translate a typed ``QuantBlockSubset`` into a modelopt config dict.

    Args:
        quant_block: the per-backend slice routed by ``factory.partition_and_dispatch``.
        ignore: submodule patterns excluded from quantization (the adapter's
            FP32 carve-outs). Each pattern becomes an ``{enable: False}``
            entry under that wildcard.
        weight_target_pattern: wildcard targeting weight-side quantizers.
        input_target_pattern: wildcard targeting input-side quantizers
            (always disabled — kdr does not quantize activations).
        key_target_pattern: wildcard targeting K output quantizers.
        value_target_pattern: wildcard targeting V output quantizers.

    Returns:
        A dict shaped ``{"quant_cfg": {wildcards: entries}, "algorithm": "max"}``.

    Raises:
        ValueError: if ``quant_block`` is empty.
    """
    if quant_block.is_empty():
        raise ValueError("cannot translate empty quant block to modelopt config")

    quant_cfg: dict[str, object] = {}

    if quant_block.weight is not None:
        # apply_quant has already enforced len(weight) == 1 (review H4).
        quant_cfg[weight_target_pattern] = _weight_modelopt_entry(
            quant_block.weight[0]
        )
        # Globally disable input quantizers — kdr does not quantize activations.
        quant_cfg[input_target_pattern] = {"enable": False}

    if quant_block.key is not None:
        quant_cfg[key_target_pattern] = _kv_modelopt_entry(quant_block.key)

    if quant_block.value is not None:
        quant_cfg[value_target_pattern] = _kv_modelopt_entry(quant_block.value)

    # FP32 carve-outs: substring-style patterns wrapped in `*…*` so they match
    # anywhere in the dotted module path (matches NativeBackend's substring rule).
    if ignore:
        for pattern in ignore:
            quant_cfg[f"*{pattern}*"] = {"enable": False}

    return {"quant_cfg": quant_cfg, "algorithm": "max"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-quantizer translation
# ─────────────────────────────────────────────────────────────────────────────


def _weight_modelopt_entry(spec: WeightPatternSpec) -> dict[str, object]:
    """Per-Linear weight quantizer config dict.

    Modelopt's per-quantizer dict carries:

      * ``num_bits``: scalar int (INT-N) or tuple ``(exp_bits, mant_bits)`` for FP
      * ``axis``: integer axis for the per-channel scale (0 for ``[out, in]``
        Linear weight = per-output-channel)
      * ``block_sizes``: optional ``{axis: size}`` mapping for block-quantization
      * ``scale_bits``: tuple for FP4 formats (E4M3 scales for NVFP4, E8M0 for MXFP4)
      * ``enable``: True to actually quantize (kdr always sets this)
    """
    base: dict[str, object] = {"axis": 0, "enable": True}
    base.update(_format_to_modelopt_dtype(spec.format, spec.bits))
    return base


def _kv_modelopt_entry(spec: KVQuantSpec) -> dict[str, object]:
    """K or V output_quantizer config dict.

    The ``axis`` reflects the granularity along the K/V tensor's typical
    ``[B, H, T, D]`` layout:

      * ``channel`` → ``axis=-1`` (per-channel along head_dim — K convention)
      * ``token``   → ``axis=-2`` (per-token along seq_len — V convention)
      * ``tensor``  → ``axis=None`` (single scalar scale)

    ``group`` and ``block`` granularities rely on ``block_sizes`` already set
    by the format (NVFP4 / MXFP4); their ``axis`` is left unset.
    """
    base: dict[str, object] = {"enable": True}
    base.update(_format_to_modelopt_dtype(spec.format, spec.bits))
    if spec.granularity == "channel":
        base["axis"] = -1
    elif spec.granularity == "token":
        base["axis"] = -2
    elif spec.granularity == "tensor":
        base["axis"] = None
    # `group` / `block`: handled via `block_sizes` from the format dispatch.
    return base


def _format_to_modelopt_dtype(fmt: Format, bits: int) -> dict[str, object]:
    """Map ``(format, bits)`` to the modelopt numeric-format dict fragment.

    Splitting this out keeps the weight and KV entries DRY — both consume the
    same numeric-format dict, only differing in axis/granularity handling.
    """
    if fmt == "nvfp4":
        # NVFP4 = E2M1 mantissa + E4M3 per-block scales (block size 16).
        return {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16},
            "scale_bits": (4, 3),
        }
    if fmt == "mxfp4":
        # MXFP4 = E2M1 mantissa + E8M0 per-block scales (block size 32, OCP spec).
        return {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32},
            "scale_bits": (8, 0),
        }
    if fmt == "fp8":
        # FP8: bits=8 → E4M3 (default); other bits not standard but exposed
        # for E5M2 if a recipe ever asks for it.
        return {"num_bits": (4, 3) if bits == 8 else (5, 2)}
    if fmt == "int":
        return {"num_bits": bits}
    raise ValueError(f"unsupported quant format: {fmt!r}")
