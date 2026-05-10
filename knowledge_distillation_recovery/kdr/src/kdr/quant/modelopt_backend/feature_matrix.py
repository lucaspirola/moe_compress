"""Declares which `(target, format, bits)` tuples ModelOpt supports (LLR-0016).

This is the single source of truth that makes modelopt API churn isolatable.
Bumping modelopt version → update this file (and `config_map.py` if the API
shape changed). Nothing else needs to change.

Three responsibilities:

  1. ``SUPPORTED_QUANTS`` — the matrix; the factory consults it to route each
     quantizer.
  2. ``is_supported(target, fmt, bits)`` — convenience predicate over the
     matrix; the factory uses it.
  3. ``resolve_converter_class(weight_format)`` — picks the compressed-tensors
     converter class for the saved weight format (LLR-0021).

The converter resolver is a thin shim over ``compressed_tensors.converters``
so that bumping that package's class names is a single-line edit here.
"""

# REQ: LLR-0016
# REQ: LLR-0021

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


def is_supported(target: Target, fmt: Format, bits: int) -> bool:
    """``True`` if modelopt covers this ``(target, fmt, bits)`` tuple."""
    bits_list = SUPPORTED_QUANTS.get((target, fmt))
    return bits_list is not None and bits in bits_list


# ─────────────────────────────────────────────────────────────────────────────
# Converter resolver (LLR-0021)
# ─────────────────────────────────────────────────────────────────────────────

# Map weight format → compressed-tensors converter class name. The class is
# imported at save-time (lazy) so kdr is importable without compressed_tensors
# installed.
_CONVERTER_CLASS_NAMES: dict[Format, str] = {
    "nvfp4": "ModelOptNvfp4Converter",
    "fp8": "ModelOptFp8Converter",
    "int": "ModelOptIntConverter",
    # MXFP4 weight-save converter: not yet upstreamed into compressed_tensors.
    # Routes to a clear error if a recipe asks for MXFP4 weight save.
}


def resolve_converter_class(weight_format: Format) -> type[object]:
    """Resolve the compressed-tensors converter class for ``weight_format``.

    The class is loaded lazily from ``compressed_tensors.converters`` so this
    module does not require that dependency at import time. Raises a clear
    ``ImportError`` (with the missing class named) if the installed
    ``compressed_tensors`` lacks the expected class for the chosen format.
    """
    cls_name = _CONVERTER_CLASS_NAMES.get(weight_format)
    if cls_name is None:
        raise ValueError(
            f"no compressed-tensors converter is wired for weight format "
            f"{weight_format!r}. Either pick a supported format "
            f"({sorted(_CONVERTER_CLASS_NAMES.keys())}) or extend "
            f"`feature_matrix._CONVERTER_CLASS_NAMES` with the new mapping."
        )

    try:
        import compressed_tensors.converters as ct_converters
    except ImportError as e:
        raise ImportError(
            "ModelOptBackend.save requires `compressed-tensors` to be "
            "installed. Run `pip install -e .[compressed]` (or "
            "`pip install compressed-tensors>=0.7.0`)."
        ) from e

    converter_cls = getattr(ct_converters, cls_name, None)
    if converter_cls is None:
        raise ImportError(
            f"compressed_tensors.converters.{cls_name} not found in the "
            f"installed compressed_tensors version. Bumping `compressed-tensors`"
            f" or pinning to a version that ships {cls_name} is required."
        )
    return converter_cls  # type: ignore[no-any-return]
