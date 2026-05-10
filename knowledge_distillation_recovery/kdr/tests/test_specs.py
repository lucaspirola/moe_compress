"""Pydantic validation tests for `kdr.quant.specs` (LLR-0009, LLR-0010, LLR-0012).

These tests run today even though the rest of kdr is stubs — Pydantic models
are inherently fully-defined the moment they exist.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kdr.quant.specs import KVQuantSpec, WeightQuantSpec


def _valid_kv_kwargs() -> dict[str, object]:
    return {"bits": 4, "format": "int", "granularity": "channel", "transform": "none"}


def _valid_weight_kwargs() -> dict[str, object]:
    return {"bits": 4, "format": "nvfp4", "granularity": "block", "transform": "none"}


# ─── KVQuantSpec ─────────────────────────────────────────────────────────────


def test_kv_spec_accepts_minimal_valid_config() -> None:
    spec = KVQuantSpec(**_valid_kv_kwargs())  # type: ignore[arg-type]
    assert spec.bits == 4
    assert spec.format == "int"
    assert spec.granularity == "channel"
    assert spec.transform == "none"


def test_kv_spec_rejects_missing_field() -> None:
    kwargs = _valid_kv_kwargs()
    del kwargs["bits"]
    with pytest.raises(ValidationError):
        KVQuantSpec(**kwargs)  # type: ignore[arg-type]


def test_kv_spec_rejects_unknown_field() -> None:
    kwargs = _valid_kv_kwargs()
    kwargs["nbits"] = 4  # typo for `bits`
    with pytest.raises(ValidationError):
        KVQuantSpec(**kwargs)  # type: ignore[arg-type]


def test_kv_spec_rejects_unknown_format() -> None:
    kwargs = _valid_kv_kwargs()
    kwargs["format"] = "int8"  # not in the Format Literal
    with pytest.raises(ValidationError):
        KVQuantSpec(**kwargs)  # type: ignore[arg-type]


def test_kv_spec_rejects_unknown_granularity() -> None:
    kwargs = _valid_kv_kwargs()
    kwargs["granularity"] = "head"
    with pytest.raises(ValidationError):
        KVQuantSpec(**kwargs)  # type: ignore[arg-type]


def test_kv_spec_rejects_v1_transforms_in_v0() -> None:
    """Hadamard / FWHT are deferred to v1+ per the plan; v0 schema accepts only `none`."""
    for deferred in ("hadamard", "fwht"):
        kwargs = _valid_kv_kwargs()
        kwargs["transform"] = deferred
        with pytest.raises(ValidationError):
            KVQuantSpec(**kwargs)  # type: ignore[arg-type]


def test_kv_spec_rejects_string_for_int_field() -> None:
    """Strict mode: no implicit string→int coercion."""
    kwargs = _valid_kv_kwargs()
    kwargs["bits"] = "4"  # YAML libs sometimes hand back strings; strict rejects.
    with pytest.raises(ValidationError):
        KVQuantSpec(**kwargs)  # type: ignore[arg-type]


# ─── WeightQuantSpec ─────────────────────────────────────────────────────────


def test_weight_spec_accepts_symmetric_int4() -> None:
    spec = WeightQuantSpec(bits=4, format="int", granularity="channel", transform="none")
    assert spec.bits == 4


def test_weight_spec_rejects_unknown_field() -> None:
    kwargs = _valid_weight_kwargs()
    kwargs["asymmetric"] = True
    with pytest.raises(ValidationError):
        WeightQuantSpec(**kwargs)  # type: ignore[arg-type]


def test_weight_spec_rejects_v1_transforms() -> None:
    for deferred in ("hadamard", "fwht"):
        kwargs = _valid_weight_kwargs()
        kwargs["transform"] = deferred
        with pytest.raises(ValidationError):
            WeightQuantSpec(**kwargs)  # type: ignore[arg-type]
