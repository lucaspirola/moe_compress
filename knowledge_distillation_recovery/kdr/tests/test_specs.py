"""Pydantic validation tests for `kdr.quant.specs` (LLR-0009, LLR-0010, LLR-0012).

These tests run today even though the rest of kdr is stubs — Pydantic models
are inherently fully-defined the moment they exist.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kdr.quant.specs import (
    KVQuantSpec,
    MixedWeightSpec,
    UniformWeightSpec,
    WeightPatternSpec,
    WeightQuantSpec,
)


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


# ─── WeightPatternSpec (Phase 7.2 Task 2) ────────────────────────────────────


def _valid_pattern_kwargs() -> dict[str, object]:
    return {
        "pattern": "experts.mlp",
        "bits": 4,
        "format": "iq4_xs",
        "granularity": "block",
        "transform": "none",
    }


def test_weight_pattern_spec_accepts_valid_config() -> None:
    spec = WeightPatternSpec(**_valid_pattern_kwargs())  # type: ignore[arg-type]
    assert spec.pattern == "experts.mlp"
    assert spec.bits == 4
    assert spec.format == "iq4_xs"
    assert spec.granularity == "block"
    assert spec.transform == "none"


def test_weight_pattern_spec_rejects_missing_field() -> None:
    kwargs = _valid_pattern_kwargs()
    del kwargs["pattern"]
    with pytest.raises(ValidationError):
        WeightPatternSpec(**kwargs)  # type: ignore[arg-type]


def test_weight_pattern_spec_rejects_unknown_field() -> None:
    kwargs = _valid_pattern_kwargs()
    kwargs["scale"] = 1.0
    with pytest.raises(ValidationError):
        WeightPatternSpec(**kwargs)  # type: ignore[arg-type]


def test_weight_pattern_spec_rejects_v1_transforms() -> None:
    for deferred in ("hadamard", "fwht"):
        kwargs = _valid_pattern_kwargs()
        kwargs["transform"] = deferred
        with pytest.raises(ValidationError):
            WeightPatternSpec(**kwargs)  # type: ignore[arg-type]


def test_weight_pattern_spec_rejects_unknown_format() -> None:
    kwargs = _valid_pattern_kwargs()
    kwargs["format"] = "fp4"  # not in the Format Literal
    with pytest.raises(ValidationError):
        WeightPatternSpec(**kwargs)  # type: ignore[arg-type]


# ─── MixedWeightSpec (Phase 7.2 Task 2) ──────────────────────────────────────


def test_mixed_weight_spec_accepts_one_entry_spec_map() -> None:
    spec = MixedWeightSpec(
        spec_map=[WeightPatternSpec(**_valid_pattern_kwargs())]  # type: ignore[arg-type]
    )
    assert len(spec.spec_map) == 1
    assert spec.spec_map[0].pattern == "experts.mlp"


def test_mixed_weight_spec_rejects_empty_spec_map() -> None:
    with pytest.raises(ValidationError):
        MixedWeightSpec(spec_map=[])


def test_mixed_weight_spec_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        MixedWeightSpec(
            spec_map=[WeightPatternSpec(**_valid_pattern_kwargs())],  # type: ignore[arg-type]
            default="iq4_xs",  # type: ignore[call-arg]
        )


def test_mixed_weight_spec_rejects_duplicate_patterns() -> None:
    """The `_no_duplicate_patterns` field validator catches identical patterns
    inside spec_map at parse time."""
    entry = WeightPatternSpec(**_valid_pattern_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        MixedWeightSpec(spec_map=[entry, entry])


# ─── Alias: WeightQuantSpec resolves to UniformWeightSpec ────────────────────


def test_weight_quant_spec_alias_resolves_to_uniform_weight_spec() -> None:
    """`WeightQuantSpec` is a re-bound alias for `UniformWeightSpec`. Existing
    imports continue to resolve and `isinstance` checks remain consistent."""
    assert WeightQuantSpec is UniformWeightSpec
