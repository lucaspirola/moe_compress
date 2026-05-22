"""Unit tests for ``moe_compress.pipeline.safe_json``."""

from __future__ import annotations

import math

from moe_compress.pipeline.safe_json import safe_float


def test_safe_float_passes_finite_floats():
    assert safe_float(0.5) == 0.5
    assert safe_float(-1.25) == -1.25
    assert safe_float(0.0) == 0.0


def test_safe_float_converts_int_to_float():
    out = safe_float(3)
    assert out == 3.0
    assert isinstance(out, float)


def test_safe_float_nan_returns_none():
    assert safe_float(float("nan")) is None


def test_safe_float_pos_inf_returns_none():
    assert safe_float(float("inf")) is None


def test_safe_float_neg_inf_returns_none():
    assert safe_float(float("-inf")) is None


def test_safe_float_isfinite_consistency():
    """Sanity: every finite ``float`` round-trips; every non-finite hits ``None``."""
    for x in (0.0, 1.0, -1.0, 1e-300, 1e300):
        assert math.isfinite(x)
        assert safe_float(x) == x
