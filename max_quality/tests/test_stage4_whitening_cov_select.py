"""Stage-4 whitening-cov selector (Task 4).

``_resolve_whitening_lookup`` picks which covariance dict EoRA whitens with,
per ``stage4_eora.whitening_cov``:
  "anchor" (default) -> A_cov  (byte-identical to history; SAME object)
  "shift"            -> shift_cov
  "anchored_adaptive"-> _AnchoredAdaptiveLookup(A_cov, shift_cov)
Unknown value raises ValueError; "shift" with no shift_cov raises ValueError.
"""
from __future__ import annotations

import pytest
import torch

from moe_compress.stage4.plugins.eora_compensation import (
    _AnchoredAdaptiveLookup,
    _resolve_whitening_lookup,
)


def _dicts():
    a = {(0, 0, "gate_proj"): torch.eye(3) * 2.0}
    s = {(0, 0, "gate_proj"): torch.eye(3) * 6.0}
    return a, s


def test_anchor_returns_same_object():
    a, s = _dicts()
    out = _resolve_whitening_lookup("anchor", a, s)
    assert out is a  # byte-identity: the SAME dict object


def test_shift_returns_shift_dict():
    a, s = _dicts()
    out = _resolve_whitening_lookup("shift", a, s)
    assert out is s


def test_shift_without_shift_cov_raises():
    a, _ = _dicts()
    with pytest.raises(ValueError):
        _resolve_whitening_lookup("shift", a, None)


def test_unknown_value_raises():
    a, s = _dicts()
    with pytest.raises(ValueError):
        _resolve_whitening_lookup("bogus", a, s)


def test_anchored_adaptive_blends():
    a, s = _dicts()
    out = _resolve_whitening_lookup("anchored_adaptive", a, s)
    assert isinstance(out, _AnchoredAdaptiveLookup)
    blended = out.get((0, 0, "gate_proj"))
    # 0.5*(2I + 6I) = 4I
    assert torch.allclose(blended, torch.eye(3) * 4.0)


def test_anchored_adaptive_falls_back_to_present_side():
    a = {(0, 0, "gate_proj"): torch.eye(3) * 2.0}
    s = {(0, 1, "gate_proj"): torch.eye(3) * 9.0}
    out = _AnchoredAdaptiveLookup(a, s)
    # anchor-only key -> anchor value; shift-only key -> shift value
    assert torch.equal(out.get((0, 0, "gate_proj")), torch.eye(3) * 2.0)
    assert torch.equal(out.get((0, 1, "gate_proj")), torch.eye(3) * 9.0)
    assert out.get((9, 9, "gate_proj")) is None
