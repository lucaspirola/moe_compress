"""Property tests for NativeBackend GGUF parametrization classes (LLR-0055).

Each of the four GGUF parametrization classes wraps a pure STE call.
The parametrization's ``forward(w)`` must be bit-identical to a direct
call of the underlying STE with ``axis=-1`` — anything else is a wiring
bug.

# REQ: LLR-0055
# VERIFIES: LLR-0055
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from kdr.quant.native_backend.backend import (
    _IQ2XSQuantWeight,
    _IQ4XSQuantWeight,
    _Q3KQuantWeight,
    _Q5KQuantWeight,
)
from kdr.quant.native_backend.ste_simulators import (
    iq2_xs_quant_ste,
    iq4_xs_quant_ste,
    q3_k_quant_ste,
    q5_k_quant_ste,
)

# Shape: last axis must be a multiple of 256 per LLR-0015. Smallest valid
# value (256) keeps the property tests fast.
_SHAPES = st.one_of(
    st.tuples(st.integers(min_value=1, max_value=4), st.just(256)),
    st.tuples(
        st.integers(min_value=1, max_value=2),
        st.integers(min_value=1, max_value=3),
        st.just(256),
    ),
)


_CASES = (
    (_IQ2XSQuantWeight, iq2_xs_quant_ste),
    (_Q3KQuantWeight, q3_k_quant_ste),
    (_IQ4XSQuantWeight, iq4_xs_quant_ste),
    (_Q5KQuantWeight, q5_k_quant_ste),
)


@settings(deadline=None, max_examples=16)
@given(shape=_SHAPES)
def test_iq2xs_param_matches_direct_ste(shape: tuple[int, ...]) -> None:
    w = torch.randn(*shape)
    direct = iq2_xs_quant_ste(w, axis=-1)
    via_param = _IQ2XSQuantWeight().forward(w)
    assert torch.equal(direct, via_param)


@settings(deadline=None, max_examples=16)
@given(shape=_SHAPES)
def test_q3k_param_matches_direct_ste(shape: tuple[int, ...]) -> None:
    w = torch.randn(*shape)
    direct = q3_k_quant_ste(w, axis=-1)
    via_param = _Q3KQuantWeight().forward(w)
    assert torch.equal(direct, via_param)


@settings(deadline=None, max_examples=16)
@given(shape=_SHAPES)
def test_iq4xs_param_matches_direct_ste(shape: tuple[int, ...]) -> None:
    w = torch.randn(*shape)
    direct = iq4_xs_quant_ste(w, axis=-1)
    via_param = _IQ4XSQuantWeight().forward(w)
    assert torch.equal(direct, via_param)


@settings(deadline=None, max_examples=16)
@given(shape=_SHAPES)
def test_q5k_param_matches_direct_ste(shape: tuple[int, ...]) -> None:
    w = torch.randn(*shape)
    direct = q5_k_quant_ste(w, axis=-1)
    via_param = _Q5KQuantWeight().forward(w)
    assert torch.equal(direct, via_param)
