"""LLR-0063: vectorised codebook unpack is bit-identical to Python-loop form.

# REQ: LLR-0063
# VERIFIES: LLR-0063
"""

from __future__ import annotations

import pytest
import torch

from kdr.quant.native_backend.gguf_codebooks import (
    IQ2XS_GRID_RAW,
    KSIGNS_IQ2XS,
    _IQ2XS_GRID_CACHE,
    _KSIGNS_IQ2XS_CACHE,
    get_iq2xs_grid,
    get_ksigns_iq2xs,
)


def _ref_iq2xs_grid(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Pre-refactor Python-loop reference replay of get_iq2xs_grid."""
    flat: list[int] = []
    for val in IQ2XS_GRID_RAW:
        for i in range(8):
            flat.append((val >> (i * 8)) & 0xFF)
    return torch.tensor(flat, dtype=dtype, device=device).view(512, 8)


def _ref_ksigns_iq2xs(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Pre-refactor Python-loop reference replay of get_ksigns_iq2xs."""
    expanded: list[list[float]] = []
    for bits in KSIGNS_IQ2XS:
        row = [-1.0 if (bits >> i) & 1 else 1.0 for i in range(8)]
        expanded.append(row)
    return torch.tensor(expanded, dtype=dtype, device=device)


@pytest.fixture(autouse=True)
def _preserve_codebook_caches() -> object:
    """Snapshot and restore the module-level caches around each test.

    Without this, `_IQ2XS_GRID_CACHE.clear()` (needed so the test
    triggers a fresh construction) would leak across tests in the
    full pytest session and break any test elsewhere that assumes
    the cache already has a fresh tensor on a given (device, dtype).
    """
    snap_grid = dict(_IQ2XS_GRID_CACHE)
    snap_ksigns = dict(_KSIGNS_IQ2XS_CACHE)
    _IQ2XS_GRID_CACHE.clear()
    _KSIGNS_IQ2XS_CACHE.clear()
    try:
        yield
    finally:
        _IQ2XS_GRID_CACHE.clear()
        _KSIGNS_IQ2XS_CACHE.clear()
        _IQ2XS_GRID_CACHE.update(snap_grid)
        _KSIGNS_IQ2XS_CACHE.update(snap_ksigns)


def test_iq2xs_grid_bit_identical_to_python_loop_cpu() -> None:
    """LLR-0063 AC: vectorised get_iq2xs_grid bit-identical to the
    Python-loop reference on CPU (value comparison)."""
    out_new = get_iq2xs_grid(torch.device("cpu"), torch.float32)
    out_ref = _ref_iq2xs_grid(torch.device("cpu"), torch.float32)
    assert torch.equal(out_new, out_ref)


def test_ksigns_iq2xs_bit_identical_to_python_loop_cpu() -> None:
    """LLR-0063 AC: vectorised get_ksigns_iq2xs bit-identical on CPU."""
    out_new = get_ksigns_iq2xs(torch.device("cpu"), torch.float32)
    out_ref = _ref_ksigns_iq2xs(torch.device("cpu"), torch.float32)
    assert torch.equal(out_new, out_ref)


def test_iq2xs_grid_meta_device_returns_correct_shape_and_dtype() -> None:
    """LLR-0063 AC: meta-device dispatch path exercises the cache+
    construct plumbing without real hardware. Meta tensors have no
    storage so `torch.equal` is N/A; verify shape + dtype + cache
    identity (a second call must return the same tensor object)."""
    out_new = get_iq2xs_grid(torch.device("meta"), torch.float32)
    assert out_new.shape == (512, 8)
    assert out_new.dtype == torch.float32
    assert out_new.device.type == "meta"
    # Cache contract: second call returns the same tensor object.
    out_again = get_iq2xs_grid(torch.device("meta"), torch.float32)
    assert out_again is out_new


def test_ksigns_iq2xs_meta_device_returns_correct_shape_and_dtype() -> None:
    """LLR-0063 AC: meta-device plumbing for ksigns — shape, dtype,
    and cache identity on a second call."""
    out_new = get_ksigns_iq2xs(torch.device("meta"), torch.float32)
    assert out_new.shape == (128, 8)
    assert out_new.dtype == torch.float32
    assert out_new.device.type == "meta"
    out_again = get_ksigns_iq2xs(torch.device("meta"), torch.float32)
    assert out_again is out_new


def test_iq2xs_grid_cache_returns_same_object() -> None:
    """LLR-0063 AC: cache identity preserved — second call returns the
    same tensor object, not a new construction."""
    a = get_iq2xs_grid(torch.device("cpu"), torch.float32)
    b = get_iq2xs_grid(torch.device("cpu"), torch.float32)
    assert a is b


def test_ksigns_iq2xs_cache_returns_same_object() -> None:
    """LLR-0063 AC: cache identity preserved for ksigns."""
    a = get_ksigns_iq2xs(torch.device("cpu"), torch.float32)
    b = get_ksigns_iq2xs(torch.device("cpu"), torch.float32)
    assert a is b
