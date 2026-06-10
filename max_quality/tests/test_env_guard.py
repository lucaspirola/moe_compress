"""Tests for the GDN-training environment guard (utils/env_guard.py).

The guard must fire EXACTLY on the unsupported combo — Hopper + Triton>=3.4 +
no tilelang (i.e. CUDA-13) — and no-op everywhere else, so it neither wastes
hours of Stage 1+2 before the inevitable Router-KD backward crash nor blocks
valid forward-only / CUDA-12 / off-Hopper runs.
"""
import pytest

from moe_compress.utils.env_guard import _gdn_reason

HOPPER = (9, 0)
BLACKWELL = (12, 0)
AMPERE = (8, 0)


def test_fires_on_hopper_triton34_no_tilelang():
    # The exact CUDA-13 failure mode that burned hours on 2026-06-10.
    reason = _gdn_reason(cap=HOPPER, triton_version="3.6.0", tilelang_present=False, cuda_build="13.0")
    assert reason is not None
    assert "UNSUPPORTED" in reason
    assert "tilelang" in reason and "CUDA-12" in reason  # message points at the fix


def test_fires_on_hopper_with_unknown_triton_failsafe():
    # Unknown triton version → assume >=3.4 (fail-safe), still fires.
    reason = _gdn_reason(cap=HOPPER, triton_version=None, tilelang_present=False, cuda_build="13.0")
    assert reason is not None


@pytest.mark.parametrize(
    "cap,triton,tilelang,why",
    [
        (HOPPER, "3.6.0", True, "tilelang present (CUDA-12 box) → correct backend"),
        (HOPPER, "3.3.0", False, "Triton < 3.4 doesn't trip the fla raise"),
        (BLACKWELL, "3.6.0", False, "off-Hopper: Triton GDN backward is correct"),
        (AMPERE, "3.6.0", False, "off-Hopper: Triton GDN backward is correct"),
        (None, "3.6.0", False, "no CUDA device (CPU/meta)"),
    ],
)
def test_noop_on_supported_envs(cap, triton, tilelang, why):
    assert _gdn_reason(cap=cap, triton_version=triton, tilelang_present=tilelang, cuda_build="x") is None, why


def test_assert_does_not_raise_off_hopper():
    # The live wrapper must be a no-op on this (non-Hopper / CPU) test host.
    from moe_compress.utils.env_guard import assert_gdn_training_supported

    assert_gdn_training_supported(context="unit-test")  # must not raise here
