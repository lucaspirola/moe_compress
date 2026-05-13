"""Numerics gate for the env-gated IQ2_XS bf16 argmin path (Group F).

When ``KDR_IQ2XS_BF16_ARGMIN=1`` the IQ2_XS codebook search runs the
``tile @ joint_flat_t`` inner-product in bf16 (≈2× throughput on B200
tensor cores) before recasting to fp32 for the
``code_sq - 2 * inner`` comparison.

The bf16 cast can flip argmins on near-tie codewords; this test asserts
the flips are rare and that aggregate reconstruction error is
comparable.

Acceptance:
  * ≥99% of chunks pick identical codewords vs. the fp32 baseline.
  * Reconstruction MSE delta (bf16 vs fp32) bounded by
    ``1e-3 × per-sub-block amax`` (squared) — a generous slack that
    still flags any systemic drift.

Skipped when CUDA is unavailable (the bf16 path is CUDA-only).
"""

from __future__ import annotations

import os

import pytest
import torch

from kdr.quant.native_backend.ste_simulators import iq2_xs_quant_snap

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bf16 argmin path is CUDA-only (see ste_simulators._iq2xs_argmin_use_bf16)",
)


def _snap(x: torch.Tensor, *, use_bf16: bool) -> torch.Tensor:
    """Run iq2_xs_quant_snap with the bf16 toggle wrapped in env state."""
    prev = os.environ.get("KDR_IQ2XS_BF16_ARGMIN")
    os.environ["KDR_IQ2XS_BF16_ARGMIN"] = "1" if use_bf16 else "0"
    try:
        return iq2_xs_quant_snap(x, axis=-1)
    finally:
        if prev is None:
            os.environ.pop("KDR_IQ2XS_BF16_ARGMIN", None)
        else:
            os.environ["KDR_IQ2XS_BF16_ARGMIN"] = prev


@pytest.mark.parametrize("seed", [1337, 9001, 4242])
def test_bf16_argmin_matches_fp32_baseline(seed: int) -> None:
    """≥99% identical chunks and bounded MSE delta vs. the fp32 baseline."""
    torch.manual_seed(seed)
    # Shape (256, 256): 256 "rows" × one IQ2_XS super-block (256 elements
    # along the in-features axis). Profile-J weights have many more rows
    # but the per-super-block snap is independent across rows, so this
    # subset is representative.
    x = torch.randn(256, 256, device="cuda", dtype=torch.float32)

    snap_fp32 = _snap(x, use_bf16=False)
    snap_bf16 = _snap(x, use_bf16=True)

    # Per-chunk equality: snapped values share the codebook → if argmins
    # agree, the snapped 8-element chunk is bit-identical.
    # Reshape to (..., 8) chunks and compare.
    fp32_chunks = snap_fp32.reshape(-1, 8)
    bf16_chunks = snap_bf16.reshape(-1, 8)
    same = (fp32_chunks == bf16_chunks).all(dim=-1)  # per-chunk bool
    pct_same = same.float().mean().item()
    assert pct_same >= 0.99, (
        f"Only {pct_same:.4f} of chunks matched the fp32 baseline; "
        "expected ≥0.99. bf16 argmin instability suspected."
    )

    # MSE delta vs. fp32 reconstruction, normalised by per-input variance.
    diff = (snap_bf16 - snap_fp32).float()
    mse_delta = (diff * diff).mean().item()
    x_var = x.float().var().item()
    # Allow at most 1e-3 × variance of the input — a near-tie flip
    # contributes only a small per-chunk error so this is a comfortable
    # cap that still catches systemic drift.
    cap = 1e-3 * x_var
    assert mse_delta <= cap, (
        f"MSE delta {mse_delta:.6g} exceeds cap {cap:.6g} "
        f"(x.var={x_var:.6g}); bf16 path has degraded reconstruction."
    )


def test_bf16_disabled_by_default() -> None:
    """Default (env var unset / 0) must take the fp32 path → exact match."""
    torch.manual_seed(7)
    x = torch.randn(32, 256, device="cuda", dtype=torch.float32)

    prev = os.environ.pop("KDR_IQ2XS_BF16_ARGMIN", None)
    try:
        snap_default = iq2_xs_quant_snap(x, axis=-1)
    finally:
        if prev is not None:
            os.environ["KDR_IQ2XS_BF16_ARGMIN"] = prev

    snap_fp32 = _snap(x, use_bf16=False)
    assert torch.equal(snap_default, snap_fp32), (
        "Default behaviour drifted from KDR_IQ2XS_BF16_ARGMIN=0 path — "
        "the env-gate default must reproduce the fp32 baseline exactly."
    )
