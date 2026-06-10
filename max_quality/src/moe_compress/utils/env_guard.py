"""Environment guards that fail loudly *before* expensive work.

The motivating failure (2026-06-10): a full pipeline run was provisioned on a
CUDA-13 H200, ran Stage 1 + Stage 2 for ~an hour, and would only have crashed
at the *first* Router-KD ``loss.backward()`` in Stage 2.5 — because fla's gated
DeltaNet backward (``chunk_bwd_dqkwg``) **raises on Hopper with Triton >= 3.4**
unless ``tilelang`` is installed (fla PR #827 / issue #640), and ``tilelang``
**cannot install on CUDA-13** (its ``apache-tvm-ffi`` double-registers the
tvm-ffi runtime that torch's ``torch_c_dlpack_ext`` already loaded → SIGABRT).

This module turns that latent, hours-deep failure into an immediate, legible
one.  The check is intentionally *precise* (Hopper + Triton>=3.4 + tilelang
absent) rather than a blunt "CUDA major == 13": inference is fine there, and a
CUDA-12 box with tilelang trains correctly.
"""
from __future__ import annotations

import importlib.util
import logging

log = logging.getLogger(__name__)

_HOPPER_CAPS = {(9, 0)}  # H100 / H200


def _gdn_reason(
    *,
    cap: tuple[int, ...] | None,
    triton_version: str | None,
    tilelang_present: bool,
    cuda_build: str | None,
) -> str | None:
    """Pure decision: reason string if GDN training is unsupported, else None.

    Separated from the environment-reading wrapper so it is unit-testable
    without a GPU. ``triton_version=None`` means "unknown" → treated as modern
    (>=3.4) to stay fail-safe (Hopper + unknown-triton + no tilelang fails).
    """
    if cap is None or tuple(cap) not in _HOPPER_CAPS:
        return None  # off-Hopper: fla's Triton gated-GDN backward is correct (no #827 guard)

    if triton_version is not None:
        try:
            from packaging.version import Version

            if Version(triton_version) < Version("3.4.0"):
                return None  # pre-3.4 triton doesn't trip the fla raise
        except Exception:  # noqa: BLE001 — unparseable → stay fail-safe (assume >=3.4)
            pass

    if tilelang_present:
        return None  # tilelang installed → the correct gated-GDN backend is available

    return (
        "Gated-DeltaNet (GDN) backward is UNSUPPORTED in this environment — "
        "Router-KD training WILL crash at loss.backward().\n"
        f"  detected: device_capability={tuple(cap)} (Hopper) "
        f"| triton={triton_version or 'unknown(>=3.4 assumed)'} "
        f"| torch.cuda={cuda_build or '?'} | tilelang=ABSENT\n"
        "  why: on Hopper (H100/H200) with Triton>=3.4, fla's chunk_bwd_dqkwg RAISES "
        "and requires `tilelang` (fla PR #827 / issue #640). `tilelang` does NOT install "
        "on CUDA-13 (apache-tvm-ffi double-registers tvm-ffi -> SIGABRT).\n"
        "  FIX: run the training stages on a CUDA-12 box where tilelang installs "
        "(requirements.txt `tilelang>=0.1.7` + scripts/setup_gpu_env.sh). Forward/"
        "inference-only stages are unaffected."
    )


def gdn_training_unsupported_reason() -> str | None:
    """Return a human-readable reason if Router-KD training will fail here, else None.

    Pure detection (no side effects, never raises). Reads torch/triton/tilelang
    from the live environment and delegates the decision to :func:`_gdn_reason`.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 — torch must exist for training; absence isn't our concern
        return None
    if not torch.cuda.is_available():
        return None  # CPU/meta — not a GPU training run
    try:
        cap = tuple(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        return None

    triton_version: str | None
    try:
        import triton

        triton_version = triton.__version__
    except Exception:  # noqa: BLE001 — assume a modern triton if unknowable
        triton_version = None

    # find_spec does NOT import tilelang — importing it on CUDA-13 would SIGABRT.
    tilelang_present = importlib.util.find_spec("tilelang") is not None

    try:
        cuda_build = torch.version.cuda
    except Exception:  # noqa: BLE001
        cuda_build = None

    return _gdn_reason(
        cap=cap,
        triton_version=triton_version,
        tilelang_present=tilelang_present,
        cuda_build=cuda_build,
    )


def assert_gdn_training_supported(context: str = "") -> None:
    """Raise RuntimeError if Router-KD training cannot run correctly here.

    Call at the entry of any stage that does a GDN backward (Router-KD training:
    Stage 2.5 heal + Stage 5). No-op on supported environments (off-Hopper,
    CUDA-12 + tilelang, CPU).
    """
    reason = gdn_training_unsupported_reason()
    if reason is None:
        return
    prefix = f"[{context}] " if context else ""
    raise RuntimeError(f"{prefix}{reason}")
