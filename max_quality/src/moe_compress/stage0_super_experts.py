"""Stage 0 — Super Expert Detection (fused-experts-aware).

NOTE: This is a stub module. Stage 0 has been merged into Stage 1 (stage1_grape.py).
This module exists only to keep old tests working that import from both stage0 and stage1.
The actual run() function is minimal and does nothing — stage1_grape.run() handles both
SE detection and GRAPE budget allocation.
"""
from __future__ import annotations

from pathlib import Path

# Re-export calibration utilities for test patching compatibility
from .utils.calibration import build_super_expert_slice

__all__ = ["run", "build_super_expert_slice"]


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
) -> Path:
    """Stub run function. Stage 0 has been merged into stage1_grape.run()."""
    # Stage 0 detection is now integrated into stage1, so this is a no-op.
    return artifacts_dir
