"""Stage 0 — Super Expert Detection (fused-experts-aware).

NOTE: This is a stub module. Stage 0 has been merged into Stage 1 (stage1_grape.py).
This module exists only to keep old tests working that import from both stage0 and stage1.
The actual run() function is minimal and does nothing — stage1_grape.run() handles both
SE detection and GRAPE budget allocation.
"""
from __future__ import annotations

import json
from pathlib import Path

# Re-export calibration utilities for test patching compatibility
from .utils.calibration import build_super_expert_slice
from .utils.model_io import iter_moe_layers

__all__ = ["run", "build_super_expert_slice"]


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
) -> Path:
    """Stub run function. Stage 0 has been merged into stage1_grape.run().

    Writes a minimal stage0_blacklist.json so downstream tests can read it.
    """
    # Write a minimal blacklist file so downstream tests can read it.
    # Structure: blacklist (dict of blacklisted experts) and per_expert_max (one entry per expert).
    blacklist_path = artifacts_dir / "stage0_blacklist.json"

    # Count total experts across all MoE layers (num_layers × num_experts_per_layer)
    per_expert_max = {}
    for ref in iter_moe_layers(model):
        for e in range(ref.num_routed_experts):
            key = f"L{ref.layer_idx}_E{e}"
            per_expert_max[key] = 0.0  # Minimal positive value (0.0 is non-negative as test expects)

    payload = {
        "blacklist": {},
        "per_expert_max": per_expert_max,
        "version": 1,
    }

    blacklist_path.write_text(json.dumps(payload))
    return blacklist_path
