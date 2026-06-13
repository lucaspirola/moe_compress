"""Stage-3 shift-cov consolidation helper (Task 1 / Task 2).

Covers ``_consolidate_shift_covariance``: merges per-layer B-cov spill files
(the post-2.5 SHIFT cov ``S = X'ᵀX'``) into one durable named artifact +
manifest for Stage-4 EoRA shift whitening.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_compress.stage3.plugins.covariance_collection import (  # noqa: E402
    _consolidate_shift_covariance,
)
from moe_compress.utils.activation_hooks import (  # noqa: E402
    InputCovarianceAccumulator,
)


def _write_spill(spill_dir: Path, layer_idx: int, cov: dict) -> None:
    """Write a format_version-1 per-layer spill via the real accumulator."""
    acc = InputCovarianceAccumulator(storage_dtype=torch.bfloat16)
    for key, tensor in cov.items():
        acc.covariance[key] = tensor
        acc.token_count[key] = 1
    acc.spill_layer_to_disk(layer_idx, spill_dir)


def test_consolidate_shift_covariance_merges_keys(tmp_path):
    spill_dir = tmp_path / "_stage3_bcov_partial"
    spill_dir.mkdir()
    d = 4
    cov0 = {
        (0, 0, "gate_proj"): torch.eye(d) * 2.0,
        (0, 0, "down_proj"): torch.eye(d) * 3.0,
        (0, 1, "gate_proj"): torch.eye(d) * 5.0,
    }
    cov1 = {
        (1, 0, "gate_proj"): torch.eye(d) * 7.0,
        (1, 0, "down_proj"): torch.eye(d) * 11.0,
    }
    _write_spill(spill_dir, 0, cov0)
    _write_spill(spill_dir, 1, cov1)

    out_path = tmp_path / "_stage3_shift_covariance.pt"
    n = _consolidate_shift_covariance(
        spill_dir, out_path, [0, 1], storage_dtype=torch.bfloat16
    )

    assert n == 5
    assert out_path.exists()
    manifest = out_path.with_suffix(out_path.suffix + ".MANIFEST.json")
    assert manifest.exists()

    payload = torch.load(out_path, map_location="cpu", weights_only=True)
    merged = payload["covariance"]
    assert payload["format_version"] == 1
    assert set(merged.keys()) == set(cov0.keys()) | set(cov1.keys())
    for key, tensor in merged.items():
        assert tensor.shape == (d, d)
        assert tensor.dtype == torch.bfloat16


def test_consolidate_skips_missing_layers(tmp_path):
    spill_dir = tmp_path / "_stage3_bcov_partial"
    spill_dir.mkdir()
    d = 3
    _write_spill(spill_dir, 0, {(0, 0, "gate_proj"): torch.eye(d)})
    out_path = tmp_path / "_stage3_shift_covariance.pt"

    # Layer 2 has no spill file — must be skipped, not raise.
    n = _consolidate_shift_covariance(
        spill_dir, out_path, [0, 2], storage_dtype=torch.bfloat16
    )
    assert n == 1
    payload = torch.load(out_path, map_location="cpu", weights_only=True)
    assert set(payload["covariance"].keys()) == {(0, 0, "gate_proj")}
