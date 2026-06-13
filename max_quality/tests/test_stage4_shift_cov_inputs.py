"""Stage-4 shift-cov input load (Task 3).

``EoraInputsPlugin.load_eora_inputs`` loads ``_stage3_shift_covariance.pt``
into the ``"shift_cov"`` ctx slot ONLY when
``stage4_eora.whitening_cov`` ∈ {"shift", "anchored_adaptive"}; the default
"anchor" path skips the load entirely (byte-identity). A requested-but-absent
artifact raises ``FileNotFoundError``.

Local helpers are redeclared on purpose (codebase discipline: tests do not
import from each other).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage4.plugins.eora_inputs import EoraInputsPlugin
from moe_compress.utils.atomic_io import atomic_torch_save, write_manifest_last


def _stub_model():
    """Minimal model whose iter_moe_layers yields nothing (non-MoE sentinel)."""
    class _StubLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = None  # rejected by _is_moe_layer (no .experts attr)

    m = nn.Module()
    m.layers = nn.ModuleList([_StubLayer()])
    return m


def _base_artifacts(tmp_path):
    """A2_cov + originals on disk + a ctx wired for load_eora_inputs."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    disk_payload = {
        "format_version": 1,
        "covariance": {(0, 0, "gate_proj"): torch.eye(3, dtype=torch.float16) * 2},
        "tokens": {(0, 0, "gate_proj"): 9},
    }
    torch.save(disk_payload, artifacts_dir / "_stage2_input_covariance.pt")
    torch.save({}, artifacts_dir / "_stage3_original_weights.pt")
    return artifacts_dir


def _write_shift_artifact(artifacts_dir, cov):
    out_path = artifacts_dir / "_stage3_shift_covariance.pt"
    atomic_torch_save(out_path, {"format_version": 1, "covariance": cov})
    manifest = out_path.with_suffix(out_path.suffix + ".MANIFEST.json")
    write_manifest_last(out_path, manifest, schema_version=1,
                        extra_meta={"n_keys": len(cov),
                                    "artifact": "stage3_shift_covariance"},
                        compute_sha256=False)
    return out_path


def _make_ctx(artifacts_dir, whitening_cov):
    config = {"stage2_reap_ream": {"covariance_storage_dtype": "float16"}}
    if whitening_cov is not None:
        config["stage4_eora"] = {"whitening_cov": whitening_cov}
    ctx = PipelineContext()
    ctx.set("artifacts_dir", artifacts_dir)
    ctx.set("config", config)
    ctx.set("model", _stub_model())
    return ctx


def test_shift_cov_loaded_when_requested(tmp_path):
    artifacts_dir = _base_artifacts(tmp_path)
    shift_cov = {
        (0, 0, "gate_proj"): torch.eye(4) * 13.0,
        (0, 0, "down_proj"): torch.eye(4) * 17.0,
    }
    _write_shift_artifact(artifacts_dir, shift_cov)

    ctx = _make_ctx(artifacts_dir, "shift")
    EoraInputsPlugin().load_eora_inputs(ctx)

    assert ctx.has("shift_cov")
    loaded = ctx.get("shift_cov")
    assert set(loaded.keys()) == set(shift_cov.keys())
    for key, tensor in loaded.items():
        assert tensor.shape == (4, 4)


@pytest.mark.parametrize("whitening_cov", [None, "anchor"])
def test_shift_cov_skipped_for_anchor(tmp_path, whitening_cov):
    artifacts_dir = _base_artifacts(tmp_path)
    # Even with the artifact present, the anchor/default path must skip it.
    _write_shift_artifact(artifacts_dir, {(0, 0, "gate_proj"): torch.eye(4)})

    ctx = _make_ctx(artifacts_dir, whitening_cov)
    EoraInputsPlugin().load_eora_inputs(ctx)

    assert not ctx.has("shift_cov")


def test_shift_cov_missing_raises(tmp_path):
    artifacts_dir = _base_artifacts(tmp_path)
    # No shift artifact written.
    ctx = _make_ctx(artifacts_dir, "shift")
    with pytest.raises(FileNotFoundError):
        EoraInputsPlugin().load_eora_inputs(ctx)
