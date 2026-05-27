"""cov_storage_dtype cross-validation for the Stage 2 profile-pass sidecar.

Covers plan §8.5. The driver flag ``--stage2-profile-cov-storage-dtype``
sets the writer's ``InputCovarianceAccumulator.storage_dtype`` AND is
embedded in the sidecar's ``cov_storage_dtype`` field. The Stage 2
reader's ``on_load`` cross-validates the sidecar value against the run's
configured ``covariance_storage_dtype`` (YAML knob, default float16).
A mismatch raises ``ValueError`` with the "Delete the sidecar to
regenerate" message.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    Stage2ProfilePayloadV3,
    load_stage2_profile_v3,
    save_stage2_profile_v3,
    sidecar_path,
)


def _jsonl(tmp_path: Path) -> Path:
    return tmp_path / "calib" / "self_traces.jsonl"


def _build_payload(cov_storage_dtype: str) -> Stage2ProfilePayloadV3:
    n_layers, n_experts, hidden = 1, 2, 4
    cov_dtype = getattr(torch, cov_storage_dtype)
    return Stage2ProfilePayloadV3(
        format_version=3,
        schema_version=SCHEMA_VERSIONS["stage2_profile"],
        model_hash="cov-dtype-test",
        n_layers=n_layers,
        n_experts=n_experts,
        top_k=2,
        cov_storage_dtype=cov_storage_dtype,
        total_tokens_per_layer=torch.zeros((n_layers,), dtype=torch.int64),
        gate_logit_profiles={0: []},
        sim_tensor=torch.zeros((n_layers, n_experts, n_experts), dtype=torch.float64),
        neuron_act_sum={},
        neuron_act_count={},
        cov_acc={
            (0, e, m): torch.eye(hidden, dtype=cov_dtype)
            for e in range(n_experts) for m in ("gate_proj", "down_proj")
        },
        cov_token_count={
            (0, e, m): 1
            for e in range(n_experts) for m in ("gate_proj", "down_proj")
        },
        layer_input_reservoir=[
            torch.zeros((4, hidden), dtype=torch.bfloat16)
            for _ in range(n_layers)
        ],
    )


def test_matching_dtype_loads_ok(tmp_path):
    """A sidecar with matching cov_storage_dtype loads without error."""
    jsonl = _jsonl(tmp_path)
    save_stage2_profile_v3(_build_payload("float16"), jsonl)
    loaded = load_stage2_profile_v3(
        jsonl, expected_cov_storage_dtype="float16",
    )
    assert loaded is not None
    assert loaded.cov_storage_dtype == "float16"


def test_mismatched_dtype_raises(tmp_path):
    """Mismatch between sidecar cov_storage_dtype and run config raises."""
    jsonl = _jsonl(tmp_path)
    save_stage2_profile_v3(_build_payload("float16"), jsonl)
    with pytest.raises(ValueError, match="Delete the sidecar to regenerate"):
        load_stage2_profile_v3(jsonl, expected_cov_storage_dtype="bfloat16")


def test_mismatched_dtype_fp32(tmp_path):
    """fp32 run requesting fp16 sidecar also fails."""
    jsonl = _jsonl(tmp_path)
    save_stage2_profile_v3(_build_payload("float16"), jsonl)
    with pytest.raises(ValueError, match="Delete the sidecar to regenerate"):
        load_stage2_profile_v3(jsonl, expected_cov_storage_dtype="float32")


def test_bfloat16_sidecar_roundtrips(tmp_path):
    """bfloat16 sidecar dtype survives the round-trip and validates."""
    jsonl = _jsonl(tmp_path)
    save_stage2_profile_v3(_build_payload("bfloat16"), jsonl)
    loaded = load_stage2_profile_v3(
        jsonl, expected_cov_storage_dtype="bfloat16",
    )
    assert loaded is not None
    assert loaded.cov_storage_dtype == "bfloat16"
    # cov entries actually stored in bf16 on disk → loaded as bf16.
    sample_key = next(iter(loaded.cov_acc))
    assert loaded.cov_acc[sample_key].dtype == torch.bfloat16


def test_cross_validation_n_layers_mismatch(tmp_path):
    """n_layers mismatch fails loud with the standard message."""
    jsonl = _jsonl(tmp_path)
    save_stage2_profile_v3(_build_payload("float16"), jsonl)
    with pytest.raises(ValueError, match="Delete the sidecar to regenerate"):
        load_stage2_profile_v3(jsonl, expected_n_layers=99)


def test_cross_validation_n_experts_mismatch(tmp_path):
    """n_experts mismatch fails loud."""
    jsonl = _jsonl(tmp_path)
    save_stage2_profile_v3(_build_payload("float16"), jsonl)
    with pytest.raises(ValueError, match="Delete the sidecar to regenerate"):
        load_stage2_profile_v3(jsonl, expected_n_experts=99)


def test_cross_validation_top_k_mismatch(tmp_path):
    """top_k mismatch fails loud."""
    jsonl = _jsonl(tmp_path)
    save_stage2_profile_v3(_build_payload("float16"), jsonl)
    with pytest.raises(ValueError, match="Delete the sidecar to regenerate"):
        load_stage2_profile_v3(jsonl, expected_top_k=99)


def test_cross_validation_model_hash_mismatch(tmp_path):
    """model_hash mismatch fails loud."""
    jsonl = _jsonl(tmp_path)
    save_stage2_profile_v3(_build_payload("float16"), jsonl)
    with pytest.raises(ValueError, match="Delete the sidecar to regenerate"):
        load_stage2_profile_v3(jsonl, expected_model_hash="otherhash")
