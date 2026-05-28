"""Stage 2 profile-pass sidecar (schema v3) — byte-identity round-trip + IO.

Covers plan §8.4 (round-trip byte identity). The cov_storage_dtype
cross-validation is tested separately in
``test_stage2_profile_sidecar_cov_storage_dtype.py``.
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


def _make_payload(
    *,
    n_layers: int = 2,
    n_experts: int = 3,
    hidden: int = 6,
    d_int: int = 4,
    cov_storage_dtype: str = "float16",
) -> Stage2ProfilePayloadV3:
    """Construct a deterministic Stage 2 profile payload (schema v3)."""
    gate_logit_profiles: dict[int, list[tuple[int, torch.Tensor]]] = {}
    neuron_act_sum: dict = {}
    neuron_act_count: dict = {}
    cov_acc: dict = {}
    cov_token_count: dict = {}
    layer_input_reservoir: list = []
    T_b = 4
    n_batches = 2
    cov_dtype = getattr(torch, cov_storage_dtype)
    for lr in range(n_layers):
        gate_logit_profiles[lr] = [
            (T_b * b,
             torch.full((T_b, n_experts), 0.1 * (lr + 1) + 0.01 * b, dtype=torch.float32))
            for b in range(n_batches)
        ]
        for e in range(n_experts):
            neuron_act_sum[(lr, e)] = torch.full((d_int,), 0.3 * (e + 1), dtype=torch.float32)
            neuron_act_count[(lr, e)] = 17 + e
            for m in ("gate_proj", "down_proj"):
                cov_acc[(lr, e, m)] = torch.eye(hidden, dtype=cov_dtype) * (e + 1)
                cov_token_count[(lr, e, m)] = 11 + e
        layer_input_reservoir.append(
            torch.arange(8 * hidden, dtype=torch.float32).reshape(8, hidden).to(torch.bfloat16)
        )

    return Stage2ProfilePayloadV3(
        format_version=3,
        schema_version=SCHEMA_VERSIONS["stage2_profile"],
        model_hash="abc123",
        n_layers=n_layers,
        n_experts=n_experts,
        top_k=2,
        cov_storage_dtype=cov_storage_dtype,
        total_tokens_per_layer=torch.full(
            (n_layers,), T_b * n_batches, dtype=torch.int64,
        ),
        gate_logit_profiles=gate_logit_profiles,
        sim_tensor=torch.arange(
            n_layers * n_experts * n_experts, dtype=torch.float64,
        ).reshape(n_layers, n_experts, n_experts),
        neuron_act_sum=neuron_act_sum,
        neuron_act_count=neuron_act_count,
        cov_acc=cov_acc,
        cov_token_count=cov_token_count,
        layer_input_reservoir=layer_input_reservoir,
    )


def test_roundtrip_byte_identity(tmp_path):
    """All payload fields round-trip byte-identically (plan §8.4)."""
    jsonl = _jsonl(tmp_path)
    original = _make_payload()
    save_stage2_profile_v3(original, jsonl)

    expected_path = sidecar_path(jsonl, "stage2_profile")
    assert expected_path.exists()

    loaded = load_stage2_profile_v3(jsonl)
    assert loaded is not None
    assert isinstance(loaded, Stage2ProfilePayloadV3)
    # Schema + identity fields.
    assert loaded.schema_version == 3
    assert loaded.format_version == 3
    assert loaded.model_hash == original.model_hash
    assert loaded.n_layers == original.n_layers
    assert loaded.n_experts == original.n_experts
    assert loaded.top_k == original.top_k
    assert loaded.cov_storage_dtype == original.cov_storage_dtype
    # Tensor fields — byte-identical.
    assert torch.equal(loaded.total_tokens_per_layer, original.total_tokens_per_layer.cpu())
    assert torch.equal(loaded.sim_tensor, original.sim_tensor.cpu())
    assert loaded.sim_tensor.dtype == torch.float64
    assert loaded.total_tokens_per_layer.dtype == torch.int64


def test_roundtrip_gate_logit_profiles_shape_preserved(tmp_path):
    """gate_logit_profiles preserves the (int, tensor) tuple structure verbatim."""
    jsonl = _jsonl(tmp_path)
    original = _make_payload()
    save_stage2_profile_v3(original, jsonl)
    loaded = load_stage2_profile_v3(jsonl)

    assert set(loaded.gate_logit_profiles) == set(original.gate_logit_profiles)
    for lr, original_batches in original.gate_logit_profiles.items():
        loaded_batches = loaded.gate_logit_profiles[lr]
        assert isinstance(loaded_batches, list)
        assert len(loaded_batches) == len(original_batches)
        for (lof, lt), (oof, ot) in zip(loaded_batches, original_batches):
            # Tuple shape: (int_offset, Tensor[T_b, E] fp32) preserved.
            assert isinstance(lof, int)
            assert isinstance(lt, torch.Tensor)
            assert int(lof) == int(oof)
            assert lt.dtype == torch.float32
            assert torch.equal(lt, ot.cpu())


def test_roundtrip_cov_and_neuron_dicts(tmp_path):
    """cov_acc / cov_token_count / neuron_act_* dicts round-trip key+value."""
    jsonl = _jsonl(tmp_path)
    original = _make_payload()
    save_stage2_profile_v3(original, jsonl)
    loaded = load_stage2_profile_v3(jsonl)

    assert set(loaded.cov_acc) == set(original.cov_acc)
    for key, ot in original.cov_acc.items():
        assert torch.equal(loaded.cov_acc[key], ot.cpu())
        # storage_dtype preserved (declared float16 → loaded float16).
        assert loaded.cov_acc[key].dtype == torch.float16
    assert dict(loaded.cov_token_count) == dict(original.cov_token_count)
    assert set(loaded.neuron_act_sum) == set(original.neuron_act_sum)
    for key, ot in original.neuron_act_sum.items():
        assert torch.equal(loaded.neuron_act_sum[key], ot.cpu())
    assert dict(loaded.neuron_act_count) == dict(original.neuron_act_count)


def test_roundtrip_layer_input_reservoir(tmp_path):
    """layer_input_reservoir is a list[Tensor[N, hidden] bf16], len==n_layers."""
    jsonl = _jsonl(tmp_path)
    original = _make_payload()
    save_stage2_profile_v3(original, jsonl)
    loaded = load_stage2_profile_v3(jsonl)

    assert isinstance(loaded.layer_input_reservoir, list)
    assert len(loaded.layer_input_reservoir) == original.n_layers
    for ot, lt in zip(original.layer_input_reservoir, loaded.layer_input_reservoir):
        assert lt.dtype == torch.bfloat16
        # bf16-cast on both sides means byte identity (both already bf16).
        assert torch.equal(lt, ot.cpu().to(torch.bfloat16))


def test_load_miss_returns_none(tmp_path):
    """Cache miss (sidecar absent) returns None gracefully."""
    jsonl = _jsonl(tmp_path)
    assert load_stage2_profile_v3(jsonl) is None


def test_schema_mismatch_raises(tmp_path, monkeypatch):
    """A sidecar with the wrong schema_version raises with the standard message."""
    jsonl = _jsonl(tmp_path)
    original = _make_payload()
    save_stage2_profile_v3(original, jsonl)
    # Bump the central version so the existing sidecar disagrees.
    monkeypatch.setitem(SCHEMA_VERSIONS, "stage2_profile", 99)
    with pytest.raises(RuntimeError, match="manifest validation FAILED"):
        load_stage2_profile_v3(jsonl)


def test_save_rejects_unknown_cov_dtype(tmp_path):
    """Writer raises on an unsupported cov_storage_dtype string."""
    jsonl = _jsonl(tmp_path)
    payload = _make_payload()
    # Hand-mutate to a bad value.
    payload.cov_storage_dtype = "bogus"
    with pytest.raises(ValueError):
        save_stage2_profile_v3(payload, jsonl)
