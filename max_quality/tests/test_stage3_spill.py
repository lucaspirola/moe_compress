"""Tests for InputCovarianceAccumulator spill/resume and Stage 3 partial-resume.

Two categories:
  1. Unit — pure tensor, no model. Cover spill_layer_to_disk / load_layer_from_disk
     correctness, atomicity guard, corrupt-file error, lock semantics.
  2. Integration — Stage 3 _collect_pruned_input_covariance with the _TinyModel
     fixture. Verifies the partial-resume path (pre-seeded layer spill is skipped).
"""
from __future__ import annotations

import sys
import threading
from dataclasses import field
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_compress.utils.activation_hooks import InputCovarianceAccumulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_acc_with_layer(layer_idx: int, num_experts: int = 4, hidden: int = 8,
                          dtype=torch.bfloat16) -> InputCovarianceAccumulator:
    """Return an accumulator with synthetic covariances for one layer pre-loaded."""
    acc = InputCovarianceAccumulator()
    acc.set_storage_dtype(dtype)
    torch.manual_seed(layer_idx)
    for e in range(num_experts):
        for mat in ("gate_proj", "down_proj"):
            cov = torch.randn(hidden, hidden, dtype=torch.float32)
            cov = (cov @ cov.T).to(dtype)          # positive semi-definite
            key = (layer_idx, e, mat)
            acc.covariance[key] = cov
            acc.token_count[key] = (e + 1) * 16
    return acc


# ---------------------------------------------------------------------------
# 1. Unit: spill round-trip
# ---------------------------------------------------------------------------


def test_spill_round_trip_data_integrity(tmp_path):
    """spill → load recovers exact tensors and token counts."""
    acc = _make_acc_with_layer(layer_idx=3)
    keys_before = set(k for k in acc.covariance if k[0] == 3)
    values_before = {k: acc.covariance[k].clone() for k in keys_before}
    counts_before = {k: acc.token_count[k] for k in keys_before}

    acc.spill_layer_to_disk(3, tmp_path)

    # In-memory dict must be empty for this layer after spill.
    assert not any(k[0] == 3 for k in acc.covariance), \
        "spill_layer_to_disk should drop layer 3 from in-memory dict"

    # Load it back.
    loaded = acc.load_layer_from_disk(3, tmp_path)
    assert loaded, "load_layer_from_disk returned False — file missing"

    for k in keys_before:
        assert k in acc.covariance, f"key {k} not restored"
        assert torch.equal(acc.covariance[k], values_before[k]), \
            f"tensor mismatch for {k}"
        assert acc.token_count[k] == counts_before[k], \
            f"token_count mismatch for {k}"


def test_spill_removes_keys_from_memory(tmp_path):
    """After spill, in-memory dict for that layer is empty."""
    acc = _make_acc_with_layer(layer_idx=0)
    acc_other = _make_acc_with_layer(layer_idx=1)
    acc.covariance.update(acc_other.covariance)
    acc.token_count.update(acc_other.token_count)

    acc.spill_layer_to_disk(0, tmp_path)

    assert not any(k[0] == 0 for k in acc.covariance), "layer 0 not removed"
    assert any(k[0] == 1 for k in acc.covariance), "layer 1 must still be present"


def test_spill_format_version_in_file(tmp_path):
    """Persisted file must include format_version=1."""
    acc = _make_acc_with_layer(layer_idx=2)
    acc.spill_layer_to_disk(2, tmp_path)
    payload = torch.load(tmp_path / "layer_2.pt", map_location="cpu")
    assert payload.get("format_version") == 1


def test_spill_no_op_when_layer_not_in_memory(tmp_path):
    """spill_layer_to_disk on absent layer creates no file."""
    acc = _make_acc_with_layer(layer_idx=5)
    acc.spill_layer_to_disk(99, tmp_path)  # layer 99 not in acc
    assert not (tmp_path / "layer_99.pt").exists()


def test_spill_atomic_no_tmp_on_success(tmp_path):
    """No .tmp file should remain after a successful spill."""
    acc = _make_acc_with_layer(layer_idx=7)
    acc.spill_layer_to_disk(7, tmp_path)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert not tmp_files, f"Stale .tmp files after spill: {tmp_files}"
    assert (tmp_path / "layer_7.pt").exists()


def test_load_missing_file_returns_false(tmp_path):
    """load_layer_from_disk returns False when file doesn't exist."""
    acc = InputCovarianceAccumulator()
    result = acc.load_layer_from_disk(42, tmp_path)
    assert result is False


def test_load_corrupt_file_raises_with_path(tmp_path):
    """load_layer_from_disk raises RuntimeError with the file path on corrupt data."""
    corrupt = tmp_path / "layer_6.pt"
    corrupt.write_bytes(b"not a valid torch pickle")
    acc = InputCovarianceAccumulator()
    with pytest.raises(RuntimeError, match=str(corrupt)):
        acc.load_layer_from_disk(6, tmp_path)


def test_load_wrong_format_version_raises(tmp_path):
    """Spill files with format_version != 1 must raise RuntimeError."""
    bad_payload = {
        "format_version": 99,
        "covariance": {},
        "tokens": {},
    }
    torch.save(bad_payload, tmp_path / "layer_4.pt")
    acc = InputCovarianceAccumulator()
    with pytest.raises(RuntimeError, match="format_version=99"):
        acc.load_layer_from_disk(4, tmp_path)


def test_spill_thread_safety(tmp_path):
    """Concurrent spill + load from two threads must not corrupt in-memory dict."""
    acc = InputCovarianceAccumulator()
    # Populate 8 layers simultaneously.
    for li in range(8):
        for e in range(2):
            key = (li, e, "gate_proj")
            acc.covariance[key] = torch.eye(4, dtype=torch.float32)
            acc.token_count[key] = 4

    errors: list[Exception] = []

    def spill_all():
        try:
            for li in range(8):
                acc.spill_layer_to_disk(li, tmp_path)
        except Exception as exc:
            errors.append(exc)

    def load_all():
        try:
            for li in range(8):
                acc.load_layer_from_disk(li, tmp_path)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=spill_all)
    t2 = threading.Thread(target=load_all)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert not errors, f"Thread-safety test raised: {errors}"


# ---------------------------------------------------------------------------
# 2. Integration: Stage 3 partial-resume via _collect_pruned_input_covariance
# ---------------------------------------------------------------------------


def test_stage3_partial_resume_skips_preseeded_layers(tiny_model, tmp_path):
    """Pre-seeding layer 0's spill file makes _collect_pruned_input_covariance skip it.

    Regression guard for the crash-resume path. If spill_dir contains
    layer_0.pt, the layer must not be re-instrumented (the covariance would
    be double-counted, corrupting the AA-SVD B matrix).
    """
    import json
    from moe_compress.stage3_svd import _collect_pruned_input_covariance
    from moe_compress.utils.model_io import iter_moe_layers

    moe_layers = list(iter_moe_layers(tiny_model))
    assert moe_layers, "No MoE layers found in tiny_model — fixture broken"

    # Pre-seed layer 0 with a dummy spill so the loop should skip it.
    layer0_idx = moe_layers[0].layer_idx
    num_experts = tiny_model.config.num_experts
    hidden = tiny_model.config.hidden_size
    spill_dir = tmp_path / "bcov"
    spill_dir.mkdir()

    dummy_acc = _make_acc_with_layer(layer0_idx, num_experts=num_experts,
                                     hidden=hidden, dtype=torch.float32)
    dummy_cov_before = {k: v.clone() for k, v in dummy_acc.covariance.items()}
    dummy_acc.spill_layer_to_disk(layer0_idx, spill_dir)

    # Run the collection phase.
    batches = [torch.randint(0, 32, (2, 8)) for _ in range(2)]
    acc = InputCovarianceAccumulator()
    acc.set_storage_dtype(torch.float32)

    _collect_pruned_input_covariance(
        tiny_model, moe_layers, batches, acc,
        device=None, spill_dir=spill_dir,
    )

    # Layer 0's pre-seeded file must be unchanged (not overwritten).
    payload = torch.load(spill_dir / f"layer_{layer0_idx}.pt", map_location="cpu")
    for k, v in dummy_cov_before.items():
        assert k in payload["covariance"], f"key {k} missing from pre-seeded spill"
        assert torch.allclose(payload["covariance"][k], v, atol=1e-4), \
            f"pre-seeded covariance for {k} was overwritten — partial-resume broken"

    # All non-skipped layers must have been spilled (files exist for layer > 0).
    if len(moe_layers) > 1:
        for ref in moe_layers[1:]:
            assert (spill_dir / f"layer_{ref.layer_idx}.pt").exists(), \
                f"layer_{ref.layer_idx}.pt missing — spill loop didn't run"
