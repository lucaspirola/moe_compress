"""Round-trip + discovery tests for ``stage2.shared_io`` and ``stage2.resume``.

Locks the Stage 2 partial-JSON schema (format_version=2) against accidental
drift, and validates that ``discover_completed_layers`` correctly cleans up
orphan ``.pt`` files and returns one ``ResumedLayerRecord`` per completed
layer. Pure CPU + stdlib; no GPU required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from moe_compress.stage2.resume import (
    ResumedLayerRecord,
    discover_completed_layers,
)
from moe_compress.stage2.shared_io import _write_merge_json


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeLayerRef:
    """Stand-in for MoELayerRef carrying only the fields ``discover_completed_layers`` reads."""

    layer_idx: int
    num_routed_experts: int


# ---------------------------------------------------------------------------
# _write_merge_json round-trip
# ---------------------------------------------------------------------------


def _representative_payload_kwargs():
    return dict(
        layer_idx=7,
        final_kept_ids=[0, 2, 5, 7],
        grouped={0: [0, 1], 2: [2, 3, 4], 5: [5, 6], 7: [7]},
        freq={i: 100 + i for i in range(8)},
        merge_map_layer={0: [0, 1], 1: [2, 3, 4], 2: [5, 6], 3: [7]},
        mean_cost_per_pair=0.1234,
        assignment_solver_used="hungarian",
        cost_alignment_used="post",
        em_rounds_completed=2,
        distill_state={"group_0": {"steps": 50, "final_loss": 0.01}},
        heal_state={"steps": 100, "accepted": True, "train_mse": 0.02},
    )


def test_write_merge_json_round_trips_every_field(tmp_path: Path):
    """Writing then reading back via json.loads must reproduce every field with the same types."""
    kw = _representative_payload_kwargs()
    _write_merge_json(
        tmp_path,
        kw["layer_idx"],
        kw["final_kept_ids"],
        kw["grouped"],
        kw["freq"],
        kw["merge_map_layer"],
        mean_cost_per_pair=kw["mean_cost_per_pair"],
        assignment_solver_used=kw["assignment_solver_used"],
        cost_alignment_used=kw["cost_alignment_used"],
        em_rounds_completed=kw["em_rounds_completed"],
        distill_state=kw["distill_state"],
        heal_state=kw["heal_state"],
    )

    final = tmp_path / f"merge_{kw['layer_idx']}.json"
    assert final.exists()
    assert not (tmp_path / f"merge_{kw['layer_idx']}.json.tmp").exists()
    data = json.loads(final.read_text())

    assert data["format_version"] == 2
    assert data["final_kept_ids"] == kw["final_kept_ids"]
    # grouped/freq/merge_map_layer keys are stringified for JSON; values are lists/ints.
    assert {int(k): list(v) for k, v in data["grouped"].items()} == kw["grouped"]
    assert {int(k): int(v) for k, v in data["freq"].items()} == kw["freq"]
    assert {int(k): list(v) for k, v in data["merge_map_layer"].items()} == kw["merge_map_layer"]
    assert data["mean_cost_per_pair"] == pytest.approx(kw["mean_cost_per_pair"])
    assert data["assignment_solver_used"] == kw["assignment_solver_used"]
    assert data["cost_alignment_used"] == kw["cost_alignment_used"]
    assert data["em_rounds_completed"] == kw["em_rounds_completed"]
    assert data["distill_state"] == kw["distill_state"]
    assert data["heal_state"] == kw["heal_state"]


def test_write_merge_json_accepts_none_optional_fields(tmp_path: Path):
    """All optional fields default safely (mean_cost_per_pair / distill_state / heal_state can be None)."""
    _write_merge_json(
        tmp_path,
        layer_idx=0,
        final_kept_ids=[0, 1],
        grouped={0: [0], 1: [1]},
        freq={0: 1, 1: 1},
        merge_map_layer={0: [0], 1: [1]},
    )
    data = json.loads((tmp_path / "merge_0.json").read_text())
    assert data["format_version"] == 2
    assert data["mean_cost_per_pair"] is None
    assert data["distill_state"] is None
    assert data["heal_state"] is None
    assert data["assignment_solver_used"] == "greedy"
    assert data["cost_alignment_used"] == "pre"
    assert data["em_rounds_completed"] == 0


# ---------------------------------------------------------------------------
# discover_completed_layers
# ---------------------------------------------------------------------------


def _write_synthetic_cov_pt(path: Path) -> None:
    """A minimal but format-version-1 conformant cov payload — content irrelevant to discover()."""
    torch.save({"format_version": 1, "covariance": {}, "tokens": {}}, path)


def test_discover_returns_two_valid_records_and_deletes_orphan(tmp_path: Path):
    """Two complete (json+pt) layers + one orphan .pt → orphan deleted, two records returned."""
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()

    moe_layers = [
        _FakeLayerRef(layer_idx=0, num_routed_experts=4),
        _FakeLayerRef(layer_idx=1, num_routed_experts=4),
        _FakeLayerRef(layer_idx=2, num_routed_experts=4),
    ]

    # Layer 0 + 1: complete pair (cov .pt + merge .json).
    for li in (0, 1):
        _write_synthetic_cov_pt(partial_dir / f"layer_{li}.pt")
        _write_merge_json(
            partial_dir,
            layer_idx=li,
            final_kept_ids=[0, 2],
            grouped={0: [0, 1], 2: [2, 3]},
            freq={0: 10, 1: 11, 2: 12, 3: 13},
            merge_map_layer={0: [0, 1], 1: [2, 3]},
        )

    # Layer 2: orphan .pt with no matching .json — must be deleted.
    orphan = partial_dir / "layer_2.pt"
    _write_synthetic_cov_pt(orphan)
    assert orphan.exists()

    # Stale .tmp file — must be cleaned up.
    stale_tmp = partial_dir / "layer_99.pt.tmp"
    stale_tmp.write_bytes(b"")
    assert stale_tmp.exists()

    records = discover_completed_layers(
        partial_dir, moe_layers, heal_enabled=False,
    )

    assert not orphan.exists(), "orphan .pt was not cleaned up"
    assert not stale_tmp.exists(), "stale .tmp file was not cleaned up"
    assert [r.layer_ref.layer_idx for r in records] == [0, 1]
    for r in records:
        assert r.final_kept_ids == [0, 2]
        assert r.grouped == {0: [0, 1], 2: [2, 3]}
        assert r.freq == {0: 10, 1: 11, 2: 12, 3: 13}
        assert r.merge_map_layer == {0: [0, 1], 1: [2, 3]}
        assert r.has_heal_weights_file is False
        assert r.resume_ream_acc is None  # no _neuron_means_layer*.pt present


def test_discover_rejects_wrong_format_version(tmp_path: Path):
    """A merge_*.json with format_version != 2 raises a clear RuntimeError."""
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()
    _write_synthetic_cov_pt(partial_dir / "layer_0.pt")
    (partial_dir / "merge_0.json").write_text(json.dumps({
        "format_version": 1,
        "final_kept_ids": [0],
        "grouped": {"0": [0]},
        "freq": {"0": 1},
        "merge_map_layer": {"0": [0]},
    }))
    with pytest.raises(RuntimeError, match="format_version=1"):
        discover_completed_layers(
            partial_dir,
            [_FakeLayerRef(layer_idx=0, num_routed_experts=1)],
            heal_enabled=False,
        )


def test_discover_skips_layers_missing_either_artefact(tmp_path: Path):
    """Layer with only merge.json (no .pt) → skipped (not a record, not an error)."""
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()
    _write_merge_json(
        partial_dir,
        layer_idx=0,
        final_kept_ids=[0],
        grouped={0: [0]},
        freq={0: 1},
        merge_map_layer={0: [0]},
    )
    # No layer_0.pt — discover() must skip the layer without raising.
    records = discover_completed_layers(
        partial_dir,
        [_FakeLayerRef(layer_idx=0, num_routed_experts=1)],
        heal_enabled=False,
    )
    assert records == []


def test_discover_flags_heal_weights_when_enabled_and_present(tmp_path: Path):
    """has_heal_weights_file True iff heal_enabled and the heal-weights .pt exists."""
    partial_dir = tmp_path / "_stage2_partial"
    partial_dir.mkdir()
    _write_synthetic_cov_pt(partial_dir / "layer_0.pt")
    _write_merge_json(
        partial_dir,
        layer_idx=0,
        final_kept_ids=[0],
        grouped={0: [0]},
        freq={0: 1},
        merge_map_layer={0: [0]},
    )
    # Heal-weights file present but heal disabled → False.
    (partial_dir / "_heal_weights_layer_0.pt").write_bytes(b"")
    refs = [_FakeLayerRef(layer_idx=0, num_routed_experts=1)]
    assert discover_completed_layers(partial_dir, refs, heal_enabled=False)[0].has_heal_weights_file is False
    # heal_enabled=True + file present → True.
    assert discover_completed_layers(partial_dir, refs, heal_enabled=True)[0].has_heal_weights_file is True
