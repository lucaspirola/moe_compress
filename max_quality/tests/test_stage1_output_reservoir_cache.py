"""Tests for ``Stage1OutputReservoirCacheProvider`` -- the Stage 1
cache-side provider for the per-(layer, expert) expert-output reservoir
sidecar (Item 6 of the calibration-v2 writers campaign).

Five tests:

1. ``test_cache_provider_miss`` -- no sidecar on disk → ``on_load``
   returns ``None`` and leaves the ctx untouched (the live
   ``ExpertOutputAccumulator`` path runs unchanged).
2. ``test_cache_provider_hit`` -- writing a sidecar with known
   reservoir contents and calling ``on_load`` with a stub ctx
   containing ``moe_layers`` populates ``ctx.output_acc`` with a
   pre-finalized ``ExpertOutputAccumulator`` whose ``_finalized`` dict
   is keyed by ``(layer_idx, expert_id)`` and whose tensor values
   match the cached slab sliced to ``valid_count`` per cell.
3. ``test_topology_mismatch_raises`` -- a sidecar reporting an
   ``n_layers`` count that disagrees with the live ``moe_layers`` list
   raises ``ValueError`` with the actionable
   "Delete it to regenerate" message.
4. ``test_zero_valid_count_excluded`` -- cells with ``valid_count == 0``
   (no tokens ever routed) are NOT inserted into the accumulator's
   ``_finalized`` dict; ``get_representations`` returns ``None`` for
   those cells, matching the live absent-key convention.
5. ``test_get_representations_on_absent`` -- after a cache hit,
   querying an absent ``(layer_idx, expert_id)`` (rank that exists in
   the sidecar but with ``valid_count == 0``, OR a layer_idx not in
   ``moe_layers``) returns ``None`` from
   ``ExpertOutputAccumulator.get_representations``.

CPU-only by construction (every tensor allocation defaults to CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.output_reservoir_cache import (
    Stage1OutputReservoirCacheProvider,
)
from moe_compress.utils.activation_hooks import ExpertOutputAccumulator
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    OutputReservoirPayload,
    save_output_reservoir,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubMoELayerRef:
    """Tiny stand-in for MoELayerRef -- the cache provider only reads
    ``.layer_idx``, so the test fixture supplies just that."""
    layer_idx: int


def _jsonl(tmp_path: Path) -> Path:
    """Return a non-existent JSONL path inside ``tmp_path`` -- the sidecar
    lives next to it under ``tmp_path/sidecars/``."""
    return tmp_path / "trace.jsonl"


def _make_payload(
    n_layers: int, n_experts: int, max_tokens: int = 4, hidden: int = 3,
    *, reservoir: torch.Tensor | None = None,
    valid_count: torch.Tensor | None = None,
    total_seen: torch.Tensor | None = None,
) -> OutputReservoirPayload:
    if reservoir is None:
        reservoir = torch.arange(
            n_layers * n_experts * max_tokens * hidden, dtype=torch.float32,
        ).reshape(n_layers, n_experts, max_tokens, hidden)
    if valid_count is None:
        valid_count = torch.full(
            (n_layers, n_experts), max_tokens, dtype=torch.int64,
        )
    if total_seen is None:
        total_seen = valid_count.clone()
    return OutputReservoirPayload(
        schema_version=SCHEMA_VERSIONS["output_reservoir"],
        n_experts=n_experts,
        n_layers=n_layers,
        reservoir=reservoir,
        valid_count=valid_count,
        total_seen=total_seen,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Test 1 -- cache provider miss
# ---------------------------------------------------------------------------


def test_cache_provider_miss(tmp_path):
    """No sidecar on disk → ``on_load`` returns None and leaves ctx
    untouched."""
    jsonl = _jsonl(tmp_path)
    # No sidecar written.

    moe_layers = [_StubMoELayerRef(layer_idx=0)]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)

    provider = Stage1OutputReservoirCacheProvider()
    returned = provider.on_load(ctx, jsonl)

    assert returned is None
    # ctx is untouched -- output_acc was not added.
    assert not ctx.has("output_acc")


# ---------------------------------------------------------------------------
# Test 2 -- cache provider hit
# ---------------------------------------------------------------------------


def test_cache_provider_hit(tmp_path):
    """Write a sidecar with known reservoir contents; call ``on_load`` with
    a stub ctx; verify the accumulator's ``_finalized`` dict is keyed by
    ``(layer_idx, expert_id)`` and the tensor values match the cached
    slab sliced to ``valid_count``."""
    jsonl = _jsonl(tmp_path)
    # Two layers with non-trivial layer_idx values so we exercise the
    # rank -> layer_idx mapping. n_experts = 2. max_tokens = 4. hidden = 3.
    moe_layers = [
        _StubMoELayerRef(layer_idx=7),
        _StubMoELayerRef(layer_idx=11),
    ]
    n_layers = 2
    n_experts = 2
    max_tokens = 4
    hidden = 3
    reservoir = torch.arange(
        n_layers * n_experts * max_tokens * hidden, dtype=torch.float32,
    ).reshape(n_layers, n_experts, max_tokens, hidden)
    # rank 0 expert 0 saw 3 tokens (slot 3 is zero-padded);
    # rank 0 expert 1 saw 4 tokens (full reservoir);
    # rank 1 expert 0 saw 2 tokens;
    # rank 1 expert 1 saw 1 token.
    valid_count = torch.tensor(
        [[3, 4], [2, 1]], dtype=torch.int64,
    )
    payload = _make_payload(
        n_layers=n_layers, n_experts=n_experts,
        max_tokens=max_tokens, hidden=hidden,
        reservoir=reservoir, valid_count=valid_count,
    )
    save_output_reservoir(payload, jsonl)

    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1OutputReservoirCacheProvider()
    returned = provider.on_load(ctx, jsonl)

    assert returned is not None
    assert returned.n_layers == n_layers
    assert returned.n_experts == n_experts
    assert returned.max_tokens == max_tokens
    # Provider populated ctx.output_acc.
    assert ctx.has("output_acc")
    acc = ctx.get("output_acc")
    assert isinstance(acc, ExpertOutputAccumulator)
    assert acc.max_tokens_per_expert == max_tokens

    # All four (layer_idx, expert_id) cells should be present (none have
    # valid_count == 0). The stored slabs match the cached reservoir
    # rows sliced to valid_count and cast to fp32. Compare against the
    # bf16-round-trip-cast value because save_output_reservoir casts
    # the tensor to bfloat16 on disk; the cache reader reads bf16 and
    # casts back to fp32.
    expected_keys = {(7, 0), (7, 1), (11, 0), (11, 1)}
    assert set(acc._finalized.keys()) == expected_keys

    # (layer_idx=7, expert=0) → rank 0, e=0, valid=3.
    expected_70 = reservoir[0, 0, :3].to(torch.bfloat16).to(torch.float32)
    assert torch.equal(acc._finalized[(7, 0)], expected_70)
    # (layer_idx=7, expert=1) → rank 0, e=1, valid=4.
    expected_71 = reservoir[0, 1, :4].to(torch.bfloat16).to(torch.float32)
    assert torch.equal(acc._finalized[(7, 1)], expected_71)
    # (layer_idx=11, expert=0) → rank 1, e=0, valid=2.
    expected_110 = reservoir[1, 0, :2].to(torch.bfloat16).to(torch.float32)
    assert torch.equal(acc._finalized[(11, 0)], expected_110)
    # (layer_idx=11, expert=1) → rank 1, e=1, valid=1.
    expected_111 = reservoir[1, 1, :1].to(torch.bfloat16).to(torch.float32)
    assert torch.equal(acc._finalized[(11, 1)], expected_111)

    # All stored tensors live on CPU as fp32.
    for t in acc._finalized.values():
        assert t.device.type == "cpu"
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 3 -- topology mismatch raises
# ---------------------------------------------------------------------------


def test_topology_mismatch_raises(tmp_path):
    """A sidecar reporting an ``n_layers`` count that disagrees with the
    live ``moe_layers`` list raises ``ValueError`` with the actionable
    "Delete it to regenerate" message."""
    jsonl = _jsonl(tmp_path)
    # Sidecar reports 3 layers; live model only has 2.
    payload = _make_payload(n_layers=3, n_experts=2)
    save_output_reservoir(payload, jsonl)

    ctx = PipelineContext()
    ctx.set("moe_layers", [
        _StubMoELayerRef(layer_idx=0),
        _StubMoELayerRef(layer_idx=1),
    ])
    provider = Stage1OutputReservoirCacheProvider()
    with pytest.raises(ValueError, match="output_reservoir cache mismatch"):
        provider.on_load(ctx, jsonl)
    # ctx is left untouched (no partial population on raise).
    assert not ctx.has("output_acc")


# ---------------------------------------------------------------------------
# Test 4 -- zero-valid-count cells excluded from the dict
# ---------------------------------------------------------------------------


def test_zero_valid_count_excluded(tmp_path):
    """Cells with ``valid_count == 0`` are NOT inserted into the
    accumulator's ``_finalized`` dict (matches the live absent-key
    convention for zero-traffic experts)."""
    jsonl = _jsonl(tmp_path)
    moe_layers = [
        _StubMoELayerRef(layer_idx=3),
        _StubMoELayerRef(layer_idx=5),
    ]
    n_layers = 2
    n_experts = 3
    max_tokens = 2
    hidden = 2
    # rank 0: e0 saw 2 tokens, e1 zero traffic, e2 saw 1 token.
    # rank 1: entire layer saw no traffic (all zeros).
    reservoir = torch.arange(
        n_layers * n_experts * max_tokens * hidden, dtype=torch.float32,
    ).reshape(n_layers, n_experts, max_tokens, hidden)
    valid_count = torch.tensor(
        [[2, 0, 1],
         [0, 0, 0]], dtype=torch.int64,
    )
    payload = _make_payload(
        n_layers=n_layers, n_experts=n_experts,
        max_tokens=max_tokens, hidden=hidden,
        reservoir=reservoir, valid_count=valid_count,
    )
    save_output_reservoir(payload, jsonl)

    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1OutputReservoirCacheProvider()
    returned = provider.on_load(ctx, jsonl)

    assert returned is not None
    acc = ctx.get("output_acc")
    # Only the two non-zero-traffic cells appear.
    assert set(acc._finalized.keys()) == {(3, 0), (3, 2)}
    # Rank-1 layer (layer_idx=5) is entirely absent from the dict.
    assert not any(k[0] == 5 for k in acc._finalized)
    # get_representations on excluded cells returns None.
    assert acc.get_representations(3, 1) is None
    assert acc.get_representations(5, 0) is None
    assert acc.get_representations(5, 1) is None
    assert acc.get_representations(5, 2) is None


# ---------------------------------------------------------------------------
# Test 5 -- get_representations on absent (li, e)
# ---------------------------------------------------------------------------


def test_get_representations_on_absent(tmp_path):
    """After a cache hit, ``get_representations`` on an absent
    ``(layer_idx, expert_id)`` returns ``None``.

    Two flavors of absence:
    * (layer_idx that exists, expert with valid_count == 0) → excluded
      from ``_finalized``; get_representations returns None.
    * (layer_idx not in moe_layers) → never inserted (the cache reader
      only iterates ranks present in moe_layers); get_representations
      returns None.
    """
    jsonl = _jsonl(tmp_path)
    moe_layers = [_StubMoELayerRef(layer_idx=2)]
    n_layers = 1
    n_experts = 3
    max_tokens = 2
    hidden = 2
    reservoir = torch.arange(
        n_layers * n_experts * max_tokens * hidden, dtype=torch.float32,
    ).reshape(n_layers, n_experts, max_tokens, hidden)
    # Only expert 1 has any traffic.
    valid_count = torch.tensor(
        [[0, 1, 0]], dtype=torch.int64,
    )
    payload = _make_payload(
        n_layers=n_layers, n_experts=n_experts,
        max_tokens=max_tokens, hidden=hidden,
        reservoir=reservoir, valid_count=valid_count,
    )
    save_output_reservoir(payload, jsonl)

    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1OutputReservoirCacheProvider()
    returned = provider.on_load(ctx, jsonl)
    assert returned is not None
    acc = ctx.get("output_acc")

    # Expert 1 IS present; the other two zero-valid-count cells are
    # absent (returns None on get_representations).
    assert acc.get_representations(2, 0) is None
    assert acc.get_representations(2, 1) is not None
    assert acc.get_representations(2, 2) is None
    # A layer_idx not in moe_layers was never written by the cache
    # reader -- get_representations returns None.
    assert acc.get_representations(99, 0) is None
    assert acc.get_representations(99, 1) is None
