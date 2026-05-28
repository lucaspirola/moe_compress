"""Tests for ``Stage1RoutingStatsCacheProvider`` -- the Stage 1 cache-
side provider for the per-(layer, expert) routing-frequency +
mean-routing-weight sidecar (Item 3 of the calibration-v2 writers
campaign).

Five tests:

1. ``test_load_miss`` -- no sidecar on disk -> ``on_load`` returns
   ``None`` and leaves ctx untouched.
2. ``test_load_hit`` -- writing a sidecar with known per-expert values
   and calling ``on_load`` with a stub ctx populates
   ``ctx.routing_stats_payload`` with the loaded payload (tensor
   equality preserved through the round-trip).
3. ``test_n_layers_mismatch_raises`` -- live model has a different MoE
   layer count than the sidecar -> ``on_load`` raises ``ValueError``
   with the actionable "delete it to regenerate" message.
4. ``test_schema_mismatch_raises`` -- bumping
   ``SCHEMA_VERSIONS["routing_stats"]`` after a sidecar is written
   makes ``load_routing_stats`` raise ``ValueError`` with the
   actionable "Delete the sidecar to regenerate" message; the cache
   provider propagates that.
5. ``test_no_ctx_mutation_on_miss`` -- no sidecar -> ctx is untouched
   (the ``routing_stats_payload`` slot is NOT set, NOT defaulted to
   None).

CPU-only by construction (every tensor allocation defaults to CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.routing_stats_cache import (
    Stage1RoutingStatsCacheProvider,
)
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    RoutingStatsPayload,
    load_routing_stats,
    save_routing_stats,
    sidecar_path,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubMoELayerRef:
    """Tiny stand-in for MoELayerRef -- the cache provider only reads
    ``len(moe_layers)`` and would read ``.layer_idx`` if it needed to
    map rank -> layer_idx (not done in Item 3)."""
    layer_idx: int


def _jsonl(tmp_path: Path) -> Path:
    return tmp_path / "trace.jsonl"


def _make_payload(n_layers: int, n_experts: int,
                  freq: torch.Tensor | None = None,
                  mean_weight: torch.Tensor | None = None,
                  ) -> RoutingStatsPayload:
    if freq is None:
        freq = torch.arange(
            n_layers * n_experts, dtype=torch.int64,
        ).reshape(n_layers, n_experts) + 1
    if mean_weight is None:
        mean_weight = torch.linspace(
            0.1, 0.9, n_layers * n_experts, dtype=torch.float32,
        ).reshape(n_layers, n_experts)
    return RoutingStatsPayload(
        schema_version=SCHEMA_VERSIONS["routing_stats"],
        n_experts=n_experts,
        n_layers=n_layers,
        freq=freq,
        mean_weight=mean_weight,
    )


# ---------------------------------------------------------------------------
# Test 1 -- load miss
# ---------------------------------------------------------------------------


def test_load_miss(tmp_path):
    """No sidecar on disk -> on_load returns None and leaves ctx
    untouched."""
    jsonl = _jsonl(tmp_path)
    moe_layers = [_StubMoELayerRef(layer_idx=0)]

    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1RoutingStatsCacheProvider()
    result = provider.on_load(ctx, jsonl)

    assert result is None
    assert not ctx.has("routing_stats_payload")


# ---------------------------------------------------------------------------
# Test 2 -- load hit
# ---------------------------------------------------------------------------


def test_load_hit(tmp_path):
    """Write a sidecar; call ``on_load`` with a stub ctx; verify
    ``ctx.routing_stats_payload`` is the loaded payload."""
    jsonl = _jsonl(tmp_path)
    moe_layers = [
        _StubMoELayerRef(layer_idx=7),
        _StubMoELayerRef(layer_idx=11),
    ]
    freq = torch.tensor(
        [[3, 5, 0], [1, 0, 7]], dtype=torch.int64,
    )
    mean_weight = torch.tensor(
        [[0.5, 0.6, 0.0], [0.7, 0.0, 0.8]], dtype=torch.float32,
    )
    payload = _make_payload(
        n_layers=2, n_experts=3, freq=freq, mean_weight=mean_weight,
    )
    save_routing_stats(payload, jsonl)

    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1RoutingStatsCacheProvider()
    returned = provider.on_load(ctx, jsonl)

    assert returned is not None
    assert returned.n_layers == 2
    assert returned.n_experts == 3
    assert ctx.has("routing_stats_payload")

    stashed = ctx.get("routing_stats_payload")
    assert torch.equal(stashed.freq, freq.cpu())
    assert torch.equal(stashed.mean_weight, mean_weight.cpu())


# ---------------------------------------------------------------------------
# Test 3 -- n_layers mismatch raises
# ---------------------------------------------------------------------------


def test_n_layers_mismatch_raises(tmp_path):
    """Live model has 3 MoE layers but the sidecar was written for 2 ->
    ``on_load`` raises ``ValueError`` pointing to the regenerate-the-
    sidecar action."""
    jsonl = _jsonl(tmp_path)
    payload = _make_payload(n_layers=2, n_experts=3)
    save_routing_stats(payload, jsonl)

    # Stub model has THREE MoE layers -> mismatch with the sidecar.
    moe_layers = [
        _StubMoELayerRef(layer_idx=0),
        _StubMoELayerRef(layer_idx=1),
        _StubMoELayerRef(layer_idx=2),
    ]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1RoutingStatsCacheProvider()
    with pytest.raises(ValueError, match="delete it to regenerate"):
        provider.on_load(ctx, jsonl)


# ---------------------------------------------------------------------------
# Test 4 -- schema-version mismatch raises
# ---------------------------------------------------------------------------


def test_schema_mismatch_raises(tmp_path, monkeypatch):
    """Bumping ``SCHEMA_VERSIONS["routing_stats"]`` after a sidecar is
    written makes ``load_routing_stats`` raise ``ValueError``; the cache
    provider propagates it (no silent fallback)."""
    jsonl = _jsonl(tmp_path)
    save_routing_stats(_make_payload(n_layers=2, n_experts=3), jsonl)

    bumped = dict(SCHEMA_VERSIONS)
    bumped["routing_stats"] = 99
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )

    moe_layers = [
        _StubMoELayerRef(layer_idx=0),
        _StubMoELayerRef(layer_idx=1),
    ]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    provider = Stage1RoutingStatsCacheProvider()
    with pytest.raises(RuntimeError, match="manifest validation FAILED"):
        provider.on_load(ctx, jsonl)


# ---------------------------------------------------------------------------
# Test 5 -- no ctx mutation on miss
# ---------------------------------------------------------------------------


def test_no_ctx_mutation_on_miss(tmp_path):
    """No sidecar -> ctx is untouched: routing_stats_payload key MUST
    NOT be set (not even to None)."""
    jsonl = _jsonl(tmp_path)
    moe_layers = [_StubMoELayerRef(layer_idx=0)]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    # Pre-seed an unrelated key so we can verify the ctx survives.
    ctx.set("untouched_marker", 12345)

    provider = Stage1RoutingStatsCacheProvider()
    result = provider.on_load(ctx, jsonl)

    assert result is None
    assert not ctx.has("routing_stats_payload")
    # Pre-existing keys survive.
    assert ctx.get("untouched_marker") == 12345
