"""Tests for ``Stage1RouterLogitsStatsCacheProvider`` -- the Stage 1
cache-side provider for the per-(layer, expert) sink-vs-normal
router-score aggregates sidecar (Item 4 of the calibration-v2 writers
campaign).

Five tests:

1. ``test_load_hit_pre_populates_sink_acc`` -- writing a sidecar with
   known sums and counts and calling ``on_load`` with a config that has
   ``sink_token_enabled=True`` and a stub ctx hydrates a pre-finalized
   SinkTokenRoutingAccumulator into ``ctx.sink_acc``; the
   ``mean_router_score_sink`` / ``mean_router_score_normal`` /
   ``freq_on_sink`` dicts contain the expected (layer_idx, expert) ->
   value pairs computed by dividing the cached sums by the cached
   counts (zero-count -> 0.0, no NaN).
2. ``test_load_miss`` -- no sidecar on disk -> ``on_load`` returns
   ``None`` and leaves ctx untouched.
3. ``test_n_layers_mismatch_raises`` -- live model has a different MoE
   layer count than the sidecar -> ``on_load`` raises ``ValueError``
   with the actionable "delete it to regenerate" message.
4. ``test_schema_mismatch_raises`` -- bumping
   ``SCHEMA_VERSIONS["router_logits_stats"]`` after a sidecar is
   written makes ``load_router_logits_stats`` raise ``ValueError``;
   the cache provider propagates it (no silent fallback).
5. ``test_sink_token_disabled_returns_none`` -- when the config sets
   ``sink_token_enabled=False`` (R3 guard), the provider returns
   ``None`` WITHOUT even consulting the sidecar so that
   ``SinkTokenDetectorPlugin.setup()``'s ``sink_acc=None`` decision is
   preserved.

CPU-only by construction (every tensor allocation defaults to CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage1.plugins.router_logits_stats_cache import (
    Stage1RouterLogitsStatsCacheProvider,
)
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    RouterLogitsStatsPayload,
    save_router_logits_stats,
)
from moe_compress.utils.sink_token_routing import SinkTokenRoutingAccumulator


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubMoELayerRef:
    """Tiny stand-in for MoELayerRef -- the cache provider reads
    ``.layer_idx`` to map rank -> layer_idx when populating the
    accumulator's (layer_idx, expert) dict keys."""
    layer_idx: int


def _jsonl(tmp_path: Path) -> Path:
    return tmp_path / "trace.jsonl"


def _make_payload(
    n_layers: int,
    n_experts: int,
    score_sink_sum: torch.Tensor | None = None,
    score_normal_sum: torch.Tensor | None = None,
    fire_on_sink: torch.Tensor | None = None,
    n_sink_tokens: torch.Tensor | None = None,
    n_normal_tokens: torch.Tensor | None = None,
    bos_token_id: int | None = 151643,
) -> RouterLogitsStatsPayload:
    if score_sink_sum is None:
        score_sink_sum = torch.arange(
            n_layers * n_experts, dtype=torch.float32
        ).reshape(n_layers, n_experts)
    if score_normal_sum is None:
        score_normal_sum = torch.linspace(
            0.0, 1.0, n_layers * n_experts, dtype=torch.float32
        ).reshape(n_layers, n_experts)
    if fire_on_sink is None:
        fire_on_sink = torch.arange(
            n_layers * n_experts, dtype=torch.int64
        ).reshape(n_layers, n_experts)
    if n_sink_tokens is None:
        n_sink_tokens = torch.tensor(
            [4 * (i + 1) for i in range(n_layers)], dtype=torch.int64,
        )
    if n_normal_tokens is None:
        n_normal_tokens = torch.tensor(
            [16 * (i + 1) for i in range(n_layers)], dtype=torch.int64,
        )
    return RouterLogitsStatsPayload(
        schema_version=SCHEMA_VERSIONS["router_logits_stats"],
        n_experts=n_experts,
        n_layers=n_layers,
        score_sink_sum=score_sink_sum,
        score_normal_sum=score_normal_sum,
        fire_on_sink=fire_on_sink,
        n_sink_tokens=n_sink_tokens,
        n_normal_tokens=n_normal_tokens,
        bos_token_id=bos_token_id,
    )


def _enabled_config() -> dict:
    return {
        "stage1_grape": {
            "super_expert_detection": {
                "sink_token_enabled": True,
            },
        },
    }


def _disabled_config() -> dict:
    return {
        "stage1_grape": {
            "super_expert_detection": {
                "sink_token_enabled": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Test 1 -- load hit pre-populates sink_acc
# ---------------------------------------------------------------------------


def test_load_hit_pre_populates_sink_acc(tmp_path):
    """Sidecar present + sink_token_enabled=True -> on_load hydrates a
    SinkTokenRoutingAccumulator into ctx.sink_acc with the expected
    per-(layer_idx, expert) means + frequencies, mapping rank ->
    layer_idx via the live moe_layers list."""
    jsonl = _jsonl(tmp_path)
    # 2 layers, 3 experts. Hand-built sums + counts so the divisions are
    # easy to read.
    score_sink_sum = torch.tensor([
        [4.0, 8.0, 0.0],
        [2.0, 0.0, 6.0],
    ], dtype=torch.float32)
    score_normal_sum = torch.tensor([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 0.0],
    ], dtype=torch.float32)
    fire_on_sink = torch.tensor([
        [2, 4, 0],
        [1, 0, 3],
    ], dtype=torch.int64)
    n_sink_tokens = torch.tensor([2, 1], dtype=torch.int64)
    n_normal_tokens = torch.tensor([4, 8], dtype=torch.int64)
    payload = _make_payload(
        n_layers=2, n_experts=3,
        score_sink_sum=score_sink_sum,
        score_normal_sum=score_normal_sum,
        fire_on_sink=fire_on_sink,
        n_sink_tokens=n_sink_tokens,
        n_normal_tokens=n_normal_tokens,
        bos_token_id=151643,
    )
    save_router_logits_stats(payload, jsonl)

    moe_layers = [
        _StubMoELayerRef(layer_idx=7),
        _StubMoELayerRef(layer_idx=11),
    ]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    ctx.set("config", _enabled_config())
    # Pre-seed sink_acc to a sentinel non-None to verify the provider
    # OVERWRITES it on hit (matches the live-path setup's behavior).
    ctx.set("sink_acc", "pre-existing-sentinel")
    provider = Stage1RouterLogitsStatsCacheProvider()
    returned = provider.on_load(ctx, jsonl)

    assert returned is not None
    assert returned.n_layers == 2
    assert returned.n_experts == 3
    acc = ctx.get("sink_acc")
    assert isinstance(acc, SinkTokenRoutingAccumulator)
    assert acc.bos_token_id == 151643
    assert acc.num_layers == 2
    assert acc.num_experts == 3

    # Layer 7 (rank 0): n_sink=2, n_normal=4.
    assert acc.mean_router_score_sink[(7, 0)] == pytest.approx(4.0 / 2.0)
    assert acc.mean_router_score_sink[(7, 1)] == pytest.approx(8.0 / 2.0)
    assert acc.mean_router_score_sink[(7, 2)] == pytest.approx(0.0)
    assert acc.mean_router_score_normal[(7, 0)] == pytest.approx(1.0 / 4.0)
    assert acc.mean_router_score_normal[(7, 1)] == pytest.approx(2.0 / 4.0)
    assert acc.mean_router_score_normal[(7, 2)] == pytest.approx(3.0 / 4.0)
    assert acc.freq_on_sink[(7, 0)] == pytest.approx(2 / 2)
    assert acc.freq_on_sink[(7, 1)] == pytest.approx(4 / 2)  # over-fires fine
    assert acc.freq_on_sink[(7, 2)] == pytest.approx(0.0)

    # Layer 11 (rank 1): n_sink=1, n_normal=8.
    assert acc.mean_router_score_sink[(11, 0)] == pytest.approx(2.0)
    assert acc.mean_router_score_sink[(11, 1)] == pytest.approx(0.0)
    assert acc.mean_router_score_sink[(11, 2)] == pytest.approx(6.0)
    assert acc.mean_router_score_normal[(11, 0)] == pytest.approx(0.0)
    assert acc.mean_router_score_normal[(11, 1)] == pytest.approx(4.0 / 8.0)
    assert acc.mean_router_score_normal[(11, 2)] == pytest.approx(0.0)
    assert acc.freq_on_sink[(11, 0)] == pytest.approx(1.0)
    assert acc.freq_on_sink[(11, 1)] == pytest.approx(0.0)
    assert acc.freq_on_sink[(11, 2)] == pytest.approx(3.0)

    # No NaNs anywhere -- zero-count denominators must produce 0.0.
    for k, v in acc.mean_router_score_sink.items():
        assert v == v, f"NaN at sink mean {k}"
    for k, v in acc.mean_router_score_normal.items():
        assert v == v, f"NaN at normal mean {k}"
    for k, v in acc.freq_on_sink.items():
        assert v == v, f"NaN at freq {k}"


# ---------------------------------------------------------------------------
# Test 2 -- load miss
# ---------------------------------------------------------------------------


def test_load_miss(tmp_path):
    """No sidecar on disk -> on_load returns None. The pre-existing
    sink_acc on ctx must remain untouched (the live-path setup wrote
    it earlier in the orchestrator and the cache provider is a no-op
    on miss)."""
    jsonl = _jsonl(tmp_path)
    moe_layers = [_StubMoELayerRef(layer_idx=0)]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    ctx.set("config", _enabled_config())
    pre_existing = SinkTokenRoutingAccumulator(
        num_layers=1, num_experts=2, bos_token_id=None,
    )
    ctx.set("sink_acc", pre_existing)

    provider = Stage1RouterLogitsStatsCacheProvider()
    result = provider.on_load(ctx, jsonl)

    assert result is None
    # The pre-existing live-path acc is untouched -- identity-preserved.
    assert ctx.get("sink_acc") is pre_existing


# ---------------------------------------------------------------------------
# Test 3 -- n_layers mismatch raises
# ---------------------------------------------------------------------------


def test_n_layers_mismatch_raises(tmp_path):
    """Live model has 3 MoE layers but the sidecar was written for 2 ->
    on_load raises ValueError pointing at the regenerate-the-sidecar
    action."""
    jsonl = _jsonl(tmp_path)
    save_router_logits_stats(_make_payload(n_layers=2, n_experts=3), jsonl)

    moe_layers = [
        _StubMoELayerRef(layer_idx=0),
        _StubMoELayerRef(layer_idx=1),
        _StubMoELayerRef(layer_idx=2),
    ]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    ctx.set("config", _enabled_config())
    provider = Stage1RouterLogitsStatsCacheProvider()
    with pytest.raises(ValueError, match="delete it to regenerate"):
        provider.on_load(ctx, jsonl)


# ---------------------------------------------------------------------------
# Test 4 -- schema-version mismatch raises
# ---------------------------------------------------------------------------


def test_schema_mismatch_raises(tmp_path, monkeypatch):
    """Bumping SCHEMA_VERSIONS['router_logits_stats'] after a sidecar is
    written makes load_router_logits_stats raise ValueError; the cache
    provider propagates the actionable message (no silent fallback)."""
    jsonl = _jsonl(tmp_path)
    save_router_logits_stats(_make_payload(n_layers=2, n_experts=3), jsonl)

    bumped = dict(SCHEMA_VERSIONS)
    bumped["router_logits_stats"] = 99
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
    ctx.set("config", _enabled_config())
    provider = Stage1RouterLogitsStatsCacheProvider()
    with pytest.raises(RuntimeError, match="manifest validation FAILED"):
        provider.on_load(ctx, jsonl)


# ---------------------------------------------------------------------------
# Test 5 -- sink_token_enabled=False returns None (R3 guard)
# ---------------------------------------------------------------------------


def test_sink_token_disabled_returns_none(tmp_path):
    """When the user has explicitly disabled sink-token detection,
    SinkTokenDetectorPlugin.setup() will have written sink_acc=None.
    The cache provider MUST honor that decision -- returning the cached
    payload would silently re-enable detection. Verify the provider
    bails BEFORE even calling load_router_logits_stats: the sidecar
    file is present + valid but the disabled-by-config branch fires
    first."""
    jsonl = _jsonl(tmp_path)
    # Write a valid sidecar so we can prove the provider's
    # short-circuit fires *before* it consults the file.
    save_router_logits_stats(_make_payload(n_layers=1, n_experts=2), jsonl)

    moe_layers = [_StubMoELayerRef(layer_idx=0)]
    ctx = PipelineContext()
    ctx.set("moe_layers", moe_layers)
    ctx.set("config", _disabled_config())
    # Pre-seed sink_acc to None (matches the live-path setup's behavior
    # when sink_token_enabled=False).
    ctx.set("sink_acc", None)

    provider = Stage1RouterLogitsStatsCacheProvider()
    result = provider.on_load(ctx, jsonl)

    assert result is None
    # sink_acc must remain None -- the cache provider must NOT have
    # overwritten it from the sidecar.
    assert ctx.get("sink_acc") is None
