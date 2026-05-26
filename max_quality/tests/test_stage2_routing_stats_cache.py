"""Tests for ``Stage2RoutingStatsCacheProvider`` (Item 3 reader).

Five tests:

1. ``test_load_miss`` -- no sidecar -> on_load returns None, ctx
   untouched.
2. ``test_load_hit`` -- sidecar present -> on_load returns non-None
   and stashes payload on ``ctx.routing_stats_payload``.
3. ``test_schema_mismatch_raises`` -- forced schema=99 -> ValueError
   with actionable "Delete the sidecar" message.
4. ``test_no_ctx_mutation_on_miss`` -- no sidecar -> ctx left
   untouched (no key written, no default None).
5. ``test_integration_alongside_reap_cache`` -- both
   Stage2ReapScoresCacheProvider and Stage2RoutingStatsCacheProvider
   on the same registry: REAP-cache hit on dispatch_first does NOT
   prevent the orchestrator's follow-up explicit call from loading
   the routing-stats payload (the orchestrator pattern is verified
   here against the live registry).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.stage2.plugins.reap_scores_cache import (
    Stage2ReapScoresCacheProvider,
)
from moe_compress.stage2.plugins.routing_stats_cache import (
    Stage2RoutingStatsCacheProvider,
)
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    RoutingStatsPayload,
    Stage2ReapPayload,
    save_reap_scores,
    save_routing_stats,
    sidecar_path,
)


def _make_payload(n_layers: int = 2, n_experts: int = 3) -> RoutingStatsPayload:
    return RoutingStatsPayload(
        schema_version=SCHEMA_VERSIONS["routing_stats"],
        n_experts=n_experts,
        n_layers=n_layers,
        freq=torch.arange(
            n_layers * n_experts, dtype=torch.int64
        ).reshape(n_layers, n_experts) + 1,
        mean_weight=torch.linspace(
            0.1, 0.9, n_layers * n_experts, dtype=torch.float32,
        ).reshape(n_layers, n_experts),
    )


def _make_reap_payload(n_layers: int = 2,
                       n_experts: int = 3) -> Stage2ReapPayload:
    return Stage2ReapPayload(
        schema_version=SCHEMA_VERSIONS["reap_scores"],
        n_experts=n_experts,
        n_layers=n_layers,
        reap_scores=torch.arange(
            n_layers * n_experts, dtype=torch.float32
        ).reshape(n_layers, n_experts),
        token_counts=torch.full(
            (n_layers, n_experts), 11, dtype=torch.int64
        ),
    )


def _jsonl(tmp_path: Path) -> Path:
    return tmp_path / "trace_0001.jsonl"


# ---------------------------------------------------------------------------
# Test 1 -- load miss
# ---------------------------------------------------------------------------


def test_load_miss(tmp_path):
    """No sidecar -> on_load returns None and does not touch ctx."""
    jsonl = _jsonl(tmp_path)
    provider = Stage2RoutingStatsCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is None
    assert not ctx.has("routing_stats_payload")


# ---------------------------------------------------------------------------
# Test 2 -- load hit
# ---------------------------------------------------------------------------


def test_load_hit(tmp_path):
    """Sidecar present -> on_load returns non-None and stashes the
    payload on ctx under ``routing_stats_payload``."""
    jsonl = _jsonl(tmp_path)
    payload = _make_payload()
    save_routing_stats(payload, jsonl)
    assert sidecar_path(jsonl, "routing_stats").exists()

    provider = Stage2RoutingStatsCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is not None
    assert ctx.has("routing_stats_payload")
    stashed = ctx.get("routing_stats_payload")
    assert stashed.n_layers == payload.n_layers
    assert stashed.n_experts == payload.n_experts
    assert torch.equal(stashed.freq, payload.freq.cpu())
    assert torch.equal(stashed.mean_weight, payload.mean_weight.cpu())


# ---------------------------------------------------------------------------
# Test 3 -- schema-version mismatch raises
# ---------------------------------------------------------------------------


def test_schema_mismatch_raises(tmp_path, monkeypatch):
    """Forced schema=99 -> on_load raises ValueError pointing to the
    actionable "Delete the sidecar" message."""
    jsonl = _jsonl(tmp_path)
    save_routing_stats(_make_payload(), jsonl)

    bumped = dict(SCHEMA_VERSIONS)
    bumped["routing_stats"] = 99
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )

    provider = Stage2RoutingStatsCacheProvider()
    ctx = PipelineContext()
    with pytest.raises(ValueError) as exc:
        provider.on_load(ctx, jsonl)
    msg = str(exc.value)
    assert "schema_version=1" in msg
    assert "expected 99" in msg
    assert "Delete the sidecar to regenerate" in msg


# ---------------------------------------------------------------------------
# Test 4 -- no ctx mutation on miss
# ---------------------------------------------------------------------------


def test_no_ctx_mutation_on_miss(tmp_path):
    """No sidecar -> ctx is untouched: routing_stats_payload key MUST
    NOT be set (not even to None). Verifies cache-miss semantics."""
    jsonl = _jsonl(tmp_path)
    provider = Stage2RoutingStatsCacheProvider()
    ctx = PipelineContext()
    ctx.set("untouched_marker", "still_here")

    result = provider.on_load(ctx, jsonl)
    assert result is None
    assert not ctx.has("routing_stats_payload")
    assert ctx.get("untouched_marker") == "still_here"


# ---------------------------------------------------------------------------
# Test 5 -- integration: routing-stats loaded alongside an existing
# REAP-cache hit; the orchestrator's explicit-call pattern is exercised.
# ---------------------------------------------------------------------------


def test_integration_alongside_reap_cache(tmp_path):
    """Both Stage2ReapScoresCacheProvider and
    Stage2RoutingStatsCacheProvider present + REAP sidecar present +
    routing-stats sidecar present.

    The orchestrator's behavior is:

      1. ``PluginRegistry.dispatch_first(plugins, "on_load", ...)``
         returns at the REAP cache (it hits, returning a non-None
         payload). The routing-stats provider's ``on_load`` is NOT
         invoked through this chain (dispatch_first is first-winner-
         takes-all).

      2. The orchestrator then calls
         ``Stage2RoutingStatsCacheProvider.on_load(...)`` EXPLICITLY
         via the ``isinstance(_plug, Stage2RoutingStatsCacheProvider)``
         lookup so the routing-stats payload also lands on ctx.

    This test replays both steps and verifies that:
      * After step 1 only ``reap_scores_payload`` is on ctx.
      * After step 2 ``routing_stats_payload`` is also on ctx.
      * The REAP provider's hit was NOT clobbered.
    """
    jsonl = _jsonl(tmp_path)
    save_reap_scores(_make_reap_payload(), jsonl)
    save_routing_stats(_make_payload(), jsonl)

    reap = Stage2ReapScoresCacheProvider()
    rts = Stage2RoutingStatsCacheProvider()
    registry = PluginRegistry([reap, rts])
    plugins = registry.enabled({})

    run_ctx = PipelineContext()

    # Step 1: dispatch_first stops at REAP.
    result = PluginRegistry.dispatch_first(
        plugins, "on_load", run_ctx, jsonl,
    )
    assert result is not None
    assert run_ctx.has("reap_scores_payload")
    # routing_stats NOT loaded yet -- dispatch_first stopped at REAP.
    assert not run_ctx.has("routing_stats_payload")

    # Step 2: orchestrator's explicit follow-up call for routing-stats.
    for _plug in plugins:
        if isinstance(_plug, Stage2RoutingStatsCacheProvider):
            _plug.on_load(run_ctx, jsonl)
            break

    # Both payloads are now on ctx.
    assert run_ctx.has("reap_scores_payload")
    assert run_ctx.has("routing_stats_payload")
    stashed_rts = run_ctx.get("routing_stats_payload")
    assert isinstance(stashed_rts, RoutingStatsPayload)
    # REAP hit was preserved.
    assert run_ctx.get("reap_scores_payload").n_layers == 2
