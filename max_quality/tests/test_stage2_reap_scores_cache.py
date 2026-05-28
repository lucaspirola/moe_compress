"""Tests for ``Stage2ReapScoresCacheProvider`` (V1+V2 reader).

Covers:
1. ``test_load_miss`` -- no sidecar → on_load returns None.
2. ``test_save_load_roundtrip`` -- synthetic payload survives save+load
   via the Item-0 helper.
3. ``test_schema_mismatch_raises`` -- forced schema=99 → ValueError with
   actionable "Delete the sidecar" message.
4. ``test_on_load_hit`` -- sidecar present → on_load returns non-None
   and stashes payload on ctx.
5. ``test_on_load_miss`` -- no sidecar → on_load returns None, no ctx
   mutation.
6. ``test_on_score_hit_populates_ctx`` -- synthetic payload on ctx →
   on_score sets scores + freq.
7. ``test_integration_cache_hit_suppresses_live`` -- full registry
   ``[cache, reap_scoring]``, cache hit → ReapScoringPlugin.on_score
   returns early via the ``ctx.has("scores")`` guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from moe_compress.pipeline.context import PipelineContext
from moe_compress.pipeline.registry import PluginRegistry
from moe_compress.stage2.plugins.reap_scores_cache import (
    Stage2ReapScoresCacheProvider,
)
from moe_compress.stage2.plugins.reap_scoring import ReapScoringPlugin
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    Stage2ReapPayload,
    load_reap_scores,
    save_reap_scores,
    sidecar_path,
)


def _make_payload(n_layers: int = 2, n_experts: int = 3) -> Stage2ReapPayload:
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
    """No sidecar → load_reap_scores returns None (cache-miss signal)."""
    jsonl = _jsonl(tmp_path)
    assert load_reap_scores(jsonl) is None


# ---------------------------------------------------------------------------
# Test 2 -- save/load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    """Synthetic payload survives the Item-0 atomic save/load."""
    jsonl = _jsonl(tmp_path)
    payload = _make_payload()
    save_reap_scores(payload, jsonl)
    assert sidecar_path(jsonl, "reap_scores").exists()

    loaded = load_reap_scores(jsonl)
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSIONS["reap_scores"]
    assert loaded.n_layers == payload.n_layers
    assert loaded.n_experts == payload.n_experts
    assert torch.equal(loaded.reap_scores, payload.reap_scores.cpu())
    assert torch.equal(loaded.token_counts, payload.token_counts.cpu())
    assert loaded.reap_scores.dtype == torch.float32
    assert loaded.token_counts.dtype == torch.int64


# ---------------------------------------------------------------------------
# Test 3 -- schema-version mismatch raises
# ---------------------------------------------------------------------------


def test_schema_mismatch_raises(tmp_path, monkeypatch):
    """Forced schema=99 → load_reap_scores raises ``RuntimeError`` from
    Pattern O manifest validation with the actionable 'Delete both ...
    re-run calibration' message."""
    jsonl = _jsonl(tmp_path)
    save_reap_scores(_make_payload(), jsonl)

    # Bump the central version *after* the write, mimicking a code upgrade.
    bumped = dict(SCHEMA_VERSIONS)
    bumped["reap_scores"] = 99
    monkeypatch.setattr(
        "moe_compress.utils.cached_calibration_signals.SCHEMA_VERSIONS",
        bumped,
    )

    with pytest.raises(RuntimeError) as exc:
        load_reap_scores(jsonl)
    msg = str(exc.value)
    assert "manifest validation FAILED" in msg
    assert "schema_version=1" in msg
    assert "expected 99" in msg
    assert "re-run calibration" in msg


# ---------------------------------------------------------------------------
# Test 4 -- on_load hit
# ---------------------------------------------------------------------------


def test_on_load_hit(tmp_path):
    """Sidecar present → on_load returns non-None and stashes the payload
    on ctx under ``reap_scores_payload``."""
    jsonl = _jsonl(tmp_path)
    payload = _make_payload()
    save_reap_scores(payload, jsonl)

    provider = Stage2ReapScoresCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is not None
    assert ctx.has("reap_scores_payload")
    stashed = ctx.get("reap_scores_payload")
    assert stashed.n_layers == payload.n_layers
    assert stashed.n_experts == payload.n_experts


# ---------------------------------------------------------------------------
# Test 5 -- on_load miss
# ---------------------------------------------------------------------------


def test_on_load_miss(tmp_path):
    """No sidecar → on_load returns None and does not touch ctx."""
    jsonl = _jsonl(tmp_path)
    provider = Stage2ReapScoresCacheProvider()
    ctx = PipelineContext()
    result = provider.on_load(ctx, jsonl)
    assert result is None
    assert not ctx.has("reap_scores_payload")


# ---------------------------------------------------------------------------
# Test 6 -- on_score populates ctx from cached payload
# ---------------------------------------------------------------------------


def test_on_score_hit_populates_ctx(tmp_path):
    """With ``reap_scores_payload`` stashed on ctx + ``_layer_rank`` set,
    on_score must populate ``scores`` (np.ndarray) and ``freq`` (dict)
    derived from the cached payload."""
    payload = _make_payload(n_layers=2, n_experts=3)
    provider = Stage2ReapScoresCacheProvider()
    ctx = PipelineContext()
    ctx.set("reap_scores_payload", payload)
    ctx.set("_layer_rank", 1)  # second layer of payload

    provider.on_score(ctx)
    assert ctx.has("scores")
    assert ctx.has("freq")

    scores = ctx.get("scores")
    freq = ctx.get("freq")
    assert isinstance(scores, np.ndarray)
    # Row 1 of [[0,1,2],[3,4,5]] is [3,4,5].
    np.testing.assert_array_equal(scores, np.array([3.0, 4.0, 5.0],
                                                    dtype=np.float32))
    # token_counts was a full-11 matrix.
    assert freq == {0: 11, 1: 11, 2: 11}


# ---------------------------------------------------------------------------
# Test 7 -- integration: cache hit suppresses live REAP forward
# ---------------------------------------------------------------------------


def test_integration_cache_hit_suppresses_live(tmp_path):
    """Full registry ``[cache, reap_scoring]``: cache hit -> live
    ReapScoringPlugin.on_score returns early via ``ctx.has("scores")``.

    We exercise the path by:
      * placing a sidecar so dispatch_first("on_load") hits the cache,
      * driving the cache's on_score (which sets ctx.scores + ctx.freq),
      * then calling ReapScoringPlugin.on_score WITHOUT having seeded
        ``reap_acc`` / ``layer_ref`` on ctx. If the early-return guard
        works, this call is a no-op; if the guard is missing, the live
        path would raise KeyError trying to read ``layer_ref``.
    """
    jsonl = _jsonl(tmp_path)
    save_reap_scores(_make_payload(n_layers=1, n_experts=4), jsonl)

    cache = Stage2ReapScoresCacheProvider()
    live = ReapScoringPlugin()
    registry = PluginRegistry([cache, live])

    run_ctx = PipelineContext()
    # Dispatch_first("on_load") -- the cache wins, live's no-op on_load
    # returns None so dispatch_first stops at the cache hit.
    result = PluginRegistry.dispatch_first(
        registry.enabled({}), "on_load", run_ctx, jsonl,
    )
    assert result is not None
    assert run_ctx.has("reap_scores_payload")

    # Per-layer context.
    layer_ctx = run_ctx.child()
    layer_ctx.set("_layer_rank", 0)

    # Cache's on_score populates scores + freq.
    cache.on_score(layer_ctx)
    assert layer_ctx.has("scores")
    assert layer_ctx.has("freq")

    # Live's on_score must NOT touch ctx.layer_ref / ctx.reap_acc on a
    # cache hit -- the guard short-circuits before any get().
    live.on_score(layer_ctx)  # no exception expected

    # And scores/freq stay equal to the cache-populated values (live
    # didn't overwrite them).
    np.testing.assert_array_equal(
        layer_ctx.get("scores"),
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
