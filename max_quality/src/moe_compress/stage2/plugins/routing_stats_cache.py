"""Stage 2 cache provider for per-(layer, expert) routing-frequency +
mean-routing-weight statistics.

Reads pre-computed routing-stats payload from a sidecar produced by the
``--capture-routing-stats`` calibration flag (Item 3 of the
calibration-v2 writers campaign). On cache hit, populates
``ctx.routing_stats_payload`` so future read-side plugins can consume
it. There is NO immediate per-layer consumer at present -- this
provider lays infrastructure only.

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0.
The Stage 2 routing-stats provider mirrors the plain (no
``BaseCacheProvider`` ABC) shape used by
``Stage2ReapScoresCacheProvider`` so registration in the existing
``PluginRegistry`` works without ABC machinery.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    RoutingStatsPayload,
    load_routing_stats,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage2RoutingStatsCacheProvider:
    """Cache-side provider for routing-stats (Stage 2).

    On hit, sets ``ctx.routing_stats_payload`` to the loaded payload
    and returns it; on miss, returns ``None`` and leaves ``ctx``
    untouched. No per-layer ``on_score`` hook -- there is no immediate
    per-layer consumer.
    """

    name: str = "routing_stats_cache"
    paper: str = (
        "Cache provider for per-(layer, expert) routing frequency + "
        "mean routing weight. Reads sidecars/routing_stats.pt produced "
        "by --capture-routing-stats (Item 3 of the calibration-v2 "
        "writers campaign). On hit: populates ctx.routing_stats_payload "
        "with the loaded RoutingStatsPayload. On miss: returns None; "
        "ctx untouched. No immediate Stage 2 consumer -- the payload "
        "is laid down as infrastructure for future routing-aware "
        "merging / cost-shaping plugins."
    )
    config_key: str = "calibration.routing_stats_cache"
    reads: tuple[str, ...] = ("routing_stats_payload",)
    writes: tuple[str, ...] = ("routing_stats_payload",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        # Always enabled: cache provider is a no-op on miss.
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(self, ctx: PipelineContext,
                jsonl_path: Path) -> RoutingStatsPayload | None:
        """Run-scope: try to load the sidecar; stash payload on ctx."""
        payload = load_routing_stats(jsonl_path)
        if payload is None:
            return None
        ctx.set("routing_stats_payload", payload)
        log.info(
            "routing-stats-cache: loaded %d-layer x %d-expert sidecar from %s",
            payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "routing_stats"),
        )
        return payload
