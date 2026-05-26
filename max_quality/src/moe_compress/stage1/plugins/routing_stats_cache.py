"""Stage 1 cache provider for per-(layer, expert) routing-frequency +
mean-routing-weight statistics.

Reads a pre-computed ``routing_stats`` payload from a sidecar produced
by the ``--capture-routing-stats`` calibration flag (Item 3 of the
calibration-v2 writers campaign, in
``vllm.calibration_routing_stats``). On cache hit, deposits the payload
onto ``ctx["routing_stats_payload"]`` for future read-side plugins;
on miss, returns None and leaves the ctx untouched.

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0.
The cached payload is indexed by ``(layer_rank, expert_id)`` on disk;
this provider validates ``n_layers`` against the live ``MoELayerRef``
list and then publishes the payload onto ``ctx`` as-is (no
rank -> layer_idx translation; consumers do the mapping at read time
if they need it).

Divergence from the canonical provider-pair pattern (see
``cached_calibration_signals.py`` module docstring): this provider has
NO live counterpart in Stage 1 and NO immediate downstream consumer.
Item 3 ships only the writer + the cache readers; the read-side
consumers (routing-aware ablation gating, mean-weight-weighted REAP
variants, ...) will be added by later items. The cache provider is
laid down now to keep the on-disk schema stable and the ctx-slot
contract testable from day 1.

Manifest-exclusion note: this plugin is intentionally NOT in
``STAGE1_PLUGIN_MANIFEST`` (see ``stage1/plugins/__init__.py``). It is
instantiated directly by ``stage1/orchestrator.py`` at STEP 4.6
because Stage 1 has no canonical "live provider" for routing stats to
fall through to via ``PluginRegistry.dispatch_first``. Future
refactors that promote routing stats to a proper plugin-pair should
also register this cache provider in the manifest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    RoutingStatsPayload,
    load_routing_stats,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage1RoutingStatsCacheProvider(BaseCacheProvider):
    """Cache-side provider for Stage 1's per-(layer, expert) routing-stats
    payload.

    On hit, sets ``ctx.routing_stats_payload`` to the loaded payload and
    returns it; on miss, returns ``None`` and leaves ``ctx`` untouched.
    There is NO immediate live consumer: this provider purely deposits
    the payload for future read-side plugins to consume.
    """

    name: str = "routing_stats_cache"
    paper: str = (
        "Cache provider for per-(layer, expert) routing frequency + "
        "mean routing weight. Reads sidecars/routing_stats.pt produced "
        "by --capture-routing-stats (Item 3 of the calibration-v2 "
        "writers campaign). On hit: populates ctx.routing_stats_payload "
        "with the loaded RoutingStatsPayload (no per-layer derivation; "
        "consumers map rank -> layer_idx at read time). On miss: "
        "returns None; ctx untouched. No immediate downstream consumer "
        "-- the payload is laid down as infrastructure for future "
        "routing-aware ablation / merging plugins."
    )
    config_key: str = "calibration.routing_stats_cache"
    reads: tuple[str, ...] = ("moe_layers",)
    writes: tuple[str, ...] = ("routing_stats_payload",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        # Always enabled: cache provider is a no-op on miss.
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(self, ctx: PipelineContext,
                jsonl_path: Path) -> RoutingStatsPayload | None:
        """Try to load the sidecar; on hit, populate
        ``ctx.routing_stats_payload``."""
        payload = load_routing_stats(jsonl_path)
        if payload is None:
            return None

        # Topology consistency: the sidecar was written for a specific
        # MoE-layer count; refuse to populate the ctx if the live model
        # disagrees (mirrors Stage 1's per_expert_max cache precedent).
        moe_layers = ctx.get("moe_layers")
        n_layers = len(moe_layers)
        if n_layers != payload.n_layers:
            raise ValueError(
                f"routing_stats cache mismatch: sidecar reports "
                f"n_layers={payload.n_layers} but the live model has "
                f"{n_layers} MoE layers. The sidecar was produced for a "
                f"different model topology -- delete it to regenerate."
            )

        ctx.set("routing_stats_payload", payload)
        log.info(
            "routing-stats-cache: loaded %d-layer x %d-expert sidecar "
            "from %s -- populated ctx.routing_stats_payload",
            payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "routing_stats"),
        )
        return payload
