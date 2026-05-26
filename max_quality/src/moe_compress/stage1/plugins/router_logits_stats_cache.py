"""Stage 1 cache provider for per-(layer, expert) sink-vs-normal router-score
aggregates.

Reads a pre-computed ``router_logits_stats`` payload from a sidecar
produced by the ``--capture-router-logits-stats`` calibration flag
(Item 4 of the calibration-v2 writers campaign, in
``vllm.calibration_router_logits_stats``). On cache hit, constructs a
:class:`SinkTokenRoutingAccumulator` whose three finalize-target dicts
(``mean_router_score_sink``, ``mean_router_score_normal``,
``freq_on_sink``) are populated directly from the cached aggregates,
then deposits it on ``ctx["sink_acc"]`` -- the SAME slot the live
``SinkTokenDetectorPlugin.setup()`` writes earlier in the orchestrator.

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0.
The Stage 1 orchestrator's STEP 4.7 instantiates this provider directly
and, on cache hit, drops ``"sink_routing"`` from the ``needed`` set so
the live router-logits + softmax + top-k HookSpec is NOT registered;
the Stage 1 calibration pass then skips the per-batch
``sink_acc.update`` work entirely. The orchestrator additionally skips
the post-pass ``finalize()`` on cache hit because the cache reader pre-
populates the same finalize-target dicts.

R3 guard: when ``stage1_grape.super_expert_detection.sink_token_enabled``
is False, ``SinkTokenDetectorPlugin.setup()`` writes ``sink_acc=None``
and the cache reader MUST honor that decision -- returning the cached
payload would let a stale accumulator override the user's explicit
disable. The provider returns ``None`` (cache miss) on
``sink_token_enabled=False`` so the orchestrator leaves the
``sink_acc=None`` ctx slot untouched.

Indexing: the cached tensor's rows are in the same order as
``iter_moe_layers`` (writer + reader share the
``named_modules() -> moe_layer_id`` discovery ordering used by every
calibration-v2 writer). The ``moe_layers`` list pulled from ctx
provides the rank -> layer_idx mapping for the SE detector's dict-keyed
output.

Manifest-exclusion note: this plugin is intentionally NOT in
``STAGE1_PLUGIN_MANIFEST`` (see ``stage1/plugins/__init__.py``). It is
instantiated directly by ``stage1/orchestrator.py`` at STEP 4.7 because
Stage 1's live sink-routing path is a setup()-built accumulator +
calibration-pass HookSpec, not a registered provider plugin -- there is
no canonical "live provider" for ``PluginRegistry.dispatch_first`` to
fall through to. Future refactors that promote the calibration pass to
a proper plugin-pair should also register this cache provider in the
manifest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    RouterLogitsStatsPayload,
    load_router_logits_stats,
    sidecar_path,
)
from ...utils.sink_token_routing import SinkTokenRoutingAccumulator

log = logging.getLogger(__name__)


class Stage1RouterLogitsStatsCacheProvider(BaseCacheProvider):
    """Cache-side provider for Stage 1's per-(layer, expert) sink-vs-normal
    router-score aggregates.

    On hit, hydrates a pre-finalized :class:`SinkTokenRoutingAccumulator`
    from the sidecar and OVERWRITES ``ctx.sink_acc`` (the live
    setup-built accumulator); returns the loaded payload so the
    orchestrator's hit-check sees a non-None winner. On miss (including
    the sink-token-disabled R3 path), returns ``None`` and leaves
    ``ctx`` untouched so the live router-logits HookSpec gets
    registered + the live accumulator gets fed at Phase B.
    """

    name: str = "router_logits_stats_cache"
    paper: str = (
        "Cache provider for per-(layer, expert) sink-vs-normal router-"
        "score aggregates. Reads sidecars/router_logits_stats.pt "
        "produced by --capture-router-logits-stats (Item 4 of the "
        "calibration-v2 writers campaign). On hit: hydrates a pre-"
        "finalized SinkTokenRoutingAccumulator into ctx.sink_acc "
        "(overwrites the live setup-built accumulator); the live "
        "router-logits HookSpec is dropped from the calibration "
        "registration set. On miss OR when sink_token_enabled=False: "
        "returns None; live sink-routing pass runs unchanged."
    )
    config_key: str = "calibration.router_logits_stats_cache"
    reads: tuple[str, ...] = ("moe_layers", "config", "sink_acc")
    writes: tuple[str, ...] = ("sink_acc",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        # Always enabled: cache provider is a no-op on miss.
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(
        self, ctx: PipelineContext, jsonl_path: Path,
    ) -> RouterLogitsStatsPayload | None:
        """Try to load the sidecar; on hit, populate ``ctx.sink_acc``."""
        # R3 guard: if the user has explicitly disabled sink-token detection
        # via the Stage 1 config, SinkTokenDetectorPlugin.setup() will have
        # written sink_acc=None earlier in STEP 4. Returning the cached
        # payload would then OVERRIDE that explicit user decision -- which
        # would be a footgun (a sidecar lying around from a previous run
        # would silently re-enable sink-token detection). Bail on cache
        # miss instead so the None sink_acc stays None.
        config = ctx.get("config")
        s1 = config.get("stage1_grape", {})
        se_cfg = s1.get("super_expert_detection", {})
        sink_token_enabled = bool(se_cfg.get("sink_token_enabled", True))
        if not sink_token_enabled:
            log.info(
                "router-logits-stats-cache: sink_token_enabled=False -- "
                "skipping cache lookup; ctx.sink_acc remains None."
            )
            return None

        payload = load_router_logits_stats(jsonl_path)
        if payload is None:
            return None

        # Topology consistency: the sidecar was written for a specific
        # MoE-layer count; refuse to hydrate the accumulator if the live
        # model disagrees (mirrors Stage 1's per_expert_max cache
        # precedent).
        moe_layers = ctx.get("moe_layers")
        n_layers = len(moe_layers)
        if n_layers != payload.n_layers:
            raise ValueError(
                f"router_logits_stats cache mismatch: sidecar reports "
                f"n_layers={payload.n_layers} but the live model has "
                f"{n_layers} MoE layers. The sidecar was produced for a "
                f"different model topology -- delete it to regenerate."
            )

        # Pre-finalize a SinkTokenRoutingAccumulator from the cached
        # aggregates. The accumulator's finalize() target dicts are the
        # only state the downstream SinkTokenDetectorPlugin.run +
        # contribute_artifact paths read; everything else
        # (_sum_score_*_per_layer, _count_*_per_layer, ...) is internal
        # to the live ``update`` path which we're bypassing here.
        acc = SinkTokenRoutingAccumulator(
            num_layers=payload.n_layers,
            num_experts=payload.n_experts,
            bos_token_id=payload.bos_token_id,
        )
        for rank in range(payload.n_layers):
            layer_idx = int(moe_layers[rank].layer_idx)
            n_sink = int(payload.n_sink_tokens[rank].item())
            n_normal = int(payload.n_normal_tokens[rank].item())
            for expert_id in range(payload.n_experts):
                ss = float(payload.score_sink_sum[rank, expert_id].item())
                sn = float(payload.score_normal_sum[rank, expert_id].item())
                fs = int(payload.fire_on_sink[rank, expert_id].item())
                # Match SinkTokenRoutingAccumulator.finalize() semantics:
                # zero-count denominator -> 0.0 (no NaN). The detector +
                # contribute_artifact paths read these dicts post-
                # finalize, so this IS the finalized form.
                acc.mean_router_score_sink[(layer_idx, expert_id)] = (
                    ss / n_sink if n_sink > 0 else 0.0
                )
                acc.mean_router_score_normal[(layer_idx, expert_id)] = (
                    sn / n_normal if n_normal > 0 else 0.0
                )
                acc.freq_on_sink[(layer_idx, expert_id)] = (
                    fs / n_sink if n_sink > 0 else 0.0
                )

        # Overwrite the live-path acc that STEP 4's
        # SinkTokenDetectorPlugin.setup() wrote.
        ctx.set("sink_acc", acc, overwrite=True)
        log.info(
            "router-logits-stats-cache: loaded %d-layer x %d-expert sidecar "
            "from %s -- ctx.sink_acc replaced with pre-finalized "
            "accumulator (bos_token_id=%s)",
            payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "router_logits_stats"),
            payload.bos_token_id,
        )
        return payload
