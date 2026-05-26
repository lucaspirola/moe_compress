"""Stage 1 cache provider for per-(layer, expert) down_proj output max L_inf.

Reads a pre-computed ``per_expert_max`` payload from a sidecar produced
by the ``--capture-per-expert-max`` calibration flag (Item 2 of the
calibration-v2 writers campaign, in
``vllm.calibration_per_expert_max``). On cache hit, constructs a
``DownProjMaxAccumulator`` and populates its ``per_expert_max`` dict
from the cached tensor, then sets it on ``ctx["max_acc"]`` so the Stage
1 detector plugins (``three_way_and``, ``aimer``, ``magnitude_topk``,
``ablation_filter``) read the cached values without running the live
Phase B forward pass to compute them.

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0. The
cached payload is indexed by ``(layer_rank, expert_id)`` on disk; this
provider maps ``rank -> layer_idx`` via ``ctx["moe_layers"]`` (an
ordered list of ``MoELayerRef``) before populating the dict so the
keying matches the live ``DownProjMaxAccumulator`` semantics.

Indexing convention: the live accumulator omits keys for experts that
received zero tokens. This provider mirrors that by filtering out
cells where ``per_expert_max[rank, expert_id] == 0.0`` (the writer's
zero-fill convention for ``-inf`` zero-traffic cells).

Note: This plugin is intentionally NOT in ``STAGE1_PLUGIN_MANIFEST``
(see ``stage1/plugins/__init__.py``). It is instantiated directly by
``stage1/orchestrator.py`` at STEP 4.5 because Stage 1's live profile
path is an accumulator-factory + calibration-pass closure, not a
registered plugin — there is no canonical "live provider" to fall
through to via ``PluginRegistry.dispatch_first``. Future refactors
that promote the calibration pass to a proper plugin should also
register this cache provider in the manifest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...pipeline.context import PipelineContext
from ...utils.activation_hooks import DownProjMaxAccumulator
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    Stage1PerExpertMaxPayload,
    load_per_expert_max,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage1PerExpertMaxCacheProvider(BaseCacheProvider):
    """Cache-side provider for Stage 1's per-(layer, expert) down_proj
    max-L_inf accumulator.

    On hit, builds a ``DownProjMaxAccumulator`` whose ``per_expert_max``
    dict is hydrated from the sidecar, sets it on ``ctx.max_acc``, and
    returns the loaded payload so the orchestrator's hit-check
    (``ctx.has("max_acc")``) sees a non-None winner. On miss, returns
    ``None`` and leaves ``ctx`` untouched so the live
    ``DownProjMaxAccumulator`` path runs unchanged.
    """

    name: str = "per_expert_max_cache"
    paper: str = (
        "Cache provider for per-(layer, expert) max|f_j(x)|_∞. Reads "
        "sidecars/per_expert_max.pt produced by --capture-per-expert-max. "
        "On hit: populates ctx.max_acc with a hydrated "
        "DownProjMaxAccumulator (zero-traffic experts excluded to match "
        "live behavior); the live max-magnitude accumulator is skipped. "
        "On miss: returns None; live DownProjMaxAccumulator path runs."
    )
    config_key: str = "calibration.per_expert_max_cache"
    reads: tuple[str, ...] = ("moe_layers",)
    writes: tuple[str, ...] = ("max_acc",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        # Always enabled: cache provider is a no-op on miss.
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(self, ctx: PipelineContext,
                jsonl_path: Path) -> Stage1PerExpertMaxPayload | None:
        """Try to load the sidecar; on hit, populate ``ctx.max_acc``."""
        payload = load_per_expert_max(jsonl_path)
        if payload is None:
            return None

        # Map rank -> layer_idx via the live MoELayerRef list. The cached
        # tensor's rows are in the same order as iter_moe_layers (same
        # convention the writer uses via named_modules() with moe_layer_id).
        moe_layers = ctx.get("moe_layers")
        n_layers = len(moe_layers)
        if n_layers != payload.n_layers:
            raise ValueError(
                f"per_expert_max cache mismatch: sidecar reports "
                f"n_layers={payload.n_layers} but the live model has "
                f"{n_layers} MoE layers. The sidecar was produced for a "
                f"different model topology -- delete it to regenerate."
            )

        acc = DownProjMaxAccumulator()
        n_inserted = 0
        for rank in range(n_layers):
            layer_idx = moe_layers[rank].layer_idx
            for expert_id in range(payload.n_experts):
                v = float(payload.per_expert_max[rank, expert_id].item())
                # Match the live accumulator's absent-key convention for
                # zero-traffic experts: skip cells whose max is exactly
                # 0.0 (the writer's zero-fill for -inf zero-traffic).
                if v > 0.0:
                    acc.per_expert_max[(layer_idx, expert_id)] = v
                    n_inserted += 1

        ctx.set("max_acc", acc)
        log.info(
            "per-expert-max-cache: loaded %d-layer × %d-expert sidecar "
            "from %s -- populated ctx.max_acc with %d non-zero entries",
            payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "per_expert_max"), n_inserted,
        )
        return payload
