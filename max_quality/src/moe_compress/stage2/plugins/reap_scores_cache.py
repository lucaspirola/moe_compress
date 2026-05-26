"""Stage 2 cache provider for REAP saliency scores.

Reads pre-computed S_j from a sidecar produced by the
``--capture-reap-scores`` calibration flag. On cache hit, populates
``ctx.scores`` and ``ctx.freq`` so ``ReapScoringPlugin.on_score`` short-
circuits its live finalize/derive step (via its ``ctx.has("scores")``
guard). The per-layer profile FORWARD pass still runs as part of
``LayerMergePlugin.on_profile`` -- it is needed for covariance
collection consumed by Stage 3/4. The cache hit only avoids the
finalize + score-derivation at the end of the profile pass; the
in-forward per-token REAP accumulation is a free side effect that
gets discarded.

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    Stage2ReapPayload,
    load_reap_scores,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage2ReapScoresCacheProvider:
    """Cache-side provider for REAP saliency scores (Stage 2)."""

    name: str = "reap_scores_cache"
    paper: str = (
        "Cache provider for REAP S_j (arXiv:2510.13999 Eq. 9). "
        "Reads sidecars/reap_scores.pt and populates ctx.scores + ctx.freq "
        "on hit so ReapScoringPlugin.on_score's ctx.has guard short-"
        "circuits the live finalize step. The per-layer profile forward "
        "still runs (LayerMergePlugin.on_profile owns it) for covariance."
    )
    config_key: str = "stage2_reap_ream"
    reads: tuple[str, ...] = ("_layer_rank", "reap_scores_payload")
    writes: tuple[str, ...] = ("reap_scores_payload", "scores", "freq")
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        # Always enabled: cache provider is a no-op on miss (returns None
        # gracefully, dispatch_first falls through to ReapScoringPlugin).
        # No need to gate on a YAML knob.
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(self, ctx: PipelineContext,
                jsonl_path: Path) -> Stage2ReapPayload | None:
        """Run-scope: try to load the sidecar; stash payload on ctx."""
        payload = load_reap_scores(jsonl_path)
        if payload is None:
            return None
        ctx.set("reap_scores_payload", payload)
        log.info(
            "reap-scores-cache: loaded %d-layer × %d-expert sidecar from %s",
            payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "reap_scores"),
        )
        return payload

    def on_score(self, ctx: PipelineContext) -> None:
        """Per-layer: populate scores + freq from the cached payload."""
        if not ctx.has("reap_scores_payload"):
            return
        payload: Stage2ReapPayload = ctx.get("reap_scores_payload")
        layer_rank = ctx.get("_layer_rank")
        scores_row = payload.reap_scores[layer_rank].numpy()
        counts_row = payload.token_counts[layer_rank]
        n_experts = int(counts_row.numel())
        ctx.set("scores", scores_row)
        ctx.set("freq", {e: int(counts_row[e].item()) for e in range(n_experts)})
