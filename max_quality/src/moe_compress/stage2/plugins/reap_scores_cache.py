"""Stage 2 cache provider for REAP saliency scores.

Reads pre-computed S_j from a sidecar produced by the
``--capture-reap-scores`` calibration flag. On cache hit, populates
``ctx.scores`` and ``ctx.freq`` so ``ReapScoringPlugin.on_score`` skips
its live accumulation (see the ctx.has("scores") guard in reap_scoring.py).
On cache miss, returns None and the live REAP-scoring path runs normally.

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
        "Cache provider for REAP saliency scores "
        "(S_j = (1/|X_j|)·Σ g_j·‖f_j‖₂, arXiv:2510.13999 Eq. 9). "
        "Reads sidecars/reap_scores.pt produced by --capture-reap-scores. "
        "On hit: populates ctx.scores + ctx.freq, suppressing the live "
        "ReapScoringPlugin.on_score forward via its ctx.has() guard. "
        "On miss: returns None; the live path runs normally."
    )
    config_key: str = "stage2_reap_ream"
    reads: tuple[str, ...] = ("layer_ref", "_layer_rank")
    writes: tuple[str, ...] = ("reap_scores_payload", "scores", "freq")
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
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
