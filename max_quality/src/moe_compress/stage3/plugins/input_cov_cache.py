"""Stage 3 cache provider for per-(layer, expert, matrix) input covariance.

Reads a pre-computed Σ_in dict from a sidecar produced by the
``--capture-input-covariance`` calibration flag (V2 writer in
``vllm.calibration_input_cov``). On cache hit, returns the
``CovariancePayload`` whose ``sigma_in`` dict is shape-compatible with
the ``A_cov`` dict that the Stage 3 orchestrator otherwise loads from
``_stage2_input_covariance.pt``. The orchestrator (see
``stage3/orchestrator.py``) tries this cache FIRST and only falls back to
``_load_stage2_covariance`` on miss.

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0. The
on-disk dict shape (keys = ``(layer_idx, expert_idx, matrix_name)``,
values = ``Tensor[d_in, d_in]`` fp16) is identical to the Stage 2
writer's so downstream loaders / ``_cov_lookup`` work unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    CovariancePayload,
    load_covariance,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage3InputCovCacheProvider(BaseCacheProvider):
    """Cache-side provider for the Stage 2/3 input-covariance sidecar.

    On hit, returns the loaded ``CovariancePayload`` so the Stage 3
    orchestrator can plug ``payload.sigma_in`` straight into ``A_cov``.
    No ctx mutation -- the orchestrator owns the slot binding because the
    legacy fallback (``_load_stage2_covariance``) also writes ``A_cov``
    inline, and the orchestrator picks one or the other.
    """

    name: str = "stage3_input_cov_cache"
    paper: str = (
        "Cache provider for the V2 input-covariance writer "
        "(calibration-v2 Item 1). Reads sidecars/covariance.pt and "
        "returns a CovariancePayload whose ``sigma_in`` dict is the "
        "drop-in replacement for the in-memory A_cov dict that the "
        "Stage 3 orchestrator otherwise loads from "
        "_stage2_input_covariance.pt."
    )
    # Informational: this provider is always-enabled (the cache is a
    # no-op on miss).
    config_key: str = "stage3_svd.aa_svd.cross_covariance"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(self, ctx: PipelineContext,
                jsonl_path: Path) -> CovariancePayload | None:
        """Try to load the sidecar; return payload on hit, None on miss.

        Does NOT mutate ctx -- the Stage 3 orchestrator inspects the
        returned payload directly (it owns the A_cov binding because the
        legacy ``_load_stage2_covariance`` fallback also writes A_cov
        inline; routing both writes through ctx.set would duplicate the
        binding and obscure the cache-vs-live decision).
        """
        payload = load_covariance(jsonl_path)
        if payload is None:
            return None
        log.info(
            "stage3-input-cov-cache: loaded %d-key sidecar (%d layers × "
            "%d experts) from %s",
            len(payload.sigma_in), payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "covariance"),
        )
        return payload
