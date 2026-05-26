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

    On hit, populates ``ctx.A_cov`` with the loaded ``sigma_in`` dict and
    returns the ``CovariancePayload`` so the Stage 3 orchestrator's
    ``dispatch_first("on_load", ...)`` call sees a non-None winner. On
    miss, returns ``None`` and leaves ``ctx`` untouched so the legacy
    ``_load_stage2_covariance`` fallback can run unchanged.

    Uniform with :class:`Stage4InputCovCacheProvider`: both providers
    write ``A_cov`` directly onto the ctx, and both orchestrators read
    ``ctx.get("A_cov")`` post-dispatch. This routes the cache-vs-live
    decision through the registry rather than the orchestrator's local
    glue (the inline-construct anti-pattern that the previous Stage 3
    implementation used).
    """

    name: str = "stage3_input_cov_cache"
    paper: str = (
        "Cache provider for the V2 input-covariance writer "
        "(calibration-v2 Item 1). Reads sidecars/covariance.pt and "
        "populates ctx.A_cov so the Stage 3 orchestrator's "
        "dispatch_first on_load sees a non-None winner and skips the "
        "legacy _load_stage2_covariance fallback."
    )
    # Informational: this provider is always-enabled (the cache is a
    # no-op on miss). The key names the actual driver knob the provider
    # depends on -- the calibration JSONL path that locates the sidecar
    # via ``sidecar_path(jsonl, "covariance")``.
    config_key: str = "calibration.input_covariance_cache"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("A_cov",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(self, ctx: PipelineContext,
                jsonl_path: Path) -> CovariancePayload | None:
        """Try to load the sidecar; on hit set ``ctx.A_cov`` and return
        the payload; on miss return ``None`` and leave ctx untouched.

        Schema mismatch surfaces as ``ValueError`` from
        ``load_covariance`` -- the caller MUST NOT mask that exception
        (the message is actionable: "Delete the sidecar to regenerate").
        """
        payload = load_covariance(jsonl_path)
        if payload is None:
            return None
        ctx.set("A_cov", payload.sigma_in)
        log.info(
            "stage3-input-cov-cache: loaded %d-key sidecar (%d layers × "
            "%d experts) from %s -- populated ctx.A_cov",
            len(payload.sigma_in), payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "covariance"),
        )
        return payload
