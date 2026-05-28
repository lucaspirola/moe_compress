"""Stage 3 cache provider for the Wanda ``scalar_row`` calibration sidecar.

W-1 (audit `tasks/AUDIT_CALIBRATION_COMPLETENESS_V2.md` §W-1, plan
`tasks/PLAN_W1_WANDA_SCALAR_ROW_CAPTURE.md`).

Reads a pre-computed per-(layer, expert, "gate_proj") ``scalar_row``
running-mean payload from a sidecar produced by the
``--capture-wanda-scalar-row`` calibration flag (W-1 writer in
``vllm.calibration_wanda_scalar_row``). On cache hit, populates
``ctx["stage3.wanda_scalar_row"]`` with the
:class:`WandaScalarRowPayload` so
``WandaIntraExpertScorePlugin.collect_wanda_scores`` short-circuits its
per-layer calibration sweep entirely (see the plugin's
D-zero-extra-forward block).

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0.
Mirror of :class:`Stage3InputCovCacheProvider` — both providers write
their respective ctx slot, both are always-enabled (the cache is a
no-op on miss), and both rely on
``moe_compress.utils.cached_calibration_signals.load_*`` for
manifest-aware validated loads (Pattern O — manifest is REQUIRED for
this green-field sidecar).
"""
from __future__ import annotations

import logging
from pathlib import Path

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    WandaScalarRowPayload,
    load_wanda_scalar_row,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage3WandaScalarRowCacheProvider(BaseCacheProvider):
    """Cache-side provider for the W-1 Wanda scalar_row sidecar.

    On hit, populates ``ctx["stage3.wanda_scalar_row"]`` with the loaded
    :class:`WandaScalarRowPayload` and returns the payload so the Stage
    3 orchestrator's ``dispatch_first("on_load", ...)`` call sees a
    non-None winner. On miss, returns ``None`` and leaves ``ctx``
    untouched so the consumer plugin's in-process calibration sweep
    (cache-MISS fallback) can run unchanged.

    Uniform with :class:`Stage3InputCovCacheProvider`: both providers
    write a dedicated ctx slot, both orchestrators read the slot
    post-dispatch, and both use ``load_*`` from
    ``utils.cached_calibration_signals`` for schema + manifest
    validation.
    """

    name: str = "stage3_wanda_scalar_row_cache"
    paper: str = (
        "Cache provider for the W-1 Wanda scalar_row sidecar "
        "(audit/PLAN_W1). Reads sidecars/wanda_scalar_row.pt (Pattern "
        "B + Pattern O -- manifest REQUIRED) and populates "
        "ctx['stage3.wanda_scalar_row'] so WandaIntraExpertScorePlugin "
        "short-circuits its per-layer calibration sweep. Mirrors "
        "Stage3InputCovCacheProvider's contract."
    )
    config_key: str = "calibration.wanda_scalar_row_cache"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("stage3.wanda_scalar_row",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        # Always-enabled: cache is a no-op on miss. Disabling the
        # consumer plugin (stage3.wanda_intra_expert.enabled=False)
        # already gates the WHOLE pipeline path; this provider is safe
        # to leave on for every Stage 3 run.
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(
        self, ctx: PipelineContext, jsonl_path: Path,
    ) -> WandaScalarRowPayload | None:
        """Try to load the W-1 sidecar; on hit set
        ``ctx["stage3.wanda_scalar_row"]`` and return the payload; on
        miss return ``None`` and leave ctx untouched.

        Schema mismatch surfaces as ``ValueError`` from
        ``load_wanda_scalar_row`` -- the caller MUST NOT mask that
        exception (the message is actionable: "Delete the sidecar to
        regenerate").

        ``ManifestMismatchError`` (raised by
        ``read_and_validate_manifest`` on missing-manifest /
        size-mismatch / torn-write) propagates as well: the operator
        deletes the sidecar + manifest and re-runs calibration.
        """
        payload = load_wanda_scalar_row(jsonl_path)
        if payload is None:
            return None
        ctx.set("stage3.wanda_scalar_row", payload)
        log.info(
            "stage3-wanda-scalar-row-cache: loaded %d-entry sidecar "
            "(%d layers x %d experts) from %s -- populated "
            "ctx['stage3.wanda_scalar_row']",
            len(payload.sigma_x_g_squared),
            payload.n_layers,
            payload.n_experts,
            sidecar_path(jsonl_path, "wanda_scalar_row"),
        )
        return payload
