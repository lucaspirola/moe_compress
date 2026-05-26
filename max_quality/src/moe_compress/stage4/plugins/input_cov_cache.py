"""Stage 4 cache provider for per-(layer, expert, matrix) input covariance.

Reads a pre-computed Σ_in dict from a sidecar produced by the
``--capture-input-covariance`` calibration flag (V2 writer in
``vllm.calibration_input_cov``). On cache hit, populates ``ctx.A_cov``
so the live :class:`EoraInputsPlugin.load_eora_inputs` hook's
``ctx.has("A_cov")`` short-circuit skips its on-disk
``_stage2_input_covariance.pt`` load.

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0. The
on-disk dict shape (keys = ``(layer_idx, expert_idx, matrix_name)``,
values = ``Tensor[d_in, d_in]`` fp16) is identical to the Stage 2
writer's so EoRA's ``_compute_eora_factors`` consumes both
interchangeably.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ...pipeline.context import PipelineContext
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    CovariancePayload,
    load_covariance,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage4InputCovCacheProvider(BaseCacheProvider):
    """Cache-side provider for the Stage 4 input-covariance sidecar.

    On hit, populates ``ctx.A_cov`` (and ``ctx.a_storage_dtype``) and
    returns the loaded ``CovariancePayload``. On miss, returns None so
    the registry's ``dispatch_first`` falls through to the live
    ``EoraInputsPlugin.load_eora_inputs`` which loads from
    ``_stage2_input_covariance.pt``. The live plugin starts with a
    ``ctx.has("A_cov")`` guard so a cache hit short-circuits the
    on-disk load.
    """

    name: str = "stage4_input_cov_cache"
    paper: str = (
        "Cache provider for the V2 input-covariance writer "
        "(calibration-v2 Item 1). Reads sidecars/covariance.pt and "
        "populates ctx.A_cov so EoraInputsPlugin's ctx.has guard "
        "short-circuits its _stage2_input_covariance.pt load."
    )
    # Informational only: this provider is always-enabled (cache is a
    # no-op on miss). The key names the actual driver knob the provider
    # depends on -- the calibration JSONL path that locates the sidecar.
    config_key: str = "calibration.input_covariance_cache"
    reads: tuple[str, ...] = ()
    # NOTE: the ``A_cov`` + ``a_storage_dtype`` slots are ALSO listed in
    # ``EoraInputsPlugin.writes``. The overlap is INTENTIONAL: the
    # orchestrator dispatches this provider's ``on_load`` BEFORE the live
    # plugin's ``load_eora_inputs`` hook, and the live plugin's
    # ``ctx.has("A_cov")`` guard short-circuits on a cache hit (with
    # ``overwrite=_cache_hit`` so the live plugin's own ctx.set is a
    # no-op when the cache populated the slots). A future contract-
    # linting tool that flags write collisions should treat this pair as
    # an allowed alias rather than a bug.
    writes: tuple[str, ...] = ("A_cov", "a_storage_dtype")
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(self, ctx: PipelineContext,
                jsonl_path: Path) -> CovariancePayload | None:
        """Try to load the sidecar; populate ctx + return payload on hit."""
        payload = load_covariance(jsonl_path)
        if payload is None:
            return None
        ctx.set("A_cov", payload.sigma_in)
        # The cached tensors are persisted as fp16 (per the Stage 2
        # writer's storage dtype convention, shared by the V2 writer).
        # Setting ``a_storage_dtype`` so the EoRA eigh-threshold tuning
        # in ``_compute_eora_factors`` picks the right noise floor.
        ctx.set("a_storage_dtype", torch.float16)
        log.info(
            "stage4-input-cov-cache: loaded %d-key sidecar (%d layers × "
            "%d experts) from %s -- populated ctx.A_cov + "
            "ctx.a_storage_dtype",
            len(payload.sigma_in), payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "covariance"),
        )
        return payload
