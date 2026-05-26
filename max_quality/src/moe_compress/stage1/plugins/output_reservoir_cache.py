"""Stage 1 cache provider for per-(layer, expert) expert-output reservoirs.

Reads a pre-computed ``output_reservoir`` payload from a sidecar
produced by the ``--capture-output-reservoir`` calibration flag
(Item 6 of the calibration-v2 writers campaign, in
``vllm.calibration_output_reservoir``). On cache hit, constructs a
pre-finalized :class:`ExpertOutputAccumulator` whose ``_finalized``
dict is populated directly from the cached reservoir tensor (sliced
to ``valid_count`` per cell), then deposits it on
``ctx["output_acc"]`` -- the SAME slot the orchestrator's STEP 5
otherwise writes from the live Phase-B accumulator.

Architecture: provider-pair pattern per
``max_quality/docs/calibration_v2_data_capture_plan.md`` Section 0.
The Stage 1 orchestrator's STEP 4.8 instantiates this provider directly
and, on cache hit, drops ``"output_reservoir"`` from the ``needed``
set so the live ``ExpertOutputAccumulator`` factory is NOT invoked
and the Phase-B forward pass skips per-expert reservoir sampling
entirely. STEP 7's ``built["output_reservoir"].finalize()`` is then
also skipped because the cache reader pre-populates the same
finalize-target dict.

Indexing convention: the cached tensor's rows are in the same order
as ``iter_moe_layers`` (writer + reader share the
``named_modules() -> moe_layer_id`` discovery ordering used by every
calibration-v2 writer). The ``moe_layers`` list pulled from ctx
provides the rank -> layer_idx mapping for the accumulator's
dict-keyed output.

Zero-valid-count exclusion: cells with ``valid_count == 0`` (no
tokens ever routed to that (layer, expert) during the calibration
run) are NOT inserted into the accumulator's ``_finalized`` dict.
This matches the live accumulator's absent-key convention:
``ExpertOutputAccumulator.get_representations(li, e)`` returns
``None`` for an unseen expert, and consumers (CKADistancePlugin,
ablation_filter) handle the ``None`` branch explicitly.

Manifest-exclusion note: this plugin is intentionally NOT in
``STAGE1_PLUGIN_MANIFEST`` (see ``stage1/plugins/__init__.py``). It is
instantiated directly by ``stage1/orchestrator.py`` at STEP 4.8 because
Stage 1's live output-reservoir path is the
``ExpertOutputAccumulator`` factory + the calibration-pass ``HookSpec``,
not a registered provider plugin -- there is no canonical "live
provider" for ``PluginRegistry.dispatch_first`` to fall through to.
Future refactors that promote the calibration pass to a proper
plugin-pair should also register this cache provider in the manifest.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from ...pipeline.context import PipelineContext
from ...utils.activation_hooks import ExpertOutputAccumulator
from ...utils.cached_calibration_signals import (
    BaseCacheProvider,
    OutputReservoirPayload,
    load_output_reservoir,
    sidecar_path,
)

log = logging.getLogger(__name__)


class Stage1OutputReservoirCacheProvider(BaseCacheProvider):
    """Cache-side provider for Stage 1's per-(layer, expert)
    expert-output reservoir accumulator.

    On hit, builds an :class:`ExpertOutputAccumulator` whose
    ``_finalized`` dict is hydrated from the cached reservoir tensor
    (sliced to ``valid_count`` per cell), sets it on ``ctx.output_acc``
    (overwriting whatever the live factory placed there), and returns
    the loaded payload so the orchestrator's hit-check sees a non-None
    winner. On miss, returns ``None`` and leaves ``ctx`` untouched so
    the live ``ExpertOutputAccumulator`` path runs unchanged.
    """

    name: str = "output_reservoir_cache"
    paper: str = (
        "Cache provider for per-(layer, expert) expert-output reservoir. "
        "Reads sidecars/output_reservoir.pt produced by "
        "--capture-output-reservoir (Item 6 of the calibration-v2 "
        "writers campaign). On hit: hydrates a pre-finalized "
        "ExpertOutputAccumulator into ctx.output_acc; the live Phase-B "
        "output_reservoir registration is dropped. On miss: returns "
        "None; live reservoir-sample pass runs unchanged."
    )
    config_key: str = "calibration.output_reservoir_cache"
    reads: tuple[str, ...] = ("moe_layers",)
    writes: tuple[str, ...] = ("output_acc",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        # Always enabled: cache provider is a no-op on miss.
        return True

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        return {}

    def on_load(
        self, ctx: PipelineContext, jsonl_path: Path,
    ) -> OutputReservoirPayload | None:
        """Try to load the sidecar; on hit, populate ``ctx.output_acc``."""
        payload = load_output_reservoir(jsonl_path)
        if payload is None:
            return None

        # Topology consistency: the sidecar was written for a specific
        # MoE-layer count; refuse to hydrate the accumulator if the live
        # model disagrees (mirrors the per_expert_max / router_logits_stats
        # cache provider precedent).
        moe_layers = ctx.get("moe_layers")
        n_layers = len(moe_layers)
        if n_layers != payload.n_layers:
            raise ValueError(
                f"output_reservoir cache mismatch: sidecar reports "
                f"n_layers={payload.n_layers} but the live model has "
                f"{n_layers} MoE layers. The sidecar was produced for a "
                f"different model topology -- delete it to regenerate."
            )

        # Build the pre-finalized accumulator. ``max_tokens_per_expert``
        # tracks the writer's cap so any downstream code that reads the
        # field sees the same value the writer used.
        acc = ExpertOutputAccumulator(
            max_tokens_per_expert=int(payload.max_tokens),
        )
        n_inserted = 0
        for rank in range(n_layers):
            layer_idx = int(moe_layers[rank].layer_idx)
            for expert_id in range(payload.n_experts):
                n_valid = int(payload.valid_count[rank, expert_id].item())
                if n_valid <= 0:
                    # Zero-traffic cell. Match the live accumulator's
                    # absent-key convention: get_representations returns
                    # None on a missing key, so omit instead of inserting
                    # an empty tensor.
                    continue
                # Slice [n_valid, hidden] from the dense reservoir, cast
                # back to fp32 to match the live ``_finalized`` dtype
                # contract (the writer stored bf16; the live accumulator
                # writes fp32 in finalize()).
                slab = payload.reservoir[rank, expert_id, :n_valid]
                acc._finalized[(layer_idx, expert_id)] = slab.to(
                    torch.float32,
                ).clone()
                n_inserted += 1

        # Overwrite any prior ctx.output_acc -- the orchestrator's STEP 5
        # publishes the live-factory accumulator unconditionally; on
        # cache hit we WANT the cached accumulator to win. The STEP 4.8
        # caller drops "output_reservoir" from ``needed`` so the live
        # factory is not invoked, but we use overwrite=True defensively
        # in case the orchestration order is reshuffled later.
        ctx.set("output_acc", acc, overwrite=True)
        log.info(
            "output-reservoir-cache: loaded %d-layer x %d-expert sidecar "
            "from %s -- populated ctx.output_acc with %d non-zero cells "
            "(cap=%d)",
            payload.n_layers, payload.n_experts,
            sidecar_path(jsonl_path, "output_reservoir"),
            n_inserted, payload.max_tokens,
        )
        return payload
