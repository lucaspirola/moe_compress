"""Skip-merge floor plugin (Task 12 of the plugin-architecture refactor).

Plugin home for Direction B — the skip-merge percentile mask. The mask logic
itself (``_apply_skip_merge_floor``) was extracted into ``pipeline.grouping`` in
Task 5 and stays there; T12 does not move it. This module adds the
``SkipMergeFloorPlugin`` that *owns the live ``apply_cost_mask`` call site* in
the decomposed phase walk.

Wiring status (S2-7): this plugin is now LIVE. ``run()`` registers it in the
``PluginRegistry`` after the three cost plugins and before the ``LegacyAdapter``,
so when it is enabled (``skip_merge_percentile < 100.0``) it wins the
``apply_cost_mask`` ``dispatch_first`` slot in ``_run_assignment``'s bump loop.
``registry.enabled(config)`` drops it at the OFF sentinel (``>= 100.0``), so
``dispatch_first`` then reaches ``LegacyAdapter.apply_cost_mask`` — its sentinel
branch returns the delta object unchanged. The monolith's ``_em_refine_assignment``
still re-applies the floor each EM round; S2-7 does not change that call site.

Config gate: enabled iff ``stage2_reap_ream.skip_merge_percentile < 100.0``.
``100.0`` is the OFF sentinel (the 100th percentile equals the max finite cost,
so nothing is strictly above it). A missing key defaults to ``100.0`` → OFF.

Circular-import note: this module imports only ``pipeline.base``,
``pipeline.context`` and ``pipeline.grouping`` — none of which import
``stage2_reap_ream``. No cycle at module load.
"""
from __future__ import annotations

import logging
from typing import Any

from ...pipeline.context import PipelineContext
from ..grouping import _apply_skip_merge_floor

log = logging.getLogger(__name__)

# OFF sentinel: percentile >= this value masks nothing (see grouping docstring).
_SKIP_MERGE_OFF = 100.0


class SkipMergeFloorPlugin:
    """Plugin home for the Direction B skip-merge percentile mask.

    S2-7 status: LIVE. ``run()`` registers this plugin after the three cost
    plugins and before the ``LegacyAdapter``; when enabled
    (``skip_merge_percentile < 100.0``) it wins the ``apply_cost_mask``
    ``dispatch_first`` slot in ``_run_assignment``'s bump loop, servicing the
    skip-merge floor that ``LegacyAdapter.apply_cost_mask`` used to own. At the
    OFF sentinel ``registry.enabled`` drops it and the LegacyAdapter's sentinel
    branch services the slot instead. ``apply_cost_mask`` delegates to
    ``grouping._apply_skip_merge_floor`` — byte-identical to the LegacyAdapter
    block it replaces.

    The percentile is stored at construction: ``apply_cost_mask`` only receives
    a ``PipelineContext`` (which carries no cfg), so the value cannot be read at
    call time. Use :func:`make_skip_merge_floor_plugin` to build the plugin
    from a config dict.
    """

    name = "skip_merge_floor"
    paper = "Direction B: skip-merge percentile mask on the cost matrix."
    config_key = "stage2_reap_ream.skip_merge_percentile"
    # S2-7: the live ``apply_cost_mask`` slot reads ``layer_ref`` for the
    # INFO log line emitted when entries are masked.
    reads: tuple[str, ...] = ("layer_ref",)
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def __init__(self, skip_merge_percentile: float = _SKIP_MERGE_OFF) -> None:
        """Store the percentile for ``apply_cost_mask``.

        Defaults to the OFF sentinel so a bare ``SkipMergeFloorPlugin()`` is a
        no-op masker (matches the config default).
        """
        self.skip_merge_percentile = float(skip_merge_percentile)

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.skip_merge_percentile`` is < 100.0.

        ``100.0`` (the default) and any missing key → False (OFF). Values
        above 100.0 are also OFF (nothing can sit strictly above a >100th
        percentile clamp). Only a value strictly below 100.0 turns it on.
        """
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        return float(s2.get("skip_merge_percentile", _SKIP_MERGE_OFF)) < _SKIP_MERGE_OFF

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def apply_cost_mask(
        self, ctx: PipelineContext, delta: Any
    ) -> tuple[Any, dict] | None:
        """Mask cost-matrix entries above the skip-merge percentile to +inf.

        Delegates to ``grouping._apply_skip_merge_floor``. At the OFF sentinel
        (percentile >= 100.0) returns the input array unchanged (no copy) to
        match the LegacyAdapter live-path semantics, which skip the helper
        entirely at the sentinel.

        Returns ``(masked_delta, info)`` with ``info = {"n_masked": int,
        "percentile": float}``. Byte-identical to
        ``LegacyAdapter.apply_cost_mask``, including the INFO log line emitted
        when entries are masked. ``ctx`` carries ``layer_ref`` for that log; it
        is ``None`` in unit tests, so the log is guarded on ``ctx is not None``.
        """
        if self.skip_merge_percentile >= _SKIP_MERGE_OFF:
            return delta, {"n_masked": 0, "percentile": self.skip_merge_percentile}
        masked, n_masked = _apply_skip_merge_floor(delta, self.skip_merge_percentile)
        if n_masked > 0 and ctx is not None:
            layer_ref = ctx.get("layer_ref")
            log.info(
                "layer %d: skip-merge floor (P%.1f) masked %d/%d "
                "cost entries to +inf — affected children fall "
                "through to orphan promotion",
                layer_ref.layer_idx, self.skip_merge_percentile,
                n_masked, masked.size,
            )
        return masked, {"n_masked": n_masked, "percentile": self.skip_merge_percentile}


def make_skip_merge_floor_plugin(cfg: dict[str, Any]) -> SkipMergeFloorPlugin:
    """Construct a :class:`SkipMergeFloorPlugin` from a Stage 2 config dict.

    Reads ``cfg["stage2_reap_ream"]["skip_merge_percentile"]`` (default
    ``100.0``). Mirrors how ``stage2_reap_ream.run()`` will wire the plugin
    once ``compute_assignment`` is decomposed (T13+).
    """
    s2 = cfg.get("stage2_reap_ream", {}) if isinstance(cfg, dict) else {}
    return SkipMergeFloorPlugin(
        skip_merge_percentile=float(s2.get("skip_merge_percentile", _SKIP_MERGE_OFF))
    )
