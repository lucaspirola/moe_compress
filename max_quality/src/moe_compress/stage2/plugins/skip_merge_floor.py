"""Skip-merge floor plugin (Task 12 of the plugin-architecture refactor).

Plugin home for Direction B — the skip-merge percentile mask. The mask logic
itself (``_apply_skip_merge_floor``) was extracted into ``pipeline.grouping`` in
Task 5 and stays there; T12 does not move it. This module only adds the
``SkipMergeFloorPlugin`` shell that *owns the call site* in the eventual
decomposed phase walk.

Wiring status (T12): the live call path still belongs to the LegacyAdapter.
``LegacyAdapter.compute_assignment`` calls ``grouping._apply_skip_merge_floor``
directly inside the bump loop, and the monolith's ``_em_refine_assignment``
re-applies it each EM round. T12 does NOT change those call sites. The
``apply_cost_mask`` hook below is fully functional (it delegates to
``_apply_skip_merge_floor``) so the plugin is testable today, but the
``Stage2Pipeline`` phase walk does not yet invoke ``apply_cost_mask`` — that
invocation arrives when ``compute_assignment`` is decomposed (T13+). Until then
this plugin is constructed but not added to the ``run()`` plugin list, exactly
like the T8–T11 cost shells.

Config gate: enabled iff ``stage2_reap_ream.skip_merge_percentile < 100.0``.
``100.0`` is the OFF sentinel (the 100th percentile equals the max finite cost,
so nothing is strictly above it). A missing key defaults to ``100.0`` → OFF.

Circular-import note: this module imports only ``pipeline.base``,
``pipeline.context`` and ``pipeline.grouping`` — none of which import
``stage2_reap_ream``. No cycle at module load.
"""
from __future__ import annotations

from typing import Any

from .._framework.base import Stage2Plugin
from .._framework.context import LayerContext
from ..grouping import _apply_skip_merge_floor

# OFF sentinel: percentile >= this value masks nothing (see grouping docstring).
_SKIP_MERGE_OFF = 100.0


class SkipMergeFloorPlugin(Stage2Plugin):
    """Plugin home for the Direction B skip-merge percentile mask.

    T12 status: functional but off the live phase walk. ``apply_cost_mask``
    delegates to ``grouping._apply_skip_merge_floor`` and is unit-tested, but
    the ``Stage2Pipeline`` does not yet call ``apply_cost_mask`` — the
    LegacyAdapter still owns the live call path inside its bump loop. Wiring
    this hook into the phase walk is deferred until the assignment phase is
    decomposed (T13+).

    The percentile is stored at construction: ``apply_cost_mask`` only receives
    a ``LayerContext`` (which carries no cfg), so the value cannot be read at
    call time. Use :func:`make_skip_merge_floor_plugin` to build the plugin
    from a config dict.
    """

    name = "skip_merge_floor"
    # Not a boolean-flag opt-in: the gate is a numeric threshold, so the base
    # AND-of-flags is_enabled cannot express "skip_merge_percentile < 100.0".
    # enabled_by stays empty and is_enabled is overridden below.
    enabled_by: tuple[str, ...] = ()

    def __init__(self, skip_merge_percentile: float = _SKIP_MERGE_OFF) -> None:
        """Store the percentile for ``apply_cost_mask``.

        Defaults to the OFF sentinel so a bare ``SkipMergeFloorPlugin()`` is a
        no-op masker (matches the config default).
        """
        self.skip_merge_percentile = float(skip_merge_percentile)

    @classmethod
    def is_enabled(cls, cfg: dict[str, Any]) -> bool:
        """True iff ``stage2_reap_ream.skip_merge_percentile`` is < 100.0.

        ``100.0`` (the default) and any missing key → False (OFF). Values
        above 100.0 are also OFF (nothing can sit strictly above a >100th
        percentile clamp). Only a value strictly below 100.0 turns it on.
        """
        s2 = cfg.get("stage2_reap_ream", {}) if isinstance(cfg, dict) else {}
        return float(s2.get("skip_merge_percentile", _SKIP_MERGE_OFF)) < _SKIP_MERGE_OFF

    def apply_cost_mask(
        self, ctx: LayerContext, delta: Any
    ) -> tuple[Any, dict] | None:
        """Mask cost-matrix entries above the skip-merge percentile to +inf.

        Delegates to ``grouping._apply_skip_merge_floor``. At the OFF sentinel
        (percentile >= 100.0) returns the input array unchanged (no copy) to
        match the LegacyAdapter live-path semantics, which skip the helper
        entirely at the sentinel.

        Returns ``(masked_delta, info)`` with ``info = {"n_masked": int,
        "percentile": float}``. ``ctx`` is unused (the percentile is an
        instance field) and accepted only to satisfy the hook signature.
        """
        if self.skip_merge_percentile >= _SKIP_MERGE_OFF:
            return delta, {"n_masked": 0, "percentile": self.skip_merge_percentile}
        masked, n_masked = _apply_skip_merge_floor(delta, self.skip_merge_percentile)
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
