"""Capacity-utilization gate plugin (Task 11 of the plugin-architecture refactor).

Home of ``_pick_effective_alignment`` — the per-layer capacity-utilization gate
(Stage 2 v2 spec § 5 step 3 / M3). Moved verbatim out of ``stage2_reap_ream.py``;
that module re-imports it so external callers and tests keep their existing
import paths.

The gate decides which cost path runs per layer: when utilization
``u = n_NC / (N'_l × C_max)`` falls below ``capacity_util_threshold`` the layer
has slack capacity and the cheap symmetric ``"pre"`` cost is used regardless of
the configured (tighter) alignment; otherwise the configured value wins.

Circular-import note: this module imports only ``pipeline.base`` and
``pipeline.context`` — neither imports ``stage2_reap_ream``. There is therefore
no cycle at module load. ``LegacyAdapter.compute_assignment`` imports
``_pick_effective_alignment`` from *this* module (its true home), not via the
monolith re-import.

``CapacityGatePlugin`` is the future plugin home for the gate. For T11 it is an
inert shell: ``compute_cost`` is a documented no-op because the legacy bump
loop (``LegacyAdapter.compute_assignment``) still calls
``_pick_effective_alignment`` directly. Wiring the gate into the phase walk is
deferred until the assignment phase is decomposed (T13+).
"""
from __future__ import annotations

from typing import Any

from .._framework.base import Stage2Plugin
from ...pipeline.context import PipelineContext


def _pick_effective_alignment(
    *,
    n_nc: int,
    n_c: int,
    max_group_cap: int,
    threshold: float,
    configured: str,
) -> str:
    """Decide SLACK vs TIGHT for the cost-matrix path (spec § 5 step 3 / M3).

    Capacity-utilization gate:
        u = n_NC / (N'_l × C_max).
    When ``u < threshold`` the layer has so much slack capacity that the
    heavyweight cost matrix is unlikely to change the assignment meaningfully
    — return ``"pre"`` regardless of the configured value.  Otherwise return
    the configured value (``"pre"``, ``"post"``, or Direction C's
    ``"output"``). The output-space cost is heavyweight too, so it is gated
    identically to ``"post"`` (downgraded to ``"pre"`` on slack layers).

    With ``max_group_cap == 0`` (uncapped, ablation-only path) we treat the
    layer as fully slack (u = 0).
    """
    if max_group_cap <= 0:
        util = 0.0
    else:
        capacity = max(n_c * max_group_cap, 1)
        util = n_nc / capacity
    if util < threshold:
        return "pre"
    return configured


class CapacityGatePlugin(Stage2Plugin):
    """Plugin home for the per-layer capacity-utilization gate (M3).

    T11 status: inert shell. ``LegacyAdapter.compute_assignment`` still calls
    ``_pick_effective_alignment`` directly inside the bump loop, so this
    plugin's ``compute_cost`` hook is a deliberate no-op. The plugin exists now
    so the gate has a stable home; wiring it into the phase walk is deferred
    until the assignment phase is decomposed (T13+).
    """

    name = "capacity_gate"
    # The gate is not a boolean-flag opt-in: it always runs (it may be a no-op
    # downgrade, but it always *decides*). enabled_by stays empty so the base
    # AND-of-flags is_enabled returns True for every config.
    enabled_by: tuple[str, ...] = ()

    def compute_cost(self, ctx: PipelineContext) -> Any | None:
        """No-op for T11. See class docstring.

        Returning ``None`` makes ``PluginRegistry.dispatch_first`` skip this
        plugin so the legacy bump loop remains the sole cost-matrix producer.
        """
        return None
