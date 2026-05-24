"""Per-layer capacity-utilization gate: SLACK vs TIGHT cost-path selection.

Paper
-----
**No paper.** Project-original — STRATEGY_NEXT § 5 step 3 / module
identifier "M3". The gate decides which cost path runs per layer; the
baseline REAM cost (arXiv:2604.04356, see
:mod:`stage2.plugins.ream_cost`) is always-on (per Eq. 5/7/8), but
this gate selects between the cheap symmetric path
(:mod:`stage2.plugins.ream_cost`) and the expensive whitened path
(:mod:`stage2.plugins.ream_cost_post`) based on per-layer capacity
utilization.

Official code
-------------
None — the gate is project-original.

Deviation: D-capacity-util-gate
-------------------------------
Compute the per-layer capacity utilization
``u = n_NC / (n_C × C_max)`` (where ``n_NC`` is the number of
non-centroids, ``n_C = N'_l`` the centroid count, and ``C_max =
max_merge_group_size``). When ``u < capacity_util_threshold``
(default ``0.25``), the layer falls back to the cheap
``cost_alignment="pre"`` path regardless of the configured value.

Uncapped (``max_group_cap == 0``) is treated as fully slack
(``u = 0``).

The post-alignment whitened cost is expensive (per-pair Hungarian + 3
Frobenius norms × K candidates per non-centroid). At low utilization
(slack capacity), most centroids have many obvious good matches and
the heavyweight cost matrix is unlikely to change the assignment
meaningfully — gating the heavy machinery on ``u ≥ 0.25`` saves ~50 %
of the per-layer compute on GRAPE-allocated heterogeneous-budget runs.
Layers near the floor (50 % reduction → ``u ≈ 0.6-0.9``) still get
the full machinery; high-budget layers (low ``u``) skip it.

Wiring
------
``CapacityGatePlugin`` is the LIVE capacity gate. S2-10 wired it into
the ``select_alignment`` assignment slot: ``_run_assignment``
dispatches ``select_alignment`` once per bump iteration BEFORE the
``compute_cost`` slot. The gate computes ``capacity_util_value`` /
``effective_cost_alignment`` / ``effective_cost_asymmetric`` and
writes them to ``ctx``; the cost plugins' ``compute_cost`` then just
READS those slots back.

Circular-import note: this module imports only ``pipeline.base`` and
``pipeline.context`` — neither imports ``stage2_reap_ream``. There is
therefore no cycle at module load.

Naming-history note
-------------------
"M3" is the STRATEGY_NEXT label. The current plugin architecture has
no module-letter taxonomy; new prose drops the label. Existing log
lines / Trackio keys preserved for dashboard back-compat.

The gate decides which cost path runs per layer: when utilization
``u = n_NC / (N'_l × C_max)`` falls below ``capacity_util_threshold`` the layer
has slack capacity and the cheap symmetric ``"pre"`` cost is used regardless of
the configured (tighter) alignment; otherwise the configured value wins.

Circular-import note: this module imports only ``pipeline.base`` and
``pipeline.context`` — neither imports ``stage2_reap_ream``. There is therefore
no cycle at module load. ``LegacyAdapter.compute_assignment`` imports
``_pick_effective_alignment`` from *this* module (its true home), not via the
monolith re-import.

``CapacityGatePlugin`` is the LIVE capacity gate. S2-10 wired it into the
``select_alignment`` assignment slot: ``_run_assignment`` dispatches
``select_alignment`` once per bump iteration BEFORE the ``compute_cost`` slot.
The gate computes ``capacity_util_value`` / ``effective_cost_alignment`` /
``effective_cost_asymmetric`` and writes them to ctx; the cost plugins'
``compute_cost`` then just READS those slots back.
"""
from __future__ import annotations

from typing import Any

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


class CapacityGatePlugin:
    """Live plugin home for the per-layer capacity-utilization gate (M3).

    S2-10 wired the gate into the ``select_alignment`` assignment slot:
    ``_run_assignment`` dispatches ``select_alignment`` once per bump iteration
    BEFORE the ``compute_cost`` slot. ``select_alignment`` computes
    ``capacity_util_value`` / ``effective_cost_alignment`` /
    ``effective_cost_asymmetric`` and writes them to ctx; the cost plugins'
    ``compute_cost`` then just READS those slots back. The gate may downgrade a
    ``post`` / ``output``-configured run to ``pre`` per layer (slack-capacity
    layers) — that is correct and intentional.
    """

    name = "capacity_gate"
    paper = (
        "Per-layer capacity-utilization gate (project-original; no paper). "
        "Deviation D-capacity-util-gate: u = n_NC/(n_C·C_max) < threshold "
        "→ cheap `pre` cost; else configured. STRATEGY_NEXT §5 step 3 / M3. "
        "See module docstring."
    )
    config_key = "stage2_reap_ream"
    # select_alignment reads the per-bump non-centroid / centroid counts and
    # writes the three gate slots the compute_cost slot reads back.
    reads: tuple[str, ...] = ("_iter_n_ream_c", "_iter_n_ream_nc")
    writes: tuple[str, ...] = (
        "capacity_util_value", "effective_cost_alignment", "effective_cost_asymmetric",
    )
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        max_group_cap: int,
        capacity_util_threshold: float,
        cost_alignment_cfg: str,
        cost_asymmetric: bool,
    ) -> None:
        # Store every knob the gate body reads. NO logic — a faithful mirror of
        # the matching subset of LegacyAdapter.__init__ / the run() locals.
        self.max_group_cap = max_group_cap
        self.capacity_util_threshold = capacity_util_threshold
        self.cost_alignment_cfg = cost_alignment_cfg
        self.cost_asymmetric = cost_asymmetric

    def is_enabled(self, config: dict) -> bool:
        """The gate always runs (it may be a no-op downgrade, but it decides)."""
        return True

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def select_alignment(self, ctx: PipelineContext) -> Any | None:
        """Slot ``select_alignment`` — the per-layer capacity-utilization gate.

        Verbatim lift of the capacity-gate block that used to live inside
        ``ream_cost._compute_cost_for_plugin``: reads the ``_iter_n_ream_*``
        bump-loop scratch counts, computes ``capacity_util_value``, calls
        ``_pick_effective_alignment``, derives ``effective_cost_asymmetric`` and
        writes all three back to ``ctx`` (``overwrite=True``). Runs BEFORE the
        ``compute_cost`` slot so the cost plugins read these slots back.

        Returns ``effective_cost_alignment`` (a non-None string) so
        ``PluginRegistry.dispatch_first`` registers this plugin as the winner.
        """
        n_ream_c = ctx.get("_iter_n_ream_c")
        n_ream_nc = ctx.get("_iter_n_ream_nc")

        # Stage 2 v2 capacity-utilization gate (M3, spec § 5 step 3):
        #   u = n_NC / (N'_l × C_max). When u < threshold, the layer
        #   has so much slack capacity that the heavyweight
        #   post-alignment cost matrix is unlikely to change the
        #   assignment meaningfully — fall back to the cheap symmetric
        #   path. This is what skips ~half the layers' compute.
        # Capture the actual u value into the layer-scope variable
        # so the per-layer Trackio emit can surface it; mirrors the
        # division done inside _pick_effective_alignment.
        if self.max_group_cap <= 0:
            capacity_util_value = 0.0
        else:
            capacity_util_value = n_ream_nc / max(n_ream_c * self.max_group_cap, 1)
        effective_cost_alignment = _pick_effective_alignment(
            n_nc=n_ream_nc,
            n_c=n_ream_c,
            max_group_cap=self.max_group_cap,
            threshold=self.capacity_util_threshold,
            configured=self.cost_alignment_cfg,
        )
        effective_cost_asymmetric = (
            self.cost_asymmetric and effective_cost_alignment == "post"
        )
        ctx.set("capacity_util_value", capacity_util_value, overwrite=True)
        ctx.set("effective_cost_alignment", effective_cost_alignment, overwrite=True)
        ctx.set("effective_cost_asymmetric", effective_cost_asymmetric, overwrite=True)
        return effective_cost_alignment
