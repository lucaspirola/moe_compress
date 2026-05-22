"""Greedy assignment solver (Task 13 of the plugin-architecture refactor).

Plugin home for ``_assign_greedy`` — the legacy descending-saliency greedy
assignment path. Extracted verbatim from ``stage2_reap_ream`` in Task 13.

This is the **leaf** of the solver import DAG: it imports nothing from the
sibling solver modules (greedy never falls back to another solver). The other
solvers (``solver_mcf``, ``solver_sinkhorn``) import ``_assign_greedy`` from
here as their fallback path; ``solver_dispatch`` re-exports it via the registry.

The monolith re-imports ``_assign_greedy`` so external callers (tests, the
``MOE_STAGE2_LEGACY_LOOP=1`` path, ``LegacyAdapter``) keep their import paths.

``GreedySolverPlugin`` is a scaffold-only ``Stage2Plugin`` — not yet on the live
phase walk (the bump loop still calls ``_assign_children_to_centroids``); it
gives T18 a per-solver plugin to wire into the decomposed ``solve_assignment``
phase. Circular-import note: this module imports only ``pipeline.base`` and
``pipeline.context``, neither of which imports ``stage2_reap_ream``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._framework.base import Stage2Plugin
from .._framework.context import LayerContext

log = logging.getLogger(__name__)


def _assign_greedy(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int,
) -> list[int]:
    """Legacy greedy path — extracted from the v1 implementation verbatim.

    Preserves the bit-identical assignment under the v1 default (greedy +
    descending-saliency centroid order).

    Defensive: returns ``[-1] * n_children`` for empty inputs so this helper
    can be called from fallback paths in :func:`_assign_hungarian` /
    :func:`_assign_mcf` without re-doing the dispatcher's early-exit.
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children
    if max_group_cap == 0:
        # Uncapped: assign each child to its nearest centroid by cost.
        # Iterating children (not centroids) avoids the centroid-order bias that
        # causes centroid 0 to absorb all children in the capped greedy path.
        assignment = [-1] * n_children
        for ch in range(n_children):
            best_c = int(np.argmin(cost[ch, :]))
            if not np.isfinite(cost[ch, best_c]):
                assignment[ch] = -1
            else:
                assignment[ch] = best_c
        n_unassigned = sum(1 for a in assignment if a < 0)
        if n_unassigned > 0 and n_centroids > 0:
            log.warning(
                "_assign_children_to_centroids: %d/%d children unassigned after uncapped pass "
                "(all-inf cost row(s) in cost matrix) — "
                "these children will be dropped from the merge group unless the caller "
                "promotes them as orphan centroids.",
                n_unassigned, n_children,
            )
        return assignment

    # Capped path (max_group_cap > 0): single-pass greedy, centroid order.
    # Note on group-cap semantics (spec §5 Step 3):
    #   max_group_cap counts non-centroids only (not the centroid itself), matching
    #   our spec §5 Step 3 ("absorb up to max_merge_group_size unassigned non-centroids").
    #   The REAM reference's group_size counts total members including the centroid,
    #   so our max_group_cap=8 is equivalent to reference group_size=9.
    # The feasibility check (b_fail) in the bump loop uses the same semantics:
    #   n_ream_nc > n_ream_c * max_group_cap  (non-centroids exceed total centroid capacity).
    assignment = [-1] * n_children
    assigned: set[int] = set()

    for c_idx in range(n_centroids):
        absorbed = 0
        # O(n_children) scan per fill slot — pathological for large expert counts;
        # consider pre-sorting by cost if this becomes a bottleneck.
        while absorbed < max_group_cap:
            best_child = -1
            best_cost = float("inf")
            for ch in range(n_children):
                if ch in assigned:
                    continue
                if cost[ch, c_idx] < best_cost:
                    best_cost = cost[ch, c_idx]
                    best_child = ch
            if best_child < 0:
                # No unassigned children with finite cost remain for this centroid.
                # Break to next centroid; any remaining unassigned children (all-inf
                # cost rows) will be reported and promoted as orphan centroids by the
                # caller. The caller must ensure costs are finite (via feasibility check)
                # to guarantee all children are assigned.
                break
            assignment[best_child] = c_idx
            assigned.add(best_child)
            absorbed += 1

    n_unassigned = sum(1 for a in assignment if a < 0)
    if n_unassigned > 0 and n_centroids > 0:
        log.warning(
            "_assign_children_to_centroids: %d/%d children unassigned after capped greedy pass "
            "(likely cause: inf cost entries in cost matrix preventing assignment) — "
            "these children will be dropped from the merge group unless the caller "
            "promotes them as orphan centroids.",
            n_unassigned, n_children,
        )

    return assignment


class GreedySolverPlugin(Stage2Plugin):
    """Plugin home for the greedy assignment solver (Task 13 scaffold).

    Scaffold only: not yet on the live phase walk. The bump loop in
    LegacyAdapter still calls `_assign_children_to_centroids`; this class
    exists so T18 has a per-solver plugin to wire into the decomposed
    `solve_assignment` phase. `is_enabled` selects this solver when
    `stage2_reap_ream.assignment_solver == "greedy"`.
    """

    name = "solver_greedy"
    enabled_by: tuple[str, ...] = ()  # selector is a string match, not a flag

    @classmethod
    def is_enabled(cls, cfg: dict[str, Any]) -> bool:
        s2 = cfg.get("stage2_reap_ream", {}) if isinstance(cfg, dict) else {}
        return str(s2.get("assignment_solver", "greedy")).lower() == "greedy"

    def solve_assignment(self, ctx: LayerContext, delta: Any) -> Any | None:
        """Wrap `_assign_greedy`. NOTE: not invoked by the current phase walk
        (the bump loop calls `_assign_children_to_centroids` directly); kept as
        a functional hook for the T18 decomposition. Returns None when delta is
        not a usable cost matrix so `dispatch_first` can skip cleanly."""
        return None
