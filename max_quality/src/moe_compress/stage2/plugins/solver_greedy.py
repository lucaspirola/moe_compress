"""REAM paper-faithful greedy assignment (descending-saliency single-pass).

Paper
-----
Liu et al., "REAM" — arXiv:2604.04356, §4.
audit/spec_compliance/01_papers/2604.04356/source.md.

Step 3 of the REAM pipeline (per project §5 Step 3): top-``N'_l``
experts by REAP score become **centroids**. Non-centroids are assigned
to centroids via a **single-pass greedy algorithm**:

  1. Iterate centroids in **descending saliency order** (most salient
     first — order is important).
  2. For each centroid, absorb up to ``max_merge_group_size`` unassigned
     non-centroids with the **lowest cost** (most similar), in order.
  3. The loop exits early once all non-centroids are assigned.

Every non-centroid is guaranteed to be assigned — the feasibility
check (see :mod:`stage2.plugins.layer_merge`, D-ream-budget-bump)
ensures full coverage.

Reference: ``ream/ream.py`` lines 63-87 in the upstream
``SamsungSAILMontreal/ream`` repository, pinned at commit
``84a3030716a0059589e9d10e2ea049e32b76cfa6`` (2026-04-16).

Official code
-------------
``SamsungSAILMontreal/ream`` @ ``84a3030716a0059589e9d10e2ea049e32b76cfa6``,
``ream/ream.py:63-87``. The plugin's ``_assign_greedy`` is a verbatim
re-implementation of that loop in NumPy.

Deviations
----------
None — this branch is paper-faithful. The alternative solvers (Hungarian,
MCF, Sinkhorn) implement deviation D-mcf-assignment /
D-sinkhorn-soft-assign; see their respective modules.

The default ``assignment_solver`` is ``"greedy"``, reproducing the
v1 baseline byte-identically.

Output context contract
-----------------------
Pure callable plugin (no state). Returns the layer's assignment from
the input cost matrix + scores + frequencies.

Naming-history note
-------------------
Step 3 of the REAM pipeline (project §5 Step 3). The current plugin
architecture has no step-numbering taxonomy; new prose drops the
labels. Existing log lines / Trackio keys preserved.

This is the **leaf** of the solver import DAG: it imports nothing from the
sibling solver modules (greedy never falls back to another solver). The other
solvers (``solver_mcf``, ``solver_sinkhorn``) import ``_assign_greedy`` from
here as their fallback path; ``solver_dispatch`` re-exports it via the registry.

The monolith re-imports ``_assign_greedy`` so external callers (tests, the
``MOE_STAGE2_LEGACY_LOOP=1`` path, ``LegacyAdapter``) keep their import paths.

``GreedySolverPlugin`` is a scaffold-only plugin — not yet on the live
phase walk (the bump loop still calls ``_assign_children_to_centroids``); it
gives T18 a per-solver plugin to wire into the decomposed ``solve_assignment``
phase. Circular-import note: this module imports only ``pipeline.base`` and
``pipeline.context``, neither of which imports ``stage2_reap_ream``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...pipeline.context import PipelineContext

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


class GreedySolverPlugin:
    """Plugin home for the greedy assignment solver.

    LIVE (S2-8): services the solve_assignment slot when assignment_solver
    selects this solver. `is_enabled` selects this solver when
    `stage2_reap_ream.assignment_solver == "greedy"`.
    """

    name = "solver_greedy"
    paper = (
        "REAM §4 descending-saliency single-pass greedy assignment — "
        "arXiv:2604.04356 (Liu et al.). Paper-faithful (no deviations). "
        "Official code: SamsungSAILMontreal/ream @ "
        "84a3030716a0059589e9d10e2ea049e32b76cfa6 (ream/ream.py L63-87). "
        "Default assignment_solver; alternative solvers implement "
        "D-mcf-assignment / D-sinkhorn-soft-assign."
    )
    config_key = "stage2_reap_ream.assignment_solver"
    # S2-8: the live solve_assignment slot reads the per-bump scratch slots.
    reads: tuple[str, ...] = ("_iter_n_ream_nc", "_iter_n_ream_c")
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        max_group_cap: int = 0,
        assignment_solver: str = "greedy",
        sinkhorn_epsilon_init: float = 1.0,
        sinkhorn_epsilon_final: float = 0.01,
        sinkhorn_iters: int = 200,
    ) -> None:
        self.max_group_cap = max_group_cap
        self.assignment_solver = assignment_solver
        self.sinkhorn_epsilon_init = sinkhorn_epsilon_init
        self.sinkhorn_epsilon_final = sinkhorn_epsilon_final
        self.sinkhorn_iters = sinkhorn_iters

    def is_enabled(self, config: dict) -> bool:
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        return str(s2.get("assignment_solver", "greedy")).lower() == "greedy"

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def solve_assignment(self, ctx: PipelineContext, delta: Any) -> Any | None:
        """Slot ``solve_assignment`` — child→centroid assignment solver.

        Delegates to the shared ``_solve_for_plugin`` helper (verbatim lift of
        ``LegacyAdapter.solve_assignment``). Reaches this plugin only when
        ``registry.enabled`` kept it, i.e. ``assignment_solver == "greedy"``."""
        from .solver_dispatch import _solve_for_plugin
        return _solve_for_plugin(self, ctx, delta)
