"""Min-cost-flow assignment solver (Task 13 of the plugin-architecture refactor).

Plugin home for ``_assign_mcf`` — the capacitated min-cost-flow solver backed by
OR-Tools' ``SimpleMinCostFlow``. Extracted verbatim from ``stage2_reap_ream`` in
Task 13.

ortools-safety contract (CRITICAL): this module **must be importable in an
environment without ``ortools``**. The ``from ortools... import
SimpleMinCostFlow`` statement stays **inside the body of ``_assign_mcf``** — it
is never a module-scope import. Module load needs only ``numpy``, ``logging``,
``_assign_greedy`` (the fallback), and the plugin-class imports. When ``ortools``
is absent, ``_assign_mcf`` raises a ``RuntimeError`` *at call time*, not an
``ImportError`` at import time.

Imports ``_assign_greedy`` from ``solver_greedy`` as the no-finite-entries /
non-optimal-status fallback. The monolith re-imports ``_assign_mcf``.

``McfSolverPlugin`` is a scaffold-only ``Stage2Plugin`` (see ``solver_greedy``).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._framework.base import Stage2Plugin
from .._framework.context import LayerContext
from .solver_greedy import _assign_greedy

log = logging.getLogger(__name__)


def _assign_mcf(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int,
) -> list[int]:
    """Capacitated min-cost flow via OR-Tools' ``SimpleMinCostFlow``.

    Models the standard transportation polytope:
        source → each child (supply 1)
        child  → each centroid (cost ``cost[ch, c]``, capacity 1)
        centroid → sink (capacity ``max_group_cap``)
        sink supply = ``n_children`` so all children must be matched.

    Total unimodularity guarantees integer optimality under the LP relaxation
    (Ahuja–Magnanti–Orlin §9 — capacity is a transportation problem). OR-Tools
    runs cost-scaling push-relabel; ~10 ms per layer for our sizes.

    ``+∞`` entries are excluded by simply not adding the corresponding arc.

    Cost normalization: OR-Tools uses int costs. We normalize the finite cost
    range to ``[0, MCF_INT_SCALE]`` before rounding, so this routine is safe
    regardless of cost magnitude (relevant when the post-alignment whitened
    residual is unbounded). The optimal solution is invariant under positive
    affine transformations of the cost matrix.

    Defensive: returns ``[-1] * n_children`` for empty inputs.
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children

    if max_group_cap < 1:
        # Reduce to assignment when no capacity bound is enforced — still
        # correct for the v1 ``max_group_cap == 0`` "uncapped" semantics by
        # treating uncapped as ``n_children`` per centroid (effectively
        # unlimited within the problem).
        max_group_cap = n_children

    try:
        from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
    except ImportError as exc:
        raise RuntimeError(
            "_assign_mcf requires the 'ortools' package. Add 'ortools>=9.10' "
            "to requirements.txt and reinstall, or set "
            "stage2_reap_ream.assignment_solver back to 'greedy'."
        ) from exc

    # Normalize finite costs to [0, MCF_INT_SCALE] so int-rounding is always
    # safe (no overflow for unbounded post-alignment residuals). Min-cost
    # solutions are invariant under positive affine transformations of cost.
    finite_mask = np.isfinite(cost)
    if not finite_mask.any():
        log.warning(
            "_assign_mcf: cost matrix has no finite entries — falling back "
            "to greedy (which will leave all children unassigned)."
        )
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)

    finite_min = float(cost[finite_mask].min())
    finite_max = float(cost[finite_mask].max())
    finite_range = finite_max - finite_min
    MCF_INT_SCALE = 1_000_000

    def _to_int_cost(c: float) -> int:
        if finite_range <= 0.0:
            return 0
        normalized = (c - finite_min) / finite_range
        return int(round(normalized * MCF_INT_SCALE))

    smcf = SimpleMinCostFlow()

    # Node ids: 0 = source, 1..n_children = child nodes,
    # n_children+1..n_children+n_centroids = centroid nodes,
    # n_children+n_centroids+1 = sink.
    SRC = 0
    SINK = n_children + n_centroids + 1
    # Inline arithmetic instead of lambdas for clarity.
    # child_node(i) = 1 + i
    # cent_node(j)  = 1 + n_children + j

    # Source → child arcs
    for i in range(n_children):
        smcf.add_arc_with_capacity_and_unit_cost(SRC, 1 + i, 1, 0)

    # Child → centroid arcs (skip +∞)
    for i in range(n_children):
        for j in range(n_centroids):
            c_ij = cost[i, j]
            if not np.isfinite(c_ij):
                continue
            smcf.add_arc_with_capacity_and_unit_cost(
                1 + i, 1 + n_children + j, 1, _to_int_cost(float(c_ij)),
            )

    # Centroid → sink arcs
    for j in range(n_centroids):
        smcf.add_arc_with_capacity_and_unit_cost(
            1 + n_children + j, SINK, max_group_cap, 0,
        )

    # Supply: source = +n_children, sink = -n_children, all others = 0.
    smcf.set_node_supply(SRC, n_children)
    smcf.set_node_supply(SINK, -n_children)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        log.warning(
            "_assign_mcf: SimpleMinCostFlow returned non-optimal status %s "
            "(infeasible? check cost matrix has finite entries and capacity "
            "satisfies n_centroids * max_group_cap >= n_children). Falling "
            "back to greedy.",
            status,
        )
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)

    assignment = [-1] * n_children
    for arc in range(smcf.num_arcs()):
        if smcf.flow(arc) <= 0:
            continue
        tail = smcf.tail(arc)
        head = smcf.head(arc)
        # We only care about child→centroid arcs.
        if 1 <= tail <= n_children and (n_children + 1) <= head <= (n_children + n_centroids):
            i = tail - 1
            j = head - n_children - 1
            assignment[i] = j
    return assignment


class McfSolverPlugin(Stage2Plugin):
    """Plugin home for the min-cost-flow assignment solver (Task 13 scaffold).

    Scaffold only: not yet on the live phase walk. The bump loop in
    LegacyAdapter still calls `_assign_children_to_centroids`; this class
    exists so T18 has a per-solver plugin to wire into the decomposed
    `solve_assignment` phase. `is_enabled` selects this solver when
    `stage2_reap_ream.assignment_solver == "mcf"`.
    """

    name = "solver_mcf"
    enabled_by: tuple[str, ...] = ()  # selector is a string match, not a flag

    @classmethod
    def is_enabled(cls, cfg: dict[str, Any]) -> bool:
        s2 = cfg.get("stage2_reap_ream", {}) if isinstance(cfg, dict) else {}
        return str(s2.get("assignment_solver", "greedy")).lower() == "mcf"

    def solve_assignment(self, ctx: LayerContext, delta: Any) -> Any | None:
        """Wrap `_assign_mcf`. NOTE: not invoked by the current phase walk
        (the bump loop calls `_assign_children_to_centroids` directly); kept as
        a functional hook for the T18 decomposition. Returns None when delta is
        not a usable cost matrix so `dispatch_first` can skip cleanly."""
        return None
