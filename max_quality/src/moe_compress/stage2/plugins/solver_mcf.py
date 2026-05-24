"""Capacitated min-cost-flow assignment solver.

Paper
-----
Standard capacitated min-cost-flow (MCF) algorithm — Ahuja, Magnanti
& Orlin, "Network Flows" §9 (the transportation polytope is totally
unimodular, so the LP relaxation has integer optima). Implemented via
OR-Tools' ``SimpleMinCostFlow``.

This plugin is the capacitated branch of deviation D-mcf-assignment
from arXiv:2604.04356 (REAM): the project adds an optimal capacitated
assignment alternative to the paper-faithful descending-saliency
greedy (see :mod:`stage2.plugins.solver_greedy`). The slack-capacity
case is handled by Hungarian (see
:mod:`stage2.plugins.solver_hungarian`).

Official code
-------------
OR-Tools ``SimpleMinCostFlow``: google/or-tools upstream
implementation (released; standard distribution). No project-specific
upstream code to pin.

Deviation: D-mcf-assignment (MCF branch)
----------------------------------------
REAM §4 (arXiv:2604.04356) uses descending-saliency single-pass greedy
with a per-centroid cap (``group_size``) — picks the highest-saliency
centroid first, absorbs up to ``C_max`` non-centroids by lowest cost.
The greedy is biased toward the highest-saliency centroid because it
picks first.

Stage 2 v2's ``assignment_solver="mcf"`` (or ``"auto"`` falling through
to mcf when ``n_NC > N'_l``) uses OR-Tools' ``SimpleMinCostFlow`` with
cost-range normalization to int.

Why MCF
-------
MCF is integer-optimal under LP relaxation (transportation polytope is
totally unimodular, Ahuja-Magnanti-Orlin §9). Synthetic counterexamples
(STRATEGY_NEXT reviewer report §2) show greedy 28-34 % above the MCF
optimum on tight-capacity instances. At project scale
(``N = 256, N'_l ∈ [128, 200], C_max = 7``) the gap is expected
smaller (loose capacity), but MCF is essentially free (~10 ms / layer
× 40 layers).

ortools-safety contract (CRITICAL)
-----------------------------------
This module **must be importable in an environment without
``ortools``**. The ``from ortools... import SimpleMinCostFlow``
statement stays **inside the body of ``_assign_mcf``** — it is never a
module-scope import. Module load needs only ``numpy``, ``logging``,
``_assign_greedy`` (the fallback), and the plugin-class imports. When
``ortools`` is absent, ``_assign_mcf`` raises a ``RuntimeError`` *at
call time*, not an ``ImportError`` at import time.

Fallback
--------
Imports ``_assign_greedy`` from
:mod:`stage2.plugins.solver_greedy` and falls back when MCF has no
finite entries / a non-optimal status return from OR-Tools.

Output context contract
-----------------------
Pure callable plugin (no state). Returns the layer's assignment from
the input cost matrix + scores + frequencies.

Naming-history note
-------------------
Step 3 of the REAM pipeline alternative-solver branch. Existing log
lines / Trackio keys preserved for dashboard back-compat.

ortools-safety contract (CRITICAL): this module **must be importable in an
environment without ``ortools``**. The ``from ortools... import
SimpleMinCostFlow`` statement stays **inside the body of ``_assign_mcf``** — it
is never a module-scope import. Module load needs only ``numpy``, ``logging``,
``_assign_greedy`` (the fallback), and the plugin-class imports. When ``ortools``
is absent, ``_assign_mcf`` raises a ``RuntimeError`` *at call time*, not an
``ImportError`` at import time.

Imports ``_assign_greedy`` from ``solver_greedy`` as the no-finite-entries /
non-optimal-status fallback. The monolith re-imports ``_assign_mcf``.

``McfSolverPlugin`` is a scaffold-only plugin (see ``solver_greedy``).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...pipeline.context import PipelineContext
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


class McfSolverPlugin:
    """Plugin home for the min-cost-flow assignment solver.

    LIVE (S2-8): services the solve_assignment slot when assignment_solver
    selects this solver. `is_enabled` selects this solver when
    `stage2_reap_ream.assignment_solver == "mcf"`.
    """

    name = "solver_mcf"
    paper = (
        "Capacitated min-cost-flow (Ahuja-Magnanti-Orlin §9) via OR-Tools "
        "SimpleMinCostFlow. Alternative to REAM §4 greedy "
        "(see :mod:`stage2.plugins.solver_greedy`); implements deviation "
        "D-mcf-assignment capacitated branch from baseline REAM "
        "arXiv:2604.04356. Falls back to greedy on infeasibility / no-finite-"
        "entries. See module docstring."
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
        return str(s2.get("assignment_solver", "greedy")).lower() == "mcf"

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def solve_assignment(self, ctx: PipelineContext, delta: Any) -> Any | None:
        """Slot ``solve_assignment`` — child→centroid assignment solver.

        Delegates to the shared ``_solve_for_plugin`` helper (verbatim lift of
        ``LegacyAdapter.solve_assignment``). Reaches this plugin only when
        ``registry.enabled`` kept it, i.e. ``assignment_solver == "mcf"``."""
        from .solver_dispatch import _solve_for_plugin
        return _solve_for_plugin(self, ctx, delta)
