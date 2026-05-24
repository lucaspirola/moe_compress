"""Hungarian (rectangular linear-sum-assignment) solver.

Paper
-----
Classic linear-sum-assignment algorithm — Kuhn (1955), Munkres (1957);
implemented via SciPy's ``scipy.optimize.linear_sum_assignment``
(per scipy docs: "a modified Jonker-Volgenant algorithm with no
initialization").

This plugin is part of deviation D-mcf-assignment from
arXiv:2604.04356 (REAM): the project adds an optimal-assignment
alternative to the paper-faithful descending-saliency greedy (see
:mod:`stage2.plugins.solver_greedy`). The Hungarian branch handles the
**slack-capacity** case (``n_children ≤ n_centroids``); the
capacitated branch dispatches to MCF (see
:mod:`stage2.plugins.solver_mcf`).

Official code
-------------
SciPy ``linear_sum_assignment``: scientific-python/scipy upstream
implementation (released; standard library). No project-specific
upstream code to pin.

Deviation: D-mcf-assignment (Hungarian branch)
----------------------------------------------
REAM §4 (arXiv:2604.04356) uses descending-saliency single-pass greedy
with a per-centroid cap (``group_size``) — picks the highest-saliency
centroid first, absorbs up to ``C_max`` non-centroids by lowest cost.
Greedy is biased toward the highest-saliency centroid because it picks
first.

Stage 2 v2 adds
``assignment_solver: "greedy" | "hungarian" | "mcf" | "auto" | "sinkhorn"``
(default ``"greedy"`` reproduces v1 bit-identically). The Hungarian
branch uses ``scipy.linear_sum_assignment`` with ``+∞`` → large-finite
sentinel for forbidden arcs.

Hungarian is the integer-optimal LP solution for the slack-capacity
1-1 assignment problem (the transportation polytope is totally
unimodular — Ahuja-Magnanti-Orlin §9). Synthetic counterexamples
(STRATEGY_NEXT reviewer report §2) show greedy 28-34 % above the
optimum on tight-capacity instances. At the project's
``N = 256, N'_l ∈ [128, 200], C_max = 7`` the gap is expected smaller
(loose capacity); Hungarian is cheap at these sizes (project-anecdotal
timing; no benchmark in tree).

Capacitated fallback
--------------------
When ``n_children > n_centroids`` Hungarian alone cannot solve the
problem (per-centroid cap > 1 needed). This module imports
``_assign_mcf`` from :mod:`stage2.plugins.solver_mcf` and falls back
when ``HungarianSolverPlugin`` is invoked on a tight-capacity layer.
The ``"auto"`` solver (see :mod:`stage2.plugins.solver_auto`)
implements this dispatch as a separate plugin.

Output context contract
-----------------------
Pure callable plugin (no state). Returns the layer's assignment from
the input cost matrix + scores + frequencies.

Naming-history note
-------------------
Step 3 of the REAM pipeline alternative-solver branch. Existing log
lines / Trackio keys preserved for dashboard back-compat.

Monolith re-export
------------------
The monolith re-imports ``_assign_hungarian`` from this module so the
legacy ``stage2_reap_ream`` entry-points see the same helper.

Back-compat notes
-----------------
``HungarianSolverPlugin`` is a scaffold-only plugin (see
``solver_greedy``). Circular-import note: this module imports only
``numpy``, ``scipy``, ``solver_mcf`` and ``pipeline.context`` — none of
which import ``stage2_reap_ream``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...pipeline.context import PipelineContext
from .solver_mcf import _assign_mcf


def _assign_hungarian(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int,
) -> list[int]:
    """Rectangular Hungarian assignment via ``scipy.linear_sum_assignment``.

    Optimal under the 1-1 capacity case (``n_children ≤ n_centroids`` with
    ``max_group_cap >= 1``). When ``n_children > n_centroids``, the problem
    becomes capacitated and Hungarian alone cannot solve it; we fall back to
    MCF. This matches the spec § 5 step 4d "auto" rule (hungarian in slack,
    mcf in tight).

    The cost matrix is shaped ``(n_children, n_centroids)``. ``+inf`` entries
    are replaced with a large finite sentinel before passing to scipy, since
    ``linear_sum_assignment`` raises on inf inputs.

    Defensive: returns ``[-1] * n_children`` for empty inputs so this helper
    can be called directly without re-doing the dispatcher's early-exit.
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children
    # Capacitated → defer to MCF. ``max_group_cap == 0`` carries the v1
    # "uncapped" semantics (each child to its argmin centroid); MCF with
    # ``max_group_cap = n_children`` reproduces that, so route there too
    # rather than letting scipy's rectangular Hungarian leave excess
    # children unassigned.
    if n_children > n_centroids:
        return _assign_mcf(cost, n_children, n_centroids, max_group_cap)

    # Replace inf with a large finite sentinel above any finite cost so that
    # scipy treats the +∞ entries as effectively forbidden but does not raise.
    finite_max = float(np.nanmax(cost[np.isfinite(cost)])) if np.isfinite(cost).any() else 1.0
    big = max(finite_max, 1.0) * 1e9
    safe_cost = np.where(np.isfinite(cost), cost, big)

    row_ind, col_ind = linear_sum_assignment(safe_cost)
    assignment = [-1] * n_children
    for r, c in zip(row_ind, col_ind):
        # Skip pairs that were forbidden by the +∞ → big sentinel — leave as
        # unassigned; the caller's orphan-promotion path handles them.
        if safe_cost[r, c] >= big * 0.5:
            continue
        assignment[int(r)] = int(c)
    return assignment


class HungarianSolverPlugin:
    """Plugin home for the Hungarian assignment solver.

    LIVE (S2-8): services the solve_assignment slot when assignment_solver
    selects this solver. `is_enabled` selects this solver when
    `stage2_reap_ream.assignment_solver == "hungarian"`.
    """

    name = "solver_hungarian"
    paper = (
        "Hungarian (Kuhn 1955 / Munkres 1957) rectangular linear-sum-"
        "assignment via scipy.optimize.linear_sum_assignment. "
        "Alternative to REAM §4 greedy (see :mod:`stage2.plugins.solver_greedy`); "
        "implements deviation D-mcf-assignment slack-capacity branch from "
        "baseline REAM arXiv:2604.04356. Capacitated case falls back to "
        ":mod:`stage2.plugins.solver_mcf`. See module docstring."
    )
    config_key = "stage2_reap_ream.assignment_solver"
    # S2-8: the live solve_assignment slot reads the per-bump scratch slots.
    reads: tuple[str, ...] = ("_iter_n_ream_nc", "_iter_n_ream_c")
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        # ``max_group_cap`` is unused by Hungarian itself (1-1 assignment) but
        # is accepted on the signature because it forwards to ``_assign_mcf``
        # for the capacitated fallback branch (see ``_assign_hungarian``).
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
        return str(s2.get("assignment_solver", "greedy")).lower() == "hungarian"

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def solve_assignment(self, ctx: PipelineContext, delta: Any) -> Any | None:
        """Slot ``solve_assignment`` — child→centroid assignment solver.

        Delegates to the shared ``_solve_for_plugin`` helper (verbatim lift of
        ``LegacyAdapter.solve_assignment``). Reaches this plugin only when
        ``registry.enabled`` kept it, i.e. ``assignment_solver == "hungarian"``."""
        from .solver_dispatch import _solve_for_plugin
        return _solve_for_plugin(self, ctx, delta)
