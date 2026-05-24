"""Hungarian assignment solver (Task 13 of the plugin-architecture refactor).

Plugin home for ``_assign_hungarian`` — the rectangular Hungarian solver backed
by ``scipy.optimize.linear_sum_assignment``. Extracted verbatim from
``stage2_reap_ream`` in Task 13.

Imports ``_assign_mcf`` from ``solver_mcf`` as the capacitated-case fallback
(when ``n_children > n_centroids`` Hungarian alone cannot solve the problem).
The monolith re-imports ``_assign_hungarian``.

``HungarianSolverPlugin`` is a scaffold-only plugin (see
``solver_greedy``). Circular-import note: this module imports only ``numpy``,
``scipy``, ``solver_mcf``, ``pipeline.base`` and ``pipeline.context`` — none of
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
    paper = "Rectangular Hungarian assignment solver via scipy."
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
