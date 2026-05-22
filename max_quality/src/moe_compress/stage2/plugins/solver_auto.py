"""Auto-pick assignment solver wrapper (Task 13 of the plugin-architecture
refactor).

Plugin home for ``_assign_auto`` — a tiny wrapper that promotes the dispatcher's
former inline ``"auto"`` branch into a first-class registry callable. It picks
``hungarian`` when ``n_children <= n_centroids`` (the slack / 1-1 case where
rectangular Hungarian is optimal) and ``mcf`` otherwise (the capacitated /
tight case). This is the spec § 5 step 4d "auto" rule.

Imports ``_assign_hungarian`` and ``_assign_mcf`` from the sibling solver
modules; it deliberately does **not** import the full ``SOLVERS`` registry —
``solver_dispatch`` is the hub, ``solver_auto`` only owns the auto heuristic.

``AutoSolverPlugin`` is a scaffold-only plugin (see ``solver_greedy``).
Circular-import note: this module imports only ``solver_hungarian``,
``solver_mcf``, ``pipeline.base`` and ``pipeline.context`` — none of which
import ``stage2_reap_ream``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ...pipeline.context import PipelineContext
from .solver_hungarian import _assign_hungarian
from .solver_mcf import _assign_mcf


def _assign_auto(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int,
) -> list[int]:
    """Auto-pick wrapper (spec § 5 step 4d).

    Picks the rectangular Hungarian solver when ``n_children <= n_centroids``
    (the slack / 1-1 capacity case where Hungarian is optimal) and the
    capacitated min-cost-flow solver otherwise. This is the same heuristic the
    dispatcher's former inline ``"auto"`` branch applied; it is now a
    first-class entry in the ``SOLVERS`` registry.
    """
    if n_children <= n_centroids:
        return _assign_hungarian(cost, n_children, n_centroids, max_group_cap)
    return _assign_mcf(cost, n_children, n_centroids, max_group_cap)


class AutoSolverPlugin:
    """Plugin home for the auto-pick assignment solver.

    LIVE (S2-8): services the solve_assignment slot when assignment_solver
    selects this solver. `is_enabled` selects this solver when
    `stage2_reap_ream.assignment_solver == "auto"`.
    """

    name = "solver_auto"
    paper = "Auto-pick assignment solver: hungarian in slack, mcf in tight."
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
        return str(s2.get("assignment_solver", "greedy")).lower() == "auto"

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def solve_assignment(self, ctx: PipelineContext, delta: Any) -> Any | None:
        """Slot ``solve_assignment`` — child→centroid assignment solver.

        Delegates to the shared ``_solve_for_plugin`` helper (verbatim lift of
        ``LegacyAdapter.solve_assignment``). Reaches this plugin only when
        ``registry.enabled`` kept it, i.e. ``assignment_solver == "auto"``."""
        from .solver_dispatch import _solve_for_plugin
        return _solve_for_plugin(self, ctx, delta)
