"""Auto-pick between Hungarian (slack) and MCF (tight) assignment.

Paper
-----
**No paper.** Project-original meta-router for deviation
D-mcf-assignment (consumed at :mod:`stage2.plugins.solver_hungarian`).

The auto-rule is the STRATEGY_NEXT §5 step 4d dispatch heuristic:

  - ``n_children < n_centroids`` (slack) or ``n_children == n_centroids``
    (square 1-1) → Hungarian (rectangular linear-sum-assignment is optimal).
  - ``n_children > n_centroids`` (tight / capacitated) → MCF
    (Hungarian alone cannot solve the capacitated problem).

When the ``assignment_solver`` config is set to ``"auto"``, this
plugin's callable dispatches into the appropriate sibling solver.

Official code
-------------
None — the heuristic is project-original. Sibling solvers'
implementations live at :mod:`stage2.plugins.solver_hungarian` (scipy
``linear_sum_assignment``, Kuhn/Munkres) and
:mod:`stage2.plugins.solver_mcf` (OR-Tools ``SimpleMinCostFlow``).

Why a separate plugin (not inline in the dispatcher)
----------------------------------------------------
The auto-rule was originally an ``if/elif`` branch inside the public
dispatcher. Promoting it to a first-class registry callable means
operators can configure ``assignment_solver="auto"`` and the dispatcher
just looks it up in the registry — no special-case branch. It also
lets the meta-router be unit-tested in isolation.

Naming-history note
-------------------
STRATEGY_NEXT § 5 step 4d label is retained throughout this module for
dashboard / Trackio back-compat; the current plugin architecture does
not require the step-numbering taxonomy, but the label is kept
consistently in prose and code comments so historical log lines and
keys keep their referent.

Imports ``_assign_hungarian`` and ``_assign_mcf`` from the sibling solver
modules; it deliberately does **not** import the full ``SOLVERS`` registry —
``solver_dispatch`` is the hub, ``solver_auto`` only owns the auto heuristic.

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

    Picks the rectangular Hungarian solver when ``n_children < n_centroids``
    (slack) or ``n_children == n_centroids`` (square 1-1, no slack), where
    Hungarian is optimal, and the capacitated min-cost-flow solver otherwise.
    This is the same heuristic the dispatcher's former inline ``"auto"`` branch
    applied; it is now a first-class entry in the ``SOLVERS`` registry.

    Note: ``_assign_hungarian`` itself already falls back to ``_assign_mcf``
    for ``n_children > n_centroids`` (see solver_hungarian.py:114-115), so
    the explicit branch below is semantically equivalent to calling
    ``_assign_hungarian`` unconditionally. We keep the branch for log /
    Trackio key clarity (the dispatch decision is observable here) rather
    than for correctness.
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
    paper = (
        "Auto-router for D-mcf-assignment: hungarian when n_children ≤ "
        "n_centroids (slack), mcf otherwise (tight). Project-original "
        "(no paper). STRATEGY_NEXT §5 step 4d. See "
        ":mod:`stage2.plugins.solver_hungarian` for the D-mcf-assignment "
        "deviation. See module docstring."
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
        # NOTE: ``assignment_solver`` and the three ``sinkhorn_*`` kwargs are
        # vestigial-but-required here — ``_assign_auto`` itself never reads
        # them, but the shared ``_solve_for_plugin`` helper (see
        # ``solver_dispatch``) expects every solver plugin to expose the same
        # ``__init__`` signature so it can hydrate plugins uniformly from the
        # stage-2 config dict. Removing them would break plugin construction.
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
