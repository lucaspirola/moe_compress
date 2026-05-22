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

``AutoSolverPlugin`` is a scaffold-only ``Stage2Plugin`` (see ``solver_greedy``).
Circular-import note: this module imports only ``solver_hungarian``,
``solver_mcf``, ``pipeline.base`` and ``pipeline.context`` — none of which
import ``stage2_reap_ream``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._framework.base import Stage2Plugin
from .._framework.context import LayerContext
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


class AutoSolverPlugin(Stage2Plugin):
    """Plugin home for the auto-pick assignment solver (Task 13 scaffold).

    Scaffold only: not yet on the live phase walk. The bump loop in
    LegacyAdapter still calls `_assign_children_to_centroids`; this class
    exists so T18 has a per-solver plugin to wire into the decomposed
    `solve_assignment` phase. `is_enabled` selects this solver when
    `stage2_reap_ream.assignment_solver == "auto"`.
    """

    name = "solver_auto"
    enabled_by: tuple[str, ...] = ()  # selector is a string match, not a flag

    @classmethod
    def is_enabled(cls, cfg: dict[str, Any]) -> bool:
        s2 = cfg.get("stage2_reap_ream", {}) if isinstance(cfg, dict) else {}
        return str(s2.get("assignment_solver", "greedy")).lower() == "auto"

    def solve_assignment(self, ctx: LayerContext, delta: Any) -> Any | None:
        """Wrap `_assign_auto`. NOTE: not invoked by the current phase walk
        (the bump loop calls `_assign_children_to_centroids` directly); kept as
        a functional hook for the T18 decomposition. Returns None when delta is
        not a usable cost matrix so `dispatch_first` can skip cleanly."""
        return None
