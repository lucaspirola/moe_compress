"""Stage 2 assignment-solver dispatcher + registry.

Paper
-----
**No paper for the dispatcher itself.** This module is the registry
hub that maps the config-driven ``assignment_solver`` choice to a
sibling solver-plugin callable:

- ``"greedy"`` (default, paper-faithful): see
  :mod:`stage2.plugins.solver_greedy` — arXiv:2604.04356 §4 baseline.
- ``"hungarian"`` / ``"mcf"`` / ``"auto"``: see
  :mod:`stage2.plugins.solver_hungarian` / ``solver_mcf`` /
  ``solver_auto`` — deviation D-mcf-assignment from baseline REAM.
- ``"sinkhorn"``: see :mod:`stage2.plugins.solver_sinkhorn` —
  deviation D-sinkhorn-soft-assign from baseline REAM.

This is the **hub** of the solver import DAG: it imports all five solver
callables (``greedy``, ``hungarian``, ``mcf``, ``sinkhorn``, ``auto``) and owns:

* ``SolverName`` — the ``Literal`` type for the ``assignment_solver`` config
  field. Its canonical home is here, next to the registry it types.
* ``SOLVERS`` — the name→callable registry dict.
* ``_assign_children_to_centroids`` — the public dispatcher, resolved through
  ``SOLVERS``. Extracted verbatim (docstring + behaviour) from
  ``stage2_reap_ream`` in Task 13; only the dispatch body was rewritten to use
  the registry instead of an if/else chain.

The monolith re-imports ``SolverName`` and ``_assign_children_to_centroids`` so
external callers (tests, the ``MOE_STAGE2_LEGACY_LOOP=1`` path,
``LegacyAdapter``'s late import, ``_em_refine_assignment``) keep their import
paths unchanged.

Circular-import note: this module imports only the sibling ``solver_*`` modules,
none of which import ``stage2_reap_ream``. No cycle at module load.
"""
from __future__ import annotations

import typing
from typing import Callable, Literal

import numpy as np

from .solver_auto import _assign_auto
from .solver_greedy import _assign_greedy
from .solver_hungarian import _assign_hungarian
from .solver_mcf import _assign_mcf
from .solver_sinkhorn import _assign_sinkhorn

# Stage 2 v2 — solver dispatch literal. Adding new solvers requires
# updating both this Literal AND the SOLVERS registry dict below.
# The module-level assert below enforces this invariant at import time.
SolverName = Literal["greedy", "hungarian", "mcf", "auto", "sinkhorn"]

# name -> callable(cost, n_children, n_centroids, max_group_cap) -> list[int]
# The "sinkhorn" entry is in the registry for completeness / is_enabled /
# introspection; the dispatcher special-cases it for the three sinkhorn_*
# keyword arguments (see _assign_children_to_centroids below).
SOLVERS: dict[str, Callable[..., list[int]]] = {
    "greedy": _assign_greedy,
    "hungarian": _assign_hungarian,
    "mcf": _assign_mcf,
    "auto": _assign_auto,
    "sinkhorn": _assign_sinkhorn,
}

# Enforce SolverName Literal ↔ SOLVERS registry agreement at import time,
# so adding a solver in one place without the other fails loudly.
assert set(SOLVERS.keys()) == set(typing.get_args(SolverName)), (
    "SolverName Literal and SOLVERS registry are out of sync: "
    f"Literal={set(typing.get_args(SolverName))}, SOLVERS={set(SOLVERS.keys())}"
)


def _assign_children_to_centroids(
    cost: np.ndarray,
    n_children: int,
    n_centroids: int,
    max_group_cap: int = 0,
    *,
    solver: SolverName = "greedy",
    sinkhorn_epsilon_init: float = 1.0,
    sinkhorn_epsilon_final: float = 0.01,
    sinkhorn_iters: int = 200,
) -> list[int]:
    """Assign non-centroid children to centroids under a per-centroid cap.

    Solver dispatch (``solver`` argument; spec § 5 Step 3 of
    ``max_quality/docs/stage2_assignment_revision.md``):

    * ``"greedy"`` — single-pass descending-saliency greedy (legacy, paper
      §4); preserves bit-identical behavior with prior Stage 2 runs. **This is
      the default and is required for the Stage 2 v1→v2 compatibility
      invariant.**
    * ``"hungarian"`` — rectangular Hungarian (``scipy.optimize.linear_sum_assignment``)
      on the cost matrix, padded to a square problem when capacity allows
      multiple absorption per centroid. Optimal under capacity-1 problems
      (``n_children ≤ n_centroids``); falls back to MCF when capacitated.
    * ``"mcf"`` — capacitated min-cost flow via OR-Tools' ``SimpleMinCostFlow``.
      Optimal under capacity ``max_group_cap`` per centroid. Drop-in replacement
      for greedy that does not bias toward the highest-saliency centroid.
    * ``"auto"`` — picks ``hungarian`` when ``n_children ≤ n_centroids``,
      else ``mcf``.
    * ``"sinkhorn"`` — capacitated entropy-regularized OT (Tier 3 / M9).
      Solved via log-domain Sinkhorn-Knopp with linear ε-annealing and a
      slack-child dummy-row construction; see :func:`_assign_sinkhorn`.

    NOTE: The greedy branch is unchanged from the v1 Stage 2; the dispatcher
    is structured so flipping ``solver`` to a non-greedy value is the only
    semantic change. With ``solver="greedy"`` the output is bit-identical to
    the prior implementation.

    The legacy greedy path:
      When ``max_group_cap == 0`` (uncapped), each child is independently
      assigned to its nearest centroid by cost (argmin over centroid columns).

      When ``max_group_cap > 0``, iterates centroids once in order
      ``0..n_centroids-1`` (caller builds centroid_ids in descending saliency
      — column 0 = highest-saliency centroid).  For each centroid, greedily
      absorbs up to ``max_group_cap`` unassigned children (lowest cost = most
      similar first).

    The caller is responsible for ensuring feasibility before calling:
    ``n_centroids * max_group_cap >= n_children`` (spec § 5 Step 3). When the
    feasibility check passes and the cost matrix is finite, every child is
    guaranteed to receive ``assignment >= 0``. This guarantee assumes
    ``n_centroids >= 1``; when ``n_centroids == 0`` all children are assigned
    ``-1`` (no centroid).

    Returns:
        List of length ``n_children`` where entry ``ch`` is:
          ``>= 0``  → centroid column index this child is merged into
          ``-1``    → child was not absorbed (should not occur under
                      feasibility + finite costs)
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children

    # `solver` is typed as the SolverName Literal (strictly lowercase), but
    # the type system is static-only — at runtime a YAML-supplied value can
    # arrive in any case. Normalise once here so human-friendly config values
    # (e.g. ``GREEDY``, ``Hungarian``) dispatch correctly.
    solver = solver.lower()

    if solver == "sinkhorn":
        # Sinkhorn needs the three sinkhorn_* kwargs; special-case it so the
        # other solver signatures stay byte-identical (no dead **kwargs).
        return _assign_sinkhorn(
            cost, n_children, n_centroids, max_group_cap,
            epsilon_init=sinkhorn_epsilon_init,
            epsilon_final=sinkhorn_epsilon_final,
            iters=sinkhorn_iters,
        )

    fn = SOLVERS.get(solver)
    if fn is None:
        raise ValueError(
            f"_assign_children_to_centroids: unknown solver {solver!r}; "
            f"expected one of {sorted(SOLVERS.keys())!r}."
        )
    return fn(cost, n_children, n_centroids, max_group_cap)


def _solve_for_plugin(plugin, ctx, delta):
    """Shared solve_assignment slot body — verbatim lift of
    LegacyAdapter.solve_assignment. Assumes `plugin` survived
    registry.enabled(), so plugin.assignment_solver == config.assignment_solver."""
    n_ream_nc = ctx.get("_iter_n_ream_nc")
    n_ream_c = ctx.get("_iter_n_ream_c")
    return _assign_children_to_centroids(
        delta, n_ream_nc, n_ream_c, plugin.max_group_cap,
        solver=plugin.assignment_solver,
        sinkhorn_epsilon_init=plugin.sinkhorn_epsilon_init,
        sinkhorn_epsilon_final=plugin.sinkhorn_epsilon_final,
        sinkhorn_iters=plugin.sinkhorn_iters,
    )
