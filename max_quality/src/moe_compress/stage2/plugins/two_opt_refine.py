"""Two-opt assignment refinement (Task 14 of the plugin-architecture refactor).

Plugin home for Direction D — the 2-opt local-refinement pass (spec §5 step 3.5).
``_two_opt_refine`` was extracted verbatim from ``stage2_reap_ream`` in Task 14;
this is a pure relocation, not a refactor — the algorithm, docstring and comments
are byte-identical to the former monolith definition.

The monolith re-imports ``_two_opt_refine`` so external callers (tests, the
``MOE_STAGE2_LEGACY_LOOP=1`` legacy-loop path, ``LegacyAdapter``) keep their
import paths working unchanged.

``TwoOptRefinePlugin`` is a scaffold-only ``Stage2Plugin`` — not yet on the live
phase walk (``LegacyAdapter.compute_assignment`` still calls ``_two_opt_refine``
directly inside the bump loop); it gives T18 a per-refiner plugin to wire into
the decomposed ``refine_assignment`` phase. Circular-import note: this module
imports only ``pipeline.base`` and ``pipeline.context``, neither of which
imports ``stage2_reap_ream``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._framework.base import Stage2Plugin
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


def _two_opt_refine(
    assignment: list[int],
    cost: np.ndarray,
    max_group_cap: int,
) -> list[int]:
    """Direction D — greedy + one 2-opt local-refinement loop (spec §5 step 3.5).

    Strictly-improving local search over an already-feasible child→centroid
    assignment. Operates purely on the assignment list, the cost matrix and the
    per-centroid capacity cap, so it is model-agnostic (no expert-count, dim,
    top-k or activation assumptions).

    ``assignment`` is a list of length ``n_children``; ``assignment[ch]`` is the
    centroid *index* in ``[0, n_centroids)`` that child ``ch`` is assigned to, or
    ``-1`` if unassigned. ``cost`` has shape ``(n_children, n_centroids)``.

    Two move types, both applied only when *strictly* lowering total cost:

      * **swap** — for a pair of children ``(i, j)`` in *different* groups,
        exchange their centroids if
        ``cost[i, g_j] + cost[j, g_i] < cost[i, g_i] + cost[j, g_j]``.
        A swap leaves every group size unchanged, so capacity is preserved by
        construction — but we still verify post-swap group sizes defensively.

      * **move** — relocate a single child ``i`` to a different centroid ``g``
        when ``g`` has spare capacity and ``cost[i, g] < cost[i, g_i]``.

    Capacity is re-checked on every accepted move; with ``max_group_cap <= 0``
    (uncapped, ablation-only path) groups are treated as unbounded. Passes
    repeat until a full pass makes no improving move; each pass is O(n²) and
    ``n`` (non-centroid expert count) is small. The function NEVER accepts a
    non-improving move, so the returned assignment's total cost is provably
    ``<=`` the input's.

    Unassigned children (``-1``) and any child whose current cost is non-finite
    are skipped — 2-opt only reshuffles already-feasible finite-cost merges and
    never assigns a previously-unassigned child.

    Returns a new assignment list (the input is not mutated).
    """
    n_children = len(assignment)
    if n_children == 0 or cost.size == 0:
        return list(assignment)

    n_centroids = cost.shape[1]
    result = list(assignment)

    # Per-centroid occupancy (number of non-centroid children currently in each
    # group). Unassigned children (-1) contribute to no group.
    group_size = [0] * n_centroids
    for g in result:
        if g >= 0:
            group_size[g] += 1

    # max_group_cap <= 0 → uncapped; use an effectively-infinite cap so the
    # capacity guards become no-ops without special-casing every check.
    cap = max_group_cap if max_group_cap > 0 else n_children

    def _cost(ch: int, g: int) -> float:
        return float(cost[ch, g])

    improved = True
    while improved:
        improved = False

        # --- single moves ---------------------------------------------------
        for i in range(n_children):
            g_i = result[i]
            if g_i < 0:
                continue
            cur = _cost(i, g_i)
            if not np.isfinite(cur):
                continue
            best_g = g_i
            best_cost = cur
            for g in range(n_centroids):
                if g == g_i:
                    continue
                if group_size[g] >= cap:
                    continue
                c = _cost(i, g)
                if np.isfinite(c) and c < best_cost:
                    best_cost = c
                    best_g = g
            if best_g != g_i:
                # Strict improvement, target has spare capacity.
                group_size[g_i] -= 1
                group_size[best_g] += 1
                result[i] = best_g
                improved = True

        # --- pairwise swaps -------------------------------------------------
        for i in range(n_children):
            g_i = result[i]
            if g_i < 0:
                continue
            for j in range(i + 1, n_children):
                g_j = result[j]
                if g_j < 0 or g_j == g_i:
                    continue
                cur = _cost(i, g_i) + _cost(j, g_j)
                new = _cost(i, g_j) + _cost(j, g_i)
                if not (np.isfinite(cur) and np.isfinite(new)):
                    continue
                if new < cur:
                    # A swap is size-neutral for both groups, so caps are
                    # preserved by construction. Assert the invariant loudly
                    # rather than silently skipping an improving swap.
                    assert group_size[g_i] <= cap and group_size[g_j] <= cap, (
                        "2-opt: group size exceeded cap before a size-neutral swap"
                    )
                    result[i] = g_j
                    result[j] = g_i
                    g_i = g_j
                    improved = True

    return result


class TwoOptRefinePlugin(Stage2Plugin):
    """Plugin home for Direction D — the 2-opt local-refinement pass (T14 scaffold).

    Scaffold only: not yet on the live phase walk. ``LegacyAdapter.compute_assignment``
    still calls ``_two_opt_refine`` directly inside the bump loop, and the
    ``MOE_STAGE2_LEGACY_LOOP=1`` path in ``stage2_reap_ream.run()`` does too.
    This class exists so T18 has a per-refiner plugin to wire into the decomposed
    ``refine_assignment`` phase.

    Config gate: enabled iff ``stage2_reap_ream.two_opt_refine`` is truthy.
    ``two_opt_refine`` is a plain bool (default ``False``), so the base
    ``Stage2Plugin.is_enabled`` (AND-of-``enabled_by``-flags) expresses the gate
    directly — no override needed.

    The greedy-only guard (``two_opt_refine`` is ignored unless
    ``assignment_solver == "greedy"``) is orchestration and STAYS in
    ``LegacyAdapter.compute_assignment``; T14 does not move it.
    """

    name = "two_opt_refine"
    enabled_by: tuple[str, ...] = ("two_opt_refine",)

    def refine_assignment(
        self, ctx: PipelineContext, asg: Any, delta: Any
    ) -> tuple[Any, Any, dict] | None:
        """Documented no-op for T14.

        The live refinement call still belongs to the LegacyAdapter bump loop
        (and the legacy-loop path), which invoke ``_two_opt_refine`` directly.
        Returning ``None`` makes ``PluginRegistry.dispatch_first`` skip this
        plugin cleanly. T18 wires the real call here once ``compute_assignment``
        is decomposed into the fine-grained phase walk.
        """
        return None
