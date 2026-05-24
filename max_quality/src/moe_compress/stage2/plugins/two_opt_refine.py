"""2-opt local-refinement of an already-feasible assignment.

Paper
-----
**No paper.** Project-original — STRATEGY_NEXT "Direction D" /
project §5 step 3.5. 2-opt is a classic combinatorial local-search
heuristic (Croes 1958, originally for TSP; cf. Lin 1965 (k-opt) and
Lin-Kernighan 1973); applied here to capacitated expert-assignment.

Baseline REAM (arXiv:2604.04356), REAP (arXiv:2510.13999), and the
solver-plugin alternatives (Hungarian / MCF / Sinkhorn) all produce a
single-shot assignment. This plugin REFINES that already-feasible
input assignment by applying 2-opt swap+move passes on top of the
chosen solver's output: greedy seeding lives upstream in
``solver_greedy`` (this plugin does not seed, only refines).

Official code
-------------
None — this implementation is project-original. 2-opt is a textbook
local-search primitive.

Why a 2-opt pass exists
-----------------------
Even the optimal single-shot capacitated solver produces a globally
optimal assignment **with respect to the static cost matrix**. The
cost matrix itself is a proxy for end-to-end merge damage; 2-opt's
locality lets the refinement chase low-cost swaps that the static
cost ordering missed (e.g. when two centroids both want the same
non-centroid but the second-choice for one is much cheaper than for
the other).

In practice the 2-opt pass is cheap — move phase is
O(n_NC × n_centroids), swap phase is O(n_NC²); total
O(n_NC × (n_NC + n_centroids)) per round — and converges in 1-2
rounds on production layers. It is opt-in and disabled by default —
see ``two_opt_refine`` (bool) in the Stage 2 config.

Refine-chain ordering
---------------------
``TwoOptRefinePlugin`` is LIVE as of S2-9 and is the **first** link
of the ``refine_assignment`` chain (two-opt THEN EM), registered ahead
of :mod:`stage2.plugins.em_refine` and the dead-fallback
``LegacyAdapter``. 2-opt operates on the **static** cost matrix
(cheap); EM re-runs assignment under tentative merged-centroid
weights (expensive). Running 2-opt first lets EM start from a
locally-optimal seed.

Naming-history note
-------------------
"Direction D" / "step 3.5" are STRATEGY_NEXT labels. The current
plugin architecture has no direction-letter taxonomy; new prose drops
the labels. Existing log lines / Trackio keys preserved for dashboard
back-compat.

Circular-import note: this module imports only ``pipeline.base`` and
``pipeline.context``, neither of which imports ``stage2_reap_ream``.
The monolith re-imports ``_two_opt_refine`` so external callers (tests,
the ``MOE_STAGE2_LEGACY_LOOP=1`` legacy-loop path, ``LegacyAdapter``)
keep their import paths working unchanged.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


def _two_opt_refine(
    assignment: list[int],
    cost: np.ndarray,
    max_group_cap: int,
) -> list[int]:
    """Greedy + one 2-opt local-refinement loop (spec §5 step 3.5).

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

    # max_group_cap <= 0 → uncapped; use a conservative oversized sentinel
    # (``n_children`` — every child could pile into one group at worst) so the
    # capacity guards become no-ops without special-casing every check. Not
    # truly infinite, just large enough that it cannot bind in this routine.
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
                        "2-opt: group size invariant violated before a "
                        "size-neutral swap (expected both <= cap)"
                    )
                    result[i] = g_j
                    result[j] = g_i
                    g_i = g_j
                    improved = True

    return result


class TwoOptRefinePlugin:
    """Plugin home for the 2-opt local-refinement pass (LIVE, S2-9).

    LIVE: the first link of the ``refine_assignment`` chain (two-opt THEN EM).
    The orchestrator's ``_run_assignment`` calls this plugin's
    ``refine_assignment`` inside the bump loop, ahead of ``EmRefinePlugin`` and
    the dead-fallback ``LegacyAdapter``.

    Config gate: enabled iff ``stage2_reap_ream.two_opt_refine`` is truthy.
    ``two_opt_refine`` is a plain bool (default ``False``).

    The greedy-only guard (``two_opt_refine`` is ignored unless
    ``assignment_solver == "greedy"``) is applied INSIDE ``refine_assignment``,
    NOT in ``is_enabled`` — a non-greedy solver still enables the plugin so the
    ``elif`` warning fires once per layer, byte-identical to the old
    ``LegacyAdapter.refine_assignment``.
    """

    name = "two_opt_refine"
    paper = "Project-original; see module docstring."
    config_key = "stage2_reap_ream.two_opt_refine"
    # Two-opt operates purely on the assignment list, the cost matrix and the
    # per-centroid cap (all passed as call args) — it reads no ctx slots.
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        two_opt_refine: bool = False,
        assignment_solver: str = "greedy",
        max_group_cap: int = 0,
    ) -> None:
        self.two_opt_refine = two_opt_refine
        self.assignment_solver = assignment_solver
        self.max_group_cap = max_group_cap

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.two_opt_refine`` is truthy.

        Gates on ``two_opt_refine`` ALONE — the greedy-only check lives in
        ``refine_assignment`` so the non-greedy ``elif`` warning still fires.
        """
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        return bool(s2.get("two_opt_refine"))

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def refine_assignment(
        self, ctx: PipelineContext, asg: Any, delta: Any
    ) -> tuple[Any, Any, dict] | None:
        """Chain link 1 — 2-opt local refinement (LIVE, S2-9).

        Verbatim lift of the 2-opt block (greedy-only guard + ``elif``
        warning) from the old ``LegacyAdapter.refine_assignment``. Returns
        ``(asg, delta, {"two_opt": True})`` — the info dict carries NO
        ``em_rounds`` key (only EM owns that count).

        Always returns a non-``None`` tuple, including when
        ``two_opt_refine`` is false at the instance level (``asg``/``delta``
        unchanged) — the refine chain skips ``None`` returns, so a non-``None``
        passthrough is uniform and harmless. In production the plugin is
        excluded by :meth:`is_enabled` before a false-flag instance is ever
        constructed; the false-flag path is exercised only by tests.
        """
        if self.two_opt_refine and self.assignment_solver == "greedy":
            asg = _two_opt_refine(asg, delta, self.max_group_cap)
        elif self.two_opt_refine:
            log.warning(
                "two_opt_refine=true is ignored: it only applies to the "
                "greedy assignment solver, but assignment_solver=%r.",
                self.assignment_solver,
            )
        return asg, delta, {"two_opt": True}
