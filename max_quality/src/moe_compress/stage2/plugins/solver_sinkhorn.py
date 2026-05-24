"""Capacitated entropy-regularized OT (Sinkhorn-Knopp) assignment.

Paper
-----
Sinkhorn-Knopp algorithm for entropy-regularized optimal transport;
Cuturi (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal
Transport". This plugin uses a log-domain Sinkhorn-Knopp with linear
ε-annealing. The linear ε-anneal is **not** in Cuturi 2013 (Algorithm
1 uses a single fixed λ); it is post-Cuturi standard numerical
practice (cf. Schmitzer 2019 "Stabilized Sparse Scaling Algorithms for
Entropy Regularized Transport Problems"; Chizat et al. 2018) (LOW-1).

This plugin is deviation D-sinkhorn-soft-assign from arXiv:2604.04356
(REAM): the project adds an OT-based alternative to the paper-faithful
descending-saliency greedy (see
:mod:`stage2.plugins.solver_greedy`). Soft-assignment via OT is then
projected to a hard assignment by argmax over real centroids per
non-centroid.

Official code
-------------
Standard Sinkhorn-Knopp pseudocode (Cuturi 2013); no upstream code to
pin. Implementation is project-original (``_assign_sinkhorn``).

Deviation: D-sinkhorn-soft-assign
---------------------------------
Solve

    min Σ T_cm · d_cm + ε · Σ T_cm log T_cm

with Sinkhorn-Knopp iterations (linear ε-anneal ``1.0 → 0.01`` over
``sinkhorn_iters`` ≈ 200), then argmax over real centroids per
non-centroid for the hard assignment.

The capacity inequality ``Σ_m T_cm ≤ C_max`` is converted to equality
via a **dummy slack child** with marginal ``n_C · C_max − n_NC`` and
uniform high cost — standard dummy-bin / unbalanced-partial-OT trick
predating Cuturi 2013 (see unbalanced-OT literature: Chizat et al.
2018 "Scaling Algorithms for Unbalanced Optimal Transport Problems";
Caffarelli & McCann 2010 "Free boundaries in optimal transport"). Cost
matrix normalized to ``[0, 1]`` before Sinkhorn iterations so ε is
scale-comparable across layers; the LP-OT optimum is exactly invariant
under positive affine transforms of the cost, while the entropic-OT
objective is invariant only up to a corresponding rescaling of ε.

This is **not** Sparsity-Constrained OT (arXiv:2209.15466), which uses
quadratic regularization with a first-order semi-dual solver and
cardinality (``‖T‖_0 ≤ k``) constraints — different scheme entirely.

Implementation detail: the STRATEGY_NEXT §5 step 4d spec frames the
construction as a dummy *centroid*; the implementation uses a dummy
*child* (rows-side dummy). The two are dual under argmax over real
centroids and produce the same hard assignment; the slack-child form
is simpler because the real-children argmax never has to filter out a
dummy column.

Sinkhorn falls back to greedy on infeasibility
(``n_C · C_max < n_NC``) with a clear warning.

LOW-3: ``iters`` is a fixed iteration count with no early-stopping on
the row-marginal residual ``‖T·1 − a‖``. This is a deliberate choice
(cost-bounded outer loop, deterministic budget); early-stopping is
left as a future-work item.

Currently opt-in (default off); gated default flip on
``A9 vs A8 ≥ +0.1 GEN-avg`` per STRATEGY_NEXT §8 ablation matrix.

Output context contract
-----------------------
Pure callable plugin (no state). Returns the layer's assignment from
the input cost matrix + scores + frequencies.

Naming-history note
-------------------
Step 3 of the REAM pipeline alternative-solver branch. Existing log
lines / Trackio keys preserved for dashboard back-compat.

Imports ``_assign_greedy`` from ``solver_greedy`` as the infeasible-slack /
no-finite-entries fallback. The monolith re-imports ``_assign_sinkhorn``.

``SinkhornSolverPlugin`` is a scaffold-only plugin (see
``solver_greedy``). Circular-import note: this module imports only ``numpy``,
``scipy``, ``solver_greedy``, ``pipeline.base`` and ``pipeline.context`` — none
of which import ``stage2_reap_ream``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.special import logsumexp

from ...pipeline.context import PipelineContext
from .solver_greedy import _assign_greedy

log = logging.getLogger(__name__)


# MEDIUM-1: distinct sentinels to disambiguate "+inf forbidden" entries from
# the dummy-slack-child kernel weight. The forbidden sentinel must be large
# enough that exp(-FORBIDDEN_KERNEL / eps) ≈ 0 even at the largest eps the
# anneal touches (eps_init = 1.0); 1e6 gives ~exp(-1e6) ≈ 0 with plenty of
# margin. The dummy-slack constant remains a moderate value so the dummy row
# absorbs leftover capacity without distorting the real-row argmax.
FORBIDDEN_KERNEL = 1e6


def _assign_sinkhorn(
    cost: np.ndarray,
    n_children: int,
    n_centroids: int,
    max_group_cap: int,
    *,
    epsilon_init: float = 1.0,
    epsilon_final: float = 0.01,
    iters: int = 200,
) -> list[int]:
    """Capacitated entropy-regularized OT via Sinkhorn-Knopp with a
    dummy-slack-child construction (spec § 5 step 4d / M9 /
    D-sinkhorn-soft-assign).

    The standard Sinkhorn-Knopp algorithm requires equality marginals on
    both sides. Our problem has demand ``n_children`` (each child needs 1)
    and supply ``n_centroids · max_group_cap`` (each centroid absorbs ≤ cap),
    so we balance by inserting one **dummy slack child** with marginal
    ``n_centroids · max_group_cap − n_children`` and uniform high cost to
    every centroid. After convergence, the dummy's mass flows to whichever
    real centroids have leftover capacity, and a simple argmax over the
    real-children rows recovers the hard assignment.

    Note: spec line 152–155 frames the construction as a *virtual centroid*
    rather than a virtual child; the two constructions are dual and produce
    the same hard assignment under argmax. The slack-child form is used
    here because it is simpler to implement: real children's argmax never
    needs to filter out a dummy column.

    Costs are normalized to ``[0, 1]`` before the Sinkhorn iterations so
    that ``epsilon`` values are independent of cost magnitude (relevant
    when post-alignment whitened residuals carry an unbounded scale). The
    LP-OT optimum is exactly invariant under positive affine transforms of
    the cost; the entropic-regularized OT objective is invariant only up
    to a redefinition of ε after scaling by ``finite_range`` (LOW-2).

    Defensive: returns ``[-1] * n_children`` for empty inputs.
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children

    if max_group_cap < 1:
        # v1 "uncapped" semantics — treat as max_group_cap = n_children so
        # the supply side has effectively unlimited capacity.
        max_group_cap = n_children

    # NITPICK-3: slack == 0 is the exact-balance case (supply == demand);
    # the dummy slack child still exists but carries zero marginal mass.
    slack = n_centroids * max_group_cap - n_children
    if slack < 0:
        log.warning(
            "_assign_sinkhorn: infeasible — n_C × C_max = %d < n_NC = %d. "
            "Falling back to greedy.",
            n_centroids * max_group_cap, n_children,
        )
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)

    finite_mask = np.isfinite(cost)
    if not finite_mask.any():
        log.warning(
            "_assign_sinkhorn: cost matrix has no finite entries — "
            "falling back to greedy."
        )
        return _assign_greedy(cost, n_children, n_centroids, max_group_cap)

    # Normalize to [0, 1] so epsilon scaling is cost-magnitude-invariant.
    finite_min = float(cost[finite_mask].min())
    finite_max = float(cost[finite_mask].max())
    finite_range = max(finite_max - finite_min, 1e-12)
    norm_cost = np.where(
        finite_mask,
        (cost - finite_min) / finite_range,
        # +∞ sentinel → very large finite value so the entry is effectively
        # forbidden but Sinkhorn-Knopp doesn't underflow exp(-inf/eps).
        # MEDIUM-1: keep this strictly larger than ``big_dummy`` so a
        # real-pair forbidden entry is *not* confused with the dummy-child
        # kernel weight; exp(-FORBIDDEN_KERNEL / eps) ≈ 0 across the anneal.
        FORBIDDEN_KERNEL,
    )

    big_dummy = 100.0  # cost of dummy slack child to every centroid

    # Expanded cost: rows 0..n_children-1 are real children, last row is dummy.
    expanded = np.zeros((n_children + 1, n_centroids), dtype=np.float64)
    expanded[:n_children, :] = norm_cost
    expanded[n_children, :] = big_dummy

    a = np.concatenate([np.ones(n_children), [float(slack)]])  # row marginals
    b = np.full(n_centroids, float(max_group_cap), dtype=np.float64)  # col marginals
    # Sanity check: balanced marginals (transportation polytope).
    assert abs(a.sum() - b.sum()) < 1e-9, (
        f"_assign_sinkhorn marginals mismatch: sum(a)={a.sum()} vs "
        f"sum(b)={b.sum()}"
    )

    log_a = np.log(np.maximum(a, 1e-30))
    log_b = np.log(np.maximum(b, 1e-30))

    # Log-domain Sinkhorn-Knopp with linear epsilon annealing.
    f = np.zeros_like(log_a)
    g = np.zeros_like(log_b)
    eps = epsilon_init
    for it in range(max(iters, 1)):
        eps = epsilon_init + (epsilon_final - epsilon_init) * (it / max(iters - 1, 1))
        log_K = -expanded / max(eps, 1e-12)
        # f_i = log_a_i - logsumexp_j(log_K_ij + g_j)
        f = log_a - logsumexp(log_K + g[np.newaxis, :], axis=1)
        # g_j = log_b_j - logsumexp_i(log_K_ij + f_i)
        g = log_b - logsumexp(log_K + f[:, np.newaxis], axis=0)

    log_K = -expanded / max(eps, 1e-12)
    log_T = f[:, np.newaxis] + log_K + g[np.newaxis, :]

    # Argmax over real centroids per real child (drop the dummy row).
    # NITPICK-1: vectorized argmax over rows (cosmetic speedup; identical
    # result to the per-row list-comprehension).
    real_log_T = log_T[:n_children, :]
    assignment = np.argmax(real_log_T, axis=1).tolist()
    # Direction B / skip-merge floor: a child whose entire cost row is +inf
    # (all candidate merges forbidden) must orphan — not be force-merged by the
    # argmax over the normalized sentinel. Match the greedy/hungarian/mcf
    # "-1 -> orphan promotion" contract so the floor holds for every solver.
    for ch in range(n_children):
        if not finite_mask[ch].any():
            assignment[ch] = -1
    return assignment


class SinkhornSolverPlugin:
    """Plugin home for the Sinkhorn assignment solver.

    LIVE (S2-8): services the solve_assignment slot when assignment_solver
    selects this solver. `is_enabled` selects this solver when
    `stage2_reap_ream.assignment_solver == "sinkhorn"`.
    """

    name = "solver_sinkhorn"
    # NITPICK-4: one-line summary; full attribution / deviation discussion /
    # ε-anneal credit (LOW-1) lives in the module docstring.
    paper = (
        "Cuturi 2013 Sinkhorn-Knopp + post-Cuturi linear ε-anneal "
        "(Schmitzer 2019) + unbalanced-OT dummy-slack child "
        "(Chizat et al. 2018). See module docstring."
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
        return str(s2.get("assignment_solver", "greedy")).lower() == "sinkhorn"

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def solve_assignment(self, ctx: PipelineContext, delta: Any) -> Any | None:
        """Slot ``solve_assignment`` — child→centroid assignment solver.

        Delegates to the shared ``_solve_for_plugin`` helper (verbatim lift of
        ``LegacyAdapter.solve_assignment``). Reaches this plugin only when
        ``registry.enabled`` kept it, i.e. ``assignment_solver == "sinkhorn"``."""
        from .solver_dispatch import _solve_for_plugin
        return _solve_for_plugin(self, ctx, delta)
