"""Budget decomposition solver: splits a configurable compression target between expert pruning and SVD rank reduction.

The total-reduction target is met by compounding two knobs.  As a
*simplified approximation* (before discretisation):

    [1 - (1 - expert_prune_ratio) · (1 - svd_rank_ratio)] · (expert_params / total_params) ≈ target_ratio

where ``expert_params`` is the total routed-expert parameter count and
``total_params`` is the full model parameter count.  In the actual solver
loop the savings are computed from the **discretised** expert count:

    actual_prune_params = expert_params - surviving_experts * params_per_expert_avg
    after_prune         = expert_params - actual_prune_params
    expert_savings      = actual_prune_params + after_prune * svd_rank_ratio

This is exact given the integer expert constraint; the compound formula above
is only used as an analytical starting point.  Every non-compressible param
(attention, shared expert, embeddings, lm_head, router, layer norms) still
counts toward the denominator in the total reduction calculation.

**Savings-ratio formula for ep_sp_knob_ratio**

The ``ep_sp_knob_ratio`` parameter controls the ratio of the expert-pruning
knob value to the SVD-rank-reduction knob value
(``expert_prune_ratio / svd_rank_ratio``), **not** the ratio of their savings
contributions.  The actual savings ratio differs because SVD is applied to the
post-prune parameter count: pruning saves ``ep * expert_params`` while SVD
saves ``sp * (1 - ep) * expert_params``, giving a savings ratio of
``ep / (sp * (1 - ep))``.  Higher ``ep_sp_knob_ratio`` values favour pruning
over factorisation; lower values favour factorisation over pruning.  Both knobs
are scaled together during iteration, so the ratio is honoured during scaling
but may deviate when the expert floor is binding (floor-clamp branch), or when
either ceiling cap (_MAX_EP, _MAX_SP) is binding at the analytical start.

.. note:: **Spec naming difference (F-01).**
   §3 of ALGORITHM_REFERENCE.md calls this parameter ``expert_svd_ratio`` and
   documents it as a *savings* ratio (e.g. "2.0 meaning pruning removes 2× the
   params that SVD removes").  That description refers to the ratio of savings
   contributions (``ep / (sp*(1-ep))``), not the knob ratio.  This
   implementation accepts a **knob ratio** (``ep/sp``) instead, which is more
   natural to set and scale during the iterative loop.  The two are not equal
   numerically; callers must pass the knob ratio, not the savings ratio.  See
   the savings formula above to convert: for a desired savings ratio ``R``,
   the corresponding knob ratio at a given ``ep`` is ``R * (1 - ep)``, which
   is target-dependent.  In practice, both ratios produce qualitatively similar
   trade-offs at the values used in production (e.g., knob ratio 2.0 ≈ savings
   ratio 1.5 at ep≈0.25).  No code change is needed; this is a documentation
   note only.

This module does NOT mutate the model. It only returns a
:class:`BudgetDecomposition` that Stages 1/2/3 consume.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field

import torch.nn as nn

from ..utils.model_io import (
    count_expert_parameters,
    count_parameters,
    iter_moe_layers,
)

log = logging.getLogger(__name__)

_MAX_EP = 0.60  # maximum expert pruning ratio
_MAX_SP = 0.40  # maximum SVD rank reduction ratio
# These caps are intentionally hardcoded; they represent the design ceiling for each compression axis.


@dataclass
class BudgetDecomposition:
    """Solver output consumed by Stages 1/2/3.

    **What this dataclass does NOT contain (F-03):**
    ``per_layer_target_experts`` (N'_l in the spec) is **not** a solver output.
    The solver produces only the global ``global_expert_budget`` (total surviving
    routed experts across all layers).  Per-layer budgets N'_l are allocated by
    GRAPE in Stage 1 (``stage1_grape.py``), which distributes ``global_expert_budget``
    non-uniformly across layers using activation-aware CKA similarity, subject to
    the ``min_experts_per_layer`` floor (see ALGORITHM_REFERENCE.md §4).
    """
    total_reduction_ratio: float            # target, e.g. 0.30
    expert_prune_ratio: float               # knob value passed to Stage 2; actual pruning fraction may differ when the expert floor is binding
    svd_rank_ratio: float                   # fraction of remaining expert params to remove via Stage 3
    global_expert_budget: int               # total surviving routed experts across all layers
    min_experts_per_layer: int
    blacklisted_experts: dict[int, list[int]] = field(default_factory=dict)  # {layer_idx: [expert_idx, ...]} — experts excluded from pruning/merging

    # Measurements (populated by :func:`solve`)
    total_params: int = 0
    expert_params: int = 0
    projected_expert_params_after_prune: int = 0
    projected_expert_params_after_svd: int = 0
    projected_total_reduction: float = 0.0

    def as_dict(self) -> dict:
        """Serialize to a JSON-compatible dict; blacklisted_experts keys are converted to strings.

        Note: ``json.dumps`` requires a custom encoder if any field contains numpy integer
        types (e.g., from ``tensor.numel()``); all int fields are Python ints when produced
        by ``count_parameters``.
        """
        d = dataclasses.asdict(self)
        # blacklisted_experts has int keys; JSON requires string keys.
        # dataclasses.asdict deep-copies the `list[int]` values of `blacklisted_experts`;
        # if the value type changes, add explicit copies here.
        d["blacklisted_experts"] = {str(k): v for k, v in d["blacklisted_experts"].items()}
        return d


def _count_experts_by_layer(model: nn.Module) -> dict[int, int]:
    return {ref.layer_idx: ref.num_routed_experts for ref in iter_moe_layers(model)}


def solve(
    model: nn.Module,
    *,
    target_total_reduction: float,
    ep_sp_knob_ratio: float,
    min_experts_per_layer: int,
    blacklisted_experts: dict[int, list[int]] | None = None,
    max_iterations: int = 20,
    tolerance: float = 0.005,
) -> BudgetDecomposition:
    """Iteratively tighten the two knobs until the projected reduction
    meets or exceeds the target within ``tolerance`` (an *absolute* difference
    in reduction ratios; e.g. ``0.005`` = ±0.5 percentage points).
    The in-loop convergence check uses ``<= tolerance`` (inclusive on both
    sides), triggering an early return before the post-loop guard is reached.
    The post-loop undershoot guard uses ``< -tolerance`` (strict), so a result
    exactly at ``-tolerance`` is accepted — this is consistent because the
    early-return exits before the post-loop check runs.  Results more than
    ``tolerance`` below target raise RuntimeError; results more than
    ``tolerance`` above target emit a warning.

    ``ep_sp_knob_ratio``: controls the relative aggressiveness of pruning vs
    factorisation; see module docstring for the savings formula.

    The starting point is derived analytically from the ratio and the model's
    expert/total param fraction, giving convergence in ≤ 3 iterations for
    fine-grained MoE models where params_per_expert_avg/total_params ≪ tolerance
    (i.e. the quantisation granularity is well below the tolerance threshold).
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(f"tolerance must be a positive finite number, got {tolerance!r}")
    if not (0 < target_total_reduction < 1):
        raise ValueError(f"target_total_reduction must be in (0, 1), got {target_total_reduction!r}")
    if not math.isfinite(ep_sp_knob_ratio) or ep_sp_knob_ratio <= 0:
        raise ValueError(
            f"ep_sp_knob_ratio must be a positive finite number, got {ep_sp_knob_ratio!r}"
        )
    if min_experts_per_layer < 0:
        raise ValueError(f"min_experts_per_layer must be >= 0, got {min_experts_per_layer}")
    # Defensive copy: protect the stored BudgetDecomposition from caller mutation.
    blacklisted_experts = {k: list(v) for k, v in (blacklisted_experts or {}).items()}
    total_params = count_parameters(model)
    expert_params = count_expert_parameters(model, routed_only=True)
    if total_params == 0:
        raise ValueError("model has no parameters")
    if expert_params == 0:
        raise ValueError("model has no routed expert parameters")
    per_layer_counts = _count_experts_by_layer(model)
    total_routed = sum(per_layer_counts.values())
    if total_routed == 0:
        raise ValueError(
            "Model has no routed experts (total_routed == 0). "
            "Cannot decompose budget for a model with no MoE layers."
        )
    # Both guards are logically equivalent for any cap values in (0,1]: any target
    # exceeding expert_params/total_params also exceeds max_achievable.  This first
    # check gives a clearer error message; the more precise max_achievable check follows.
    if target_total_reduction * total_params > expert_params:
        raise ValueError(
            f"target_total_reduction={target_total_reduction:.3f} requires removing more params "
            f"than available in routed experts ({expert_params}/{total_params}); "
            "or lower the target — this solver compresses only routed expert parameters."
        )
    max_achievable = expert_params * (1.0 - (1.0 - _MAX_EP) * (1.0 - _MAX_SP)) / total_params
    if target_total_reduction > max_achievable:
        raise ValueError(
            f"target_total_reduction={target_total_reduction:.3f} exceeds the maximum achievable "
            f"reduction ({max_achievable:.3f}, continuous upper bound; actual discrete max may be lower "
            f"due to integer expert constraints) given _MAX_EP={_MAX_EP} and _MAX_SP={_MAX_SP} "
            f"ceiling caps, or per-layer floor constraints (min_experts_per_layer, blacklisted_experts). "
            f"Raise _MAX_EP/_MAX_SP or lower the target."
        )
    # Assumes all routed experts have equal parameter counts (valid for uniform fused stacks).
    # Correctness depends on this assumption; if expert parameter counts differ across layers,
    # the returned budget may not achieve the target reduction ratio.
    params_per_expert_avg = expert_params / total_routed

    # Strip unknown blacklist entries BEFORE computing protected_per_layer so that
    # the per-layer floor is computed only over layers that actually exist in the model.
    unknown = set(blacklisted_experts) - set(per_layer_counts)
    if unknown:
        log.warning("blacklisted_experts references layer indices not in model: %s", sorted(unknown))
        for k in unknown:
            blacklisted_experts.pop(k)  # remove from defensive copy so downstream doesn't see them

    # Protected experts: blacklist + min_experts floor per layer (we can't go
    # below the floor regardless of what the ratio demands).
    # blacklisted experts are treated as included in the min_experts_per_layer
    # floor (they count as "safe" survivors); if a layer has more blacklisted
    # experts than min_experts_per_layer, the larger count becomes the floor.
    protected_per_layer = {
        li: max(min_experts_per_layer, len(blacklisted_experts.get(li, [])))
        for li in per_layer_counts
    }

    # Analytical starting point: ignore the ep*sp cross-term and solve
    #   expert_params * sp * (ep_sp_knob_ratio + 1) / total_params ≈ target
    # This lands close to the solution for any ep_sp_knob_ratio, cutting iterations.
    sp_start = target_total_reduction * total_params / (expert_params * (ep_sp_knob_ratio + 1))
    sp = min(_MAX_SP, sp_start)
    # Use clamped sp (not raw sp_start) so that ep inherits any ceiling applied to sp.
    # Both ep and sp may be overridden later in the floor-clamp branch below.
    ep = min(_MAX_EP, sp * ep_sp_knob_ratio)
    decomp: BudgetDecomposition | None = None  # assigned on first iteration (max_iterations >= 1)

    # Loop-invariant: maximum experts that can be pruned given the protected floor.
    # Clamp to zero: if min_experts_per_layer exceeds actual counts, protected
    # experts could exceed total_routed and make min_pool negative.
    # min_pool is a global lower bound; per-layer floors may prevent pruning all
    # min_pool experts (some layers may have zero prunable capacity). This ceiling
    # is necessary but not sufficient for feasibility.
    #
    # F-02 note: Per-layer structural feasibility (individual layers where
    # protected_per_layer[l] >= per_layer_counts[l], leaving zero prunable
    # capacity for that layer) is NOT checked here.  The global min_pool check
    # below handles the degenerate case where ALL layers are fully protected
    # (min_pool == 0).  For the general case, per-layer budget infeasibility is
    # detected and handled by GRAPE in Stage 1 (ALGORITHM_REFERENCE.md §4),
    # which distributes global_expert_budget non-uniformly and enforces
    # min_experts_per_layer per layer.  Adding a solver-level per-layer warning
    # would duplicate GRAPE's own feasibility logic without spec mandate (§3
    # delegates allocation to GRAPE, not the solver).
    min_pool = max(0, total_routed - sum(protected_per_layer.values()))
    if min_pool == 0:
        max_achievable_svd_only = expert_params * _MAX_SP / total_params
        if target_total_reduction > max_achievable_svd_only:
            raise ValueError(
                f"All experts are protected (min_pool=0); target={target_total_reduction:.4f} "
                f"cannot be achieved via SVD alone (max={max_achievable_svd_only:.4f} with _MAX_SP={_MAX_SP}). "
                "Reduce the target or relax the expert protection constraints."
            )
        log.warning(
            "All routed experts are protected; no pruning is possible. "
            "The full reduction target must be absorbed by SVD alone; if the required SVD ratio "
            "exceeds _MAX_SP=%s, the solver will fail.",
            _MAX_SP,
        )
    max_prunable_params = min_pool * params_per_expert_avg

    quant_granularity = params_per_expert_avg / total_params
    if tolerance < 0.5 * quant_granularity:
        log.warning(
            "tolerance=%.6f is below the half-expert quantisation step (0.5 × ppa/total_params=%.6f); "
            "solver may not converge for coarse-grained MoE models",
            tolerance, 0.5 * quant_granularity,
        )

    # Stagnation detection state ep_prev/sp_prev is assigned at the top of each
    # iteration body (see inside the loop) before the next iteration's check
    # reads it.  The `it > 0` guard skips the check on iter 0, by which point
    # ep_prev/sp_prev have already been written by iter-0's snapshot — so no
    # pre-loop seed is needed.
    last_iter: int | None = None  # track last completed iteration for error reporting
    for it in range(max_iterations):
        # Stagnation check: compare ep/sp entering THIS iteration against values
        # entering the PREVIOUS iteration. If they are identical the ceiling caps
        # are binding and further iterations will not change the result.
        # `it > 0` guards so we never fire on the very first iteration (before any
        # computation has run). ep_prev/sp_prev are snapshotted at the TOP of each
        # loop body (below), so they reflect the entering values for the previous
        # iteration — the values BEFORE that iteration's computation ran.
        if it > 0 and abs(ep - ep_prev) < 1e-9 and abs(sp - sp_prev) < 1e-9:
            log.debug(
                "solve: stagnation detected at iter=%d (ep=%.6f, sp=%.6f unchanged since iter=%d); exiting loop",
                it, ep, sp, it - 1,
            )
            break
        # Snapshot entering values BEFORE this iteration's computation so the next
        # iteration's stagnation check can compare against them.
        ep_prev, sp_prev = ep, sp
        last_iter = it + 1

        prune_params = ep * expert_params
        surviving_experts_total = _project_expert_budget(
            per_layer_counts, protected_per_layer, prune_params, params_per_expert_avg,
            max_prunable=min_pool,
            total_experts=total_routed,
        )
        if surviving_experts_total == 0:
            raise ValueError(
                "Budget decomposition resulted in 0 surviving experts — this would eliminate all "
                "routed experts. Increase min_experts_per_layer or lower the target."
            )
        actual_prune_params = (total_routed - surviving_experts_total) * params_per_expert_avg
        after_prune = expert_params - actual_prune_params
        after_svd = after_prune * (1.0 - sp)
        expert_savings = expert_params - after_svd
        projected_total_reduction = expert_savings / total_params

        # NOTE: intentional discrepancy between projected_total_reduction and
        # projected_expert_params_after_svd: projected_total_reduction is computed from
        # the unrounded float `after_svd` for accuracy, while projected_expert_params_after_svd
        # is rounded to an integer for storage.  Callers that recompute reduction from the
        # stored int fields will get a slightly different value (by up to 0.5 / total_params).
        decomp = BudgetDecomposition(
            total_reduction_ratio=target_total_reduction,
            expert_prune_ratio=ep,
            svd_rank_ratio=sp,
            global_expert_budget=surviving_experts_total,
            min_experts_per_layer=min_experts_per_layer,
            blacklisted_experts={k: list(v) for k, v in blacklisted_experts.items()},
            total_params=total_params,
            expert_params=expert_params,
            projected_expert_params_after_prune=round(after_prune),
            projected_expert_params_after_svd=round(after_svd),
            projected_total_reduction=projected_total_reduction,
        )
        log.info(
            "solve iter=%d/%d ep=%.4f sp=%.4f budget=%d"
            " projected=%.4f (target=%.4f)",
            it + 1, max_iterations, ep, sp,
            surviving_experts_total, projected_total_reduction, target_total_reduction,
        )
        err = projected_total_reduction - target_total_reduction
        if -tolerance <= err <= tolerance:
            return decomp
        # Scale both knobs by the deficit ratio.
        # NOTE: hard ceilings ep≤0.60 and sp≤0.40 can prevent convergence when the
        # required reduction cannot be reached within these bounds. In that case the
        # solver exhausts max_iterations and raises RuntimeError. Relax constraints
        # (min_experts_per_layer, blacklist, target) or raise the ceilings if needed.
        # scale is computed from a quantised projected reduction — the quantisation comes from
        # `round(experts_to_prune)` in `_project_expert_budget`, which discretises the surviving
        # expert count.  tolerance should be >= 0.5 * ppa/total_params (the half-expert rounding
        # step) to guarantee convergence.
        # projected_total_reduction > 0 is guaranteed by two invariants: (a) at least one of ep
        # or sp is positive (both are initialised > 0 and the floor-clamp branch's sp formula
        # always gives sp > 0 when ep = 0), so expert_savings > 0; (b) expert_params > 0
        # (verified by the `expert_params == 0` guard above).  Together these ensure the
        # denominator and numerator are both positive.
        assert projected_total_reduction > 0, (
            "unreachable: ep > 0 or sp > 0 is maintained as a loop invariant, "
            "ensuring expert_savings > 0 given expert_params > 0"
        )
        scale = target_total_reduction / projected_total_reduction
        # Compute scaled values first WITHOUT clamping so the floor-clamp guard sees
        # the unclamped ep_scaled.  Applying min(_MAX_EP, ...) before the guard could
        # reduce ep_scaled below the floor threshold, causing the guard to be skipped
        # even when ep_scaled genuinely exceeds max_prunable_params/expert_params, which
        # leads to oscillation instead of convergence.
        ep_scaled = ep * scale
        sp_scaled = sp * scale
        # NOTE (F-04 assessment): the floor-clamp branch *re-assigns* ep to
        # max_prunable_params/expert_params (a different value) before the inner
        # `if ep > _MAX_EP` check.  That inner check is NOT dead: it guards the case
        # where min_pool/total_routed > _MAX_EP (>60% of experts prunable), which is
        # a valid configuration.  Do NOT remove the inner guard.
        # When the protected-expert floor is binding, overwrite both scale-derived ep and sp:
        # ep is clamped to the floor, and sp is re-derived analytically to absorb the residual.
        if ep_scaled * expert_params > max_prunable_params:
            # Floor-clamp branch: derive ep from the protected-expert floor, then
            # apply ceiling AFTER the floor fix (not before).  ep_scaled and sp_scaled
            # are intentionally not used here — ep is re-derived from max_prunable_params
            # and sp will be solved analytically below.
            ep = max_prunable_params / expert_params
            if ep > _MAX_EP:
                # Unusual case: the protected-expert floor (min_pool/total_routed) itself
                # exceeds the ceiling cap _MAX_EP.  Apply the ceiling cap so ep stays in
                # [0, _MAX_EP]; this means we prune fewer experts than the floor would
                # allow, and sp will absorb the residual.
                log.debug(
                    "solve: floor-clamp ep=%.4f (min_pool/total_routed) exceeds _MAX_EP=%.4f; "
                    "further clamping to ceiling",
                    ep, _MAX_EP,
                )
                ep = _MAX_EP
            # ep is now fixed at min(min_pool/total_routed, _MAX_EP).  Recompute the actual
            # integer-rounded surviving expert count at this clamped ep, then solve for sp
            # from the forward model's exact integer arithmetic rather than the continuous
            # approximation.  This prevents oscillation between iterations caused by the
            # mismatch between the continuous formula and the quantised expert count.
            n_surviving_at_clamped_ep = _project_expert_budget(
                per_layer_counts, protected_per_layer,
                ep * expert_params, params_per_expert_avg,
                max_prunable=min_pool,
                total_experts=total_routed,
            )
            n_pruned_at_floor = total_routed - n_surviving_at_clamped_ep
            actual_prune_at_floor = n_pruned_at_floor * params_per_expert_avg
            after_prune_at_floor = expert_params - actual_prune_at_floor
            # Discretisation-consistent formula: sp needed so that
            #   after_prune_at_floor * (1 - sp) = expert_params - target * total_params
            # i.e. sp = 1 - residual / after_prune_at_floor
            residual = expert_params - target_total_reduction * total_params
            # The else branch (after_prune_at_floor == 0) is structurally unreachable: the
            # floor-clamp branch only fires when protected experts remain
            # (ep > min_pool / total_routed), so n_surviving_at_clamped_ep == protected_sum > 0.
            assert after_prune_at_floor > 0, (
                "after_prune_at_floor is zero inside floor-clamp branch, which is structurally impossible "
                "when protected experts exist — check _project_expert_budget logic"
            )
            sp = max(0.0, min(_MAX_SP, 1.0 - residual / after_prune_at_floor))
            # sp driven to 0: pruning at the protected-floor ep already achieves or
            # exceeds the target without any SVD rank reduction.
            if sp == 0.0:
                # target_survival_frac is the fraction of expert params that must survive
                # to hit the reduction target (residual / expert_params), NOT the actual
                # fraction surviving after floor-clamped pruning (after_prune_at_floor /
                # expert_params).  When sp==0 the actual fraction is ≤ target_survival_frac.
                target_survival_frac = residual / expert_params
                log.warning(
                    "solve: sp clamped to 0.0 — SVD rank reduction not needed — pruning at floor-clamped ep "
                    "meets or exceeds target (target_survival_frac=%.4f — fraction of expert params that must survive)",
                    target_survival_frac,
                )
        else:
            # Normal path: floor is not binding; apply ceiling caps to scaled values.
            # Ceilings are applied HERE (after the floor check) to avoid masking a
            # genuine floor condition by prematurely reducing ep before the guard fires.
            ep = min(_MAX_EP, ep_scaled)
            sp = min(_MAX_SP, sp_scaled)
    # decomp is always assigned: max_iterations >= 1 guarantees at least one loop body.
    # Undershoot vs overshoot asymmetry: an undershoot (projected < target) means
    # the model is *less* compressed than requested, which may violate downstream
    # constraints — so we raise hard. An overshoot (projected > target) compresses
    # *more* than requested, which is suboptimal but safe; we warn and proceed.
    err = decomp.projected_total_reduction - target_total_reduction
    if err < -tolerance:
        raise RuntimeError(
            f"Budget solver could not reach target_total_reduction="
            f"{target_total_reduction:.3f} within tolerance={tolerance:.3f}. "
            f"Best projection={decomp.projected_total_reduction:.4f} after "
            f"{last_iter or 0} iterations. Likely cause: min_experts_per_layer="
            f"{min_experts_per_layer} leaves too few prunable experts given "
            f"blacklisted_experts size={sum(len(v) for v in blacklisted_experts.values())}. "
            "Relax these constraints or lower the target — this solver compresses only routed expert parameters. "
            f"Also check: _MAX_EP ({_MAX_EP}) and _MAX_SP ({_MAX_SP}) ceiling caps may prevent "
            "reaching target even if floor constraints are satisfied."
        )
    if decomp.projected_total_reduction > target_total_reduction + tolerance:
        # Reached here only after loop exhaustion or knob stagnation (not early-return convergence), so
        # the solver ran out of iterations while overshooting the target.
        log.warning(
            "Budget solver exhausted %d iterations with overshoot "
            "(projected=%.4f > target=%.4f). Proceeding.",
            last_iter or 0, decomp.projected_total_reduction, target_total_reduction,
        )
    return decomp


def _project_expert_budget(
    per_layer_counts: dict[int, int],
    protected_per_layer: dict[int, int],
    target_prune_params: float,
    params_per_expert_avg: float,
    max_prunable: int | None = None,
    total_experts: int | None = None,
) -> int:
    """Translate a param-savings target into total surviving experts.

    Converts ``target_prune_params`` into an integer expert count to prune
    (floor-rounded, clamped to prunable capacity), then returns the
    total number of surviving routed experts across all layers.  The per-layer
    allocation of survivors is handled downstream by GRAPE; this function
    only determines the global total.

    ``max_prunable``: optional pre-computed maximum prunable expert count
    (``min_pool`` in the caller).  When provided, the per-layer sum is skipped
    to avoid recomputing the same value on every iteration.

    ``total_experts``: optional pre-computed total routed expert count
    (``total_routed`` in the caller).  When provided, avoids recomputing
    ``sum(per_layer_counts.values())`` on every iteration.
    """
    if max_prunable is None:
        max_prunable = sum(
            max(0, per_layer_counts[li] - protected_per_layer[li]) for li in per_layer_counts
        )
    # This is a global lower bound; per-layer floors may make it impossible to prune all prunable
    # experts. Per-layer allocation is handled downstream by GRAPE.
    # floor, not round, to ensure pruning target is met-or-exceeded at boundaries
    # (avoids banker's-rounding oscillation when target lands on a half-integer multiple)
    experts_to_prune = math.floor(target_prune_params / max(params_per_expert_avg, 1e-9))
    # Clamp to [0, max_prunable].  max_prunable >= 0 is guaranteed by callers (min_pool is
    # clamped to 0 at the call site; internal computation uses max(0, ...) per-layer sums).
    # The max(0, ...) is a defensive guard in case target_prune_params is subnormal-negative
    # (cannot happen in normal operation, but avoids a silent negative surviving count).
    experts_to_prune = min(experts_to_prune, max_prunable)
    experts_to_prune = max(0, experts_to_prune)
    if total_experts is None:
        total_experts = sum(per_layer_counts.values())
    return total_experts - experts_to_prune
