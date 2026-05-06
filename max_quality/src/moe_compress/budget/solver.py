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

    ep_prev, sp_prev = None, None  # used for stagnation detection
    last_iter: int | None = None  # track last completed iteration for error reporting
    for it in range(max_iterations):
        # Stagnation check at TOP of loop so we skip the redundant decomp build
        # when ep/sp haven't changed since the previous iteration (ceilings hit).
        if ep_prev is not None and abs(ep - ep_prev) < 1e-9 and abs(sp - sp_prev) < 1e-9:
            log.debug(
                "solve: stagnation detected at iter=%d (ep=%.6f, sp=%.6f unchanged since iter=%d); exiting loop",
                it, ep, sp, it - 1,
            )
            break
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
        actual_prune_params = expert_params - surviving_experts_total * params_per_expert_avg
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
        # expert_params > 0 is verified at line 143, so projected_total_reduction cannot be 0.
        assert projected_total_reduction > 0, "unreachable: expert_params > 0 guaranteed at entry"
        scale = target_total_reduction / projected_total_reduction
        ep = min(_MAX_EP, ep * scale)
        sp = min(_MAX_SP, sp * scale)  # scale sp independently, then clamp (avoids deriving from possibly-clamped ep)
        # When the protected-expert floor is binding, overwrite both scale-derived ep and sp:
        # ep is clamped to the floor, and sp is re-derived analytically to absorb the residual.
        if ep * expert_params > max_prunable_params:
            ep = max_prunable_params / expert_params
            if ep > _MAX_EP:
                log.debug("solve: floor-clamp ep=%.4f exceeds _MAX_EP=%.4f; clamping", ep, _MAX_EP)
                ep = _MAX_EP
            # ep is now fixed by the protected-expert floor (and additionally ceiling-clamped
            # to _MAX_EP above), so the effective ep used is min(floor_ep, _MAX_EP).  Recompute
            # the actual integer-rounded surviving expert count at this clamped ep, then solve for
            # sp from the forward model's exact integer arithmetic rather than the continuous
            # approximation.  This prevents oscillation between iterations caused by the mismatch
            # between the continuous formula and the quantised expert count.
            n_surviving_at_clamped_ep = _project_expert_budget(
                per_layer_counts, protected_per_layer,
                ep * expert_params, params_per_expert_avg,
                max_prunable=min_pool,
                total_experts=total_routed,
            )
            n_pruned_at_floor = total_routed - n_surviving_at_clamped_ep
            actual_prune_at_floor = n_pruned_at_floor * params_per_expert_avg
            after_prune_at_floor = expert_params - actual_prune_at_floor
            # ep = max_prunable_params / expert_params = min_pool / total_routed < 1.0,
            # since the floor-clamp only fires when protected experts exist (n_surviving_at_floor >= 1),
            # guaranteeing min_pool < total_routed. ep is re-clamped to _MAX_EP above.
            if ep > _MAX_EP:
                raise RuntimeError(
                    f"ep={ep:.6f} exceeds _MAX_EP={_MAX_EP}; this should be unreachable "
                    f"(ep is clamped to _MAX_EP at the floor-clamp branch above)"
                )
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
                required_survival_frac = residual / expert_params  # for the warning below
                log.warning(
                    "solve: sp clamped to 0.0 (required_survival_frac=%.4f — expert pruning alone meets the target)",
                    required_survival_frac,
                )
        # Update prev values at the END of the loop body, after all knob adjustments,
        # so the stagnation check next iteration compares against the fully-updated values.
        ep_prev, sp_prev = ep, sp

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
    (nearest-integer rounded, clamped to prunable capacity), then returns the
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
    # round, not ceil, to avoid systematic over-pruning bias
    experts_to_prune = round(target_prune_params / max(params_per_expert_avg, 1e-9))
    experts_to_prune = min(experts_to_prune, max_prunable)
    experts_to_prune = max(0, experts_to_prune)
    if total_experts is None:
        total_experts = sum(per_layer_counts.values())
    return total_experts - experts_to_prune
