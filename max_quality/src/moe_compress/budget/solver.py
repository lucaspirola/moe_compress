"""Budget decomposition for a 30% total parameter reduction.

The total-reduction target is met by compounding two knobs:

    (1 - expert_prune_ratio) · (1 - svd_rank_ratio) ≈ (1 - target_ratio)

applied to the *compressible* parameter pool — routed experts only. Every
non-compressible param (attention, shared expert, embeddings, lm_head, router,
layer norms) still counts toward the denominator in the total reduction
calculation, so we iterate: pick initial knobs, project savings, compare to
target, bump knobs, repeat.

This module does NOT mutate the model. It only returns a
:class:`BudgetDecomposition` that Stages 1/2/3 consume.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch.nn as nn

from ..utils.model_io import (
    count_expert_parameters,
    count_parameters,
    iter_moe_layers,
    iter_routed_experts,
)

log = logging.getLogger(__name__)


@dataclass
class BudgetDecomposition:
    total_reduction_ratio: float            # target, e.g. 0.30
    expert_prune_ratio: float               # fraction of routed-expert params to remove via Stage 2
    svd_rank_ratio: float                   # fraction of remaining expert params to remove via Stage 3
    global_expert_budget: int               # total surviving routed experts across all layers
    min_experts_per_layer: int
    blacklisted_experts: dict[int, list[int]] = field(default_factory=dict)

    # Measurements (populated by :func:`solve`)
    total_params: int = 0
    expert_params: int = 0
    projected_expert_params_after_prune: int = 0
    projected_expert_params_after_svd: int = 0
    projected_total_reduction: float = 0.0

    def as_dict(self) -> dict:
        return {
            "total_reduction_ratio": self.total_reduction_ratio,
            "expert_prune_ratio": self.expert_prune_ratio,
            "svd_rank_ratio": self.svd_rank_ratio,
            "global_expert_budget": self.global_expert_budget,
            "min_experts_per_layer": self.min_experts_per_layer,
            "blacklisted_experts": {str(k): v for k, v in self.blacklisted_experts.items()},
            "total_params": self.total_params,
            "expert_params": self.expert_params,
            "projected_expert_params_after_prune": self.projected_expert_params_after_prune,
            "projected_expert_params_after_svd": self.projected_expert_params_after_svd,
            "projected_total_reduction": self.projected_total_reduction,
        }


def _count_experts_by_layer(model: nn.Module) -> dict[int, int]:
    return {
        ref.layer_idx: len(list(iter_routed_experts(ref))) for ref in iter_moe_layers(model)
    }


def solve(
    model: nn.Module,
    *,
    target_total_reduction: float,
    initial_expert_reduction: float,
    initial_svd_reduction: float,
    min_experts_per_layer: int,
    blacklisted_experts: dict[int, list[int]] | None = None,
    max_iterations: int = 20,
    tolerance: float = 0.005,
) -> BudgetDecomposition:
    """Iteratively tighten the two knobs until the projected reduction
    meets or exceeds the target within ``tolerance``.

    We keep ``expert_prune_ratio : svd_rank_ratio`` at roughly the initial
    ratio — if the initial knobs undershoot, we scale both up; if they
    overshoot, we scale both down.
    """
    blacklisted_experts = blacklisted_experts or {}
    total_params = count_parameters(model)
    expert_params = count_expert_parameters(model, routed_only=True)
    per_layer_counts = _count_experts_by_layer(model)
    num_layers = len(per_layer_counts)
    total_routed = sum(per_layer_counts.values())
    params_per_expert_avg = expert_params / max(total_routed, 1)

    # Protected experts: blacklist + min_experts floor per layer (we can't go
    # below the floor regardless of what the ratio demands).
    protected_per_layer = {
        li: max(min_experts_per_layer, len(blacklisted_experts.get(li, [])))
        for li in per_layer_counts
    }

    ep = initial_expert_reduction
    sp = initial_svd_reduction
    ratio = sp / max(ep, 1e-9)        # preserve initial split
    decomp: BudgetDecomposition | None = None

    for it in range(max_iterations):
        prune_params = ep * expert_params
        surviving_experts_total = _project_expert_budget(
            per_layer_counts, protected_per_layer, prune_params, params_per_expert_avg
        )
        actual_prune_params = expert_params - surviving_experts_total * params_per_expert_avg
        after_prune = expert_params - actual_prune_params
        after_svd = after_prune * (1.0 - sp)
        expert_savings = expert_params - after_svd
        projected_total_reduction = expert_savings / total_params

        decomp = BudgetDecomposition(
            total_reduction_ratio=target_total_reduction,
            expert_prune_ratio=ep,
            svd_rank_ratio=sp,
            global_expert_budget=surviving_experts_total,
            min_experts_per_layer=min_experts_per_layer,
            blacklisted_experts=blacklisted_experts,
            total_params=total_params,
            expert_params=expert_params,
            projected_expert_params_after_prune=int(after_prune),
            projected_expert_params_after_svd=int(after_svd),
            projected_total_reduction=projected_total_reduction,
        )
        log.info(
            "solve iter=%d ep=%.4f sp=%.4f budget=%d projected=%.4f (target=%.4f)",
            it, ep, sp, surviving_experts_total, projected_total_reduction, target_total_reduction,
        )
        err = projected_total_reduction - target_total_reduction
        if -tolerance <= err <= tolerance:
            return decomp
        # Scale both knobs by the deficit ratio. Cap at 0.6 / 0.4 hard ceilings
        # so we never try to prune half the model in one go.
        scale = target_total_reduction / max(projected_total_reduction, 1e-9)
        ep = min(0.60, ep * scale)
        sp = min(0.40, ep * ratio)
        # Also don't let ep drop below floor imposed by protected experts
        min_pool = total_routed - sum(protected_per_layer.values())
        max_prunable_params = min_pool * params_per_expert_avg
        if ep * expert_params > max_prunable_params:
            ep = max_prunable_params / expert_params
            sp = max(0.0, (1 - (1 - target_total_reduction) * total_params / (expert_params * (1 - ep))))
            sp = min(0.40, sp)

    assert decomp is not None
    # FIX (review bug #10): silent undershoot is worse than a loud failure —
    # if the solver cannot hit the requested target within tolerance, raise.
    # Caller can relax `min_experts_per_layer`, widen `max_blacklisted_per_layer`,
    # or reduce `total_reduction_ratio` to make progress.
    err = decomp.projected_total_reduction - target_total_reduction
    if err < -tolerance:
        raise RuntimeError(
            f"Budget solver could not reach target_total_reduction="
            f"{target_total_reduction:.3f} within tolerance={tolerance:.3f}. "
            f"Best projection={decomp.projected_total_reduction:.4f} after "
            f"{max_iterations} iterations. Likely cause: min_experts_per_layer="
            f"{min_experts_per_layer} leaves too few prunable experts given "
            f"blacklist size={sum(len(v) for v in blacklisted_experts.values())}. "
            "Relax these constraints or lower the target."
        )
    log.warning(
        "Budget solver converged with overshoot (projected=%.4f > target=%.4f). "
        "Proceeding.", decomp.projected_total_reduction, target_total_reduction,
    )
    return decomp


def _project_expert_budget(
    per_layer_counts: dict[int, int],
    protected_per_layer: dict[int, int],
    target_prune_params: float,
    params_per_expert_avg: float,
) -> int:
    """Translate a param-savings target into total surviving experts.

    Uniform initial distribution across layers; GRAPE will redistribute later
    but the *total* budget doesn't change.
    """
    total_prunable_experts = sum(
        per_layer_counts[li] - protected_per_layer[li] for li in per_layer_counts
    )
    experts_to_prune = math.ceil(target_prune_params / max(params_per_expert_avg, 1e-9))
    experts_to_prune = min(experts_to_prune, total_prunable_experts)
    total_experts = sum(per_layer_counts.values())
    return total_experts - experts_to_prune
