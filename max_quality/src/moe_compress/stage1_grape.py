"""Stage 1 — GRAPE non-uniform per-layer expert budgets (fused-experts-aware).

Same math as before; only the weight-flattening step reads from
``ExpertMatrixBank.get(e)`` instead of per-expert ``nn.Linear.weight``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .budget.solver import BudgetDecomposition
from .utils.model_io import (
    MATRIX_NAMES,
    build_banks,
    iter_moe_layers,
    save_json_artifact,
)
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


def run(
    model,
    config: dict,
    artifacts_dir: Path,
    decomposition: BudgetDecomposition,
) -> Path:
    s1 = config["stage1_grape"]
    moe_layers = list(iter_moe_layers(model))

    log.info("Stage 1: computing layer redundancy over %d MoE layers", len(moe_layers))
    redundancies: dict[int, float] = {}
    for k, ref in enumerate(moe_layers):
        log.info("Stage 1 redundancy: layer %d/%d (idx=%d)",
                 k + 1, len(moe_layers), ref.layer_idx)
        D = _pairwise_distance_matrix(ref, metric=s1["similarity_metric"])
        n = D.shape[0]
        if n <= 1:
            redundancies[ref.layer_idx] = 0.0
            continue
        off = (D.sum() - D.diag().sum()) / (n * (n - 1))
        redundancies[ref.layer_idx] = 1.0 - float(off.item())
        _trackio_log({
            "stage1/layer": k + 1,
            "stage1/layer_idx": ref.layer_idx,
            "stage1/redundancy": redundancies[ref.layer_idx],
        })

    budgets = _allocate_budgets(
        redundancies=redundancies,
        global_budget=decomposition.global_expert_budget,
        per_layer_counts={ref.layer_idx: ref.num_routed_experts for ref in moe_layers},
        min_experts=s1["min_experts_per_layer"],
        blacklist=decomposition.blacklisted_experts,
        early_bonus=s1["early_layer_bonus"],
        early_bonus_depth=s1["early_layer_bonus_depth"],
        late_bonus=s1.get("late_layer_bonus", 0),
        late_bonus_depth=s1.get("late_layer_bonus_depth", 0),
    )
    out = {
        "per_layer_target_experts": {str(k): v for k, v in budgets.items()},
        "per_layer_redundancy": {str(k): v for k, v in redundancies.items()},
        "global_budget": sum(budgets.values()),
        "config": s1,
    }
    path = artifacts_dir / "stage1_budgets.json"
    save_json_artifact(out, path)
    log.info(
        "Stage 1 complete — per-layer budgets range=[%d..%d] mean=%.1f → %s",
        min(budgets.values()), max(budgets.values()),
        np.mean(list(budgets.values())), path,
    )
    return path


def _pairwise_distance_matrix(layer_ref, *, metric: str) -> torch.Tensor:
    banks = build_banks(layer_ref)
    vecs: list[torch.Tensor] = []
    for e in range(layer_ref.num_routed_experts):
        parts = [banks[name].get(e).detach().to(torch.float32).flatten()
                 for name in MATRIX_NAMES]
        vecs.append(torch.cat(parts))
    if not vecs:
        return torch.zeros(0, 0)
    W = torch.stack(vecs)
    if metric == "cosine":
        W = torch.nn.functional.normalize(W, dim=1)
        sim = W @ W.transpose(0, 1)
        dist = (1.0 - sim).clamp(min=0.0, max=2.0) / 2.0
    elif metric == "mse":
        sq = (W * W).sum(dim=1)
        dot = W @ W.transpose(0, 1)
        dist = (sq[:, None] + sq[None, :] - 2 * dot).clamp(min=0.0)
        dist = dist / (dist.max().clamp(min=1e-8))
    elif metric == "cka":
        n = W.size(0)
        K = W @ W.transpose(0, 1)
        H = torch.eye(n, device=W.device) - 1.0 / n
        Kc = H @ K @ H
        # Use the un-centred gram K for the HSIC normalization denominator so
        # the denominator is always PSD (K.diag() >= 0 by construction).
        # Kc.diag() can go negative for n < d after double-centering, which
        # would collapse the denominator and flip the similarity sign.
        diag_safe = K.diag().clamp(min=0.0)
        denom = torch.sqrt(diag_safe.unsqueeze(0) * diag_safe.unsqueeze(1)).clamp(min=1e-8)
        dist = (1.0 - (Kc / denom)).clamp(min=0.0, max=1.0)
        dist.fill_diagonal_(0.0)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    return dist


def _allocate_budgets(
    redundancies: dict[int, float],
    *,
    global_budget: int,
    per_layer_counts: dict[int, int],
    min_experts: int,
    blacklist: dict[int, list[int]],
    early_bonus: int,
    early_bonus_depth: int,
    late_bonus: int = 0,
    late_bonus_depth: int = 0,
) -> dict[int, int]:
    max_feasible = sum(per_layer_counts.values())
    min_feasible = sum(max(min_experts, len(blacklist.get(li, []))) for li in per_layer_counts)
    if not (min_feasible <= global_budget <= max_feasible):
        raise ValueError(
            f"global_budget={global_budget} is outside the feasible range "
            f"[{min_feasible}, {max_feasible}] given per-layer constraints. "
            f"Adjust global_expert_budget or min_experts_per_layer in the config."
        )

    vals = np.array([redundancies[li] for li in sorted(redundancies)])
    if vals.max() > vals.min():
        r_tilde = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        r_tilde = np.zeros_like(vals)
    inv = 1.0 - r_tilde
    total_inv = inv.sum() or 1.0

    sorted_ids = sorted(redundancies)
    n_moe_layers = len(sorted_ids)
    budgets: dict[int, int] = {}
    for idx, li in enumerate(sorted_ids):
        proto = global_budget * (inv[idx] / total_inv)
        if early_bonus_depth > 0 and li < early_bonus_depth:
            proto += early_bonus
        if late_bonus_depth > 0 and idx >= n_moe_layers - late_bonus_depth:
            proto += late_bonus
        floor = max(min_experts, len(blacklist.get(li, [])))
        ceil = per_layer_counts[li]
        budgets[li] = int(min(ceil, max(floor, round(proto))))

    diff = global_budget - sum(budgets.values())
    if diff != 0:
        inv_by_id = {li: inv[idx] for idx, li in enumerate(sorted_ids)}
        order = sorted(sorted_ids, key=lambda l: inv_by_id[l], reverse=(diff > 0))
        # Cap: (|diff| + 1) * len(order) guarantees convergence even when only
        # one layer can donate at a time — e.g. when early_bonus inflates the
        # initial allocation and all other layers are already at their floor.
        max_iters = (abs(diff) + 1) * len(order)
        i = 0
        while diff != 0 and i < max_iters:
            li = order[i % len(order)]
            step = 1 if diff > 0 else -1
            new_val = budgets[li] + step
            floor = max(min_experts, len(blacklist.get(li, [])))
            ceil = per_layer_counts[li]
            if floor <= new_val <= ceil:
                budgets[li] = new_val
                diff -= step
            i += 1
    if diff != 0:
        raise RuntimeError(
            f"_allocate_budgets: could not reconcile expert budget "
            f"(diff={diff} remaining). "
            f"Constraints may be infeasible: global_budget={global_budget}, "
            f"sum={sum(budgets.values())}."
        )
    return budgets
