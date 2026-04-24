"""Stage 1 — GRAPE non-uniform per-layer expert budgets.

Given a global expert budget ``B`` (surviving routed experts summed over
layers), distribute it per layer using pairwise expert-weight redundancy:

    D^l_{ij} = 1 - cos(w_i, w_j)     (or MSE / CKA per config)
    R^l      = mean_{i≠j} D^l_{ij}
    R̃^l     = (R^l - min R) / (max R - min R)
    N'_l     = max(min_experts, round(B · (1 - R̃^l) / Σ_{l'} (1 - R̃^{l'})))
    + early_layer_bonus if l < early_layer_bonus_depth

``w_i`` is the flattened concatenation ``[gate_proj; up_proj; down_proj]``.
Shared expert is excluded (per config flag).

Artifact: ``stage1_budgets.json`` — ``{layer_idx: target_expert_count}``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .budget.solver import BudgetDecomposition
from .utils.model_io import (
    get_expert_matrices,
    iter_moe_layers,
    iter_routed_experts,
    save_json_artifact,
)

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
    for ref in moe_layers:
        D = _pairwise_distance_matrix(ref, metric=s1["similarity_metric"])
        # Off-diagonal mean
        n = D.shape[0]
        if n <= 1:
            redundancies[ref.layer_idx] = 0.0
            continue
        off = (D.sum() - D.diag().sum()) / (n * (n - 1))
        # Lower distance = more redundant → convert to "redundancy score" in [0,1]
        redundancy = 1.0 - float(off.item())
        redundancies[ref.layer_idx] = redundancy
        log.debug("Layer %d redundancy=%.4f", ref.layer_idx, redundancy)

    budgets = _allocate_budgets(
        redundancies=redundancies,
        global_budget=decomposition.global_expert_budget,
        per_layer_counts={ref.layer_idx: len(ref.experts) for ref in moe_layers},
        min_experts=s1["min_experts_per_layer"],
        blacklist=decomposition.blacklisted_experts,
        early_bonus=s1["early_layer_bonus"],
        early_bonus_depth=s1["early_layer_bonus_depth"],
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
    """Compute a dense ``[N, N]`` distance matrix over routed experts."""
    vecs: list[torch.Tensor] = []
    for _, expert in iter_routed_experts(layer_ref):
        mats = get_expert_matrices(expert)
        parts = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            if name in mats:
                parts.append(mats[name].weight.detach().to(torch.float32).flatten())
        if not parts:
            continue
        vecs.append(torch.cat(parts))
    if not vecs:
        return torch.zeros(0, 0)

    W = torch.stack(vecs)                             # [N, P]  may be huge for 256 experts
    if metric == "cosine":
        W = torch.nn.functional.normalize(W, dim=1)
        sim = W @ W.transpose(0, 1)                   # [N, N] in [-1, 1]
        # Distance in [0, 1] — same scale as MSE fallback.
        dist = (1.0 - sim).clamp(min=0.0, max=2.0) / 2.0
    elif metric == "mse":
        # Batched pairwise: ||a - b||² = ||a||² + ||b||² - 2 a·b
        sq = (W * W).sum(dim=1)
        dot = W @ W.transpose(0, 1)
        dist = (sq[:, None] + sq[None, :] - 2 * dot).clamp(min=0.0)
        dist = dist / (dist.max().clamp(min=1e-8))     # normalize to [0, 1]
    elif metric == "cka":
        dist = _cka_distance(W)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    return dist


def _cka_distance(W: torch.Tensor) -> torch.Tensor:
    """Approximate linear CKA distance on flattened weights."""
    n = W.size(0)
    K = W @ W.transpose(0, 1)                         # [N, N] Gram
    # Center gram
    H = torch.eye(n, device=W.device) - 1.0 / n
    Kc = H @ K @ H
    denom = torch.sqrt(Kc.diag().unsqueeze(0) * Kc.diag().unsqueeze(1)).clamp(min=1e-8)
    cka = Kc / denom
    return (1.0 - cka).clamp(min=0.0, max=1.0)


def _allocate_budgets(
    redundancies: dict[int, float],
    *,
    global_budget: int,
    per_layer_counts: dict[int, int],
    min_experts: int,
    blacklist: dict[int, list[int]],
    early_bonus: int,
    early_bonus_depth: int,
) -> dict[int, int]:
    """Convert redundancy scores into per-layer target experts."""
    # Normalize redundancy across layers
    vals = np.array([redundancies[li] for li in sorted(redundancies)])
    if vals.max() > vals.min():
        r_tilde = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        r_tilde = np.zeros_like(vals)
    inv = 1.0 - r_tilde
    total_inv = inv.sum() or 1.0

    sorted_ids = sorted(redundancies)
    budgets: dict[int, int] = {}
    for idx, li in enumerate(sorted_ids):
        proto = global_budget * (inv[idx] / total_inv)
        if li < early_bonus_depth:
            proto += early_bonus
        # Protect at least (min_experts + blacklist), but never exceed layer size
        floor = max(min_experts, len(blacklist.get(li, [])))
        ceil = per_layer_counts[li]
        budgets[li] = int(min(ceil, max(floor, round(proto))))

    # If rounding pushed us off the global budget, rebalance greedily by redundancy.
    diff = global_budget - sum(budgets.values())
    if diff != 0:
        order = sorted(sorted_ids, key=lambda l: inv[sorted_ids.index(l)], reverse=(diff > 0))
        i = 0
        while diff != 0 and i < 10 * len(order):
            li = order[i % len(order)]
            step = 1 if diff > 0 else -1
            new_val = budgets[li] + step
            floor = max(min_experts, len(blacklist.get(li, [])))
            ceil = per_layer_counts[li]
            if floor <= new_val <= ceil:
                budgets[li] = new_val
                diff -= step
            i += 1
    return budgets
