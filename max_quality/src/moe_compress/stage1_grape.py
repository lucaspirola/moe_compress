"""Stage 1 — GRAPE non-uniform per-layer expert budgets (fused-experts-aware).

Implements GRAPE Algorithm 1 (2604.06542, §3.3): entropy-aware greedy merge
with restart. The algorithm iteratively merges the most-similar expert pair
from the most-redundant layer, subject to an entropy constraint that prevents
over-pruning any single layer.

The pairwise distance matrices D^l use a pluggable metric (cosine/MSE/CKA on
flattened weight vectors, per paper §3.2 which explicitly allows "CKA, MSE,
or other similarity measures").
"""
from __future__ import annotations

import logging
import math
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

    log.info("Stage 1: computing pairwise distance matrices over %d MoE layers", len(moe_layers))

    # Step 1: Compute pairwise distance matrices D^l for all MoE layers.
    D_matrices: dict[int, torch.Tensor] = {}
    per_layer_counts: dict[int, int] = {}
    for k, ref in enumerate(moe_layers):
        log.info("Stage 1 distance matrix: layer %d/%d (idx=%d)",
                 k + 1, len(moe_layers), ref.layer_idx)
        D = _pairwise_distance_matrix(ref, metric=s1["similarity_metric"])
        D_matrices[ref.layer_idx] = D
        per_layer_counts[ref.layer_idx] = ref.num_routed_experts

    # Step 2: Run GRAPE Algorithm 1 (entropy-aware greedy merge).
    global_budget = decomposition.global_expert_budget
    min_experts = s1["min_experts_per_layer"]
    blacklist = decomposition.blacklisted_experts
    gamma = float(s1.get("entropy_tolerance", 0.1))  # γ in paper Eq. 10

    budgets = _grape_greedy_merge(
        D_matrices=D_matrices,
        global_budget=global_budget,
        per_layer_counts=per_layer_counts,
        min_experts=min_experts,
        blacklist=blacklist,
        gamma=gamma,
        early_bonus=s1["early_layer_bonus"],
        early_bonus_depth=s1["early_layer_bonus_depth"],
        late_bonus=s1.get("late_layer_bonus", 0),
        late_bonus_depth=s1.get("late_layer_bonus_depth", 0),
    )

    # Compute redundancies for logging (Eq. 2: mean off-diagonal).
    redundancies: dict[int, float] = {}
    for li, D in D_matrices.items():
        n = D.shape[0]
        if n <= 1:
            redundancies[li] = 0.0
        else:
            # Use similarity (1 - dist) for redundancy interpretation.
            sim = 1.0 - D
            off = (sim.sum() - sim.diag().sum()) / (n * (n - 1))
            redundancies[li] = float(off.item())
        _trackio_log({
            "stage1/layer_idx": li,
            "stage1/redundancy": redundancies[li],
            "stage1/budget": budgets[li],
        })

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


def _grape_greedy_merge(
    *,
    D_matrices: dict[int, torch.Tensor],
    global_budget: int,
    per_layer_counts: dict[int, int],
    min_experts: int,
    blacklist: dict[int, list[int]],
    gamma: float,
    early_bonus: int,
    early_bonus_depth: int,
    late_bonus: int,
    late_bonus_depth: int,
) -> dict[int, int]:
    """GRAPE Algorithm 1 (2604.06542, §3.3): entropy-aware greedy merge with restart.

    Returns per-layer surviving expert counts (budgets).
    """
    sorted_layers = sorted(per_layer_counts.keys())
    n_moe_layers = len(sorted_layers)
    total_experts = sum(per_layer_counts.values())

    # Initialize: each expert is its own cluster. Track cluster count per layer.
    cluster_counts: dict[int, int] = dict(per_layer_counts)

    # R^l = sum of off-diagonal distances (Eq. 11, sum form).
    R: dict[int, float] = {}
    for li in sorted_layers:
        D = D_matrices[li]
        n = D.shape[0]
        if n <= 1:
            R[li] = 0.0
        else:
            R[li] = float((D.sum() - D.diag().sum()).item())

    # Working copies of distance matrices (we zero out merged pairs).
    D_work: dict[int, np.ndarray] = {
        li: D_matrices[li].cpu().numpy().copy() for li in sorted_layers
    }

    # Floor constraints: can't go below min_experts or blacklist size.
    floors: dict[int, int] = {
        li: max(min_experts, len(blacklist.get(li, [])))
        for li in sorted_layers
    }
    # Also apply early/late layer bonuses as floor increases.
    for idx, li in enumerate(sorted_layers):
        if early_bonus_depth > 0 and idx < early_bonus_depth:
            floors[li] = max(floors[li], per_layer_counts[li] - max(0, per_layer_counts[li] - floors[li] - early_bonus))
        if late_bonus_depth > 0 and idx >= n_moe_layers - late_bonus_depth:
            floors[li] = max(floors[li], per_layer_counts[li] - max(0, per_layer_counts[li] - floors[li] - late_bonus))

    # Initial entropy: uniform distribution over layers.
    def _entropy(counts: dict[int, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in counts.values()])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    E_init = _entropy(cluster_counts)
    E_hat = E_init * (1.0 - gamma)  # Eq. 10: entropy threshold

    frozen: set[int] = set()
    current_total = sum(cluster_counts.values())

    log.info("GRAPE: global_budget=%d, current_total=%d, gamma=%.2f, E_hat=%.4f",
             global_budget, current_total, gamma, E_hat)

    max_iterations = current_total * n_moe_layers  # safety bound
    for iteration in range(max_iterations):
        if current_total <= global_budget:
            break

        # Restart: if all layers frozen, unfreeze all.
        if len(frozen) >= n_moe_layers:
            frozen.clear()
            log.info("GRAPE iter %d: all layers frozen → restart", iteration)

        # Pick l* = argmax R^l among unfrozen layers that can still donate.
        best_layer = None
        best_R = -1.0
        for li in sorted_layers:
            if li in frozen:
                continue
            if cluster_counts[li] <= floors[li]:
                continue  # can't prune further
            if R[li] > best_R:
                best_R = R[li]
                best_layer = li

        if best_layer is None:
            log.warning("GRAPE: no unfrozen layer can donate — stopping at %d (target %d)",
                        current_total, global_budget)
            break

        # Pick (i*, j*) = argmax D^{l*}_{ij} (most similar pair).
        D_l = D_work[best_layer]
        n = D_l.shape[0]
        # Zero diagonal to avoid self-merge.
        np.fill_diagonal(D_l, 0.0)
        flat_idx = int(np.argmax(D_l))
        i_star, j_star = divmod(flat_idx, n)

        if D_l[i_star, j_star] <= 0:
            # No more mergeable pairs in this layer.
            frozen.add(best_layer)
            continue

        # Merge: zero out the merged pair, update R^l.
        contribution = float(D_l[i_star, j_star]) + float(D_l[j_star, i_star])
        R[best_layer] -= contribution
        D_l[i_star, j_star] = 0.0
        D_l[j_star, i_star] = 0.0
        # Also zero out all interactions with j_star (it's absorbed into i_star).
        R[best_layer] -= float(D_l[j_star, :].sum() + D_l[:, j_star].sum())
        D_l[j_star, :] = 0.0
        D_l[:, j_star] = 0.0

        cluster_counts[best_layer] -= 1
        current_total -= 1

        # Entropy check: if layer entropy drops below threshold, freeze it.
        E_current = _entropy(cluster_counts)
        if E_current < E_hat:
            frozen.add(best_layer)

    log.info("GRAPE: converged at %d total experts (target %d) after %d iterations",
             current_total, global_budget, min(iteration + 1, max_iterations))

    if current_total > global_budget:
        log.warning(
            "GRAPE: could not reach global_budget=%d (achieved=%d). "
            "Consider reducing early/late_bonus, min_experts_per_layer, "
            "or the target reduction ratio.",
            global_budget, current_total,
        )

    return dict(cluster_counts)


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



