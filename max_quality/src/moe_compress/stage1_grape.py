"""Stage 1 — Super Expert Detection + GRAPE non-uniform per-layer expert budgets.

Merged stage: a single 512-sample calibration forward pass simultaneously
collects (a) max |down_proj_output| for super expert detection and (b) expert
output representations for CKA pairwise similarity matrices, which feed GRAPE
Algorithm 1 (2604.06542, §3.3).

Super expert detection follows 2507.23279 with per-layer z-score thresholding
(deviation D1) and safety caps (deviation D2).

GRAPE uses CKA similarity (paper §3.2 explicitly allows "CKA, MSE, or other
similarity measures"). Floor constraint: num_routed_experts // 2 (deviation D5).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch

from .budget.solver import BudgetDecomposition
from .utils.activation_hooks import (
    DownProjMaxAccumulator,
    ExpertOutputAccumulator,
    instrument_experts,
    run_calibration,
)
from .utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from .utils.model_io import (
    MATRIX_NAMES,
    build_banks,
    iter_moe_layers,
    save_json_artifact,
)
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    decomposition: BudgetDecomposition,
    *,
    device=None,
) -> tuple[Path, Path]:
    """Run the merged Stage 1: SE detection + GRAPE budget allocation.

    Returns (blacklist_path, budgets_path).
    """
    s1 = config["stage1_grape"]
    cal = config["calibration"]
    moe_layers = list(iter_moe_layers(model))
    n_per_layer = moe_layers[0].num_routed_experts if moe_layers else 0

    # ------------------------------------------------------------------
    # Phase A: Single-pass calibration (512 samples)
    # ------------------------------------------------------------------
    spec = spec_from_config(
        cal,
        num_sequences_override=s1.get("num_calibration_samples"),
        seed_offset=1,
    )
    calib = build_calibration_tensor(
        tokenizer, spec,
        cache_dir=artifacts_dir / "_calibration_cache",
    )
    batches = iter_batches(calib, batch_size=1)

    log.info(
        "Stage 1 Phase A: profiling %d layers × %d experts on %d samples",
        len(moe_layers), n_per_layer, len(batches),
    )

    max_acc = DownProjMaxAccumulator()
    output_acc = ExpertOutputAccumulator()

    def down_cb(li, e, tensor, ctx):
        max_acc.update(li, e, tensor)
        output_acc.update(li, e, tensor)

    import contextlib as _ctx
    with _ctx.ExitStack() as stack:
        for ref in moe_layers:
            stack.enter_context(instrument_experts(ref, {"down": down_cb}))
        run_calibration(model, batches, device=device)

    max_acc.finalize()
    output_acc.finalize()

    # ------------------------------------------------------------------
    # Phase B: Super Expert Detection (2507.23279)
    # ------------------------------------------------------------------
    se_cfg = s1["super_expert_detection"]
    per_experts_by_layer = {ref.layer_idx: ref.num_routed_experts for ref in moe_layers}

    blacklist = _threshold_per_layer(
        max_acc.per_expert_max,
        num_experts_per_layer=per_experts_by_layer,
        zscore=se_cfg["zscore_threshold"],
        cap_per_layer=se_cfg["max_blacklisted_per_layer"],
    )
    total_experts = sum(per_experts_by_layer.values())
    cap_pct = float(se_cfg["global_blacklist_cap_pct"])
    if not (0.0 < cap_pct <= 1.0):
        raise ValueError(
            f"global_blacklist_cap_pct={cap_pct} must be a fraction in (0, 1], "
            "e.g. 0.05 for 5%. Got a value outside this range — did you pass an "
            "integer percentage (e.g. 5) instead of a decimal (0.05)?"
        )
    global_cap = int(cap_pct * total_experts)
    blacklist = _apply_global_cap(blacklist, max_acc.per_expert_max, global_cap)

    blacklist_out = {str(li): sorted(es) for li, es in blacklist.items() if es}
    blacklist_path = artifacts_dir / "stage1_blacklist.json"
    save_json_artifact(
        {
            "blacklist": blacklist_out,
            "per_expert_max": {f"{k[0]}_{k[1]}": v for k, v in max_acc.per_expert_max.items()},
            "config": se_cfg,
        },
        blacklist_path,
    )
    log.info(
        "Stage 1 Phase B: blacklisted %d / %d super experts → %s",
        sum(len(v) for v in blacklist_out.values()), total_experts, blacklist_path,
    )

    # Trackio: SE detection stats
    import statistics as _stats
    for ref in moe_layers:
        vals = [
            max_acc.per_expert_max.get((ref.layer_idx, e), 0.0)
            for e in range(ref.num_routed_experts)
        ]
        if not vals:
            continue
        _trackio_log({
            "stage1/se_layer_idx": ref.layer_idx,
            "stage1/se_down_max_mean": float(_stats.fmean(vals)),
            "stage1/se_down_max_std": float(_stats.pstdev(vals)) if len(vals) > 1 else 0.0,
            "stage1/se_down_max_max": float(max(vals)),
            "stage1/se_blacklisted": float(len(blacklist.get(ref.layer_idx, []))),
        })

    # ------------------------------------------------------------------
    # Phase C: CKA Similarity Matrices
    # ------------------------------------------------------------------
    log.info("Stage 1 Phase C: computing CKA pairwise similarity matrices")

    D_matrices: dict[int, torch.Tensor] = {}
    per_layer_counts: dict[int, int] = {}
    for k, ref in enumerate(moe_layers):
        D = _cka_distance_matrix(output_acc, ref)
        D_matrices[ref.layer_idx] = D
        per_layer_counts[ref.layer_idx] = ref.num_routed_experts
        log.info("  CKA matrix: layer %d/%d (idx=%d)", k + 1, len(moe_layers), ref.layer_idx)

    # Free the output accumulator (can be large: 40 layers × 256 experts × repr vectors)
    del output_acc

    # Also support weight-space fallback metrics for testing/ablation.
    metric = s1.get("similarity_metric", "cka")
    if metric != "cka":
        log.info("Stage 1: overriding CKA with weight-space metric '%s' (ablation mode)", metric)
        for k, ref in enumerate(moe_layers):
            D_matrices[ref.layer_idx] = _pairwise_distance_matrix(ref, metric=metric)

    # ------------------------------------------------------------------
    # Phase D: GRAPE Algorithm 1 (entropy-aware greedy merge with restart)
    # ------------------------------------------------------------------
    global_budget = decomposition.global_expert_budget
    gamma = float(s1.get("entropy_tolerance", 0.1))

    # Floor = num_routed_experts // 2 (no early/late bonuses)
    min_experts = n_per_layer // 2 if n_per_layer > 0 else s1.get("min_experts_per_layer", 128)

    budgets = _grape_greedy_merge(
        D_matrices=D_matrices,
        global_budget=global_budget,
        per_layer_counts=per_layer_counts,
        min_experts=min_experts,
        blacklist=blacklist,
        gamma=gamma,
    )

    # Logging: per-layer redundancy
    redundancies: dict[int, float] = {}
    for li, D in D_matrices.items():
        n = D.shape[0]
        if n <= 1:
            redundancies[li] = 0.0
        else:
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
    budgets_path = artifacts_dir / "stage1_budgets.json"
    save_json_artifact(out, budgets_path)
    log.info(
        "Stage 1 complete — budgets range=[%d..%d] mean=%.1f → %s",
        min(budgets.values()), max(budgets.values()),
        np.mean(list(budgets.values())), budgets_path,
    )
    return blacklist_path, budgets_path


# ---------------------------------------------------------------------------
# Super Expert Detection helpers
# ---------------------------------------------------------------------------


def _threshold_per_layer(
    per_expert_max: dict[tuple[int, int], float],
    *,
    num_experts_per_layer: dict[int, int],
    zscore: float,
    cap_per_layer: int,
) -> dict[int, list[int]]:
    blacklist: dict[int, list[int]] = {}
    for li, n_experts in num_experts_per_layer.items():
        vals = np.array([per_expert_max.get((li, e), 0.0) for e in range(n_experts)])
        mean, std = vals.mean(), vals.std()
        if std <= 0:
            blacklist[li] = []
            continue
        thresh = mean + zscore * std
        flagged = [int(e) for e in range(n_experts) if vals[e] > thresh]
        flagged.sort(key=lambda e: -vals[e])
        blacklist[li] = flagged[:cap_per_layer]
    return blacklist


def _apply_global_cap(
    blacklist: dict[int, list[int]],
    per_expert_max: dict[tuple[int, int], float],
    cap: int,
) -> dict[int, list[int]]:
    flat = [
        (li, e, per_expert_max.get((li, e), 0.0))
        for li, es in blacklist.items()
        for e in es
    ]
    if len(flat) <= cap:
        return blacklist
    flat.sort(key=lambda x: -x[2])
    kept = flat[:cap]
    out: dict[int, list[int]] = {}
    for li, e, _ in kept:
        out.setdefault(li, []).append(e)
    return out


# ---------------------------------------------------------------------------
# CKA distance matrix from collected expert output representations
# ---------------------------------------------------------------------------


def _cka_distance_matrix(
    output_acc: 'ExpertOutputAccumulator',
    layer_ref,
) -> torch.Tensor:
    """Compute pairwise CKA distance matrix for all experts in a layer.

    Uses expert output representations collected during the calibration
    forward pass. CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    where HSIC uses linear kernels.
    """
    n_experts = layer_ref.num_routed_experts
    li = layer_ref.layer_idx

    # Collect representation matrices: [n_tokens, d_out] per expert
    repr_matrices = []
    for e in range(n_experts):
        R = output_acc.get_representations(li, e)  # [n_tokens, d_out]
        if R is None or R.shape[0] == 0:
            # Expert was never activated — use zero vector
            R = torch.zeros(1, 1, dtype=torch.float32)
        repr_matrices.append(R.to(torch.float32))

    # Compute CKA pairwise using linear kernel
    n = n_experts
    dist = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        Xi = repr_matrices[i]
        mi = Xi.shape[0]
        for j in range(i + 1, n):
            Xj = repr_matrices[j]
            mj = Xj.shape[0]
            # Cross-HSIC: need same number of samples — truncate to min length
            m_common = min(mi, mj)
            if m_common <= 1:
                dist[i, j] = dist[j, i] = 1.0
                continue
            Xi_c = Xi[:m_common]
            Xj_c = Xj[:m_common]
            H = torch.eye(m_common) - 1.0 / m_common
            Ki = H @ (Xi_c @ Xi_c.T) @ H
            Kj = H @ (Xj_c @ Xj_c.T) @ H
            hsic_ij = float((Ki * Kj).sum().item())
            hsic_ii = float((Ki * Ki).sum().item())
            hsic_jj = float((Kj * Kj).sum().item())
            denom = math.sqrt(max(hsic_ii, 1e-12) * max(hsic_jj, 1e-12))
            cka = hsic_ij / denom if denom > 0 else 0.0
            d = max(0.0, min(1.0, 1.0 - cka))
            dist[i, j] = d
            dist[j, i] = d

    return dist


# ---------------------------------------------------------------------------
# Weight-space distance matrix fallback (for ablation / testing)
# ---------------------------------------------------------------------------


def _pairwise_distance_matrix(layer_ref, *, metric: str) -> torch.Tensor:
    """Weight-space pairwise distance matrix (fallback for ablation)."""
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
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    return dist


# ---------------------------------------------------------------------------
# GRAPE Algorithm 1 (entropy-aware greedy merge with restart)
# ---------------------------------------------------------------------------


def _grape_greedy_merge(
    *,
    D_matrices: dict[int, torch.Tensor],
    global_budget: int,
    per_layer_counts: dict[int, int],
    min_experts: int,
    blacklist: dict[int, list[int]],
    gamma: float,
) -> dict[int, int]:
    """GRAPE Algorithm 1 (2604.06542, §3.3).

    Returns per-layer surviving expert counts (budgets).
    Floor = min_experts (expected: num_routed_experts // 2). No bonuses.
    """
    sorted_layers = sorted(per_layer_counts.keys())
    n_moe_layers = len(sorted_layers)

    # Entropy is computed over active (non-blacklisted) experts only.
    # Blacklisted experts are not available for merging, so including them in
    # cluster_counts would inflate E_init and cause premature layer freezing.
    cluster_counts: dict[int, int] = {
        li: per_layer_counts[li] - len(blacklist.get(li, []))
        for li in per_layer_counts
    }

    # global_budget (from BudgetDecomposition) counts TOTAL surviving experts including
    # blacklisted ones. GRAPE tracks only non-blacklisted experts in cluster_counts, so
    # the termination condition must compare against the non-blacklisted budget.
    total_blacklisted = sum(len(v) for v in blacklist.values())
    effective_budget = max(0, global_budget - total_blacklisted)

    # R^l = sum of off-diagonal distances (Eq. 11, sum form).
    # D_matrices contains DISTANCES (0=identical, large=different) from
    # _pairwise_distance_matrix / _cka_distance_matrix. Small R means experts
    # are mutually similar (redundant); large R means diverse experts.
    # Layer selection uses argmin R (most redundant = smallest distance sum),
    # NOT argmax — this is correct for distance matrices despite GRAPE's paper
    # notation which uses argmax R over a SIMILARITY-based R.
    R: dict[int, float] = {}
    for li in sorted_layers:
        D = D_matrices[li]
        n = D.shape[0]
        R[li] = float((D.sum() - D.diag().sum()).item()) if n > 1 else 0.0

    D_work: dict[int, np.ndarray] = {
        li: D_matrices[li].cpu().numpy().copy() for li in sorted_layers
    }

    # Floor: max(min_experts, blacklist size per layer)
    floors: dict[int, int] = {
        li: max(min_experts, len(blacklist.get(li, [])))
        for li in sorted_layers
    }

    def _entropy(counts: dict[int, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in counts.values()])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    E_init = _entropy(cluster_counts)
    E_hat = E_init * (1.0 - gamma)

    frozen: set[int] = set()
    current_total = sum(cluster_counts.values())

    log.info("GRAPE: global_budget=%d (non-bl effective=%d), current_total=%d, gamma=%.2f, E_hat=%.4f, floor=%d",
             global_budget, effective_budget, current_total, gamma, E_hat, min_experts)

    max_iterations = current_total * n_moe_layers
    for iteration in range(max_iterations):
        if current_total <= effective_budget:
            break

        if len(frozen) >= n_moe_layers:
            frozen.clear()
            log.info("GRAPE iter %d: all layers frozen → restart", iteration)

        best_layer = None
        best_R = float('inf')
        for li in sorted_layers:
            if li in frozen:
                continue
            if cluster_counts[li] <= floors[li]:
                continue
            if R[li] < best_R:
                best_R = R[li]
                best_layer = li

        if best_layer is None:
            log.warning("GRAPE: no unfrozen layer can donate — stopping at %d (target %d)",
                        current_total, global_budget)
            break

        D_l = D_work[best_layer]
        n = D_l.shape[0]
        # For a distance matrix: find the most similar (smallest distance) off-diagonal pair.
        # Diagonal is 0 (self-distance) and already-merged pairs are zeroed out — exclude both.
        tmp = D_l.copy()
        np.fill_diagonal(tmp, np.inf)
        tmp[D_l == 0] = np.inf
        if not np.isfinite(tmp).any():
            frozen.add(best_layer)
            continue
        flat_idx = int(np.argmin(tmp))
        i_star, j_star = divmod(flat_idx, n)

        if D_l[i_star, j_star] <= 0:
            frozen.add(best_layer)
            continue

        # D4: zero entire row/column of absorbed expert (not just the pair)
        contribution = float(D_l[i_star, j_star]) + float(D_l[j_star, i_star])
        R[best_layer] -= contribution
        D_l[i_star, j_star] = 0.0
        D_l[j_star, i_star] = 0.0
        R[best_layer] -= float(D_l[j_star, :].sum() + D_l[:, j_star].sum())
        D_l[j_star, :] = 0.0
        D_l[:, j_star] = 0.0

        cluster_counts[best_layer] -= 1
        current_total -= 1

        E_current = _entropy(cluster_counts)
        if E_current < E_hat:
            frozen.add(best_layer)

    log.info("GRAPE: converged at %d non-blacklisted experts (target %d) after %d iterations",
             current_total, effective_budget, min(iteration + 1, max_iterations))

    if current_total > effective_budget:
        log.warning(
            "GRAPE: could not reach effective_budget=%d non-blacklisted (achieved=%d). "
            "Consider reducing min_experts_per_layer or the target reduction ratio.",
            effective_budget, current_total,
        )

    # Stage 2 reads per-layer budgets as TOTAL centroid count (blacklisted + non-blacklisted).
    # Add blacklisted experts back so Stage 2's effective_target is inclusive.
    return {
        li: cluster_counts[li] + len(blacklist.get(li, []))
        for li in cluster_counts
    }
