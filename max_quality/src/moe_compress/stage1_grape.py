"""Stage 1 — Super Expert Detection + GRAPE non-uniform per-layer expert budgets.

Two sequential forward passes over 256 calibration samples:
  Phase A: detect MA-formation layers (set L)
  Phase B: collect max down_proj output magnitude (l ∈ L) + CKA representations (all layers)

Super expert detection follows 2507.23279 Eq. 6: three-way AND criterion
(> P99.5(A) AND > 0.1·a_max AND l ∈ L). No per-layer caps, no global caps.

GRAPE uses CKA similarity (paper §3.2 explicitly allows "CKA, MSE, or other
similarity measures"). Floor constraint: num_routed_experts // 2 (deviation D5).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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

_MA_RATIO = 100.0        # max / Q99 threshold — absolute outlier check for first MoE layer
_MA_GROWTH_RATIO = 5.0   # max|H_l| / max|H_{l-1}| threshold — growth check for subsequent layers


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
    """Run Stage 1: SE detection + GRAPE budget allocation.

    Returns (blacklist_path, budgets_path).
    """
    s1 = config["stage1_grape"]
    cal = config["calibration"]
    moe_layers = list(iter_moe_layers(model))
    if not moe_layers:
        raise ValueError(
            "Stage 1: model has no MoE layers — check iter_moe_layers() compatibility "
            "with this model architecture."
        )
    n_per_layer = moe_layers[0].num_routed_experts

    se_cfg = s1["super_expert_detection"]

    # Warn on deprecated config keys (kept for backwards compat; not used).
    for old_key in ("zscore_threshold", "max_blacklisted_per_layer", "global_blacklist_cap_pct"):
        if old_key in se_cfg:
            log.warning(
                "Stage 1: config key '%s' is deprecated and ignored. "
                "Super expert detection now uses the paper's three-way AND criterion "
                "(P99.5 + 0.1·a_max). Remove this key from your config.",
                old_key,
            )

    ma_ratio = float(se_cfg.get("ma_ratio", _MA_RATIO))
    ma_growth_ratio = float(se_cfg.get("ma_growth_ratio", _MA_GROWTH_RATIO))
    a_max_fraction = float(se_cfg.get("a_max_fraction", 0.1))

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
    n_batches = len(batches)

    # ------------------------------------------------------------------
    # Phase A: MA-Formation Layer Detection (Pass 1)
    # ------------------------------------------------------------------
    log.info(
        "Stage 1 Phase A: detecting MA-formation layers over %d samples (%d MoE layers)",
        n_batches, len(moe_layers),
    )
    L = _detect_ma_layers(model, batches, moe_layers, device, ma_ratio=ma_ratio, ma_growth_ratio=ma_growth_ratio)
    log.info("Stage 1 Phase A: MA-formation layers L = %s", sorted(L))

    # ------------------------------------------------------------------
    # Phase B: Expert Magnitude + CKA (Pass 2)
    # ------------------------------------------------------------------
    batches = iter_batches(calib, batch_size=1)

    log.info(
        "Stage 1 Phase B: profiling %d layers × %d experts on %d samples "
        "(magnitude for L=%s, CKA for all layers)",
        len(moe_layers), n_per_layer, n_batches, sorted(L),
    )

    max_acc = DownProjMaxAccumulator()
    output_acc = ExpertOutputAccumulator()

    def down_cb(li, e, tensor, ctx):
        # Magnitude collection restricted to MA-formation layers per spec.
        if li in L:
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
    # Phase C: Super Expert Detection (2507.23279, Eq. 6)
    # ------------------------------------------------------------------
    per_experts_by_layer = {ref.layer_idx: ref.num_routed_experts for ref in moe_layers}

    p995, a_max = _compute_se_thresholds(max_acc.per_expert_max, L)
    blacklist = _apply_paper_criterion(
        max_acc.per_expert_max, L, p995, a_max_fraction * a_max,
    )

    blacklist_out = {str(li): sorted(es) for li, es in blacklist.items() if es}
    total_experts = sum(per_experts_by_layer.values())

    blacklist_config = {
        "a_max_fraction": a_max_fraction,
        "ma_ratio": ma_ratio,
        "ma_formation_layers": sorted(L),
        "p995_value": p995,
        "a_max_value": float(a_max),
        "a_max_threshold": float(a_max_fraction * a_max),
    }
    blacklist_path = artifacts_dir / "stage1_blacklist.json"
    save_json_artifact(
        {
            "blacklist": blacklist_out,
            "per_expert_max": {f"L{k[0]}E{k[1]}": v for k, v in max_acc.per_expert_max.items()},
            "config": blacklist_config,
        },
        blacklist_path,
    )
    log.info(
        "Stage 1 Phase C: blacklisted %d / %d super experts (P99.5=%.3g, 0.1·a_max=%.3g) → %s",
        sum(len(v) for v in blacklist_out.values()), total_experts,
        p995, a_max_fraction * a_max, blacklist_path,
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
    # Phase D: CKA Similarity Matrices
    # ------------------------------------------------------------------
    log.info("Stage 1 Phase D: computing CKA pairwise distance matrices (D = 1 − CKA)")

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
    # Phase E: GRAPE Algorithm 1 (entropy-aware greedy merge with restart)
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

    # Logging: per-layer redundancy R̃^l (spec §4, Eq. 3)
    # Build D_work_logging: zero blacklisted rows/cols in a copy of D_matrices
    # (mirrors what _grape_greedy_merge does internally, for consistency).
    D_work_logging: dict[int, np.ndarray] = {}
    for li, D in D_matrices.items():
        d = D.cpu().numpy().copy()
        for bl_e in blacklist.get(li, []):
            d[bl_e, :] = 0.0
            d[:, bl_e] = 0.0
        D_work_logging[li] = d

    # R^l = Σ_{i≠j} D^l_{ij}  (sum of off-diagonal distances)
    R_raw: dict[int, float] = {}
    for li, d in D_work_logging.items():
        n = d.shape[0]
        R_raw[li] = float(d.sum() - np.diag(d).sum()) if n > 1 else 0.0

    # Min-max normalise across layers → R̃^l ∈ [0, 1]
    r_min = min(R_raw.values())
    r_max = max(R_raw.values())
    denom = r_max - r_min if r_max > r_min else 1.0

    redundancies: dict[int, float] = {
        li: (R_raw[li] - r_min) / denom for li in R_raw
    }

    for li in D_matrices:
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
# Phase A: MA-formation layer detection
# ---------------------------------------------------------------------------


def _detect_ma_layers(
    model: nn.Module,
    batches,
    moe_layers,
    device,
    *,
    ma_ratio: float = _MA_RATIO,
    ma_growth_ratio: float = _MA_GROWTH_RATIO,
) -> set[int]:
    """Forward pass 1: identify decoder layers that form (amplify) massive activations.

    Uses a hybrid check to distinguish formation from propagation:
    - First MoE layer: absolute outlier check — add to L if max|H_l| > ma_ratio × Q99(|H_l|).
    - Subsequent layers: growth check — add to L if max|H_l| / max|H_{l-1}| > ma_growth_ratio.

    MAs propagate stably through residuals after they form; the old absolute check would
    flag all post-formation layers. The growth check flags only amplification events.
    Both maxima are tracked across all calibration batches (MAs are input-stable per the paper).
    """
    sorted_layer_indices = sorted(ref.layer_idx for ref in moe_layers)
    if not sorted_layer_indices:
        return set()
    first_layer_idx = sorted_layer_indices[0]

    moe_layer_modules = {ref.layer_module: ref.layer_idx for ref in moe_layers}
    layer_max: dict[int, float] = {idx: 0.0 for idx in sorted_layer_indices}
    layer_q99: dict[int, float] = {idx: 0.0 for idx in sorted_layer_indices}
    handles: list = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor):
                return
            h_abs = h.detach().abs().float()
            curr_max = h_abs.max().item()
            if curr_max > layer_max[layer_idx]:
                layer_max[layer_idx] = curr_max
            if layer_idx == first_layer_idx:
                curr_q99 = torch.quantile(h_abs.flatten(), 0.99).item()
                if curr_q99 > layer_q99[layer_idx]:
                    layer_q99[layer_idx] = curr_q99
        return _hook

    for module, layer_idx in moe_layer_modules.items():
        h = module.register_forward_hook(_make_hook(layer_idx))
        handles.append(h)

    try:
        run_calibration(model, batches, device=device)
    finally:
        for h in handles:
            h.remove()

    L: set[int] = set()
    for i, layer_idx in enumerate(sorted_layer_indices):
        if i == 0:
            q99 = layer_q99[layer_idx]
            if q99 > 0 and layer_max[layer_idx] > ma_ratio * q99:
                L.add(layer_idx)
        else:
            prev_max = layer_max[sorted_layer_indices[i - 1]]
            if prev_max > 0 and layer_max[layer_idx] / prev_max > ma_growth_ratio:
                L.add(layer_idx)

    return L


# ---------------------------------------------------------------------------
# Phase C: Super Expert Detection helpers
# ---------------------------------------------------------------------------


def _compute_se_thresholds(
    per_expert_max: dict[tuple[int, int], float],
    L: set[int],
) -> tuple[float, float]:
    """Compute P99.5 and a_max over all (l, e) with l ∈ L."""
    A = [v for (li, _e), v in per_expert_max.items() if li in L]
    if not A:
        return 0.0, 0.0
    arr = np.array(A, dtype=np.float64)
    p995 = float(np.percentile(arr, 99.5))
    a_max = float(arr.max())
    return p995, a_max


def _apply_paper_criterion(
    per_expert_max: dict[tuple[int, int], float],
    L: set[int],
    p995: float,
    a_max_threshold: float,
) -> dict[int, list[int]]:
    """Apply Eq. 6 three-way AND: a > P99.5 AND a > 0.1·a_max AND l ∈ L."""
    blacklist: dict[int, list[int]] = {}
    for (li, e), v in per_expert_max.items():
        if li not in L:
            raise RuntimeError(
                f"per_expert_max contains layer {li} which is not in L={L}; "
                "down_cb should only call max_acc.update() for li in L"
            )
        if v > p995 and v > a_max_threshold:
            blacklist.setdefault(li, []).append(e)
    return blacklist


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
    #
    # Blacklisted experts are zeroed out in D_work so they never participate
    # in pair selection as either centroid (i_star) or absorbed expert (j_star),
    # and their distances do not inflate R (which would bias layer selection).
    D_work: dict[int, np.ndarray] = {}
    for li in sorted_layers:
        d = D_matrices[li].cpu().numpy().copy()
        for bl_e in blacklist.get(li, []):
            d[bl_e, :] = 0.0
            d[:, bl_e] = 0.0
        D_work[li] = d

    R: dict[int, float] = {}
    for li in sorted_layers:
        d = D_work[li]
        n = d.shape[0]
        R[li] = float((d.sum() - np.diag(d).sum())) if n > 1 else 0.0

    # Floor: non-blacklisted portion = max(min_experts - BL, 0).
    # Total floor = non-blacklisted floor + BL = max(min_experts, BL). Correct.
    floors: dict[int, int] = {
        li: max(min_experts - len(blacklist.get(li, [])), 0)
        for li in sorted_layers
    }

    def _entropy(counts: dict[int, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in counts.values()])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    if gamma <= 0.0:
        log.warning(
            "GRAPE: gamma=%.4f ≤ 0 — entropy constraint will freeze every layer after "
            "the first merge (E_hat=E_init). Set gamma > 0 to allow meaningful reduction.",
            gamma,
        )

    E_init = _entropy(cluster_counts)
    E_hat = E_init * (1.0 - gamma)

    frozen: set[int] = set()
    # Pre-populate floor_blocked for layers already at their floor before merging starts.
    # Lazy-add inside the loop handles layers that reach their floor mid-run, but
    # without this pre-population, layers simultaneously at-floor and in `frozen`
    # (from a prior restart cycle) would never be added to floor_blocked, causing
    # _non_floor_blocked to overcount and trigger spurious restarts.
    floor_blocked: set[int] = {li for li in sorted_layers if cluster_counts.get(li, 0) <= floors.get(li, 0)}
    current_total = sum(cluster_counts.values())

    log.info("GRAPE: global_budget=%d (non-bl effective=%d), current_total=%d, gamma=%.2f, E_hat=%.4f, floor=%d",
             global_budget, effective_budget, current_total, gamma, E_hat, min_experts)

    # Per-layer sets of absorbed (merged-away) expert indices.  Using an explicit
    # set — rather than checking D_l == 0 — avoids misidentifying genuinely
    # zero-distance (identical-weight) expert pairs as already-merged.
    # Pre-populate with blacklisted experts: their D_work rows/cols are 0.0
    # (zeroed during D_work initialization), so without this pre-population
    # argmin would select them as j_star and corrupt cluster_counts.
    merged: dict[int, set[int]] = {
        li: set(blacklist.get(li, [])) for li in sorted_layers
    }

    max_iterations = current_total * n_moe_layers
    iteration = -1
    for iteration in range(max_iterations):
        if current_total <= effective_budget:
            break

        # Restart only when entropy-frozen layers block all non-floor-blocked layers.
        # Floor constraints are permanent — clearing floor_blocked can't help and
        # causes a spurious restart loop when all layers are at their floor.
        _non_floor_blocked = n_moe_layers - len(floor_blocked)
        if _non_floor_blocked > 0 and len(frozen) >= _non_floor_blocked:
            frozen.clear()
            log.info("GRAPE iter %d: all non-floor-blocked layers frozen → restart", iteration)

        best_layer = None
        best_R = float('inf')
        for li in sorted_layers:
            if li in frozen:
                continue
            if cluster_counts[li] <= floors[li]:
                floor_blocked.add(li)
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
        # For a distance matrix: find the most similar (smallest distance) pair where
        # neither expert has already been absorbed.  Track absorbed experts explicitly
        # so that genuinely zero-distance (identical-weight) pairs remain selectable.
        absorbed = merged[best_layer]
        tmp = D_l.copy()
        np.fill_diagonal(tmp, np.inf)
        for a in absorbed:
            tmp[a, :] = np.inf
            tmp[:, a] = np.inf
        if not np.isfinite(tmp).any():
            frozen.add(best_layer)
            continue
        flat_idx = int(np.argmin(tmp))
        i_star, j_star = divmod(flat_idx, n)

        # D4: zero entire row/column of absorbed expert and update R
        contribution = float(D_l[i_star, j_star]) + float(D_l[j_star, i_star])
        R[best_layer] -= contribution
        D_l[i_star, j_star] = 0.0
        D_l[j_star, i_star] = 0.0
        R[best_layer] -= float(D_l[j_star, :].sum() + D_l[:, j_star].sum())
        D_l[j_star, :] = 0.0
        D_l[:, j_star] = 0.0
        absorbed.add(j_star)

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
