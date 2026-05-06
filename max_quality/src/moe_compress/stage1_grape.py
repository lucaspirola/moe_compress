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

import contextlib
import logging
import math
import statistics
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
_CKA_EPSILON = 1e-12     # numerical floor for HSIC denominators to avoid division by zero
_SIMILARITY_METRIC_DEFAULT = "cka"


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
    if any(ref.num_routed_experts != n_per_layer for ref in moe_layers[1:]):
        log.warning(
            "stage1_grape: layers have heterogeneous expert counts; "
            "min_experts floor is derived from layer 0 count (%d) only",
            n_per_layer,
        )

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
    # Re-create iterator for Phase B pass
    batches = iter_batches(calib, batch_size=1)

    log.info(
        "Stage 1 Phase B: profiling %d layers × up to %d experts on %d samples "
        "(magnitude for L=%s, CKA for all layers)",
        len(moe_layers), n_per_layer, n_batches, sorted(L),
    )

    max_acc = DownProjMaxAccumulator()
    output_acc = ExpertOutputAccumulator()

    def down_cb(li, e, tensor, _ctx):  # _ctx required by the instrument_experts CallbackFn protocol; unused here
        # Magnitude collection restricted to MA-formation layers per spec.
        if li in L:
            # Using pre-routing-weight magnitude (down_proj output before top_k_weight
            # scaling). Paper Eq. 6 is ambiguous; post-weight magnitude would require
            # passing routing weights through the hook.
            max_acc.update(li, e, tensor)
        output_acc.update(li, e, tensor)

    with contextlib.ExitStack() as stack:
        for ref in moe_layers:
            stack.enter_context(instrument_experts(ref, {"down": down_cb}))
        run_calibration(model, batches, device=device)

    max_acc.finalize()
    output_acc.finalize()

    # ------------------------------------------------------------------
    # Phase C: Super Expert Detection (2507.23279, Eq. 6)
    # ------------------------------------------------------------------
    p995, a_max = _compute_se_thresholds(max_acc.per_expert_max, L)
    blacklist = _apply_paper_criterion(
        max_acc.per_expert_max, L, p995, a_max_fraction * a_max,
    )

    # _apply_paper_criterion only appends to a key when an expert passes the criterion,
    # so all present keys map to non-empty lists; blacklist_out may itself be empty if
    # no expert qualifies.
    blacklist_out = {str(li): sorted(es) for li, es in blacklist.items()}
    total_experts = sum(ref.num_routed_experts for ref in moe_layers)

    blacklist_config = {
        "a_max_fraction": a_max_fraction,
        "ma_ratio": ma_ratio,
        "ma_growth_ratio": ma_growth_ratio,
        "ma_formation_layers": sorted(L),
        "p995_threshold": float(p995),
        "a_max_absolute": float(a_max),
        "a_max_threshold": float(a_max_fraction * a_max),
    }
    blacklist_path = artifacts_dir / "stage1_blacklist.json"
    save_json_artifact(
        {
            "blacklist": blacklist_out,
            "per_expert_max": {
                f"L{layer_i}E{expert_i}": v
                for (layer_i, expert_i), v in max_acc.per_expert_max.items()
            },
            "config": blacklist_config,
        },
        blacklist_path,
    )
    log.info(
        "Stage 1 Phase C: blacklisted %d / %d super experts (P99.5=%.3g, a_max_threshold=%.3g) → %s",
        sum(len(v) for v in blacklist_out.values()), total_experts,
        p995, a_max_fraction * a_max, blacklist_path,
    )

    # Trackio: SE detection stats
    for ref in moe_layers:
        in_ma_layer = ref.layer_idx in L
        entry: dict = {
            "stage1/se_layer_idx": ref.layer_idx,
            "stage1/se_blacklisted": len(blacklist.get(ref.layer_idx, [])),
            "stage1/se_in_ma_layer": float(in_ma_layer),
        }
        # Magnitude stats are only meaningful for MA-formation layers (set L); for non-L
        # layers max_acc.per_expert_max has no entries, so omit these keys entirely rather
        # than emitting ambiguous 0.0 values.
        if in_ma_layer:
            # Only include experts that were actually activated; absent keys mean zero
            # activations on all calibration samples, which would bias the statistics.
            vals = [v for (li, _e), v in max_acc.per_expert_max.items() if li == ref.layer_idx]
            if vals:
                entry["stage1/se_down_max_mean"] = float(statistics.fmean(vals))
                entry["stage1/se_down_max_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
                entry["stage1/se_down_max_max"] = float(max(vals))
        _trackio_log(entry)

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
    metric = s1.get("similarity_metric", _SIMILARITY_METRIC_DEFAULT)
    if metric != _SIMILARITY_METRIC_DEFAULT:
        log.info("Stage 1: overriding %s with weight-space metric '%s' (ablation mode)", _SIMILARITY_METRIC_DEFAULT, metric)
        for k, ref in enumerate(moe_layers):
            D_matrices[ref.layer_idx] = _pairwise_distance_matrix(ref, metric=metric)

    # ------------------------------------------------------------------
    # Phase E: GRAPE Algorithm 1 (entropy-aware greedy merge with restart)
    # ------------------------------------------------------------------
    global_budget = decomposition.global_expert_budget
    gamma = float(s1.get("entropy_tolerance", 0.1))

    # Floor = num_routed_experts // 2 (no early/late bonuses).
    # NOTE: for heterogeneous architectures where layers have different expert counts,
    # this single global floor derived from layer 0 is an approximation.  All layers
    # are treated as having the same floor, which may be too permissive for layers
    # with fewer experts than layer 0, or overly conservative for layers with more.
    # A per-layer floor would require extending _grape_greedy_merge to accept a
    # per-layer min_experts dict; for the current homogeneous-architecture target
    # this is acceptable (deviation D5).
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
        # Keep in sync with the corresponding zeroing block in _grape_greedy_merge;
        # consider extracting to `_zero_blacklisted(d, blacklist_for_layer)` if logic diverges.
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
    if r_max == r_min:
        # All layers have identical R_raw — redundancy collapses to 0.0 for all.
        # This is expected and benign for single-layer models (only one data point,
        # so min-max normalisation is undefined and R̃^l=0 for all layers). Also
        # expected when all layers have identical distance-sum profiles (e.g. uniform
        # random init in tests).
        log.debug(
            "Stage 1: all layers have identical R_raw=%.4g; R̃^l=0 for all layers "
            "(expected for single-layer models or uniform-init tests)",
            r_min,
        )

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
        "achieved_budget": sum(budgets.values()),
        "requested_budget": decomposition.global_expert_budget,
        "config": dict(s1),
    }
    budgets_path = artifacts_dir / "stage1_budgets.json"
    save_json_artifact(out, budgets_path)
    if budgets:
        log.info(
            "Stage 1 complete — budgets range=[%d..%d] mean=%.1f → %s",
            min(budgets.values()), max(budgets.values()),
            np.mean(list(budgets.values())), budgets_path,
        )
    else:
        log.info("Stage 1 complete — no budgets (empty model?) → %s", budgets_path)
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

    Q99 is approximated as the running max of per-batch Q99 values (conservative upper bound).
    """
    sorted_layer_indices = sorted(ref.layer_idx for ref in moe_layers)
    if not sorted_layer_indices:
        return set()
    first_layer_idx = sorted_layer_indices[0]

    moe_layer_modules = {ref.layer_module: ref.layer_idx for ref in moe_layers}
    layer_max: dict[int, float] = {idx: 0.0 for idx in sorted_layer_indices}
    # Single-element list so the hook closure can mutate via index assignment.
    # Only the first layer is ever written; subsequent layers use layer_max only.
    first_layer_q99_val: list[float] = [0.0]
    handles: list = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor):
                log.debug(
                    "_detect_ma_layers hook: unexpected output type %s for layer %d; skipping",
                    type(h).__name__, layer_idx,
                )
                return
            h_abs = h.detach().abs().float()
            curr_max = h_abs.max().item()
            if curr_max > layer_max[layer_idx]:
                layer_max[layer_idx] = curr_max
            if layer_idx == first_layer_idx:
                curr_q99 = torch.quantile(h_abs.flatten(), 0.99).item()
                # NB: running max of per-batch Q99 values; this overestimates the true
                # cross-batch Q99 but is a conservative upper bound for MA detection.
                if curr_q99 > first_layer_q99_val[0]:
                    first_layer_q99_val[0] = curr_q99
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
            q99_val = first_layer_q99_val[0]
            if q99_val <= 0:
                log.debug("Stage 1: first-layer Q99 is %.2e for layer %d; excluding from MA-formation candidate set L", q99_val, layer_idx)
            if q99_val > 0 and layer_max[layer_idx] > ma_ratio * q99_val:
                L.add(layer_idx)
        else:
            prev_max = layer_max[sorted_layer_indices[i - 1]]
            # Note: prev_max is from the previous MoE layer in sorted_layer_indices; for
            # sparsely-placed MoE layers, this ratio captures compound growth across multiple
            # non-MoE layers, not just the immediately-preceding transformer layer.
            # prev_max == 0 means the preceding layer produced no non-zero output on any
            # batch; genuine MA detection requires a nonzero baseline, so we skip this check.
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
    if not L:
        return {}
    # defensive — unreachable in normal operation (update() only called for li in L).
    # External callers or accumulator reuse may legitimately pass entries outside L;
    # collect unexpected indices first and warn once per layer rather than once per expert.
    unexpected_layers = {li for (li, _e) in per_expert_max if li not in L}
    for li in sorted(unexpected_layers):
        log.warning(
            "_apply_paper_criterion: layer %d not in MA-layer set L; skipping all experts for this layer", li
        )
    blacklist: dict[int, list[int]] = {}
    for (li, e), v in per_expert_max.items():
        if li in unexpected_layers:
            continue
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
            # Expert was never activated — use a zero placeholder. This causes
            # CKA(zero, X) = 0 for any X, so distance = 1 - CKA = 1.0 (maximum
            # distance). Unactivated experts are treated as maximally dissimilar
            # from all others; will not be preferentially selected as j_star when
            # more-similar pairs exist; if all pairwise distances equal 1.0,
            # selection is arbitrary.
            # Each unactivated expert contributes distance 1.0 to all pairs in its
            # row/column, inflating R[li]. Since argmin-R selects the most-redundant
            # layer first, layers with unactivated experts are deprioritized for
            # merging — semantically backwards but tolerated because unactivated
            # experts are also filtered in the per-pair argmin (j_star selection).
            # Shape [1, 1] is safe here: the m_common <= 1 guard (below) ensures this
            # placeholder is never used in a matmul — the pair is short-circuited to
            # distance 1.0 before Xi_c @ Xi_c.T is evaluated.
            R = torch.zeros(1, 1, dtype=torch.float32)
        repr_matrices.append(R.detach().cpu().to(torch.float32))

    # Compute CKA pairwise using linear kernel.
    # Diagonal is 0 because torch.zeros initializes it and the loop only writes i<j pairs;
    # CKA(X,X)=1→distance=0 is consistent but the code never evaluates it.
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
                # m_common=1 gives H=0, making HSIC=0 and CKA undefined; treat as
                # maximum dissimilarity.
                dist[i, j] = dist[j, i] = 1.0
                continue
            trunc_ratio = m_common / max(mi, mj)
            if trunc_ratio < 0.5:
                log.debug(
                    "_cka_distance_matrix: layer %d experts (%d, %d) truncated to %d/%d tokens (%.0f%%)",
                    li, i, j, m_common, max(mi, mj), trunc_ratio * 100,
                )
            Xi_c = Xi[:m_common]
            Xj_c = Xj[:m_common]
            # Direct doubly-centred gram matrix: avoids allocating a [m, m] H matrix on
            # every pair. This is standard HSIC centering: K_c = K - row_mean - col_mean + grand_mean.
            Ki_raw = Xi_c @ Xi_c.T
            Ki_row = Ki_raw.mean(dim=1, keepdim=True)
            Ki_col = Ki_raw.mean(dim=0, keepdim=True)
            Ki_grand = Ki_raw.mean()
            Ki = Ki_raw - Ki_row - Ki_col + Ki_grand

            Kj_raw = Xj_c @ Xj_c.T
            Kj_row = Kj_raw.mean(dim=1, keepdim=True)
            Kj_col = Kj_raw.mean(dim=0, keepdim=True)
            Kj_grand = Kj_raw.mean()
            Kj = Kj_raw - Kj_row - Kj_col + Kj_grand
            hsic_ij = float((Ki * Kj).sum().item())
            hsic_ii = float((Ki * Ki).sum().item())
            hsic_jj = float((Kj * Kj).sum().item())
            denom = math.sqrt(max(hsic_ii, _CKA_EPSILON) * max(hsic_jj, _CKA_EPSILON))
            cka = hsic_ij / denom
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
    # Cosine distance is in [0, 1]; MSE distance is normalised to [0, 1] by dividing by
    # its max; if all experts are identical (max=0), clamp keeps denominator at 1e-8 and
    # all distances stay near zero. These two modes produce incommensurable R values;
    # do not compare across metric runs.
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

    Returns per-layer surviving expert counts (budgets). Floor is num_routed_experts // 2
    with no per-position adjustment (deviation D5 from paper).
    """
    sorted_layers = sorted(per_layer_counts.keys())
    n_moe_layers = len(sorted_layers)

    # Validate that no layer's blacklist exceeds its total expert count.
    for li in sorted_layers:
        bl_count = len(blacklist.get(li, []))
        layer_count = per_layer_counts.get(li, 0)
        if bl_count > layer_count:
            raise ValueError(
                f"layer {li}: blacklist has {bl_count} experts but layer only has {layer_count}"
            )
        if bl_count == layer_count:
            log.warning(
                "layer %d: all %d experts are blacklisted; this layer cannot contribute any merges",
                li, layer_count,
            )

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
    if total_blacklisted > global_budget:
        log.warning(
            "GRAPE: total_blacklisted=%d > global_budget=%d — the mandatory super-expert set "
            "already exceeds the requested budget; effective_budget forced to 0. "
            "Consider increasing global_budget or reducing a_max_fraction.",
            total_blacklisted, global_budget,
        )

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
        # Keep in sync with the corresponding zeroing block in run() (D_work_logging);
        # consider extracting to `_zero_blacklisted(d, blacklist_for_layer)` if logic diverges.
        for bl_e in blacklist.get(li, []):
            d[bl_e, :] = 0.0
            d[:, bl_e] = 0.0
        D_work[li] = d

    for li, D in D_work.items():
        diag = np.diag(D)
        if not np.allclose(diag, 0.0):
            log.warning("Stage 1: D_work[layer %d] diagonal is non-zero (max=%.2e); R update may double-count", li, float(np.abs(diag).max()))

    R: dict[int, float] = {}
    for li in sorted_layers:
        d = D_work[li]
        n = d.shape[0]
        R[li] = float((d.sum() - np.diag(d).sum())) if n > 1 else 0.0

    # floors[li] is the NON-BLACKLISTED portion of the hard floor, i.e. the
    # minimum number of non-blacklisted experts that must survive in layer li.
    # It is NOT the total expert floor; total floor = floors[li] + len(blacklist[li])
    # = max(min_experts, len(blacklist[li])). GRAPE tracks only non-blacklisted
    # experts in cluster_counts, so cluster_counts[li] must not drop below floors[li].
    floors: dict[int, int] = {
        li: max(min_experts - len(blacklist.get(li, [])), 0)
        for li in sorted_layers
    }

    def _entropy(counts: dict[int, int]) -> float:
        if any(v < 0 for v in counts.values()):
            neg_entries = {k: v for k, v in counts.items() if v < 0}
            raise ValueError(f"_entropy: negative count(s) encountered: {neg_entries}")
        total = sum(counts.values())
        if total == 0:  # covers both empty dict and all-zero dict
            return 0.0
        probs = np.fromiter((c / total for c in counts.values()), dtype=np.float64, count=len(counts))
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    if gamma == 0:
        log.warning(
            "GRAPE: gamma=0 — entropy gate at initial entropy — every entropy-reducing merge will trigger a freeze.",
        )
    elif gamma < 0.0:
        log.warning(
            "GRAPE: gamma=%.4f < 0: E_hat > E_init — every merge reduces entropy below the "
            "inflated threshold, so every layer freezes after one merge per restart cycle; "
            "the loop produces approximately %d merges per restart cycle (one per MoE layer); "
            "convergence may require far more iterations than the normal (gamma>0) case",
            gamma, n_moe_layers,
        )

    E_init = _entropy(cluster_counts)
    # E_hat is intentionally fixed to the pre-loop baseline; it is not updated per restart (conservative approximation).
    E_hat = E_init * (1.0 - gamma)

    frozen: set[int] = set()
    # Layers where all valid (non-absorbed) pairs have been exhausted — no merge is ever
    # possible again regardless of entropy state.  Unlike `frozen`, this set is NOT
    # cleared on entropy restart so that exhausted layers are never re-selected.
    structurally_blocked: set[int] = set()
    # Pre-populate floor_blocked for layers already at their floor before merging starts.
    # Lazy-add inside the loop handles layers that reach their floor mid-run, but
    # without this pre-population, layers simultaneously at-floor and in `frozen`
    # (from a prior restart cycle) would never be added to floor_blocked, causing
    # _non_floor_blocked to overcount and trigger spurious restarts.
    # both dicts are keyed over sorted_layers; .get() defaults are unreachable
    floor_blocked: set[int] = {li for li in sorted_layers if cluster_counts[li] <= floors[li]}
    current_total = sum(cluster_counts.values())

    log.info("GRAPE: global_budget=%d (non-bl effective=%d), current_total=%d, gamma=%.4g, E_hat=%.4f, floor=%d",
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

    if current_total == 0:
        log.debug("GRAPE: all unfrozen experts blacklisted; skipping greedy merge loop")
        return {li: cluster_counts[li] + len(blacklist.get(li, [])) for li in cluster_counts}

    # Tight case: at most current_total merge-iterations plus at most n_moe_layers
    # structurally-blocked skip-iterations (each layer joins structurally_blocked at most
    # once; restarts do not consume a separate iteration — frozen.clear() and the next
    # merge both happen in the same iteration body). The factor n_moe_layers * 2 is well
    # above this tight bound.
    max_iterations = current_total * n_moe_layers * 2
    log.debug("GRAPE max_iterations=%d (current_total=%d, n_moe_layers=%d)",
              max_iterations, current_total, n_moe_layers)
    # last_merge_iter is the loop ordinal of the last successful merge, not the count of
    # successful merges — structural-blocking iterations advance `_iter` without updating
    # last_merge_iter.  Set to _iter on budget-satisfied break (_iter merges were completed;
    # _iter+1 would be a ghost iteration where no merge occurs). At max-iterations exit keep
    # _iter+1 since all iterations were genuine merge attempts.
    last_merge_iter = 0
    for _iter in range(max_iterations):
        if current_total <= effective_budget:
            last_merge_iter = _iter  # _iter merges completed; _iter+1 would be a ghost
            break

        # Restart only when entropy-frozen layers block all non-floor-blocked,
        # non-structurally-blocked layers.  Floor and structural constraints are
        # permanent — clearing frozen can't help those, and must not touch
        # structurally_blocked (which persists across restarts).
        # floor_blocked is populated lazily during layer selection; pre-seeded in the
        # initialization block above for layers already at their floor, but
        # mid-run additions lag by one iteration. permanently_blocked may undercount if
        # layers are skipped via structurally_blocked before reaching the floor check.
        # Additionally, a layer whose cluster_count was just decremented to floor[li] in
        # the current iteration is not yet in floor_blocked — it is added only at its
        # next examination, causing a one-iteration lag independent of the
        # structurally_blocked path.
        # permanently_blocked = |floor_blocked ∪ structurally_blocked|
        permanently_blocked = len(floor_blocked) + len(structurally_blocked - floor_blocked)
        non_perm_blocked = n_moe_layers - permanently_blocked
        if non_perm_blocked > 0 and len(frozen - structurally_blocked - floor_blocked) >= non_perm_blocked:
            log.info("GRAPE iter %d: all non-permanently-blocked layers frozen → restart", _iter)
            frozen.clear()
            # After clearing frozen, the current iteration immediately runs layer selection
            # and may merge a layer that re-triggers the entropy gate on the following iteration.
            # This one extra merge per restart-cycle is by design — it allows GRAPE to escape
            # local optima.

        best_layer = None
        best_R = math.inf
        for li in sorted_layers:
            if li in structurally_blocked:
                continue
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
                        current_total, effective_budget)
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
            structurally_blocked.add(best_layer)
            # not added to frozen — structurally_blocked takes precedence.
            continue
        flat_idx = int(np.argmin(tmp))
        # n is the original matrix dimension (total experts), not the count of
        # remaining unabsorbed experts.
        # i_star (absorbing centroid) is intentionally discarded here; Stage 2 re-derives
        # the merge tree from covariance data.
        _, j_star = divmod(flat_idx, n)

        # D4: zero entire row/column of absorbed expert j_star and update R.
        # R = Σ_{i≠j} D_l[i, j] (sum of all off-diagonal entries). When j_star
        # is absorbed we must remove its full contribution: D_l[j_star, k] and
        # D_l[k, j_star] for all k. Read the full row/column sum BEFORE zeroing
        # so that D_l[i_star, j_star] / D_l[j_star, i_star] are still included.
        j_contribution = float(D_l[j_star, :].sum() + D_l[:, j_star].sum())
        # The diagonal D_l[j_star, j_star] is double-counted but always zero,
        # so result is exact.
        # Diagonal is 0 by construction: torch.zeros init + CKA/MSE metrics produce dist(x,x)=0.
        R[best_layer] -= j_contribution
        pre_clamp_R = R[best_layer]
        R[best_layer] = max(0.0, pre_clamp_R)
        if pre_clamp_R < 0.0:
            log.debug("_grape_greedy_merge: pre-clamp R[%d]=%.2e clamped to 0.0 (FP drift)", best_layer, pre_clamp_R)
        D_l[j_star, :] = 0.0
        D_l[:, j_star] = 0.0
        absorbed.add(j_star)

        cluster_counts[best_layer] -= 1
        current_total -= 1

        E_current = _entropy(cluster_counts)
        if E_current < E_hat:
            frozen.add(best_layer)

        # Record the ordinal of this successful merge (budget-satisfied exit sets last_merge_iter
        # before breaking; max-iterations exit falls through with this value).
        last_merge_iter = _iter + 1

    log.info("GRAPE: converged at %d non-blacklisted experts (target %d) after %d merges",
             current_total, effective_budget, last_merge_iter)

    if current_total > effective_budget:
        log.warning(
            "GRAPE: could not reach effective_budget=%d non-blacklisted (achieved=%d) "
            "(max_iterations=%d reached). "
            "Consider reducing min_experts_per_layer or the target reduction ratio.",
            effective_budget, current_total, max_iterations,
        )

    # Stage 2 reads per-layer budgets as TOTAL centroid count (blacklisted + non-blacklisted).
    # Add blacklisted experts back so Stage 2's effective_target is inclusive.
    return {
        li: cluster_counts[li] + len(blacklist.get(li, []))
        for li in cluster_counts
    }
