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
    iter_decoder_layers,
    iter_moe_layers,
    save_json_artifact,
)
from .utils.trackio_log import trackio_flush as _trackio_flush
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)

_MA_RATIO = 100.0                     # max / Q99 threshold — first MoE layer absolute outlier check
_MA_GROWTH_RATIO = 3.0                # was 5.0; calibrated for Qwen3.5/3.6 attn_output_gate=true
_MOE_OUTPUT_GROWTH_RATIO = 2.0        # ungated MoE-block-output secondary signal (OR with residual)
_PHASE_A_BATCH_SIZE = 4               # Phase A only tracks max magnitudes — batch-size invariant
_CKA_EPSILON = 1e-12                  # numerical floor for HSIC denominators
_SIMILARITY_METRIC_DEFAULT = "cka"


def _make_calibration_progress_cb(phase_tag: str, n_total: int, log_every: int = 64):
    """Build a per-batch callback that streams Stage 1 calibration progress to
    Trackio every ``log_every`` batches.

    Phase A and Phase B both run a calibration forward pass over the full
    sample set (4000 by default). At ~0.4 sec/forward on H200 that is ~28 min
    per phase, with no Trackio metric emits in the existing flow until each
    phase's *end-of-phase* summary fires. Operators staring at the Trackio
    dashboard see a flat zero for ~1 hour. This callback emits a fractional
    progress signal so the dashboard shows a live ramp 0→1 per phase, helping
    distinguish "still working" from "stuck".

    Tags: ``stage1/{phase_tag}/calibration_progress`` (fraction in [0,1]) and
    ``stage1/{phase_tag}/calibration_step`` (raw batch index).

    Cost: one trackio_log + one trackio_flush call per ``log_every`` batches
    (~63 emits per phase at 4000 samples / log_every=64). Negligible vs
    forward-pass cost. The flush ensures (a) trackio's background sender
    thread is kept alive and (b) the in-process queue drains on a known
    cadence — without it, a silently-dead sender thread would let queued
    emits pile up while the dashboard shows nothing.
    """
    def _cb(i: int) -> None:
        n_done = i + 1
        if log_every > 0 and n_done % log_every == 0:
            _trackio_log({
                f"stage1/{phase_tag}/calibration_progress": n_done / n_total,
                f"stage1/{phase_tag}/calibration_step": n_done,
            })
            _trackio_flush()
    return _cb


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

    Writes ``stage1_blacklist.json`` with exactly three top-level keys:
      - ``blacklist`` : dict[str(layer_idx) -> list[expert_idx]] — the SE blacklist.
      - ``per_expert_max`` : dict[str -> float] — per-expert max-magnitude over all
        calibration batches, keyed by the format ``"L{layer_i}E{expert_i}"`` (e.g.
        ``"L3E17"`` for layer 3, expert 17). All MoE layers are instrumented per
        spec §4 Phase B, so any expert that fired on at least one calibration sample
        is present here regardless of whether its layer is in L; the L-restriction
        is applied later when computing P99.5 / a_max for the SE criterion. (A-C-N-2)
      - ``config`` : dict — thresholds and detector parameters used to produce
        the blacklist (a_max_fraction, ma_ratio, ma_growth_ratio,
        ma_formation_layers, p995_threshold, a_max_absolute, a_max_threshold).

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
            "GRAPE floor is computed per-layer as num_routed_experts // 2"
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
    moe_output_growth_ratio = float(se_cfg.get("moe_output_growth_ratio", _MOE_OUTPUT_GROWTH_RATIO))

    spec = spec_from_config(
        cal,
        num_sequences_override=s1.get("num_calibration_samples"),
        seed_offset=1,
    )
    calib = build_calibration_tensor(
        tokenizer, spec,
        cache_dir=artifacts_dir / "_calibration_cache",
    )
    phase_a_batch_size = int(s1.get("phase_a_batch_size", _PHASE_A_BATCH_SIZE))
    batches = iter_batches(calib, batch_size=phase_a_batch_size)
    n_batches = len(batches)

    # ------------------------------------------------------------------
    # Phase A: MA-Formation Layer Detection (Pass 1)
    # ------------------------------------------------------------------
    log.info(
        "Stage 1 Phase A: detecting MA-formation layers over %d samples (%d MoE layers)",
        n_batches, len(moe_layers),
    )
    L, residual_growth, moe_output_growth, moe_output_max = _detect_ma_layers(
        model, batches, moe_layers, device,
        ma_ratio=ma_ratio,
        ma_growth_ratio=ma_growth_ratio,
        moe_output_growth_ratio=moe_output_growth_ratio,
    )
    log.info("Stage 1 Phase A: MA-formation layers L = %s", sorted(L))

    # ------------------------------------------------------------------
    # Phase B: Expert Magnitude + CKA (Pass 2)
    # ------------------------------------------------------------------
    # iter_batches returns a list; reuse the same list for Phase B.
    batches = iter_batches(calib, batch_size=1)

    log.info(
        "Stage 1 Phase B: profiling %d layers × up to %d experts on %d samples "
        "(magnitude for L=%s, CKA for all layers)",
        len(moe_layers), n_per_layer, n_batches, sorted(L),
    )

    max_acc = DownProjMaxAccumulator()
    # Hard-pin to 256 per spec §12 D-ma-detector ("CKA reservoir cap = 256 tokens
    # per expert"). Passing the value explicitly ensures a future default change
    # in activation_hooks.py does not silently drift the spec-pinned cap.
    output_acc = ExpertOutputAccumulator(max_tokens_per_expert=256)

    def down_cb(li, e, tensor, _ctx):  # _ctx required by the instrument_experts CallbackFn protocol; unused here
        # All MoE layers are instrumented simultaneously per spec §4 Phase B
        # ("All MoE layers are instrumented simultaneously"). The L-restriction is
        # applied later, in _compute_se_thresholds (P99.5 / a_max are computed only
        # over l ∈ L per Algorithm 1 line 16). Hook collects magnitudes for ALL
        # MoE layers; downstream filtering enforces the SE-criterion's l ∈ L gate.
        # Using pre-routing-weight magnitude (down_proj output before top_k_weight
        # scaling). Paper Eq. 6 is ambiguous; post-weight magnitude would require
        # passing routing weights through the hook.
        max_acc.update(li, e, tensor)
        output_acc.update(li, e, tensor)

    with contextlib.ExitStack() as stack:
        for ref in moe_layers:
            stack.enter_context(instrument_experts(ref, {"down": down_cb}))
        run_calibration(
            model, batches, device=device,
            per_batch_callback=_make_calibration_progress_cb(
                "phase_b", n_total=len(batches),
            ),
        )

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

    # One-shot Trackio emit of Phase A/C summary (additive; all variables in
    # scope from the prose log.info above).
    _trackio_log({
        "stage1/ma_formation_layers_count": len(L),
        "stage1/total_experts": int(total_experts),
        "stage1/p995_threshold": float(p995),
        "stage1/a_max": float(a_max),
        "stage1/a_max_threshold": float(a_max_fraction * a_max),
        "stage1/n_blacklisted": int(sum(len(v) for v in blacklist_out.values())),
    })

    # Trackio: SE detection stats
    for ref in moe_layers:
        in_ma_layer = ref.layer_idx in L
        entry: dict = {
            "stage1/se_layer_idx": ref.layer_idx,
            "stage1/se_blacklisted": len(blacklist.get(ref.layer_idx, [])),
            "stage1/se_in_ma_layer": float(in_ma_layer),
        }
        # Magnitude stats are only meaningful for MA-formation layers (set L); for non-L
        # layers we now also collect magnitudes (spec §4 Phase B: "All MoE layers are
        # instrumented simultaneously"), but they never enter the SE three-way AND, so
        # we still omit these stats for non-L layers rather than emitting ambiguous values.
        if in_ma_layer:
            # Only include experts that were actually activated; absent keys mean zero
            # activations on all calibration samples, which would bias the statistics.
            vals = [v for (li, _e), v in max_acc.per_expert_max.items() if li == ref.layer_idx]
            if vals:
                entry["stage1/se_down_max_mean"] = float(statistics.fmean(vals))
                entry["stage1/se_down_max_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
                entry["stage1/se_down_max_max"] = float(max(vals))
        _trackio_log(entry)
    # End-of-Phase-A/C: drain queue + keep sender thread alive before the
    # multi-minute Phase D CKA loop (which has no per-batch flush of its own).
    _trackio_flush()

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
        for ref in moe_layers:
            D_matrices[ref.layer_idx] = _pairwise_distance_matrix(ref, metric=metric)

    # ------------------------------------------------------------------
    # Phase E: GRAPE Algorithm 1 (entropy-aware greedy merge with restart)
    # ------------------------------------------------------------------
    global_budget = decomposition.global_expert_budget
    gamma = float(s1.get("entropy_tolerance", 0.1))

    budgets = _grape_greedy_merge(
        D_matrices=D_matrices,
        global_budget=global_budget,
        per_layer_counts=per_layer_counts,
        blacklist=blacklist,
        gamma=gamma,
    )

    # Logging: per-layer redundancy R̃^l (spec §4, Eq. 3)
    # Build D_work_logging: zero blacklisted rows/cols in a copy of D_matrices
    # (mirrors what _grape_greedy_merge does internally, for consistency).
    D_work_logging: dict[int, np.ndarray] = {
        li: _zero_blacklisted(D.cpu().numpy().copy(), blacklist.get(li, []))
        for li, D in D_matrices.items()
    }

    # R^l = Σ_{i≠j} D^l_{ij}  (sum of off-diagonal distances)
    R_raw: dict[int, float] = {}
    for li, d in D_work_logging.items():
        n = d.shape[0]
        R_raw[li] = float(d.sum() - np.diag(d).sum()) if n > 1 else 0.0

    # Min-max normalise across layers → R̃^l ∈ [0, 1]
    r_min = min(R_raw.values())
    r_max = max(R_raw.values())
    if r_max > r_min:
        denom = r_max - r_min
    else:
        # All layers have identical R_raw — redundancy collapses to 0.0 for all.
        # This is expected and benign for single-layer models (only one data point,
        # so min-max normalisation is undefined and R̃^l=0 for all layers). Also
        # expected when all layers have identical distance-sum profiles (e.g. uniform
        # random init in tests).
        denom = 1.0
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
    # End-of-Phase-D/E per-layer emit: drain before Stage 1 returns control
    # to its caller.
    _trackio_flush()

    out = {
        "per_layer_target_experts": {str(k): v for k, v in budgets.items()},
        "per_layer_redundancy": {str(k): v for k, v in redundancies.items()},
        "achieved_budget": sum(budgets.values()),
        "requested_budget": decomposition.global_expert_budget,
        "config": dict(s1),
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


def _flag_layer_dual_signal(
    *,
    residual_ratio: float,
    moe_ratio: float,
    residual_threshold: float,
    moe_threshold: float,
) -> bool:
    """OR-rule flagging: True iff EITHER signal exceeds its threshold (D-ma-detector)."""
    return (residual_ratio > residual_threshold) or (moe_ratio > moe_threshold)


def _detect_ma_layers(
    model: nn.Module,
    batches,
    moe_layers,
    device,
    *,
    ma_ratio: float = _MA_RATIO,
    ma_growth_ratio: float = _MA_GROWTH_RATIO,
    moe_output_growth_ratio: float = _MOE_OUTPUT_GROWTH_RATIO,
) -> tuple[set[int], dict[int, float], dict[int, float], dict[int, float]]:
    """Forward pass 1: identify MA-formation layers via dual-signal OR rule.

    Returns (L, residual_growth, moe_output_growth, moe_output_max):
      L                   — set of MA-formation layer indices.
      residual_growth     — per-MoE-layer max|H_l|/max|H_{l-1}| (residual stream).
                            First MoE layer entry is float('nan') (no predecessor).
      moe_output_growth   — per-MoE-layer max|MoE_l|/max|MoE_{l-1}| (post-routing-weighted-sum,
                            pre-residual-add). First MoE layer entry is 0.0 (no predecessor).
      moe_output_max      — per-MoE-layer raw max|MoE_l| (for diagnostics).

    See ALGORITHM_REFERENCE.md §4 Phase A and D-ma-detector for the OR rule rationale.
    """
    sorted_moe_layer_indices = sorted(ref.layer_idx for ref in moe_layers)
    if not sorted_moe_layer_indices:
        return set(), {}, {}, {}
    first_moe_layer_idx = sorted_moe_layer_indices[0]
    moe_layer_by_idx = {ref.layer_idx: ref for ref in moe_layers}

    decoder_layers: list[tuple[int, nn.Module]] = list(iter_decoder_layers(model))
    if not decoder_layers:
        raise ValueError("_detect_ma_layers: no decoder layers found")
    decoder_layer_modules = {layer: idx for idx, layer in decoder_layers}
    if len(decoder_layer_modules) != len(decoder_layers):
        raise ValueError("_detect_ma_layers: weight-tied decoder layers detected")
    sorted_decoder_layer_indices = sorted(decoder_layer_modules.values())

    # Residual-stream max: hooked on every decoder layer
    layer_max: dict[int, float] = {idx: 0.0 for idx in sorted_decoder_layer_indices}
    # MoE-block-output max: hooked on each MoE layer's `mlp` (Qwen3_5MoeSparseMoeBlock)
    moe_block_max: dict[int, float] = {idx: 0.0 for idx in sorted_moe_layer_indices}
    first_layer_q99_buffer: list[np.ndarray] = []
    handles: list = []

    def _make_decoder_hook(layer_idx: int):
        def _hook(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor):
                return
            h_abs = h.detach().abs().float()
            curr_max = h_abs.max().item()
            if curr_max > layer_max[layer_idx]:
                layer_max[layer_idx] = curr_max
            if layer_idx == first_moe_layer_idx:
                first_layer_q99_buffer.append(h_abs.flatten().cpu().numpy())
        return _hook

    def _make_moe_hook(layer_idx: int):
        def _hook(_module, _input, output):
            # Qwen3_5MoeSparseMoeBlock.forward returns (hidden_states, router_logits) tuple.
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor):
                return
            curr_max = h.detach().abs().float().max().item()
            if curr_max > moe_block_max[layer_idx]:
                moe_block_max[layer_idx] = curr_max
        return _hook

    for module, layer_idx in decoder_layer_modules.items():
        handles.append(module.register_forward_hook(_make_decoder_hook(layer_idx)))
    for layer_idx in sorted_moe_layer_indices:
        ref = moe_layer_by_idx[layer_idx]
        handles.append(ref.mlp.register_forward_hook(_make_moe_hook(layer_idx)))

    try:
        run_calibration(
            model, batches, device=device,
            per_batch_callback=_make_calibration_progress_cb(
                "phase_a", n_total=len(batches),
            ),
        )
    finally:
        for h in handles:
            h.remove()

    if first_layer_q99_buffer:
        first_layer_q99 = float(
            np.percentile(np.concatenate(first_layer_q99_buffer), 99.0)
        )
    else:
        first_layer_q99 = 0.0

    L: set[int] = set()
    residual_growth: dict[int, float] = {}
    moe_output_growth: dict[int, float] = {}
    decoder_index_pos = {idx: pos for pos, idx in enumerate(sorted_decoder_layer_indices)}

    for layer_idx in sorted_moe_layer_indices:
        if layer_idx == first_moe_layer_idx:
            residual_growth[layer_idx] = float("nan")
            moe_output_growth[layer_idx] = 0.0
            if first_layer_q99 <= 0:
                log.warning(
                    "Stage 1: first-MoE-layer Q99 is %.2e for layer %d; model output may be "
                    "degenerate — excluding from MA-formation candidate set L",
                    first_layer_q99, layer_idx,
                )
            elif layer_max[layer_idx] > ma_ratio * first_layer_q99:
                L.add(layer_idx)
            continue
        # Residual ratio against the immediately preceding decoder layer
        pos = decoder_index_pos[layer_idx]
        prev_decoder_idx = sorted_decoder_layer_indices[pos - 1]
        prev_max = layer_max[prev_decoder_idx]
        res_ratio = (layer_max[layer_idx] / prev_max) if prev_max > 0 else 0.0
        residual_growth[layer_idx] = res_ratio
        # MoE ratio against the previous MoE layer in the sorted order
        prev_moe_pos = sorted_moe_layer_indices.index(layer_idx) - 1
        prev_moe_idx = sorted_moe_layer_indices[prev_moe_pos]
        prev_moe_max = moe_block_max[prev_moe_idx]
        moe_ratio = (moe_block_max[layer_idx] / prev_moe_max) if prev_moe_max > 0 else 0.0
        moe_output_growth[layer_idx] = moe_ratio
        if _flag_layer_dual_signal(
            residual_ratio=res_ratio,
            moe_ratio=moe_ratio,
            residual_threshold=ma_growth_ratio,
            moe_threshold=moe_output_growth_ratio,
        ):
            L.add(layer_idx)

    if not L:
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
        total_layers = getattr(text_cfg, "num_hidden_layers", None) or len(sorted_decoder_layer_indices)
        cutoff = round(0.75 * float(total_layers))
        fallback_layers = [li for li in sorted_moe_layer_indices if li < cutoff]
        log.warning(
            "Stage 1 Phase A: dual-signal detector returned ∅; falling back to "
            "layer_idx < round(0.75 × %d) = %d. Fallback L = %s",
            total_layers, cutoff, fallback_layers,
        )
        L = set(fallback_layers)

    return L, residual_growth, moe_output_growth, dict(moe_block_max)


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
        if L:
            log.warning(
                "_compute_se_thresholds: MA-formation layers L=%s but no expert fired "
                "on any calibration sample in those layers; SE detection will find nothing. "
                "Consider increasing the calibration set size or checking the model.",
                sorted(L),
            )
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
        if per_expert_max:
            log.warning(
                "Phase C: L is empty; skipping SE detection (no MA-formation layers found, "
                "even after fallback)."
            )
        return {}
    # Magnitudes are collected for ALL MoE layers (spec §4 Phase B: "All MoE layers
    # are instrumented simultaneously"); the SE three-way AND is then enforced here
    # by silently skipping any (l, e) with l ∉ L (Eq. 6's `l ∈ L` clause).
    blacklist: dict[int, list[int]] = {}
    for (li, e), v in per_expert_max.items():
        if li not in L:
            continue
        if v > p995 and v > a_max_threshold:
            blacklist.setdefault(li, []).append(e)
    return blacklist


# ---------------------------------------------------------------------------
# CKA distance matrix from collected expert output representations
# ---------------------------------------------------------------------------


_CKA_M_MIN_VECTORIZED_FLOOR = 32  # below this, the GPU uniform-m path is unsafe


def _cka_distance_matrix(
    output_acc: ExpertOutputAccumulator,
    layer_ref,
) -> torch.Tensor:
    """Compute pairwise CKA distance matrix for all experts in a layer.

    Uses expert output representations collected during the calibration
    forward pass. CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    where HSIC uses linear kernels and the biased centering of Gretton (2005):
    K_c = K - row_mean - col_mean + grand_mean. Distance = (1 − CKA), clamped
    to [0, 1].

    Dispatches between two implementations based on the reservoir fill across
    the active expert set:

    - **GPU vectorized** (default for prod): subsamples every active expert
      to a single m_min over the active set, batches the Gram matrices, and
      computes the full N×N HSIC table in O(N · m² · d) GPU work. ~1 sec/layer
      on H200 vs ~10 min/layer for the CPU per-pair path.
    - **CPU per-pair fallback**: original implementation. Activated when the
      vectorized path is unsafe — m_min < 32 OR m_min < m_max // 4. Used by
      tests with tiny calibration sets and as a safety net when reservoir
      under-fill would force every pair to use a low m.

    With the prod default of ``stage1_grape.num_calibration_samples=1024`` and
    the ExpertOutputAccumulator reservoir cap of 256 tokens/expert, all active
    experts saturate at m=256 and the GPU path is bit-equivalent (within fp32
    tolerance) to the original.

    Unactivated experts (m_e ≤ 1) get distance 1.0 in their full row and
    column, preserving the original placeholder semantics in both paths.
    """
    n_experts = layer_ref.num_routed_experts
    li = layer_ref.layer_idx

    # Pre-pass: gather active reservoirs and decide which path to take.
    active_indices: list[int] = []
    active_reprs: list[torch.Tensor] = []
    active_lengths: list[int] = []
    for e in range(n_experts):
        R = output_acc.get_representations(li, e)  # [m_e, d_out] CPU fp32 or None
        if R is None or R.shape[0] < 2:
            continue
        active_indices.append(e)
        active_reprs.append(R.detach().to(torch.float32))
        active_lengths.append(R.shape[0])

    # Initialize result: max dissimilarity 1.0, self-distance 0.0. Inactive
    # experts retain their full row/col at 1.0 — bit-identical to the original
    # zero-placeholder behavior.
    dist = torch.ones(n_experts, n_experts, dtype=torch.float32)
    dist.fill_diagonal_(0.0)

    if len(active_indices) < 2:
        return dist

    m_min = min(active_lengths)
    m_max = max(active_lengths)
    if m_min < _CKA_M_MIN_VECTORIZED_FLOOR or m_min < m_max // 4:
        # Reservoir is under-filled or skewed enough that uniform m_min would
        # silently degrade every pair's CKA precision. Fall back to the
        # per-pair m_common path so each pair retains its full intersection.
        log.info(
            "_cka_distance_matrix: layer %d active reservoir lengths span [%d..%d]; "
            "below vectorized floor (m_min ≥ %d and ≥ m_max//4) — falling back to "
            "CPU per-pair m_common path. Cause: small calibration / routing imbalance.",
            li, m_min, m_max, _CKA_M_MIN_VECTORIZED_FLOOR,
        )
        return _cka_distance_matrix_cpu_per_pair(
            active_indices, active_reprs, active_lengths, n_experts, dist
        )

    # ----- GPU vectorized path (uniform m_min over the active set) -----
    # Uniform-stride subsample to m_min — spreads token coverage across the
    # reservoir rather than front-slicing, identical to the original.
    X_list: list[torch.Tensor] = []
    for R in active_reprs:
        m = R.shape[0]
        if m == m_min:
            X_list.append(R)
        else:
            step = m / m_min
            idx = [int(k * step) for k in range(m_min)]
            X_list.append(R[idx])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.stack(X_list, dim=0).to(device)  # [N_active, m_min, d_out]

    # Batched linear-kernel Gram matrices: K_e = X_e @ X_e.T.
    K = torch.bmm(X, X.transpose(-2, -1))  # [N_active, m_min, m_min]

    # Biased HSIC centering: K_c = K - row_mean - col_mean + grand_mean.
    row_mean = K.mean(dim=2, keepdim=True)            # [N, m, 1]
    col_mean = K.mean(dim=1, keepdim=True)            # [N, 1, m]
    grand_mean = K.mean(dim=(1, 2), keepdim=True)     # [N, 1, 1]
    Kc = K - row_mean - col_mean + grand_mean

    # HSIC matrix H[i,j] = ⟨Kc_i, Kc_j⟩_F = (Kc_flat @ Kc_flat^T)[i,j].
    Kc_flat = Kc.reshape(Kc.shape[0], -1)             # [N, m*m]
    H = Kc_flat @ Kc_flat.t()                         # [N, N]

    # CKA = H[i,j] / sqrt(max(H[i,i], ε) · max(H[j,j], ε)).
    diag = H.diagonal().clamp(min=_CKA_EPSILON)
    norm = torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
    CKA = H / norm
    D_active = (1.0 - CKA).clamp(0.0, 1.0)
    D_active.fill_diagonal_(0.0)

    # Scatter the active-active sub-block back into the full distance matrix.
    idx_t = torch.tensor(active_indices, dtype=torch.long)
    dist[idx_t.unsqueeze(1), idx_t.unsqueeze(0)] = D_active.detach().cpu()

    return dist


def _cka_distance_matrix_cpu_per_pair(
    active_indices: list[int],
    active_reprs: list[torch.Tensor],
    active_lengths: list[int],
    n_experts: int,
    dist: torch.Tensor,
) -> torch.Tensor:
    """CPU per-pair m_common fallback. Preserves the original O(N²) Python loop
    used when the vectorized GPU path is unsafe (very small reservoirs / skewed
    fill). Fills the active-active sub-block of ``dist``; inactive rows/cols
    retain their pre-initialized 1.0 (with diagonal 0.0)."""
    n_active = len(active_indices)
    for ii in range(n_active):
        ei = active_indices[ii]
        Xi = active_reprs[ii]
        mi = active_lengths[ii]
        for jj in range(ii + 1, n_active):
            ej = active_indices[jj]
            Xj = active_reprs[jj]
            mj = active_lengths[jj]
            m_common = min(mi, mj)
            if m_common <= 1:
                # H=0 → CKA undefined → maximum distance.
                dist[ei, ej] = dist[ej, ei] = 1.0
                continue
            if mi > m_common:
                step = mi / m_common
                Xi_c = Xi[[int(k * step) for k in range(m_common)]]
            else:
                Xi_c = Xi
            if mj > m_common:
                step = mj / m_common
                Xj_c = Xj[[int(k * step) for k in range(m_common)]]
            else:
                Xj_c = Xj
            # Biased HSIC centering (Gretton 2005), identical to the GPU path.
            Ki_raw = Xi_c @ Xi_c.T
            Ki = Ki_raw - Ki_raw.mean(dim=1, keepdim=True) - Ki_raw.mean(dim=0, keepdim=True) + Ki_raw.mean()
            Kj_raw = Xj_c @ Xj_c.T
            Kj = Kj_raw - Kj_raw.mean(dim=1, keepdim=True) - Kj_raw.mean(dim=0, keepdim=True) + Kj_raw.mean()
            hsic_ij = float((Ki * Kj).sum().item())
            hsic_ii = float((Ki * Ki).sum().item())
            hsic_jj = float((Kj * Kj).sum().item())
            denom = math.sqrt(max(hsic_ii, _CKA_EPSILON) * max(hsic_jj, _CKA_EPSILON))
            cka = hsic_ij / denom
            d = max(0.0, min(1.0, 1.0 - cka))
            dist[ei, ej] = d
            dist[ej, ei] = d
    # Diagonal is already 0.0 from the caller.
    return dist


# ---------------------------------------------------------------------------
# Weight-space distance matrix fallback (for ablation / testing)
# ---------------------------------------------------------------------------


# Scale note (A-C-N-1): cosine is scaled to [0, 1] by (1 - sim) / 2; MSE is normalized by
# its max. The two scales are NOT directly comparable across runs that switch metrics.
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


def _zero_blacklisted(d: np.ndarray, bl_experts: list[int]) -> np.ndarray:
    """Zero rows and columns of distance matrix `d` for blacklisted experts in-place."""
    for e in bl_experts:
        d[e, :] = 0.0
        d[:, e] = 0.0
    return d


# ---------------------------------------------------------------------------
# GRAPE Algorithm 1 (entropy-aware greedy merge with restart)
# ---------------------------------------------------------------------------


def _grape_greedy_merge(
    *,
    D_matrices: dict[int, torch.Tensor],
    global_budget: int,
    per_layer_counts: dict[int, int],
    blacklist: dict[int, list[int]],
    gamma: float,
) -> dict[int, int]:
    """GRAPE Algorithm 1 (2604.06542, §3.3).

    Returns per-layer surviving expert counts (budgets). Floor is per_layer_counts[li] // 2
    computed independently for each layer, so heterogeneous architectures are handled correctly.
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
    D_work: dict[int, np.ndarray] = {
        li: _zero_blacklisted(D_matrices[li].cpu().numpy().copy(), blacklist.get(li, []))
        for li in sorted_layers
    }

    for li, D in D_work.items():
        diag = np.diag(D)
        if not np.allclose(diag, 0.0):
            log.debug("Stage 1: D_work[layer %d] diagonal is non-zero (max=%.2e); R update may double-count", li, float(np.abs(diag).max()))

    R: dict[int, float] = {}
    for li in sorted_layers:
        d = D_work[li]
        n = d.shape[0]
        R[li] = float((d.sum() - np.diag(d).sum())) if n > 1 else 0.0

    # floors[li] is the NON-BLACKLISTED portion of the hard floor, i.e. the
    # minimum number of non-blacklisted experts that must survive in layer li.
    # Total floor = floors[li] + len(blacklist[li]).
    # GRAPE tracks only non-blacklisted experts in cluster_counts, so
    # cluster_counts[li] must not drop below floors[li].
    # NOTE: `min_experts_per_layer` is a config key consumed by the budget solver
    # for global feasibility, but Stage 1 hardcodes the per-layer floor as
    # `per_layer_counts[li] // 2` per spec §12 D5 ("min_experts_per_layer =
    # num_routed_experts // 2"). Stage 1 does not read the config value here.
    floors: dict[int, int] = {
        li: max(per_layer_counts[li] // 2 - len(blacklist.get(li, [])), 0)
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

    if gamma == 0.0:
        log.warning(
            "GRAPE: gamma=0.0 — entropy gate at initial entropy — every entropy-reducing merge will trigger a freeze.",
        )
    elif gamma < 0.0:
        log.warning(
            "GRAPE: gamma=%.4f < 0: E_hat > E_init — every merge reduces entropy below the "
            "inflated threshold, so most layers freeze after the first merge per restart cycle; "
            "the loop produces approximately %d merges per restart cycle (one per MoE layer); "
            "convergence may require far more iterations than the normal (gamma>0) case",
            gamma, n_moe_layers,
        )
    elif gamma >= 1.0:
        log.warning(
            "GRAPE: gamma=%.4f >= 1.0: E_hat <= 0 — entropy gate permanently disabled; "
            "GRAPE will merge greedily to floor without entropy constraints",
            gamma,
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

    log.info("GRAPE: global_budget=%d (non-bl effective=%d), current_total=%d, gamma=%.4g, E_hat=%.4f",
             global_budget, effective_budget, current_total, gamma, E_hat)

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
    # once). Top-of-loop restarts fall through to layer selection in the same iteration;
    # lag-corrected restarts use `continue` and burn one iteration without a merge.
    # The factor n_moe_layers * 2 is well above this tight bound in both cases.
    max_iterations = current_total * n_moe_layers * 2
    log.debug("GRAPE max_iterations=%d (current_total=%d, n_moe_layers=%d)",
              max_iterations, current_total, n_moe_layers)
    # n_merges counts successful merge operations only.  Structural-blocking skip-iterations
    # advance iter_ without incrementing n_merges, so n_merges <= iter_ always.
    n_merges = 0
    exit_reason = "max_iter"
    for iter_ in range(max_iterations):
        if current_total <= effective_budget:
            exit_reason = "budget"
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
            log.info("GRAPE iter %d: all non-permanently-blocked layers frozen → restart", iter_)
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
            # floor_blocked was updated lazily inside the selection loop above; re-evaluate
            # the restart condition with the now-complete floor_blocked before giving up —
            # the per-iteration lag may have prevented the check above from firing.
            permanently_blocked = len(floor_blocked) + len(structurally_blocked - floor_blocked)
            non_perm_blocked = n_moe_layers - permanently_blocked
            if non_perm_blocked > 0 and len(frozen - structurally_blocked - floor_blocked) >= non_perm_blocked:
                log.info("GRAPE iter %d: post-selection restart (lag-corrected) — unfreezing frozen layers", iter_)
                frozen.clear()
                continue
            log.warning("GRAPE: no unfrozen layer can donate — stopping at %d (target %d)",
                        current_total, effective_budget)
            exit_reason = "no_layer"
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
        # i_star is intentionally discarded: GRAPE's contribution to Stage 2 is the per-layer budget N'_l, not pair assignments — Stage 2 re-derives centroids via covariance (spec §4 line 271).
        _, j_star = divmod(flat_idx, n)

        # D4: zero entire row/column of absorbed expert j_star and update R.
        # R = Σ_{i≠j} D_l[i, j] (sum of all off-diagonal entries). When j_star
        # is absorbed we must remove its full contribution: D_l[j_star, k] and
        # D_l[k, j_star] for all k. Read the full row/column sum BEFORE zeroing
        # so that D_l[i_star, j_star] / D_l[j_star, i_star] are still included.
        # Defensively subtract the (always-zero) diagonal once so the formula is
        # robust if any future metric ever yielded a non-zero self-distance.
        j_contribution = float(
            D_l[j_star, :].sum() + D_l[:, j_star].sum() - D_l[j_star, j_star]
        )
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

        n_merges += 1

    log.info("GRAPE: converged at %d non-blacklisted experts (target %d) after %d merges (exit=%s)",
             current_total, effective_budget, n_merges, exit_reason)

    if current_total > effective_budget and exit_reason == "max_iter":
        log.warning(
            "GRAPE: could not reach effective_budget=%d non-blacklisted (achieved=%d) "
            "after %d iterations (max_iterations=%d). "
            "Consider increasing global_budget or reducing the target compression ratio "
            "(floors are per_layer_counts[li] // 2 per layer).",
            effective_budget, current_total, iter_ + 1, max_iterations,
        )

    # One-shot Trackio emit of GRAPE summary. All variables already in scope
    # — pure additive emit, no new state computed.
    _trackio_log({
        "stage1/effective_budget": int(effective_budget),
        "stage1/global_budget": int(global_budget),
        "stage1/total_blacklisted": int(total_blacklisted),
        "stage1/entropy_initial": float(E_init),
        "stage1/entropy_threshold": float(E_hat),
        "stage1/gamma": float(gamma),
        "stage1/n_merges_executed": int(n_merges),
        "stage1/exit_reason": exit_reason,
        "stage1/final_total": int(current_total),
    })
    # End-of-GRAPE-solver: drain final summary before returning.
    _trackio_flush()

    # Stage 2 reads per-layer budgets as TOTAL centroid count (blacklisted + non-blacklisted).
    # Add blacklisted experts back so Stage 2's effective_target is inclusive.
    return {
        li: cluster_counts[li] + len(blacklist.get(li, []))
        for li in cluster_counts
    }
