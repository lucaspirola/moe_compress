"""Stage 2 — REAP scoring + REAM pseudo-pruning, fused-experts-aware.

Key differences from the pre-refactor version:
  - Weights live in stacked tensors on ``Qwen3_5MoeExperts``; pruning means
    slicing those tensors and the router's ``gate.weight`` rows.
  - Scoring hooks go through :func:`instrument_experts` which monkey-patches
    the fused forward with per-expert callbacks.
  - Input covariance for Stage 3 is collected on two tap points:
      ``gate_up_in``    → covariance used by gate_proj + up_proj SVD
      ``intermediate``  → covariance used by down_proj SVD
    We save these under the (layer, expert, matrix_name) key space that
    Stage 3 consumes.

REAM cost matrix (paper 2604.04356, reference ream/ream.py):
  - δ_gate (Eq. 5): similarity ∈ [0,1] between L2-row-normalized pre-softmax
    gate logit profile vectors — Euclidean distance converted via dist2sim.
  - δ̃_expert (Eq. 8): mean cosine similarity of full-softmax-gated expert
    outputs σ(x)_i · E_i(x), rescaled to [0,1] via (cosine+1)/2.
  - δ_REAM = (δ_gate + δ̃_expert) / 2 ∈ [0,1]; cost = 1 − δ_REAM.
  - Grouping: single-pass greedy per paper §4, descending centroid saliency,
    full assignment guaranteed by upfront feasibility check.

Frequency-weighted merge with neuron permutation alignment is preserved.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from .utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
    ReapAccumulator,
    _EarlyExitException,
    capture_router_outputs,
    early_exit_after_layer,
    instrument_experts,
    record_reap,
    run_calibration,
)
from .utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from .utils.model_io import (
    MATRIX_NAMES,
    MoELayerRef,
    build_banks,
    iter_moe_layers,
    load_json_artifact,
    save_compressed_checkpoint,
    save_json_artifact,
)
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
    stage1_budget_path: Path | None = None,
    no_resume: bool = False,
) -> Path:
    s2 = config["stage2_reap_ream"]
    cal = config["calibration"]

    if stage1_budget_path is None:
        stage1_budget_path = artifacts_dir / "stage1_budgets.json"
    budgets_payload = load_json_artifact(stage1_budget_path)
    per_layer_target = {
        int(k): int(v) for k, v in budgets_payload["per_layer_target_experts"].items()
    }
    blacklist_payload = load_json_artifact(artifacts_dir / "stage1_blacklist.json")
    blacklist = {int(k): list(v) for k, v in blacklist_payload.get("blacklist", {}).items()}

    spec = spec_from_config(cal, num_sequences_override=s2["num_calibration_samples"])
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    # iter_batches returns a list (not a generator): confirmed by inspecting
    # calibration.py — it returns [calib_ids[i:i+batch_size] for i in range(...)].
    # The same `batches` list is safely re-iterable across all layer profiling passes.
    batches = iter_batches(calib, batch_size=s2["batch_size"])

    moe_layers = list(iter_moe_layers(model))
    cov_acc = InputCovarianceAccumulator()
    # Spec §5 "Covariance Side-Collection": FP32 storage certified by Swift-SVD
    # paper 2604.01609; avoids numerical degradation in eigendecomposition.
    cov_dtype = getattr(torch, s2.get("covariance_storage_dtype", "float32"))
    cov_acc.set_storage_dtype(cov_dtype)
    merge_map: dict[int, dict[int, list[int]]] = {}

    # -----------------------------------------------------------------------
    # Crash-resume: scan partial_dir for layers already completed in a prior
    # interrupted run. Re-apply merges in layer order (fast, no forward pass).
    # -----------------------------------------------------------------------
    completed_layers: set[int] = set()
    _layer_mean_costs: list[float] = []  # running history for cost-threshold gate (Strategy C)

    if no_resume:
        partial_dir = None
        # Delete stale partial dir so a future non-no-resume run cannot resume
        # from this run's incomplete (or absent) checkpoints.
        stale = artifacts_dir / "_stage2_partial"
        if stale.exists():
            import shutil as _shutil
            _shutil.rmtree(stale, ignore_errors=True)
    else:
        partial_dir = artifacts_dir / "_stage2_partial"
        partial_dir.mkdir(parents=True, exist_ok=True)
        for _stale in partial_dir.glob("*.tmp"):
            _stale.unlink(missing_ok=True)

        # Crash safety: delete any .pt whose matching .json is absent.
        # A .pt without .json means the process died between _snapshot_cov_layer
        # and _write_merge_json. The covariance has been remapped but not recorded.
        # Reprocessing the .pt would double-remap — silent numerical corruption.
        for ref in moe_layers:
            pt_path = partial_dir / f"layer_{ref.layer_idx}.pt"
            json_path = partial_dir / f"merge_{ref.layer_idx}.json"
            if pt_path.exists() and not json_path.exists():
                log.warning(
                    "Stage 2 resume: orphaned %s (no matching JSON) — "
                    "deleting and reprocessing layer %d",
                    pt_path.name, ref.layer_idx,
                )
                pt_path.unlink()

        for ref in moe_layers:
            merge_path = partial_dir / f"merge_{ref.layer_idx}.json"
            cov_path = partial_dir / f"layer_{ref.layer_idx}.pt"
            if not (merge_path.exists() and cov_path.exists()):
                continue
            data = json.loads(merge_path.read_text())
            fv = int(data.get("format_version", 0))
            if fv != 1:
                raise RuntimeError(
                    f"_stage2_partial/merge_{ref.layer_idx}.json has format_version={fv} "
                    "(expected 1) — delete _stage2_partial/ and re-run Stage 2"
                )
            # Migration guard: old partial dirs wrote "centroid_ids"; new ones
            # write "final_kept_ids". Accept both for backward compatibility.
            if "final_kept_ids" in data:
                centroid_ids = [int(x) for x in data["final_kept_ids"]]
            elif "centroid_ids" in data:
                log.warning(
                    "Stage 2 resume layer %d: found deprecated 'centroid_ids' field "
                    "(expected 'final_kept_ids') — using it for backward compatibility. "
                    "Delete _stage2_partial/ to regenerate with the new format.",
                    ref.layer_idx,
                )
                centroid_ids = [int(x) for x in data["centroid_ids"]]
            else:
                raise RuntimeError(
                    f"_stage2_partial/merge_{ref.layer_idx}.json missing both "
                    "'final_kept_ids' and 'centroid_ids' keys — file is corrupt. "
                    "Delete _stage2_partial/ and re-run Stage 2."
                )
            grouped = {int(k): list(v) for k, v in data["grouped"].items()}
            freq = {int(k): int(v) for k, v in data["freq"].items()}
            merge_map_layer = {int(k): list(v) for k, v in data["merge_map_layer"].items()}

            n_pre_merge = len(freq)
            if ref.num_routed_experts != n_pre_merge:
                raise RuntimeError(
                    f"Stage 2 resume layer {ref.layer_idx}: expected {n_pre_merge} "
                    f"experts (pre-merge) but model has {ref.num_routed_experts}. "
                    "The model passed to stage2.run() must be the Stage 1 output, "
                    "not a partially-merged model."
                )

            _merge_experts_inplace(ref, grouped, freq,
                                   freq_weighted=s2["ream"]["frequency_weighted_merge"])
            banks = build_banks(ref)
            for bank in banks.values():
                bank.select(centroid_ids)
            _resize_router_for_kept_experts(ref, centroid_ids)

            try:
                cov_acc.load_layer_from_disk(ref.layer_idx, partial_dir)
            except Exception as _exc:
                raise RuntimeError(
                    f"Stage 2 resume: failed to load covariance for layer {ref.layer_idx} "
                    f"from _stage2_partial/ ({_exc}). "
                    "The in-memory model has already been partially mutated — "
                    "restart with a fresh Stage 1 model and delete _stage2_partial/."
                ) from _exc
            merge_map[ref.layer_idx] = merge_map_layer
            completed_layers.add(ref.layer_idx)
            log.info("Stage 2: layer %d resumed from partial (skipping profile + merge)",
                     ref.layer_idx)
            val = data.get("mean_cost_per_pair")
            if val is not None and val > 0.0:
                _layer_mean_costs.append(float(val))

        if completed_layers:
            log.info("Stage 2: resumed %d / %d layers from %s",
                     len(completed_layers), len(moe_layers), partial_dir)

    max_group_cap: int = s2.get("max_merge_group_size", 0) or 0
    cost_sigma: float = s2.get("ream_cost_sigma_threshold", float("inf"))
    cost_bump_ratio: float = s2.get("ream_cost_bump_ratio", 0.10)
    min_active_tokens: int = s2.get("reap_min_active_tokens", 0)

    for k, layer_ref in enumerate(moe_layers):
        if layer_ref.layer_idx in completed_layers:
            log.info(
                "Stage 2 layer %d/%d (idx=%d) — skipped (resumed from partial)",
                k + 1, len(moe_layers), layer_ref.layer_idx,
            )
            continue

        target = per_layer_target[layer_ref.layer_idx]
        log.info(
            "Stage 2 layer %d/%d (idx=%d) — profiling then merging to %d experts",
            k + 1, len(moe_layers), layer_ref.layer_idx, target,
        )
        reap_acc = ReapAccumulator()
        ream_acc = ReamCostAccumulator()
        torch.cuda.empty_cache()
        _profile_layer(
            model, layer_ref, batches, reap_acc, cov_acc, ream_acc,
            device=device,
        )
        reap_acc.finalize_layer(layer_ref.layer_idx)
        cov_acc.finalize_layer(layer_ref.layer_idx)

        n_experts = layer_ref.num_routed_experts
        protected = set(blacklist.get(layer_ref.layer_idx, []))
        scores = np.array([reap_acc.score(layer_ref.layer_idx, e) for e in range(n_experts)])
        freq = {e: reap_acc.freq.get((layer_ref.layer_idx, e), 0) for e in range(n_experts)}

        # Protected experts (super experts + shared experts from stage1_blacklist.json)
        # are completely excluded from REAM — not centroids, not non-centroids.
        # Their weights pass through Stage 2 unchanged (spec §5 "Blacklisted Expert Exclusion").
        n_protected = len(protected)

        effective_target = target
        ream_centroid_ids: list[int] = []
        ream_noncentroid_ids: list[int] = []
        grouped: dict[int, list[int]] = {}
        delta = np.empty(0)
        assignment: list[int] = []
        running_mean: float = 0.0
        mean_assigned_cost: float = 0.0
        assigned_cost: float = 0.0

        for _bump_attempt in range(n_experts - target + 1):
            # REAM centroid count = total target minus the protected slots.
            ream_target = max(effective_target - n_protected, 0)

            # Select top-ream_target non-protected experts by REAP score (descending).
            # This is the greedy centroid selection order: highest-saliency centroid
            # gets priority in the assignment pass (spec §5 Step 3).
            ream_centroid_ids = []
            for _e in np.argsort(-scores):
                if len(ream_centroid_ids) >= ream_target:
                    break
                e = int(_e)
                if e in protected:
                    continue
                if freq[e] < min_active_tokens:
                    continue
                ream_centroid_ids.append(e)

            if len(ream_centroid_ids) < ream_target:
                log.warning(
                    "  layer %d: REAM centroid selection yielded %d < %d — "
                    "%d candidate(s) filtered by reap_min_active_tokens=%d",
                    layer_ref.layer_idx, len(ream_centroid_ids), ream_target,
                    ream_target - len(ream_centroid_ids), min_active_tokens,
                )

            ream_centroid_set = set(ream_centroid_ids)
            ream_noncentroid_ids = [
                e for e in range(n_experts)
                if e not in protected and e not in ream_centroid_set
            ]

            n_ream_c  = len(ream_centroid_ids)
            n_ream_nc = len(ream_noncentroid_ids)

            # Feasibility check (spec §5 Step 3, reference ream/ream.py L60-62):
            # every non-centroid must be absorbable within the per-centroid cap.
            b_fail = (max_group_cap > 0) and (n_ream_nc > n_ream_c * max_group_cap)

            delta = np.empty(0)
            assignment = []
            mean_cost = 0.0
            c_fail = False

            if not b_fail:
                delta = _ream_cost_matrix(
                    layer_ref, ream_noncentroid_ids, ream_centroid_ids,
                    ream_acc=ream_acc,
                )
                assignment = _assign_children_to_centroids(
                    delta, n_ream_nc, n_ream_c, max_group_cap,
                )
                _iter_n_assigned = sum(1 for a in assignment if a >= 0)
                _iter_assigned_cost = (
                    sum(float(delta[ch, assignment[ch]])
                        for ch in range(n_ream_nc) if assignment[ch] >= 0)
                    if delta.size > 0 else 0.0
                )
                mean_cost = _iter_assigned_cost / max(_iter_n_assigned, 1)
                if len(_layer_mean_costs) >= 4:
                    running_mean = float(np.mean(_layer_mean_costs))
                    c_fail = mean_cost > running_mean * (1.0 + cost_sigma)

            if not b_fail and not c_fail:
                break

            bump = 1
            if c_fail:
                bump = max(bump, math.ceil(effective_target * cost_bump_ratio))
            new_effective = min(effective_target + bump, n_experts)
            if b_fail:
                log.warning(
                    "  layer %d: infeasible (ream_c=%d × cap=%d < nc=%d) — "
                    "bumping target %d→%d",
                    layer_ref.layer_idx, n_ream_c, max_group_cap, n_ream_nc,
                    effective_target, new_effective,
                )
            if c_fail:
                log.warning(
                    "  layer %d: mean_cost=%.4f > threshold=%.4f — bumping target %d→%d",
                    layer_ref.layer_idx, mean_cost,
                    running_mean * (1.0 + cost_sigma),
                    effective_target, new_effective,
                )
            effective_target = new_effective
            if effective_target >= n_experts:
                break

        # Finding 11 fallback: if the bump loop exhausted without achieving feasibility
        # (b_fail still True and no assignment was built), log a WARNING and fall back
        # to keeping all non-protected experts as centroids (zero merges). This is the
        # safest fallback — it produces the least compression but loses no expert weights.
        if b_fail and not assignment and ream_noncentroid_ids:
            log.warning(
                "  layer %d: bump loop exhausted (effective_target=%d == n_experts=%d) "
                "without achieving feasibility — falling back to zero-merge "
                "(all non-protected experts kept as centroids). "
                "No expert weights are lost, but compression target is not met.",
                layer_ref.layer_idx, effective_target, n_experts,
            )
            # Explicitly set ream_centroid_ids to all non-protected experts (zero-merge
            # fallback). We cannot rely on the last bump iteration's ream_centroid_ids
            # because the loop broke before recomputing it with the final effective_target.
            ream_centroid_ids = [
                e for e in range(n_experts) if e not in protected
            ]
            ream_noncentroid_ids = []

        # Build REAM merge groups (keyed by REAM centroid only — protected experts
        # are not in grouped and their weights are not touched by _merge_experts_inplace).
        grouped = {c: [c] for c in ream_centroid_ids}
        for child_pos, centroid_pos in enumerate(assignment):
            if centroid_pos >= 0:
                grouped[ream_centroid_ids[centroid_pos]].append(
                    ream_noncentroid_ids[child_pos]
                )

        assigned_cost = (
            sum(float(delta[ch, assignment[ch]])
                for ch in range(len(ream_noncentroid_ids)) if assignment[ch] >= 0)
            if delta.size > 0 else 0.0
        )
        n_assigned = sum(1 for a in assignment if a >= 0)
        mean_assigned_cost = assigned_cost / max(n_assigned, 1)

        if n_assigned > 0:
            _layer_mean_costs.append(mean_assigned_cost)

        _merge_experts_inplace(
            layer_ref, grouped, freq,
            freq_weighted=s2["ream"]["frequency_weighted_merge"],
            ream_acc=ream_acc,
        )

        # Final kept set = protected experts (untouched) + REAM centroids (post-merge).
        # Protected experts' rows are preserved in gate.weight and expert tensors.
        final_kept_ids = sorted(list(protected) + ream_centroid_ids)

        banks = build_banks(layer_ref)
        for bank in banks.values():
            bank.select(final_kept_ids)
        _resize_router_for_kept_experts(layer_ref, final_kept_ids)

        ream_acc.clear_layer(layer_ref.layer_idx)

        merge_map[layer_ref.layer_idx] = {
            new_idx: ([eid] if eid in protected else sorted(grouped[eid]))
            for new_idx, eid in enumerate(final_kept_ids)
        }
        _remap_covariance_for_layer(cov_acc, layer_ref.layer_idx, final_kept_ids)

        if partial_dir is not None:
            _snapshot_cov_layer(cov_acc, layer_ref.layer_idx, partial_dir)
            _write_merge_json(
                partial_dir, layer_ref.layer_idx, final_kept_ids, grouped, freq,
                merge_map[layer_ref.layer_idx],
                mean_cost_per_pair=mean_assigned_cost,
            )

        max_group = max((len(g) for g in grouped.values()), default=1)
        mean_group = len(ream_noncentroid_ids) / max(len(ream_centroid_ids), 1)
        log.info(
            "  kept %d / %d experts (protected=%d, ream_centroids=%d) — "
            "Σ cost=%.4f, max_group=%d, mean_group=%.2f",
            len(final_kept_ids), n_experts, n_protected, len(ream_centroid_ids),
            assigned_cost, max_group, mean_group,
        )
        _trackio_log({
            "stage2/layer_idx": layer_ref.layer_idx,
            "stage2/kept_experts": len(final_kept_ids),
            "stage2/protected_experts": n_protected,
            "stage2/ream_centroids": len(ream_centroid_ids),
            "stage2/total_experts": n_experts,
            "stage2/sum_assignment_cost": assigned_cost,
            "stage2/mean_cost_per_pair": mean_assigned_cost,
            "stage2/max_merge_group_size": max_group,
            "stage2/mean_merge_group_size": mean_group,
            "stage2/effective_target": len(final_kept_ids),
            "stage2/stage1_target": target,
        })

    out_dir = artifacts_dir / "stage2_pruned"
    _save_covariance(cov_acc, artifacts_dir / "_stage2_input_covariance.pt")
    save_compressed_checkpoint(
        model, tokenizer, out_dir,
        pipeline_stage="stage2_pruned",
        extra_metadata={"merge_map_file": "merge_map.json"},
    )
    save_json_artifact(merge_map, out_dir / "merge_map.json")
    if partial_dir is not None:
        shutil.rmtree(partial_dir, ignore_errors=True)
    log.info("Stage 2 complete — pruned checkpoint at %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Partial-resume helpers
# ---------------------------------------------------------------------------


def _snapshot_cov_layer(
    cov_acc: InputCovarianceAccumulator,
    layer_idx: int,
    partial_dir: Path,
) -> None:
    with cov_acc._lock:
        keys = [k for k in cov_acc.covariance if k[0] == layer_idx]
        if not keys:
            return
        payload = {
            "format_version": 1,
            "covariance": {k: cov_acc.covariance[k].clone() for k in keys},
            "tokens": {k: cov_acc.token_count.get(k, 0) for k in keys},
        }
    tmp = partial_dir / f"layer_{layer_idx}.pt.tmp"
    final = partial_dir / f"layer_{layer_idx}.pt"
    torch.save(payload, tmp)
    # Spec §11: durable write — fsync file bytes, then fsync parent dir entry,
    # then atomic rename so a crash never leaves a truncated final file.
    # O_WRONLY|O_APPEND is used for the .tmp file so fsync flushes write data
    # (O_RDONLY on a regular file does not guarantee flushing write buffers on POSIX).
    # The parent dir must use O_RDONLY (directories cannot be opened for write).
    fd = os.open(str(tmp), os.O_WRONLY | os.O_APPEND)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)
    parent_fd = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _write_merge_json(
    partial_dir: Path,
    layer_idx: int,
    final_kept_ids: list[int],
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    merge_map_layer: dict[int, list[int]],
    *,
    mean_cost_per_pair: float = 0.0,
) -> None:
    """Write the per-layer merge record to a durable JSON file.

    Args:
        partial_dir:      Directory for partial/crash-resume checkpoints.
        layer_idx:        MoE layer index.
        final_kept_ids:   Sorted list of all kept expert IDs after merging
                          (protected experts + REAM centroids). Stored under
                          ``"final_kept_ids"`` (renamed from the old
                          ``"centroid_ids"`` field in format_version 1; the
                          resume path accepts both names for backward compat).
        grouped:          Merge groups keyed by centroid expert ID.
        freq:             Per-expert token frequency counts.
        merge_map_layer:  New-index → original-expert-ids mapping for this layer.
        mean_cost_per_pair: Mean REAM assignment cost, for the budget-bump history.
    """
    payload = {
        "format_version": 1,
        "final_kept_ids": final_kept_ids,
        "grouped": {str(k): list(v) for k, v in grouped.items()},
        "freq": {str(k): int(v) for k, v in freq.items()},
        "merge_map_layer": {str(k): list(v) for k, v in merge_map_layer.items()},
        "mean_cost_per_pair": mean_cost_per_pair,
    }
    tmp = partial_dir / f"merge_{layer_idx}.json.tmp"
    final = partial_dir / f"merge_{layer_idx}.json"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    # Spec §11: durable write — fsync file bytes, then fsync parent dir entry,
    # then atomic rename so a crash never leaves a truncated final file.
    # O_WRONLY|O_APPEND is used for the .tmp file so fsync flushes write data
    # (O_RDONLY on a regular file does not guarantee flushing write buffers on POSIX).
    # The parent dir must use O_RDONLY (directories cannot be opened for write).
    fd = os.open(str(tmp), os.O_WRONLY | os.O_APPEND)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)
    parent_fd = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


# ---------------------------------------------------------------------------
# Per-layer profiling
# ---------------------------------------------------------------------------


def _profile_layer(
    model,
    layer_ref: MoELayerRef,
    batches,
    reap_acc: ReapAccumulator,
    cov_acc: InputCovarianceAccumulator,
    ream_acc: ReamCostAccumulator,
    *,
    device=None,
) -> None:
    """Profile a single MoE layer with early-exit forward.

    REAM sequential merging (paper 2604.04356, §4, Fig 1(b)) requires
    that each layer is profiled on hidden states reflecting all prior
    merges.  All metrics (REAP scores, REAM δ_gate/δ̃_expert, input
    covariance) depend only on hidden states arriving *at* this layer,
    not on downstream layers.  We therefore abort the forward pass
    immediately after this layer completes via :func:`early_exit_after_layer`,
    avoiding O(40−L) unnecessary layer-forwards per batch.

    Total layer-forwards across 40 sequential profiling passes:
    1+2+…+40 = 820 (vs 40×40 = 1600 without early exit).
    """
    layer_idx = layer_ref.layer_idx
    n_experts = layer_ref.num_routed_experts
    model.eval()

    # Cumulative token offset: tracks the global start index of each batch.
    # Using cumulative addition (not batch_idx * fixed_size) handles the last
    # partial batch when num_calibration_samples % batch_size != 0.
    _batch_offset = [0]
    _next_offset = [0]

    def input_cb(li, e, tensor, ctx):
        cov_acc.update(li, e, "gate_proj", tensor)

    def intermediate_cb(li, e, tensor, ctx):
        cov_acc.update(li, e, "down_proj", tensor)
        ream_acc.record_neuron_activations(li, e, tensor)

    def down_cb(li, e, tensor, ctx):
        record_reap(reap_acc, li, e, ctx["top_k_weights"], tensor)
        ream_acc.record_gated_output(
            li, e, ctx["top_k_weights"], tensor,
            ctx["token_idx"], _batch_offset[0],
        )

    with instrument_experts(
        layer_ref,
        {"input": input_cb, "intermediate": intermediate_cb, "down": down_cb},
    ), capture_router_outputs([layer_ref]) as router_logits_storage, \
         early_exit_after_layer(model, layer_idx):
        for batch_idx, batch in enumerate(batches):
            if device is not None:
                batch = batch.to(device)
            _batch_offset[0] = _next_offset[0]
            router_logits_storage[layer_idx].clear()
            with torch.no_grad():
                try:
                    model(input_ids=batch)
                except _EarlyExitException:
                    pass  # expected — target layer completed
            if router_logits_storage[layer_idx]:
                batch_logits = router_logits_storage[layer_idx][-1]
                ream_acc.record_router_logits(layer_idx, batch_logits, _batch_offset[0])
            ream_acc.finalize_batch(layer_idx, n_experts)
            ream_acc.record_batch_token_count(layer_idx, batch.shape[0] * batch.shape[1])
            _next_offset[0] += batch.shape[0] * batch.shape[1]


# ---------------------------------------------------------------------------
# REAM cost + assignment
# ---------------------------------------------------------------------------


def _ream_cost_matrix(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    ream_acc: ReamCostAccumulator,
) -> np.ndarray:
    if not noncentroid_ids or not centroid_ids:
        return np.zeros((len(noncentroid_ids), len(centroid_ids)))

    li = layer_ref.layer_idx
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)

    # Compute δ_gate for all (noncentroid, centroid) pairs in one matrix call.
    # The observed-max dist2sim requires a full N×N pairwise distance matrix;
    # a per-pair call cannot implement this correctly (Finding 1).
    # We build a combined list [noncentroid_ids..., centroid_ids...] and slice
    # the (noncentroid × centroid) submatrix from the full similarity matrix.
    all_ids = noncentroid_ids + centroid_ids
    sim_gate_full = ream_acc.compute_gate_similarity_matrix(li, all_ids)
    # Submatrix: rows = noncentriods (0..n_nc-1), cols = centroids (n_nc..n_nc+n_c-1)
    sim_gate_sub = sim_gate_full[:n_nc, n_nc:].numpy().astype(np.float64)  # (n_nc, n_c)

    cost = np.zeros((n_nc, n_c), dtype=np.float64)

    for ci in range(n_nc):
        child = noncentroid_ids[ci]
        for cj in range(n_c):
            centroid = centroid_ids[cj]
            sim_gate   = float(sim_gate_sub[ci, cj])
            sim_expert = ream_acc.compute_delta_expert(li, child, centroid)
            # δ_REAM = (δ_gate + δ̃_expert) / 2 ∈ [0,1]; cost = 1 − δ_REAM ∈ [0,1].
            # Lower cost = more similar (spec §5 Step 2, reference ream/ream.py L46-53).
            cost[ci, cj] = 1.0 - (sim_gate + sim_expert) / 2.0

    return cost


def _assign_children_to_centroids(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int = 0,
) -> list[int]:
    """Single-pass greedy assignment of non-centroid children to centroids.

    Iterates centroids once in order 0..n_centroids-1 (caller builds centroid_ids
    in descending saliency — column 0 = highest-saliency centroid).  For each
    centroid, greedily absorbs up to *max_group_cap* unassigned children (lowest
    cost = most similar first).  When max_group_cap=0 the cap is disabled and each
    centroid absorbs as many remaining unassigned children as exist.

    The caller is responsible for ensuring feasibility before calling:
    n_centroids * max_group_cap >= n_children (spec §5 Step 3). When the
    feasibility check passes, every child is guaranteed to receive assignment >= 0.

    Returns a list of length n_children where entry ch is:
      >= 0  → centroid column index this child is merged into
      -1    → child was not absorbed (should not occur if feasibility holds)
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children

    assignment = [-1] * n_children
    assigned: set[int] = set()

    # Single pass: iterate centroids in descending saliency order (column 0 first).
    # Note on group-cap semantics (spec §5 Step 3):
    #   max_group_cap counts non-centroids only (not the centroid itself), matching
    #   our spec §5 Step 3 ("absorb up to max_merge_group_size unassigned non-centroids").
    #   The REAM reference's group_size counts total members including the centroid,
    #   so our max_group_cap=8 is equivalent to reference group_size=9.
    # The feasibility check (b_fail) in the bump loop uses the same semantics:
    #   n_ream_nc > n_ream_c * max_group_cap  (non-centroids exceed total centroid capacity).
    for c_idx in range(n_centroids):
        # Slots available for this centroid: unlimited when cap disabled (0).
        slots = max_group_cap if max_group_cap > 0 else n_children
        absorbed = 0
        while absorbed < slots:
            best_child = -1
            best_cost = float("inf")
            for ch in range(n_children):
                if ch in assigned:
                    continue
                if cost[ch, c_idx] < best_cost:
                    best_cost = cost[ch, c_idx]
                    best_child = ch
            if best_child < 0:
                break  # no more unassigned children
            assignment[best_child] = c_idx
            assigned.add(best_child)
            absorbed += 1

    return assignment


# ---------------------------------------------------------------------------
# Merge + router resize + covariance I/O
# ---------------------------------------------------------------------------


def _permutation_align_to_centroid(
    ref_gate: torch.Tensor,
    ref_up: torch.Tensor,
    child_gate: torch.Tensor,
    child_up: torch.Tensor,
    ref_act_mean: torch.Tensor | None = None,
    child_act_mean: torch.Tensor | None = None,
) -> np.ndarray:
    C = (
        torch.cdist(ref_gate.cpu(), child_gate.cpu())
        + torch.cdist(ref_up.cpu(), child_up.cpu())
    )
    if ref_act_mean is not None and child_act_mean is not None:
        C = C + torch.cdist(
            ref_act_mean.cpu().unsqueeze(-1),
            child_act_mean.cpu().unsqueeze(-1),
        )
    _, col_ind = linear_sum_assignment(C.numpy())
    return col_ind


def _merge_experts_inplace(
    layer_ref: MoELayerRef,
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    *,
    freq_weighted: bool,
    ream_acc: ReamCostAccumulator | None = None,
) -> None:
    banks = build_banks(layer_ref)
    li = layer_ref.layer_idx
    with torch.no_grad():
        for centroid, members in grouped.items():
            if len(members) <= 1:
                continue
            weights = np.array([max(freq.get(m, 0), 1) for m in members], dtype=np.float64)
            if not freq_weighted:
                weights[:] = 1.0
            weights = weights / weights.sum()

            ref_gate = banks["gate_proj"].get(centroid).to(torch.float32)
            ref_up   = banks["up_proj"].get(centroid).to(torch.float32)
            ref_act  = ream_acc.get_neuron_mean(li, centroid) if ream_acc else None

            accs: dict[str, torch.Tensor | None] = {name: None for name in banks}
            for w, m in zip(weights, members):
                gate_m = banks["gate_proj"].get(m).to(torch.float32)
                up_m   = banks["up_proj"].get(m).to(torch.float32)
                child_act = ream_acc.get_neuron_mean(li, m) if ream_acc else None
                perm = (
                    None if m == centroid
                    else _permutation_align_to_centroid(
                        ref_gate, ref_up, gate_m, up_m,
                        ref_act_mean=ref_act, child_act_mean=child_act,
                    )
                )
                for name, bank in banks.items():
                    if name == "gate_proj":
                        Wm = gate_m
                    elif name == "up_proj":
                        Wm = up_m
                    else:
                        Wm = bank.get(m).to(torch.float32)
                    if perm is not None:
                        Wm = Wm[perm, :] if name in ("gate_proj", "up_proj") else Wm[:, perm]
                    accs[name] = Wm * w if accs[name] is None else accs[name] + Wm * w

            for name, bank in banks.items():
                bank.set(centroid, accs[name])


def _resize_router_for_kept_experts(layer_ref: MoELayerRef, kept_ids: list[int]) -> None:
    router = layer_ref.router
    idx = torch.as_tensor(kept_ids, device=router.weight.device, dtype=torch.long)
    with torch.no_grad():
        new_w = router.weight.data.index_select(0, idx).contiguous().clone()
        router.weight = nn.Parameter(new_w, requires_grad=router.weight.requires_grad)
        if getattr(router, "bias", None) is not None:
            new_b = router.bias.data.index_select(0, idx).contiguous().clone()
            router.bias = nn.Parameter(new_b, requires_grad=router.bias.requires_grad)
    router.num_experts = len(kept_ids)
    if hasattr(router, "top_k") and router.top_k > len(kept_ids):
        router.top_k = len(kept_ids)

    mlp = layer_ref.mlp
    if hasattr(mlp, "num_experts"):
        mlp.num_experts = len(kept_ids)


def _save_covariance(cov: InputCovarianceAccumulator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"covariance": cov.covariance, "tokens": dict(cov.token_count)}, path)
    log.info("Saved Stage 2 input covariance to %s", path)


def _remap_covariance_for_layer(
    cov: InputCovarianceAccumulator,
    layer_idx: int,
    centroid_ids: list[int],
) -> None:
    id_to_new = {old: new for new, old in enumerate(centroid_ids)}
    new_cov: dict = {}
    new_tokens: dict = {}
    with cov._lock:
        for key, val in list(cov.covariance.items()):
            li, eidx, name = key
            if li != layer_idx:
                new_cov[key] = val
                new_tokens[key] = cov.token_count.get(key, 0)
                continue
            if eidx not in id_to_new:
                continue
            new_key = (li, id_to_new[eidx], name)
            new_cov[new_key] = val
            new_tokens[new_key] = cov.token_count.get(key, 0)
        cov.covariance, cov.token_count = new_cov, new_tokens
