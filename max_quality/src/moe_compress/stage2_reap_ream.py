"""Stage 2 — REAP scoring + REAM pseudo-pruning, fused-experts-aware.

Key differences from the pre-refactor version:
  - Weights live in stacked tensors on ``Qwen3_5MoeExperts``; pruning means
    slicing those tensors and the router's ``gate.weight`` rows.
  - Scoring hooks go through :func:`instrument_experts` which monkey-patches
    the fused forward with per-expert callbacks.
  - Input covariance for Stage 3 is collected on two tap points:
      ``"gate_proj"``   → covariance used by gate_proj + up_proj SVD
      ``"down_proj"``   → covariance used by down_proj SVD
    Keys match those used by ``InputCovarianceAccumulator`` (``'gate_proj'``
    covers gate+up projections; ``'down_proj'`` covers the down projection).
    We save these under the (layer, expert, matrix_name) key space that
    Stage 3 consumes.

REAM cost matrix (paper 2604.04356, reference ream/ream.py):
  - δ_gate (Eq. 5): similarity ∈ [0,1] between L2-row-normalized pre-softmax
    gate logit profile vectors — Euclidean distance converted via dist2sim.
  - δ̃_expert (Eq. 8): mean cosine similarity of expert outputs (sparse top-k
    approximation; see `compute_delta_expert` in `activation_hooks.py`),
    rescaled to [0,1] via (cosine+1)/2.
  - δ_REAM = (δ_gate + δ̃_expert) / 2 ∈ [0,1]; cost = 1 − δ_REAM.
  - Grouping: single-pass greedy procedure matching paper §4 exactly (descending
    centroid saliency, absorb up to C nearest unassigned non-centroids per centroid).
    The paper prescribes greedy, not optimal matching; this is spec-compliant.
    Full assignment guaranteed by upfront feasibility check.

Frequency-weighted merge with neuron permutation alignment is preserved.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    batches = iter_batches(calib, batch_size=s2["batch_size"])
    assert isinstance(batches, list), "iter_batches must return a list for multi-pass re-iteration"

    moe_layers = list(iter_moe_layers(model))
    cov_acc = InputCovarianceAccumulator()
    # Spec §5 "Covariance Side-Collection": FP32 storage is recommended by
    # Swift-SVD paper 2604.01609 (avoids numerical degradation in eigendecomposition);
    # the dtype is configurable via covariance_storage_dtype.
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
            shutil.rmtree(stale, ignore_errors=True)
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
                if merge_path.exists() and not cov_path.exists():
                    log.warning("layer %d: found merge JSON but missing covariance .pt; re-running layer", ref.layer_idx)
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
                final_kept_ids = [int(x) for x in data["final_kept_ids"]]
            elif "centroid_ids" in data:
                log.warning(
                    "Stage 2 resume layer %d: found deprecated 'centroid_ids' field "
                    "(expected 'final_kept_ids') — using it for backward compatibility. "
                    "Delete _stage2_partial/ to regenerate with the new format.",
                    ref.layer_idx,
                )
                final_kept_ids = [int(x) for x in data["centroid_ids"]]
            else:
                raise RuntimeError(
                    f"_stage2_partial/merge_{ref.layer_idx}.json missing both "
                    "'final_kept_ids' and 'centroid_ids' keys — file is corrupt. "
                    "Delete _stage2_partial/ and re-run Stage 2."
                )
            grouped = {int(k): list(v) for k, v in data["grouped"].items()}
            freq = {int(k): int(v) for k, v in data["freq"].items()}
            _freq_keys = set(freq.keys())
            if _freq_keys != set(range(len(freq))):
                raise RuntimeError(
                    f"Layer {ref.layer_idx}: loaded freq has non-contiguous or unexpected keys — "
                    "delete partial checkpoint and re-run from scratch"
                )
            merge_map_layer = {int(k): list(v) for k, v in data["merge_map_layer"].items()}

            _keys = set(merge_map_layer.keys())
            if _keys != set(range(len(merge_map_layer))):
                raise RuntimeError(
                    f"Layer {ref.layer_idx}: loaded merge_map has non-contiguous keys {_keys} — "
                    "delete partial checkpoint and re-run from scratch"
                )
            if any(not v for v in merge_map_layer.values()):
                raise RuntimeError(
                    f"Layer {ref.layer_idx}: loaded merge_map has empty member lists — "
                    "delete partial checkpoint and re-run from scratch"
                )

            # n_pre_merge is derived from len(freq) rather than a dedicated persisted
            # field. This is safe because freq is written with exactly one key per
            # expert (range(n_experts)) at calibration time, so len(freq) always equals
            # the original expert count for this layer before any merging.
            n_pre_merge = len(freq)
            if ref.num_routed_experts != n_pre_merge:
                raise RuntimeError(
                    f"Stage 2 resume layer {ref.layer_idx}: expected {n_pre_merge} "
                    f"experts (pre-merge) but model has {ref.num_routed_experts}. "
                    "The model passed to stage2.run() must be the Stage 1 output, "
                    "not a partially-merged model."
                )

            # ream_acc is not passed here: neuron means are not persisted to disk,
            # so permutation alignment on resume uses only gate/up weights.
            # B-C-L-1 TODO: persisting neuron-mean tensors alongside merge_{layer}.json
            # would weaken the spec invariant C = C_wt + C_act on resume. Currently
            # the resume path uses C = C_wt only (no C_act); fix would require a
            # larger artifact-format change to save/restore the neuron-mean tensors.
            log.warning(
                "layer %d (resume): neuron-mean activation data not available on resume — "
                "permutation alignment uses weight-only cost (C_gate + C_up, no C_act). "
                "Merged weights may differ from a fresh run.",
                ref.layer_idx,
            )
            _merge_experts_inplace(ref, grouped, freq,
                                   freq_weighted=s2["ream"]["frequency_weighted_merge"])
            # build_banks again: _merge_experts_inplace already called it internally, but
            # bank.select() was never called on any of those banks, so the _last_kept_ids_*
            # sentinel is still unset and this select() call is safe.
            banks = build_banks(ref)
            for bank in banks.values():
                bank.select(final_kept_ids)
            _resize_router_for_kept_experts(ref, final_kept_ids)

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

    # B-C-H-1: default to 8 (D5a value) so REAM merging always has a per-centroid
    # cap, preventing degenerate one-centroid-absorbs-all groupings. Setting to 0
    # explicitly disables the cap (uncapped path); users must opt in to that.
    max_group_cap: int = int(s2.get("max_merge_group_size", 8) or 8)
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
        ream_acc = ReamCostAccumulator()  # fresh accumulator per layer; discarded after this layer's pass
        torch.cuda.empty_cache()
        _profile_layer(
            model, layer_ref, batches, reap_acc, cov_acc, ream_acc,
            device=device,
        )
        # These two finalize calls are independent of each other and could be
        # parallelised (e.g., via concurrent.futures) if profiling shows this
        # is a bottleneck in future.
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

        if target > n_experts:
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: budget target {target} > n_experts {n_experts}; "
                "budget allocation is inconsistent with layer expert count"
            )
        if target == n_experts:
            log.warning(
                "layer %d: budget target (%d) equals total expert count (%d) — "
                "no merging will occur; check budget configuration.",
                layer_ref.layer_idx, target, n_experts,
            )

        effective_target = target
        ream_centroid_ids: list[int] = []
        ream_noncentroid_ids: list[int] = []
        grouped: dict[int, list[int]] = {}
        delta = np.empty((0, 0))
        assignment: list[int] = []
        running_mean: float = float("nan")
        mean_assigned_cost: float = 0.0
        assigned_cost: float = 0.0
        # Invariant: after the bump loop, assignment is either:
        #   (a) a list of length len(ream_noncentroid_ids) with centroid indices (normal path), or
        #   (b) [] with ream_noncentroid_ids also [] (zero-merge fallback path).
        # (c) c_fail last-resort: assignment holds the last above-threshold assignment
        #     (len == len(ream_noncentroid_ids)); applied as best-available merge below.
        # b_fail / c_fail are initialized here so the post-loop fallback check never raises
        # NameError if the range were somehow empty.
        b_fail: bool = False
        c_fail: bool = False
        _warned_ream_target_zero: bool = False

        _original_ream_target = max(effective_target - n_protected, 0)  # target on first attempt

        # Loop runs (1 + n_experts - target) times: 1 initial attempt plus up to
        # (n_experts - target) bumps, one per additional kept expert.
        for _bump_attempt in range(n_experts - target + 1):
            # REAM centroid count = total target minus the protected slots.
            ream_target = max(effective_target - n_protected, 0)

            if ream_target == 0:
                if not _warned_ream_target_zero:
                    log.warning(
                        "layer %d: ream_target=0 — all %d non-protected experts will be dropped "
                        "(budget fully consumed by %d protected experts); "
                        "check budget configuration.",
                        layer_ref.layer_idx, n_experts - len(protected), len(protected),
                    )
                    _warned_ream_target_zero = True
                break

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

            delta = np.empty((0, 0))
            assignment = []
            mean_cost = 0.0
            c_fail = False

            if not b_fail:
                delta = _ream_cost_matrix(
                    layer_ref, ream_noncentroid_ids, ream_centroid_ids,
                    ream_acc=ream_acc,
                    blacklisted_ids=protected,
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
                if _iter_n_assigned == 0 and n_ream_nc == 0:
                    # No non-centroid experts exist — nothing to merge, cost is
                    # genuinely zero.  Skip the c_fail gate entirely: there is no
                    # merge to gate on, and inf would cause a spurious bump.
                    mean_cost = 0.0
                    # c_fail remains False (already set above); do not evaluate gate.
                else:
                    # When nothing was assigned despite having non-centroids, use inf
                    # rather than 0.0: a zero mean_cost would be a false negative,
                    # making an unassigned layer look cheaper than any real merge and
                    # preventing the cost-threshold bump from triggering.
                    mean_cost = (
                        _iter_assigned_cost / _iter_n_assigned
                        if _iter_n_assigned > 0 else float("inf")
                    )
                    # Require at least 4 prior-layer samples before applying the cost-sigma
                    # gate: fewer samples make the running mean too noisy to be meaningful.
                    # Invariant: running_mean is always computed in the same branch as
                    # c_fail = True, so running_mean is guaranteed to be set before
                    # c_fail can become True. Future refactors must preserve this ordering
                    # to avoid referencing running_mean when it is still 0.0 (its default).
                    if len(_layer_mean_costs) >= 4:
                        running_mean = float(np.mean(_layer_mean_costs))
                        c_fail = mean_cost > running_mean * (1.0 + cost_sigma)

            if not b_fail and not c_fail:
                break

            bump = 1
            if c_fail:
                # max(..., 1) serves as a floor: ceil(0 * ratio)=0 is unreachable in
                # practice (effective_target starts at target >= 1), but kept for safety.
                bump = max(bump, math.ceil(effective_target * cost_bump_ratio))
            new_effective = min(effective_target + bump, n_experts)
            if b_fail:
                log.warning(
                    "  layer %d: infeasible (ream_c=%d × cap=%d < nc=%d) — "
                    "bumping target %d→%d",
                    layer_ref.layer_idx, n_ream_c, max_group_cap, n_ream_nc,
                    effective_target, new_effective,
                )
            # running_mean is always current here: c_fail=True can only be set inside the
            # cost block (not b_fail path), which assigns running_mean before setting c_fail.
            if c_fail:
                assert not math.isnan(running_mean), (
                    "running_mean must be set before c_fail can be True; "
                    "check that the c_fail assignment is co-located with the running_mean assignment"
                )
                log.warning(
                    "  layer %d: mean_cost=%.4f > threshold=%.4f — bumping target %d→%d",
                    layer_ref.layer_idx, mean_cost,
                    running_mean * (1.0 + cost_sigma),
                    effective_target, new_effective,
                )
            effective_target = new_effective
            # We break BEFORE computing a new assignment at effective_target==n_experts;
            # the last assignment from the previous iteration is used as the fallback.
            if effective_target >= n_experts:
                break

        # Post-loop: if the loop exited because effective_target >= n_experts but c_fail
        # was still True (cost gate never cleared), the last above-threshold assignment
        # is used as last resort. Warn so this silent state is observable.
        if c_fail and effective_target >= n_experts:
            log.warning(
                "REAM layer %d: bump loop exhausted (c_fail=True, b_fail=%s, effective_target=%d >= n_experts=%d); "
                "applying above-threshold assignment as last resort",
                layer_ref.layer_idx, b_fail, effective_target, n_experts,
            )
        # Fallback: if the bump loop exhausted without achieving feasibility
        # (b_fail still True and no assignment was built), log a WARNING and fall back
        # to keeping all non-protected experts as centroids (zero merges). This is the
        # safest fallback — it produces the least compression but loses no expert weights.

        # Zero-target case: budget fully consumed by protected experts — no REAM
        # centroids or non-centroids should exist and no merges should be produced.
        # The bump loop broke out early, so ream_centroid_ids/ream_noncentroid_ids/
        # assignment may still hold stale values from a previous attempt (or their
        # initial [] defaults). Reset them explicitly so the grouping code below
        # produces an empty grouped dict and all protected experts flow to final_kept_ids.
        if _original_ream_target == 0:
            ream_centroid_ids = []
            ream_noncentroid_ids = []
            assignment = []
            delta = np.empty((0, 0))
            b_fail = False
            c_fail = False

        # When b_fail: assignment is [] (reset at iteration top; b_fail skips _assign_children_to_centroids).
        # When c_fail last-resort (effective_target >= n_experts break): assignment holds the last
        # computed above-threshold result and is intentionally applied in the grouping step below.
        if b_fail and ream_noncentroid_ids:
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
            assignment = []
            delta = np.empty((0, 0))

        if not ream_centroid_ids and ream_noncentroid_ids and _original_ream_target > 0:
            log.warning(
                "REAM layer %d: no centroids selected (all non-protected experts may have failed "
                "min_active_tokens or cost gate); promoting all non-protected experts to singleton "
                "centroids (zero-merge fallback).",
                layer_ref.layer_idx,
            )
            ream_centroid_ids = list(ream_noncentroid_ids)
            ream_noncentroid_ids = []
            assignment = []
            delta = np.empty((0, 0))

        # Build REAM merge groups (keyed by REAM centroid only — protected experts
        # are not in grouped and their weights are not touched by _merge_experts_inplace).
        grouped = {c: [c] for c in ream_centroid_ids}
        # Protected experts should never appear as REAM centroids; verify the invariant.
        _protected_centroids = [eid for eid in protected if eid in grouped]
        if _protected_centroids:
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: protected expert(s) {_protected_centroids} "
                "appeared as REAM centroids — invariant violated"
            )
        for child_pos, centroid_pos in enumerate(assignment):
            if centroid_pos >= 0:
                grouped[ream_centroid_ids[centroid_pos]].append(
                    ream_noncentroid_ids[child_pos]
                )

        for child_pos, centroid_pos in enumerate(assignment):
            if centroid_pos < 0:
                # Unassigned non-centroid: promote to singleton centroid to avoid weight loss.
                orphan_eid = ream_noncentroid_ids[child_pos]
                log.warning(
                    "layer %d: non-centroid expert %d unassigned in capped grouping — "
                    "promoted to singleton centroid to avoid weight loss",
                    layer_ref.layer_idx, orphan_eid,
                )
                grouped[orphan_eid] = [orphan_eid]
                ream_centroid_ids.append(orphan_eid)
        ream_centroid_ids = sorted(set(ream_centroid_ids))

        assigned_cost = (
            sum(float(delta[ch, assignment[ch]])
                for ch in range(len(ream_noncentroid_ids)) if assignment[ch] >= 0)
            if delta.size > 0 else 0.0
        )
        n_assigned = sum(1 for a in assignment if a >= 0)
        mean_assigned_cost = assigned_cost / max(n_assigned, 1)

        # Guard mirrors the resume-path condition (val > 0.0): exclude zero costs
        # so that layers with all-zero pair costs don't bias the running mean low
        # and suppress the cost-sigma bump gate for subsequent layers.
        # Also exclude last-resort c_fail assignments (bump loop exhausted with
        # effective_target >= n_experts) — those costs would inflate the running mean
        # and progressively suppress the c_fail gate for subsequent layers.
        if n_assigned > 0 and mean_assigned_cost > 0.0 and not (c_fail and effective_target >= n_experts):
            _layer_mean_costs.append(mean_assigned_cost)

        _merge_experts_inplace(
            layer_ref, grouped, freq,
            freq_weighted=s2["ream"]["frequency_weighted_merge"],
            ream_acc=ream_acc,
        )

        # Final kept set = protected experts (untouched) + REAM centroids (post-merge).
        # Protected experts' rows are preserved in gate.weight and expert tensors.
        final_kept_ids = sorted(list(protected) + ream_centroid_ids)

        if not final_kept_ids:
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: final_kept_ids is empty after merge — "
                "target may be inconsistent with protected/blacklisted expert counts"
            )

        banks = build_banks(layer_ref)
        for bank in banks.values():
            bank.select(final_kept_ids)
        _resize_router_for_kept_experts(layer_ref, final_kept_ids)

        # Correctness depends on the RuntimeError guard above ensuring no protected expert
        # appears in grouped. Without that guard, the else-branch would silently emit [eid]
        # instead of the full merge group for a protected expert that was also a centroid.
        merge_map[layer_ref.layer_idx] = {
            new_idx: (sorted(grouped[eid]) if eid in grouped else [eid])
            for new_idx, eid in enumerate(final_kept_ids)
        }
        # Ordering critical: remap to post-merge indices BEFORE snapshotting.
        # Writing pre-remap covariance would silently corrupt the resume path.
        _remap_covariance_for_layer(cov_acc, layer_ref.layer_idx, final_kept_ids)

        if partial_dir is not None:
            _snapshot_cov_layer(cov_acc, layer_ref.layer_idx, partial_dir)
            _write_merge_json(
                partial_dir, layer_ref.layer_idx, final_kept_ids, grouped, freq,
                merge_map[layer_ref.layer_idx],
                mean_cost_per_pair=(
                    mean_assigned_cost
                    if n_assigned > 0 and mean_assigned_cost > 0.0 and not (c_fail and effective_target >= n_experts)
                    else None
                ),
            )

        max_group = max((len(g) for g in grouped.values()), default=0)
        n_noncentroid_members = sum(len(g) - 1 for g in grouped.values())
        mean_group = n_noncentroid_members / len(grouped) if grouped else 0.0
        log.info(
            "  kept %d / %d experts (protected=%d, ream_centroids=%d) — "
            "Σ cost=%.4f, max_group=%d, mean_group=%.2f",
            len(final_kept_ids), n_experts, n_protected, len(ream_centroid_ids),
            assigned_cost, max_group, mean_group,
        )
        _trackio_log({
            "stage2/layer_idx": layer_ref.layer_idx,
            "stage2/protected_experts": n_protected,
            "stage2/ream_centroids": len(ream_centroid_ids),
            "stage2/total_experts": n_experts,
            "stage2/sum_assignment_cost": assigned_cost,
            "stage2/mean_cost_per_pair": mean_assigned_cost if n_assigned > 0 else float("nan"),
            "stage2/max_merge_group_size": max_group,
            "stage2/mean_merge_group_size": mean_group,
            "stage2/effective_target": effective_target,
            "stage2/actual_kept_experts": len(final_kept_ids),
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


def _durable_rename(tmp: Path, final: Path) -> None:
    """Fsync *tmp*, atomically rename it to *final*, then fsync the parent dir.

    Spec §11: durable write — fsync file bytes, then fsync parent dir entry,
    then atomic rename so a crash never leaves a truncated final file.
    O_WRONLY|O_APPEND is used for the .tmp file so fsync flushes write data
    (O_RDONLY on a regular file does not guarantee flushing write buffers on POSIX).
    The parent dir must use O_RDONLY (directories cannot be opened for write).

    Note: the tmp file must already be closed (all Python I/O buffers flushed to
    the kernel) before calling _durable_rename; the fsync it performs flushes
    kernel buffers, not Python-level buffers.
    """
    fd = os.open(str(tmp), os.O_WRONLY | os.O_APPEND)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)
    # tmp and final share the same parent directory (both are created in the same
    # directory by all callers), so final.parent == tmp.parent and the fsync below
    # correctly flushes the directory entry for the rename regardless of which path
    # is used.
    parent_fd = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _snapshot_cov_layer(
    cov_acc: InputCovarianceAccumulator,
    layer_idx: int,
    partial_dir: Path,
) -> None:
    with cov_acc._lock:
        keys = [k for k in cov_acc.covariance if k[0] == layer_idx]
        if not keys:
            log.debug("_snapshot_cov_layer: no covariance entries for layer %d; skipping snapshot", layer_idx)
            return
        payload = {
            "format_version": 1,
            "covariance": {k: cov_acc.covariance[k].clone() for k in keys},
            "tokens": {k: cov_acc.token_count.get(k, 0) for k in keys},
        }
    tmp = partial_dir / f"layer_{layer_idx}.pt.tmp"
    final = partial_dir / f"layer_{layer_idx}.pt"
    torch.save(payload, tmp)
    _durable_rename(tmp, final)


def _write_merge_json(
    partial_dir: Path,
    layer_idx: int,
    final_kept_ids: list[int],
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    merge_map_layer: dict[int, list[int]],
    *,
    mean_cost_per_pair: float | None = None,
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
        # list(v) ensures JSON gets a plain list, not a subclass that might not serialize
        "grouped": {str(k): list(v) for k, v in grouped.items()},
        "freq": {str(k): int(v) for k, v in freq.items()},
        # list(v) ensures JSON gets a plain list, not a subclass that might not serialize
        "merge_map_layer": {str(k): list(v) for k, v in merge_map_layer.items()},
        "mean_cost_per_pair": mean_cost_per_pair,
    }
    tmp = partial_dir / f"merge_{layer_idx}.json.tmp"
    final = partial_dir / f"merge_{layer_idx}.json"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    _durable_rename(tmp, final)


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
    was_training = model.training
    model.eval()

    # Cumulative token offset: tracks the global start index of each batch.
    # Using cumulative addition (not batch_idx * fixed_size) handles the last
    # partial batch when num_calibration_samples % batch_size != 0.
    _batch_offset = 0  # cumulative token start of current batch
    _next_offset = 0   # cumulative token count after current batch
    # B-C-C-1: full-softmax cache for the current batch's router logits.
    # Spec §5 line 339 + D-ream-sparse-routing require σ(x)_e (the
    # un-renormalized full softmax over ALL experts), not the top-k
    # renormalized weights returned by Qwen3_5MoeTopKRouter.forward.
    # Populated by an experts-module pre-forward hook that runs AFTER the
    # router pre-forward hook (which captures the raw logits) but BEFORE any
    # expert forward (which fires down_cb). down_cb reads _full_softmax[0]
    # to obtain σ(x)_e at active token positions.
    _full_softmax: list[torch.Tensor | None] = [None]

    def input_cb(li, e, tensor, ctx):
        cov_acc.update(li, e, "gate_proj", tensor)

    def intermediate_cb(li, e, tensor, ctx):
        cov_acc.update(li, e, "down_proj", tensor)
        ream_acc.record_neuron_activations(li, e, tensor)

    def down_cb(li, e, tensor, ctx):
        # _batch_offset is only read here, never assigned; no nonlocal declaration needed.
        record_reap(reap_acc, li, e, ctx["top_k_weights"], tensor)
        # B-C-C-1: pass σ(x)_e (full softmax over all experts) at active token
        # positions for this expert, NOT ctx["top_k_weights"] (renormalized to
        # sum=1 over top-k). The pre-forward hook installed below populates
        # _full_softmax[0] before any expert forward fires.
        token_idx = ctx["token_idx"]
        fs = _full_softmax[0]
        if fs is not None:
            # Index the cached [T, n_experts] full-softmax tensor at the
            # active token positions for this expert. Result shape: [|active|].
            # Move to device of expert_output to avoid a CPU↔GPU mismatch in
            # the (gate * expert_output) multiplication inside record_gated_output.
            sigma_e = fs[token_idx.cpu(), e].to(tensor.device)
        else:
            log.warning(
                "down_cb: full-softmax cache empty for layer %d expert %d — "
                "falling back to top_k_weights (renormalized; spec-degraded).",
                li, e,
            )
            sigma_e = ctx["top_k_weights"]
        ream_acc.record_gated_output(
            li, e, sigma_e, tensor,
            token_idx, _batch_offset,
        )

    # B-C-C-1: pre-forward hook on the experts module that computes the full
    # softmax from the latest captured router logits. Runs after the router
    # pre-forward hook (which appends to router_logits_storage[layer_idx])
    # but before the experts forward (which fires down_cb). Because
    # capture_router_outputs's hook is a *router* pre-forward hook and this
    # one is an *experts* pre-forward hook, ordering is guaranteed by the
    # decoder layer's call sequence (router runs first, dispatches to experts).
    def _populate_full_softmax(_module, _inputs):
        if router_logits_storage[layer_idx]:
            batch_logits = router_logits_storage[layer_idx][-1]
            # F.softmax over the last (expert) dim → [T, n_experts] σ(x)_e values.
            # .float() avoids dtype mismatch when the router runs in bf16.
            _full_softmax[0] = F.softmax(batch_logits.float(), dim=-1).cpu()
        else:
            _full_softmax[0] = None

    try:
        with instrument_experts(
            layer_ref,
            {"input": input_cb, "intermediate": intermediate_cb, "down": down_cb},
        ), capture_router_outputs([layer_ref]) as router_logits_storage, \
             early_exit_after_layer(model, layer_idx):
            # Install the experts pre-forward hook AFTER capture_router_outputs
            # so the router hook fires first per batch.
            _experts_handle = layer_ref.experts_module.register_forward_pre_hook(
                _populate_full_softmax
            )
            try:
                for batch in batches:
                    if device is not None:
                        batch = batch.to(device)
                    _batch_offset = _next_offset
                    router_logits_storage[layer_idx].clear()
                    _full_softmax[0] = None
                    with torch.no_grad():
                        try:
                            model(input_ids=batch)
                        except _EarlyExitException:
                            pass  # expected — target layer completed
                    if router_logits_storage[layer_idx]:
                        batch_logits = router_logits_storage[layer_idx][-1]
                        ream_acc.record_router_logits(layer_idx, batch_logits, _batch_offset)
                    ream_acc.finalize_batch(layer_idx, n_experts)
                    ream_acc.record_batch_token_count(layer_idx, batch.shape[0] * batch.shape[1])
                    _next_offset += batch.shape[0] * batch.shape[1]
            finally:
                _experts_handle.remove()
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# REAM cost + assignment
# ---------------------------------------------------------------------------


def _ream_cost_matrix(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    ream_acc: ReamCostAccumulator,
    blacklisted_ids: set[int] | None = None,
) -> np.ndarray:
    if not noncentroid_ids or not centroid_ids:
        # Early return produces shape (0, n_c) or (n_nc, 0) rather than (0, 0),
        # which is intentional. Callers guard with `delta.size > 0`, which correctly
        # handles all three degenerate shapes without special-casing each.
        return np.zeros((len(noncentroid_ids), len(centroid_ids)))

    li = layer_ref.layer_idx
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)
    n_experts_total = layer_ref.num_routed_experts

    # Compute δ_gate over the non-protected expert population so that dist2sim
    # normalizes by the global maximum distance among non-protected experts
    # (spec §5 Step 2, REAM ref ream/ream.py lines 37-41). Including protected
    # (super-expert) IDs would let their extreme gate-logit distances dominate
    # d.max(), compressing all noncentroid–centroid similarities toward 1.0
    # — DIST2SIM-PROTECTED-BIAS.
    protected_set = set(blacklisted_ids) if blacklisted_ids else set()
    all_n_ids = [e for e in range(n_experts_total) if e not in protected_set]
    _nc_protected = set(noncentroid_ids) & protected_set
    _c_protected  = set(centroid_ids)    & protected_set
    if _nc_protected or _c_protected:
        raise ValueError(
            f"_ream_cost_matrix: noncentroid_ids or centroid_ids overlap with blacklisted_ids "
            f"(nc={_nc_protected}, c={_c_protected})"
        )
    sim_gate_full = ream_acc.compute_gate_similarity_matrix(li, all_n_ids)
    # id_to_full_row maps expert ID → row index in all_n_ids.
    # Invariant: Stage 2 profiles each layer before merging it, so expert IDs are
    # always pre-merge [0, n_experts_total) when _ream_cost_matrix is called.
    id_to_full_row = {e: i for i, e in enumerate(all_n_ids)}
    # Extract the (n_nc × n_c) submatrix from the full N×N matrix.
    nc_rows = [id_to_full_row[e] for e in noncentroid_ids]
    c_cols  = [id_to_full_row[e] for e in centroid_ids]
    sim_gate_sub = sim_gate_full[np.ix_(nc_rows, c_cols)].numpy().astype(np.float64)  # (n_nc, n_c)

    cost = np.zeros((n_nc, n_c), dtype=np.float64)

    for ci in range(n_nc):
        child = noncentroid_ids[ci]
        for cj in range(n_c):
            centroid = centroid_ids[cj]
            sim_gate   = float(sim_gate_sub[ci, cj])
            sim_expert = ream_acc.compute_delta_expert(li, child, centroid)
            if math.isnan(sim_expert):
                # B-C-N-2: When δ̃_expert returns NaN (no joint activations),
                # substitute 0.0 (no similarity); biases the cost up to
                # 0.5 + 0.5*δ_gate per D-ream-sparse-routing.
                sim_expert = 0.0  # no profiling data; treat as no similarity
            # δ_REAM = (δ_gate + δ̃_expert) / 2 ∈ [0,1]; cost = 1 − δ_REAM ∈ [0,1].
            # Lower cost = more similar (spec §5 Step 2, reference ream/ream.py L46-53).
            cost[ci, cj] = 1.0 - (sim_gate + sim_expert) / 2.0

    np.clip(cost, 0.0, 1.0, out=cost)
    return cost


def _assign_children_to_centroids(
    cost: np.ndarray, n_children: int, n_centroids: int, max_group_cap: int = 0,
) -> list[int]:
    """Single-pass greedy assignment of non-centroid children to centroids.

    NOTE: This implements the greedy pseudo-pruning procedure described in paper §4
    exactly — the paper prescribes greedy (not optimal matching) for centroid grouping.
    Optimal matching (Hungarian) appears only in intra-group neuron permutation alignment.

    When max_group_cap == 0 (uncapped), each child is independently assigned to its
    nearest centroid by cost (argmin over centroid columns), so every child is
    assigned and no centroid monopolises all children due to iteration order.

    When max_group_cap > 0, iterates centroids once in order 0..n_centroids-1
    (caller builds centroid_ids in descending saliency — column 0 = highest-saliency
    centroid).  For each centroid, greedily absorbs up to *max_group_cap* unassigned
    children (lowest cost = most similar first).

    The caller is responsible for ensuring feasibility before calling:
    n_centroids * max_group_cap >= n_children (spec §5 Step 3). When the
    feasibility check passes, every child is guaranteed to receive assignment >= 0.
    This guarantee assumes `n_centroids >= 1`; when `n_centroids == 0` all children
    are assigned -1 (no centroid).

    Returns a list of length n_children where entry ch is:
      >= 0  → centroid column index this child is merged into
      -1    → child was not absorbed (should not occur if feasibility holds)
    """
    if n_children == 0 or n_centroids == 0:
        return [-1] * n_children

    if max_group_cap == 0:
        # Uncapped: assign each child to its nearest centroid by cost.
        # Iterating children (not centroids) avoids the centroid-order bias that
        # causes centroid 0 to absorb all children in the capped greedy path.
        assignment = [-1] * n_children
        for ch in range(n_children):
            best_c = int(np.argmin(cost[ch, :]))
            if not np.isfinite(cost[ch, best_c]):
                assignment[ch] = -1
            else:
                assignment[ch] = best_c
        n_unassigned = sum(1 for a in assignment if a < 0)
        if n_unassigned > 0 and n_centroids > 0:
            log.warning(
                "_assign_children_to_centroids: %d/%d children unassigned after uncapped pass "
                "(all-inf cost row(s) in cost matrix) — "
                "these children will be dropped from the merge group unless the caller "
                "promotes them as orphan centroids.",
                n_unassigned, n_children,
            )
        return assignment

    # Capped path (max_group_cap > 0): single-pass greedy, centroid order.
    # Note on group-cap semantics (spec §5 Step 3):
    #   max_group_cap counts non-centroids only (not the centroid itself), matching
    #   our spec §5 Step 3 ("absorb up to max_merge_group_size unassigned non-centroids").
    #   The REAM reference's group_size counts total members including the centroid,
    #   so our max_group_cap=8 is equivalent to reference group_size=9.
    # The feasibility check (b_fail) in the bump loop uses the same semantics:
    #   n_ream_nc > n_ream_c * max_group_cap  (non-centroids exceed total centroid capacity).
    assignment = [-1] * n_children
    assigned: set[int] = set()

    for c_idx in range(n_centroids):
        absorbed = 0
        # O(n_children) scan per fill slot — pathological for large expert counts;
        # consider pre-sorting by cost if this becomes a bottleneck.
        while absorbed < max_group_cap:
            best_child = -1
            best_cost = float("inf")
            for ch in range(n_children):
                if ch in assigned:
                    continue
                if cost[ch, c_idx] < best_cost:
                    best_cost = cost[ch, c_idx]
                    best_child = ch
            if best_child < 0:
                # No unassigned children with finite cost remain for this centroid.
                # Break to next centroid; any remaining unassigned children (all-inf
                # cost rows) will be reported and promoted as orphan centroids by the
                # caller. The caller must ensure costs are finite (via feasibility check)
                # to guarantee all children are assigned.
                break
            assignment[best_child] = c_idx
            assigned.add(best_child)
            absorbed += 1

    n_unassigned = sum(1 for a in assignment if a < 0)
    if n_unassigned > 0 and n_centroids > 0:
        log.warning(
            "_assign_children_to_centroids: %d/%d children unassigned after capped greedy pass "
            "(likely cause: inf cost entries in cost matrix preventing assignment) — "
            "these children will be dropped from the merge group unless the caller "
            "promotes them as orphan centroids.",
            n_unassigned, n_children,
        )

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
    def _safe_norm(M):
        # B-C-L-2: when M is all-zero (or constant), m_max == m_min and we fall
        # through to torch.zeros_like(M). This means a zero-distance pair stays
        # zero (no cost contribution from that component) — the desired behavior
        # for Hungarian assignment where ties resolve arbitrarily.
        m_min = float(M.min())
        m_max = float(M.max())
        if m_max > m_min:
            return (M - m_min) / (m_max - m_min)
        return torch.zeros_like(M)

    C_gate = torch.cdist(ref_gate.cpu(), child_gate.cpu())
    C_up   = torch.cdist(ref_up.cpu(), child_up.cpu())
    if ref_act_mean is not None and child_act_mean is not None:
        # L2-normalize both activation-mean vectors along the neuron dimension
        # before computing L2 distance (spec §5, F2-PERM-ALIGN-NORM).
        # eps=1e-8 guards against zero-norm vectors (all-zero activations);
        # F.normalize returns a zero vector for those, which is the safest
        # fallback (zero-norm input → zero output, no NaN).
        ref_act_n   = torch.nn.functional.normalize(ref_act_mean.cpu().float(),   p=2, dim=0, eps=1e-8)
        child_act_n = torch.nn.functional.normalize(child_act_mean.cpu().float(), p=2, dim=0, eps=1e-8)
        C_act = torch.cdist(
            ref_act_n.unsqueeze(-1),
            child_act_n.unsqueeze(-1),
        )
        # Scale each cost component to [0, 1] before summing so that
        # L2-normalized activation distances (O(1/√d_ffn)) are not
        # negligible relative to gate/up weight distances (O(√d_hidden))
        # — spec §5, PERM-ACT-SCALE.
        # B-C-M-1: spec §5 / D5b defines C = C_act + C_wt where C_wt is the
        # gate+up Frobenius distance treated as a SINGLE component (sum first,
        # then normalize once), not two separately-normalized components.
        C_act = _safe_norm(C_act)
        C_wt = _safe_norm(C_gate + C_up)
        C = _safe_norm(C_act) + C_wt
    else:
        # B-C-M-1: same single-component treatment for the no-activation path.
        C = _safe_norm(C_gate + C_up)
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
            if freq_weighted:
                weights = np.array([max(freq.get(m, 0), 0) for m in members], dtype=np.float64)
                # Guard: if all members have zero calibration frequency (pathological
                # edge case), fall back to equal weights rather than dividing by zero
                # (spec freq_i / Σ freq_j formula requires Σ > 0 — F2-FREQ-WEIGHT-FLOOR).
                if weights.sum() <= 0.0:
                    log.warning(
                        "layer %d centroid %d: all %d merge members have zero calibration "
                        "frequency — falling back to equal weights",
                        li, centroid, len(members),
                    )
                    weights[:] = 1.0
                weights /= weights.sum()
            else:
                # B-C-M-2: spec mandates frequency-weighted merge per REAM Eq. 6;
                # equal-weights is for ablation only and produces non-spec-compliant
                # merges. Warn loudly so this is observable in logs.
                log.warning(
                    "Spec mandates frequency-weighted merge per REAM Eq. 6; "
                    "equal-weights is for ablation only and produces "
                    "non-spec-compliant merges. (layer=%d centroid=%d members=%d)",
                    li, centroid, len(members),
                )
                weights = np.ones(len(members), dtype=np.float64)
                weights /= weights.sum()  # equal weights; no zero-sum risk

            # The centroid serves a dual role: it is the permutation-alignment reference
            # (via ref_gate/ref_up) AND a member of the weighted average (members[0]).
            # This is intentional — all reads from the weight bank precede the single
            # write-back (bank.set at the end), so the read-then-write-once ordering
            # guarantees correctness: the centroid's original weights are consumed before
            # being overwritten with the merged result.
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
    # Guard: not all router implementations expose top_k (e.g., custom routers).
    if hasattr(router, "top_k") and router.top_k > len(kept_ids):
        router.top_k = len(kept_ids)

    mlp = layer_ref.mlp
    if hasattr(mlp, "num_experts"):
        mlp.num_experts = len(kept_ids)


def _save_covariance(cov: InputCovarianceAccumulator, path: Path) -> None:
    """Save the full covariance accumulator state to *path*.

    Caller must ensure no active profiling threads are writing to `cov` during
    this call, or hold `cov._lock` externally.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with cov._lock:
        # Clone tensors inside the lock so the snapshot is a deep copy, not a
        # shallow dict of shared tensor references that could be mutated concurrently.
        cov_snapshot = {k: v.clone() for k, v in cov.covariance.items()}
        tok_snapshot = dict(cov.token_count)
    torch.save({"covariance": cov_snapshot, "tokens": tok_snapshot}, tmp)
    _durable_rename(tmp, path)
    log.info("Saved Stage 2 input covariance to %s", path)


def _remap_covariance_for_layer(
    cov: InputCovarianceAccumulator,
    layer_idx: int,
    kept_ids: list[int],
) -> None:
    # kept_ids contains both REAM centroids and protected experts (the full post-merge
    # kept set), not just REAM centroids.
    id_to_new = {old: new for new, old in enumerate(kept_ids)}
    new_cov: dict = {}
    new_tokens: dict = {}
    n_dropped = 0
    dropped_expert_ids: set[int] = set()
    with cov._lock:
        for key, val in list(cov.covariance.items()):
            li, eidx, name = key
            if li != layer_idx:
                new_cov[key] = val
                new_tokens[key] = cov.token_count.get(key, 0)
                continue
            if eidx not in id_to_new:
                n_dropped += 1
                dropped_expert_ids.add(eidx)
                continue
            new_key = (li, id_to_new[eidx], name)
            new_cov[new_key] = val
            new_tokens[new_key] = cov.token_count.get(key, 0)
        orphan_token_keys = set(cov.token_count.keys()) - set(cov.covariance.keys())
        if orphan_token_keys:
            log.warning(
                "_remap_covariance_for_layer layer %d: %d orphaned token_count keys "
                "not in covariance will be dropped: %s",
                layer_idx, len(orphan_token_keys), orphan_token_keys,
            )
        cov.covariance, cov.token_count = new_cov, new_tokens
    if n_dropped > 0:
        n_dropped_experts = len(dropped_expert_ids)
        log.warning(
            "  layer %d: _remap_covariance_for_layer dropped %d covariance "
            "entries (= %d unique experts × ~2 matrices/expert); "
            "dropping %d experts from covariance; keeping %d experts; unexpected if "
            "n_dropped_experts > (n_keys_before - n_kept).",
            layer_idx, n_dropped, n_dropped_experts,
            n_dropped_experts, len(kept_ids),
        )
