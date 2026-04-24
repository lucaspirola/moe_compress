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

The REAM Hungarian + frequency-weighted-merge logic is unchanged
algorithmically.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from .utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReapAccumulator,
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

log = logging.getLogger(__name__)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
    stage1_budget_path: Path | None = None,
) -> Path:
    s2 = config["stage2_reap_ream"]
    cal = config["calibration"]

    if stage1_budget_path is None:
        stage1_budget_path = artifacts_dir / "stage1_budgets.json"
    budgets_payload = load_json_artifact(stage1_budget_path)
    per_layer_target = {
        int(k): int(v) for k, v in budgets_payload["per_layer_target_experts"].items()
    }
    blacklist_payload = load_json_artifact(artifacts_dir / "stage0_blacklist.json")
    blacklist = {int(k): list(v) for k, v in blacklist_payload.get("blacklist", {}).items()}

    spec = spec_from_config(cal, num_sequences_override=s2["num_calibration_samples"])
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    batches = iter_batches(calib, batch_size=s2["batch_size"])

    moe_layers = list(iter_moe_layers(model))
    cov_acc = InputCovarianceAccumulator()
    # Addresses review P0-3 (cov storage OOM): default to bf16 on-disk cov.
    cov_dtype = getattr(torch, s2.get("covariance_storage_dtype", "bfloat16"))
    cov_acc.set_storage_dtype(cov_dtype)
    merge_map: dict[int, dict[int, list[int]]] = {}

    for k, layer_ref in enumerate(moe_layers):
        target = per_layer_target[layer_ref.layer_idx]
        log.info(
            "Stage 2 layer %d/%d (idx=%d) — profiling then merging to %d experts",
            k + 1, len(moe_layers), layer_ref.layer_idx, target,
        )
        reap_acc = ReapAccumulator()
        _profile_layer(
            model, layer_ref, batches, reap_acc, cov_acc,
            device=device,
        )

        n_experts = layer_ref.num_routed_experts
        protected = set(blacklist.get(layer_ref.layer_idx, []))
        scores = np.array([reap_acc.score(layer_ref.layer_idx, e) for e in range(n_experts)])

        centroid_ids = sorted(protected)
        for e in np.argsort(-scores):
            e = int(e)
            if e in protected:
                continue
            centroid_ids.append(e)
            if len(centroid_ids) >= target:
                break
        centroid_ids = sorted(centroid_ids)
        noncentroid_ids = [e for e in range(n_experts) if e not in set(centroid_ids)]

        delta = _ream_cost_matrix(
            layer_ref, noncentroid_ids, centroid_ids,
            gate_weight=s2["ream"]["gate_weight"],
            expert_weight=s2["ream"]["expert_weight"],
        )
        assignment = _assign_children_to_centroids(
            delta, len(noncentroid_ids), len(centroid_ids),
        )
        freq = {e: reap_acc.freq.get((layer_ref.layer_idx, e), 0) for e in range(n_experts)}
        grouped: dict[int, list[int]] = {c: [c] for c in centroid_ids}
        for child_pos, centroid_pos in enumerate(assignment):
            grouped[centroid_ids[centroid_pos]].append(noncentroid_ids[child_pos])

        _merge_experts_inplace(
            layer_ref, grouped, freq,
            freq_weighted=s2["ream"]["frequency_weighted_merge"],
        )

        banks = build_banks(layer_ref)
        for bank in banks.values():
            bank.select(centroid_ids)
        _resize_router_for_kept_experts(layer_ref, centroid_ids)

        merge_map[layer_ref.layer_idx] = {
            new_idx: sorted(grouped[centroid])
            for new_idx, centroid in enumerate(centroid_ids)
        }
        _remap_covariance_for_layer(cov_acc, layer_ref.layer_idx, centroid_ids)

        log.info(
            "  kept %d / %d experts (blacklist=%d) — Σ cost=%.4f",
            len(centroid_ids), n_experts, len(protected),
            float(delta.sum()) if delta.size else 0.0,
        )

    out_dir = artifacts_dir / "stage2_pruned"
    save_compressed_checkpoint(
        model, tokenizer, out_dir,
        pipeline_stage="stage2_pruned",
        extra_metadata={"merge_map_file": "merge_map.json"},
    )
    save_json_artifact(merge_map, out_dir / "merge_map.json")
    _save_covariance(cov_acc, artifacts_dir / "_stage2_input_covariance.pt")
    log.info("Stage 2 complete — pruned checkpoint at %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Per-layer profiling: REAP + input covariance via instrument_experts
# ---------------------------------------------------------------------------


def _profile_layer(
    model,
    layer_ref: MoELayerRef,
    batches,
    reap_acc: ReapAccumulator,
    cov_acc: InputCovarianceAccumulator,
    *,
    device=None,
) -> None:
    layer_idx = layer_ref.layer_idx

    def input_cb(li, e, tensor, ctx):
        # Input to gate_proj + up_proj share the same tensor; the accumulator
        # aliases them so a single gate_proj update covers both.
        cov_acc.update(li, e, "gate_proj", tensor)

    def intermediate_cb(li, e, tensor, ctx):
        # Input to down_proj.
        cov_acc.update(li, e, "down_proj", tensor)

    def down_cb(li, e, tensor, ctx):
        # REAP contribution per expert dispatch event.
        record_reap(reap_acc, li, e, ctx["top_k_weights"], tensor)

    with instrument_experts(
        layer_ref,
        {"input": input_cb, "intermediate": intermediate_cb, "down": down_cb},
    ):
        run_calibration(model, batches, device=device)


# ---------------------------------------------------------------------------
# REAM cost + Hungarian assignment
# ---------------------------------------------------------------------------


def _ream_cost_matrix(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    gate_weight: float,
    expert_weight: float,
) -> np.ndarray:
    if not noncentroid_ids or not centroid_ids:
        return np.zeros((len(noncentroid_ids), len(centroid_ids)))

    router = layer_ref.router
    gate_w = router.weight.detach().to(torch.float32)
    gate_w_n = torch.nn.functional.normalize(gate_w, dim=1)
    gate_sim = gate_w_n @ gate_w_n.transpose(0, 1)
    delta_gate_full = (1.0 - gate_sim).clamp(min=0.0, max=2.0) / 2.0

    # Expert similarity via flattened combined weights (gate, up, down).
    banks = build_banks(layer_ref)
    all_ids = noncentroid_ids + centroid_ids
    flat_vecs: list[torch.Tensor] = []
    for e in all_ids:
        parts = [banks[n].get(e).detach().to(torch.float32).flatten() for n in MATRIX_NAMES]
        flat_vecs.append(torch.cat(parts))
    W = torch.stack(flat_vecs)
    Wn = torch.nn.functional.normalize(W, dim=1)
    sim = Wn @ Wn.transpose(0, 1)
    expert_sim = sim[: len(noncentroid_ids), len(noncentroid_ids):]
    delta_expert = (1.0 - expert_sim).clamp(min=0.0, max=2.0) / 2.0

    dg = delta_gate_full[np.ix_(noncentroid_ids, centroid_ids)]
    cost = gate_weight * dg + expert_weight * delta_expert
    return cost.cpu().numpy()


def _assign_children_to_centroids(
    cost: np.ndarray, n_children: int, n_centroids: int,
) -> list[int]:
    if n_children == 0 or n_centroids == 0:
        return []
    assignment = [-1] * n_children
    remaining = list(range(n_children))
    while remaining:
        batch = remaining[:n_centroids]
        sub = cost[np.ix_(batch, range(n_centroids))]
        row_ind, col_ind = linear_sum_assignment(sub)
        for r, c in zip(row_ind, col_ind):
            assignment[batch[r]] = int(c)
        remaining = remaining[n_centroids:]
    return assignment


# ---------------------------------------------------------------------------
# Frequency-weighted merge (bank-aware)
# ---------------------------------------------------------------------------


def _merge_experts_inplace(
    layer_ref: MoELayerRef,
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    *,
    freq_weighted: bool,
) -> None:
    """Write the frequency-weighted average of each group into the centroid
    slot of the stacked tensors. ``select`` afterwards drops non-centroid rows.
    """
    banks = build_banks(layer_ref)
    with torch.no_grad():
        for centroid, members in grouped.items():
            if len(members) <= 1:
                continue
            weights = np.array([max(freq.get(m, 0), 1) for m in members], dtype=np.float64)
            if not freq_weighted:
                weights[:] = 1.0
            weights = weights / weights.sum()
            for name, bank in banks.items():
                acc = None
                for w, m in zip(weights, members):
                    Wm = bank.get(m).to(torch.float32)
                    acc = Wm * float(w) if acc is None else acc + Wm * float(w)
                bank.set(centroid, acc)


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


# ---------------------------------------------------------------------------
# Covariance I/O + post-merge remap (preserves Round-1 fix)
# ---------------------------------------------------------------------------


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
    cov.covariance = new_cov
    cov.token_count = new_tokens
