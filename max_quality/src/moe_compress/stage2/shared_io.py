"""Stage 2 shared IO helpers: durable writes, partial-dir snapshots, heal-weights checkpoint.

Extracted from ``stage2_reap_ream.py`` in Task 2 of the plugin-architecture
refactor. Public surface is unchanged: ``stage2_reap_ream`` re-imports every
symbol below at module scope so external call-sites (tests, sibling modules)
keep working without modification.

The partial-JSON schema written by ``_write_merge_json`` is FROZEN at
``format_version=2`` — round-tripped by ``test_pipeline_shared_io.py`` so any
accidental drift fails CI.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from ..utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from ..utils.model_io import MATRIX_NAMES, MoELayerRef, build_banks

log = logging.getLogger(__name__)

# Bump when the on-disk heal-weights payload changes shape; loader validates.
_HEAL_WEIGHTS_FORMAT_VERSION = 2


def _durable_rename(tmp: Path, final: Path) -> None:
    """Fsync *tmp*, atomically rename it to *final*, then fsync the parent dir.

    Spec §11: durable write — fsync file bytes, then fsync parent dir entry,
    then atomic rename so a crash never leaves a truncated final file.

    Backward-compat shim (audit/calibration-durability): delegates to the
    shared :func:`utils.atomic_io.durable_rename`. Existing call sites
    that ``from .shared_io import _durable_rename`` (or the in-module
    callers below) continue to work unchanged. The shared helper uses
    ``O_RDONLY`` for the file fsync rather than ``O_WRONLY|O_APPEND``;
    POSIX requires fsync() to flush all buffered modifications on the
    fd regardless of open mode, and O_RDONLY survives on FUSE mounts
    (HF Jobs bucket) that reject opening a regular file O_WRONLY.

    Note: ``tmp`` must already be closed at the Python level (all
    userspace I/O buffers flushed to the kernel) before this call; the
    fsync here flushes kernel page-cache, not Python buffers.
    """
    from ..utils.atomic_io import durable_rename as _shared_durable_rename
    _shared_durable_rename(tmp, final)


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


def _snapshot_neuron_means_layer(
    ream_acc: ReamCostAccumulator,
    layer_idx: int,
    partial_dir: Path,
) -> None:
    """Persist per-expert mean activation vectors for resume-time C_act.

    B-iter5-M-2: spec D5b mandates `C = C_wt + C_act` for permutation alignment.
    Without this artifact, resume falls back to weight-only alignment and merged
    weights diverge from a fresh run. This helper snapshots only the small
    per-expert mean vectors (`[d_intermediate]` per expert), not the full
    intermediate-activation history (which is large and not needed downstream).

    Format version 1: `{"format_version": 1, "neuron_means": {expert_idx: tensor}}`.
    Missing-on-resume → loud ERROR + weight-only fallback (preserves run completion).
    """
    with ream_acc._lock:
        keys = [k for k in ream_acc._neuron_act_sum if k[0] == layer_idx]
        if not keys:
            log.debug("_snapshot_neuron_means_layer: no neuron-mean entries for layer %d; "
                      "skipping snapshot (no merges in this layer)", layer_idx)
            return
        means: dict[int, torch.Tensor] = {}
        for k in keys:
            s = ream_acc._neuron_act_sum[k]
            c = ream_acc._neuron_act_count.get(k, 0)
            if c == 0:
                continue
            means[k[1]] = (s.clone() / c).contiguous()
    if not means:
        return
    payload = {"format_version": 1, "neuron_means": means}
    tmp = partial_dir / f"_neuron_means_layer{layer_idx}.pt.tmp"
    final = partial_dir / f"_neuron_means_layer{layer_idx}.pt"
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
    assignment_solver_used: str = "greedy",
    cost_alignment_used: str = "pre",
    em_rounds_completed: int = 0,
    distill_state: dict | None = None,
    heal_state: dict | None = None,
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
        heal_state:       Per-layer merge-heal outcome dict (None when the
                          opt-in merge-heal feature is disabled).
    """
    payload = {
        "format_version": 2,
        "final_kept_ids": final_kept_ids,
        # list(v) ensures JSON gets a plain list, not a subclass that might not serialize
        "grouped": {str(k): list(v) for k, v in grouped.items()},
        "freq": {str(k): int(v) for k, v in freq.items()},
        # list(v) ensures JSON gets a plain list, not a subclass that might not serialize
        "merge_map_layer": {str(k): list(v) for k, v in merge_map_layer.items()},
        "mean_cost_per_pair": mean_cost_per_pair,
        # Stage 2 v2 (spec § 12.1): forensic / resume fields. ``em_rounds_completed``
        # and ``distill_state`` are reserved for Phases 2 and 3; included here
        # so Phase-1-completed partials are forward-compatible with later phases.
        "assignment_solver_used": assignment_solver_used,
        "cost_alignment_used": cost_alignment_used,
        "em_rounds_completed": em_rounds_completed,
        "distill_state": distill_state,
        # Stage-2 per-layer merge-heal (opt-in). None when healing is off;
        # otherwise the small JSON-able state dict returned by _heal_layer
        # (steps / train_mse / train_mse_at_best / holdout_mse / heal_gap /
        # stop_reason). The healed weights themselves live in
        # _heal_weights_layer_{N}.pt.
        "heal_state": heal_state,
    }
    tmp = partial_dir / f"merge_{layer_idx}.json.tmp"
    final = partial_dir / f"merge_{layer_idx}.json"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    _durable_rename(tmp, final)


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
    torch.save({"format_version": 1, "covariance": cov_snapshot, "tokens": tok_snapshot}, tmp)
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


def _write_heal_weights(
    partial_dir: Path,
    layer_ref: MoELayerRef,
    final_kept_ids: list[int],
    *,
    accepted: bool,
) -> None:
    """Persist the post-heal expert tensors + router weight/bias for one layer.

    Post-heal weights are NOT reconstructible from ``merge_N.json`` (the heal
    fine-tunes them), so a per-layer ``_heal_weights_layer_{N}.pt`` is written
    into ``partial_dir`` — atomically, and BEFORE ``_write_merge_json`` so the
    ``.pt``-before-``.json`` resume invariant holds.

    EVERY kept expert is stored (all kept experts are trained by the heal);
    when the layer's heal was rejected the banks already hold the plain-merged
    weights, so the file faithfully captures whatever final state the layer is
    in. ``accepted`` is recorded as telemetry.
    """
    layer_idx = layer_ref.layer_idx
    banks = build_banks(layer_ref)
    router = layer_ref.router

    # Keyed by post-select POSITION (banks were re-indexed to 0..n_kept-1 by
    # bank.select()); _load_heal_weights replays by the same position. All
    # kept experts are stored.
    healed: dict[str, dict[str, torch.Tensor]] = {}
    for pos, _cid in enumerate(final_kept_ids):
        healed[str(pos)] = {
            name: banks[name].get(pos).detach().cpu().clone()
            for name in MATRIX_NAMES
        }
    payload = {
        "format_version": _HEAL_WEIGHTS_FORMAT_VERSION,
        "layer_idx": layer_idx,
        "accepted": bool(accepted),
        "healed_experts": healed,
        "router_weight": router.weight.detach().cpu().clone(),
        "router_bias": (
            router.bias.detach().cpu().clone()
            if getattr(router, "bias", None) is not None else None
        ),
    }
    tmp = partial_dir / f"_heal_weights_layer_{layer_idx}.pt.tmp"
    final = partial_dir / f"_heal_weights_layer_{layer_idx}.pt"
    torch.save(payload, tmp)
    _durable_rename(tmp, final)


def _load_heal_weights(
    partial_dir: Path,
    layer_ref: MoELayerRef,
    final_kept_ids: list[int],
) -> None:
    """Apply a persisted ``_heal_weights_layer_{N}.pt`` to the banks + router.

    Used by the resume path: a layer that completed its heal in a prior run
    has a healed-weights file; reload it so the in-memory model matches the
    state the heal left. A missing file is fatal — the operator must delete
    ``_stage2_partial/`` and re-run.

    For an ACCEPTED layer the persisted weights are genuinely post-heal and
    are NOT reconstructible from ``merge_*.json``. For a REJECTED layer the
    banks were reverted to the plain merge, so the file simply re-persists
    that plain-merged state. The file is written for every *merged* layer
    regardless of the accept/reject outcome, and reloaded here on resume
    whenever it is present; a 0-merge layer has no heal-weights file (the
    heal is skipped), so the caller gates this load on the file existing.
    """
    layer_idx = layer_ref.layer_idx
    path = partial_dir / f"_heal_weights_layer_{layer_idx}.pt"
    if not path.exists():
        raise RuntimeError(
            f"Stage-2 merge-heal resume: layer {layer_idx} completed its merge "
            f"but {path.name} is missing. Healed weights are not "
            "reconstructible from merge_*.json — delete _stage2_partial/ and "
            "re-run Stage 2."
        )
    # weights_only=True is safe: the payload is only tensors + ints + str + None.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fv = int(payload.get("format_version", 0))
    if fv != _HEAL_WEIGHTS_FORMAT_VERSION:
        raise RuntimeError(
            f"{path} has format_version={fv} "
            f"(expected {_HEAL_WEIGHTS_FORMAT_VERSION}) — delete "
            "_stage2_partial/ and re-run Stage 2."
        )
    banks = build_banks(layer_ref)
    bank_dtype = banks["gate_proj"].get(0).dtype
    n_kept = len(final_kept_ids)
    with torch.no_grad():
        # healed_experts is keyed by post-select position (see _write_heal_weights):
        # banks were re-indexed to 0..n_kept-1 by bank.select().
        for pos_str, mats in payload["healed_experts"].items():
            pos = int(pos_str)
            if not (0 <= pos < n_kept):
                raise RuntimeError(
                    f"{path}: healed-expert position {pos} out of range "
                    f"[0, {n_kept}) — heal-weights file inconsistent with "
                    "merge_*.json."
                )
            for name in MATRIX_NAMES:
                banks[name].set(pos, mats[name].to(bank_dtype))
        router = layer_ref.router
        router.weight = nn.Parameter(
            payload["router_weight"].to(
                device=router.weight.device, dtype=router.weight.dtype,
            ),
            requires_grad=router.weight.requires_grad,
        )
        if payload.get("router_bias") is not None and getattr(router, "bias", None) is not None:
            router.bias = nn.Parameter(
                payload["router_bias"].to(
                    device=router.bias.device, dtype=router.bias.dtype,
                ),
                requires_grad=router.bias.requires_grad,
            )
