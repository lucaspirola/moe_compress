"""Stage 0 — Super Expert Detection.

Profile ``down_proj`` maximum activations on the 100-sample calibration slice
and flag per-layer outliers. Blacklisted experts are protected from pruning in
Stages 1–2 (GRAPE budget solver already subtracts them; REAM keeps them as
forced centroids).

Artifact: ``stage0_blacklist.json`` mapping ``layer_idx -> [expert_idx, ...]``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .utils.activation_hooks import DownProjMaxAccumulator, hook_down_proj_max, run_calibration
from .utils.calibration import CalibrationSpec, build_super_expert_slice, iter_batches
from .utils.model_io import iter_moe_layers, save_json_artifact

log = logging.getLogger(__name__)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
) -> Path:
    s0 = config["stage0_super_experts"]
    cal = config["calibration"]
    spec = CalibrationSpec(
        num_sequences=cal["num_sequences"],
        sequence_length=cal["sequence_length"],
        seed=cal["seed"],
        domain_mix=cal["domain_mix"],
        c4_dataset=cal["dataset"],
        c4_subset=cal["subset"],
    )
    calib = build_super_expert_slice(
        tokenizer, spec, num_samples=cal["super_expert_num_samples"],
        cache_dir=artifacts_dir / "_calibration_cache",
    )
    # Stage 0 walks batches of 1 sample — the hook is just a running max,
    # so memory doesn't accumulate.
    batches = iter_batches(calib, batch_size=1)

    moe_layers = list(iter_moe_layers(model))
    acc = DownProjMaxAccumulator()
    log.info("Stage 0: profiling down_proj max on %d layers × %d experts each (≈%d samples)",
             len(moe_layers), len(moe_layers[0].experts) if moe_layers else 0, len(batches))

    with hook_down_proj_max(moe_layers, acc):
        run_calibration(model, batches, device=device)

    blacklist = _threshold_per_layer(
        acc.per_expert_max,
        num_layers=len(moe_layers),
        num_experts_per_layer={ref.layer_idx: len(ref.experts) for ref in moe_layers},
        zscore=s0["zscore_threshold"],
        cap_per_layer=s0["max_blacklisted_per_layer"],
    )
    # Global cap (total blacklist ≤ cap_pct × total_routed_experts)
    total_experts = sum(len(ref.experts) for ref in moe_layers)
    global_cap = int(s0["global_blacklist_cap_pct"] * total_experts)
    blacklist = _apply_global_cap(blacklist, acc.per_expert_max, global_cap)

    out = {
        str(layer_idx): sorted(experts)
        for layer_idx, experts in blacklist.items()
        if experts
    }
    path = artifacts_dir / "stage0_blacklist.json"
    save_json_artifact({
        "blacklist": out,
        "per_expert_max": {
            f"{k[0]}_{k[1]}": v for k, v in acc.per_expert_max.items()
        },
        "config": s0,
    }, path)
    log.info("Stage 0 complete — blacklisted %d / %d experts → %s",
             sum(len(v) for v in out.values()), total_experts, path)
    return path


def _threshold_per_layer(
    per_expert_max: dict[tuple[int, int], float],
    *,
    num_layers: int,
    num_experts_per_layer: dict[int, int],
    zscore: float,
    cap_per_layer: int,
) -> dict[int, list[int]]:
    """Z-score per layer → flag experts above ``mean + zscore · std``,
    capped at ``cap_per_layer``."""
    blacklist: dict[int, list[int]] = {}
    for li, n_experts in num_experts_per_layer.items():
        vals = np.array([per_expert_max.get((li, e), 0.0) for e in range(n_experts)])
        mean = vals.mean()
        std = vals.std()
        if std <= 0:
            blacklist[li] = []
            continue
        thresh = mean + zscore * std
        flagged = [int(e) for e in range(n_experts) if vals[e] > thresh]
        # Rank by magnitude, keep top cap_per_layer
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
        for li, experts in blacklist.items()
        for e in experts
    ]
    if len(flat) <= cap:
        return blacklist
    flat.sort(key=lambda x: -x[2])
    kept = flat[:cap]
    out: dict[int, list[int]] = {}
    for li, e, _ in kept:
        out.setdefault(li, []).append(e)
    return out
