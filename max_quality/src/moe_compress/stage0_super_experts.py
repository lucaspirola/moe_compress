"""Stage 0 — Super Expert Detection (fused-experts-aware).

Profile ``max(|down_proj_output|)`` per (layer, expert) over a 100-sample
C4 slice, z-score each layer, emit a blacklist capped per-layer and globally.

Instrumentation: we monkey-patch each MoE layer's ``experts.forward`` via
:func:`instrument_experts` with a single ``down`` callback. The wrapped
forward mirrors the reference dispatch so we do not perturb model output.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .utils.activation_hooks import (
    DownProjMaxAccumulator,
    instrument_experts,
    run_calibration,
)
from .utils.calibration import build_super_expert_slice, iter_batches, spec_from_config
from .utils.model_io import iter_moe_layers, save_json_artifact
from .utils.trackio_log import trackio_log as _trackio_log

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
    spec = spec_from_config(cal)
    calib = build_super_expert_slice(
        tokenizer, spec, num_samples=cal["super_expert_num_samples"],
        cache_dir=artifacts_dir / "_calibration_cache",
    )
    batches = iter_batches(calib, batch_size=1)

    moe_layers = list(iter_moe_layers(model))
    n_per_layer = moe_layers[0].num_routed_experts if moe_layers else 0
    log.info(
        "Stage 0: profiling down_proj max on %d layers × %d experts each (≈%d samples)",
        len(moe_layers), n_per_layer, len(batches),
    )

    acc = DownProjMaxAccumulator()

    def down_cb(li, e, tensor, ctx):
        acc.update(li, e, tensor)

    # Instrument every MoE layer simultaneously so one forward pass collects
    # per-expert maxes across all of them.
    import contextlib as _ctx
    with _ctx.ExitStack() as stack:
        for ref in moe_layers:
            stack.enter_context(instrument_experts(ref, {"down": down_cb}))
        run_calibration(model, batches, device=device)
    acc.finalize()

    per_experts_by_layer = {ref.layer_idx: ref.num_routed_experts for ref in moe_layers}
    blacklist = _threshold_per_layer(
        acc.per_expert_max,
        num_experts_per_layer=per_experts_by_layer,
        zscore=s0["zscore_threshold"],
        cap_per_layer=s0["max_blacklisted_per_layer"],
    )
    total_experts = sum(per_experts_by_layer.values())
    global_cap = int(s0["global_blacklist_cap_pct"] * total_experts)
    blacklist = _apply_global_cap(blacklist, acc.per_expert_max, global_cap)

    out = {str(li): sorted(es) for li, es in blacklist.items() if es}
    path = artifacts_dir / "stage0_blacklist.json"
    save_json_artifact(
        {
            "blacklist": out,
            "per_expert_max": {f"{k[0]}_{k[1]}": v for k, v in acc.per_expert_max.items()},
            "config": s0,
        },
        path,
    )
    log.info(
        "Stage 0 complete — blacklisted %d / %d experts → %s",
        sum(len(v) for v in out.values()), total_experts, path,
    )
    # Trackio: per-layer distribution stats of the down-proj max activation
    # (the signal Stage 0 thresholds on). High max + high std = a real super
    # expert; low std + high cap = the threshold is doing nothing.
    import statistics as _stats
    for ref in moe_layers:
        vals = [
            acc.per_expert_max.get((ref.layer_idx, e), 0.0)
            for e in range(ref.num_routed_experts)
        ]
        if not vals:
            continue
        _trackio_log({
            "stage0/layer_idx": ref.layer_idx,
            "stage0/down_max_mean": float(_stats.fmean(vals)),
            "stage0/down_max_std": float(_stats.pstdev(vals)) if len(vals) > 1 else 0.0,
            "stage0/down_max_max": float(max(vals)),
            "stage0/blacklisted": float(len(blacklist.get(ref.layer_idx, []))),
        })
    return path


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
