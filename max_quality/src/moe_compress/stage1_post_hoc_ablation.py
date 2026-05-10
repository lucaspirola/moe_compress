"""Phase F — post-hoc causal-ablation validation of the Stage 1 SE blacklist.

After Phase A/B/C/C+ produces the final blacklist, this module measures the
PPL impact of ablating (zeroing the down_proj output of) each blacklisted
expert AND the top-K non-blacklisted experts in each MA-formation layer. The
report is written to ``stage1_post_hoc_ablation.json`` and is REPORT-ONLY — no
automatic blacklist mutation.

Reviewer (max_quality/docs/stage1_review_10.05.2026.txt §Q4): "Ablate each
blacklisted expert AND the top-5 non-blacklisted experts (by per_expert_max)
in L34/L38 to confirm no critical experts were missed."

Cost (Qwen3.6-35B-A3B, H200): ~(|blacklist| + |L|·top_k) forward passes over
a small held-out slice. With |L|=3, |blacklist|≈10, top_k=5 → ~25 forwards
≈ 30 minutes.

Ablation semantics
------------------
The ``down`` callback in :func:`instrument_experts` receives the down_proj
output by reference *before* it is multiplied by routing weights and
``index_add_``-ed into the layer's output. Calling ``tensor.zero_()`` on the
callback argument therefore zeroes the tensor that the next two lines of
the wrapped forward consume — see ``activation_hooks.py`` lines 1099-1103
(factored) and 1132-1135 (fused). The forward runs under
``torch.no_grad()`` so in-place mutation is safe.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from .utils.activation_hooks import instrument_experts
from .utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from .utils.model_io import iter_moe_layers, save_json_artifact

log = logging.getLogger(__name__)


def rank_top_nonblacklisted(
    per_expert_max: dict[tuple[int, int], float],
    blacklist: dict[int, list[int]],
    L: set[int],
    top_k: int,
) -> dict[int, list[int]]:
    """For each l ∈ L, return top-`top_k` non-blacklisted expert ids by per_expert_max desc."""
    blacklisted = {(li, e) for li, lst in blacklist.items() for e in lst}
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for (li, e), v in per_expert_max.items():
        if li not in L or (li, e) in blacklisted:
            continue
        by_layer.setdefault(li, []).append((e, v))
    out: dict[int, list[int]] = {}
    for li, lst in by_layer.items():
        lst.sort(key=lambda t: -t[1])
        out[li] = [e for e, _ in lst[:top_k]]
    return out


def _measure_corpus_nll(model, batches, device) -> float:
    """Mean per-token NLL across the held-out slice. Cross-entropy on shifted labels."""
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device) if device is not None else batch
            out = model(input_ids=batch, labels=batch)
            ntok = (batch.shape[0] * (batch.shape[1] - 1))  # shift-by-1
            total_nll += float(out.loss.item()) * ntok
            total_tokens += ntok
    return total_nll / max(total_tokens, 1)


def _ablate_expert_context(layer_ref, expert_idx: int):
    """Context manager that zeros the named expert's down_proj output for its lifetime.

    Relies on ``instrument_experts``'s ``down`` callback receiving the
    down_proj output by reference *before* it is multiplied by routing
    weights — see module docstring for the activation_hooks reference.
    """
    def _zero_cb(li, e, tensor, _ctx):
        if e == expert_idx:
            tensor.zero_()
    return instrument_experts(layer_ref, {"down": _zero_cb})


def run_phase_f(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    blacklist: dict[int, list[int]],
    per_expert_max: dict[tuple[int, int], float],
    L: set[int],
    *,
    device=None,
) -> Path:
    """Run Phase F and write ``stage1_post_hoc_ablation.json``."""
    s1 = config["stage1_grape"]
    cal = config["calibration"]
    pf = s1.get("post_hoc_ablation", {})
    if not bool(pf.get("enabled", True)):
        log.info("Stage 1 Phase F: disabled in config; skipping")
        return artifacts_dir / "stage1_post_hoc_ablation.json"

    holdout_samples = int(pf.get("holdout_samples", 100))
    top_k = int(pf.get("topk_nonblacklisted", 5))

    # Held-out slice: deterministic seed offset distinct from Phase A/B
    spec = spec_from_config(cal, num_sequences_override=holdout_samples, seed_offset=999)
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache_phase_f",
    )
    eval_batches = iter_batches(calib, batch_size=1)

    moe_layers = {ref.layer_idx: ref for ref in iter_moe_layers(model)}

    # Baseline: no ablation
    baseline_nll = _measure_corpus_nll(model, eval_batches, device)
    log.info("Stage 1 Phase F: baseline mean NLL = %.4f", baseline_nll)

    impacts: dict[str, dict] = {"blacklisted": {}, "top_nonblacklisted": {}}

    # Blacklisted ablations
    for li, exps in blacklist.items():
        ref = moe_layers[int(li)]
        for e in exps:
            with _ablate_expert_context(ref, e):
                nll = _measure_corpus_nll(model, eval_batches, device)
            impacts["blacklisted"][f"L{li}E{e}"] = nll - baseline_nll

    # Top-K non-blacklisted candidates per l ∈ L
    candidates = rank_top_nonblacklisted(per_expert_max, blacklist, L, top_k=top_k)
    for li, exps in candidates.items():
        ref = moe_layers[li]
        for e in exps:
            with _ablate_expert_context(ref, e):
                nll = _measure_corpus_nll(model, eval_batches, device)
            impacts["top_nonblacklisted"][f"L{li}E{e}"] = nll - baseline_nll

    out_path = artifacts_dir / "stage1_post_hoc_ablation.json"
    save_json_artifact(
        {
            "baseline_mean_nll": baseline_nll,
            "delta_nll": impacts,
            "config": {
                "holdout_samples": holdout_samples,
                "topk_nonblacklisted": top_k,
                "ma_formation_layers": sorted(L),
            },
        },
        out_path,
    )
    log.info("Stage 1 Phase F: wrote %d ablation results to %s",
             len(impacts["blacklisted"]) + len(impacts["top_nonblacklisted"]),
             out_path)
    return out_path
