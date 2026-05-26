"""Stage 2 merge engine + router resize.

Extracted from ``stage2_reap_ream.py`` in Task 4 of the plugin-architecture
refactor. The two operations live together because both mutate the layer in
place at the end of grouping:

  * ``_merge_experts_inplace`` -- REAM Eq. 6 merge with per-pair Hungarian alignment.
    Supports two weight modes: frequency-weighted (``freq_weighted=True``, default;
    REAM paper Eq. 6) and saliency-weighted (``freq_weighted=False``; Cerebras REAP
    default, ``merger.py:563``). See function docstring for the algebraic equivalence.
  * ``_resize_router_for_kept_experts`` -- slice the router's weight rows
    (and bias, if present) down to the centroid set; update ``num_experts``
    and clamp ``top_k``.

``stage2_reap_ream`` re-imports both names at module scope so existing call
sites and tests keep working unchanged.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from ..utils.activation_hooks import ReamCostAccumulator
from ..utils.model_io import MoELayerRef, build_banks
from .permutation_align import _PermAlignCache, _permutation_align_to_centroid

log = logging.getLogger(__name__)


def _merge_experts_inplace(
    layer_ref: MoELayerRef,
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    *,
    freq_weighted: bool,
    scores: np.ndarray | None = None,
    ream_acc: ReamCostAccumulator | None = None,
    perm_cache: "_PermAlignCache | None" = None,
) -> None:
    """Merge non-centroid experts into their centroid in place.

    Parameters
    ----------
    freq_weighted:
        ``True`` (default, REAM Eq. 6): merge weights are
        ``freq_m / Σ_j freq_j`` — raw calibration token counts, renormalized
        over the merge group. Equivalent to
        ``S^freq_m / Σ_j S^freq_j`` where ``S^freq = freq/|X|``, because
        ``|X|`` (token count) cancels in the ratio.

        ``False`` (Cerebras REAP default, ``merger.py:563`` @
        ``CerebrasResearch/reap@1970473c``): merge weights are
        ``S_m / Σ_j S_j`` where ``S_j`` is the REAP Eq. 9 per-expert average
        saliency ``(1/|X_j|)·Σ g_j·‖f_j‖₂``. ``scores`` must be provided.

        Algebraic equivalence: both modes compute ``weight_m / Σ weight_j``
        where the weighting quantity is already a per-expert average (freq/|X|
        for freq mode; S_j for saliency mode). No additional normalization
        by token count is needed in either case.

    scores:
        1-D ``np.ndarray`` indexed by expert id. Required when
        ``freq_weighted=False``; ignored (may be ``None``) when
        ``freq_weighted=True``.
    """
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
                # Saliency-weighted merge — Cerebras REAP default (merger.py:563,
                # CerebrasResearch/reap @ 1970473c51ca3caeb98c10392f15b3a08a672974).
                # weights = S_m / Σ S_j, where S_j is the REAP Eq. 9 per-expert
                # average saliency (already a per-expert AVERAGE over dispatched
                # tokens, so no |X| normalization is needed here — the Σ
                # denominator handles it).
                if scores is None:
                    raise ValueError(
                        f"Stage 2: ream.frequency_weighted_merge=False requires "
                        f"a saliency scores array (scores=None at layer={li} "
                        f"centroid={centroid}). Ensure ReapScoringPlugin ran "
                        f"before LayerMergePlugin and that ctx.get('scores') is "
                        f"non-None. NB: scores are not persisted on disk, so a "
                        f"saliency-mode run cannot be resumed from a partial "
                        f"stage-2 checkpoint."
                    )
                weights = np.array(
                    [max(float(scores[m]), 0.0) for m in members],
                    dtype=np.float64,
                )
                if weights.sum() <= 0.0:
                    log.warning(
                        "layer %d centroid %d: all %d merge members have zero "
                        "saliency score — falling back to equal weights",
                        li, centroid, len(members),
                    )
                    weights[:] = 1.0
                weights /= weights.sum()

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
                if m == centroid:
                    perm = None
                else:
                    # Stage 2 v2 (M1): reuse the perm computed during cost-matrix
                    # construction if the cache hit. This avoids a second
                    # Hungarian solve per merge member.
                    cached = (
                        perm_cache.get((li, centroid, m))
                        if perm_cache is not None
                        else None
                    )
                    if cached is not None:
                        perm = cached[0]
                    else:
                        perm = _permutation_align_to_centroid(
                            ref_gate, ref_up, gate_m, up_m,
                            ref_act_mean=ref_act, child_act_mean=child_act,
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
