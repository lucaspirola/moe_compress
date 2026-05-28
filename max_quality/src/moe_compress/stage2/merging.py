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

from ..utils.activation_hooks import InputCovarianceAccumulator, ReamCostAccumulator
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
    merge_step: str = "freq_weighted",
    layer_inputs: torch.Tensor | None = None,
    token_cap: int = 1024,
    cov_acc: "InputCovarianceAccumulator | None" = None,
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

    merge_step:
        ``"freq_weighted"`` (default, byte-identical to legacy behaviour) —
        merged ``gate_proj`` / ``up_proj`` / ``down_proj`` are all
        ``Σ_j b_j · perm_j(W_j)`` (REAM Eq. 6 with the cluster's freq or
        saliency weights ``b_j``).

        ``"mergemoe"`` — paper-original MergeMoE closed-form merge
        (arXiv:2510.14436 Eqs. 3–6 / Theorem 1). Gate and up remain
        ``Σ_j b_j · perm_j(W_j)`` (algebraically identical to the
        freq-weighted path because T₂=T₃ are the freq-weighting block
        matrices; paper Eq. 4). The down-projection is replaced by the
        closed-form least-squares solution
        ``W_D^merged = Σ_j b_j · perm_j(W_D^j) · T₁_block_j`` with
        ``T₁ = Q·P†`` solved per-cluster from the layer-input calibration
        tokens. See :mod:`stage2.mergemoe` for the math and deviations.
        Requires ``layer_inputs`` to be non-None and non-empty; on
        ``layer_inputs is None`` falls back to ``"freq_weighted"`` with a
        WARNING log line (defensive — the caller should populate this
        when configuring ``merge_step="mergemoe"``).

        ``"regmean"`` — RegMean closed-form per-Linear least-squares
        merge (Jin et al., ICLR 2023, arXiv:2212.09849, Eq. 2). All
        three projections (``gate_proj`` / ``up_proj`` / ``down_proj``)
        are replaced by the closed-form
        ``W_M = (Σ G_i)⁻¹ · Σ G_i W_i`` with ``G_i = X_i^T·X_i`` the
        per-source input Gram. Requires ``cov_acc`` to be non-None and
        populated for every (layer, expert, matrix) key in the cluster;
        on missing keys, the cluster falls back to ``"freq_weighted"``
        with a WARNING log line. See :mod:`stage2.regmean` for the math
        and deviations.

    layer_inputs:
        ``(T_full, d_hidden)`` calibration tokens captured by
        :class:`stage2.profiling._LayerInputAccumulator`. Required for
        ``merge_step="mergemoe"``; ignored for ``"freq_weighted"`` and
        ``"regmean"``.

    token_cap:
        Cap on the per-layer calibration-token subsample used by the
        MergeMoE solve. Ignored for ``"freq_weighted"`` and ``"regmean"``.
        Default 1024 matches the SC output-cost token budget (see
        ``cost_output_token_cap``).

    cov_acc:
        :class:`InputCovarianceAccumulator` populated during the Stage 2
        profile pass (per-(layer, expert, matrix) input Gram
        ``X^T·X``). Required for ``merge_step="regmean"``; ignored for
        ``"freq_weighted"`` and ``"mergemoe"``. On ``cov_acc is None``
        with ``merge_step="regmean"`` falls back to ``"freq_weighted"``
        with a WARNING (defensive — the caller should populate it).
    """
    if merge_step not in ("freq_weighted", "mergemoe", "regmean"):
        raise ValueError(
            f"_merge_experts_inplace: merge_step={merge_step!r}; "
            "expected 'freq_weighted', 'mergemoe', or 'regmean'."
        )
    # MergeMoE requires the layer-input calibration buffer. If the caller
    # forgot to enable it, fall back loudly to the freq-weighted path so the
    # merge still completes — but log a WARNING so the misconfiguration is
    # visible. Per-cluster fallback also fires inside
    # ``_mergemoe_compute_merged_down`` on cond(P) > 1e8.
    effective_merge_step = merge_step
    if merge_step == "mergemoe" and (layer_inputs is None or layer_inputs.numel() == 0):
        log.warning(
            "layer %d: merge_step='mergemoe' but layer_inputs is empty/None — "
            "falling back to freq-weighted merge for this layer.",
            layer_ref.layer_idx,
        )
        effective_merge_step = "freq_weighted"
    # RegMean requires the per-(layer, expert, matrix) Gram via cov_acc.
    # Defensive whole-layer fallback when cov_acc is missing entirely
    # (per-cluster, per-member None-Gram fallback fires per cluster below;
    # see D-regmean-zero-cov-fallback in :mod:`stage2.regmean`).
    if merge_step == "regmean" and cov_acc is None:
        log.warning(
            "layer %d: merge_step='regmean' but cov_acc is None — "
            "falling back to freq-weighted merge for this layer. "
            "Ensure InputCovarianceAccumulator is wired through "
            "LayerMergePlugin.merge (the Stage 2 profile pass populates it).",
            layer_ref.layer_idx,
        )
        effective_merge_step = "freq_weighted"

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
            # MergeMoE bookkeeping (only populated when
            # ``effective_merge_step == "mergemoe"``): permutation-aligned
            # per-member weight tensors. Re-used after the loop to call
            # ``_mergemoe_compute_merged_down``. Stays empty for the default
            # freq-weighted path so the legacy code is byte-identical.
            mm_gates: list[torch.Tensor] = []
            mm_ups:   list[torch.Tensor] = []
            mm_downs: list[torch.Tensor] = []
            # RegMean bookkeeping (only populated when
            # ``effective_merge_step == "regmean"``): permutation-aligned
            # per-member weight tensors AND per-member input Gram matrices.
            # gate_proj and up_proj share the same input (the layer hidden
            # state) so they share the same Gram from
            # ``cov_acc.get((li, m, "gate_proj"))``. down_proj has its own
            # Gram (the SwiGLU intermediate input), which must be permuted
            # along BOTH axes to align with the permuted down_proj weight.
            # Stays empty for the default freq-weighted path.
            rm_gates: list[torch.Tensor] = []
            rm_ups:   list[torch.Tensor] = []
            rm_downs: list[torch.Tensor] = []
            rm_G_hidden:  list[torch.Tensor | None] = []  # shared by gate/up
            rm_G_down:    list[torch.Tensor | None] = []
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
                    # MergeMoE: stash the aligned per-member tensors for the
                    # per-cluster lstsq solve below. We branch only on the
                    # short-circuit boolean — the freq-weighted accumulation
                    # above is unaffected.
                    if effective_merge_step == "mergemoe":
                        if name == "gate_proj":
                            mm_gates.append(Wm)
                        elif name == "up_proj":
                            mm_ups.append(Wm)
                        else:
                            mm_downs.append(Wm)
                    # RegMean: stash the aligned per-member tensors. The
                    # per-member Grams are stashed after the inner loop
                    # (one Gram per member, not one per matrix).
                    if effective_merge_step == "regmean":
                        if name == "gate_proj":
                            rm_gates.append(Wm)
                        elif name == "up_proj":
                            rm_ups.append(Wm)
                        else:
                            rm_downs.append(Wm)

                # RegMean per-member Gram capture. Done OUTSIDE the
                # per-matrix loop — gate_proj/up_proj share their Gram
                # (same input = layer hidden state) and down_proj has its
                # own. The down_proj Gram axes correspond to the SwiGLU
                # intermediate dimension; permuting the down_proj weight
                # along its input (column) axis requires permuting BOTH
                # axes of its Gram for consistency, since the Gram is the
                # outer product of the *labeled* intermediate-neuron
                # activations.
                if effective_merge_step == "regmean":
                    # cov_acc was validated non-None at the function top
                    # for the regmean branch; the .get below can still
                    # return None for a member that received zero
                    # calibration traffic (D-regmean-zero-cov-fallback).
                    assert cov_acc is not None, (
                        "regmean branch reached with cov_acc=None — the "
                        "top-of-function fallback should have demoted "
                        "effective_merge_step to 'freq_weighted'"
                    )
                    G_hidden_m = cov_acc.get((li, m, "gate_proj"))
                    G_down_m = cov_acc.get((li, m, "down_proj"))
                    if G_down_m is not None and perm is not None:
                        # Permute both axes of the (d_int, d_int) Gram so
                        # neuron labels align with the permuted weight's
                        # column axis. Use the same long-tensor index the
                        # weight permutation used.
                        perm_t = torch.as_tensor(
                            perm, dtype=torch.long, device=G_down_m.device,
                        )
                        G_down_m = G_down_m.index_select(0, perm_t).index_select(1, perm_t)
                    rm_G_hidden.append(G_hidden_m)
                    rm_G_down.append(G_down_m)

            if effective_merge_step == "mergemoe":
                # Replace the freq-weighted ``down_proj`` accumulator with
                # the MergeMoE closed-form result. Gate/up keep the
                # freq-weighted result — algebraically identical to MergeMoE
                # because T₂ = T₃ = freq-weights (paper Eq. 4).
                from .mergemoe import _mergemoe_compute_merged_down
                accs["down_proj"] = _mergemoe_compute_merged_down(
                    member_gates=mm_gates,
                    member_ups=mm_ups,
                    member_downs=mm_downs,
                    weights=list(weights),
                    layer_inputs=layer_inputs,
                    token_cap=token_cap,
                    seed=li,
                )

            if effective_merge_step == "regmean":
                # Replace ALL THREE accumulators with the RegMean closed-form
                # solve (paper Eq. 2, arXiv:2212.09849 — Jin et al. ICLR 2023).
                # Each Linear is solved independently:
                #   gate_proj / up_proj: shared input Gram (layer hidden state)
                #   down_proj:           own Gram (SwiGLU intermediate),
                #                        per-axis permuted by the same neuron-perm
                #                        applied to the down_proj column axis.
                # Per-cluster D-regmean-zero-cov-fallback: if any member's Gram
                # is None (zero calibration traffic to that expert), KEEP the
                # freq-weighted accs[name] (already filled above) and log a
                # WARNING. The freq-weighted accumulator was built in-band
                # during the per-member loop, so the cluster falls back
                # gracefully with no second pass needed.
                from .regmean import _regmean_solve_one_linear
                missing_gate_grams = any(g is None for g in rm_G_hidden)
                missing_down_grams = any(g is None for g in rm_G_down)
                if missing_gate_grams or missing_down_grams:
                    log.warning(
                        "layer %d centroid %d: D-regmean-zero-cov-fallback "
                        "(missing per-member Gram: hidden=%s, down=%s) — "
                        "falling back to freq-weighted merge for this cluster.",
                        li, centroid, missing_gate_grams, missing_down_grams,
                    )
                else:
                    alpha_list = [float(a) for a in weights]
                    # gate_proj uses rm_G_hidden grams; up_proj shares them.
                    accs["gate_proj"] = _regmean_solve_one_linear(
                        weights_per_member=rm_gates,
                        grams_per_member=rm_G_hidden,
                        alpha_per_member=alpha_list,
                    )
                    accs["up_proj"] = _regmean_solve_one_linear(
                        weights_per_member=rm_ups,
                        grams_per_member=rm_G_hidden,
                        alpha_per_member=alpha_list,
                    )
                    accs["down_proj"] = _regmean_solve_one_linear(
                        weights_per_member=rm_downs,
                        grams_per_member=rm_G_down,
                        alpha_per_member=alpha_list,
                    )

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
