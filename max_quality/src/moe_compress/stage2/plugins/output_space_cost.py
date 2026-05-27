"""Direction C: output-space REAM cost matrix.

Paper
-----
**No paper for this cost form.** Direction C is project-original —
introduced in the project's ``STRATEGY_NEXT`` document §C as a third
``cost_alignment`` mode (alongside ``pre`` and ``post``) for the Stage 2
v2 cost-matrix machinery. The baseline REAM symmetric cost comes from
arXiv:2604.04356 (see :mod:`stage2.plugins.ream_cost`); this branch
replaces the weight-/activation-space proxies with a forward-pass
MoE-block output residual.

Official code
-------------
None — this cost form is project-original. The reference REAM
implementation (``SamsungSAILMontreal/ream``) implements only the
symmetric ``δ_REAM`` cost.

Why "output space"
------------------
The ``pre`` and ``post`` cost forms are proxies for end-to-end merge
damage:

- ``pre`` (REAM Eq. 7) measures input-side similarity through router
  logits + gated-expert-output cosine — alignment-invariant and cheap,
  but doesn't capture the block-level interaction with other surviving
  experts.
- ``post`` (``D-whitened-cost`` — see
  :mod:`stage2.plugins.ream_cost` and the ``post`` branch in
  ``_aligned_whitened_residual``) measures weight-space residual along
  the activation-covariance eigenbasis — captures the AA-SVD-style
  whitened ΔW norm but is still a weight-only proxy.
- ``output`` (this branch, Direction C) measures the actual expert
  output divergence under a **tentative merge**: for each candidate
  ``(c, m)`` pair, build the freq-weighted permutation-aligned merge
  of ``{c, m}``, run the layer-input batch through a single-expert
  ``_swiglu_forward`` for both ``E_m`` and ``E_merged`` (the ORIGINAL
  router σ_m, no full-block replica and no resized router), and
  compare. Cost = routing-weight-weighted mean squared output distance
  ``E_m`` vs ``E_merged``, weighted by σ_m from the original router
  with the dispatch-time top-k membership mask applied.

The four module-level helpers implement this:

- ``_swiglu_forward`` — faithful single-expert SwiGLU forward
  (gate / up / down). Also called by ``_distill_merged_group``
  (``stage2.plugins.expert_distill`` L240/L250) and
  ``_heal_student_moe_output`` (``stage2.plugins.merge_heal`` L552),
  so the live re-export from this module via
  ``stage2.orchestrator`` (L133-L138) is load-bearing.
- ``_router_routing_weights`` — returns the **full unmasked softmax**
  ``σ(x)_e`` over all experts (REAM convention), with no top-k masking
  and no renormalization. The merge-cost call site
  (``_output_space_cost``) then multiplies by a 0/1 top-k membership
  mask to obtain ``σ_m(x)·1[m∈topk(x)]`` — i.e. REAP's masked gate
  ``g_j(x)`` **un-renormalized**. This deviates from both REAM's raw
  ``σ(x)`` and from ``D-reap-routing-weight`` (which renormalizes the
  top-k softmax, as used by :mod:`stage2.plugins.reap_scoring`); call
  this hybrid ``D-output-space-routing-weight``. Kept un-renormalized
  on purpose: the cost is a routing-weight-weighted mean (denominator
  = Σ_t gate_m(t)), so a per-token renormalization would cancel out of
  both numerator and denominator and only matter as an overall scale.
- ``_tentative_merged_weights`` — in-memory freq-weighted weighted
  average of ``{c, m}``'s gate/up/down (no model mutation), with the
  same Hungarian neuron permutation alignment used by the merge step
  (cached for that consumer; see D5b).
- ``_output_space_cost`` — the top-level cost-matrix builder that runs
  the K-prefilter (``cost_topk_filter``, default 48) and computes the
  output-residual cost for the surviving (c, m) pairs.

Deviation: D-output-space-routing-weight
----------------------------------------
The per-token weight used by ``_output_space_cost`` is

    gate_m(t) = σ_m(x_t) · 1[m ∈ topk(x_t)]

where ``σ`` is the **full unmasked softmax** from
``_router_routing_weights`` and the indicator is the dispatch-time
top-k membership mask. This is a deliberate hybrid:

- REAM convention uses raw ``σ(x)`` (full softmax, no top-k mask).
- ``D-reap-routing-weight`` (:mod:`stage2.plugins.reap_scoring`) uses
  the **renormalized** top-k softmax — the post-renormalization
  dispatch weight.
- ``D-output-space-routing-weight`` (this branch) keeps ``σ``
  **un-renormalized** but masks by top-k membership, yielding REAP's
  masked gate ``g_j(x)`` minus the renormalization.

Kept un-renormalized on purpose: the cost is a routing-weight-weighted
mean (denominator ``Σ_t gate_m(t)``), so a per-token renormalization
over the top-k set would cancel out of both numerator and denominator
and only matter as an overall scale.

Why this is opt-in (not the default)
------------------------------------
Output-space cost requires a forward pass per candidate pair (or per
top-K-prefiltered pair); even with the K-prefilter and the layer-input
reservoir, it is materially more expensive than ``post`` (per-pair
Frobenius norm) and ``pre`` (cosines off a single calibration pass).
Gated behind ``cost_alignment="output"`` (default: ``"pre"`` → baseline
REAM); used selectively when ablations indicate the merge damage is
sensitive to the block-level interaction the proxies miss.

Naming-history note
-------------------
"Direction C" is the project's STRATEGY_NEXT document label for this
cost form. The current plugin architecture has no direction-letter
taxonomy. New prose drops the label; existing log lines and Trackio
keys retain "Direction C" identifiers for dashboard back-compat.

Circular-import note: this module's top-level imports are
``...utils.activation_hooks`` (``ReamCostAccumulator``),
``...utils.model_io`` (``MATRIX_NAMES``, ``MoELayerRef``, ``build_banks``),
``...pipeline.context`` (``PipelineContext``),
``..permutation_align`` (``_PermAlignCache``,
``_permutation_align_to_centroid``) and ``.ream_cost``
(``_COST_PLUGIN_READS``, ``_COST_PLUGIN_WRITES``) — none of
which import ``stage2_reap_ream``, ``ream_cost`` or ``output_space_cost``. There
is therefore no cycle at module load. ``ream_cost._ream_cost_matrix``'s
``output`` branch imports ``_output_space_cost`` from *this* module at *function
scope*, kept symmetric with the ``post`` branch's ``_post_alignment_cost``
import; either could be module-top now, but symmetry costs nothing once the
module is cached.

``OutputSpaceCostPlugin`` is the live plugin home for the ``output`` cost path.
S2-6 wired its ``compute_cost`` hook into the ``compute_cost`` assignment slot
via the shared ``ream_cost._compute_cost_for_plugin`` helper: when
``cost_alignment`` resolves to ``"output"`` this plugin is registered ahead of
the ``LegacyAdapter`` and wins ``PluginRegistry.dispatch_first`` for the slot.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ...utils.activation_hooks import ReamCostAccumulator  # noqa: F401 — string type hint
from ...utils.model_io import MATRIX_NAMES, ExpertMatrixBank, MoELayerRef, build_banks
from ...pipeline.context import PipelineContext
from ..permutation_align import (
    _PermAlignCache,  # noqa: F401 — resolves the string type hint
    _permutation_align_to_centroid,
)
# Shared cost-plugin metadata (reads/writes slot tuples). ream_cost imports
# output_space_cost only at function scope, so this module-top import is
# cycle-free.
from .ream_cost import _COST_PLUGIN_READS, _COST_PLUGIN_WRITES


# ---------------------------------------------------------------------------
# Direction C — output-space merge cost (cost_alignment == "output")
# ---------------------------------------------------------------------------


def _swiglu_forward(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Standard SwiGLU FFN forward used by Qwen3-MoE experts.

    PyTorch nn.Linear weight shapes (used by the bank get/set):
        W_gate, W_up : (d_int, hidden)      — applied as ``F.linear(x, W)``
        W_down       : (hidden, d_int)
    Input ``x`` has shape ``(*, hidden)``; output has shape ``(*, hidden)``.
    """
    gate = F.linear(x, W_gate)
    up = F.linear(x, W_up)
    intermediate = F.silu(gate) * up
    return F.linear(intermediate, W_down)


def _router_routing_weights(
    layer_ref: MoELayerRef,
    x: torch.Tensor,
) -> torch.Tensor:
    """Full **unmasked** softmax routing weights ``σ(x)_e`` for every expert.

    Recomputes the layer router's pre-softmax routing scores from the layer
    input ``x`` and applies a softmax over the expert axis — the same
    bias-adjusted pre-softmax → softmax path that ``capture_router_outputs``
    uses during profiling. No top-k masking and no renormalization are
    applied here: this returns the raw REAM ``σ(x)`` distribution over all
    experts. The merge-cost call site (``_output_space_cost``) multiplies
    this by a 0/1 top-k membership mask to obtain the un-renormalized
    masked gate used as a per-token weight (see ``D-output-space-routing-weight``
    in the module docstring; distinct from ``D-reap-routing-weight``, which
    renormalizes the top-k softmax).

    Model-agnostic: it reads ``layer_ref.router``'s ``weight`` / ``bias`` /
    ``e_score_correction_bias`` attributes, none of which are Qwen-specific
    (any linear MoE router exposes ``weight``; the bias terms are simply
    skipped when absent).

    ``x`` : ``(n_tokens, hidden)``. Returns ``(n_tokens, n_experts)`` float32.
    """
    router = layer_ref.router
    w = router.weight
    logits = F.linear(x.to(w.dtype), w)
    bias = getattr(router, "bias", None)
    if bias is not None:
        logits = logits + bias
    esc = getattr(router, "e_score_correction_bias", None)
    if esc is not None:
        logits = logits + esc
    return F.softmax(logits.float(), dim=-1)


def _tentative_merged_weights(
    layer_ref: MoELayerRef,
    centroid_id: int,
    child_id: int,
    freq: dict[int, int],
    ream_acc: "ReamCostAccumulator | None",
    perm_cache: "_PermAlignCache | None",
    banks: dict[str, ExpertMatrixBank],
    *,
    merge_step: str = "freq_weighted",
    layer_inputs: torch.Tensor | None = None,
    token_cap: int = 1024,
) -> dict[str, torch.Tensor]:
    """Freq-weighted, permutation-aligned merge of ``child_id`` into
    ``centroid_id`` — the two-member case of ``_em_compute_tentative_weights``.

    ``W_merged = w_c · W_c + w_m · perm_m(W_m)`` where the weights are
    ``freq_e / (freq_c + freq_m)`` and ``perm_m`` aligns the child's
    intermediate neurons to the centroid (identity for the centroid itself).
    Returns weights in the model's native dtype keyed by ``MATRIX_NAMES``. Model-agnostic: expert
    weights come through ``build_banks`` / ``MATRIX_NAMES``.

    Read-only on model params — wrapped in ``torch.no_grad()`` so the leaf
    ``nn.Parameter``s' ``requires_grad=True`` does not build an autograd graph
    (mirrors ``_post_alignment_cost``).

    Per SC_FAST_PLAN_V3.md §4-B1: caches the freshly-computed permutation
    under ``(li, centroid_id, child_id)`` for reuse by the eventual merge
    step (``_merge_experts_inplace``). Side-effect only; cost matrix is
    byte-identical.

    Per SC_FAST_PLAN_V3.md §4-B4: ``banks`` is now passed by the caller,
    hoisted out of the per-pair call to amortize the ``build_banks`` cost
    over the full cost-matrix loop. Callers must pass the same ``banks``
    dict that ``build_banks(layer_ref)`` would return — this function no
    longer builds it internally.

    Per SC_FAST_PLAN_V3.md §4-B2: the six ``.to(torch.float32)`` upcasts on
    ``ref_gate``, ``ref_up``, ``child_gate``, ``child_up``, ``W_c``, ``W_m``
    have been removed. Merge arithmetic runs in the model's native dtype
    (bf16 for Qwen3.6-35B-A3B; float32 for tests). Callers needing float32
    outputs apply ``.to(device, torch.float32)`` on the returned dict
    (see ``_output_space_cost``: the W_m and merged dict comprehensions).
    Documented relative drift O(1e-3) bounded by
    ``test_output_cost_bf16_drift_under_threshold``.

    Note on dtype promotion: ``w_c * W_c + w_m * W_m`` where ``w_c``/``w_m``
    are Python float scalars and ``W_c``/``W_m`` are bf16 tensors stays bf16
    (PyTorch scalar-type rule: result dtype follows the tensor's dtype,
    Python float scalars do NOT promote bf16 → fp32).

    MergeMoE branch (``merge_step="mergemoe"``)
    -------------------------------------------
    When ``merge_step="mergemoe"`` and ``layer_inputs`` is non-empty, the
    merged ``down_proj`` is replaced by the MergeMoE closed-form least-squares
    solution ``T₁ = Q · P†`` (paper arXiv:2510.14436 Eqs. 3–6 — see
    :mod:`stage2.mergemoe`). Gate and up projections remain the freq-weighted
    average (algebraically identical to MergeMoE T₂/T₃ per paper Eq. 4).

    The default (``merge_step="freq_weighted"``) is byte-identical to legacy
    behaviour. The only caller from this module (``_output_space_cost``) does
    NOT opt into MergeMoE — the cost matrix continues to use the freq-weighted
    tentative-merge proxy. The MergeMoE branch is exposed here so the
    function signature is symmetric with ``_merge_experts_inplace`` (both
    accept the same merge_step config knob); external callers that want to
    evaluate MergeMoE-style merge damage as a cost proxy can opt in.
    """
    li = layer_ref.layer_idx

    f_c = max(int(freq.get(centroid_id, 0)), 0)
    f_m = max(int(freq.get(child_id, 0)), 0)
    denom = f_c + f_m
    if denom > 0:
        w_c, w_m = f_c / denom, f_m / denom
    else:
        w_c, w_m = 0.5, 0.5  # both zero — neutral average

    with torch.no_grad():
        ref_gate   = banks["gate_proj"].get(centroid_id)
        ref_up     = banks["up_proj"].get(centroid_id)
        child_gate = banks["gate_proj"].get(child_id)
        child_up   = banks["up_proj"].get(child_id)

        cached = perm_cache.get((li, centroid_id, child_id)) if perm_cache is not None else None
        if cached is not None:
            perm = cached[0]
        else:
            ref_act   = ream_acc.get_neuron_mean(li, centroid_id) if ream_acc else None
            child_act = ream_acc.get_neuron_mean(li, child_id) if ream_acc else None
            perm = _permutation_align_to_centroid(
                ref_gate, ref_up, child_gate, child_up,
                ref_act_mean=ref_act, child_act_mean=child_act,
            )
            # Persist the freshly-computed permutation so the eventual merge
            # step reuses it instead of re-running LAP for the same pair.
            # Mirrors ream_cost_post.py:285. residual=None because the output
            # path does not compute a whitened Frobenius residual.
            # See SC_FAST_PLAN_V3.md §4-B1.
            if perm_cache is not None:
                perm_cache.put((li, centroid_id, child_id), perm, residual=None)
        perm_t = torch.as_tensor(perm, dtype=torch.long, device=ref_gate.device)

        merged: dict[str, torch.Tensor] = {}
        for name in MATRIX_NAMES:
            W_c = banks[name].get(centroid_id)
            W_m = banks[name].get(child_id)
            # gate/up permute the intermediate (row) axis; down permutes the
            # intermediate (column) axis. Mirrors _aligned_whitened_residual.
            if name == "down_proj":
                W_m = W_m[:, perm_t]
            else:
                W_m = W_m[perm_t, :]
            merged[name] = w_c * W_c + w_m * W_m

        # MergeMoE override (opt-in; default-off branch). Replace ``down_proj``
        # with the closed-form T₁=Q·P† solution; gate/up unchanged (paper
        # T₂/T₃ collapse to the freq-weighted average per Eq. 4, so the
        # already-computed freq-weighted gate/up tensors are correct as-is).
        # ``layer_inputs`` empty → silent stay-on-freq-weighted (this is a
        # cost-matrix tentative proxy, not the final merge; the actual merge
        # at ``_merge_experts_inplace`` enforces the contract loudly).
        if merge_step == "mergemoe" and layer_inputs is not None and layer_inputs.numel() > 0:
            from ..mergemoe import _mergemoe_compute_merged_down
            # Cast aligned member tensors to fp32 for the solve (matches the
            # numerical posture of ``_merge_experts_inplace``).
            gate_c_fp32 = banks["gate_proj"].get(centroid_id).to(torch.float32)
            up_c_fp32   = banks["up_proj"].get(centroid_id).to(torch.float32)
            down_c_fp32 = banks["down_proj"].get(centroid_id).to(torch.float32)
            gate_m_fp32 = banks["gate_proj"].get(child_id).to(torch.float32)[perm_t, :]
            up_m_fp32   = banks["up_proj"].get(child_id).to(torch.float32)[perm_t, :]
            down_m_fp32 = banks["down_proj"].get(child_id).to(torch.float32)[:, perm_t]
            merged["down_proj"] = _mergemoe_compute_merged_down(
                member_gates=[gate_c_fp32, gate_m_fp32],
                member_ups=[up_c_fp32, up_m_fp32],
                member_downs=[down_c_fp32, down_m_fp32],
                weights=[w_c, w_m],
                layer_inputs=layer_inputs,
                token_cap=token_cap,
                seed=li,
            )
    return merged


def _output_space_cost(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    cheap_cost: np.ndarray,
    ream_acc: "ReamCostAccumulator | None",
    perm_cache: "_PermAlignCache | None",
    topk: int,
    freq: dict[int, int] | None,
    layer_inputs: torch.Tensor | None,
    token_cap: int,
) -> np.ndarray:
    """Direction C — output-space merge cost matrix (spec STRATEGY_NEXT §C).

    For each non-centroid ``m`` and each of its top-K cheapest candidate
    centroids ``c``, the cost is the routing-weighted change in expert ``m``'s
    *gated routed output* on the calibration tokens when ``m`` is tentatively
    merged into ``c``. Concretely, with the per-token weight

        gate_m(t) = σ_m(x_t) · 1[m ∈ topk(x_t)]

    (full unmasked softmax from ``_router_routing_weights``, then masked
    by top-k membership at this call site — ``D-output-space-routing-weight``
    in the module docstring; un-renormalized), the cost is a
    **routing-weight-weighted mean** over tokens::

        cost(m→c) = Σ_t gate_m(t) · ‖ E_m(x_t) − E_merged(x_t) ‖²
                    ─────────────────────────────────────────────
                                  Σ_t gate_m(t)

    where ``E_m`` is expert ``m``'s SwiGLU forward and ``E_merged`` is the
    forward of the freq-weighted permutation-aligned tentative merge of
    ``m`` into ``c``. The denominator is ``Σ_t gate_m(t)`` (not the token
    count ``T``); tokens that do not route to ``m`` carry ``gate_m=0`` and
    drop out of both sums — the cost counts only tokens that actually
    reach expert ``m`` (the topk-routing bound from the spec). Because
    both numerator and denominator share the same per-token weight,
    any per-token renormalization of ``σ`` over the top-k set would
    cancel — which is why we keep ``σ`` un-renormalized.

    This is a strictly better merge-damage proxy than the weight-space
    ``pre`` / ``post`` costs: it measures the realised change in what the
    layer emits, not a norm on the weights.

    All non-candidate entries get ``+inf`` so the assignment solver treats
    them as forbidden, exactly like the ``post`` path.

    Model-agnostic: expert count / dims come from ``build_banks`` /
    ``MATRIX_NAMES``; ``top_k`` from ``layer_ref.top_k``; routing weights from
    ``layer_ref.router``; the FFN forward is ``_swiglu_forward`` — the same
    activation abstraction the ``post`` and distillation paths already use.
    """
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)

    if topk < 1:
        raise ValueError(
            f"_output_space_cost: cost_topk_filter={topk} < 1 — must be at "
            "least the per-centroid capacity to leave a feasible assignment."
        )
    if layer_inputs is None or layer_inputs.shape[0] == 0:
        raise RuntimeError(
            "_output_space_cost: no layer-input calibration tokens were "
            "captured — the _LayerInputAccumulator must be enabled when "
            "cost_alignment == 'output' (check the Stage 2 driver)."
        )
    if freq is None:
        raise ValueError(
            "_output_space_cost: freq dict is required (the tentative merge "
            "is freq-weighted)."
        )

    out = np.full((n_nc, n_c), np.inf, dtype=np.float64)

    # Edge guards (LOW-2):
    #  * If both ``centroid_ids`` and ``noncentroid_ids`` are empty the device-probe
    #    below would IndexError; upstream invariants should prevent this (the
    #    bump loop only calls this cost path when at least one side is non-
    #    empty), but we return the empty matrix to make the contract explicit.
    #  * If either side is empty there is no (m, c) pair to score: return early.
    # Empty matrix → no work; return the all-+∞ out array as-is. The
    # argpartition below would raise on shape[1]==0 otherwise.
    if n_nc == 0 or n_c == 0:
        return out

    # Hoist per-row argpartition out of the ci loop: cheap_cost is read-only
    # and k_cand is loop-invariant, so this is equivalent to per-row form.
    # See SC_FAST_PLAN_V3.md §4-B3.
    k_cand = min(topk, n_c)
    topk_per_ci = np.argpartition(cheap_cost, k_cand - 1, axis=1)[:, :k_cand]  # (n_nc, k_cand)

    banks = build_banks(layer_ref)
    # Resolve the compute device from the model's own expert weights so the
    # cost computation runs wherever the model lives (CPU or GPU), with no
    # hardcoded device. (Dtype is hardcoded to fp32 at the SwiGLU call sites
    # below.) Safe to index ``centroid_ids[0]`` — the empty case is handled
    # by the early-return above.
    _probe = banks["gate_proj"].get(centroid_ids[0])
    device = _probe.device

    with torch.no_grad():
        # Calibration tokens: deterministic per-layer subsample, capped so the
        # per-pair SwiGLU forward stays bounded (~1k tokens by default).
        x_all = layer_inputs.reshape(-1, layer_inputs.shape[-1])
        n_tokens = x_all.shape[0]
        if n_tokens > token_cap:
            rng = torch.Generator(device="cpu").manual_seed(layer_ref.layer_idx)
            idx = torch.randperm(n_tokens, generator=rng)[:token_cap]
            x_all = x_all[idx]
        x_all = x_all.to(device, dtype=torch.float32)

        # Full-softmax routing weights from ``_router_routing_weights`` — no
        # masking and no renormalization. We multiply by a 0/1 top-k membership
        # mask below to obtain the un-renormalized masked gate ``g_j(x)``
        # (``D-output-space-routing-weight``; see module docstring).
        # ``router_top_k`` (the router's dispatch-time top-k) is named to
        # disambiguate from the ``topk`` parameter above, which is the
        # cost-matrix candidate-prefilter K (``cost_topk_filter``).
        sigma = _router_routing_weights(layer_ref, x_all)  # (T, n_experts)
        router_top_k = layer_ref.top_k
        # clamp in case top_k exceeds n_experts on degenerate configs
        k = min(router_top_k, sigma.shape[-1])
        topk_idx = torch.topk(sigma, k=k, dim=-1).indices  # (T, k)

        # Per non-centroid m: σ_m masked to tokens that route to m, and m's
        # own expert output E_m(x). Computed once per m (reused across its
        # K candidate centroids).
        for ci in range(n_nc):
            m_id = noncentroid_ids[ci]
            # routing-weighted mask: σ_m(x) on tokens that route to m, else 0.
            routed_m = (topk_idx == m_id).any(dim=-1)  # (T,) bool
            gate_m = sigma[:, m_id] * routed_m.to(sigma.dtype)  # (T,)
            # gate_m = softmax(·) * {0,1} is nonneg, so == 0.0 would suffice;
            # kept as <= 0.0 defensively in case of fp underflow / NaN guards.
            if float(gate_m.sum()) <= 0.0:
                # No calibration token routes to m — the output cost is
                # undefined (every merge is "free" on unseen inputs). The
                # cheap-cost top-K still picks candidates, and a finite
                # fallback is needed so the assignment solver retains
                # feasible arcs for this row — fill those top-K entries
                # with the cheap symmetric REAM cost (finite fallback),
                # leaving the remaining entries at the row's initial +∞.
                for cj in topk_per_ci[ci].tolist():
                    out[ci, cj] = float(cheap_cost[ci, cj])
                continue

            W_m = {name: banks[name].get(m_id).to(device, torch.float32)
                   for name in MATRIX_NAMES}
            E_m = _swiglu_forward(
                W_m["gate_proj"], W_m["up_proj"], W_m["down_proj"], x_all,
            )  # (T, hidden)

            top_cj = topk_per_ci[ci].tolist()  # 1-D K-element Python int list — eliminates per-iteration int() cast
            for cj in top_cj:
                c_id = centroid_ids[cj]
                merged = _tentative_merged_weights(
                    layer_ref, c_id, m_id, freq, ream_acc, perm_cache, banks,
                )
                merged = {name: merged[name].to(device, torch.float32)
                          for name in MATRIX_NAMES}
                E_merged = _swiglu_forward(
                    merged["gate_proj"], merged["up_proj"], merged["down_proj"],
                    x_all,
                )  # (T, hidden)
                # Routing-weighted mean squared output change for expert m.
                per_token = (E_m - E_merged).pow(2).sum(dim=-1)  # (T,)
                cost = (gate_m * per_token).sum() / gate_m.sum()
                out[ci, cj] = float(cost)

    return out


class OutputSpaceCostPlugin:
    """Live plugin home for the REAM output-space (Direction C) cost path.

    S2-6 wired ``compute_cost`` into the ``compute_cost`` assignment slot. When
    ``cost_alignment`` resolves to ``"output"`` the orchestrator registers this
    plugin ahead of the ``LegacyAdapter`` so it wins
    ``PluginRegistry.dispatch_first`` for the slot. The slot body — the
    ``_ream_cost_matrix`` call — lives in the shared module-level helper
    ``ream_cost._compute_cost_for_plugin`` (one source of truth for all three
    cost plugins). S2-10 moved the capacity-util gate out into
    ``CapacityGatePlugin.select_alignment`` (which may still downgrade
    ``output``→``pre`` per layer on slack-capacity layers, run earlier in the
    bump iteration); this slot reads the gate's decision back off ``ctx``.
    """

    name = "output_space_cost"
    paper = (
        "Output-space merge-cost matrix (project-original; no paper). "
        "Direction C from STRATEGY_NEXT §C. Replaces baseline REAM "
        "arXiv:2604.04356 (see :mod:`stage2.plugins.ream_cost`). "
        "Routing-weight-weighted mean squared output distance under "
        "tentative merge — opt-in via cost_alignment='output'; default "
        "'pre'. See module docstring."
    )
    config_key = "stage2_reap_ream.cost_alignment"
    reads: tuple[str, ...] = _COST_PLUGIN_READS
    writes: tuple[str, ...] = _COST_PLUGIN_WRITES  # () — S2-10 moved the gate out
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        cov_acc,
        cost_alignment_cfg: str,
        cost_whitening: str,
        cost_topk_filter: int,
        cost_output_token_cap: int,
    ) -> None:
        # Store every knob the shared compute_cost body reads. NO logic — a
        # faithful mirror of the matching subset of LegacyAdapter.__init__.
        # ``cost_alignment_cfg`` is retained for ``is_enabled``; the capacity
        # gate's knobs moved to CapacityGatePlugin (S2-10).
        self.cov_acc = cov_acc
        self.cost_alignment_cfg = cost_alignment_cfg
        self.cost_whitening = cost_whitening
        self.cost_topk_filter = cost_topk_filter
        self.cost_output_token_cap = cost_output_token_cap

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.cost_alignment`` resolves to ``"output"``.

        ``"output"`` is not the default, so a missing key or a missing
        ``stage2_reap_ream`` block leaves this plugin disabled. Case-insensitive
        to match the ``str(...).lower()`` normalization done in
        ``stage2_reap_ream.run()`` (config validation).
        """
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        return str(s2.get("cost_alignment", "pre")).lower() == "output"

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def compute_cost(self, ctx: PipelineContext) -> Any | None:
        """Slot ``compute_cost`` — REAM cost matrix.

        Delegates to the shared ``ream_cost._compute_cost_for_plugin`` helper
        (same body as the other two cost plugins). Reads the capacity-gate
        decision off ``ctx`` (published by ``CapacityGatePlugin.select_alignment``
        earlier in the bump iteration). Returns the cost matrix ``delta``.
        """
        from .ream_cost import _compute_cost_for_plugin
        return _compute_cost_for_plugin(self, ctx)
