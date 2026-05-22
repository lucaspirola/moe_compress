"""Output-space (Direction C) REAM cost plugin (Task 10 of the
plugin-architecture refactor).

Home of the four output-space-cost helpers — ``_swiglu_forward``,
``_router_routing_weights``, ``_tentative_merged_weights`` and
``_output_space_cost`` — that together implement the ``cost_alignment="output"``
branch of the REAM cost matrix (Direction C, spec STRATEGY_NEXT §C). All four
were moved verbatim out of ``stage2_reap_ream.py``; that module re-imports them
at module scope so external callers, tests and the ``MOE_STAGE2_LEGACY_LOOP=1``
path keep their existing import paths. ``_swiglu_forward`` is additionally
called by monolith-resident code (``_distill_merged_group``,
``_heal_student_moe_output``), so its re-import is load-bearing.

Circular-import note: this module imports only ``pipeline.permutation_align``,
``pipeline.base``, ``pipeline.context`` and ``moe_compress.utils.*`` — none of
which import ``stage2_reap_ream``, ``ream_cost`` or ``output_space_cost``. There
is therefore no cycle at module load. ``ream_cost._ream_cost_matrix``'s
``output`` branch imports ``_output_space_cost`` from *this* module at *function
scope*, kept symmetric with the ``post`` branch's ``_post_alignment_cost``
import; either could be module-top now, but symmetry costs nothing once the
module is cached.

``OutputSpaceCostPlugin`` is the future plugin home for the ``output`` cost
path. For T10 it is an inert shell: its ``compute_cost`` hook is a documented
no-op because the legacy bump loop still calls ``_ream_cost_matrix`` directly.
Wiring ``compute_cost`` into the phase walk is deferred until the assignment
phase is decomposed (T13+).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ...utils.activation_hooks import ReamCostAccumulator  # noqa: F401 — string type hint
from ...utils.model_io import MATRIX_NAMES, MoELayerRef, build_banks
from .._framework.base import Stage2Plugin
from ...pipeline.context import PipelineContext
from ..permutation_align import (
    _PermAlignCache,  # noqa: F401 — resolves the string type hint
    _permutation_align_to_centroid,
)


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
    """Full-softmax routing weights ``σ(x)_e`` for every routed expert.

    Recomputes the layer router's pre-softmax routing scores from the layer
    input ``x`` and applies a softmax over the expert axis — the same
    bias-adjusted pre-softmax → softmax path that ``capture_router_outputs``
    uses during profiling. Model-agnostic: it reads ``layer_ref.router``'s
    ``weight`` / ``bias`` / ``e_score_correction_bias`` attributes, none of
    which are Qwen-specific (any linear MoE router exposes ``weight``; the
    bias terms are simply skipped when absent).

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
) -> dict[str, torch.Tensor]:
    """Freq-weighted, permutation-aligned merge of ``child_id`` into
    ``centroid_id`` — the two-member case of ``_em_compute_tentative_weights``.

    ``W_merged = w_c · W_c + w_m · perm_m(W_m)`` where the weights are
    ``freq_e / (freq_c + freq_m)`` and ``perm_m`` aligns the child's
    intermediate neurons to the centroid (identity for the centroid itself).
    Returns float32 weights keyed by ``MATRIX_NAMES``. Model-agnostic: expert
    weights come through ``build_banks`` / ``MATRIX_NAMES``.

    Read-only on model params — wrapped in ``torch.no_grad()`` so the leaf
    ``nn.Parameter``s' ``requires_grad=True`` does not build an autograd graph
    (mirrors ``_post_alignment_cost``).
    """
    li = layer_ref.layer_idx
    banks = build_banks(layer_ref)

    f_c = max(int(freq.get(centroid_id, 0)), 0)
    f_m = max(int(freq.get(child_id, 0)), 0)
    denom = f_c + f_m
    if denom > 0:
        w_c, w_m = f_c / denom, f_m / denom
    else:
        w_c, w_m = 0.5, 0.5  # both zero — neutral average

    with torch.no_grad():
        ref_gate = banks["gate_proj"].get(centroid_id).to(torch.float32)
        ref_up   = banks["up_proj"].get(centroid_id).to(torch.float32)
        child_gate = banks["gate_proj"].get(child_id).to(torch.float32)
        child_up   = banks["up_proj"].get(child_id).to(torch.float32)

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
        perm_t = torch.as_tensor(perm, dtype=torch.long, device=ref_gate.device)

        merged: dict[str, torch.Tensor] = {}
        for name in MATRIX_NAMES:
            W_c = banks[name].get(centroid_id).to(torch.float32)
            W_m = banks[name].get(child_id).to(torch.float32)
            # gate/up permute the intermediate (row) axis; down permutes the
            # intermediate (column) axis. Mirrors _aligned_whitened_residual.
            if name == "down_proj":
                W_m = W_m[:, perm_t]
            else:
                W_m = W_m[perm_t, :]
            merged[name] = w_c * W_c + w_m * W_m
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
    merged into ``c``::

        cost(m→c) = mean_t [ σ_m(x_t) · ‖ E_m(x_t) − E_merged(x_t) ‖² ]

    where ``E_m`` is expert ``m``'s SwiGLU forward, ``E_merged`` is the
    forward of the freq-weighted permutation-aligned tentative merge of
    ``m`` into ``c``, and ``σ_m(x_t)`` is token ``t``'s full-softmax routing
    weight for expert ``m``, masked to zero on tokens where ``m`` is not in
    the token's top-``top_k`` routed set (so the cost only counts tokens that
    actually reach expert ``m`` — the topk-routing bound from the spec).

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

    banks = build_banks(layer_ref)
    # Resolve the compute device + dtype from the model's own expert weights
    # so the cost computation runs wherever the model lives (CPU or GPU),
    # with no hardcoded device.
    _probe = banks["gate_proj"].get(centroid_ids[0] if n_c else noncentroid_ids[0])
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

        # Full-softmax routing weights + the token's top-k routed expert set.
        sigma = _router_routing_weights(layer_ref, x_all)  # (T, n_experts)
        top_k = layer_ref.top_k
        k = min(top_k, sigma.shape[-1])
        topk_idx = torch.topk(sigma, k=k, dim=-1).indices  # (T, k)

        # Per non-centroid m: σ_m masked to tokens that route to m, and m's
        # own expert output E_m(x). Computed once per m (reused across its
        # K candidate centroids).
        for ci in range(n_nc):
            m_id = noncentroid_ids[ci]
            # routing-weighted mask: σ_m(x) on tokens that route to m, else 0.
            routed_m = (topk_idx == m_id).any(dim=-1)  # (T,) bool
            gate_m = sigma[:, m_id] * routed_m.to(sigma.dtype)  # (T,)
            if float(gate_m.sum()) <= 0.0:
                # No calibration token routes to m — the output cost is
                # undefined (every merge is "free" on unseen inputs). Leave
                # the whole row at +inf so the bump loop / orphan-promotion
                # path handles it, exactly as the post path does for a child
                # with no finite candidate. The cheap-cost top-K still picks
                # candidates, but a finite fallback is needed so the solver
                # has a feasible arc — fall back to the cheap symmetric cost.
                k_cand = min(topk, n_c)
                for cj in np.argpartition(cheap_cost[ci], k_cand - 1)[:k_cand]:
                    out[ci, int(cj)] = float(cheap_cost[ci, int(cj)])
                continue

            W_m = {name: banks[name].get(m_id).to(device, torch.float32)
                   for name in MATRIX_NAMES}
            E_m = _swiglu_forward(
                W_m["gate_proj"], W_m["up_proj"], W_m["down_proj"], x_all,
            )  # (T, hidden)

            k_cand = min(topk, n_c)
            top_cj = np.argpartition(cheap_cost[ci], k_cand - 1)[:k_cand]
            for cj in top_cj:
                cj = int(cj)
                c_id = centroid_ids[cj]
                merged = _tentative_merged_weights(
                    layer_ref, c_id, m_id, freq, ream_acc, perm_cache,
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


class OutputSpaceCostPlugin(Stage2Plugin):
    """Plugin home for the REAM output-space (Direction C) cost path.

    T10 status: inert shell. The legacy bump loop still calls
    ``_ream_cost_matrix`` directly, so this plugin's ``compute_cost`` hook is a
    deliberate no-op. The plugin exists now so the ``output`` path has a stable
    home; wiring ``compute_cost`` into the phase walk is deferred until the
    assignment phase is decomposed (T13+).
    """

    name = "output_space_cost"
    # enabled_by stays empty: the pre/post/output choice is a single tri-state
    # config value, not a set of boolean flags, so the base AND-of-flags
    # is_enabled cannot express "cost_alignment == 'output'". We override below.
    enabled_by: tuple[str, ...] = ()

    @classmethod
    def is_enabled(cls, cfg: dict[str, Any]) -> bool:
        """True iff ``stage2_reap_ream.cost_alignment`` resolves to ``"output"``.

        ``"output"`` is not the default, so a missing key or a missing
        ``stage2_reap_ream`` block leaves this plugin disabled. Case-insensitive
        to match the ``str(...).lower()`` normalization done in
        ``stage2_reap_ream.run()`` (config validation).
        """
        s2 = cfg.get("stage2_reap_ream", {}) if isinstance(cfg, dict) else {}
        return str(s2.get("cost_alignment", "pre")).lower() == "output"

    def compute_cost(self, ctx: PipelineContext) -> Any | None:
        """No-op for T10. See class docstring.

        Returning ``None`` makes ``PluginRegistry.dispatch_first`` skip this
        plugin so the legacy bump loop remains the sole cost-matrix producer.
        """
        return None
