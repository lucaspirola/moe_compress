"""REAM cost-matrix construction (Task 8 of the plugin-architecture refactor).

Home of ``_ream_cost_matrix`` — the REAM cost-matrix builder with three
alignment modes (``pre`` / ``post`` / ``output``) — and its vectorized helper
``_extract_sim_expert_matrix_from_tensor``. Both moved verbatim out of
``stage2_reap_ream.py``; that module re-imports them so external callers and
tests keep their existing import paths.

Circular-import note: ``stage2_reap_ream`` imports *this* module at load time,
so a module-top ``from ...stage2_reap_ream import ...`` here would deadlock the
import — neither branch does that. Both the ``post`` (``ream_cost_post``, T9)
and the ``output`` (``output_space_cost``, T10) branches now import their cost
helper from a cycle-free sibling plugin module via a single-dot **function-scope
import**, kept symmetric. Either could be a module-top import now, but the
function-scope form costs nothing once the module is cached.

``ReamCostPrePlugin`` is the live plugin home for the ``pre`` cost path. S2-6
wired its ``compute_cost`` hook into the ``compute_cost`` assignment slot: when
``cost_alignment`` resolves to ``"pre"`` this plugin is registered ahead of the
``LegacyAdapter`` and wins ``PluginRegistry.dispatch_first`` for the slot. All
three cost plugins share the verbatim slot body via the module-level helper
``_compute_cost_for_plugin`` below to avoid three drifting copies.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...utils.activation_hooks import (
    InputCovarianceAccumulator,  # noqa: F401 — resolves the string type hint
    ReamCostAccumulator,
)
from ...utils.model_io import MoELayerRef
from ...pipeline.context import PipelineContext
from ..permutation_align import _PermAlignCache  # noqa: F401 — string type hint


def _extract_sim_expert_matrix_from_tensor(
    sim_tensor: "torch.Tensor | None",
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    total_tokens: int,
) -> np.ndarray:
    """Vectorized δ̃_expert(child, centroid) submatrix from the dense sim tensor.

    Replaces the former ``n_nc × n_c`` double loop over
    ``ReamCostAccumulator.compute_delta_expert`` (one lock-protected call per
    pair, ~14.6K calls/layer at Qwen scale). Reads the dense ``[E, E]``
    float64 ``_sim_tensor`` once and broadcasts the REAM Eq. 8 rescale over
    the whole submatrix.

    Args:
        sim_tensor: the layer's ``[E, E]`` float64 gated-output cosine-sum
            accumulator (``ReamCostAccumulator._sim_tensor[layer_idx]``), or
            ``None`` if no batch was finalized for the layer.
        noncentroid_ids: child (row) expert IDs.
        centroid_ids: centroid (column) expert IDs.
        total_tokens: |X|, the Eq. 8 denominator (total calibration tokens).

    Returns:
        ``(n_nc, n_c)`` float64 ndarray of δ̃_expert similarities ∈ [0, 1].

    Equivalence with the old per-pair path: ``compute_delta_expert`` returns
        * ``NaN`` when ``total_tokens == 0`` (or no sim data) — the caller
          substituted ``0.5`` (neutral after the (cos+1)/2 rescale);
        * else ``clip((sim_val / total + 1) / 2, 0, 1)``.
      A ``None`` sim_tensor means ``sim_val == 0`` for every pair, so the old
      path yielded ``clip((0/total + 1)/2, 0, 1) == 0.5`` — identical to the
      ``total_tokens == 0`` neutral fill. We therefore collapse both
      degenerate cases to a full-0.5 matrix.
    """
    n_nc = len(noncentroid_ids)
    n_c = len(centroid_ids)
    if total_tokens == 0 or sim_tensor is None:
        # Degenerate: no joint-activation data → neutral 0.5 everywhere
        # (matches compute_delta_expert's NaN→0.5 substitution and the
        # sim_val==0 → 0.5 algebra; see docstring).
        return np.full((n_nc, n_c), 0.5, dtype=np.float64)
    # Index the [E, E] accumulator down to the (child, centroid) submatrix.
    # advanced indexing on the first axis then the second produces the
    # n_nc × n_c block in C-row order matching the old nested loop.
    nc_idx = torch.as_tensor(noncentroid_ids, dtype=torch.long)
    c_idx = torch.as_tensor(centroid_ids, dtype=torch.long)
    sub = sim_tensor.to(torch.float64)[nc_idx][:, c_idx]  # (n_nc, n_c)
    sim = ((sub / total_tokens + 1.0) / 2.0).clamp_(0.0, 1.0)
    return sim.numpy().copy()


def _ream_cost_matrix(
    layer_ref: MoELayerRef,
    noncentroid_ids: list[int],
    centroid_ids: list[int],
    *,
    ream_acc: ReamCostAccumulator,
    blacklisted_ids: set[int] | None = None,
    cost_alignment: str = "pre",
    cost_whitening: str = "none",
    cost_asymmetric: bool = False,
    cost_topk_filter: int = 48,
    freq: dict[int, int] | None = None,
    cov_acc: "InputCovarianceAccumulator | None" = None,
    perm_cache: "_PermAlignCache | None" = None,
    tentative_centroid_weights: dict[int, dict[str, torch.Tensor]] | None = None,
    layer_inputs: torch.Tensor | None = None,
    output_token_cap: int = 1024,
) -> np.ndarray:
    """Compute the (n_nc × n_c) REAM cost matrix.

    Three modes (Stage 2 v2 spec § 5 step 4 + Direction C):

    - ``cost_alignment="pre"`` (default, v1 behavior): symmetric δ_REAM cost
      ``1 - (δ_gate + δ̃_expert)/2`` over all pairs.
    - ``cost_alignment="post"`` (Tier 2 / v2 path): for each non-centroid m,
      compute the cheap symmetric cost first; take the top-K candidates by
      cheap cost; for those candidates only, compute the per-pair Hungarian
      alignment cost and the whitened Frobenius residual
      ``R_cm = ‖(W_c − P_cm·W_m) · A^{1/2}‖_F`` (sum over gate/up/down per
      § 5 step 4T(c)(ii)). All other entries get +∞ so the assignment solver
      treats them as forbidden. Permutations and residuals are stashed in
      ``perm_cache`` for the merge step to reuse (M1).
    - ``cost_alignment="output"`` (Direction C): for each non-centroid m,
      take the top-K candidates by cheap cost; for those candidates compute
      the *output-space* cost — the routing-weighted change in expert m's
      gated routed output on the captured calibration tokens when m is
      tentatively merged into the centroid. See ``_output_space_cost``.
      Requires ``layer_inputs`` (the layer-input calibration buffer) and
      ``freq`` (for the freq-weighted tentative merge).

    When ``cost_asymmetric=True`` and ``freq`` is provided, the post-alignment
    residual is multiplied by ``freq_m / (freq_c + freq_m)`` (spec § 5 step
    4T(c)(iii) / D-asymmetric-freq). This is valid only under the
    freq-weighted merge path; the caller is responsible for that invariant.
    """
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

    # δ̃_expert submatrix (REAM Eq. 8): vectorized read of the dense [E, E]
    # gated-output cosine-sum accumulator, replacing a former n_nc × n_c
    # double loop over compute_delta_expert (~14.6K lock-protected calls per
    # layer at Qwen3.6 scale). The NaN→0.5 substitution that the old loop
    # applied per pair is folded into _extract_sim_expert_matrix_from_tensor
    # (degenerate total_tokens==0 / no-data → full-0.5 matrix); see its
    # docstring for the equivalence argument.
    #
    # Lock discipline (R4): snapshot the total-token count and the sim
    # tensor reference under ream_acc._lock, then compute outside the lock.
    # This is safe because Stage 2 finalizes ALL batches for a layer before
    # calling _ream_cost_matrix — there is no concurrent finalize_batch
    # mutating _sim_tensor[li] during this read. Snapshotting the dict
    # reference under the lock still guards against a concurrent clear_layer.
    with ream_acc._lock:
        total_tokens = ream_acc._total_tokens_by_layer.get(li, 0)
        sim_t = ream_acc._sim_tensor.get(li)
    sim_expert_matrix = _extract_sim_expert_matrix_from_tensor(
        sim_t, noncentroid_ids, centroid_ids, total_tokens,
    )  # (n_nc, n_c) float64

    # δ_REAM = (δ_gate + δ̃_expert) / 2 ∈ [0,1]; cost = 1 − δ_REAM ∈ [0,1].
    # Lower cost = more similar (spec §5 Step 2, reference ream/ream.py L46-53).
    cost = 1.0 - (sim_gate_sub + sim_expert_matrix) / 2.0
    np.clip(cost, 0.0, 1.0, out=cost)

    if cost_alignment == "pre":
        return cost

    if cost_alignment == "output":
        # Function-scope import: _output_space_cost lives in the sibling plugin
        # module output_space_cost (T10). output_space_cost is itself cycle-free,
        # but the import is kept function-scope to stay symmetric with the
        # ``post`` branch's _post_alignment_cost import and costs nothing once
        # the module is cached.
        from .output_space_cost import _output_space_cost
        # Direction C — output-space merge cost. ``cost`` (the cheap symmetric
        # δ_REAM) is reused only as the top-K candidate filter, exactly like
        # the "post" path uses it.
        return _output_space_cost(
            layer_ref,
            noncentroid_ids,
            centroid_ids,
            cheap_cost=cost,
            ream_acc=ream_acc,
            perm_cache=perm_cache,
            topk=cost_topk_filter,
            freq=freq,
            layer_inputs=layer_inputs,
            token_cap=output_token_cap,
        )

    if cost_alignment != "post":
        raise ValueError(
            f"_ream_cost_matrix: unknown cost_alignment={cost_alignment!r}; "
            "expected 'pre', 'post', or 'output'."
        )

    # Function-scope import: _post_alignment_cost lives in the sibling plugin
    # module ream_cost_post (T9). ream_cost_post is itself cycle-free, but the
    # import is kept function-scope to stay symmetric with the still-monolith
    # ``output`` branch above and costs nothing once the module is cached.
    from .ream_cost_post import _post_alignment_cost
    # Stage 2 v2: post-alignment whitened residual path (spec § 5 step 4T).
    return _post_alignment_cost(
        layer_ref,
        noncentroid_ids,
        centroid_ids,
        cheap_cost=cost,
        ream_acc=ream_acc,
        cov_acc=cov_acc,
        perm_cache=perm_cache,
        whitening_mode=cost_whitening,
        asymmetric=cost_asymmetric,
        topk=cost_topk_filter,
        freq=freq,
        tentative_centroid_weights=tentative_centroid_weights,
    )


def _compute_cost_for_plugin(plugin, ctx: PipelineContext) -> np.ndarray:
    """Shared ``compute_cost`` slot body for the three live cost plugins.

    Verbatim lift of ``LegacyAdapter.compute_cost`` — the capacity-utilization
    gate + ``_ream_cost_matrix`` call from the old ``compute_assignment``
    bump-loop ``if not b_fail`` branch. The cost config knobs are read off
    ``plugin.`` (each cost plugin stores them as instance attributes mirroring
    ``LegacyAdapter.__init__``) instead of ``self.``; everything else — the ctx
    reads, the capacity gate, the three ``ctx.set(..., overwrite=True)`` writes,
    the ``_ream_cost_matrix`` call — is byte-identical to the legacy adapter.

    The gate may downgrade a ``post``/``output``-configured plugin to ``pre``
    per layer (slack-capacity layers) — that is correct and intentional, so all
    three plugins keep the full gate. Writes ``capacity_util_value`` /
    ``effective_cost_alignment`` / ``effective_cost_asymmetric`` back to ``ctx``
    (``overwrite=True``). Reads the ``_iter_*`` scratch slots published by the
    orchestrator. Returns the cost matrix ``delta``.
    """
    from .capacity_gate import _pick_effective_alignment

    layer_ref = ctx.get("layer_ref")
    ream_acc = ctx.get("ream_acc")
    perm_cache = ctx.get("perm_cache")
    layer_input_acc = ctx.get("layer_input_acc")
    cov_acc = plugin.cov_acc
    freq = ctx.get("freq")
    protected = set(ctx.get("protected"))
    ream_centroid_ids = list(ctx.get("_iter_ream_centroid_ids"))
    ream_noncentroid_ids = list(ctx.get("_iter_ream_noncentroid_ids"))
    n_ream_c = ctx.get("_iter_n_ream_c")
    n_ream_nc = ctx.get("_iter_n_ream_nc")

    # Stage 2 v2 capacity-utilization gate (M3, spec § 5 step 3):
    #   u = n_NC / (N'_l × C_max). When u < threshold, the layer
    #   has so much slack capacity that the heavyweight
    #   post-alignment cost matrix is unlikely to change the
    #   assignment meaningfully — fall back to the cheap symmetric
    #   path. This is what skips ~half the layers' compute.
    # Capture the actual u value into the layer-scope variable
    # so the per-layer Trackio emit can surface it; mirrors the
    # division done inside _pick_effective_alignment.
    if plugin.max_group_cap <= 0:
        capacity_util_value = 0.0
    else:
        capacity_util_value = n_ream_nc / max(n_ream_c * plugin.max_group_cap, 1)
    effective_cost_alignment = _pick_effective_alignment(
        n_nc=n_ream_nc,
        n_c=n_ream_c,
        max_group_cap=plugin.max_group_cap,
        threshold=plugin.capacity_util_threshold,
        configured=plugin.cost_alignment_cfg,
    )
    effective_cost_asymmetric = (
        plugin.cost_asymmetric and effective_cost_alignment == "post"
    )
    ctx.set("capacity_util_value", capacity_util_value, overwrite=True)
    ctx.set("effective_cost_alignment", effective_cost_alignment, overwrite=True)
    ctx.set("effective_cost_asymmetric", effective_cost_asymmetric, overwrite=True)
    delta = _ream_cost_matrix(
        layer_ref, ream_noncentroid_ids, ream_centroid_ids,
        ream_acc=ream_acc,
        blacklisted_ids=protected,
        cost_alignment=effective_cost_alignment,
        cost_whitening=plugin.cost_whitening,
        cost_asymmetric=effective_cost_asymmetric,
        cost_topk_filter=plugin.cost_topk_filter,
        freq=(
            freq
            if (effective_cost_asymmetric
                or effective_cost_alignment == "output")
            else None
        ),
        cov_acc=cov_acc if effective_cost_alignment == "post" else None,
        perm_cache=perm_cache,
        # Direction C: calibration tokens for the output-space cost.
        # None for "pre"/"post" — those paths never read it.
        layer_inputs=(
            layer_input_acc.get()
            if (effective_cost_alignment == "output"
                and layer_input_acc is not None)
            else None
        ),
        output_token_cap=plugin.cost_output_token_cap,
    )
    return delta


# The ctx slots ``_compute_cost_for_plugin`` reads — shared by all three cost
# plugins' ``reads`` metadata. ``writes`` is the three capacity-gate slots.
_COST_PLUGIN_READS: tuple[str, ...] = (
    "layer_ref", "ream_acc", "perm_cache", "layer_input_acc", "freq",
    "protected", "_iter_ream_centroid_ids", "_iter_ream_noncentroid_ids",
    "_iter_n_ream_c", "_iter_n_ream_nc",
)
_COST_PLUGIN_WRITES: tuple[str, ...] = (
    "capacity_util_value", "effective_cost_alignment", "effective_cost_asymmetric",
)


class ReamCostPrePlugin:
    """Live plugin home for the REAM symmetric ``pre`` cost path.

    S2-6 wired ``compute_cost`` into the ``compute_cost`` assignment slot. When
    ``cost_alignment`` resolves to ``"pre"`` the orchestrator registers this
    plugin ahead of the ``LegacyAdapter`` so it wins
    ``PluginRegistry.dispatch_first`` for the slot. The slot body — the
    capacity-util gate + ``_ream_cost_matrix`` call — lives in the shared
    module-level helper ``_compute_cost_for_plugin`` (one source of truth for
    all three cost plugins).
    """

    name = "ream_cost_pre"
    paper = "REAM symmetric pre-alignment cost matrix builder."
    config_key = "stage2_reap_ream.cost_alignment"
    reads: tuple[str, ...] = _COST_PLUGIN_READS
    writes: tuple[str, ...] = _COST_PLUGIN_WRITES
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        cov_acc,
        max_group_cap: int,
        capacity_util_threshold: float,
        cost_alignment_cfg: str,
        cost_asymmetric: bool,
        cost_whitening: str,
        cost_topk_filter: int,
        cost_output_token_cap: int,
    ) -> None:
        # Store every knob the shared compute_cost body reads. NO logic — a
        # faithful mirror of the matching subset of LegacyAdapter.__init__.
        self.cov_acc = cov_acc
        self.max_group_cap = max_group_cap
        self.capacity_util_threshold = capacity_util_threshold
        self.cost_alignment_cfg = cost_alignment_cfg
        self.cost_asymmetric = cost_asymmetric
        self.cost_whitening = cost_whitening
        self.cost_topk_filter = cost_topk_filter
        self.cost_output_token_cap = cost_output_token_cap

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.cost_alignment`` resolves to ``"pre"``.

        ``"pre"`` is also the default, so a missing key / missing
        ``stage2_reap_ream`` block enables this plugin. Case-insensitive to
        match the ``str(...).lower()`` normalization done in
        ``stage2_reap_ream.run()`` (config validation).
        """
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        return str(s2.get("cost_alignment", "pre")).lower() == "pre"

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def compute_cost(self, ctx: PipelineContext) -> Any | None:
        """Slot ``compute_cost`` — capacity-util gate + REAM cost matrix.

        Delegates to the shared ``_compute_cost_for_plugin`` helper (same body
        as the other two cost plugins and the dead ``LegacyAdapter.compute_cost``
        fallback). Returns the cost matrix ``delta``.
        """
        return _compute_cost_for_plugin(self, ctx)
