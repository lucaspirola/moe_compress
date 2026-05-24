"""REAM symmetric pre-alignment cost matrix builder (Eqs. 5, 7, 8).

Paper
-----
Liu et al., "REAM: Routing Expert Activation Merging for MoE
Compression" — arXiv:2604.04356.
audit/spec_compliance/01_papers/2604.04356/source.md.

This module implements the REAM Step 2 cost matrix (project §5 Step 2)
combining two activation-space similarities into a single greedy-assignment
cost:

- δ_gate(i, j) — REAM Eq. 5. Similarity between pre-softmax router-logit
  profile vectors. Each expert's profile is a vector of length ``|X|``
  (one pre-softmax logit per calibration token). Profiles are
  L2-row-normalized, then pairwise Euclidean distances are computed;
  ``dist2sim`` converts to similarity via ``1 − d / max(d)``.
  ``δ_gate ∈ [0, 1]`` — higher = more similar.

- δ̃_expert(i, j) — REAM Eq. 8. Mean per-token cosine similarity of the
  two experts' full-softmax-gated outputs:
  ``δ̃_expert(i, j) = (1/|X|) · Σ_x sim(σ(x)_i · E_i(x), σ(x)_j · E_j(x))``
  where ``σ(x)_e`` is the FULL UNMASKED softmax weight (not the
  dispatched top-k weight). The raw cosine ∈ [−1, 1] is rescaled to
  [0, 1] via ``(cos + 1) / 2``.

- δ_REAM(i, j) — combination. The working similarity for greedy
  assignment. ``δ_REAM(i, j) = (δ_gate + δ̃_expert) / 2 ∈ [0, 1]``.
  The cost passed to the assignment solver is
  ``cost(i, j) = 1 − δ_REAM(i, j) ∈ [0, 1]``.

Routing-weight notation
-----------------------
- REAM (this paper) uses ``σ(x)_j`` for the **full unmasked softmax**
  weight (strictly positive for every expert on every token).
- REAP (arXiv:2510.13999) uses ``g_j(x)`` for the **masked** top-k
  weight (zero for non-top-k experts).

This plugin implements the REAM-side ``σ(x)_j`` view for the Eq. 8
numerator (full unmasked softmax over all experts with no top-k mask,
no renormalization; confirmed by the reference implementation at
``ream/moe_utils.py`` lines 157-158, 173-174 in the upstream repo).
:mod:`stage2.plugins.reap_scoring` owns the REAP-side ``g_j(x)`` view.

Official code
-------------
``SamsungSAILMontreal/ream`` @ commit
``84a3030716a0059589e9d10e2ea049e32b76cfa6`` (2026-04-16) —
github.com/SamsungSAILMontreal/ream. Cross-checked formulas against
``ream/ream.py`` lines 37-41 (δ_gate), 99-113 (δ̃_expert),
46-53 (δ_REAM aggregation), 60-87 (greedy assignment).

Deviation: D-ream-similarity-rescale
------------------------------------
Paper Eqs. 4-5 use the raw cosine similarity directly. This plugin
rescales both components to [0, 1]:

- δ_gate: L2-row-normalize profiles → pairwise Euclidean distance →
  ``dist2sim`` (= ``1 − d / max(d)``).
- δ̃_expert: raw cosine ∈ [−1, 1] rescaled as ``(cos + 1) / 2`` ∈ [0, 1].

Both transforms are monotone in the underlying cosine, so greedy
ranking (and therefore the centroid→non-centroid assignment) is
preserved. The [0, 1] range is cross-stage-comparable, lets δ_gate and
δ̃_expert be averaged on the same footing (D-ream-aggregation), and
matches the bounded similarity scales used elsewhere (e.g. Stage 3).

For δ_gate specifically: the L2-norm + Euclidean + ``dist2sim`` chain
is a monotone transform of the paper's raw cosine into [0, 1] (since
``dist = √(2 − 2·cos)`` on unit vectors, and ``1 − dist/max(dist)`` is
monotone-decreasing in ``dist``); greedy ranking is preserved.

Deviation: D-ream-aggregation
-----------------------------
Paper Eq. 7 sums the two components as ``δ_REAM = δ_g + δ̃_E``. This
plugin uses the **mean**: ``δ_REAM = (δ_gate + δ̃_expert) / 2``.

Monotone in each component, so the joint greedy ranking matches the
paper's exactly when components agree on the pair; it is a
project-original re-weighting otherwise. The /2 normalization keeps
cost values in [0, 1] for cross-stage diagnostic comparability
(Stage 3 also uses [0, 1] similarities) and lets the cost-threshold
logic (``ream_cost_sigma_threshold``) operate on bounded,
mean-relative quantities.

Deviation: D-ream-sparse-routing
--------------------------------
Paper Eq. 8 is defined over expert outputs ``E_e(x)``; it does not
specify behavior when an expert is not dispatched on token ``x``. This
plugin's convention:

- For jointly-active tokens: compute ``σ(x)_e · E_e(x)`` with the
  full-softmax weight ``σ(x)_e`` and the actual computed expert output
  ``E_e(x)``.
- For non-jointly-active tokens: contribute zero to the numerator
  (expert output not computed under top-k dispatch) while keeping the
  full ``|X|`` in the denominator.
- NaN-handling: if a jointly-active token produces a zero gated-output
  vector for one expert (extremely rare; cosine undefined), the
  per-token cosine is treated as ``0`` (after rescale: ``0.5``,
  neutral) rather than excluded from ``|X|``.

The convention deflates ``δ̃_expert`` in proportion to the
jointly-active fraction, biasing greedy assignment toward expert pairs
that co-fire — desirable, since pairs that rarely co-activate carry
little merge signal.

Deviation: D-ream-resume-fallback
---------------------------------
A separate deviation for the merge step (D5b mandates
``C = C_act + C_wt`` for the Hungarian intermediate-neuron alignment).
Resume from a partial directory created **before** the B-iter5-M-2
fix (which began persisting ``_neuron_means_layer{li}.pt`` per merged
layer) cannot reconstruct ``C_act`` from disk — the implementation
logs ERROR and falls back to weight-only
``C = _safe_norm(C_gate + C_up)``. Merges produced from pre-2026-05-07
partial directories diverge cosmetically from a fresh run on the
affected layers. Documented here because the plugin is the entry point
for the C_act consumer; the actual fallback path lives in the merge
helper :mod:`stage2.merging`. New runs from scratch always persist
neuron-means and resume spec-compliantly.

Output context contract
-----------------------
``_compute_cost_for_plugin`` is the shared body for all three cost
plugins (``pre`` / ``post`` / ``output``); each plugin selects the
appropriate alignment branch via the capacity-utilization gate's
``effective_cost_alignment`` decision on ``ctx``.

Original module-header circular-import note
-------------------------------------------
``stage2_reap_ream`` imports *this* module at load time, so a module-top
``from ...stage2_reap_ream import ...`` here would deadlock the import —
neither branch does that. Both the ``post`` (``ream_cost_post``, T9) and
the ``output`` (``output_space_cost``, T10) branches now import their
cost helper from a cycle-free sibling plugin module via a single-dot
**function-scope import**, kept symmetric. Either could be a module-top
import now, but the function-scope form costs nothing once the module
is cached.

``ReamCostPrePlugin`` is the live plugin home for the ``pre`` cost path.
S2-6 wired its ``compute_cost`` hook into the ``compute_cost`` assignment
slot: when ``cost_alignment`` resolves to ``"pre"`` this plugin is
registered ahead of the ``LegacyAdapter`` and wins
``PluginRegistry.dispatch_first`` for the slot. All three cost plugins
share the verbatim slot body via the module-level helper
``_compute_cost_for_plugin`` below to avoid three drifting copies. S2-10
moved the per-layer capacity-utilization gate out into
``CapacityGatePlugin.select_alignment`` (a separate slot that runs
earlier in the bump iteration); ``_compute_cost_for_plugin`` now just
reads the gate's decision (``effective_cost_alignment`` /
``effective_cost_asymmetric``) back off ``ctx``.
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

    The cost config knobs are read off ``plugin.`` (each cost plugin stores
    them as instance attributes mirroring ``LegacyAdapter.__init__``); the ctx
    reads and the ``_ream_cost_matrix`` call are byte-identical to the legacy
    adapter.

    S2-10: the per-layer capacity-utilization gate moved out into
    ``CapacityGatePlugin.select_alignment``, which the orchestrator dispatches
    earlier in the same bump iteration (before this ``compute_cost`` slot). This
    helper therefore just READS ``effective_cost_alignment`` /
    ``effective_cost_asymmetric`` back off ``ctx`` (already published by the
    gate) and threads them into the ``_ream_cost_matrix`` call exactly as
    before. Reads the ``_iter_*`` scratch slots published by the orchestrator.
    Returns the cost matrix ``delta``.
    """
    layer_ref = ctx.get("layer_ref")
    ream_acc = ctx.get("ream_acc")
    perm_cache = ctx.get("perm_cache")
    layer_input_acc = ctx.get("layer_input_acc")
    cov_acc = plugin.cov_acc
    freq = ctx.get("freq")
    protected = set(ctx.get("protected"))
    ream_centroid_ids = list(ctx.get("_iter_ream_centroid_ids"))
    ream_noncentroid_ids = list(ctx.get("_iter_ream_noncentroid_ids"))

    # Capacity-gate outputs published by CapacityGatePlugin.select_alignment
    # earlier in this bump iteration (S2-10). The gate may have downgraded a
    # post/output-configured run to "pre" on a slack-capacity layer — that is
    # correct and intentional; this slot simply consumes its decision.
    effective_cost_alignment = ctx.get("effective_cost_alignment")
    effective_cost_asymmetric = ctx.get("effective_cost_asymmetric")
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
# plugins' ``reads`` metadata. S2-10: the capacity-gate slots
# ``effective_cost_alignment`` / ``effective_cost_asymmetric`` are now read in
# (published by CapacityGatePlugin.select_alignment earlier in the bump
# iteration); the cost plugins write nothing.
_COST_PLUGIN_READS: tuple[str, ...] = (
    "layer_ref", "ream_acc", "perm_cache", "layer_input_acc", "freq",
    "protected", "_iter_ream_centroid_ids", "_iter_ream_noncentroid_ids",
    "effective_cost_alignment", "effective_cost_asymmetric",
)
_COST_PLUGIN_WRITES: tuple[str, ...] = ()


class ReamCostPrePlugin:
    """Live plugin home for the REAM symmetric ``pre`` cost path.

    S2-6 wired ``compute_cost`` into the ``compute_cost`` assignment slot. When
    ``cost_alignment`` resolves to ``"pre"`` the orchestrator registers this
    plugin ahead of the ``LegacyAdapter`` so it wins
    ``PluginRegistry.dispatch_first`` for the slot. The slot body — the
    ``_ream_cost_matrix`` call — lives in the shared module-level helper
    ``_compute_cost_for_plugin`` (one source of truth for all three cost
    plugins). S2-10 moved the capacity-util gate out into
    ``CapacityGatePlugin.select_alignment``, which runs earlier in the bump
    iteration; this slot just reads the gate's decision back off ``ctx``.
    """

    name = "ream_cost_pre"
    paper = (
        "REAM Eqs. 5/7/8: δ_REAM = (δ_gate + δ̃_expert)/2 — arXiv:2604.04356 "
        "(Liu et al.). Official code: SamsungSAILMontreal/ream @ "
        "84a3030716a0059589e9d10e2ea049e32b76cfa6 "
        "(ream/ream.py L37-41/L99-113/L46-53, moe_utils.py L157-158/L173-174). "
        "Deviations: D-ream-aggregation (mean vs paper's sum, monotone-rank-"
        "preserving), D-ream-similarity-rescale (both components rescaled to "
        "[0,1], monotone), D-ream-sparse-routing (non-jointly-active tokens "
        "contribute 0 numerator + full |X| denominator), D-ream-resume-fallback "
        "(weight-only C on pre-2026-05-07 partials). See module docstring."
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
        """Slot ``compute_cost`` — REAM cost matrix.

        Delegates to the shared ``_compute_cost_for_plugin`` helper (same body
        as the other two cost plugins and the dead ``LegacyAdapter.compute_cost``
        fallback). Reads the capacity-gate decision off ``ctx`` (published by
        ``CapacityGatePlugin.select_alignment`` earlier in the bump iteration).
        Returns the cost matrix ``delta``.
        """
        return _compute_cost_for_plugin(self, ctx)
