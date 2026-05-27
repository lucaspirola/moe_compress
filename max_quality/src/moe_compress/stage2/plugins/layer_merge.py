"""Stage 2 per-layer merge spine — REAM Eq. 6 + sequential-profiling orchestration.

This plugin owns the always-on per-layer merge orchestration: the
accumulator construction, the early-exit forward profile, the REAM
Eq. 6 merge (frequency- or saliency-weighted) with Hungarian intermediate-neuron
alignment, the budget-bump feasibility/quality loop, the router
resize, the covariance snapshot, and the artifact write. Specific
cost/solver/refine/distill/heal plugins live as siblings in this
package; this plugin is the merge spine that wires them through the
six live phase hooks.

Paper
-----
Liu et al., "REAM: Routing Expert Activation Merging for MoE
Compression" — arXiv:2604.04356.
audit/spec_compliance/01_papers/2604.04356/source.md.

Equation 6 (frequency-weighted merge):

    W_merged = Σ_i (freq_i / Σ_j freq_j) · P_i(W_i)

where ``P_i`` is the Hungarian neuron-permutation alignment of
``W_i`` to the centroid expert (intermediate-neuron axis), and
``Σ_j freq_j`` sums over the merge-group members only (group
renormalization — see D-ream-aggregation in
:mod:`stage2.plugins.ream_cost` for the renormalization rationale).
Note: the paper's per-expert ``S^freq_i = freq_i / |X|`` (selection
frequency normalized by token count) cancels the ``|X|`` factor
through the ``Σ_j`` denominator, so using raw ``freq_i`` here is
algebraically equivalent — the ratios ``freq_i / Σ_j freq_j`` and
``S^freq_i / Σ_j S^freq_j`` are identical.

Plus the surrounding §3-§4 sequential-profiling pipeline:

- Sequential merging (paper §4 / Fig. 1(b)): after merging layer ℓ,
  activations are recomputed through the merged layer before profiling
  layer ℓ+1. The paper's ablation (§5.4) measures ΔAVG = −1.0 when
  sequential merging is removed.
- Early-exit forward pass (project-pragmatic optimization, paper-
  equivalent): for each layer ``L``, run forward from input embedding
  through layers 0...L only, collecting metrics from layer ``L``'s
  hooks. Layers L+1..N-1 are skipped via an exception-raising hook on
  the next decoder layer. Pure ~2× wall-clock speedup; semantics
  preserved exactly because every layer-L metric depends only on
  hidden states arriving AT layer ``L``.

Official code
-------------
``SamsungSAILMontreal/ream`` @ commit
``84a3030716a0059589e9d10e2ea049e32b76cfa6`` (2026-04-16) —
github.com/SamsungSAILMontreal/ream. The reference implementation
(``ream/ream.py`` lines 60-87 for assignment, plus the broader
sequential pipeline in the same repo) is the basis for this plugin.

Deviation: D5a — per-centroid cap
---------------------------------
**Semantics difference (read carefully).** The paper's ``group_size``
parameter and this plugin's ``max_merge_group_size`` count different
things:

- Official REAM (``ream/ream.py`` L75-82 @
  ``84a3030716a0059589e9d10e2ea049e32b76cfa6``): ``group = [centroid]``
  then ``break when len(group) >= group_size`` — i.e. ``group_size`` is
  the **total** group size **including** the centroid itself.
- This plugin: ``max_merge_group_size`` is a cap on **non-centroids
  only** (the centroid is implicit and not counted).

Equivalence: plugin ``max_merge_group_size = 8`` → total group of 9
(centroid + 8 non-centroids); paper ``group_size = 16`` → 15
non-centroids (in plugin semantics); paper ``group_size = 32`` → 31
non-centroids.

REAM §4 / experiments use ``group_size = 16`` at 25 % reduction
(Qwen3-30B-A3B, 96 centroids on 128 experts) and ``group_size = 32``
at 50 % reduction (64 centroids on 128 experts). In plugin semantics
that is ~15 and ~31 non-centroids respectively. This plugin uses
``max_merge_group_size = 8`` (configurable) — so the cap is roughly
**half as permissive** as the paper's 25 %-reduction recipe (8 vs
~15 in plugin semantics).

Rationale: smaller groups reduce destructive averaging on long-tail
experts. At the project's floor budget (256 → 128, ~30 % reduction
on a 256-expert pool), the per-centroid average absorption is 1.0
non-centroid, so ``C = 8`` provides 8× headroom above the average
while still bounding any single survivor's merge breadth. The
budget-bump loop (D-ream-budget-bump) catches feasibility violations
and bumps ``effective_target`` upward as needed, so no expert weights
are silently dropped when the tighter cap would otherwise be
infeasible.

Deviation: D5b — cost matrix for neuron permutation alignment
-------------------------------------------------------------
REAM §4 (L399-415 of ``source.md``) explicitly defines the Hungarian
cost matrix as ``C = C_act + C_wt`` — a **raw sum** of the activation
and weight components, with ``[C_wt]_pq = ‖W^p_{c_i} − W^q_j‖_2`` (a
single ``W`` per expert, the paper does not specify which projection
of an SwiGLU expert is meant).

This plugin's deviation from the paper is twofold:

1. **Normalization.** This plugin min-max normalizes both ``C_wt`` and
   ``C_act`` to ``[0, 1]`` independently **before** summing, rather
   than summing the raw distances as written in the paper. Rationale:
   the two components live on very different numerical scales (weight
   Frobenius distances vs activation L2 distances on normalized
   rows); raw summation would let whichever component has the larger
   dynamic range dominate the Hungarian assignment. Min-max
   normalization equalizes their influence.

2. **SwiGLU gate+up aggregation.** The paper writes a single ``W`` per
   expert and does not say which projection of an SwiGLU expert is
   used. This plugin combines gate and up via separate Frobenius
   distances summed (L1-of-Frobenius), not via a block-Frobenius over
   ``[W_gate, W_up]``:

   ``C_wt[p, q] = ‖W_gate^p − W_gate^q‖_F + ‖W_up^p − W_up^q‖_F``

   This is **not** the block-Frobenius
   ``‖[W_gate^p, W_up^p] − [W_gate^q, W_up^q]‖_F``.

Components (each min-max normalized to ``[0, 1]`` before summing):

- ``C_wt``: gate+up Frobenius weight distance, as defined above.
- ``C_act``: per-neuron mean-activation L2 distance. Gate-output rows
  are L2-normalized per-row before computing pairwise Euclidean
  distances (equivalent to cosine distance on the normalized rows).

Rationale: ``C_wt`` captures structural similarity; ``C_act``
captures functional importance; normalization keeps both components
comparable. (Ablation of cost matrix choice — ``C_wt`` only vs
``C_wt + C_act`` vs ``C_act`` only, raw vs normalized — is a TODO
pending Stage 6 evals.)

Deviation: D-ream-budget-bump — feasibility / quality gates
-----------------------------------------------------------
REAM does not describe a feasibility-bump loop or cost-threshold
quality gate. Two project-original gates raise the per-layer effective
centroid count:

  (1) **Feasibility gate**: if
      ``N'_l × max_merge_group_size < N_l − N'_l`` (the per-centroid
      cap, which counts non-centroids only, cannot absorb every
      non-centroid), bump ``effective_target`` by
      ``max(1, ceil(effective_target × cost_bump_ratio))`` and retry.
      Falls back to zero-merge if ``effective_target`` reaches
      ``n_experts`` without feasibility — guaranteeing no expert
      weights are silently dropped.
  (2) **Quality gate**: if mean assigned cost exceeds
      ``running_mean × (1 + ream_cost_sigma_threshold)`` with
      ``ream_cost_sigma_threshold = 1.5`` (mean-relative multiplier;
      inactive for the first 4 layers that contribute valid mean-cost
      samples while the running mean stabilizes), bump target.

**Quality-gate exhaustion (last-resort apply-anyway):** distinct from
the feasibility-fallback above, if the quality gate is still active
when ``effective_target = n_experts``, the plugin applies the most
recent above-threshold assignment instead of zero-merging — rationale:
quality-gate failures often coincide with naturally high-cost layers
where any single-cap-respecting assignment is the best available, and
zero-merging would unnecessarily cost compression.

**Orphan-singleton promotion:** if the capped greedy assignment leaves
any non-centroid unassigned (rare edge case where every centroid's cap
is saturated by lower-cost candidates), the orphan is promoted to a
singleton centroid for that layer (no merge) — defensive safety net;
the feasibility check is designed to prevent this.

Blacklisted-expert exclusion
----------------------------
Before any REAP/REAM computation, super experts (SEs) are excluded
from the routed expert pool — they are not candidates for the
centroid set and not candidates for the non-centroid set; their
weights pass through Stage 2 unchanged. SEs are identified by the
``(layer, expert)`` pairs in ``stage1_blacklist.json``. Placing an SE
in the centroid set would allow non-centroid weights to be merged
into it, modifying the SE's weights — defeating the purpose of
blacklisting.

Shared experts (``mlp.shared_expert``) are never in scope: they live
in a separate model attribute, are not indexed as routed experts, and
are never iterated by ``iter_moe_layers``. No explicit exclusion logic
needed.

All counts here (``N'_l``, feasibility checks, group sizes) refer to
**non-SE routed experts only**.

Router resize (Step 5)
----------------------
Remove merged non-centroid experts' rows from ``gate.weight``. Update
``num_experts`` on the MoE block. SE rows are **not removed** — they
remain in the router and expert list unchanged.

Covariance side-collection (also used by Stage 3 / 4)
-----------------------------------------------------
During the profiling forward pass, two covariance matrices are
accumulated per ``(layer, expert)``:

- ``A_gate_up`` (``gate_proj``): input covariance for ``gate_proj``
  and ``up_proj`` (shared tensor).
- ``A_down`` (``down_proj``): input covariance for ``down_proj``
  (intermediate activations).

Stored in ``_stage2_input_covariance.pt`` (fp16 persisted dtype per
D-cov-storage-fp16 — SHARED with Stage 3; consumed there at
:mod:`stage3.plugins.covariance_collection` / ``swift_svd_alpha``).
Eigendecomposition still runs in fp64 in-memory in Stage 3, so
numerical conditioning is preserved.

Resume
------
Per-layer atomic checkpointing to ``_stage2_partial/`` (see project
§11 for the ``.tmp + os.replace`` idiom and ``.pt``-before-``.json``
ordering invariant):

- ``merge_{layer_idx}.json``: centroid IDs, groupings, frequencies,
  merge map.
- ``layer_{layer_idx}.pt``: covariance snapshot for this layer.

On resume, completed layers are replayed from partial files (fast, no
forward pass). The model must be passed in pre-merge state (Stage 1
output) — a guard checks ``num_routed_experts`` matches the pre-merge
count.

**Critical invariant**: covariance remapping
(``_remap_covariance_for_layer``) must happen BEFORE the snapshot.
Snapshotting before remapping persists pre-merge expert keys,
corrupting Stage 3 inputs on resume.

Resume schema v2: ``_stage2_partial/merge_{layer_idx}.json`` bumped
from ``format_version: 1`` to ``format_version: 2`` to carry the new
forensic / resume fields: ``assignment_solver_used``,
``cost_alignment_used``, ``em_rounds_completed``, ``distill_state``
(per merged-group dict). **No backward-compat shim** — operators
upgrading mid-pipeline must finish a stage on one version or restart
cleanly (per project §11 strict version match).

Sequential profiling with early-exit (project-pragmatic optimization)
---------------------------------------------------------------------
The REAM paper §4 introduces *sequential merging* as a core
contribution: after merging layer ℓ, activations are recomputed
through the merged layer before profiling layer ℓ+1, ensuring each
layer's REAP scores and REAM cost matrices reflect the actual input
distribution it will see at inference time (not stale pre-merge
statistics). The paper's ablation (§5.4) measures ΔAVG = −1.0 when
sequential merging is removed.

Implementation: for each layer ``L`` (processed in order 0→39), the
profiling forward pass runs from the input embedding through layers
0...L, collecting REAP/REAM/covariance data from layer L's hooks.
Layers L+1...39 are **not executed** — their computation is pure
waste because all metrics collected for layer ``L`` (REAP scores,
δ_gate, δ̃_expert, input covariance) depend only on the hidden states
that *arrive at* layer ``L``, not on what happens after it. An
**early-exit forward hook** is registered on decoder layer ``L+1``'s
input forward hook (i.e. it fires as ``L+1`` begins to consume the
hidden state ``L`` just produced, before any ``L+1`` compute runs);
the hook raises a sentinel exception that aborts the forward pass
cleanly. The profiling runs under ``torch.no_grad()``, so no autograd
graph is corrupted.

This gives a ~2× wall-clock speedup over the naïve approach (running
all 40 layers for each of the 40 profiling passes): the total
layer-forward count drops from 40 × 40 = 1600 to
1 + 2 + 3 + ... + 40 = 820. The REAM paper's sequential merging
semantics are preserved exactly — each layer is profiled on hidden
states that reflect all prior merges.

Six live phase hooks (S2-12a)
-----------------------------
S2-12a relocates the SIX live phase hooks of the retired
``LegacyAdapter`` (``on_layer_setup`` / ``on_profile`` / ``merge`` /
``post_merge`` / ``write_artifacts`` / ``on_layer_teardown``) into
this one always-on plugin. Each hook is a verbatim slice of the
legacy loop body, with long explanatory comments preserved (the
original lines are the load-bearing documentation of the accumulator
/ merge / artifact semantics).

The dead ``dispatch_first``-slot fallbacks (``compute_cost`` /
``apply_cost_mask`` / ``solve_assignment`` / ``refine_assignment`` /
``pre_merge_snapshot``) are NOT relocated — they stay behind on the
(now 100 %-dead) ``LegacyAdapter`` until S2-12b deletes that file.

Naming-history note
-------------------
"Step 1-5" labels (project §5) and "Phase F" (legacy stage-1 monolith
terminology occasionally reused in stage-2 logs) are
naming-historical. The current plugin architecture has no
step-numbering taxonomy; new prose drops the labels. Existing log
lines / Trackio keys preserved for dashboard back-compat.

Run-scope mutable scratchpad (``cov_acc``, ``merge_map``,
``_layer_mean_costs``, ``partial_dir``) lives as instance attributes on this
plugin. The plugin is constructed once per ``run()`` invocation, so the
per-plugin scratchpad is single-run-scoped with no concurrency hazard.
Per-layer scratchpad lives on the per-layer :class:`PipelineContext` (a
``child()`` scope), addressed by named slots via ``ctx.get`` / ``ctx.set``.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch

from ...utils.activation_hooks import ReamCostAccumulator
from ...utils.model_io import build_banks
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext
from ..merging import _merge_experts_inplace, _resize_router_for_kept_experts
from ..permutation_align import _PermAlignCache
from ..profiling import _LayerInputAccumulator
from ..shared_io import (
    _remap_covariance_for_layer,
    _snapshot_cov_layer,
    _snapshot_neuron_means_layer,
    _write_heal_weights,
    _write_merge_json,
)

log = logging.getLogger(__name__)


class LayerMergePlugin:
    """Always-on Stage-2 plugin owning the per-layer merge spine."""

    name = "layer_merge"
    paper = (
        "REAM Eq. 6 merge (frequency- or saliency-weighted) + §3-4 "
        "sequential profiling — arXiv:2604.04356 (Liu et al.). "
        "Official code: SamsungSAILMontreal/ream "
        "@ 84a3030716a0059589e9d10e2ea049e32b76cfa6. "
        "Deviations: D5a (max_merge_group_size=8 vs paper C=16/32), "
        "D5b (C = C_wt + C_act for Hungarian neuron-alignment; "
        "paper leaves cost unspecified), D-ream-budget-bump (feasibility + "
        "quality gates around N'_l). Calibration: D11 + D-cal-size "
        "(see :mod:`stage2.plugins.reap_scoring`); covariance: "
        "D-cov-storage-fp16 (shared with Stage 3). See module docstring."
    )
    config_key = "stage2_reap_ream"
    # reads / writes carried forward from LegacyAdapter, trimmed to exactly
    # the ctx slots the SIX live hooks touch (S2-12a). ``provides`` is empty.
    reads: tuple[str, ...] = (
        "layer_ref", "reap_acc", "ream_acc", "layer_input_acc", "perm_cache",
        "target", "freq", "scores", "grouped", "protected",
        "ream_centroid_ids", "final_kept_ids",
        "heal_state", "distill_state", "n_experts", "n_protected",
        "assigned_cost", "n_assigned", "c_fail", "em_rounds_done",
        "effective_cost_alignment", "effective_cost_asymmetric",
        "capacity_util_value", "effective_target", "mean_assigned_cost",
    )
    writes: tuple[str, ...] = (
        "ream_acc", "perm_cache", "layer_input_acc",
        "distill_state", "final_kept_ids", "heal_state", "reap_acc",
    )
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always on; the per-layer merge spine runs on every Stage-2 run."""
        return True

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def __init__(
        self,
        *,
        s2_cfg: dict[str, Any],
        heal_cfg,
        batches,
        model,
        cov_acc,
        merge_map: dict[int, dict[int, list[int]]],
        layer_mean_costs: list[float],
        partial_dir: Path | None,
        max_group_cap: int,
        cost_sigma: float,
        cost_bump_ratio: float,
        min_active_tokens: int,
        assignment_solver: str,
        cost_alignment_cfg: str,
        cost_output_token_cap: int,
        cost_asymmetric: bool,
        expert_distill_steps: int,
        expert_distill_token_cap: int,
        blacklist: dict[int, list[int]],
        device,
    ) -> None:
        # Store every knob the SIX live hooks read off ``self`` PLUS the eight
        # attributes ``orchestrator._run_assignment`` reads off this plugin
        # instance (``_layer_mean_costs`` / ``blacklist`` / ``cost_alignment_cfg``
        # / ``cost_asymmetric`` / ``min_active_tokens`` / ``max_group_cap`` /
        # ``cost_sigma`` / ``cost_bump_ratio``). NO logic in __init__ — a
        # faithful re-host of the original local variables. Knobs only the dead
        # ``LegacyAdapter`` fallbacks read are NOT carried over.
        self.s2 = s2_cfg
        self.heal_cfg = heal_cfg
        self.batches = batches
        self.model = model
        # Run-scope mutable scratchpad (was held in run()'s local frame).
        # Held here on the plugin instance; in-place mutations on these
        # references are visible to run() after the per-layer loop exits.
        self.cov_acc = cov_acc
        self.merge_map = merge_map
        self._layer_mean_costs = layer_mean_costs
        self.partial_dir = partial_dir
        # Parsed flag knobs (see stage2_reap_ream.run for the parsing logic).
        self.max_group_cap = max_group_cap
        self.cost_sigma = cost_sigma
        self.cost_bump_ratio = cost_bump_ratio
        self.min_active_tokens = min_active_tokens
        self.assignment_solver = assignment_solver
        self.cost_alignment_cfg = cost_alignment_cfg
        self.cost_output_token_cap = cost_output_token_cap
        self.cost_asymmetric = cost_asymmetric
        self.expert_distill_steps = expert_distill_steps
        self.expert_distill_token_cap = expert_distill_token_cap
        self.blacklist = blacklist
        self.device = device

    # ------------------------------------------------------------------
    # Phase 1: on_layer_setup
    # ------------------------------------------------------------------
    def on_layer_setup(self, ctx: PipelineContext) -> None:
        """Build per-layer accumulators + perm cache + (optional) layer-input acc.

        Verbatim slice of lines 695–727 of stage2_reap_ream.run() (pre-T6).
        """
        layer_ref = ctx.get("layer_ref")
        # ctx.reap_acc is created earlier in this phase by ReapScoringPlugin
        # (registered first in stage2_reap_ream.py); we only construct the
        # REAM/perm caches and (optionally) the layer-input accumulator here.
        ream_acc = ReamCostAccumulator()  # fresh accumulator per layer; discarded after this layer's pass
        # Stage 2 v2 (M1): cache (perm, residual) per (layer, centroid, noncentroid)
        # so the cost-matrix builder and merge step share Hungarian alignments.
        # Cleared at the start of every layer.
        perm_cache = _PermAlignCache()
        # Phase 3 (M8): capture layer-input hidden states only when
        # per-expert distillation is enabled, to keep host-RAM cost zero
        # for runs that don't use the feature.
        # Direction C: the output-space cost (cost_alignment == "output") also
        # needs the layer-input calibration tokens, so the accumulator is
        # likewise enabled in that mode. When BOTH are active the buffer must
        # be large enough for the larger consumer. The accumulator stays None
        # (no capture, no host-RAM cost) for every "pre"/"post" run — keeping
        # those paths byte-identical to main.
        _need_layer_inputs = self.expert_distill_steps > 0 or self.cost_alignment_cfg == "output"
        _layer_input_cap = (
            max(
                self.expert_distill_token_cap if self.expert_distill_steps > 0 else 0,
                self.cost_output_token_cap if self.cost_alignment_cfg == "output" else 0,
            )
            if _need_layer_inputs
            else 0
        )
        layer_input_acc = (
            _LayerInputAccumulator(
                max_samples=_layer_input_cap,
                seed=layer_ref.layer_idx,  # per-layer seed for bit-reproducibility
            )
            if _need_layer_inputs
            else None
        )
        torch.cuda.empty_cache()
        ctx.set("ream_acc", ream_acc)
        ctx.set("perm_cache", perm_cache)
        ctx.set("layer_input_acc", layer_input_acc)

    # ------------------------------------------------------------------
    # Phase 2: on_profile
    # ------------------------------------------------------------------
    def on_profile(self, ctx: PipelineContext) -> None:
        """Forward-pass profile: reap + cov + ream accumulators populated.

        Verbatim slice of lines 728–737 of stage2_reap_ream.run() (pre-T6).

        Plugin #12 REDO (Optimization A): on a full-hit from
        :class:`Stage2ProfileCacheProvider`, the per-layer ream_acc /
        cov_acc / layer_input_acc have already been hydrated from the
        sidecar in ``on_layer_setup``. We early-return BEFORE the live
        forward pass and BEFORE ``cov_acc.finalize_layer`` — both are in
        this method body and both must be skipped. ``cov_acc`` is already
        finalized inside the payload (writer-side serialization order in
        plan §10).
        """
        # Pattern A — cache-aware skip. Must be at the top so the live
        # forward AND finalize_layer are both elided on full hit. The slot
        # is only set (to True) on a full hit by
        # :meth:`Stage2ProfileCacheProvider.on_layer_setup`; ``ctx.has``
        # guards against the cache-disabled case (no provider, slot never
        # written) and the partial-hit case (slot set to True only on full).
        if ctx.has("stage2_profile_full_hit") and ctx.get("stage2_profile_full_hit"):
            return
        # Look up ``_profile_layer`` via the stage2.orchestrator namespace so
        # existing tests (e.g. test_smoke_stage2_resume.py) that
        # ``monkeypatch.setattr(stage2.orchestrator, "_profile_layer", ...)``
        # still take effect through the pipeline path. The plain
        # module-level import would bind the symbol at import time and
        # bypass the monkey-patch.
        from .. import orchestrator as _srr
        layer_ref = ctx.get("layer_ref")
        _srr._profile_layer(
            self.model, layer_ref, self.batches,
            ctx.get("reap_acc"), self.cov_acc, ctx.get("ream_acc"),
            device=self.device,
            layer_input_acc=ctx.get("layer_input_acc"),
        )
        # cov_acc.finalize_layer is independent of reap finalization (which
        # has moved to ReapScoringPlugin.on_score, the very next phase) and
        # could be parallelised (e.g., via concurrent.futures) if profiling
        # shows this is a bottleneck in future.
        self.cov_acc.finalize_layer(layer_ref.layer_idx)

    # ------------------------------------------------------------------
    # Phase 6: merge
    # ------------------------------------------------------------------
    def merge(self, ctx: PipelineContext) -> None:
        """Merge experts in place (trimmed S2-11).

        Verbatim slice of the ``_merge_experts_inplace`` call from
        stage2_reap_ream.run() (pre-T6). The per-merge-group distillation block
        MOVED OUT to ``ExpertDistillPlugin.merge`` as of S2-11 (registered after
        this adapter, so its ``merge`` hook runs after ``_merge_experts_inplace``
        and before ``bank.select``). This sets ``distill_state=None`` only as a
        DEFAULT — ``ExpertDistillPlugin.merge`` overwrites it when distillation
        is enabled, and the default prevents a ``KeyError`` in
        ``write_artifacts`` / ``on_layer_teardown`` when distill is disabled
        (``ExpertDistillPlugin`` is dropped by ``registry.enabled``).
        """
        layer_ref = ctx.get("layer_ref")
        grouped = ctx.get("grouped")
        freq = ctx.get("freq")
        scores = ctx.get("scores")
        ream_acc = ctx.get("ream_acc")
        perm_cache = ctx.get("perm_cache")

        _merge_experts_inplace(
            layer_ref, grouped, freq,
            freq_weighted=self.s2["ream"]["frequency_weighted_merge"],
            scores=scores,
            ream_acc=ream_acc,
            perm_cache=perm_cache,
        )

        ctx.set("distill_state", None)

    # ------------------------------------------------------------------
    # Phase 7: post_merge
    # ------------------------------------------------------------------
    def post_merge(self, ctx: PipelineContext) -> None:
        """bank.select + router resize (trimmed S2-11).

        Verbatim slice of the ``final_kept_ids`` / ``bank.select`` /
        ``_resize_router_for_kept_experts`` block from stage2_reap_ream.run()
        (pre-T6). The ``_heal_layer`` merge-heal block MOVED OUT to
        ``MergeHealPlugin.post_merge`` as of S2-11 (registered after this
        adapter, so its ``post_merge`` hook runs after ``bank.select`` + the
        router resize). This sets ``heal_state=None`` only as a DEFAULT —
        ``MergeHealPlugin.post_merge`` overwrites it when healing is enabled,
        and the default prevents a ``KeyError`` in ``write_artifacts`` when
        merge-heal is disabled (``MergeHealPlugin`` is dropped by
        ``registry.enabled``).
        """
        layer_ref = ctx.get("layer_ref")
        protected = list(ctx.get("protected"))
        ream_centroid_ids = list(ctx.get("ream_centroid_ids"))

        # Final kept set = protected experts (untouched) + REAM centroids (post-merge).
        # Protected experts' rows are preserved in gate.weight and expert tensors.
        final_kept_ids = sorted(list(protected) + ream_centroid_ids)

        if not final_kept_ids:
            raise RuntimeError(
                f"Layer {layer_ref.layer_idx}: final_kept_ids is empty after merge — "
                "target may be inconsistent with protected/blacklisted expert counts"
            )

        banks = build_banks(layer_ref)
        for bank in banks.values():
            bank.select(final_kept_ids)
        _resize_router_for_kept_experts(layer_ref, final_kept_ids)

        ctx.set("final_kept_ids", tuple(final_kept_ids))
        ctx.set("heal_state", None)

    # ------------------------------------------------------------------
    # Phase 8: write_artifacts
    # ------------------------------------------------------------------
    def write_artifacts(self, ctx: PipelineContext) -> dict[str, Any]:
        """Mutate run-scope merge_map; cov remap; write partial JSON + .pt.

        Verbatim slice of lines 1327–1409 of stage2_reap_ream.run() (pre-T6).
        ``partial_dir`` is read from the per-layer context slot
        (``ctx.get("partial_dir")``, set on the run-scope context by the
        orchestrator and inherited by the layer child); it is ``None`` in
        no-resume mode.
        """
        from .merge_heal import _summarize_distill_state

        partial_dir = ctx.get("partial_dir")
        layer_ref = ctx.get("layer_ref")
        grouped = ctx.get("grouped")
        freq = ctx.get("freq")
        final_kept_ids = list(ctx.get("final_kept_ids"))
        ream_centroid_ids = list(ctx.get("ream_centroid_ids"))
        ream_acc = ctx.get("ream_acc")
        merge_map = self.merge_map
        cov_acc = self.cov_acc
        heal_state = ctx.get("heal_state")
        distill_state = ctx.get("distill_state")
        # Read bump-loop outputs from the per-layer context slots.
        n_experts = ctx.get("n_experts")
        n_protected = ctx.get("n_protected")
        assigned_cost = ctx.get("assigned_cost")
        n_assigned = ctx.get("n_assigned")
        c_fail = ctx.get("c_fail")
        em_rounds_done = ctx.get("em_rounds_done")
        effective_cost_alignment = ctx.get("effective_cost_alignment")
        effective_cost_asymmetric = ctx.get("effective_cost_asymmetric")
        capacity_util_value = ctx.get("capacity_util_value")
        effective_target = ctx.get("effective_target")
        _mean_assigned_cost = ctx.get("mean_assigned_cost")
        mean_assigned_cost = _mean_assigned_cost if _mean_assigned_cost is not None else 0.0

        # Correctness depends on the RuntimeError guard above ensuring no protected expert
        # appears in grouped. Without that guard, the else-branch would silently emit [eid]
        # instead of the full merge group for a protected expert that was also a centroid.
        merge_map[layer_ref.layer_idx] = {
            new_idx: (sorted(grouped[eid]) if eid in grouped else [eid])
            for new_idx, eid in enumerate(final_kept_ids)
        }
        # Ordering critical: remap to post-merge indices BEFORE snapshotting.
        # Writing pre-remap covariance would silently corrupt the resume path.
        _remap_covariance_for_layer(cov_acc, layer_ref.layer_idx, final_kept_ids)

        if partial_dir is not None:
            _snapshot_cov_layer(cov_acc, layer_ref.layer_idx, partial_dir)
            # B-iter5-M-2: persist per-expert neuron means BEFORE the merge JSON
            # so that .pt-before-.json ordering invariant (spec §11) holds for
            # the new artifact too. Resume detects missing means by file absence.
            _snapshot_neuron_means_layer(ream_acc, layer_ref.layer_idx, partial_dir)
            # Merge-heal: healed weights are not reconstructible from
            # merge_*.json, so persist them in their own .pt — written BEFORE
            # _write_merge_json so the .pt-before-.json resume invariant holds.
            if self.heal_cfg.enabled and heal_state is not None:
                _write_heal_weights(
                    partial_dir, layer_ref, final_kept_ids,
                    accepted=bool(heal_state["accepted"]),
                )
            _write_merge_json(
                partial_dir, layer_ref.layer_idx, final_kept_ids, grouped, freq,
                merge_map[layer_ref.layer_idx],
                mean_cost_per_pair=(
                    mean_assigned_cost
                    if n_assigned > 0 and mean_assigned_cost > 0.0 and not (c_fail and effective_target >= n_experts)
                    else None
                ),
                assignment_solver_used=self.assignment_solver,
                cost_alignment_used=self.cost_alignment_cfg,
                em_rounds_completed=em_rounds_done,
                distill_state=(
                    {str(k): v for k, v in distill_state.items()}
                    if distill_state is not None
                    else None
                ),
                heal_state=heal_state,
            )

        max_group = max((len(g) for g in grouped.values()), default=0)
        n_noncentroid_members = sum(len(g) - 1 for g in grouped.values())
        mean_group = n_noncentroid_members / len(grouped) if grouped else 0.0
        log.info(
            "  kept %d / %d experts (protected=%d, ream_centroids=%d) — "
            "Σ cost=%.4f, max_group=%d, mean_group=%.2f",
            len(final_kept_ids), n_experts, n_protected, len(ream_centroid_ids),
            assigned_cost, max_group, mean_group,
        )
        _trackio_log({
            # v1 keys — kept verbatim for backward-compatibility with
            # existing Trackio dashboards. Do not rename or remove.
            "stage2/layer_idx": layer_ref.layer_idx,
            "stage2/protected_experts": n_protected,
            "stage2/ream_centroids": len(ream_centroid_ids),
            "stage2/total_experts": n_experts,
            "stage2/sum_assignment_cost": assigned_cost,
            "stage2/mean_cost_per_pair": mean_assigned_cost if n_assigned > 0 else float("nan"),
            "stage2/max_merge_group_size": max_group,
            "stage2/mean_merge_group_size": mean_group,
            "stage2/effective_target": effective_target,
            "stage2/actual_kept_experts": len(final_kept_ids),
            "stage2/stage1_target": ctx.get("target"),
            # v2 keys (spec § 5 / § 6) — per-layer runtime state from the
            # new dispatcher / capacity gate / EM / distillation paths.
            "stage2/assignment_solver_used": self.assignment_solver,
            "stage2/cost_alignment_effective": effective_cost_alignment,
            "stage2/cost_asymmetric_effective": effective_cost_asymmetric,
            "stage2/capacity_util": capacity_util_value,
            "stage2/capacity_regime": (
                "tight" if effective_cost_alignment == "post" else "slack"
            ),
            "stage2/em_rounds_done": em_rounds_done,
            # Distillation aggregates: keys appear only on layers where
            # distillation actually ran (non-empty distill_state). The
            # **{} no-op keeps the emit slim on disabled / singleton-only
            # layers, avoiding dashboard noise.
            **_summarize_distill_state(distill_state),
        })
        return {}

    # ------------------------------------------------------------------
    # Phase 9: on_layer_teardown
    # ------------------------------------------------------------------
    def on_layer_teardown(self, ctx: PipelineContext) -> None:
        """Drop per-layer accumulators + force CUDA cache empty.

        Verbatim slice of lines 1411–1428 of stage2_reap_ream.run() (pre-T6).
        """
        # End-of-layer cleanup: drop Python refs to the per-layer accumulators
        # and force the CUDA caching allocator to release unreferenced blocks
        # back to the driver. Two prior segfaults inside CUDA kernels (silu at
        # layer ~34, layer 7 in an earlier run) were traced to allocator
        # fragmentation that accumulated over the long Stage 2 pass: even with
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments, freed-but-cached blocks
        # are not returned to the driver, so a future large allocation can
        # still fail mid-kernel. Forcing gc.collect() + empty_cache() at every
        # layer boundary keeps the working set bounded.
        # Null all per-layer slots in place, uniformly and unconditionally.
        # ``overwrite=True`` is an upsert: it works whether the slot was ever
        # set or not and whether its current value is None or not, so no
        # ``ctx.get(...) is not None`` guard is needed. Dropping that guard
        # also removes a KeyError hazard for slots a reduced test harness
        # never set. The teardown tests assert every slot resolves to None.
        ctx.set("reap_acc", None, overwrite=True)
        ctx.set("ream_acc", None, overwrite=True)
        ctx.set("perm_cache", None, overwrite=True)
        ctx.set("layer_input_acc", None, overwrite=True)
        ctx.set("pre_merge_weights", None, overwrite=True)
        ctx.set("distill_state", None, overwrite=True)
        # Plugin #12 REDO: null per-layer cache slots so a future layer
        # cannot see stale full/partial-hit state from the previous layer.
        # The reader writes True on hit; this teardown wipes them between
        # layers regardless of whether the cache was enabled.
        ctx.set("stage2_profile_full_hit", None, overwrite=True)
        ctx.set("stage2_profile_partial_hit", None, overwrite=True)
        # Drop large transient writers / accumulators as well so gc.collect()
        # can reclaim their underlying buffers immediately.
        ctx.set("nemo_writer", None, overwrite=True)
        ctx.set("xd_writer", None, overwrite=True)
        gc.collect()
        torch.cuda.empty_cache()
