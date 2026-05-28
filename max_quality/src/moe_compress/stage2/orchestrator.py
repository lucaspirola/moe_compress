"""Stage 2 — REAP scoring + REAM pseudo-pruning, fused-experts-aware.

Key differences from the pre-refactor version:
  - Weights live in stacked tensors on ``Qwen3_5MoeExperts``; pruning means
    slicing those tensors and the router's ``gate.weight`` rows.
  - Scoring hooks go through :func:`instrument_experts` which monkey-patches
    the fused forward with per-expert callbacks.
  - Input covariance for Stage 3 is collected on two tap points:
      ``"gate_proj"``   → covariance used by gate_proj + up_proj SVD
      ``"down_proj"``   → covariance used by down_proj SVD
    Keys match those used by ``InputCovarianceAccumulator`` (``'gate_proj'``
    covers gate+up projections; ``'down_proj'`` covers the down projection).
    We save these under the (layer, expert, matrix_name) key space that
    Stage 3 consumes.

REAM cost matrix (paper 2604.04356, reference ream/ream.py):
  - δ_gate (Eq. 5): similarity ∈ [0,1] between L2-row-normalized pre-softmax
    gate logit profile vectors — Euclidean distance converted via dist2sim.
  - δ̃_expert (Eq. 8): mean cosine similarity of expert outputs (sparse top-k
    approximation; see `compute_delta_expert` in `activation_hooks.py`),
    rescaled to [0,1] via (cosine+1)/2.
  - δ_REAM = (δ_gate + δ̃_expert) / 2 ∈ [0,1]; cost = 1 − δ_REAM.
  - Grouping: single-pass greedy procedure matching paper §4 exactly (descending
    centroid saliency, absorb up to C nearest unassigned non-centroids per centroid).
    The paper prescribes greedy, not optimal matching; this is spec-compliant.
    Full assignment guaranteed by upfront feasibility check.

Frequency-weighted merge with neuron permutation alignment is preserved.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
    ReapAccumulator,
    _EarlyExitException,
    capture_router_outputs,
    early_exit_after_layer,
    instrument_experts,
    record_reap,
)
from ..utils.activation_shards import (
    HealActivationDataset,
    ShardManifest,
    ShardWriter,
    load_manifest,
)
from ..utils.calibration import (
    build_calibration_tensor,
    iter_batches,
    shared_calibration_cache_dir,
    spec_from_config,
)
from ..utils.model_io import (
    MATRIX_NAMES,
    MoELayerRef,
    build_banks,
    iter_moe_layers,
    load_json_artifact,
    save_compressed_checkpoint,
    save_json_artifact,
)
from ..utils.runtime_monitor import snapshot_telemetry as _rt_snap, update as _rt_update
from ..utils.trackio_log import trackio_log as _trackio_log

# ===========================================================================
# Backward-compatibility re-exports
# ---------------------------------------------------------------------------
# Every Stage 2 algorithm now lives under ``stage2/`` (plugin-architecture
# refactor, Tasks 2-17). These re-exports keep the historical
# ``moe_compress.stage2_reap_ream`` import paths working for external callers
# (run_pipeline.py, run_ablations.py, budget_retune.py, stage4_eora.py) and for
# the test suite, which imports many of these ``_``-prefixed internals directly.
# ``_HealConfig`` is additionally constructed inside ``run()`` itself, so its
# import is load-bearing twice over. Removing any name here is a breaking
# change — see max_quality/docs/stage2_plugin_guide.md.
# ===========================================================================

# Shared IO + crash-resume helpers (stage2.resume / stage2.shared_io).
from .resume import ResumedLayerRecord, discover_completed_layers  # noqa: F401
from .shared_io import (  # noqa: F401
    _HEAL_WEIGHTS_FORMAT_VERSION,
    _durable_rename,
    _load_heal_weights,
    _remap_covariance_for_layer,
    _save_covariance,
    _snapshot_cov_layer,
    _snapshot_neuron_means_layer,
    _write_heal_weights,
    _write_merge_json,
)

# Per-layer profiling harness (stage2.profiling).
from .profiling import _LayerInputAccumulator, _profile_layer  # noqa: F401

# Merge engine + permutation alignment (stage2.permutation_align / merging).
from .permutation_align import (  # noqa: F401
    _PermAlignCache,
    _aligned_whitened_residual,
    _permutation_align_to_centroid,
)
from .merging import (  # noqa: F401
    _merge_experts_inplace,
    _resize_router_for_kept_experts,
)

# Grouping primitives (stage2.grouping).
from .grouping import (  # noqa: F401
    _apply_skip_merge_floor,
    _build_grouped_from_assignment,
    _promote_orphans,
)

# REAM cost matrix + cost variants (stage2.plugins.ream_cost*).
from .plugins.ream_cost import (  # noqa: F401
    _extract_sim_expert_matrix_from_tensor,
    _ream_cost_matrix,
)
from .plugins.ream_cost_post import _post_alignment_cost  # noqa: F401
from .plugins.output_space_cost import (  # noqa: F401
    _output_space_cost,
    _router_routing_weights,
    _swiglu_forward,
    _tentative_merged_weights,
)
from .plugins.capacity_gate import _pick_effective_alignment  # noqa: F401

# Assignment solvers + refinement (stage2.plugins.solver_* / two_opt / em).
from .plugins.solver_dispatch import (  # noqa: F401
    SolverName,
    _assign_children_to_centroids,
)
from .plugins.solver_greedy import _assign_greedy  # noqa: F401
from .plugins.solver_hungarian import _assign_hungarian  # noqa: F401
from .plugins.solver_mcf import _assign_mcf  # noqa: F401
from .plugins.solver_sinkhorn import _assign_sinkhorn  # noqa: F401
from .plugins.solver_auto import _assign_auto  # noqa: F401
from .plugins.two_opt_refine import _two_opt_refine  # noqa: F401
from .plugins.em_refine import (  # noqa: F401
    _em_compute_tentative_weights,
    _em_refine_assignment,
)

# Per-group expert distillation + per-layer merge-heal toolchain.
from .plugins.expert_distill import (  # noqa: F401
    _distill_merged_group,
    _snapshot_pre_merge_layer_experts,
)
from .plugins.merge_heal import (  # noqa: F401
    _HealConfig,
    _capture_mlp_io,
    _heal_layer,
    _heal_lr_at_step,
    _heal_student_moe_output,
    _make_shared_out_fn,
    _summarize_distill_state,
)

log = logging.getLogger(__name__)

# Canonical per-layer phase schedule for Stage 2. Copied byte-for-byte from the
# retired ``Stage2Pipeline.phases`` 9-tuple; ``walk_phases`` drives plugins
# through these phases in phase-major / plugin-minor order (see
# max_quality/docs/stage2_plugin_guide.md).
#
# S2-5: ``compute_assignment`` is no longer a plain ``walk_phases`` phase — the
# bump loop is an explicit multi-pass driver (``_run_assignment``) that the
# per-layer loop calls between the pre-assign and post-assign phase walks. The
# schedule is split into the two halves below; ``_STAGE2_LAYER_PHASES`` stays as
# a derived back-compat constant so external callers / tests that expect the
# full 10-tuple keep working.
#
# ``post_merge`` vs ``on_post_merge``:
#   post_merge    — in-layer reaction to the merge (MergeHealPlugin,
#                   ExpertDistillPlugin observe/repair the merged weight tensor).
#   on_post_merge — inter-layer cache invalidation (S2_SEQ / REAM sequential:
#                   clears cov_acc, ream_acc, layer_input_acc so the next layer's
#                   on_layer_setup → on_profile sees fresh state).
#                   Per SC_STAGE12 §582.
#
# Position B (after write_artifacts, before on_layer_teardown): chosen because
# write_artifacts reads ream_acc._lock (via _snapshot_neuron_means_layer).
# An earlier "Position A" choice (between post_merge and write_artifacts)
# caused AttributeError when sequential_reprofile=True. Per Plugin #10 review.
_STAGE2_PRE_ASSIGN_PHASES: tuple[str, ...] = (
    "on_layer_setup",
    "on_profile",
    "on_score",
)
_STAGE2_POST_ASSIGN_PHASES: tuple[str, ...] = (
    "pre_merge_snapshot",
    "merge",
    "post_merge",
    "write_artifacts",
    "on_post_merge",   # SC_STAGE12 §582 — inter-layer cache invalidation.
                       # Position B (after write_artifacts): write_artifacts
                       # reads ream_acc._lock, so invalidation MUST run after.
                       # See Plugin #10 review (cov_acc clarification).
    "on_layer_teardown",
)
# Derived back-compat constant: the full 10-phase schedule with the compound
# ``compute_assignment`` slot wedged between the two halves. Not walked directly
# anymore (``_run_assignment`` owns that slot) — kept so the canonical-order
# contract test and any external importer still see the historical tuple.
_STAGE2_LAYER_PHASES: tuple[str, ...] = (
    _STAGE2_PRE_ASSIGN_PHASES + ("compute_assignment",) + _STAGE2_POST_ASSIGN_PHASES
)


def _run_assignment(plugins, ctx) -> None:
    """Stage-2 assignment driver — the bump loop, decomposed into four slots.

    Reproduces the body of the retired ``LegacyAdapter.compute_assignment``
    line-for-line, EXCEPT the per-bump cost / mask / solve / refine work is now
    reached via ``PluginRegistry.dispatch_first`` over the fine-grained slots
    ``select_alignment`` / ``compute_cost`` / ``apply_cost_mask`` /
    ``solve_assignment`` / ``refine_assignment``. S2-6..S2-10 wired the real
    capacity-gate / cost / solver / refinement plugins, which win every slot;
    behaviour is byte-identical to the pre-S2-5 monolithic hook.

    ``_run_assignment`` owns the bump-loop control flow, the b_fail / c_fail
    gates, the orphan-promotion grouping, and the final ``ctx.set`` of all
    per-layer output slots — exactly the responsibilities the monolithic
    ``compute_assignment`` carried.
    """
    from ..pipeline.registry import PluginRegistry
    from .plugins.layer_merge import LayerMergePlugin
    from .plugins.reap_scoring import select_centroids_by_reap

    layer_ref = ctx.get("layer_ref")
    reap_acc = ctx.get("reap_acc")
    ream_acc = ctx.get("ream_acc")
    perm_cache = ctx.get("perm_cache")
    layer_input_acc = ctx.get("layer_input_acc")
    target = ctx.get("target")
    # scores / freq are published by ReapScoringPlugin.on_score (T7); read
    # them off the ctx slots rather than re-deriving from reap_acc here.
    scores = ctx.get("scores")
    freq = ctx.get("freq")
    n_experts = ctx.get("n_experts")

    # The LayerMergePlugin instance owns the run-scope scratchpad (blacklist,
    # _layer_mean_costs). Locate it in the plugin list — S2-12a relocated the
    # run-scope state off ``LegacyAdapter`` onto this always-on plugin.
    adapter = next(p for p in plugins if isinstance(p, LayerMergePlugin))
    _layer_mean_costs = adapter._layer_mean_costs

    protected = set(adapter.blacklist.get(layer_ref.layer_idx, []))
    # Protected experts (super experts + shared experts from stage1_blacklist.json)
    # are completely excluded from REAM — not centroids, not non-centroids.
    # Their weights pass through Stage 2 unchanged (spec §5 "Blacklisted Expert Exclusion").
    n_protected = len(protected)
    # Publish ``protected`` on ctx up front so the cost / refine slots can read
    # it during the bump loop. The pre-S2-5 ``compute_assignment`` set this slot
    # only at the very end; here it is set once early and the final output-slot
    # block below leaves it as-is (already a tuple of the same value).
    ctx.set("protected", tuple(sorted(protected)))

    if target > n_experts:
        raise RuntimeError(
            f"Layer {layer_ref.layer_idx}: budget target {target} > n_experts {n_experts}; "
            "budget allocation is inconsistent with layer expert count"
        )
    if target == n_experts:
        log.warning(
            "layer %d: budget target (%d) equals total expert count (%d) — "
            "no merging will occur; check budget configuration.",
            layer_ref.layer_idx, target, n_experts,
        )

    effective_target = target
    ream_centroid_ids: list[int] = []
    ream_noncentroid_ids: list[int] = []
    grouped: dict[int, list[int]] = {}
    delta = np.empty((0, 0))
    assignment: list[int] = []
    running_mean: float = float("nan")
    em_rounds_done: int = 0  # populated by _em_refine_assignment in the bump loop
    # Stage 2 v2: hoist effective_cost_alignment / effective_cost_asymmetric
    # from the bump-loop's "if not b_fail" branch to layer scope so the
    # per-layer Trackio emit at the bottom of the loop sees them whether
    # or not the bump loop's success branch ran (b_fail / zero-merge
    # fallback leaves the defaults as-is, which is the right thing to
    # log: "no cost matrix was actually built for this layer"). Same for
    # capacity_util_value — defaults to 0.0 (uncapped / fully-slack).
    effective_cost_alignment: str = adapter.cost_alignment_cfg
    effective_cost_asymmetric: bool = adapter.cost_asymmetric
    capacity_util_value: float = 0.0
    mean_assigned_cost: float = 0.0
    assigned_cost: float = 0.0
    # Invariant: after the bump loop, assignment is either:
    #   (a) a list of length len(ream_noncentroid_ids) with centroid indices (normal path), or
    #   (b) [] with ream_noncentroid_ids also [] (zero-merge fallback path).
    # (c) c_fail last-resort: assignment holds the last above-threshold assignment
    #     (len == len(ream_noncentroid_ids)); applied as best-available merge below.
    # b_fail / c_fail are initialized here so the post-loop fallback check never raises
    # NameError if the range were somehow empty.
    b_fail: bool = False
    c_fail: bool = False
    _warned_ream_target_zero: bool = False

    _original_ream_target = max(effective_target - n_protected, 0)  # target on first attempt

    # Loop runs (1 + n_experts - target) times: 1 initial attempt plus up to
    # (n_experts - target) bumps, one per additional kept expert.
    for _bump_attempt in range(n_experts - target + 1):
        # F1 fix: reset em_rounds_done per bump iteration so the value
        # persisted in the partial JSON reflects the iteration whose
        # assignment is actually committed (not a stale value from a
        # prior bump iteration).
        em_rounds_done = 0
        # REAM centroid count = total target minus the protected slots.
        ream_target = max(effective_target - n_protected, 0)

        if ream_target == 0:
            if not _warned_ream_target_zero:
                log.warning(
                    "layer %d: ream_target=0 — all %d non-protected experts will be dropped "
                    "(budget fully consumed by %d protected experts); "
                    "check budget configuration.",
                    layer_ref.layer_idx, n_experts - len(protected), len(protected),
                )
                _warned_ream_target_zero = True
            break

        # Select top-ream_target non-protected experts by REAP score (descending).
        # This is the greedy centroid selection order: highest-saliency centroid
        # gets priority in the assignment pass (spec §5 Step 3). The pure
        # helper also emits the under-budget warning when the
        # min_active_tokens filter eliminates candidates.
        ream_centroid_ids = select_centroids_by_reap(
            scores,
            freq,
            ream_target=ream_target,
            min_active_tokens=adapter.min_active_tokens,
            protected=protected,
            layer_idx=layer_ref.layer_idx,
            log=log,
        )

        ream_centroid_set = set(ream_centroid_ids)
        ream_noncentroid_ids = [
            e for e in range(n_experts)
            if e not in protected and e not in ream_centroid_set
        ]

        n_ream_c  = len(ream_centroid_ids)
        n_ream_nc = len(ream_noncentroid_ids)

        # Feasibility check (spec §5 Step 3, reference ream/ream.py L60-62):
        # every non-centroid must be absorbable within the per-centroid cap.
        b_fail = (adapter.max_group_cap > 0) and (n_ream_nc > n_ream_c * adapter.max_group_cap)

        delta = np.empty((0, 0))
        assignment = []
        mean_cost = 0.0
        c_fail = False

        if not b_fail:
            # Publish the per-bump scratch slots the four assignment slots
            # read. Always ``overwrite=True`` — a fresh value every bump
            # iteration, the previous iteration's value is stale.
            ctx.set("_iter_ream_centroid_ids", tuple(ream_centroid_ids), overwrite=True)
            ctx.set("_iter_ream_noncentroid_ids", tuple(ream_noncentroid_ids), overwrite=True)
            ctx.set("_iter_n_ream_c", n_ream_c, overwrite=True)
            ctx.set("_iter_n_ream_nc", n_ream_nc, overwrite=True)
            # Slot 0: select_alignment — the per-layer capacity-utilization
            # gate (CapacityGatePlugin). Runs BEFORE compute_cost; it publishes
            # capacity_util_value / effective_cost_alignment /
            # effective_cost_asymmetric to ctx, which the cost slot reads back.
            _alignment = PluginRegistry.dispatch_first(plugins, "select_alignment", ctx)
            assert _alignment is not None, "select_alignment slot returned None"
            # Slot 1: compute_cost — REAM cost matrix (reads the gate slots).
            delta = PluginRegistry.dispatch_first(plugins, "compute_cost", ctx)
            assert delta is not None, "compute_cost slot returned None"
            # Slot 2: apply_cost_mask — Direction B skip-merge floor. The
            # masker may decline (None) — keep delta unchanged in that case.
            masked = PluginRegistry.dispatch_first(plugins, "apply_cost_mask", ctx, delta)
            if masked is not None:
                delta, _mask_info = masked
            # Slot 3: solve_assignment — child→centroid assignment solver.
            assignment = PluginRegistry.dispatch_first(plugins, "solve_assignment", ctx, delta)
            assert assignment is not None, "solve_assignment slot returned None"
            # Slot 4: refine_assignment — 2-opt local search + EM refinement.
            # S2-9: unlike the single-winner cost / mask / solve slots,
            # refine_assignment is a CHAIN: BOTH refiners may run, in registry
            # order (TwoOptRefinePlugin then EmRefinePlugin). The chain calls
            # EVERY enabled plugin's refine_assignment — there is no
            # dispatch_first early-return — so each refiner threads its result
            # forward to the next. A plugin declining the slot — returning
            # None, e.g. a refiner whose own gate is off this layer — is
            # skipped.
            # em_rounds_done was already reset to 0 at the top of this bump
            # iteration (F1: the persisted JSON reflects the committed bump);
            # the chain below only overwrites it when EmRefinePlugin runs.
            for p in plugins:
                hook = getattr(p, "refine_assignment", None)
                if not callable(hook):
                    continue
                result = hook(ctx, assignment, delta)
                if result is None:
                    continue
                assignment, delta, info = result
                if "em_rounds" in info:
                    em_rounds_done = int(info["em_rounds"])
            # The slots wrote effective_cost_alignment / effective_cost_asymmetric
            # / capacity_util_value back to ctx — pull them back into the loop's
            # layer-scope variables so the post-loop output emit sees them.
            effective_cost_alignment = ctx.get("effective_cost_alignment")
            effective_cost_asymmetric = ctx.get("effective_cost_asymmetric")
            capacity_util_value = ctx.get("capacity_util_value")
            _iter_n_assigned = sum(1 for a in assignment if a >= 0)
            _iter_assigned_cost = (
                sum(float(delta[ch, assignment[ch]])
                    for ch in range(n_ream_nc) if assignment[ch] >= 0)
                if delta.size > 0 else 0.0
            )
            if _iter_n_assigned == 0 and n_ream_nc == 0:
                # No non-centroid experts exist — nothing to merge, cost is
                # genuinely zero.  Skip the c_fail gate entirely: there is no
                # merge to gate on, and inf would cause a spurious bump.
                mean_cost = 0.0
                # c_fail remains False (already set above); do not evaluate gate.
            else:
                # When nothing was assigned despite having non-centroids, use inf
                # rather than 0.0: a zero mean_cost would be a false negative,
                # making an unassigned layer look cheaper than any real merge and
                # preventing the cost-threshold bump from triggering.
                mean_cost = (
                    _iter_assigned_cost / _iter_n_assigned
                    if _iter_n_assigned > 0 else float("inf")
                )
                # Require at least 4 prior-layer samples before applying the cost-sigma
                # gate: fewer samples make the running mean too noisy to be meaningful.
                # Invariant: running_mean is always computed in the same branch as
                # c_fail = True, so running_mean is guaranteed to be set before
                # c_fail can become True. Future refactors must preserve this ordering
                # to avoid referencing running_mean when it is still 0.0 (its default).
                if len(_layer_mean_costs) >= 4:
                    running_mean = float(np.mean(_layer_mean_costs))
                    c_fail = mean_cost > running_mean * (1.0 + adapter.cost_sigma)

        if not b_fail and not c_fail:
            break

        # Spec D-ream-budget-bump: BOTH gates use the same bump formula
        # max(1, ceil(effective_target * cost_bump_ratio)) — applies to
        # feasibility (b_fail) AND quality (c_fail) gates uniformly.
        # Previously the ratio was only applied on c_fail, making
        # b_fail-only iterations bump by exactly 1 (slow convergence).
        bump = max(1, math.ceil(effective_target * adapter.cost_bump_ratio))
        new_effective = min(effective_target + bump, n_experts)
        if b_fail:
            log.warning(
                "  layer %d: infeasible (ream_c=%d × cap=%d < nc=%d) — "
                "bumping target %d→%d",
                layer_ref.layer_idx, n_ream_c, adapter.max_group_cap, n_ream_nc,
                effective_target, new_effective,
            )
        # running_mean is always current here: c_fail=True can only be set inside the
        # cost block (not b_fail path), which assigns running_mean before setting c_fail.
        if c_fail:
            assert not math.isnan(running_mean), (
                "running_mean must be set before c_fail can be True; "
                "check that the c_fail assignment is co-located with the running_mean assignment"
            )
            log.warning(
                "  layer %d: mean_cost=%.4f > threshold=%.4f — bumping target %d→%d",
                layer_ref.layer_idx, mean_cost,
                running_mean * (1.0 + adapter.cost_sigma),
                effective_target, new_effective,
            )
        effective_target = new_effective
        # We break BEFORE computing a new assignment at effective_target==n_experts;
        # the last assignment from the previous iteration is used as the fallback.
        if effective_target >= n_experts:
            break

    # Post-loop: if the loop exited because effective_target >= n_experts but c_fail
    # was still True (cost gate never cleared), the last above-threshold assignment
    # is used as last resort. Warn so this silent state is observable.
    if c_fail and effective_target >= n_experts:
        log.warning(
            "REAM layer %d: bump loop exhausted (c_fail=True, b_fail=%s, effective_target=%d >= n_experts=%d); "
            "applying above-threshold assignment as last resort",
            layer_ref.layer_idx, b_fail, effective_target, n_experts,
        )
    # Fallback: if the bump loop exhausted without achieving feasibility
    # (b_fail still True and no assignment was built), log a WARNING and fall back
    # to keeping all non-protected experts as centroids (zero merges). This is the
    # safest fallback — it produces the least compression but loses no expert weights.

    # Zero-target case: budget fully consumed by protected experts — no REAM
    # centroids or non-centroids should exist and no merges should be produced.
    # The bump loop broke out early, so ream_centroid_ids/ream_noncentroid_ids/
    # assignment may still hold stale values from a previous attempt (or their
    # initial [] defaults). Reset them explicitly so the grouping code below
    # produces an empty grouped dict and all protected experts flow to final_kept_ids.
    if _original_ream_target == 0:
        ream_centroid_ids = []
        ream_noncentroid_ids = []
        assignment = []
        delta = np.empty((0, 0))
        b_fail = False
        c_fail = False

    # When b_fail: assignment is [] (reset at iteration top; b_fail skips _assign_children_to_centroids).
    # When c_fail last-resort (effective_target >= n_experts break): assignment holds the last
    # computed above-threshold result and is intentionally applied in the grouping step below.
    if b_fail and ream_noncentroid_ids:
        log.warning(
            "  layer %d: bump loop exhausted (effective_target=%d == n_experts=%d) "
            "without achieving feasibility — falling back to zero-merge "
            "(all non-protected experts kept as centroids). "
            "No expert weights are lost, but compression target is not met.",
            layer_ref.layer_idx, effective_target, n_experts,
        )
        # Explicitly set ream_centroid_ids to all non-protected experts (zero-merge
        # fallback). We cannot rely on the last bump iteration's ream_centroid_ids
        # because the loop broke before recomputing it with the final effective_target.
        ream_centroid_ids = [
            e for e in range(n_experts) if e not in protected
        ]
        ream_noncentroid_ids = []
        assignment = []
        delta = np.empty((0, 0))

    if not ream_centroid_ids and ream_noncentroid_ids and _original_ream_target > 0:
        log.warning(
            "REAM layer %d: no centroids selected (all non-protected experts may have failed "
            "min_active_tokens or cost gate); promoting all non-protected experts to singleton "
            "centroids (zero-merge fallback).",
            layer_ref.layer_idx,
        )
        ream_centroid_ids = list(ream_noncentroid_ids)
        ream_noncentroid_ids = []
        assignment = []
        delta = np.empty((0, 0))

    # Build REAM merge groups (keyed by REAM centroid only — protected experts
    # are not in grouped and their weights are not touched by _merge_experts_inplace).
    grouped = {c: [c] for c in ream_centroid_ids}
    # Protected experts should never appear as REAM centroids; verify the invariant.
    _protected_centroids = [eid for eid in protected if eid in grouped]
    if _protected_centroids:
        raise RuntimeError(
            f"Layer {layer_ref.layer_idx}: protected expert(s) {_protected_centroids} "
            "appeared as REAM centroids — invariant violated"
        )
    for child_pos, centroid_pos in enumerate(assignment):
        if centroid_pos >= 0:
            grouped[ream_centroid_ids[centroid_pos]].append(
                ream_noncentroid_ids[child_pos]
            )

    _promote_orphans(
        grouped,
        ream_centroid_ids,
        ream_noncentroid_ids,
        assignment,
        layer_idx=layer_ref.layer_idx,
        log=log,
    )
    ream_centroid_ids = sorted(set(ream_centroid_ids))

    assigned_cost = (
        sum(float(delta[ch, assignment[ch]])
            for ch in range(len(ream_noncentroid_ids)) if assignment[ch] >= 0)
        if delta.size > 0 else 0.0
    )
    n_assigned = sum(1 for a in assignment if a >= 0)
    mean_assigned_cost = assigned_cost / max(n_assigned, 1)

    # Guard mirrors the resume-path condition (val > 0.0): exclude zero costs
    # so that layers with all-zero pair costs don't bias the running mean low
    # and suppress the cost-sigma bump gate for subsequent layers.
    # Also exclude last-resort c_fail assignments (bump loop exhausted with
    # effective_target >= n_experts) — those costs would inflate the running mean
    # and progressively suppress the c_fail gate for subsequent layers.
    if n_assigned > 0 and mean_assigned_cost > 0.0 and not (c_fail and effective_target >= n_experts):
        _layer_mean_costs.append(mean_assigned_cost)

    # Surface bump-loop outputs on ctx for downstream phases. ``scores`` and
    # ``freq`` are NOT re-published here: ReapScoringPlugin.on_score already
    # wrote those slots to the same objects (set-once would reject a second
    # write, and the re-assignment was a behavior-preserving no-op anyway).
    # ``protected`` is also NOT re-set here: it was published up front (before
    # the bump loop) so the cost / refine slots could read it; the value is the
    # same tuple, so the early set already satisfies this slot.
    ctx.set("ream_centroid_ids", tuple(ream_centroid_ids))
    ctx.set("ream_noncentroid_ids", tuple(ream_noncentroid_ids))
    ctx.set("assignment", assignment)
    ctx.set("delta", delta)
    ctx.set("grouped", grouped)
    ctx.set("mean_assigned_cost", mean_assigned_cost)
    ctx.set("n_protected", n_protected)
    ctx.set("assigned_cost", assigned_cost)
    ctx.set("n_assigned", n_assigned)
    ctx.set("b_fail", b_fail)
    ctx.set("c_fail", c_fail)
    ctx.set("em_rounds_done", em_rounds_done)
    # effective_cost_alignment / effective_cost_asymmetric / capacity_util_value
    # were written to ctx by the compute_cost slot during the bump loop's
    # success branch (overwrite=True). On the b_fail / zero-merge fallback path
    # the slot never ran, so set them here from the layer-scope defaults — at
    # most one of these two writes happens per layer per slot, so set-once is
    # safe only for the slots the bump loop did not already write. Use
    # overwrite=True uniformly to cover both paths.
    ctx.set("effective_cost_alignment", effective_cost_alignment, overwrite=True)
    ctx.set("effective_cost_asymmetric", effective_cost_asymmetric, overwrite=True)
    ctx.set("capacity_util_value", capacity_util_value, overwrite=True)
    ctx.set("effective_target", effective_target)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    *,
    device=None,
    stage1_budget_path: Path | None = None,
    no_resume: bool = False,
) -> Path:
    s2 = config["stage2_reap_ream"]
    cal = config["calibration"]

    # Stage 2 v2 (spec § 6 / D-asymmetric-freq): cost_asymmetric is valid
    # only under freq-weighted merge — the asymmetric factor freq_m/(freq_c+freq_m)
    # is the per-pair version of the merge weight. Reject the combination
    # at the very top of `run` so misconfigured pipelines fail fast before
    # spending compute on calibration / Stage-1 artifact loading. CRITICAL
    # to keep this BEFORE the `_set_experts_implementation` Blackwell workaround
    # below: that call touches `model.config` and would surface a confusing
    # AttributeError on misconfigured tests / dry-runs that pass a stub model,
    # masking the real ValueError we want users to see.
    if bool(s2.get("cost_asymmetric", False)) and not s2["ream"]["frequency_weighted_merge"]:
        raise ValueError(
            "stage2_reap_ream.cost_asymmetric=True requires "
            "ream.frequency_weighted_merge=True. Asymmetric-cost with "
            "saliency-weighted merge (frequency_weighted_merge=False) is not "
            "implemented in P2 — the analogous factor sal_m/(sal_c+sal_m) would "
            "require threading scores into ream_cost_post.py. Set "
            "cost_asymmetric=false or frequency_weighted_merge=true "
            "(spec § 5 step 4T(c)(iii) / D-asymmetric-freq)."
        )

    # Plugin #9 / S2_MM (MergeMoE arXiv:2510.14436) +
    # Plugin RegMean (Jin et al., ICLR 2023, arXiv:2212.09849): merge_step
    # switches the merge math between legacy freq-weighted (default;
    # byte-identical), the MergeMoE closed-form T₁=Q·P† least-squares
    # solution, and the RegMean closed-form per-Linear least-squares
    # solution W_M = (Σ G_i)⁻¹ Σ G_i W_i. Validated at the top of run() —
    # same fail-fast posture as ``cost_asymmetric`` above — so
    # misconfigured pipelines surface the error before spending compute on
    # calibration loading. The canonical resolved value is re-read further
    # below where it is plumbed into ``LayerMergePlugin``.
    _merge_step_check = str(s2.get("merge_step", "freq_weighted")).lower()
    if _merge_step_check not in ("freq_weighted", "mergemoe", "regmean"):
        raise ValueError(
            f"stage2_reap_ream.merge_step={_merge_step_check!r}; "
            "expected 'freq_weighted', 'mergemoe', or 'regmean'."
        )

    # Plugin #14 audit (HIGH-2): hard mutual-exclusion between Plugin #10's
    # ``sequential_reprofile`` and Plugin #12's ``profile_sidecar.enabled``.
    # REAM §4 warns that pre-collected statistics go stale after a merge.
    # The sidecar serves *pre-merge* state on full hit; the sequential
    # invalidator clears state at on_post_merge. Combining them is NOT a
    # mild race — the invalidator becomes a complete no-op:
    #
    #   1. on_layer_setup: LayerMergePlugin constructs empty accs, then
    #      Stage2ProfileCacheProvider re-hydrates them with **pre-merge stats**
    #      (stale for every layer ℓ ≥ 1, because upstream layers were merged).
    #   2. on_profile: LayerMergePlugin.on_profile early-returns on
    #      ``stage2_profile_full_hit`` — the live forward pass is **skipped**.
    #   3. merge: uses the stale-hydrated cost.
    #   4. on_post_merge: Stage2ReamSequentialPlugin sets accs to None
    #      (invalidator fires).
    #   5. Next layer: the cycle repeats — sidecar re-hydrates with stale
    #      stats AGAIN, defeating the invalidator entirely.
    #
    # Net effect: EVERY layer ℓ ≥ 1 sees stale (pre-upstream-merge) cost,
    # not just "layer N's cost is wrong while N+1 reprofiles correctly".
    # The invalidator's clear-at-on_post_merge is structurally unable to
    # win against the sidecar's re-hydration at the next layer's
    # on_layer_setup. Fail fast at top of run() rather than let the silent
    # corruption land in stage2_merge_map.json.
    _sequential_reprofile_enabled = bool(s2.get("sequential_reprofile", False))
    _profile_sidecar_enabled = bool(
        (s2.get("profile_sidecar") or {}).get("enabled", False)
    )
    if _sequential_reprofile_enabled and _profile_sidecar_enabled:
        raise ValueError(
            "stage2_reap_ream.sequential_reprofile=True AND "
            "stage2_reap_ream.profile_sidecar.enabled=True is invalid; "
            "REAM sequential reprofile invalidates per-layer state after every "
            "merge, but the profile sidecar serves pre-merge state on full hit. "
            "The two are mutually exclusive — set at most one. See Plugin #14 "
            "sidecar audit HIGH-2 + tasks/PLAN_PLUGIN_14_sidecar_audit.md."
        )

    # Blackwell sm_100 workaround: transformers' default MoE forward uses
    # `torch.nn.functional.grouped_mm`, which deadlocks on B200 partway
    # through Stage 2 (reproduced as a 2-min main-thread hang then SIGSEGV
    # at layer 13 batch ~60 on Qwen3.6-35B-A3B, 2026-05-13). Stages 5 and 6
    # already force `batched_mm` via `_set_experts_implementation` — Stage 2
    # was the only path still going through the broken kernel. Mirror the
    # same override here: env var `EXPERTS_IMPLEMENTATION` wins, then the
    # YAML knob `stage2_reap_ream.experts_implementation`, default
    # `batched_mm`. See memory/project_grouped_mm_blackwell.md.
    from ..stage5_router_kd import _set_experts_implementation
    _experts_impl = os.environ.get(
        "EXPERTS_IMPLEMENTATION", s2.get("experts_implementation", "batched_mm")
    )
    _set_experts_implementation(model, _experts_impl)

    if stage1_budget_path is None:
        stage1_budget_path = artifacts_dir / "stage1_budgets.json"
    budgets_payload = load_json_artifact(stage1_budget_path)
    per_layer_target = {
        int(k): int(v) for k, v in budgets_payload["per_layer_target_experts"].items()
    }
    blacklist_payload = load_json_artifact(artifacts_dir / "stage1_blacklist.json")
    blacklist = {int(k): list(v) for k, v in blacklist_payload.get("blacklist", {}).items()}

    spec = spec_from_config(cal, num_sequences_override=s2["num_calibration_samples"])
    calib = build_calibration_tensor(
        tokenizer, spec,
        cache_dir=(os.environ.get("MOE_CALIB_CACHE_DIR") or shared_calibration_cache_dir(artifacts_dir)),
    )
    batches = iter_batches(calib, batch_size=s2["batch_size"])
    assert isinstance(batches, list), "iter_batches must return a list for multi-pass re-iteration"

    moe_layers = list(iter_moe_layers(model))
    cov_acc = InputCovarianceAccumulator()
    # Spec §5 "Covariance Side-Collection": FP32 storage is recommended by
    # Swift-SVD paper 2604.01609 (avoids numerical degradation in eigendecomposition);
    # the dtype is configurable via covariance_storage_dtype.
    # Default fp16 per §12 D-cov-storage-fp16 (10 mantissa bits, half the
    # disk vs fp32, no measurable downstream PPL drift on Qwen3-30B-A3B).
    # Production config also pins this to fp16; the default is here for
    # config-omitted invocations.
    cov_dtype = getattr(torch, s2.get("covariance_storage_dtype", "float16"))
    cov_acc.set_storage_dtype(cov_dtype)
    merge_map: dict[int, dict[int, list[int]]] = {}

    # -----------------------------------------------------------------------
    # Stage-2 per-layer merge-heal setup (opt-in; inert when disabled).
    # When `merge_heal_enabled` is False, heal_cfg.enabled is False and every
    # heal code path below is skipped — Stage 2 behaviour is byte-identical to
    # the pre-feature code (no capture, no heal). When enabled, each layer's
    # (input, target) pairs are captured in-process just before the merge —
    # no teacher object, no sidecar, no cascade buffer.
    # -----------------------------------------------------------------------
    heal_cfg = _HealConfig(s2)
    _heal_device: torch.device | None = None
    # Cross-domain (WikiText) batches — pre-tokenised once and reused across
    # layers when `cross_domain_holdout_enabled`. None when disabled, so the
    # per-layer capture block below short-circuits.
    xd_batches: list | None = None
    if heal_cfg.enabled:
        _heal_device = device
        if _heal_device is None:
            try:
                _heal_device = next(model.parameters()).device
            except StopIteration:
                _heal_device = torch.device("cpu")
        log.info(
            "Stage-2 merge-heal enabled: self-distillation, "
            "pool=%d tokens/layer, train_router=%s, lr=%g",
            heal_cfg.token_cap, heal_cfg.train_router, heal_cfg.lr,
        )
        if heal_cfg.cross_domain_holdout_enabled:
            # Read WikiText corpus identity from the thermometer config so the
            # heal-time cross-domain holdout matches the BPT eval corpus
            # exactly (same dataset / subset / split). Defaults mirror the
            # thermometer defaults; the `_thermo_wikitext_tensor` call is the
            # same tokenisation path the BPT eval uses.
            from ..stage6alt.plugins.thermo_corpus import _thermo_wikitext_tensor
            therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
            wt = therm.get("wikitext", {}) or {}
            wt_dataset = wt.get("dataset", "wikitext")
            wt_subset = wt.get("subset", "wikitext-2-raw-v1")
            wt_split = wt.get("split", "test")
            wt_seq_len = int(therm.get("sequence_length", 2048))
            # Sequences = ceil(xd_holdout_tokens / wt_seq_len) + 4 absolute
            # pad. `_capture_mlp_io`'s pool cap stops the forward as soon as
            # the target row count is reached, so over-provisioning the
            # source sequence count is essentially free.
            wt_n_seq = max(
                1, (heal_cfg.xd_holdout_tokens + wt_seq_len - 1) // wt_seq_len + 4
            )
            log.info(
                "Stage-2 merge-heal: building WikiText cross-domain holdout "
                "(%d seqs × %d tokens, target %d-row pool/layer)",
                wt_n_seq, wt_seq_len, heal_cfg.xd_holdout_tokens,
            )
            xd_calib = _thermo_wikitext_tensor(
                tokenizer, num_sequences=wt_n_seq, sequence_length=wt_seq_len,
                dataset=wt_dataset, subset=wt_subset, split=wt_split,
            )
            xd_batches = iter_batches(xd_calib, batch_size=s2["batch_size"])

    # -----------------------------------------------------------------------
    # Crash-resume: scan partial_dir for layers already completed in a prior
    # interrupted run. Re-apply merges in layer order (fast, no forward pass).
    # File IO (orphan cleanup + JSON parse + neuron-means load) lives in
    # ``stage2.resume.discover_completed_layers``. The model-mutation loop
    # below replays each record against the live model.
    # -----------------------------------------------------------------------
    completed_layers: set[int] = set()
    # R-1 v2 / Mutation 3(a): hoist the configured merge_step parse here
    # (from the post-resume config block) so the eager init below has the
    # configured value in scope. Plugin #9 / S2_MM (MergeMoE arXiv:2510.14436):
    # merge_step was already validated at the top of run() (alongside the
    # cost_asymmetric / freq invariant). This is a re-read of the canonical
    # resolved value passed into LayerMergePlugin and emitted to Trackio
    # below — no second raise to avoid duplicate error paths. See
    # tasks/PLAN_PLUGIN_09_s2_mm.md and :mod:`stage2.mergemoe`.
    merge_step: str = str(s2.get("merge_step", "freq_weighted")).lower()
    # R-1 v2 / Mutation 1: per-layer effective merge_step pin for Trackio
    # (§2.1, §2.2 of tasks/PLAN_R1_REGMEAN_RESUME_TRACKIO.md). Pre-populated
    # with the configured value for every MoE layer so the one-shot emit at
    # the run-scope Trackio block has full coverage regardless of resume
    # state. The resume loop below only OVERWRITES resumed entries to
    # "freq_weighted" — non-resumed layers keep the configured default,
    # clean (non-resume) runs see all configured values. Hoisted out of
    # the ``else:`` branch so ``--no-resume`` (the if-branch below) also
    # sees the binding.
    _effective_merge_step_per_layer: dict[int, str] = {
        ref.layer_idx: merge_step for ref in moe_layers
    }
    _layer_mean_costs: list[float] = []  # running history for cost-threshold gate (Strategy C)

    if no_resume:
        partial_dir = None
        # Delete stale partial dir so a future non-no-resume run cannot resume
        # from this run's incomplete (or absent) checkpoints.
        stale = artifacts_dir / "_stage2_partial"
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    else:
        partial_dir = artifacts_dir / "_stage2_partial"
        partial_dir.mkdir(parents=True, exist_ok=True)

        resumed_records = discover_completed_layers(
            partial_dir, moe_layers, heal_enabled=heal_cfg.enabled,
        )

        # Plugin #9 / S2_MM (D-mergemoe-resume-fallback): when the configured
        # ``merge_step`` is ``"mergemoe"`` but the run is being resumed from
        # ``partial_dir``, the per-layer ``_LayerInputAccumulator`` calibration
        # buffer is not on disk — so the lstsq T₁=Q·P† solve cannot run for the
        # replayed layers. We force ``merge_step="freq_weighted"`` for the
        # resume loop below (see kwarg on the ``_merge_experts_inplace`` call).
        # The in-function fallback log.warning in ``merging.py`` never fires for
        # these layers because the forced ``"freq_weighted"`` short-circuits the
        # ``layer_inputs is None`` check. Emit the deviation warning HERE so the
        # operator sees one log line per resumed run that surfaces the silent
        # downgrade — and note that ``stage2/config/merge_step`` Trackio key
        # still records the configured (not effective per-layer) value.
        if (
            str(s2.get("merge_step", "freq_weighted")).lower() == "mergemoe"
            and resumed_records
        ):
            log.warning(
                "Stage 2 resume: forcing merge_step=freq_weighted for %d "
                "replayed layers (D-mergemoe-resume-fallback). The "
                "_LayerInputAccumulator buffer is not persisted across resume, "
                "so MergeMoE's lstsq solve cannot run on the replayed layers. "
                "New layers (post-resume-point) will use the configured "
                "merge_step. Trackio key stage2/config/merge_step still records "
                "the configured value, not the effective per-layer choice.",
                len(resumed_records),
            )

        # RegMean resume fallback — same posture as the MergeMoE block above.
        # The per-layer cov_acc IS persisted to ``_stage2_partial/`` (via
        # ``_snapshot_cov_layer``) and is reloaded later in this loop via
        # ``cov_acc.load_layer_from_disk``, BUT that load happens AFTER the
        # ``_merge_experts_inplace`` call below. We force the replayed
        # layers to ``freq_weighted`` for determinism (matching what the
        # in-function fallback path would have produced anyway) rather than
        # re-ordering the resume loop. Operators who need RegMean on a
        # mid-run crash can re-run with ``--no-resume``.
        if (
            str(s2.get("merge_step", "freq_weighted")).lower() == "regmean"
            and resumed_records
        ):
            log.warning(
                "Stage 2 resume: forcing merge_step=freq_weighted for %d "
                "replayed layers (D-regmean-resume-fallback). The replay "
                "loop applies _merge_experts_inplace BEFORE cov_acc is "
                "reloaded from disk, so the RegMean solve cannot see the "
                "per-member Gram for the replayed layers. New layers "
                "(post-resume-point) will use the configured merge_step. "
                "Trackio key stage2/config/merge_step still records the "
                "configured value, not the effective per-layer choice. "
                "Re-run with --no-resume if RegMean fidelity is required.",
                len(resumed_records),
            )

        for record in resumed_records:
            ref = record.layer_ref
            # scores=None on the resume path: saliency scores are not
            # persisted on disk. A saliency-mode run that resumes here will
            # hit the new ValueError inside _merge_experts_inplace for the
            # first multi-member merge group — that is the correct surfacing
            # of the gap (rather than silently applying wrong weights).
            #
            # Plugin #9 / S2_MM (D-mergemoe-resume-fallback): the
            # ``_LayerInputAccumulator`` calibration buffer is NOT persisted
            # across runs, so a resumed MergeMoE run would silently fall back
            # to freq-weighted per cluster (via the in-function warning) for
            # every replayed layer. Force ``merge_step="freq_weighted"``
            # explicitly here so the resume is deterministic and matches
            # what the freq-weighted fallback path would have produced anyway
            # — same posture as the saliency ``scores=None`` deviation above.
            # MergeMoE-mode crashes therefore require ``--no-resume`` to be
            # re-run with the original calibration buffer. The single
            # log.warning above (gated on configured merge_step == "mergemoe"
            # and non-empty resumed_records) surfaces this deviation once per
            # resumed run rather than once per layer.
            _merge_experts_inplace(
                ref, record.grouped, record.freq,
                freq_weighted=s2["ream"]["frequency_weighted_merge"],
                scores=None,
                ream_acc=record.resume_ream_acc,
                merge_step="freq_weighted",
            )
            # build_banks again: _merge_experts_inplace already called it
            # internally, but bank.select() was never called on any of those
            # banks, so the _last_kept_ids_* sentinel is still unset and this
            # select() call is safe.
            banks = build_banks(ref)
            for bank in banks.values():
                bank.select(record.final_kept_ids)
            _resize_router_for_kept_experts(ref, record.final_kept_ids)

            # Merge-heal resume: a completed layer's post-heal weights are not
            # reconstructible from merge_*.json — reload them so the in-memory
            # model matches the state the heal left. A 0-merge layer has no
            # heal-weights file (the heal is skipped, heal_state stays None);
            # its banks are already fully reconstructed above, so gate the load
            # on the file existing — mirror the _write_heal_weights condition.
            if record.has_heal_weights_file:
                _load_heal_weights(partial_dir, ref, record.final_kept_ids)

            try:
                cov_acc.load_layer_from_disk(ref.layer_idx, partial_dir)
            except Exception as _exc:
                raise RuntimeError(
                    f"Stage 2 resume: failed to load covariance for layer {ref.layer_idx} "
                    f"from _stage2_partial/ ({_exc}). "
                    "The in-memory model has already been partially mutated — "
                    "restart with a fresh Stage 1 model and delete _stage2_partial/."
                ) from _exc

            merge_map[ref.layer_idx] = record.merge_map_layer
            completed_layers.add(ref.layer_idx)
            # R-1 v2 / Mutation 2: surface the forced freq_weighted downgrade
            # for this replayed layer (overwrites the eager-init default
            # written at line ~825). The forced ``merge_step="freq_weighted"``
            # kwarg on ``_merge_experts_inplace`` above IS the truth we're
            # recording; copying the string keeps drift impossible. See
            # tasks/PLAN_R1_REGMEAN_RESUME_TRACKIO.md § 3.1 Mutation 2.
            _effective_merge_step_per_layer[ref.layer_idx] = "freq_weighted"
            log.info(
                "Stage 2: layer %d resumed from partial (skipping profile + merge)",
                ref.layer_idx,
            )
            val = record.mean_cost_per_pair
            if val is not None and val > 0.0:
                _layer_mean_costs.append(float(val))

        if completed_layers:
            log.info(
                "Stage 2: resumed %d / %d layers from %s",
                len(completed_layers), len(moe_layers), partial_dir,
            )

    # B-C-H-1: default to 8 (D5a value) so REAM merging always has a per-centroid
    # cap, preventing degenerate one-centroid-absorbs-all groupings. Setting to 0
    # explicitly disables the cap (uncapped path; rare, ablation-only); users must
    # opt in to that. The `or 8` collapse from earlier was dropped so an explicit
    # `0` in config is honored as "uncapped" rather than silently overridden to 8.
    max_group_cap: int = int(s2.get("max_merge_group_size", 8))
    # B-iter5-L-1 (code): default to 1.5 (D-ream-budget-bump value) so the quality
    # gate is active out-of-the-box; setting to a very large value (e.g. inf) in
    # config disables the gate.
    cost_sigma: float = s2.get("ream_cost_sigma_threshold", 1.5)
    cost_bump_ratio: float = s2.get("ream_cost_bump_ratio", 0.10)
    min_active_tokens: int = s2.get("reap_min_active_tokens", 0)
    # Stage 2 v2 — assignment solver dispatch. Default "greedy" reproduces v1
    # behavior exactly. See max_quality/docs/stage2_assignment_revision.md § 6.
    # Validate the YAML value at the boundary so typos are caught at the
    # config-load site rather than at the per-layer call.
    _solver_value = str(s2.get("assignment_solver", "greedy")).lower()
    _valid_solvers = ("greedy", "hungarian", "mcf", "auto", "sinkhorn")
    if _solver_value not in _valid_solvers:
        raise ValueError(
            f"stage2_reap_ream.assignment_solver={_solver_value!r} is not a "
            f"valid solver name; expected one of {_valid_solvers}."
        )
    assignment_solver: SolverName = _solver_value  # type: ignore[assignment]

    # Stage 2 v2 cost matrix variants (spec § 5 step 4 / § 6).
    # Direction C adds "output": an output-space merge cost that measures the
    # actual change in the layer's gated routed-expert output on calibration
    # tokens when a child is tentatively merged into a centroid. A strictly
    # better merge-damage proxy than the weight-space "pre"/"post" costs.
    cost_alignment_cfg: str = str(s2.get("cost_alignment", "pre")).lower()
    if cost_alignment_cfg not in ("pre", "post", "output"):
        raise ValueError(
            f"stage2_reap_ream.cost_alignment={cost_alignment_cfg!r}; "
            "expected 'pre', 'post', or 'output'."
        )
    # Plugin #9 / S2_MM (MergeMoE arXiv:2510.14436): merge_step was already
    # validated at the top of run() (alongside the cost_asymmetric / freq
    # invariant). The canonical resolved value (passed into LayerMergePlugin
    # and emitted to Trackio below) is parsed at the top of the resume block
    # (line 825 region) so the R-1 v2 eager-init of
    # ``_effective_merge_step_per_layer`` has it in scope before the resume
    # loop runs. See tasks/PLAN_PLUGIN_09_s2_mm.md, :mod:`stage2.mergemoe`,
    # and tasks/PLAN_R1_REGMEAN_RESUME_TRACKIO.md § 3.1 Mutation 3(a).
    # Direction C — output-space cost calibration-token cap. Only consumed when
    # cost_alignment == "output"; bounds the per-pair SwiGLU residual compute.
    cost_output_token_cap: int = int(s2.get("cost_output_token_cap", 1024))
    if cost_output_token_cap < 1:
        raise ValueError(
            f"stage2_reap_ream.cost_output_token_cap={cost_output_token_cap}; "
            "must be >= 1 (number of calibration tokens for the output cost)."
        )
    cost_whitening: str = str(s2.get("cost_whitening", "none")).lower()
    if cost_whitening not in ("none", "diag", "full"):
        raise ValueError(
            f"stage2_reap_ream.cost_whitening={cost_whitening!r}; "
            "expected 'none', 'diag', or 'full'."
        )
    cost_asymmetric: bool = bool(s2.get("cost_asymmetric", False))
    cost_topk_filter: int = int(s2.get("cost_topk_filter", 48))
    capacity_util_threshold: float = float(s2.get("capacity_util_threshold", 0.25))
    em_refinement_rounds: int = int(s2.get("em_refinement_rounds", 0))
    em_convergence_break: bool = bool(s2.get("em_convergence_break", True))
    # Direction D — greedy + 2-opt local refinement. Only active when
    # assignment_solver == "greedy"; strictly-improving so it cannot regress.
    two_opt_refine: bool = bool(s2.get("two_opt_refine", False))
    sinkhorn_epsilon_init: float = float(s2.get("sinkhorn_epsilon_init", 1.0))
    sinkhorn_epsilon_final: float = float(s2.get("sinkhorn_epsilon_final", 0.01))
    sinkhorn_iters: int = int(s2.get("sinkhorn_iters", 200))
    # Direction B — skip-merge floor. Per-layer percentile P over the *finite*
    # entries of the cost matrix; every entry strictly above P is masked to
    # +inf so those pairs fall through to orphan promotion (singleton kept
    # experts). OFF sentinel: 100.0 (the 100th percentile is the max finite
    # cost, so nothing is strictly above it -> no entry masked -> byte-identical
    # to the unmasked run). Valid range [0.0, 100.0].
    skip_merge_percentile: float = float(s2.get("skip_merge_percentile", 100.0))
    if not (0.0 <= skip_merge_percentile <= 100.0):
        raise ValueError(
            f"stage2_reap_ream.skip_merge_percentile={skip_merge_percentile}; "
            "must be in [0.0, 100.0] (100.0 = off, mask nothing)."
        )
    if em_refinement_rounds < 0:
        raise ValueError(
            f"stage2_reap_ream.em_refinement_rounds={em_refinement_rounds}; "
            "must be >= 0 (set 0 to disable)."
        )
    # Phase 3 (M8): per-merge-group expert distillation flags.
    expert_distill_steps: int = int(s2.get("expert_distill_steps", 0))
    expert_distill_lr: float = float(s2.get("expert_distill_lr", 1e-4))
    _betas_raw = s2.get("expert_distill_betas", [0.9, 0.95])
    expert_distill_betas: tuple[float, float] = (float(_betas_raw[0]), float(_betas_raw[1]))
    expert_distill_token_cap: int = int(s2.get("expert_distill_token_cap", 8192))
    expert_distill_skip_singletons: bool = bool(s2.get("expert_distill_skip_singletons", True))
    expert_distill_plateau_steps: int = int(s2.get("expert_distill_loss_plateau_steps", 50))
    expert_distill_plateau_eps: float = float(s2.get("expert_distill_loss_plateau_eps", 1e-4))
    # Lift 1 — D-expert-distill-ce-term: paper Eq. 10's L_KD = L_CE + λ · MSE.
    # Defaults: CE ON (default True post-lift), λ = 1.0 (paper line 414 does not
    # pin a numeric default; parity weighting is the safe ON-path starting point).
    expert_distill_use_ce_term: bool = bool(s2.get("expert_distill_use_ce_term", True))
    expert_distill_ce_lambda: float = float(s2.get("expert_distill_ce_lambda", 1.0))
    # Lift 2 — D-expert-distill-paper-lift: target version. "v2" is the
    # paper-faithful TopK-gated + per-token routing-weighted target (Eqs. 1-3,
    # paper lines 133-152) and is the default post-lift. "v1" preserves the
    # legacy freq-weighted-only target for A0..A11 ablation parity.
    expert_distill_target_version: str = str(
        s2.get("expert_distill_target_version", "v2")
    )
    if expert_distill_target_version not in ("v1", "v2"):
        raise ValueError(
            "stage2_reap_ream.expert_distill_target_version="
            f"{expert_distill_target_version!r}; must be 'v1' or 'v2'."
        )
    if expert_distill_steps < 0:
        raise ValueError(
            f"stage2_reap_ream.expert_distill_steps={expert_distill_steps}; "
            "must be >= 0 (set 0 to disable)."
        )
    if expert_distill_ce_lambda < 0.0:
        raise ValueError(
            f"stage2_reap_ream.expert_distill_ce_lambda="
            f"{expert_distill_ce_lambda}; must be >= 0 (set 0 (with "
            "`use_ce_term=True`) to silence the MSE term while still "
            "running CE; with `use_ce_term=False` the run is pure MSE "
            "and lambda has no effect)."
        )
    # cost_asymmetric × freq_weighted_merge invariant is checked at the very
    # top of run() (fail-fast); we rely on that here.

    # Stage 2 v2 (spec § 6) — one-shot Trackio emit of the static config so
    # the dashboard run-summary reflects which features are active without
    # parsing per-layer logs. All v2 config flags + the partial-JSON
    # format_version are surfaced under the "stage2/config/*" namespace.
    _trackio_log({
        "stage2/config/assignment_solver": assignment_solver,
        "stage2/config/cost_alignment": cost_alignment_cfg,
        "stage2/config/cost_whitening": cost_whitening,
        "stage2/config/cost_asymmetric": cost_asymmetric,
        "stage2/config/cost_topk_filter": cost_topk_filter,
        "stage2/config/capacity_util_threshold": capacity_util_threshold,
        "stage2/config/em_refinement_rounds": em_refinement_rounds,
        "stage2/config/em_convergence_break": em_convergence_break,
        "stage2/config/two_opt_refine": two_opt_refine,
        "stage2/config/expert_distill_steps": expert_distill_steps,
        "stage2/config/expert_distill_token_cap": expert_distill_token_cap,
        "stage2/config/expert_distill_lr": expert_distill_lr,
        "stage2/config/expert_distill_use_ce_term": expert_distill_use_ce_term,
        "stage2/config/expert_distill_ce_lambda": expert_distill_ce_lambda,
        "stage2/config/expert_distill_target_version": expert_distill_target_version,
        "stage2/config/sinkhorn_iters": sinkhorn_iters,
        "stage2/config/merge_step": merge_step,
        # R-1 v2 / Mutation 3(b): per-layer effective merge_step + downgrade
        # forensics. The eager-init contract (§2.2 of
        # tasks/PLAN_R1_REGMEAN_RESUME_TRACKIO.md) guarantees one entry per
        # MoE layer in ``_effective_merge_step_per_layer`` by the time this
        # emit fires — replayed-under-resume entries carry "freq_weighted",
        # all others carry the configured ``merge_step``. Operators alert on
        # the scalar downgrade counter; the per-layer JSON dict is for the
        # post-mortem notebook; the reason string distinguishes the two
        # D-* deviations (RegMean-resume vs MergeMoE-resume) without log
        # grepping.
        "stage2/effective/merge_step_per_layer": json.dumps(
            {str(k): v for k, v in sorted(_effective_merge_step_per_layer.items())},
            sort_keys=True,
        ),
        "stage2/effective/merge_step_downgrades_total": (
            _downgrades_total := sum(
                1 for v in _effective_merge_step_per_layer.values() if v != merge_step
            )
        ),
        # R-1 v2 LOW-finding fold: emit a non-empty reason string ONLY when
        # the downgrade counter is nonzero. The original v1 formulation
        # keyed the reason on ``merge_step`` (configured) inside an
        # unconditional ternary, so a future change that allowed
        # ``freq_weighted`` to also produce per-layer downgrades would
        # silently mislabel them as the MergeMoE reason. Gating on
        # ``_downgrades_total > 0`` makes the keying explicit: no
        # downgrades → empty string; downgrades exist → reason follows
        # configured ``merge_step``. (Today both branches are reachable
        # only via RegMean-resume or MergeMoE-resume, so the inner ternary
        # is unchanged.)
        "stage2/effective/merge_step_downgrade_reason": (
            ""
            if _downgrades_total == 0
            else (
                "regmean_resume_no_cov_load_before_merge"
                if merge_step == "regmean"
                else "mergemoe_resume_no_calibration_buffer"
            )
        ),
        "stage2/config/format_version": 2,
    })

    # ---- Per-layer work is driven by the universal phase walker -------------
    # Each layer flows through the 9-phase plugin walk (see
    # max_quality/docs/stage2_plugin_guide.md). Shared setup (above) and the
    # final-checkpoint save (below) run once, around the layer loop.
    from ..pipeline.context import PipelineContext
    from ..pipeline.registry import PluginRegistry
    from ..tools.phase_walker import walk_phases
    from .plugins.capacity_gate import CapacityGatePlugin
    from .plugins.expert_distill import ExpertDistillPlugin
    from .plugins.layer_merge import LayerMergePlugin
    from .plugins.merge_heal import MergeHealPlugin
    from .plugins.output_space_cost import OutputSpaceCostPlugin
    from .plugins.ream_cost import ReamCostPrePlugin
    from .plugins.ream_cost_post import ReamCostPostPlugin
    from .plugins.ream_sequential import Stage2ReamSequentialPlugin
    from .plugins.reap_scoring import ReapScoringPlugin
    from .plugins.regmean_merge import RegMeanMergeStepPlugin
    from .plugins.reap_scores_cache import Stage2ReapScoresCacheProvider
    from .plugins.routing_stats_cache import Stage2RoutingStatsCacheProvider
    from .plugins.stage2_profile_cache import Stage2ProfileCacheProvider
    from .plugins.skip_merge_floor import SkipMergeFloorPlugin
    from .plugins.solver_auto import AutoSolverPlugin
    from .plugins.solver_greedy import GreedySolverPlugin
    from .plugins.solver_hungarian import HungarianSolverPlugin
    from .plugins.solver_mcf import McfSolverPlugin
    from .plugins.solver_sinkhorn import SinkhornSolverPlugin
    from .plugins.two_opt_refine import TwoOptRefinePlugin
    from .plugins.em_refine import EmRefinePlugin

    # The run-scope context is the root PipelineContext; each layer opens a
    # child() scope. Run-scope mutable scratchpad (cov_acc, merge_map,
    # _layer_mean_costs, partial_dir) lives on the LayerMergePlugin instance
    # instead — the plugin is constructed once per run() invocation and is
    # the natural home for single-run-scoped state.
    run_ctx = PipelineContext()
    run_ctx.set("model", model)
    run_ctx.set("tokenizer", tokenizer)
    run_ctx.set("config", config)
    run_ctx.set("artifacts_dir", artifacts_dir)
    # Store the TRUE partial_dir, including ``None`` in no-resume mode:
    # ``write_artifacts`` branches on ``if partial_dir is not None:`` and must
    # see ``None`` (not a fallback path) when resume is disabled.
    run_ctx.set("partial_dir", partial_dir)
    # The "device" ctx slot holds the *stringified* device ("cpu" / "cuda:0"),
    # not a torch.device — the original device object is passed separately to
    # the stage-2 plugins (the `device=device` kwarg below). Readers of
    # run_ctx.get("device") must not expect a torch.device.
    run_ctx.set("device", str(device) if device is not None else "cpu")
    # S2-12a: the per-layer merge spine. ``LayerMergePlugin`` now owns the SIX
    # live phase hooks (on_layer_setup / on_profile / merge / post_merge /
    # write_artifacts / on_layer_teardown) that ``LegacyAdapter`` used to carry.
    # Constructed from the SAME parsed run() locals — the knob/scratchpad
    # superset = (a) every knob the 6 live hooks read off ``self`` PLUS (b) the
    # eight attributes ``_run_assignment`` reads off the plugin instance.
    layer_merge = LayerMergePlugin(
        s2_cfg=s2, heal_cfg=heal_cfg,
        batches=batches, model=model,
        cov_acc=cov_acc, merge_map=merge_map,
        layer_mean_costs=_layer_mean_costs,
        partial_dir=partial_dir,
        max_group_cap=max_group_cap, cost_sigma=cost_sigma,
        cost_bump_ratio=cost_bump_ratio, min_active_tokens=min_active_tokens,
        assignment_solver=assignment_solver, cost_alignment_cfg=cost_alignment_cfg,
        cost_output_token_cap=cost_output_token_cap,
        cost_asymmetric=cost_asymmetric,
        expert_distill_steps=expert_distill_steps,
        expert_distill_token_cap=expert_distill_token_cap,
        blacklist=blacklist, device=device,
        # Plugin #9 / S2_MM — see PLAN_PLUGIN_09_s2_mm.md.
        merge_step=merge_step,
    )
    # Registration order matters: ReapScoringPlugin.on_layer_setup must run
    # BEFORE LayerMergePlugin.on_profile (which reads ctx.reap_acc into
    # _profile_layer). ``walk_phases`` dispatches each phase to every plugin
    # in sequence order, so listing ReapScoringPlugin first satisfies the
    # dependency.
    #
    # S2-6: the three live cost plugins are registered BETWEEN ReapScoringPlugin
    # and the merge spine so the enabled one wins the ``compute_cost``
    # ``dispatch_first`` slot. Each is constructed with the SAME parsed cost
    # knobs + the SAME ``cov_acc`` object. ``registry.enabled(config)`` drops
    # the two cost plugins whose ``is_enabled`` gate is False, leaving exactly
    # one cost plugin (the one matching ``cost_alignment``).
    # S2-10: the capacity-util gate moved out of the cost plugins into
    # CapacityGatePlugin (the ``select_alignment`` slot). The cost plugins no
    # longer take ``max_group_cap`` / ``capacity_util_threshold`` /
    # ``cost_asymmetric`` — those knobs are passed to CapacityGatePlugin
    # instead.
    _cost_plugin_kwargs = dict(
        cov_acc=cov_acc,
        cost_alignment_cfg=cost_alignment_cfg,
        cost_whitening=cost_whitening,
        cost_topk_filter=cost_topk_filter,
        cost_output_token_cap=cost_output_token_cap,
    )
    # S2-7: the skip-merge floor plugin is registered AFTER the three cost
    # plugins so it wins the ``apply_cost_mask`` ``dispatch_first`` slot.
    # Constructed directly from the already-parsed and range-validated
    # ``skip_merge_percentile`` local. ``registry.enabled`` drops it at the OFF
    # sentinel (>= 100.0); with no plugin servicing the slot ``dispatch_first``
    # returns None and ``_run_assignment`` leaves the cost matrix unmasked.
    # S2-8: the five solver plugins are registered AFTER the skip-merge floor
    # plugin so the enabled one wins the ``solve_assignment`` ``dispatch_first``
    # slot. Each is constructed with the SAME parsed assignment knobs.
    # ``registry.enabled`` gates each on ``assignment_solver``, leaving exactly
    # one solver plugin (the one matching the configured solver).
    _solver_plugin_kwargs = dict(
        max_group_cap=max_group_cap,
        assignment_solver=assignment_solver,
        sinkhorn_epsilon_init=sinkhorn_epsilon_init,
        sinkhorn_epsilon_final=sinkhorn_epsilon_final,
        sinkhorn_iters=sinkhorn_iters,
    )
    registry = PluginRegistry([
        # V1+V2 (REAP-exact via vLLM hooks): cache provider runs first.
        # ``on_load`` tries to hydrate ``ctx.reap_scores_payload`` from a
        # sidecar produced by ``--capture-reap-scores``; on a per-layer
        # ``on_score`` hit it populates ``scores`` + ``freq`` so the live
        # ReapScoringPlugin.on_score (registered next) short-circuits via
        # its ``ctx.has("scores")`` guard. On miss, this provider is a no-op
        # and the live REAP path runs unchanged.
        Stage2ReapScoresCacheProvider(),
        # V2 (routing_stats infrastructure): cache provider registered
        # immediately AFTER Stage2ReapScoresCacheProvider so it joins the
        # same run-scope ``dispatch_first("on_load", ...)`` chain. On hit
        # it deposits the payload on ``ctx.routing_stats_payload`` for
        # future read-side plugins; on miss it returns None gracefully.
        # No per-layer ``on_score`` hook -- there is no immediate per-
        # layer consumer (Item 3 lays infrastructure only).
        Stage2RoutingStatsCacheProvider(),
        ReapScoringPlugin(),
        # S2-10: the capacity-utilization gate. Registered AFTER ReapScoringPlugin
        # and BEFORE the three cost plugins so its ``select_alignment`` slot runs
        # earlier in the bump iteration and publishes the gate decision
        # (effective_cost_alignment / effective_cost_asymmetric /
        # capacity_util_value) the cost plugins' ``compute_cost`` slot reads back.
        CapacityGatePlugin(
            max_group_cap=max_group_cap,
            capacity_util_threshold=capacity_util_threshold,
            cost_alignment_cfg=cost_alignment_cfg,
            cost_asymmetric=cost_asymmetric,
        ),
        ReamCostPrePlugin(**_cost_plugin_kwargs),
        ReamCostPostPlugin(**_cost_plugin_kwargs),
        OutputSpaceCostPlugin(**_cost_plugin_kwargs),
        SkipMergeFloorPlugin(skip_merge_percentile=skip_merge_percentile),
        GreedySolverPlugin(**_solver_plugin_kwargs),
        HungarianSolverPlugin(**_solver_plugin_kwargs),
        McfSolverPlugin(**_solver_plugin_kwargs),
        SinkhornSolverPlugin(**_solver_plugin_kwargs),
        AutoSolverPlugin(**_solver_plugin_kwargs),
        # S2-9: the two refinement plugins are registered AFTER the solver
        # plugins. refine_assignment is a CHAIN (two-opt THEN EM), so unlike
        # the single-winner cost / mask / solve slots both may run — registry
        # order is chain order, two-opt first. Each is constructed from the
        # SAME parsed knobs (notably the SAME cov_acc object). registry.enabled
        # drops whichever refiner's gate is off.
        TwoOptRefinePlugin(
            two_opt_refine=two_opt_refine,
            assignment_solver=assignment_solver,
            max_group_cap=max_group_cap,
        ),
        EmRefinePlugin(
            em_refinement_rounds=em_refinement_rounds,
            em_convergence_break=em_convergence_break,
            max_group_cap=max_group_cap,
            assignment_solver=assignment_solver,
            cost_whitening=cost_whitening,
            cost_asymmetric=cost_asymmetric,
            cost_topk_filter=cost_topk_filter,
            skip_merge_percentile=skip_merge_percentile,
            cov_acc=cov_acc,
            sinkhorn_epsilon_init=sinkhorn_epsilon_init,
            sinkhorn_epsilon_final=sinkhorn_epsilon_final,
            sinkhorn_iters=sinkhorn_iters,
        ),
        # S2-12: the per-layer merge spine. ``LayerMergePlugin`` carries the
        # SIX live phase hooks relocated out of the retired ``LegacyAdapter``
        # (S2-12a) — its registry position is exactly where the adapter's used
        # to be, so the phase-major walk lands its hooks unchanged. S2-12b
        # deleted the ``LegacyAdapter`` class entirely.
        layer_merge,
        # RegMean (Jin et al., ICLR 2023, arXiv:2212.09849) — metadata /
        # config-validation shim for ``merge_step="regmean"``. Carries NO
        # phase hooks; the actual closed-form solve is invoked inline by
        # ``LayerMergePlugin.merge`` -> ``_merge_experts_inplace`` ->
        # ``_regmean_solve_one_linear``. The shim's ``is_enabled`` gate is
        # True only when ``stage2_reap_ream.merge_step == "regmean"``, so
        # the plugin self-deselects on every non-regmean run.
        # Position: AFTER ``layer_merge`` mirroring MergeMoE's pattern
        # (MergeMoE has no separate plugin class — its math lives on
        # LayerMergePlugin's merge_step knob, which RegMean extends).
        RegMeanMergeStepPlugin(),
        # Profile-sidecar cache reader (Optimization A REDO — Plugin #12).
        # Single config knob: stage2_reap_ream.profile_sidecar.enabled.
        # Same flag governs gate-logit hydration, cov_acc hydration, AND
        # layer_input_acc hydration — all from ONE sidecar written by
        # --capture-stage2-profile (Bug #8 fix is structural: this is the
        # only reader for the profile sidecar).
        # Registration AFTER layer_merge is REQUIRED so
        # Stage2ProfileCacheProvider.on_layer_setup runs SECOND (OQ-1
        # Option A — in-place hydration of the fresh ream_acc /
        # layer_input_acc that LayerMergePlugin.on_layer_setup
        # constructs). Registering before layer_merge would let the
        # fresh empty accumulators overwrite the hydrated ones.
        # On full hit: hydrates ream_acc + cov_acc + layer_input_acc so
        # LayerMergePlugin.on_profile early-returns (Pattern A skip).
        # On partial/miss: no-op; live forward path runs unchanged.
        *(
            [Stage2ProfileCacheProvider(
                cov_acc=cov_acc,
                expected_cov_storage_dtype=s2.get(
                    "covariance_storage_dtype", "float16",
                ),
            )]
            if (s2.get("profile_sidecar") or {}).get("enabled", False)
            else []
        ),
        # S2-11: the per-merge-group expert distillation + per-layer merge-heal
        # plugins are registered AFTER the merge spine — the LAST elements,
        # reversed vs. the S2-6..S2-10 ordering. ``pre_merge_snapshot`` /
        # ``merge`` / ``post_merge`` are ``walk_phases`` PHASES (every plugin's
        # hook runs, phase-major / plugin-minor), not ``dispatch_first`` slots.
        # These two plugins must run AFTER the merge spine so
        # ``ExpertDistillPlugin.merge`` lands after
        # ``LayerMergePlugin._merge_experts_inplace`` and
        # ``MergeHealPlugin.post_merge`` lands after ``LayerMergePlugin``'s
        # ``bank.select`` + router resize. ``registry.enabled(config)`` drops
        # each when its gate is off (expert_distill_steps == 0 /
        # merge_heal_enabled == False) — inert by default.
        ExpertDistillPlugin(
            expert_distill_steps=expert_distill_steps,
            expert_distill_lr=expert_distill_lr,
            expert_distill_betas=expert_distill_betas,
            expert_distill_token_cap=expert_distill_token_cap,
            expert_distill_skip_singletons=expert_distill_skip_singletons,
            expert_distill_plateau_steps=expert_distill_plateau_steps,
            expert_distill_plateau_eps=expert_distill_plateau_eps,
            expert_distill_use_ce_term=expert_distill_use_ce_term,
            expert_distill_ce_lambda=expert_distill_ce_lambda,
            expert_distill_target_version=expert_distill_target_version,
        ),
        MergeHealPlugin(
            heal_cfg=heal_cfg,
            heal_device=_heal_device,
            xd_batches=xd_batches,
            batches=batches,
            model=model,
            artifacts_dir=artifacts_dir,
            device=device,
        ),
        # Plugin #10 (row S2_SEQ) — REAM sequential merging cache invalidator.
        # Registered LAST in the PluginRegistry list so its ``on_post_merge``
        # hook fires after any other plugin's ``on_post_merge`` (none of the
        # other 18 Stage 2 plugins implement that hook today, but ordering
        # last is the safe default for any future plugin that wants to
        # READ the caches in ``on_post_merge`` before they get cleared).
        # ``LayerMergePlugin`` does NOT implement ``on_post_merge`` at all
        # (its merge-spine work happens in ``merge`` / ``post_merge``), so
        # the "must run AFTER LayerMergePlugin in on_post_merge phase"
        # contract is satisfied trivially. Default-OFF gate
        # (``stage2_reap_ream.sequential_reprofile``): ``registry.enabled``
        # drops this plugin at the default, preserving byte-identical
        # existing behavior. See arXiv:2604.04356 §4 / SC_STAGE12 §523-532.
        Stage2ReamSequentialPlugin(),
    ])
    plugins = registry.enabled(config)
    walk_phases(("on_run_setup",), plugins, run_ctx)

    # Run-scope sidecar load. Stage2ReapScoresCacheProvider.on_load tries
    # to read the REAP-scores sidecar at <jsonl_dir>/sidecars/reap_scores.pt.
    # On hit: per-layer Stage2ReapScoresCacheProvider.on_score populates
    # ctx.scores + ctx.freq so ReapScoringPlugin.on_score's ctx.has("scores")
    # guard short-circuits the live finalize step. On miss: the live path
    # runs normally.
    #
    # Use the calibration loader's default JSONL path when `jsonl_path` is
    # absent from the YAML, mirroring the loader's resolution so a user who
    # wrote the sidecar to the default location is not silently bypassed.
    from pathlib import Path as _Path
    from ..utils.calibration import _DEFAULT_SELF_TRACES_PATH
    _calib_source = cal.get("jsonl_path", _DEFAULT_SELF_TRACES_PATH)
    _calib_jsonl_path = _Path(_calib_source)
    if not _calib_jsonl_path.is_absolute():
        _calib_jsonl_path = _Path.cwd() / _calib_jsonl_path
    PluginRegistry.dispatch_first(
        plugins, "on_load", run_ctx, _calib_jsonl_path,
    )

    # Item 3 routing-stats cache load. ``dispatch_first`` above stops at
    # the FIRST non-None result (its semantics are "first winner takes
    # all"), so if Stage2ReapScoresCacheProvider hits, the
    # Stage2RoutingStatsCacheProvider's on_load is NEVER invoked through
    # that chain. We therefore call its on_load EXPLICITLY here so the
    # routing-stats payload always gets a chance to populate ctx,
    # regardless of REAP-cache outcome. This mirrors Stage 1's STEP 4.6
    # divergence (Item 3 is infrastructure-only, no live counterpart).
    for _plug in plugins:
        if isinstance(_plug, Stage2RoutingStatsCacheProvider):
            _plug.on_load(run_ctx, _calib_jsonl_path)
            break

    # Plugin #12 REDO: stage2-profile cache provider on_load. Same
    # dispatch_first-stops-at-first-non-None hazard as routing-stats above,
    # so call its on_load EXPLICITLY here. The provider is only present
    # when ``profile_sidecar.enabled`` was True at registration time
    # (registry.enabled drops it otherwise), so the explicit loop is
    # naturally a no-op on disabled runs.
    for _plug in plugins:
        if isinstance(_plug, Stage2ProfileCacheProvider):
            _plug.on_load(run_ctx, _calib_jsonl_path)
            break

    for k, layer_ref in enumerate(moe_layers):
        if layer_ref.layer_idx in completed_layers:
            log.info(
                "Stage 2 layer %d/%d (idx=%d) — skipped (resumed from partial)",
                k + 1, len(moe_layers), layer_ref.layer_idx,
            )
            continue
        target = per_layer_target[layer_ref.layer_idx]
        log.info(
            "Stage 2 layer %d/%d (idx=%d) — profiling then merging to %d experts",
            k + 1, len(moe_layers), layer_ref.layer_idx, target,
        )
        ctx = run_ctx.child()
        ctx.set("layer_idx", layer_ref.layer_idx)
        ctx.set("layer_ref", layer_ref)
        ctx.set("n_experts", layer_ref.num_routed_experts)
        ctx.set("target", target)
        ctx.set("blacklist", tuple(blacklist.get(layer_ref.layer_idx, [])))
        # ``_layer_rank`` is the 0-based index into the ordered MoE layer
        # list — the same indexing convention the REAP-scores cache writer
        # uses. Stage2ReapScoresCacheProvider.on_score reads this to slice
        # the per-layer row out of the loaded payload.
        ctx.set("_layer_rank", k)
        # S2-5: the assignment phase is no longer a plain ``walk_phases`` slot.
        # The pre-assign phases run, then ``_run_assignment`` drives the bump
        # loop over the four fine-grained assignment slots, then the
        # post-assign phases run. ``_STAGE2_LAYER_PHASES`` (the derived 9-tuple)
        # is the back-compat view of this same canonical order.
        walk_phases(_STAGE2_PRE_ASSIGN_PHASES, plugins, ctx)
        _run_assignment(plugins, ctx)
        walk_phases(_STAGE2_POST_ASSIGN_PHASES, plugins, ctx)
    walk_phases(("on_run_teardown",), plugins, run_ctx)

    out_dir = artifacts_dir / "stage2_pruned"
    if os.environ.get("MOE_SKIP_STAGE2_COV_SAVE") == "1":
        log.info("Skipping _stage2_input_covariance.pt save "
                 "(MOE_SKIP_STAGE2_COV_SAVE=1; Stages 3/4 disabled, file unused)")
    else:
        _save_covariance(cov_acc, artifacts_dir / "_stage2_input_covariance.pt")
    save_compressed_checkpoint(
        model, tokenizer, out_dir,
        pipeline_stage="stage2_pruned",
        extra_metadata={"merge_map_file": "merge_map.json"},
    )
    save_json_artifact(merge_map, out_dir / "merge_map.json")
    if partial_dir is not None:
        if os.environ.get("MOE_KEEP_STAGE2_PARTIAL") == "1":
            # Direction A — budget retune reads per-layer measured damage from
            # _stage2_partial/merge_*.json. Keep the dir so a baseline run's
            # damage signal survives for the retune tool.
            log.info("Keeping %s (MOE_KEEP_STAGE2_PARTIAL=1) for budget retune",
                     partial_dir)
        else:
            shutil.rmtree(partial_dir, ignore_errors=True)
    log.info("Stage 2 complete — pruned checkpoint at %s", out_dir)
    return out_dir
