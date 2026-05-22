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
# full 9-tuple keep working.
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
    "on_layer_teardown",
)
# Derived back-compat constant: the full 9-phase schedule with the compound
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
    reached via ``PluginRegistry.dispatch_first`` over the four fine-grained
    slots ``compute_cost`` / ``apply_cost_mask`` / ``solve_assignment`` /
    ``refine_assignment``. Because the Stage-2 registry is
    ``[ReapScoringPlugin(), adapter]`` and ReapScoringPlugin declares none of
    those slots, every dispatch lands on ``LegacyAdapter``'s extracted methods
    — behaviour is byte-identical to the pre-S2-5 monolithic hook. S2-6+ wires
    the real cost / solver / refine plugins ahead of the adapter so they win
    the slot.

    ``_run_assignment`` owns the bump-loop control flow, the b_fail / c_fail
    gates, the orphan-promotion grouping, and the final ``ctx.set`` of all
    per-layer output slots — exactly the responsibilities the monolithic
    ``compute_assignment`` carried.
    """
    from ..pipeline.registry import PluginRegistry
    from .plugins.legacy_adapter import LegacyAdapter
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

    # The LegacyAdapter instance owns the run-scope scratchpad (blacklist,
    # _layer_mean_costs). Locate it in the plugin list — it is the plugin
    # exposing the four assignment slots.
    adapter = next(p for p in plugins if isinstance(p, LegacyAdapter))
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
            # forward to the next. A plugin declining the slot (returns None,
            # e.g. LegacyAdapter, which trails the chain as a dead fallback,
            # or a refiner whose own gate is off this layer) is skipped.
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

    # Stage 2 v2 (spec § 6 / D-asymmetric-freq): cost_asymmetric is valid
    # only under freq-weighted merge — the asymmetric factor freq_m/(freq_c+freq_m)
    # is the per-pair version of the merge weight. Reject the combination
    # at the very top of `run` so misconfigured pipelines fail fast before
    # spending compute on calibration / Stage-1 artifact loading.
    if bool(s2.get("cost_asymmetric", False)) and not s2["ream"]["frequency_weighted_merge"]:
        raise ValueError(
            "stage2_reap_ream.cost_asymmetric=True requires "
            "ream.frequency_weighted_merge=True (spec § 5 step 4T(c)(iii) "
            "/ D-asymmetric-freq)."
        )

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
            from ..stage6alt_thermometer import _thermo_wikitext_tensor
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

        for record in resumed_records:
            ref = record.layer_ref
            _merge_experts_inplace(
                ref, record.grouped, record.freq,
                freq_weighted=s2["ream"]["frequency_weighted_merge"],
                ream_acc=record.resume_ream_acc,
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
    if expert_distill_steps < 0:
        raise ValueError(
            f"stage2_reap_ream.expert_distill_steps={expert_distill_steps}; "
            "must be >= 0 (set 0 to disable)."
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
        "stage2/config/sinkhorn_iters": sinkhorn_iters,
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
    from .plugins.legacy_adapter import LegacyAdapter
    from .plugins.output_space_cost import OutputSpaceCostPlugin
    from .plugins.ream_cost import ReamCostPrePlugin
    from .plugins.ream_cost_post import ReamCostPostPlugin
    from .plugins.reap_scoring import ReapScoringPlugin
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
    # _layer_mean_costs, partial_dir) lives on the LegacyAdapter instance
    # instead — the adapter is constructed once per run() invocation and is
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
    # LegacyAdapter (the `device=device` kwarg below). Readers of
    # run_ctx.get("device") must not expect a torch.device.
    run_ctx.set("device", str(device) if device is not None else "cpu")
    adapter = LegacyAdapter(
        s2_cfg=s2, heal_cfg=heal_cfg,
        heal_device=_heal_device, xd_batches=xd_batches,
        batches=batches, model=model,
        cov_acc=cov_acc, merge_map=merge_map,
        layer_mean_costs=_layer_mean_costs,
        partial_dir=partial_dir,
        max_group_cap=max_group_cap, cost_sigma=cost_sigma,
        cost_bump_ratio=cost_bump_ratio, min_active_tokens=min_active_tokens,
        assignment_solver=assignment_solver, cost_alignment_cfg=cost_alignment_cfg,
        cost_output_token_cap=cost_output_token_cap, cost_whitening=cost_whitening,
        cost_asymmetric=cost_asymmetric, cost_topk_filter=cost_topk_filter,
        capacity_util_threshold=capacity_util_threshold,
        em_refinement_rounds=em_refinement_rounds,
        em_convergence_break=em_convergence_break,
        two_opt_refine=two_opt_refine,
        sinkhorn_epsilon_init=sinkhorn_epsilon_init,
        sinkhorn_epsilon_final=sinkhorn_epsilon_final,
        sinkhorn_iters=sinkhorn_iters,
        skip_merge_percentile=skip_merge_percentile,
        expert_distill_steps=expert_distill_steps,
        expert_distill_lr=expert_distill_lr,
        expert_distill_betas=expert_distill_betas,
        expert_distill_token_cap=expert_distill_token_cap,
        expert_distill_skip_singletons=expert_distill_skip_singletons,
        expert_distill_plateau_steps=expert_distill_plateau_steps,
        expert_distill_plateau_eps=expert_distill_plateau_eps,
        per_layer_target=per_layer_target, blacklist=blacklist,
        artifacts_dir=artifacts_dir, device=device,
    )
    # Registration order matters: ReapScoringPlugin.on_layer_setup must run
    # BEFORE LegacyAdapter.on_layer_setup (which now reads ctx.reap_acc into
    # _profile_layer via on_profile). ``walk_phases`` dispatches each phase to
    # every plugin in sequence order, so listing ReapScoringPlugin first
    # satisfies the dependency.
    #
    # S2-6: the three live cost plugins are registered BETWEEN ReapScoringPlugin
    # and the adapter so they win the ``compute_cost`` ``dispatch_first`` slot
    # over the (now-dead) ``LegacyAdapter.compute_cost`` fallback. Each is
    # constructed with the SAME parsed cost knobs + the SAME ``cov_acc`` object
    # the adapter received. ``registry.enabled(config)`` drops the two cost
    # plugins whose ``is_enabled`` gate is False, leaving exactly one cost
    # plugin (the one matching ``cost_alignment``) ahead of the adapter.
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
    # plugins and BEFORE the adapter so it wins the ``apply_cost_mask``
    # ``dispatch_first`` slot over the (now-dead-for-<100.0) ``LegacyAdapter.
    # apply_cost_mask`` fallback. Constructed directly from the already-parsed
    # and range-validated ``skip_merge_percentile`` local. ``registry.enabled``
    # drops it at the OFF sentinel (>= 100.0), leaving the adapter's sentinel
    # branch to service the slot.
    # S2-8: the five solver plugins are registered AFTER the skip-merge floor
    # plugin and BEFORE the adapter so the enabled one wins the
    # ``solve_assignment`` ``dispatch_first`` slot over the (now-dead)
    # ``LegacyAdapter.solve_assignment`` fallback. Each is constructed with the
    # SAME parsed assignment knobs the adapter received. ``registry.enabled``
    # gates each on ``assignment_solver``, leaving exactly one solver plugin
    # (the one matching the configured solver) ahead of the adapter.
    _solver_plugin_kwargs = dict(
        max_group_cap=max_group_cap,
        assignment_solver=assignment_solver,
        sinkhorn_epsilon_init=sinkhorn_epsilon_init,
        sinkhorn_epsilon_final=sinkhorn_epsilon_final,
        sinkhorn_iters=sinkhorn_iters,
    )
    registry = PluginRegistry([
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
        # plugins and BEFORE the adapter. refine_assignment is a CHAIN
        # (two-opt THEN EM), so unlike the single-winner cost / mask / solve
        # slots both may run — registry order is chain order, two-opt first.
        # Each is constructed from the SAME parsed knobs the adapter received
        # (notably the SAME cov_acc object). registry.enabled drops whichever
        # refiner's gate is off; the neutered LegacyAdapter.refine_assignment
        # trails as a dead fallback that always returns None.
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
        adapter,
    ])
    plugins = registry.enabled(config)
    walk_phases(("on_run_setup",), plugins, run_ctx)
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
