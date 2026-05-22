"""Single all-in-one Stage 2 plugin holding the legacy per-layer loop body.

Temporary scaffolding plugin for Task 6 of the plugin-architecture refactor:
it gives the universal ``walk_phases`` phase walk something to drive while
preserving byte-identical behaviour vs. the pre-refactor inline loop. Each
phase hook on this class is a verbatim slice of the legacy loop body, with long
explanatory comments preserved (the original lines are the load-bearing
documentation of the assignment / bump / heal / distill semantics).

Tasks T7–T17 peel one algorithm out of this adapter into its own real plugin
(e.g. ``plugins/reap_scoring.py``, ``plugins/solver_greedy.py``,
``plugins/expert_distill.py``); Task T18 deletes this file and the
``MOE_STAGE2_LEGACY_LOOP`` env-var escape hatch in ``stage2_reap_ream.py``.

Run-scope mutable scratchpad (``cov_acc``, ``merge_map``, ``_layer_mean_costs``,
``partial_dir``) lives as instance attributes on this adapter. The adapter is
constructed once per ``run()`` invocation, so the per-adapter scratchpad is
single-run-scoped with no concurrency hazard. Per-layer scratchpad lives on the
per-layer :class:`PipelineContext` (a ``child()`` scope), addressed by named
slots via ``ctx.get`` / ``ctx.set``.
"""
from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...utils.activation_hooks import ReamCostAccumulator
from ...utils.activation_shards import ShardManifest, ShardWriter
from ...utils.model_io import build_banks
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext
from ..grouping import _apply_skip_merge_floor, _promote_orphans
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
from .reap_scoring import select_centroids_by_reap

log = logging.getLogger(__name__)


class LegacyAdapter:
    """All-in-one adapter — every phase hook is a verbatim slice of the legacy loop."""

    name = "legacy_adapter"
    paper = "All-in-one adapter holding the legacy per-layer loop body."
    config_key = "stage2_reap_ream"
    reads: tuple[str, ...] = (
        "layer_ref", "reap_acc", "ream_acc", "layer_input_acc", "perm_cache",
        "target", "scores", "freq", "grouped", "pre_merge_weights", "protected",
        "ream_centroid_ids", "nemo_writer", "xd_writer", "final_kept_ids",
        "heal_state", "distill_state", "n_experts", "n_protected",
        "assigned_cost", "n_assigned", "c_fail", "em_rounds_done",
        "effective_cost_alignment", "effective_cost_asymmetric",
        "capacity_util_value", "effective_target", "mean_assigned_cost",
    )
    writes: tuple[str, ...] = (
        "ream_acc", "perm_cache", "layer_input_acc", "protected",
        "ream_centroid_ids", "ream_noncentroid_ids", "assignment", "delta",
        "grouped", "mean_assigned_cost", "n_protected", "assigned_cost",
        "n_assigned", "b_fail", "c_fail", "em_rounds_done",
        "effective_cost_alignment", "effective_cost_asymmetric",
        "capacity_util_value", "effective_target", "nemo_writer", "xd_writer",
        "pre_merge_weights", "layer_merged", "distill_state", "final_kept_ids",
        "heal_state", "reap_acc",
    )
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always on; pipeline composes exactly [LegacyAdapter] for T6."""
        return True

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def __init__(
        self,
        *,
        s2_cfg: dict[str, Any],
        heal_cfg,
        heal_device,
        xd_batches,
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
        cost_whitening: str,
        cost_asymmetric: bool,
        cost_topk_filter: int,
        capacity_util_threshold: float,
        em_refinement_rounds: int,
        em_convergence_break: bool,
        two_opt_refine: bool,
        sinkhorn_epsilon_init: float,
        sinkhorn_epsilon_final: float,
        sinkhorn_iters: int,
        skip_merge_percentile: float,
        expert_distill_steps: int,
        expert_distill_lr: float,
        expert_distill_betas: tuple[float, float],
        expert_distill_token_cap: int,
        expert_distill_skip_singletons: bool,
        expert_distill_plateau_steps: int,
        expert_distill_plateau_eps: float,
        per_layer_target: dict[int, int],
        blacklist: dict[int, list[int]],
        artifacts_dir: Path,
        device,
    ) -> None:
        # Store every knob the legacy loop body reads. NO logic in __init__ —
        # we are a faithful re-host of the original local variables.
        self.s2 = s2_cfg
        self.heal_cfg = heal_cfg
        self.heal_device = heal_device
        self.xd_batches = xd_batches
        self.batches = batches
        self.model = model
        # Run-scope mutable scratchpad (was held in run()'s local frame).
        # Held here on the adapter instance; in-place mutations on these
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
        self.cost_whitening = cost_whitening
        self.cost_asymmetric = cost_asymmetric
        self.cost_topk_filter = cost_topk_filter
        self.capacity_util_threshold = capacity_util_threshold
        self.em_refinement_rounds = em_refinement_rounds
        self.em_convergence_break = em_convergence_break
        self.two_opt_refine = two_opt_refine
        self.sinkhorn_epsilon_init = sinkhorn_epsilon_init
        self.sinkhorn_epsilon_final = sinkhorn_epsilon_final
        self.sinkhorn_iters = sinkhorn_iters
        self.skip_merge_percentile = skip_merge_percentile
        self.expert_distill_steps = expert_distill_steps
        self.expert_distill_lr = expert_distill_lr
        self.expert_distill_betas = expert_distill_betas
        self.expert_distill_token_cap = expert_distill_token_cap
        self.expert_distill_skip_singletons = expert_distill_skip_singletons
        self.expert_distill_plateau_steps = expert_distill_plateau_steps
        self.expert_distill_plateau_eps = expert_distill_plateau_eps
        self.per_layer_target = per_layer_target
        self.blacklist = blacklist
        self.artifacts_dir = artifacts_dir
        self.device = device

    # ------------------------------------------------------------------
    # Run-scope hooks (no-op; final save lives in run(), not pipeline).
    # ------------------------------------------------------------------
    def on_run_setup(self, run_ctx: PipelineContext) -> None:
        """No-op. Run-scope state was wired in __init__ (the kwargs above)."""

    def on_run_teardown(self, run_ctx: PipelineContext) -> None:
        """No-op. Final checkpoint save happens in run(), not in the pipeline."""

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
        """
        # Look up ``_profile_layer`` via the stage2_reap_ream namespace so
        # existing tests (e.g. test_smoke_stage2_resume.py) that
        # ``monkeypatch.setattr(stage2_reap_ream, "_profile_layer", ...)``
        # still take effect through the pipeline path. The plain
        # module-level import would bind the symbol at import time and
        # bypass the monkey-patch.
        from ... import stage2_reap_ream as _srr
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
    # Phase 4: compute_assignment — DECOMPOSED (S2-5)
    # ------------------------------------------------------------------
    # The monolithic ``compute_assignment`` hook is gone. Its bump-loop body
    # now lives in the orchestrator's module-level ``_run_assignment``, which
    # drives the four fine-grained slots below via ``dispatch_first``. Because
    # the registry is still ``[ReapScoringPlugin(), adapter]`` and
    # ReapScoringPlugin declares none of these four slots, ``dispatch_first``
    # always lands on the verbatim slices below — behaviour is byte-identical
    # to the pre-S2-5 ``compute_assignment``. S2-6+ wires the real
    # cost/solver/refine plugins ahead of this adapter so they win the slot.
    #
    # Each slot is a VERBATIM lift of the matching slice of the old
    # ``compute_assignment``. The orchestrator publishes the per-bump scratch
    # slots (``_iter_ream_centroid_ids`` / ``_iter_ream_noncentroid_ids`` /
    # ``_iter_n_ream_c`` / ``_iter_n_ream_nc``) on ``ctx`` before each call,
    # and these methods read them back. The function-scope late imports are
    # preserved per-method (the ``stage2_reap_ream`` import-cycle reason).
    # ------------------------------------------------------------------
    def compute_cost(self, ctx: PipelineContext):
        """Slot ``compute_cost`` — capacity-util gate + REAM cost matrix.

        DEAD FALLBACK as of S2-6: the three live cost plugins
        (``ReamCostPrePlugin`` / ``ReamCostPostPlugin`` / ``OutputSpaceCostPlugin``)
        are now registered ahead of this adapter and win the ``compute_cost``
        ``dispatch_first`` slot for every configured ``cost_alignment``, so this
        method is never reached on the production path. It is kept intact
        (byte-identical to ``ream_cost._compute_cost_for_plugin``) only as a
        defensive fallback; S2-12 deletes the whole ``LegacyAdapter`` class.

        Verbatim lift of the capacity-utilization gate + ``_ream_cost_matrix``
        call from the old ``compute_assignment`` bump-loop ``if not b_fail``
        branch. Writes ``capacity_util_value`` / ``effective_cost_alignment`` /
        ``effective_cost_asymmetric`` back to ``ctx`` (``overwrite=True``).
        Reads the ``_iter_*`` scratch slots published by the orchestrator.
        Returns the cost matrix ``delta``.
        """
        from .capacity_gate import _pick_effective_alignment
        from .ream_cost import _ream_cost_matrix

        layer_ref = ctx.get("layer_ref")
        ream_acc = ctx.get("ream_acc")
        perm_cache = ctx.get("perm_cache")
        layer_input_acc = ctx.get("layer_input_acc")
        cov_acc = self.cov_acc
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
        if self.max_group_cap <= 0:
            capacity_util_value = 0.0
        else:
            capacity_util_value = n_ream_nc / max(n_ream_c * self.max_group_cap, 1)
        effective_cost_alignment = _pick_effective_alignment(
            n_nc=n_ream_nc,
            n_c=n_ream_c,
            max_group_cap=self.max_group_cap,
            threshold=self.capacity_util_threshold,
            configured=self.cost_alignment_cfg,
        )
        effective_cost_asymmetric = (
            self.cost_asymmetric and effective_cost_alignment == "post"
        )
        ctx.set("capacity_util_value", capacity_util_value, overwrite=True)
        ctx.set("effective_cost_alignment", effective_cost_alignment, overwrite=True)
        ctx.set("effective_cost_asymmetric", effective_cost_asymmetric, overwrite=True)
        delta = _ream_cost_matrix(
            layer_ref, ream_noncentroid_ids, ream_centroid_ids,
            ream_acc=ream_acc,
            blacklisted_ids=protected,
            cost_alignment=effective_cost_alignment,
            cost_whitening=self.cost_whitening,
            cost_asymmetric=effective_cost_asymmetric,
            cost_topk_filter=self.cost_topk_filter,
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
            output_token_cap=self.cost_output_token_cap,
        )
        return delta

    def apply_cost_mask(self, ctx: PipelineContext, delta):
        """Slot ``apply_cost_mask`` — Direction B skip-merge floor.

        Verbatim lift of the skip-merge-floor block from the old
        ``compute_assignment``. When ``skip_merge_percentile < 100.0`` the
        cost matrix is masked and ``(delta, info)`` is returned. At the
        ``100.0`` OFF sentinel the delta object is returned UNCHANGED with no
        copy — matching ``SkipMergeFloorPlugin.apply_cost_mask``'s documented
        sentinel behaviour (the old live path skipped ``_apply_skip_merge_floor``
        entirely at the sentinel). Returns ``(delta, info)``.
        """
        layer_ref = ctx.get("layer_ref")
        # Direction B — skip-merge floor. Mask high-cost pairs to +inf
        # so they fall through to orphan promotion. When the flag is at
        # its OFF sentinel (100.0) this is a no-op on delta's values.
        if self.skip_merge_percentile < 100.0:
            delta, _n_skip_masked = _apply_skip_merge_floor(
                delta, self.skip_merge_percentile,
            )
            if _n_skip_masked > 0:
                log.info(
                    "layer %d: skip-merge floor (P%.1f) masked %d/%d "
                    "cost entries to +inf — affected children fall "
                    "through to orphan promotion",
                    layer_ref.layer_idx, self.skip_merge_percentile,
                    _n_skip_masked, delta.size,
                )
            return delta, {"n_masked": _n_skip_masked,
                           "percentile": self.skip_merge_percentile}
        # OFF sentinel (100.0): return the delta object unchanged, no copy.
        return delta, {"n_masked": 0, "percentile": self.skip_merge_percentile}

    def solve_assignment(self, ctx: PipelineContext, delta):
        """Slot ``solve_assignment`` — child→centroid assignment solver.

        Verbatim lift of the ``_assign_children_to_centroids`` call from the
        old ``compute_assignment``. Reads the ``_iter_n_ream_nc`` /
        ``_iter_n_ream_c`` scratch slots. Returns the assignment list.
        """
        from ...stage2_reap_ream import _assign_children_to_centroids

        n_ream_nc = ctx.get("_iter_n_ream_nc")
        n_ream_c = ctx.get("_iter_n_ream_c")
        assignment = _assign_children_to_centroids(
            delta, n_ream_nc, n_ream_c, self.max_group_cap,
            solver=self.assignment_solver,
            sinkhorn_epsilon_init=self.sinkhorn_epsilon_init,
            sinkhorn_epsilon_final=self.sinkhorn_epsilon_final,
            sinkhorn_iters=self.sinkhorn_iters,
        )
        return assignment

    def refine_assignment(self, ctx: PipelineContext, asg, delta):
        """Slot ``refine_assignment`` — 2-opt local search + EM refinement.

        Verbatim lift of the Direction-D 2-opt block (including the
        greedy-only guard + ``elif`` warning) and the Stage 2 v2 EM
        refinement from the old ``compute_assignment``. Returns
        ``(assignment, delta, em_rounds_done)``.
        """
        from ...stage2_reap_ream import _em_refine_assignment, _two_opt_refine

        layer_ref = ctx.get("layer_ref")
        ream_acc = ctx.get("ream_acc")
        perm_cache = ctx.get("perm_cache")
        cov_acc = self.cov_acc
        freq = ctx.get("freq")
        protected = set(ctx.get("protected"))
        ream_centroid_ids = list(ctx.get("_iter_ream_centroid_ids"))
        ream_noncentroid_ids = list(ctx.get("_iter_ream_noncentroid_ids"))
        effective_cost_alignment = ctx.get("effective_cost_alignment")
        effective_cost_asymmetric = ctx.get("effective_cost_asymmetric")

        assignment = asg
        # Direction D — greedy + 2-opt local refinement (spec §5 step 3.5).
        # Strictly-improving local search; runs only for the greedy solver
        # and only when the flag is set. It cannot regress vs. the greedy
        # assignment, so the EM step below still sees a feasible input.
        if self.two_opt_refine and self.assignment_solver == "greedy":
            assignment = _two_opt_refine(
                assignment, delta, self.max_group_cap,
            )
        elif self.two_opt_refine:
            log.warning(
                "two_opt_refine=true is ignored: it only applies to the "
                "greedy assignment solver, but assignment_solver=%r.",
                self.assignment_solver,
            )
        # Stage 2 v2 EM refinement (spec § 5 step 4T(e) / M4).
        # Runs only when cost_alignment == "post": "pre" is a no-op
        # (cost is centroid-independent) and "output" is deferred (EM
        # would help but needs layer_inputs threaded — see
        # _em_refine_assignment). It guards on this internally.
        assignment, delta, em_rounds_done = _em_refine_assignment(
            layer_ref,
            initial_assignment=assignment,
            initial_delta=delta,
            skip_merge_percentile=self.skip_merge_percentile,
            ream_centroid_ids=ream_centroid_ids,
            ream_noncentroid_ids=ream_noncentroid_ids,
            perm_cache=perm_cache,
            ream_acc=ream_acc,
            cov_acc=cov_acc if effective_cost_alignment == "post" else None,
            freq=freq,
            max_group_cap=self.max_group_cap,
            cost_alignment=effective_cost_alignment,
            cost_whitening=self.cost_whitening,
            cost_asymmetric=effective_cost_asymmetric,
            cost_topk_filter=self.cost_topk_filter,
            assignment_solver=self.assignment_solver,
            em_rounds=self.em_refinement_rounds,
            em_break=self.em_convergence_break,
            blacklisted_ids=protected,
            sinkhorn_epsilon_init=self.sinkhorn_epsilon_init,
            sinkhorn_epsilon_final=self.sinkhorn_epsilon_final,
            sinkhorn_iters=self.sinkhorn_iters,
        )
        return assignment, delta, em_rounds_done

    # ------------------------------------------------------------------
    # Phase 5: pre_merge_snapshot
    # ------------------------------------------------------------------
    def pre_merge_snapshot(self, ctx: PipelineContext) -> None:
        """Snapshot pre-merge expert weights (for distill) + capture mlp I/O (for heal).

        Verbatim slice of lines 1146–1198 of stage2_reap_ream.run() (pre-T6).
        """
        from ...stage2_reap_ream import (
            _capture_mlp_io,
            _snapshot_pre_merge_layer_experts,
        )

        layer_ref = ctx.get("layer_ref")
        grouped = ctx.get("grouped")

        # Stage-2 merge-heal: capture this layer's pre-merge (input, target)
        # pairs for self-distillation. Done BEFORE _merge_experts_inplace,
        # while layer_ref.mlp is still its original self — so the captured
        # output is the self-distillation target. Skipped for 0-merge layers:
        # a layer that merged nothing is unchanged, so the heal would be a
        # guaranteed no-op the accept/reject guard rejects anyway.
        #
        # Captures stream to bf16 safetensors shards on disk so peak RAM is
        # bounded by a small LRU cache, NOT by ``token_cap``. One layer at a
        # time = bounded total disk use. Companion ``shared_*`` shards are
        # computed AFTER bank.select / router resize (the shared expert is
        # untouched by the merge, so timing is purely a convenience).
        nemo_writer: ShardWriter | None = None
        xd_writer: ShardWriter | None = None
        layer_merged = any(len(m) > 1 for m in grouped.values())
        if self.heal_cfg.enabled and layer_merged:
            heal_shard_root = (
                Path(self.heal_cfg.shard_dir) if self.heal_cfg.shard_dir
                else self.artifacts_dir / "_stage2_heal_shards"
            )
            layer_shard_dir = heal_shard_root / f"layer_{layer_ref.layer_idx}"
            hidden_dim = layer_ref.router.weight.shape[-1]
            nemo_writer = ShardWriter(
                layer_shard_dir / "nemo",
                layer_idx=layer_ref.layer_idx,
                hidden_dim=hidden_dim,
                shard_rows=self.heal_cfg.shard_rows,
            )
            _capture_mlp_io(
                self.model, layer_ref, self.batches,
                device=self.heal_device, pool_size=self.heal_cfg.token_cap,
                shard_writer=nemo_writer,
            )
            if self.xd_batches is not None:
                xd_writer = ShardWriter(
                    layer_shard_dir / "xd",
                    layer_idx=layer_ref.layer_idx,
                    hidden_dim=hidden_dim,
                    shard_rows=self.heal_cfg.shard_rows,
                )
                _capture_mlp_io(
                    self.model, layer_ref, self.xd_batches,
                    device=self.heal_device, pool_size=self.heal_cfg.xd_holdout_tokens,
                    shard_writer=xd_writer,
                )

        # Phase 3 (M8): snapshot pre-merge expert weights BEFORE the merge
        # mutates the bank. The snapshot is consumed only by the per-group
        # distillation step below; released as soon as that finishes for
        # this layer (Python GC since no module-level reference is held).
        pre_merge_weights: dict[int, dict[str, torch.Tensor]] | None = None
        if self.expert_distill_steps > 0:
            pre_merge_weights = _snapshot_pre_merge_layer_experts(layer_ref)

        ctx.set("nemo_writer", nemo_writer)
        ctx.set("xd_writer", xd_writer)
        ctx.set("pre_merge_weights", pre_merge_weights)
        ctx.set("layer_merged", layer_merged)

    # ------------------------------------------------------------------
    # Phase 6: merge
    # ------------------------------------------------------------------
    def merge(self, ctx: PipelineContext) -> None:
        """Merge experts in place + per-group distillation.

        Verbatim slice of lines 1200–1244 of stage2_reap_ream.run() (pre-T6).
        """
        from ...stage2_reap_ream import _distill_merged_group

        layer_ref = ctx.get("layer_ref")
        grouped = ctx.get("grouped")
        freq = ctx.get("freq")
        ream_acc = ctx.get("ream_acc")
        perm_cache = ctx.get("perm_cache")
        layer_input_acc = ctx.get("layer_input_acc")
        pre_merge_weights = ctx.get("pre_merge_weights")

        _merge_experts_inplace(
            layer_ref, grouped, freq,
            freq_weighted=self.s2["ream"]["frequency_weighted_merge"],
            ream_acc=ream_acc,
            perm_cache=perm_cache,
        )

        # Phase 3 (M8): per-merge-group expert distillation (spec § 5 step 7b).
        distill_state: dict[int, dict] | None = None
        if self.expert_distill_steps > 0 and pre_merge_weights is not None:
            layer_inputs_buf = (
                layer_input_acc.get() if layer_input_acc is not None else None
            )
            if layer_inputs_buf is None or layer_inputs_buf.shape[0] == 0:
                log.warning(
                    "layer %d: expert distillation enabled but no layer-input "
                    "samples were captured during profile — skipping.",
                    layer_ref.layer_idx,
                )
            else:
                distill_state = {}
                target_device = layer_ref.layer_module.parameters().__next__().device
                for centroid, members in grouped.items():
                    if self.expert_distill_skip_singletons and len(members) <= 1:
                        continue
                    state = _distill_merged_group(
                        layer_ref=layer_ref,
                        centroid_id=centroid,
                        members=members,
                        freq=freq,
                        pre_merge_weights=pre_merge_weights,
                        layer_inputs=layer_inputs_buf,
                        steps=self.expert_distill_steps,
                        lr=self.expert_distill_lr,
                        betas=self.expert_distill_betas,
                        plateau_steps=self.expert_distill_plateau_steps,
                        plateau_eps=self.expert_distill_plateau_eps,
                        token_cap=self.expert_distill_token_cap,
                        device=target_device,
                    )
                    distill_state[centroid] = state
                log.info(
                    "  layer %d distillation: %d non-singleton groups distilled",
                    layer_ref.layer_idx, len(distill_state),
                )

        ctx.set("distill_state", distill_state)

    # ------------------------------------------------------------------
    # Phase 7: post_merge
    # ------------------------------------------------------------------
    def post_merge(self, ctx: PipelineContext) -> None:
        """bank.select, router resize, optional merge-heal.

        Verbatim slice of lines 1246–1325 of stage2_reap_ream.run() (pre-T6).
        """
        from ...stage2_reap_ream import (
            _heal_layer,
            _make_shared_out_fn,
        )

        layer_ref = ctx.get("layer_ref")
        protected = list(ctx.get("protected"))
        ream_centroid_ids = list(ctx.get("ream_centroid_ids"))
        nemo_writer: ShardWriter | None = ctx.get("nemo_writer")
        xd_writer: ShardWriter | None = ctx.get("xd_writer")

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

        # Stage-2 per-layer merge-heal (opt-in). Heal this layer's kept
        # experts (+ optionally the router) by self-distillation toward its
        # OWN pre-merge MoE-block output — right after the router resize,
        # BEFORE the checkpoint block so the persisted weights and the
        # heal_state field reflect the healed layer.
        heal_state: dict | None = None
        if self.heal_cfg.enabled:
            # `nemo_writer is not None` iff this layer had merges
            # (`heal_cfg.enabled and layer_merged` above). `_capture_mlp_io`
            # raises when it captures 0 rows, so reaching this point with a
            # non-None writer means there's something to heal — the
            # `n_captured > 0` sub-condition is dead code.
            if nemo_writer is not None:
                try:
                    # The shared expert is Stage-2 protected (untouched by
                    # merge/bank.select/resize), so we can run it on the captured
                    # inputs now and store the result in companion shards. Then
                    # finalize each writer with a layer-idx-seeded whole-shard
                    # 90/10 split (controlled by ``holdout_fraction``).
                    _shared_fn = _make_shared_out_fn(layer_ref)
                    nemo_writer.compute_shared_companions(_shared_fn)
                    nemo_manifest = nemo_writer.finalize(
                        split_ratio=1.0 - self.heal_cfg.holdout_fraction,
                        seed=layer_ref.layer_idx,
                    )
                    xd_manifest: ShardManifest | None = None
                    if xd_writer is not None:
                        xd_writer.compute_shared_companions(_shared_fn)
                        xd_manifest = xd_writer.finalize(
                            split_ratio=1.0 - self.heal_cfg.holdout_fraction,
                            seed=layer_ref.layer_idx,
                        )
                    heal_state = _heal_layer(
                        layer_ref=layer_ref,
                        final_kept_ids=final_kept_ids,
                        manifest=nemo_manifest,
                        manifest_dir=nemo_writer.out_dir,
                        xd_manifest=xd_manifest,
                        xd_manifest_dir=(
                            xd_writer.out_dir if xd_writer is not None else None
                        ),
                        heal_cfg=self.heal_cfg,
                        device=(
                            self.device if self.device is not None
                            else layer_ref.router.weight.device
                        ),
                    )
                finally:
                    # Bounded disk use: drop the layer's shard dir even on
                    # exception. `cleanup()` is idempotent and safe when the
                    # writer never created its out_dir (lazy mkdir).
                    # `keep_shards=True` opts into debugging mode and keeps
                    # shards on disk.
                    if not self.heal_cfg.keep_shards:
                        nemo_writer.cleanup()
                        if xd_writer is not None:
                            xd_writer.cleanup()
            else:
                # nemo_writer is None => layer had 0 merges (the layer is
                # unchanged, so there is nothing to heal).
                log.info(
                    "  merge-heal layer %d: skipped (0 merges — layer "
                    "unchanged, nothing to heal)",
                    layer_ref.layer_idx,
                )

        ctx.set("final_kept_ids", tuple(final_kept_ids))
        ctx.set("heal_state", heal_state)

    # ------------------------------------------------------------------
    # Phase 8: write_artifacts
    # ------------------------------------------------------------------
    def write_artifacts(self, ctx: PipelineContext) -> dict[str, Any]:
        """Mutate run-scope merge_map; cov remap; write partial JSON + .pt.

        Verbatim slice of lines 1327–1409 of stage2_reap_ream.run() (pre-T6).
        ``partial_dir`` is read from the per-layer context slot
        (``ctx.get("partial_dir")``, set on the run-scope context by the
        orchestrator and inherited by the layer child); it is ``None`` in
        no-resume mode. The ``self.partial_dir`` instance attribute is kept
        for the resume path elsewhere in the adapter.
        """
        from ...stage2_reap_ream import _summarize_distill_state

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
        # Drop large transient writers / accumulators as well so gc.collect()
        # can reclaim their underlying buffers immediately.
        ctx.set("nemo_writer", None, overwrite=True)
        ctx.set("xd_writer", None, overwrite=True)
        gc.collect()
        torch.cuda.empty_cache()
