"""Stage 2 ``LegacyAdapter`` — 100% DEAD CODE as of S2-12a, pending S2-12b deletion.

S2-12a relocated the SIX live phase hooks (``on_layer_setup`` / ``on_profile``
/ ``merge`` / ``post_merge`` / ``write_artifacts`` / ``on_layer_teardown``) and
dropped the two no-op run-scope hooks (``on_run_setup`` / ``on_run_teardown``)
out of this class into the always-on ``plugins/layer_merge.LayerMergePlugin``.
What remains here is pure dead code: the class is still constructed and still
registered in the orchestrator's ``PluginRegistry`` ONLY so the byte-identical
gate can prove the relocation is exact before S2-12b removes this file.

The class now keeps ONLY:
  * metadata attrs (``name`` / ``paper`` / ``config_key`` / ``reads`` /
    ``writes`` / ``provides``);
  * ``__init__`` (still constructed by the orchestrator);
  * ``is_enabled`` / ``contribute_artifact``;
  * the dead ``dispatch_first``-slot fallbacks ``compute_cost`` /
    ``apply_cost_mask`` / ``solve_assignment`` / ``refine_assignment`` /
    ``pre_merge_snapshot`` — never reached on the production path (the live
    cost / solver / refine plugins win every slot ahead of this adapter).

S2-12b deletes this entire file.

Run-scope mutable scratchpad (``cov_acc``, ``merge_map``, ``_layer_mean_costs``,
``partial_dir``) lives as instance attributes on this adapter. The adapter is
constructed once per ``run()`` invocation, so the per-adapter scratchpad is
single-run-scoped with no concurrency hazard. Per-layer scratchpad lives on the
per-layer :class:`PipelineContext` (a ``child()`` scope), addressed by named
slots via ``ctx.get`` / ``ctx.set``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...pipeline.context import PipelineContext
from ..grouping import _apply_skip_merge_floor

log = logging.getLogger(__name__)


class LegacyAdapter:
    """100% DEAD CODE as of S2-12a — only the dead slot fallbacks remain (S2-12b deletes)."""

    name = "legacy_adapter"
    paper = "All-in-one adapter holding the legacy per-layer loop body."
    config_key = "stage2_reap_ream"
    # S2-11: ``pre_merge_weights`` / ``nemo_writer`` / ``xd_writer`` /
    # ``layer_merged`` dropped — those slots are owned by ExpertDistillPlugin /
    # MergeHealPlugin now. The adapter still READS ``heal_state`` /
    # ``distill_state`` in ``write_artifacts`` and WRITES them as defaults in
    # ``merge`` / ``post_merge`` (the two plugins overwrite the live values).
    reads: tuple[str, ...] = (
        "layer_ref", "reap_acc", "ream_acc", "layer_input_acc", "perm_cache",
        "target", "scores", "freq", "grouped", "protected",
        "ream_centroid_ids", "final_kept_ids",
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
        "capacity_util_value", "effective_target",
        "distill_state", "final_kept_ids", "heal_state", "reap_acc",
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
    # S2-12a: ``on_layer_setup`` / ``on_profile`` (and the two no-op run-scope
    # hooks ``on_run_setup`` / ``on_run_teardown``) were RELOCATED to
    # ``plugins/layer_merge.LayerMergePlugin``. Only the dead ``dispatch_first``
    # slot fallbacks remain below.
    # ------------------------------------------------------------------

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

        DEAD FALLBACK as of S2-7 for ``< 100.0``; still services the
        ``>= 100.0`` sentinel until S2-12. The live ``SkipMergeFloorPlugin`` is
        registered ahead of this adapter and wins the ``apply_cost_mask`` slot
        whenever it is enabled (``skip_merge_percentile < 100.0``); at the OFF
        sentinel that plugin is dropped by ``registry.enabled`` and this
        method's sentinel branch services the slot.

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

        DEAD FALLBACK as of S2-8 — the five solver plugins
        (``GreedySolverPlugin`` / ``HungarianSolverPlugin`` / ``McfSolverPlugin``
        / ``SinkhornSolverPlugin`` / ``AutoSolverPlugin``) are registered ahead
        of this adapter and one always wins the ``solve_assignment``
        ``dispatch_first`` slot; ``assignment_solver`` is validated to one of
        those five so this method is unreachable on the production path. Kept
        intact (byte-identical to ``solver_dispatch._solve_for_plugin``) only as
        a defensive fallback; S2-12 deletes the whole ``LegacyAdapter`` class.

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
        """Slot ``refine_assignment`` — DEAD FALLBACK as of S2-9.

        ``refine_assignment`` is a CHAIN serviced by ``TwoOptRefinePlugin``
        then ``EmRefinePlugin``, both registered ahead of this adapter; a chain
        calls every enabled plugin's ``refine_assignment`` (no
        ``dispatch_first`` early-return), so this adapter must decline the slot
        — otherwise the 2-opt + EM work would run a SECOND time. Returns
        ``None`` to decline. The method is kept (a drift-guard test asserts it
        is ``callable``); S2-12 deletes the whole ``LegacyAdapter`` class.
        """
        return None

    # ------------------------------------------------------------------
    # Phase 5: pre_merge_snapshot
    # ------------------------------------------------------------------
    def pre_merge_snapshot(self, ctx: PipelineContext) -> None:
        """DEAD as of S2-11 — ``pre_merge_snapshot`` is a ``walk_phases`` phase;
        ``ExpertDistillPlugin`` + ``MergeHealPlugin`` own it; this declines to
        avoid double-run; S2-12 deletes the class.
        """
        return None

    # ------------------------------------------------------------------
    # S2-12a: ``merge`` / ``post_merge`` / ``write_artifacts`` /
    # ``on_layer_teardown`` were RELOCATED VERBATIM to
    # ``plugins/layer_merge.LayerMergePlugin``. Nothing of the per-layer merge
    # spine remains on this class — only the dead slot fallbacks above.
    # ------------------------------------------------------------------
