"""Stage-2 per-layer merge spine — the always-on ``LayerMergePlugin``.

S2-12a relocates the SIX live phase hooks of the retired ``LegacyAdapter``
(``on_layer_setup`` / ``on_profile`` / ``merge`` / ``post_merge`` /
``write_artifacts`` / ``on_layer_teardown``) into this one always-on plugin.
Each hook below is a verbatim slice of the legacy loop body, with long
explanatory comments preserved (the original lines are the load-bearing
documentation of the accumulator / merge / artifact semantics).

The dead ``dispatch_first``-slot fallbacks (``compute_cost`` /
``apply_cost_mask`` / ``solve_assignment`` / ``refine_assignment`` /
``pre_merge_snapshot``) are NOT relocated — they stay behind on the (now
100%-dead) ``LegacyAdapter`` until S2-12b deletes that file.

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
        "Stage-2 per-layer merge spine: accumulators, profiling, in-place "
        "merge, kept-set selection, artifacts."
    )
    config_key = "stage2_reap_ream"
    # reads / writes carried forward from LegacyAdapter, trimmed to exactly
    # the ctx slots the SIX live hooks touch (S2-12a). ``provides`` is empty.
    reads: tuple[str, ...] = (
        "layer_ref", "reap_acc", "ream_acc", "layer_input_acc", "perm_cache",
        "target", "freq", "grouped", "protected",
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
        ream_acc = ctx.get("ream_acc")
        perm_cache = ctx.get("perm_cache")

        _merge_experts_inplace(
            layer_ref, grouped, freq,
            freq_weighted=self.s2["ream"]["frequency_weighted_merge"],
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
