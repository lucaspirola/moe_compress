"""EM refinement (Task 15 of the plugin-architecture refactor).

Home of ``_em_refine_assignment`` — the Stage 2 v2 EM refinement loop
(spec § 5 step 4T(e) / M4) — and its helper ``_em_compute_tentative_weights``.
Both moved verbatim out of ``stage2_reap_ream.py``; that module re-imports them
so external callers and tests keep their existing import paths.

EM is an iterative re-assignment refiner: each round rebuilds the current
groups, computes tentative freq-weighted merged centroid weights, recomputes
the post-alignment cost matrix against those tentative centroids, re-solves the
assignment, and (optionally) breaks on convergence.

Circular-import note: this module imports only ``pipeline.base``,
``pipeline.context``, ``pipeline.permutation_align``, ``pipeline.grouping``,
``pipeline.plugins.solver_dispatch``, ``pipeline.plugins.ream_cost`` and
``moe_compress.utils.*`` — none of which import ``stage2_reap_ream`` or
``em_refine``. There is therefore no cycle at module load, and every import
below is a plain module-top import (no function-scope late imports needed).

``EmRefinePlugin`` is a scaffold-only plugin — not yet on the live
phase walk (``LegacyAdapter.compute_assignment`` still calls
``_em_refine_assignment`` directly inside the bump loop, and the
``MOE_STAGE2_LEGACY_LOOP=1`` path does too); it gives T18 a per-refiner plugin
to wire into the decomposed ``refine_assignment`` phase.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...utils.activation_hooks import (
    InputCovarianceAccumulator,  # noqa: F401 — resolves the string type hint
    ReamCostAccumulator,
)
from ...utils.model_io import MoELayerRef, build_banks
from ...pipeline.context import PipelineContext
from ..grouping import _apply_skip_merge_floor, _build_grouped_from_assignment
from ..permutation_align import (
    _PermAlignCache,  # noqa: F401 — resolves the string type hint
    _permutation_align_to_centroid,
)
from .ream_cost import _ream_cost_matrix
from .solver_dispatch import (
    SolverName,  # noqa: F401 — resolves the string type hint
    _assign_children_to_centroids,
)


def _em_compute_tentative_weights(
    layer_ref: MoELayerRef,
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    ream_acc: ReamCostAccumulator | None,
    perm_cache: "_PermAlignCache | None",
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute the tentative freq-weighted merged centroid weights for every
    non-singleton group, WITHOUT mutating the bank.

    For each centroid c with members [c, m1, m2, ...]:
        W_c_tentative = Σ (freq_e / Σ freq) · perm_e(W_e)

    Permutations come from ``perm_cache`` if available; otherwise computed
    fresh via ``_permutation_align_to_centroid`` (the centroid contributes
    with identity permutation).

    Used by EM refinement (spec § 5 step 4T(e)) to recompute the cost matrix
    against the tentative merged centroid before reassigning.
    """
    li = layer_ref.layer_idx
    banks = build_banks(layer_ref)
    out: dict[int, dict[str, torch.Tensor]] = {}

    for centroid, members in grouped.items():
        if len(members) <= 1:
            continue  # singleton — nothing to merge

        weights = np.array([max(freq.get(m, 0), 0) for m in members], dtype=np.float64)
        if weights.sum() <= 0.0:
            weights[:] = 1.0
        weights /= weights.sum()

        ref_gate = banks["gate_proj"].get(centroid).to(torch.float32)
        ref_up   = banks["up_proj"].get(centroid).to(torch.float32)
        ref_act  = ream_acc.get_neuron_mean(li, centroid) if ream_acc else None

        accs: dict[str, torch.Tensor | None] = {name: None for name in banks}
        for w, m in zip(weights, members):
            gate_m = banks["gate_proj"].get(m).to(torch.float32)
            up_m   = banks["up_proj"].get(m).to(torch.float32)
            child_act = ream_acc.get_neuron_mean(li, m) if ream_acc else None

            if m == centroid:
                perm = None
            else:
                cached = (
                    perm_cache.get((li, centroid, m))
                    if perm_cache is not None
                    else None
                )
                if cached is not None:
                    perm = cached[0]
                else:
                    perm = _permutation_align_to_centroid(
                        ref_gate, ref_up, gate_m, up_m,
                        ref_act_mean=ref_act, child_act_mean=child_act,
                    )

            for name, bank in banks.items():
                if name == "gate_proj":
                    Wm = gate_m
                elif name == "up_proj":
                    Wm = up_m
                else:
                    Wm = bank.get(m).to(torch.float32)
                if perm is not None:
                    Wm = Wm[perm, :] if name in ("gate_proj", "up_proj") else Wm[:, perm]
                accs[name] = Wm * w if accs[name] is None else accs[name] + Wm * w

        out[centroid] = {name: accs[name] for name in banks}

    return out


def _em_refine_assignment(
    layer_ref: MoELayerRef,
    *,
    initial_assignment: list[int],
    initial_delta: np.ndarray,
    ream_centroid_ids: list[int],
    ream_noncentroid_ids: list[int],
    perm_cache: "_PermAlignCache",
    ream_acc: ReamCostAccumulator,
    cov_acc: "InputCovarianceAccumulator | None",
    freq: dict[int, int],
    max_group_cap: int,
    cost_alignment: str,
    cost_whitening: str,
    cost_asymmetric: bool,
    cost_topk_filter: int,
    assignment_solver: SolverName,
    em_rounds: int,
    em_break: bool,
    blacklisted_ids: set[int] | None,
    sinkhorn_epsilon_init: float = 1.0,
    sinkhorn_epsilon_final: float = 0.01,
    sinkhorn_iters: int = 200,
    skip_merge_percentile: float = 100.0,
) -> tuple[list[int], np.ndarray, int]:
    """EM refinement loop (spec § 5 step 4T(e) / M4).

    For each round r in 1..em_rounds:
      1. Build current groups from ``assignment``.
      2. Compute tentative merged centroid weights (freq-weighted average of
         current group members, using cached perms where available).
      3. Recompute the cost matrix with the tentative centroids substituted.
      4. Re-solve the assignment.
      5. If ``em_break`` and the new assignment equals the old, stop early.

    Returns ``(final_assignment, final_delta, rounds_completed)``. ``rounds_completed``
    is the number of rounds where step 4 actually ran (≥ 1 if em_rounds ≥ 1).

    EM is a no-op when:
      - ``em_rounds <= 0``
      - ``cost_alignment == "pre"`` (the cheap symmetric cost does not depend
        on centroid weights, so a tentative merge does not change the cost
        matrix and the assignment cannot improve).
      - ``cost_alignment == "output"`` — the output-space cost *does* depend on
        the (tentative) centroid weights, so EM would be meaningful here; it is
        deferred only because ``_em_refine_assignment`` does not thread the
        per-layer ``layer_inputs`` calibration tensors that ``_output_space_cost``
        needs. See the TODO at the cost-matrix recompute below.
    """
    if em_rounds <= 0 or cost_alignment != "post":
        return initial_assignment, initial_delta, 0

    n_nc = len(ream_noncentroid_ids)
    n_c = len(ream_centroid_ids)
    assignment = list(initial_assignment)
    delta = initial_delta
    rounds_done = 0

    for r in range(em_rounds):
        grouped = _build_grouped_from_assignment(
            assignment, ream_centroid_ids, ream_noncentroid_ids,
        )
        tentative = _em_compute_tentative_weights(
            layer_ref, grouped, freq, ream_acc, perm_cache,
        )
        if not tentative:
            # No non-singleton groups → tentative is identical to original →
            # cost matrix would be unchanged. Stop early.
            break

        new_delta = _ream_cost_matrix(
            layer_ref, ream_noncentroid_ids, ream_centroid_ids,
            ream_acc=ream_acc,
            blacklisted_ids=blacklisted_ids,
            cost_alignment=cost_alignment,
            cost_whitening=cost_whitening,
            cost_asymmetric=cost_asymmetric,
            cost_topk_filter=cost_topk_filter,
            # freq is also needed by the "output" cost (freq-weighted tentative
            # merge), not just the asymmetric "post" cost — keep consistent with
            # the main _ream_cost_matrix call site. TODO: admitting "output" to
            # the EM guard above additionally requires threading layer_inputs
            # here so _output_space_cost has its calibration tokens.
            freq=freq if (cost_asymmetric or cost_alignment == "output") else None,
            cov_acc=cov_acc,
            perm_cache=perm_cache,
            tentative_centroid_weights=tentative,
        )
        # Direction B — re-apply the skip-merge floor each EM round; the freshly
        # recomputed cost matrix would otherwise un-mask the high-cost pairs.
        if skip_merge_percentile < 100.0:
            new_delta, _ = _apply_skip_merge_floor(new_delta, skip_merge_percentile)
        new_assignment = _assign_children_to_centroids(
            new_delta, n_nc, n_c, max_group_cap,
            solver=assignment_solver,
            sinkhorn_epsilon_init=sinkhorn_epsilon_init,
            sinkhorn_epsilon_final=sinkhorn_epsilon_final,
            sinkhorn_iters=sinkhorn_iters,
        )
        rounds_done = r + 1
        # F2 fix: commit ``delta = new_delta`` BEFORE the break check so
        # downstream assigned_cost reporting uses the EM-refined cost matrix
        # even when the assignment converged this round.
        delta = new_delta
        if em_break and new_assignment == assignment:
            break
        assignment = new_assignment

    return assignment, delta, rounds_done


class EmRefinePlugin:
    """Plugin home for Stage 2 v2 EM refinement (spec § 5 step 4T(e) / M4).

    T15 status: scaffold only — NOT on the live phase walk.
    ``LegacyAdapter.compute_assignment`` still calls ``_em_refine_assignment``
    directly inside the bump loop, and the ``MOE_STAGE2_LEGACY_LOOP=1`` path in
    ``stage2_reap_ream.run()`` does too. This class exists so T18 has a
    per-refiner plugin to wire into the decomposed ``refine_assignment`` phase.

    Config gate: enabled iff ``stage2_reap_ream.em_refinement_rounds`` is a
    positive integer. ``em_refinement_rounds`` is a numeric knob (default 0).
    """

    name = "em_refine"
    paper = "Stage 2 v2 EM refinement loop (spec § 5 step 4T(e) / M4)."
    config_key = "stage2_reap_ream.em_refinement_rounds"
    # () until a later task wires the live hook
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """True iff ``stage2_reap_ream.em_refinement_rounds`` > 0.

        Defaults to 0 (EM off) → a missing key / block leaves the plugin
        disabled. Coerced via ``int(...)`` to match the ``em_rounds <= 0``
        guard inside ``_em_refine_assignment``; a non-numeric value falls back
        to disabled rather than crashing config discovery.
        """
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        try:
            return int(s2.get("em_refinement_rounds", 0)) > 0
        except (TypeError, ValueError):
            return False

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def refine_assignment(
        self, ctx: PipelineContext, asg: Any, delta: Any
    ) -> tuple[Any, Any, dict] | None:
        """Documented no-op for T15.

        The live EM call still belongs to the LegacyAdapter bump loop (and the
        ``MOE_STAGE2_LEGACY_LOOP=1`` legacy-loop path), which invoke
        ``_em_refine_assignment`` directly. Returning ``None`` makes
        ``PluginRegistry.dispatch_first`` skip this plugin cleanly. T18 wires
        the real call here once ``compute_assignment`` is decomposed into the
        fine-grained phase walk.
        """
        return None
