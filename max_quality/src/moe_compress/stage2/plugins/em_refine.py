"""EM-style assignment refinement (iterative re-solve under tentative merges).

Paper
-----
Inspiration: Sub-MoE (arXiv:2506.23266) demonstrates iterative
refinement of K-means-style expert merging by re-running assignment
after each tentative merge. This plugin adapts that idea to
**capacitated assignment** (the Stage 2 v2 cost-matrix machinery).

Baseline REAM (arXiv:2604.04356) and the alternative single-shot
solvers (REAP arXiv:2510.13999, GRAPE arXiv:2604.06542) all use
**single-shot** assignment with no iterative refinement.

Official code
-------------
None for this specific Sub-MoE-inspired EM loop. The Sub-MoE paper
(arXiv:2506.23266) is the conceptual reference for the
re-assignment-after-tentative-merge pattern; the implementation in
``_em_refine_assignment`` + ``_em_compute_tentative_weights`` is
project-original adapted for the Stage 2 v2 capacitated-assignment
setting.

Deviation: D-em-refinement
--------------------------
Stage 2 v2 adds ``em_refinement_rounds`` (default ``0``) iterations
of:

  1. Tentatively merge each non-singleton group with the current
     assignment (no model mutation; freq-weighted weights computed
     in-memory).
  2. Recompute the cost matrix against the tentative merged centroids.
  3. Re-solve the assignment.

Stops early on ``em_convergence_break=True`` (default) and assignment
stability. EM is a no-op under ``cost_alignment="pre"`` (the cheap
symmetric cost doesn't depend on centroid weights).

Why this exists
---------------
The merge formula is non-linear in inputs but linear in weights:
``forward(linear_combo(W_e)) ≠ linear_combo(forward(W_e))``. After one
merge, the centroid's weights are no longer the original — a new
assignment under the new centroid weights may produce a lower-cost
matching.

Inspiration vs adaptation: Sub-MoE (arXiv:2506.23266) runs classical
K-means **over expert output vectors** with k-means++ seeding (paper
line 245), cosine-similarity assignment (Eq. 2), and a **uniform**
arithmetic mean of outputs at the centroid-update step (paper line 250,
objective in Eq. 3); a **single-shot** SVD + frequency-weighted V-matrix
merge (Eq. 7, paper line 326) is then applied **after** clustering
converges — Sub-MoE does **not** iteratively re-cluster around tentative
merged weights. Our plugin is *adapted from* (not a re-implementation
of) Sub-MoE: we borrow only the E→M→converge shape and apply it in
**weight space** under capacitated assignment — re-solving the
assignment against tentative **freq-weighted merged centroid weights**
(per-round). The freq weighting itself is paper-attested, but only at
Sub-MoE's V-matrix merge step (Eq. 7), not at its K-means centroid
update (Eq. 3, which is uniform).

Cache invariant: the cached perm becomes stale under tentative
weights, so the inner cost recomputes the perm; the cache is **not**
updated with tentative residuals so the merge step's perm-cache reuse
is preserved. Inside ``_em_compute_tentative_weights`` we also skip
cache reads after round 0 — see HIGH-1 fix in audit history — because
the cached perm was aligned against the *original* centroid weights,
which have drifted under prior EM rounds.

EM is an iterative re-assignment refiner: each round rebuilds the current
groups, computes tentative freq-weighted merged centroid weights, recomputes
the post-alignment cost matrix against those tentative centroids, re-solves the
assignment, and (optionally) breaks on convergence.

Wiring
------
``EmRefinePlugin`` is LIVE as of S2-9: it is the second link of the
``refine_assignment`` slot chain (two-opt THEN EM), registered after
``TwoOptRefinePlugin`` and ahead of the dead-fallback ``LegacyAdapter``.

Circular-import note: this module imports only ``pipeline.base``,
``pipeline.context``, ``pipeline.permutation_align``, ``pipeline.grouping``,
``pipeline.plugins.solver_dispatch``, ``pipeline.plugins.ream_cost`` and
``moe_compress.utils.*`` — none of which import ``stage2_reap_ream`` or
``em_refine``. There is therefore no cycle at module load, and every
import below is a plain module-top import (no function-scope late imports
needed).

Naming-history note
-------------------
The historical "M4" / step-4T(e) labels (STRATEGY_NEXT § 5) are no longer
used in prose; the current plugin architecture has no module-letter
taxonomy. Existing log lines and Trackio keys preserve those identifiers
for dashboard back-compat.

Convergence criterion
---------------------
Early-exit uses strict assignment equality (``new_assignment ==
assignment``). With Sinkhorn solvers a single low-confidence flip can
prevent termination; a Hamming-distance tolerance would be a natural
future-work knob (see MEDIUM-2 in audit history). Kept strict today to
preserve byte-identical determinism with the greedy solver. Note that
Sub-MoE's paper does not use "EM" / "E-step" / "M-step" terminology
(its §3.2 names the four steps "Means Initialization / Clusters
Assignment / Means Update / Convergence"); our "EM" / "M4" naming is an
internal Stage 2 v2 convention, not a paper-traceable label.

Initialization context (D-em-init-reap)
---------------------------------------
This plugin is a *refiner*, not an initializer: it operates on an
assignment already seeded upstream. Sub-MoE explicitly uses k-means++
on expert outputs for centroid seeding (paper line 245); the Stage 2 v2
pipeline initializes from REAP-saliency-ranked centroids via
``select_centroids_by_reap`` (see ``orchestrator.py:344``; REAP itself
is arXiv:2510.13999). The k-means++ → REAP-saliency swap is therefore
an upstream deviation visible to em_refine only as the starting
assignment it inherits.

Pattern H: paper-only re-implementation
---------------------------------------
No upstream code exists for the Sub-MoE-inspired EM loop. Paper
re-verification stamp: 2026-05-28.

Upstream URL audit (2026-05-28):

* ``github.com/lliai/MoERazor`` — paper-cited reference repo →
  HTTP 404 (verified via ``curl``); the repository does not exist.
* ``github.com/siruihan2024/Sub-MoE`` — placeholder repo → HTTP 200,
  but contains only ``README.md`` + ``LICENSE`` (MIT, repo size = 1 KB,
  default_branch = main, last updated 2026-03-14T15:46:59Z, verified via
  GitHub API). No source code is published.

The arXiv:2506.23266 PDF is therefore the sole authoritative source for
Sub-MoE; this plugin is paper-only re-implementation with no upstream
to clean-room from.
"""
from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)

# Per-layer set of layer_idx values for which we have already emitted the
# all-zero-freq fallback warning. Module-global because the warning is a
# one-shot diagnostic (LOW-3), not a per-call counter.
_ZERO_FREQ_WARNED: set[int] = set()


def _em_compute_tentative_weights(
    layer_ref: MoELayerRef,
    grouped: dict[int, list[int]],
    freq: dict[int, int],
    ream_acc: ReamCostAccumulator | None,
    perm_cache: "_PermAlignCache | None",
    em_round: int = 0,
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute the tentative freq-weighted merged centroid weights for every
    non-singleton group, WITHOUT mutating the bank.

    For each centroid c with members [c, m1, m2, ...]:
        W_c_tentative = Σ (freq_e / Σ freq) · perm_e(W_e)

    Permutations come from ``perm_cache`` ONLY on round 0; after round 0 the
    centroid's weights have drifted under prior tentative merges, so the
    cached perm (keyed on the *original* centroid) is stale (HIGH-1 fix).
    From round 1 onward we always recompute the perm against the just-
    refreshed centroid weights via ``_permutation_align_to_centroid``. The
    centroid itself always contributes with identity permutation.

    Used by EM refinement to recompute the cost matrix against the tentative
    merged centroid before reassigning.

    Memory note (LOW-4): we materialize FP32 copies of gate/up/down per
    member per round. The accumulator is kept in FP32 and each contribution
    is cast on add; we deliberately do not pre-allocate a per-name FP32
    buffer because ``banks[name].get(m)`` may return a view we must not alias.
    """
    li = layer_ref.layer_idx
    banks = build_banks(layer_ref)
    out: dict[int, dict[str, torch.Tensor]] = {}

    for centroid, members in grouped.items():
        if len(members) <= 1:
            continue  # singleton — nothing to merge

        weights = np.array([max(freq.get(m, 0), 0) for m in members], dtype=np.float64)
        if weights.sum() <= 0.0:
            # LOW-3: surface the degenerate all-zero-freq case once per layer.
            if li not in _ZERO_FREQ_WARNED:
                log.warning(
                    "em_refine: layer %d centroid %d has all-zero freq across "
                    "members %s — falling back to uniform tentative weights.",
                    li, centroid, list(members),
                )
                _ZERO_FREQ_WARNED.add(li)
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
                # HIGH-1: only trust the perm cache on round 0; after that
                # the cached perm aligns against the *original* centroid,
                # not the drifted tentative centroid.
                cached = (
                    perm_cache.get((li, centroid, m))
                    if perm_cache is not None and em_round == 0
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
                # Cast on add: keep accumulator in FP32, add the FP32-cast contribution.
                contribution = (Wm * w).to(torch.float32)
                accs[name] = contribution if accs[name] is None else accs[name] + contribution

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
    """EM refinement loop.

    For each round r in 1..em_rounds:
      1. Build current groups from ``assignment``.
      2. Compute tentative merged centroid weights (freq-weighted average of
         current group members). On round 0 cached perms are reused; after
         round 0 perms are recomputed against the drifted centroid (HIGH-1).
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
        the (tentative) centroid weights, so EM would be meaningful here; it
        is deferred only because ``_em_refine_assignment`` does not thread the
        per-layer ``layer_inputs`` calibration tensors that
        ``_output_space_cost`` needs.

    Convergence: the strict ``new_assignment == assignment`` check at the
    bottom of the loop can fail to terminate under Sinkhorn when a single
    low-confidence assignment flips back and forth. A Hamming-tolerance
    fallback is future-work (MEDIUM-2 in audit history); the strict check is
    retained today to preserve byte-identical determinism with the greedy
    solver.
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
            layer_ref, grouped, freq, ream_acc, perm_cache, em_round=r,
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
            # LOW-1: the EM guard above forces cost_alignment == "post", so
            # only the cost_asymmetric branch can require freq here.
            freq=freq if cost_asymmetric else None,
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
    """Plugin home for Stage 2 v2 EM refinement.

    LIVE as of S2-9: the second link of the ``refine_assignment`` chain
    (two-opt THEN EM). The orchestrator's ``_run_assignment`` calls this
    plugin's ``refine_assignment`` inside the bump loop, after
    ``TwoOptRefinePlugin`` and ahead of the dead-fallback ``LegacyAdapter``.

    Config gate: enabled iff ``stage2_reap_ream.em_refinement_rounds`` is a
    positive integer. ``em_refinement_rounds`` is a numeric knob (default 0).
    """

    name = "em_refine"
    paper = (
        "EM-style iterative re-assignment under tentative merges. "
        "Inspired by Sub-MoE arXiv:2506.23266 (no official code). "
        "Deviation D-em-refinement vs baseline REAM arXiv:2604.04356 "
        "(single-shot). See module docstring."
    )
    config_key = "stage2_reap_ream.em_refinement_rounds"
    # S2-9: the live refine_assignment slot reads the per-bump scratch slots
    # the orchestrator publishes plus the per-layer cost-alignment slots.
    # NITPICK-3: ``_iter_*`` keys are per-bump scratch slots the orchestrator
    # publishes once per inner loop iteration; the un-prefixed keys are
    # per-layer slots that outlive a single bump. Both flavours appear here.
    reads: tuple[str, ...] = (
        "layer_ref", "ream_acc", "perm_cache", "freq", "protected",
        "_iter_ream_centroid_ids", "_iter_ream_noncentroid_ids",
        "effective_cost_alignment", "effective_cost_asymmetric",
    )
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        em_refinement_rounds: int = 0,
        em_convergence_break: bool = True,
        max_group_cap: int = 0,
        assignment_solver: str = "greedy",
        cost_whitening: str = "none",
        cost_asymmetric: bool = False,
        cost_topk_filter: int = 48,
        skip_merge_percentile: float = 100.0,
        cov_acc=None,
        sinkhorn_epsilon_init: float = 1.0,
        sinkhorn_epsilon_final: float = 0.01,
        sinkhorn_iters: int = 200,
    ) -> None:
        self.em_refinement_rounds = em_refinement_rounds
        self.em_convergence_break = em_convergence_break
        self.max_group_cap = max_group_cap
        self.assignment_solver = assignment_solver
        self.cost_whitening = cost_whitening
        self.cost_asymmetric = cost_asymmetric
        self.cost_topk_filter = cost_topk_filter
        self.skip_merge_percentile = skip_merge_percentile
        self.cov_acc = cov_acc
        self.sinkhorn_epsilon_init = sinkhorn_epsilon_init
        self.sinkhorn_epsilon_final = sinkhorn_epsilon_final
        self.sinkhorn_iters = sinkhorn_iters

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
        """Chain link 2 — Stage 2 v2 EM refinement (LIVE, S2-9).

        Verbatim lift of the EM block from the old
        ``LegacyAdapter.refine_assignment``: reads the per-bump scratch slots
        + per-layer cost-alignment slots off ``ctx``, calls
        ``_em_refine_assignment`` directly, and returns
        ``(assignment, delta, {"em_rounds": em_rounds_done})``. The orchestrator
        reads ``em_rounds`` out of the info dict.
        """
        layer_ref = ctx.get("layer_ref")
        ream_acc = ctx.get("ream_acc")
        perm_cache = ctx.get("perm_cache")
        freq = ctx.get("freq")
        protected = set(ctx.get("protected"))
        ream_centroid_ids = list(ctx.get("_iter_ream_centroid_ids"))
        ream_noncentroid_ids = list(ctx.get("_iter_ream_noncentroid_ids"))
        effective_cost_alignment = ctx.get("effective_cost_alignment")
        effective_cost_asymmetric = ctx.get("effective_cost_asymmetric")

        assignment, delta, em_rounds_done = _em_refine_assignment(
            layer_ref,
            initial_assignment=asg,
            initial_delta=delta,
            skip_merge_percentile=self.skip_merge_percentile,
            ream_centroid_ids=ream_centroid_ids,
            ream_noncentroid_ids=ream_noncentroid_ids,
            perm_cache=perm_cache,
            ream_acc=ream_acc,
            # LOW-2: ``_em_refine_assignment`` early-returns when
            # cost_alignment != "post", so the cov_acc selection is read only
            # in the post branch. The conditional is preserved purely for
            # readability of intent ("cov_acc is a post-only knob"); it has no
            # behavioural effect for the other alignments.
            cov_acc=self.cov_acc if effective_cost_alignment == "post" else None,
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
        return assignment, delta, {"em_rounds": em_rounds_done}
