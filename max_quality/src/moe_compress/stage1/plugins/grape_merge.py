"""Phase F — GRAPE Algorithm 1 (entropy-aware greedy merge with restart).

Paper: 2604.06542 §3.3. Migrated from the legacy Stage 1 module (Phase-F
block) in sub-task 3 of the Stage 1 → plugin-architecture refactor.

The plugin's externally observable behaviour is **byte-identical** to the
legacy inline Phase F: same per-layer budget allocation, same redundancy
normalisation, same artifact schema. Verified via the golden snapshot
test.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from ...utils.trackio_log import trackio_flush as _trackio_flush
from ...utils.trackio_log import trackio_log as _trackio_log
from ..context import Stage1Context

log = logging.getLogger(__name__)


class GrapeMergePlugin:
    """GRAPE greedy merge plugin (Phase F).

    Reads CKA distance matrices, the SE blacklist, per-layer expert counts,
    and the global budget; writes per-layer target counts + redundancy +
    achieved/requested budgets. Contributes the ``stage1_budgets.json``
    payload via :meth:`contribute_artifact`.
    """

    name: str = "grape_merge"
    paper: str = "Liu et al., GRAPE 2024 (arXiv 2604.06542), Algorithm 1"
    config_key: str = "stage1_grape"
    reads: tuple[str, ...] = (
        "D_matrices",
        "blacklist",
        "per_layer_targets",
        "decomposition",
        "config",
    )
    writes: tuple[str, ...] = (
        "per_layer_target_experts",
        "per_layer_redundancy",
        "achieved_budget",
        "requested_budget",
        "grape_config",
    )
    # Phase F is a pure post-processing step over D + blacklist.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """GRAPE is mandatory — every Stage 1 run executes Phase F."""
        return True

    def run(self, ctx: Stage1Context) -> None:
        """Execute Phase F end-to-end.

        Reads slots ``D_matrices``, ``blacklist``, ``per_layer_targets``,
        ``decomposition``, ``config`` from ``ctx``;
        writes ``per_layer_target_experts``, ``per_layer_redundancy``,
        ``achieved_budget``, ``requested_budget``, ``grape_config`` back.

        Emits per-layer Trackio metrics + GRAPE summary identically to
        the legacy inline block.
        """
        D_matrices: dict[int, torch.Tensor] = ctx.get("D_matrices")
        blacklist: dict[int, list[int]] = ctx.get("blacklist")
        per_layer_counts: dict[int, int] = ctx.get("per_layer_targets")
        decomposition = ctx.get("decomposition")
        config: dict = ctx.get("config")
        s1 = config["stage1_grape"]

        global_budget = decomposition.global_expert_budget
        gamma = float(s1.get("entropy_tolerance", 0.1))

        # Direction-A second-phase knobs. Both default to a strict no-op:
        # `grape_floor_divisor` absent -> 2 (the spec §12 D5 `N // 2` floor);
        # `merge_cost_prior` absent -> None (selection stays `min R[li]`).
        # When the config keys are unset the GRAPE output is byte-identical
        # to the historical behaviour.
        grape_floor_divisor = int(s1.get("grape_floor_divisor", 2))
        if grape_floor_divisor > 2:
            log.warning(
                "Stage 1: grape_floor_divisor=%d > 2 — GRAPE may drive layers "
                "BELOW the N//2 spec invariant. This is an opt-in, unvalidated "
                "quality regime intended for the Direction-A second Stage-1 pass.",
                grape_floor_divisor,
            )
        elif grape_floor_divisor < 2:
            log.warning(
                "Stage 1: grape_floor_divisor=%d (< 2) sets the per-layer floor to "
                "(near) the full expert count — GRAPE can merge little or nothing. "
                "Almost certainly a misconfiguration.",
                grape_floor_divisor,
            )
        raw_cost_prior = s1.get("merge_cost_prior")
        merge_cost_prior: dict[int, float] | None = None
        if raw_cost_prior:
            # Config stores it as {str(layer_idx): cost}; GRAPE keys layers
            # by int. Every layer GRAPE will consider must have a prior
            # entry.
            merge_cost_prior = {int(k): float(v) for k, v in raw_cost_prior.items()}
            missing = sorted(set(per_layer_counts) - set(merge_cost_prior))
            if missing:
                raise ValueError(
                    f"merge_cost_prior is missing entries for layers {missing}; "
                    "the prior must cover every MoE layer GRAPE allocates."
                )
            log.info(
                "Stage 1: merge_cost_prior supplied for %d layers — GRAPE "
                "best-layer selection minimises R[li] * cost_prior[li].",
                len(merge_cost_prior),
            )

        budgets = _grape_greedy_merge(
            D_matrices=D_matrices,
            global_budget=global_budget,
            per_layer_counts=per_layer_counts,
            blacklist=blacklist,
            gamma=gamma,
            floor_divisor=grape_floor_divisor,
            merge_cost_prior=merge_cost_prior,
        )

        # Logging: per-layer redundancy R̃^l (spec §4, Eq. 3).
        # Build D_work_logging: zero blacklisted rows/cols in a copy of
        # D_matrices (mirrors what _grape_greedy_merge does internally,
        # for consistency).
        D_work_logging: dict[int, np.ndarray] = {
            li: _zero_blacklisted(D.cpu().numpy().copy(), blacklist.get(li, []))
            for li, D in D_matrices.items()
        }

        # R^l = Σ_{i≠j} D^l_{ij}  (sum of off-diagonal distances)
        R_raw: dict[int, float] = {}
        for li, d in D_work_logging.items():
            n = d.shape[0]
            R_raw[li] = float(d.sum() - np.diag(d).sum()) if n > 1 else 0.0

        # Min-max normalise across layers → R̃^l ∈ [0, 1].
        r_min = min(R_raw.values())
        r_max = max(R_raw.values())
        if r_max > r_min:
            denom = r_max - r_min
        else:
            # All layers have identical R_raw — redundancy collapses to
            # 0.0 for all. This is expected and benign for single-layer
            # models (only one data point, so min-max normalisation is
            # undefined and R̃^l=0 for all layers). Also expected when
            # all layers have identical distance-sum profiles (e.g.
            # uniform random init in tests).
            denom = 1.0
            log.debug(
                "Stage 1: all layers have identical R_raw=%.4g; R̃^l=0 for all layers "
                "(expected for single-layer models or uniform-init tests)",
                r_min,
            )
        redundancies: dict[int, float] = {
            li: (R_raw[li] - r_min) / denom for li in R_raw
        }

        for li in D_matrices:
            _trackio_log({
                "stage1/layer_idx": li,
                "stage1/redundancy": redundancies[li],
                "stage1/budget": budgets[li],
            })
        # End-of-Phase-E/F per-layer emit: drain before Stage 1 returns
        # control to its caller.
        _trackio_flush()

        ctx.set(
            "per_layer_target_experts",
            {str(k): v for k, v in budgets.items()},
        )
        ctx.set(
            "per_layer_redundancy",
            {str(k): v for k, v in redundancies.items()},
        )
        ctx.set("achieved_budget", sum(budgets.values()))
        ctx.set("requested_budget", decomposition.global_expert_budget)
        ctx.set("grape_config", dict(s1))

    def contribute_artifact(self, ctx: Stage1Context) -> dict:
        """Return the ``stage1_budgets.json`` payload.

        The orchestrator calls this method to obtain the dict that
        becomes ``stage1_budgets.json``; it writes the returned dict to
        ``artifacts_dir / "stage1_budgets.json"`` via
        :func:`utils.model_io.save_json_artifact`.

        **Note on the artifact-write boundary.** Per the overarching
        plan, ``contribute_artifact`` returns a dict — the *caller*
        decides where to write it. For Phase F the dict is its own file
        (``stage1_budgets.json``), not a fragment of the shared
        ``stage1_blacklist.json``. Both writer patterns coexist:

        - Fragment contributors (Phase A/C₂/C₃/...) return a dict that
          :class:`ArtifactBuilder` merges into ``stage1_blacklist.json``.
        - Whole-file contributors (Phase F here, Phase D in sub-task 5)
          return the complete file payload; the orchestrator writes it
          directly with :func:`save_json_artifact` under a path it owns.

        The plugin itself never writes to disk.

        Returns
        -------
        dict
            Exactly five top-level keys, byte-identical to the
            historical ``stage1_budgets.json`` schema:

            - ``per_layer_target_experts`` : dict[str, int]
            - ``per_layer_redundancy``     : dict[str, float]
            - ``achieved_budget``          : int
            - ``requested_budget``         : int
            - ``config``                   : dict (the Stage 1 YAML
              sub-config, identical to the legacy ``"config": dict(s1)``
              field)
        """
        return {
            "per_layer_target_experts": ctx.get("per_layer_target_experts"),
            "per_layer_redundancy": ctx.get("per_layer_redundancy"),
            "achieved_budget": ctx.get("achieved_budget"),
            "requested_budget": ctx.get("requested_budget"),
            "config": ctx.get("grape_config"),
        }


# ---------------------------------------------------------------------------
# Blacklist masking helper (consumed by ``_grape_greedy_merge`` below).
# ---------------------------------------------------------------------------


def _zero_blacklisted(d: np.ndarray, bl_experts: list[int]) -> np.ndarray:
    """Zero rows and columns of distance matrix `d` for blacklisted experts in-place."""
    for e in bl_experts:
        d[e, :] = 0.0
        d[:, e] = 0.0
    return d


# ---------------------------------------------------------------------------
# GRAPE Algorithm 1 (entropy-aware greedy merge with restart)
# ---------------------------------------------------------------------------


def _grape_greedy_merge(
    *,
    D_matrices: dict[int, torch.Tensor],
    global_budget: int,
    per_layer_counts: dict[int, int],
    blacklist: dict[int, list[int]],
    gamma: float,
    floor_divisor: int = 2,
    merge_cost_prior: dict[int, float] | None = None,
) -> dict[int, int]:
    """GRAPE Algorithm 1 (2604.06542, §3.3).

    Returns per-layer surviving expert counts (budgets). Floor is
    ``per_layer_counts[li] // floor_divisor`` computed independently for each
    layer, so heterogeneous architectures are handled correctly.

    Optional Direction-A second-phase knobs (both default to a strict no-op):

      * ``floor_divisor`` (default ``2``): the per-layer floor divisor. ``2``
        reproduces the spec §12 D5 ``N // 2`` invariant exactly. A value
        ``> 2`` lets GRAPE drive layers below ``N // 2`` so the budget-retune
        tool and a second Stage-1 pass agree on the floor.
      * ``merge_cost_prior`` (default ``None``): an optional ``layer_idx ->
        cost`` map. When supplied, ``best_layer`` selection minimises
        ``R[li] * merge_cost_prior[li]`` instead of ``R[li]`` alone, so
        Stage 1 can be biased by *measured* merge damage from a prior Stage-2
        run. When ``None`` selection is byte-identical to the original
        ``min R[li]`` behaviour.
    """
    if floor_divisor < 1:
        raise ValueError(
            f"_grape_greedy_merge: floor_divisor must be >= 1, got {floor_divisor}."
        )
    sorted_layers = sorted(per_layer_counts.keys())
    n_moe_layers = len(sorted_layers)

    # Validate that no layer's blacklist exceeds its total expert count.
    for li in sorted_layers:
        bl_count = len(blacklist.get(li, []))
        layer_count = per_layer_counts.get(li, 0)
        if bl_count > layer_count:
            raise ValueError(
                f"layer {li}: blacklist has {bl_count} experts but layer only has {layer_count}"
            )
        if bl_count == layer_count:
            log.warning(
                "layer %d: all %d experts are blacklisted; this layer cannot contribute any merges",
                li, layer_count,
            )

    # Entropy is computed over active (non-blacklisted) experts only.
    # Blacklisted experts are not available for merging, so including them in
    # cluster_counts would inflate E_init and cause premature layer freezing.
    cluster_counts: dict[int, int] = {
        li: per_layer_counts[li] - len(blacklist.get(li, []))
        for li in per_layer_counts
    }

    # global_budget (from BudgetDecomposition) counts TOTAL surviving experts including
    # blacklisted ones. GRAPE tracks only non-blacklisted experts in cluster_counts, so
    # the termination condition must compare against the non-blacklisted budget.
    total_blacklisted = sum(len(v) for v in blacklist.values())
    effective_budget = max(0, global_budget - total_blacklisted)
    if total_blacklisted > global_budget:
        log.warning(
            "GRAPE: total_blacklisted=%d > global_budget=%d — the mandatory super-expert set "
            "already exceeds the requested budget; effective_budget forced to 0. "
            "Consider increasing global_budget or reducing a_max_fraction.",
            total_blacklisted, global_budget,
        )

    # R^l = sum of off-diagonal distances (Eq. 11, sum form).
    # D_matrices contains DISTANCES (0=identical, large=different) from
    # _pairwise_distance_matrix / _cka_distance_matrix. Small R means experts
    # are mutually similar (redundant); large R means diverse experts.
    # Layer selection uses argmin R (most redundant = smallest distance sum),
    # NOT argmax — this is correct for distance matrices despite GRAPE's paper
    # notation which uses argmax R over a SIMILARITY-based R.
    #
    # Blacklisted experts are zeroed out in D_work so they never participate
    # in pair selection as either centroid (i_star) or absorbed expert (j_star),
    # and their distances do not inflate R (which would bias layer selection).
    D_work: dict[int, np.ndarray] = {
        li: _zero_blacklisted(D_matrices[li].cpu().numpy().copy(), blacklist.get(li, []))
        for li in sorted_layers
    }

    for li, D in D_work.items():
        diag = np.diag(D)
        if not np.allclose(diag, 0.0):
            log.debug("Stage 1: D_work[layer %d] diagonal is non-zero (max=%.2e); R update may double-count", li, float(np.abs(diag).max()))

    R: dict[int, float] = {}
    for li in sorted_layers:
        d = D_work[li]
        n = d.shape[0]
        R[li] = float((d.sum() - np.diag(d).sum())) if n > 1 else 0.0

    # floors[li] is the NON-BLACKLISTED portion of the hard floor, i.e. the
    # minimum number of non-blacklisted experts that must survive in layer li.
    # Total floor = floors[li] + len(blacklist[li]).
    # GRAPE tracks only non-blacklisted experts in cluster_counts, so
    # cluster_counts[li] must not drop below floors[li].
    # NOTE: `min_experts_per_layer` is a config key consumed by the budget solver
    # for global feasibility; Stage 1's per-layer floor is
    # `per_layer_counts[li] // floor_divisor`. With the default
    # `floor_divisor == 2` this is exactly the spec §12 D5 `N // 2` invariant
    # ("min_experts_per_layer = num_routed_experts // 2") — byte-identical to
    # the historical hardcoded floor. `floor_divisor` is only ever non-2 when
    # a caller explicitly opts in (Direction-A second pass).
    floors: dict[int, int] = {
        li: max(per_layer_counts[li] // floor_divisor - len(blacklist.get(li, [])), 0)
        for li in sorted_layers
    }

    def _entropy(counts: dict[int, int]) -> float:
        if any(v < 0 for v in counts.values()):
            neg_entries = {k: v for k, v in counts.items() if v < 0}
            raise ValueError(f"_entropy: negative count(s) encountered: {neg_entries}")
        total = sum(counts.values())
        if total == 0:  # covers both empty dict and all-zero dict
            return 0.0
        probs = np.fromiter((c / total for c in counts.values()), dtype=np.float64, count=len(counts))
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    if gamma == 0.0:
        log.warning(
            "GRAPE: gamma=0.0 — entropy gate at initial entropy — every entropy-reducing merge will trigger a freeze.",
        )
    elif gamma < 0.0:
        log.warning(
            "GRAPE: gamma=%.4f < 0: E_hat > E_init — every merge reduces entropy below the "
            "inflated threshold, so most layers freeze after the first merge per restart cycle; "
            "the loop produces approximately %d merges per restart cycle (one per MoE layer); "
            "convergence may require far more iterations than the normal (gamma>0) case",
            gamma, n_moe_layers,
        )
    elif gamma >= 1.0:
        log.warning(
            "GRAPE: gamma=%.4f >= 1.0: E_hat <= 0 — entropy gate permanently disabled; "
            "GRAPE will merge greedily to floor without entropy constraints",
            gamma,
        )

    E_init = _entropy(cluster_counts)
    # E_hat is intentionally fixed to the pre-loop baseline; it is not updated per restart (conservative approximation).
    E_hat = E_init * (1.0 - gamma)

    frozen: set[int] = set()
    # Layers where all valid (non-absorbed) pairs have been exhausted — no merge is ever
    # possible again regardless of entropy state.  Unlike `frozen`, this set is NOT
    # cleared on entropy restart so that exhausted layers are never re-selected.
    structurally_blocked: set[int] = set()
    # Pre-populate floor_blocked for layers already at their floor before merging starts.
    # Lazy-add inside the loop handles layers that reach their floor mid-run, but
    # without this pre-population, layers simultaneously at-floor and in `frozen`
    # (from a prior restart cycle) would never be added to floor_blocked, causing
    # _non_floor_blocked to overcount and trigger spurious restarts.
    # both dicts are keyed over sorted_layers; .get() defaults are unreachable
    floor_blocked: set[int] = {li for li in sorted_layers if cluster_counts[li] <= floors[li]}
    current_total = sum(cluster_counts.values())

    log.info("GRAPE: global_budget=%d (non-bl effective=%d), current_total=%d, gamma=%.4g, E_hat=%.4f",
             global_budget, effective_budget, current_total, gamma, E_hat)

    # Per-layer sets of absorbed (merged-away) expert indices.  Using an explicit
    # set — rather than checking D_l == 0 — avoids misidentifying genuinely
    # zero-distance (identical-weight) expert pairs as already-merged.
    # Pre-populate with blacklisted experts: their D_work rows/cols are 0.0
    # (zeroed during D_work initialization), so without this pre-population
    # argmin would select them as j_star and corrupt cluster_counts.
    merged: dict[int, set[int]] = {
        li: set(blacklist.get(li, [])) for li in sorted_layers
    }

    if current_total == 0:
        log.debug("GRAPE: all unfrozen experts blacklisted; skipping greedy merge loop")
        return {li: cluster_counts[li] + len(blacklist.get(li, [])) for li in cluster_counts}

    # Tight case: at most current_total merge-iterations plus at most n_moe_layers
    # structurally-blocked skip-iterations (each layer joins structurally_blocked at most
    # once). Top-of-loop restarts fall through to layer selection in the same iteration;
    # lag-corrected restarts use `continue` and burn one iteration without a merge.
    # The factor n_moe_layers * 2 is well above this tight bound in both cases.
    max_iterations = current_total * n_moe_layers * 2
    log.debug("GRAPE max_iterations=%d (current_total=%d, n_moe_layers=%d)",
              max_iterations, current_total, n_moe_layers)
    # n_merges counts successful merge operations only.  Structural-blocking skip-iterations
    # advance iter_ without incrementing n_merges, so n_merges <= iter_ always.
    n_merges = 0
    exit_reason = "max_iter"
    for iter_ in range(max_iterations):
        if current_total <= effective_budget:
            exit_reason = "budget"
            break

        # Restart only when entropy-frozen layers block all non-floor-blocked,
        # non-structurally-blocked layers.  Floor and structural constraints are
        # permanent — clearing frozen can't help those, and must not touch
        # structurally_blocked (which persists across restarts).
        # floor_blocked is populated lazily during layer selection; pre-seeded in the
        # initialization block above for layers already at their floor, but
        # mid-run additions lag by one iteration. permanently_blocked may undercount if
        # layers are skipped via structurally_blocked before reaching the floor check.
        # Additionally, a layer whose cluster_count was just decremented to floor[li] in
        # the current iteration is not yet in floor_blocked — it is added only at its
        # next examination, causing a one-iteration lag independent of the
        # structurally_blocked path.
        # permanently_blocked = |floor_blocked ∪ structurally_blocked|
        permanently_blocked = len(floor_blocked) + len(structurally_blocked - floor_blocked)
        non_perm_blocked = n_moe_layers - permanently_blocked
        if non_perm_blocked > 0 and len(frozen - structurally_blocked - floor_blocked) >= non_perm_blocked:
            log.info("GRAPE iter %d: all non-permanently-blocked layers frozen → restart", iter_)
            frozen.clear()
            # After clearing frozen, the current iteration immediately runs layer selection
            # and may merge a layer that re-triggers the entropy gate on the following iteration.
            # This one extra merge per restart-cycle is by design — it allows GRAPE to escape
            # local optima.

        best_layer = None
        best_R = math.inf
        for li in sorted_layers:
            if li in structurally_blocked:
                continue
            if li in frozen:
                continue
            if cluster_counts[li] <= floors[li]:
                floor_blocked.add(li)
                continue
            # Selection score. With merge_cost_prior unset (the default) this
            # is `R[li]` verbatim — byte-identical to the original
            # `if R[li] < best_R` greedy. When a prior IS supplied (Direction-A
            # second pass) the score is `R[li] * merge_cost_prior[li]`, biasing
            # selection toward layers that are both activation-redundant AND
            # cheap to merge per measured Stage-2 damage.
            if merge_cost_prior is None:
                score = R[li]
            else:
                score = R[li] * merge_cost_prior[li]
            if score < best_R:
                best_R = score
                best_layer = li

        if best_layer is None:
            # floor_blocked was updated lazily inside the selection loop above; re-evaluate
            # the restart condition with the now-complete floor_blocked before giving up —
            # the per-iteration lag may have prevented the check above from firing.
            permanently_blocked = len(floor_blocked) + len(structurally_blocked - floor_blocked)
            non_perm_blocked = n_moe_layers - permanently_blocked
            if non_perm_blocked > 0 and len(frozen - structurally_blocked - floor_blocked) >= non_perm_blocked:
                log.info("GRAPE iter %d: post-selection restart (lag-corrected) — unfreezing frozen layers", iter_)
                frozen.clear()
                continue
            log.warning("GRAPE: no unfrozen layer can donate — stopping at %d (target %d)",
                        current_total, effective_budget)
            exit_reason = "no_layer"
            break

        D_l = D_work[best_layer]
        n = D_l.shape[0]
        # For a distance matrix: find the most similar (smallest distance) pair where
        # neither expert has already been absorbed.  Track absorbed experts explicitly
        # so that genuinely zero-distance (identical-weight) pairs remain selectable.
        absorbed = merged[best_layer]
        tmp = D_l.copy()
        np.fill_diagonal(tmp, np.inf)
        for a in absorbed:
            tmp[a, :] = np.inf
            tmp[:, a] = np.inf
        if not np.isfinite(tmp).any():
            structurally_blocked.add(best_layer)
            # not added to frozen — structurally_blocked takes precedence.
            continue
        flat_idx = int(np.argmin(tmp))
        # n is the original matrix dimension (total experts), not the count of
        # remaining unabsorbed experts.
        # i_star is intentionally discarded: GRAPE's contribution to Stage 2 is the per-layer budget N'_l, not pair assignments — Stage 2 re-derives centroids via covariance (spec §4 line 271).
        _, j_star = divmod(flat_idx, n)

        # D4: zero entire row/column of absorbed expert j_star and update R.
        # R = Σ_{i≠j} D_l[i, j] (sum of all off-diagonal entries). When j_star
        # is absorbed we must remove its full contribution: D_l[j_star, k] and
        # D_l[k, j_star] for all k. Read the full row/column sum BEFORE zeroing
        # so that D_l[i_star, j_star] / D_l[j_star, i_star] are still included.
        # Defensively subtract the (always-zero) diagonal once so the formula is
        # robust if any future metric ever yielded a non-zero self-distance.
        j_contribution = float(
            D_l[j_star, :].sum() + D_l[:, j_star].sum() - D_l[j_star, j_star]
        )
        R[best_layer] -= j_contribution
        pre_clamp_R = R[best_layer]
        R[best_layer] = max(0.0, pre_clamp_R)
        if pre_clamp_R < 0.0:
            log.debug("_grape_greedy_merge: pre-clamp R[%d]=%.2e clamped to 0.0 (FP drift)", best_layer, pre_clamp_R)
        D_l[j_star, :] = 0.0
        D_l[:, j_star] = 0.0
        absorbed.add(j_star)

        cluster_counts[best_layer] -= 1
        current_total -= 1

        E_current = _entropy(cluster_counts)
        if E_current < E_hat:
            frozen.add(best_layer)

        n_merges += 1

    log.info("GRAPE: converged at %d non-blacklisted experts (target %d) after %d merges (exit=%s)",
             current_total, effective_budget, n_merges, exit_reason)

    if current_total > effective_budget and exit_reason == "max_iter":
        log.warning(
            "GRAPE: could not reach effective_budget=%d non-blacklisted (achieved=%d) "
            "after %d iterations (max_iterations=%d). "
            "Consider increasing global_budget or reducing the target compression ratio "
            "(floors are per_layer_counts[li] // 2 per layer).",
            effective_budget, current_total, iter_ + 1, max_iterations,
        )

    # One-shot Trackio emit of GRAPE summary. All variables already in scope
    # — pure additive emit, no new state computed.
    _trackio_log({
        "stage1/effective_budget": int(effective_budget),
        "stage1/global_budget": int(global_budget),
        "stage1/total_blacklisted": int(total_blacklisted),
        "stage1/entropy_initial": float(E_init),
        "stage1/entropy_threshold": float(E_hat),
        "stage1/gamma": float(gamma),
        "stage1/n_merges_executed": int(n_merges),
        "stage1/exit_reason": exit_reason,
        "stage1/final_total": int(current_total),
    })
    # End-of-GRAPE-solver: drain final summary before returning.
    _trackio_flush()

    # Stage 2 reads per-layer budgets as TOTAL centroid count (blacklisted + non-blacklisted).
    # Add blacklisted experts back so Stage 2's effective_target is inclusive.
    return {
        li: cluster_counts[li] + len(blacklist.get(li, []))
        for li in cluster_counts
    }
