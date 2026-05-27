"""Per-layer damage curve + DP knapsack budget allocation (S1_DP).

Papers
------
This plugin combines two anchors:

1. **R4 — Additivity theorem (arxiv:2308.10438, Aug 2023).**
   *Efficient Joint Optimization of Layer-Adaptive Weight Pruning in
   Deep Neural Networks*, Theorem 1: under Taylor + i.i.d. perturbation
   assumptions,

       E[‖f(x; W) − f(x; W̃)‖²] ≈ Σᵢ E[δᵢ]

   where δᵢ is the per-layer output distortion. **Additivity** makes
   the per-layer budget allocation a 1D knapsack:

       min Σ_ℓ D_ℓ(k_ℓ)   s.t.   Σ_ℓ k_ℓ = global_merge_target

   solvable exactly by O(L · K · G) DP.

2. **R8 — HC-SMoE (arxiv:2410.08589, ICML 2024).**
   *Retraining-Free Merging of Sparse MoE via Hierarchical Clustering*,
   Appendix B.1: the earliest cited precedent for "vary the per-layer
   budget instead of holding it uniform". HC-SMoE's implementation is
   crude (global frequency threshold determines per-layer counts as a
   side-effect); the DP-knapsack-on-damage-curve here is a principled
   refinement of the same idea.

Spec
----
``tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md`` §7 A1 — "S1_DP" / Rec 2:
per-layer damage curve ``D_ℓ(k)`` + DP knapsack → populates
``stage1_grape.merge_cost_prior`` consumed by
:class:`~moe_compress.stage1.plugins.grape_merge.GrapeMergePlugin`'s
inert hook.

Official code
-------------
No published reference implementation for the combined S1_DP recipe.
The DP-knapsack primitive is textbook (R4 §3 sketches the algorithm in
~10 lines of math) and is implemented reference-free against the
recurrence in the paper.

Algorithm
=========

For each MoE layer ℓ, sort the **off-diagonal**, non-blacklisted
entries of ``D_matrices[ℓ]`` in ascending order (most similar / smallest
distance first) and cumulate::

    D_ℓ(k) = Σ_{i=1..k} sort_asc(off_diag(D_matrices[ℓ]))[i]
    D_ℓ(0) = 0

This is the per-layer **damage curve**. Under R4 additivity it is an
additive cost-of-merging estimate. ``D_ℓ`` is monotone non-decreasing
in ``k`` by construction (cumsum of non-negative sorted-ascending
values).

Then solve the layer-knapsack DP::

    dp[0][0]     = 0
    dp[0][b > 0] = +inf
    dp[i+1][b]   = min_{k ∈ [0, k_max_i]} (dp[i][b - k] + D_i(k))

over ``i ∈ [0, L)``, ``b ∈ [0, G]``, where ``G`` = global merge target
(= ``Σ_ℓ N_ℓ − global_expert_budget``) and ``k_max_ℓ`` =
``N_ℓ − floor_ℓ``. Traceback gives the optimal merge count ``k*_ℓ``
per layer.

The **prior** published to ``stage1_grape.merge_cost_prior`` is the
**marginal damage at the optimum**::

    prior_ℓ = D_ℓ(k*_ℓ + 1) − D_ℓ(k*_ℓ)     if k*_ℓ < k_max_ℓ
    prior_ℓ = +∞                            if k*_ℓ == k_max_ℓ  (at floor)

GRAPE's selection rule becomes ``argmin R[li] · prior[li]``. Layers at
their DP-optimum floor get ``+∞`` and are excluded from further merges;
layers below their floor still have a finite marginal damage and remain
in play. ``prior_ℓ = 0`` is clamped to a small positive epsilon
(:data:`_PRIOR_EPS`) so the multiplicative selection never collapses.

Deviations
==========

D-cka-substitute-for-output-mse
-------------------------------
R4 / Rec 2 of the SC plan prescribe the **output-space** MSE
(``stage2.output_space_cost._output_space_cost``) as the per-layer
damage signal. That cost requires (a) a Stage 2 cost-matrix
configuration (centroid / non-centroid partition, freq dict) that does
not yet exist at Stage 1; and (b) per-layer input reservoirs +
permutation caches the Stage 2 driver builds. To keep S1_DP shippable as
a "cheap baseline" (per ``SC_STAGE12_COMPREHENSIVE_PLAN.md`` §5.4: *"S1_DP
is positioned as a cheap baseline against S1_RCO, not the headline
Stage 1 method"*), this plugin **substitutes the CKA off-diagonal
distance** — already computed by ``CKADistancePlugin`` — for the
output-space MSE. Empirically the two costs share the GRAPE-style
"smaller distance ⇒ smaller merge damage" monotone ordering at small k;
the substitution is paper-consistent with GRAPE's own choice of CKA as
its merge primitive. A future ``S1_DP_OUTPUT`` variant could swap in
the Stage 2 cost via the same plugin scaffold.

D-dp-prior-as-marginal
----------------------
The DP optimum ``k*_ℓ`` itself is a complete per-layer count vector and
*could* be used directly as the budget (skip GRAPE entirely). Instead
we publish the **marginal damage at the optimum** as a multiplicative
prior into GRAPE's existing ``merge_cost_prior`` hook
(``grape_merge.py:171-176``), letting GRAPE's entropy-aware greedy
refine the DP starting point. Rationale: GRAPE's entropy gate (γ)
already encodes a regulariser the DP does not see; using the DP as a
*biaser* keeps that gate active and preserves the proven SC=0.1293
baseline behaviour when DP is disabled.

D-prior-floor-eps
-----------------
``prior_ℓ = 0`` (free further merge available — degenerate D_ℓ profile)
is clamped to :data:`_PRIOR_EPS` = ``1e-12`` so GRAPE's multiplicative
``R[li] · prior[li]`` rule remains discriminative. Without this clamp a
single zero-prior layer would attract every greedy step regardless of
its CKA redundancy.

The value ``1e-12`` is intentionally **far below** the natural CKA
distance scale (off-diagonal CKA distances are in ``[0, 1]``; typical
pair distances at the head of the sorted curve are O(0.1), and the
smallest non-degenerate marginal differences observed empirically are
O(1e-3..1e-4)). The clamp therefore only activates on *exact-zero*
marginals — degenerate identical-expert pairs and FP-drift cases — and
leaves every realistic marginal untouched. A larger ``_PRIOR_EPS``
(say ``1e-6``) would risk masking genuine small marginals at the tail
of the damage curve when many near-duplicate experts collide there.
Kept as a module constant rather than a YAML knob because changing it
without also auditing the empirical marginal-distribution histogram
would silently shift GRAPE's bias.

D-independent-pairs-assumption
------------------------------
``D_ℓ(k) = Σ_{i=1..k} sort_asc(off_diag(D_matrices[ℓ]))[i]`` sums the
``k`` smallest off-diagonal pair distances **as if those pairs were
realisable simultaneously**. In reality R4's per-layer ``δᵢ`` treats
``k`` merge *steps* where each absorption changes the expert
membership — two pairs that share an expert (e.g. ``(1, 2)`` and
``(1, 3)``) cannot both be true merge steps. R4 itself does **not**
assume independent pairs; the sorted-pair cumsum is a strict UPPER
BOUND on the realisable per-layer cost, biasing the DP toward layers
with sparser low-distance pair structure (layers whose low-distance
pair list contains many disjoint pairs score artificially low because
no overlap penalty is paid). Acceptable per
``SC_STAGE12_COMPREHENSIVE_PLAN.md`` §5.4 — S1_DP is positioned as a
cheap baseline against S1_RCO, not the headline Stage 1 method; the
upper-bound bias is the price paid for the O(L · K · G) closed-form
DP. A future ``S1_DP_GREEDY`` variant could rebuild the damage curve
incrementally after each candidate absorption to remove the
assumption, at the cost of O(L · K² · G) runtime.

Output context contract
-----------------------
- ``reads``: ``D_matrices``, ``blacklist``, ``per_layer_targets``,
  ``decomposition``, ``config``.
- ``writes``: ``damage_curves`` (dict[int, np.ndarray]),
  ``dp_optimum`` (dict[int, int]), ``merge_cost_prior_computed``
  (dict[int, float]). Also mutates
  ``config["stage1_grape"]["merge_cost_prior"]`` so GRAPE's existing
  inert hook activates.
- ``provides``: ``()`` — pure post-process over ``D_matrices``.
- ``contribute_artifact``: returns ``{}`` (diagnostics live on ctx
  slots; no separate JSON file).

When disabled (``stage1_grape.damage_curve_dp.enabled == false``, the
default), the plugin returns immediately from ``run()`` and writes
nothing. With every other plugin held fixed the Stage 1 output is
byte-identical to the historical GRAPE-only path.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from ...pipeline.context import PipelineContext
from ._floor import per_layer_floor

log = logging.getLogger(__name__)

# Clamp value for ``prior_ℓ = 0`` so GRAPE's ``R · prior`` selection rule
# stays discriminative. See D-prior-floor-eps in the module docstring.
_PRIOR_EPS: float = 1.0e-12


class DamageCurveDpPlugin:
    """Per-layer damage curve + DP knapsack for Stage 1 budget allocation.

    Reads CKA distance matrices, blacklist, per-layer expert counts, and
    the global budget; computes per-layer cumulative damage curves
    ``D_ℓ(k)``; solves the 1D DP knapsack
    ``min Σ_ℓ D_ℓ(k_ℓ) s.t. Σ_ℓ k_ℓ = G``; and publishes the marginal
    damage at the optimum as ``stage1_grape.merge_cost_prior`` for
    :class:`GrapeMergePlugin` to consume.

    Disabled by default. See module docstring for the R4 / R8 paper
    citations and the four deviations:
    ``D-cka-substitute-for-output-mse``, ``D-dp-prior-as-marginal``,
    ``D-prior-floor-eps``, and ``D-independent-pairs-assumption``.
    """

    name: str = "damage_curve_dp"
    paper: str = (
        "R4 Additivity theorem arXiv:2308.10438 Theorem 1 (DP formal basis); "
        "R8 HC-SMoE arXiv:2410.08589 Appendix B.1 (non-uniform-budget precedent). "
        "No official code published. "
        "Deviations: D-cka-substitute-for-output-mse (damage uses CKA off-diagonal "
        "distance cumsum vs paper Rec 2's output-space MSE — _output_space_cost "
        "machinery lives in Stage 2 and isn't available at Stage 1); "
        "D-dp-prior-as-marginal (prior published into merge_cost_prior is the "
        "marginal damage D_ℓ(k*+1)−D_ℓ(k*) at the DP optimum, not the cumulative "
        "value — preserves GRAPE's entropy gate); D-prior-floor-eps (prior=0 "
        "clamped to 1e-12 so R·prior stays discriminative); "
        "D-independent-pairs-assumption (sorted-pair cumsum treats merged pairs "
        "as independent → strict upper bound on R4's per-layer δᵢ; biases DP "
        "toward layers with sparser low-distance pair structure; acceptable "
        "per §5.4 cheap-baseline positioning). See module docstring for full "
        "algorithm + per-deviation derivations."
    )
    config_key: str = "stage1_grape.damage_curve_dp.enabled"
    reads: tuple[str, ...] = (
        "D_matrices",
        "blacklist",
        "per_layer_targets",
        "decomposition",
        "config",
    )
    writes: tuple[str, ...] = (
        "damage_curves",
        "dp_optimum",
        "merge_cost_prior_computed",
    )
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Gate on ``config["stage1_grape"]["damage_curve_dp"]["enabled"]``.

        Default is ``False`` — when missing or false the plugin no-ops and
        Stage 1 is byte-identical to the GRAPE-only path. The orchestrator
        always calls :meth:`run`; the run itself short-circuits when
        disabled (mirrors the legacy plugin pattern — see
        ``magnitude_topk.py``).
        """
        s1 = config.get("stage1_grape", {})
        dp = s1.get("damage_curve_dp", {})
        return bool(dp.get("enabled", False))

    def run(self, ctx: PipelineContext) -> None:
        """Execute damage-curve DP end-to-end.

        Short-circuits when :meth:`is_enabled` is false. Otherwise:

        1. Builds per-layer cumulative damage curves from
           ``D_matrices`` (off-diagonal distances of non-blacklisted
           experts, sorted ascending, cumsummed).
        2. Solves the 1D layer-knapsack DP for the global merge target.
        3. Publishes the marginal damage at the optimum as
           ``merge_cost_prior`` on the in-ctx config, and stashes the
           curves + optimum on dedicated ctx slots for downstream
           inspection.
        """
        config: dict = ctx.get("config")
        if not self.is_enabled(config):
            log.debug("Stage 1 damage_curve_dp: disabled, skipping")
            return

        D_matrices: dict[int, torch.Tensor] = ctx.get("D_matrices")
        blacklist: dict[int, list[int]] = ctx.get("blacklist")
        per_layer_counts: dict[int, int] = ctx.get("per_layer_targets")
        decomposition = ctx.get("decomposition")

        s1 = config["stage1_grape"]
        # Note (L3): the floor_divisor validation only fires when
        # `enabled=True` — when disabled the plugin short-circuits above
        # before reading any config beyond the gate. This matches the
        # project convention (validation gated on plugin activation) so
        # a disabled-default Stage 1 never trips on an invalid value
        # that GRAPE itself validates in parallel.
        floor_divisor = int(s1.get("grape_floor_divisor", 2))
        if floor_divisor < 1:
            raise ValueError(
                f"damage_curve_dp: grape_floor_divisor must be >= 1, got "
                f"{floor_divisor}"
            )

        sorted_layers = sorted(per_layer_counts.keys())
        global_budget = int(decomposition.global_expert_budget)
        total_experts = sum(per_layer_counts[li] for li in sorted_layers)
        total_blacklisted = sum(
            len(blacklist.get(li, [])) for li in sorted_layers
        )
        # Global merge target: total experts to remove across all layers.
        # L1 clamp: the merge count cannot exceed the number of
        # non-blacklisted experts (since blacklisted experts are
        # immovable). Mirrors GRAPE's effective_budget =
        # max(0, global_budget − total_blacklisted) in
        # ``grape_merge.py`` and keeps the DP feasibility-guard
        # consistent in the degenerate case where global_budget <
        # total_blacklisted.
        global_merges = max(0, total_experts - global_budget)
        global_merges = min(global_merges, total_experts - total_blacklisted)

        # Per-layer max-merges = N_ℓ − total_floor_ℓ; floor is computed via
        # the shared ``per_layer_floor`` helper so the DP plans merges
        # against the SAME total floor GRAPE enforces downstream
        # (``grape_merge.py`` D5).
        k_max: dict[int, int] = {}
        for li in sorted_layers:
            n = per_layer_counts[li]
            bl = len(blacklist.get(li, []))
            k_max[li] = per_layer_floor(n, bl, floor_divisor).k_max

        # Build per-layer damage curves D_ℓ(k) of length k_max[li] + 1.
        damage_curves: dict[int, np.ndarray] = _build_damage_curves(
            D_matrices=D_matrices,
            blacklist=blacklist,
            sorted_layers=sorted_layers,
            k_max=k_max,
        )

        # Solve the DP. Feasibility guards: if Σ k_max < global_merges the
        # target is infeasible against the floors — fall back to k_max
        # everywhere and warn (downstream GRAPE will then enforce its own
        # floor anyway; the prior just biases ordering).
        sum_kmax = sum(k_max.values())
        if global_merges > sum_kmax:
            log.warning(
                "damage_curve_dp: global_merges=%d exceeds Σ k_max=%d "
                "(over-blacklisted or floor too tight); pinning each layer "
                "at its k_max and setting prior=+inf everywhere.",
                global_merges, sum_kmax,
            )
            dp_optimum = {li: k_max[li] for li in sorted_layers}
        elif global_merges == 0:
            log.info(
                "damage_curve_dp: global_merges=0 (compression target met "
                "by the initial expert counts); DP trivial — all k*=0."
            )
            dp_optimum = {li: 0 for li in sorted_layers}
        else:
            dp_optimum = _solve_knapsack_dp(
                sorted_layers=sorted_layers,
                damage_curves=damage_curves,
                k_max=k_max,
                global_merges=global_merges,
            )

        # Marginal-damage prior at the optimum (D-dp-prior-as-marginal).
        prior: dict[int, float] = {}
        for li in sorted_layers:
            k_star = dp_optimum[li]
            curve = damage_curves[li]
            # curve has length k_max[li] + 1, indexed 0..k_max[li].
            if k_star >= k_max[li]:
                # At-floor: no further merge available, GRAPE must not pick.
                prior[li] = math.inf
            else:
                marginal = float(curve[k_star + 1] - curve[k_star])
                # Defensive: curve is monotone non-decreasing by construction
                # (cumsum of sorted-ascending non-negative distances). Clamp
                # any FP drift below 0 to 0, then apply _PRIOR_EPS floor.
                if marginal < 0.0:
                    log.debug(
                        "damage_curve_dp: layer %d marginal=%.2e clamped to 0 "
                        "(FP drift on sorted cumsum)", li, marginal,
                    )
                    marginal = 0.0
                prior[li] = max(marginal, _PRIOR_EPS)

        # Publish: dedicated ctx slots for inspection + mutate the in-ctx
        # config so GRAPE's existing inert hook activates without any GRAPE
        # code change. GRAPE reads `s1.get("merge_cost_prior")` (string-keyed
        # per its contract); we honour that schema.
        ctx.set("damage_curves", damage_curves)
        ctx.set("dp_optimum", dp_optimum)
        ctx.set("merge_cost_prior_computed", prior)
        s1["merge_cost_prior"] = {str(li): float(prior[li]) for li in sorted_layers}

        finite = [p for p in prior.values() if math.isfinite(p)]
        log.info(
            "damage_curve_dp: solved DP knapsack — global_merges=%d, "
            "Σ k*=%d, prior range (finite) [%.3g..%.3g] over %d layers "
            "(%d at-floor/+inf).",
            global_merges, sum(dp_optimum.values()),
            min(finite) if finite else 0.0,
            max(finite) if finite else 0.0,
            len(sorted_layers),
            len(sorted_layers) - len(finite),
        )

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return ``{}`` — damage_curve_dp contributes no JSON artifact.

        Diagnostics live on ctx slots (``damage_curves``, ``dp_optimum``,
        ``merge_cost_prior_computed``) for downstream inspection. The
        published prior is consumed by GRAPE and surfaced in
        ``stage1_budgets.json`` indirectly via its effect on the
        budget allocation.
        """
        return {}


# ---------------------------------------------------------------------------
# Damage-curve construction
# ---------------------------------------------------------------------------


def _build_damage_curves(
    *,
    D_matrices: dict[int, torch.Tensor],
    blacklist: dict[int, list[int]],
    sorted_layers: list[int],
    k_max: dict[int, int],
) -> dict[int, np.ndarray]:
    """Return per-layer cumulative damage curves ``D_ℓ(k)``.

    For each layer ℓ, gather the off-diagonal upper-triangular entries
    of ``D_matrices[ℓ]`` (each unordered pair counted once), excluding
    any pair involving a blacklisted expert; sort ascending; cumulative-
    sum to ``D_ℓ(k_max[ℓ])``. Pad with the last value if k_max[ℓ]
    exceeds the number of available pairs (degenerate over-floored case
    — guarded by k_max derivation but defensively handled here).

    Returns a dict keyed by layer index; each value is a NumPy array of
    length ``k_max[ℓ] + 1`` with ``curve[0] = 0`` and
    ``curve[k] = Σ_{i=1..k} sorted_distances[i-1]``.
    """
    curves: dict[int, np.ndarray] = {}
    for li in sorted_layers:
        d = D_matrices[li].detach().cpu().numpy().astype(np.float64)
        n = d.shape[0]
        bl_set = set(blacklist.get(li, []))
        # Upper-triangular off-diagonal pair list, excluding blacklist pairs.
        # ``triu(k=1)`` indices: each unordered pair counted exactly once.
        if n >= 2:
            iu, ju = np.triu_indices(n, k=1)
            # Vectorised mask: drop pairs that touch a blacklisted expert.
            if bl_set:
                bl_arr = np.fromiter(bl_set, dtype=np.int64, count=len(bl_set))
                keep = ~(np.isin(iu, bl_arr) | np.isin(ju, bl_arr))
                iu = iu[keep]
                ju = ju[keep]
            pair_vals = d[iu, ju]
            # CKA distances are in [0, 1]; clamp any FP drift below 0.
            pair_vals = np.clip(pair_vals, 0.0, None)
            pair_vals.sort()    # ascending: smallest / most-similar first
        else:
            pair_vals = np.empty(0, dtype=np.float64)
        cum = np.zeros(k_max[li] + 1, dtype=np.float64)
        # cum[0] = 0; cum[k] = sum of first k sorted pairs (or pad with last).
        n_pairs_take = min(k_max[li], pair_vals.size)
        if n_pairs_take > 0:
            cum[1: n_pairs_take + 1] = np.cumsum(pair_vals[:n_pairs_take])
        # Defensive: pad the tail when k_max exceeds available pairs (e.g.
        # n*(n-1)/2 < k_max). ``k_max`` is derived from per_layer_counts +
        # floor_divisor, so under normal config the bound is respected; this
        # branch only fires on pathological inputs (single-expert layers,
        # over-blacklisted layers). Padding with the last running value
        # makes the curve monotone non-decreasing with zero marginal at
        # the tail.
        if n_pairs_take < k_max[li]:
            cum[n_pairs_take + 1:] = cum[n_pairs_take]
        curves[li] = cum
    return curves


# ---------------------------------------------------------------------------
# DP knapsack solver
# ---------------------------------------------------------------------------


def _solve_knapsack_dp(
    *,
    sorted_layers: list[int],
    damage_curves: dict[int, np.ndarray],
    k_max: dict[int, int],
    global_merges: int,
) -> dict[int, int]:
    """Solve the 1D layer-knapsack DP exactly.

    Recurrence::

        dp[i+1][b] = min_{k ∈ [0, min(k_max_i, b)]} (dp[i][b - k] + D_i(k))

    Runtime: O(L · K · G); memory: O(L · G) for the traceback table.
    The per-layer transition is **fully vectorised**: for each layer
    ``i`` a single ``(G+1, km+1)`` matrix of ``dp[i, b-k] + curve[k]``
    candidates is built via broadcasting and reduced with a single
    ``np.argmin`` along the ``k`` axis.

    Returns a dict ``{layer_idx: k*_ℓ}`` minimising Σ D_ℓ(k_ℓ) subject
    to Σ k_ℓ = global_merges.
    """
    L = len(sorted_layers)
    G = int(global_merges)
    # dp[i][b] = min cost using layers [0..i) and budget b merges total.
    dp = np.full((L + 1, G + 1), math.inf, dtype=np.float64)
    # back[i][b] = k chosen at layer (i-1) when filling dp[i][b].
    back = np.full((L + 1, G + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    b_axis = np.arange(G + 1)
    for i, li in enumerate(sorted_layers):
        curve = damage_curves[li]
        km = int(k_max[li])
        # Fully vectorised transition. Build the candidate matrix
        #     M[b, k] = dp[i, b − k] + curve[k]    for k = 0..km, b = 0..G,
        # masking the invalid region (k > b) with +inf so it never wins
        # the argmin.
        k_axis = np.arange(km + 1)
        # B, K have shape (G+1, km+1); valid when K <= B.
        diff = b_axis[:, None] - k_axis[None, :]
        valid = diff >= 0
        # Index dp[i, b−k] safely: pad invalid lookups with 0 (cell never
        # consulted because `valid` masks them to +inf below).
        prev = np.where(valid, dp[i, np.where(valid, diff, 0)], np.inf)
        # Broadcast curve along the b axis. Result is (G+1, km+1).
        candidates = prev + curve[None, : km + 1]
        # Argmin over the k axis gives the optimal merge count per b.
        k_star = np.argmin(candidates, axis=1)
        dp[i + 1, :] = candidates[b_axis, k_star]
        back[i + 1, :] = k_star

    # Traceback from dp[L, G].
    if not math.isfinite(dp[L, G]):
        # Infeasibility — caller already guards against this via
        # global_merges > sum(k_max). Raise loudly if we reach here.
        raise RuntimeError(
            f"damage_curve_dp: DP infeasible at dp[L={L}, G={G}] = inf; "
            "this should be caught by the global_merges > Σ k_max guard "
            "upstream."
        )
    optimum: dict[int, int] = {}
    b = G
    for i in range(L, 0, -1):
        k = int(back[i, b])
        optimum[sorted_layers[i - 1]] = k
        b -= k
    # N5: post-traceback structural assertion — every layer must appear
    # exactly once in the optimum. Catches any future bug where the
    # traceback loop fails to walk back through all L layers (e.g. an
    # off-by-one in the range or a malformed back-pointer).
    assert set(optimum) == set(sorted_layers), (
        f"damage_curve_dp: traceback missed layers — "
        f"optimum={sorted(optimum)} vs sorted_layers={sorted_layers}"
    )
    return optimum
