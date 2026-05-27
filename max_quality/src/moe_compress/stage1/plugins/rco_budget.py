"""RCO — Riemannian-manifold budget allocator (Stage 1 refinement).

Paper
-----
IST-DASLab, *Model Compression with Exact Budget Constraints via
Riemannian Manifolds* — arxiv:2605.00649 (May 2026). §3 Algorithm 1.

Slot in this codebase
---------------------
**Stage 1 budget refinement.** This plugin runs AFTER
:class:`stage1.plugins.grape_merge.GrapeMergePlugin` and produces a
*refined* per-layer budget vector `{layer_idx -> surviving expert count}`
on a NEW context slot ``per_layer_target_experts_rco``. GRAPE's original
slot ``per_layer_target_experts`` is left untouched — downstream stages
choose which budget to consume via the standard config knob (the row
recipe sets the slot name; default is GRAPE).

Algorithm summary (paper §3, Algorithm 1)
-----------------------------------------
RCO recasts the discrete budget-allocation problem as a smooth
optimization over a **Riemannian manifold** defined by an *exact* budget
constraint. State is a per-layer matrix of logits ``α ∈ ℝ^{L × K}`` where
``L`` is the MoE layer count and ``K`` is the number of candidate
per-layer budgets. ``p_lk = softmax(α_l)_k`` is the soft allocation and
the constraint is ``Σ_l Σ_k p_lk · c_lk = B`` with ``c_lk`` = surviving
expert count of option ``k`` for layer ``l`` and ``B`` = global expert
budget.

Three manifold primitives drive the search (paper §3.1):

1. **Tangent projection** of a Euclidean gradient ``g``:
   ``g_tangent = g − (⟨g, n⟩ / ⟨n, n⟩) · n``
   where ``n_lk = p_lk · (c_lk − E_p[c_l])`` is the gradient of the
   constraint w.r.t. ``α`` (i.e. the constraint normal).

2. **Retraction** onto the manifold via 1-D bisection along the cost
   direction: find ``t`` such that ``Σ_l Σ_k softmax(α_l + t·c_l)_k
   · c_lk = B``. Bracket-doubling until the budget straddles zero, then
   bisection to tolerance.

3. **Vector transport** of Adam's first-moment buffer to the new tangent
   plane after a step + retraction (re-project ``m`` using the same
   formula as the tangent projection on the new ``α``).

Forward pass (fitness evaluation): Gumbel-STE samples a per-layer
argmax, then a **multiple-choice knapsack 1-D DP** projects the argmax
assignment to the closest budget-exact discrete vector under the per-
layer cost grid. Backward flows through Gumbel-softmax probabilities.

Fitness signal (`SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 last bullet):
**output-space MSE** — the same Stage 2 cost the SC row already
optimizes. If a per-layer damage curve
``D_l(k) = output-space MSE damage at k surviving experts`` is supplied
on ``ctx["per_layer_damage_curve"]`` (the future S1_DP plugin would
populate this), RCO reads it directly. When the slot is absent the
plugin falls back to a *synthetic linear-redundancy curve*
``D_l(k) = R̃^l · (per_layer_count_l − k)`` using GRAPE's R̃^l output,
which preserves GRAPE's qualitative ranking — worst case RCO ≈ GRAPE.

Upstream code & licensing
-------------------------
Reference implementation: https://github.com/IST-DASLab/RCO. **The
upstream repo ships without a LICENSE file** (verified 2026-05-27 via
the GitHub API: ``"license": null``). This implementation is therefore
a clean-room re-implementation from the paper's algorithm prose and
from the manifold-primitive descriptions in
``tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md`` §5.3 — no source files are
copied. The repo is cited as an attribution / cross-check target only.
See deviation D-clean-room.

Deviations
==========

D-clean-room — re-implementation, not vendor
--------------------------------------------
The plugin is a clean-room re-implementation of the paper's Algorithm 1
from the prose description; no code is copied from the upstream repo
(which has no LICENSE file). Each algorithmic choice the paper leaves
under-specified is flagged with its own deviation tag below.

D-init-grape — GRAPE initialization (vs paper's REAP saliency)
--------------------------------------------------------------
Paper §3.3 initializes ``α`` from REAP saliency scores. This plugin
initializes from **GRAPE budgets** (`SC_STAGE12_COMPREHENSIVE_PLAN.md`
§7 A5 mandate). Concretely: for each layer ``l`` with GRAPE budget
``g_l ∈ {floor_l, ..., per_layer_count_l}``, the option-index
corresponding to ``g_l`` gets logit ``init_peak_logit`` (default 2.0)
and every other option gets logit 0. ``softmax`` of that gives a sharp
peak on ``g_l`` with small mass on neighbors — RCO can refine.

D-fitness-mse — output-space MSE fitness (vs paper's end-to-end loss)
---------------------------------------------------------------------
Paper §4.2 uses end-to-end task loss as the fitness signal (cheap with
their L1/vLLM rollout substrate). This plugin uses output-space MSE
(`SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 last bullet); the actual-loss
upgrade is gated on `L1_FOR_SC_PLAN.md`. The risk mitigation
(`SC_STAGE12_COMPREHENSIVE_PLAN.md` §9 R3, fitness-vs-bpt_gap mismatch)
is operator-driven: the plugin logs implied + projected discrete
budget vectors so the operator can spot-check rankings through Stage 2.

D-synthetic-curve — synthetic linear-redundancy fallback
--------------------------------------------------------
When no per-layer damage curve is on the ctx (typical Stage-1-only
run; S1_DP is a separate work item), RCO uses
``D_l(k) = R̃^l · (per_layer_count_l − k)`` with R̃^l = GRAPE's
``per_layer_redundancy`` slot. This is a convex decreasing curve in
``k`` that preserves GRAPE's ranking. Worst case: RCO output ≈ GRAPE
(no regression); best case: RCO redistributes 1-2 experts at the
margin where logit gradients agree across iterations. The plugin
emits a WARNING when the fallback fires so operators are not surprised
by quiet behaviour.

D-floor-projection — floor baked into option grid
-------------------------------------------------
``min_experts_per_layer`` is a project invariant (see
``MOE_COMPRESS_REPORT.md`` §5.1; not reopened by this plan). Each
layer's per-layer option grid is restricted to
``{floor_l, floor_l+1, ..., per_layer_count_l}`` so the floor is part
of the manifold's intrinsic geometry — RCO cannot escape it. The
alternative (post-hoc clipping) would break the budget-exact retraction.

D-ragged-K — per-layer K varies; ragged tensors with a mask
-----------------------------------------------------------
Because the floor + per_layer_count combination can differ across
layers (e.g. one MoE layer at 256 experts and one at 128), K_l varies.
The implementation pads the per-layer option grid up to ``K_max =
max_l (per_layer_count_l - floor_l + 1)`` and uses a 0/1 mask to zero
out the padding columns wherever the algorithm computes a sum or
gradient. The retraction bisection is done independently per layer
(no cross-layer coupling beyond the global budget constraint).

D-bisection-budget — joint vs per-layer retraction
--------------------------------------------------
Paper Algorithm 1 line 8 (retraction) is global: one scalar ``t`` such
that the *total* budget equals ``B``. This is the form implemented
here. The upstream repo's ``manifold.py`` exposes a per-layer variant
(``*_per_layer``) for independent per-layer constraints, which is not
applicable here — our constraint is global.

D-disabled-default — opt-in via ``stage1.rco_budget.enabled``
-------------------------------------------------------------
Default ``enabled: false``. The plugin is only consumed by the
``S1_RCO`` row of the S-series ablation; default-off keeps every other
row (S0_GRAPE, SC, SCD, ...) byte-identical to pre-RCO behaviour.
When the flag is false the plugin's ``run`` is never called and
``contribute_artifact`` returns ``{}``.

Output context contract
-----------------------
- ``reads``:
    - ``per_layer_target_experts`` — GRAPE budgets (dict[str, int]).
    - ``per_layer_redundancy`` — GRAPE R̃^l (dict[str, float]),
      consumed by the synthetic-curve fallback.
    - ``per_layer_targets`` — pre-Stage-1 per-layer expert counts
      (dict[int, int]).
    - ``decomposition`` — :class:`BudgetDecomposition` (the global
      budget B is on ``.global_expert_budget``).
    - ``config`` — the run config; this plugin reads
      ``config["stage1"]["rco_budget"]``.

- ``writes``:
    - ``per_layer_target_experts_rco`` — refined budgets
      (dict[str, int]). Distinct slot from GRAPE's so downstream
      consumers explicitly opt in.
    - ``rco_metadata`` — solver-state summary
      (dict with init/final budget vectors, fitness, iter count).

- Optional read (no KeyError if absent):
    - ``per_layer_damage_curve`` — dict[int, dict[int, float]] of
      ``D_l(k)`` per-layer cost curves. When present, RCO uses these
      directly; when absent, the synthetic fallback fires.

- ``contribute_artifact`` returns
  ``{"rco_budgets": <budget dict>, "rco_metadata": <summary>}`` (an
  empty dict when the plugin is disabled). The orchestrator writes
  this to ``stage1_rco_budgets.json`` when enabled.

Naming
------
"S1_RCO" is the ablation-row name in `SC_STAGE12_COMPREHENSIVE_PLAN.md`
§6.1; "rco_budget" is the plugin id.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch

from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


# Numerical tolerances for the budget retraction bisection. Tight enough
# to round to the correct integer after the final DP projection (which
# only needs ~0.5-of-an-expert resolution to disambiguate), loose enough
# to converge in <60 bisection steps for L ≤ 64, K ≤ 256.
_BISECT_TOL = 1e-4
_BISECT_MAX_ITERS = 60
# Cap on the bracket-doubling phase before bisection. 32 doublings span
# 2^31 in either direction — more than enough for any realistic
# (cost, logit) scale.
_BRACKET_MAX_DOUBLINGS = 32


class RCOBudgetPlugin:
    """RCO Stage-1 budget refinement plugin.

    See module docstring for the paper citation (arxiv:2605.00649
    Algorithm 1), the clean-room re-implementation note (upstream
    unlicensed), and the seven deviations:

    - **D-clean-room** (no verbatim vendoring)
    - **D-init-grape** (initialize from GRAPE, not REAP saliency)
    - **D-fitness-mse** (output-space MSE, not end-to-end loss)
    - **D-synthetic-curve** (linear-redundancy fallback when no damage curve)
    - **D-floor-projection** (floor baked into the option grid)
    - **D-ragged-K** (per-layer K varies, padded with mask)
    - **D-bisection-budget** (global retraction, not per-layer)
    - **D-disabled-default** (opt-in via ``stage1.rco_budget.enabled``)
    """

    name: str = "rco_budget"
    paper: str = (
        "RCO: IST-DASLab arxiv:2605.00649 §3 Algorithm 1. "
        "Upstream code at github.com/IST-DASLab/RCO ships without a LICENSE "
        "file; this is a clean-room re-implementation from the paper's prose. "
        "Deviations: D-clean-room (no verbatim vendoring), "
        "D-init-grape (initialize from GRAPE budgets, not REAP saliency), "
        "D-fitness-mse (output-space MSE fitness, not end-to-end loss), "
        "D-synthetic-curve (linear-redundancy fallback when no damage curve), "
        "D-floor-projection (floor baked into per-layer option grid), "
        "D-ragged-K (per-layer K varies, padded with a mask), "
        "D-bisection-budget (global retraction, not per-layer), "
        "D-disabled-default (opt-in via stage1.rco_budget.enabled). "
        "See module docstring for full per-deviation derivations."
    )
    config_key: str = "stage1.rco_budget"
    reads: tuple[str, ...] = (
        "per_layer_target_experts",
        "per_layer_redundancy",
        "per_layer_targets",
        "decomposition",
        "config",
    )
    writes: tuple[str, ...] = (
        "per_layer_target_experts_rco",
        "rco_metadata",
    )
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Gate on ``config["stage1"]["rco_budget"]["enabled"]`` (default False).

        Default OFF so the plugin is a strict no-op for every row that
        does not explicitly request RCO (S0_GRAPE, SC, SCD, ...).
        """
        try:
            return bool(config["stage1"]["rco_budget"]["enabled"])
        except (KeyError, TypeError):
            return False

    def run(self, ctx: PipelineContext) -> None:
        """Refine GRAPE budgets via RCO Algorithm 1.

        Reads ``per_layer_target_experts`` (GRAPE), the optional
        ``per_layer_damage_curve`` (Plugin #8 / S1_DP would populate
        this; absent in typical runs), and the per-layer expert counts;
        writes ``per_layer_target_experts_rco`` + ``rco_metadata``.

        Raises KeyError with the slot name if any required slot is
        missing (no silent degradation — `SC_STAGE12_COMPREHENSIVE_PLAN.md`
        §1 mandate: "errors clearly").
        """
        grape_budgets_str: dict[str, int] = ctx.get("per_layer_target_experts")
        grape_redundancy_str: dict[str, float] = ctx.get("per_layer_redundancy")
        per_layer_counts: dict[int, int] = ctx.get("per_layer_targets")
        decomposition = ctx.get("decomposition")
        config: dict = ctx.get("config")

        rco_cfg: dict = config.get("stage1", {}).get("rco_budget", {})
        n_iterations: int = int(rco_cfg.get("n_iterations", 500))
        learning_rate: float = float(rco_cfg.get("learning_rate", 0.1))
        gumbel_tau_init: float = float(rco_cfg.get("gumbel_tau_init", 5.0))
        gumbel_tau_final: float = float(rco_cfg.get("gumbel_tau_final", 0.5))
        init_peak_logit: float = float(rco_cfg.get("init_peak_logit", 2.0))
        floor_divisor: int = int(rco_cfg.get("floor_divisor", 2))
        seed: int = int(rco_cfg.get("seed", 0))
        adam_beta1: float = float(rco_cfg.get("adam_beta1", 0.9))
        adam_beta2: float = float(rco_cfg.get("adam_beta2", 0.999))
        adam_eps: float = float(rco_cfg.get("adam_eps", 1e-8))

        global_budget: int = int(decomposition.global_expert_budget)

        # Coerce GRAPE outputs (str keys → int).
        grape_budgets: dict[int, int] = {int(k): int(v) for k, v in grape_budgets_str.items()}
        grape_redundancy: dict[int, float] = {
            int(k): float(v) for k, v in grape_redundancy_str.items()
        }

        sorted_layers = sorted(per_layer_counts.keys())
        if not sorted_layers:
            raise ValueError("RCO: per_layer_targets is empty — no MoE layers to allocate.")

        # Build per-layer option grids: k_options[li] = {floor_l, ..., N_l}.
        # D-floor-projection: floor is part of the manifold's intrinsic geometry.
        k_options: dict[int, list[int]] = {}
        for li in sorted_layers:
            N_l = int(per_layer_counts[li])
            floor_l = max(N_l // floor_divisor, 1)
            k_options[li] = list(range(floor_l, N_l + 1))
            if not k_options[li]:
                raise ValueError(
                    f"RCO: layer {li}: option grid empty (N={N_l}, floor={floor_l})."
                )

        # D-ragged-K: pad to K_max with a 0/1 mask.
        K_max = max(len(opts) for opts in k_options.values())
        L = len(sorted_layers)
        layer_to_row = {li: idx for idx, li in enumerate(sorted_layers)}

        # cost_grid[row, k_idx] = surviving expert count for that option,
        # or 0 in padding columns.
        cost_grid = torch.zeros((L, K_max), dtype=torch.float64)
        mask = torch.zeros((L, K_max), dtype=torch.float64)
        for li in sorted_layers:
            row = layer_to_row[li]
            opts = k_options[li]
            for k_idx, k_val in enumerate(opts):
                cost_grid[row, k_idx] = float(k_val)
                mask[row, k_idx] = 1.0

        # Build the damage-cost matrix D[row, k_idx]: fitness contribution
        # of choosing option k for layer li. Damage curve from ctx if
        # present, synthetic linear fallback otherwise.
        damage_grid = self._build_damage_grid(
            ctx=ctx,
            sorted_layers=sorted_layers,
            k_options=k_options,
            K_max=K_max,
            grape_redundancy=grape_redundancy,
            per_layer_counts=per_layer_counts,
        )

        # D-init-grape: initial logits peak at the GRAPE option-index.
        alpha = torch.zeros((L, K_max), dtype=torch.float64)
        for li in sorted_layers:
            row = layer_to_row[li]
            opts = k_options[li]
            g_l = int(grape_budgets[li])
            # Clamp GRAPE budget into the option grid (defensive — GRAPE
            # SHOULD respect the same floor, but bugs happen). Add a
            # warning if clamping fires.
            g_l_clamped = max(opts[0], min(opts[-1], g_l))
            if g_l_clamped != g_l:
                log.warning(
                    "RCO: layer %d: GRAPE budget %d clamped to option grid "
                    "[%d, %d] (floor mismatch with floor_divisor=%d)",
                    li, g_l, opts[0], opts[-1], floor_divisor,
                )
            k_idx = opts.index(g_l_clamped)
            alpha[row, k_idx] = init_peak_logit
            # Make padding columns hard-impossible: large negative logit
            # so softmax probability is ~0 even before mask multiplies it.
            for pad_k in range(len(opts), K_max):
                alpha[row, pad_k] = -1e9

        # Retract initial logits onto the constraint surface — GRAPE's
        # budget is integer-feasible but the soft-budget at τ→0+ may
        # drift; bisection pins ``Σ p · c = B`` at iteration 0.
        alpha = self._retract(alpha, cost_grid, mask, global_budget)

        # Initial fitness + budget vector for the metadata + R3 log lever.
        init_fitness, init_budget_vec = self._evaluate_discrete(
            alpha=alpha,
            cost_grid=cost_grid,
            mask=mask,
            damage_grid=damage_grid,
            k_options=k_options,
            sorted_layers=sorted_layers,
            global_budget=global_budget,
            tau=gumbel_tau_final,
            seed=seed,
            stochastic=False,
        )
        log.info(
            "RCO init: global_budget=%d, fitness=%.6g, budget_sum=%d, "
            "init_iterations=%d, lr=%.3g",
            global_budget, init_fitness, sum(init_budget_vec.values()),
            n_iterations, learning_rate,
        )

        # Adam state.
        m_buf = torch.zeros_like(alpha)
        v_buf = torch.zeros_like(alpha)
        rng = torch.Generator().manual_seed(seed)

        # Main RCO loop.
        for it in range(n_iterations):
            # Cosine anneal Gumbel τ: high → low (explore → exploit).
            progress = it / max(n_iterations - 1, 1)
            tau = gumbel_tau_final + 0.5 * (gumbel_tau_init - gumbel_tau_final) * (
                1.0 + math.cos(math.pi * (1.0 - progress))
            )

            # Forward: Gumbel-STE → soft probabilities → fitness estimate.
            grad = self._gradient_estimate(
                alpha=alpha,
                cost_grid=cost_grid,
                mask=mask,
                damage_grid=damage_grid,
                tau=tau,
                rng=rng,
            )

            # Tangent projection: remove the constraint-normal component.
            normal = self._constraint_normal(alpha, cost_grid, mask)
            grad_tangent = self._project_off_normal(grad, normal, mask)

            # Adam (in tangent space).
            m_buf = adam_beta1 * m_buf + (1.0 - adam_beta1) * grad_tangent
            v_buf = adam_beta2 * v_buf + (1.0 - adam_beta2) * (grad_tangent * grad_tangent)
            m_hat = m_buf / (1.0 - adam_beta1 ** (it + 1))
            v_hat = v_buf / (1.0 - adam_beta2 ** (it + 1))
            step = -learning_rate * m_hat / (torch.sqrt(v_hat) + adam_eps)
            # Padding columns are forced impossible by the very-negative
            # init logit; zero out updates to them so they stay impossible.
            step = step * mask
            alpha = alpha + step

            # Retract onto the manifold (1-D bisection along cost direction).
            alpha = self._retract(alpha, cost_grid, mask, global_budget)

            # Vector transport: re-project Adam's first moment onto the
            # new tangent plane.
            normal_new = self._constraint_normal(alpha, cost_grid, mask)
            m_buf = self._project_off_normal(m_buf, normal_new, mask)

        # Final discrete read: τ → 0 (deterministic argmax) and DP project
        # to budget-exact.
        final_fitness, final_budget_vec = self._evaluate_discrete(
            alpha=alpha,
            cost_grid=cost_grid,
            mask=mask,
            damage_grid=damage_grid,
            k_options=k_options,
            sorted_layers=sorted_layers,
            global_budget=global_budget,
            tau=gumbel_tau_final,
            seed=seed,
            stochastic=False,
        )
        log.info(
            "RCO final: fitness=%.6g (init=%.6g, Δ=%.3g), budget_sum=%d (target=%d), "
            "iterations=%d",
            final_fitness, init_fitness, init_fitness - final_fitness,
            sum(final_budget_vec.values()), global_budget, n_iterations,
        )

        # `SC_STAGE12_COMPREHENSIVE_PLAN.md` §9 R3 inspection lever: log
        # init + final budget-vector hashes so the operator can run the
        # 3 spot-check budget vectors through end-to-end Stage 2.
        init_vec_str = ",".join(str(init_budget_vec[li]) for li in sorted_layers)
        final_vec_str = ",".join(str(final_budget_vec[li]) for li in sorted_layers)
        log.info("RCO init budget vector: %s", init_vec_str)
        log.info("RCO final budget vector: %s", final_vec_str)

        ctx.set(
            "per_layer_target_experts_rco",
            {str(li): int(v) for li, v in final_budget_vec.items()},
        )
        ctx.set(
            "rco_metadata",
            {
                "init_fitness": float(init_fitness),
                "final_fitness": float(final_fitness),
                "init_budget_vector": {str(li): int(v) for li, v in init_budget_vec.items()},
                "final_budget_vector": {str(li): int(v) for li, v in final_budget_vec.items()},
                "n_iterations": int(n_iterations),
                "achieved_budget": int(sum(final_budget_vec.values())),
                "requested_budget": int(global_budget),
                "fitness_source": (
                    "damage_curve" if ctx.has("per_layer_damage_curve") else "synthetic"
                ),
            },
        )

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return the ``stage1_rco_budgets.json`` payload (empty if disabled).

        The orchestrator writes this dict to
        ``artifacts_dir / "stage1_rco_budgets.json"`` ONLY when the
        plugin is enabled; the empty-dict return on disabled paths is
        a defensive belt-and-suspenders so a stray write would produce
        a well-formed empty JSON instead of corrupting state.
        """
        if not ctx.has("per_layer_target_experts_rco"):
            return {}
        return {
            "rco_budgets": ctx.get("per_layer_target_experts_rco"),
            "rco_metadata": ctx.get("rco_metadata"),
        }

    # ------------------------------------------------------------------
    # Damage-curve construction (real curve from ctx OR synthetic fallback)
    # ------------------------------------------------------------------

    def _build_damage_grid(
        self,
        *,
        ctx: PipelineContext,
        sorted_layers: list[int],
        k_options: dict[int, list[int]],
        K_max: int,
        grape_redundancy: dict[int, float],
        per_layer_counts: dict[int, int],
    ) -> torch.Tensor:
        """Build the ``[L, K_max]`` damage grid.

        Primary path: read ``ctx["per_layer_damage_curve"]`` if present
        (a future S1_DP plugin would populate this with output-space MSE
        costs from a Stage-2 profile pass).

        Fallback path (D-synthetic-curve): build a synthetic linear
        curve ``D_l(k) = R̃^l · (per_layer_count_l − k)``. R̃^l is
        GRAPE's per-layer redundancy slot in ``[0, 1]``; the curve is
        zero at ``k = per_layer_count_l`` (no compression, no damage)
        and grows linearly as ``k`` shrinks. The +1 offset on R̃^l
        below ensures even layers with R̃ = 0 get a nonzero
        compression cost so RCO does not see a flat objective.
        """
        L = len(sorted_layers)
        damage_grid = torch.zeros((L, K_max), dtype=torch.float64)

        if ctx.has("per_layer_damage_curve"):
            curve: dict[int, dict[int, float]] = ctx.get("per_layer_damage_curve")
            missing_layers = [li for li in sorted_layers if li not in curve]
            if missing_layers:
                raise ValueError(
                    f"RCO: per_layer_damage_curve is missing entries for layers "
                    f"{missing_layers}; if supplied, the curve must cover every "
                    "MoE layer."
                )
            for row, li in enumerate(sorted_layers):
                layer_curve = curve[li]
                opts = k_options[li]
                missing_k = [k for k in opts if k not in layer_curve]
                if missing_k:
                    raise ValueError(
                        f"RCO: per_layer_damage_curve[{li}] is missing entries "
                        f"for option values {missing_k}; curve must cover "
                        f"{{floor, ..., per_layer_count}}."
                    )
                for k_idx, k_val in enumerate(opts):
                    damage_grid[row, k_idx] = float(layer_curve[k_val])
        else:
            log.warning(
                "RCO: per_layer_damage_curve slot not present on ctx — falling "
                "back to synthetic linear-redundancy curve "
                "D_l(k) = (R̃^l + 1) · (per_layer_count_l - k). This is a "
                "qualitative-rank-preserving fallback; the real damage curve "
                "(Plugin S1_DP) is recommended for production use."
            )
            for row, li in enumerate(sorted_layers):
                # +1 offset so layers with R̃=0 still get a nonzero
                # compression cost — otherwise their gradient is zero
                # and RCO can over-allocate them with no penalty.
                alpha_redundancy = float(grape_redundancy.get(li, 0.0)) + 1.0
                N_l = int(per_layer_counts[li])
                opts = k_options[li]
                for k_idx, k_val in enumerate(opts):
                    damage_grid[row, k_idx] = alpha_redundancy * float(N_l - k_val)

        return damage_grid

    # ------------------------------------------------------------------
    # Manifold primitives (paper §3.1)
    # ------------------------------------------------------------------

    def _masked_softmax(self, alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Row-wise softmax with padding columns masked to 0 probability.

        Padding columns already carry very-negative logits (set at init),
        so the unmasked softmax has near-zero mass on them; the explicit
        mask multiplication + renormalisation guarantees exactly zero.
        """
        # Numerical-stability shift: subtract per-row max before exp.
        alpha_shift = alpha - alpha.max(dim=1, keepdim=True).values
        exp = torch.exp(alpha_shift) * mask
        norm = exp.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return exp / norm

    def _constraint_normal(
        self,
        alpha: torch.Tensor,
        cost_grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient of the constraint Σ_l Σ_k p_lk · c_lk w.r.t. α.

        ``n_lk = p_lk · (c_lk − E_p[c_l])`` where ``E_p[c_l] = Σ_k p_lk · c_lk``.

        Derivation: d/d α_lk of ``softmax(α_l)_j · c_lj`` summed over j:
        ``Σ_j (∂p_lj / ∂α_lk) · c_lj``. With the standard softmax
        Jacobian ``∂p_lj/∂α_lk = p_lj (δ_jk − p_lk)``, this collapses to
        ``p_lk · (c_lk − Σ_j p_lj c_lj)`` = ``p_lk · (c_lk − E_p[c_l])``.
        """
        p = self._masked_softmax(alpha, mask)
        e_c = (p * cost_grid).sum(dim=1, keepdim=True)
        return p * (cost_grid - e_c) * mask

    def _project_off_normal(
        self,
        g: torch.Tensor,
        normal: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Remove the component of ``g`` along ``normal``.

        Standard Gram-Schmidt: ``g_tangent = g − (⟨g, n⟩ / ⟨n, n⟩) · n``.
        Treats the full ``[L, K_max]`` tensor as one vector — the budget
        constraint is global so the projection is global.
        """
        # Zero out padding columns BEFORE the inner product so they
        # do not contribute to either numerator or denominator.
        g_masked = g * mask
        normal_masked = normal * mask
        num = (g_masked * normal_masked).sum()
        den = (normal_masked * normal_masked).sum().clamp_min(1e-30)
        return (g_masked - (num / den) * normal_masked) * mask

    def _budget_residual(
        self,
        alpha: torch.Tensor,
        cost_grid: torch.Tensor,
        mask: torch.Tensor,
        global_budget: float,
    ) -> float:
        """Compute ``(Σ_l Σ_k softmax(α_l)_k · c_lk) − B``.

        Sign matters for the bisection: positive residual means the
        soft-budget exceeds the target so ``α`` should shift *away from*
        high-cost options (subtract t·c), and vice versa.
        """
        p = self._masked_softmax(alpha, mask)
        soft_budget = float((p * cost_grid).sum().item())
        return soft_budget - float(global_budget)

    def _retract(
        self,
        alpha: torch.Tensor,
        cost_grid: torch.Tensor,
        mask: torch.Tensor,
        global_budget: float,
    ) -> torch.Tensor:
        """Bisect along the cost direction to restore Σ p · c = B.

        Parametrise: ``α(t) = α − t · c_grid``. The residual
        ``f(t) = (Σ softmax(α(t)) · c) − B`` is monotonically decreasing
        in ``t`` (higher t penalizes high-cost options more, shrinking
        the soft budget), so a 1-D bisection converges.

        Bracket-doubling phase finds an interval where ``f`` straddles
        zero, then bisection halves it to tolerance.
        """
        res_zero = self._budget_residual(alpha, cost_grid, mask, global_budget)
        if abs(res_zero) <= _BISECT_TOL:
            return alpha

        # If residual > 0, soft budget too high → need t > 0 to shrink it.
        # If residual < 0, soft budget too low → need t < 0 to grow it.
        if res_zero > 0:
            t_lo, t_hi = 0.0, 1.0
            f_lo, f_hi = res_zero, self._budget_residual(
                alpha - t_hi * cost_grid, cost_grid, mask, global_budget
            )
            doublings = 0
            while f_hi > 0 and doublings < _BRACKET_MAX_DOUBLINGS:
                t_lo, f_lo = t_hi, f_hi
                t_hi *= 2.0
                f_hi = self._budget_residual(
                    alpha - t_hi * cost_grid, cost_grid, mask, global_budget
                )
                doublings += 1
            if f_hi > 0:
                log.warning(
                    "RCO retract: bracket-doubling exhausted at t_hi=%.3g "
                    "with f_hi=%.3g still positive; returning best alpha "
                    "(soft-budget overshoot)", t_hi, f_hi,
                )
                return alpha - t_hi * cost_grid
        else:
            t_hi, t_lo = 0.0, -1.0
            f_hi, f_lo = res_zero, self._budget_residual(
                alpha - t_lo * cost_grid, cost_grid, mask, global_budget
            )
            doublings = 0
            while f_lo < 0 and doublings < _BRACKET_MAX_DOUBLINGS:
                t_hi, f_hi = t_lo, f_lo
                t_lo *= 2.0
                f_lo = self._budget_residual(
                    alpha - t_lo * cost_grid, cost_grid, mask, global_budget
                )
                doublings += 1
            if f_lo < 0:
                log.warning(
                    "RCO retract: bracket-doubling exhausted at t_lo=%.3g "
                    "with f_lo=%.3g still negative; returning best alpha "
                    "(soft-budget undershoot)", t_lo, f_lo,
                )
                return alpha - t_lo * cost_grid

        # Bisection.
        for _ in range(_BISECT_MAX_ITERS):
            t_mid = 0.5 * (t_lo + t_hi)
            alpha_mid = alpha - t_mid * cost_grid
            f_mid = self._budget_residual(alpha_mid, cost_grid, mask, global_budget)
            if abs(f_mid) <= _BISECT_TOL:
                return alpha_mid
            # f is decreasing in t: f > 0 means t too small.
            if f_mid > 0:
                t_lo = t_mid
            else:
                t_hi = t_mid
        return alpha - 0.5 * (t_lo + t_hi) * cost_grid

    # ------------------------------------------------------------------
    # Gradient estimator (Gumbel-STE forward + analytic backward)
    # ------------------------------------------------------------------

    def _gradient_estimate(
        self,
        *,
        alpha: torch.Tensor,
        cost_grid: torch.Tensor,
        mask: torch.Tensor,
        damage_grid: torch.Tensor,
        tau: float,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Stochastic gradient of ``E_p[Σ_l D_l(k_l)]`` w.r.t. ``α``.

        Uses the soft Gumbel-softmax to approximate the discrete
        distribution at temperature ``tau``, then computes the analytic
        gradient of ``Σ_l Σ_k p̃_lk · D_lk`` where ``p̃`` is the
        Gumbel-softmax output.

        The straight-through estimator typically uses ``argmax`` in the
        forward + ``softmax`` in the backward. Here we use the soft
        ``p̃`` for both (a Gumbel-softmax relaxation, not strict STE),
        since the fitness signal is a sum over options and the soft
        path gives a lower-variance gradient. This matches the paper's
        practical recipe at moderate τ; the discrete projection happens
        only at the final read-out.
        """
        # Sample Gumbel noise: g_lk = -log(-log(u_lk)) with u ~ Uniform.
        # The shape matches alpha; padding columns get noise too but the
        # mask zeroes them out before softmax.
        u = torch.rand(alpha.shape, generator=rng, dtype=alpha.dtype).clamp_min(1e-20)
        gumbel = -torch.log(-torch.log(u))
        # Padding columns must stay impossible after gumbel noise. We
        # add gumbel to the *masked* logits then re-impose the very-
        # negative pad value, so the soft probabilities on pads are ~0.
        alpha_perturbed = alpha + tau * gumbel
        # Re-impose the pad: where mask == 0, set to a very-negative
        # value so the masked softmax gives ~0 mass.
        very_neg = torch.full_like(alpha_perturbed, -1e9)
        alpha_perturbed = torch.where(mask > 0, alpha_perturbed, very_neg)

        p_tilde = self._masked_softmax(alpha_perturbed, mask)

        # Analytic ∇_α (Σ p̃ · D) under the standard softmax Jacobian:
        # ∂(Σ_j p̃_lj D_lj)/∂α_lk = p̃_lk · (D_lk − E_p̃[D_l]).
        # The 1/tau factor would normally appear because α' = α/tau before
        # softmax, but we treat that as a constant scale absorbed into the
        # Adam learning rate.
        e_d = (p_tilde * damage_grid).sum(dim=1, keepdim=True)
        grad = p_tilde * (damage_grid - e_d) * mask
        return grad

    # ------------------------------------------------------------------
    # Discrete readout (multiple-choice knapsack DP)
    # ------------------------------------------------------------------

    def _evaluate_discrete(
        self,
        *,
        alpha: torch.Tensor,
        cost_grid: torch.Tensor,
        mask: torch.Tensor,
        damage_grid: torch.Tensor,
        k_options: dict[int, list[int]],
        sorted_layers: list[int],
        global_budget: int,
        tau: float,
        seed: int,
        stochastic: bool,
    ) -> tuple[float, dict[int, int]]:
        """Project the soft logits to a budget-exact discrete vector via DP.

        Multiple-choice knapsack with ``L`` items (layers), each having
        ``K_l`` mutually-exclusive options. The DP minimises a *combined*
        score that mixes ``damage_grid`` (the fitness signal we want to
        minimise) with a small negative-log-probability bias toward the
        soft logits (so the discrete projection respects the optimizer's
        belief at small τ).

        Concretely: select one option per layer minimising
        ``Σ_l (D_lk - β · log p_lk)`` subject to ``Σ_l c_lk = B``, with
        β small (default 1e-3). β → 0 recovers a pure damage-DP solve;
        β → ∞ recovers a pure argmax projection. The small default
        means damage dominates, but the soft logits break ties.

        Returns (fitness, budget_vector). budget_vector keyed by layer_idx.
        """
        del stochastic, seed, tau  # kept in signature for future stochastic readout

        # Per-option score: damage + small soft-logit nudge.
        p = self._masked_softmax(alpha, mask)
        log_p = torch.log(p.clamp_min(1e-30))
        beta = 1e-3
        score_grid = damage_grid - beta * log_p
        # Padding columns get +inf score so DP never picks them.
        score_grid_np = score_grid.detach().cpu().numpy().copy()
        mask_np = mask.detach().cpu().numpy()
        score_grid_np[mask_np == 0] = float("inf")
        cost_grid_np = cost_grid.detach().cpu().numpy()

        L = len(sorted_layers)
        B = int(global_budget)

        # DP table: best_score[i][b] = min sum of scores for first i
        # layers using exactly budget b. choice[i][b] = option index
        # taken at layer i to achieve it.
        INF = float("inf")
        best = np.full((L + 1, B + 1), INF, dtype=np.float64)
        choice = np.full((L + 1, B + 1), -1, dtype=np.int64)
        best[0, 0] = 0.0
        for i, li in enumerate(sorted_layers):
            opts = k_options[li]
            K_l = len(opts)
            for b in range(B + 1):
                if not math.isfinite(best[i, b]):
                    continue
                for k_idx in range(K_l):
                    c = int(cost_grid_np[i, k_idx])
                    s = float(score_grid_np[i, k_idx])
                    nb = b + c
                    if nb > B:
                        continue
                    cand = best[i, b] + s
                    if cand < best[i + 1, nb]:
                        best[i + 1, nb] = cand
                        choice[i + 1, nb] = k_idx

        if not math.isfinite(best[L, B]):
            # Infeasibility: no combination of per-layer options hits
            # exactly B. Fall back to the closest feasible budget.
            feasible = [b for b in range(B + 1) if math.isfinite(best[L, b])]
            if not feasible:
                raise ValueError(
                    f"RCO DP: no feasible budget assignment for global_budget={B}."
                )
            chosen_B = max(feasible, key=lambda b: (b, -abs(b - B)))
            log.warning(
                "RCO DP: budget %d infeasible; falling back to nearest feasible "
                "budget %d.", B, chosen_B,
            )
        else:
            chosen_B = B

        # Backtrack.
        budget_vec: dict[int, int] = {}
        b = chosen_B
        for i in range(L, 0, -1):
            li = sorted_layers[i - 1]
            k_idx = int(choice[i, b])
            if k_idx < 0:
                raise RuntimeError(
                    f"RCO DP backtrack failed at layer index {i} budget {b}."
                )
            chosen_k = int(k_options[li][k_idx])
            budget_vec[li] = chosen_k
            b -= chosen_k

        # Final fitness uses the *damage* alone (not the score), so it is
        # comparable across calls regardless of β.
        fitness = 0.0
        for i, li in enumerate(sorted_layers):
            k_idx = k_options[li].index(budget_vec[li])
            fitness += float(damage_grid[i, k_idx].item())
        return fitness, budget_vec


__all__ = ["RCOBudgetPlugin"]
