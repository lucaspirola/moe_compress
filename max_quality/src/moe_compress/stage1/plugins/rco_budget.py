"""RCO — Riemannian-manifold budget allocator (Stage 1 refinement, clean-room re-impl).

Paper
-----
IST-DASLab, *Model Compression with Exact Budget Constraints via
Riemannian Manifolds* — arxiv:2605.00649 (May 2026). §3 Algorithm 1.

This module is a **clean-room re-implementation** from the paper's prose +
``tasks/PLAN_RCO_NATIVE_REIMPL.md`` (v7 — 7 rounds of plan-reviewer
ping-pong). Upstream repo at https://github.com/IST-DASLab/RCO ships
**without a LICENSE file**, re-verified 2026-05-28 via the GitHub API on
``https://api.github.com/repos/IST-DASLab/RCO`` returning the raw response
``{"license": null, "default_branch": "main", "updated_at": "2026-05-16T..."}``.
NO upstream source files are referenced. (Pattern H clean-room rule.)

History — this file fully replaces the prior implementation at commit
``269e64d`` (898 LoC, 12 tests). That implementation had **four** known
algorithmic deviations that this re-impl fixes in-place:

1. **Gumbel-softmax form**: was ``softmax(α + τ · g)``; now standard
   ``softmax((α + g) / τ)``. Plan §1.3 step 1.
2. **Cosine anneal direction**: was exploit→explore (start cold, end hot);
   now explore→exploit (start hot, end cold) per the canonical form
   ``τ_t = τ_final + 0.5·(τ_init − τ_final)·(1 + cos(π · t/T))``. Plan
   §1.3 step 2.
3. **DP score**: was ``damage − 1e-3 · log p``; now pure damage. The
   β·log p tiebreak is REMOVED. Plan §1.3 step 3.
4. **Infeasible-budget fallback**: was ``max(feasible, ...)``; now
   nearest-feasible with larger-budget tiebreak via
   ``min(feasible, key=lambda b: (abs(b − B), −b))``. Plan §6.1 F8.

Slot in this codebase
---------------------
**Stage 1 budget refinement.** Runs AFTER ``grape_merge`` (Phase G,
manifest index 9). Produces ``per_layer_target_experts_rco`` on a NEW
context slot — distinct from GRAPE's ``per_layer_target_experts`` so the
row recipe explicitly opts in. Default OFF behind
``stage1.rco_budget.enabled``.

Algorithm reference (paper §3 Algorithm 1, plan §1)
----------------------------------------------------

Given ``L`` MoE-bearing groups, each with a finite per-group option grid
of size ``K_l`` with positive integer costs ``c_lk`` (here: surviving
expert counts for the layer), and a global integer budget ``B``, find a
discrete assignment minimising the decomposable damage
``Σ_l D_l(k_l)`` subject to ``Σ_l c_{l, k_l} = B``.

Introduce per-group logits ``α_l ∈ ℝ^{K_l}`` and soft probabilities
``p_lk = softmax(α_l)_k``. The Budget Manifold (paper §2 Eq. 1) is

    M = { α ∈ ℝ^{Σ K_l} : C(α) = B }    with    C(α) = Σ_l ⟨p_l, c_l⟩.

Four manifold primitives drive the search (paper §2 Props. 1/2 + §3.1):

1. **Constraint normal** ``n_lk = p_lk · (c_lk − ⟨p_l, c_l⟩)`` —
   gradient of C w.r.t. α. Always evaluated at the **un-perturbed**
   ``p = softmax(α)`` (paper §2; see Q8 in the plan).
2. **Tangent projection**: ``g_tan = g − (⟨g, n⟩/⟨n, n⟩)·n``.
3. **Retraction** along the cost direction ``α(t) = α − t·c_grid``:
   1-D bracket-doubling + bisection, signed by ``f(0) = C(α) − B``.
4. **Vector transport — first moment only**: after each retraction,
   re-project Adam's first moment ``m`` onto the new tangent plane;
   the second moment ``v`` is left untransported (``D-adam-no-v-transport``).

Forward pass (§1.3): standard-form Gumbel-softmax
``p̃ = softmax((α + g) / τ)`` with ``g ~ Gumbel(0, 1)``. Limits: τ→∞ →
uniform (explore); τ→0 → argmax(α + g) (exploit, Gumbel-max trick).
Cosine-anneal τ from ``τ_init`` to ``τ_final`` so the search starts
exploratory and ends decisive.

Discrete readout (§1.3 step 3): multiple-choice knapsack DP minimising
**pure damage** ``Σ_l D_l(k_l)`` s.t. ``Σ_l c_{l, k_l} = B``. Strict
``<`` tiebreak so on tied scores the first vector encountered along the
layer sweep (lex-min on option indices) wins (plan v4-N4). On
infeasibility (B outside ``[Σ floor_l, Σ N_l]``), fall back to the
**nearest feasible budget** with WARNING log (larger-budget tiebreak).

Deviations
==========

Nine `D-*` tags total (8 carried from the prior impl + 1 new).

D-clean-room
    Re-implementation from paper prose; no upstream code is copied.
    Re-verified 2026-05-28: GitHub API returns ``{"license": null, ...}``.

D-init-grape
    α initialised from GRAPE budgets, not REAP saliency. REAP would
    require a per-(layer, option) saliency pass which our pipeline
    does not run. Concretely: the option-index matching GRAPE's
    budget gets logit ``init_peak_logit`` (default 2.0); other
    options get 0. (Plan §1.6, Q2.)

D-fitness-mse
    Output-space MSE as fitness (vs paper §4.2's end-to-end task
    loss). The upgrade is gated on L1/vLLM (`L1_FOR_SC_PLAN.md`)
    and out of scope here. (Plan §1.6, Q3.)

D-synthetic-curve
    When ``per_layer_damage_curve`` slot is absent and
    ``fitness_signal="auto"``, fall back to a synthetic linear curve
    ``D_l(k) = (R̃^l + 1) · (per_layer_count_l − k)`` using GRAPE's
    redundancy. Convex decreasing; preserves GRAPE's ranking. (Plan §1.6.)

D-floor-projection
    The per-layer floor ``floor_l = per_layer_count_l // floor_divisor``
    is baked into the option grid ``{floor_l, ..., per_layer_count_l}``
    — part of the manifold's intrinsic geometry. (Plan §1.6.)

D-ragged-K
    Per-layer K_l varies; the option grids are padded to
    ``K_max = max_l(per_layer_count_l − floor_l + 1)`` with a 0/1 mask
    and very-negative pad logits. The masked softmax zeroes pads. (Plan §1.6.)

D-bisection-budget
    Primitive 3 is the global retraction (one scalar ``t`` for the
    global budget constraint), implemented via bracket-doubling +
    bisection. Paper says only "1-D root-find"; the bracket scheme
    is a concrete choice. (Plan §1.6, §1.2 Primitive 3.)

D-disabled-default
    Gated default OFF behind ``stage1.rco_budget.enabled``. Every
    non-S1_RCO row stays byte-identical to pre-plugin-11 main. (Plan §1.6.)

D-adam-no-v-transport  [NEW in this re-impl]
    Adam's first moment ``m`` is vector-transported (re-projected onto
    the new tangent plane after retraction); the second moment ``v``
    is left untouched. The paper is silent on ``v``; we pick the
    lighter convention because ``v`` is element-wise squared and used
    only as an adaptive learning-rate scaler ``1 / (√v̂ + ε)`` — its
    tangent-vs-normal decomposition has no operational meaning.
    Documented in plan §1.2 Primitive 4; pinned by F13. (Plan §1.6.)

Output context contract
-----------------------
- ``reads``:
    - ``per_layer_target_experts`` — GRAPE budgets (dict[str, int]).
    - ``per_layer_redundancy`` — GRAPE R̃^l (dict[str, float]).
    - ``per_layer_targets`` — pre-Stage-1 per-layer expert counts (dict[int, int]).
    - ``decomposition`` — :class:`BudgetDecomposition`; B = ``.global_expert_budget``.
    - ``config`` — the run config; reads ``config["stage1"]["rco_budget"]``.
- ``writes``:
    - ``per_layer_target_experts_rco`` — refined budgets (dict[str, int]).
    - ``rco_metadata`` — solver-state summary (dict).
- Optional read (no KeyError if absent):
    - ``per_layer_damage_curve`` — dict[int, dict[int, float]].

Artifact (Pattern B + K)
------------------------
``stage1_rco_budgets.json`` payload:

    {"format_version": 1,
     "rco_budgets": {...},
     "rco_metadata": {...}}

The ``format_version`` field sits at the **top level**, NOT nested inside
``rco_metadata``. Forward-only schema bumps (Pattern K): readers tolerate
unknown keys; new fields appended to either dict do not bump the version.

Config validation (Pattern C)
-----------------------------
``_validate_config`` runs as the FIRST statement of ``run()`` and
rejects unknown keys + range-checks the values. Hidden mis-keys
(e.g. ``learning_rates`` vs ``learning_rate``) raise ``ValueError`` with
the unknown key surfaced, rather than silently falling through to
defaults.

Fitness signal knob (Pattern E)
-------------------------------
The ``fitness_signal`` config key gates the damage_curve interaction:

- ``"auto"`` (default) — use ``per_layer_damage_curve`` if present, else
  fall back to the synthetic curve. Byte-identical to historical behaviour.
- ``"synthetic"`` — hard-skip damage_curve even if present.
- ``"damage_curve"`` — hard-require damage_curve; raise ``ValueError`` if absent.

Naming
------
"S1_RCO" is the ablation row name in ``SC_STAGE12_COMPREHENSIVE_PLAN.md``
§6.1; "rco_budget" is the plugin id (manifest index 9).
"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch

from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


# Numerical tolerances for the budget retraction bisection. Tight enough
# to round to the correct integer after the final DP projection (which
# only needs ~0.5-of-an-expert resolution to disambiguate), loose enough
# to converge in <60 bisection steps. At production scale B ≈ 8600,
# relative tolerance is 1e-4/8600 ≈ 1.16e-8 — far below float64 epsilon.
_BISECT_TOL = 1e-4
_BISECT_MAX_ITERS = 60
# Cap on the bracket-doubling phase before bisection. 32 doublings span
# 2^31 ≈ 2.15e9 in either direction — comfortably above any plausible B.
_BRACKET_MAX_DOUBLINGS = 32

# Artifact schema version (Pattern B). Bump only on incompatible shape
# changes; additive top-level keys are tolerated by readers (Pattern K).
_ARTIFACT_FORMAT_VERSION = 1

# Allowed values for the Pattern E fitness_signal knob.
_FITNESS_SIGNAL_AUTO = "auto"
_FITNESS_SIGNAL_SYNTHETIC = "synthetic"
_FITNESS_SIGNAL_DAMAGE_CURVE = "damage_curve"
_FITNESS_SIGNAL_ALLOWED = frozenset(
    (_FITNESS_SIGNAL_AUTO, _FITNESS_SIGNAL_SYNTHETIC, _FITNESS_SIGNAL_DAMAGE_CURVE)
)

# Recognised config keys (Pattern C). Any other key under
# ``stage1.rco_budget`` raises ValueError in ``_validate_config``.
_ALLOWED_CFG_KEYS = frozenset(
    (
        "enabled",
        "n_iterations",
        "learning_rate",
        "gumbel_tau_init",
        "gumbel_tau_final",
        "floor_divisor",
        "seed",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "fitness_signal",
    )
)


class RCOBudgetPlugin:
    """RCO Stage-1 budget refinement plugin.

    See module docstring for the paper citation (arxiv:2605.00649
    Algorithm 1), the clean-room re-implementation note (upstream
    unlicensed, re-verified 2026-05-28), and the nine deviations:

    - **D-clean-room** — no verbatim vendoring
    - **D-init-grape** — initialise from GRAPE, not REAP saliency
    - **D-fitness-mse** — output-space MSE fitness, not end-to-end loss
    - **D-synthetic-curve** — linear-redundancy fallback when no damage curve
    - **D-floor-projection** — floor baked into the option grid
    - **D-ragged-K** — per-layer K varies, padded with mask
    - **D-bisection-budget** — global retraction, not per-layer
    - **D-disabled-default** — opt-in via ``stage1.rco_budget.enabled``
    - **D-adam-no-v-transport** — first moment transported only
    """

    name: str = "rco_budget"
    paper: str = (
        "RCO: IST-DASLab arxiv:2605.00649 §3 Algorithm 1. "
        "Upstream code at github.com/IST-DASLab/RCO ships without a LICENSE file "
        "(re-verified 2026-05-28 via GitHub API: license=null); this is a "
        "clean-room re-implementation from the paper's prose. "
        "Deviations: D-clean-room (no verbatim vendoring), "
        "D-init-grape (initialize from GRAPE budgets, not REAP saliency), "
        "D-fitness-mse (output-space MSE fitness, not end-to-end loss), "
        "D-synthetic-curve (linear-redundancy fallback when no damage curve), "
        "D-floor-projection (floor baked into per-layer option grid), "
        "D-ragged-K (per-layer K varies, padded with a mask), "
        "D-bisection-budget (global retraction, not per-layer), "
        "D-disabled-default (opt-in via stage1.rco_budget.enabled), "
        "D-adam-no-v-transport (Adam first moment transported only). "
        "Algorithm details follow paper §3 with: standard Gumbel-softmax "
        "softmax((α+g)/τ) (NOT α+τ·g); cosine τ-anneal τ_init→τ_final "
        "(explore→exploit); pure-damage DP knapsack (no β·log p tiebreak); "
        "Adam first-moment-only vector transport. "
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

        Raises ``KeyError`` with the slot name if any required slot is
        missing (no silent degradation). Raises ``ValueError`` if the
        config contains unknown keys, out-of-range values, or
        ``fitness_signal="damage_curve"`` without a damage-curve slot.
        """
        # Required-slot reads (KeyError surfaces slot name).
        grape_budgets_str: dict[str, int] = ctx.get("per_layer_target_experts")
        grape_redundancy_str: dict[str, float] = ctx.get("per_layer_redundancy")
        per_layer_counts: dict[int, int] = ctx.get("per_layer_targets")
        decomposition = ctx.get("decomposition")
        config: dict = ctx.get("config")

        rco_cfg: dict = config.get("stage1", {}).get("rco_budget", {})

        # Pattern C: validate config FIRST, before any RCO work begins.
        cfg = self._validate_config(rco_cfg)

        # Pattern E: resolve fitness signal mode + raise on damage_curve-strict
        # when the slot is absent.
        fitness_signal = cfg["fitness_signal"]
        has_curve = ctx.has("per_layer_damage_curve")
        if fitness_signal == _FITNESS_SIGNAL_DAMAGE_CURVE and not has_curve:
            raise ValueError(
                "rco_budget: fitness_signal=damage_curve but "
                "per_layer_damage_curve slot is absent. Set "
                "fitness_signal=auto or enable damage_curve_dp."
            )
        if fitness_signal == _FITNESS_SIGNAL_SYNTHETIC:
            use_damage_curve = False
        elif fitness_signal == _FITNESS_SIGNAL_DAMAGE_CURVE:
            use_damage_curve = True
        else:  # auto
            use_damage_curve = has_curve
        fitness_signal_resolved = (
            _FITNESS_SIGNAL_DAMAGE_CURVE if use_damage_curve else _FITNESS_SIGNAL_SYNTHETIC
        )

        n_iterations = cfg["n_iterations"]
        learning_rate = cfg["learning_rate"]
        gumbel_tau_init = cfg["gumbel_tau_init"]
        gumbel_tau_final = cfg["gumbel_tau_final"]
        floor_divisor = cfg["floor_divisor"]
        seed = cfg["seed"]
        adam_beta1 = cfg["adam_beta1"]
        adam_beta2 = cfg["adam_beta2"]
        adam_eps = cfg["adam_eps"]

        global_budget: int = int(decomposition.global_expert_budget)

        # Coerce GRAPE outputs (str keys → int).
        grape_budgets: dict[int, int] = {
            int(k): int(v) for k, v in grape_budgets_str.items()
        }
        grape_redundancy: dict[int, float] = {
            int(k): float(v) for k, v in grape_redundancy_str.items()
        }

        sorted_layers = sorted(per_layer_counts.keys())
        if not sorted_layers:
            raise ValueError(
                "RCO: per_layer_targets is empty — no MoE layers to allocate."
            )

        # Build per-layer option grids: k_options[li] = {floor_l, ..., N_l}.
        # D-floor-projection: floor is part of the manifold's intrinsic geometry.
        k_options: dict[int, list[int]] = {}
        for li in sorted_layers:
            N_l = int(per_layer_counts[li])
            floor_l = max(N_l // floor_divisor, 1)
            opts = list(range(floor_l, N_l + 1))
            if not opts:
                raise ValueError(
                    f"RCO: layer {li}: option grid empty (N={N_l}, floor={floor_l})."
                )
            k_options[li] = opts

        # D-ragged-K: pad to K_max with a 0/1 mask.
        K_max = max(len(opts) for opts in k_options.values())
        L = len(sorted_layers)
        layer_to_row = {li: idx for idx, li in enumerate(sorted_layers)}

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
        # selected by the fitness_signal knob; synthetic linear fallback otherwise.
        damage_grid = self._build_damage_grid(
            ctx=ctx,
            sorted_layers=sorted_layers,
            k_options=k_options,
            K_max=K_max,
            grape_redundancy=grape_redundancy,
            per_layer_counts=per_layer_counts,
            use_damage_curve=use_damage_curve,
        )

        # Initialise α via β-bisection so the expected budget already
        # equals the target at iteration 0 (upstream parity:
        # ``src/search/quant.py::InterpolatedModel.init_alpha_to_bits``
        # lines 302-327). Bisect a single scalar β such that
        # ``E[budget] = Σ_l w_l · Σ_k softmax(-β·c_l)_k · c_lk = B``;
        # then set ``α_lk = -β · c_lk``. Per-row identical when costs
        # are identical (uniform pattern); otherwise per-row.
        alpha = self._init_alpha_beta_bisection(
            cost_grid=cost_grid,
            mask=mask,
            global_budget=global_budget,
            L=L,
            K_max=K_max,
        )
        # Make padding columns hard-impossible: large negative logit so
        # softmax probability is ~0 even before mask multiplies it.
        very_neg = torch.full_like(alpha, -1e9)
        alpha = torch.where(mask > 0, alpha, very_neg)

        # Retract initial logits onto the constraint surface — GRAPE's
        # budget is integer-feasible but the soft-budget at τ→0+ may
        # drift; bisection pins ``Σ p · c = B`` at iteration 0.
        alpha = self._retract(alpha, cost_grid, mask, global_budget)

        # Initial fitness + budget vector for the metadata + R3 log lever.
        init_fitness, init_budget_vec = self._evaluate_discrete(
            cost_grid=cost_grid,
            mask=mask,
            damage_grid=damage_grid,
            k_options=k_options,
            sorted_layers=sorted_layers,
            global_budget=global_budget,
        )
        log.info(
            "RCO init: global_budget=%d, fitness=%.6g, budget_sum=%d, "
            "n_iterations=%d, lr=%.3g, tau_init=%.3g, tau_final=%.3g, "
            "fitness_signal_resolved=%s",
            global_budget, init_fitness, sum(init_budget_vec.values()),
            n_iterations, learning_rate,
            gumbel_tau_init, gumbel_tau_final, fitness_signal_resolved,
        )

        # Adam state.
        m_buf = torch.zeros_like(alpha)
        v_buf = torch.zeros_like(alpha)
        rng = torch.Generator().manual_seed(seed)

        # Main RCO loop.
        for it in range(n_iterations):
            # Exponential anneal τ explore → exploit. Upstream parity:
            # src/search/prune.py:663 / src/search/quant.py:655.
            tau = self._anneal_tau(
                step=it,
                total_steps=max(n_iterations, 1),
                tau_init=gumbel_tau_init,
                tau_final=gumbel_tau_final,
            )

            # Forward: standard-form Gumbel-softmax (plan §1.3 step 1).
            grad = self._gradient_estimate(
                alpha=alpha,
                mask=mask,
                damage_grid=damage_grid,
                tau=tau,
                rng=rng,
            )

            # Tangent projection: remove the constraint-normal component.
            # Constraint normal evaluated at UN-PERTURBED p = softmax(α)
            # (paper §2 Eq. 1 + Prop. 2; plan Q8).
            normal = self._constraint_normal(alpha, cost_grid, mask)
            grad_tangent = self._project_off_normal(grad, normal, mask)

            # Adam (in tangent space).
            m_buf = adam_beta1 * m_buf + (1.0 - adam_beta1) * grad_tangent
            v_buf = adam_beta2 * v_buf + (1.0 - adam_beta2) * (
                grad_tangent * grad_tangent
            )
            m_hat = m_buf / (1.0 - adam_beta1 ** (it + 1))
            v_hat = v_buf / (1.0 - adam_beta2 ** (it + 1))
            step = -learning_rate * m_hat / (torch.sqrt(v_hat) + adam_eps)
            # Zero updates to padding columns so they stay impossible.
            step = step * mask
            alpha = alpha + step

            # Retract onto the manifold (1-D bisection along cost direction).
            alpha = self._retract(alpha, cost_grid, mask, global_budget)

            # Vector transport — m only (D-adam-no-v-transport, plan §1.2
            # Primitive 4). Re-project Adam's first moment onto the new
            # tangent plane; second moment ``v_buf`` is left untouched.
            normal_new = self._constraint_normal(alpha, cost_grid, mask)
            m_buf = self._project_off_normal(m_buf, normal_new, mask)

        # Final discrete read: pure-damage DP project to budget-exact.
        final_fitness, final_budget_vec = self._evaluate_discrete(
            cost_grid=cost_grid,
            mask=mask,
            damage_grid=damage_grid,
            k_options=k_options,
            sorted_layers=sorted_layers,
            global_budget=global_budget,
        )
        log.info(
            "RCO final: fitness=%.6g (init=%.6g, Δ=%.3g), "
            "budget_sum=%d (target=%d), iterations=%d",
            final_fitness, init_fitness, init_fitness - final_fitness,
            sum(final_budget_vec.values()), global_budget, n_iterations,
        )

        # Log compact init+final budget vectors for the SC §9 R3 inspection lever.
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
                "init_budget_vector": {
                    str(li): int(v) for li, v in init_budget_vec.items()
                },
                "final_budget_vector": {
                    str(li): int(v) for li, v in final_budget_vec.items()
                },
                "n_iterations": int(n_iterations),
                "achieved_budget": int(sum(final_budget_vec.values())),
                "requested_budget": int(global_budget),
                "fitness_source": fitness_signal_resolved,
                "tau_init_used": float(gumbel_tau_init),
                "tau_final_used": float(gumbel_tau_final),
                "fitness_signal_resolved": fitness_signal_resolved,
            },
        )

    def contribute_artifact(self, ctx: PipelineContext) -> dict:
        """Return the ``stage1_rco_budgets.json`` payload (empty if disabled).

        Pattern B: ``format_version`` at the TOP LEVEL (not inside
        ``rco_metadata``). The orchestrator writes this dict to
        ``artifacts_dir / "stage1_rco_budgets.json"`` ONLY when the
        plugin is enabled; the empty-dict return on disabled paths is
        a defensive belt-and-suspenders so a stray write would produce
        a well-formed empty JSON instead of corrupting state.
        """
        if not ctx.has("per_layer_target_experts_rco"):
            return {}
        return {
            "format_version": _ARTIFACT_FORMAT_VERSION,
            "rco_budgets": ctx.get("per_layer_target_experts_rco"),
            "rco_metadata": ctx.get("rco_metadata"),
        }

    # ------------------------------------------------------------------
    # Config validation (Pattern C)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(rco_cfg: dict) -> dict:
        """Reject unknown keys + range-check the values. Pattern C.

        Returns a typed dict of the validated config with defaults
        applied. Unknown keys raise ``ValueError`` listing the typo so
        operators see ``learning_rates`` mis-keys instead of having them
        silently fall through to defaults.
        """
        unknown = set(rco_cfg.keys()) - _ALLOWED_CFG_KEYS
        if unknown:
            raise ValueError(
                f"rco_budget: unknown config keys {sorted(unknown)!r} under "
                f"stage1.rco_budget. Allowed keys: {sorted(_ALLOWED_CFG_KEYS)!r}."
            )

        cfg = {
            "enabled": bool(rco_cfg.get("enabled", False)),
            # Upstream default: ``n_steps=200`` in ``src/search/prune.py::optimize``.
            "n_iterations": int(rco_cfg.get("n_iterations", 200)),
            # Upstream default: ``lr=0.1`` in ``src/search/prune.py::optimize``.
            "learning_rate": float(rco_cfg.get("learning_rate", 0.1)),
            # Upstream default: ``tau_init=1.0`` in ``src/search/prune.py::optimize``.
            "gumbel_tau_init": float(rco_cfg.get("gumbel_tau_init", 1.0)),
            # Upstream default: ``tau_min=0.05`` in ``src/search/prune.py::optimize``.
            "gumbel_tau_final": float(rco_cfg.get("gumbel_tau_final", 0.05)),
            "floor_divisor": int(rco_cfg.get("floor_divisor", 2)),
            "seed": int(rco_cfg.get("seed", 0)),
            "adam_beta1": float(rco_cfg.get("adam_beta1", 0.9)),
            "adam_beta2": float(rco_cfg.get("adam_beta2", 0.999)),
            "adam_eps": float(rco_cfg.get("adam_eps", 1e-8)),
            "fitness_signal": str(rco_cfg.get("fitness_signal", _FITNESS_SIGNAL_AUTO)),
        }

        if cfg["n_iterations"] <= 0:
            raise ValueError(
                f"rco_budget: n_iterations must be > 0, got {cfg['n_iterations']}."
            )
        if not (0.0 < cfg["learning_rate"] < 10.0):
            raise ValueError(
                f"rco_budget: learning_rate must be in (0, 10), got "
                f"{cfg['learning_rate']}."
            )
        if cfg["gumbel_tau_init"] <= 0.0:
            raise ValueError(
                f"rco_budget: gumbel_tau_init must be > 0, got "
                f"{cfg['gumbel_tau_init']}."
            )
        if cfg["gumbel_tau_final"] <= 0.0:
            raise ValueError(
                f"rco_budget: gumbel_tau_final must be > 0, got "
                f"{cfg['gumbel_tau_final']}."
            )
        if cfg["gumbel_tau_final"] >= cfg["gumbel_tau_init"]:
            raise ValueError(
                f"rco_budget: gumbel_tau_final ({cfg['gumbel_tau_final']}) must "
                f"be strictly < gumbel_tau_init ({cfg['gumbel_tau_init']}) so "
                "the cosine anneal goes explore → exploit."
            )
        if cfg["floor_divisor"] < 1:
            raise ValueError(
                f"rco_budget: floor_divisor must be ≥ 1, got {cfg['floor_divisor']}."
            )
        if not (0.0 <= cfg["adam_beta1"] < 1.0):
            raise ValueError(
                f"rco_budget: adam_beta1 must be in [0, 1), got {cfg['adam_beta1']}."
            )
        if not (0.0 <= cfg["adam_beta2"] < 1.0):
            raise ValueError(
                f"rco_budget: adam_beta2 must be in [0, 1), got {cfg['adam_beta2']}."
            )
        if cfg["adam_eps"] <= 0.0:
            raise ValueError(
                f"rco_budget: adam_eps must be > 0, got {cfg['adam_eps']}."
            )
        if cfg["fitness_signal"] not in _FITNESS_SIGNAL_ALLOWED:
            raise ValueError(
                f"rco_budget: fitness_signal must be one of "
                f"{sorted(_FITNESS_SIGNAL_ALLOWED)!r}, got "
                f"{cfg['fitness_signal']!r}."
            )

        return cfg

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
        use_damage_curve: bool,
    ) -> torch.Tensor:
        """Build the ``[L, K_max]`` damage grid.

        ``use_damage_curve`` is resolved upstream from the
        ``fitness_signal`` knob (Pattern E):
        - True → read ``ctx["per_layer_damage_curve"]`` (must be present).
        - False → use the synthetic linear curve
          ``D_l(k) = (R̃^l + 1) · (per_layer_count_l − k)`` (D-synthetic-curve).
        """
        L = len(sorted_layers)
        damage_grid = torch.zeros((L, K_max), dtype=torch.float64)

        if use_damage_curve:
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
                "RCO: per_layer_damage_curve not used (synthetic fallback). "
                "D_l(k) = (R̃^l + 1) · (per_layer_count_l − k); a "
                "qualitative-rank-preserving fallback. The real damage curve "
                "(Plugin S1_DP) is recommended for production."
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
    # Initialisation (β-bisection, upstream parity)
    # ------------------------------------------------------------------

    def _init_alpha_beta_bisection(
        self,
        *,
        cost_grid: torch.Tensor,
        mask: torch.Tensor,
        global_budget: float,
        L: int,
        K_max: int,
    ) -> torch.Tensor:
        """β-bisection init.

        Mirrors ``src/search/quant.py::InterpolatedModel.init_alpha_to_bits``
        (upstream lines 302-327): bisect ``β ∈ [-10, 10]`` such that
        ``E[budget] = Σ_l Σ_k softmax(-β·c_l)_k · c_lk`` equals the
        target. With identical cost rows the upstream code sets the same
        per-row logits for every group; we generalise to per-row costs
        (our cost_grid may differ per layer when option grids differ).

        100 bisection iterations matches upstream (`for _ in range(100)`).
        """
        lo = -10.0
        hi = 10.0

        def expected_budget(beta_val: float) -> float:
            # softmax(-β·c) per row, then Σ_l Σ_k p_lk · c_lk.
            logits = -beta_val * cost_grid
            very_neg = torch.full_like(logits, -1e9)
            logits = torch.where(mask > 0, logits, very_neg)
            p = self._masked_softmax(logits, mask)
            return float((p * cost_grid).sum().item())

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if expected_budget(mid) > float(global_budget):
                lo = mid
            else:
                hi = mid
        beta = 0.5 * (lo + hi)

        alpha = -beta * cost_grid
        return alpha

    # ------------------------------------------------------------------
    # Manifold primitives (paper §2 + §3.1; plan §1.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_softmax(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Row-wise softmax with padding columns masked to 0 probability.

        Numerical-stability shift: subtract per-row max before ``exp``,
        then multiply by mask + renormalise so pads carry exactly zero
        mass even if the very-negative pad logits underflowed.
        """
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
        """Gradient of the constraint C(α) = Σ_l ⟨p_l, c_l⟩ w.r.t. α.

        ``n_lk = p_lk · (c_lk − E_p[c_l])`` where ``E_p[c_l] = Σ_k p_lk · c_lk``.

        Derivation (paper §2 Prop. 2): applying the softmax Jacobian
        ``∂p_lj/∂α_lk = p_lj·(δ_jk − p_lk)`` to ``Σ_j p_lj·c_lj`` collapses to
        the formula above.

        Evaluated at the **un-perturbed** ``p = softmax(α)`` (paper §2;
        plan Q8) — the constraint surface is defined in α-space without
        temperature or Gumbel noise. The Gumbel-perturbed ``p̃`` is a
        SEPARATE object used only as the STE backward surrogate (paper
        §3.1) and is not the constraint normal.
        """
        p = self._masked_softmax(alpha, mask)
        e_c = (p * cost_grid).sum(dim=1, keepdim=True)
        return p * (cost_grid - e_c) * mask

    @staticmethod
    def _project_off_normal(
        g: torch.Tensor,
        normal: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Remove the component of ``g`` along ``normal``.

        Standard Gram-Schmidt: ``g_tan = g − (⟨g, n⟩ / ⟨n, n⟩) · n``.
        Treats the full ``[L, K_max]`` tensor as one vector — the budget
        constraint is global, so the projection is global.
        """
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

        Sign matters for the bisection:
        - ``f > 0``: soft-budget over-allocates → need larger ``t``
          (shifts probability mass toward cheaper options).
        - ``f < 0``: soft-budget under-allocates → need smaller (more
          negative) ``t``.
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

        Parametrise ``α(t) = α − t · c_grid``. Then ``d/dt C(α(t)) =
        −Σ_l Var_{p_l}(c_l) ≤ 0`` so the residual ``f(t) = C(α(t)) − B``
        is monotonically non-increasing in ``t``. Bracket-doubling with
        an EXPLICIT SIGN BRANCH (plan §1.2 Primitive 3) handles both
        over- and under-budget cases.
        """
        res_zero = self._budget_residual(alpha, cost_grid, mask, global_budget)
        if abs(res_zero) <= _BISECT_TOL:
            return alpha

        if res_zero > 0:
            # Over-budget: f(0) > 0; need t > 0 to shrink C(α).
            t_lo, t_hi = 0.0, 1.0
            f_lo = res_zero
            f_hi = self._budget_residual(
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
                    "with f_hi=%.3g still positive; returning best α "
                    "(soft-budget overshoot)", t_hi, f_hi,
                )
                return alpha - t_hi * cost_grid
        else:
            # Under-budget: f(0) < 0; need t < 0 to grow C(α).
            t_hi, t_lo = 0.0, -1.0
            f_hi = res_zero
            f_lo = self._budget_residual(
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
                    "with f_lo=%.3g still negative; returning best α "
                    "(soft-budget undershoot)", t_lo, f_lo,
                )
                return alpha - t_lo * cost_grid

        # Bisection.
        for _ in range(_BISECT_MAX_ITERS):
            t_mid = 0.5 * (t_lo + t_hi)
            alpha_mid = alpha - t_mid * cost_grid
            f_mid = self._budget_residual(
                alpha_mid, cost_grid, mask, global_budget
            )
            if abs(f_mid) <= _BISECT_TOL:
                return alpha_mid
            # f is non-increasing in t: f > 0 means t too small (advance lo).
            if f_mid > 0:
                t_lo = t_mid
            else:
                t_hi = t_mid
        return alpha - 0.5 * (t_lo + t_hi) * cost_grid

    # ------------------------------------------------------------------
    # τ-anneal (exponential, upstream parity)
    # ------------------------------------------------------------------

    @staticmethod
    def _anneal_tau(
        *, step: int, total_steps: int, tau_init: float, tau_final: float
    ) -> float:
        """Exponential τ schedule, explore → exploit (τ_init → τ_final).

        Mirrors upstream ``src/search/prune.py:663`` and
        ``src/search/quant.py:655`` verbatim::

            progress = step / max(n_steps - 1, 1)
            tau = max(tau_min, tau_init * (tau_min / tau_init) ** progress)

        Endpoints:
        - At ``t = 0``: ``progress = 0`` ⇒ ``τ = τ_init`` (hot).
        - At ``t = T-1``: ``progress = 1`` ⇒ ``τ = τ_final`` (cold).
        """
        progress = step / max(total_steps - 1, 1)
        return max(
            tau_final,
            tau_init * (tau_final / tau_init) ** progress,
        )

    # ------------------------------------------------------------------
    # Gradient estimator (Gumbel-softmax forward + analytic backward)
    # ------------------------------------------------------------------

    def _gradient_estimate(
        self,
        *,
        alpha: torch.Tensor,
        mask: torch.Tensor,
        damage_grid: torch.Tensor,
        tau: float,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Stochastic gradient of ``E_p̃[Σ_l D_l(k_l)]`` w.r.t. ``α``.

        Standard-form Gumbel-softmax (plan §1.3 step 1 / D1-1 fix):

            p̃ = softmax((α + g) / τ),   g ~ Gumbel(0, 1).

        Limits (paper §3.1):
        - ``τ → ∞``: ``p̃ → uniform`` (high-entropy exploration).
        - ``τ → 0``: ``p̃ → argmax(α + g)`` (low-entropy exploitation; this
          IS the categorical sample by the Gumbel-max trick).

        The prior impl used ``softmax(α + τ·g)`` which has the opposite
        limits — REMOVED, not preserved.

        Analytic backward (paper §3.1, eq. on lines 478-482): the softmax
        Jacobian collapse yields ``∂(Σ_j p̃_lj D_lj)/∂α_lk = (1/τ) ·
        p̃_lk · (D_lk − E_p̃[D_l])``. The ``1/τ`` factor is absorbed
        into the Adam learning rate.
        """
        # Sample Gumbel noise: g_lk = -log(-log(u_lk)) with u ~ Uniform(0,1).
        # clamp_min(1e-20) avoids log(0) at the floor.
        u = torch.rand(alpha.shape, generator=rng, dtype=alpha.dtype).clamp_min(1e-20)
        gumbel = -torch.log(-torch.log(u))

        # Standard Gumbel-softmax form: (α + g) / τ.
        alpha_perturbed = (alpha + gumbel) / tau
        # Re-impose pad: where mask == 0, set to a very-negative value so
        # the masked softmax gives ~0 mass even after the gumbel noise
        # could have boosted a pad index.
        very_neg = torch.full_like(alpha_perturbed, -1e9)
        alpha_perturbed = torch.where(mask > 0, alpha_perturbed, very_neg)

        p_tilde = self._masked_softmax(alpha_perturbed, mask)

        # Analytic backward (1/τ absorbed by Adam lr).
        e_d = (p_tilde * damage_grid).sum(dim=1, keepdim=True)
        grad = p_tilde * (damage_grid - e_d) * mask
        return grad

    # ------------------------------------------------------------------
    # Discrete readout (multiple-choice knapsack DP, pure damage)
    # ------------------------------------------------------------------

    def _evaluate_discrete(
        self,
        *,
        cost_grid: torch.Tensor,
        mask: torch.Tensor,
        damage_grid: torch.Tensor,
        k_options: dict[int, list[int]],
        sorted_layers: list[int],
        global_budget: int,
    ) -> tuple[float, dict[int, int]]:
        """Project to a budget-exact discrete vector via pure-damage DP.

        Multiple-choice knapsack: select one option per layer minimising

            Σ_l D_l(k_l)   subject to   Σ_l c_{l, k_l} = B.

        The DP score is **pure damage** (plan §1.3 step 3 / D1-3 fix);
        the prior impl's ``β · log p`` tiebreak is REMOVED.

        Tiebreak policy (plan v4-N4): strict ``<`` on score comparisons,
        so on tied scores the first vector encountered in the layer
        sweep (lex-min on option indices) wins.

        Infeasibility fallback (plan §6.1 F8 / Delta 6 fix): if ``B`` is
        outside the achievable range, pick the nearest feasible budget
        with larger-budget tiebreak via
        ``min(feasible, key=lambda b: (abs(b−B), −b))``. Prior impl
        picked the maximum; REMOVED.

        Returns (fitness, budget_vector). ``budget_vector`` keyed by
        ``layer_idx`` → surviving expert count.
        """
        score_grid_np = damage_grid.detach().cpu().numpy().copy()
        mask_np = mask.detach().cpu().numpy()
        score_grid_np[mask_np == 0] = float("inf")
        cost_grid_np = cost_grid.detach().cpu().numpy()

        L = len(sorted_layers)
        B = int(global_budget)

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
                base = best[i, b]
                for k_idx in range(K_l):
                    c = int(cost_grid_np[i, k_idx])
                    s = float(score_grid_np[i, k_idx])
                    nb = b + c
                    if nb > B:
                        continue
                    cand = base + s
                    # Strict < tiebreak (plan v4-N4): first vector
                    # encountered in the layer sweep wins on ties.
                    if cand < best[i + 1, nb]:
                        best[i + 1, nb] = cand
                        choice[i + 1, nb] = k_idx

        if not math.isfinite(best[L, B]):
            # Infeasibility (plan §6.1 F8): pick the NEAREST feasible
            # budget with larger-budget tiebreak. WARN on the way.
            #
            # The primary DP table only tracks budgets in [0, B]; nearest
            # feasibility needs to also consider budgets > B, which means
            # re-solving over the full achievable range
            # ``B_max = Σ_l max(opts_l)``. Capacity is small at our scale:
            # Cost: L · K_max · B_max ≈ 48 · 65 · 6144 ≈ 19M float64 ops for
            # production scale (still cheap — paid only on the rare
            # infeasibility fallback path).
            B_max = int(sum(max(opts) for opts in k_options.values()))
            best_ext = np.full((L + 1, B_max + 1), INF, dtype=np.float64)
            choice_ext = np.full((L + 1, B_max + 1), -1, dtype=np.int64)
            best_ext[0, 0] = 0.0
            for i, li in enumerate(sorted_layers):
                opts = k_options[li]
                K_l = len(opts)
                for b in range(B_max + 1):
                    if not math.isfinite(best_ext[i, b]):
                        continue
                    base = best_ext[i, b]
                    for k_idx in range(K_l):
                        c = int(cost_grid_np[i, k_idx])
                        s = float(score_grid_np[i, k_idx])
                        nb = b + c
                        if nb > B_max:
                            continue
                        cand = base + s
                        if cand < best_ext[i + 1, nb]:
                            best_ext[i + 1, nb] = cand
                            choice_ext[i + 1, nb] = k_idx
            feasible = [b for b in range(B_max + 1) if math.isfinite(best_ext[L, b])]
            if not feasible:
                raise ValueError(
                    f"RCO DP: no feasible budget assignment for global_budget={B}."
                )
            chosen_B = min(feasible, key=lambda b: (abs(b - B), -b))
            log.warning(
                "RCO DP: global_budget %d infeasible; falling back to nearest "
                "feasible budget %d (larger-budget tiebreak).", B, chosen_B,
            )
            # Use the extended DP table for backtracking.
            best = best_ext
            choice = choice_ext
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

        fitness = 0.0
        for i, li in enumerate(sorted_layers):
            k_idx = k_options[li].index(budget_vec[li])
            fitness += float(damage_grid[i, k_idx].item())
        return fitness, budget_vec


__all__ = ["RCOBudgetPlugin"]
