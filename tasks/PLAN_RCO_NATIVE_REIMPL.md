# Plan — RCO Native Re-Implementation (clean-room, third path) — **v4**

**Status**: Planning. Pre-implementation. v4 folds the 6 round-3 deltas — two serious (v4-N1 fabricated Q8 quote, v4-N2 mathematically infeasible F15 construction) plus 4 smaller. Every paper quote in v4 verified via `grep -F` against `pdftotext`-extracted `paper.txt` (3768 lines from arxiv.org/pdf/2605.00649). Awaiting plan-reviewer round 4 per [[paper-fidelity-review-loop]].
**Branch**: `feat/plugin_11_rco_native_reimpl` (cut from `main@d7bfa22`).
**Parent plans**: `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 (R3 — RCO); `tasks/PLAN_PLUGIN_11_s1_rco.md` (the earlier clean-room plan, already implemented and merged at `269e64d`).
**Paper anchor**: arxiv 2605.00649 — *Model Compression with Exact Budget Constraints via Riemannian Manifolds* (IST-DASLab, May 2026). §3 Algorithm 1.
**Upstream code**: https://github.com/IST-DASLab/RCO — re-verified 2026-05-28 via GitHub API: `{"license": null, "default_branch": "main", "updated_at": "2026-05-16T..."}`. NO LICENSE file. (See §5 Pattern H row for raw response.)

---

## 0. Why this plan exists (the "third path")

The product brief enumerates three options for ingesting RCO:

1. **Vendor verbatim from upstream.** Blocked: upstream repo has no LICENSE file. Verbatim copy of unlicensed code into our repo is not legally clean. Verified `"license": null` via the GitHub API on 2026-05-27 and re-verified 2026-05-28.
2. **Wait for upstream + author consent.** A `feat/plugin_11_rco_revendor` branch exists with three small Stage-2 hoists but never received the consented vendor — the user has decided not to wait for legal clarity from a third party. Author consent without a repository-level LICENSE is still legally ambiguous because consent does not bind future readers of the original repo nor establish a redistribution licence for our forks.
3. **Clean-room re-implementation in moe_compress's plugin architecture.** ← The chosen path.

The repo *already contains* a clean-room implementation at `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` (898 LoC, merged to main as `269e64d`, with 12 tests at `max_quality/tests/test_stage1_plugin_rco_budget.py` — 325 lines). It was written from paper prose, has 8 documented `D-*` deviations, and is gated default-OFF behind `stage1.rco_budget.enabled`.

**Crucial**: the merge commit `269e64d` body explicitly flags **two known algorithm bugs** in the existing impl that v1 of this plan missed:

> *NOTE: clean-room has 2 algorithmic bugs (reversed cosine annealing, non-standard Gumbel-softmax) that will be fixed by the upcoming re-vendor from upstream RCO. The DP readout hides them on easy cases; plugin interface is re-vendor-ready […].*

Since the re-vendor path (option 2) is now abandoned, this re-impl plan **owns** fixing both bugs. They are folded into §1 (paper-fidelity algorithm reference) and §6.1 (new paper-fidelity tests F11-F14, plus F15 added in v3 and arithmetic-corrected in v4 to pin the pure-damage-DP-vs-logit-tiebreak fidelity).

This plan's job is therefore:

- **Re-derive RCO from paper prose** at full algorithmic fidelity (§1) — overriding the two flagged bugs in the existing impl;
- **Audit pattern adoption** (§5) — Patterns B / C / E from `[[architectural-patterns]]`;
- **Replace the existing file in-place** (§7) — same plugin id, same contract on the OFF path;
- **Surface open questions** (§8) where the upstream paper is ambiguous.

The plan is reviewed *before* code lands per [[paper-fidelity-review-loop]] step 2.

---

## 1. Algorithm reference (paper-only, clean-room)

This section is the canonical native description of RCO, derived from arxiv 2605.00649 prose + `SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 + the paper-summary returned by an authenticated query to the upstream README on 2026-05-28. NO upstream `.py` code is referenced. Where the paper is under-specified, the open question is logged in §8.

### 1.1 Problem statement

Given:
- `L` groups (in our setting: MoE-bearing decoder layers; for Qwen3.6-35B-A3B, `L = 48`).
- For each group `l ∈ {1, ..., L}`, a finite set of `K_l` discrete options, each with a positive integer cost `c_lk` (in our setting: a candidate surviving-expert count for that layer; with 256 experts per layer and floor `N // 2 = 128`, `K_max = 256 − 128 + 1 = 129`).
- A global integer budget `B > 0` (in our setting: `decomposition.global_expert_budget`; at 30% reduction with 256 experts × 48 layers ≈ 12,288 → `B ≈ 8600`).
- A non-decomposable objective `J(k_1, ..., k_L)` mapping a per-group option vector to a real-valued damage to minimise (in our setting: a sum of per-layer damage costs `D_l(k_l)`, which is decomposable — see Open Question Q1).

Find a budget-exact discrete assignment

```
k_l ∈ {1, ..., K_l} ∀ l,    Σ_l c_{l, k_l} = B,    minimising J(k_1, ..., k_L).
```

### 1.2 Soft relaxation onto a Riemannian manifold (paper §3)

Introduce per-group logits `α_l ∈ ℝ^{K_l}` and soft probabilities `p_lk = softmax(α_l)_k`. The *expected* cost is

```
C(α) = Σ_l Σ_k p_lk · c_lk = Σ_l ⟨p_l, c_l⟩.
```

The manifold of budget-feasible logits is

```
M = { α ∈ ℝ^{Σ K_l} : C(α) = B }.
```

The paper shows (Algorithm 1 + §3.1) that `M` admits three primitives in closed form:

#### Primitive 1: closed-form constraint normal

```
n_lk = ∂C / ∂α_lk = p_lk · (c_lk − ⟨p_l, c_l⟩).
```

Derivation: applying the standard softmax Jacobian `∂p_lj/∂α_lk = p_lj (δ_jk − p_lk)` to `Σ_j p_lj c_lj` collapses to the formula above.

#### Primitive 2: tangent projection

For any Euclidean gradient `g`, the projection that removes the constraint-normal component is

```
g_tangent = g − (⟨g, n⟩ / ⟨n, n⟩) · n
```

where the inner products are *global* (treat the full `Σ K_l`-dim stack as one vector) because the budget constraint is global. This step is standard Gram-Schmidt.

#### Primitive 3: retraction along the cost direction

After a tangent step, the new `α` may have drifted off `M`. The retraction parametrises a 1-D curve `α(t) = α − t · c_grid` (where `c_grid` is the cost tensor) and seeks `t` such that `C(α(t)) = B`. Because

```
d/dt C(α(t)) = − Σ_l Var_{p_l}(c_l) ≤ 0,
```

the residual `f(t) = C(α(t)) − B` is monotonically non-increasing in `t` (strictly decreasing on any group where the cost has variance under `p_l`). A 1-D bisection therefore converges in `O(log(1/ε))` steps:

1. **Bracket-doubling — explicit sign branch** (D7-2 fix). At `t = 0` compute the residual `f(0) = C(α) − B`.
   - If `f(0) > 0`: the current α over-budgets; we need to **increase** `t` (which moves probability mass toward cheaper options) so the bracket is `[0, +∞)`. Double `t_hi ← max(1, 2 · t_hi)` until `f(t_hi) ≤ 0`.
   - If `f(0) < 0`: under-budget; bracket is `(−∞, 0]`. Double `t_lo ← min(−1, 2 · t_lo)` until `f(t_lo) ≥ 0`.
   - If `|f(0)| ≤ tol`: return immediately.
2. **Bisection.** Halve the bracket until `|f(t)| ≤ tol`.

Termination: 32 doublings × 60 bisections is plenty for any realistic (`c`, `α`) scale (`2^31 ≈ 2.1e9` upper bound on the doubling range, comfortably above any plausible `B`).

#### Primitive 4: vector transport — Adam moment-transport policy

Adam carries TWO buffers: first moment `m` (gradient EMA) and second moment `v` (squared-gradient EMA). The paper's manifold story is unambiguous for `m` — it lives in the tangent space at `α` and must be re-projected after retraction:

```
m_new = m − (⟨m, n_new⟩ / ⟨n_new, n_new⟩) · n_new.
```

For `v`, the paper is **silent**. The literature on Riemannian Adam (Becigneul & Ganea 2019, "Riemannian Adaptive Optimization Methods") typically transports both `m` and `v` via the same vector-transport map, but several follow-ups (e.g. RsgdMomentum-style) transport `m` only and treat `v` as a scalar-like magnitude estimator that does not need to live in any specific tangent space.

**Decision for the re-impl**: **transport `m` only**; leave `v` untransported (D4-1 / Delta 4 option a). Rationale:
- `v` is element-wise squared and used only as an adaptive learning-rate scaler `1 / (√v̂ + ε)`. Its sign and tangent-vs-normal decomposition carry no operational meaning.
- The two flagged bugs in the existing impl are *not* about `v` — the existing code already does only-`m` transport. Keeping this preserves byte-identity on the (rare) cases where the algorithm currently converges correctly.
- A new deviation tag `D-adam-no-v-transport` is added in §1.6 to mark this as a paper-deviation (paper does not prescribe; we pick the lighter convention).
- The interaction with `v̂` ↔ retracted `m̂` is documented (D7-3 / Delta 14): in iteration `t`, the `step = −lr · m̂ / (√v̂ + ε)` is computed BEFORE the retraction; the retraction then projects `α + step` back onto `M`; finally `m` is transported to the new tangent plane and `v` is left as-is. So `v̂` "lags" `m̂` by one transport step. The lag is one iteration and gets absorbed into Adam's EMA dynamics; not a stability hazard at our scale.

Pinned by test F13 (`test_adam_v_buf_transport_policy`).

### 1.3 Discrete fitness signal (Gumbel-softmax + DP knapsack)

The paper §3.2 evaluates fitness on a *discrete* sample because the objective `J` is typically not differentiable in `α`. The flow is:

1. **Gumbel perturbation — standard form** (D1-1 / Delta 1 fix). Sample i.i.d. Gumbel noise `g_lk = −log(−log(u_lk))` with `u ~ Uniform(0, 1)`. Form perturbed logits `α' = (α + g) / τ`. **Not** `α + τ · g` (which is what the existing impl uses — see §7 migration). The standard Gumbel-softmax form is:

   ```
   p̃ = softmax((α + g) / τ).
   ```

   Limits:
   - `τ → ∞`: `(α + g)/τ → 0`, so `p̃ → uniform` (high-entropy exploration). ✓
   - `τ → 0`: `(α + g)/τ → ±∞`, the argmax dominates and `p̃ → argmax(α + g)` (low-entropy exploitation, with Gumbel noise breaking ties — this IS the categorical sample by the Gumbel-max trick). ✓

   The existing impl's `softmax(α + τ · g)` has the opposite limits (τ → ∞ → argmax of pure noise `g`, τ → 0 → softmax(α), neither of which matches paper intent).

2. **Cosine-anneal direction — explore → exploit** (D1-2 / Delta 2 fix). Pin the schedule as `τ_init → τ_final` with `τ_init > τ_final` (default 5.0 → 0.5). The cleaned-up cosine formula is

   ```
   τ_t = τ_final + 0.5 · (τ_init − τ_final) · (1 + cos(π · t / T))
   ```

   where `t` ranges over `0, 1, ..., T−1` (with `T = n_iterations`). At `t = 0`: `cos(0) = 1` ⇒ `τ_0 = τ_init`. At `t = T − 1` (or `t = T` in the limit): `cos(π) = −1` ⇒ `τ_T = τ_final`. So the schedule starts hot (uniform p̃, explore) and ends cool (argmax-like p̃, exploit).

   The existing impl at `rco_budget.py:404-407` reverses this — it computes `cos(π · (1.0 − progress))` which evaluates to `cos(π) = −1` at `t=0` (giving `τ_0 = τ_final`) and `cos(0) = +1` at `t=T-1` (giving `τ_T = τ_init`). That is the exploit-first / explore-last order, exactly backward. **The re-impl uses the canonical form above.** Pinned by F12.

3. **DP score is pure damage** (D1-3 / Delta 3 fix, choice (a)). The multiple-choice knapsack solves

   ```
   minimise Σ_l D_l(k_l)  subject to  Σ_l c_{l, k_l} = B.
   ```

   The existing impl at `rco_budget.py:822-825` mixes `score_grid = damage_grid − β · log p` with `β = 1e-3` as a soft tiebreak. **The re-impl removes this term**: the DP is over pure damage. Rationale:
   - The paper §3.2 describes the discrete projection as a budget-exact damage minimisation; the `β · log p` term is not in the paper.
   - At small `τ_final = 0.5`, `log p` of the argmax option is already ~0 and of the others is large negative, so β=1e-3 affects only deep-tie cases. The tiebreaking value is marginal and obscures the spec.
   - Removing it makes RCO degenerate cleanly to Plugin #8 (`damage_curve_dp`) when fitness comes from the damage_curve slot — a desirable property for the S1_DP-vs-S1_RCO ablation.

   Documented as a fix-up in §7 (behaviour change on the ON path); pinned by F15 on a hand-graded tied-damage instance where the β tiebreak would otherwise pick a different vector.

   **DP tiebreak policy (v4-N4 pin)**: the DP uses **strict `<` on score comparisons**, so on ties the first vector encountered along the layer sweep (lexicographic by layer 0 option, then layer 1 option, …) wins. Equivalently: the table entry is replaced only when a strictly-smaller damage is found. This becomes load-bearing the moment two feasible vectors have equal pure-damage sum; absent any other rule the layer-sweep order picks lex-min on the option indices. F15's construction has no ties (damage gap = 0.001), so the rule does not affect F15's reference value — but it MUST be specified, both for reviewer auditability and so a future DP re-write doesn't silently swap to `≤` and produce a different answer on tied cases.

4. **Soft objective for backward.** Compute `J_soft = Σ_l ⟨p̃_l, D_l⟩` and differentiate analytically: the softmax Jacobian collapse yields `∂J_soft / ∂α_lk = (1/τ) · p̃_lk · (D_lk − ⟨p̃_l, D_l⟩)`. The `1/τ` factor is absorbed into the Adam learning rate (so the implementation drops the explicit `1/τ` from the analytic gradient and lets Adam adapt).

### 1.4 Outer loop (Algorithm 1)

```
init α from REAP saliency (paper §3.3) OR GRAPE budgets (our D-init-grape)
init Adam buffers m = 0, v = 0
for t = 0..T−1:
    # Explore → exploit anneal (D1-2 fix):
    τ_t = τ_final + 0.5 · (τ_init − τ_final) · (1 + cos(π · t / T))

    # Standard-form Gumbel-softmax (D1-1 fix):
    sample g ~ Gumbel(0, 1) i.i.d. on the (L, K_max) grid
    p̃ = softmax((α + g) / τ_t)

    # Analytic gradient on the soft objective (1/τ absorbed by Adam lr):
    grad = p̃ · (D − ⟨p̃, D⟩)              # per-group Jacobian collapse

    # Tangent projection (Primitive 2):
    n   = constraint_normal(α)             # p · (c − ⟨p, c⟩) at the un-perturbed p = softmax(α); see Q8
    grad_tangent = grad − (⟨grad, n⟩ / ⟨n, n⟩) · n

    # Adam update (in tangent space):
    m = β1 · m + (1 − β1) · grad_tangent
    v = β2 · v + (1 − β2) · (grad_tangent ⊙ grad_tangent)
    m̂ = m / (1 − β1^(t+1))
    v̂ = v / (1 − β2^(t+1))
    step = −lr · m̂ / (√v̂ + ε)
    α   = α + step

    # Retract (Primitive 3) — explicit sign branch:
    α = retract(α, c_grid, B)

    # Vector transport (Primitive 4) — m only:
    n_new = constraint_normal(α)
    m     = m − (⟨m, n_new⟩ / ⟨n_new, n_new⟩) · n_new
    # v left untransported (D-adam-no-v-transport)
return discrete_argmax_then_DP(α, c_grid, B, damage_grid_only)   # pure-damage DP (D1-3 fix); strict `<` tiebreak (v4-N4)
```

**Hyperparameters exposed by the paper**: `n_iterations` (default 500-2000 depending on problem size), `learning_rate` (default 0.01-0.1), `gumbel_tau_init` (default 5.0), `gumbel_tau_final` (default 0.5), Adam (β1=0.9, β2=0.999, ε=1e-8), `seed`. The paper notes (abstract): *"avoids constraint-specific hyperparameters"* — the constraint primitives are parameter-free.

### 1.5 Cost & complexity at production scale

Production scale per `max_quality/configs/qwen36_35b_a3b_30pct.yaml`:
- `L = 48` MoE-bearing decoder layers
- `K_max = 256 − 128 + 1 = 129` per-layer options (full surviving-expert grid)
- `B ≈ 8600` (30% reduction over 256 × 48 ≈ 12,288 experts)

Per iteration:
- Forward `softmax + Gumbel sample`: `O(Σ K_l) = O(L · K_max) ≈ 6.2k` float64 ops
- Backward (analytic Jacobian collapse): same, `≈ 6.2k`
- Retraction (Primitive 3): `O(log(1/ε)) · O(Σ K_l) ≈ 60 · 6.2k ≈ 370k`
- Vector transport (Primitive 4): one inner product + one scalar-vector subtraction, `≈ 6.2k`

Over `T = 500` outer iterations: `≈ 500 · (4 · 6.2k + 370k) ≈ 200 M` float64 ops in tight numpy/torch kernels.

DP knapsack (once at end): `O(L · K_max · B) = 48 · 129 · 8600 ≈ 53 M` ops, executed as a Python triple-loop — at ~1 M ops/sec in pure-Python branch-heavy code, **≈ 50 seconds wall-clock**.

**Total RCO run**: ~50 s for the DP + ~1-2 s for the outer loop (vectorised torch on CPU) = **~1 minute end-to-end**. This is the *RCO-internal* time; the spec table at `SC_STAGE12_COMPREHENSIVE_PLAN.md:475` quotes "+1 RCO run (~85 min)" because that includes the calibration teacher forward that produces the damage curve, plus row-rendering overhead, not just RCO's internal solve.

Implication: the v1 plan's "sub-second on CPU at our scale" claim was **wrong** (it used B ≤ 256 instead of B ≈ 8600). At B ≈ 8600, the DP knapsack dominates and ~50 s is meaningful enough to motivate inner-loop vectorisation — see Q5 re-justified.

### 1.6 Deviation tag list (final, post-deltas)

Carried forward from the existing impl:
- `D-clean-room` — re-derived from paper prose (no verbatim upstream code)
- `D-init-grape` — α initialised from GRAPE budgets (Q2 ablation pending)
- `D-fitness-mse` — output-space MSE as the fitness signal (Q3)
- `D-synthetic-curve` — fallback synthetic linear damage curve when `per_layer_damage_curve` absent
- `D-floor-projection` — per-layer floor baked into the option grid (manifold's intrinsic geometry)
- `D-ragged-K` — per-layer K_l varies; padded with very-negative logits + a 0/1 mask
- `D-bisection-budget` — Primitive 3 implemented as bracket-doubling + bisection (paper says only "1-D root-find")
- `D-disabled-default` — gated default-OFF behind `stage1.rco_budget.enabled`

**NEW** tags introduced by v2:
- `D-adam-no-v-transport` (Delta 4 / D2-3) — only the first Adam moment `m` is vector-transported after retraction; the second moment `v` is left untouched. Paper is silent on `v`; we pick the lighter convention.

**NOT introduced** (because the re-impl removes the non-standard behaviour rather than documenting it):
- ~~`D-gumbel-perturb-form`~~ — re-impl uses standard `softmax((α+g)/τ)`; non-standard form is REMOVED, not documented.
- ~~`D-cosine-anneal-reversed`~~ — re-impl uses canonical explore→exploit; reversed schedule is FIXED, not documented.
- ~~`D-dp-soft-logit-tiebreak`~~ — re-impl removes the `β · log p` term; not documented as a deviation.
- ~~`D-fluctuation-budget-nearest`~~ — re-impl picks nearest feasible budget (Delta 6 / F8 fix); the "max feasible" behaviour is FIXED, not preserved.

Total final deviation count: **9** (8 carried + 1 new).

---

## 2. Slot mapping in moe_compress

| Slot dimension | Decision |
|---|---|
| **Pipeline stage** | Stage 1 (budget allocation). Spec `SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 places R3 (RCO) at "Stage 1 budget allocation". |
| **Phase / manifest position** | Phase G — strictly AFTER Phase F (`grape_merge`). The 10-tuple manifest order is the FROZEN contract; verified by `max_quality/tests/test_stage1_orchestrator.py:119-131` (`test_plugin_manifest_order`). |
| **Existing plugin id** | `rco_budget` at manifest index 9. Same id retained — see §7 (replace, not augment). |
| **Orchestrator hook** | `stage1/orchestrator.py` STEP 10b (already present at lines 604-612). Gated on `plugin.is_enabled(config)`. |
| **Artifact** | `artifacts_dir / "stage1_rco_budgets.json"` — already present at orchestrator lines 727-741. |
| **Default state** | OFF. The `S1_RCO` ablation row enables it; every other row stays byte-identical. |
| **Downstream consumers** | None inside moe_compress (verified by `grep -rn per_layer_target_experts_rco max_quality/src`). The refined budget is a **new** ctx slot (`per_layer_target_experts_rco`) sitting alongside GRAPE's existing `per_layer_target_experts`. Stage 2 reads whichever slot the row recipe names — the redirection is a row-config concern, not a plugin concern. |

**Decision: no new manifest position is needed.** The existing slot is correct (between GRAPE and the end of Stage 1; opt-in via the same gate). The implementation will *replace* the file at `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` in-place rather than introduce a `rco_budget_v2.py`.

---

## 3. Plugin contract

The contract is FROZEN against the existing plugin so byte-identical regression tests still pass on the default OFF path. Note the **paper** docstring is updated to reflect the new deviation list (§1.6):

```python
class RCOBudgetPlugin:
    name: str = "rco_budget"
    paper: str = (
        "RCO: IST-DASLab arxiv:2605.00649 §3 Algorithm 1. "
        "Upstream code at github.com/IST-DASLab/RCO ships without a LICENSE file "
        "(re-verified 2026-05-28 GitHub API: license=null); this is a clean-room "
        "re-implementation from the paper's prose. "
        "Deviations: D-clean-room, D-init-grape, D-fitness-mse, D-synthetic-curve, "
        "D-floor-projection, D-ragged-K, D-bisection-budget, D-disabled-default, "
        "D-adam-no-v-transport. "
        "Algorithm details follow paper §3 with: standard Gumbel-softmax "
        "softmax((α+g)/τ) (NOT α+τ·g); cosine τ-anneal τ_init→τ_final "
        "(explore→exploit); pure-damage DP knapsack (no β·log p tiebreak); "
        "Adam first-moment-only vector transport. "
        "See module docstring for full per-deviation derivations."
    )
    config_key: str = "stage1.rco_budget"
    reads: tuple[str, ...] = (
        "per_layer_target_experts",     # GRAPE budgets (init)
        "per_layer_redundancy",          # GRAPE R̃^l (synthetic curve fallback)
        "per_layer_targets",             # per-layer expert counts
        "decomposition",                 # BudgetDecomposition (global_expert_budget)
        "config",                        # the run config
    )
    writes: tuple[str, ...] = (
        "per_layer_target_experts_rco",  # refined budgets
        "rco_metadata",                  # init/final fitness, budget vectors, iters
    )
    provides: tuple[str, ...] = ()       # no accumulators
    def is_enabled(self, config: dict) -> bool: ...        # gates on config["stage1"]["rco_budget"]["enabled"]
    def run(self, ctx: PipelineContext) -> None: ...
    def contribute_artifact(self, ctx: PipelineContext) -> dict: ...
```

**Optional read (no KeyError if absent)**: `per_layer_damage_curve` — `dict[int, dict[int, float]]` of `D_l(k)` produced by Plugin #8 (`damage_curve_dp`). When present, RCO consumes it (the principled path); when absent, RCO falls back to the synthetic linear curve from GRAPE's `per_layer_redundancy` (D-synthetic-curve). Interaction with the new `fitness_signal` knob is specified in §5 Pattern E.

**No phase-hook methods.** RCO is a single-shot transform in `run()`; no `on_pre_assign`, `on_post_merge`, etc.

---

## 4. Data flow

### 4.1 Config → ctx inputs

```yaml
stage1:
  rco_budget:
    enabled: false           # MUST be true to fire (default OFF)
    n_iterations: 500        # outer loop iters
    learning_rate: 0.1       # Adam lr (in tangent space)
    gumbel_tau_init: 5.0     # cosine anneal start (explore)
    gumbel_tau_final: 0.5    # cosine anneal end (exploit); REQUIRED τ_init > τ_final
    init_peak_logit: 2.0     # GRAPE-option peak height
    floor_divisor: 2         # floor_l = per_layer_count_l // floor_divisor
    seed: 0                  # RNG for Gumbel samples
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-8
    fitness_signal: auto     # "auto" | "synthetic" | "damage_curve"   ← NEW (Pattern E)
```

### 4.2 Required ctx reads (raise KeyError if missing)

| Slot | Type | Source plugin |
|---|---|---|
| `per_layer_target_experts` | `dict[str, int]` | `grape_merge` (Phase F) |
| `per_layer_redundancy` | `dict[str, float]` | `grape_merge` (Phase F) |
| `per_layer_targets` | `dict[int, int]` | upstream of Stage 1 |
| `decomposition` | `BudgetDecomposition` | `budget.solver` |
| `config` | `dict` | orchestrator |

### 4.3 Optional ctx read

| Slot | Type | Source plugin |
|---|---|---|
| `per_layer_damage_curve` | `dict[int, dict[int, float]]` (layer_idx → {surviving_k → damage}) | `damage_curve_dp` (Phase E.5, Plugin #8). Optional; falls back to synthetic curve when `fitness_signal="auto"` AND slot absent. |

### 4.4 ctx writes

| Slot | Type | Consumer |
|---|---|---|
| `per_layer_target_experts_rco` | `dict[str, int]` | Stage 2 (when the row recipe selects RCO over GRAPE) |
| `rco_metadata` | `dict` (with `format_version: 1`, see §4.5) | Operator logs + the `stage1_rco_budgets.json` artifact |

### 4.5 Artifact (Pattern B + Pattern K)

`artifacts_dir / "stage1_rco_budgets.json"` — written by `_write_artifacts` ONLY when the plugin is enabled. Schema (D4-1 / Delta 13 — `format_version` at the **top-level** dict, not nested inside `rco_metadata`):

```json
{
  "format_version": 1,
  "rco_budgets": {"0": 3, "1": 4, ...},
  "rco_metadata": {
    "init_fitness": float, "final_fitness": float,
    "init_budget_vector": {"0": 3, ...}, "final_budget_vector": {"0": 3, ...},
    "n_iterations": int,
    "achieved_budget": int, "requested_budget": int,
    "fitness_source": "damage_curve" | "synthetic",
    "tau_init_used": float, "tau_final_used": float,
    "fitness_signal_resolved": "synthetic" | "damage_curve"
  }
}
```

**Pattern K policy** (forward-only schema bumps): readers MUST tolerate unknown keys at the top level. New fields appended to either the top-level dict OR `rco_metadata` do not bump `format_version`. A `format_version` bump is required ONLY when an existing key changes shape or is removed.

---

## 5. Architectural pattern adoption (Patterns A-M)

Re-evaluating the existing plugin against `[[architectural-patterns]]`:

| Pattern | Verdict | Action in re-impl |
|---|---|---|
| **A — Cache-aware plugin skip** | Not applicable. RCO has no precomputed sidecar of its own. | None. |
| **B — Versioned sidecar payload** | **Adopt.** Top-level `format_version: 1` field on the artifact (NOT inside `rco_metadata`); pair with Pattern K for forward-only schema bump policy (see §4.5). | New: emit `format_version: 1`; tests pin both readers (C12). |
| **C — Config-validation-at-top-of-run()** | **Currently weak.** The existing impl reads config keys lazily (`rco_cfg.get(...)` with defaults). Hidden mis-keys (e.g. `learning_rates` vs `learning_rate`) silently fall through to defaults. | New: add a `_validate_config(cfg)` call as the FIRST statement of `run()` that rejects unknown keys + range-checks (n_iterations > 0, 0 < lr < 10, 0 < τ_final < τ_init, 0 ≤ β1, β2 < 1, ε > 0, floor_divisor ≥ 1, `fitness_signal ∈ {"auto", "synthetic", "damage_curve"}`). |
| **D — Pre-flight config override** | Not applicable. RCO does not need to mutate config for a downstream plugin. | None. |
| **E — Algorithm branch knob + default-byte-identical** | **Adopt.** New `fitness_signal` knob: see specification below the table. | New knob with default `"auto"` (preserves current behaviour); C11 test. |
| **F — on_post_merge hook + Position B invalidation** | Not applicable. RCO is a single-shot Stage-1 transform; no inter-layer caches to invalidate. | None. |
| **G — Marginal prior as `in_ctx_config` slot** | Not applicable. RCO writes a budget dict, not a prior. | None. |
| **H — Clean-room + license-check before vendoring** | **Already adopted.** This entire plan IS Pattern H. **Re-verification 2026-05-28**: GitHub API on `https://api.github.com/repos/IST-DASLab/RCO` returned `{"license": null, "default_branch": "main", "updated_at": "2026-05-16T..."}`. File docstring includes this raw response timestamp. | Re-affirm in docstring with 2026-05-28 timestamp + the literal `license: null` API response. |
| **I — Layer-disjoint key invariant** | RCO reads all layers' GRAPE budgets + writes all layers' RCO budgets in a single call. The invariant ("per-layer state read only at current layer's key") is satisfied trivially — there is no inter-layer state mutation. | None; document in module docstring. |
| **J — Plan-reviewer step** | This plan IS the plan-reviewer trigger. | Spawn plan-reviewer (per project workflow). |
| **K — Forward-only schema bumps** | Pairs with Pattern B above; explicit policy in §4.5. | None new beyond §4.5. |
| **L — Reviewer sub-agents in worktrees** | Workflow concern, not implementation. | None. |
| **M — Post-fix reviewer worktree freshness** | Workflow concern. | None. |

**Pattern E — `fitness_signal` knob spec** (D4-2 / Delta 13):

- **Validation point**: top-of-`run()`, inside `_validate_config(cfg)`, BEFORE any RCO work begins.
- **Interaction with `ctx.has("per_layer_damage_curve")`**:
  - `fitness_signal="auto"` (default) → use `per_layer_damage_curve` if present, else fall back to the synthetic curve from `per_layer_redundancy`. Logs which path it took at INFO. **Byte-identical to historical behaviour.**
  - `fitness_signal="synthetic"` → hard-skip damage_curve even if present; always use the synthetic curve. (Useful for the S0_GRAPE-like baseline reproducibility under the RCO flag.)
  - `fitness_signal="damage_curve"` → hard-require damage_curve; if `ctx.has("per_layer_damage_curve")` is false, raise `ValueError("rco_budget: fitness_signal=damage_curve but per_layer_damage_curve slot is absent. Set fitness_signal=auto or enable damage_curve_dp.")`. (Useful for the S1_RCO ablation where the damage curve must be authoritative.)
- **Default value**: `"auto"`.
- **Test**: C11 `test_fitness_signal_strict_mode_raises_when_curve_absent` covers the damage_curve-strict path; default-auto path covered by existing `test_run_consumes_damage_curve_when_present` + `test_run_synthetic_2layer_handcheck`.

**Plan-reviewer C-checks** (from `[[architectural-patterns]]`):

- **C-INV (Invalidation-target consumer audit)**: RCO does not invalidate any slot — it only adds a new slot. ✓ No action required.
- **C-PROD-CFG (Production-default state audit)**: Default `enabled: false`. Both the `run()` call and the `contribute_artifact` write are gated on `is_enabled`. Production-default path: zero work, zero artifact, identical state to pre-plugin-#11 main. ✓ — **BUT** see §7 migration: on the ON path, the re-impl introduces three behavioural changes (Gumbel form, anneal direction, β·log p removal, F8 nearest-vs-max).
- **C-DTYPE-OPS (Bf16 portability per-op audit)**: RCO operates in `torch.float64` throughout (`_BISECT_TOL = 1e-4` requires the precision; expert counts up to 256 × 48 layers = 12,288 entries fit trivially). No bf16 dispatch concerns. ✓

---

## 6. Test plan

### 6.1 Paper-fidelity tests (algorithm correctness)

These tests are content-level: they prove the implementation does what arxiv 2605.00649 §3 Algorithm 1 says.

| # | Test | What it verifies |
|---|---|---|
| F1 | `test_constraint_normal_closed_form` | For random α, mask, c, `_constraint_normal` equals the symbolic gradient `p · (c − E_p[c])` (within 1e-12 of a `torch.autograd` reference). |
| F2 | `test_tangent_projection_orthogonality` | `⟨_project_off_normal(g, n), n⟩ ≈ 0` (within 1e-10) for random `g`, `n`. |
| F3 | `test_retraction_budget_exactness` | After `_retract`, the soft budget `Σ p · c` equals `B` within `_BISECT_TOL = 1e-4` (across 50 random α, B combinations). |
| F4 | `test_retraction_monotonicity` | The residual `f(t)` is monotonically non-increasing in `t` on a fine grid (catches a sign-flip bug in the bracket-doubling phase). |
| F5 | `test_vector_transport_preserves_tangency` | After a step + retract, projecting `m` via `_project_off_normal` at the new α makes `⟨m_new, n_new⟩ ≈ 0`. |
| F6 | `test_gradient_estimate_jacobian_collapse` | The analytic Gumbel-softmax gradient equals `p̃ · (D − E_p̃[D])`, verified against `torch.autograd` on a small random instance. |
| F7 | `test_dp_knapsack_optimality` | The multiple-choice DP finds the optimum on a hand-graded **3-layer × 4-option** instance (= **64 brute-force enumerations**), cross-checked against `itertools.product`. |
| F8 | `test_dp_handles_infeasible_budget` | When B falls outside `[Σ floor_l, Σ N_l]`, the DP fails over to the **nearest feasible budget** with a WARNING log (no silent corruption). **Tie-break: prefer the larger budget.** The selector is `min(feasible, key=lambda b: (abs(b − B), −b))` — NOT the existing impl's `max(feasible, key=...)` which picks the maximum (D2-2 / Delta 6 fix). |
| F9 | `test_algorithm_converges_on_quadratic_proxy` | For a synthetic quadratic damage curve where the analytic minimum is known, RCO converges to within ε of the optimum in ≤200 iterations. |
| F10 | `test_seed_reproducibility` | Two runs with the same seed produce identical final budget vectors **within the same torch version** (catches any non-deterministic ops; cross-version drift NOT pinned because torch RNG implementation can shift between minor versions). |
| F11 | `test_gumbel_softmax_tau_limits` | **NEW (Delta 1 / D1-1).** At very large τ (e.g. 1e6) and fixed (α, g), the sampled `p̃ ≈ uniform(K)` within 1e-3 per element. At very small τ (e.g. 1e-3) and fixed (α, g), `p̃` converges to a one-hot at `argmax(α + g)`. Pins the standard-form Gumbel-softmax. |
| F12 | `test_cosine_anneal_endpoints` | **NEW (Delta 2 / D1-2).** Construct the cosine schedule with `(τ_init=5.0, τ_final=0.5, T=500)`. Assert `τ_t[0] == pytest.approx(τ_init, abs=1e-12)` and `τ_t[T−1] == pytest.approx(τ_final, abs=2e-2)` (the `T−1` endpoint is `cos(π · (T−1)/T) ≈ cos(π)` with a small offset; use a generous tol). Pins the explore→exploit direction. |
| F13 | `test_adam_v_buf_transport_policy` | **NEW (Delta 4 / D1-4).** After one outer step + retract, the first moment `m` has `⟨m, n_new⟩ ≈ 0` (transported); the second moment `v` does NOT (untransported). Pins `D-adam-no-v-transport`. |
| F14 | `test_alpha_stability_under_large_logits` | **NEW (Delta 12 / D6-5).** Feed α with magnitude ~1e6 into the masked softmax → no NaN/Inf in `p̃`, in `_constraint_normal`, in retraction. Standard plugin-audit stability sanity. |
| F15 | `test_dp_pure_damage_not_logit_tiebreak` | **NEW (v3-Δ3 / Delta 3 fidelity pin; v4-N2 construction).** Construct a 2-layer × 2-option instance that genuinely contrasts β=0 vs β=1e-3. **Setup**: L=2, K=2; `c = [[1, 2], [1, 2]]`; `B = 3`; `D = [[1.0, 1.0], [1.0, 1.001]]`; `α = [[10.0, 0.0], [0.0, 10.0]]`. **Hand-derived arithmetic** (worked into the test docstring): (a) **Feasibility at B = 3**: per-layer choice (k₀, k₁) sums `c[0][k₀] + c[1][k₁]`. (0,0): 1+1=2 — infeasible. (0,1): 1+2=3 — FEASIBLE. (1,0): 2+1=3 — FEASIBLE. (1,1): 2+2=4 — infeasible. Two feasible vectors. (b) **β=0 pure-damage DP**: damage_sum(0,1) = D[0][0] + D[1][1] = 1.0 + 1.001 = 2.001; damage_sum(1,0) = D[0][1] + D[1][0] = 1.0 + 1.0 = 2.0. Minimum is 2.0 → pure-damage DP picks **(1, 0)**. (c) **β=1e-3 reference**: `log softmax([10, 0]) ≈ [−4.54e-5, −10.0000454]`; `log softmax([0, 10]) ≈ [−10.0000454, −4.54e-5]`. Score = damage − β · log p. score(0,1) = 2.001 − 1e-3 · (−4.54e-5 + −4.54e-5) = 2.001 + 9.08e-8 ≈ 2.001000091. score(1,0) = 2.0 − 1e-3 · (−10.0000454 + −10.0000454) = 2.0 + 0.0200000908 ≈ 2.0200000908. Minimum is ≈2.001 → β=1e-3 picks **(0, 1)**. **Crucially β=0 and β=1e-3 disagree** because the damage gap (0.001) is comparable to β · |Δ log p| (1e-3 · 20 = 0.02). **Test assertions**: (i) Re-impl (which has no β knob — removed in v2-Δ3) returns the β=0 reference `(1, 0)`. (ii) Re-impl's output is NOT `(0, 1)` (the β=1e-3 hand-computed answer; computed off-line because the re-impl has no β knob to re-create the broken regime). Both reference vectors spelled out verbatim in the test docstring. **No DP-tiebreak ambiguity**: the construction has no tied scores, so v4-N4's tiebreak policy doesn't kick in here. Pins §1.3 step 3 against future re-introduction of the β tiebreak. |

### 6.2 Code-quality tests (config validation, edge cases)

Real existing test inventory in `max_quality/tests/test_stage1_plugin_rco_budget.py` (12 tests, 325 lines — verified by `grep -n "^def test_"` on 2026-05-28). The v1 plan's C-tests fabricated names that don't exist in the file; v2 reconciles to ACTUAL names:

| # | Test | Status | What it verifies |
|---|---|---|---|
| C1 | `test_plugin_protocol_attributes` | **EXISTS** (keep verbatim) | Class attrs match the contract. Will need a docstring touch if the `paper:` string changes (see §3). |
| C2 | `test_plugin_is_runtime_checkable_pipelineplugin` | **EXISTS** (keep verbatim) | `isinstance(plugin, PipelinePlugin)`. |
| C3 | `test_plugin_disabled_by_default` | **EXISTS** (keep verbatim) | Default-OFF semantics. |
| C4 | `test_plugin_enabled_when_flag_true` | **EXISTS** (keep verbatim) | Flag-on enables. |
| C5 | `test_run_rejects_missing_slot[5 params]` | **EXISTS** (keep verbatim) | KeyError contract over 5 slots. |
| C6 | `test_config_rejects_unknown_keys` | **NEW** (Pattern C) | `learning_rates=0.1` (typo) raises `ValueError` listing the unknown key. |
| C7 | `test_config_range_checks` | **NEW** (Pattern C) | Invalid ranges (`n_iterations=0`, `tau_init=-1`, `floor_divisor=0`, `tau_final >= tau_init`) all raise `ValueError`. |
| C8 | `test_run_synthetic_2layer_handcheck` | **EXISTS — RENAMED** from v1's fictional name; this is the existing hand-graded 2-layer regression test. Will need expected-value RE-VERIFICATION because the algorithm changes (Gumbel form, anneal direction, β removal) shift the DP output; the hand-check values get re-computed under the corrected algorithm. |
| C9 | `test_run_consumes_damage_curve_when_present` | **EXISTS — VERIFY** | Damage-curve path. Behavioural test; may need expected-value re-derivation under the corrected algorithm. |
| C10 | `test_run_respects_floor` | **EXISTS — KEEP** (this is the test v1 mis-named as "pathological damage curve" — it's the per-layer floor invariant). | RCO never drops below `per_layer_count_l // floor_divisor`. |
| C11 | `test_fitness_signal_strict_mode_raises_when_curve_absent` | **NEW** (Pattern E) | When `fitness_signal: "damage_curve"` is set and the slot is absent, raise `ValueError` (no silent fallback). |
| C12 | `test_artifact_includes_format_version` | **NEW** (Pattern B) | The `stage1_rco_budgets.json` includes `format_version: 1` at the top level (not nested in `rco_metadata`). |
| C13 | `test_contribute_artifact_when_disabled` | **EXISTS — KEEP** | Default-OFF artifact empty. |
| C14 | `test_contribute_artifact_when_enabled` | **EXISTS — KEEP / EXTEND** | Post-run artifact correctness; extend the assertion to include the new top-level `format_version` key. |
| C15 | `test_plugin_registered_in_manifest` | **EXISTS — KEEP** | Manifest order (this is in the rco test file; the cross-plugin manifest order is in `test_stage1_orchestrator.py:119-131`). |
| C16 | `test_run_sums_to_global_budget` | **EXISTS — KEEP** (single test, NOT parametrised in current file) | Verifies RCO returns budget-exact assignments. |

**Re-tallied test count** (v3 arithmetic, with F15 added per v3-Δ3):
- Existing code-quality (kept verbatim or with expected-value re-verification): **12** — C1-C5, C8-C10, C13-C16
- New code-quality: **4** — C6, C7, C11, C12 (Patterns B/C/E)
- Paper-fidelity (all NEW; none of F1-F15 exist in the current test file — verified by `grep -n "^def test_"` on 2026-05-28): **15** — F1-F15 (was 14 in v2; F15 added in v3-Δ3 to pin the pure-damage DP vs β·log-p tiebreak fidelity)
- Regression: **3** — R1-R3 (R1 is the Stage-1 byte-equality golden snapshot; R2 and R3 are repurposed existing C-tests with re-derived expected values, listed separately in §6.3 because they fill the regression role even though the rows reference C-test names)

**Contiguous F + C + R tally: 15 + 16 + 3 = 34** (was 33 in v2 pre-F15). This is the authoritative total.

### 6.3 Regression / byte-identity tests

| # | Test | What it verifies |
|---|---|---|
| R1 | Full Stage-1 default-OFF byte-equality vs `main@d7bfa22` (**Stage-1 golden snapshot, single fixture**, NOT the all-stage 20-snapshot suite — D7-4 / Delta 14) | Run `stage1_orchestrator` on the canonical Qwen3 fixture with `stage1.rco_budget.enabled: false` — every file in `artifacts_dir` from Stage 1 must be byte-identical to the pre-re-impl artifacts. The Stage-1 portion of the existing 20-snapshot plugin-audit harness covers this. |
| R2 | `test_run_synthetic_2layer_handcheck` + `test_run_consumes_damage_curve_when_present` | Must pass — but with expected values RE-DERIVED under the corrected algorithm (Gumbel form, anneal direction, β removal). These are NOT byte-identity guards on the ON path. |
| R3 | `test_run_sums_to_global_budget` | Existing single test; budget-exactness is invariant under algorithm changes, so this passes as-is. |

**Pass criteria**: F1-F15 + C1-C16 + R1-R3 green. Total: 15 + 16 + 3 = **34 tests**.

---

## 7. Migration / deprecation

**Decision: REPLACE the existing Plugin #11 in-place, not augment.**

Rationale:
- Same algorithm (RCO §3 Algorithm 1) — adding a second plugin would split readers between the two and accrete cruft.
- Same contract (name, reads, writes, gate key) — no downstream consumer changes.
- The improvements (Patterns B, C, E) are forward-compatible deltas, not algorithm pivots.
- The two flagged bugs from the `269e64d` commit body (reversed cosine, non-standard Gumbel) are FIXED in-place rather than left for a "future re-vendor".
- The β · log p DP tiebreak and the "max feasible" infeasibility fallback are also FIXED in-place (Deltas 3 + 6).
- The existing branch `feat/plugin_11_rco_revendor` (the consent-vendor path) is **abandoned** by this plan, but BEFORE closing it the implementer MUST audit for any unique unmerged content (D7-5 / Delta 14):
  ```bash
  git fetch origin feat/plugin_11_rco_revendor
  git log --oneline main..origin/feat/plugin_11_rco_revendor
  git diff --stat main...origin/feat/plugin_11_rco_revendor
  ```
  v1 of this plan asserted "three small Stage-2 hoists, already backported to main". This must be VERIFIED, not trusted; surface any unique commits before closing.

**Behaviour changes on the ON path** (i.e., when `stage1.rco_budget.enabled: true`):

1. **Gumbel form** — was `softmax(α + τ · g)`, now `softmax((α + g) / τ)`. Limits flip from "argmax of pure noise vs softmax(α)" to "uniform vs argmax(α + g)". Affects every iteration's gradient signal direction at all τ.
2. **Cosine anneal direction** — was exploit→explore (start cold, end hot), now explore→exploit (start hot, end cold). Affects every iteration's τ value.
3. **DP score** — was `damage − 1e-3 · log p`, now `damage` only. Affects deep-tie cases at the final DP readout.
4. **Infeasible-budget fallback** — was `max(feasible, ...)`, now `min(feasible, key=(|b−B|, −b))` (nearest, larger-tiebreak). Affects only the rare infeasibility path; warning log preserved.

Combined effect: the final `per_layer_target_experts_rco` vector returned for any given config is expected to **change** between v1-existing and v2-reimpl. The expected-value assertions in C8 + C9 must be re-derived. **There is no byte-identity guarantee on the ON path** (only on the default-OFF path via R1).

**Migration steps**:

1. Re-write `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` on this branch (`feat/plugin_11_rco_native_reimpl`). The file is REPLACED, not appended. Existing API (`name`, `reads`, `writes`, `is_enabled`, `run`, `contribute_artifact`) is preserved bit-for-bit on the OFF path.
2. Extend `max_quality/tests/test_stage1_plugin_rco_budget.py` with F1-F15 + C6, C7, C11, C12; re-derive expected values in C8 + C9 under the corrected algorithm; keep all other existing tests as-is.
3. No changes to `max_quality/src/moe_compress/stage1/orchestrator.py` (the gated call sites at STEP 10b and `_write_artifacts` already exist).
4. No changes to `max_quality/src/moe_compress/stage1/plugins/__init__.py` (manifest entry already present; verified by `test_plugin_manifest_order` at `test_stage1_orchestrator.py:119-131`).
5. Update the module docstring of `rco_budget.py` to:
   - cite this plan (`tasks/PLAN_RCO_NATIVE_REIMPL.md`);
   - re-anchor the clean-room note with the 2026-05-28 re-verification timestamp + the literal GitHub API `license: null` response;
   - re-list the **nine** `D-*` deviations per §1.6 (8 carried + 1 new `D-adam-no-v-transport`);
   - explicitly call out the three FIXES vs the previous body (Gumbel form, cosine direction, β·log p removal, F8 nearest-vs-max) so future readers don't re-introduce them.
6. **Audit and close `feat/plugin_11_rco_revendor`**:
   ```bash
   git fetch origin
   git log --oneline main..origin/feat/plugin_11_rco_revendor
   ```
   If the diff is empty (all commits already in main), `git push origin --delete feat/plugin_11_rco_revendor`. If non-empty, surface the unique content to the user before closing.

**Deprecation:** none. The old plugin was a draft; the new one is the canonical native implementation.

---

## 8. Open questions

These need a plan-reviewer or user decision before the implementer can proceed cleanly.

### Q1 — `J` decomposability

The paper (§3.1) introduces RCO for *non-decomposable* objectives `J(k_1, ..., k_L)` (e.g. end-to-end task loss). In our setting the fitness `Σ_l D_l(k_l)` is fully decomposable per layer — meaning RCO degenerates to "minimise a sum of per-layer scalar damage curves under a global budget", which a plain integer DP solves exactly in `O(L · K_max · B)` with NO gradient descent, NO manifold, NO Gumbel.

**Question**: Are we using RCO because (a) we anticipate a future non-decomposable fitness (the L1/vLLM end-to-end-loss path from `L1_FOR_SC_PLAN.md`), or (b) because the spec mandates it for the `S1_RCO` ablation row regardless of decomposability?

**Existing impl's answer**: (b) — the docstring cites spec §6.1 row `S1_RCO` and notes worst-case "RCO ≈ GRAPE" for the synthetic-curve fallback. The plan-reviewer should re-affirm.

### Q2 — Initialisation: REAP saliency vs GRAPE budgets

Paper §3.3 initialises α from REAP saliency scores; our implementation initialises from GRAPE per-layer budgets via `D-init-grape`.

**Concrete reason for the deviation**: REAP saliency is per-*expert*, not per-budget-*option*. To use REAP, we'd need a per-(layer, surviving_k) saliency, which the upstream paper *constructs* from a forward pass over candidate budgets. Our pipeline does not run that pass — GRAPE's output is the cheapest reasonable initialisation.

**Status (v3 correction)**: the v2 wording claimed the GRAPE-vs-uniform-init basin question was "folded into F11/F12". That was misleading: F11 pins the Gumbel-softmax τ limits and F12 pins the cosine anneal endpoints — neither addresses the basin-of-attraction question for the initialisation choice. The ablation is **deferred**, not folded:

- Q2 is a basin-of-attraction ablation, not a paper-fidelity property. Adding an F-test for it would mis-categorise the test (F-tests pin paper-spec behaviour; basin questions are empirical / post-hoc).
- The existing `D-init-grape` deviation already documents the divergence from paper §3.3.
- The ablation can be run post-hoc against the new impl if the user requests it (config-only change: swap `init_peak_logit` for a uniform-init code path; no algorithm changes needed).
- **No new F-test is added.** The "drop `D-init-grape` from the deviation list" decision is gated on running that ablation, which is out of scope for this plan.

### Q3 — Fitness signal: output-space MSE vs end-to-end task loss

Paper §4.2 uses end-to-end task loss (validated via vLLM rollout) as the fitness signal. Our implementation uses output-space MSE (`D-fitness-mse`).

**Recommendation**: defer the `RCOFitnessProvider` hook until L1 lands. YAGNI bites.

### Q4 — Bisection tolerance scale (re-computed for B ≈ 8600)

`_BISECT_TOL = 1e-4` at production scale `B ≈ 8600`:
- Relative tolerance: `1e-4 / 8600 ≈ 1.16e-8` — well below float64 epsilon margin for any plausible accumulation.
- Bracket-doubling: 32 doublings → `2^31 ≈ 2.15e9`, far above B.
- DP readout precision: the DP picks integer expert counts; 1e-4 residual on the soft `C(α)` rounds cleanly to the nearest integer everywhere `Σ Var_p(c) > 0.5` (true at non-degenerate α).

**Conclusion** (D5-1 / Delta 9 fix): **keep `_BISECT_TOL = 1e-4` fixed**, but with the arithmetic above documented in the constant's comment. v1's "B ≤ 512 in practice" was incorrect; the actual scale is `B ≈ 8600` and the tolerance is still comfortably adequate.

### Q5 — Vectorise the DP knapsack inner loop (re-justified, D5-2 / Delta 10)

At production scale, the DP knapsack is **~50 seconds** wall-clock (53 M Python branch-heavy ops × ~1 µs each). v1's rationale "keep explicit indexing to preserve byte-identity" was wrong because:
- On the OFF path, no DP runs at all → no byte-identity concern.
- On the ON path, the algorithm is changing (Gumbel form, β removal); there is no prior artifact to byte-match.

**Updated rationale considerations**:
- **Readability**: the Python triple-loop (`for i, for b, for k_idx`) maps cleanly to the paper's recurrence `best[i+1, b+c] = min(best[i+1, b+c], best[i, b] + s_lk)`. A fully vectorised form using `numpy.minimum.outer` + masked assignments obscures this.
- **Numerical stability**: float64 throughout; vectorisation does not introduce accumulation hazards.
- **Performance**: vectorising the inner `k_idx` loop alone (keeping `i` and `b` explicit) gives ~10×-20× speedup → ~5 s instead of ~50 s; vectorising both `k_idx` AND `b` is harder because of the `b + c` shift and adds complexity for marginal gain.

**Recommendation** (commit to one): **vectorise the inner `k_idx` loop only**. Concrete pattern: for each `(i, b)` pair where `best[i, b]` is finite, compute `cand = best[i, b] + score_grid_np[i, :]` and `nb = b + cost_grid_np[i, :]`, mask `nb > B`, then for each valid k advance `best[i+1, nb]` via a vectorised `np.minimum` against the current row. Net wall-clock target: < 10 s at B ≈ 8600. Surface to plan-reviewer for sign-off if a different cut-line is preferred.

### Q6 — `D-init-grape` defensive clamp behaviour: warn vs ValueError

Current code clamps GRAPE budgets that fall outside the option grid and emits a warning:

```python
g_l_clamped = max(opts[0], min(opts[-1], g_l))
if g_l_clamped != g_l:
    log.warning("RCO: layer %d: GRAPE budget %d clamped...", ...)
```

**Verification step before promoting to ValueError** (D5-3 / Delta 11): GRAPE's floor handling lives in a different config key namespace — `max_quality/src/moe_compress/stage1/plugins/grape_merge.py:304` reads `s1.get("grape_floor_divisor", 2)` under `stage1_grape`, while RCO reads `stage1.rco_budget.floor_divisor`. Both default to `2` (same `N // 2` floor), so under default config the clamp can fire only if there is an actual upstream invariant violation. **HOWEVER**: a row recipe MAY legitimately set `stage1_grape.grape_floor_divisor: 3` (warned-only, opt-in) AND keep `stage1.rco_budget.floor_divisor: 2`; in that case GRAPE could legitimately allocate a layer below RCO's floor, the clamp would legitimately fire, and a hard `ValueError` would crash an otherwise-valid run.

**Conclusion**: keep as warning, BUT extend the warning message to include both floor-divisor values (`stage1_grape.grape_floor_divisor=X, stage1.rco_budget.floor_divisor=Y`) so the operator immediately sees the divergent-config root cause. Add an integration test that exercises mismatched divisors to pin the warning-only behaviour. **Do NOT promote to ValueError.**

### Q7 — Recommendation strength for the re-impl: do it or not?

**The existing implementation has TWO known bugs** (commit `269e64d` body) PLUS two more identified by plan-reviewer round 1 (β·log p mix-in; max-vs-nearest infeasibility fallback). The existing tests don't catch any of them because the DP readout hides the τ-anneal/Gumbel-form bugs on easy cases and the infeasibility path is exercised only at edge budgets.

This is no longer a "quality-of-life" question — it is a **correctness fix**. The re-impl is required (Posture A).

- **Posture A (full re-impl, REQUIRED)**: rewrite `rco_budget.py` end-to-end on this branch, apply all Pattern-B/C/E upgrades, add F1-F15 paper-fidelity tests, fix all four algorithm bugs. Estimated effort: 1 implementation session + 1 paper-fidelity review + 1 code-quality review = ~5-7 hours of agent time.
- ~~Posture B (incremental)~~: insufficient; cannot patch the four bugs without rewriting most of the iteration loop and DP body.
- ~~Posture C (no change)~~: ruled out — the existing impl is known-buggy per the merge commit body.

### Q8 — Constraint normal at un-perturbed `p` vs Gumbel-perturbed `p̃`

In §1.4 the pseudocode computes `n = constraint_normal(α)` at the deterministic, un-perturbed softmax `p = softmax(α)` — NOT at the Gumbel-perturbed `p̃ = softmax((α + g) / τ)`. This was dangling-referenced in v2 as "Q-grad-α-vs-α'" but never resolved.

**Question**: should the constraint normal be computed at `p` (deterministic) or `p̃` (perturbed)?

**Existing impl**: un-perturbed `p`. At `rco_budget.py:420`, `self._constraint_normal(alpha, cost_grid, mask)` is called with the un-perturbed `alpha` (NOT a pre-computed `p`); the softmax is computed *internally* at line 611 as `p = self._masked_softmax(alpha, mask)` — no τ scaling, no Gumbel noise. (v4-N6 corrects v3's line-citation paraphrase.)

**Paper §2 (The Budget Manifold)**, Eq. (1) + Propositions 1 and 2 (paper.txt lines 251-308). The paper defines `pi = softmax(αi)` (paper.txt line 265, `grep -F`-verified) and constructs the manifold as

> M = {α ∈ R^{NK} : C(α) = B}  with  C(α) = Σᵢ wᵢ ⟨pᵢ, c⟩  and  pᵢ = softmax(αᵢ)

(Eq. 1, paper.txt lines 264-291). Proposition 2 then gives the closed-form normal `∇C(α)_{ik} = wᵢ pᵢₖ (cₖ − E_{pᵢ}[c])` (paper.txt line 303, `grep -F` hit on `"the gradient of C with respect to α"` and on `Proposition 2 (Normal vector)`). The whole §2 derivation is in `α`-space with `p(α) = softmax(α)` — no temperature, no Gumbel noise, no `α̂`. Verbatim `grep -F`-verified quote from paper.txt line 256 (Section 2 opener):

> *"Optimizing on a manifold requires three operations: tangent projection"*

— and §2 explicitly defines those three operations (tangent projection, retraction, vector transport) against the un-perturbed `p`.

**§3.1 contrast (NOT in tension)**. Paper §3.1 introduces a *separate* object: the perturbed logits `α̂ᵢₖ = (αᵢₖ + Gᵢₖ)/τ` for the forward-pass DP sampler, with `p̂ᵢ = softmax(α̂ᵢ)` as the **STE backward-gradient surrogate** for the soft objective `J_soft`. The paper makes the choice of `p̂` (perturbed) over un-perturbed `softmax(α)` explicit at paper.txt lines 478-482 (`grep -F`-verified):

> *"gradients flow through p̂ᵢₖ = softmax(α̂ᵢ)ₖ, the softmax of the same perturbed logits that produced z\*. This ensures the surrogate concentrates on the sampled mode, so the STE bias vanishes as τ → 0 and independent Gumbel samples yield independent Jacobians; the unperturbed softmax(αᵢ) would decouple the surrogate from the sampled assignment and suppress both effects."*

This `p̂` is used for the **backward pass of the STE on the loss gradient `∇L(α)`** — it is the surrogate that lets ∂L/∂α flow through the discrete arg-max. It is NOT the constraint normal.

**The two objects are distinct and the paper uses them in different places**:

| Object | Where | Computed at |
|---|---|---|
| Constraint normal `n = ∇C(α)` (forward manifold projection) | §2, Prop. 2 | Un-perturbed `p = softmax(α)` |
| STE backward gradient surrogate `p̂` (backward through DP arg-max) | §3.1, lines 478-482 | Perturbed `p̂ = softmax(α̂)`, `α̂ = (α + G)/τ` |

The re-impl mirrors this: `_constraint_normal` uses un-perturbed `α` (paper §2); the analytic backward gradient on `J_soft` in §1.3 step 4 uses `p̃` (paper §3.1, lines 478-482). There is no paper-fidelity tension between the two — they answer different questions about different terms.

**Recommendation**: keep un-perturbed `p` for the constraint normal. Pinned by F1 (`test_constraint_normal_closed_form` — F1's symbolic reference `p · (c − E_p[c])` already encodes this: F1 constructs a deterministic `p = softmax(α)` and checks `_constraint_normal` against autograd of `Σ p · c`, so any drift toward `p̃` in the impl would surface as an F1 failure). F6 (`test_gradient_estimate_jacobian_collapse`) covers the §3.1 contrast (`p̃`-based backward).

**No new test required**: F1 + F6 already cover both objects. Q8 documents that the §2-vs-§3.1 distinction is intentional and not a paper-fidelity bug.

---

## 9. Estimate

### 9.1 File list

| File | Action | Approx LoC |
|---|---|---|
| `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` | REPLACE (in-place rewrite) | ~950 (was 898) |
| `max_quality/tests/test_stage1_plugin_rco_budget.py` | EXTEND (add F1-F15, C6, C7, C11, C12; re-derive C8/C9 expected values) | ~820 (was 325) |
| `max_quality/src/moe_compress/stage1/orchestrator.py` | NO CHANGE | — |
| `max_quality/src/moe_compress/stage1/plugins/__init__.py` | NO CHANGE | — |
| `tasks/PLAN_RCO_NATIVE_REIMPL.md` | THIS FILE (v4 commit) | ~830 |

### 9.2 Test list (34 total)

- Paper-fidelity (F1-F15): 15 NEW (F15 added in v3-Δ3 to pin pure-damage DP vs β·log-p tiebreak)
- Code-quality: C1-C5 (5 existing), C6/C7 (2 NEW Pattern C), C8/C9 (2 existing, expected-values re-derived), C10 (1 existing), C11 (1 NEW Pattern E), C12 (1 NEW Pattern B), C13/C14 (2 existing; C14 extended for format_version), C15/C16 (2 existing) → **16**
- Regression: R1 (Stage-1 byte-equality), R2 (2 existing behavioural tests, re-derived), R3 (1 existing budget-sum test) → **3**

**Total: 34** (was 33 in v2; +1 from F15 in v3-Δ3).

### 9.3 Performance — production scale (L=48, K_max=129, B≈8600)

| Phase | Ops | Wall-clock (Python/torch CPU) |
|---|---|---|
| Outer loop (500 iters × ~25k ops/iter) | ~12.5 M | ~1-2 s (vectorised torch) |
| DP knapsack (current Python triple-loop) | ~53 M | **~50 s** |
| DP knapsack (after Q5 inner-loop vectorisation) | ~53 M | **~5-10 s** |

**Total RCO-internal time at production scale: ~10-60 s** depending on Q5 sign-off. Spec `SC_STAGE12_COMPREHENSIVE_PLAN.md:475`'s "~85 min" entry for the `S1_RCO` row includes the calibration teacher forward pass for the damage curve + row-rendering overhead, NOT just RCO's internal solve.

### 9.4 Coupling to existing plugins

| Plugin | Coupling | Risk |
|---|---|---|
| `grape_merge` (Phase F) | RCO reads `per_layer_target_experts` and `per_layer_redundancy` from GRAPE's writes. | Low — contract stable. |
| `damage_curve_dp` (Phase E.5, Plugin #8) | RCO optionally reads `per_layer_damage_curve` from S1_DP. | Low — `ctx.has()` guarded; `fitness_signal="auto"` preserves current auto-fallback behaviour. |
| `budget.solver.BudgetDecomposition` | RCO reads `global_expert_budget`. | None — public dataclass. |
| Stage 2 (any plugin) | RCO writes `per_layer_target_experts_rco`; Stage 2 reads whichever budget slot the row recipe names. | Zero — no Stage 2 plugin currently reads the `_rco` slot. |

### 9.5 Estimated effort (Posture A, full re-impl with bug fixes)

- Plan-reviewer cycle round 2: ~30 min.
- Implementer: 1 session, ~2.5 hours (rewrite + re-derive C8/C9 expected values + new tests + run gates).
- Paper-fidelity reviewer: 1 round (likely 0-2 fix iterations), ~45 min.
- Code-quality reviewer: 1 round (likely 0-1 fix iteration), ~45 min.
- Commit + merge to main (no PR per [[no-pr-language]]): ~5 min.

**Total: ~5-7 hours of agent time across the workflow.**

---

## 10. Out of scope

- Stage 2-6 changes.
- `grape_merge.py` or any other Stage 1 plugin behaviour.
- The L1/vLLM rollout substrate.
- Vendoring upstream `IST-DASLab/RCO` (the chosen "third path" rules this out).
- Adding a new manifest slot — the existing `rco_budget` position is correct.
- Running the actual `S1_RCO` ablation row (config bundle + GPU run; that is a separate work item).
- Pursuing upstream LICENSE re-engagement — if upstream lands a permissive license later, a follow-up plan could revisit verbatim vendor; this plan is independent of that.

---

## Appendix A — Cross-reference to existing artefacts

| Artefact | Path | Status |
|---|---|---|
| Existing clean-room impl | `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` (39 KB, 898 lines) | Merged to main at `269e64d`. **Has 4 known bugs** (see §1.3 + §7). To be REPLACED. |
| Existing tests | `max_quality/tests/test_stage1_plugin_rco_budget.py` (325 lines, 12 tests) | Merged. To be EXTENDED + C8/C9 expected values re-derived. |
| Existing plan | `tasks/PLAN_PLUGIN_11_s1_rco.md` | Done. This plan supersedes it for the re-impl effort. |
| Predecessor merge commit | `269e64d` — body explicitly flags 2 of the 4 known bugs | Read by §0 + §7. |
| Abandoned consent-vendor branch | `feat/plugin_11_rco_revendor` (3 commits per v1 claim — TO BE VERIFIED in §7 step 6) | Audit before closing. |
| Manifest-order test | `max_quality/tests/test_stage1_orchestrator.py:119-131` (`test_plugin_manifest_order`) | Pins the 10-tuple as the FROZEN contract. |
| Spec authority | `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 (R3 — RCO); row `S1_RCO` at line 475 | "~85 min" includes calib forward + row render, not just RCO internal solve. |
| Architecture patterns | `[[architectural-patterns]]` (memory: `project_architectural_patterns`) | Pattern catalogue; §5 enumerates the relevant ones. |
| Workflow | `[[paper-fidelity-review-loop]]`, `[[review-fix-loop-protocol]]` | This plan triggers step 2 (plan-reviewer). |

## Appendix B — Five-line algorithm summary

RCO casts discrete budget allocation as soft-relaxed optimisation on a Riemannian manifold `M = { α : Σ p · c = B }` in logit space where `p = softmax(α)`. Three closed-form primitives — constraint-normal `n = p · (c − E_p[c])`, Gram-Schmidt tangent projection, and bisection-along-cost retraction with explicit sign branch — wrap a standard Adam optimiser so every iterate is budget-exact by construction. A standard-form Gumbel-softmax `p̃ = softmax((α + g) / τ)` with explore→exploit cosine-annealed τ (τ_init → τ_final) samples discrete assignments for the fitness signal; the analytic backward differentiates the same softmax-Jacobian collapse used by `n`. Adam's first moment is vector-transported by re-projection after each retraction; the second moment is left untransported (`D-adam-no-v-transport`). A final pure-damage multiple-choice knapsack DP locks the soft logits to a budget-exact integer assignment with nearest-feasible-budget fallback on infeasibility, completing one RCO run in ~10-60 seconds on CPU at our production scale (L=48, K_max=129, B≈8600).

---

## Appendix C — v1→v2 delta change-log

The 14 deltas surfaced by plan-reviewer round 1, each mapped to the v2 section that folds it:

| Delta | Severity | v1 issue | v2 fold location | Status |
|---|---|---|---|---|
| D1 (D1-1) | Critical | v1 used `softmax(α + τ · g)` (non-standard); paper uses `softmax((α + g) / τ)` | §1.3 step 1 + §1.4 pseudocode + §1.6 (no new D-tag — fixed in-place) | FOLDED |
| D2 (D1-2) | Critical | v1 + existing impl run cosine anneal exploit→explore (reversed); should be explore→exploit | §1.3 step 2 + §1.4 + F12 | FOLDED |
| D3 (D1-3) | High | v1 ignored existing impl's `β · log p` DP tiebreak (deviates from paper) | §1.3 step 3 + §1.6 (β · log p REMOVED) + §7 behaviour change list | FOLDED — option (a), pure-damage DP |
| D4 (D1-4) | Medium | v1 ignored Adam `v_buf` transport policy | §1.2 Primitive 4 + §1.4 + §1.6 (`D-adam-no-v-transport`) + F13 | FOLDED — transport only `m` |
| D5 (D2-1) | High | v1 fabricated test names that don't exist in the file | §6.2 reconciled against real `grep -n "^def test_"` output | FOLDED |
| D6 (D2-2) | Medium | v1's F8 said "nearest feasible" but existing impl picks max feasible | §6.1 F8 + §7 behaviour change list (FIXED to nearest with larger-tiebreak) | FOLDED — option (a) |
| D7 (D2-3) | Medium | v1 said "re-list eight D-* PLUS new ones" but §8 surfaced no new tags | §1.6 final tag list (9 total: 8 carried + 1 new) | FOLDED |
| D8 (D5-1, D7-1) | High | v1 claimed "sub-second on CPU"; correct figure is ~50 s for DP at B≈8600 | §1.5 + §9.3 + Q5 re-justification | FOLDED |
| D9 (D5-1) | Medium | v1's Q4 used "B ≤ 512 in practice"; actual scale is ~8600 | §8 Q4 re-derived | FOLDED |
| D10 (D5-2) | Medium | v1's Q5 rationale "byte-identical fp accumulation" doesn't apply on ON path | §8 Q5 re-justified on readability + perf grounds; commits to inner-k vectorisation | FOLDED |
| D11 (D5-3) | Low | v1's Q6 wanted warn → ValueError without checking GRAPE-RCO floor-divisor coupling | §8 Q6 verified GRAPE uses different key namespace + identified the legit mismatch case → KEEP warning | FOLDED — kept as warning |
| D12 (D6-1 through D6-6) | Medium | v1 missing F11-F14 paper-fidelity tests | §6.1 F11/F12/F13/F14 added; F7 sizing pinned (3×4); F10 same-torch-version guard | FOLDED |
| D13 (D4-1, D4-2) | Low | v1's Pattern B/E specs underspecified | §4.5 (`format_version` at top level); §5 Pattern E concrete spec | FOLDED |
| D14 (D2-4, D3-2, D7-2 through D7-6) | Low | v1 polish items | §1.2 (sign branch + v_hat lag); §1.4 (cleaned cos formula); §6.3 R1 (Stage-1 only); §5 H (license: null timestamp); §7 (revendor audit); §2 (manifest test cite) | FOLDED |

**No deltas were RECONSIDERED or DEFERRED. All 14 folded.**

---

## Appendix D — v2→v3 delta change-log

Plan-reviewer round 2 surfaced 5 new findings — all of them inconsistencies that v2 itself introduced when folding round 1's 14 deltas. v3 resolves them:

| v3 Delta | Severity | v2 issue | v3 fold location | Status |
|---|---|---|---|---|
| v3-Δ1 | Medium | §6.2 said "New: 6" (should be 4 — C6, C7, C11, C12); §9.2 line 627 said "33"; §6.3 line 482 had a 35-vs-33 reconciliation paragraph that became unnecessary once the arithmetic was fixed. | §6.2 re-tally rewritten with explicit 12 + 4 + 15 + 3 = 34 arithmetic; §9.2 updated to 34; §6.3 line 482 reconciliation prose dropped per v3-Δ5. | FOLDED — recommendation (a) (recompute total) |
| v3-Δ2 | Medium | §1.4 line 184 cited a non-existent "Q-grad-α-vs-α'" question. The substantive choice (un-perturbed `p` vs Gumbel-perturbed `p̃` for the constraint normal) was un-anchored. | §1.4 line 184 inline citation rewritten to "see Q8"; new Q8 added to §8 anchored at **Paper §2 (The Budget Manifold), Eq. (1) + Prop. 1/2** (not §3.1 — §3.1 is the algorithm section, which uses perturbed `p̂` for a different purpose, the STE backward; v4-N1/N3/N5 corrects v3's section citation). Recommendation: keep un-perturbed `p`. F1 + F6 already pin both objects; no new F-test needed. See Q8 for the full §2-vs-§3.1 distinction. | FOLDED — paper §2 anchor (corrected from §3.1 in v4-N3) |
| v3-Δ3 | Medium | §1.3 step 3 line 163 promised "pinned by F-test on a hand-graded instance where β tiebreak would otherwise pick a different vector" but no such F-test existed in F1-F14. | F15 (`test_dp_pure_damage_not_logit_tiebreak`) added to §6.1 covering tied-damage instance with β=1e-3 vs β=0 disagreement. §1.3 step 3 line 163 amended to cite F15. §6.1, §6.3, §9.2 test counts updated 14 → 15 (F), 33 → 34 (total). | FOLDED — recommendation (a) (add F15) |
| v3-Δ4 | Low | §8 Q2 line 553 said the GRAPE-init basin ablation was "folded into F11/F12" but F11 (Gumbel τ limits) and F12 (cosine endpoints) don't cover the basin question. Misleading "folded" claim. | Q2 rewritten: ablation explicitly **deferred** (not folded). Rationale: it's a basin-of-attraction empirical question, not a paper-fidelity property. No new F-test added; ablation runnable post-hoc by user request. | FOLDED — recommendation (defer) |
| v3-Δ5 | Nitpick | §0 line 25 mentioned only "F11 and F12" when v2 actually added F11-F14. §6.3 line 482 had reconciliation prose that v3-Δ1 makes unnecessary. | §0 line 25 expanded to "F11-F14, plus F15 in v3-Δ3". §6.3 line 482 reconciliation prose dropped. | FOLDED |

**No deltas were RECONSIDERED or DEFERRED. All 5 folded.** Round-3 reviewer should re-spawn against this v3 plan.

---

## Appendix E — v3→v4 delta change-log

Plan-reviewer round 3 surfaced 6 findings, including two serious bugs (one fabricated paper quote in Q8, one mathematically infeasible F15 construction). v4 folds them. Every paper quote in this v4 was verified via `grep -F` against `pdftotext`-extracted `paper.txt` from `arxiv.org/pdf/2605.00649` (3768 lines, downloaded fresh 2026-05-28).

| v4 Delta | Severity | v3 issue | v4 fold location | Status |
|---|---|---|---|---|
| v4-N1 | HIGH | Q8 contained a verbatim quote (*"the geometric constraint defining the manifold itself operates on the clean softmax probability distribution, ensuring the constraint surface has a well-defined mathematical structure independent of stochastic perturbations."*) that the round-3 reviewer empirically demonstrated does NOT appear in the paper (downloaded PDF + `pdftotext` + `grep -F`: no hit). This was a hallucination from a WebFetch "verification" that wasn't a verification. | Q8 rewritten end-to-end. Fabricated quote struck. Replaced with `grep -F`-verified paraphrase + citation of Paper §2 Eq. (1) + Props. 1/2 (paper.txt lines 251-308). Added a clarifying paragraph distinguishing the **constraint normal** (§2, un-perturbed `p` — what `_constraint_normal` uses) from the **STE backward gradient surrogate** (§3.1, perturbed `p̂` — what the loss-gradient backward pass uses), with the `grep -F`-verified §3.1 quote (paper.txt lines 478-482): *"gradients flow through p̂ᵢₖ = softmax(α̂ᵢ)ₖ, the softmax of the same perturbed logits that produced z\*. This ensures the surrogate concentrates on the sampled mode, so the STE bias vanishes as τ → 0 and independent Gumbel samples yield independent Jacobians; the unperturbed softmax(αᵢ) would decouple the surrogate from the sampled assignment and suppress both effects."* The two objects are NOT in tension — they answer different questions about different terms. | FOLDED — both quotes `grep -F`-verified |
| v4-N2 | CRITICAL | F15's v3 construction was mathematically infeasible: with `c = [[1, 2], [1, 2]]` and `B = 2`, only `(0, 0)` is feasible (1+1=2); `(0,1)`, `(1,0)`, `(1,1)` all violate the budget (sum 3, 3, 4). The v3 claim that β=1e-3 picks `[0, 1]` vs β=0 picks `[0, 0]` is impossible — the DP has only one feasible vector. | F15 entirely rewritten with hand-derived arithmetic. New setup: `c = [[1, 2], [1, 2]]`, `B = 3`, `D = [[1.0, 1.0], [1.0, 1.001]]`, `α = [[10, 0], [0, 10]]`. Feasibility worked out: (0,1) and (1,0) are the two feasible vectors at B=3. β=0 hand-derivation: damage_sum(0,1)=2.001, damage_sum(1,0)=2.0 → β=0 picks (1,0). β=1e-3 hand-derivation (computed off-line because the re-impl has no β knob — removed in v2-Δ3): score(0,1)≈2.001, score(1,0)≈2.020 → β=1e-3 picks (0,1). β=0 and β=1e-3 genuinely disagree because the damage gap (0.001) is comparable to β·|Δ log p| (1e-3 · 20 = 0.02). Both reference vectors verbatim in F15's docstring. Test asserts re-impl returns (1, 0) and is NOT (0, 1). | FOLDED — arithmetic hand-derived |
| v4-N3 | MEDIUM | Q8 cited "Paper §3.1" for the manifold definition. The actual paper has §2 = "The Budget Manifold" (paper.txt line 251; manifold definition lines 263-304); §3.1 = "Algorithm" (paper.txt line 447). | Q8 corrected to cite **Paper §2 (The Budget Manifold), Eq. (1) + Prop. 1/2**. Appendix D v3-Δ2 row also corrected (was §3.1, now §2). Section structure verified by `grep -nE "^[0-9]+\..|^[A-Z][a-z].* Manifold$|^Algorithm$"` on paper.txt. | FOLDED |
| v4-N4 | LOW | §1.3 step 3 and §1.4 pseudocode did not specify the DP tiebreak policy. F15 explicitly states it has no ties, but the rule must still be pinned for general auditability and to prevent a future DP rewrite from silently swapping `<` for `≤` on tied cases. | §1.3 step 3 amended with a tiebreak-policy paragraph: strict `<` on score comparisons, so the first vector encountered along the layer sweep wins on ties (equivalently: lex-min on option indices). §1.4 pseudocode comment line `discrete_argmax_then_DP(...)` extended to "strict `<` tiebreak (v4-N4)". F15 docstring notes its construction has no ties so the rule is informational for that test. | FOLDED |
| v4-N5 | MEDIUM | Appendix D v3-Δ2 row repeated the v4-N1 fabricated quote verbatim. | Fabricated quote struck from Appendix D v3-Δ2 row. Row replaced with a reference to the corrected Q8 (Paper §2 anchor) and the v4-N1/N3 lineage. | FOLDED |
| v4-N6 | NITPICK | Q8 in v3 said "the call site reads `p = torch.softmax(alpha, dim=-1)` immediately before `constraint_normal(p, c)`". This does not match the code: `rco_budget.py:420` calls `self._constraint_normal(alpha, cost_grid, mask)` with `alpha`, NOT a pre-computed `p`; the softmax happens *inside* at line 611 via `_masked_softmax(alpha, mask)`. | Q8's "existing impl" line rewritten to match the code: "At `rco_budget.py:420`, `self._constraint_normal(alpha, cost_grid, mask)` is called with the un-perturbed `alpha`; the softmax is computed internally at line 611 as `p = self._masked_softmax(alpha, mask)` — no τ scaling, no Gumbel noise." Both line numbers verified against the file in this branch. | FOLDED |

**Paper-quote verification protocol used for v4** (per round-3 reviewer's discipline):
```
mkdir -p /tmp/rco_paper && cd /tmp/rco_paper
curl -L https://arxiv.org/pdf/2605.00649 -o paper.pdf  # 2.08 MB
pdftotext paper.pdf paper.txt                          # 3768 lines
# For every quote used:
grep -Fn "<exact string>" paper.txt                    # must return a hit
# For multi-line quotes (wrapped by pdftotext):
tr '\n' ' ' < paper.txt | grep -oF "<joined string>"   # must return a hit
```

`grep -F` hits confirmed for both quotes used in v4 Q8:
- §2 quote (paper.txt line 256): `"Optimizing on a manifold requires three operations: tangent projection"`
- §3.1 quote (paper.txt lines 478-482, joined): `"gradients flow through p̂ᵢₖ = softmax(α̂ᵢ)ₖ, the softmax of the same perturbed logits that produced z*. This ensures the surrogate concentrates on the sampled mode, so the STE bias vanishes as τ → 0 and independent Gumbel samples yield independent Jacobians; the unperturbed softmax(αᵢ) would decouple the surrogate from the sampled assignment and suppress both effects."`

**No deltas were RECONSIDERED or DEFERRED. All 6 folded.** Loop should close on round-4 review.
