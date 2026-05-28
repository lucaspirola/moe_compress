# RCO Upstream Bit-by-Bit Alignment Audit

Repo state:
- Upstream: `IST-DASLab/RCO` cloned at `/tmp/upstream_rco_align/RCO` (depth=1).
- Ours:    `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` (≈1114 LoC).
- Branch:  `fix/rco-upstream-alignment` off `main` (e1426b4).

Definitions:
- **ALIGN** = our code changes to match upstream exactly.
- **KEEP-DEVIATION** = upstream's behavior cannot be matched because our pipeline's API surface requires a different shape; documented as a D-tag with justification.
- **OPEN-QUESTION** = uncertain whether the deviation is forced or optional; needs user input.

---

## Item 1 — Per-group weights `w_i`

**Upstream**: `src/manifold.py:32-45` (`budget_normal`), 48-65 (`project_gradient`), 68-118 (`retraction`), 121-145 (`vector_transport`). Each accepts `weights: Optional[torch.Tensor]`; the budget formula is `C(α) = Σ_i w_i · Σ_k p_ik · c_k`. In `src/search/quant.py:329-332` the weights are `group_param_fracs` (per-group parameter-count fraction).

**Ours**: No `weights` parameter in `_constraint_normal`, `_project_off_normal`, `_retract`, `_budget_residual`. Implicit `w_i = 1`.

**Verdict**: **ALIGN**. Add optional `weights` parameter to all manifold primitives. For our MoE expert-pruning use, derive `w_i` as the per-layer **MoE parameter count** (sum of expert weights = `num_experts × hidden_size × moe_intermediate_size × 3` for gate+up+down). The orchestrator exposes the `model` slot in ctx; we add a `_layer_param_counts` helper and pass `weights` shaped `[L]` into the primitives.

If `model` is absent (unit-test path), pass `weights=None` and treat as ones (back-compat).

---

## Item 2 — Annealing schedule shape

**Upstream**: `src/search/prune.py:663` and `src/search/quant.py:655`:
```
tau = max(tau_min, tau_init * (tau_min / tau_init) ** progress)
```
Exponential, with `progress = step / max(n_steps - 1, 1)`. At step 0 → `tau_init`; at step `n_steps-1` → `tau_min`.

**Ours**: `_cosine_tau` uses `τ_t = τ_final + 0.5·(τ_init − τ_final)·(1 + cos(π·t/T))`.

**Verdict**: **ALIGN**. Rename `_cosine_tau` → `_anneal_tau`, replace formula with exponential, update F12 endpoints, remove `D-anneal-cosine` references in docstring (replace with no deviation note).

---

## Item 3 — Retraction parameterization sign

**Upstream**: `src/manifold.py:115-117`:
```
c = 0.5 * (lo + hi)
with torch.no_grad():
    alpha.add_(c * costs)
```
Positive `+c · costs`. Inside the bisection, when `E = C(mid) > target` → `hi = mid`; when `E < target` → `lo = mid`. Increasing the shift `mid` along `+costs` direction shifts probability mass toward more expensive options (because `softmax(α + s·c)` weights options with higher `c` more as `s` grows), so `C` is monotonically **non-decreasing** in `mid`.

Wait — re-checking: `softmax(α + s·c)` with positive `s` and positive cost `c` puts more mass on high-cost options ⇒ `C` increases as `s` increases. So `if E > target: hi = mid` decreases the upper bound (we want smaller `s`). That matches upstream.

**Ours**: parametrize `α(t) = α − t·c`, so `C` is monotonically **non-increasing** in `t`. `if f > 0: t_lo = t_mid` (advance lower bound). The math is symmetric.

**Verdict**: **ALIGN**. Flip our convention to `α(t) = α + t·c` to match upstream's positive-shift convention so the code reads identically. Update the monotonicity argument, sign branches, and bracket-doubling accordingly. Tests F3/F4 update the residual function passed in.

---

## Item 4 — Retraction tolerance

**Upstream**: `src/manifold.py:73` `retraction(..., tol=0.05)`; `:174` `retraction_per_layer(..., tol=1e-3)`.

**Ours**: `_BISECT_TOL = 1e-4`.

**Verdict**: **ALIGN**. Use `tol = 0.05` for the global-budget retraction (matching `retraction`). If per-layer retraction is ever wired in (we don't currently use it), use `tol = 1e-3`. Update F3 expected tolerance.

---

## Item 5 — Out-of-range fallback tiebreak direction

**Upstream**: `src/search/quant.py:411-419`:
```
best_j = budget
if dp[best_j] == NEG_INF:
    for delta in range(1, budget):
        for j in [budget - delta, budget + delta]:
            if 0 <= j <= budget and dp[j] > NEG_INF:
                best_j = j
                break
        if dp[best_j] > NEG_INF:
            break
```
Lower-side preferred on tied distance. Note: upstream caps at `budget` (does not search beyond), so larger-side is only the `+delta` slot when within `[0, budget]`.

**Ours**: `min(feasible, key=lambda b: (abs(b - B), -b))` — explicitly larger-side wins on ties. Our impl also extends the DP table to `B_max > B` to consider supra-budget options.

**Verdict**: **ALIGN**. Match upstream's lower-side-preferred tiebreak. Change to `min(feasible, key=lambda b: (abs(b - B), b))` so ties prefer the smaller budget. Update F8 test (B=5 → 4, B=7 → 6).

The extension of DP table to `B_max` (to consider supra-budget) is a separate concern. Upstream caps at `budget` and only considers `[budget - delta, budget + delta]` within `[0, budget]`, i.e. effectively `[0, budget]`. Keep our extension because our pipeline's `B = global_expert_budget` is an exact target the row recipe asks for; we WANT to consider larger budgets too — but tiebreak resolved smaller-side as upstream does.

---

## Item 6 — Initialization

**Upstream**: `src/search/quant.py:302-327` `init_alpha_to_bits`:
```
def expected_bits(beta):
    logits = -beta * bits
    probs = torch.softmax(logits, dim=0)
    return (probs * bits).sum().item()

for _ in range(100):
    mid = (lo + hi) / 2.0
    if expected_bits(mid) > target_bits:
        lo = mid
    else:
        hi = mid

beta = (lo + hi) / 2.0
init_logits = -beta * bits
alpha.copy_(init_logits.unsqueeze(0).expand(self.n_groups, -1)...)
```
Same per-row init `α_l = -β · c` for ALL rows; `β` bisected so `E[bits] = target`.

**Ours**: `D-init-grape`: initialise `α_lk = init_peak_logit` at the option-index matching GRAPE's budget; 0 elsewhere; pad columns = -1e9.

**Verdict**: **ALIGN**. Replace GRAPE-driven init with β-bisection init. Remove `D-init-grape` deviation tag.

GRAPE's budgets are ignored for init (matching upstream); they remain useful for synthetic-damage fallback (`grape_redundancy` → damage). The plan's `init_peak_logit` config knob becomes obsolete; remove it. Update F9 (convergence test) to confirm β-init also converges to the optimum.

---

## Item 7 — Constraint formula and DP scoring polarity

**Upstream**: `src/search/quant.py:362-430` `round_with_budget_dp` MAXIMIZES `dp[prev] + value` where `value = log(prob)`. Higher prob = optimizer-preferred.

**Ours**: We MINIMIZE `dp + damage`. Damage is a positive quantity to minimize.

**Verdict**: KEEP-DEVIATION (forced by our pipeline). Our pipeline's `damage_grid` is a true damage (loss-increase) value, not a log-probability. Upstream's polarity (maximize log-prob) is equivalent to minimizing negative log-prob. Both directions select the option with the lowest cost per the underlying signal; the difference is only in input form (continuous prob vs explicit damage scalar) and is forced by what Stage 1 actually computes.

D-tag this as `D-dp-damage-not-logp` (replaces the silent assumption that the signals are interchangeable).

---

## Item 8 — D-tag list cleanup

**`D-adam-no-v-transport`**: Upstream `manifold.py:121-145` `vector_transport` ALSO transports `m` only, leaving `v` untouched. NOT a deviation. **REMOVE.**

**`D-anneal-cosine`** (implicit in docstring): becomes obsolete after Item 2 fix. Removed by docstring rewrite.

**`D-init-grape`**: removed by Item 6 fix.

**`D-bisection-budget`**: KEEP — upstream's global retraction IS bisection, so this matches. Re-word to note it matches upstream.

**`D-fitness-mse`**: KEEP — upstream uses end-to-end KL/CE on calibration data with autograd through the model, we use precomputed damage curves. This is forced by our pipeline architecture (Stage 1 has no model-in-the-loop in the orchestrator path).

**`D-synthetic-curve`**: KEEP — upstream has no synthetic-fallback because they always run the model; we need one when Plugin S1_DP is disabled.

**`D-floor-projection`**: KEEP — option grids are floor-clamped; upstream's option grid is `[0, ..., bitwidths_max]` without a floor concept.

**`D-ragged-K`**: KEEP — per-layer K varies in our MoE-pruning setting; upstream's per-group K is uniform (fixed bitwidth set).

**`D-disabled-default`**: KEEP — plugin-architecture gating, orthogonal to upstream.

**`D-clean-room`**: KEEP — upstream is unlicensed; this is a clean-room re-implementation. Re-verify license status when bumping.

**NEW**: `D-dp-damage-not-logp` (Item 7).

---

## Item 9 — Other differences found

### 9.1 — Default `tau_init`

**Upstream**: `tau_init=1.0` default in `optimize` function (`prune.py:591`).
**Ours**: `gumbel_tau_init: 5.0` default.

**Verdict**: **ALIGN**. Set default `gumbel_tau_init = 1.0`. (Tests use explicit values.)

### 9.2 — Default `n_steps`

**Upstream**: `n_steps=200` for prune (`prune.py:591`); `n_steps=400` for quant (`quant.py:582`).
**Ours**: `n_iterations=500` default.

**Verdict**: **ALIGN** to prune's default (the pruning case is ours): `n_iterations = 200`.

### 9.3 — Default `lr`

**Upstream**: `lr=0.1` for prune (`prune.py:591`); `lr=0.05` for quant (`quant.py:583`).
**Ours**: `learning_rate: 0.1` — already matches prune.

**Verdict**: NO CHANGE.

### 9.4 — Gumbel noise sampling: clamp value

**Upstream prune.py:103-104**: `u = torch.rand_like(alpha).clamp(1e-20)` then `gumbel = -log(-log(u) + 1e-20)`. Note BOTH the `1e-20` clamp on `u` AND a `+1e-20` inside the inner log.
**Upstream quant.py:672-673**: `u = torch.rand_like(...).clamp(min=1e-20, max=1 - 1e-20)` then `gumbel = -log(-log(u) + 1e-20)`. ALSO has the inner `+1e-20`.

**Ours**: `u = torch.rand(...).clamp_min(1e-20)`; `gumbel = -log(-log(u))`. NO inner `+1e-20`.

**Verdict**: **ALIGN**. Add `+ 1e-20` to the inner log to match upstream. Use `clamp(min=1e-20, max=1 - 1e-20)` to also bound `u` from above (matches quant.py — slightly tighter than prune's clamp-only-min).

### 9.5 — Initial bracket size for retraction

**Upstream prune+quant**: bracket starts at `lo, hi = -1.0, 1.0`.
**Ours**: bracket starts at the same `-1.0, 1.0` (post-flip in Item 3 fix).

**Verdict**: NO CHANGE after Item 3.

### 9.6 — Bracket doubling iterations

**Upstream**: 40 iterations of bracket-doubling (`prune.py:96`, `manifold.py:96`).
**Ours**: `_BRACKET_MAX_DOUBLINGS = 32`.

**Verdict**: **ALIGN**. Change to 40.

### 9.7 — Bisection iterations (`max_iter`)

**Upstream**: default `max_iter=60` in `retraction`.
**Ours**: `_BISECT_MAX_ITERS = 60`. Match.

**Verdict**: NO CHANGE.

### 9.8 — Variance-reduction option (antithetic Gumbel sampling)

**Upstream**: `optimize` in `prune.py:594` accepts `antithetic=False`; if True, halves base samples and pairs each draw with its negation.

**Ours**: No antithetic option.

**Verdict**: KEEP-DEVIATION. Our gradient estimator is a single Gumbel sample per iteration (no `n_gumbel_samples` loop) because we have no model-in-the-loop noise; the gradient is purely analytic over a small `[L, K_max]` tensor. Antithetic is unnecessary in our setting (no monte-carlo noise to reduce). Document as `D-no-antithetic` (or just skip — it's an upstream optional knob with no operational effect when `n_gumbel_samples=1` and analytic gradient).

Re-evaluate: this isn't even a deviation since we don't have `n_gumbel_samples > 1`. Just document in docstring that we use a single-sample analytic gradient.

### 9.9 — Optimizer choice (Adam vs AdamW)

**Upstream**: `torch.optim.Adam` (`prune.py:634`, `quant.py:616`).
**Ours**: manual Adam (no decoupled weight decay).

**Verdict**: NO CHANGE. Both are Adam.

### 9.10 — Adam epsilon

**Upstream**: PyTorch default `eps=1e-8`.
**Ours**: `adam_eps: 1e-8`. Match.

**Verdict**: NO CHANGE.

### 9.11 — `1e-12` epsilon in `project_gradient` / `vector_transport`

**Upstream**: `nf @ nf + 1e-12` (`manifold.py:62`); `nsq = ... + 1e-12` (`:138`).
**Ours**: `clamp_min(1e-30)` in `_project_off_normal`.

**Verdict**: **ALIGN**. Use `1e-12` for the projection denominator (matches upstream).

### 9.12 — `1e-30` epsilon in `_masked_softmax` normalizer

**Upstream**: no masked softmax (uniform option grids); ordinary `torch.softmax`.

**Ours**: `_masked_softmax` divides by `sum.clamp_min(1e-30)`.

**Verdict**: KEEP — needed for our ragged-K mask. Same epsilon, no deviation flag (it's a numerical-stability detail that doesn't differ from "no upstream analog").

### 9.13 — Random number generation: device + seed

**Upstream**: `torch.rand_like(alpha)` (uses default RNG on alpha's device).
**Ours**: `torch.rand(shape, generator=rng, dtype=alpha.dtype)` with a `torch.Generator()` (CPU by default).

**Verdict**: KEEP. Our implementation runs entirely on CPU for the small `[L, K_max]` tensor. Using an explicit per-plugin `torch.Generator` is required for `seed` reproducibility (Pattern Q from architectural patterns).

### 9.14 — `init_alpha_from_router_scores` variant

**Upstream**: `src/search/prune.py:485-533` — alternative init that biases α away from high-routing-frequency experts using a `spread` knob. Optional path; only activated when `router_scores` is passed.

**Ours**: No router-score init.

**Verdict**: KEEP-DEVIATION. Our plugin runs as a budget refinement on top of GRAPE; routing-stats may be available via `ctx["routing_stats"]` from earlier plugins, but the upstream variant is opt-in and orthogonal. The β-bisection init (Item 6) gives equivalent per-row coverage and matches the default upstream path. Defer router-score init to a future enhancement; document as `D-no-router-prior`.

### 9.15 — Gradient computation: STE vs analytic

**Upstream prune.py:106-114**: STE through the soft Gumbel-softmax sampled in forward; gradient flows from the actual KL loss through the model.

**Ours**: Pure analytic gradient `p̃ · (D − E_p̃[D])` from the Gumbel-softmax expected damage; no model in the loop.

**Verdict**: KEEP-DEVIATION — already documented as `D-fitness-mse` (Item 8). The analytic backward is mathematically the gradient of `E_p̃[D]` w.r.t. α (paper §3.1, eq. on lines 478-482 of our docstring); upstream's STE+autograd gives the same expectation in the limit of many Gumbel samples. We use a single-sample analytic form which is exact and cheap. Re-document as `D-analytic-grad`.

### 9.16 — Bias-correction in Adam

**Upstream**: `torch.optim.Adam` defaults `amsgrad=False`; standard bias-corrected updates.

**Ours**: Manual Adam with bias-corrected `m_hat / v_hat`. Match (functionally identical to upstream's Adam).

**Verdict**: NO CHANGE.

### 9.17 — Step zeroing on pad columns

**Upstream**: no pad columns (uniform K).
**Ours**: `step = step * mask` to zero updates on pad columns.

**Verdict**: KEEP — forced by `D-ragged-K`.

### 9.18 — `_BISECT_TOL = 1e-4` comment claims it's tight enough — Item 4 fix invalidates this.

**Verdict**: Comment updated alongside Item 4 fix.

### 9.19 — Initial retraction call (line 450 in our impl)

**Ours**: After GRAPE-init, call `_retract` to pin Σp·c = B.

**Verdict**: KEEP (upstream also does this in `init_alpha_to_bits` via the `expected_bits` bisection; after β-init the budget is already on-manifold so the retract becomes a no-op).

---

## Action plan (commits)

1. **Item 6 + Item 9.1 + 9.2** (init + defaults): replace GRAPE-init with β-bisection init; default `tau_init=1.0`, `n_iterations=200`. Remove `init_peak_logit` config knob.
2. **Item 2** (anneal): replace cosine with exponential; rename `_cosine_tau` → `_anneal_tau`; update F12.
3. **Item 3 + Item 4 + Item 9.6 + Item 9.11** (retraction): flip sign convention to `α + t·c`; bisection tolerance 0.05; bracket-doubling 40 iters; projection denominator `+1e-12`. Update F3/F4.
4. **Item 5** (infeasibility tiebreak): lower-side preferred; update F8.
5. **Item 1** (per-group weights): add `weights` parameter to all manifold primitives, derive layer parameter counts from `model` slot.
6. **Item 8 + Item 9.4** (cleanup + Gumbel clamp): remove `D-adam-no-v-transport`, `D-init-grape`, `D-anneal-cosine`; add `D-dp-damage-not-logp`, `D-analytic-grad`, `D-no-router-prior`. Match Gumbel clamp/inner-log to upstream.
7. Test sweep: rerun `tests/test_stage1_plugin_rco_budget.py` + Stage 1 tests; pin all upstream-aligned values.

---

## Open Questions

None. All deviations are either:
- forced by our pipeline (KEEP-DEVIATION with D-tag), or
- ALIGN to upstream.

The `weights` derivation from MoE layer dims (Item 1) reads `model` from ctx (always present in the orchestrator). In tests that pass synthetic data without a model, `weights=None` falls back to `w_i=1` (back-compat).
