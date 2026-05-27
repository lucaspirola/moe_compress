# Plugin #8 — S1_DP — Damage-curve DP for global budget allocation

## Spec citation

- `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §7 A1 — "S1_DP" / Rec 2:
  per-layer `D_ℓ(k)` damage curve + DP knapsack → populates
  `merge_cost_prior` consumed by `GrapeMergePlugin`.
- R8 (HC-SMoE arxiv:2410.08589) — non-uniform-budget precedent.
- R4 (Additivity theorem arxiv:2308.10438) — formal basis: under
  Taylor + i.i.d. assumptions the total distortion decomposes
  additively across layers, enabling a tractable 1D DP knapsack.

## Goal

Replace the inert default `merge_cost_prior = None` in
`stage1_grape` with a measured per-layer prior so GRAPE's greedy
queue is biased by per-layer merge damage rather than CKA
redundancy alone.

## Algorithm

### Damage-curve `D_ℓ(k)` per layer

For each MoE layer ℓ, sort intra-layer expert-to-expert distances
in `D_matrices[ℓ]` (excluding self-distance + blacklisted experts)
in **ascending order** (most similar first). Define

    D_ℓ(k) = Σ_{i=1..k} sort_asc(off_diag(D_matrices[ℓ]))[i]

i.e. the cumulative sum of the `k` smallest off-diagonal
distances. Interpretation: under additivity (R4), one merge ≈ a
small Taylor perturbation; the cumulative cost of the `k`
cheapest merges in ℓ is an additive estimate of the per-layer
output damage from removing `k` experts in layer ℓ.

This **substitutes CKA distance for output-space MSE** because
the output-space cost machinery (`_output_space_cost`) lives in
Stage 2 and consumes centroid/non-centroid assignments that
don't yet exist in Stage 1. The substitution is paper-consistent
with GRAPE's choice of CKA as the redundancy primitive — it
makes S1_DP an "ablation of GRAPE that reallocates budget by DP
instead of greedy entropy-gated selection," holding the
similarity primitive fixed. We document this as deviation
**D-cka-substitute-for-output-mse** in the plugin docstring.

### Budget per-layer counts and floors

For layer ℓ with `N_ℓ` total experts, blacklist count `B_ℓ`, and
floor divisor `f` (default 2):

  - `floor_ℓ = max(N_ℓ // f, B_ℓ)`  (minimum survivors; respect
    blacklist invariant)
  - `keep_max_ℓ = N_ℓ`              (maximum survivors)
  - `k_range_ℓ = (N_ℓ - keep_max_ℓ) .. (N_ℓ - floor_ℓ)` =
    `0 .. (N_ℓ - floor_ℓ)` merges allowed

`D_ℓ(0) = 0` (no merges, no damage). `D_ℓ(k)` is monotone
non-decreasing in `k`.

### DP knapsack recurrence

Decision variables: `k_ℓ ∈ [0, k_max_ℓ]` = merges in layer ℓ.
Constraint: `Σ_ℓ k_ℓ = G` (global merge target).
Objective: minimise `Σ_ℓ D_ℓ(k_ℓ)`.

Standard 1D knapsack recurrence over layers (in any fixed
order):

    dp[i+1][b] = min over k ∈ [0, k_max_i] of (dp[i][b - k] + D_i(k))
    dp[0][0]   = 0
    dp[0][b>0] = +inf

with `i ∈ [0, L)` and `b ∈ [0, G]`. Runtime: O(L · K · G) where
K = max k_max_ℓ. For our scenario L=40, K=128, G≈40·128=5120 →
~26M ops — sub-second on CPU.

Traceback gives the optimal `k*_ℓ` per layer; the **prior** to
publish into `merge_cost_prior` is then derived from the
*marginal damage* at the optimum:

    prior_ℓ = D_ℓ(k*_ℓ + 1) - D_ℓ(k*_ℓ)

i.e. the cost of the *next* merge in layer ℓ. Layers where the
DP placed the optimum at the floor get `prior_ℓ = +inf`
(forbidden by GRAPE's `R[li] * prior[li]` rule — GRAPE will
never pick a +inf-prior layer to merge next).

Layers with `prior_ℓ = 0` (free further merges available) are
clamped to a small positive epsilon so GRAPE's selection
`R · prior` does not collapse to 0.

### Output

`merge_cost_prior: dict[int, float]` keyed by layer index,
published into the **config** at `stage1_grape.merge_cost_prior`
so `GrapeMergePlugin.run` picks it up via its existing inert
hook.

## API contract

- **Name**: `damage_curve_dp`
- **Paper**: `R8 HC-SMoE arxiv:2410.08589 (non-uniform budget precedent); R4 Additivity arxiv:2308.10438 (DP formal basis). Deviation D-cka-substitute-for-output-mse: damage curve uses CKA off-diagonal distance sums instead of paper Rec 2's output-space MSE (the Stage 2 _output_space_cost machinery isn't available at Stage 1). Deviation D-dp-prior-as-marginal: prior published is the marginal damage at the DP optimum, not the cumulative damage.`
- **config_key**: `stage1_grape.damage_curve_dp.enabled`
- **reads**: `("D_matrices", "blacklist", "per_layer_targets", "decomposition", "config")`
- **writes**: `("damage_curves", "dp_optimum", "merge_cost_prior_computed")` — and also writes the prior into `config["stage1_grape"]["merge_cost_prior"]` so the inert GRAPE hook activates.
- **provides**: `()` — pure post-process over `D_matrices`.
- **is_enabled**: returns true iff `config["stage1_grape"]["damage_curve_dp"]["enabled"]`.
- **run order**: must run AFTER `cka_distance` (produces D_matrices) and BEFORE `grape_merge` (consumes the prior).
- **`contribute_artifact`**: returns `{}` — diagnostics live on ctx slots; no JSON artifact.

## Files to touch

1. **New**: `max_quality/src/moe_compress/stage1/plugins/damage_curve_dp.py` (~200 LoC)
2. **Edit**: `max_quality/src/moe_compress/stage1/plugins/__init__.py` — add `DamageCurveDpPlugin` to manifest in the correct position (after CKA, before GRAPE).
3. **Edit**: `max_quality/src/moe_compress/stage1/orchestrator.py` — call the new plugin's `.run(ctx)` in STEP 10 between `cka_distance.run` and `grape_merge.run`.
4. **Edit**: `max_quality/configs/qwen36_35b_a3b_30pct.yaml` — add `stage1_grape.damage_curve_dp.enabled: false` (default OFF, byte-identical to current behaviour).
5. **New**: `max_quality/tests/test_stage1_plugin_damage_curve_dp.py` — unit + integration tests.

## Tests

### Unit
- **U1** damage curve formula: synthetic 4-layer × 8-expert distance matrices with hand-checked cumulative sums.
- **U2** DP knapsack on small toy (3 layers × 4 experts × budget=5) with hand-checked optimum.
- **U3** floor respected: DP cannot exceed `k_max_ℓ` per layer.
- **U4** plugin Protocol attributes match contract.
- **U5** `is_enabled` correctly gates on `enabled: True / False / missing`.
- **U6** marginal-prior derivation: at optimum `k*`, prior == `D(k*+1) - D(k*)`; layers at floor get `+inf`.

### Integration
- **I1** when disabled, `merge_cost_prior` slot is **not** set on the config (byte-identical to GRAPE-only path).
- **I2** when enabled, `merge_cost_prior` is populated for all layers; type is `dict[int, float]`; values are non-negative finite or `+inf`.
- **I3** chained with `GrapeMergePlugin`: GRAPE consumes the produced prior and the run succeeds.

### Gates
- `pytest max_quality/tests/test_stage1_plugin_damage_curve_dp.py -v` — all new tests green.
- `pytest max_quality/tests/ -q --timeout=600` — no regressions (1550+N passed, 13 skipped).

## Risk + halt triggers
- **R-DP-inf-trap**: if every layer's `k_max_ℓ = 0` (over-blacklisted), DP has no feasible budget; return prior_ℓ = +inf everywhere and log a warning. GRAPE will then merge nothing → caller's existing global-feasibility check fires.
- **R-monotonicity**: if some `D_ℓ(k+1) < D_ℓ(k)` (numerically — should not happen since we cumsum sorted-ascending distances), clamp to monotone and warn.
- **R-cka-substitute**: doc-only — flagged as a known deviation. The S1_DP row in the plan is positioned as a "cheap baseline" anyway (§5.4 R4 explicitly says "this is positioned as a cheap baseline against S1_RCO, not the headline").

## Out of scope
- Output-space MSE damage curves (those need Stage 2 machinery; deferred to S1_RCO or a future S1_DP_OUTPUT variant).
- Modifying `GrapeMergePlugin` — the inert hook is already there, we only populate it.
