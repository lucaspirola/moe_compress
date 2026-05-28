# Plan — RCO Native Re-Implementation (clean-room, third path)

**Status**: Planning. Pre-implementation. Awaiting plan-reviewer per [[paper-fidelity-review-loop]] step 2.
**Branch**: `feat/plugin_11_rco_native_reimpl` (cut from `main@d7bfa22`).
**Parent plans**: `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 (R3 — RCO); `tasks/PLAN_PLUGIN_11_s1_rco.md` (the earlier clean-room plan, already implemented and merged at `269e64d`).
**Paper anchor**: arxiv 2605.00649 — *Model Compression with Exact Budget Constraints via Riemannian Manifolds* (IST-DASLab, May 2026). §3 Algorithm 1.
**Upstream code**: https://github.com/IST-DASLab/RCO (re-verified 2026-05-28: GitHub API `"license": null`; default branch `main`; last `updated_at` 2026-05-16). NO LICENSE file.

---

## 0. Why this plan exists (the "third path")

The product brief enumerates three options for ingesting RCO:

1. **Vendor verbatim from upstream.** Blocked: upstream repo has no LICENSE file. Verbatim copy of unlicensed code into our repo is not legally clean. Verified `"license": null` via the GitHub API on 2026-05-27 and re-verified 2026-05-28.
2. **Wait for upstream + author consent.** A `feat/plugin_11_rco_revendor` branch exists with three small Stage-2 hoists but never received the consented vendor — the user has decided not to wait for legal clarity from a third party. Author consent without a repository-level LICENSE is still legally ambiguous because consent does not bind future readers of the original repo nor establish a redistribution licence for our forks.
3. **Clean-room re-implementation in moe_compress's plugin architecture.** ← The chosen path.

The repo *already contains* a clean-room implementation at `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` (~900 LoC, merged to main as `269e64d`, with 16 tests at `max_quality/tests/test_stage1_plugin_rco_budget.py`). It was written from paper prose, has 8 documented `D-*` deviations, and is gated default-OFF behind `stage1.rco_budget.enabled`.

This plan's job is therefore **not** "write RCO from zero" but:

- **Audit the existing clean-room implementation against the paper** and confirm it is the canonical native implementation;
- **Identify gaps / improvements** the prior plugin sweep (Patterns A-M in `[[architectural-patterns]]`) and the audit findings (`tasks/PLAN_PLUGIN_14_sidecar_audit.md`) surfaced;
- **Decide replace-vs-augment** and produce a concrete delta plan;
- **Surface open questions** where the upstream paper is ambiguous and the existing implementation made a choice the next reviewer should re-affirm.

The plan is reviewed *before* code lands per [[paper-fidelity-review-loop]] step 2 (plan-reviewer step).

---

## 1. Algorithm reference (paper-only, clean-room)

This section is the canonical native description of RCO, derived from arxiv 2605.00649 prose + `SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 + the paper-summary returned by an authenticated query to the upstream README on 2026-05-28. NO upstream `.py` code is referenced. Where the paper is under-specified, the open question is logged in §8.

### 1.1 Problem statement

Given:
- `L` groups (in our setting: MoE-bearing decoder layers).
- For each group `l ∈ {1, ..., L}`, a finite set of `K_l` discrete options, each with a positive integer cost `c_lk` (in our setting: a candidate surviving-expert count for that layer).
- A global integer budget `B > 0` (in our setting: `decomposition.global_expert_budget`).
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

Derivation: applying the standard softmax Jacobian `∂p_lj/∂α_lk = p_lj (δ_jk − p_lk)` to `Σ_j p_lj c_lj` collapses to the formula above. The intuition: shifting α_lk up makes option `k` more probable, weighted by the deviation of its cost from the group's expected cost.

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

1. **Bracket-doubling.** Starting from `t=0` (residual = current drift), double `|t|` in the residual-killing direction until the sign flips.
2. **Bisection.** Halve the bracket until `|f(t)| ≤ tol`.

Termination: 32 doublings × 60 bisections is plenty for any realistic (`c`, `α`) scale.

#### Primitive 4: vector transport

Adam's first moment buffer `m` lives in tangent space at the *old* `α`. After a step + retraction, the tangent space changes; transporting `m` to the new manifold point is just the same Gram-Schmidt projection applied at the new `α`:

```
m_new = m − (⟨m, n_new⟩ / ⟨n_new, n_new⟩) · n_new.
```

This is computationally negligible — one inner product + one scalar-vector subtraction.

### 1.3 Discrete fitness signal (Gumbel-STE + DP knapsack)

The paper §3.2 evaluates fitness on a *discrete* sample because the objective `J` is typically not differentiable in `α` (e.g. quantization error of a packed checkpoint, end-to-end task loss after a discrete prune). The flow is:

1. **Gumbel perturbation.** Sample i.i.d. Gumbel noise `g_lk = −log(−log(u_lk))` with `u ~ Uniform(0, 1)`. Form perturbed logits `α' = α + τ · g`.
2. **Softmax relaxation.** Compute `p̃ = softmax(α')`. At τ → ∞, p̃ is uniform; at τ → 0, p̃ is argmax. Cosine-anneal τ from `τ_init` (default 5.0) to `τ_final` (default 0.5).
3. **Discrete projection (training-time, optional).** A multiple-choice knapsack DP over `(L, B)` projects p̃'s argmax to the closest budget-exact discrete vector. Time: `O(L · K_max · B)`.
4. **Soft objective for backward.** Compute `J_soft = Σ_l ⟨p̃_l, D_l⟩` (or the paper's task-loss surrogate) and differentiate analytically: the same softmax Jacobian collapse yields `∂J_soft/∂α_lk = p̃_lk · (D_lk − ⟨p̃_l, D_l⟩)`.

The 1/τ scale factor that would normally come from `softmax(α/τ)` is absorbed into the Adam learning rate.

### 1.4 Outer loop (Algorithm 1)

```
init α from REAP saliency (paper §3.3) OR GRAPE budgets (our D-init-grape)
init Adam buffers m=0, v=0
for t = 1..T_iter:
    τ_t = cosine_anneal(τ_init, τ_final, t/T_iter)
    grad = ∇_α [Gumbel-STE soft objective] at temperature τ_t
    grad_tangent = project_off_normal(grad, normal(α))
    (m, v) = adam_update(grad_tangent, m, v)
    step = -lr · m_hat / (sqrt(v_hat) + ε)
    α = α + step
    α = retract(α, c_grid, B)
    m = project_off_normal(m, normal(α))   # vector transport
return discrete_argmax_then_DP(α, c_grid, B)
```

**Hyperparameters exposed by the paper**: `n_iterations` (default 500-2000 depending on problem size), `learning_rate` (default 0.01-0.1), `gumbel_tau_init` (default 5.0), `gumbel_tau_final` (default 0.5), Adam (β1=0.9, β2=0.999, ε=1e-8), `seed`. The paper explicitly notes (abstract): *"avoids constraint-specific hyperparameters"* — the constraint primitives are parameter-free.

### 1.5 Cost & complexity

- Per iteration: one forward `softmax + Gumbel sample` (`O(Σ K_l)`), one backward (analytic), one retraction (`O(log(1/ε)) · O(Σ K_l)`), one vector transport (`O(Σ K_l)`).
- DP projection (once at end, or once per fitness probe): `O(L · K_max · B)`.
- For our scale (L = 40-48 MoE layers, K_max ≤ 128, B ≤ 256), one full RCO run is sub-second on CPU.

---

## 2. Slot mapping in moe_compress

| Slot dimension | Decision |
|---|---|
| **Pipeline stage** | Stage 1 (budget allocation). Spec `SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 places R3 (RCO) at "Stage 1 budget allocation". |
| **Phase / manifest position** | Phase G — strictly AFTER Phase F (`grape_merge`). RCO initialises α from GRAPE's output (D-init-grape). |
| **Existing plugin id** | `rco_budget` at manifest index 9. Same id retained — see §7 (replace, not augment). |
| **Orchestrator hook** | `stage1/orchestrator.py` STEP 10b (already present at lines 604-612). Gated on `plugin.is_enabled(config)`. |
| **Artifact** | `artifacts_dir / "stage1_rco_budgets.json"` — already present at orchestrator lines 727-741. |
| **Default state** | OFF. The `S1_RCO` ablation row enables it; every other row stays byte-identical. |
| **Downstream consumers** | None inside moe_compress (verified by `grep -rn per_layer_target_experts_rco max_quality/src`). The refined budget is a **new** ctx slot (`per_layer_target_experts_rco`) sitting alongside GRAPE's existing `per_layer_target_experts`. Stage 2 reads whichever slot the row recipe names — the redirection is a row-config concern, not a plugin concern. |

**Decision: no new manifest position is needed.** The existing slot is correct (between GRAPE and the end of Stage 1; opt-in via the same gate). The implementation will *replace* the file at `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` in-place rather than introduce a `rco_budget_v2.py`.

---

## 3. Plugin contract

The contract is FROZEN against the existing plugin so byte-identical regression tests still pass on the default OFF path:

```python
class RCOBudgetPlugin:
    name: str = "rco_budget"
    paper: str = (
        "RCO: IST-DASLab arxiv:2605.00649 §3 Algorithm 1. "
        "Upstream code at github.com/IST-DASLab/RCO ships without a LICENSE file; "
        "this is a clean-room re-implementation from the paper's prose. "
        "Deviations: D-clean-room, D-init-grape, D-fitness-mse, D-synthetic-curve, "
        "D-floor-projection, D-ragged-K, D-bisection-budget, D-disabled-default. "
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

**Optional read (no KeyError if absent)**: `per_layer_damage_curve` — `dict[int, dict[int, float]]` of `D_l(k)` produced by Plugin #8 (`damage_curve_dp`). When present, RCO consumes it (the principled path); when absent, RCO falls back to the synthetic linear curve from GRAPE's `per_layer_redundancy` (D-synthetic-curve).

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
    gumbel_tau_init: 5.0     # cosine anneal start
    gumbel_tau_final: 0.5    # cosine anneal end
    init_peak_logit: 2.0     # GRAPE-option peak height
    floor_divisor: 2         # floor_l = per_layer_count_l // floor_divisor
    seed: 0                  # RNG for Gumbel samples
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-8
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
| `per_layer_damage_curve` | `dict[int, dict[int, float]]` (layer_idx → {surviving_k → damage}) | `damage_curve_dp` (Phase E.5, Plugin #8). Optional; falls back to synthetic curve. |

### 4.4 ctx writes

| Slot | Type | Consumer |
|---|---|---|
| `per_layer_target_experts_rco` | `dict[str, int]` | Stage 2 (when the row recipe selects RCO over GRAPE) |
| `rco_metadata` | `dict` | Operator logs + the `stage1_rco_budgets.json` artifact |

### 4.5 Artifact

`artifacts_dir / "stage1_rco_budgets.json"` — written by `_write_artifacts` ONLY when the plugin is enabled. Schema:

```json
{
  "rco_budgets": {"0": 3, "1": 4, ...},     // per_layer_target_experts_rco
  "rco_metadata": {
    "init_fitness": float, "final_fitness": float,
    "init_budget_vector": {"0": 3, ...}, "final_budget_vector": {"0": 3, ...},
    "n_iterations": int,
    "achieved_budget": int, "requested_budget": int,
    "fitness_source": "damage_curve" | "synthetic"
  }
}
```

---

## 5. Architectural pattern adoption (Patterns A-M)

Re-evaluating the existing plugin against `[[architectural-patterns]]`:

| Pattern | Verdict | Action in re-impl |
|---|---|---|
| **A — Cache-aware plugin skip** | Not applicable. RCO has no precomputed sidecar of its own. | None. |
| **B — Versioned sidecar payload** | **Adopt for `rco_metadata`.** Add a `format_version: 1` (forward-only) to the metadata dict + the artifact, so a future RCO v2 reader can route old/new outputs cleanly. | New: bump artifact schema to add `format_version: 1`; tests pin both readers. |
| **C — Config-validation-at-top-of-run()** | **Currently weak.** The existing impl reads config keys lazily (`rco_cfg.get(...)` with defaults). Hidden mis-keys (e.g. `learning_rates` vs `learning_rate`) silently fall through to defaults. | New: add a `_validate_config(cfg)` call as the FIRST statement of `run()` that rejects unknown keys + range-checks (n_iterations > 0, 0 < lr < 10, 0 < τ_init, τ_final ≤ τ_init, 0 ≤ β1, β2 < 1, ε > 0, floor_divisor ≥ 1). |
| **D — Pre-flight config override** | Not applicable. RCO does not need to mutate config for a downstream plugin. | None. |
| **E — Algorithm branch knob + default-byte-identical** | **Consider.** Adding a `fitness_signal: "synthetic" \| "damage_curve" \| "auto"` knob with default `"auto"` preserves current behaviour (auto-detects via `ctx.has`) but lets row recipes force one path explicitly for ablation purity. | New: add the knob with default `"auto"`; mark `"damage_curve"` strict mode (raises if slot absent) for the ablation. |
| **F — on_post_merge hook + Position B invalidation** | Not applicable. RCO is a single-shot Stage-1 transform; no inter-layer caches to invalidate. | None. |
| **G — Marginal prior as `in_ctx_config` slot** | Not applicable. RCO writes a budget dict, not a prior. | None. |
| **H — Clean-room + license-check before vendoring** | **Already adopted.** This entire plan IS Pattern H. | Re-affirm in docstring; cite this plan. |
| **I — Layer-disjoint key invariant** | RCO reads all layers' GRAPE budgets + writes all layers' RCO budgets in a single call. The invariant ("per-layer state read only at current layer's key") is satisfied trivially — there is no inter-layer state mutation. | None; document in module docstring. |
| **J — Plan-reviewer step** | This plan IS the plan-reviewer trigger. | Spawn plan-reviewer (per project workflow). |
| **K — Forward-only schema bumps** | Pairs with Pattern B above. | None new. |
| **L — Reviewer sub-agents in worktrees** | Workflow concern, not implementation. | None. |
| **M — Post-fix reviewer worktree freshness** | Workflow concern. | None. |

**Plan-reviewer C-checks** (from `[[architectural-patterns]]`):

- **C-INV (Invalidation-target consumer audit)**: RCO does not invalidate any slot — it only adds a new slot. ✓ No action required.
- **C-PROD-CFG (Production-default state audit)**: Default `enabled: false`. Both the `run()` call and the `contribute_artifact` write are gated on `is_enabled`. Production-default path: zero work, zero artifact, identical state to pre-plugin-#11 main. ✓
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
| F7 | `test_dp_knapsack_optimality` | The multiple-choice DP finds the optimum on a hand-graded 3-layer × 4-option instance (cross-check against a brute-force enumeration). |
| F8 | `test_dp_handles_infeasible_budget` | When B falls outside `[Σ floor_l, Σ N_l]`, the DP fails over to the nearest feasible budget with a WARNING log (no silent corruption). |
| F9 | `test_algorithm_converges_on_quadratic_proxy` | For a synthetic quadratic damage curve where the analytic minimum is known, RCO converges to within ε of the optimum in ≤200 iterations. |
| F10 | `test_seed_reproducibility` | Two runs with the same seed produce identical final budget vectors (catches any non-deterministic ops). |

### 6.2 Code-quality tests (config validation, edge cases)

| # | Test | What it verifies |
|---|---|---|
| C1 | `test_plugin_protocol_attributes` | (Existing.) Class attrs match the contract. |
| C2 | `test_plugin_is_runtime_checkable_pipelineplugin` | (Existing.) `isinstance(plugin, PipelinePlugin)`. |
| C3 | `test_plugin_disabled_by_default` | (Existing.) Default-OFF semantics. |
| C4 | `test_plugin_enabled_when_flag_true` | (Existing.) Flag-on enables. |
| C5 | `test_run_rejects_missing_slot[5 params]` | (Existing.) KeyError contract. |
| C6 | `test_config_rejects_unknown_keys` | **NEW** (per Pattern C). `n_iterations=500, learning_rates=0.1` (typo) raises `ValueError` listing the unknown key. |
| C7 | `test_config_range_checks` | **NEW** (per Pattern C). Invalid ranges (`n_iterations=0`, `tau_init=-1`, `floor_divisor=0`) all raise `ValueError`. |
| C8 | `test_empty_layer_set_raises` | `per_layer_targets = {}` raises `ValueError` (existing path already does this). |
| C9 | `test_grape_budget_outside_grid_warns_and_clamps` | Pre-existing — verify the log line. |
| C10 | `test_floor_respected_with_pathological_damage_curve` | (Existing.) Even with `D_l(k=2) = 0; D_l(k=floor+1) = ∞`, RCO never goes below floor. |
| C11 | `test_fitness_signal_strict_mode_raises_when_curve_absent` | **NEW** (Pattern E). When `fitness_signal: "damage_curve"` is set and the slot is absent, raise (no silent fallback). |
| C12 | `test_artifact_includes_format_version` | **NEW** (Pattern B). The `stage1_rco_budgets.json` includes `format_version: 1`. |
| C13 | `test_artifact_empty_when_disabled` | (Existing.) Default-OFF artifact empty. |
| C14 | `test_artifact_populated_when_enabled` | (Existing.) Post-run artifact correctness. |
| C15 | `test_manifest_position_after_grape_merge` | (Existing.) Manifest order. |

### 6.3 Regression / byte-identity tests

| # | Test | What it verifies |
|---|---|---|
| R1 | Full Stage-1 default-OFF byte-equality vs main@d7bfa22 | Run `stage1_orchestrator` on the canonical Qwen3 fixture with `stage1.rco_budget.enabled: false` — every file in `artifacts_dir` must be byte-identical to the pre-re-impl artifacts. Existing 20-snapshot harness from the plugin audit catches this. |
| R2 | Existing `test_run_synthetic_2layer_handcheck` and `test_run_consumes_damage_curve_when_present` | Must still pass with the same expected allocations (4+2 = 6, etc.). |
| R3 | `test_run_sums_to_global_budget[6, 7, 8]` | Existing parametrised test. |

**Pass criteria**: all F1-F10 + C1-C15 + R1-R3 green. Total: 10 + 15 + 3 = **28 tests** (12 of which already exist; 16 are new or upgraded).

---

## 7. Migration / deprecation

**Decision: REPLACE the existing Plugin #11 in-place, not augment.**

Rationale:
- Same algorithm (RCO §3 Algorithm 1) — adding a second plugin would split readers between the two and accrete cruft.
- Same contract (name, reads, writes, gate key) — no downstream consumer changes.
- The improvements (Patterns B, C, E) are forward-compatible deltas, not algorithm pivots.
- The existing branch `feat/plugin_11_rco_revendor` (the consent-vendor path) is **abandoned** by this plan; close it without merging.

**Migration steps**:

1. Re-write `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` on this branch (`feat/plugin_11_rco_native_reimpl`). The file is REPLACED, not appended. Existing API (`name`, `reads`, `writes`, `is_enabled`, `run`, `contribute_artifact`) is preserved bit-for-bit on the OFF path.
2. Extend `max_quality/tests/test_stage1_plugin_rco_budget.py` with the new F1-F10 + C6, C7, C11, C12 tests; keep all existing tests as-is.
3. No changes to `max_quality/src/moe_compress/stage1/orchestrator.py` (the gated call sites at STEP 10b and `_write_artifacts` already exist).
4. No changes to `max_quality/src/moe_compress/stage1/plugins/__init__.py` (manifest entry already present).
5. Update the module docstring of `rco_budget.py` to:
   - cite this plan (`tasks/PLAN_RCO_NATIVE_REIMPL.md`);
   - re-anchor the clean-room note with the 2026-05-28 re-verification timestamp;
   - re-list the eight `D-*` deviations PLUS the new ones (see §8 open questions).
6. Abandon `feat/plugin_11_rco_revendor` (the consent-vendor branch). It contains three Stage-2 hoists unrelated to RCO that have already been backported to main; the branch carries no unique value.

**Deprecation:** none. The old plugin was a draft; the new one is the canonical native implementation.

---

## 8. Open questions

These need a plan-reviewer or user decision before the implementer can proceed cleanly.

### Q1 — `J` decomposability

The paper (§3.1) introduces RCO for *non-decomposable* objectives `J(k_1, ..., k_L)` (e.g. end-to-end task loss). In our setting the fitness `Σ_l D_l(k_l)` is fully decomposable per layer — meaning RCO degenerates to "minimise a sum of per-layer scalar damage curves under a global budget", which a plain integer DP solves exactly in `O(L · K_max · B)` with NO gradient descent, NO manifold, NO Gumbel.

**Question**: Are we using RCO because (a) we anticipate a future non-decomposable fitness (the L1/vLLM end-to-end-loss path from `L1_FOR_SC_PLAN.md`), or (b) because the spec mandates it for the `S1_RCO` ablation row regardless of decomposability?

If (a): the synthetic-curve fallback is the *only* path that exercises the non-trivial RCO machinery; we should mark it clearly as "exercising a future feature".

If (b): the implementation is correct as-is but the value-add over Plugin #8 (`damage_curve_dp`'s pure DP) is purely *future-proofing*. Worth a docstring callout.

**Existing impl's answer**: (b) — the docstring cites spec §6.1 row `S1_RCO` and notes worst-case "RCO ≈ GRAPE" for the synthetic-curve fallback. The plan-reviewer should re-affirm this and decide whether to add a `D-decomposable-fitness` deviation tag explicitly.

### Q2 — Initialisation: REAP saliency vs GRAPE budgets

Paper §3.3 initialises α from REAP saliency scores (a per-expert importance score that ranks experts within each layer). Our implementation initialises from GRAPE per-layer budgets via `D-init-grape`.

**Concrete reason for the deviation**: REAP saliency is per-*expert*, not per-budget-*option*. To use REAP, we'd need a per-(layer, surviving_k) saliency, which the upstream paper *constructs* from a forward pass over candidate budgets. Our pipeline does not run that pass — GRAPE's output is the cheapest reasonable initialisation.

**Question**: Is the GRAPE initialisation actually meaningfully different from a uniform initialisation at high `τ_init = 5.0`? At τ=5, the softmax is nearly uniform, so a tiny init perturbation (peak height 2.0) gets washed out within ~30 iterations. The annealed-τ schedule then re-concentrates the distribution toward whatever the gradient signal prefers.

If the answer is "GRAPE init makes <1% difference in final allocation", then `D-init-grape` is mostly cosmetic and could be removed (use uniform `α=0` init). If the answer is "GRAPE init biases toward GRAPE's basin and matters at convergence", keep it.

**Recommendation**: add a Q2 ablation test (F11): `test_init_grape_vs_uniform_at_high_tau_converges_to_same_basin`. If it passes, drop `D-init-grape` from the deviation list. If it fails, keep the deviation and document the basin difference.

### Q3 — Fitness signal: output-space MSE vs end-to-end task loss

Paper §4.2 uses end-to-end task loss (validated via vLLM rollout) as the fitness signal. Our implementation uses output-space MSE (`D-fitness-mse`). The L1/vLLM substrate is a separate work item (`L1_FOR_SC_PLAN.md`).

**Question**: Should the re-impl carry a `fitness_provider` hook so the L1 work, when it lands, can plug in end-to-end task loss without re-writing RCO? A small abstraction would be:

```python
class RCOFitnessProvider(Protocol):
    def damage_grid(self, alpha: Tensor, k_options, ...) -> Tensor: ...

# default: SyntheticOrCurveProvider (reads ctx slot OR builds synthetic)
# L1 future: VllmRolloutProvider (calls into the L1 substrate)
```

**Recommendation**: defer until L1 lands. Adding the hook now costs little but YAGNI bites. Keep the in-line implementation; if the hook is added later it's a `D-fitness-provider` deviation, not a contract break.

### Q4 — Bisection tolerance scale

`_BISECT_TOL = 1e-4` is tight enough for the DP projection (which only needs ~0.5-expert resolution) but is hardcoded. At very large budgets (B > 10,000) or unusual cost scales, this could be loose.

**Question**: Should `_BISECT_TOL` scale with B (e.g. `tol = max(1e-4, 1e-7 · B)`) or stay fixed? Current scale (B ≤ 512 in practice) makes it irrelevant; future-proofing argues for scaling.

**Recommendation**: leave fixed for now; document the assumption in the constant's comment.

### Q5 — Multi-precision / vectorisation

The existing impl runs in `torch.float64` on CPU. For our scale (12k entries × 500 iters) this is sub-second so optimisation is unnecessary. But the impl uses small-batch CPU loops (e.g. the DP knapsack triple-nested loop) that are O(L·K·B) which at L=48 × K=64 × B=256 = 786k ops — fine, but easy to vectorise.

**Question**: Vectorise the DP loop, or keep the explicit indexing for readability? Vectorisation would make the byte-identical regression brittle (different fp accumulation order).

**Recommendation**: keep explicit indexing. Byte-identical regression is more valuable than a sub-millisecond speedup.

### Q6 — `D-init-grape` defensive clamp behaviour

Current code:
```python
g_l_clamped = max(opts[0], min(opts[-1], g_l))
if g_l_clamped != g_l:
    log.warning("RCO: layer %d: GRAPE budget %d clamped...", ...)
```

GRAPE *should* respect the same floor / per-layer count as RCO. If clamping ever fires, it indicates an upstream invariant violation.

**Question**: Should this be a hard `ValueError` instead of a warning? Silent warnings get lost in long log streams.

**Recommendation**: promote to `ValueError`. The plan-reviewer should confirm — if there's any pipeline configuration where GRAPE's `floor_divisor` legitimately differs from RCO's, this would need to stay a warning.

### Q7 — Recommendation strength for the re-impl: do it or not?

**The existing implementation is correct and well-tested.** The proposed deltas are quality-of-life improvements (Pattern B versioning, Pattern C config validation, Pattern E branch knob) — not bug fixes. None of the eight existing `D-*` deviations are problematic per paper §3.

**Question for the user**: is the re-impl effort worth it given that the existing one works? Three possible postures:

- **Posture A (full re-impl)**: rewrite `rco_budget.py` end-to-end on this branch, apply all Pattern-B/C/E upgrades, add F1-F10 paper-fidelity tests. Estimated effort: 1 implementation session + 1 paper-fidelity review + 1 code-quality review = ~4-6 hours of agent time.

- **Posture B (incremental)**: keep the existing file, layer in only the Pattern-C config validation + a single new F-test for Algorithm 1 mathematical correctness end-to-end. ~1 hour.

- **Posture C (no change)**: declare the existing Plugin #11 as the canonical native implementation; close this branch; the "third path" is already taken. ~0 hours.

The current plan defaults to Posture A on the assumption that the user's "third path" framing signals genuine appetite for a rewrite. The plan-reviewer should escalate this choice to the user if the answer is not unambiguously A.

---

## 9. Estimate

### 9.1 File list

| File | Action | Approx LoC |
|---|---|---|
| `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` | REPLACE (in-place rewrite) | ~950 (was ~900) |
| `max_quality/tests/test_stage1_plugin_rco_budget.py` | EXTEND (add F1-F10, C6, C7, C11, C12) | ~700 (was ~330) |
| `max_quality/src/moe_compress/stage1/orchestrator.py` | NO CHANGE | — |
| `max_quality/src/moe_compress/stage1/plugins/__init__.py` | NO CHANGE | — |
| `tasks/PLAN_RCO_NATIVE_REIMPL.md` | THIS FILE (created in this commit) | ~700 |

### 9.2 Test list (28 total)

- Paper-fidelity: F1-F10 (10 NEW)
- Code-quality: C1-C5 (5 existing), C6, C7 (2 NEW Pattern C), C8 (1 existing), C9-C10 (2 existing), C11 (1 NEW Pattern E), C12 (1 NEW Pattern B), C13-C15 (3 existing)
- Regression: R1 (existing 20-snapshot byte-equality harness), R2 (2 existing tests), R3 (1 existing parametrised test)

### 9.3 Coupling to existing plugins

| Plugin | Coupling | Risk |
|---|---|---|
| `grape_merge` (Phase F) | RCO reads `per_layer_target_experts` and `per_layer_redundancy` from GRAPE's writes. | Low — contract stable. |
| `damage_curve_dp` (Phase E.5, Plugin #8) | RCO optionally reads `per_layer_damage_curve` from S1_DP. | Low — `ctx.has()` guarded. |
| `budget.solver.BudgetDecomposition` | RCO reads `global_expert_budget`. | None — public dataclass. |
| Stage 2 (any plugin) | RCO writes `per_layer_target_experts_rco`; Stage 2 reads whichever budget slot the row recipe names. | Zero — no Stage 2 plugin currently reads the `_rco` slot. |

### 9.4 Estimated effort (Posture A, full re-impl)

- Plan-reviewer cycle: 1 round, ~30 min.
- Implementer: 1 session, ~2 hours (rewrite + new tests + run gates).
- Paper-fidelity reviewer: 1 round (likely 0-2 fix iterations), ~45 min.
- Code-quality reviewer: 1 round (likely 0-1 fix iteration on a fresh rewrite), ~45 min.
- Commit + merge to main (no PR per [[no-pr-language]]): ~5 min.

**Total: ~4.5 hours of agent time across the workflow.**

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
| Existing clean-room impl | `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` (39 KB, 899 lines) | Merged to main at `269e64d`. To be REPLACED in §7. |
| Existing tests | `max_quality/tests/test_stage1_plugin_rco_budget.py` (16 tests, ~330 lines) | Merged. To be EXTENDED in §6. |
| Existing plan | `tasks/PLAN_PLUGIN_11_s1_rco.md` | Done. This plan supersedes it for the re-impl effort. |
| Abandoned consent-vendor branch | `feat/plugin_11_rco_revendor` (3 commits, Stage-2 hoists already in main) | Close without merging per §7. |
| Spec authority | `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 (R3 — RCO) | Unchanged. |
| Architecture patterns | `[[architectural-patterns]]` (memory: `project_architectural_patterns`) | Pattern catalogue; §5 enumerates the relevant ones. |
| Workflow | `[[paper-fidelity-review-loop]]`, `[[review-fix-loop-protocol]]` | This plan triggers step 2 (plan-reviewer). |

## Appendix B — Five-line algorithm summary

RCO casts discrete budget allocation as soft-relaxed optimisation on a Riemannian manifold `M = { α : Σ p · c = B }` in logit space where `p = softmax(α)`. Three closed-form primitives — constraint-normal `n = p · (c − E_p[c])`, Gram-Schmidt tangent projection, and bisection-along-cost retraction — wrap a standard Adam optimiser so every iterate is budget-exact by construction. A Gumbel-STE forward pass with cosine-annealed temperature samples discrete assignments for the fitness signal; the analytic backward differentiates the same softmax-Jacobian collapse used by `n`. Adam's first moment is vector-transported by re-projection after each retraction. A final multiple-choice knapsack DP locks the soft logits to a budget-exact integer assignment, completing one RCO run in `O(T · L · K_max + L · K_max · B)` sub-second on CPU at our scale.
