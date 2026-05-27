# Plugin #11 — S1_RCO (RCO budget allocator)

**Status**: Implemented on `feat/plugin_11_s1_rco`.
**Author**: ml-intern protocol (session planner, 2026-05-27)
**Parent plan**: `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3 (R3 — RCO) + §6.1 (row `S1_RCO`) + §7 A5 (workflow).
**Paper anchor**: arxiv 2605.00649 — *Model Compression with Exact Budget Constraints via Riemannian Manifolds* (IST-DASLab, May 2026).
**Upstream code**: https://github.com/IST-DASLab/RCO.

---

## 1. Goal

Add a Stage-1 plugin `stage1/plugins/rco_budget.py` that produces per-layer expert budgets via Riemannian-manifold optimization with an *exact* budget constraint baked into the geometry, **initialized from GRAPE** and minimizing **output-space MSE** as a fitness proxy.

When `stage1.rco_budget.enabled` is **false** (default), the plugin is a strict no-op: it does not register hooks, does not run, does not touch GRAPE's budget output. When **true**, RCO post-processes GRAPE-produced budgets and writes refined per-layer budgets to a NEW ctx slot `per_layer_target_experts_rco` that downstream Stage-2 may consume in place of GRAPE's allocation.

Constraints respected:
- Do NOT modify `grape_merge.py` or any other existing Stage-1 plugin.
- Default OFF; only the `S1_RCO` row enables the plugin.
- Stages 2-6 untouched.

---

## 2. Upstream IP / Licensing

**The upstream RCO repo (https://github.com/IST-DASLab/RCO) ships without a LICENSE file** (verified 2026-05-27 via the GitHub API: `"license": null`). Verbatim vendoring of unlicensed code into this codebase is therefore not legally clean.

This plugin is a **clean-room re-implementation from the paper's algorithm description in arxiv 2605.00649 §3 Algorithm 1** and the manifold-operation prose in `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §5.3. The upstream repo is cited as an attribution / cross-check target only. Each algorithmic choice the paper leaves under-specified is flagged as a deviation in the plugin docstring (see §6 below).

---

## 3. Algorithm Summary (arxiv 2605.00649 §3 Algorithm 1)

RCO recasts the discrete per-layer budget-allocation problem as a smooth optimization over a Riemannian manifold defined by the *exact* budget constraint. State is a per-layer logit matrix `α ∈ ℝ^{L × K}` where L is the MoE-layer count and K is the number of candidate per-layer budgets. `p_lk = softmax(α_l)_k` is the soft allocation; the constraint is `Σ_l Σ_k p_lk · c_lk = B` where `c_lk` is the surviving-expert count of option k at layer l, and `B` is the global expert budget.

Three manifold primitives drive the search:

1. **Tangent projection** of the Euclidean gradient `g`:
   `g_tangent = g − (⟨g, n⟩ / ⟨n, n⟩) · n` where `n_lk = p_lk · (c_lk − E_p[c_l])` is the constraint normal (the gradient of the constraint w.r.t. α; derivation in the plugin docstring).

2. **Retraction** via 1-D bisection along the cost direction: find `t` such that `Σ softmax(α − t·c)_k · c_lk = B`. Bracket-doubling until the budget straddles zero, then bisection.

3. **Vector transport** of Adam's first moment buffer to the new tangent plane after a step + retraction (re-project using the same formula).

The discrete forward (fitness evaluation) is **Gumbel-STE → soft probabilities** at temperature τ with **cosine annealing** of τ (high → low: explore → exploit), and a final **multiple-choice knapsack DP** that projects to a budget-exact discrete vector.

**Fitness signal** (per spec §5.3 last bullet): output-space MSE — either read from an optional `ctx["per_layer_damage_curve"]` slot (future S1_DP plugin), or built from a synthetic linear-redundancy fallback using GRAPE's R̃^l output.

---

## 4. GRAPE Initialization

GRAPE writes per-layer budgets `g_l` to `ctx["per_layer_target_experts"]` as `dict[str, int]`. RCO initialises α so that `softmax(α_l)` peaks on the option-index corresponding to `g_l`. Concretely: the option grid is `{floor_l, ..., per_layer_count_l}`; the GRAPE-chosen option gets logit `init_peak_logit` (default 2.0), all others get 0. This is a sharp distribution on `g_l` with small mass on neighbors that RCO can shift.

The `min_experts_per_layer` floor is baked into the option grid (D-floor-projection) — RCO cannot escape it.

---

## 5. Plugin Contract

| Attribute | Value |
|---|---|
| `name` | `"rco_budget"` |
| `config_key` | `"stage1.rco_budget"` |
| `reads` | `("per_layer_target_experts", "per_layer_redundancy", "per_layer_targets", "decomposition", "config")` |
| `writes` | `("per_layer_target_experts_rco", "rco_metadata")` |
| `provides` | `()` |

`is_enabled(config)` reads `config["stage1"]["rco_budget"]["enabled"]` (default false).

`run(ctx)` reads GRAPE budgets + per-layer counts + decomposition + optional damage curve; runs the RCO loop; writes refined budgets + metadata.

`contribute_artifact(ctx)` returns `{"rco_budgets": ..., "rco_metadata": ...}` when run, `{}` otherwise.

---

## 6. Deviations (D-tags)

| Tag | Deviation | Rationale |
|---|---|---|
| **D-clean-room** | Re-implemented from prose, not vendored | Upstream repo has no LICENSE (verified 2026-05-27). |
| **D-init-grape** | Initialize α from GRAPE budgets (not REAP) | Spec §7 A5 mandate. |
| **D-fitness-mse** | Output-space MSE fitness (not end-to-end loss) | Spec §5.3 last bullet; actual-loss requires the L1/vLLM substrate. |
| **D-synthetic-curve** | Linear-redundancy fallback when no damage curve on ctx | Plugin #8 (S1_DP) not yet integrated; the synthetic curve preserves GRAPE's ranking so worst case is RCO ≈ GRAPE. |
| **D-floor-projection** | Floor baked into the option grid | `min_experts_per_layer` is a project invariant (`MOE_COMPRESS_REPORT.md` §5.1). |
| **D-ragged-K** | Per-layer K varies; padded with a 0/1 mask | Layers may have different `per_layer_count − floor` ranges. |
| **D-bisection-budget** | Global retraction (1 scalar t), not per-layer | Constraint is global. |
| **D-disabled-default** | Opt-in via `stage1.rco_budget.enabled` | Default off keeps every existing row byte-identical. |

---

## 7. Orchestrator Integration

In `stage1/orchestrator.py`:
1. STEP 10b (new): after `grape_merge.run(ctx)`, call `rco_plugin.run(ctx)` gated on `is_enabled`.
2. `_write_artifacts`: after the GRAPE budget write, optionally write `stage1_rco_budgets.json` gated on `is_enabled`.

Both additions are dead code on the default path.

---

## 8. Tests (`test_stage1_plugin_rco_budget.py`)

16 tests:
1. Protocol-attribute contract (name, paper string with D-tags, reads/writes).
2. `isinstance(plugin, PipelinePlugin)`.
3. `is_enabled` false on default configs (4 sub-cases).
4. `is_enabled` true when explicit flag set.
5. Missing-slot KeyError contract (parametrised across 5 reads).
6. Synthetic 2-layer hand-check (sum-to-B, floor respected).
7. Damage-curve consumption: asymmetric curve shifts allocation toward steep layer.
8. Floor respected even with curve favouring violation.
9. Sum-to-B at B ∈ {6, 7, 8}.
10. Artifact empty when disabled.
11. Artifact populated after run.
12. Manifest registration sanity (after grape_merge).

**Pass criteria**: all 16 new tests green; full suite no regressions.

---

## 9. Files Touched

| File | Action |
|---|---|
| `max_quality/src/moe_compress/stage1/plugins/rco_budget.py` | NEW — ~650 LoC including docstring. |
| `max_quality/src/moe_compress/stage1/plugins/__init__.py` | Add import + manifest entry after `GrapeMergePlugin`. |
| `max_quality/src/moe_compress/stage1/orchestrator.py` | Gated `run` call + gated artifact write. |
| `max_quality/tests/test_stage1_plugin_rco_budget.py` | NEW — 16 tests. |
| `tasks/PLAN_PLUGIN_11_s1_rco.md` | This file. |

---

## 10. Out of Scope

- Stage 2-6 untouched.
- `grape_merge.py` untouched.
- The L1/vLLM rollout substrate.
- The DP damage-curve plugin (Plugin #8 / S1_DP) — RCO reads its slot opportunistically.
- The actual `S1_RCO` recovery row (config bundle + run, not plugin code).
