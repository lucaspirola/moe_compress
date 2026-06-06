# PLAN — Faithful-to-Upstream REAP Pruner (pure expert PRUNING)

**Branch:** `feat/reap-faithful-pruner`
**Status:** PLANNING ONLY (no production code in this commit). Plan-review loop next.
**Upstream:** CerebrasResearch/reap (cloned `/tmp/reap_upstream`, depth-1, default branch `main` as of 2026-06-06). Paper: arXiv:2510.13999.

**Plan-review revision (rev-2, 2026-06-06):** Six independently-confirmed findings
folded in. See §10 for the point-by-point resolution. Headline corrections:
- Resume is **broken** in the naive design — `resume.py:135` requires both
  `merge_{idx}.json` AND `layer_{idx}.pt`; faithful mode collects no covariance
  so `_snapshot_cov_layer` (`shared_io.py:77-79`) never writes the `.pt`, so every
  layer is silently re-run on resume. Fix in §5 (faithful mode writes an empty
  sentinel `.pt`; option (a)).
- `LayerMergePlugin.write_artifacts` (`layer_merge.py:658-668`) `ctx.get()`s ~12
  bump-loop slots that the faithful early-branch never sets, and `ctx.get` raises
  `KeyError` on a missing slot (`context.py:91`). So we adopt wiring (a):
  `ReapPrunePlugin` owns `merge`/`post_merge`/`write_artifacts` and emits ONLY the
  fields `resume.py` reads — see §3.7.
- `ctx.get("target")` (`orchestrator.py:249`) is the Stage-1 **GRAPE-allocated**
  per-layer expert budget (`per_layer_target_experts`), NOT a 35% expert fraction.
  Provenance corrected (nit C): `budget/solver.py:113-119` explicitly states
  `per_layer_target_experts` is **NOT a solver output** — the solver yields only the
  global `global_expert_budget`; GRAPE in Stage 1 distributes it non-uniformly
  across layers (activation-aware CKA, `min_experts_per_layer` floor). It is driven
  by the overall reduction config, not a clean 35% expert drop. Faithful mode
  BYPASSES it and computes `n_prune` from an explicit `prune_fraction` — §3.3.
- Protected-experts exclusion is a **genuine divergence** from upstream's
  default-OFF super-expert preservation (`args.py:515,523` both `default=False`) —
  re-documented in §7 with a `protected=∅` upstream-formula byte-match test.

---

## 0. TL;DR — Ground-truth finding

**A faithful drop+router-slice pruner does NOT exist in our code today, BUT every
low-level primitive it needs already exists and is already wired.** The gap is
narrow and purely *algorithmic*: how `final_kept_ids` is computed.

- Our Stage 2 is **merge-only by construction**. Its design invariant is "expert
  weights are *not silently dropped*" (`grouping.py:15,87`), enforced by
  `_promote_orphans` (`grouping.py:77-103`) which promotes any non-merged
  non-centroid to a **singleton kept centroid**. So no expert is ever actually
  removed — the bottom-saliency experts are merged into centroids or kept as
  singletons.
- The advertised "REAP-exact / pure-prune" config
  (`configs/qwen36_35b_a3b_reap_exact.yaml`) achieves "pure-prune" by setting
  `skip_merge_percentile: 0.0` (mask ALL cost entries → solver assigns nothing).
  **This does not drop experts** — it makes every non-centroid an orphan, which
  `_promote_orphans` then KEEPS as a singleton, and the zero-assignment also
  trips the `c_fail` cost-bump loop (`orchestrator.py:438-446, 457-491`) that
  *raises* `effective_target` toward `n_experts`. Net effect: it keeps ~all
  experts, the opposite of a 35% prune. "Pure-prune" in that config name refers
  to *no-SVD, no-distill, no-heal pipeline-stage selection* (stages 1-2 only →
  stage6alt), **not** a real drop algorithm.
- Therefore: build a NEW Stage-2 plugin that computes `final_kept_ids` as the
  **top-K experts by REAP score** (drop the bottom `n_prune`) and feeds the
  EXISTING `bank.select` + `_resize_router_for_kept_experts` path, bypassing the
  cost/solver/merge/bump/heal machinery entirely.

**This is faithful to upstream**, which also does a pure structural drop with no
post-drop router rescale (see §1).

---

## 1. Upstream algorithm (READ from their .py — file:line)

Both upstream entrypoints (`prune.py:main`, `layerwise_prune.py:main`) call the
**same** `prune()` in `src/reap/prune.py` — confirmed
`layerwise_prune.py:52  from reap.prune import prune as prune_model`.

### 1.1 Saliency (REAP Eq. 9)
`src/reap/pruning_metrics.py:172-211`:
```
routing_weights = F.softmax(router_logits, dim=1, ...)               # :172
if renormalize_router_weights and selected.numel() > 0:              # :175
    topk_weights = gather(routing_weights, 1, selected_experts)
    routing_weights = routing_weights / topk_weights.sum(-1, keepdim) # :181  top-k renorm
for i in range(num_experts):
    ean_norm = ||selected_activations||_2                            # :193
    reap[i] = (ean_norm * active_router_weights).mean()              # :198  <-- Eq.9 S_j
layer_state["reap"].update(reap, expert_frequency)                   # :209  running mean
```
So `S_j = mean_over_tokens( g_j(x) · ||f_j(x)||_2 )`, `g_j` = post-softmax routing
weight (top-k renormalized when the CLI flag is on; default ON —
`src/reap/args.py:141-145`).

### 1.2 Drop loop (the actual pruning)
`src/reap/prune.py:82-145`:
```
saliency_data = observer_data[layer]["reap"]                          # :95
_, experts_to_prune = torch.topk(saliency_data, n_experts_to_prune, largest=False)  # :101
retained_expert_indicies = [i for i in range(num_experts) if i not in experts_to_prune]  # :105

# fused path (Llama-4 style; Qwen3 uses the unfused path, see below):
moe.experts.gate_up_proj.data = moe.experts.gate_up_proj[retained]   # :137
moe.experts.down_proj.data    = moe.experts.down_proj[retained]      # :140
moe.num_experts               = len(retained)                        # :141
moe.router.weight.data        = moe.router.weight.data[retained]     # :142
moe.router.out_features       = len(retained)                        # :143
moe.router.num_experts        = len(retained)                        # :145
```
Unfused path (`prune.py:110-134`): drops `ModuleList` entries, slices
`router.weight.data[retained]`, sets `router.out_features`.

### 1.3 Router rescale after drop — THERE IS NONE
**Critical fidelity finding (quoted from their code):** upstream `prune()`
performs **NO renormalization / rescale of the surviving router rows** after the
drop. It only *slices* `router.weight.data[retained]`. The
`renormalize_router_weights` flag (`args.py:141`) is purely an **observer-time
saliency-weighting** knob (`pruning_metrics.py:175-184`), NOT a post-prune router
fix-up.

The implicit "router rescale" the task brief refers to is supplied by the
**model's own forward** when `config.norm_topk_prob=True`: after experts are
dropped and `router.weight` is sliced, the model re-softmaxes the surviving
logits and renormalizes the top-k weights at inference time. Upstream relies on
this; it does not bake a rescale into the weights. Their e2e test
(`tests/test_pruning_e2e.py:210-218`) asserts **only shapes** (expert count +
`router.out_features`), never weight values — confirming "drop + slice, no
weight rescale".

### 1.4 Target → n_experts_to_prune (computed ONCE, globally)
`src/reap/prune.py:250-263`:
```
n_experts_to_prune = prune_args.n_experts_to_prune                    # :250
if n_experts_to_prune is None:                                        # :251
    total_experts = len(observer_data[layer0]["expert_frequency"])    # :258 (layer 0 only)
    n_experts_to_prune = int(total_experts * compression_ratio)       # :261
```
**Confirmed (file:line):** `n_experts_to_prune` is a single scalar computed ONCE
before the per-layer loop, from layer 0's expert count (`prune.py:258`). It is then
passed unchanged into the loop and applied at every layer via
`torch.topk(saliency, n_experts_to_prune, largest=False)` (`prune.py:101-103`). So
upstream drops the **same ABSOLUTE count** of experts in every layer — NOT a
per-layer fraction re-evaluated against each layer's own expert count. (For the
homogeneous Qwen3.x stack every layer has the same `num_experts`, so absolute-count
and fixed-fraction coincide; we still match upstream by computing one global
`n_prune` from `prune_fraction × n_experts(layer 0)` and reusing it — see §3.3.)

### 1.5 Qwen3 model attrs (our target family)
`src/reap/model_util.py:8-17` — `Qwen3MoeForCausalLM`: `fused=False`,
`router="gate"`, `num_experts="num_experts"`. Upstream treats Qwen3 as unfused
ModuleList. **Our** Qwen3.5/3.6 stores experts in the **fused**
`Qwen3_5MoeExperts` (`gate_up_proj` / `down_proj` stacked tensors —
`utils/model_io.py:10,260-275`). So our pruner uses the fused-slice primitive
(`ExpertMatrixBank.select`), which is the exact analog of upstream's fused path
(`prune.py:137-140`).

---

## 2. Exact DELTA — what we have vs what a faithful pruner needs

| Capability | Upstream | Our code today | Faithful-pruner need |
|---|---|---|---|
| REAP Eq.9 saliency | `pruning_metrics.py:198` | `ReapScoringPlugin` (`plugins/reap_scoring.py`) — faithful, top-k-renorm `g_j` (deviation D-reap-routing-weight, matches upstream default) | **reuse as-is** |
| top-K-by-score selection | `prune.py:101` | only via `select_centroids_by_reap` (centroid candidacy, feeds merge) | **NEW: drop bottom n_prune** |
| fused expert-tensor slice | `prune.py:137-140` | `ExpertMatrixBank.select(kept_ids)` (`model_io.py:418`) | **reuse as-is** |
| router row slice (no rescale) | `prune.py:126,142` | `_resize_router_for_kept_experts` (`merging.py:421-437`) — slices rows, updates `num_experts`/`top_k`, **no rescale** (matches upstream) | **reuse as-is** |
| wire select+resize into spine | `prune.py` loop | `LayerMergePlugin.post_merge` (`layer_merge.py:610-629`) calls both off `final_kept_ids` | **NEW: alternative `final_kept_ids` producer** |
| merge/solve/cost/heal | n/a (no merge) | full machinery | **MUST be bypassed** in faithful mode |
| target → n_prune | `prune.py:261` (`int(total_experts*compression_ratio)`) | `ctx.get("target")` is the **Stage-1 GRAPE-allocated** `per_layer_target_experts` (NOT a solver output — `budget/solver.py:113-119`), driven by the SVD-aware reduction config, not a 35% expert drop | **BYPASS `target`; compute `n_prune` from explicit `prune_fraction` (§3.3)** |

**Net delta = ONE new plugin + ONE config mode flag** that:
1. computes `final_kept_ids = top-(n_experts - n_prune) experts by REAP score`
   (∪ protected, which are never dropped — matches our existing protected
   invariant), and
2. short-circuits the cost/solver/merge/heal slots so the orchestrator goes
   straight from scoring to `bank.select` + router resize.

---

## 3. New plugin design

### 3.1 File + class
`max_quality/src/moe_compress/stage2/plugins/reap_prune.py`
```
class ReapPrunePlugin:
    name = "reap_prune"
    config_key = "stage2_reap_ream"          # same section; gated by a new sub-flag
    reads  = ("layer_ref", "scores", "freq", "protected", "n_experts", "partial_dir")
    writes = ("final_kept_ids", "grouped", "ream_centroid_ids", "pruned_expert_ids",
              "distill_state", "heal_state")
    # Owns these phases (wiring (a), §3.7): compute_assignment, merge (no-op),
    # post_merge (the drop), write_artifacts (own payload + sentinel .pt).
    def is_enabled(self, config) -> bool:
        return config["stage2_reap_ream"].get("prune_mode", "merge") == "faithful_prune"
```
(Holds the run-scope `merge_map` dict it mutates in `write_artifacts`, mirroring
`LayerMergePlugin`'s ownership of that scratchpad.)

### 3.2 Where it slots in (registry order)
Registered in `orchestrator.py` (the plugin-list builder around lines 1344-1416),
**after** `ReapScoringPlugin()` (so `scores`/`freq` are published) and **in place
of / before** `LayerMergePlugin`. Because `is_enabled` is mutually exclusive with
the merge path, the clean wiring is:

- When `prune_mode == "faithful_prune"`: register `ReapPrunePlugin` and let
  `registry.enabled(config)` **drop** the cost plugins, all solvers,
  skip-merge-floor, the two refiners, ExpertDistill, MergeHeal, and RegMean
  (they already gate off via their own `is_enabled`/config defaults — verify each
  in §6). **`LayerMergePlugin` MUST be dropped in faithful mode** (wiring (a),
  adopted — see §3.5 and §3.7): its `write_artifacts` `ctx.get()`s ~12 bump-loop
  slots (`layer_merge.py:658-668`: `n_experts`, `n_protected`, `assigned_cost`,
  `n_assigned`, `c_fail`, `em_rounds_done`, `effective_cost_alignment`,
  `effective_cost_asymmetric`, `capacity_util_value`, `effective_target`,
  `mean_assigned_cost`, plus `grouped`/`freq`/`ream_centroid_ids`) that are written
  ONLY in the tail of `_run_assignment` (`orchestrator.py:604-626`) — which the
  faithful early-branch skips. Since `ctx.get` raises `KeyError` on a missing slot
  (`context.py:91`), reusing `LayerMergePlugin.write_artifacts` would crash.
  `ReapPrunePlugin` therefore owns `merge`/`post_merge`/`write_artifacts`.
- When `prune_mode == "merge"` (default): byte-identical to today (plugin not
  registered / `is_enabled=False`).

**Exact call site (CONFIRMED file:line — finding #4):** the bump loop is NOT a
registry slot. The per-layer loop body (`orchestrator.py:~1538-1568`) does:
```
target = per_layer_target[layer_ref.layer_idx]   # :1545  (budget, NOT 35% — see §3.3)
ctx.set("target", target)                         # :1554
walk_phases(_STAGE2_PRE_ASSIGN_PHASES, plugins, ctx)   # :1566  on_layer_setup/on_profile/on_score
_run_assignment(plugins, ctx)                          # :1567  <-- the inline bump-loop driver
walk_phases(_STAGE2_POST_ASSIGN_PHASES, plugins, ctx)  # :1568  pre_merge_snapshot/merge/post_merge/write_artifacts/...
```
The faithful early-branch is keyed to `prune_mode == "faithful_prune"` and:
1. lets `_STAGE2_PRE_ASSIGN_PHASES` run (it publishes `scores`/`freq` via
   `on_score`; `_STAGE2_PRE_ASSIGN_PHASES` = `("on_layer_setup","on_profile",
   "on_score")`, `orchestrator.py:199-203`),
2. **SKIPS** `_run_assignment(plugins, ctx)` (`:1567`) entirely (no cost matrix,
   no solver, no bump loop, no `_promote_orphans`),
3. lets `_STAGE2_POST_ASSIGN_PHASES` run (`:1568`,
   = `("pre_merge_snapshot","merge","post_merge","write_artifacts","on_post_merge",
   "on_layer_teardown")`, `orchestrator.py:204-214`) — but with `LayerMergePlugin`
   absent, the `merge`/`post_merge`/`write_artifacts` phases dispatch to
   `ReapPrunePlugin` instead (§3.7).

   Concretely: instead of skipping `_run_assignment` with an `if`, the cleanest
   implementation is to compute `final_kept_ids` in a NEW `ReapPrunePlugin`
   `compute_assignment` hook and run the standard
   `walk_phases(_STAGE2_LAYER_PHASES, plugins, ctx)` 10-phase schedule
   (`orchestrator.py:219-221`) for faithful layers — `_run_assignment` is only
   invoked on the split schedule (`:1566-1568`), so routing faithful layers
   through `walk_phases(_STAGE2_LAYER_PHASES, ...)` (which dispatches the plain
   `compute_assignment` phase, never `_run_assignment`) is the surgical change.
   Decide in review between (i) an `if prune_mode==faithful` branch around
   `:1566-1568` that calls `walk_phases(_STAGE2_LAYER_PHASES, plugins, ctx)`, vs
   (ii) the explicit skip-`_run_assignment` form. Both pin to the same line range.

### 3.3 Phase hooks
The orchestrator's per-layer walk has the bump loop + grouping inline in
`orchestrator.py` (NOT a plugin slot), then dispatches `merge` / `post_merge` /
`write_artifacts` to `LayerMergePlugin`. The cleanest faithful path:

- **`compute_assignment` hook (faithful)**: `ReapPrunePlugin.compute_assignment`
  computes `final_kept_ids` WITHOUT touching the bump loop or grouping.
  **CRITICAL (finding #3) — do NOT use `ctx.get("target")`.** That slot
  (`orchestrator.py:249`, set at `:1554` from `per_layer_target[layer_idx]`,
  `:1545`) traces to `stage1_budgets.json["per_layer_target_experts"]`
  (`orchestrator.py:747-748`). Provenance (nit C, CONFIRMED `budget/solver.py:113-119`):
  `per_layer_target_experts` is **NOT a `solve()` output** — the docstring says so
  verbatim ("``per_layer_target_experts`` (N'_l in the spec) is **not** a solver
  output"). The solver yields only the global `global_expert_budget`
  (`solver.py:124`); **GRAPE in Stage 1** distributes it non-uniformly across layers
  (activation-aware CKA similarity, subject to `min_experts_per_layer`,
  `solver.py:116-119`). So the per-layer value is a GRAPE allocation of the overall
  reduction target (`total_reduction_ratio` 0.30 / `expert_prune_ratio` /
  `svd_rank_ratio`, `solver.py:121-123`), NOT a clean 35%-of-experts drop. Using it
  would prune "whatever GRAPE allocated for the SVD-aware param budget", silently
  ignoring §4's `prune_fraction`.

  Faithful mode instead computes `n_prune` from the explicit `prune_fraction`
  (§4), mirroring upstream's direct `int(total_experts * compression_ratio)`
  (`prune.py:261`) — computed ONCE from layer 0 and reused per layer (finding #5):
  ```
  # n_prune is a SINGLE global scalar, computed once before the layer loop:
  n_experts0 = n_experts_of_layer0          # ctx.get("n_experts") on first MoE layer
  n_prune    = int(n_experts0 * prune_fraction)   # == upstream prune.py:261
  # per layer (n_experts is homogeneous in Qwen3.x, so this holds every layer):
  faithful_target = n_experts - n_prune     # kept count; BYPASSES ctx.get("target")
  order    = np.argsort(-scores)            # descending saliency (ties: stable → low idx first)
  kept     = [e for e in order if e not in protected][: (faithful_target - len(protected))]
  final_kept_ids = sorted(set(protected) | set(kept))
  grouped  = {e: [e] for e in final_kept_ids}     # singleton groups → no merge math
  ream_centroid_ids = [e for e in final_kept_ids if e not in protected]
  pruned_expert_ids = sorted(set(range(n_experts)) - set(final_kept_ids))  # §10 item 8 / Q4
  ```
  This mirrors upstream `topk(saliency, n_prune, largest=False)` exactly: dropping
  the bottom `n_prune` ≡ keeping the top `faithful_target`. The `protected=∅` case
  (upstream default, finding #6) reduces to a pure `topk(largest=False)` —
  byte-matched in the §6 test. **Resolves open Q1.** Reconciled with §3.6/§4: the
  ONE source of truth for `n_prune` is `prune_fraction`, never the budget `target`.

- **`merge` slot (wiring (a), ADOPTED)**: `LayerMergePlugin` is **dropped** in
  faithful mode (it cannot run — see §3.2 and §3.7). `ReapPrunePlugin.merge` is a
  NO-OP that just sets the defaults the standard schedule expects downstream:
  `ctx.set("distill_state", None)` and returns. No `_merge_experts_inplace`, no
  covariance touch.

- **`post_merge` slot** (the drop): reuse the SAME primitives the existing block
  uses (`layer_merge.py:624-630`), now living in `ReapPrunePlugin.post_merge`:
  ```
  layer_ref = ctx.get("layer_ref")
  final_kept_ids = list(ctx.get("final_kept_ids"))   # published by compute_assignment
  banks = build_banks(layer_ref)
  for bank in banks.values():
      bank.select(final_kept_ids)
  _resize_router_for_kept_experts(layer_ref, final_kept_ids)
  ctx.set("heal_state", None)
  ```
  (`final_kept_ids` is set by `ReapPrunePlugin.compute_assignment`, not here.)

### 3.4 How the drop touches the fused tensors + router (concrete)
- **Experts**: `build_banks(layer_ref)` returns gate_proj/up_proj/down_proj banks
  over the fused `gate_up_proj`/`down_proj`. `bank.select(final_kept_ids)`
  `index_select`s the expert axis (`model_io.py:418-477`) — the fused analog of
  upstream `prune.py:137-140`.
- **Router**: `_resize_router_for_kept_experts(layer_ref, final_kept_ids)`
  (`merging.py:421-437`) does `router.weight = index_select(0, kept)`, updates
  `router.num_experts` and clamps `top_k`, and sets `mlp.num_experts`. This is the
  exact analog of upstream `prune.py:126,142-145`. **No rescale** — faithful.

### 3.5 Consuming the `reap_scores` sidecar
`ReapScoringPlugin.on_score` already publishes `scores`/`freq`, and
`Stage2ReapScoresCacheProvider` (`orchestrator.py:1344`) hydrates them from the
`--capture-reap-scores` sidecar on cache-hit (short-circuits via
`ctx.has("scores")`). The faithful pruner reads `ctx.get("scores")` — **same
contract, no new sidecar plumbing**. (The faithful pruner does NOT need
covariance/imatrix; those can be skipped in faithful mode for speed — note in
§7.)

### 3.6 Reaching the 35% target (RESOLVED — explicit `prune_fraction`, NOT the budget)
**Q1 is resolved (finding #3): faithful mode BYPASSES `ctx.get("target")`.** The
brief's "35%" is a per-layer **expert** drop fraction (upstream `compression_ratio`
semantics, `prune.py:261`). The orchestrator's `target` slot is a different number
— the **Stage-1 GRAPE-allocated** per-layer expert budget
(`per_layer_target_experts`). Provenance (nit C): `budget/solver.py:113-119`
explicitly states this is NOT a `solve()` output; the solver emits only the global
`global_expert_budget` (`solver.py:124`) and **GRAPE** distributes it per-layer
(CKA-weighted, `min_experts_per_layer` floor). It is a function of the overall
reduction config, NOT a clean 35%-of-experts drop. They are not the same quantity
and must not be conflated.

So the ONE source of truth for the drop count is the explicit `prune_fraction`
under `stage2_reap_ream` (§4):
```
n_prune  = int(n_experts(layer 0) * prune_fraction)   # == upstream prune.py:261
kept     = n_experts - n_prune                         # per layer (homogeneous stack)
```
There is **no bump loop** in faithful mode (upstream has none). The orchestrator's
budget machinery (`stage1_budgets.json`, `_run_assignment`, `effective_target`
bumping) is entirely off the faithful path. This reconciles §3.3 (the
implementation) with §4 (the config knob): both refer to the same
`prune_fraction`, never to the budget `target`.

### 3.7 `ReapPrunePlugin` OWNS `write_artifacts` (finding #2 — wiring (a))
`LayerMergePlugin.write_artifacts` cannot be reused: it `ctx.get()`s ~12 slots that
only `_run_assignment`'s tail writes (`layer_merge.py:658-668` ←
`orchestrator.py:604-626`), and `ctx.get` raises `KeyError` on a missing slot
(`context.py:91`). The faithful early-branch never runs `_run_assignment`, so those
slots are unset → crash. `LayerMergePlugin` is dropped (§3.2); `ReapPrunePlugin`
provides its own `write_artifacts` that emits ONLY the fields `resume.py` reads.

**Exact payload `ReapPrunePlugin.write_artifacts` writes** (enumerated against
`resume.py:139-263` and the writer `shared_io.py:_write_merge_json:168-199`). The
merge JSON (`merge_{idx}.json`) MUST contain, with these exact shapes:

| key | value (faithful) | resume.py read / assert |
|---|---|---|
| `format_version` | `2` (literal int) | `:140-150` — must equal 2 or RuntimeError |
| `final_kept_ids` | `list[int]` = the kept set | `:153-154` |
| `grouped` | `{str(e): [e]}` for each kept `e` (singletons) | `:169` — read as-is (no multi-member assumption; singletons OK) |
| `freq` | `{str(e): int}` for ALL **original** experts `range(n_experts)` | `:170-176` — asserts keys == `range(len(freq))` (contiguous); also `n_pre_merge=len(freq)` must equal `ref.num_routed_experts` (`:195-202`) ⇒ freq MUST cover the FULL pre-prune expert set, not just kept |
| `merge_map_layer` | `{str(new_idx): [orig_eid]}` for `enumerate(final_kept_ids)` (singleton, NON-EMPTY member list) | `:177-189` — asserts keys contiguous `range(len)` AND no empty member lists (singleton `[eid]` passes) |
| `mean_cost_per_pair` | `None` (no merges → no cost) | `:255` `data.get(...)` — None tolerated |
| `assignment_solver_used` | `"none"` (or `"greedy"` default) | `:258` `data.get(..., "greedy")` — any string |
| `cost_alignment_used` | `"pre"` (neutral default) | `:259` `data.get(..., "pre")` |
| `em_rounds_completed` | `0` | `:260` `int(data.get(..., 0))` |
| `distill_state` | `None` | `:261` `data.get(...)` — None tolerated |
| `heal_state` | `None` | `:262` `data.get(...)` — None tolerated; also drives `has_heal_weights_file` only when `heal_enabled` (off) |
| `pruned_expert_ids` | `list[int]` = `range(n_experts) \ final_kept_ids` | NEW additive field (Q4 / §10 item 8); `resume.py` ignores unknown keys — safe |

Notes that make this correct:
- **`freq` covers the full original expert set** (not just kept). This is required:
  `resume.py:195` derives `n_pre_merge = len(freq)` and `:196-202` asserts it equals
  the model's `num_routed_experts` (the pre-prune count). `ReapScoringPlugin`
  already publishes `freq` over all experts via `on_score`; pass it through
  unchanged.
- **`grouped` and `merge_map_layer` are singletons** but with NON-EMPTY member
  lists (`[eid]`), satisfying the `:185-189` "no empty member lists" guard.
- **No bump-loop fields are written** (`n_protected`, `c_fail`, `effective_target`,
  etc.) — `resume.py` never reads them; they existed only for forensic logging on
  the merge path.
- `write_artifacts` calls `_write_merge_json(...)` (`shared_io.py:129`) with the
  neutral defaults above, AND write the sentinel `.pt` (§5, finding #1). It must
  NOT call `_remap_covariance_for_layer` / `_snapshot_cov_layer` /
  `_snapshot_neuron_means_layer` (no covariance exists in faithful mode).
- **`pruned_expert_ids` requires a `_write_merge_json` EDIT (blocker B).**
  `_write_merge_json` (`shared_io.py:129-199`) currently has NO `pruned_expert_ids`
  parameter and never serializes it, so the §3.7 payload table CANNOT be satisfied
  by reusing it unchanged. The fix is an **additive optional kwarg**, mirroring the
  existing `stage2_run_id` omitted-when-None pattern (CONFIRMED `shared_io.py:194-195`:
  `if stage2_run_id is not None: payload["stage2_run_id"] = stage2_run_id`):
  ```
  def _write_merge_json(..., stage2_run_id=None, pruned_expert_ids=None):  # signature :129-144
      payload = { ... }                       # format_version stays 2
      if stage2_run_id is not None:           # existing, :194-195
          payload["stage2_run_id"] = stage2_run_id
      if pruned_expert_ids is not None:       # NEW — same omit-when-None contract
          payload["pruned_expert_ids"] = list(pruned_expert_ids)
  ```
  Because the key is OMITTED when None, the merge path (which passes None) keeps the
  merge JSON **byte-identical** — every existing stage2 golden / resume round-trip
  test stays green. `resume.py` ignores unknown keys, so reading is safe. This
  makes `_write_merge_json` an **EDIT**, not a reuse-unchanged (corrected in §8).

---

## 4. Config wiring — faithful-REAP mode (pure-prune, no merge/SVD)

New config `configs/qwen36_35b_a3b_reap_faithful.yaml`, derived from
`qwen36_35b_a3b_reap_exact.yaml`, changing:
```yaml
stage2_reap_ream:
  prune_mode: faithful_prune        # NEW flag (default "merge" elsewhere)
  prune_fraction: 0.35              # NEW: per-layer expert drop fraction (upstream compression_ratio)
  renormalize_router_weights: true  # observer-time saliency g_j renorm (matches upstream default)
  # merge/solver/cost/heal knobs become INERT in faithful mode (assert in code).
pipeline:
  skip_intermediate_stages: true    # stages 1+2 → stage6alt (same as reap_exact)
  evaluator: stage6alt
```
Code must **assert/refuse** the contradictory combo (faithful_prune +
non-default merge knobs, or + `skip_intermediate_stages: false` with stage3 SVD
expecting merged centroids) so a misconfig fails loud, not silently.

**Env (nit D):** set `MOE_SKIP_STAGE2_COV_SAVE=1` for the faithful run (in the
runner / launch env, or document it alongside this config). In faithful mode
`cov_acc` is empty, so the end-of-run `_save_covariance`
(`orchestrator.py:1572-1576`) would otherwise write a useless empty
`_stage2_input_covariance.pt`. Stage 3 (its only consumer) is skipped anyway, so
the guard just avoids the wasted write — harmless if omitted, see §5.

`prune_mode` default is `"merge"` everywhere → **every existing config and golden
snapshot is byte-identical** (the new plugin's `is_enabled` returns False).

---

## 5. Per-layer resume compatibility — RESUME IS BROKEN WITHOUT A FIX (finding #1)

**The naive design silently re-runs every layer on resume.** Confirmed chain
(file:line):
1. `resume.py:135` — `if not (merge_path.exists() and cov_path.exists()): continue`
   ⇒ a layer is only treated as completed when BOTH `merge_{idx}.json` AND
   `layer_{idx}.pt` exist. `:136-137` logs "found merge JSON but missing
   covariance .pt; re-running layer".
2. `_snapshot_cov_layer` (`shared_io.py:75-79`) early-returns WITHOUT writing the
   `.pt` when there are no covariance entries for the layer — and faithful mode
   collects NO covariance. So `layer_{idx}.pt` is never written.
3. Result: `merge_{idx}.json` exists but `layer_{idx}.pt` does not ⇒ `resume.py`
   skips the "completed" record and **re-runs the layer from scratch** every time.
   (Note: the `orchestrator.py:976-984` `cov_acc.load_layer_from_disk` raise is
   NOT the trigger — `load_layer_from_disk` returns `False` on an absent file
   (`activation_hooks.py:1218-1231`), it only raises on a *corrupt* payload. The
   real break is the `resume.py:135` gate, which never even reaches the replay.)
4. Secondary hazard: the orphan-cleanup at `resume.py:123-129` deletes a `.pt`
   that has no matching JSON — harmless here (we will have JSON, no `.pt`), but
   confirms the resume design assumes `.pt`-with-JSON pairing.

**FIX (option (a), ADOPTED — justification below): faithful mode writes an empty
sentinel `layer_{idx}.pt`.** `ReapPrunePlugin.write_artifacts` writes the merge
JSON (§3.7) and ALSO writes a minimal valid `.pt` payload so the
`resume.py:135` `cov_path.exists()` gate passes and the layer is recognized as
completed. The sentinel payload mirrors `_snapshot_cov_layer`'s schema
(`shared_io.py:80-88`) with empty maps so any future reader stays schema-valid:
```
payload = {"format_version": 1, "covariance": {}, "tokens": {}}
# write via the same atomic tmp+durable_rename used by _snapshot_cov_layer
```
On replay, `orchestrator.py:976-977` `cov_acc.load_layer_from_disk` reads the
sentinel: `_read_layer_payload` returns the empty-cov payload, `_accumulate_payload`
accumulates nothing (no keys), returns `True`. The replay block
(`orchestrator.py:962-965`) reconstructs banks from `record.final_kept_ids` and
calls `bank.select` + `_resize_router_for_kept_experts` — exactly the right
behavior for a pure prune. **Confirm in review** that `_accumulate_payload` tolerates
empty `covariance`/`tokens` dicts (it iterates the maps; empty ⇒ no-op).

**Replay also calls `_merge_experts_inplace` — no-op for faithful, but ONLY by
construction (nit A).** BEFORE the `bank.select` above, the replay path
unconditionally runs `_merge_experts_inplace(ref, record.grouped, record.freq, …)`
(`orchestrator.py:951-956`). For a faithful layer `record.grouped` is all
singletons, and `_merge_experts_inplace` skips any group with `len(members) <= 1`
(CONFIRMED `merging.py:161-163`: `for centroid, members in grouped.items(): if
len(members) <= 1: continue`). So it touches zero weights — the per-expert tensors
reach `bank.select` byte-identical to a fresh run. This is the load-bearing
invariant behind the §6 test-6 (faithful resume round-trip) byte-equality
assertion: if a future change ever made `_merge_experts_inplace` mutate singleton
groups, the resumed weights would diverge silently. Test-6 must assert byte-equal
weights (not just equal `final_kept_ids`) so it locks `merging.py:162` in place.

**Why (a) over (b):** option (b) (teach `resume.py` to treat present
`merge_*.json` + no-`.pt` as a completed faithful layer) requires `resume.py` to
KNOW the run is in faithful mode, which it does not have in scope at `:135`
(it iterates files on disk, mode-agnostic), and it would weaken the orphan/torn
checks for ALL runs. (a) is a self-contained additive write in the new plugin,
keeps `resume.py` and the merge-path resume contract byte-identical, and needs zero
changes to the shared resume reader. Lower blast radius, no cross-mode coupling.

**Resume record shape (§3.7 payload) is already resume-compatible:** singleton
`grouped`/`merge_map_layer` (non-empty `[eid]` member lists) pass the
`resume.py:185-189` guards; full-`range`-keyed `freq` passes `:170-176` and the
`n_pre_merge` check `:195-202`. No multi-member assumption exists in `resume.py`.

**Covariance remap is NOT called in faithful mode:** `write_artifacts` skips
`_remap_covariance_for_layer` (`layer_merge.py:680`) and the two snapshot helpers
(no covariance collected) — see §3.7. This removes the §9-Q3 open item.

**End-of-run covariance save (nit D — harmless, but set the guard):** at run end
`orchestrator.py:1576` `_save_covariance(cov_acc, "_stage2_input_covariance.pt")`
runs unless `MOE_SKIP_STAGE2_COV_SAVE == "1"` (`:1572-1574`). In faithful mode
`cov_acc` is empty, so this writes an empty/degenerate `_stage2_input_covariance.pt`.
It is harmless — Stage 3 (its only consumer) is skipped in faithful mode
(`skip_intermediate_stages: true`, §4) — but wasteful. The faithful config/runner
should set `MOE_SKIP_STAGE2_COV_SAVE=1` (§4) so the empty file is never written.

This is exercised by the §6 faithful-mode resume round-trip test.

---

## 6. Test plan (CPU-only; tiny fixtures)

All tests run on CPU with a 2-3 layer tiny `Qwen3_5Moe`-shaped fixture (mirror
the upstream `_make_qwen3_model` config in `tests/test_pruning_e2e.py:66-75`:
hidden=16, num_experts=3-8, num_experts_per_tok=1). **No GPU.**

1. **Unit — top-K selection** (`test_reap_prune_selection.py`):
   given a hand-built `scores` vector + `protected` set + `prune_fraction`,
   assert `final_kept_ids` == expected (drop bottom-saliency, never drop
   protected, kept count == target). Mirrors upstream `topk(largest=False)`.
2. **Unit — drop applies to tensors** (`test_reap_prune_apply.py`):
   build the tiny fused model, run `bank.select` + `_resize_router_for_kept_experts`
   via the plugin's `post_merge`, assert:
   - `gate_up_proj.shape[0]` and `down_proj.shape[0]` == kept count,
   - `gate.weight.shape[0]` == kept count, `router.num_experts` == kept,
   - **router weight rows are UNCHANGED (no rescale)** for surviving experts
     (byte-equal to the original rows at the kept indices) — locks in upstream
     "drop-only, no rescale" fidelity.
3. **Integration — tiny CPU end-to-end** (`test_reap_prune_integration.py`):
   run Stage 2 in `prune_mode: faithful_prune` on the tiny model with a tiny
   calibration batch; assert post-prune model forward runs, expert count dropped
   to target on every layer, merge machinery never invoked (assert no
   covariance accumulated / no cost matrix built — e.g. via a spy/None-check on
   `cov_acc`), and merge_map has only singleton groups.
4. **Config test** (`test_reap_faithful_config.py`):
   YAML-load the new config; assert `prune_mode == "faithful_prune"`,
   `prune_fraction` present, pipeline skip/evaluator correct, and the inert-knob
   assertions hold. Mirror `test_reap_exact_config.py`.
5. **Golden snapshot** (`test_reap_prune_golden.py`):
   deterministic tiny model + fixed `scores` → assert `final_kept_ids` per layer
   and the sliced `gate.weight` / `gate_up_proj` byte-hash match a committed
   golden (regenerate-on-purpose pattern used by the repo's other stage2 golden
   tests). Also a **no-regression golden**: with `prune_mode` absent/`"merge"`,
   an existing stage2 golden is byte-identical (proves the new plugin is inert by
   default).
6. **NEW — faithful-mode resume round-trip** (`test_reap_prune_resume.py`)
   (finding #1 / #7a): run Stage 2 faithful on the tiny model with a `partial_dir`,
   killing/stopping after writing layer 0's artifacts; assert:
   - `merge_0.json` AND the sentinel `layer_0.pt` both exist (the §5 fix);
   - a second Stage-2 invocation with the same `partial_dir` RECOGNIZES layer 0 as
     completed (does NOT re-run it — e.g. spy that the faithful drop is not
     re-applied / `resume.py` returns a record for layer 0);
   - `cov_acc.load_layer_from_disk(0, partial_dir)` returns `True` and accumulates
     no covariance (empty sentinel);
   - the resumed model's per-layer expert count == fresh-run count, with **byte-equal
     surviving expert tensors AND `final_kept_ids`** (not just equal counts). The
     byte-equality of weights is the assertion that locks `merging.py:162` (nit A):
     replay runs `_merge_experts_inplace(record.grouped,…)` (`orchestrator.py:951`)
     before `bank.select`, and it is a no-op for faithful singletons ONLY because
     `merging.py:161-163` skips `len(members) <= 1` groups. If that skip ever
     regressed, this assertion fails.
   This is the regression lock for the "resume silently re-runs every layer" bug.
7. **NEW — protected=∅ upstream-formula byte-match** (`test_reap_prune_upstream_formula.py`)
   (finding #6 / #7b): with `protected = ∅` (upstream default, both
   `perserve_super_experts`/`perserve_outliers` `default=False`, `args.py:515,523`),
   hand-compute `experts_to_prune = torch.topk(scores, n_prune, largest=False)` and
   `retained = [i for i in range(n) if i not in experts_to_prune]` EXACTLY as
   upstream `prune.py:101-106`, then assert `ReapPrunePlugin.compute_assignment`'s
   `final_kept_ids == sorted(retained)` — byte-identical selection. Pin tie-break
   to match `torch.topk` ordering so the byte-match is deterministic. This proves
   that, absent our protected divergence, our selection equals upstream's formula.

**Keep the no-rescale assertion** (test 2: surviving router rows byte-equal to the
original rows at the kept indices) — it locks upstream "drop-only, no rescale"
fidelity (`prune.py:142`, no post-drop renorm).

---

## 7. Deviations from upstream (stated explicitly)

- **D-fp-fused-storage:** upstream Qwen3 is unfused ModuleList (`model_util.py:14`);
  we operate on the fused `Qwen3_5MoeExperts` stacked tensors via
  `ExpertMatrixBank.select`. Mathematically identical drop (same kept rows), just
  the storage layout our model family uses (upstream's own fused path,
  `prune.py:137-140`, does the same on Llama-4).
- **D-reap-routing-weight (inherited):** `g_j` is the top-k-renormalized routing
  weight (matches upstream CLI default `renormalize_router_weights=True`,
  `args.py:141`). Documented already in `reap_scoring.py:24-43`.
- **D-prune-no-imatrix/cov:** faithful mode skips covariance/imatrix collection
  (upstream `prune()` collects none for the drop). Faster; no quality impact on a
  pure prune. Stage 3+ are skipped in faithful mode anyway.
- **D-prune-fraction-source:** we expose an explicit `prune_fraction` rather than
  deriving the expert-drop count from the param-split budget
  (`total_reduction_ratio` × `expert_svd_ratio`), to match upstream's direct
  `compression_ratio` semantics (`prune.py:261`). RESOLVED — see §3.6 (we bypass
  `ctx.get("target")` entirely).
- **D-protected-experts — GENUINE DIVERGENCE (not "same intent"):** we always
  hard-exclude protected (super/shared) experts from the drop candidate list (our
  Stage-1 blacklist invariant). Upstream's super-expert preservation is **OFF by
  default**: it only pins super-expert saliency to `+inf` when
  `prune_args.perserve_super_experts or prune_args.perserve_outliers`
  (`prune.py:63`), and BOTH flags `default=False` (`args.py:514-515, 522-523`). So
  the upstream **default** run has NO protected set — it drops the bottom
  `n_experts_to_prune` by pure `torch.topk(saliency, n, largest=False)`
  (`prune.py:101-103`) with no exclusions. Our pruner's unconditional protected
  exclusion is therefore a real behavioral divergence from the upstream default,
  not a re-implementation of an always-on feature. We accept it because our
  protected set is a Stage-1 contract (shared/super experts must never be dropped),
  but we document it as a divergence and PROVE the underlying selection matches
  upstream by the `protected=∅` byte-match test (§6 test 7): with no protected
  experts our `compute_assignment` output is byte-identical to a hand-computed
  upstream `topk(largest=False)` + `retained` complement. (Upstream's `+inf` pin
  and our hard-exclude are *equivalent* mechanisms, but they are equivalent only
  when the upstream flag is ON — which it is not by default.)

---

## 8. Files touched (when implemented — NOT in this commit)

- NEW `max_quality/src/moe_compress/stage2/plugins/reap_prune.py` — owns
  `compute_assignment` (top-K selection, BYPASSES `ctx.get("target")`), `merge`
  (no-op + `distill_state=None`), `post_merge` (the drop), and `write_artifacts`
  (own payload §3.7 + sentinel `.pt` §5).
- EDIT `max_quality/src/moe_compress/stage2/orchestrator.py` — (1) register
  `ReapPrunePlugin` after `ReapScoringPlugin` and drop `LayerMergePlugin` +
  cost/solver/refine/distill/heal/regmean in faithful mode; (2) route faithful
  layers through `walk_phases(_STAGE2_LAYER_PHASES, plugins, ctx)` instead of the
  `_run_assignment` split at `:1566-1568` (so the plain `compute_assignment` phase
  dispatches to `ReapPrunePlugin`, never the bump loop). Pin to `:1538-1568`.
- **EDIT `max_quality/src/moe_compress/stage2/shared_io.py`** — add the additive
  optional `pruned_expert_ids=None` kwarg to `_write_merge_json` (`:129-199`),
  omitted-when-None so the merge path JSON stays byte-identical (blocker B; mirrors
  the `stage2_run_id` pattern at `:194-195`).
- NEW `max_quality/configs/qwen36_35b_a3b_reap_faithful.yaml`
- NEW tests (§6, **7 files** — incl. the new resume round-trip and the
  protected=∅ upstream-formula byte-match).
- **NOT touched:** `resume.py` (the §5 fix lives entirely in
  `ReapPrunePlugin.write_artifacts` via the sentinel `.pt` — option (a) keeps the
  shared resume reader byte-identical). `shared_io.py:_snapshot_cov_layer` /
  `_remap_covariance_for_layer` are simply NOT CALLED in faithful mode.
- Reused unchanged: `plugins/reap_scoring.py`, `utils/model_io.py`
  (`ExpertMatrixBank.select`), `stage2/merging.py`
  (`_resize_router_for_kept_experts`), and the `shared_io.py` atomic-write helpers
  (for the sentinel `.pt`). NOTE: `_write_merge_json` is an EDIT (see above), not a
  reuse-unchanged — the `pruned_expert_ids` field requires the new kwarg.

---

## 9. Open questions for plan review — ALL RESOLVED (rev-2)

1. **(RESOLVED — §3.3/§3.6)** The 35% is a per-layer *expert* fraction (upstream
   `compression_ratio` semantics). Faithful mode uses an explicit `prune_fraction`
   and **BYPASSES `ctx.get("target")`** — which is the Stage-1 GRAPE-allocated
   per-layer budget, not a 35% expert drop (confirmed `orchestrator.py:249,747-748`;
   provenance corrected per nit C: `budget/solver.py:113-119` says
   `per_layer_target_experts` is NOT a solver output — GRAPE allocates it in Stage 1).
2. **(RESOLVED — §3.2/§3.5/§3.7)** Wiring **(a) ADOPTED**: `ReapPrunePlugin` owns
   `merge`/`post_merge`/`write_artifacts`; `LayerMergePlugin` is dropped in
   faithful mode. Required, not just preferred — `LayerMergePlugin.write_artifacts`
   `ctx.get()`s ~12 unset bump-loop slots and `ctx.get` raises `KeyError`
   (`context.py:91`), so (b) would crash.
3. **(RESOLVED — §5)** `resume.py` is INCOMPATIBLE with the naive design (the
   `:135` both-files gate vs no-cov `.pt`); fixed by writing a sentinel
   `layer_{idx}.pt` (`{"format_version":1,"covariance":{},"tokens":{}}`).
   `_remap_covariance_for_layer` / `_snapshot_cov_layer` are NOT called in faithful
   mode. Confirmed `_accumulate_payload` (`activation_hooks.py:1198-`) tolerates
   empty cov/tokens dicts.
4. **(RESOLVED — adopted)** Faithful mode **DOES** emit `pruned_expert_ids` per
   layer (complement of `final_kept_ids`) in the merge JSON (§3.7 payload table) for
   downstream parity with upstream's `experts_to_prune`. Additive field; `resume.py`
   ignores unknown keys.
