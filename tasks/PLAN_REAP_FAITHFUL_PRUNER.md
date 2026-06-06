# PLAN — Faithful-to-Upstream REAP Pruner (pure expert PRUNING)

**Branch:** `feat/reap-faithful-pruner`
**Status:** PLANNING ONLY (no production code in this commit). Plan-review loop next.
**Upstream:** CerebrasResearch/reap (cloned `/tmp/reap_upstream`, depth-1, default branch `main` as of 2026-06-06). Paper: arXiv:2510.13999.

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

### 1.4 Target → n_experts_to_prune
`src/reap/prune.py:251-264`:
```
total_experts = len(observer_data[layer0]["expert_frequency"])       # :258
n_experts_to_prune = int(total_experts * compression_ratio)          # :261
```
Per-layer, uniform `n_prune` per layer (same fraction every layer).

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
| target → n_prune | `prune.py:261` | budget logic computes per-layer `target` (kept count) in `orchestrator.py` | **reuse the target arithmetic, drop the bump loop** |

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
    reads  = ("layer_ref", "scores", "freq", "protected")
    writes = ("final_kept_ids", "grouped", "ream_centroid_ids")
    def is_enabled(self, config) -> bool:
        return config["stage2_reap_ream"].get("prune_mode", "merge") == "faithful_prune"
```

### 3.2 Where it slots in (registry order)
Registered in `orchestrator.py` (the plugin-list builder around lines 1344-1416),
**after** `ReapScoringPlugin()` (so `scores`/`freq` are published) and **in place
of / before** `LayerMergePlugin`. Because `is_enabled` is mutually exclusive with
the merge path, the clean wiring is:

- When `prune_mode == "faithful_prune"`: register `ReapPrunePlugin` and let
  `registry.enabled(config)` **drop** the cost plugins, all solvers,
  skip-merge-floor, the two refiners, ExpertDistill, MergeHeal, and RegMean
  (they already gate off via their own `is_enabled`/config defaults — verify each
  in §6). `LayerMergePlugin` must also be dropped (or made to early-return) in
  faithful mode — see §3.5.
- When `prune_mode == "merge"` (default): byte-identical to today (plugin not
  registered / `is_enabled=False`).

### 3.3 Phase hooks
The orchestrator's per-layer walk has the bump loop + grouping inline in
`orchestrator.py` (NOT a plugin slot), then dispatches `merge` / `post_merge` /
`write_artifacts` to `LayerMergePlugin`. The cleanest faithful path:

- **`compute_assignment`-equivalent**: add an early branch in the orchestrator's
  per-layer body (guarded by `prune_mode == "faithful_prune"`) that SKIPS the
  entire `for _bump_attempt` loop (`orchestrator.py:320-491`) and the grouping +
  `_promote_orphans` block (`:554-578`). Instead:
  ```
  n_prune  = n_experts - target                  # target = kept count from budget
  order    = np.argsort(-scores)                  # descending saliency
  kept     = [e for e in order if e not in protected][: (target - len(protected))]
  final_kept_ids = sorted(set(protected) | set(kept))
  grouped  = {e: [e] for e in final_kept_ids}     # singleton groups → no merge math
  ream_centroid_ids = [e for e in final_kept_ids if e not in protected]
  ```
  This mirrors upstream `topk(..., largest=False)` (drop bottom) exactly:
  dropping the bottom `n_prune` ≡ keeping the top `target`.

- **`merge` slot**: in faithful mode `LayerMergePlugin.merge` must be a NO-OP
  (singleton `grouped` means `_merge_experts_inplace` has nothing to merge, but
  we should not even call it — it would touch covariance). Provide
  `ReapPrunePlugin.merge` that sets `distill_state=None` and returns, OR gate
  `LayerMergePlugin` off and let `ReapPrunePlugin` own `merge`/`post_merge`.

- **`post_merge` slot** (the drop): reuse VERBATIM the existing block
  (`layer_merge.py:624-629`):
  ```
  banks = build_banks(layer_ref)
  for bank in banks.values():
      bank.select(final_kept_ids)
  _resize_router_for_kept_experts(layer_ref, final_kept_ids)
  ctx.set("final_kept_ids", tuple(final_kept_ids)); ctx.set("heal_state", None)
  ```

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

### 3.6 Reaching the 35% target
Reuse the orchestrator's existing per-layer `target` (kept-expert count) budget
arithmetic. In faithful mode there is **no bump loop** (upstream has none): the
kept count is exactly `target` (uniform fraction per layer, matching upstream
`int(total_experts * compression_ratio)` at `prune.py:261`). Confirm in review
that our `target` derivation equals `round((1 - reduction_ratio_for_experts) *
n_experts)` and that the expert-only ratio (vs the 0.30 *total* param ratio with
`expert_svd_ratio`) is the intended 35%-of-experts knob — **open question for
plan review** (the brief says 35%; the config's `total_reduction_ratio: 0.30`
with `expert_svd_ratio: 2.0` is a *param* budget, not a clean 35%-expert drop).
The faithful pruner should accept an explicit `prune_fraction` (per-layer expert
drop fraction) under `stage2_reap_ream` to match upstream's direct
`compression_ratio` semantics, rather than inferring it from the SVD-split param
budget.

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

`prune_mode` default is `"merge"` everywhere → **every existing config and golden
snapshot is byte-identical** (the new plugin's `is_enabled` returns False).

---

## 5. Per-layer resume compatibility

The merge path writes per-layer partial JSON + `.pt` (`write_artifacts`,
`layer_merge.py:635-`) and the resume path replays `record.final_kept_ids` into
`bank.select` + `_resize_router_for_kept_experts` (`orchestrator.py:960-986`).
Faithful mode must write the **same** `final_kept_ids` record shape so resume
works unchanged:
- `ReapPrunePlugin.write_artifacts` writes `{layer_idx, final_kept_ids,
  ream_centroid_ids=final_kept_ids\protected, grouped=singletons, freq}` matching
  the fields `resume.py` reads.
- `merge_map.json` envelope: singleton groups (each kept expert maps to itself,
  no absorbed members). Downstream `merge_map.json` consumers (Stage 2.5
  merge-repair) see "no merges" — correct for a pure prune.
- **Open item for review:** confirm `resume.py` does not assume `grouped` has
  multi-member groups; it should treat singletons fine, but verify the
  `_remap_covariance_for_layer` call (`layer_merge.py:680`) is skipped or
  no-ops in faithful mode (no covariance is collected).

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
  `compression_ratio` semantics (`prune.py:261`). Open for review (§3.6).
- **D-protected-experts:** we never drop protected (super/shared) experts (our
  Stage-1 blacklist invariant). Upstream has an analogous
  `perserve_super_experts` path (`prune.py:63-80`) that pins super-expert
  saliency to `+inf` so they are never pruned — same intent, our protected set is
  applied as a hard exclude from the drop candidate list.

---

## 8. Files touched (when implemented — NOT in this commit)

- NEW `max_quality/src/moe_compress/stage2/plugins/reap_prune.py`
- EDIT `max_quality/src/moe_compress/stage2/orchestrator.py` (register plugin +
  faithful-mode early branch around the bump loop)
- NEW `max_quality/configs/qwen36_35b_a3b_reap_faithful.yaml`
- NEW tests (§6, 5 files)
- Reused unchanged: `plugins/reap_scoring.py`, `utils/model_io.py`
  (`ExpertMatrixBank.select`), `stage2/merging.py`
  (`_resize_router_for_kept_experts`).

---

## 9. Open questions for plan review

1. §3.6 — is the 35% target a per-layer *expert* fraction (upstream semantics)
   or derived from the `total_reduction_ratio`/`expert_svd_ratio` param budget?
   Plan assumes the former via an explicit `prune_fraction`.
2. §3.2/§3.5 — preferred wiring: (a) `ReapPrunePlugin` owns `merge`/`post_merge`
   and `LayerMergePlugin` is dropped in faithful mode, vs (b) keep
   `LayerMergePlugin` and feed it singleton `grouped` so its existing
   `post_merge` does the drop. (a) is cleaner (no merge code on the prune path);
   (b) reuses more. Recommend (a).
3. §5 — confirm `resume.py` + `_remap_covariance_for_layer` tolerate the
   no-covariance faithful path.
4. Should faithful mode also emit a `pruned_expert_ids` list per layer (the
   complement of `final_kept_ids`) for downstream analysis parity with upstream's
   `experts_to_prune`?
