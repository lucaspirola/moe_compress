# Plugin #14 — Cross-Plugin Sidecar Audit

**Status**: read-only audit. No code changed. Recommendations only.
**Repo**: `/home/lucas/ai/moe_compress` (main @ `9330988`).
**Date**: 2026-05-28.

## 1. Goal & Context

After the 12-plugin sweep landed, the user asked: *"Re-analyse what are the
needs of all plugins, and see if we need to collect any different side-cars
during the calibration phase to speed up the run."*

The calibration phase (driven by
`max_quality/scripts/build_self_traces_calib_vllm.py`) is the single most
expensive recurring cost in the pipeline: every model × calibration-mix
combo pays it again unless we cache its outputs. Each capture flag turns on
a teacher-side accumulator that writes a sidecar next to the JSONL; each
downstream stage's `*_cache.py` provider tries `load_*` first and falls
through to a live computation on miss.

This audit answers three questions:

1. What sidecars do we currently capture, and what consumes them?
2. Of the 12 merged plugins, which still pay an avoidable per-run cost
   that a sidecar would amortize?
3. Where (if anywhere) is it worth adding a new sidecar?

Method: read-only. Citations are file:line from this branch's HEAD. Where
wall-clock numbers come from plan docs, the source is named; where they
are my own estimate, that is called out.

---

## 2. Sidecar Inventory (current state)

All sidecars are written to `<jsonl_path.parent>/sidecars/<name>.pt`
(or per-shard subdirs) via the atomic `tmp + os.replace` contract in
`max_quality/src/moe_compress/utils/cached_calibration_signals.py:459-481`.
`SCHEMA_VERSIONS` (lines 98-116) is the central registry.

| Sidecar | Schema | Captures | Consumed by | Disk (est.) |
|---|---|---|---|---|
| `phase_b` | 1 | Stage 1 Phase-B accumulators: per-(layer, expert) `per_expert_max`, `routing_freq`, `mean_routing_weight`, `output_reservoir [n_layers, n_experts, R, hidden]`. Combined legacy payload. | Stage 1 Phase-B fast-path (now superseded by the per-signal sidecars below). | 10-15 GB (dominated by reservoir slab) |
| `reap_scores` | 1 | REAP Eq. 9 saliencies `S_j = (1/|X_j|)·Σ g_j·‖f_j‖₂` per (layer, expert) + token counts. | `Stage2ReapScoresCacheProvider` → hydrates `ctx.scores`/`ctx.freq`; `ReapScoringPlugin.on_score` early-returns on `ctx.has("scores")`. | small (~MB) |
| `per_expert_max` | 1 | Per-(layer, expert) `max(‖f_j(x)‖_∞)` + token counts. | `Stage1PerExpertMaxCacheProvider` at STEP 4.5; drops `downproj_max` from live HookSpec set. | small |
| `routing_stats` | 1 | Per-(layer, expert) `freq` int64 + `mean_weight` fp32 (zero where freq==0). | Stage 1 `Stage1RoutingStatsCacheProvider` (STEP 4.6) + Stage 2 `Stage2RoutingStatsCacheProvider`. **NO live consumer yet** — payload deposited on `ctx.routing_stats_payload` for future plugins. | small |
| `router_logits_stats` | 1 | Per-(layer, expert) sink-vs-normal router-score aggregates + per-layer sink/normal token counts + `bos_token_id`. | `Stage1RouterLogitsStatsCacheProvider` (STEP 4.7) → hydrates `SinkTokenRoutingAccumulator`; orchestrator drops `"sink_routing"` from `needed`. | small |
| `output_reservoir` | 1 | `[n_layers, n_experts, max_tokens=256, hidden] bf16` reservoir-sampled unweighted expert outputs + `valid_count` + `total_seen`. | `Stage1OutputReservoirCacheProvider` (STEP 4.8) → hydrates `ExpertOutputAccumulator`; orchestrator drops `"output_reservoir"` from `needed`. | **large** (~10-15 GB at typical configs; see `OutputReservoirPayload` docstring lines 381-385) |
| `covariance` | 2 | Dict `(layer_idx, expert_idx, matrix_name) → fp16 Σ_in[d_in, d_in]` (gate_proj/down_proj; up_proj aliases gate_proj). | Stage 3 `Stage3InputCovCacheProvider` (before `_load_stage2_covariance`); Stage 4 `Stage4InputCovCacheProvider` (`EoraInputsPlugin.load_eora_inputs` short-circuits on `ctx.has("A_cov")`). | medium (n_layers × n_experts × d²·2 bytes; e.g. 40 × 256 × 2048² × 2 ≈ ~85 GB at d_in=2048 — **only gate_proj is populated**, so effective ~half that. Spec note: Σ is finalized after all tokens; **fp16 storage**.) |
| `stage2_profile` | 3 | Plugin #12 REDO bundle. Fields: `model_hash`, `top_k`, `cov_storage_dtype`, `total_tokens_per_layer[n_layers]`, `gate_logit_profiles dict[rank→list[(off, [T_b,E] fp32)]]`, `sim_tensor[n_layers,E,E] fp64`, `neuron_act_sum/_count`, `cov_acc/_token_count` (finalized), `layer_input_reservoir list[Tensor[N, hidden] bf16]`. | `Stage2ProfileCacheProvider` (registered after `LayerMergePlugin` per the OQ-1 Option A in-place hydration pattern). Full hit → `LayerMergePlugin.on_profile` early-returns (Pattern A skip). | large (sim_tensor is `n_layers·E²·8` bytes — 40·256²·8 ≈ 2 GB; cov_acc dominates at ~40 GB fp16 like `covariance`; gate_logit_profiles a few GB; reservoir 1-15 GB) |
| `block_hidden` | 1 (per-layer shards) | Per-MoE-block post-block hidden states on a fixed N-prompt subset (default 128). Written as `[n_tokens, hidden] bf16` slabs at `sidecars/block_hidden/layer_{idx:04d}.pt`. | `Stage3BlockHiddenCacheProvider` (block-refine cache, before `walk_phases(("refine_blocks",)…)`); ctx slot `teacher_targets_cache`. Stage 2.5 `merge_repair` reader was scope-cut (see MANIFEST §"Scope-cut consumer notes"). | medium-large (128 prompts × ~512 tokens × 2048 hidden × 2 B × n_layers ≈ a few GB) |
| `router_kd_logits` | 1 (per attempt_idx .npz) | Sharded `[n_tokens] token_ids, [n_tokens, top_k] top_ids, top_logprobs`. Streaming-writer pattern. | Stage 5 Router-KD trainer teacher branch. | large (per-attempt; varies with mix size) |
| `teacher_eval` | 1 | Eval-harness teacher results dict + param counts, keyed by SHA-256 cache_key. | Stage 6 baseline evaluator. | tiny |

**Schema-version table** is the v3 bump for `stage2_profile`; everything else
is v1 except `covariance: 2` (v1→v2 forward-only; v1 never persisted).

---

## 3. Per-Plugin Consumption Analysis

### Plugin #1 — Opt C (vectorized reservoir)
File: `stage2/profiling.py:34-167`.

* **Consumes**: nothing from sidecars at runtime. `_LayerInputAccumulator`
  is populated live by a forward-pre hook on the decoder layer during the
  Stage 2 profile pass.
* **Re-computes per run**: the entire layer-input reservoir buffer
  (`max_samples=8192` tokens × hidden, bf16, per layer).
* **Sidecar coverage today**: Plugin #12's `stage2_profile` schema HAS a
  `layer_input_reservoir` field (cached_calibration_signals.py:215), and
  the reader at `stage2_profile_cache.py:282-293` will hydrate
  `layer_input_acc.buffer` on full hit — BUT the vLLM patch currently
  registers no `layer_in` callback, so production sidecars carry empty
  `(0, 0)` placeholders (see `vllm_calibration_stage2_profile.patch:474-482`
  and the runtime warning at lines 512-527 of the same file).
* **Gap**: with the placeholder in place, Plugin #1's vectorized fast-add
  still runs on every SC row when `cost_alignment="output"` because the
  reader's `reservoir_t.numel() > 0` guard at
  `stage2_profile_cache.py:291` keeps falling through to the live forward.
  Fixing this is Gap-1 below (~6.8 min/SC row, per
  `PLAN_OPT_C_vectorized_reservoir.md` line 14).

### Plugin #2 — Opt B1 (perm_cache write)
File: `stage2/plugins/output_space_cost.py:310-311`.

* **Consumes**: `perm_cache` (run-scope dict). NOT a sidecar.
* **Re-computes per run**: writes Hungarian-LAP results to the cache so
  the subsequent merge step at `_merge_experts_inplace` reuses them.
* **Sidecar coverage**: `perm_cache` is intentionally a per-run in-memory
  artifact. Persisting it would help only if the same `(layer, c_id, m_id)`
  triple is hit across runs, which is rare because the merge clustering
  (and therefore the candidate pairs) differs between strategy rows.
* **Gap**: NONE. Persisting `perm_cache` would not pay back — saving is
  ~1 min/row total (PLAN_OPT_B1 line 13), only a fraction would carry
  across runs.

### Plugin #3 — Opt B3 (argpartition hoist)
File: `stage2/plugins/output_space_cost.py:441-445`.

* **Consumes**: `cheap_cost` (in-memory).
* **Re-computes per run**: collapsing `n_NC × n_C` per-row argpartitions
  into a single 2-D call.
* **Sidecar coverage**: N/A — pure CPU-side reorganization.
* **Gap**: NONE here. (See Pattern-H follow-up in §6 below — the
  symmetric site at `ream_cost_post.py:214` still needs the same hoist.)

### Plugin #4 — Opt B4 (build_banks hoist)
File: `stage2/plugins/output_space_cost.py:447`.

* **Consumes**: `layer_ref.experts` (live model parameters).
* **Re-computes per run**: avoids re-calling `build_banks` 98k times by
  hoisting it out of the inner loop.
* **Sidecar coverage**: N/A — `build_banks` returns views over the
  model's own `nn.Parameter`s, so a disk cache would defeat the point
  (the parameters mutate in place during sequential merging).
* **Gap**: NONE.

### Plugin #5 — Opt B2 (bf16 weighted merge)
Files: `stage2/plugins/output_space_cost.py:282-324`; `stage2/permutation_align.py`.

* **Consumes**: weight tensors (live model).
* **Re-computes per run**: per-pair weighted merge of `(W_c, W_m)` in
  bf16 instead of fp32.
* **Sidecar coverage**: N/A.
* **Gap**: NONE. Open nit: the `(Dtype is hardcoded to fp32 at the SwiGLU
  call sites below.)` comment at lines 450-451 of `output_space_cost.py`
  is the stale-comment cited as a known follow-up (§6).

### Plugin #6 — `on_post_merge` hook
Files: `stage2/orchestrator.py:186-208`, framework-level.

* **Consumes**: nothing.
* **Re-computes per run**: N/A. This is a new phase in
  `_STAGE2_POST_ASSIGN_PHASES`; the per-layer cost accumulators are
  invalidated by Plugin #10 inside this phase.
* **Sidecar coverage**: N/A. The framework change is plumbing only.
* **Gap**: NONE.

### Plugin #7 — RKD-paper recipe (Row P)
File: `router_kd/plugins/rkd_paper_recipe.py:96-200`.

* **Consumes**: `config["stage5_router_kd"]`, `config["calibration"]`. No
  ctx slots, no sidecars.
* **Re-computes per run**: nothing the plugin owns; it mutates `config`
  in place with the 4 paper-recipe deltas.
* **Sidecar coverage**: N/A.
* **Gap**: NONE. (Note: the multi-epoch contract at line 177 forces
  `s5["teacher_logits_cache"] = None` because `orchestrator.py:585` raises
  if epochs>1 and the logits cache is non-None — so Row P deliberately
  pays the full teacher logit recompute on epoch 2.)

### Plugin #8 — S1_DP damage-curve DP
File: `stage1/plugins/damage_curve_dp.py:274-383`.

* **Consumes**: `D_matrices` (CKA distance matrices, `ctx.get("D_matrices")`),
  `blacklist`, `per_layer_targets`, `decomposition`.
* **Re-computes per run**: the per-layer damage curve
  (sorted off-diagonal cumsum) and the L × K × G DP knapsack solve.
* **Sidecar coverage**: `D_matrices` come from
  `CKADistancePlugin`, which itself depends on the per-(layer, expert)
  output reservoir (`output_reservoir.pt` ✓ already cached) and
  `MagnitudeTopkPlugin`'s outputs. So the **inputs** to S1_DP are already
  fully cacheable.
* **Gap**: the DP-knapsack itself is cheap (CPU, O(L·K·G) — under a
  second for typical L≈40, K≤256, G≤8000). Caching the **converged
  optimum** (`dp_optimum`, `damage_curves`, `merge_cost_prior_computed`)
  would save <1 second/run — **not worth the cache schema**. Caching the
  CKA inputs is already covered.

### Plugin #9 — S2_MM MergeMoE step
File: `stage2/mergemoe.py:155-302`.

* **Consumes**: `layer_inputs` (per-layer reservoir buffer — the same one
  Plugin #1 fills); member gate/up/down weights (live).
* **Re-computes per run**: per merge-cluster `T₁ = Q·P†` lstsq solve.
* **Sidecar coverage**: `layer_inputs` is the same reservoir Plugin #1
  produces; covered (or rather, **should be** covered — see Gap-1 since
  the vLLM patch doesn't capture it today).
* **Gap**: caching T₁ per (model, calibration_set, merge_cluster) is
  problematic — the cluster identity changes with strategy row (different
  budget → different merges), so the cache hit rate is low. **Skip**.

### Plugin #10 — S2_SEQ REAM sequential
File: `stage2/plugins/ream_sequential.py:215-260`.

* **Consumes**: nothing (sets three ctx slots to `None`).
* **Re-computes per run**: forces the **next layer's** profile pass to
  rebuild `cov_acc` / `ream_acc` / `layer_input_acc` against the
  just-merged model. By design, this is the opposite of caching: it
  invalidates accumulators on purpose.
* **Sidecar coverage**: N/A — and importantly, **a stage2_profile cache
  hit is incompatible with `sequential_reprofile=True`**. When the user
  opts into REAM sequential merging, the cached pre-merge stats are
  stale by construction (REAM §4 — "modified outputs render the
  statistics for subsequent layers as stale"). Section 7 lists this as
  a known invariant.
* **Gap**: NONE; this plugin is intentionally a cache-buster. **Risk**:
  see Recommendations §5 — when `sequential_reprofile=True`,
  `stage2_reap_ream.profile_sidecar.enabled` MUST be false (or only the
  layer-0 hit is correct). Worth a documentation note in MANIFEST + a
  loud-warn at orchestrator startup.

### Plugin #11 — S1_RCO budget
File: `stage1/plugins/rco_budget.py:266-470`.

* **Consumes**: `per_layer_target_experts` (GRAPE output),
  `per_layer_redundancy`, `per_layer_targets`, `per_layer_damage_curve`
  (the S1_DP output when present; synthetic fallback otherwise),
  `decomposition`, `config`.
* **Re-computes per run**: `n_iterations = 500` (rco_budget.py:285) of
  Adam-on-the-manifold with bracketed bisection retraction and
  Gumbel-STE sampling.
* **Sidecar coverage**: NONE. RCO is post-GRAPE and runs entirely on the
  per-layer cost grid + damage curve. The inputs are deterministic given
  GRAPE's output + damage curve, so the converged `α` is reusable across
  identical input states.
* **Gap**: **caching the converged RCO logits is plausible** — Gap-3
  below. The save is small (`α: L × K_max float64`, a few KB), but the
  500-iteration loop runs CPU-bound on float64 — wall-clock impact is
  order-of-seconds to ~1 minute, **not the big lever**.

### Plugin #12 — Opt A profile sidecar
Files: `stage2/plugins/stage2_profile_cache.py`,
`calibration/stage2_profile_writer.py`,
`patches/vllm_calibration_stage2_profile.patch`.

* **Consumes**: writes the `stage2_profile.pt` sidecar (during
  calibration) and reads it (during Stage 2). Provides the strongest
  single cache lever in the pipeline.
* **Re-computes per run**: nothing on a full hit — `LayerMergePlugin.on_profile`
  early-returns when `ctx["stage2_profile_full_hit"]` is set.
* **Sidecar coverage**: covers `ream_acc` (gate_logit_profiles, sim_tensor,
  total_tokens, neuron_act sum/count), `cov_acc` (finalized covariance +
  token_count), AND `layer_input_acc.buffer` (Plugin #1's reservoir).
* **Gap**: **the layer_input_reservoir field is shipped but never
  populated in production**. See Gap-1 below — this is the single most
  important deferred sidecar feature, because it gates the SC strategy's
  big win.

---

## 4. Gap Analysis (candidate new sidecars)

### Gap-1: `layer_in` callback hook in vLLM patch + populate `layer_input_reservoir`
**Status**: deferred follow-up from Plugin #12 (writer patch lines
474-482, 512-527). Schema slot already exists.

* **What it would cache**: per-layer pre-MoE hidden states fed into
  `_LayerInputAccumulator`. The buffer is currently rebuilt live on every
  Stage 2 profile pass for SC rows.
* **Beneficiaries**:
  * Plugin #1 (Opt C reservoir) — main user; ~6.8 min/SC row
    (PLAN_OPT_C_vectorized_reservoir.md:14).
  * Plugin #9 (MergeMoE T₁ solve) — reuses `layer_inputs` (mergemoe.py:155).
  * `_output_space_cost` (output_space_cost.py:416-465) — primary
    consumer; the whole point of `cost_alignment="output"`.
* **Estimated wall-clock saved per Stage 2 SC row**: combining the Opt-A
  numerator (30-50 min/row, plan §1) with Opt-C's reservoir build
  (6.8 min/row), a full Stage 2 cache hit becomes possible **only when
  this hook lands** — until then, SC `cost_alignment="output"` rows always
  fall back to the live forward for `_output_space_cost`, eliminating
  most of Opt-A's quoted win on SC strategy rows.
* **Implementation effort**: small-medium. The site is identified:
  `vllm/model_executor/models/qwen3_moe.py` `Qwen3MoeSparseMoeBlock.forward`
  already has the input `hidden_states` in scope before the
  `self.experts(...)` call (see `vllm_calibration_hooks.patch:10049-10073`
  — the `block_out` hook is dispatched on the OUTPUT; a parallel
  `layer_in` dispatch on the INPUT is one-callback-site shorter).
  Required pieces:
  - New env gate `VLLM_CALIB_CAPTURE_LAYER_IN`.
  - One dispatch site in `Qwen3MoeSparseMoeBlock.forward` (input branch).
  - Driver-side: wire `_state.layer_input_reservoir[layer_idx]` from the
    new callback (mirror `_router_handler`).
  - Reservoir-sampling math (Vitter Algorithm R) — reuse Plugin #1's
    vectorized implementation at `stage2/profiling.py:88-166` (CPU-side;
    no vLLM-side reservoir math needed if we capture every Nth batch's
    full input, but a CPU reservoir keeps payload bounded).
  - Schema: NO bump needed — the `Stage2ProfilePayloadV3.layer_input_reservoir`
    field already exists (cached_calibration_signals.py:215).
  - Storage shape contract: `list[Tensor[N, hidden] bf16]` of length
    `n_layers`; reader expects `reservoir_t.numel() > 0` to hydrate.
* **Risks/tradeoffs**:
  - Disk: ~3 GB at `max_samples=8192` × `n_layers=40` × `hidden=2048` × bf16;
    up to ~15 GB if SC experiments raise the cap (plan §6 / OQ-2 quote).
    Already budgeted by the plan.
  - Correctness: must use the SAME reservoir RNG seed contract as
    `_LayerInputAccumulator` so the buffer is statistically equivalent
    (Plugin #1's `_generator` seed = `layer_idx`).
  - Pattern K: the field already exists at v3; this is a populate-only
    change, no schema bump.
* **Priority**: **HIGH**. Closes the loop on the biggest single lever in
  the pipeline (Opt A + Opt C combined).

### Gap-2: REAM `δ_gate` standalone sidecar separation
**Status**: speculation by the user. The current `stage2_profile`
payload bundles `gate_logit_profiles`, `sim_tensor`, `cov_acc`, and
`layer_input_reservoir`.

* **What it would cache**: just the gate-logit profiles + sim_tensor (the
  REAM `δ_gate` half of the payload).
* **Beneficiaries**: a hypothetical Stage 2 path that wants `ream_acc`
  without `cov_acc` (e.g. `cost_alignment="pre"` with no Stage 3 cov
  follow-on). Today no such path exists — every cost mode either uses
  both or skips both.
* **Estimated wall-clock saved**: zero today. **Negative** in fact: it
  would duplicate ~5-15% of `stage2_profile`'s data on disk and force
  two separate readers when one suffices.
* **Implementation effort**: medium (schema split, new writer module,
  new reader, OQ-1 partial-hit retest).
* **Risks/tradeoffs**: re-introduces the Plugin #12 Bug #8 hazard
  (paired-reader double-counting) that the redo specifically removed by
  consolidating into a single writer/reader pair (see
  `PLAN_PLUGIN_12_opt_a_redo.md:321-323`). **The structural one-reader
  invariant is more valuable than the marginal disk savings.**
* **Priority**: **LOW / NOT WORTH IT**.

### Gap-3: Stage 1 RCO state cache
**Status**: not implemented.

* **What it would cache**: the converged α logits + budget vector from
  the RCO `n_iterations=500` Adam-on-manifold loop
  (`rco_budget.py:402-456`).
* **Beneficiaries**: Plugin #11 (S1_RCO). Cache key would be
  `(model_hash, GRAPE budget hash, per_layer_damage_curve hash,
  global_budget, rco_cfg hash)` — change ANY of these and re-solve.
* **Estimated wall-clock saved**: 500 iterations × per-iter cost. The
  loop is CPU-bound float64 with L×K_max matrices (typical 40 × ~128,
  ~5k elements). My estimate: 30-90 seconds per run. **Not a primary
  lever.**
* **Implementation effort**: small (one schema entry, one
  cache_or_live provider pair).
* **Risks/tradeoffs**:
  - Disk: trivial (`α: L × K_max float64` ≈ 40 KB).
  - Cache invalidation is non-trivial: the converged `α` depends on
    `per_layer_damage_curve` (potentially from S1_DP) — if the damage
    curve changes, the cache must miss. Hash-keying handles this but
    adds plumbing.
  - The 500-iteration loop is already deterministic given the seed,
    so reproducibility is preserved.
* **Priority**: **LOW**. Useful only if RCO becomes a hot iteration
  point (e.g. budget sweeps).

### Gap-4: Stage 1 damage curve cache
**Status**: not implemented as a sidecar; the curve is derived from
`D_matrices` (CKA distances) which themselves come from the reservoir
sidecar.

* **What it would cache**: `damage_curves dict[layer → np.ndarray[k]]`
  + `dp_optimum dict[layer → int]` + `merge_cost_prior_computed dict`.
* **Beneficiaries**: Plugin #8 (S1_DP), and downstream `GrapeMergePlugin`'s
  `merge_cost_prior` inert hook.
* **Estimated wall-clock saved**: sub-second (DP is O(L·K·G), runs on
  CPU). **Not worth a sidecar.**
* **Implementation effort**: small but unnecessary.
* **Risks/tradeoffs**: adds a cache that depends transitively on the
  reservoir sidecar; another invalidation knob.
* **Priority**: **LOW / NOT WORTH IT**. Cache the inputs (already done
  via `output_reservoir`), not the trivial CPU derivation.

### Gap-5: Stage 3/4 covariance separate captures
**Status**: covered today via the `covariance` sidecar (schema v2) AND
the `stage2_profile.cov_acc` field (in the `stage2_profile` v3 sidecar).

* **What it would cache**: anything beyond `Σ_in` per (layer, expert,
  matrix_name)?
* **Beneficiaries**: Stage 3 / Stage 4 already hit the `covariance.pt`
  sidecar at startup (see Stage 3 `Stage3InputCovCacheProvider` and
  Stage 4 `Stage4InputCovCacheProvider`).
* **Estimated wall-clock saved**: zero — coverage is complete. The
  *post-prune* `B_acc` covariance (collected by Stage 3's
  `_collect_covariances`, vendored at `stage3/plugins/covariance_collection.py:194-336`)
  cannot be precomputed during teacher calibration because by definition
  it depends on the pruned student.
* **Priority**: **NOT APPLICABLE** — already covered by existing sidecars.

### Gap-6: MergeMoE T₁ closed-form result cache
**Status**: not implemented.

* **What it would cache**: per (model, calibration set, cluster
  membership) the `T₁ = Q·P†` solve result.
* **Beneficiaries**: Plugin #9.
* **Estimated wall-clock saved**: the lstsq solve runs in fp32 on at
  most `cost_output_token_cap=1024` tokens × N·d_int columns. Per cluster
  this is ~50-200 ms on CPU/~10 ms on GPU. With ~40 layers × handful of
  merge clusters per layer, ~1-5 min/SC row.
* **Implementation effort**: medium. Cache key MUST include the cluster
  membership (`tuple(sorted(member_ids))`), permutation alignment, and
  the freq weights — all of which depend on the SC strategy row's
  greedy assignment. Cluster identity changes across rows.
* **Risks/tradeoffs**:
  - Cache key complexity: clusters change per strategy row, so cross-row
    hit rate is near zero. Cross-run hit rate (re-running same strategy
    on same model) is high but limited.
  - Disk: `(N·d_int, d_int)` fp32 per cluster ≈ `(2·512, 512)·4` = 1 MB
    per cluster, ~hundreds of MB per run.
  - Numerical: the cond-fallback at threshold 1e8 (`mergemoe.py:59`) is
    sample-dependent — caching freezes that decision.
* **Priority**: **MEDIUM**, but only if MergeMoE becomes the default
  merge step. Today it's opt-in (`merge_step="mergemoe"`), so the
  payoff is rare.

### Gap-7: per-layer `effective_end` / max_layer-aware sidecar shards
**Status**: partial — the L2 `VLLM_CALIB_MAX_LAYER` early-exit lets the
calibration driver run only layers 0..N; existing sidecars degrade
gracefully (MANIFEST lines 154-160). Sharded sidecars (per-layer files)
already exist for `block_hidden`.

* **What it would cache**: per-layer shards for the `stage2_profile`
  payload (currently a single monolithic `.pt`).
* **Beneficiaries**: enables piecewise calibration capture (run layers
  0-19 on one rental, 20-39 on another) without rewriting the whole
  sidecar each time. Useful for spot-preemption recovery.
* **Estimated wall-clock saved**: zero direct savings; this is a
  **resilience** lever, not a speed lever. Already handled today by the
  checkpoint mechanism (`dump_stage2_profile_checkpoint` /
  `load_stage2_profile_checkpoint`).
* **Risks/tradeoffs**: significant schema work (split + glue reader)
  for no first-order time win.
* **Priority**: **LOW**.

---

## 5. Recommendations (prioritized)

Wall-clock estimates ranked by SC strategy row impact (the production hot
path). Where the source is a plan doc, the line is cited; otherwise the
number is my estimate, flagged.

| Pri | Item | Wall-clock saved per SC row | Effort | Disk | Risk |
|---|---|---|---|---|---|
| **H** | **Gap-1**: vLLM `layer_in` hook + populate `layer_input_reservoir` (closes the loop on Plugin #1 + Plugin #12) | ~6.8 min (Opt C, PLAN_OPT_C:14) **plus** unblocks the SC-row share of Opt A's 30-50 min (PLAN_PLUGIN_12:5) | small-medium (~1 new dispatch site, ~50 LOC writer + driver wiring; no schema bump) | +3 GB (up to 15 GB at SC cap) | low — field already exists; reader already gated |
| **M** | **Gap-6**: MergeMoE T₁ cache (only if `merge_step="mergemoe"` becomes default) | ~1-5 min (estimate) | medium (cluster-keyed cache, ~200 LOC) | hundreds of MB | medium — cluster identity changes per strategy row |
| **L** | **Gap-3**: RCO converged-α cache | ~30-90 s (estimate) | small | ~40 KB | low |
| **L** | **Gap-7**: stage2_profile per-layer shards (resilience, not speed) | 0 direct | medium | same total disk | low |
| **DROP** | **Gap-2**: REAM `δ_gate` standalone sidecar (re-introduces Bug #8 hazard) | 0 / negative | medium | duplicates ~5-15% | **high** — undoes Plugin #12 redo invariant |
| **DROP** | **Gap-4**: Stage 1 damage curve cache | sub-second | small | trivial | adds an invalidation knob; no payoff |
| **N/A** | **Gap-5**: Stage 3/4 covariance separate captures | already covered | n/a | n/a | n/a |

**Bold takeaway**: only Gap-1 (`layer_in` hook) is a primary lever
worth landing in the next sprint. Every other candidate either is
already covered, saves a tiny amount, or undoes the Plugin #12 single-
reader invariant.

**Cross-cutting risk (Plugin #10 ↔ Plugin #12 interaction)**: when
`stage2_reap_ream.sequential_reprofile=True`, the cached `stage2_profile`
payload becomes **stale for layers ≥ 1** because every preceding layer's
merge modifies the upstream context the sidecar was captured against
(REAM §4 semantics — see `ream_sequential.py:14-21`). The current
`profile_sidecar.enabled` knob does not interlock with
`sequential_reprofile`. Recommended next step (small but important):
add an orchestrator-startup assertion that raises if both are truthy, and
document the mutual exclusion in MANIFEST.md.

---

## 6. Follow-ups Already Queued

Comparing the known follow-up backlog against the new gap analysis:

| Item | Priority vs Gaps 1-7 | Notes |
|---|---|---|
| **RCO re-vendor** from upstream (Plugin #11; user has author consent; clean-room has 2 algorithmic bugs the re-vendor fixes) | **HIGH** — orthogonal to caching; correctness fix, not a perf lever. Should land before Gap-3 (no point caching a buggy result). | Cited at `rco_budget.py:62-69` (D-clean-room). |
| **vLLM `layer_in` callback hook** (Plugin #12 reservoir capture) | **= Gap-1**. This IS the audit's top recommendation. | Patch site identified: `qwen3_moe.py` `Qwen3MoeSparseMoeBlock.forward` before `self.experts(...)`; H-1 footnote in writer patch lines 474-482. |
| **Pattern H backport to `ream_cost_post.py:214`** (argpartition hoist, same as Plugin #3 already applied) | **MEDIUM** — direct ~1 min/SC row win (mirrors PLAN_OPT_B3:13). Not a sidecar gap; pure CPU-side win. Should land alongside any other perf work. | Confirmed by reading `ream_cost_post.py:209-214`. |
| **Plugin #1 M1 docstring** (no `.item()` calls → no GPU sync points) | **LOW** — documentation only. | |
| **Plugin #5 M1 stale comment** (Dtype hardcoded to fp32 in `output_space_cost.py:447-450`) | **LOW** — comment tweak. | Confirmed lines 450-451: "*Dtype is hardcoded to fp32 at the SwiGLU call sites below.*" |

The follow-up backlog overlaps Gap-1 (the most important new
recommendation). The remaining items are either independent correctness
work (RCO re-vendor) or low-risk hygiene.

---

## 7. Summary

**Top 3 next actions, ranked by leverage:**

1. **Land Gap-1 (`layer_in` hook in vLLM patch + populate
   `layer_input_reservoir`).** This is the deferred half of Plugin #12
   and the missing input to Plugin #1's reservoir cache. Without it, the
   Stage 2 SC row still pays the per-layer forward for `_output_space_cost`
   even on a "full hit" `stage2_profile` cache, defeating most of the
   30-50 min/row Opt-A win on SC strategy rows. Effort is small (one
   dispatch site, no schema bump — the field already exists at
   `cached_calibration_signals.py:215`).

2. **Add the `sequential_reprofile` ⊕ `profile_sidecar.enabled`
   mutual-exclusion guard** at the Stage 2 orchestrator start. This is
   a correctness fence, not a cache: Plugin #10 invalidates accumulators
   precisely because the cached pre-merge stats are stale under REAM
   sequential merging. Letting both knobs run together silently merges
   a stale sidecar into the post-merge state. Small change; high value.

3. **RCO re-vendor (Plugin #11 follow-up)**. Two algorithmic bugs in
   the clean-room implementation are known; the author has given consent
   to vendor the upstream repo. Land this BEFORE any caching of RCO's
   converged α (Gap-3), so the cache doesn't freeze the buggy result.

**Everything else** — Gap-2, Gap-4, Gap-5, Gap-6, Gap-7 — is either
not worth the effort, already covered, or actively undoes the Plugin
#12 single-reader invariant.

**Sidecar inventory health**: the current set (`reap_scores`,
`per_expert_max`, `routing_stats`, `router_logits_stats`,
`output_reservoir`, `covariance`, `stage2_profile`, `block_hidden`,
`router_kd_logits`, `teacher_eval`, plus the legacy `phase_b`) covers
every consumer plugin's needs except the one missing field
(`layer_input_reservoir`) on the one sidecar (`stage2_profile`).
Schema versions are centralised, atomicity is uniform, multi-arch
portability is enforced — the architectural foundation is sound.
