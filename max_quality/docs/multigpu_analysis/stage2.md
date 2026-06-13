# Stage 2 (REAP/REAM prune-merge) — Multi-GPU Feasibility Analysis

Read-only deep analysis. Code root: `max_quality/src/moe_compress/stage2/`.
Templates consulted: Stage-3 `covariance_collection.py` (DATA-PARALLEL `mp.spawn`
replicas + key-wise spill reduce), Stage-4 `eora_compensation.py` (TASK-PARALLEL
per-expert fan-out), `utils/auto_batch.py` (per-GPU VRAM-aware batch sizing).

---

## TL;DR — the one structural fact that governs everything

**Stage 2's per-layer loop is UNCONDITIONALLY layer-sequential, and there is NO
multi-GPU machinery anywhere in `stage2/`.**

The driver loop (`orchestrator.py:1673-1712`) does, per MoE layer in order:

1. **profile** — `LayerMergePlugin.on_profile` → `_profile_layer`
   (`profiling.py:169`) runs an **early-exit forward through layers `0..L` of the
   LIVE model** (`profiling.py:338-342`, `early_exit_after_layer`).
2. **assign** — `_run_assignment` bump loop (cost → mask → solve → refine).
3. **merge** — `merge` phase → `_merge_experts_inplace` **mutates the model in
   place** (`merging.py` writes sliced/merged expert tensors + resizes the router).

Because step 3 mutates the model and step 1 of layer `L+1` forwards through
`0..L` of *that mutated model*, **layer `L+1`'s profile always consumes layer
`L`'s merged weights**. This dependency is intrinsic to the in-place-merge design
and is **independent of the `sequential_reprofile` flag**. That flag
(`Stage2ReamSequentialPlugin`, `ream_sequential.py:215-240`) only controls whether
the *accumulators* are invalidated between layers (i.e. whether you may reuse a
profile sidecar); it does NOT remove the forward-path data dependency, because the
forward always runs against the live, already-mutated model.

> Consequence: **per-LAYER task-parallelism across GPUs is illegal** in the
> default design (it would profile every layer against the *unmerged* model,
> changing the result — REAM §4 sequential merging is the whole point). The
> exploitable multi-GPU axis is **DATA-PARALLEL within each layer's profile
> forward** — shard the calibration set across GPUs, reduce the additive
> accumulators key-wise — which is *exactly* the Stage-3 covariance pattern and is
> result-preserving.

Everything below is organized around that fact.

---

## Per-plugin table

Legend for (d) multi-GPU scheme:
`DP` = data-parallel (shard calibration, reduce); `TASK-PAR` = task-parallel
(per-layer/per-expert/per-group); `DDP` = distributed-data-parallel training;
`N/A` = CPU/tiny, multi-GPU moot; `NWI` = not-worth-it.

| Plugin / file | (a) Computes | (b) Profile | (c) Device today | (d) MG scheme | (e) Result-preserving mechanism | (g) Effort / risk / speedup |
|---|---|---|---|---|---|---|
| **profiling.py** `_profile_layer` | THE per-layer calibration forward; feeds REAP+REAM+cov+distill-input in one pass (`profiling.py:296-360`) | calibration-forward-scoring | GPU forward, early-exit 0..L (`:338-342`) | **DP** (shard `batches` `:327`) | All accumulators are additive Σ over tokens; reduce numerator + token-count separately, mean at read | **L / med / HIGH** (the bottleneck) |
| **reap_scoring.py** | REAP Eq.9 saliency `S_j=(1/|X_j|)Σ g_j·‖f_j‖₂`; builds/finalizes `ReapAccumulator`, ranks centroids | tiny (orchestration) | CPU; `np.argsort(-scores,kind='stable')` `:276` | rides DP of profiling | `sums` + `counts` both additive (`activation_hooks.py:881-889`), mean-at-read `score()` `:923` | S / low / — (rides forward) |
| **reap_prune.py** | Faithful top-K structural prune (drop low-saliency experts, no rescale). INERT unless `prune_mode=="faithful_prune"` | tiny / merge-op | CPU select `:207`; GPU tensor-slice in `post_merge` `:394-409` | N/A (consumer) | Reads already-reduced `scores`; one-shot slice | S / low / — |
| **reap_scores_cache.py** | Loads `sidecars/reap_scores.pt` → populates `ctx.scores/freq` on hit | cache-IO | CPU `:78` | N/A | Proves saliency is already serialized as a flat shardable `[layer][expert]` Σ | S / low / — |
| **routing_stats_cache.py** | Loads routing-freq sidecar. **No production consumer** (`:11-17`) | cache-IO | CPU | N/A | — | — (dead) |
| **ream_cost.py** (`pre`) | δ_gate + δ̃_expert cost matrix, combine `1−(δ_g+δ_e)/2` | tiny (numpy) | **numpy-CPU** `:312,:226-228,:339` over `[E,E]≤256²` | N/A | — | NWI (tiny CPU) |
| **ream_cost_post.py** (`post`) | Whitened Frobenius residual cost over top-K centroids (eigh + matmul), Hungarian per row | GPU-linalg + CPU-solver, **thread-per-row** | torch on model device; eigh CPU-LAPACK `:206,:228`; `parallel_map` rows `:371` | intra-layer only (rows) | already row-threaded | NWI for MG (per-layer, sequential) |
| **output_space_cost.py** (`output`) | Per candidate pair: tentative merge + SwiGLU forward; routing-weighted ‖ΔE‖²; Hungarian. Dominant cost mode | GPU calibration-forward + CPU-solver, **thread-per-row** | GPU forwards on model device `:493-588`; `parallel_map` rows `:596` | intra-layer only (rows) | row-disjoint threading | NWI for MG (intra-layer, bounded by sequential loop) |
| **ream_sequential.py** | Clears `cov_acc/ream_acc/layer_input_acc` on `on_post_merge` | tiny | none `:238-240` | N/A | — | — (it IS the sequential enforcer for accumulators) |
| **layer_merge.py** | The per-layer merge SPINE: profile→merge→bank.select→router resize→artifact | orchestration + forward + merge-op | GPU profile `:528`; weight-slice `:624` | **DP within profile** | delegates math to siblings | rides profiling DP |
| **merging.py** `_merge_experts_inplace` | REAM Eq.6 freq/saliency-weighted merge + Hungarian align, in place | merge-op + (regmean/mergemoe GPU) + CPU-Hungarian | torch on weight device; threaded Hungarian `:247`; serial fp accumulate `:217-225` | intra-layer (groups) | split-phase: solve in pool, accumulate serial for bit-repro | NWI for MG (few groups/layer) |
| **regmean.py** | RegMean closed-form `W=(ΣG)⁻¹ΣGW` per Linear/group | GPU-linalg | `torch.linalg.solve` `:269`, `cond` `:239`, fp32, weight device | TASK-PAR intra-layer (groups×3 Linears) — small fan-out | independent self-contained solves | NWI/S (small fan-out, sequential outer) |
| **mergemoe.py** | MergeMoE `T₁=Q·P†` down-proj via lstsq on calib tokens | GPU-linalg | `torch.linalg.lstsq` `:359`, `cond` `:311`, fp32, weight device | TASK-PAR intra-layer (groups) | independent per-group lstsq | NWI/S |
| **regmean_merge.py** | Metadata/validation shim (`cov_acc` present) | tiny | none `:153-181` | N/A | — | — |
| **expert_distill.py** | **TRAINS** merged centroid's 3 SwiGLU matrices (AdamW MSE+optional KL) to mimic pre-merge group on calib tokens | **training** | AdamW `:625`, `backward` `:809`, single `device` | **TASK-PAR per-group** (not DDP) | per-group independent; deterministic group order | M / med / med (per-layer bounded) |
| **merge_heal.py** | **TRAINS** all kept experts (+router) of ONE layer by self-distillation to pre-merge MoE-block output; accept/reject vs holdout | **training** | AdamW `:770`, `backward` `:902`, streams bf16 shards; single `device` | **TASK-PAR per-layer** (blocked by sequential dep) / or **DP minibatch within a layer** | per-layer independent IF dep broken; else DP the heal minibatch | M-L / med-high / med |
| **em_refine.py** | EM re-assignment: tentative merge (M) + cost recompute + re-solve (E), iterative. Default rounds=0 (inert) | CPU-solver + GPU-linalg, **NOT training** | tentative weights fp32 on bank device `:219-258`; solver CPU `:365`; loop `:333` | intra-layer | iterative, per-layer | NWI for MG |
| **two_opt_refine.py** | 2-opt local search over assignment. Default OFF | CPU-solver (pure Python) | CPU triple-loop `:136-192` | N/A | — | NWI |
| **capacity_gate.py** | Scalar SLACK/TIGHT path selector | tiny | pure Python scalar | N/A | — | — |
| **skip_merge_floor.py** | Percentile cost-mask. Default OFF | tiny (numpy percentile) | CPU | N/A | — | — |
| **solver_dispatch.py** | name→solver registry/dispatch | tiny | pure Python/numpy | N/A | — | — |
| **solver_greedy.py** (DEFAULT) | Descending-saliency greedy assign | CPU-solver | **numpy** `np.argmin` `:156,:202`; `[E×C]≈256×180` | N/A | — | NWI (µs CPU) |
| **solver_hungarian.py** | Rectangular LSA | CPU-solver | **scipy** `linear_sum_assignment` `:83,:123` | N/A | — | NWI (<1ms) |
| **solver_mcf.py** | Capacitated min-cost flow | CPU-solver | **ortools** `SimpleMinCostFlow` `:142` (~10ms) | N/A | — | NWI |
| **solver_sinkhorn.py** | Log-domain Sinkhorn-Knopp OT (200 iters). Default OFF | CPU-solver (iterative) | **numpy + scipy.special.logsumexp** `:91,:92`, loop `:211-217`, `[257×180]` fp64 — **NOT torch, NOT GPU** | N/A | — | NWI (too small for GPU) |
| **solver_auto.py** | Hungarian/MCF meta-router | tiny | Python branch | N/A | — | — |
| **shared_io.py** | Durable artifact IO (merge JSON, cov snapshot, heal weights) | cache-IO | CPU-only (`.cpu()` before save `:82,:376`) | N/A | — | — |
| **stage2_profile_cache.py** | Hydrates `ream_acc/cov_acc/layer_input_acc` from `stage2_profile.pt` sidecar (skip live profile) | cache-IO | device-agnostic, CPU load | N/A | de-quadratic bucketing `:180` | — |

---

## (f) How per-GPU auto-batch sizing is preserved (the HARD requirement)

The DP scheme reuses the Stage-3 cov contract **verbatim** — that is its whole
appeal. In `covariance_collection.py` each replica:

- pins itself to one GPU via `CUDA_VISIBLE_DEVICES` (`_cov_replica_worker`,
  `covariance_collection.py:1074`), so `CudaMemProbe` sees ITS OWN free VRAM;
- runs the auto path (`_cov_is_auto`, `:450`) — `size_batch` probes the live
  `max_memory_allocated` with the resident model, sizes the per-replica forward
  batch, and `run_with_oom_backoff` halves to `floor=1` on OOM
  (`:936-992`). **Each replica sizes independently** — see the explicit comment
  block `covariance_collection.py:373-447`: "every replica probes its OWN
  pinned-device VRAM and sizes INDEPENDENTLY".

For Stage 2 the identical wiring applies to `_profile_layer`'s batch loop
(`profiling.py:327`). The per-GPU sizing is preserved by construction:

1. Each replica is a separate `mp.spawn` process pinned to one GPU → its own
   `CudaMemProbe.total()`/`allocated()`.
2. The forward-batch sizing uses `size_batch(cost_probe_fn, floor=1,
   headroom_frac, max_cap, mem=CudaMemProbe(device))` against that replica's VRAM,
   then `run_with_oom_backoff` to `floor=1`.
3. **Reduction-pin makes it result-preserving across batch sizes** — the same
   per-sequence pin Stage-3 uses (`InputCovarianceAccumulator.update_grouped`,
   split by `token_idx // seq_len`, `profiling.py`-side cov already uses
   `cov_acc.update`; the grouped variant exists in `activation_hooks.py:1032`).
   REAP's `ReapAccumulator` is even simpler: `record_reap` sums per-token
   `g_j·‖f_j‖₂` and a token count — both are forward-batch-size-invariant under a
   per-sequence pin because addition of the same per-sequence operands in the same
   order is identical regardless of how the forward batched them.

So **no new auto-batch logic is needed** — the DP Stage-2 forward inherits the
Stage-3 per-replica `size_batch` + `run_with_oom_backoff` + per-sequence-pin
machinery unchanged. The classes already live in `utils/auto_batch.py` and
`utils/activation_hooks.py`.

---

## The 4 KEY QUESTIONS

### Q1. Is the REAP/REAM scoring calibration forward the bottleneck, and can it be DATA-PARALLEL (shard calibration, reduce saliency) with a replica-independent reduction + per-replica auto-batch — like Stage-3 cov?

**YES on both counts — this is THE win.**

- **Bottleneck:** There is exactly ONE calibration forward per layer,
  `_profile_layer` (`profiling.py:169`), an `instrument_experts`-hooked
  `model(input_ids=batch)` early-exit forward (`:338-342`). It co-produces REAP
  scores, REAM δ_gate/δ̃_expert, input covariance (Stages 3/4 consume it), AND the
  distillation-input reservoir — all in one pass. The forward + per-expert
  instrumented matmuls dominate Stage-2 wall time. Total layer-forwards across 40
  sequential passes = 820 (`profiling.py:190-191`).

- **Replica-independent reduction — proven additive.** REAP saliency is a pure
  additive sum with a separate additive count, mean applied only at read:
  - per-token contribution (`activation_hooks.py:1433-1435`):
    `contrib = (gate_vals.float() * expert_outs.float().norm(dim=-1)).sum()`
  - accumulate (`activation_hooks.py:881-889`): `cur.add_(contrib)`,
    `counts[k] += n_tokens`, `freq[k] += n_tokens`
  - read (`activation_hooks.py:923-930`): `s / n`
  ⇒ `S_j = (Σ_r numerator_r) / (Σ_r count_r)`. Both numerator and denominator are
  linear sums over shards → reduce key-wise (sum `sums`, sum `counts`), then
  `score()` once. **No mean-weighting trap, no order dependence** beyond the ~1e-5
  fp32 non-associativity the cov path already tolerates
  (`activation_hooks.py:308-311`). REAM is identical: `_gate_gram` is `G += bᵀb`
  (`:179-186`), `_sim_tensor` is `t.add_(sim_sum)` (`:462`), with
  `_total_tokens_by_layer` (`:235`) as the separate additive denominator. Input
  covariance is the *same* Gram Stage-3 already mp.spawn-reduces via
  `_reduce_spilled_cov_dirs` (`covariance_collection.py:211-289`).

- **The implementation already exists at the right cut point.** Shard `batches`
  (`profiling.py:327`) across N replicas exactly as `_shard_calib`
  (`covariance_collection.py:1052`) does; each replica builds its own
  `ReapAccumulator + InputCovarianceAccumulator + ReamCostAccumulator`; key-wise
  reduce all three; finalize/normalize once after the reduce. **One sharded
  forward serves REAP + REAM + cov + distill-input together** — you do NOT
  parallelize REAP separately from cov; you parallelize the single forward and
  reduce all accumulators.

- **BUT it must be done PER LAYER, inside the sequential loop** (not across
  layers — see Q4). i.e. for each layer: spawn N replicas → each forwards its
  calib shard through `0..L` of the (already-merged-upstream) model → reduce
  accumulators → assign+merge on the parent → repeat for `L+1`. This respects the
  layer dependency while parallelizing the expensive forward. The replica spawn/
  teardown cost amortizes if you keep a persistent replica pool across layers
  (the model copies stay resident; only the active layer's hooks change).

- **Per-replica auto-batch:** inherited verbatim from Stage-3 (see (f) above).

**Effort L, risk medium, speedup HIGH (near-linear in GPUs on the forward, which
is the bottleneck).** Main complexity: a persistent per-layer replica pool (vs
Stage-3's once-per-run spawn) and re-broadcasting the merged upstream weights to
replicas after each layer's merge — that broadcast is the new cost Stage-3 didn't
have (Stage-3 collects cov once on a fixed student). A simpler v1 is respawn-free
by having replicas reload the just-merged layer's weights from the parent each
iteration.

### Q2. Are the merge solvers CPU-bound (multi-GPU moot) or GPU?

**ALL CPU-bound — multi-GPU is MOOT for the entire solver layer.** Definitively,
with citations:
- greedy (default): **numpy** `np.argmin` (`solver_greedy.py:156,202`)
- hungarian: **scipy** `linear_sum_assignment` (`solver_hungarian.py:83,123`)
- mcf: **ortools** `SimpleMinCostFlow` (`solver_mcf.py:142`, ~10ms/layer)
- **sinkhorn (the one to check): numpy + `scipy.special.logsumexp`**
  (`solver_sinkhorn.py:91,92`, iteration loop `:211-217`), `[257×180]` fp64 —
  **NOT torch, NOT GPU.** 200 iters of two small logsumexp reductions = ms; a GPU
  would lose to kernel-launch overhead.
- auto/two_opt/capacity_gate/skip_merge: pure Python / numpy scalar.

Cost matrices are `~256×180` per layer → microseconds-to-~10ms on CPU. The right
parallel axis for solvers is CPU multiprocessing across layers — but even that is
pointless because the solvers are a rounding-error fraction of the forward cost.
**Skip them entirely for multi-GPU.**

### Q3. Does any plugin TRAIN (→ DDP candidate)?

**Two plugins train; NEITHER is a good DDP candidate — both want TASK-PARALLEL.**

- **expert_distill.py** — trains the merged centroid's **3 SwiGLU matrices only**
  (AdamW `:625`, `loss.backward()` `:809`, MSE `:796` + optional feature-KL
  `:805`), **per merge-group, per layer**. Unit is tiny (3 matrices, ≤8192
  tokens, default 500 steps). The parallelism axis is the `for centroid, members
  in grouped.items()` loop (`orchestrator.py:1081`) — embarrassingly parallel
  across groups with no shared mutable state. **TASK-PARALLEL per-group across
  GPUs**, NOT DDP (DDP shards one model's batch — wrong shape for a 3-matrix
  unit). Effort M, risk medium, speedup medium (bounded by groups/layer and the
  sequential outer loop).

- **merge_heal.py** — trains **all kept experts of ONE layer (+router)** by
  self-distillation to the pre-merge MoE-block output (AdamW `:770`,
  `loss.backward()` `:902`, streams bf16 activation shards, default 2000 steps,
  accept/reject vs holdout). Unit = one layer's expert bank (bigger, but still a
  single MoE block). **TASK-PARALLEL per-layer** is the natural axis — but it is
  **blocked by the layer-sequential dependency** (Q4): healing layer L changes the
  weights layer L+1 profiles against. The legal multi-GPU option here is **DP the
  heal minibatch within a single layer** (shard `sample_minibatch` `:898` across
  GPUs, all-reduce grads) — a genuine DDP-style use, but per-layer-at-a-time. Or
  accept a quality approximation (heal all layers against the pre-merge model in
  parallel) — a deviation, not result-preserving. Effort M-L, risk medium-high.

- **em_refine.py** does **NOT** train (no optimizer/backward; iterative
  M-step tentative-merge + E-step re-solve, `:333`). Default rounds=0 (inert).

### Q4. Can per-LAYER merge work be task-parallel across GPUs (independent per layer)?

**NO — not in the default design.** The per-layer loop carries an unconditional
forward-path data dependency:

- merge mutates the model in place (`_merge_experts_inplace`, `merging.py`;
  `bank.select` + `_resize_router_for_kept_experts`, `layer_merge.py:624-627`);
- each layer's profile forwards through `0..L` of that mutated model
  (`profiling.py:338-342`, `early_exit_after_layer`);
- so layer `L+1`'s profile **always** sees layer `L`'s merged weights — this is
  REAM §4 sequential merging and is the intended semantics
  (`profiling.py:182-188` docstring).

`sequential_reprofile` (`ream_sequential.py:215-240`) does NOT change this — it
only invalidates *accumulators* so a fresh reprofile runs (vs reusing a sidecar);
the forward already runs against the live merged model either way. Running layers
in parallel would profile each against the *unmerged* upstream → a different
(worse) result → not result-preserving.

**Exploitable parallelism is INTRA-layer:** (i) the DATA-PARALLEL profile forward
(Q1 — the real win), (ii) thread-per-row cost matrices (already implemented:
`ream_cost_post.py:371`, `output_space_cost.py:596`, `merging.py:247`), (iii)
per-group regmean/mergemoe solves and expert_distill (small fan-out). None of
these crosses the layer boundary.

---

## Recommended multi-GPU roadmap for Stage 2 (priority order)

1. **DATA-PARALLEL the per-layer profile forward** (Q1). This is the bottleneck
   and the only large, clearly result-preserving win. Reuse Stage-3's
   `mp.spawn` replica + `_reduce_spilled_cov_dirs`-style key-wise reduce +
   per-replica `size_batch`/`run_with_oom_backoff` + per-sequence reduction-pin —
   applied PER LAYER inside the sequential loop, reducing REAP+REAM+cov+distill
   accumulators together. Effort L, risk medium, speedup near-linear on the
   forward. New cost vs Stage-3: re-syncing merged upstream weights to replicas
   each layer.
2. **TASK-PARALLEL expert_distill across groups** (Q3) — embarrassingly parallel,
   no shared state, bounded by groups/layer. Effort M, medium speedup.
3. **(optional) DP the merge_heal minibatch within a layer** (Q3) — genuine DDP
   shape but per-layer-at-a-time; quality-preserving only if grads all-reduce
   exactly. Effort M-L, higher risk.
4. **Skip everything else** — all solvers are CPU/numpy/scipy/ortools (Q2);
   cost-matrix builds are tiny or already row-threaded; regmean/mergemoe are
   small per-group GPU solves bounded by the sequential loop.

**Caveat on REAP in production:** REAP scores often come from the vLLM
`--capture-reap-scores` sidecar (`reap_scores_cache.py`; `reap_prune.py:309`
fails loud without it), so the HF `_profile_layer` REAP accumulation can be a
discarded side-effect. The HF forward is still mandatory for **covariance**
(Stages 3/4) and REAM — which is the real reason to data-parallel it. If you
shard, REAP rides along for free.
