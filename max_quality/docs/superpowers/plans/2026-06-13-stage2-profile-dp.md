# Stage 2 — DP per-layer profile forward + merge-anchor union fix

**Branch:** `feat/stage2-profile-dp` · **Worktree:** `/home/lucas/ai/wt-s2mg` · **Code root:** `max_quality/`
**Status:** PLAN (do not implement from this doc without the review loop).
**Date:** 2026-06-13

Two independent features, both Stage 2, bundled in this worktree:

- **(A)** Data-parallel the Stage-2 per-layer profiling forward (the bottleneck). Shard
  calibration **by sequence** across GPUs, key-wise reduce the additive accumulators
  (REAP + REAM grams + input covariance), normalize once. RESULT-PRESERVING, opt-in,
  default byte-identical.
- **(B)** Fix the REAM **merge-anchor wart**: the post-merge covariance remap copies the
  **centroid's own** input Gram into the survivor slot instead of the **group-UNION**
  (`Σ_j G_j` over the merge group). A real correctness bug surfaced by the acov research,
  independent of multi-GPU. Lives in Stage 2 → bundled here (NOT the acov branch).

---

## 0. Load-bearing facts established by the spec + code read (cite + verified)

Spec: `max_quality/docs/multigpu_analysis/stage2.md`. Verified against the actual code below.

### The sequential-merge constraint is LOAD-BEARING (no cross-layer parallelism)

The driver loop (`stage2/orchestrator.py:1673-1712`) processes MoE layers **strictly in
order**. Per layer it runs: profile → assign (`_run_assignment`) → merge
(`LayerMergePlugin.merge` → `_merge_experts_inplace`, `merging.py:423-424` `bank.set`) →
post_merge (`bank.select` + router resize, `layer_merge.py:624-627`). The merge **mutates
the live model in place**. Layer `L+1`'s profile forwards through layers `0..L` of *that
mutated model* (`profiling.py:338-342`, `model(input_ids=batch)` under
`early_exit_after_layer(model, layer_idx)`). So **layer L+1's profile always consumes layer
L's merged weights** — REAM §4 sequential merging (`layer_merge.py` docstring `:237-266`).

> **Therefore: task-parallelism ACROSS layers is ILLEGAL (−1.0 AVG, REAM §5.4 ablation).
> The plan MUST NOT propose cross-layer parallelism.** The only exploitable axis is
> DATA-PARALLEL **within each layer's profile forward**, applied PER LAYER inside the
> sequential loop. Every task below respects this.

### The forward is the bottleneck and is exactly DP-able

There is exactly ONE calibration forward per layer: `_profile_layer` (`profiling.py:169`),
an `instrument_experts`-hooked early-exit `model(input_ids=batch)`
(`profiling.py:327-348`). It co-produces REAP scores, REAM δ_gate/δ̃_expert, input
covariance, and the distill-input reservoir in one pass. Total layer-forwards across the
sequential passes (40-layer model) = 1+2+…+40 = 820 (`profiling.py:190-191`).

### All accumulators are additive Σ over tokens with a separate additive count — proven (verified)

| Accumulator | Numerator (additive) | Count (additive) | Read (mean-at-read) |
|---|---|---|---|
| **REAP** `ReapAccumulator.add_gpu` | `_gpu_sums[k].add_(contrib)` `activation_hooks.py:885-887`; `contrib = (gate·‖f‖).sum()` `:1433-1434` | `counts[k]+=n`, `freq[k]+=n` `:888-889` | `score() = s/n` `:923-930` |
| **REAM δ_gate** `record_router_logits` | `_gate_gram[li].add_(bᵀb)` `:182-186` | (Gram diag carries ‖v‖²; no separate count) | `compute_gate_similarity_matrix` `:464` |
| **REAM δ̃_expert** `finalize_batch` | `_sim_tensor[li].add_(sim_sum)` `:460-462` | `_total_tokens_by_layer[li]+=n` `:235` (via `record_batch_token_count`) | `compute_delta_expert = sim/total` `:586` |
| **REAM C_act** `record_neuron_activations` | `_neuron_act_sum[k]+=batch_sum` `:604-608` | `_neuron_act_count[k]+=n` `:609` | `get_neuron_mean = s/c` `:619` |
| **Input cov** `InputCovarianceAccumulator.update` | `_pending[k].add_(cov)` `:1024-1029` (cov = `flatᵀ@flat`) | `_gpu_token_count[k]+=n` `:1030` | `finalize_layer` sums + casts `:1082-1112` |

Every numerator is a **linear sum over tokens**; every count is a **linear sum over
tokens**. ⇒ For a sequence-disjoint shard set, the reduce is exactly
`Σ_r numerator_r` and `Σ_r count_r`, mean applied once after the reduce. This is the
**same additive-Gram property** Stage-3 already mp.spawn-reduces. The only non-determinism
is fp32 accumulation non-associativity (~1e-5/1e-6), which the existing pipeline already
tolerates (`activation_hooks.py:308-311`) and which the byte-identical default path avoids
entirely (1 replica ⇒ no reduce).

### The Stage-3 DP template to mirror (verified, `stage3/plugins/covariance_collection.py`)

1. **Driver:** `run_dp_covariance_collection(... replicas, ...)` `:1213-1289` — shards
   calib, spawns replicas, joins, reduces spill dirs.
2. **Shard:** `_shard_calib(calib, replicas)` `:1052-1071` — **contiguous dim-0 slices
   (by sequence), token-disjoint**, last shard takes remainder. "each replica owns its own
   token_idx space; we never share … only sum the final per-(layer,expert) Gram matrices".
3. **Replica worker:** `_cov_replica_worker(...)` `:1074-1210` — pins via
   `os.environ["CUDA_VISIBLE_DEVICES"]=visible_devices` `:1096`, **reloads model from
   disk** (`_load_compressed_model(student_path,…)` `:1119`), builds its own accumulator,
   runs `_collect_covariances(calib=shard, cov_auto=…)`, spills per-layer `layer_{idx}.pt`
   to its **per-replica** dir.
4. **Reduce:** `_reduce_spilled_cov_dirs(replica_dirs, out_dir, storage_dtype)` `:211-289`
   — key-wise **fp32 sum** of `payload["covariance"]`, token counts sum as ints, processed
   in **sorted replica-dir order** (determinism), write canonical `layer_{li}.pt`.
5. **Per-replica auto-batch:** double-gate `_cov_is_auto` (`cov_batch_size=="auto"` AND
   `auto_batch.enabled`) `:450-464`; each replica probes its OWN
   `CudaMemProbe(device)` and sizes INDEPENDENTLY (`covariance_collection.py:373-447`
   comment: "every replica probes its OWN pinned-device VRAM … NO cross-replica min").
6. **Per-sequence reduction-pin:** `InputCovarianceAccumulator.update_grouped(... seq_ids)`
   `activation_hooks.py:1032-1049` splits rows by `token_idx // seq_len` in ascending seq
   order ⇒ the finalized per-key Gram is **batch-size-invariant**, so each replica can
   auto-batch independently and the reduce stays exact.
7. **Lifecycle:** fresh `mp.spawn` per call (no persistent pool); synchronous `join()` +
   exit-code check `:1271-1282`; results flow through the **filesystem** (spill dirs).

### The merge-anchor wart (B) — exact location + correctness statement (verified)

`_remap_covariance_for_layer` (`stage2/shared_io.py:301-344`), called from
`LayerMergePlugin.write_artifacts` (`layer_merge.py:678-680`) BEFORE the cov snapshot,
remaps the post-merge survivor index. For a survivor slot that is a **merge centroid**, it
copies the **centroid's OWN** Gram `(li, centroid, name)` verbatim into the new index
(`shared_io.py:320-326`) and **drops** every non-centroid member's Gram (`:320-322`,
`if eidx not in id_to_new: continue`).

But the survivor weight is `W_merged = Σ_j b_j·perm_j(W_j)` (`merging.py:308`,
`bank.set(centroid, accs[name])` `:424`) — a centroid of the whole group. The acov
research states this precisely (`docs/research/2026-06-13-acov-capture-point.md:280-288`):

> "a survivor slot is a centroid of a merge GROUP, but the stored `A` is the **single
> original expert's** Gram copied verbatim into that slot (`stage2/shared_io.py:324-326`),
> **not** averaged over the group. … the anchor `A` is genuinely mis-attributed — the
> merged weight `W_merged` is anchored to ONE constituent's input distribution, not the
> merged slot's true (group-union) distribution. This is a real correctness wart for the
> merge arms (REAM)."

**The fix:** the survivor's anchor Gram must be the **group UNION** `A_survivor = Σ_{j∈group} G_j`
(sum of all constituents' input Grams). The Gram is `X^TX` summed over tokens, so the
union of the group's input distributions is *exactly* the sum of the per-member Grams —
no normalization, no averaging, purely additive (matches the AA-SVD anchor `A=X` semantics
which is an unnormalized Gram; Stage 3 consumes it directly). `down_proj` Grams must be
permuted to the centroid's neuron axis before summing (the same `perm` the merge used) —
see Task B2.

---

## 1. Scope / Out of scope

**In scope:**
- (A) Per-layer DP profile forward: shard-by-sequence, key-wise reduce of REAP+REAM+cov+
  distill accumulators, merged-weight re-sync to replicas per layer, per-replica auto-batch
  inherited, default-off byte-identical golden gate.
- (B) Merge-anchor union fix in `_remap_covariance_for_layer` (+ the down_proj-perm union)
  with a test proving the survivor's `A` is the group union, not one constituent.

**Out of scope (explicit):**
- Live ≥2-GPU validation (DEFERRED to GPU — see Task A8; this plan ships 1-GPU
  byte-identical + a mocked-spawn reduce test).
- TASK-PARALLEL `expert_distill` across groups (Stage-2 roadmap item 2) and DP `merge_heal`
  minibatch (item 3) — separate future work, NOT here.
- Any change to the solver layer (all CPU-bound, MG-moot per spec Q2).
- Cross-layer parallelism — ILLEGAL (§0); never propose it.
- Changing REAP semantics: in production REAP rides the vLLM sidecar
  (`reap_scores_cache.py`), so the HF forward's value is **cov + REAM**; REAP rides the
  shard for free.

---

## 2. Feature (B) FIRST — the merge-anchor union fix (independent, no multi-GPU)

Do (B) first: it is small, independent, and de-risks the golden baseline that (A) must
preserve. **TDD: write the failing test, then the fix.**

### Task B1 — Failing test: survivor anchor must be the group UNION

**File:** `max_quality/tests/test_stage2_merge_anchor_union.py` (NEW)

Construct an `InputCovarianceAccumulator` populated for a layer with three experts
(centroid `c=0`, members `m1=1`, `m2=2`) for `gate_proj` only (down_proj covered in B2),
with **distinct, known** Gram tensors `G0, G1, G2`. Call the remap with
`kept_ids=[0]` and a `grouped={0:[0,1,2]}` map (the merge group). Assert the survivor's
remapped `(layer, 0, "gate_proj")` Gram equals `G0+G1+G2` (the union), NOT `G0`.

```python
import torch
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
from moe_compress.stage2.shared_io import _remap_covariance_for_layer

def _put(cov, li, e, name, G, ntok):
    cov.covariance[(li, e, name)] = G.clone()
    cov.token_count[(li, e, name)] = ntok

def test_survivor_anchor_is_group_union_gate():
    cov = InputCovarianceAccumulator()
    G = {e: torch.full((4, 4), float(e + 1)) for e in (0, 1, 2)}
    for e in (0, 1, 2):
        _put(cov, 7, e, "gate_proj", G[e], ntok=10 * (e + 1))
    # Merge group: centroid 0 absorbs 1 and 2. Survivor kept set = [0].
    _remap_covariance_for_layer(cov, 7, kept_ids=[0], grouped={0: [0, 1, 2]})
    A = cov.covariance[(7, 0, "gate_proj")]
    assert torch.equal(A, G[0] + G[1] + G[2]), "anchor must be the group UNION Σ_j G_j"
    # token_count is the union sum too (additive denominator).
    assert cov.token_count[(7, 0, "gate_proj")] == 10 + 20 + 30

def test_non_merged_survivor_unchanged_byte_identical():
    # A protected / singleton survivor (no group) keeps its own Gram verbatim.
    cov = InputCovarianceAccumulator()
    G = torch.arange(9.0).reshape(3, 3)
    _put(cov, 3, 5, "gate_proj", G, ntok=42)
    _remap_covariance_for_layer(cov, 3, kept_ids=[5], grouped={})
    assert torch.equal(cov.covariance[(3, 0, "gate_proj")], G)
    assert cov.token_count[(3, 0, "gate_proj")] == 42
```

Run (expect FAIL on the union assert before B2/B3):
`cd max_quality && python -m pytest tests/test_stage2_merge_anchor_union.py -x -q`

### Task B2 — Failing test: down_proj union must be PERMUTED to the centroid axis

**File:** same test module.

The down_proj Gram lives on the SwiGLU intermediate-neuron axis. The merge permutes each
member's neuron axis to the centroid (`merging.py:307` `Wm[:, perm]`,
`merging.py:357-360` already permutes the member's down Gram for RegMean). The union must
sum each member's down Gram **after** applying that member's `perm` to BOTH axes, so neuron
labels align. Test: give members non-identity perms; assert the survivor down Gram equals
`G0 + perm(G1) + perm(G2)`.

```python
def test_survivor_anchor_down_is_permuted_union():
    cov = InputCovarianceAccumulator()
    d = 3
    G = {e: torch.arange(d * d, dtype=torch.float32).reshape(d, d) + 100 * e for e in (0, 1, 2)}
    for e in (0, 1, 2):
        _put(cov, 2, e, "down_proj", G[e], ntok=5)
    # Per-member neuron permutations applied by the merge (centroid perm = identity).
    perms = {0: None, 1: [2, 0, 1], 2: [1, 2, 0]}
    def permute_both(t, p):
        idx = torch.as_tensor(p, dtype=torch.long)
        return t.index_select(0, idx).index_select(1, idx)
    expected = G[0] + permute_both(G[1], perms[1]) + permute_both(G[2], perms[2])
    _remap_covariance_for_layer(
        cov, 2, kept_ids=[0], grouped={0: [0, 1, 2]}, member_perms={0: perms},
    )
    assert torch.allclose(cov.covariance[(2, 0, "down_proj")], expected)
```

> **Design note:** `_remap_covariance_for_layer` does not currently know the per-member
> down-proj perms. Two options — decide in review:
> - **B2-opt-A (preferred):** thread the perms from the merge. `_merge_experts_inplace`
>   already computes/uses them (`merging.py:280-298`, `perm_cache`). Have
>   `LayerMergePlugin.merge` capture a `{centroid: {member: perm}}` map into a ctx slot
>   (`merge_member_perms`) and `write_artifacts` pass it to the remap. This is exact.
> - **B2-opt-B (fallback):** if threading perms is deemed too invasive for v1, sum the
>   gate/up union (B1, axis-free) but **leave down_proj as the centroid's own Gram** with
>   an explicit `# WART: down union pending perm-threading` comment + a logged WARNING, and
>   a skipped xfail test. NOT preferred — it leaves half the wart. Raise to user if B2-opt-A
>   looks larger than ~40 LOC.

### Task B3 — Implement the union remap

**File:** `max_quality/src/moe_compress/stage2/shared_io.py`,
`_remap_covariance_for_layer` (`:301-344`).

Change the signature to accept the merge grouping (and, per B2-opt-A, the per-member
perms), and accumulate the union into the survivor slot instead of copying the centroid's
Gram. Sketch:

```python
def _remap_covariance_for_layer(
    cov, layer_idx, kept_ids, *, grouped=None, member_perms=None,
):
    grouped = grouped or {}
    member_perms = member_perms or {}
    id_to_new = {old: new for new, old in enumerate(kept_ids)}
    new_cov, new_tokens = {}, {}
    with cov._lock:
        # 1) pass through other layers unchanged (verbatim).
        # 2) for each kept survivor:
        for old in kept_ids:
            new = id_to_new[old]
            members = grouped.get(old)            # None ⇒ protected/singleton (byte-identical path)
            for name in ("gate_proj", "down_proj"):   # up_proj aliases gate_proj (see acc)
                key = (layer_idx, old, name)
                if members is None or len(members) <= 1:
                    # Singleton / protected — verbatim copy (NO behaviour change).
                    val = cov.covariance.get(key)
                    if val is None:
                        continue
                    new_cov[(layer_idx, new, name)] = val
                    new_tokens[(layer_idx, new, name)] = cov.token_count.get(key, 0)
                    continue
                # MERGED survivor — UNION over the group.
                acc, ntok = None, 0
                for m in members:
                    g = cov.covariance.get((layer_idx, m, name))
                    if g is None:
                        continue
                    if name == "down_proj":
                        p = member_perms.get(old, {}).get(m)
                        if p is not None:
                            idx = torch.as_tensor(p, dtype=torch.long, device=g.device)
                            g = g.index_select(0, idx).index_select(1, idx)
                    g32 = g.to(torch.float32)
                    acc = g32 if acc is None else acc + g32
                    ntok += cov.token_count.get((layer_idx, m, name), 0)
                if acc is None:
                    continue
                new_cov[(layer_idx, new, name)] = acc.to(cov.storage_dtype)
                new_tokens[(layer_idx, new, name)] = ntok
        cov.covariance, cov.token_count = new_cov, new_tokens
```

**Invariants to preserve:**
- Non-merged survivors (protected experts, singleton centroids) keep their Gram **byte-
  identical** to today (the `members is None or len<=1` branch). This is what keeps every
  existing golden green and is asserted by `test_non_merged_survivor_unchanged_byte_identical`.
- Dropped-expert logging (`shared_io.py:335-344`) preserved (recompute `n_dropped` from
  keys not landing in `new_cov`).
- `up_proj` aliases `gate_proj` in the accumulator (`InputCovarianceAccumulator._alias_gate_up`,
  `:1001-1002` writes are skipped, `get` redirects `:1310-1314`) — so only `gate_proj` +
  `down_proj` keys exist; iterating those two is complete.
- fp32-sum-then-cast-to-`storage_dtype` mirrors `finalize_layer` `:1109-1111` and the
  Stage-3 reduce `:266-280`.

### Task B4 — Thread `grouped` + `member_perms` from the merge to the remap

**Files:** `stage2/plugins/layer_merge.py` (`merge` `:543-591`, `write_artifacts`
`:635-768`); `stage2/merging.py` (`_merge_experts_inplace` `:34-424`).

- In `merge`, capture the per-`(centroid, member)` perm actually used (the cache hit
  `merging.py:291` or the Phase-B solve `merging.py:298`). Simplest: have
  `_merge_experts_inplace` return / populate a `member_perms` dict (centroid → {member →
  perm or None}); set it on `ctx` (`merge_member_perms`). Add the slot to
  `LayerMergePlugin.reads/writes`.
- In `write_artifacts`, pass `grouped=ctx.get("grouped")` and
  `member_perms=ctx.get("merge_member_perms")` to `_remap_covariance_for_layer`
  (`layer_merge.py:680`). `grouped` is already in ctx (`:557`, `:648-649`).

> **Note on existing callers:** make `grouped`/`member_perms` keyword-only with `None`
> defaults so any test/caller passing only `(cov, layer_idx, kept_ids)` gets the
> byte-identical singleton path (B3's `members is None` branch). Verify no other call site
> of `_remap_covariance_for_layer` exists: `grep -rn _remap_covariance_for_layer max_quality/src max_quality/tests`.

### Task B5 — Run the affected goldens + suite for (B)

```
cd max_quality && python -m pytest tests/test_stage2_merge_anchor_union.py \
  tests/test_stage2_cov_manifest.py tests/test_stage2_plugin_layer_merge.py \
  tests/test_pipeline_shared_io.py tests/test_smoke_stage2_resume.py -q
```

**Golden expectation:** the merged-survivor cov CHANGES (that is the fix). Any golden that
pins the **post-merge cov of a merged layer** must be regenerated and the diff inspected to
confirm it now equals the union. Goldens that pin **merge JSON / kept-ids / weights** must
stay byte-identical (the fix touches only the cov sidecar). List the regenerated goldens
explicitly in the commit body; do NOT blanket-regen.

---

## 3. Feature (A) — DP per-layer profile forward

Built ON TOP of (B). Default-off; 1-replica path is byte-identical. The hard new
requirement vs Stage-3: **the model is mutated in place each layer**, so each layer's
replicas must see the **already-merged upstream weights**.

### A0 — Architecture decision: how replicas get the merged upstream model

Stage-3 replicas reload a *fixed* student from disk once. Stage-2 cannot: the upstream
layers `0..L-1` are merged in place on the parent between layers. Two designs:

- **A0-design-1 — per-layer merged-weight re-sync (PRIMARY).** Keep a persistent replica
  pool. Each replica holds a full model copy. Before profiling layer `L`, the parent
  **broadcasts only the weights that changed since the last layer** — i.e. layer `L-1`'s
  merged expert bank + resized router (`bank.select` + `_resize_router_for_kept_experts`
  outputs). Replicas apply them to their copy, then profile. This is the "re-syncing the
  merged upstream weights to replicas each layer" cost the spec flagged
  (`stage2.md:178-179`, `:263-264`). Broadcast volume per layer = one merged MoE layer's
  expert tensors (bounded; the bulk of the model is unchanged).
- **A0-design-2 — replay-from-partial (FALLBACK / v1-simplest).** Each replica reloads the
  Stage-1 base model from disk and replays the **already-written** `merge_{0..L-1}.json` +
  `_heal_weights_layer_*.pt` partials (the parent writes these in `write_artifacts` BEFORE
  the next layer's profile) to reconstruct the merged upstream state, then profiles layer
  `L`. Requires `partial_dir` (resume mode) to be enabled. Avoids live IPC weight broadcast
  but pays a per-layer disk reload + replay (expensive; respawn-per-layer).

**Recommendation:** ship **A0-design-1** (persistent pool + per-layer delta broadcast) — it
is the only one that is cheap per layer and does not couple DP to resume-mode. But it is
the larger build (IPC weight transfer). If the review judges the IPC broadcast too risky
for v1, fall back to **A0-design-2 respawn-per-layer** which reuses the Stage-3 spawn
template almost verbatim (each layer = one `run_dp_*` call against a from-disk+replayed
model). **RAISE this decision to the user before building A4-A6** — it is the load-bearing
architectural choice and the spec explicitly leaves "persistent pool vs respawn-free" open
(`stage2.md:168-179`).

> Either design keeps the **sequential constraint**: the parent merges layer L on itself
> AND re-syncs to replicas before layer L+1; replicas never run ahead.

### A1 — `stage2_reap_ream.profile_dp` config block (default OFF)

**File:** `stage2/orchestrator.py` (run() parse block, near the other knob parses
`:1100-1215`).

Add an opt-in block mirroring Stage-3's `replicas` resolution:

```yaml
stage2_reap_ream:
  profile_dp:
    enabled: false          # master switch — OFF ⇒ serial _profile_layer, byte-identical
    replicas: auto          # "auto" ⇒ torch.cuda.device_count(); or an int
    shards_per_model: 1      # GPUs per replica (mirror stage3 shards_per_model)
  auto_batch: { enabled: false, ... }   # inherited per-replica (already parsed)
```

Parse to a small `Stage2ProfileDpConfig` (enabled, replicas resolved to int via
`torch.cuda.device_count()` when "auto", clamped ≥1). **`enabled=False` OR `replicas<=1`
⇒ the existing serial `_profile_layer` path runs unchanged** (the byte-identical gate).
Reject `enabled=True` with `replicas==1` only if you want a loud config error; otherwise
silently fall back to serial (prefer fall-back, matching Stage-3's `_shard_calib` clamp).

### A2 — Sequence-disjoint shard helper (mirror `_shard_calib`)

**File:** `stage2/profiling.py` (new module-level helper) or reuse Stage-3's `_shard_calib`
by import.

`batches` at `profiling.py:327` is an iterable of `[bs, seq_len]` tensors. The shard must
be **by sequence** (each token's full top-k set stays whole — REAP saliency, REAM grams,
cov are all per-token sums, so a sequence-disjoint partition makes every per-key reduction
exact and replica-independent). Reuse the Stage-3 contiguous dim-0 slice
(`covariance_collection.py:1052-1071`) at the **calibration-sequence** granularity (split
the underlying calib sequence list, not the pre-batched tensors), then each replica
re-batches its shard with its own (auto-)batch size. This is the same cut as Stage-3
(`shard = calib[shard_start:shard_end]`, `covariance_collection.py:1125-1126`).

> **Per-sequence reduction-pin is inherited, not rebuilt.** Cov already routes through
> `update`/`update_grouped`; for the DP path, pin cov via `update_grouped(seq_ids)` exactly
> as Stage-3 (`covariance_collection.py:657-659`) so each replica's finalized Gram is
> batch-size-invariant ⇒ replicas auto-batch independently and the reduce is exact. REAP
> and REAM numerators are per-token sums with no cross-token coupling, so they are already
> batch-invariant under a per-sequence shard (addition of the same per-sequence operands).

### A3 — Per-replica spill: serialize ALL four accumulators (not just cov)

**File:** new `stage2/profile_dp.py` (the Stage-2 analog of the Stage-3 DP driver +
worker + reduce). Stage-3 only spills `InputCovarianceAccumulator`. Stage-2 must spill +
reduce **four** additive structures per replica per layer:

| Structure | Spill payload | Reduce op |
|---|---|---|
| `InputCovarianceAccumulator` | `{covariance:{k:T}, tokens:{k:int}}` (reuse `spill_layer_to_disk` `:1135`) | fp32 key-wise sum (reuse `_reduce_spilled_cov_dirs` `:211`) |
| `ReapAccumulator` | `{sums:{k:float}, counts:{k:int}, freq:{k:int}}` (finalize_layer first → CPU floats) | sum sums, sum counts, sum freq |
| `ReamCostAccumulator._gate_gram` | `{li: [E,E] fp64}` | key-wise fp64 sum |
| `ReamCostAccumulator._sim_tensor` + `_total_tokens_by_layer` + `_neuron_act_sum/_count` | dense [E,E] fp64 + int + per-(l,e) [d_int] sums + int counts | key-wise sum (fp64 sims, int counts) |

Write new `_spill_reap_layer` / `_reduce_reap_dirs` / `_spill_ream_layer` /
`_reduce_ream_dirs` helpers (small; pattern-identical to `_reduce_spilled_cov_dirs`). The
reduce must run **per layer** (inside the sequential loop), reconstructing a single merged
set of accumulators that the assign/merge phases then consume exactly as the serial path
does. **Determinism:** process replica dirs in sorted order (Stage-3 contract
`:231`); REAM `_gate_gram`/`_sim_tensor` are fp64 so the reduce is bit-exact regardless of
order; cov + REAP are fp32 (~1e-6 drift, same class the serial path already tolerates).

> The distill-input reservoir (`_LayerInputAccumulator`) is a RESERVOIR SAMPLE, not an
> additive reduction — it is NOT trivially reducible across shards. For v1, **disable DP
> when any layer-input consumer is active** (`expert_distill_steps>0` or
> `cost_alignment=="output"` or `merge_step=="mergemoe"` — see `layer_merge.py:468-489`):
> fall back to serial profiling for the whole run with a loud INFO. Reducing a reservoir
> across shards correctly (merge by global `seen` weights) is future work. State this
> guard explicitly in A1's config validation.

### A4 — DP per-layer driver, wired into `on_profile`

**File:** `stage2/plugins/layer_merge.py` `on_profile` (`:498-538`); driver in
`stage2/profile_dp.py`.

Replace the single `_srr._profile_layer(...)` call (`layer_merge.py:528-533`) with a
DP-or-serial branch:

```python
def on_profile(self, ctx):
    if ctx.has("stage2_profile_full_hit") and ctx.get("stage2_profile_full_hit"):
        return
    layer_ref = ctx.get("layer_ref")
    if self.profile_dp.enabled and self.profile_dp.replicas > 1 and not self._dp_disabled:
        from ..profile_dp import run_dp_profile_layer
        run_dp_profile_layer(
            replica_pool=self._replica_pool,        # persistent pool (A0-design-1)
            layer_ref=layer_ref,
            shards=self._calib_shards,              # sequence-disjoint, built once
            reap_acc=ctx.get("reap_acc"),
            cov_acc=self.cov_acc,
            ream_acc=ctx.get("ream_acc"),
            device=self.device,
            auto_batch_cfg=self.auto_batch_cfg,
        )
    else:
        from .. import orchestrator as _srr
        _srr._profile_layer(self.model, layer_ref, self.batches,
                            ctx.get("reap_acc"), self.cov_acc, ctx.get("ream_acc"),
                            device=self.device, layer_input_acc=ctx.get("layer_input_acc"))
    self.cov_acc.finalize_layer(layer_ref.layer_idx)
```

`run_dp_profile_layer` (in `profile_dp.py`):
1. **Re-sync** layer `L-1`'s merged weights to replicas (A0-design-1) — skip on the first
   layer (nothing merged yet).
2. Each replica profiles its `shards[r]` through `0..L` of its (re-synced) model, building
   its own four accumulators, finalizes them per layer, spills to its per-replica dir.
3. Parent **joins**, then **reduces** the four spill dir sets into the parent's
   `reap_acc` / `cov_acc` / `ream_acc` (the very ctx instances the assign/merge phases
   read). After the reduce the parent state is identical to a serial single-GPU profile of
   the full calib set (modulo the documented fp32 drift).

The parent's own model is what gets assigned/merged (`merge` mutates `self.model`); the
re-sync in step 1 propagates that merge to replicas before the NEXT layer. This keeps the
sequential constraint exact.

### A5 — Per-replica auto-batch (inherited verbatim, NOT rebuilt)

Each replica pins itself via `CUDA_VISIBLE_DEVICES` (Stage-3 `:1096`) so
`CudaMemProbe(device)` sees its OWN free VRAM. The replica's batch loop uses
`size_batch(cost_probe_fn, floor=1, headroom_frac, max_cap, mem=CudaMemProbe(device))` +
`run_with_oom_backoff(..., floor=1)` from `utils/auto_batch.py` (`:143-203`), gated by the
inherited `auto_batch.enabled`. **No new auto-batch logic.** The per-sequence reduction-pin
(A2) makes the finalized per-key reductions batch-size-invariant, so each replica sizes
INDEPENDENTLY with NO cross-replica min-agreement — identical to the Stage-3 contract
(`covariance_collection.py:373-447`). Default `auto_batch.enabled=false` ⇒ each replica
uses the fixed batch ⇒ no probe.

### A6 — Persistent replica pool lifecycle (A0-design-1) OR respawn (A0-design-2)

**File:** `stage2/profile_dp.py` + `LayerMergePlugin.__init__`/teardown.

- **A0-design-1:** build the pool once (lazily on first DP `on_profile`), tear it down at
  run end (a `finally` in `orchestrator.run` after the layer loop, mirroring Stage-3's
  `procs.join()` discipline `:1277-1282`). The pool holds N spawned processes, each with a
  resident model copy; the parent communicates per-layer via a `mp` queue/pipe (re-sync
  weights down, "done" up) + filesystem spill (accumulators up). Ensure clean teardown on
  exception (the `verify-teardown` discipline — no leaked child processes).
- **A0-design-2:** no pool; each layer calls a `run_dp_profile_layer` that spawns N
  short-lived workers (Stage-3 template verbatim), each reloading base+replaying partials.
  Simpler lifecycle, higher per-layer cost.

### A7 — Tests for (A) (no live multi-GPU — mocked spawn + reduce-math)

**File:** `max_quality/tests/test_stage2_profile_dp.py` (NEW).

1. **Shard correctness:** `_shard_calib`-equivalent splits N sequences into R disjoint
   contiguous shards covering every sequence exactly once (mirror Stage-3 shard test).
2. **Reduce math — REAP:** build two `ReapAccumulator`s with disjoint per-(l,e) sums/counts;
   spill + reduce; assert merged `sums/counts/freq` == element-wise sums and
   `score()==Σsum/Σcount` (the mean-at-read identity).
3. **Reduce math — REAM gate_gram:** two `_gate_gram[li]` fp64 tensors; reduce; assert
   bit-exact sum; assert `compute_gate_similarity_matrix` on the merged Gram == the matrix
   from a single-accumulator that saw both shards' logits (fp64 ⇒ bit-exact).
4. **Reduce math — REAM sim_tensor + total_tokens + neuron means:** disjoint shards; assert
   `_sim_tensor` sum (fp64 bit-exact), `_total_tokens_by_layer` int sum, `_neuron_act_sum/_count`
   sums; assert `compute_delta_expert`/`get_neuron_mean` equal the single-pass values.
5. **End-to-end equivalence (mocked, 2 fake replicas, CPU):** monkeypatch the replica
   "spawn" to run two shards **in-process on CPU** against a tiny stub MoE model; reduce;
   assert the merged `reap_acc/cov_acc/ream_acc` match a serial `_profile_layer` over the
   full calib set within `allclose` (fp32 cov/REAP ~1e-5; fp64 REAM bit-exact). Use the
   existing `test_stage2_pipeline_run_layer.py` stub model as the fixture.
6. **Byte-identical default gate:** with `profile_dp.enabled=false`, assert
   `on_profile` calls the serial `_profile_layer` path (no DP import/spawn) — pin via a
   spy. This is the golden guardrail proving default-off is a no-op.
7. **Layer-input-consumer guard:** with `expert_distill_steps>0` AND
   `profile_dp.enabled=true`, assert DP is disabled (serial fallback) + the loud INFO
   (A3 reservoir guard).

```
cd max_quality && python -m pytest tests/test_stage2_profile_dp.py -q
```

### A8 — Out of scope: live ≥2-GPU validation (DEFERRED)

Real ≥2-GPU run on a real model is DEFERRED to GPU (no box in this plan). The 1-GPU
default-off path is byte-identical (A7.6); the reduce math is proven in-process (A7.2-5);
the first live ≥2-GPU run validates the IPC re-sync + spawn lifecycle. Document this in the
commit body and in `profile_dp.py`'s module docstring (mirror the Stage-3/4 multigpu
"CAVEAT untested on real ≥2-GPU" memory note).

### A9 — Full-suite + whole-impl review before claiming green

```
cd max_quality && python -m pytest tests/ -q -k "stage2 or cov or profil or shared_io or merge or pipeline"
```
Then the per-task review → fixer ping-pong (all 5 categories incl. nitpick) → all-none, and
a whole-implementation review + the full caller/integration suite (the
`run_full_suite_and_final_review` memory rule). Confirm no other `_profile_layer` /
`_remap_covariance_for_layer` caller regressed.

---

## 4. Task list (ordered, bite-sized, TDD)

| # | Task | Files | Gate |
|---|---|---|---|
| **B1** | Failing test: survivor anchor = group UNION (gate_proj) | `tests/test_stage2_merge_anchor_union.py` (NEW) | test FAILS pre-fix |
| **B2** | Failing test: down_proj union PERMUTED to centroid axis | same | test FAILS pre-fix |
| **B3** | Implement union remap (fp32-sum-then-cast; singleton byte-identical branch) | `stage2/shared_io.py:301-344` | B1/B2 pass; singleton test green |
| **B4** | Thread `grouped`+`member_perms` from merge → remap (kw-only, None default) | `plugins/layer_merge.py:543-591,635-768`; `merging.py:34-424` | no other caller breaks |
| **B5** | Regenerate ONLY merged-layer cov goldens; inspect diff; suite | (goldens) | merge JSON/weights byte-identical; cov = union |
| **A0** | DECISION: design-1 (delta broadcast pool) vs design-2 (replay-from-disk) — **RAISE to user** | (doc) | user picks |
| **A1** | `profile_dp` config block (default OFF) + layer-input-consumer guard | `stage2/orchestrator.py` | default-off parses to serial |
| **A2** | Sequence-disjoint shard helper (reuse Stage-3 `_shard_calib`) + per-seq cov pin | `stage2/profiling.py`/`profile_dp.py` | shard test green |
| **A3** | Per-replica spill + per-layer reduce of ALL FOUR accumulators | `stage2/profile_dp.py` (NEW) | reduce-math tests green |
| **A4** | DP-or-serial branch in `on_profile`; `run_dp_profile_layer` | `plugins/layer_merge.py:498-538`; `profile_dp.py` | e2e mocked equivalence |
| **A5** | Per-replica auto-batch inherited (CUDA_VISIBLE_DEVICES + size_batch + backoff) | `profile_dp.py` | per-replica independent sizing |
| **A6** | Pool lifecycle + clean teardown (or respawn per A0-design-2) | `profile_dp.py`; `orchestrator.run` finally | no leaked children |
| **A7** | DP tests (shard, 4× reduce-math, e2e mocked, byte-identical gate, reservoir guard) | `tests/test_stage2_profile_dp.py` (NEW) | all green |
| **A8** | Out-of-scope note: live ≥2-GPU deferred | `profile_dp.py` docstring + commit | documented |
| **A9** | Full suite + whole-impl review loop to all-none | (all) | green + reviewed |

---

## 5. Key decisions (for the reviewer)

1. **Per-layer DP, never cross-layer.** The merge mutates the live model in place
   (`merging.py:424`) and layer L+1 forwards through layer L's merged weights
   (`profiling.py:338-342`). Cross-layer parallelism is −1.0 AVG (REAM §5.4). The DP axis
   is strictly **within** each layer's forward, inside the sequential loop.
2. **Merged-weight re-sync per layer is the new cost** (vs Stage-3's once-from-disk).
   A0-design-1 (persistent pool + per-layer delta broadcast) is primary; A0-design-2
   (respawn + replay-from-partial) is the simpler fallback. **This is the one decision to
   RAISE to the user before building A4-A6.**
3. **Shard by sequence** (`_shard_calib` contiguous dim-0 slice, token-disjoint). Every
   accumulator is a per-token additive Σ with a separate additive count ⇒ the reduce is
   `Σ_r num_r / Σ_r count_r`, exact and replica-independent. Per-sequence cov pin
   (`update_grouped`) inherited from Stage-3 ⇒ replicas auto-batch independently, no
   cross-replica min.
4. **Reduce all FOUR accumulators per layer** (cov + REAP + REAM-gate-gram + REAM-sim/
   neuron), not just cov. REAM fp64 ⇒ bit-exact; cov/REAP fp32 ⇒ ~1e-6 (already-tolerated
   class). Spill-dir + sorted-order reduce mirrors `_reduce_spilled_cov_dirs`.
5. **Distill-input reservoir is NOT additively reducible** ⇒ v1 disables DP when a
   layer-input consumer is active (loud serial fallback). Reservoir-merge across shards is
   future work.
6. **(B) merge-anchor union** = `Σ_{j∈group} G_j`, with `down_proj` Grams permuted to the
   centroid neuron axis (same perm the merge used). The Gram is an unnormalized `X^TX`, so
   the union is exactly the additive sum — matching AA-SVD's anchor `A=X` semantics (Stage 3
   consumes it directly). Non-merged survivors stay byte-identical.
7. **Default-off byte-identical everywhere.** `profile_dp.enabled=false` ⇒ serial
   `_profile_layer` (A7.6 spy test). (B) changes only the cov of *merged* layers; merge
   JSON / weights / singleton cov stay byte-identical (B5).

---

## 6. Open questions (resolve in review / with user)

1. **A0 design choice** (delta-broadcast pool vs respawn-from-disk replay) — RAISE to user.
   Drives the size/risk of A4-A6. Spec leaves it open (`stage2.md:168-179`).
2. **B2 down_proj perms:** thread the exact merge perms (B2-opt-A, preferred, exact) or
   ship gate/up-only union + xfail down (B2-opt-B). If perm-threading exceeds ~40 LOC,
   raise before committing to A.
3. **REAM `_gate_gram` non-determinism under reduce:** it is fp64 (`activation_hooks.py:178`)
   so the key-wise sum is order-independent and bit-exact. Confirm the cov + REAP fp32
   ~1e-6 drift is acceptable for the DP-on path (it is NOT on the byte-identical default,
   which has 1 replica and no reduce). Same drift class the serial path already accepts
   (`activation_hooks.py:308-311`).
4. **`shards_per_model` > 1** (tensor-sharded replica): mirror Stage-3's `shards_per_model`
   knob but is it needed for Stage-2 model sizes? Default 1; defer multi-GPU-per-replica.
5. **Resume interaction:** DP profiling writes the same per-layer partials
   (`write_artifacts`); confirm a DP run and a serial resume are interchangeable (the
   reduced accumulators feed the identical `write_artifacts`). Add a resume smoke if cheap.
