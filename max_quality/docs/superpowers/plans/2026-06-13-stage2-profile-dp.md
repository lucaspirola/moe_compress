# Stage 2 — persistent-pool DP profile forward (structural replay) + merge-anchor union fix

**Branch:** `feat/stage2-profile-dp` · **Worktree:** `/home/lucas/ai/wt-s2mg` · **Code root:** `max_quality/`
**Status:** PLAN (do not implement from this doc without the review loop).
**Date:** 2026-06-13 · **Rev 2** (post plan-review CHANGES REQUESTED — A0 resolved to design-1, C1/H1/H2/H3 + B-polish folded in)

Two features, both Stage 2, bundled in this worktree:

- **(A)** Data-parallel the Stage-2 per-layer profiling forward (the bottleneck) via a
  **PERSISTENT worker pool** spawned once at Stage-2 start, driven per layer over a
  command/reduce IPC channel. Shard calibration **by sequence**, key-wise reduce the four
  additive accumulators, normalize once. RESULT-PRESERVING, opt-in, default byte-identical.
- **(B)** Fix the REAM **merge-anchor wart**: the post-merge covariance remap copies the
  **centroid's own** input Gram into the survivor slot instead of the **group-UNION**
  (`Σ_j G_j`). Real correctness bug surfaced by the acov research, independent of multi-GPU.

> **Effort re-estimate (A0 resolved):** Feature A is **XL / high-risk greenfield**. There
> is **no persistent-pool / live-IPC template in this repo** — Stage-3 spawns *fresh*
> workers per call and reduces via disk (`covariance_collection.py:1271-1289`), it never
> keeps workers alive or sends them live commands. Feature A builds a **new IPC subsystem**
> (persistent pool + per-layer parent→worker command channel + worker→parent reduce
> channel + structural-replay re-sync + lifecycle/teardown). Feature B is **S / low-risk**
> and lands first to de-risk the baseline.

---

## 0. Load-bearing facts (cited + verified against the actual code)

Spec: `max_quality/docs/multigpu_analysis/stage2.md`. Verified below.

### The sequential-merge constraint is LOAD-BEARING — no cross-layer parallelism

Driver loop `stage2/orchestrator.py:1673-1712` processes MoE layers **strictly in order**:
profile → assign (`_run_assignment`) → merge (`_merge_experts_inplace`,
`merging.py:423-424` `bank.set`) → post_merge (`bank.select` + router resize,
`layer_merge.py:624-627`). The merge **mutates the live model in place**; layer `L+1`'s
profile forwards through `0..L` of *that mutated model* (`profiling.py:338-342` under
`early_exit_after_layer(model, layer_idx)`). REAM §4 sequential merging
(`layer_merge.py:237-266`).

> **Task-parallelism ACROSS layers is ILLEGAL (−1.0 AVG, REAM §5.4). The plan MUST NOT
> propose it.** The only exploitable axis is DATA-PARALLEL **within each layer's profile
> forward**, applied PER LAYER inside the sequential loop. Every task respects this.

### The forward is the bottleneck and is exactly DP-able

One calibration forward per layer: `_profile_layer` (`profiling.py:169`), an
`instrument_experts`-hooked early-exit `model(input_ids=batch)`
(`profiling.py:327-348`), co-producing REAP + REAM + cov + distill-input in one pass.
820 layer-forwards across the 40 sequential passes (`profiling.py:190-191`).

### All FOUR accumulators are additive Σ-over-tokens + separate additive count (verified)

| Accumulator | Numerator (additive) | Count (additive) | Read |
|---|---|---|---|
| **REAP** `add_gpu` | `_gpu_sums[k].add_(contrib)` `activation_hooks.py:885-887`; `contrib=(gate·‖f‖).sum()` `:1433-1434` | `counts[k]+=n`, `freq[k]+=n` `:888-889` | `score()=s/n` `:923-930` |
| **REAM δ_gate** `record_router_logits` | `_gate_gram[li].add_(bᵀb)` (**fp64**) `:178-186` | (Gram diag = ‖v‖²) | `compute_gate_similarity_matrix` `:464` |
| **REAM δ̃_expert** `finalize_batch` | `_sim_tensor[li].add_(sim_sum_f64)` (**fp64**) `:459-462` | `_total_tokens_by_layer[li]+=n` `:235` | `compute_delta_expert=sim/total` `:586` |
| **REAM C_act** `record_neuron_activations` | `_neuron_act_sum[k]+=batch_sum` `:604-608` | `_neuron_act_count[k]+=n` `:609` | `get_neuron_mean=s/c` `:619` |
| **Input cov** `update` | `_pending[k].add_(cov)` (**fp32**) `:1024-1029`, cov=`flatᵀ@flat` | `_gpu_token_count[k]+=n` `:1030` | `finalize_layer` sums+casts `:1082-1112` |

Every numerator is a linear sum over tokens; every count a linear sum over tokens. For a
**sequence-disjoint** shard set the reduce is exactly `Σ_r num_r / Σ_r count_r`, mean once
after the reduce. REAM grams are **fp64 ⇒ the reduce is bit-exact regardless of order**;
cov + REAP are fp32 ⇒ ~1e-6 drift, the same class the serial path already tolerates
(`activation_hooks.py:308-311`) — and absent on the byte-identical 1-replica default.

### Stage-3 DP template to mirror (DISK-reduce only; NOT a persistent pool)

`stage3/plugins/covariance_collection.py`: driver
`run_dp_covariance_collection(...replicas...)` `:1213-1289`; shard `_shard_calib`
`:1052-1071` (contiguous dim-0 / by-sequence, token-disjoint); worker `_cov_replica_worker`
`:1074-1210` (pins via `CUDA_VISIBLE_DEVICES` `:1096`, **reloads model from disk** `:1119`,
spills per-layer); reduce `_reduce_spilled_cov_dirs` `:211-289` (fp32 key-wise sum, sorted
dirs); per-replica auto-batch double-gate `_cov_is_auto` `:450-464`; per-seq pin
`update_grouped` `activation_hooks.py:1032-1049`. **What we reuse:** the shard math, the
per-key disk-spill reduce, the per-replica auto-batch wiring, the per-seq pin. **What we
must BUILD NEW (no template):** persistent pool, live per-layer command channel, structural
re-sync — see C1/A0.

---

## 1. CRITICAL — C1: the per-layer re-sync is STRUCTURAL SURGERY, not a value delta

**The Rev-1 "delta broadcast" framing was WRONG.** Verified, the per-layer merge changes
TENSOR SHAPES and REPLACES Parameter objects — a worker cannot value-copy into its resident
tensors because those tensors change shape and identity:

- `merging.py:424` `bank.set(centroid, accs[name])` — writes merged centroid VALUES into the
  stacked expert tensor (value change, still pre-select shape).
- `layer_merge.py:626` `bank.select(final_kept_ids)` → `merging`/`model_io` **SLICES the
  stacked expert tensor down to a smaller SHAPE** (n_experts → n_kept rows).
- `merging.py:435-440` `_resize_router_for_kept_experts`: `router.weight =
  nn.Parameter(router.weight.data.index_select(0, idx)...)` **REPLACES the Parameter
  object** + mutates `router.num_experts` (`:440`), `router.top_k` (`:442-443`),
  `mlp.num_experts` (`:446-447`).

**FIX — structural replay (not delta broadcast).** Each layer, the parent sends each worker
the *recipe* to reproduce the structural mutation on its own resident copy:

1. parent merges layer `L` on itself (the normal serial merge/post_merge),
2. parent broadcasts to every worker: `final_kept_ids`, `grouped`, and the **merged
   centroid tensors** for layer `L` (`{name: bank.get(centroid)}` for each centroid, the
   post-`bank.set` pre-`select` values — or equivalently the post-select kept tensors),
3. each worker **REPLAYS the same structural ops** on its model copy:
   `_merge_experts_inplace` is NOT re-run (no profile data on the worker); instead the
   worker (a) patches the centroid bank rows with the broadcast merged values, then (b) runs
   `bank.select(final_kept_ids)` + `_resize_router_for_kept_experts(layer_ref,
   final_kept_ids)` — the *identical* structural surgery the parent ran. After replay the
   worker's layer `L` is shape- and value-identical to the parent's.

> **Simplification to consider in review (RAISE):** instead of "patch centroid values then
> replay select+resize", broadcast the **already-merged, already-selected kept tensors**
> (parent's post-`post_merge` `bank.get(pos)` for every kept position + the resized
> `router.weight`/`bias`) and have the worker `bank.set` + replace the router Parameter +
> set `num_experts`/`top_k` directly. This is the same wire volume (one merged MoE layer)
> and skips re-deriving the slice on the worker — strictly simpler and less divergence-
> prone. Recommend this variant; either is structural replay, NOT a value delta.
>
> **Drop all "delta broadcast" language.** The re-sync is broadcasting a structurally-merged
> layer + replaying the shape change.

The bulk of the model (all other layers) is unchanged each step, so the per-layer wire
volume is one merged MoE layer's tensors — bounded, but this is a **live mp IPC transfer**
(new subsystem), not a disk reload.

---

## 2. Feature (B) FIRST — merge-anchor union fix (independent, S/low-risk)

TDD: failing tests, then the fix. Lands first; de-risks the baseline A must preserve.

### Task B1 — Failing test: survivor anchor = group UNION (gate_proj)

**File:** `max_quality/tests/test_stage2_merge_anchor_union.py` (NEW).

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
    _remap_covariance_for_layer(cov, 7, kept_ids=[0], grouped={0: [0, 1, 2]})
    A = cov.covariance[(7, 0, "gate_proj")]
    assert torch.equal(A, G[0] + G[1] + G[2]), "anchor must be group UNION Σ_j G_j"
    assert cov.token_count[(7, 0, "gate_proj")] == 10 + 20 + 30

def test_non_merged_survivor_unchanged_byte_identical():
    cov = InputCovarianceAccumulator()
    G = torch.arange(9.0).reshape(3, 3)
    _put(cov, 3, 5, "gate_proj", G, ntok=42)
    _remap_covariance_for_layer(cov, 3, kept_ids=[5], grouped={})
    assert torch.equal(cov.covariance[(3, 0, "gate_proj")], G)
    assert cov.token_count[(3, 0, "gate_proj")] == 42
```

`cd max_quality && python -m pytest tests/test_stage2_merge_anchor_union.py -x -q` → FAIL.

### Task B2 — Failing test: down_proj union PERMUTED to centroid axis (B2-opt-A, the ONLY option)

**B2-opt-B (gate/up-only fallback) is DROPPED** — it would leave half the wart. We commit
to exact perm threading. The down_proj Gram is on the SwiGLU intermediate-neuron axis; the
merge permutes each member's neuron axis to the centroid (`merging.py:307` `Wm[:, perm]`).
The union must permute **BOTH axes** of each member's down Gram by **the exact `perm` the
merge used** before summing — mirroring the RegMean down-Gram permutation already in the
code (`merging.py:352-360`, `G_down_m.index_select(0, perm_t).index_select(1, perm_t)`).

```python
def test_survivor_anchor_down_is_permuted_union():
    cov = InputCovarianceAccumulator()
    d = 3
    G = {e: torch.arange(d * d, dtype=torch.float32).reshape(d, d) + 100 * e for e in (0, 1, 2)}
    for e in (0, 1, 2):
        _put(cov, 2, e, "down_proj", G[e], ntok=5)
    perms = {0: None, 1: [2, 0, 1], 2: [1, 2, 0]}   # centroid perm = None (identity)
    def pb(t, p):
        idx = torch.as_tensor(p, dtype=torch.long)
        return t.index_select(0, idx).index_select(1, idx)
    expected = G[0] + pb(G[1], perms[1]) + pb(G[2], perms[2])
    _remap_covariance_for_layer(
        cov, 2, kept_ids=[0], grouped={0: [0, 1, 2]}, member_perms={0: perms},
    )
    assert torch.allclose(cov.covariance[(2, 0, "down_proj")], expected)
```

### Task B3 — Implement the union remap

**File:** `stage2/shared_io.py`, `_remap_covariance_for_layer` (`:301-344`). Signature gains
**keyword-only `grouped=None, member_perms=None`** (None ⇒ today's verbatim singleton path,
keeping `test_stage2_merge.py:17-32` green — see H2). Sketch:

```python
def _remap_covariance_for_layer(cov, layer_idx, kept_ids, *, grouped=None, member_perms=None):
    grouped = grouped or {}
    member_perms = member_perms or {}
    id_to_new = {old: new for new, old in enumerate(kept_ids)}
    new_cov, new_tokens = {}, {}
    with cov._lock:
        # pass through OTHER layers verbatim (unchanged from :316-318)
        for key, val in list(cov.covariance.items()):
            if key[0] != layer_idx:
                new_cov[key] = val
                new_tokens[key] = cov.token_count.get(key, 0)
        for old in kept_ids:
            new = id_to_new[old]
            members = grouped.get(old)
            for name in ("gate_proj", "down_proj"):     # up_proj aliases gate_proj
                key = (layer_idx, old, name)
                if not members or len(members) <= 1:     # singleton/protected — byte-identical
                    val = cov.covariance.get(key)
                    if val is None:
                        continue
                    new_cov[(layer_idx, new, name)] = val
                    new_tokens[(layer_idx, new, name)] = cov.token_count.get(key, 0)
                    continue
                acc, ntok = None, 0                       # MERGED — UNION Σ_j G_j
                for m in members:
                    g = cov.covariance.get((layer_idx, m, name))
                    if g is None:
                        continue
                    if name == "down_proj":
                        p = member_perms.get(old, {}).get(m)
                        if p is not None:
                            idx = torch.as_tensor(p, dtype=torch.long, device=g.device)
                            g = g.index_select(0, idx).index_select(1, idx)   # BOTH axes
                    g32 = g.to(torch.float32)
                    acc = g32 if acc is None else acc + g32
                    ntok += cov.token_count.get((layer_idx, m, name), 0)
                if acc is None:
                    continue
                new_cov[(layer_idx, new, name)] = acc.to(cov.storage_dtype)
                new_tokens[(layer_idx, new, name)] = ntok
        cov.covariance, cov.token_count = new_cov, new_tokens
    # preserve the dropped-key WARNING (:335-344): recompute from keys absent in new_cov
```

Invariants: singleton/protected byte-identical (B1 second test + `test_stage2_merge.py:17-32`);
fp32-sum-then-cast mirrors `finalize_layer:1109-1111`; `up_proj` aliases gate so iterating
`{gate_proj, down_proj}` is complete (`activation_hooks.py:1001-1002,1310-1314`); keep the
orphan-token + dropped-expert logging.

### Task B4 — Thread `grouped` + EXACT `member_perms` from the merge to the remap

**Files:** `stage2/merging.py` (`_merge_experts_inplace`), `stage2/plugins/layer_merge.py`
(`merge` `:543-591`, `write_artifacts:680`).

- **Capture the EXACT perm the merge used**, NOT a recomputed one. At `merging.py:280/291/298`
  the merge already binds `perm` per member (`None` for centroid `:280`, `cached[0]` on
  perm-cache hit `:291`, `_miss_perms[m]` otherwise `:298`). Collect these into
  `member_perms[centroid][m] = perm` inside the existing member loop and have
  `_merge_experts_inplace` return (or out-param) the `{centroid: {member: perm}}` map. Use
  the SAME object the weight permutation consumed (`:306-307`) so the Gram axis-permutation
  is provably consistent with the weight axis-permutation.
- `LayerMergePlugin.merge` stores it on ctx (`merge_member_perms`); add the slot to
  `reads/writes`. `write_artifacts` passes `grouped=ctx.get("grouped")` (already present
  `:557,:648-649`) and `member_perms=ctx.get("merge_member_perms")` to
  `_remap_covariance_for_layer` (`layer_merge.py:680`).
- **kw-only None defaults** ⇒ the single other lexical reference is the call at `:680`
  (verified: `grep -rn _remap_covariance_for_layer max_quality/src max_quality/tests` →
  one production caller + the test). Existing tests passing only `(cov, li, kept_ids)` hit
  the byte-identical singleton path.

### Task B5 — Run the REAL Stage-2 guardrails (H2: there is NO stage2 cov golden to regen)

The Rev-1 "regenerate merged-layer cov goldens" step is **DELETED** — `test_stage2_golden_snapshot.py`
and `golden/stage2` **do not exist**. The real guardrails (baseline **36 passed**):

```
cd max_quality && python -m pytest \
  tests/test_stage2_merge_anchor_union.py \
  tests/test_stage2_merge.py \
  tests/test_stage2_cov_manifest.py \
  tests/test_stage2_shared_io.py \
  tests/test_stage2_plugin_layer_merge.py \
  tests/test_smoke_stage2_resume.py -q
```

B's correctness is pinned by the NEW `test_stage2_merge_anchor_union.py` PLUS
`test_stage2_merge.py::test_remap_covariance_keeps_only_centroids` (`:17-32`) staying green
(it calls remap with NO `grouped` ⇒ singleton path ⇒ the kw-only default handles it — that
test is the compat anchor; do NOT modify it).

### H3 — Resume path needs NO change (positive)

The resume loader at `orchestrator.py:1051` does `cov_acc.load_layer_from_disk(layer_idx,
partial_dir)` — it reads the **already-remapped union cov** off disk (written by
`write_artifacts` → `_snapshot_cov_layer` AFTER the remap, `layer_merge.py:680-683`). Since
the union is computed at write time, resume is consistent **for free**: a resumed run loads
the union, a fresh run computes the union, and they match. Record this in B4 + Q5.

---

## 3. Feature (A) — persistent-pool DP per-layer profile forward (XL / high-risk)

Built ON TOP of (B). Default-off; the 1-replica path is byte-identical (the serial
`_profile_layer`). **A0 RESOLVED: design-1 (persistent pool + per-layer structural re-sync).**

### A0 — Lifecycle of the persistent worker pool + IPC channels (NEW subsystem)

No template exists; design it explicitly. **File:** `stage2/profile_dp.py` (NEW).

**Spawn ONCE at Stage-2 start** (lazily on first DP `on_profile`, or eagerly in
`orchestrator.run` right after model load). `mp.get_context("spawn")`; N workers, each:
pins via `CUDA_VISIBLE_DEVICES` (Stage-3 `:1096`), **loads its OWN model copy from disk**
(`config["model"]["name_or_path"]` / the Stage-1 artifact path — same source the parent
loaded), enters a **command loop** blocking on its command channel.

**Two channels per worker:**
- **Command channel (parent→worker):** an `mp.Queue` (or `Pipe`) per worker carrying typed
  messages: `RESYNC(layer_idx, final_kept_ids, grouped, merged_layer_tensors)` (structural
  replay, C1), `PROFILE(layer_idx, shard_id, auto_batch_cfg)` (run the early-exit forward
  on this worker's shard, spill the four accumulators to the worker's per-replica dir),
  `SHUTDOWN`. Workers process commands in order.
- **Reduce channel (worker→parent):** an `mp.Queue` carrying `DONE(layer_idx, shard_id,
  spill_dir)` / `ERROR(layer_idx, traceback)`. Large accumulator payloads travel via the
  **filesystem spill dir** (Stage-3 contract), NOT serialized through the queue — the queue
  carries only the "spill ready" signal + the dir path. This keeps the IPC small and reuses
  the proven disk-reduce. (Worker model weights load from disk; only small structured
  control messages cross the queues, and the broadcast merged-layer tensors in RESYNC are
  plain `torch.save`/`torch.load` to a per-layer scratch file referenced by path — never a
  raw serialized object through the queue.)

**Per-layer protocol (inside the sequential loop):**
1. parent does layer `L-1` merge/post_merge on itself (normal),
2. parent sends `RESYNC(L-1, ...)` to all workers, **waits for all ACKs** (structural replay
   done — workers now have the merged upstream),
3. parent sends `PROFILE(L, shard_r, ...)` to worker `r`,
4. workers profile their shard through `0..L`, finalize their four accumulators per layer,
   spill to per-replica dirs, send `DONE`,
5. parent **joins on the reduce channel** (all `DONE`), then **reduces** the four spill-dir
   sets into the parent's `reap_acc`/`cov_acc`/`ream_acc` (A3). Assign/merge run on the
   parent exactly as serial.

**Teardown at Stage-2 end** (a `finally` in `orchestrator.run` after the layer loop): send
`SHUTDOWN`, `join()` every worker with a timeout, check `exitcode==0` for each (mirror
Stage-3 `:1277-1282`), `terminate()` + raise on any non-zero/timeout. **Verify no leaked
children** (the verify-teardown discipline). Any worker `ERROR` mid-run → parent aborts the
run, tears the pool down, re-raises with the worker traceback (no silent partial result).

> **Sequential constraint preserved:** step 2's RESYNC-then-ACK barrier means no worker
> profiles layer `L` until every worker has the merged layer `L-1`. Workers never run ahead.

### A1 — `stage2_reap_ream.profile_dp` config (default OFF) + reservoir guard AT RESOLUTION

**File:** `stage2/orchestrator.py` (knob-parse block `:1100-1215`).

```yaml
stage2_reap_ream:
  profile_dp: { enabled: false, replicas: auto, shards_per_model: 1 }
  auto_batch: { enabled: false }   # inherited per-replica (already parsed)
```

`enabled=False` OR resolved `replicas<=1` ⇒ serial `_profile_layer`, byte-identical.
`replicas: auto` → `torch.cuda.device_count()`, clamped ≥1.

**Reservoir guard fires HERE (config resolution), with `log.warning` naming the consumer —
NOT 40 layers deep.** The distill-input reservoir (`_LayerInputAccumulator`) is a RESERVOIR
SAMPLE, not additive, so it is NOT reducible across shards. When ANY layer-input consumer is
active — `expert_distill_steps > 0` OR `cost_alignment == "output"` OR `merge_step ==
"mergemoe"` (the exact disjunction at `layer_merge.py:468-472`) — DP profiling is **disabled
for the whole run** with a single `log.warning` at resolution time naming which consumer
forced the fallback. Implement as: if `profile_dp.enabled and (those consumers)` → set
`profile_dp.enabled=False`, warn once, proceed serial. (Reservoir-merge across shards by
global-`seen` weighting is future work.)

### A2 — Sequence-disjoint shard (reuse Stage-3 `_shard_calib`) + cov per-seq pin (see H1)

`batches` at `profiling.py:327` are `[bs, seq_len]` tensors. Shard **by sequence** (each
token's full top-k set stays whole ⇒ REAP/REAM/cov per-key sums are exact and replica-
independent). Reuse Stage-3's contiguous dim-0 slice (`covariance_collection.py:1052-1071`)
at the **calibration-sequence** granularity (split the calib sequence list, not pre-batched
tensors); each worker re-batches its shard with its own (auto-)batch size. Built ONCE at
pool spawn and indexed by `shard_id`.

### H1 — NEW plumbing: cov per-sequence pin is NOT inherited; it must be added

The live cov path uses `cov_acc.update` (`profiling.py:230` `input_cb`, `:236`
`intermediate_cb`), **NOT `update_grouped`**. For A5's premise (each replica auto-batches
INDEPENDENTLY) the cov reduction must be **batch-invariant**, which requires the per-sequence
pin (`update_grouped(..., seq_ids)`). REAP/REAM numerators are genuinely batch-invariant
(per-token sums, no cross-token coupling) — **only cov needs the pin.** This is NEW code:

- **`input_cb` has NO `token_idx` today.** `down_cb`/REAP have `ctx["token_idx"]`
  (`profiling.py:248`; supplied by `instrument_experts` `ctx={... "token_idx": token_idx}`
  `activation_hooks.py:1512-1513,1551-1552`), but `input_cb`/`intermediate_cb` receive only
  `(li, e, tensor, ctx)` and the current bodies ignore `ctx`. Both `input` and
  `intermediate` ARE called with the SAME `ctx` that carries `token_idx`
  (`_cb("input", ...)` / `_cb("intermediate", ...)` fire inside the per-expert loop that
  already has `token_idx`, `activation_hooks.py:1514,1524 / 1553,1559`).
- **New task:** in `profiling.py`, when DP is active, thread a `seq_len` into `_profile_layer`
  and switch `input_cb`/`intermediate_cb` to:
  ```python
  def input_cb(li, e, tensor, ctx):
      seq_ids = ctx["token_idx"] // seq_len if seq_len else None
      cov_acc.update_grouped(li, e, "gate_proj", tensor, seq_ids)
  def intermediate_cb(li, e, tensor, ctx):
      seq_ids = ctx["token_idx"] // seq_len if seq_len else None
      cov_acc.update_grouped(li, e, "down_proj", tensor, seq_ids)
      ream_acc.record_neuron_activations(li, e, tensor)   # unchanged (additive, batch-invariant)
  ```
  `update_grouped` (`activation_hooks.py:1032-1049`) splits rows by ascending `seq_ids` and
  reduces to plain `update` calls per sequence ⇒ the finalized per-key Gram is
  batch-size-invariant. **Guard:** verify `token_idx` indexes the same rows as `tensor`
  (both are the per-expert active-token view — `sel = hidden_states[token_idx]`
  `activation_hooks.py:1511,1550`, so `tensor` rows ↔ `token_idx` 1:1). On the SERIAL path
  keep plain `update` (default `seq_len=None` ⇒ `update_grouped` calls `update` once,
  byte-identical) so the non-DP golden is untouched.

> **NIT to confirm in review:** `update_grouped` with a real multi-seq split changes the
> fp32 reduction grouping vs a single `update` over the whole batch — that is the *point*
> (it pins the grouping to bs=1 order). On the SERIAL default we pass `seq_len=None` ⇒ exactly
> one `update` ⇒ byte-identical. The DP path's per-seq grouping is the batch-invariant
> reference (matches Stage-3's cov pin rationale, `covariance_collection.py:373-447`).

### A3 — Per-replica spill + per-layer reduce of ALL FOUR accumulators

**File:** `stage2/profile_dp.py`. Stage-3 spills only cov; Stage-2 spills + reduces four
additive structures per replica per layer:

| Structure | Spill payload | Reduce |
|---|---|---|
| `InputCovarianceAccumulator` | reuse `spill_layer_to_disk` `:1135` | reuse `_reduce_spilled_cov_dirs` `:211` (fp32 sum) |
| `ReapAccumulator` | finalize_layer→`{sums,counts,freq}` (CPU floats/ints) | sum sums (fp32), sum counts, sum freq |
| `ReamCostAccumulator._gate_gram` | `{li: [E,E] fp64}` | fp64 key-wise sum (bit-exact) |
| `_sim_tensor` + `_total_tokens_by_layer` + `_neuron_act_sum/_count` | dense [E,E] fp64 + int + per-(l,e)[d_int] + int | fp64/int sums (bit-exact / exact) |

New `_spill_reap_layer`/`_reduce_reap_dirs`/`_spill_ream_layer`/`_reduce_ream_dirs` (small;
pattern-identical to `_reduce_spilled_cov_dirs`, **sorted dir order** `:231`). Reduce runs
**per layer** inside the loop, reconstructing the parent's four accumulators so the
assign/merge phases consume them exactly as serial. fp64 REAM ⇒ bit-exact; fp32 cov/REAP ⇒
~1e-6 (tolerated; absent at 1 replica).

### A4 — DP-or-serial branch in `on_profile`

**File:** `stage2/plugins/layer_merge.py` `on_profile` (`:498-538`).

Replace the single `_srr._profile_layer(...)` (`:528-533`) with: if
`profile_dp.enabled and replicas>1` → `run_dp_profile_layer(pool, layer_ref, shards,
reap_acc, cov_acc, ream_acc, device, auto_batch_cfg)` (issues RESYNC(L-1)+PROFILE(L),
joins, reduces the four spill sets into the ctx/`self` accumulators); else the serial
`_profile_layer` unchanged. `self.cov_acc.finalize_layer(layer_idx)` stays after the branch
(the DP reduce already finalized per-shard; the parent-side finalize is a no-op on an
already-CPU-resident cov OR is folded into the reduce — confirm in review which side owns
the final cast — see Q2).

### A5 — Per-replica auto-batch (inherited verbatim; premise provided by H1's pin)

Each worker pins via `CUDA_VISIBLE_DEVICES` ⇒ its own `CudaMemProbe(device)`; uses
`size_batch(... floor=1, mem=CudaMemProbe(device))` + `run_with_oom_backoff(..., floor=1)`
(`utils/auto_batch.py:143-203`), gated by inherited `auto_batch.enabled`. **No new
auto-batch logic.** Independent per-replica sizing is SOUND **because H1 pins cov per-seq**
(finalized Grams batch-invariant) and REAP/REAM are inherently batch-invariant ⇒ no
cross-replica min-agreement (Stage-3 contract `covariance_collection.py:373-447`). Default
`auto_batch.enabled=false` ⇒ fixed batch, no probe.

### A6 — (folded into A0) pool lifecycle + clean teardown

Covered by A0: spawn-once, per-layer command/reduce protocol, SHUTDOWN + join + exitcode
check + leak verification in `orchestrator.run`'s `finally`. Worker `ERROR` aborts the run.

### A7 — Tests (no live multi-GPU — mocked in-process workers + reduce-math)

**File:** `max_quality/tests/test_stage2_profile_dp.py` (NEW).

1. **Shard:** N sequences → R disjoint contiguous shards, every sequence covered once.
2. **Reduce — REAP:** two accs, disjoint per-(l,e) sums/counts; spill+reduce; assert
   element-wise sums + `score()==Σsum/Σcount`.
3. **Reduce — REAM gate_gram:** two fp64 `_gate_gram[li]`; reduce; **bit-exact** sum;
   `compute_gate_similarity_matrix` on merged == single-acc-over-both-shards.
4. **Reduce — REAM sim/total/neuron:** disjoint shards; fp64 `_sim_tensor` bit-exact, int
   `_total_tokens_by_layer`, `_neuron_act_sum/_count`; `compute_delta_expert`/`get_neuron_mean`
   == single-pass.
5. **E2E equivalence (mocked, 2 in-process CPU "workers"):** monkeypatch the spawn to run
   two shards in-process against the `test_stage2_pipeline_run_layer.py` stub MoE; reduce;
   assert merged `reap_acc/cov_acc/ream_acc` == serial `_profile_layer` over full calib
   (`allclose` ~1e-5 cov/REAP; fp64 REAM bit-exact).
6. **Structural replay (C1):** unit-test the RESYNC handler — give a worker-side model copy
   + a parent's `(final_kept_ids, grouped, merged tensors)`; replay; assert the worker
   layer's expert-bank SHAPE + router `num_experts`/`top_k`/`mlp.num_experts` + tensor
   VALUES match a parent that ran `bank.select` + `_resize_router_for_kept_experts`.
7. **Byte-identical default gate:** `profile_dp.enabled=false` ⇒ `on_profile` calls serial
   `_profile_layer`, no DP import/spawn (spy).
8. **Reservoir guard:** `expert_distill_steps>0` + `profile_dp.enabled=true` ⇒ DP disabled
   at resolution + the `log.warning` names `expert_distill` (assert via `caplog`).
9. **Cov per-seq pin (H1):** `_profile_layer` with `seq_len` set routes `input_cb`/
   `intermediate_cb` through `update_grouped`; assert a 2-batch-vs-1-batch run yields the
   SAME finalized cov Gram (batch-invariance), and `seq_len=None` is byte-identical to a
   plain `update`.

`cd max_quality && python -m pytest tests/test_stage2_profile_dp.py -q`

### A8 — Out of scope: live ≥2-GPU validation (DEFERRED)

Real ≥2-GPU run DEFERRED to GPU. 1-GPU default-off byte-identical (A7.7); reduce + replay
math proven in-process (A7.2-6,9); first live ≥2-GPU run validates the persistent-pool IPC +
RESYNC barrier + teardown. Document in `profile_dp.py` docstring + commit body (mirror the
Stage-3/4 "untested on real ≥2-GPU" memory caveat).

### A9 — Full suite + whole-impl review before green

```
cd max_quality && python -m pytest tests/ -q -k "stage2 or cov or profil or shared_io or merge or pipeline"
```
Then per-task review→fixer ping-pong (all 5 categories incl. nitpick)→all-none + a
whole-implementation review + the full caller/integration suite (the
`run_full_suite_and_final_review` rule). Confirm no other `_profile_layer` /
`_remap_covariance_for_layer` caller regressed.

---

## 4. Task list (ordered, TDD)

| # | Task | Files | Gate |
|---|---|---|---|
| **B1** | Failing test: survivor anchor = group UNION (gate) | `tests/test_stage2_merge_anchor_union.py` (NEW) | FAILS pre-fix |
| **B2** | Failing test: down_proj union PERMUTED (both axes, exact perm) | same | FAILS pre-fix |
| **B3** | Union remap (kw-only grouped/member_perms=None; fp32-sum-cast; singleton byte-identical) | `shared_io.py:301-344` | B1/B2 pass; singleton green |
| **B4** | Thread `grouped`+EXACT `member_perms` (capture perm at `merging.py:280/291/298`) → remap | `merging.py:34-424`; `layer_merge.py:543-591,680` | one caller; resume unchanged (H3) |
| **B5** | Run REAL guardrails (NO golden regen — H2) | `test_stage2_merge.py`,`_cov_manifest`,`_shared_io`,`_plugin_layer_merge`,`_smoke_resume` | 36+ green; `:17-32` green |
| **A0** | Persistent pool lifecycle + command/reduce IPC + teardown/exitcode | `profile_dp.py` (NEW); `orchestrator.run` finally | replay unit-test green; no leaks |
| **C1** | Structural replay handler (RESYNC: patch centroids + `bank.select` + router resize) | `profile_dp.py` worker | A7.6 |
| **A1** | `profile_dp` config (default OFF) + reservoir guard AT RESOLUTION (log.warning names consumer) | `orchestrator.py:1100-1215` | default-off serial; A7.8 |
| **A2** | Sequence-disjoint shard (reuse `_shard_calib`), built once | `profile_dp.py` | A7.1 |
| **H1** | NEW cov per-seq pin: thread `seq_len`, switch `input_cb`/`intermediate_cb`→`update_grouped(ctx["token_idx"]//seq_len)` | `profiling.py:228-239` | A7.9; serial byte-identical |
| **A3** | Per-replica spill + per-layer reduce of ALL FOUR accumulators | `profile_dp.py` | A7.2-4 |
| **A4** | DP-or-serial branch in `on_profile` | `layer_merge.py:498-538` | A7.5 |
| **A5** | Per-replica auto-batch inherited (premise = H1 pin) | `profile_dp.py` | independent sizing |
| **A7** | DP tests (shard, 4× reduce, e2e, structural-replay, default-gate, reservoir, pin) | `tests/test_stage2_profile_dp.py` (NEW) | all green |
| **A8** | Out-of-scope note: live ≥2-GPU deferred | `profile_dp.py` docstring + commit | documented |
| **A9** | Full suite + whole-impl review to all-none | (all) | green + reviewed |

---

## 5. Key decisions (for the reviewer)

1. **Per-layer DP, never cross-layer** — merge mutates in place (`merging.py:424`); L+1
   forwards through L's merged weights (`profiling.py:338-342`). −1.0 AVG otherwise.
2. **A0 = persistent pool + per-layer STRUCTURAL re-sync (design-1, user-accepted XL).** NEW
   IPC subsystem (no repo template). Spawn-once; per-layer command/reduce channels; RESYNC
   barrier before each layer; SHUTDOWN+exitcode teardown.
3. **C1: re-sync is structural surgery, NOT a value delta.** `bank.select` reshapes;
   `_resize_router_for_kept_experts` REPLACES the router Parameter + mutates counts. Workers
   **replay** `bank.select`+`_resize_router_for_kept_experts` (after patching centroid values),
   or — recommended — receive the already-selected kept tensors + resized router and set them
   directly. Same wire volume; "delta broadcast" language dropped.
4. **H1: cov per-seq pin is NEW, not inherited.** Live cov uses `update` (`profiling.py:230,236`);
   switch to `update_grouped(ctx["token_idx"]//seq_len)`. `input_cb` gains `token_idx` use
   (the ctx already carries it). REAP/REAM stay as-is (batch-invariant). Premise of A5.
5. **Reduce all FOUR accumulators per layer.** REAM fp64 bit-exact; cov/REAP fp32 ~1e-6
   (tolerated; absent at 1 replica). Disk-spill + sorted reduce mirrors `_reduce_spilled_cov_dirs`.
6. **Reservoir guard at config resolution** (`log.warning` naming `expert_distill`/`output`/
   `mergemoe`), not 40 layers deep. Reservoir-merge across shards is future work.
7. **(B) union = `Σ_j G_j`**, down_proj permuted BOTH axes by the EXACT merge perm
   (`merging.py:280/291/298`, mirrors RegMean `:352-360`). Non-merged survivors byte-identical
   (kw-only `grouped=None`). The Gram is unnormalized `X^TX` ⇒ union = additive sum (matches
   AA-SVD anchor `A=X`).
8. **H2: no stage2 cov golden exists.** Guardrails are the 5 named real tests (baseline 36
   passed) + the new union test; `test_stage2_merge.py:17-32` is the singleton compat anchor.
9. **H3: resume is consistent for free** — `orchestrator.py:1051` loads the already-remapped
   union cov off disk; no resume change.
10. **Default-off byte-identical everywhere** — `profile_dp.enabled=false` ⇒ serial; (B)
    changes only merged-layer cov; merge JSON/weights/singleton cov byte-identical.

---

## 6. Open questions (resolve in review)

1. **C1 variant:** "patch centroids then replay select+resize" vs "send already-selected
   kept tensors + resized router, set directly" (recommended). Same wire volume; pick the
   simpler.
2. **A4 finalize ownership:** does the DP reduce or the parent `cov_acc.finalize_layer`
   apply the final storage-dtype cast? Pick one to avoid a double-cast.
3. **fp32 cov/REAP ~1e-6 under reduce** acceptable on the DP-ON path (NOT on the 1-replica
   default, which has no reduce). Same class as `activation_hooks.py:308-311`.
4. **`shards_per_model>1`** (tensor-sharded replica): mirror Stage-3's knob; default 1, defer.
5. **Resume × DP interchange (H3 closes the cov side):** a DP run writes the same per-layer
   partials via `write_artifacts`; a serial resume loads the reduced+remapped union cov and
   matches a fresh run. Add a cheap resume smoke if practical.
