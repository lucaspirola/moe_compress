# SC Stage 2 Bottleneck — Diagnosis & Optimization Plan

**Branch**: `feat/calibration-v2` (HEAD `6ff3636`)
**Status**: Read-only diagnosis; no implementation yet.
**Scope**: Identify the *actual* dominant cost sources in a 3 h SC Stage-2 row
on Qwen3.6-35B-A3B, then propose targeted optimizations that preserve the
existing byte-identical golden snapshot.

---

## 0. Critical correction to the prior framing

The prior verification agent and `MOE_COMPRESS_REPORT.md:250-254` claim
"~3 h / row, dominated by per-pair Hungarian neuron-permutation solve in
`_tentative_merged_weights` (~25 ms / pair, ~96% of per-pair cost)". The
per-pair number is plausible (cdist + scipy LAP + fp32 weighted-merge alloc),
but the **total** does not reconcile:

| Pair-count estimate | Source | Total at 25 ms/pair |
|---|---|---|
| 45,000 pairs/row | prior agent (unsourced) | ~19 min |
| n_NC × K × n_layers ≈ 88 × 48 × 40 ≈ 169,000 pairs | code inspection at N=256, K=48 | ~70 min |
| 21 merge-amenable layers × 88 × 48 ≈ 88,700 pairs | accounting for some sparsely-merged layers | ~37 min |

Even the *upper* estimate is < 70 min. **~2 h of the 3 h SC row is something
other than the per-pair cost-matrix work.** The Hungarian framing is at best
partially correct; the missing time has to come from elsewhere.

---

## 1. Phase 1 — Verified timing breakdown

No archived per-phase Trackio dumps or per-row logs from the SC = 0.1293
production run survive in either checkout (`/home/lucas/ai/moe_compress` or
`/home/lucas/moe_compress`); the only artifact is the result summary in
`tasks/MOE_COMPRESS_REPORT.md`. The breakdown below is therefore
**code-inspection back-of-envelope** with confidence flags (HIGH / MEDIUM /
LOW).

### 1.1 Model & config anchors (HIGH confidence)

From `max_quality/docs/stage2_assignment_revision.md:293` and
`max_quality/configs/qwen36_35b_a3b_30pct.yaml`:

- N = 256 experts/layer, top_k = 8, hidden = 2048, `moe_intermediate_size = 512`
- 40 MoE layers (Qwen3.6-35B-A3B)
- `num_calibration_samples = 4000`, `batch_size = 32` → 125 batches × 2048 seq = 8.2M tokens/layer-profile-pass
- `cost_output_token_cap = 1024` (default; the per-pair SwiGLU forward sees 1024 tokens)
- `cost_topk_filter K = 48` (SC inherits the default)
- SC override: `cost_alignment="output"`, `capacity_util_threshold=0` (every layer runs the output cost; no slack-downgrade to "pre")
- SC defaults (no override): `em_refinement_rounds=0`, `two_opt_refine=False`, `expert_distill_steps=0`, `merge_heal_enabled=False`
- At ~35% compression: n_NC ≈ 80–100 per merge-amenable layer, n_C ≈ 156–176

### 1.2 Pair count for SC's cost matrix (HIGH confidence)

`_output_space_cost` (`max_quality/src/moe_compress/stage2/plugins/output_space_cost.py:399-442`):
outer loop runs `len(noncentroid_ids)` times (n_NC ≈ 88 on average), inner
loop runs `k_cand = min(K, n_C) ≈ 48` times — **per layer ≈ 4,224 pairs.**
At 40 MoE layers: **~169,000 pairs total** (the floor; some layers may have
larger n_NC under aggressive compression).

### 1.3 EM rounds is NOT a multiplier for SC (HIGH confidence)

`em_refine.py:277`:
```python
if em_rounds <= 0 or cost_alignment != "post":
    return initial_assignment, initial_delta, 0
```
The EM refiner **early-returns** when `cost_alignment != "post"`. Under SC
(`cost_alignment="output"`) EM is a no-op regardless of `em_refinement_rounds`.
Bump-loop iterations could in principle multiply work, but the SC YAML's bump
gates rarely trip on a clean post-Stage-1 budget — typically the bump-loop
runs 1 iteration per layer.

### 1.4 perm_cache is NEVER hit for SC (HIGH confidence)

`perm_cache.put` is called only from `ream_cost_post.py:285` (the `post`
branch). The `output` branch reads the cache (`output_space_cost.py:250`,
`em_refine.py:190`, `merging.py:140`) but **never writes** to it. Under SC
every cache lookup misses; every call to `_tentative_merged_weights` and
every call inside `_merge_experts_inplace` does a fresh
`_permutation_align_to_centroid` (cdist + scipy LAP). The "M1 cache reuse"
optimization is dead code under SC.

### 1.5 Per-cost component breakdown (~3 h envelope)

| # | Phase | What it does | Cost shape | Estimate | Confidence | SC-specific? |
|---|---|---|---|---|---|---|
| A | Per-layer profile-pass forward | 1+2+...+40 = 820 layer-forwards × 125 batches = 102,500 layer-forwards | bs=32 × seq=2048 grouped_mm forward on 35B teacher | **30–60 min** | MEDIUM | No (universal) |
| B | `_LayerInputAccumulator.add` Python token loop (`stage2/profiling.py:58-80`) | Python `for i in range(n)` over 65,536 tokens per batch × 125 batches × 40 layers = **328M iterations** | CPU-only, per-iter ~15–40 µs (torch.randint + .item() + indexed assign) | **~1.5–3 h** | MEDIUM–HIGH | **YES — SC-specific** |
| C | Per-pair output-space cost (`_output_space_cost` + `_tentative_merged_weights`) | ~169k pairs × (cdist 512×512 GPU ~1 ms + scipy LAP CPU ~10 ms + 3× fp32 weighted-merge alloc ~3 ms + SwiGLU forward ~0.4 ms) | ≈ 15 ms / pair | **~40–60 min** | MEDIUM–HIGH | **YES — SC-specific** |
| D | `_merge_experts_inplace` Hungarian for the chosen assignment | ~85 merges × 15 ms × 40 layers (perm_cache empty for SC, see §1.4) | one Hungarian per merge member | ~1 min | HIGH | (SC pays it twice — same per-pair work as in C since C never wrote the cache) |
| E | `record_neuron_activations` `.cpu()` syncs (`activation_hooks.py:589`) | 256 experts × 125 batches × 40 layers = 1.28M `.cpu()` syncs, but only for the target layer per profile = 256 × 125 × 40 = 1.28M; cost ≈ 20 µs/sync | CPU↔GPU host stall per call | ~1 min | MEDIUM | No (universal) |
| F | `cov_acc.update` (input covariance accumulator) | per-expert-per-batch on target layer | GPU matmul + accumulator add | ~5 min | LOW | No (universal) |
| G | `_ream_cost_matrix` cheap symmetric δ_REAM (the K-prefilter base) | matrix ops at n_NC × n_C × layer | fully vectorized after `finalize_batch` rewrite | <1 min | HIGH | No (universal) |
| H | I/O — covariance snapshot + merge JSON + heal weights | per-layer write of fp16 cov + tiny JSON | I/O + a 0.1–0.5 GB write/layer | ~1–2 min | MEDIUM | No (universal) |
| I | Bump-loop / orphan promotion / other orchestration | python | <1 min | HIGH | No (universal) |

**Rough sum**: A + B + C ≈ 2 h 10 min – 4 h 0 min. The 3 h envelope falls
inside this range with the **`_LayerInputAccumulator` Python loop (B) as
the single dominant accidental cost**, the per-pair Hungarian + SwiGLU (C)
as the second largest, and the universal profile-pass forward (A) as the
third.

### 1.6 Top-3 cost consumers (ranked)

| Rank | Bottleneck | Share of 3 h | Confidence |
|---|---|---|---|
| **#1** | `_LayerInputAccumulator.add` per-token Python reservoir loop | ~50–80 % (1.5–2.5 h) | MEDIUM–HIGH |
| **#2** | Per-pair Hungarian + fp32 weighted-merge alloc inside `_output_space_cost` | ~20–35 % (40–60 min) | MEDIUM–HIGH |
| **#3** | Universal profile-pass forward (Stage 2 baseline cost) | ~15–25 % (30–60 min) | MEDIUM |

The prior agent's "96% Hungarian" framing collapses to rank #2 and represents
~30 % of the row, not 96 %. **The largest cost is an accidental Python loop
in the reservoir sampler that fires only under `cost_alignment="output"`.**

---

## 2. Phase 2 — Diagnose each top-3 bottleneck

### Bottleneck #1 — `_LayerInputAccumulator.add` (`stage2/profiling.py:58-80`)

**Why expensive**:

```python
def add(self, hidden: torch.Tensor) -> None:
    flat = hidden.reshape(-1, hidden.shape[-1]).detach().to("cpu")  # GPU→CPU sync, 8 MB
    n = flat.shape[0]                                                # n = bs × seq = 65,536
    if self.buffer is None:
        take = min(n, self.max_samples)
        self.buffer = flat[:take].contiguous().clone()
        self.seen = n
        return
    for i in range(n):                                               # ~65k Python iterations / batch
        self.seen += 1
        if self.buffer.shape[0] < self.max_samples:
            self.buffer = torch.cat([self.buffer, flat[i:i + 1]], dim=0)
        else:
            j = int(torch.randint(0, self.seen, (1,),
                                  generator=self._generator).item())   # CPU rand + .item() sync
            if j < self.max_samples:
                self.buffer[j] = flat[i]                              # CPU row copy of 2048 floats
```

- Per-batch tokens: `n = batch_size × seq_len = 32 × 2048 = 65,536`.
- Per-layer batches: 125.
- Per-layer iterations: 8.19 M.
- Across 40 layers: **~328 M Python iterations** of a per-row torch op
  sequence that cannot be JITed and is purely CPU.
- The `add()` callback fires once per batch via
  `layer_ref.layer_module.register_forward_pre_hook(_capture_layer_input)`
  (`stage2/profiling.py:233`), so it serializes against the forward pass
  (blocks the forward stream waiting for the CPU loop to return).

**Is the work fundamental or accidental**: ACCIDENTAL.

`max_samples = cost_output_token_cap = 1024`. The reservoir sampler is
*intended* to give a uniform subsample of 1024 tokens across the full
calibration distribution. But:

1. The downstream consumer (`_output_space_cost`) doesn't require
   global uniformity — it needs a representative token sample for a
   per-layer mean. A reservoir sample across batches is one valid
   strategy, but **the same statistical property is achieved by
   sampling 1024 tokens once at the END of profiling from the
   concatenated tokens** (`buffer = concat_all_tokens[randperm(N)[:1024]]`)
   with a constant cap on RAM via per-batch truncation.

2. The reservoir algorithm itself doesn't need a Python loop — it can
   be vectorized in PyTorch with a single masked-scatter: per batch
   generate `n` candidate replacement indices via `torch.randint`,
   compare against `seen + arange(n)`, mask the in-bounds replacements,
   and `index_copy_` the survivors into the buffer in one CUDA-side op
   (or one host-side vector op since the buffer lives on CPU anyway).

3. Even simpler — since `n ≫ max_samples` after the first batch, the
   acceptance probability falls below 2 % after ~5 batches; sampling
   per-batch instead of per-token (Algorithm L / vitter's optimal reservoir
   sampling) collapses the loop from 65,536 iterations/batch to
   ~`max_samples × log(seen/max_samples)` iterations, ≈ a few hundred per
   batch instead of 65k. Pure CPU but ~100× fewer Python iterations.

**Optimization surface area**: `stage2/profiling.py:58-80`. One function,
~25 lines. The downstream consumers are
`output_space_cost._output_space_cost` (reads `layer_inputs` as a `(T, H)`
tensor) and `expert_distill._distill_merged_group` (same shape contract).
Both treat the buffer as an opaque token bag — no ordering or
batch-locality assumption.

**Speedup ceiling**: ~50–200× wall-clock on this hook. Two viable
replacement designs:

  - **Design A — vectorized reservoir (Algorithm L)**: collapses the
    Python loop to O(max_samples × log(seen/max_samples)) iterations.
    Expected speedup ~100×. Statistical output identical (uniform
    random sample of size 1024) but the specific sample identities
    will differ → **breaks byte-identical golden snapshot**. Needs a
    new golden re-baselining.
  - **Design B — vectorized "fill, then per-batch truncate"**: replace
    the reservoir with a fixed-size head buffer that takes
    `min(max_samples - filled, n_new)` tokens per batch, then early-
    exits the profile pass once the buffer is full. No randomness, no
    Python loop. Expected speedup ~500×+ on this hook but **changes the
    statistical distribution** (first-tokens-only sample, not a uniform
    sample across batches). Quality impact unknown until measured.
  - **Design C — pure-tensor reservoir on a single in-flight buffer**:
    keep the random-replacement semantics but rewrite the per-batch
    body in pure tensor ops:
    ```python
    flat = hidden.reshape(-1, hidden.shape[-1]).detach()
    n = flat.shape[0]
    seen_before = self.seen
    self.seen += n
    pos = torch.arange(n) + seen_before          # global positions of these tokens
    # Algorithm R: each new token at position p has probability max_samples/p of replacement
    keep_probs = (self.max_samples / pos.float()).clamp_max(1.0)
    keeps = torch.rand(n, generator=self._generator) < keep_probs
    n_keep = int(keeps.sum())
    if n_keep == 0:
        return
    # For each kept token, choose a random index in [0, max_samples) and overwrite
    j = torch.randint(0, self.max_samples, (n_keep,), generator=self._generator)
    self.buffer.index_copy_(0, j, flat[keeps])
    ```
    Pure-tensor, no Python loop, statistically identical to Algorithm R
    (the textbook reservoir). Expected speedup ~50–100×. Sample
    identities will differ from the current loop's bit-exact output
    (Python `.item()` ordering of `randint` calls produces a particular
    sequence; the vectorized form draws all `n` randoms in one call).
    **Breaks byte-identity** but preserves statistical contract.

**Recommendation**: Design C. It preserves the reservoir-sampling
statistical contract used downstream and is the minimum behavioural
change. The byte-identity snapshot can be re-baselined cheaply if the
statistical contract is preserved.

**Risk surface**:
- *Correctness*: trivially testable — verify uniform-over-all-tokens
  property on a 10k-token synthetic batch; verify per-token presence
  probability matches `max_samples / total_seen`.
- *Memory*: identical (one buffer of `max_samples × hidden`).
- *Numerical drift*: tokens differ by identity, but the downstream
  consumer is a routing-weighted MEAN over the sample (`_output_space_cost`
  line 441: `cost = (gate_m * per_token).sum() / gate_m.sum()`). Mean
  is invariant under random sampling at the same sample size. Expect
  the cost values to drift by O(1/√1024) ≈ 3 % — well within the
  natural numerical noise of fp32 SwiGLU forwards.

---

### Bottleneck #2 — Per-pair Hungarian + fp32 weighted-merge inside `_output_space_cost`

**Why expensive**: At ~169,000 pairs × ~15 ms each:

1. `_permutation_align_to_centroid` (`stage2/permutation_align.py:114`) for each pair:
   - GPU `torch.cdist` of two 512×2048 matrices (gate then up), ~1 ms.
   - `.cpu().numpy()` sync, ~0.5 ms.
   - `scipy.optimize.linear_sum_assignment` on 512×512, ~10 ms (matches doc
     estimate at line 300 of `stage2_assignment_revision.md`).
   - Result → torch tensor on GPU.
2. `_tentative_merged_weights` (`output_space_cost.py:212`):
   - 6× `bank.get(...).to(torch.float32)` calls — fp16/bf16 → fp32 promotion,
     each allocating ~6 MB.
   - perm-apply via `W[perm_t, :]` on gate / up (row gather) and
     `W[:, perm_t]` on down (column gather) — ~6 MB each.
   - Linear combination — 6 MB temp output per matrix.
   - Total ~50 MB of throwaway fp32 allocations per pair.
3. `_swiglu_forward` (`output_space_cost.py:157`) on T=1024 tokens, ~0.4 ms.

The cost stack is: **Hungarian LAP (~10 ms) > fp32 weighted-merge ops
(~3–4 ms) ≫ SwiGLU forward (~0.4 ms)**. The prior agent's "96% Hungarian"
is approximately right within the per-pair budget; the framing error is
that the per-pair budget is ~30 % of the row, not 96 %.

**Is the work fundamental or accidental**: MOSTLY fundamental, with two
significant accidental costs piggy-backed.

The Hungarian solve itself is the inevitable cost of finding the optimal
neuron permutation per pair. Replacement options:

  - **GPU LAP** (`auction_lap`, `lap-jv` CUDA, etc.): a 512×512 LAP on
    GPU is ~1–2 ms vs scipy's 10 ms. ~5× per-pair speedup. **Available
    via PyPI (`auction_lap`)** — but currently no CI / numerical-
    stability validation in the repo; needs a careful golden re-baseline
    because LAP solutions are not unique under cost ties (different
    solvers return different permutations among equally-good ones).
  - **Cache reuse across pairs (M1)**: the `_PermAlignCache` infrastructure
    is already present and would amortize ~half the Hungarian solves
    across the bump loop's repeated calls — **but SC never hits the
    cache because `output_space_cost.py` never calls `perm_cache.put`**
    (verified at §1.4). Adding the `.put` call is a 5-line change and
    halves the Hungarian count on bump-loop retries + the merge step.
    For SC with no bump-loop retries this only saves the merge-step
    Hungarian (~1 min) — small.
  - **Approximate alignment** (Sinkhorn / soft-assignment): can replace
    Hungarian with an O(n²·iters) iterative solve that runs on GPU.
    Quality cost depends on iter count; for n=512 and ~10 Sinkhorn
    iterations expect ~1 ms/pair (~10× speedup) but cost values shift
    by ~1–5 % which may move SC's `bpt_gap` enough to invalidate the
    0.1293 result. Risky.
  - **Reduce K from 48**: re-run the K-correlation experiment from
    `stage2_assignment_revision.md`. At K=24 the per-pair count halves
    (~85 → 42 min); at K=16 it drops by 3× (~30 min). The doc cites
    rank-correlation 0.332 between cheap and expensive costs at K=16
    — too low to safely shrink K without an ablation. Not a free win.

The accidental costs:

  - **fp32 weighted-merge allocation**: each pair allocates ~50 MB of
    fp32 temp tensors. The merge math is linear in the weights — it
    could equally be done in bf16 (the model's native dtype) at ~half
    the memory and ~half the alloc/copy cost. The fp32 upcast was
    added for numerical safety against accumulator drift in the
    `_post_alignment_cost` whitened-residual path; the `output` path
    has no such drift concern (it's a single weighted sum, not a
    repeated accumulator). **Switching to bf16 here saves ~2 ms/pair
    × 169k pairs ≈ 5.6 min, with no semantic change.**
  - **`build_banks(layer_ref)` is called once per `_tentative_merged_weights`
    call** — i.e. ~169k times per row. `build_banks` rebuilds an
    `ExpertMatrixBank` dict object (`utils/model_io.py:515`). The
    function doesn't materialize tensors (it returns lightweight bank
    refs that hold the stacked-tensor reference), so each call is
    cheap (~50 µs), but at 169k calls that's ~8.5 sec — small but
    free to fix by hoisting the call once per layer.

**Optimization surface area**:
- `stage2/plugins/output_space_cost.py:212-273` (`_tentative_merged_weights`)
- `stage2/plugins/output_space_cost.py:276-444` (`_output_space_cost`)
- `stage2/permutation_align.py:114-177` (`_permutation_align_to_centroid`)

**Speedup ceiling**: ~3–4× if GPU LAP + bf16 weighted-merge + write the
perm cache → halves the Hungarian count on retries + the merge step. Per-
pair drops from ~15 ms to ~3–5 ms → per-row drops from ~45 min to ~12–15 min.

**Risk surface**:
- Byte-identity break (any LAP swap, any dtype change).
- Numerical drift from bf16 weighted merge (small — single linear combo,
  no accumulator).
- GPU LAP package vendor risk (`auction-lap`, `lap-jv`) — not currently
  in the dependency set.

---

### Bottleneck #3 — Universal profile-pass forward (`stage2/profiling.py:_profile_layer`)

**Why expensive**: 40 sequential profile passes with early-exit at the
target layer, plus full instrumentation hooks:

- 1+2+...+40 = 820 layer-forwards × 125 batches = ~102,500 layer-forwards.
- Per layer-forward at bs=32 × seq=2048 ≈ 9.8 TFLOP active (3B active
  params × 2 × 65,536 tokens / 40 layers).
- At realised ~50 ms / layer-forward on H200 grouped_mm: ~50 min total.
- Hook overhead (cov + ream + reap callbacks, full-softmax cache,
  layer-input capture) adds ~10–20 % on top.

**Is the work fundamental or accidental**: ARCHITECTURALLY fundamental
(REAM's sequential per-layer profiling protocol requires this — see
paper 2604.04356 §4 Fig 1(b)), but the **specific forward path** has
significant accidental cost:

  - **40 separate forward calls** instead of one with intermediate-state
    capture. Each call re-runs the embedding + attention work for layers
    0..L-1 that were already computed in the previous layer's pass.
    A single forward with `output_hidden_states=True` and per-layer
    callback would do all 40 layers' profiling in one pass (~30 sec/batch
    × 125 batches = ~63 min) vs the current cumulative-prefix scheme.
    But this conflicts with REAM's sequential-merge protocol: layer L+1's
    profile must run on the **merged** model state of layer L. Bypassing
    this requires either:
    1. Profile all 40 layers on the unmerged base model (BREAKING change
       — different cost values; un-validated against 0.1293 result).
    2. Keep the sequential protocol but vectorize the early-exit
       overhead — incremental ~10 % win at most.
  - **Pre-Stage-2.5 V1+V2 cache writers** (`tasks/calib_v2_writers_todo.md`)
    already capture REAP scores + covariance + routing stats from vLLM in
    a single fast pass. Stage 2's `Stage2ReapScoresCacheProvider` and
    `Stage2RoutingStatsCacheProvider` hydrate `scores` / `freq` from those
    sidecars BUT the live `_profile_layer` ALSO runs the cov + REAM + neuron-
    mean accumulators that the sidecars don't cover yet. Wiring the
    remaining accumulators into the calibration-v2 writer (and adding a
    `Stage2InputCovarianceCacheProvider` + `Stage2NeuronMeansCacheProvider`)
    would let `_profile_layer` short-circuit entirely. This is the
    "calibration-v2 writers consumed by Stage 2" extension already
    referenced in `tasks/calib_v2_writers_todo.md:14-26` — feasible but
    larger-than-the-table-stakes scope.

**Optimization surface area**:
- `stage2/profiling.py:86-286` (`_profile_layer`).
- `stage2/plugins/layer_merge.py:473-496` (`on_profile` phase entry).
- New cache-provider plugins for cov / neuron means (extends the
  Stage2ReapScoresCacheProvider pattern).

**Speedup ceiling**: ~3–5× if the cached-sidecar route lands (~50 min →
~10–15 min for the profile pass). Universal across Stage 2 — saves time
on every row, not just SC.

**Risk surface**:
- Sidecar correctness: the cached values must be byte-identical to live
  values, modulo documented fp32-reduction-order drift.
- Cache invalidation: any model dtype / experts_implementation change
  must invalidate the cache.
- Calibration-corpus identity: cache keyed on `(model_repo, calib_corpus,
  seq_len, num_seqs)`; mismatch must hard-fail.

---

## 3. Phase 3 — Optimization plan

### 3.1 Combined-savings estimate

Realistic outcomes (with the existing 3 h envelope):

| Variant | Lands | Estimated row time |
|---|---|---|
| Baseline | nothing | 3 h |
| Bottleneck #1 only | reservoir vectorized (Design C) | **~1 h** |
| Bottleneck #1 + #2 | reservoir + (GPU LAP OR bf16 merge + perm-cache write) | **~30–40 min** |
| Bottleneck #1 + #2 + #3 | + sidecar-cached profile | **~15–25 min** |

The fixed minimum is the per-pair SwiGLU forward (~7 min) + the K-prefilter
matrix + the merge step + I/O — call it ~10 min floor.

### 3.2 Phased implementation plan

**Phase A — Bottleneck #1 fix (HIGHEST ROI)**

- Files: `max_quality/src/moe_compress/stage2/profiling.py` (one function,
  `_LayerInputAccumulator.add`).
- Approach: Design C (vectorized Algorithm R reservoir, all-tensor ops).
- Expected speedup on this hook: ~50–100× (i.e. ~80–150 min saved on the
  3 h row).
- Risk: medium. The byte-identical Stage 2 golden snapshot
  (`max_quality/tests/test_stage2_*.py`, the SC-relevant ones at
  `test_stage2_output_space_cost_*.py` and `test_smoke_stage2_*.py`)
  WILL break because the sample identities differ. **Re-baseline is
  required.** The statistical contract (uniform random sample of size
  1024) is preserved.
- Unit-test gate:
  - New test `test_layer_input_accumulator_uniform.py` verifying that
    over 100k synthetic tokens with `max_samples=1024`, each token's
    final-buffer probability is within 3σ of `1024/100,000`.
  - Existing `test_stage2_output_space_cost_*.py` re-baselined against
    fresh golden output values.
- Validation gate (NEEDS USER AUTHORIZATION FOR GPU SPEND): full SC row
  on a single H200, target `bpt_gap` within 0.02 of 0.1293
  (covers the ~3 % sample-mean noise from the new reservoir
  identities — same tolerance the L1 plan uses for FP8 teacher drift
  at `L1_FOR_SC_PLAN.md:299`).
- Effort: ~4–6 h dev + 1 H200-row validation.

**Phase B — Bottleneck #2 fix (MEDIUM ROI, GATED ON A)**

Three sub-options, in order of recommended landing:

  B1. **Write the perm cache from the output path** (quick win):
      - Files: `output_space_cost.py:_tentative_merged_weights`. Add the
        `perm_cache.put((li, c, m), perm, residual=None)` call after the
        Hungarian solve, mirroring `ream_cost_post.py:285`.
      - Saves the merge step's Hungarians (~1 min).
      - Risk: zero — symmetric with the post path; no semantic change.
      - Effort: ~1 h dev.

  B2. **bf16 weighted-merge** (medium win):
      - Files: `output_space_cost.py:_tentative_merged_weights` lines 245-272.
      - Replace the `.to(torch.float32)` upcasts with the model's native
        bf16 dtype (the merge math is a single weighted sum — no
        accumulator drift risk).
      - Saves ~5–6 min/row.
      - Risk: low. Numerical drift bounded by one bf16 mul+add per element.
      - Effort: ~2 h dev.

  B3. **GPU LAP via `auction-lap` or hand-rolled**:
      - Files: `permutation_align.py:_permutation_align_to_centroid`.
      - Adds `auction-lap` (PyPI) to deps.
      - Saves ~30 min/row.
      - Risk: high — non-determinism under cost ties; needs a careful
        golden re-baseline AND a numerical-stability validation against
        scipy across a randomized cost-matrix corpus.
      - Effort: ~6–8 h dev + 2 H200-row validations (smoke + SC).

**Recommendation**: land B1 + B2 (cheap, ~6–7 min saved, low risk).
Defer B3 unless the row-time after A+B1+B2 is still > 30 min.

**Phase C — Bottleneck #3 fix (LARGE BUT BROAD-SCOPE)**

- Files: new `stage2/plugins/input_covariance_cache.py`,
  `stage2/plugins/neuron_means_cache.py`; extensions to the calibration-v2
  writers in the patched vLLM wheel (`tasks/calib_v2_writers_todo.md`).
- Approach: extend the existing cache-provider pattern
  (`Stage2ReapScoresCacheProvider`) to cov + neuron means; short-circuit
  `_profile_layer` when all sidecars hit.
- Expected speedup: ~30–40 min saved on every Stage 2 row (not just SC).
- Risk: high (touches the patched-vLLM wheel; needs a wheel bump per the
  campaign's wheel-discipline rules in `tasks/calib_v2_writers_todo.md`).
  Bigger-than-A+B in scope.
- Effort: ~20–40 h dev (matches the original L1-cache writers' effort)
  + 1 calibration-v2-writer GPU run + 1 H200-row validation.

**Recommendation**: defer until after A+B is in production. The campaign
has already paid most of this cost (V1+V2 writers exist); the marginal
cost of two more cache-providers is small but the wheel-bump discipline
is heavy. Don't open that surface unless the row-time after A+B is
still > 30 min.

### 3.3 Halt-trigger list

- **Phase A statistical-contract test fails**: the vectorized reservoir
  does not match `Bernoulli(max_samples/total_seen)` per-token presence.
  STOP, revert to the loop, surface the math gap.
- **Phase A SC-row validation drifts > 0.02 from 0.1293**: the sample
  distribution change actually matters numerically. STOP, surface the
  drift, consider Design A (Algorithm L) instead — slower but
  statistically equivalent.
- **Phase B3 LAP solver returns non-byte-identical results on the
  randomized corpus**: STOP, defer B3 indefinitely.
- **Phase C wheel bump required for a 5 % improvement**: the wheel
  bump is too heavy for the gain — STOP and surface to user before
  bumping the MANIFEST.

### 3.4 Go / no-go matrix

| Phase | Go condition | No-go fallback |
|---|---|---|
| A | Unit test passes + golden re-baseline accepted by user | Revert; document the loop as the floor and pivot to B/C |
| B1 | A is green; perm-cache write doesn't alter cost matrix values | Drop B1; symmetric with post path so unlikely to fail |
| B2 | A+B1 green; bf16 merge values within fp32-noise of fp32 | Revert B2 only |
| B3 | A+B1+B2 green AND row time still > 30 min AND `auction-lap` golden-stable on randomized corpus | Drop B3; row time after A+B1+B2 is the new floor |
| C | A+B green AND user authorizes wheel-bump discipline AND sidecar tests byte-identical | Drop C; row time after A+B is the long-term floor |

### 3.5 Existing test surface

- `max_quality/tests/test_stage2_output_space_cost_*.py` — direct unit
  tests of the cost function; will need golden re-baselining after Phase A.
- `max_quality/tests/test_stage2_*.py` — broader Stage 2 tests.
- `max_quality/tests/test_smoke_stage2_resume.py` — the per-layer
  partial-resume invariant; not behaviourally affected by A or B
  (covariance / merge JSON formats unchanged).

---

## 4. Honest risks to raise before any implementation

1. **The reservoir Python loop is conjectured to be ~50–80 % of the row
   from code inspection alone — not measured**. The conviction is high
   (the math says 328 M Python iterations at ~10–40 µs each fits the
   2 h+ gap) but the prior agent's 96 % Hungarian framing was also
   "high-conviction" and got the share wrong. The first concrete step
   should be **a 10-minute profile of the reservoir on a single layer
   on CPU** (no GPU needed) — measure `add()`'s wall-clock at the
   actual bs=32 × seq=2048 shape. This is a free check before any code
   change.
2. **Phase A breaks byte-identity**. Every Stage 2 golden snapshot that
   touches the layer-inputs reservoir will need re-baselining. The
   project's "byte-identical golden snapshot" discipline is load-
   bearing per `MEMORY.md` (plugin-audit complete with 20 golden
   snapshots byte-identical). Re-baselining is the user's call.
3. **Phase B3's GPU LAP option carries a non-uniqueness risk**. LAP
   solutions under cost ties are arbitrary; scipy and GPU solvers will
   pick *different* tied-optimal permutations. The downstream merge
   math is linear in the permutation only modulo the tied set, so the
   end-to-end SC `bpt_gap` *should* be invariant — but this is not
   guaranteed and a single rounding-noise tiebreaker could flip a
   permutation and shift the cost matrix non-trivially.

---

## 5. Constraints honoured

- READ-ONLY codebase analysis — no edits below this file.
- NO GPU spend — no `hf jobs run`, no DataCrunch instance launches.
- No PR language — Phase work, when authorized, lands directly on
  `feat/calibration-v2`.
- No monkey-patching of vLLM (Phase C touches the calibration-v2
  writers properly with a wheel bump + MANIFEST update).
- File paths cited liberally for spot-checking:
  `max_quality/src/moe_compress/stage2/profiling.py:58`,
  `:233`,
  `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py:212`,
  `:276`,
  `:399`,
  `:430`,
  `max_quality/src/moe_compress/stage2/permutation_align.py:114`,
  `:176`,
  `max_quality/src/moe_compress/stage2/plugins/ream_cost_post.py:285`,
  `max_quality/src/moe_compress/stage2/plugins/em_refine.py:277`,
  `max_quality/src/moe_compress/stage2/merging.py:140`,
  `max_quality/src/moe_compress/stage2/plugins/layer_merge.py:448-456`,
  `max_quality/src/moe_compress/stage2/orchestrator.py:1188-1219`,
  `max_quality/configs/qwen36_35b_a3b_30pct.yaml:144-187`,
  `max_quality/docs/stage2_assignment_revision.md:293-307`.

---

*Generated 2026-05-27. Read-only diagnosis; no code changes accompany this
plan.*
