# Stage-3 covariance: single-pass-all-layers feasibility & fidelity

**Date:** 2026-06-14
**Branch:** `research/stage3-single-pass-cov`
**Question:** Stage-3 cov collection windows the MoE layers (`cov_window_size=G`)
and forwards the calibration set **once per window** (`ceil(N_layers/G)` passes).
On a 2×H200 dual-model cross-cov run this multi-pass is a major cost. Can we
capture **all layers in a single forward pass** by offloading the per-layer Gram
accumulators to CPU RAM (≈3 TB box), and is the result the SAME as the
paper/current method (bit-for-bit or within fp tolerance)?

**Verdict: SINGLE-PASS FEASIBLE & PAPER-FAITHFUL** (scoped: the gate_proj B +
cross-cov C keys are *bitwise-identical* to today's per-window result **always**;
the factored down_proj B is bitwise-identical only **at a fixed forward shape**
(same `cov_batch_size`, e.g. the golden `=1`) and otherwise *allclose ~1e-6* —
that residual is **already present today** at any `cov_batch_size>1` and is NOT
introduced by single-pass). Single-pass = setting `G = N_layers` (one window). It
needs the GPU-resident Gram (~30 GB) to fit, OR a modest offload of the
*finalized* Gram to CPU — and an offload-to-CPU path *partially exists already*
via the finalize→spill machinery, though the **hot `_pending` Gram is
GPU-resident by design** and a true CPU-accumulate path does not yet exist.

> ## ⚠️ CRITICAL IMPLEMENTATION CONSTRAINT (read before building the CPU-offload)
>
> **The bitwise-identical guarantee holds ONLY IF the per-token outer-product
> GEMM stays ON GPU and only the pinned-order running-SUM moves to CPU.** The
> required design is exactly:
>
> > **matmul-on-GPU (`flatᵀ@flat` on the input device) → `.to('cpu')` the
> > result → accumulate the running-sum (`add_`) on CPU.**
>
> Do **NOT** move the matmul itself to CPU. A CPU fp32 matmul is **not**
> bit-for-bit equal to a GPU fp32 matmul (different reduction kernels/order), so
> computing `xᵀx` on CPU would **VOID** the bitwise guarantee for *every* key —
> including gate B and cross C — turning a "pure perf change" into a
> results-changing one. The per-token product must be produced by the same GPU
> kernel as today; only the cheap `[d,d]` running-sum accumulation relocates to
> CPU, and the pinned per-sequence accumulation order (`update_grouped`) must be
> preserved on the CPU side.

---

## 1. Why does it window? (accumulator-VRAM bounding, NOT correctness)

The premise is correct: the window bounds **GPU** accumulator memory, and it is
a pure perf/memory knob, not a correctness requirement.

**The window forwards the calib set once per window** — `ceil(N/G)` passes, not
`N` passes:
- `covariance_collection.py:854` — `for window in _iter_windows(indexed, G):`
  iterates contiguous windows of `G` MoE layers.
- `covariance_collection.py:919-947` (non-auto path) — inside each window, a
  single loop `for batch_idx, batch in enumerate(batches):` runs the full
  calibration set once (teacher forward line 945, student forward line 947).
- `covariance_collection.py:552` — docstring: *"forwards the calibration set
  ONCE per window — `ceil(N/G)` passes instead of N."*
- `_iter_windows` (`covariance_collection.py:333-344`) — *"A window of `G` MoE
  layers is hooked … and the calibration set is forwarded ONCE per window."*

So `G=1` ⇒ N passes; `G=N` ⇒ **1 pass** (single-pass-all-layers). The lever
already exists.

**The window's purpose is GPU-Gram bounding** — explicit in the resolver:
- `_resolve_cov_window` (`covariance_collection.py:347-398`): *"each hooked layer
  holds a persistent on-device fp32 Gram … until its window's `finalize_layer`
  runs at window end, plus transient gathered activations. We size
  `G ≈ floor((free − headroom) / per_layer_bytes)`."* (lines 356-360)
- The per-layer cost it budgets: `per_layer_bytes = d_hid²·4 + d_int²·4`
  (`covariance_collection.py:388`) — i.e. the **fp32 gate Gram + fp32 down Gram
  per layer**, summed over all 256 experts implicitly (see §1.2).

**The Gram is GPU-resident until finalize.** `InputCovarianceAccumulator.update`
forms `cov = flatᵀ@flat` **on the input device** and keeps it in `_pending`
on-GPU (`activation_hooks.py:1018-1029`); the docstring is explicit: *"The
covariance tensor stays on the input device (typically GPU) for the lifetime of
a layer's profile; `finalize_layer` does the single GPU→CPU transfer per key"*
(`activation_hooks.py:992-996`). `finalize_layer` is the **only** GPU→CPU move,
casting to `storage_dtype` then `.cpu()` (`activation_hooks.py:1097`). It runs
**at window end** (`covariance_collection.py:1031-1034`).

⇒ Windowing exists solely to cap how many layers' worth of GPU-resident
`_pending` Grams are live at once. Nothing about a smaller window changes the
math (proven in §3).

### 1.2 Accumulator footprint — does all-layers-at-once fit?

Model: Qwen3.6-35B-A3B (`configs/qwen36_35b_a3b_reap_exact.yaml:21`), d_hid≈5120,
d_int≈2048, 256 experts. The accumulator docstring gives a per-layer total of
**≈30 GB** (`activation_hooks.py:954-963`).

> **⚠️ Pre-existing docstring mislabel (`activation_hooks.py:957-958`):** that
> docstring literally writes *"gate_proj : 256 × [2048, 2048] … ≈ 4 GB"* and
> *"down_proj : 256 × [5120, 5120] … ≈ 25.6 GB"* — which is **physically
> backwards**. The Gram dimension is the *matrix input* dim: gate_proj's input is
> the pre-routing hidden state (d_hid=5120) ⇒ `[5120,5120]`; down_proj's input is
> the intermediate activation (d_int=2048) ⇒ `[2048,2048]`. So the **per-expert
> byte figures are swapped between the two rows** in the docstring. The
> *physically correct* mapping is used below. The ~30 GB/layer total and the
> ~2 TB all-layers total are **unaffected** (it is the same two numbers, just
> attached to the correct matrix). This is a pre-existing labeling bug in
> `activation_hooks.py`, noted here so the footprint reasoning isn't mistrusted.

The Gram is **per (layer, expert, matrix)** but **only the experts a token routes
to** get a Gram, and the per-expert Gram dimension is the *matrix input* dim
(physically-correct mapping):
- gate_proj input = the pre-routing hidden state ⇒ `[d_hid, d_hid]` =
  `[5120,5120]` fp32 = 105 MB/expert.
- down_proj input = the intermediate activation ⇒ `[d_int, d_int]` =
  `[2048,2048]` fp32 = 16.8 MB/expert.
- The student (post-prune) has ~180–200 experts/layer, not 256.

**All-layers-at-once GPU footprint** (B-cov, student): worst case 256 experts ×
40 layers × (105 + 16.8) MB ≈ **1.25 TB fp32**. That does **NOT** fit even an
H200's 143 GB. This is exactly why the current default windows down to a handful
of layers (the resolver budgets ~30 GB/layer ⇒ `G≈3–4` on a 143 GB H200 after
the ~70 GB model).

**With cross-cov C** (gate-only, per D6 — `covariance_collection.py:69-95`): C
adds one more `[5120,5120]` fp32 per (layer, student-expert), ≈ same magnitude as
the gate B term ⇒ roughly **+0.8 TB**. Plus the teacher dense `[T,d_in]` capture
buffers per window layer (`covariance_collection.py:617-620, 663-673`). Total
all-layers B+C ≈ **~2 TB fp32 GPU-resident** — far beyond VRAM, but **fits the
3 TB CPU RAM** (and ~1 TB in fp16, the persisted dtype —
`covariance_collection.py:97-111`).

⇒ **Single-pass-all-layers is infeasible with today's GPU-resident `_pending`**
(needs ~2 TB VRAM). It becomes feasible only if the per-layer Gram is
**offloaded to CPU** after each window's tokens for that layer are consumed —
which is the whole question (§2).

---

## 2. Is single-pass-all-layers feasible? Does an offload path exist?

**Structurally, hooking all layers in one pass is trivially supported.** Each
forward already computes every layer's activations; `capture_experts` is a
pure `forward_pre_hook` that fires per layer as the native forward reaches it
(`activation_hooks.py:1577-1737`), and the window loop already
`ExitStack`-enters a hook on *every layer in the window*
(`covariance_collection.py:882-910`). Setting `G=N` hooks all 40 layers and the
single batch loop accumulates every layer's Gram in one pass. No new capture
code is needed.

**The blocker is purely accumulator residency**, not pass structure. Today
`_pending` is GPU-resident (§1). For single-pass you must move each layer's Gram
to CPU so VRAM holds only model + the *current* layer's transient activation.

### Does an offload path already exist? PARTIALLY.

- **Finalized-Gram → CPU → disk spill EXISTS and is the live default.**
  `finalize_layer` already moves the finalized per-layer Gram to CPU
  (`activation_hooks.py:1097`); `spill_layer_to_disk`
  (`activation_hooks.py:1135-1194`) then writes it to disk and **evicts it from
  CPU RAM** (Phase 3, lines 1183-1188). The window loop spills each layer right
  after finalize via a background `ThreadPoolExecutor`
  (`covariance_collection.py:1036-1047`). So the *finalized* Gram never
  accumulates unboundedly in RAM — it streams to disk per layer. This is the
  exact "stream-each-shard-to-disk-and-free" fix the old 172 GB memory note
  called for (see §2.1).

- **A true CPU-resident *hot accumulator* (`_pending` on CPU) does NOT exist.**
  `update` hard-codes the matmul + in-place add on the input device
  (`activation_hooks.py:1018-1029`); there is no flag to place `_pending` on CPU.
  For single-pass you would either (a) add a `cpu_accumulate` mode to `update`
  (matmul on GPU, `.to("cpu")` the *result*, accumulate into a CPU `_pending`),
  or (b) — cleaner — keep the existing per-window finalize+spill but **raise G to
  N and finalize+spill each layer as soon as its last token is processed**. Note
  the current loop finalizes ALL window layers only at window *end*
  (`covariance_collection.py:1031`), so with `G=N` it would still hold all 40
  layers' GPU `_pending` simultaneously → OOM. Single-pass therefore needs the
  CPU-accumulate variant (a), not just `G=N`.

### 2.1 Does the old "172 GB wall" still apply? NO — different codebase + fixed.

The 172 GB wall (memory `project_input_cov_offload_172gb_wall`) was in
`build_self_traces_calib_vllm.py` (the **Stage-2/vLLM calibration** path), NOT
this Stage-3 plugin. Two distinct problems there, **both already solved in the
Stage-3 plugin's design**:
1. *RAM-accumulate-all-windows-then-rewrite-whole-dict* (43→86→129→172 GB
   monotonic growth + full rewrite each window). The Stage-3 plugin instead
   **spills each layer straight to disk and frees it**
   (`spill_layer_to_disk` Phase-3 eviction, `activation_hooks.py:1183-1188`;
   per-layer spill in the window loop, `covariance_collection.py:1036-1047`).
2. *Size*: 172 GB fp32. The Stage-3 plugin persists **fp16** (halved —
   `covariance_collection.py:97-111`) and never holds the full dense set in RAM.

The remaining constraint for single-pass is only the **GPU**-resident hot
`_pending` (≈2 TB B+C at all-layers, §1.2). On a 3 TB-RAM box a CPU-accumulate
`_pending` fits comfortably (≈1 TB fp16 / ≈2 TB fp32 for B+C across all layers,
and even that is only the transient pre-spill set if you spill per layer
on-the-fly). **The 3 TB RAM removes the wall; the GPU residency is the only thing
to change.**

---

## 3. Fidelity — THE KEY QUESTION

**Verdict: the final Gram is bitwise-identical for gate_proj B and cross-cov C;
allclose ~1e-6 for factored down_proj B — and the down_proj residual is NOT
caused by single-pass.** Pass-count and accumulator-location are pure
perf/memory; they do not change the math.

### (a) The math is identical regardless of pass-count or accumulator location

The Gram is a **linear sum of per-token outer products**:
`S = Σ_t xₜ·xₜᵀ`, `C = Σ_t x_pre,ₜ·x_post,ₜᵀ`. This is stated and relied upon
throughout:
- `update` forms `flatᵀ@flat` and `add_`s it into `_pending`
  (`activation_hooks.py:1021, 1029`).
- The DP cross-replica reduce is **exactly `B = Σ_r B_r`** because *"The Gram
  accumulator is a linear sum of per-token outer products, so the cross-replica
  reduce is exactly `B = Σ_r B_r` — summed key-wise in fp32"*
  (`covariance_collection.py:256-260`). If summing across **processes** is exact,
  summing the same tokens across **passes** or on a **different device** is the
  same exact sum.
- Windowing adds **zero error**: *"The per-(layer,expert,matrix) accumulators are
  additive and order-stable per key, so windowing adds zero error on top of the
  native baseline"* (`covariance_collection.py:560-562`). A single window (`G=N`)
  is the degenerate case of "zero error."

Whether a layer's tokens are summed in pass-3-of-5 (current) or pass-1-of-1
(single) does not change **which tokens** are summed (the full calib set either
way) nor the **per-key accumulation order** (see (b)). CPU-fp32 vs GPU-fp32
`add_` of the *same operands in the same order* gives the same IEEE-754 result.

### (b) fp reduction ORDER is pinned and pass-count-independent

The accumulation order within a key is fixed by the **per-sequence reduction
pin**, independent of the forward batch size:
- `update_grouped` splits captured rows by source sequence and accumulates *"one
  update per source sequence in ASCENDING seq order … making the running Gram
  independent of the forward batch's sequence-merging"*
  (`activation_hooks.py:1032-1049`; routed in `covariance_collection.py:692-695,
  818-821`).
- Consequence (`_resolve_cov_batch_size` docstring,
  `covariance_collection.py:430-442`): *"gate_proj/up B and the cross-cov C: the
  Gram is now BITWISE-INVARIANT to `cov_batch_size`."*

The window/pass structure does not reorder a key's per-sequence sum: each
(layer, expert) key receives exactly the same sequences in the same ascending
order whether collected in 1 pass or 5. So **pass-count does not perturb the
reduction order** for gate B / cross C.

### (c) What DOES change the result — and single-pass does not touch it

The **only** non-bitwise term is the **factored down_proj B**, and the cause is
upstream **forward-activation** drift, not the accumulator:
- `intermediate_cb` ACCURACY NOTE (`covariance_collection.py:806-816`): the
  down_proj operand `act_fn(gate)*up` is produced by a *"PADDED batched `bmm` …
  whose fp reduction is perturbed by the forward batch SHAPE upstream"* ⇒
  down_proj Gram is *"allclose (~1e-6), NOT bitwise, across cov_batch_size"*.

This residual depends on the **forward batch shape** (`cov_batch_size`), which is
*orthogonal* to pass-count and accumulator location. Single-pass-all-layers at
`cov_batch_size=1` (the golden setting) leaves the forward shape identical to
today's per-window `cov_batch_size=1` run ⇒ **even the down_proj B is
bitwise-identical to today**. If single-pass is combined with a *larger*
`cov_batch_size` for throughput, the down_proj B picks up the same bounded
~1e-6 drift that today's `cov_batch_size>1` already has — a pre-existing,
quality-neutral property, not a new one.

⇒ **Accumulator location (GPU vs CPU) and pass-count (N vs 1) change neither the
token set, the per-key reduction order, nor the operands. The Gram is the same.**
fp32 `add_` is associative-order-sensitive only, and the order is pinned.

---

## 4. Cross-cov C in a single pass — feasible, no new obstruction

Cross-cov C needs teacher and student activations **aligned per token**. The
current code already does this **within one forward-pair per window**, and
single-pass (one window of all layers) is just the same mechanism with more
layers hooked:

- Per batch, the teacher forward fills `_teacher_dense[layer] : [T, d_in]` for
  **every hooked window layer** (`covariance_collection.py:663-673`); the student
  forward then gathers the matching teacher rows via one `index_select` and
  accumulates `C += X_preᵀ@X_post` per (layer, student-expert)
  (`covariance_collection.py:754-795`).
- The teacher store is keyed by layer and **cleared per BATCH, not per layer**,
  *"so it holds all G window layers' teacher rows during the student forward"*
  (`covariance_collection.py:558-561, 928-931`). At `G=N` it simply holds all 40
  layers' teacher rows for the current batch — sized `T = rows·seq`
  (`covariance_collection.py:931, 1019`), i.e. the **per-batch** token count, NOT
  the whole calib set. So the teacher buffer scales with batch size, not with G:
  **all-layers does not blow up the teacher store** (it is `N · [T,d_in]` fp32 ⇒
  at bs=1, T=seq≈2–4 K, that is 40 × few-MB = ~GBs, trivially on-GPU or
  offloadable).
- The position-join (teacher pre-routing hidden gathered at the positions the
  *student* routes to expert e) is per-(layer, student-expert) and **purely
  intra-batch** (`covariance_collection.py:754-765`); it does not span passes, so
  single-pass does not break it. The DP path already proves per-shard C grams sum
  exactly across processes (`covariance_collection.py:256-260, 1320-1324`),
  which is the same linearity that makes single-pass C correct.

⇒ **Cross-cov is single-pass-safe.** The only growth at `G=N` is N teacher
`[T,d_in]` buffers (batch-sized, not calib-sized) plus the N layers' C
`_pending` Grams — the latter is the same ~0.8 TB GPU-residency issue as B
(§1.2), addressed by CPU-accumulate, not by anything cross-cov-specific.

---

## 5. Net perf cost/benefit (honest order-of-magnitude)

**Benefit:** `ceil(N/G)` → 1 dual-forwards. With the current default
auto-window (`G≈3–4` on a 143 GB H200 ⇒ ~10–13 windows over 40 layers), or even
a conservative `G=8` ⇒ 5 windows, single-pass saves **~5–13× the dual-forward
work**. On a 2×H200 dual-model cross-cov run the dual-forward is the dominant
cost, so this is a **~5–13× reduction in the forward portion** — large.

**New cost introduced by CPU-offloaded single-pass:**
1. **GPU→CPU transfer of finalized Grams.** This *already happens once per layer*
   today (`finalize_layer` `.cpu()`, `activation_hooks.py:1097`) and is spilled
   to disk async (`covariance_collection.py:1036-1047`). Single-pass does the
   same total volume of GPU→CPU once per layer — **no net increase** vs today's
   per-window finalize. (Today already finalizes every layer exactly once.)
2. **CPU-side hot accumulation** (if using the `cpu_accumulate` `_pending`
   variant): each (layer,expert) key's `add_` runs on CPU instead of GPU. CPU
   fp32 `[5120,5120] add_` ≈ 26 M FLOP, ~sub-ms; over 256 experts × 40 layers ×
   (calib batches) this is real but **bounded by the per-token outer-product
   count, which is identical to today's GPU accumulation** — you are moving the
   same `add_`s from GPU to CPU. On a many-core 3 TB box the CPU `add_`s overlap
   the next GPU forward and are unlikely to dominate. Order-of-magnitude: the
   added CPU-accumulate + D2H copy is **a small fraction** of one saved
   dual-forward, and you save `N/G − 1` of them.

**Honest verdict on perf:** single-pass is **clearly net-faster** on the 2×H200
dual-model run — the saved `~5–13×` dual-forwards dwarf the CPU-accumulate/D2H
overhead (which is largely already paid today and overlaps the forward). The
*implementation* cost is adding a CPU-resident `_pending` mode to
`InputCovarianceAccumulator.update` (or an on-the-fly per-layer
finalize+spill-as-you-go), NOT new capture logic.

---

## 6. Bottom line

| Question | Answer |
|---|---|
| Does it really need a pass per window? | The window forwards calib **once per window** (`ceil(N/G)` passes). `G=N` = single pass. The window exists only to bound **GPU**-resident `_pending` Gram (~30 GB/layer), not for correctness. (`covariance_collection.py:347-398, 552, 560-562`) |
| Single-pass-all-layers feasible? | **Yes**, but needs the hot `_pending` Gram **off GPU** — all-layers B+C ≈ ~2 TB fp32 won't fit VRAM but fits the 3 TB CPU RAM (~1 TB fp16). |
| Does an offload path already exist? | **Partially**: finalized-Gram→CPU→disk-spill-and-evict EXISTS and is the live default (`finalize_layer` `.cpu()` + `spill_layer_to_disk` eviction). A **CPU-resident hot `_pending`** does NOT exist — `update` hard-codes the matmul/accumulate on the input device (`activation_hooks.py:1018-1029`). That ~10-line addition is the only code gap. |
| Old 172 GB wall still apply? | **No** — that was the vLLM Stage-2 calib script, with a RAM-accumulate-all-windows flaw + fp32. The Stage-3 plugin already streams per-layer to disk and frees it, and persists fp16. 3 TB RAM removes the size wall. |
| Fidelity verdict | **Always bitwise-identical** for gate_proj B + cross-cov C (pinned per-sequence reduction order, `update_grouped`, batch/pass-invariant). Factored down_proj B is bitwise-identical **only at a fixed forward shape** (same `cov_batch_size`) and otherwise **allclose ~1e-6** — that residual is an upstream forward-shape effect already present at `cov_batch_size>1` today, NOT introduced by single-pass. ⇒ at the golden `cov_batch_size=1`, single-pass is bitwise-identical to today **end-to-end** (all keys). (`covariance_collection.py:256-260, 430-442, 560-562, 806-816`) |
| Cross-cov single-pass-safe? | **Yes** — teacher store is cleared per-batch and already holds all window layers' rows; position-join is intra-batch; DP linearity proves the sum is exact. Teacher buffer scales with batch size, not G. (`covariance_collection.py:558-561, 663-673, 754-795`) |
| Perf | **Net faster** — saves `~5–13×` dual-forwards on the 2×H200 run; the added CPU-accumulate/D2H is small and largely already paid + overlappable. |

**FINAL: SINGLE-PASS FEASIBLE & PAPER-FAITHFUL.** gate_proj B + cross-cov C are
*always* bitwise-identical to today; the factored down_proj B is too at a fixed
forward shape — so at the golden `cov_batch_size=1` it is a pure perf change with
**zero** quality/fidelity impact (bitwise-identical end-to-end, all keys). The
one required code change is a CPU-resident hot-accumulator mode for
`InputCovarianceAccumulator.update` — **matmul on GPU, `.to('cpu')` the result,
`add_` on CPU** (see the CRITICAL IMPLEMENTATION CONSTRAINT box above; doing the
matmul on CPU would VOID the bitwise guarantee for every key). Everything else —
the per-layer hooks, finalize, fp16 spill-and-evict, linearity, reduction pin —
is already in place. **Do NOT modify main while the ablation runs** (per task
constraint); this is research only.
