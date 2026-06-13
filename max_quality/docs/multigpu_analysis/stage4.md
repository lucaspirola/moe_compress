# Stage 4 (EoRA compensation) — Multi-GPU State Analysis

Read-only analysis. Files read in full:
- `src/moe_compress/stage4/plugins/eora_compensation.py` (791 lines)
- `src/moe_compress/stage4/plugins/eora_inputs.py` (353 lines)
- `src/moe_compress/stage4/plugins/input_cov_cache.py` (104 lines)
- `src/moe_compress/stage4/orchestrator.py` (268 lines)
- `src/moe_compress/stage4/context.py`, `stage4/stage.py`
- `src/moe_compress/utils/auto_batch.py` (full)

Cross-ref: landed N-GPU EoRA at main `e395ad0` (merge of `9841971`).

---

## TL;DR

Stage 4 is **pure linear algebra** (eigh + Gram-SVD per expert matrix). It does
**ZERO forward passes** → **auto-batch is N/A for the entire stage** (there is no
forward batch to size; the relevant resource is per-GPU VRAM for the eigh/SVD
working set, which is bounded by `d_in`/`d_out`, not by any batch dimension).

The compensation plugin ALREADY has N-GPU **task-parallel** scaffolding (lever 1,
`eora_workers`): each eligible expert is *placed* on a worker device and solved
there, then gathered as a disjoint row back to the layer home device. It
auto-detects N GPUs, is 1-GPU byte-identical, and all rank/budget decisions are
device-independent.

**BUT the single most important finding:** the per-expert fan-out is **placed but
not concurrently executed**. The dispatch loop `for e in eligible:`
(`eora_compensation.py:693`) calls `_solve_expert_tile(...)` and immediately
`U_corr[e] = Uc.to(device=dev, ...)` (`:709-710`) — a **blocking** device→home
copy — every iteration. There is **no** `ThreadPoolExecutor`, no
`torch.cuda.Stream`, no process pool, no async anywhere in `stage4/` (grep
confirms zero matches). So experts banded to `cuda:1..cuda:N-1` are computed
**one at a time, serially**, each blocking on its own gather before the next
starts. The N-GPU lever as landed gives **near-zero wall-clock speedup** — it is a
correct device-distribution skeleton missing its concurrency engine.

---

## Per-plugin table

| Plugin | (a) Computes | (b) Compute profile | (c) Current device usage / already-MG? | (d) Scheme | (g) Effort / Risk / Speedup |
|---|---|---|---|---|---|
| **eora_inputs** | Loads A-cov dict + Stage-3 originals `.pt` (~50 GB) + ranks snapshot; manifest validation; builds MoE layer list | Pure disk I/O (`torch.load` map_location="cpu") + dict ops. No GPU linalg. | All CPU. Single-process. NOT multi-GPU. | **NOT-WORTH-IT** (I/O-bound serial load; could parallel-read but disk is the wall) | low / low / negligible |
| **input_cov_cache** | Cache-side provider: loads V2 covariance sidecar → populates `ctx.A_cov`; infers storage dtype | Disk I/O via `load_covariance`. No linalg. | All CPU. Single-process. | **NOT-APPLICABLE** (pure cache lookup) | — / — / — |
| **eora_compensation** | Per-(layer,expert,matrix) EoRA Algorithm-1 residual kernel: `eigh(A)` whitening + Gram-side SVD of `ΔW·Q'`, back-project, widen `U/V` | Heavy GPU linalg: 1× `eigh(A)` [d_in×d_in] per (expert, gate/down) + 1× `eigh(Gram)` [min(d_out,n_keep)²] per (expert,matrix). No forward pass. | **ALREADY MG (lever 1, task-parallel) — but PLACED-NOT-CONCURRENT.** Per-expert solve relocated to worker device; serial blocking gather. | **ALREADY-MULTI-GPU (incomplete)** → upgrade to true concurrency (streams/threads) | med / med / **high (latent)** |

---

## eora_compensation — the EoRA kernel (the only stage-4 compute plugin)

### (a) What it computes
For each MoE layer, for each matrix `name ∈ {gate_proj, up_proj, down_proj}`:
1. Per-matrix EoRA rank budget `r_per_expert` from `compensation_budget_pct` of
   Stage-3 savings, capped at `eigenspace_rank_cap=128` and `min(d_out,d_in)`
   (`:627-634`). **Pure integer arithmetic on shapes — device-independent.**
2. For each *eligible* expert (one with an `originals` entry): the EoRA
   Algorithm-1 kernel `_solve_expert_tile` → `_compute_eora_factors`
   (`:389-467`, `:235-386`): build `ΔW = W_orig − U_e·V_e`, eigendecompose
   covariance `A = QΛQ^T`, project `ΔW·Q√Λ`, Gram-side rank-r SVD, back-project
   `U_corr/V_corr`, gather into `U_corr[e]/V_corr[e]`.
3. `fe.widen_rank(name, U_corr, V_corr, ...)` appends the correction columns.
4. Trackio emit + per-layer spill (`_spill_layer`).

### (b) Compute profile — pure linear algebra, NO forward pass
- **`eigh(A)`** on `[d_in, d_in]` fp32 — once per (expert, gate_proj) and once
  per (expert, down_proj). up_proj **reuses** gate's spectrum (Lever A memo,
  `:446-454`, `:622`,`:699`,`:706`) — bit-identical, shares the same `A` object.
- **`eigh(Gram)`** on the smaller of `[n_keep,n_keep]` or `[d_out,d_out]`
  (`:352-367`) — Lever C avoids materialising the full SVD.
- A few matmuls (`delta @ Q_prime`, back-projection).
- **No `model(...)` call, no generate, no token batch.** The covariance `A` was
  produced upstream (Stage 2 / V2 calibration capture); Stage 4 only consumes it.

→ **This is the answer to KEY QUESTION 2:** Stage 4 does **no forward passes**.
It is purely cov-driven linear algebra. The resource to manage per GPU is the
**eigh/SVD working-set VRAM**, not a forward batch.

### (c) Current device usage — already-multi-GPU (lever 1), but serial execution

**The lever (`eora_workers`):**
- `orchestrator._resolve_eora_workers` (`orchestrator.py:69-85`) reads
  `multi_gpu.eora_workers` (default 1), clamps to `min(requested, device_count())`
  floor 1. Mirrors Stage 3's `_resolve_cov_replicas`. **Auto-detects N GPUs**
  (`torch.cuda.device_count()`), no 1x/2x special-casing. Set once on root ctx
  (`orchestrator.py:120`), read by every `compensate_layer` (`:604`).
- Per-layer (`:678-691`): `effective_workers = min(eora_workers, len(eligible))`;
  experts split into **contiguous bands by sorted index** across worker devices
  (`device_of` map, `:687-691`). `_resolve_worker_devices` (`:470-486`) derives
  ascending `cuda:0..cuda:N-1`, or uses the explicit `eora_worker_devices` test
  seam (CPU stand-in for CI), or degrades to N× home device.

**How it shards — EXACTLY (KEY QUESTION on sharding granularity):**
- **PER-EXPERT, within a (layer, matrix) pass.** NOT per-layer, NOT per-matrix.
  The unit of distribution is one expert's `(W_orig, U_e, V_e, A)` → one
  `_solve_expert_tile` call (`:389`, the "task-parallel unit").
- Bands are **contiguous slices of the sorted eligible-expert list**
  (`eligible[w*per:(w+1)*per]`, `:690`). Worker `w` owns experts in band `w`.
- The gate→up spectrum memo is keyed **per-expert** (`gate_spectra[e]`, `:622`,
  `:706`) and survives the whole gate→up window; gate and up resolve their bands
  **independently**, so an expert's up_proj pass may land on a *different* device
  than its gate_proj pass — hence `_compute_eora_factors` relocates a supplied
  spectrum to `delta.device` (`:321-325`) to avoid a cross-device matmul. This is
  the cross-device-spectrum bug the impl review caught (per merge message).

**How it gathers:**
- `U_corr[e] = Uc.to(device=dev, dtype=dtype)` / `V_corr[e] = Vc.to(...)`
  (`:709-710`) — each tile copied back to the **layer home device** `dev`
  (`fe.gate_proj_U.device`, `:596`) into its **disjoint row** `e`.
- Gather is in **ascending-e order** with NO cross-expert reduction
  (`:651-657`, `:708`), so serial and N-GPU fills are byte-for-byte identical.
- `test_eora_gather_order_deterministic` pins this.

**Device-independence of rank/budget decisions (the "M2-guard" analog):**
- `r_per_expert`, `take_eff`, the noise-floor `keep_mask`, and the eligible set
  are all computed from **shapes and the covariance values**, never from device
  identity. `take_eff = min(r, min(d_out, n_keep))` (`:350`) is identical on any
  device. The Gram-SVD deliberately does NOT add an `evals>0` positivity filter
  (`:344-348`) precisely so `take_eff` matches production exactly regardless of
  device. The double-widen `assert` (`:723-728`) compares against `stage3_ranks`,
  a device-independent snapshot. **All rank decisions stay device-independent.** ✓

**1-GPU byte-identical:** ✓ confirmed by construction and by
`test_eora_taskparallel_equivalence`. When `eora_workers<=1` (default), `device_of
= {e: dev for e in eligible}` (`:683`) → every expert on home device, same serial
order as before the lever landed. The `.to(device=dev)` gather is a no-op when the
tile already lives on `dev`. Goldens green (merge message + memory note
`project_multigpu_stage3_landed`).

### (d) Scheme: ALREADY-MULTI-GPU (task-parallel) — but execution is SERIAL

**The gap.** Grep for `ThreadPool|ProcessPool|Executor|threading|multiprocessing|
Stream|spawn|concurrent` across `stage4/` returns **ZERO matches**. The dispatch
is the plain blocking loop:

```python
for e in eligible:                                    # :693
    tgt = device_of[e]
    Uc, Vc, take_eff, ... = _solve_expert_tile(..., tgt, ...)   # :696  runs on tgt
    U_corr[e] = Uc.to(device=dev, dtype=dtype)        # :709  BLOCKING gather
    V_corr[e] = Vc.to(device=dev, dtype=dtype)        # :710  BLOCKING gather
```

Because CUDA ops on `tgt` enqueue async but the `.to(device=dev)` gather (`:709`)
**synchronizes the producing stream** before the next `_solve_expert_tile`
begins, experts banded to different GPUs run **one after another**, never
overlapped. The landed lever is a correct *device-placement* skeleton; it is
**missing the concurrency engine** that would actually overlap the per-GPU eigh/
SVD work. As shipped, wall-clock ≈ serial single-GPU (minus nothing, plus a
little cross-device copy overhead).

→ **KEY QUESTION 1 answer:** the existing task-parallel EoRA is **NOT optimal** —
the *entire* per-expert solve is "still single-GPU" in the wall-clock sense
because nothing runs concurrently. There is no separate L-BFGS / activation-
weighted refine in this implementation (EoRA here is closed-form eigh+SVD, not an
iterative solver), so the serialization is the whole compute, not a tail.

### (e) Result-preserving mechanism
`_solve_expert_tile` reads **ONLY** this expert's tensors (`W_orig/U_e/V_e/A`) —
nothing from any other expert (`:411-412` docstring + body). It is a **pure
function of its inputs**; relocating it to any device is a pure relocation (same
kernels, same arch ⇒ bit-identical). Per-expert SVDs are **independent** → true
task-parallelism is **exact**, no reduction, no approximation. The disjoint-row
ascending-e gather guarantees the assembled `U_corr/V_corr` is identical to serial.
This is the cleanest possible parallelism: embarrassingly parallel, exact.

### (f) HOW auto-batch per-GPU sizing is preserved
**It is N/A and that is correct.** `_compute_eora_factors` / `_solve_expert_tile`
have **no forward batch** — the work is `eigh`/`svd`/`matmul` on weight-shaped
tensors (`[d_in,d_in]`, `[d_out,d_in]`). `auto_batch.size_batch` exists to size a
**forward micro-batch** of *sequences* (its `cost_probe_fn(micro_batch)` runs "ONE
forward of micro_batch sequences", `auto_batch.py:147-151`). Stage 4 feeds no
sequences, so:
- There is **no `resolve_batch` call** anywhere in `stage4/` (grep: zero), and
  there should not be.
- `EoraCompensationPlugin` declares no `FidelityClass`; it is not in
  `_V1_ELIGIBLE`, so even if some glue called `resolve_batch` it would no-op
  return `fixed_batch` (`auto_batch.py:182-183`).
- The auto-batch **HARD REQUIREMENT** ("each device probes its own VRAM") is
  satisfied **vacuously**: there is no batch dimension to size. The per-GPU
  resource is the eigh/SVD working set, which is fixed by `d_in`/`d_out` and the
  number of *experts placed on that card*, not by a batch.

→ **KEY QUESTION 3 answer:** the per-expert distribution does **NOT** probe
per-GPU VRAM to decide how many experts to place per card. Banding is purely
**count-based and contiguous** (`per = ceil(len(eligible)/effective_workers)`,
`:688`) — even split, VRAM-oblivious. This is the **task-parallel analog of
auto-batch** and it is currently **absent**. On heterogeneous GPUs, or when one
card already holds the resident model, a naive even split can OOM the fuller card
while the others idle. A VRAM-aware band assignment (place fewer experts on the
home card that holds the model, more on the empty workers) would be the correct
analog — but note the per-expert working set is small (a few `[d_in,d_in]` fp32
matrices), so OOM risk is low in practice; this is a refinement, not a blocker.

### (g) Effort / Risk / Speedup
- **Make the placed fan-out actually concurrent** (the real win):
  - **Effort: medium.** Wrap the per-expert solves in a `ThreadPoolExecutor`
    (one thread per worker device — Python threads suffice because the heavy work
    is in CUDA kernels that release the GIL and run async on per-device streams),
    OR issue all band solves on independent `torch.cuda.Stream`s and gather after
    a single sync. The gather must still write rows in ascending-e order to
    preserve the byte-identical golden — collect tiles, then assemble.
  - **Risk: medium.** Concurrency + CUDA streams + determinism is the classic
    trap. Must keep: ascending-e gather order, the per-expert gate→up spectrum
    memo (now shared across threads — needs a thread-safe dict or per-band memo),
    and the trackio/residual accumulation (`res_*_acc`, `:714-715`) which is an
    on-device running sum (order-independent for the sum, but the `.item()` sync
    must come after all joins). 1-GPU path must stay byte-identical (gate behind
    `effective_workers>1`).
  - **Speedup: high (latent).** Up to ≈N× on the per-layer expert loop for
    N workers, since the solves are independent and currently fully serialized.
    This is the entire point of the lever and it is unrealized today.
- **VRAM-aware banding** (the auto-batch analog): low effort, low risk, low-to-
  modest payoff (only matters on heterogeneous cards / tight VRAM). Probe each
  worker's free VRAM, weight band sizes inversely to resident load. Optional.

---

## eora_inputs — input loader

- **(a)** Loads A-cov (V2 cache hit short-circuits, else `_stage2_input_covariance.pt`),
  Stage-3 originals (`~50 GB .pt`), manifest-validates both, builds MoE layer list,
  sets up partial dir, snapshots `stage3_ranks` from in-memory `FactoredExperts.ranks`.
- **(b)** Pure **disk I/O** + dict construction; `torch.load(..., map_location="cpu")`.
  No GPU, no linalg, no forward pass.
- **(c)** All CPU, single process. NOT multi-GPU.
- **(d) NOT-WORTH-IT.** The wall is sequential disk read of a ~50 GB file; sharding
  the read across processes is possible but disk bandwidth, not compute, is the
  limit. No multi-GPU lever applies.
- **(f)** Auto-batch N/A (no forward).

## input_cov_cache — V2 covariance cache provider

- **(a)** On cache hit, loads the V2 sidecar covariance dict → `ctx.A_cov` +
  `ctx.a_storage_dtype`. On miss returns `None` (live loader falls through).
- **(b)** Pure disk I/O (`load_covariance`). No linalg, no forward pass.
- **(c)** CPU, single process.
- **(d) NOT-APPLICABLE.** A cache lookup; nothing to parallelize across GPUs.
- **(f)** Auto-batch N/A.

---

## Answers to the three KEY QUESTIONS

1. **Is the existing task-parallel EoRA optimal? Any serialized parts?**
   **No, not optimal.** The per-expert solves are *placed* on N devices but
   *executed serially* — the dispatch loop blocks on each tile's gather before
   the next solve (`eora_compensation.py:693-710`); there is **no thread/stream/
   process concurrency anywhere in `stage4/`** (grep-confirmed). The *whole*
   per-expert compute is effectively single-GPU wall-clock. There is no separate
   L-BFGS / iterative refine — EoRA here is closed-form `eigh`+Gram-SVD — so the
   serialization is the entire kernel, not a tail. Realizing the lever needs a
   concurrency engine (ThreadPoolExecutor or per-device CUDA streams) over the
   already-correct device-placement skeleton.

2. **Forward passes (auto-batch applies) or pure cov-driven linear algebra?**
   **Pure linear algebra.** Zero forward passes; covariance `A` is consumed from
   upstream. **Auto-batch is N/A for the entire stage** — there is no sequence
   batch to size. The relevant per-GPU resource is the eigh/SVD working set
   (bounded by `d_in`/`d_out` and #experts-per-card), not a batch dimension.
   Consequently no `resolve_batch` is or should be called in `stage4/`.

3. **Does the per-expert distribution probe per-GPU VRAM to decide placement?**
   **No.** Banding is contiguous count-based even-split
   (`per = ceil(len(eligible)/workers)`, `:688`), VRAM-oblivious. The
   task-parallel analog of auto-batch (VRAM-aware band sizing) is **absent**. Low
   practical risk because per-expert working sets are small, but it is the correct
   refinement for heterogeneous GPUs / tight VRAM (e.g. the home card also holds
   the resident model).

---

## What's ALREADY multi-GPU vs. what REMAINS

**Already (landed `e395ad0`):**
- `eora_compensation`: N-GPU **device-placement** of per-expert solves
  (auto-detect N, contiguous bands, disjoint-row ascending-e gather, cross-device
  spectrum relocation). 1-GPU byte-identical. Rank/budget/double-widen decisions
  all device-independent.

**Remains:**
1. **Concurrency engine** (the high-value item): make the banded solves run
   *concurrently* via ThreadPoolExecutor or per-device CUDA streams; today they
   are serial. This is the only change that converts the existing skeleton into
   actual wall-clock speedup.
2. **VRAM-aware banding** (optional refinement): the auto-batch analog — weight
   band sizes by each worker's free VRAM instead of an even count split.
3. **eora_inputs / input_cov_cache**: I/O-bound, NOT-WORTH-IT / NOT-APPLICABLE.

**Auto-batch:** N/A stage-wide and correctly so — no forward batch exists. The
"per-GPU VRAM-aware sizing" hard requirement maps onto item 2 (VRAM-aware
expert-per-card placement), which is currently not implemented.
