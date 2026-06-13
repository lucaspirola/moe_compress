# Stage 3 (AA-SVD) — Multi-GPU State Analysis

Read-only deep dive. Every plugin in `max_quality/src/moe_compress/stage3/plugins/*.py`
plus `orchestrator.py`, `context.py`, `stage.py` read in full. Stage 3 is the stage
with the MOST existing multi-GPU support; this documents exactly what is there and
what is NOT, per plugin, with the auto-batch per-GPU-sizing story.

Code root: `max_quality/src/moe_compress/`. All file:line cites below are relative to
that root unless absolute.

---

## TL;DR

Stage 3 has TWO production multi-GPU levers, BOTH only on the covariance-collection
phase:

1. **MODEL-SHARD (memory fit, not speed)** — `model.device_map: "balanced"` /
   `"auto"` shards teacher + student across N GPUs via accelerate so the
   dual-forward cross-cov pass *fits* (a 70 GB teacher + 50 GB student won't co-fit
   1×H200 at useful batch). This is a naive pipeline (one layer's compute active at
   a time), NOT a throughput speedup. The plugins are cross-device-safe via
   per-callback `.to(tensor.device)` coercions.

2. **DATA-PARALLEL covariance (real speedup)** — `multi_gpu.cov_replicas > 1` fans
   out G replica processes over disjoint calibration shards
   (`mp.spawn` → `_cov_replica_worker`), each spilling per-layer Gram to its own
   subdir; the parent key-wise sums them (`_reduce_spilled_cov_dirs`). Correct
   because the per-SEQUENCE reduction pin (`InputCovarianceAccumulator.update_grouped`)
   makes each replica's finalized per-key Gram batch-independent, so the cross-replica
   fp32 sum is exact (B+C bitwise, factored down_proj allclose ~1e-6). **Live-validated:**
   sharded B-cov bitwise-identical to 1-GPU; cross-cov runs cross-device.

Everything DOWNSTREAM of covariance collection is **single-GPU today**:
- `d_rank_allocate` — per-expert `svdvals` (CPU-fp64, single process).
- `swift_svd_alpha` — alpha grid of 11 end-to-end forward evals, SERIAL on one device.
- `aa_svd_factor` — per-(layer, expert) eigh+SVD, SERIAL on one device.
- `block_refine` — per-block AdamW training, SERIAL on one device.
- `wanda_intra_expert_score` — forward-based score sweep, SERIAL (cache-MISS path).

The biggest un-parallelized wins are **swift_svd_alpha** (alpha-grid → DATA/TASK-PARALLEL),
**block_refine** (training → DDP/TASK-PARALLEL per block), and **aa_svd_factor +
d_rank_allocate** (per-expert SVD → TASK-PARALLEL like Stage 4 EoRA).

---

## Per-Plugin Table

| Plugin | (a) Computes | (b) Compute profile | (c) Device usage / already-MG? | (d) Scheme | (e) Result-preserving | (f) Auto-batch per-GPU preserved | (g) Effort / Risk / Speedup |
|---|---|---|---|---|---|---|---|
| **covariance_collection** | Post-prune B-cov (S=XᵀX) + cross-cov C (gate-only) per (layer,expert,matrix) via dual-forward | cov-collection-forward (+ in-graph Gram) | **ALREADY MULTI-GPU**: MODEL-SHARD (device_map) AND DATA-PARALLEL (`cov_replicas`) | ALREADY-MULTI-GPU | per-sequence reduction pin → key-wise fp32 sum exact; disjoint shards | YES — each DP replica probes its OWN pinned-device VRAM (`size_batch` after `CUDA_VISIBLE_DEVICES`) | done / low / N× on cov pass |
| **d_rank_allocate** | T_budget + per-group rank allocation; whitened per-expert `svdvals` for eff-rank | SVD-factor-linear-algebra (CPU-fp64) | single-process, CPU-fp64 (device-independent) | TASK-PARALLEL (per-(layer,matrix) group) — **NOT-WORTH-IT today** | deterministic per-group; mean-of-spectra is order-free | N/A (CPU, no forward, no auto-batch) | med / low / small (seconds-to-minutes; not a wall-clock bottleneck) |
| **swift_svd_alpha** | Global α (11-cand WikiText-2 PPL grid) + per-type spectral-proxy α + per-expert rank redistribution | alpha-search-eval-forward (validation path) + SVD-proxy (spectral path) | **single-GPU, SERIAL** over 11 α candidates | DATA-PARALLEL or TASK-PARALLEL (alphas across GPUs) | each α independent: factor→eval→restore; PPL is per-α deterministic | YES — `_evaluate_wikitext2_ppl` has fixed `validation_batch_size`; a per-replica auto-batch would probe own VRAM | high / med / ~N× on the ~31-min validation grid |
| **aa_svd_factor** | Per-(layer,expert) AA-SVD rank-k factor (eigh(B)+SVD(W·rhs)+back-solve); installs FactoredExperts | SVD-factor-linear-algebra | **single-GPU, SERIAL** per-layer `loop_over`; eigh/SVD on `dev` | TASK-PARALLEL (per-layer or per-expert SVD, like Stage 4 EoRA) | factor is a pure fn of (W,A,B,C,k); independent per (layer,expert) | N/A (no forward batch; linear-algebra only) | high / med / ~N× on factor loop |
| **block_refine** (gated, default OFF) | Per-block AdamW MSE refine of U/V + RMSNorm vs teacher block target | block-refine-training | **single-GPU, SERIAL** per-block; MODEL-SHARD-safe (out.to(target.device)) | DDP (within a block) or TASK-PARALLEL (blocks pipelined) — sequential cross-block dependency limits this | **METRIC-PINNED** — batch grouping changes trained weights; DDP must replicate gradients exactly | NO auto-batch by design (METRIC_PINNED, no `auto_batch` wiring; see L172-176) | very high / high / partial (cross-block serial chain) |
| **wanda_intra_expert_score** (gated, default OFF) | Routing-weighted Wanda `|W|·√E[(x·g)²]` per (layer,expert,matrix) | wanda-score-forward (cache-MISS) / cache-IO (HIT) | **single-GPU, SERIAL** per-layer sweep (MISS); CPU accum | DATA-PARALLEL (MISS path, like cov) | running-mean accumulator is additive per key | partial — uses fixed `batches`; would need per-replica probe like cov | med / low / N× (only on MISS path; HIT is free) |
| **input_cov_cache** | Loads A_cov (Σ_in) sidecar | cache-IO (CPU torch.load) | CPU only | NOT-APPLICABLE | manifest-validated load | N/A | none |
| **wanda_scalar_row_cache** | Loads Wanda scalar_row sidecar | cache-IO (CPU torch.load) | CPU only | NOT-APPLICABLE | manifest-validated load | N/A | none |
| **block_hidden_cache** | Loads teacher block-output targets sidecar | cache-IO (CPU torch.load + reshape) | CPU only | NOT-APPLICABLE | shape-checked load | N/A | none |

---

## Detailed Findings

### 1. covariance_collection — ALREADY MULTI-GPU (both levers)

**MODEL-SHARD** (memory fit). The teacher+student dual-forward (Theorem 3.2 cross-cov)
needs both models resident. On 1×H200 (141 GB) a 70 GB teacher + 50 GB student leaves
~21 GB → bs≈4. `model.device_map: "balanced"` shards both across N GPUs
(`orchestrator.py:222-228` comment block; teacher loaded with
`device_map=config["model"]["device_map"]` at `orchestrator.py:390-396, 447-454, 467-474`;
`load_model` honors `"balanced"`/`"auto"` per `utils/model_io.py:80-140`,
`_resolve_4bit_device_map` at :54). Cross-device safety in the cov callbacks:
`tgt_device = tensor.device` then `.to(tgt_device)` before the `Xᵀ@X` matmul and the
`update_cross` in-place add — `covariance_collection.py:671, 682, 728, 745` (dense path),
:682 (dict path). This is the naive pipeline (NOT a throughput speedup); the comment is
explicit (`orchestrator.py:226-228`: "the achievable ceiling is activation-bound on the
hot layer's device and is MEASURED on the first sharded run, not assumed").

**DATA-PARALLEL** (real speedup). `_resolve_cov_replicas` (`orchestrator.py:79-104`) reads
`multi_gpu.cov_replicas`, returns `min(requested, n_gpu // shards_per_model)` floored to 1
(`shards_per_model` hard-coded 1 → whole-model replicas). When `_dp_replicas > 1` the
no-resume branch (`orchestrator.py:401-458`) calls `run_dp_covariance_collection`
(`covariance_collection.py:1213-1288`):
- `_shard_calib` (`:1052-1071`) splits the calib tensor into disjoint contiguous shards
  (token-disjoint → exact cross-replica sum).
- `mp.get_context("spawn")` launches one `_cov_replica_worker` (`:1074-1210`) per replica;
  each sets `CUDA_VISIBLE_DEVICES` (`:1097`), reloads teacher+student, runs
  `_collect_covariances` over its shard, spills per-layer Gram to `_replica_{r}/`.
- Parent joins, then `_reduce_spilled_cov_dirs` (`:211-289`) key-wise sums the replica
  subdirs in **fp32** (sorted order for determinism, `:231`) → canonical spill dirs.
- The factor phase then lazy-loads from the canonical spills exactly like a 1-pass run.

**Why correct (the reduction pin):** The Gram is a linear sum of per-token outer products,
so `B = Σ_r B_r` exactly. The per-SEQUENCE pin (`update_grouped`, routed at
`covariance_collection.py:657-659` for B input, :783-785 for down_proj, :701/746 for
cross-cov) makes each replica's FINALIZED per-key Gram independent of its forward batch
size. Result: **gate_proj/up B + cross-cov C are bitwise-invariant**; factored down_proj B
is allclose ~1e-6 (bounded, N-INDEPENDENT upstream forward-activation drift, NOT reduction
drift — `:394-406, 761-780`). Therefore NO cross-replica `min(candidate)` agreement is
needed (supersedes the old spec §6). **Live-validated** (per task brief): sharded B-cov
bitwise-identical to 1-GPU; cross-cov runs cross-device.

### 5. cov DP per-replica auto-batch — CONFIRMED CORRECT AND COMPLETE

The auto-batch story for the DP path is complete and correct:
- `_resolve_cov_batch_size` (`:365-447`) returns the inherited int FLOOR (=1 default) in
  every case — including `"auto"` — so the golden is byte-identical when auto is off.
- The actual VRAM sizing is done INSIDE `_collect_covariances` (`:912-992`), gated by
  `cov_auto=_cov_is_auto(s3)` (double-gate: `cov_batch_size=="auto"` AND
  `auto_batch.enabled`, `:450-464`). It calls `size_batch(cost_probe_fn, cov_floor,
  headroom_frac=..., max_cap=_COV_MAX_CAP(256), mem=CudaMemProbe(device))` with the G
  window **already resident** so the probe baseline absorbs the window commitment, then
  `run_with_oom_backoff(..., floor=cov_floor=1)`.
- **Per-replica independence:** the worker sets `CUDA_VISIBLE_DEVICES` FIRST (`:1097`),
  THEN passes `cov_auto=_cov_is_auto(...)` + `auto_batch_cfg` + `calib=shard` to
  `_collect_covariances` (`:1192-1209`). So `CudaMemProbe(device)` probes the replica's
  OWN pinned-device free VRAM; each replica sizes INDEPENDENTLY. The per-sequence pin makes
  the key-wise reduce batch-independent, so independent per-replica batches are SAFE — no
  agreement protocol. This is the HARD REQUIREMENT ("each replica probes its own VRAM") and
  it is met. Worker comment block `:1142-1157, 1192-1210` documents this precisely.
- `_COV_MAX_CAP=256` (`:177`) is a cov-specific backstop (NOT v1's 4096), because a 4096-seq
  dual-forward over G window layers each holding `[T,d_in]` fp32 teacher tensors would OOM
  thrash; `headroom_frac` is the real limiter, the cap just bounds a degenerate probe.

`auto_batch.py` itself was recently FIXED (main f8f7108 / size_candidate): `usable = total -
headroom`, NO `allocated_baseline` subtraction (the model bytes are already inside `fixed =
2·peak1 − peak2`); double-subtracting drove the result negative → silent bs=1. The cov path
consumes the fixed resolver, so the DP path benefits from the fix.

### 2. d_rank_allocate — TASK-PARALLEL candidate, NOT-WORTH-IT today

`_group_stat` (`d_rank_allocate.py:347-421`) per (layer,matrix) group computes an fp64
Cholesky of A_g (CPU) then per-expert `svdvals(L_A @ W.T)` (CPU-fp64, `:385-395`), then
mean-of-spectra → effective rank. `_compute_T_budget` + `_d_rank_allocate` are pure scalar
arithmetic. The whole thing is CPU-fp64 by deliberate device-independence policy
(D-drank-fp64-spectrum, `:193-219`) — rank decisions must agree across CPU/GPU to ~1e-14.
The per-(layer,matrix) groups (~120 entries) are embarrassingly parallel and the per-expert
`svdvals` loop is too — a `ProcessPoolExecutor` or per-GPU shard would work. **But:** this
runs CPU-fp64 and is seconds-to-low-minutes, not a wall-clock bottleneck; parallelizing adds
spawn/serialization complexity for marginal gain. No forward → no auto-batch concern.
Verdict: TASK-PARALLEL-able but NOT-WORTH-IT unless profiling flags it.

### 3. swift_svd_alpha — best un-parallelized win (alpha-grid DATA/TASK-PARALLEL)

Two paths, both SERIAL on one GPU today:
- **Validation path** (`_swift_svd_plus_alpha_search_validation`, `:654-806`): the
  paper-exact 11-candidate grid. For EACH α: `_redistribute_ranks` → `_factor_model_at_ranks`
  (factor whole model in-place at α's ranks) → `_evaluate_wikitext2_ppl` (end-to-end forward)
  → `_restore_fused_experts`. ~31 min on H200 (`:693-698`). The 11 candidates are
  **independent** (each does factor→eval→restore from the shared CPU `originals` snapshot).
- **Spectral-proxy path** (`_swift_svd_plus_alpha_search`, `:809-1022`): per-expert
  `svdvals(W@L_A)` (CPU-fp64) then scores all 11 α — no forward, fast.

**Scheme — KEY QUESTION 2:** The validation grid is the prime candidate. Two options:
- **DATA-PARALLEL (each GPU evaluates the SAME α on a calib shard):** would split the
  `validation_samples` PPL eval across GPUs, summing NLL — but PPL eval is already cheap
  (~0.3 min) vs the factor (~2 min), so this barely helps.
- **TASK-PARALLEL (each GPU owns a SUBSET of the 11 α candidates):** the big win. Replica r
  factors+evals α candidates `r::N`, returns `(α, ppl)`; parent takes argmin. Each replica
  needs the CPU `originals` snapshot + B/C spill dirs (already on disk) + its own model copy.
  Near-N× on the 31-min grid for N≤11.
  - Result-preserving: PPL is a deterministic fn of (factored model, val_tensor); the A2
    eigh-cache (`_stage3_alpha_eigh_cache`, `:735-750`) is (B,A,C)-determined and identical
    across candidates, so each replica can rebuild it locally (or share read-only). argmin of
    `(α,ppl)` is order-free.
  - **Auto-batch per-GPU preserved:** `_evaluate_wikitext2_ppl` takes `validation_batch_size`
    (fixed 16, `:347/710`). A per-replica VRAM-aware sizing would probe its own device exactly
    like the cov path (`CudaMemProbe(device)` after pinning) — but eval is BATCH_INVARIANT
    (NLL sum over tokens is grouping-independent), so auto-batch is cleanly applicable here.
- Effort high (needs a spawn driver mirroring `run_dp_covariance_collection`, a per-replica
  model reload, and result merge); risk med (the in-place factor/restore must be replica-local
  so no cross-replica model mutation); speedup ~N×.

### 4. aa_svd_factor — TASK-PARALLEL (per-layer/per-expert SVD), like Stage 4 EoRA

**KEY QUESTION 1:** Yes, the per-expert SVD factorization IS task-parallelizable across GPUs,
and it is currently single-GPU. The factor loop is driven by `loop_over(moe_layers, plugins,
("factor_layer",), ...)` (`orchestrator.py:806`) — one layer per child ctx, SERIAL. Inside
`factor_layer` (`aa_svd_factor.py:480-718`): per expert, `_precompute_eigh(B,A,C)` then per
matrix `_aa_svd_precomputed`/`_aa_svd` (eigh + SVD(W@rhs) + back-solve) on `dev`
(`:629-650`). Each (layer,expert,matrix) factor is a **pure function** of its inputs and
independent — identical structure to Stage 4 EoRA's per-expert eigh/SVD that already went
N-GPU TASK-parallel (per MEMORY `multigpu_stage3_landed` / `stage4` row-gather). 

Scheme: TASK-PARALLEL — shard (layer,expert) work across GPUs; each GPU factors its subset
from the CPU `originals` snapshot + its own B/C spill loads, gathers U/V back. The shared
mutable `rank_map` dict (HAZARD H1, `:489-491`) and the in-model `setattr(ref.mlp,'experts',
new_factored)` (`:688`) are the only cross-cutting state — a TASK-parallel version must merge
rank_map (order-free dict of independent keys) and install factored modules in the parent.
Result-preserving: factor is deterministic; FactoredExperts install is per-layer independent.
No forward batch → **no auto-batch concern** (linear-algebra only; the per-GPU concern is just
VRAM fit of one expert's eigh/SVD, which is tiny). Effort high, risk med, speedup ~N× on the
factor loop. The eigh-reuse (gate/up share `_precompute_eigh`, `:618-626`) and the B-cov
prefetcher (Tier-1 item 9, `:545-554`) must be preserved per-replica.

### 6. block_refine — DDP candidate, but cross-block-serial + METRIC-PINNED

**KEY QUESTION 3:** Yes, it TRAINS — per-block AdamW (`_phase_c5_block_refine`,
`block_refine.py:165-656`). For each decoder block sequentially: compute teacher target,
train U/V + RMSNorm via AdamW MSE for `epochs`, advance both streams. This makes it a DDP
candidate WITHIN a block (replicate the block, shard the calib minibatches, all-reduce grads).
BUT two hard limits:
- **Cross-block serial dependency:** block i+1's input `X'_{i+1}` is produced by the REFINED
  block i (`_advance_streams`, `:647-651`). You cannot pipeline blocks naively — block i must
  finish before i+1's targets are correct. So TASK-PARALLEL across blocks is unsound; only
  intra-block DDP is.
- **METRIC-PINNED (auto-batch forbidden):** the docstring is explicit (`:172-176`):
  block_refine is minibatch-SGD; changing batch_size changes the minibatch grouping and the
  trained weights, so it gets NO `auto_batch` wiring. A DDP version must keep the EFFECTIVE
  global batch + step order bit-identical (per-rank micro-batch × N ranks = the original
  global batch, gradient all-reduce in a fixed order) to preserve the trained weights — this
  is the result-preserving mechanism and it is non-trivial.
- It IS already MODEL-SHARD-safe (`out.to(target.device)` at `:563-564`; pin gated on CUDA at
  `:224-228, 670-674`). Effort very high, risk high, speedup partial (serial chain dominates).

### 7. wanda_intra_expert_score — DATA-PARALLEL (MISS path only)

Gated OFF by default. On cache HIT (`collect_wanda_scores`, `:657-686`) it hydrates from the
sidecar — pure CPU, zero forward, no MG needed. On cache MISS (`:687-727`) it runs a per-layer
`instrument_experts` calibration sweep (`model(input_ids=batch)` forward per layer) into a
running-mean `_WandaScalarRowAccumulator` — same structure as `_collect_covariances`.
That accumulator is additive per key, so it is DATA-PARALLEL exactly like the cov DP path
(disjoint shards → sum). Auto-batch: uses the fixed `batches` slot; a DP version would probe
per-replica VRAM like cov. Effort med, risk low, speedup N× on the MISS path only.

### 8-10. Cache providers — NOT-APPLICABLE

`input_cov_cache`, `wanda_scalar_row_cache`, `block_hidden_cache` are all CPU-side
manifest-validated `torch.load` of sidecars (`load_covariance` / `load_wanda_scalar_row` /
block-output reshape). No compute, no device, no batch. NOT-APPLICABLE for multi-GPU.

---

## Auto-batch invariant (applies to every forward-bearing plugin)

The HARD REQUIREMENT — multi-GPU must preserve per-GPU VRAM-aware batch sizing, each replica
probes its OWN VRAM — is currently met ONLY in `covariance_collection` (the DP worker probes
`CudaMemProbe(device)` after `CUDA_VISIBLE_DEVICES`, `:1097` + `:1184-1209`). Any future
parallelization of `swift_svd_alpha` (BATCH_INVARIANT eval — clean) or `wanda` (MISS sweep)
MUST replicate that pattern: pin device first, construct `CudaMemProbe(device)`/`size_batch`
inside the worker, OOM-backoff to floor=1. `block_refine` is METRIC_PINNED and must NOT get
auto-batch (`auto_batch.py:_V1_ELIGIBLE = {BATCH_INVARIANT}` excludes it). `aa_svd_factor` /
`d_rank_allocate` have no forward batch, so auto-batch is N/A there.
