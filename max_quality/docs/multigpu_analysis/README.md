# Per-Stage Multi-GPU Convertibility Analysis

Cross-stage roll-up of a per-plugin audit (one agent per stage, every plugin's full
code read). For each plugin: compute profile, current device usage, multi-GPU scheme,
result-preserving mechanism, **how the per-GPU auto-batch sizing is preserved**, and
effort/risk/speedup. Per-stage detail in `stage{1,2,3,4,6}.md` + `router_kd.md` (= Stages 2.5 + 5).

## Hard requirement (held throughout): auto-batch preserved per-GPU
The VRAM-aware max-batch resolver (`utils/auto_batch.py`: `size_batch` + `CudaMemProbe(device)`
+ `run_with_oom_backoff`, with the `size_candidate` baseline-double-count fix, main `f8f7108`)
is **already device-scoped** → under data-parallel, each replica sets `CUDA_VISIBLE_DEVICES`
then calls the resolver locally, so **each GPU probes its own VRAM and sizes its own max batch
independently**. The per-sequence reduction pin makes the cross-replica reduce batch/replica-
independent. **No change to `auto_batch.py` is needed.** For METRIC-PINNED stages (Router-KD,
generation) the batch is fixed by the science (`global_batch / n_gpu`), so auto-batch is
correctly only an OOM floor there, not a maximizer.

## Already multi-GPU
- **Stage 3 cov collection** — DP-replicas (`cov_replicas` + `_reduce_spilled_cov_dirs`) +
  model-shard (`device_map=balanced`). Live-validated (sharded B-cov bitwise-identical to 1-GPU).
- **Stage 4 EoRA** — per-expert device *placement* exists, **but runs SERIALLY** (no
  ThreadPool / CUDA streams): experts on different GPUs execute one-at-a-time. Declared
  multi-GPU, delivers ~0× speedup as shipped. → see rank 2.

## Ranked recommendations

| # | Target | Scheme | Speedup | Result-preserving | Effort | Notes |
|---|--------|--------|---------|-------------------|--------|-------|
| 1 | **Router-KD (2.5 + 5)** | **DDP** | ~linear | ✅ grad-avg ≡ single-GPU full-batch | M | The runtime long pole. Teacher 70GB won't co-fit a student replica → teacher-logit cache (1-epoch) or 4-bit teacher (multi-epoch, quality trade). Step-count global, rank-0 I/O, synchronized early-stop. |
| 2 | **Stage 4 EoRA concurrency** | ThreadPool / CUDA streams over existing placement | ~N× | ✅ per-expert independent | S | Skeleton already shards per-expert; just not overlapped. Cheapest real win. |
| 3 | **Stage 3 `swift_svd_alpha`** | task-parallel the 11-α grid | ~N× | ✅ | M | ~31-min α-search runs serial today; eval is BATCH_INVARIANT so per-replica auto-batch applies cleanly. |
| 3 | **Stage 3 `aa_svd_factor`** | task-parallel per-expert SVD (like EoRA) | ~N× | ✅ | M | merge the shared `rank_map` (HAZARD H1) in parent. |
| 4 | **Stage 6 eval** (HumanEval + MATH-500) | DP eval-shard | ~linear | ✅ **iff** shard boundary = multiple of 8 | M | gen is METRIC-PINNED (`PINNED_GEN_BATCH_SIZE=8`); WikiText-PPL is BATCH_INVARIANT (clean per-GPU auto-batch). |
| 5 | **Stage 2 profile forward** | DP (shard calib, reduce REAP/REAM/cov) | ~linear on scoring | ✅ shard-by-sequence | L | **per-layer** (sequential in-place merge — NOT parallelizable across layers); re-sync merged weights to replicas each layer. In production REAP rides the vLLM sidecar, so the HF forward's value is cov + REAM. |
| — | **Stage 1** | DP (Phase B = 1 shared pass for 5 plugins; Phase D = task-parallel candidates) | ~N× | ⚠️ CKA reservoir randomized → tolerance, not byte | L | **Zero value for by-the-book ablations** (Stage 1 stubbed). |

## Not multi-GPU-able (correctly)
- All 6 Stage-2 merge solvers (greedy / hungarian / mcf / sinkhorn / auto / two-opt) — CPU
  (numpy / scipy / ortools). Stage-1 damage-curve / RCO / floor — CPU.
- The Stage-2 per-layer merge loop is **strictly sequential** (in-place merge: layer L+1
  profiles through merged layer L — REAM §4). Only *intra*-layer parallelism is legal.
- All `*_cache` providers — pure cache-IO (CPU). On a sidecar hit the live accumulator and
  its calibration forward are dropped entirely (an alternative to multi-GPU for that signal).

## Correctness notes surfaced by the audit
- Stage-2 agent **corrected an over-claim** that the per-layer loop is task-parallel — it is
  not (the in-place merge is load-bearing; paper ablation −1.0 AVG when removed).
- Stage-4 agent found the "N-GPU EoRA" (main `e395ad0`) is a correct placement **skeleton
  missing its concurrency engine** — the headline finding for cheap wins.
