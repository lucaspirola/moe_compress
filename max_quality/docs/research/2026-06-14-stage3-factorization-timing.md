# Stage-3 factorization timing after the cov-efficiency merge (#1/#3/#4)

**Date:** 2026-06-14
**Branch:** `research/stage3-factorization-timing`
**Scope:** read-only analysis. Does the merged Stage-3 cov-efficiency work
(`merge d967b16`: #1 gentler OOM backoff, #3 single-pass CPU-accumulator,
#4 `cov_num_sequences`) speed up the ~90 min **factorization** phase that
paced the previous s234 run?

## VERDICT

**No. The ~90 min group-stat (rank-deciding SPECTRA) phase is UNCHANGED by
#1/#3/#4.** None of the three merged changes touch the factorization code
path. They attacked **covariance COLLECTION** (the ~3.5 hr part). The
fp64-CPU spectra (~137 s/layer × ~40 ≈ ~90 min, both GPUs idle) is the
**unimplemented** improvement #5 and remains untouched.

There is **no indirect speedup** either: the α-search forward passes use
the int default `validation_batch_size=16` (not auto), so #1's gentler
backoff never engages there; and `validation_samples` (512 WikiText seqs)
is a **different knob** from `cov_num_sequences`, so #4 does not shrink the
α-search.

Proof from the merge scope — `git diff --stat d967b16^ d967b16` touches
only:

| file | role |
|---|---|
| `stage3/orchestrator.py` | wiring (`_resolve_bcov_spec`, `_maybe_cpu_hot_accum`) |
| `stage3/plugins/covariance_collection.py` | **cov collection** |
| `utils/activation_hooks.py` | the cov accumulator |
| `utils/auto_batch.py` | backoff step (x0.75) |
| `tests/*` | tests |

The three factorization plugins — `aa_svd_factor.py`,
`swift_svd_alpha.py`, `d_rank_allocate.py` — are **NOT in the diff**.
`grep run_with_oom_backoff|single_pass|hot_accumulator|cov_num_sequences`
over those three files returns **nothing**.

## Stage-3 post-cov-collection timeline (orchestrator.py order)

After cov collection finishes, the orchestrator runs, in order:

1. **Group-stat / rank-deciding SPECTRA** — `orchestrator.py:656-677`
   loops every MoE layer × {gate,up,down} and calls
   `_group_stat(...)` (`d_rank_allocate.py:347-421`). This is **THE ~90 min
   pole, GPUs idle**. Per group it:
   - Cholesky-factorises the averaged Stage-2 A-cov in **fp64 on CPU**
     (`d_rank_allocate.py:370-374`), then
   - runs a **per-expert** `torch.linalg.svdvals(L_A @ W.T)` in **fp64 on
     CPU** (`d_rank_allocate.py:381-395`) — `n_experts` SVDs of a
     `[d_in, d_out]` matrix per group, ~200 experts × 3 matrices × ~40
     layers ≈ 24k fp64-CPU svdvals. fp64 + CPU-resident is INTENTIONAL
     (deviation `D-drank-fp64-spectrum`, lines 193-219: device-independent
     0-rank-flip decision). This is the GPUs-idle / CPU-bound pole the
     "group-stat layer k/40" log lines (`orchestrator.py:659`) were pacing.
2. **`allocate_ranks`** — `_compute_T_budget` + `_d_rank_allocate`
   (`d_rank_allocate.py:453-567`). Microseconds; pure arithmetic on the
   group stats. Negligible.
3. **α-search** — `swift_svd_alpha.py:753-909`
   (`_swift_svd_plus_alpha_search_validation`). 11-α grid; each candidate
   factors the whole model (`_factor_model_at_ranks`) + evaluates WikiText-2
   PPL on `validation_samples=512` seqs + restores. Docstring estimate
   ~31 min on H200 (line 792-793). **GPU-bound** (forward passes + SVD), so
   this is NOT the GPUs-idle pole. (Eigh-decomp is cached across the 11
   candidates — `_factor_model_at_ranks` A2 cache, lines 495-523/605-618.)
4. **Actual per-expert SVD FACTORING** — `aa_svd_factor.py:586-841`
   (`factor_layer` → `_factor_expert_tile` → `_aa_svd_precomputed`). One
   pass over all layers. GPU-resident fp32 SVD. Fast relative to the
   group-stat pole.

**Dominant pole:** phase (1), the fp64-CPU group-stat spectra
(~137 s/layer ≈ ~90 min, both GPUs at 0%). Phase (3) is the second pole
(~31 min, GPU-bound). Phases (2) and (4) are minor.

> Naming note: the user's "group-stat log lines" are literally
> `orchestrator.py:659` `"  group-stat layer %d/%d"`, i.e. phase (1) — the
> D-Rank rank-deciding spectra, NOT the α-search and NOT the final factor
> loop. So the ~90 min the user clocked is precisely the fp64-CPU spectra.

## Per-change impact on factorization

### #3 — single-pass CPU-accumulator: **does NOT touch factorization**
`_maybe_cpu_hot_accum` (`orchestrator.py:145-158`) calls
`acc.set_hot_accumulator_device("cpu")` only on the **B/C accumulators
during cov collection** (call sites `orchestrator.py:291,457,517,558`, all
in the collection block). It changes *where the running Gram sum lives
while being accumulated*. Factorization consumes the **finalized**
`B_acc.covariance` / `C_acc.covariance` dict (`aa_svd_factor.py:741-742`),
which is the same d×d Gram regardless of how it was accumulated. **No
effect on factorization input or time.** (Expected — confirmed.)

### #4 — `cov_num_sequences`: **does NOT change the factorization input**
`_resolve_bcov_spec` (`orchestrator.py:161-175`) reads
`stage3_svd.cov_num_sequences` and overrides `num_sequences` **for the cov
pass only**. It reduces how many calibration sequences flow through the cov
forward (the ~3.5 hr collection). The covariance is `Xᵀ·X`, a
**`d_in × d_in` Gram** (e.g. 2048×2048 for gate/up) — its dimension is the
**hidden width, not the sequence count**. 512 vs 4000 sequences both
produce the identical-shape Gram; only its *statistical content* changes
(fewer samples → noisier estimate), never its size or the factorization
cost. So `_group_stat`'s Cholesky+svdvals, `_aa_svd`'s eigh+SVD, and the
α-grid all see the **same-shape** inputs and run in the **same time**.
**No factorization speedup.**

**`cov_num_sequences` ≠ `validation_samples` — DIFFERENT knobs:**
- `cov_num_sequences` (`orchestrator.py:170`, key `stage3_svd.cov_num_sequences`)
  → cov-COLLECTION calib size. Default **None** → `cal["num_sequences"]`
  unchanged (byte-identical). **Not present in any config yet.**
- `validation_samples` (`orchestrator.py:700` / `swift_svd_alpha.py:808`,
  key `stage3_svd.swift_svd_plus.validation_samples`, **=512** in all four
  `configs/qwen36_*.yaml`) → the α-search WikiText-2 PPL grid size.
The orchestrator comment at lines 265-267 explicitly warns the two must
not be conflated. So #4 leaves the α-search's 512 unchanged — no indirect
α-search speedup.

### #1 — gentler OOM backoff (x0.75): **does NOT affect factorization**
`run_with_oom_backoff` (`auto_batch.py:188-203`, now x0.75 per-attempt)
is invoked at **exactly one site**: `covariance_collection.py:1051` — the
**cov-collection forward**. The factorization plugins never call it
(grep-confirmed empty). It only changes how the cov forward recovers from
an OOM during collection.

Does the α-search use auto-batch backoff? **No, not by default.** The
α-eval forward (`_resolve_alpha_eval_batch`, `swift_svd_alpha.py:374-414`)
probes only when `validation_batch_size == "auto"` AND `auto_batch.enabled`;
the config default is the **int 16** (`swift_svd_alpha.py:809`,
`configs/*` set `validation_samples: 512` with the int batch), so the probe
is skipped and `run_with_oom_backoff` is not on the α-search path at all.
Even if it were, that path is the ~31 min GPU-bound α-search, not the
~90 min GPUs-idle pole.

## What WOULD speed up the ~90 min — improvement #5 (NOT implemented)

The pole is `_group_stat`'s **fp64 svdvals on CPU**. Options:

1. **GPU the rank-deciding spectra.** The current code is deliberately
   fp64-CPU for **device-independence** (`D-drank-fp64-spectrum`,
   `d_rank_allocate.py:193-219`): fp32-GPU flips 2-3/216 ranks vs the CPU
   golden (fragile per-device re-bless), while fp64 agrees CPU↔GPU to
   ~1e-14 (0 flips). So a naive fp32-GPU move is a **quality/repro trade**,
   not a free win — it would need a re-bless of the golden.
   - The MEMORY note `reference_fp64_gpu_svdvals_blackwell` says fp64
     svdvals is **~14× slower than fp32-GPU on consumer Blackwell (RTX
     5080)** but only **~½ speed (≈2×) on H200** — i.e. H200 fp64-GPU is
     viable where RTX 5080 fp64-GPU is not. **Caveat:** the ~137 s/layer /
     ~90 min was measured on the **2×H200** box and the code runs the
     spectra **on CPU**, so the ~90 min is the **CPU fp64** number, not a
     consumer-GPU artifact. Moving the fp64 svdvals to **H200 GPU** (fp64,
     ~2× the H200 fp32 cost but still device-independent at fp64) is the
     plausible #5 win — keeps the 0-flip fp64 guarantee while leaving the
     idle GPUs. This is a code change, not yet done.
2. **Overlap the CPU group-stat with GPU SVD factoring.** The group-stat
   loop and the later factor loop are sequential phases today; the GPUs sit
   idle during (1) and the CPU sits idle during (4). Pipelining them (or at
   least running the per-expert fp64 svdvals across CPU worker threads /
   processes — they are embarrassingly parallel per expert, see
   `D-drank-mean-spectra` lines 165-176) would cut wall time without
   touching the numerics. Also not done.

Both are #5-class work outside the merged cov-efficiency scope.

## Bottom line

The cov-efficiency merge made **cov COLLECTION** cheaper (fewer sequences
via #4, OOM-resilience via #1, VRAM headroom via #3). The **~90 min
fp64-CPU group-stat factorization-prep pole is exactly as slow as before**.
On the H200 resume box that ~90 min is the H200/CPU number — to attack it
you need improvement #5 (fp64-on-H200-GPU spectra, or CPU-parallel /
GPU-overlap), which was not part of this merge.
