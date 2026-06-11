# Per-Plugin VRAM-Aware Auto-Batch-Sizing — Design

Date: 2026-06-11
Status: Design (pre-implementation)
Branch: `feat/auto-batch-sizing`

## 1. Problem & Motivation

Every forward/calibration plugin in the pipeline carries a **hardcoded `batch_size` constant** (e.g. `stage1 ablation_filter.batch_size=8`, `stage3 cov_batch_size=1`, `block_refine.batch_size=32`, `phase_a/phase_b_batch_size`, `stage6 ppl_batch_size=8`). These constants are set **conservatively for the worst case**: `ablation_filter` was dropped 32→8 after an OOM, and `cov_batch_size` is pinned to **1** *"so the 1-GPU golden is byte-identical."*

Two problems result:

1. **Wasteful on big GPUs.** A constant tuned to survive one phase on one GPU leaves an H200 (141 GB) massively under-utilized. The Stage-3 cross-covariance path runs at `bs=4`, ~27 min/layer (~16 h/arm) — the wall that motivated the multi-GPU Stage-3 work.
2. **The OOM was phase-dependent, not constant.** `ablation_filter` Phase D OOM'd at `bs=32` because its baseline NLL forward (~9.5 GB fp32 activations) ran with the **35B model + earlier-pipeline accumulators (covariance/reservoir/MA buffers) co-resident**. The *free* VRAM at a plugin's moment of execution is not a function of model size alone — it depends on what else is already on the device at that phase.

Globally lowering a shared constant penalizes every plugin for the tightest one. The correct direction is the opposite: **each plugin should maximize its own forward batch against the VRAM actually free at the moment it runs**, while never changing a numeric result.

## 2. Goals / Non-Goals

**Goals**
- Each opted-in plugin auto-sizes its **forward** batch to the free VRAM measured *at its phase*, on the actual host.
- Works unchanged across single-GPU (RTX5080 16 GB), H200 (141 GB), and the multi-GPU sharded path.
- **Never silently changes a result.** Numeric safety is decided by fidelity class (only provably-safe classes are wired); memory fit is validated by running the *real* pass with OOM-resilient halving down to the fixed batch. No subset self-test (§4c).
- Quality-neutral: for v1's `batch-invariant` plugins drift is structurally **zero** (grouping-independent reduction); for v2's pinned reductions it is the residual hardware-GEMM term only.

**Non-Goals**
- We do **not** attempt bitwise reproducibility across hosts/batches. On GPU the forward GEMM itself is batch-shape-dependent (cuBLAS tiling), so the activations are not bitwise-invariant to batch size — see §3. This is a hardware fact, not a design choice.
- We do **not** touch **metric-pinned** plugins (e.g. `gen_batch_size` — batched generate ≠ bs=1 changes the *metric*, the landed bs-invariance lesson). Those opt out entirely.
- No global "one auto batch for everything." Sizing is per-plugin.

## 3. Fidelity Model

There are **two** sources of batch-dependence, and they are categorically different:

| Source | Owned by | Invariant achievable? | Our handling |
|---|---|---|---|
| **Forward** (cuBLAS GEMM tiling) | hardware | **No** — kernel selection depends on batch shape; bs=64 activations ≠ bs=1 activations bitwise (~1e-6 fp32) | **Tolerate** — for `batch-invariant` the downstream reduction (max/percentile) is robust to it; for `reduction-accumulating` it's the residual after v2 pinning |
| **Reduction** (Gram/NLL accumulation over tokens) | our code | **Yes** — we choose the grouping | **Validate** (v1) → optionally **pin** (v2) |

**Per-plugin classification (explicit declaration in code):**

- **`metric-pinned`** — batch size changes the reported number, not just fp order (generation geometry). Auto-batch **disabled**; uses the fixed constant. (`stage6 gen`, `lm_eval` already has its own `auto:N`.)
- **`batch-invariant`** — output has no fp reduction over the batch dim, so batch size cannot change the result at all. Auto-size freely. (Few/none expected among the fp-reduction plugins; included for completeness.)
- **`reduction-bounded`** — output is a deterministic function of activations via an fp reduction whose drift does **not** grow with token count (bounded independent of N). Eligible in principle, but **the resolver is wired to one only when a concrete bounded-drift argument exists for that plugin** — there is no v1 caller, and no probe auto-promotes one (see §4c). Until then it stays at the fixed batch.
- **`reduction-accumulating`** — output is an fp reduction that **accumulates over tokens**, so its reassociation drift **grows with N** (covariance Gram `Σ_tokens x xᵀ`, NLL mean `Σ_tokens nll / N` — `activation_hooks.py:308-311` documents ~5e-4 rel at large token counts). The slow, valuable class. **No subset probe can bound full-run drift here** (see §5). These plugins MUST have their reduction **pinned** (§5 v2) before a big batch is adopted — pinning makes the reduction grouping batch-independent → 0 reduction drift. Until pinned, they stay at the fixed batch.

The declaration **gates whether auto-batch is wired at all** — it is a load-bearing correctness declaration, not just an opt-in flag, and is the first thing the plan verifies per plugin (cov and NLL are accumulating → not v1; ma_detection phase-a is batch-invariant → v1). There is **no runtime numeric self-test that could rescue a mis-declaration** (§4c removed the subset allclose); a plugin wrongly declared `batch-invariant` would silently drift. So the classification must be *proven* per plugin (for `batch-invariant`: the reduction is a grouping-independent selection like max/percentile, not an fp sum), not assumed. Memory fit is the only thing validated at runtime — by the §4b OOM-resilient real run.

## 4. Architecture — the Auto-Batch Resolver

The resolver decomposes into **two orthogonal responsibilities** — *predict* a batch (cheap, from a cost model) and *run safely* at it (validate fit on the real workload). It does NOT run a separate subset "self-test" forward — that mechanism was removed (see the design note below). The plugin supplies its forward at the call site; the resolver owns sizing + the OOM-resilient run wrapper.

### 4a. Sizing — `size_batch(...)` (cost-model prediction, cheap)

1. **Cost probe — identical forward (CONTRACT).** A tiny probe (just bs=1 and bs=2, over ≥2 sequences) that runs the **byte-for-byte same forward signature** the plugin uses in its real loop — same `labels=`/`use_cache=`/dtype/loss path. Load-bearing: `ablation_filter`'s peak is dominated by the HF `ForCausalLMLoss` bf16→fp32 **logits upcast** (`batch·seq·vocab·4` ≈ 38 GB at bs=32; `ablation_filter.py:96-102`); a probe that omits `labels=` under-measures and sizes into an OOM. Real dtype, not synthetic. `reset_peak_memory_stats()` before each; cost = `max_memory_allocated()` peak.
2. **Two-point fit.** `cost(b) = fixed + b·per_sample`. `fixed` (cuBLAS/SDPA workspaces) is sub-linear; `per_sample` is linear. A one-point probe can't separate them → mis-sizes when `fixed` is appreciable (small/short-seq). bs=1 and bs=2 peaks fit both: `per_sample = peak2−peak1`, `fixed = 2·peak1−peak2`.
3. **Precise free signal.** Raw `mem_get_info()` counts PyTorch's **reserved-but-unallocated cache** as used (under-reports reusable memory); a read right after `empty_cache()` over-reports. Define `usable = total − memory_allocated(baseline) − headroom`, `baseline` captured after the model + resident accumulators load and before the probe, `headroom = headroom_frac · total` (fragmentation only). `candidate = floor((usable − fixed) / per_sample)`, clamped `[fixed_batch, max_cap]` (`fixed` subtracted once, not in headroom). **Supersedes** `_resolve_cov_window`'s raw-`mem_get_info`×0.75 where it overlaps (§6).
4. `size_batch` returns the **predicted** candidate (an int ≥ `fixed_batch`). A non-increasing / noisy probe (`peak2 ≤ peak1`, e.g. `probe_samples<2`) MUST be caught and degrade to `fixed_batch` — sizing **never raises** into the caller.

### 4b. Fit-validation — `run_with_oom_backoff(run_fn, start_batch, floor)` (the real run IS the test)

There is no separate confirmatory probe. The plugin's **actual** calibration pass runs at `start_batch` (the prediction); the wrapper executes `run_fn(batch)` and, on `torch.cuda.OutOfMemoryError`, calls `empty_cache()`, **halves** the batch, and **reruns the pass** (idempotent — fresh accumulators each attempt), down to `floor` (the always-safe fixed batch). The production loop at full width over real data is the genuine, no-waste fit test; the two-point prediction is conservative — and its margin is **tunable via `headroom_frac`** (§7) — so a retry is rare. This is the established `lm_eval auto:N` pattern. Every adopt/retry/fallback is `log()`-ged — never silent.

**CUDA-context recoverability caveat.** A CUDA OOM does not always leave a cleanly recoverable context — `empty_cache()` may not fully release, or the next kernel launch may fault (this is precisely why `lm_eval auto:N` pre-computes a descending schedule rather than catching in place). The wrapper therefore treats a *re-raised* OOM after `empty_cache()` (or any non-OOM CUDA error during a retry) as a **hard fail and falls back to the `floor`/fixed-batch path** rather than spinning the retry loop — process-level safety, not just batch-level. The idempotent-rerun requirement also constrains *which* plugins may use the wrapper: only passes whose accumulators are freshly built per attempt (ma_detection phase-a qualifies) — a plugin that mutates shared state mid-pass must not be wrapped.

### 4c. Drift-safety comes from classification, NOT a probe

A subset `allclose` self-test (the original §4 step) was **removed**. Rationale: (a) to run a *real* W-wide forward needs ≥W rows, and to compare to bs=1 needs bs=1 over the *same* rows — ~W forwards, the expensive shadow run we're avoiding; and (b) per §5 it is **blind to N-scaling reduction drift** anyway, so it was never a proof. Numeric safety is therefore decided **by fidelity class**: `batch-invariant` (e.g. ma_detection's max/percentile) is grouping-independent → no check needed; `reduction-accumulating` (cov/NLL) gets drift-safety from **v2 reduction-pinning**, not a probe; `reduction-bounded` (when a real caller appears) is handled per its own bounded-drift argument at that time. The resolver does not pretend a probe validates drift.

Interface: `size_batch(cost_probe_fn, fixed_batch, *, headroom_frac, max_cap, mem) -> int` and `run_with_oom_backoff(run_fn, start_batch, floor) -> result`. A plugin sizes once, then runs its pass through the backoff wrapper. `auto_batch.enabled=False` ⇒ neither is invoked (the plugin keeps its fixed batch).

## 5. v1 vs v2 (Incremental)

**Why drift-safety is a classification decision, not a probe (the N-scaling problem).** Forward-kernel drift is per-element and does **not** grow with token count. Reduction-reassociation drift **grows with N** (`activation_hooks.py:308-311`: ~5e-4 rel at large token counts). A subset probe over a few samples therefore can never *prove* a full-N accumulating reduction stays in tolerance, and a full-N shadow re-run costs the bs=1 run we're avoiding. So drift-safety is established structurally:

- **v1 — sizing + OOM-resilient run, for `batch-invariant` plugins.** `batch-invariant` (e.g. ma_detection's max/percentile) has *no* reduction drift at all — grouping-independent by construction — so auto-batch is safe with **zero** numeric check; only memory fit matters, which §4b's OOM-resilient real run validates. v1 ships `size_batch`, `run_with_oom_backoff`, the classification scaffolding, and wires the one batch-invariant plugin. `auto_batch` default-off. (`reduction-bounded` is *declared* but has no v1 caller; the resolver simply won't be wired to one until its bounded-drift argument is made concrete — it is NOT auto-enabled by a probe.)
- **v2 — pin the reduction grouping for `reduction-accumulating` plugins (cov / NLL).** Refactor the reduction to a **fixed grouping** (per-sequence / fixed token-chunk, independent of forward batch). The Gram matmul / NLL sum is cheap vs the forward, so this is nearly free; it drives reduction drift to **0**, leaving only forward drift (small, N-independent, and for these plugins quality-neutral). **This is the precondition that makes auto-batch safe there**, not a later optimization. The cov wall (§10) is a **v2** target from the start.

So: v1 = sizing/run infra + batch-invariant plugins (safe with no numeric gate); v2 = the pinned-reduction adaptation that unlocks the accumulating plugins (cov, NLL — the slow ones). We never pretend a subset probe makes an accumulating reduction safe; that safety comes only from pinning.

## 6. Multi-GPU / Sharded Path

`mem_get_info()`/`memory_allocated()` are per-device. Two distinct paths:

- **Data-parallel cov (replicas).** Each replica owns a disjoint shard (`_shard_calib`) and the final artifact is the cross-replica **sum** (`_reduce_spilled_cov_dirs`). If replicas adopt **different** batches (different free VRAM per device), their per-shard reductions use **different groupings**, and the drift composes in the sum — so per-replica-independent sizing is correct for *not OOMing* but does **not** by itself keep the *reduced* Gram within the blessed tolerance. Therefore on the DP path the resolver must **agree on one batch across replicas** (all-reduce `min(candidate)` before adopting), keeping the grouping uniform. Because the DP cov reduction is `reduction-accumulating`, it is a **v2 (pinned-reduction)** target — so the "reduced reference" here is the *pinned-grouping* reference (per-sequence fixed order, batch-independent), NOT a full bs=1 shadow run; uniform `min(candidate)` + pinned grouping is what keeps the summed Gram within tolerance without paying the bs=1 cost. This **supersedes** the currently-deferred DP branch of `_resolve_cov_batch_size` (`covariance_collection.py:384-396`); the spec explicitly takes over that deferral.
- **Model-sharded (whole-layer, accelerate).** Forward batch is bounded by the **tightest** shard device; size against `min` usable-free across the shard's devices.
- **Interaction with the existing window auto-sizer.** `_resolve_cov_window` (`covariance_collection.py:297-348`) already `mem_get_info`-auto-sizes the *layer window* `G` (0.75 headroom); the compound peak is `G · cov_bs · seq · d_in · 4` (`:370-376`). An auto-sized batch stacked on an auto-sized `G` double-counts free VRAM → OOM. The resolver MUST **co-resolve**: fix `G` first, then size the batch against the free VRAM *remaining after* `G` is committed (or fold both into one peak budget). The plan must reconcile the two sizers, not run them blind to each other.

## 7. Config & Knobs

- Per-plugin: `<plugin>.batch_size` stays as the **fixed-batch floor / fallback** (back-compat; existing configs keep working unchanged).
- New optional per-plugin (or global default) block, e.g. `auto_batch: { enabled: bool, headroom_frac: float, max_cap: int, probe_samples: int }`. **Default `enabled: false`** for the first rollout (opt-in allowlist behavior via config), so nothing changes until explicitly turned on. (No `rtol`/`atol` — there is no allclose self-test; `probe_samples` ≥2 is just the cost-probe size.)
- `metric-pinned` plugins ignore `auto_batch` entirely.

## 8. Testing Strategy

- **Existing goldens stay byte-identical** — they run at the fixed batch with `auto_batch` disabled (the default). No re-bless, no CI change. This is the regression tripwire.
- **`size_batch` unit tests** (CPU + mocked `memory_allocated`/`mem_get_info`): two-point cost fit (`fixed`+`per_sample`), usable-free formula, clamp to floor/cap, headroom, and **non-increasing/noisy probe → degrade to floor, never raise**.
- **`run_with_oom_backoff` unit tests** (no GPU — `run_fn` raises `torch.cuda.OutOfMemoryError` synthetically): start at the predicted batch; on OOM `empty_cache` + halve + rerun down to the floor; succeed → return the run result at the adopted batch; exhaust → run at floor. Assert the exact batch sequence (e.g. start 32 OOM → 16 OOM → 8 ok) and that each retry actually re-invokes `run_fn` at the smaller batch (real-run validation, not an omniscient probe).
- **Classification gating test**: a stub `reduction-accumulating` (and `metric-pinned`) plugin must NOT be auto-batched in v1 (assert it keeps the fixed batch). There is no numeric self-test, so this purely-structural gate is the safety — pin it.
- **Integration on a tiny model** (`batch-invariant` = ma_detection phase-a): with `auto_batch.enabled=true`, the pass runs through `run_with_oom_backoff` and produces the **same** blacklist as the fixed-batch run (grouping-independent), and a forced-OOM `run_fn` halves correctly.
- **v2 parity test (later):** pinned-reduction output at varying forward batch equals the fixed-grouping reference within a *tight* tolerance.
- **DP test (later):** replicas converge on `min(candidate)` and the reduced Gram matches the reduced fixed-batch reference.

## 9. Risks & Open Questions

- **Borderline threshold flips** (ablation_filter `ΔNLL > threshold`): even small fp drift could flip a candidate sitting exactly on the threshold, AND ablation_filter's NLL is `reduction-accumulating` (drift grows with the holdout token count). So ablation_filter stays at fixed batch until v2 pins its NLL reduction, and even then needs a decision-margin check. Do **not** auto-batch ablation_filter in v1.
- **Determinism mode**: the pipeline does **not** set `torch.use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` (zero hits in `src/`; noted at `activation_hooks.py:311`). Auto-batch assumes non-deterministic mode. Enabling determinism would change the fixed-batch reference too (goldens unaffected since they'd re-bless together) but tightens the achievable forward-drift floor; out of scope, stated so the tolerance budget is understood.
- **Probe cost**: the two-point (bs=1, bs=2) reference probe adds a few forwards per plugin per run. Bounded and dwarfed by the full calibration it accelerates.
- **Fragmentation**: a successful probe size can still fragment over a long run. Mitigation: `headroom_frac` margin + `empty_cache()` between phases; the fixed-batch fallback remains on a mid-run OOM.

## 10. Rollout

1. **v1 infra**: build `size_batch` (two-point sizing, precise free signal, identical-forward cost probe, degrade-to-floor on bad probe) + `run_with_oom_backoff` (real-run halving) + classification scaffolding + tests. `auto_batch` default-off. Apply only to `batch-invariant` plugins (drift structurally zero) — establishes the mechanism without touching any accumulating reduction and with no subset self-test.
2. **v2 cov (the wall)**: classify Stage-3 covariance as `reduction-accumulating`, pin its Gram reduction to a fixed grouping (co-resolved with `_resolve_cov_window` G), then enable auto-batch for cov and measure adoption/drift on a real GPU. This is the primary payoff.
3. **DP cov**: extend the cov resolver to the data-parallel path with the all-replica `min(candidate)` agreement + reduced-reference gate (superseding the deferred `_resolve_cov_batch_size` auto branch).
4. **Broaden**: `block_refine`, `phase_b` per their class; `ablation_filter` only after its NLL reduction is pinned (v2) and the threshold-margin tolerance is set. Never `gen` (metric-pinned).
5. **Model-sharded path** (§6, `min` usable-free across shard devices) is deferred **beyond** DP — out of v1/v2/DP scope; listed so its omission from steps 1–4 reads as intentional, not missed.

Per project standard, every code step above goes through the **plan/review loop** then the **implementation/review loop** — reviewer→fixer ping-pong addressing all five categories (Critical→Nitpick), each loop closing only on all-none.
