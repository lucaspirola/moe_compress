# Stage-3 rank-deciding spectra speedup (improvement #5)

**Date:** 2026-06-14 (amended 2026-06-15 per reviewer corrections)
**Branch:** `research/stage3-spectra-speedup`
**Scope:** design-only. Cut the ~90 min fp64-CPU group-stat phase WITHOUT
changing any rank decision (0-rank-flip, byte-identical-rank requirement).
A reviewer checks this; plan/impl loops build it.

**Reviewer corrections folded in (verdict unchanged — Option B):** (1) the
spawn ProcessPool MUST force `multiprocessing.get_context("spawn")` —
fork-after-CUDA-init deadlocks (precedent `humaneval.py:374-381`); (2) config
default is `spectra_workers=1` (serial, byte-identical) — parallelism is opt-in
until a one-time golden bless; (3) 1-thread-per-worker pinning is a FIDELITY
invariant (multi-thread BLAS re-associates fp sums → measured ~1e-11 drift),
not just perf; (4) `effective_rank` is the bit-sensitive pivot and the real
fidelity gate; (5) speedup quoted ~8-13 min (spawn startup + straggler tail
folded in).

---

## 1. Root cause (one-liner)

The ~90 min pole is a **serial Python loop of ~24k fp64 `svdvals` run on
CPU** (`orchestrator.py:658-677` → `d_rank_allocate.py:381-395`), while both
H200 GPUs sit idle by design — the fp64-CPU residency is the
device-independence / 0-rank-flip guarantee, NOT an inherent serial
requirement. The work is embarrassingly parallel per `(layer, matrix, expert)`
and is being throttled because LAPACK's single-matrix SVD scales poorly past
~8-10 intra-op threads, so the default 20-/200-thread intra-op call leaves
most cores idle on each individual svdvals.

### Exactly what runs today

`orchestrator.py:656-680`:

```python
for k, ref in enumerate(moe_layers):                       # ~40 layers, SERIAL
    banks = build_banks(ref)
    for name in MATRIX_NAMES:                               # gate/up/down, SERIAL
        ...                                                 # average Stage-2 A-cov
        group_stats[(ref.layer_idx, name)] = _group_stat(   # the pole
            ref.num_routed_experts, banks[name], A_g=A_g)
```

`d_rank_allocate.py:362-395` (inside `_group_stat`):

```python
A64 = A_g.to(device="cpu", dtype=torch.float64)             # CPU fp64
A64 = 0.5 * (A64 + A64.T)
L_A = torch.linalg.cholesky(A64 + jitter)                   # one Cholesky / group
svs = []
for e in range(n_experts):                                  # ~200 experts, SERIAL
    W = bank.get(e).detach().to(device="cpu", dtype=torch.float64)
    M = L_A @ W.T                                           # [d_in, d_out]
    s = torch.linalg.svdvals(M)                             # CPU fp64 svdvals — POLE
    svs.append(_pad(s, min(d_out, d_in)))
mean_s = torch.stack(svs).mean(0)                           # mean-of-spectra
# -> eff_rank (Eq.1/2) -> _d_rank_allocate -> per-group integer ranks (rank_map)
```

Total svdvals ≈ `~40 layers × 3 matrices × ~200 experts ≈ 24k`, each a
`svdvals` of a `[d_in, d_out]` matrix (e.g. `[2048, 1536]`), all fp64, all on
CPU, all in one serial Python loop.

### Why fp64-CPU (the rationale, confirmed from git)

Commit **`f556aa3`** ("Tier-2 device-independent fp64 spectra") and the
deviation block `D-drank-fp64-spectrum` (`d_rank_allocate.py:193-219`):

> *"a 3-seed measurement on real shapes showed an FP32-GPU spectrum flips
> 2–3/216 ranks vs the FP32-CPU golden ... whereas FP64 agrees across CPU and
> GPU to ~1e-14 (0 rank flips)."*

So the choice is **device-independence**, not a fp64-GPU correctness defect.
This is load-bearing for the options below: it is *fp32-on-GPU* that flips
ranks, **not fp64-on-GPU**. The fp64 result is what makes the rank decision
device-portable in the first place.

The downstream consumer is the **integer rank** (`eff_rank` → Eq.7 →
`int(round())` in `_d_rank_allocate.py:520`). The 0-flip requirement is on the
**integer `rank_map`**, not on the bit pattern of the singular values — the
singular values feed a `round()` whose output is what must be preserved.

### The bit-sensitive pivot: `effective_rank` (the real fidelity gate)

The single load-bearing scalar is **`_GroupStats.effective_rank`** — the Python
`float` produced at `d_rank_allocate.py:410` (`exp(-Σ p log p)` over the
squared-SV distribution). It is the *only* value that crosses from the fp64
spectrum into the rank decision: `_d_rank_allocate` divides it by `omega`
(int), feeds `math.sqrt`, scales by `T_budget`, then `int(round())`s the result
(`d_rank_allocate.py:494, 519-520`). **A sub-ULP drift in `effective_rank` is
exactly what tips a borderline `round()` and flips a rank.** So the real
fidelity gate for ANY change here is **exact-float (`torch.equal`-grade)
equality on `effective_rank` per group** — NOT merely equality on the final
integer dict. The integer dict can coincidentally agree at one `T_budget` while
the float silently drifts, leaving a latent flip that surfaces at a different
`T_budget` / `svd_rank_ratio`. Every option below is judged against bit-exact
`effective_rank` (and its upstream `singular_values_mean`), and the test plan
(§5) makes `effective_rank` the primary assertion, not the integer dict.

---

## 2. The parallelism opportunity (empirically measured)

The 24k svdvals are fully independent per `(layer, matrix, expert)`. Three
measured facts (RTX-5080 dev box, 20 cores, fp64, shape `[2048,1536]`; the
resume box is a 200-core H200 host — see §6 scaling):

**Fact A — a single fp64 svdvals does NOT scale past ~10 threads.**

| intra-op threads | ms / svdvals |
|---|---|
| 1 | 739.8 |
| 10 | 114.7 |
| 20 | 134.9 (slower than 10) |

LAPACK gesdd saturates ~8-10 threads on one matrix; 20 threads is *worse*
than 10. So the current code's default-thread serial loop wastes the box: it
runs ONE svdvals at a time across all cores, and that one call cannot use them.

**Fact B — `ThreadPoolExecutor` makes it WORSE (svdvals holds the GIL / contends).**

| approach (40 svdvals) | wall |
|---|---|
| serial, 20 intra-threads | 6.15 s |
| ThreadPool 8 workers ×1 thread | 12.21 s |
| ThreadPool 16 workers ×1 thread | 14.76 s |

`torch.linalg.svdvals` does not release the GIL cleanly enough for a thread
pool to help — threads serialize and add overhead. **A ThreadPool is the wrong
tool; reject it.**

**Fact C — `ProcessPoolExecutor` (1 intra-op thread per worker) wins ~3.8×.**

| approach (40 svdvals incl. matmul) | wall |
|---|---|
| serial, in-proc, 20 intra-threads | 39.65 s |
| ProcessPool 8 workers ×1 thread | 11.35 s |
| ProcessPool 16 workers ×1 thread | 10.44 s (**3.8×**) |
| ProcessPool 20 workers ×1 thread | 10.56 s |

(The serial number here is larger than Fact B's because it includes the
`L@W.T` matmul per call; the relative ProcessPool win is the point.)

**Conclusion:** the win comes from **many independent processes, each pinned to
1 intra-op thread** (`OMP_NUM_THREADS=1` / `torch.set_num_threads(1)` in the
worker), NOT from threads and NOT from cranking intra-op threads on a serial
loop. This is consistent with the existing
[[reference_fp64_gpu_svdvals_blackwell]] device note and with how Stage-6
HumanEval already uses a `ProcessPoolExecutor`.

**1-thread-per-worker is a FIDELITY invariant, not just a perf tuning.** The
current serial path runs each `svdvals`/`mean(0)` under the default multi-thread
BLAS. To stay byte-identical, the parallel workers must reproduce the SAME
floating-point reduction order. A multi-threaded BLAS **re-associates** the
sums inside `svdvals` and the `mean(0)` accumulation across a
thread-count-dependent tiling — fp addition is non-associative, so
`set_num_threads(N≠1)` in a worker can shift `effective_rank` in the low bits
and silently flip a borderline rank. The only way to guarantee the worker's
arithmetic matches the reference is to pin **exactly 1 intra-op thread per
worker**, which also happens to be the fastest configuration (Fact A/C). So the
pinning is doubly mandatory: it is the speedup AND the bit-for-bit guarantee.
**Measured (this box):** the same fp64 `svdvals`+`mean(0)` under 1 vs 20 threads
is NOT bit-identical — `mean_s` diverges by ~1.3e-11 and `effective_rank` drifts
in the last bits (`796.78621239945` @1-thread vs `796.7862123994486`
@20-threads). So multi-threaded BLAS DOES re-associate the spectrum at the
~1e-11 level — well above the ~1e-14 GPU/CPU fp64 floor and large enough to be
the dominant fidelity term. The integer `round()` almost never flips at 1e-11,
but "almost never" is not the 0-flip bar.

**Consequence for the byte-identical claim (corrected — see §4):** because the
current serial path runs under the default 20 threads, pinning workers to 1
thread changes the bits relative to today's golden. B is therefore **NOT
zero-re-bless**; it requires a **ONE-TIME golden re-bless to the canonical
1-thread reduction**. After that single reviewed bless, every path — serial
`spectra_workers=1` and parallel — pins 1 thread, so the result is **bit-stable
and thread-count- AND core-count-independent forever** (and still
device-independent: 1-thread fp64 LAPACK is the reference on any box). This is
still far safer than Option A's *per-device* re-bless + GPU coupling: B's bless
is one-time and the bits never move again. The §5 equality test pins both legs
to 1 thread so they share the canonical order.)

---

## 3. The three options + fidelity analysis

### Option A — move the fp64 spectra onto the H200 GPU

**Idea:** run the per-expert `svdvals(L_A @ W.T)` in **fp64 on the H200 GPU**
(cuSOLVER) instead of CPU. H200 fp64 is only ~2× the H200 fp32 cost (vs ~14×
on consumer Blackwell — [[reference_fp64_gpu_svdvals_blackwell]]), and the
GPU's throughput plus batching could absorb 24k small SVDs.

**Fidelity — the make-or-break question:** does fp64 cuSOLVER svdvals produce a
rank decision identical to fp64 LAPACK svdvals?

- **The bit pattern will differ.** svdvals is iterative (Jacobi / QR-based);
  cuSOLVER on GPU and LAPACK on CPU use different algorithms and reduction
  orders. IEEE-754 fp64 is deterministic *per implementation* but not *across*
  implementations — the singular values will differ in the last few bits
  (~1e-14 relative, exactly the magnitude commit `f556aa3` measured).
- **BUT the rank decision is already proven device-independent at fp64.**
  There is an existing, committed test —
  `tests/test_stage3_tier2.py::test_d_rank_alloc_fp64_cpu_equals_fp64_gpu`
  (lines 86-95) — that runs the *entire* `_group_stat` → `_d_rank_allocate`
  phase on CPU and on CUDA with identical inputs and asserts
  **`rm_cpu == rm_gpu`** (integer rank_map equality). It was written precisely
  as "the guard against a future regression that moves a fp32 spectrum back
  onto the GPU." So fp64-GPU == fp64-CPU *at the integer-rank level* is already
  a tested invariant of this codebase.

**Why Option A is still the riskier choice:**

1. The test proves equality on a *toy* shape (`d_out=16, d_in=12, n_experts=4`,
   2 groups). It does NOT prove that on the **real** 24k-svdvals production
   shapes there is never a borderline expert whose `eff_rank` sits within
   ~1e-12 of an `int(round())` boundary where the ~1e-14 GPU/CPU spectrum
   difference tips the rounding. The 0-flip guarantee on production shapes
   would need a **fresh full-model fp64-CPU-vs-fp64-GPU rank_map diff** to
   bless — i.e. a re-bless gate, exactly what Tier-2 was trying to avoid.
2. The current golden (`tests/test_stage3_golden_snapshot.py`) is pinned to the
   **fp64-CPU** spectrum. Even if every rank is identical, switching the
   producer to GPU changes which path mints the golden; any single borderline
   flip on a future architecture port silently breaks reproducibility and the
   failure mode is a per-device re-bless — the precise fragility Tier-2
   removed.
3. It still leaves the *Cholesky* and the cross-device `.to()` traffic, and
   adds GPU-memory pressure during a phase currently chosen to be CPU-resident
   so it co-locates with the `map_location="cpu"` covariances.

**Robustness guard (if Option A is ever pursued):** round the singular values
to N significant digits before the energy/`eff_rank` computation
(`s = (s * 10**N).round() / 10**N` in fp64, with N chosen so 1e-14 GPU/CPU
noise is quantized away but the spectrum's discriminating power is intact,
e.g. N≈10). This would make GPU and CPU agree by construction at the cost of a
new (also re-bless-requiring) golden. **Not recommended** — it trades one
re-bless for another and adds a magic constant.

**Verdict on A:** technically the biggest single-knob win and *probably*
0-flip given the existing test, but it carries fidelity risk on real shapes
(borderline `round()`), requires a full-model re-bless, and reintroduces the
device-coupling Tier-2 deliberately removed. **Not the recommendation.**

### Option B — parallelize the fp64 svdvals across CPU cores (RECOMMENDED)

**Idea:** keep the computation **bit-for-bit identical** (same fp64 LAPACK
svdvals, same CPU, same Cholesky, same `mean(0)`, same `round()`) and only
change *concurrency*: dispatch the independent per-expert (and/or per-group)
svdvals across a **`ProcessPoolExecutor`**, each worker pinned to 1 intra-op
thread.

**Fidelity — 0-flip and bit-stable, after a one-time 1-thread re-bless.** The
arithmetic operator is unchanged: the same `torch.linalg.svdvals(L_A @ W.T)` in
fp64 on CPU runs in a worker process instead of the main process. fp64 LAPACK is
deterministic for a given matrix *at a fixed intra-op thread count*. Two
ordering invariants make the result bit-stable:

1. **1 intra-op thread per worker** — pins the BLAS reduction order so the
   spectrum/`mean(0)` bits do not move with worker count (the measured 1-vs-20
   thread ~1e-11 drift, §2). This is the canonical reduction.
2. **Expert-index-ordered `mean(0)`** — the per-group `torch.stack(svs).mean(0)`
   must reassemble `svs[0..n-1]` in expert index order (fp reduction is
   order-sensitive at ~1e-16); the chosen `(layer, matrix)`-group unit keeps the
   `mean(0)` *inside one worker*, so this is automatic.

Under both, the result is **byte-identical across any core/thread/worker count
and any box**. It is NOT byte-identical to *today's* golden, which was minted
under default-threaded BLAS — so B carries a **single, one-time, reviewed golden
re-bless to the 1-thread canonical reduction** (§4). After that bless the bits
never move again. The integer-rank decision is robust at the ~1e-11 level either
way, but the bar is bit-exact `effective_rank`, which 1-thread pinning + ordered
mean delivers.

**Concrete design:**

- **Pool type:** `concurrent.futures.ProcessPoolExecutor` (NOT threads —
  Fact B proves threads regress). Reuse the project's existing
  process-pool idiom (Stage-6 HumanEval uses one).
- **CUDA-fork safety (mandatory):** the parent process is **already
  CUDA-initialized** by the time the group-stat loop runs (Stage 3 is
  GPU-resident; the banks live on GPU and the cov pass ran on CUDA). A default
  `fork` start method **after CUDA init deadlocks the child**. The pool MUST be
  created with `mp_context=multiprocessing.get_context("spawn")` — exact in-repo
  precedent at `stage6/plugins/humaneval.py:374-381` ("FORCE spawn (host default
  may be fork — fork-after-CUDA-init can deadlock the child)"). To bound the
  spawn re-import cost, put the worker entry point `_group_stat_payload` (and the
  tiny duck-typed bank it rebuilds) in a **lean, torch-light top-level module**
  so each spawned child re-imports the minimum — the same discipline humaneval's
  "torch-free worker leaf module" uses. Spawn startup (interpreter + imports per
  worker, one-time) is folded into the §4 estimate.
- **Worker pinning:** each worker sets `torch.set_num_threads(1)` /
  `OMP_NUM_THREADS=1` at startup (an `initializer=`). This is BOTH the perf
  crux (Fact A: 1-thread workers × many processes beats few processes ×
  many-thread svdvals) AND the fidelity invariant (§2: it pins the BLAS
  reduction order so the bits are reproducible).
- **Unit of work:** the cleanest granularity is **per `(layer, matrix)`
  group** = one Cholesky + its `n_experts` svdvals + the `mean(0)`, returning
  the finished `_GroupStats`. This keeps the order-sensitive `mean(0)`
  *inside* one worker (so reassembly across workers is just a dict keyed by
  `(layer_idx, name)` — order-independent), minimizes IPC (ship `A_g` and the
  `n_experts` weight tensors in, ship one small `_GroupStats` out), and amounts
  to **~120 independent tasks** (40 layers × 3 matrices) — ample for a
  20-to-200-core box. Finer per-expert granularity is possible but reintroduces
  the cross-worker ordered-mean and multiplies IPC; **group-level is the right
  unit**.
- **Worker count:** `min(n_groups, n_cpu)` capped by a config knob
  (default e.g. `min(os.cpu_count(), 64)`); on the 200-core H200 host this
  fans the ~120 groups out near-fully.
- **IPC payload:** the worker needs `A_g` (one `d_in×d_in` fp32 cov, e.g.
  2048² ×4B ≈ 16 MB) + `n_experts` weight slabs. To avoid serializing GPU
  tensors across the process boundary, move each group's `A_g` and weights to
  **CPU** in the *parent* before submit (they already go to CPU inside
  `_group_stat`; just hoist the `.cpu()` to the dispatch site so only CPU
  tensors cross the boundary). The weight gather (`bank.get(e)`) must happen in
  the parent because `build_banks`/`ExpertMatrixBank` holds live module
  references that cannot be serialized.
- **Determinism / shared-state hazards:** none. `group_stats` is the only
  shared structure and it is write-once per key from the parent as futures
  complete — no concurrent mutation of `rank_map`, no shared cov dict in
  workers (each worker gets its own `A_g` copy). `_d_rank_allocate` runs once
  in the parent **after** all groups complete (it is microseconds — timing doc
  phase (2)).

**Where the code changes:**

- **`stage3/orchestrator.py:656-680`** — the serial
  `for k, ref in enumerate(moe_layers): ... for name in MATRIX_NAMES:` loop
  becomes: (1) a parent-side gather building a list of serializable per-group
  payloads `(layer_idx, name, A_g_cpu_fp32, [W_e_cpu]...)`; (2) a
  `ProcessPoolExecutor(mp_context=get_context("spawn"),
  initializer=_pin_one_thread)` `.map`/`as_completed` over a new top-level
  `_group_stat_payload(payload) -> ((layer_idx,name), _GroupStats)` wrapper in a
  lean torch-light module; (3) reassemble `group_stats` dict from results. The
  `_group_stat` body in `d_rank_allocate.py` is **unchanged** (the wrapper just
  reconstructs a tiny duck-typed bank from the shipped weight list, exactly as
  `tests/test_stage3_tier2.py::_FakeBank` already does).
- A **config gate** `stage3_svd.spectra_workers` that **defaults to `1`
  (serial, single-thread-pinned, byte-identical) — parallelism is OPT-IN**.
  Project rule: byte-identical default. The default flips to parallel
  (e.g. `min(cpu_count, 64)`) **only after** the one-time golden diff blesses
  the 1-thread canonical reduction (§4/§5). Until then, `spectra_workers > 1`
  is an explicit, reviewed opt-in for the H200 run; `1` stays the shipped
  default so CI and any unaudited run get the canonical serial path.
- No change to `d_rank_allocate.py`'s numerics.

### Option C — overlap the CPU group-stat with the GPU α-search/factoring

**Idea:** pipeline phases so the idle GPUs do useful work during the CPU
spectra.

**Data-dependency analysis (from the timing doc + orchestrator order):**

```
group-stat (CPU)  →  allocate_ranks  →  α-search (GPU)  →  factor loop (GPU)
   phase 1              phase 2            phase 3            phase 4
```

- α-search (`swift_svd_alpha.py`) consumes `ranks` (the `rank_map`), which is
  produced by `allocate_ranks`, which consumes **all** `group_stats`. So
  α-search **cannot start until the entire group-stat phase is done** — there
  is no phase-1↔phase-3 overlap to exploit. The dependency is hard.
- The only intra-phase-1 overlap is: while the CPU computes group `g`'s
  svdvals, the GPU could pre-stage the *next* layer's weights / pre-compute the
  α-search's cached eigh. This is marginal and entangles two plugins.

**Verdict on C:** the hard `group_stats → ranks → α-search` dependency means
there is **no clean phase-level overlap**. C reduces to "do phase 1 faster,"
which is exactly Option B. **Not independently worthwhile; B subsumes the win.**

---

## 4. Recommendation

**Option B — ProcessPool-parallelize the fp64-CPU svdvals at `(layer, matrix)`
group granularity, 1 intra-op thread per worker.**

- **Fidelity:** bit-stable and 0-rank-flip via two invariants — **1 intra-op
  thread per worker** (pins the BLAS reduction order; the measured 1-vs-20
  thread drift is ~1e-11, §2) and **expert-index-ordered `mean(0)`** (kept
  inside one worker by the group-granularity unit). The bit-sensitive pivot is
  `effective_rank` (§1), and 1-thread pinning + ordered mean make it
  reproducible across any core/box. **Caveat (corrected):** because today's
  golden was minted under default-threaded BLAS, B requires **one** reviewed
  golden re-bless to the 1-thread canonical reduction; after that the bits never
  move again (thread/core/box-independent). This is still far safer than A's
  *per-device* re-bless + GPU coupling — B's bless is one-time and permanent.
- **Default is `spectra_workers=1` (serial, byte-identical) — parallelism is
  OPT-IN.** The default flips to parallel only after the one-time golden diff
  blesses it (project byte-identical-default rule).
- **Expected speedup:** **~3.8× measured on 20 cores**; on the **200-core H200
  host** the ~120 independent group tasks fan out near-fully until bounded by
  the single slowest group (max-expert layer × largest matrix) + the **`spawn`
  pool startup (interpreter + per-worker re-import, one-time)** + the IPC /
  straggler tail. Folding those overheads in, the realistic landing is the
  ~90 min phase → **~8-13 min** (not a flat 20×; the straggler group and spawn
  cost dominate once cores ≫ groups).
- **Exact change location:** `stage3/orchestrator.py:656-680` (the serial
  group-stat loop → spawn-context ProcessPool dispatch + ordered reassemble) +
  a new serializable top-level wrapper `_group_stat_payload` in a lean
  torch-light module + a `stage3_svd.spectra_workers` config gate (default
  `1`). `d_rank_allocate.py` numerics **untouched**.
- **Why not A:** A is a bigger single knob but carries real fidelity risk on
  production shapes (borderline `int(round())` tipped by the GPU/CPU spectrum
  difference), needs a **per-device** re-bless, and reintroduces the
  device-coupling Tier-2 deliberately removed. B's re-bless is one-time and
  permanent; B gets most of the win with a bounded, reviewed fidelity cost.
- **B + C:** C adds nothing over B (hard dependency, no phase overlap). Ship B
  alone.

**Optional follow-up (not in scope of #5):** if B's ~8-13 min is still a pole on
a future much-bigger model, *then* revisit A behind the existing
`test_d_rank_alloc_fp64_cpu_equals_fp64_gpu` guard plus a full-model rank_map
diff bless — but only as a deliberate, separately-reviewed re-bless.

---

## 5. Fidelity test plan

The new parallel path MUST produce a **bit-identical** `effective_rank` /
`singular_values_mean` (and hence identical `rank_map`) to the canonical
**1-thread** fp64-CPU path. The primary gate is `effective_rank` (the
bit-sensitive pivot, §1), NOT the integer dict (which can coincidentally agree
while the float drifts). All legs pin `torch.set_num_threads(1)` so they share
the canonical reduction. Tests:

1. **`test_group_stat_parallel_equals_serial` (NEW, PRIMARY — `effective_rank`
   gate).** Build a small multi-group fixture (reuse `_FakeBank` from
   `test_stage3_tier2.py`). Run the group-stat phase (a) serially (1-thread
   pinned) and (b) through the spawn ProcessPool wrapper with
   `spectra_workers > 1` (each worker 1-thread pinned). Assert, in priority
   order:
   - **`serial_gs[g].effective_rank == parallel_gs[g].effective_rank`** as
     exact Python-float equality for every group — THE fidelity gate (a sub-ULP
     drift here is a latent rank flip),
   - `torch.equal(serial_gs[g].singular_values_mean,
     parallel_gs[g].singular_values_mean)` (bit-identical spectra — proves the
     `mean(0)` reassembly is expert-index-order-exact),
   - `_d_rank_allocate(serial_gs, T) == _d_rank_allocate(parallel_gs, T)` for a
     **sweep of `T_budget`/`svd_rank_ratio`** (the int dict — checked across
     several budgets so a borderline `round()` cannot hide a float drift at one
     budget).
2. **`test_thread_pinning_holds_the_bits` (NEW — proves #3, the fidelity
   invariant).** Compute one group's `effective_rank`+`singular_values_mean`
   under `set_num_threads(1)` and under `set_num_threads(>1)` and assert they
   **DIFFER** (documents that multi-thread BLAS re-associates the sum — the
   measured ~1e-11 drift), THEN assert the pool's worker initializer yields
   `torch.get_num_threads() == 1` inside the worker AND that the worker's
   `effective_rank` matches the 1-thread reference bit-for-bit. This pins the
   pinning as a correctness contract: if a future change drops
   `set_num_threads(1)` in the worker, this test goes red on the bit mismatch,
   not just on a perf regression.
3. **`test_group_stat_worker_order_invariant` (NEW).** Run with
   `spectra_workers ∈ {1, 2, 4}` and a deterministic seed; assert all three
   produce the **same** `effective_rank` (exact) and `singular_values_mean`
   (`torch.equal`) — guards a refactor that lets per-expert results reassemble
   out of order inside a group, or that makes the result worker-count-dependent.
4. **Existing guard kept green:**
   `test_stage3_tier2.py::test_d_rank_alloc_fp64_cpu_equals_fp64_gpu` continues
   to pass (B does not touch device residency — still CPU-fp64; pin its legs to
   1 thread to match the canonical reduction).
5. **Golden re-bless + lock (one-time).** Because today's golden was minted
   under default-threaded BLAS, the bless is: (a) run the real
   `configs/qwen36_*.yaml` Stage-3 with `spectra_workers=1` pinned to 1 thread,
   mint the canonical `rank_map.{fp32,bf16}.json`; (b) human-review the diff vs
   the current golden (expected: ≤ a handful of borderline ranks moved by the
   1e-11 reduction change — each one inspected); (c) pin that as the new golden;
   (d) assert the **parallel** path (`spectra_workers>1`) reproduces it with
   **0 diff** (test 1 guarantees this). After the bless, `spectra_workers=1` and
   any `>1` are byte-identical forever.
6. **Worker pinning assertion** (subsumed into test 2): the pool initializer
   sets `torch.get_num_threads() == 1` inside the worker.

**The all-or-nothing fidelity bar:** exact-float equality on `effective_rank`
per group (test 1, first assertion) + `torch.equal` on `singular_values_mean`.
The integer `rank_map` equality is the *downstream* check, swept across budgets
— necessary but not sufficient on its own (a float drift can pass one budget).
There is no "allclose" tolerance for a rank decision.

---

## 6. Scaling note (dev box vs resume box)

All measurements above are on the **20-core RTX-5080 dev box**. The resume box
is a **200-core H200 host**. The 24k svdvals are independent and the design
fans them out at ~120 group-tasks; with 200 cores ≫ 120 groups, every group
runs concurrently and wall time collapses to roughly **the single slowest
group** (the max-expert layer's largest matrix) **plus the `spawn` pool startup
(per-worker interpreter + re-import, one-time) plus the IPC + straggler tail**.
Folding those overheads in, the realistic landing is **~8-13 min** (~7-11×), not
a flat 20× — the straggler group and spawn/IPC cost, not core count, set the
floor. Keeping the worker leaf module torch-light bounds the spawn re-import
cost (the humaneval precedent). If finer scaling is ever needed, split the
largest groups' per-expert svdvals into sub-tasks (reintroduces the
expert-index-ordered mean across workers — only worth it if a single group
dominates the tail).
