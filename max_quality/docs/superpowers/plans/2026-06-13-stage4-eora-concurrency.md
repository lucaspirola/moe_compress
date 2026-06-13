# Plan: Stage-4 EoRA concurrency engine (make the N-GPU placement actually overlap)

**Date:** 2026-06-13
**Branch / worktree:** `feat/stage4-eora-mg` @ `/home/lucas/ai/wt-stage4` (code root: `max_quality/`)
**Status:** plan — not yet implemented

---

## Goal

Stage-4 EoRA already *places* each per-expert solve on an N-GPU worker band, but
executes the bands **serially**: the dispatch loop
(`max_quality/src/moe_compress/stage4/plugins/eora_compensation.py:693-718`)
calls `_solve_expert_tile(...)` and immediately does a **blocking** device→home
gather `U_corr[e] = Uc.to(device=dev, ...)` (`:709-710`) every iteration, which
synchronizes the producing device's stream before the next solve begins. Result:
experts banded to `cuda:1..cuda:N-1` run one-at-a-time → **~0× speedup** despite
the "N-GPU" label. This is documented as the central finding of the analysis:
"a correct device-distribution skeleton **missing its concurrency engine**"
(`max_quality/docs/multigpu_analysis/stage4.md:28-37`, `:131-157`, `:282-288`).

**This plan adds the concurrency engine** so the placed bands run concurrently,
while keeping the assembled `U_corr`/`V_corr` **byte-identical** to today's
serial output.

### Non-negotiable correctness invariants (a live ablation may restart-resume from Stage 4)

1. **Result byte-identical to current output.** Per-expert solves are independent
   pure functions (`eora_compensation.py:405-412` docstring;
   analysis `stage4.md:159-166`). Concurrency must not change numerics:
   - The row-gather into `U_corr[e]`/`V_corr[e]` MUST stay in **ascending-expert
     order** regardless of completion order (gather writes disjoint rows, no
     cross-expert reduction — `eora_compensation.py:651-657`, `:708-710`).
   - fp accumulation order for the log-only residual sums
     (`res_before_acc`/`res_after_acc`, `:647-648`, `:714-715`) MUST be
     preserved (sum in ascending-e, single `.item()` sync after all joins).
   - The gate→up whitening-spectrum memo (`gate_spectra[e]`, `:622`, `:699`,
     `:706`) MUST remain per-expert and survive the gate→up window, now under
     concurrent writers.
2. **1-GPU path unchanged / byte-identical.** Default `eora_workers=1`
   (`orchestrator.py:69-85`, `:120`) → `effective_workers<=1` → every expert on
   the home device, same serial order as before this change. The concurrency
   layer is a **no-op** in that case (gated behind `effective_workers > 1`).
3. **Golden gate.** `tests/test_stage4_golden_snapshot.py` (the integer-rank /
   param-count byte-identical snapshot) MUST stay green **without**
   `MOE_REGEN_GOLDEN`. The float U/V byte-identity is enforced by the existing +
   new equivalence tests in `tests/test_stage4_multigpu.py`.

---

## Architecture

### Primitive choice: `ThreadPoolExecutor`, one worker thread per worker-device — NOT process spawn, NOT raw CUDA streams

**Why not process spawn (the Stage-3 pattern).** Stage-3 covariance collection
uses `torch.multiprocessing` spawn
(`stage3/plugins/covariance_collection.py:1213-1288`, `run_dp_covariance_collection`
→ `ctx.Process(target=_cov_replica_worker)`) **because it runs forward passes** —
each replica needs its own resident model + CUDA context + `CUDA_VISIBLE_DEVICES`
pinning, and reduces results across processes via on-disk spill. **Stage-4 EoRA
runs ZERO forward passes** (`stage4.md:66-79`, `:257-262`); it is in-process
linear algebra on weight-shaped tensors already in this process's memory
(`originals`, `fe.*_U/_V`, `A_cov`). Spawning processes would force re-loading
those tensors (50+ GB `originals`) into each child — absurd. The work is in CUDA
kernels (`eigh`/`svd`/`matmul`) that **release the GIL and run async on the
target device's stream**, so Python threads achieve real device overlap.

**Why threads over hand-rolled `torch.cuda.Stream`.** Both work. Threads are the
strictly simpler and lower-risk option that matches an **existing, reviewed repo
primitive**: `utils/lsa_pool.py::parallel_map` is an order-preserving threaded
`map` with an explicit byte-identicality contract ("only changes *when*
independent pure functions run, never *what* they compute. Callers reassemble
results by precomputed index … so worker completion order is unobservable",
`lsa_pool.py:32-35`). We reuse that exact discipline. Raw streams would require
manual event/sync bookkeeping for the gather and a hand-managed default-stream
context per op — more surface for a determinism bug, with no upside here because
each worker device's solves already serialize on that device's default stream
within its own thread.

**Granularity: one thread per *worker BAND*, not one thread per *expert*.** The
unit of true parallelism is the *worker* (one band = one device's slice), not the
task — N experts on one card still serialize on that card. So we spawn one thread
per **band ordinal** `w` (the `0..effective_workers-1` index already used to
build `device_of` at `eora_compensation.py:687-691`), each owning the
**contiguous band** `eligible[w*per:(w+1)*per]` and carrying its band's target
device `devices[w]`. Within a thread, experts run in **ascending-e** order
(preserving the gate→up memo and the per-expert spectrum reuse). This keeps the
thread count tiny (= #workers), avoids GIL thrash, and means each thread touches
exactly one device → no cross-thread same-device contention.

> **CRITICAL — band by INDEX `w`, never by device OBJECT.** The CPU test seam
> passes `worker_devices=["cpu","cpu","cpu"]`, and
> `torch.device("cpu") == torch.device("cpu")` (hash-equal). Grouping experts by
> the device *object* would collapse all three CPU "workers" into ONE band → the
> threaded branch would never run and every CPU equivalence test would pass
> trivially as serial==serial, shipping the byte-identity-under-threading
> guarantee **untested**. The band ordinal `w` is distinct per worker regardless
> of device identity, so `["cpu","cpu","cpu"]` yields 3 real bands → 3 real
> threads → a faithful 3-GPU model on CPU. The RED tests (Tasks 1-2) assert
> `len(bands) == 3` so this can never silently regress to inline.

### Data-flow of the concurrent path (replaces the serial `for e in eligible:` block)

```
band_results: dict[e -> (Uc_home, Vc_home, take_eff, res_before, res_after)]   # filled by threads, keyed by e
gate_spectra: dict[e -> spectrum]                                              # filled by gate-band threads (per-e, disjoint across bands)

# bands keyed by ORDINAL w (NOT device object): {w: [experts...ascending]}, plus device_of[e]=devices[w]
# One thread per band ordinal w; ["cpu","cpu","cpu"] → 3 distinct w → 3 real threads.
def _run_band(experts):                      # experts for ONE band ordinal, ascending e
    for e in experts:                        # serial WITHIN a band (same-device work serializes anyway)
        tgt = device_of[e]
        Uc, Vc, take_eff, rb, ra, spec = _solve_expert_tile(..., tgt=tgt, gate_spectrum=gate_spectra.get(e), ...)
        if name == "gate_proj": gate_spectra[e] = spec        # disjoint key e — no cross-thread race on same key
        Uc_home = Uc.to(device=dev, dtype=dtype)              # gather to home INSIDE the thread (peer copy, async on tgt)
        Vc_home = Vc.to(device=dev, dtype=dtype)
        band_results[e] = (Uc_home, Vc_home, int(take_eff), rb, ra)   # disjoint key e — no race

# main thread, AFTER all bands join (+ guarded cuda.synchronize over worker devices if log_residuals):
for e in eligible:                           # ASCENDING-e — the byte-identity guarantee
    Uc_home, Vc_home, take_eff, rb, ra = band_results[e]
    U_corr[e] = Uc_home;  V_corr[e] = Vc_home
    eff_per_expert[e] = take_eff
    if log_residuals: res_before_acc += rb.to(dev); res_after_acc += ra.to(dev)   # ascending-e fp sum
```

**Why this is byte-identical.** The numerics of each tile come from
`_solve_expert_tile`, a pure function of its inputs (`eora_compensation.py:405-412`);
running it on a thread does not change its kernels or inputs. The ONLY
order-sensitive writes are (a) the `U_corr[e]/V_corr[e]` row placement and (b)
the residual fp accumulation — both are pulled OUT of the thread bodies and done
on the main thread in **ascending-e order**, exactly as the serial loop does
today. `dict[e]` writes from threads are to **disjoint keys** (each expert in
exactly one band) so there is no data race and no order dependence on the dict
fill. CPython dict item assignment to distinct keys is thread-safe under the GIL;
we additionally never read a key in a thread that another thread writes
(`gate_spectra[e]` is written by e's gate-band thread and read by e's up-band
thread, and the gate phase **fully joins** before the up phase begins — see Task 4).

**Memo tensors are read-only to consumers.** When the up pass consumes
`gate_spectra[e]`, `_compute_eora_factors` only ever *rebinds local names* to the
spectrum tensors and, if a device move is needed, produces NEW tensors via `.to`
(`eora_compensation.py:321-325`: `eigvecs_keep = eigvecs_keep.to(...)` etc.) — it
never mutates the memo's tensors in place. So a gate-band thread's stored
spectrum cannot be corrupted by the up pass, and concurrent up-pass reads of
distinct `gate_spectra[e]` entries are safe.

### GPU thread-safety details

- **Multi-device CUDA from multiple threads is safe with care.** Each thread
  issues ops to exactly **one** device (its band's `tgt`), so there is no
  cross-device op interleaving within a thread. The `.to(device=dev)` gather is
  the only cross-device copy and it happens inside the owning thread (peer copy
  from `tgt`→`dev`), enqueued on that thread's device context.
- **No explicit per-op stream needed.** Each thread uses the default stream of
  its device; PyTorch maintains a per-thread current-device, so setting
  `torch.cuda.device(tgt)` (or relying on the device carried by the tensors) is
  sufficient. We do NOT share a single stream across threads.
- **Final sync.** `ThreadPoolExecutor.__exit__` / `future.result()` joins the
  Python threads but does NOT by itself synchronize CUDA. The subsequent
  `Uc.to(device=dev)` gather (done inside each worker thread) is a copy that
  orders after the producing kernels on the **same** device (CUDA stream
  ordering), so each gathered tile is correct. For the U/V golden path the single
  per-matrix `widen_rank` consumption and the existing `res_*_acc.item()` host
  sync (`:736-737`) order the device work — **no extra
  `torch.cuda.synchronize()` is required for the U/V bytes.**
- **Residual path (`log_residuals=True`) — RECOMMENDED guarded sync.** The
  residual scalars take a different route: each worker thread produces `rb`/`ra`
  on its **worker** device, then the main-thread assembly does `rb.to(dev)`
  (`:714-715`) — a cross-device copy from `tgt`→`dev` issued on the **main**
  thread, which can race the worker's still-in-flight producing kernel on `tgt`.
  Because the residual scalars are log-only (never the golden), a silent race
  there would corrupt a logged number without tripping any byte gate. **Therefore
  RECOMMEND** a single guarded `torch.cuda.synchronize(d)` over the distinct
  worker devices, issued AFTER the pool joins and BEFORE the ascending-e
  assembly, gated `if log_residuals and concurrent`. One sync per matrix,
  default-off (residual logging is off by default), zero golden impact — cheap
  insurance against a silently-wrong residual scalar. See Task 4c and
  Open-Question #4.

### BLAS throttle note (CPU-stand-in path)

`lsa_pool.parallel_map` pins `torch.set_num_threads(1)` for the pool duration to
avoid `outer×cores` BLAS oversubscription. The CI EoRA equivalence tests run on
**CPU** (`eora_worker_devices=["cpu","cpu"]`), so concurrent CPU `eigh`/`svd`
would oversubscribe identically. The new helper MUST apply the same global
save/set(1)/restore-in-`finally` discipline **on the main thread only** (never
per worker — it is process-global). This is timing-only / byte-irrelevant for a
fixed BLAS build (`lsa_pool.py:21-27`). On real CUDA workers it is a harmless
no-op (the heavy work is on-GPU).

---

## Tech stack

- Python `concurrent.futures.ThreadPoolExecutor` (stdlib; already used at
  `utils/lsa_pool.py:41`, `utils/activation_hooks.py:1348`,
  `stage3/.../covariance_collection.py:795`).
- `torch` CUDA per-device context (`torch.cuda.device`), default streams.
- Reuse the design contract (not necessarily the literal function) of
  `utils/lsa_pool.py::parallel_map`.
- Tests: `pytest` on CPU (no GPU needed for the equivalence/golden gates); the
  real ≥2-GPU overlap is validated separately on hardware (deferred — see Out of
  scope).

---

## Files touched

| File | Change |
|---|---|
| `max_quality/src/moe_compress/stage4/plugins/eora_compensation.py` | Replace the serial `for e in eligible:` dispatch (`:693-718`) with the band-threaded engine + ascending-e assembly. New private helper `_run_expert_bands(...)`. |
| `max_quality/tests/test_stage4_multigpu.py` | Add tests: concurrent==serial byte-identity, gate→up memo correctness under threading, residual-accumulation order, single-worker no-op, (optional) stream/thread-safety smoke. |
| `max_quality/tests/test_stage4_golden_snapshot.py` | **No edit** — used as-is as the byte-identical guardrail (Task 6 runs it). |
| `max_quality/src/moe_compress/stage4/orchestrator.py` | **No edit for core** — `_resolve_eora_workers` and ctx wiring are unchanged. (Edited only in the OPTIONAL VRAM-aware task.) |

---

## Tasks (bite-sized TDD: write failing test → verify fail → minimal impl → verify pass → commit)

> Run all `pytest` from `max_quality/`. The interpreter here is `python3` (no
> `python` on PATH): `cd /home/lucas/ai/wt-stage4/max_quality && python3 -m pytest ...`.

### Task 0 — Baseline: confirm the current suite is green and the goldens exist

**No code change. Establish the gate before touching anything.**

```bash
cd /home/lucas/ai/wt-stage4/max_quality
python3 -m pytest tests/test_stage4_multigpu.py tests/test_stage4_golden_snapshot.py -q
```
Expected for the **combined** command: `8 passed, 1 skipped` — that is the 6
`test_stage4_multigpu.py` tests + the 2 `test_stage4_golden_snapshot.py` cases
(`[fp32]` + `[bf16]`), with the 1 skip being
`test_compute_eora_factors_relocates_cross_device_spectrum` (needs ≥2 CUDA
devices; CI has 0). Run alone, `test_stage4_multigpu.py` shows `6 passed, 1
skipped` and `test_stage4_golden_snapshot.py` shows `2 passed`. If the golden
snapshot test FAILS or reports "Golden snapshot missing", STOP — the
byte-identical baseline is not seeded and the plan's gate is invalid. Do not
proceed.

**Commit:** none (baseline only).

---

### Task 1 — Failing test: concurrent (W=3) must byte-match serial AND prove the threaded branch actually ran

The existing `test_eora_taskparallel_equivalence` (W=2) already passes against
the **serial** dispatch because today's "parallel" path IS serial. We need a test
that (a) exercises a genuine multi-band split (3 workers × 3 experts) with the
gather assembled out of completion order, (b) asserts **exact** equality
`torch.equal` on CPU (CPU↔CPU is bit-identical), AND (c) **proves the
ThreadPoolExecutor branch ran** — without (c), the band-by-device bug (C1) could
collapse all CPU workers to one band and the test would pass as serial==serial,
shipping the concurrency guarantee untested.

**Observability seam (no monkeypatch — repo rule).** Task 4a makes
`_run_expert_bands` record the number of bands it built into a module-level
read-only diagnostic `_LAST_BAND_COUNT` (an `int`, plus `_LAST_RAN_THREADED:
bool`). Tests read these directly off the module after the run. This is a passive
introspection counter, not a behavior patch.

Add to `tests/test_stage4_multigpu.py`:

```python
from moe_compress.stage4.plugins import eora_compensation as _eora_mod

def test_eora_concurrent_exact_equals_serial_cpu():
    """W=3 concurrent path is BIT-identical to serial on CPU AND really threaded.

    Uses torch.equal (not assert_close): CPU eigh/svd is deterministic, so the
    concurrency engine — which only changes WHEN pure per-expert solves run, not
    WHAT they compute, and assembles rows in ascending-e on the main thread —
    must reproduce the serial bytes exactly. The len(bands)==3 / ran-threaded
    assertions are the C1 guard: ["cpu","cpu","cpu"] MUST yield 3 real bands
    (banded by ordinal w, not device object), proving the ThreadPoolExecutor
    branch executed rather than silently collapsing to inline serial.
    """
    case = _build_case(n_experts=9)          # 9 experts → 3 full bands of 3 under W=3
    fe_serial, rm_s, cp_s = _run_compensate(*case, eora_workers=1)

    fe_par, rm_p, cp_p = _run_compensate(
        *case, eora_workers=3, worker_devices=["cpu", "cpu", "cpu"],
    )
    # C1 guard: the threaded branch actually ran with 3 distinct bands.
    assert _eora_mod._LAST_BAND_COUNT == 3, (
        f"expected 3 bands (1 thread/worker), got {_eora_mod._LAST_BAND_COUNT} — "
        "band-by-device-object bug would collapse CPU workers to 1 band")
    assert _eora_mod._LAST_RAN_THREADED is True

    assert fe_serial.ranks == fe_par.ranks
    assert fe_serial.effective_ranks == fe_par.effective_ranks
    assert rm_s == rm_p and cp_s == cp_p
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            a = getattr(fe_serial, f"{name}_{proj}").data
            b = getattr(fe_par, f"{name}_{proj}").data
            assert torch.equal(a, b), f"{name}_{proj} bytes differ serial vs concurrent"
```

**RED first.** Before Task 4 exists, `_run_expert_bands` /`_LAST_BAND_COUNT`
/`_LAST_RAN_THREADED` are undefined, so this test errors at import/attribute
access — a true failing test. (The byte-equality assertions would pass against
serial, but the band-count + ran-threaded assertions cannot be satisfied until
the real engine lands — this is the regression gate that forces a genuinely
threaded impl.)

```bash
python3 -m pytest tests/test_stage4_multigpu.py::test_eora_concurrent_exact_equals_serial_cpu -q
```
Expected (pre-Task-4): `1 failed` (`AttributeError`/`ImportError` on the
diagnostic / `_run_expert_bands`). Post-Task-4: `1 passed`.

**Commit:** `test(stage4-eora): bit-exact concurrent==serial gate + 3-band ran-threaded proof`

---

### Task 2 — Failing test: gate→up spectrum memo must be correct under genuine threading (H1)

The gate→up memo (`gate_spectra[e]`) is the one cross-pass shared state. Under
threading it is written by e's gate-band thread and read by e's up-band thread.
This test forces gate and up eligibility to differ (so a shared expert is banded
to a different worker ordinal across the two passes — the existing
`_drop_up_originals` seam at `tests/test_stage4_multigpu.py:262-269`), under W=3,
**and asserts the gate pass actually ran threaded** (the H1 ask: this must
exercise the memo under genuine concurrency, not collapsed-to-serial).

**Why this is safe under threading (H1 citation).** The memo tensors are
**read-only to the up-pass consumer**: `_compute_eora_factors` only rebinds local
names and, when a device move is needed, allocates NEW tensors via `.to`
(`eora_compensation.py:321-325`) — it never mutates the stored spectrum in place.
So concurrent up-pass reads of distinct `gate_spectra[e]` entries cannot corrupt
each other or the gate-band writer's tensors.

```python
def test_eora_concurrent_gate_up_memo_skew_exact():
    """Concurrent path with gate-eligible != up-eligible is bit-exact to serial.

    Forces a shared expert onto different worker bands across the gate vs up
    passes (the cross-device gate-spectrum reuse case). The threaded engine must
    (a) build each gate spectrum on its band thread, (b) hand it to the up pass
    via the per-expert memo (read-only; eora_compensation.py:321-325 rebinds
    locals / allocs new tensors, never mutates the memo), (c) let
    _compute_eora_factors relocate it to the up delta's device. Bit-exact on CPU,
    AND genuinely threaded (gate pass yields >1 band)."""
    make_fe, ref_factory, originals, A_cov, config = _build_case(n_experts=9)
    originals_skew = _drop_up_originals(originals, {1, 4, 7})
    fe_s, rm_s, cp_s = _run_compensate(
        make_fe, ref_factory, originals_skew, A_cov, config, eora_workers=1)
    fe_p, rm_p, cp_p = _run_compensate(
        make_fe, ref_factory, originals_skew, A_cov, config,
        eora_workers=3, worker_devices=["cpu", "cpu", "cpu"])
    # H1/C1 guard: the LAST band run (the down_proj pass, 6 experts) ran threaded.
    # The gate pass (9 experts) split into 3 bands too; the memo was exercised
    # across real threads. _LAST_RAN_THREADED reflects the final per-matrix run.
    assert _eora_mod._LAST_RAN_THREADED is True
    assert _eora_mod._LAST_BAND_COUNT >= 2
    assert fe_s.ranks == fe_p.ranks and rm_s == rm_p and cp_s == cp_p
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            assert torch.equal(
                getattr(fe_s, f"{name}_{proj}").data,
                getattr(fe_p, f"{name}_{proj}").data), f"{name}_{proj} skew mismatch"
```

```bash
python3 -m pytest tests/test_stage4_multigpu.py::test_eora_concurrent_gate_up_memo_skew_exact -q
```
Expected (pre-Task-4): `1 failed` (diagnostic undefined). Post-Task-4: `1 passed`.

> Note for the implementer: `_LAST_BAND_COUNT` / `_LAST_RAN_THREADED` reflect the
> **most recent** `_run_expert_bands` call, i.e. the down_proj pass (the last
> matrix in `MATRIX_NAMES`). With 9 experts and `{1,4,7}` up-dropped, down_proj
> has 9 eligible → 3 bands; gate_proj 9 → 3 bands; up_proj 6 → 3 bands. So every
> per-matrix pass is genuinely threaded here.

**Commit:** `test(stage4-eora): gate→up memo bit-exact under genuine threading (H1)`

---

### Task 3 — Failing test: residual-accumulation order is preserved (log_residuals=True)

The residual sums `res_before_acc`/`res_after_acc` are fp adds; their order is
observable (fp add is non-associative). The serial loop adds in ascending-e
(`:714-715`). The concurrent engine must too. Add a test with `log_residuals`
on, asserting the per-matrix residual stats (which feed trackio, not the golden,
but we pin them anyway as a determinism guard) match serial exactly. Since the
residual scalars are not in `rank_map`/`compensated_params`, capture them via the
trackio emit or recompute: simplest is to assert the **rank_map + U/V bytes**
match with `log_residuals=True` (the residual path must not perturb the golden),
AND add a direct unit check that an out-of-order fp sum can differ so the test is
meaningful.

```python
def test_eora_concurrent_log_residuals_bytes_unchanged():
    """log_residuals=True must not perturb U/V/ranks vs log_residuals=False, and
    concurrent==serial under it. Guards that the residual fp accumulation (pulled
    onto the main thread in ascending-e) stays byte-irrelevant to the golden."""
    case = _build_case(n_experts=9)
    cfg_on = {"stage4_eora": {**case[4]["stage4_eora"], "log_residuals": True}}
    case_on = (*case[:4], cfg_on)
    fe_s, rm_s, cp_s = _run_compensate(*case_on, eora_workers=1)
    fe_p, rm_p, cp_p = _run_compensate(
        *case_on, eora_workers=3, worker_devices=["cpu", "cpu", "cpu"])
    assert rm_s == rm_p and cp_s == cp_p and fe_s.ranks == fe_p.ranks
    for name in ("gate_proj", "up_proj", "down_proj"):
        for proj in ("U", "V"):
            assert torch.equal(getattr(fe_s, f"{name}_{proj}").data,
                               getattr(fe_p, f"{name}_{proj}").data)
```

```bash
python3 -m pytest tests/test_stage4_multigpu.py::test_eora_concurrent_log_residuals_bytes_unchanged -q
```
Expected: `1 passed`.

**Commit:** `test(stage4-eora): log_residuals byte-irrelevant + concurrent-stable`

---

### Task 4 — Impl: replace the serial dispatch with the band-threaded engine

This is the core change. Add a private helper `_run_expert_bands(...)` and call
it in place of the `for e in eligible:` block. **Minimal, surgical** — touch ONLY
`eora_compensation.py:693-718`; everything above (`device_of` banding) and below
(`widen_rank`, trackio) is unchanged.

**4a — Add the helper** (module scope, near `_solve_expert_tile`, after
`:467`). The helper takes a list of **(band-ordinal, experts) bands already
grouped by worker index** and a callable that solves one expert (returning the
tile **already gathered to home device**). It runs one thread per band ordinal,
each thread walking its band ascending-e; then the caller assembles in ascending-e
on the main thread.

> **C1 — band by ORDINAL, not device object.** The grouping is done by the band
> ordinal `w`, NOT by `device_of[e]`. The CPU test seam uses
> `["cpu","cpu","cpu"]` and `torch.device("cpu")==torch.device("cpu")`
> (hash-equal); grouping by device would collapse those to ONE band and never run
> the pool. Banding by `w` keeps the 3 workers distinct → 3 threads → faithful
> 3-GPU model on CPU. We pass the pre-grouped `bands` in (built at the call site
> from the existing `w`-indexed `device_of` construction) so the helper never
> re-derives keys from device identity.

```python
# Module-level read-only DIAGNOSTICS for tests (passive introspection — NOT a
# behavior patch / monkeypatch). Reflect the most recent _run_expert_bands call.
_LAST_BAND_COUNT: int = 0
_LAST_RAN_THREADED: bool = False


def _run_expert_bands(
    bands: list[tuple[int, list[int]]],   # [(w, [experts ascending]), ...] — keyed by ORDINAL w, NOT device
    device_of: dict[int, "torch.device"],
    solve_one,                       # (e, tgt) -> (Uc_home, Vc_home, take_eff, rb, ra, spec_out)
    *,
    name: str,
    set_gate_spectrum,               # (e, spectrum_out) -> None   (called inside band thread)
    concurrent: bool,
    home_device,                     # gather target (for the optional residual sync)
    log_residuals: bool = False,
) -> dict:
    """Run the per-expert solves grouped into per-WORKER bands (banded by ordinal).

    Concurrency engine for N-GPU lever 1. Each band ordinal ``w`` gets ONE thread
    that runs its contiguous band in ASCENDING-e order; threads overlap across
    workers (CUDA kernels release the GIL and run async on each device's default
    stream). The gather to the home device happens INSIDE each thread
    (``solve_one`` returns home-resident tiles), so the only main-thread work is
    the deterministic ascending-e assembly the caller does after join.

    Banded by ORDINAL ``w`` (C1): the caller passes ``bands`` pre-grouped by the
    worker index used in ``device_of`` construction, so two workers that happen
    to map to the SAME device object (e.g. the CPU test stand-in
    ``["cpu","cpu","cpu"]``) remain DISTINCT bands → real threads. Grouping by
    ``device_of[e]`` here would silently collapse them and skip the pool.

    Byte-identicality: ``solve_one`` is a pure function of (e, tgt); results are
    keyed by ``e`` into a dict (disjoint keys, no race) and the CALLER reassembles
    in ascending-e — so worker completion order is unobservable, exactly like
    serial. ``concurrent=False`` OR a single band runs inline on the calling
    thread → byte-identical to the legacy serial loop, and the 1-GPU default
    (effective_workers<=1 ⇒ one band) never enters a thread.

    Returns ``band_results: {e: (Uc_home, Vc_home, take_eff, rb, ra)}``.
    """
    global _LAST_BAND_COUNT, _LAST_RAN_THREADED
    import torch as _torch
    from concurrent.futures import ThreadPoolExecutor

    band_results: dict = {}
    _LAST_BAND_COUNT = len(bands)

    def _run_band(experts: list[int]) -> None:
        for e in experts:                    # ascending-e within the band
            tgt = device_of[e]
            Uc_home, Vc_home, take_eff, rb, ra, spec_out = solve_one(e, tgt)
            if name == "gate_proj":
                set_gate_spectrum(e, spec_out)   # disjoint key e — thread-safe under GIL
            band_results[e] = (Uc_home, Vc_home, take_eff, rb, ra)

    if not concurrent or len(bands) <= 1:
        _LAST_RAN_THREADED = False
        for _w, experts in bands:
            _run_band(experts)
        return band_results

    # Threaded: one worker thread per band ordinal. Pin intra-op BLAS to 1 on the
    # main thread (process-global; restored in finally) to avoid (bands x cores)
    # oversubscription on the CPU stand-in path. No-op cost on real GPUs.
    _LAST_RAN_THREADED = True
    prev_threads = _torch.get_num_threads()
    try:
        _torch.set_num_threads(1)
        with ThreadPoolExecutor(max_workers=len(bands)) as pool:
            futures = [pool.submit(_run_band, experts) for _w, experts in bands]
            for f in futures:
                f.result()                   # re-raise any worker exception here
    finally:
        _torch.set_num_threads(prev_threads)

    # 4c — RECOMMENDED guarded residual sync (M1). The residual scalars rb/ra are
    # produced on each WORKER device and later copied to ``home_device`` by the
    # main-thread assembly (rb.to(dev)), which can race the worker's in-flight
    # kernel. Log-only ⇒ a race would silently corrupt a logged number without
    # tripping any byte gate. Sync the distinct worker devices once per matrix,
    # ONLY under log_residuals (default off ⇒ zero cost on the golden path).
    if log_residuals:
        seen = set()
        for e in band_results:
            d = device_of[e]
            if d.type == "cuda" and d not in seen:
                seen.add(d)
                _torch.cuda.synchronize(d)
    return band_results
```

Notes embedded for the implementer:
- `bands` is built at the call site directly from the existing ordinal-indexed
  loop (`eora_compensation.py:687-691`): `bands = [(w, eligible[w*per:(w+1)*per])
  for w in range(effective_workers)]` in the W>1 branch, or `[(0, eligible)]` in
  the `effective_workers<=1` branch. This reuses the already-correct `w`/`per`
  banding and guarantees distinct ordinals.
- `solve_one` is a closure built in `compensate_layer` that wraps
  `_solve_expert_tile` AND the home-device gather (so the gather runs on the
  worker thread, overlapping the next worker's solve). It returns the
  gate-spectrum-out as the 6th element only for the gate pass.
- `set_gate_spectrum` is a tiny closure `lambda e, s: gate_spectra.__setitem__(e, s)`.
  Passing it (rather than mutating `gate_spectra` directly in the helper) keeps
  the helper agnostic of the memo dict's identity and makes the disjoint-key
  contract explicit at the call site.
- The `_LAST_BAND_COUNT` / `_LAST_RAN_THREADED` module globals are passive test
  diagnostics only — no production code reads them; they exist so Tasks 1-2/5 can
  prove the threaded vs inline branch was taken without monkeypatching.

**4b — Rewrite the dispatch block** at `eora_compensation.py:693-718`. Replace:

```python
            for e in eligible:
                tgt = device_of[e]
                W_orig, U_e, V_e, A = _inputs_for(e)
                Uc, Vc, take_eff, res_before, res_after, gate_spec_out = _solve_expert_tile(...)
                if name == "gate_proj":
                    gate_spectra[e] = gate_spec_out
                U_corr[e] = Uc.to(device=dev, dtype=dtype)
                V_corr[e] = Vc.to(device=dev, dtype=dtype)
                eff_per_expert[e] = int(take_eff)
                if log_residuals:
                    res_before_acc += res_before.to(dev)
                    res_after_acc += res_after.to(dev)
                n_eligible += 1
                if (e + 1) % 32 == 0:
                    log.info("  L%d/%s expert %d/%d", ref.layer_idx, name, e + 1, N)
```

with:

```python
            # Concurrency engine (N-GPU lever 1): run each worker's band
            # CONCURRENTLY (one thread per band ORDINAL w — NOT per device object,
            # so the CPU ["cpu","cpu"] seam stays multi-band), gathering each tile
            # to the home device INSIDE its band thread. Assembly into U_corr/
            # V_corr and the residual fp sum stay on THIS thread in ascending-e
            # order, so the output is byte-identical to the serial path regardless
            # of which device finishes first. effective_workers<=1 ⇒ one band ⇒
            # inline (no thread), byte-identical to single-GPU today.
            #
            # Build bands by ORDINAL w, reusing the same w/per split that produced
            # device_of above (eora_compensation.py:687-691). C1: keyed by w, so
            # two workers mapping to the same device object remain distinct bands.
            if effective_workers <= 1:
                bands = [(0, eligible)]
            else:
                bands = [
                    (w, eligible[w * per:(w + 1) * per])
                    for w in range(effective_workers)
                ]
                bands = [(w, ex) for (w, ex) in bands if ex]   # drop empty trailing bands

            def _solve_one(e, tgt):
                W_orig, U_e, V_e, A = _inputs_for(e)
                Uc, Vc, take_eff, rb, ra, spec_out = _solve_expert_tile(
                    name, e, ref.layer_idx, W_orig, U_e, V_e, A, d_in,
                    r_per_expert, tgt, a_storage_dtype,
                    gate_spectrum=gate_spectra.get(e),
                    log_residuals=log_residuals,
                )
                # Gather to home device on the WORKER thread (overlaps next band).
                Uc_home = Uc.to(device=dev, dtype=dtype)
                Vc_home = Vc.to(device=dev, dtype=dtype)
                return Uc_home, Vc_home, int(take_eff), rb, ra, spec_out

            band_results = _run_expert_bands(
                bands, device_of, _solve_one,
                name=name,
                set_gate_spectrum=gate_spectra.__setitem__,
                concurrent=(effective_workers > 1),
                home_device=dev,
                log_residuals=log_residuals,
            )

            # Deterministic ascending-e assembly (the byte-identity guarantee).
            for e in eligible:
                Uc_home, Vc_home, take_eff, rb, ra = band_results[e]
                U_corr[e] = Uc_home
                V_corr[e] = Vc_home
                eff_per_expert[e] = take_eff
                if log_residuals:
                    res_before_acc += rb.to(dev)
                    res_after_acc += ra.to(dev)
                n_eligible += 1
                if (e + 1) % 32 == 0:
                    log.info("  L%d/%s expert %d/%d", ref.layer_idx, name, e + 1, N)
```

**4c — Guarded residual sync** is implemented INSIDE `_run_expert_bands` (see
4a): one `torch.cuda.synchronize(d)` per distinct worker device, gated on
`log_residuals`, after the pool joins and before the helper returns (so it
precedes the main-thread ascending-e `rb.to(dev)` assembly). Default-off, zero
golden impact. (M1.)

**`per` scope note:** `per = (len(eligible) + effective_workers - 1) //
effective_workers` is currently computed only inside the W>1 branch at
`eora_compensation.py:688`. The new `bands` builder reuses `per` in that same W>1
branch, so it stays in scope — keep the `bands` construction immediately after
`device_of` is built (inside the existing `else:` at `:684-691`), or hoist `per`
just above. Do NOT reference `per` in the `effective_workers<=1` branch (it uses
`[(0, eligible)]`).

**Important behavioral preservations (verify each against the diff):**
- `gate_spectra[e]` is written inside `_run_expert_bands` via
  `set_gate_spectrum` BEFORE the up pass runs — and the gate pass for the whole
  matrix joins before `compensate_layer` advances to `name == "up_proj"` (the
  `for name in MATRIX_NAMES:` loop is outer, so the gate matrix fully completes —
  threads joined — before the up matrix begins). So the up pass always sees a
  fully-populated memo. **This ordering is preserved; do not move the memo
  read/write across the matrix-loop boundary.**
- The `n_eligible` counter and the `(e+1) % 32` progress log now run on the main
  thread in ascending-e — same values as serial.
- `res_before`/`res_after` are `None` when `log_residuals=False`; the assembly
  loop only touches them under the flag, identical to today.

**Verify the RED tests from Tasks 1-3 now pass AND the originals stay green:**
```bash
python3 -m pytest tests/test_stage4_multigpu.py -q
```
Expected: `9 passed, 1 skipped` (6 original passing + 3 new; the 1 skip is the
≥2-CUDA spectrum test). If ANY equivalence test flips to FAIL, the concurrency
broke byte-identity — STOP and debug (do NOT regen any golden).

**Commit:** `feat(stage4-eora): concurrency engine — ordinal-banded threads (byte-identical)`

---

### Task 5 — Test: 1-GPU default takes the inline (no-thread) branch (M2)

Guard that `effective_workers<=1` (the 1-GPU default) is a literal no-op — ONE
band, inline, no pool — not a 1-thread pool. Two assertions: (a) a direct unit
test on the helper's inline branch (new ordinal-banded signature), and (b) the
**M2 invariant via the full `compensate_layer` path**: with `eora_workers=1` the
engine must report exactly one band and `_LAST_RAN_THREADED is False`.

```python
def test_run_expert_bands_single_band_inline():
    """A single band runs inline — no pool — and reports not-threaded."""
    from moe_compress.stage4.plugins import eora_compensation as m
    calls = []
    def solve_one(e, tgt):
        calls.append(e)
        return (torch.zeros(2, 2), torch.zeros(2, 2), 1, None, None, None)
    eligible = [0, 1, 2, 3]
    device_of = {e: torch.device("cpu") for e in eligible}
    bands = [(0, eligible)]                       # ONE band ordinal
    res = m._run_expert_bands(
        bands, device_of, solve_one, name="down_proj",
        set_gate_spectrum=(lambda e, s: None),
        concurrent=True,                          # even concurrent=True: 1 band ⇒ inline
        home_device=torch.device("cpu"),
    )
    assert calls == [0, 1, 2, 3]                  # ascending-e, inline
    assert set(res.keys()) == {0, 1, 2, 3}
    assert m._LAST_BAND_COUNT == 1
    assert m._LAST_RAN_THREADED is False


def test_eora_single_worker_is_one_band_not_threaded():
    """M2 invariant: eora_workers=1 ⇒ exactly one band, never the threaded branch."""
    from moe_compress.stage4.plugins import eora_compensation as m
    case = _build_case(n_experts=6)
    fe, rm, cp = _run_compensate(*case, eora_workers=1)
    assert m._LAST_BAND_COUNT == 1
    assert m._LAST_RAN_THREADED is False
```

```bash
python3 -m pytest tests/test_stage4_multigpu.py::test_run_expert_bands_single_band_inline \
                  tests/test_stage4_multigpu.py::test_eora_single_worker_is_one_band_not_threaded -q
```
Expected: `2 passed`.

**Commit:** `test(stage4-eora): 1-GPU default is one-band inline, never threaded (M2)`

---

### Task 6 — GOLDEN GUARDRAIL: byte-identical Stage-4 snapshot, no regen

Run the integer-rank/param byte-identical golden (fp32 + bf16 params) with the
default `eora_workers=1`. This is the hard gate for "1-GPU path unchanged /
byte-identical". **NEVER set `MOE_REGEN_GOLDEN`.**

```bash
cd /home/lucas/ai/wt-stage4/max_quality
python3 -m pytest tests/test_stage4_golden_snapshot.py -v
```
Expected: `test_stage4_eora_ranks_byte_identical[fp32]` and `[bf16]` both PASS.
If either reports "golden snapshot drift detected", the change altered the
default-path bytes — a hard failure; debug, do NOT regen.

> Note on scope of the golden: `eora_ranks.json` pins integer `rank_map` +
> `compensated_params` + the config block (`test_stage4_golden_snapshot.py:20-26`).
> The **float U/V** byte-identity is NOT in this JSON; it is enforced by the
> `torch.equal` equivalence tests (Tasks 1-3). Both gates together cover
> "result byte-identical".

**Commit:** none (verification only) — or if a `.gitkeep`/golden already exists,
no new bytes. If Task 0 found the golden unseeded, that is a pre-existing gap to
raise with the user, NOT something this plan regenerates.

---

### Task 7 — Full Stage-4 + caller suite regression

Run the broader Stage-4 surface to catch seam regressions (orchestrator, resume,
smoke), per the MEMORY lesson "run the FULL suite + whole-impl final review".

```bash
cd /home/lucas/ai/wt-stage4/max_quality
python3 -m pytest tests/ -k "stage4 or eora" -q
```
Expected: all green (modulo the pre-existing ≥2-CUDA skip). Also run any
orchestrator/resume smoke (`tests/test_smoke_stage4_resume.py` if present).

**Commit:** none (verification). If green, the feature branch is ready for the
review/fix loop.

---

### Task 8 (OPTIONAL, separate follow-on) — VRAM-aware banding (the auto-batch analog)

**Flag: out of core scope. Do as a distinct task only if requested.** The
analysis notes banding is a VRAM-oblivious even count split
(`per = ceil(len(eligible)/effective_workers)`, `eora_compensation.py:688`;
`stage4.md:185-194`, `:284-288`). On heterogeneous cards — or when the home card
also holds the resident model — an even split can OOM the fuller card while
others idle. The auto-batch analog is to weight band sizes by each worker's
**free VRAM** (`torch.cuda.mem_get_info(dev)`), placing fewer experts on the
loaded home card.

- **This is a PLACEMENT change (`device_of`), NOT a numerics change** — the
  ascending-e gather still makes it byte-identical, so the same equivalence tests
  (Tasks 1-3) gate it.
- Practical payoff is low (per-expert working sets are small `[d_in,d_in]` fp32),
  so this is a refinement, not a blocker (`stage4.md:192-194`).
- Implementation sketch: replace the even-`per` band construction
  (`:687-691`) with a free-VRAM-weighted assignment that still produces
  **contiguous ascending-e bands** (so determinism holds). Gate behind a config
  knob `multi_gpu.eora_vram_aware_banding` (default False) so the default path is
  untouched and byte-identical.
- **Auto-batch (`utils/auto_batch.py`) is N/A and stays N/A.** Stage 4 has NO
  forward pass and feeds NO sequence batch (`stage4.md:168-194`, `:257-262`):
  `auto_batch.size_batch` sizes a *forward micro-batch of sequences*; there is no
  `resolve_batch` call anywhere in `stage4/` and there must not be. The per-GPU
  resource is the eigh/SVD working set, addressed by Task 8's band sizing — NOT
  by auto-batch. **State this explicitly so no one wires `resolve_batch` here.**

**Commit (if done):** `feat(stage4-eora): opt-in VRAM-aware expert banding (byte-identical placement)`

---

## Out of scope

- **`eora_inputs` / `input_cov_cache` plugins.** Pure disk I/O (`torch.load`
  map_location="cpu"), no GPU linalg; I/O-bound, NOT-WORTH-IT / NOT-APPLICABLE
  for multi-GPU (`stage4.md:45-46`, `:220-240`, `:289`).
- **Auto-batch / `resolve_batch` in Stage 4.** N/A stage-wide — no forward pass,
  no sequence batch (`stage4.md:168-194`). Explicitly must NOT be added.
- **Process-spawn / `torch.multiprocessing`.** Rejected: EoRA is in-process
  linear algebra with no forward pass; spawning would re-load 50+ GB `originals`
  per child (see Architecture).
- **Raw `torch.cuda.Stream` hand-management.** Rejected in favor of the
  thread-per-device primitive (existing reviewed `lsa_pool` discipline);
  re-evaluate only if profiling on real ≥2-GPU shows thread overhead dominates
  (it won't for #devices threads).
- **Changing the EoRA numerics / rank-budget / noise-floor.** This plan is
  purely a *scheduling* change; the kernel (`_compute_eora_factors`,
  `_eigh_spectrum`) and all rank/budget decisions are untouched and remain
  device-independent (`stage4.md:115-122`).
- **Live ≥2-GPU wall-clock validation.** The equivalence/golden gates run on CPU
  (the `eora_worker_devices=["cpu","cpu"]` seam) and prove byte-identity. Real
  multi-GPU **overlap/speedup measurement** is deferred to a hardware run (one
  GPU box at a time per the shared-workstation rule); it validates timing, not
  correctness, so it does not block merge of the correctness-complete change.
- **Modifying `orchestrator._resolve_eora_workers` or the ctx wiring.**
  Unchanged; the concurrency engine consumes the already-resolved
  `eora_workers` / `eora_worker_devices` exactly as today.

---

## Review / fix loop (per project protocol)

After Task 7 green: run the two-stage per-plugin review (paper/spec fidelity
FIRST — confirm zero numeric deviation from the EoRA kernel; code-quality
SECOND), then reviewer→fixer ping-pong addressing ALL five categories
(Critical/High/Medium/Low/Nitpick) until all-none. The supervisor sanity-checks
line cites + any parallel git races. Do a final whole-implementation review +
full caller suite before claiming green (MEMORY: "run the FULL suite + whole-impl
final review").

---

## Open questions

1. **Real-GPU overlap validation timing.** The plan proves byte-identity on CPU;
   the actual N× speedup is unverified until a ≥2-GPU run. Acceptable to merge
   the correctness-complete change and validate timing on the next GPU box, or
   does the user want a GPU run gating merge? (Recommendation: merge on
   correctness gates; timing is a non-correctness follow-up. Per
   "deferred-GPU-validation is NOT a stop point" this should not block.)
2. **Golden seeding state.** Task 0 must confirm `tests/golden/stage4/eora_ranks.{fp32,bf16}.json`
   exist. If they are NOT seeded in this worktree, that is a pre-existing gap —
   raise it; this plan will NOT regenerate goldens (the live ablation needs the
   existing bytes preserved).
3. **Thread count cap.** `effective_workers` = #worker devices (small). No cap
   needed. If a future config allows `eora_workers` > device_count for some
   logical oversubscription, revisit — but `_resolve_eora_workers` already clamps
   to `device_count()` (`orchestrator.py:82-85`), so this cannot happen today.
4. **Guarded CUDA sync under `log_residuals` (RECOMMENDED, M1).** For the U/V
   golden path no extra sync is needed (the per-device gather copy + the
   per-matrix `widen_rank`/`.item()` order the device work). For the **residual
   path only**, the main-thread `rb.to(dev)` (`:714-715`) copies a worker-device
   scalar that may still be in-flight on that worker — a race that would silently
   corrupt a log-only number without tripping any byte gate. The plan therefore
   **RECOMMENDS** (Task 4a/4c) a single `torch.cuda.synchronize(d)` over the
   distinct worker devices, issued after the pool joins and before the ascending-e
   assembly, gated `if log_residuals` — one sync per matrix, default-off, zero
   golden impact. This is implemented inside `_run_expert_bands`, never inside a
   worker thread. Confirm on the first real ≥2-GPU run that the residual stats are
   stable; if profiling shows the guarded sync is unnecessary it may be dropped,
   but it ships ON by default-of-the-flag as cheap insurance.
```
