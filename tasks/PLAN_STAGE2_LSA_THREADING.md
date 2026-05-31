# PLAN — Stage-2 LSA Threading (workstream A)

**Status:** PLAN (not implementation). Reviewed by plan-reviewer before any code lands.
**Branch:** `plan/stage2-lsa-threading`
**Constraint:** Every change is **byte-identical** to the serial path. The only observable
difference is wall-clock. No golden snapshot, no merged-weight tensor, no assignment, no
cache content may change by a single bit.

---

## 0. Why this is safe to thread (the one load-bearing fact)

`scipy.optimize.linear_sum_assignment` (LSA / Hungarian) is implemented in C
(`_lsap_module`). **scipy ≥ 1.12 releases the GIL** for the duration of the solve; on
scipy < 1.12 (the current host has **1.11.4**) it holds the GIL, so a Python `ThreadPool`
serializes the solves and *regresses* (thread-management overhead on top of serial work).

Measured (agent's experiments, recorded in the workstream brief):

| Item | Loop | scipy≥1.12 speedup | Fires when |
|------|------|--------------------|-----------|
| 1 | merge-time Hungarian (`merging.py` → `permutation_align.py`) | ~12.5× @16 thr | ALL configs (incl. SC, 30pct) |
| 2 | post-cost residual loop (`ream_cost_post.py`) | 6.4× @8 thr, ×(1+em_rounds) | `cost_alignment="post"` |
| 3 | output-space cost Hungarian (`output_space_cost.py`) | (embedded LSA) | SC (`cost_alignment="output"`) |
| 4 | `eigh` in `compute_a_sqrt("full")` (`cov_sqrt.py`) | 9.3× @16 thr | `cost_whitening="full"` |

**Therefore the scipy floor bump (item K) is a HARD PREREQUISITE.** Threading must be
**gated on the installed scipy version at runtime**, not merely on the pin — see §1.

---

## 1. Item K — `scipy>=1.12` floor bump (PREREQUISITE)

### Files touched
1. `max_quality/requirements.txt:31` — `scipy>=1.11.0` → `scipy>=1.12.0`
   (the comment `# Hungarian assignment in REAM` stays; this is the canonical pin).
2. `max_quality/hf_jobs/entrypoint_ablations.py:38` — the **PEP 723 inline metadata**
   has its *own independent* `"scipy>=1.11.0",` → `"scipy>=1.12.0",`.
   **This is the trap.** `requirements.txt` header says it is "mirrored by" the
   entrypoint; the HF Jobs UV resolver reads the PEP 723 block, NOT `requirements.txt`.
   Bumping only `requirements.txt` would leave the HF Jobs path on 1.11 and silently
   fall into the GIL-serialized regression.

### numpy<2 compatibility
scipy 1.12.0 requires `numpy>=1.22.4,<1.29` and builds against numpy 1.x. The current
floor `numpy>=1.26.0` (both files) is compatible; **no numpy change needed**. (scipy
1.12 is the last line that still supports numpy 1.x; scipy 1.13+ also works but 1.12 is
the minimal floor that releases the GIL in LSAP, so pin the floor at exactly 1.12.0.)

### "Verify the DEPLOYED IMAGE actually installs ≥1.12, not just the pin"
The pin is necessary but **not sufficient**:
- **Docker path** (`docker/Dockerfile:55`): `pip install -r requirements.txt -c torch-constraint.txt`.
  The `torch-constraint.txt` constrains torch only — it does not cap scipy. Verify post-build:
  `docker run <image> python -c "import scipy; print(scipy.__version__)"` ⇒ must print ≥1.12.
- **HF Jobs path**: UV resolves the PEP 723 block. Verify in the job log that the resolved
  lockline shows `scipy==1.12.x` (or higher), not 1.11.
- **Runtime self-check (belt-and-suspenders, see §3):** the thread-pool helper reads
  `scipy.__version__` at import and **disables threading on <1.12**, logging a one-shot
  WARNING. So even a stale image is correct (it just runs serial), never wrong.

---

## 2. The four threaded loops — independence & determinism analysis

### Shared invariant for all four
LSA is **deterministic per cost matrix** (same matrix in ⇒ same `col_ind` out; scipy's
solver is not randomized). `eigh` (item 4) is likewise deterministic per input matrix on
CPU LAPACK. Results are **reassembled by index** (each worker writes to `out[ci, cj]` or
returns `(idx, value)`), so **thread completion order never affects the result**. This is
the crux of byte-identicality: we parallelize *independent pure functions of disjoint
inputs* and scatter their outputs by precomputed index.

### Item 1 — merge-time Hungarian (`stage2/merging.py:245-260` → `permutation_align.py:188`)
- **Loop:** inside `_merge_experts_inplace`, the `for w, m in zip(weights, members)` loop
  calls `_permutation_align_to_centroid` (whose tail is the LSA at
  `permutation_align.py:188`) once per non-centroid member, ONLY on cache miss
  (`perm_cache.get((li, centroid, m))` is None).
- **Independence:** each member `m` produces a `perm` that is then *consumed in the same
  iteration* to accumulate `accs[name] += Wm[perm] * w`. The accumulation `accs` is
  cross-iteration mutable state ⇒ **the accumulation cannot be naively threaded**.
- **What IS independent:** the *Hungarian solve itself* (`perm = align(...)`). It is a
  pure function of `(ref_gate, ref_up, gate_m, up_m, ref_act, child_act)`, none of which
  the loop mutates.
- **Design (split phase):**
  1. **Phase A (threaded):** for the cache-miss members, compute all `perm`s in a
     ThreadPool, keyed by member id. Each `perm` is reassembled into a `dict[m] -> perm`
     by member index (deterministic). Pure, no shared mutation.
  2. **Phase B (serial, unchanged):** the existing `for w, m in zip(...)` loop runs as
     today but reads `perm` from the phase-A dict (or `perm_cache`) instead of computing
     it. The accumulation order is **identical to today** (still `zip(weights, members)`),
     so `accs` sums in the same order ⇒ bit-identical float accumulation.
  - Rationale for split rather than threading the whole body: float addition is
    non-associative; reordering `accs[name] += ...` would change the low bits. Keeping
    Phase B serial in the original member order guarantees byte-identicality.
- **perm_cache writes:** Phase A may also `perm_cache.put` the computed perms (matching
  today's implicit behavior where merge-time perms are NOT currently cached — confirm:
  today merging.py only *reads* the cache, it does not put. So Phase A does NOT put;
  it just hands perms to Phase B. **No cache write in item 1** ⇒ no cache-safety concern.)

### Item 2 — post-cost residual loop (`stage2/plugins/ream_cost_post.py:222-314`)
- **Loop:** `for ci in range(n_nc):` (outer, over non-centroid rows) × `for cj in top_cj`
  (inner, K candidate centroids). Body: cache lookup → `_permutation_align_to_centroid`
  (LSA) → `_aligned_whitened_residual` (which itself may trigger item-4 `eigh` via
  `_get_a_sqrt`) → write `out[ci, cj]` and `perm_cache.put((li, c_id, m_id), perm, residual)`.
- **Independence:**
  - Each `ci` owns a distinct non-centroid `m_id = noncentroid_ids[ci]`; m_ids are unique.
  - `out[ci, cj]` writes are to disjoint `(ci, cj)` cells ⇒ no output collision.
  - `perm_cache.put` key is `(li, c_id, m_id)`. Across rows, `m_id` differs ⇒ **keys are
    disjoint between rows.** Within a row, the K `cj`s have distinct `c_id`s ⇒ disjoint
    within-row too. **No two iterations ever write the same cache key.**
  - `perm_cache.get` reads keys `(li, c_id, m_id)` that, on EM round 0, were written by
    the *upstream* `_ream_cost_matrix`/output pass — i.e. reads precede this loop, no
    in-loop writer targets a key another row reads.
  - `a_sqrt_cache` (the `CovSqrtCache` closure) is **read-AND-write shared state** across
    rows (keyed `(li, eid, name, mode)`, shared because many rows reuse the same centroid
    `c_id`'s a_sqrt). This is a genuine concurrent read-modify-write ⇒ needs protection
    (§3, a_sqrt cache lock OR per-thread compute-then-publish).
- **Thread granularity:** thread the **outer `ci` loop** (rows). The inner `cj` loop and
  all per-pair work runs inside the worker. Each worker writes its own `out[ci, :]` slice
  and its own disjoint cache keys.
- **This is the big one:** 464.9s/layer, ×(1+em_refinement_rounds) because the whole
  matrix is rebuilt each EM round. 6.4×@8 thr.

### Item 3 — output-space cost Hungarian (`stage2/plugins/output_space_cost.py:491-531`)
- **Loop:** `for ci in range(n_nc):` × `for cj in top_cj`. The embedded LSA is inside
  `_tentative_merged_weights` (`output_space_cost.py:290` → `permutation_align.py:188`)
  on cache miss. SwiGLU forwards (`_swiglu_forward`) are already on GPU.
- **Independence:** same structure as item 2:
  - distinct `m_id` per row, disjoint `out[ci, cj]` writes.
  - `perm_cache.put((li, centroid_id, child_id), perm, residual=None)` — disjoint keys
    by the same argument as item 2 (row owns child_id=m_id).
- **Subtlety — GPU concurrency:** the worker bodies launch GPU SwiGLU forwards. CUDA is
  thread-safe but kernels on the default stream serialize; the *parallelism win here is
  the CPU-side LSA overlapping with GPU work*, not GPU parallelism. The LSA releases the
  GIL (scipy≥1.12) so while one worker's GPU forward runs, another worker's CPU Hungarian
  proceeds. This is byte-identical: each forward is independent, results scattered by
  index. **Worker `torch.set_num_threads(1)` does NOT throttle CUDA** (it caps CPU intra-op
  threads only) — the GPU forwards keep full device throughput.
- **Thread the outer `ci` loop**, same as item 2.

### Item 4 — `eigh` in `compute_a_sqrt("full")` (`utils/cov_sqrt.py:140-146`)
- **Where the calls originate:** `ream_cost_post._get_a_sqrt` (`ream_cost_post.py:178`)
  calls `compute_a_sqrt(A, mode="full")` which runs `torch.linalg.eigh(A_work)` at
  `cov_sqrt.py:145`. LAPACK `syevd`/`heevd` **releases the GIL** during the
  decomposition.
- **Independence:** each `_get_a_sqrt(eid, name)` is a pure function of the covariance
  matrix `A = cov_acc.covariance[(li, eid, name)]` (read-only). Distinct keys ⇒
  independent. The only shared state is the `a_sqrt_cache` (see item 2 note).
- **CPU, not GPU:** the brief is explicit — **keep `eigh` on CPU.** GPU `eigh` (cuSOLVER)
  produces *different low bits* than CPU LAPACK and would require re-blessing every
  `cost_whitening="full"` golden artifact. CPU threading is byte-identical because it is
  the *same* LAPACK call, just issued from multiple threads, each scattering its result
  by `(eid, name)` key. `A_work = A.to(torch.float32)` already keeps it on whatever device
  `A` lives on; confirm `cov_acc.covariance` tensors are CPU (they are — stored CPU-side
  per the InputCovarianceAccumulator), so no device change is needed.
- **Design:** item 4 is **subsumed by item 2's row-threading.** When the post-cost rows
  run in parallel (item 2), the `eigh` calls inside them already run concurrently and the
  GIL is released during each. We do **not** add a separate ThreadPool around
  `_get_a_sqrt`; that would nest pools and oversubscribe. Item 4 is realized *for free*
  by item 2's threading **provided** the a_sqrt_cache is thread-safe (§3). The only
  item-4-specific action: pre-warming. Optionally, before the row loop, compute all unique
  `(c_id, "gate_proj")` and `(c_id, "down_proj")` a_sqrts in one ThreadPool to populate
  the cache, so rows then hit the cache lock-free. **Recommended:** the pre-warm approach
  (see §3 decision) because it makes the a_sqrt_cache effectively read-only during the
  row loop, eliminating the in-loop lock contention entirely.

---

## 3. Design decisions to settle (the meat the reviewer must sign off)

### D1 — Shared thread-pool helper: WHERE it lives + worker count

**Decision:** add a small module `max_quality/src/moe_compress/utils/lsa_pool.py`
(NOT extend `utils/futures.py`, which is scoped to background I/O drain semantics; the
LSA pool is a compute-parallel map with version-gating and BLAS-throttling — a different
concern). It exposes:

```python
def lsa_threads_enabled() -> bool:
    """True iff scipy>=1.12 (LSA releases the GIL). Cached one-shot. Logs a
    one-time WARNING when disabled so a stale image is visibly serial, not wrong."""

def parallel_map(fn, items, *, max_workers=None):
    """ThreadPool map preserving input order in the returned list. Falls back
    to a serial list-comprehension when lsa_threads_enabled() is False or
    len(items) <= 1. Each worker sets torch.set_num_threads(_INNER) on entry."""
```

- **Worker count default:** `min(8, os.cpu_count())`. The brief's measurement: the
  16-thread plateau is **intra-op BLAS oversubscription**, not LSA saturation. 8 outer
  workers × throttled inner BLAS is the sweet spot for the BLAS-heavy bodies (items 2/3/4
  do cdist + Frobenius + eigh + SwiGLU). A config knob
  `stage2_reap_ream.lsa_threads` (default 8, 0/1 ⇒ serial) lets the SC sweep tune it
  without code change; **the knob must NOT appear in any golden-snapshot key** (it is a
  pure perf knob — confirm it is excluded from the manifest, see §4).
- **`torch.set_num_threads(1)` inside workers:** each worker calls
  `torch.set_num_threads(_INNER)` with `_INNER=1` (or 2) on entry. This prevents the 8
  Python threads from each spawning 16 BLAS threads (8×16=128 ⇒ thrash). **Caveat to
  verify:** `torch.set_num_threads` is **process-global**, not thread-local. Setting it
  from worker threads mutates global state racily and bleeds into the main thread after
  the pool exits. **Correct pattern:** set `torch.set_num_threads(_INNER)` ONCE on the
  main thread before submitting, wrapped in a `try/finally` that **restores the original
  count after the pool joins**. Do NOT set it per-worker. The reviewer must confirm this
  — a per-worker `set_num_threads` is a latent global-state bug.
  - Alternative if 1-2 global threads starve the serial Phase-B accumulation: use the
    `threadpoolctl` library (already a transitive dep via scikit/scipy? — VERIFY; if not
    present, do NOT add a dep, use the global save/restore pattern).

### D2 — perm_cache thread-safety: LOCK vs collect-then-merge

**Analysis (the disjoint-key proof):** in items 2 and 3, every threaded iteration writes
`perm_cache.put(key, ...)` with `key = (li, c_id, m_id)` where the row owns a **unique**
`m_id`. No two rows (no two threads) ever write the same key. So the *logical* contents
are independent of execution order. The ONLY hazard is the **non-atomic dict mutation**:
CPython `dict.__setitem__` can trigger a resize concurrently with another thread's
`__setitem__`/`__getitem__`, which is technically not guaranteed safe across threads even
under the GIL when the GIL is released mid-operation by the C-LSA running in a *different*
thread.

**Decision: collect-then-merge (NOT a lock).** Each worker returns its
`(row_index, out_row_slice, list_of_(key, perm, residual))`. After the pool joins, the
**main thread** serially:
1. scatters `out[ci, :] = out_row_slice` for each row, and
2. replays `perm_cache.put(key, perm, residual)` for every collected entry.

- **Why collect-then-merge wins:**
  - **Byte-identical cache CONTENTS** are guaranteed (disjoint keys ⇒ same final dict
    regardless of order).
  - **Byte-identical INSERTION ORDER:** replay the puts in **row-major order** (ci
    ascending, then cj/candidate order within the row) — *exactly the order the serial
    loop would have inserted them*. This matters because `_PermAlignCache._store` is a
    plain `dict` whose iteration order = insertion order; if any downstream consumer
    iterates the cache (it does not today — all access is keyed `.get`/`.has` — VERIFY in
    review), insertion order would be observable. Replaying in serial order makes it moot.
  - **No lock contention** in the hot path; workers touch only thread-local structures.
  - **No `_PermAlignCache` API change.** The cache stays a plain dict, mutated only by the
    main thread ⇒ the cache class needs NO lock, NO `threading` import, stays byte-identical
    for every other (serial) caller (item 1, em_refine, ream_cost pre-path).
- **Rejected alternative — lock inside `_PermAlignCache.put`/`.get`:** would serialize
  cache access, add a `threading.Lock` to a class used by serial paths too, and still
  needs the insertion-order argument. Collect-then-merge is strictly cleaner and provably
  order-deterministic.

### D3 — a_sqrt_cache (`CovSqrtCache`) thread-safety (item 2 + item 4)

**Decision: pre-warm, then treat as read-only.** Before the threaded `ci` loop in
`_post_alignment_cost`, compute every unique a_sqrt the loop will need —
`{(c_id, "gate_proj"), (c_id, "down_proj") for c_id in centroid_ids}` — via
`parallel_map` (this IS where item-4 eigh threading is realized), populating
`a_sqrt_cache`. During the row loop, `_get_a_sqrt` then only ever **reads** an
already-populated cache (every `c_id` a row touches was pre-warmed) ⇒ no concurrent
mutation ⇒ no lock needed. Confirm the pre-warm set is a superset of what rows request
(it is: rows only ever ask for `c_id ∈ centroid_ids` via `_get_a_sqrt(c_id, ...)`).
- The `compute_a_sqrt`/`eigh` result is a pure function of `A` ⇒ pre-warm vs in-loop
  compute yield bit-identical tensors; only timing differs.
- Edge: the `tentative_active` branch (EM rounds) recomputes against tentative weights but
  **still calls `_get_a_sqrt(c_id, ...)` with the ORIGINAL c_id** (per the comment at
  `ream_cost_post.py:276-282`) ⇒ pre-warm by `centroid_ids` still covers it.

### D4 — determinism confirmation (explicit, per loop)
- **LSA:** deterministic per matrix (scipy `linear_sum_assignment` is not randomized; ties
  broken deterministically by the C implementation). ✔
- **eigh:** deterministic per matrix on a fixed LAPACK build (CPU). Threading issues the
  *same* call from N threads; each is independent. ✔ (GPU eigh would differ — explicitly
  excluded, item 4.)
- **Reassembly:** all four scatter by precomputed index (member id / `(ci,cj)` / cache
  key). No reduction across threads. The one float-accumulation reduction (merge-time
  `accs += `, item 1) stays **serial in original order** (D1/item-1 Phase B). ✔
- **Net:** thread completion order is unobservable in every output. ✔

### D5 — which loops are genuinely independent (confirmed)
| Loop | Cross-iteration mutable state? | Verdict |
|------|-------------------------------|---------|
| item 1 body | YES — `accs` accumulation | split: thread the LSA only, keep accumulation serial |
| item 2 rows | NO (disjoint out cells + disjoint cache keys; a_sqrt_cache pre-warmed) | thread outer `ci` |
| item 3 rows | NO (same as item 2; GPU forwards independent) | thread outer `ci` |
| item 4 eigh | NO (pure fn of A; cache pre-warm) | thread via item-2 pre-warm |

---

## 4. Golden snapshots & byte-identicality verification

### Stage-2 golden artifacts that MUST stay byte-identical
There is **no standalone `golden/stage2/*` file** — Stage 2's merge output is pinned
*transitively* through the Stage 2.5 router-KD snapshot:
- `max_quality/tests/golden/router_kd/compressed_metadata.stage2p5.json`
- `max_quality/tests/golden/router_kd/loss_trace.stage2p5.json`
  guarded by `max_quality/tests/test_router_kd_golden_snapshot.py` (the stage2p5 case)
  and the smoke `max_quality/tests/test_smoke_stage2_to_stage2p5.py`.

### Stage-2 unit/integration tests that pin merge artifacts & cost matrices (regression gate)
These must pass unchanged after threading (they encode the per-pair / per-matrix math):
- `test_stage2_merge.py`, `test_stage2_merging.py` — `_merge_experts_inplace` (item 1)
- `test_stage2_plugin_layer_merge.py`, `test_stage2_plugin_mergemoe_step.py`,
  `test_stage2_plugin_regmean_merge.py` — merge step variants (item 1 paths)
- `test_stage2_plugin_ream_cost_post.py`, `test_stage2_vec_cost_matrix.py` — post-cost (item 2)
- `test_stage2_plugin_output_space_cost.py`, `test_stage2_output_cost.py`,
  `test_stage2_output_space_perm_cache_write.py` — output-space + perm_cache writes (item 3)
- `test_cov_sqrt.py` — `compute_a_sqrt`/`eigh` (item 4)
- `test_stage2_plugin_em_refine.py` — EM round path that re-invokes item-2 ×(1+rounds)
- `test_stage2_assignment_v2.py`, `test_stage2_pipeline_run_layer.py` — end-to-end assignment
- `test_smoke_stage2_resume.py` — resume determinism

### New test to ADD (threaded == serial)
`max_quality/tests/test_stage2_lsa_threading_parity.py`:
1. Build a small synthetic MoE layer (reuse the fixtures from
   `test_stage2_merging.py` / `test_stage2_plugin_ream_cost_post.py`).
2. For each of items 1/2/3/4: run the path with `lsa_threads=1` (serial) and
   `lsa_threads=8`, assert **`torch.equal`** (bit-exact) on:
   - merged weight tensors (item 1),
   - the full cost matrix `out` (items 2/3),
   - the a_sqrt tensors (item 4),
   - and **`perm_cache._store` equality** including key insertion order
     (`list(serial._store.items()) == list(threaded._store.items())`).
3. Force-exercise the version gate: monkeypatch is PROHIBITED (per repo policy) — instead
   inject the version decision through the `lsa_threads` config knob / a public
   `lsa_threads_enabled` override arg so both branches run without patching scipy.
4. Run this test on a CI box with `OMP_NUM_THREADS`/host scipy whatever — parity must hold
   even on scipy<1.12 (there the threaded path *is* the serial fallback ⇒ trivially equal).

### Verification gate before merge (not part of this plan's code, but the plan mandates it)
- `pytest max_quality/tests -k "stage2 or cov_sqrt or router_kd_golden"` green.
- Golden snapshots byte-identical (no `--bless`).
- A real-layer A/B on the deployed image: dump the merged `down_proj` / cost matrix for one
  layer serial vs threaded, `cmp`/`torch.equal` ⇒ identical; record wall-clock to confirm
  the speedup actually materialized (proves scipy≥1.12 landed in the image, per §1).

---

## 5. Config knob (perf-only, excluded from determinism keys)
- `stage2_reap_ream.lsa_threads: int = 8` (0 or 1 ⇒ serial; >1 ⇒ ThreadPool workers).
- **MUST be excluded** from any golden/manifest hash (it is a pure perf knob; including it
  would make snapshots depend on host core count). Review item: grep the stage2 cov/profile
  manifest writers (`test_stage2_cov_manifest.py`, `stage2/shared_io.py`) to confirm the
  config-hash allowlist does not pick up `lsa_threads`.

---

## 6. Files touched (summary)

**Source (behavior = parallelism only, byte-identical):**
1. `max_quality/requirements.txt` — scipy floor 1.11→1.12 (item K).
2. `max_quality/hf_jobs/entrypoint_ablations.py` — PEP 723 scipy floor 1.11→1.12 (item K).
3. `max_quality/src/moe_compress/utils/lsa_pool.py` — **NEW** version-gated pool helper (D1).
4. `max_quality/src/moe_compress/stage2/merging.py` — item 1: split-phase perm precompute.
5. `max_quality/src/moe_compress/stage2/plugins/ream_cost_post.py` — item 2 (rows) + item 4
   (a_sqrt pre-warm), collect-then-merge cache replay (D2/D3).
6. `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py` — item 3 (rows),
   collect-then-merge cache replay (D2).
7. `max_quality/src/moe_compress/stage2/orchestrator.py` — thread the `lsa_threads` config
   knob through to the cost/merge plugins (read at `s2.get(...)`, default 8).

**No change** to `permutation_align.py` (`_PermAlignCache` stays a plain, lock-free dict —
mutated only by the main thread under collect-then-merge), `cov_sqrt.py` (eigh unchanged;
threaded by the caller's pre-warm), or `em_refine.py` (it calls the now-threaded
`_post_alignment_cost`/`_ream_cost_matrix` and gets the speedup transitively).

**Tests:**
8. `max_quality/tests/test_stage2_lsa_threading_parity.py` — **NEW** threaded==serial bit-exact.

---

## 7. Open questions for the plan-reviewer
1. `torch.set_num_threads` is process-global — confirm the save/restore-around-pool pattern
   (D1) over any per-worker setter, and decide `_INNER ∈ {1,2}`.
2. Confirm no current consumer iterates `_PermAlignCache._store` (only `.get`/`.has`/`.put`
   today) — if true, insertion-order replay (D2) is belt-and-suspenders; if false, it is
   load-bearing.
3. Confirm `cov_acc.covariance` tensors are CPU-resident so item-4 eigh stays on CPU LAPACK
   (the brief asserts this; verify in `InputCovarianceAccumulator`).
4. Decide whether item 3's GPU SwiGLU forwards benefit enough from CPU-LSA overlap to be
   worth threading at all on the SC config, or whether CUDA-stream serialization caps the
   gain (the brief lists item 3 without a measured multiplier — flag for a quick measure).
5. Confirm `lsa_threads` is excluded from the stage2 config-hash/manifest allowlist (§5).
