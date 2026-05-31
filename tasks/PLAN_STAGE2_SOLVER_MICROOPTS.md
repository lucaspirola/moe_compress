# Plan — Stage-2 Solver Micro-Optimizations (Workstream C)

**Branch:** `plan/stage2-solver-microopts`
**Status:** PLAN ONLY — not an implementation.
**Scope:** Two micro-optimizations in the Stage-2 assignment solvers, both
independently **measured NEGLIGIBLE**. This doc assesses real speedup vs.
effort/risk, byte-identical correctness, and gives a **KEEP / SKIP** call per
item.

All code references below are read from `origin/main` blobs (the local worktree
tree is stale on `fix/svc-audit-script`). Canonical paths:

- `max_quality/src/moe_compress/stage2/plugins/solver_greedy.py`
- `max_quality/src/moe_compress/stage2/plugins/solver_mcf.py`
- `max_quality/requirements.txt`

> **Path note:** the workstream brief cites paths as
> `stage2/plugins/solver_*.py`. On `origin/main` these live under
> `max_quality/src/moe_compress/stage2/plugins/`. Line numbers in the brief
> (greedy `181-203`, mcf `194-210`) match `origin/main` exactly — verified
> below.

---

## TL;DR recommendations

| # | Item | Speedup | Effort | Risk | Recommendation |
|---|------|---------|--------|------|----------------|
| 7 | Vectorize capped-greedy argmin scan (`solver_greedy.py`) | sub-ms → ~order faster, **tiny absolute** | Low (~15 LOC) | Low — but tie-break must stay strict-`<` lowest-index **and** NaN-sanitize must not clobber real `+inf` | **KEEP** (trivial, safe; byte-identity test-gated) |
| 8 | OR-Tools bulk arc-add (`solver_mcf.py`) | ~2.69 ms/layer on a **non-default** solver | Low–Med | Low (API already shipped in pinned `ortools>=9.10`) | **SKIP** (value too small to justify the churn; **no new dependency is involved — see correction below**) |

**Honest bottom line:** #7 is a trivial, safe vectorize that is worth doing for
code-clarity-plus-marginal-speed; its byte-identical contract is **achievable
but must be test-gated** (the obvious-looking NaN-sanitize one-liner has a
correctness trap — see Item #7), not assumed proven on inspection.
#8's premise in the brief ("adds an OR-Tools dependency") is **factually wrong**
— OR-Tools is already pinned (`requirements.txt:32`) — but even with the
dependency cost removed, ~2.69 ms/layer on an opt-in, rarely-used solver does
not justify touching a correctness-sensitive integer-MCF path. **SKIP #8.**

---

## Item #7 — Capped-greedy argmin vectorization (DEFAULT solver)

### Current code (`solver_greedy.py`, `origin/main` lines 181-203)

```python
for c_idx in range(n_centroids):
    absorbed = 0
    # O(n_children) scan per fill slot — pathological for large expert counts;
    # consider pre-sorting by cost if this becomes a bottleneck.
    while absorbed < max_group_cap:
        best_child = -1
        best_cost = float("inf")
        for ch in range(n_children):
            if ch in assigned:
                continue
            if cost[ch, c_idx] < best_cost:
                best_cost = cost[ch, c_idx]
                best_child = ch
        if best_child < 0:
            break
        assignment[best_child] = c_idx
        assigned.add(best_child)
        absorbed += 1
```

- Complexity: `O(n_children · n_centroids · max_group_cap)` Python-level
  iteration (docstring lines 122-127). The brief's ~107K iter/layer figure is
  consistent with this nested loop at project expert counts.
- This is the **DEFAULT** solver (`solver_dispatch.py:59` maps `"greedy"` →
  `_assign_greedy`; `solver=` defaults to `"greedy"`), so it runs on every
  Stage-2 layer in the standard recipe.

### The load-bearing invariant — tie-break = strict-`<`, lowest child index

The inner scan uses `if cost[ch, c_idx] < best_cost` (**strict** `<`) over
`ch` iterated in **ascending** order. Consequence: when two unassigned children
have equal cost to a centroid, the **lowest child index wins** (the first one
seen establishes `best_cost`; a later equal one does not displace it).

This is the byte-identical contract enforced by:

- `test_stage2_assignment_v2.py::test_dispatcher_default_solver_is_greedy_and_bit_identical`
  (the v1→v2 compatibility invariant — default dispatch must equal
  `_assign_greedy` bit-for-bit), and
- `test_stage2_assignment_v2.py:1236` (saliency mode must produce
  byte-identical merged output).

**Any vectorization MUST reproduce lowest-index-on-tie.**

### Why the vectorize is *expected* byte-identical (to be PROVEN by tests, not asserted)

`np.argmin` returns the **first** (lowest) index of the minimum on ties —
exactly the same tie policy as the strict-`<` ascending scan. So the inner
"find lowest-cost unassigned child for this centroid" reduces to a masked
`np.argmin` over `cost[:, c_idx]`, with assigned children masked to `+inf`.

The recipe below is the *intended* equivalent, not a proven one: it has the
right tie/inf/NaN handling **only if implemented exactly as written** (the bare
`np.nan_to_num(col, nan=np.inf)` variant, for instance, is NOT equivalent — see
the NaN-sanitize warning under the recipe). The byte-identical contract MUST be
**established by the regression tests mandated below** (asserting against
`_assign_greedy`'s own output) before merge — do not treat the recipe as
"proven byte-identical" on inspection alone. An equality harness over random
ties+inf+NaN matrices is an acceptable alternative gate.

Recipe (per centroid column, replacing the inner `while`/`for`):

```python
col = cost[:, c_idx].astype(float).copy()  # 1-D copy of this centroid's costs
col[np.isnan(col)] = np.inf          # NaN → +inf so argmin never picks a NaN slot;
                                     # MUST NOT touch real +inf (see condition (3))
col[assigned_mask] = np.inf          # exclude already-assigned children
for _ in range(max_group_cap):
    ch = int(np.argmin(col))         # lowest-index on tie — matches strict-<
    if not np.isfinite(col[ch]):     # all remaining are inf → break (== best_child<0)
        break
    assignment[ch] = c_idx
    col[ch] = np.inf                 # mark assigned for subsequent picks
    assigned_mask[ch] = True         # keep global mask in sync across centroids
```

> **NaN-sanitize must preserve real `+inf` — do NOT use bare
> `np.nan_to_num(col, nan=np.inf)`.** Its default `posinf=` rewrites genuine
> `+inf` → `1.7976931348623157e308` (**finite**), so the
> `if not np.isfinite(col[ch]): break` guard never fires and infeasible
> (inf-cost) children get wrongly assigned — silently breaking the all-inf /
> partial-inf orphan-promotion path (this plan's own highest-risk case). Use
> `col[np.isnan(col)] = np.inf` (shown above), or — if `np.nan_to_num` is
> preferred for style — pass `posinf=` explicitly:
> `np.nan_to_num(col, nan=np.inf, posinf=np.inf)`. Both forms are equivalent
> for this loop; the bare-default form is a correctness bug.

Where `assigned_mask` is a single `np.zeros(n_children, dtype=bool)` carried
across the centroid loop (replacing the `assigned: set[int]`).

**Three correctness conditions that MUST be checked in implementation + tests:**

1. **Tie → lowest index.** Guaranteed by `np.argmin`'s first-min semantics.
   Equal-cost children must verify byte-identical assignment vs. the current
   loop (mandated test case (a) below).
2. **All-inf / exhausted → break.** When every unassigned child for a centroid
   is `+inf`, the old code sets `best_child = -1` and breaks. The vectorized
   form must treat `np.argmin` returning an `inf`-valued slot as the break
   condition (`if not np.isfinite(col[ch]): break`) — otherwise it would
   wrongly assign an infeasible child. This is the single most likely
   correctness regression; it preserves the orphan-promotion contract
   (`solver_greedy.py:194-200`).
3. **NaN cost.** The current `<` comparison is `False` for NaN, so NaN-cost
   children are never selected (effectively skipped). `np.argmin` does **not**
   skip NaN — it can return a NaN index. Implementation MUST sanitize NaN to
   `+inf` via `col[np.isnan(col)] = np.inf` (or
   `np.nan_to_num(col, nan=np.inf, posinf=np.inf)` — note the explicit
   `posinf=np.inf`; the bare `np.nan_to_num(col, nan=np.inf)` is a bug because
   its default `posinf` rewrites real `+inf` to a finite value and defeats the
   all-inf break) before `argmin`, to preserve the existing NaN-skip behavior.
   The MCF path already builds a `finite_mask` for exactly this reason; greedy
   must not silently diverge.

> **Note on the level of vectorization.** The `np.argmin`-per-pick form above
> stays a Python loop over `max_group_cap` picks per centroid but removes the
> inner `O(n_children)` Python scan (the dominant 107K-iter cost). A fully
> argsort-based form (upstream's `O(n log n)` per centroid, docstring line 125)
> is also possible but is **not recommended** here: argsort tie-ordering is
> stable so it can match lowest-index, but it complicates the cross-centroid
> "already assigned" exclusion and raises regression risk for a sub-ms win.
> Prefer the minimal masked-`argmin` form.

### Mandated regression tests (the load-bearing cases)

The byte-identical contract is established by these tests, NOT by inspection.
The load-bearing cases are **all-inf-then-finite** (orphan promotion) and
**partial-inf at cap ≥ 2** — exactly the cases the obvious NaN-sanitize bug (C1)
would silently break — not a single all-inf column. Every case MUST assert the
vectorized output against **`_assign_greedy`'s own output** on the same matrix
(`assert vectorized(...) == _assign_greedy(...)`), so the oracle is the real
helper, never a hand-written expectation that could itself encode the bug:

- **(a) Tie → lowest index.** A `cost` column with two equal minima at indices
  `i < j` (cap ≥ 1): assert child `i` is chosen before `j`.
- **(b) Fully-inf column then finite column.** Column `c0` all `+inf`, column
  `c1` finite, cap ≥ 1: the all-inf centroid must absorb nothing (orphan
  promotion left to the caller), and the finite centroid must fill normally.
  This is the case the bare-`np.nan_to_num` bug regresses.
- **(c) Partial-inf column at cap ≥ 2.** A column with a mix of finite and
  `+inf` children, cap ≥ 2: the centroid absorbs the finite children up to cap
  and breaks before pulling an `+inf` slot.
- **(d) NaN + finite at cap ≥ 2.** A column mixing NaN and finite costs, cap ≥
  2: NaN children must be skipped (never selected), matching the old `<`-is-
  `False`-for-NaN behavior, while finite children fill up to cap.

> **The existing `test_dispatcher_default_solver_is_greedy_and_bit_identical`
> (`test_stage2_assignment_v2.py:49`) CANNOT catch a vectorize regression.** It
> uses `rng.random((6, 3))` — continuous floats with **no ties, no `+inf`, no
> NaN** — and compares the dispatcher against `_assign_greedy` (both the *same*
> pre-vectorize code). Cases (a)–(d) above are what close the actual gap; add
> them, do not rely on `:49`. An equality harness sweeping random
> ties+inf+NaN matrices against `_assign_greedy` is an acceptable complement.

### Speedup vs. effort/risk

- **Speedup:** Real but **tiny in absolute terms** — the brief measured the
  whole scan as sub-ms/layer. Removing ~107K Python iterations/layer in favor
  of `max_group_cap` `argmin` calls is a meaningful *relative* win (Python loop
  → C-level reduction) but the absolute Stage-2 wall-clock impact is in the
  noise next to merge/SVD/eval costs.
- **Effort:** Low (~15 LOC swap, plus the four mandated regression tests
  (a)–(d) above).
- **Risk:** Low, contingent on the three conditions above **and on the tests
  actually proving them** — the `:49` bit-identical test does NOT cover ties /
  inf / NaN, so it is not a tripwire for this change (see the boxed note under
  "Mandated regression tests"). The saliency-byte-identical and Stage-2 golden
  suites are backstops, but the local contract is established by cases (a)–(d).

### Docstring + inline-comment updates required by the vectorize

The vectorize changes the algorithmic shape, so two pieces of in-file prose go
stale and MUST be updated in the same commit (else the docstring lies):

- **Complexity docstring (`solver_greedy.py:122-127`).** It currently reads
  `O(n_children · n_centroids · max_group_cap)` *Python-level* "linear scan per
  fill slot". After the vectorize the inner `O(n_children)` Python scan is gone:
  each pick is a single C-level `np.argmin` over the column, so the Python-level
  cost drops to `O(n_centroids · max_group_cap)` `argmin` calls (each `argmin`
  is `O(n_children)` in C). Rewrite the Complexity block to state the masked-
  `argmin`-per-pick form and stop claiming the Python triple-nested cost.
- **Stale inline note (`solver_greedy.py:183-184`).** The
  `# O(n_children) scan per fill slot … consider pre-sorting by cost if this
  becomes a bottleneck` comment describes the scan being removed and a
  pre-sorting idea that the vectorize supersedes. Retire it (the docstring
  references it as "see the inline note on pre-sorting" — drop that phrase too).

### Recommendation: **KEEP**

Trivial, safe, self-documenting improvement. Worth doing primarily for clarity
plus retiring the `O(n)` inner scan (and the now-stale Complexity docstring /
pre-sorting note above); the speed win is a bonus, not the justification.

---

## Item #8 — OR-Tools bulk arc-add for MCF (opt-in solver)

### Current code (`solver_mcf.py`, `origin/main` lines 194-210)

```python
# Source → child arcs
for i in range(n_children):
    smcf.add_arc_with_capacity_and_unit_cost(SRC, 1 + i, 1, 0)

# Child → centroid arcs (skip +∞ / NaN)
for i in range(n_children):
    for j in range(n_centroids):
        if not finite_mask[i, j]:
            continue
        smcf.add_arc_with_capacity_and_unit_cost(
            1 + i, 1 + n_children + j, 1, int_cost[i, j],
        )

# Centroid → sink arcs
for j in range(n_centroids):
    smcf.add_arc_with_capacity_and_unit_cost(
        1 + n_children + j, SINK, max_group_cap, 0,
    )
```

The proposed optimization: replace the Python double-loop with OR-Tools' bulk
`add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities,
unit_costs)` vectorized (numpy) API.

### CORRECTION to the brief's premise — no new dependency

The brief states #8 "**adds an OR-Tools dependency**." **This is incorrect.**
OR-Tools is **already a pinned, first-class dependency**:

```
max_quality/requirements.txt:32: ortools>=9.10   # SimpleMinCostFlow for Stage 2 v2 capacitated assignment
```

`solver_mcf.py` already imports and uses `SimpleMinCostFlow` unconditionally on
the MCF path. The bulk arc-add API
(`add_arcs_with_capacity_and_unit_cost`, numpy-array signature) ships in the
same `ortools.graph.python.min_cost_flow.SimpleMinCostFlow` class and has been
available since the 9.x line — i.e. it is covered by the existing `>=9.10` pin.
So **#8 requires no dependency change and no `requirements.txt` edit.** The
"dependency cost" the brief asks me to weigh does not exist.

(Sandbox note: `ortools` is not importable in this planning environment, so the
exact pinned-version API surface was not executed here. Implementation MUST
confirm `add_arcs_with_capacity_and_unit_cost` exists on the resolved
`ortools>=9.10` build before relying on it; if a pre-9.x-numpy build is ever
resolved, fall back to the loop. This is a one-line `hasattr` guard, not a new
dep.)

### Why SKIP anyway

Even with the dependency objection removed, the value does not justify the
change:

1. **Non-default, rarely-exercised path.** `mcf` is opt-in
   (`assignment_solver="mcf"`, or `"auto"` falling through when
   `n_NC > N'_l`). The default recipe never runs it. The 2.69 ms/layer is paid
   only when a user explicitly selects MCF.
2. **Absolute cost is negligible.** ~2.69 ms/layer is in the noise vs. the MCF
   `solve()` call itself and the surrounding merge/SVD work; bulk arc-add would
   shave a fraction of that 2.69 ms, not eliminate a bottleneck.
3. **Correctness-sensitive surface for ~nil gain.** The arc-add loop encodes
   the node-id layout (`SRC`, child `1+i`, centroid `1+n_children+j`, `SINK`),
   per-arc capacities (1 for child arcs, `max_group_cap` for centroid→sink),
   and the `finite_mask` skip of `+∞`/NaN entries. Reproducing all of that via
   bulk numpy arrays (building filtered start/end/cap/cost arrays from
   `finite_mask`) is a non-trivial rewrite of a path whose integer-optimal
   output must stay exactly correct. Risk/reward is poor.
4. **No standing pressure.** Both items are explicitly measured negligible and
   MCF is not on the hot path; there is no profiling signal demanding this.

### Recommendation: **SKIP**

Not worth the churn. The honest framing: the brief's stated blocker (new
dependency) is moot, but the remaining cost/benefit — a sub-3 ms shave on an
opt-in solver, against editing a correctness-critical integer-MCF graph build —
still lands on SKIP. If MCF ever becomes the default or shows up hot in a real
profile, revisit; until then, leave it.

If a future maintainer *does* implement it, the minimal-risk shape is: build
`int_cost`/`finite_mask` as today, then assemble flat numpy `start/end/cap/cost`
arrays (source arcs + masked child→centroid arcs + sink arcs) and make a single
`add_arcs_with_capacity_and_unit_cost` call guarded by a `hasattr` check, with a
byte-identical assignment test vs. the current loop on the existing MCF
fixtures (`test_stage2_plugin_solvers.py:92`, `:187`).

---

## Golden-snapshot impact

- **#7 (KEEP):** **Zero** intended golden-snapshot change. Greedy is the
  default solver and feeds `compressed_metadata.stage2p5.json` /
  `loss_trace.stage2p5.json` (`tests/golden/router_kd/`). The vectorization is
  asserted byte-identical, so these snapshots **must remain byte-identical**;
  any diff is a regression, not an expected re-baseline. The implementation
  gate is: run the Stage-2 golden suite and confirm **no** snapshot delta.
  Treat a snapshot diff as a hard failure of the byte-identical contract
  (most likely the all-inf-break or NaN-skip condition was missed).
- **#8 (SKIP):** No code change ⇒ no snapshot impact. (Had it been kept: MCF is
  not exercised by the default golden recipe, so MCF arc-add would not touch
  `stage2p5` snapshots either — but it is SKIP regardless.)

---

## Files touched (if recommendations followed)

| File | #7 KEEP | #8 SKIP |
|------|---------|---------|
| `max_quality/src/moe_compress/stage2/plugins/solver_greedy.py` | edit (vectorize lines 181-203; update Complexity docstring 122-127; retire stale inline note 183-184) | — |
| `max_quality/tests/test_stage2_assignment_v2.py` (or `test_stage2_plugin_solvers.py`) | add mandated regression tests (a) tie, (b) all-inf-then-finite, (c) partial-inf cap≥2, (d) NaN+finite cap≥2 — each asserting vs. `_assign_greedy` output | — |
| `max_quality/src/moe_compress/stage2/plugins/solver_mcf.py` | — | no change |
| `max_quality/requirements.txt` | — | no change (ortools already pinned) |

Net: **#7 is the only implementation item.** It is a contained vectorize of the
default solver's inner scan whose byte-identical contract is **gated by the
mandated regression tests below** (not proven by inspection — the NaN-sanitize
line has a known trap). #8 is documented-and-declined.
