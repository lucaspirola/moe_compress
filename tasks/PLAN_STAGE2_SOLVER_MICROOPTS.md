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
| 7 | Vectorize capped-greedy argmin scan (`solver_greedy.py`) | sub-ms → ~order faster, **tiny absolute** | Low (~15 LOC) | Low — but tie-break must stay strict-`<` lowest-index | **KEEP** (trivial, safe, byte-identical) |
| 8 | OR-Tools bulk arc-add (`solver_mcf.py`) | ~2.69 ms/layer on a **non-default** solver | Low–Med | Low (API already shipped in pinned `ortools>=9.10`) | **SKIP** (value too small to justify the churn; **no new dependency is involved — see correction below**) |

**Honest bottom line:** #7 is a trivial, safe vectorize that is worth doing for
code-clarity-plus-marginal-speed and carries a clean byte-identical guarantee.
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

### Why the vectorize is safe (byte-identical)

`np.argmin` returns the **first** (lowest) index of the minimum on ties —
exactly the same tie policy as the strict-`<` ascending scan. So the inner
"find lowest-cost unassigned child for this centroid" reduces to a masked
`np.argmin` over `cost[:, c_idx]`, with assigned children masked to `+inf`.

Recipe (per centroid column, replacing the inner `while`/`for`):

```python
col = cost[:, c_idx].copy()          # 1-D view of this centroid's costs
col[assigned_mask] = np.inf          # exclude already-assigned children
for _ in range(max_group_cap):
    ch = int(np.argmin(col))         # lowest-index on tie — matches strict-<
    if not np.isfinite(col[ch]):     # all remaining are inf → break (== best_child<0)
        break
    assignment[ch] = c_idx
    col[ch] = np.inf                 # mark assigned for subsequent picks
    assigned_mask[ch] = True         # keep global mask in sync across centroids
```

Where `assigned_mask` is a single `np.zeros(n_children, dtype=bool)` carried
across the centroid loop (replacing the `assigned: set[int]`).

**Three correctness conditions that MUST be checked in implementation + tests:**

1. **Tie → lowest index.** Guaranteed by `np.argmin`'s first-min semantics.
   Equal-cost children must verify byte-identical assignment vs. the current
   loop. Add a dedicated tie-break test: e.g. `cost` column with two equal
   minima at indices `i<j`, assert child `i` is chosen first.
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
   `+inf` (or use `np.nan_to_num(col, nan=np.inf)`) before `argmin` to preserve
   the existing NaN-skip behavior. The MCF path already builds a `finite_mask`
   for exactly this reason; greedy must not silently diverge.

> **Note on the level of vectorization.** The `np.argmin`-per-pick form above
> stays a Python loop over `max_group_cap` picks per centroid but removes the
> inner `O(n_children)` Python scan (the dominant 107K-iter cost). A fully
> argsort-based form (upstream's `O(n log n)` per centroid, docstring line 125)
> is also possible but is **not recommended** here: argsort tie-ordering is
> stable so it can match lowest-index, but it complicates the cross-centroid
> "already assigned" exclusion and raises regression risk for a sub-ms win.
> Prefer the minimal masked-`argmin` form.

### Speedup vs. effort/risk

- **Speedup:** Real but **tiny in absolute terms** — the brief measured the
  whole scan as sub-ms/layer. Removing ~107K Python iterations/layer in favor
  of `max_group_cap` `argmin` calls is a meaningful *relative* win (Python loop
  → C-level reduction) but the absolute Stage-2 wall-clock impact is in the
  noise next to merge/SVD/eval costs.
- **Effort:** Low (~15 LOC swap, plus 1–2 targeted tie/all-inf/NaN tests).
- **Risk:** Low, contingent on the three conditions above. The existing
  bit-identical and saliency-byte-identical tests act as a tripwire; add the
  explicit tie-break + all-inf + NaN tests to make the contract local and
  permanent.

### Recommendation: **KEEP**

Trivial, safe, self-documenting improvement that also lets the stale
"consider pre-sorting if this becomes a bottleneck" inline note (lines 183-184)
be retired. Worth doing primarily for clarity + retiring the O(n) inner scan;
the speed win is a bonus, not the justification.

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
| `max_quality/src/moe_compress/stage2/plugins/solver_greedy.py` | edit (vectorize lines 181-203; retire stale inline note 183-184) | — |
| `max_quality/tests/test_stage2_assignment_v2.py` (or `test_stage2_plugin_solvers.py`) | add tie-break + all-inf + NaN regression tests | — |
| `max_quality/src/moe_compress/stage2/plugins/solver_mcf.py` | — | no change |
| `max_quality/requirements.txt` | — | no change (ortools already pinned) |

Net: **#7 is the only implementation item.** It is a contained,
byte-identical-guaranteed vectorize of the default solver's inner scan plus
three small regression tests. #8 is documented-and-declined.
