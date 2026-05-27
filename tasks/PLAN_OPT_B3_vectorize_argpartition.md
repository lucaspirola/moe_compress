# PLAN_OPT_B3 — Vectorize `argpartition` + `cheap_cost[ci]` indexing

**Status**: Ready for implementation
**Implementer deliverables**: (1) code change, (2) test additions to `test_stage2_output_cost.py`.
**Implementer does NOT run pytest.** Supervisor runs gates after both review loops close.

---

## 1. Goal & Spec Citation

Hoist the per-row `np.argpartition(cheap_cost[ci], k_cand-1)[:k_cand]` call out of the outer `for ci in range(n_nc):` loop in `_output_space_cost` and replace it with a single full-matrix call `np.argpartition(cheap_cost, k_cand-1, axis=1)[:, :k_cand]` before the loop.

**Spec**: `SC_FAST_PLAN_V3.md §4-B3` lines 251-263. Saving: ~1 min/row (98k argpartition-of-168 calls collapse to one partition of `(n_NC × n_C)`, ~1 ms). Risk: zero — argpartition returns the same indices in vectorized form modulo unordered partition; downstream picks the same K candidates regardless of order.

Spec-anchored only (no external paper).

---

## 2. Loop-Invariant Analysis (verified by code-reading)

In `_output_space_cost`:
- **`k_cand = min(topk, n_c)`** appears at line 426 and line 437. Both are identical. `topk` is a call parameter; `n_c = len(centroid_ids)` is set at line 342 and never reassigned. **Confirmed loop-invariant.**
- **`cheap_cost`** is a 2D numpy array of shape `(n_nc, n_c)` constructed before the loop. Confirmed read-only throughout the function body — no in-place mutation, no callee receives it for mutation.
- Therefore `np.argpartition(cheap_cost, k_cand-1, axis=1)[:, :k_cand]` is safe to hoist; produces semantically identical K-smallest-indices-per-row.

---

## 3. The Exact Change

**File**: `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py`, function `_output_space_cost`.

### 3a. Hoist k_cand + topk_per_m

After the early-return guard `if n_nc == 0 or n_c == 0: return out` (around line 372), add (place near where `cheap_cost` is finalized):

```python
# B3 hoist: compute the K-smallest centroid columns per non-centroid row
# ONCE before the loop. Loop-invariant: k_cand depends only on (topk, n_c);
# cheap_cost is read-only after construction. Semantically equivalent to
# np.argpartition(cheap_cost[ci], k_cand-1)[:k_cand] per-row.
# See SC_FAST_PLAN_V3.md §4-B3.
k_cand = min(topk, n_c)
topk_per_m = np.argpartition(cheap_cost, k_cand - 1, axis=1)[:, :k_cand]  # (n_nc, k_cand)
```

### 3b. No-route branch (line 426-428 region)

Current:
```python
k_cand = min(topk, n_c)
for cj in np.argpartition(cheap_cost[ci], k_cand - 1)[:k_cand]:
    out[ci, int(cj)] = float(cheap_cost[ci, int(cj)])
```

Replace with:
```python
for cj in topk_per_m[ci]:
    out[ci, int(cj)] = float(cheap_cost[ci, int(cj)])
```

(Remove the `k_cand = ...` recomputation; use the hoisted constant.)

### 3c. Main branch (line 437-438 region)

Current:
```python
k_cand = min(topk, n_c)
top_cj = np.argpartition(cheap_cost[ci], k_cand - 1)[:k_cand]
for cj in top_cj:
```

Replace with:
```python
top_cj = topk_per_m[ci]
for cj in top_cj:
```

(Remove the `k_cand = ...` recomputation; use the hoisted constant. `top_cj` is now a reference to a row of `topk_per_m`.)

---

## 4. Verified Semantic Equivalence

`np.argpartition(M, k-1, axis=1)[:, :k]` returns, for each row, the K-smallest indices (unordered partition). Per-row `np.argpartition(M[i], k-1)[:k]` returns the same K-smallest indices, also unordered. The SET of returned indices is bit-identical for any input with strict ordering; under ties, the ORDER may differ but the SET is the same.

Downstream consumers:
- 3b: iterates `for cj in ...` and writes `out[ci, cj] = float(cheap_cost[ci, cj])` — order-independent (each cj produces an independent write).
- 3c: iterates `for cj in top_cj` and computes a value per cj — order-independent.

Therefore the change is **byte-identical** for any cheap_cost matrix without ties, and **set-identical** under ties. The output cost matrix `out` is byte-identical.

---

## 5. Files to Touch

**Modify** (1 file):
- `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py` — hoist + 2 replacements; net ~5 lines added (hoist block) − 4 lines removed (duplicate k_cand + 2 per-row argpartitions) = +1 line net.

**Modify** (1 file):
- `max_quality/tests/test_stage2_output_cost.py` — append the new test (§6).

**NOT touching**: any other file.

---

## 6. Test Cases

### 6a. New test in `test_stage2_output_cost.py` (append to end of file)

```python
def test_output_cost_topk_hoisting_byte_identical():
    """B3: hoisting np.argpartition out of the per-row loop must produce
    byte-identical output cost matrices. Constructs a synthetic 4-NC × 6-C
    cheap_cost with no ties, asserts that for each row, the set of K
    smallest indices selected matches np.argpartition per-row.

    Per SC_FAST_PLAN_V3.md §4-B3.
    """
    rng = np.random.default_rng(seed=42)
    n_nc, n_c = 4, 6
    k_cand = 3
    cheap_cost = rng.random((n_nc, n_c)).astype(np.float64)
    # Ensure no ties (well-separated random floats)

    # Vectorized form (B3):
    vectorized = np.argpartition(cheap_cost, k_cand - 1, axis=1)[:, :k_cand]
    assert vectorized.shape == (n_nc, k_cand)

    # Per-row form (pre-B3 baseline):
    per_row = np.array([
        np.argpartition(cheap_cost[ci], k_cand - 1)[:k_cand]
        for ci in range(n_nc)
    ])

    # SET equality per row (order may differ for ties; none here, but be defensive):
    for ci in range(n_nc):
        assert set(vectorized[ci].tolist()) == set(per_row[ci].tolist()), (
            f"row {ci}: vectorized {vectorized[ci]} != per-row {per_row[ci]}"
        )
        # Additionally: every selected index is actually one of the K smallest.
        sorted_indices = np.argsort(cheap_cost[ci])[:k_cand]
        assert set(vectorized[ci].tolist()) == set(sorted_indices.tolist()), (
            f"row {ci}: selected indices are not the K smallest"
        )
```

### 6b. Existing tests must remain byte-identical (no modifications)

- `test_stage2_output_cost.py::test_output_cost_matches_independent_recomputation` — full numeric reproduction; will catch any semantic drift.
- `test_stage2_output_cost.py::test_output_cost_hand_checked_scalar` — known scalar values; will catch any change.
- `test_stage2_output_cost.py::test_output_cost_of_identical_experts_is_zero` — symmetry check.
- `test_stage2_plugin_output_space_cost.py` — full file (plugin enable/disable logic).
- `test_stage2_output_space_perm_cache_write.py` — Opt B1's tests (T1-T4); cache-write path unchanged.

---

## 7. Risk Register

**R1 — Ties in `cheap_cost`** (CLOSED): per-row vs full-matrix argpartition can return ties in different orders. Downstream uses indices as a SET (independent writes to `out[ci, cj]`). No order-dependency. Zero risk.

**R2 — Memory** (CLOSED): `topk_per_m` shape `(n_nc, k_cand)` int64. For Qwen3.6-35B: ~178 × 48 × 8 bytes = ~68 KB. Negligible.

**R3 — `cheap_cost` mutability** (CLOSED): verified by reading `_output_space_cost` end-to-end — `cheap_cost` is constructed before the loop and not mutated. Hoist is safe.

---

## 8. Acceptance Gates (SUPERVISOR runs after both review loops close)

- **G1**: `pytest max_quality/tests/test_stage2_output_cost.py::test_output_cost_topk_hoisting_byte_identical -v` — new test green
- **G2**: `pytest max_quality/tests/test_stage2_output_cost.py max_quality/tests/test_stage2_plugin_output_space_cost.py max_quality/tests/test_stage2_output_space_perm_cache_write.py -v` — all green, byte-identical to before
- **G3**: `pytest max_quality/tests/ -q --timeout=600` — full suite (expected: 1514 passed, 13 skipped — baseline 1513 + 1 new test)
- **G4**: commit on `main`, push

---

## 9. Out of Scope

- The `_tentative_merged_weights` function — untouched.
- Cost-matrix semantics — unchanged.
- The inner `_swiglu_forward` calls — unchanged.
- Opts B2 (bf16) and B4 (build_banks hoist) — separate plans.
- Any file outside the two listed in §5.

---

## 10. Workflow Reminder

**Implementer writes**:
1. Code change to `output_space_cost.py` (§3).
2. Test addition to `test_stage2_output_cost.py` (§6a).

**Implementer does NOT run pytest.**

**Supervisor runs gates G1–G4** after BOTH review loops (paper-fidelity + code-quality) close.
