# PLAN_OPT_B1 — Write the perm cache from the output path (free)

**Status**: Ready for implementation
**Implementer deliverables**: (1) the code change, (2) the test file.
**Implementer does NOT run pytest.** Supervisor runs gates G1–G5 after both review loops close.

---

## 1. Goal & Spec Citation

This change adds a single logical write to `_PermAlignCache` inside `_tentative_merged_weights` in `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py`. The write persists the Hungarian permutation computed during the output-space cost-matrix pass so that the eventual merge step (`_merge_experts_inplace` via `merging.py:140`) and the EM-refine pass (`em_refine.py:190`) can reuse it instead of re-running `scipy.optimize.linear_sum_assignment` (LAP) for the same expert pair.

**Spec anchor**: `SC_FAST_PLAN_V3.md §4-B1` lines 211–229. Risk: zero. Saving: ~1 min/row (final merge step only).

**Symmetric reference**: `ream_cost_post.py:285`:
```python
if perm_cache is not None and not tentative_active:
    perm_cache.put(cache_key, perm, residual)
```
The post path stores a real `residual` float. The output path stores `residual=None` because it doesn't compute a whitened Frobenius residual. `_PermAlignCache.put` is typed `(key, perm, residual: float | None)` (`permutation_align.py:48`).

Spec-anchored only — no external paper.

---

## 2. The Exact Change

**File**: `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py`
**Function**: `_tentative_merged_weights` (lines 212–273 post-Opt-C)

**Insertion**: 4 lines inside the `else:` branch (cache-miss path), after the closing `)` of `_permutation_align_to_centroid`, before `perm_t = torch.as_tensor(...)`.

```python
        cached = perm_cache.get((li, centroid_id, child_id)) if perm_cache is not None else None
        if cached is not None:
            perm = cached[0]
        else:
            ref_act   = ream_acc.get_neuron_mean(li, centroid_id) if ream_acc else None
            child_act = ream_acc.get_neuron_mean(li, child_id) if ream_acc else None
            perm = _permutation_align_to_centroid(
                ref_gate, ref_up, child_gate, child_up,
                ref_act_mean=ref_act, child_act_mean=child_act,
            )
            # NEW (B1): persist the freshly-computed permutation so the eventual
            # merge step reuses it instead of re-running LAP for the same pair.
            # Mirrors ream_cost_post.py:285. residual=None because the output
            # path does not compute a whitened Frobenius residual.
            if perm_cache is not None:
                perm_cache.put((li, centroid_id, child_id), perm, residual=None)
        perm_t = torch.as_tensor(perm, dtype=torch.long, device=ref_gate.device)
```

The write is reachable ONLY on cache-miss. On cache-hit (`if cached is not None: perm = cached[0]`), no write occurs — the perm is already in the cache.

**No `tentative_active` guard.** `_tentative_merged_weights` is called from `_output_space_cost` which has no EM-round semantics; every call uses the original centroid bank weights. The cached perm is always valid for the merge step.

---

## 3. Verified Symmetric Pattern

- `ream_cost_post.py:285`: `perm_cache.put(cache_key, perm, residual)` where `cache_key = (li, c_id, m_id)` — same tuple structure as ours.
- `permutation_align.py:48`: `def put(self, key, perm: np.ndarray, residual: float | None) -> None`. `None` accepted; cache docstring at lines 34–36 explicitly documents `None` as a valid residual for paths that compute only the permutation.

---

## 4. Files to Touch

**Modify** (1 file):
- `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py` — add the 4-line block in §2.

**Create** (1 file):
- `max_quality/tests/test_stage2_output_space_perm_cache_write.py` — see §6.

**NOT touching** (downstream consumers, not modification targets):
- `max_quality/src/moe_compress/stage2/merging.py:140` — reads the cache; benefits from the write.
- `max_quality/src/moe_compress/stage2/plugins/em_refine.py:190` — reads the cache; benefits from the write.
- `max_quality/src/moe_compress/stage2/permutation_align.py` — `_PermAlignCache` class; no signature change.
- `max_quality/src/moe_compress/stage2/plugins/ream_cost_post.py` — symmetric write already present; no change.

---

## 5. Plugin Docstring Update

Append to the `_tentative_merged_weights` docstring (before the closing `"""`):

```
    Per SC_FAST_PLAN_V3.md §4-B1: caches the freshly-computed permutation
    under ``(li, centroid_id, child_id)`` for reuse by the eventual merge
    step (``_merge_experts_inplace``). Side-effect only; cost matrix is
    byte-identical.
```

---

## 6. Test Cases

### New file: `max_quality/tests/test_stage2_output_space_perm_cache_write.py`

Use the synthetic-MoE pattern from `test_stage2_output_cost.py` (hidden=4, d_int=3, n_exp=2, top_k=1).

**T1 — cache-miss path writes the cache**:
- Construct fresh `_PermAlignCache()`.
- Call `_tentative_merged_weights(layer_ref, centroid_id=0, child_id=1, freq={0:2,1:2}, ream_acc=None, perm_cache=cache)`.
- Assert `cache.has((layer_ref.layer_idx, 0, 1))` is `True`.
- Assert the stored perm has shape `(d_int,)` — a 1-D integer array of length `intermediate_size`.
- Assert `cache.get((layer_ref.layer_idx, 0, 1))[1] is None` — residual is `None`.

**T2 — cached perm matches the perm used in the merge**:
- Fresh cache; first call (cache miss → write).
- Retrieve `perm_stored, _ = cache.get((li, 0, 1))`.
- Second call with same args (cache hit → perm read from cache).
- For all `MATRIX_NAMES`, second-call merged weights `torch.allclose` to first-call merged weights (atol=1e-6).

**T3 — cache-hit path does NOT overwrite (idempotency)**:
- Fresh cache; first call populates `(li, 0, 1)`.
- Manually `cache.put((li, 0, 1), perm_stored, residual=42.0)`.
- Second call (cache hit; existing entry used, NOT overwritten).
- Assert `cache.get((li, 0, 1))[1] == 42.0` — sentinel residual still present.

**T4 — `perm_cache=None` path unchanged**:
- Call `_tentative_merged_weights(..., perm_cache=None)`.
- Assert no exception, returned merged weights are finite, correct shapes.

### Existing tests — must remain byte-identical (no change)
- `max_quality/tests/test_stage2_plugin_output_space_cost.py`
- `max_quality/tests/test_stage2_output_cost.py`
- `max_quality/tests/test_stage2_plugin_em_refine.py`
- `max_quality/tests/test_stage2_merging.py`

Per spec lines 222–225: cost matrix byte-identical; only the merge-step's LAP outputs change (and they're deterministic identical to the cost-matrix LAP outputs at the same key).

---

## 7. Risk Register

**R1 — `PermCache.put` signature** (CLOSED): verified `permutation_align.py:48` accepts `residual: float | None`. Docstring explicitly documents `None`.

**R2 — Cache key collision** (CLOSED): the post path also writes `(li, c_id, m_id)`. Both compute identical perms deterministically. Output path writes first (residual=None); post path, if running, overwrites with a real residual. Downstream consumers use `cached[0]` only (the perm), never the residual. No behavioral impact.

**R3 — Memory blow-up** (LOW): cache docstring (`permutation_align.py:38–39`): "~6 MB/layer" — bound is `N × K` where `K = cost_topk_filter = 48`, not `N_experts = 256`. Cache instantiated per-layer (`layer_merge.py:438`), GC'd between layers. Peak memory is per-layer, not cumulative. No new entries beyond what the cost loop already traverses.

**R4 — Thread safety** (NOT APPLICABLE): `_PermAlignCache` is a plain dict. Stage 2 cost-matrix loop is single-threaded per layer.

---

## 8. Acceptance Gates (SUPERVISOR runs AFTER both review loops close)

**G1**: `pytest max_quality/tests/test_stage2_output_space_perm_cache_write.py -v` — 4 passed.

**G2**: `pytest max_quality/tests/test_stage2_plugin_output_space_cost.py max_quality/tests/test_stage2_output_cost.py -v` — all green, byte-identical.

**G3**: `pytest max_quality/tests/test_stage2_plugin_em_refine.py max_quality/tests/test_stage2_merging.py -v` — downstream consumers green.

**G4**: `pytest max_quality/tests/ -q --timeout=600` — full suite. Expected: **1513 passed, 13 skipped** (1509 baseline + 4 new tests).

**G5**: commit on `main`, push.

---

## 9. Out of Scope

- The read side of the cache.
- The `_PermAlignCache` class signature.
- `tentative_active` flag in the output path.
- Opts B2/B3/B4 — separate plans.
- Any file outside the two listed in §4.

---

## 10. Workflow Reminder

**Implementer writes**:
1. Code change to `output_space_cost.py` (§2 + §5).
2. New test file `test_stage2_output_space_perm_cache_write.py` (§6).

**Implementer does NOT run pytest.**

**Supervisor runs gates G1–G5** after BOTH review loops (paper-fidelity + code-quality) close with all-none findings.
