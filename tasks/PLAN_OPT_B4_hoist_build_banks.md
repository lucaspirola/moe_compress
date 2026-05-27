# PLAN_OPT_B4 — Hoist `build_banks` out of `_tentative_merged_weights`

**Status**: Ready for implementation
**Implementer deliverables**: (1) code change in `output_space_cost.py`, (2) test updates across two test files, (3) one new defensive test.
**Implementer does NOT run pytest.** Supervisor runs gates after both review loops close.

---

## 1. Goal & Spec Citation

`_tentative_merged_weights` is called ~98,000 times per row during the output-space cost matrix computation. Each call currently executes `banks = build_banks(layer_ref)` at line 239 of `output_space_cost.py`. The outer `_output_space_cost` already builds the identical `banks` dict once at line 380, before entering the loop. The per-pair call is pure overhead.

**Spec**: `SC_FAST_PLAN_V3.md §4-B4` lines 265-273. Saving: ~1 min/row (98k × ~50 µs). Risk: zero — `build_banks` returns view objects from unchanged model parameters; the same dict semantically.

Spec-anchored only.

---

## 2. API Change Decision — Option A (required positional)

**Option A (chosen)**: `banks` is a required positional parameter, appended after `perm_cache`.

**Option B (rejected)**: `banks` is optional kwarg defaulting to `None`; fall back to `build_banks(layer_ref)` if `None`.

**Rationale**: spec says "remove the per-pair call". Option A enforces this at the type level. Option B preserves dead fallback that only executes in tests — masks the regression the defensive test (§6.2) is designed to catch.

---

## 3. Exact Changes

### 3.1 `output_space_cost.py`

**New signature** (replaces line 212–219):
```python
def _tentative_merged_weights(
    layer_ref: MoELayerRef,
    centroid_id: int,
    child_id: int,
    freq: dict[int, int],
    ream_acc: "ReamCostAccumulator | None",
    perm_cache: "_PermAlignCache | None",
    banks: dict,
) -> dict[str, torch.Tensor]:
```

**Delete line 239**: `banks = build_banks(layer_ref)`.

**Update internal call site at lines 444-446**:
```python
                merged = _tentative_merged_weights(
                    layer_ref, c_id, m_id, freq, ream_acc, perm_cache, banks,
                )
```
`banks` already in scope from line 380.

**Append to docstring** (before closing `"""`):
```
    Per SC_FAST_PLAN_V3.md §4-B4: ``banks`` is now passed by the caller,
    hoisted out of the per-pair call to amortize the ``build_banks`` cost
    over the full cost-matrix loop. Callers must pass the same ``banks``
    dict that ``build_banks(layer_ref)`` would return — this function no
    longer builds it internally.
```

### 3.2 `test_stage2_output_cost.py` — 4 call sites

All have `banks = build_banks(layer_ref)` already in enclosing scope; append `banks` arg.

| Line | Change |
|------|--------|
| 67 | append `, banks` |
| 117–120 (kwargs) | add `banks=banks,` |
| 141–144 (kwargs) | add `banks=banks,` |
| 335 | append `, banks` |

`banks` exists in scope at lines 59, 111, 131, 317 respectively (verified by planner).

### 3.3 `test_stage2_output_space_perm_cache_write.py` — 8 call sites across 4 functions

**test_cache_miss_path_writes_perm_to_cache** (1 call at line 96):
- Add `banks = build_banks(layer_ref)` after `layer_ref = _make_layer_ref()`.
- Pass `banks=banks` to call.

**test_cached_perm_yields_identical_merge_on_second_call** (2 calls at lines 156, 170):
- Add `banks = build_banks(layer_ref)` after `layer_ref = _make_layer_ref()`.
- Pass `banks=banks` to both calls.

**test_cache_hit_path_does_not_overwrite_existing_entry** (2 calls at lines 202, 217):
- Add `banks = build_banks(layer_ref)` after `layer_ref = _make_layer_ref()`.
- Pass `banks=banks` to both calls.

**test_none_perm_cache_path_unchanged** (3 calls at lines 248, 272, 280):
- **Hoist** existing `banks = build_banks(layer_ref)` from line 257 to immediately before line 248.
- Pass `banks=banks` to all three calls.
- Existing shape/finite assertions at lines 257–266 unaffected.

`build_banks` already imported at line 29.

---

## 4. Files to Touch

| File | Nature |
|------|--------|
| `max_quality/src/moe_compress/stage2/plugins/output_space_cost.py` | signature + delete line 239 + update call at line 444 + docstring |
| `max_quality/tests/test_stage2_output_cost.py` | update 4 call sites + add 1 new defensive test |
| `max_quality/tests/test_stage2_output_space_perm_cache_write.py` | add `banks = build_banks(...)` in 3 functions + hoist in 1 function + update 8 call sites |

`orchestrator.py:137` re-export transparently picks up new signature. No change needed.

---

## 5. Plugin Docstring Update

See §3.1 above (B4 citation paragraph appended to `_tentative_merged_weights` docstring).

---

## 6. Test Cases

### 6.1 Existing tests (updated, not new)

The 12 call-site updates in §3.2 + §3.3.

### 6.2 New defensive test: `test_tentative_merged_weights_uses_passed_banks`

Note: prior version of this test mutated shared model storage via `ExpertMatrixBank.set()` — corrected to use a non-mutating `_ScaledBankView` mock; see B4 fix review D.2.

Add to `test_stage2_output_cost.py`. Proves the function uses caller's `banks`, not an internally-built one.

```python
def test_tentative_merged_weights_uses_passed_banks(tiny_model):
    """Per SC_FAST_PLAN_V3.md §4-B4: the function must use the ``banks``
    argument for weight lookups, NOT call build_banks(layer_ref) internally.

    Uses a non-mutating mock that returns a ×2-scaled COPY (not an in-place
    mutation) so the real and sentinel banks point to genuinely different
    tensor values at call time. If the function were to ignore the passed
    banks and call build_banks internally, both calls would return the
    same merged weights — the assertion catches that.
    """
    layer_ref = list(iter_moe_layers(tiny_model))[0]
    banks = build_banks(layer_ref)

    # Non-mutating wrapper: returns a 2× COPY of the underlying tensor.
    # Does NOT call .set() — never touches the model's storage.
    class _ScaledBankView:
        def __init__(self, real_bank, scale):
            self._real_bank = real_bank
            self._scale = scale

        def get(self, eid):
            return self._real_bank.get(eid) * self._scale  # returns new tensor

    sentinel_banks = {name: _ScaledBankView(banks[name], 2.0) for name in MATRIX_NAMES}

    freq = {0: 1, 1: 1}

    merged_real = _tentative_merged_weights(
        layer_ref, centroid_id=0, child_id=1,
        freq=freq, ream_acc=None, perm_cache=None, banks=banks,
    )
    merged_sentinel = _tentative_merged_weights(
        layer_ref, centroid_id=0, child_id=1,
        freq=freq, ream_acc=None, perm_cache=None, banks=sentinel_banks,
    )

    # down_proj is a linear function of the bank weights; the 2× scaling
    # of sentinel_banks must propagate into merged_sentinel["down_proj"].
    # If the function ignored the passed banks and called build_banks
    # internally, both merges would use the real (unmutated) weights and
    # return identical down_proj — that's the regression this test catches.
    assert not torch.allclose(
        merged_real["down_proj"], merged_sentinel["down_proj"], atol=1e-6,
    ), (
        "merged_real and merged_sentinel match — "
        "_tentative_merged_weights appears to ignore the passed banks argument"
    )
```

---

## 7. Risk Register

**R1 — API break for external callers**: only callers are the internal `_output_space_cost` site + test files. `orchestrator.py:137` is a re-export (`# noqa: F401`) and transparent. All updated per §3.

**R2 — Semantic difference**: `build_banks` is idempotent (returns view objects from same params). Hoisting changes nothing semantically.

**R3 — Test fixture overhead**: one extra `build_banks` call per test (<1 ms). Negligible.

**R4 — Missed call site**: a 6-arg call raises `TypeError` immediately at test time. Loud, no silent regression.

---

## 8. Acceptance Gates (SUPERVISOR runs after review loops close)

- **G1**: `pytest max_quality/tests/test_stage2_output_cost.py max_quality/tests/test_stage2_output_space_perm_cache_write.py max_quality/tests/test_stage2_plugin_output_space_cost.py -v` — all green incl. new defensive test.
- **G2**: `pytest max_quality/tests/ -q --timeout=600` — full suite. Expected: 1515 passed (1514 baseline + 1 new defensive test), 13 skipped.
- **G3**: commit on `main`, push.

---

## 9. Out of Scope

- NOT changing `build_banks` itself.
- NOT touching B1/B2/B3.
- NOT introducing a shared fixture-level banks cache.
- NOT touching `MoELayerRef` construction.
- NOT modifying `orchestrator.py`.

---

## 10. Workflow Reminder

Implementer writes code + test updates + new defensive test. Implementer does NOT run pytest. Supervisor runs gates after BOTH review loops close.
