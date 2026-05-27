# PLAN_PLUGIN_06 — `on_post_merge` Hook in Stage 2 Phase Schedule

**Status**: Ready for implementation
**Implementer deliverables**: (1) schedule update in `orchestrator.py`, (2) updates to 3 existing tests, (3) 1 new test verifying hook firing order.
**Implementer does NOT run pytest.** Supervisor runs gates after both review loops close.

---

## 1. Goal & Spec Citation

Add an `on_post_merge` hook to the Stage 2 per-layer phase schedule. Fires after `post_merge` and before `write_artifacts` — Position A. Gives downstream plugins a dedicated phase to invalidate per-layer state caches that become stale after the merge.

**Spec citations**:
- `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §582 (Risk R2): "If drift detected, add an `on_post_merge` plugin hook that invalidates layer-input + ream + cov caches." User elected to apply unconditionally.
- `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §523-532 (S2_SEQ): "requires invalidating cov_acc, ream_acc, and layer_input reservoir caches after each layer's merge — add `on_post_merge` plugin hook." Primary consumer is Plugin #10 (`ream_sequential.py`, not implemented here).

---

## 2. Hook Position Decision — Position A

**Position**: between `post_merge` and `write_artifacts`. Rationale:

| Position | Per-layer ctx slots live? | Verdict |
|----------|--------------------------|---------|
| A: after `post_merge`, before `write_artifacts` | YES — caches alive, can be invalidated with `ctx.set(..., None, overwrite=True)` | **CHOSEN** |
| B: after `write_artifacts`, before `on_layer_teardown` | YES | gratuitously late; no benefit over A |
| C: after `on_layer_teardown` | NO — per-layer slots gone; only run-scope state | contradicts §582 intent |

**Semantic boundary (for orchestrator docstring)**:
- `post_merge` = in-layer reaction (MergeHealPlugin, ExpertDistillPlugin observe/repair merged tensor)
- `on_post_merge` = inter-layer cache invalidation (S2_SEQ clears cov_acc/ream_acc/layer_input_acc)

---

## 3. Schedule Update

**File**: `max_quality/src/moe_compress/stage2/orchestrator.py`

### Change 1 — `_STAGE2_POST_ASSIGN_PHASES` (lines 190-196)

Insert `"on_post_merge"` between `"post_merge"` and `"write_artifacts"`:

```python
_STAGE2_POST_ASSIGN_PHASES: tuple[str, ...] = (
    "pre_merge_snapshot",
    "merge",
    "post_merge",
    "on_post_merge",   # NEW: SC_STAGE12 §582 — inter-layer cache invalidation
    "write_artifacts",
    "on_layer_teardown",
)
```

### Change 2 — `_STAGE2_LAYER_PHASES` (lines 197-203)

**No code change needed.** The constant is derived from the two halves; it automatically expands to a 10-tuple.

### Change 3 — Add semantic-boundary comment

Add to the preamble comment block (lines 174-183 area) explaining `post_merge` vs `on_post_merge`:

```
# ``post_merge`` vs ``on_post_merge``:
#   post_merge    — in-layer reaction to the merge (MergeHealPlugin,
#                   ExpertDistillPlugin observe/repair the merged weight tensor).
#   on_post_merge — inter-layer cache invalidation (S2_SEQ / REAM sequential:
#                   clears cov_acc, ream_acc, layer_input_acc so the next layer's
#                   on_layer_setup → on_profile sees fresh state).
#                   Per SC_STAGE12 §582.
```

---

## 4. Plugin Protocol Implication — NO CHANGE NEEDED

`pipeline/plugin.py` PipelinePlugin Protocol declares NO phase-hook methods (design note 3: "Phase-hook names are an open vocabulary"). `BasePlugin` also has no phase hooks. `tools/phase_walker.py` uses `getattr(plugin, phase, None)` with a `callable(hook)` guard. All 18 existing Stage 2 plugins lack `on_post_merge`; they are silently skipped. **Zero existing plugin files touched.**

---

## 5. Test Updates — 3 Existing Tests

### 5.1 `test_stage2_pipeline_scaffold.py` — `test_pipeline_phases_are_declared_in_canonical_order`

Lines 73-94. Update the asserted tuple to include `"on_post_merge"` between `"post_merge"` and `"write_artifacts"`. Update any "9 entries" docstring → "10 entries".

### 5.2 `test_stage2_pipeline_run_layer.py` — `test_phases_tuple_matches_t6_canonical_order`

Lines 192-228. Update BOTH asserted tuples:
- `_STAGE2_POST_ASSIGN_PHASES` literal: add `"on_post_merge"` between `"post_merge"` and `"write_artifacts"`.
- `_STAGE2_LAYER_PHASES` literal: same insertion (now 10-tuple).

### 5.3 `test_stage2_pipeline_run_layer.py` — `_CountingPlugin` and `test_run_layer_visits_each_phase_in_canonical_order`

Lines 118-158 (`_CountingPlugin`) and 161-186 (the test).

**Add to `_CountingPlugin`** (after the `post_merge` method, before `write_artifacts`):
```python
def on_post_merge(self, ctx):
    self.calls.append("on_post_merge")
```

The test at line 181-186 asserts `plugin.calls` against a list derived from the phase tuples — once `_STAGE2_POST_ASSIGN_PHASES` includes `on_post_merge` AND `_CountingPlugin` implements it, the test passes without further changes.

---

## 6. New Test — Hook Firing Order

**File**: `max_quality/tests/test_stage2_pipeline_run_layer.py` (append at end of file).

```python
class _PostMergeProbePlugin:
    """Records phase calls; implements on_post_merge to verify it fires."""

    name = "post_merge_probe"

    def __init__(self):
        self.calls: list[str] = []

    def pre_merge_snapshot(self, ctx):
        self.calls.append("pre_merge_snapshot")

    def merge(self, ctx):
        self.calls.append("merge")

    def post_merge(self, ctx):
        self.calls.append("post_merge")

    def on_post_merge(self, ctx):
        self.calls.append("on_post_merge")

    def write_artifacts(self, ctx):
        self.calls.append("write_artifacts")
        return {}

    def on_layer_teardown(self, ctx):
        self.calls.append("on_layer_teardown")


def test_on_post_merge_fires_after_post_merge_before_write_artifacts(tmp_path):
    """on_post_merge fires in the correct position within _STAGE2_POST_ASSIGN_PHASES.

    Asserts strict ordering:
      merge → post_merge → on_post_merge → write_artifacts → on_layer_teardown.
    Per SC_STAGE12 §582.
    """
    plugin = _PostMergeProbePlugin()
    run_ctx = _make_run_ctx(
        model=object(), tokenizer=object(), config={},
        artifacts_dir=tmp_path, partial_dir=tmp_path, device="cpu",
    )
    layer_ctx = _make_layer_ctx(run_ctx, layer_idx=0, layer_ref=object(),
                                n_experts=4, target=2)
    walk_phases(_STAGE2_POST_ASSIGN_PHASES, [plugin], layer_ctx)

    assert plugin.calls == [
        "pre_merge_snapshot",
        "merge",
        "post_merge",
        "on_post_merge",
        "write_artifacts",
        "on_layer_teardown",
    ]
    assert "on_post_merge" in _STAGE2_POST_ASSIGN_PHASES
    post_idx = _STAGE2_POST_ASSIGN_PHASES.index("post_merge")
    opm_idx = _STAGE2_POST_ASSIGN_PHASES.index("on_post_merge")
    wa_idx = _STAGE2_POST_ASSIGN_PHASES.index("write_artifacts")
    assert post_idx < opm_idx < wa_idx
```

`_make_run_ctx`, `_make_layer_ctx`, and `walk_phases` are already in scope at the top of the file.

---

## 7. Reference Implementation for S2_SEQ (NOT to implement here)

For the future `ream_sequential.py` plugin (Plugin #10):
```python
def on_post_merge(self, ctx):
    # Invalidate caches stale after this layer's merge so the next layer's
    # on_layer_setup → on_profile sees fresh state. Per SC_STAGE12 §582.
    ctx.set("cov_acc", None, overwrite=True)
    ctx.set("ream_acc", None, overwrite=True)
    ctx.set("layer_input_acc", None, overwrite=True)
```

---

## 8. Files to Touch

| File | Nature |
|------|--------|
| `max_quality/src/moe_compress/stage2/orchestrator.py` | insert `"on_post_merge"` in `_STAGE2_POST_ASSIGN_PHASES` + semantic comment in preamble |
| `max_quality/tests/test_stage2_pipeline_scaffold.py` | update `test_pipeline_phases_are_declared_in_canonical_order` tuple |
| `max_quality/tests/test_stage2_pipeline_run_layer.py` | update both canonical-order tests + `_CountingPlugin` + append new firing-order test |
| `pipeline/plugin.py` | **NO CHANGE** |
| `tools/phase_walker.py` | **NO CHANGE** |

---

## 9. Risk Register

| ID | Risk | Resolution |
|----|------|-----------|
| R1 | Schedule change breaks existing 18 Stage 2 plugins | None — walker uses `getattr(..., None)`; missing hook silently skipped |
| R2 | 3 canonical-order tests fail | Handled in §5 — all 3 updates listed |
| R3 | Performance impact | Negligible — one extra dict lookup per plugin per layer |
| R4 | Future ambiguity (`post_merge` vs `on_post_merge`) | Resolved by semantic-boundary comment in §3 |

---

## 10. Acceptance Gates (SUPERVISOR after review loops close)

- **G1**: new `test_on_post_merge_fires_after_post_merge_before_write_artifacts` green
- **G2**: `test_phases_tuple_matches_t6_canonical_order` green (with updated tuples)
- **G3**: `test_pipeline_phases_are_declared_in_canonical_order` green (with updated tuple)
- **G4**: `test_run_layer_visits_each_phase_in_canonical_order` green (with updated `_CountingPlugin`)
- **G5**: full suite green. Expected: 1516 + 1 new = **1517 passed, 13 skipped**
- **G6**: commit + push

---

## 11. Out of Scope

- S2_SEQ implementation (Plugin #10).
- Any existing plugin behavior change.
- Stage 1 / 3 / 4 / 5 / 6.
- `dispatch_first` slots.

---

## 12. Workflow Reminder

Implementer writes code + tests, does NOT run pytest. Supervisor runs gates after BOTH review loops close.
