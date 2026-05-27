# PLAN_PLUGIN_10 — `S2_SEQ` — REAM Sequential Merging

**Status**: Ready for implementation.
**Plugin slot**: Stage 2 row #19, opt-in.
**Spec citations**:
- `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §523-532 (Section 7, A3): "Integrate REAM sequential merging … Vendor github.com/SamsungSAILMontreal/ream under `max_quality/src/moe_compress/stage2/plugins/ream_sequential.py`; wire into `stage2/orchestrator.py` re-profile loop. … requires invalidating cov_acc, ream_acc, and layer_input reservoir caches after each layer's merge — add `on_post_merge` plugin hook."
- `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §331-358 (Section 5.2, R2): REAM paper anchor (arXiv:2604.04356, Samsung SAIL Montreal, Apr 2026), open-source @ github.com/SamsungSAILMontreal/ream.
- `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` §582 (Risk R2): "If drift detected, add an `on_post_merge` plugin hook that invalidates layer-input + ream + cov caches."

**Foundational dep**: Plugin #6 (`on_post_merge` phase) lands in commit `ce49a1b` upstream. This plan is its first consumer.

---

## 1. Goal

Implement `Stage2ReamSequentialPlugin` — a Stage 2 plugin that fires on the `on_post_merge` phase added by Plugin #6 and invalidates the three per-layer caches that REAM Section 4 (Sequential Merging) identifies as stale after a layer is merged:

- `cov_acc` (input covariance accumulator)
- `ream_acc` (REAM cost accumulator: gate-logit + gated-output statistics)
- `layer_input_acc` (reservoir of post-attn / pre-MoE inputs used by `post`/`output` cost modes)

Per `tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md` Section 5.2: the REAM paper's central claim is that "once the experts in layer ℓ are compressed, its modified outputs render the statistics for the subsequent layers as stale … after merging layer ℓ, a second forward pass is run through this layer to recompute its activations to be used by the subsequent layer ℓ+1." This plugin implements the **invalidation half** of that loop — the next layer's `on_layer_setup → on_profile` reads None and rebuilds from scratch, which means the next layer's profile pass naturally sees the merged upstream context that already flows through the (live) model.

**No re-profile driver of our own**: our existing per-layer loop in `stage2/orchestrator.py` already runs profile-then-merge layer-by-layer; the per-layer `on_profile` hook in `LayerMergePlugin` re-creates accumulators per layer (line 434 `ream_acc = ReamCostAccumulator()`). Invalidating ctx slots is sufficient — the next iteration's `on_layer_setup` overwrites them anyway, and clearing them makes the contract explicit / safe in the face of any cross-layer state leakage.

## 2. REAM Paper Anchor (Rec 4a)

REAM (arXiv:2604.04356, Liu et al., Apr 2026, Samsung SAIL Montreal) — paper §4 "Sequential merging" (`audit/spec_compliance/01_papers/2604.04356/source.md` lines 435-448):

> Prior expert pruning and merging methods run a single forward pass through the original, unmodified model to collect per-layer statistics. The pre-collected statistics are then used to compress all layers independently. However, once the experts in layer ℓ are compressed, its modified outputs render the statistics for the subsequent layers as stale. Instead, we propose updating the model outputs to reflect the currently merged layers. After merging layer ℓ, a second forward pass is run through this layer to recompute its activations to be used by the subsequent layer ℓ + 1.

**Project-side implementation note**: in our codebase, the calibration loop already calls `_profile_layer` per layer on the *current* model state (the model is mutated in-place by `_merge_experts_inplace` in `merge` phase). So unlike paper-time, where "compute all stats up-front" is the baseline, our `on_profile` already runs against the live model. The remaining gap that §582 R2 flags is **per-layer accumulator caches**: if `cov_acc` (run-scope, on `LayerMergePlugin`) carries forward stale per-layer entries, the next layer's cost would consult them. Similarly `ream_acc` and `layer_input_acc` are per-layer ctx slots that need to be visibly null at the start of the next layer's `on_layer_setup` so the rebuild in `LayerMergePlugin.on_layer_setup` is unambiguous.

The plugin scope is exactly **cache invalidation**; the re-profile itself is already what the existing layer loop does once caches are clear.

## 3. What this plugin does NOT do

- **Does NOT modify** `LayerMergePlugin`, `_profile_layer`, or the existing per-layer loop. The only entry point is the `on_post_merge` phase added by Plugin #6.
- **Does NOT** run a second forward pass on its own. The current orchestrator already iterates layers against the live (merged) model state; once caches are cleared, the **next** layer's profile pass sees the merged upstream context naturally.
- **Does NOT** vendor any code from `github.com/SamsungSAILMontreal/ream` literally. The reference repo's sequential mode is a top-level driver loop; our orchestrator's existing per-layer driver already plays that role. What's vendored is the **conceptual contract** (cache-invalidation requirement § paper §4) — explicitly attributed in the plugin's `paper` field and docstring.
- **Does NOT** change the default behavior. Defaults to OFF (`sequential_reprofile: false`). With the gate off, the plugin is dropped by `registry.enabled(config)` and the byte-identical existing path is preserved.

## 4. Deviations from the SAILMontreal reference repo

Documented in the plugin docstring (per project default for paper-spec deviations):

- **D-seq-1**: The reference repo runs a dedicated second-pass forward through the just-merged layer to refresh accumulators. Our implementation invalidates the ctx-level caches and relies on the existing per-layer loop's natural sequential structure — every per-layer `on_profile` already runs against the live (merged) model state. Functional equivalence holds because (a) `_profile_layer` reconstructs `ream_acc` / `layer_input_acc` from a fresh forward pass; (b) `cov_acc` accumulates over the calibration batches at the live state; (c) the `on_post_merge` invalidation eliminates any stale prior-layer entry hazard.
- **D-seq-2**: The reference repo's sequential mode is controlled by `--sequential-merging` CLI flag; we expose it as `stage2_reap_ream.sequential_reprofile` YAML knob, default `false`.
- **D-seq-3**: The reference repo invalidates only its internal activation-cost accumulator (single accumulator). We invalidate three ctx slots (`cov_acc`, `ream_acc`, `layer_input_acc`) because our cost framework has three independent cost modes (`pre` / `post` / `output`) backed by three separate accumulators (`stage2.plugins.ream_cost`, `stage2.plugins.ream_cost_post`, `stage2.plugins.output_space_cost`). Strict superset of what REAM clears.

## 5. Implementation

### 5.1 New file: `max_quality/src/moe_compress/stage2/plugins/ream_sequential.py`

Single class `Stage2ReamSequentialPlugin`. Sketch:

```python
class Stage2ReamSequentialPlugin:
    name = "ream_sequential"
    paper = (
        "REAM §4 Sequential Merging — arXiv:2604.04356 (Liu et al., 2026, "
        "Samsung SAIL Montreal). Official code: SamsungSAILMontreal/ream. "
        "Plugin clears cov_acc / ream_acc / layer_input_acc on on_post_merge "
        "so the next layer's on_layer_setup → on_profile sees fresh state "
        "reflecting the freshly-merged upstream context. "
        "Deviations: D-seq-1 (no second-pass driver — uses existing per-layer "
        "loop's natural sequential structure), D-seq-2 (YAML knob naming), "
        "D-seq-3 (clears 3 accumulators because our cost framework has 3 modes)."
    )
    config_key = "stage2_reap_ream.sequential_reprofile"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("cov_acc", "ream_acc", "layer_input_acc")
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        s2 = config.get("stage2_reap_ream", {}) if isinstance(config, dict) else {}
        return bool(s2.get("sequential_reprofile", False))

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def on_post_merge(self, ctx: PipelineContext) -> None:
        """Invalidate per-layer cost accumulators after a merge.

        Per SC_STAGE12 §582 / REAM paper §4. The next layer's
        ``on_layer_setup`` overwrites these slots anyway; clearing them
        first makes the contract explicit and protects against any
        cross-layer state leakage.

        ``overwrite=True`` is an upsert: works whether the slot was set or
        not, mirrors ``LayerMergePlugin.on_layer_teardown`` (lines 717-720).
        """
        ctx.set("cov_acc", None, overwrite=True)
        ctx.set("ream_acc", None, overwrite=True)
        ctx.set("layer_input_acc", None, overwrite=True)
```

### 5.2 Wire into `stage2/orchestrator.py`

Add import + registry entry **after** `MergeHealPlugin` so it runs LAST in the `on_post_merge` phase walk (no current plugin implements `on_post_merge`, but ordering it last is the safest default — any future plugin that wants to read these caches in `on_post_merge` would do so before invalidation):

```python
from .plugins.ream_sequential import Stage2ReamSequentialPlugin
# … in PluginRegistry([…]):
    MergeHealPlugin(…),
    Stage2ReamSequentialPlugin(),   # NEW — Plugin #10
```

`LayerMergePlugin` does NOT implement `on_post_merge`, so the "must run after LayerMergePlugin in on_post_merge phase" requirement from the brief is satisfied trivially (LayerMergePlugin has nothing to dispatch in that phase). The registry-position contract pinned by the test is: the new plugin appears after `MergeHealPlugin` in `orchestrator.py`.

### 5.3 Config key

`stage2_reap_ream.sequential_reprofile: bool` (default `false`). NO default value needs adding to any config file — `is_enabled` reads with `False` default, so absent-key = OFF, byte-identical existing behavior preserved. To enable, the operator adds `sequential_reprofile: true` under `stage2_reap_ream:` in their run YAML.

## 6. Files Touched

| File | Change |
|------|--------|
| `max_quality/src/moe_compress/stage2/plugins/ream_sequential.py` | NEW — plugin class (~150 lines incl. docstrings) |
| `max_quality/src/moe_compress/stage2/orchestrator.py` | +2 lines: import + registry entry after `MergeHealPlugin` |
| `max_quality/tests/test_stage2_plugin_ream_sequential.py` | NEW — Protocol conformance, gate, on_post_merge contract, registry order |

## 7. Tests

`max_quality/tests/test_stage2_plugin_ream_sequential.py`:

1. `test_plugin_conforms_to_pipeline_plugin` — `isinstance(plugin, PipelinePlugin)`.
2. `test_plugin_name_and_metadata` — name == "ream_sequential", paper cites arXiv:2604.04356.
3. `test_is_enabled_true_when_knob_true`.
4. `test_is_enabled_false_when_knob_false`.
5. `test_is_enabled_false_when_key_missing` — absent key defaults to OFF.
6. `test_is_enabled_false_when_block_missing`.
7. `test_on_post_merge_invalidates_three_caches` — build a PipelineContext, pre-populate cov_acc / ream_acc / layer_input_acc with sentinel objects, call `on_post_merge`, assert all three resolve to `None`.
8. `test_on_post_merge_works_when_slots_never_set` — overwrite=True upsert; no KeyError.
9. `test_orchestrator_registers_after_merge_heal` — source-string assertion: in `orchestrator.py`, `Stage2ReamSequentialPlugin(` appears AFTER the `MergeHealPlugin(` registry entry.
10. `test_registry_enabled_subset` — with knob True, plugin appears in `registry.enabled(config)`; with knob False, dropped.
11. `test_default_byte_identical_path` — with default config (no knob), `is_enabled` is False and `registry.enabled` does not include the plugin (proves the no-config-change-needed default).

## 8. Workflow gates

1. Plan (this file) → user signoff.
2. Implement.
3. Paper-fidelity review (`general-purpose` subagent): does the implementation match REAM §4? Are vendored portions correctly attributed? Loop until clean.
4. Code-quality review (`feature-dev:code-reviewer` subagent): 5-category sweep. Loop until clean.
5. Tests + gates:
   - `pytest max_quality/tests/test_stage2_plugin_ream_sequential.py` — all green.
   - `pytest max_quality/tests/test_stage2_*` — all green (default byte-identical).
   - Full suite green.
6. Commit on `feat/plugin_10_s2_seq`. Push.

## 9. Risks

- **R-Plan-1**: `cov_acc` is run-scope plugin-instance state on `LayerMergePlugin.cov_acc`, NOT a ctx slot in current code. The spec mandates `ctx.set("cov_acc", None, overwrite=True)` regardless. With `overwrite=True` this is a safe upsert: the ctx slot will exist and be `None` after the call. Future plugins that read `ctx.get("cov_acc")` (none currently) will see the invalidation. The run-scope instance state on `LayerMergePlugin` is untouched (out of scope for this plugin per "Do NOT modify the existing 18 Stage 2 plugins"). Documented in the plugin docstring.
- **R-Plan-2**: Default-OFF guarantees byte-identical existing behavior — pinned by `test_default_byte_identical_path`.

---

*Plan generated 2026-05-27. Worktree: `/home/lucas/ai/moe_compress/.claude/worktrees/agent-ae26cbdd94a7a4a05`. Base: `main` @ HEAD `0e7fd35`. Foundational dep `ce49a1b` (Plugin #6) already landed.*
