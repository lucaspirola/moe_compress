# Stage 2 Plugin Author Guide

How to read, extend, and test the Stage 2 (REAP + REAM) plugin architecture.

---

## 1. Overview

Stage 2 prunes and merges MoE experts. After the plugin-architecture refactor
(Tasks 2-18) the orchestrator `stage2.orchestrator.run()` is thin: it parses and
validates config, scans `_stage2_partial/` for resume state, builds a
`Stage2Pipeline`, walks every MoE layer through that pipeline, and saves the
final compressed checkpoint. Shared run-scope setup (calibration, covariance
accumulator, merge-heal config) and the final-checkpoint save run **once**,
around the per-layer loop.

The concrete algorithms — REAP scoring, the REAM cost matrix and its variants,
the assignment solvers, two-opt and EM refinement, per-group expert
distillation, per-layer merge-heal — all live under `stage2/plugins/*.py`.
The shared mechanics they reuse (merging, permutation alignment, grouping,
profiling, IO, resume) live under `stage2/*.py`. `stage2/orchestrator.py` itself
is now `module docstring + imports + a backward-compatibility re-export block +
run()` and nothing else.

The re-export block at the top of `stage2/orchestrator.py` keeps the historical
`moe_compress.stage2_reap_ream._<name>` import paths working for external
callers and the test suite. **Do not remove a name from it** — it is the public
compatibility contract.

---

## 2. The pipeline objects

| Object | Module | Role |
|---|---|---|
| `Stage2Pipeline` | `stage2/_framework/pipeline.py` | Holds an ordered list of plugin instances. `run_setup` / `run_layer` / `run_teardown`. `run_layer` walks `Stage2Pipeline.phases` and dispatches each phase to every plugin in registration order. |
| `Stage2Plugin` | `stage2/_framework/base.py` | Base class. Every hook is a no-op default. Class attrs `name`, `enabled_by`; classmethod `is_enabled(cfg)`. |
| `PluginRegistry` | `stage2/_framework/registry.py` | Ordered list of plugin *classes*. `register`, `classes`, `active(cfg)` (instantiates the `is_enabled`-true subset), and the static `dispatch_first` helper for slot-style hooks. |
| `PipelineContext` | `pipeline/context.py` | Named-slot state object. The root instance is the per-run context passed to `on_run_setup` / `on_run_teardown`; each layer opens a `child()` scope passed to every per-layer hook. Run-scope mutable scratchpad lives on the plugin instance (as `LegacyAdapter` does). |

### `Stage2Pipeline.phases`

The fixed per-layer phase walk, in execution order:

```
on_layer_setup
on_profile
on_score
compute_assignment
pre_merge_snapshot
merge
post_merge
write_artifacts
on_layer_teardown
```

Plus two run-scope phases dispatched outside the layer loop: `on_run_setup`
(in `run_setup`) and `on_run_teardown` (in `run_teardown`).

`run_layer` iterates `phases` and, for each phase name, calls that method on
every registered plugin in registration order. A plugin that does not override
a hook inherits the no-op default and is silently skipped at no cost.

---

## 3. The plugin lifecycle hook contract

### Per-layer + run-scope phases (these are on the live walk)

| Hook | When it fires | Typical use |
|---|---|---|
| `on_run_setup(run_ctx)` | Once, before any layer | Allocate run-scope resources, read `run_ctx.config`. |
| `on_layer_setup(ctx)` | Start of each layer | Allocate per-layer accumulators (e.g. `ctx.reap_acc`, `ctx.ream_acc`, `ctx.perm_cache`). |
| `on_profile(ctx)` | After setup | Run the calibration forward pass; fill the accumulators. |
| `on_score(ctx)` | After profiling | Publish per-expert saliency — `ReapScoringPlugin` writes `ctx.scores` / `ctx.freq` here. |
| `compute_assignment(ctx)` | After scoring | Build the cost matrix, solve the child→centroid assignment, refine it, produce the grouped result on `ctx`. |
| `pre_merge_snapshot(ctx)` | Before merge | Snapshot pre-merge expert weights (needed by expert-distill / merge-heal). |
| `merge(ctx)` | Apply the merge | Fuse experts in place, resize the router, run expert distillation. |
| `post_merge(ctx)` | After merge | Per-layer merge-heal, telemetry. |
| `write_artifacts(ctx, partial_dir)` | After post-merge | Write the per-layer partial checkpoint to `partial_dir`; return a dict of artifact metadata. |
| `on_layer_teardown(ctx)` | End of each layer | Drop per-layer accumulators, free memory. |
| `on_run_teardown(run_ctx)` | Once, after all layers | Release run-scope resources. |

### Fine-grained slot hooks (declared, NOT yet on the live walk)

`Stage2Plugin` also declares four fine-grained sub-hooks that decompose
`compute_assignment`:

| Slot hook | Return-value contract |
|---|---|
| `compute_cost(ctx)` | Return a cost matrix, or `None` to defer to the next plugin. |
| `apply_cost_mask(ctx, cost)` | Return a masked cost matrix, or `None` to defer. |
| `solve_assignment(ctx, cost)` | Return an assignment, or `None` to defer. |
| `refine_assignment(ctx, assignment)` | Return a refined assignment, or `None` to defer. |

These are *slot* hooks: `PluginRegistry.dispatch_first` walks the plugins and
picks the **first non-`None` return**. A plugin returns `None` to say "not my
job, ask the next one".

**Current wiring caveat — read this before adding a cost/solver/refine
plugin.** `LegacyAdapter` still drives `compute_assignment` as a single
**compound** phase: it runs the whole cost → mask → solve → refine → group →
capacity-bump loop internally. The four slot hooks above are declared on
`Stage2Plugin` and the cost/solver/refine plugins are individually unit-tested,
but the slot hooks are **not in `Stage2Pipeline.phases`**. A future task
decomposes `compute_assignment` into the four-hook walk so those plugins drive
the assignment directly. Until then a cost/solver/refine plugin is constructed
and tested but is **not** on the live phase walk — its logic is reached because
`LegacyAdapter` re-imports the canonical helper.

---

## 4. How to add a new paper as a plugin

1. **Create the module.** `stage2/plugins/<name>.py`.
2. **Subclass `Stage2Plugin`.** Set the class attr `name = "<name>"`.
3. **Pick the hook(s).** Override only the lifecycle hooks the algorithm needs;
   inherit the no-op default for the rest.
4. **Gate it.** Either set `enabled_by = ("<config_flag>",)` — the default
   `is_enabled` returns true iff every named flag under
   `config["stage2_reap_ream"]` is truthy (AND-of-truthy-flags) — or override
   `is_enabled(cls, cfg)` for non-boolean gates (numeric thresholds, enum
   values). See `SkipMergeFloorPlugin` (gate is `skip_merge_percentile < 100.0`)
   and `ReamCostPlugin` for the override pattern.
5. **Wire the config flag.** Add the flag under `stage2_reap_ream.*` and
   validate it at the top of `run()` — fail-fast at the config boundary, the
   way `assignment_solver` and `cost_alignment` are validated.
6. **Register it.** Add the plugin to the `run()` pipeline build (currently
   `Stage2Pipeline(plugins=[ReapScoringPlugin(), adapter])`).
7. **Avoid import cycles.** A plugin module must **not** do a module-top
   `from ..orchestrator import ...` — `stage2/orchestrator.py` re-imports from
   `stage2/`, so a top-level back-import deadlocks. Import siblings under
   `stage2/` directly, or use a function-scope late import (see the
   circular-import note in `ream_cost.py`).

---

## 5. Shared tools under `stage2/` — reuse, do not duplicate

| Module | Reusable helpers |
|---|---|
| `stage2/merging.py` | `_merge_experts_inplace`, `_resize_router_for_kept_experts` |
| `stage2/permutation_align.py` | `_permutation_align_to_centroid`, `_aligned_whitened_residual`, `_PermAlignCache` |
| `stage2/grouping.py` | `_build_grouped_from_assignment`, `_promote_orphans`, `_apply_skip_merge_floor` |
| `stage2/profiling.py` | `_profile_layer`, `_LayerInputAccumulator` |
| `stage2/shared_io.py` | Covariance / heal-weights / merge-JSON IO helpers |
| `stage2/resume.py` | `discover_completed_layers`, `ResumedLayerRecord` |

If your algorithm needs to merge experts, align permutations, build groups,
profile a layer, or write artifacts — call the canonical helper above. Holding
a private copy is the bug class the T9-T11 monkeypatch-drift tests exist to
catch.

---

## 6. Worked example skeleton

A minimal plugin that hooks one phase, gated by a boolean flag:

```python
"""Example plugin — <one-line paper summary>."""
from __future__ import annotations

from typing import Any

from .._framework.base import Stage2Plugin
from ...pipeline.context import PipelineContext


class ExampleScorePlugin(Stage2Plugin):
    """Adjusts per-expert saliency before assignment (example skeleton)."""

    name = "example_score"
    enabled_by = ("example_score_enabled",)  # stage2_reap_ream.example_score_enabled

    def on_score(self, ctx: PipelineContext) -> None:
        # ctx.scores / ctx.freq are published here by ReapScoringPlugin;
        # mutate or augment them, then leave them on ctx for downstream phases.
        ...
```

Matching config-validation snippet for `run()` (next to the existing
`assignment_solver` / `cost_alignment` validation):

```python
    example_score_enabled: bool = bool(s2.get("example_score_enabled", False))
    # ... validate any numeric companions here, fail-fast on bad values ...
```

Matching registration snippet in the `run()` pipeline build:

```python
    plugins = [ReapScoringPlugin(), ExampleScorePlugin(), adapter]
    pipeline = Stage2Pipeline(plugins=plugins)
```

For a non-boolean gate, drop `enabled_by` and override `is_enabled`:

```python
    enabled_by: tuple[str, ...] = ()

    @classmethod
    def is_enabled(cls, cfg: dict[str, Any]) -> bool:
        s2 = cfg.get("stage2_reap_ream", {})
        return float(s2.get("example_threshold", 0.0)) > 0.0
```

---

## 7. The testing pattern

Each plugin gets `tests/test_plugin_<name>.py`. Use
`tests/test_plugin_skip_merge_floor.py` as the reference. Mandatory checks:

- **`is_enabled` truth table** — flag on / flag off / missing key.
- **Plugin contract** — `issubclass(<Plugin>, Stage2Plugin)`, the `name`
  attribute, the `enabled_by` attribute.
- **Re-import identity** — if the algorithm's `_`-prefixed functions are
  re-exported from `stage2/orchestrator.py`, assert
  `orchestrator._fn is <plugin_module>._fn` so an accidental divergence is
  caught at test time.
- **Monkeypatch-drift guard** — assert the plugin delegates to the canonical
  helper rather than holding a stale private copy (the T9-T11 lesson).
- **Hook behaviour** — call the hook directly with a constructed
  `PipelineContext` (a `child()` scope for per-layer hooks) and assert the
  expected `ctx` mutation.

---

## 8. Current state / roadmap

`LegacyAdapter` (`stage2/plugins/legacy_adapter.py`) is a **transitional**
plugin. It still owns `compute_assignment` as one compound hook — the whole
cost → solve → refine → group → capacity-bump loop runs inside it.

The eleven algorithm plugins (T7-T17) are extracted and individually tested,
but only `ReapScoringPlugin` (`on_score`) is on the live phase walk alongside
`LegacyAdapter`. The cost / solver / refine plugins are inert shells: their
canonical logic is reached because `LegacyAdapter` re-imports it, not because
they sit on the walk.

The next milestone decomposes `compute_assignment` into the four fine-grained
slot hooks (`compute_cost`, `apply_cost_mask`, `solve_assignment`,
`refine_assignment`) so the cost / solver / refine plugins drive the assignment
walk directly and `LegacyAdapter` can be retired.

The `MOE_STAGE2_LEGACY_LOOP` env-var escape hatch and the `_run_legacy_layer_loop`
helper were deleted in Task 18 — the `Stage2Pipeline` plugin walk is now the
sole per-layer path.
