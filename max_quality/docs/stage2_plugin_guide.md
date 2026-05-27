# Stage 2 Plugin Author Guide

How to read, extend, and test the Stage 2 (REAP scoring + REAM expert merging)
plugin architecture, as it stands after the universal-plugin migration
(tasks S2-1 … S2-13b complete).

---

## 1. Overview

Stage 2 scores MoE experts (REAP) and merges the low-saliency ones into
high-saliency centroids (REAM). The orchestrator entrypoint
`stage2.orchestrator.run()` is thin: it parses and validates config, scans
`_stage2_partial/` for crash-resume state (via `stage2.resume`), builds a
universal `PluginRegistry`, walks every MoE layer through the phase walker, and
saves the final compressed checkpoint. Shared run-scope setup (calibration,
covariance accumulator, merge-heal config, optional WikiText cross-domain
holdout) and the final-checkpoint save run **once**, around the per-layer loop.

`run()` is re-exported as `moe_compress.stage2.run`. Stage 2 is also exposed as
the `STAGE2` `Stage` object (`stage2/stage.py`) — a thin `Stage`-conforming
adapter that unwraps a `PipelineContext` into the `run()` call and writes
`stage2_pruned_path` back onto the context. The future universal orchestrator
drives Stage 2 the same way it drives every other stage: `STAGE2.is_enabled` →
`STAGE2.run`.

The concrete algorithms — REAP scoring, the REAM cost matrix and its variants,
the assignment solvers, two-opt and EM refinement, per-group expert
distillation, per-layer merge-heal — all live under `stage2/plugins/*.py`. The
shared mechanics they reuse (merging, permutation alignment, grouping,
profiling, IO, resume) live under `stage2/*.py`. `stage2/orchestrator.py`
itself is `module docstring + imports + a backward-compatibility re-export
block + the _run_assignment driver + the phase-schedule constants + run()`.

The re-export block at the top of `stage2/orchestrator.py` keeps the historical
`moe_compress.stage2_reap_ream._<name>` import paths working for external
callers and the test suite. **Do not remove a name from it** — it is the public
compatibility contract (see §8).

> Historical note: the pre-refactor monolith `stage2_reap_ream.py` was deleted
> in S2-13b; the stage-2 `_framework/` package, the `LegacyAdapter`
> (`legacy_adapter.py`), and the `MOE_STAGE2_LEGACY_LOOP` env-var hatch were all
> deleted earlier in the migration. None of those names describe current
> behaviour — if you find them in older notes, ignore them.

---

## 2. The universal framework objects

Stage 2 plugins are universal `PipelinePlugin`s. The framework lives under
`pipeline/` and `tools/` and is shared by every stage, not just Stage 2.

| Object | Module | Role |
|---|---|---|
| `PipelinePlugin` | `pipeline/plugin.py` | A `@runtime_checkable` structural `Protocol`. A plugin is anything carrying the metadata attributes (`name`, `paper`, `config_key`, `reads`, `writes`, `provides`) plus the two universal-core methods `is_enabled(config)` and `contribute_artifact(ctx)`. It declares **no** phase hooks — phase-hook names are an open vocabulary. |
| `BasePlugin` | `pipeline/plugin.py` | Optional convenience base: metadata defaults + no-op `is_enabled`/`contribute_artifact`. You may subclass it *or* satisfy `PipelinePlugin` structurally — both are first-class. Stage-2 plugins are plain classes (structural conformance). |
| `PipelineContext` | `pipeline/context.py` | Dict-backed, set-once shared-state holder. A second `set(name, …)` raises unless `overwrite=True`. Supports parent/child scopes: the run-scope root is passed to `on_run_setup`/`on_run_teardown`; each layer opens a `child()` scope. `get`/`has` resolve through the parent chain; `set`/`drop`/`keys` are local-scope only. |
| `PluginRegistry` | `pipeline/registry.py` | Ordered, **immutable-after-construction** collection (tuple-backed). Construction order = execution order. No `register()`. `enabled(config)` returns the `is_enabled`-true subset in order; `provides(config)` unions enabled plugins' `provides`; the `dispatch_first` static helper drives single-winner slot hooks. Duplicate plugin `name`s raise at construction. |
| `walk_phases` | `tools/phase_walker.py` | Reflective phase scheduler. Given an ordered phase list and a plugin sequence, for each phase it `getattr`s that name on every plugin and calls it (with `ctx`) if present and callable. **Phase-major / plugin-minor**: every plugin runs phase `A` before any plugin runs phase `B`. |
| `Stage` | `pipeline/stage.py` | Orchestrator-facing Protocol (`stage_id` + `is_enabled` + `run(ctx)`). One whole compression stage. `STAGE2` (`stage2/stage.py`) conforms to it. |

There are **no** stage-2-private framework classes. The deleted
`stage2/_framework/` package (`_framework/base.py`'s `Stage2Plugin`,
`_framework/pipeline.py`'s `Stage2Pipeline`, `_framework/registry.py`) has been
fully replaced by the universal objects above.

### The `PipelinePlugin` contract

```python
name: str            # Unique plugin id, e.g. "reap_scoring"
paper: str           # Citation / one-line role
config_key: str      # Dotted YAML path that gates the plugin
reads: tuple[str,…]  # Context slots the plugin consumes
writes: tuple[str,…] # Context slots the plugin produces
provides: tuple[str,…]  # Named accumulators a calibration pass must run

def is_enabled(self, config: dict) -> bool: ...
def contribute_artifact(self, ctx) -> dict: ...
```

`reads`/`writes` are inspectable metadata for a future static
producer-before-consumer check; `provides` feeds the calibration-pass
multiplexer. Phase hooks (`on_score`, `merge`, `compute_cost`, …) are *not* on
the Protocol — they are discovered reflectively by `walk_phases` and
`dispatch_first`.

---

## 3. The per-layer schedule

Stage 2 does not use a single fixed phase walk. The per-layer schedule is split
into two `walk_phases` halves with the assignment driver wedged between them.
The three schedule constants live in `stage2/orchestrator.py`:

```python
_STAGE2_PRE_ASSIGN_PHASES  = ("on_layer_setup", "on_profile", "on_score")
_STAGE2_POST_ASSIGN_PHASES = ("pre_merge_snapshot", "merge",
                              "post_merge", "on_post_merge",
                              "write_artifacts", "on_layer_teardown")
# Derived back-compat constant — the full 10-tuple with the compound
# "compute_assignment" slot wedged in. NOT walked directly; kept so the
# canonical-order contract test and external importers still see the
# historical schedule.
_STAGE2_LAYER_PHASES = (_STAGE2_PRE_ASSIGN_PHASES
                        + ("compute_assignment",)
                        + _STAGE2_POST_ASSIGN_PHASES)
```

Per layer, `run()` does:

```python
walk_phases(_STAGE2_PRE_ASSIGN_PHASES,  plugins, ctx)   # setup → profile → score
_run_assignment(plugins, ctx)                           # the bump-loop driver
walk_phases(_STAGE2_POST_ASSIGN_PHASES, plugins, ctx)   # snapshot → merge → … → teardown
```

Plus two **run-scope** phases dispatched outside the layer loop, on the root
context: `on_run_setup` (before the first layer) and `on_run_teardown` (after
the last). `_run_assignment` is **not** a `walk_phases` phase — it is an
explicit multi-pass driver (see §4); `compute_assignment` therefore never
appears in a `walk_phases` call.

| Phase | Scope | When | Typical use |
|---|---|---|---|
| `on_run_setup(ctx)` | run | Once, before any layer | Allocate run-scope resources, read `ctx.config`. |
| `on_layer_setup(ctx)` | layer | Start of each layer | Allocate per-layer accumulators (`reap_acc`, `ream_acc`, `perm_cache`, `layer_input_acc`). |
| `on_profile(ctx)` | layer | After setup | Run the calibration forward pass; fill the accumulators. |
| `on_score(ctx)` | layer | After profiling | Publish per-expert saliency — `ReapScoringPlugin` writes `ctx.scores` / `ctx.freq`. |
| *(assignment)* | layer | After scoring | `_run_assignment` runs the bump loop — see §4. |
| `pre_merge_snapshot(ctx)` | layer | Before merge | Snapshot pre-merge expert weights (needed by expert-distill / merge-heal). |
| `merge(ctx)` | layer | Apply the merge | Fuse experts in place, resize the router, run expert distillation. |
| `post_merge(ctx)` | layer | After merge | Per-layer merge-heal, telemetry. |
| `on_post_merge(ctx)` | layer | After `post_merge` | Inter-layer cache invalidation — clear `cov_acc`, `ream_acc`, `layer_input_acc` so the next layer's `on_layer_setup` → `on_profile` sees fresh state. Per SC_STAGE12 §582. |
| `write_artifacts(ctx)` | layer | After post-merge | Write the per-layer partial checkpoint to `partial_dir` (read off `ctx`). |
| `on_layer_teardown(ctx)` | layer | End of each layer | Drop per-layer accumulators, free memory. |
| `on_run_teardown(ctx)` | run | Once, after all layers | Release run-scope resources. |

A plugin that does not implement a hook is silently skipped at that phase
(`walk_phases` does a `getattr` + `callable` guard) — no base class, no no-op
override needed.

---

## 4. The `_run_assignment` slots

`_run_assignment(plugins, ctx)` (in `stage2/orchestrator.py`) owns the
child→centroid assignment. It is the **bump-loop driver**: it runs up to
`1 + (n_experts - target)` attempts — one initial attempt plus one bump per
additional kept expert — re-solving the assignment each time a gate trips:

- **`b_fail`** (feasibility): `n_noncentroids > n_centroids × max_group_cap`.
  The per-centroid cap cannot absorb every non-centroid.
- **`c_fail`** (quality): the mean assigned cost exceeds the running per-layer
  mean by `(1 + cost_sigma)`. Requires ≥4 prior layers of cost history before
  it can fire.

When either gate trips the driver bumps `effective_target` by
`max(1, ceil(effective_target × cost_bump_ratio))` and retries. If the loop
exhausts at `effective_target == n_experts`, it falls back to the last
above-threshold assignment (`c_fail`) or to a zero-merge "keep all non-protected
experts as centroids" fallback (`b_fail`). The driver owns the bump control
flow, orphan-promotion grouping, and the final `ctx.set` of every per-layer
output slot (`ream_centroid_ids`, `ream_noncentroid_ids`, `assignment`,
`delta`, `grouped`, …).

Inside each non-`b_fail` bump iteration, `_run_assignment` reaches the cost /
solver / refinement logic through **five fine-grained slot hooks**, dispatched
over the enabled plugin list:

| Slot | Dispatch | Contract |
|---|---|---|
| `select_alignment(ctx)` | **single-winner** `dispatch_first` | First non-`None` wins. The capacity-utilization gate; runs **before** `compute_cost` and publishes `effective_cost_alignment` / `effective_cost_asymmetric` / `capacity_util_value` to `ctx` for the cost slot to read back. |
| `compute_cost(ctx)` | **single-winner** `dispatch_first` | First non-`None` wins. Returns the REAM cost matrix `delta`. |
| `apply_cost_mask(ctx, cost)` | **single-winner** `dispatch_first` | First non-`None` wins. Returns `(masked_cost, mask_info)`, or `None` to leave the matrix unmasked. |
| `solve_assignment(ctx, cost)` | **single-winner** `dispatch_first` | First non-`None` wins. Returns the child→centroid assignment list. |
| `refine_assignment(ctx, assignment, delta)` | **CHAIN** | Every enabled plugin's hook runs, in registry order. Each refiner threads `(assignment, delta, info)` forward to the next; a refiner whose own gate is off this layer returns `None` and is skipped. There is **no** `dispatch_first` early-return here. |

`dispatch_first` (single-winner) walks the enabled plugins in **registry
order** and takes the first non-`None` return — so registry order decides which
plugin wins a slot. `refine_assignment` is the only slot that is a chain: both
`TwoOptRefinePlugin` and `EmRefinePlugin` may run, two-opt first.

`select_alignment` / `compute_cost` / `solve_assignment` are asserted non-`None`
inside the driver — exactly one enabled plugin must service each. `registry`
gating (§5) guarantees that.

---

## 5. The live plugin roster

`run()` builds one `PluginRegistry([...])`. **Registration order is execution
order** and it is load-bearing — `walk_phases` is phase-major (so for a shared
phase, earlier-registered plugins run first), and `dispatch_first` takes the
first enabled winner in registry order. The roster, in order:

| # | Plugin | `is_enabled` gate | Hooks / slots | Role |
|---|---|---|---|---|
| 1 | `ReapScoringPlugin` | always on | `on_layer_setup`, `on_score` | Allocates `reap_acc`; publishes per-expert `scores` / `freq`. Registered first so its `on_layer_setup` runs before `LayerMergePlugin.on_profile` reads `reap_acc`. |
| 2 | `CapacityGatePlugin` | always on | `select_alignment` slot | SLACK-vs-TIGHT capacity-utilization gate. Runs before the cost plugins; publishes the effective cost-alignment decision. |
| 3 | `ReamCostPrePlugin` | `cost_alignment == "pre"` | `compute_cost` slot | REAM symmetric pre-alignment cost matrix. |
| 4 | `ReamCostPostPlugin` | `cost_alignment == "post"` | `compute_cost` slot | REAM post-alignment whitened-residual cost matrix. |
| 5 | `OutputSpaceCostPlugin` | `cost_alignment == "output"` | `compute_cost` slot | Direction C output-space merge-cost matrix. |
| 6 | `SkipMergeFloorPlugin` | `skip_merge_percentile < 100.0` | `apply_cost_mask` slot | Direction B skip-merge percentile mask. Registered after the cost plugins so it wins the mask slot. |
| 7 | `GreedySolverPlugin` | `assignment_solver == "greedy"` | `solve_assignment` slot | Greedy descending-saliency assignment. |
| 8 | `HungarianSolverPlugin` | `assignment_solver == "hungarian"` | `solve_assignment` slot | Rectangular Hungarian (scipy). |
| 9 | `McfSolverPlugin` | `assignment_solver == "mcf"` | `solve_assignment` slot | Capacitated min-cost-flow (OR-Tools). |
| 10 | `SinkhornSolverPlugin` | `assignment_solver == "sinkhorn"` | `solve_assignment` slot | Capacitated entropy-regularized OT (Sinkhorn-Knopp). |
| 11 | `AutoSolverPlugin` | `assignment_solver == "auto"` | `solve_assignment` slot | Auto-pick: Hungarian in slack, MCF in tight. |
| 12 | `TwoOptRefinePlugin` | `two_opt_refine` | `refine_assignment` chain | Direction D greedy + 2-opt local refinement. `is_enabled` gates on `two_opt_refine` alone; the greedy-only guard is enforced inside `refine_assignment` (a non-greedy solver logs a one-shot warning instead). |
| 13 | `EmRefinePlugin` | `em_refinement_rounds > 0` | `refine_assignment` chain | EM refinement loop. Chained after two-opt. |
| 14 | `LayerMergePlugin` | always on | `on_layer_setup`, `on_profile`, `merge`, `post_merge`, `write_artifacts`, `on_layer_teardown` | The per-layer **merge spine** — owns the six live `walk_phases` phase hooks and the run-scope scratchpad (`cov_acc`, `merge_map`, `_layer_mean_costs`, `partial_dir`, `blacklist`, the bump-loop knobs `_run_assignment` reads off the instance). |
| 15 | `ExpertDistillPlugin` | `expert_distill_steps > 0` | `pre_merge_snapshot`, `merge` | Per-merge-group expert distillation. Registered after the spine so its `merge` lands after `LayerMergePlugin`'s in-place merge. |
| 16 | `MergeHealPlugin` | `merge_heal_enabled` | `pre_merge_snapshot`, `post_merge` | Per-layer merge-heal by self-distillation toward the pre-merge output. Registered last so its `post_merge` lands after the spine's `bank.select` + router resize. |

Exactly one of plugins 3–5 (cost) and exactly one of 7–11 (solver) is enabled
per run — `registry.enabled(config)` drops the rest. `SkipMergeFloorPlugin`,
`TwoOptRefinePlugin`, `EmRefinePlugin`, `ExpertDistillPlugin` and
`MergeHealPlugin` are all opt-in and inert by default. `ReapScoringPlugin`,
`CapacityGatePlugin` and `LayerMergePlugin` are always on.

Why order matters, concretely:
- The cost plugins (3–5) sit **between** `ReapScoringPlugin` and the merge
  spine and **after** `CapacityGatePlugin`, so the gate publishes its decision
  before `compute_cost` reads it back.
- `SkipMergeFloorPlugin` (6) is **after** the cost plugins so it wins the
  `apply_cost_mask` slot (the cost plugins do not service it).
- The solvers (7–11) are **after** the mask so they see the masked matrix.
- The refiners (12–13) are **after** the solvers; their registry order is the
  chain order — two-opt then EM.
- `ExpertDistillPlugin` (15) and `MergeHealPlugin` (16) are **after**
  `LayerMergePlugin` so, for the shared `merge` / `post_merge` phases, their
  hooks run after the spine's (phase-major / plugin-minor).

---

## 6. How to add a new paper as a `PipelinePlugin`

1. **Create the module.** `stage2/plugins/<name>.py`.
2. **Write a plain class conforming to `PipelinePlugin`.** Declare the metadata
   class attrs, an `is_enabled`, a `contribute_artifact`, and only the phase
   hook(s) / slot(s) the algorithm needs — there is **no base class to
   subclass** (you may subclass `BasePlugin` for the no-op defaults if you
   prefer, but most stage-2 plugins are structural).
3. **Gate it** via `is_enabled(self, config)` — read your dotted `config_key`
   out of `config["stage2_reap_ream"]` and return a `bool`.
4. **Wire the config flag.** Add the flag under `stage2_reap_ream.*` and
   validate it at the top of `run()` — fail-fast at the config boundary, the
   way `assignment_solver` and `cost_alignment` are validated.
5. **Register it.** Add the plugin to the `PluginRegistry([...])` list in
   `run()` at the position its hooks/slots require (see §5 on ordering).
6. **Avoid import cycles.** A plugin module must **not** do a module-top
   `from ..orchestrator import ...` — `orchestrator.py` re-imports from
   `stage2/`, so a top-level back-import deadlocks. Import siblings under
   `stage2/` directly, or use a function-scope late import.

Worked example — a plugin that adjusts saliency on the `on_score` phase, gated
by a boolean flag:

```python
"""Example plugin — <one-line paper summary>."""
from __future__ import annotations

from ...pipeline.context import PipelineContext


class ExampleScorePlugin:
    """Adjusts per-expert saliency before assignment (example skeleton)."""

    name = "example_score"
    paper = "Example: per-expert saliency adjustment."
    config_key = "stage2_reap_ream.example_score_enabled"
    reads: tuple[str, ...] = ("scores", "freq")
    writes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        s2 = config.get("stage2_reap_ream", {})
        return bool(s2.get("example_score_enabled", False))

    def contribute_artifact(self, ctx) -> dict:
        return {}

    def on_score(self, ctx: PipelineContext) -> None:
        # ctx.scores / ctx.freq are published here by ReapScoringPlugin;
        # mutate the objects in place — set-once forbids re-binding the slot.
        ...
```

Matching config-validation snippet in `run()` (next to the existing
`assignment_solver` / `cost_alignment` validation):

```python
    example_score_enabled: bool = bool(s2.get("example_score_enabled", False))
    # ... validate any numeric companions here, fail-fast on bad values ...
```

Matching registration snippet in the `run()` `PluginRegistry([...])` build:

```python
    registry = PluginRegistry([
        ReapScoringPlugin(),
        ExampleScorePlugin(),       # after scoring, before the cost plugins
        CapacityGatePlugin(...),
        # ... the rest of the roster ...
    ])
```

For a slot-style plugin (cost / solver / mask / alignment), implement the slot
hook instead of a phase hook — return the slot value, or `None` to defer to the
next plugin (`dispatch_first` takes the first non-`None`). For a refinement
plugin, implement `refine_assignment(ctx, assignment, delta)` and return
`(assignment, delta, info)` (or `None` to decline) — it is a chain, so be a
well-behaved link.

> The retired `Stage2Plugin` base class, its `enabled_by` tuple, and the
> `stage2/_framework/` package no longer exist. If you are porting an old
> plugin, drop `enabled_by` for an explicit `is_enabled` and import the
> universal `PipelineContext` from `pipeline/context.py`.

---

## 7. Shared stage-2 modules — reuse, do not duplicate

| Module | Reusable helpers |
|---|---|
| `stage2/merging.py` | `_merge_experts_inplace`, `_resize_router_for_kept_experts` |
| `stage2/permutation_align.py` | `_permutation_align_to_centroid`, `_aligned_whitened_residual`, `_PermAlignCache` |
| `stage2/grouping.py` | `_build_grouped_from_assignment`, `_promote_orphans`, `_apply_skip_merge_floor` |
| `stage2/profiling.py` | `_profile_layer`, `_LayerInputAccumulator` |
| `stage2/shared_io.py` | Covariance / heal-weights / merge-JSON IO helpers (`_save_covariance`, `_load_heal_weights`, `_write_merge_json`, …) |
| `stage2/resume.py` | `discover_completed_layers`, `ResumedLayerRecord` |

If your algorithm needs to merge experts, align permutations, build groups,
profile a layer, or write artifacts — call the canonical helper above. Holding
a private copy is the bug class the monkeypatch-drift tests exist to catch.

---

## 8. The `orchestrator.py` re-export block

`stage2/orchestrator.py` opens with a backward-compatibility re-export block:
every `_`-prefixed Stage 2 internal — pulled from `stage2/resume.py`,
`shared_io.py`, `profiling.py`, `permutation_align.py`, `merging.py`,
`grouping.py`, and the `stage2/plugins/*` modules — is re-imported under the
`orchestrator` namespace. This keeps the historical
`moe_compress.stage2_reap_ream._<name>` import paths working for external
callers (`run_pipeline.py`, `run_ablations.py`, `budget_retune.py`,
`stage4_eora.py`) and for the test suite, which imports many of these internals
directly. `_HealConfig` is additionally constructed inside `run()` itself.

**Removing any name from that block is a breaking change.** It is the public
compatibility surface — treat it as API.

---

## 9. Testing

- **Per-plugin tests.** Each plugin has a `test_stage2_plugin_<name>.py`
  (`test_stage2_plugin_reap_scoring.py`, `…_layer_merge.py`, `…_capacity_gate.py`,
  `…_ream_cost.py`, `…_ream_cost_post.py`, `…_output_space_cost.py`,
  `…_skip_merge_floor.py`, `…_solvers.py`, `…_two_opt_refine.py`,
  `…_em_refine.py`, `…_refine.py`, `…_expert_distill.py`, `…_merge_heal.py`).
  Mandatory checks: the `is_enabled` truth table (flag on / off / missing key);
  the `PipelinePlugin` contract (`isinstance(plugin, PipelinePlugin)`, the
  metadata attrs); re-import identity (`orchestrator._fn is <plugin>._fn`) so an
  accidental divergence is caught; the monkeypatch-drift guard (the plugin
  delegates to the canonical shared helper, not a stale copy); and hook
  behaviour — call the hook directly with a constructed `PipelineContext`
  (a `child()` scope for per-layer hooks) and assert the `ctx` mutation.
- **Phase-order tests.** `test_stage2_pipeline_run_layer.py` asserts the
  `walk_phases` schedule: `_STAGE2_PRE_ASSIGN_PHASES` then
  `_STAGE2_POST_ASSIGN_PHASES` visit every phase once per plugin in canonical
  order, `compute_assignment` is **not** a walked phase, and
  `_STAGE2_LAYER_PHASES` is the derived 10-tuple. It also covers `_run_assignment`
  and `LayerMergePlugin` wiring.
- **`Stage` conformance.** `test_stage2_stage.py` asserts `STAGE2.stage_id`,
  `STAGE2.is_enabled`, and `isinstance(STAGE2, Stage)`.
- **Byte-identical resume gate.** `test_smoke_stage2_resume.py` crashes a run
  after layer 0, resumes, and asserts the resumed run's `merge_map.json` is
  identical to a clean run's — the determinism contract for crash-resume.
