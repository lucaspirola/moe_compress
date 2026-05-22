# Stage 1 Plugins — Authoring Guide

Stage 1 is a thin orchestrator (`stage1/orchestrator.py`) that runs an ordered
sequence of single-paper plugins. Each file under this directory implements
exactly one paper / detector / phase. Adding a new paper to Stage 1 means
**dropping a file here, registering it in `__init__.py`, and adding one
explicit `run()` call in `orchestrator.py` at the correct phase position** —
the orchestrator is the single place the pipeline sequence is declared, by
design (see §5.1). Plugins never import each other; all shared state flows
through one `Stage1Context` instance threaded through every phase. This guide
is a working document: read it top-to-bottom once, then use the checklist (§6)
and the copy-paste template (§7) as a recipe. The checklist tells you exactly
which lines to add to `orchestrator.py` — you do not need to reverse-engineer
its body to add a plugin.

---

## 1. Architecture overview

Six moving parts. Know which one your new code touches.

- **`stage1/orchestrator.py`** — the thin phase sequencer. Owns only *glue*:
  the accumulator factory (`_build_accumulator`), artifact assembly
  (`_write_artifacts`), and telemetry (`_emit_telemetry`). It threads one
  `Stage1Context` through Phase A → calibration pass → 4 detectors → ablation
  → CKA → GRAPE → artifact write. It contains **no** phase logic — every phase
  lives inside a plugin. The orchestrator invokes each plugin via an
  **explicit, hand-written `run()` call in a fixed sequence** — it does *not*
  iterate the manifest to call `run()`. This is by design: the phase sequence
  is load-bearing, so the calls are spelled out in one reviewable place
  (see §5.1).

- **`stage1/plugins/__init__.py`** — the `STAGE1_PLUGIN_MANIFEST` tuple. The
  orchestrator feeds this tuple straight to `PluginRegistry`, which it uses
  for `enabled()` resolution and `provides()`, and to build the
  `by_name` lookup table its explicit `run()` calls index into. The manifest
  does **not** drive `run()` invocation — that is the orchestrator's explicit
  call sequence (§5.1). By convention the manifest tuple is ordered to match
  that call sequence so the two read consistently, but it is the explicit
  calls that execute. Adding a plugin = one import line + one tuple entry +
  one `__all__` entry here, **plus** the explicit `run()` call in
  `orchestrator.py` (§6 step 5).

- **`pipeline/plugin.py`** — the universal plugin framework. Defines the
  `PipelinePlugin` Protocol (the contract every plugin satisfies — §4).
  **`pipeline/registry.py`** defines `PluginRegistry` (ordered,
  immutable-after-construction; exposes `enabled()` and `provides()`). The
  registry rejects duplicate plugin names at construction time.

- **`stage1/context.py` → `Stage1Context`** — the typed shared-state holder.
  Backed by a `dict[str, Any]` with strict `get` / `set` / `drop` / `has`
  accessors (defined on the `PipelineContext` base in `pipeline/context.py`).
  Reads of an unwritten slot raise `KeyError`; `set` is **set-once** (pass
  `overwrite=True` to replace a binding). Shared mutable state (e.g. a
  `CandidateBag`) is `set` once and then mutated in place.

- **`_framework/calibration_engine.py` → `CalibrationEngine`** — the shared
  Phase-B profiling driver. Plugins declare which named accumulators they need
  via their `provides` tuple; the orchestrator maps each name to a concrete
  `(accumulator, HookSpec)` pair via `_build_accumulator`, and the engine wires
  **all** hooks in **one** forward pass over the calibration batches.

- **`_framework/candidates.py` → `CandidateBag`** and
  **`_framework/artifact_assembly.py` → `ArtifactBuilder`** — the candidate-union
  data structure (the 4 detectors all `add()` into one shared bag) and the
  `stage1_blacklist.json` assembler (validates the 7-top-level-key schema —
  see `REQUIRED_BLACKLIST_TOP_LEVEL_KEYS`). **`_framework/safe_json.py` →
  `safe_float`** turns NaN / ±Inf into JSON `null`; call it on every float
  inside `contribute_artifact` before the float enters a JSON fragment.

---

## 2. The contract diagram

Data flow, top to bottom. This mirrors `orchestrator.run` STEP 3-11. Use it to
see where a new plugin slots in.

```
                       Stage1Context  (one instance threaded through all phases)
                              |
   +--------------------------+-------------------------------------------------+
   |                                                                           |
Phase A: ma_detection.run()         -- own dedicated early-exit pass            |
   |  writes: L, residual_growth, moe_output_growth, moe_output_max             |
   v                                                                           |
setup() on setup-capable plugins    -- sink_token.setup() builds                |
   |  writes: sink_acc                  the SinkTokenRoutingAccumulator         |
   v                                                                           |
CalibrationEngine  (ONE shared forward pass)                                   |
   |  registry.provides(config) -> {downproj_max,                               |
   |      output_reservoir, sink_routing}                                       |
   |  orchestrator._build_accumulator(name) -> (accumulator, HookSpec)          |
   |  writes: max_acc, output_acc  (sink_acc already on ctx)                    |
   v                                                                           |
Phase C detectors (orchestrator STEP 8 call order), all mutate ONE CandidateBag:|
   three_way_and.run() -> bag.add(l,e,"phase_c")    writes p995/a_max...        |
   aimer.run()         -> bag.add(l,e,"aimer")                                  |
   sink_token.run()    -> bag.add(l,e,"sink_token")                             |
   magnitude_topk.run()-> bag.add(l,e,"magnitude_topk")                         |
   |  candidates = bag.to_provenance_dict()  -> ctx["candidates"]               |
   v                                                                           |
Phase D: ablation_filter.run()      -- own ablation forward pass                |
   |  writes: blacklist, candidate_deltas, baseline_nll                         |
   v                                                                           |
Phase E: cka_distance.run()         writes: D_matrices                          |
   |  (orchestrator then drop()s output_acc)                                    |
   v                                                                           |
Phase F: grape_merge.run()          writes: per_layer_target_experts ...        |
   |                                                                           |
   v                                                                           v
_write_artifacts():  ArtifactBuilder assembles stage1_blacklist.json
   - orchestrator-owned top-level keys: blacklist, per_expert_max,
     config, blacklist_provenance
   - plugin fragments: dual_signal (ma_detection), aimer, sink_token
   - whole-file contributors: stage1_ablation_filter.json (ablation_filter),
     stage1_budgets.json (grape_merge)
```

---

## 3. Phases and current manifest

`STAGE1_PLUGIN_MANIFEST` (from `__init__.py`). The manifest tuple is ordered,
by convention, to match the orchestrator's explicit `run()` call sequence —
but it is those explicit calls in `orchestrator.py`, not the tuple, that
execute the pipeline (§5.1). The orchestrator's call order is:

| # | Plugin class | `name` | Phase | Notes |
|---|---|---|---|---|
| 1 | `MADetectionPlugin` | `ma_detection` | A | Own forward pass; always enabled |
| 2 | `ThreeWayAndPlugin` | `three_way_and` | C1 | Mandatory paper criterion, no flag |
| 3 | `AimerDetectorPlugin` | `aimer` | C2 | AIMER bottom-pct detector |
| 4 | `SinkTokenDetectorPlugin` | `sink_token` | C3 | Has `setup()`; sink-token routing |
| 5 | `MagnitudeTopkPlugin` | `magnitude_topk` | C4 | Magnitude top-K detector |
| 6 | `AblationFilterPlugin` | `ablation_filter` | D | Own ablation forward pass; always enabled |
| 7 | `CKADistancePlugin` | `cka_distance` | E | CKA distance matrices |
| 8 | `GrapeMergePlugin` | `grape_merge` | F | GRAPE greedy merge |

A new detector goes among the Phase-C block (positions 2-5). A new
post-processing step goes after `ablation_filter`. In both cases you add the
plugin to this tuple **and** add a matching explicit `run()` call to
`orchestrator.py` at the same sequential position — see §6 step 5.

---

## 4. The `PipelinePlugin` Protocol contract

A plugin is a plain class — **no base class to inherit**. It structurally
satisfies the `PipelinePlugin` Protocol from `pipeline/plugin.py`:

```python
@runtime_checkable
class PipelinePlugin(Protocol):
    name: str
    paper: str
    config_key: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    provides: tuple[str, ...]

    def is_enabled(self, config: dict) -> bool: ...
    def run(self, ctx: "Any") -> None: ...
    def contribute_artifact(self, ctx: "Any") -> dict: ...
```

Every member, with its exact type and meaning:

| Member | Type | Meaning |
|---|---|---|
| `name` | `str` | Unique plugin id, e.g. `"ma_detection"`. Duplicate names in the manifest raise `ValueError` at `PluginRegistry` construction. Used as the `by_name` key in the orchestrator. |
| `paper` | `str` | One-line citation. Informational only. |
| `config_key` | `str` | Dotted path into the YAML where the plugin's sub-config / flag lives, e.g. `"stage1_grape.super_expert_detection.aimer_enabled"`. Documentation / introspection only — `is_enabled` reads the config itself; nothing parses this string. |
| `reads` | `tuple[str, ...]` | `Stage1Context` slots the plugin consumes. Honesty contract — enables a future static check that nothing reads a slot no prior plugin wrote. |
| `writes` | `tuple[str, ...]` | `Stage1Context` slots the plugin produces. List a slot here even if it is mutated in place (not rebound) — e.g. `candidate_bag`. |
| `provides` | `tuple[str, ...]` | Named accumulators the shared `CalibrationEngine` must run for this plugin. Empty `()` if the plugin needs no Phase-B data (it runs its own forward pass, or is weight-only). |
| `is_enabled(config) -> bool` | method | Whether the plugin is "on" per config. **See §5.1 ("`is_enabled` gates `provides`, NOT `run`") — `run()` is always called; `is_enabled` only drives `provides`.** |
| `run(ctx) -> None` | method | The phase logic. Reads slots in `reads`, writes slots in `writes`. Must produce well-formed output even on the disabled path. |
| `contribute_artifact(ctx) -> dict` | method | A JSON-ready dict — a *fragment* merged into `stage1_blacklist.json`, a *whole-file* payload, or `{}`. See subtlety §5.2. |

### Optional `setup` method

Not on the Protocol. If a plugin needs to build an accumulator *before* the
calibration pass, it implements:

```python
def setup(self, ctx: Stage1Context) -> None: ...
```

The orchestrator calls it via `getattr(plugin, "setup", None)` on **every**
plugin before the calibration engine runs (STEP 4, before STEP 5). Only
`sink_token` uses it today — it builds the `SinkTokenRoutingAccumulator` onto
`ctx["sink_acc"]`, which the accumulator factory then reads back so the engine
feeds the *same* instance the plugin later reads.

---

## 5. Three behavioural subtleties a new author MUST know

These are non-obvious and have bitten the refactor.

### 5.1 The orchestrator invokes plugins by explicit, sequential `run()` calls

`orchestrator.run()` does **not** loop over the manifest to invoke plugins.
It calls each plugin's `run()` by an explicit, hand-written statement, in a
fixed sequence — the orchestrator builds a `by_name = {p.name: p for p in
registry}` lookup and then indexes it by literal plugin name. This is
**intentional**: the phase sequence (Phase A → setup → calibration pass →
4 detectors → ablation → CKA → GRAPE → artifacts) is load-bearing, and
spelling the calls out in one place makes the whole pipeline ordering
reviewable at a glance.

The actual sequence, as written in `orchestrator.run()`:

1. **STEP 3 — Phase A:** `by_name["ma_detection"].run(ctx)` — its own
   dedicated early-exit forward pass.
2. **STEP 4 — setup:** loops the registry calling `getattr(plugin, "setup",
   None)` on every plugin (only `sink_token` has one today). This *is* a loop,
   but it calls the optional `setup()`, not `run()`.
3. **STEP 5-7 — calibration:** builds the `CalibrationEngine`, registers the
   accumulators from `registry.provides(config)`, runs the single
   shared forward pass, finalizes the accumulators.
4. **STEP 8 — the 4 detectors:** a `for name in ("three_way_and", "aimer",
   "sink_token", "magnitude_topk"): by_name[name].run(ctx)` loop over a
   **hard-coded literal tuple of names** — not the manifest. The tuple's order
   is the detector execution order.
5. **STEP 9 — ablation:** `by_name["ablation_filter"].run(ctx)`.
6. **STEP 10 — CKA then GRAPE:** `by_name["cka_distance"].run(ctx)`, then the
   orchestrator `drop()`s `output_acc`, then `by_name["grape_merge"].run(ctx)`.
7. **STEP 11 — artifacts/telemetry:** `_write_artifacts` and `_emit_telemetry`.

`STAGE1_PLUGIN_MANIFEST` is used to construct the `PluginRegistry` and thereby
to drive `enabled()`, `provides()`, and the `by_name` table — it
is **not** iterated to call `run()`. Adding a plugin therefore always includes
adding its explicit `run()` call to `orchestrator.py` (§6 step 5).

### `is_enabled` gates `provides`, NOT `run`

Every `run()` call above is **unconditional** — the orchestrator never checks
`is_enabled` before calling `run()`. A plugin must short-circuit *internally*
on its own flag — e.g. `aimer.run` returns empty results when
`aimer_enabled=False`; `ablation_filter.run` uses the candidate set verbatim
when disabled. `is_enabled` is consumed *only* by
`PluginRegistry.provides` — so a disabled plugin does not make the
`CalibrationEngine` wire a hook nobody reads. Your new plugin's `run()` must
therefore produce well-formed (possibly empty) outputs even on the disabled
path.

### 5.2 `contribute_artifact` has three modes

- **(a) Fragment** — returns a dict merged under a top-level key of
  `stage1_blacklist.json` by `ArtifactBuilder`: `ma_detection` → `dual_signal`,
  `aimer` → `aimer`, `sink_token` → `sink_token`.
- **(b) Whole-file** — returns the complete payload of its own file:
  `ablation_filter` → `stage1_ablation_filter.json`, `grape_merge` →
  `stage1_budgets.json`. The orchestrator writes the payload directly.
- **(c) No contribution** — returns `{}`: `three_way_and`, `magnitude_topk`,
  `cka_distance`.

A plugin **never writes to disk itself** — the orchestrator owns all writes.

### 5.3 Set-once context + in-place mutation

`ctx.set(name, ...)` raises `KeyError` on a second write unless
`overwrite=True`. Shared mutable state — the `CandidateBag` — is `set` **once**
by the orchestrator, and every detector `add()`s into the same instance. List
such a slot in **both** `reads` and `writes` (you read the instance, you mutate
it). For a private bookkeeping slot that only your own `contribute_artifact`
later consumes, still list it in `writes` so the contract stays honest — see
`ablation_filter`, which stashes config onto the ctx for its own whole-file
payload.

---

## 6. Checklist — how to add a new plugin

1. **Create the file.** `stage1/plugins/<your_plugin>.py`. One paper per file.
   Module docstring: the paper citation + which phase it belongs to.

2. **Implement the Protocol.** Write a plain class declaring all six
   class-level attributes (`name`, `paper`, `config_key`, `reads`, `writes`,
   `provides`) and the three methods (`is_enabled`, `run`,
   `contribute_artifact`). No base class. Import the Protocol only for the
   type-checker: `from ..pipeline.plugin import PipelinePlugin  # noqa: F401`.

3. **Declare `provides` if you need calibration data.** If your plugin
   reads per-expert activations or router logits, name the accumulator(s) you
   need. Three names exist today: `"downproj_max"`, `"output_reservoir"`,
   `"sink_routing"`. If you need a *new* accumulator name, see step 6. If your
   plugin is weight-only or runs its own forward pass, use `provides = ()`.

4. **Register in `STAGE1_PLUGIN_MANIFEST`.** In `stage1/plugins/__init__.py`:
   add the `from .<your_plugin> import <YourPlugin>` import, insert the
   `<YourPlugin>()` entry in the `STAGE1_PLUGIN_MANIFEST` tuple **at the
   correct phase position** (a detector among positions 2-5; a post-processing
   step after `ablation_filter`), and add the class name to `__all__`. This
   registers the plugin with the `PluginRegistry` (so it participates in
   `enabled()`, `provides()`, and the `by_name` lookup). It does
   **not**, on its own, make the plugin run — see step 5.

5. **Add the explicit `run()` call to `orchestrator.py`.** The orchestrator
   invokes plugins by explicit, hand-written calls in a fixed sequence (§5.1),
   not by iterating the manifest — so registering in step 4 alone never
   executes your plugin. Add the call at the correct sequential position for
   your plugin's phase. This is the normal, expected way to add a plugin: the
   orchestrator is the one place the pipeline sequence is declared, and that is
   intentional. Where to put the call:
   - **A new candidate detector (Phase C):** add the plugin's `name` to the
     hard-coded literal tuple in STEP 8 —
     `for name in ("three_way_and", "aimer", "sink_token", "magnitude_topk")` —
     at the position matching its manifest slot. The detectors all `add()`
     into the one shared `CandidateBag`, so any position inside this group is
     valid; keep it consistent with the manifest order.
   - **A new post-processing step:** add an explicit
     `by_name["<your_plugin>"].run(ctx)` line. Put it after the STEP 9
     `ablation_filter` call and around the STEP 10 `cka_distance` /
     `grape_merge` calls, at the point your phase's `reads`/`writes`
     dependencies are satisfied — e.g. after `ablation_filter` if you consume
     `blacklist`, before `grape_merge` if you must influence the budgets.
   - **A new pre-calibration / Phase-A-like pass:** call its `run()` near
     STEP 3, before the calibration engine pass.

6. **Wire any new accumulator name into the orchestrator's factory.** If step 3
   introduced a name not in `{downproj_max, output_reservoir, sink_routing}`,
   add a branch to `_build_accumulator` in `stage1/orchestrator.py` mapping the
   name to a concrete `(accumulator, HookSpec)`. Pick the right `HookKind`(s)
   from `_framework/calibration_engine.py`: `DOWN_PROJ`, `EXPERT_INPUT`,
   `EXPERT_INTERMEDIATE`, `EXPERT_GATE_UP_OUT`, `ROUTER_LOGITS_PER_BATCH`,
   `INPUT_IDS_PER_BATCH`. A `HookSpec` carries `kinds` (a `frozenset` of
   `HookKind`), an optional per-expert `expert_callback`, and an optional
   `per_batch` callback. This is the **one** place where adding a new
   accumulator touches the orchestrator's calibration glue — by design.

7. **Write the plugin's test file.**
   `max_quality/tests/test_stage1_plugin_<your_plugin>.py`, following the
   existing `test_stage1_plugin_*.py` files. Cover: the disabled path produces
   well-formed empty output, the enabled path produces correct output, and
   `contribute_artifact` returns the right schema. The plugin classes are also
   covered by `test_stage1_e2e.py` and the golden snapshot.

8. **If your plugin contributes to `stage1_blacklist.json`, return a
   fragment** from `contribute_artifact` AND add its top-level key name to
   `REQUIRED_BLACKLIST_TOP_LEVEL_KEYS` in `_framework/artifact_assembly.py` AND
   wire `builder.add_fragment("<key>", plugin.contribute_artifact(ctx))` into
   the orchestrator's `_write_artifacts`. If instead your plugin emits a
   whole-file artifact, wire its `contribute_artifact` call + a
   `save_json_artifact(...)` into `_write_artifacts` the way `ablation_filter`
   and `grape_merge` are wired. **This changes the artifact schema** — see §9
   (the golden-snapshot note) and the `test_blacklist_schema_seven_top_level_keys`
   test, which is locked to the current 7 keys.

---

## 7. Copy-paste plugin template

A complete skeleton — paste it, rename it, fill it in. Accurate against the
current `PipelinePlugin` Protocol.

```python
"""Phase X — <paper name> detector.

Paper: <citation>. One paper per file.
"""
from __future__ import annotations

import logging

from ..pipeline.plugin import PipelinePlugin  # noqa: F401  (Protocol, type-checkers only)
from ..context import Stage1Context
# Allowed imports: .._framework.*, ...utils.*, ..context, stdlib, torch/numpy.
# NEVER:  from .other_plugin import ...   (the architectural invariant — see §8)

log = logging.getLogger(__name__)


class MyDetectorPlugin:
    """One-paragraph summary: what this plugin detects / does."""

    name: str = "my_detector"
    paper: str = "<citation>"
    config_key: str = "stage1_grape.super_expert_detection.my_detector_enabled"
    reads: tuple[str, ...] = ("max_acc", "L", "candidate_bag", "config")
    writes: tuple[str, ...] = ("candidate_bag",)
    provides: tuple[str, ...] = ("downproj_max",)  # or () if none

    def is_enabled(self, config: dict) -> bool:
        """Reflect the YAML flag. NOTE: this gates provides,
        NOT run() -- run() is always called and must short-circuit itself."""
        se = config.get("stage1_grape", {}).get("super_expert_detection", {})
        return bool(se.get("my_detector_enabled", True))

    def run(self, ctx: Stage1Context) -> None:
        """Phase logic. Read ctx slots in `reads`, write those in `writes`.
        Must produce well-formed output even when is_enabled() is False."""
        config = ctx.get("config")
        if not self.is_enabled(config):
            return  # disabled path -- leave the shared bag untouched
        candidate_bag = ctx.get("candidate_bag")
        # ... detection logic ...
        # candidate_bag.add(layer_idx, expert_idx, "my_detector")

    def contribute_artifact(self, ctx: Stage1Context) -> dict:
        """Return a JSON fragment, a whole-file payload, or {} (no
        contribution). Scrub non-finite floats with safe_float()."""
        return {}


# Optional -- only if the plugin builds an accumulator before Phase B:
#   def setup(self, ctx: Stage1Context) -> None: ...
```

---

## 8. The architectural invariant

> **Plugins NEVER import each other.** No file under `stage1/plugins/` may
> contain `from .<other_plugin> import ...`. A plugin imports only from
> `_framework/` (the cross-stage framework), `utils/` (cross-stage utilities),
> `stage1/context` (the shared-state *type* — importing `Stage1Context` is the
> prescribed pattern, not a violation), the standard library, and third-party
> libraries (torch, numpy). All inter-plugin data flow goes through the one
> `Stage1Context` instance — never a direct import.
>
> If two plugins need the same helper, it belongs in `_framework/` or `utils/`,
> not in one plugin imported by the other. CI enforces this:
>
> ```bash
> for p in max_quality/src/moe_compress/stage1/plugins/*.py; do
>   grep -E "from .*stage1\.plugins" "$p" && echo "VIOLATION in $p"
> done
> ```
>
> This grep must produce empty output.

`stage1/plugins/__init__.py` imports the 8 plugin classes to build the
manifest — that is the manifest module's job, not a plugin importing a plugin,
so it is not a violation.

---

## 9. The golden-snapshot regression note

`max_quality/tests/test_stage1_golden_snapshot.py` runs the full Stage 1 on the
tiny-model fixture and asserts the three artifacts — `stage1_blacklist.json`,
`stage1_budgets.json`, `stage1_ablation_filter.json` — are **byte-identical** to
the checked-in snapshots under `max_quality/tests/golden/stage1/`.

Adding a new plugin that is **enabled by default and changes any artifact** (a
new detector adds candidates → the blacklist / budgets change; a new fragment
adds a top-level key) will make this test fail **correctly**. The fix is a
deliberate, reviewed snapshot regeneration — **never silence the test**.

Regeneration workflow (from the snapshot test's docstring):

```bash
# 1. regenerate the three golden files
MOE_REGEN_GOLDEN=1 pytest max_quality/tests/test_stage1_golden_snapshot.py -v
#    (the test skips, reporting that the goldens were regenerated and that you
#    should inspect the git diff before committing)

# 2. verify the regenerated bytes pass on the SAME machine/env
pytest max_quality/tests/test_stage1_golden_snapshot.py -v

# 3. review the change, then stage the goldens (the three JSON files AND the
#    .gitkeep) and commit them
git diff max_quality/tests/golden/stage1/
git add max_quality/tests/golden/stage1/
git commit -m "<describe the artifact change>"
```

Determinism caveat: the regen step and the verify step must run on the same
machine with the same Python/torch wheel — CPU float reprs are bit-identical
only under those conditions. A purely additive plugin that is **disabled by
default** leaves the golden untouched and needs no regeneration.
