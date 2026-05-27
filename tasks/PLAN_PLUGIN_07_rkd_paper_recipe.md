# Plugin #7 — RKD Paper Recipe (Row P config overrides)

**Status**: Spec. NOT implemented.
**Author**: ml-intern protocol (session planner, 2026-05-27)
**Parent plan**: `tasks/RKD_AB_PLAN.md` §4 (Row P)

---

## 1. Goal & Spec Citation

Implement a NEW plugin `router_kd/plugins/rkd_paper_recipe.py` that, when
`rkd_recipe: "paper"` is set in the `stage5_router_kd` config block, applies
the four paper-recipe deltas to the live config dict BEFORE the Router-KD
orchestrator captures any local variable from it. The existing Stage 2.5
plugins — `vocab_kd.py`, `kd_optimizer.py`, `early_stop.py`,
`trainable_scope.py`, `teacher.py`, `merge_repair.py` — are NOT modified.

**Spec sources:**
- `tasks/RKD_AB_PLAN.md` §4 (Row P definition, delta table, pre-flight checks)
- arxiv **2603.02217** — "Is Retraining-Free Enough? The Necessity of Router
  Calibration for Efficient MoE Compression" (Hyeon & Do, Mar 2026). The paper's
  Eq. 3 defines `L_RKD = (τ²/N_x) · Σ_t m_{t+1} · D_KL(p_T ‖ p_S)`.
- Hinton et al. 2015 — original KD paper establishing the τ² scaling convention
  (softened distributions scale the gradient by τ²; the paper's canonical choice
  is τ=4).

---

## 2. Pre-flight Verification Results

Examined `max_quality/src/moe_compress/router_kd/plugins/vocab_kd.py` (the
`_chunked_vocab_kl` kernel) and supporting modules.

### 2a. Forward-KL direction — PASS

`vocab_kd.py:124-126`:
```python
t_p = F.softmax(t_chunk / temperature, dim=-1)
s_lp = F.log_softmax(s_chunk / temperature, dim=-1)
chunk_kl = F.kl_div(s_lp, t_p, reduction="none").sum(dim=-1)
```
PyTorch `F.kl_div(input, target)` computes `Σ target × (log target − input)`.
With `input = log_softmax(student/τ)` and `target = softmax(teacher/τ)`, this
gives `Σ softmax(teacher/τ) × (log_softmax(teacher/τ) − log_softmax(student/τ))`
= D_KL(teacher ‖ student) = forward-KL (teacher → student), which is exactly
the paper's form. **PASS.**

### 2b. τ² scaling — PASS

`vocab_kd.py:132`:
```python
return (total_kl / max(n_tokens, 1)) * (temperature ** 2)
```
The τ² multiplier is explicitly present. **PASS.**

### 2c. Padding mask — PASS (with invariant)

`vocab_kd.py:97-105` explicitly documents: "No per-position mask `m_{t+1}`.
Under the fully-packed invariant `m_{t+1}=1` everywhere, so `Σ_t m_{t+1} =
n_tokens` and the mask multiplier collapses to a no-op." A `torch._assert` at
function entry enforces `teacher_logits.shape == student_logits.shape`, which
guarantees the fully-packed invariant is respected.

The `wikitext-103-raw-v1` adapter registered below uses the same
`build_calibration_tensor` → `_tokenize_to_fixed_length` pathway as all other
sources — producing fixed-length packed sequences with no padding tokens.
**PASS with invariant: the wikitext adapter must not produce variable-length or
padded tensors.** The standard tokenizer pipeline already enforces this.

### 2d. No critical bugs found

All three checks pass. The new plugin is a **hyperparameter-override +
calibration-source swap** plugin, NOT a loss-kernel replacement.

The pre-flight conclusion: Row P's behavior difference vs Row C is purely the 4
config deltas + the calibration source. No production code is broken.

---

## 3. The 4 Deltas (Row P vs Row C)

| Knob | Row C (current) | Row P (paper) | Location in code |
|---|---|---|---|
| `kd_temperature` | 1.0 | **4.0** | `orchestrator.py:291` reads `s5.get("kd_temperature", 1.0)` |
| `weight_decay` | 0.01 | **0.0** | `kd_optimizer.py:229` reads `s5.get("weight_decay", 0.0)` |
| `epochs` | 1 | **2** | `orchestrator.py:310` reads `s5["epochs"]` |
| `early_stop_patience` | 8 | **0** (disabled) | `early_stop.py:292` reads `s5.get("early_stop_patience", 0)` |
| Calibration source | `qwen3-pretrain-mix-v2` | **`wikitext-103-raw`** | `orchestrator.py:276` calls `spec_from_config(cal, ...)` |

**Multi-epoch + cache guard:** `orchestrator.py:585` raises if
`epochs > 1 and teacher_logits_cache is not None`. Row P sets `epochs=2`,
so `apply_config_overrides` must explicitly set `teacher_logits_cache` to
`None` (or remove the key) to prevent the guard from firing when the user
accidentally has a cache configured.

---

## 4. Architecture Decision: Pre-Flight Config Mutation

### Why not a walk_phases hook?

The Router-KD orchestrator captures all config locals at the very top of
`run()` before any plugins are dispatched:

```python
# orchestrator.py:172-173
s5 = config["stage5_router_kd"]
cal = config["calibration"]
```

Any `walk_phases`-dispatched hook runs AFTER these captures. A hook that
writes overrides to `ctx["config"]` at that point is too late — `s5` and `cal`
are already bound to their original values.

### Chosen approach: `apply_config_overrides(config)` called by the orchestrator before capture

The plugin exposes a method `apply_config_overrides(config: dict) -> None`
that mutates `config` in-place. The orchestrator calls this method at the very
top of `run()`, before `s5 = config["stage5_router_kd"]` is captured.

Injection point in `orchestrator.py:run()`:
```python
# Line to add BEFORE "s5 = config["stage5_router_kd"]"
from .plugins.rkd_paper_recipe import RkdPaperRecipePlugin as _RkdPaperPlugin
_RkdPaperPlugin().apply_config_overrides(config)
```

This is a two-line change to the orchestrator. It is unconditional (always
called) but the method is a no-op when `rkd_recipe != "paper"`, so Row C runs
are byte-identical to pre-plugin behavior.

### Contract

1. `apply_config_overrides` reads `config["stage5_router_kd"].get("rkd_recipe", "current")`.
2. If the value is `"current"` (or any non-"paper" value), the method returns immediately without touching `config`.
3. If the value is `"paper"`, it applies the 4 deltas by mutating `config["stage5_router_kd"]` in-place and sets `config["calibration"]["source"] = "wikitext-103-raw"`.
4. The existing Stage 2.5 plugins (`vocab_kd`, `kd_optimizer`, `early_stop`) then read their effective values from the mutated `config` — no changes needed in those plugins.

### Trade-offs

- **Pro**: zero changes to existing Stage 2.5 plugins.
- **Pro**: the override is applied ONCE, before any config captures; no ordering or hook-timing risk.
- **Pro**: Row C behavior is byte-identical (method is a no-op on `rkd_recipe="current"`).
- **Con**: mutates the caller's config dict. Acceptable — the config is the mutable single source of truth for the run; the caller who sets `rkd_recipe: "paper"` is explicitly requesting overrides.

---

## 5. New Configuration Knob

### Field definition

```yaml
# In stage5_router_kd: block of qwen36_35b_a3b_30pct.yaml
rkd_recipe: "current"   # "current" = production Stage 2.5 (all existing plugins, unchanged).
                         # "paper"   = arxiv 2603.02217 recipe: τ=4, wd=0, epochs=2,
                         #             early_stop_patience=0, calib=wikitext-103-raw.
                         #             Applied by RkdPaperRecipePlugin.apply_config_overrides()
                         #             before orchestrator.run() captures config locals.
                         # Default "current" — all existing rows unaffected.
```

The YAML currently has no `rkd_recipe` key. The new plugin reads it with
`.get("rkd_recipe", "current")` so existing config files that don't set it
continue to run the Row C (current) recipe unchanged.

### Config schema

There is no formal JSON Schema / Pydantic validator for the YAML config in
this project (config is consumed as a plain `dict`). No schema file to update.
The config YAML comment above is the authoritative documentation.

---

## 6. Files to Create / Modify

### 6a. NEW: `max_quality/src/moe_compress/router_kd/plugins/rkd_paper_recipe.py`

**Purpose:** Owns the Row P config-override logic and the `is_enabled` gate.
**Responsibilities:**
- `RkdPaperRecipePlugin` class satisfying `PipelinePlugin` Protocol.
- `is_enabled(config)` returns `True` iff `config["stage5_router_kd"].get("rkd_recipe", "current") == "paper"`. Used by the registry for audit/reporting only — the method `apply_config_overrides` drives the actual behavior.
- `apply_config_overrides(config: dict) -> None`: the core method. No-op when `rkd_recipe != "paper"`. When `"paper"`, mutates:
  - `s5["kd_temperature"] = 4.0`
  - `s5["weight_decay"] = 0.0`
  - `s5["epochs"] = 2`
  - `s5["early_stop_patience"] = 0`
  - `s5["teacher_logits_cache"] = None` (prevent multi-epoch + cache guard)
  - `config["calibration"]["source"] = "wikitext-103-raw"`
- Class metadata: `name = "rkd_paper_recipe"`, `paper = "arXiv:2603.02217 + Hinton 2015"`, `config_key = "stage5_router_kd.rkd_recipe"`.
- `reads = ("config",)`, `writes = ()`, `provides = ()`.
- Circular-import contract (same as other plugins): never import `stage5_router_kd` or `router_kd.orchestrator` at any scope.
- Module docstring: spec citation, the 4 deltas, the injection-point contract.

### 6b. MODIFY: `max_quality/src/moe_compress/router_kd/orchestrator.py`

**Change:** Add import and call of `apply_config_overrides` at the very top of `run()`, before `s5 = config["stage5_router_kd"]` (currently line 172).

Exact insertion point: after the `def run(...)` signature and its docstring, before the first line of function body.

```python
# NEW import at module top (with the other plugin imports, ~line 114):
from .plugins.rkd_paper_recipe import RkdPaperRecipePlugin

# NEW call at the very start of run() body, before s5 = config[...]:
RkdPaperRecipePlugin().apply_config_overrides(config)
```

The plugin is NOT added to the `PluginRegistry([...])` list (line 199-207) because `RkdPaperRecipePlugin` has no `walk_phases` hooks — it only has `apply_config_overrides` and `is_enabled`. Adding it to the registry would be pure documentation overhead with no runtime function; the method call at line 172 is the functional entry point.

**Net diff:** 2 lines added (1 import, 1 call). No existing lines changed.

### 6c. MODIFY: `max_quality/src/moe_compress/utils/calibration.py`

**Change:** Add a new `CorpusAdapter` for `wikitext-103-raw` (the `wikitext/wikitext-103-raw-v1` HuggingFace dataset).

Following the exact same pattern as `tulu3-sft-mix` (lines 875-937):

```python
# _parse_yaml_wikitext_103_raw: CalibrationSpec with source="wikitext-103-raw",
#   dataset=cal_cfg.get("dataset", "wikitext/wikitext-103-raw-v1")
# _stream_texts_wikitext_103_raw: load_dataset("wikitext/wikitext-103-raw-v1",
#   name="wikitext-103-raw-v1", split="train", streaming=True),
#   stream row["text"] fields (raw paragraphs; no chat template applied —
#   wikitext is raw encyclopedic text, not instruction-tuned),
#   skip empty rows, circuit-breaker at _CIRCUIT_BREAKER_MULTIPLIER * num_sequences.
# register_corpus(CorpusAdapter(name="wikitext-103-raw", ...))
```

Key distinction from instruct-tuned sources: wikitext rows are plain text
strings. No `_render_messages` / `apply_chat_template` call. Just use
`row["text"]` directly after stripping empty strings.

**Net diff:** ~45 lines added at the bottom of `calibration.py`, after the
existing `register_corpus` blocks (after line ~1825). Zero existing lines changed.

### 6d. MODIFY: `max_quality/configs/qwen36_35b_a3b_30pct.yaml`

**Change:** Add `rkd_recipe: "current"` to the `stage5_router_kd:` block
with a documentation comment. Recommended placement: right after line 305
(`stage5_router_kd:`), as the first knob in the block so it's visually
prominent.

```yaml
stage5_router_kd:
  rkd_recipe: "current"          # "current" = production Stage 2.5 recipe (unchanged).
                                  # "paper"   = arxiv 2603.02217 Row P: τ=4, wd=0, epochs=2,
                                  #             early_stop_patience=0, calib=wikitext-103-raw.
                                  # See router_kd/plugins/rkd_paper_recipe.py.
  optimizer: adamw
  ...
```

The value stays `"current"` in the committed config. Row P runs are invoked
by passing a config override (e.g., `--config-override stage5_router_kd.rkd_recipe=paper`
or a sibling YAML that overrides the one key).

### 6e. NEW: `max_quality/tests/test_router_kd_plugin_rkd_paper_recipe.py`

See §7 for full test specification.

### NOT MODIFIED (per-spec constraint):
- `vocab_kd.py` — not touched
- `kd_optimizer.py` — not touched
- `early_stop.py` — not touched
- `trainable_scope.py` — not touched
- `teacher.py` — not touched
- `merge_repair.py` — not touched

---

## 7. Test Specification

File: `max_quality/tests/test_router_kd_plugin_rkd_paper_recipe.py`

All tests use only `torch`, `pytest`, and project-internal imports. No GPU
required (all tensors on CPU). No live dataset loading (mock / no-op where needed).

### Group A: Plugin scaffolding

**`test_plugin_imports`**: `RkdPaperRecipePlugin` imports from `router_kd.plugins.rkd_paper_recipe`.

**`test_plugin_satisfies_protocol`**: `isinstance(RkdPaperRecipePlugin(), PipelinePlugin)` is True.

**`test_plugin_metadata`**: `name == "rkd_paper_recipe"`, `"2603.02217" in paper`, `config_key == "stage5_router_kd.rkd_recipe"`, `reads`, `writes`, `provides` are tuples.

**`test_no_forbidden_import`**: AST walk of the plugin module's source; assert it never imports `stage5_router_kd` or `router_kd.orchestrator` at any scope (same pattern as existing plugin tests).

### Group B: `is_enabled` gate

**`test_is_enabled_paper`**: `is_enabled({"stage5_router_kd": {"rkd_recipe": "paper"}})` returns `True`.

**`test_is_enabled_current`**: `is_enabled({"stage5_router_kd": {"rkd_recipe": "current"}})` returns `False`.

**`test_is_enabled_default`**: `is_enabled({"stage5_router_kd": {}})` returns `False` (default is `"current"`).

**`test_is_enabled_missing_block`**: `is_enabled({})` returns `False` (graceful on missing block).

### Group C: `apply_config_overrides` — no-op on non-paper recipe

**`test_apply_config_overrides_noop_on_current`**: Build a config with all Row C
values, call `apply_config_overrides`, assert all values unchanged. Tests that
`rkd_recipe="current"` is a true no-op.

**`test_apply_config_overrides_noop_on_default`**: Same but without `rkd_recipe`
key (missing = default = "current"). Assert no mutation.

### Group D: `apply_config_overrides` — paper recipe applied

**`test_apply_config_overrides_paper_sets_temperature`**: Start with
`kd_temperature: 1.0`, call with `rkd_recipe="paper"`, assert `kd_temperature == 4.0`.

**`test_apply_config_overrides_paper_sets_weight_decay`**: Start with
`weight_decay: 0.01`, call with `rkd_recipe="paper"`, assert `weight_decay == 0.0`.

**`test_apply_config_overrides_paper_sets_epochs`**: Start with `epochs: 1`,
call with `rkd_recipe="paper"`, assert `epochs == 2`.

**`test_apply_config_overrides_paper_disables_early_stop`**: Start with
`early_stop_patience: 8`, call with `rkd_recipe="paper"`, assert
`early_stop_patience == 0`.

**`test_apply_config_overrides_paper_sets_calib_source`**: Start with
`calibration.source: "qwen3-pretrain-mix-v2"`, call with `rkd_recipe="paper"`,
assert `calibration["source"] == "wikitext-103-raw"`.

**`test_apply_config_overrides_paper_clears_teacher_cache`**: Start with
`teacher_logits_cache: "/some/path"`, call with `rkd_recipe="paper"`, assert
`s5.get("teacher_logits_cache") is None` (multi-epoch + cache guard).

**`test_apply_config_overrides_paper_all_deltas_together`**: Single call that
verifies all 4 numeric deltas + calibration source + teacher_logits_cache in
one assertion block.

### Group E: Existing plugins see correct values post-override

**`test_kd_optimizer_reads_overridden_weight_decay`**: Build a config with
`rkd_recipe="paper"`, call `apply_config_overrides`, then construct a
`KdOptimizerPlugin` and invoke its `build_optimizer` hook against a minimal ctx.
Assert the optimizer's `defaults["weight_decay"]` is `0.0`.

**`test_early_stop_reads_overridden_patience`**: Build a config with
`rkd_recipe="paper"`, call `apply_config_overrides`, then construct an
`EarlyStopPlugin` and invoke its `setup_early_stop` hook against a minimal ctx.
Assert `ctx.get("early_stop_patience") == 0`.

(These two tests prove the contract: overrides applied BEFORE plugin hooks run → plugins read the correct values.)

### Group F: wikitext-103-raw corpus adapter (in calibration.py)

**`test_wikitext_103_raw_corpus_registered`**: `from moe_compress.utils.calibration import registered_corpora; assert "wikitext-103-raw" in registered_corpora()`.

**`test_wikitext_103_raw_spec_from_config`**: Call `spec_from_config({"source": "wikitext-103-raw", "num_sequences": 4, "sequence_length": 16, "seed": 0})`, assert `spec.source == "wikitext-103-raw"` and `spec.dataset == "wikitext/wikitext-103-raw-v1"`.

**`test_wikitext_103_raw_stream_no_chat_template`**: Monkeypatch `datasets.load_dataset` to return a fake iterable of `{"text": "Hello world."}` rows. Call `_stream_texts_wikitext_103_raw(spec, tokenizer)` and assert that no `apply_chat_template` was called (verify `tokenizer.apply_chat_template` mock was never invoked). Assert returned list contains the raw text strings.

---

## 8. Data Flow

```
YAML config["stage5_router_kd"]["rkd_recipe"] = "paper"
        |
        v
orchestrator.run(student, tokenizer, config, ...)
        |
        +-- RkdPaperRecipePlugin().apply_config_overrides(config)
        |       |
        |       +-- config["stage5_router_kd"]["kd_temperature"] = 4.0
        |       +-- config["stage5_router_kd"]["weight_decay"] = 0.0
        |       +-- config["stage5_router_kd"]["epochs"] = 2
        |       +-- config["stage5_router_kd"]["early_stop_patience"] = 0
        |       +-- config["stage5_router_kd"]["teacher_logits_cache"] = None
        |       +-- config["calibration"]["source"] = "wikitext-103-raw"
        |
        +-- s5 = config["stage5_router_kd"]   # now sees τ=4, wd=0, epochs=2
        +-- cal = config["calibration"]        # now sees source="wikitext-103-raw"
        |
        +-- spec_from_config(cal, ...)
        |       |
        |       +-- get_corpus_adapter("wikitext-103-raw")   # NEW adapter
        |       +-- build_calibration_tensor(tokenizer, spec) # wikitext text rows
        |
        +-- total_steps = (len(batches) // grad_accum) * s5["epochs"]  # = 750 not 375
        |
        +-- walk_phases(("build_optimizer",), plugins, ctx)
        |       |
        |       +-- KdOptimizerPlugin.build_optimizer(ctx)
        |               reads s5.get("weight_decay", 0.0)  # now 0.0
        |
        +-- walk_phases(("setup_early_stop",), plugins, ctx)
        |       |
        |       +-- EarlyStopPlugin.setup_early_stop(ctx)
        |               reads s5.get("early_stop_patience", 0)  # now 0 → disabled
        |
        +-- epoch loop (epochs=2 iterations)
                T = s5.get("kd_temperature", 1.0)  # now 4.0 → τ² = 16
```

---

## 9. Build Sequence (Implementation Checklist)

### Phase 1: Corpus adapter (prerequisite)

- [ ] Add `_parse_yaml_wikitext_103_raw` and `_stream_texts_wikitext_103_raw`
  functions to `calibration.py` (after the existing `register_corpus` blocks,
  around line 1825).
- [ ] Register with `register_corpus(CorpusAdapter(name="wikitext-103-raw", ...))`.
- [ ] Write Group F tests (`test_wikitext_103_raw_*`).

### Phase 2: Plugin implementation

- [ ] Create `max_quality/src/moe_compress/router_kd/plugins/rkd_paper_recipe.py`.
  - [ ] Module docstring (spec citations, 4 deltas, injection-point contract,
    circular-import contract).
  - [ ] `RkdPaperRecipePlugin` class with all metadata attributes.
  - [ ] `is_enabled(config)` gate.
  - [ ] `apply_config_overrides(config)` method with all 5 mutations.
  - [ ] `contribute_artifact(ctx)` no-op (returns `{}`).
- [ ] Write Group A–E tests.

### Phase 3: Orchestrator integration

- [ ] Add `from .plugins.rkd_paper_recipe import RkdPaperRecipePlugin` to the
  imports section of `orchestrator.py` (after the `early_stop` import, ~line 113).
- [ ] Add `RkdPaperRecipePlugin().apply_config_overrides(config)` as the very
  first line of `run()` function body, before `s5 = config["stage5_router_kd"]`
  (currently line 172).
- [ ] Verify: no other changes to orchestrator needed.

### Phase 4: Config documentation

- [ ] Add `rkd_recipe: "current"` with comment to the `stage5_router_kd:` block
  in `max_quality/configs/qwen36_35b_a3b_30pct.yaml` (as the first knob after
  the section header).

### Phase 5: plugins `__init__.py` update (optional doc update)

- [ ] Update `router_kd/plugins/__init__.py` docstring to mention
  `rkd_paper_recipe.py` (the new plugin extraction RK-9, or just "new").

---

## 10. Risk Register

| Risk | Detection | Mitigation |
|---|---|---|
| **R1** Pre-flight reveals KL bug | See §2 — all checks PASS. No KL direction, mask, or τ² bug found. | N/A — no fix needed before Row P run. |
| **R2** wikitext-103-raw data loader plumbing | New `CorpusAdapter` in `calibration.py` needed; no adapter existed before | Implement adapter in Phase 1. Test with monkeypatched `load_dataset`. The adapter is `~45 lines`, following the exact `tulu3-sft-mix` pattern. |
| **R3** Existing Stage 2.5 tests must still pass | `rkd_recipe` defaults to `"current"` → `apply_config_overrides` is no-op | `tiny_config` fixture has no `rkd_recipe` key → no-op path. Existing tests unaffected. |
| **R4** Multi-epoch + teacher_logits_cache guard fires on Row P | `orchestrator.py:585` raises if `epochs > 1 and teacher_logits_cache is not None` | `apply_config_overrides` explicitly sets `s5["teacher_logits_cache"] = None`. Test in Group D covers this. |
| **R5** Calibration source mutation is visible in logs / trackio | `orchestrator.py:544-566` logs `calib_num_batches`, `epochs`, etc. after the spec build — with overrides applied, the log correctly shows wikitext stats and epochs=2 | No change needed; the override is applied before spec build. |
| **R6** `wikitext-103-raw` rows contain empty strings | wikitext-103 has many empty-line separators that parse as empty `row["text"]` | The stream function skips `if not text.strip()`, same as other adapters. |

---

## 11. Acceptance Gates (SUPERVISOR — runs after both review loops close)

- **G1**: `pytest max_quality/tests/test_router_kd_plugin_rkd_paper_recipe.py` green (new test file).
- **G2**: `pytest max_quality/tests/test_router_kd_plugin_vocab_kd.py max_quality/tests/test_router_kd_plugin_optimizer.py max_quality/tests/test_router_kd_plugin_early_stop.py` green (existing plugins unchanged).
- **G3**: Full suite green: `pytest max_quality/tests/` with no regressions.
- **G4**: `git commit -m "feat(router_kd): Plugin #7 — rkd_paper_recipe config-override plugin (Row P)"` and `git push` direct to `main`.

---

## 12. Out of Scope

- NOT running the A/B comparison (GPU job, deferred until after G4).
- NOT modifying any existing Stage 2.5 plugin's algorithm (`vocab_kd`, `kd_optimizer`, `early_stop`, etc.).
- NOT introducing new `datasets` library dependencies (already used by all other corpus adapters).
- NOT wiring `RkdPaperRecipePlugin` into the `PluginRegistry([...])` list in `orchestrator.py` — it has no `walk_phases` hooks, and adding it would be dead weight in the registry walk.
- NOT implementing the drill-down rows P-τ, P-data, P-time, P-wd (§8 of RKD_AB_PLAN.md — only on inconclusive outcome).

---

## 13. File Reference (absolute paths)

**Read during pre-flight:**
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/router_kd/plugins/vocab_kd.py` — `_chunked_vocab_kl` lines 72-132
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/router_kd/plugins/kd_optimizer.py` — `weight_decay` at line 229
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/router_kd/plugins/early_stop.py` — `early_stop_patience` at line 292
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/router_kd/orchestrator.py` — `s5` capture at line 172, `cal` at 173, `epochs` at 310, multi-epoch+cache guard at 585
- `/home/lucas/ai/moe_compress/max_quality/configs/qwen36_35b_a3b_30pct.yaml` — `stage5_router_kd:` at line 305, `calibration:` at line 29
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/utils/calibration.py` — `CorpusAdapter` at line 136, `register_corpus` at line 156, adapter pattern at lines 875-937

**To create:**
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/router_kd/plugins/rkd_paper_recipe.py`
- `/home/lucas/ai/moe_compress/max_quality/tests/test_router_kd_plugin_rkd_paper_recipe.py`

**To modify:**
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/router_kd/orchestrator.py` — 2 lines (1 import + 1 call)
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/utils/calibration.py` — ~45 lines added (wikitext adapter)
- `/home/lucas/ai/moe_compress/max_quality/configs/qwen36_35b_a3b_30pct.yaml` — ~5 lines added (`rkd_recipe` knob + comment)

---

*Generated 2026-05-27 under ml-intern protocol. Pre-flight complete (§2 all PASS, no bugs). Implementer writes code + tests, does NOT run pytest. Supervisor runs G1–G4 after both review loops close.*
