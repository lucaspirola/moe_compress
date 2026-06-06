# PLAN — 6-model healing probe machinery

**Date:** 2026-06-06
**Branch:** `feat/probe-machinery`
**Status:** planning only — NO production code written yet
**Design spec:** `tasks/2026-06-06-reap-ream-healing-probe-design.md` (read in full)
**Repo state:** `main` @ `3d715d6` (REAP faithful pruner merged); all paths below are
under `max_quality/` unless noted.

> The probe RUN is BLOCKED on empty calibration sidecars (separate issue). This plan
> builds the CODE + configs so the probe is launch-ready the moment sidecars are
> re-captured. Nothing here runs on GPU.

---

## 0. OPEN DECISIONS (read first — these need the user)

### OD-1 (CRITICAL) — REAM has **no clean "35% expert reduction" knob**

**Finding (verified):**
- The faithful-prune path has a direct, clean per-layer knob:
  `stage2_reap_ream.prune_fraction = 0.35` →
  `n_prune = int(n_experts(layer0) * 0.35)`, computed once and reused per layer,
  **bypassing the Stage-1 GRAPE budget**
  (`stage2/orchestrator.py:1338`, comment at config
  `configs/qwen36_35b_a3b_reap_faithful.yaml:166-170`).
- The REAM **merge** path has NO equivalent. Its per-layer survivor count comes from
  `per_layer_target = budgets_payload["per_layer_target_experts"]`
  (`stage2/orchestrator.py:787-789`), which GRAPE allocates in Stage 1.
- GRAPE's `per_layer_target_experts` is a non-uniform distribution of a single
  **global param budget** `global_expert_budget`
  (`budget/solver.py:109-126` — the `BudgetDecomposition` dataclass docstring
  explicitly states "`per_layer_target_experts` … is **not** a solver output …
  allocated by GRAPE"). That global budget is solved from
  `target.total_reduction_ratio` × `target.expert_svd_ratio`
  (the ep:sp knob ratio), measured against **total model params**, not expert count
  (`budget/solver.py:14-26, 40-67`; consumed at `run_pipeline.py:177-178`).
- Two hard constraints distort any attempt to map a param target onto a clean
  35% expert fraction:
  - `min_experts_per_layer` floor (default `num_routed_experts // 2` = **128 of
    256** = a 50% hard floor) — `budget/solver.py:28-31`.
  - `_MAX_EP = 0.60` ceiling on the expert-prune knob — `budget/solver.py:104`.

**So:** "REAM at 35% expert reduction" is **not directly expressible**. The
param-budget solver is the only lever, and it targets params, not an expert
fraction. This matches the prior memory finding (target = total_reduction × svd_ratio
param budget, NOT a clean expert fraction).

**Existing precedent (verified):** `run_ablations.py:234-235` already forces the
budget solver to put ~all savings into expert pruning by setting
`total_reduction_ratio = 0.35` + `expert_svd_ratio = 100.0`. But note this is a
**35% TOTAL-PARAM** reduction, which at expert_svd_ratio=100 lands ~all on experts —
yet still measured in *params*, and still subject to the 128-floor / 0.60-ceiling.
It does NOT guarantee "35% of experts dropped per layer."

**Options for the user (pick one before launch):**

- **Option A (RECOMMENDED) — match the faithful-prune survivor count.**
  Run the REAP faithful-prune base FIRST. It drops a fixed
  `int(256 * 0.35) = 89` experts → **167 survivors per layer**. Then build a
  custom Stage-1 budget JSON for the REAM runs that pins
  `per_layer_target_experts[layer] = 167` for every MoE layer (uniform), and
  feed it via the existing `stage1_budget_path` arg
  (`stage2/orchestrator.py:784-789` reads `artifacts_dir/stage1_budgets.json`,
  overridable). This makes REAM merge down to the **identical survivor count** as
  REAP prune → an apples-to-apples 35%-expert comparison, sidestepping the param
  solver entirely. Tiny helper script writes the JSON; no solver change.
  *Caveat:* 167 > the 128 floor, so the floor never binds — safe.

- **Option B — param-budget proxy.** Use `total_reduction_ratio=0.35` +
  `expert_svd_ratio=100.0` (the `run_ablations.py` recipe) and accept that "35%"
  means 35% total-param, landing ~all on experts. Simplest (zero new code) but
  the two bases are then NOT at the same expert count, weakening the
  REAP-vs-REAM comparison the probe exists to make.

- **Option C — add a clean REAM expert-fraction knob.** New code: a
  `merge_target_fraction` that, like `prune_fraction`, bypasses GRAPE and pins
  per-layer survivor counts. Most faithful to the design's intent but is new
  production code with its own tests/review — heavier than a probe warrants.

**My recommendation: Option A.** It reuses the faithful-prune survivor count as
the ground truth, needs only a ~20-line JSON-writer helper (not a solver change),
and gives the cleanest REAP-vs-REAM read. **DO NOT proceed past config-stubbing
on the REAM bases until the user picks A / B / C.** Where this plan needs a
concrete value it assumes Option A and flags it `[OD-1:A]`.

### OD-2 — Run naming + storage layout

Proposed names for the 6 runs (used as artifact subdirs + HF repo suffixes):
`reap_base`, `reap_heal_ours`, `reap_heal_paper`, `ream_base`,
`ream_heal_ours`, `ream_heal_paper`. **Confirm naming + whether each gets its own
HF repo or one repo with 6 subfolders** (see §6). Recommendation: one driver,
artifact subdirs under a single `--artifacts-dir`, per-run HF repos via the
existing `upload_stage_to_hub` `-stageN` suffix convention (§6).

### OD-3 — Exact paper-core REAM plugin toggles

The design §3 lists the paper-core REAM plugin set conceptually; the concrete
config keys are mapped in §4 below. **Confirm the §4 toggle table** — in
particular that `sequential_reprofile=true` (the §4-sequential-propagation
plugin) is wanted ON for the "paper-faithful" base. It defaults OFF
(`ream_sequential.py:209`) and is **mutually exclusive** with the profile sidecar
(`stage2/orchestrator.py:759-767`), so turning it ON forces
`profile_sidecar.enabled=false`.

---

## 1. Piece 1 — RKD dials-only mode (`paper_dials_only`)

### Current behaviour (verified)
`router_kd/plugins/rkd_paper_recipe.py`:
- `apply_config_overrides(config)` (line 141) early-returns unless
  `s5["rkd_recipe"] == "paper"` (lines 162-166).
- On `"paper"` it sets the 4 dials (`kd_temperature=4.0`, `weight_decay=0.0`,
  `epochs=2`, `early_stop_patience=0`) at lines 169-172, clears
  `teacher_logits_cache=None` (line 177, the epochs>1 guard), AND forces
  `cal["source"] = "wikitext-103-raw"` (lines 184-195).
- The orchestrator calls this as the FIRST statement of `run()`
  (`router_kd/orchestrator.py:180`), before the `s5`/`cal` captures
  (lines 182-183). `is_enabled` keys off the same value (line 134).

### Change — add a third mode that keeps our calibration source

The dials-only path applies the same 4 numeric dials + the epochs>1 cache-clear,
but **does NOT touch `cal["source"]`** (keeps `qwen3-pretrain-mix-v2`).

**File:** `src/moe_compress/router_kd/plugins/rkd_paper_recipe.py`

1. **`apply_config_overrides` (lines 162-202)** — replace the gate + branch so it
   recognises both `"paper"` and `"paper_dials_only"`:
   - Read `recipe = s5.get("rkd_recipe", "current")` once.
   - `if recipe not in ("paper", "paper_dials_only"): return`.
   - Apply the 4 dials + `teacher_logits_cache=None` unconditionally for BOTH
     modes (lines 169-177 stay as-is).
   - Wrap the calibration-source swap (lines 184-195) in
     `if recipe == "paper":` so `paper_dials_only` SKIPS it.
   - Update the two `log.info` lines (197-202) to report the active recipe and
     whether the source was swapped.

2. **`is_enabled` (lines 123-134)** — return True for both `"paper"` and
   `"paper_dials_only"`:
   `return s5.get("rkd_recipe", "current") in ("paper", "paper_dials_only")`.

3. **Docstrings** — class docstring "Contract" (lines 57-87) + method docstring
   (lines 142-160): document the third mode and that it preserves
   `calibration.source`. Keep the existing "paper" wording.

No orchestrator change needed — `router_kd/orchestrator.py:180` already calls
`apply_config_overrides` unconditionally; the new branch is internal.

### Test plan

**File:** `tests/test_router_kd_plugin_rkd_paper_recipe.py` (extend; do not rewrite).

Add a Group D-bis. Build on the existing `_row_c_config()` helper (line 225) with
`rkd_recipe = "paper_dials_only"`:

- `test_dials_only_sets_4_dials` — kd_temperature==4.0, weight_decay==0.0,
  epochs==2, early_stop_patience==0.
- `test_dials_only_clears_teacher_cache` — `teacher_logits_cache is None`
  (epochs>1 guard still honoured).
- `test_dials_only_KEEPS_calibration_source` — **the key new invariant**:
  `config["calibration"]["source"] == "qwen3-pretrain-mix-v2"` (NOT
  `wikitext-103-raw`). This is the entire point of the mode.
- `test_dials_only_is_enabled` — `is_enabled({...rkd_recipe:"paper_dials_only"})`
  is True.
- `test_dials_only_idempotent` — mirror existing `..._paper_idempotent`
  (line 426).
- Regression guard: keep the existing `test_apply_config_overrides_paper_sets_calib_source`
  (line 380) green — `"paper"` MUST still swap to wikitext.

Run: `pytest tests/test_router_kd_plugin_rkd_paper_recipe.py -q` (CPU-only,
no GPU/model).

---

## 2. Piece 2 — stage6alt eval corpus = wikitext

### Status: WORKS as-is. No code change.

Verified in `stage6alt/plugins/thermo_corpus.py`:
- `_build_thermo_corpus` reads `therm.get("corpus", "nemotron")` (line 195) and
  dispatches to the wikitext branch (lines 205-233) when
  `thermometer.corpus == "wikitext"`.
- The wikitext branch reads `thermometer.wikitext.{dataset,subset,split}`
  (lines 207-209) and pins the revision off
  `stage6_validate.dataset_revisions.wikitext_ppl` (lines 215-217) — the same
  canonical SHA key Stage 6's PPL gate uses.
- `ThermoCorpusPlugin.is_enabled` always returns True (lines 285-291); the
  config key only selects WHICH corpus.
- The existing `configs/qwen36_35b_a3b_reap_faithful.yaml:549` already sets
  `thermometer.corpus: wikitext` with the wikitext sub-block (550-553) and the
  pinned SHA (`dataset_revisions.wikitext_ppl`, line 632).

### Gap / caveat (carry into configs, not a code change)
The wikitext branch passes `revision` through to `load_dataset`
(`thermo_corpus.py:161, 217`). If `dataset_revisions.wikitext_ppl` is null AND
`strict_revision_pinning: true` (line 624), Stage 6 aborts. The faithful config
already pins the SHA, so the 6 probe configs MUST inherit that pinned SHA. **Action:
all 6 probe configs set `thermometer.corpus: wikitext` + carry the pinned
`dataset_revisions.wikitext_ppl` SHA** (copy from the faithful config). No code.

---

## 3. Piece 3 — 6-model orchestration + configs

### 3.1 Config layout

**Base config = `configs/qwen36_35b_a3b_reap_faithful.yaml`** (already exists, already
correct for: Stage 1+2 only, `pipeline.skip_intermediate_stages: true`,
`pipeline.evaluator: stage6alt`, `thermometer.corpus: wikitext`, our
calibration source).

Two strategy approaches — RECOMMENDATION: **programmatic deltas in a driver**
(mirror `run_ablations.py`), NOT 6 hand-maintained YAMLs. Rationale: the 6 runs
differ only by a small delta set; `run_ablations.py:214-278` already proves the
deep-copy-base + apply-deltas pattern, with idempotent skip + leaderboard + bucket
upload. Reuse it.

Define a 6-row spec table (analogous to the `_ABLATIONS` list at
`run_ablations.py:204-206`):

| run name | base | Stage-2 deltas | heal? | rkd_recipe |
|---|---|---|---|---|
| `reap_base`        | faithful-prune | `prune_mode=faithful_prune`, `prune_fraction=0.35` | no | — (skip stage 5) |
| `reap_heal_ours`   | faithful-prune | same | yes | `current` |
| `reap_heal_paper`  | faithful-prune | same | yes | `paper_dials_only` |
| `ream_base`        | merge          | see §4 toggle table `[OD-1:A]` | no | — (skip stage 5) |
| `ream_heal_ours`   | merge          | same | yes | `current` |
| `ream_heal_paper`  | merge          | same | yes | `paper_dials_only` |

- **`no-heal` rows:** keep `pipeline.skip_intermediate_stages: true` so the run
  jumps Stage 2 → stage6alt, no Router-KD (faithful config default, line 678).
- **`heal` rows:** flip `pipeline.skip_intermediate_stages: false` and run Stage 5
  (Router-KD `merge_repair`/`router_kd`) as the ONLY post-Stage-2 stage. Stages 3/4
  stay off. **Confirm the exact mechanism for "Stage 5 only" with the user** — the
  faithful config's `skip_intermediate_stages` flag is all-or-nothing (skips 2.5/3/4/5
  together, `run_pipeline.py` reads it). Healing rows need 5-only.
  `[OD-4 below]`.

> **OD-4 (decision):** the faithful config exposes only the binary
> `pipeline.skip_intermediate_stages`. The heal rows need "skip 3+4, run 5". Options:
> (a) add a finer-grained `pipeline.run_stages: [1,2,5]` knob to `run_pipeline.py`
> (small, but new production code), or (b) reuse the proven `run_ablations.py`
> shape, which already runs Stage 1→2→2.5→6 and skips 3/4/5 — and adapt it to run
> Stage 5 healing instead of 2.5. **Recommend (b):** the ablation harness already
> does "Stage 2 + KD + stage6alt, no SVD." The probe driver is a near-clone with
> 6 rows and the REAP/REAM base switch. Confirm before coding.

### 3.2 The REAP base (rows 1-3)

Direct, no decisions:
```yaml
stage2_reap_ream:
  prune_mode: faithful_prune     # ReapPrunePlugin pure drop (reap_prune.py)
  prune_fraction: 0.35           # → 89 dropped / 167 kept per 256-expert layer
```
This is already the faithful config's Stage-2 block (lines 159-166). Rows 1-3
share it; rows 2-3 add the heal stage.

### 3.3 The REAM base (rows 4-6) — paper-core merge `[OD-1:A]`

Set `prune_mode: merge` (the default; `stage2/orchestrator.py:694-711` validates
`prune_fraction` only in faithful mode). Survivor count pinned to 167/layer via
the Option-A custom `stage1_budgets.json` (see OD-1). Plugin toggles per §4.

### 3.4 The heal stage (rows 2,3,5,6)

Same Router-KD mechanism for all four; differ only in `rkd_recipe`:
- `reap_heal_ours` / `ream_heal_ours`: `stage5_router_kd.rkd_recipe = "current"`
  (the production dials; T=1.0, wd=0.01, epochs=1, patience=8 —
  `configs/...faithful.yaml:356-467` defaults).
- `reap_heal_paper` / `ream_heal_paper`: `stage5_router_kd.rkd_recipe =
  "paper_dials_only"` (Piece 1 — paper dials, OUR data).
- For REAM heal rows, `merge_repair.enabled: true` is appropriate (the merged
  centroids exist; `configs/...faithful.yaml:514-519`). For REAP heal rows there are
  no merged centroids (pure drop) → keep `merge_repair.enabled: false`
  (router-only KD). **Confirm with user** — flagged in OD-3.

---

## 4. Paper-core REAM plugin toggle table (config keys → verified)

All keys live under `stage2_reap_ream`. Plugin `config_key`s verified by grep of
`stage2/plugins/*.py`. "ON" = the value that activates the paper-core plugin;
"OFF" = the inert default the design §3 wants.

| Design plugin (§3) | config key | probe value | source (verified) |
|---|---|---|---|
| `reap_scoring` (saliency) | `stage2_reap_ream` | always on (scoring step) | `reap_scoring.py:157` |
| `ream_cost` (Eqs 5/7/8) | `cost_alignment` | `"pre"` (δ_REAM) | `ream_cost.py:507` |
| `solver_greedy` | `assignment_solver` | `"greedy"` | `solver_greedy.py:246` |
| `layer_merge` (Eq 6) | `stage2_reap_ream` | `prune_mode: merge` | `layer_merge.py:341` |
| `ream_sequential` (§4 propagation) | `sequential_reprofile` | `true` `[OD-3]` | `ream_sequential.py:181,209` |
| merge op | `merge_step` | default freq-weighted (`frequency_weighted_merge: true`) | `layer_merge.py:460`, config:202-206 |
| **OFF** `merge_heal` | `merge_heal_enabled` | `false` | `merge_heal.py:1105` |
| **OFF** `expert_distill` | `expert_distill_steps` | `0` | `expert_distill.py:893` |
| **OFF** `em_refine` | `em_refinement_rounds` | `0` | `em_refine.py:403` |
| **OFF** `two_opt_refine` | `two_opt_refine` | falsy/absent | `two_opt_refine.py:215` |
| **OFF** `capacity_gate` | `stage2_reap_ream` (capacity_util_threshold) | leave default; gate inert | `capacity_gate.py:134` |
| **OFF** `skip_merge_floor` | `skip_merge_percentile` | `100.0` (OFF sentinel) | `skip_merge_floor.py:101-102` |
| **OFF** `output_space_cost` | `cost_alignment` | not `"post"` (use `"pre"`) | `output_space_cost.py:638` |
| **OFF** `ream_cost_post` | `cost_alignment` | not `"post"` | `ream_cost_post.py:397` |
| **OFF** `regmean_merge` | `merge_step` | not `"regmean"` | `regmean_merge.py:132` |
| **OFF** `solver_hungarian/mcf/sinkhorn/auto` | `assignment_solver` | `"greedy"` (not these) | `solver_*.py` |

> **Mutual-exclusion guard (verified):** `sequential_reprofile=true` +
> `profile_sidecar.enabled=true` is rejected at `stage2/orchestrator.py:759-767`.
> If OD-3 keeps sequential ON, the probe config MUST set
> `profile_sidecar.enabled: false`.

> Note the faithful config currently has `skip_merge_percentile: 0.0`
> (config:232) and `cost_asymmetric: false` (226) — those are REAP-PRUNE-exact
> values. The REAM rows OVERRIDE `skip_merge_percentile` back to `100.0` (OFF) so
> merges are allowed (the whole point of REAM). The driver applies this as a delta.

---

## 5. Orchestration / launch (one command)

**New file:** `src/moe_compress/run_probe.py` — a near-clone of `run_ablations.py`
adapted to the 6-row probe. Reuses verbatim:
- the env-var preamble (`run_ablations.py:19-60`),
- `_build_*_config` deep-copy + delta pattern (214-278),
- the idempotent skip on existing `stage6alt_eval.json`
  (`_stage6_artifact`, 286-293; the file is `stage6alt_eval.json` for thermometer
  mode),
- the leaderboard table ranked by `bpt_gap` ascending (318-357),
- the bucket/summary upload (379-432).

Differences from `run_ablations.py`:
1. **6 rows, not 12** — the §3.1 spec table, with a `base` field selecting
   REAP-faithful vs REAM-merge Stage-2 deltas + the per-row `rkd_recipe`.
2. **Two bases** — the driver builds REAP rows from `prune_mode=faithful_prune`
   and REAM rows from `prune_mode=merge` + the Option-A pinned budget JSON.
3. **Heal toggle** — no-heal rows skip Stage 5; heal rows run Stage 5 with the
   row's `rkd_recipe` (per OD-4 mechanism).
4. **Pre-flight: REAP base FIRST** `[OD-1:A]` — run `reap_base`, read its dropped
   count (167 survivors), write `stage1_budgets.json` pinning 167/layer for the
   REAM rows. ~20-line helper `_pin_uniform_budget(survivors:int)`.

**Launch (once sidecars exist):**
```bash
MOE_SKIP_STAGE2_COV_SAVE=1 \
PIPELINE_HUB_RESULT_REPO_BASE=pirola/probe-reap-ream \
python -m moe_compress.run_probe \
    --config configs/qwen36_35b_a3b_reap_faithful.yaml \
    --artifacts-dir ./artifacts/probe \
    --num-sequences 4000
```
(`MOE_SKIP_STAGE2_COV_SAVE=1` per the faithful config header — faithful mode
collects no covariance; required for the REAP rows.)

### Crash/preempt resilience (design §6)
Already covered by existing machinery: Stage 2 checkpoints per layer
(`stage2/resume.py`), the driver's idempotent per-row skip, stage6alt is cheap +
re-runnable. A spot-H200 preemption resumes from the last completed row + layer.

---

## 6. Per-model HF upload + winner-pick + local download of winner

### Upload (verified API)
`utils/hub_upload.py:112 upload_stage_to_hub(...)` creates
`{repo_base}-stage{stage_idx}` (line 147), `private=True` (151), waits for the
upload (`wait_for_pending_uploads`, 186). `hub_repo_base_from_env()` (197) reads
`PIPELINE_HUB_RESULT_REPO_BASE`.

`run_ablations.py` deliberately DISABLES per-row model uploads (only summary +
bucket) to avoid 36 junk repos (docstring 12-15). **Decision OD-2:** for the
probe we likely DO want each winner-candidate model retrievable. Recommendation:
- Upload the **stage6alt_eval.json + config + a tiny manifest** for all 6 rows to
  one summary repo (cheap), AND
- Upload the **full model weights only for the picked winner** (after the table is
  computed) to `{repo_base}-winner`.
- Confirm with user (OD-2) whether all 6 full models should be uploaded or just the
  winner.

### Winner-pick
Reuse `run_ablations.py`'s leaderboard: rank the 6 by `bpt_gap` ascending
(lower = less compression damage; `run_ablations.py:332-357`). The probe's
decision (design §8) is two-axis — pick (a) better base REAP vs REAM and
(b) better recipe ours vs paper-dials. The driver writes
`_probe_leaderboard.md` with all 6 + an explicit "winning base / winning recipe"
section derived from the table. `top1_agreement` is reported alongside (fair on
wikitext).

### Local download of winner (design)
After the table, `huggingface_hub.snapshot_download(repo_id=f"{repo_base}-winner")`
into a local dir. Add a `--download-winner-to PATH` flag to `run_probe.py`; default
off (so a headless spot run doesn't block on a large download). Document the manual
fallback command in the run README/log.

---

## 7. Test plan summary (all CPU, no GPU)

1. **Piece 1:** extend `tests/test_router_kd_plugin_rkd_paper_recipe.py`
   (§1 test plan) — `pytest tests/test_router_kd_plugin_rkd_paper_recipe.py -q`.
2. **Piece 2:** add a config-shape test mirroring
   `tests/test_reap_faithful_config.py` asserting each of the 6 generated probe
   configs has `thermometer.corpus == "wikitext"` + a non-null
   `dataset_revisions.wikitext_ppl`.
3. **Piece 3 / driver:** a `run_probe` config-builder unit test (no GPU): assert
   the 6-row spec table produces configs with the expected
   `prune_mode` / `prune_fraction` / `rkd_recipe` / heal-toggle / §4 OFF-sentinels.
   Mirror `run_ablations.py`'s existing config-builder test if one exists; else add
   `tests/test_run_probe_config.py`.
4. **No-regression:** existing golden snapshots
   (`test_router_kd_golden_snapshot.py`) must stay byte-identical — the
   `paper_dials_only` branch is additive and the `"current"`/`"paper"` paths are
   untouched.

---

## 8. Work order (gated on OD-1..OD-4 answers)

1. Resolve **OD-1** (REAM 35% knob), **OD-2** (naming/upload), **OD-3** (REAM
   plugin toggles + REAP merge_repair), **OD-4** (Stage-5-only mechanism).
2. Implement Piece 1 (`paper_dials_only`) + tests — fully unblocked, no decisions.
3. Build the 6-row driver `run_probe.py` per the resolved decisions.
4. Add config-shape + driver-config tests.
5. Dry-run the driver's config-builder on CPU (no model load) to confirm the 6
   configs materialise correctly.
6. Hold for sidecar re-capture, then launch (§5).

---

## Appendix — file:line evidence index

- RKD recipe plugin: `router_kd/plugins/rkd_paper_recipe.py:134,141,162-202`
- RKD orchestrator call site: `router_kd/orchestrator.py:180`
- RKD tests: `tests/test_router_kd_plugin_rkd_paper_recipe.py` (380 = calib-source test)
- Thermo corpus wikitext branch: `stage6alt/plugins/thermo_corpus.py:195,205-233`
- Thermo strict-revision abort: `configs/qwen36_35b_a3b_reap_faithful.yaml:624,632`
- Faithful-prune knob: `configs/qwen36_35b_a3b_reap_faithful.yaml:159-166`;
  `stage2/orchestrator.py:694-711,1338,1591-1592`
- REAM budget path: `stage2/orchestrator.py:784-789`;
  `budget/solver.py:14-31,40-67,104-126`; `run_pipeline.py:177-178`
- run_ablations precedent (svd_ratio=100 trick + driver shape):
  `run_ablations.py:5,234-235,214-278,286-293,318-357,379-432`
- Stage-2 plugin config keys: see §4 table (each verified by grep)
- Mutual-exclusion guard: `stage2/orchestrator.py:759-767`
- Hub upload API: `utils/hub_upload.py:112,147,186,197`

---
## RESOLUTIONS (user decisions 2026-06-06) — scope LOCKED, no open questions

6 models, ALL on qwen3-pretrain-mix-v2, 35% fewer experts (keep round(0.65*256)=166/layer):
- Group REAP (by-the-book): prune_mode=faithful_prune, prune_fraction=0.35
- Group REAM (by-the-book): direct merge_size=166 knob (OD-1 — mirrors upstream `--merge_size`, an ABSOLUTE per-layer kept count; bypass GRAPE, parallel to REAP's prune_fraction). Confirmed from SamsungSAILMontreal/ream merge.py `--merge_size` + ream/ream.py pseudo_group(k=...).
- Each group: base (no heal) + stage-2.5-heal + RKD-heal = 3 per group → 6 total.
  - "stage-2.5 healing" = CURRENT production router-KD (default dials, our calib; rkd_recipe absent).
  - "RKD healing" = rkd_paper_recipe paper_dials_only (paper dials, our calib; NO wikitext).
  - merge_repair is NOT involved (was my confusion; closed).
- OD-2 naming: HF dataset/model repos pirola/calib-v2-probe-{reap,ream}-{base,heal25,rkd}; stage6alt eval on wikitext each.
- OD-4: add a real per-stage enable toggle (heal/Stage-5 ON while SVD/Stage-3/4 OFF) — replaces the all-or-nothing skip_intermediate_stages for the probe.
- B0: fix vLLM capture-hook to bind Qwen3.6 fused/GDN MoE + add calib-driver fail-fast (assert nonzero captured entries after chunk 1). Re-capture via forward-only replay of the 8000 saved prompts.

## PAPER-CORE = NO REFINEMENTS (user-confirmed 2026-06-06) — HARD REQUIREMENT
Goal: establish the by-the-book REAP/REAM baseline (what popular compressors give) BEFORE adding our refinements. ALL 6 probe configs MUST explicitly disable (not rely on defaults):
  em_refinement_rounds: 0
  expert_distill_steps: 0
  merge_heal_enabled: false
  two_opt_refine: off (confirm exact key in registry)
  capacity_gate: off (confirm exact key)
  skip_merge_floor: off  (REAM paper-core uses plain greedy assignment; no percentile masking)
REAP group: faithful_prune bypasses the merge machinery so these are structurally inert — set them anyway for clarity. REAM group: these ARE merge-path plugins → explicit-off is load-bearing for faithfulness. The orchestration (run_probe.py + 6 configs) build step must assert these in every config + ideally a test that greps the generated configs for the disabled set.
