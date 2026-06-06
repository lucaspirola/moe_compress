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

6 models, ALL on qwen3-pretrain-mix-v2, 35% fewer experts. **Single survivor count
K for BOTH groups: drop = round(0.35 × 256) = 90 → KEEP 166/layer** (see
PLAN-REVIEW §R1 for the REAP fix this requires).
- Group REAP (by-the-book): prune_mode=faithful_prune, prune_fraction=0.35
- Group REAM (by-the-book): direct merge_size=166 knob (OD-1 — mirrors upstream
  `--merge_size`, an ABSOLUTE per-layer kept count; bypass GRAPE, parallel to
  REAP's prune_fraction). Confirmed from SamsungSAILMontreal/ream
  `merge.py --merge_size` (default 96, `merger.py:32`) + `ream/ream.py`
  `pseudo_group(k=...)`. **This knob is already implemented on the branch** —
  see PLAN-REVIEW §R2.
- Each group: base (no heal) + stage-2.5-heal + RKD-heal = 3 per group → 6 total.
  - "stage-2.5 healing" = CURRENT production router-KD (default dials, our calib; rkd_recipe absent).
  - "RKD healing" = rkd_paper_recipe paper_dials_only (paper dials, our calib; NO wikitext).
  - merge_repair is NOT involved (was my confusion; closed).
- OD-2 naming: HF dataset/model repos pirola/calib-v2-probe-{reap,ream}-{base,heal25,rkd}; stage6alt eval on wikitext each.
- OD-4: heal/no-heal is controlled by the EXISTING `pipeline.stages` per-stage
  enable map — no new toggle (see PLAN-REVIEW §R3/§R4).
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

---

# PLAN-REVIEW RESOLUTIONS (2026-06-06) — supersedes OD-1..OD-4 above

> Scope is LOCKED (RESOLUTIONS block). The reviewer's Critical/High/Medium/Low
> findings are resolved below. **Tiebreak = faithfulness to upstream
> SamsungSAILMontreal/ream** (intent: "by-the-book REAP/REAM on our calib, 35%
> fewer experts, no refinements"). Every claim is re-verified against the
> *current branch* code (`feat/probe-machinery` @ 7fbeed1) and the re-read
> upstream clone at `/tmp/ream_upstream`. **Line numbers in §0–§8 above were
> written against an older `main` state and have drifted; the file:line cites in
> THIS section are authoritative.**

## R1 — CRITICAL: single survivor count K=166 for BOTH groups

**Decision: drop = `round(0.35 × 256) = 90` → KEEP exactly 166/layer for BOTH
REAP and REAM.** REAM keeps 166 via `merge_size: 166` (R2). REAP currently
keeps the wrong count.

**Verified bug.** `reap_prune.py:309`:
```python
self._n_prune = int(n_experts * self.prune_fraction)
```
`int(256 × 0.35) = int(89.6) = 89` dropped → **167 kept**, off-by-one vs REAM's
166. (The plugin comment at `reap_prune.py:306-307` cites upstream
`prune.py:258-261`, which also uses `int()`, but upstream REAP prunes to a
*fraction*, not to match a merge survivor count — for an apples-to-apples probe
the two groups MUST land on the identical K.)

**Plan change (production code, R-phase — NOT written in this planning pass):**
change the REAP drop-count derivation so REAP and REAM both land at 166. Two
acceptable forms; pick ONE at implement time:
- **(a) round():** `self._n_prune = round(n_experts * self.prune_fraction)` →
  `round(89.6) = 90` → keeps 166. Minimal diff at `reap_prune.py:309`.
- **(b) explicit keep-count:** add a `stage2_reap_ream.keep_experts` (absolute,
  parallel to REAM's `merge_size`) that, when set, makes
  `n_prune = n_experts - keep_experts` and pins REAP to keep=166 directly. More
  symmetric with REAM's knob but more code.

**Recommendation: (a) round().** One-line change, faithful to the "35% fewer
experts" intent (round-to-nearest), and the probe sets `prune_fraction: 0.35`
so it lands on 166 automatically. The 6 configs set `prune_fraction: 0.35`; a
config-grep/derivation test asserts the derived drop == 90 / keep == 166.

> Note: changing `int()`→`round()` at `reap_prune.py:309` is behaviour-changing
> ONLY when `n_experts × prune_fraction` is non-integer (e.g. 256×0.35=89.6 →
> 89 vs 90). VERIFIED: no existing test exercises such a case —
> `test_reap_prune_golden.py` uses `prune_fraction` 0.5/0.0 (8×0.5=4 exact),
> `test_reap_prune_integration.py:72,134` uses 0.5 (8→4 exact),
> `test_reap_prune_upstream_formula.py` passes `n_prune` directly (bypasses the
> derivation), and `test_run_pipeline_reap_exact.py` uses tiny synthetic
> dims/budgets (not the 256→K production count). So `int()`→`round()` breaks NO
> current test. The R-phase MUST nonetheless add a derived-count test asserting
> 256×0.35 → drop 90 / keep 166, and confirm (by regen) that the production
> faithful-prune goldens — if any are keyed on the 256-expert count — move
> 167→166. Flag for the implementer: behavioural for the real config, inert for
> the current test suite.

## R2 — CRITICAL: merge_size is a THIN budget-pin, ALREADY IMPLEMENTED

**Decision: `stage2_reap_ream.merge_size` is a config key whose ONLY effect is
to pin `target` (the per-layer survivor count) UNIFORMLY = merge_size, on the
existing GRAPE-budget consumption path. No new merge machinery, no grouping /
_promote_orphans / resume changes.** This is the reviewer-approved "Option A"
budget-pin, exposed as the direct knob the user asked for.

**Verified: this is already on the branch — the plan only needs to DOCUMENT it,
not invent it.**
- Validation block: `stage2/orchestrator.py:731-756` — rejects `merge_size`
  under `prune_mode=faithful_prune` (it is a merge-path knob), and asserts a
  positive int.
- The pin itself: `stage2/orchestrator.py:1674-1686`:
  ```python
  target = per_layer_target[layer_ref.layer_idx]   # GRAPE budget (1674)
  if _merge_size is not None:
      ...                                            # ceiling check vs n_exp
      target = int(_merge_size)                      # uniform pin (1686)
  ```
  `target` is then written to ctx at `orchestrator.py:1695` and consumed by the
  merge loop exactly as the GRAPE value would be. When `merge_size` is absent the
  GRAPE `per_layer_target` (`orchestrator.py:814-816`) is used unchanged —
  byte-identical default.

**Plan change:** the §3.3 / §4 prose that described OD-1 Option A as a "custom
`stage1_budgets.json` JSON-writer helper" is OBSOLETE — there is NO budget-JSON
helper and NO `_pin_uniform_budget()` step. The REAM rows simply set
`stage2_reap_ream.merge_size: 166`. Delete the JSON-writer from the §5 driver
spec (it was Difference #4 there). The pin is written by the orchestrator at the
single site `orchestrator.py:1686`; the driver only injects the config key.

## R3 — HIGH: heal mechanism = EXISTING `pipeline.stages` map (no new toggle)

**Decision: DROP the proposed new per-stage toggle. The branch already has a
clean native per-stage enable map; the heal rows use it.** This is even cleaner
than the reviewer's "reuse run_ablations stage-windowing" because it runs
entirely inside `run_pipeline.run()` in ONE invocation.

**Verified.** `run_pipeline.py:434-493` `_resolve_stage_enables(pipe_cfg,
skip_intermediate)`:
- With `pipeline.stages` ABSENT → all intermediate (2.5/3/4/5) derive from
  `skip_intermediate_stages` (byte-identical legacy behaviour, `:439-442,456-457`).
- With `pipeline.stages` PRESENT → a dict `{stage2p5|stage3|stage4|stage5: bool}`
  overrides per-stage (`:443-447,472-482`); contradiction guard at `:488-493`.
- The resolved map gates each stage: `run_pipeline.py:269` (2.5), `:289` (3),
  `:301` (4), `:312` (5).

**Mapping the two heal variants to the existing stage runners (both call
`stage5_router_kd.run`):**
- **stage-2.5 heal** (CURRENT production router-KD, default dials): enable
  Stage 2.5 only. `run_pipeline.py:280` runs
  `stage5_router_kd.run(..., stage_key="stage2p5")` — independent of Stage 3/4.
  Config: `pipeline.stages: {stage2p5: true, stage3: false, stage4: false,
  stage5: false}`, `rkd_recipe` absent.
- **RKD heal** (paper_dials_only): enable Stage 5 only.
  `run_pipeline.py:312-318` runs `stage5_router_kd.run(...)` (default
  `stage_key`). Config: `pipeline.stages: {stage2p5: false, stage3: false,
  stage4: false, stage5: true}`, `stage5_router_kd.rkd_recipe:
  "paper_dials_only"`.

No new production toggle. (The reviewer's run_ablations.py:633 cite refers to a
state of that file that no longer matches; the native `pipeline.stages` map is
the superior in-pipeline equivalent and is what the probe uses.)

## R4 — HIGH: base (no-heal) rows still produce stage6alt_eval.json — natively

**Decision: base rows disable ALL intermediate stages via `pipeline.stages` and
flow straight to Stage 6 in the SAME invocation. No `--skip-stage2p5`, no second
`--resume-from-stage 6` process.** This sidesteps the reviewer's concern
entirely (the `--skip-stage2p5` early-return is never on the probe's path).

**Verified.** When every intermediate stage is off, `_all_intermediate_off=True`
(`run_pipeline.py:104`). The mid-pipeline "stop after stage N" early-returns at
`run_pipeline.py:296-299, 307-310, 319-322` are guarded by
`... and not _all_intermediate_off`, so with all-off they are SKIPPED and control
reaches Stage 6 at `run_pipeline.py:324`. With `stage6_validate.mode` resolving
to `thermometer` (forced by `pipeline.evaluator: stage6alt` at
`run_pipeline.py:111-117`), Stage 6 runs `stage6alt_thermometer.run`
(`run_pipeline.py:329-331`), which writes `stage6alt_eval.json`.

So base rows = `pipeline.stages: {stage2p5: false, stage3: false, stage4: false,
stage5: false}` (or equivalently `skip_intermediate_stages: true` with no
`stages` block), `pipeline.evaluator: stage6alt`, run once with default
`--stop-after-stage 6`. They reach stage6alt without the split-machine
`--skip-stage2p5` chain.

> `--skip-stage2p5` (the reviewer's bail path, `run_pipeline.py:272-276`) is a
> SEPARATE split-machine feature and is NOT used by `run_probe.py`. The driver
> must NOT pass it. (If a future split-GPU flow needs base rows across two boxes,
> the Stage-2-only run + a `--resume-from-stage 6` second process is the
> documented fallback — Stage 6 loads `stage2_pruned/` via the resume shortcut at
> `run_pipeline.py:156-166` — but the single-box probe does not need it.)

## R5 — MEDIUM: REAM by-the-book — match upstream params + document deviations

The REAM rows MUST set the upstream-default REAM params, NOT silently inherit our
production defaults. Verified upstream defaults (`/tmp/ream_upstream`):

| upstream param | upstream default | source | our knob | probe value |
|---|---|---|---|---|
| `group_size` (C) | **16** | `merger.py:40`, `merge.py:72`, `ream.py:25` | `stage2_reap_ream.max_merge_group_size` | **15** (see ⚠ below) |
| `sequential` | **True** | `merger.py:41`, `merge.py:83` | `stage2_reap_ream.sequential_reprofile` | **true** |
| `merging` | `logits+weights` | `merger.py:34` | (cost path, baked) | n/a — δ_REAM `cost_alignment: "pre"` |
| `saliency` | `reap` | `merger.py:35` | always-on REAP scoring | n/a |
| `grouping` | `ream` | `merger.py:33` | `assignment_solver: "greedy"` + pseudo-group | greedy |
| `use_gate_output` | **True** | `merger.py:42` | (no knob — see ⚠ below) | baked-on |
| `gated_sim` | **True** | `merger.py:43` | (no knob — baked) | baked-on |
| freq-weighted merge | (REAM merge op) | — | `frequency_weighted_merge` | **true** |

**⚠ DEVIATION #1 (group_size semantics — CORRECTS the task's "set C=16").**
Our `max_merge_group_size` counts **non-centroids only**; upstream `group_size`
counts the **total group including the centroid**
(`layer_merge.py:55-79`, citing `ream/ream.py:75-82`). Equivalence:
`max_merge_group_size = N` ⇔ upstream `group_size = N+1`. Therefore matching
upstream `group_size=16` requires `max_merge_group_size: 15`, **not 16** (16
would be upstream group_size=17). Our config default is 8
(`layer_merge.py:77`), explicitly ~half the paper's 25%-reduction recipe.
**By-the-book ⇒ set `max_merge_group_size: 15`.**

**⚠ DEVIATION #2 (no `use_gate_output`/`gated_sim` knob).** Our pipeline has NO
gate-output toggle. The gated-softmax expert view is baked into the δ_REAM cost
under `cost_alignment="pre"` (`ream_cost.py:40-46`, mirroring upstream
`ream/moe_utils.py:146-147,170-171`). So upstream's `use_gate_output=True` /
`gated_sim=True` are matched implicitly by `cost_alignment: "pre"`; there is
nothing to set and nothing to turn off. **Accepted deviation (parity preserved
by construction).**

**⚠ DEVIATION #3 (sequential propagation cost — FLAG FOR USER).**
`sequential_reprofile: true` re-forwards each layer after every merge
(`ream_sequential.py`, mirroring `merger.py:303,468` `if self.sequential`). This
is **slower and heavier on GPU** than the one-shot pre-collected-stats path.
This is the price of by-the-book REAM (upstream default is sequential=True).
**Flagged prominently for the user before launch.**

**Mutual-exclusion consequence:** `sequential_reprofile=true` is rejected
together with `profile_sidecar.enabled=true`
(`orchestrator.py:782-794`). The REAM configs MUST set
`profile_sidecar.enabled: false`.

**Remaining accepted deviations from upstream `pseudo_group`
(`ream/ream.py:21-34`):**
- Calibration data: upstream `c4+math+code @ 0/0.3/0.7` (`merge.py:59-67`); ours
  is `qwen3-pretrain-mix-v2`. **Accepted — the whole point of the probe is "on
  OUR calib".**
- Solver: upstream `pseudo_group` greedy nearest-centroid (`ream.py:74-94`); ours
  is `assignment_solver: "greedy"`. Parity intended; any algorithmic gap is a
  documented-not-fixed item for the R-phase reviewer.
- saliency tie-break / argsort order: upstream `np.argsort(saliency)[::-1][:k]`
  (`ream.py:64`); confirm our scoring picks the same top-K centroids — note as
  a verify-at-implement item.

## R6 — MEDIUM: post-run assertion that survivors == 166/layer (BOTH groups)

**Verified gap.** No survivor-count assertion exists in `orchestrator.py`
(grep for `num_experts ==` / `len(...kept)` / `survivor` returns only the
unrelated cost asserts at `:386,389,397`). Upstream HARD-asserts
`moe_layer.num_experts == self.merge_size` (`/tmp/ream_upstream/ream/merger.py:463`).

**Verified risk.** The bump loop raises the kept count above the configured
target when the cost gate fails: `orchestrator.py:480-491` logs
`"bumping target %d→%d"` and sets `effective_target = new_effective`, and the
post-loop fallback (`orchestrator.py:493+`) commits an above-target assignment as
last resort. So a `merge_size: 166` run can silently keep >166.

**Plan change (R-phase production code):** add a post-merge assertion — after the
per-layer merge commits, assert the realised survivor count == target (166) for
EVERY MoE layer, raising on mismatch (mirroring upstream `merger.py:463`). This
guards both the REAM bump-loop overshoot and the REAP K (R1). The driver/config
test additionally asserts the *intended* counts (REAP keep==166 via
prune_fraction derivation, REAM `merge_size==166`); the orchestrator assertion
guards the *realised* counts at run time.

> Caveat to document: with a hard 166 assertion AND the bump loop active, a
> cost-gate bump would now ABORT the run rather than silently overshoot. That is
> the desired by-the-book behaviour (upstream aborts on count mismatch). If the
> probe's cost gate is expected to bump, the bump loop must be configured inert
> for the REAM rows — note as an implement-time check.

## R7 — LOW / NITPICK

- **Piece 1 (RKD `paper_dials_only`) is DONE.** Already on
  `feat/probe-machinery` @ 7fbeed1, reviewed. Mark §1 complete; no R-phase work.
- **HF naming is CUSTOM, not the stage-idx default.** Required repos:
  `pirola/calib-v2-probe-{reap,ream}-{base,heal25,rkd}` (6 repos). This is NOT
  the `upload_stage_to_hub` default `f"{repo_base}-stage{stage_idx}"`
  (`utils/hub_upload.py:147`). The driver must compute the custom repo id per
  row and pass it explicitly (or set `PIPELINE_HUB_RESULT_REPO_BASE` per row);
  do NOT rely on the stage-idx suffix.
- **`capacity_gate` has no off-key — inert via `cost_alignment="pre"`.**
  `capacity_gate.py:23-25`: under `cost_alignment="pre"` the gate forces the pre
  path "regardless of the configured value". So setting `cost_alignment: "pre"`
  (already required for δ_REAM) makes capacity_gate inert; there is no separate
  toggle to set. Document, don't invent a key.
- **REAM rows set `skip_merge_percentile: 100.0` (OFF sentinel).**
  `skip_merge_floor.py:101-102` (100.0 = OFF). The faithful config ships
  `skip_merge_percentile: 0.0` (REAP-prune-exact); the REAM rows OVERRIDE to
  100.0 so merges are allowed. `two_opt_refine` is OFF when the key is
  falsy/absent (`two_opt_refine.py:215,239-240`,
  `config_key="stage2_reap_ream.two_opt_refine"`).
- **Config-grep test asserts ALL disable sentinels on the GENERATED configs**
  (not just source defaults): per the PAPER-CORE block —
  `em_refinement_rounds: 0`, `expert_distill_steps: 0`,
  `merge_heal_enabled: false`, `two_opt_refine` falsy, `skip_merge_percentile:
  100.0` (REAM) / inert (REAP), plus the R1 derived-keep==166 and R5 REAM params
  (`max_merge_group_size: 15`, `sequential_reprofile: true`,
  `cost_alignment: "pre"`, `frequency_weighted_merge: true`,
  `profile_sidecar.enabled: false`). Mirror `tests/test_reap_faithful_config.py`.

## R-phase work order (gated on this revised plan)

1. (R1) REAP keep==166: `reap_prune.py:309` `int()`→`round()`; regenerate
   faithful-prune goldens + fix `tests/test_run_pipeline_reap_exact.py` (167→166).
2. (R6) post-merge survivor==target assertion in `orchestrator.py`.
3. `run_probe.py` 6-row driver: REAP rows (`prune_fraction: 0.35`), REAM rows
   (`merge_size: 166` + R5 params), per-row `pipeline.stages` (R3/R4), custom HF
   repo ids (R7), PAPER-CORE OFF set.
4. Config-grep + driver-config + derived-count tests (R7).
5. CPU dry-run of the config-builder; hold for sidecar re-capture (B0); launch.
