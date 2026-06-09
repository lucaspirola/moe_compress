# PLAN — 6-model healing probe machinery (CLEAN-CODE REVISION)

**Date:** 2026-06-07
**Branch:** `feat/probe-machinery`
**Status:** planning only — NO production code written yet (one prod change specified: §3).
**Design spec:** `tasks/2026-06-06-reap-ream-healing-probe-design.md`
**Repo state:** `feat/probe-machinery` @ tip; working tree CLEAN. All paths under
`max_quality/src/moe_compress/` unless noted.

> **This revision re-derives every file:line against the CLEAN committed code.** A
> prior agent's uncommitted edits (a `merge_size` config knob, a
> `pipeline.stages` per-stage enable map, `_resolve_stage_enables`) were stashed
> as a SUPERSEDED approach and are GONE from the tree. They DO NOT exist in the
> code. The earlier plan revision (eeffb1e) was written against those edits and
> cited non-existent symbols/lines. Everything below is verified present on clean
> code. The probe RUN is BLOCKED on calibration sidecar re-capture (separate
> issue, "B0"); this plan makes the machinery launch-ready. Nothing here runs on
> GPU.

---

## 0. SCOPE (LOCKED — no open decisions)

6 models, ALL on `qwen3-pretrain-mix-v2`, 35% fewer experts, **single survivor
count K = 166/layer for BOTH groups** (drop = `round(0.35 × 256) = 90`). Paper-core
= NO refinements. Two groups × three arms:

| arm | group REAP | group REAM | heal mechanism |
|---|---|---|---|
| base (no heal) | faithful_prune, prune_fraction=0.35 | merge, budget-pin 166 | none |
| heal25 (stage-2.5) | + auto Stage-2.5 router-KD | + auto Stage-2.5 router-KD | current dials, our calib |
| rkd (paper-dials) | + Stage-2.5 router-KD | + Stage-2.5 router-KD | `rkd_recipe: paper_dials_only` |

HF repos: `pirola/calib-v2-probe-{reap,ream}-{base,heal25,rkd}` (6). stage6alt
wikitext BPT eval each. Winner = lowest BPT → HF + local download.

---

## 1. heal-on / SVD-off uses the EXISTING mechanism — NO new code

The clean pipeline has only the binary `pipeline.skip_intermediate_stages` flag
plus `--resume-from-stage` / `--stop-after-stage` windowing. There is **no**
`pipeline.stages` map and **no** `_resolve_stage_enables`. The probe's three arms
map onto this existing machinery exactly:

### 1a. base (no-heal) rows — ONE invocation, `skip_intermediate_stages=true`

Config sets `pipeline.skip_intermediate_stages: true` (faithful config default,
`configs/qwen36_35b_a3b_reap_faithful.yaml:678`). Then:
- Stage 2.5 is skipped — `run_pipeline.py:253-257` (`if _skip_intermediate: skip_stage25=True`).
- The "stop after stage N" early-returns at `run_pipeline.py:270, 282, 293, 305`
  are each guarded `and not _skip_intermediate`, so with the flag set they are
  ALL bypassed; control flows past Stages 3/4/5 straight to Stage 6 at
  `run_pipeline.py:310` (`if start <= 6 <= stop:`).
- The evaluator is forced to thermometer: `pipeline.evaluator: stage6alt`
  (`...faithful.yaml:683`) → `run_pipeline.py:107` sets
  `stage6_validate.mode = "thermometer"` → `run_pipeline.py:315-317` calls
  `stage6alt_thermometer.run(...)`, which writes **`stage6alt_eval.json`**.

So base rows run with default `--stop-after-stage 6` (do **not** pass
`--stop-after-stage 2`; `run_pipeline.py:121-128` warns that `skip_intermediate +
stop<6` never reaches the evaluator). SVD/Stage3/4/5-proper never run. SINGLE
process, no resume chain, no `--skip-stage2p5` (that bail path,
`run_pipeline.py:258-262`, is a split-machine feature the probe must NOT use).

### 1b. heal arms (heal25 and rkd) — TWO invocations, `skip_intermediate_stages=false`

This is exactly `run_ablations.py`'s shape (Stage 1→2→auto-2.5, then resume-6):

1. **First process:** `python -m moe_compress.run_pipeline --config <cfg>
   --artifacts-dir <dir> --resume-from-stage 2 --stop-after-stage 2`
   (mirrors `run_ablations.py:640-651`). With `skip_intermediate_stages=false`:
   Stage 2 runs, then the auto Stage-2.5 router-KD runs immediately
   (`run_pipeline.py:263-269`: `stage5_router_kd.run(..., stage_key="stage2p5")`),
   then the `stop < 3 and not _skip_intermediate` early-return at
   `run_pipeline.py:270` fires → process exits after writing `stage2p5_final/`.
   (`run_ablations.py:632-633` confirms "`--stop-after-stage 2` runs both" Stage 2
   AND Stage 2.5.) `--resume-from-stage 2` is used because Stage 1 artifacts —
   incl. the uniform-166 budget pin (§2) — are pre-seeded into the dir, not
   re-solved.
2. **Second process:** `python -m moe_compress.run_pipeline --config <cfg>
   --artifacts-dir <dir> --resume-from-stage 6 --stop-after-stage 6`
   (mirrors `run_ablations.py:680-687`). Stage 6 loads `stage2p5_final/` via the
   resume fallback at `run_pipeline.py:520-532` and runs `stage6alt_thermometer`
   (mode=thermometer), writing `stage6alt_eval.json`.

This skips SVD/Stage3/4/5-proper: with `skip_intermediate_stages=false` the only
post-Stage-2 work in the `--stop-after-stage 2` window is the auto Stage-2.5; the
second process resumes directly at Stage 6.

> **Note on the eval-mode override.** The `pipeline.evaluator → stage6_validate.mode`
> override at `run_pipeline.py:102-113` is gated by `if _skip_intermediate:`. For
> the heal arms `skip_intermediate_stages=false`, so that override block does NOT
> run — the heal configs MUST set `stage6_validate.mode: thermometer` **directly**
> (not rely on `evaluator: stage6alt`). The base configs get it via the override;
> the heal configs set it explicitly. (Verify both at config-gen.)

### 1c. arm → rkd_recipe wiring

Both heal arms run through the SAME auto Stage-2.5 call
(`run_pipeline.py:266`, `stage_key="stage2p5"`). They differ ONLY by the
`stage5_router_kd.rkd_recipe` config key:
- **heal25** = deprecated production router-KD. **[2026-06-09:** the absent-key default flipped `current → "paper_dials_only"`, so `run_probe.py` now pins `rkd_recipe: "current"` EXPLICITLY for heal25 — absent would otherwise become a second paper arm.]
- **rkd** = `stage5_router_kd.rkd_recipe: "paper_dials_only"` (paper dials, OUR
  calib — already implemented, 7fbeed1).

**Verified the override is honored on the stage2p5 path:**
`router_kd/orchestrator.py:180` calls `RkdPaperRecipePlugin().apply_config_overrides(config)`
as the FIRST statement of `run()`, BEFORE any `stage_key` branching
(`router_kd/orchestrator.py:187`). The plugin reads
`config["stage5_router_kd"]["rkd_recipe"]` (`rkd_paper_recipe.py:193`) and applies
`paper_dials_only` (`rkd_paper_recipe.py:194` accepts both `"paper"` and
`"paper_dials_only"`; the wikitext swap is gated to `"paper"` only). So the same
single Stage-2.5 invocation serves both heal arms; no per-stage map needed.

---

## 2. REAM expert count = 166 via a UNIFORM `stage1_budgets.json` pin

**Mechanism: pin `per_layer_target_experts[layer] = 166` for EVERY MoE layer in
`stage1_budgets.json` before Stage 2 runs.** A ~20-line helper in `run_probe.py`
writes it; NO orchestrator change, NO config knob.

**Verified the merge path reads `per_layer_target` from `stage1_budgets.json`:**
- `stage2/orchestrator.py:785-789`:
  `stage1_budget_path = artifacts_dir / "stage1_budgets.json"`;
  `budgets_payload = load_json_artifact(...)`;
  `per_layer_target = {int(k): int(v) for k,v in budgets_payload["per_layer_target_experts"].items()}`.
- `stage2/orchestrator.py:1647`: `target = per_layer_target[layer_ref.layer_idx]`
  inside the per-layer loop; pushed to ctx at `:1656` (`ctx.set("target", target)`)
  and consumed by the merge bump-loop as the survivor count (`_run_assignment`,
  `target = ctx.get("target")` at `:249`).
- `stage1_budget_path` is an overridable kwarg of `stage2/orchestrator.run`
  (signature `:296-304`, `stage1_budget_path: Path | None = None`).

So writing a uniform 166 into `per_layer_target_experts` forces every layer's
merge `target` to 166 → kept = 166. This is the reviewer-approved budget-pin
("Option A"), realised via the artifact GRAPE already consumes — NOT a new knob.
The probe's helper mirrors `run_ablations.py:525-538` `_seed_stage1_artifacts`
(which hardlinks `stage1_budgets.json` per-dir); the probe instead WRITES a
166-pinned `stage1_budgets.json` (plus copies `stage1_blacklist.json` +
`budget_decomposition.json` from the shared Stage-1 run).

> 166 > the `min_experts_per_layer` floor (`num_routed_experts // 2 = 128`), so
> the floor never binds. The helper must also write a `stage1_blacklist.json`
> (read at `orchestrator.py:790-791`); use the shared Stage-1 blacklist.

> **REAM rows set `prune_mode: merge`** (default; the faithful config ships
> `faithful_prune`). The merge path is the only one that consumes
> `per_layer_target`; faithful_prune bypasses it (§3).

---

## 3. REAP count = exactly 166 — the ONLY production code change

**Verified bug.** `stage2/plugins/reap_prune.py:309`:
```python
self._n_prune = int(n_experts * self.prune_fraction)
```
`int(256 × 0.35) = int(89.6) = 89` dropped → **167 kept**, off-by-one vs REAM's 166.

**Change (one line, `reap_prune.py:309`):** derive the drop count so it keeps
exactly `round((1 − 0.35) × 256) = 166` (drop 90). Equivalent forms:
```python
# keep round((1-f)*N) → drop = N - round((1-f)*N)
self._n_prune = n_experts - round(n_experts * (1.0 - self.prune_fraction))
```
For N=256, f=0.35: `256 - round(166.4) = 256 - 166 = 90` → keeps 166. (Simpler
`round(n_experts * self.prune_fraction)` = `round(89.6) = 90` also yields 166 and
is a more minimal diff; pick at implement time — the keep-rounded form above is
the more faithful "keep round((1-f)·N)" statement of intent.) The configs set
`prune_fraction: 0.35`.

**New test (R-phase, with its own code-review loop):** assert the derived count
for the production case — `n_experts=256, prune_fraction=0.35` → drop == 90, keep
== 166. Verified NO existing test breaks: `test_reap_prune_golden.py` uses
fractions giving exact integers (8×0.5=4), `test_reap_prune_upstream_formula.py`
passes `n_prune` directly (bypasses the derivation),
`test_run_pipeline_reap_exact.py` uses tiny synthetic dims. The change is
behaviour-changing ONLY when `n_experts × fraction` is non-integer — inert for the
current suite, behavioural for the real 256-expert config. The R-phase MUST also
regenerate any faithful-prune golden keyed on the 256 count (167→166) if present.

**This is a real production code change → its own review/fix loop (all 5
categories incl. nitpick).**

---

## 4. REAM by-the-book params (set in the REAM configs)

Verified upstream defaults (`SamsungSAILMontreal/ream`) vs our knobs:

| upstream param | upstream default | our knob (verified) | probe value |
|---|---|---|---|
| `group_size` (C) | 16 (total incl. centroid) | `stage2_reap_ream.max_merge_group_size` | **15** (see ⚠1) |
| `sequential` | True | `stage2_reap_ream.sequential_reprofile` (`ream_sequential.py:181,209`, default false) | **true** |
| grouping/cost (δ_REAM) | logits+weights, gated | `cost_alignment: "pre"` (`ream_cost.py:238,342`) | **"pre"** |
| freq-weighted merge | (REAM merge op) | `stage2_reap_ream.frequency_weighted_merge` (config `:206`) | **true** |

**⚠1 group_size semantics (CORRECTS "set C=16").** Our `max_merge_group_size`
counts **non-centroids only**; upstream `group_size` counts the TOTAL group incl.
the centroid (`layer_merge.py:55-71`). Equivalence: `max_merge_group_size = N` ⇔
upstream `group_size = N+1`. Matching upstream `group_size = 16` ⇒
**`max_merge_group_size: 15`** (NOT 16). Our config default is 8
(`...faithful.yaml:194`, `layer_merge.py:77`).

**⚠2 sequential = slower/heavier GPU (FLAG FOR USER).** `sequential_reprofile:
true` re-forwards each layer after every merge (`ream_sequential.py`), the
by-the-book upstream default but materially slower/heavier than the one-shot
pre-collected-stats path. Flagged.

**⚠3 mutual-exclusion (verified).** `sequential_reprofile=true` +
`profile_sidecar.enabled=true` is HARD-rejected at
`stage2/orchestrator.py:759-767`. The REAM configs MUST set
`profile_sidecar.enabled: false`.

**Remaining deviations from upstream `pseudo_group`, documented-not-fixed:**
- Calibration: ours `qwen3-pretrain-mix-v2` vs upstream c4+math+code. ACCEPTED —
  the whole point of the probe.
- `use_gate_output`/`gated_sim`: no separate knob; baked into δ_REAM under
  `cost_alignment="pre"` (`ream_cost.py`). Parity by construction.
- Greedy assignment / saliency tie-break / argsort order: confirm our
  `assignment_solver: "greedy"` + REAP scoring pick the same top-K centroids —
  a verify-at-implement item for the R-phase reviewer.

---

## 5. Paper-core = ALL 6 refinements OFF in ALL configs (explicit, not defaults)

| refinement | config key | probe value | verified |
|---|---|---|---|
| EM refine | `stage2_reap_ream.em_refinement_rounds` | `0` | `em_refine.py:25` |
| expert distill | `stage2_reap_ream.expert_distill_steps` | `0` | `expert_distill.py:865-866` |
| merge heal | `stage2_reap_ream.merge_heal_enabled` | `false` | `merge_heal.py:99` |
| two-opt | `stage2_reap_ream.two_opt_refine` | falsy/absent | `two_opt_refine.py:215` (config_key) |
| capacity gate | (no off-key) inert via `cost_alignment: "pre"` | — | `capacity_gate.py:25,100,110` ("`pre` regardless") |
| skip-merge floor | `stage2_reap_ream.skip_merge_percentile` | **100.0** (REAM rows) | `skip_merge_floor.py:39` (enabled iff `< 100.0`) |

> The faithful config ships `skip_merge_percentile: 0.0` (`:232`, REAP-prune-exact);
> the REAM rows OVERRIDE to **100.0** (OFF sentinel) so merges proceed (the whole
> point of REAM). The driver applies this as a delta.

> REAP group: faithful_prune structurally bypasses the merge machinery — these are
> inert. Set them anyway for clarity; note `orchestrator.py:719-728` already
> HARD-rejects `expert_distill_steps>0` and `merge_heal_enabled=true` under
> faithful_prune, so the REAP configs MUST keep them at 0/false.

**Test:** config-grep over the GENERATED 6 configs asserts every disable sentinel
above PLUS the §4 REAM params (`max_merge_group_size: 15`,
`sequential_reprofile: true`, `cost_alignment: "pre"`,
`frequency_weighted_merge: true`, `profile_sidecar.enabled: false`) on REAM rows,
and §3's derived keep==166. Mirror `tests/test_reap_faithful_config.py`.

---

## 6. Post-run assertion: survivors == 166/layer (BOTH groups)

**Verified gap.** No survivor-count assertion exists in `stage2/orchestrator.py`
(grep for `num_experts ==` / `== target` / `survivor` → none). Upstream
HARD-asserts `moe_layer.num_experts == self.merge_size`.

**Verified risk.** The merge bump loop RAISES the kept count above the configured
target when a feasibility/cost gate fails: `orchestrator.py:465-487`
(`bump = max(1, ceil(...))`; `effective_target = new_effective`; logs
`"bumping target %d→%d"`). So a 166-pinned REAM run can silently keep >166.

**Plan change (R-phase production code, part of §3's review loop or its own):**
add a post-merge assertion — after each layer's merge commits, assert the realised
survivor count == target (166) for EVERY MoE layer, raising on mismatch (mirrors
upstream). Guards both the REAM bump-loop overshoot AND the REAP K. The
config/driver test asserts the INTENDED counts; the orchestrator assertion guards
the REALISED counts at run time.

> Caveat: a hard 166 assertion + an active bump loop means a cost-gate bump
> ABORTS rather than silently overshooting (desired by-the-book behaviour). If the
> REAM cost gate is expected to bump, it must be configured inert for the probe —
> implement-time check.

---

## 7. run_probe.py — orchestration (new file, near-clone of run_ablations.py)

6 rows: `(reap|ream) × (base|heal25|rkd)`. Reuses from `run_ablations.py`: the
env preamble, deep-copy-base + apply-deltas config builder
(`run_ablations.py:214-285`), idempotent skip on existing `stage6alt_eval.json`
(`_stage6_artifact`/`_is_complete`, `:286-305`), leaderboard ranked by BPT
ascending (`:314-377`), and the per-dir Stage-1 seeding pattern (`:525-538`).

Per-row driver logic:
- **Stage-1 seed:** write `stage1_budgets.json` per row. REAM rows: UNIFORM 166
  pin (§2). REAP rows: faithful_prune ignores `per_layer_target`, so any valid
  budget works — but for symmetry seed the same shared Stage-1 budget (the
  REAP keep is fixed by §3's derivation, not the budget). Copy
  `stage1_blacklist.json` + `budget_decomposition.json` from the shared Stage-1.
- **Stage-2 deltas:** REAP rows `prune_mode: faithful_prune`, `prune_fraction: 0.35`.
  REAM rows `prune_mode: merge` + §4 params + §5 OFF set + `skip_merge_percentile: 100.0`.
- **arm windowing:** base rows = `skip_intermediate_stages: true`, single
  `--stop-after-stage 6` (§1a). heal rows = `skip_intermediate_stages: false`,
  `stage6_validate.mode: thermometer` set directly, two processes
  (`--stop-after-stage 2`, then `--resume-from-stage 6`) per §1b. heal25:
  `rkd_recipe: "current"` (pinned explicitly since the 2026-06-09 default flip;
  was absent); rkd: `rkd_recipe: "paper_dials_only"`.
- **subprocess, not in-process** (memory; `run_ablations.py:634-639,677-679`).

**Launch (after sidecar B0):**
```bash
MOE_SKIP_STAGE2_COV_SAVE=1 \
python -m moe_compress.run_probe \
    --config configs/qwen36_35b_a3b_reap_faithful.yaml \
    --artifacts-dir ./artifacts/probe \
    --num-sequences 4000
```
(`MOE_SKIP_STAGE2_COV_SAVE=1` per the faithful config header — faithful mode
collects no covariance.)

---

## 8. Per-model HF upload + winner-pick + local download

- **Custom repo ids (NOT the stage-idx default).** Required:
  `pirola/calib-v2-probe-{reap,ream}-{base,heal25,rkd}` (6). This is NOT
  `upload_stage_to_hub`'s `f"{repo_base}-stage{stage_idx}"`
  (`utils/hub_upload.py:147`); the driver computes the custom id per row (set
  `PIPELINE_HUB_RESULT_REPO_BASE` per row, or pass the repo id explicitly). Each
  row uploads its model + `stage6alt_eval.json`.
- **stage6alt wikitext eval each.** All 6 configs use `thermometer.corpus:
  wikitext` (`thermo_corpus.py:195,205-233`) + the pinned SHA
  `dataset_revisions.wikitext_ppl` (`...faithful.yaml:632`, the
  `b08601e0...` Salesforce/wikitext sha) — inherited from the faithful base.
- **Winner = lowest BPT** (reuse `run_ablations.py`'s leaderboard ranked
  ascending; lower = less damage). Winner model → `pirola/calib-v2-probe-winner`
  (or copy/tag) + local `huggingface_hub.snapshot_download` into
  `artifacts/probe_winner/`. Add `--download-winner-to PATH` (default off so a
  headless spot run doesn't block on a large download).

---

## 9. Test plan (all CPU, no GPU)

1. **Piece 1 (RKD `paper_dials_only`): DONE** on `feat/probe-machinery` @ 7fbeed1.
   Existing `tests/test_router_kd_plugin_rkd_paper_recipe.py` covers it. No work.
2. **§3 REAP keep==166:** new derived-count test (256, 0.35 → drop 90 / keep 166);
   own review loop.
3. **§5 + §4 config-grep test** over the 6 GENERATED configs (mirror
   `tests/test_reap_faithful_config.py`): all disable sentinels + REAM params +
   `thermometer.corpus == wikitext` + non-null `wikitext_ppl` SHA + heal rows have
   `stage6_validate.mode == thermometer`.
4. **§7 driver-config test:** the 6-row spec produces configs with expected
   `prune_mode` / `prune_fraction` / `rkd_recipe` / `skip_intermediate_stages` per
   arm; REAM rows get the uniform-166 budget pin written.
5. **No-regression:** golden snapshots stay byte-identical (the `paper_dials_only`
   branch is additive; `current`/`paper` paths untouched; §3 is inert for the
   current suite).

---

## 10. Work order

1. **(prod, §3)** REAP keep==166 at `reap_prune.py:309` + derived-count test →
   own review/fix loop (all 5 categories).
2. **(prod, §6)** post-merge survivor==target assertion in `orchestrator.py` →
   review loop (can fold into §3's loop or its own).
3. **(driver, §7)** `run_probe.py`: 6-row builder, uniform-166 budget-pin helper
   (§2), per-arm windowing (§1), custom HF repo ids (§8), PAPER-CORE OFF set (§5),
   REAM by-the-book params (§4).
4. **(tests, §9)** config-grep + driver-config + derived-count tests.
5. CPU dry-run of the config builder (no model load); hold for sidecar B0; launch.

---

## Appendix — clean-code file:line evidence index

- Skip-intermediate → Stage-6 flow: `run_pipeline.py:95,102-113,253-257,270,282,293,305,310,315-317`
- Skip+stop<6 warning: `run_pipeline.py:121-128`
- Auto Stage-2.5 (stage_key=stage2p5): `run_pipeline.py:263-269`
- Stage-6 resume fallback (stage2p5_final): `run_pipeline.py:520-532`
- RKD override on stage2p5 path: `router_kd/orchestrator.py:158-194` (call at :180, stage_key at :187); `rkd_paper_recipe.py:155,193-194`
- heal-arm two-process shape (precedent): `run_ablations.py:632-633,640-651,680-687`
- REAM budget read / per-layer target pin: `stage2/orchestrator.py:296-304(run sig + stage1_budget_path),785-789,1647,1656`; `_run_assignment` target: `:249`
- REAP drop derivation (the §3 change): `stage2/plugins/reap_prune.py:309`
- faithful_prune validation (rejects distill/heal): `stage2/orchestrator.py:694-729`
- sequential/profile_sidecar mutual-excl: `stage2/orchestrator.py:759-767`
- bump-loop overshoot (§6 risk): `stage2/orchestrator.py:465-487`
- max_merge_group_size non-centroid semantics: `stage2/plugins/layer_merge.py:55-71,77`
- sequential_reprofile knob: `stage2/plugins/ream_sequential.py:181,209`
- cost_alignment="pre" (δ_REAM): `stage2/plugins/ream_cost.py:238,342`
- capacity_gate inert under pre: `stage2/plugins/capacity_gate.py:25,100,110`
- skip_merge floor OFF sentinel: `stage2/plugins/skip_merge_floor.py:39`
- two_opt config_key: `stage2/plugins/two_opt_refine.py:215`
- thermo wikitext branch: `stage2 → stage6alt/plugins/thermo_corpus.py:195,205-233`
- faithful config keys: `configs/qwen36_35b_a3b_reap_faithful.yaml:159,166,194,206,220,232,236,241,259,549-550,632,678,683`
- Stage-1 seeding precedent: `run_ablations.py:525-538`
- Hub upload custom-id note: `utils/hub_upload.py:147`

---

## DELETED vs prior revision (eeffb1e) — stashed superseded mechanisms

Removed entirely (these symbols DO NOT exist in clean code — verified by grep
returning nothing):
- **`stage2_reap_ream.merge_size`** config knob + its orchestrator validation
  (claimed `orchestrator.py:731-756`) and pin (claimed `:1674-1686`). GONE.
  Replaced by the uniform `stage1_budgets.json` pin (§2), which is the actual
  clean mechanism (`orchestrator.py:785-789,1647`).
- **`pipeline.stages` per-stage enable map** + **`_resolve_stage_enables`**
  (claimed `run_pipeline.py:434-493`) + **`_all_intermediate_off`** (claimed
  `:104,296-322`). GONE. Replaced by the existing `skip_intermediate_stages` +
  stop/resume windowing (§1).
- All OD-1..OD-4 / R2 / R3 / R4 prose predicated on the above.
