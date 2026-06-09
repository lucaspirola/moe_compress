# Plan: Stage-4-aware solver + full-pipeline REAP-vs-REAM @ net-35%

Date: 2026-06-09. Status: **PLAN-REVIEW CLOSED (round 4 all-none) — READY TO IMPLEMENT.**

## Established facts (verified)
- **Solver gap** (2 independent agents): `budget/solver.py::solve()` originally hit the target on Stages 2+3 only; Stage-4 EoRA regrows ~`compensation_budget_pct`(0.03) × Stage-3 savings → final landed ~0.2–0.3pp UNDER target (≈34.7% for a 35% ask), one-directional. **NOW FIXED by WS1 (opt-in `net_of_eora`).**
- **Stage 2.5 is already router-kd** — `stage5_router_kd.run(stage_key="stage2p5")` → `router_kd.orchestrator.run`. Stage 5 is the same call (`stage_key="stage5"`). They differ by output-dir name, seed offset, and merge-repair (on only at 2p5).
- **Winner recipe** = `rkd_recipe: paper_dials_only` (rkd_paper_recipe.py:194-226): kd_temperature=4.0, weight_decay=0.0, epochs=2, early_stop_patience=0, teacher_logits_cache cleared. (LR stays config 5e-5 — paper recipe does NOT override it.)
- **REAP/REAM iso-count** (run_probe.py): both pinned to K=166 survivors/layer (35% expert reduction). REAP=`prune_mode=faithful_prune` (prune_fraction 0.35), REAM=`prune_mode=merge` to a uniform-166 `stage1_budgets.json` pin. Paper-core refinements OFF for both.
- The winning probe ran **Stage 2 → 2.5 → 6alt** (skipped 3/4/5). The new ask is the **full** solver-driven pipeline.

## GOAL (clarified 2026-06-09) — read first
Scientific question: at iso-35% compression, does **keeping more experts + compressing them with Stage-3 SVD + Stage-4 EoRA** beat **bluntly dropping/merging experts (paper, Stage-2 only)**?
- **Paper baselines ALREADY EXIST** (last ablation, run_probe.py): `reap-rkd` (student_bpt 3.1686), `ream-rkd`, etc. — 35% via Stage-2 alone (prune/merge to K_paper=166), + Stage-2.5 router-kd. **DO NOT rebuild these.**
- **NEW work = build 2 "ours" arms only:** `reap-s234` and `ream-s234`. Each: Stage-2 by-the-paper-only (faithful_prune / merge, refinements OFF) to **K_solver > 166** (lighter Stage-2; the new Stage-4-aware solver splits 35% across Stage 2+3+4), + Stage-2.5 rkd (paper dials) + Stage-3 SVD (solver sp) + Stage-4 EoRA + Stage-5 rkd (paper dials) → **net 35%** → Stage-6 thermometer.
- **Compare:** the 2 new arms' student_bpt vs the existing reap-rkd/ream-rkd baselines. reap-s234 vs reap-rkd, ream-s234 vs ream-rkd (each method against its own paper baseline), all at iso-35%.
- NOTE this REPLACES the earlier framing: K is NOT 166 for the new arms — it is the solver's K_solver (higher, since SVD+EoRA absorb part of the 35%). The "iso-K between REAP and REAM" still holds (both new arms share the same solver K_solver + sp); iso-K is NOT shared with the paper baselines (those are at 166).

## Decisions (from user)
1. RKD: make `paper_dials_only` the default recipe at BOTH 2.5 and 5 (the proven dials). [interpretation from "see the last ablation"; confirm]
2. Ablations: FULL pipeline for both (Stage2 → 2.5-rkd → 3-svd → 4-eora → 5-rkd → 6-eval), net 35%.
3. REAP & REAM: same per-layer survivor count (iso-count, paper methodology); differ only in Stage-2 method.

---

## Workstream 1 — Stage-4-aware solver — ✅ DONE (verify-only)
**STATUS: IMPLEMENTED + TESTED on main-track (commit pending review-loop close). Not "to do".**
- `budget/solver.py`: `solve(..., eora_overhead_pct=0.0)` outer fixed-point (`_EORA_OUTER_ITERS=5`, `_growth_frac`, `max_ach` clamp); `BudgetDecomposition` carries `eora_overhead_pct`/`target_net_reduction`/`projected_net_reduction`.
- `run_pipeline.py`: both `solve()` call sites pass `eora_overhead_pct` derived from `stage4_eora.compensation_budget_pct` gated on `config["target"]["net_of_eora"]`.
- `tests/test_budget_solver.py`: `test_eora_zero_reproduces_legacy` (golden byte-identical), `_populates_net_fields`, `_inflates_gross_target`, `_negative_raises`. 94 budget/stage1 tests green.
- Remaining WS1 action = NONE beyond confirming `net_of_eora:true` is set at runtime (that's WS3/M-C). Per the project workflow these landed tests will be re-blessed AFTER the fidelity + code-quality review loops on the WS1 code.

**Design (for reference — already built):** user's `target.total_reduction_ratio` = NET compression after EoRA regrowth.

**Design (budget/solver.py::solve):**
- Add an EoRA-overhead model to the projection. EoRA adds back ≈ `min(comp_pct × svd_savings, rank_cap_growth)`. Use the *budget-cap* form `eora_growth ≈ comp_pct × svd_savings` (upper bound on real growth, since the rank-128 cap only ever reduces it → lands at-or-slightly-over target, never under).
- Net projection:
  `projected_net_reduction = (expert_savings − eora_growth) / total_params`
  where `eora_growth = comp_pct × svd_savings`, `svd_savings = after_prune × sp`.
- Converge `projected_net_reduction → target` (replaces the current Stages-2+3-only convergence). Same iterative scale-both-knobs loop.
- Read `comp_pct` from `config["stage4_eora"]["compensation_budget_pct"]`; pass into `solve()` via `run_pipeline.py` call site (new kwarg, default 0.0 = old behavior). If Stage 4 disabled/absent → 0.0 → identical to today (backward compatible).
- Record both gross and net reduction in `BudgetDecomposition` + `budget_decomposition.json` for transparency.

**Tests:** unit test that with comp_pct=0.03 and target=0.35, the solver's gross target rises so projected_net≈0.35 (±0.5% tol); comp_pct=0 reproduces current numbers byte-identically (golden).

**Caveat to flag:** the gap (~0.3pp) is within the solver's own ±0.5% tolerance, so the practical effect is small — but it's a correctness fix and makes "35%" mean net-35%.

**EoRA growth model (plan-review L1):** at production dims (d_model=2048, moe_intermediate≈768, sp small ~0.07) the EoRA add-back is **budget-bound, not cap-bound** — `r_per_expert ≈ 0.03·saved/(N·(d_out+d_in)) ≈ 1`, so the `eigenspace_rank_cap=128` is INERT. `eora_growth = comp_pct × svd_savings` is thus a tight estimate; the only slack is sub-1-rank integer floor-division, which makes real growth ≤ estimate → solver lands at-or-slightly-OVER net target (safe direction), never under. The `max_achievable` clamp in the fixed-point cannot bite at 35% net (gross ~35.3% ≪ ceiling ~0.6·expert_frac).

## Plan-review log
- **Round 4 — CLOSING (reviewer ad77da6b): ALL FIVE CATEGORIES NONE.** Verified H-A uniform-K pin for both arms is conflict-free (prune_fraction→K AND budget→K both resolve to K; survivor guard + budget load both succeed), no internal contradictions, full 2→2.5→3→4→5→6 net-35% iso-K run unblocked. Verdict: READY. **Plan-review loop CLOSED.**
- **Round 1** (reviewer a58f683b): Critical none. High H1 (epochs=2 contradicts 2026-05-17 revert — documented; save_best is the load-bearing safety net), H2 (no shared base config — inject recipe in runner). Med M1 (set net_of_eora:true or WS1 bypassed), M2 (reuse write_uniform_budget + assert_survivors_match_target for exact iso-K), M3 (Stage-4 covariance wiring is a NEW integration surface). Low L1/L2 (EoRA cap inert; clamp can't bite — no action). Verdict: sound, no blocking design errors. → all High/Med folded into WS2/WS3 above.
- **Round 3** (reviewer a8f20129): Critical none. **High H3-1** (WS1 already implemented — plan framed it as to-do → rewrote WS1 as DONE/verify-only + fixed stale facts line 6), **High H3-2** (H-A needs the uniform-K budget pin for the REAP arm too, not just prune_fraction — survivor guard reads stage1_budgets.json for both → folded into H-A step 4). Med M-C (resolved; added config["target"] placement note). Low L3-1 (fp16 fallback dtype is legitimate — added clarification). Nit N3-1 (full config filenames — added), N3-2 (pass survivors=K explicitly — added). Verification-only: prune_fraction inverse exact, full 2→2.5→3→4→5→6 dir handoffs well-formed (_STAGE_IO), survivor guard pins exactly K. Verdict: ready once H3-1/H3-2 folded. → folded.
- **Round 2** (reviewer a41385c4): Critical none. **High H-A** (no code wires solver `ep`→K; REAP uses static prune_fraction, REAM uses stage1_budgets.json → arms diverge unless runner explicitly translates ep→{prune_fraction, uniform budget}), **High H-B** (MOE_SKIP_STAGE2_COV_SAVE=1 removes Stage-4 fallback → crash on sidecar miss). Med M-C (configs lack net_of_eora + still 0.30; --target-ratio doesn't set net_of_eora), M-D (minor line-cite drift; dials are rkd_paper_recipe.py:199-207, survivor-guard orchestrator.py:737, budget-read :820-823). Low L-E/L-F (WS1 solve() verified correct + backward-compat; WS2 injection sound — no action). Verdict: NOT READY — 2 HIGH blockers. → H-A, H-B, M-C folded into WS3 above; line cites corrected.

## Workstream 2 — RKD recipe = paper_dials_only at both positions
- **There is NO shared base config** (plan-review H2): only `qwen36_35b_a3b_30pct.yaml` carries `rkd_recipe`; the two configs WS3 uses (`reap_exact.yaml`, `reap_faithful.yaml`) have NO `rkd_recipe` key → default `"current"` via the plugin hardcode. Flipping `30pct.yaml` alone does NOT reach them.
- **Therefore:** the WS3 runner explicitly injects `cfg["stage5_router_kd"]["rkd_recipe"]="paper_dials_only"` for BOTH arms (mirroring how `run_probe.py` injects `PROBE_REAM_PARAMS`). Do NOT flip the plugin hardcode default (`rkd_paper_recipe.py:155,193`) — that would change every other run repo-wide.
- Stage 5 (`stage_key="stage5"`) reads the same `stage5_router_kd` block → honors the injected recipe. Both positions get paper dials.
- **H1 — epochs=2 cost + safety net (MUST document):** `paper_dials_only` sets `epochs=2`, which CONTRADICTS the documented 2026-05-17 revert (3→1 epochs; raw_kl rises 7× over training — late steps overfit). It is NOT a quality regression ONLY because `save_best: true` (present in all configs, untouched by the recipe) exports the EMA-best (~step250) and discards the late-step overfit. Cost: ~40 min/row of redundant 2nd-epoch teacher forwards. **Guard:** assert `save_best` is true whenever a paper recipe is active; never combine `paper_dials_only` with `save_best: false`.
- Keep the `epochs>1 + teacher_logits_cache` guard (paper recipe clears the cache — orchestrator.py:585).

## Workstream 3 — Full-pipeline REAP-vs-REAM @ net-35% harness
- The current `run_probe.py` only does 2→2.5→6. Need a full-pipeline runner (new `run_reap_ream_35pct.py`, reusing run_probe.py helpers) that for each of {reap, ream}:
  1. Solver (WS1, net-35%) → uniform per-layer survivor count K + Stage-3 sp.
  2. Stage 2: REAP=`faithful_prune` to K; REAM=`merge` to K (same K both).
  3. Stage 2.5 router-kd (paper_dials_only).
  4. Stage 3 SVD (solver sp), Stage 4 EoRA, Stage 5 router-kd (paper_dials_only).
  5. Stage 6 **thermometer** eval (stage6alt, student_bpt — comparable to the 3.1686 winner).
- Both arms consume the SAME solver output (same K, same sp) → clean iso-compression; only Stage-2 method differs.

### WS3 MUST-DO integration items (from plan-review)
- **M1 — turn WS1 ON:** set `target.net_of_eora: true` in both reap configs (or inject in the runner). Without it the solver runs gross-only and lands ~0.3pp under net-35% — defeating WS1. Both reap configs already carry `compensation_budget_pct: 0.03` + `eigenspace_rank_cap: 128`.
- **H2 — inject recipe:** runner sets `stage5_router_kd.rkd_recipe="paper_dials_only"` for both arms (see WS2).
- **M2 — enforce exact iso-K:** the merge bump-loop can overshoot the budget (stage2/orchestrator.py:734-736). Reuse `run_probe.py`'s `write_uniform_budget` to seed a uniform-K `stage1_budgets.json` for the MERGE arm, and set `assert_survivors_match_target: true` for BOTH arms. "Same solver output" alone does NOT guarantee equal K. REAP keep = `round_half_up((1-ep)*256)` (reap_prune.py:339); merge targets the per-layer pin (orchestrator.py:818-821); merge collapses on the intermediate-neuron axis so survivors keep identical `[d_model × moe_intermediate]` dims → Stage 3 (same sp) + Stage 4 act on identical shapes → equal final param count, IFF both hit exactly K.
- **M3 — Stage-4 covariance wiring (NEW integration surface):** WS3 is the FIRST harness to run Stage 4 here (the probe skipped 3/4/5). Stage 4 EoRA consumes the input covariance via `stage4/plugins/input_cov_cache.py` → `load_covariance(jsonl)` reading `sidecars/<stem>/covariance.pt`. Must wire the runner's data path to the **bf16 capture output** (`self_traces_489ee0e1b17b43b0.jsonl` + its `sidecars/.../covariance.pt`) being produced now on the box. Confirm the cache plugin finds it and `a_storage_dtype` resolves to bfloat16 (per the WS-input_cov fix).

### H-A — explicit ep→K→{prune_fraction, uniform budget} translation (round-2 blocker)
There is NO code path auto-deriving K from the solver's `ep`. REAP keep = `round_half_up((1-prune_fraction)*256)` from the **config-static** `prune_fraction` (reap_prune.py:241,247; reap_faithful.yaml:168=0.35); REAM keep = `per_layer_target_experts` from `stage1_budgets.json` (orchestrator.py:820-823). Only Stage-3 `sp` flows from the solver automatically (stage3/orchestrator.py:84,223). So the runner MUST translate explicitly:
  1. Run WS1 solver (net-35%, `net_of_eora:true`) → get `ep`, `sp`.
  2. `K = round((1-ep)*256)` (uniform per layer; model is homogeneous 256 experts/layer — reap_prune.py:322-323).
  3. REAP arm: inject `stage2.prune_fraction = 1 - K/256` so faithful keep == K (exact inverse of `n_keep=round_half_up((1-pf)*256)`, reap_prune.py:339).
  4. **BOTH arms (H3-2):** `write_uniform_budget(K)` → uniform `stage1_budgets.json` pin + `assert_survivors_match_target:true`. The survivor guard (orchestrator.py:646, :1716-1719) runs for BOTH arms and compares against `per_layer_target` from `stage1_budgets.json` (:820-821) — so REAP needs the uniform-K pin too, NOT just prune_fraction (else it crashes at :646). The probe does this for both groups via `seed_stage1_artifacts` (run_probe.py:262-264). Pass `survivors=K` EXPLICITLY to `write_uniform_budget` (default is PROBE_SURVIVORS=166).
  5. Both arms: `sp` already flows from the solver decomposition → identical Stage 3.
This makes K genuinely solver/net-35%-derived AND exactly iso-K across arms. (The earlier "ep directly becomes K" wording was wrong — it requires this explicit runner wiring.)

### H-B — Stage-4 covariance must resolve OR keep the fallback (round-2 blocker)
Stage 4 loads covariance cache-first with a disk fallback to `artifacts_dir/_stage2_input_covariance.pt` (eora_inputs.py:145,162). The probe sets `MOE_SKIP_STAGE2_COV_SAVE=1` (run_probe.py:50) which SKIPS writing that fallback (orchestrator.py:1724-1726). WS3 is the first harness to actually run Stage 4, so on a bf16-sidecar miss it would crash with neither cache nor fallback. Runner MUST:
  - **Fail-fast pre-check**: before launch, assert `load_covariance(jsonl)` (cached_calibration_signals.py:1452, standalone-callable, returns None on miss) resolves the bf16 `sidecars/<stem>/covariance.pt` (the capture output). On the sidecar-HIT path, `a_storage_dtype` is inferred from the loaded tensor dtype (input_cov_cache.py:91-94) → auto-resolves to bfloat16. **L3-1 nuance:** the bf16 assertion applies ONLY to the sidecar-hit path; the fp16 disk fallback (`_stage2_input_covariance.pt`, eora_inputs.py:152,160,162) is legitimately float16 (`covariance_storage_dtype: float16` under stage2_reap_ream) — it's the safety net, not the intended path.
  - **AND do NOT set** `MOE_SKIP_STAGE2_COV_SAVE=1` for the full-pipeline runner (belt-and-suspenders: keeps the fp16 fallback written at stage2/orchestrator.py:1724 if the sidecar ever misses).

### M-C — config wiring (round-2)
Both reap configs (`configs/qwen36_35b_a3b_reap_exact.yaml`, `configs/qwen36_35b_a3b_reap_faithful.yaml`) still have `total_reduction_ratio: 0.30` and NO `net_of_eora` key. `--target-ratio` overrides only the ratio, NOT `net_of_eora` (run_pipeline.py:89-90,181). Runner MUST inject BOTH `target.total_reduction_ratio=0.35` AND `target.net_of_eora=true` for both arms — placed under `config["target"]` (run_pipeline.py:181 reads `config["target"].get("net_of_eora", False)`). Both configs already carry `compensation_budget_pct: 0.03` + `eigenspace_rank_cap: 128`.

### Uniform-K (user decision)
K is UNIFORM per layer; derived from solver `ep` per H-A step 2. GRAPE non-uniform NOT used.

## Confirmations (RESOLVED by user 2026-06-09)
- WS2: paper dials at **BOTH 2.5 and 5**.
- WS3: Stage-6 evaluator = **thermometer (fast BPT)** (stage6alt, comparable to the 3.1686 winner).
- WS3: per-layer K = **UNIFORM** (by-the-book), e.g. K=round((1-ep)*256)/layer for every MoE layer. GRAPE non-uniform NOT used. → solver's `ep` is applied as a uniform prune/merge fraction; both arms share the same K.
- Run target: SAME vast H200 box (contract 40173566) after the input_cov capture finishes + uploads.

**Implication for WS1:** with uniform-K, the solver's `ep` directly becomes the uniform expert-drop fraction (K=round((1-ep)*256)); `sp` is the Stage-3 SVD ratio; both net of the WS1 EoRA-overhead term. The GRAPE non-uniform `_project_expert_budget` path is bypassed for these ablations (as in the probe).

## Sequencing
1. WS1 solver — ✅ DONE (code + tests landed; pending fidelity + code-quality review loops then re-bless).
2. WS3 harness (new `run_reap_ream_35pct.py`) — the bulk of remaining work; folds WS2 recipe injection + H-A/H-B/M-C. Implement CODE ONLY first (tests last per project workflow).
3. WS2 recipe — realized as runner injection inside WS3 (no standalone config flip).
4. Per project paper-fidelity-review-loop: implementer (code only) → fidelity review loop → code-quality review loop → tests last → commit FF-only. Then run on the box once the bf16 capture is uploaded.
