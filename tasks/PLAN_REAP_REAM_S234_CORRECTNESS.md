# Audit — REAP-vs-REAM @ net-35% runner correctness (S2→3→4→5→6alt)

**Target file:** `max_quality/src/moe_compress/run_reap_ream_35pct.py` (landed main `a0a8a7d`).
**Branch:** `plan/reap-ream-s234-correctness` off main `d56bc75`.
**Method:** read the runner completely + every load-bearing helper (`run_probe.py`,
`budget/solver.py`, `stage2/orchestrator.py`, `run_pipeline.py`,
`router_kd/plugins/rkd_paper_recipe.py`, `stage2/plugins/reap_prune.py`,
`utils/cached_calibration_signals.py`). All file:line cites verified with grep/Read.

**HEADLINE VERDICT: the runner is CORRECT on all three concerns. ONE optional
cleanup (CONCERN 2): the explicit `rkd_recipe="paper_dials_only"` injection is now
redundant because it equals the live default — harmless but no longer load-bearing.
No structural rewrite. Fix-in-place is sufficient; the single edit is OPTIONAL.**

---

## CONCERN 1 — Stage-1 leakage — VERDICT: CORRECT (high confidence)

**What Stage-1 code actually runs in this ablation: NONE in the per-arm pipeline.**

The runner drives each arm with exactly two `run_pipeline` subprocesses
(`_pipeline_argv`, run_reap_ream_35pct.py:439-447), invoked at:
- `run_one_arm` resume=2 stop=2 (Stage 2 + auto Stage 2.5) — line 488-491
- `run_one_arm` resume=3 stop=6 (Stage 3→4→5→6) — line 502-505

In `run_pipeline.py`, Stage 1 is gated on the resume start index:

> run_pipeline.py:118  `start = args.resume_from_stage`
> run_pipeline.py:170  `if start <= 1 <= stop:`  ← Stage 1 GRAPE/RCO body
> run_pipeline.py:213  `else:` → loads `budget_decomposition.json` from disk (214-219)

Both subprocesses pass `start ∈ {2,3}`, so `start <= 1` is **always false** ⇒ the
Stage-1 GRAPE/RCO/Super-Expert/CKA body (run_pipeline.py:171-212, incl. the two
`budget_solver.solve` calls + `stage1.run`) **never executes per arm**. The only
solver call that happens is the runner's own one-shot `derive_solver_budget`
(run_reap_ream_35pct.py:600), on the CPU base model, purely to get K+sp; it is freed
before any subprocess (lines 603-610).

**Where the budget comes from instead — the seeded uniform pin:**
- `seed_stage1_artifacts(arm_dir, shared_dir, group="ream", survivors=budget["K"])`
  (run_reap_ream_35pct.py:483) copies `stage1_blacklist.json` +
  `budget_decomposition.json` verbatim from `_shared/` and writes a **uniform-K**
  `stage1_budgets.json` via `write_uniform_budget` (run_probe.py:280-286 →
  233-257): `per_layer_target_experts[layer] = K` for **every** MoE layer
  (run_probe.py:253-255).
- Stage 2 reads that pin: `per_layer_target = {int(k): int(v) for ...
  budgets_payload["per_layer_target_experts"]}` (stage2/orchestrator.py:820-822),
  and per layer `target = per_layer_target[layer_ref.layer_idx]`
  (stage2/orchestrator.py:1680).

**Can non-uniform allocation sneak in? No.**
1. GRAPE never runs (gated out, above), so it cannot override the pin.
2. The pin itself is uniform-by-construction (single scalar K written to every
   layer key).
3. REAP keep-count is derived from the scalar `prune_fraction`, not per-layer:
   `n_keep = math.floor(n_experts*(1-prune_fraction)+0.5)` (reap_prune.py:339) with
   `prune_fraction = 1 - K/n_experts` (run_reap_ream_35pct.py:250, 324) ⇒
   `n_keep == K` for every layer (integer K, no tie). Verified by the parametrized
   test `test_prune_fraction_is_exact_inverse_of_K` (test_run_reap_ream_35pct.py:118-129).

**The survivor guard is wired and enforced for BOTH arms:**
- `build_arm_config` sets `assert_survivors_match_target = True` unconditionally
  (run_reap_ream_35pct.py:319) — applied before the REAP/REAM branch, so both arms
  carry it.
- Orchestrator reads it: `_assert_survivors = bool(s2.get("assert_survivors_match_target", False))`
  (stage2/orchestrator.py:737) and enforces per layer:
  `if _assert_survivors: _assert_survivor_count(len(ctx.get("final_kept_ids")), target, ..., faithful=_faithful_prune)`
  (stage2/orchestrator.py:1716-1720). The guard hard-raises on `n_survivors != target`
  (stage2/orchestrator.py:629-657) for BOTH `faithful_prune` and `merge`. So even a
  hypothetical divergence aborts rather than silently producing non-uniform counts.

**Conclusion:** uniform by-the-book (same K every layer) is guaranteed end-to-end
for both REAP and REAM. The user's "saw it running something stage-1-related" is
explained by the runner's **own** one-shot `derive_solver_budget` solver call
(run_reap_ream_35pct.py:589-600, log line "Loading base model to derive the solver
budget") — that is the legitimate (b) solver-for-K step, NOT Stage-1 GRAPE
allocation, and it does not touch per-layer counts.

---

## CONCERN 2 — Stage 2.5 / Stage 5 must use CURRENT defaults — VERDICT: CORRECT (high confidence); ONE optional cleanup

**Live default for `rkd_recipe` (post-959fab3, 2026-06-09):** `"paper_dials_only"`.

> rkd_paper_recipe.py:209  `recipe = s5.get("rkd_recipe", "paper_dials_only")`
> rkd_paper_recipe.py:205-208 comment: "DEFAULT FLIPPED 2026-06-09: absent
> rkd_recipe now resolves to 'paper_dials_only' ... reached only by an EXPLICIT
> rkd_recipe: 'current'".

The paper recipe applies exactly these 4 dials (rkd_paper_recipe.py:213-218):

> `s5["kd_temperature"] = 4.0`
> `s5["weight_decay"] = 0.0`
> `s5["epochs"] = 2`
> `s5["early_stop_patience"] = 0`

plus a defensive `teacher_logits_cache=None` (line 223). The calibration-source
swap to wikitext is applied **only** for the full `"paper"` recipe, NOT
`"paper_dials_only"` (rkd_paper_recipe.py:230-242) — so the project's own
calibration source is preserved.

**The runner injects `rkd_recipe = "paper_dials_only"`**
(run_reap_ream_35pct.py:333). This is **identical to the live default**. The recipe
is honored at BOTH Stage 2.5 and Stage 5 (single config key consumed by the
orchestrator — orchestrator.py:178-181).

**Does the runner override any OTHER Stage 2.5 / Stage 5 param? No.**
- `grep` of the runner shows the ONLY `stage5_router_kd` write is line 333
  (`rkd_recipe`). It sets no `kd_temperature`, `weight_decay`, `epochs`,
  `early_stop_patience`, `save_best`, `lr`, etc.
- `assert_paper_recipe_safety` (run_reap_ream_35pct.py:348-367) only READS
  `save_best`/`rkd_recipe`; it mutates nothing. Its default branch matches
  early_stop.py's `save_best` default True, so it never trips on omission.
- Base config `qwen36_35b_a3b_reap_faithful.yaml` already has `save_best: true`
  (line 422), so the safety net is satisfied by the real config.

**The user's rule ("run on CURRENT DEFAULT parameters") is satisfied:** the injected
value equals the current default, and no other 2.5/5 param is overridden.

**OPTIONAL cleanup (NOT a bug):** since the default flip, line 333's explicit
injection is redundant. Per the user's stated *best outcome* ("inject nothing, or
inject exactly the current default"), both the current code and the no-op variant
are compliant. Recommend KEEPING the explicit injection for robustness against a
future default re-flip (same rationale run_probe.py:206-209 documents for its `rkd`
arm). If the user prefers the minimal form, the edit is:

  - File: `run_reap_ream_35pct.py:333`
    BEFORE: `cfg.setdefault("stage5_router_kd", {})["rkd_recipe"] = "paper_dials_only"`
    AFTER (option A, keep + comment): add inline note "== live default since 959fab3
    (2026-06-09); pinned explicitly for robustness to default re-flips".
    AFTER (option B, drop): delete line 333 and the docstring WS2 bullet; rely on the
    default. **Not recommended** (loses robustness; the safety net at line 361 already
    assumes the paper recipe is active).

**Recommendation: KEEP as-is (CORRECT). At most apply option-A comment for clarity.**

---

## CONCERN 3 — by-the-book / iso-compression / covariance — VERDICT: CORRECT (high confidence)

**(a) REAP faithful_prune, exact inverse, uniform K:**
`s2["prune_mode"]="faithful_prune"`, `s2["prune_fraction"]=1-K/n_experts`
(run_reap_ream_35pct.py:321-324; `prune_fraction` computed at :250). Inverse is exact
(`n_keep==K`, reap_prune.py:339; test :118-129). Single scalar ⇒ uniform.

**(b) REAM uniform merge-to-K:** `s2.update(_REAM_BY_THE_BOOK)`
(run_reap_ream_35pct.py:326; params at run_probe.py:123-133:
`prune_mode="merge"`, `max_merge_group_size=15`, `sequential_reprofile=True`,
`cost_alignment="pre"`, `skip_merge_percentile=100.0`) + `profile_sidecar.enabled=False`
(line 330, avoids the orchestrator HARD reject at orchestrator.py:792-800). The merge
target is the same uniform-K pin (`per_layer_target[layer]=K`), enforced exactly by the
survivor guard (orchestrator.py:1716-1720). So REAM is uniform-K and **iso-K to REAP**
(both consume `budget["K"]` from the single solver run — run_reap_ream_35pct.py:483,
475).

**(c) Net accounting:** `target.total_reduction_ratio=0.35` + `target.net_of_eora=true`
injected (run_reap_ream_35pct.py:312-313). Consumed:
- runner's own solver: `eora_overhead_pct` from `stage4_eora.compensation_budget_pct`
  (=0.03 in base config, line 358), passed to `budget_solver.solve(..., eora_overhead_pct=...)`
  (run_reap_ream_35pct.py:224-237). Solver's net path is gated on
  `eora_overhead_pct > 0` (solver.py:215) and returns `projected_net_reduction`
  (solver.py:149).
- per-arm Stage 1 would (if ever run) read `net_of_eora` at run_pipeline.py:179-183 —
  but Stage 1 is gated out (Concern 1), so this is moot; the seeded
  `budget_decomposition.json` carries sp. `--target-ratio 0.35` is also passed
  (run_reap_ream_35pct.py:445) and sets only the ratio (run_pipeline.py:89-90), with
  `net_of_eora` carried by the written arm config.
- **A1 provenance warning** (run_reap_ream_35pct.py:571-582): if the shared
  `budget_decomposition.json` was produced gross-only (`eora_overhead_pct==0`), the
  runner WARNS (sp ~0.3pp light) but does not block — iso-K still holds, so the
  comparison is unaffected. This is documented and intentional.

**(d) H-B covariance pre-check REQUIRES the fresh bf16 covariance.pt (fail-fast):**
`assert_covariance_resolves` (run_reap_ream_35pct.py:374-419) resolves
`calibration.jsonl_path` (falls back to `_DEFAULT_SELF_TRACES_PATH` =
`artifacts/_shared/self_traces.jsonl`, calibration.py:1939 — base config has no
explicit jsonl_path), then calls `load_covariance` (cached_calibration_signals.py:1452,
returns None on miss). On `payload is None` it RAISES (run_reap_ream_35pct.py:403-413),
naming the expected `sidecars/<stem>/covariance.pt`. Called at
run_reap_ream_35pct.py:586 BEFORE the model load / any GPU work, and the runner does
NOT set `MOE_SKIP_STAGE2_COV_SAVE=1`, so the fp16 fallback is kept but a
stale/missing covariance fails fast rather than silently degrading.
**NOTE (provenance, not a code bug):** the pre-check asserts a covariance.pt *exists
and loads*; it does not cryptographically verify it is the *fresh* bf16 recapture vs an
older bf16 sidecar. Freshness is a launch-time operational gate (per MEMORY: GPU run
gated on fresh recapture). Flagged for the operator; no code change required.

**(e) Stages 3/4/5 really run + Stage 6alt → student_bpt:**
`pipeline.skip_intermediate_stages=False` (run_reap_ream_35pct.py:337) — Stages
3/4/5 are gated on `not _skip_intermediate` (run_pipeline.py:288,300,311), so they
run. `pipeline.evaluator="stage6alt"` (line 338) + `stage6_validate.mode="thermometer"`
(line 339) ⇒ run_pipeline.py:328-329 calls `stage6alt_thermometer.run`, producing
`stage6alt_eval.json` with `student_bpt` (stage6alt/stage.py:24,46;
run_ablations.py:290-293). Completion gate `is_complete` checks that file
(run_reap_ream_35pct.py:450-452, 182). `_student_bpt` extracts it (run_probe.py:516-519).

---

## VERDICT — FIX-IN-PLACE (the single edit is OPTIONAL)

The runner is **structurally correct on all three concerns**. No rewrite. There is
**no required code change**. One optional clarity edit:

### Optional change set (apply only if the user wants the cleanup)
1. `max_quality/src/moe_compress/run_reap_ream_35pct.py:333` — append inline comment
   documenting that `"paper_dials_only"` is now the live default (since 959fab3) and is
   pinned explicitly for robustness against a future default re-flip. (Mirrors
   run_probe.py:206-209.)
   - BEFORE: `cfg.setdefault("stage5_router_kd", {})["rkd_recipe"] = "paper_dials_only"`
   - AFTER:  same line + preceding comment
     `# == the live Stage-2.5/5 default since 959fab3 (2026-06-09); pinned`
     `# EXPLICITLY for robustness to a future default re-flip (no override of`
     `# any other 2.5/5 dial — those stay at the plugin defaults).`
   - Also (optional) update docstring WS2 bullet (lines 42-45) to note the value now
     equals the default.

**Do NOT drop the injection** — the `assert_paper_recipe_safety` net (line 361) and
the documented robustness rationale make explicit pinning the safer choice.

---

## TEST PLAN

Existing suite `max_quality/tests/test_run_reap_ream_35pct.py` (22 passed, 1 skipped —
re-run confirmed green this audit) already covers all three concerns:

**CONCERN 2 (defaults):**
- `test_build_arm_config_reap_injects_paper_pure_and_net_accounting` (:25) — asserts
  `cfg["stage5_router_kd"]["rkd_recipe"] == "paper_dials_only"` (:39).
- `test_build_arm_config_ream_uses_by_the_book_merge` (:46) — same assertion (:57).
- `test_paper_recipe_safety_*` (:76-104) — save_best safety net incl. default path.
- ADD (recommended, 1 test): `test_runner_rkd_recipe_equals_live_default` — import
  `rkd_paper_recipe`, assert the runner's injected literal equals the plugin's default
  fallback string (`s5.get("rkd_recipe", "paper_dials_only")`), so a future default
  flip that diverges from the runner trips CI.

**CONCERN 1 (uniform / no stage-1 leak / guard):**
- `test_build_arm_config_*` assert `assert_survivors_match_target is True` (:31) for
  both arms.
- `test_prune_fraction_is_exact_inverse_of_K` (:118) — uniform-K ↔ prune_fraction.
- ADD (recommended, 1 test): assert `_pipeline_argv(...)` always emits
  `--resume-from-stage` ∈ {2,3} (never 1) for both subprocess calls — locks the
  "Stage 1 never runs per arm" property at the argv level.

**CONCERN 3 (iso-compression / net / covariance / stages run):**
- `test_build_arm_config_reap_injects_paper_pure_and_net_accounting` (:25) — asserts
  `net_of_eora is True` (:37), `skip_intermediate_stages is False` (:41).
- ADD (recommended, 1 test): assert both arms in a single `derive_solver_budget` →
  `run_one_arm` flow consume the SAME `budget["K"]` (iso-K) — a small unit test on the
  budget dict identity is enough.
- `assert_covariance_resolves` miss-path: ADD a unit test that a temp JSONL with no
  sidecar raises RuntimeError (covers H-B fail-fast). The file's module docstring
  (:5-7) notes these I/O paths are presently untested.

All recommended ADDs are net-new and do not change production code.

---

## DEFINITIVE "what Stage-1 code runs" trace (answers the user's worry)

1. Runner start → `assert_covariance_resolves` (no model) → **one** `load_model` +
   `derive_solver_budget` → **one** `budget_solver.solve` on the CPU base model
   (run_reap_ream_35pct.py:600). THIS is the only thing that "looks like Stage 1"
   (it logs "Loading base model to derive the solver budget"). It computes a single
   global K + sp; it does NOT run GRAPE, Super-Expert detection, CKA, or per-layer
   allocation. Model is freed (603-610).
2. Per arm: `seed_stage1_artifacts` writes a **uniform-K** `stage1_budgets.json`
   (every layer = K) + copies blacklist/budget_decomposition from `_shared/`.
3. Subprocess A: `run_pipeline --resume-from-stage 2 --stop-after-stage 2`. Stage-1
   body (run_pipeline.py:170) is skipped (`start=2 > 1`); pipeline loads the seeded
   `budget_decomposition.json` (run_pipeline.py:214). Stage 2 + auto Stage 2.5 run.
4. Subprocess B: `--resume-from-stage 3 --stop-after-stage 6`. Stage 1 skipped again;
   Stages 3→4→5→6alt run.

**Net: GRAPE/RCO non-uniform Stage-1 allocation NEVER executes for either arm.** The
only solver invocation is the runner's intentional one-shot K-derivation, and its
output is forced uniform via the per-layer pin + the per-layer survivor guard. The
user's observation is the expected solver-for-K step, not a leak.
