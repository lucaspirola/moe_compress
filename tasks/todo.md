# TODO — Universal Plugin Interface + Stage Adaptation

Branch: `feat/universal-plugin-interface` (off `main` @ `3b9db2c`)
Master plan: `~/.claude/plans/our-tool-has-two-streamed-minsky.md`

Execution per task: planner agent → implementer agent → (reviewer → fixer)* until clean → run tests → commit.

## Framework (Part 2)
- [x] F-1  PipelinePlugin + BasePlugin (`pipeline/plugin.py`) — committed a7c0e08
- [x] F-2  PipelineContext (`pipeline/context.py`) — committed 4665b06
- [x] F-3  PluginRegistry (`pipeline/registry.py`) — committed 6eff639
- [x] F-4  Stage protocol (`pipeline/stage.py`) — committed fbf5eb6
- [x] F-5  tools/ + phase_walker + artifact_builder — committed 5deb26f
- [~] F-6  tools/calibration_pass + whitening + eigh_decomp — DEFERRED, folded into S1-3 (calibration_pass) / S3 (whitening, eigh_decomp)
- [~] F-7  tools/kd_loop + model_factor — DEFERRED, folded into RK / S3
- [~] F-8  tools/eval_harness + eval_environment + teacher_cache — DEFERRED, folded into S6 / S6alt

Rationale: plan Part 2 sanctions folding F-6..F-8 into the first consuming
stage. These are concrete numerical/training/eval modules — built with their
real consumer so the API fits and byte-identical correctness is gated by the
stage golden snapshot, not a speculative standalone test.

## Stage 1 adaptation (Part 3)
- [x] S1-1  Port stage-1 package — committed 743073f
- [x] S1-2  Migrate 8 plugins to PipelinePlugin — committed 20f8e0b
- [x] S1-3a Create tools/calibration_pass.py (folded F-6) — committed f3b2219
- [x] S1-3b Relocate candidates + safe_json into pipeline/ — committed d510d7c
- [x] S1-3c Rewire orchestrator + collapse Stage1Context + dismantle _framework/ — committed 19cf449
- [x] S1-4a Expose STAGE1 Stage object (stage1/stage.py) — committed 267cc4c
- [x] S1-4b Rewire run_pipeline.py to plugin stage1 + delete monoliths — committed 4e3979b
      (+ ccfe5a6 chore: fixed pre-existing stale stage-5 format_version test)
- [x] S1-5  Stage-1 regression sweep — 339 passed (stage1 + pipeline + tools)

## Stage 2 adaptation (Part 4)
- [x] S2-1a Port stage-2 framework (_framework/) + helpers — committed 9c76be9
- [x] S2-1b Port stage-2 plugins (stage2/plugins/, 19 files) — committed dcae454
- [x] S2-1c Port slim orchestrator + run_layer behavioral gate — committed db340cc
- [x] S2-2  Migrate context typed→dict — committed 379bc0d
- [x] S2-3a Make Stage2Pipeline phase walk tolerant (getattr+callable) — committed 0abc333
- [x] S2-3b Migrate 16 plugins to PipelinePlugin + tests + delete _framework/base.py & registry.py — committed dca1ca3
- [x] S2-4  Migrate registry + pipeline (walk_phases, delete _framework/) — committed ffef12e
- [x] S2-5  Decompose compute_assignment — committed 675cec5
- [x] S2-6  Wire cost plugins live — committed 56a5d46
- [x] S2-7  Wire SkipMergeFloorPlugin live — committed f24b257
- [x] S2-8  Wire solver plugins live — committed 4cc8487
- [x] S2-9  Wire refinement plugins live — committed b269495
- [x] S2-10 Wire CapacityGatePlugin live — committed df0a406
- [x] S2-11 Wire post-merge plugins live — ExpertDistill (merge phase) + MergeHeal (post_merge phase)
- [x] S2-12a Introduce LayerMergePlugin (6 live hooks), wire live, neuter LegacyAdapter
- [x] S2-12b+c Delete legacy_adapter.py + remove orchestrator refs + retarget 12 test files
- [x] S2-13a Expose STAGE2 Stage object (stage2/stage.py) + __init__ exports + tests
- [x] S2-13b Rewire run_pipeline.py to stage2 pkg + delete stage2_reap_ream.py monolith (19 test files retargeted)
- [x] S2-13c Rewrite stage2_plugin_guide.md for the final S2-12/S2-13 architecture

## Stage 3 — SVD (Part 5.1)
- [x] S3-0  Capture stage-3 golden — test_stage3_golden_snapshot.py (rank_map.json, fp32+bf16)
- [x] S3-1  Scaffold stage3/ package — __init__/context/orchestrator/plugins, run delegates to legacy
- [x] S3-2  covariance_collection plugin — _collect_covariances + _load_stage2_covariance relocated
- [x] S3-3  d_rank_allocate plugin — _GroupStats/_group_stat/_pad/_compute_T_budget/_d_rank_allocate relocated
- [x] S3-4  swift_svd_alpha plugin — both α-searches + redistribute + snapshot/restore + wikitext PPL relocated
- [x] S3-5  aa_svd_factor plugin — AA-SVD core (_EighDecomp/_precompute_eigh/_aa_svd*/_cov_lookup) relocated
- [x] S3-6  block_refine plugin — _phase_c5_block_refine/_advance_streams relocated, first config-gated stage-3 plugin
- [x] S3-7a Rewrite stage-3 orchestrator on the plugin schedule + flip stage3_svd.run to a shim
- [x] S3-7b Expose STAGE3 Stage object + stage3/__init__ export
- [x] S3-8  Stage-3 orchestrator test — test_stage3_orchestrator.py (registry/phase-order/artifact-set)

## Stage 4 — EoRA (Part 5.2)
- [x] S4-0  Capture stage-4 golden — test_stage4_golden_snapshot.py (eora_ranks.json, fp32+bf16)
- [x] S4-1  Scaffold stage4/ package — __init__/context/orchestrator/plugins, run delegates to legacy
- [x] S4-2  eora_inputs plugin — EoraInputsPlugin + inert load_eora_inputs hook (monolith untouched)
- [x] S4-3  eora_compensation plugin — _compute_eora_factors/_spill_layer relocated + tools/dtype_noise_floor
- [x] S4-4a Rewrite stage-4 orchestrator on the plugin schedule + flip stage4_eora.run to a shim
- [x] S4-4b Expose STAGE4 Stage object + stage4/__init__ export
- [x] S4-5  Stage-4 orchestrator test — test_stage4_orchestrator.py (registry/phase-order/sidecar-deletion)

## Router-KD — stages 2.5 ≡ 5 (Part 5.3)
- [x] RK-0  Capture router-KD goldens ×2 — test_router_kd_golden_snapshot.py (stage2p5+stage5; metadata+loss-trace)
- [x] RK-1  Scaffold router_kd/ package — __init__/context/orchestrator/plugins + make_router_kd_stage factory
- [x] RK-2  trainable_scope plugin — _freeze_non_routers relocated + conflict-check reproduced in inert hook
- [x] RK-3  kd_optimizer plugin — split-group AdamW + _lr_lambda reproduced; _move_optimizer_state_to_device relocated
- [x] RK-4  vocab_kd plugin — _chunked_vocab_kl/_combine_kd_loss + NaN probes relocated
- [x] RK-5  teacher_cache + teacher_live slot plugins — provide_teacher_logits slot, cache wins under dispatch_first
- [x] RK-6  merge_repair plugin — 7 symbols relocated; stage-gated is_enabled (stage2p5-only)
- [x] RK-7  early_stop plugin — _save_best_router_state relocated; best-tracker EMA + patience reproduced
- [x] RK-8  Wire orchestrator + resume + factory — router_kd.orchestrator drives the plugin schedule; stage5_router_kd.run shimmed
- [x] RK-9  Router-KD dual-invocation test — test_router_kd_orchestrator.py (8 tests; registry roster/order, dual-factory Stage conformance, stage-gated merge_repair, stage_id→dir-name propagation ×2)

## Stage 6 — validation (Part 5.4)
- [x] S6-0  Capture stage-6 golden — test_stage6_golden_snapshot.py (stage6_eval.json byte-identical; all evals disabled + teacher cache-hit forced → integer/bool-only artifact)
- [x] S6-1  Scaffold stage6/ package — __init__/context/orchestrator + plugins/; orchestrator delegates to legacy stage6_validate.run + test_stage6_scaffold.py
- [x] S6-2  eval_environment plugin — EvalEnvironmentPlugin; Pattern-A relocates 8 env-setup symbols, inert setup_environment hook (experts-impl, model.eval, revision-pin, imatrix corpus, kernel patches, torch.compile, masking_utils patch)
- [x] S6-3  wikitext_ppl + zero_shot_lm_eval plugins — WikitextPplPlugin + ZeroShotLmEvalPlugin; Pattern-A relocates _wikitext2_ppl / _lm_eval_tasks / _ZERO_SHOT_TASKS, inert eval_task hooks
- [x] S6-4  humaneval + math500 plugins — HumanEvalPlugin + Math500Plugin + new tools/eval_harness.py (shared batched-gen + chat-format); 14 symbols + sympy guard relocated, inert eval_task hooks
- [x] S6-5  teacher_provider plugin — TeacherProviderPlugin; Pattern-A relocates TEACHER_CACHE_FORMAT_VERSION + 5 functions (_safe_pkg_version/_teacher_cache_key/_load/_save/_preload_teacher_to_cpu), inert provide_teacher_side hook
- [x] S6-6  imatrix_export plugin — ImatrixExportPlugin; Pattern-A relocates 5 functions + _EVAL_TEXT_CONCAT_FILENAME; Option-C two-hook design (start_gguf_convert + export_imatrix); is_enabled defaults True matching monolith
- [x] S6-7  validation_report plugin — ValidationReportPlugin; Pattern-A relocates _deltas/_measured_reduction/_check_thresholds (§8 NaN hotspot preserved verbatim); inert assemble_report hook reproduces JSON-assembly + Trackio flatten
- [ ] S6-8  Wire orchestrator + STAGE6
- [ ] S6-9  Stage-6 orchestrator test

## Stage 6alt — thermometer (Part 5.5)
- [ ] S6A-0 Capture stage-6alt golden
- [ ] S6A-1 Scaffold stage6alt/ package
- [ ] S6A-2 thermo_environment + thermo_corpus plugins
- [ ] S6A-3 bpt_metric + zero_shot_subset plugins
- [ ] S6A-4 thermo_teacher_provider plugin
- [ ] S6A-5 thermo_report plugin
- [ ] S6A-6 Wire orchestrator + STAGE6ALT
- [ ] S6A-7 Stage-6alt orchestrator test

## Cleanup
- [ ] Z-1  Delete StagePlugin/Stage2Plugin back-compat shims; full-suite green

## Review
(completion notes added here as tasks land)
