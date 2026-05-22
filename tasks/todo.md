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
- [ ] S3-2  covariance_collection plugin
- [ ] S3-3  d_rank_allocate plugin
- [ ] S3-4  swift_svd_alpha plugin
- [ ] S3-5  aa_svd_factor plugin
- [ ] S3-6  block_refine plugin
- [ ] S3-7  Wire orchestrator + STAGE3
- [ ] S3-8  Stage-3 orchestrator test

## Stage 4 — EoRA (Part 5.2)
- [ ] S4-0  Capture stage-4 golden
- [ ] S4-1  Scaffold stage4/ package
- [ ] S4-2  eora_inputs plugin
- [ ] S4-3  eora_compensation plugin
- [ ] S4-4  Wire orchestrator + STAGE4
- [ ] S4-5  Stage-4 orchestrator test

## Router-KD — stages 2.5 ≡ 5 (Part 5.3)
- [ ] RK-0  Capture router-KD goldens ×2
- [ ] RK-1  Scaffold router_kd/ package
- [ ] RK-2  trainable_scope plugin
- [ ] RK-3  kd_optimizer plugin
- [ ] RK-4  vocab_kd plugin
- [ ] RK-5  teacher_cache + teacher_live slot plugins
- [ ] RK-6  merge_repair plugin
- [ ] RK-7  early_stop plugin
- [ ] RK-8  Wire orchestrator + resume + factory
- [ ] RK-9  Router-KD dual-invocation test

## Stage 6 — validation (Part 5.4)
- [ ] S6-0  Capture stage-6 golden
- [ ] S6-1  Scaffold stage6/ package
- [ ] S6-2  eval_environment plugin
- [ ] S6-3  wikitext_ppl + zero_shot_lm_eval plugins
- [ ] S6-4  humaneval + math500 plugins
- [ ] S6-5  teacher_provider plugin
- [ ] S6-6  imatrix_export plugin
- [ ] S6-7  validation_report plugin
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
