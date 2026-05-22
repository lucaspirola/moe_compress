# TODO — Universal Plugin Interface + Stage Adaptation

Branch: `feat/universal-plugin-interface` (off `main` @ `3b9db2c`)
Master plan: `~/.claude/plans/our-tool-has-two-streamed-minsky.md`

Execution per task: planner agent → implementer agent → (reviewer → fixer)* until clean → run tests → commit.

## Framework (Part 2)
- [ ] F-1  PipelinePlugin + BasePlugin (`pipeline/plugin.py`)
- [ ] F-2  PipelineContext (`pipeline/context.py`)
- [ ] F-3  PluginRegistry (`pipeline/registry.py`)
- [ ] F-4  Stage protocol (`pipeline/stage.py`)
- [ ] F-5  tools/ + phase_walker + artifact_builder
- [ ] F-6  tools/calibration_pass + whitening + eigh_decomp
- [ ] F-7  tools/kd_loop + model_factor
- [ ] F-8  tools/eval_harness + eval_environment + teacher_cache

## Stage 1 adaptation (Part 3)
- [ ] S1-1  Port stage-1 package
- [ ] S1-2  Migrate 8 plugins to PipelinePlugin
- [ ] S1-3  Re-wire orchestrator on shared primitives
- [ ] S1-4  Expose STAGE1 Stage object
- [ ] S1-5  Stage-1 regression sweep

## Stage 2 adaptation (Part 4)
- [ ] S2-1  Port stage-2 package
- [ ] S2-2  Migrate context typed→dict
- [ ] S2-3  Migrate Stage2Plugin→PipelinePlugin
- [ ] S2-4  Migrate registry + pipeline
- [ ] S2-5  Decompose compute_assignment
- [ ] S2-6  Wire cost plugins live
- [ ] S2-7  Wire SkipMergeFloorPlugin live
- [ ] S2-8  Wire solver plugins live
- [ ] S2-9  Wire refinement plugins live
- [ ] S2-10 Wire CapacityGatePlugin live
- [ ] S2-11 Wire post-merge plugins live
- [ ] S2-12 Delete LegacyAdapter
- [ ] S2-13 Expose STAGE2 Stage object

## Stage 3 — SVD (Part 5.1)
- [ ] S3-0  Capture stage-3 golden
- [ ] S3-1  Scaffold stage3/ package
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
