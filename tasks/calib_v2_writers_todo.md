# Calibration-v2 Writers Campaign — todo.md

**Branch**: `feat/calibration-v2`
**Started**: 2026-05-26
**Scope**: Pending-writers items #0-#10 + REAP-exact bundle (P1-P4, V1-V2) + L1/L2 + L3.
**Workflow per item**: planner agent → my sanity check → implementer agent → my integration → review/fix loop to convergence.

## Phase 1 — Cheap wins, no patch change
- [x] **P1**: Fix stale docstring in `stage2/plugins/reap_scoring.py` — point at `config["calibration"]` instead of hardcoded "Nemotron-Cascade". Commits `0a95560` + `e374c00` (review-fix iter 1). Loop closed at iter 2.
- [x] **P2**: Wire saliency-weighted merge branch in `stage2/merging.py` (replace `raise ValueError`), thread `scores` from `layer_merge.py`. Commits `4239304` + `3aae40c` (review-fix iter 1). Loop closed at iter 2. 5 new tests, byte-identical equivalence to freq mode with matching weights confirmed.

## Phase 2 — Infrastructure + first writer (proves the pattern)
- [x] **Item 0**: `utils/cached_calibration_signals.py` provider-pair module + schema/loader. Commit `b7e2b72`. Loop closed at iter 1 (no fixer needed). 432 LoC module + 544 LoC tests + MANIFEST "Schema bumps" section. 6 dataclass payloads, 12 load/save fns, 2 ABCs, central `SCHEMA_VERSIONS` dict, atomic-write via tmp+os.replace, multi-arch portable, npz double-extension trap defeated via file-handle pattern.
- [x] **V1+V2**: REAP-exact via vLLM hooks — writer in patch (router + expert_out_unweighted scatter-reduce); stage-2 cache-reader plugin (`dispatch_first` slot pattern). Commits `1fd9cc4` + `36f08af` (review-fix iter 1). Loop closed at iter 2. Patch now 4409 lines, MD5 `e3fba22dc2bb0f5db3822c75a8182ad5`. New tag `calib-v2-reap-scores-writer`. 6 vLLM smoke tests + 7 reader tests, all green. Existing stage 2 tests untouched.

## Phase 3 — REAP-exact preset
- [x] **P3+P4** (combined): REAP-exact YAML preset + skip-intermediate orchestration. Commits `48cfc01` + `bdbc2d9`. Loop closed at iter 2. YAML-only entry point (no new CLI flag), `pipeline.skip_intermediate_stages: true` + `pipeline.evaluator: stage6alt`. 6 new tests + docs. 81 regression tests still green.

## Phase 4 — Remaining 8 writers
- [x] **Item 1**: Teacher per-expert `Σ_in[layer][expert]` writer + Stage 3/4 cache readers. Commits `08562c7` + `389a923` (iter-1 fixes for 8 findings). Loop closed at iter 2. Patch now 5350 lines, MD5 `c35dc497cd3e9268c7448410bdddf80c`. New tag `calib-v2-input-cov-writer-chained-callbacks`. **CRITICAL: vllm/calibration_hooks.py registry was restructured to support chained callbacks** (was single-slot, now list-valued with identity-dedup) — required to fix the latent expert_in collision between imatrix + input-cov writers.
- [x] **Item 2**: Per-expert `down_proj` output magnitudes writer + Stage 1 readers. Commits `1a6d2cf` + `a375c98`. Loop closed at iter 2 (4 nitpicks fixed). Patch now 6097 lines, MD5 `15163e64ad096eb8e1e24f961b7f3543`. New tag `calib-v2-per-expert-max-writer`. Reuses `expert_out_unweighted` hook chained with REAP via Item 1's multi-callback fix.
- [x] **Item 3**: Routing freq + mean weight writer + Stage 1/2 readers. Commit `628fe3f`. Loop closed at iter 1 (no fixer needed; 3 cosmetic suggestions noted but non-blocking). Patch now 6841 lines, MD5 `1c5602d20f5a2b268e6edb0f969e4cfe`. New tag `calib-v2-routing-stats-writer`. Router-hook-only (no FLASHINFER dep). Infrastructure for future downstream consumers — current sidecar deposits to `ctx.routing_stats_payload` for any plugin that wants it.
- [x] **Item 4**: Per-layer router-logits aggregate stats writer + Stage 1 SinkToken reader. Commits `f534993` + `65ac809`. Loop closed at iter 2 (1 Important + 1 nit fixed). Patch now 7824 lines, MD5 `2f05332467dafd54e7083494e4aa7823`. New tag `calib-v2-router-logits-stats-writer`. **Storage choice: AGGREGATE stats** (per-(layer, expert) sink/normal score sums + fire-on-sink counts) — raw per-token would have been ~3.3 TB; aggregates are ~256 KB.
- [x] **Item 5**: ~~Per-layer expert top-K + post-softmax weights writer + Stage 2 readers.~~ **SCOPE CUT — implementation-redundant**. Planner analysis confirmed all four named consumers are already served: `ReapScoringPlugin` via V1+V2 cache; `OutputSpaceCostPlugin` recomputes from live router; `ExpertDistillPlugin` v1 drops routing weights entirely; `MergeHealPlugin` captures live post-merge I/O. Also: three different routing-weight conventions in the consumers make a single sidecar impossible. Plan doc updated. Reversible: if a future consumer needs per-token data, revisit. NO code shipped.
- [x] **Item 6**: Per-expert output reservoir writer + Stage 1 CKA reader. Commit `4a26401`. Loop closed at iter 1. Patch now 8774 lines, MD5 `423b8267b18fc8f0990cd7c9b9f24828`. New tag `calib-v2-output-reservoir-writer`. ~17 GB bf16 sidecar (real signal — CKA needs per-token outputs). Reservoir-samples per (layer, expert) up to 256 tokens, matching live `ExpertOutputAccumulator.update()` semantics exactly.
- [x] **Item 7**: Per-MoE-block output (128-prompt subset) writer + Stage 3 `block_refine` cache reader. Commits `91b8844` + `c5ce0df` (iter-1: C1 critical batch_size mismatch + I1/I2/I3) + `26c1181` (iter-2: writer truncates at subset boundary). Loop closed at iter 3. **Stage 2.5 `merge_repair` reader SCOPE-CUT** (router_kd trainer uses its own dataloader). Patch 9828 lines, MD5 `1375b965b02b4ce7ac0a35fd7f7b23cf`. New tag `calib-v2-block-outputs-writer`.
- [x] **Items 8+9 (combined)**: JSONL row metadata schema bump (v7→v8) + `seed_idx`. Commit `a0084d4`. 6 new fields per row (n_prompt_tokens, n_gen_tokens, has_think, refusal_flag, subset, seed_idx); duplicate-key strategy preserves backward-compat. 20 test cases green. NO vLLM patch change.
- [x] **Item 10**: ~~Stage 6 teacher cache pre-populate (WikiText-2, ARC-C, HellaSwag, HumanEval, MATH-500).~~ **SCOPE CUT — already implemented.** The `_teacher_cache_key` + `_load_teacher_cache`/`_save_teacher_cache` + `TeacherProviderPlugin.provide_teacher_side` mechanism in `stage6/plugins/teacher_provider.py` is the pre-population mechanism, operating lazily on first Stage 6 run. First run writes all 5 baselines (wikitext2_ppl, ARC-C, HellaSwag, humaneval_pass_at_1, math500_accuracy) to `teacher_eval_cache.json` under a 10-component SHA-256 key; subsequent ablations are cache-hits with zero GPU cost. An offline `precompute_teacher_baselines.py` would shift identical compute earlier but save nothing AND require duplicating eval-environment setup. NO code shipped. Reversible if cross-model pre-warming becomes a concrete need.

## Phase 5 — Last (your explicit ordering)
- [x] **L2**: `max_layer` early-exit source patch to vLLM model runner. Commit `f8511cb`. Loop closed at iter 1. Patch 10101 lines, MD5 `a8da5e321ac7fb30f1648fba3476bea6`. New tag `calib-v2-max-layer-early-exit`. torch.compile-safe (modifies islice bound; no data-dependent break). Public API: `set_calibration_max_layer(N)` + `VLLM_CALIB_MAX_LAYER` env var. 8 in-patch tests.
- [x] **L1**: ~~REAP+REAM (default) refactor — N vLLM passes with `update_weights` between rounds.~~ **SCOPE CUT (halt-trigger invoked)**. Planner identified 3 architectural blockers: (1) `update_weights` CUDA-graph semantics unverified (worst case: 32 min of graph re-captures × 64 layers), (2) vLLM's `expert_in` hook fires at MoE block input but Stage 2 needs per-expert gate_proj inputs (after dispatch/gather) — semantic mismatch requires solving signal-mapping problem, (3) HF↔vLLM expert weight stacking representations differ; arch-specific adapter needed. All 3 require live GPU validation which the autonomous campaign doesn't have. Per the user's pre-authorized halt-triggers ("architectural incompatibility ... `update_weights` doesn't survive cudagraph capture for L1"), L1 is deferred. **Better alternative path documented**: a `Stage2ProfileCacheProvider` (capture REAP/REAM/cov via the one-shot vLLM calibration pass, read in Stage 2 to bypass the profile forward entirely) is the architecturally-cleaner next step IF Stage 2 profile time becomes a bottleneck. The signal-mapping problem must be solved either way; doing it ONCE in the writer (not N times with weight updates) is the simpler path. Reversible: revisit when GPU-equipped engineering session is available.

## Phase 6 — Deploy
- [ ] Regenerate patch + bump MANIFEST + new tag.
- [ ] Push to GitHub.
- [ ] Kick off HF Jobs build, `/loop` every 15 min.
- [ ] On BUILD COMPLETE: verify wheel + kill any lingering CPU.
- [ ] Surface final to user.

## Invariants (apply to every item)

- **Plugin format**: each stage's existing contract (all stages are now on the universal `PipelinePlugin` Protocol — confirmed `pipeline/` + `tools/` packages present).
- **Atomic writes**: `tmp + os.replace`, matching shipped `.npz`/`.imatrix.dat`/`.imatrix.ckpt`.
- **Docstrings**: every behavior change gets a docstring update; reference `config[...]` over hardcoded values where appropriate.
- **Review/fix loop**: runs to convergence (no iteration cap), all 5 categories incl. nitpick.
- **No monkey-patching**: vLLM changes are source-patches (per `feedback_raise_dont_substitute`).
- **No PR language**: commits direct to `feat/calibration-v2`, FF-only.
- **Halt triggers**: golden snapshot breaks unintentionally / sub-agent structural error / discovered architectural incompatibility / patch fails clean re-apply on fresh tree.
