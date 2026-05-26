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
- [ ] **Item 1**: Teacher per-expert `Σ_in[layer][expert]` writer + Stage 3/4 cache readers.
- [ ] **Item 2**: Per-expert `down_proj` output magnitudes writer + Stage 1 readers.
- [ ] **Item 3**: Routing freq + mean weight writer + Stage 1/2 readers.
- [ ] **Item 4**: Per-layer pre-softmax router logits writer + Stage 1 SinkToken reader.
- [ ] **Item 5**: Per-layer expert top-K + post-softmax weights writer + Stage 2 readers.
- [ ] **Item 6**: Per-expert output reservoir writer + Stage 1 CKA reader.
- [ ] **Item 7**: Per-MoE-block output (500-prompt subset) writer + Stage 2.5/3 readers.
- [ ] **Item 8**: JSONL row metadata schema bump in `build_self_traces_calib_vllm.py`.
- [ ] **Item 9**: Per-prompt deterministic `seed_idx`.
- [ ] **Item 10**: Stage 6 teacher cache pre-populate (WikiText-2, ARC-C, HellaSwag, HumanEval, MATH-500).

## Phase 5 — Last (your explicit ordering)
- [ ] **L2**: `max_layer` early-exit source patch to vLLM model runner.
- [ ] **L1**: REAP+REAM (default) refactor — N vLLM passes with `update_weights` between rounds.

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
