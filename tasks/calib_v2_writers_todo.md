# Calibration-v2 Writers Campaign — todo.md

**Branch**: `feat/calibration-v2`
**Started**: 2026-05-26
**Scope**: Pending-writers items #0-#10 + REAP-exact bundle (P1-P4, V1-V2) + L1/L2 + L3.
**Workflow per item**: planner agent → my sanity check → implementer agent → my integration → review/fix loop to convergence.

## Phase 1 — Cheap wins, no patch change
- [ ] **P1**: Fix stale docstring in `stage2/plugins/reap_scoring.py` — point at `config["calibration"]` instead of hardcoded "Nemotron-Cascade".
- [ ] **P2**: Wire saliency-weighted merge branch in `stage2/merging.py` (replace `raise ValueError`), thread `scores` from `layer_merge.py`.

## Phase 2 — Infrastructure + first writer (proves the pattern)
- [ ] **Item 0**: `utils/cached_calibration_signals.py` provider-pair module + schema/loader.
- [ ] **V1+V2**: REAP-exact via vLLM hooks — writer in patch (router + expert_out_unweighted scatter-reduce); stage-2 cache-reader plugin (`dispatch_first` slot pattern).

## Phase 3 — REAP-exact preset
- [ ] **P3**: `configs/qwen36_35b_a3b_reap_exact.yaml` — keep qwen3-pretrain-mix, skip 2.5/3/4/5.
- [ ] **P4**: `run_pipeline.py` `--reap-exact` flag + stage-6/6alt input remap from `stage2_pruned/`.

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
