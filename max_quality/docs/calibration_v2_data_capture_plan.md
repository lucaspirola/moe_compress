# Calibration v2 — Data capture plan

Spec for the next iteration of the Qwen3.6-35B-A3B self-distillation
calibration pass. Goal: capture **every teacher-derived signal that any
plugin in any stage could ever need** in a single calibration run, so
downstream ablations stop re-loading the 70 GB teacher per stage.

Audience: future-Claude / future-Lucas planning the next calibration rental.
Companion to `vllm_multiarch_cache_recipe.md` (cold-start cache recipe).

Status: **spec, not implemented**. Implementation gated on user trigger.
Source audit: `feature-dev:code-explorer` agent run 2026-05-25 against the
three repo checkouts (`/home/lucas/ai/moe_compress` mainline,
`/home/lucas/moe_compress` stage 1 refactor, `/home/lucas/tmp/moe_compress`
stage 2 refactor).

---

## Why this exists

Today every downstream stage that needs a teacher signal **re-loads the
70 GB teacher** and runs its own forward pass:

| Stage | Re-runs teacher for | Approx wall-clock |
|---|---|---|
| 1 Phase B | Per-expert `down_proj` max + router logits + output reservoir | ~25 min |
| 2 profiling | REAP/REAM accumulators + routing weights + layer-input cov | ~20 min |
| 3 covariance | Teacher-side `Σ_in` + dual-forward cross-cov `C` | ~15 min cov + α-search |
| 2.5 merge_repair | Per-layer MoE-block outputs `ℒ_i(x_t)` (live hooks) | full teacher resident ~30 min/layer |
| 3 block_refine | Same — teacher resident during AdamW | full teacher resident |
| 6 / 6alt | Teacher PPL/HumanEval/MATH-500/lm-eval baselines | ~95 min first ablation |

Over an N=12 ablation sweep, this duplicated teacher work compounds to
~22 hours of wall-clock waste plus the operational pain of needing 141 GB
VRAM (H200-class only) for the merge_repair/block_refine paths.

**v2 trades ~1 extra hour of one-time calibration + 15 GB of sidecars for
~22 hours saved across the sweep + 80 GB GPU class flexibility.**

---

## Section 0 — Architectural prerequisite (READ FIRST)

**No pipeline plugin currently reads the calibration JSONL for teacher
signals.** The JSONL is consumed only as a source of prompt text via
`build_calibration_tensor`. Every plugin doing teacher work today collects
signals from a live in-memory teacher.

Implication: capturing more data in the JSONL/sidecars is necessary but not
sufficient. **We also need a consumer path** that lets stages read cached
signals in place of running live teacher forwards.

### The cache-or-live pattern via provider plugins (NOT every-plugin churn)

The master plan §1.5 already defines `PluginRegistry.dispatch_first(hook)`:
when multiple plugins implement the same hook, the registry tries them in
order and the **first non-`None` return wins**. The Router-KD section (§5.3,
RK-5) already names this pattern for teacher logits: a `teacher_cache`
plugin registered first; if cache hits it returns the data, otherwise it
returns `None` and the registry falls through to `teacher_live`.

We generalize this pattern to every cacheable teacher signal. Each
cacheable signal gets a **provider pair**:

| Layer | Changes? |
|---|---|
| Consumer plugins (~40 across stages) | **NO** — they already read context slots via `ctx.get()`. They don't care whether the slot was filled by cache or live. |
| Provider plugins | **YES — one cache + one live per signal** Both write to the same context slot. |
| Stage orchestrator | Register the cache provider **before** the live provider in the plugin sequence. |
| `utils/cached_calibration_signals.py` | **NEW** — schema + on-disk format + loader. Cache providers call this; consumers don't. |

Cache-miss fallback is automatic — a cache provider that finds the file
missing returns `None`, the registry dispatches to the next provider, and
consumer plugins never see the difference.

### Six provider pairs to add

| Signal | Today | Cache provider | Live provider (wrap existing) |
|---|---|---|---|
| Stage 1 Phase B accumulators | `CalibrationEngine.run_phase_b` | `Stage1PhaseBCacheProvider` | `Stage1PhaseBLiveProvider` |
| Stage 2 profiling accumulators | Stage 2 profiling forward | `Stage2ProfileCacheProvider` | `Stage2ProfileLiveProvider` |
| Stage 3 covariance | `stage3_svd._collect_covariances` | `CovarianceCacheProvider` (teacher Σ_in only — see Gap 3) | `CovarianceLiveProvider` (still does dual-forward for `C`) |
| Router-KD teacher logits | `_get_teacher` | `teacher_cache` (already in master plan) | `teacher_live` (already in master plan) |
| Stage 2.5/3 MoE-block hidden states | `_LayerOutputCapture` hooks | `BlockOutputCacheProvider` | `BlockOutputLiveProvider` |
| Stage 6 teacher eval | `_teacher_cache_key` mechanism | `Stage6TeacherEvalCacheProvider` (extend existing cache) | `Stage6TeacherEvalLiveProvider` |

**Total new code**: ~600-800 LoC across 12 provider plugin files + the
loader library. Compare to modifying every consumer plugin (~5000 LoC of
churn). Net: 10× smaller diff and clean cache-miss semantics.

### Partial cache hits

For signals that cannot be fully cached (Stage 3's AA-SVD cross-cov `C`
requires SIMULTANEOUS teacher+student forward — pre-baking from teacher
alone is impossible), the cache provider returns whatever IS cached
(teacher-side `Σ_in`), AND the live provider still runs for the part that
can't be cached (student-side `Σ_in` + cross-cov `C`). They write to
**different sub-slots** of the same context namespace — no race.

---

## Section 1 — What to capture (revised 11-item list)

Audit-revised list. Items #1-#11 of my earlier proposal were largely
over-claims (no consumer reads pre-baked PPL, think indices, prompt
logprobs, etc.). The list below was produced by enumerating every plugin
across all stages and asking *what would it actually consume?*

### P0 (cheap + broad value)

| # | Capture | Consuming plugin(s) | How |
|---|---|---|---|
| 0 | `utils/cached_calibration_signals.py` + 6 provider pairs | All cached signals | New module + ~12 provider plugins (Section 0) |
| 8 | Per-row JSONL metadata `{n_prompt_tokens, n_gen_tokens, has_think, refusal_flag, subset, seed_idx}` | Any plugin using `build_calibration_tensor`; `swift_svd_alpha`, `merge_repair`, Stage 6alt `thermo_corpus` | **IMPLEMENTED** (combined with Item 9) at JSONL `schema_version=8` — fields added at write time in `build_self_traces_calib_vllm.py::_process_outputs`; cache_key folds `schema_version=8` so v7 runs do not cache-hit v8 runs. No vLLM patch change. Loader (`build_calibration_tensor`) is tolerant of additional JSONL keys. |
| 9 | Per-prompt deterministic `seed_idx` (the row's position in the shuffled `CalibrationSpec` source) | Reproducibility audit; any plugin selecting exact subsets | **IMPLEMENTED** as part of the Item-8 metadata bundle at JSONL `schema_version=8` — duplicate key of the existing `_attempt_idx` (kept for back-compat); same int value, plan-doc name. |

### P1 (highest wall-clock + VRAM unlock per row of code)

| # | Capture | Consuming plugin(s) | How |
|---|---|---|---|
| 1 | Teacher per-expert `Σ_in[layer][expert]` | Stage 3 `d_rank_allocate`, `swift_svd_alpha`; Stage 2 `ReamCostPostPlugin` (warm-start prior) | vLLM hook on expert `up_proj`/`gate_proj` input projection; accumulate covariance in fp32 |
| 2 | Teacher per-expert output magnitudes (`per_expert_max`) | Stage 1 `ThreeWayAndPlugin`, `MagnitudeTopkPlugin` | vLLM hook on each expert's `down_proj` output |
| 3 | Teacher routing freq + mean weight per (layer, expert) | Stage 2 `ReapScoringPlugin` (centroid priors); `CapacityGatePlugin` (frequency seeds); Stage 1 `SinkTokenDetectorPlugin` | vLLM hook on each MoE block's router gate output |
| 7 | Teacher per-MoE-block output hidden states on **fixed 128-prompt subset** `(n_tokens × hidden_dim)` per layer | Stage 3 `block_refine` (anchored objective). **Stage 2.5 `merge_repair` consumer SCOPE-CUT** — see note below. | vLLM hook capturing `mlp` module output pre-residual-add; sample on fixed 128-prompt subset only (size budget ~64 GiB on Qwen3-30B-A3B) |
| 10 | Extend Stage 6 `_teacher_cache_key` to pre-populate WikiText-2 PPL, ARC-C, HellaSwag, HumanEval, MATH-500 baselines | `wikitext_ppl`, `humaneval`, `math500`, `zero_shot_lm_eval`, `teacher_provider`, Stage 6alt `bpt_metric`/`zero_shot_subset` | Run benchmark corpora through teacher at calibration time; write to existing `_teacher_cache_key` format |

### P2 (specific plugin wins)

| # | Capture | Consuming plugin(s) | How |
|---|---|---|---|
| 4 | Per-layer pre-softmax teacher router logits per batch | Stage 1 `SinkTokenDetectorPlugin` (`ROUTER_LOGITS_PER_BATCH`) | Same hook as #3 (joint capture) |
| 5 | ~~Per-layer expert top-K + post-softmax routing weights per token~~ | **IMPLEMENTATION-REDUNDANT — NOT IMPLEMENTED (campaign decision)**. All four named consumers are already served without this data: `ReapScoringPlugin` uses its live `ReapAccumulator` (or V1+V2 REAP-scores cache); `OutputSpaceCostPlugin` recomputes `σ(x)` from `_router_routing_weights` against the live router + `layer_inputs` reservoir; `ExpertDistillPlugin` v1 drops per-token routing weights entirely (only `freq` from context is used per `D-expert-distill-mse-v1`); `MergeHealPlugin` captures I/O in-process and recomputes routing from the post-resize live router. The three active routing-weight conventions (renormalized top-K, un-renormalized masked, absent) are mutually incompatible — no single pre-baked representation serves all consumers. Per-token storage (8 GB sampled, 410 GB full) was not justified. See `tasks/calib_v2_writers_todo.md`. | — |
| 6 | Teacher per-expert output activation reservoir `(m_e, d_out)` per `(layer, expert)` | Stage 1 `CKADistancePlugin` (warm-start `output_reservoir`) | vLLM hook on each expert's `down_proj` output, reservoir-sampling |

### Scope-cut consumer notes

**Item 7 — Stage 2.5 `merge_repair` cache reader: NOT IMPLEMENTED.** The
Item 7 writer ships and is consumed by Stage 3 `block_refine` only. The
Stage 2.5 `merge_repair` plugin captures teacher block outputs via live
forward hooks on the teacher's `.mlp` module during the router_kd training
loop (`router_kd/plugins/merge_repair.py::_LayerOutputCapture`), driven by
the router_kd trainer's OWN dataloader — NOT the calibration JSONL prompts.
Aligning the Item-7 sidecar (flat `[n_tokens, hidden]` keyed by vLLM
calibration prompt order) with the router_kd trainer's per-batch
`input_ids` would require all three of: (a) the trainer to switch its
dataset to the calibration JSONL, (b) byte-identical chat-template
rendering between the vLLM driver and the trainer, (c) deterministic batch
shuffle order. None of these hold today, and forcing them would entangle
router_kd's dataset selection with the calibration writer choice. The
Stage 3 reader alone captures the high-cost win (skipping the live teacher
block forward inside Phase C.5's `_phase_c5_block_refine`); the Stage 2.5
side stays on its in-process live capture. Reversible: if a future
router_kd refactor switches the trainer to the calibration JSONL with a
deterministic order, the cache reader becomes tractable and can be added
without a schema bump.

### Removed (no consumer in any plugin)

These were in my pre-audit proposal but the explorer found no plugin reads
them today, nor in the planned-future plugin decompositions:

- `prompt_logprobs=50` on prompt tokens — no consumer reads prompt-token top-K from JSONL
- Return-token rank metadata — no consumer
- Sequence-level perplexity per row — computable from existing per-token logprobs; no consumer reads pre-baked PPL
- `<think>` open/close + answer-end token indices — no consumer
- Per-layer residual-stream hidden states — wrong shape for what CKA/block_refine/merge_repair actually need (they want expert outputs or MoE-block outputs, not residual stream)

### Footnote — speculative future

If attention-aware Stage 3 ever lands (the master plan mentions a
`scope: moe_experts_only` switch that could be flipped), per-head attention
entropy statistics from the teacher pass would enable non-uniform rank
allocation across attention heads. Not implementing now; flagged as
future-optional.

---

## Section 2 — Synergy: vLLM patch + multi-arch compile in ONE rental

Items #1-#7 above require vLLM internal hooks that **do not exist in
stock vLLM 0.21.0**. This means a forked `vllm-patched` build. That same
forked build naturally regenerates all the flashinfer JIT fatbinaries,
which is exactly the moment to bake the **multi-arch cache** from
`vllm_multiarch_cache_recipe.md`.

### Combined recipe (one rental, two artifacts)

```bash
# 1. Rent a GPU host with one of {A100, H100, H200, B200, RTX 6000 Pro}.
#    Any one will do — nvcc compiles for ALL listed archs from a single
#    source via TORCH_CUDA_ARCH_LIST, but vLLM warmup needs a live GPU
#    to dispatch the kernels.

# 2. Clone vllm at the pinned canonical SHA + our hooks patch.
git clone --branch v0.21.0 https://github.com/vllm-project/vllm /tmp/vllm-patched
cd /tmp/vllm-patched
git apply /home/lucas/ai/moe_compress/max_quality/patches/vllm_calibration_hooks.patch

# 3. Set the multi-arch list. nvcc compiles for ALL archs in one shot.
export TORCH_CUDA_ARCH_LIST="8.0;9.0a;10.0;12.0"

# 4. Build vllm from source — pinned-versions venv.
python3 -m venv ~/venv-vllm-build && source ~/venv-vllm-build/bin/activate
pip install --upgrade pip wheel
pip install -e . \
    torch==2.11.0 transformers==5.9.0 datasets==4.8.5

# 5. Run the v2 calibration pass with hooks enabled.
#    The hooks dump router logits, expert outputs, block hidden states
#    to ~/artifacts/_shared/self_traces_v2_<key>_sidecars/
python max_quality/scripts/build_self_traces_calib_vllm.py \
    --teacher Qwen/Qwen3.6-35B-A3B \
    --num-prompts 5000 \
    --max-new-tokens 16384 \
    --reasoning-budget 4096 \
    --logits-top-k 50 \
    --capture-router-logits \
    --capture-expert-outputs \
    --capture-block-hidden-states \
    --block-hidden-states-subset-size 500 \
    --teacher-eval-corpora wikitext,arc,hellaswag,humaneval,math500 \
    --output ~/artifacts/_shared/self_traces_v2.jsonl

# 6. Pack BOTH artifacts.
pip freeze > /tmp/requirements.txt
# 6a. The multi-arch vLLM cache (from the build + warmup).
tar czf vllm_cache_multiarch.tgz -C ~ .cache/flashinfer .cache/vllm \
    -C /tmp requirements.txt
hf upload pirola/qwen3-6-35b-a3b-vllm-cache vllm_cache_multiarch.tgz
# 6b. The v2 calibration artifacts (JSONL + signal sidecars).
tar czf calibration_v2.tgz \
    ~/artifacts/_shared/self_traces_v2.jsonl \
    ~/artifacts/_shared/self_traces_v2_*_sidecars/ \
    ~/artifacts/_shared/teacher_eval_baselines/
hf upload pirola/qwen3-6-35b-a3b-self-traces-v2 calibration_v2.tgz \
    --repo-type dataset
```

### What the patch needs to expose

The forked vLLM must add (and only add) these hook surfaces:

| Hook target | Module | Captures |
|---|---|---|
| MoE block router output | `Qwen3MoeSparseMoeBlock.gate` | items #3, #4, #5 (router stats + logits + top-K) |
| Per-expert output | `Qwen3MoeSparseMoeBlock.experts[e].down_proj` | items #1, #2, #6 (Σ_in via input, max via output, reservoir via output) |
| MoE block aggregate output | `Qwen3MoeSparseMoeBlock` forward return | item #7 (block hidden state pre-residual) |

All hooks should be **opt-in via env var** (`VLLM_CAPTURE_*`), zero overhead
when disabled. Patch surface area: ~150 LoC across 3-4 files.

### Cost amortization

- **vLLM patch + build + multi-arch compile**: ~75 min one-time at the v2 calibration rental.
- **v2 calibration run with hooks**: +25-40% wall-clock vs current (the hooks aren't free); ~4-5 h total.
- **Storage**: ~12-15 GB sidecars + ~750 MB vLLM cache.
- **One rental, two HF artifacts**: `pirola/qwen3-6-35b-a3b-vllm-cache` (multi-arch) + `pirola/qwen3-6-35b-a3b-self-traces-v2` (data).

---

## Section 3 — Migration order

Strict ordering. Each step is independently committable.

| # | Task | Pre-requisite |
|---|---|---|
| M1 | Land master plan §F-1..F-8 (PipelinePlugin, PipelineContext, PluginRegistry, dispatch_first, tools/) | — |
| M2 | Implement `utils/cached_calibration_signals.py` (schema + loader + version pinning) | M1 |
| M3 | Define the on-disk sidecar format (one HDF5/.safetensors per stage; index file at root) | M2 |
| M4 | Implement Stage 6 teacher-eval cache provider pair (lowest risk — existing cache infra) | M3 |
| M5 | Implement Router-KD `teacher_cache` / `teacher_live` per master plan RK-5 | M3 |
| M6 | Patch vLLM with the calibration hooks (env-var gated) — surface area ~150 LoC | M3 |
| M7 | Wire the hooks into `build_self_traces_calib_vllm.py` behind `--capture-*` flags | M6 |
| M8 | Implement Stage 1 / Stage 2 / Stage 3 provider pairs | M3, master plan stage migrations done |
| M9 | One-rental: build patched vLLM with `TORCH_CUDA_ARCH_LIST="8.0;9.0a;10.0;12.0"`, run v2 calibration, upload both artifacts | M6, M7, M8 |
| M10 | Each subsequent ablation pulls both artifacts and runs on 80 GB cards | M9 |

Estimated total engineering: ~5-7 days. The bottleneck is M6 (vLLM patch
review).

---

## Aggregate impact

For an N=12 ablation sweep:

| Bucket | Per-ablation saving | × N=12 |
|---|---|---|
| Stage 1 Phase B (items #2, #4, #6) | ~25 min | 5 h |
| Stage 2 profiling priors (items #3, #5) | ~20 min | 4 h |
| Stage 3 covariance warm-start (item #1) | ~40 min | 8 h |
| Stage 2.5 / 3 teacher-resident → sidecar (item #7) | ~25 min + 70 GB VRAM freed | 5 h + 80 GB GPU access |
| Stage 6 eval pre-bake (item #10) | ~95 min on first ablation, then 0 | ~95 min one-time |
| **Total wall-clock saved** | | **~22 h across the sweep** |
| **Hardware unlock** | | Stage 2.5/3 ablations possible on 80 GB GPUs (A100, H100) instead of 141 GB H200 |

Less-quantifiable wins:

- **Reproducibility**: `seed_idx` + per-row metadata means any future
  researcher can reproduce a specific ablation slice from JSONL + commit.
- **Subset filtering**: enables novel experiments like "router-KD on
  reasoning-only prompts" without re-generating calibration data.
- **Architectural cleanup**: forcing the provider-pair pattern surfaces
  which plugins are coupled to live teacher load vs which can run on
  cached signals — likely uncovers Stage 2.5 / 3 simplifications.
- **Single-artifact deployments**: future calibration re-runs are
  `hf download` + `tar xzf` + `python build_self_traces_calib_vllm.py`.

---

## Related

- `vllm_multiarch_cache_recipe.md` — the multi-arch vLLM cache build recipe
  this plan consumes as a byproduct.
- Master plan: `~/.claude/plans/our-tool-has-two-streamed-minsky.md` — the
  Universal Plugin Interface plan, §1.5 (`dispatch_first`), §5.3 RK-5
  (`teacher_cache` / `teacher_live` pattern we generalize here).
- `build_self_traces_calib_vllm.py` — the script that gains `--capture-*`
  flags and the v2 data write paths.
