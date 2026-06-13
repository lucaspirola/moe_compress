# Stage 6 (evaluation) — multi-GPU analysis

Read-only analysis. Scope: every plugin in `src/moe_compress/stage6/plugins/`,
the orchestrator/context, and the shared generation harness
`tools/eval_harness.py`. Citations are `path:line`. No source was modified.

The model fits on one GPU (~50 GB). So this is **never** a model-shard problem.
The only multi-GPU lever Stage 6 admits is **data-parallel eval-shard**: replicate
the full model on each GPU, split the *independent* eval examples across replicas,
generate/score each shard, merge. The question per eval is whether the shard is
(a) GPU-bound enough to be worth it and (b) result-preserving.

---

## TL;DR

Stage 6 today is **strictly single-GPU**. `run()` is handed one `device`
(`run_pipeline.py:347-348` -> `orchestrator.run(..., device=device)`), threads it
through ctx, and every forward/generate call lands on that one device. Nothing
shards across GPUs.

The dominant cost is **generative decoding** (HumanEval + MATH-500), run
**twice** (student then teacher, unless the teacher cache hits). That is the only
part where multi-GPU buys a near-linear win.

**The decisive constraint:** the generative metrics are `gen` /
**METRIC_PINNED** (`auto_batch.py:42`, `FidelityClass.METRIC_PINNED`). They are
NOT batch-invariant — bf16 + left-pad reduction drift flips ~11/16 near-tied
argmax completions at bs=8 vs bs=1 (`eval_harness.py:42-52, 60-72`). The whole
metric is reproducible *only* because the batch geometry is **pinned** to
`PINNED_GEN_BATCH_SIZE = 8` (`eval_harness.py:52`). This is what kills naive
auto-batch on this path — and it also constrains a DP shard.

**The saving grace for DP:** the pinned geometry is per *micro-batch of 8*, and
the per-example completion is a deterministic function of *which other 7 prompts
share its bs=8 group*, in order. A DP shard that **preserves the bs=8 grouping
and the within-group order** produces byte-identical completions to the
single-GPU run. So eval-shard *can* be exact — but only if the shard boundary
falls on a multiple of 8 **and** the prompt ordering inside each group of 8 is
left untouched (see "bs-invariance interaction" below). Per-GPU auto-batch
sizing of the generative path is therefore **forbidden** (it would change the
group geometry and the metric) — each replica must keep the fixed pinned bs=8.

---

## Per-eval table

| Eval | What it measures | Compute profile | Device today | Multi-GPU scheme | Effort | Speedup |
|---|---|---|---|---|---|---|
| **HumanEval** (`humaneval.py`) | code-gen pass@1 | `model.generate()` greedy, 164 prompts, max_new<=2048; CPU exec-scoring | single GPU generate; CPU ProcessPool *only for code-exec* | **DP eval-shard** (shard the 164 prompts on a multiple of 8) | Medium | ~linear (approx 1.9x on 2 GPU) on the generate phase |
| **MATH-500** (`math500.py`) | math accuracy | `model.generate()` greedy, 500 prompts, max_new<=4096; CPU SymPy grading | single GPU generate; CPU grading inline | **DP eval-shard** (shard 500 on multiple of 8) | Medium | ~linear; biggest absolute win (longest decode budget) |
| **WikiText-2 PPL** (`wikitext_ppl.py`) | perplexity | forward-only `model(input_ids, labels)`, `out.loss`; ~hundreds of 2048-tok chunks | single GPU forward | **DP eval-shard** (chunks fully independent; reduction is a sum) | Low–Med | ~linear, but small absolute cost vs generation |
| **Zero-shot lm-eval** (`zero_shot_lm_eval.py`) | ARC-C + HellaSwag acc | loglikelihood fwd via `lm_eval.HFLM` | single GPU (HFLM(pretrained=model)) | **NOT-WORTH-IT** in-tree (delegated to lm-eval; use lm-eval's own data_parallel/accelerate launch instead) | High | n/a here |
| **Teacher side** (`teacher_provider.py`) | same 4 evals on the uncompressed baseline | re-runs all of the above | single GPU; cache-hit short-circuit | inherits whatever the 4 evals get; **cache hit makes it free** | — | cache-first |
| **imatrix / GGUF** (`imatrix_export.py`) | GGUF imatrix export | **external `llama-imatrix` subprocess** + CPU GGUF convert | own `--ngl` GPU offload, outside PyTorch | **NOT-APPLICABLE** (separate binary; not our device mgmt) | — | — |
| **eval_environment** (`eval_environment.py`) | setup/patches/corpus build | CPU + in-place model patches | single model | **NOT-APPLICABLE** (per-replica setup) | — | — |
| **validation_report** (`validation_report.py`) | deltas, thresholds, JSON | pure CPU (one optional CPU teacher load for param-count fallback) | CPU | **NOT-APPLICABLE** (IO/report) | — | — |

---

## The ProcessPool is NOT GPU parallelism (Key Question 4)

`humaneval.py:329-435` (`_score_all_humaneval`) uses a
`ProcessPoolExecutor(mp_context=spawn)` — but it submits
`_humaneval_worker._score_humaneval_one` (`_humaneval_worker.py:75`), a
**torch-free, CPU-only** function that runs the generated Python against the
reference unit test. The worker module is deliberately torch-free
(`_humaneval_worker.py:1-26`) precisely so spawned children do NOT re-import
torch. This pool exists for **code-execution isolation + a kill boundary on
runaway snippets** (`humaneval.py:333-345`), NOT for GPU-parallel generation. All
`model.generate()` work already finished serially on one GPU before scoring
starts (`humaneval.py:311-324` runs `_generate_batched`, *then*
`_score_all_humaneval`). So Stage 6 has **zero** GPU parallelism today; the only
existing parallelism is CPU exec-fanout for HumanEval scoring.

---

## Where the GPU work actually is

All three generation/forward paths funnel device selection the same way: an
explicit `device` arg, falling back to `next(model.parameters()).device`:

- Generation: `eval_harness.py:60-167` (`_generate_batched`) — `_gen_dev` at
  `:100-106`, `encoded.to(_gen_dev)` at `:127`, single `model.generate(**encoded)`
  at `:130`. One device, one stream.
- PPL forward: `wikitext_ppl.py:227-262` — `_ppl_dev` at `:228-233`,
  `batch.to(_ppl_dev)` at `:241`, `model(input_ids=batch, labels=batch)` at `:244`.
- lm-eval: `zero_shot_lm_eval.py:177` — `HFLM(pretrained=model, ...)`; device is
  wherever the model lives.

Each is GPU-bound and replicable. None currently touches a second device.

---

## bs-invariance interaction (Key Questions 3, 5) — the crux

`_generate_batched` (`eval_harness.py:60-167`) processes prompts in **contiguous
slices of `batch_size`** (`:113` `for i in range(0, len(prompts), batch_size)`),
left-padding each slice to its own longest member (`:121-125`,
`pad_to_multiple_of=64`). Greedy decode is deterministic *for a fixed batch
geometry* but NOT across geometries (`eval_harness.py:64-72`). The metric is only
reproducible because callers **pin `gen_batch_size = PINNED_GEN_BATCH_SIZE = 8`**
(`humaneval.py:586-591`, `math500.py:456-461`, `teacher_provider.py:571-576` all
WARN if off-pin; `eval_harness.py:42-52`).

Consequence for a DP eval-shard:

1. **A completion is a pure function of its bs=8 group and within-group order.**
   Prompt *i* lands in group `i // 8` and its left-pad width is set by the
   longest prompt in `[8*(i//8) : 8*(i//8)+8]`. To stay byte-identical to the
   single-GPU run, a shard MUST keep every group of 8 intact and in the same
   internal order.

2. **Therefore the shard boundary must be a multiple of 8.** Split on group
   boundaries: GPU0 gets groups `[0..k)`, GPU1 gets groups `[k..n_groups)`, i.e.
   the prompt-index cut is `k*8`. Pick `k = round(n_groups/2)`. Each replica runs
   `_generate_batched` over its contiguous sub-list with the SAME pinned bs=8;
   the groups it forms are identical to the single-GPU groups it owns. Merge by
   concatenation in original index order -> byte-identical completions ->
   identical pass@1 / accuracy. **This preserves the bs-invariance correctness
   fix exactly.**

3. **Per-GPU auto-batch sizing is forbidden on the generative path.** Because the
   metric depends on the group geometry, a replica may NOT size its own
   generative batch from VRAM — that would change bs away from 8 and flip
   near-tied argmax. This is exactly why `resolve_batch` gates on
   `_V1_ELIGIBLE = {BATCH_INVARIANT}` (`auto_batch.py:48-52, 178-185`) and the
   generative class is METRIC_PINNED, not eligible. Under DP, each replica keeps
   the **fixed pinned bs=8**; the only "VRAM-awareness" available is
   `run_with_oom_backoff` (`auto_batch.py:188-204`) — but **even that is unsafe
   here** because halving bs changes the metric. So the generative DP shard runs
   a hard-pinned bs=8 on every replica with NO sizing and NO backoff. (Memory is
   not a concern: bs=8 on a 50 GB model already fits the single-GPU run.)

So the auto-batch "HARD REQUIREMENT" (preserve per-GPU VRAM-aware sizing)
interacts as follows: **on the generative evals there is no VRAM-aware sizing to
preserve in the first place** — the geometry is pinned by the metric, not chosen
by VRAM. DP simply replicates the same pinned bs=8 per GPU. The requirement is
honored vacuously: each replica independently *could* call `resolve_batch`, which
correctly returns the fixed bs=8 untouched because the class is not v1-eligible.

### PPL is the one path where per-GPU auto-batch genuinely applies

WikiText PPL reads `out.loss`, a batch-invariant mean reduction
(`wikitext_ppl.py:26-39, 242, 260-262`): the code explicitly rescales the mean by
`(batch.numel() - batch.shape[0])` to recover the exact NLL **sum**
(`:260`), and the final PPL is `exp(nll_sum / tok_count)` (`:293`). This is
order-independent and batch-size-independent -> it is the natural
`BATCH_INVARIANT` candidate. A DP shard of the chunk rows is **exact regardless
of where the boundary falls** (no group constraint), and **each GPU replica may
size its own forward batch from its own VRAM** via the standard
`resolve_batch`/`size_batch` path (`auto_batch.py:143-185`) and still merge the
two partial `(nll_sum, tok_count)` sums to the identical global PPL. This is the
clean case where the multi-GPU + per-GPU-auto-batch requirement is fully and
correctly satisfiable. (Caveat: PPL is the cheapest of the GPU paths, so the
absolute win is small.)

---

## Key-question answers

**(1) Which evals are generation-bound vs forward-PPL vs CPU?**
- GPU-generation-bound: **HumanEval, MATH-500** (`model.generate`,
  `eval_harness.py:130`). These dominate wall-clock (max_new 2048 / 4096, x2 for
  teacher).
- Forward-PPL: **WikiText-2** (`wikitext_ppl.py:244`, `out.loss`).
- Loglikelihood-forward: **zero-shot lm-eval** (`zero_shot_lm_eval.py:177-182`),
  but inside the external lm-eval harness.
- CPU-only: HumanEval **scoring** (`_humaneval_worker.py`), MATH SymPy grading
  (`math500.py:294-335`), validation_report, imatrix subprocess.
- (Thermometer/stage6alt is a separate evaluator path, `run_pipeline.py:342-344`,
  out of scope here.)

**(2) Can eval examples be sharded across 2 GPUs (full replica each), near-linear?**
Yes for HumanEval / MATH-500 / WikiText — the examples are independent and the
plugins already collect results into per-index lists then reduce (pass-count sum
`humaneval.py:385-408`; accuracy count `math500.py:241-251`; NLL sum
`wikitext_ppl.py:260-293`). Replicate the model per GPU, shard the prompt/chunk
list, merge. Generation is the expensive part, so the win is ~linear on the
generative evals (approx 1.9x on 2 GPU after replica-load overhead). lm-eval
should be parallelized through its OWN mechanism, not re-sharded in-tree.

**(3) Does sharding preserve the bs-invariance correctness fix (per-example determinism)?**
- HumanEval / MATH-500: **only if** the shard boundary is on a multiple of 8 and
  each group-of-8's internal order is preserved (see crux above). Done that way,
  it is byte-identical. Done naively (arbitrary split, re-padding, reordering) it
  **breaks** the metric.
- WikiText PPL: **always** preserves it (true batch-invariant reduction; no group
  constraint).
- Greedy `do_sample=False` (`eval_harness.py:143`) means no RNG/seed plumbing is
  needed across replicas.

**(4) Does the existing ProcessPool give any GPU parallelism?**
No. It is a torch-free **CPU** spawn pool for safely running model-generated code
with a kill boundary (`humaneval.py:329-435`, `_humaneval_worker.py:1-26,
75-97`). Generation already completed on one GPU before it runs. Purely CPU
exec-isolation, not GPU-parallel generation.

**(5) How does per-GPU auto-batch coexist with the gen METRIC_PINNED constraint?**
On the generative evals it does **not** coexist — and must not. The metric pins
the geometry to bs=8; per-GPU VRAM sizing (and even OOM-backoff halving) would
change the group geometry and flip the metric. So under a generative DP shard,
every replica runs a hard-pinned bs=8 (which is exactly what `resolve_batch`
returns for a non-`BATCH_INVARIANT` class — a no-op floor, `auto_batch.py:178-185`).
Per-GPU VRAM-aware auto-batch is real and correct **only on the PPL path**
(`BATCH_INVARIANT`), where each replica sizes its own forward batch and the
partial NLL sums merge exactly.

---

## Recommendation / effort-risk

Highest-value, lowest-risk target: **DP eval-shard of HumanEval + MATH-500
generation** (and the teacher's, when the cache misses), with:
- shard boundary forced to a multiple of `PINNED_GEN_BATCH_SIZE` (group-aligned),
- pinned bs=8 per replica (no sizing, no backoff on the gen path),
- merge by original-index concatenation, then run the existing CPU scoring
  unchanged.

This is exact (byte-identical completions), independent of the teacher cache, and
captures essentially all the wall-clock (decode budget dominates). Risk is
entirely in the **shard-alignment discipline** — an off-by-group split silently
changes the reported metric, so any implementation needs a golden byte-identical
gate (single-GPU completions == concat of 2-GPU shard completions) before it can
be trusted.

Secondary: **WikiText PPL DP-shard with per-GPU auto-batch** — fully correct and
the clean home for the VRAM-aware sizing requirement, but small absolute payoff.

Out of scope / not worth in-tree: zero-shot lm-eval (use lm-eval's own DP launch),
imatrix subprocess (own `--ngl`), report/setup (CPU/per-replica).

Notable hazard for any DP implementation: `_generate_batched` **mutates shared
tokenizer state** (`padding_side`, `pad_token_id`) and restores it in `finally`
(`eval_harness.py:73-96, 163-165`) — flagged H2 as not concurrency-safe. Per-GPU
replicas must each use their own tokenizer (or run in separate processes) to
avoid racing that global state. The cleanest DP design is process-per-GPU (mirror
of the spawn discipline already used for scoring), which sidesteps the tokenizer
race and CUDA-context sharing entirely.
