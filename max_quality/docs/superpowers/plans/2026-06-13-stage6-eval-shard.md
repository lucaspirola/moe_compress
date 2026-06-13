# Implementation Plan — Data-Parallel Eval-Shard for Stage 6 (HumanEval, MATH-500, WikiText-PPL)

Status: PLAN ONLY — do not implement until reviewed. Branch `feat/stage6-eval-shard`,
worktree `/home/lucas/ai/wt-s6mg`, code root `max_quality/`.

This feature is for FUTURE runs. It is opt-in and **default single-GPU byte-identical**;
the live ablation path is untouched. Correctness > speed throughout.

Spec source (read fully): `max_quality/docs/multigpu_analysis/stage6.md`.
Stage-3 DP spawn precedent mirrored here:
`src/moe_compress/stage3/plugins/covariance_collection.py:1052-1288`
(`_shard_calib` / `_cov_replica_worker` / `run_dp_covariance_collection`).

---

## Goal

Replicate the (single-GPU-fitting, ~50 GB) Stage 6 model on each of `G` GPUs, split the
**independent** eval examples across replicas, generate/score each shard, and merge —
~linear speedup on the generation-bound evals (HumanEval, MATH-500) plus WikiText-PPL,
while being **RESULT-PRESERVING (byte-identical completions / exact PPL)**.

The model fits on one GPU, so this is never a model-shard problem; the only multi-GPU
lever Stage 6 admits is **data-parallel eval-shard** (`stage6.md:7-11`). The dominant
cost is generative decoding (HumanEval `max_new<=2048`, MATH-500 `max_new<=4096`), run
twice (student then teacher unless the teacher cache hits) — that is the near-linear win
(`stage6.md:22-31, 50-51`).

### The correctness crux (cite + verify against code)

Generative metrics are **METRIC_PINNED**, NOT batch-invariant: bf16 matmul reduction
order + left-pad placement flip ~11/16 near-tied argmax completions at bs=8 vs bs=1
(`auto_batch.py:42` `FidelityClass.METRIC_PINNED`; `eval_harness.py:42-52, 60-72`;
`humaneval.py:40-45`; `math500.py` docstring). The metric is reproducible **only**
because callers hard-pin `gen_batch_size = PINNED_GEN_BATCH_SIZE = 8`
(`eval_harness.py:52`; warned-if-off at `humaneval.py:586-591`, `math500.py:456-461`).

Verified in code: `_generate_batched` processes prompts in **contiguous slices of
`batch_size`** (`eval_harness.py:113` `for i in range(0, len(prompts), batch_size)`),
left-pads each slice to its own longest member with `pad_to_multiple_of=64`
(`eval_harness.py:121-125`), and decodes greedily `do_sample=False`
(`eval_harness.py:143`). Therefore:

1. **A completion is a pure function of its group-of-8 and within-group order.** Prompt
   `i` lands in group `i // 8`; its left-pad width is set by the longest prompt in
   `[8*(i//8) : 8*(i//8)+8]` (`eval_harness.py:113-125, 150`). Verified: `input_len` is
   the padded width, shared across the batch (`eval_harness.py:150`).
2. **Therefore the shard boundary MUST be a multiple of 8** and the within-group order
   MUST be untouched. Split on group boundaries: GPU0 gets prompts `[0 : k*8)`, GPU1 gets
   `[k*8 : n)` — each replica forms IDENTICAL groups to the single-GPU run it owns.
   Merge by concatenation in original index order → byte-identical completions
   (`stage6.md:106-121, 183-192`).
3. **Per-GPU auto-batch sizing AND OOM-backoff are FORBIDDEN on the gen path** — both
   change the group geometry and flip the metric (`stage6.md:123-140`). Each replica runs
   a hard-pinned bs=8 with NO `size_batch`, NO `run_with_oom_backoff`. This is exactly
   what `resolve_batch` already returns for a non-`BATCH_INVARIANT` class — the floor,
   untouched (`auto_batch.py:52, 178-185`). Memory is a non-issue: bs=8 already fit the
   single-GPU run (`stage6.md:133`).

### WikiText-PPL is the clean BATCH_INVARIANT path

`out.loss` is a batch-invariant mean; the code rescales by `(batch.numel() -
batch.shape[0])` to recover the exact NLL **sum** (`wikitext_ppl.py:260-262`), and the
final PPL is `exp(nll_sum / tok_count)` (`wikitext_ppl.py:293`). Order- and batch-size-
independent (`wikitext_ppl.py:26-39`; `stage6.md:142-156`). A DP shard of the chunk rows
is **exact regardless of boundary** (no group-of-8 constraint), and **each replica MAY
size its own forward batch from its own VRAM** via `resolve_batch`/`size_batch`
(`auto_batch.py:143-185`) — the two partial `(nll_sum, tok_count)` sums merge to the
identical global PPL. This is the ONLY path where the per-GPU auto-batch requirement is
satisfiable; the gen path pins bs=8.

### Why process-per-GPU (not threads)

`_generate_batched` **mutates shared tokenizer state** (`padding_side`, `pad_token_id`)
and restores it in a `finally` (`eval_harness.py:73-96, 163-165`; flagged H2). Two
replicas on one tokenizer would race that global state. Process-per-GPU gives each
replica its own tokenizer + CUDA context and sidesteps the race entirely
(`stage6.md:233-239`). This mirrors the Stage-3 spawn discipline exactly.

The existing `ProcessPoolExecutor` in HumanEval (`humaneval.py:329-435`) is **CPU
code-exec isolation, NOT GPU parallelism** — the torch-free worker
(`_humaneval_worker.py`) runs the generated Python against the reference test AFTER all
generation finished serially on one GPU (`stage6.md:61-74, 193-197`). It is untouched by
this plan; scoring still runs in the PARENT after the merge.

---

## Architecture

### Integration point

Stage 6 runs as one in-process orchestrator (`stage6/orchestrator.py:117`), walking the
eval plugins via `walk_phases(("eval_task",), ...)` (`orchestrator.py:184`). Each plugin
calls `_humaneval` / `_math500` / `_wikitext2_ppl`, which call
`_generate_batched` / `model(...)` on one device. There is no `torch.distributed` in the
repo for inference; the Stage-3 precedent uses raw `torch.multiprocessing` spawn +
`CUDA_VISIBLE_DEVICES` pins + a filesystem reduce (`covariance_collection.py:1271-1288`),
which we mirror. **No NCCL** is needed (no gradients; the reduce is a trivial
concat / sum done in the parent).

### Chosen approach: a single new harness module + 3 thin plugin call-sites

New module `src/moe_compress/tools/eval_shard.py` (leaf utility; stdlib + torch + a
function-local import of the eval helpers — see the import-contract note below). It owns:

- `_group_aligned_split(n, replicas, group=PINNED_GEN_BATCH_SIZE) -> list[(start,end)]`
  — gen-path boundaries, every boundary a multiple of 8.
- `_even_split(n, replicas) -> list[(start,end)]` — PPL-path boundaries (no constraint).
- `_gen_replica_worker(...)` — spawn target for HumanEval/MATH-500: reload student, pin
  GPU, run `_generate_batched` on its prompt sub-list at **hard-pinned bs=8**, write
  completions to its replica file.
- `_ppl_replica_worker(...)` — spawn target for WikiText-PPL: reload student, pin GPU,
  forward its chunk sub-rows at a **per-replica auto-batch** (BATCH_INVARIANT), write its
  partial `(nll_sum, tok_count)` to its replica file.
- `run_dp_generate(prompts, ...) -> list[str]` — fan out gen replicas, join, merge
  completions by original index (concatenation of contiguous, group-aligned shards).
- `run_dp_ppl(chunks, ...) -> float` — fan out PPL replicas, join, sum partials, return
  `exp(sum_nll / sum_tok)`.

The three plugins gain ONE branch each: when DP is enabled and `G>1`, route the
generate / forward step through the harness; else fall through to the existing in-process
call **unchanged** (the default path is byte-for-byte the current instruction stream).

### Default-OFF discipline (the byte-identical guarantee)

A new config block `stage6_validate.eval_shard` (parsed via a small frozen dataclass,
`AutoBatchConfig`-style coerce, `auto_batch.py:55-72`):

```yaml
stage6_validate:
  eval_shard:
    enabled: false        # master switch; default false → resolver never spawns
    replicas: 0           # 0/1 → single-GPU in-process path (no spawn)
    gpus_per_replica: 1
```

When `enabled` is false OR `replicas <= 1` OR `torch.cuda.device_count() < 2`, the
plugins take the existing in-process path — **no spawn, no new tensor ops, no reordering**
— so every Stage 6 golden stays byte-identical (`test_stage6_golden_snapshot.py`). DP is
strictly opt-in. The gen replica count is additionally clamped so each shard is a whole
number of groups (`min(replicas, n_groups)`), and the WARN at `humaneval.py:586` /
`math500.py:456` fires unchanged (bs is still 8 per replica).

### Per-replica student materialization

Each gen/PPL worker reloads the **compressed student from disk** (mirror of the Stage-3
worker, `covariance_collection.py:1119-1124`, `load_compressed_model`) with
`attn_implementation="eager"` (the gen path's hard requirement, `eval_harness.py:85-91`;
`run_pipeline.py:477-490`) and re-runs the generative-side environment restores
**inside the worker** (the contract published by `eval_environment.py` and consumed at
`humaneval.py:614-648` / `math500.py:469-492`): restore the uncompiled `pre_compile_forward`
and switch `experts_implementation` to the generative impl via
`_set_experts_implementation_s6`. The parent passes the resolved
`experts_implementation_generative` string + the `student_path` + per-eval cfg through
spawn args (all picklable). PPL workers keep the model **compiled / PPL experts-impl**
(PPL is forward-only and batch-invariant; `wikitext_ppl.py:155-165` only asserts eager
attn, which the load pins).

> Open question O1 (below): does Stage 6 already have a canonical on-disk student path at
> eval time, or must the orchestrator materialize the live student to a temp dir first
> (as the Stage-5 DDP plan does)? Resolve before Task 4.

### Import-contract note

`tools/eval_harness.py` is pipeline-free by contract (`eval_harness.py:9-16`). The new
`tools/eval_shard.py` is also a `tools/` leaf, but its workers need
`_set_experts_implementation_s6` (lives in `stage6/plugins/eval_environment.py`) and
`load_compressed_model` (`utils/model_io`). To avoid an import cycle, the workers do these
imports **function-locally inside the spawn target** (exactly as
`_cov_replica_worker` re-imports everything inside, `covariance_collection.py:1096-1110`).
The module top imports only stdlib + torch + `eval_harness`.

---

## Task list (bite-sized, TDD; CPU-simulated replicas / tiny model)

Each task: write the test FIRST, watch it fail, implement, watch it pass. Run the exact
command shown. Tests use CPU + a tiny stub model/tokenizer (no real GPU) — replicas are
simulated by running the worker bodies in-process / via spawn with `CUDA_VISIBLE_DEVICES=""`
so `device_count()==0` and the workers run on CPU. Live ≥2-GPU validation is **out of
scope** (deferred to a real multi-GPU box).

### Task 1 — Group-aligned + even split helpers (pure, no torch)

- File: `tools/eval_shard.py` (new). Add `_group_aligned_split` and `_even_split`.
- Test: `tests/test_eval_shard_split.py` (new).
- Assertions:
  - `_group_aligned_split(164, 2)` → `[(0,84),(84,164)]` — wait: 164 prompts = 20 full
    groups + 4 remainder. The LAST group is short (HumanEval 164 is not a multiple of 8);
    its 4 prompts form their own final group in the single-GPU run too. The split must put
    the boundary on a multiple of 8 (`k*8`), and the short tail stays attached to whichever
    shard owns it. Concretely: `n_groups = ceil(164/8) = 21`; `k = round(21/2) = 11`;
    boundary = `min(11*8, 164) = 88`. Result `[(0,88),(88,164)]`. The single-GPU run forms
    groups `[0:8],[8:16],...,[160:164]`; GPU0 owns groups 0..10 = prompts `[0:88]`, GPU1
    owns groups 11..20 = prompts `[88:164]` incl. the short last group `[160:164]`. **No
    group is ever split.** Pin this exact arithmetic in the test.
  - Every boundary in a `_group_aligned_split` result is `== 0 (mod 8)` OR `== n`.
  - `_group_aligned_split(n, replicas)` reconstructs `[0, n)` with no gap/overlap, in order.
  - `replicas` is clamped to `n_groups` (e.g. `_group_aligned_split(8, 5)` → one shard
    `[(0,8)]`, since there is only 1 group).
  - `_even_split(500, 2)` → `[(0,250),(250,500)]`; remainder rule = last shard absorbs it
    (mirror `_shard_calib`, `covariance_collection.py:1066-1070`).
- Command:
  `pytest max_quality/tests/test_eval_shard_split.py -v`

Complete code for the non-trivial split (the load-bearing part):

```python
PINNED_GEN_BATCH_SIZE = 8  # re-exported from tools.eval_harness; single source

def _group_aligned_split(n, replicas, group=PINNED_GEN_BATCH_SIZE):
    """Contiguous shards whose boundaries are multiples of ``group`` (or n).

    Guarantees every group-of-``group`` formed by the single-GPU
    ``_generate_batched`` slicing stays intact on exactly one shard, so the
    per-example completion (a pure function of its group + within-group order)
    is byte-identical to the single-GPU run after index-ordered concatenation.
    """
    if replicas <= 1 or n == 0:
        return [(0, n)]
    n_groups = (n + group - 1) // group          # ceil → the short tail is its own group
    replicas = min(replicas, n_groups)
    base = n_groups // replicas
    bounds, start_g = [], 0
    for r in range(replicas):
        end_g = n_groups if r == replicas - 1 else start_g + base
        start = start_g * group
        end = n if end_g == n_groups else end_g * group   # last shard → real n (short tail)
        bounds.append((start, end))
        start_g = end_g
    return bounds
```

### Task 2 — Gen merge by index (order-preserving concat)

- Add `run_dp_generate`'s **merge** as a separately-testable pure function
  `_merge_completions(shard_results) -> list[str]` (shards are `(start, end, completions)`,
  concatenated in `start` order).
- Test `tests/test_eval_shard_merge.py`: given 3 shards covering `[0,8),[8,16),[16,20)`
  with sentinel completion strings, `_merge_completions` returns the 20 in original index
  order; assert it `== ` a reference single-list. Assert a gap/overlap raises (defensive).
- Command:
  `pytest max_quality/tests/test_eval_shard_merge.py -v`

### Task 3 — PPL partial-NLL exact merge

- Add `_merge_ppl(partials) -> float` where `partials` is a list of `(nll_sum, tok_count)`;
  returns `exp(sum_nll / sum_tok)`, with the SAME `tok_count==0 → inf` and `OverflowError
  → inf` guards as `wikitext_ppl.py:273-295`.
- Test `tests/test_eval_shard_ppl_merge.py`: split a known chunk tensor's rows into 2
  shards, compute each partial with the SAME mean→sum rescale
  (`wikitext_ppl.py:260`), merge, and assert `math.isclose(merged_ppl,
  single_pass_ppl, rel_tol=0, abs_tol=0)` — i.e. **byte-exact** for the partial-sum merge
  given identical per-row forward (BATCH_INVARIANT property). (The per-row forward in the
  test is a deterministic stub, so this is a true equality, not a tolerance.)
- Command:
  `pytest max_quality/tests/test_eval_shard_ppl_merge.py -v`

### Task 4 — Gen replica worker + spawn driver (`run_dp_generate`)

- Implement `_gen_replica_worker` (module-level, picklable; mirror
  `_cov_replica_worker`, `covariance_collection.py:1074-1097`): set
  `CUDA_VISIBLE_DEVICES`, function-local imports, reload compressed student eager-attn,
  run the env restores (`pre_compile_forward`, generative experts-impl), call
  `_generate_batched(model, tok, prompts_shard, max_new=..., device=dev,
  batch_size=PINNED_GEN_BATCH_SIZE)` — **hard-pinned bs=8, no `size_batch`, no
  `run_with_oom_backoff`** — and write its completions list (JSON) to its replica file.
- Implement `run_dp_generate(prompts, *, student_path, replicas, gpus_per_replica,
  max_new, gen_impl, cfg, out_dir) -> list[str]`: `_group_aligned_split` → spawn
  (`torch.multiprocessing.get_context("spawn")`, `.Process(...).start()/join()`, exit-code
  check) mirroring `covariance_collection.py:1271-1282` → read replica files →
  `_merge_completions`.
- Test `tests/test_eval_shard_gen_driver.py`: monkeypatch the worker's generate to a
  deterministic `f(prompt)->completion` stub, run with `replicas=2` (CPU-spawn,
  `CUDA_VISIBLE_DEVICES=""`), and assert the merged list equals the single-process stub
  over the full prompt list. (Tests the spawn+merge plumbing, NOT a real model.)
- Command:
  `pytest max_quality/tests/test_eval_shard_gen_driver.py -v`

### Task 5 — PPL replica worker + driver (`run_dp_ppl`) with per-replica auto-batch

- Implement `_ppl_replica_worker`: pin GPU, reload student, forward its chunk sub-rows.
  Per-replica auto-batch: `resolve_batch(cost_probe_fn, fixed_batch=ppl_bs,
  FidelityClass.BATCH_INVARIANT, cfg=AutoBatchConfig.from_dict(...))`
  (`auto_batch.py:178-185`) probing THIS replica's pinned-device VRAM — exactly the
  Stage-3 cov-replica pattern (`covariance_collection.py:1206-1209`). Wrap the forward in
  `run_with_oom_backoff` (safe here — PPL is batch-invariant, halving bs does not change
  the metric). Accumulate `(nll_sum, tok_count)` with the `wikitext_ppl.py:260` rescale,
  write the partial to its replica file.
- Implement `run_dp_ppl(chunks, ...)`: `_even_split` → spawn → read partials →
  `_merge_ppl`.
- Test `tests/test_eval_shard_ppl_driver.py`: deterministic per-row loss stub, `replicas=2`
  CPU-spawn, assert merged PPL equals the single-process PPL (exact). Also assert that with
  `auto_batch.enabled=true` a (mocked) larger per-replica batch yields the **same** merged
  PPL (batch-invariance preserved).
- Command:
  `pytest max_quality/tests/test_eval_shard_ppl_driver.py -v`

### Task 6 — Config dataclass + default-OFF resolver

- Add `EvalShardConfig` (frozen dataclass, `from_dict` with `enabled` bool-coerce; copy the
  pattern from `AutoBatchConfig`, `auto_batch.py:55-72`) in `tools/eval_shard.py`.
- Add `_should_shard(cfg, n_examples) -> int` returning the effective replica count
  (`0/1` = in-process) after clamping to `device_count()` and (for gen) `n_groups`.
- Test `tests/test_eval_shard_config.py`: `enabled:false`→0; string `"true"`+`replicas:4`
  on a (mocked) 2-GPU box → 2; `replicas:8` with 3 groups (gen) → 3; `device_count()<2` →
  0 (forces in-process).
- Command:
  `pytest max_quality/tests/test_eval_shard_config.py -v`

### Task 7 — Wire the three plugins (one branch each, default unchanged)

- `humaneval.py` `_humaneval`: after prompts are built (`humaneval.py:277-296`) and
  before `_generate_batched` (`humaneval.py:311-314`), if `_should_shard(...) > 1` call
  `run_dp_generate(prompts, ...)` instead; else the existing in-process call **verbatim**.
  Scoring (`_score_all_humaneval`, `humaneval.py:321`) runs unchanged in the parent.
- `math500.py` `_math500`: same branch around `_generate_batched` (`math500.py:236-239`);
  SymPy grading (`math500.py:241-251`) unchanged in the parent.
- `wikitext_ppl.py` `_wikitext2_ppl`: after `chunks` is built (`wikitext_ppl.py:216`),
  branch to `run_dp_ppl(chunks, ...)` instead of the in-process forward loop
  (`wikitext_ppl.py:237-272`) when sharded; the in-process path is the default.
- The `eval_shard` cfg is read from `s6` in each plugin's `eval_task` hook and threaded
  into the `_humaneval`/`_math500`/`_wikitext2_ppl` call (new kwarg, default `None` = OFF),
  alongside the resolved `experts_implementation_generative` / `student_path` (the worker
  reload inputs).
- Tests:
  - `tests/test_humaneval_shard_passthrough.py`: with `eval_shard.enabled=false`, assert
    `_humaneval` calls `_generate_batched` exactly once in-process (no spawn) and the
    result is unchanged. (Mirror for math500 / wikitext.)
  - Each plugin's existing golden / unit tests still pass unchanged.
- Commands:
  `pytest max_quality/tests/test_humaneval_shard_passthrough.py max_quality/tests/test_math500_shard_passthrough.py max_quality/tests/test_wikitext_shard_passthrough.py -v`

### Task 8 — GOLDEN GATE: 1-GPU completions == concat of 2-shard completions

This is the load-bearing guardrail (the risk is entirely shard-alignment; an off-by-group
split silently changes the metric — `stage6.md:222-225`).

- Test `tests/test_eval_shard_golden_gate.py`: with a tiny deterministic stub model +
  tokenizer (left-pad, greedy, `pad_to_multiple_of=64`) so completions are a real function
  of group geometry:
  1. Build a 20-prompt list with **deliberately varied lengths** so group-of-8 padding
     widths differ across groups (this is what would expose an order/boundary bug).
  2. Run `_generate_batched(..., batch_size=8)` single-process → `golden`.
  3. Run `run_dp_generate(..., replicas=2)` (CPU-spawn) → `sharded`.
  4. Assert `sharded == golden` **byte-identical** (list equality of decoded strings).
  5. Negative control: assert a NAIVE even split (`_even_split` on the gen path, boundary
     NOT a multiple of 8, e.g. `[(0,10),(10,20)]`) produces a DIFFERENT result from
     `golden` on the crafted prompts — proving the gate actually detects a broken split
     (so the gate is not vacuous). If the stub is too benign to drift, document that the
     real-model drift is asserted by the `eval_harness` bs-invariance reasoning and keep
     the positive byte-identical assertion as the binding gate.
  6. PPL analogue: `run_dp_ppl(chunks, replicas=2)` exact-equals the single-pass PPL.
- Command:
  `pytest max_quality/tests/test_eval_shard_golden_gate.py -v`

### Task 9 — Stage 6 golden snapshot stays byte-identical (regression guard)

- With `eval_shard` absent / `enabled:false` (the fixture default), re-run the existing
  `test_stage6_golden_snapshot.py` and `test_stage6_orchestrator.py` — they MUST pass
  unchanged (default path is the current instruction stream).
- Command:
  `pytest max_quality/tests/test_stage6_golden_snapshot.py max_quality/tests/test_stage6_orchestrator.py -v`

### Task 10 — Full Stage 6 + harness suite green

- Command:
  `pytest max_quality/tests/ -k "stage6 or eval_shard or eval_harness or humaneval or math500 or wikitext" -v`

---

## Key decisions (the binding guarantees)

1. **Group-of-8 boundary guarantee (gen path).** Gen shards are cut ONLY on multiples of
   `PINNED_GEN_BATCH_SIZE=8` (`_group_aligned_split`), within-group order untouched, so
   each replica forms the IDENTICAL group geometry to the single-GPU run for the groups it
   owns. The short final group (164 not divisible by 8) is never split — it rides whole on
   the last shard. This is the only thing that makes the METRIC_PINNED metric byte-identical
   under DP (`stage6.md:106-121`).

2. **Merge.** Gen: concatenate contiguous shard completion-lists in `start` order →
   original index order (a pure function-of-group result, `stage6.md:108-121`). PPL: sum
   the per-shard `(nll_sum, tok_count)` partials and take `exp(sum_nll/sum_tok)` — exact
   because `out.loss`→sum rescale is order/batch-independent (`wikitext_ppl.py:260-293`).

3. **Auto-batch ONLY on PPL.** Gen replicas run a **hard-pinned bs=8** with NO `size_batch`
   and NO `run_with_oom_backoff` (both would move bs off 8 and flip near-tied argmax —
   `stage6.md:123-140`). PPL replicas DO size per-GPU via `resolve_batch`(BATCH_INVARIANT)
   + OOM-backoff (safe; batch-invariant), each probing its own pinned-device VRAM — the
   exact Stage-3 cov-replica pattern (`covariance_collection.py:1206-1209`).

4. **Process-per-GPU.** Sidesteps the shared-tokenizer-state race in `_generate_batched`
   (`eval_harness.py:73-96, 163-165`, H2) and CUDA-context sharing; mirrors the Stage-3
   spawn driver. No NCCL — the reduce is a parent-side concat / sum over replica files.

5. **Default single-GPU byte-identical.** `enabled:false` / `replicas<=1` /
   `device_count()<2` → the existing in-process call runs unchanged (no spawn, no reorder,
   no new ops). Stage 6 goldens stay green (Task 9). Opt-in only.

6. **CPU `ProcessPool` untouched.** HumanEval's scoring pool is exec-isolation, not GPU
   parallelism (`stage6.md:61-74`); scoring still runs serially in the parent after merge.

---

## Out of scope

- **Live ≥2-GPU validation** (real RTX/H200 run) — deferred to a real multi-GPU box; all
  tests here are CPU-simulated replicas / tiny stub model.
- **Teacher-side DP.** The teacher re-runs the same evals (`teacher_provider.py`); it
  inherits whatever the gen/PPL drivers provide, but wiring the teacher hook is a
  follow-up (cache-hit makes the teacher free anyway — `stage6.md:54, 214`). Note: the
  teacher's own `gen_batch_size` pin (`teacher_provider.py:571-576`) mirrors the student's.
- **zero-shot lm-eval** (`zero_shot_lm_eval.py`) — delegate to lm-eval's OWN
  data-parallel / accelerate launch, NOT re-sharded in-tree (`stage6.md:53, 230`).
- **imatrix / GGUF** subprocess (own `--ngl`), `eval_environment`, `validation_report` —
  not applicable to eval-shard (`stage6.md:55-57`).
- **Thermometer / stage6alt** path (`run_pipeline.py:342-344`) — separate evaluator,
  out of scope (`stage6.md:171-172`).

---

## Open questions (resolve in review, before coding the relevant task)

- **O1 (Task 4/5 blocker):** Does Stage 6 have a canonical on-disk student path at eval
  time that a spawned worker can `load_compressed_model` from, or must the orchestrator
  first materialize the live (possibly-compiled) student to a temp dir (as the Stage-5 DDP
  plan does)? The Stage-3 worker reloads from `student_path`
  (`covariance_collection.py:1119`); Stage 6 receives a live `model` object
  (`orchestrator.py:117, 129`). If no canonical path exists, add a parent-side
  `model.save_pretrained(tmp)` step before spawning (and confirm the saved student
  reproduces eager-attn + the generative experts-impl on reload).

- **O2:** Confirm the reload faithfully reproduces the generative-side environment a
  worker needs: eager attn (load-pin, `run_pipeline.py:483`) + `pre_compile_forward`
  restore + generative experts-impl (`humaneval.py:614-648`). `pre_compile_forward` is a
  bound method (not picklable) — the worker must REBUILD the uncompiled forward by simply
  NOT compiling (load → no torch.compile → forward is already uncompiled), so the parent
  passes only the `experts_implementation_generative` STRING + cfg, never the closure.
  Verify this matches the in-process contract's effect.

- **O3:** Spawn-overhead vs. payoff: each replica pays a full ~50 GB student reload
  (`stage6.md:180` "after replica-load overhead"). For the cheap WikiText-PPL path the
  reload may dominate — gate PPL-DP behind an explicit opt-in and/or a min-chunk threshold
  so it never regresses wall-clock. (Gen path decode budget dominates, so reload is
  amortized — `stage6.md:51, 155-156`.)

- **O4:** `max_workers` / GPU pinning when `gpus_per_replica > 1` — Stage-3 pins
  contiguous device subsets (`covariance_collection.py:1254-1256`). For a model that fits
  on ONE GPU, `gpus_per_replica` should default to 1; decide whether to support >1 at all
  (likely not needed — drop to keep it simple unless review wants it).
