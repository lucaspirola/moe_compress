# Router-KD multi-GPU analysis (Stage 2.5 heal + Stage 5 final)

Read-only deep analysis. Code root: `max_quality/src/moe_compress/router_kd/`.
Same code serves both Stage 2.5 (`stage_key="stage2p5"`) and Stage 5
(`stage_key="stage5"`) — selected by the `stage.py` factory
(`router_kd/stage.py:41`). Everything below applies to BOTH invocations.

This is the suspected highest-value multi-GPU opportunity: Router-KD is the
only **training** stage (real backward + optimizer steps over a 3000-sample
calibration set, default 2 epochs), so it is the pipeline runtime long pole.

---

## 1. Executive verdict (read first)

**Training is currently single-process, single optimizer.** There is **no
`torch.distributed`, no `DistributedDataParallel`, no `DataParallel`, no
`init_process_group`/`all_reduce` anywhere in the codebase** (grep over
`src/moe_compress/` returns nothing — verified). The only "multi-GPU" the loop
supports today is HF `device_map` (naive **pipeline-parallel / model-sharded**),
which the resume code explicitly tolerates (`orchestrator.py:469-481` handles
"trainable parameters span N devices"). The live ablation runs
`device_map=balanced` purely for **memory** (fit the student + ~70 GB BF16
teacher), and pipeline-MP is **sequential** — only one GPU computes at a time —
so it is **slower than single-GPU** for the compute itself; it buys capacity,
not speed.

**DDP is the right tool and it is result-preserving.** The loss is a per-token
mean vocab-KL (`vocab_kd.py:132`) and the optimizer does exactly one
`optim.step()` per grad-accum window (`orchestrator.py:876`). DDP that splits
the per-step batch across replicas and **all-reduce-averages the gradients**
reconstructs the identical full-batch gradient (modulo fp reduction order), so
the trained result matches single-GPU at the **same effective batch**. This is
fundamentally different from *raising* `batch_size` (which IS forbidden — see
§3), because DDP keeps the effective batch fixed.

**Effort: M (medium).** No algorithmic change; the loop is already
plugin-clean. The work is: process-group bootstrap, wrap student in DDP,
shard the calibration iterator per rank, gate all I/O + decisions on rank 0 +
broadcast, and (the one real subtlety) keep the **teacher** present per rank or
shared. Plus a one-time decision on whether the **teacher** is replicated
(VRAM) or kept pipeline-sharded.

**The fit question (does the student fit on one GPU?) decides the teacher
strategy, not DDP viability** — see §5.

---

## 2. Per-plugin / per-component table

Legend — **DDP-compat**: does the component work unchanged under DDP at the
same effective batch? **Must change**: what (if anything) DDP requires.

| Component (file:line) | What it computes | Train vs support | DDP-compat | Must change |
|---|---|---|---|---|
| **orchestrator.run** (`orchestrator.py:158`) | The whole loop: epoch/batch iteration, teacher dispatch, student fwd, loss, backward, `optim.step`, scheduler, checkpoint, early-stop break | TRAINING (the loop itself) | Partial | Wrap student in DDP; shard `batches` per rank; rank-0-gate all logging/trackio/checkpoint; broadcast early-stop decision. See §4/§6. |
| **RkdPaperRecipePlugin** (`rkd_paper_recipe.py:170`) | Pre-flight config mutation (τ=4, wd=0, epochs=2, patience=0, cache=None). Pure dict edit, no tensors. | support | ✅ identical | None — runs on every rank deterministically, same result. |
| **TrainableScopePlugin** (`trainable_scope.py:137`) | Freezes all non-router params; pattern-conflict check. `requires_grad_` in place. | support (setup) | ✅ identical | None — every rank freezes the same params; DDP wrap must happen AFTER this (it does, see §4 ordering). |
| **KdOptimizerPlugin** (`kd_optimizer.py:177`) | Builds AdamW (split group if merge-repair) + warmup/cosine `LambdaLR`. LR/warmup keyed to `total_optim_steps`. | TRAINING (optimizer) | ✅ result-preserving | `total_optim_steps` must be the **global** step count = `(len(GLOBAL_batches)//grad_accum)*epochs`, computed identically on every rank. Since each rank sees `1/n` of the batches but the step counter advances once per *global* grad-accum window, compute it from the global batch count (it already is — `len(batches)` is global pre-shard; keep that). Optimizer wraps DDP-managed `.parameters()` — unchanged. |
| **VocabKdPlugin** (`vocab_kd.py:291` / `_chunked_vocab_kl:72`) | Per-token mean KL over `[B,L,\|V\|]`, ÷ `n_tokens`, ×τ². The loss. | TRAINING (loss) | ✅ result-preserving **with caveat** | The KL is a **per-token MEAN with a LOCAL denominator** `n_tokens = B_local·(L-1)`. DDP averages *gradients*, which for a per-replica mean loss gives the **correct global mean ONLY if every replica has the same token count** — true here because batches are uniform/fully-packed (`_chunked_vocab_kl` asserts shape parity; calibration is packed, no padding). With equal `B_local` per rank and equal `L`, `mean(grad over ranks) == grad of global mean`. **Gotcha:** the global batch must be divisible by `n_gpu` so every rank gets equal `B_local` (see §3/§6). If a rank ever got a short batch, the per-token mean weighting would skew — must drop-last or pad-equal. |
| **TeacherCachePlugin** (`teacher.py:170`) | Loads precompute sidecar, slices `[token_start:token_end]` per batch. No teacher fwd. | support (teacher source) | ✅ with index fix | Per-batch slice indexes by **global** `(epoch*num_batches+batch_index)` (`teacher.py:498`). Under DDP each rank must index by its **global** batch position, not its local position. The mmap cache is read-only (shareable across ranks on one node). Note: paper-dials default sets `epochs=2` which **disables the cache** (`rkd_paper_recipe.py:223` clears it, and `orchestrator.py:607` rejects epochs>1+cache) — so on the DEFAULT recipe the cache is never active and the live teacher path (below) is what matters. |
| **TeacherLivePlugin** (`teacher.py:510` / `_load_teacher:562`) | Lazy-loads ~70 GB BF16 teacher, no-grad forward per batch → `[B,L,\|V\|]` logits. | TRAINING-adjacent (forward only, no grad) | ✅ but VRAM-heavy | This is the crux. Each DDP rank needs teacher logits for ITS local batch. Two options (§5): **(A) replicate** the teacher on each GPU (needs teacher+student to co-fit per card), or **(B) keep the teacher pipeline-sharded** across a *separate* GPU set / shared. The forward is stateless + no-grad, so it is trivially data-parallel; the only question is VRAM. Device-map derivation (`teacher.py:622-641`) already co-locates teacher with the student device — under DDP that becomes per-rank. |
| **MergeRepairPlugin** (`merge_repair.py:410`, Stage 2.5 only, opt-in default-OFF) | Unfreezes merged centroid rows + grad-mask hook; per-layer MoE-block MSE vs teacher block outputs (forward hooks on `mlp`). | TRAINING (extra params + extra loss term) | ⚠️ needs care | The grad-mask `register_hook` (`merge_repair.py:281`) runs in autograd → fires per replica → DDP all-reduces the **masked** grads → correct (mask is deterministic per row). The MSE term (`_merge_repair_mse:350`) is a `mean` over layers of `F.mse_loss` (element-mean) → same per-replica-mean→global-mean argument as the KL holds **iff** equal `B_local`. The `_LayerOutputCapture` forward hooks (`merge_repair.py:290`) attach to `student`/`teacher` module trees — **must attach to the DDP-wrapped student's `.module`** (the underlying model), since DDP wraps and the hooks key on `iter_moe_layers`. Default-OFF, so DDP can ship without merge-repair first and add it as a follow-up. |
| **EarlyStopPlugin** (`early_stop.py:179`) | EMA of raw-KL, `best.pt` save-on-improve, patience early-stop decision, end reload. | support (convergence policy) | ⚠️ MUST synchronize | EMA/patience are computed from `raw_kl_val` = the **window-mean loss** (`orchestrator.py:893`). Under DDP each rank only sees its local loss; the `raw_kl_val` fed to the tracker must be the **all-reduced (mean) loss across ranks** so every rank makes the SAME decision — else ranks diverge on when to `break` → deadlock (one rank exits the loop, others wait on its all-reduce). `best.pt` write must be **rank-0 only** (`_save_best_router_state:126`). `reload_best_checkpoint:467` must run on all ranks (or rank-0 reload + broadcast) so every replica ends with identical weights. Patience default is 0 on the paper recipe → block is inert, but the EMA/best save still runs. |

---

## 3. The METRIC_PINNED constraint and how DDP threads through it

Router-KD batch is **METRIC_PINNED** in the FidelityClass taxonomy
(`utils/auto_batch.py:42`). The taxonomy is confirmed by the sibling training
stage's note: `stage3/plugins/block_refine.py:172-176` —

> "block_refine is METRIC-PINNED (minibatch-SGD). Changing batch_size changes
> the minibatch grouping and therefore the trained weights, so it gets NO
> auto_batch wiring."

and `stage1/plugins/ablation_filter.py:126` repeats it for block_refine. The
auto_batch resolver is **gated to `BATCH_INVARIANT` only** (`_V1_ELIGIBLE =
frozenset({FidelityClass.BATCH_INVARIANT})`, `auto_batch.py:52`), and
`resolve_batch` is a hard no-op for any other class (`auto_batch.py:182`).

**Router-KD is not even wired to auto_batch** — grep for
`auto_batch`/`resolve_batch`/`AutoBatchConfig` in `router_kd/` returns nothing.
The forward batch is the fixed config `s5["batch_size"]` (`orchestrator.py:302`),
exactly because it is metric-pinned: the loop does one `optim.step` per
grad-accum window (`orchestrator.py:852-879`) and the LR schedule is keyed to
that step count (`kd_optimizer.py:262-275`), so changing the batch changes the
number of steps, the LR-per-token, and the SGD trajectory → a different model.

### Why DDP is exempt where raising batch_size is not

- **Raising `batch_size`**: fewer optimizer steps, each over more tokens → the
  SGD trajectory and the LR schedule's per-token meaning both change → DIFFERENT
  trained model. Forbidden. This is the whole reason METRIC_PINNED exists.
- **DDP at the SAME effective batch**: split the per-step global batch of `G`
  sequences across `n` replicas (`G/n` each), all-reduce-**average** the
  gradients. The averaged gradient over the `n` shards = the gradient of the
  loss over the full `G` sequences (the per-token mean is linear in the
  per-replica means when token counts are equal). Same number of optimizer
  steps, same LR schedule, same trajectory → **same trained model** (modulo
  fp reduction order, the usual DDP tolerance). The "minibatch grouping" that
  METRIC_PINNED protects is **unchanged**: the global minibatch is still `G`
  sequences per step; DDP only parallelizes their gradient computation.

**Hard requirement to preserve the result: `per_gpu_batch = global_batch /
n_gpu`, with `global_batch` held equal to the single-GPU `s5["batch_size"]`.**
This means `s5["batch_size"]` must be **divisible by `n_gpu`** (or use a
DistributedSampler with drop_last and accept the dropped tail, matching the
existing `trailing` drop at `orchestrator.py:560`). With the paper recipe's
`batch_size=2`, `n_gpu` can be 2 (1 seq/GPU) but not >2 without raising the
global batch — which would break the metric pin. So **DDP scaling is capped at
`n_gpu ≤ global_batch`** unless the operator deliberately co-scales
grad_accum down (keeping effective batch fixed) — see §6 gotchas.

### How per-GPU auto-batch interacts (the user's explicit requirement)

The user requires that any multi-GPU scheme **preserve per-GPU VRAM-aware batch
sizing**. For Router-KD the honest answer is:

- The per-GPU forward batch is **NOT VRAM-auto-sized** here and must not be —
  it is **fixed by the science** to `global_batch / n_gpu` (metric-pinned).
  Auto-batch's *maximizer* role (grow the batch to fill VRAM) is
  **categorically wrong** for this stage and is correctly already disabled
  (router_kd is not wired to it, and even if it were, `resolve_batch` no-ops
  for non-`BATCH_INVARIANT` classes).
- Auto-batch's role here is **only the OOM-safety floor**, i.e. the
  `run_with_oom_backoff` halving-toward-floor mechanism (`auto_batch.py:188`).
  But under the metric-pin, the per-GPU batch CANNOT be halved without changing
  the result, so even the backoff is not freely usable: an OOM at
  `global_batch/n_gpu` per rank must be resolved by **adding GPUs** (smaller
  per-rank batch at fixed global batch) or **raising grad_accum** (smaller
  per-microbatch at fixed effective batch) — both of which keep the metric pin
  — NOT by silently halving one rank's batch.
- **Therefore: per-GPU VRAM-awareness is preserved as a FLOOR/admission check,
  not a maximizer.** Concretely: compute `per_gpu_batch = global_batch /
  n_gpu`; if that does not fit a rank's VRAM, the correct levers are (raise
  grad_accum at fixed effective batch) or (more GPUs), surfaced as a loud
  precondition — never an auto-resize of the metric-pinned batch. This matches
  exactly how block_refine (the other METRIC_PINNED training path) refuses
  auto_batch wiring.

---

## 4. Is the training loop DDP-compatible & result-preserving? (the deep read)

Walking the loop (`orchestrator.py:665-1028`) component by component:

**Optimizer step / LR schedule / step counter** — `optim.step()` once per
grad-accum window (`:876`), `scheduler.step()` + `step += 1` immediately after
(`:878-879`). The schedule is `(current_step+1)/warmup` then cosine
(`kd_optimizer.py:264-275`), keyed to `total_optim_steps`
(`orchestrator.py:332`). **Result-preserving under DDP** provided `step` stays
the **global** step (it does: every rank runs the identical loop with the same
grad-accum boundaries, so `step` advances in lockstep) and `total_optim_steps`
is computed from the **global** batch count (it is — `len(batches)` is the full
calibration tensor's batch count, computed before any per-rank shard). DDP's
gradient all-reduce happens inside `(loss/grad_accum).backward()` (`:850`); with
gradient accumulation, the standard DDP idiom is `no_sync()` on the
non-boundary microbatches and a live all-reduce on the boundary one — a
**correctness-and-throughput** change (without `no_sync` you all-reduce every
microbatch, still correct, just slower). Must add `no_sync()` context around
the `(i+1)%grad_accum != 0` microbatches.

**The loss** — per-token mean (`_chunked_vocab_kl:132`), local denominator.
DDP averages gradients → correct global-mean gradient **iff equal token count
per rank**. Calibration is fully-packed (uniform `L`, asserted at
`vocab_kd.py:112-117`), so equal `B_local` ⇒ equal token count ⇒ exact. This is
the single most important correctness fact and it **holds** for this codebase.

**Teacher forward** — no-grad (`teacher.py:764`), stateless. Trivially
data-parallel: each rank computes teacher logits for its local batch. The only
cost is VRAM (§5).

**Early-stop / EMA / best-checkpoint** — the ONE place that breaks silently
under naive DDP. `raw_kl_val` is the window-mean of the **local** loss
(`orchestrator.py:889-895`). Each rank would compute a *different* `raw_kl_val`
→ different EMA → different `early_stop_should_stop` → one rank `break`s while
others block on the next all-reduce → **hang**. Fix: all-reduce-mean the window
loss before feeding `update_best_tracker`, OR compute the decision on rank 0 and
**broadcast** the stop flag. Either keeps the decision identical to single-GPU
(single-GPU's `raw_kl_val` is the mean over the full global batch; the
all-reduced multi-GPU value equals it). `best.pt` save → rank-0 only.

**Checkpoint / resume** — `_save_stage5_checkpoint` (`:1197`) +
`_Stage5CheckpointWriter` (async, `:1130`) → **rank-0 only**. Resume restore
(`:403-511`) must run on all ranks (each loads the same router_state into its
replica) OR rank-0 loads + DDP broadcasts at wrap time (DDP broadcasts module
state on construction, so rank-0-load-then-wrap is the clean idiom). The
existing multi-device resume handling (`:469-481`) is for device_map sharding,
not DDP, and can stay as a fallback.

**`save_compressed_checkpoint`** (`:1053`) — rank-0 only; unwrap DDP
(`student.module`) the same way it already unwraps `torch.compile`
(`getattr(student, "_orig_mod", student)` → add `getattr(..., "module", ...)`).

**Ordering constraint** — DDP must wrap **after** `setup_trainable_scope`
(freeze) and **after** `build_optimizer`, but the optimizer must hold the
DDP-managed params. Cleanest: freeze → build optimizer over the raw params →
wrap student in DDP (DDP finds the same leaf params; `requires_grad` already
set) → train. `torch.compile` currently wraps after optimizer
(`:359-368`); DDP+compile ordering is `DDP(compile(model))` or
`compile(DDP(model))` — both supported, but pick one and pin it.

---

## 5. Does the student fit on one GPU? → teacher strategy (the fit question)

DDP requires a **full model replica per GPU**. So the gating question is
whether `student + teacher` co-fit per card.

- The **student** is the *compressed/pruned* model (Stage 2.5 runs post-merge;
  Stage 5 runs post-Stage-4). At the project's net-35% compression the pruned
  student is on the order of ~50 GB (the prompt's figure). On an 80 GB H200 the
  student replica fits with room to spare.
- The **teacher** is the ~70 GB BF16 *uncompressed* model (`teacher.py:44-46`,
  the 4-bit and FP8-override paths exist precisely to relieve this). Student
  (~50 GB) **+** teacher (~70 GB) = ~120 GB does **NOT** co-fit on one 80 GB
  card. This is why the live ablation uses `device_map=balanced` — it is a
  *memory* spill, not a speed choice.

Two DDP-compatible teacher strategies:

- **(A) Replicate-teacher DDP (cleanest, needs the VRAM).** Each rank holds its
  own student replica + its own teacher. Requires teacher+student per card.
  Viable only with a **4-bit / FP8 teacher** (`teacher_load_in_4bit=true` or a
  `teacher_model_repo` FP8 override → ~18–35 GB) so ~50 GB student + ~20–35 GB
  teacher fits 80 GB. The KD target is then an *approximation* of θ_T (the
  module docstring flags this as a deviation, `teacher.py:42-55`), so this is a
  **quality trade** the user must sanction — NOT result-preserving vs the BF16
  teacher. **OR** use the **precompute cache** (`TeacherCachePlugin`): logits
  precomputed once, mmap'd read-only, shared across ranks → **zero teacher VRAM
  per rank**, BF16-faithful, and result-preserving. The cache is the ideal DDP
  teacher path — BUT it is incompatible with the default `epochs=2` paper recipe
  (`orchestrator.py:607`) and with merge-repair (`merge_repair.py:539`). For a
  1-epoch, no-merge-repair Stage 5 run, **cache + DDP is the clean win**.

- **(B) Hybrid: DDP students + shared/pipeline teacher.** Keep the student
  data-parallel across `n` GPUs (one replica each) and run a **single** BF16
  teacher pipeline-sharded across a *separate* GPU group (or the same GPUs with
  the teacher's layers spilled). Each rank sends its local batch to the shared
  teacher and gets logits back. More plumbing (teacher becomes a cross-rank
  service); the teacher forward is the serial bottleneck again. Less attractive
  than (A)-with-cache.

**Bottom line on fit:** DDP of the *student* is clearly viable (it fits one
card). The teacher VRAM is the only obstacle, and it has a **clean
result-preserving answer (precompute cache, shared mmap)** for the 1-epoch /
no-merge-repair case, and an **approximate answer (4-bit/FP8 replicated
teacher)** for the multi-epoch / merge-repair case that needs user sign-off as a
quality trade.

---

## 6. DDP feasibility verdict — Router-KD

**Result-preserving?** YES, at the same effective batch, for the vocab-KL loss
(the production path), provided the gradient is all-reduce-**averaged** and the
per-rank token count is equal (guaranteed by packed calibration + equal
`B_local`). The early-stop/best-tracker decision must be driven by the
all-reduced (global) window loss so it matches single-GPU bit-for-decision.
The teacher choice determines whether the *KD target* is faithful: BF16 live
or BF16 precompute cache = faithful; 4-bit/FP8 = approximate (quality trade).

**Effort: M.** No algorithm change; the loop is plugin-clean and already
separates policy (plugins) from control flow (orchestrator). The work:
1. Process-group bootstrap (`torchrun` / `init_process_group("nccl")`), rank/
   world-size plumbing into `run()`.
2. `DistributedSampler`-style shard of the `batches` list per rank (or slice
   `iter_batches` output); keep `len(batches)` (global) for `total_optim_steps`.
3. Wrap student in DDP after freeze+optimizer-build; add `no_sync()` around
   non-boundary grad-accum microbatches.
4. All-reduce-mean the window loss before `update_best_tracker`; gate `best.pt`
   + `step_*.pt` + `save_compressed_checkpoint` + all logging/trackio on rank 0;
   broadcast the early-stop flag (or derive it from the reduced loss so it's
   identical on every rank).
5. Teacher: wire the precompute-cache path for the 1-epoch case (zero per-rank
   teacher VRAM); document the 4-bit/FP8 replicated-teacher path as the
   multi-epoch fallback (quality trade, user sign-off).
6. Unwrap `.module` alongside the existing `_orig_mod` unwrap at every
   `named_parameters()` / save / iter_moe_layers site.

**Gotchas (each maps to a concrete code site):**

- **Global step count.** `total_optim_steps` (`orchestrator.py:332`) and the LR
  schedule (`kd_optimizer.py:262`) MUST stay keyed to the **global** batch
  count, not the per-rank shard. The step counter `step` advances once per
  global grad-accum window and stays in lockstep across ranks — verify the
  shard does not change the number of grad-accum windows per epoch.
- **Effective batch fixed (METRIC_PINNED).** `per_gpu_batch = global_batch /
  n_gpu`. `global_batch` must equal the single-GPU `s5["batch_size"]`. With
  `batch_size=2` (paper default) DDP caps at `n_gpu=2` unless grad_accum is
  co-scaled DOWN to keep effective batch constant. Do NOT let auto-batch raise
  the per-GPU batch — it is metric-pinned (it is already not wired; keep it so).
- **Gradient all-reduce must be MEAN, not SUM.** DDP defaults to averaging,
  which is exactly the per-token-mean-loss semantics. Confirm no manual
  `loss * world_size` rescale sneaks in.
- **Equal token count per rank.** Drop-last or pad-equal so no rank gets a short
  final batch (the per-token mean denominator `n_tokens`, `vocab_kd.py:119`,
  is local — unequal counts skew the global mean). The existing `trailing` drop
  (`orchestrator.py:560`) already discards the ragged tail; extend it to also
  enforce per-rank evenness.
- **Synchronized early-stop.** Feed `update_best_tracker` the all-reduced
  window loss; gate `best.pt` on rank 0; broadcast `early_stop_should_stop`
  before the `break` (`orchestrator.py:998`, `:1024`). A per-rank-local decision
  → deadlock (one rank breaks, the rest hang on the next all-reduce).
- **Rank-0-only I/O.** `_Stage5CheckpointWriter` (async thread, `:1130`),
  `_save_best_router_state` (`early_stop.py:126`), trackio (`:943`), and
  `save_compressed_checkpoint` (`:1053`) must be rank-0-guarded or N ranks race
  the same files (torn writes / N× the I/O).
- **Resume + DDP.** Cleanest is rank-0 loads router_state then DDP broadcasts on
  wrap; the existing CPU-checkpoint multi-device migration (`:469-481`) is
  device_map-era and can be retired for the DDP path.
- **Teacher cache index is global.** `token_start` (`teacher.py:498`) must use
  the rank's **global** batch index, not its local loop index.
- **Merge-repair (Stage 2.5, opt-in).** Forward-capture hooks
  (`merge_repair.py:290`) attach to module subtrees — must attach to
  `ddp_student.module`. Grad-mask hook + MSE term are DDP-safe (deterministic
  mask, mean MSE with equal `B_local`). Ship DDP first WITHOUT merge-repair
  (default-OFF), add it as a follow-up.
- **`torch.compile` + DDP ordering.** Pin one of `DDP(compile(m))` /
  `compile(DDP(m))`; the loop already special-cases `_orig_mod` unwrapping, so
  add the `.module` unwrap consistently.

**Expected speedup.** Near-linear in `n_gpu` for the *compute* (student fwd+bwd
+ teacher fwd are the per-step cost and parallelize cleanly), capped by
`n_gpu ≤ global_batch` under the metric pin, minus all-reduce overhead (router
gradients are small — `best.pt` is ~10–50 MB of router params per
`early_stop.py:135`, so the all-reduced gradient is tiny → low comm cost → DDP
efficiency should be high). This is a genuine, result-preserving training
speedup, unlike the current `device_map=balanced` which serializes the GPUs.

---

## 7. Summary

- **Currently single-GPU training.** No DDP/distributed anywhere. The only
  multi-GPU is HF `device_map` (pipeline-MP) used for memory; it is *slower*
  than single-GPU compute.
- **DDP is viable and result-preserving** at the same effective batch — the
  per-token-mean vocab-KL loss + average-gradient DDP reconstruct the
  single-GPU full-batch gradient exactly (equal token counts hold via packed
  calibration). METRIC_PINNED forbids *raising* the batch but **not** DDP.
- **Per-GPU batch is fixed by science** (`global/n_gpu`), not VRAM-auto-sized;
  auto-batch's role is an OOM **floor/admission check**, never a maximizer here
  — and even the OOM-halve is unusable on a metric-pinned batch (resolve OOM via
  more GPUs or higher grad_accum, not by resizing the batch).
- **Teacher VRAM is the only real obstacle.** Clean answer = precompute cache
  (BF16-faithful, shared mmap, zero per-rank teacher VRAM) for 1-epoch /
  no-merge-repair; fallback = 4-bit/FP8 replicated teacher (quality trade,
  needs sign-off) for the multi-epoch paper-default recipe.
- **Effort M;** the gotchas are the standard DDP checklist (global step,
  rank-0 I/O, synchronized early-stop, equal per-rank batch, `.module` unwrap),
  all mapping to concrete call sites above.
