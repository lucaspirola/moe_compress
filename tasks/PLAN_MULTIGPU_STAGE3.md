# PLAN — N-GPU multi-device support for Stage 3 (cov collection + model load)

Branch: `plan/multigpu-stage3` off `main` @ `6aa568f`.
Scope owner: Stage 3 AA-SVD covariance collection (the dual-forward) + the two
load functions it depends on. Author: planner pass, code NOT written here.

All file:line citations verified by Read/grep against the working tree at
`6aa568f` on 2026-06-11.

---

## 0. TL;DR — the chosen design

**Two independent levers, both auto-detected from `torch.cuda.device_count()`,
both no-ops at 1 GPU:**

1. **Model-sharding lever (the immediate VRAM unblock).** Today the *student*
   model in Stage 3 is loaded by `load_compressed_model`, which is HARD-PINNED
   to a single device (`model_io.py:1455-1458`). The *teacher* is loaded by
   `load_model` with `device_map="auto"`, which **already shards** across all
   visible GPUs for the non-4bit path (the forced `{"":0}` at
   `model_io.py:98-99` is ONLY the 4bit branch). So today's bottleneck is
   asymmetric: teacher can shard, student cannot. Fix = teach
   `load_compressed_model` to honor a real `accelerate` `device_map`
   ("auto"/"balanced"/explicit `max_memory`) so the *student* shards too.

   **What sharding actually buys (H1 — weight-VRAM vs activation-VRAM,
   precisely):** the MoE experts are **fused stacked `nn.Parameter`s** — one
   `gate_up_proj : [num_experts, 2*moe_intermediate, hidden]` and one
   `down_proj : [num_experts, hidden, moe_intermediate]` per layer
   (`model_io.py:10-13`; axis convention `[num_experts, d_out, d_in]` at `:13`).
   accelerate's `infer_auto_device_map` / `dispatch_model` place at **module
   granularity** and **cannot split a single stacked parameter**. So sharding
   distributes **whole decoder layers** across the N GPUs: **weight VRAM drops
   ~N-fold** (real, per-layer, certain). But the **activation** memory of the
   *currently-executing* layer still concentrates on **that layer's one device**
   — accelerate streams the hidden state to wherever the next layer lives and runs
   it there, one device hot at a time. Therefore:
   - **Weight-VRAM relief is guaranteed** and is what frees room on each card.
   - **The `batch_size` ceiling is set by peak per-layer activation memory on a
     single device**, which sharding relieves only indirectly (the freed weight
     VRAM on the hot layer's device is now available for activations). The
     *magnitude* of the batch_size gain is therefore **NOT pre-validated by this
     plan** — it depends on how much of the ~120 GB dual-model weight footprint
     leaves each card. **We commit to MEASURING batch headroom on the first
     sharded run before sizing lever-2 scope** (see §6 step 4); the plan claims
     only the certain part (weight-VRAM N-fold drop ⇒ *some* batch headroom),
     not a specific multiple.

   **This is the primary, lowest-risk win and is numerically identical to today
   (sharding relocates ops, it does not change the math; accelerate moves
   activations across the device boundary transparently — see §4).**

2. **Data-parallel-over-calibration lever (the throughput multiplier).**
   Replicate the (optionally sharded) teacher+student onto G *replica groups*,
   split the calibration tensor's batches across replicas, run the dual-forward
   independently per replica, and **sum the per-replica `B_acc`/`C_acc`/`A`
   Gram accumulators at the end.** Gram accumulation is a linear sum of
   per-batch `XᵀX` / `X_preᵀX_post` outer products — reducing across replicas is
   exactly `B = Σ_r B_r`, which is *order-tolerant* and *bit-meaningful* (same
   set of token outer-products, possibly summed in a different order). This is
   the highest-throughput option but only pays off when N ≥ 2× the per-model
   shard count (i.e. you have enough GPUs to hold ≥2 full model replicas). It is
   **gated behind the sharding lever** and is **phase 2** of the build.

**Why this split:** lever 1 is a correctness-neutral plumbing change that
unblocks the live H200 run immediately and needs zero new reduction code. Lever 2
is the real wall-time multiplier but introduces a cross-replica reduction that
must be proven bit-faithful against the golden tests, so it ships second behind
its own config flag.

**Rejected alternative — "model-placement split" (teacher on GPU-set A, student
on GPU-set B):** analyzed in §2.3. Rejected as the default because (a) it caps
useful parallelism at 2 and wastes GPUs beyond that, (b) the cross-cov matmul
`X_preᵀ @ X_post` then *always* straddles the A/B boundary forcing a per-batch
cross-device gather of the teacher hidden state (the code at
`covariance_collection.py:300-313` already does this move, but making it the
hot path is strictly worse than co-sharding), and (c) it does not generalize to
"any N". Balanced co-sharding (lever 1) + DP (lever 2) dominates it on every
axis. Kept documented because the existing `update_cross` device-coercion
machinery makes it *possible* as a fallback if a model ever can't co-shard.

---

## 1. Verified single-GPU blocks (file:line)

### B1 — `load_model` 4bit pin (NOT the general path) — `utils/model_io.py:98-99`
```
98:        if kwargs["device_map"] == "auto":
99:            kwargs["device_map"] = {"": 0}
```
This rewrite is **inside the `if load_in_4bit:` block** (opened at
`model_io.py:89`). For the **non-4bit** Stage 3 teacher load, `device_map="auto"`
(default at `:59`, passed through at `:84`) reaches transformers/accelerate
UNMODIFIED and **does shard** across all visible GPUs. So `load_model` is
*already N-GPU-capable for the teacher*; only the 4bit teacher is force-pinned.
→ Change: make the 4bit pin device-aware (mirror `router_kd/plugins/teacher.py:624-639`
which already does `{"": str(device)}` instead of `{"":0}`), but bnb 4bit cannot
itself shard — documented limitation, see §6.

### B2 — `load_compressed_model` single-device hard-pin — `utils/model_io.py:1406-1458`
```
1406:    if device_map not in ("auto", "cuda", "cpu"):
1407:        raise ValueError(... "use 'auto', 'cuda', or 'cpu'")
...
1411:    if device_map == "auto":
1413:        log.debug("device_map='auto': loading to GPU 0 if CUDA is available, else CPU (no multi-GPU balancing)")
...
1455:    target_device = (
1456:        torch.device("cuda") if torch.cuda.is_available() and device_map != "cpu"
1457:        else torch.device("cpu"))
```
The streaming loader resolves a **single** `target_device` and streams every
tensor to it (`model_io.py:1543` `safe_open(..., device=str(target_device))`,
`:1556` `_assign_storage(model, key, t)`, `:1618` `model.to(canonical_target)`).
Arbitrary `device_map` dicts are rejected at `:1406`. **This is the real
student-side block** — the Stage 3 student is ALWAYS loaded via this function
(`run_pipeline.py:527-532`, the `stage == 3` branch). → Change: §3.B2.

### B3 — orchestrator threads ONE `device` — `stage3/orchestrator.py:170,224,316-347`
```
run_pipeline.py:170:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
orchestrator.py:224:    run_ctx.set("device", device)
orchestrator.py:316-323 / 336-343:  teacher_model = load_model(..., device_map=config["model"]["device_map"], ...)
```
`device` is a non-indexed `cuda` (== `cuda:0` semantics for tensor placement).
The teacher load already passes `config["model"]["device_map"]` (so it shards),
but the single `device` is then used to place calibration batches
(`covariance_collection.py:404-405` `batch.to(device)`) and as the block_refine
target device. Under sharding, placing the *input batch* on `cuda:0` is correct —
accelerate's pre-hook on the embedding layer relocates as needed. → The single
`device` stays valid as the **input/entry device**; no change needed for lever 1.
For lever 2 the orchestrator gains a replica-spawn wrapper (§3.D).

### B4 — teacher residency assumes 1 GPU budget comment — `stage3/orchestrator.py:174-177`
Comment hard-codes the H200 single-GPU arithmetic (orig ~70 GB + pruned ~50 GB =
~120 GB on one card). Not a code block — but the `batch_size=4` ceiling it
documents is exactly what sharding removes. → Update comment + the
`bcov_batch_size` default story in §5.

### B5 — cov-collection dual-forward — `stage3/plugins/covariance_collection.py:402-413`
```
404:                if device is not None:
405:                    batch = batch.to(device)
...
408:                    if teacher_model is not None:
409:                        with torch.no_grad():
410:                            teacher_model(input_ids=batch)
412:                    with torch.no_grad():
413:                        model(input_ids=batch)
```
Single `device` for the batch; teacher then student forwarded sequentially on the
SAME batch. **Already cross-device-safe** for the cross-cov term:
`covariance_collection.py:300-313` coerces each teacher hidden-state row onto
`tensor.device` (the student tensor's device) before the `X_preᵀ@X_post` matmul,
and `InputCovarianceAccumulator.update_cross` (`activation_hooks.py:1051`) does
`cur.add_(cross_f32.to(device=cur.device))` so the FIRST writer fixes the
accumulator device and all later adds coerce. → Lever 1 needs **no change here**;
this is the load-bearing reason sharding is numerically transparent. Lever 2 adds
a post-loop cross-replica reduce (§3.C).

**M2 — why the Stage-3 student forward cannot intra-layer device-mismatch:**
the genuine un-coerced chain in the codebase is `wrapped_factored`'s
`F.linear(F.linear(sel, V), U)` (`activation_hooks.py:1474-1475, 1483-1484`) —
NO `.to()` between the two linears, so a `V`/`U` split across devices would fault.
But the Stage-3 student is loaded from `stage2p5_final`/`stage2_pruned`
(`run_pipeline.py:515-533`) which is **pruned but NOT yet factored** — Stage 3 is
what *produces* the factorization. `_resize_moe_stack_to_metadata` therefore does
**not** install `FactoredExperts` for the Stage-3 student; its experts are the
**fused stacked** params, whose forward (`wrapped_fused`, `:1491`) is a single
`F.linear` per matrix with no cross-device chain. Combined with the fused-param
constraint (H1) forcing `no_split_module_classes` to keep each layer whole, the
Stage-3 student cov pass **cannot** hit the un-coerced factored chain. The §3.B
out-of-scope guard refuses a multi-device map on an already-factored checkpoint to
keep this invariant explicit for Stage-4+ resumes (not covered here).

### B6 — accumulator device contract — `utils/activation_hooks.py:983-1052, 1054-1084`
`InputCovarianceAccumulator.update` (`:1012`) computes `XᵀX` on the input's
device; `update_cross` (`:1051`) coerces to `cur.device`; `finalize_layer`
(`:1069`) casts to `storage_dtype` and moves to CPU — **final covariance dict is
CPU-resident**. This means the *reduction surface* for lever 2 is already CPU
tensors keyed by `(layer, expert, matrix)` → a trivial key-wise sum. No GPU
collective needed.

### B7 — `instrument_experts` token_idx is BATCH-LOCAL — `utils/activation_hooks.py:1468-1471, 1505-1507`
```
1468:                top_k_pos, token_idx = torch.where(mask[e])
1470:                ctx = {... "token_idx": token_idx}
```
`token_idx` indexes into the *current batch's* flattened token axis, not a global
ID. The cross-cov keying `_teacher_hidden[li][tidx]`
(`covariance_collection.py:267-274, 311-313`) is therefore self-consistent ONLY
within one teacher+student forward of the SAME batch on the SAME replica. **This
is the invariant that makes lever-2 data-parallel safe:** each replica processes
disjoint batches with its own local `token_idx` space; cross-replica there is no
token-id collision because we never *share* `_teacher_hidden` across replicas —
we only sum the final per-(layer,expert) Gram matrices. → Lever 2 must keep
`_teacher_hidden` per-replica (it already is — it's a local dict in
`_collect_covariances`).

### B8 — block_refine cross-device loss risk — `stage3/plugins/block_refine.py:549-555`
```
549:    x_s = X_student[bi].to(device=device, ...)
550:    target = teacher_targets[bi].to(device=device, ...)
...
554:    loss = nn.functional.mse_loss(out.to(torch.float32), target.to(torch.float32))
```
`x_s`/`target` are explicitly placed on `device`, but `out` (the student block's
forward output) under accelerate sharding lands on **the last shard's device**,
which may differ from `device`. `mse_loss(out, target)` would then raise a
device-mismatch. Today this is latent (single GPU → all on `cuda:0`). → Lever 1
must add `out = out.to(device)` before the loss (§3.E). Low-risk one-liner;
numerically identical (a device copy, not a dtype/precision change).

### B9 — config `model.device_map` — `configs/*.yaml`
```
configs/qwen36_35b_a3b_30pct.yaml:14:  device_map: auto   # ... on H200 everything fits on 1 GPU
configs/qwen36_35b_a3b_reap_exact.yaml:28 / reap_faithful.yaml:39: same
```
Already `auto`. Consumed at `run_pipeline.py:496,505,529,568,592` and
`orchestrator.py:320,340`. → §5 extends the surface (no breaking change; `auto`
keeps working).

---

## 2. Design analysis — the four candidates

### 2.1 Balanced model-sharding (`device_map="auto"`/`"balanced"`/`max_memory`) — CHOSEN (lever 1)
- Each big model spreads across all N GPUs; accelerate inserts device-boundary
  hooks that relocate activations automatically. The dual-forward becomes
  VRAM-cheap per card → raise `batch_size`.
- **Numerically identical** to single-GPU: sharding changes *where* a matmul runs,
  not the matmul. The only cross-device event is the cross-cov teacher-row gather,
  already handled (B5).
- Cost: a per-boundary activation copy during forward (accelerate overhead),
  negligible vs the VRAM unblock.
- `"balanced"` vs `"auto"`: `"auto"` = balanced_low_0 (fills later GPUs first to
  leave GPU0 headroom for generation); `"balanced"` = even split. For two
  co-resident models we want **even** split → expose both, default keep `"auto"`
  for back-compat, recommend `"balanced"` in the 35B multi-GPU config. Explicit
  `max_memory={i: "Xgib"}` is the escape hatch for heterogeneous GPUs.

### 2.2 Data-parallel over calibration batches — CHOSEN (lever 2, phase 2)
- Highest throughput when GPUs ≥ 2×(shards-per-model). Split the
  `build_calibration_tensor` output into G contiguous batch-shards, one per
  replica group; each replica runs the full per-layer dual-forward over its shard;
  **sum the CPU-resident covariance dicts key-wise** at the end.
- Reduction correctness: `B = Σ_tokens xᵀx = Σ_replicas Σ_{tokens∈r} xᵀx`. Float
  addition is non-associative, so the *sum order* differs from single-GPU → result
  is **within fp tolerance, not bit-identical** (§4). The golden Stage-3 snapshot
  test runs at 1 GPU/1 replica → unaffected.
- Implementation surface is small *because the accumulator already drains to CPU*
  (B6): the reduce is `for k: B[k] = Σ_r B_r[k]` plus `token_count[k]` summed.
- Process model: torch `multiprocessing` spawn (one process per replica, each
  pinned to its GPU subset via `CUDA_VISIBLE_DEVICES`) is preferred over threads
  (avoids the GIL on the per-token Python loop at
  `covariance_collection.py:273,311` which is already flagged as the CPU
  bottleneck, MEDIUM-2). Replicas return their spilled per-layer cov files; the
  parent sums them. **Reuses the existing spill-to-disk path** (`spill_dir`,
  `ccov_spill_dir`) — each replica spills to its own subdir, parent sums the
  per-layer `.pt` files. This is the elegant move: no live IPC of 5 GB tensors,
  just disk handoff + a key-wise CPU sum that mirrors the existing resume merge.

### 2.3 Model-placement split (teacher set A / student set B) — REJECTED as default
- Caps at 2-way; wastes GPUs ≥3; forces the cross-cov gather onto the hot path
  every batch. Documented as a *possible* fallback only because `update_cross`'s
  device coercion already supports it. Not built.

### 2.4 Dynamic GPU count + graceful 1-GPU degrade — REQUIRED, threaded through both
- `n_gpu = torch.cuda.device_count()`. `n_gpu <= 1` → both levers collapse to
  today's exact path (lever 1: `device_map` resolves to single device; lever 2:
  G=1 → no spawn, no reduce). **No 1x/2x special-casing** — the replica count is
  `min(requested_replicas, n_gpu // shards_per_model)` and shards-per-model is
  whatever accelerate picks; G=1 is the natural floor.

---

## 3. Per-file change list

### 3.A `utils/model_io.py` — `load_model` (B1)
Make the 4bit device pin device-aware (non-4bit already shards):
- `load_model`'s signature (`model_io.py:54-63`) has **NO `device` parameter** —
  it only sees `device_map`. So the single-device hint **must be parsed from the
  incoming `device_map`** (L2): if `device_map` is a single-device dict
  (`{"": "cuda:1"}`) or a `"cuda:N"` string, reuse that device for the 4bit pin;
  otherwise fall back to `{"": 0}`. Replace the unconditional `{"": 0}` at `:99`
  with that parse. This mirrors `router_kd/plugins/teacher.py:624-639` (which
  resolves `{"": str(device)}` from a single-device map). bnb 4bit cannot shard
  (bnb limitation, documented; 4bit is not the Stage 3 cross-cov path).
- No change to the non-4bit path — it already honors `"auto"`/`"balanced"`/dict.

### 3.B `utils/model_io.py` — `load_compressed_model` (B2) — THE core change (non-trivial)
**Accurate description of today's loader (H2 correction):** the function does
**NOT** build a meta skeleton. It (a) `from_config` under `no_init_weights()`
on **CPU** (`model_io.py:1496-1497`), (b) `model.tie_weights()` (`:1498`),
(c) **resizes the MoE stack ON `target_device`** via
`_resize_moe_stack_to_metadata(model, meta, dtype=dtype, device=target_device)`
(`:1500`), then (d) streams each shard tensor to the single `target_device`
(`safe_open(..., device=str(target_device))` `:1543`, `_assign_storage` `:1556`),
and (e) `model.to(canonical_target)` for non-persistent buffers (`:1618`).

The multi-device change therefore has **two substeps of different risk**:

1. **Low-risk — per-key device routing (keeps resize-on-device, option (a) of
   H2).** Relax the guard at `:1406` to also accept `"balanced"`, a `dict`, and
   `max_memory`. Compute the per-module device map from the **existing module
   tree** (not a meta rebuild): call `accelerate.infer_auto_device_map(model,
   max_memory=..., no_split_module_classes=[decoder-layer class])` on the
   CPU-resident skeleton (or honor the explicit dict), yielding
   `module_path -> device`. Then in the stream loop, resolve each `key`'s target
   device by longest-prefix match against that map and pass it to
   `safe_open(..., device=<per-key device>)` + `_assign_storage`. **The
   one-tensor-peak streaming discipline is preserved per device.** Resize stays
   as-is but must target each stack's mapped device instead of one global
   `target_device` — `_resize_moe_stack_to_metadata` gains a per-layer device
   lookup (small, contained change).
   `no_split_module_classes` MUST include the Qwen3.5 decoder-layer class so
   accelerate keeps each layer (and its fused stacked expert params) intact on one
   device — this is mandatory given the fused-`nn.Parameter` constraint (H1).

2. **Non-trivial — accelerate hook dispatch.** After streaming, run
   `accelerate.dispatch_model(model, device_map=resolved_map)` so cross-layer
   forward relocates activations automatically (this is what
   `from_pretrained(device_map=...)` does internally). This replaces the single
   `model.to(canonical_target)` at `:1618` on the multi-device path. **This is a
   genuine new code path, not "the loop changes only in the device arg"** — it
   owns: re-establishing the `embed_tokens↔lm_head` tie **before** dispatch
   (the existing `tie_weights()` at `:1498` runs pre-resize; verify the tie
   survives resize+dispatch, see R1), and the per-device meta-leftover check
   (generalize `:1597-1610` to assert no `is_meta` param on ANY device). **It gets
   its own unit test** (`test_load_compressed_multidevice_routing`, §7) and the
   sharded-forward integration check. Budget it as a real substep, not a one-liner.

- **1-GPU / `"auto"`-on-1-GPU / `"cuda"` / `"cpu"` paths: UNCHANGED** (single
  `target_device`, original loop at `:1540-1564`, original `model.to` at `:1618`).
  The multi-device branch is entered only when the resolved map spans ≥2 devices.
- **Out-of-scope guard (M2):** if the checkpoint being loaded is **already
  factored** (`compressed_metadata.json` lists non-empty `factored_layers`) AND a
  multi-device map is requested, **log a loud warning and refuse / fall back to
  single-device**. Reason: an already-factored checkpoint installs
  `FactoredExperts` whose `wrapped_factored` forward chains
  `F.linear(F.linear(sel, V), U)` with **no `.to()` between the two linears**
  (`activation_hooks.py:1474-1475, 1483-1484`); if `V` and `U` landed on different
  devices that chain would device-mismatch. The Stage-3 student is NOT factored at
  load time (M2 below), so this guard never trips for the motivating run — but it
  protects a Stage-4+ multi-device resume that this plan does NOT cover.

### 3.C `stage3/plugins/covariance_collection.py` (B5) — lever-2 reduce only
- Lever 1: **no change** (cross-device already handled, B5).
- Lever 2: add a module-level `_reduce_spilled_cov_dirs(replica_dirs, out_dir)`
  that, per layer file, loads each replica's `layer_{idx}.pt`, sums the
  per-`(expert,matrix)` covariance tensors (fp32 accumulate → cast back to
  `storage_dtype`) and the `token_count`s, and writes the merged `layer_{idx}.pt`
  to `out_dir`. This mirrors `finalize_layer`'s merge math
  (`activation_hooks.py:1081-1084`) exactly. Pure CPU, deterministic given a fixed
  replica order.

### 3.D `stage3/orchestrator.py` (B3) — lever-2 spawn wrapper
- Add an opt-in branch around the `collect_covariances` dispatch
  (`orchestrator.py:365`): if `multi_gpu.cov_replicas > 1` and
  `device_count() >= replicas * shards_per_model`, fan out G child processes
  (torch.multiprocessing spawn), each with `CUDA_VISIBLE_DEVICES` set to its GPU
  subset, each running the existing `_collect_covariances` over its batch-shard
  with its own `spill_dir`/`ccov_spill_dir` subdir. Parent waits, then calls
  `_reduce_spilled_cov_dirs` (§3.C) into the canonical `bcov_spill_dir` /
  `ccov_spill_dir`, then proceeds to the factor phase reading the merged spill
  exactly as today.
- `replicas <= 1` → current in-process path verbatim (no spawn).
- Batch-shard split: slice the `calib` tensor (`orchestrator.py:163-167`) into G
  contiguous shards by sequence; each replica builds its own `iter_batches`. Equal
  token budget per replica (last replica takes remainder).

### 3.E `stage3/plugins/block_refine.py` (B8)
- One-liner: `out = out.to(device)` before `mse_loss` at `:554` (`out` is the
  student block forward output produced at `:553`; under sharding it lands on the
  last shard's device). The teacher side is already device-safe: the teacher
  block input `x_t` is placed on `device` at `:526` and the teacher target is
  spilled via `out.detach().to(dtype=torch.bfloat16, device="cpu")` at `:533`,
  then re-loaded to `device` at `:550`. Numerically a copy; enables a sharded
  student block output to reduce against the `device`-resident target. Guarded so
  it is a no-op when `out` is already on `device`.

### 3.F `run_pipeline.py` / config plumbing (B9, §5)
- Pass the (possibly multi-device) `device_map` through unchanged — it already
  flows to both load sites. Add reading of the new `multi_gpu` block (§5) and
  thread `cov_replicas` into the Stage 3 orchestrator call at `:305`.

---

## 4. Cross-device / cross-replica correctness + fp tolerance

**Lever 1 (sharding) — bit-identical claim.** Sharding relocates a matmul's
operands to another device but performs the *same* fp operations in the *same
order* (per-layer, per-batch accumulation order is unchanged — there is still one
sequential pass over batches per layer, `covariance_collection.py:403`). The only
new op is a device-to-device *copy* of the teacher hidden row
(`covariance_collection.py:313`) and of the cross term into the accumulator
(`activation_hooks.py:1051`) — copies are bit-preserving. **Therefore lever 1
covariances are bit-identical to single-GPU on the SAME hardware backend** within
the same dtype, and the Stage-3 golden snapshot
(`tests/test_stage3_golden_snapshot.py`) must remain unchanged when run on 1 GPU.
Tolerance budget, split by backend (consistent with §7/N1):
- **Same-arch 2-GPU** (`cuda:0` vs `cuda:1`, identical kernels): a pure relocation
  is bit-exact → **`atol=0`** for the same batch schedule. This is the strong
  claim, validated only in the multi-GPU integration check.
- **CPU-stand-in second device** (CI without a 2nd GPU): a CPU matmul and a CUDA
  matmul are NOT bit-identical → assert **`rtol=1e-5, atol=1e-6`**, NOT `atol=0`
  (that would falsely fail). This still proves device-relocation correctness.

**Lever 2 (data-parallel) — within-tolerance claim.** The math identity
`B = Σ_r Σ_{t∈shard_r} x_tᵀ x_t` is exact in ℝ; in fp the per-replica partial sums
are summed in a *different grouping* than the single-replica sequential sum, so
the result differs by floating-point reassociation only. Bound: the covariance is
accumulated in **fp32** (`activation_hooks.py:1009,1043,1069 cast to
storage_dtype only at finalize`) then stored at `storage_dtype` (bf16/fp16). The
reassociation error of summing K fp32 terms in a different tree is ≤ (K·ε_fp32)
relative ≈ 1e-5–1e-6 for K≈4000 sequences, **far below** the bf16/fp16 storage
quantization that both paths already incur. → **Equivalence tolerance for the
DP test: `rtol=1e-4, atol=1e-5` in fp32 before storage cast** (loosen to the
storage dtype's eps when comparing persisted artifacts). Token counts sum
exactly (integers). The golden snapshot test is NOT run under DP (it's a 1-replica
fixture) → never perturbed.

**Determinism knob:** fix the replica→shard assignment and the reduce order
(sorted replica dir order in `_reduce_spilled_cov_dirs`) so a given (N_gpu, G,
seed) is reproducible run-to-run.

---

## 5. Config surface

Extend `model:` and add an optional `multi_gpu:` block (all auto-detect by
default; absent block == today's behavior):
```yaml
model:
  device_map: auto        # auto | balanced | cpu | cuda | {explicit dict}
  # NEW (optional): max_memory for heterogeneous GPUs, passed to accelerate
  max_memory: null        # e.g. {0: "70GiB", 1: "70GiB"} or null

multi_gpu:                 # NEW, entirely optional
  enabled: auto            # auto -> derive from torch.cuda.device_count()
  shard_models: true       # lever 1: let device_map span all visible GPUs
  cov_replicas: 1          # lever 2: data-parallel replica groups for Stage 3 cov
                           #   1 = in-process (today). >1 = spawn DP.
                           #   effective = min(cov_replicas, n_gpu // shards_per_model)
```
- `multi_gpu` absent OR `n_gpu<=1` → behavior byte-identical to today.
- `shard_models: true` + `device_map: balanced` is the recommended 35B
  multi-GPU production setting (even split, max batch headroom).
- `cov_replicas` only affects Stage 3 covariance collection (the documented
  bottleneck); everything else inherits the sharded placement for free.

---

## 6. Build / implementation sequence (ordered)

1. **Lever 1 — `load_compressed_model` multi-device streaming** (§3.B). Land
   behind the existing `device_map` arg; add the multi-device branch, keep the
   single-device branch byte-identical. Unit-test the per-key device routing on a
   tiny 2-layer fixture with a forced 2-bucket device map under
   `CUDA_VISIBLE_DEVICES` mocking (§7).
2. **`load_model` 4bit device-aware pin** (§3.A) — small, mirrors existing
   router_kd logic.
3. **block_refine `out.to(device)`** (§3.E) — one-liner, unblocks sharded C.5.
4. **Config surface** (§5) + thread-through in `run_pipeline.py` (§3.F). At this
   point lever 1 is live: `device_map: balanced` shards both models, NO new
   reduction code. **First-sharded-run validation (H1):** on the live H200×N box,
   (a) confirm weight VRAM drops ~N-fold per card, (b) **MEASURE the achievable
   `stage3_svd.batch_size`** before committing lever-2 scope — the batch ceiling
   is activation-bound on the hot layer's device, so the gain is measured, not
   assumed — and (c) confirm covariances match 1-GPU (same-arch: `atol=0` on
   identical batch schedule, per §4).
5. **Lever 2 — `_reduce_spilled_cov_dirs`** (§3.C) + orchestrator spawn wrapper
   (§3.D). Land behind `cov_replicas > 1`. Prove DP-vs-1GPU equivalence test
   (§7) within the §4 tolerance.
6. **Docs/config**: update the H200 budget comment
   (`orchestrator.py:174-177`, B4) and ship a `*_multigpu.yaml` example.

Levers 1 and 4 are the immediate unblock; 5 is the throughput multiplier and can
follow after the first sharded run is validated.

---

## 7. Test plan

**Unit (no real multi-GPU box needed):**
- `test_load_compressed_multidevice_routing` — **scope-limited to placement, not
  intra-layer forward (M1).** Build a tiny config-only Qwen3.5-MoE skeleton
  (2 layers, 4 experts — reuse the smoke fixture from `tests/test_smoke_stage3.py`),
  force a 2-bucket map putting **layer 0** on `cuda:0` and **layer 1** on `cpu`
  (whole-layer buckets, honoring `no_split_module_classes` — NOT half-modules-of-a-
  layer, which is impossible for the fused stacked params, H1). Assert: (i) each
  param landed on its layer's mapped device, (ii) `dispatch_model` ran and a plain
  `model(input_ids=...)` forward completes (accelerate relocates the hidden state
  across the layer-0→layer-1 boundary). **This test passes trivially for the fused
  experts and intentionally does NOT claim to exercise intra-layer cross-device
  safety** — the fused experts keep each layer intact, so there is no
  intra-layer split to test. The genuine cross-device guard is
  `test_cov_sharding_equivalence` below (the cov-collection forward via
  `instrument_experts`, where the un-coerced `F.linear` chain would live IF the
  experts were factored — they are not, see M2).
- `test_reduce_spilled_cov_dirs`: synth 3 replica spill dirs with known
  per-`(layer,expert,matrix)` cov shards; assert the merged file equals the
  single-pass sum within fp32 `rtol=1e-6` and token_counts sum exactly. Pure CPU.
- `test_load_model_4bit_device_aware`: call with `device_map={"": "cuda:1"}` (a
  single-device dict) and assert the resolved 4bit `device_map` is `{"": "cuda:1"}`
  not `{"":0}` (inspect via a thin seam returning the resolved kwargs; the hint is
  parsed from `device_map`, §3.A/L2 — there is no `device` param to read).

**Equivalence (the load-bearing ones):**
- `test_cov_sharding_equivalence` (lever 1) — **the real cross-device guard.** Run
  `_collect_covariances` twice over the SAME batch schedule on a tiny model — once
  all on one device, once with the model's two layers split across two devices via
  a forced device map — and compare `B_acc.covariance` / `C_acc.covariance`. This
  drives the cov-collection forward through `instrument_experts`
  (`activation_hooks.py:1474-1485`), exercising the teacher-row device coercion
  (`covariance_collection.py:310-313`) and the accumulator coercion
  (`activation_hooks.py:1051`). **Tolerance is split by device backend (N1):**
  - **CPU-stand-in variant (runs on 1 GPU or 0 GPU): `rtol=1e-5, atol=1e-6`** —
    a CPU matmul vs a CUDA matmul is NOT bit-identical, so `atol=0` would
    falsely FAIL. This variant proves *device-relocation correctness*, within fp
    tolerance.
  - **True same-arch 2-GPU variant (integration, requires 2 physical GPUs):
    `atol=0` bit-identical** — `cuda:0` and `cuda:1` run the same kernels, so a
    pure relocation IS bit-exact. This is the strong claim from §4; it runs only
    in the multi-GPU integration check, not in CI.
- `test_cov_dp_equivalence` (lever 2): run the in-process single-replica path over
  the full tiny calibration tensor, then run the DP path with G=2 (two batch
  shards reduced via `_reduce_spilled_cov_dirs`) and assert covariances match
  within fp32 `rtol=1e-4, atol=1e-5` (§4). Disk handoff so it runs without 2 GPUs
  (spawn 2 CPU "replicas").

**Without a real multi-GPU box:** all CI tests use either (a) `CUDA_VISIBLE_DEVICES`
to mask, (b) CPU as a stand-in second device in a forced device map (with the
`rtol` tolerance of N1, never `atol=0`), or (c) disk-based replica handoff with CPU
replicas. The real 2×/4×/8× H200 run is the integration check (same covariances —
the `atol=0` same-arch variant — plus measured weight-VRAM relief and wall-time
drop, and the batch-headroom measurement of H1/§6 step 4).

**Regression guards (must stay green, unchanged):**
- `tests/test_stage3_golden_snapshot.py`, `test_stage3_cross_cov.py`,
  `test_stage3_plugin_covariance.py`, `test_stage3_spill.py`,
  `test_aa_svd_correctness.py`, `test_smoke_stage3.py`,
  `test_run_pipeline_reap_exact.py` (uses `device_map: cpu`),
  `test_input_cov_offload_streaming.py`. These exercise the 1-GPU / single-device
  path which §3 keeps byte-identical.

---

## 8. Risks + scope boundaries

**In scope (this plan):** Stage 3 AA-SVD covariance collection (lever 1 sharding +
lever 2 DP), `load_compressed_model` multi-device load, `load_model` 4bit pin,
block_refine cross-device loss fix, config surface.

**Out of scope (noted, not built here):**
- **Stage 4 EoRA** (`stage4/orchestrator.py`): operates on already-factored
  `FactoredExperts` and inherits `fe.gate_proj_U.device` (`stage4/orchestrator.py:156`)
  — it works *correctly* under sharding for free (each expert's residual computed
  on that expert's shard) but does NOT need DP (it's not the bottleneck). Benefits
  from lever 1 automatically; recommend a follow-up only if EoRA wall-time
  becomes the long pole.
- **Stage 5 Router-KD** (`stage5_router_kd.py`, `router_kd/plugins/teacher.py`):
  already partly multi-GPU-aware (`teacher.py:624-639` co-locates the 4bit teacher
  with the student device). Training (backprop) under model-parallel sharding is a
  *different* problem (pipeline/tensor parallel, not the inference-only Gram
  collection here) — explicitly OUT, flagged as its own future epic.
- **Stage 1/2/6**: Stage 2 faithful_prune is already GPU-optimal and device-
  inherited (see `tasks/REAP_PERF_GPU_FINDINGS.md` — index_select runs on the
  layer's resident device, NO-OP on H200). Stage 6 eval inherits placement. None
  are the bottleneck; lever 1 helps them passively, no dedicated work.

**Risks:**
- R1 — `accelerate.dispatch_model` on the streamed skeleton must reproduce the tie
  (`embed_tokens ↔ lm_head`) that `load_compressed_model` re-establishes at
  `:1498 model.tie_weights()`. Mitigation: call `tie_weights()` BEFORE
  `dispatch_model`, and assert no meta leftovers (the existing check at
  `:1597-1610`) per device.
- R2 — bnb 4bit cannot shard (bitsandbytes limitation). Documented; the Stage 3
  cross-cov teacher is loaded in BF16 (not 4bit) so this does not block the
  motivating run. A 4bit teacher on a tiny box stays single-device (degraded but
  correct).
- R3 — DP fp reassociation (§4): bounded well under storage quantization;
  mitigated by fp32 accumulate + fixed reduce order + the tolerance-not-bit-exact
  test. Golden snapshot is 1-replica → never perturbed.
- R4 — spawn overhead / CUDA-context-per-process: amortized over a ~16h run;
  spawn (not fork) required for CUDA. Each replica re-loads the model (~60s) —
  negligible vs the per-layer pass it parallelizes.
- R5 — heterogeneous GPUs: covered by explicit `max_memory` in the config (§5).

**Boundary invariant:** every change is gated so that `torch.cuda.device_count()
<= 1` (or `multi_gpu` absent) reproduces the current code path with no behavioral
delta — the live single-GPU run and all golden tests are untouched.
