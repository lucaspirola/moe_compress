# PLAN — N-GPU multi-device support for Stage 4 (EoRA residual compensation)

Branch: `plan/multigpu-stage4` off `main` @ `0a59ff7`.
Scope owner: Stage 4 EoRA compensation — the per-(layer, expert, matrix)
weight-space linear algebra and the model placement it inherits.
Author: planner pass, code NOT written here.

All file:line citations verified by Read/grep against the working tree at
`0a59ff7` on 2026-06-11. The Stage-3 multi-GPU feature it builds beside
landed at the same SHA (`tasks/PLAN_MULTIGPU_STAGE3.md`,
`stage3/orchestrator.py`, `stage3/plugins/covariance_collection.py`).

---

## 0. TL;DR — the chosen design and why it differs from Stage 3

**Stage 4's compute profile is fundamentally different from Stage 3's, so the
parallelization axis is different.**

| | Stage 3 (cov collection) | Stage 4 (EoRA) |
|---|---|---|
| Dominant work | **forward passes** (dual teacher+student calibration) | **weight-space linear algebra** (`eigh`/Gram-SVD per (layer, expert, matrix)) |
| Inputs | streamed calibration batches | CPU-resident `A_cov` dict + `originals` dict (both `map_location="cpu"`) |
| Models resident | **two** (teacher BF16 + student) → VRAM-bound | **one** (the factored student) → **NOT VRAM-bound** |
| Natural parallel axis | **data-parallel** over calibration batches | **task-parallel** over (layer, expert) tiles |
| Reduction | sum Gram accumulators (`B = Σ_r B_r`) | **none** — disjoint output tiles, concatenate/place |
| Single-GPU block | VRAM ceiling on batch_size | **wall-time**: serial `eigh`+SVD over all (layer×expert×matrix) tiles on one device |

**Therefore Stage 4 does NOT reuse Stage 3's data-parallel lever.** EoRA has no
calibration forward to split and no cross-replica Gram sum. The work is an
embarrassingly-parallel set of independent per-(layer, expert) factor solves
that today run serially on one device (§1). The N-GPU design is a
**task-parallel shard of the (layer, expert) work across N devices**, gathering
the resulting `U_corr`/`V_corr` correction tiles back to the layer's home
device for the in-place `fe.widen_rank` (§2).

**Two design levers, both auto-detected from `torch.cuda.device_count()`, both
no-ops at 1 GPU:**

1. **Task-parallel EoRA lever (the primary, the wall-time win).** Distribute the
   per-(layer, expert, matrix) `_compute_eora_factors` calls across N CUDA
   devices. Each device computes a disjoint subset of experts' correction tiles
   for the current layer from the **CPU-resident** `A_cov`/`originals` (which
   every device can read independently — no model weights needed for the SOLVE,
   only `W_orig`, `U_e`, `V_e`, and `A`). The parent gathers the per-expert
   `(U_corr[e], V_corr[e], take_eff[e])` tiles to the layer's home device and
   calls `fe.widen_rank` exactly as today. **This is the only lever that moves
   wall-time.** Gated behind `multi_gpu.eora_workers > 1`; phase-2-style opt-in.

2. **Model-placement lever (free, inherited, NOT the bottleneck).** Today
   Stage 4 holds ONE factored model on a single device (§1.B2). Sharding it
   across N GPUs via accelerate is **possible but blocked by the M2 guard**
   (§1.M2) AND **unnecessary** — Stage 4 is not VRAM-bound (one ~50 GB model on
   an 80 GB card). The plan **does NOT shard the Stage 4 model**; instead the
   factored experts stay on their single home device, and the task-parallel
   lever ships each tile's *inputs* (CPU tensors) to a worker device and ships
   the *result* tile back. **This sidesteps the M2 guard entirely** — the
   factored model is never placed multi-device, so the un-coerced
   `F.linear(F.linear(sel, V), U)` chain (`activation_hooks.py:1474-1485`) never
   straddles a device boundary. M2 is preserved untouched (§2.3).

**Why lever 1 is task-parallel, not data-parallel (the core insight):** the EoRA
solve for expert `e`, matrix `name` reads ONLY
`originals[(li, e, name)]`, `fe.{name}_U[e]`, `fe.{name}_V[e]`, and
`A_cov[(li, e, gate_or_self)]` — **nothing from any other expert**
(`eora_compensation.py:522-557`). The only intra-loop shared state is the
gate→up spectrum memo `gate_spectra` (`eora_compensation.py:494,541-545`), which
is **per-expert keyed** (`gate_spectra[e]`) and so shards cleanly with the
expert. There is **no cross-expert reduction** — `U_corr`/`V_corr` are
pre-allocated `[N, ...]` tensors written at disjoint expert rows `U_corr[e]=Uc`
(`:552-553`). Splitting the `for e in range(N)` loop across devices and gathering
the rows is **numerically identical to the serial loop** (the rows are
independent; §4).

---

## 1. Verified single-GPU blocks (file:line)

### B1 — Stage 4 holds ONE model on ONE device — `run_pipeline.py:314-317` + `_load_for_stage` fall-through `:581-600`
```
314:    if start <= 4 <= stop and not _skip_intermediate:
315:        log.info("=== Stage 4 — EoRA ===")
317:        stage4_eora.run(model, tokenizer, config, artifacts_dir, no_resume=args.no_resume)
```
Two entry paths for the Stage 4 model:
- **Full-pipeline path** (`start ≤ 3 ≤ stop`): Stage 4 receives the **in-memory**
  model object straight from `stage3_svd.run` (`run_pipeline.py:305`), already
  factored, already resident on the single `device` (`run_pipeline.py:170`).
- **Resume-at-4 path**: `_load_for_stage(4, ...)` hits the generic fall-through
  (`run_pipeline.py:581-600`, since stage 4 is not 2/3/6), loading
  `STAGE_REGISTRY[4][1] == "stage3_svd"` (`run_pipeline.py:70`) via
  `load_compressed_model` (`:593-599`) — **a FACTORED checkpoint** →
  **trips the M2 guard** (§1.M2) → single-device fallback today.

Either way the Stage 4 model is **single-device**. The experts inherit that
device: `dev = fe.gate_proj_U.device` (`stage4/orchestrator.py:156`,
`eora_compensation.py:479`).

### B2 — EoRA compute is serial per (layer, expert, matrix) — `stage4/orchestrator.py:152-198` + `eora_compensation.py:496-560`
The orchestrator loops layers with a **plain `for`** (`orchestrator.py:152`):
```
152:    for k, ref in enumerate(layers):
156:        dev = fe.gate_proj_U.device
198:        walk_phases(("compensate_layer",), plugins, run_ctx)
```
Each `compensate_layer` call loops matrices then experts **serially on `dev`**:
```
eora_compensation.py:496:        for name in MATRIX_NAMES:        # gate_proj, up_proj, down_proj
:522:            for e in range(N):                              # N experts, SERIAL
:544:                spectrum = _eigh_spectrum(A, d_in, dev, ...) # eigh on dev
:548:                Uc, Vc, take_eff = _compute_eora_factors(delta, A, ..., dev, spectrum=...)
```
`_compute_eora_factors` (`eora_compensation.py:235-373`) runs, **per expert**:
a `delta @ Q'` projection (`:322`), a Gram-side `eigh` (`:342` or `:350`), and
back-projection matmuls (`:357,:363`). `_eigh_spectrum` (`:182-232`) runs a
**full `torch.linalg.eigh(A)` on the `[d_in, d_in]` covariance** (`:215`) once
per (layer, expert) for gate_proj, reused for up_proj (Lever A memo).

**This is the wall-time bottleneck:** `N_layers × N_experts × {2 eigh (cov +
Gram) for gate/up, 1 for down} × matmuls`, all on ONE device, fully serial. For
Qwen3.6 (~48 MoE layers × ~128 experts) that is ~6000 expert-solves × 3 matrices
serialized on a single GPU. **No VRAM pressure** (each solve touches one
expert's `[d_out, d_in]` + one `[d_in, d_in]` cov), so the single device sits
compute-serial-bound, not memory-bound. → This is the unit lever 1 parallelizes.

### B3 — inputs are CPU-resident (the enabler) — `eora_inputs.py:217,296` + `input_cov_cache.py:84`
```
eora_inputs.py:217:  A_cov = torch.load(A_cov_path, map_location="cpu").get("covariance", {})
eora_inputs.py:296:  originals: dict = torch.load(originals_path, map_location="cpu")
input_cov_cache.py:84: ctx.set("A_cov", payload.sigma_in)   # load_covariance → CPU tensors
```
Both `A_cov` (keyed `(layer, expert, matrix) → Tensor[d_in,d_in]`) and
`originals` (keyed `(layer, expert, matrix) → W_orig`) are **CPU dicts**. The
per-expert solve copies the needed slice to `dev` on demand
(`eora_compensation.py:527 W_orig.to(device=dev, ...)`, `:207 A.to(device=dev,
...)`). **This is why task-parallel works with zero model sharding:** any worker
device can pull its experts' CPU input slices independently — there is no need
to place the factored model on multiple devices at all.

### B4 — the per-expert write surface is disjoint — `eora_compensation.py:510-557`
```
510:            U_corr = torch.zeros(N, d_out, r_per_expert, dtype=dtype, device=dev)
511:            V_corr = torch.zeros(N, r_per_expert, d_in, dtype=dtype, device=dev)
552:                U_corr[e] = Uc.to(dtype)        # disjoint row e
553:                V_corr[e] = Vc.to(dtype)
554:                eff_per_expert[e] = int(take_eff)
```
The pre-allocated `[N, ...]` tensors are written at **disjoint expert rows**.
There is **no read-after-write across experts** and **no accumulation** — the
only post-loop reductions are `res_before_acc`/`res_after_acc`
(`:519-520,535,557`) which feed **only log/trackio, never the golden**
(`:518` "These feed ONLY log.info / trackio — never the golden"). → The expert
loop is data-parallel-trivial; gather is a row-place, not a sum (§4).

### M2 — the factored-model multi-device guard (the CRUX) — `utils/model_io.py:1647-1666`
```
1655:    if multi_device and meta.get("factored_layers"):
1656:        log.warning("load_compressed_model: checkpoint ... is already factored ...
1659:                    "a multi-device map could split a factored expert's V/U ...")
1663:        multi_device = False                      # ← refuse → single-device fallback
```
The Stage-3 multi-device loader **refuses** a multi-device map on an
already-factored checkpoint, because `FactoredExperts.wrapped_factored` chains
two linears with **no `.to()` between them**
(`activation_hooks.py:1474-1475, 1483-1484`):
```
1474:            tmp = torch.nn.functional.linear(sel, V_e)     # V on its device
1475:            out = torch.nn.functional.linear(tmp, U_e)     # U must match — no .to()
```
Stage 4's resume-at-4 model is loaded from `stage3_svd/` which **is factored**
(`run_pipeline.py:593`, `STAGE_REGISTRY[4][1]=="stage3_svd"`), so a naive
"shard the Stage 4 model" approach would **trip M2 → fall back to single
device** anyway. **The chosen design does not fight this** — it leaves the model
single-device and parallelizes the *math*, not the *placement* (§2.3). M2 is
neither modified nor circumvented; it stays a correct guard.

### B5 — orchestrator is the dispatch point — `stage4/orchestrator.py:148-198`
The plain per-layer loop (`:152`) dispatches `compensate_layer` against the ROOT
ctx (`:198`). The resume-from-spill branch (`:162-192`) `continue`s past
already-done layers. **Lever 1 wraps the per-expert loop *inside*
`compensate_layer`, not the layer loop** — keeping the resume branch, the
`rank_map`/`compensated_params` root accumulation (`:184-189`), and the spill
(`eora_compensation.py:622-623`) byte-identical. The layer loop stays serial
(layers are cheap relative to the per-expert eigh work and share the layer's
`fe`); experts within a layer fan out.

### B6 — golden + smoke tests that MUST stay green — `max_quality/tests/`
- `test_stage4_golden_snapshot.py` — byte-identical `eora_ranks.json` on the
  `tiny_model` CPU fixture (regen/verify same-machine caveat documented in its
  header). **1-worker path must reproduce these bytes exactly.**
- `test_stage4_eora_opt_levers.py` — pins Lever A/B/C equivalence (the existing
  eigh-reuse / deferred-sync / Gram-SVD opts). Lever 1 must not perturb these.
- `test_smoke_stage4_resume.py`, `test_stage4_orchestrator.py`,
  `test_stage4_plugin_compensation.py`, `test_stage4_plugin_inputs.py`,
  `test_stage4_input_cov_cache.py`, `test_stage4_stage.py`,
  `test_stage4_scaffold.py`.

### B7 — config knob today — `config["stage4_eora"]`
`eora_compensation.py:475` reads `s4 = config["stage4_eora"]`; knobs used:
`compensation_budget_pct` (`:503`), `eigenspace_rank_cap` (`:505`). No
`multi_gpu` block consumed by Stage 4 today. → §5 adds an OPTIONAL
`multi_gpu.eora_workers`.

---

## 2. Design analysis — the candidates

### 2.1 Task-parallel per-(layer, expert) EoRA across N devices — CHOSEN (lever 1)
**Mechanism.** For the current layer's matrix `name`, partition the `N` experts
into `W = min(eora_workers, n_gpu, N)` contiguous device-buckets. Each worker
device `d` owns expert subset `E_d`. A worker, for each `e ∈ E_d`:
- pulls `originals[(li,e,name)]`, `fe.{name}_U[e]`, `fe.{name}_V[e]`,
  `A_cov[cov_key]` (CPU tensors) and moves them to device `d`;
- runs the **identical** `_eigh_spectrum` + `_compute_eora_factors` math on `d`
  (gate→up spectrum memo stays per-expert, lives entirely within `d`);
- returns `(Uc, Vc, take_eff)` for its experts.

The parent gathers all `(e → Uc, Vc, take_eff)` to the **layer home device**
`dev` and writes the pre-allocated `U_corr[e]`/`V_corr[e]` rows exactly as the
serial code does today (`eora_compensation.py:552-553`), then runs the unchanged
`fe.widen_rank` (`:571`) and spill (`:623`).

**Why numerically identical (§4):** the per-expert solve is a pure function of
that expert's inputs; rows are independent; the gather is a row-place not a sum;
the gate→up memo is per-expert and never crosses workers. On the **same GPU
arch** (`cuda:0`==`cuda:1` kernels) the per-expert result is **bit-identical** to
the serial run; the order of *writing* rows does not affect any row's value.

**Cost.** Each expert solve ships ~`(d_out·d_in + d_in²)` fp32 to a device and
~`(d_out·r + r·d_in)` back — small vs the eigh compute it parallelizes. Worker
spin-up amortized over ~6000 solves.

**Process model — two sub-options, pick by measurement (§6 step 3):**
- **(a) In-process multi-stream / multi-device dispatch (PREFERRED default).**
  The parent process owns all N CUDA contexts; it issues each expert's solve on
  its assigned device via an explicit `torch.cuda.device(d)` context and
  collects results. Because the heavy ops (`eigh`, SVD-Gram) release the GIL
  during the CUDA kernel, round-robining experts across devices overlaps the
  kernels. **No spawn, no pickling, no model reload** — the model is already
  in-memory (B1 full-pipeline path). Simplest; preserves the in-memory handoff.
  Risk: the Python per-token/per-expert glue (`:522-560`) is GIL-bound; if launch
  latency dominates the kernel time the overlap is poor → fall to (b).
- **(b) `torch.multiprocessing` spawn workers (fallback, mirrors Stage 3).**
  Reuse the **exact pattern** from
  `stage3/plugins/covariance_collection.py:580-643` (`_cov_replica_worker`):
  module-level picklable worker, `CUDA_VISIBLE_DEVICES` pin, disk handoff of
  result tiles. Each worker owns a layer-band or expert-band and writes its
  correction tiles to a per-worker spill dir; the parent loads + places them.
  Needed only if (a)'s GIL launch latency is the wall. Heavier (the model must be
  re-materialized per worker on the resume-at-4 path, or the workers operate
  purely on the CPU `originals`/`A_cov` + the per-expert `U_e`/`V_e` slices
  passed by value — the latter avoids any model reload since the SOLVE needs only
  tensors, not the nn.Module).

**Decision:** ship **(a)** behind `eora_workers>1`; keep (b) documented as the
escape hatch if step-3 timing shows GIL-bound launch. Both are gated; `==1`
reproduces today's serial path byte-for-byte.

### 2.2 Shard the Stage 4 model across N GPUs — REJECTED
Would require either defeating the M2 guard (correctness hazard — the un-coerced
factored `F.linear` chain, §1.M2) or accepting the M2 single-device fallback
(no-op). And it buys **nothing**: Stage 4 is not VRAM-bound (one ~50 GB model).
Sharding would also force every per-expert solve to read `fe.{name}_U[e]` /
`fe.{name}_V[e]` from whichever shard owns that layer — but those are exactly the
tensors lever 1 already moves to the worker device on demand. Model sharding adds
the M2 hazard for zero benefit. **Not built.**

### 2.3 How the chosen design handles the factored-model / M2-guard issue
**It never places the factored model multi-device.** The model stays on its
single home device (`dev`). Lever 1 ships per-expert *tensor slices*
(`U_e`,`V_e`,`W_orig`,`A`) — not nn.Modules, not the factored expert object — to
worker devices, runs the pure-function solve there, and ships *result tiles*
back. The `FactoredExperts` object and its `wrapped_factored` forward are never
exercised across a device boundary (Stage 4 does no forward pass at all — it is
weight-space algebra, §0). **Therefore the M2 guard is irrelevant to lever 1 and
is left exactly as-is.** On the resume-at-4 path, `load_compressed_model` still
hits M2 and loads single-device (correct, unchanged); lever 1 then parallelizes
the math off that single-device model.

### 2.4 Dynamic GPU count + graceful 1-GPU degrade — REQUIRED
`n_gpu = torch.cuda.device_count()`. `effective_workers = min(eora_workers,
n_gpu, N_experts)`. `effective_workers <= 1` → the **today** serial in-process
loop verbatim (no device fan-out, no gather indirection). No 1x/2x special-case;
`W=1` is the natural floor. Any N is `min(...)`-clamped.

---

## 3. Per-file change list

### 3.A `stage4/plugins/eora_compensation.py` — the core change (lever 1)
The per-expert loop (`:522-560`) is refactored into a **device-parallel map** over
experts, behind a worker-count gate. Concretely:

1. **Extract a pure per-expert solve helper** (new module-level function):
   `_solve_expert_tile(name, e, li, W_orig_cpu, U_e_cpu, V_e_cpu, A_cpu,
   r_per_expert, target_device, a_storage_dtype, gate_spectrum=None) ->
   (Uc, Vc, take_eff, gate_spectrum_out)`. Body is **verbatim** the current
   `:527-557` inner block (delta, spectrum select, `_compute_eora_factors`), just
   parameterized by `target_device` instead of the closure `dev`. The gate→up
   memo becomes an explicit in/out arg so it stays per-expert and crosses no
   worker boundary. **This helper is the unit each worker device runs.**
2. **Dispatch branch in `compensate_layer`:**
   - `effective_workers <= 1` (or `multi_gpu` absent): call `_solve_expert_tile`
     in the **same serial order** with `target_device=dev` and the existing
     `gate_spectra[e]` memo — **byte-identical to today** (the helper is the same
     statements). The golden path is this branch.
   - `effective_workers > 1`: assign experts round-robin/contiguous to the
     `W` worker devices; run each expert's `_solve_expert_tile` on its device
     (in-process option 2.1a: `with torch.cuda.device(d):`); gather
     `(Uc, Vc, take_eff)` per expert to `dev`; write `U_corr[e]`,`V_corr[e]`,
     `eff_per_expert[e]` in **ascending e order** (deterministic gather, §4).
3. **`res_before_acc`/`res_after_acc`** (log-only, B4): accumulate per-expert
   contributions returned alongside the tile (or recompute on `dev` after gather
   — they never touch the golden, so either is fine; recompute-on-`dev` keeps the
   worker return payload minimal). Single `.item()` sync per matrix preserved
   (Lever B, `:575-576`).
4. **`fe.widen_rank`, trackio emit, spill** (`:562-623`): **unchanged** — they run
   on `dev` after the gather, identical to today.

`_eigh_spectrum`, `_compute_eora_factors`, `_spill_layer`: **unchanged bodies**
(lever 1 calls them with a `target_device` arg they already accept via the
`device` parameter).

### 3.B `stage4/orchestrator.py` — wiring + worker resolution
- Add a small `_resolve_eora_workers(config) -> int` mirroring
  `stage3/orchestrator.py:79-99` `_resolve_cov_replicas`: read
  `config.get("multi_gpu", {}).get("eora_workers", 1)`, clamp to
  `min(requested, torch.cuda.device_count() or 1)`. Thread the result onto
  `run_ctx` (`run_ctx.set("eora_workers", W)`) **before** the per-layer loop
  (`:148`) so `compensate_layer` reads it off the ctx.
- The per-layer loop (`:152-198`), the resume branch (`:162-192`), and finalize
  (`:200-241`) are **unchanged**.
- Add `"eora_workers"` to `EoraCompensationPlugin.reads`
  (`eora_compensation.py:411-414`).

### 3.C `run_pipeline.py` — no functional change for lever 1
Stage 4 already receives the in-memory model (full-pipeline) or single-device
load (resume, via M2 fallback). The new `multi_gpu` block (§5) is read inside the
Stage 4 orchestrator, not here. **No change to `:314-317` or `_load_for_stage`.**
(If a future epic wants the resume-at-4 model itself sharded, that is the M2
follow-up — explicitly out of scope, §8.)

### 3.D (option-b only, if step-3 timing demands it) spawn driver
Mirror `stage3/plugins/covariance_collection.py:554-643`: a module-level
`_eora_layer_worker(worker_idx, visible_devices, expert_band, cpu_input_refs,
out_dir, ...)` picklable target that pins `CUDA_VISIBLE_DEVICES`, solves its
expert band on the CPU inputs, and spills result tiles to `out_dir`; a parent
`run_eora_layer_parallel(...)` that spawns, waits, loads tiles, places rows.
**Only built if 2.1a is GIL-bound (measured).** Reuses Stage 3's spawn/disk
plumbing wholesale.

---

## 4. Correctness / equivalence argument + fp tolerance

**Lever 1 — the per-expert solve is a pure function of disjoint inputs.**
`_compute_eora_factors(delta_e, A_e, r, device)` reads only expert `e`'s tensors
(`eora_compensation.py:527-548`) and writes only row `e`
(`:552-554`). No expert reads another expert's result; the only post-loop
reductions are log-only norms (B4). Therefore:

- **Bit-identical on same arch.** `cuda:0` and `cuda:1` (or `cuda:d`) run the
  **same kernels** for `eigh`/matmul; relocating expert `e`'s solve to `cuda:d`
  performs the *same fp ops in the same per-expert order*. The gather writes rows
  in **ascending e order** (deterministic), so `U_corr`/`V_corr` are
  byte-identical to the serial fill. → **same-arch N-GPU `eora_ranks.json` is
  byte-identical to 1-GPU** (the golden is integer ranks + param counts anyway —
  `test_stage4_golden_snapshot.py` header — so even fp-level drift in factor
  *values* would not move the golden; but on same-arch there is none).
- **Gate→up spectrum memo invariance.** The memo is per-expert
  (`gate_spectra[e]`) and travels with expert `e` to its worker device; up_proj
  reuses gate's spectrum **on the same device** — bit-identical to recompute
  (the existing Lever-A contract, `eora_compensation.py:280-282`). No worker
  shares a memo with another worker.
- **Cross-arch / CPU-stand-in tolerance.** When a worker "device" is CPU (CI
  stand-in) or a different GPU arch, `eigh`/SVD are NOT bit-identical to the home
  device. Per-expert factor values then differ by floating-point only. Bound:
  the solve is fp32 throughout (`eora_compensation.py:287,207`), cast to storage
  `dtype` only at `U_corr[e]=Uc.to(dtype)` (`:552`). The eigenspace noise-floor
  truncation (`_eigh_spectrum` `:222`) could in principle change `n_keep` (and
  thus `take_eff`, an integer the golden DOES record via `rank_map`) if an
  eigenvalue sits exactly on the `thresh` boundary on one backend but not the
  other — a measure-zero event, but the reason the **CI equivalence test asserts
  on the same backend** (CPU vs CPU) for the integer ranks, and uses
  `rtol=1e-5, atol=1e-6` for the **float factor values**, never `atol=0` across
  backends. The **same-arch 2-GPU integration check** asserts `atol=0` and
  integer-rank-identical.

**No reduction → no reassociation error.** Unlike Stage 3's DP (which sums Gram
partials in a different tree, §4 of the Stage-3 plan), Stage 4's gather is a
**placement**, not a sum. There is no float-addition reassociation anywhere in
lever 1. This is strictly safer than Stage 3's DP lever.

**Determinism knob:** fix the expert→worker assignment (contiguous bands by
sorted expert index) and the gather write order (ascending `e`) so a given
`(n_gpu, eora_workers, layer)` is reproducible run-to-run and identical to the
serial order's per-row values.

---

## 5. Config surface

Extend with an OPTIONAL `multi_gpu.eora_workers` (absent ⇒ today's serial
behavior):
```yaml
multi_gpu:                  # OPTIONAL block (shared with Stage 3's keys)
  eora_workers: 1           # lever 1: task-parallel EoRA worker devices.
                            #   1 = serial in-process (today, golden path).
                            #   >1 = fan per-(layer,expert) solves across devices.
                            #   effective = min(eora_workers, cuda.device_count(), N_experts)
```
- `multi_gpu` absent OR `n_gpu<=1` OR `eora_workers<=1` → byte-identical to today.
- Coexists with Stage 3's `cov_replicas`/`shard_models` in the same `multi_gpu`
  block; Stage 4 reads only `eora_workers`, Stage 3 reads only its keys. No
  collision (different keys, different stages).
- Stage 4 does NOT add a `device_map`/`max_memory` surface — it does not shard
  the model (§2.2), so the model-placement knobs are irrelevant here.

---

## 6. Build / implementation sequence (ordered)

1. **Extract `_solve_expert_tile`** (§3.A.1) as a pure helper, refactor the
   serial loop to call it with `target_device=dev`. **No behavior change** — run
   the full Stage 4 test suite (B6) and confirm `test_stage4_golden_snapshot.py`
   + `test_stage4_eora_opt_levers.py` still byte-/value-match. This is a pure
   refactor commit, golden-gated, before any parallelism.
2. **Worker resolution** (§3.B): `_resolve_eora_workers`, thread `eora_workers`
   onto ctx, add to `reads`. Still serial (`W=1` default). Tests green.
3. **In-process device fan-out (2.1a)** behind `eora_workers>1` (§3.A.2). Add the
   gather + ascending-e row-place. **MEASURE** on the live H200×N box: per-layer
   wall-time at W=1 vs W=2/4/8, and the same-arch `atol=0` equivalence of
   `eora_ranks.json` + a sampled factor tile. If launch latency dominates (poor
   overlap), proceed to step 4; else stop here.
4. **(conditional) spawn fallback (2.1b / §3.D)** — only if step-3 timing shows
   GIL-bound launch. Reuse Stage 3's spawn/disk plumbing.
5. **Config + a `*_multigpu.yaml` example** carrying `multi_gpu.eora_workers`,
   and update any Stage-4 budget/runtime comment.

Steps 1-3 are the feature; step 4 is conditional on measurement.

---

## 7. Test plan (runnable WITHOUT a real multi-GPU box)

**Pure-refactor gate (step 1):**
- Existing `test_stage4_golden_snapshot.py` + `test_stage4_eora_opt_levers.py`
  re-run after the `_solve_expert_tile` extraction — must stay byte/value
  identical. This proves the helper is a faithful extraction before any fan-out.

**Unit (no real multi-GPU box needed):**
- `test_solve_expert_tile_pure` — call `_solve_expert_tile` for a single expert
  twice (once with `target_device='cpu'`, once via a forced second "device" that
  is also CPU under `CUDA_VISIBLE_DEVICES=""`) and assert identical `(Uc, Vc,
  take_eff)` — proves device-parameterization is a pure relocation.
- `test_eora_workers_resolution` — assert `_resolve_eora_workers` clamps to
  `min(requested, device_count, N)` and returns 1 when `multi_gpu` absent /
  `device_count()<=1` (monkeypatch-free: drive via config dict +
  `torch.cuda.device_count` already returning 0 in CI).

**Equivalence (the load-bearing one), CPU-stand-in for the second device:**
- `test_eora_taskparallel_equivalence` — run `compensate_layer` on a tiny
  fixture (reuse the `tiny_model` from `test_stage4_golden_snapshot.py` / the
  smoke fixture) **twice**: once serial (`eora_workers=1`), once with
  `eora_workers=2` where the two "worker devices" are both **CPU** (forced via a
  device list `["cpu","cpu"]` injected by the test seam, since CI has no 2nd
  GPU). Assert:
  - **integer ranks** (`fe.ranks`, `rank_map`, `compensated_params`) are
    **exactly equal** (same backend → same `n_keep`/`take_eff`);
  - **float factor tensors** (`U_corr`/`V_corr` via `fe.{name}_U/V`) match within
    `rtol=1e-5, atol=1e-6` (CPU-vs-CPU is bit-identical in practice, but the
    tolerance is stated for cross-arch generalization — never `atol=0` across
    nominal backends).
  This drives the real gather + ascending-e row-place path on CPU, proving the
  task-split is equivalent to serial **without a GPU**.
- `test_eora_gather_order_deterministic` — run the W=2 path twice and assert
  identical output (fixed expert→worker bands + ascending-e gather).

**Integration (requires ≥2 physical GPUs — the live box, not CI):**
- Same-arch 2/4/8-GPU run vs 1-GPU on the real model: assert `eora_ranks.json`
  **byte-identical** (`atol=0` on integer ranks) and a sampled factor tile
  bit-identical, plus the **measured per-layer wall-time drop** (the actual point
  of the feature, step-3 measurement of §6).

**Regression guards (must stay green, unchanged):** the full B6 list. All
exercise the `eora_workers<=1` serial path which §3 keeps byte-identical.

---

## 8. Risks + scope boundaries

**In scope (this plan):** Stage 4 EoRA task-parallel per-(layer, expert) solve
across N devices (lever 1), the worker-resolution + config surface, the
pure-helper extraction. 1-GPU byte-identical; N-GPU same-arch byte-identical
(integer ranks) / bit-identical (factor values).

**Out of scope (noted, not built):**
- **Sharding the Stage 4 model multi-device** (§2.2) — rejected (M2 hazard, zero
  VRAM benefit). If a future model is too large for one card *even factored*, the
  M2 guard would need a real fix (per-expert V/U co-placement assertion in
  `wrapped_factored` + a factored-aware device map) — a separate epic, flagged
  here, NOT undertaken.
- **Stage 3** (already landed, `tasks/PLAN_MULTIGPU_STAGE3.md`) — Stage 4 reuses
  its `_resolve_*` worker-clamp pattern and (if step-4 needed) its spawn/disk
  plumbing, but adds no Stage-3 code.
- **Cross-layer parallelism** — lever 1 parallelizes experts *within* a layer
  (they share `fe`, and `widen_rank` mutates `fe` in place). Parallelizing whole
  layers across workers is possible (layers are independent until finalize) but
  not needed: the per-layer expert fan-out already saturates N devices for
  N_experts≫N_gpu. Documented as a future option, not built.

**Risks:**
- R1 — **In-process GIL launch latency** (2.1a): if the Python per-expert glue
  dominates the CUDA kernel time, device fan-out won't overlap well. Mitigation:
  measure at step 3; fall to spawn (2.1b) if needed. The heavy ops release the
  GIL, so overlap is expected to be good for the ~`d_in²` eigh.
- R2 — **CPU input copy bandwidth**: every expert solve copies `W_orig`+`A` from
  CPU to its worker device. For `d_in≈2048`, `A` is `~16 MB` fp32 — negligible
  vs the eigh. Confirmed small; not a bottleneck.
- R3 — **noise-floor boundary on cross-arch** (§4): a different backend could
  flip `n_keep` for an eigenvalue exactly on `thresh`, changing an integer rank.
  Measure-zero; CI asserts integer equality only on the **same backend**; the
  same-arch integration check is the real guarantee.
- R4 — **the gather must preserve ascending-e order** or the (log-only) residual
  norms and (golden) rank map could reorder. Mitigation: explicit
  ascending-`e` row-place; `test_eora_gather_order_deterministic`.

**Boundary invariant:** every change is gated so that
`torch.cuda.device_count() <= 1` OR `multi_gpu.eora_workers <= 1` OR `multi_gpu`
absent reproduces the current serial code path with **zero behavioral delta** —
the live single-GPU run and all Stage-4 golden/smoke tests are untouched. The M2
guard and all Stage-3 work are not modified.
