# Plan: Stage-3 task-parallel wins — α-grid + per-expert SVD factor

**Date:** 2026-06-13
**Branch / worktree:** `feat/stage3-taskparallel` @ `/home/lucas/ai/wt-s3mg` (code root: `max_quality/`)
**Status:** plan — not yet implemented
**Spec source:** `max_quality/docs/multigpu_analysis/stage3.md` §3 (swift_svd_alpha) + §4 (aa_svd_factor); verified against the actual code below.

---

## Goal

Two RESULT-PRESERVING, default-OFF, byte-identical multi-GPU levers in Stage 3, each ~N×:

1. **`swift_svd_alpha` — α-grid task-parallel.** The paper-exact 11-candidate WikiText-2 PPL grid (`_swift_svd_plus_alpha_search_validation`, `src/moe_compress/stage3/plugins/swift_svd_alpha.py:654-806`) evaluates each α SERIALLY on one device (~31 min on H200). Distribute the α candidates across N GPUs (replica `r` owns candidates `r::N`); each replica factors→evals→returns `(α, ppl)`; the parent argmins on `(ppl, α)`. Each α evaluation is an independent pure function (factor whole model at the α's per-expert ranks from the shared CPU `originals` snapshot → end-to-end PPL → restore). PPL eval is BATCH_INVARIANT, so per-GPU auto-batch applies cleanly.

2. **`aa_svd_factor` — per-expert SVD factor task-parallel.** The per-layer factor loop (`orchestrator.py:806` `loop_over(moe_layers, ..., ("factor_layer",))`) is SERIAL; inside `factor_layer` (`aa_svd_factor.py:480-718`) the per-(expert,matrix) `eigh(B)`+`SVD(W·rhs)`+back-solve is a pure function of `(W,A,B,C,k)`. Task-parallel the per-expert solves across GPUs **inside one layer's factor** — MIRROR Stage-4 EoRA's proven concurrency engine (`stage4/plugins/eora_compensation.py:_run_expert_bands` / `_resolve_worker_devices` / ascending-e assembly). Merge the shared mutable `rank_map` (HAZARD H1) in ascending-e on the main thread; install `FactoredExperts` in the parent. No forward batch → no auto-batch.

### Non-negotiable correctness invariants

Both levers may have a live ablation restart-resume off Stage 3, so:

1. **Result byte-identical to current single-GPU output** when the lever is OFF (default), and **bit-identical-in-practice** (CPU-vs-CPU `torch.equal`; cross-arch tolerance `rtol=1e-5, atol=1e-6`) when ON. The work units are pure functions; parallelism changes only *when* they run, never *what* they compute.
2. **1-GPU / no-`multi_gpu` path unchanged.** Default knobs (`alpha_workers=1`, `factor_workers=1`, or `multi_gpu` absent) → serial in-process path, the exact code that runs today.
3. **Golden gate.** `tests/test_stage3_golden_snapshot.py` MUST stay green WITHOUT `MOE_REGEN_GOLDEN` for BOTH levers. (The golden runs `block_refine` OFF + `alpha_grid` length 1 → the α-grid lever's `len(alpha_grid) > 1` guard never fires; the factor lever defaults to 1 worker → serial. So the golden is untouched by construction; the test still pins it.)
4. **Per-GPU auto-batch preserved** on the α-eval path (BATCH_INVARIANT — each replica probes its OWN pinned-device VRAM, mirroring the cov DP worker). The SVD-factor path has NO forward batch → auto-batch N/A.

---

## Scope decisions & key architectural choices

### Lever 1 (α-grid): process spawn, mirroring `run_dp_covariance_collection`

**Why processes, NOT threads (opposite of Lever 2).** Each α candidate runs an **end-to-end model forward** (`_evaluate_wikitext2_ppl`, `swift_svd_alpha.py:346-371`: `model(input_ids=batch, labels=batch)`) AND mutates the model in place (`_factor_model_at_ranks` swaps `FactoredExperts` into every layer; `_restore_fused_experts` reverses it). Two α candidates sharing one resident model in one process would corrupt each other's in-place factor/restore. Each replica therefore needs its OWN resident model copy + its own CUDA context + `CUDA_VISIBLE_DEVICES` pinning — exactly the cov DP pattern. This is `torch.multiprocessing` spawn, mirroring `run_dp_covariance_collection` (`covariance_collection.py:1249-1325`) and `_cov_replica_worker` (`:1110+`).

**What each replica needs (all already on disk or rebuildable):**
- The student checkpoint path (reload its own model) — resolved exactly like the cov DP branch (`orchestrator.py:413-424`: try `stage2p5_final` then `stage2_pruned`).
- The CPU `originals` snapshot — already persisted to `_stage3_original_weights.pt` (+ manifest) at `orchestrator.py:654-690`, BEFORE α-search. Replicas `torch.load` it (map to CPU).
- `A_cov` — reload from the Stage-2 sidecar / `_stage2_input_covariance.pt` (same path resolution the parent used).
- B/C cov spill dirs — already on disk (`bcov_spill_dir` / `ccov_spill_dir`), lazy-loaded per layer inside `_factor_model_at_ranks`.
- `group_stats` / `base_ranks(=ranks)` — plain dicts; pass as spawn args.
- The eigh-cache (`_stage3_alpha_eigh_cache`): it is `(B,A,C)`-determined and IDENTICAL across all candidates (`swift_svd_alpha.py:735-750`). Each replica rebuilds it LOCALLY in its own cache subdir (`_stage3_alpha_eigh_cache/_replica_{r}`) so candidate-0-on-replica-r fills it and r's other candidates reuse it. Do NOT share one cache dir across replicas (write race + the unconditional rmtree-at-entry would clobber peers).

**(α,ppl) merge — result-preserving argmin.** Each replica returns its list of `(alpha, ppl)` for the candidates it owns (its disjoint `alpha_grid[r::N]` slice). The parent concatenates and picks the winner with the SAME tie-break the serial loop uses today. **Critical tie-break fidelity (H-α1):** the serial loop (`swift_svd_alpha.py:791-793`) keeps the FIRST α (lowest grid index) on a strict-`<` improvement, i.e. ties go to the earlier-evaluated (lower-α) candidate. To reproduce this exactly, the parent MUST iterate the merged results in ASCENDING GRID-INDEX order (not completion order, not per-replica order) and apply the identical `if ppl < best_ppl` rule. Implement by returning `(grid_idx, alpha, ppl)` per candidate and sorting by `grid_idx` before the argmin fold. This makes the winner independent of which replica finished first → byte-identical α selection.

**Per-GPU auto-batch (BATCH_INVARIANT).** `_evaluate_wikitext2_ppl` sums NLL over tokens (`nll_sum += loss * n_tokens` then `exp(nll_sum/tok_count)`) — grouping-independent, so the PPL is batch-invariant (a true `FidelityClass.BATCH_INVARIANT`). The per-replica auto-batch MUST mirror the cov DP worker: set `CUDA_VISIBLE_DEVICES` FIRST, THEN build `CudaMemProbe(device)` / `size_batch(..., max_cap=..., mem=CudaMemProbe(local_device))` inside the worker with the factored model resident, `run_with_oom_backoff(..., floor=validation_batch_size)`. Gate identically to cov: double-gate on `swift_svd_plus.validation_batch_size == "auto"` AND `auto_batch.enabled`. Default (int `validation_batch_size`, =16) → no probe → byte-identical eval. **This is the HARD REQUIREMENT** ("each replica probes its own VRAM") and is met by the `CUDA_VISIBLE_DEVICES`-first ordering.

**Why not data-parallel the PPL eval.** Splitting `validation_samples` across GPUs (sum NLL) barely helps: eval is ~0.3 min vs the ~2 min factor per candidate (`swift_svd_alpha.py:693`). Task-parallel over the 11 candidates is the real ~N× win (for N ≤ 11). REJECTED data-parallel; this plan does task-parallel only.

### Lever 2 (per-expert SVD factor): ThreadPool, MIRROR Stage-4 EoRA

**Why threads, NOT processes (opposite of Lever 1).** The factor is pure in-process linear algebra: `eigh(B)` + `SVD(W @ rhs)` + back-solve on `dev` (`aa_svd_factor.py:618-650`). NO forward pass. The CUDA kernels release the GIL and run async on each device's default stream, so Python threads achieve real device overlap WITHOUT re-loading the 50 GB `originals` into child processes. This is EXACTLY the Stage-4 EoRA situation; reuse its proven engine verbatim in shape:
- `_resolve_worker_devices(worker_devices, n, home_device)` (eora_compensation.py:571-587) — device resolution + the `["cpu","cpu"]` CI seam.
- `_run_expert_bands(bands, device_of, solve_one, ...)` (eora_compensation.py:482-568) — banded-by-ORDINAL ThreadPool, ascending-e within each band, gather-to-home INSIDE the thread, main-thread assembly after join. C1 discipline (band by ordinal `w`, not device object) so the CPU `["cpu",...]` seam stays multi-band.
- Pin intra-op BLAS to 1 on the main thread during the pool (eora_compensation.py:545-553) to avoid `bands × cores` oversubscription on the CPU stand-in.

**The task-parallel unit — extract `_factor_expert_tile`.** Mirror EoRA's `_solve_expert_tile` (eora_compensation.py:395-473): extract the per-expert inner block of `factor_layer` (`aa_svd_factor.py:611-686`) into a pure function parameterized by `target_device` (replacing the closure `dev`). It reads ONLY this expert's tensors (`originals[(layer,e,name)]`, the per-expert `B/A/C` lookups) and returns the per-(matrix) `(U_k, V_k, k_eff, k)` tiles plus the zero-pad-to-slot already applied — gathered to the home device INSIDE the worker thread (like EoRA gathers `Uc_home`). 

**Gate/up eigh memo is SIMPLER than EoRA — no cross-thread plumbing.** In `factor_layer` the loop is matrix-INNER, expert-OUTER (`for e: ... for name in MATRIX_NAMES:`, `aa_svd_factor.py:611,628`), so `gate_up_decomp` (`_precompute_eigh`, `:618-626`) is computed once per expert and consumed within that expert's iteration. It therefore stays a LOCAL inside `_factor_expert_tile` (computed on `target_device`, used for gate+up of the same expert) and NEVER crosses threads. EoRA's loop is matrix-OUTER so its `gate_spectrum` memo must survive the gate→up window across the band engine; Lever 2 has no such need — drop the `set_gate_spectrum` plumbing (pass a no-op `lambda e,s: None`).

**HAZARD H1 — the shared mutable `rank_map` merge.** `rank_map` is ONE shared dict on the root ctx (`orchestrator.py:269,308`); `factor_layer` mutates it in place across loop iterations (`aa_svd_factor.py:684`: `rank_map[f"L{layer}_E{e}_{name}"] = k`). Under per-expert parallelism the writes are to DISJOINT string keys (`L{layer}_E{e}_{name}` — unique per (e,name)), so even concurrent writes would not collide — BUT to guarantee byte-identical dict iteration order and avoid any GIL-edge race, the band engine returns per-expert results keyed by `e` (a dict, disjoint keys — EoRA's exact `band_results[e]` pattern), and the MAIN thread writes `rank_map` in ASCENDING-e order after join. Same for `new_factored.set_factors(e, name, ...)` (`aa_svd_factor.py:683`): the `FactoredExperts` slot fill is a disjoint per-e row write, performed on the main thread in ascending-e from the gathered tiles — never inside the worker thread. The trackio `err_sum`/`n_per_matrix`/`k_eff_clip_count` accumulators (`aa_svd_factor.py:608-610,651-652,685-686`) are log-only (never the golden) and accumulated on the main thread in ascending-e (mirror EoRA's residual-acc discipline). **`setattr(ref.mlp,'experts',new_factored)` + `ref.experts_module = new_factored` (`aa_svd_factor.py:688-689`) stay on the MAIN thread**, once, after all experts of this layer are assembled — these mutate the model, the parent owns them.

**Granularity: per-expert WITHIN a layer, NOT per-layer across layers.** The layer loop stays serial (`loop_over` per-layer, driven by the orchestrator) — each layer lazy-loads its own ~5 GB B/C cov spill and the B-cov prefetcher (Tier-1 item 9, `orchestrator.py:802-809`) overlaps the next layer's disk read. Parallelize the ~200-expert inner loop of each layer across GPUs. This keeps the prefetcher + per-layer cov residency model intact (preserved per the spec, `stage3.md:194-195`) and matches EoRA's per-layer-internal banding exactly. (Per-layer-across-GPUs would fight the single-layer cov residency + prefetcher and is NOT chosen.)

**Byte-identity of the SVD assembly.** Each `_factor_expert_tile` is a pure fn of its inputs; `FactoredExperts` rows are filled in ascending-e on the main thread; the zero-pad-to-slot (`aa_svd_factor.py:653-683`) is applied inside the tile on `target_device` then gathered home — bit-identical to the serial fill because the slot width `ranks_layer[name]` is a per-LAYER constant computed BEFORE the per-expert loop (`aa_svd_factor.py:522-533`), independent of expert order. Worker completion order is unobservable.

---

## Out of scope (explicitly excluded)

- **Live ≥2-GPU validation** — DEFERRED to a real multi-GPU box (CI has 0 GPUs). All tests here run CPU-simulated multi-device via the `worker_devices=["cpu","cpu",...]` ctx seam (Lever 2) and a single-process in-thread fan-out stand-in for the spawn driver (Lever 1) — see Testing.
- **`d_rank_allocate`** — CPU-fp64, device-independent, seconds-to-minutes, NOT a wall-clock bottleneck (`stage3.md:131-142`). TASK-PARALLEL-able but NOT-WORTH-IT. Excluded.
- **`block_refine`** — METRIC-PINNED + cross-block serial dependency (`stage3.md:197-215`); a different (DDP) problem, out of scope.
- **`wanda_intra_expert_score`** — DATA-PARALLEL MISS-path only, gated OFF by default; out of scope.
- **Data-parallel PPL eval** for Lever 1 (rejected above — task-parallel candidates is the real win).
- **acov shift-cov files** — the recently-landed shift-cov changes live in `covariance_collection.py` + `orchestrator.py` (the `persist_shift_covariance` ride-along, `orchestrator.py:887-901`; `_consolidate_shift_covariance`). This plan touches `swift_svd_alpha.py`, `aa_svd_factor.py`, and the factor-loop wiring in `orchestrator.py:784-809` (the `loop_over` call + new worker resolution) — **different lines**, NO overlap (confirmed via git log + line ranges below).

---

## Files to change

| File | Change |
|---|---|
| `src/moe_compress/stage3/orchestrator.py` | Add `_resolve_alpha_workers` + `_resolve_factor_workers` (mirror `_resolve_cov_replicas` / Stage-4 `_resolve_eora_workers`). Thread `factor_workers` (+ optional `factor_worker_devices` seam) onto `run_ctx` before the `loop_over` at `:806`. Branch the α-dispatch (`:746-752`) to the spawn driver when `alpha_workers > 1` (else unchanged). |
| `src/moe_compress/stage3/plugins/swift_svd_alpha.py` | Add `run_dp_alpha_search(...)` spawn driver + `_alpha_replica_worker(...)` (mirror `run_dp_covariance_collection` / `_cov_replica_worker`). Factor the argmin fold into `_argmin_alpha(results)` so serial + DP paths share the identical tie-break. Per-replica auto-batch (BATCH_INVARIANT) inside the worker. |
| `src/moe_compress/stage3/plugins/aa_svd_factor.py` | Extract `_factor_expert_tile(...)` (pure per-expert solve, `target_device`-parameterized). Import + reuse `_resolve_worker_devices` / `_run_expert_bands` (see Q1). Rewrite the `factor_layer` per-expert loop to: resolve `factor_workers` + bands, run the band engine, assemble ascending-e on the main thread. |
| `tests/test_stage3_taskparallel.py` (NEW) | All equivalence + resolution + determinism tests (below). |
| `tests/test_stage3_golden_snapshot.py` | UNCHANGED — re-run to prove byte-identity (no regen). |

---

## Bite-sized TDD tasks

> Discipline: each task = write/extend the test FIRST (red), implement (green), then run the golden + full stage3/stage4 suites. Worker count default = 1 everywhere → the serial path is the byte-identical reference all equivalence tests diff against.

### Lever 2 (per-expert SVD factor) — do FIRST (simpler, no spawn, directly mirrors landed EoRA)

**T2.0 — confirm the EoRA engine is importable / decide reuse vs relocate (Q1).**
- Test: `test_factor_engine_imports` — `from moe_compress.stage4.plugins.eora_compensation import _run_expert_bands, _resolve_worker_devices` succeeds (or, if relocated, from the shared module). Decide Q1 (below) before writing code.

**T2.1 — extract `_factor_expert_tile` (pure, device-parameterized).**
- Test: `test_factor_expert_tile_pure` — build a tiny `(W, A, B, C, k)` for one expert; call `_factor_expert_tile` on two CPU "devices"; assert `torch.equal` on `(U_k, V_k)` and equal `(k_eff, k)`. Mirror `test_solve_expert_tile_pure` (test_stage4_multigpu.py:133-161).
- Impl: lift `aa_svd_factor.py:612-686` inner block into a function:
  ```python
  def _factor_expert_tile(
      layer_idx, e, target_device, *,
      originals, A_cov, B_cov, C_cov, per_expert_ranks, ranks_layer,
      B_cov_dtype,
  ):
      """Pure per-expert AA-SVD solve — the task-parallel unit (mirror of
      stage4 _solve_expert_tile). Reads ONLY expert e's tensors; relocating
      to any device is a pure relocation (same kernels => bit-identical).
      Returns {name: (U_k_padded, V_k_padded, k_eff, k, rel_err)} for the
      three matrices, all on target_device, zero-padded to ranks_layer[name]."""
      from ...stage3_svd import _cov_lookup  # lazy (import-cycle; see module docstring)
      tiles = {}
      B_shared = _cov_lookup(B_cov, layer_idx, e, "gate_proj")
      A_shared = _cov_lookup(A_cov, layer_idx, e, "gate_proj")
      C_shared = _cov_lookup(C_cov, layer_idx, e, "gate_proj") if C_cov is not None else None
      gate_up_decomp = None
      if B_shared is not None:
          try:
              gate_up_decomp = _precompute_eigh(
                  B_shared, A_shared, C_shared,
                  device=target_device, storage_dtype=B_cov_dtype)
          except ValueError:
              pass
      for name in MATRIX_NAMES:
          W = originals[(layer_idx, e, name)].to(device=target_device, dtype=torch.float32)
          k = (per_expert_ranks.get((layer_idx, name, e), ranks_layer[name])
               if per_expert_ranks is not None else ranks_layer[name])
          if name in ("gate_proj", "up_proj") and gate_up_decomp is not None:
              U_k, V_k, rel_err, k_eff = _aa_svd_precomputed(W, gate_up_decomp, k, device=target_device)
          else:
              A = _cov_lookup(A_cov, layer_idx, e, name)
              B = _cov_lookup(B_cov, layer_idx, e, name)
              C = _cov_lookup(C_cov, layer_idx, e, name) if C_cov is not None else None
              U_k, V_k, rel_err, k_eff = _aa_svd(W, A, B, k, C=C, device=target_device, storage_dtype=B_cov_dtype)
          # zero-pad to slot (verbatim aa_svd_factor.py:664-682), incl. the width<=slot assert
          slot = ranks_layer[name]
          ...  # (pad exactly as today)
          tiles[name] = (U_k, V_k, int(k_eff), int(k), float(rel_err))
      return tiles
  ```
  Note: pass `B_cov = B_acc.covariance`, `C_cov = C_acc.covariance if C_acc else None` (the tile must NOT touch the accumulator's load/unload — the main thread keeps owning per-layer load/unload + the prefetcher).

**T2.2 — band-engine equivalence (serial == W=2 CPU).**
- Test: `test_factor_taskparallel_equivalence` — build a tiny layer fixture (FactoredExperts + originals + A/B/C cov dicts), run `factor_layer` twice (serial `factor_workers=1`; `factor_workers=2` + `factor_worker_devices=["cpu","cpu"]`). Assert `rank_map` dicts EXACTLY equal, and all `FactoredExperts.*_U/_V` within `rtol=1e-5, atol=1e-6`. Mirror `test_eora_taskparallel_equivalence`.
- Impl: rewrite the per-expert loop in `factor_layer` to: compute `ranks_layer` (unchanged, before the loop), build `eligible = list(range(N))`, resolve `effective_workers = min(factor_workers, max(1,N))` + `device_of` contiguous bands + `bands` by ordinal (verbatim EoRA pattern, eora_compensation.py:830-869), define `_solve_one(e, tgt)` calling `_factor_expert_tile(...)` and gathering tiles to `dev`, call `_run_expert_bands(bands, device_of, _solve_one, name="factor", set_gate_spectrum=lambda e,s:None, concurrent=(effective_workers>1))`, then ascending-e main-thread assembly: `for e in eligible: for name: new_factored.set_factors(e,name,U,V,effective_rank=k_eff); rank_map[...] = k; accumulate trackio sums`. Then `setattr(...)` once.
  - The engine's `solve_one` returns a 6-tuple `(Uc,Vc,take_eff,rb,ra,spec_out)`. For factor, pack the 3-matrix tile dict into slot-0 and pass `(payload, None, 0, None, None, None)` (Q2) — zero engine change.

**T2.3 — determinism + C1 multi-band proof.**
- Test: `test_factor_gather_order_deterministic` (W=3 run twice → `torch.equal`) + `test_factor_concurrent_exact_equals_serial_cpu` (9 experts, W=3, assert `_LAST_BAND_COUNT==3` + `_LAST_RAN_THREADED` + `torch.equal` vs serial). Mirror test_stage4_multigpu.py:332-362.

**T2.4 — worker resolution.**
- Test: `test_factor_workers_resolution` — `_resolve_factor_workers({})==1`, `{"multi_gpu":{"factor_workers":1}}==1`, requested-8 clamps to `min(8, device_count())` (CI: 1). Mirror test_stage4_multigpu.py:196-213.
- Impl: `_resolve_factor_workers(config)` in `orchestrator.py`, byte-for-byte clone of `_resolve_eora_workers` (stage4/orchestrator.py:69-85) with key `factor_workers`. Thread onto `run_ctx` before `loop_over` (`orchestrator.py:804`-ish): `run_ctx.set("factor_workers", _resolve_factor_workers(config))` and optionally a `factor_worker_devices` seam (set by tests only).

**T2.5 — golden gate (Lever 2).**
- Run `pytest tests/test_stage3_golden_snapshot.py -v` (no `MOE_REGEN_GOLDEN`) → green. Default `factor_workers=1` ⇒ one band ⇒ inline ⇒ byte-identical.

### Lever 1 (α-grid) — do SECOND (spawn driver)

**T1.1 — factor the serial argmin fold + per-candidate results.**
- Test: `test_alpha_argmin_tiebreak` — feed a fake results list with a tie (`[(0,0.0,5.0),(1,0.1,5.0)]`) into `_argmin_alpha(results)`; assert it returns α=0.0 (lower grid idx wins on tie — strict-`<` fold over ascending grid_idx). Pins H-α1.
- Impl: factor the fold at `swift_svd_alpha.py:791-793` into `_argmin_alpha(results)` that sorts by `grid_idx` then folds `if ppl < best_ppl`. Have `_swift_svd_plus_alpha_search_validation` build `results=[(idx,alpha,ppl),...]` and return `_argmin_alpha(results)`. **Byte-identical to today** (same iteration order, same rule) — guarded by the existing stage3 α tests + the golden.

**T1.2 — the spawn driver + worker (single-process equivalence first).**
- Test: `test_alpha_dp_equivalence_inproc` — a CPU/in-process stand-in: run the serial validation search vs a fan-out that splits `alpha_grid` into 2 disjoint slices, runs each slice through the SAME `_factor/eval/restore` cycle in sequence (no real spawn), merges via `_argmin_alpha`. Assert identical winning α + identical per-candidate `(idx,alpha,ppl)` set. Proves the SLICE + MERGE math without 2 GPUs / real processes. (Real spawn is exercised only on the live box — Out of scope here.)
- Impl: `run_dp_alpha_search(*, config, artifacts_dir, student_path, group_stats, ranks, alpha_grid, originals_path, bcov_spill_dir, ccov_spill_dir, ...)` mirroring `run_dp_covariance_collection` (covariance_collection.py:1249-1325): `spawn_args` per replica carry the replica's `alpha_grid[r::N]` slice (as `(grid_idx, alpha)` pairs) + `CUDA_VISIBLE_DEVICES` string + a per-replica eigh-cache subdir; `ctx.Process(target=_alpha_replica_worker)`; join; collect each replica's `[(idx,alpha,ppl),...]` (per-replica result JSON, mirror cov's spill-to-disk + parent-read — Q3); `_argmin_alpha(all_results)`. `_alpha_replica_worker` sets `CUDA_VISIBLE_DEVICES` FIRST, reloads model + originals + A_cov, then per owned `(idx,alpha)`: redistribute→factor→eval (auto-batch per-replica)→restore→record `(idx,alpha,ppl)`.

**T1.3 — per-replica auto-batch (BATCH_INVARIANT).**
- Test: `test_alpha_eval_batch_invariant` — `_evaluate_wikitext2_ppl(model, val, device=cpu, batch_size=4)` == `batch_size=7` (tiny model, CPU) to ~1e-6. Proves the eval is BATCH_INVARIANT so per-replica sizing is safe.
- Impl: inside `_alpha_replica_worker`, when `validation_batch_size=="auto"` AND `auto_batch.enabled`, after the model is factored at the candidate's ranks, size with `CudaMemProbe(local_device)` + `run_with_oom_backoff(floor=16)`. Default int → no probe → unchanged. Document the `CUDA_VISIBLE_DEVICES`-first ordering (mirror covariance_collection.py:1178-1231 comment block).

**T1.4 — orchestrator wiring + α-resume interaction.**
- Test: `test_alpha_dp_resume_skips_search` — when `_stage3_alpha_result.json` exists, the DP path is NOT entered (the existing resume branch at `orchestrator.py:731-742` short-circuits before `select_alpha`). Assert no spawn driver call.
- Impl: add `_resolve_alpha_workers` (clone of `_resolve_cov_replicas` with key `alpha_workers`). In the `if not _alpha_loaded:` branch (`:746`), when `alpha_workers > 1` AND `validation_samples > 0` AND `len(alpha_grid) > 1`, call `run_dp_alpha_search(...)` to get `best_global_alpha`, then run the (cheap, CPU-fp64) spectral-proxy + redistribute exactly as `select_alpha` does for `per_group_type` — the DP driver replaces ONLY the expensive validation grid; the rest of the α-dispatch is unchanged. Else fall through to the existing `walk_phases(("select_alpha",), ...)`.
  - **Subtlety (H-α2):** when `per_group_type=True` (production default) the validation α is DISCARDED for factoring (`swift_svd_alpha.py:110-138` honest-cost note L2) — the per-type spectral proxy drives factoring. So in that config the DP grid is pure audit/telemetry; the FACTORING ranks come from the spectral proxy either way → byte-identical model. Document so a reviewer doesn't think the DP α leaks into the model.

**T1.5 — golden gate (Lever 1).**
- Re-run `pytest tests/test_stage3_golden_snapshot.py -v` → green. The golden uses `alpha_grid` length 1 → `len(alpha_grid) > 1` false → DP path never entered → byte-identical.

---

## Test commands

```bash
cd /home/lucas/ai/wt-s3mg
# new task-parallel suite
pytest max_quality/tests/test_stage3_taskparallel.py -v
# golden guardrail — MUST pass WITHOUT regen, for BOTH levers
pytest max_quality/tests/test_stage3_golden_snapshot.py -v
# full stage3 + stage4 (Lever 2 reuses the EoRA engine — guard cross-stage)
pytest max_quality/tests/ -k "stage3 or stage4 or multigpu" -q
# argmin tie-break + batch-invariance units
pytest max_quality/tests/test_stage3_taskparallel.py -k "tiebreak or batch_invariant" -v
```

## Golden guardrails

- `test_stage3_golden_snapshot.py` — UNCHANGED, NO `MOE_REGEN_GOLDEN`. Both levers default-OFF so the snapshot bytes are untouched by construction; the test still pins it after each task.
- `test_stage4_golden_snapshot.py` + `test_stage4_multigpu.py` — must stay green if Q1 resolves to "reuse `eora_compensation` engine in place" (Lever 2 imports it; don't perturb it). If Q1 resolves to "relocate to a shared module", update the EoRA import site + re-run the Stage-4 golden.

---

## Open questions

- **Q1 — reuse vs relocate the band engine.** Lever 2 needs `_run_expert_bands` + `_resolve_worker_devices`, currently in `stage4/plugins/eora_compensation.py`. A stage3→stage4 import is a cross-stage dependency (mild smell, but stage4 already imports `tools/dtype_noise_floor` shared with stage3). Options: (a) import from `stage4.plugins.eora_compensation` directly (zero churn, couples stage3→stage4); (b) relocate both helpers to `utils/band_engine.py` (or `tools/`) and re-import from BOTH stages (cleaner; touches the landed EoRA file + its golden). **Recommendation: (b)** — the engine is stage-agnostic and the relocation is byte-identical (mirror the `_NOISE_FLOOR_BY_DTYPE` relocation precedent, eora_compensation.py:122-133). Confirm before T2.0.
- **Q2 — engine payload generality.** `_run_expert_bands`'s `solve_one` contract returns a fixed 6-tuple (EoRA-shaped). Lever 2's per-expert result is a 3-matrix dict. **Recommendation: pack the dict into slot 0** and pass `None` for the rest (zero risk to the landed engine); generalize to an opaque payload only if a third consumer appears.
- **Q3 — Lever-1 result transport.** Replica → parent `(idx,alpha,ppl)` transport: `mp.Queue` vs per-replica result JSON file (mirror cov's spill-to-disk + parent-read). **Recommendation: per-replica JSON file** under `artifacts_dir/_stage3_alpha_dp/_replica_{r}.json` — matches the cov DP idiom, crash-inspectable, no Queue serialization-of-CUDA-context risk. Confirm.
- **Q4 — per-candidate PPL telemetry ordering.** The DP path changes WHEN each candidate's `stage3/alpha_search/{alpha,ppl}` trackio emit fires (interleaved across replicas → out of grid-order). If any dashboard asserts ordering, re-emit in grid-order from the parent after merge. Likely benign (telemetry only) — confirm no downstream ordering dependency.

---

## Confirmed: no overlap with the just-landed acov shift-cov work

`git log` on `covariance_collection.py` / `orchestrator.py` shows the shift-cov ride-along (`_consolidate_shift_covariance`, `persist_shift_covariance` at `orchestrator.py:887-901`) living in the finalize block + cov-collection branch. This plan edits: `orchestrator.py:746-752` (α-dispatch) + `:804-806` (factor `loop_over` wiring) + new resolver fns; `swift_svd_alpha.py` (new DP driver/worker + argmin factor); `aa_svd_factor.py` (`_factor_expert_tile` + band-engine rewrite of `factor_layer`). Disjoint line ranges, disjoint concerns.
