# Auto-Batch v2 step 3 — Enable Auto Cov Batch on the DP Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Let the data-parallel (multi-replica) cov capture auto-size its forward batch (per replica, against each replica's own GPU VRAM), instead of the current hard `cov_auto=False`. Gated behind `cov_batch_size:"auto"` + `auto_batch.enabled` (same gate as the in-process path). Default byte-identical.

**The key insight (why this is SMALL — the pin subsumes min-agreement):** The spec §6 originally wanted an all-replica `min(candidate)` agreement so every replica used the SAME forward batch (to keep the reduction grouping uniform). **That requirement is now MOOT.** The per-sequence reduction pin (v2 step 1) makes each replica's per-key Gram **independent of that replica's forward batch size** — the reduction grouping is per-sequence regardless of bs. The DP reduce (`_reduce_spilled_cov_dirs`) is a key-wise SUM of finalized per-replica Grams: `B_total[k] = Σ_r B_r[k]`, and each `B_r[k] = Σ_{seq in shard_r} (per-seq Gram)` is the same whatever bs the replica chose. So replicas can size **independently** and the sum is correct (reduction-order drift = 0; the only residual is the bounded ~1e-6 forward-activation drift, N-independent, quality-neutral — exactly the single-GPU property). **No min-agreement, no cross-replica coordination.**

**Architecture:**
- Each DP worker calls `_collect_covariances` in its own process with `CUDA_VISIBLE_DEVICES` pinned to its GPU subset (existing). Change `cov_auto=False` → `cov_auto=_cov_is_auto(s3)` and pass `auto_batch_cfg=AutoBatchConfig.from_dict(s3.get("auto_batch"))`. The probe then measures THIS replica's GPU free VRAM (`CudaMemProbe` reads the pinned device) → each replica sizes to its own budget. The cov-wire machinery (v2 step 2) is already in `_collect_covariances`; this just flips the worker's gate.
- Default (no `"auto"`) → `_cov_is_auto` False → worker stays on the inherited int (bs=1) → DP reduce byte-identical (the multigpu golden/A4 tests stay green).

**Tech Stack:** PyTorch, pytest. Code root `max_quality/`. CPU-only (the multigpu test harness CPU-simulates replicas); the LIVE multi-GPU cov-wall speedup needs real ≥2 GPUs (deferred, like the existing multi-GPU Stage-3 validation).

**Spec:** `docs/.../2026-06-11-...design.md` §6 (note: this plan SUPERSEDES the §6 "min(candidate) agreement" with "pin makes independent sizing correct"). Builds on v2 step 1 (pin, `59666d8`) + step 2 (cov-wire, `86352a5`).

---

## File Structure
- **Modify** `src/moe_compress/stage3/plugins/covariance_collection.py` — the DP worker call site (~`:1175-1193`): `cov_auto=_cov_is_auto(config["stage3_svd"])`, `auto_batch_cfg=AutoBatchConfig.from_dict(config["stage3_svd"].get("auto_batch"))`. Update the stale "DP worker stays inherited" comment to the pin-subsumes-min-agreement rationale.
- **Modify** `tests/test_multigpu_stage3.py` — add a DP-reduce-with-heterogeneous-replica-batches test (the core correctness claim).
- **Goldens** — NOT TOUCHED (default not "auto").

---

## Task 1: Enable auto on the DP worker + prove the reduce is batch-independent

**Files:** Modify `covariance_collection.py` worker call site; extend `tests/test_multigpu_stage3.py`.

- [ ] **Step 1: Failing test** — the load-bearing correctness claim. In `tests/test_multigpu_stage3.py` (CPU-simulated replicas, tiny model), add `test_dp_reduce_heterogeneous_replica_batches_matches_bs1`: run the DP cov collection where replica 0 captures at cov_bs=A and replica 1 at cov_bs=B (≠A) — both with the pin active — then `_reduce_spilled_cov_dirs`; assert the reduced per-key Gram equals the reduced result from a uniform bs=1 run: `torch.equal` for gate_proj/up B + cross-cov C keys, `torch.allclose` (rtol/atol ~1e-5) for factored down_proj B keys (forward-drift residual). This proves replicas need NOT agree on a batch. **Drive the REAL auto path (Review L1):** call each replica's `_collect_covariances(calib=shard_r, cov_auto=True, auto_batch_cfg=...)` with a fake `CudaMemProbe`/forced `cov_bs` so the actual auto re-slice (`iter_batches(shard_r, bs)`) + `run_with_oom_backoff` + per-sequence pin are exercised — do NOT just monkeypatch `_resolve_cov_batch_size` on the non-auto branch (that would leave the auto re-slice untested). The CPU harness (cf. `test_cov_dp_equivalence`) already simulates shards via disk handoff.
- [ ] **Step 2: Run** → it should ALREADY pass if the pin is correct and the wiring lets a worker auto-size (the pin is active regardless of auto). If it FAILS, the heterogeneous-batch reduce diverges beyond tolerance → STOP and report (a real pin/reduce bug). If it passes pre-change, it documents the property the worker-gate flip relies on.
- [ ] **Step 3: Flip the worker gate.** At the worker call site (~`:1188`): `cov_auto=_cov_is_auto(config["stage3_svd"])` (was `False`); add `auto_batch_cfg=AutoBatchConfig.from_dict(config["stage3_svd"].get("auto_batch"))`. Confirm `_cov_is_auto`/`AutoBatchConfig` are imported in the module (they are, from step 2). Update the `:1183` comment.
- [ ] **Step 4: Default-off test** — add `test_dp_default_no_auto_inherited`: with no `cov_batch_size:"auto"`, the worker's `_cov_is_auto` is False → `cov_auto=False` → the probe/backoff never run (spy on `size_batch`); the DP reduce is byte-identical to today. (The existing `test_a4`/multigpu golden already pin the default.)
- [ ] **Step 5: Run** the new tests → PASS.
- [ ] **Step 6: GOLDEN GUARDRAIL** — `cd max_quality && python3 -m pytest tests/test_multigpu_stage3.py tests/test_stage3_golden_snapshot.py tests/test_cov_reduction_pin.py tests/test_cov_autobatch_wire.py -q` MUST pass UNCHANGED, no `MOE_REGEN_GOLDEN`. Default has no `"auto"` → worker inherited → byte-identical. If it changes → STOP, report.
- [ ] **Step 7: Commit** `feat(dp-cov): auto-size cov batch per DP replica (pin makes the reduce batch-independent; no min-agreement)`.

---

## Task 2: Docs
- [ ] Update the docstrings AND the now-stale comments that assert "DP stays inherited / min-agreement is a later step" (Review N1/N2): the worker comment (~`:1123-1127` + the old `:1183-1187`), and the `_resolve_cov_batch_size` "Multi-GPU auto-raise … (Deferred)" block (~`:426-435`) + its `"auto"` bullet (~`:379-380`,`:408-410`). New copy: DP cov now auto-sizes PER REPLICA when `"auto"` (each replica probes its own pinned-device VRAM); the per-sequence pin makes the key-wise reduce independent of each replica's batch, so NO min(candidate) agreement is needed (supersedes spec §6). `_resolve_cov_batch_size` still returns the inherited int FLOOR (the actual sizing is in `_collect_covariances`/`size_batch`). Default inherited = byte-identical. Live ≥2-GPU speedup deferred. Commit `docs(dp-cov): per-replica auto sizing, pin subsumes min-agreement`.

---

## Out of scope
- ablation_filter / block_refine pins (their own plans next).
- Re-blessing any golden. The LIVE ≥2-GPU speedup validation (deferred).

## After this plan
Standard plan/review → impl/review loops, all-none. Live multi-GPU validation deferred to a real ≥2-GPU cov run.
