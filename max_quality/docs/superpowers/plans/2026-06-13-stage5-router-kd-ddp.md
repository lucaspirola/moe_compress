# Implementation Plan — DistributedDataParallel for Router-KD (Stage 2.5 + Stage 5)

Status: PLAN ONLY — do not implement until reviewed. Branch `feat/stage5-rkd-ddp`,
worktree `/home/lucas/ai/wt-stage5`, code root `max_quality/`.

This feature is for FUTURE runs. It will NOT be injected into the currently-running
live ablation (too risky for a science run). Correctness > speed throughout.

---

## Goal

Add **DistributedDataParallel (DDP)** to the Router-KD training loop (the same code
serves Stage 2.5 heal AND Stage 5 final — `router_kd/stage.py:41`), so training
parallelizes across GPUs at ~linear speedup, while being **RESULT-PRESERVING**: the
average-gradient over the split global batch is mathematically identical to the
single-GPU full-batch gradient (modulo fp reduction order). Training is the pipeline
runtime long pole (real backward + optimizer steps over ~3000 calibration samples,
default 2 epochs), so this is the highest-value multi-GPU win — and the highest risk.

**Result-preservation is the correctness crux.** The vocab-KL loss is a per-token mean
with a *local* denominator (`vocab_kd.py:132`: `(total_kl / max(n_tokens, 1)) * τ²`),
and the optimizer does exactly one `optim.step()` per grad-accum window
(`orchestrator.py:876`). DDP that splits the per-step global batch across replicas and
**all-reduce-AVERAGES** the gradients reconstructs the identical full-batch gradient
**iff every replica has the same token count** — which holds here because calibration
is fully-packed (uniform `L`, asserted at `vocab_kd.py:112-117`) and we force equal
per-rank micro-batches. This is fundamentally different from *raising* `batch_size`
(forbidden — METRIC_PINNED), because DDP keeps the effective global batch FIXED.

Default (1 GPU, or DDP not requested) MUST be the existing single-process path,
**byte-identical** — the Router-KD golden stays green. DDP is strictly opt-in.

---

## Architecture

### The integration problem
The pipeline runs as a SINGLE process: `run_pipeline.py:293-294` (Stage 2.5) and
`run_pipeline.py:328` (Stage 5) call `stage5_router_kd.run(...)` as plain in-process
function calls in one `main()` process. The orchestrator `run()`
(`router_kd/orchestrator.py:158`) is an in-process plain-Python `for epoch / for batch`
loop (`:665` / `:683`) with `(loss/grad_accum).backward()` at `:850`. There is **no
`torch.distributed`, no DDP, no `init_process_group` anywhere in the repo** (verified —
grep returns nothing in production code). DDP would be net-new distributed infra.

### Chosen approach: IN-PROCESS spawn (NOT external torchrun)
The Router-KD stage internally spawns N rank workers via
`torch.multiprocessing.get_context("spawn")` (the **exact precedent** already in the
repo: Stage 3 `covariance_collection.py:1213` `run_dp_covariance_collection` uses raw
`ctx.Process(...).start()/join()` with `CUDA_VISIBLE_DEVICES` pins and a filesystem
reduce — `:1271-1288`). We mirror that spawn style but layer NCCL collectives on top
(`init_process_group("nccl")`). Each worker:
1. bootstraps the process group (`init_process_group`, rank, world_size, NCCL),
2. pins its GPU (`CUDA_VISIBLE_DEVICES` or `torch.cuda.set_device(rank)`),
3. builds the student replica + freezes scope + builds optimizer + wraps in DDP,
4. trains its calibration shard (per-rank batches, global step count),
5. rank-0 performs ALL I/O (checkpoint/log/trackio/best.pt/final export),
6. rank-0's result (the `out_dir` Path) is returned to the parent via a `mp.SimpleQueue`.

We pick in-process spawn (not requiring an external `torchrun`) so `run_pipeline` stays
a single entrypoint — no operator workflow change, no separate launcher.

**Why NOT torchrun:** torchrun re-executes the WHOLE pipeline script N times, which would
re-run Stages 1-4 N times (or require deep refactor to make every prior stage rank-aware).
In-process spawn confines DDP to exactly the Router-KD stage; the rest of the pipeline is
untouched and runs once on the parent.

### Default-OFF contract (backward compat)
A new config knob `stage5_router_kd.ddp` (mapping; default absent → disabled). When
absent / `enabled: false` / resolved `world_size <= 1`, `orchestrator.run` takes the
EXISTING single-process code path **unchanged** (no spawn, no DDP wrap, byte-identical).
DDP engages only when `ddp.enabled: true` AND `world_size >= 2`. This mirrors how
Stage 3 (`multi_gpu.cov_replicas` default 1) and Stage 4 (`multi_gpu.eora_workers`)
gate their multi-GPU paths — Router-KD currently reads NO `multi_gpu` key (grep
confirms), so this is a new, parallel knob.

### Teacher VRAM (the load-bearing obstacle) — DECISION
DDP needs a full student replica per GPU (~50 GB at net-35%). The teacher is the ~70 GB
BF16 uncompressed model (`teacher.py:44-46`). Student (~50) + teacher (~70) = ~120 GB
does NOT co-fit one 80 GB H200 — which is exactly why the live ablation uses
`device_map=balanced` (a memory spill, not a speed choice). Two DDP-compatible teacher
strategies, and which recipe needs which:

- **(A) Precompute teacher-logit cache — the FAITHFUL, IDEAL path for 1-epoch.**
  `TeacherCachePlugin` (`teacher.py:170`) loads a precomputed sidecar, mmap'd read-only,
  shareable across ranks → **zero teacher VRAM per rank**, BF16-faithful,
  result-preserving. This is the clean DDP teacher. BUT it is incompatible with the
  default `epochs=2` paper recipe: `orchestrator.py:607` hard-rejects `epochs>1 + cache`,
  and `rkd_paper_recipe.py:223` clears the cache slot when applying paper dials. So cache
  + DDP is the win **only for a 1-epoch, no-merge-repair run**.

- **(B) 4-bit / FP8 replicated teacher — the QUALITY-TRADE path for multi-epoch.**
  Each rank holds its own student replica + its own quantized teacher
  (`teacher_load_in_4bit=true` → ~18-35 GB, or a `teacher_model_repo` FP8 override →
  ~20-35 GB), so ~50 GB student + ~20-35 GB teacher fits 80 GB. The KD target is then an
  *approximation* of θ_T (`teacher.py:39-55` flags both as project-level, NOT
  paper-sanctioned deviations) → **NOT result-preserving vs the BF16 teacher** → needs
  explicit user sign-off as a quality trade.

**Which the default recipe needs:** the consolidated default is `paper_dials_only`
(`rkd_paper_recipe.py:162`) → `epochs=2`. epochs=2 **disables the cache** (path A is
unavailable). Therefore the default-recipe DDP path REQUIRES path (B) — a quantized
replicated teacher — OR the operator must switch to `epochs=1` to use path (A). The plan
implements BOTH and makes the teacher strategy an explicit, validated precondition
(see Task 7): on `epochs>1` + DDP we require either a quantized teacher
(`teacher_load_in_4bit` or `teacher_model_repo`) configured, or we raise a loud error
naming the two valid choices. We do NOT silently fall back to a BF16 replicated teacher
that would OOM.

### Tech stack
- `torch.distributed` (NCCL backend on GPU; `gloo` backend for the CPU tolerance test).
- `torch.nn.parallel.DistributedDataParallel`.
- `torch.multiprocessing.get_context("spawn")` (Stage-3 precedent) + `mp.SimpleQueue`
  for the rank-0 → parent result handoff.
- No new third-party deps.

### Unwrap chain (critical, recurs at every save/param site)
DDP wraps the student in `.module`; torch.compile wraps in `_orig_mod`. The codebase
already unwraps `_orig_mod` at EVERY save / `named_parameters()` / `iter_moe_layers`
site (`orchestrator.py:328,428,447,869,1056,1218`; `early_stop.py:140,497`;
`teacher.py:588,675,709`; `vocab_kd.py:231`). DDP adds a SECOND wrapper. The ordering
when DDP wraps `torch.compile(student)` is `DDP(compile(model))`, so the unwrap must
peel BOTH: `module` first, then `_orig_mod`. We introduce ONE helper
`_unwrap_student(student)` and replace every ad-hoc `getattr(student, "_orig_mod",
student)` in the Router-KD package with it (Task 2), so there is a single source of
truth and no site is missed.

---

## Result-preservation gate (the acceptance test)

A **tolerance test** (NOT byte-identical — DDP all-reduce reorders fp): run the SAME
tiny Router-KD config (a) single-process and (b) under DDP `world_size=2` on CPU
(`gloo`), with `global_batch=2`, `per_gpu_batch=1`, and assert the final loss trace and
the final trainable (router) weights match within `rel_tol=1e-5, abs_tol=1e-7` (the same
bar the existing golden uses, `test_router_kd_golden_snapshot.py:328-329`). This is the
single most important test; it proves average-gradient DDP ≡ single-GPU at the same
effective batch for this loss. See Task 10.

---

## Hard requirements (the correctness checklist) — each maps to a task

1. Result-preserving via average-gradient DDP at the SAME effective global batch.
   `per_gpu_batch = global_batch / n_gpu`; with grad_accum, co-scale so the effective
   batch is invariant. → Task 4 (shard), Task 5 (DDP wrap + no_sync), Task 10 (gate).
2. Step-count stays GLOBAL (LR schedule keyed to global step). `total_optim_steps`
   computed from the GLOBAL `len(batches)` BEFORE per-rank shard. → Task 4.
3. Rank-0-only I/O (checkpoint, logging, trackio, best.pt, final export). → Task 6.
4. Synchronized early-stop/EMA: all-reduce-MEAN the window loss so every rank decides
   identically; broadcast the stop flag (else ranks desync/deadlock). best.pt write
   rank-0 + reload broadcast. → Task 6.
5. Teacher VRAM: cache (faithful, 1-epoch) vs 4-bit/FP8 replicated (quality trade,
   multi-epoch). Validated precondition. → Task 7.
6. In-process spawn process-group bootstrap + teardown. → Task 3.
7. `.module` unwrap alongside `_orig_mod`. → Task 2.
8. Backward-compat: default path byte-identical; DDP opt-in. → Task 1, Task 9 (golden
   guardrail).

---

## Tasks (bite-sized, TDD)

> Conventions: tests live in `max_quality/tests/`, run from repo root as
> `pytest max_quality/tests/<file>.py -v`. There is no `pytest.ini`/`pyproject`/Makefile;
> `tests/conftest.py:23` inserts `src` on `sys.path`. New tests redeclare local helpers
> verbatim (codebase discipline forbids cross-test imports). Mirror the scaffold in
> `test_router_kd_golden_snapshot.py` / `test_router_kd_orchestrator.py`: `tiny_model`,
> `tiny_config` fixtures (conftest `:167,:173`), monkeypatch `build_calibration_tensor`
> + `load_model` + `_trackio_log` on BOTH `utils.*` and `router_kd.orchestrator`/
> `router_kd.plugins.teacher` (they bind by direct import).

### Task 0 — Config schema + resolver for the new `ddp` knob (no behavior change yet)
**Files:** new `max_quality/src/moe_compress/router_kd/ddp_config.py`.
**Test first:** `max_quality/tests/test_router_kd_ddp_config.py`
- `test_ddp_disabled_by_default`: `DdpConfig.from_config({})` → `enabled is False`,
  `world_size == 1`.
- `test_ddp_explicit_world_size`: `{"stage5_router_kd": {"ddp": {"enabled": True,
  "world_size": 2}}}` → `enabled True, world_size 2`.
- `test_ddp_auto_world_size`: `enabled: true` with no `world_size` resolves to
  `min(requested_or_all, torch.cuda.device_count())` (mock `device_count` → 4 → 4).
- `test_ddp_world_size_capped_by_global_batch`: with `batch_size=2`, a requested
  `world_size=4` is REJECTED (raise `ValueError` naming the METRIC_PINNED cap
  `n_gpu <= global_batch` unless grad_accum is co-scaled) — see §"effective batch fixed".
- `test_ddp_string_false_coerced`: `enabled: "false"` (YAML string) → `False` (mirror
  `AutoBatchConfig.from_dict` coercion, `auto_batch.py:70-71`).
**Implementation:**
```python
# router_kd/ddp_config.py
from __future__ import annotations
import dataclasses

@dataclasses.dataclass(frozen=True)
class DdpConfig:
    enabled: bool = False
    world_size: int = 1
    # Teacher strategy is validated separately (Task 7); kept here for surfacing.
    backend: str = "nccl"  # "gloo" for the CPU tolerance test

    @classmethod
    def from_config(cls, config: dict, *, device_count_fn=None) -> "DdpConfig":
        s5 = config.get("stage5_router_kd", {}) or {}
        raw = s5.get("ddp", {}) or {}
        enabled = (raw.get("enabled") is True) or (
            str(raw.get("enabled", "")).strip().lower() == "true")
        if not enabled:
            return cls(enabled=False, world_size=1)
        if device_count_fn is None:
            import torch
            device_count_fn = torch.cuda.device_count
        avail = int(device_count_fn())
        requested = raw.get("world_size")
        ws = int(requested) if requested is not None else avail
        ws = min(ws, avail) if avail > 0 else ws  # avail==0 only on the CPU test path
        backend = str(raw.get("backend", "nccl"))
        # METRIC_PINNED cap: per_gpu_batch = global_batch / world_size must be an
        # integer >= 1, and the effective batch must stay == single-GPU global_batch.
        global_batch = int(s5["batch_size"])
        if ws > 1 and global_batch % ws != 0:
            raise ValueError(
                f"Router-KD DDP: stage5_router_kd.batch_size={global_batch} is not "
                f"divisible by ddp.world_size={ws}. The effective global batch is "
                "METRIC_PINNED (it determines the trained result); per_gpu_batch = "
                "global_batch / world_size must be an integer. Either set world_size "
                "to a divisor of batch_size, raise batch_size to a multiple of "
                "world_size AND co-scale gradient_accumulation DOWN to keep the "
                "effective batch fixed, or reduce world_size."
            )
        return cls(enabled=(ws > 1), world_size=ws, backend=backend)
```
**Expected:** `pytest max_quality/tests/test_router_kd_ddp_config.py -v` → all pass.
**Note:** when `enabled: true` but resolved `world_size == 1` (e.g. one GPU available),
`enabled` collapses to `False` so the orchestrator takes the single-process path —
backward-compat preserved on single-GPU hosts even with the flag on.

### Task 1 — Orchestrator dispatch fork (single-process vs DDP), default path untouched
**Files:** `router_kd/orchestrator.py` (top of `run`, after `apply_config_overrides`).
**Test first:** `test_router_kd_ddp_dispatch.py`
- `test_default_takes_single_process`: with no `ddp` key, monkeypatch a sentinel onto the
  new `_run_ddp` to assert it is NEVER called; assert the run completes via the existing
  path (reuse the tiny scaffold; assert `out_dir` is the expected `{stage_key}_final`).
- `test_ddp_enabled_dispatches_spawn`: with `ddp.enabled true, world_size 2, backend
  gloo` on CPU, monkeypatch `_spawn_ddp_workers` to a stub returning a fake Path; assert
  it IS called with `world_size=2` and the orchestrator returns that Path.
**Implementation:** extract the entire existing loop body of `run()` (from `s5 =
config[...]` through `return out_dir`) into a new private `_run_single_process(student,
tokenizer, config, artifacts_dir, *, device, no_resume, stage_key, rank=0, world_size=1,
ddp=None)`. The public `run()` becomes:
```python
def run(student, tokenizer, config, artifacts_dir, *, device=None,
        no_resume=False, stage_key="stage5"):
    from ..utils.env_guard import assert_gdn_training_supported
    assert_gdn_training_supported(context=f"Router-KD {stage_key}")
    RkdPaperRecipePlugin().apply_config_overrides(config)
    ddp = DdpConfig.from_config(config)
    if not ddp.enabled:
        # EXISTING single-process path — byte-identical. ddp=None signals "no DDP".
        return _run_single_process(student, tokenizer, config, artifacts_dir,
                                   device=device, no_resume=no_resume,
                                   stage_key=stage_key)
    return _spawn_ddp_workers(tokenizer, config, artifacts_dir,
                              no_resume=no_resume, stage_key=stage_key, ddp=ddp)
```
**CRITICAL backward-compat detail:** `apply_config_overrides` must run BEFORE
`DdpConfig.from_config` (it sets `epochs`, may clear the cache) so the DDP config sees
the effective recipe. The `ddp=None` default of `_run_single_process` means every
`_unwrap_student` / no_sync / all-reduce site (added in later tasks) is a no-op on the
single-process path — guaranteeing byte-identity (see Task 9 guardrail).
**Expected:** `pytest max_quality/tests/test_router_kd_ddp_dispatch.py -v` → pass.

### Task 2 — Single `_unwrap_student` helper; replace all `_orig_mod` peels
**Files:** new `router_kd/_unwrap.py`; edits across `orchestrator.py`, `early_stop.py`,
`teacher.py`, `vocab_kd.py`.
**Test first:** `test_router_kd_unwrap.py`
- `test_unwrap_plain`: a bare `nn.Module` → returned as-is.
- `test_unwrap_compile_only`: object with `_orig_mod` → returns `_orig_mod`.
- `test_unwrap_ddp_only`: object with `.module` → returns `.module`.
- `test_unwrap_ddp_over_compile`: `module._orig_mod` nesting (DDP(compile(m))) → returns
  the innermost `m`.
- `test_unwrap_compile_over_ddp`: `_orig_mod.module` nesting → also returns innermost
  (peel both orders).
**Implementation:**
```python
# router_kd/_unwrap.py
def unwrap_student(model):
    """Peel DDP (.module) and torch.compile (._orig_mod) wrappers, in any order,
    to the underlying nn.Module. Idempotent on a bare module."""
    seen = set()
    while True:
        nxt = getattr(model, "module", None) or getattr(model, "_orig_mod", None)
        if nxt is None or id(nxt) in seen:
            return model
        seen.add(id(model)); model = nxt
```
Then replace EVERY `getattr(student, "_orig_mod", student)` (and the `_t` variants in
teacher.py) in the Router-KD package with `unwrap_student(...)`. Sites to change (verify
each by grep `_orig_mod` in `router_kd/`): `orchestrator.py:149,328,428,447,869,1056,1218`,
`early_stop.py:140,497`, `teacher.py:161,588,675,709`, `vocab_kd.py:231`.
**Guardrail:** because the single-process student has neither `.module` nor (when
compile off) `_orig_mod`, `unwrap_student` returns it unchanged → byte-identical on the
default path. The golden (Task 9) proves this.
**Expected:** `pytest max_quality/tests/test_router_kd_unwrap.py -v` → pass; the existing
golden still green after the mechanical replacement (run Task 9 command).

### Task 3 — Process-group bootstrap + teardown + worker entrypoint skeleton
**Files:** new `router_kd/ddp_runtime.py`.
**Test first:** `test_router_kd_ddp_runtime.py` (CPU, `gloo`)
- `test_bootstrap_teardown_world1`: `_init_pg(rank=0, world_size=1, backend="gloo",
  master_port=<free>)` then `_destroy_pg()`; assert `dist.is_initialized()` toggles
  True→False and `dist.get_rank()==0` while up.
- `test_spawn_two_workers_collect_result`: `_spawn_ddp_workers` with a stub `worker_fn`
  that returns `f"rank{rank}"`; assert the parent receives rank-0's value via the queue
  and both processes exit 0. (Use `backend="gloo"`, `world_size=2`, devices=`["cpu","cpu"]`.)
- `test_worker_nonzero_exit_raises`: a worker that raises → parent re-raises a
  `RuntimeError` naming the failed rank (mirror Stage-3 exit-code check
  `covariance_collection.py:1279-1282`).
**Implementation sketch:**
```python
# router_kd/ddp_runtime.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def _init_pg(rank, world_size, *, backend, master_addr="127.0.0.1", master_port):
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def _destroy_pg():
    if dist.is_initialized():
        dist.destroy_process_group()

def _free_port():
    import socket
    s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close(); return p

def _worker_entry(rank, world_size, backend, master_port, result_q, payload, worker_fn):
    try:
        _init_pg(rank, world_size, backend=backend, master_port=master_port)
        out = worker_fn(rank=rank, world_size=world_size, **payload)
        if rank == 0:
            result_q.put(("ok", out))
    except BaseException as exc:  # noqa: BLE001
        result_q.put(("err", rank, repr(exc)))
        raise
    finally:
        _destroy_pg()

def spawn_ddp_workers(world_size, *, backend, payload, worker_fn):
    ctx = mp.get_context("spawn")
    result_q = ctx.SimpleQueue()
    port = _free_port()
    procs = [ctx.Process(target=_worker_entry,
                args=(r, world_size, backend, port, result_q, payload, worker_fn))
             for r in range(world_size)]
    for p in procs: p.start()
    # Drain rank-0 result / first error.
    status = result_q.get()
    for p in procs: p.join()
    for p in procs:
        if p.exitcode not in (0, None):
            raise RuntimeError(f"Router-KD DDP: a worker exited with code {p.exitcode}")
    if status[0] == "err":
        raise RuntimeError(f"Router-KD DDP rank {status[1]} failed: {status[2]}")
    return status[1]
```
**Note on serialization:** the spawn target and `payload` must be serializable for the
spawn handoff. The student/teacher models are NOT passed through the queue — each worker
loads its own via `load_model` (Stage-3 precedent: workers re-load inside the child,
`covariance_collection.py:1099-1124`). `payload` carries only `config`, `artifacts_dir`,
`stage_key`, `no_resume`, the resolved model repo + revision (so each rank loads the
same student checkpoint Stage 4 produced) and the `DdpConfig`. See Task 8 for how the
student is materialized per rank.
**Expected:** `pytest max_quality/tests/test_router_kd_ddp_runtime.py -v` → pass.

### Task 4 — Per-rank batch shard + GLOBAL step count (result-preserving partition)
**Files:** `router_kd/orchestrator.py` (`_run_single_process` batch setup, ~`:302-332`,
`:683`).
**Test first:** `test_router_kd_ddp_shard.py`
- `test_total_optim_steps_is_global`: with `len(batches)=8, grad_accum=1, epochs=1,
  world_size=2`, assert `total_optim_steps == 8` (NOT 4) — computed from the GLOBAL batch
  count before the shard. (Call the extracted helper directly with a fake DdpConfig.)
- `test_shard_partition_disjoint_and_even`: a `_rank_batch_indices(num_batches=8, rank,
  world_size=2)` helper returns `[0,2,4,6]` for rank 0 and `[1,3,5,7]` for rank 1
  (strided so each grad-accum window draws one sequence per rank from the SAME global
  window — see note). Disjoint, equal length.
- `test_shard_drops_uneven_tail`: `num_batches=7, world_size=2` → each rank gets 3 (the
  trailing odd batch is dropped so token counts stay equal across ranks), and a WARNING
  is logged naming the dropped count (extend the existing `trailing` drop semantics,
  `orchestrator.py:560,589-595`).
**Implementation:**
- `total_optim_steps = (len(batches) // grad_accum) * epochs` STAYS computed from the
  GLOBAL `len(batches)` (it already is — `orchestrator.py:332`; do NOT change it). This
  is the load-bearing fact: every rank computes the identical `total_optim_steps`, so the
  LR schedule (`kd_optimizer.py:262-275`, keyed to `total_optim_steps`) and the global
  `step` counter advance in lockstep across ranks.
- Introduce the partition: the per-rank batch list is `batches[rank::world_size]` (strided)
  so that grad-accum window `w` consumes, across the N ranks, the SAME N×grad_accum
  sequences a single GPU would have consumed in that window. Equal `B_local` per rank →
  equal `n_tokens` per rank → average-gradient ≡ global mean (the `vocab_kd.py:132` mean
  is linear in the per-replica means when token counts are equal).
  - **Divisibility:** require `len(batches) % world_size == 0`; drop the ragged tail
    BEFORE sharding (extend the `trailing` logic) so all ranks get equal length. Log the
    drop. Combined with the existing `len(batches) % grad_accum` trailing drop, this keeps
    each global grad-accum window complete and identical across ranks.
- The `step` counter, the grad-accum boundary test `(i + 1) % grad_accum == 0`
  (`orchestrator.py:852`), and `scheduler.step()` are UNCHANGED — they operate on the
  per-rank loop index `i`, which now ranges over `len(batches)//world_size`, but the
  GLOBAL window cadence is identical because every rank steps on the same boundaries.
**Expected:** `pytest max_quality/tests/test_router_kd_ddp_shard.py -v` → pass.

### Task 5 — DDP wrap (after freeze + optimizer) + grad-accum `no_sync`
**Files:** `router_kd/orchestrator.py` (after `build_optimizer` ~`:340`, and the
microbatch backward `:850`).
**Test first:** `test_router_kd_ddp_wrap.py` (CPU, `gloo`, world_size=2)
- `test_ddp_wrap_after_freeze`: assert the optimizer holds the SAME leaf param tensors as
  `ddp_student.module.parameters()` (DDP must not introduce new params); assert
  `requires_grad` flags are already set (freeze happened first).
- `test_no_sync_on_nonboundary_microbatch`: with `grad_accum=2`, assert
  `ddp.no_sync()` is entered on microbatch 0 and NOT on microbatch 1 (spy on a wrapper);
  assert one all-reduce per grad-accum window, not per microbatch.
- `test_grad_avg_matches_local` (the micro result-preservation unit): on a 1-layer toy
  model, feed rank-0 batch A and rank-1 batch B; after backward+all-reduce, assert each
  rank's `.grad` equals the gradient of `mean(loss(A), loss(B))` within tol — i.e. AVERAGE
  not SUM.
**Implementation:**
- Ordering (per analysis §4): freeze (`setup_trainable_scope`) → build optimizer over the
  raw `requires_grad` params → OPTIONAL `torch.compile` → wrap in DDP. So:
  ```python
  if use_compile:
      student = torch.compile(student, mode="default")
  if ddp is not None:
      student = torch.nn.parallel.DistributedDataParallel(
          student,
          device_ids=([device.index] if (device is not None and device.type=="cuda")
                      else None),  # None for gloo/CPU
          find_unused_parameters=False,  # router-only scope is fully used; see note
          gradient_as_bucket_view=True,
      )
  ```
  The optimizer already holds the leaf params (built before the wrap); DDP wraps the same
  module so its `parameters()` are identical objects — no optimizer rebuild needed
  (analysis §4 ordering).
- **`find_unused_parameters`:** the trainable scope is router-only and every trainable
  param participates in the loss every step, so `False` is correct AND faster. BUT
  merge-repair (Task 12, follow-up) unfreezes expert centroid rows behind a grad-mask —
  if those ever don't receive grad, DDP would error. Since merge-repair DDP is deferred,
  ship with `False` and add a test asserting no "unused parameter" error on the
  router-only path.
- **`no_sync` around grad-accum:** wrap the non-boundary microbatch backward so the
  all-reduce fires only on the boundary microbatch (correctness-AND-throughput; without it
  you all-reduce every microbatch — still correct, just slower):
  ```python
  is_boundary = (i + 1) % grad_accum == 0
  cm = student.no_sync() if (ddp is not None and not is_boundary) else contextlib.nullcontext()
  with cm:
      (loss / grad_accum).backward()
  ```
- **MEAN not SUM:** DDP defaults to averaging gradients across ranks — exactly the
  per-token-mean semantics. Add an assertion/comment that NO manual `loss * world_size`
  rescale exists anywhere (grep guard in the test).
**Expected:** `pytest max_quality/tests/test_router_kd_ddp_wrap.py -v` → pass.

### Task 6 — Synchronized early-stop/EMA + rank-0-only I/O + best.pt broadcast
**Files:** `router_kd/orchestrator.py` (log-window block `:882-1018`, final export
`:1036-1058`); `router_kd/ddp_runtime.py` (add `all_reduce_mean`, `broadcast_flag`,
`broadcast_module_state` helpers).
**Test first:** `test_router_kd_ddp_sync.py` (CPU, `gloo`, world_size=2)
- `test_window_loss_all_reduced_mean`: give rank-0 a window loss of 2.0 and rank-1 of
  4.0; assert the value fed to `update_best_tracker` is 3.0 on BOTH ranks (so both make
  the identical best/EMA/patience decision). This makes the multi-GPU `raw_kl_val`
  equal the single-GPU full-batch window mean (the gate's correctness, analysis §4).
- `test_early_stop_flag_broadcast`: force rank-0's patience to trip while rank-1's local
  view would not; assert BOTH ranks see `early_stop_should_stop=True` after the broadcast
  → both `break` → no deadlock (one rank breaking while others wait on the next all-reduce
  is the classic DDP hang).
- `test_rank0_only_writes`: assert `best.pt`, `step_*.pt`, trackio, and
  `save_compressed_checkpoint` are invoked ONLY on rank 0 (spy; ranks 1+ must not touch
  the filesystem — N ranks racing the same files = torn writes / N× I/O).
- `test_best_reload_broadcast`: rank-0 loads `best.pt`; after
  `broadcast_module_state`, assert all ranks' trainable params are bit-identical to
  rank-0's (so every replica ends with the same exported weights).
**Implementation:**
- `all_reduce_mean(scalar_tensor)`: `dist.all_reduce(t, op=SUM); t /= world_size`. Apply
  to BOTH `loss_val` and `raw_kl_val` at the log boundary (`orchestrator.py:889,893`)
  BEFORE publishing `raw_kl_val` to `run_ctx` for `update_best_tracker`. Do the reduce on
  a GPU/CPU scalar tensor, then `.item()`.
- **Rank-0 gates** (guard with `if rank == 0:`):
  - `_trackio_log(...)` (`:566,943`) — rank-0 only (no-op on other ranks).
  - `_save_best_router_state` via `update_best_tracker` (`early_stop.py:397`) — rank-0
    only. **Subtlety:** `update_best_tracker` MUTATES EMA/patience state that EVERY rank
    must keep identical for the synchronized decision. So: ALL ranks run the EMA/patience
    arithmetic (it is deterministic given the all-reduced `raw_kl_val` — identical inputs
    → identical state), but only rank-0 performs the `best.pt` WRITE. Implement by passing
    `rank` into the ctx and guarding ONLY the `_save_best_router_state` call inside
    `update_best_tracker` (add a `ctx.get("ddp_rank", 0) == 0` guard around
    `early_stop.py:397`). The EMA/`best_step`/`no_improve_windows` updates stay on all ranks.
  - `check_early_stop` runs on all ranks; then `broadcast_flag(early_stop_should_stop,
    src=0)` so the BREAK decision is unanimous. (Because the all-reduced `raw_kl_val` is
    identical on all ranks, the flag is ALSO identical pre-broadcast — the broadcast is a
    belt-and-suspenders guard against any divergence in the float-compare; keep it.)
  - `_save_stage5_checkpoint` (`:977-988`, `:1007-1017`) + `_Stage5CheckpointWriter`
    (`:660-664`) — rank-0 only. Other ranks skip checkpoint writes entirely.
  - `save_compressed_checkpoint` (`:1053`) — rank-0 only; unwrap via
    `unwrap_student(student)` (now peels `.module` then `_orig_mod`).
- **best.pt reload broadcast** (`reload_best_checkpoint`, `early_stop.py:467`): rank-0
  loads `best.pt` + swaps params; then `broadcast_module_state(student, src=0)` so every
  replica's trainable params match rank-0 before export. (Alternative: all ranks load the
  same `best.pt` from the shared filesystem — but rank-0-load+broadcast avoids a read race
  and is the clean idiom; analysis §4.)
- **Resume + DDP:** rank-0 loads the resume checkpoint (`orchestrator.py:403-511`) and the
  DDP construction broadcasts module state from rank-0 on wrap (DDP broadcasts on
  construction). So restore on rank-0 BEFORE the DDP wrap; the existing multi-device
  optimizer-state migration (`:469-481`, device_map-era) is bypassed on the DDP path (each
  rank loads the same optim state independently — see Task 8). Keep the old block as the
  non-DDP fallback.
**Expected:** `pytest max_quality/tests/test_router_kd_ddp_sync.py -v` → pass.

### Task 7 — Teacher VRAM strategy: validated precondition + per-rank teacher
**Files:** new validation in `router_kd/orchestrator.py` (DDP branch, before spawn) or
`ddp_config.py`; uses existing `teacher.py` paths unchanged.
**Test first:** `test_router_kd_ddp_teacher.py`
- `test_ddp_epoch1_allows_cache`: `epochs=1` + `teacher_logits_cache` set + DDP → OK
  (path A; the cache is read-only mmap, shareable across ranks → zero per-rank teacher
  VRAM, BF16-faithful).
- `test_ddp_multiepoch_requires_quantized_teacher`: `epochs=2` (paper default) + DDP +
  NO `teacher_load_in_4bit` and NO `teacher_model_repo` → RAISE, error message naming the
  two valid choices (4-bit OR FP8 repo) and stating BF16 replicated teacher will OOM
  (~50 student + ~70 teacher > 80 GB) and is a quality trade needing sign-off.
- `test_ddp_multiepoch_4bit_ok`: `epochs=2` + `teacher_load_in_4bit: true` + DDP → OK
  (path B, quality-trade, allowed because explicitly configured).
- `test_ddp_cache_rejected_multiepoch`: `epochs=2` + cache set → the EXISTING
  orchestrator guard (`:607`) already raises; assert it still fires under DDP (cache is
  cleared by paper-dials anyway, `rkd_paper_recipe.py:223`).
**Implementation:** a `validate_ddp_teacher_strategy(config, ddp)` called in the DDP
branch before spawn:
```python
s5 = config["stage5_router_kd"]
epochs = int(s5["epochs"])
has_cache = bool(s5.get("teacher_logits_cache"))
quantized = bool(s5.get("teacher_load_in_4bit")) or bool(s5.get("teacher_model_repo"))
if epochs == 1 and has_cache:
    return  # path A — faithful, zero per-rank teacher VRAM
if epochs > 1 and not quantized:
    raise RuntimeError(
        "Router-KD DDP with epochs>1 (the paper_dials_only default) cannot use the "
        "teacher-logit cache (orchestrator rejects epochs>1 + cache) and a BF16 "
        "replicated teacher (~70 GB) will not co-fit a student replica (~50 GB) on "
        "one 80 GB card. Choose ONE: (A) set epochs=1 and configure "
        "stage5_router_kd.teacher_logits_cache (faithful, zero per-rank teacher VRAM); "
        "or (B) set stage5_router_kd.teacher_load_in_4bit=true OR teacher_model_repo "
        "(FP8) — a QUALITY TRADE (KD target is an approximation of theta_T, NOT "
        "result-preserving vs BF16) that needs explicit sign-off.")
# epochs==1 without cache + quantized/BF16: each rank loads its own teacher (Task 8);
# allowed (BF16 only if it fits — operator's call).
```
Per rank, the teacher is materialized by the EXISTING `TeacherLivePlugin._load_teacher`
(`teacher.py:562`) inside the worker — its device-map derivation (`:622-641`) already
co-locates the teacher with the student device, which under DDP is the rank's device. The
cache path (`TeacherCachePlugin`) is read-only mmap — naturally shared across ranks on one
node. No teacher code changes needed; only the precondition + per-rank load wiring (Task 8).
**Expected:** `pytest max_quality/tests/test_router_kd_ddp_teacher.py -v` → pass.

### Task 8 — `_run_ddp_worker`: assemble the per-rank training (the integration)
**Files:** `router_kd/orchestrator.py` (`_spawn_ddp_workers` + `_run_ddp_worker`).
**Test first:** covered end-to-end by Task 10 (the result-preservation gate). Add a
focused `test_ddp_worker_loads_own_student.py`:
- `test_each_rank_loads_student`: monkeypatch `load_model` to count calls per rank; assert
  each of the 2 ranks loads the student once (re-load-in-child pattern, Stage-3 precedent).
- `test_rank0_returns_out_dir`: stub the loop; assert rank-0 puts the `{stage_key}_final`
  Path on the result queue and parent returns it.
**Implementation:**
```python
def _spawn_ddp_workers(tokenizer, config, artifacts_dir, *, no_resume, stage_key, ddp):
    validate_ddp_teacher_strategy(config, ddp)
    # Resolve the student source so each rank loads the SAME checkpoint Stage 4 wrote.
    student_src = config["model"]["name_or_path"]  # or the Stage-4 output dir, per caller
    payload = dict(config=config, artifacts_dir=str(artifacts_dir),
                   no_resume=no_resume, stage_key=stage_key,
                   student_src=student_src, ddp_world_size=ddp.world_size,
                   backend=ddp.backend)
    return spawn_ddp_workers(ddp.world_size, backend=ddp.backend,
                             payload=payload, worker_fn=_run_ddp_worker)

def _run_ddp_worker(*, rank, world_size, config, artifacts_dir, no_resume, stage_key,
                    student_src, ddp_world_size, backend):
    import torch
    device = (torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu"))
    # Re-load the student inside the child (Stage-3 precedent; models aren't shipped).
    student, tokenizer = load_model(student_src, ...)  # mirror run_pipeline load
    ddp_cfg = DdpConfig(enabled=True, world_size=world_size, backend=backend)
    out = _run_single_process(student, tokenizer, config, Path(artifacts_dir),
                              device=device, no_resume=no_resume, stage_key=stage_key,
                              rank=rank, world_size=world_size, ddp=ddp_cfg)
    return out  # only rank 0's is consumed by the parent (others return same Path)
```
`_run_single_process` is the extracted loop (Task 1) threaded with `rank`/`world_size`/
`ddp`: it publishes `ddp_rank` on `run_ctx` (for the best.pt write guard, Task 6), shards
batches (Task 4), wraps in DDP (Task 5), all-reduces the window loss + gates I/O (Task 6).
**Serialization:** `config` (nested dicts/lists/scalars), `artifacts_dir` (str),
`student_src` (str) carry across the spawn boundary fine. The student model + tokenizer
are NOT passed — loaded in-child. `worker_fn` is a module-level function.
**Expected:** `pytest max_quality/tests/test_ddp_worker_loads_own_student.py -v` → pass.

### Task 9 — DEFAULT-PATH GOLDEN GUARDRAIL (byte-identical, non-negotiable)
**Files:** none new — re-run the existing golden after Tasks 1-8 land.
**Command:**
```
pytest max_quality/tests/test_router_kd_golden_snapshot.py -v
```
**Expected output:** both params (`stage2p5`, `stage5`) PASS. The golden asserts
`compressed_metadata.{stage_id}.json` BYTE-identical (`:306`) and the loss trace within
`rel_tol=1e-5, abs_tol=1e-7` (`:328-329`) — both on the DEFAULT (no-`ddp`-key,
`rkd_recipe="current"`) single-process path. This proves the mechanical `_unwrap_student`
replacement (Task 2) and the dispatch fork (Task 1) did NOT perturb the existing path.
**If it fails:** the single-process path was changed — STOP. Do not widen the tolerance
(the golden docstring `:23-31` forbids masking same-machine drift). Root-cause the
regression (most likely an `unwrap_student` site that changed behavior, or `ddp=None` not
short-circuiting a new code branch).
Also run the full Router-KD suite to catch seam regressions:
```
pytest max_quality/tests/test_router_kd_orchestrator.py \
       max_quality/tests/test_router_kd_plugin_optimizer.py \
       max_quality/tests/test_router_kd_plugin_early_stop.py \
       max_quality/tests/test_router_kd_plugin_vocab_kd.py \
       max_quality/tests/test_router_kd_teacher_slot.py \
       max_quality/tests/test_smoke_stage5_resume.py -v
```
**Expected:** all pass (no behavior change on the default path).

### Task 10 — RESULT-PRESERVATION GATE (1-GPU vs 2-GPU tolerance test) — THE acceptance test
**Files:** `max_quality/tests/test_router_kd_ddp_result_preserving.py` (CPU, `gloo`).
**Test:**
- `test_ddp2_matches_single_process`:
  1. Build `tiny_model` + `tiny_config` (conftest `:167,:173`); set
     `stage5_router_kd.batch_size=2, gradient_accumulation=1, epochs=1,
     max_calibration_samples=8, rkd_recipe="current"` (defeat paper-dials so epochs stays
     1 and we can use a deterministic single-process baseline), `log_every_n_steps=1`,
     teacher==student via the `load_model` monkeypatch (golden `:208-222`).
  2. Run (a) single-process (no `ddp` key) → capture the final trainable (router) state
     dict + the captured loss trace (via the `_trackio_log` capture monkeypatch).
  3. Run (b) `ddp: {enabled: true, world_size: 2, backend: "gloo"}`,
     `batch_size=2` → `per_gpu_batch=1` → capture rank-0's final exported router state +
     loss trace. (DDP spawns 2 CPU workers; rank-0 writes the final checkpoint.)
  4. Assert: per-step `loss` and `raw_kl` match within `math.isclose(rel_tol=1e-5,
     abs_tol=1e-7)` (the golden bar); the final router weight tensors match within
     `torch.allclose(rtol=1e-5, atol=1e-7)`. NOT byte-identical (DDP all-reduce reorders
     fp — documented).
**Why this is the crux:** it empirically proves average-gradient DDP at the same effective
batch ≡ single-GPU full-batch for the vocab-KL loss. If it fails, the result-preservation
claim is false and DDP must NOT ship.
**Determinism notes for the test:** seed everything (`torch.manual_seed`), force CPU
single-thread (`torch.set_num_threads(1)`) so the AdamW float math is reproducible across
the two runs (mirror the golden's same-machine caveat). Use `gloo` (NCCL needs CUDA).
**Command:** `pytest max_quality/tests/test_router_kd_ddp_result_preserving.py -v`
**Expected:** PASS within tolerance.

### Task 11 — Deadlock / failure-mode tests (defensive)
**Files:** `test_router_kd_ddp_failure_modes.py` (CPU, `gloo`).
**Tests:**
- `test_uneven_batch_count_drops_tail_no_hang`: `max_calibration_samples` giving
  `num_batches=7, world_size=2` → tail dropped, run completes, no hang.
- `test_worker_exception_propagates`: inject a NaN-loss on rank-1 (the NaN tripwire,
  `orchestrator.py:815`) → the parent re-raises naming the rank; no orphaned processes
  (assert all child exitcodes set).
- `test_early_stop_no_desync`: `early_stop_patience>0`, force a trip → both ranks break,
  process group destroyed cleanly.
**Command:** `pytest max_quality/tests/test_router_kd_ddp_failure_modes.py -v`
**Expected:** PASS.

### Task 12 — (FOLLOW-UP, OUT OF THIS PLAN's critical path) merge-repair under DDP
Ship DDP FIRST without merge-repair (it is Stage-2.5-only, opt-in, default-OFF —
`MergeRepairPlugin`, `merge_repair.py`). When added later: the forward-capture hooks
(`merge_repair.py:290`) must attach to `unwrap_student(student)` (the DDP `.module`), not
the wrapper; the grad-mask `register_hook` and the mean-MSE term are DDP-safe (deterministic
mask, equal `B_local`); `find_unused_parameters` may need re-evaluation if masked rows ever
receive no grad. Tracked as a separate plan. **This plan asserts merge-repair + DDP raises a
clear "not yet supported" error** (add a guard in the DDP branch when
`merge_repair.enabled` at stage2p5).

---

## Out of scope
- **Merge-repair (Direction E) under DDP** — deferred to a follow-up (Task 12 guards it
  off with a clear error). Stage-2.5-only, opt-in, default-OFF.
- **Multi-node DDP** — single-node, multi-GPU only (one `MASTER_ADDR=127.0.0.1`). The
  in-process spawn is single-host by construction.
- **Raising the effective batch / changing the trained result** — forbidden
  (METRIC_PINNED). DDP keeps the global batch fixed; that is the whole point.
- **Auto-batch as a maximizer** — Router-KD is METRIC_PINNED; the per-GPU batch is fixed
  by the science (`global_batch / world_size`), NOT VRAM-auto-sized. Auto-batch's role
  here is an OOM FLOOR / admission check only, and even the OOM-halve is NOT freely usable
  (halving a metric-pinned batch changes the result) — resolve an OOM via more GPUs or
  higher grad_accum, never by resizing the batch. Router-KD is already NOT wired to
  auto_batch (grep confirms); keep it so. No change to `utils/auto_batch.py`.
- **torchrun / external launcher** — rejected in favor of in-process spawn so the pipeline
  stays one entrypoint.
- **FP8/4-bit teacher quality validation** — the plan gates it behind an explicit operator
  opt-in + sign-off; measuring the KD-quality delta is a separate science task.
- **Pipeline-parallel teacher service (option B-hybrid)** — analysis §5 documents it; we
  pick replicate-teacher (cache or quantized) instead. Not implemented.
- **Live ≥2-GPU validation on real H200** — deferred to the first real run (the CPU `gloo`
  tolerance gate is the CI proof; a live NCCL run validates on hardware), consistent with
  how Stage-3/Stage-4 multi-GPU landed (1-GPU golden + deferred live validation).

---

## Risk register (the gotcha checklist, each mapped to a code site + task)
| Gotcha | Site | Mitigation (task) |
|---|---|---|
| Global step count must stay global | `orchestrator.py:332`, `kd_optimizer.py:262` | Compute from global `len(batches)` pre-shard (Task 4) |
| Effective batch fixed (METRIC_PINNED) | `orchestrator.py:302` | `per_gpu = global/world_size`; reject non-divisible (Task 0/4) |
| Grad all-reduce MEAN not SUM | DDP default | No manual `*world_size` rescale; unit test (Task 5) |
| Equal token count per rank | `vocab_kd.py:119,132` | Drop ragged tail; strided even shard (Task 4) |
| Early-stop desync → deadlock | `orchestrator.py:998,1024`; `early_stop.py` | All-reduce window loss + broadcast flag (Task 6) |
| Rank-0-only I/O | `:566,943,977,1053,1130`; `early_stop.py:397` | rank guards (Task 6) |
| Resume + DDP | `:403-511,469-481` | rank-0 load → DDP broadcast on wrap (Task 6) |
| Teacher cache index is GLOBAL | `teacher.py:498` | strided shard preserves global `batch_index`; pass global index (Task 4/8) |
| `.module` + `_orig_mod` unwrap | many sites | single `unwrap_student` helper (Task 2) |
| compile + DDP ordering | `:359-368` | pin `DDP(compile(m))` (Task 5) |
| no_sync on grad-accum | `:850` | `no_sync()` on non-boundary microbatch (Task 5) |
| Default path must stay byte-identical | whole loop | `ddp=None` short-circuit + golden guardrail (Task 1/9) |

---

## Open questions (need a decision before / during implementation)
1. **Default recipe is `epochs=2` → no cache → the FAITHFUL teacher path (A) is
   unavailable.** Does the operator accept the 4-bit/FP8 quality trade (path B) for the
   default multi-epoch DDP run, OR should the default DDP run pin `epochs=1` to use the
   faithful cache? The plan REQUIRES an explicit choice (Task 7 raises otherwise) but does
   not pick the project default. **Recommendation: for the FAITHFUL result, run DDP only
   at epochs=1 + cache; use 4-bit teacher DDP only with sign-off.**
2. **Student source per rank.** Task 8 re-loads the student in each child from
   `config["model"]["name_or_path"]`. But at Stage 5 the student is the Stage-4 OUTPUT
   (post-EoRA), not the original checkpoint — confirm the caller passes the Stage-4 output
   dir as the student source (or that each rank can reconstruct it). The single-process
   path receives the live `student` object; DDP must materialize the SAME weights per rank.
   **Needs the run_pipeline wiring decision** (does `_spawn_ddp_workers` get a path, or do
   we save+reload the live student to a temp dir first?).
3. **`world_size > global_batch`.** With the paper `batch_size=2`, DDP caps at
   `world_size=2` unless `grad_accum` is co-scaled down (keeping effective batch fixed).
   Should the plan implement automatic grad_accum co-scaling, or just reject and require
   the operator to raise `batch_size` to a multiple of `world_size`? (Task 0 currently
   REJECTS; co-scaling is a possible enhancement but adds result-preservation surface.)
4. **NCCL determinism for the LIVE result-preservation check.** The CPU `gloo` gate
   (Task 10) proves the math; a live NCCL run on real H200s should be spot-checked
   (1-GPU vs 2-GPU final loss within tolerance) before trusting a production science run.
   Defer to first real run, or block on a dedicated 2-GPU validation?
5. **`find_unused_parameters=False`** assumes every trainable router param participates
   every step. Confirm no architecture variant (e.g. a layer whose router is frozen by a
   `frozen_name_patterns` quirk) leaves a trainable-but-unused param — would error under
   DDP. (Router-only scope makes this unlikely; verify on the target model.)
