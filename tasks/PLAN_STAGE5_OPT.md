# Stage-5 (Router KD) — Tier-1 Throughput / Memory Optimization Plan

**Status:** PLAN ONLY — no production edits in this branch.
**Base:** `origin/main` @ `588ec5e`.
**Stage 5 = the `router_kd/` package** (`router_kd/orchestrator.py`, `router_kd/plugins/teacher.py`, …);
`stage5_router_kd.py` is a thin shim. All file paths below are under
`max_quality/src/moe_compress/`.

All three levers are **Tier-1: byte-identical / golden-safe**. They are
throughput/memory wins on a GPU-bound training stage and MUST NOT change any
trained numerics. **Zero golden re-bless is permitted** — if any golden drifts,
the lever is wrong and must be reverted, not re-blessed.

Explicitly **out of scope** (do NOT touch in this work): the bf16 loss kernel
(Tier-2, user-held — 7%/23% numeric divergence, changes trained weights), fused
KL (skipped), `kd_seq_chunk_size` (= 32, already optimal), window-loss
accumulation, teacher caching, and the dataloader (32 KB batches → host transfer
negligible; the one host lever is gated on a disabled teacher-cache).

---

## 0. Verification of the research numbers against real blobs (`588ec5e`)

All sites below were re-read from `git show origin/main:…` and confirmed. Where
the research line numbers had drifted, the **verified** numbers are recorded.

| Item | Research said | Verified @ `588ec5e` |
|------|---------------|----------------------|
| Teacher forward | `teacher.py:~763` | `teacher.py:763` — `out = teacher(input_ids=input_ids)` inside `with torch.no_grad():` (762), in `TeacherLivePlugin.provide_teacher_logits` (def @ 720). No `attention_mask`, no `use_cache`. |
| Student forward | `orchestrator.py:~733` | `orchestrator.py:733` — `student_out = student(input_ids=batch)`. No `use_cache`. |
| grad-norm metric | `orchestrator.py:~828-833` | `orchestrator.py:828-834` — `grad_norm = float(clip_grad_norm_([p for p in _params_for_norm.parameters() if p.requires_grad and p.grad is not None], float('inf')))`, inside the `if (i+1) % grad_accum == 0:` block (825). |
| grad-norm consumption | `~:884/:898` | **Only** `884/886` (`log.info`) and `898` (trackio payload) — both **inside** the `if step % log_every_n_steps == 0:` block (840). `grep -n grad_norm orchestrator.py` → exactly lines 828, 884, 886, 898. **Nothing else reads it.** |
| Checkpoint glue | `~:928-938` + `_save_stage5_checkpoint ~:1004-1067` | Periodic save call @ `928-939`; early-stop save call @ `957-968`; `_save_stage5_checkpoint` def @ `1004`, body `.cpu().clone()` @ `1026`, `torch.save` @ `1055`, two `os.fsync` @ `1058`/`1064`, `os.replace` @ `1062`. |
| Final export (golden target) | n/a | `save_compressed_checkpoint(...)` @ `994` writes `{stage_key}_final/compressed_metadata.json` — the byte-pinned golden artifact. |

**Golden contract (read from `tests/test_router_kd_golden_snapshot.py`):**
- pins `compressed_metadata.json` **byte-for-byte** (`read_bytes()` compare, line 299);
- pins a loss trace of `{step, loss, raw_kl}` with `rel_tol=1e-5, abs_tol=1e-7`
  (lines 314-321). `loss`/`raw_kl` are window means of the `loss` / `kl_loss`
  tensors. **`grad_norm` is NOT in the pinned trace** — only `step/loss/raw_kl`
  (line 269). Parametrized over `["stage2p5", "stage5"]`.
- `tests/test_stage5_merge_repair.py` — unit tests of merge-map / freeze-set /
  gradient-mask / MSE; none of the three levers touch that surface, but it is in
  scope as a must-stay-green regression.

Implication: Levers B and C cannot affect the golden by construction (grad_norm
is unpinned; checkpoint bytes are not compared by the golden). **Lever A is the
only one that touches numerics-producing forwards**, so its `diff = 0.0` claim
must be re-proven on the golden fixture, not just asserted.

---

## Lever A — `use_cache=False` on both forwards (Tier-1, byte-identical)

### Files / functions
- `router_kd/plugins/teacher.py` — `TeacherLivePlugin.provide_teacher_logits`,
  line **763**.
- `router_kd/orchestrator.py` — `run()` training loop, line **733**.

### Before → after
Teacher (`teacher.py:763`):
```python
# before
out = teacher(input_ids=input_ids)
# after
out = teacher(input_ids=input_ids, use_cache=False)
```
Student (`orchestrator.py:733`):
```python
# before
student_out = student(input_ids=batch)
# after
student_out = student(input_ids=batch, use_cache=False)
```

### Why
Both are single full-sequence forwards; the returned `logits` are consumed and
the model output object is discarded (`del student_out` @ 735; teacher returns
`out.logits.detach()`). The KV cache is **never read** — there is no incremental
decode. HF defaults `use_cache=True`, which allocates a `past_key_values` buffer
each forward. Research measured the freed allocation at ~0.86 GB/forward at
production scale (B=8, L=512, 40 layers, bf16) and logits max-abs-diff = 0.0.
Passing `use_cache=False` suppresses the allocation. The KV cache does not feed
into the logits math for a single full-sequence pass → output is bit-identical.

Note: the freed VRAM *could* later enable a larger training batch (a real
throughput multiplier), but that is a **separate config decision** and is NOT
part of this change. This change only removes the wasted allocation.

### Risk
- Some HF configs *require* `use_cache=True` when gradient-checkpointing is on
  (they emit a warning and silently flip it). Verify the student model does not
  use gradient checkpointing under the Stage-5 config; if it does, confirm the
  flag interaction is a no-op warning, not a behavior change.
- A custom model `forward` that does not accept `use_cache` would raise
  `TypeError`. Both teacher and student are HF `*ForCausalLM` instances whose
  `forward` accepts `use_cache`; confirm during smoke.

### Test that proves it (the load-bearing one for Tier-1)
1. **diff = 0.0 re-proof on the golden fixture.** Before editing, run the golden
   once to confirm it is green at `588ec5e`. After editing, re-run; the
   byte-identical `compressed_metadata.json` + `rel_tol=1e-5` loss trace passing
   *is* the diff=0.0 proof on the exact fixture the project pins:
   ```
   pytest max_quality/tests/test_router_kd_golden_snapshot.py -v
   ```
   Both `stage2p5` and `stage5` params must pass with **no** `MOE_REGEN_GOLDEN`.
2. (Optional, stronger) a throwaway local assertion (not committed) that, on the
   tiny golden model, `model(input_ids).logits` is `torch.equal` with and
   without `use_cache=False`. The golden pass subsumes this.

---

## Lever B — gate the grad-norm metric behind the log window (Tier-1, byte-identical)

### File / function
`router_kd/orchestrator.py` — `run()` training loop, the optimizer-step block
`if (i + 1) % grad_accum == 0:` (line 825); `grad_norm` computed @ **828-834**.

### Before → after (sketch)
Today (verified): `grad_norm` is computed **every** optimizer step (828-834),
then `optim.step()` / `zero_grad()` / `scheduler.step()` / `step += 1`, then it
is read **only** inside `if step % log_every_n_steps == 0:` (840) at 886 + 898.

Plan: compute `grad_norm` only on log-window steps. Move the
`clip_grad_norm_(..., inf)` call inside the `if step % log_every_n_steps == 0:`
block, **after** `step` has been incremented (so the modulo test is against the
same `step` used for the log line). Sketch:
```python
if (i + 1) % grad_accum == 0:
    optim.step()
    optim.zero_grad()
    scheduler.step()
    step += 1
    _rt_update(...)
    if step % config["logging"]["log_every_n_steps"] == 0:
        _params_for_norm = getattr(student, "_orig_mod", student)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in _params_for_norm.parameters()
             if p.requires_grad and p.grad is not None],
            float('inf'),
        ))
        ...  # existing window-mean code, log.info, trackio payload
```
**Ordering caveat to resolve during implementation:** `clip_grad_norm_` reads
`.grad`. The verified order today is *grad-norm BEFORE `optim.zero_grad()`* — so
grads are still populated. If the call is moved to **after** `optim.step()`/
`zero_grad()`, the grads are gone and the norm would be ~0/stale. **Therefore the
gated computation must run BEFORE `optim.zero_grad()` clears grads.** Two valid
shapes:
  - (preferred) keep the `step += 1` increment first, but guard with the
    *predicted* next step value: compute the gate as
    `(step + 1) % log_every_n_steps == 0` evaluated **before** `step += 1` and
    before `zero_grad()`, i.e. wrap only the `clip_grad_norm_` in
    `if (step + 1) % log_every_n_steps == 0:` placed where the unconditional
    computation is today (before `optim.step()`), then reuse the value below; OR
  - capture `.grad` norms before `zero_grad()` into `grad_norm` guarded by the
    same window predicate, and read it in the existing log block.

The implementer MUST pick whichever keeps the `grad_norm` value reported on a
given window **identical** to today's (today: the norm of the grads that
produced *that window's* `optim.step()`). Verify by temporarily logging
`grad_norm` every step under both versions on the tiny model and confirming the
window-boundary values match. This is a reporting-fidelity check, not a
golden-pinned value.

### Why
`clip_grad_norm_(..., inf)` with `max_norm=inf` is a **pure metric**: the
clip-coefficient is `max_norm / (total_norm + eps)`; with `max_norm=inf` the
coefficient is `inf`, which is `>= 1`, so PyTorch's guard
(`if clip_coef < 1: grads.mul_(clip_coef)`) skips the in-place scale — grads are
untouched. Computing it every step forces a device→host sync (the `float(...)`)
plus a full `parameters()` comprehension every optimizer step, but the value is
consumed only on log windows. Gating it removes (window-size − 1)/window-size of
those syncs and comprehensions.

### Risk
- **Low.** The only behavioral surface is the *reported* `grad_norm` value and
  the timing of a device sync. The numerically-trained weights are unchanged
  (inf-clip is a no-op). The golden does not pin `grad_norm`.
- The one real trap is the ordering relative to `zero_grad()` (see caveat). Get
  it wrong and the logged `grad_norm` becomes ~0 — a cosmetic regression, caught
  by the reporting-fidelity check above.

### Test that proves it
```
pytest max_quality/tests/test_router_kd_golden_snapshot.py -v   # byte-identical, no regen
pytest max_quality/tests/test_stage5_merge_repair.py -v
```
Plus the manual window-boundary `grad_norm` parity check described above.

---

## Lever C — async checkpoint (Tier-1)

### Files / functions
`router_kd/orchestrator.py` — `_save_stage5_checkpoint` (def @ **1004**, body
1026-1064) and its two call sites: periodic @ **928-939**, early-stop @
**957-968**; plus the post-save prune (`sorted(partial_dir.glob("step_*.pt"))` +
`unlink`) @ **942-948**; and the post-loop teardown/final-export region
(`teardown_merge_repair` @ ~986, `save_compressed_checkpoint` @ **994**).

### Before → after (design)
Today `_save_stage5_checkpoint` runs entirely on the training thread:
`.cpu().clone()` of router params (1026-1030) → `torch.save` to `*.pt.tmp`
(1055) → `os.fsync(file)` (1058) → `os.replace` (1062) → `os.fsync(parent_dir)`
(1064). Research measured ~815 ms blocking × ~3/run ≈ 2.4 s.

Plan — **split synchronous snapshot from asynchronous write**:
1. **Synchronous, on the training thread (must stay sync for a consistent
   snapshot):**
   - the `router_state = {name: p.data.cpu().clone() …}` dict (already a CPU
     copy — safe to hand to another thread);
   - **`optim.state_dict()` is NOT a CPU copy.** It returns references to the
     **live GPU optimizer moment tensors**. If `torch.save` is backgrounded, the
     next `optim.step()` mutates those tensors → torn/garbage checkpoint. So the
     sync part MUST deep-CPU-copy the optimizer state (e.g. recursively
     `.detach().cpu().clone()` every tensor in `optim.state_dict()`), and the
     scheduler/best-tracker scalars (already plain Python). Build the **entire
     `payload` dict from CPU/host values synchronously**, then hand the finished
     `payload` to the writer thread.
2. **Asynchronous, on a background writer thread:** `torch.save(payload, tmp)` +
   `os.fsync(file)` + `os.replace` + `os.fsync(parent_dir)` + the prune
   (`glob`/`unlink`) — *or* keep the prune synchronous but make it
   race-safe (see below). Bytes written are identical to today's.

### Thread lifecycle / correctness (must-address)
- **Single writer, no overlap.** Use one persistent worker (a `threading.Thread`
  draining a `queue.Queue(maxsize=1)`) or a one-slot executor. Before enqueuing a
  new save, **block-join the previous one** (or rely on `maxsize=1` `put()`
  blocking). Two checkpoints must never write concurrently. At production cadence
  (~3 saves/run, hundreds of steps apart) the previous save is long done, so the
  join is effectively free — but it MUST be there for correctness.
- **Join on shutdown / finalize.** Before the post-loop work that depends on the
  partial dir being settled — specifically **before `save_compressed_checkpoint`
  @ 994** and before `run()` returns @ 1001 — drain/join the writer so (a) all
  step checkpoints are durably on disk if the process exits, and (b) no pending
  write races the final export. Also join on the early-stop `break` path (957-968
  writes a final checkpoint, then breaks) — that save should complete before the
  function proceeds to export. Simplest: a `finally:` around the epoch loop that
  joins the writer; the early-stop checkpoint can be written **synchronously**
  (it is the last one and we want it durable before break) while periodic ones go
  async.
- **Error propagation.** A `torch.save`/`fsync` failure on the worker thread must
  not be swallowed. Capture the exception on the worker and re-raise it on the
  training thread at the next enqueue **and** at the final join (so a failed
  checkpoint halts the run rather than silently losing crash-resume state). Log
  with the same `log.info`/`log.error` channel.
- **Prune race.** The post-save prune (942-948) lists `step_*.pt` and unlinks all
  but the newest two. If the just-queued save has not yet written its `step_N.pt`
  (only the async writer creates it), the prune sees an older set. Resolve by
  **doing the prune inside the writer thread, after `os.replace`** (so the new
  file exists before the glob), keeping the "keep newest two" semantics. This
  also keeps prune off the training thread.
- **`.tmp` cleanup invariant.** Run-start already cleans stale `*.tmp`
  (387-390). The async writer still writes `*.pt.tmp` → `os.replace`; on a crash
  mid-write a stale `.tmp` is left, cleaned next run. Unchanged.

### Why
Removes ~815 ms × ~3 of training-thread blocking per run by overlapping the
serialize+fsync with subsequent training steps. The synchronous CPU snapshot
preserves a consistent point-in-time state; only the disk write moves off-thread.

### Risk + rollback
- **Highest-risk lever** (concurrency). Failure modes: torn optimizer state (if
  the deep-CPU-copy is missed — the single most important detail), overlapping
  writes, a swallowed write error losing crash-resume, or a pending write racing
  the final export.
- **Rollback:** the change is contained to `_save_stage5_checkpoint` + a small
  amount of orchestrator glue (writer creation + join). Rollback = revert to the
  synchronous body. Provide a kill-switch env/config (e.g.
  `STAGE5_ASYNC_CKPT=0`) that forces the synchronous path, so a production run
  can disable async without a code revert.
- **Resume correctness is the real acceptance gate**, not the golden (the golden
  never exercises mid-run checkpoints). See testing plan.

### Test that proves it
- `pytest max_quality/tests/test_smoke_stage5_resume.py -v` — exercises the
  write→resume round trip; the resumed optimizer/router/scheduler state must
  match (this catches a torn optimizer snapshot).
- `pytest max_quality/tests/test_stage5_early_stop.py -v` — exercises the
  early-stop final-checkpoint path (the synchronous-on-break case).
- `pytest max_quality/tests/test_router_kd_golden_snapshot.py -v` — must stay
  byte-identical (sanity: checkpoint changes don't leak into the export).
- A targeted test (add only if the existing resume smoke does not already force
  it): save async, immediately mutate `optim`/router params on the GPU, join the
  writer, reload the checkpoint, assert reloaded state == the **pre-mutation**
  snapshot. This is the torn-snapshot guard.

---

## Ordering of the three levers

1. **Lever A** first — smallest, two-line, the only one touching numerics-
   producing forwards. Land it and re-prove the golden (diff=0.0) in isolation so
   any golden drift is unambiguously attributable.
2. **Lever B** second — single-function, pure-metric gating, golden-neutral by
   construction. Independent of A.
3. **Lever C** last — the concurrency change. Largest surface, own acceptance
   gate (resume/early-stop tests), benefits from A/B already being green so a
   golden failure here would be surprising and easy to localize.

A and B are mutually independent; C is independent of both but ordered last for
risk isolation. Each lever is committed separately with its proving test run
recorded in the commit message.

---

## Testing plan (runnable on host RTX 5080)

All `pytest` targets are the project's tiny-model CPU/GPU-light tests — they run
on the host without the production teacher. Run from repo root.

**Baseline (must do FIRST, before any edit):** confirm green at `588ec5e`:
```
pytest max_quality/tests/test_router_kd_golden_snapshot.py \
       max_quality/tests/test_stage5_merge_repair.py \
       max_quality/tests/test_smoke_stage5_resume.py \
       max_quality/tests/test_stage5_early_stop.py -v
```

**Per lever:** re-run the suite above; for A the golden byte/trace pass is the
diff=0.0 proof; for B add the manual `grad_norm` window-boundary parity check;
for C the resume + early-stop tests are the acceptance gate plus the torn-
snapshot guard.

**Final (all three landed):** full suite green + **no `MOE_REGEN_GOLDEN`** (a
regen would mean a lever drifted — that is a failure, not a re-bless). Optionally
record a wall-clock delta on a short host run to confirm the throughput/memory
wins materialized (A: VRAM via `nvidia-smi`; C: reduced training-thread stall),
but the correctness gate is the byte-identical golden + green resume tests.

---

## Constraints restated (do NOT cross)
- Zero golden re-bless. `compressed_metadata.json` (both stage keys) byte-
  identical; loss trace within `rel_tol=1e-5`.
- No bf16 loss kernel, no fused KL, no `kd_seq_chunk_size` change, no window-loss
  / teacher-cache / dataloader changes.
- Plan-only branch — production edits happen in a follow-up branch.
