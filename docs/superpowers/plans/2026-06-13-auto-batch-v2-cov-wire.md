# Auto-Batch v2 step 2 — Wire the Resolver into Cov Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Wire the auto-batch resolver (`size_batch` + `run_with_oom_backoff`) into the Stage-3 covariance capture so the cov forward batch auto-sizes to VRAM (killing the ~27 min/layer cov wall), gated behind `cov_batch_size: "auto"` + `auto_batch.enabled`. The cov Gram reduction is already PINNED (v2 step 1, main `59666d8`) so a bigger batch is reduction-grouping-invariant. **Default (no `"auto"`) is byte-identical** — the resolver never runs. Single-replica (1-GPU) only; DP min-agreement is a later step.

**Architecture (from the lifecycle map):**
- **Idempotent boundary:** wrap ONLY the `with stack: for batch in batches` dual-forward block (`covariance_collection.py` ~819-847) in `run_with_oom_backoff(run_fn, start_batch, floor=1)`. `run_fn(bs)`: (1) `discard_layer` the window's layers on `B_acc`/`C_acc` (pop `_pending`/`_gpu_token_count`, NO covariance write); (2) re-slice `batches = iter_batches(calib, batch_size=bs)`; (3) run the dual-forward loop. **Finalize + spill stay OUTSIDE** the wrapped region — the OOM-prone forward region is strictly upstream of `finalize_layer`, so `covariance`/spill are never touched on an OOM and never re-entered on retry.
- **CRITICAL — closure scoping (Review C1).** The cov callbacks `input_cb`/`intermediate_cb`/`_teacher_input_cb` read `_teacher_hidden`/`_teacher_dense`/`_teacher_filled`/`_teacher_T`/`_seq_len` as FREE VARIABLES bound to `_collect_covariances`'s scope. If `run_fn` is a nested `def` that ASSIGNS `_teacher_T`/`_seq_len`, Python makes them function-locals of `run_fn` and the callbacks still see the OUTER `_seq_len` (which stays `0`) → `input_cb`'s `if _seq_len ...` is False → it falls to the UN-pinned `update` and the cross-cov `// _seq_len` divides by zero/skips. **This silently abandons the reduction pin.** The nested `run_fn` MUST declare `nonlocal _teacher_T, _seq_len` (and not shadow the teacher dicts — `.clear()` mutates in place, no rebind, so dicts are fine). The unit test must assert `_seq_len` is non-zero inside the callback under the auto path.
- **CRITICAL — `calib`/`iter_batches` are NOT in `_collect_covariances` today (Review C2).** It receives the already-materialized `batches` list, not `calib`, and does not import `iter_batches`. The auto path must re-slice at the sized bs, so thread `calib` + the `auto_batch` config into `_collect_covariances`: add params, add to the plugin `reads` tuple + the `collect_covariances` hook, update BOTH call sites (orchestrator ~`:1241`, worker ~`:1034`), and `from ...utils.calibration import iter_batches` in the collector. Non-auto path still uses the passed `batches` unchanged.
- **Probe once, G-resident:** run `size_batch` once, inside the first non-skipped window AFTER the hooks are installed (G layers resident, so `CudaMemProbe.allocated()` baseline absorbs the G commitment — no double-count with `_resolve_cov_window`'s G sizing, no hand-derived `G·bs·seq·d` formula). The cost probe = one dual-forward at 1 seq, then 2 seq, measuring `max_memory_allocated`. ORDERING (Review H2/M1): before calling `size_batch`, `discard_layer` ALL window layers + clear the three teacher dicts so the baseline is clean; `size_batch` internally probes/frees; after sizing, `discard_layer` + clear again so the probe's Gram never contaminates the real run. Reuse the sized bs for all windows (G constant → per-forward peak window-independent). Use a **cov-specific `max_cap`** (sequences, NOT the v1 default 4096 — that would blow VRAM; clamp to e.g. ≤ a few hundred, or document operator-set) (Review L3).
- **Honest fidelity (Review M3):** call `size_batch`/`run_with_oom_backoff` DIRECTLY (not `resolve_batch`) — cov is `REDUCTION_ACCUMULATING` (down_proj is allclose-not-bitwise even pinned), so don't mislabel it `BATCH_INVARIANT` to pass `_V1_ELIGIBLE`. Do NOT import `FidelityClass`/`resolve_batch` into cov; the test asserts this.
- **Default byte-identical, DOUBLE-gated (Review H1):** the auto path fires only when `cov_batch_size == "auto"` AND `AutoBatchConfig.from_dict(s3.get("auto_batch")).enabled` (mirror `ma_detection.py:367-370`; `headroom_frac`/`max_cap` come from that cfg). Default (`cov_batch_size` absent) → `_resolve_cov_batch_size` returns inherited `1`, `cov_auto=False`, resolver/probe never run → bs=1 → pin no-op → stage3 golden unchanged. DP worker path keeps returning `inherited` (unchanged; the DP reduce sums finalized fp32 Grams key-wise so it's bs-agnostic — future DP-auto is unblocked).

**Tech Stack:** PyTorch, pytest. Code root `max_quality/`. CPU-only for design+impl+byte-identity tests; the LIVE cov-wall speedup needs a 35B/H200 (deferred, like the multi-GPU validation).

**Spec:** `docs/.../2026-06-11-per-plugin-vram-aware-auto-batch-sizing-design.md` §5/§6 (G co-resolve), §10 step 2. Builds on v1 (`utils/auto_batch.py`) + v2-pin (`update_grouped`, main `59666d8`).

---

## File Structure
- **Modify** `src/moe_compress/utils/activation_hooks.py` — add `InputCovarianceAccumulator.discard_layer(layer_idx)` (pops `_pending`/`_gpu_token_count` for the layer, NO covariance write — mirrors `finalize_layer`'s pop minus the CPU accumulate). One small method.
- **Modify** `src/moe_compress/stage3/plugins/covariance_collection.py` —
  - `_resolve_cov_batch_size`: keep returning a valid POSITIVE int (floor). Add a sibling `_cov_is_auto(s3) -> bool` returning `s3.get("cov_batch_size")=="auto" and AutoBatchConfig.from_dict(s3.get("auto_batch")).enabled` (Review N1: do NOT return a `-1` sentinel — the orchestrator does `iter_batches(calib, batch_size=...)` which rejects non-positive).
  - Thread `calib` + `auto_batch` cfg into `_collect_covariances` (new params), the plugin `reads` tuple, the `collect_covariances` hook, and BOTH call sites (orchestrator ~`:1241`, worker ~`:1034`). `from ...utils.calibration import iter_batches` in the collector.
  - In `_collect_covariances`: when `cov_auto`, probe-once `size_batch` in the first window (clean baseline) → `cov_bs`; wrap the dual-forward block in `run_with_oom_backoff(run_fn, cov_bs, floor=1)` where `run_fn` declares `nonlocal _teacher_T, _seq_len`, discards window layers, re-slices `iter_batches(calib, batch_size=bs)`, runs the loop. Finalize/spill stay outside. Non-auto path unchanged.
  - `from ..utils.auto_batch import size_batch, run_with_oom_backoff, AutoBatchConfig, CudaMemProbe` (NOT `resolve_batch`/`FidelityClass`).
- **Create** `tests/test_cov_autobatch_wire.py` — unit tests: `discard_layer` correctness; the auto-path run_fn idempotency (discard+rerun == single run); default-off byte-identity at the call-site level (with fakes, no GPU).
- **Goldens** `tests/golden/stage3*` — NOT TOUCHED (default not "auto" → byte-identical).

---

## Conventions
- Logger `log = logging.getLogger(__name__)`. No GPU in unit tests (fake `MemProbe`, synthetic forwards / monkeypatched model). Commit per task; trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: `InputCovarianceAccumulator.discard_layer` (forward-free)

**Files:** Modify `activation_hooks.py`; Test `tests/test_cov_autobatch_wire.py`.

`discard_layer(layer_idx)` pops every `_pending` key with `k[0]==layer_idx` and the matching `_gpu_token_count`, under `self._lock`, WITHOUT writing to `self.covariance` (mirrors `finalize_layer`'s pop loop minus the CPU accumulate). Lets an OOM-retry reset a window's in-flight Gram so re-running the forwards doesn't double-count.

- [ ] **Step 1: Failing tests**
```python
import torch
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

def test_discard_layer_clears_pending_no_covariance_write():
    a = InputCovarianceAccumulator(); a.set_storage_dtype(torch.float32)
    a.update(3, 0, "gate_proj", torch.randn(4, 8))
    a.update(3, 1, "gate_proj", torch.randn(4, 8))
    a.update(5, 0, "gate_proj", torch.randn(4, 8))     # different layer — must survive
    a.discard_layer(3)
    assert (3, 0, "gate_proj") not in a._pending and (3, 1, "gate_proj") not in a._pending
    assert (5, 0, "gate_proj") in a._pending            # other layer untouched
    assert not a.covariance                              # NOTHING finalized
    assert all(k[0] != 3 for k in a._gpu_token_count)

def test_discard_then_reaccumulate_equals_fresh_bytewise():
    x = torch.randn(6, 8)
    fresh = InputCovarianceAccumulator(); fresh.set_storage_dtype(torch.float32)
    fresh.update(0, 0, "gate_proj", x)
    re = InputCovarianceAccumulator(); re.set_storage_dtype(torch.float32)
    re.update(0, 0, "gate_proj", torch.randn(6, 8))     # junk first attempt
    re.discard_layer(0)
    re.update(0, 0, "gate_proj", x)                     # retry
    assert torch.equal(re._pending[(0,0,"gate_proj")], fresh._pending[(0,0,"gate_proj")])
```
- [ ] **Step 2: Run** `python3 -m pytest tests/test_cov_autobatch_wire.py -q` → FAIL (`discard_layer` undefined).
- [ ] **Step 3: Implement** — read `finalize_layer` (~`:1082`), add `discard_layer` reusing its pop pattern minus the covariance write.
- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(cov-wire): InputCovarianceAccumulator.discard_layer`.

---

## Task 2: Wire size_batch + run_with_oom_backoff into the cov loop

**Files:** Modify `covariance_collection.py` (wiring + auto path) + the two call sites; extend the test.

Read `_collect_covariances` (~410-868), the window/batch loops (~761-868, the per-batch teacher-dict clears ~829-831, `_seq_len`/`_teacher_T` set ~842), finalize/spill (~850-866), `_resolve_cov_batch_size` (~351-415), the plugin `reads`/`collect_covariances` hook (~1206-1241), and the worker (~987,1034).

- [ ] **Step 1: Plumb the gate + `calib`/`iter_batches` (Review C2/H1/N1).**
  - Add `_cov_is_auto(s3) -> bool`: `return s3.get("cov_batch_size") == "auto" and AutoBatchConfig.from_dict(s3.get("auto_batch")).enabled`. `_resolve_cov_batch_size` UNCHANGED (still returns a positive int floor; for `"auto"` it returns the inherited floor, used as `floor=` for backoff and the non-auto fallback).
  - `from ...utils.calibration import iter_batches` and `from ..utils.auto_batch import size_batch, run_with_oom_backoff, AutoBatchConfig, CudaMemProbe` in the collector. Do NOT import `resolve_batch`/`FidelityClass`.
  - Add params `calib=None, cov_auto=False, auto_batch_cfg=None` to `_collect_covariances`; thread them through the `collect_covariances` hook + the plugin `reads` tuple (add `"calib"`) + both call sites: orchestrator (~`:1241`, has `calib` on run_ctx ~`:281`) and worker (~`:1034`, `calib` local ~`:980`). The worker call site (~`:1034`) adds `calib=calib, cov_auto=False` (it has the `calib` local at ~`:980`; `auto_batch_cfg` unused on the DP path) so the signature matches — DP stays on the inherited int path, byte-identical. `_COV_MAX_CAP` (e.g. 256 seqs) is a BACKSTOP only — `size_batch`'s `headroom_frac` fit is the real limiter; don't mistake 256 for a tuned value.

- [ ] **Step 2: Probe-once sizing (Review H2/M1/L3).** When `cov_auto`, on the FIRST non-skipped window, after `with stack:` installs the hooks (G resident, ~`:817`):
```python
def _clear_teacher():
    _teacher_hidden.clear(); _teacher_dense.clear(); _teacher_filled.clear()
def _discard_window():
    for _k, ref in to_collect:
        B_acc.discard_layer(ref.layer_idx)
        if C_acc is not None: C_acc.discard_layer(ref.layer_idx)   # C_acc is None when cross-cov off

def cost_probe_fn(mb):
    _discard_window(); _clear_teacher()
    torch.cuda.reset_peak_memory_stats(device)
    b = calib[:mb]                                   # mb sequences
    nonlocal _teacher_T, _seq_len
    _teacher_T = b.shape[0]*b.shape[1]; _seq_len = int(b.shape[1])
    with torch.no_grad():
        if teacher_model is not None: teacher_model(input_ids=b)   # None on B-only path
        model(input_ids=b)
    return int(torch.cuda.max_memory_allocated(device))

_discard_window(); _clear_teacher()                  # clean baseline BEFORE size_batch
ab = auto_batch_cfg or AutoBatchConfig()
cov_bs = size_batch(cost_probe_fn, fixed_batch=cov_floor,
                    headroom_frac=ab.headroom_frac, max_cap=_COV_MAX_CAP,   # cov-specific cap, NOT 4096
                    mem=CudaMemProbe(device))
_discard_window(); _clear_teacher()                  # probe Gram must NOT contaminate the real run
log.info("cov auto-batch: sized cov_bs=%d (floor=%d)", cov_bs, cov_floor)
```
`_COV_MAX_CAP` = a sane module constant in *sequences* (e.g. 256) — a 4096-seq dual-forward over G layers would OOM-backoff repeatedly. Cache `cov_bs` on the first window; reuse for all windows (G constant). `cost_probe_fn` declares `nonlocal _teacher_T, _seq_len` so the callbacks' free vars update (Review C1).

- [ ] **Step 3: OOM-backoff around the dual-forward block (Review C1/M2).** Replace the inner `for batch in batches:` body (~`:819-847`) — when `cov_auto`:
```python
def run_window_forwards(bs):
    nonlocal _teacher_T, _seq_len                    # CRITICAL: callbacks read these as free vars
    _discard_window()                                # idempotent: reset any aborted attempt's _pending
    for batch in iter_batches(calib, batch_size=bs):
        _clear_teacher()
        _teacher_T = batch.shape[0]*batch.shape[1]; _seq_len = int(batch.shape[1])
        with torch.no_grad():
            if teacher_model is not None: teacher_model(input_ids=batch)   # guard B-only path
            model(input_ids=batch)
run_with_oom_backoff(run_window_forwards, start_batch=cov_bs, floor=cov_floor)
# finalize (:850-853) + spill (:855+) stay HERE, OUTSIDE the wrapper — never retried
```
NON-auto path: the existing `with stack: for batch in batches:` loop, verbatim — no nonlocal/probe/wrapper. (Keep the two branches clearly separated so the non-auto path is provably the original.)

- [ ] **Step 4: Failing tests** (`tests/test_cov_autobatch_wire.py`, GPU-free — fake CudaMemProbe + a tiny monkeypatched dual-model, or a focused harness around the accumulators):
  - **`_seq_len` reaches the callback under auto** (Review C1): drive the auto `run_window_forwards`/`cost_probe_fn` and assert the cov callback observed `_seq_len == batch.shape[1]` (NOT 0) — i.e. the pin actually fires under the wrapper. A regression of the closure scoping must fail this.
  - **Mid-batch OOM idempotency** (Review M2): a `run_window_forwards` whose model raises `torch.cuda.OutOfMemoryError` on batch N of M at the start bs, then succeeds at bs//2 → the finalized Gram (after the post-wrapper finalize) is byte-identical to a single clean pass at bs//2. Proves `_discard_window` resets the aborted attempt.
  - **Default-off:** no `"auto"` → `_cov_is_auto` False → original loop; assert `size_batch`/`run_with_oom_backoff` NOT called (spies).
  - **No fidelity mislabel** (Review M3): assert `covariance_collection` does NOT import `resolve_batch`/`FidelityClass` (`"resolve_batch" not in dir(module)` / source grep).
  - **Probe non-contamination** (Review N-new1): an auto run whose `cost_probe_fn` actually accumulated into `_pending` produces a final finalized Gram byte-identical to a run that skipped the probe — i.e. the pre+post `_discard_window()` fully removes the probe's tokens (assert the finalized covariance, not just `_pending` emptiness).

- [ ] **Step 5: Implement** Steps 1-3. **Step 6: Run** the new tests → PASS.

- [ ] **Step 7: GOLDEN GUARDRAIL** — `python3 -m pytest tests/test_stage3_golden_snapshot.py tests/test_smoke_stage3.py tests/test_cov_reduction_pin.py tests/test_multigpu_stage3.py tests/test_stage2_cov_manifest.py -q` MUST pass UNCHANGED, no `MOE_REGEN_GOLDEN` (default has no `cov_batch_size:"auto"` → resolver never runs → byte-identical; `test_a6_cov_batch_size_close` in the multigpu suite pins the gate/C-bitwise + down_proj-allclose pin contract). If it changes → STOP, report.

- [ ] **Step 8: Commit** `feat(cov-wire): auto-size cov forward batch via size_batch+run_with_oom_backoff (gated, default byte-identical)`.

## Task 3: Config docs

**Files:** `covariance_collection.py` `_resolve_cov_batch_size` docstring + `stage3/__init__` config doc.
- [ ] Document `cov_batch_size: "auto"` (+ `auto_batch.enabled`): probes VRAM with the G window resident, auto-sizes the cov forward batch, OOM-backoff to floor=1. Default int/inherited = unchanged & byte-identical. 1-GPU only (DP returns inherited until min-agreement lands). down_proj allclose / gate+C bitwise per the pin. Commit `docs(cov-wire): document cov_batch_size auto`.

---

## Task 4: (GPU, manual, DEFERRED) live cov-wall smoke
- [ ] On a real ≥40GB GPU with a real MoE model: set `cov_batch_size:"auto"`, confirm the resolver sizes bs>1, the wall-clock per layer drops vs bs=1, and the cov result is allclose to the bs=1 run (gate+C bitwise, down_proj ~1e-6). NOT a CI test (needs a 35B/H200 — the RTX5080 won't fit). Note pending; ride along the next real cov run.

---

## Out of scope
- DP `min(candidate)` cross-replica agreement (later step — the worker path stays `inherited`).
- ablation_filter / block_refine wiring (their own pins first).
- Re-blessing any golden.

## After this plan
Standard plan/review loop → impl/review loop, all-none. Live GPU validation deferred (Task 4).
