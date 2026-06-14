# Stage-3 Covariance Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four independent, default-OFF knobs that make the Stage-3 covariance pass ~40-100x cheaper at paper-faithful quality and fit a single H200 (single-pass CPU-accumulator + 512-seq calib + gentler OOM backoff).

**Architecture:** (A) a CPU-resident hot-accumulator mode in `InputCovarianceAccumulator` (GEMM stays on GPU, only the running sum + result move to CPU) — bitwise at cov_batch_size=1; (B) single-pass G=N window that auto-enables A so one forward captures all layers; (C) a `cov_num_sequences` override distinct from the global calib count; (D) a gentler OOM backoff (x0.75, not /2). All default-OFF → byte-identical goldens.

**Tech Stack:** PyTorch, the existing moe_compress Stage-3 cov collection (`utils/activation_hooks.py`, `stage3/plugins/covariance_collection.py`, `stage3/orchestrator.py`, `utils/auto_batch.py`). Tests run from `max_quality/` on tiny synthetic tensors (the 35B does NOT fit this host; real-35B validation is DEFERRED to the resume GPU box).

---

## Goal

Four independent, default-OFF knobs that reduce Stage-3 covariance-pass wall time and VRAM on a single H200 without touching quality or the byte-identical golden:

| ID | Short name | What it does |
|----|-----------|--------------|
| A | CPU hot-accumulator | Running-sum (`_pending`) migrated to CPU; GEMM stays GPU. Frees per-layer GPU Gram VRAM so a wider window fits. |
| B | Single-pass G=N | Sets window G = n\_layers AND auto-enables A, so all 40 layers accumulate in one forward pass instead of ⌈N/G⌉ passes. |
| C | `cov_num_sequences` knob | Independent sequence-count override for the Stage-3 B/C calib build; intended fast value 512 (vs global 2048+). |
| D | Gentler OOM backoff | Replace halve (`// 2`) with `× 0.75` (floor of `attempt-1`) in `run_with_oom_backoff`; reclaims throughput on marginal OOMs (18→14→10 instead of 18→9). |

### Non-negotiable correctness invariants

1. **Byte-identical default path.** All four knobs default OFF / absent. Existing goldens (`test_stage3_golden_snapshot.py`, `test_stage4_golden_snapshot.py`, etc.) MUST pass without `MOE_REGEN_GOLDEN`.
2. **GEMM NEVER moves to CPU** (Task A). The `flat_f32.T @ flat_f32` matmul in `update()` / cross-tensor handling in `update_cross()` MUST execute on the GPU. Only the result tensor and the running sum move.
3. **Single-pass bitwise guarantee.** At `cov_batch_size=1`, matmul + reduction order are unchanged by the running-sum device; finalized covs must be `torch.equal` to the windowed baseline.
4. **No interaction between knobs.** B implies A; A, C, D are fully independent of each other. Enabling C does NOT change the seed or any parameter other than `num_sequences` for the Stage-3 B/C calib spec.

---

## Files → responsibilities

| File | Task | Change |
|------|------|--------|
| `src/moe_compress/utils/activation_hooks.py` | A | Add `_hot_accum_device` field + `set_hot_accumulator_device()` method on `InputCovarianceAccumulator`; modify `update()` (≈:1019-1029) and `update_cross()` (≈:1071-1080) to move result to CPU when flag set; adapt `finalize_layer()` Phase-2 (≈:1097) to no-op-cast an already-CPU pending tensor |
| `src/moe_compress/stage3/plugins/covariance_collection.py` | B | Extend `_resolve_cov_window()` (≈:347-398) to accept `"all"` as a string value or read `stage3_svd.cov_single_pass: true`; return `n_layers` and set a sentinel indicating CPU-accum must be enabled; wire that sentinel into the `B_acc` / `C_acc` setup in the orchestrator |
| `src/moe_compress/stage3/orchestrator.py` | B, C | (B) After `B_acc` / `C_acc` instantiation (≈:254-255), call `B_acc.set_hot_accumulator_device("cpu")` when single-pass is active; same for `C_acc`. (C) Read `s3.get("cov_num_sequences")` (≈:235); if set, deep-copy `cal` dict, override `num_sequences`, call `spec_from_config` on the copy |
| `src/moe_compress/utils/auto_batch.py` | D | Replace `:201` `new = max(attempt // 2, floor)` with `new = max(min(int(attempt * 0.75), attempt - 1), floor)` |
| `tests/test_stage3_cov_efficiency.py` | A,B,C,D | New file: all TDD tests for the four tasks |

---

## Detailed design

### Task A — CPU hot-accumulator

**Mechanism.** `InputCovarianceAccumulator` gets one new field:

```python
_hot_accum_device: str | None = None   # None = current GPU path (default)
```

And one new method:

```python
def set_hot_accumulator_device(self, device: str) -> None:
    self._hot_accum_device = device
```

**`update()` change** (`activation_hooks.py:1019-1029`). After `cov = flat_f32.transpose(0, 1) @ flat_f32` (GEMM ON GPU, UNCHANGED), insert:

```python
if self._hot_accum_device is not None:
    cov = cov.to(self._hot_accum_device)
```

Then `_pending[key]` accumulates on `_hot_accum_device` (CPU). The `cur.add_(cov)` call already works on CPU because `add_` is device-local.

**`update_cross()` change** (`activation_hooks.py:1071-1080`). `cross_f32 = cross.to(torch.float32)` stays on GPU (cross is passed in already computed). After that cast, insert the same device migration:

```python
if self._hot_accum_device is not None:
    cross_f32 = cross_f32.to(self._hot_accum_device)
```

The `_pending[key].add_(cross_f32.to(device=cur.device))` line already handles the device match for the existing case; the existing `.to(device=cur.device)` in the else-branch ensures no device mismatch regardless.

**`finalize_layer()` change** (`activation_hooks.py:1082-1112`). Phase 2 currently does:

```python
cpu_items = [(k, gpu_cov.to(storage_dtype).cpu(), n_tok) ...]
```

When `_pending[key]` is already on CPU (hot-accum path), `.cpu()` is a no-op. The `.to(storage_dtype)` cast still fires correctly. No special-casing needed — the existing code handles it transparently.

**Lock discipline.** Unchanged. The GEMM fires OUTSIDE the lock (as today). The `cov.to("cpu")` call happens BEFORE the lock is acquired. The only change is that `_pending[key]` lives on CPU; the in-place add is still under the lock.

**Bitwise guarantee.** At `cov_batch_size=1` the GEMM operands are identical; the fp32 result is accumulated in the same order (batch-sequential); only the accumulator buffer's device changes. The final `finalize_layer` Phase-3 merge (`prev.to(float32) + cpu_cov.to(float32)`) is device-agnostic (both are CPU). Byte-identity at `cov_batch_size=1` holds.

### Task B — single-pass G=N

**Config surface.** Two equivalent spellings (either accepted):
- `stage3_svd.cov_single_pass: true`
- `multi_gpu.cov_window_size: "all"`

**`_resolve_cov_window()` change** (`covariance_collection.py:347-398`). At the top of the function, before the existing `mg = config.get("multi_gpu") or {}` block, add:

```python
s3 = config.get("stage3_svd") or {}
if s3.get("cov_single_pass", False):
    return n_layers   # caller must also enable CPU accumulator
```

In the `req` parsing block, add `"all"` as a recognized string sentinel that maps to `n_layers` (alongside the existing `"auto"` branch):

```python
if isinstance(req, str):
    if req.strip().lower() == "all":
        return n_layers
    elif req.strip().lower() != "auto":
        ...
```

**Orchestrator wiring** (`orchestrator.py`). After `B_acc` is instantiated (≈:254) and after `C_acc` would be instantiated (inside the `if cross_cov_enabled:` block, ≈:268+), add:

```python
_single_pass = (
    s3.get("cov_single_pass", False)
    or (config.get("multi_gpu") or {}).get("cov_window_size", "auto") == "all"
)
if _single_pass:
    B_acc.set_hot_accumulator_device("cpu")
    if C_acc is not None:
        C_acc.set_hot_accumulator_device("cpu")
```

This is the only place the coupling between B and A is enforced. `_resolve_cov_window` returns `n_layers`; the window loop runs exactly once.

**VRAM argument.** With 40 layers and CPU accumulation: the GPU only holds the active forward-pass weights (~120 GB BF16 for teacher+student) plus transient activations. The Gram buffers (`_pending`) live on CPU RAM (~4 GB total for gate_proj at d_hid=5120). Without CPU accum, 40 layers × ~4 GB/layer of GPU Grams = ~160 GB GPU — impossible on 141 GB H200. Single-pass only works WITH Task A.

### Task C — `cov_num_sequences` knob

**Config key:** `stage3_svd.cov_num_sequences` (int, absent = use `cal["num_sequences"]`).

**Orchestrator change** (`orchestrator.py:231-238`). Replace the single `spec = spec_from_config(cal, seed_offset=2)` call with:

```python
_cov_num_seq = s3.get("cov_num_sequences")
if _cov_num_seq is not None:
    import copy as _copy
    _cal_cov = _copy.copy(cal)          # shallow copy is sufficient: only one int key changes
    _cal_cov = dict(_cal_cov)           # make it mutable if needed
    _cal_cov["num_sequences"] = int(_cov_num_seq)
    spec = spec_from_config(_cal_cov, seed_offset=2)
else:
    spec = spec_from_config(cal, seed_offset=2)
```

The `spec_from_config` function already accepts `num_sequences_override` as a kwarg (`calibration.py:336`), so this can be simplified to:

```python
_cov_num_seq = s3.get("cov_num_sequences")
spec = spec_from_config(
    cal,
    seed_offset=2,
    num_sequences_override=int(_cov_num_seq) if _cov_num_seq is not None else None,
)
```

This is cleaner and avoids a dict copy entirely. The `spec_from_config` already handles `num_sequences_override` correctly (`calibration.py:357-358`).

**Default absent = byte-identical.** `num_sequences_override=None` → `spec_from_config` reads `cal["num_sequences"]` as before.

**Independence.** Does NOT affect Stage-2's calibration build, Stage-5's calibration build, or the global `cal` dict. The same `cal` dict is used downstream for `validation_samples` etc.

### Task D — gentler OOM backoff

**Change** (`auto_batch.py:201`). Replace:

```python
new = max(attempt // 2, floor)
```

with:

```python
new = max(min(int(attempt * 0.75), attempt - 1), floor)
```

**Correctness proof of termination.** At any `attempt > floor`, `int(attempt * 0.75) ≤ attempt - 1` when `attempt ≥ 4` (since `0.75*4=3 < 4`). For `attempt ∈ {2,3}`: `int(2*0.75)=1` and `int(3*0.75)=2`, both `< attempt`. The `min(..., attempt-1)` guarantees strict decrease even for `attempt=1` (though `floor` is always ≥ 1 and the `if attempt <= floor: raise` guard fires first). So every step strictly decreases `attempt` toward `floor`, and the loop terminates. Update the docstring to note "18→14→10 instead of 18→9→..." as the intended faster-recovery sequence.

---

## Bite-sized TDD tasks

> Discipline: write the failing test first (red), then implement (green), then commit. No skipping steps. All tests use tiny synthetic tensors on CPU or (for CUDA assertions) skip-if-no-CUDA marks. Do NOT touch the existing goldens.

### Task A — CPU hot-accumulator

**TA.1 — failing test: flag exists and `set_hot_accumulator_device` is callable**

- File: `tests/test_stage3_cov_efficiency.py` (create)
- Test `test_cpu_accum_flag_exists`:
  ```python
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
  acc = InputCovarianceAccumulator()
  assert hasattr(acc, "_hot_accum_device")
  assert acc._hot_accum_device is None
  assert callable(getattr(acc, "set_hot_accumulator_device", None))
  ```
- Run: `cd max_quality && python -m pytest tests/test_stage3_cov_efficiency.py::test_cpu_accum_flag_exists -x` → expect `AttributeError` / FAILED.
- Impl: add `_hot_accum_device: str | None = None` field and `set_hot_accumulator_device` method to `InputCovarianceAccumulator` in `activation_hooks.py`.
- Run again → PASSED.
- Commit: `feat(stage3-cov-eff): Task A — _hot_accum_device field + setter`.

**TA.2 — failing test: `update()` bitwise-equal GPU-hot vs CPU-hot**

- Test `test_update_gpu_hot_vs_cpu_hot_bitwise`:
  ```python
  import torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  torch.manual_seed(0)
  d = 16
  # Build two identical accumulators
  acc_gpu = InputCovarianceAccumulator()
  acc_cpu = InputCovarianceAccumulator()
  acc_cpu.set_hot_accumulator_device("cpu")

  # Feed 3 identical small batches
  for _ in range(3):
      x = torch.randn(8, d)      # CPU tensor (no CUDA required)
      acc_gpu.update(0, 0, "gate_proj", x)
      acc_cpu.update(0, 0, "gate_proj", x)

  acc_gpu.finalize_layer(0)
  acc_cpu.finalize_layer(0)

  k = (0, 0, "gate_proj")
  assert torch.equal(acc_gpu.covariance[k], acc_cpu.covariance[k]), (
      "GPU-hot and CPU-hot paths must produce bitwise-equal finalized covariance"
  )
  ```
- Run → FAILED (no hot-accum logic yet).
- Impl: in `update()`, after `cov = flat_f32.transpose(0, 1) @ flat_f32`, insert the `if self._hot_accum_device is not None: cov = cov.to(self._hot_accum_device)` block.
- Run → PASSED.
- Commit: `feat(stage3-cov-eff): Task A — update() hot-accum device migration`.

**TA.3 — failing test: `update()` GEMM stays on input device (GPU invariant)**

- Test `test_update_gemm_stays_on_gpu` (skip if no CUDA):
  ```python
  import pytest, torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  pytest.importorskip("torch.cuda")
  if not torch.cuda.is_available():
      pytest.skip("no CUDA")

  acc = InputCovarianceAccumulator()
  acc.set_hot_accumulator_device("cpu")

  # Monkey-read the matmul to check device: use a small GPU tensor
  x = torch.randn(4, 8, device="cuda")
  # We verify by asserting _pending[key] is on CPU AFTER update,
  # and that no CPU matmul occurred (the result device is cpu but
  # that's because we migrated AFTER the gpu matmul).
  acc.update(0, 0, "gate_proj", x)
  k = (0, 0, "gate_proj")
  # _pending must be on CPU (the hot-accum migration happened)
  assert acc._pending[k].device.type == "cpu", (
      "_pending must be on CPU when hot_accum_device='cpu'"
  )
  # The covariance values must still be correct (non-zero)
  assert acc._pending[k].abs().max() > 0
  ```
- Run → FAILED (pending still on GPU before TA.2 impl, or asserted wrong thing).
- Impl: TA.2 impl already covers this; run confirms PASSED.
- No separate commit needed — covered by TA.2.

**TA.4 — failing test: `update_cross()` bitwise-equal GPU-hot vs CPU-hot**

- Test `test_update_cross_gpu_hot_vs_cpu_hot_bitwise`:
  ```python
  import torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  torch.manual_seed(7)
  d = 12
  acc_gpu = InputCovarianceAccumulator()
  acc_cpu = InputCovarianceAccumulator()
  acc_cpu.set_hot_accumulator_device("cpu")

  for _ in range(4):
      cross = torch.randn(d, d)    # precomputed cross-cov tensor
      acc_gpu.update_cross(0, 0, "gate_proj", cross, n_tokens=8)
      acc_cpu.update_cross(0, 0, "gate_proj", cross, n_tokens=8)

  acc_gpu.finalize_layer(0)
  acc_cpu.finalize_layer(0)

  k = (0, 0, "gate_proj")
  assert torch.equal(acc_gpu.covariance[k], acc_cpu.covariance[k]), (
      "update_cross GPU-hot and CPU-hot must be bitwise-equal"
  )
  ```
- Run → FAILED (`update_cross` not yet migrated).
- Impl: in `update_cross()`, after `cross_f32 = cross.to(torch.float32)`, insert `if self._hot_accum_device is not None: cross_f32 = cross_f32.to(self._hot_accum_device)`. Also update the `cur.add_(cross_f32.to(device=cur.device))` call — when both are on CPU, `cur.device == cross_f32.device` and `.to(device=cur.device)` is a no-op, so this is transparent.
- Run → PASSED.
- Commit: `feat(stage3-cov-eff): Task A — update_cross() hot-accum migration`.

**TA.5 — failing test: `finalize_layer` handles already-CPU pending (no double-transfer)**

- Test `test_finalize_layer_already_cpu_pending`:
  ```python
  import torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  acc = InputCovarianceAccumulator()
  acc.set_hot_accumulator_device("cpu")

  x = torch.randn(6, 10)
  acc.update(0, 2, "gate_proj", x)

  # Manually verify _pending is on CPU before finalize
  k = (0, 2, "gate_proj")
  assert acc._pending[k].device.type == "cpu"

  # finalize_layer must not raise and must produce correct covariance
  acc.finalize_layer(0)
  assert k in acc.covariance
  assert acc.covariance[k].device.type == "cpu"
  expected = (x.T @ x).to(acc.storage_dtype)
  assert torch.allclose(acc.covariance[k], expected, atol=1e-6)
  ```
- Run → may PASS already after TA.2 (the `.to(storage_dtype).cpu()` is a no-op when already CPU). If PASSED, note it and skip separate impl. Commit: `test(stage3-cov-eff): Task A — finalize_layer already-CPU pending test`.

**TA.6 — run full suite gate**

- Run: `cd max_quality && python -m pytest tests/test_stage3_cov_efficiency.py tests/test_stage3_golden_snapshot.py tests/test_utils_hooks.py tests/test_activation_hooks_finalize_batch.py -x`
- Expected: all PASSED, no `MOE_REGEN_GOLDEN`.
- Commit: (none — this is a verification step, not a code change).

---

### Task B — single-pass G=N

**TB.1 — failing test: `_resolve_cov_window` returns `n_layers` for `cov_single_pass: true`**

- Test `test_resolve_cov_window_single_pass`:
  ```python
  from moe_compress.stage3.plugins.covariance_collection import _resolve_cov_window

  n = 40
  config = {"stage3_svd": {"cov_single_pass": True}}
  assert _resolve_cov_window(config, n) == n

  # Also check "all" string in multi_gpu block
  config2 = {"multi_gpu": {"cov_window_size": "all"}}
  assert _resolve_cov_window(config2, n) == n

  # Default path unchanged
  config3 = {}
  # auto on CPU-only box degrades to 1
  result = _resolve_cov_window(config3, n)
  assert isinstance(result, int) and result >= 1
  ```
- Run → FAILED (`_resolve_cov_window` does not handle `cov_single_pass` or `"all"`).
- Impl: in `_resolve_cov_window` (`covariance_collection.py:347-398`):
  1. At top (after `if n_layers <= 0: return 1`), add:
     ```python
     s3 = config.get("stage3_svd") or {}
     if s3.get("cov_single_pass", False):
         return n_layers
     ```
  2. In the `req` string-dispatch block (≈:367-373), add before the existing `"auto"` check:
     ```python
     if req.strip().lower() == "all":
         return n_layers
     ```
- Run → PASSED.
- Commit: `feat(stage3-cov-eff): Task B — _resolve_cov_window single-pass + "all" sentinel`.

**TB.2 — failing test: single-pass G=N produces `torch.equal` cov vs windowed G=1 on tiny synthetic multi-layer collection**

- Test `test_single_pass_vs_windowed_bitwise_tiny`:
  ```python
  """
  Build a tiny InputCovarianceAccumulator; simulate what _collect_covariances
  does for 4 synthetic 'layers' (each with 2 experts) under two window modes:
    - windowed G=2 (2 passes of 2 layers each), GPU-hot (default)
    - single-pass G=4, CPU-hot (Task A + Task B)
  Verify torch.equal on all finalized gate_proj covariances.

  This is a unit test of the accumulator mechanics, NOT a real cov collection.
  """
  import torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  torch.manual_seed(42)
  n_layers, n_experts, d = 4, 2, 8

  # Pre-generate identical activation data for each (layer, expert) slot
  data = {
      (li, ei): torch.randn(6, d)
      for li in range(n_layers)
      for ei in range(n_experts)
  }

  def _run_accumulation(window_size, use_cpu_accum):
      acc = InputCovarianceAccumulator()
      if use_cpu_accum:
          acc.set_hot_accumulator_device("cpu")
      # Simulate windowed passes: each window covers `window_size` layers
      for w_start in range(0, n_layers, window_size):
          w_end = min(w_start + window_size, n_layers)
          # One forward pass feeds all layers in [w_start, w_end)
          for li in range(w_start, w_end):
              for ei in range(n_experts):
                  acc.update(li, ei, "gate_proj", data[(li, ei)])
          # Finalize at window end
          for li in range(w_start, w_end):
              acc.finalize_layer(li)
      return acc.covariance

  cov_windowed = _run_accumulation(window_size=2, use_cpu_accum=False)
  cov_single   = _run_accumulation(window_size=4, use_cpu_accum=True)

  assert set(cov_windowed.keys()) == set(cov_single.keys()), "key sets must match"
  for k in cov_windowed:
      assert torch.equal(cov_windowed[k], cov_single[k]), (
          f"key {k}: single-pass vs windowed must be bitwise-equal"
      )
  ```
- Run → FAILED before TB.1 is done (wrong window resolver), PASSED after TB.1 + TA.2.
- No separate impl needed if TB.1 and TA.2 are done.
- Commit: `test(stage3-cov-eff): Task B — single-pass vs windowed bitwise equality`.

**TB.3 — failing test: orchestrator wires CPU accumulator when `cov_single_pass: true`**

- This is an orchestrator integration test. Use a minimal config dict, mock out the heavy machinery, and assert `B_acc.set_hot_accumulator_device` was called.
- Test `test_orchestrator_single_pass_enables_cpu_accum` (use `unittest.mock.patch`):
  ```python
  """Assert that when cov_single_pass is true the orchestrator calls
  set_hot_accumulator_device('cpu') on B_acc before collection begins.
  Achieved by patching InputCovarianceAccumulator to record calls."""
  import pytest
  from unittest.mock import patch, MagicMock

  # We test the _resolve logic only — not the full orchestrator run
  # (the full run requires model + tokenizer).
  # Test the helper that resolves the flag and sets the device:
  # directly call _resolve_cov_window and verify the n_layers return.

  from moe_compress.stage3.plugins.covariance_collection import _resolve_cov_window

  n = 40
  cfg = {"stage3_svd": {"cov_single_pass": True}}
  g = _resolve_cov_window(cfg, n)
  assert g == n, f"single-pass must return G=n_layers={n}, got {g}"
  ```
  (Deeper orchestrator wiring is covered by the smoke test in TB.4.)
- Run → PASSED (covered by TB.1 impl).
- No separate impl. Commit: `test(stage3-cov-eff): Task B — orchestrator single-pass flag test`.

**TB.4 — run full suite gate for Task B**

- Run: `cd max_quality && python -m pytest tests/test_stage3_cov_efficiency.py tests/test_stage3_golden_snapshot.py tests/test_stage3_plugin_covariance.py -x`
- Expected: all PASSED, no regen.

---

### Task C — `cov_num_sequences` knob

**TC.1 — failing test: orchestrator passes `num_sequences_override` when `cov_num_sequences` is set**

- Test `test_cov_num_sequences_override_changes_spec`:
  ```python
  """Assert that stage3_svd.cov_num_sequences overrides only the Stage-3 cov
  spec's num_sequences, not the global cal dict. Uses a mock calib builder
  to capture the spec it was called with."""
  from unittest.mock import MagicMock, patch, call
  from moe_compress.utils.calibration import spec_from_config, CalibrationSpec

  # Minimal cal config
  cal = {
      "num_sequences": 2048,
      "sequence_length": 512,
      "seed": 0,
      "source": "nvidia-cascade",
      "subset_weights": {"math": 1.0},
  }
  s3_with_override = {"cov_num_sequences": 512}
  s3_without_override = {}

  # With override: spec.num_sequences must be 512
  spec_with = spec_from_config(cal, seed_offset=2, num_sequences_override=512)
  assert spec_with.num_sequences == 512, (
      f"override must set num_sequences=512, got {spec_with.num_sequences}"
  )

  # Without override: spec.num_sequences must be 2048 (unchanged)
  spec_without = spec_from_config(cal, seed_offset=2, num_sequences_override=None)
  assert spec_without.num_sequences == 2048, (
      f"default must use cal num_sequences=2048, got {spec_without.num_sequences}"
  )

  # The seed is identical in both (seed_offset=2 applied to same base seed)
  assert spec_with.seed == spec_without.seed, "seed must be unchanged by num_seq override"
  ```
- Run → PASSED (tests `spec_from_config` which already supports `num_sequences_override`).
- If PASSED, this confirms the implementation hook exists; the orchestrator change is the only impl step.
- Commit: `test(stage3-cov-eff): Task C — cov_num_sequences spec override test`.

**TC.2 — failing test: orchestrator reads `cov_num_sequences` from `stage3_svd` and passes override**

- Test `test_orchestrator_reads_cov_num_sequences`:
  ```python
  """Directly exercise the spec-building logic in the orchestrator by mocking
  spec_from_config and asserting it is called with num_sequences_override=512."""
  from unittest.mock import patch, MagicMock

  # We directly call the code path that should exist post-impl:
  # orchestrator._build_bcov_spec(s3, cal) — or inline the logic as a helper.
  # Since the orchestrator doesn't yet expose this as a function, we verify via
  # the resolver pattern: mock spec_from_config in the orchestrator's namespace
  # and check the call args.

  # Pre-condition: patch spec_from_config in stage3.orchestrator to capture call
  import moe_compress.stage3.orchestrator as orch_mod
  import moe_compress.utils.calibration as cal_mod

  captured = []
  original = cal_mod.spec_from_config

  def _capturing_spec_from_config(cal, *, seed_offset=0, num_sequences_override=None, **kw):
      captured.append({"seed_offset": seed_offset, "num_sequences_override": num_sequences_override})
      return original(cal, seed_offset=seed_offset, num_sequences_override=num_sequences_override, **kw)

  # To test the orchestrator logic without a model, we extract the resolution
  # logic into a helper (see impl below) and call it directly.
  # HELPER to extract: _resolve_bcov_spec(s3, cal) -> CalibrationSpec
  # This is the impl target for TC.2.
  try:
      from moe_compress.stage3.orchestrator import _resolve_bcov_spec
  except ImportError:
      import pytest; pytest.fail("_resolve_bcov_spec not yet implemented")

  cal = {"num_sequences": 2048, "sequence_length": 512, "seed": 0, "source": "nvidia-cascade", "subset_weights": {"math": 1.0}}
  s3_with = {"cov_num_sequences": 512}
  s3_without = {}

  spec_with = _resolve_bcov_spec(s3_with, cal)
  assert spec_with.num_sequences == 512

  spec_without = _resolve_bcov_spec(s3_without, cal)
  assert spec_without.num_sequences == 2048
  ```
- Run → FAILED (`_resolve_bcov_spec` does not exist).
- Impl: Extract a small helper in `orchestrator.py` before the `run()` function:
  ```python
  def _resolve_bcov_spec(s3: dict, cal: dict):
      """Build the B/C calibration spec, applying cov_num_sequences override if set."""
      _cov_num_seq = s3.get("cov_num_sequences")
      return spec_from_config(
          cal,
          seed_offset=2,
          num_sequences_override=int(_cov_num_seq) if _cov_num_seq is not None else None,
      )
  ```
  Then in `run()` at `:235`, replace `spec = spec_from_config(cal, seed_offset=2)` with `spec = _resolve_bcov_spec(s3, cal)`.
- Run → PASSED.
- Commit: `feat(stage3-cov-eff): Task C — cov_num_sequences override via _resolve_bcov_spec`.

**TC.3 — run full suite gate for Task C**

- Run: `cd max_quality && python -m pytest tests/test_stage3_cov_efficiency.py tests/test_stage3_golden_snapshot.py tests/test_smoke_stage3.py -x`
- Expected: all PASSED. The smoke test uses a fixture with `cal["num_sequences"]` set but no `cov_num_sequences` in `stage3_svd`, so it follows the unchanged default path.

---

### Task D — gentler OOM backoff

**TD.1 — failing test: backoff sequence is gentler than halving and terminates at floor**

- Test `test_run_with_oom_backoff_gentler_sequence`:
  ```python
  """
  run_with_oom_backoff with a fake run_fn that OOMs for batch > threshold.
  Assert:
  1. The attempt sequence is ×0.75 steps (gentler than ×0.5 halving).
  2. Terminates at floor without infinite loop.
  3. The returned result is from the first batch at or below the threshold.
  """
  import pytest
  try:
      import torch.cuda
      _has_cuda = torch.cuda.is_available()
  except Exception:
      _has_cuda = False

  from moe_compress.utils.auto_batch import run_with_oom_backoff

  # Simulate CUDA OOM by raising the real exception class.
  # If CUDA not available, we skip the OOM simulation test.
  if not _has_cuda:
      pytest.skip("no CUDA — OOM exception class requires cuda")

  import torch

  threshold = 10   # OOM for batch > 10
  attempts = []

  def _fake_run(batch):
      attempts.append(batch)
      if batch > threshold:
          raise torch.cuda.OutOfMemoryError("simulated OOM")
      return batch   # success returns the batch size

  result = run_with_oom_backoff(_fake_run, start_batch=18, floor=4)

  # First attempt: 18 (OOM), then ×0.75 steps
  # 18 → int(18*0.75)=13 → int(13*0.75)=9 → 9 ≤ 10 → success at 9
  # (or via min(int(18*.75), 17) = min(13,17) = 13, then min(int(13*.75),12) = min(9,12) = 9)
  assert result == 9, f"expected success at batch=9, got {result}"
  assert attempts == [18, 13, 9], f"expected attempts=[18,13,9], got {attempts}"

  # Verify it's strictly less aggressive than halving (halving would give [18,9])
  assert len(attempts) > 2, "gentler backoff must take more steps than halving"
  ```
- Run → FAILED (current impl halves: attempts=[18,9]).
- Impl: in `auto_batch.py:201`, replace `new = max(attempt // 2, floor)` with `new = max(min(int(attempt * 0.75), attempt - 1), floor)`. Update the docstring.
- Run → PASSED.
- Commit: `feat(stage3-cov-eff): Task D — gentler OOM backoff (×0.75 vs ÷2)`.

**TD.2 — failing test: backoff terminates at floor (raises at floor)**

- Test `test_run_with_oom_backoff_raises_at_floor`:
  ```python
  """If even the floor batch OOMs, run_with_oom_backoff must re-raise."""
  import pytest, torch

  if not torch.cuda.is_available():
      pytest.skip("no CUDA")

  from moe_compress.utils.auto_batch import run_with_oom_backoff

  def _always_oom(batch):
      raise torch.cuda.OutOfMemoryError("always")

  with pytest.raises(torch.cuda.OutOfMemoryError):
      run_with_oom_backoff(_always_oom, start_batch=4, floor=4)
  ```
- Run → PASSED already (existing impl re-raises at floor, behavior unchanged by D impl).
- Commit: `test(stage3-cov-eff): Task D — floor re-raise test`.

**TD.3 — run full suite gate for Task D**

- Run: `cd max_quality && python -m pytest tests/test_stage3_cov_efficiency.py -x -k "backoff"` then `cd max_quality && python -m pytest tests/ -x --ignore=tests/test_stage3_cov_efficiency.py -q`
- Expected: all PASSED.

---

## Final integration gate

Run after all four tasks are committed:

```bash
cd max_quality && python -m pytest \
    tests/test_stage3_cov_efficiency.py \
    tests/test_stage3_golden_snapshot.py \
    tests/test_stage4_golden_snapshot.py \
    tests/test_stage6_golden_snapshot.py \
    tests/test_stage6alt_golden_snapshot.py \
    tests/test_utils_hooks.py \
    tests/test_activation_hooks_finalize_batch.py \
    tests/test_stage3_plugin_covariance.py \
    tests/test_smoke_stage3.py \
    -v
```

Expected: all PASSED, zero `MOE_REGEN_GOLDEN`.

---

## Deferred — real-35B GPU validation (NOT runnable on this host)

The following must be verified on a GPU box with the real 35B model before enabling knobs in production. DO NOT attempt on RTX5080 (16 GB — model does not fit).

1. **Byte-identical default gate.** Run Stage-3 at default config (no new knobs set) on the 35B calibration and assert `torch.equal` on every finalized cov key vs a pre-recorded reference. This proves the four code changes are truly no-ops at default settings.

2. **Single-pass CPU-hot path vs windowed GPU-hot baseline.** With `cov_single_pass: true` enabled, run Stage-3 cov collection on 1-2 layers and `torch.equal`-compare the finalized `B_cov` / `C_cov` dicts vs the existing windowed result. Expected: byte-identical at `cov_batch_size=1` (per the Task A bitwise guarantee).

3. **`cov_num_sequences: 512` spectrum quality gate.** Run the full Stage-3 SVD pipeline (cov → d-rank → α-search → factor) with `cov_num_sequences: 512` and confirm: (a) the resulting `rank_map.json` ranks differ from the 2048-sequence baseline by ≤ 5% of keys, and (b) end-to-end PPL on WikiText-2 degrades by ≤ 0.05 nats vs the 2048-sequence baseline. These are the per-doc #4 quality gates.

4. **`run_with_oom_backoff` real-world OOM sequence.** On a near-capacity forward batch, confirm the gentler backoff (×0.75) completes successfully at a higher batch size than the old ÷2 would have, and that total runtime is lower.

---

## Build sequence checklist

- [ ] **TA.1** — add `_hot_accum_device` field + `set_hot_accumulator_device()` → test passes
- [ ] **TA.2** — `update()` device migration after GEMM → bitwise test passes
- [ ] **TA.3** — GPU-CUDA invariant test (skip if no CUDA) → confirmed
- [ ] **TA.4** — `update_cross()` device migration → bitwise test passes
- [ ] **TA.5** — `finalize_layer` already-CPU test → passes (transparent)
- [ ] **TA.6** — full suite gate for Task A
- [ ] **TB.1** — `_resolve_cov_window` single-pass + `"all"` → test passes
- [ ] **TB.2** — single-pass vs windowed bitwise equality test → passes
- [ ] **TB.3** — orchestrator flag test → passes
- [ ] **TB.4** — full suite gate for Task B
- [ ] **TC.1** — `spec_from_config num_sequences_override` test → passes
- [ ] **TC.2** — `_resolve_bcov_spec` helper + orchestrator wire → test passes
- [ ] **TC.3** — full suite gate for Task C
- [ ] **TD.1** — gentler backoff sequence test → passes (SKIP if no CUDA)
- [ ] **TD.2** — floor re-raise test → passes (SKIP if no CUDA)
- [ ] **TD.3** — full suite gate for Task D
- [ ] **Final integration gate** — all goldens green, zero regen
```

Now I need to write this as an actual file. Given I can only use the tools listed, and none of them write files, I'll produce the final content as the response for the user to create manually. But wait — the task says "committed on feat/stage3-cov-efficiency". Let me check whether I can maybe produce this via an indirect approach, but I don't have a Write or Bash tool.

I'll deliver the plan content in my response and provide the exact git commands to save and commit it.

---

Here is the complete plan. Save it to `/home/lucas/ai/moe_compress/max_quality/docs/superpowers/plans/2026-06-14-stage3-cov-efficiency.md` (the directory already exists):

```markdown
# Plan: Stage-3 covariance efficiency — CPU accumulator + single-pass + cov_num_sequences + gentler OOM backoff

**Date:** 2026-06-14
**Branch:** `feat/stage3-cov-efficiency` (main checkout `/home/lucas/ai/moe_compress`, code root `max_quality/`)
**Status:** plan — not yet implemented
**Spec sources:**
- `research/stage3-single-pass-cov:max_quality/docs/research/2026-06-14-stage3-single-pass-cov.md`
- `research/stage3-bc-calibration-size:max_quality/docs/research/2026-06-14-stage3-bc-calibration-size.md`
- `research/single-h200-stage3-strategy:max_quality/docs/research/2026-06-14-single-h200-stage3-strategy.md`

---

## Goal

Four independent, default-OFF knobs that reduce Stage-3 covariance-pass wall time and VRAM on a single H200 without touching quality or the byte-identical golden:

| ID | Short name | What it does |
|----|-----------|--------------|
| A | CPU hot-accumulator | Running-sum (`_pending`) migrated to CPU after GPU GEMM; frees per-layer GPU Gram VRAM so a wider window fits |
| B | Single-pass G=N | Sets window G = n\_layers AND auto-enables A, so all 40 layers accumulate in one forward pass |
| C | `cov_num_sequences` knob | Independent sequence-count override for the Stage-3 B/C calib build; intended fast value 512 |
| D | Gentler OOM backoff | Replace halve (`// 2`) with `× 0.75` floor `attempt-1` in `run_with_oom_backoff` |

### Non-negotiable correctness invariants

1. **Byte-identical default path.** All four knobs default OFF / absent. Existing goldens (`test_stage3_golden_snapshot.py`, `test_stage4_golden_snapshot.py`, `test_stage6_golden_snapshot.py`, `test_stage6alt_golden_snapshot.py`) MUST pass without `MOE_REGEN_GOLDEN`.
2. **GEMM NEVER moves to CPU (Task A).** The `flat_f32.transpose(0, 1) @ flat_f32` matmul in `update()` and the `cross_f32` handling in `update_cross()` execute on the GPU. Only the cov result tensor and the running sum move to CPU afterward.
3. **Single-pass bitwise guarantee (Task B).** At `cov_batch_size=1`, matmul and reduction order are unchanged; finalized covs must be `torch.equal` to the windowed baseline.
4. **No cross-knob interaction.** B implies A (auto-enables it). A, C, D are fully independent of each other and of B.

---

## Files → responsibilities

| File | Task | Change |
|------|------|--------|
| `src/moe_compress/utils/activation_hooks.py` | A | Add `_hot_accum_device: str \| None = None` field and `set_hot_accumulator_device()` method on `InputCovarianceAccumulator`; modify `update()` (≈:1019-1029) and `update_cross()` (≈:1071-1080) to move result to `_hot_accum_device` when set; `finalize_layer()` Phase-2 (≈:1097) already handles already-CPU pending transparently |
| `src/moe_compress/stage3/plugins/covariance_collection.py` | B | Extend `_resolve_cov_window()` (:347-398) to accept `stage3_svd.cov_single_pass: true` or `multi_gpu.cov_window_size: "all"` → return `n_layers` |
| `src/moe_compress/stage3/orchestrator.py` | B, C | (B) After `B_acc` / `C_acc` instantiation (≈:254-268), call `.set_hot_accumulator_device("cpu")` when single-pass is active. (C) Extract `_resolve_bcov_spec(s3, cal)` helper; replace `:235` `spec = spec_from_config(cal, seed_offset=2)` with `spec = _resolve_bcov_spec(s3, cal)` |
| `src/moe_compress/utils/auto_batch.py` | D | Replace `:201` `new = max(attempt // 2, floor)` with `new = max(min(int(attempt * 0.75), attempt - 1), floor)` |
| `tests/test_stage3_cov_efficiency.py` | A,B,C,D | New file — all TDD tests |

---

## Detailed design

### Task A — CPU hot-accumulator

**`InputCovarianceAccumulator` changes** (`activation_hooks.py`):

1. Add field after `_lock` (≈:978): `_hot_accum_device: str | None = None`

2. Add method after `set_storage_dtype()` (≈:982):
   ```python
   def set_hot_accumulator_device(self, device: str) -> None:
       self._hot_accum_device = device
   ```

3. In `update()` (≈:1019-1029), after `cov = flat_f32.transpose(0, 1) @ flat_f32` (GEMM unchanged, stays on GPU):
   ```python
   if self._hot_accum_device is not None:
       cov = cov.to(self._hot_accum_device)
   ```
   The subsequent lock + `_pending[key]` accumulation runs normally; `cur.add_(cov)` works on CPU because both sides are on `_hot_accum_device`.

4. In `update_cross()` (≈:1071-1080), after `cross_f32 = cross.to(torch.float32)`:
   ```python
   if self._hot_accum_device is not None:
       cross_f32 = cross_f32.to(self._hot_accum_device)
   ```
   The existing `cur.add_(cross_f32.to(device=cur.device))` already handles device matching; when both are on CPU this is a no-op cast.

5. `finalize_layer()` (≈:1082-1112): Phase 2's `gpu_cov.to(storage_dtype).cpu()` is transparent when `gpu_cov` is already on CPU — `.cpu()` is a no-op. No special-casing needed.

**Bitwise guarantee.** At `cov_batch_size=1`, the GEMM operands and their batch-sequential accumulation order are identical to the GPU-hot path. The only change is where the running sum lives. The Phase-3 merge in `finalize_layer` (`prev.to(float32) + cpu_cov.to(float32)`) is device-agnostic.

### Task B — single-pass G=N

**`_resolve_cov_window()` changes** (`covariance_collection.py:347-398`):

1. At the top of the function body (after `if n_layers <= 0: return 1`):
   ```python
   s3 = config.get("stage3_svd") or {}
   if s3.get("cov_single_pass", False):
       return n_layers
   ```

2. In the `req` string-dispatch block (≈:367-373), add before the existing `"auto"` branch:
   ```python
   if req.strip().lower() == "all":
       return n_layers
   ```

**Orchestrator wiring** (`orchestrator.py`). After `B_acc = InputCovarianceAccumulator()` (≈:254) and the `C_acc` conditional setup (≈:268+), add:
```python
_single_pass = (
    s3.get("cov_single_pass", False)
    or (config.get("multi_gpu") or {}).get("cov_window_size", "auto") == "all"
)
if _single_pass:
    B_acc.set_hot_accumulator_device("cpu")
    if C_acc is not None:
        C_acc.set_hot_accumulator_device("cpu")
```

**VRAM argument.** Without CPU accum, G=40 would hold 40 layers × ~4 GB/layer of fp32 Grams on GPU ≈ 160 GB — impossible on 141 GB H200 with both models resident. Task A is the prerequisite; B enforces it automatically.

### Task C — `cov_num_sequences` knob

**New helper** in `orchestrator.py` (add before `run()`):
```python
def _resolve_bcov_spec(s3: dict, cal: dict):
    """Build the Stage-3 B/C calibration spec.

    Reads ``stage3_svd.cov_num_sequences``; if set, uses it as the sequence
    count for the cov pass only — does not mutate the global ``cal`` dict and
    does not affect any other stage's calib spec. Absent → unchanged behavior
    (``cal["num_sequences"]``, byte-identical default).
    """
    _cov_num_seq = s3.get("cov_num_sequences")
    return spec_from_config(
        cal,
        seed_offset=2,
        num_sequences_override=int(_cov_num_seq) if _cov_num_seq is not None else None,
    )
```

Then replace `orchestrator.py:235`:
```python
spec = spec_from_config(cal, seed_offset=2)
```
with:
```python
spec = _resolve_bcov_spec(s3, cal)
```

`spec_from_config` already accepts `num_sequences_override` (`calibration.py:336-358`). No dict copy needed; `spec_from_config` reads the integer from the kwarg, not from a mutated dict.

### Task D — gentler OOM backoff

**`auto_batch.py:201`** — replace:
```python
new = max(attempt // 2, floor)
```
with:
```python
new = max(min(int(attempt * 0.75), attempt - 1), floor)
```

Update the function docstring (`run_with_oom_backoff`, :188-204) to note "18→14→10 instead of 18→9" as the recovery example.

**Termination proof.** For any `attempt > floor ≥ 1`: `int(attempt * 0.75) ≤ attempt - 1` holds for `attempt ≥ 4` (0.75·4=3<4). For `attempt ∈ {2,3}`: `int(2·0.75)=1 < 2`, `int(3·0.75)=2 < 3`. The `min(..., attempt-1)` clamp guarantees strict decrease even for small values. The floor re-raise path (`if attempt <= floor: raise`) is unchanged.

---

## Bite-sized TDD tasks

> Discipline: write failing test first (red), implement (green), commit. All tests run on CPU or skip-if-no-CUDA. No model required. No golden regen.

### Task A — CPU hot-accumulator

**TA.1 — `_hot_accum_device` field + setter exist**

- File: `tests/test_stage3_cov_efficiency.py` (create)
- Test `test_cpu_accum_flag_exists`:
  ```python
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
  acc = InputCovarianceAccumulator()
  assert hasattr(acc, "_hot_accum_device") and acc._hot_accum_device is None
  assert callable(getattr(acc, "set_hot_accumulator_device", None))
  ```
- Run: `cd /home/lucas/ai/moe_compress/max_quality && python -m pytest tests/test_stage3_cov_efficiency.py::test_cpu_accum_flag_exists -x`
- Expected: **FAILED** (AttributeError).
- Impl: add field + method to `InputCovarianceAccumulator` in `activation_hooks.py`.
- Rerun: **PASSED**.
- Commit: `feat(stage3-cov-eff): Task A — _hot_accum_device field + setter`

**TA.2 — `update()` bitwise-equal GPU-hot vs CPU-hot**

- Test `test_update_gpu_hot_vs_cpu_hot_bitwise`:
  ```python
  import torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  torch.manual_seed(0)
  d = 16
  acc_gpu = InputCovarianceAccumulator()
  acc_cpu = InputCovarianceAccumulator()
  acc_cpu.set_hot_accumulator_device("cpu")

  for _ in range(3):
      x = torch.randn(8, d)   # CPU tensor; no CUDA needed
      acc_gpu.update(0, 0, "gate_proj", x)
      acc_cpu.update(0, 0, "gate_proj", x)

  acc_gpu.finalize_layer(0)
  acc_cpu.finalize_layer(0)

  k = (0, 0, "gate_proj")
  assert torch.equal(acc_gpu.covariance[k], acc_cpu.covariance[k])
  ```
- Run: `python -m pytest tests/test_stage3_cov_efficiency.py::test_update_gpu_hot_vs_cpu_hot_bitwise -x`
- Expected: **FAILED** (no hot-accum logic yet).
- Impl: in `update()` after `cov = flat_f32.transpose(0, 1) @ flat_f32`, add `if self._hot_accum_device is not None: cov = cov.to(self._hot_accum_device)`.
- Rerun: **PASSED**.
- Commit: `feat(stage3-cov-eff): Task A — update() hot-accum device migration`

**TA.3 — `_pending` on CPU when flag set (CUDA test)**

- Test `test_update_pending_on_cpu_after_gpu_compute`:
  ```python
  import pytest, torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  if not torch.cuda.is_available():
      pytest.skip("no CUDA")

  acc = InputCovarianceAccumulator()
  acc.set_hot_accumulator_device("cpu")

  x = torch.randn(4, 8, device="cuda")
  acc.update(0, 0, "gate_proj", x)

  k = (0, 0, "gate_proj")
  assert acc._pending[k].device.type == "cpu"
  assert acc._pending[k].abs().max() > 0
  ```
- Run: PASSED after TA.2 impl (or skipped if no CUDA). No separate impl.
- Commit: `test(stage3-cov-eff): Task A — GPU-resident GEMM + CPU _pending invariant`

**TA.4 — `update_cross()` bitwise-equal GPU-hot vs CPU-hot**

- Test `test_update_cross_gpu_hot_vs_cpu_hot_bitwise`:
  ```python
  import torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  torch.manual_seed(7)
  d = 12
  acc_gpu = InputCovarianceAccumulator()
  acc_cpu = InputCovarianceAccumulator()
  acc_cpu.set_hot_accumulator_device("cpu")

  for _ in range(4):
      cross = torch.randn(d, d)
      acc_gpu.update_cross(0, 0, "gate_proj", cross, n_tokens=8)
      acc_cpu.update_cross(0, 0, "gate_proj", cross, n_tokens=8)

  acc_gpu.finalize_layer(0)
  acc_cpu.finalize_layer(0)

  k = (0, 0, "gate_proj")
  assert torch.equal(acc_gpu.covariance[k], acc_cpu.covariance[k])
  ```
- Run: **FAILED** (`update_cross` not yet migrated).
- Impl: in `update_cross()` after `cross_f32 = cross.to(torch.float32)`, add `if self._hot_accum_device is not None: cross_f32 = cross_f32.to(self._hot_accum_device)`.
- Rerun: **PASSED**.
- Commit: `feat(stage3-cov-eff): Task A — update_cross() hot-accum migration`

**TA.5 — `finalize_layer` transparent with already-CPU pending**

- Test `test_finalize_layer_already_cpu_pending`:
  ```python
  import torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  acc = InputCovarianceAccumulator()
  acc.set_hot_accumulator_device("cpu")
  x = torch.randn(6, 10)
  acc.update(0, 2, "gate_proj", x)

  k = (0, 2, "gate_proj")
  assert acc._pending[k].device.type == "cpu"

  acc.finalize_layer(0)

  assert k in acc.covariance
  assert acc.covariance[k].device.type == "cpu"
  expected = (x.T @ x).to(acc.storage_dtype)
  assert torch.allclose(acc.covariance[k], expected, atol=1e-6)
  ```
- Run after TA.2 impl: expected **PASSED** (`.cpu()` is no-op on already-CPU tensor). If FAILED, fix `finalize_layer` to no-op when pending is already on target.
- Commit: `test(stage3-cov-eff): Task A — finalize_layer already-CPU pending`

**TA.6 — full suite gate**

```bash
cd /home/lucas/ai/moe_compress/max_quality
python -m pytest \
    tests/test_stage3_cov_efficiency.py \
    tests/test_stage3_golden_snapshot.py \
    tests/test_utils_hooks.py \
    tests/test_activation_hooks_finalize_batch.py \
    -v
```
Expected: all PASSED, zero regen.

---

### Task B — single-pass G=N

**TB.1 — `_resolve_cov_window` returns `n_layers` for single-pass config**

- Test `test_resolve_cov_window_single_pass_spellings`:
  ```python
  from moe_compress.stage3.plugins.covariance_collection import _resolve_cov_window

  n = 40
  # Spelling 1: stage3_svd.cov_single_pass
  assert _resolve_cov_window({"stage3_svd": {"cov_single_pass": True}}, n) == n

  # Spelling 2: multi_gpu.cov_window_size = "all"
  assert _resolve_cov_window({"multi_gpu": {"cov_window_size": "all"}}, n) == n

  # Default path: returns something ≥ 1 (auto-probe or 1)
  result = _resolve_cov_window({}, n)
  assert 1 <= result <= n
  ```
- Run: **FAILED** (neither sentinel handled).
- Impl: modify `_resolve_cov_window` (`covariance_collection.py:347-398`) as described in design above.
- Rerun: **PASSED**.
- Commit: `feat(stage3-cov-eff): Task B — _resolve_cov_window single-pass + "all" sentinel`

**TB.2 — single-pass G=N vs windowed G=2 bitwise equality (tiny synthetic)**

- Test `test_single_pass_vs_windowed_bitwise_tiny`:
  ```python
  import torch
  from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

  torch.manual_seed(42)
  n_layers, n_experts, d = 4, 2, 8

  data = {
      (li, ei): torch.randn(6, d)
      for li in range(n_layers) for ei in range(n_experts)
  }

  def _run(window_size, use_cpu_accum):
      acc = InputCovarianceAccumulator()
      if use_cpu_accum:
          acc.set_hot_accumulator_device("cpu")
      for w_start in range(0, n_layers, window_size):
          w_end = min(w_start + window_size, n_layers)
          for li in range(w_start, w_end):
              for ei in range(n_experts):
                  acc.update(li, ei, "gate_proj", data[(li, ei)])
          for li in range(w_start, w_end):
              acc.finalize_layer(li)
      return acc.covariance

  cov_windowed = _run(window_size=2, use_cpu_accum=False)
  cov_single   = _run(window_size=4, use_cpu_accum=True)

  assert set(cov_windowed.keys()) == set(cov_single.keys())
  for k in cov_windowed:
      assert torch.equal(cov_windowed[k], cov_single[k]), f"mismatch at {k}"
  ```
- Run: **PASSED** after TA.2 impl (single-pass is just window=n_layers with CPU accum — the accumulator tests already verify this).
- Commit: `test(stage3-cov-eff): Task B — single-pass vs windowed bitwise equality`

**TB.3 — orchestrator flag resolution**

- Test `test_resolve_cov_window_n_layers_edge_cases`:
  ```python
  from moe_compress.stage3.plugins.covariance_collection import _resolve_cov_window

  # n_layers=0 → always returns 1 (guard unchanged)
  assert _resolve_cov_window({"stage3_svd": {"cov_single_pass": True}}, 0) == 1

  # n_layers=1 → single-pass trivially equals windowed
  assert _resolve_cov_window({"stage3_svd": {"cov_single_pass": True}}, 1) == 1

  # "ALL" (caps) accepted
  assert _resolve_cov_window({"multi_gpu": {"cov_window_size": "ALL"}}, 5) == 5
  ```
- Run: PASSED after TB.1 impl (`.lower()` handles caps; `n_layers=0` returns 1 via existing guard before the new code).
- Commit: `test(stage3-cov-eff): Task B — edge cases for single-pass resolver`

**TB.4 — full suite gate**

```bash
python -m pytest \
    tests/test_stage3_cov_efficiency.py \
    tests/test_stage3_golden_snapshot.py \
    tests/test_stage3_plugin_covariance.py \
    -v
```
Expected: all PASSED.

---

### Task C — `cov_num_sequences` knob

**TC.1 — `spec_from_config` already supports `num_sequences_override` (contract test)**

- Test `test_spec_from_config_num_sequences_override`:
  ```python
  from moe_compress.utils.calibration import spec_from_config

  cal = {
      "num_sequences": 2048, "sequence_length": 512, "seed": 0,
      "source": "nvidia-cascade", "subset_weights": {"math": 1.0},
  }

  spec_with = spec_from_config(cal, seed_offset=2, num_sequences_override=512)
  assert spec_with.num_sequences == 512

  spec_without = spec_from_config(cal, seed_offset=2, num_sequences_override=None)
  assert spec_without.num_sequences == 2048

  # Seed unchanged by override
  assert spec_with.seed == spec_without.seed
  ```
- Run: **PASSED** (confirming the hook exists before adding orchestrator code).
- Commit: `test(stage3-cov-eff): Task C — spec_from_config num_sequences_override contract`

**TC.2 — `_resolve_bcov_spec` helper and orchestrator wire**

- Test `test_resolve_bcov_spec_helper`:
  ```python
  import pytest

  try:
      from moe_compress.stage3.orchestrator import _resolve_bcov_spec
  except ImportError:
      pytest.fail("_resolve_bcov_spec not yet implemented in orchestrator")

  cal = {
      "num_sequences": 2048, "sequence_length": 512, "seed": 0,
      "source": "nvidia-cascade", "subset_weights": {"math": 1.0},
  }

  spec_with = _resolve_bcov_spec({"cov_num_sequences": 512}, cal)
  assert spec_with.num_sequences == 512

  spec_without = _resolve_bcov_spec({}, cal)
  assert spec_without.num_sequences == 2048

  # Does not mutate the cal dict
  assert cal["num_sequences"] == 2048
  ```
- Run: **FAILED** (`_resolve_bcov_spec` ImportError).
- Impl: add `_resolve_bcov_spec` to `orchestrator.py` and replace `:235` `spec = spec_from_config(cal, seed_offset=2)` with `spec = _resolve_bcov_spec(s3, cal)`.
- Rerun: **PASSED**.
- Commit: `feat(stage3-cov-eff): Task C — _resolve_bcov_spec + orchestrator wire`

**TC.3 — global `cal` dict is not mutated**

- Covered by the `assert cal["num_sequences"] == 2048` assertion in TC.2. No additional test needed.

**TC.4 — full suite gate**

```bash
python -m pytest \
    tests/test_stage3_cov_efficiency.py \
    tests/test_stage3_golden_snapshot.py \
    tests/test_smoke_stage3.py \
    -v
```
Expected: all PASSED.

---

### Task D — gentler OOM backoff

**TD.1 — backoff sequence is ×0.75, not ÷2**

- Test `test_run_with_oom_backoff_gentler_sequence` (CUDA required):
  ```python
  import pytest, torch
  from moe_compress.utils.auto_batch import run_with_oom_backoff

  if not torch.cuda.is_available():
      pytest.skip("requires CUDA for OutOfMemoryError")

  threshold = 10
  attempts = []

  def _fake_run(batch):
      attempts.append(batch)
      if batch > threshold:
          raise torch.cuda.OutOfMemoryError("simulated")
      return batch

  result = run_with_oom_backoff(_fake_run, start_batch=18, floor=4)

  # 18 → min(int(18*0.75), 17) = min(13,17) = 13 (OOM)
  # 13 → min(int(13*0.75), 12) = min(9,12) = 9 (success)
  assert attempts == [18, 13, 9], f"expected [18,13,9], got {attempts}"
  assert result == 9
  # Gentler than halving ([18, 9]) — more steps
  assert len(attempts) > 2
  ```
- Run: **FAILED** (current impl halves → `[18, 9]`).
- Impl: in `auto_batch.py:201` replace `new = max(attempt // 2, floor)` with `new = max(min(int(attempt * 0.75), attempt - 1), floor)`. Update docstring.
- Rerun: **PASSED**.
- Commit: `feat(stage3-cov-eff): Task D — gentler OOM backoff ×0.75 vs ÷2`

**TD.2 — floor re-raise unchanged**

- Test `test_run_with_oom_backoff_floor_reraise` (CUDA required):
  ```python
  import pytest, torch
  from moe_compress.utils.auto_batch import run_with_oom_backoff

  if not torch.cuda.is_available():
      pytest.skip("requires CUDA for OutOfMemoryError")

  def _always_oom(batch):
      raise torch.cuda.OutOfMemoryError("always")

  with pytest.raises(torch.cuda.OutOfMemoryError):
      run_with_oom_backoff(_always_oom, start_batch=4, floor=4)
  ```
- Run: **PASSED** (existing re-raise unchanged).
- Commit: `test(stage3-cov-eff): Task D — floor re-raise test`

**TD.3 — termination at floor from high start**

- Test `test_run_with_oom_backoff_terminates_at_floor` (CUDA required):
  ```python
  import pytest, torch
  from moe_compress.utils.auto_batch import run_with_oom_backoff

  if not torch.cuda.is_available():
      pytest.skip("requires CUDA for OutOfMemoryError")

  attempts = []

  def _run(batch):
      attempts.append(batch)
      if batch > 2:
          raise torch.cuda.OutOfMemoryError("oom")
      return batch

  result = run_with_oom_backoff(_run, start_batch=100, floor=2)
  assert result == 2
  # All intermediate attempts must be strictly decreasing
  for i in range(1, len(attempts)):
      assert attempts[i] < attempts[i-1], f"not strictly decreasing: {attempts}"
  # Floor is the last attempt that succeeded
  assert attempts[-1] == 2
  ```
- Run: **PASSED** after TD.1 impl.
- Commit: `test(stage3-cov-eff): Task D — strict-decrease termination test`

**TD.4 — full suite gate**

```bash
python -m pytest \
    tests/test_stage3_cov_efficiency.py \
    -v -k "backoff"

python -m pytest \
    tests/ \
    --ignore=tests/test_stage3_cov_efficiency.py \
    -q --tb=short
```
Expected: all PASSED.

---

## Final integration gate

Run after all four tasks committed:

```bash
cd /home/lucas/ai/moe_compress/max_quality
python -m pytest \
    tests/test_stage3_cov_efficiency.py \
    tests/test_stage3_golden_snapshot.py \
    tests/test_stage4_golden_snapshot.py \
    tests/test_stage6_golden_snapshot.py \
    tests/test_stage6alt_golden_snapshot.py \
    tests/test_utils_hooks.py \
    tests/test_activation_hooks_finalize_batch.py \
    tests/test_stage3_plugin_covariance.py \
    tests/test_smoke_stage3.py \
    -v
```

Expected: all PASSED. Zero `MOE_REGEN_GOLDEN`.

---

## Deferred — real-35B GPU validation (NOT runnable on RTX 5080 / this host)

Execute on a GPU box where the 35B model fits (H200 or equivalent):

1. **Byte-identical default gate.** Run Stage-3 at default config (no new knobs). Assert `torch.equal` on every key of the finalized `B_cov` / `C_cov` dicts vs a pre-recorded reference.

2. **Single-pass CPU-hot vs windowed GPU-hot.** With `cov_single_pass: true`, run Stage-3 cov collection on 1-2 layers; assert `torch.equal` on `B_cov` and `C_cov` vs the windowed reference at `cov_batch_size=1`.

3. **`cov_num_sequences: 512` quality gate.** Run the full pipeline (cov → d-rank → α-search → factor) with the override; confirm: (a) `rank_map.json` ranks differ from the 2048-sequence baseline by ≤ 5% of keys; (b) WikiText-2 PPL degrades ≤ 0.05 nats.

4. **Gentler backoff real-world test.** On a near-capacity batch, confirm the ×0.75 backoff lands at a higher batch size than ÷2 would have and that total runtime is lower.

---

## Build sequence checklist

- [ ] **TA.1** `_hot_accum_device` field + `set_hot_accumulator_device()` added; test PASSES
- [ ] **TA.2** `update()` moves result to `_hot_accum_device`; bitwise equality test PASSES
- [ ] **TA.3** GPU `_pending`→CPU invariant test (skip if no CUDA) confirmed
- [ ] **TA.4** `update_cross()` moves result to `_hot_accum_device`; bitwise equality test PASSES
- [ ] **TA.5** `finalize_layer` already-CPU pending test PASSES
- [ ] **TA.6** full suite gate — all PASSED, zero regen
- [ ] **TB.1** `_resolve_cov_window` single-pass + `"all"` → test PASSES
- [ ] **TB.2** single-pass vs windowed bitwise equality test PASSES
- [ ] **TB.3** edge-case resolver tests PASS
- [ ] **TB.4** full suite gate — all PASSED
- [ ] **TC.1** `spec_from_config num_sequences_override` contract test PASSES
- [ ] **TC.2** `_resolve_bcov_spec` helper + orchestrator wire → test PASSES
- [ ] **TC.4** full suite gate — all PASSED
- [ ] **TD.1** ×0.75 backoff sequence test PASSES (CUDA) or SKIPPED (no CUDA)
- [ ] **TD.2** floor re-raise test PASSES (CUDA) or SKIPPED
- [ ] **TD.3** strict-decrease termination test PASSES
- [ ] **TD.4** full suite gate — all PASSED
- [ ] **Final integration gate** — all goldens green, zero regen
```

Then commit with:
```bash
cd /home/lucas/ai/moe_compress
git checkout feat/stage3-cov-efficiency
mkdir -p max_quality/docs/superpowers/plans
# write the file
git add max_quality/docs/superpowers/plans/2026-06-14-stage3-cov-efficiency.md
git commit -m "plan(stage3-cov-eff): single-pass + CPU-accum + cov_num_sequences + gentler backoff"
```

---

Since I cannot write files directly, here is the summary of what I found and what the plan covers:

**Plan path:** `/home/lucas/ai/moe_compress/max_quality/docs/superpowers/plans/2026-06-14-stage3-cov-efficiency.md`

**Task list:**
- **Task A — CPU-resident hot accumulator** (`activation_hooks.py:InputCovarianceAccumulator`): add `_hot_accum_device` field + `set_hot_accumulator_device()` method; migrate result tensor to CPU after GPU GEMM in `update()` (:1021) and `update_cross()` (:1071); `finalize_layer` transparent (`.cpu()` is no-op on already-CPU). 5 TDD steps (TA.1–TA.5).
- **Task B — single-pass G=N** (`covariance_collection.py:_resolve_cov_window` :347-398 + `orchestrator.py`): `cov_single_pass: true` or `cov_window_size: "all"` → return `n_layers`; orchestrator auto-calls `.set_hot_accumulator_device("cpu")` on `B_acc`/`C_acc`. 3 TDD steps (TB.1–TB.3).
- **Task C — `cov_num_sequences` knob** (`orchestrator.py:235`): extract `_resolve_bcov_spec(s3, cal)` helper that calls `spec_from_config(cal, seed_offset=2, num_sequences_override=...)` using the already-existing kwarg; replace the single `spec_from_config` call at :235. 2 TDD steps (TC.1–TC.2).
- **Task D — gentler OOM backoff** (`auto_batch.py:201`): replace `max(attempt // 2, floor)` with `max(min(int(attempt * 0.75), attempt - 1), floor)`; 18→14→10 instead of 18→9. 3 TDD steps (TD.1–TD.3).

Key code references confirmed: `activation_hooks.py:1001-1029` (`update`), `:1051-1080` (`update_cross`), `:1082-1112` (`finalize_layer`), `:933-978` (dataclass fields); `covariance_collection.py:347-398` (`_resolve_cov_window`); `orchestrator.py:235` (`spec = spec_from_config`), `:254-255` (B_acc instantiation); `auto_batch.py:188-203` (`run_with_oom_backoff`); `calibration.py:333-366` (`spec_from_config` — already has `num_sequences_override` kwarg).agentId: a90ed2400ff722e66 (use SendMessage with to: 'a90ed2400ff722e66' to continue this agent)
<usage>subagent_tokens: 96481
tool_uses: 71
duration_ms: 577562</usage>
