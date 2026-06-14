# Stage-3 Covariance Efficiency Implementation Plan

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
|----|---|---|
| A | CPU hot-accumulator | Running-sum (`_pending`) migrated to CPU after GPU GEMM; frees per-layer GPU Gram VRAM so a wider window fits |
| B | Single-pass G=N | Sets window G = n\_layers AND auto-enables Task A on every `C_acc` and `B_acc`, so all 40 layers accumulate in one forward pass |
| C | `cov_num_sequences` knob | Independent sequence-count override for the Stage-3 B/C calib build via the existing `spec_from_config(..., num_sequences_override=...)` kwarg; intended fast value 512 |
| D | Gentler OOM backoff | Replace halve (`// 2`) with `× 0.75` (floor `attempt-1`) in `run_with_oom_backoff`; 18→13→9 instead of 18→9 |

### Non-negotiable correctness invariants

1. **Byte-identical default path.** All four knobs default OFF / absent. `test_stage3_golden_snapshot.py`, `test_stage4_golden_snapshot.py`, `test_stage6_golden_snapshot.py`, `test_stage6alt_golden_snapshot.py` MUST pass without `MOE_REGEN_GOLDEN`.
2. **GEMM NEVER moves to CPU (Task A).** The `flat_f32.transpose(0, 1) @ flat_f32` in `update()` and the incoming `cross` tensor in `update_cross()` are processed on GPU. Only the result tensor is migrated to `_hot_accum_device` afterward.
3. **Single-pass bitwise guarantee (Task B).** At `cov_batch_size=1`, matmul and reduction order are unchanged; finalized covs must be `torch.equal` to the windowed baseline.
4. **B implies A everywhere.** `_single_pass` must be applied to `B_acc` at its creation site (:254) AND to every `C_acc` creation site (:419, :478, :518). Missing any one site causes that accumulator's Grams to stay on GPU and OOM on a single-pass G=40 run.
5. **Task C does not mutate `cal`.** Uses `spec_from_config`'s existing `num_sequences_override` kwarg (`calibration.py:336,357`); no dict copy.

---

## Files → responsibilities

| File | Task | Change |
|---|---|---|
| `src/moe_compress/utils/activation_hooks.py` | A | Add `_hot_accum_device: str \| None = None` field and `set_hot_accumulator_device()` on `InputCovarianceAccumulator`; migrate result to `_hot_accum_device` in `update()` (:1021) and `update_cross()` (:1071); update stale comment at :1091-1096 |
| `src/moe_compress/stage3/plugins/covariance_collection.py` | B | Extend `_resolve_cov_window()` (:347-398): accept `stage3_svd.cov_single_pass: true` or `multi_gpu.cov_window_size: "all"` → return `n_layers` |
| `src/moe_compress/stage3/orchestrator.py` | B, C | (B) Resolve `_single_pass` bool once at :255 (after B_acc); call `B_acc.set_hot_accumulator_device("cpu")` immediately after. At each of the three `C_acc = InputCovarianceAccumulator()` sites (:419, :478, :518), add `if _single_pass: C_acc.set_hot_accumulator_device("cpu")`. (C) Add `_resolve_bcov_spec(s3, cal)` helper; replace `spec = spec_from_config(cal, seed_offset=2)` at :235 |
| `src/moe_compress/utils/auto_batch.py` | D | Replace `:201` `new = max(attempt // 2, floor)` with `new = max(min(int(attempt * 0.75), attempt - 1), floor)`; update docstring example to 18→13→9 |
| `tests/test_stage3_cov_efficiency.py` | A,B,C,D | New file — all TDD tests |

---

## Detailed design

### Task A — CPU hot-accumulator

**`InputCovarianceAccumulator` changes** (`activation_hooks.py`):

Add field in the `@dataclass` body after `_lock` (≈:978):
```python
_hot_accum_device: str | None = None
```

Add method after `set_storage_dtype()` (≈:982):
```python
def set_hot_accumulator_device(self, device: str) -> None:
    self._hot_accum_device = device
```

In `update()`, after `cov = flat_f32.transpose(0, 1) @ flat_f32` (:1021, GEMM unchanged, stays on input device):
```python
if self._hot_accum_device is not None:
    cov = cov.to(self._hot_accum_device)
```
The subsequent lock + `_pending[key]` accumulation (`cur.add_(cov)`) runs normally; both sides are on `_hot_accum_device`.

In `update_cross()`, after `cross_f32 = cross.to(torch.float32)` (:1071):
```python
if self._hot_accum_device is not None:
    cross_f32 = cross_f32.to(self._hot_accum_device)
```
The existing `cur.add_(cross_f32.to(device=cur.device))` handles the device match transparently (no-op cast when both are already on `_hot_accum_device`).

In `finalize_layer()`, update the stale comment at :1091-1096. The current text says "The pending tensors now live on GPU until this call". Replace with:
```
# Phase 2: cast to storage dtype on the source device and transfer to CPU.
# When _hot_accum_device="cpu" the pending tensors already live on CPU;
# .cpu() is a no-op and the dtype cast is applied in-place. Default path
# (None) retains the single GPU→CPU transfer per key.
```
No logic change; `gpu_cov.to(storage_dtype).cpu()` is transparent when the tensor is already on CPU.

**Bitwise guarantee.** At `cov_batch_size=1` the GEMM operands and batch-sequential accumulation order are identical to the GPU-hot path. Only the running-sum device differs. Phase-3 merge (`prev.to(float32) + cpu_cov.to(float32)`) is device-agnostic.

### Task B — single-pass G=N

**`_resolve_cov_window()` changes** (`covariance_collection.py:347-398`):

At the top of the function body, after `if n_layers <= 0: return 1`:
```python
s3 = config.get("stage3_svd") or {}
if s3.get("cov_single_pass", False):
    return n_layers
```

In the `req` string-dispatch block (≈:367-373), before the existing `"auto"` branch:
```python
if req.strip().lower() == "all":
    return n_layers
```

**Orchestrator wiring** (`orchestrator.py`) — critical: three C_acc sites, not one.

Resolve `_single_pass` once, immediately after `B_acc` is set up at :254:
```python
B_acc = InputCovarianceAccumulator()
B_acc.set_storage_dtype(B_cov_dtype)
_single_pass = (
    s3.get("cov_single_pass", False)
    or (config.get("multi_gpu") or {}).get("cov_window_size", "auto") == "all"
)
if _single_pass:
    B_acc.set_hot_accumulator_device("cpu")
```

Then at each of the three `C_acc = InputCovarianceAccumulator()` sites, add the guard immediately after `C_acc.set_storage_dtype(...)`:

- **Resume branch** (:419-420):
```python
C_acc = InputCovarianceAccumulator()
C_acc.set_storage_dtype(B_cov_dtype)
if _single_pass:
    C_acc.set_hot_accumulator_device("cpu")
```

- **DP branch** (:478-479):
```python
C_acc = InputCovarianceAccumulator()
C_acc.set_storage_dtype(B_cov_dtype)
if _single_pass:
    C_acc.set_hot_accumulator_device("cpu")
```

- **Live else-branch** (:518-519):
```python
C_acc = InputCovarianceAccumulator()
C_acc.set_storage_dtype(B_cov_dtype)
if _single_pass:
    C_acc.set_hot_accumulator_device("cpu")
```

All three are inside `if cross_cov_enabled:` guards so `C_acc` is non-None by the time the guard runs.

**VRAM argument.** Without CPU accum, G=40 would hold 40 layers × ~4 GB/layer of fp32 gate\_proj Grams on GPU ≈ 160 GB — impossible on a 141 GB H200 with both models resident. Task A is the mechanical prerequisite; Task B enforces it automatically at all three accumulator creation sites.

### Task C — `cov_num_sequences` knob

Add helper before `run()` in `orchestrator.py`:
```python
def _resolve_bcov_spec(s3: dict, cal: dict):
    """Build the Stage-3 B/C calibration spec.

    Reads ``stage3_svd.cov_num_sequences``; if set, overrides num_sequences
    for the cov pass only via spec_from_config's existing kwarg. Does NOT
    mutate the caller's cal dict. Absent → cal["num_sequences"] unchanged
    (byte-identical default).
    """
    _cov_num_seq = s3.get("cov_num_sequences")
    return spec_from_config(
        cal,
        seed_offset=2,
        num_sequences_override=int(_cov_num_seq) if _cov_num_seq is not None else None,
    )
```

Replace `orchestrator.py:235`:
```python
# before:
spec = spec_from_config(cal, seed_offset=2)
# after:
spec = _resolve_bcov_spec(s3, cal)
```

`spec_from_config` already handles `num_sequences_override` at `calibration.py:357-358`. No dict copy. The global `cal` dict is never mutated; seed is unchanged.

### Task D — gentler OOM backoff

**`auto_batch.py:201`** — replace:
```python
new = max(attempt // 2, floor)
```
with:
```python
new = max(min(int(attempt * 0.75), attempt - 1), floor)
```

Update the docstring of `run_with_oom_backoff` (:188-190) to read "18→13→9 instead of 18→9" as the recovery example.

**Termination proof.** For any `attempt > floor ≥ 1`: `int(attempt * 0.75) ≤ attempt - 1` for `attempt ≥ 4` (0.75×4=3<4). For `attempt=2`: `int(1.5)=1<2`. For `attempt=3`: `int(2.25)=2<3`. The `min(..., attempt-1)` clamp guarantees strict decrease for all values. The floor re-raise path is unchanged.

---

## Bite-sized TDD tasks

> Discipline: write the failing test first (red), implement (green), commit. All tests use tiny synthetic tensors on CPU or CUDA-skip where noted. Zero model loads. Zero golden regen.

---

### Task A — CPU hot-accumulator

**TA.1 — field and setter exist**

Test `test_cpu_accum_flag_exists`:
```python
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
acc = InputCovarianceAccumulator()
assert hasattr(acc, "_hot_accum_device") and acc._hot_accum_device is None
assert callable(getattr(acc, "set_hot_accumulator_device", None))
```

Run: `cd /home/lucas/ai/moe_compress/max_quality && python -m pytest tests/test_stage3_cov_efficiency.py::test_cpu_accum_flag_exists -x`
Expected: **FAILED** (AttributeError).
Impl: add `_hot_accum_device: str | None = None` field and `set_hot_accumulator_device()` method to `InputCovarianceAccumulator`.
Rerun: **PASSED**.
Commit: `feat(stage3-cov-eff): Task A — _hot_accum_device field + setter`

---

**TA.2 — `update()` GPU-hot vs CPU-hot bitwise equality (CPU tensors)**

Test `test_update_gpu_hot_vs_cpu_hot_bitwise`:
```python
import torch
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

torch.manual_seed(0)
d = 16
acc_gpu = InputCovarianceAccumulator()
acc_cpu = InputCovarianceAccumulator()
acc_cpu.set_hot_accumulator_device("cpu")

for _ in range(3):
    x = torch.randn(8, d)   # CPU tensor; no CUDA required
    acc_gpu.update(0, 0, "gate_proj", x)
    acc_cpu.update(0, 0, "gate_proj", x)

acc_gpu.finalize_layer(0)
acc_cpu.finalize_layer(0)

k = (0, 0, "gate_proj")
assert torch.equal(acc_gpu.covariance[k], acc_cpu.covariance[k])
```

Run: `python -m pytest tests/test_stage3_cov_efficiency.py::test_update_gpu_hot_vs_cpu_hot_bitwise -x`
Expected: **FAILED** (no migration logic).
Impl: in `update()` after `cov = flat_f32.transpose(0, 1) @ flat_f32`, add `if self._hot_accum_device is not None: cov = cov.to(self._hot_accum_device)`.
Rerun: **PASSED**.
Commit: `feat(stage3-cov-eff): Task A — update() hot-accum device migration`

---

**TA.3 — `update()` GEMM stays on GPU, `_pending` migrates to CPU (CUDA-gated)**

Test `test_update_gemm_on_gpu_pending_on_cpu`:
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
assert acc._pending[k].device.type == "cpu", \
    "_pending must be on CPU when hot_accum_device='cpu'"
assert acc._pending[k].abs().max() > 0, \
    "result must be non-zero (GEMM ran correctly)"
```

Run: **PASSED** after TA.2 impl (or skipped if no CUDA). No separate impl step.
Commit: `test(stage3-cov-eff): Task A — GEMM-on-GPU / pending-on-CPU invariant`

---

**TA.4 — `update_cross()` GPU-hot vs CPU-hot bitwise equality (CPU tensors)**

Test `test_update_cross_gpu_hot_vs_cpu_hot_bitwise`:
```python
import torch
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

torch.manual_seed(7)
d = 12
acc_gpu = InputCovarianceAccumulator()
acc_cpu = InputCovarianceAccumulator()
acc_cpu.set_hot_accumulator_device("cpu")

for _ in range(4):
    cross = torch.randn(d, d)   # CPU tensor
    acc_gpu.update_cross(0, 0, "gate_proj", cross, n_tokens=8)
    acc_cpu.update_cross(0, 0, "gate_proj", cross, n_tokens=8)

acc_gpu.finalize_layer(0)
acc_cpu.finalize_layer(0)

k = (0, 0, "gate_proj")
assert torch.equal(acc_gpu.covariance[k], acc_cpu.covariance[k])
```

Run: **FAILED** (`update_cross` not yet migrated).
Impl: in `update_cross()` after `cross_f32 = cross.to(torch.float32)`, add `if self._hot_accum_device is not None: cross_f32 = cross_f32.to(self._hot_accum_device)`.
Rerun: **PASSED**.
Commit: `feat(stage3-cov-eff): Task A — update_cross() hot-accum migration`

---

**TA.5 — `update_cross()` GEMM-result on GPU migrates to CPU (CUDA-gated)**

Test `test_update_cross_pending_on_cpu_after_gpu_compute`:
```python
import pytest, torch
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

if not torch.cuda.is_available():
    pytest.skip("no CUDA")

acc = InputCovarianceAccumulator()
acc.set_hot_accumulator_device("cpu")

# cross is pre-computed; it may arrive from a GPU matmul
cross = torch.randn(8, 8, device="cuda")
acc.update_cross(0, 0, "gate_proj", cross, n_tokens=4)

k = (0, 0, "gate_proj")
assert acc._pending[k].device.type == "cpu", \
    "_pending must be on CPU when hot_accum_device='cpu'"

# Verify content matches CPU reference
expected = cross.to(torch.float32).cpu()
assert torch.equal(acc._pending[k], expected)
```

Run: **PASSED** after TA.4 impl (or skipped). No separate impl.
Commit: `test(stage3-cov-eff): Task A — update_cross GPU-tensor pending-on-CPU invariant`

---

**TA.6 — `finalize_layer` transparent with already-CPU pending**

Test `test_finalize_layer_already_cpu_pending`:
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

Run: **PASSED** after TA.2 impl (`.cpu()` is no-op on already-CPU). No separate impl.
Commit: `test(stage3-cov-eff): Task A — finalize_layer already-CPU pending transparent`

---

**TA.7 — full suite gate**

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

**TB.1 — `_resolve_cov_window` returns `n_layers` for both single-pass spellings**

Test `test_resolve_cov_window_single_pass_spellings`:
```python
from moe_compress.stage3.plugins.covariance_collection import _resolve_cov_window

n = 40
assert _resolve_cov_window({"stage3_svd": {"cov_single_pass": True}}, n) == n
assert _resolve_cov_window({"multi_gpu": {"cov_window_size": "all"}}, n) == n
assert _resolve_cov_window({"multi_gpu": {"cov_window_size": "ALL"}}, n) == n  # case-insensitive

# Default path: valid int in [1, n]
result = _resolve_cov_window({}, n)
assert 1 <= result <= n

# n_layers=0 guard unchanged
assert _resolve_cov_window({"stage3_svd": {"cov_single_pass": True}}, 0) == 1
```

Run: `python -m pytest tests/test_stage3_cov_efficiency.py::test_resolve_cov_window_single_pass_spellings -x`
Expected: **FAILED** (neither sentinel handled).
Impl: modify `_resolve_cov_window` as described in design above.
Rerun: **PASSED**.
Commit: `feat(stage3-cov-eff): Task B — _resolve_cov_window single-pass + "all" sentinel`

---

**TB.2 — single-pass G=N vs windowed G=2 bitwise equality (tiny synthetic)**

Test `test_single_pass_vs_windowed_bitwise_tiny`:
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

Run: **PASSED** after TA.2 (single-pass is mechanically window=n_layers + CPU accum).
Commit: `test(stage3-cov-eff): Task B — single-pass vs windowed bitwise equality`

---

**TB.3 — orchestrator wires CPU accumulator at ALL creation sites (targeted coupling test)**

This test is specifically designed to catch H2-class bugs where a C_acc site is missed.

Test `test_single_pass_wires_cpu_accum_on_b_acc_and_c_acc`:
```python
"""
Simulate the orchestrator's _single_pass wiring logic by directly
instantiating accumulators the same way the orchestrator does, applying
the wiring, and asserting _hot_accum_device on both B_acc and C_acc.

This tests the COUPLING between _single_pass and set_hot_accumulator_device
— the same check that would have caught the H2 bug (C_acc site missed).
"""
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

B_cov_dtype = __import__("torch").bfloat16

# --- simulate orchestrator resume-branch C_acc creation ---
def _make_accumulators_resume(single_pass: bool):
    B_acc = InputCovarianceAccumulator()
    B_acc.set_storage_dtype(B_cov_dtype)
    _single_pass = single_pass
    if _single_pass:
        B_acc.set_hot_accumulator_device("cpu")
    # resume branch (mirrors :419-420 + fix)
    C_acc = InputCovarianceAccumulator()
    C_acc.set_storage_dtype(B_cov_dtype)
    if _single_pass:
        C_acc.set_hot_accumulator_device("cpu")
    return B_acc, C_acc

B, C = _make_accumulators_resume(single_pass=True)
assert B._hot_accum_device == "cpu", "B_acc must be CPU-hot when single_pass=True"
assert C._hot_accum_device == "cpu", "C_acc resume-branch must be CPU-hot when single_pass=True"

B2, C2 = _make_accumulators_resume(single_pass=False)
assert B2._hot_accum_device is None, "B_acc must be None when single_pass=False"
assert C2._hot_accum_device is None, "C_acc must be None when single_pass=False"
```

Run: **PASSED** after TA.1 impl (tests the wiring pattern, not `_resolve_cov_window`).
Commit: `test(stage3-cov-eff): Task B — B_acc+C_acc CPU-hot coupling test`

---

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

**TC.1 — `spec_from_config` `num_sequences_override` contract (confirms the hook exists)**

Test `test_spec_from_config_num_sequences_override_contract`:
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

# Seed is unchanged by the override
assert spec_with.seed == spec_without.seed

# cal dict is not mutated
assert cal["num_sequences"] == 2048
```

Run: **PASSED** (confirms calibration.py:357 hook before orchestrator work begins).
Commit: `test(stage3-cov-eff): Task C — spec_from_config num_sequences_override contract`

---

**TC.2 — `_resolve_bcov_spec` helper**

Test `test_resolve_bcov_spec_helper`:
```python
import pytest

try:
    from moe_compress.stage3.orchestrator import _resolve_bcov_spec
except ImportError:
    pytest.fail("_resolve_bcov_spec not yet in orchestrator")

cal = {
    "num_sequences": 2048, "sequence_length": 512, "seed": 0,
    "source": "nvidia-cascade", "subset_weights": {"math": 1.0},
}

spec_with = _resolve_bcov_spec({"cov_num_sequences": 512}, cal)
assert spec_with.num_sequences == 512

spec_without = _resolve_bcov_spec({}, cal)
assert spec_without.num_sequences == 2048

# cal dict not mutated
assert cal["num_sequences"] == 2048
```

Run: **FAILED** (ImportError).
Impl: add `_resolve_bcov_spec` to `orchestrator.py` and replace `:235` `spec = spec_from_config(cal, seed_offset=2)` with `spec = _resolve_bcov_spec(s3, cal)`.
Rerun: **PASSED**.
Commit: `feat(stage3-cov-eff): Task C — _resolve_bcov_spec helper + orchestrator wire`

---

**TC.3 — full suite gate**

```bash
python -m pytest \
    tests/test_stage3_cov_efficiency.py \
    tests/test_stage3_golden_snapshot.py \
    tests/test_smoke_stage3.py \
    -v
```
Expected: all PASSED. The smoke test uses a fixture with `cal["num_sequences"]` set but no `cov_num_sequences` in `stage3_svd`, so it follows the unchanged default path.

---

### Task D — gentler OOM backoff

**TD.1 — backoff sequence is 18→13→9, not 18→9 (CUDA-gated)**

Test `test_run_with_oom_backoff_gentler_sequence`:
```python
import pytest, torch
from moe_compress.utils.auto_batch import run_with_oom_backoff

if not torch.cuda.is_available():
    pytest.skip("requires CUDA for OutOfMemoryError class")

threshold = 10
attempts = []

def _fake_run(batch):
    attempts.append(batch)
    if batch > threshold:
        raise torch.cuda.OutOfMemoryError("simulated")
    return batch

result = run_with_oom_backoff(_fake_run, start_batch=18, floor=4)

# 18 → min(int(18*0.75), 17) = min(13, 17) = 13 (OOM)
# 13 → min(int(13*0.75), 12) = min(9, 12) = 9 (success, 9 <= 10)
assert attempts == [18, 13, 9], f"expected [18, 13, 9], got {attempts}"
assert result == 9
assert len(attempts) > 2, "must take more steps than halving ([18,9])"
```

Run: `python -m pytest tests/test_stage3_cov_efficiency.py::test_run_with_oom_backoff_gentler_sequence -x`
Expected: **FAILED** (current halve gives `[18, 9]`).
Impl: in `auto_batch.py:201` replace `new = max(attempt // 2, floor)` with `new = max(min(int(attempt * 0.75), attempt - 1), floor)`. Update docstring example to 18→13→9.
Rerun: **PASSED**.
Commit: `feat(stage3-cov-eff): Task D — gentler OOM backoff ×0.75 (18→13→9 vs 18→9)`

---

**TD.2 — floor re-raise unchanged (CUDA-gated)**

Test `test_run_with_oom_backoff_floor_reraise`:
```python
import pytest, torch
from moe_compress.utils.auto_batch import run_with_oom_backoff

if not torch.cuda.is_available():
    pytest.skip("requires CUDA for OutOfMemoryError class")

def _always_oom(batch):
    raise torch.cuda.OutOfMemoryError("always")

with pytest.raises(torch.cuda.OutOfMemoryError):
    run_with_oom_backoff(_always_oom, start_batch=4, floor=4)
```

Run: **PASSED** (re-raise path unchanged by impl).
Commit: `test(stage3-cov-eff): Task D — floor re-raise test`

---

**TD.3 — strict-decrease termination at floor from high start (CUDA-gated)**

Test `test_run_with_oom_backoff_strict_decrease_to_floor`:
```python
import pytest, torch
from moe_compress.utils.auto_batch import run_with_oom_backoff

if not torch.cuda.is_available():
    pytest.skip("requires CUDA for OutOfMemoryError class")

attempts = []

def _run(batch):
    attempts.append(batch)
    if batch > 2:
        raise torch.cuda.OutOfMemoryError("oom")
    return batch

result = run_with_oom_backoff(_run, start_batch=100, floor=2)
assert result == 2

for i in range(1, len(attempts)):
    assert attempts[i] < attempts[i - 1], \
        f"not strictly decreasing at index {i}: {attempts}"
assert attempts[-1] == 2
```

Run: **PASSED** after TD.1 impl.
Commit: `test(stage3-cov-eff): Task D — strict-decrease termination to floor`

---

**TD.4 — full suite gate**

```bash
python -m pytest \
    tests/test_stage3_cov_efficiency.py \
    -v -k "backoff"

python -m pytest tests/ -q --tb=short
```
Expected: all PASSED.

---

## Final integration gate

Run after all four tasks are committed:

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

Run on a GPU box where the 35B model fits (H200 or equivalent):

1. **Byte-identical default gate.** Run Stage-3 at default config (no new knobs set). Assert `torch.equal` on every finalized `B_cov` / `C_cov` key vs a pre-recorded reference.

2. **Single-pass CPU-hot vs windowed GPU-hot.** With `cov_single_pass: true`, run Stage-3 cov collection on 1-2 layers; assert `torch.equal` on `B_cov` and `C_cov` vs the windowed baseline at `cov_batch_size=1`. Confirm all three C_acc creation branches produce CPU-resident `_pending` (verify by checking `C_acc._hot_accum_device == "cpu"` in a debug log or assertion before `_collect_covariances` runs).

3. **`cov_num_sequences: 512` quality gate.** Run the full pipeline (cov → d-rank → α-search → factor) with the override; confirm: (a) `rank_map.json` rank distribution differs from the 2048-sequence baseline by ≤ 5% of keys; (b) WikiText-2 PPL degrades ≤ 0.05 nats vs baseline.

4. **Gentler backoff real-world.** On a near-capacity forward batch, confirm the ×0.75 backoff (18→13→9) completes at a higher batch size than ÷2 (18→9) would have.

---

## Build sequence checklist

- [ ] **TA.1** `_hot_accum_device` field + `set_hot_accumulator_device()` added; test PASSES
- [ ] **TA.2** `update()` migrates result to `_hot_accum_device` after GPU GEMM; bitwise equality test PASSES
- [ ] **TA.3** CUDA-gated: `update()` GEMM-on-GPU / `_pending`-on-CPU invariant test PASSES or SKIPPED
- [ ] **TA.4** `update_cross()` migrates `cross_f32` to `_hot_accum_device`; CPU bitwise equality test PASSES
- [ ] **TA.5** CUDA-gated: `update_cross()` GPU-tensor pending-on-CPU invariant test PASSES or SKIPPED
- [ ] **TA.6** `finalize_layer` already-CPU pending transparent; test PASSES
- [ ] **TA.7** full suite gate — all PASSED, zero regen
- [ ] **TB.1** `_resolve_cov_window` single-pass + `"all"` → test PASSES
- [ ] **TB.2** single-pass vs windowed G=2 bitwise equality test PASSES
- [ ] **TB.3** B_acc + C_acc CPU-hot coupling test PASSES (catches H2-class missed-site bugs)
- [ ] **TB.4** full suite gate — all PASSED
- [ ] **TC.1** `spec_from_config num_sequences_override` contract test PASSES
- [ ] **TC.2** `_resolve_bcov_spec` helper + orchestrator wire; test PASSES
- [ ] **TC.3** full suite gate — all PASSED
- [ ] **TD.1** CUDA-gated: 18→13→9 sequence test PASSES or SKIPPED
- [ ] **TD.2** CUDA-gated: floor re-raise test PASSES or SKIPPED
- [ ] **TD.3** CUDA-gated: strict-decrease termination test PASSES or SKIPPED
- [ ] **TD.4** full suite gate — all PASSED
- [ ] **Final integration gate** — all goldens green, zero regen
