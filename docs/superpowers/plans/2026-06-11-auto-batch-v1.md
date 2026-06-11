# Auto-Batch v1 (Resolver Infra + Bounded-Plugin Wiring) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build per-plugin VRAM-aware auto-batch infra — **`size_batch`** (cost-model PREDICTION) + **`run_with_oom_backoff`** (run the REAL pass; on OOM halve & rerun to the fixed floor) — default-OFF, and wire it into one `batch-invariant` plugin (`ma_detection` phase-a) as the integration proof, with goldens byte-identical because the feature is off by default.

**Architecture:** A standalone, GPU-free-testable `utils/auto_batch.py`. There is **NO subset allclose self-test** (the spec rev4 removed it — a subset probe can't run a real W-wide forward over <W rows, and is blind to N-scaling drift). Instead: (1) `size_batch` runs a tiny bs=1/bs=2 **cost probe** (identical forward) to fit `cost(b)=fixed+b·per_sample`, sizes `candidate` from `usable=total−allocated(baseline)−headroom`, and returns a prediction (degrading to the floor on a bad probe, never raising); (2) `run_with_oom_backoff` runs the plugin's **actual** pass at the prediction and, on `torch.cuda.OutOfMemoryError`, `empty_cache()`s, halves, and **reruns** down to the fixed floor (a re-raised OOM after empty_cache is a hard-fail → floor). **Drift-safety is by fidelity class, not a probe**: only `batch-invariant` plugins (ma_detection's max/percentile — grouping-independent) are wired in v1; `reduction-accumulating` (cov/NLL) is deferred to v2 pinning. A thin `resolve_batch(...)` = class-gate + `size_batch` keeps `_V1_ELIGIBLE` meaningful (refuses non-eligible/disabled → fixed batch).

**Tech Stack:** Python 3, PyTorch (`torch.cuda` memory APIs), pytest. Code root `max_quality/`; all commands run from there.

**Spec:** `docs/superpowers/specs/2026-06-11-per-plugin-vram-aware-auto-batch-sizing-design.md` (commit `0b7c82c`, rev5). Implements §3 (classes), §4a/4b/4c (size + run + drift-by-class), §10 step 1 (v1 infra + batch-invariant plugin) ONLY. §5 v2 / cov / NLL / DP / model-shard out of scope.

> **REWORK NOTE:** Tasks 1–3 are already implemented + reviewed (commits `b7b0cdb`, `6b48770`, `b1a2d87`, `880f5b0`). Task 1's `AutoBatchConfig` must DROP `rtol`/`atol` (no allclose anymore). Tasks 2 (fit_cost/size_candidate) + 3 (MemProbe) survive AS-IS. The committed `resolve_batch` (with allclose self-test, commits `1cf8b3f`/`8946d03`) and the ma_detection wiring (`2208738`) are REPLACED by Tasks 4–6 below. Implementers refactor the existing file, not greenfield.

---

## File Structure

- **Create/refactor** `src/moe_compress/utils/auto_batch.py` — `FidelityClass`, `AutoBatchConfig` (no rtol/atol), `MemProbe`/`CudaMemProbe`, `fit_cost`/`size_candidate` (kept), `size_batch(...)` (cost-probe → prediction), `resolve_batch(...)` (class-gate + size_batch), `run_with_oom_backoff(...)`. No plugin-specific logic.
- **Create/refactor** `tests/test_auto_batch.py` — GPU-free unit coverage with fake `MemProbe` + synthetic `cost_probe_fn`/`run_fn`. Remove the old allclose-self-test tests; add `size_batch` (incl. bad-probe→floor) and `run_with_oom_backoff` (asserting the real re-invocation sequence at smaller batch).
- **Modify** `src/moe_compress/stage1/plugins/ma_detection.py` — at the **phase-a** `phase_a_batch_size` site **only** (`ma_detection.py:366`), call `resolve_batch` ONLY when `auto_batch.enabled`; otherwise the existing constant. Phase A declared `FidelityClass.BATCH_INVARIANT` (per-layer `abs().max()` running max + Q99 percentile over the full token multiset are grouping-independent). **Phase B is explicitly EXCLUDED from v1** — its `run_calibration_pass` builds covariance (Gram), output-reservoir, and downproj-max accumulators, which are `reduction-accumulating` (drift grows with N) → deferred to v2 alongside cov. Do NOT touch `phase_b_batch_size` (`stage1/orchestrator.py:541`).
- **Create** `tests/test_ma_detection_auto_batch.py` — (a) default-off ⇒ resolver never called, batch == constant; (b) flag-on with injected fake probe ⇒ resolver invoked, returns ≥ fixed floor.
- **Modify** `tests/golden/stage1/` — NOT TOUCHED. Goldens must stay byte-identical (feature default-off). This is a guardrail, listed so the implementer knows not to re-bless.

---

## Conventions (read once)

- Module logger: `import logging; log = logging.getLogger(__name__)` (matches `ma_detection.py:235`).
- `OutOfMemoryError`: `torch.cuda.OutOfMemoryError` (alias `torch.OutOfMemoryError`). Tests raise it from the fake `probe_fn`.
- Never call real `torch.cuda` in unit tests — inject the fake `MemProbe` and a synthetic `probe_fn`. CI has no GPU.
- Commit after every green step. Use `feat(auto-batch): …` messages, end with the Co-Authored-By trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 1: Data structures — `FidelityClass`, `AutoBatchConfig`

**Files:**
- Create: `src/moe_compress/utils/auto_batch.py`
- Test: `tests/test_auto_batch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auto_batch.py
import pytest
from moe_compress.utils.auto_batch import FidelityClass, AutoBatchConfig

def test_fidelity_classes_exist():
    assert {c.value for c in FidelityClass} == {
        "metric_pinned", "batch_invariant",
        "reduction_bounded", "reduction_accumulating",
    }

def test_autobatchconfig_defaults_disabled():
    c = AutoBatchConfig()
    assert c.enabled is False            # default-OFF: goldens byte-identical
    assert c.headroom_frac == 0.1
    assert c.max_cap == 4096
    assert c.probe_samples == 4
    assert not hasattr(c, "rtol") and not hasattr(c, "atol")  # no allclose self-test

def test_autobatchconfig_from_dict_ignores_unknown():
    c = AutoBatchConfig.from_dict({"enabled": True, "headroom_frac": 0.2, "bogus": 9})
    assert c.enabled is True and c.headroom_frac == 0.2
    assert c.max_cap == 4096             # untouched default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_auto_batch.py -q`
Expected: FAIL (`ModuleNotFoundError: moe_compress.utils.auto_batch`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/moe_compress/utils/auto_batch.py
"""VRAM-aware per-plugin auto-batch resolver (v1: bounded/invariant plugins).

See docs/superpowers/specs/2026-06-11-per-plugin-vram-aware-auto-batch-sizing-design.md.
Default-OFF: when AutoBatchConfig.enabled is False the resolver is never invoked
and every golden stays byte-identical. NOT for reduction-accumulating plugins
(cov Gram / NLL) — those need the v2 pinned-reduction work first.
"""
from __future__ import annotations
import dataclasses
import enum
import logging

log = logging.getLogger(__name__)


class FidelityClass(enum.Enum):
    METRIC_PINNED = "metric_pinned"
    BATCH_INVARIANT = "batch_invariant"
    REDUCTION_BOUNDED = "reduction_bounded"
    REDUCTION_ACCUMULATING = "reduction_accumulating"


# Classes the v1 resolver may auto-size. Only BATCH_INVARIANT is proven-safe with
# NO numeric check (grouping-independent reduction); REDUCTION_BOUNDED is declared
# but NOT auto-wired in v1 (needs its own bounded-drift argument first, per spec
# §4c) — there is no subset self-test to fall back on, so we don't trust it yet.
_V1_ELIGIBLE = frozenset({FidelityClass.BATCH_INVARIANT})


@dataclasses.dataclass(frozen=True)
class AutoBatchConfig:
    enabled: bool = False
    headroom_frac: float = 0.1
    max_cap: int = 4096
    probe_samples: int = 4

    @classmethod
    def from_dict(cls, d: dict | None) -> "AutoBatchConfig":
        d = d or {}
        fields = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in fields}
        # Safety contract is "OFF unless explicitly on": coerce enabled to a real
        # bool so a stray YAML string ("false") cannot silently enable the resolver
        # and break the byte-identical guarantee.
        if "enabled" in kw:
            kw["enabled"] = (kw["enabled"] is True) or (str(kw["enabled"]).strip().lower() == "true")
        return cls(**kw)
```

Add to the Task-1 test (Step 1): `assert AutoBatchConfig.from_dict({"enabled": "false"}).enabled is False` and `assert AutoBatchConfig.from_dict({"enabled": "true"}).enabled is True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_auto_batch.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/moe_compress/utils/auto_batch.py tests/test_auto_batch.py
git commit -m "feat(auto-batch): FidelityClass + AutoBatchConfig (default-off)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Pure sizing math — two-point cost fit + usable-free → candidate

**Files:**
- Modify: `src/moe_compress/utils/auto_batch.py`
- Test: `tests/test_auto_batch.py`

Rationale (spec §4 steps 2–3): `cost(b)=fixed+b·per_sample`; from peaks at b=1,2: `per_sample=peak2−peak1`, `fixed=2·peak1−peak2`. `usable=total−allocated_baseline−headroom_frac·total`; `candidate=floor((usable−fixed)/per_sample)` clamped to `[fixed_batch, max_cap]`. `fixed` is subtracted exactly once (NOT folded into headroom).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auto_batch.py  (append)
from moe_compress.utils.auto_batch import fit_cost, size_candidate

def test_fit_cost_separates_fixed_and_per_sample():
    # peak1 = fixed + 1*per ; peak2 = fixed + 2*per
    fixed, per = fit_cost(peak1=30, peak2=50)
    assert fixed == 10 and per == 20

def test_fit_cost_rejects_nonincreasing():
    # bs=2 must cost more than bs=1; otherwise measurement is unusable
    with pytest.raises(ValueError):
        fit_cost(peak1=50, peak2=50)

def test_size_candidate_basic():
    # usable = 100 - 10 - 0.1*100 = 80 ; (80-10)/20 = 3.5 -> floor 3
    cand = size_candidate(total=100, allocated_baseline=10, headroom_frac=0.1,
                          fixed=10, per_sample=20, fixed_batch=1, max_cap=4096)
    assert cand == 3

def test_size_candidate_clamps_to_floor_and_cap():
    # tiny usable -> never below fixed_batch
    assert size_candidate(total=20, allocated_baseline=18, headroom_frac=0.1,
                          fixed=5, per_sample=5, fixed_batch=8, max_cap=4096) == 8
    # huge usable -> capped
    assert size_candidate(total=10**12, allocated_baseline=0, headroom_frac=0.0,
                          fixed=0, per_sample=1, fixed_batch=1, max_cap=64) == 64
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_auto_batch.py -q` → FAIL (`fit_cost`/`size_candidate` undefined).

- [ ] **Step 3: Implement**

```python
# auto_batch.py  (append)
import math


def fit_cost(*, peak1: int, peak2: int) -> tuple[float, float]:
    """Fit cost(b)=fixed+b*per_sample from single-forward peaks at b=1 and b=2."""
    per_sample = float(peak2) - float(peak1)
    if per_sample <= 0:
        raise ValueError(f"non-increasing probe peaks: peak1={peak1} peak2={peak2}")
    # Spec §4 formula is fixed = 2*peak1 - peak2 (unclamped). We clamp to >=0 to
    # guard against a noisy probe yielding a spurious negative fixed; this does
    # not change the b=1,2 slope (per_sample) used for sizing.
    fixed = 2.0 * float(peak1) - float(peak2)
    return max(fixed, 0.0), per_sample


def size_candidate(*, total: int, allocated_baseline: int, headroom_frac: float,
                   fixed: float, per_sample: float, fixed_batch: int,
                   max_cap: int) -> int:
    """Largest batch whose predicted peak fits usable VRAM, clamped to [floor, cap].

    Precondition: fixed_batch <= max_cap. The floor wins over the cap (we never
    return below the caller's proven-safe fixed batch), so a misconfigured
    fixed_batch > max_cap would defeat the cap — reject it loudly instead.
    """
    if fixed_batch > max_cap:
        raise ValueError(f"fixed_batch={fixed_batch} exceeds max_cap={max_cap}")
    headroom = headroom_frac * float(total)
    usable = float(total) - float(allocated_baseline) - headroom
    raw = math.floor((usable - fixed) / per_sample) if per_sample > 0 else fixed_batch
    return int(max(fixed_batch, min(raw, max_cap)))
```

Add to Task-2 test (Step 1): `with pytest.raises(ValueError): size_candidate(total=100, allocated_baseline=0, headroom_frac=0.0, fixed=0, per_sample=1, fixed_batch=8192, max_cap=4096)`.

- [ ] **Step 4: Run** `python3 -m pytest tests/test_auto_batch.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(auto-batch): two-point cost fit + usable-free sizing math

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `MemProbe` abstraction (injectable CUDA-mem accessor)

**Files:**
- Modify: `src/moe_compress/utils/auto_batch.py`
- Test: `tests/test_auto_batch.py`

Rationale: keep the resolver GPU-free-testable. `MemProbe` exposes `total()`, `allocated()`, `reset_peak()`, `peak()`. Default `CudaMemProbe` wraps `torch.cuda`; tests inject a fake.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auto_batch.py (append)
from moe_compress.utils.auto_batch import MemProbe

class FakeMem(MemProbe):
    def __init__(self, total, allocated): self._t=total; self._a=allocated; self._peak=0
    def total(self): return self._t
    def allocated(self): return self._a
    def reset_peak(self): self._peak = self._a
    def peak(self): return self._peak
    def set_peak(self, v): self._peak = v   # test helper

def test_memprobe_is_subclassable():
    m = FakeMem(total=100, allocated=10)
    m.reset_peak(); m.set_peak(42)
    assert m.total()==100 and m.allocated()==10 and m.peak()==42
```

- [ ] **Step 2: Run** → FAIL (`MemProbe` undefined).

- [ ] **Step 3: Implement**

```python
# auto_batch.py (append)
class MemProbe:
    """Injectable CUDA-memory accessor (bytes). Override in tests."""
    def total(self) -> int: raise NotImplementedError
    def allocated(self) -> int: raise NotImplementedError
    def reset_peak(self) -> None: raise NotImplementedError
    def peak(self) -> int: raise NotImplementedError


class CudaMemProbe(MemProbe):
    def __init__(self, device=None):
        import torch
        self._torch = torch
        self._device = device
    def total(self) -> int:
        _free, total = self._torch.cuda.mem_get_info(self._device)
        return int(total)
    def allocated(self) -> int:
        return int(self._torch.cuda.memory_allocated(self._device))
    def reset_peak(self) -> None:
        self._torch.cuda.reset_peak_memory_stats(self._device)
    def peak(self) -> int:
        return int(self._torch.cuda.max_memory_allocated(self._device))
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(auto-batch): injectable MemProbe (CudaMemProbe default)`.

---

## Task 4: `size_batch` (cost-model prediction) + `resolve_batch` (class-gate)

**Files:** Modify `src/moe_compress/utils/auto_batch.py`; Test `tests/test_auto_batch.py`.

**REWORK:** delete the committed `resolve_batch` allclose-self-test body and its tests; replace with the below. The cost probe now returns **only a peak** (no reference output — there is no allclose).

Contract: `cost_probe_fn(micro_batch:int) -> int` runs ONE forward of `micro_batch` sequences with the plugin's identical forward signature and returns the peak allocated bytes. `size_batch` calls it at 1 and 2 only.

- [ ] **Step 1: Failing tests**

```python
# tests/test_auto_batch.py (replace the old resolve_batch tests)
import torch
from moe_compress.utils.auto_batch import size_batch, resolve_batch, FidelityClass, AutoBatchConfig

def test_size_batch_predicts_from_cost_probe():
    mem = FakeMem(total=1000, allocated=100)            # usable=1000-100-0.1*1000=800
    peaks = {1: 200, 2: 300}                            # per=100, fixed=100 -> (800-100)/100=7
    calls = []
    def cost_probe_fn(mb): calls.append(mb); return peaks[mb]
    bs = size_batch(cost_probe_fn, fixed_batch=1, headroom_frac=0.1, max_cap=4096, mem=mem)
    assert bs == 7 and calls == [1, 2]                  # ONLY the two cost probes

def test_size_batch_bad_probe_degrades_to_floor_no_raise():
    # non-increasing peaks (e.g. probe_samples<2 / noise) -> fit_cost would raise -> must return floor
    def cost_probe_fn(mb): return 500
    bs = size_batch(cost_probe_fn, fixed_batch=8, headroom_frac=0.1, max_cap=4096,
                    mem=FakeMem(1000, 0))
    assert bs == 8                                       # degraded, did NOT raise

def test_size_batch_cost_probe_oom_degrades_to_floor():
    def cost_probe_fn(mb):
        if mb == 2: raise torch.cuda.OutOfMemoryError("boom")
        return 200
    bs = size_batch(cost_probe_fn, fixed_batch=4, headroom_frac=0.1, max_cap=4096,
                    mem=FakeMem(1000, 0))
    assert bs == 4

def test_resolve_batch_class_gate():
    cp = lambda mb: {1:200,2:300}[mb]
    cfg_on = AutoBatchConfig(enabled=True)
    # eligible + on -> sized
    assert resolve_batch(cp, fixed_batch=1, fidelity_class=FidelityClass.BATCH_INVARIANT,
                         cfg=cfg_on, mem=FakeMem(1000,100)) == 7
    # disabled -> fixed, no probe
    seen = []
    cp2 = lambda mb: seen.append(mb) or 200
    assert resolve_batch(cp2, fixed_batch=8, fidelity_class=FidelityClass.BATCH_INVARIANT,
                         cfg=AutoBatchConfig(enabled=False), mem=FakeMem(1000,0)) == 8 and seen == []
    # ineligible class (accumulating) -> fixed, no probe
    assert resolve_batch(cp2, fixed_batch=8, fidelity_class=FidelityClass.REDUCTION_ACCUMULATING,
                         cfg=cfg_on, mem=FakeMem(1000,0)) == 8 and seen == []
```

- [ ] **Step 2: Run** → FAIL (`size_batch`/new `resolve_batch` undefined).

- [ ] **Step 3: Implement**

```python
# auto_batch.py
def size_batch(cost_probe_fn, fixed_batch: int, *, headroom_frac: float,
               max_cap: int, mem: MemProbe | None = None) -> int:
    """Predict the largest forward batch from a two-point cost probe. Never raises:
    a bad/non-increasing/OOMing probe degrades to fixed_batch."""
    import torch
    if mem is None:
        mem = CudaMemProbe()
    baseline = mem.allocated()
    try:
        mem.reset_peak(); peak1 = cost_probe_fn(1)
        mem.reset_peak(); peak2 = cost_probe_fn(2)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.warning("auto_batch: OOM during cost probe; using floor %d", fixed_batch)
        return int(fixed_batch)
    try:
        fixed, per = fit_cost(peak1=peak1, peak2=peak2)
    except ValueError as exc:
        log.warning("auto_batch: cost probe unusable (%s); using floor %d", exc, fixed_batch)
        return int(fixed_batch)
    candidate = size_candidate(total=mem.total(), allocated_baseline=baseline,
                               headroom_frac=headroom_frac, fixed=fixed, per_sample=per,
                               fixed_batch=fixed_batch, max_cap=max_cap)
    log.info("auto_batch: predicted batch=%d (floor=%d fixed=%.3g per=%.3g)",
             candidate, fixed_batch, fixed, per)
    return int(candidate)


def resolve_batch(cost_probe_fn, fixed_batch: int, fidelity_class: FidelityClass,
                  cfg: AutoBatchConfig, mem: MemProbe | None = None) -> int:
    """Class-gated sizing. Returns fixed_batch (no probe) when disabled or the
    fidelity class is not v1-eligible; else the size_batch prediction."""
    if not cfg.enabled or fidelity_class not in _V1_ELIGIBLE:
        return int(fixed_batch)
    return size_batch(cost_probe_fn, fixed_batch, headroom_frac=cfg.headroom_frac,
                      max_cap=cfg.max_cap, mem=mem)
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(auto-batch): size_batch prediction + class-gated resolve_batch (no self-test)`.

---

## Task 5: `run_with_oom_backoff` — the real run IS the fit test

**Files:** Modify `src/moe_compress/utils/auto_batch.py`; Test `tests/test_auto_batch.py`.

Contract: `run_fn(batch:int) -> result` runs the plugin's ACTUAL pass at `batch` (idempotent — fresh accumulators). `run_with_oom_backoff` runs it at `start_batch`; on `torch.cuda.OutOfMemoryError` it `empty_cache()`s, halves toward `floor`, and reruns; if the floor itself OOMs it re-raises (genuinely unrecoverable). Returns `run_fn`'s result at the adopted batch.

- [ ] **Step 1: Failing tests**

```python
# tests/test_auto_batch.py
from moe_compress.utils.auto_batch import run_with_oom_backoff

def test_run_backoff_adopts_start_when_fits():
    calls = []
    def run_fn(b): calls.append(b); return f"ok@{b}"
    assert run_with_oom_backoff(run_fn, start_batch=32, floor=1) == "ok@32" and calls == [32]

def test_run_backoff_halves_on_oom_and_reruns():
    calls = []
    def run_fn(b):
        calls.append(b)
        if b > 8: raise torch.cuda.OutOfMemoryError("boom")   # 32,16 OOM; 8 fits
        return f"ok@{b}"
    assert run_with_oom_backoff(run_fn, start_batch=32, floor=1) == "ok@8"
    assert calls == [32, 16, 8]                                 # REAL re-invocation at each smaller batch

def test_run_backoff_never_below_floor_and_reraises_if_floor_ooms():
    calls = []
    def run_fn(b):
        calls.append(b); raise torch.cuda.OutOfMemoryError("boom")  # everything OOMs
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_with_oom_backoff(run_fn, start_batch=8, floor=4)
    assert calls == [8, 4]                                       # halved 8->4, floor OOM -> reraise, never <4
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement:**

```python
# auto_batch.py
def run_with_oom_backoff(run_fn, start_batch: int, floor: int):
    """Run run_fn(batch) at start_batch; on CUDA OOM empty_cache + halve toward
    floor and rerun. The real pass is the fit test. Re-raises if the floor OOMs."""
    import torch
    attempt = max(int(start_batch), int(floor))
    while True:
        try:
            return run_fn(attempt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt <= floor:
                log.error("auto_batch: OOM at floor batch=%d; unrecoverable", floor)
                raise
            new = max(attempt // 2, floor)
            log.warning("auto_batch: OOM at batch=%d; retrying at %d", attempt, new)
            attempt = new
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(auto-batch): run_with_oom_backoff (real-run halving to floor)`.

---

## Task 6: Wire `ma_detection` phase-a (default-off) + golden guardrail

**Files:** Modify `src/moe_compress/stage1/plugins/ma_detection.py` (phase-a site `:366`, ONLY); Test `tests/test_ma_detection_auto_batch.py`.

**REWORK** the committed wiring (`2208738`). Keep the factored `_run_phase_a_collect(model, batches, device) -> (layer_max, moe_block_max, q99)` (already extracted). New control flow at the phase-a site:

- `cfg = AutoBatchConfig.from_dict(s1.get("auto_batch"))`.
- **Disabled (default):** EXACTLY the original path — `bs = int(s1.get("phase_a_batch_size", _PHASE_A_BATCH_SIZE))`, then run the pass once via `_run_phase_a_collect(model, iter_batches(calib, bs), device)`. No probe, no wrapper → byte-identical, golden unchanged.
- **Enabled:** `bs = resolve_batch(cost_probe_fn, fixed_batch=_PHASE_A_BATCH_SIZE, fidelity_class=FidelityClass.BATCH_INVARIANT, cfg=cfg)` where `cost_probe_fn(mb)` runs ONE forward of `mb` sequences (`_run_phase_a_collect(model, iter_batches(calib[:max(2, mb)], mb), device)`) and returns `torch.cuda.max_memory_allocated(device)`; then run the real pass via `run_with_oom_backoff(lambda b: _run_phase_a_collect(model, iter_batches(calib, b), device), start_batch=bs, floor=_PHASE_A_BATCH_SIZE)`. The `(layer_max, moe_block_max, q99)` result feeds the unchanged downstream ratio logic.

Helper `_resolve_phase_batch` is no longer needed in its old form; if kept, it must reflect class-gated `resolve_batch` (sizing only) — the RUN now goes through `run_with_oom_backoff`, which is the structural change.

- [ ] **Step 1: Failing tests** (`tests/test_ma_detection_auto_batch.py`): (a) `monkeypatch` `M.resolve_batch` to a spy; with no `auto_batch` config the spy is NOT called and `M.run_with_oom_backoff` is NOT called (default-off path is the original single run); (b) with `auto_batch.enabled=true`, `resolve_batch` IS called with `fixed_batch=_PHASE_A_BATCH_SIZE` + `FidelityClass.BATCH_INVARIANT`, and the pass runs through `run_with_oom_backoff`. Use fakes; no GPU.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** the control flow above.

- [ ] **Step 4: Unit** `python3 -m pytest tests/test_stage1_plugin_ma_detection.py tests/test_ma_detection_auto_batch.py -q` → green.

- [ ] **Step 5: GOLDEN GUARDRAIL** `python3 -m pytest tests/test_stage1_golden_snapshot.py -q` → MUST pass UNCHANGED, no `MOE_REGEN_GOLDEN`. If it changes → STOP, report (the default-off path is not inert). 

- [ ] **Step 6: Commit** `feat(auto-batch): wire ma_detection phase-a via size+run-backoff (default-off)`.

## Task 7: Config docs + plugin/stage __init__ note

**Files:**
- Modify: `src/moe_compress/stage1/__init__.py` (the batch-size doc block — `grep -n "ablation_filter_batch_size\|batch_size" src/moe_compress/stage1/__init__.py` to find it; do NOT trust a fixed line number, it drifts)
- Modify: `src/moe_compress/utils/auto_batch.py` (module docstring: config keys)

- [ ] **Step 1:** Add a short doc paragraph: the optional `auto_batch: {enabled, headroom_frac, max_cap, probe_samples}` block (default `enabled:false`; NO rtol/atol — there is no allclose self-test), that it applies ONLY to `batch_invariant` plugins (phase-a today), and that it is a no-op when off. No test (docs only); run `python3 -m pytest tests/test_auto_batch.py tests/test_ma_detection_auto_batch.py -q` to confirm still green.
- [ ] **Step 2: Commit** `docs(auto-batch): document the auto_batch config block`.

---

## Task 8: (GPU, manual) Real-probe smoke on the local RTX5080

**Files:** none (manual validation; document result in the PR/commit message of Task 6 follow-up)

Rationale: unit tests use fakes; this confirms the real `CudaMemProbe` cost probe + `run_with_oom_backoff` path actually sizes the batch and runs the phase-a pass (adopting a batch ≥ floor, or halving cleanly on a forced OOM) on hardware, with the blacklist identical to the default-off run. NOT a CI test (needs GPU).

- [ ] **Step 1:** On a GPU host, set `auto_batch.enabled=true` for a tiny-model `ma_detection` run; confirm the log shows ADOPT with a batch ≥ floor and the blacklist output matches the default-off run within the plugin's tolerance (BATCH_INVARIANT ⇒ identical selection). Record the adopted batch + drift in the commit message. If it backs off unexpectedly, capture the log and STOP for review.

---

## Out of scope (explicit — do NOT implement here)

- Any change to `covariance_collection.py` / the Gram reduction / `_resolve_cov_window` / `_resolve_cov_batch_size` (cov is `reduction-accumulating` → v2).
- **`ma_detection` Phase B / `phase_b_batch_size` (`stage1/orchestrator.py:541`)** — its `run_calibration_pass` builds covariance/reservoir/downproj-max accumulators (`reduction-accumulating`) → v2 (alongside cov). Only phase-a is wired in v1.
- `ablation_filter` auto-batch (NLL is accumulating → v2; threshold-margin risk).
- The data-parallel `min(candidate)` agreement / pinned-grouping reduction (v2 + DP plan).
- Re-blessing ANY golden.

## After this plan

This plan goes through our standard **plan/review loop** (reviewer→fixer ping-pong, all 5 categories — Critical/High/Medium/Low/Nitpick — closing only on all-none) BEFORE execution, then the **implementation/review loop** with the same rules during execution.
