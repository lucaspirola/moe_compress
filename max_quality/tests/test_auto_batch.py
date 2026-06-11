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
    assert AutoBatchConfig.from_dict({"enabled": "false"}).enabled is False
    assert AutoBatchConfig.from_dict({"enabled": "true"}).enabled is True


# --- Task 2: pure sizing math ---
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

def test_size_candidate_rejects_floor_above_cap():
    with pytest.raises(ValueError):
        size_candidate(total=100, allocated_baseline=0, headroom_frac=0.0,
                       fixed=0, per_sample=1, fixed_batch=8192, max_cap=4096)


# --- Task 3: MemProbe abstraction ---
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


# --- Task 4: size_batch (cost-model prediction) + class-gated resolve_batch ---
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


# --- Task 5: run_with_oom_backoff — the real run IS the fit test ---
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
