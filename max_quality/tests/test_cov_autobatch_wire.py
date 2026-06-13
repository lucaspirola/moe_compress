import types

import pytest
import torch

from moe_compress.utils.activation_hooks import InputCovarianceAccumulator


# ---------------------------------------------------------------------------
# Task 1 — discard_layer (forward-free reset)
# ---------------------------------------------------------------------------
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


# ===========================================================================
# Task 2 — size_batch + run_with_oom_backoff wiring into the cov loop
# ===========================================================================
#
# GPU-free harness. The real ``_collect_covariances`` closures are exercised
# by monkeypatching ``capture_experts`` (in the cov-module namespace) with a
# fake context manager that registers the input/intermediate callbacks into a
# shared registry; a fake model's ``__call__`` then fires those callbacks with
# synthetic activations + a ``token_idx``, exactly as the native forward would.
# This drives the REAL ``input_cb`` (incl. the ``_seq_len`` free-var pin) under
# the auto wrapper, so a closure-scoping regression (missing ``nonlocal``) is
# observable as ``_seq_len == 0`` in the callback.

import moe_compress.stage3.plugins.covariance_collection as cc


D_IN = 4


class _FakeRef:
    """Minimal MoELayerRef stand-in (only ``layer_idx`` is read by the loop)."""
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.experts_module = object()


class _Registry:
    """Holds the callbacks registered via the fake ``capture_experts`` CM."""
    def __init__(self):
        # layer_idx -> {"input": cb, "intermediate": cb}
        self.student = {}


def _make_fake_capture(registry):
    import contextlib

    @contextlib.contextmanager
    def fake_capture_experts(layer_ref, callbacks, *, capture_intermediate=True):
        registry.student[layer_ref.layer_idx] = callbacks
        try:
            yield
        finally:
            registry.student.pop(layer_ref.layer_idx, None)
    return fake_capture_experts


def _make_fake_model(registry, *, n_experts=2, seq_len, oom_on=None, counter=None):
    """A callable that, per forward, fires each hooked layer's input callback
    once per expert with a deterministic activation derived from the batch.

    ``oom_on`` (optional dict {"remaining": k}) raises ``OutOfMemoryError`` on
    the k-th batch forward (decrementing) to simulate a mid-window OOM.
    """
    def model(input_ids=None):
        rows = int(input_ids.shape[0])
        sl = int(input_ids.shape[1])
        n_tok = rows * sl
        if oom_on is not None:
            oom_on["remaining"] -= 1
            if oom_on["remaining"] == 0:
                raise torch.cuda.OutOfMemoryError("fake mid-window OOM")
        if counter is not None:
            counter["forwards"] += 1
        for li, cbs in list(registry.student.items()):
            for e in range(n_experts):
                # token_idx: this expert "routes" all tokens (deterministic);
                # activation = a fixed function of (input_ids, expert) so the
                # finalized Gram is reproducible across batchings.
                token_idx = torch.arange(n_tok, dtype=torch.long)
                # deterministic per-(token,expert) activation, independent of bs
                base = (input_ids.reshape(-1, 1).to(torch.float32) + 1.0)
                act = (base.expand(n_tok, D_IN) + float(e)) * 0.1
                ctx = {"token_idx": token_idx, "_seq_len_seen": []}
                cbs["input"](li, e, act, ctx)
        return None
    return model


def _drive_collect(monkeypatch, *, calib, batches, cov_auto, n_layers=1,
                   probe_peaks=None, oom_on=None, mem_total=10**12):
    """Run the real ``_collect_covariances`` with the fake harness. Returns the
    finalized B_acc covariance dict and a captured-_seq_len list."""
    registry = _Registry()
    monkeypatch.setattr(cc, "capture_experts", _make_fake_capture(registry))

    seq_len = int(calib.shape[1])
    seen_seq_len = []

    counter = {"forwards": 0}
    base_model = _make_fake_model(
        registry, seq_len=seq_len, oom_on=oom_on, counter=counter,
    )

    # Fake CudaMemProbe: a two-point probe with a fixed per-sample slope so
    # size_batch predicts a bs > 1 (mem_total huge), clamped by _COV_MAX_CAP.
    class _FakeMem:
        def __init__(self, device=None):
            pass
        def total(self):
            return mem_total
        def allocated(self):
            return 0
        def reset_peak(self):
            pass
        def peak(self):
            return 0
    monkeypatch.setattr(cc, "CudaMemProbe", _FakeMem)

    # torch.cuda probe shims used by cost_probe_fn (device=None path).
    peaks = list(probe_peaks or [1000, 2000])
    pk = {"i": 0}
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: None)

    def _max_alloc(d=None):
        i = min(pk["i"], len(peaks) - 1)
        val = peaks[i]
        pk["i"] += 1
        return val
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _max_alloc)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    B_acc = InputCovarianceAccumulator(); B_acc.set_storage_dtype(torch.float32)
    moe_layers = [_FakeRef(i) for i in range(n_layers)]

    cc._collect_covariances(
        base_model, moe_layers, batches, B_acc, device=None,
        teacher_model=None, teacher_moe_layers=None, C_acc=None,
        cov_window_size=n_layers,
        calib=calib, cov_auto=cov_auto,
        auto_batch_cfg=cc.AutoBatchConfig(enabled=True, headroom_frac=0.0),
    )
    B_acc.finalize_all()
    return B_acc.covariance, seen_seq_len, counter


def _make_calib(n_seq, seq_len, vocab=7):
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, vocab, (n_seq, seq_len), generator=g)


# --- (a) _seq_len reaches the callback under auto (pin fires; NOT 0) --------
def test_seq_len_reaches_callback_under_auto(monkeypatch):
    """Under the auto wrapper, the cov pin must fire: the per-sequence split is
    active, which it can only be if ``_seq_len`` (a free var of the OUTER
    ``_collect_covariances``) is updated by ``run_window_forwards`` via
    ``nonlocal``. We prove the pin fired by comparing the multi-sequence auto
    Gram against a bs=1 reference: equal ⇒ the per-sequence accumulation
    happened (the pin was live). A regression that drops ``nonlocal`` leaves
    ``_seq_len == 0`` → un-pinned ``update`` → a DIFFERENT accumulation order.
    """
    calib = _make_calib(4, 3)            # 4 sequences, seq_len 3
    batches_bs1 = cc.iter_batches(calib, batch_size=1)

    # bs=1 reference (non-auto, original loop).
    cov_ref, _, _ = _drive_collect(
        monkeypatch, calib=calib, batches=batches_bs1, cov_auto=False,
    )
    # auto path (probe sizes a larger bs; pin must re-pin per-sequence).
    cov_auto, _, _ = _drive_collect(
        monkeypatch, calib=calib, batches=batches_bs1, cov_auto=True,
        probe_peaks=[1000, 1100],        # gentle slope → bs raised > 1
    )
    for k in cov_ref:
        assert torch.equal(cov_ref[k], cov_auto[k]), (
            f"key {k}: auto Gram diverged from bs=1 — the per-sequence pin did "
            f"not fire under the wrapper (closure-scoping regression?)"
        )


def test_seq_len_nonzero_via_direct_probe(monkeypatch):
    """Direct proof the callback observes the real per-sequence length (NOT 0)
    under auto. We spy on ``B_acc.update_grouped`` (the pinned path) and assert
    the ``sids`` it receives span >1 sequence — only possible if ``_seq_len``
    (free var of the OUTER ``_collect_covariances``) was the real seq_len at
    callback time, which requires ``run_window_forwards`` to declare
    ``nonlocal _seq_len``. A missing nonlocal leaves ``_seq_len == 0`` → the
    callback takes the un-pinned ``update`` branch → ``update_grouped`` is never
    called → this test fails."""
    calib = _make_calib(3, 5)            # 3 sequences, seq_len 5

    reg = _Registry()
    import contextlib

    @contextlib.contextmanager
    def fake_capture(layer_ref, callbacks, *, capture_intermediate=True):
        reg.student[layer_ref.layer_idx] = callbacks
        try:
            yield
        finally:
            reg.student.pop(layer_ref.layer_idx, None)
    monkeypatch.setattr(cc, "capture_experts", fake_capture)

    def model(input_ids=None):
        rows = int(input_ids.shape[0]); sl = int(input_ids.shape[1])
        n_tok = rows * sl
        for li, cbs in list(reg.student.items()):
            token_idx = torch.arange(n_tok, dtype=torch.long)
            act = torch.ones(n_tok, D_IN)
            cbs["input"](li, 0, act, {"token_idx": token_idx})
        return None

    class _FakeMem:
        def __init__(self, d=None): pass
        def total(self): return 10**12
        def allocated(self): return 0
        def reset_peak(self): pass
        def peak(self): return 0
    monkeypatch.setattr(cc, "CudaMemProbe", _FakeMem)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: None)
    pk = {"i": 0}; peaks = [1000, 1100]

    def _max_alloc(d=None):
        v = peaks[min(pk["i"], len(peaks) - 1)]; pk["i"] += 1; return v
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _max_alloc)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    # Spy on update_grouped to capture the sids (per-sequence split).
    B_acc = InputCovarianceAccumulator(); B_acc.set_storage_dtype(torch.float32)
    real_ug = B_acc.update_grouped
    seen_sids = []

    def spy_ug(li, e, m, t, sids):
        seen_sids.append(sids.clone())
        return real_ug(li, e, m, t, sids)
    B_acc.update_grouped = spy_ug

    cc._collect_covariances(
        model, [_FakeRef(0)], cc.iter_batches(calib, batch_size=2), B_acc,
        device=None, calib=calib, cov_auto=True,
        auto_batch_cfg=cc.AutoBatchConfig(enabled=True, headroom_frac=0.0),
    )
    assert seen_sids, "update_grouped never called — pin did not fire (_seq_len==0?)"
    # At least one real (non-probe) forward batched ≥2 sequences ⇒ sids span >1.
    assert any(int(s.max()) > 0 for s in seen_sids), (
        "all sids collapsed to 0 — _seq_len was 0 in the callback (missing nonlocal)"
    )


# --- (b) mid-batch OOM idempotency: retry == clean pass --------------------
def test_mid_window_oom_idempotent(monkeypatch):
    """The start-bs run OOMs partway through the window (AFTER an earlier batch
    accumulated into ``_pending``); ``run_with_oom_backoff`` halves and
    ``run_window_forwards`` ``_discard_window``s the aborted attempt before
    re-running. The finalized Gram must be byte-identical to a single clean pass
    at the smaller bs — proving no double-count of the partial accumulation.

    ``_COV_MAX_CAP`` is pinned to 2 so the start bs is 2 → the 4-sequence calib
    yields 2 window batches; the OOM fires on the 2nd forward (batch 1 already
    accumulated), forcing a backoff to bs=1 that must discard the partial."""
    monkeypatch.setattr(cc, "_COV_MAX_CAP", 2)
    calib = _make_calib(4, 2)

    # Clean reference at bs=1 (the bs the backoff lands on after halving from 2).
    cov_clean, _, _ = _drive_collect(
        monkeypatch, calib=calib, batches=cc.iter_batches(calib, batch_size=1),
        cov_auto=False,
    )

    # Auto: start bs=2 (cap), 2 window batches; OOM on the 2nd forward of the
    # window (after batch 1 accumulated) → retry restarts cleanly at bs=1.
    # Probe runs cost_probe_fn(1)+cost_probe_fn(2) first = 2 forwards; the window
    # batches are forwards 3 (batch 0) and 4 (batch 1) → OOM the 4th forward.
    cov_oom, _, _ = _drive_collect(
        monkeypatch, calib=calib, batches=cc.iter_batches(calib, batch_size=1),
        cov_auto=True, probe_peaks=[1000, 1500],   # slope → bs raised to cap=2
        oom_on={"remaining": 4},                    # OOM on 2nd window forward
    )
    for k in cov_clean:
        assert torch.equal(cov_clean[k], cov_oom[k]), (
            f"key {k}: OOM-retry Gram != clean pass — _discard_window did not "
            f"reset the aborted attempt (double-count)."
        )


# --- (c) default-off: size_batch / run_with_oom_backoff NOT called ----------
def test_default_off_no_resolver_calls(monkeypatch):
    calib = _make_calib(3, 2)
    calls = {"size_batch": 0, "backoff": 0}
    real_sb = cc.size_batch
    real_bo = cc.run_with_oom_backoff
    monkeypatch.setattr(cc, "size_batch",
                        lambda *a, **k: (calls.__setitem__("size_batch", calls["size_batch"] + 1), real_sb(*a, **k))[1])
    monkeypatch.setattr(cc, "run_with_oom_backoff",
                        lambda *a, **k: (calls.__setitem__("backoff", calls["backoff"] + 1), real_bo(*a, **k))[1])

    _drive_collect(
        monkeypatch, calib=calib, batches=cc.iter_batches(calib, batch_size=1),
        cov_auto=False,
    )
    assert calls == {"size_batch": 0, "backoff": 0}


# --- (d) no resolve_batch / FidelityClass import (honest fidelity) ----------
def test_no_fidelity_mislabel_import():
    assert not hasattr(cc, "resolve_batch"), "cov must NOT import resolve_batch"
    assert not hasattr(cc, "FidelityClass"), "cov must NOT import FidelityClass"
    src = open(cc.__file__).read()
    assert "resolve_batch" not in src
    assert "FidelityClass" not in src


# --- (e) probe non-contamination: probe Gram never reaches the final Gram ---
def test_probe_non_contamination(monkeypatch):
    """The cost-probe's two dual-forwards accumulate into ``_pending``; the
    pre+post ``_discard_window()`` must remove them so the finalized Gram is
    byte-identical to a run that never probed (bs=1 reference)."""
    calib = _make_calib(4, 2)
    cov_ref, _, _ = _drive_collect(
        monkeypatch, calib=calib, batches=cc.iter_batches(calib, batch_size=1),
        cov_auto=False,
    )
    cov_auto, _, _ = _drive_collect(
        monkeypatch, calib=calib, batches=cc.iter_batches(calib, batch_size=1),
        cov_auto=True, probe_peaks=[1000, 1100],
    )
    for k in cov_ref:
        assert torch.equal(cov_ref[k], cov_auto[k]), (
            f"key {k}: probe Gram contaminated the final Gram — pre/post "
            f"_discard_window did not fully clear the probe's tokens."
        )


# --- cross-cov: _teacher_T nonlocal correctness under auto ------------------
def test_cross_cov_teacher_T_threaded_under_auto(monkeypatch):
    """The dense cross-cov path sizes ``_teacher_dense`` as ``[_teacher_T, d_in]``
    and scatters rows via ``index_copy_(0, token_idx, ...)``. ``_teacher_T`` is a
    free var of the OUTER ``_collect_covariances``; ``run_window_forwards`` must
    set it via ``nonlocal _teacher_T`` to the REAL batch's row count. If it
    doesn't, ``_teacher_T`` stays at the probe's last value (2*seq) while the
    real batch's ``token_idx`` ranges over ``cov_bs*seq`` rows → ``index_copy_``
    overflows / the gather mismatches → the auto Gram diverges from (or raises
    against) the clean bs=1 cross-cov pass. This test pins that wiring."""
    calib = _make_calib(4, 3)            # 4 sequences, seq_len 3

    student_reg = _Registry()
    teacher_reg = _Registry()
    import contextlib

    @contextlib.contextmanager
    def fake_capture(layer_ref, callbacks, *, capture_intermediate=True):
        # The teacher-only hook registers {"input": _teacher_input_cb} (no
        # "intermediate"); the student registers input+intermediate. Route by
        # presence of the "intermediate" key.
        target = student_reg if "intermediate" in callbacks else teacher_reg
        target.student[layer_ref.layer_idx] = callbacks
        try:
            yield
        finally:
            target.student.pop(layer_ref.layer_idx, None)
    monkeypatch.setattr(cc, "capture_experts", fake_capture)

    def _make_dual_model(s_reg, t_reg, *, is_teacher):
        reg = t_reg if is_teacher else s_reg
        def model(input_ids=None):
            rows = int(input_ids.shape[0]); sl = int(input_ids.shape[1])
            n_tok = rows * sl
            for li, cbs in list(reg.student.items()):
                token_idx = torch.arange(n_tok, dtype=torch.long)
                base = (input_ids.reshape(-1, 1).to(torch.float32) + 1.0)
                if is_teacher:
                    act = base.expand(n_tok, D_IN) * 0.07
                    cbs["input"](li, 0, act, {"token_idx": token_idx})
                else:
                    act = base.expand(n_tok, D_IN) * 0.1
                    cbs["input"](li, 0, act, {"token_idx": token_idx})
            return None
        return model

    class _FakeMem:
        def __init__(self, d=None): pass
        def total(self): return 10**12
        def allocated(self): return 0
        def reset_peak(self): pass
        def peak(self): return 0
    monkeypatch.setattr(cc, "CudaMemProbe", _FakeMem)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: None)
    pk = {"i": 0}; peaks = [1000, 1100]

    def _max_alloc(d=None):
        v = peaks[min(pk["i"], len(peaks) - 1)]; pk["i"] += 1; return v
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _max_alloc)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def _run(cov_auto):
        student_reg.student.clear(); teacher_reg.student.clear()
        pk["i"] = 0
        student = _make_dual_model(student_reg, teacher_reg, is_teacher=False)
        teacher = _make_dual_model(student_reg, teacher_reg, is_teacher=True)
        B = InputCovarianceAccumulator(); B.set_storage_dtype(torch.float32)
        C = InputCovarianceAccumulator(); C.set_storage_dtype(torch.float32)
        cc._collect_covariances(
            student, [_FakeRef(0)], cc.iter_batches(calib, batch_size=1), B,
            device=None, teacher_model=teacher, teacher_moe_layers=[_FakeRef(0)],
            C_acc=C, cov_window_size=1, calib=calib, cov_auto=cov_auto,
            auto_batch_cfg=cc.AutoBatchConfig(enabled=True, headroom_frac=0.0),
        )
        B.finalize_all(); C.finalize_all()
        return dict(B.covariance), dict(C.covariance)

    b_ref, c_ref = _run(False)
    b_auto, c_auto = _run(True)
    assert c_ref, "cross-cov produced no C keys (harness wiring bug)"
    for k in c_ref:
        assert torch.equal(c_ref[k], c_auto[k]), (
            f"cross-cov key {k}: auto C != bs=1 C — _teacher_T not threaded "
            f"(missing nonlocal) corrupted the dense teacher gather."
        )
    for k in b_ref:
        assert torch.equal(b_ref[k], b_auto[k])


# --- gate logic unit checks -------------------------------------------------
def test_cov_is_auto_double_gate():
    assert cc._cov_is_auto({"cov_batch_size": "auto",
                            "auto_batch": {"enabled": True}}) is True
    # "auto" but auto_batch disabled → False
    assert cc._cov_is_auto({"cov_batch_size": "auto",
                            "auto_batch": {"enabled": False}}) is False
    assert cc._cov_is_auto({"cov_batch_size": "auto"}) is False   # no auto_batch
    # enabled but not "auto" → False
    assert cc._cov_is_auto({"cov_batch_size": 4,
                            "auto_batch": {"enabled": True}}) is False
    assert cc._cov_is_auto({}) is False                            # default
