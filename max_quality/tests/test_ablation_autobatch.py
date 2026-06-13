"""Auto-batch v2 tests for ``stage1/plugins/ablation_filter`` (CPU, GPU-free).

Covers:

T1  ``_measure_corpus_nll_pinned`` per-sequence grouping-independence: B
    sequences batched together produce the EXACT same mean NLL as the same B
    sequences fed one-per-batch (the per-seq CE reduction is pinned), and the
    pinned token-mean matches the fused ``_measure_corpus_nll`` within a
    generous allclose tolerance (same shift/all-token/fp32 semantics).

T2a default-off -> fused ``_measure_corpus_nll`` path; ``size_batch`` /
    ``run_with_oom_backoff`` are NOT called (spies).

T2b auto -> ``size_batch`` called exactly once, every NLL (baseline + each
    candidate) measured through ``run_with_oom_backoff``.

T2c dNLL batch-invariance under the pin: baseline-vs-ablated computed at two
    different forced batch sizes yields an identical dNLL.

T2d the plugin imports ``size_batch``/``run_with_oom_backoff``/etc DIRECTLY -
    no ``resolve_batch`` / ``FidelityClass`` (honest fidelity, cov-class).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import moe_compress.stage1.plugins.ablation_filter as af_mod
from moe_compress.stage1.plugins.ablation_filter import (
    _measure_corpus_nll,
    _measure_corpus_nll_pinned,
    run_ablation_filter,
)


VOCAB = 17


class _Out:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class _TinyLM:
    """Deterministic causal LM: a fixed cos-table indexed by token id."""

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self._mode = False
        ids = torch.arange(vocab, dtype=torch.float64).reshape(1, vocab)
        self._table = torch.cos(
            (torch.arange(vocab, dtype=torch.float64).reshape(vocab, 1) + 1.0)
            * (ids + 1.0)
            * 0.1
        )

    def eval(self):
        self._mode = True
        return self

    def _logits(self, input_ids):
        rows = self._table[input_ids.long()]
        return rows.to(torch.float32)

    def __call__(self, input_ids=None, labels=None, **kw):
        logits = self._logits(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.vocab).float()
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
        return _Out(logits, loss)


def _make_batch(bs: int, seq: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (bs, seq), generator=g)


def test_pinned_nll_grouping_independent():
    model = _TinyLM()
    seq = 12
    full = _make_batch(8, seq, seed=1)

    grouped = _measure_corpus_nll_pinned(model, [full], device=None)
    per_seq = _measure_corpus_nll_pinned(
        model, [full[i : i + 1] for i in range(full.shape[0])], device=None
    )
    assert grouped == per_seq

    mixed = _measure_corpus_nll_pinned(model, [full[:3], full[3:]], device=None)
    assert grouped == mixed


def test_pinned_matches_fused_within_tol():
    model = _TinyLM()
    full = _make_batch(6, 10, seed=2)
    pinned = _measure_corpus_nll_pinned(model, [full], device=None)
    fused = _measure_corpus_nll(model, [full], device=None)
    assert torch.allclose(
        torch.tensor(pinned), torch.tensor(fused), atol=1e-5
    ), f"pinned={pinned} fused={fused}"


def _patch_calib(monkeypatch, calib: torch.Tensor):
    monkeypatch.setattr(af_mod, "spec_from_config", lambda *a, **k: object())
    monkeypatch.setattr(af_mod, "build_calibration_tensor", lambda *a, **k: calib)

    class _Ref:
        def __init__(self, li):
            self.layer_idx = li

    monkeypatch.setattr(af_mod, "iter_moe_layers", lambda model: [_Ref(5), _Ref(10)])
    import contextlib

    monkeypatch.setattr(
        af_mod, "_ablate_expert_context", lambda ref, e: contextlib.nullcontext()
    )


def _cfg(batch_size, auto_enabled):
    return {
        "calibration": {},
        "stage1_grape": {
            "ablation_filter": {
                "enabled": True,
                "holdout_samples": 8,
                "blacklist_threshold": 0.001,
                "batch_size": batch_size,
            },
            "auto_batch": {"enabled": auto_enabled},
        },
    }


def test_default_off_uses_fused_no_autobatch(monkeypatch, tmp_path):
    model = _TinyLM()
    calib = _make_batch(8, 10, seed=3)
    _patch_calib(monkeypatch, calib)

    calls = {"size_batch": 0, "backoff": 0, "fused": 0, "pinned": 0}
    real_fused = af_mod._measure_corpus_nll
    real_pinned = af_mod._measure_corpus_nll_pinned

    def spy_fused(*a, **k):
        calls["fused"] += 1
        return real_fused(*a, **k)

    def spy_pinned(*a, **k):
        calls["pinned"] += 1
        return real_pinned(*a, **k)

    monkeypatch.setattr(af_mod, "_measure_corpus_nll", spy_fused)
    monkeypatch.setattr(af_mod, "_measure_corpus_nll_pinned", spy_pinned)
    monkeypatch.setattr(
        af_mod,
        "size_batch",
        lambda *a, **k: calls.__setitem__("size_batch", calls["size_batch"] + 1),
    )
    monkeypatch.setattr(
        af_mod,
        "run_with_oom_backoff",
        lambda *a, **k: calls.__setitem__("backoff", calls["backoff"] + 1),
    )

    candidates = {(5, 0): ["aimer"], (10, 1): ["sink"]}
    run_ablation_filter(
        model, None, _cfg(8, auto_enabled=False), tmp_path,
        candidates=candidates, device=None,
    )

    assert calls["size_batch"] == 0
    assert calls["backoff"] == 0
    assert calls["fused"] >= 1
    assert calls["pinned"] == 0


def test_auto_uses_size_batch_and_backoff(monkeypatch, tmp_path):
    model = _TinyLM()
    calib = _make_batch(8, 10, seed=4)
    _patch_calib(monkeypatch, calib)

    calls = {"size_batch": 0, "backoff": 0}

    def fake_size_batch(cost_probe_fn, fixed_batch, **k):
        calls["size_batch"] += 1
        return fixed_batch

    real_backoff = af_mod.run_with_oom_backoff

    def spy_backoff(run_fn, start_batch, floor):
        calls["backoff"] += 1
        return real_backoff(run_fn, start_batch=start_batch, floor=floor)

    monkeypatch.setattr(af_mod, "size_batch", fake_size_batch)
    monkeypatch.setattr(af_mod, "run_with_oom_backoff", spy_backoff)

    candidates = {(5, 0): ["aimer"], (10, 1): ["sink"]}
    run_ablation_filter(
        model, None, _cfg("auto", auto_enabled=True), tmp_path,
        candidates=candidates, device=None,
    )

    assert calls["size_batch"] == 1
    assert calls["backoff"] == 1 + len(candidates)


def test_dnll_batch_invariant_under_pin(monkeypatch, tmp_path):
    model = _TinyLM()
    calib = _make_batch(8, 10, seed=5)
    _patch_calib(monkeypatch, calib)

    class _Ablate:
        def __init__(self):
            self._orig = None

        def __enter__(self):
            self._orig = model._logits

            def biased(input_ids):
                return self._orig(input_ids) + 0.5

            model._logits = biased
            return self

        def __exit__(self, *exc):
            model._logits = self._orig
            return False

    monkeypatch.setattr(af_mod, "_ablate_expert_context", lambda ref, e: _Ablate())

    def run_at(forced_bs):
        monkeypatch.setattr(af_mod, "size_batch", lambda *a, **k: forced_bs)
        candidates = {(5, 0): ["aimer"]}
        bl, deltas, baseline = run_ablation_filter(
            model, None, _cfg("auto", auto_enabled=True), tmp_path,
            candidates=candidates, device=None,
        )
        return deltas[(5, 0)]

    d1 = run_at(8)
    d2 = run_at(3)
    assert d1 == d2, f"dNLL not batch-invariant: bs8={d1} bs3={d2}"


def test_no_resolve_batch_or_fidelityclass_import():
    src = open(af_mod.__file__).read()
    assert "resolve_batch" not in src
    assert "FidelityClass" not in src
    assert "size_batch" in src
    assert "run_with_oom_backoff" in src
    assert "AutoBatchConfig" in src
    assert "CudaMemProbe" in src
