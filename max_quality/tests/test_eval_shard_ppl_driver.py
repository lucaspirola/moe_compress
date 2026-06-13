"""Task 5 — PPL replica worker + driver (run_dp_ppl) with per-replica auto-batch.

WikiText-PPL is BATCH_INVARIANT, so a row shard is exact regardless of boundary,
and each replica MAY size its own forward batch. Tests the spawn + even-split +
exact partial-sum merge plumbing with a deterministic per-row loss stub, and
asserts that a (mocked) larger per-replica batch yields the SAME merged PPL.
"""
from __future__ import annotations

import math

import pytest
import torch

from moe_compress.tools import eval_shard
from moe_compress.tools.eval_shard import run_dp_ppl, _ppl_forward_all


def _single_pass_ppl(chunks):
    # Mirror the deterministic stub the worker uses for one whole batch.
    numel = chunks.numel()
    n_rows = chunks.shape[0]
    per_tok = (chunks.to(torch.float64) % 7).sum() / max(numel - n_rows, 1)
    nll = float(per_tok) * (numel - n_rows)
    return math.exp(nll / (numel - n_rows))


def test_run_dp_ppl_exact_equals_single_pass(tmp_path):
    torch.manual_seed(5)
    chunks = torch.randint(0, 50, (10, 8), dtype=torch.long)
    single = _single_pass_ppl(chunks)
    merged = run_dp_ppl(
        chunks,
        tmp_dir=str(tmp_path / "src"),
        replicas=2,
        gpus_per_replica=1,
        ppl_bs=1,
        experts_impl_generative="batched_mm",
        cfg=None,
        out_dir=str(tmp_path / "out"),
        _stub_ppl=eval_shard._STUB_PPL_MOD7,
    )
    assert math.isclose(merged, single, rel_tol=0.0, abs_tol=0.0)


def test_run_dp_ppl_single_replica(tmp_path):
    torch.manual_seed(6)
    chunks = torch.randint(0, 50, (7, 8), dtype=torch.long)
    single = _single_pass_ppl(chunks)
    merged = run_dp_ppl(
        chunks,
        tmp_dir=str(tmp_path / "src"),
        replicas=1,
        gpus_per_replica=1,
        ppl_bs=1,
        experts_impl_generative="batched_mm",
        cfg=None,
        out_dir=str(tmp_path / "out"),
        _stub_ppl=eval_shard._STUB_PPL_MOD7,
    )
    assert math.isclose(merged, single, rel_tol=0.0, abs_tol=0.0)


def test_run_dp_ppl_batch_invariant_across_replica_batch(tmp_path):
    # The deterministic stub accumulates per-row regardless of the forward
    # batch, so a larger ppl_bs (the per-replica auto-batch outcome) yields the
    # SAME merged PPL — the batch-invariance property the gen path lacks.
    torch.manual_seed(9)
    chunks = torch.randint(0, 50, (12, 8), dtype=torch.long)
    base = run_dp_ppl(
        chunks, tmp_dir=str(tmp_path / "s1"), replicas=2, gpus_per_replica=1,
        ppl_bs=1, experts_impl_generative="batched_mm", cfg=None,
        out_dir=str(tmp_path / "o1"), _stub_ppl=eval_shard._STUB_PPL_MOD7,
    )
    bigger = run_dp_ppl(
        chunks, tmp_dir=str(tmp_path / "s2"), replicas=2, gpus_per_replica=1,
        ppl_bs=4, experts_impl_generative="batched_mm", cfg=None,
        out_dir=str(tmp_path / "o2"), _stub_ppl=eval_shard._STUB_PPL_MOD7,
    )
    assert math.isclose(base, bigger, rel_tol=0.0, abs_tol=0.0)


# ---------------------------------------------------------------------------
# M1 — _ppl_forward_all skip-accounting (production skip-semantics parity).
#
# This is the module-level helper the worker's forward loop delegates to. We
# drive it with a tiny stub model (NO real model / NO spawned process) whose
# forward returns, across 3 one-row batches: None loss, then a non-finite
# (nan) loss, then a finite loss. Production (wikitext_ppl.py:264-288) skips
# the first two and counts only the finite batch — so skipped==2 and only the
# finite batch contributes to nsum/tcount.
# ---------------------------------------------------------------------------


class _Out:
    def __init__(self, loss):
        self.loss = loss


class _ScriptedModel:
    """Stub model: returns a scripted sequence of `.loss` values, one per call.

    A `None` -> production skips; a non-finite float -> production skips; a
    finite float -> contributes. `loss` is exposed as a 0-dim tensor so the
    helper's `out.loss.item()` works (it only `.item()`s after the None check).
    """
    def __init__(self, losses):
        self._losses = list(losses)
        self._i = 0

    def __call__(self, *, input_ids, labels):
        v = self._losses[self._i]
        self._i += 1
        loss = None if v is None else torch.tensor(float(v))
        return _Out(loss)


def _one_row_batches(shard, batch_size):
    # Stub iter_batches: yield each row as its own [1, seq] batch so the helper
    # makes exactly one model call per row (drives the scripted loss sequence).
    for r in range(shard.shape[0]):
        yield shard[r : r + 1]


def test_ppl_forward_all_skips_none_and_nonfinite_counts_only_finite():
    shard = torch.arange(3 * 8, dtype=torch.long).view(3, 8)  # 3 one-row batches
    finite_loss = 1.25
    model = _ScriptedModel([None, float("nan"), finite_loss])
    nsum, tcount, skipped = _ppl_forward_all(
        model, shard, device=torch.device("cpu"), bs=1,
        iter_batches=_one_row_batches,
    )
    # Only the 3rd (finite) batch contributes: one row of seq_len=8 -> 7 tokens.
    assert skipped == 2
    expected_tok = 8 - 1
    assert tcount == expected_tok
    assert math.isclose(nsum, finite_loss * expected_tok, rel_tol=0.0, abs_tol=0.0)


def test_ppl_forward_all_skips_raising_batch():
    # A forward that raises a generic RuntimeError is skip-counted (not fatal),
    # mirroring wikitext_ppl.py:285-288.
    shard = torch.arange(2 * 8, dtype=torch.long).view(2, 8)

    class _Raiser:
        def __init__(self):
            self._i = 0

        def __call__(self, *, input_ids, labels):
            self._i += 1
            if self._i == 1:
                raise RuntimeError("boom")
            return _Out(torch.tensor(2.0))

    nsum, tcount, skipped = _ppl_forward_all(
        _Raiser(), shard, device=torch.device("cpu"), bs=1,
        iter_batches=_one_row_batches,
    )
    assert skipped == 1
    assert tcount == 7
    assert math.isclose(nsum, 2.0 * 7, rel_tol=0.0, abs_tol=0.0)


def test_ppl_forward_all_skips_batch_whose_loss_item_raises():
    # H1: the deferred-CUDA-assert case — model() returns a fine output object,
    # but out.loss.item() RAISES on the sync. Production wraps .item() inside the
    # same try (wikitext_ppl.py:265-288), so the batch must be skip-counted, NOT
    # propagated (a narrower try would crash the replica).
    shard = torch.arange(2 * 8, dtype=torch.long).view(2, 8)

    class _RaisingItem:
        def item(self):
            raise RuntimeError("deferred device-side assert surfaced on .item()")

    class _ItemRaiserModel:
        def __init__(self):
            self._i = 0

        def __call__(self, *, input_ids, labels):
            self._i += 1
            if self._i == 1:
                return _Out(_RaisingItem())   # not None, but .item() throws
            return _Out(torch.tensor(2.0))

    nsum, tcount, skipped = _ppl_forward_all(
        _ItemRaiserModel(), shard, device=torch.device("cpu"), bs=1,
        iter_batches=_one_row_batches,
    )
    assert skipped == 1
    assert tcount == 7
    assert math.isclose(nsum, 2.0 * 7, rel_tol=0.0, abs_tol=0.0)


def test_ppl_forward_all_all_finite_zero_skipped():
    shard = torch.arange(3 * 8, dtype=torch.long).view(3, 8)
    model = _ScriptedModel([1.0, 2.0, 3.0])
    nsum, tcount, skipped = _ppl_forward_all(
        model, shard, device=torch.device("cpu"), bs=1,
        iter_batches=_one_row_batches,
    )
    assert skipped == 0
    assert tcount == 3 * 7
    assert math.isclose(nsum, (1.0 + 2.0 + 3.0) * 7, rel_tol=0.0, abs_tol=0.0)
