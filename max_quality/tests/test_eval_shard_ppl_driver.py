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
from moe_compress.tools.eval_shard import run_dp_ppl


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
