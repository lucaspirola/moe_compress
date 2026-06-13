"""Task 3 — PPL partial-NLL exact merge.

WikiText-PPL is BATCH_INVARIANT: ``out.loss`` is a batch-mean NLL, rescaled by
``(numel - n_rows)`` to recover the exact NLL **sum** (wikitext_ppl.py:260), and
the final PPL is ``exp(nll_sum / tok_count)`` (wikitext_ppl.py:293). A DP shard
of the chunk rows is therefore exact regardless of boundary — the two partial
``(nll_sum, tok_count)`` sums merge to the identical global PPL.

``_merge_ppl`` carries the SAME guards as wikitext_ppl.py:273-295:
``tok_count == 0 -> inf`` and ``OverflowError -> inf``.
"""
from __future__ import annotations

import math

import pytest
import torch

from moe_compress.tools.eval_shard import _merge_ppl


def _deterministic_partial(chunk_rows):
    """Mirror the wikitext_ppl per-batch accumulation EXACTLY for a stub.

    Returns ``(nll_sum, tok_count)`` over the given rows using a deterministic
    per-row 'loss' that is a pure function of the row content — so the merge
    of two row-disjoint shards is a *true equality* (not a tolerance) against
    the single-pass forward. Uses the same ``mean_loss * (numel - n_rows)``
    rescale the production code uses.
    """
    nll_sum = 0.0
    tok_count = 0
    # one batch = all rows (geometry is irrelevant for BATCH_INVARIANT)
    batch = chunk_rows
    numel = batch.numel()
    n_rows = batch.shape[0]
    # deterministic mean loss over the (numel - n_rows) predicted tokens
    per_tok = (batch.to(torch.float64) % 7).sum() / max(numel - n_rows, 1)
    mean_loss = float(per_tok)
    nll = mean_loss * (numel - n_rows)
    nll_sum += nll
    tok_count += numel - n_rows
    return nll_sum, tok_count


def test_merge_ppl_exact_equals_single_pass():
    torch.manual_seed(7)
    chunks = torch.randint(0, 50, (10, 8), dtype=torch.long)
    # single pass
    single_nll, single_tok = _deterministic_partial(chunks)
    single_ppl = math.exp(single_nll / single_tok)
    # two shards by rows (boundary anywhere — no group constraint)
    p0 = _deterministic_partial(chunks[:4])
    p1 = _deterministic_partial(chunks[4:])
    merged = _merge_ppl([p0, p1])
    assert math.isclose(merged, single_ppl, rel_tol=0.0, abs_tol=0.0)


def test_merge_ppl_sum_is_associative_three_shards():
    torch.manual_seed(11)
    chunks = torch.randint(0, 50, (9, 8), dtype=torch.long)
    single = math.exp(
        _deterministic_partial(chunks)[0] / _deterministic_partial(chunks)[1]
    )
    parts = [
        _deterministic_partial(chunks[:3]),
        _deterministic_partial(chunks[3:7]),
        _deterministic_partial(chunks[7:]),
    ]
    assert math.isclose(_merge_ppl(parts), single, rel_tol=0.0, abs_tol=0.0)


def test_merge_ppl_zero_tokens_is_inf():
    assert _merge_ppl([(0.0, 0), (0.0, 0)]) == float("inf")


def test_merge_ppl_overflow_is_inf():
    # nll_sum/tok_count huge → exp overflows → inf (guarded, not raised).
    assert _merge_ppl([(1e9, 1)]) == float("inf")
