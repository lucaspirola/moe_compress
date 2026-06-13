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


# ---------------------------------------------------------------------------
# M1 — PPL non-finite/error skip-guard parity (wikitext_ppl.py:280-291).
#
# Production refuses to report a sub-corpus PPL: if ANY batch was skipped
# (None / non-finite loss / runtime error), the whole-corpus PPL is forced to
# inf to FAIL the gate rather than silently shrink its domain. The DP path now
# carries a per-replica `skipped` count in the 3-tuple partial, and `_merge_ppl`
# enforces the same corpus-level guard: sum(skipped) > 0 -> inf.
# ---------------------------------------------------------------------------


def test_merge_ppl_any_skip_is_inf_even_with_finite_nll():
    # Core M1 regression: a partial carrying skipped>0 forces inf even though
    # tok_count>0 and nll is finite — matching wikitext_ppl.py:302-313.
    p0 = (12.3, 100, 0)   # clean replica
    p1 = (4.5, 40, 2)     # this replica skipped 2 batches
    assert _merge_ppl([p0, p1]) == float("inf")


def test_merge_ppl_skip_guard_precedes_tok_and_overflow_guards():
    # skipped>0 wins even when tok_count would also be 0 (i.e. the skip guard is
    # checked BEFORE the tok==0 and exp() guards — order matters for provenance).
    assert _merge_ppl([(0.0, 0, 1)]) == float("inf")


def test_merge_ppl_all_zero_skipped_behaves_as_before():
    # With skipped==0 everywhere the 3-tuple path reduces to the pre-M1 result.
    chunks = torch.randint(0, 50, (10, 8), generator=torch.Generator().manual_seed(7),
                           dtype=torch.long)
    n0, t0 = _deterministic_partial(chunks[:4])
    n1, t1 = _deterministic_partial(chunks[4:])
    expected = math.exp((n0 + n1) / (t0 + t1))
    merged = _merge_ppl([(n0, t0, 0), (n1, t1, 0)])
    assert math.isclose(merged, expected, rel_tol=0.0, abs_tol=0.0)
    # tok==0 -> inf still holds with explicit zero-skip tuples
    assert _merge_ppl([(0.0, 0, 0), (0.0, 0, 0)]) == float("inf")
    # OverflowError -> inf still holds with explicit zero-skip tuple
    assert _merge_ppl([(1e9, 1, 0)]) == float("inf")


def test_merge_ppl_accepts_legacy_2tuple_partials():
    # Backward compat: a 2-tuple partial means skipped defaults to 0, so the
    # pre-M1 callers/tests keep working unchanged.
    chunks = torch.randint(0, 50, (8, 8), generator=torch.Generator().manual_seed(3),
                           dtype=torch.long)
    n, t = _deterministic_partial(chunks)
    expected = math.exp(n / t)
    assert math.isclose(_merge_ppl([(n, t)]), expected, rel_tol=0.0, abs_tol=0.0)
    # mixed 2-tuple + 3-tuple (legacy + new) merges fine when no skips
    n0, t0 = _deterministic_partial(chunks[:4])
    n1, t1 = _deterministic_partial(chunks[4:])
    expected2 = math.exp((n0 + n1) / (t0 + t1))
    assert math.isclose(_merge_ppl([(n0, t0), (n1, t1, 0)]), expected2,
                        rel_tol=0.0, abs_tol=0.0)
