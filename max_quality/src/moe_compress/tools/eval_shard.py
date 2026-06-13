"""``tools/eval_shard`` — data-parallel eval-shard for Stage 6 (HumanEval,
MATH-500, WikiText-PPL).

Opt-in, **default single-GPU byte-identical**. Replicate the (single-GPU-
fitting ~50 GB) Stage-6 student on each of ``G`` GPUs, split the independent
eval examples across replicas, generate/score each shard, and merge —
RESULT-PRESERVING (byte-identical completions / exact PPL).

The gen path is METRIC_PINNED: a completion is a pure function of its
group-of-``PINNED_GEN_BATCH_SIZE`` and within-group order (left-pad geometry
+ bf16 reduction order flip near-tied argmax). Therefore gen shards are cut
ONLY on multiples of 8 (``_group_aligned_split``) at a HARD-PINNED bs=8 with NO
auto-batch / OOM-backoff. The PPL path is BATCH_INVARIANT: any row boundary is
exact and each replica MAY size its own forward batch.

Per the ``tools/`` package contract this module is a leaf utility: the module
top imports only stdlib + torch + ``tools.eval_harness``. The spawn-target
workers do their stage-module imports (``_set_experts_implementation_s6``,
``load_compressed_model``) **function-locally** to avoid an import cycle, exactly
as ``stage3.plugins.covariance_collection._cov_replica_worker`` does.
"""
from __future__ import annotations

from .eval_harness import PINNED_GEN_BATCH_SIZE


# ---------------------------------------------------------------------------
# Split helpers (pure arithmetic — no torch)
# ---------------------------------------------------------------------------


def _group_aligned_split(n, replicas, group=PINNED_GEN_BATCH_SIZE):
    """Contiguous shards whose boundaries are multiples of ``group`` (or n).

    Guarantees every group-of-``group`` formed by the single-GPU
    ``_generate_batched`` slicing stays intact on exactly one shard, so the
    per-example completion (a pure function of its group + within-group order)
    is byte-identical to the single-GPU run after index-ordered concatenation.
    The short final group (n not divisible by group) is never split — it rides
    whole on the last shard.
    """
    if replicas <= 1 or n == 0:
        return [(0, n)]
    n_groups = (n + group - 1) // group          # ceil → the short tail is its own group
    replicas = min(replicas, n_groups)
    base = n_groups // replicas
    bounds, start_g = [], 0
    for r in range(replicas):
        end_g = n_groups if r == replicas - 1 else start_g + base
        start = start_g * group
        end = n if end_g == n_groups else end_g * group   # last shard → real n (short tail)
        bounds.append((start, end))
        start_g = end_g
    return bounds


def _even_split(n, replicas):
    """Contiguous, disjoint shards of ``[0, n)`` (no group constraint) — the
    PPL-path boundaries. Last shard absorbs the remainder (mirror
    ``covariance_collection._shard_calib``). Clamped to ``n``.
    """
    if replicas <= 1 or n == 0:
        return [(0, n)]
    replicas = min(replicas, n)
    base = n // replicas
    bounds, start = [], 0
    for r in range(replicas):
        end = n if r == replicas - 1 else start + base
        bounds.append((start, end))
        start = end
    return bounds


# ---------------------------------------------------------------------------
# Merge helpers (pure)
# ---------------------------------------------------------------------------


def _merge_completions(shard_results):
    """Concatenate contiguous gen-shard results into original index order.

    ``shard_results`` is a list of ``(start, end, completions)`` tuples. Shards
    are sorted by ``start`` and validated to tile ``[0, N)`` with no gap/overlap
    (defensive: an off-by-group split is exactly the silent-metric-flip risk
    this whole feature guards against), then their completion lists are
    concatenated. Because each completion is a pure function of its
    group-of-8 + within-group order, the concatenation reproduces the
    single-GPU completion list byte-for-byte.
    """
    ordered = sorted(shard_results, key=lambda t: t[0])
    merged: list[str] = []
    expected_start = 0
    for start, end, completions in ordered:
        if start != expected_start:
            raise ValueError(
                f"_merge_completions: shards are not contiguous from 0 — expected "
                f"start={expected_start}, got start={start} (gap/overlap). "
                f"Shard bounds: {[(s, e) for s, e, _ in ordered]}"
            )
        if len(completions) != end - start:
            raise ValueError(
                f"_merge_completions: shard [{start}:{end}] has {len(completions)} "
                f"completions but expected {end - start} (length mismatch)."
            )
        merged.extend(completions)
        expected_start = end
    return merged


__all__ = [
    "PINNED_GEN_BATCH_SIZE",
    "_group_aligned_split",
    "_even_split",
    "_merge_completions",
]
