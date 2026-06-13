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

import tempfile
from pathlib import Path

from .eval_harness import PINNED_GEN_BATCH_SIZE


# ---------------------------------------------------------------------------
# Per-replica student materialization + worker reload (C1 / NEW-C1)
# ---------------------------------------------------------------------------


def _materialize_student(model, tokenizer):
    """Serialize the LIVE Stage-6 student to a fresh temp dir for worker reload.

    Stage 6 receives a live in-memory model and has NO on-disk student, so a
    spawned worker has nothing to load. We write a compressed checkpoint via
    ``save_compressed_checkpoint`` -- NOT ``model.save_pretrained``, which unpacks
    the stacked ``FactoredExperts`` tensors into ~24k per-expert keys and breaks
    the ``load_compressed_model`` resume path ("80 missing keys",
    model_io.py:1185-1197). Unwraps a ``torch.compile`` wrapper first (as
    humaneval.py:640 / math500.py:489). Returns the temp dir ``Path``.

    Called by the orchestrator ONLY on the eval_shard ENABLED branch, BEFORE any
    of the mutating steps (model.to("cpu") / forward-swap /
    _set_experts_implementation_s6). Best-effort ``shutil.rmtree`` after the join.
    """
    from ..utils.model_io import save_compressed_checkpoint

    tmp_dir = Path(tempfile.mkdtemp(prefix="stage6_eval_shard_src_"))
    model_unwrapped = getattr(model, "_orig_mod", model)
    save_compressed_checkpoint(
        model_unwrapped, tokenizer, tmp_dir,
        pipeline_stage="stage6_eval_shard_src",
    )
    return tmp_dir


def _reload_student_for_worker(tmp_dir, *, experts_impl_generative,
                               for_generate, device_map="cpu",
                               torch_dtype="float32"):
    """Reload the materialized student in a worker process + re-apply the FULL
    generative-env contract.

    A bare ``load_compressed_model`` is NOT enough: the saved checkpoint persists
    weights + config only, not the runtime patches ``EvalEnvironmentPlugin``
    applied. Reproduces, in order:

      1. ``load_compressed_model(..., attn_implementation="eager")`` -- eager is
         the gen path's hard requirement (eval_harness.py:85-91).
      2. ``model`` set to inference mode.
      3. ``_apply_stage6_kernel_patches(model, role="student")`` -- the
         cu130/Hopper fla GatedDeltaNet fix (eval_environment.py:594).
      4. Register the ``linear_attention -> full_attention`` mask passthrough in
         ``transformers.masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING``
         (eval_environment.py:638-646), else ``generate()`` raises
         ``KeyError: 'linear_attention'``. The mapping is process-global and the
         fresh worker has the unpatched mapping.
      5. (generate only) ``_set_experts_implementation_s6(model,
         experts_impl_generative)`` -- switch to the generative experts impl
         (humaneval.py:648 / math500.py:492). PPL keeps its forward-only impl.

    NEVER calls ``torch.compile`` -- the worker's ``model.forward`` is already the
    uncompiled forward the generative block needs (the equivalent of the
    in-process ``model.forward = _pre`` restore). All imports are function-local
    to keep ``tools/eval_shard`` a leaf utility (mirror ``_cov_replica_worker``).
    """
    from ..stage6.plugins.eval_environment import (
        _apply_stage6_kernel_patches,
        _set_experts_implementation_s6,
    )
    from ..utils.model_io import load_compressed_model

    model, tokenizer, _meta = load_compressed_model(
        tmp_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.eval()
    _apply_stage6_kernel_patches(model, role="student")

    # (4) linear_attention -> full_attention mask passthrough.
    try:
        from transformers import masking_utils as _mu
        _mapping = getattr(_mu, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", None)
        if isinstance(_mapping, dict) and "linear_attention" not in _mapping:
            if "full_attention" in _mapping:
                _mapping["linear_attention"] = _mapping["full_attention"]
    except Exception:  # noqa: BLE001 -- passthrough benign for non-Qwen3.5-MoE
        pass

    if for_generate:
        _set_experts_implementation_s6(model, experts_impl_generative)
    return model, tokenizer


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


def _merge_ppl(partials):
    """Merge per-shard ``(nll_sum, tok_count)`` partials into the global PPL.

    Sums the disjoint partials and returns ``exp(sum_nll / sum_tok)``. Exact
    because the per-row forward is BATCH_INVARIANT and the ``mean_loss *
    (numel - n_rows)`` rescale recovers the true NLL sum (wikitext_ppl.py:260).
    Carries the SAME guards as wikitext_ppl.py:273-295: ``tok_count == 0 -> inf``
    (PPL undefined) and ``OverflowError -> inf``.
    """
    import math

    sum_nll = 0.0
    sum_tok = 0
    for nll, tok in partials:
        sum_nll += float(nll)
        sum_tok += int(tok)
    if sum_tok == 0:
        return float("inf")
    try:
        return math.exp(sum_nll / sum_tok)
    except OverflowError:
        return float("inf")


__all__ = [
    "PINNED_GEN_BATCH_SIZE",
    "_group_aligned_split",
    "_even_split",
    "_merge_completions",
    "_merge_ppl",
    "_materialize_student",
    "_reload_student_for_worker",
]
