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

import dataclasses
import tempfile
from pathlib import Path

from .eval_harness import PINNED_GEN_BATCH_SIZE


# ---------------------------------------------------------------------------
# Config dataclass + default-OFF resolver
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EvalShardConfig:
    """Frozen config for the Stage-6 DP eval-shard (``stage6_validate.eval_shard``).

    Default-OFF: ``enabled=False`` / ``replicas<=1`` / ``device_count()<2`` all
    route to the existing in-process path (byte-identical). ``ppl`` gates the
    PPL-DP path SEPARATELY from the gen path (M3): enabling eval-shard for the
    gen win never silently spawns N x ~50 GB reloads for the cheap PPL forward.
    """
    enabled: bool = False
    replicas: int = 0
    gpus_per_replica: int = 1
    ppl: bool = False

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        fields = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in fields}
        # Safety contract "OFF unless explicitly on": coerce bool flags to real
        # bools so a stray YAML string ("false") cannot silently enable the path.
        for flag in ("enabled", "ppl"):
            if flag in kw:
                kw[flag] = (kw[flag] is True) or (str(kw[flag]).strip().lower() == "true")
        if "replicas" in kw:
            kw["replicas"] = int(kw["replicas"])
        if "gpus_per_replica" in kw:
            kw["gpus_per_replica"] = int(kw["gpus_per_replica"])
        return cls(**kw)


def _should_shard(cfg, n_examples, *, group_aligned=True):
    """Return the effective replica count (``0`` = in-process single-GPU path).

    Returns 0 when disabled, when ``replicas<=1``, or when fewer than 2 CUDA
    devices are visible. Otherwise clamps the requested replica count to the
    visible device count, to ``n_examples``, and -- on the gen path
    (``group_aligned=True``) -- to the number of whole groups-of-8, so every shard
    is a whole number of groups (the METRIC_PINNED boundary guarantee).
    """
    import torch

    if not cfg.enabled or cfg.replicas <= 1 or n_examples <= 0:
        return 0
    n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_dev < 2:
        return 0
    eff = min(cfg.replicas, n_dev, n_examples)
    if group_aligned:
        n_groups = (n_examples + PINNED_GEN_BATCH_SIZE - 1) // PINNED_GEN_BATCH_SIZE
        eff = min(eff, n_groups)
    return eff if eff > 1 else 0


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
    """Merge per-shard ``(nll_sum, tok_count, skipped)`` partials into the global PPL.

    Sums the disjoint partials and returns ``exp(sum_nll / sum_tok)``. Exact
    because the per-row forward is BATCH_INVARIANT and the ``mean_loss *
    (numel - n_rows)`` rescale recovers the true NLL sum (wikitext_ppl.py:260).

    Carries the SAME guards as wikitext_ppl.py:280-317, IN THE SAME ORDER:

      1. ``sum(skipped) > 0 -> inf`` (M1, wikitext_ppl.py:295-313): a replica
         that skipped any batch (None / non-finite loss / runtime error) makes
         the merged PPL a sub-corpus metric. Spec §9 mandates PPL over the FULL
         retained-chunk corpus, so ANY skip forces inf to FAIL the gate rather
         than silently report a partial-corpus PPL. Checked FIRST so it wins
         even when ``sum_tok`` also happens to be 0.
      2. ``sum_tok == 0 -> inf`` (PPL undefined; wikitext_ppl.py:295-301).
      3. ``OverflowError -> inf`` (wikitext_ppl.py:314-317).

    Backward-compatible: a 2-tuple partial ``(nll, tok)`` means ``skipped=0``
    (tolerant unpack), so pre-M1 callers/tests keep working unchanged.
    """
    import math

    sum_nll = 0.0
    sum_tok = 0
    sum_skipped = 0
    for p in partials:
        nll, tok, *rest = p
        skipped = rest[0] if rest else 0
        sum_nll += float(nll)
        sum_tok += int(tok)
        sum_skipped += int(skipped)
    if sum_skipped > 0:
        # Sub-corpus PPL refused (wikitext_ppl.py:302-313) — checked BEFORE the
        # tok==0 and exp() guards so a skip always forces inf.
        return float("inf")
    if sum_tok == 0:
        return float("inf")
    try:
        return math.exp(sum_nll / sum_tok)
    except OverflowError:
        return float("inf")


# ---------------------------------------------------------------------------
# Gen replica worker + spawn driver (run_dp_generate)
# ---------------------------------------------------------------------------

# Test-only seam: a picklable stub-generate key. When passed to the worker, the
# worker SKIPS the model reload + _generate_batched and instead applies the
# named deterministic stub to each prompt. This lets the spawn + split + merge
# plumbing be tested on CPU with no real model. Production callers never pass it.
_STUB_GENERATE_PROMPTLEN = "promptlen"
# Width-derived stub MODEL key (golden gate): the worker reconstructs the
# stub model + tokenizer and runs the REAL _generate_batched so the left-pad
# + pad_to_multiple_of group geometry is exercised end-to-end.
_STUB_WIDTH_MODEL = "width_model"


def _apply_stub_generate(key, prompts):
    if key == _STUB_GENERATE_PROMPTLEN:
        return [f"<{p}|len{len(p)}>" for p in prompts]
    raise ValueError(f"_apply_stub_generate: unknown stub key {key!r}")


def _gen_replica_worker(
    replica_idx,
    visible_devices,
    tmp_dir,
    prompts_shard,
    max_new,
    experts_impl_generative,
    out_file,
    stub_generate,
):
    """Spawn target: one data-parallel GEN replica (mirror ``_cov_replica_worker``).

    Pins itself to its GPU via ``CUDA_VISIBLE_DEVICES``, reloads the student with
    the FULL generative-env contract (``_reload_student_for_worker(...,
    for_generate=True)``), runs ``_generate_batched`` on its prompt sub-list at a
    HARD-PINNED bs=8 (NO size_batch, NO run_with_oom_backoff — both would move bs
    off 8 and flip near-tied argmax), and writes its completions (JSON) to
    ``out_file``. Module-level (picklable) so it is a valid spawn target;
    re-imports inside so the child has a clean import graph.
    """
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    import json as _json
    import torch as _torch
    from .eval_harness import _generate_batched, PINNED_GEN_BATCH_SIZE as _PIN
    from .eval_shard import (
        _apply_stub_generate as _stub,
        _reload_student_for_worker as _reload,
        _STUB_WIDTH_MODEL,
    )

    prompts_shard = list(prompts_shard)
    if stub_generate == _STUB_WIDTH_MODEL:
        from .eval_shard_stub import (
            build_width_stub_model as _bm,
            build_width_stub_tokenizer as _bt,
        )
        completions = _generate_batched(
            _bm(), _bt(), prompts_shard, max_new=max_new,
            device=None, batch_size=_PIN,
        )
    elif stub_generate is not None:
        completions = _stub(stub_generate, prompts_shard)
    else:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        model, tokenizer = _reload(
            tmp_dir, experts_impl_generative=experts_impl_generative,
            for_generate=True,
            device_map=("cuda" if _torch.cuda.is_available() else "cpu"),
            torch_dtype="bfloat16",
        )
        # HARD-PINNED bs=8 — no auto-batch, no OOM-backoff (METRIC_PINNED gen path).
        completions = _generate_batched(
            model, tokenizer, prompts_shard, max_new=max_new,
            device=device, batch_size=_PIN,
        )
    with open(out_file, "w", encoding="utf-8") as _f:
        _json.dump(completions, _f)


def run_dp_generate(prompts, *, tmp_dir, replicas, gpus_per_replica, max_new,
                    experts_impl_generative, cfg, out_dir, _stub_generate=None,
                    _stub_model=None, _force_bounds=None):
    """Data-parallel HumanEval/MATH-500 generation.

    Fan out ``replicas`` child processes (torch.multiprocessing spawn), each
    pinned to its GPU subset, generating its group-aligned prompt shard at a
    hard-pinned bs=8; then read the per-replica completion files and merge them
    in original index order (``_merge_completions``). Boundaries come from
    ``_group_aligned_split`` so every group-of-8 stays whole on one replica and
    the merged completions are byte-identical to the single-GPU run.

    ``_stub_generate`` is the test-only seam (see ``_STUB_GENERATE_PROMPTLEN``).
    """
    import json
    from pathlib import Path as _Path
    import torch.multiprocessing as _mp

    n = len(prompts)
    # _force_bounds is the test-only NEGATIVE-CONTROL seam (a NON-mod-8 split);
    # production always uses the group-aligned boundaries.
    bounds = _force_bounds if _force_bounds is not None else _group_aligned_split(n, replicas)
    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spawn_args = []
    out_files = []
    for r, (start, end) in enumerate(bounds):
        out_file = out_dir / f"_gen_replica_{r}.json"
        out_files.append((start, end, out_file))
        dev_lo = r * gpus_per_replica
        dev_hi = dev_lo + gpus_per_replica
        visible = ",".join(str(d) for d in range(dev_lo, dev_hi))
        _stub_arg = _stub_model if _stub_model is not None else _stub_generate
        spawn_args.append(
            (r, visible, str(tmp_dir), list(prompts[start:end]), max_new,
             experts_impl_generative, str(out_file), _stub_arg)
        )

    ctx = _mp.get_context("spawn")
    procs = []
    for args in spawn_args:
        p = ctx.Process(target=_gen_replica_worker, args=args)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Stage 6 DP generate: replica process exited with code {p.exitcode}"
            )

    shard_results = []
    for start, end, out_file in out_files:
        with open(out_file, "r", encoding="utf-8") as _f:
            completions = json.load(_f)
        shard_results.append((start, end, completions))
    return _merge_completions(shard_results)


# ---------------------------------------------------------------------------
# PPL replica worker + spawn driver (run_dp_ppl)
# ---------------------------------------------------------------------------

# Test-only seam: a picklable stub-PPL key. When passed, the worker SKIPS the
# model reload + forward and computes a deterministic per-row partial instead,
# so the spawn + even-split + exact partial-sum merge plumbing is testable on
# CPU with no real model. Production callers never pass it.
_STUB_PPL_MOD7 = "mod7"


def _stub_ppl_partial(key, chunk_rows):
    """Deterministic (nll_sum, tok_count) over the given rows, using the SAME
    mean_loss * (numel - n_rows) rescale the production code uses. The per-row
    contribution is independent of the forward batch, so the merge of disjoint
    shards equals the single pass exactly (BATCH_INVARIANT property).

    NEVER skips a batch: every row is deterministic and finite, so the worker's
    skip count is implicitly 0 for the stub path (no None / non-finite / raising
    forward can occur)."""
    if key != _STUB_PPL_MOD7:
        raise ValueError(f"_stub_ppl_partial: unknown stub key {key!r}")
    numel = int(chunk_rows.numel())
    n_rows = int(chunk_rows.shape[0])
    denom = max(numel - n_rows, 1)
    per_tok = (chunk_rows.to(_torch_float64()) % 7).sum() / denom
    mean_loss = float(per_tok)
    nll = mean_loss * (numel - n_rows)
    return nll, (numel - n_rows)


def _torch_float64():
    import torch as _t
    return _t.float64


def _ppl_forward_all(model, shard, device, bs, iter_batches):
    """Forward every batch of ``shard`` and accumulate ``(nll_sum, tok_count,
    skipped)`` with PRODUCTION skip-semantics (wikitext_ppl.py:264-288).

    Per batch, mirroring the single-GPU loop EXACTLY:
      * ``out.loss is None`` -> skip (count it), continue.
      * ``not math.isfinite(loss_val)`` -> skip (count it), continue.
      * a runtime error in the forward -> skip (count it), continue.

    A skipped batch contributes NOTHING to ``nll_sum`` / ``tok_count`` (so a
    None / nan / inf loss can never poison the sum, and a raising batch can
    never crash the replica). ``_merge_ppl`` then forces the WHOLE-corpus PPL
    to inf when ``sum(skipped) > 0`` (wikitext_ppl.py:302-313).

    The ENTIRE per-batch block — ``model(...)``, the None-check, ``.item()``,
    the isfinite check, AND the nll accumulation — sits inside ONE ``try`` to
    match production's single-try scope (wikitext_ppl.py:265-288). This matters
    because ``out.loss.item()`` forces a CUDA sync that can surface a DEFERRED
    device-side assert as a raise; production skip-counts that, so we must too
    (a narrower try would let it escape and crash the replica).

    CRUCIAL: ``torch.cuda.OutOfMemoryError`` is RE-RAISED, never swallowed —
    the enclosing ``run_with_oom_backoff`` (auto_batch.py:188) catches OOM to
    halve the batch and rerun. The OOM clause MUST precede the generic
    ``Exception`` clause so OOM propagates instead of being skip-counted.

    Returns ``(nll_sum, tok_count, skipped)``.
    """
    import math
    import torch as _torch

    nsum, tcount, skipped = 0.0, 0, 0
    with _torch.no_grad():
        for batch in iter_batches(shard, batch_size=bs):
            batch = batch.to(device)
            try:
                out = model(input_ids=batch, labels=batch)
                if out.loss is None:
                    skipped += 1
                    continue
                loss_val = float(out.loss.item())
                if not math.isfinite(loss_val):
                    skipped += 1
                    continue
                nsum += loss_val * (batch.numel() - batch.shape[0])
                tcount += batch.numel() - batch.shape[0]
            except _torch.cuda.OutOfMemoryError:
                raise
            except Exception:  # noqa: BLE001 — skip the batch, mirror wikitext_ppl.py:285
                skipped += 1
                continue
    return nsum, tcount, skipped


def _ppl_replica_worker(
    replica_idx,
    visible_devices,
    tmp_dir,
    chunks_file,
    shard_start,
    shard_end,
    ppl_bs,
    experts_impl_generative,
    auto_batch_cfg_dict,
    out_file,
    stub_ppl,
):
    """Spawn target: one data-parallel PPL replica.

    Pins its GPU, reloads the student (``_reload_student_for_worker(...,
    for_generate=False)`` — eager attn + kernel patch + mask passthrough; keeps
    the PPL experts impl, NO compile), forwards its chunk sub-rows. Per-replica
    auto-batch: ``resolve_batch(..., FidelityClass.BATCH_INVARIANT, ...)`` probes
    THIS replica's pinned-device VRAM, and the forward is wrapped in
    ``run_with_oom_backoff`` (safe — PPL is batch-invariant). Accumulates
    ``(nll_sum, tok_count)`` with the ``wikitext_ppl.py:260`` rescale and writes
    the partial to ``out_file``. Module-level (picklable); re-imports inside.
    """
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    import json as _json
    import torch as _torch
    from .eval_shard import (
        _stub_ppl_partial as _stub,
        _reload_student_for_worker as _reload,
    )

    chunks = _torch.load(chunks_file)
    shard = chunks[shard_start:shard_end]

    if stub_ppl is not None:
        # Stub path never produces a None/non-finite/raising batch, so skipped=0.
        nll_sum, tok_count = _stub(stub_ppl, shard)
        skipped = 0
    else:
        from .eval_shard import _ppl_forward_all as _fwd_all
        from ..utils.auto_batch import (
            AutoBatchConfig as _ABC, FidelityClass as _FC,
            resolve_batch as _resolve_batch, run_with_oom_backoff as _backoff,
        )
        from ..utils.calibration import iter_batches as _iter_batches

        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        model, _tok = _reload(
            tmp_dir, experts_impl_generative=experts_impl_generative,
            for_generate=False,
            device_map=("cuda" if _torch.cuda.is_available() else "cpu"),
            torch_dtype="bfloat16",
        )
        _abc = _ABC.from_dict(auto_batch_cfg_dict)

        def _cost_probe(micro_batch):
            mb = shard[:micro_batch].to(device)
            with _torch.no_grad():
                model(input_ids=mb, labels=mb)
            if _torch.cuda.is_available():
                return int(_torch.cuda.max_memory_allocated(device))
            return 0

        eff_bs = _resolve_batch(_cost_probe, int(ppl_bs), _FC.BATCH_INVARIANT, _abc)

        # _ppl_forward_all carries production skip-semantics (None/non-finite/
        # error -> skip+count) AND re-raises CUDA OOM so the backoff can halve.
        nll_sum, tok_count, skipped = _backoff(
            lambda bs: _fwd_all(model, shard, device, bs, _iter_batches),
            eff_bs, int(ppl_bs),
        )

    with open(out_file, "w", encoding="utf-8") as _f:
        _json.dump([nll_sum, tok_count, skipped], _f)


def run_dp_ppl(chunks, *, tmp_dir, replicas, gpus_per_replica, ppl_bs,
               experts_impl_generative, cfg, out_dir, auto_batch_cfg_dict=None,
               _stub_ppl=None):
    """Data-parallel WikiText-PPL.

    Fan out ``replicas`` child processes over an even row-split (no group
    constraint — BATCH_INVARIANT), each forwarding its chunk sub-rows at a
    per-replica auto-batch; then sum the per-shard ``(nll_sum, tok_count)``
    partials and return ``exp(sum_nll / sum_tok)`` (``_merge_ppl``). Exact
    regardless of boundary or per-replica batch.

    ``_stub_ppl`` is the test-only seam (see ``_STUB_PPL_MOD7``).
    """
    import json
    from pathlib import Path as _Path
    import torch as _torch
    import torch.multiprocessing as _mp

    n = int(chunks.shape[0])
    bounds = _even_split(n, replicas)
    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_file = out_dir / "_ppl_chunks.pt"
    _torch.save(chunks, chunks_file)

    spawn_args = []
    out_files = []
    for r, (start, end) in enumerate(bounds):
        out_file = out_dir / f"_ppl_replica_{r}.json"
        out_files.append(out_file)
        dev_lo = r * gpus_per_replica
        dev_hi = dev_lo + gpus_per_replica
        visible = ",".join(str(d) for d in range(dev_lo, dev_hi))
        spawn_args.append(
            (r, visible, str(tmp_dir), str(chunks_file), start, end, ppl_bs,
             experts_impl_generative, auto_batch_cfg_dict, str(out_file), _stub_ppl)
        )

    ctx = _mp.get_context("spawn")
    procs = []
    for args in spawn_args:
        p = ctx.Process(target=_ppl_replica_worker, args=args)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Stage 6 DP PPL: replica process exited with code {p.exitcode}"
            )

    partials = []
    for out_file in out_files:
        with open(out_file, "r", encoding="utf-8") as _f:
            rec = json.load(_f)
        # Tolerant unpack: 3-element [nll, tok, skipped] (current worker) or a
        # 2-element [nll, tok] legacy file (skipped defaults to 0).
        nll, tok, *rest = rec
        skipped = rest[0] if rest else 0
        partials.append((nll, tok, skipped))
    return _merge_ppl(partials)


__all__ = [
    "PINNED_GEN_BATCH_SIZE",
    "EvalShardConfig",
    "_should_shard",
    "_group_aligned_split",
    "_even_split",
    "_merge_completions",
    "_merge_ppl",
    "_materialize_student",
    "_reload_student_for_worker",
    "_gen_replica_worker",
    "run_dp_generate",
    "_STUB_GENERATE_PROMPTLEN",
    "_STUB_WIDTH_MODEL",
    "_ppl_replica_worker",
    "_ppl_forward_all",
    "run_dp_ppl",
    "_STUB_PPL_MOD7",
]
