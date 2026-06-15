"""AA-SVD cross-covariance + B-covariance collection (Theorem 3.2 / Corollary 3.3).

Paper
-----
"AA-SVD: Activation-Aware SVD with Cross-Covariance Calibration" —
arXiv:2604.02119 (audit/spec_compliance/01_papers/2604.02119/source.md).

Theorem 3.2 of AA-SVD prescribes the factorization
``M = W·C·S⁻¹·L_B^T`` where ``B`` is the post-prune input data matrix
``X_post`` (rows = tokens — code/numpy convention; the paper uses
cols = tokens so its ``B·B^T`` becomes ``B^T·B`` here),
``S = B^T·B = E[X_post^T X_post]`` is the post-prune input
*covariance*, ``L_B`` is the symmetric factor of ``S`` (i.e.
``S = L_B · L_B^T``), and ``C = E[X_pre^T X_post]`` is the
cross-covariance between pre- and post-prune layer inputs. Corollary
3.3 (the special case ``A = S``) reduces to ``M = W · L_B^T`` when only
the post-prune covariance is available.

Naming bridge (paper ↔ code): the paper's covariance ``S`` is what
this codebase calls ``B_cov`` / ``B_acc`` / the ``B`` accumulator —
i.e. the variable named ``B`` in the *code* IS paper-``S``, not the
paper's data matrix. From here on this docstring uses paper notation
(``S`` for the covariance) exclusively when discussing math, and the
code symbol (``B_acc``, ``_bcov_*.pt``) only when discussing artifacts.
The paper-data-matrix sense of ``B`` is NOT reused after this
paragraph.

This plugin owns the post-prune covariance pass: for every MoE
``(layer, expert, matrix)`` tuple, accumulate ``S`` (always; into the
code accumulator ``B_acc``) and ``C`` (gate_proj only — see deviation
D6 below) during a single
dual-forward pass against the teacher (pre-prune model) and the
post-prune student. Also loads the Stage 2 input-covariance sidecar
(``A_cov`` from ``_stage2_input_covariance.pt``) which downstream
factorisation (``aa_svd_factor.py``) consults but does NOT substitute
into the Theorem 3.2 cross-cov slot (see "Path 2 retirement" below).

Official code
-------------
``atulkumarin/AA-SVD`` @ commit
``1fa1b686cd9b13a77607a676564e37d438a176c8`` (2026-04-22) —
github.com/atulkumarin/AA-SVD.

Live factorisation paths (after Path 2 retirement)
--------------------------------------------------
Downstream ``aa_svd_factor.py`` (the ``_aa_svd`` factor function)
implements two paths only:

  * **Path 1 — Theorem 3.2 (paper-exact)**: ``M = W·C·S⁻¹·L_B^T`` when
    both ``C`` (cross-cov from this plugin's dual-forward) and ``S``
    (post-prune input covariance — code symbol ``B_acc`` / ``B_cov``
    that IS paper-``S``, per the Naming bridge above) are available.
    This is the default when ``aa_svd.cross_covariance: true`` (the
    configured default).
  * **Path 3 — Corollary 3.3 (S-only fallback)**: ``M = W·L_B^T`` when
    ``C`` is unavailable. Used when ``aa_svd.cross_covariance: false``
    or when the teacher load is suppressed.

An earlier "Path 2" substituted the pre-prune *auto*-covariance ``A``
(from Stage 2) into the Theorem 3.2 slot in place of ``C``. That path
was retired: it produced ``U·V ≈ W·A·S⁻¹·L_B^T`` rather than
approximating ``W``, breaking ``FactoredExperts`` forward and Stage 4
EoRA residual (see the Path-2-retirement comment block in ``_aa_svd``
of ``aa_svd_factor.py`` / tests in ``test_aa_svd_correctness.py``).
The ``A_cov`` sidecar load is still performed because L-BFGS
refinement (Stage 4) consumes it, but the Stage 3 rank-k factor uses
only ``C`` (Path 1) or omits it (Path 3).

Deviation: D6 — cross-covariance scope (gate-only, MoE-specific)
----------------------------------------------------------------
Paper Theorem 3.2 requires cross-covariance C for all linear layers
and uses a single shared-sample formulation per layer (one ``X_pre``
/ ``X_post`` per token). This plugin's MoE-specific resolution:

  * Cross-covariance C is collected per ``(layer, student_expert)``
    on ``gate_proj`` inputs only (``up_proj`` shares the same hidden
    state pre-routing — covered by the gate_proj entry via the
    factorisation-time ``_cov_lookup`` fallback). ``down_proj`` has
    no cross-cov because the teacher's per-expert intermediate
    activations would need full expert-dispatch instrumentation that
    the project does not implement; ``down_proj`` therefore falls
    back to Path 3 (Corollary 3.3, B-only) at factorisation time.
  * The per-expert formulation is asymmetric with the paper's
    shared-sample C: each student expert ``e`` accumulates the cross
    term over the teacher's representation of *the token positions
    that the student routes to e*. This is the natural MoE
    generalisation — teacher and student route different subsets, so
    a single shared C per layer would mis-attribute cross terms
    between experts. The asymmetry is the price of having any
    cross-cov at all when routing diverges.

Rationale: gate/up share the same hidden state pre-routing so one
capture covers both; down_proj is expert-internal (post gate+up) and
differs between teacher and student expert sets. Per-expert
attribution is required because teacher/student routing diverges.

Deviation: D-cov-storage-fp16 (SHARED with Stage 2)
---------------------------------------------------
Stage 2 covariance + Stage 3 B-cov persisted in **fp16** (not fp32).
Paper §5 (covariance side-collection) originally stated fp32 storage
citing Swift-SVD certification. The project persists fp16 (10
mantissa bits, strictly higher than bf16's 7 bits) for both
``_stage2_input_covariance.pt`` and ``_bcov_*.pt``; eigendecomposition
still runs in fp64 in-memory.

Rationale: fp16 produces cleaner Stage 3 rank-deficiency outcomes
than bf16 in spot checks. Halves the persisted-covariance disk
footprint vs fp32 (~2× saving on the gigabyte-scale covariance
artifact) without measurable downstream PPL / zero-shot drift.
Switching back to fp32 is a one-line config flip if a future model
exposes precision sensitivity.

Naming-history note
-------------------
"Phase A" (legacy Stage 3 monolith terminology) is naming-historical.
The current plugin architecture has no phase taxonomy; new prose
drops the labels. Existing log lines / Trackio keys preserved for
dashboard back-compat.

Tool inventory (relocated verbatim):

* ``_collect_covariances`` — collects post-prune input covariance B and
  (optionally) cross-covariance C per (layer, expert, matrix);
* ``_collect_pruned_input_covariance`` — public alias of ``_collect_covariances``;
* ``_load_stage2_covariance`` — loads the Stage-2 covariance payload from disk;
* the RSS/memory helpers ``_proc_rss_gb``, ``_maxrss_gb``, ``_fmt`` used by the
  per-layer telemetry inside ``_collect_covariances``.

All six symbols are byte-identical copies of the monolith bodies; the monolith
re-imports them (``# noqa: F401`` block in ``stage3_svd.py``) so external
callers and tests keep their existing import paths — e.g.
``test_stage3_spill.py`` imports ``_collect_pruned_input_covariance`` from
``moe_compress.stage3_svd``.

Circular-import note (mirror of ``stage2/plugins/ream_cost.py``): this module
imports only from ``...utils.*``, ``...pipeline.*`` and stdlib — NEVER from
``stage3_svd`` or ``stage3.orchestrator``. ``stage3_svd`` imports *this* module
at load time, so a module-top ``from ..stage3_svd import ...`` here would
deadlock the import; nothing in this module does that.

``CovarianceCollectionPlugin`` is wired into the live Stage 3 plugin
sequencer (``stage3/orchestrator.py``) as the first phase hook
(``collect_covariances``). The legacy "S3-2 INERT / S3-7 wiring"
milestone labels are naming-historical.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ...utils.activation_hooks import (
    InputCovarianceAccumulator,
    capture_experts,
    instrument_experts,
)
from ...utils.auto_batch import (
    AutoBatchConfig,
    CudaMemProbe,
    run_with_oom_backoff,
    size_batch,
)
from ...utils.calibration import iter_batches
from ...utils.futures import drain_done_futures as _drain_done_futures
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)

# Cov-specific BACKSTOP cap on the auto-sized forward batch, in *sequences*
# (NOT v1's 4096 default — a 4096-seq dual-forward over G window layers, each
# holding a full ``[T, d_in]`` fp32 teacher tensor, would OOM-backoff
# repeatedly). ``size_batch``'s ``headroom_frac`` VRAM fit is the real limiter;
# this is only an upper clamp so a degenerate probe can't return an absurd bs.
_COV_MAX_CAP = 256


def _proc_rss_gb() -> float | None:
    """Per-process RSS in GB. Tighter bound on the pipeline's own memory
    footprint than ``virtual_memory().used`` (which is host-wide and
    floats with page cache from other tenants / cold mmap pages).
    Returns None if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:                                # noqa: BLE001
        return None


def _maxrss_gb() -> float | None:
    """Peak RSS since process start, monotonically non-decreasing.
    Best signal for ``did this layer's accumulator actually grow``."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:                                # noqa: BLE001
        return None


def _fmt(x):
    return f"{x:.1f}" if x is not None else "?"


# ---------------------------------------------------------------------------
# Lever 2 (data-parallel) — cross-replica spill reduce
# ---------------------------------------------------------------------------


def _consolidate_shift_covariance(
    spill_dir, out_path, layer_indices, *, storage_dtype,
) -> int:
    """Consolidate per-layer B-cov spills (the post-2.5 shift cov S=X'ᵀX')
    into one durable named artifact for Stage-4 EoRA shift whitening.

    Reads each ``spill_dir/layer_{idx}.pt`` (format_version 1, keyed
    (layer,expert,matrix) -> Tensor[d,d]; up_proj aliased to gate_proj
    upstream), merges the per-layer ``covariance`` dicts, and atomically
    saves ``{"format_version": 1, "covariance": {...}}`` + a manifest sibling.
    Near-zero cost: pure disk read+merge, no forward pass. Returns key count.
    """
    merged: dict = {}
    for li in layer_indices:
        p = Path(spill_dir) / f"layer_{li}.pt"
        if not p.exists():
            continue
        payload = torch.load(p, map_location="cpu", weights_only=True)
        cov = payload.get("covariance", {}) if isinstance(payload, dict) else {}
        for k, t in cov.items():
            merged[k] = t.to(storage_dtype)
    from ...utils.atomic_io import atomic_torch_save, write_manifest_last
    out_path = Path(out_path)
    atomic_torch_save(out_path, {"format_version": 1, "covariance": merged})
    manifest = out_path.with_suffix(out_path.suffix + ".MANIFEST.json")
    try:
        manifest.unlink(missing_ok=True)
    except OSError:
        pass
    write_manifest_last(out_path, manifest, schema_version=1,
                        extra_meta={"n_keys": len(merged),
                                    "artifact": "stage3_shift_covariance"},
                        compute_sha256=False)
    return len(merged)


def _reduce_spilled_cov_dirs(replica_dirs, out_dir, *, storage_dtype=None) -> list[int]:
    """Sum per-layer covariance spills from G data-parallel replicas into one
    canonical spill dir.

    Each replica processed a DISJOINT batch-shard and spilled per-layer files
    ``layer_{idx}.pt`` in the same on-disk format as
    :meth:`InputCovarianceAccumulator.spill_layer_to_disk`
    (``{"format_version": 1, "covariance": {key: tensor}, "tokens": {key: int}}``,
    key = ``(layer, expert, matrix)``). The Gram accumulator is a linear sum of
    per-token outer products, so the cross-replica reduce is exactly
    ``B = Σ_r B_r`` — summed key-wise in **fp32** then cast back to
    ``storage_dtype``, mirroring ``finalize_layer`` / ``_accumulate_payload``
    (activation_hooks.py:1081-1084, 1212-1216). Token counts sum exactly
    (integers).

    Determinism: replica dirs are processed in **sorted** order so a given
    (replicas, seed) is reproducible run-to-run (§4 determinism knob). Pure CPU.

    Returns the sorted list of layer indices written to ``out_dir``.
    """
    replica_dirs = [Path(d) for d in sorted(str(d) for d in replica_dirs)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Union of layer indices present across replicas (a replica may legitimately
    # be missing a layer file only if ALL are — otherwise it's a partial spill
    # bug we surface by KeyError-free union + per-file existence below).
    layer_ids: set[int] = set()
    for d in replica_dirs:
        for p in d.glob("layer_*.pt"):
            try:
                layer_ids.add(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue

    written: list[int] = []
    for li in sorted(layer_ids):
        merged_cov: dict = {}
        merged_tok: dict = {}
        resolved_dtype = storage_dtype
        for d in replica_dirs:
            p = d / f"layer_{li}.pt"
            if not p.exists():
                # A replica that saw no tokens for this layer contributes zero;
                # skip it (its absence is an additive identity).
                continue
            payload = torch.load(p, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or "covariance" not in payload:
                raise RuntimeError(
                    f"_reduce_spilled_cov_dirs: spill file {p} has unexpected "
                    f"layout (keys={list(payload.keys()) if isinstance(payload, dict) else 'n/a'})."
                )
            for k, t in payload["covariance"].items():
                if resolved_dtype is None:
                    resolved_dtype = t.dtype
                prev = merged_cov.get(k)
                if prev is None:
                    merged_cov[k] = t.to(torch.float32)
                else:
                    merged_cov[k] = prev + t.to(torch.float32)
            for k, n in payload.get("tokens", {}).items():
                merged_tok[k] = merged_tok.get(k, 0) + int(n)

        if not merged_cov:
            continue
        if resolved_dtype is None:
            resolved_dtype = torch.float32
        out_payload = {
            "format_version": 1,
            "covariance": {k: v.to(resolved_dtype) for k, v in merged_cov.items()},
            "tokens": merged_tok,
        }
        out_path = out_dir / f"layer_{li}.pt"
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(out_payload, tmp)
        import os as _os
        _os.replace(tmp, out_path)
        written.append(li)
    return written


# ---------------------------------------------------------------------------
# A1 — windowed single-pass covariance collection helpers
# ---------------------------------------------------------------------------


def _iter_windows(seq, window_size: int):
    """Yield contiguous windows of ``window_size`` items from ``seq`` (the last
    window absorbs the remainder). ``window_size <= 0`` is treated as 1.

    A window of ``G`` MoE layers is hooked with A7 ``capture_experts`` and the
    calibration set is forwarded ONCE per window — ``ceil(N/G)`` passes instead
    of N. ``G=1`` reproduces the per-layer structure (but on the native
    forward, NOT the old Python-loop golden — see PLAN §4.1).
    """
    g = max(1, int(window_size))
    for start in range(0, len(seq), g):
        yield seq[start:start + g]


def _resolve_single_pass(config: dict) -> bool:
    """Single source of truth for the single-pass G=N detection.

    True when ``stage3_svd.cov_single_pass`` is set, OR when
    ``multi_gpu.cov_window_size`` is the case/whitespace-insensitive sentinel
    ``"all"``. Used by BOTH the in-process orchestrator (to engage the CPU
    hot-accumulator on its B_acc/C_acc handles) AND the DP replica worker (to
    engage it on the worker's OWN collection accumulators). Keeping the
    detection here — the module the orchestrator already imports from — ensures
    the two paths can never drift, which would re-open the all-40-layer
    GPU-resident-Gram OOM that Task B exists to prevent.
    """
    s3 = config.get("stage3_svd") or {}
    raw = (config.get("multi_gpu") or {}).get("cov_window_size", "auto")
    return bool(
        s3.get("cov_single_pass", False)
        or (isinstance(raw, str) and raw.strip().lower() == "all")
    )


def _cov_per_layer_gram_bytes(d_hid: int, d_int: int, n_exp: int, cross_cov: bool) -> float:
    """Persistent on-GPU fp32 Gram footprint for ONE cov-window layer.

    The accumulator holds a fp32 Gram PER (expert, matrix): per expert the input
    covariance B = gate_proj ``[d_hid, d_hid]`` + down_proj ``[d_int, d_int]``;
    when cross-cov is enabled the cross covariance C adds gate_proj
    ``[d_hid, d_hid]``. ``up_proj`` aliases ``gate_proj`` (same d_hid input) — no
    separate Gram. So the per-LAYER footprint is ``n_exp ×`` the per-expert bytes,
    NOT a single expert's. Omitting the ``× n_exp`` factor (the historical bug)
    under-counted ~``n_exp``× and made ``cov_window_size: auto`` pick a G whose
    Grams overflow VRAM. fp32 → 4 bytes.
    """
    per_expert = (float(d_hid) ** 2 + float(d_int) ** 2) * 4.0
    if cross_cov:
        per_expert += float(d_hid) ** 2 * 4.0
    return float(n_exp) * per_expert


def _resolve_cov_window(config: dict, n_layers: int) -> int:
    """Resolve the cov collection window size G ∈ [1, n_layers].

    Reads ``multi_gpu.cov_window_size`` (same block as ``cov_replicas``):
      * absent / ``"auto"`` → VRAM-probe via ``torch.cuda.mem_get_info()``;
        on a CPU-only box (no CUDA) ``auto`` degrades to ``G=1``.
      * explicit int → clamp to ``[1, n_layers]``.

    Unlike ``orchestrator._resolve_cov_replicas`` (config-ONLY), ``auto`` here
    adds a real per-device VRAM probe: each hooked layer holds a persistent
    on-device fp32 Gram sized by ``_cov_per_layer_gram_bytes`` —
    ``n_exp · ((d_hid² + d_int²) + d_hid²·[cross]) · 4`` bytes — until its window's
    ``finalize_layer`` runs at window end, plus transient gathered activations. We
    size ``G ≈ floor((free − headroom) / per_layer_bytes)``.
    """
    if n_layers <= 0:
        return 1
    s3 = config.get("stage3_svd") or {}
    if s3.get("cov_single_pass", False):
        return n_layers
    mg = config.get("multi_gpu") or {}
    req = mg.get("cov_window_size", "auto")
    if req is None:
        req = "auto"
    if isinstance(req, str):
        if req.strip().lower() == "all":
            return n_layers
        if req.strip().lower() != "auto":
            try:
                req = int(req)
            except ValueError:
                req = "auto"
    if isinstance(req, int):
        return max(1, min(int(req), n_layers))

    # auto: VRAM probe. CPU-only / no CUDA ⇒ G=1 (clean degrade).
    if not torch.cuda.is_available():
        return 1
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:                                # noqa: BLE001
        return 1
    # Per-layer persistent Gram estimate. d_hid / d_int / n_exp are seeded from the
    # model config by the caller (orchestrator / DP worker); the literal fallbacks
    # are conservative over-estimates used only if seeding failed. (The old
    # ``config["model"].get("hidden_size")`` fallback was dead — that block is the
    # run config, not the HF model config, and num_experts is nested under
    # text_config — so it silently returned the defaults; dropped.)
    d_hid = int(config.get("_d_hidden") or 8192)
    d_int = int(config.get("_d_intermediate") or 8192)
    n_exp = int(config.get("_n_experts") or 256)
    s3 = config.get("stage3_svd") or {}
    cross_cov = bool((s3.get("aa_svd") or {}).get("cross_covariance", True))
    per_layer_bytes = _cov_per_layer_gram_bytes(d_hid, d_int, n_exp, cross_cov)
    # Reserve headroom for transient activations + the native forward's own
    # working set (conservative: keep 25% of free VRAM in reserve).
    usable = free * 0.75
    g = int(usable // per_layer_bytes) if per_layer_bytes > 0 else 1
    g = max(1, min(g, n_layers))
    log.info(
        "Stage 3 cov window auto-size: free=%.1fGB per_layer~%.1fMB -> G=%d (N=%d)",
        free / 1e9, per_layer_bytes / 1e6, g, n_layers,
    )
    return g


def _resolve_cov_batch_size(s3: dict) -> int:
    """Resolve the cov-collection batch size (A6).

    Reads the cov-specific ``stage3_svd.cov_batch_size`` key, defaulting to the
    inherited ``stage3_svd.batch_size`` (default 1) so the 1-GPU golden is
    untouched (the golden was generated at ``batch_size``):

      * absent              → inherits ``batch_size`` (no behavior change).
      * explicit int / str  → passes through (operator opt-in on a sharded box).
      * ``"auto"``          → this resolver returns the inherited floor; the
        VRAM-auto-sizing happens in ``_collect_covariances`` (gated by
        ``_cov_is_auto`` = ``cov_batch_size=="auto"`` AND ``auto_batch.enabled``),
        which probes free VRAM with the G window resident, auto-sizes the cov
        forward batch, and OOM-backs-off to this floor (=1). This applies to BOTH
        the in-process (1-GPU) path AND each DP (≥2-GPU) replica: every replica
        probes its OWN pinned-device VRAM and sizes INDEPENDENTLY — the
        per-sequence pin makes the key-wise DP reduce batch-independent, so NO
        cross-replica min(candidate) agreement is needed (supersedes spec §6).
        The returned int is always positive (it is the backoff ``floor`` + the
        non-auto fallback); ``_resolve_cov_batch_size`` never returns the
        ``"auto"`` sentinel.

    Reduction-pin (per-sequence grouping): with the cov Gram now accumulated via
    the per-sequence pinned split (``InputCovarianceAccumulator.update_grouped``,
    routed in ``_collect_covariances``), raising ``cov_batch_size`` NO LONGER
    changes the reduction GROUPING — each source sequence's tokens are reduced in
    the same bs=1 order regardless of how the forward batch merged them. The
    consequences differ per matrix:

      * gate_proj/up B and the cross-cov C: the Gram is now BITWISE-INVARIANT to
        ``cov_batch_size`` (the per-sequence operands and their accumulation order
        are identical at any batch size).
      * factored down_proj B: ``allclose`` (~1e-6), NOT bitwise — the residual is
        upstream forward-activation drift (GPU matmul output is batch-shape
        dependent), bounded and N-INDEPENDENT, not the old N-scaling reduction
        drift. This is quality-neutral.

    So a bigger cov batch is now quality-neutral rather than golden-breaking as it
    was before the pin (gate_proj/up B + cross-cov C bitwise; factored down_proj B
    allclose ~1e-6). The DEFAULT (int / inherited, no ``"auto"``) still returns
    ``1`` / inherited so the bs=1 golden stays byte-identical; nothing below
    raises bs and the auto path never runs.

    Auto-batch (v2 step 2 + step 3): when ``cov_batch_size=="auto"`` AND
    ``auto_batch.enabled`` (see ``_cov_is_auto``), ``_collect_covariances`` probes
    VRAM with the G window resident, auto-sizes the cov forward batch, and
    OOM-backs-off to ``floor=1`` (this resolver supplies that floor). This runs on
    BOTH the in-process path and the DP worker path (v2 step 3): each DP replica
    auto-sizes against its OWN pinned-device VRAM, independently, because the
    per-sequence pin makes the key-wise DP reduce batch-independent — no
    cross-replica min-agreement (supersedes spec §6). ``_resolve_cov_batch_size``
    still returns only the inherited int FLOOR in every case.

    Compound-peak constraint (plan M2): the A1×A4×A6 dense-teacher peak is
    ``G · cov_bs · seq · d_in · 4`` bytes per hot device (every G window layer
    holds a full ``[T, d_in]`` fp32 teacher tensor, T = cov_bs·seq). The auto
    sizing budgets THIS (not just the forward activation) because ``size_batch``
    probes the live ``max_memory_allocated`` with the G window already resident,
    so the probe baseline absorbs the G commitment; OOM-backoff to ``floor=1``
    is the safety net if the sized bs still over-commits the hot device.
    """
    inherited = int(s3.get("batch_size", 1))
    req = s3.get("cov_batch_size", inherited)
    if req is None:
        return inherited
    if isinstance(req, str):
        if req.strip().lower() == "auto":
            # ``"auto"`` resolves here to the inherited int FLOOR only — the
            # actual VRAM-measured sizing is NOT done in this resolver. It is
            # done inside ``_collect_covariances`` (``size_batch`` + the G-window
            # probe + OOM-backoff to this floor), on BOTH the in-process path and
            # each DP replica (v2 step 3). Every replica probes its OWN
            # pinned-device VRAM and sizes independently; the per-sequence pin
            # makes the key-wise DP reduce batch-independent, so there is no
            # cross-replica min(candidate) agreement (supersedes spec §6). This
            # floor is what the OOM-backoff falls back to (=1 by default), so the
            # hot device is never OOM'd even if the probe over-commits.
            return inherited
        try:
            return int(req)
        except ValueError:
            return inherited
    return int(req)


def _cov_is_auto(s3: dict) -> bool:
    """True iff the cov forward batch should be VRAM-auto-sized (v2 step 2).

    DOUBLE-gated (Review H1, mirrors ``ma_detection``): the auto path fires only
    when ``stage3_svd.cov_batch_size == "auto"`` AND
    ``AutoBatchConfig.from_dict(stage3_svd.auto_batch).enabled`` is true. Either
    condition absent → False → the original bs=``_resolve_cov_batch_size`` loop
    runs unchanged (no probe, no backoff wrapper) → stage3 golden byte-identical.
    ``_resolve_cov_batch_size`` itself is UNCHANGED and still returns a positive
    int floor (used as the backoff ``floor`` and the non-auto fallback).
    """
    return (
        s3.get("cov_batch_size") == "auto"
        and AutoBatchConfig.from_dict(s3.get("auto_batch")).enabled
    )


# ---------------------------------------------------------------------------
# Post-prune input covariance (for AA-SVD B matrix)
# ---------------------------------------------------------------------------


def _collect_covariances(
    model, moe_layers, batches, B_acc: InputCovarianceAccumulator, *, device,
    spill_dir=None,
    teacher_model=None,
    teacher_moe_layers=None,
    C_acc: InputCovarianceAccumulator | None = None,
    ccov_spill_dir=None,
    cov_window_size: int = 1,
    cov_capture_mode: str = "capture",
    cov_cross_impl: str = "dense",
    calib=None,
    cov_auto: bool = False,
    auto_batch_cfg: AutoBatchConfig | None = None,
) -> None:
    """Collect post-prune input covariance S and (optionally) cross-covariance C.

    **S-covariance** (always; code symbol ``B_acc`` IS paper-``S`` — see
    module-docstring Naming bridge): ``S = X_post^T X_post`` per
    (layer, expert, matrix), collected by hooking the pruned (student)
    model's expert inputs. ``InputCovarianceAccumulator`` redirects
    ``matrix_name="up_proj"`` to the ``gate_proj`` entry internally
    (auto-cov share, same hidden state pre-routing).

    **Cross-covariance** (when teacher_model provided): ``C = X_pre^T X_post``
    per ``(layer, student_expert)`` on **gate_proj inputs only**, collected by
    running both original (teacher) and pruned (student) models on the same
    calibration batch. The teacher's expert inputs give X_pre; the student's
    give X_post. C is accumulated as ``X_pre^T @ X_post`` per batch.
    ``up_proj`` cross-cov is served by ``_cov_lookup``'s gate->up fallback in
    ``aa_svd_factor.py``; ``down_proj`` falls back to Path 3 (B-only,
    Corollary 3.3) — see module docstring D6. This implements the
    MoE-resolved cross-cov required by AA-SVD Theorem 3.2 (paper 2604.02119).

    **Expert mapping challenge**: The teacher has 256 experts per layer; the
    student has ~180-200 (post Stage 2 merge). Expert indices don't correspond
    1:1. The cross-covariance is collected per (layer, student_expert) — for
    each student expert, we need the teacher's activation at the *same token
    positions* that the student routes to that expert.

    **Implementation (A7 + A1)**: covariance is captured from the model's REAL
    native forward via A7 ``capture_experts`` (a side-effect-free
    ``forward_pre_hook``), NOT the per-expert Python-loop ``instrument_experts``
    forward swap. A1 windows the MoE layers into contiguous groups of
    ``cov_window_size`` (G) layers and forwards the calibration set ONCE per
    window — ``ceil(N/G)`` passes instead of N. For each window, for each batch:
    1. Forward teacher → collect {(layer, token_idx) → X_pre} for all G window
       layers via the teacher capture hooks.
    2. Forward student → for each (layer, expert, token_idx) in the window, look
       up the corresponding X_pre and accumulate C += X_pre^T @ X_post.

    ``_teacher_hidden`` is keyed by layer and cleared per BATCH (not per layer),
    so it holds all G window layers' teacher rows during the student forward.
    The per-(layer,expert,matrix) accumulators are additive and order-stable
    per key, so windowing adds zero error on top of the native baseline (PLAN
    §1.2 / §2.1). ``cov_capture_mode="instrument"`` selects the legacy
    forward-swap path as a fallback.

    Since experts in teacher and student see different token subsets (routing
    differs), the cross-covariance captures the teacher's representation of
    the tokens that the *student* routes to each expert — exactly what
    Theorem 3.2 needs: "what would the original model have produced for the
    inputs that the compressed model actually receives."

    With ``spill_dir`` set, after each layer's finalize the layer's entries
    are written to disk and dropped from memory.

    .. note::

       A7+A1 is byte-identical to a NEW all-native golden, NOT the legacy
       Python-loop golden (which baked in ``instrument_experts``'s fp reduction
       order). ``cov_window_size=1`` (``G=1``) is the per-layer structure on the
       native forward — still the new golden. See
       ``tasks/PLAN_A7_A1_WINDOWED_COV.md`` and the D-A7 deviation entry.
    """
    if cov_capture_mode not in ("capture", "instrument"):
        raise ValueError(
            f"cov_capture_mode must be 'capture' (A7) or 'instrument' "
            f"(legacy fallback), got {cov_capture_mode!r}"
        )
    if cov_cross_impl not in ("dense", "dict"):
        raise ValueError(
            f"cov_cross_impl must be 'dense' (A4, default) or 'dict' "
            f"(legacy, retained only for the A4 equivalence test), "
            f"got {cov_cross_impl!r}"
        )
    if cov_auto and calib is None:
        raise ValueError(
            "_collect_covariances: cov_auto=True requires the calib tensor "
            "(the auto path re-slices iter_batches(calib, batch_size=cov_bs))"
        )
    # Backoff floor + sized batch (auto path). ``cov_floor`` is the proven-safe
    # bs=1 forward (also the non-auto golden setting); ``run_with_oom_backoff``
    # never drops below it and ``size_batch``'s ``fixed_batch`` is this floor.
    cov_floor = 1
    cov_bs: int | None = None  # lazily sized on the first non-skipped window

    # --- Storage for teacher's per-layer hidden states (for cross-cov) ---
    # A4 (default ``cov_cross_impl="dense"``): one dense ``[T, d_in]`` fp32
    # tensor per window layer (``teacher_dense``) plus a boolean ``[T]``
    # ``filled`` mask, scattered via ``index_copy_`` and consumed by a single
    # ``index_select`` in ``input_cb`` — replacing the per-token Python loop.
    # ``T = batch.shape[0] * batch.shape[1]`` (rebound per batch in the
    # enclosing batch loop and read by the dense-path closures as a free
    # variable — NOT a ``nonlocal`` declaration, and none is required). Legacy
    # ``cov_cross_impl="dict"`` (the ``{token_idx → row}`` nested dict) is
    # RETAINED unchanged, reachable ONLY through this kwarg, PURELY so the
    # A4 equivalence test (``test_a4_cross_cov_dense_equals_dict``) can compare
    # dense-vs-dict bit-for-bit — both ``capture`` and ``instrument`` route the
    # SAME closures, so there is no other legacy dict path to diff against.
    _teacher_hidden: dict[int, dict[int, torch.Tensor]] = {}
    # Dense-path per-layer state (A4): layer_idx → dense [T, d_in] / [T] mask.
    _teacher_dense: dict[int, torch.Tensor] = {}
    _teacher_filled: dict[int, torch.Tensor] = {}
    # Per-batch token count T (= rows * seq), set in the batch loop below so
    # the lazily-allocated dense tensors are sized correctly each batch.
    _teacher_T: int = 0
    # Per-batch sequence length (= batch.shape[1]), set in the batch loop below.
    # Read by the cov callbacks as a free variable to derive per-row source
    # sequence ids (``token_idx // _seq_len``) for the reduction-pin split.
    _seq_len: int = 0

    def _teacher_input_cb(li, e, tensor, ctx):
        """Teacher hook: store the full hidden state for this layer.
        We only need gate_proj input (= hidden state entering the MoE experts).
        Since all experts in a layer receive the same hidden state (pre-routing),
        we capture it once from any expert and key by (layer, token_positions).

        NOTE on B_acc.update's gate/up aliasing: that share applies to
        auto-covariance (same input on the student side). The cross-cov
        below is built per-expert against the teacher's hidden state and
        is written under matrix_name="gate_proj" only; ``up_proj``
        cross-cov is served at factorisation time by the
        ``_cov_lookup`` gate->up fallback in ``aa_svd_factor.py``.
        """
        # Store the raw hidden state indexed by token position.
        # The teacher routes tokens to different experts than the student,
        # but the *input* to the MoE layer (before routing) is the same for
        # all experts. We need to capture it per-token for cross-cov lookup.
        token_idx = ctx["token_idx"]
        key = li
        det = tensor.detach().to(torch.float32)
        if cov_cross_impl == "dict":
            if key not in _teacher_hidden:
                # Will be populated incrementally per expert dispatch
                _teacher_hidden[key] = {}
            for i, tidx in enumerate(token_idx.tolist()):
                _teacher_hidden[key][tidx] = det[i]
            return
        # A4 dense path: lazily allocate a dense [T, d_in] fp32 tensor (+ a
        # boolean [T] ``filled`` mask) for this layer on first dispatch this
        # batch, then scatter the dispatched rows in one ``index_copy_``.
        # ``token_idx`` from ``torch.where(mask[e])`` is UNIQUE within a single
        # expert dispatch (no repeated-index hazard for ``index_copy_``); across
        # experts the teacher's pre-routing layer-input row at a given position
        # is identical, so a re-write is value-preserving.
        if key not in _teacher_dense:
            d_in = det.shape[1]
            _teacher_dense[key] = torch.zeros(
                (_teacher_T, d_in), dtype=torch.float32, device=det.device
            )
            _teacher_filled[key] = torch.zeros(
                (_teacher_T,), dtype=torch.bool, device=det.device
            )
        tok = token_idx.to(det.device)
        _teacher_dense[key].index_copy_(0, tok, det)
        _teacher_filled[key][tok] = True

    def input_cb(li, e, tensor, ctx):
        # The student-side B accumulation uses the InputCovarianceAccumulator's
        # built-in up_proj→gate_proj alias on the auto-cov path (one entry
        # serves both, since gate/up share the pre-routing hidden state).
        # The cross-cov path below has no such alias inside the accumulator
        # — ``update_cross`` writes the exact ``matrix_name`` — so we key
        # the cross term under "gate_proj" here and rely on the
        # factor-time ``_cov_lookup`` gate→up fallback in
        # ``aa_svd_factor.py`` to serve ``up_proj``.
        #
        # Reduction-pin: split the captured rows by source sequence so the Gram
        # accumulates in sequence-ascending order, independent of the cov
        # forward batch size. ``input``-key ``tensor`` is ``hidden_states[tok]``
        # (unpadded, ``tensor.shape[0] == tok.shape[0]``), so the prefix slice
        # is a no-op here; it matters for the PADDED factored down_proj below.
        # ``update_grouped`` owns the single-vs-split decision (no pre-guard).
        tok = ctx.get("token_idx")
        if _seq_len and tok is not None:
            B_acc.update_grouped(
                li, e, "gate_proj", tensor[: tok.shape[0]], tok // _seq_len
            )
        else:
            B_acc.update(li, e, "gate_proj", tensor)
        # Cross-covariance: C += X_pre^T @ X_post for matching token positions.
        # Cross-device safety: with ``device_map="auto"`` teacher and student
        # copies of the same MoE layer can land on different GPUs. ``det_post``
        # lives on the student tensor's device; the teacher rows were detached
        # on the teacher's device. Coerce onto ``tensor.device`` before the
        # ``X_pre.T @ X_post`` matmul (and the in-place add inside
        # ``update_cross``) so the op is single-device. The ``.to()`` is a
        # no-op when devices already match (single-GPU) and a cheap D2D copy
        # under sharding.
        tgt_device = tensor.device
        if cov_cross_impl == "dict":
            if C_acc is not None and li in _teacher_hidden:
                token_idx = ctx["token_idx"].tolist()
                teacher_store = _teacher_hidden[li]
                pre_vecs = []
                post_vecs = []
                matched_tids = []                    # seq-id source, 1:1 with vecs
                det_post = tensor.detach().to(torch.float32)
                for i, tidx in enumerate(token_idx):
                    if tidx in teacher_store:
                        pre_vecs.append(teacher_store[tidx].to(device=tgt_device))
                        post_vecs.append(det_post[i])
                        matched_tids.append(tidx)
                if pre_vecs:
                    X_pre = torch.stack(pre_vecs)    # [n_match, d_in]
                    X_post = torch.stack(post_vecs)  # [n_match, d_in]
                    # Reduction-pin (legacy dict path): split the matched
                    # operands by source sequence so the dict path stays
                    # byte-identical to the dense path (A4 equivalence) under the
                    # pin. ``matched_tids`` is 1:1 with the stacked rows; its seq
                    # ids are ``tid // _seq_len``. Ascending per-sequence
                    # accumulation = the bs=1 grouping.
                    sids = (
                        torch.tensor(matched_tids, dtype=torch.long) // _seq_len
                        if _seq_len else None
                    )
                    if sids is not None and torch.unique(sids).numel() > 1:
                        for s in torch.unique(sids, sorted=True).tolist():
                            m = (sids == s).to(tgt_device)
                            C_acc.update_cross(
                                li, e, "gate_proj",
                                X_pre[m].T @ X_post[m], n_tokens=int(m.sum().item()),
                            )
                    else:
                        C_acc.update_cross(
                            li, e, "gate_proj", X_pre.T @ X_post,
                            n_tokens=len(pre_vecs),
                        )
            return
        # A4 dense path: a single ``index_select`` over the dense teacher
        # tensor reproduces the dict's in-order matched-position gather. The
        # ``filled`` mask reproduces the legacy ``if tidx in teacher_store``
        # skip exactly; boolean masking is order-preserving and
        # ``index_select(0, sel_idx)`` gathers in ``sel_idx`` order, so the
        # rows align row-for-row with the old in-order Python loop ⇒ the
        # ``X_pre.T @ X_post`` GEMM is bit-identical to the dict path.
        if C_acc is not None and li in _teacher_dense:
            token_idx = ctx["token_idx"]
            filled = _teacher_filled[li]
            tok = token_idx.to(filled.device)
            keep = filled[tok]
            sel_idx = tok[keep]
            if sel_idx.numel() > 0:
                det_post = tensor.detach().to(torch.float32)
                X_pre = _teacher_dense[li].index_select(
                    0, sel_idx.to(_teacher_dense[li].device)
                ).to(tgt_device)
                X_post = det_post[keep.to(det_post.device)]
                # Reduction-pin (cross-cov). ``update_cross`` receives an
                # already-formed product and cannot be split internally, so we
                # split the pre-matmul OPERANDS here. CRITICAL: ``X_pre``/
                # ``X_post`` are the ``keep``-filtered rows (``keep.sum()`` rows,
                # 1:1 with ``sel_idx = tok[keep]``), NOT ``len(tok)`` rows — so
                # the per-row sequence ids MUST come from ``sel_idx // _seq_len``
                # (the kept-row identities), aligned with the operand rows.
                # Ascending per-sequence accumulation reproduces the bs=1 Gram.
                n_kept = int(keep.sum().item())
                # ``sel_idx`` lives on ``filled.device``; the operands live on
                # ``tgt_device`` — move the per-seq boolean mask to the operand
                # device before indexing (no-op single-GPU, cheap D2D sharded).
                sids = sel_idx // _seq_len if _seq_len else None
                if sids is not None and torch.unique(sids).numel() > 1:
                    for s in torch.unique(sids, sorted=True).tolist():
                        m = (sids == s).to(tgt_device)
                        C_acc.update_cross(
                            li, e, "gate_proj",
                            X_pre[m].T @ X_post[m], n_tokens=int(m.sum().item()),
                        )
                else:
                    # bs=1 / single-sequence: UNCHANGED single product + count.
                    # Public entry: holds C_acc._lock around _pending writes and
                    # routes through finalize_layer's storage_dtype cast. The
                    # token count feeds persisted ``_gpu_token_count`` metadata
                    # (not the Gram) but must stay correct (review L1).
                    C_acc.update_cross(
                        li, e, "gate_proj", X_pre.T @ X_post,
                        n_tokens=n_kept,
                    )

    def intermediate_cb(li, e, tensor, ctx):
        # Reduction-pin (down_proj). CRITICAL: for FactoredExperts ``tensor`` is
        # the PADDED ``inter_padded[i]`` (``[max_tokens, d_int]`` with
        # ``len(tok) <= max_tokens``); the trailing pad rows are zero
        # (silu(0)*0 = 0) and contribute nothing to the Gram. Split on the
        # UNPADDED prefix ``tensor[:tok.shape[0]]`` so the per-sequence row sets
        # are correct — dropping the zero pad rows is byte-safe. The fused
        # (non-factored) path passes an unpadded tensor, where the prefix slice
        # is a no-op. ``update_grouped`` owns the single-vs-split decision.
        #
        # ACCURACY NOTE (down_proj only): the pin makes the *reduction grouping*
        # cov-batch-size-invariant, but UNLIKE gate_proj/cross-cov the down_proj
        # operand here (``act_fn(gate)*up``, ``inter_padded[i]``) is produced by
        # a PADDED batched ``bmm`` in ``capture_experts`` whose fp reduction is
        # perturbed by the forward batch SHAPE upstream of this callback. So the
        # down_proj Gram is allclose (~1e-6), NOT bitwise, across cov_batch_size
        # — the pin removes the reduction-order drift; the residual is the
        # unavoidable upstream forward-activation drift (same as v1). gate_proj B
        # and all cross-cov C keys (pure-gather operands) ARE bitwise-invariant.
        # (test_a6_cov_batch_size_close pins both behaviours.)
        tok = ctx.get("token_idx")
        if _seq_len and tok is not None:
            B_acc.update_grouped(
                li, e, "down_proj", tensor[: tok.shape[0]], tok // _seq_len
            )
        else:
            B_acc.update(li, e, "down_proj", tensor)
        # Cross-covariance for down_proj: teacher's intermediate → student's intermediate.
        # This requires hooking teacher's intermediate too — more complex.
        # For now, cross-cov is collected only for gate_up (input-side).
        # down_proj cross-cov would need teacher's act_fn(gate)*up output per expert,
        # which requires full teacher expert dispatch instrumentation.
        # The B-only Corollary 3.3 fallback handles down_proj adequately.

    from concurrent.futures import ThreadPoolExecutor
    spill_executor: ThreadPoolExecutor | None = None
    spill_futures: list = []
    if spill_dir is not None:
        spill_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bcov-spill",
        )

    # A1: window the MoE layers into contiguous groups of G layers and forward
    # the calibration set ONCE per window (ceil(N/G) passes instead of N). Each
    # window hooks all its layers with A7 capture_experts (capture-only — the
    # native forward runs untouched, so upstream layers stay native and the
    # residual stream is the real inference stream; PLAN §2.1/§4.1). The
    # per-(layer,expert,matrix) accumulators are additive + order-stable per key
    # (PLAN §1.2), so windowing adds zero error on top of the native baseline.
    # ``cov_window_size=1`` reproduces the per-layer structure on the native
    # forward (the NEW golden, not the legacy Python-loop golden).
    import contextlib
    n = len(moe_layers)
    G = max(1, int(cov_window_size))
    indexed = list(enumerate(moe_layers))  # [(k, ref), ...]
    done_count = 0
    try:
        for window in _iter_windows(indexed, G):
            # Resume: drop window layers already fully spilled (both B + C, when
            # cross-cov spill is configured). A layer NOT in ``to_collect`` is
            # neither hooked nor finalized — identical resume semantics to the
            # old per-layer skip, just at window granularity.
            to_collect = []
            for k, ref in window:
                if spill_dir is not None:
                    b_spilled = (spill_dir / f"layer_{ref.layer_idx}.pt").exists()
                    if ccov_spill_dir is None:
                        c_spilled = True
                    else:
                        c_spilled = (ccov_spill_dir / f"layer_{ref.layer_idx}.pt").exists()
                    if b_spilled and c_spilled:
                        log.info("Stage 3 cov layer %d/%d (idx=%d) — already spilled, skipping",
                                 k + 1, n, ref.layer_idx)
                        done_count += 1
                        continue
                to_collect.append((k, ref))
            if not to_collect:
                continue

            idxs = [ref.layer_idx for _, ref in to_collect]
            log.info("Stage 3 cov window [%s] (%d layer(s)) — %s calibration pass (G=%d)",
                     ",".join(str(i) for i in idxs), len(to_collect),
                     "dual-forward" if teacher_model is not None else "B-cov only", G)

            # Hook every layer in the window on BOTH models (student + teacher).
            stack = contextlib.ExitStack()
            for _k, ref in to_collect:
                if cov_capture_mode == "instrument":
                    stack.enter_context(
                        instrument_experts(
                            ref, {"input": input_cb, "intermediate": intermediate_cb}
                        )
                    )
                else:
                    stack.enter_context(
                        capture_experts(
                            ref, {"input": input_cb, "intermediate": intermediate_cb}
                        )
                    )
                if teacher_model is not None and teacher_moe_layers is not None:
                    teacher_ref = teacher_moe_layers[_k]
                    assert teacher_ref.layer_idx == ref.layer_idx, \
                        f"Teacher/student layer index mismatch: {teacher_ref.layer_idx} vs {ref.layer_idx}"
                    if cov_capture_mode == "instrument":
                        stack.enter_context(
                            instrument_experts(teacher_ref, {"input": _teacher_input_cb})
                        )
                    else:
                        stack.enter_context(
                            capture_experts(
                                teacher_ref, {"input": _teacher_input_cb},
                                capture_intermediate=False,
                            )
                        )

            if not cov_auto:
                # ---- NON-AUTO PATH (default): the original loop, VERBATIM ----
                # No nonlocal / probe / backoff wrapper. ``_teacher_T`` /
                # ``_seq_len`` are assigned at the ``_collect_covariances`` scope
                # here, so the cov callbacks read them correctly as free vars.
                # This branch is provably byte-identical to the pre-wiring code.
                with stack:
                    for batch_idx, batch in enumerate(batches):
                        if device is not None:
                            batch = batch.to(device)
                        # Clear per-BATCH (not per-layer): the teacher stores hold
                        # all G window layers' teacher rows for this batch's
                        # student forward, then are dropped before the next batch
                        # (PLAN §4.2). ``_teacher_T`` = rows*seq is threaded into
                        # the dense-path closures so each layer's lazily-allocated
                        # ``[T, d_in]`` tensor is sized for THIS batch.
                        _teacher_hidden.clear()
                        _teacher_dense.clear()
                        _teacher_filled.clear()
                        _teacher_T = int(batch.shape[0]) * int(batch.shape[1])
                        # Cov reduction-pin seam: the per-sequence length for THIS
                        # batch. ``iter_batches`` slices rows only, so ``seq_len``
                        # is uniform across the batch. The cov callbacks (closures)
                        # read this free var to split each expert's captured rows
                        # by source sequence (``seq_id = token_idx // _seq_len``)
                        # and accumulate the Gram in sequence-ascending order,
                        # making the accumulation independent of the cov forward
                        # batch size. At ``cov_batch_size=1`` (the golden's
                        # setting) every captured row shares one seq id → the split
                        # is a no-op → byte-identical.
                        _seq_len = int(batch.shape[1])
                        if teacher_model is not None:
                            with torch.no_grad():
                                teacher_model(input_ids=batch)
                        with torch.no_grad():
                            model(input_ids=batch)
            else:
                # ---- AUTO PATH (cov_batch_size:"auto" + auto_batch.enabled) ----
                # VRAM-auto-size the cov forward batch with the G window resident
                # (so ``CudaMemProbe.allocated()`` baseline absorbs the G
                # commitment), then run the dual-forward block under OOM-backoff.
                # Per-window helpers used by BOTH the probe and the backoff run.
                def _clear_teacher():
                    _teacher_hidden.clear()
                    _teacher_dense.clear()
                    _teacher_filled.clear()

                def _discard_window():
                    # Idempotent reset of this window's in-flight Gram so an
                    # aborted attempt (OOM) or the cost probe never double-counts.
                    for _dk, _dref in to_collect:
                        B_acc.discard_layer(_dref.layer_idx)
                        if C_acc is not None:          # None when cross-cov off
                            C_acc.discard_layer(_dref.layer_idx)

                with stack:
                    if cov_bs is None:
                        # Probe ONCE, on the first non-skipped window, with the G
                        # hooks installed (G resident). Cache for all windows (G
                        # constant → per-forward peak window-independent).
                        def cost_probe_fn(mb):
                            # CRITICAL (Review C1): the cov callbacks read
                            # ``_teacher_T`` / ``_seq_len`` as free vars of
                            # ``_collect_covariances``; without ``nonlocal`` a
                            # nested assignment makes them locals → callbacks see
                            # ``_seq_len == 0`` → the per-sequence PIN silently
                            # dies (un-pinned ``update``, cross-cov ``// 0``).
                            nonlocal _teacher_T, _seq_len
                            _discard_window(); _clear_teacher()
                            torch.cuda.reset_peak_memory_stats(device)
                            b = calib[:mb]
                            if device is not None:
                                b = b.to(device)
                            _teacher_T = int(b.shape[0]) * int(b.shape[1])
                            _seq_len = int(b.shape[1])
                            with torch.no_grad():
                                if teacher_model is not None:
                                    teacher_model(input_ids=b)
                                model(input_ids=b)
                            return int(torch.cuda.max_memory_allocated(device))

                        # Clean baseline BEFORE size_batch (no in-flight Gram).
                        _discard_window(); _clear_teacher()
                        ab = auto_batch_cfg or AutoBatchConfig()
                        cov_bs = size_batch(
                            cost_probe_fn, cov_floor,
                            headroom_frac=ab.headroom_frac,
                            max_cap=_COV_MAX_CAP,
                            mem=CudaMemProbe(device),
                        )
                        # The probe's Gram must NOT contaminate the real run.
                        _discard_window(); _clear_teacher()
                        log.info(
                            "Stage 3 cov auto-batch: sized cov_bs=%d (floor=%d, "
                            "max_cap=%d)", cov_bs, cov_floor, _COV_MAX_CAP,
                        )

                    def run_window_forwards(bs):
                        # CRITICAL (Review C1): callbacks read these as free vars.
                        nonlocal _teacher_T, _seq_len
                        # Idempotent: reset any aborted attempt's in-flight Gram
                        # before (re-)running, so OOM-retry never double-counts.
                        _discard_window()
                        for batch in iter_batches(calib, batch_size=bs):
                            if device is not None:
                                batch = batch.to(device)
                            _clear_teacher()
                            _teacher_T = int(batch.shape[0]) * int(batch.shape[1])
                            _seq_len = int(batch.shape[1])
                            with torch.no_grad():
                                if teacher_model is not None:
                                    teacher_model(input_ids=batch)
                                model(input_ids=batch)

                    run_with_oom_backoff(
                        run_window_forwards, start_batch=cov_bs, floor=cov_floor,
                    )

            # Finalize + spill every collected layer in the window.
            for _k, ref in to_collect:
                B_acc.finalize_layer(ref.layer_idx)
                if C_acc is not None:
                    C_acc.finalize_layer(ref.layer_idx)

                if spill_executor is not None:
                    _drain_done_futures(spill_futures)
                    fut = spill_executor.submit(
                        B_acc.spill_layer_to_disk, ref.layer_idx, spill_dir,
                    )
                    spill_futures.append(fut)
                if C_acc is not None and ccov_spill_dir is not None:
                    if spill_executor is not None:
                        fut_c = spill_executor.submit(
                            C_acc.spill_layer_to_disk, ref.layer_idx, ccov_spill_dir,
                        )
                        spill_futures.append(fut_c)

                done_count += 1
                proc_rss = _proc_rss_gb()
                maxrss = _maxrss_gb()
                host_ram = None
                try:
                    import psutil
                    host_ram = psutil.virtual_memory().used / 1e9
                except Exception:                            # noqa: BLE001
                    pass
                log.info(
                    "  Stage 3 cov layer %d/%d done — proc_rss=%sGB maxrss=%sGB host_ram=%sGB",
                    done_count, n, _fmt(proc_rss), _fmt(maxrss), _fmt(host_ram),
                )
                _trackio_log({
                    "stage3/bcov_layer": done_count,
                    "stage3/bcov_layer_idx": ref.layer_idx,
                    "stage3/bcov_proc_rss_gb": proc_rss if proc_rss is not None else float("nan"),
                    "stage3/bcov_maxrss_gb": maxrss if maxrss is not None else float("nan"),
                    "stage3/bcov_ram_used_gb": host_ram if host_ram is not None else float("nan"),
                })
    finally:
        if spill_executor is not None:
            log.info("Waiting for %d background spill(s) to flush before factor phase",
                     sum(1 for f in spill_futures if not f.done()))
            for f in spill_futures:
                f.result()
            spill_executor.shutdown(wait=True)
            log.info("All cov layer spills durable on disk.")


# Public alias for tests that import the B-only covariance collection path.
_collect_pruned_input_covariance = _collect_covariances


# ---------------------------------------------------------------------------
# Lever 2 (data-parallel) — replica spawn driver
# ---------------------------------------------------------------------------


def _shard_calib(calib, replicas: int) -> list:
    """Split the calibration tensor along dim 0 into ``replicas`` contiguous,
    disjoint shards (last shard takes the remainder). Token-disjoint shards are
    what make the cross-replica Gram sum exact (B7): each replica owns its own
    ``token_idx`` space; we never share ``_teacher_hidden`` across replicas, only
    sum the final per-(layer,expert) Gram matrices.
    """
    n = calib.size(0)
    if replicas <= 1 or n == 0:
        return [calib]
    replicas = min(replicas, n)
    base = n // replicas
    shards = []
    start = 0
    for r in range(replicas):
        # Last replica absorbs the remainder so every sequence is covered once.
        end = n if r == replicas - 1 else start + base
        shards.append(calib[start:end])
        start = end
    return shards


def _cov_replica_worker(
    replica_idx: int,
    visible_devices: str,
    config: dict,
    artifacts_dir,
    student_path: str,
    shard_start: int,
    shard_end: int,
    bcov_replica_dir: str,
    ccov_replica_dir,
    cross_cov_enabled: bool,
    bcov_storage_dtype: str,
    cov_num_sequences_override: int | None = None,
) -> None:
    """Spawn target: one data-parallel replica. Pins itself to its GPU subset
    via ``CUDA_VISIBLE_DEVICES``, reloads teacher+student, runs
    :func:`_collect_covariances` over its contiguous calibration shard, and
    spills per-layer covariance to its own replica subdir. The parent reduces
    the replica subdirs key-wise via :func:`_reduce_spilled_cov_dirs`.

    Module-level (picklable) so it is a valid ``torch.multiprocessing`` spawn
    target. Re-imports inside so the child has a clean import graph.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    import torch as _torch
    from pathlib import Path as _Path
    from ...utils.calibration import (
        build_calibration_tensor as _bct,
        spec_from_config as _spec_from_config,
        iter_batches as _iter_batches,
    )
    from ...utils.model_io import (
        load_model as _load_model,
        load_compressed_model as _load_compressed_model,
        iter_moe_layers as _iter_moe_layers,
    )

    artifacts_dir = _Path(artifacts_dir)
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    B_dtype = getattr(_torch, bcov_storage_dtype)

    # Rebuild the SAME calibration tensor, then slice this replica's shard.
    # MUST thread cov_num_sequences_override so the replica's rebuilt spec has
    # the SAME num_sequences as the parent's _resolve_bcov_spec-built calib;
    # otherwise the parent's shard bounds (computed against the override-length
    # tensor) slice a differently-sized worker calib → corrupted reduction.
    cal = config["calibration"]
    spec = _spec_from_config(
        cal, seed_offset=2, num_sequences_override=cov_num_sequences_override
    )
    student, tokenizer, _ = _load_compressed_model(
        student_path,
        device_map=config["model"]["device_map"],
        torch_dtype=config["model"]["torch_dtype"],
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
    )
    calib = _bct(tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache")
    shard = calib[shard_start:shard_end]
    # A6: DP replica reads the SAME cov-specific key as the in-process path so
    # the two agree (cross-replica Gram sum is bs-independent — finalized
    # per-key Grams are summed). ``config["stage3_svd"]`` is this site's own
    # local dict (review L2); ``_resolve_cov_batch_size`` returns the inherited
    # int FLOOR (golden untouched) — the actual per-replica VRAM auto-sizing,
    # when ``cov_batch_size:"auto"``, happens inside ``_collect_covariances``
    # (``size_batch``), gated by ``cov_auto=_cov_is_auto(...)`` at the call site
    # below. Each replica probes its OWN pinned-device VRAM and sizes
    # independently; the per-sequence pin makes the key-wise DP reduce
    # batch-independent, so NO cross-replica min(candidate) agreement is needed.
    batch_size = _resolve_cov_batch_size(config["stage3_svd"])
    batches = _iter_batches(shard, batch_size=batch_size)

    moe_layers = list(_iter_moe_layers(student))

    # A1 VRAM auto-sizing runs PER REPLICA, AFTER the CUDA_VISIBLE_DEVICES pin
    # above (each replica probes only its own GPU subset; PLAN §4.4). Seed
    # d_hidden/d_intermediate from the student config so the per-layer Gram
    # estimate is accurate.
    _cfg_for_window = dict(config)
    try:
        _scfg = getattr(student, "config", None)
        _tcfg = getattr(_scfg, "text_config", _scfg)
        if _tcfg is not None:
            _cfg_for_window["_d_hidden"] = int(getattr(_tcfg, "hidden_size", 0)) or None
            _cfg_for_window["_d_intermediate"] = int(
                getattr(_tcfg, "moe_intermediate_size", 0)
            ) or None
            _cfg_for_window["_n_experts"] = int(getattr(_tcfg, "num_experts", 0)) or None
    except Exception:                                # noqa: BLE001
        pass
    cov_window_size = _resolve_cov_window(_cfg_for_window, len(moe_layers))

    # Single-pass (G=N) engages the CPU hot-accumulator on the worker's OWN
    # collection accumulators — same detection the orchestrator uses for its
    # parent-side handles (shared _resolve_single_pass). Without this, a
    # cov_single_pass + cov_replicas>1 run accumulates all N layers' Grams
    # GPU-resident inside the worker → the exact OOM Task B prevents. Default
    # (no single-pass) → False → no migration → byte-identical.
    single_pass = _resolve_single_pass(config)

    teacher_model = None
    teacher_moe_layers = None
    C_acc = None
    B_acc = InputCovarianceAccumulator()
    B_acc.set_storage_dtype(B_dtype)
    if single_pass:
        B_acc.set_hot_accumulator_device("cpu")
    if cross_cov_enabled:
        teacher_model, _ = _load_model(
            config["model"]["name_or_path"],
            revision=config["model"]["revision"],
            torch_dtype=config["model"]["torch_dtype"],
            device_map=config["model"]["device_map"],
            attn_implementation=config["model"]["attn_implementation"],
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_moe_layers = list(_iter_moe_layers(teacher_model))
        C_acc = InputCovarianceAccumulator()
        C_acc.set_storage_dtype(B_dtype)
        if single_pass:
            C_acc.set_hot_accumulator_device("cpu")

    _Path(bcov_replica_dir).mkdir(parents=True, exist_ok=True)
    if ccov_replica_dir is not None:
        _Path(ccov_replica_dir).mkdir(parents=True, exist_ok=True)

    _collect_covariances(
        student, moe_layers, batches, B_acc, device=device,
        spill_dir=_Path(bcov_replica_dir),
        teacher_model=teacher_model,
        teacher_moe_layers=teacher_moe_layers,
        C_acc=C_acc,
        ccov_spill_dir=_Path(ccov_replica_dir) if ccov_replica_dir is not None else None,
        cov_window_size=cov_window_size,
        # v2 step 3: this replica AUTO-sizes its cov forward batch when
        # ``cov_batch_size:"auto"`` + ``auto_batch.enabled`` (double-gated via
        # ``_cov_is_auto``). The probe reads THIS replica's pinned-device VRAM
        # (CUDA_VISIBLE_DEVICES is set above), so each replica sizes to its OWN
        # budget — NO cross-replica min(candidate) agreement (supersedes spec
        # §6). The per-sequence reduction pin makes each replica's finalized
        # per-key Gram independent of its forward batch, so the key-wise DP
        # reduce (``_reduce_spilled_cov_dirs``, a fp32 sum of finalized Grams)
        # is batch-independent: gate_proj/up B + cross-cov C bitwise, factored
        # down_proj B allclose ~1e-6 (bounded, N-independent fwd drift — the
        # single-GPU property). ``calib=shard`` so the auto path re-slices THIS
        # replica's shard. Default (no "auto") → ``cov_auto`` False → inherited
        # bs=1 → byte-identical DP reduce (golden/A4 untouched).
        calib=shard,
        cov_auto=_cov_is_auto(config["stage3_svd"]),
        auto_batch_cfg=AutoBatchConfig.from_dict(
            config["stage3_svd"].get("auto_batch")
        ),
    )


def run_dp_covariance_collection(
    *,
    config: dict,
    artifacts_dir,
    student_path: str,
    calib,
    replicas: int,
    shards_per_model: int,
    cross_cov_enabled: bool,
    bcov_spill_dir,
    ccov_spill_dir,
    bcov_storage_dtype: str = "bfloat16",
) -> None:
    """Data-parallel Stage-3 covariance collection (lever 2).

    Fan out ``replicas`` child processes (torch.multiprocessing spawn), each
    pinned to ``shards_per_model`` GPUs via ``CUDA_VISIBLE_DEVICES`` and
    processing a disjoint contiguous calibration shard into its own per-replica
    spill subdir; then key-wise sum the subdirs into the canonical
    ``bcov_spill_dir`` / ``ccov_spill_dir`` so the factor phase reads them
    exactly like a single-pass (or resume) run.
    """
    from pathlib import Path as _Path
    import torch as _torch
    import torch.multiprocessing as _mp

    artifacts_dir = _Path(artifacts_dir)
    shards = _shard_calib(calib, replicas)
    replicas = len(shards)  # may be clamped by _shard_calib

    bcov_spill_dir = _Path(bcov_spill_dir)
    ccov_spill_dir = _Path(ccov_spill_dir) if ccov_spill_dir is not None else None
    store_dtype = getattr(_torch, bcov_storage_dtype)

    # Mirror _resolve_bcov_spec's read of stage3_svd.cov_num_sequences so every
    # replica builds the SAME spec as the in-process path. None → no override.
    _s3 = config.get("stage3_svd") or {}
    _cov_num_seq = _s3.get("cov_num_sequences")
    _cov_num_seq_override = int(_cov_num_seq) if _cov_num_seq is not None else None

    bcov_replica_dirs = []
    ccov_replica_dirs = []
    spawn_args = []
    start = 0
    for r in range(replicas):
        end = start + shards[r].size(0)
        # Each replica gets a contiguous, non-overlapping GPU subset.
        dev_lo = r * shards_per_model
        dev_hi = dev_lo + shards_per_model
        visible = ",".join(str(d) for d in range(dev_lo, dev_hi))
        b_dir = bcov_spill_dir / f"_replica_{r}"
        c_dir = (ccov_spill_dir / f"_replica_{r}") if ccov_spill_dir is not None else None
        bcov_replica_dirs.append(b_dir)
        if c_dir is not None:
            ccov_replica_dirs.append(c_dir)
        spawn_args.append(
            (r, visible, config, str(artifacts_dir), str(student_path),
             start, end, str(b_dir), (str(c_dir) if c_dir is not None else None),
             cross_cov_enabled, bcov_storage_dtype, _cov_num_seq_override)
        )
        start = end

    log.info("Stage 3 DP cov: spawning %d replica(s), %d GPU(s)/replica",
             replicas, shards_per_model)
    procs = []
    ctx = _mp.get_context("spawn")
    for args in spawn_args:
        p = ctx.Process(target=_cov_replica_worker, args=args)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Stage 3 DP cov: replica process exited with code {p.exitcode}"
            )

    log.info("Stage 3 DP cov: reducing %d replica spill dir(s) → canonical spill",
             replicas)
    _reduce_spilled_cov_dirs(bcov_replica_dirs, bcov_spill_dir, storage_dtype=store_dtype)
    if ccov_spill_dir is not None and ccov_replica_dirs:
        _reduce_spilled_cov_dirs(ccov_replica_dirs, ccov_spill_dir, storage_dtype=store_dtype)


def _load_stage2_covariance(path: Path):
    if not path.exists():
        log.warning("Stage 2 covariance not found at %s — AA-SVD fallback", path)
        return {}
    # S-2: validate the MANIFEST.json sidecar before loading the multi-GB
    # .pt. Stage 2 writes the manifest LAST, after the .pt's fsync, so a
    # torn .pt (mid-write SIGKILL) leaves NO manifest. Missing or
    # mismatched manifest = fail loudly + delete-and-re-run, NEVER
    # silently consume a partial file. Mirrors F-S3-1's
    # eora_inputs.py:199-243 contract.
    manifest_path = path.with_suffix(path.suffix + ".MANIFEST.json")
    if manifest_path.exists():
        from moe_compress.utils.atomic_io import (
            ManifestMismatchError,
            read_and_validate_manifest,
        )
        try:
            read_and_validate_manifest(
                path,
                manifest_path,
                expected_schema_version=1,
            )
        except ManifestMismatchError as exc:
            raise RuntimeError(
                f"Stage 3: Stage 2 covariance manifest validation FAILED — {exc}. "
                "This is the classic torn-write signature on a multi-GB "
                f"artifact. Delete both {path.name} and "
                f"{manifest_path.name} from {path.parent} and re-run Stage 2."
            ) from exc
    else:
        # MEDIUM-S2 TODO(post-2026-Q3): remove this backward-compat shim
        # once all in-flight runs that produced pre-S-2 .pt files are
        # regenerated under the new writer. Mirrors MEDIUM-8 in
        # eora_inputs.py:230-236. The fallback exists because pre-S-2
        # Stage 2 writers produced .pt files without sibling manifests;
        # once those in-flight runs complete, ALL Stage 2 writers emit a
        # manifest and the missing-manifest branch becomes
        # dead-code-loud-fail territory.
        log.warning(
            "Stage 3: %s has no MANIFEST.json sibling (pre-S-2 Stage 2 "
            "writer?). Proceeding without manifest validation; if "
            "torch.load errors below, the .pt may be torn — delete it "
            "and re-run Stage 2.",
            path,
        )
    payload = torch.load(path, map_location="cpu")
    return payload.get("covariance", {})


class CovarianceCollectionPlugin:
    """Stage 3 covariance-collection plugin (live in the orchestrator sequencer).

    Owns the post-prune covariance-collection phase: S-covariance
    ``S = X_post^T X_post`` (always; code symbol ``B_acc`` IS paper-``S``)
    and, when a teacher model is supplied, the
    AA-SVD cross-covariance ``C = X_pre^T X_post`` (Theorem 3.2, paper
    2604.02119). The phase logic lives in the module-level
    ``_collect_covariances``. The Stage 3 orchestrator (``stage3/orchestrator.py``)
    dispatches ``collect_covariances`` as the first phase hook; the legacy
    "S3-2 INERT / S3-7 wires it in" milestone labels are naming-historical
    (the wiring landed; this docstring is the post-wiring snapshot).
    """

    name = "covariance_collection"
    paper = (
        "AA-SVD Theorem 3.2 cross-covariance + Corollary 3.3 — "
        "arXiv:2604.02119 (atulkumarin/AA-SVD @ "
        "1fa1b686cd9b13a77607a676564e37d438a176c8). "
        "Live factor paths: Path 1 (W·C·S⁻¹·L_B^T, default; code symbol "
        "``B_acc`` IS paper-S) and Path 3 (W·L_B^T, S-only fallback). "
        "Path 2 (auto-cov-for-cross-cov) was retired — see the "
        "Path-2-retirement comment block in ``_aa_svd`` of "
        "aa_svd_factor.py. "
        "Deviations: D6 (gate-only cross-cov, per-expert MoE resolution; "
        "down falls back to Corollary 3.3), D-cov-storage-fp16 (SHARED "
        "with Stage 2 — fp16 persisted, fp64 in-memory eigh), "
        "D-A7 (cov captured from the REAL native forward via capture_experts, "
        "was the instrument_experts Python loop; windowed G layers/pass — "
        "byte-identical to a NEW all-native golden, more inference-faithful; "
        "definitions unchanged. See tasks/PLAN_A7_A1_WINDOWED_COV.md). "
        "See module docstring."
    )
    config_key = "stage3_svd.aa_svd.cross_covariance"
    reads: tuple[str, ...] = (
        "model", "moe_layers", "batches", "B_acc", "device",
        "bcov_spill_dir", "teacher_model", "teacher_moe_layers",
        "C_acc", "ccov_spill_dir", "cov_window_size", "calib",
    )
    writes: tuple[str, ...] = ("B_acc", "C_acc")
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — covariance collection is UNCONDITIONAL.

        B-covariance is mandatory for every AA-SVD factorization, so this
        phase always runs. ``config_key`` gates only the *cross-covariance*
        branch (the optional teacher dual-forward), which is an internal
        decision inside ``_collect_covariances`` — it does not disable the
        plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def collect_covariances(self, ctx: PipelineContext) -> None:
        """Phase hook — covariance collection.

        Reads the calibration args off ``ctx`` and delegates to the
        module-level ``_collect_covariances``. The Stage 3 orchestrator
        invokes this hook in place of the legacy monolith's inline call;
        the (legacy) "S3-2 INERT / S3-7 wiring" milestone labels are
        naming-historical.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. Optional slots (params that default to None in
        # _collect_covariances) are has()-guarded so the B-only path (no
        # teacher, no C_acc, no spill) does not KeyError on an unset slot.
        _collect_covariances(
            ctx.get("model"),
            ctx.get("moe_layers"),
            ctx.get("batches"),
            ctx.get("B_acc"),
            device=ctx.get("device"),
            spill_dir=(
                ctx.get("bcov_spill_dir") if ctx.has("bcov_spill_dir") else None
            ),
            teacher_model=(
                ctx.get("teacher_model") if ctx.has("teacher_model") else None
            ),
            teacher_moe_layers=(
                ctx.get("teacher_moe_layers")
                if ctx.has("teacher_moe_layers")
                else None
            ),
            C_acc=ctx.get("C_acc") if ctx.has("C_acc") else None,
            ccov_spill_dir=(
                ctx.get("ccov_spill_dir") if ctx.has("ccov_spill_dir") else None
            ),
            cov_window_size=(
                ctx.get("cov_window_size") if ctx.has("cov_window_size") else 1
            ),
            # v2 step 2 (Review C2/H1): thread ``calib`` + the double-gated auto
            # flag so the auto path can re-slice ``iter_batches(calib, cov_bs)``.
            # ``calib`` is on run_ctx (orchestrator). The gate is DOUBLE: the
            # cov-specific ``cov_batch_size:"auto"`` AND ``auto_batch.enabled``.
            # Default (no "auto") → cov_auto False → original loop, no probe.
            calib=ctx.get("calib") if ctx.has("calib") else None,
            cov_auto=_cov_is_auto(
                ctx.get("config").get("stage3_svd", {})
                if ctx.has("config") else {}
            ),
            auto_batch_cfg=AutoBatchConfig.from_dict(
                ctx.get("config").get("stage3_svd", {}).get("auto_batch")
                if ctx.has("config") else None
            ),
        )
