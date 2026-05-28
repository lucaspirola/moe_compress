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

from ...utils.activation_hooks import InputCovarianceAccumulator, instrument_experts
from ...utils.futures import drain_done_futures as _drain_done_futures
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


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
# Post-prune input covariance (for AA-SVD B matrix)
# ---------------------------------------------------------------------------


def _collect_covariances(
    model, moe_layers, batches, B_acc: InputCovarianceAccumulator, *, device,
    spill_dir=None,
    teacher_model=None,
    teacher_moe_layers=None,
    C_acc: InputCovarianceAccumulator | None = None,
    ccov_spill_dir=None,
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

    **Implementation**: We hook ALL layers on BOTH models simultaneously.
    For each batch:
    1. Forward teacher → collect {(layer, token_idx) → X_pre} via hooks
    2. Forward student → for each (layer, expert, token_idx), look up the
       corresponding X_pre from the teacher's output and accumulate
       C += X_pre^T @ X_post for the same token positions.

    Since experts in teacher and student see different token subsets (routing
    differs), the cross-covariance captures the teacher's representation of
    the tokens that the *student* routes to each expert — exactly what
    Theorem 3.2 needs: "what would the original model have produced for the
    inputs that the compressed model actually receives."

    With ``spill_dir`` set, after each layer's finalize the layer's entries
    are written to disk and dropped from memory.
    """

    # --- Storage for teacher's per-layer hidden states (for cross-cov) ---
    # Structure: layer_idx → {token_idx → row Tensor [d_in]}. The nested
    # dict is populated incrementally by ``_teacher_input_cb`` (one entry
    # per token position the teacher dispatches through this MoE layer)
    # and consumed by ``input_cb`` via per-token lookup.
    _teacher_hidden: dict[int, dict[int, torch.Tensor]] = {}

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
        if key not in _teacher_hidden:
            # Will be populated incrementally per expert dispatch
            _teacher_hidden[key] = {}
        det = tensor.detach().to(torch.float32)
        for i, tidx in enumerate(token_idx.tolist()):
            _teacher_hidden[key][tidx] = det[i]

    def input_cb(li, e, tensor, ctx):
        # The student-side B accumulation uses the InputCovarianceAccumulator's
        # built-in up_proj→gate_proj alias on the auto-cov path (one entry
        # serves both, since gate/up share the pre-routing hidden state).
        # The cross-cov path below has no such alias inside the accumulator
        # — ``update_cross`` writes the exact ``matrix_name`` — so we key
        # the cross term under "gate_proj" here and rely on the
        # factor-time ``_cov_lookup`` gate→up fallback in
        # ``aa_svd_factor.py`` to serve ``up_proj``.
        B_acc.update(li, e, "gate_proj", tensor)
        # Cross-covariance: C += X_pre^T @ X_post for matching token positions.
        if C_acc is not None and li in _teacher_hidden:
            token_idx = ctx["token_idx"].tolist()
            teacher_store = _teacher_hidden[li]
            # Collect teacher activations for the same token positions.
            # PERF (MEDIUM-2): the per-token Python loop here and in
            # _teacher_input_cb is the dominant CPU cost of cross-cov
            # capture; a vectorised replacement that indexes the teacher
            # tensor with the student's token_idx (instead of building
            # a {tidx: row} dict and stacking row-by-row) would remove
            # ~256 small tensor builds per batch. Not correctness-critical.
            pre_vecs = []
            post_vecs = []
            det_post = tensor.detach().to(torch.float32)
            # Cross-device safety: with ``device_map="auto"`` teacher and
            # student copies of the same MoE layer can land on different
            # GPUs. ``det_post`` lives on the student tensor's device;
            # teacher rows in ``_teacher_hidden`` were detached on the
            # teacher's device. Coerce each teacher row onto
            # ``tensor.device`` before stacking so the X_pre.T @ X_post
            # matmul (and the in-place add inside ``update_cross``) is
            # single-device. The .to() is a no-op when devices already
            # match (common single-GPU case) and a cheap H2D/D2D copy
            # under sharding.
            tgt_device = tensor.device
            for i, tidx in enumerate(token_idx):
                if tidx in teacher_store:
                    pre_vecs.append(teacher_store[tidx].to(device=tgt_device))
                    post_vecs.append(det_post[i])
            if pre_vecs:
                X_pre = torch.stack(pre_vecs)   # [n_match, d_in]
                X_post = torch.stack(post_vecs)  # [n_match, d_in]
                # Cross term on the input device so the in-place add inside
                # update_cross stays on-device (matches B_acc.update's
                # contract; finalize_layer does the single GPU→CPU
                # transfer per key, applying ``storage_dtype``). The per-row
                # ``.to(tgt_device)`` above already pinned every X_pre vector
                # to ``tensor.device``, so the matmul output lives there by
                # construction — no trailing ``.to(tensor.device)`` needed.
                cross = X_pre.T @ X_post
                # Public entry: holds C_acc._lock around _pending writes
                # and routes through finalize_layer's storage_dtype cast
                # (D-cov-storage-fp16). Direct ``_gpu``/``_pending``
                # mutation here would (a) bypass the lock and (b) leave
                # GPU fp32 tensors alive past finalize.
                C_acc.update_cross(
                    li, e, "gate_proj", cross, n_tokens=len(pre_vecs),
                )

    def intermediate_cb(li, e, tensor, ctx):
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

    # NOTE: This function runs one full calibration pass PER MoE layer
    # (sequential, not simultaneous). The original spec described a single
    # simultaneous pass over all MoE layers; the sequential design was
    # chosen because holding all layers' hook state simultaneously is
    # memory-intensive — per-layer spill to disk bounds peak RAM to ~one
    # layer's covariance at a time (~5 GB). Wall-clock cost is ~N× the
    # simultaneous design (for N MoE layers), but GPU memory stays within
    # H200 budget. Documented as an allowed deviation in the project
    # deviation log (legacy label: D9; "Phase A" naming-historical).
    n = len(moe_layers)
    try:
        for k, ref in enumerate(moe_layers):
            if spill_dir is not None:
                b_spilled = (spill_dir / f"layer_{ref.layer_idx}.pt").exists()
                # When ccov_spill_dir is None (cross-cov disabled OR no
                # spill destination configured), there is no C-cov file
                # to wait on — treat it as satisfied so the B-only resume
                # path skips early instead of redoing the calibration pass.
                if ccov_spill_dir is None:
                    c_spilled = True
                else:
                    c_spilled = (ccov_spill_dir / f"layer_{ref.layer_idx}.pt").exists()
                if b_spilled and c_spilled:
                    log.info("Stage 3 cov layer %d/%d (idx=%d) — already spilled, skipping",
                             k + 1, n, ref.layer_idx)
                    continue
            log.info("Stage 3 cov layer %d/%d (idx=%d) — %s calibration pass",
                     k + 1, n, ref.layer_idx,
                     "dual-forward" if teacher_model is not None else "B-cov only")

            # Clear teacher hidden state storage for this layer.
            _teacher_hidden.clear()

            # Build context managers for instrumentation.
            import contextlib
            stack = contextlib.ExitStack()
            # Always hook the student (pruned model).
            stack.enter_context(
                instrument_experts(ref, {"input": input_cb, "intermediate": intermediate_cb})
            )
            # Optionally hook the teacher for cross-covariance.
            if teacher_model is not None and teacher_moe_layers is not None:
                # Find the matching teacher layer by index.
                teacher_ref = teacher_moe_layers[k]
                assert teacher_ref.layer_idx == ref.layer_idx, \
                    f"Teacher/student layer index mismatch: {teacher_ref.layer_idx} vs {ref.layer_idx}"
                stack.enter_context(
                    instrument_experts(teacher_ref, {"input": _teacher_input_cb})
                )

            with stack:
                for batch_idx, batch in enumerate(batches):
                    if device is not None:
                        batch = batch.to(device)
                    _teacher_hidden.clear()
                    # Forward teacher first (if present) to populate _teacher_hidden.
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_model(input_ids=batch)
                    # Forward student — hooks fire and accumulate B + C.
                    with torch.no_grad():
                        model(input_ids=batch)

            B_acc.finalize_layer(ref.layer_idx)
            if C_acc is not None:
                C_acc.finalize_layer(ref.layer_idx)

            # Background spill for B-cov.
            if spill_executor is not None:
                _drain_done_futures(spill_futures)
                fut = spill_executor.submit(
                    B_acc.spill_layer_to_disk, ref.layer_idx, spill_dir,
                )
                spill_futures.append(fut)
            # Spill cross-cov too.
            if C_acc is not None and ccov_spill_dir is not None:
                if spill_executor is not None:
                    fut_c = spill_executor.submit(
                        C_acc.spill_layer_to_disk, ref.layer_idx, ccov_spill_dir,
                    )
                    spill_futures.append(fut_c)

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
                k + 1, n, _fmt(proc_rss), _fmt(maxrss), _fmt(host_ram),
            )
            _trackio_log({
                "stage3/bcov_layer": k + 1,
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
        "with Stage 2 — fp16 persisted, fp64 in-memory eigh). "
        "See module docstring."
    )
    config_key = "stage3_svd.aa_svd.cross_covariance"
    reads: tuple[str, ...] = (
        "model", "moe_layers", "batches", "B_acc", "device",
        "bcov_spill_dir", "teacher_model", "teacher_moe_layers",
        "C_acc", "ccov_spill_dir",
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
        )
