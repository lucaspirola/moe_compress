"""Covariance collection (S3-2 of the Stage 3 plugin-architecture refactor).

Home of the post-prune covariance-collection logic relocated VERBATIM from
the legacy ``stage3_svd.py`` monolith:

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

``CovarianceCollectionPlugin`` is registered-but-INERT at S3-2 — no walk or
test invokes its ``collect_covariances`` hook. S3-7 wires it into the live
Stage 3 plugin sequencer.
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
    """Collect post-prune input covariance B and (optionally) cross-covariance C.

    **B-covariance** (always): ``B = X_post^T X_post`` per (layer, expert, matrix),
    collected by hooking the pruned (student) model's expert inputs.

    **Cross-covariance** (when teacher_model provided): ``C = X_pre^T X_post``
    per (layer, expert, matrix), collected by running both original (teacher)
    and pruned (student) models on the same calibration batch. The teacher's
    expert inputs give X_pre; the student's give X_post. C is accumulated as
    ``X_pre^T @ X_post`` per batch. This implements the exact covariance pair
    required by AA-SVD Theorem 3.2 (paper 2604.02119).

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
    # Key: layer_idx → Tensor [n_tokens_in_batch, d_in]
    _teacher_hidden: dict[int, torch.Tensor] = {}

    def _teacher_input_cb(li, e, tensor, ctx):
        """Teacher hook: store the full hidden state for this layer.
        We only need gate_proj input (= hidden state entering the MoE experts).
        Since all experts in a layer receive the same hidden state (pre-routing),
        we capture it once from any expert and key by (layer, token_positions)."""
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
        B_acc.update(li, e, "gate_proj", tensor)  # up_proj aliases to gate_proj
        # Cross-covariance: C += X_pre^T @ X_post for matching token positions.
        if C_acc is not None and li in _teacher_hidden:
            token_idx = ctx["token_idx"].tolist()
            teacher_store = _teacher_hidden[li]
            # Collect teacher activations for the same token positions
            pre_vecs = []
            post_vecs = []
            det_post = tensor.detach().to(torch.float32)
            for i, tidx in enumerate(token_idx):
                if tidx in teacher_store:
                    pre_vecs.append(teacher_store[tidx])
                    post_vecs.append(det_post[i])
            if pre_vecs:
                X_pre = torch.stack(pre_vecs)   # [n_match, d_in]
                X_post = torch.stack(post_vecs)  # [n_match, d_in]
                # Accumulate cross-covariance C = X_pre^T @ X_post
                cross = X_pre.T @ X_post  # [d_in, d_in]
                ckey = (li, e, "gate_proj")
                cur = C_acc._gpu.get(ckey)
                if cur is None:
                    C_acc._gpu[ckey] = cross.to(device=tensor.device)
                else:
                    cur.add_(cross.to(device=cur.device))
                C_acc._gpu_token_count[ckey] = C_acc._gpu_token_count.get(ckey, 0) + len(pre_vecs)

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
    # (sequential, not simultaneous). This differs from the spec §6 Phase A
    # which describes a single simultaneous pass over all 40 layers. The
    # sequential design was chosen because holding all 40 layers' hook state
    # simultaneously is memory-intensive; per-layer spill to disk bounds peak
    # RAM to ~one layer's covariance at a time (~5 GB). Wall-clock cost is
    # ~40× the simultaneous design, but GPU memory stays within H200 budget.
    # This deviation is documented in §12 (allowed deviation D9).
    n = len(moe_layers)
    try:
        for k, ref in enumerate(moe_layers):
            if spill_dir is not None:
                b_spilled = (spill_dir / f"layer_{ref.layer_idx}.pt").exists()
                c_spilled = (
                    ccov_spill_dir is None
                    or (ccov_spill_dir / f"layer_{ref.layer_idx}.pt").exists()
                )
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
    payload = torch.load(path, map_location="cpu")
    return payload.get("covariance", {})


class CovarianceCollectionPlugin:
    """Stage 3 covariance-collection plugin (S3-2 — registered-but-INERT).

    Owns the post-prune covariance-collection phase: B-covariance
    ``B = X_post^T X_post`` (always) and, when a teacher model is supplied, the
    AA-SVD cross-covariance ``C = X_pre^T X_post`` (Theorem 3.2, paper
    2604.02119). The phase logic lives in the module-level ``_collect_covariances``
    relocated verbatim from the monolith.

    S3-2 wires this class into the plugin registry as metadata only — no walk
    or test invokes ``collect_covariances``. S3-7 plugs the hook into the live
    Stage 3 plugin sequencer.
    """

    name = "covariance_collection"
    paper = "AA-SVD cross-covariance C = X_pre^T X_post (Theorem 3.2, paper 2604.02119)."
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
        """Phase hook — covariance collection (S3-7 wiring surface).

        INERT at S3-2: no orchestrator walk or test invokes this hook. S3-7
        replaces the Stage 3 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline
        ``_collect_covariances`` call. The body reads the calibration args off
        ``ctx`` and delegates to the relocated ``_collect_covariances``.
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
