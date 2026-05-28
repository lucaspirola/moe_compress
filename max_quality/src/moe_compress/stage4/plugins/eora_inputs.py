"""EoRA inputs loader — A-cov + Stage 3 originals + ranks snapshot.

Paper
-----
"EoRA: Eigenspace Low-Rank Approximation for Post-Training Compression
of LLMs" — arXiv:2410.21271. Algorithm 1 reads the input
auto-covariance ``A = X̃^T X̃`` (with ``X̃ ∈ ℝ^{N×d_in}`` the per-token
activation matrix, rows=tokens — so ``A ∈ ℝ^{d_in × d_in}`` is the
d_in × d_in right-side activation moment; see the shape check in
:func:`stage4.plugins.eora_compensation._compute_eora_factors` against
``(d_in, d_in)``). Convention source-of-truth:
:mod:`stage4.plugins.eora_compensation` (module docstring L38 and
``_compute_eora_factors`` Step 2) — keep this notation aligned with
that sibling plugin. Algorithm 1 also reads the factorization residual
``ΔW = W_orig − Ŵ`` (where ``Ŵ`` is the post-Stage-3 factored
expert), then computes the √Λ-scaled eigenspace projection of ΔW and
back-projects a rank-r correction.

This plugin owns the **inputs** half of the Stage 4 pipeline: load the
post-Stage-3 A-covariance (reusing the Stage 2 covariance sidecar
``_stage2_input_covariance.pt`` per D-drank-premerge-A — consumed at
:mod:`stage3.plugins.d_rank_allocate`), load the Stage 3 original
expert weights behind the file-deleted double-widen guard, build the
MoE-layer list, set up the crash-resume partial directory, and
snapshot the Stage-3 per-matrix ranks **directly from the in-memory
``FactoredExperts.ranks`` dict of each MoE layer** (no on-disk
``rank_map.json`` sidecar — ``rank_map`` is a separate run-scope slot
written downstream by
:mod:`stage4.plugins.eora_compensation`).

The compensation half (the per-(layer, expert) residual SVD and factor
widening) lives at :mod:`stage4.plugins.eora_compensation`.

Official code
-------------
``NVlabs/EoRA`` @ commit
``6a42e2edcc7559422d14ccf79b0105b2d8a78c76`` (2026-04-21) —
github.com/NVlabs/EoRA. Reference implementation for the
√Λ-eigenspace projection + correction.

Wiring
------
Live: registered and walked via :mod:`stage4.orchestrator` (imports
``EoraInputsPlugin``, registers it on the ``PluginRegistry``, and
walks ``load_eora_inputs`` as the first phase of the Stage 4 schedule).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ...utils.model_io import MATRIX_NAMES, FactoredExperts, iter_moe_layers
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


class EoraInputsPlugin:
    """Stage 4 EoRA input-load plugin.

    Owns the EoRA ``load_eora_inputs`` phase: loading the Stage-2 input
    activation covariance (``_stage2_input_covariance.pt``, plain-SVD
    fallback downstream when absent — see Step 2 short-circuit in
    :func:`stage4.plugins.eora_compensation._compute_eora_factors`),
    loading the Stage-3 original expert weights
    (``_stage3_original_weights.pt``) behind the file-deleted
    double-widen guard, building the MoE-layer list, setting up the
    crash-resume partial directory, and snapshotting the Stage-3
    per-matrix ranks (read from each layer's in-memory
    ``FactoredExperts.ranks`` dict) before any widening occurs.
    """

    name = "eora_inputs"
    paper = (
        "EoRA inputs loader (Algorithm 1 prereqs) — arXiv:2410.21271 "
        "(NVlabs/EoRA @ 6a42e2edcc7559422d14ccf79b0105b2d8a78c76). "
        "Loads post-Stage-3 A-cov + Stage 3 originals + ranks snapshot. "
        "See module docstring."
    )
    # ``config_key`` is informational only — ``is_enabled`` returns True
    # unconditionally (EoRA input load is mandatory for every Stage 4 run)
    # and this body never reads the key. The downstream per-matrix
    # compensation budget is parametrised by the same key in
    # :class:`stage4.plugins.eora_compensation.EoraCompensationPlugin`.
    config_key = "stage4_eora.compensation_budget_pct"
    reads: tuple[str, ...] = (
        "model", "config", "artifacts_dir",
    )
    writes: tuple[str, ...] = (
        "A_cov", "a_storage_dtype", "originals", "layers",
        "partial_dir", "stage3_ranks",
    )
    # Empty: EoraInputsPlugin needs no calibration pass — every Stage 4 input
    # (A-covariance, Stage-3 originals) is a precomputed on-disk artifact.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — the EoRA input load is UNCONDITIONAL.

        Every Stage 4 run must load the A-covariance and the Stage-3
        originals before any residual compensation can happen, so this phase
        always runs. ``config_key`` only parametrises the per-matrix
        compensation budget downstream; it never gates this plugin.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def load_eora_inputs(self, ctx: PipelineContext) -> None:
        """Phase hook — EoRA input load.

        Resolves ``a_storage_dtype`` from the Stage-2 config, loads the
        A-covariance (logging the plain-SVD-fallback notice when the sidecar
        is absent), loads the Stage-3 originals behind the file-deleted
        double-widen guard, builds the MoE-layer list, sets up the
        crash-resume partial directory, and snapshots the Stage-3 per-matrix
        ranks from each layer's in-memory ``FactoredExperts.ranks`` dict.
        Does NOT include the in-process per-matrix double-widen ``assert`` —
        that belongs to ``EoraCompensationPlugin.compensate_layer``.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. Optional slots are has()-guarded (get() raises
        # KeyError on an unset slot).
        artifacts_dir = ctx.get("artifacts_dir")
        config = ctx.get("config")
        model = ctx.get("model")
        no_resume = ctx.get("no_resume") if ctx.has("no_resume") else False

        # A-cov was persisted by Stage 2 in this storage dtype; the eigh
        # threshold in `_compute_eora_factors` must be tuned to that dtype's
        # quantization noise floor or it will keep noise-inflated directions.
        #
        # V2 cache short-circuit: when the calibration-v2 cache provider
        # (:class:`stage4.plugins.input_cov_cache.Stage4InputCovCacheProvider`)
        # already populated ``ctx.A_cov`` + ``ctx.a_storage_dtype``, skip
        # the on-disk ``_stage2_input_covariance.pt`` load entirely. The
        # cached dict has the same ``(layer_idx, expert_idx, matrix_name)``
        # → ``Tensor[d_in, d_in]`` shape so downstream ``_cov_lookup`` /
        # ``_compute_eora_factors`` work unchanged.
        if ctx.has("A_cov"):
            log.info("Stage 4: V2 input-cov cache HIT (%d keys) — skipping "
                     "_stage2_input_covariance.pt load",
                     len(ctx.get("A_cov")))
            # Compute a_storage_dtype only if the cache provider did not.
            if not ctx.has("a_storage_dtype"):
                s2 = config.get("stage2_reap_ream", {})
                ctx.set("a_storage_dtype", getattr(
                    torch, s2.get("covariance_storage_dtype", "float16"),
                ))
            A_cov = ctx.get("A_cov")
            a_storage_dtype = ctx.get("a_storage_dtype")
        else:
            s2 = config.get("stage2_reap_ream", {})
            a_storage_dtype = getattr(
                torch, s2.get("covariance_storage_dtype", "float16")
            )
            A_cov_path = artifacts_dir / "_stage2_input_covariance.pt"
            A_cov = {}
            if A_cov_path.exists():
                # S-2: validate the MANIFEST.json sidecar before loading
                # the multi-GB .pt. Stage 2 writes the manifest LAST,
                # after the .pt's fsync, so a torn .pt (mid-write
                # SIGKILL) leaves NO manifest sibling. Same contract as
                # the Stage 3 originals block below (lines 199-243).
                #
                # Stage 2 cov has no pre-rename legacy artifacts, so the
                # legacy-suffix back-compat block (analogous to lines
                # 179-184 for Stage 3 originals) is intentionally absent
                # here — Reader 2's new block has ONE manifest path.
                A_cov_manifest_path = A_cov_path.with_suffix(
                    A_cov_path.suffix + ".MANIFEST.json"
                )
                if A_cov_manifest_path.exists():
                    from moe_compress.utils.atomic_io import (
                        ManifestMismatchError,
                        read_and_validate_manifest,
                    )
                    try:
                        read_and_validate_manifest(
                            A_cov_path,
                            A_cov_manifest_path,
                            expected_schema_version=1,
                        )
                    except ManifestMismatchError as exc:
                        raise RuntimeError(
                            f"Stage 4: Stage 2 covariance manifest validation "
                            f"FAILED — {exc}. This is the classic torn-write "
                            f"signature on a multi-GB artifact. Delete both "
                            f"{A_cov_path.name} and "
                            f"{A_cov_manifest_path.name} from {artifacts_dir} "
                            "and re-run Stage 2."
                        ) from exc
                else:
                    # MEDIUM-S2 TODO(post-2026-Q3): remove this
                    # backward-compat shim once all in-flight runs that
                    # produced pre-S-2 .pt files are regenerated under
                    # the new writer. Mirrors MEDIUM-8 in lines 230-236
                    # below. The fallback exists because pre-S-2 Stage 2
                    # writers produced .pt files without sibling
                    # manifests; once those in-flight runs complete, ALL
                    # Stage 2 writers emit a manifest and the
                    # missing-manifest branch becomes
                    # dead-code-loud-fail territory.
                    log.warning(
                        "Stage 4: %s has no MANIFEST.json sibling "
                        "(pre-S-2 Stage 2 writer?). Proceeding without "
                        "manifest validation; if torch.load errors "
                        "below, the .pt may be torn — delete it and "
                        "re-run Stage 2.",
                        A_cov_path,
                    )
                A_cov = torch.load(A_cov_path, map_location="cpu").get("covariance", {})
            else:
                log.warning("Stage 4: no Stage 2 covariance at %s — plain-SVD fallback",
                            A_cov_path)

        originals_path = artifacts_dir / "_stage3_original_weights.pt"
        # LOW-5: manifest naming consistency with F-RK-1
        # (``.pt.MANIFEST.json`` — appended after the payload suffix). The
        # legacy ``_stage3_original_weights.MANIFEST.json`` is also
        # consulted for backward compat (in-flight runs that wrote the
        # manifest before this rename landed).
        originals_manifest_path = (
            artifacts_dir / "_stage3_original_weights.pt.MANIFEST.json"
        )
        legacy_originals_manifest_path = (
            artifacts_dir / "_stage3_original_weights.MANIFEST.json"
        )
        if not originals_manifest_path.exists() and legacy_originals_manifest_path.exists():
            # Honour the legacy manifest from a pre-LOW-5 Stage 3 run.
            originals_manifest_path = legacy_originals_manifest_path
        if not originals_path.exists():
            # If Stage 4 already completed, the originals are intentionally
            # deleted. Re-entering Stage 4 on an already-widened model is a
            # double-widen.
            if (artifacts_dir / "stage4_eora" / "eora_ranks.json").exists():
                raise AssertionError(
                    "Stage 4 double-widen detected: _stage3_original_weights.pt was deleted "
                    "after a prior successful Stage 4 run. "
                    "widen_rank() has already been applied to this model."
                )
            raise FileNotFoundError(
                f"Stage 4 requires Stage 3 original weights at {originals_path}. "
                "Re-run Stage 3 first."
            )
        # F-S3-1: validate the MANIFEST.json sidecar before loading the
        # ~50 GB .pt. Stage 3 writes the manifest LAST, after the .pt's
        # fsync, so a torn .pt (mid-write SIGKILL) leaves NO manifest.
        # Missing or mismatched manifest = fail loudly + delete-and-re-run,
        # NEVER silently consume a partial file.
        #
        # Backward-compat: a .pt produced by a pre-F-S3-1 Stage 3 has no
        # manifest sibling — we accept it with a single WARNING (skipping
        # validation) so the fix doesn't invalidate completed runs already
        # in flight. Once the next full pipeline rev lands, this fallback
        # can be removed.
        if originals_manifest_path.exists():
            from moe_compress.utils.atomic_io import (
                ManifestMismatchError,
                read_and_validate_manifest,
            )
            try:
                read_and_validate_manifest(
                    originals_path,
                    originals_manifest_path,
                    expected_schema_version=1,
                )
            except ManifestMismatchError as exc:
                raise RuntimeError(
                    f"Stage 4: Stage 3 originals manifest validation FAILED — {exc}. "
                    "This is the classic torn-write signature on a "
                    f"~50 GB artifact. Delete both {originals_path.name} and "
                    f"{originals_manifest_path.name} from {artifacts_dir} "
                    "and re-run Stage 3."
                ) from exc
        else:
            # MEDIUM-8 TODO(post-2026-Q3): remove this backward-compat
            # shim once all sidecars under /opt/output/* are regenerated
            # with manifests. The fallback exists because pre-F-S3-1
            # runs produced .pt files without sibling manifests; once
            # those in-flight runs complete, ALL Stage 3 writers emit a
            # manifest and the missing-manifest branch becomes
            # dead-code-loud-fail territory.
            log.warning(
                "Stage 4: %s has no MANIFEST.json sibling (pre-F-S3-1 "
                "calibration run?). Proceeding without manifest validation; "
                "if Stage 4 errors during originals lookup, the .pt may be "
                "torn — delete it and re-run Stage 3.",
                originals_path,
            )
        originals: dict = torch.load(originals_path, map_location="cpu")

        layers = list(iter_moe_layers(model))
        log.info("Stage 4: EoRA residual compensation over %d MoE layers", len(layers))

        # One-shot Trackio emit: run-level shape constants. Both values are
        # already computed from the model and used in the log.info above.
        # Uniform-MoE assumption: only layer 0 is sampled for the
        # per-layer expert count; mixed-expert-count topologies would
        # silently emit only the first layer's value (0 if the attr is
        # missing).
        _n_experts_first = (
            layers[0].experts_module.num_experts
            if layers and hasattr(layers[0].experts_module, "num_experts")
            else 0
        )
        _trackio_log({
            "stage4/config/n_moe_layers": len(layers),
            "stage4/config/n_experts_per_layer": int(_n_experts_first),
            "stage4/config/no_resume": bool(no_resume),
        })

        if no_resume:
            partial_dir = None
        else:
            partial_dir = artifacts_dir / "_stage4_partial"
            partial_dir.mkdir(parents=True, exist_ok=True)
            for _stale in partial_dir.glob("*.tmp"):
                _stale.unlink(missing_ok=True)

        # Snapshot Stage 3 ranks before any widening occurs — read from
        # each layer's in-memory ``FactoredExperts.ranks`` dict (NOT
        # from an on-disk ``rank_map.json``; that filename is not a
        # Stage 4 input). Used by the per-matrix double-widen guard in
        # ``EoraCompensationPlugin.compensate_layer`` to detect in-process
        # re-runs.
        stage3_ranks: dict[int, dict[str, int]] = {}
        for ref in layers:
            fe = ref.experts_module
            if isinstance(fe, FactoredExperts):
                stage3_ranks[ref.layer_idx] = {
                    name: fe.ranks[name] for name in MATRIX_NAMES
                }

        # ``overwrite=("A_cov" in ctx)`` -- when the V2 cache provider
        # already populated A_cov + a_storage_dtype on the ctx, our
        # set() must use overwrite=True (or skip the set; we'd write the
        # same object back, so overwrite is the simpler form). When the
        # cache missed, A_cov / a_storage_dtype are absent on the ctx and
        # the set is a fresh bind.
        _cache_hit = "A_cov" in ctx
        ctx.set("A_cov", A_cov, overwrite=_cache_hit)
        ctx.set("a_storage_dtype", a_storage_dtype, overwrite=_cache_hit)
        ctx.set("originals", originals)
        ctx.set("layers", layers)
        ctx.set("partial_dir", partial_dir)
        ctx.set("stage3_ranks", stage3_ranks)
