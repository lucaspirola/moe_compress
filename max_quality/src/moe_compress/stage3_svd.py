"""Stage 3 — Non-uniform SVD, fused-experts-aware.

At this point each MoE layer still has a fused ``Qwen3_5MoeExperts`` but with
``num_experts = N'_l`` (post-prune). We:

1. Compute per-group statistics (D-Rank) over banks.
2. Choose per-group uniform rank ``k_g`` via D-Rank allocation targeting
   the global ``T_budget`` derived from ``decomposition.svd_rank_ratio``.
3. Swift-SVD+ α selection via WikiText-2 PPL validation (paper 2604.01609,
   §3.2.2). For each α ∈ {0.0, ..., 1.0}, factor the full model at the
   corresponding per-expert ranks and evaluate end-to-end perplexity on a
   validation set; pick the α with lowest PPL. Falls back to spectral
   energy proxy when ``validation_samples=0``.
4. For each layer, **factor every expert at the chosen group rank**
   via AA-SVD using the Stage-2 covariance and a fresh pruned-model
   calibration. Install a :class:`FactoredExperts` in place of the fused module.
5. Per-matrix activation-weighted L-BFGS refine on (U, V) pairs.

Uniform per-group rank lets us keep all factored banks as clean stacked
tensors of shape ``[N, d_out, k]`` and ``[N, k, d_in]`` — no padding.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .budget.solver import BudgetDecomposition
from .utils.activation_hooks import (
    InputCovarianceAccumulator,
    run_calibration,
)
from .utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from .utils.model_io import (
    MATRIX_NAMES,
    FactoredExperts,
    build_banks,
    iter_moe_layers,
    load_model,
    load_json_artifact,
    save_compressed_checkpoint,
    save_json_artifact,
)
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)

# S3-2: covariance collection relocated to stage3/plugins/covariance_collection.
# Re-imported so run() + external callers/tests keep their import paths.
from .stage3.plugins.covariance_collection import (  # noqa: F401
    _collect_covariances,
    _collect_pruned_input_covariance,
    _load_stage2_covariance,
)

# S3-3: D-Rank group-stats + rank allocation relocated to
# stage3/plugins/d_rank_allocate. Re-imported so run() keeps its import paths.
from .stage3.plugins.d_rank_allocate import (  # noqa: F401
    _GroupStats,
    _group_stat,
    _pad,
    _compute_T_budget,
    _d_rank_allocate,
)

# S3-4: Swift-SVD+ alpha-search (both searches), rank redistribution,
# snapshot/restore + WikiText-2 PPL validation relocated to
# stage3/plugins/swift_svd_alpha. Re-imported so run() keeps its paths.
from .stage3.plugins.swift_svd_alpha import (  # noqa: F401
    _snapshot_originals,
    _build_wikitext2_validation,
    _evaluate_wikitext2_ppl,
    _factor_model_at_ranks,
    _restore_fused_experts,
    _swift_svd_plus_alpha_search_validation,
    _swift_svd_plus_alpha_search,
    _redistribute_ranks_swift_svd_plus,
)

# S3-5: AA-SVD core (eigendecomposition cache, rank-k factorization, the
# per-bank covariance lookup, the storage-dtype noise-floor table) relocated
# to stage3/plugins/aa_svd_factor. Re-imported so run() + external callers/
# tests + S3-4's swift_svd_alpha lazy import keep their stage3_svd paths.
from .stage3.plugins.aa_svd_factor import (  # noqa: F401
    _NOISE_FLOOR_BY_DTYPE,
    _EighDecomp,
    _precompute_eigh,
    _aa_svd_precomputed,
    _aa_svd,
    _cov_lookup,
)

# S3-6: Phase C.5 block-level joint refinement (_phase_c5_block_refine +
# _advance_streams) relocated to stage3/plugins/block_refine. Re-imported so
# run() keeps its import path. This is the LAST stage-3 plugin extraction.
from .stage3.plugins.block_refine import (  # noqa: F401
    _phase_c5_block_refine,
    _advance_streams,
)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    decomposition: BudgetDecomposition,
    *,
    device=None,
    no_resume: bool = False,
) -> Path:
    s3 = config["stage3_svd"]
    cal = config["calibration"]

    moe_layers = list(iter_moe_layers(model))
    log.info("Stage 3: %d MoE layers in scope", len(moe_layers))

    # A covariance from Stage 2 (pre-prune inputs per surviving expert).
    # Used by EoRA (Stage 4) and L-BFGS refine (Phase D). Also used for
    # activation-weighted ε* in Swift-SVD+ (D8 fix).
    A_cov = _load_stage2_covariance(artifacts_dir / "_stage2_input_covariance.pt")

    # B covariance + cross-covariance: fresh calibration through both models.
    # Use cal["num_sequences"] directly — do NOT reuse validation_samples here,
    # because validation_samples=0 means "disable PPL α-search, use spectral proxy"
    # and must NOT zero out the B-cov calibration pass.
    spec = spec_from_config(cal, seed_offset=2)
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    bcov_batch_size = int(s3.get("batch_size", 1))
    batches = iter_batches(calib, batch_size=bcov_batch_size)
    B_cov_dtype = getattr(torch, s3.get("bcov_storage_dtype", "bfloat16"))

    B_acc = InputCovarianceAccumulator()
    B_acc.set_storage_dtype(B_cov_dtype)

    # Cross-covariance C = X_pre^T @ X_post (AA-SVD Theorem 3.2, paper 2604.02119).
    # Requires both the original (teacher) model and the pruned (student) model
    # to forward the same calibration batch simultaneously. On H200 (141 GB VRAM):
    # original BF16 (~70 GB) + pruned BF16 (~50 GB) = ~120 GB, leaving ~21 GB
    # for activations and covariance accumulation.
    cross_cov_enabled = s3.get("aa_svd", {}).get("cross_covariance", True)
    C_acc: InputCovarianceAccumulator | None = None
    teacher_model = None
    teacher_moe_layers = None

    # One-shot Trackio emit: AA-SVD path indicator. All values in scope.
    _trackio_log({
        "stage3/config/cross_cov_enabled": bool(cross_cov_enabled),
        "stage3/config/scope": str(s3.get("scope", "moe_experts_only")),
    })

    if no_resume:
        import shutil as _shutil
        for _d in ["_stage3_bcov_partial", "_stage3_ccov_partial"]:
            _p = artifacts_dir / _d
            if _p.exists():
                _shutil.rmtree(_p, ignore_errors=True)

    bcov_spill_dir = artifacts_dir / "_stage3_bcov_partial"
    bcov_spill_dir.mkdir(parents=True, exist_ok=True)
    for _stale in bcov_spill_dir.glob("*.tmp"):
        _stale.unlink(missing_ok=True)
    ccov_spill_dir = artifacts_dir / "_stage3_ccov_partial" if cross_cov_enabled else None
    if ccov_spill_dir is not None:
        ccov_spill_dir.mkdir(parents=True, exist_ok=True)
        for _stale in ccov_spill_dir.glob("*.tmp"):
            _stale.unlink(missing_ok=True)

    # On resume: if all B-cov spill files exist (and all C-cov spills when cross-cov is
    # enabled), skip Phase A entirely — including the ~60s / 70 GB teacher model load.
    # Not checking C-cov completeness here would silently fall back to B-only paths for
    # a run that switched cross_covariance: false → true.
    _all_bcov_spills_exist = (
        not no_resume
        and all(
            (bcov_spill_dir / f"layer_{ref.layer_idx}.pt").exists()
            for ref in moe_layers
        )
        and (
            not cross_cov_enabled
            or (
                ccov_spill_dir is not None
                and all(
                    (ccov_spill_dir / f"layer_{ref.layer_idx}.pt").exists()
                    for ref in moe_layers
                )
            )
        )
    )

    if _all_bcov_spills_exist:
        log.info(
            "Stage 3: all %d B-cov spill files found — skipping Phase A "
            "(covariance collection)",
            len(moe_layers),
        )
        # On resume we skip the else-branch where C_acc is normally created.
        # Re-create it here so the factoring loop can still lazy-load
        # the existing C-cov spill files (AA-SVD Path 1 / Theorem 3.2).
        if cross_cov_enabled:
            C_acc = InputCovarianceAccumulator()
            C_acc.set_storage_dtype(B_cov_dtype)
        # Load the teacher even on resume if Phase C.5 is enabled — its block
        # forwards are required for the anchored MSE objective. Skip the
        # ~60 s / 70 GB load only when block_refine is off.
        if bool(s3.get("block_refine", {}).get("enabled", False)):
            log.info("Stage 3: resume + block_refine — loading original model "
                     "for Phase C.5 anchored objective")
            teacher_model, _ = load_model(
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
            teacher_moe_layers = list(iter_moe_layers(teacher_model))
    else:
        # Teacher is needed when (a) cross-covariance is enabled (Phase A
        # dual-forward, Theorem 3.2), or (b) block_refine is enabled
        # (Phase C.5 anchored MSE objective). Load once and use for both.
        _need_teacher = cross_cov_enabled or bool(s3.get("block_refine", {}).get("enabled", False))
        if _need_teacher:
            log.info("Stage 3: loading original model (cross_cov=%s, block_refine=%s)",
                     cross_cov_enabled, bool(s3.get("block_refine", {}).get("enabled", False)))
            teacher_model, _ = load_model(
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
            teacher_moe_layers = list(iter_moe_layers(teacher_model))
            if cross_cov_enabled:
                C_acc = InputCovarianceAccumulator()
                C_acc.set_storage_dtype(B_cov_dtype)
                log.info("Stage 3: dual-forward covariance collection (B + cross-cov C), batch_size=%d",
                         bcov_batch_size)
            else:
                log.info("Stage 3: B-cov only collection; teacher resident for Phase C.5, batch_size=%d",
                         bcov_batch_size)
        else:
            log.info("Stage 3: B-cov only (no cross-cov, no block_refine), batch_size=%d",
                     bcov_batch_size)
        _collect_covariances(
            model, moe_layers, batches, B_acc, device=device,
            spill_dir=bcov_spill_dir,
            teacher_model=teacher_model,
            teacher_moe_layers=teacher_moe_layers,
            C_acc=C_acc,
            ccov_spill_dir=ccov_spill_dir,
        )

    # Teacher residency: spec §6 keeps the teacher in VRAM through Phase C.5
    # so its block forwards can be invoked on-demand for the anchored objective
    # ‖ℒ_i(X) − ℒ'_i(X')‖². If Phase C.5 (block_refine) is disabled, free now.
    _block_refine_enabled = bool(s3.get("block_refine", {}).get("enabled", False))
    if teacher_model is not None and not _block_refine_enabled:
        teacher_model.to("cpu")
        del teacher_model, teacher_moe_layers
        teacher_model = None  # noqa: F841 - reused below
        teacher_moe_layers = None
        torch.cuda.empty_cache()
        log.info("Stage 3: freed original model after cross-covariance collection (block_refine disabled)")

    # 1. Per-(layer, matrix) group stats and rank allocation.
    log.info("Stage 3: computing per-group stats over %d layers", len(moe_layers))
    group_stats: dict[tuple[int, str], _GroupStats] = {}
    for k, ref in enumerate(moe_layers):
        log.info("  group-stat layer %d/%d (idx=%d)", k + 1, len(moe_layers), ref.layer_idx)
        banks = build_banks(ref)
        # Group-average pre-prune input covariance for D-Rank whitening.
        # Spec §6 Phase B.1: gate/up share `A_gate_up` (hidden-state input);
        # down_proj uses `A_down` (intermediate-activation input). These
        # come from Stage 2 `_stage2_input_covariance.pt` (already loaded
        # into `A_cov`). We average across experts in the group since the
        # spec stipulates a single covariance per (layer, matrix_type).
        for name in MATRIX_NAMES:
            cov_key_name = "gate_proj" if name == "up_proj" else name
            covs = []
            for e in range(ref.num_routed_experts):
                t = _cov_lookup(A_cov, ref.layer_idx, e, cov_key_name)
                if t is not None:
                    covs.append(t.to(torch.float32))
            A_g = torch.stack(covs).mean(0) if covs else None
            group_stats[(ref.layer_idx, name)] = _group_stat(
                ref.num_routed_experts, banks[name], A_g=A_g,
            )

    T_budget = _compute_T_budget(group_stats, decomposition.svd_rank_ratio)
    proj_weights = config.get("stage3_svd", {}).get("d_rank", {}).get("per_projection_weight", {})
    ranks = _d_rank_allocate(group_stats, T_budget, proj_weights=proj_weights or None)

    # Swift-SVD+ α selection (paper 2604.01609, §3.2.2 / Algorithm 2).
    # Within each (layer, matrix_type) group, redistribute the group's total
    # rank budget across individual experts using the blending score:
    #   s_i = β_i^α · (log(e + ε*_i))^{1-α}
    # where β_i = spectral energy proportion and ε*_i = reconstruction error
    # at a reference rank. α balances the two signals.
    #
    # Primary path (paper-exact): when validation_samples > 0, select the
    # global α by factoring the full model at each candidate's ranks and
    # evaluating WikiText-2 PPL end-to-end.
    #
    # Fallback (spectral proxy): when validation_samples = 0, select α by
    # minimising total activation-weighted tail spectral energy (no forward
    # passes, seconds).
    svd_plus_cfg = s3.get("swift_svd_plus", {})
    alpha_grid = svd_plus_cfg.get("alpha_grid")
    per_group_type = svd_plus_cfg.get("per_group_type", True)
    validation_samples = int(svd_plus_cfg.get("validation_samples", 0))

    # Snapshot originals to CPU BEFORE α search. Used by both the
    # validation-based α search (factor → eval → restore cycle) and
    # Stage 4 EoRA residual computation. Moving it here means both
    # consumers share one snapshot and the factoring loop no longer
    # needs to build it inline.
    #
    # Memory: ~50 GB CPU RAM for Qwen3.6-35B-A3B post-prune.
    # H200 has 256 GB host RAM; A-cov (~68 GB) + originals (~50 GB)
    # + OS (~10 GB) ≈ ~128 GB → ~128 GB headroom.
    originals = _snapshot_originals(moe_layers)
    log.info("Snapshotted %d original expert matrices to CPU for "
             "α-search and Stage 4 residuals", len(originals))

    # Persist originals before Phase D. Stage 4 reads this file, so it must
    # exist even if Stage 3 crashes mid-factoring. Written once here so
    # Stage 4 can always resume cleanly from the correct original weights.
    _orig_path = artifacts_dir / "_stage3_original_weights.pt"
    torch.save(originals, _orig_path)
    log.info("Saved Stage 3 original weights snapshot (%d matrices) → %s",
             len(originals), _orig_path)

    # Pre-flight RAM check: paper-compliance contract (spec §6 Phase B.2)
    # requires the end-to-end PPL α-search per Swift-SVD §3.2.2. If host RAM
    # cannot host the snapshot + eval working set, fail fast rather than
    # silently degrade to a spectral proxy that produces a non-paper-compliant
    # model. Operators must provision ≥15 GB headroom or reduce
    # validation_samples to fit.
    if validation_samples > 0:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "Stage 3 α-search requires psutil for the host-RAM pre-flight "
                "check (spec §6 Phase B.2 paper-compliance contract). Install "
                "psutil or set stage3_svd.validation_samples=0 to skip the "
                "α-search entirely."
            ) from exc
        avail_gb = psutil.virtual_memory().available / 1e9
        # Default headroom (15 GB) is sized for production: ~50 GB snapshot in
        # CPU RAM + ~5 GB B-cov per layer + ~5 GB eval working set on a 30 B
        # base model. Smoke tests on toy models override via swift_svd_plus
        # config to skip the gate while still exercising the α-search path.
        min_headroom_gb = float(svd_plus_cfg.get("alpha_search_min_host_ram_gb", 15.0))
        if avail_gb < min_headroom_gb:
            raise RuntimeError(
                f"Stage 3 α-search: only {avail_gb:.1f} GB host RAM available, "
                f"need ≥{min_headroom_gb:.0f} GB headroom for the paper-exact "
                f"end-to-end PPL grid (spec §6 Phase B.2). Provision more RAM "
                f"or reduce stage3_svd.validation_samples to fit. The previous "
                f"silent spectral-proxy fallback (D9) was removed because it "
                f"produced a non-paper-compliant model."
            )

    # Resume: if α was already selected (and saved) in a previous interrupted
    # run, reload it and skip the ~33 min α search entirely.
    _alpha_cache_path = artifacts_dir / "_stage3_alpha_result.json"
    _alpha_loaded = False
    if not no_resume and _alpha_cache_path.exists() and alpha_grid and len(alpha_grid) > 1:
        try:
            _cached_alpha = load_json_artifact(_alpha_cache_path)
            alpha_by_type = _cached_alpha.get("alpha_by_type")
            if alpha_by_type is not None:
                log.info("Stage 3: loaded cached α from %s — skipping α search", _alpha_cache_path)
                per_expert_ranks = _redistribute_ranks_swift_svd_plus(
                    moe_layers, group_stats, ranks, alpha_by_type,
                    grouped_svs_cache=None, A_cov=A_cov,
                )
                _alpha_loaded = True
        except Exception as _exc:
            log.warning("Stage 3: failed to load α cache (%s) — running α search", _exc)

    if not _alpha_loaded:
        if alpha_grid and len(alpha_grid) > 1:
            if validation_samples > 0:
                # Paper-exact: global α via WikiText-2 PPL validation (§3.2.2).
                log.info("Stage 3: Swift-SVD+ α selection via validation "
                         "(%d samples, %d candidates)",
                         validation_samples, len(alpha_grid))
                best_global_alpha = _swift_svd_plus_alpha_search_validation(
                    model, tokenizer, moe_layers, group_stats, ranks,
                    alpha_grid, originals, A_cov, B_acc, bcov_spill_dir,
                    C_acc, ccov_spill_dir, config, device=device,
                    storage_dtype=B_cov_dtype,
                )
                if per_group_type:
                    # Per-type refinement using spectral proxy, seeded from the
                    # validation-selected global α. The paper uses a single α
                    # for all projections; per-type is our extension.
                    log.info("Stage 3: per-type α refinement via spectral "
                             "proxy (seed=%.1f)", best_global_alpha)
                    alpha_by_type = _swift_svd_plus_alpha_search(
                        moe_layers, group_stats, ranks, alpha_grid,
                        per_group_type=True,
                        A_cov=A_cov,
                    )
                    log.info("Stage 3: Swift-SVD+ per-type α = %s "
                             "(global validation best = %.1f)",
                             alpha_by_type, best_global_alpha)
                else:
                    alpha_by_type = {"all": best_global_alpha}
                    log.info("Stage 3: Swift-SVD+ selected α = %s",
                             alpha_by_type)
            else:
                # Fallback: spectral proxy only (no forward passes).
                log.info("Stage 3: Swift-SVD+ α search via spectral proxy "
                         "(%d candidates, per_group_type=%s)",
                         len(alpha_grid), per_group_type)
                alpha_by_type = _swift_svd_plus_alpha_search(
                    moe_layers, group_stats, ranks, alpha_grid,
                    per_group_type=per_group_type,
                    A_cov=A_cov,
                )
                log.info("Stage 3: Swift-SVD+ selected α = %s", alpha_by_type)
            # Redistribute per-expert ranks within each group using the selected α.
            per_expert_ranks = _redistribute_ranks_swift_svd_plus(
                moe_layers, group_stats, ranks, alpha_by_type,
                grouped_svs_cache=None,
                A_cov=A_cov,
            )
        else:
            alpha_by_type = None
            per_expert_ranks = None  # uniform: every expert gets ranks[(li, name)]

    # One-shot Trackio emit of D-Rank / Swift-SVD+ summary. All values in scope.
    _stage3_alpha_emit: dict[str, float | int | str] = {
        "stage3/config/t_budget": int(T_budget),
        "stage3/config/alpha_candidates_count": int(len(alpha_grid) if alpha_grid else 0),
    }
    if isinstance(alpha_by_type, dict):
        for _k, _v in alpha_by_type.items():
            try:
                _stage3_alpha_emit[f"stage3/config/alpha_by_type/{_k}"] = float(_v)
            except (TypeError, ValueError):
                _stage3_alpha_emit[f"stage3/config/alpha_by_type/{_k}"] = str(_v)
    _trackio_log(_stage3_alpha_emit)

    # Persist α result so a crash during Phase D doesn't force re-running
    # the ~33 min α search on resume.
    if not no_resume:
        save_json_artifact(
            {"alpha_by_type": alpha_by_type},
            artifacts_dir / "_stage3_alpha_result.json",
        )

    # 2. Factor per-layer using the pre-built originals snapshot.
    rank_map: dict[str, int] = {}

    for ref in moe_layers:
        # When Swift-SVD+ gives per-expert ranks, allocate at the max rank
        # across experts for each matrix type (the slot width). Experts with
        # lower rank will be zero-padded; effective_ranks tracks the true rank.
        if per_expert_ranks is not None:
            ranks_layer = {
                name: max(
                    per_expert_ranks.get((ref.layer_idx, name, e), ranks[(ref.layer_idx, name)])
                    for e in range(ref.num_routed_experts)
                )
                for name in MATRIX_NAMES
            }
        else:
            ranks_layer = {
                name: ranks[(ref.layer_idx, name)] for name in MATRIX_NAMES
            }
        # Lazy-load this layer's B-cov from the per-layer spill files.
        # Keeps in-memory cov bounded to ~one layer (~3-5 GB at bf16).
        # Assert (not silent fall-through) — a missing spill at this
        # point would mean _aa_svd silently falls back to plain SVD for
        # this whole layer's experts, ignoring the activation-aware
        # weighting; we'd ship a degraded model. Crash loud instead.
        loaded = B_acc.load_layer_from_disk(ref.layer_idx, bcov_spill_dir)
        if not loaded:
            raise RuntimeError(
                f"Stage 3 factor: B-cov spill missing for layer {ref.layer_idx} "
                f"at {bcov_spill_dir}/layer_{ref.layer_idx}.pt. The B-cov phase "
                "should have produced this file. Investigate before proceeding."
            )
        # Also load cross-covariance C for this layer (if dual-forward was run).
        if C_acc is not None and ccov_spill_dir is not None:
            c_loaded = C_acc.load_layer_from_disk(ref.layer_idx, ccov_spill_dir)
            if not c_loaded:
                log.warning(
                    "Stage 3 factor: cross-cov spill missing for layer %d — "
                    "falling back to auto-covariance for this layer.",
                    ref.layer_idx,
                )
        # Build FactoredExperts on the same device / dtype.
        # Originals are already snapshotted to CPU (before α search);
        # offload the dense expert module before allocating FactoredExperts
        # to avoid brief double-occupancy OOM on 80 GB A100s.
        ex = ref.experts_module
        dtype = ex.gate_up_proj.dtype
        dev = ex.gate_up_proj.device
        ex.to("cpu")
        torch.cuda.empty_cache()
        new_factored = FactoredExperts(
            num_experts=ref.num_routed_experts,
            hidden_dim=ex.gate_up_proj.shape[-1],
            intermediate_dim=ex.gate_up_proj.shape[1] // 2,
            ranks=ranks_layer, dtype=dtype, device=dev,
        )
        # Fill factors by per-expert AA-SVD. Track relative reconstruction
        # error per (layer, matrix) so the dashboard shows whether the chosen
        # rank is enough — a "convergence in spirit" signal for the SVD.
        # Per-expert weighted relative error: mean of ||(W-UV)L_B||/||WL_B|| across experts.
        #
        # Optimization: gate_proj and up_proj share the same B and C covariance
        # (``_cov_lookup`` falls back from up_proj to gate_proj).  We precompute
        # the eigh(B) decomposition + rhs product once per expert and reuse it
        # for both projections, eliminating one eigh(2048×2048) call per expert
        # (~7,200 redundant calls across 40 layers).
        err_sum: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        n_per_matrix: dict[str, int] = {n: 0 for n in MATRIX_NAMES}
        k_eff_clip_count: dict[str, int] = {n: 0 for n in MATRIX_NAMES}
        for e in range(ref.num_routed_experts):
            # --- Precompute shared eigh for gate_proj / up_proj ---
            B_shared = _cov_lookup(B_acc.covariance, ref.layer_idx, e, "gate_proj")
            A_shared = _cov_lookup(A_cov, ref.layer_idx, e, "gate_proj")
            C_shared = None
            if C_acc is not None:
                C_shared = _cov_lookup(C_acc.covariance, ref.layer_idx, e, "gate_proj")
            gate_up_decomp: _EighDecomp | None = None
            if B_shared is not None:
                try:
                    gate_up_decomp = _precompute_eigh(
                        B_shared, A_shared, C_shared,
                        device=dev, storage_dtype=B_cov_dtype,
                    )
                except ValueError:
                    pass  # falls through to plain SVD per matrix below

            for name in MATRIX_NAMES:
                W = originals[(ref.layer_idx, e, name)].to(device=dev, dtype=torch.float32)
                # Per-expert rank from Swift-SVD+ if available, else group-uniform.
                if per_expert_ranks is not None:
                    k = per_expert_ranks.get((ref.layer_idx, name, e), ranks_layer[name])
                else:
                    k = ranks_layer[name]
                if name in ("gate_proj", "up_proj") and gate_up_decomp is not None:
                    # Reuse the precomputed eigh for gate_proj and up_proj.
                    U_k, V_k, rel_err, k_eff = _aa_svd_precomputed(
                        W, gate_up_decomp, k, device=dev,
                    )
                else:
                    # down_proj has its own B (intermediate-dim covariance),
                    # or gate_up_decomp failed — fall back to full _aa_svd.
                    A = _cov_lookup(A_cov, ref.layer_idx, e, name)
                    B = _cov_lookup(B_acc.covariance, ref.layer_idx, e, name)
                    C = None
                    if C_acc is not None:
                        C = _cov_lookup(C_acc.covariance, ref.layer_idx, e, name)
                    U_k, V_k, rel_err, k_eff = _aa_svd(
                        W, A, B, k, C=C, device=dev, storage_dtype=B_cov_dtype,
                    )
                if k_eff < k:
                    k_eff_clip_count[name] += 1
                new_factored.set_factors(e, name, U_k, V_k, effective_rank=k_eff)
                rank_map[f"L{ref.layer_idx}_E{e}_{name}"] = k
                err_sum[name] += rel_err
                n_per_matrix[name] += 1
        # Swap in.
        setattr(ref.mlp, "experts", new_factored)
        ref.experts_module = new_factored
        recon_metrics: dict[str, float] = {"stage3/recon_layer_idx": float(ref.layer_idx)}
        for name in MATRIX_NAMES:
            if n_per_matrix[name] > 0:
                rel = err_sum[name] / n_per_matrix[name]
                # Renamed from `recon_rel_err` post-bf16 fix: this is the
                # B-weighted singular-value-tail ratio of M = W·L_B. The old
                # key is dual-emitted as an alias so existing trackio
                # dashboards keep working — TODO(post-launch): drop the alias
                # once dashboards are migrated to `b_weighted_tail_ratio`.
                recon_metrics[f"stage3/b_weighted_tail_ratio/{name}"] = rel
                recon_metrics[f"stage3/recon_rel_err/{name}"] = rel
                recon_metrics[f"stage3/k_eff_clip_count/{name}"] = float(k_eff_clip_count[name])
                recon_metrics[f"stage3/k_eff_clip_ratio/{name}"] = (
                    k_eff_clip_count[name] / max(n_per_matrix[name], 1)
                )
                # `b_weighted_tail_ratio` = ‖tail_S(M)‖/‖S(M)‖, the singular-
                # value-tail proxy for ‖(W−UV)L_B‖/‖WL_B‖. Pre-fix code logged
                # this same key as `rel_recon_err`; numbers from before commit
                # e7e0fbf are not directly comparable.
                log.info("  L%d %s rank=%d b_weighted_tail_ratio=%.4f k_eff_clipped=%d/%d",
                         ref.layer_idx, name, ranks_layer[name], rel,
                         k_eff_clip_count[name], n_per_matrix[name])
        _trackio_log(recon_metrics)
        log.info("  layer %d factored at ranks=%s", ref.layer_idx, ranks_layer)
        # Drop this layer's B-cov and C-cov from memory now that we're done factoring
        # it. The next iteration will lazy-load the next layer's spill.
        B_acc.unload_layer(ref.layer_idx)
        if C_acc is not None:
            C_acc.unload_layer(ref.layer_idx)

    # 4. Phase C.5 — block-level joint refinement (paper 2604.02119 §3.3).
    # Replaces the legacy per-matrix L-BFGS refine; trains the block's
    # FactoredExperts U/V slots and the two RMSNorm scales jointly via
    # AdamW (fp32) on the anchored MSE objective ‖ℒ_i(X) − ℒ'_i(X')‖².
    if _block_refine_enabled:
        if teacher_model is None or teacher_moe_layers is None:
            raise RuntimeError(
                "Stage 3 Phase C.5 requires the teacher model to be resident. "
                "Either disable stage3_svd.block_refine.enabled or ensure "
                "Phase A loaded the teacher (check aa_svd.cross_covariance "
                "and the resume path)."
            )
        br = s3["block_refine"]
        _phase_c5_block_refine(
            model, teacher_model, moe_layers, teacher_moe_layers, calib,
            batch_size=int(br.get("batch_size", 32)),
            learning_rate=float(br.get("learning_rate", 1.0e-4)),
            epochs=int(br.get("epochs", 25)),
            warmup_ratio=float(br.get("warmup_ratio", 0.1)),
            weight_decay=float(br.get("weight_decay", 0.0)),
            artifacts_dir=artifacts_dir, no_resume=no_resume, device=device,
        )

    # Free teacher after Phase C.5 (or after factoring if C.5 disabled and the
    # earlier branch left it resident — defensive).
    if teacher_model is not None:
        teacher_model.to("cpu")
        del teacher_model, teacher_moe_layers
        torch.cuda.empty_cache()
        log.info("Stage 3: freed original model after Phase C.5")

    out_dir = artifacts_dir / "stage3_svd"
    save_compressed_checkpoint(
        model, tokenizer, out_dir,
        pipeline_stage="stage3_svd",
        extra_metadata={"alpha_by_type": alpha_by_type, "T_budget": T_budget},
    )
    save_json_artifact({
        "rank_map": rank_map,
        "T_budget": T_budget,
        "alpha_by_type": alpha_by_type,
        "per_layer_ranks": {
            str(ref.layer_idx): {n: ranks[(ref.layer_idx, n)] for n in MATRIX_NAMES}
            for ref in moe_layers
        },
        "config": s3,
    }, out_dir / "rank_map.json")
    # Clean up the per-layer B-cov spill dir on successful Stage 3 finish.
    # Otherwise a future re-run (e.g. with a different svd_rank_ratio) would
    # silently reuse the stale cov instead of recomputing. The spill dir's
    # purpose is mid-stage crash-resume only; once Stage 3 has completed
    # cleanly its outputs live in stage3_svd/ and originals.pt.
    import shutil
    if (artifacts_dir / "_stage3_bcov_partial").exists():
        shutil.rmtree(artifacts_dir / "_stage3_bcov_partial", ignore_errors=True)
        log.info("Removed Stage 3 B-cov spill dir (no longer needed post-success).")
    if (artifacts_dir / "_stage3_ccov_partial").exists():
        shutil.rmtree(artifacts_dir / "_stage3_ccov_partial", ignore_errors=True)
        log.info("Removed Stage 3 cross-cov spill dir (no longer needed post-success).")
    log.info("Stage 3 complete → %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Group stats + allocations
# ---------------------------------------------------------------------------
# S3-3: ``_GroupStats`` / ``_group_stat`` / ``_pad`` / ``_compute_T_budget`` /
# ``_d_rank_allocate`` relocated to ``stage3/plugins/d_rank_allocate`` and
# re-imported above (see the S3-3 ``# noqa: F401`` block).


# S3-4: Swift-SVD+ alpha-search (both searches), rank redistribution,
# snapshot/restore + WikiText-2 PPL validation relocated to
# ``stage3/plugins/swift_svd_alpha``. They are re-imported in the
# top-of-module import block above (see the S3-4 ``# noqa: F401`` block).


# ---------------------------------------------------------------------------
# AA-SVD per matrix
# ---------------------------------------------------------------------------
# S3-5: ``_NOISE_FLOOR_BY_DTYPE`` / ``_EighDecomp`` / ``_precompute_eigh`` /
# ``_aa_svd_precomputed`` / ``_aa_svd`` / ``_cov_lookup`` relocated to
# ``stage3/plugins/aa_svd_factor`` and re-imported in the top-of-module import
# block above (see the S3-5 ``# noqa: F401`` block).


# ---------------------------------------------------------------------------
# Post-prune input covariance (for AA-SVD B matrix)
# ---------------------------------------------------------------------------
# S3-2: ``_collect_covariances`` / ``_collect_pruned_input_covariance`` /
# ``_load_stage2_covariance`` relocated to stage3/plugins/covariance_collection.
# They are re-imported in the top-of-module import block above.


# ---------------------------------------------------------------------------
# Per-matrix L-BFGS reconstruction refine
# ---------------------------------------------------------------------------


# S3-6: ``_phase_c5_block_refine`` / ``_advance_streams`` relocated to
# ``stage3/plugins/block_refine`` and re-imported in the top-of-module import
# block above (see the S3-6 ``# noqa: F401`` block). This is the LAST stage-3
# plugin extraction.


