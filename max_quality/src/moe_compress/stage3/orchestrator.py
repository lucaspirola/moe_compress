"""Stage 3 orchestrator — the real plugin-driven phase sequencer (S3-7a).

S3-1 shipped this module as a thin delegation to the legacy
``stage3_svd.run`` monolith. S3-2..S3-6 extracted the SVD algorithm
(covariance collection, D-Rank allocation, Swift-SVD+ alpha-search, AA-SVD
factorization, Phase C.5 block-refine) into ``stage3/plugins/``. S3-7a
flips the relationship: :func:`run` here is now the REAL orchestrator and
``stage3_svd.run`` is a thin shim that delegates to it.

The schedule
------------
``collect_covariances -> allocate_ranks -> select_alpha ->
LOOP layers[factor_layer] -> refine_blocks -> finalize``.

Division of labour (mirror of stage 1)
--------------------------------------
The five plugin hooks own the ALGORITHM -- verbatim slices relocated from
the monolith ``run()``. This orchestrator owns the GLUE: config parse,
teacher-model lifecycle, the per-(layer, matrix) ``group_stats`` loop, the
``originals`` snapshot / alpha-cache / RAM pre-flight / telemetry, and the
finalize block (checkpoint save, ``rank_map.json``, spill cleanup). Every
glue line is a verbatim copy from the monolith ``run()``, just reorganized
into run-glue around the ``walk_phases`` / ``loop_over`` calls.

Monkeypatch survival (HAZARD H3)
--------------------------------
The golden / smoke tests ``monkeypatch.setattr`` on the SOURCE module that
binds each name -- ``utils.calibration`` for ``build_calibration_tensor``,
``utils.model_io`` for ``save_compressed_checkpoint``, and the ``stage3_svd``
module object for ``load_model``. So this orchestrator calls those three
module-qualified (``cal_mod.build_calibration_tensor`` / ``mio.save_*`` /
``_stage3_svd.load_model``) -- the patches reach them without the test
fixture needing to also patch ``stage3.orchestrator``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..budget.solver import BudgetDecomposition
from ..pipeline.context import PipelineContext
from ..pipeline.registry import PluginRegistry
from ..tools.phase_walker import loop_over, walk_phases
from ..utils.activation_hooks import InputCovarianceAccumulator
from ..utils import calibration as cal_mod
from ..utils.calibration import iter_batches, spec_from_config
from ..utils import model_io as mio
from ..utils.model_io import (
    MATRIX_NAMES,
    build_banks,
    iter_moe_layers,
    load_json_artifact,
    save_json_artifact,
)
from ..utils.trackio_log import trackio_log as _trackio_log

from .plugins.aa_svd_factor import AaSvdFactorPlugin, _cov_lookup
from .plugins.block_hidden_cache import Stage3BlockHiddenCacheProvider
from .plugins.block_refine import BlockRefinePlugin
from .plugins.covariance_collection import (
    CovarianceCollectionPlugin,
    _load_stage2_covariance,
)
from .plugins.d_rank_allocate import DRankAllocatePlugin, _GroupStats, _group_stat
from .plugins.input_cov_cache import Stage3InputCovCacheProvider
from .plugins.swift_svd_alpha import (
    SwiftSvdAlphaPlugin,
    _redistribute_ranks_swift_svd_plus,
    _snapshot_originals,
)
from .plugins.wanda_intra_expert_score import WandaIntraExpertScorePlugin

log = logging.getLogger(__name__)


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
    """Run Stage 3 -- non-uniform SVD factorization -- via the plugin pipeline.

    Threads one :class:`PipelineContext` through the six-phase schedule.
    Returns the Stage 3 output directory (``artifacts_dir / "stage3_svd"``),
    same as the legacy monolith ``run()``.
    """
    s3 = config["stage3_svd"]
    cal = config["calibration"]

    moe_layers = list(iter_moe_layers(model))
    log.info("Stage 3: %d MoE layers in scope", len(moe_layers))

    # A covariance from Stage 2 (pre-prune inputs per surviving expert).
    # Used by EoRA (Stage 4) and L-BFGS refine (Phase D). Also used for
    # activation-weighted eps* in Swift-SVD+ (D8 fix).
    #
    # V2 cache-first: try the calibration-v2 sidecar at
    # ``<jsonl_dir>/sidecars/covariance.pt`` BEFORE the legacy
    # _stage2_input_covariance.pt load. On hit, the dict-shaped payload
    # is byte-compatible with the legacy file's "covariance" field and
    # plugs into ``A_cov`` directly; on miss, fall through to the
    # legacy artifact. The provider is path-scoped (it reads from the
    # calibration JSONL's sibling sidecars/ dir, not from artifacts_dir),
    # so we resolve the calibration JSONL the same way Stage 2 does.
    #
    # Wiring: route the cache lookup through the plugin registry via
    # ``dispatch_first("on_load", ...)`` -- uniform with Stage 4. The
    # cache provider sets ``ctx.A_cov`` on hit; on miss the ctx is
    # untouched and the legacy ``_load_stage2_covariance`` runs below.
    # A throw-away ``cache_ctx`` here keeps the cache binding decoupled
    # from the main ``run_ctx`` (constructed after the teacher-load
    # branching) -- we promote ``A_cov`` onto ``run_ctx`` together with
    # the rest of the slots.
    A_cov = None
    _cache_ctx = PipelineContext()
    try:
        from pathlib import Path as _Path
        from ..utils.calibration import _DEFAULT_SELF_TRACES_PATH
        _calib_source = cal.get("jsonl_path", _DEFAULT_SELF_TRACES_PATH)
        _calib_jsonl_path = _Path(_calib_source)
        if not _calib_jsonl_path.is_absolute():
            _calib_jsonl_path = _Path.cwd() / _calib_jsonl_path
        # The cache provider is one of the plugins in the registry built
        # below; reuse the same registry here so an introspection tool
        # observes one wiring, not two.
        _cache_only_plugins = [Stage3InputCovCacheProvider()]
        PluginRegistry.dispatch_first(
            _cache_only_plugins, "on_load", _cache_ctx, _calib_jsonl_path,
        )
        if _cache_ctx.has("A_cov"):
            A_cov = _cache_ctx.get("A_cov")
            log.info(
                "Stage 3: V2 input-cov cache HIT (%d keys) -- skipping "
                "_stage2_input_covariance.pt load", len(A_cov),
            )
    except (FileNotFoundError, OSError) as _exc:
        # Cache attempt MUST NOT block the legacy fallback for routine
        # filesystem misses. ValueError from _check_schema is NOT caught
        # here -- a schema mismatch is an actionable user error
        # ("Delete the sidecar to regenerate") and silently falling back
        # to the legacy .pt file would mask the upgrade path.
        log.warning("Stage 3: V2 input-cov cache lookup failed (%s) -- "
                    "falling back to _stage2_input_covariance.pt", _exc)
    if A_cov is None:
        A_cov = _load_stage2_covariance(artifacts_dir / "_stage2_input_covariance.pt")

    # B covariance + cross-covariance: fresh calibration through both models.
    # Use cal["num_sequences"] directly -- do NOT reuse validation_samples here,
    # because validation_samples=0 means "disable PPL alpha-search, use spectral
    # proxy" and must NOT zero out the B-cov calibration pass.
    spec = spec_from_config(cal, seed_offset=2)
    calib = cal_mod.build_calibration_tensor(
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

    # ---- one PipelineContext: input slots + run-glue intermediates --------
    # rank_map: ONE shared mutable dict set on the ROOT ctx. The per-layer
    # factor_layer loop opens a child scope per layer but MUTATES this same
    # dict in place across iterations (HAZARD H1 -- never a per-child rank_map).
    run_ctx = PipelineContext()
    run_ctx.set("model", model)
    run_ctx.set("tokenizer", tokenizer)
    run_ctx.set("config", config)
    run_ctx.set("artifacts_dir", artifacts_dir)
    run_ctx.set("decomposition", decomposition)
    run_ctx.set("device", device)
    run_ctx.set("no_resume", no_resume)
    run_ctx.set("moe_layers", moe_layers)
    run_ctx.set("A_cov", A_cov)
    run_ctx.set("calib", calib)
    run_ctx.set("batches", batches)
    run_ctx.set("B_acc", B_acc)
    run_ctx.set("B_cov_dtype", B_cov_dtype)
    run_ctx.set("cross_cov_enabled", cross_cov_enabled)
    run_ctx.set("bcov_spill_dir", bcov_spill_dir)
    run_ctx.set("ccov_spill_dir", ccov_spill_dir)
    run_ctx.set("rank_map", {})

    # Cache provider registered FIRST so a future ``dispatch_first(plugins,
    # "on_load", ...)`` call (or a downstream phase that wants to consult
    # the cache through the registry) sees it before the live plugins.
    # The actual cache lookup for the run-scope A_cov happens above (run
    # glue), but registering the provider here keeps the plugin sequence
    # introspectable + parity with the Stage 2 / Stage 4 wiring.
    registry = PluginRegistry([
        Stage3InputCovCacheProvider(),
        Stage3BlockHiddenCacheProvider(),
        CovarianceCollectionPlugin(),
        # Routing-weighted Wanda intra-expert importance score (MoE-Pruner
        # arXiv:2410.12013, clean-room from fusion_bench upstream). Default
        # OFF — registry filter drops the plugin when
        # ``stage3.wanda_intra_expert.enabled`` is false, making the
        # ``collect_wanda_scores`` walk_phases call a no-op.
        WandaIntraExpertScorePlugin(),
        DRankAllocatePlugin(),
        SwiftSvdAlphaPlugin(),
        AaSvdFactorPlugin(),
        BlockRefinePlugin(),
    ])
    # ``registry.enabled(config)`` drops BlockRefinePlugin when
    # ``stage3_svd.block_refine.enabled`` is false -- then the
    # ``walk_phases(("refine_blocks",), ...)`` call is a no-op, byte-identical
    # to the monolith's ``if _block_refine_enabled:`` skip.
    plugins = registry.enabled(config)

    # ---- collect_covariances --------------------------------------------
    # The teacher load + resume branching is RUN-GLUE; only the
    # _collect_covariances call lives inside CovarianceCollectionPlugin.

    # On resume: if all B-cov spill files exist (and all C-cov spills when cross-cov is
    # enabled), skip Phase A entirely -- including the ~60s / 70 GB teacher model load.
    # Not checking C-cov completeness here would silently fall back to B-only paths for
    # a run that switched cross_covariance: false -> true.
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

    # Function-local import: stage3_svd.run is the shim that delegates HERE,
    # so a module-top ``from .. import stage3_svd`` would close an import
    # cycle. We need the module OBJECT (not just load_model) because the
    # smoke test patches ``stage3_svd.load_model`` -- calling it module-
    # qualified through this object honours that patch (HAZARD H3).
    from .. import stage3_svd as _stage3_svd

    if _all_bcov_spills_exist:
        log.info(
            "Stage 3: all %d B-cov spill files found -- skipping Phase A "
            "(covariance collection)",
            len(moe_layers),
        )
        # On resume we skip the else-branch where C_acc is normally created.
        # Re-create it here so the factoring loop can still lazy-load
        # the existing C-cov spill files (AA-SVD Path 1 / Theorem 3.2).
        if cross_cov_enabled:
            C_acc = InputCovarianceAccumulator()
            C_acc.set_storage_dtype(B_cov_dtype)
        # Load the teacher even on resume if Phase C.5 is enabled -- its block
        # forwards are required for the anchored MSE objective. Skip the
        # ~60 s / 70 GB load only when block_refine is off.
        if bool(s3.get("block_refine", {}).get("enabled", False)):
            log.info("Stage 3: resume + block_refine -- loading original model "
                     "for Phase C.5 anchored objective")
            teacher_model, _ = _stage3_svd.load_model(
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
            teacher_model, _ = _stage3_svd.load_model(
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
        # collect_covariances phase -- CovarianceCollectionPlugin.collect_covariances
        # delegates to the relocated _collect_covariances. Publish the slots it
        # reads onto the ctx, then walk the phase.
        run_ctx.set("teacher_model", teacher_model)
        run_ctx.set("teacher_moe_layers", teacher_moe_layers)
        run_ctx.set("C_acc", C_acc)
        walk_phases(("collect_covariances",), plugins, run_ctx)

    # Teacher residency: spec section 6 keeps the teacher in VRAM through
    # Phase C.5 so its block forwards can be invoked on-demand for the anchored
    # objective. If Phase C.5 (block_refine) is disabled, free now.
    _block_refine_enabled = bool(s3.get("block_refine", {}).get("enabled", False))
    if teacher_model is not None and not _block_refine_enabled:
        teacher_model.to("cpu")
        del teacher_model, teacher_moe_layers
        teacher_model = None
        teacher_moe_layers = None
        torch.cuda.empty_cache()
        log.info("Stage 3: freed original model after cross-covariance collection (block_refine disabled)")

    # The teacher/C_acc slots may have been written by the collect_covariances
    # branch above (or never, on the resume path). Republish the post-Phase-A
    # truth -- overwrite=True because the no-resume branch already set them.
    run_ctx.set("teacher_model", teacher_model, overwrite=True)
    run_ctx.set("teacher_moe_layers", teacher_moe_layers, overwrite=True)
    run_ctx.set("C_acc", C_acc, overwrite=True)

    # ---- collect_wanda_scores -------------------------------------------
    # Routing-weighted Wanda intra-expert score (MoE-Pruner arXiv:2410.12013).
    # Default OFF -- WandaIntraExpertScorePlugin.is_enabled gates on
    # stage3.wanda_intra_expert.enabled, so the registry filter drops it
    # and this walk_phases call is a byte-identical no-op for runs that
    # do not opt in. When enabled, the plugin runs its own calibration
    # pass through the student model and publishes
    # ctx["stage3.wanda_intra_expert_score"] (a nested
    # {layer: {expert: {matrix: Tensor}}} score map). See
    # stage3.plugins.wanda_intra_expert_score for the math + upstream
    # citation. Placed between collect_covariances and allocate_ranks so
    # a future allocator that wants to consult the score can read it
    # through the parent chain.
    walk_phases(("collect_wanda_scores",), plugins, run_ctx)

    # ---- allocate_ranks --------------------------------------------------
    # The per-(layer, matrix) group_stats loop is RUN-GLUE (it needs
    # build_banks / _cov_lookup); only _compute_T_budget + _d_rank_allocate
    # live inside DRankAllocatePlugin.allocate_ranks.
    log.info("Stage 3: computing per-group stats over %d layers", len(moe_layers))
    group_stats: dict[tuple[int, str], _GroupStats] = {}
    for k, ref in enumerate(moe_layers):
        log.info("  group-stat layer %d/%d (idx=%d)", k + 1, len(moe_layers), ref.layer_idx)
        banks = build_banks(ref)
        # Group-average pre-prune input covariance for D-Rank whitening.
        # Spec section 6 Phase B.1: gate/up share `A_gate_up` (hidden-state
        # input); down_proj uses `A_down` (intermediate-activation input).
        # These come from Stage 2 `_stage2_input_covariance.pt` (already loaded
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

    run_ctx.set("group_stats", group_stats)
    walk_phases(("allocate_ranks",), plugins, run_ctx)
    T_budget = run_ctx.get("T_budget")
    ranks = run_ctx.get("ranks")

    # Swift-SVD+ alpha selection (paper 2604.01609, section 3.2.2 / Algorithm 2).
    # Within each (layer, matrix_type) group, redistribute the group's total
    # rank budget across individual experts using the blending score
    # s_i = beta_i^alpha * (log(e + eps*_i))^(1-alpha), where beta_i = spectral
    # energy proportion and eps*_i = reconstruction error at a reference rank.
    # alpha balances the two signals.
    #
    # Primary path (paper-exact): when validation_samples > 0, select the
    # global alpha by factoring the full model at each candidate's ranks and
    # validating WikiText-2 PPL end-to-end.
    #
    # Fallback (spectral proxy): when validation_samples = 0, select alpha by
    # minimising total activation-weighted tail spectral energy (no forward
    # passes, seconds).
    svd_plus_cfg = s3.get("swift_svd_plus", {})
    alpha_grid = svd_plus_cfg.get("alpha_grid")
    validation_samples = int(svd_plus_cfg.get("validation_samples", 0))

    # Snapshot originals to CPU BEFORE alpha search. Used by both the
    # validation-based alpha search (factor -> eval -> restore cycle) and
    # Stage 4 EoRA residual computation. Moving it here means both
    # consumers share one snapshot and the factoring loop no longer
    # needs to build it inline.
    #
    # Memory: ~50 GB CPU RAM for Qwen3.6-35B-A3B post-prune.
    # H200 has 256 GB host RAM; A-cov (~68 GB) + originals (~50 GB)
    # + OS (~10 GB) ~= ~128 GB -> ~128 GB headroom.
    originals = _snapshot_originals(moe_layers)
    log.info("Snapshotted %d original expert matrices to CPU for "
             "alpha-search and Stage 4 residuals", len(originals))

    # Persist originals before Phase D. Stage 4 reads this file, so it must
    # exist even if Stage 3 crashes mid-factoring. Written once here so
    # Stage 4 can always resume cleanly from the correct original weights.
    #
    # F-S3-1 fix: previously a bare in-place torch.save with no tmp+rename,
    # no fsync, no manifest. A SIGKILL mid-write (~50 GB on
    # Qwen3.6-35B-A3B) left a TRUNCATED .pt at the final path that Stage 4
    # would either error on (loud) OR partially load (silent corruption if
    # a future Stage 4 ever adds a `.get(..., 0)` fallback).
    #
    # Now: atomic_torch_save (tmp + fsync + os.replace + fsync(parent)) +
    # write_manifest_last so Stage 4 keys on the manifest's existence,
    # NOT the .pt's. A torn write produces a .pt without its
    # MANIFEST.json sibling — Stage 4 fails loudly with an actionable
    # message instead of consuming the partial file.
    from moe_compress.utils.atomic_io import atomic_torch_save, write_manifest_last
    _orig_path = artifacts_dir / "_stage3_original_weights.pt"
    # LOW-5: manifest naming consistency — append ``.MANIFEST.json`` AFTER
    # the payload suffix so the manifest sorts alphabetically right after
    # the .pt (Pattern O: manifest-LAST upload ordering relies on this).
    # F-RK-1 already uses this convention.
    _orig_manifest_path = artifacts_dir / "_stage3_original_weights.pt.MANIFEST.json"
    # If a previous run left a stale manifest, remove it first so an
    # interrupted re-write here doesn't briefly look "good" to Stage 4.
    try:
        _orig_manifest_path.unlink(missing_ok=True)
    except OSError:
        pass
    atomic_torch_save(_orig_path, originals)
    # Manifest is written LAST, after the .pt is fsync'd. Stage 4 reads
    # the manifest first; a missing/mismatched manifest means the .pt
    # is torn and must be re-captured.
    write_manifest_last(
        _orig_path,
        _orig_manifest_path,
        schema_version=1,
        extra_meta={
            "n_matrices": len(originals),
            "artifact": "stage3_original_weights",
        },
        # 50 GB SHA-256 on resume would add minutes — size + schema is
        # the validation budget; sha256 here is computed once at write
        # time and stored for opt-in deep validation by operators.
        compute_sha256=True,
    )
    log.info("Saved Stage 3 original weights snapshot (%d matrices) -> %s "
             "(manifest -> %s)",
             len(originals), _orig_path, _orig_manifest_path)

    # Pre-flight RAM check: paper-compliance contract (spec section 6 Phase B.2)
    # requires the end-to-end PPL alpha-search per Swift-SVD section 3.2.2. If
    # host RAM cannot host the snapshot + eval working set, fail fast rather
    # than silently degrade to a spectral proxy that produces a
    # non-paper-compliant model. Operators must provision >=15 GB headroom or
    # reduce validation_samples to fit.
    if validation_samples > 0:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "Stage 3 alpha-search requires psutil for the host-RAM "
                "pre-flight check (spec section 6 Phase B.2 paper-compliance "
                "contract). Install psutil or set "
                "stage3_svd.validation_samples=0 to skip the alpha-search "
                "entirely."
            ) from exc
        avail_gb = psutil.virtual_memory().available / 1e9
        # Default headroom (15 GB) is sized for production: ~50 GB snapshot in
        # CPU RAM + ~5 GB B-cov per layer + ~5 GB eval working set on a 30 B
        # base model. Smoke tests on toy models override via swift_svd_plus
        # config to skip the gate while still exercising the alpha-search path.
        min_headroom_gb = float(svd_plus_cfg.get("alpha_search_min_host_ram_gb", 15.0))
        if avail_gb < min_headroom_gb:
            raise RuntimeError(
                f"Stage 3 alpha-search: only {avail_gb:.1f} GB host RAM "
                f"available, need >={min_headroom_gb:.0f} GB headroom for the "
                f"paper-exact end-to-end PPL grid (spec section 6 Phase B.2). "
                f"Provision more RAM or reduce stage3_svd.validation_samples "
                f"to fit. The previous silent spectral-proxy fallback (D9) was "
                f"removed because it produced a non-paper-compliant model."
            )

    # Resume: if alpha was already selected (and saved) in a previous
    # interrupted run, reload it and skip the ~33 min alpha search entirely.
    _alpha_cache_path = artifacts_dir / "_stage3_alpha_result.json"
    _alpha_loaded = False
    alpha_by_type = None
    per_expert_ranks = None
    if not no_resume and _alpha_cache_path.exists() and alpha_grid and len(alpha_grid) > 1:
        try:
            _cached_alpha = load_json_artifact(_alpha_cache_path)
            alpha_by_type = _cached_alpha.get("alpha_by_type")
            if alpha_by_type is not None:
                log.info("Stage 3: loaded cached alpha from %s -- skipping alpha search",
                         _alpha_cache_path)
                per_expert_ranks = _redistribute_ranks_swift_svd_plus(
                    moe_layers, group_stats, ranks, alpha_by_type,
                    grouped_svs_cache=None, A_cov=A_cov,
                )
                _alpha_loaded = True
        except Exception as _exc:
            log.warning("Stage 3: failed to load alpha cache (%s) -- running alpha search", _exc)

    if not _alpha_loaded:
        # select_alpha phase -- SwiftSvdAlphaPlugin.select_alpha owns the
        # alpha-dispatch (validation vs spectral-proxy path + redistribution).
        # The originals snapshot + RAM pre-flight above and the alpha-cache I/O
        # below are run-glue. Publish the slots the hook reads, then walk.
        run_ctx.set("originals", originals)
        walk_phases(("select_alpha",), plugins, run_ctx)
        alpha_by_type = run_ctx.get("alpha_by_type")
        per_expert_ranks = run_ctx.get("per_expert_ranks")
    else:
        # alpha-cache hit: the select_alpha walk is skipped. Publish the
        # cache-derived values onto the ctx so the factor_layer loop reads
        # them through the parent chain.
        run_ctx.set("originals", originals)
        run_ctx.set("alpha_by_type", alpha_by_type)
        run_ctx.set("per_expert_ranks", per_expert_ranks)

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

    # Persist alpha result so a crash during Phase D doesn't force re-running
    # the ~33 min alpha search on resume.
    if not no_resume:
        save_json_artifact(
            {"alpha_by_type": alpha_by_type},
            artifacts_dir / "_stage3_alpha_result.json",
        )

    # ---- factor_layer (per-layer loop) ----------------------------------
    # The WHOLE per-layer factoring loop body lives in
    # AaSvdFactorPlugin.factor_layer. ``loop_over`` opens a fresh child ctx
    # per layer, binds the layer ref under "layer_ref", and walks the
    # factor_layer phase. The hook reads ranks / per_expert_ranks / A_cov /
    # B_acc / C_acc / B_cov_dtype / rank_map / device / originals through
    # the parent chain. rank_map is the single shared dict set on run_ctx --
    # every child mutates it in place (HAZARD H1).
    loop_over(moe_layers, plugins, ("factor_layer",), run_ctx, item_key="layer_ref")

    # 4. Phase C.5 -- block-level joint refinement (paper 2604.02119
    # section 3.3). Replaces the legacy per-matrix L-BFGS refine; trains the
    # block's FactoredExperts U/V slots and the two RMSNorm scales jointly via
    # AdamW (fp32) on the anchored MSE objective. BlockRefinePlugin gates on
    # stage3_svd.block_refine.enabled -- when off it is not in ``plugins`` and
    # this walk is a byte-identical no-op.
    #
    # V2 cache-first: try the block-hidden sidecar via dispatch_first on the
    # Stage3BlockHiddenCacheProvider. On hit, populates
    # ``ctx.teacher_targets_cache`` so block_refine's per-layer loop can
    # consume the cached teacher targets and skip the live teacher block
    # forward. On miss (no sidecars or token-count mismatch), ctx is
    # untouched and block_refine falls through to the live teacher forward.
    # Provide_block_targets is a Stage-3-local phase name that ONLY the
    # block-hidden cache provider implements; the registry's
    # ``dispatch_first`` returns the first non-None payload winner.
    if bool(s3.get("block_refine", {}).get("enabled", False)):
        try:
            from pathlib import Path as _Path
            from ..utils.calibration import _DEFAULT_SELF_TRACES_PATH
            _cal_source = cal.get("jsonl_path", _DEFAULT_SELF_TRACES_PATH)
            _cal_jsonl_path = _Path(_cal_source)
            if not _cal_jsonl_path.is_absolute():
                _cal_jsonl_path = _Path.cwd() / _cal_jsonl_path
            _bh_providers = [Stage3BlockHiddenCacheProvider()]
            PluginRegistry.dispatch_first(
                _bh_providers, "on_load", run_ctx, _cal_jsonl_path,
            )
            if run_ctx.has("teacher_targets_cache"):
                log.info(
                    "Stage 3: block-hidden cache HIT "
                    "(%d layers) -- block_refine will skip the live "
                    "teacher block forward",
                    len(run_ctx.get("teacher_targets_cache")),
                )
        except (FileNotFoundError, OSError) as _exc:
            log.warning(
                "Stage 3: block-hidden cache lookup failed (%s) -- "
                "block_refine will fall through to the live teacher "
                "block forward.", _exc,
            )

    walk_phases(("refine_blocks",), plugins, run_ctx)

    # Free teacher after Phase C.5 (or after factoring if C.5 disabled and the
    # earlier branch left it resident -- defensive).
    if teacher_model is not None:
        teacher_model.to("cpu")
        del teacher_model, teacher_moe_layers
        torch.cuda.empty_cache()
        log.info("Stage 3: freed original model after Phase C.5")

    # ---- finalize --------------------------------------------------------
    rank_map = run_ctx.get("rank_map")
    out_dir = artifacts_dir / "stage3_svd"
    mio.save_compressed_checkpoint(
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
    log.info("Stage 3 complete -> %s", out_dir)
    return out_dir


__all__ = ["run"]
