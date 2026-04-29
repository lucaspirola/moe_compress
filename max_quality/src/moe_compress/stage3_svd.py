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
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .budget.solver import BudgetDecomposition
from .utils.activation_hooks import (
    InputCovarianceAccumulator,
    instrument_experts,
    run_calibration,
)
from .utils.calibration import build_calibration_tensor, iter_batches, spec_from_config
from .utils.model_io import (
    MATRIX_NAMES,
    FactoredExperts,
    MoELayerRef,
    build_banks,
    iter_moe_layers,
    load_model,
    save_compressed_checkpoint,
    save_json_artifact,
)
from .utils.futures import drain_done_futures as _drain_done_futures
from .utils.trackio_log import trackio_log as _trackio_log

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


@dataclass
class _GroupStats:
    d_out: int
    d_in: int
    n_experts: int
    singular_values_mean: torch.Tensor
    effective_rank: float
    omega: float


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
    decomposition: BudgetDecomposition,
    *,
    device=None,
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
    spec = spec_from_config(
        cal,
        num_sequences_override=s3["swift_svd_plus"]["validation_samples"],
        seed_offset=2,
    )
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

    if cross_cov_enabled:
        log.info("Stage 3: loading original model for cross-covariance dual-forward (Theorem 3.2)")
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
        C_acc = InputCovarianceAccumulator()
        C_acc.set_storage_dtype(B_cov_dtype)
        log.info("Stage 3: dual-forward covariance collection (B + cross-cov C), batch_size=%d",
                 bcov_batch_size)
    else:
        log.info("Stage 3: B-cov only (cross-covariance disabled), batch_size=%d",
                 bcov_batch_size)

    bcov_spill_dir = artifacts_dir / "_stage3_bcov_partial"
    bcov_spill_dir.mkdir(parents=True, exist_ok=True)
    ccov_spill_dir = artifacts_dir / "_stage3_ccov_partial" if C_acc is not None else None
    if ccov_spill_dir is not None:
        ccov_spill_dir.mkdir(parents=True, exist_ok=True)

    _collect_covariances(
        model, moe_layers, batches, B_acc, device=device,
        spill_dir=bcov_spill_dir,
        teacher_model=teacher_model,
        teacher_moe_layers=teacher_moe_layers,
        C_acc=C_acc,
        ccov_spill_dir=ccov_spill_dir,
    )

    # Free the teacher model after covariance collection — not needed for factoring.
    if teacher_model is not None:
        teacher_model.to("cpu")
        del teacher_model, teacher_moe_layers
        torch.cuda.empty_cache()
        log.info("Stage 3: freed original model after cross-covariance collection")

    # 1. Per-(layer, matrix) group stats and rank allocation.
    log.info("Stage 3: computing per-group stats over %d layers", len(moe_layers))
    group_stats: dict[tuple[int, str], _GroupStats] = {}
    for k, ref in enumerate(moe_layers):
        log.info("  group-stat layer %d/%d (idx=%d)", k + 1, len(moe_layers), ref.layer_idx)
        banks = build_banks(ref)
        for name in MATRIX_NAMES:
            group_stats[(ref.layer_idx, name)] = _group_stat(
                ref.num_routed_experts, banks[name]
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

    # Pre-flight RAM check: if the system is low on memory, fall back
    # to the spectral proxy rather than OOM during the α-search loop.
    if validation_samples > 0:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / 1e9
            # Estimate memory needed: one layer's B-cov (~5 GB) + eval
            # overhead (~5 GB) on top of what we already hold.
            min_headroom_gb = 15.0
            if avail_gb < min_headroom_gb:
                log.warning(
                    "Stage 3 α-search: only %.1f GB host RAM available "
                    "(need ≥%.0f GB headroom). Falling back to spectral "
                    "proxy for α selection.",
                    avail_gb, min_headroom_gb,
                )
                validation_samples = 0
        except ImportError:
            pass  # psutil unavailable — proceed optimistically

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

    # 3. Save originals for Stage 4.
    torch.save(originals, artifacts_dir / "_stage3_original_weights.pt")
    log.info("Saved Stage 3 original weights snapshot (%d matrices)", len(originals))

    # 4. Block refine (optional).
    if s3["block_refine"]["enabled"]:
        _per_matrix_refine(
            moe_layers, originals, A_cov,
            lbfgs_steps=s3["block_refine"]["lbfgs_steps"],
            lbfgs_history=s3["block_refine"]["lbfgs_history"],
            B_acc=B_acc, bcov_spill_dir=bcov_spill_dir,
        )

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


def _group_stat(n_experts: int, bank) -> _GroupStats:
    d_out, d_in = bank.shape()
    svs: list[torch.Tensor] = []
    for e in range(n_experts):
        W = bank.get(e).detach().to(torch.float32)
        s = torch.linalg.svdvals(W)
        svs.append(_pad(s, min(d_out, d_in)))
    mean_s = torch.stack(svs).mean(0)
    p = mean_s ** 2
    p = p / p.sum().clamp(min=1e-12)
    eff_rank = float(torch.exp(-(p * p.clamp(min=1e-12).log()).sum()).item())
    omega = n_experts * (d_out + d_in)
    return _GroupStats(d_out, d_in, n_experts, mean_s, eff_rank, omega)


def _pad(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.numel() >= n:
        return x[:n]
    return torch.cat([x, torch.zeros(n - x.numel(), device=x.device, dtype=x.dtype)])


def _compute_T_budget(group_stats: dict, svd_rank_ratio: float) -> int:
    """T_budget solved so that reconstructed weight cost is ~= (1 - svd_rank_ratio) · original."""
    total_full = 0
    costs: list[float] = []
    for g, s in group_stats.items():
        total_full += s.n_experts * s.d_out * s.d_in
        costs.append(s.n_experts * (s.d_out + s.d_in))
    target_params = total_full * (1.0 - svd_rank_ratio)
    avg_cost = np.mean(costs) if costs else 1.0
    return int(max(1, target_params / max(avg_cost, 1.0)))


def _d_rank_allocate(
    group_stats: dict,
    T_budget: int,
    proj_weights: dict[str, float] | None = None,
) -> dict:
    """Distribute T_budget rank across all (layer, matrix) groups.

    proj_weights biases the allocation toward specific projection types without
    changing the total budget — e.g. {"gate_proj": 1.75, "up_proj": 1.35,
    "down_proj": 0.35} gives gate/up more rank at down_proj's expense.
    """
    pw = proj_weights or {}

    def _weight(g, s):
        return math.sqrt(s.effective_rank / s.omega) * pw.get(g[1], 1.0)

    def _cap(s):
        return min(s.d_out, s.d_in) - 1

    denom = sum(_weight(g, s) for g, s in group_stats.items()) or 1.0
    raw: dict = {g: _weight(g, s) * T_budget / denom for g, s in group_stats.items()}
    out: dict = {g: max(1, min(int(round(raw[g])), _cap(s)))
                 for g, s in group_stats.items()}

    # Correction: rounding+clamping perturbs the total away from T_budget.
    # Redistribute the residual to under-allocated groups (those still below
    # their cap) up to a configurable tolerance.
    target = int(T_budget)
    actual = sum(out.values())
    diff = target - actual
    if diff != 0:
        # Sort by (cap_room when adding, allocated rank when subtracting) so
        # the largest groups absorb most of the correction proportionally.
        sign = 1 if diff > 0 else -1
        # Iterate while there's residual to assign and at least one group
        # can accept it. Bounded by T_budget iterations as a safety.
        for _ in range(abs(diff)):
            if sign > 0:
                # Pick the group with the largest fractional remainder that
                # still has room below its cap.
                cands = [
                    (raw[g] - out[g], g) for g, s in group_stats.items()
                    if out[g] < _cap(s)
                ]
                if not cands:
                    break
                cands.sort(reverse=True)
                _, g = cands[0]
                out[g] += 1
            else:
                # Pick the group with the smallest fractional remainder that
                # still has room above the floor (rank ≥ 1).
                cands = [(raw[g] - out[g], g) for g in group_stats if out[g] > 1]
                if not cands:
                    break
                cands.sort()
                _, g = cands[0]
                out[g] -= 1

    final_total = sum(out.values())
    drift = abs(final_total - target)
    if drift > 0:
        log.warning("D-Rank budget conservation: residual drift %d after "
                    "correction (target=%d, actual=%d) — bounded by per-group "
                    "rank caps", drift, target, final_total)
    elif diff != 0:
        log.info("D-Rank budget redistributed %+d ranks across groups "
                 "(target=%d, conserved)", diff, target)
    return out



def _snapshot_originals(
    moe_layers: list[MoELayerRef],
) -> dict[tuple[int, int, str], torch.Tensor]:
    """CPU snapshot of all expert weights.

    Used by (a) validation-based α search (factor → eval → restore) and
    (b) Stage 4 EoRA residual computation. Moved before the α search so
    both consumers share the same snapshot.

    Memory: ~50 GB CPU RAM for Qwen3.6-35B-A3B post-prune (~200 experts ×
    40 layers × 3 matrices × [512,2048] bf16). H200 has 256 GB host RAM;
    combined with A-cov (~68 GB) this leaves ~128 GB headroom.
    """
    originals: dict[tuple[int, int, str], torch.Tensor] = {}
    for ref in moe_layers:
        banks = build_banks(ref)
        for e in range(ref.num_routed_experts):
            for name in MATRIX_NAMES:
                originals[(ref.layer_idx, e, name)] = (
                    banks[name].get(e).detach().cpu().clone()
                )
    return originals


def _build_wikitext2_validation(
    tokenizer,
    n_seqs: int,
    seq_len: int = 2048,
) -> torch.LongTensor:
    """Build a WikiText-2 validation tensor for α search.

    Uses the standard WikiText-2 raw test set: concatenate with EOS
    between documents, chunk to fixed ``seq_len``, return the first
    ``n_seqs`` full-length chunks.

    This mirrors Stage 6's ``_wikitext2_ppl`` tokenization exactly so
    the α-search PPL and Stage 6's final PPL are directly comparable.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eos = tokenizer.eos_token_id or 0
    all_ids: list[int] = []
    for row in ds:
        text = row.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(eos)

    n_full = len(all_ids) // seq_len
    if n_full == 0:
        log.warning("WikiText-2 has no full-length sequences; α search will "
                    "fall back to spectral proxy.")
        return torch.empty(0, seq_len, dtype=torch.long)
    n_use = min(n_full, n_seqs)
    return torch.tensor(
        all_ids[: n_use * seq_len], dtype=torch.long,
    ).view(n_use, seq_len)


def _evaluate_wikitext2_ppl(
    model, val_tensor: torch.LongTensor, *, device, batch_size: int = 16,
) -> float:
    """Compute WikiText-2 perplexity on pre-tokenized sequences.

    Matches Stage 6's ``_wikitext2_ppl`` methodology: next-token NLL
    averaged over all non-first positions, then exp(mean_NLL).
    """
    if val_tensor.numel() == 0:
        return float("inf")
    model.eval()
    nll_sum = 0.0
    tok_count = 0
    for i in range(0, val_tensor.size(0), batch_size):
        batch = val_tensor[i : i + batch_size]
        if device is not None:
            batch = batch.to(device)
        with torch.no_grad():
            out = model(input_ids=batch, labels=batch)
        # out.loss is the mean NLL over (seq_len - 1) positions per sequence.
        n_tokens = batch.numel() - batch.shape[0]
        nll_sum += float(out.loss.item()) * n_tokens
        tok_count += n_tokens
    if tok_count == 0:
        return float("inf")
    return math.exp(nll_sum / tok_count)


def _factor_model_at_ranks(
    model,
    moe_layers: list[MoELayerRef],
    originals: dict[tuple[int, int, str], torch.Tensor],
    per_expert_ranks: dict[tuple[int, str, int], int],
    base_ranks: dict[tuple[int, str], int],
    A_cov: dict,
    B_acc,
    bcov_spill_dir: Path,
    C_acc,
    ccov_spill_dir: Path | None,
    *,
    device,
    storage_dtype: torch.dtype = torch.float16,
) -> None:
    """Factor all MoE layers in-place at the given per-expert ranks.

    Used by the validation-based α search: for each candidate α, this
    function installs FactoredExperts at the candidate's rank allocation.
    After evaluation, ``_restore_fused_experts`` reverses the swap.

    Covariance is lazy-loaded per layer from spill files and immediately
    unloaded, keeping in-memory footprint bounded to one layer (~5 GB).
    """
    for ref in moe_layers:
        # Load covariances for this layer.
        B_acc.load_layer_from_disk(ref.layer_idx, bcov_spill_dir)
        if C_acc is not None and ccov_spill_dir is not None:
            C_acc.load_layer_from_disk(ref.layer_idx, ccov_spill_dir)

        # Slot width = max per-expert rank within this layer/matrix.
        ranks_layer = {
            name: max(
                per_expert_ranks.get(
                    (ref.layer_idx, name, e),
                    base_ranks[(ref.layer_idx, name)],
                )
                for e in range(ref.num_routed_experts)
            )
            for name in MATRIX_NAMES
        }

        ex = ref.experts_module
        dtype = ex.gate_up_proj.dtype
        # Offload dense experts to CPU before allocating FactoredExperts
        # to avoid brief double-occupancy.
        ex.to("cpu")
        torch.cuda.empty_cache()

        new_factored = FactoredExperts(
            num_experts=ref.num_routed_experts,
            hidden_dim=ex.gate_up_proj.shape[-1],
            intermediate_dim=ex.gate_up_proj.shape[1] // 2,
            ranks=ranks_layer,
            dtype=dtype,
            device=device,
        )

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
                        device=device, storage_dtype=storage_dtype,
                    )
                except ValueError:
                    pass  # falls through to full _aa_svd below

            for name in MATRIX_NAMES:
                W = originals[(ref.layer_idx, e, name)].to(
                    device=device, dtype=torch.float32,
                )
                k = per_expert_ranks.get(
                    (ref.layer_idx, name, e),
                    base_ranks[(ref.layer_idx, name)],
                )
                if name in ("gate_proj", "up_proj") and gate_up_decomp is not None:
                    U_k, V_k, _, k_eff = _aa_svd_precomputed(
                        W, gate_up_decomp, k, device=device,
                    )
                else:
                    A = _cov_lookup(A_cov, ref.layer_idx, e, name)
                    B = _cov_lookup(B_acc.covariance, ref.layer_idx, e, name)
                    C = None
                    if C_acc is not None:
                        C = _cov_lookup(C_acc.covariance, ref.layer_idx, e, name)
                    U_k, V_k, _, k_eff = _aa_svd(
                        W, A, B, k, C=C, device=device,
                        storage_dtype=storage_dtype,
                    )
                new_factored.set_factors(
                    e, name, U_k, V_k, effective_rank=k_eff,
                )

        # Swap in.
        setattr(ref.mlp, "experts", new_factored)
        ref.experts_module = new_factored

        # Free this layer's covariance from memory.
        B_acc.unload_layer(ref.layer_idx)
        if C_acc is not None:
            C_acc.unload_layer(ref.layer_idx)


def _restore_fused_experts(
    model,
    moe_layers: list[MoELayerRef],
    originals: dict[tuple[int, int, str], torch.Tensor],
    *,
    device,
) -> None:
    """Restore original fused experts from the CPU snapshot.

    Reverses ``_factor_model_at_ranks``: replaces each layer's
    FactoredExperts with the original ``Qwen3_5MoeExperts`` fused module
    reconstructed from the ``originals`` dict.

    The fused module is rebuilt manually as a ``SimpleNamespace``-style
    module with the correct ``gate_up_proj`` and ``down_proj`` stacked
    tensors that ``build_banks`` expects.
    """
    for ref in moe_layers:
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        dtype = fe.gate_proj_U.dtype
        n = ref.num_routed_experts
        d_int = fe.intermediate_dim
        d_hid = fe.hidden_dim

        # Free FactoredExperts from GPU.
        fe.to("cpu")
        torch.cuda.empty_cache()

        # Rebuild fused storage on GPU from CPU originals.
        gate_up = torch.zeros(n, 2 * d_int, d_hid, dtype=dtype, device=device)
        down = torch.zeros(n, d_hid, d_int, dtype=dtype, device=device)

        for e in range(n):
            gate_w = originals[(ref.layer_idx, e, "gate_proj")].to(
                dtype=dtype, device=device,
            )
            up_w = originals[(ref.layer_idx, e, "up_proj")].to(
                dtype=dtype, device=device,
            )
            down_w = originals[(ref.layer_idx, e, "down_proj")].to(
                dtype=dtype, device=device,
            )
            gate_up[e, :d_int] = gate_w
            gate_up[e, d_int:] = up_w
            down[e] = down_w

        # Reconstruct the fused experts module. We need a module that
        # ``build_banks`` / ``_is_fused_experts`` recognises: it must have
        # ``gate_up_proj`` and ``down_proj`` as Parameters and a
        # ``num_experts`` attribute.
        fused = nn.Module()
        fused.gate_up_proj = nn.Parameter(gate_up, requires_grad=False)
        fused.down_proj = nn.Parameter(down, requires_grad=False)
        fused.num_experts = n
        # Copy the act_fn and forward from the original class if available.
        # Not strictly necessary — we only need the weights accessible via
        # build_banks for the final factoring loop. The forward will be
        # replaced when the final FactoredExperts is installed.
        from transformers.activations import ACT2FN
        fused.act_fn = ACT2FN["silu"]

        setattr(ref.mlp, "experts", fused)
        ref.experts_module = fused


def _swift_svd_plus_alpha_search_validation(
    model,
    tokenizer,
    moe_layers: list[MoELayerRef],
    group_stats: dict[tuple[int, str], _GroupStats],
    base_ranks: dict[tuple[int, str], int],
    alpha_grid: list[float],
    originals: dict[tuple[int, int, str], torch.Tensor],
    A_cov: dict,
    B_acc,
    bcov_spill_dir: Path,
    C_acc,
    ccov_spill_dir: Path | None,
    config: dict,
    *,
    device,
) -> float:
    """Paper-exact α selection via end-to-end WikiText-2 PPL validation.

    Implements Swift-SVD (2604.01609) §3.2.2:

        "Swift-SVD uses a fixed retention ratio δ=0.5 and 11 scaling
        factors α=[0, 0.1, 0.2, ..., 1] to generate 11 candidate rank
        allocations. For each candidate corresponding to α_i, the optimal
        low-rank approximation of every layer is computed using the
        closed-form solution in (3). The resulting compressed models are
        then evaluated on a validation set, and the candidate that yields
        the best end-to-end performance is selected."

    For each α in ``alpha_grid``:
      1. Compute per-expert rank redistribution (Algorithm 2 blending score)
      2. Factor the full model layer-by-layer via AA-SVD at those ranks
      3. Evaluate WikiText-2 PPL on ``validation_samples`` sequences
      4. Restore original fused experts from CPU snapshot

    Returns the α with the lowest PPL.

    **Cost**: 11 candidates × (~2 min factor + ~0.3 min eval + ~0.5 min
    restore) ≈ ~31 min on H200 for Qwen3.6-35B-A3B.

    **Memory**: CPU RAM holds originals (~50 GB) + A-cov (~68 GB) ≈ ~128 GB.
    H200 has 256 GB host RAM → ~128 GB headroom. VRAM holds the factored
    model (~34 GB) during eval → ~107 GB headroom on 141 GB.
    """
    svd_plus_cfg = config["stage3_svd"]["swift_svd_plus"]
    validation_samples = int(svd_plus_cfg.get("validation_samples", 512))
    validation_batch_size = int(svd_plus_cfg.get("validation_batch_size", 16))

    # Build WikiText-2 validation tensor (same tokenization as Stage 6).
    log.info("Stage 3 α-search: building WikiText-2 validation set "
             "(%d sequences, seq_len=2048)", validation_samples)
    val_tensor = _build_wikitext2_validation(
        tokenizer, n_seqs=validation_samples, seq_len=2048,
    )
    if val_tensor.numel() == 0:
        log.warning("Stage 3 α-search: empty validation set — falling back "
                    "to spectral proxy.")
        return 0.5  # neutral fallback

    log.info("Stage 3 α-search: %d validation sequences (%d tokens)",
             val_tensor.size(0), val_tensor.numel())

    best_alpha = 0.5
    best_ppl = float("inf")
    results: list[tuple[float, float]] = []

    for idx, alpha in enumerate(alpha_grid):
        log.info("Stage 3 α-search: candidate %d/%d (α=%.1f)",
                 idx + 1, len(alpha_grid), alpha)

        # 1. Compute per-expert ranks for this α (single α for all types).
        alpha_by_type = {"all": alpha}
        per_expert_ranks = _redistribute_ranks_swift_svd_plus(
            moe_layers, group_stats, base_ranks, alpha_by_type,
            A_cov=A_cov,
        )

        # 2. Factor the full model at these ranks.
        _factor_model_at_ranks(
            model, moe_layers, originals, per_expert_ranks, base_ranks,
            A_cov, B_acc, bcov_spill_dir, C_acc, ccov_spill_dir,
            device=device,
        )

        # 3. Evaluate WikiText-2 PPL.
        ppl = _evaluate_wikitext2_ppl(
            model, val_tensor, device=device,
            batch_size=validation_batch_size,
        )
        results.append((alpha, ppl))
        log.info("  α=%.1f → WikiText-2 PPL=%.4f", alpha, ppl)
        _trackio_log({
            "stage3/alpha_search/alpha": alpha,
            "stage3/alpha_search/ppl": ppl,
        })

        # 4. Restore original fused experts for the next candidate.
        _restore_fused_experts(model, moe_layers, originals, device=device)

        if ppl < best_ppl:
            best_ppl = ppl
            best_alpha = alpha

    log.info("Stage 3 α-search complete: best α=%.1f (PPL=%.4f)", best_alpha, best_ppl)
    log.info("  full results: %s",
             ", ".join(f"α={a:.1f}→{p:.4f}" for a, p in results))
    _trackio_log({
        "stage3/alpha_search/best_alpha": best_alpha,
        "stage3/alpha_search/best_ppl": best_ppl,
    })
    return best_alpha


def _swift_svd_plus_alpha_search(
    moe_layers: list,
    group_stats: dict[tuple[int, str], _GroupStats],
    base_ranks: dict[tuple[int, str], int],
    alpha_grid: list[float],
    *,
    per_group_type: bool = True,
    A_cov: dict | None = None,
) -> dict[str, float]:
    """Swift-SVD+ (2604.01609, Algorithm 2): select α per projection type.

    For each candidate α, compute the blending score for every expert within
    each (layer, matrix_type) group:

        s_i = β_i^α · (log(e + ε*_i))^{1-α}

    where:
      - β_i = σ_i² / Σ_j σ_j²  (spectral energy proportion — how much of the
        group's total spectral energy this expert contributes)
      - ε*_i = √(Σ_{j>k̄} σ_j² / Σ_j σ_j²)  (reconstruction error at the
        group's mean rank k̄ — higher = this expert needs more rank)

    Then redistribute the group's total rank budget proportionally to s_i.
    The α that minimises the total weighted reconstruction error across all
    experts in the group wins.

    Returns {matrix_type: best_α} if per_group_type, else {"all": best_α}.
    """
    import math as _math

    # Collect per-expert singular value spectra, grouped by matrix type.
    # When A_cov is available (D8 fix), compute activation-weighted SVD
    # (SVD of W @ L_A) instead of raw SVD. This gives ε* that reflects
    # actual reconstruction error weighted by input distribution.
    # grouped_svs[name][(layer_idx, expert_idx)] = singular_values tensor
    grouped_svs: dict[str, dict[tuple[int, int], torch.Tensor]] = {
        n: {} for n in MATRIX_NAMES
    }
    for (li, name), gs in group_stats.items():
        banks = build_banks([ref for ref in moe_layers if ref.layer_idx == li][0])
        for e in range(gs.n_experts):
            W = banks[name].get(e).detach().to(torch.float32)
            # D8 fix: activation-weighted singular values when A_cov available.
            A = _cov_lookup(A_cov, li, e, name) if A_cov else None
            if A is not None:
                A_f32 = A.to(torch.float32)
                A_f32 = 0.5 * (A_f32 + A_f32.T)
                eigvals_a, eigvecs_a = torch.linalg.eigh(A_f32)
                keep_a = eigvals_a > eigvals_a.max() * 1e-6
                if keep_a.any():
                    L_A = eigvecs_a[:, keep_a] * eigvals_a[keep_a].clamp_min(1e-12).sqrt().unsqueeze(0)
                    M_A = W @ L_A
                    svs = torch.linalg.svdvals(M_A)
                else:
                    svs = torch.linalg.svdvals(W)
            else:
                svs = torch.linalg.svdvals(W)
            grouped_svs[name][(li, e)] = svs

    def _evaluate_alpha(name: str, alpha: float) -> float:
        """Total weighted reconstruction error for this α across all experts
        in the given projection type."""
        group_keys = [(li, n) for (li, n) in base_ranks if n == name]
        total_err = 0.0
        for (li, n) in group_keys:
            gs = group_stats[(li, n)]
            k_group = base_ranks[(li, n)]
            # Collect per-expert scores.
            expert_ids = list(range(gs.n_experts))
            betas: list[float] = []
            epsilons: list[float] = []
            energies: list[float] = []
            for e in expert_ids:
                svs = grouped_svs[n][(li, e)]
                s2 = (svs * svs)
                total_energy = float(s2.sum().clamp_min(1e-30).item())
                energies.append(total_energy)
                # ε*_i at reference rank k_group
                tail = float(s2[k_group:].sum().item()) if k_group < len(s2) else 0.0
                epsilons.append((tail / total_energy) ** 0.5)
            # β_i = energy_i / total_energy_in_group
            group_energy = sum(energies) or 1.0
            betas = [e_val / group_energy for e_val in energies]
            # Blending scores
            scores = []
            for beta, eps in zip(betas, epsilons):
                s = (beta ** alpha) * (_math.log(_math.e + eps) ** (1.0 - alpha))
                scores.append(max(s, 1e-12))
            # Redistribute group rank budget proportionally to scores.
            total_score = sum(scores) or 1.0
            total_group_rank = k_group * gs.n_experts
            per_expert_ranks = [
                max(1, min(min(gs.d_out, gs.d_in) - 1,
                           int(round(total_group_rank * (sc / total_score)))))
                for sc in scores
            ]
            # Evaluate: sum of tail energy at allocated rank per expert.
            for e, k_e in zip(expert_ids, per_expert_ranks):
                svs = grouped_svs[n][(li, e)]
                s2 = svs * svs
                tail = float(s2[k_e:].sum().item()) if k_e < len(s2) else 0.0
                total_err += tail
        return total_err

    if per_group_type:
        best_alphas: dict[str, float] = {}
        for name in MATRIX_NAMES:
            best_alpha = 0.5
            best_err = float("inf")
            for alpha in alpha_grid:
                err = _evaluate_alpha(name, alpha)
                if err < best_err:
                    best_err = err
                    best_alpha = alpha
            best_alphas[name] = best_alpha
            log.info("  Swift-SVD+ %s: best α=%.1f (err=%.4e)", name, best_alpha, best_err)
        return best_alphas
    else:
        best_alpha = 0.5
        best_err = float("inf")
        for alpha in alpha_grid:
            err = sum(_evaluate_alpha(n, alpha) for n in MATRIX_NAMES)
            if err < best_err:
                best_err = err
                best_alpha = alpha
        log.info("  Swift-SVD+ global: best α=%.1f (err=%.4e)", best_alpha, best_err)
        return {"all": best_alpha}


def _redistribute_ranks_swift_svd_plus(
    moe_layers: list,
    group_stats: dict[tuple[int, str], _GroupStats],
    base_ranks: dict[tuple[int, str], int],
    alpha_by_type: dict[str, float],
    *,
    grouped_svs_cache=None,
    A_cov: dict | None = None,
) -> dict[tuple[int, str, int], int]:
    """Given the selected α per type, compute per-expert ranks.

    Returns {(layer_idx, matrix_name, expert_idx): rank}.
    The total rank within each (layer, matrix_type) group is conserved
    (sum of per-expert ranks = base_rank × n_experts).
    """
    import math as _math

    out: dict[tuple[int, str, int], int] = {}
    for (li, name), gs in group_stats.items():
        k_group = base_ranks[(li, name)]
        alpha = alpha_by_type.get(name, alpha_by_type.get("all", 0.5))

        # Collect per-expert singular values (activation-weighted when A_cov available).
        banks = build_banks([ref for ref in moe_layers if ref.layer_idx == li][0])
        energies: list[float] = []
        epsilons: list[float] = []
        for e in range(gs.n_experts):
            W = banks[name].get(e).detach().to(torch.float32)
            # D8 fix: activation-weighted SVD when A_cov available.
            A = _cov_lookup(A_cov, li, e, name) if A_cov else None
            if A is not None:
                A_f32 = A.to(torch.float32)
                A_f32 = 0.5 * (A_f32 + A_f32.T)
                eigvals_a, eigvecs_a = torch.linalg.eigh(A_f32)
                keep_a = eigvals_a > eigvals_a.max() * 1e-6
                if keep_a.any():
                    L_A = eigvecs_a[:, keep_a] * eigvals_a[keep_a].clamp_min(1e-12).sqrt().unsqueeze(0)
                    svs = torch.linalg.svdvals(W @ L_A)
                else:
                    svs = torch.linalg.svdvals(W)
            else:
                svs = torch.linalg.svdvals(W)
            s2 = svs * svs
            total_e = float(s2.sum().clamp_min(1e-30).item())
            energies.append(total_e)
            tail = float(s2[k_group:].sum().item()) if k_group < len(s2) else 0.0
            epsilons.append((tail / total_e) ** 0.5)

        group_energy = sum(energies) or 1.0
        betas = [e_val / group_energy for e_val in energies]
        scores = [
            max((b ** alpha) * (_math.log(_math.e + eps) ** (1.0 - alpha)), 1e-12)
            for b, eps in zip(betas, epsilons)
        ]
        total_score = sum(scores) or 1.0
        total_group_rank = k_group * gs.n_experts
        cap = min(gs.d_out, gs.d_in) - 1

        per_e = [
            max(1, min(cap, int(round(total_group_rank * (sc / total_score)))))
            for sc in scores
        ]
        # Reconcile rounding residual.
        diff = total_group_rank - sum(per_e)
        if diff != 0:
            order = sorted(range(gs.n_experts),
                           key=lambda i: scores[i], reverse=(diff > 0))
            for idx in order:
                if diff == 0:
                    break
                step = 1 if diff > 0 else -1
                new_val = per_e[idx] + step
                if 1 <= new_val <= cap:
                    per_e[idx] = new_val
                    diff -= step

        for e, k_e in enumerate(per_e):
            out[(li, name, e)] = k_e
    return out

# ---------------------------------------------------------------------------
# AA-SVD per matrix
# ---------------------------------------------------------------------------


_NOISE_FLOOR_BY_DTYPE: dict[torch.dtype, float] = {
    # Relative threshold above which an eigenvalue of B is considered signal
    # rather than storage-quantization noise. Driven by the storage dtype's
    # mantissa bits: bf16 has 7 (~2⁻⁷ ≈ 8e-3 noise), fp16 has 10 (~2⁻¹⁰ ≈ 1e-3),
    # fp32 has 23 (~2⁻²³). Set the floor a small margin above noise to ensure
    # we don't keep noise-inflated directions.
    torch.bfloat16: 1e-2,
    torch.float16:  1e-3,
    torch.float32:  1e-6,
    torch.float64:  1e-12,
}


@dataclass
class _EighDecomp:
    """Cached eigendecomposition of a covariance matrix B, plus the
    pre-multiplied right-hand side for the M = W @ rhs formulation.

    This allows gate_proj and up_proj — which share the same B and C
    covariance (``_cov_lookup`` falls back from up_proj to gate_proj) —
    to skip the redundant ``eigh(B)`` call and the ``C @ Q`` or ``A @ Q``
    product.  The only per-matrix work that remains is ``W @ rhs`` and
    the subsequent SVD + back-solve.

    Attributes:
        eigvals_keep: Eigenvalues of B above the noise floor, clamped ≥0.  [r_eff]
        eigvecs_keep: Corresponding eigenvectors.                          [d_in, r_eff]
        inv_sqrt:     1/√(eigvals_keep), for the back-solve.              [r_eff]
        rhs:          The right-hand-side matrix such that M = W @ rhs.
                      Shape [d_in, r_eff].  Content depends on the path:
                      - Path 1 (Theorem 3.2): CQ · diag(1/√λ)
                      - Path 2 (auto-cov):    AQ · diag(1/√λ)
                      - Path 3 (Cor. 3.3):    L_B = Q · diag(√λ)
        r_eff:        Number of retained eigenvalues (= rhs.shape[1]).
    """
    eigvals_keep: torch.Tensor
    eigvecs_keep: torch.Tensor
    inv_sqrt: torch.Tensor
    rhs: torch.Tensor
    r_eff: int


def _precompute_eigh(
    B: torch.Tensor,
    A: torch.Tensor | None,
    C: torch.Tensor | None,
    *,
    device,
    storage_dtype: torch.dtype | None = None,
) -> _EighDecomp:
    """Eigendecompose B and build the right-hand-side matrix for AA-SVD.

    This is the expensive part of ``_aa_svd`` that depends only on the
    covariance matrices (B, A, C) and NOT on the weight matrix W.  Since
    gate_proj and up_proj share the same B and C (via ``_cov_lookup``
    fallback), callers can call this once per expert and reuse the result
    for both projections — eliminating one ``eigh(2048×2048)`` call per
    expert.

    Raises ``ValueError`` if B has no positive eigenvalues above the noise
    floor (same behaviour as ``_aa_svd``).
    """
    B = B.to(device=device, dtype=torch.float32)
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = torch.linalg.eigh(B)                     # ascending
    sigma_max = float(eigvals[-1].clamp_min(0).item())
    rel_floor = _NOISE_FLOOR_BY_DTYPE.get(storage_dtype or torch.float32, 1e-6)
    thresh = max(sigma_max * rel_floor, 1e-12)
    keep = eigvals > thresh
    r_eff = int(keep.sum().item())
    if r_eff == 0:
        raise ValueError("B has no positive eigenvalues above threshold")
    eigvals_keep = eigvals[keep].clamp_min(0)
    eigvecs_keep = eigvecs[:, keep]
    inv_sqrt = eigvals_keep.clamp_min(1e-30).rsqrt()             # [r_eff]

    if C is not None:
        # Path 1: Paper-exact Theorem 3.2 — rhs = C @ Q · diag(1/√λ)
        C = C.to(device=device, dtype=torch.float32)
        CQ = C @ eigvecs_keep                                    # [d_in, r_eff]
        rhs = CQ * inv_sqrt.unsqueeze(0)                         # [d_in, r_eff]
    elif A is not None:
        # Path 2: Auto-covariance — rhs = A @ Q · diag(1/√λ)
        A = A.to(device=device, dtype=torch.float32)
        A = 0.5 * (A + A.T)
        AQ = A @ eigvecs_keep                                    # [d_in, r_eff]
        rhs = AQ * inv_sqrt.unsqueeze(0)                         # [d_in, r_eff]
    else:
        # Path 3: Corollary 3.3 — rhs = L_B = Q · diag(√λ)
        rhs = eigvecs_keep * eigvals_keep.sqrt().unsqueeze(0)    # [d_in, r_eff]

    return _EighDecomp(
        eigvals_keep=eigvals_keep,
        eigvecs_keep=eigvecs_keep,
        inv_sqrt=inv_sqrt,
        rhs=rhs,
        r_eff=r_eff,
    )


def _aa_svd_precomputed(
    W: torch.Tensor,
    decomp: _EighDecomp,
    k: int,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Rank-k factorization of W using a pre-computed eigendecomposition.

    Mathematically identical to ``_aa_svd`` — the only difference is that
    the eigendecomposition of B and the rhs product (CQ·inv_sqrt, AQ·inv_sqrt,
    or L_B) are supplied via ``decomp`` rather than recomputed.

    Returns (U_k, V_k, rel_err, k_eff).
    """
    d_out, d_in = W.shape
    k = max(1, min(k, min(d_out, d_in) - 1))
    try:
        M = W @ decomp.rhs                                      # [d_out, r_eff]

        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        k_eff = max(1, min(k, decomp.r_eff))
        U_eff = U[:, :k_eff] * S[:k_eff]
        # Back-solve: V = Vh[:k_eff] @ diag(1/√λ) @ Q^T
        V_eff = (Vh[:k_eff, :] * decomp.inv_sqrt.unsqueeze(0)) @ decomp.eigvecs_keep.T  # [k_eff, d_in]
        # Numerically stable rel_err: tail singular values of M.
        S2 = S * S
        denom = S2.sum().clamp_min(1e-30)
        if k_eff < S2.numel():
            rel_err = float((S2[k_eff:].sum() / denom).sqrt().item())
        else:
            rel_err = 0.0
        # Always return shape [d_out, k] / [k, d_in] — caller's FactoredExperts
        # slot is pre-allocated at `k`. Zero-pad when effective rank < k.
        if k_eff < k:
            U_k = torch.zeros(d_out, k, device=device, dtype=U_eff.dtype)
            V_k = torch.zeros(k, d_in, device=device, dtype=V_eff.dtype)
            U_k[:, :k_eff] = U_eff
            V_k[:k_eff, :] = V_eff
        else:
            U_k, V_k = U_eff, V_eff
    except Exception as err:                         # noqa: BLE001
        log.warning("AA-SVD (precomputed) fallback to plain SVD (%s)", err)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = Vh[:k, :]
        with torch.no_grad():
            R = W - U_k @ V_k
            w_norm = W.norm()
            rel_err = float((R.norm() / w_norm).item()) if w_norm > 0 else 0.0
        k_eff = k
    return U_k, V_k, rel_err, k_eff


def _aa_svd(
    W: torch.Tensor,
    A: torch.Tensor | None,
    B: torch.Tensor | None,
    k: int,
    *,
    C: torch.Tensor | None = None,
    device,
    storage_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Activation-aware rank-k factorization of W.

    Three paths, in priority order:

    1. **Paper-exact (Theorem 3.2)**: when cross-covariance C = X_pre^T X_post
       and B = X_post^T X_post are both available:
         M = W · C · B^{-1} · L_B
       where L_B satisfies B = L_B · L_B^T. This is the exact AA-SVD solution
       that anchors to original outputs while adapting to shifted inputs.

    2. **Auto-covariance approximation**: when A = X_pre^T X_pre and B are
       available but C is not:
         M = W · A · B^{-1} · L_B
       Substitutes pre-prune auto-covariance for cross-covariance. The two
       coincide when pre/post distributions are similar (light pruning).

    3. **Corollary 3.3 fallback**: when only B is available:
         M = W · L_B
       Shift-aware variant that adapts to post-prune distribution only.

    Returns (U_k, V_k, rel_err, k_eff).

    .. note::

       When factoring both gate_proj and up_proj for the same expert, prefer
       ``_precompute_eigh`` + ``_aa_svd_precomputed`` to avoid the redundant
       ``eigh(B)`` call — gate_proj and up_proj share the same B and C
       covariance via ``_cov_lookup`` fallback.
    """
    d_out, d_in = W.shape
    k = max(1, min(k, min(d_out, d_in) - 1))
    try:
        decomp = _precompute_eigh(B, A, C, device=device, storage_dtype=storage_dtype)
        return _aa_svd_precomputed(W, decomp, k, device=device)
    except Exception as err:                         # noqa: BLE001
        log.warning("AA-SVD fallback to plain SVD (%s)", err)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = Vh[:k, :]
        with torch.no_grad():
            R = W - U_k @ V_k
            w_norm = W.norm()
            rel_err = float((R.norm() / w_norm).item()) if w_norm > 0 else 0.0
        k_eff = k
    return U_k, V_k, rel_err, k_eff


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

    n = len(moe_layers)
    try:
        for k, ref in enumerate(moe_layers):
            if spill_dir is not None:
                existing = (spill_dir / f"layer_{ref.layer_idx}.pt").exists()
                if existing:
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


def _load_stage2_covariance(path: Path):
    if not path.exists():
        log.warning("Stage 2 covariance not found at %s — AA-SVD fallback", path)
        return {}
    payload = torch.load(path, map_location="cpu")
    return payload.get("covariance", {})


def _cov_lookup(cov: dict, layer_idx: int, expert_idx: int, matrix_name: str):
    """Bank-aware lookup: falls back to gate_proj when asked for up_proj."""
    key = (layer_idx, expert_idx, matrix_name)
    if key in cov:
        return cov[key]
    if matrix_name == "up_proj":
        return cov.get((layer_idx, expert_idx, "gate_proj"))
    return None


# ---------------------------------------------------------------------------
# Per-matrix L-BFGS reconstruction refine
# ---------------------------------------------------------------------------


def _per_matrix_refine(
    moe_layers: list[MoELayerRef],
    originals: dict[tuple[int, int, str], torch.Tensor],
    A_cov: dict,
    *,
    lbfgs_steps: int,
    lbfgs_history: int,
    B_acc=None,
    bcov_spill_dir: Path | None = None,
) -> None:
    # The L-BFGS objective here is the A-weighted (pre-prune input cov)
    # reconstruction loss ‖(W−UV)·A^{1/2}‖_F², which is *not* the same as
    # the B-weighted (post-prune input cov) loss ‖(W−UV)·L_B‖_F² that
    # ``_aa_svd`` minimised. A and B both live in input space [d_in, d_in]
    # but capture different signal distributions (full pre-prune vs. the
    # subset that survives Stage 2 expert merge). The two norms can
    # disagree; refine can in principle improve the A-weighted metric
    # while slightly degrading the B-weighted metric. We compute the
    # B-weighted residual pre/post-refine per (layer, matrix) and warn if
    # any matrix regresses, so silent quality loss in the B-weighted norm
    # is observable in trackio rather than invisible.
    log.info(
        "Stage 3.D: activation-weighted refine (%d layers × 3 matrices × N experts)",
        len(moe_layers),
    )
    bw_check = B_acc is not None and bcov_spill_dir is not None
    for ref in moe_layers:
        # Lazy-load this layer's B-cov for the pre/post B-weighted check.
        # Same pattern as the factor loop: load → use → unload to keep the
        # in-memory cov bounded to ~one layer.
        if bw_check:
            loaded = B_acc.load_layer_from_disk(ref.layer_idx, bcov_spill_dir)
            if not loaded:
                log.warning(
                    "Refine: B-cov spill missing for layer %d at %s; "
                    "skipping B-weighted regression check for this layer.",
                    ref.layer_idx, bcov_spill_dir,
                )
        fe: FactoredExperts = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        # Aggregate convergence: sum of (initial_loss, final_loss) over experts
        # for this layer × matrix. Per-expert is too noisy for the dashboard.
        layer_init: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        layer_final: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        layer_count: dict[str, int] = {n: 0 for n in MATRIX_NAMES}
        # B-weighted residual tr((W−UV) B (W−UV)^T) = ((R @ B) * R).sum(),
        # pre/post-refine. Same trace form as the A-weighted line; B is the
        # post-prune *input* covariance ([d_in, d_in]), the same B that
        # ``_aa_svd`` minimised against — not an output-side cov.
        bw_init: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        bw_final: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        bw_count: dict[str, int] = {n: 0 for n in MATRIX_NAMES}
        for e in range(fe.num_experts):
            for name in MATRIX_NAMES:
                key = (ref.layer_idx, e, name)
                A = _cov_lookup(A_cov, ref.layer_idx, e, name)
                if A is None:
                    continue
                W = originals[key].to(device=fe.gate_proj_U.device, dtype=torch.float32)
                A_d = A.to(device=fe.gate_proj_U.device, dtype=torch.float32)
                B_d = None
                if bw_check:
                    B_t = _cov_lookup(B_acc.covariance, ref.layer_idx, e, name)
                    if B_t is not None:
                        B_d = B_t.to(device=fe.gate_proj_U.device, dtype=torch.float32)
                U_p = getattr(fe, f"{name}_U").data[e].clone().to(torch.float32).requires_grad_(True)
                V_p = getattr(fe, f"{name}_V").data[e].clone().to(torch.float32).requires_grad_(True)
                # Initial loss (pre-refine) for the convergence diff.
                # Held in locals — only committed to the layer accumulators
                # after a successful ``opt.step``, so an LBFGS exception
                # cannot leave bw_init incremented without a matching
                # bw_final / bw_count.
                with torch.no_grad():
                    R0 = W - U_p @ V_p
                    init_loss = float(((R0 @ A_d) * R0).sum().item())
                    bw_init_local = (
                        float(((R0 @ B_d) * R0).sum().item())
                        if B_d is not None else None
                    )
                opt = torch.optim.LBFGS(
                    [U_p, V_p], history_size=lbfgs_history,
                    max_iter=lbfgs_steps, line_search_fn="strong_wolfe",
                )

                def closure():
                    opt.zero_grad()
                    R = W - U_p @ V_p
                    loss = ((R @ A_d) * R).sum()
                    loss.backward()
                    return loss

                try:
                    final_t = opt.step(closure)
                    final_loss = float(final_t.item()) if final_t is not None else init_loss
                except Exception as err:
                    log.debug("refine skipped for %s: %s", key, err)
                    continue
                with torch.no_grad():
                    getattr(fe, f"{name}_U").data[e].copy_(U_p.to(getattr(fe, f"{name}_U").dtype))
                    getattr(fe, f"{name}_V").data[e].copy_(V_p.to(getattr(fe, f"{name}_V").dtype))
                    if B_d is not None and bw_init_local is not None:
                        R1 = W - U_p @ V_p
                        bw_init[name] += bw_init_local
                        bw_final[name] += float(((R1 @ B_d) * R1).sum().item())
                        bw_count[name] += 1
                layer_init[name] += init_loss
                layer_final[name] += final_loss
                layer_count[name] += 1
        # Per-layer LBFGS convergence: how much did the activation-weighted
        # loss drop, averaged over experts? Negative = worse (shouldn't happen).
        metrics: dict[str, float] = {"stage3/refine_layer_idx": float(ref.layer_idx)}
        for name in MATRIX_NAMES:
            if layer_count[name] == 0:
                continue
            init_m = layer_init[name] / layer_count[name]
            final_m = layer_final[name] / layer_count[name]
            rel = (init_m - final_m) / max(init_m, 1e-12)
            metrics[f"stage3/refine_loss_init/{name}"] = init_m
            metrics[f"stage3/refine_loss_final/{name}"] = final_m
            metrics[f"stage3/refine_rel_drop/{name}"] = rel
            log.info("  L%d %s LBFGS: %.4e → %.4e (drop %.1f%%)",
                     ref.layer_idx, name, init_m, final_m, 100 * rel)
            # B-weighted regression check: refine optimises the A-weighted
            # objective, but the B-weighted norm is what ``_aa_svd`` (and,
            # by proxy, the model's actual quality on calibration-like
            # input) targets. Warn loudly if refine made the B-weighted
            # residual worse on this layer × matrix.
            if bw_count[name] > 0:
                bw_i = bw_init[name] / bw_count[name]
                bw_f = bw_final[name] / bw_count[name]
                bw_rel = (bw_i - bw_f) / max(bw_i, 1e-12)
                metrics[f"stage3/refine_bw_loss_init/{name}"] = bw_i
                metrics[f"stage3/refine_bw_loss_final/{name}"] = bw_f
                metrics[f"stage3/refine_bw_rel_drop/{name}"] = bw_rel
                if bw_rel < -1e-3:
                    log.warning(
                        "  L%d %s refine REGRESSED B-weighted norm: "
                        "%.4e → %.4e (%.2f%%); A-weighted dropped %.1f%%. "
                        "Refine objective ≠ B-weighted target; consider "
                        "reducing lbfgs_steps or disabling block_refine.",
                        ref.layer_idx, name, bw_i, bw_f, 100 * bw_rel,
                        100 * rel,
                    )
                else:
                    log.info("  L%d %s B-weighted: %.4e → %.4e (drop %.1f%%)",
                             ref.layer_idx, name, bw_i, bw_f, 100 * bw_rel)
        _trackio_log(metrics)
        if bw_check:
            B_acc.unload_layer(ref.layer_idx)
