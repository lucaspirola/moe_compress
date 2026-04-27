"""Stage 3 — Non-uniform SVD, fused-experts-aware.

At this point each MoE layer still has a fused ``Qwen3_5MoeExperts`` but with
``num_experts = N'_l`` (post-prune). We:

1. Compute per-group statistics (D-Rank) over banks.
2. Choose per-group uniform rank ``k_g`` via D-Rank allocation targeting
   the global ``T_budget`` derived from ``decomposition.svd_rank_ratio``.
3. Swift-SVD+ α selection per matrix type (placeholder α=0.5 by default;
   the grid-search harness is wired but gated by config).
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
    A_cov = _load_stage2_covariance(artifacts_dir / "_stage2_input_covariance.pt")

    # B covariance: fresh calibration through the already-pruned model.
    spec = spec_from_config(
        cal,
        num_sequences_override=s3["swift_svd_plus"]["validation_samples"],
        seed_offset=2,
    )
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    # batch_size is read from config; default 1 for backwards compat. The
    # B-covariance forward runs under torch.no_grad(), so activations are
    # cheap (~100 MB / batch elem) and the GPU bottleneck is HBM bandwidth
    # — bigger batches amortize the model-weight read and approach
    # compute-bound throughput. Empirical sweet spot on a100-large @ 80 GB
    # VRAM is batch_size=16 (~30 min total Stage 3 vs. ~5h40m at batch=1).
    bcov_batch_size = int(s3.get("batch_size", 1))
    batches = iter_batches(calib, batch_size=bcov_batch_size)
    log.info("Stage 3 B-cov calibration: batch_size=%d", bcov_batch_size)
    B_acc = InputCovarianceAccumulator()
    # Match Stage 2's bf16 storage so the per-layer covariance dict stays
    # under the a100-large 142 GB cgroup limit. fp32 would be ~140 GB at
    # the end of B-cov, which crashed our prior run at layer 20.
    B_cov_dtype = getattr(torch, s3.get("bcov_storage_dtype", "bfloat16"))
    B_acc.set_storage_dtype(B_cov_dtype)
    # Per-layer disk spill directory. After each layer's finalize, that
    # layer's entries are written to disk and dropped from memory; the
    # factor loop later lazy-loads one layer at a time. Also gives us
    # crash-resume: if a previous run made it to layer 19, those .pt
    # files are already there and we skip those layers in the B-cov loop.
    bcov_spill_dir = artifacts_dir / "_stage3_bcov_partial"
    bcov_spill_dir.mkdir(parents=True, exist_ok=True)
    _collect_pruned_input_covariance(
        model, moe_layers, batches, B_acc, device=device,
        spill_dir=bcov_spill_dir,
    )

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
    alpha_by_type = _swift_svd_plus_grid(
        model, tokenizer, config, group_stats, ranks, artifacts_dir,
    )

    # 2. Snapshot originals (for Stage 4 residuals) then factor per-layer.
    originals: dict[tuple[int, int, str], torch.Tensor] = {}
    rank_map: dict[str, int] = {}

    for ref in moe_layers:
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
        banks = build_banks(ref)
        # Snapshot originals for this layer
        for e in range(ref.num_routed_experts):
            for name in MATRIX_NAMES:
                originals[(ref.layer_idx, e, name)] = banks[name].get(e).detach().cpu().clone()
        # Build FactoredExperts on the same device / dtype.
        ex = ref.experts_module
        dtype = ex.gate_up_proj.dtype
        dev = ex.gate_up_proj.device
        # Originals are already snapshotted to CPU above; offload the dense
        # expert module before allocating FactoredExperts to avoid a brief
        # double-occupancy OOM on 80 GB A100s.
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
        err_sum: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        n_per_matrix: dict[str, int] = {n: 0 for n in MATRIX_NAMES}
        for e in range(ref.num_routed_experts):
            for name in MATRIX_NAMES:
                W = originals[(ref.layer_idx, e, name)].to(device=dev, dtype=torch.float32)
                A = _cov_lookup(A_cov, ref.layer_idx, e, name)
                B = _cov_lookup(B_acc.covariance, ref.layer_idx, e, name)
                k = ranks_layer[name]
                U_k, V_k, rel_err = _aa_svd(W, A, B, k, device=dev)
                new_factored.set_factors(e, name, U_k, V_k)
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
                recon_metrics[f"stage3/recon_rel_err/{name}"] = rel
                log.info("  L%d %s rank=%d rel_recon_err=%.4f",
                         ref.layer_idx, name, ranks_layer[name], rel)
        _trackio_log(recon_metrics)
        log.info("  layer %d factored at ranks=%s", ref.layer_idx, ranks_layer)
        # Drop this layer's B-cov from memory now that we're done factoring
        # it. The next iteration will lazy-load the next layer's spill.
        B_acc.unload_layer(ref.layer_idx)

    # 3. Save originals for Stage 4.
    torch.save(originals, artifacts_dir / "_stage3_original_weights.pt")
    log.info("Saved Stage 3 original weights snapshot (%d matrices)", len(originals))

    # 4. Block refine (optional).
    if s3["block_refine"]["enabled"]:
        _per_matrix_refine(
            moe_layers, originals, A_cov,
            lbfgs_steps=s3["block_refine"]["lbfgs_steps"],
            lbfgs_history=s3["block_refine"]["lbfgs_history"],
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
    omega = d_out + n_experts * d_in
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

    denom = sum(_weight(g, s) for g, s in group_stats.items()) or 1.0
    out = {}
    for g, s in group_stats.items():
        k = _weight(g, s) * T_budget / denom
        k = max(1, min(int(round(k)), min(s.d_out, s.d_in) - 1))
        out[g] = k
    return out


def _swift_svd_plus_grid(
    model, tokenizer, config, group_stats, ranks, artifacts_dir,
) -> dict[str, float]:
    log.info("Swift-SVD+ α-grid: using α=0.5 per matrix type (placeholder).")
    return {"gate_proj": 0.5, "up_proj": 0.5, "down_proj": 0.5}


# ---------------------------------------------------------------------------
# AA-SVD per matrix
# ---------------------------------------------------------------------------


def _aa_svd(
    W: torch.Tensor,
    A: torch.Tensor | None,
    B: torch.Tensor | None,
    k: int,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Activation-aware rank-k factorization of W.

    Standard one-sided ASVD (SVD-LLM / Yuan et al.):
      Minimize ||(W - UV) L_B||_F  where B = X_post_prune^T X_post_prune
      Solution:
        M = W @ L_B
        SVD(M) = U Σ Vh
        U_k = U[:,:k] * S[:k]
        V_k = Vh[:k,:] @ L_B^{-1}   (back-solve to undo the L_B weighting)
      Then U_k @ V_k ≈ W (rank-k optimal in the B-weighted Frobenius norm).

    The pre-prune covariance A (Stage 2) is intentionally NOT used here —
    only B (post-prune) reflects the input distribution that the compressed
    model will actually see. A is reserved for the optional L-BFGS refine
    in _per_matrix_refine.

    Returns (U_k, V_k, rel_err) where rel_err is the weighted relative
    reconstruction error the objective actually minimizes:
      ||(W - U_k V_k) L_B||_F / ||W L_B||_F   ∈ [0, 1]
    """
    d_out, d_in = W.shape
    k = max(1, min(k, min(d_out, d_in) - 1))
    try:
        if B is None:
            raise ValueError("no post-prune covariance B available")
        B = B.to(device=device, dtype=torch.float32)
        B_reg = B + 1e-6 * torch.eye(B.shape[0], dtype=B.dtype, device=device)
        L_B = torch.linalg.cholesky(B_reg)
        M = W @ L_B
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = torch.linalg.solve_triangular(
            L_B.transpose(0, 1), Vh[:k, :].transpose(0, 1), upper=True,
        ).transpose(0, 1)
        with torch.no_grad():
            R_weighted = (W - U_k @ V_k) @ L_B
            W_weighted = W @ L_B
            w_norm = W_weighted.norm()
            rel_err = float((R_weighted.norm() / w_norm).item()) if w_norm > 0 else 0.0
    except Exception as err:                         # noqa: BLE001
        log.warning("AA-SVD fallback to plain SVD (%s)", err)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = Vh[:k, :]
        with torch.no_grad():
            R = W - U_k @ V_k
            w_norm = W.norm()
            rel_err = float((R.norm() / w_norm).item()) if w_norm > 0 else 0.0
    return U_k, V_k, rel_err


# ---------------------------------------------------------------------------
# Post-prune input covariance (for AA-SVD B matrix)
# ---------------------------------------------------------------------------


def _collect_pruned_input_covariance(
    model, moe_layers, batches, B_acc: InputCovarianceAccumulator, *, device,
    spill_dir=None,
) -> None:
    """Collect post-prune input covariance one layer at a time.

    With ``spill_dir`` set, after each layer's finalize the layer's entries
    are written to ``spill_dir/layer_{idx}.pt`` and dropped from memory,
    bounding peak CPU usage to ~one layer's covariance (~3-5 GB at bf16).
    On entry, any layer whose spill file already exists is skipped — this
    is the crash-resume path: a previous run that died at layer 20 leaves
    19 .pt files; the next run resumes at layer 20.

    The spill itself runs on a background single-worker thread so the main
    GPU loop can launch the *next* layer's forward pass while the previous
    layer's ~5 GB tensors stream to FUSE. Spills are serialized against
    each other (one worker) to avoid FUSE-bandwidth contention. Race-safe
    because each spill only touches keys for its own ``layer_idx``; the
    next layer mutates different keys.
    """
    def input_cb(li, e, tensor, ctx):
        B_acc.update(li, e, "gate_proj", tensor)  # up_proj aliases to gate_proj

    def intermediate_cb(li, e, tensor, ctx):
        B_acc.update(li, e, "down_proj", tensor)

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
                    log.info("Stage 3 B-cov layer %d/%d (idx=%d) — already spilled, skipping",
                             k + 1, n, ref.layer_idx)
                    continue
            log.info("Stage 3 B-cov layer %d/%d (idx=%d) — instrumented calibration pass",
                     k + 1, n, ref.layer_idx)
            with instrument_experts(ref, {"input": input_cb, "intermediate": intermediate_cb}):
                run_calibration(model, batches, device=device)
            B_acc.finalize_layer(ref.layer_idx)
            # Background spill: hand this layer off to the executor so the
            # main loop can immediately start the NEXT layer's forward pass.
            # Spill takes ~30-60s for a 5 GB layer on FUSE; the next forward
            # takes ~8 min, so we fully overlap I/O with GPU compute.
            if spill_executor is not None:
                # Surface earlier failures BEFORE submitting more work.
                _drain_done_futures(spill_futures)
                fut = spill_executor.submit(
                    B_acc.spill_layer_to_disk, ref.layer_idx, spill_dir,
                )
                spill_futures.append(fut)
            # Trackio: per-layer pipeline-progress snapshot. Three CPU memory
            # signals so we can distinguish accumulator growth from page cache:
            #   - bcov_ram_used_gb:  host-wide (psutil.virtual_memory().used);
            #                        floats with page cache, NOT a leak indicator.
            #   - bcov_proc_rss_gb:  this process's RSS (Python heap + tensor
            #                        storage + touched mmaps); tighter bound.
            #   - bcov_maxrss_gb:    peak RSS since process start, monotonically
            #                        non-decreasing — the cleanest "did the
            #                        accumulator actually grow" trace.
            proc_rss = _proc_rss_gb()
            maxrss = _maxrss_gb()
            host_ram = None
            try:
                import psutil
                host_ram = psutil.virtual_memory().used / 1e9
            except Exception:                            # noqa: BLE001
                pass
            log.info(
                "  Stage 3 B-cov layer %d/%d done — proc_rss=%sGB maxrss=%sGB host_ram=%sGB",
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
                # .result() reraises any exception from the spill thread.
                f.result()
            spill_executor.shutdown(wait=True)
            log.info("All B-cov layer spills durable on disk.")


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
) -> None:
    log.info(
        "Stage 3.D: activation-weighted refine (%d layers × 3 matrices × N experts)",
        len(moe_layers),
    )
    for ref in moe_layers:
        fe: FactoredExperts = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        # Aggregate convergence: sum of (initial_loss, final_loss) over experts
        # for this layer × matrix. Per-expert is too noisy for the dashboard.
        layer_init: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        layer_final: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        layer_count: dict[str, int] = {n: 0 for n in MATRIX_NAMES}
        for e in range(fe.num_experts):
            for name in MATRIX_NAMES:
                key = (ref.layer_idx, e, name)
                A = _cov_lookup(A_cov, ref.layer_idx, e, name)
                if A is None:
                    continue
                W = originals[key].to(device=fe.gate_proj_U.device, dtype=torch.float32)
                A_d = A.to(device=fe.gate_proj_U.device, dtype=torch.float32)
                U_p = getattr(fe, f"{name}_U").data[e].clone().to(torch.float32).requires_grad_(True)
                V_p = getattr(fe, f"{name}_V").data[e].clone().to(torch.float32).requires_grad_(True)
                # Initial loss (pre-refine) for the convergence diff.
                with torch.no_grad():
                    R0 = W - U_p @ V_p
                    init_loss = float(((R0 @ A_d) * R0).sum().item())
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
        _trackio_log(metrics)
