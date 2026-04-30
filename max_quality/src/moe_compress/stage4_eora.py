"""Stage 4 — EoRA residual compensation, fused-experts-aware.

For each (layer, expert, matrix) factored in Stage 3, we compute the
residual ΔW_e = W_orig_e - U_e @ V_e, project it through the **√Λ-scaled
eigenspace** of the input activation covariance (paper Algorithm 1, step 3:
Q' = Q·√Λ), take a rank-r SVD of the *full* projected error ΔW'= ΔW·Q'
(shape [d_out, d_in], NOT pre-truncated to r), then back-project via
V_corr = V'^T · (√Λ)^{-1} · Q^T and **widen** the corresponding
``FactoredExperts`` U / V along the rank dim.

Reference: EoRA (2410.21271) Algorithm 1.

The compensation budget per matrix is capped at
``compensation_budget_pct × (stage3 saved params for this matrix)``, so
Stage 4 never re-inflates by more than that fraction.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from .utils.model_io import (
    MATRIX_NAMES,
    FactoredExperts,
    iter_moe_layers,
    save_compressed_checkpoint,
    save_json_artifact,
)
from .utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


def run(
    model,
    tokenizer,
    config: dict,
    artifacts_dir: Path,
) -> Path:
    s4 = config["stage4_eora"]
    # A-cov was persisted by Stage 2 in this storage dtype; the eigh threshold
    # in `_compute_eora_factors` must be tuned to that dtype's quantization
    # noise floor or it will keep noise-inflated directions.
    s2 = config.get("stage2_reap_ream", {})
    a_storage_dtype = getattr(torch, s2.get("covariance_storage_dtype", "float32"))
    A_cov_path = artifacts_dir / "_stage2_input_covariance.pt"
    A_cov = {}
    if A_cov_path.exists():
        A_cov = torch.load(A_cov_path, map_location="cpu").get("covariance", {})
    else:
        log.warning("Stage 4: no Stage 2 covariance at %s — isotropic fallback",
                    A_cov_path)

    originals_path = artifacts_dir / "_stage3_original_weights.pt"
    if not originals_path.exists():
        # If Stage 4 already completed, the originals are intentionally deleted.
        # Re-entering Stage 4 on an already-widened model is a double-widen.
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
    originals: dict = torch.load(originals_path, map_location="cpu")

    rank_map: dict[str, int] = {}
    compensated_params = 0
    layers = list(iter_moe_layers(model))
    log.info("Stage 4: EoRA residual compensation over %d MoE layers", len(layers))

    partial_dir = artifacts_dir / "_stage4_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot Stage 3 ranks before any widening occurs.
    # Used by the double-widen guard below to detect in-process re-runs.
    stage3_ranks: dict[int, dict[str, int]] = {}
    for ref in layers:
        fe = ref.experts_module
        if isinstance(fe, FactoredExperts):
            stage3_ranks[ref.layer_idx] = {name: fe.ranks[name] for name in MATRIX_NAMES}

    for k, ref in enumerate(layers):
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        dev = fe.gate_proj_U.device
        dtype = fe.gate_proj_U.dtype
        N = fe.num_experts

        # Crash-resume: load saved layer state if present.
        spill_path = partial_dir / f"layer_{ref.layer_idx}.pt"
        if spill_path.exists():
            try:
                payload = torch.load(spill_path, map_location="cpu")
            except Exception as exc:
                raise RuntimeError(
                    f"Stage 4 resume: failed to load {spill_path}: {exc}"
                ) from exc
            fv = int(payload.get("format_version", 0))
            if fv != 1:
                raise RuntimeError(
                    f"Stage 4 resume: {spill_path} has format_version={fv} "
                    "(expected 1) — delete _stage4_partial/ and re-run Stage 4"
                )
            for name in MATRIX_NAMES:
                u = payload[f"{name}_U"].to(device=dev, dtype=dtype)
                v = payload[f"{name}_V"].to(device=dev, dtype=dtype)
                setattr(fe, f"{name}_U", nn.Parameter(u, requires_grad=False))
                setattr(fe, f"{name}_V", nn.Parameter(v, requires_grad=False))
                fe.ranks[name] = int(payload["ranks"][name])
                if "effective_ranks" in payload:
                    fe.effective_ranks[name] = [int(r) for r in payload["effective_ranks"][name]]
            rank_map.update(payload["rank_map_layer"])
            compensated_params += int(payload["compensated_params_layer"])
            log.info("Stage 4 layer %d/%d (idx=%d) — resumed from partial",
                     k + 1, len(layers), ref.layer_idx)
            continue

        log.info("Stage 4 layer %d/%d (idx=%d, %d experts)", k + 1, len(layers), ref.layer_idx, N)
        layer_compensated_params = 0
        rank_map_layer: dict[str, int] = {}

        for name in MATRIX_NAMES:
            # Per-matrix-type, per-layer: pool per-expert residuals independently.
            # Budget: ≤ compensation_budget_pct of saved params for this matrix.
            d_out, d_in = fe.matrix_shape(name)
            cur_rank = fe.ranks[name]
            saved_per_expert = d_out * d_in - cur_rank * (d_out + d_in)
            saved_for_matrix = max(0, saved_per_expert) * N
            max_added_ranks = int(s4["compensation_budget_pct"] * saved_for_matrix)
            r_per_expert = max_added_ranks // max(N * (d_out + d_in), 1)
            r_per_expert = min(r_per_expert, s4["eigenspace_rank_cap"], min(d_out, d_in) - 1)
            r_per_expert = max(0, r_per_expert)
            if r_per_expert <= 0:
                continue

            U_corr = torch.zeros(N, d_out, r_per_expert, dtype=dtype, device=dev)
            V_corr = torch.zeros(N, r_per_expert, d_in, dtype=dtype, device=dev)
            # Per-expert effective rank of the EoRA correction. Defaults to
            # 0 for experts without an `originals` entry (no correction
            # applied → no parameters added in effective terms).
            eff_per_expert: list[int] = [0] * N
            # Track residual norm before/after to verify EoRA actually helps.
            res_before_sum = 0.0
            res_after_sum = 0.0
            n_eligible = 0
            for e in range(N):
                key = (ref.layer_idx, e, name)
                W_orig = originals.get(key)
                if W_orig is None:
                    continue
                W_orig_f = W_orig.to(device=dev, dtype=torch.float32)
                U_e = fe.gate_proj_U.data[e] if name == "gate_proj" else \
                       fe.up_proj_U.data[e]   if name == "up_proj"   else \
                       fe.down_proj_U.data[e]
                V_e = fe.gate_proj_V.data[e] if name == "gate_proj" else \
                       fe.up_proj_V.data[e]   if name == "up_proj"   else \
                       fe.down_proj_V.data[e]
                delta = W_orig_f - (U_e.to(torch.float32) @ V_e.to(torch.float32))
                res_before_sum += float(delta.norm().item() ** 2)
                # up_proj shares the gate_proj input covariance (same fused tensor).
                cov_key = (ref.layer_idx, e, "gate_proj") if name == "up_proj" else key
                A = A_cov.get(cov_key)
                Uc, Vc, take_eff = _compute_eora_factors(
                    delta, A, r_per_expert, dev, storage_dtype=a_storage_dtype,
                )
                U_corr[e] = Uc.to(dtype)
                V_corr[e] = Vc.to(dtype)
                eff_per_expert[e] = int(take_eff)
                # Residual after applying the planned correction (Uc @ Vc).
                res_after = delta - (Uc.to(torch.float32) @ Vc.to(torch.float32))
                res_after_sum += float(res_after.norm().item() ** 2)
                n_eligible += 1
                if (e + 1) % 32 == 0:
                    log.info("  L%d/%s expert %d/%d", ref.layer_idx, name, e + 1, N)

            # Double-widen guard: assert ranks haven't been modified yet.
            # Protects against in-process re-runs (notebooks, test harnesses)
            # where widen_rank() would double-apply EoRA correction.
            assert fe.ranks[name] == stage3_ranks.get(ref.layer_idx, {}).get(name, fe.ranks[name]), (
                f"Stage 4 double-widen detected: layer={ref.layer_idx}, matrix={name}, "
                f"current_rank={fe.ranks[name]}, "
                f"stage3_rank={stage3_ranks.get(ref.layer_idx, {}).get(name)}. "
                "widen_rank() has already been applied in this process."
            )
            fe.widen_rank(name, U_corr, V_corr, added_effective_per_expert=eff_per_expert)
            rank_map_layer[f"L{ref.layer_idx}_{name}"] = fe.ranks[name]
            layer_compensated_params += int(U_corr.numel() + V_corr.numel())
            res_before = (res_before_sum / max(n_eligible, 1)) ** 0.5
            res_after = (res_after_sum / max(n_eligible, 1)) ** 0.5
            rel_drop = (res_before - res_after) / max(res_before, 1e-12)
            log.info("  L%d/%s widened by r=%d → new rank=%d; residual %.4e→%.4e (-%.1f%%)",
                     ref.layer_idx, name, r_per_expert, fe.ranks[name],
                     res_before, res_after, 100 * rel_drop)
            _trackio_log({
                "stage4/layer_idx": ref.layer_idx,
                f"stage4/{name}_added_rank": r_per_expert,
                f"stage4/{name}_new_rank": fe.ranks[name],
                f"stage4/{name}_residual_before": res_before,
                f"stage4/{name}_residual_after": res_after,
                f"stage4/{name}_residual_rel_drop": rel_drop,
                "stage4/compensated_params": compensated_params + layer_compensated_params,
            })

        rank_map.update(rank_map_layer)
        compensated_params += layer_compensated_params

        # Atomically persist this layer's FactoredExperts state for crash-resume.
        _spill_layer(partial_dir, ref.layer_idx, fe, rank_map_layer, layer_compensated_params)

    out_dir = artifacts_dir / "stage4_eora"
    save_compressed_checkpoint(
        model, tokenizer, out_dir,
        pipeline_stage="stage4_eora",
        extra_metadata={"compensated_params": compensated_params},
    )
    save_json_artifact({
        "rank_map": rank_map,
        "compensated_params": compensated_params,
        "config": s4,
    }, out_dir / "eora_ranks.json")

    shutil.rmtree(partial_dir, ignore_errors=True)

    # Stage 4 is the last consumer of both `_stage3_original_weights.pt` and
    # `_stage2_input_covariance.pt`. Both are already durable on the per-stage
    # Hub repos (`<base>-stage2`, `<base>-stage3`) — leaving them on the
    # bucket only causes the entrypoint's job-exit aux upload to push ~140 GB
    # of already-uploaded data to the aggregate result repo. Delete on Stage 4
    # success only; on failure they stay so a re-run can pick up cleanly.
    for sidecar in ("_stage3_original_weights.pt", "_stage2_input_covariance.pt"):
        p = artifacts_dir / sidecar
        if p.exists():
            try:
                p.unlink()
                log.info("Deleted %s (no longer needed past Stage 4; durable on Hub)", p)
            except OSError as exc:
                log.warning("Could not delete %s: %s", p, exc)
    log.info("Stage 4 complete — EoRA added %d params → %s", compensated_params, out_dir)
    return out_dir


def _spill_layer(
    partial_dir: Path,
    layer_idx: int,
    fe: FactoredExperts,
    rank_map_layer: dict[str, int],
    compensated_params_layer: int,
) -> None:
    payload = {
        "format_version": 1,
        "layer_idx": layer_idx,
        "ranks": dict(fe.ranks),
        "effective_ranks": {n: list(v) for n, v in fe.effective_ranks.items()},
        "rank_map_layer": rank_map_layer,
        "compensated_params_layer": compensated_params_layer,
    }
    for name in MATRIX_NAMES:
        payload[f"{name}_U"] = getattr(fe, f"{name}_U").data.cpu()
        payload[f"{name}_V"] = getattr(fe, f"{name}_V").data.cpu()
    tmp = partial_dir / f"layer_{layer_idx}.pt.tmp"
    final = partial_dir / f"layer_{layer_idx}.pt"
    torch.save(payload, tmp)
    os.replace(tmp, final)


def _compute_eora_factors(
    delta: torch.Tensor,
    A: torch.Tensor | None,
    r: int,
    device,
    *,
    storage_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Paper-correct EoRA (2410.21271) Algorithm 1.

    Steps (matching paper notation, restricted to signal eigenspace):
      1. ΔW = W_orig − Ŵ                           [d_out × d_in]  (caller)
      2. Eigendecompose X̃X̃ᵀ = QΛQᵀ                [d_in × d_in]
         Keep n_keep eigenvectors above noise floor.
      3. Q' = Q_keep · √Λ_keep                      [d_in × n_keep]
      4. ΔW' = ΔW · Q'                              [d_out × n_keep]  (full projection)
      5. SVD(ΔW') full_matrices=False → top take_eff=min(r, min(d_out, n_keep))
      6. B' = U'[:,:take_eff] · Σ'[:take_eff]       [d_out × take_eff]
         A  = Vh'[:take_eff] · diag(1/√Λ_keep) @ Q_keep^T   [take_eff × d_in]

    The √Λ scaling is the core innovation of EoRA: it importance-weights the
    eigenspace so SVD concentrates rank budget on high-variance input directions.
    Without it, this degenerates to the Act-S baseline.

    Returns (U, V, take_eff) where `take_eff <= r` is the effective rank.
    When `take_eff < r`, U/V are zero-padded to width r.
    """
    if r <= 0:
        return (torch.zeros(delta.shape[0], 0, device=device),
                torch.zeros(0, delta.shape[1], device=device), 0)
    delta = delta.to(device=device, dtype=torch.float32)
    d_out, d_in = delta.shape

    def _plain_svd_padded() -> tuple[torch.Tensor, torch.Tensor, int]:
        # Fallback: plain SVD without activation weighting.
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        rk = min(r, U.shape[1])
        if rk == r:
            return U[:, :r] * S[:r], Vh[:r, :], r
        U_out = torch.zeros(d_out, r, device=device, dtype=delta.dtype)
        V_out = torch.zeros(r, d_in, device=device, dtype=delta.dtype)
        U_out[:, :rk] = U[:, :rk] * S[:rk]
        V_out[:rk, :] = Vh[:rk, :]
        return U_out, V_out, rk

    if A is None:
        return _plain_svd_padded()

    A = A.to(device=device, dtype=torch.float32)
    A = 0.5 * (A + A.T)
    if A.shape != (d_in, d_in):
        log.warning("EoRA: covariance shape %s != (%d,%d), falling back to plain SVD",
                    A.shape, d_in, d_in)
        return _plain_svd_padded()

    # Step 2: Eigendecompose activation covariance X̃X̃ᵀ = QΛQᵀ  (A shape [d_in × d_in])
    eigvals, eigvecs = torch.linalg.eigh(A)  # ascending order

    sigma_max = float(eigvals[-1].clamp_min(0).item())
    # Dtype-aware noise floor — see _NOISE_FLOOR_BY_DTYPE in stage3_svd.
    from moe_compress.stage3_svd import _NOISE_FLOOR_BY_DTYPE
    rel_floor = _NOISE_FLOOR_BY_DTYPE.get(storage_dtype or torch.float32, 1e-6)
    thresh = max(sigma_max * rel_floor, 1e-12)

    keep_mask = eigvals > thresh
    if not keep_mask.any():
        return _plain_svd_padded()

    # Keep only directions above the noise floor for the projection.
    eigvals_keep = eigvals[keep_mask].clamp_min(0)
    eigvecs_keep = eigvecs[:, keep_mask]         # [d_in, n_keep]
    n_keep = int(eigvals_keep.numel())

    # Step 3: Q' = Q · √Λ  (paper Algorithm 1 step 3).
    # FULL projection matrix — NOT truncated to r. The SVD in step 5
    # will optimally select the best r directions from the FULL d_in-
    # dimensional projected error. Pre-truncating to r would eliminate
    # the joint optimisation that distinguishes EoRA from Act-S.
    sqrt_lambda = eigvals_keep.sqrt()                         # [n_keep]
    Q_prime = eigvecs_keep * sqrt_lambda.unsqueeze(0)         # [d_in, n_keep]

    # Step 4: ΔW' = ΔW · Q'  (FULL projection, [d_out × n_keep])
    delta_prime = delta @ Q_prime                             # [d_out, n_keep]

    # Step 5: rank-r SVD of the full projected error
    U_p, S_p, Vh_p = torch.linalg.svd(delta_prime, full_matrices=False)
    take_eff = min(r, int(U_p.shape[1]))

    # Step 6a: B' = U' Σ'
    U_corr = U_p[:, :take_eff] * S_p[:take_eff]              # [d_out, take_eff]

    # Step 6b: Back-project A = V'^T · Q'^{-1}
    # Q'^{-1} = diag(1/√Λ) · Q^T  (since Q has orthonormal columns)
    # So: A = V'^T · diag(1/√Λ) · Q^T = (Vh_p[:take_eff] · diag(1/√Λ)) @ Q^T
    inv_sqrt_lambda = eigvals_keep.clamp_min(1e-30).rsqrt()   # [n_keep]
    V_corr = (Vh_p[:take_eff, :] * inv_sqrt_lambda.unsqueeze(0)) @ eigvecs_keep.T
    # V_corr shape: [take_eff, d_in]

    # Zero-pad to fixed r so caller's pre-allocated tensors stay shape-stable.
    if take_eff >= r:
        return U_corr[:, :r], V_corr[:r, :], r
    U_out = torch.zeros(d_out, r, device=device, dtype=U_corr.dtype)
    V_out = torch.zeros(r, d_in, device=device, dtype=V_corr.dtype)
    U_out[:, :take_eff] = U_corr
    V_out[:take_eff, :] = V_corr
    return U_out, V_out, take_eff
