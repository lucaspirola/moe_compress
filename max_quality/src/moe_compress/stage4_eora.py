"""Stage 4 — EoRA residual compensation, fused-experts-aware.

For each (layer, expert, matrix) factored in Stage 3, we compute the
residual ΔW_e = W_orig_e - U_e @ V_e, project it into the top-r eigenspace
of the input activation covariance, take a rank-r SVD, and **widen** the
corresponding ``FactoredExperts`` U / V along the rank dim.

The compensation budget per matrix is capped at
``compensation_budget_pct × (stage3 saved params for this matrix)``, so
Stage 4 never re-inflates by more than that fraction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

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
    A_cov_path = artifacts_dir / "_stage2_input_covariance.pt"
    A_cov = {}
    if A_cov_path.exists():
        A_cov = torch.load(A_cov_path, map_location="cpu").get("covariance", {})
    else:
        log.warning("Stage 4: no Stage 2 covariance at %s — isotropic fallback",
                    A_cov_path)

    originals_path = artifacts_dir / "_stage3_original_weights.pt"
    originals: dict = {}
    if originals_path.exists():
        originals = torch.load(originals_path, map_location="cpu")
    else:
        log.warning("Stage 4: no Stage 3 originals at %s — skipping compensation",
                    originals_path)

    rank_map: dict[str, int] = {}
    compensated_params = 0
    layers = list(iter_moe_layers(model))
    log.info("Stage 4: EoRA residual compensation over %d MoE layers", len(layers))
    for k, ref in enumerate(layers):
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        dev = fe.gate_proj_U.device
        dtype = fe.gate_proj_U.dtype
        N = fe.num_experts
        log.info("Stage 4 layer %d/%d (idx=%d, %d experts)", k + 1, len(layers), ref.layer_idx, N)
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
                A = A_cov.get(key)
                Uc, Vc = _compute_eora_factors(delta, A, r_per_expert, dev)
                U_corr[e] = Uc.to(dtype)
                V_corr[e] = Vc.to(dtype)
                # Residual after applying the planned correction (Uc @ Vc).
                res_after = delta - (Uc.to(torch.float32) @ Vc.to(torch.float32))
                res_after_sum += float(res_after.norm().item() ** 2)
                n_eligible += 1
                if (e + 1) % 32 == 0:
                    log.info("  L%d/%s expert %d/%d", ref.layer_idx, name, e + 1, N)

            fe.widen_rank(name, U_corr, V_corr)
            rank_map[f"L{ref.layer_idx}_{name}"] = fe.ranks[name]
            compensated_params += int(U_corr.numel() + V_corr.numel())
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
                "stage4/compensated_params": compensated_params,
            })

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


def _compute_eora_factors(
    delta: torch.Tensor,
    A: torch.Tensor | None,
    r: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if r <= 0:
        return (torch.zeros(delta.shape[0], 0, device=device),
                torch.zeros(0, delta.shape[1], device=device))
    delta = delta.to(device=device, dtype=torch.float32)
    if A is None:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        return U[:, :r] * S[:r], Vh[:r, :]
    A = A.to(device=device, dtype=torch.float32)
    eigvals, eigvecs = torch.linalg.eigh(A + 1e-6 * torch.eye(A.shape[0], device=device))
    order = torch.argsort(eigvals, descending=True)
    P = eigvecs[:, order[:r]]
    projected = delta @ P
    U, S, Vh = torch.linalg.svd(projected, full_matrices=False)
    V_back = Vh @ P.transpose(0, 1)
    return U[:, :r] * S[:r], V_back[:r, :]
