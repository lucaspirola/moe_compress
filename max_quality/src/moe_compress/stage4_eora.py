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
    for ref in iter_moe_layers(model):
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        dev = fe.gate_proj_U.device
        dtype = fe.gate_proj_U.dtype
        N = fe.num_experts
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
                A = A_cov.get(key)
                Uc, Vc = _compute_eora_factors(delta, A, r_per_expert, dev)
                U_corr[e] = Uc.to(dtype)
                V_corr[e] = Vc.to(dtype)

            fe.widen_rank(name, U_corr, V_corr)
            rank_map[f"L{ref.layer_idx}_{name}"] = fe.ranks[name]
            compensated_params += int(U_corr.numel() + V_corr.numel())
            log.info("  L%d/%s widened by r=%d → new rank=%d",
                     ref.layer_idx, name, r_per_expert, fe.ranks[name])

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
