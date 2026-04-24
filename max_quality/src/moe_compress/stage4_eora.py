"""Stage 4 — EoRA (training-free compensation).

Per expert matrix ``W`` that was factored in Stage 3 into ``W_c = U · V``:

1. Form the residual ``ΔW = W_orig - W_c``.
2. Project ``ΔW`` into the top-``r`` eigenspace of the input activation
   covariance ``Σ = Σ_x xxᵀ`` (loaded from Stage 2's cached covariance).
3. Produce a rank-``r`` SVD of ``ΔW @ P_eig``.
4. Merge the low-rank correction into the factored module in-place:

        down_new = cat([V; Vcorr])
        up_new   = cat([U, Ucorr], dim=1)

   i.e. we widen the factored rank by ``r`` and load the correction into the
   new dimensions. This keeps the forward pass identical to the paper's
   "Ŵ·x + B'·A·x" while avoiding runtime adapters.

Artifact: ``stage4_eora/`` safetensors + ``eora_ranks.json``.

Caveat (from VALIDATED_STRATEGIES §Stage 4): EoRA was only tested on dense
LLaMA. For MoE we do per-expert EoRA. Per-matrix compensation budget is
capped at ``compensation_budget_pct`` of the params saved in Stage 3 for
that matrix.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from .stage3_svd import _FactoredLinear
from .utils.model_io import (
    get_expert_matrices,
    iter_moe_layers,
    iter_routed_experts,
    save_checkpoint,
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
    if not A_cov_path.exists():
        log.warning("Stage 4: no Stage 2 covariance at %s — running isotropic fallback", A_cov_path)
        A_cov: dict = {}
    else:
        A_cov = torch.load(A_cov_path, map_location="cpu").get("covariance", {})

    # Find originals: we saved them on disk during Stage 3 if a snapshot file
    # was created; otherwise reconstruct ΔW using the factored Linear's
    # current composition versus the pre-stage3 weight. We prefer the latter
    # because Stage 3's refinement step already adjusted U,V toward the
    # original — the delta here captures residual error.
    originals_path = artifacts_dir / "_stage3_original_weights.pt"
    originals: dict[tuple[int, int, str], torch.Tensor] = {}
    if originals_path.exists():
        originals = torch.load(originals_path, map_location="cpu")
    else:
        log.warning(
            "Stage 4: no original-weight snapshot at %s — EoRA will use "
            "zero-residual (no compensation). Run Stage 3 with "
            "save_originals=True to enable.",
            originals_path,
        )

    rank_map: dict[str, int] = {}
    compensated_params = 0
    for ref in iter_moe_layers(model):
        for e_idx, expert in iter_routed_experts(ref):
            for name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(expert, name, None)
                if not isinstance(mod, _FactoredLinear):
                    continue
                key = (ref.layer_idx, e_idx, name)
                A = A_cov.get(key)
                W_orig = originals.get(key)
                if W_orig is None:
                    # No delta available — skip this matrix.
                    continue
                W_orig = W_orig.to(torch.float32)
                W_cur = (mod.up.weight @ mod.down.weight).detach().to(torch.float32).cpu()
                delta = W_orig - W_cur
                if delta.abs().max().item() < 1e-8:
                    continue
                r = _pick_eora_rank(
                    delta=delta,
                    factored=mod,
                    compensation_budget_pct=s4["compensation_budget_pct"],
                    eigenspace_rank_cap=s4["eigenspace_rank_cap"],
                )
                if r <= 0:
                    continue
                U_corr, V_corr = _compute_eora_factors(delta, A, r)
                _merge_into_factored(mod, U_corr, V_corr)
                rank_map[f"L{ref.layer_idx}_E{e_idx}_{name}"] = r
                compensated_params += r * (mod.down.weight.shape[1] + mod.up.weight.shape[0])

    out_dir = artifacts_dir / "stage4_eora"
    save_checkpoint(model, tokenizer, out_dir)
    save_json_artifact({
        "rank_map": rank_map,
        "compensated_params": compensated_params,
        "config": s4,
    }, out_dir / "eora_ranks.json")
    log.info("Stage 4 complete — EoRA added %d params across %d matrices → %s",
             compensated_params, len(rank_map), out_dir)
    return out_dir


def _pick_eora_rank(
    *,
    delta: torch.Tensor,
    factored: _FactoredLinear,
    compensation_budget_pct: float,
    eigenspace_rank_cap: int,
) -> int:
    """Compute per-matrix EoRA rank.

    Budget = ``compensation_budget_pct × (stage3_saved_params)``. In a factored
    matrix stored as d2×k plus k×d1, savings vs the full W (d2→d1) are
    ``d1*d2 - k*(d1+d2)``. EoRA adds ``r*(d1+d2)``.
    """
    d1, k_current = factored.up.weight.shape
    k_current2, d2 = factored.down.weight.shape
    assert k_current == k_current2
    saved = max(0, d1 * d2 - k_current * (d1 + d2))
    max_added = int(compensation_budget_pct * saved)
    r_by_budget = max_added // max(d1 + d2, 1)
    r = min(r_by_budget, eigenspace_rank_cap, min(d1, d2) - 1)
    return max(0, r)


def _compute_eora_factors(
    delta: torch.Tensor, A: torch.Tensor | None, r: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-``r`` SVD of ΔW projected into the top-``r`` eigenspace of A."""
    if r <= 0:
        return torch.zeros(delta.shape[0], 0), torch.zeros(0, delta.shape[1])
    if A is None:
        # Isotropic fallback — plain truncated SVD of the residual.
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        return (U[:, :r] * S[:r]), Vh[:r, :]
    A = A.to(torch.float32)
    eigvals, eigvecs = torch.linalg.eigh(A + 1e-6 * torch.eye(A.shape[0], dtype=A.dtype))
    # Sort descending, take top-r eigenvectors
    order = torch.argsort(eigvals, descending=True)
    P = eigvecs[:, order[:r]]                        # [d2, r]
    # Project ΔW into eigenspace → [d1, r]
    projected = delta @ P
    U, S, Vh = torch.linalg.svd(projected, full_matrices=False)
    # Lift Vh from eigenspace back to original d2
    V_back = Vh @ P.transpose(0, 1)
    U_corr = U[:, :r] * S[:r]
    V_corr = V_back[:r, :]
    return U_corr, V_corr


def _merge_into_factored(mod: _FactoredLinear, U_corr: torch.Tensor, V_corr: torch.Tensor) -> None:
    """Widen ``mod.up`` and ``mod.down`` by the correction rank."""
    dtype = mod.up.weight.dtype
    device = mod.up.weight.device
    U_corr = U_corr.to(dtype=dtype, device=device)
    V_corr = V_corr.to(dtype=dtype, device=device)

    old_k = mod.up.weight.shape[1]
    r = U_corr.shape[1]
    if r <= 0:
        return
    new_up = nn.Linear(old_k + r, mod.up.weight.shape[0], bias=(mod.up.bias is not None))
    new_down = nn.Linear(mod.down.weight.shape[1], old_k + r, bias=False)
    with torch.no_grad():
        new_up.weight[:, :old_k].copy_(mod.up.weight)
        new_up.weight[:, old_k:].copy_(U_corr)
        if mod.up.bias is not None:
            new_up.bias.copy_(mod.up.bias)
        new_down.weight[:old_k, :].copy_(mod.down.weight)
        new_down.weight[old_k:, :].copy_(V_corr)
    mod.up = new_up.to(device)
    mod.down = new_down.to(device)
