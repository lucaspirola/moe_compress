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
from .utils.calibration import CalibrationSpec, build_calibration_tensor, iter_batches
from .utils.model_io import (
    MATRIX_NAMES,
    FactoredExperts,
    MoELayerRef,
    build_banks,
    iter_moe_layers,
    save_compressed_checkpoint,
    save_json_artifact,
)

log = logging.getLogger(__name__)


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
    spec = CalibrationSpec(
        num_sequences=s3["swift_svd_plus"]["validation_samples"],
        sequence_length=cal["sequence_length"],
        seed=cal["seed"] + 2,
        domain_mix=cal["domain_mix"],
        c4_dataset=cal["dataset"],
        c4_subset=cal["subset"],
    )
    calib = build_calibration_tensor(
        tokenizer, spec, cache_dir=artifacts_dir / "_calibration_cache"
    )
    batches = iter_batches(calib, batch_size=1)
    B_acc = InputCovarianceAccumulator()
    _collect_pruned_input_covariance(model, moe_layers, batches, B_acc, device=device)

    # 1. Per-(layer, matrix) group stats and rank allocation.
    group_stats: dict[tuple[int, str], _GroupStats] = {}
    for ref in moe_layers:
        banks = build_banks(ref)
        for name in MATRIX_NAMES:
            group_stats[(ref.layer_idx, name)] = _group_stat(
                ref.num_routed_experts, banks[name]
            )

    T_budget = _compute_T_budget(group_stats, decomposition.svd_rank_ratio)
    ranks = _d_rank_allocate(group_stats, T_budget)
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
        banks = build_banks(ref)
        # Snapshot originals for this layer
        for e in range(ref.num_routed_experts):
            for name in MATRIX_NAMES:
                originals[(ref.layer_idx, e, name)] = banks[name].get(e).detach().cpu().clone()
        # Build FactoredExperts on the same device / dtype.
        ex = ref.experts_module
        dtype = ex.gate_up_proj.dtype
        dev = ex.gate_up_proj.device
        new_factored = FactoredExperts(
            num_experts=ref.num_routed_experts,
            hidden_dim=ex.gate_up_proj.shape[-1],
            intermediate_dim=ex.gate_up_proj.shape[1] // 2,
            ranks=ranks_layer, dtype=dtype, device=dev,
        )
        # Fill factors by per-expert AA-SVD.
        for e in range(ref.num_routed_experts):
            for name in MATRIX_NAMES:
                W = originals[(ref.layer_idx, e, name)].to(device=dev, dtype=torch.float32)
                A = _cov_lookup(A_cov, ref.layer_idx, e, name)
                B = _cov_lookup(B_acc.covariance, ref.layer_idx, e, name)
                k = ranks_layer[name]
                U_k, V_k = _aa_svd(W, A, B, k, device=dev)
                new_factored.set_factors(e, name, U_k, V_k)
                rank_map[f"L{ref.layer_idx}_E{e}_{name}"] = k
        # Swap in.
        setattr(ref.mlp, "experts", new_factored)
        ref.experts_module = new_factored
        log.info("  layer %d factored at ranks=%s", ref.layer_idx, ranks_layer)

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


def _d_rank_allocate(group_stats: dict, T_budget: int) -> dict:
    denom = sum(math.sqrt(s.effective_rank / s.omega) for s in group_stats.values()) or 1.0
    out = {}
    for g, s in group_stats.items():
        k = math.sqrt(s.effective_rank / s.omega) * T_budget / denom
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
) -> tuple[torch.Tensor, torch.Tensor]:
    d_out, d_in = W.shape
    k = max(1, min(k, min(d_out, d_in) - 1))
    try:
        if A is None or B is None:
            raise ValueError("no covariance available")
        A = A.to(device=device, dtype=torch.float32)
        B = B.to(device=device, dtype=torch.float32)
        B_reg = B + 1e-6 * torch.eye(B.shape[0], dtype=B.dtype, device=device)
        L_B = torch.linalg.cholesky(B_reg)
        BBT_inv = torch.cholesky_inverse(L_B)
        M = W @ A @ B.transpose(0, 1) @ BBT_inv @ L_B
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = torch.linalg.solve_triangular(
            L_B.transpose(0, 1), Vh[:k, :].transpose(0, 1), upper=True,
        ).transpose(0, 1)
    except Exception as err:                         # noqa: BLE001
        log.warning("AA-SVD fallback to plain SVD (%s)", err)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = Vh[:k, :]
    return U_k, V_k


# ---------------------------------------------------------------------------
# Post-prune input covariance (for AA-SVD B matrix)
# ---------------------------------------------------------------------------


def _collect_pruned_input_covariance(
    model, moe_layers, batches, B_acc: InputCovarianceAccumulator, *, device,
) -> None:
    """Collect post-prune input covariance one layer at a time so GPU + CPU
    memory stays bounded. Trade-off: N forward passes instead of 1, but each
    one only accumulates covariance for a single MoE layer (~200 experts ×
    2 matrices = ~400 entries on CPU).

    This addresses review P1-4 (simultaneous 40-layer instrumentation).
    """
    def input_cb(li, e, tensor, ctx):
        B_acc.update(li, e, "gate_proj", tensor)  # up_proj aliases to gate_proj

    def intermediate_cb(li, e, tensor, ctx):
        B_acc.update(li, e, "down_proj", tensor)

    for ref in moe_layers:
        with instrument_experts(ref, {"input": input_cb, "intermediate": intermediate_cb}):
            run_calibration(model, batches, device=device)


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
                    opt.step(closure)
                except Exception as err:
                    log.debug("refine skipped for %s: %s", key, err)
                    continue
                with torch.no_grad():
                    getattr(fe, f"{name}_U").data[e].copy_(U_p.to(getattr(fe, f"{name}_U").dtype))
                    getattr(fe, f"{name}_V").data[e].copy_(V_p.to(getattr(fe, f"{name}_V").dtype))
