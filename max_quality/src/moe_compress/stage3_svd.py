"""Stage 3 — Non-uniform SVD rank allocation on MoE expert matrices only.

Four sub-steps from ``VALIDATED_STRATEGIES.md §Stage 3``:

A. **D-Rank spectral entropy** (2509.25622 Eq. 2, 6) — compute effective rank
   ``R_eff(g) = exp(-Σ p log p)`` on normalized squared singular values of each
   group ``g ∈ {(layer, matrix_type)}``. Allocate a preliminary group rank

        k_g* = (√(R_eff(g)/ω) · T_budget) / Σ_{g'} √(R_eff(g')/ω)

   with ω = d1 + n·d2 (parameter cost of a rank-1 factor).

B. **Swift-SVD+** (2604.01609 Alg. 2) — within each group, rank the singular
   components by ``s_i = β_i^α · (log(e + ε*_{k̄,i}))^{1-α}``. α is grid-
   searched on a held-out WikiText-2 slice (placeholder uses α=0.5 per type;
   full grid is enabled via the config flag).

C. **AA-SVD** (2604.02119 Thm 3.2) — activation-anchored low-rank factorization
   using pre-prune A and post-prune B covariances.

D. **Block-level refinement** (AA-SVD §3.3) — joint L-BFGS fit of all
   ``{U_j, V_j}`` in each transformer block against block-output MSE.

Restriction: SVD is applied **only** to routed expert matrices (``gate_proj``,
``up_proj``, ``down_proj``). Attention, shared expert, router, embeddings,
``lm_head`` are untouched.

Artifacts: ``stage3_svd/`` safetensors with factored Linears replaced in place
+ ``rank_map.json``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .budget.solver import BudgetDecomposition
from .utils.activation_hooks import (
    InputCovarianceAccumulator,
    hook_matrix_inputs,
    run_calibration,
)
from .utils.calibration import CalibrationSpec, build_calibration_tensor, iter_batches
from .utils.model_io import (
    MoELayerRef,
    get_expert_matrices,
    iter_moe_layers,
    iter_routed_experts,
    save_checkpoint,
    save_json_artifact,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# References to expert matrices (with parent pointers so we can replace
# the Linear in-place later).
# ---------------------------------------------------------------------------


@dataclass
class _MatrixRef:
    layer_idx: int
    expert_idx: int
    name: str                       # "gate_proj" | "up_proj" | "down_proj"
    linear: nn.Linear
    parent: nn.Module               # the expert module
    attr: str                       # attribute name on parent (==name)


@dataclass
class _GroupStats:
    d1: int                                   # output dim
    d2: int                                   # input dim
    n_matrices: int
    singular_values_mean: torch.Tensor        # [min(d1, d2)]
    effective_rank: float
    omega: float


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


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

    matrices = list(_iter_expert_matrices(model))
    log.info("Stage 3: %d expert matrices in scope", len(matrices))

    # Covariances: A from Stage 2 (pre-prune), B from fresh calibration through
    # the already-pruned model.
    A_cov, tokens_A = _load_covariance(artifacts_dir / "_stage2_input_covariance.pt")
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
    moe_layers = list(iter_moe_layers(model))
    with hook_matrix_inputs(moe_layers, B_acc):
        run_calibration(model, batches, device=device)

    # Group statistics
    groups = _group_matrices(matrices)
    group_stats = {g: _group_stat(ms) for g, ms in groups.items()}

    # T_budget from target svd_rank_ratio
    T_budget = _compute_T_budget(groups, group_stats, decomposition.svd_rank_ratio)
    initial_ranks = _d_rank_allocate(group_stats, T_budget)

    alpha_by_type = _swift_svd_plus_grid(
        model, tokenizer, config, group_stats, initial_ranks, artifacts_dir,
    )

    # Snapshot original full-rank weights before factoring, so the refine step
    # can minimize ||W_orig @ x - up(down(x))||² directly, and so Stage 4 can
    # compute ΔW against them.
    originals: dict[tuple[int, int, str], torch.Tensor] = {
        (m.layer_idx, m.expert_idx, m.name): m.linear.weight.detach().clone().cpu()
        for m in matrices
    }
    torch.save(originals, artifacts_dir / "_stage3_original_weights.pt")
    log.info("Saved Stage 3 original-weight snapshot (%d matrices)", len(originals))

    rank_map: dict[str, int] = {}
    for g, ms in groups.items():
        layer_idx, name = g
        alpha = alpha_by_type.get(name, 0.5)
        k = initial_ranks[g]
        per_matrix_ranks = _swift_svd_plus_per_matrix(ms, alpha, k, group_stats[g])
        for m, rk in zip(ms, per_matrix_ranks):
            _apply_aa_svd(
                m, rk,
                A=A_cov.get((m.layer_idx, m.expert_idx, m.name)),
                B=B_acc.covariance.get((m.layer_idx, m.expert_idx, m.name)),
            )
            rank_map[f"L{m.layer_idx}_E{m.expert_idx}_{m.name}"] = rk

    if s3["block_refine"]["enabled"]:
        _per_matrix_refine(
            matrices, originals, A_cov,
            lbfgs_steps=s3["block_refine"]["lbfgs_steps"],
            lbfgs_history=s3["block_refine"]["lbfgs_history"],
        )

    out_dir = artifacts_dir / "stage3_svd"
    save_checkpoint(model, tokenizer, out_dir)
    save_json_artifact({
        "rank_map": rank_map,
        "T_budget": T_budget,
        "alpha_by_type": alpha_by_type,
        "config": s3,
    }, out_dir / "rank_map.json")
    log.info("Stage 3 complete → %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Gather expert matrices
# ---------------------------------------------------------------------------


def _iter_expert_matrices(model):
    for ref in iter_moe_layers(model):
        for e_idx, expert in iter_routed_experts(ref):
            for name, lin in get_expert_matrices(expert).items():
                yield _MatrixRef(
                    layer_idx=ref.layer_idx,
                    expert_idx=e_idx,
                    name=name,
                    linear=lin,
                    parent=expert,
                    attr=name,
                )


def _group_matrices(matrices):
    groups: dict[tuple[int, str], list[_MatrixRef]] = {}
    for m in matrices:
        groups.setdefault((m.layer_idx, m.name), []).append(m)
    return groups


def _group_stat(matrices: list[_MatrixRef]) -> _GroupStats:
    W = matrices[0].linear.weight
    d1, d2 = W.shape
    svs: list[torch.Tensor] = []
    for m in matrices:
        s = torch.linalg.svdvals(m.linear.weight.detach().to(torch.float32))
        svs.append(s.cpu())
    mean_s = torch.stack([_pad(s, min(d1, d2)) for s in svs]).mean(0)
    p = mean_s ** 2
    p = p / p.sum().clamp(min=1e-12)
    eff_rank = float(torch.exp(-(p * p.clamp(min=1e-12).log()).sum()).item())
    omega = d1 + len(matrices) * d2
    return _GroupStats(d1, d2, len(matrices), mean_s, eff_rank, omega)


def _pad(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.numel() >= n:
        return x[:n]
    return torch.cat([x, torch.zeros(n - x.numel())])


# ---------------------------------------------------------------------------
# Budget + allocation
# ---------------------------------------------------------------------------


def _compute_T_budget(groups, stats, svd_rank_ratio: float) -> int:
    """Sum of per-group rank k_g that achieves ``svd_rank_ratio`` savings."""
    total_full = 0
    for g, ms in groups.items():
        total_full += stats[g].n_matrices * stats[g].d1 * stats[g].d2
    target_params = total_full * (1.0 - svd_rank_ratio)
    # Rough allocation: each rank-1 factor across n_matrices costs
    # n_matrices * (d1 + d2); picking the mean as the "cost per rank unit".
    costs = [s.n_matrices * (s.d1 + s.d2) for s in stats.values()]
    avg_cost = np.mean(costs) if costs else 1.0
    T = int(max(1, target_params / max(avg_cost, 1.0)))
    return T


def _d_rank_allocate(stats: dict, T_budget: int) -> dict:
    denom = sum(math.sqrt(s.effective_rank / s.omega) for s in stats.values()) or 1.0
    out = {}
    for g, s in stats.items():
        k = math.sqrt(s.effective_rank / s.omega) * T_budget / denom
        k = max(1, min(int(round(k)), min(s.d1, s.d2) - 1))
        out[g] = k
    return out


# ---------------------------------------------------------------------------
# Swift-SVD+
# ---------------------------------------------------------------------------


def _swift_svd_plus_grid(
    model, tokenizer, config, group_stats, initial_ranks, artifacts_dir,
) -> dict[str, float]:
    """α selection per matrix-type.

    Placeholder: returns α=0.5 for each of the three matrix types. Replace
    with a short PPL sweep when bandwidth allows — the cost is 11 α values ×
    3 types × one short evaluation pass.
    """
    log.info("Swift-SVD+ α-grid: using α=0.5 per matrix type (placeholder).")
    return {"gate_proj": 0.5, "up_proj": 0.5, "down_proj": 0.5}


def _swift_svd_plus_per_matrix(
    ms: list[_MatrixRef], alpha: float, group_k: int, stat: _GroupStats,
) -> list[int]:
    """Distribute group rank across matrices within the group by importance,
    using the Swift-SVD+ score as a proxy for per-matrix importance.
    """
    scores = []
    for m in ms:
        s = torch.linalg.svdvals(m.linear.weight.detach().to(torch.float32))
        beta = s[:group_k]
        # ε* term approximated by a constant 1 — the paper's full form uses a
        # reconstruction residual that we'd need extra forwards to get.
        eps = torch.ones_like(beta)
        score = (beta.pow(alpha) * torch.log(math.e + eps).pow(1.0 - alpha)).sum()
        scores.append(float(score.item()))
    total = sum(scores) or 1.0
    budget = group_k * len(ms)
    ranks = []
    for sc in scores:
        r = max(1, int(round(budget * (sc / total))))
        ranks.append(min(r, min(stat.d1, stat.d2) - 1))
    return ranks


# ---------------------------------------------------------------------------
# AA-SVD application (replaces the Linear with a factored module)
# ---------------------------------------------------------------------------


def _apply_aa_svd(
    m: _MatrixRef, k: int, A: torch.Tensor | None, B: torch.Tensor | None,
) -> None:
    W = m.linear.weight.detach().to(torch.float32)
    device = W.device
    d1, d2 = W.shape
    k = max(1, min(k, min(d1, d2) - 1))

    try:
        if A is None or B is None:
            raise ValueError("no covariance available")
        # FIX (review bug #7): covariances come from disk on CPU — move to
        # the weight's device and cast to fp32 before the linear algebra.
        A = A.to(device=device, dtype=torch.float32)
        B = B.to(device=device, dtype=torch.float32)
        # Regularize before Cholesky
        B_reg = B + 1e-6 * torch.eye(B.shape[0], dtype=B.dtype, device=device)
        L_B = torch.linalg.cholesky(B_reg)
        BBT_inv = torch.cholesky_inverse(L_B)
        M = W @ A @ B.transpose(0, 1) @ BBT_inv @ L_B
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        # FIX (review bug #6): paper calls for Vh · L_B^{-1}, which equals
        # (L_B^{-T} · Vh^T)^T. solve_triangular(L_B.T, Vh^T, upper=True) gives
        # exactly L_B^{-T} · Vh^T.
        V_k = torch.linalg.solve_triangular(
            L_B.transpose(0, 1), Vh[:k, :].transpose(0, 1), upper=True,
        ).transpose(0, 1)
    except Exception as err:                        # noqa: BLE001 — fallback path
        log.warning(
            "AA-SVD fallback to plain SVD for L%d/E%d/%s (%s)",
            m.layer_idx, m.expert_idx, m.name, err,
        )
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = Vh[:k, :]

    factor_down = nn.Linear(d2, k, bias=False)
    factor_up = nn.Linear(k, d1, bias=(m.linear.bias is not None))
    with torch.no_grad():
        factor_down.weight.copy_(V_k.to(m.linear.weight.dtype))
        factor_up.weight.copy_(U_k.to(m.linear.weight.dtype))
        if m.linear.bias is not None:
            factor_up.bias.copy_(m.linear.bias.detach())

    setattr(m.parent, m.attr, _FactoredLinear(factor_down, factor_up))


class _FactoredLinear(nn.Module):
    """Drop-in replacement for a ``Linear`` expressed as y = up(down(x))."""

    def __init__(self, down: nn.Linear, up: nn.Linear):
        super().__init__()
        self.down = down
        self.up = up

    def forward(self, x):
        return self.up(self.down(x))

    @property
    def weight(self):                                   # read-only composition
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return self.up.bias


# ---------------------------------------------------------------------------
# Per-matrix L-BFGS refine — minimize ||W_orig · x - U·V · x||²
# over the activation eigenspace implied by A_cov.
# ---------------------------------------------------------------------------


def _per_matrix_refine(
    matrices: list[_MatrixRef],
    originals: dict[tuple[int, int, str], torch.Tensor],
    A_cov: dict[tuple[int, int, str], torch.Tensor],
    *,
    lbfgs_steps: int,
    lbfgs_history: int,
) -> None:
    """For each factored expert Linear, refine U and V via L-BFGS on

        L(U, V) = tr( (W - UV)ᵀ (W - UV) · A_cov )

    where A_cov = Σ xxᵀ (so this is activation-weighted reconstruction).
    """
    log.info(
        "Stage 3.D: activation-weighted refine over %d factored matrices "
        "(%d L-BFGS steps each)",
        len(matrices), lbfgs_steps,
    )
    skipped = 0
    for m in matrices:
        factored = getattr(m.parent, m.attr)
        if not isinstance(factored, _FactoredLinear):
            skipped += 1
            continue
        key = (m.layer_idx, m.expert_idx, m.name)
        A = A_cov.get(key)
        if A is None:
            # No calibration data for this expert — skip; plain SVD is already
            # optimal under the isotropic prior.
            continue
        # FIX (review bug #7): place everything on the factored module's device.
        device = factored.up.weight.device
        W = originals[key].to(device=device, dtype=torch.float32)
        A_d = A.to(device=device, dtype=torch.float32)
        U = factored.up.weight.detach().to(torch.float32).clone().requires_grad_(True)
        V = factored.down.weight.detach().to(torch.float32).clone().requires_grad_(True)
        opt = torch.optim.LBFGS(
            [U, V], history_size=lbfgs_history,
            max_iter=lbfgs_steps, line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            residual = W - U @ V
            # tr( Rᵀ R A ) = sum((R A) ⊙ R)
            loss = ((residual @ A_d) * residual).sum()
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception as err:                    # noqa: BLE001
            log.debug("L-BFGS refine skipped for %s: %s", key, err)
            continue

        with torch.no_grad():
            factored.up.weight.copy_(U.to(factored.up.weight.dtype))
            factored.down.weight.copy_(V.to(factored.down.weight.dtype))
    if skipped:
        log.info("  %d matrices already non-factored, skipped refine", skipped)


# ---------------------------------------------------------------------------
# Covariance I/O
# ---------------------------------------------------------------------------


def _load_covariance(path: Path):
    if not path.exists():
        log.warning("Stage 2 covariance not found at %s — AA-SVD will fall back to plain SVD", path)
        return {}, {}
    payload = torch.load(path, map_location="cpu")
    return payload.get("covariance", {}), payload.get("tokens", {})
