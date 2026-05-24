"""D-Rank rank allocation (S3-3 of the Stage 3 plugin-architecture refactor).

Home of the D-Rank effective-rank allocation logic relocated VERBATIM from
the legacy ``stage3_svd.py`` monolith:

* ``_GroupStats`` — the ``@dataclass`` holding per-(layer, matrix) group stats;
* ``_group_stat`` — computes per-group statistics (whitened SVD, effective
  rank, omega) for D-Rank allocation (paper 2509.25622 Eq. 1);
* ``_pad`` — the sole private helper of ``_group_stat`` (pads/truncates a
  singular-value vector to a fixed length); called only inside ``_group_stat``;
* ``_compute_T_budget`` — solves the global rank budget T_budget from the
  target SVD rank ratio;
* ``_d_rank_allocate`` — distributes T_budget across all (layer, matrix)
  groups, with optional per-projection-weight biasing.

All five symbols are byte-identical copies of the monolith bodies; the monolith
re-imports them (``# noqa: F401`` block in ``stage3_svd.py``) so ``run()`` and
external callers/tests keep their existing import paths.

Circular-import note (mirror of ``stage3/plugins/covariance_collection.py``):
this module imports only stdlib / numpy / torch / ``...pipeline.context`` —
NEVER from ``stage3_svd`` or ``stage3.orchestrator``. ``stage3_svd`` imports
*this* module at load time, so a module-top ``from ..stage3_svd import ...``
here would deadlock the import; nothing in this module does that.

Unlike S3-2, the relocated D-Rank code has NO monolith-resident dependency:
``_cov_lookup`` stays in the monolith and is NEVER called by any symbol here —
``A_g`` arrives pre-resolved as a parameter to ``_group_stat``. So this module
does not import anything from the monolith at all.

``DRankAllocatePlugin`` is registered-but-INERT at S3-3 — no walk or test
invokes its ``allocate_ranks`` hook. S3-7 wires it into the live Stage 3
plugin sequencer.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


@dataclass
class _GroupStats:
    d_out: int
    d_in: int
    n_experts: int
    singular_values_mean: torch.Tensor
    effective_rank: float
    omega: float


# ---------------------------------------------------------------------------
# Group stats + allocations
# ---------------------------------------------------------------------------


def _group_stat(n_experts: int, bank, A_g: torch.Tensor | None = None) -> _GroupStats:
    """Per-group statistics for D-Rank allocation.

    Spec §6 Phase B.1/B.2 (paper 2509.25622 Eq. 1): the effective rank must
    be computed from the **whitened** SVD `sv(S_g · W_g^T)` where
    `S_g = chol(A_g)` (FP64) and `A_g = X_g^T X_g` is the group-average
    pre-prune input covariance from Stage 2. Raw `sv(W)` is not
    input-distribution-aware.

    When `A_g` is None (no Stage 2 covariance available), fall back to raw
    `sv(W)` and warn — this is a degraded path that should only fire on
    test fixtures or Stage 2 reruns.
    """
    d_out, d_in = bank.shape()
    L_A = None
    if A_g is not None:
        try:
            A64 = A_g.to(torch.float64)
            A64 = 0.5 * (A64 + A64.T)
            jitter = 1e-6 * A64.diag().mean().clamp_min(1e-12) * torch.eye(
                A64.shape[0], dtype=torch.float64, device=A64.device)
            L_A = torch.linalg.cholesky(A64 + jitter).to(torch.float32)
        except Exception as exc:
            log.warning("D-Rank whitening: Cholesky on A_g failed (%s); "
                        "falling back to raw SVD for this group.", exc)
            L_A = None

    svs: list[torch.Tensor] = []
    for e in range(n_experts):
        W = bank.get(e).detach().to(torch.float32)  # [d_out, d_in]
        if L_A is not None:
            # Spec §6 Step B.2: σ_i = sv(S_g · W_g^T). PyTorch stores W as
            # [d_out × d_in]; transpose to [d_in × d_out] then S @ that
            # gives the whitened operator whose singular values match the
            # paper's whitened spectrum.
            M = L_A @ W.T  # [d_in, d_out]
            s = torch.linalg.svdvals(M)
        else:
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


class DRankAllocatePlugin:
    """Stage 3 D-Rank rank-allocation plugin (S3-3 — registered-but-INERT).

    Owns the D-Rank effective-rank allocation phase: per-(layer, matrix) group
    stats (whitened SVD, effective rank, omega), the global rank budget
    ``T_budget`` derived from ``decomposition.svd_rank_ratio``, and the
    distribution of that budget across all groups (paper 2509.25622, Eq. 1).
    The phase logic lives in the module-level ``_group_stat`` /
    ``_compute_T_budget`` / ``_d_rank_allocate`` relocated verbatim from the
    monolith.

    S3-3 wires this class into the plugin registry as metadata only — no walk
    or test invokes ``allocate_ranks``. S3-7 plugs the hook into the live
    Stage 3 plugin sequencer.
    """

    name = "d_rank_allocate"
    paper = "D-Rank effective-rank allocation (paper 2509.25622, Eq. 1)."
    config_key = "stage3_svd.d_rank.per_projection_weight"
    reads: tuple[str, ...] = ("group_stats", "decomposition", "config")
    writes: tuple[str, ...] = ("ranks", "T_budget")
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — D-Rank allocation is UNCONDITIONAL.

        Every Stage 3 run needs a rank budget and a per-group allocation, so
        this phase always runs. ``config_key`` only *biases* the allocation
        (the optional per-projection weights) — it does not gate the plugin as
        a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def allocate_ranks(self, ctx: PipelineContext) -> None:
        """Phase hook — D-Rank rank allocation (S3-7 wiring surface).

        INERT at S3-3: no orchestrator walk or test invokes this hook. S3-7
        replaces the Stage 3 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline
        ``_compute_T_budget`` / ``_d_rank_allocate`` calls. The body reads the
        inputs off ``ctx`` and delegates to the relocated functions.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. ``config`` is optional (the per-projection-weight
        # biasing defaults to empty) so it is has()-guarded — get() raises
        # KeyError on an unset slot.
        group_stats = ctx.get("group_stats")
        decomposition = ctx.get("decomposition")
        config = ctx.get("config") if ctx.has("config") else {}
        proj_weights = (
            config.get("stage3_svd", {}).get("d_rank", {}).get(
                "per_projection_weight", {})
        )
        T_budget = _compute_T_budget(group_stats, decomposition.svd_rank_ratio)
        ranks = _d_rank_allocate(group_stats, T_budget, proj_weights=proj_weights or None)
        ctx.set("T_budget", T_budget)
        ctx.set("ranks", ranks)
