"""D-Rank effective-rank budget allocation (Eq. 1 + Eq. 7).

Paper
-----
Mi, Sun et al., "Layer-wise Dynamic Rank for Compressing Large Language
Models" (D-Rank) — arXiv:2509.25622.
audit/spec_compliance/01_papers/2509.25622/source.md.

Equation 1 (FP64 Cholesky whitening): the effective rank for a group
``g`` is computed from the SVD of ``S_g · W_g`` (whitened) where
``S_g = Cholesky(X_g^T X_g)`` is the input-activation Cholesky factor.

Equation 7 (rank budget): ``ω = d₁ + n · d₂`` (where ``d₁`` is the
shared dimension across layers in the group and ``d₂`` the
non-shared dimension; the paper formulation targets shared-basis
layer groups).

This plugin's responsibilities:
- ``_group_stat`` (and its ``_pad`` helper) — computes per-(layer,
  matrix-type) group statistics: whitened SVD via fp64 Cholesky, the
  effective-rank vector, and ``ω``.
- ``_compute_T_budget`` — solves the global rank budget ``T_budget``
  from the target SVD rank ratio.
- ``_d_rank_allocate`` — distributes ``T_budget`` across all
  ``(layer, matrix-type)`` groups, with optional per-projection-weight
  biasing (see deviation D7a below).

Official code
-------------
None published — the D-Rank paper (arXiv:2509.25622, source.md) does
not link to a code repository, and a GitHub search of the first
author's profile (Zhendong Mi, Stevens Institute) returned no D-Rank
repo at retirement time (verified 2026-05). The Eq. 1 + Eq. 7
implementation here is project-original following the paper text.

Deviation: D7 — ω adapted for MoE
---------------------------------
Paper Eq. 7 is ``ω = d₁ + n · d₂``: ``n`` is the number of layers
sharing a basis, ``d₁`` is the shared dimension, ``d₂`` the
non-shared dimension. D-Rank targets shared-basis layer groups.

This plugin adapts ``ω`` for MoE expert groups:
``ω = n_experts × (d_out + d_in)`` — every expert contributes a full
``d_out × d_in`` parameter slab; ``n`` becomes the number of experts
in the group, ``d_out + d_in`` becomes the per-expert per-matrix
parameter cost.

Deviation: D7a — per-projection rank bias + ``k̄`` semantics for ε*
------------------------------------------------------------------
Paper Eq. 7 produces a single ``k_g`` per ``(layer, matrix_type)``
group; no per-projection-type multiplier. Swift-SVD
(arXiv:2604.01609) defines ``k̄ = (m·n)/(m+n)·ρ`` as the plain uniform
rank entering ε*.

This plugin scales group ranks from Eq. 7 by per-projection
multipliers ``(gate = 1.33, up = 0.67, down = 1.0)`` (sum = 3.0; the
multipliers are approximately parameter-budget-preserving — exactly
preserved when gate/up/down receive the same ``k_g`` AND share the
same ``ω_g``, which holds under SwiGLU symmetry where gate/up have
identical input dimensions; in the general case, the multipliers
redistribute rank between projection types and may shift the
post-bias parameter total by a few percent) before per-expert
redistribution. The bias-adjusted ``k̄`` also flows into the
Swift-SVD ε* computation (see :mod:`stage3.plugins.swift_svd_alpha`).

Rationale: adapted from jangq's GGUF bit-allocation insight
(``gate:up:down ≈ 4:2:3`` — see project ``397B-MLP-ASYMMETRY.md``
§3.1). SwiGLU forward couples gate errors multiplicatively via SiLU,
while down errors propagate to the residual stream. The ratio
translates the same physical asymmetry from bit space to rank space.
(*TODO: empirical re-tune from clean per-projection ``recon_rel_err``
once Stage 6 evals are available; current values inherited unchanged
from a prior bf16-bug-tainted run and are theoretically- (not
empirically-) grounded.*)

Deviation: D-drank-premerge-A — Stage 2 A-covariance reuse
----------------------------------------------------------
Paper §3.2.1 assumes the whitening factor ``S_g`` is computed from
activations of the model being compressed (post-merge for this
pipeline). This plugin uses ``A_gate_up`` and ``A_down`` from Stage 2's
``_stage2_input_covariance.pt``, collected during Stage 2 calibration
on the **pre-merge** expert population. After REAM merging, the
surviving experts produce slightly different intermediate activations
than the pre-merge experts they replaced; the down_proj input
distribution shift is the larger of the two.

Rationale: project-pragmatic. Re-running a Stage 3-specific
calibration pass to collect post-merge A would cost a full
teacher+student forward and ~140 GB of new covariance on disk, with
marginal expected impact: REAM's frequency-weighted merge (REAM Eq. 6)
preserves expected activations by construction, so the pre/post-merge
A on a per-(layer, matrix-type) average basis is close to identity
under expected-merge invariance. Stage 4's EoRA residual compensation
(see :mod:`stage4.plugins.eora_compensation`, paper 2410.21271) absorbs
any residual whitening mismatch via the activation-aware √Λ projection
on the **post-merge** Stage 4 covariance reuse. Trade-off accepted;
revisit if Stage 6 PPL regresses unexpectedly on a future architecture
port.

Naming-history note
-------------------
"Phase B" (legacy Stage 3 monolith terminology) is naming-historical.
The current plugin architecture has no phase taxonomy; new prose
drops the labels. Existing log lines / Trackio keys preserved for
dashboard back-compat.

Tool inventory (relocated verbatim):

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
    paper = (
        "D-Rank effective-rank allocation Eqs. 1 + 7 — arXiv:2509.25622 "
        "(Mi, Sun et al.). No official code published. Deviations: "
        "D7 (ω = n_experts·(d_out+d_in) for MoE expert groups), "
        "D7a (per-projection bias gate=1.33/up=0.67/down=1.0), "
        "D-drank-premerge-A (Stage 2 A-cov reuse on post-merge weights). "
        "See module docstring."
    )
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
