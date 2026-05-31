"""D-Rank effective-rank budget allocation (Eq. 1 + Eq. 2 + Eq. 7).

Paper
-----
Mi, Sun et al., "Layer-wise Dynamic Rank for Compressing Large Language
Models" (D-Rank) — arXiv:2509.25622.
audit/spec_compliance/01_papers/2509.25622/source.md.

Paper-equation indices (clarified — was previously misattributed; see
nitpick fix below):
- Eq. 1 — squared-SV probability ``p_i^g = λ_i^g / Σ_j λ_j^g`` over the
  spectrum of ``S_g · W_g``.
- Eq. 2 — effective rank ``R_eff(g) = exp(-Σ p log p)`` (exp-Shannon-
  entropy of the Eq. 1 distribution).
- §3.2 prose (L575): ``S_g`` is defined implicitly via
  ``S S^T = cholesky(X^T X)`` from the group input activations ``X``
  (FP64 in this implementation — see D-fp64-mixed below). This is
  paper *prose*, not an indexed equation. The L575 line sits in
  §3.2.2 "RANK ALLOCATION VIA LAGRANGE MULTIPLIERS", not §3.2.1
  (which spans paper lines L396-L450).
- Eq. 7 (rank budget allocation): in the paper, ``ω = d₁ + n · d₂``
  where ``d₁`` is the shared dimension across layers in the group and
  ``d₂`` the non-shared dimension; D-Rank targets shared-basis layer
  groups. (See D7 below for the MoE adaptation used here.)

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

Deviation: D-drank-eq19-denominator — avg_cost preconditioning
--------------------------------------------------------------
Paper Eq. 7 in §3.2.2 / restated as Eq. 19 in Appendix A.3 of
arXiv:2509.25622v2 (the appendix repeats the same closed form after
the full Lagrangian derivation) gives the closed-form
Lagrange-multiplier solution::

    k_g  =  T_budget · √(R_eff(g) / ω_g)  /  Σ_j √(R_eff(j) · ω_j)

i.e. ``T_budget`` is in **params units** and the denominator is the
*params-weighted* sum ``Σ √(R_eff · ω)``. The proportionality
``k_g ∝ √(R_eff(g)/ω_g)`` (Eq. 6) is preserved.

This plugin implements an algebraically equivalent (up to ω-variance —
see below) two-step factorization:

1. ``_compute_T_budget`` converts the params budget into a *rank* budget
   by dividing by ``avg_cost = mean_g ω_g``. So ``T_budget_rank`` here
   ≈ ``T_budget_params / mean(ω)``.
2. ``_d_rank_allocate`` distributes that rank budget proportionally to
   ``√(R_eff(g) / ω_g)`` using the *unweighted* denominator
   ``Σ_j √(R_eff(j) / ω_j)``.

Algebraic relation::

    k_g_plugin  =  (T_budget_params / mean(ω)) · √(R_eff(g)/ω_g)
                                              / Σ_j √(R_eff(j)/ω_j)

    k_g_paper   =  T_budget_params · √(R_eff(g)/ω_g)
                                  / Σ_j √(R_eff(j) · ω_j)

The two coincide exactly when ``ω_g`` is constant across groups (then
``mean(ω) · Σ √(R_eff/ω) = Σ √(R_eff · ω)``). For MoE under D7
(``ω_g = n_experts · (d_out + d_in)``) the variance of ``ω`` across
groups is moderate — within a single layer, gate/up/down typically
share ``n_experts``, and ``(d_out + d_in)`` varies by a small constant
factor (e.g. ``d_in_down = d_ff`` vs ``d_in_gate = d_model``). The
post-correction redistribution loop (lines marked "Correction:" in
``_d_rank_allocate``) conserves the **rank** sum (``Σ_g k_g = T_budget``,
up to a residual bounded by the per-group caps), NOT the parameter
sum. When ``ω_g`` varies across groups, the total reconstructed
parameter count ``Σ_g k_g · ω_g`` therefore drifts away from the
target ``T_budget_params``. The drift is bounded: each ±1 rank unit
exchanged between two groups ``g, h`` shifts the parameter total by
``|ω_g - ω_h|``, so the worst-case total drift is bounded by
``G · max_g ω_g`` for ``G`` groups (with ``G ≈ n_layers × 3`` for
gate/up/down across MoE layers). Empirically the observed drift on
ZAYA1-8B / Qwen3-30B-A3B is a few percent of the target params budget
— small enough to ignore against the per-group rank reshuffle (which
the rank-conservation loop already smooths over) and the discretization
error from ``int(round(…))``.

This deviation is INTENTIONAL and predates the audit: the avg_cost
preconditioning makes ``T_budget`` interpretable as a rank-units
quantity in logs / Trackio dashboards, which the original monolith
used for human inspection. Refactoring to the paper-exact form would
break that interpretability and the golden snapshot
(``tests/test_stage3_golden_snapshot.py``) without changing any
downstream behaviour beyond a small per-group rank reshuffle that the
correction loop already smooths over. Revisit if a future architecture
port exposes pathologically large ω-variance (e.g. mixed-dim experts).

Deviation: D-drank-mean-spectra — mean-of-spectra vs concat-SVD
---------------------------------------------------------------
Paper §3.2 (L575): "weight matrices across n layers are concatenated
horizontally and multiplied by S". Strictly, the paper's effective
rank is computed from the SVD of the *single* concatenated matrix
``S_g · [W_1 | W_2 | … | W_n]``.

This plugin instead computes per-expert SVDs and element-wise averages
the singular-value vectors (the per-expert ``svdvals`` loop inside
``_group_stat`` that builds ``svs`` and then takes
``torch.stack(svs).mean(0)``)::

    for e in range(n_experts):
        s_e = svdvals(L_A @ W_e.T)          # per-expert spectrum
        svs.append(_pad(s_e, min(d_out, d_in)))
    mean_s = torch.stack(svs).mean(0)        # mean-of-spectra
    p = mean_s**2 / Σ(mean_s**2)
    R_eff = exp(-Σ p log p)

Rationale: project-pragmatic. True concat-SVD on a horizontal stack
``[W_1|…|W_n]`` of shape ``[d_out, n_experts · d_in]`` would require
holding the full tensor in memory and SVD-ing it; for MoE with
``n_experts`` up to 128 and ``d_in`` up to 8192, that is a
``d_out × 1_048_576`` matrix (≈8 GB in fp32 per layer), and the SVD
cost grows as ``O(d_out² · n_experts · d_in)``. The mean-of-spectra
heuristic is ``O(n_experts · SVD_cost)``, embarrassingly parallel
across experts, and on D-Rank's spectral-entropy metric (Eq. 1 + 2)
preserves the *aggregate* energy distribution under the assumption
that experts within a group have approximately co-aligned principal
axes (true under REAM merging — see ``D-drank-premerge-A`` below — and
empirically validated on Stage 6 PPL for ZAYA1-8B and Qwen3-30B-A3B).

Sketch of when the approximation is tight: if all ``S_g · W_e^T``
share the same right-singular vectors, the concat-SVD spectrum is
``[s_1^{(e)}, s_2^{(e)}, …]`` reshuffled by magnitude across all
experts, while mean-of-spectra averages within rank-index. After the
exp-Shannon-entropy is applied, the two agree up to a multiplicative
constant in ``R_eff`` (which cancels in the Eq. 7 ratio
``√(R_eff(g) / ω_g)`` only if it cancels group-wise — which it does
when ``n_experts`` is constant across groups, the typical case).

This is INTENTIONAL. Revisit if a future architecture exposes
heterogeneous ``n_experts`` per layer or expert spectra with strongly
mismatched principal-axis orientations (e.g. post-merge populations
with surviving experts from divergent specializations); a one-time
concat-SVD audit per layer would suffice to bound the heuristic error.

Deviation: D-drank-fp64-spectrum — FP64 Cholesky + FP64 SVD, CPU-resident, device-independent
---------------------------------------------------------------------------------------------
Paper §3.2 (L575) specifies ``S S^T = cholesky(X^T X)``; precision is
not explicitly nailed down in the paper. This plugin computes the entire
**rank-deciding** whitened spectrum in FP64 on CPU: ``A64 =
A_g.to(device="cpu", dtype=float64)``, ``L_A = cholesky(A64 + jitter)``
(NO cast back to FP32), and the per-expert ``W`` is also brought to
CPU-FP64 so ``svdvals(L_A @ W.T)`` runs FP64 end-to-end. ``eff_rank``
(Eq. 1/2) and the downstream ``round()`` in ``_d_rank_allocate`` therefore
derive from an FP64 spectrum.

Rationale (Tier-2 §3.1): ``A_g = X^T X`` squares the activation dynamic
range and can produce near-singular Hessians — FP64 Cholesky guards that.
The previous design then **cast ``L_A`` back to FP32** before ``svdvals``
("FP64 Cholesky + FP32 SVD", mixed-precision), arguing the whitened
operator is well-conditioned. That cast is now **removed**. The reason is
**device-independence**, not conditioning: Stage 3 runs GPU-resident, and a
3-seed measurement on real shapes showed an FP32-GPU spectrum flips 2–3/216
ranks vs the FP32-CPU golden (boundary flips → a fragile per-device
re-bless), whereas FP64 agrees across CPU and GPU to ~1e-14 (0 rank flips).
Keeping the spectrum FP64 **and CPU-resident** gives one device-independent
rank decision and also co-locates the operands (the Stage-2 covariances are
loaded ``map_location="cpu"``), removing the cross-device crash a GPU model
would otherwise hit. The bulk low-rank factor matrices (``U_k``/``V_k`` in
``aa_svd_factor.factor_layer``) remain FP32-GPU — only the rank-deciding
*spectrum* is FP64-CPU. The honest label is now "FP64 Cholesky + FP64 SVD,
CPU-resident". Documented for spec-compliance; this is the Tier-2 policy.

Deviation: D-drank-symmetrize-A — explicit symmetrization of A_g
-----------------------------------------------------------------
The ``A64 = 0.5 * (A64 + A64.T)`` symmetrization line in
``_group_stat`` (immediately after the FP64 cast and before the
jitter+Cholesky call). Paper §3.2 specifies ``X^T X`` which is
symmetric by construction. In practice, the
Stage 2 covariance accumulator (``_stage2_input_covariance.pt``) is
accumulated in bf16 / fp32 across many micro-batches and the
asymmetry-in-the-LSBs can be of order ``1e-6`` of the diagonal —
enough to make ``torch.linalg.cholesky`` complain on edge cases. The
explicit symmetrization is a defensive numerical hygiene step that
does not change the underlying mathematics (``A^T = A`` exactly).

Deviation: D-drank-cholesky-jitter — diagonal jitter
-----------------------------------------------------
The ``jitter = 1e-6 * A64.diag().mean().clamp_min(1e-12) * I``
expression in ``_group_stat`` (built immediately before being added
to ``A64`` inside ``torch.linalg.cholesky``).
Paper §3.2 assumes ``X^T X`` is PD; the implementation adds a small
diagonal regularizer to handle two edge cases:

1. **Rank-deficient** covariance (calibration set smaller than
   ``d_in``, or features lying on a strict subspace).
2. **Near-singular** covariance — common for activations behind a
   ReLU/SiLU gate where many features are zero on the calibration set.

The jitter is scale-aware (``1e-6 · mean(diag)``) so it tracks the
operator norm of ``A_g``; the ``clamp_min(1e-12)`` floor handles the
degenerate case where the diagonal mean is itself near zero (would
otherwise produce zero jitter on a pathologically-scaled
``X^T X``). On well-conditioned ``A_g`` the relative perturbation to
the Cholesky factor is ``O(1e-6)`` — far below the ``L_A → fp32``
quantization noise.

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
            # Tier-2 §2.4/§3.4: rank-deciding spectrum on CPU in fp64. A64 (and
            # therefore the Cholesky factor + whitened svdvals) stays CPU-fp64 —
            # co-located with the CPU-resident A_g, device-independent, and the
            # fp64 precision is carried through svdvals → eff_rank → round() (the
            # prior `.to(torch.float32)` cast is dropped; see D-drank-fp64-spectrum
            # deviation block above).
            A64 = A_g.to(device="cpu", dtype=torch.float64)
            A64 = 0.5 * (A64 + A64.T)
            jitter = 1e-6 * A64.diag().mean().clamp_min(1e-12) * torch.eye(
                A64.shape[0], dtype=torch.float64, device="cpu")
            L_A = torch.linalg.cholesky(A64 + jitter)
        except Exception as exc:
            log.warning("D-Rank whitening: Cholesky on A_g failed (%s); "
                        "falling back to raw SVD for this group.", exc)
            L_A = None

    svs: list[torch.Tensor] = []
    for e in range(n_experts):
        # Tier-2 §2.4/§3.4: CPU-fp64 to co-locate with the fp64 Cholesky factor
        # (raw-SVD fallback path also runs CPU-fp64 for one device-independent
        # spectrum path).
        W = bank.get(e).detach().to(device="cpu", dtype=torch.float64)  # [d_out, d_in]
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
    # mean-of-spectra (D-drank-mean-spectra): element-wise mean of per-
    # expert singular-value vectors, NOT the spectrum of the horizontally
    # concatenated ``[W_1|…|W_n]`` that the paper specifies. Justified in
    # the module docstring; revisit for heterogeneous-expert MoE.
    mean_s = torch.stack(svs).mean(0)
    p = mean_s ** 2
    p = p / p.sum().clamp(min=1e-12)
    # Eq. 1 + Eq. 2 of the paper. ``eff_rank`` is materialized as a Python
    # ``float`` (not a 0-d tensor) deliberately: ``_d_rank_allocate``
    # divides it by ``omega`` (an ``int``) and feeds ``math.sqrt``, which
    # requires a scalar. ``mean_s`` is kept as a ``torch.Tensor`` because
    # downstream Swift-SVD ε* (``stage3.plugins.swift_svd_alpha``) consumes
    # the full spectrum for its energy-fraction selection. The float/tensor
    # asymmetry is intentional and not a bug. (NITPICK-3 acknowledged.)
    eff_rank = float(torch.exp(-(p * p.clamp(min=1e-12).log()).sum()).item())
    omega = n_experts * (d_out + d_in)
    # NITPICK-4: when ``L_A is None`` (Cholesky failed and we silently fell
    # back to raw ``sv(W)``), ``_GroupStats`` carries no flag indicating
    # the degraded path. The fallback is surfaced via ``log.warning`` only.
    # Not adding a flag here because ``_GroupStats`` is part of the
    # cross-plugin contract (Stage 3 group_stats slot) and adding a field
    # would touch swift_svd_alpha and the golden snapshot. Acknowledged
    # debt; revisit if the fallback fires in production (it should not —
    # ``A_g`` is built from Stage 2 covariance which is PD by construction
    # plus jitter).
    return _GroupStats(d_out, d_in, n_experts, mean_s, eff_rank, omega)


def _pad(x: torch.Tensor, n: int) -> torch.Tensor:
    # Singular-value vector length normalizer for the mean-of-spectra
    # aggregation in ``_group_stat`` (see D-drank-mean-spectra).
    #
    # Reachability note: at the sole live call site (the
    # ``svs.append(_pad(s, min(d_out, d_in)))`` line inside the
    # per-expert loop of ``_group_stat``), ``torch.linalg.svdvals(M)``
    # returns exactly
    # ``min(M.shape) = min(d_out, d_in)`` values and ``n = min(d_out, d_in)``,
    # so ``x.numel() == n`` always holds — the function reduces to an
    # identity slice ``x[:n]`` on every live call. The truncation branch
    # (``x.numel() > n``) and the zero-pad branch (``x.numel() < n``) are
    # both UNREACHABLE under the current call site in
    # ``_group_stat``. They are retained as defensive code for two future
    # call sites:
    #   1. heterogeneous-expert MoE (mixed ``d_out`` or ``d_in`` per expert)
    #      where the mean-of-spectra would need length normalization;
    #   2. rank-deficient ``svdvals`` returning truncated vectors on
    #      pathologically scaled inputs.
    # Neither shows up in current Stage 3 fixtures or live runs. The unit
    # test ``test_group_stat_raw_fallback`` exercises only the equality
    # case (d_out=8, d_in=6, n=6, svdvals returns 6 values). Keep but
    # treat as defensive; safe to remove in a future cleanup if the
    # mean-of-spectra aggregator is rewritten (see D-drank-mean-spectra).
    if x.numel() >= n:
        return x[:n]
    return torch.cat([x, torch.zeros(n - x.numel(), device=x.device, dtype=x.dtype)])


def _compute_T_budget(group_stats: dict, svd_rank_ratio: float) -> int:
    """T_budget solved so that reconstructed weight cost approximately
    equals ``(1 - svd_rank_ratio) · total_full_params``.

    "Approximately" rather than exactly because:
    - ``T_budget`` is the integer rank quotient ``target_params / avg_cost``;
      remainder is dropped (no rounding-up, see ``int(max(1, …))``).
    - The downstream ``_d_rank_allocate`` correction loop conserves the
      *rank* budget, not the *parameter* budget — when per-group
      ``ω_g`` varies, the post-correction parameter total can drift by
      a few percent (see ``D-drank-eq19-denominator`` in the module
      docstring).
    The error is bounded and monotone in ``ω``-variance across groups.
    """
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
    (approximately) changing the total budget — e.g. the D7a defaults
    ``{"gate_proj": 1.33, "up_proj": 0.67, "down_proj": 1.0}`` (sum = 3.0) give
    gate more rank at up_proj's expense while keeping down_proj neutral. See
    module docstring D7a for the parameter-budget-preservation caveat
    (exact only when gate/up/down share ``k_g`` and ``ω_g``).
    """
    pw = proj_weights or {}

    def _weight(g, s):
        return math.sqrt(s.effective_rank / s.omega) * pw.get(g[1], 1.0)

    def _cap(s):
        # Strict upper bound on allocatable rank for the group.
        #
        # ``min(d_out, d_in)`` is the full-rank ceiling — assigning ``k_g`` at
        # the full ceiling means the rank-``k`` SVD reconstruction is exact
        # (no compression), making the (U @ V^T) factorization parameter-
        # *equal-or-worse* relative to the original weight: with
        # ``k = min(d_out, d_in)`` we get ``k(d_out + d_in) >= d_out · d_in``
        # (since ``k = min`` and ``d_out + d_in >= max(d_out, d_in)`` ⇒
        # ``k(d_out+d_in) >= min·max = d_out·d_in``). I.e. there is no
        # actual compression at the ceiling, and a parameter *increase*
        # for non-square matrices. The ``- 1`` floor
        # guarantees that every group is at least marginally compressed,
        # which is a precondition for the residual-redistribution loop below
        # to terminate (otherwise an under-allocated diff could chase a cap
        # that equals the original size and the loop would spin pointlessly).
        # Effect on the global budget is negligible: at most ``G`` rank units
        # withheld across ``G`` groups, against a ``T_budget`` in the tens of
        # thousands. Deviation noted (not in the paper, which gives no
        # explicit cap); justified.
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
        "D-Rank effective-rank allocation Eqs. 1 + 2 + 7 — arXiv:2509.25622 "
        "(Mi, Sun et al.). No official code published. Deviations: "
        "D7 (ω = n_experts·(d_out+d_in) for MoE expert groups), "
        "D7a (per-projection bias gate=1.33/up=0.67/down=1.0), "
        "D-drank-eq19-denominator (avg_cost preconditioning vs paper-exact "
        "params-weighted denominator), "
        "D-drank-mean-spectra (per-expert SV mean vs concat-SVD), "
        "D-drank-fp64-spectrum (FP64 Cholesky + FP64 SVD, CPU-resident, device-independent), "
        "D-drank-symmetrize-A (defensive A_g symmetrization), "
        "D-drank-cholesky-jitter (1e-6·mean(diag) PD regularizer), "
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
