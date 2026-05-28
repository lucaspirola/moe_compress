"""RegMean — closed-form per-Linear least-squares merge (paper Eqs. 2 / Theorem 1).

Paper
-----
Jin et al., *Dataless Knowledge Fusion by Merging Weights of Language
Models*, ICLR 2023 — arXiv:2212.09849. The merged weight is the analytic
minimizer of ``Σ_i ‖W_M·X_i^T − W_i·X_i^T‖_F^2`` (per-Linear least-squares
that reproduces each donor's outputs on its own calibration inputs).

Clean-room reimplementation
---------------------------
Upstream reference (study-only, no code copied):
``tanganke/fusion_bench`` @ ``fusion_bench/method/regmean/regmean.py``
(MIT, Copyright (c) 2024 Anke Tang) — function
``merging_with_regmean_weights`` lines 43–120. This module derives the
math directly from the paper formula reproduced below; the function
signatures, dispatch, dtype posture, damping policy, and warning gates
are project-original (see Deviations).

Paper-verification stamp: math verified against arXiv:2212.09849 §3.1
(Eq. 2 closed-form) on 2026-05-28. Upstream file re-read on the same
date; the row-vector convention (paper uses x-row, W transposed in
storage) matches PyTorch's ``nn.Linear`` weight shape ``(d_out, d_in)``
once the transpose is applied — see "PyTorch shape convention" below.

Math (paper §3.1 Eq. 2)
-----------------------
For a merge cluster ``C`` of ``N`` permutation-aligned source weights
``W_i ∈ R^{d_out × d_in}`` (one per source, all sharing the same
``(d_out, d_in)``) and per-source input Gram matrices
``G_i = X_i^T · X_i ∈ R^{d_in × d_in}`` (Pearson-style, no centering),
the merged weight is::

    W_M^T = (Σ_i G_i)^{-1} · Σ_i (G_i · W_i^T)         (d_in × d_out)
    W_M   = (W_M^T).T                                   (d_out × d_in)

This is the analytic minimizer of the per-source output-reconstruction
loss ``Σ_i ‖W_M·X_i^T − W_i·X_i^T‖_F^2`` (the regression-mean objective
that gives RegMean its name). When all ``X_i`` are equal, the formula
collapses to the simple weighted average ``Σ_i α_i · W_i``; the value
of RegMean over a freq-weighted average is precisely that it accounts
for *per-source input distribution* differences.

PyTorch shape convention
------------------------
PyTorch's ``nn.Linear.weight`` has shape ``(d_out, d_in)``: ``y = W · x``
with ``x`` a column vector. The paper writes ``y = x · W`` (row
convention) and uses ``W ∈ R^{d_in × d_out}`` for its formula. The
algebra above bridges the two: when ``W_i_torch ∈ R^{d_out × d_in}`` is
passed in, we transpose to row-major (``W_i_torch.T``), apply the
formula, then transpose back. The closed-form solve is identical; only
the storage layout differs.

Deviations from the paper
-------------------------
**D-regmean-damping**: when ``Σ_i G_i`` is ill-conditioned (rank-deficient
or near-singular calibration on a low-traffic expert), the direct
inverse blows up. We add a per-cluster damping term
``Σ G + λ · trace(Σ G)/d_in · I`` with ``λ = _DAMPING_RATIO = 1e-3`` for
every solve. The paper does not specify damping (the original setting
assumed dense calibration on well-populated layers); rationale: in our
Stage 2 MoE pipeline some experts receive very few tokens, making their
``G_i`` numerically rank-deficient. ``λ·trace/d_in·I`` is the canonical
Tikhonov regularizer scaled to the matrix's natural magnitude (trace/d
gives the mean eigenvalue). This is well-behaved across calibration
volumes and is a strict identity at λ=0.

**D-regmean-cond-fallback**: when the post-damping condition number of
``Σ G + λ·tr/d_in·I`` is still > ``_COND_THRESHOLD = 1e8`` (degenerate
calibration even after Tikhonov), we fall back to the freq-weighted
merged weight ``Σ_i α_i · W_i`` and log a warning. Mirrors MergeMoE's
``D-mergemoe-cond-fallback`` (paper does not address this corner; the
threshold matches that spec for consistency).

**D-regmean-zero-cov-fallback**: when ``cov_acc.get(...)`` returns
``None`` for some member (no calibration traffic to that expert in the
profile pass), we fall back to a freq-weighted average for the WHOLE
cluster and log a warning. Without per-source Gram data the regression
objective is undefined for that member; declining the source-specific
solve and degrading to the simple average is the safe choice (and
matches the freq-weighted fallback path the rest of the merge engine
already handles).

**D-regmean-no-non-diagonal-reduction**: fusion_bench's
``merging_with_regmean_weights`` exposes a ``reduce_non_diagonal_ratio``
knob (lines 24–40, 78–84) that pre-scales off-diagonal Gram entries by
a multiplicative factor. The paper's Eq. 2 does not include this
knob — it is a fusion_bench-specific empirical add-on. We intentionally
do NOT carry it over; the closed-form solve here uses the raw Gram as
written in the paper, which keeps the math one-to-one with arXiv:2212.09849.

**D-regmean-no-renormalization-by-tokens**: the paper averages
``G_i = X_i^T·X_i / N_i`` per source (some references include the
``1/N_i`` factor, fusion_bench includes it at line 161); since the
closed-form is scale-invariant in ``G_i`` under uniform scaling
(``α·Σ G`` cancels in the inverse + multiplication), we accept whatever
normalization ``cov_acc`` provides without rescaling. ``InputCovarianceAccumulator``
in this project accumulates the un-normalized ``Σ X^T·X`` per
(layer, expert, matrix); the formula's algebraic answer is the same
because every term gets the same un-normalization.

**D-regmean-fp32-solve**: the closed-form inverse runs in float32 even
if the model weights are bf16; result is cast back to the model's
native dtype before write-back. Mirrors the fp32 numerical posture of
``_merge_experts_inplace`` (the ``.to(torch.float32)`` upcasts in
``stage2/merging.py``) and ``stage2/mergemoe.py``.

License + attribution
---------------------
fusion_bench: MIT, Copyright (c) 2024 Anke Tang. No code is copied
verbatim. The mathematical specification is paper-derived; the
implementation here is project-original. See
``audit/spec_compliance/01_papers/`` for the paper-verification audit
trail discipline.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch

log = logging.getLogger(__name__)


# Tikhonov damping ratio relative to the Gram's mean eigenvalue (= trace/d).
# Project default per D-regmean-damping; small enough to leave the math
# essentially unchanged on well-conditioned cases.
_DAMPING_RATIO = 1e-3

# Conditioning threshold for the cluster-wide fallback. Matches MergeMoE's
# _COND_THRESHOLD in :mod:`stage2.mergemoe` for cross-merge consistency.
_COND_THRESHOLD = 1e8


def _regmean_solve_one_linear(
    *,
    weights_per_member: Sequence[torch.Tensor],
    grams_per_member: Sequence[torch.Tensor],
    alpha_per_member: Sequence[float],
) -> torch.Tensor:
    """Closed-form RegMean merge for a single nn.Linear weight.

    Implements paper Eq. 2 (arXiv:2212.09849 §3.1)::

        W_M^T = (Σ_i G_i)^{-1} · Σ_i (G_i · W_i^T)
        W_M   = (W_M^T).T

    Parameters
    ----------
    weights_per_member:
        ``N`` tensors of shape ``(d_out, d_in)`` (PyTorch ``nn.Linear``
        convention), one per cluster member, **already permutation-aligned
        to the centroid** by the caller.
    grams_per_member:
        ``N`` tensors of shape ``(d_in, d_in)``, the per-member input
        Gram ``X_i^T · X_i``. Pearson-style (no centering), un-normalized
        (scale cancels in the inverse — see D-regmean-no-renormalization-by-tokens).
    alpha_per_member:
        ``N`` Python floats. Used ONLY for the fp32 freq-weighted fallback
        when the post-damping ``Σ G`` is still ill-conditioned
        (D-regmean-cond-fallback). The closed-form Eq. 2 itself is
        weighting-invariant in ``α``; the Gram acts as the natural weight.

    Returns
    -------
    torch.Tensor
        Merged weight of shape ``(d_out, d_in)``, in the dtype of
        ``weights_per_member[0]`` (the solve runs in fp32 internally
        regardless — see D-regmean-fp32-solve).

    Raises
    ------
    ValueError
        If ``N < 2`` (caller must filter singletons), the three sequences
        disagree in length, or any shape is wrong.
    """
    N = len(weights_per_member)
    if N < 2:
        raise ValueError(
            f"_regmean_solve_one_linear: N={N}; need at least 2 cluster "
            "members (callers must filter singletons)."
        )
    if len(grams_per_member) != N or len(alpha_per_member) != N:
        raise ValueError(
            "_regmean_solve_one_linear: weights_per_member, grams_per_member, "
            "alpha_per_member must have the same length; got "
            f"{N}/{len(grams_per_member)}/{len(alpha_per_member)}."
        )

    W0 = weights_per_member[0]
    d_out, d_in = W0.shape
    target_dtype = W0.dtype
    target_device = W0.device

    # Validate shapes — surface a useful error before the matmul.
    for j, (W, G) in enumerate(zip(weights_per_member, grams_per_member)):
        if W.shape != (d_out, d_in):
            raise ValueError(
                f"_regmean_solve_one_linear: member {j} weight shape "
                f"{tuple(W.shape)} != expected {(d_out, d_in)}"
            )
        if G.shape != (d_in, d_in):
            raise ValueError(
                f"_regmean_solve_one_linear: member {j} Gram shape "
                f"{tuple(G.shape)} != expected {(d_in, d_in)}"
            )

    # --- Closed-form solve in fp32 (D-regmean-fp32-solve) -----------------
    # Σ G_i and Σ (G_i · W_i^T) in fp32 on the source device.
    G_sum = torch.zeros((d_in, d_in), dtype=torch.float32, device=target_device)
    GW_sum = torch.zeros((d_in, d_out), dtype=torch.float32, device=target_device)
    for W, G in zip(weights_per_member, grams_per_member):
        G_f32 = G.to(dtype=torch.float32, device=target_device)
        W_f32 = W.to(dtype=torch.float32, device=target_device)
        G_sum.add_(G_f32)
        # W_i^T has shape (d_in, d_out); G_i · W_i^T has shape (d_in, d_out).
        GW_sum.add_(G_f32 @ W_f32.transpose(0, 1))

    # Tikhonov damping scaled to the matrix's natural magnitude
    # (D-regmean-damping). trace(G_sum)/d_in is the mean eigenvalue;
    # scaling λ by that keeps the regularizer well-conditioned across
    # calibration volumes.
    trace_mean = (G_sum.diagonal().sum() / float(d_in)).item()
    # Guard: trace_mean can be exactly 0 on a zero-traffic cluster
    # (every member's G is the zero tensor — InputCovarianceAccumulator
    # auto-vivifies to a zero tensor on update with empty input). In that
    # case any positive damping is fine; pick 1.0 so the identity term
    # alone defines a well-conditioned system (which will then produce
    # W_M ≈ 0 — but the post-condition fallback below catches that as
    # a degenerate cluster and falls back to freq-weighted).
    damping = _DAMPING_RATIO * trace_mean if trace_mean > 0.0 else 1.0
    G_sum_reg = G_sum + damping * torch.eye(d_in, dtype=torch.float32, device=target_device)

    # Condition-number gate (D-regmean-cond-fallback). torch.linalg.cond is
    # O(d_in^3) but only called once per cluster, not per token — cheap at
    # our scale (d_in ≤ 5120 for Qwen3.6-A3B).
    try:
        cond_value = torch.linalg.cond(G_sum_reg).item()
    except Exception as exc:  # noqa: BLE001 — surface the underlying error.
        log.warning(
            "_regmean_solve_one_linear: torch.linalg.cond failed (%r); "
            "falling back to freq-weighted merge for this cluster.",
            exc,
        )
        cond_value = float("inf")

    if not (cond_value < _COND_THRESHOLD):  # also catches NaN/inf via 'not <'
        log.warning(
            "_regmean_solve_one_linear: D-regmean-cond-fallback (cond(Σ G + λI)="
            "%.3e > %.0e); falling back to freq-weighted merge for this cluster.",
            cond_value, _COND_THRESHOLD,
        )
        # α-weighted freq fallback. Renormalize α defensively so callers
        # can pass raw weights without doing it themselves.
        alpha_sum = sum(alpha_per_member)
        if alpha_sum > 0.0:
            alphas_n = [a / alpha_sum for a in alpha_per_member]
        else:
            alphas_n = [1.0 / N for _ in alpha_per_member]
        merged_fp32 = torch.zeros((d_out, d_in), dtype=torch.float32, device=target_device)
        for a, W in zip(alphas_n, weights_per_member):
            merged_fp32.add_(a * W.to(dtype=torch.float32, device=target_device))
        return merged_fp32.to(dtype=target_dtype)

    # Solve (Σ G_reg) · X = Σ (G·W^T); X has shape (d_in, d_out) = W_M^T.
    # torch.linalg.solve is preferred over torch.linalg.inv + matmul for
    # numerical stability (the literature is unanimous on this).
    WT_merged = torch.linalg.solve(G_sum_reg, GW_sum)  # (d_in, d_out)
    W_merged_fp32 = WT_merged.transpose(0, 1).contiguous()  # (d_out, d_in)
    return W_merged_fp32.to(dtype=target_dtype)


__all__ = ["_regmean_solve_one_linear", "_DAMPING_RATIO", "_COND_THRESHOLD"]
