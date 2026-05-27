"""MergeMoE — closed-form T₁ = Q·P† merged down-projection (paper Eq. 6).

Paper
-----
Miao et al., *MergeMoE: Efficient Compression of MoE Models via Expert
Output Merging*, arXiv:2510.14436 (Oct 2025). Equations 3–6 and
Theorem 1.

Used by ``merge_step="mergemoe"`` (config-knob default is
``"freq_weighted"`` → byte-identical to legacy code). This module is
loaded **only** when the user opts into the new merge math; both
:mod:`stage2.merging` and :mod:`stage2.plugins.output_space_cost`
import ``_mergemoe_compute_merged_down`` at function scope so the
default freq-weighted path pays zero overhead.

Math (paper §3.2–§4)
--------------------
For a merge cluster ``C_i`` with ``N`` permutation-aligned experts and
frequency weights ``b_j = f_j / Σ_k f_k``:

::

    W'_G = [W_G^1; W_G^2; …; W_G^N]            (vertical stack, (N·d_int, d_hidden))
    W'_U = [W_U^1; W_U^2; …; W_U^N]            (vertical stack)
    W'_D = [b_1·W_D^1, b_2·W_D^2, …, b_N·W_D^N]  (horizontal stack, (d_hidden, N·d_int))
    T₂ = T₃ = [b_1·I, b_2·I, …, b_N·I]         (d_int, N·d_int)

The merged expert is ``E'(x) = W'_D · T₁ · (σ(T₂·W'_G·x) ⊙ (T₃·W'_U·x))``
(Eq. 3). Because ``T₂·W'_G = Σ_j b_j·W_G^j`` and ``T₃·W'_U = Σ_j b_j·W_U^j``,
the merged gate/up projections collapse to the freq-weighted average
— byte-identical to legacy code. Only the down-projection changes.

The least-squares system (Eq. 5)::

    P = σ(T₂·W'_G·X̂) ⊙ (T₃·W'_U·X̂)             shape (d_int,    T)
    Q = σ(W'_G·X̂)   ⊙ (W'_U·X̂)                  shape (N·d_int,  T)
    T₁ · P = Q     →     T₁ = Q · P†                (Eq. 6)

solves for the ``(N·d_int, d_int)`` mixing matrix that makes the merged
expert reproduce the cluster's freq-weighted output on the calibration
tokens ``X̂``. The final merged down-projection is::

    W_D^merged = W'_D · T₁ = Σ_j b_j · W_D^j · T₁_block_j         (d_hidden, d_int)

where ``T₁_block_j`` is the ``j``-th ``(d_int, d_int)`` row-block of T₁.

PyTorch lstsq convention note
-----------------------------
``torch.linalg.lstsq(A, B).solution`` returns ``X`` solving ``A·X = B``;
i.e. ``X = A† · B``. With ``A = Pᵀ`` (shape ``(T, d_int)``) and
``B = Qᵀ`` (shape ``(T, N·d_int)``) we get
``X = (Pᵀ)† · Qᵀ = (P†)ᵀ · Qᵀ = (Q · P†)ᵀ = T₁ᵀ``. So the lstsq
solution transposed gives the paper's T₁ directly. This module operates
in "rows are tokens" layout throughout (PyTorch standard) and transposes
at the very end of the solve.

Deviations from the paper
-------------------------
**D-mergemoe-cond-fallback**: when ``cond(P) > 1e8`` the lstsq is
ill-posed (rank-deficient calibration); we fall back to the
freq-weighted merged down ``Σ_j b_j · W_D^j`` and log a warning. Not in
the paper — added per ``tasks/SC_STAGE12_COMPREHENSIVE_PLAN.md`` §581
risk-mitigation R1. Threshold ``1e8`` matches that spec.

**D-mergemoe-perm-alignment**: the paper clusters by cosine-similarity
on ``[W_U; W_G]`` and does not specify a per-cluster neuron alignment.
This project uses SC's output-cost assignment and the existing
``_permutation_align_to_centroid`` Hungarian neuron alignment. The
caller passes already-aligned member weight tensors — the paper's
``W_G^j`` is our ``perm_j(W_G^j)``. The math is unchanged.

**D-mergemoe-token-cap**: ``X̂`` is sub-sampled to at most
``cost_output_token_cap`` tokens (default 1024). Paper §3.3 uses
~262k tokens; we match the existing SC output-cost token budget so
the calibration buffer is shared. The closed-form lstsq is well-posed
when ``T ≥ d_int``; for the project's ``d_int = 512`` and ``T = 1024``
this holds with 2× margin.

**D-mergemoe-fp32-solve**: the lstsq solve runs in float32 even if the
model weights are bf16; result is cast back to the model's native dtype
before return. Mirrors the existing fp32 numerical posture of
``_merge_experts_inplace`` — see the ``ref_gate``/``ref_up``
upcasts in ``merging.py`` (the ``.to(torch.float32)`` calls inside
``_merge_experts_inplace``) and the matching per-member upcasts in the
``for w, m in zip(weights, members):`` loop.

**D-mergemoe-resume-fallback**: when Stage 2 is being resumed from
``_stage2_partial/``, the per-layer ``_LayerInputAccumulator`` calibration
buffer is not on disk. The orchestrator therefore forces
``merge_step="freq_weighted"`` for every replayed layer before calling
``_merge_experts_inplace`` — see the resume loop in
``stage2/orchestrator.py`` around the ``for record in resumed_records:``
loop. A single ``log.warning`` is emitted once per resumed run (gated on
the configured ``merge_step == "mergemoe"`` and a non-empty
``resumed_records``) rather than once per layer. New layers processed
after the resume cursor use the configured ``merge_step``; the
``stage2/config/merge_step`` Trackio key records the configured value,
not the effective per-layer choice. MergeMoE-mode crashes therefore
require ``--no-resume`` to re-run with the original calibration buffer.

**D-mergemoe-saliency-interaction**: ``merge_step="mergemoe"`` with the
project's saliency-mode merge (``ream.frequency_weighted_merge=False`` —
weights derived from saliency scores rather than calibration frequency)
is a project extension beyond paper Theorem 1. The paper's Eq. 4
collapse ``T₂·W'_G = Σ_j b_j·W_G^j`` relies on the ``b_j`` interpretation
as a normalized merge weight, which holds for both frequency and
saliency weights (any positive weights summing to 1). The resulting
``W_D^merged`` is still the closed-form least-squares solution that makes
the merged expert reproduce the cluster's weighted output on the
calibration tokens; only the source of the ``b_j`` changes.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# Conditioning threshold for the lstsq fallback — see D-mergemoe-cond-fallback.
_COND_THRESHOLD = 1e8


def _swiglu_intermediate(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU intermediate activation ``σ(W_gate·x) ⊙ (W_up·x)``.

    Returns the ``(T, d_int)`` tensor that feeds the down-projection.
    Distinct from :func:`stage2.plugins.output_space_cost._swiglu_forward`
    which returns the full down-projected output. MergeMoE needs the
    pre-down intermediate — that is the ``P`` / ``Q`` of paper Eq. 5.

    Shapes:
        ``W_gate``, ``W_up`` : ``(d_int, d_hidden)``   (PyTorch nn.Linear convention)
        ``x``                : ``(T, d_hidden)``
        return               : ``(T, d_int)``
    """
    gate = F.linear(x, W_gate)
    up = F.linear(x, W_up)
    return F.silu(gate) * up


def _mergemoe_compute_merged_down(
    *,
    member_gates: Sequence[torch.Tensor],
    member_ups: Sequence[torch.Tensor],
    member_downs: Sequence[torch.Tensor],
    weights: Sequence[float],
    layer_inputs: torch.Tensor,
    token_cap: int,
    seed: int,
) -> torch.Tensor:
    """Solve T₁ = Q·P† and return the MergeMoE merged ``down_proj`` matrix.

    Parameters
    ----------
    member_gates, member_ups, member_downs
        N-length sequences of permutation-aligned weight tensors in the
        model's native dtype, all on the same device. The caller is
        responsible for applying the centroid-aligned permutation to
        non-centroid members (so all members share the centroid's
        intermediate-neuron ordering). For the centroid itself the
        permutation is the identity.

        Shapes (PyTorch nn.Linear convention):
            gate, up : (d_int, d_hidden)
            down     : (d_hidden, d_int)

    weights
        N-length sequence of normalized merge weights ``b_j``; must sum
        to ~1.0 (caller normalizes per F2-FREQ-WEIGHT-FLOOR, same as the
        legacy freq-weighted path). Used for both the freq-weighted
        gate/up averages (implicit in the math; gate/up are computed by
        the caller and not by this helper) and the W'_D block weighting.

    layer_inputs
        ``(T_full, d_hidden)`` calibration tokens captured by
        :class:`stage2.profiling._LayerInputAccumulator`. Will be
        sub-sampled to ``token_cap`` tokens via a deterministic
        per-layer permutation (``seed`` parameter).

    token_cap
        Maximum number of calibration tokens to use for the solve.
        Bounds per-layer wall-clock; the lstsq is well-posed when
        ``T ≥ d_int`` so 1024 is comfortable for d_int=512 (the
        project's Qwen3.6-35B-A3B target). See D-mergemoe-token-cap.

    seed
        RNG seed for the deterministic per-layer token sub-sample.
        Callers pass the layer index so the sample is bit-reproducible
        across runs and matches the seed used by
        ``_output_space_cost`` on the same layer.

    Returns
    -------
    merged_down
        ``(d_hidden, d_int)`` tensor in the same dtype/device as
        ``member_downs[0]``. Typical caller (``_merge_experts_inplace``)
        upcasts inputs to fp32 before passing, so the returned tensor is
        fp32; ``ExpertMatrixBank.set`` then casts back to the model's
        native dtype on write. On a conditioning-failure fallback
        (``cond(P) > 1e8``) returns ``Σ_j b_j · W_D^j`` (the legacy
        freq-weighted result) and logs a warning.
    """
    N = len(member_gates)
    if N < 2:
        raise ValueError(
            f"_mergemoe_compute_merged_down: N={N}; need at least 2 cluster "
            "members. Caller must filter singleton groups before calling."
        )
    if not (len(member_ups) == N and len(member_downs) == N and len(weights) == N):
        raise ValueError(
            "_mergemoe_compute_merged_down: member_gates / member_ups / "
            "member_downs / weights must all have the same length "
            f"(got {N} / {len(member_ups)} / {len(member_downs)} / {len(weights)})."
        )

    device = member_downs[0].device
    native_dtype = member_downs[0].dtype
    d_hidden, d_int = member_downs[0].shape

    # Deterministic per-layer token sub-sample. Mirrors the
    # `_output_space_cost` sampler so the two paths consume the same
    # calibration subset on the same layer.
    x_all = layer_inputs.reshape(-1, layer_inputs.shape[-1])
    n_tokens = x_all.shape[0]
    if n_tokens == 0:
        raise ValueError(
            "_mergemoe_compute_merged_down: layer_inputs has zero tokens — "
            "the _LayerInputAccumulator must be enabled when merge_step == "
            "'mergemoe' (check the Stage 2 driver)."
        )
    if n_tokens > token_cap:
        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randperm(n_tokens, generator=rng)[:token_cap]
        X_hat = x_all[idx]
    else:
        X_hat = x_all
    X_hat = X_hat.to(device, dtype=torch.float32)

    # Cast member weights to fp32 for the solve (D-mergemoe-fp32-solve), then
    # move to the compute device. ``member_downs[0].device`` is the canonical
    # target; if the caller passed a mix of devices that is a bug — surface
    # via the existing PyTorch device-mismatch errors on the first .to() that
    # cannot promote silently.
    W_G = [W.to(device=device, dtype=torch.float32) for W in member_gates]
    W_U = [W.to(device=device, dtype=torch.float32) for W in member_ups]
    W_D = [W.to(device=device, dtype=torch.float32) for W in member_downs]
    b = [float(w) for w in weights]

    # Freq-weighted merged gate/up — same math T₂·W'_G yields and same as
    # the legacy path. We only need these to build P (the merged expert's
    # intermediate activation on X̂). Python-scalar weights so the
    # accumulator dtype stays fp32 (no autograd graph, no leaf promotion).
    W_G_merged = W_G[0] * b[0]
    W_U_merged = W_U[0] * b[0]
    for j in range(1, N):
        W_G_merged = W_G_merged + W_G[j] * b[j]
        W_U_merged = W_U_merged + W_U[j] * b[j]

    # P = σ(W_G_merged·X̂) ⊙ (W_U_merged·X̂)         shape (T, d_int)
    P = _swiglu_intermediate(W_G_merged, W_U_merged, X_hat)

    # Conditioning guard (D-mergemoe-cond-fallback). cond is well-defined on
    # the rectangular P (T ≥ d_int expected); for T < d_int we still compute
    # it (cond falls back to the largest/smallest singular ratio).
    try:
        cond_P = float(torch.linalg.cond(P).item())
    except torch.linalg.LinAlgError as exc:  # pragma: no cover — defensive; cond() on
        # a finite real matrix should not raise, but rank-deficient inputs can
        # trigger a LinAlgError inside the SVD. Fail safe to freq-weighted.
        # Narrowed from bare ``Exception`` per reviewer feedback so unrelated
        # bugs (CUDA OOM, device errors) propagate instead of being swallowed.
        log.warning(
            "_mergemoe_compute_merged_down: torch.linalg.cond(P) raised "
            "(%s) — falling back to freq-weighted merged down.",
            exc,
        )
        return _freq_weighted_down(W_D, b).to(native_dtype)

    if not (cond_P < _COND_THRESHOLD):  # NaN-safe: NaN fails the < test
        log.warning(
            "_mergemoe_compute_merged_down: cond(P)=%.3e exceeds threshold "
            "%.0e (D-mergemoe-cond-fallback) — falling back to freq-weighted "
            "merged down for this cluster.",
            cond_P, _COND_THRESHOLD,
        )
        return _freq_weighted_down(W_D, b).to(native_dtype)

    # Build Q row-blocks (one per cluster member) and stack into the
    # T × (N·d_int) "rows are tokens" layout. We never materialize the
    # paper's transposed Q = (N·d_int, T) variant — the lstsq below is
    # in the same "tokens are rows" layout.
    Q_cols = [_swiglu_intermediate(W_G[j], W_U[j], X_hat) for j in range(N)]  # each (T, d_int)
    Q_stack_T = torch.cat(Q_cols, dim=1)  # (T, N·d_int)  — Qᵀ in paper notation

    # Solve for T₁ via PyTorch's lstsq. Let A = P (our layout, shape (T, d_int))
    # and B = Q_stack_T (shape (T, N·d_int)). Then ``lstsq(A, B).solution`` is
    #     X = A† · B = P† · Q_stack_T,   shape (d_int, N·d_int).
    # Paper's P has shape (d_int, T) — i.e. A = paper_Pᵀ — and paper's Q is
    # (N·d_int, T) — i.e. B = paper_Qᵀ. Using the identity (Mᵀ)† = (M†)ᵀ:
    #     paper_T₁ = paper_Q · paper_P†
    #              = (Bᵀ) · (Aᵀ)†
    #              = (Bᵀ) · (A†)ᵀ
    #              = (A† · B)ᵀ
    #              = Xᵀ,                shape (N·d_int, d_int).
    # So T1 = X.transpose(0, 1) is exactly paper Eq. 6's T₁.
    #
    # driver="gelsd" — SVD-based, robust to rank-deficient P. We already
    # short-circuit on cond > 1e8 above, but gelsd is the safer driver for
    # the borderline-conditioned case below the threshold.
    sol = torch.linalg.lstsq(P, Q_stack_T, driver="gelsd")
    X = sol.solution                       # (d_int, N·d_int)
    T1 = X.transpose(0, 1).contiguous()    # (N·d_int, d_int)

    # Split T1 row-wise into N blocks of shape (d_int, d_int) and compute
    # W_D_merged = Σ_j b_j · W_D^j · T1_block_j.
    T1_blocks = T1.view(N, d_int, d_int)
    merged_down = torch.zeros(d_hidden, d_int, dtype=torch.float32, device=device)
    for j in range(N):
        merged_down = merged_down + b[j] * (W_D[j] @ T1_blocks[j])

    return merged_down.to(native_dtype)


def _freq_weighted_down(
    member_downs_fp32: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    """Σ_j b_j · W_D^j — the legacy freq-weighted merged down-projection.

    Used only on the conditioning fallback path
    (``D-mergemoe-cond-fallback``). Operates in the input fp32 dtype;
    caller casts to native dtype before return.
    """
    acc = member_downs_fp32[0] * weights[0]
    for j in range(1, len(weights)):
        acc = acc + member_downs_fp32[j] * weights[j]
    return acc
