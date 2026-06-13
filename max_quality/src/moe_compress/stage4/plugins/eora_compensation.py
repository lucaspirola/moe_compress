"""EoRA per-layer residual compensation kernel (Algorithm 1).

Paper
-----
"EoRA: Eigenspace Low-Rank Approximation for Post-Training Compression
of LLMs" — arXiv:2410.21271 (NVlabs).
audit/spec_compliance/01_papers/2410.21271/source.md.

Algorithm 1 (the EoRA kernel): given a factorization residual
``ΔW = W_orig − Ŵ`` (where ``Ŵ = U · V`` is the post-Stage-3 factored
expert), compute the activation-weighted eigenspace correction by:

  1. Eigendecompose the input auto-covariance ``A = Q Λ Q^T``.
  2. Project the residual onto the eigenbasis: ``ΔW' = ΔW · Q · √Λ``.
  3. Rank-r SVD of ``ΔW'`` (where r is the EoRA budget for this matrix).
  4. Back-project the rank-r left+right factors through ``Q · √Λ⁻¹``.
  5. **Widen** the FactoredExperts ``U / V`` slots by appending the
     correction columns.

Theorem 1 guarantees exactness if Q is full (``QQ^T = I``). See
deviation D10 below for the noise-floor truncation that takes the
implementation slightly off the exact Theorem-1 path.

Official code
-------------
``NVlabs/EoRA`` @ commit
``6a42e2edcc7559422d14ccf79b0105b2d8a78c76`` (2026-04-21) —
github.com/NVlabs/EoRA. Reference implementation for the √Λ-eigenspace
projection + correction.

Deviation: D10 — Eigenspace noise-floor truncation
--------------------------------------------------
Paper Algorithm 1 uses the full ``Q ∈ ℝ^{k×k}``; ``QQ^T = I``
guarantees Theorem 1 exactness.

This plugin's reading: ``X̃ ∈ ℝ^{N×d_in}`` is the matrix of per-token
activation samples for tokens routed to the expert, so
``A = X̃^T X̃ ∈ ℝ^{d_in × d_in}`` has rank ≤ ``min(N, d_in)``
(typically ≫ 1 under the project's calibration volume). The
noise-floor threshold keeps only eigenvalues above a dtype-aware
floor; small-eigenvalue directions below the floor are discarded.

This is a **real, deliberate deviation** from Theorem 1 exactness —
not exact. When A has rank > 1, noise-floor truncation discards
small-eigenvalue directions (noise-dominated activation modes). The
discarded **activation-weighted reconstruction-error component** is
upper-bounded by ``‖ΔW‖_2² · Σ_{j > n_keep} λ_j`` (where
``ΔW = W_orig − Ŵ``). Loewner-trace majorant: the discarded
contribution to ``tr(ΔW · A_tail · ΔW^T)`` is bounded by
``‖ΔW‖_2² · Σ_{j > n_keep} λ_j`` where
``A_tail = Σ_{j > n_keep} λ_j · q_j q_j^T``. For the eigendirections
kept, Theorem-1-style exactness holds in the kept subspace.

This residual is dominated by Stage 3 SVD residual / quantization
residual at moderate-to-high compression ratios. The trade-off is
intentional — preserving every tiny eigendirection would waste rank
budget on noise; the rank cap (``eigenspace_rank_cap = 128``) further
bounds ``take_eff`` so the correction concentrates on the
highest-energy directions.

Deviation: D-eora-budget-pct — 3 % of Stage 3 per-matrix savings
----------------------------------------------------------------
Paper sweeps fixed correction ranks ``{64, 128, 256, 512}`` per
matrix in its experiments; no "% of savings" rule.

This plugin uses ``compensation_budget_pct = 3 %`` of Stage 3
per-matrix parameter savings, then caps at
``eigenspace_rank_cap = 128`` rank. Project-chosen, **not from
paper**. 3 % empirically selected to keep Stage 4's added parameter
footprint small relative to Stage 3 savings (net compression remains
favorable while still recovering quality on the most-truncated
matrices). The cap at 128 keeps per-matrix EoRA rank within the
paper's evaluated range ``{64, 128, 256, 512}``.

(*TODO: ablate 1 % / 3 % / 5 % once Stage 6 evals are available.*)

Activation-cov reuse
--------------------
This plugin reads the **post-merge** A-covariance from Stage 2's
sidecar. The "pre-merge A on post-merge weights" mismatch documented
under D-drank-premerge-A (consumed at
:mod:`stage3.plugins.d_rank_allocate`) is intentionally absorbed
here: the EoRA √Λ projection is computed on the **post-merge** A
re-collected for Stage 4, so the activation-aware projection sees
the true post-merge distribution.

Naming-history note
-------------------
"S4-3" / "Phase D" (legacy Stage 4 monolith terminology) are
naming-historical. The current plugin architecture has no phase
taxonomy; new prose drops the labels. Existing log lines / Trackio
keys preserved for dashboard back-compat.

Original module header retained:

S4-3 is a MIXED relocation — it has three parts:

(1) Two STANDALONE functions are relocated VERBATIM from the
    ``stage4_eora.py`` monolith (the S3-2/S3-3 pattern):

    * ``_spill_layer``        — atomic per-layer crash-resume spill;
    * ``_compute_eora_factors`` — the paper-correct EoRA residual kernel
      (2410.21271, Algorithm 1).

    The monolith re-imports both via a ``# noqa: F401`` block so ``run()`` and
    external callers/tests keep their ``stage4_eora`` import paths.

(2) The per-matrix budget calc + per-expert widen loop in the monolith
    ``run()`` body (lines ~152-250) is NOT a standalone function — it is
    inline ``run()`` code. It is therefore REPRODUCED in the inert
    ``compensate_layer`` hook below rather than relocated; the monolith
    ``run()`` is left BYTE-IDENTICAL for those statements (the S4-2 pattern).
    S4-4 deletes the monolith ``run()`` and wires this hook live; the
    duplication resolves at that point.

(3) The dtype noise-floor table ``_NOISE_FLOOR_BY_DTYPE`` was relocated by
    S4-3 to ``tools/dtype_noise_floor`` (a pure literal shared by stage 3 and
    stage 4); see the deviation note below.

THE ONE DEVIATION FROM VERBATIM. The monolith's ``_compute_eora_factors``
resolves the dtype noise floor through a function-scope
``from moe_compress.stage3_svd import _NOISE_FLOOR_BY_DTYPE`` (monolith
~line 369) — a stage4→stage3 cross-import. In this relocated copy that
function-scope import is DELETED; the name is supplied instead by the
module-top ``from ...tools.dtype_noise_floor import _NOISE_FLOOR_BY_DTYPE``.
This is behavior-identical — the same dict object, the same lookup — and is
gated by the S4-0 golden snapshot. No other change to the relocated bodies.

Circular-import note (mirror of ``stage4/plugins/eora_inputs``): this module
imports only from ``...utils.*``, ``...pipeline.*``, ``...tools.*`` and
stdlib/torch — NEVER from ``stage4_eora`` or ``stage4.orchestrator`` at any
scope (module-top OR function-local). The monolith ``stage4_eora`` imports
*this* module at load time (the S4-3 ``# noqa: F401`` re-import block), so a
``from ..stage4_eora import ...`` here — at any scope — would deadlock the
import cycle; nothing here does that.

``EoraCompensationPlugin`` is registered-but-INERT at S4-3 — no orchestrator
walk or test invokes its ``compensate_layer`` hook. S4-4 plugs the hook into
the live Stage 4 plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from ...pipeline.context import PipelineContext
from ...tools.dtype_noise_floor import _NOISE_FLOOR_BY_DTYPE
from ...utils.model_io import MATRIX_NAMES, FactoredExperts
from ...utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)


def _spill_layer(
    partial_dir: Path,
    layer_idx: int,
    fe: FactoredExperts,
    rank_map_layer: dict[str, int],
    compensated_params_layer: int,
) -> None:
    payload = {
        "format_version": 1,
        "layer_idx": layer_idx,
        "ranks": dict(fe.ranks),
        "effective_ranks": {n: list(v) for n, v in fe.effective_ranks.items()},
        "rank_map_layer": rank_map_layer,
        "compensated_params_layer": compensated_params_layer,
    }
    for name in MATRIX_NAMES:
        payload[f"{name}_U"] = getattr(fe, f"{name}_U").data.cpu()
        payload[f"{name}_V"] = getattr(fe, f"{name}_V").data.cpu()
    tmp = partial_dir / f"layer_{layer_idx}.pt.tmp"
    final = partial_dir / f"layer_{layer_idx}.pt"
    torch.save(payload, tmp)
    os.replace(tmp, final)


def _eigh_spectrum(
    A: torch.Tensor,
    d_in: int,
    device,
    storage_dtype: torch.dtype | None = None,
):
    """Compute the EoRA whitening spectrum of covariance ``A`` (Lever A helper).

    Performs the EXACT prologue ``_compute_eora_factors`` runs at the original
    lines :239-:240 BEFORE ``eigh`` (cast to fp32 + symmetrize), then the
    noise-floor truncation. The returned spectrum is keyed on the
    *post-cast, post-symmetrize* matrix — memoizing it and reusing it on the
    up_proj pass (which shares gate_proj's identical ``A`` object) is
    bit-identical to recomputing ``eigh`` on the same input.

    Returns
    -------
    ``None``
        when the caller must fall back to plain (unweighted) SVD — i.e. the
        covariance shape mismatches ``(d_in, d_in)`` or no eigenvalue clears
        the noise floor. Mirrors the original :241-:244 / :255-:256 fallbacks.
    ``(eigvecs_keep, eigvals_keep, sqrt_lambda, inv_sqrt_lambda)``
        on success. ``eigvecs_keep`` is ``[d_in, n_keep]``; the three vectors
        are ``[n_keep]``. All on ``device`` in fp32.
    """
    A = A.to(device=device, dtype=torch.float32)
    A = 0.5 * (A + A.T)
    if A.shape != (d_in, d_in):
        log.warning("EoRA: covariance shape %s != (%d,%d), falling back to plain SVD",
                    A.shape, d_in, d_in)
        return None

    # Step 2: Eigendecompose activation covariance A = X̃^T X̃ = QΛQ^T  (A shape [d_in × d_in])
    eigvals, eigvecs = torch.linalg.eigh(A)  # ascending order

    lambda_max = float(eigvals[-1].clamp_min(0).item())
    # Dtype-aware noise floor — see _NOISE_FLOOR_BY_DTYPE in tools.dtype_noise_floor.
    rel_floor = _NOISE_FLOOR_BY_DTYPE.get(storage_dtype or torch.float32, 1e-6)
    thresh = max(lambda_max * rel_floor, 1e-12)

    keep_mask = eigvals > thresh
    if not keep_mask.any():
        return None

    # Keep only directions above the noise floor for the projection.
    eigvals_keep = eigvals[keep_mask].clamp_min(0)
    eigvecs_keep = eigvecs[:, keep_mask]         # [d_in, n_keep]

    sqrt_lambda = eigvals_keep.sqrt()                        # [n_keep]
    inv_sqrt_lambda = eigvals_keep.clamp_min(1e-30).rsqrt()  # [n_keep]
    return eigvecs_keep, eigvals_keep, sqrt_lambda, inv_sqrt_lambda


def _compute_eora_factors(
    delta: torch.Tensor,
    A: torch.Tensor | None,
    r: int,
    device,
    *,
    storage_dtype: torch.dtype | None = None,
    spectrum: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Paper-correct EoRA (2410.21271) Algorithm 1.

    Steps (matching paper notation, restricted to signal eigenspace):
      1. ΔW = W_orig − Ŵ                           [d_out × d_in]  (caller)
      2. Eigendecompose A = X̃^T X̃ = QΛQ^T          [d_in × d_in]
         Keep n_keep eigenvectors above noise floor.
      3. Q' = Q_keep · √Λ_keep                      [d_in × n_keep]
      4. ΔW' = ΔW · Q'                              [d_out × n_keep]  (full projection)
      5. SVD(ΔW') full_matrices=False → top take_eff=min(r, min(d_out, n_keep))
      6. U_corr = U'[:,:take_eff] · Σ'[:take_eff]       [d_out × take_eff]
         V_corr = Vh'[:take_eff] · diag(1/√Λ_keep) @ Q_keep^T   [take_eff × d_in]

    The √Λ scaling is the core innovation of EoRA: it importance-weights the
    eigenspace so SVD concentrates rank budget on high-variance input directions.
    Without it, this degenerates to the Act-S baseline.

    Returns (U, V, take_eff) where `take_eff <= r` is the effective rank.
    When `take_eff < r`, U/V are zero-padded to width r.

    Silent-degrade note: if ``A.shape != (d_in, d_in)`` the kernel logs a
    WARNING and falls back to plain (unweighted) SVD rather than raising.
    The contract is that ``EoraInputsPlugin`` delivers a matching covariance
    (i.e. this fallback indicates an upstream wiring bug, not user input);
    callers depending on activation-aware EoRA semantics should treat the
    warning as a hard error and investigate the input pipeline.
    TODO: consider upgrading the warning to ``raise ValueError`` once the
    eora_inputs contract is locked down further.

    Lever A — ``spectrum``
    ----------------------
    Optional precomputed whitening spectrum
    ``(eigvecs_keep, eigvals_keep, sqrt_lambda, inv_sqrt_lambda)`` from
    :func:`_eigh_spectrum`. When supplied, the kernel SKIPS its own
    ``eigh(A)`` and reuses these tensors. The gate_proj / up_proj passes of
    the same ``(layer, expert)`` share the IDENTICAL ``A`` object, so passing
    gate's spectrum to up is bit-identical to recomputing — by construction,
    since the memoized spectrum is the spectrum of the *post-cast,
    post-symmetrize* matrix that ``eigh`` would consume. When ``spectrum`` is
    ``None`` the kernel computes it itself (unchanged behaviour).
    """
    if r <= 0:
        return (torch.zeros(delta.shape[0], 0, device=device),
                torch.zeros(0, delta.shape[1], device=device), 0)
    delta = delta.to(device=device, dtype=torch.float32)
    d_out, d_in = delta.shape

    def _plain_svd_padded() -> tuple[torch.Tensor, torch.Tensor, int]:
        # Fallback: plain SVD without activation weighting.
        # Zero-pad to fixed r so caller's pre-allocated tensors stay shape-stable.
        # (When rk == r the pad slices are empty — single return path handles both.)
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        rk = min(r, U.shape[1])
        U_out = torch.zeros(d_out, r, device=device, dtype=delta.dtype)
        V_out = torch.zeros(r, d_in, device=device, dtype=delta.dtype)
        U_out[:, :rk] = U[:, :rk] * S[:rk]
        V_out[:rk, :] = Vh[:rk, :]
        return U_out, V_out, rk

    if A is None and spectrum is None:
        return _plain_svd_padded()

    # Lever A: reuse a precomputed spectrum when supplied; else compute it now.
    # ``_eigh_spectrum`` returns None on the same shape-mismatch / empty-keep
    # conditions that previously fell back inline (:241-:244, :255-:256).
    if spectrum is None:
        spectrum = _eigh_spectrum(A, d_in, device, storage_dtype)
    if spectrum is None:
        return _plain_svd_padded()
    eigvecs_keep, eigvals_keep, sqrt_lambda, inv_sqrt_lambda = spectrum
    # A SUPPLIED spectrum (Lever A reuse) may have been computed on a DIFFERENT
    # device than ``delta`` — on real multi-GPU the gate_proj pass can band
    # expert ``e`` to a different worker device than the up_proj pass (the two
    # passes resolve their bands independently from per-matrix eligibility sets,
    # which differ under partial/resumed Stage-3 originals). Relocate the few
    # small fp32 spectrum tensors to ``delta``'s device so step 4's
    # ``delta @ Q_prime`` is never a cross-device matmul. Bit-identical when the
    # spectrum already lives on ``device`` (``.to`` is a no-op).
    if eigvecs_keep.device != delta.device:
        eigvecs_keep = eigvecs_keep.to(device=device)
        eigvals_keep = eigvals_keep.to(device=device)
        sqrt_lambda = sqrt_lambda.to(device=device)
        inv_sqrt_lambda = inv_sqrt_lambda.to(device=device)

    # Step 3: Q' = Q · √Λ  (paper Algorithm 1 step 3).
    # FULL projection matrix — NOT truncated to r. The SVD in step 5
    # will optimally select the best r directions from the FULL d_in-
    # dimensional projected error. Pre-truncating to r would eliminate
    # the joint optimisation that distinguishes EoRA from Act-S.
    Q_prime = eigvecs_keep * sqrt_lambda.unsqueeze(0)         # [d_in, n_keep]

    # Step 4: ΔW' = ΔW · Q'  (FULL projection, [d_out × n_keep])
    delta_prime = delta @ Q_prime                             # [d_out, n_keep]

    # Step 5: rank-r SVD of the full projected error — Lever C (Gram-side).
    # Instead of torch.linalg.svd(delta_prime) (which materialises ALL
    # min(d_out, n_keep) singular triplets), eigh the SMALLER Gram and extract
    # the top-take_eff triplets. ``take_eff`` MIRRORS production EXACTLY:
    #   production was  take_eff = min(r, U_p.shape[1]) with U_p from
    #   svd(delta_prime, full_matrices=False) ⇒ U_p.shape[1] == min(d_out, n_keep)
    # so take_eff = min(r, min(d_out, n_keep)). We do NOT add an (evals>0)
    # positivity filter — that would shrink take_eff below production whenever a
    # kept direction is zero/near-zero/Gram-negative, a deterministic rank
    # reduction the full SVD never does → would break the golden by construction.
    # We clamp the singular VALUES (not the count) and guard the left/right
    # divide with an eps tensor on ``device``.
    d_out_, n_keep_ = delta_prime.shape
    take_eff = min(r, min(d_out_, n_keep_))
    eps = torch.tensor(1e-30, device=device, dtype=torch.float32)
    if n_keep_ <= d_out_:
        # Right Gram is the smaller [n_keep, n_keep] matrix.
        G = delta_prime.T @ delta_prime                       # [n_keep, n_keep]
        evals, evecs = torch.linalg.eigh(G)                   # ascending
        idx = torch.arange(n_keep_ - 1, n_keep_ - 1 - take_eff, -1, device=device)
        s = evals[idx].clamp_min(0).sqrt()                    # σ = √λ, [take_eff]
        Vh_p = evecs[:, idx].T                                # [take_eff, n_keep]
        U_p = (delta_prime @ evecs[:, idx]) / s.clamp_min(eps)  # [d_out, take_eff]
    else:
        # Left Gram is the smaller [d_out, d_out] matrix.
        G = delta_prime @ delta_prime.T                       # [d_out, d_out]
        evals, evecs = torch.linalg.eigh(G)                   # ascending
        idx = torch.arange(d_out_ - 1, d_out_ - 1 - take_eff, -1, device=device)
        s = evals[idx].clamp_min(0).sqrt()                    # σ = √λ, [take_eff]
        U_p = evecs[:, idx]                                   # [d_out, take_eff]
        Vh_p = ((delta_prime.T @ evecs[:, idx]) / s.clamp_min(eps)).T  # [take_eff, n_keep]

    # Step 6a: B' = U' Σ'
    U_corr = U_p * s                                         # [d_out, take_eff]

    # Step 6b: Back-project V_corr = V'^T · Q'^{+}
    # Q'^{+} = diag(1/√Λ) · Q^T  (pseudo-inverse — eigvecs_keep is [d_in × n_keep],
    # non-square after the noise-floor mask, so this is Moore-Penrose, not true inverse)
    # So: V_corr = V'^T · diag(1/√Λ) · Q^T = (Vh_p[:take_eff] · diag(1/√Λ)) @ Q^T
    V_corr = (Vh_p * inv_sqrt_lambda.unsqueeze(0)) @ eigvecs_keep.T
    # V_corr shape: [take_eff, d_in]

    # Zero-pad to fixed r so caller's pre-allocated tensors stay shape-stable.
    if take_eff >= r:
        return U_corr[:, :r], V_corr[:r, :], r
    U_out = torch.zeros(d_out, r, device=device, dtype=U_corr.dtype)
    V_out = torch.zeros(r, d_in, device=device, dtype=V_corr.dtype)
    U_out[:, :take_eff] = U_corr
    V_out[:take_eff, :] = V_corr
    return U_out, V_out, take_eff


def _solve_expert_tile(
    name: str,
    e: int,
    layer_idx: int,
    W_orig,
    U_e,
    V_e,
    A,
    d_in: int,
    r_per_expert: int,
    target_device,
    a_storage_dtype,
    *,
    gate_spectrum=None,
    log_residuals: bool = False,
):
    """Pure per-expert EoRA solve — the task-parallel unit (N-GPU lever 1).

    Verbatim extraction of the serial per-expert inner block
    (``compensate_layer`` :523-557), parameterized by ``target_device`` in
    place of the closure ``dev``. Reads ONLY this expert's tensors
    (``W_orig`` / ``U_e`` / ``V_e`` / ``A``) — nothing from any other expert —
    so it is a pure function of its inputs and relocating it to any device is a
    pure relocation (same kernels on the same arch ⇒ bit-identical).

    The gate→up whitening-spectrum memo (Lever A) is threaded as an explicit
    in/out arg ``gate_spectrum``, keyed PER-EXPERT: a worker computes the
    spectrum on the gate_proj pass (returns it) and the caller hands it back on
    the up_proj pass (the loop is matrix-outer / expert-inner, so the memo
    survives the whole gate→up window). The memo is NOT pinned to a single
    device: gate_proj and up_proj resolve their worker bands INDEPENDENTLY from
    per-matrix eligibility (which can differ under partial/resumed Stage-3
    originals), so the spectrum may arrive on a different device than this
    pass's ``delta``. ``_compute_eora_factors`` therefore relocates a supplied
    spectrum to ``target_device`` before use (bit-identical no-op when they
    already match), so step 4's projection is never a cross-device matmul.
    down_proj passes ``gate_spectrum=None`` and gets ``None`` back.

    Returns
    -------
    ``(Uc, Vc, take_eff, res_before, res_after, gate_spectrum_out)``
        ``Uc`` / ``Vc`` are the correction factors on ``target_device``;
        ``take_eff`` is the integer effective rank; ``res_before`` / ``res_after``
        are scalar fp32 squared residual norms (log-only, never the golden) when
        ``log_residuals=True``, else both are ``None`` (the extra ``Uc@Vc`` recompute
        is skipped — default);
        ``gate_spectrum_out`` is the memo to retain for this expert's up_proj
        pass (the gate spectrum on gate_proj, else ``None``).
    """
    W_orig_f = W_orig.to(device=target_device, dtype=torch.float32)
    delta = W_orig_f - (
        U_e.to(device=target_device, dtype=torch.float32)
        @ V_e.to(device=target_device, dtype=torch.float32)
    )
    res_before = delta.norm() ** 2 if log_residuals else None
    # Lever A: gate_proj computes+memoizes the spectrum; up_proj reuses gate's
    # (identical A object) → skips the redundant eigh.
    if name == "up_proj":
        spectrum = gate_spectrum
        gate_spectrum_out = None
    elif name == "gate_proj":
        spectrum = (
            _eigh_spectrum(A, d_in, target_device, a_storage_dtype)
            if A is not None else None
        )
        gate_spectrum_out = spectrum
    else:
        spectrum = None
        gate_spectrum_out = None
    Uc, Vc, take_eff = _compute_eora_factors(
        delta, A, r_per_expert, target_device, storage_dtype=a_storage_dtype,
        spectrum=spectrum,
    )
    # Residual after applying the planned correction (Uc @ Vc).
    res_after = (
        (delta - (Uc.to(torch.float32) @ Vc.to(torch.float32))).norm() ** 2
        if log_residuals else None
    )
    return Uc, Vc, int(take_eff), res_before, res_after, gate_spectrum_out


# Module-level read-only DIAGNOSTICS for tests (passive introspection — NOT a
# behavior patch / monkeypatch). Reflect the most recent _run_expert_bands call.
_LAST_BAND_COUNT: int = 0
_LAST_RAN_THREADED: bool = False


def _run_expert_bands(
    bands,                           # [(w, [experts ascending]), ...] — keyed by ORDINAL w, NOT device
    device_of,                       # {e: torch.device}
    solve_one,                       # (e, tgt) -> (Uc_home, Vc_home, take_eff, rb, ra, spec_out)
    *,
    name: str,
    set_gate_spectrum,               # (e, spectrum_out) -> None   (called inside band thread)
    concurrent: bool,
    log_residuals: bool = False,
) -> dict:
    """Run the per-expert solves grouped into per-WORKER bands (banded by ordinal).

    Concurrency engine for N-GPU lever 1. Each band ordinal ``w`` gets ONE thread
    that runs its contiguous band in ASCENDING-e order; threads overlap across
    workers (CUDA kernels release the GIL and run async on each device's default
    stream). The gather to the home device happens INSIDE each thread
    (``solve_one`` returns home-resident tiles), so the only main-thread work is
    the deterministic ascending-e assembly the caller does after join.

    Banded by ORDINAL ``w`` (C1): the caller passes ``bands`` pre-grouped by the
    worker index used in ``device_of`` construction, so two workers that happen
    to map to the SAME device object (e.g. the CPU test stand-in
    ``["cpu","cpu","cpu"]``) remain DISTINCT bands → real threads. Grouping by
    ``device_of[e]`` here would silently collapse them and skip the pool.

    Byte-identicality: ``solve_one`` is a pure function of (e, tgt); results are
    keyed by ``e`` into a dict (disjoint keys, no race) and the CALLER reassembles
    in ascending-e — so worker completion order is unobservable, exactly like
    serial. ``concurrent=False`` OR a single band runs inline on the calling
    thread → byte-identical to the legacy serial loop, and the 1-GPU default
    (effective_workers<=1 ⇒ one band) never enters a thread.

    Returns ``band_results: {e: (Uc_home, Vc_home, take_eff, rb, ra)}``.
    """
    global _LAST_BAND_COUNT, _LAST_RAN_THREADED
    import torch as _torch
    from concurrent.futures import ThreadPoolExecutor

    band_results: dict = {}
    _LAST_BAND_COUNT = len(bands)

    def _run_band(experts):
        # N2: this closure captures ``experts`` from the submit-time loop var,
        # which is SAFE because the helper joins all threads synchronously below
        # within this single call — threads never outlive _run_expert_bands, so
        # no late-binding hazard (each future carries its own ``experts`` arg).
        for e in experts:                    # ascending-e within the band
            tgt = device_of[e]
            Uc_home, Vc_home, take_eff, rb, ra, spec_out = solve_one(e, tgt)
            if name == "gate_proj":
                set_gate_spectrum(e, spec_out)   # disjoint key e — thread-safe under GIL
            band_results[e] = (Uc_home, Vc_home, take_eff, rb, ra)

    if not concurrent or len(bands) <= 1:
        _LAST_RAN_THREADED = False
        for _w, experts in bands:
            _run_band(experts)
        return band_results

    # Threaded: one worker thread per band ordinal. Pin intra-op BLAS to 1 on the
    # main thread (process-global; restored in finally) to avoid (bands x cores)
    # oversubscription on the CPU stand-in path. No-op cost on real GPUs.
    _LAST_RAN_THREADED = True
    prev_threads = _torch.get_num_threads()
    try:
        _torch.set_num_threads(1)
        with ThreadPoolExecutor(max_workers=len(bands)) as pool:
            futures = [pool.submit(_run_band, experts) for _w, experts in bands]
            for f in futures:
                f.result()                   # re-raise any worker exception here
    finally:
        _torch.set_num_threads(prev_threads)

    # 4c — RECOMMENDED guarded residual sync (M1). The residual scalars rb/ra are
    # produced on each WORKER device and later copied to the home device by the
    # main-thread assembly (rb.to(dev)), which can race the worker's in-flight
    # kernel. Log-only ⇒ a race would silently corrupt a logged number without
    # tripping any byte gate. Sync the distinct worker devices once per matrix,
    # ONLY under log_residuals (default off ⇒ zero cost on the golden path).
    if log_residuals:
        seen = set()
        for e in band_results:
            d = device_of[e]
            if d.type == "cuda" and d not in seen:
                seen.add(d)
                _torch.cuda.synchronize(d)
    return band_results


def _resolve_worker_devices(worker_devices, n: int, home_device) -> list:
    """Resolve the list of ``n`` target devices for the expert fan-out.

    If ``worker_devices`` is supplied (the test/integration seam — e.g.
    ``["cuda:0","cuda:1"]`` or the CI CPU stand-in ``["cpu","cpu"]``), it is
    used verbatim (truncated/cycled to ``n``). Otherwise derive ascending CUDA
    devices ``cuda:0..cuda:(n-1)`` when CUDA is available, else fall back to
    ``n`` copies of the home device (graceful 1-GPU / CPU degrade).
    """
    if worker_devices:
        devs = [torch.device(d) for d in worker_devices]
        if len(devs) < n:
            devs = [devs[i % len(devs)] for i in range(n)]
        return devs[:n]
    if torch.cuda.is_available() and torch.cuda.device_count() >= n:
        return [torch.device(f"cuda:{i}") for i in range(n)]
    return [home_device for _ in range(n)]


class _AnchoredAdaptiveLookup:
    """anchored_adaptive whitening: blend anchor + shift into a single [d,d]
    cov for the EoRA √Λ basis. Reduction (NOT the full AA-SVD M=W·C·S⁻¹·R —
    that needs cross-cov C and a different kernel; see plan §9 open question).
    The blend is A_eff = A_anchor (when shift missing) else
    0.5*(A_anchor+A_shift), keeping EoRA's single-cov contract."""

    def __init__(self, A_cov, shift_cov):
        self._a, self._s = A_cov, shift_cov or {}

    def get(self, key):
        a = self._a.get(key)
        s = self._s.get(key)
        if a is None:
            return s
        if s is None:
            return a
        return 0.5 * (a.to(torch.float32) + s.to(torch.float32))


def _resolve_whitening_lookup(whitening_cov: str, A_cov, shift_cov):
    """Select the EoRA whitening covariance dict per stage4_eora.whitening_cov.

    'anchor' (default) -> the original-calibration anchor A_cov (byte-identical
    to historical behaviour; the SAME object). 'shift' -> the post-2.5 SHIFT
    cov (upstream-EoRA-faithful; spec 2026-06-13-acov-capture-point.md).
    'anchored_adaptive' -> _AnchoredAdaptiveLookup(A_cov, shift_cov) (plan §7).
    Unknown value raises.
    """
    if whitening_cov == "anchor":
        return A_cov
    if whitening_cov == "shift":
        if not shift_cov:
            raise ValueError(
                "stage4_eora.whitening_cov='shift' but no shift_cov was loaded "
                "(EoraInputsPlugin did not populate it — wiring bug)."
            )
        return shift_cov
    if whitening_cov == "anchored_adaptive":
        return _AnchoredAdaptiveLookup(A_cov, shift_cov)
    raise ValueError(
        f"stage4_eora.whitening_cov must be 'anchor', 'shift', or "
        f"'anchored_adaptive', got {whitening_cov!r}"
    )


class EoraCompensationPlugin:
    """Stage 4 EoRA residual-compensation plugin (S4-3 — registered-but-INERT).

    Owns the EoRA per-layer ``compensate_layer`` phase: for each matrix type,
    the per-matrix compensation-budget calculation (capped at
    ``compensation_budget_pct`` of the Stage-3 saved params), the per-expert
    ``_compute_eora_factors`` residual kernel loop, the in-process
    double-widen ``assert``, the ``FactoredExperts.widen_rank`` call, the
    trackio emit, and the per-layer crash-resume spill (``_spill_layer``).

    The residual kernel ``_compute_eora_factors`` and the spill helper
    ``_spill_layer`` are relocated VERBATIM from the ``stage4_eora.py``
    monolith (one deviation — see the module docstring); the monolith
    re-imports them. The per-matrix budget + widen loop is NOT a standalone
    function in the monolith — it is inline ``run()`` code — so the
    ``compensate_layer`` hook below REPRODUCES it; the monolith ``run()`` is
    left byte-identical for those statements.

    S4-3 wires this class into the plugin registry as metadata only — no walk
    or test invokes ``compensate_layer``. S4-4 plugs the hook into the live
    Stage 4 plugin sequencer and deletes the monolith ``run()``.
    """

    name = "eora_compensation"
    paper = (
        "EoRA Algorithm 1 residual compensation — arXiv:2410.21271 "
        "(NVlabs/EoRA @ 6a42e2edcc7559422d14ccf79b0105b2d8a78c76). "
        "Deviations: D10 (eigenspace noise-floor truncation — bounded "
        "by ‖ΔW‖_2²·Σ_{j>n_keep} λ_j), D-eora-budget-pct (3% of Stage 3 "
        "per-matrix savings, cap rank=128). See module docstring."
    )
    config_key = "stage4_eora.compensation_budget_pct"
    # ``compensate_layer`` runs inside a per-layer scope: it reads the layer
    # ref under ``layer_ref`` (the loop item key) and the remaining run-scope
    # slots through the parent ctx chain.
    reads: tuple[str, ...] = (
        "layer_ref", "originals", "A_cov", "a_storage_dtype", "config",
        "partial_dir", "stage3_ranks", "rank_map", "compensated_params",
        # N-GPU lever 1: task-parallel EoRA worker count (+ optional explicit
        # device list seam). Both optional — absent ⇒ serial single-GPU path.
        "eora_workers", "eora_worker_devices",
        # Optional: the post-2.5 shift cov for whitening_cov in {shift,
        # anchored_adaptive}. Absent ⇒ default "anchor" path (A_cov).
        "shift_cov",
    )
    # ``rank_map`` is a shared mutable dict the hook MUTATES in place across
    # loop iterations (mirror of ``aa_svd_factor.factor_layer``'s HAZARD H1)
    # rather than rebinding via ``ctx.set``; ``compensated_params`` is the
    # running total the hook advances. Both remain this plugin's declared
    # write surface.
    writes: tuple[str, ...] = ("rank_map", "compensated_params")
    # Empty: EoraCompensationPlugin needs no calibration pass — the residual
    # compensation consumes only precomputed inputs (Stage-3 originals, the
    # A-covariance) already loaded by EoraInputsPlugin.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — EoRA compensation is UNCONDITIONAL.

        Every Stage 4 run applies residual compensation. The per-matrix
        budget calc may compute ``r_per_expert <= 0`` and ``continue`` past an
        individual matrix internally, but the plugin as a whole always runs;
        ``config_key`` only parametrises the per-matrix compensation budget,
        it never gates the plugin.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def compensate_layer(self, ctx: PipelineContext) -> None:
        """Phase hook — EoRA per-layer residual compensation (S4-4 wiring surface).

        INERT at S4-3: no orchestrator walk or test invokes this hook. S4-4
        replaces the Stage 4 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith ``run()``'s inline
        per-layer ``for name in MATRIX_NAMES:`` block (``stage4_eora.py``
        lines ~152-250). The body below reproduces that block faithfully — it
        is dead code at S4-3 but S4-4 relies on it once the monolith is
        deleted.

        Reproduces (in monolith order): the per-matrix compensation-budget
        calc, the per-expert ``_compute_eora_factors`` loop, the in-process
        double-widen ``assert`` (INCLUDED — S4-4 relies on it), the
        ``fe.widen_rank`` call, the trackio emit, and the per-layer tail
        (``rank_map.update`` / ``compensated_params +=`` / ``_spill_layer``).

        The layer ref arrives under ``ctx["layer_ref"]`` (the loop item key);
        ``originals`` / ``A_cov`` / ``a_storage_dtype`` / ``config`` /
        ``stage3_ranks`` / ``rank_map`` / ``compensated_params`` resolve
        through the parent ctx chain. ``partial_dir`` is ``has()``-guarded —
        it is ``None`` when ``no_resume=True``.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. ``partial_dir`` is optional (has()-guarded).
        ref = ctx.get("layer_ref")
        originals = ctx.get("originals")
        A_cov = ctx.get("A_cov")
        a_storage_dtype = ctx.get("a_storage_dtype")
        config = ctx.get("config")
        stage3_ranks = ctx.get("stage3_ranks")
        rank_map = ctx.get("rank_map")
        compensated_params = ctx.get("compensated_params")
        partial_dir = ctx.get("partial_dir") if ctx.has("partial_dir") else None

        s4 = config["stage4_eora"]
        log_residuals = bool(s4.get("log_residuals", False))
        # Whitening-cov selection (default "anchor" -> whitening_lookup IS
        # A_cov -> _inputs_for byte-identical to history).
        whitening_cov = str(s4.get("whitening_cov", "anchor"))
        shift_cov = ctx.get("shift_cov") if ctx.has("shift_cov") else None
        whitening_lookup = _resolve_whitening_lookup(
            whitening_cov, A_cov, shift_cov
        )
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            return
        dev = fe.gate_proj_U.device
        dtype = fe.gate_proj_U.dtype
        N = fe.num_experts

        # N-GPU lever 1 (task-parallel EoRA): the orchestrator resolves the
        # effective worker count and (optionally) an explicit worker-device
        # list onto the ctx. ``eora_workers <= 1`` (or absent) ⇒ the serial
        # in-process path below, byte-identical to single-GPU today.
        eora_workers = int(ctx.get("eora_workers")) if ctx.has("eora_workers") else 1
        # ``eora_worker_devices`` is an optional test/integration seam: an
        # explicit list of target devices to fan experts across (e.g.
        # ["cuda:0", "cuda:1"], or ["cpu", "cpu"] for the CI CPU stand-in).
        # When absent and workers>1, derive ascending CUDA devices.
        worker_devices = ctx.get("eora_worker_devices") if ctx.has("eora_worker_devices") else None

        layer_compensated_params = 0
        rank_map_layer: dict[str, int] = {}
        # Lever A: memoize the gate_proj whitening spectrum per expert so the
        # up_proj pass — which shares the IDENTICAL covariance object (cov_key
        # rewrite below) — reuses it instead of recomputing eigh(A). Reusing
        # the same spectrum on the same input is bit-identical to recomputing.
        # Variant A1 (matrix-outer loops preserved): keyed by expert index,
        # built during the gate_proj pass and read during up_proj. The dict
        # holds the N gate spectra across the gate→up window (kept eigvecs are
        # [d_in, n_keep] fp32) and is cleared right after the up_proj pass so it
        # never outlives its use or reaches down_proj.
        gate_spectra: dict[int, Any] = {}

        for name in MATRIX_NAMES:
            # Per-matrix-type, per-layer: pool per-expert residuals independently.
            # Budget: ≤ compensation_budget_pct of saved params for this matrix.
            d_out, d_in = fe.matrix_shape(name)
            cur_rank = fe.ranks[name]
            saved_per_expert = d_out * d_in - cur_rank * (d_out + d_in)
            saved_for_matrix = max(0, saved_per_expert) * N
            param_budget = int(s4["compensation_budget_pct"] * saved_for_matrix)
            r_per_expert = param_budget // max(N * (d_out + d_in), 1)
            r_per_expert = min(r_per_expert, s4["eigenspace_rank_cap"], min(d_out, d_in))
            r_per_expert = max(0, r_per_expert)
            if r_per_expert <= 0:
                continue

            U_corr = torch.zeros(N, d_out, r_per_expert, dtype=dtype, device=dev)
            V_corr = torch.zeros(N, r_per_expert, d_in, dtype=dtype, device=dev)
            # Per-expert effective rank of the EoRA correction. Defaults to
            # 0 for experts without an `originals` entry (no correction
            # applied → no parameters added in effective terms).
            eff_per_expert: list[int] = [0] * N
            # Lever B: accumulate squared residual norms ON-DEVICE and sync once
            # per matrix (below), instead of a per-expert .item() host sync.
            # These feed ONLY log.info / trackio — never the golden.
            res_before_acc = torch.zeros((), device=dev, dtype=torch.float32)
            res_after_acc = torch.zeros((), device=dev, dtype=torch.float32)
            n_eligible = 0

            # Eligible experts (have an ``originals`` entry) in ascending order.
            # The gather always writes rows in ascending ``e`` (§4 determinism),
            # so the serial and N-GPU paths fill ``U_corr``/``V_corr`` identically.
            eligible = [
                e for e in range(N)
                if originals.get((ref.layer_idx, e, name)) is not None
            ]

            def _inputs_for(e):
                key = (ref.layer_idx, e, name)
                W_orig = originals.get(key)
                U_e = fe.gate_proj_U.data[e] if name == "gate_proj" else \
                       fe.up_proj_U.data[e]   if name == "up_proj"   else \
                       fe.down_proj_U.data[e]
                V_e = fe.gate_proj_V.data[e] if name == "gate_proj" else \
                       fe.up_proj_V.data[e]   if name == "up_proj"   else \
                       fe.down_proj_V.data[e]
                # up_proj shares the gate_proj input covariance (same fused tensor).
                cov_key = (ref.layer_idx, e, "gate_proj") if name == "up_proj" else key
                A = whitening_lookup.get(cov_key)
                return W_orig, U_e, V_e, A

            # Resolve the effective per-layer worker count + the device each
            # eligible expert is solved on. ``effective_workers <= 1`` ⇒ every
            # expert runs on ``dev`` (the home device) in the same serial order
            # as single-GPU today (byte-identical golden path). >1 ⇒ fan experts
            # contiguously across the worker devices (deterministic bands).
            effective_workers = min(eora_workers, max(1, len(eligible)))
            if effective_workers > 1:
                devices = _resolve_worker_devices(worker_devices, effective_workers, dev)
                effective_workers = min(effective_workers, len(devices))
            if effective_workers <= 1:
                device_of = {e: dev for e in eligible}
            else:
                # Contiguous bands by sorted expert index — reproducible
                # run-to-run and identical per-row to the serial fill.
                device_of = {}
                per = (len(eligible) + effective_workers - 1) // effective_workers
                for w in range(effective_workers):
                    for e in eligible[w * per:(w + 1) * per]:
                        device_of[e] = devices[w]

            # Concurrency engine (N-GPU lever 1): run each worker's band
            # CONCURRENTLY (one thread per band ORDINAL w — NOT per device object,
            # so the CPU ["cpu","cpu"] seam stays multi-band), gathering each tile
            # to the home device INSIDE its band thread. Assembly into U_corr/
            # V_corr and the residual fp sum stay on THIS thread in ascending-e
            # order, so the output is byte-identical to the serial path regardless
            # of which device finishes first. effective_workers<=1 ⇒ one band ⇒
            # inline (no thread), byte-identical to single-GPU today.
            #
            # Build bands by ORDINAL w, reusing the same w/per split that produced
            # device_of above. C1: keyed by w, so two workers mapping to the same
            # device object remain distinct bands.
            if effective_workers <= 1:
                bands = [(0, eligible)]
            else:
                bands = [
                    (w, eligible[w * per:(w + 1) * per])
                    for w in range(effective_workers)
                ]
                bands = [(w, ex) for (w, ex) in bands if ex]   # drop empty trailing bands

            def _solve_one(e, tgt):
                W_orig, U_e, V_e, A = _inputs_for(e)
                Uc, Vc, take_eff, rb, ra, spec_out = _solve_expert_tile(
                    name, e, ref.layer_idx, W_orig, U_e, V_e, A, d_in,
                    r_per_expert, tgt, a_storage_dtype,
                    gate_spectrum=gate_spectra.get(e),
                    log_residuals=log_residuals,
                )
                # Gather to home device on the WORKER thread (overlaps next band).
                Uc_home = Uc.to(device=dev, dtype=dtype)
                Vc_home = Vc.to(device=dev, dtype=dtype)
                return Uc_home, Vc_home, int(take_eff), rb, ra, spec_out

            band_results = _run_expert_bands(
                bands, device_of, _solve_one,
                name=name,
                set_gate_spectrum=gate_spectra.__setitem__,
                concurrent=(effective_workers > 1),
                log_residuals=log_residuals,
            )

            # Deterministic ascending-e assembly (the byte-identity guarantee).
            for e in eligible:
                Uc_home, Vc_home, take_eff, rb, ra = band_results[e]
                # Place the disjoint row (ascending-e order, no cross-expert reduction).
                U_corr[e] = Uc_home
                V_corr[e] = Vc_home
                eff_per_expert[e] = take_eff
                # log-only residual norms (B4) — accumulate on dev in ascending-e.
                if log_residuals:
                    res_before_acc += rb.to(dev)
                    res_after_acc += ra.to(dev)
                n_eligible += 1
                if (e + 1) % 32 == 0:
                    log.info("  L%d/%s expert %d/%d", ref.layer_idx, name, e + 1, N)

            # Double-widen guard: assert ranks haven't been modified yet.
            # Protects against in-process re-runs (notebooks, test harnesses)
            # where widen_rank() would double-apply EoRA correction.
            assert fe.ranks[name] == stage3_ranks.get(ref.layer_idx, {}).get(name, fe.ranks[name]), (
                f"Stage 4 double-widen detected: layer={ref.layer_idx}, matrix={name}, "
                f"current_rank={fe.ranks[name]}, "
                f"stage3_rank={stage3_ranks.get(ref.layer_idx, {}).get(name)}. "
                "widen_rank() has already been applied in this process."
            )
            fe.widen_rank(name, U_corr, V_corr, added_effective_per_expert=eff_per_expert)
            rank_map_layer[f"L{ref.layer_idx}_{name}"] = fe.ranks[name]
            layer_compensated_params += int(U_corr.numel() + V_corr.numel())
            # Lever B: single device→host sync per matrix (was per-expert).
            # Log-only — gated behind stage4_eora.log_residuals (default off).
            residual_fields = {}
            if log_residuals:
                res_before_sum = float(res_before_acc.item())
                res_after_sum = float(res_after_acc.item())
                res_before = (res_before_sum / max(n_eligible, 1)) ** 0.5
                res_after = (res_after_sum / max(n_eligible, 1)) ** 0.5
                rel_drop = (res_before - res_after) / max(res_before, 1e-12)
                log.info("  L%d/%s widened by r=%d → new rank=%d; residual %.4e→%.4e (-%.1f%%)",
                         ref.layer_idx, name, r_per_expert, fe.ranks[name],
                         res_before, res_after, 100 * rel_drop)
                residual_fields = {
                    f"stage4/{name}_residual_unweighted_before": res_before,
                    f"stage4/{name}_residual_unweighted_after": res_after,
                    f"stage4/{name}_residual_unweighted_rel_drop": rel_drop,
                }
            else:
                log.info("  L%d/%s widened by r=%d → new rank=%d",
                         ref.layer_idx, name, r_per_expert, fe.ranks[name])
            _eff_list = [v for v in eff_per_expert if v is not None]
            _trackio_log({
                "stage4/layer_idx": ref.layer_idx,
                f"stage4/{name}_added_rank": r_per_expert,
                f"stage4/{name}_new_rank": fe.ranks[name],
                # NOTE: these are plain Frobenius ‖ΔW‖_F (unweighted), NOT the
                # activation-weighted objective tr(ΔW·A·ΔW^T)^{1/2} that EoRA
                # actually optimises (arXiv:2410.21271 Eq. 6, projected
                # residual ‖ΔW·Q'‖_F). Key renamed to surface that distinction
                # on dashboards.
                **residual_fields,
                "stage4/compensated_params": compensated_params + layer_compensated_params,
                # Additive v2 keys: per-layer aggregates of in-scope variables.
                f"stage4/{name}_n_eligible_experts": int(n_eligible),
                f"stage4/{name}_eff_rank_mean": (
                    float(sum(_eff_list) / len(_eff_list)) if _eff_list else 0.0
                ),
                f"stage4/{name}_eff_rank_max": int(max(_eff_list)) if _eff_list else 0,
                f"stage4/{name}_eff_rank_min": int(min(_eff_list)) if _eff_list else 0,
                # Per-matrix contribution (not the per-layer running total —
                # which is already in `stage4/compensated_params`).
                f"stage4/{name}_matrix_compensated_params": int(U_corr.numel() + V_corr.numel()),
            })

            # Lever A: the memoized gate spectra are consumed by the up_proj
            # pass; drop them before down_proj so they don't outlive their use.
            if name == "up_proj":
                gate_spectra.clear()

        rank_map.update(rank_map_layer)
        compensated_params += layer_compensated_params
        # S4-4: dispatched against the ROOT ctx by a plain for-loop (not
        # loop_over); the overwrite=True rebind of the root scalar accumulates
        # across layers.
        ctx.set("compensated_params", compensated_params, overwrite=True)

        # Atomically persist this layer's FactoredExperts state for crash-resume.
        if partial_dir is not None:
            _spill_layer(partial_dir, ref.layer_idx, fe, rank_map_layer, layer_compensated_params)
