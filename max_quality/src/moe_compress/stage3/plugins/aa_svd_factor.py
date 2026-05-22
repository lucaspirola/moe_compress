"""AA-SVD rank-k factorization core (S3-5 of the Stage 3 plugin refactor).

Home of the activation-aware SVD core relocated VERBATIM from the legacy
``stage3_svd.py`` monolith:

* ``_NOISE_FLOOR_BY_DTYPE`` — module-level dict mapping a storage dtype to the
  relative eigenvalue noise floor (mantissa-bits driven);
* ``_EighDecomp`` — ``@dataclass`` caching the eigendecomposition of a
  covariance matrix B plus the pre-multiplied right-hand side;
* ``_precompute_eigh`` — eigendecomposes B and builds the AA-SVD rhs (the
  expensive, W-independent part of ``_aa_svd``);
* ``_aa_svd_precomputed`` — rank-k factorization of W from a pre-computed
  ``_EighDecomp``;
* ``_aa_svd`` — activation-aware rank-k factorization of W (paper 2604.02119,
  Theorem 3.2 / Corollary 3.3);
* ``_cov_lookup`` — bank-aware per-expert covariance dict lookup with an
  ``up_proj`` → ``gate_proj`` fallback.

All six symbols are byte-identical copies of the monolith bodies; the monolith
re-imports them (``# noqa: F401`` block in ``stage3_svd.py``) so ``run()`` and
external callers/tests keep their existing import paths.

Circular-import note (mirror of ``stage3/plugins/swift_svd_alpha.py``): the
AA-SVD core is SELF-CONTAINED — it imports nothing from ``stage3_svd`` or
``stage3.orchestrator``, and (unlike S3-4's ``swift_svd_alpha``) it needs NO
lazy / function-scope imports at all. ``stage3_svd`` imports *this* module at
load time, so a module-top ``from ...stage3_svd import ...`` here would deadlock
the import cycle — but no such import is needed.

Notes on what S3-5 deliberately does NOT touch:

* ``_pad`` — the master plan's S3-5 line mentions ``_pad``, but ``_pad`` was
  already relocated by S3-3 into ``d_rank_allocate.py`` and the AA-SVD core
  never calls it. S3-5 does NOT touch ``_pad``.
* S3-4's ``swift_svd_alpha.py`` lazy-imports ``_cov_lookup`` / ``_precompute_eigh``
  / ``_aa_svd`` / ``_aa_svd_precomputed`` from ``stage3_svd`` at function scope.
  Those still resolve after S3-5 because the monolith re-exports them via the
  S3-5 ``# noqa: F401`` block. S3-5 does NOT repoint S3-4's lazy imports.
* The per-layer factoring LOOP BODY in ``run()`` (the loop that populates
  ``rank_map`` by walking the MoE layers and installing ``FactoredExperts``) is
  NOT relocated by S3-5 — it stays inline in the monolith and is deferred to
  S3-7 (same as S3-4 left the α-cache I/O inline).

``AaSvdFactorPlugin`` is registered-but-INERT at S3-5 — no walk or test invokes
its ``factor_layer`` hook. S3-7 wires it into the live Stage 3 plugin sequencer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from ...pipeline.context import PipelineContext

log = logging.getLogger(__name__)


_NOISE_FLOOR_BY_DTYPE: dict[torch.dtype, float] = {
    # Relative threshold above which an eigenvalue of B is considered signal
    # rather than storage-quantization noise. Driven by the storage dtype's
    # mantissa bits: bf16 has 7 (~2⁻⁷ ≈ 8e-3 noise), fp16 has 10 (~2⁻¹⁰ ≈ 1e-3),
    # fp32 has 23 (~2⁻²³). Set the floor a small margin above noise to ensure
    # we don't keep noise-inflated directions.
    torch.bfloat16: 1e-2,
    torch.float16:  1e-3,
    torch.float32:  1e-6,
    torch.float64:  1e-12,
}


@dataclass
class _EighDecomp:
    """Cached eigendecomposition of a covariance matrix B, plus the
    pre-multiplied right-hand side for the M = W @ rhs formulation.

    This allows gate_proj and up_proj — which share the same B and C
    covariance (``_cov_lookup`` falls back from up_proj to gate_proj) —
    to skip the redundant ``eigh(B)`` call and the ``C @ Q`` or ``A @ Q``
    product.  The only per-matrix work that remains is ``W @ rhs`` and
    the subsequent SVD + back-solve.

    Attributes:
        eigvals_keep: Eigenvalues of B above the noise floor, clamped ≥0.  [r_eff]
        eigvecs_keep: Corresponding eigenvectors.                          [d_in, r_eff]
        inv_sqrt:     1/√(eigvals_keep), for the back-solve.              [r_eff]
        rhs:          The right-hand-side matrix such that M = W @ rhs.
                      Shape [d_in, r_eff].  Content depends on the path:
                      - Path 1 (Theorem 3.2): CQ · diag(1/√λ)
                      - Path 3 (Cor. 3.3):    L_B = Q · diag(√λ)
        rhs_pinv:     Pseudo-inverse of rhs, shape [r_eff, d_in].
                      Used in the back-solve: V_k = Vh[:k] @ rhs_pinv.
                      - Path 3: exact inverse = diag(1/√λ) · Q^T (no extra SVD)
                      - Path 1: torch.linalg.pinv(rhs)
        r_eff:        Number of retained eigenvalues (= rhs.shape[1]).
    """
    eigvals_keep: torch.Tensor
    eigvecs_keep: torch.Tensor
    inv_sqrt: torch.Tensor
    rhs: torch.Tensor
    rhs_pinv: torch.Tensor
    r_eff: int


def _precompute_eigh(
    B: torch.Tensor,
    A: torch.Tensor | None,
    C: torch.Tensor | None,
    *,
    device,
    storage_dtype: torch.dtype | None = None,
) -> _EighDecomp:
    """Eigendecompose B and build the right-hand-side matrix for AA-SVD.

    This is the expensive part of ``_aa_svd`` that depends only on the
    covariance matrices (B, A, C) and NOT on the weight matrix W.  Since
    gate_proj and up_proj share the same B and C (via ``_cov_lookup``
    fallback), callers can call this once per expert and reuse the result
    for both projections — eliminating one ``eigh(2048×2048)`` call per
    expert.

    Raises ``ValueError`` if B has no positive eigenvalues above the noise
    floor (same behaviour as ``_aa_svd``).
    """
    B = B.to(device=device, dtype=torch.float32)
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = torch.linalg.eigh(B)                     # ascending
    sigma_max = float(eigvals[-1].clamp_min(0).item())
    rel_floor = _NOISE_FLOOR_BY_DTYPE.get(storage_dtype or torch.float32, 1e-6)
    thresh = max(sigma_max * rel_floor, 1e-12)
    keep = eigvals > thresh
    r_eff = int(keep.sum().item())
    if r_eff == 0:
        raise ValueError("B has no positive eigenvalues above threshold")
    eigvals_keep = eigvals[keep].clamp_min(0)
    eigvecs_keep = eigvecs[:, keep]
    inv_sqrt = eigvals_keep.clamp_min(1e-30).rsqrt()             # [r_eff]

    # Path selection: paper 2604.02119 has only two sanctioned paths —
    #   • Path 1 (Theorem 3.2)    when cross-covariance C is available
    #   • Path 3 (Corollary 3.3)  fallback using only B
    # An earlier "Path 2" auto-covariance variant (rhs = A · Q · Λ^{-1/2})
    # corrupted the rank-k target — it produces U·V ≈ W·A·B^{-1}·L_B rather
    # than approximating W in the B-weighted norm. Three tests in
    # tests/test_aa_svd_correctness.py pin A as reserved for L-BFGS refinement
    # only, never used in the SVD step. The variable ``A`` is kept in the
    # signature for backward compatibility but is intentionally unused here.
    del A  # explicit: A must not influence the rank-k factorization
    if C is not None:
        # Path 1: Paper-exact Theorem 3.2 — rhs = C @ Q · diag(1/√λ).
        C = C.to(device=device, dtype=torch.float32)
        CQ = C @ eigvecs_keep                                    # [d_in, r_eff]
        rhs = CQ * inv_sqrt.unsqueeze(0)                         # [d_in, r_eff]
        # pinv(C·Q·Λ^{-1/2}) ≠ Λ^{-1/2}·Q^T — must compute pseudo-inverse explicitly.
        rhs_pinv = torch.linalg.pinv(rhs)                        # [r_eff, d_in]
    else:
        # Path 3: Corollary 3.3 — rhs = L_B = Q · diag(√λ).
        rhs = eigvecs_keep * eigvals_keep.sqrt().unsqueeze(0)    # [d_in, r_eff]
        # Exact inverse: (Q·Λ^{1/2})^{-1} = Λ^{-1/2}·Q^T — no extra SVD.
        rhs_pinv = inv_sqrt.unsqueeze(1) * eigvecs_keep.T        # [r_eff, d_in]

    return _EighDecomp(
        eigvals_keep=eigvals_keep,
        eigvecs_keep=eigvecs_keep,
        inv_sqrt=inv_sqrt,
        rhs=rhs,
        rhs_pinv=rhs_pinv,
        r_eff=r_eff,
    )


def _aa_svd_precomputed(
    W: torch.Tensor,
    decomp: _EighDecomp,
    k: int,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Rank-k factorization of W using a pre-computed eigendecomposition.

    Mathematically identical to ``_aa_svd`` — the only difference is that
    the eigendecomposition of B and the rhs product (Path 1: CQ·inv_sqrt,
    or Path 3: L_B) are supplied via ``decomp`` rather than recomputed.

    Returns (U_k, V_k, rel_err, k_eff).
    """
    d_out, d_in = W.shape
    k = max(1, min(k, min(d_out, d_in) - 1))
    try:
        M = W @ decomp.rhs                                      # [d_out, r_eff]

        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        k_eff = max(1, min(k, decomp.r_eff))
        U_eff = U[:, :k_eff] * S[:k_eff]
        # Back-solve: V_k = Vh[:k_eff] @ rhs_pinv where rhs_pinv = pinv(rhs).
        # Path 3: rhs_pinv = Λ^{-1/2}·Q^T (exact, precomputed analytically).
        # Path 1: rhs_pinv = pinv(C·Q·Λ^{-1/2}) (precomputed via torch.linalg.pinv).
        # Using the path-specific rhs_pinv is critical for Path 1 — the naive
        # Λ^{-1/2}·Q^T back-solve ignores the C factor and produces wrong V_k.
        V_eff = Vh[:k_eff, :] @ decomp.rhs_pinv                 # [k_eff, d_in]
        # Numerically stable rel_err: tail singular values of M.
        S2 = S * S
        denom = S2.sum().clamp_min(1e-30)
        if k_eff < S2.numel():
            rel_err = float((S2[k_eff:].sum() / denom).sqrt().item())
        else:
            rel_err = 0.0
        # Always return shape [d_out, k] / [k, d_in] — caller's FactoredExperts
        # slot is pre-allocated at `k`. Zero-pad when effective rank < k.
        if k_eff < k:
            U_k = torch.zeros(d_out, k, device=device, dtype=U_eff.dtype)
            V_k = torch.zeros(k, d_in, device=device, dtype=V_eff.dtype)
            U_k[:, :k_eff] = U_eff
            V_k[:k_eff, :] = V_eff
        else:
            U_k, V_k = U_eff, V_eff
    except Exception as err:                         # noqa: BLE001
        log.warning("AA-SVD (precomputed) fallback to plain SVD (%s)", err)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = Vh[:k, :]
        with torch.no_grad():
            R = W - U_k @ V_k
            w_norm = W.norm()
            rel_err = float((R.norm() / w_norm).item()) if w_norm > 0 else 0.0
        k_eff = k
    return U_k, V_k, rel_err, k_eff


def _aa_svd(
    W: torch.Tensor,
    A: torch.Tensor | None,
    B: torch.Tensor | None,
    k: int,
    *,
    C: torch.Tensor | None = None,
    device,
    storage_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Activation-aware rank-k factorization of W.

    Two paths from paper 2604.02119, in priority order:

    1. **Paper-exact (Theorem 3.2)**: when cross-covariance C = X_pre^T X_post
       and B = X_post^T X_post are both available:
         M = W · C · B^{-1} · L_B
       where L_B satisfies B = L_B · L_B^T. This is the exact AA-SVD solution
       that anchors to original outputs while adapting to shifted inputs.

    2. **Corollary 3.3 fallback**: when C is unavailable:
         M = W · L_B
       Shift-aware variant that adapts to post-prune distribution only.

    The ``A`` argument (pre-prune auto-covariance) is reserved for L-BFGS
    refinement and is NOT used in the rank-k factorization. An earlier
    "Path 2" using A as a proxy for C produced U·V ≈ W·A·B^{-1}·L_B rather
    than approximating W, breaking downstream consumers (FactoredExperts
    forward, Stage 4 EoRA residual). The contract is pinned by tests in
    ``tests/test_aa_svd_correctness.py``.

    Returns (U_k, V_k, rel_err, k_eff).

    .. note::

       When factoring both gate_proj and up_proj for the same expert, prefer
       ``_precompute_eigh`` + ``_aa_svd_precomputed`` to avoid the redundant
       ``eigh(B)`` call — gate_proj and up_proj share the same B and C
       covariance via ``_cov_lookup`` fallback.
    """
    d_out, d_in = W.shape
    k = max(1, min(k, min(d_out, d_in) - 1))
    try:
        decomp = _precompute_eigh(B, A, C, device=device, storage_dtype=storage_dtype)
        return _aa_svd_precomputed(W, decomp, k, device=device)
    except Exception as err:                         # noqa: BLE001
        log.warning("AA-SVD fallback to plain SVD (%s)", err)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k] * S[:k]
        V_k = Vh[:k, :]
        with torch.no_grad():
            R = W - U_k @ V_k
            w_norm = W.norm()
            rel_err = float((R.norm() / w_norm).item()) if w_norm > 0 else 0.0
        k_eff = k
    return U_k, V_k, rel_err, k_eff


def _cov_lookup(cov: dict, layer_idx: int, expert_idx: int, matrix_name: str):
    """Bank-aware lookup: falls back to gate_proj when asked for up_proj."""
    key = (layer_idx, expert_idx, matrix_name)
    if key in cov:
        return cov[key]
    if matrix_name == "up_proj":
        return cov.get((layer_idx, expert_idx, "gate_proj"))
    return None


class AaSvdFactorPlugin:
    """Stage 3 AA-SVD rank-k factorization plugin (S3-5 — registered-but-INERT).

    Owns the activation-aware SVD core: the storage-dtype noise-floor table
    (``_NOISE_FLOOR_BY_DTYPE``), the cached eigendecomposition (``_EighDecomp``
    / ``_precompute_eigh``), the rank-k factorization (``_aa_svd`` /
    ``_aa_svd_precomputed``) and the per-bank covariance lookup
    (``_cov_lookup``). The core lives in the module-level symbols relocated
    verbatim from the monolith (AA-SVD paper 2604.02119, Theorem 3.2 /
    Corollary 3.3).

    S3-5 wires this class into the plugin registry as metadata only — no walk
    or test invokes ``factor_layer``. S3-7 plugs the hook into the live Stage 3
    plugin sequencer.
    """

    name = "aa_svd_factor"
    paper = (
        "AA-SVD activation-aware rank-k factorization — M = W·C·B⁻¹·L_B "
        "(Theorem 3.2) / M = W·L_B (Corollary 3.3) (paper 2604.02119)."
    )
    config_key = "stage3_svd.aa_svd.cross_covariance"
    reads: tuple[str, ...] = (
        "model", "moe_layers", "ranks", "per_expert_ranks", "A_cov",
        "B_acc", "C_acc", "config", "device",
    )
    writes: tuple[str, ...] = ("rank_map",)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — AA-SVD rank-k factorization is UNCONDITIONAL.

        AA-SVD is the core of Stage 3: every Stage 3 run factors the MoE
        experts via this code path. ``config_key`` only selects the Path 1
        (cross-covariance) vs. Path 3 (B-only) variant of the factorization;
        it never gates the plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def factor_layer(self, ctx: PipelineContext) -> None:
        """Phase hook — AA-SVD per-layer factoring (S3-7 wiring surface).

        INERT at S3-5: no orchestrator walk or test invokes this hook. The
        per-layer factoring LOOP BODY in the monolith ``run()`` — the loop that
        walks the MoE layers, calls ``_precompute_eigh`` / ``_aa_svd`` /
        ``_aa_svd_precomputed`` per expert and installs ``FactoredExperts`` while
        populating ``rank_map`` — is NOT relocated by S3-5. S3-7 replaces the
        Stage 3 orchestrator body with the plugin sequencer and fills this hook
        with that loop.

        Dead code at S3-5; kept minimal — S3-7 builds it out.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. Optional slots are has()-guarded.
        if not ctx.has("moe_layers"):
            return
