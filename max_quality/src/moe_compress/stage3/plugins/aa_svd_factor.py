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
from ...tools.dtype_noise_floor import _NOISE_FLOOR_BY_DTYPE  # noqa: F401
from ...utils.model_io import MATRIX_NAMES, FactoredExperts
from ...utils.trackio_log import trackio_log as _trackio_log

log = logging.getLogger(__name__)

# S4-3: ``_NOISE_FLOOR_BY_DTYPE`` relocated to tools/dtype_noise_floor (a pure
# literal shared by stage 3 AA-SVD and stage 4 EoRA). Re-imported above so
# ``_precompute_eigh`` below + ``stage3_svd``'s S3-5 re-export block + the
# stage-3 plugin tests keep their existing ``aa_svd_factor`` import paths.


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
    # ``factor_layer`` runs inside a per-layer ``loop_over`` child scope: it
    # reads the layer ref under ``layer_ref`` (the loop item key) and the
    # remaining slots through the parent ctx chain.
    reads: tuple[str, ...] = (
        "layer_ref", "ranks", "per_expert_ranks", "A_cov", "B_acc", "C_acc",
        "B_cov_dtype", "rank_map", "device", "originals",
        "bcov_spill_dir", "ccov_spill_dir",
    )
    # ``rank_map`` is the slot this plugin produces — it is a shared mutable
    # dict the hook MUTATES in place across loop iterations (HAZARD H1)
    # rather than rebinding via ``ctx.set``, but it remains this plugin's
    # declared write surface.
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
        """Phase hook — AA-SVD per-layer factoring (S3-7a live).

        Filled at S3-7a with the VERBATIM per-layer factoring loop body from
        the monolith ``run()`` (the ``for ref in moe_layers:`` body). The
        layer ref arrives under ``ctx["layer_ref"]`` (the ``loop_over``
        item key); ``ranks`` / ``per_expert_ranks`` / ``A_cov`` / ``B_acc`` /
        ``C_acc`` / ``B_cov_dtype`` / ``rank_map`` / ``device`` / ``originals``
        resolve through the parent ctx chain. ``rank_map`` is the ONE shared
        mutable dict set on the root ctx — this hook mutates it in place
        across loop iterations (HAZARD H1).
        """
        ref = ctx.get("layer_ref")
        ranks = ctx.get("ranks")
        per_expert_ranks = ctx.get("per_expert_ranks")
        A_cov = ctx.get("A_cov")
        B_acc = ctx.get("B_acc")
        C_acc = ctx.get("C_acc")
        B_cov_dtype = ctx.get("B_cov_dtype")
        rank_map = ctx.get("rank_map")
        device = ctx.get("device")
        originals = ctx.get("originals")
        bcov_spill_dir = ctx.get("bcov_spill_dir")
        ccov_spill_dir = ctx.get("ccov_spill_dir")

        # ---- VERBATIM per-layer factoring loop body from the monolith run() --
        # When Swift-SVD+ gives per-expert ranks, allocate at the max rank
        # across experts for each matrix type (the slot width). Experts with
        # lower rank will be zero-padded; effective_ranks tracks the true rank.
        if per_expert_ranks is not None:
            ranks_layer = {
                name: max(
                    per_expert_ranks.get((ref.layer_idx, name, e), ranks[(ref.layer_idx, name)])
                    for e in range(ref.num_routed_experts)
                )
                for name in MATRIX_NAMES
            }
        else:
            ranks_layer = {
                name: ranks[(ref.layer_idx, name)] for name in MATRIX_NAMES
            }
        # Lazy-load this layer's B-cov from the per-layer spill files.
        # Keeps in-memory cov bounded to ~one layer (~3-5 GB at bf16).
        # Assert (not silent fall-through) — a missing spill at this
        # point would mean _aa_svd silently falls back to plain SVD for
        # this whole layer's experts, ignoring the activation-aware
        # weighting; we'd ship a degraded model. Crash loud instead.
        loaded = B_acc.load_layer_from_disk(ref.layer_idx, bcov_spill_dir)
        if not loaded:
            raise RuntimeError(
                f"Stage 3 factor: B-cov spill missing for layer {ref.layer_idx} "
                f"at {bcov_spill_dir}/layer_{ref.layer_idx}.pt. The B-cov phase "
                "should have produced this file. Investigate before proceeding."
            )
        # Also load cross-covariance C for this layer (if dual-forward was run).
        if C_acc is not None and ccov_spill_dir is not None:
            c_loaded = C_acc.load_layer_from_disk(ref.layer_idx, ccov_spill_dir)
            if not c_loaded:
                log.warning(
                    "Stage 3 factor: cross-cov spill missing for layer %d — "
                    "falling back to auto-covariance for this layer.",
                    ref.layer_idx,
                )
        # Build FactoredExperts on the same device / dtype.
        # Originals are already snapshotted to CPU (before α search);
        # offload the dense expert module before allocating FactoredExperts
        # to avoid brief double-occupancy OOM on 80 GB A100s.
        ex = ref.experts_module
        dtype = ex.gate_up_proj.dtype
        dev = ex.gate_up_proj.device
        ex.to("cpu")
        torch.cuda.empty_cache()
        new_factored = FactoredExperts(
            num_experts=ref.num_routed_experts,
            hidden_dim=ex.gate_up_proj.shape[-1],
            intermediate_dim=ex.gate_up_proj.shape[1] // 2,
            ranks=ranks_layer, dtype=dtype, device=dev,
        )
        # Fill factors by per-expert AA-SVD. Track relative reconstruction
        # error per (layer, matrix) so the dashboard shows whether the chosen
        # rank is enough — a "convergence in spirit" signal for the SVD.
        # Per-expert weighted relative error: mean of ||(W-UV)L_B||/||WL_B|| across experts.
        #
        # Optimization: gate_proj and up_proj share the same B and C covariance
        # (``_cov_lookup`` falls back from up_proj to gate_proj).  We precompute
        # the eigh(B) decomposition + rhs product once per expert and reuse it
        # for both projections, eliminating one eigh(2048×2048) call per expert
        # (~7,200 redundant calls across 40 layers).
        err_sum: dict[str, float] = {n: 0.0 for n in MATRIX_NAMES}
        n_per_matrix: dict[str, int] = {n: 0 for n in MATRIX_NAMES}
        k_eff_clip_count: dict[str, int] = {n: 0 for n in MATRIX_NAMES}
        for e in range(ref.num_routed_experts):
            # --- Precompute shared eigh for gate_proj / up_proj ---
            B_shared = _cov_lookup(B_acc.covariance, ref.layer_idx, e, "gate_proj")
            A_shared = _cov_lookup(A_cov, ref.layer_idx, e, "gate_proj")
            C_shared = None
            if C_acc is not None:
                C_shared = _cov_lookup(C_acc.covariance, ref.layer_idx, e, "gate_proj")
            gate_up_decomp: _EighDecomp | None = None
            if B_shared is not None:
                try:
                    gate_up_decomp = _precompute_eigh(
                        B_shared, A_shared, C_shared,
                        device=dev, storage_dtype=B_cov_dtype,
                    )
                except ValueError:
                    pass  # falls through to plain SVD per matrix below

            for name in MATRIX_NAMES:
                W = originals[(ref.layer_idx, e, name)].to(device=dev, dtype=torch.float32)
                # Per-expert rank from Swift-SVD+ if available, else group-uniform.
                if per_expert_ranks is not None:
                    k = per_expert_ranks.get((ref.layer_idx, name, e), ranks_layer[name])
                else:
                    k = ranks_layer[name]
                if name in ("gate_proj", "up_proj") and gate_up_decomp is not None:
                    # Reuse the precomputed eigh for gate_proj and up_proj.
                    U_k, V_k, rel_err, k_eff = _aa_svd_precomputed(
                        W, gate_up_decomp, k, device=dev,
                    )
                else:
                    # down_proj has its own B (intermediate-dim covariance),
                    # or gate_up_decomp failed — fall back to full _aa_svd.
                    A = _cov_lookup(A_cov, ref.layer_idx, e, name)
                    B = _cov_lookup(B_acc.covariance, ref.layer_idx, e, name)
                    C = None
                    if C_acc is not None:
                        C = _cov_lookup(C_acc.covariance, ref.layer_idx, e, name)
                    U_k, V_k, rel_err, k_eff = _aa_svd(
                        W, A, B, k, C=C, device=dev, storage_dtype=B_cov_dtype,
                    )
                if k_eff < k:
                    k_eff_clip_count[name] += 1
                new_factored.set_factors(e, name, U_k, V_k, effective_rank=k_eff)
                rank_map[f"L{ref.layer_idx}_E{e}_{name}"] = k
                err_sum[name] += rel_err
                n_per_matrix[name] += 1
        # Swap in.
        setattr(ref.mlp, "experts", new_factored)
        ref.experts_module = new_factored
        recon_metrics: dict[str, float] = {"stage3/recon_layer_idx": float(ref.layer_idx)}
        for name in MATRIX_NAMES:
            if n_per_matrix[name] > 0:
                rel = err_sum[name] / n_per_matrix[name]
                # Renamed from `recon_rel_err` post-bf16 fix: this is the
                # B-weighted singular-value-tail ratio of M = W·L_B. The old
                # key is dual-emitted as an alias so existing trackio
                # dashboards keep working — TODO(post-launch): drop the alias
                # once dashboards are migrated to `b_weighted_tail_ratio`.
                recon_metrics[f"stage3/b_weighted_tail_ratio/{name}"] = rel
                recon_metrics[f"stage3/recon_rel_err/{name}"] = rel
                recon_metrics[f"stage3/k_eff_clip_count/{name}"] = float(k_eff_clip_count[name])
                recon_metrics[f"stage3/k_eff_clip_ratio/{name}"] = (
                    k_eff_clip_count[name] / max(n_per_matrix[name], 1)
                )
                # `b_weighted_tail_ratio` = ‖tail_S(M)‖/‖S(M)‖, the singular-
                # value-tail proxy for ‖(W−UV)L_B‖/‖WL_B‖. Pre-fix code logged
                # this same key as `rel_recon_err`; numbers from before commit
                # e7e0fbf are not directly comparable.
                log.info("  L%d %s rank=%d b_weighted_tail_ratio=%.4f k_eff_clipped=%d/%d",
                         ref.layer_idx, name, ranks_layer[name], rel,
                         k_eff_clip_count[name], n_per_matrix[name])
        _trackio_log(recon_metrics)
        log.info("  layer %d factored at ranks=%s", ref.layer_idx, ranks_layer)
        # Drop this layer's B-cov and C-cov from memory now that we're done factoring
        # it. The next iteration will lazy-load the next layer's spill.
        B_acc.unload_layer(ref.layer_idx)
        if C_acc is not None:
            C_acc.unload_layer(ref.layer_idx)
